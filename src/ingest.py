"""Loading, cleaning and auditing the electricity-bill history.

Real utility exports are messy, and this one is no exception.  The cleaning
here is deliberately *explicit*: every repair is recorded as a boolean column
and summarised in a data-quality report, so nothing silently changes the ROI
number downstream.

Issues handled
--------------
duplicate month     Two bills issued inside one calendar month (a catch-up
                    bill).  Consumption and amount are summed; generation,
                    which is a monthly total, is not.
bi-monthly repeats  The utility bills every two months and the export writes
                    the same reading to both months.  Flagged, then
                    down-weighted so one meter reading is not counted twice
                    when fitting the tariff.
credit adjustments  Negative bill amounts (refunds/arrears reversals).  Kept
                    for cash-flow accounting, excluded from tariff fitting.
missing months      Gaps in the export.  Reindexed onto a continuous monthly
                    axis, seasonally interpolated, and flagged as imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"year", "month", "grid_kwh", "bill_amount"}


@dataclass
class DataQualityReport:
    n_raw_rows: int = 0
    n_clean_rows: int = 0
    duplicate_months: list[str] = field(default_factory=list)
    repeated_readings: list[str] = field(default_factory=list)
    credit_adjustments: list[str] = field(default_factory=list)
    imputed_months: list[str] = field(default_factory=list)
    billing_anomalies: list[str] = field(default_factory=list)
    date_range: tuple[str, str] = ("", "")
    solar_start: str | None = None

    def to_lines(self) -> list[str]:
        return [
            f"Rows read: {self.n_raw_rows} -> {self.n_clean_rows} after cleaning",
            f"Coverage: {self.date_range[0]} to {self.date_range[1]}",
            f"Solar commissioned: {self.solar_start or 'not detected'}",
            f"Duplicate months merged: {len(self.duplicate_months)} {self.duplicate_months or ''}",
            f"Repeated (bi-monthly) readings flagged: {len(self.repeated_readings)}",
            f"Credit-adjustment bills flagged: {len(self.credit_adjustments)} {self.credit_adjustments or ''}",
            f"Missing months imputed: {len(self.imputed_months)} {self.imputed_months or ''}",
            f"Billing anomalies (arrears / true-up) flagged: "
            f"{len(self.billing_anomalies)} {self.billing_anomalies or ''}",
        ]


def _stamp(year, month) -> str:
    return f"{int(year)}-{int(month):02d}"


def load_bills(source) -> pd.DataFrame:
    """Read the bill CSV from a path or file-like object and validate columns."""
    df = pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"billdata.csv is missing required column(s): {sorted(missing)}. "
            f"Expected at least {sorted(REQUIRED_COLUMNS)}."
        )
    if "solar_kwh" not in df.columns:
        df["solar_kwh"] = 0.0
    for c in ("year", "month", "grid_kwh", "bill_amount", "solar_kwh"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["year", "month"])
    df = df[(df["month"] >= 1) & (df["month"] <= 12)]
    return df.reset_index(drop=True)


def clean_bills(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Clean the raw bill frame onto a continuous monthly axis."""
    report = DataQualityReport(n_raw_rows=len(df))

    df = df.copy()
    df["period"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=df.year.astype(int), month=df.month.astype(int), day=1)), freq="M"
    )

    # --- duplicate calendar months: sum the money and the metered units ------
    dupes = df[df.duplicated("period", keep=False)]["period"].unique()
    report.duplicate_months = [str(p) for p in sorted(dupes)]
    agg = (
        df.groupby("period")
        .agg(
            grid_kwh=("grid_kwh", "sum"),
            bill_amount=("bill_amount", "sum"),
            solar_kwh=("solar_kwh", "max"),  # a monthly total, never additive
            n_bills=("grid_kwh", "size"),
        )
        .sort_index()
    )

    # --- reindex onto a continuous monthly axis -----------------------------
    full_index = pd.period_range(agg.index.min(), agg.index.max(), freq="M")
    missing = full_index.difference(agg.index)
    report.imputed_months = [str(p) for p in missing]
    agg = agg.reindex(full_index)
    agg["is_imputed"] = agg["grid_kwh"].isna()

    # Seasonal-aware fill: use the same calendar month either side where we can,
    # otherwise linear interpolation.  Only ever fills a handful of months.
    # Generation is never interpolated: a month with no reading generated
    # nothing we can bill against, and seasonally averaging it would invent
    # solar output years before the array existed.
    for col in ("grid_kwh", "bill_amount"):
        month_of = agg.index.month
        filled = agg[col].copy()
        for pos in np.where(agg[col].isna())[0]:
            same_month = agg[col][(month_of == month_of[pos])].dropna()
            if len(same_month) >= 2:
                filled.iloc[pos] = float(same_month.mean())
        agg[col] = filled.interpolate(limit_direction="both")
    agg["n_bills"] = agg["n_bills"].fillna(0)

    # --- bi-monthly repeats --------------------------------------------------
    same_as_prev = (
        agg["grid_kwh"].diff().eq(0)
        & agg["bill_amount"].diff().eq(0)
        & ~agg["is_imputed"]
        & ~agg["is_imputed"].shift(fill_value=False)
    )
    agg["is_repeat_reading"] = same_as_prev.fillna(False)
    report.repeated_readings = [str(p) for p in agg.index[agg["is_repeat_reading"]]]

    # --- credit adjustments --------------------------------------------------
    agg["is_credit_adjustment"] = agg["bill_amount"] < 0
    report.credit_adjustments = [str(p) for p in agg.index[agg["is_credit_adjustment"]]]

    agg = agg.reset_index().rename(columns={"index": "period"})
    agg["year"] = agg["period"].dt.year
    agg["month"] = agg["period"].dt.month
    agg["date"] = agg["period"].dt.to_timestamp()

    # --- derived quantities --------------------------------------------------
    agg["solar_kwh"] = agg["solar_kwh"].fillna(0.0).clip(lower=0)
    agg["has_solar"] = agg["solar_kwh"] > 0
    # What the meter would have registered with no array on the roof.  With net
    # metering the billed units are already net of export, so adding back
    # generation reconstructs gross household demand.
    agg["total_consumption_kwh"] = agg["grid_kwh"] + agg["solar_kwh"]
    agg["self_sufficiency"] = np.where(
        agg["total_consumption_kwh"] > 0, agg["solar_kwh"] / agg["total_consumption_kwh"], 0.0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["effective_rate"] = np.where(
            (agg["grid_kwh"] > 0) & (agg["bill_amount"] > 0),
            agg["bill_amount"] / agg["grid_kwh"],
            np.nan,
        )

    # --- billing anomalies ---------------------------------------------------
    # A month holding two bills, or one whose rupees-per-unit is wildly out of
    # line, is an arrears or true-up adjustment: money that belongs to other
    # months.  It is real cash, so it stays in the record, but it must not be
    # read as that month's consumption, and it would poison a tariff fit.
    rate = agg["effective_rate"]
    median_rate = float(rate[rate.notna()].median()) if rate.notna().any() else np.nan
    # The units floor matters: on a near-zero-import month the fixed charge is
    # spread over a handful of units, so the rupees-per-unit ratio explodes for
    # a perfectly ordinary bill.  That is a feature of block tariffs, not an
    # anomaly, and flagging it would throw away the best solar months.
    agg["is_billing_anomaly"] = (agg["n_bills"] > 1) | (
        rate.notna() & np.isfinite(median_rate) & (agg["grid_kwh"] >= 30) & (rate > 3 * median_rate)
    )
    report.billing_anomalies = [str(p) for p in agg.loc[agg["is_billing_anomaly"], "period"]]

    solar_rows = agg[agg["has_solar"]]
    report.solar_start = str(solar_rows["period"].iloc[0]) if len(solar_rows) else None
    report.n_clean_rows = len(agg)
    report.date_range = (str(agg["period"].iloc[0]), str(agg["period"].iloc[-1]))
    return agg, report


def merge_generation(bills: pd.DataFrame, extracted: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Fold screenshot-extracted generation into the bill history.

    The bill CSV stops when the last bill was issued; the app's charts run to
    the current month.  Where both sources cover a month we keep the CSV value
    (a metered figure beats a pixel measurement) and record the disagreement as
    a cross-validation statistic.  Where only the screenshots reach, they
    extend the history.
    """
    stats = {"overlap_n": 0, "overlap_mape": float("nan"), "months_added": 0, "months_updated": 0}
    if extracted is None or len(extracted) == 0:
        return bills, stats

    ext = extracted.dropna(subset=["month"]).copy()
    if ext.empty:
        return bills, stats
    ext["period"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=ext.year.astype(int), month=ext.month.astype(int), day=1)), freq="M"
    )
    ext = ext.groupby("period", as_index=False)["solar_kwh"].mean()

    out = bills.set_index("period")
    overlap = ext[ext["period"].isin(out.index)]
    both = overlap.set_index("period").join(out[["solar_kwh"]], rsuffix="_csv")
    both = both[both["solar_kwh_csv"] > 0]
    if len(both):
        stats["overlap_n"] = int(len(both))
        stats["overlap_mape"] = float(
            (both["solar_kwh"] - both["solar_kwh_csv"]).abs().div(both["solar_kwh_csv"]).mean() * 100
        )

    # Months the CSV never saw, or saw with zero generation.
    new_periods = [p for p in ext["period"] if p not in out.index]
    ext_indexed = ext.set_index("period")["solar_kwh"]

    if new_periods:
        last = out.index.max()
        extend_to = max(new_periods)
        if extend_to > last:
            new_index = pd.period_range(out.index.min(), extend_to, freq="M")
            out = out.reindex(new_index)
            out.index.name = "period"
            stats["months_added"] = int(len(new_index) - len(bills))
            out["year"] = out.index.year
            out["month"] = out.index.month
            out["date"] = out.index.to_timestamp()
            for flag, default in (
                ("is_imputed", True),
                ("is_repeat_reading", False),
                ("is_credit_adjustment", False),
                ("is_billing_anomaly", False),
            ):
                out[flag] = out[flag].fillna(default).astype(bool)

    fill_mask = out["solar_kwh"].fillna(0).le(0) & out.index.isin(ext_indexed.index)
    if fill_mask.any():
        out.loc[fill_mask, "solar_kwh"] = ext_indexed.reindex(out.index[fill_mask]).values
        stats["months_updated"] = int(fill_mask.sum())

    out["solar_kwh"] = out["solar_kwh"].fillna(0.0)
    out["has_solar"] = out["solar_kwh"] > 0
    out["grid_kwh"] = out["grid_kwh"].fillna(np.nan)
    out["total_consumption_kwh"] = out["grid_kwh"] + out["solar_kwh"]
    out["self_sufficiency"] = np.where(
        out["total_consumption_kwh"] > 0, out["solar_kwh"] / out["total_consumption_kwh"], 0.0
    )
    return out.reset_index(), stats


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows trustworthy enough to fit a tariff or consumption model on.

    Drops imputed months, credit-adjustment bills and repeated meter readings,
    each of which would otherwise distort the money-per-unit relationship.
    """
    keep = (
        ~df["is_imputed"].fillna(True)
        & ~df["is_credit_adjustment"].fillna(False)
        & ~df["is_repeat_reading"].fillna(False)
        & ~df.get("is_billing_anomaly", pd.Series(False, index=df.index)).fillna(False)
        & df["bill_amount"].notna()
        & df["grid_kwh"].notna()
        & (df["bill_amount"] > 0)
    )
    return df[keep].copy()
