"""Load and clean the electricity-bill CSV.

The CSV has one row per bill: year, month, grid_kwh (units bought from the
grid), bill_amount (Rs paid) and solar_kwh (units the panels made, if known).

Real utility exports are messy. Every fix below is recorded as a True/False
"flag" column, so nothing is changed silently:

    duplicate month   two bills in one month          -> add them together
    missing month     a gap in the history            -> fill with that month's usual value
    repeat reading    same reading copied to 2 months -> flag, don't train on it
    credit bill       negative bill (a refund)        -> flag, don't train on it
    arrears bill      a bill far too big for its units-> flag, don't train on it
"""

import numpy as np
import pandas as pd

REQUIRED = {"year", "month", "grid_kwh", "bill_amount"}


def load_bills(source):
    """Read the CSV (a path or an uploaded file) and check the columns."""
    df = pd.read_csv(source)
    df.columns = [c.strip().lower().lstrip("﻿") for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing column(s): {sorted(missing)}")
    if "solar_kwh" not in df.columns:
        df["solar_kwh"] = 0.0
    return df


def clean_bills(df):
    """Return (clean monthly table, report dict)."""
    report = {"rows_read": len(df)}
    df = df.copy()
    df["period"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=df.year, month=df.month, day=1)), freq="M"
    )

    # 1. Duplicate months: add units and rupees. Solar is a monthly total
    #    written on both bills, so take it once (max), never add it twice.
    report["duplicate_months"] = sorted(str(p) for p in df.loc[df.duplicated("period", keep=False), "period"].unique())
    monthly = df.groupby("period").agg(
        grid_kwh=("grid_kwh", "sum"),
        bill_amount=("bill_amount", "sum"),
        solar_kwh=("solar_kwh", "max"),
        n_bills=("grid_kwh", "size"),
    )

    # 2. Missing months: put every month on the calendar, then fill gaps with
    #    the average of the same calendar month in other years.
    #    Solar is NEVER filled in - guessing it could invent solar output
    #    before the panels even existed.
    all_months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(all_months)
    monthly["is_imputed"] = monthly["grid_kwh"].isna()
    report["imputed_months"] = [str(p) for p in all_months[monthly["is_imputed"]]]
    for col in ("grid_kwh", "bill_amount"):
        same_month_avg = monthly.groupby(monthly.index.month)[col].transform("mean")
        monthly[col] = monthly[col].fillna(same_month_avg)
    monthly["solar_kwh"] = monthly["solar_kwh"].fillna(0.0)
    monthly["n_bills"] = monthly["n_bills"].fillna(0)

    # 3. Repeat readings: the utility bills every 2 months and copies the
    #    same numbers into both months.
    monthly["is_repeat"] = (
        monthly["grid_kwh"].diff().eq(0) & monthly["bill_amount"].diff().eq(0) & ~monthly["is_imputed"]
    )
    report["repeat_readings"] = int(monthly["is_repeat"].sum())

    # 4. Credit bills: negative amount = refund, not a real price.
    monthly["is_credit"] = monthly["bill_amount"] < 0
    report["credit_bills"] = [str(p) for p in monthly.index[monthly["is_credit"]]]

    # 5. Arrears bills: two bills landed in one month, or Rs-per-unit is more
    #    than 3x normal. This money belongs to other months.
    rate = monthly["bill_amount"] / monthly["grid_kwh"].where(monthly["grid_kwh"] > 0)
    normal_rate = rate[rate > 0].median()
    monthly["is_arrears"] = (monthly["n_bills"] > 1) | ((monthly["grid_kwh"] >= 30) & (rate > 3 * normal_rate))
    report["arrears_bills"] = [str(p) for p in monthly.index[monthly["is_arrears"]]]

    monthly = monthly.rename_axis("period").reset_index()
    monthly["year"] = monthly["period"].dt.year
    monthly["month"] = monthly["period"].dt.month
    monthly = add_derived_columns(monthly)

    report["months_after_cleaning"] = len(monthly)
    report["date_range"] = f"{monthly.period.iloc[0]} to {monthly.period.iloc[-1]}"
    return monthly, report


def add_derived_columns(df):
    """Columns calculated from the others."""
    df["has_solar"] = df["solar_kwh"] > 0
    # With net metering the meter only counts what you BOUGHT. Adding back
    # what the panels made gives what the house really USED.
    df["total_use_kwh"] = df["grid_kwh"] + df["solar_kwh"]
    return df


def merge_screenshots(bills, screenshot_rows):
    """Add generation read from screenshots into the bill table.

    - Month in both: keep the CSV (meter) value, and measure how far the
      screenshot was from it -> this is how we check screenshot accuracy.
    - Month only in screenshots (after the last bill): add it.
    Returns (table, stats).
    """
    stats = {"overlap_months": 0, "screenshot_error_pct": float("nan"), "months_added": 0}
    if not screenshot_rows:
        return bills, stats

    shots = pd.DataFrame(screenshot_rows)
    shots["period"] = pd.PeriodIndex(pd.to_datetime(dict(year=shots.year, month=shots.month, day=1)), freq="M")
    shots = shots.groupby("period")["solar_kwh"].mean()

    table = bills.set_index("period")
    both = shots[shots.index.isin(table.index)]
    metered = table.loc[both.index, "solar_kwh"]
    check = metered > 0
    if check.any():
        stats["overlap_months"] = int(check.sum())
        stats["screenshot_error_pct"] = float(((both[check] - metered[check]).abs() / metered[check]).mean() * 100)

    # Extend the calendar to the last screenshot month.
    end = max(table.index.max(), shots.index.max())
    full = pd.period_range(table.index.min(), end, freq="M")
    stats["months_added"] = len(full) - len(table)
    table = table.reindex(full)
    for flag in ("is_imputed", "is_repeat", "is_credit", "is_arrears"):
        table[flag] = table[flag].astype("boolean").fillna(False).astype(bool)

    # Fill solar where the CSV had none.
    no_meter = table["solar_kwh"].fillna(0).le(0) & table.index.isin(shots.index)
    table.loc[no_meter, "solar_kwh"] = shots.reindex(table.index[no_meter]).values
    table["solar_kwh"] = table["solar_kwh"].fillna(0.0)

    table = table.rename_axis("period").reset_index()
    table["year"] = table["period"].dt.year
    table["month"] = table["period"].dt.month
    return add_derived_columns(table), stats


def training_rows(df):
    """Only the rows we trust enough to train a model on."""
    bad = df["is_imputed"] | df["is_repeat"] | df["is_credit"] | df["is_arrears"]
    return df[~bad & df["bill_amount"].notna() & (df["bill_amount"] > 0)].copy()
