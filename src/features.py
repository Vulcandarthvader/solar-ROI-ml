"""Feature engineering.

Three models need three different feature views, and they are kept apart on
purpose:

tariff features       billed units + calendar time.  Learns "what does the
                      utility charge for N units in month M of year Y".
consumption features  calendar only.  Deliberately excludes anything measured
                      after the array was installed, because this model has to
                      answer a counterfactual question.
generation features   calendar + panel age.  Age carries the degradation trend.

Leakage note: no lagged bill or lagged consumption feature is used anywhere.
A lag would make the in-sample fit look excellent and make the 25-year
projection meaningless, because at forecast time there is no lag to feed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Coastal Goa has a hot-humid summer (Mar-May), a monsoon (Jun-Sep) that guts
# solar output, and a mild dry winter.  A one-hot season lets tree models split
# on it directly instead of discovering it from sin/cos.
SEASONS = {
    1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon", 12: "winter",
}


def add_calendar_features(df: pd.DataFrame, origin: pd.Period | None = None) -> pd.DataFrame:
    """Add cyclical month encodings, a linear time index and season dummies."""
    out = df.copy()
    period = out["period"] if "period" in out else pd.PeriodIndex(out.index, freq="M")
    origin = origin if origin is not None else period.min()

    out["t_months"] = [(p - origin).n for p in period]
    out["t_years"] = out["t_months"] / 12.0
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    # A second harmonic captures the double-peak of a monsoon climate.
    out["month_sin2"] = np.sin(4 * np.pi * out["month"] / 12)
    out["month_cos2"] = np.cos(4 * np.pi * out["month"] / 12)
    out["season"] = out["month"].map(SEASONS)
    for s in ("summer", "monsoon", "post_monsoon"):  # winter is the reference level
        out[f"season_{s}"] = (out["season"] == s).astype(int)
    return out


TARIFF_FEATURES = [
    "grid_kwh",
    "kwh_over_100",
    "kwh_over_200",
    "kwh_over_300",
    "kwh_over_400",
    "t_years",
    "month_sin",
    "month_cos",
    "season_summer",
    "season_monsoon",
    "season_post_monsoon",
]

CONSUMPTION_FEATURES = [
    "t_years",
    "month_sin",
    "month_cos",
    "month_sin2",
    "month_cos2",
    "season_summer",
    "season_monsoon",
    "season_post_monsoon",
]

GENERATION_FEATURES = [
    "panel_age_years",
    "month_sin",
    "month_cos",
    "month_sin2",
    "month_cos2",
    "season_summer",
    "season_monsoon",
    "season_post_monsoon",
]


def add_slab_features(df: pd.DataFrame, column: str = "grid_kwh") -> pd.DataFrame:
    """Hinge features at the domestic tariff slab boundaries.

    Indian domestic tariffs are block-rate: the per-unit price steps up past
    100, 200, 300 and 400 units.  Encoding units-above-each-threshold turns the
    piecewise-linear tariff into something a *linear* model can represent
    exactly, so the linear baseline competes honestly with the tree models
    instead of losing to an artefact of feature design.
    """
    out = df.copy()
    units = out[column].astype(float)
    for thr in (100, 200, 300, 400):
        out[f"kwh_over_{thr}"] = np.maximum(units - thr, 0.0)
    return out


def build_tariff_frame(df: pd.DataFrame, origin: pd.Period | None = None) -> pd.DataFrame:
    return add_slab_features(add_calendar_features(df, origin), "grid_kwh")


def build_counterfactual_input(df: pd.DataFrame, origin: pd.Period | None = None) -> pd.DataFrame:
    """Same rows, but with billed units replaced by *gross* demand.

    This is the counterfactual query put to the tariff model: charge me for
    every unit the house actually used, as if no array existed.
    """
    out = add_calendar_features(df, origin)
    out["grid_kwh"] = out["total_consumption_kwh"]
    return add_slab_features(out, "grid_kwh")


def build_generation_frame(
    df: pd.DataFrame, commissioning: pd.Period, origin: pd.Period | None = None
) -> pd.DataFrame:
    out = add_calendar_features(df, origin)
    period = out["period"] if "period" in out else pd.PeriodIndex(out.index, freq="M")
    out["panel_age_years"] = [(p - commissioning).n / 12.0 for p in period]
    return out


def future_frame(
    start: pd.Period, n_months: int, commissioning: pd.Period, origin: pd.Period
) -> pd.DataFrame:
    """Calendar scaffold for projecting forward, with no measured data in it."""
    idx = pd.period_range(start, periods=n_months, freq="M")
    df = pd.DataFrame({"period": idx, "year": idx.year, "month": idx.month})
    df = add_calendar_features(df, origin)
    df["panel_age_years"] = [(p - commissioning).n / 12.0 for p in idx]
    return df
