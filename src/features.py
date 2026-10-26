"""Turn dates and units into number columns a model can learn from.

    t_years             years since the first bill (tariffs rise over time)
    month_sin/month_cos the month placed on a circle, so Dec and Jan are neighbours
    kwh_over_100 ...    units above each tariff slab (100, 200, 300, 400)

Why slab features? Indian tariffs charge more per unit after 100, 200, 300
and 400 units. With "units above each slab" as columns, a straight-line
model can learn a different price for each slab.
    Example: 250 units -> kwh_over_100 = 150, kwh_over_200 = 50, others 0.
"""

import numpy as np

SLABS = (100, 200, 300, 400)

# Model A (tariff model): what does N units cost in a given month?
TARIFF_FEATURES = ["grid_kwh", "kwh_over_100", "kwh_over_200", "kwh_over_300",
                   "kwh_over_400", "t_years", "month_sin", "month_cos"]

# Model B (pre-solar model): only the calendar. It must never see solar data.
CALENDAR_FEATURES = ["t_years", "month_sin", "month_cos"]


def add_features(df, start_period, units_column="grid_kwh"):
    """Add all feature columns. `units_column` is the units to price."""
    out = df.copy()
    out["t_years"] = [(p - start_period).n / 12 for p in out["period"]]
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    units = out[units_column].astype(float)
    out["grid_kwh"] = units
    for slab in SLABS:
        out[f"kwh_over_{slab}"] = np.maximum(units - slab, 0)
    return out
