"""Central configuration and economic assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# Month index of the first month with recorded solar generation.  Detected from
# the data at runtime; this is only the fallback.
DEFAULT_COMMISSIONING = (2023, 6)

# Grid emission factor for the Indian grid, CEA CO2 Baseline Database
# (~0.71 kg CO2 / kWh for the combined margin).
GRID_EMISSION_FACTOR_KG_PER_KWH = 0.71


@dataclass
class FinancialAssumptions:
    """Assumptions behind every ROI number the app reports."""

    capex: float = 170_000.0
    """Total installed cost in INR (panels, inverter, structure, labour)."""

    discount_rate: float = 0.07
    """Real discount rate. Roughly a post-tax fixed-deposit / debt-fund return,
    i.e. the opportunity cost of the money sunk into the array."""

    tariff_escalation: float = 0.055
    """Annual electricity tariff escalation. Estimated from the user's own
    bills at runtime; this is the prior used when history is too short."""

    degradation_rate: float = 0.006
    """Annual panel output loss. Typical linear-warranty slope for mono-PERC
    modules is 0.5-0.7 %/yr after first-year settling."""

    annual_om_cost: float = 2_000.0
    """Cleaning, minor repairs and inspection, in INR/year."""

    inverter_replacement_cost: float = 25_000.0
    inverter_replacement_year: int = 12
    """String inverters typically need replacing once inside a 25-year life."""

    system_life_years: int = 25

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MonteCarloAssumptions:
    """Uncertainty ranges for the payback simulation.

    Spreads are deliberately wide: the point of the simulation is to show how
    fragile a single-number payback estimate is, not to manufacture precision.
    """

    n_simulations: int = 10_000
    generation_sigma: float = 0.08
    """Year-on-year generation variability (monsoon strength, soiling)."""

    tariff_escalation_sigma: float = 0.02
    """Uncertainty on the annual tariff escalation rate."""

    degradation_mean: float = 0.006
    degradation_sigma: float = 0.002

    om_cost_sigma: float = 0.30
    """Relative uncertainty on annual O&M spend."""

    outage_prob: float = 0.03
    """Per-year probability of a fault costing a chunk of the year's output."""

    outage_loss: float = 0.25
    """Fraction of annual generation lost when a fault occurs."""

    seed: int = 42


# Model-selection settings shared by every supervised model in the project.
CV_SPLITS = 5
RANDOM_STATE = 42
