"""All the assumptions in one place.

Every number the money calculation depends on lives here, so it is easy to
find, explain and change.
"""

from dataclasses import dataclass, asdict
#dataclass is the decorator

@dataclass
class Assumptions:
    capex: float = 170_000.0          # what the panels cost in total (Rs)
    tariff_escalation: float = 0.055  # electricity price rise per year (5.5%)
    discount_rate: float = 0.07       # what a fixed deposit would earn (7%) - used for NPV
    degradation_rate: float = 0.006   # panels lose 0.6% output per year
    annual_om_cost: float = 2_000.0   # cleaning and small repairs per year (Rs)
    inverter_cost: float = 25_000.0   # one inverter replacement (Rs)
    inverter_year: int = 12           # ...in this year of the system's life
    system_life_years: int = 25       # how long panels are expected to last

    def to_dict(self) -> dict:
        return asdict(self)

"""
It converts the assumptions dataclass into a dictionary using asdict. 
Right now nothing in the project calls it. I kept it for exporting the settings, for example as JSON from an API.
"""
# Indian grid: about 0.71 kg of CO2 per kWh (CEA CO2 baseline database).
CO2_KG_PER_KWH = 0.71

# Three simple "what if" scenarios instead of a full simulation.
# Each one changes how fast tariffs rise and how much sun the panels get.
SCENARIOS = {
    "Pessimistic": {"tariff_escalation": 0.035, "generation_factor": 0.90},
    "Expected": {"tariff_escalation": 0.055, "generation_factor": 1.00},
    "Optimistic": {"tariff_escalation": 0.075, "generation_factor": 1.10},
}

# Number of folds used when comparing models (TimeSeriesSplit).
CV_SPLITS = 5
