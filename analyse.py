"""Run the whole analysis from the terminal, without the website.

    python analyse.py
    python analyse.py --capex 200000 --escalation 6
"""

import argparse
import glob
import warnings

from src.config import Assumptions
from src.pipeline import run_analysis
from src.vision import extract_many

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="When do my solar panels pay for themselves?")
parser.add_argument("--bills", default="data/billdata.csv")
parser.add_argument("--screenshots", default="data/samples")
parser.add_argument("--capex", type=float, default=170_000)
parser.add_argument("--escalation", type=float, default=5.5, help="tariff rise per year, in %%")
args = parser.parse_args()

# 1. Read screenshots
files = sorted(glob.glob(f"{args.screenshots}/*.png") + glob.glob(f"{args.screenshots}/*.jpg"))
rows, errors = extract_many(files)
print(f"Screenshots: {len(files)} files -> {len(rows)} monthly values")
for e in errors:
    print("  !", e)

# 2. Run everything
R = run_analysis(args.bills, rows, Assumptions(capex=args.capex, tariff_escalation=args.escalation / 100))
past = R["past"]

# 3. Print the results
print(f"\nData: {R['report']['months_after_cleaning']} months, panels installed {R['install_month']}")
s = R["screenshot_stats"]
print(f"Screenshot accuracy: {s['screenshot_error_pct']:.2f}% average error on {s['overlap_months']} metered months")

print(f"\nModel A (tariff):    {R['model_a'].name}")
print(R["model_a"].leaderboard.round(1).to_string(index=False))
if R["model_b"] is not None:
    print(f"\nModel B (pre-solar): {R['model_b'].name}")
    print(R["model_b"].leaderboard.round(1).to_string(index=False))

print("\nResults")
print(f"  Saved so far            Rs {past['total_saving']:,.0f}  ({past['capex_recovered_pct']:.1f}% of cost, {past['years']:.1f} years)")
print(f"  Average saving / month  Rs {past['avg_monthly_saving']:,.0f}")
if "agreement_pct" in past:
    print(f"  Model A vs B agreement  {past['agreement_pct']:.0f}%")
print(f"  Value of 1 solar unit   Rs {past['value_per_kwh']:.2f} per kWh")
print(f"  Typical year            {R['annual_kwh']:,.0f} kWh")
print(f"  Payback                 {R['payback_years']:.1f} years")
print(f"  NPV @ 7%                Rs {R['npv']:,.0f}   (negative = a 7% deposit would earn more)")
print(f"  25-year ROI             {R['roi_pct']:.0f}%")
print(f"  CO2 avoided so far      {past['co2_avoided_kg'] / 1000:.1f} tonnes")
print("\nScenarios")
table = R["scenarios"].copy()
table["payback (years)"] = table["payback (years)"].map(lambda v: "not within 25 yrs" if v != v else f"{v:.1f}")
print(table.to_string(index=False))
