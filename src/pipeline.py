"""Runs every step in order. Both app.py and analyse.py call run_analysis(),
so the website and the command line always give the same numbers.

    1. Load + clean bills                       (cleaning.py)
    2. Add screenshot generation                (vision.py -> cleaning.py)
    3. Find the month the panels started
    4. Train Model A (tariff) and Model B (pre-solar)   (models.py)
    5. Monthly savings so far                   (finance.py)
    6. Typical year of generation
    7. 25-year cash flow, payback, NPV, ROI, scenarios (finance.py)
"""

from .cleaning import clean_bills, load_bills, merge_screenshots, training_rows
from .config import Assumptions
from .finance import monthly_savings, run_finance, summarise_past, typical_year_kwh
from .models import PreSolarModel, TariffModel


def run_analysis(bill_csv, screenshot_rows=None, assumptions=None):
    a = assumptions or Assumptions()

    # 1-2. Data
    bills, report = clean_bills(load_bills(bill_csv))
    bills, screenshot_stats = merge_screenshots(bills, screenshot_rows)

    # 3. Install month = first month with any solar generation
    solar_months = bills[bills["solar_kwh"] > 0]
    if solar_months.empty:
        raise ValueError("No solar generation found, so there is nothing to analyse.")
    install = solar_months["period"].iloc[0]
    start = bills["period"].iloc[0]

    # 4. Models
    train = training_rows(bills)
    model_a = TariffModel().fit(train, start)
    pre_solar = train[train["period"] < install]
    model_b = PreSolarModel().fit(pre_solar, start) if len(pre_solar) >= 24 else None

    # 5. Savings in the months that have a bill
    billed_solar = bills[(bills["period"] >= install) & bills["bill_amount"].notna()]
    savings = monthly_savings(billed_solar, model_a, model_b)
    past = summarise_past(savings, a.capex)

    # 6. Generation: typical year, from every month with solar (incl. screenshots)
    by_month, annual_kwh = typical_year_kwh(solar_months)

    # 7. Money
    money = run_finance(past, annual_kwh, a)

    return {
        "assumptions": a,
        "bills": bills,
        "report": report,
        "screenshot_stats": screenshot_stats,
        "install_month": str(install),
        "model_a": model_a,
        "model_b": model_b,
        "savings": savings,
        "past": past,
        "kwh_by_month": by_month,
        "annual_kwh": annual_kwh,
        "pre_solar_avg_grid_kwh": float(bills.loc[bills["period"] < install, "grid_kwh"].mean()),
        "post_solar_avg_grid_kwh": float(billed_solar["grid_kwh"].mean()),
        **money,
    }
