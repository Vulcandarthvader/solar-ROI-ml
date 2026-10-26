"""The money part: savings so far, a 25-year forecast, payback, NPV and ROI.

Rule: MEASURE THE PAST, FORECAST ONLY THE FUTURE.
  - Months that already happened use real savings.
  - Only the years after the last bill are forecast.
"""

import numpy as np
import pandas as pd

from .config import CO2_KG_PER_KWH, SCENARIOS, Assumptions


# ----------------------------------------------------------- the past ------

def monthly_savings(solar_df, model_a, model_b=None):
    """Saving each month = bill without solar - real bill.

    Two estimates (Model A and Model B) are averaged. How close they are
    tells us how much to trust the result.
    """
    df = solar_df.copy()
    df["no_solar_bill_A"] = model_a.bill_without_solar(df)
    df["saving_A"] = df["no_solar_bill_A"] - df["bill_amount"]
    if model_b is not None:
        df["no_solar_bill_B"] = model_b.bill_without_solar(df)
        df["saving_B"] = df["no_solar_bill_B"] - df["bill_amount"]
        df["saving"] = (df["saving_A"] + df["saving_B"]) / 2
    else:
        df["saving"] = df["saving_A"]
    df["cumulative_saving"] = df["saving"].cumsum()
    return df


def summarise_past(savings_df, capex):
    """Totals for the months that already happened.

    One arrears bill (money for other months landing in one month) would
    distort the total, so the total = average saving of normal months x
    number of months.
    """
    normal = savings_df[~savings_df["is_arrears"]]
    months = len(savings_df)
    total = normal["saving"].mean() * months
    kwh = savings_df["solar_kwh"].sum()

    # What one solar unit is worth today: last 12 normal months.
    last12 = normal.tail(12)
    value_per_kwh = last12["saving"].sum() / last12["solar_kwh"].sum()

    out = {
        "months": months,
        "years": months / 12,
        "total_saving": total,
        "avg_monthly_saving": total / months,
        "solar_kwh_so_far": kwh,
        "value_per_kwh": value_per_kwh,
        "capex_recovered_pct": 100 * total / capex,
        "co2_avoided_kg": kwh * CO2_KG_PER_KWH,
    }
    if "saving_B" in savings_df:
        a, b = normal["saving_A"].mean(), normal["saving_B"].mean()
        out["saving_A_avg"], out["saving_B_avg"] = a, b
        out["agreement_pct"] = 100 - abs(a - b) / ((a + b) / 2) * 100
    return out


# --------------------------------------------------------- the future ------

def typical_year_kwh(solar_df):
    """Expected kWh in each calendar month = average of that month so far.
    The first month is skipped because the panels started part-way through it."""
    full_months = solar_df.iloc[1:]
    by_month = full_months.groupby("month")["solar_kwh"].mean()
    return by_month, float(by_month.sum())


def forecast_cash_flows(past, annual_kwh, a: Assumptions, tariff_escalation=None, generation_factor=1.0):
    """One row per future year until the end of the system's life.

        kWh(k)        = annual_kwh x (1 - degradation)^k
        Rs per kWh(k) = today's value x (1 + tariff rise)^k
        cash(k)       = kWh x Rs per kWh - upkeep - inverter (once)
    """
    esc = a.tariff_escalation if tariff_escalation is None else tariff_escalation
    years_done = past["years"]
    n_future = int(a.system_life_years - years_done)   # whole years left in the system's life

    rows = []
    for k in range(1, n_future + 1):
        age_at_end = years_done + k                     # system age at end of this year
        kwh = annual_kwh * generation_factor * (1 - a.degradation_rate) ** k
        value = past["value_per_kwh"] * (1 + esc) ** k
        upkeep = a.annual_om_cost * (1 + esc) ** k
        inverter = a.inverter_cost if age_at_end - 1 < a.inverter_year <= age_at_end else 0.0
        rows.append({
            "year_of_life": round(age_at_end, 1),
            "kwh": kwh,
            "rs_per_kwh": value,
            "saving": kwh * value,
            "upkeep": upkeep,
            "inverter": inverter,
            "net_cash": kwh * value - upkeep - inverter,
        })
    table = pd.DataFrame(rows)
    table["cumulative"] = past["total_saving"] + table["net_cash"].cumsum()
    return table


def payback_years(past, cash_table, capex):
    """When does cumulative saving reach the capex?

    Find the year it crosses, then add the fraction of that year needed.
    Example: Rs 1,60,000 saved at the start of a year that earns Rs 20,000
    -> half that year is needed to reach Rs 1,70,000.
    """
    if past["total_saving"] >= capex:
        return past["years"] * capex / past["total_saving"]
    before = past["total_saving"]
    for i, row in cash_table.iterrows():
        if row["cumulative"] >= capex:
            fraction = (capex - before) / row["net_cash"]
            return past["years"] + i + fraction
        before = row["cumulative"]
    return float("nan")   # never pays back within the system's life


def npv(past, cash_table, a: Assumptions):
    """Net Present Value: all savings in today's money, minus the cost.

    Rs 107 next year is worth Rs 100 today if a deposit pays 7%.
    So each future year's cash is divided by (1 + 7%)^(years from install).
    Savings already received are counted at face value.
    """
    years_from_install = past["years"] + np.arange(1, len(cash_table) + 1)
    discounted = cash_table["net_cash"] / (1 + a.discount_rate) ** years_from_install
    return -a.capex + past["total_saving"] + discounted.sum()


def run_finance(past, annual_kwh, a: Assumptions):
    """Payback, NPV, ROI for the expected case, plus 3 scenarios."""
    cash = forecast_cash_flows(past, annual_kwh, a)
    lifetime_saving = past["total_saving"] + cash["net_cash"].sum()
    result = {
        "cash_table": cash,
        "payback_years": payback_years(past, cash, a.capex),
        "npv": npv(past, cash, a),
        "roi_pct": 100 * (lifetime_saving - a.capex) / a.capex,
        "lifetime_saving": lifetime_saving,
        "lifetime_co2_tonnes": (past["solar_kwh_so_far"] + cash["kwh"].sum()) * CO2_KG_PER_KWH / 1000,
    }

    scenario_rows = []
    for name, s in SCENARIOS.items():
        c = forecast_cash_flows(past, annual_kwh, a, s["tariff_escalation"], s["generation_factor"])
        scenario_rows.append({
            "scenario": name,
            "tariff rise": f"{s['tariff_escalation']:.1%}",
            "sunlight": f"{s['generation_factor']:.0%}",
            "payback (years)": payback_years(past, c, a.capex),
        })
    result["scenarios"] = pd.DataFrame(scenario_rows)
    return result
