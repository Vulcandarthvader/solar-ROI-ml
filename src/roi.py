"""Financial engine: realised savings, 25-year cash flows, and payback risk.

Design principle: **measure the past, model only the future.**  Months that
have already happened contribute their realised savings; the projection starts
the month after the last reading.  A payback estimate built entirely from
projections is a spreadsheet fantasy, and one built entirely from history
cannot see past the last bill.

Every headline figure is reported alongside its uncertainty.  The Monte Carlo
exists because a single payback year, quoted to one decimal place, implies a
precision that a solar array in a monsoon climate does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    GRID_EMISSION_FACTOR_KG_PER_KWH,
    FinancialAssumptions,
    MonteCarloAssumptions,
)


# --------------------------------------------------------------------------- #
# Realised savings
# --------------------------------------------------------------------------- #


def historical_savings(
    solar_df: pd.DataFrame, tariff_model, synthetic=None
) -> pd.DataFrame:
    """Month-by-month realised savings since commissioning.

    Two independent counterfactuals are computed where possible.  Their
    agreement is itself a result: two methods resting on different assumptions
    landing in the same place is far more persuasive than one method quoted
    confidently.
    """
    df = solar_df.copy()
    df["counterfactual_bill_tariff"] = tariff_model.counterfactual_bill(df)
    df["marginal_rate"] = tariff_model.marginal_rate(df)

    if synthetic is not None:
        df["counterfactual_bill_synthetic"] = synthetic.predict(df)
    else:
        df["counterfactual_bill_synthetic"] = np.nan

    df["savings_tariff"] = df["counterfactual_bill_tariff"] - df["bill_amount"]
    df["savings_synthetic"] = df["counterfactual_bill_synthetic"] - df["bill_amount"]

    both = df[["savings_tariff", "savings_synthetic"]]
    df["savings"] = both.mean(axis=1, skipna=True)
    df["savings_spread"] = (df["savings_tariff"] - df["savings_synthetic"]).abs()

    # A month can genuinely go negative (a credit-adjustment bill, or a
    # correction), so no clipping - but the value per unit is only meaningful
    # where the array actually produced something.
    df["value_per_kwh"] = np.where(df["solar_kwh"] > 0, df["savings"] / df["solar_kwh"], np.nan)
    if "is_billing_anomaly" not in df:
        df["is_billing_anomaly"] = False
    df["cumulative_savings"] = df["savings"].cumsum()
    df["cumulative_kwh"] = df["solar_kwh"].cumsum()
    return df


def realised_summary(hist: pd.DataFrame, capex: float) -> dict:
    """Aggregate the realised era, robust to arrears and true-up bills.

    Summing the savings column naively lets a single arrears bill - money that
    belongs to earlier months but landed in one - swallow a fifth of the
    measured return.  The headline total is therefore the mean saving over
    *clean* months scaled to the observed span, with the raw as-billed figure
    kept alongside it so nothing is hidden.
    """
    months = len(hist)
    as_billed = float(hist["savings"].sum())
    clean = hist[~hist.get("is_billing_anomaly", pd.Series(False, index=hist.index)).fillna(False)]
    if len(clean) >= max(6, months // 2):
        total = float(clean["savings"].mean() * months)
    else:  # too few clean months to extrapolate from; fall back to as-billed
        total = as_billed

    kwh = float(hist["solar_kwh"].sum())
    last12 = clean.tail(12) if len(clean) >= 12 else hist.tail(12)
    return {
        "months_observed": months,
        "years_observed": months / 12,
        "total_savings": total,
        "total_savings_as_billed": as_billed,
        "n_anomalous_months": int(months - len(clean)),
        "total_generation_kwh": kwh,
        "avg_monthly_savings": total / months if months else 0.0,
        "last12_savings": float(last12["savings"].mean() * 12) if len(last12) else 0.0,
        "last12_generation_kwh": float(last12["solar_kwh"].sum()),
        "realised_value_per_kwh": total / kwh if kwh else np.nan,
        "capex_recovered_pct": 100 * total / capex if capex else np.nan,
        "capex_outstanding": capex - total,
        "co2_avoided_kg": kwh * GRID_EMISSION_FACTOR_KG_PER_KWH,
        "counterfactual_agreement_pct": float(
            100 - (hist["savings_spread"].mean() / max(abs(hist["savings"].mean()), 1e-9) * 100)
        )
        if hist["savings_synthetic"].notna().any()
        else np.nan,
    }


# --------------------------------------------------------------------------- #
# Forward projection
# --------------------------------------------------------------------------- #


def project_cashflows(
    projected_generation: pd.DataFrame,
    value_per_kwh: float,
    assumptions: FinancialAssumptions,
    start_year_offset: float = 0.0,
) -> pd.DataFrame:
    """Turn projected monthly kWh into annual net cash flows.

    ``start_year_offset`` is the array's age when the projection begins, so
    escalation and the inverter replacement stay pinned to the install date
    rather than to the first projected month.
    """
    gen = projected_generation.copy()
    gen["year_index"] = np.floor(start_year_offset + np.arange(len(gen)) / 12).astype(int)

    rows = []
    for k, grp in gen.groupby("year_index"):
        kwh = float(grp["solar_kwh"].sum())
        rate = value_per_kwh * (1 + assumptions.tariff_escalation) ** k
        gross = kwh * rate
        om = assumptions.annual_om_cost * (1 + assumptions.tariff_escalation) ** k
        capex_event = (
            assumptions.inverter_replacement_cost
            if k + 1 == assumptions.inverter_replacement_year
            else 0.0
        )
        rows.append(
            {
                "year_index": k,
                "generation_kwh": kwh,
                "value_per_kwh": rate,
                "gross_savings": gross,
                "om_cost": om,
                "capex_event": capex_event,
                "net_cashflow": gross - om - capex_event,
                "months": len(grp),
            }
        )
    return pd.DataFrame(rows)


def npv(cashflows: np.ndarray, rate: float, capex: float) -> float:
    """Net present value of the whole project, capex at t=0."""
    t = np.arange(1, len(cashflows) + 1)
    return float(-capex + np.sum(cashflows / (1 + rate) ** t))


def irr(cashflows: np.ndarray, capex: float, lo: float = -0.9, hi: float = 1.5) -> float:
    """Internal rate of return by bisection.

    Bisection rather than a polynomial root-finder: the cash-flow sign pattern
    here is a single sign change, so the root is unique and bracketing is
    guaranteed - no risk of returning a complex or spurious root.
    """
    f = lambda r: npv(cashflows, r, capex)  # noqa: E731
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float((lo + hi) / 2)


def lcoe(
    projected_generation_kwh: np.ndarray, assumptions: FinancialAssumptions, realised_kwh: float = 0.0
) -> float:
    """Levelised cost of energy: what a unit from this roof really costs.

    Comparing LCOE with the grid tariff is the cleanest one-number answer to
    "was this a good idea".
    """
    r = assumptions.discount_rate
    t = np.arange(1, len(projected_generation_kwh) + 1)
    disc_kwh = np.sum(projected_generation_kwh / (1 + r) ** t) + realised_kwh
    om = assumptions.annual_om_cost * (1 + assumptions.tariff_escalation) ** (t - 1)
    inv = np.zeros(len(t))
    if assumptions.inverter_replacement_year <= len(t):
        inv[assumptions.inverter_replacement_year - 1] = assumptions.inverter_replacement_cost
    disc_cost = assumptions.capex + np.sum((om + inv) / (1 + r) ** t)
    return float(disc_cost / disc_kwh) if disc_kwh > 0 else float("nan")


def _crossing_year(cumulative: np.ndarray, target: float) -> float:
    """Elapsed years at which a cumulative annual series first reaches ``target``.

    ``cumulative[i]`` is the balance at the *end* of year ``i + 1``, so a
    crossing at index ``i`` means ``i`` whole years have passed plus the
    fraction of year ``i + 1`` needed to close the gap.  Adding a further year
    here is an off-by-one that quietly reports payback a year late.
    """
    idx = int(np.argmax(cumulative >= target))
    if cumulative[idx] < target:
        return float("nan")
    prev = cumulative[idx - 1] if idx > 0 else 0.0
    step = cumulative[idx] - prev
    frac = (target - prev) / step if step > 0 else 0.0
    return float(idx + frac)


@dataclass
class PaybackResult:
    simple_payback_years: float
    discounted_payback_years: float
    npv: float
    irr: float
    lcoe: float
    roi_pct_25y: float
    total_lifetime_savings: float
    co2_lifetime_tonnes: float
    realised_years: float
    realised_savings: float
    schedule: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("schedule")
        return d


def compute_payback(
    realised: dict,
    projection: pd.DataFrame,
    assumptions: FinancialAssumptions,
) -> PaybackResult:
    """Blend realised savings with projected cash flows into ROI metrics."""
    realised_years = realised["years_observed"]
    realised_total = realised["total_savings"]

    # Annual view of the realised era, so the schedule is continuous.
    n_full = int(np.floor(realised_years))
    realised_annual = [realised_total / realised_years] * n_full if realised_years >= 1 else []
    stub = realised_total - sum(realised_annual)

    net = projection["net_cashflow"].to_numpy()
    if stub > 0 and len(net):
        net = net.copy()
        net[0] += stub  # partial first year folded into the next full year

    all_flows = np.concatenate([np.array(realised_annual), net])
    cumulative = np.cumsum(all_flows)

    simple = _crossing_year(cumulative, assumptions.capex)

    r = assumptions.discount_rate
    t = np.arange(1, len(all_flows) + 1)
    disc_cum = np.cumsum(all_flows / (1 + r) ** t)
    discounted = _crossing_year(disc_cum, assumptions.capex)

    horizon = assumptions.system_life_years
    flows_h = all_flows[:horizon]
    total_savings = float(np.sum(flows_h))
    gen_h = np.concatenate(
        [
            np.full(len(realised_annual), realised["total_generation_kwh"] / max(realised_years, 1e-9)),
            projection["generation_kwh"].to_numpy(),
        ]
    )[:horizon]

    schedule = pd.DataFrame(
        {
            "year": np.arange(1, len(flows_h) + 1),
            "net_cashflow": flows_h,
            "cumulative": np.cumsum(flows_h),
            "cumulative_discounted": np.cumsum(flows_h / (1 + r) ** np.arange(1, len(flows_h) + 1)),
            "generation_kwh": gen_h,
            "source": ["realised"] * len(realised_annual) + ["projected"] * (len(flows_h) - len(realised_annual)),
        }
    )
    schedule["remaining_to_breakeven"] = (assumptions.capex - schedule["cumulative"]).clip(lower=0)

    return PaybackResult(
        simple_payback_years=simple,
        discounted_payback_years=discounted,
        npv=npv(flows_h, r, assumptions.capex),
        irr=irr(flows_h, assumptions.capex),
        lcoe=lcoe(projection["generation_kwh"].to_numpy()[: max(horizon - len(realised_annual), 1)], assumptions, realised["total_generation_kwh"]),
        roi_pct_25y=100 * (total_savings - assumptions.capex) / assumptions.capex,
        total_lifetime_savings=total_savings,
        co2_lifetime_tonnes=float(np.sum(gen_h) * GRID_EMISSION_FACTOR_KG_PER_KWH / 1000),
        realised_years=realised_years,
        realised_savings=realised_total,
        schedule=schedule,
    )


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #


@dataclass
class MonteCarloResult:
    paybacks: np.ndarray
    npvs: np.ndarray
    p10: float
    p50: float
    p90: float
    prob_within_life: float
    prob_positive_npv: float
    mean_npv: float

    def summary(self) -> dict:
        return {
            "P10_payback_years": self.p10,
            "P50_payback_years": self.p50,
            "P90_payback_years": self.p90,
            "prob_payback_within_life_%": self.prob_within_life * 100,
            "prob_positive_npv_%": self.prob_positive_npv * 100,
            "mean_npv": self.mean_npv,
        }


def monte_carlo_payback(
    realised: dict,
    base_annual_kwh: float,
    value_per_kwh: float,
    assumptions: FinancialAssumptions,
    mc: MonteCarloAssumptions | None = None,
) -> MonteCarloResult:
    """Simulate payback under uncertain weather, tariffs, degradation and faults.

    Vectorised over simulations: 10,000 paths x 25 years is a single array
    operation, which keeps the UI responsive enough to re-run on every slider
    change.

    The realised era is held **fixed** - it already happened, and pretending
    otherwise would launder measured savings into simulated ones.
    """
    mc = mc or MonteCarloAssumptions()
    rng = np.random.default_rng(mc.seed)
    n, life = mc.n_simulations, assumptions.system_life_years

    realised_years = realised["years_observed"]
    realised_total = realised["total_savings"]
    n_realised = int(np.floor(realised_years))
    remaining = max(life - n_realised, 1)

    esc = rng.normal(assumptions.tariff_escalation, mc.tariff_escalation_sigma, n)[:, None]
    deg = np.clip(rng.normal(mc.degradation_mean, mc.degradation_sigma, n), 0.0, 0.05)[:, None]
    om_mult = np.clip(rng.normal(1.0, mc.om_cost_sigma, n), 0.2, 3.0)[:, None]

    k = np.arange(n_realised, n_realised + remaining)[None, :]

    # Weather / soiling: multiplicative lognormal noise, one draw per year.
    weather = rng.lognormal(-0.5 * mc.generation_sigma**2, mc.generation_sigma, (n, remaining))
    # Faults: an outage costs a slice of that year's output.
    outage = rng.random((n, remaining)) < mc.outage_prob
    weather *= np.where(outage, 1 - mc.outage_loss, 1.0)

    gen = base_annual_kwh * (1 - deg) ** k * weather
    rate = value_per_kwh * (1 + esc) ** k
    om = assumptions.annual_om_cost * om_mult * (1 + esc) ** k

    inv = np.zeros((1, remaining))
    inv_idx = assumptions.inverter_replacement_year - 1 - n_realised
    if 0 <= inv_idx < remaining:
        inv[0, inv_idx] = assumptions.inverter_replacement_cost

    flows = gen * rate - om - inv
    realised_block = np.full((n, n_realised), realised_total / max(n_realised, 1)) if n_realised else np.zeros((n, 0))
    all_flows = np.concatenate([realised_block, flows], axis=1)
    cum = np.cumsum(all_flows, axis=1)

    reached = cum >= assumptions.capex
    first = np.argmax(reached, axis=1).astype(float)
    ever = reached.any(axis=1)

    prev = np.where(first > 0, cum[np.arange(n), np.maximum(first.astype(int) - 1, 0)], 0.0)
    hit = cum[np.arange(n), first.astype(int)]
    step = np.where(hit - prev > 0, hit - prev, 1.0)
    frac = (assumptions.capex - prev) / step
    payback = np.where(ever, first + frac, np.nan)

    t = np.arange(1, all_flows.shape[1] + 1)
    npvs = -assumptions.capex + np.sum(all_flows / (1 + assumptions.discount_rate) ** t, axis=1)

    valid = payback[~np.isnan(payback)]
    return MonteCarloResult(
        paybacks=payback,
        npvs=npvs,
        p10=float(np.percentile(valid, 10)) if len(valid) else float("nan"),
        p50=float(np.percentile(valid, 50)) if len(valid) else float("nan"),
        p90=float(np.percentile(valid, 90)) if len(valid) else float("nan"),
        prob_within_life=float(ever.mean()),
        prob_positive_npv=float((npvs > 0).mean()),
        mean_npv=float(npvs.mean()),
    )


def deterministic_payback(
    realised: dict,
    base_annual_kwh: float,
    value_per_kwh: float,
    assumptions: FinancialAssumptions,
    discounted: bool = False,
) -> float:
    """Closed-form payback with no simulation noise.

    Shares the accounting rules of the Monte Carlo (realised years fixed,
    escalating tariff, degrading output, inverter replacement) but with every
    random draw set to its mean, so it is the right tool for one-at-a-time
    sensitivity work where sampling noise would swamp the effect being
    measured.
    """
    life = assumptions.system_life_years
    realised_years = realised["years_observed"]
    n_realised = int(np.floor(realised_years))
    remaining = max(life - n_realised, 1)

    k = np.arange(n_realised, n_realised + remaining)
    gen = base_annual_kwh * (1 - assumptions.degradation_rate) ** k
    rate = value_per_kwh * (1 + assumptions.tariff_escalation) ** k
    om = assumptions.annual_om_cost * (1 + assumptions.tariff_escalation) ** k

    inv = np.zeros(remaining)
    inv_idx = assumptions.inverter_replacement_year - 1 - n_realised
    if 0 <= inv_idx < remaining:
        inv[inv_idx] = assumptions.inverter_replacement_cost

    flows = gen * rate - om - inv
    realised_block = (
        np.full(n_realised, realised["total_savings"] / max(n_realised, 1)) if n_realised else np.zeros(0)
    )
    all_flows = np.concatenate([realised_block, flows])
    if discounted:
        t = np.arange(1, len(all_flows) + 1)
        all_flows = all_flows / (1 + assumptions.discount_rate) ** t
    return _crossing_year(np.cumsum(all_flows), assumptions.capex)


def sensitivity_analysis(
    realised: dict,
    base_annual_kwh: float,
    value_per_kwh: float,
    assumptions: FinancialAssumptions,
) -> pd.DataFrame:
    """One-at-a-time tornado analysis around the base payback estimate.

    Complements the Monte Carlo: the simulation says how wide the outcome is,
    this says *which* assumption is responsible for the width.  Note that the
    discount rate is absent - it moves NPV and discounted payback, but simple
    cash payback is a question about rupees in hand, not their present value.
    """

    def pb(**over) -> float:
        gen = over.pop("_gen", base_annual_kwh)
        rate = over.pop("_rate", value_per_kwh)
        a = FinancialAssumptions(**{**assumptions.to_dict(), **over})
        return deterministic_payback(realised, gen, rate, a)

    base = pb()
    rows = [
        {"factor": "Tariff escalation: 3.5% vs 7.5%",
         "worse": pb(tariff_escalation=assumptions.tariff_escalation - 0.02),
         "better": pb(tariff_escalation=assumptions.tariff_escalation + 0.02)},
        {"factor": "Capex +/- 15%",
         "worse": pb(capex=assumptions.capex * 1.15),
         "better": pb(capex=assumptions.capex * 0.85)},
        {"factor": "Annual generation +/- 10%",
         "worse": pb(_gen=base_annual_kwh * 0.90),
         "better": pb(_gen=base_annual_kwh * 1.10)},
        {"factor": "Value per unit +/- 10%",
         "worse": pb(_rate=value_per_kwh * 0.90),
         "better": pb(_rate=value_per_kwh * 1.10)},
        {"factor": "O&M +/- 50%",
         "worse": pb(annual_om_cost=assumptions.annual_om_cost * 1.5),
         "better": pb(annual_om_cost=assumptions.annual_om_cost * 0.5)},
        {"factor": "Degradation 1.0% vs 0.3%/yr",
         "worse": pb(degradation_rate=0.010),
         "better": pb(degradation_rate=0.003)},
        {"factor": "Inverter swap 40k vs none",
         "worse": pb(inverter_replacement_cost=40_000),
         "better": pb(inverter_replacement_cost=0)},
    ]
    out = pd.DataFrame(rows)
    out["base"] = base
    out["swing"] = (out["worse"] - out["better"]).abs()
    return out[["factor", "worse", "base", "better", "swing"]].sort_values(
        "swing", ascending=False
    ).reset_index(drop=True)
