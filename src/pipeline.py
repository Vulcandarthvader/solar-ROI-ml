"""End-to-end orchestration.

One function, ``run_analysis``, takes raw inputs and returns every artefact the
UI, the notebook and the tests need.  Keeping the pipeline here rather than in
the Streamlit script means the analysis is testable headlessly and identical
whether it is driven by the app or by a script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FinancialAssumptions, MonteCarloAssumptions
from .ingest import DataQualityReport, clean_bills, load_bills, merge_generation, modelling_frame
from .models import (
    GenerationModel,
    SyntheticCounterfactual,
    TariffModel,
    detect_anomalies,
)
from .roi import (
    MonteCarloResult,
    PaybackResult,
    compute_payback,
    historical_savings,
    monte_carlo_payback,
    project_cashflows,
    realised_summary,
    sensitivity_analysis,
)


@dataclass
class AnalysisResult:
    bills: pd.DataFrame
    quality: DataQualityReport
    extraction_stats: dict
    commissioning: pd.Period
    tariff_model: TariffModel
    synthetic: SyntheticCounterfactual | None
    generation_fit: object
    history: pd.DataFrame
    realised: dict
    anomalies: pd.DataFrame
    projection: pd.DataFrame
    cashflows: pd.DataFrame
    payback: PaybackResult
    monte_carlo: MonteCarloResult
    sensitivity: pd.DataFrame
    assumptions: FinancialAssumptions
    value_per_kwh: float
    warnings: list[str] = field(default_factory=list)

    @property
    def leaderboards(self) -> dict[str, pd.DataFrame]:
        out = {"Tariff model (bill vs units)": self.tariff_model.fitted.leaderboard}
        if self.synthetic is not None:
            out["Synthetic counterfactual (pre-solar bills)"] = self.synthetic.fitted.leaderboard
        out["Generation model (monthly kWh)"] = self.generation_fit.model.leaderboard
        return out

    def headline(self) -> dict:
        p = self.payback
        return {
            "Payback (blended, years)": p.simple_payback_years,
            "Payback P50 (simulated)": self.monte_carlo.p50,
            "Payback P10 - P90": f"{self.monte_carlo.p10:.1f} - {self.monte_carlo.p90:.1f}",
            "Capex recovered so far (%)": self.realised["capex_recovered_pct"],
            "Realised savings to date (INR)": self.realised["total_savings"],
            "NPV @ 7% (INR)": p.npv,
            "IRR (%)": p.irr * 100 if np.isfinite(p.irr) else np.nan,
            "LCOE (INR/kWh)": p.lcoe,
            "25-year ROI (%)": p.roi_pct_25y,
            "CO2 avoided to date (kg)": self.realised["co2_avoided_kg"],
        }


def run_analysis(
    bill_source,
    extracted_generation: pd.DataFrame | None = None,
    assumptions: FinancialAssumptions | None = None,
    mc_assumptions: MonteCarloAssumptions | None = None,
) -> AnalysisResult:
    """Run the full pipeline from a bill CSV plus optional extracted generation."""
    assumptions = assumptions or FinancialAssumptions()
    warnings: list[str] = []

    raw = load_bills(bill_source)
    bills, quality = clean_bills(raw)
    bills, ext_stats = merge_generation(bills, extracted_generation)

    solar_rows = bills[bills["solar_kwh"] > 0]
    if solar_rows.empty:
        raise ValueError(
            "No solar generation found in either the bill CSV or the screenshots, "
            "so there is no return to analyse."
        )
    commissioning = solar_rows["period"].iloc[0]
    origin = bills["period"].iloc[0]

    # --- Estimator A: tariff surrogate --------------------------------------
    model_df = modelling_frame(bills)
    if len(model_df) < 24:
        warnings.append(
            f"Only {len(model_df)} clean months available to fit the tariff model; "
            "results will be sensitive to individual bills."
        )
    tariff = TariffModel(origin=origin).fit(model_df)

    # --- Estimator B: synthetic counterfactual ------------------------------
    pre = model_df[model_df["period"] < commissioning]
    synthetic = None
    if len(pre) >= 24:
        synthetic = SyntheticCounterfactual(origin=origin).fit(pre)
    else:
        warnings.append(
            "Fewer than 24 clean pre-solar months, so the independent synthetic "
            "counterfactual was skipped; savings rest on the tariff model alone."
        )

    # --- Realised savings ----------------------------------------------------
    solar_era = bills[bills["period"] >= commissioning].copy()
    billed = solar_era[solar_era["bill_amount"].notna()].copy()
    history = historical_savings(billed, tariff, synthetic)
    realised = realised_summary(history, assumptions.capex)

    # --- Generation model, projection and anomalies -------------------------
    gen_rows = bills[bills["solar_kwh"] > 0].copy()
    gen_model = GenerationModel(commissioning=commissioning, origin=origin)
    gen_fit = gen_model.fit(gen_rows)
    anomalies = detect_anomalies(gen_fit)

    # The projection must start where the *realised* accounting stops - the
    # month after the last bill - or the months that have generation data but no
    # bill yet fall into neither half and a year of value vanishes.
    start = history["period"].max() + 1
    age_at_start = (start - commissioning).n / 12.0
    months_left = max(int(round((assumptions.system_life_years - age_at_start) * 12)), 12)
    projection = gen_model.project(start, months_left, assumptions.degradation_rate)

    # Where the screenshots already measured a month the projection covers, the
    # measurement replaces the model's guess.
    measured = gen_rows.set_index("period")["solar_kwh"]
    overlap = projection["period"].isin(measured.index)
    if overlap.any():
        projection.loc[overlap, "solar_kwh"] = measured.reindex(projection.loc[overlap, "period"]).to_numpy()

    # Value of a generated unit: measured, not assumed.  The last 12 months
    # reflect the current tariff slab the household actually sits in.
    clean_history = history[~history["is_billing_anomaly"].fillna(False)]
    recent = (clean_history if len(clean_history) >= 12 else history).tail(12)
    value_per_kwh = float(
        recent["savings"].sum() / recent["solar_kwh"].sum()
        if recent["solar_kwh"].sum() > 0
        else realised["realised_value_per_kwh"]
    )
    if not np.isfinite(value_per_kwh) or value_per_kwh <= 0:
        value_per_kwh = float(np.nanmedian(history["marginal_rate"]))
        warnings.append("Realised value per unit was not usable; fell back to the modelled marginal rate.")

    cashflows = project_cashflows(projection, value_per_kwh, assumptions, start_year_offset=age_at_start)
    payback = compute_payback(realised, cashflows, assumptions)

    mc = monte_carlo_payback(
        realised,
        base_annual_kwh=gen_fit.annual_yield_kwh,
        value_per_kwh=value_per_kwh,
        assumptions=assumptions,
        mc=mc_assumptions,
    )
    sens = sensitivity_analysis(realised, gen_fit.annual_yield_kwh, value_per_kwh, assumptions)

    n_out = tariff.out_of_range_months(history)
    if n_out:
        warnings.append(
            f"{n_out} month(s) have a gross demand outside the {tariff.units_range[0]:.0f}-"
            f"{tariff.units_range[1]:.0f} kWh range the tariff model was fitted on; the "
            "counterfactual for those months is an extrapolation."
        )

    if ext_stats.get("overlap_n", 0) >= 6 and ext_stats.get("overlap_mape", 100) > 10:
        warnings.append(
            f"Screenshot-extracted generation disagrees with the CSV by "
            f"{ext_stats['overlap_mape']:.1f}% on overlapping months - check the extraction."
        )

    return AnalysisResult(
        bills=bills,
        quality=quality,
        extraction_stats=ext_stats,
        commissioning=commissioning,
        tariff_model=tariff,
        synthetic=synthetic,
        generation_fit=gen_fit,
        history=history,
        realised=realised,
        anomalies=anomalies,
        projection=projection,
        cashflows=cashflows,
        payback=payback,
        monte_carlo=mc,
        sensitivity=sens,
        assumptions=assumptions,
        value_per_kwh=value_per_kwh,
        warnings=warnings,
    )


def extractions_to_frame(extractions: list) -> pd.DataFrame:
    """Flatten ChartExtraction objects into a tidy generation frame."""
    rows = []
    for e in extractions:
        rows.extend(e.as_records())
    if not rows:
        return pd.DataFrame(columns=["year", "month", "solar_kwh"])
    df = pd.DataFrame(rows)
    return df.dropna(subset=["month"]) if "month" in df else df
