"""Tests for model selection, projection and the financial engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FinancialAssumptions, MonteCarloAssumptions
from src.features import add_calendar_features, build_generation_frame
from src.models import (
    EXTRAPOLATING_MODELS,
    GenerationModel,
    SyntheticCounterfactual,
    detect_anomalies,
    evaluate_models,
    select_and_fit,
)
from src.roi import (
    _crossing_year,
    deterministic_payback,
    irr,
    monte_carlo_payback,
    npv,
)


def _monthly_frame(n_months: int, start="2018-01", trend=5.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    season = 200 + 60 * np.sin(2 * np.pi * idx.month / 12)
    return pd.DataFrame(
        {
            "period": idx,
            "year": idx.year,
            "month": idx.month,
            "bill_amount": season * 3 + trend * np.arange(n_months) + rng.normal(0, 20, n_months),
            "solar_kwh": np.maximum(season + rng.normal(0, 12, n_months), 0),
            "grid_kwh": season,
        }
    )


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #


def test_cross_validation_returns_a_ranked_leaderboard():
    df = add_calendar_features(_monthly_frame(72))
    board, oof = evaluate_models(df[["t_years", "month_sin", "month_cos"]], df["bill_amount"])
    assert board["MAE"].is_monotonic_increasing, "leaderboard must be sorted best-first"
    assert {"MAE", "RMSE", "MAPE_%", "R2"} <= set(board.columns)
    # Out-of-fold predictions exist for the later folds and are absent for the
    # first block, which is training-only under a rolling origin.
    for preds in oof.values():
        assert np.isnan(preds[0]), "the earliest months must never be predicted out-of-fold"
        assert not np.isnan(preds[-1])


def test_extrapolation_requirement_excludes_tree_models():
    df = add_calendar_features(_monthly_frame(72))
    fit = select_and_fit(
        df, df["bill_amount"], ["t_years", "month_sin", "month_cos"], "bill_amount",
        require_extrapolation=True,
    )
    assert fit.name in EXTRAPOLATING_MODELS
    # The excluded models still appear, so the trade-off stays auditable.
    assert (~fit.leaderboard["selectable"]).any()
    assert fit.leaderboard["selected"].sum() == 1


def test_counterfactual_keeps_rising_past_the_training_window():
    """A tree would flat-line here; the point of the restriction is that it doesn't."""
    train = _monthly_frame(72, trend=8.0)
    model = SyntheticCounterfactual().fit(train)
    future = _monthly_frame(24, start="2024-01", trend=8.0)
    preds = model.predict(future)
    early, late = preds[:12].mean(), preds[12:].mean()
    assert late > early, "the fitted trend must continue into unseen years"


# --------------------------------------------------------------------------- #
# Generation and anomalies
# --------------------------------------------------------------------------- #


def test_projection_declines_at_the_configured_degradation_rate():
    hist = _monthly_frame(36, start="2023-01")
    commissioning = pd.Period("2023-01", freq="M")
    gm = GenerationModel(commissioning=commissioning)
    gm.fit(hist)
    proj = gm.project(pd.Period("2026-01", freq="M"), 240, degradation=0.01)
    first_year = proj.iloc[:12]["solar_kwh"].sum()
    tenth_year = proj.iloc[108:120]["solar_kwh"].sum()
    ratio = tenth_year / first_year
    assert 0.86 < ratio < 0.94, f"10 years at 1%/yr should lose ~9%, got {1 - ratio:.1%}"


def test_anomaly_detector_flags_a_real_shortfall_and_spares_the_first_month():
    hist = _monthly_frame(36, start="2023-01", seed=3)
    hist.loc[20, "solar_kwh"] *= 0.45  # a month with a fault
    gm = GenerationModel(commissioning=pd.Period("2023-01", freq="M"))
    fit = gm.fit(hist)
    flags = detect_anomalies(fit)

    assert flags.iloc[0]["severity"] == "ramp-up", "commissioning month is partial by construction"
    assert flags.iloc[20]["severity"] in {"high", "watch"}
    assert flags.iloc[20]["shortfall_pct"] > 20
    # An unusually good month is not a maintenance alarm.
    assert "high" not in flags[flags["shortfall_pct"] < 0]["severity"].tolist()


# --------------------------------------------------------------------------- #
# Financial engine
# --------------------------------------------------------------------------- #


def test_npv_and_irr_agree_by_construction():
    flows = np.full(20, 15_000.0)
    rate = irr(flows, capex=150_000)
    assert np.isfinite(rate)
    assert npv(flows, rate, 150_000) == pytest.approx(0.0, abs=1e-3)


def test_crossing_year_interpolates_within_the_year():
    cumulative = np.array([40.0, 80.0, 120.0])
    # 100 sits exactly halfway through the third year's 40-unit step, i.e. two
    # whole years plus half of the third.
    assert _crossing_year(cumulative, 100.0) == pytest.approx(2.5)
    # Hitting the target exactly at a year end reports that whole year.
    assert _crossing_year(cumulative, 40.0) == pytest.approx(1.0)
    assert np.isnan(_crossing_year(cumulative, 999.0))


def test_flat_cashflows_pay_back_when_arithmetic_says_they_should():
    realised = {"years_observed": 0.0, "total_savings": 0.0, "total_generation_kwh": 0.0}
    a = FinancialAssumptions(
        capex=100_000, tariff_escalation=0.0, degradation_rate=0.0,
        annual_om_cost=0.0, inverter_replacement_cost=0.0, system_life_years=25,
    )
    # 2,000 kWh at Rs 5 = Rs 10,000/yr against Rs 100,000 -> exactly 10 years.
    years = deterministic_payback(realised, base_annual_kwh=2_000, value_per_kwh=5.0, assumptions=a)
    assert years == pytest.approx(10.0, abs=0.01)


def test_discounting_can_only_lengthen_payback():
    realised = {"years_observed": 0.0, "total_savings": 0.0, "total_generation_kwh": 0.0}
    a = FinancialAssumptions(capex=100_000, discount_rate=0.08, tariff_escalation=0.0,
                             degradation_rate=0.0, annual_om_cost=0.0,
                             inverter_replacement_cost=0.0)
    plain = deterministic_payback(realised, 2_000, 5.0, a)
    discounted = deterministic_payback(realised, 2_000, 5.0, a, discounted=True)
    assert discounted > plain


def test_monte_carlo_is_reproducible_and_brackets_the_deterministic_case():
    realised = {"years_observed": 2.0, "total_savings": 12_000.0, "total_generation_kwh": 5_000.0}
    a = FinancialAssumptions(capex=170_000)
    mc_cfg = MonteCarloAssumptions(n_simulations=3_000, seed=7)

    first = monte_carlo_payback(realised, 2_400, 3.0, a, mc_cfg)
    second = monte_carlo_payback(realised, 2_400, 3.0, a, mc_cfg)
    assert first.p50 == second.p50, "a fixed seed must give a repeatable answer"
    assert first.p10 < first.p50 < first.p90

    point = deterministic_payback(realised, 2_400, 3.0, a)
    assert first.p10 <= point <= first.p90, "the point estimate should sit inside the band"


def test_realised_savings_are_not_resimulated():
    """Measured years are held fixed; only the future is uncertain."""
    realised = {"years_observed": 3.0, "total_savings": 30_000.0, "total_generation_kwh": 7_000.0}
    a = FinancialAssumptions(capex=30_000)  # already paid back inside the measured era
    res = monte_carlo_payback(realised, 2_400, 3.0, a, MonteCarloAssumptions(n_simulations=500, seed=1))
    assert np.nanstd(res.paybacks) == pytest.approx(0.0, abs=1e-9)
    assert res.prob_within_life == 1.0
