"""End-to-end tests on the real dataset."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from src.config import FinancialAssumptions, MonteCarloAssumptions
from src.pipeline import extractions_to_frame, run_analysis
from src.vision import extract_chart

ROOT = pathlib.Path(__file__).resolve().parents[1]
BILLS = ROOT / "data" / "billdata.csv"
SAMPLES = sorted((ROOT / "data" / "samples").glob("shinephone_*"))

pytestmark = pytest.mark.skipif(not BILLS.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def result():
    extractions = [extract_chart(str(p)) for p in SAMPLES]
    return run_analysis(
        BILLS,
        extractions_to_frame(extractions),
        FinancialAssumptions(capex=170_000),
        MonteCarloAssumptions(n_simulations=2_000, seed=42),
    )


def test_commissioning_month_is_detected(result):
    assert str(result.commissioning) == "2023-06"


def test_screenshots_agree_with_the_meter(result):
    stats = result.extraction_stats
    assert stats["overlap_n"] >= 25
    assert stats["overlap_mape"] < 3.0
    assert stats["months_added"] > 0, "the charts should extend past the last bill"


def test_headline_numbers_are_finite_and_plausible(result):
    h = result.headline()
    assert 5 < result.monte_carlo.p50 < 40, "payback outside any believable range"
    assert result.monte_carlo.p10 < result.monte_carlo.p50 < result.monte_carlo.p90
    assert 0 < result.value_per_kwh < 20, "value of a generated unit is implausible"
    assert np.isfinite(h["NPV @ 7% (INR)"])
    assert result.realised["total_savings"] > 0


def test_both_counterfactuals_land_in_the_same_neighbourhood(result):
    hist = result.history
    a = hist["savings_tariff"].mean()
    b = hist["savings_synthetic"].mean()
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(a - b) / max(a, b) < 0.5, "independent counterfactuals disagree wildly"


def test_savings_are_positive_on_the_typical_month(result):
    hist = result.history
    # The commissioning month is partial - the array was energised part-way
    # through it - so it is excluded here for the same reason the anomaly
    # detector labels it "ramp-up".
    clean = hist[~hist["is_billing_anomaly"].fillna(False)].iloc[1:]
    assert (clean["savings"] > 0).all(), "every clean full month should show a saving"
    assert clean["savings"].median() > 200
    # The as-billed series may dip where an arrears bill landed; the robust
    # headline total must not inherit that dip.
    assert result.realised["total_savings"] >= result.realised["total_savings_as_billed"]


def test_cashflow_schedule_is_coherent(result):
    s = result.payback.schedule
    # Cumulative savings rise every year except the one the inverter is
    # replaced in, where a lump cost legitimately pushes the balance down.
    dips = s.index[s["cumulative"].diff() < 0].tolist()
    assert len(dips) <= 1, f"unexpected dips in cumulative savings at {dips}"
    if dips:
        assert s.loc[dips[0], "net_cashflow"] < 0
    assert s["cumulative"].iloc[-1] > s["cumulative"].iloc[0]
    assert (s["cumulative_discounted"] <= s["cumulative"] + 1e-6).all()
    assert set(s["source"]) <= {"realised", "projected"}
    assert (s["source"] == "realised").sum() >= 1


def test_projection_covers_the_remaining_system_life(result):
    total_years = len(result.payback.schedule)
    assert total_years == result.assumptions.system_life_years


def test_leaderboards_name_a_selected_model(result):
    for name, board in result.leaderboards.items():
        assert len(board) >= 3, f"{name} has too few candidates to be a comparison"
        if "selected" in board.columns:
            assert board["selected"].sum() == 1


def test_analysis_still_runs_without_any_screenshots():
    r = run_analysis(BILLS, None, FinancialAssumptions(capex=170_000),
                     MonteCarloAssumptions(n_simulations=500, seed=1))
    assert np.isfinite(r.monte_carlo.p50)
    assert r.extraction_stats["overlap_n"] == 0
