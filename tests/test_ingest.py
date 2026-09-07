"""Tests for bill cleaning and the generation merge."""

from __future__ import annotations

import io
import pathlib

import pandas as pd
import pytest

from src.ingest import clean_bills, load_bills, merge_generation, modelling_frame

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def _csv(rows: str) -> io.StringIO:
    return io.StringIO("year,month,grid_kwh,bill_amount,solar_kwh\n" + rows)


def test_missing_required_column_is_a_clear_error():
    with pytest.raises(ValueError, match="missing required column"):
        load_bills(io.StringIO("year,month\n2020,1\n"))


def test_duplicate_months_are_summed_not_dropped():
    df = load_bills(_csv("2023,7,559,2265,122.2\n2023,7,57,2469,122.2\n2023,8,197,698,231.6\n"))
    clean, report = clean_bills(df)
    july = clean[clean["month"] == 7].iloc[0]
    assert july["grid_kwh"] == 616  # 559 + 57
    assert july["bill_amount"] == 4734  # 2265 + 2469
    assert july["solar_kwh"] == 122.2  # a monthly total, never added twice
    assert report.duplicate_months == ["2023-07"]


def test_gaps_are_filled_and_flagged():
    df = load_bills(_csv("2021,1,200,500,0\n2021,3,220,560,0\n2021,4,210,540,0\n"))
    clean, report = clean_bills(df)
    assert len(clean) == 4  # Jan, Feb (imputed), Mar, Apr
    assert report.imputed_months == ["2021-02"]
    assert bool(clean.loc[clean["month"] == 2, "is_imputed"].iloc[0])
    assert clean["grid_kwh"].notna().all()


def test_imputation_never_invents_solar_generation():
    """Regression: a seasonal fill once gave a pre-solar month solar output.

    That moved the detected commissioning date three years early and corrupted
    every savings figure downstream.
    """
    df = load_bills(
        _csv(
            "2020,4,300,700,0\n"      # gap at 2020-05
            "2020,6,310,720,0\n"
            "2023,5,260,900,0\n"
            "2023,6,200,800,150\n"    # the array actually starts here
        )
    )
    clean, _ = clean_bills(df)
    pre = clean[clean["period"] < pd.Period("2023-06", freq="M")]
    assert (pre["solar_kwh"] == 0).all()
    assert clean[clean["has_solar"]]["period"].min() == pd.Period("2023-06", freq="M")


def test_repeated_and_negative_bills_are_flagged():
    df = load_bills(_csv("2019,10,200,500,0\n2019,11,200,500,0\n2019,12,180,-300,0\n"))
    clean, report = clean_bills(df)
    assert report.repeated_readings == ["2019-11"]
    assert report.credit_adjustments == ["2019-12"]
    # Both are excluded from anything the models are fitted on.
    model_rows = modelling_frame(clean)
    assert set(model_rows["month"]) == {10}


def test_total_consumption_reconstructs_gross_demand():
    df = load_bills(_csv("2024,1,50,300,200\n"))
    clean, _ = clean_bills(df)
    row = clean.iloc[0]
    assert row["total_consumption_kwh"] == 250
    assert row["self_sufficiency"] == pytest.approx(0.8)


def test_merge_extends_history_and_scores_the_overlap():
    df = load_bills(_csv("2024,1,50,300,200\n2024,2,40,280,210\n"))
    clean, _ = clean_bills(df)
    extracted = pd.DataFrame(
        [
            {"year": 2024, "month": 1, "solar_kwh": 202.0},  # overlaps the meter
            {"year": 2024, "month": 3, "solar_kwh": 240.0},  # beyond the last bill
        ]
    )
    merged, stats = merge_generation(clean, extracted)
    assert stats["overlap_n"] == 1
    assert stats["overlap_mape"] == pytest.approx(1.0, abs=0.01)
    assert len(merged) == 3
    # The metered value wins where both sources exist.
    assert merged.loc[merged["month"] == 1, "solar_kwh"].iloc[0] == 200
    assert merged.loc[merged["month"] == 3, "solar_kwh"].iloc[0] == 240


@pytest.mark.skipif(not (DATA / "billdata.csv").exists(), reason="sample data missing")
def test_real_dataset_cleans_without_losing_months():
    clean, report = clean_bills(load_bills(DATA / "billdata.csv"))
    assert clean["period"].is_monotonic_increasing
    assert clean["period"].diff().dropna().apply(lambda d: d.n).eq(1).all()
    assert report.solar_start == "2023-06"
    assert len(modelling_frame(clean)) > 50
