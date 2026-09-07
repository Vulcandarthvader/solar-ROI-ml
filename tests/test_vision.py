"""Tests for the chart-to-data extraction.

The accuracy test is the important one: the CSV carries metered generation for
30 of the months shown in the sample screenshots, so the pixel measurement can
be scored against ground truth rather than against itself.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from src.vision import (
    _label_similarity,
    _ransac_line,
    _snap_to_nice,
    extract_chart,
)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
SAMPLES = DATA / "samples"
MONTH_SHOTS = sorted(SAMPLES.glob("shinephone_month_*"))


def test_snap_to_nice_includes_four_and_three_mantissas():
    # A 240-max axis divided into six ticks gives a step of 40.  An earlier
    # version only knew 1/2/2.5/5, snapped 40 to 50, and read every 2023 bar
    # 25% too high.
    assert _snap_to_nice(40) == 40
    assert _snap_to_nice(300) == 300
    assert _snap_to_nice(51) == 50
    assert _snap_to_nice(0.9) == 1.0


def test_label_similarity_rewards_single_digit_substitution():
    assert _label_similarity("80.0", "80.0") == 1.0
    assert _label_similarity("80.0", "30.0") == pytest.approx(0.6)  # 8 read as 3
    assert _label_similarity("80.0", "120.0") == 0.0  # different length
    assert _label_similarity("80.0", "") == 0.0


def test_ransac_line_ignores_an_outlier():
    xs = np.array([1, 2, 3, 4, 5], dtype=float)
    ys = 10 * xs + 3
    ys[2] = 999  # a misread tick label
    a, b, inliers = _ransac_line(xs, ys, tol=2.0)
    assert b == pytest.approx(10.0, abs=1e-6)
    assert a == pytest.approx(3.0, abs=1e-6)
    assert inliers.sum() == 4


@pytest.mark.skipif(not MONTH_SHOTS, reason="sample screenshots not present")
@pytest.mark.parametrize("path", [str(p) for p in MONTH_SHOTS])
def test_month_charts_parse(path):
    r = extract_chart(path)
    assert r.kind == "month"
    assert r.year is not None and 2000 <= r.year <= 2100
    assert 1 <= len(r.values) <= 12
    assert all(1 <= p <= 12 for p in r.periods)
    assert len(set(r.periods)) == len(r.periods), "each bar must map to its own month"
    assert r.confidence > 0.6


@pytest.mark.skipif(not MONTH_SHOTS, reason="sample screenshots not present")
def test_extraction_matches_metered_values():
    """Pixel-measured generation must track the meter to within a few percent."""
    bills = pd.read_csv(DATA / "billdata.csv")
    truth = bills.groupby(["year", "month"])["solar_kwh"].max()

    errors = []
    for path in MONTH_SHOTS:
        r = extract_chart(str(path))
        for period, value in zip(r.periods, r.values):
            actual = truth.get((r.year, period))
            if actual and actual > 0:
                errors.append(abs(value - actual) / actual * 100)

    assert len(errors) >= 25, "expected the samples to overlap the metered history"
    assert np.mean(errors) < 3.0, f"mean absolute error {np.mean(errors):.2f}% is too high"
    assert np.max(errors) < 8.0, f"worst month off by {np.max(errors):.2f}%"


@pytest.mark.skipif(not (SAMPLES / "shinephone_year.jpg").exists(), reason="sample missing")
def test_year_chart_is_recognised_as_yearly():
    r = extract_chart(str(SAMPLES / "shinephone_year.jpg"))
    assert r.kind == "year"
    assert all(2000 <= p <= 2100 for p in r.periods)
    # Annual totals should be in the low thousands of kWh for a domestic array.
    assert all(500 < v < 10_000 for v in r.values)


def test_a_non_chart_image_is_rejected_cleanly():
    blank = np.full((400, 400, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        extract_chart(blank)
