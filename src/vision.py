"""
Chart-to-data extraction for ShinePhone (Growatt) app screenshots.

The ShinePhone mobile app renders monthly / yearly generation as a bar chart but
offers no CSV export on the free tier.  This module recovers the underlying
numbers straight from a screenshot:

    1. Segment the green bars in HSV space and locate the plot band.
    2. Detect y-axis gridline labels, OCR them, and calibrate kWh-per-pixel.
    3. Detect x-axis tick labels, OCR them, and calibrate pixel-to-period.
    4. Convert every bar's height into kWh.

OCR on a phone screenshot is noisy (a '5' is routinely read as a '9' at this
font size), so *neither* axis trusts OCR directly.  Both are calibrated with a
RANSAC-style consensus fit that exploits the strong structural priors of a
chart axis: gridlines are evenly spaced, the bottom gridline is zero, and axis
steps are "nice" numbers (1, 2, 2.5, 5 x 10^k).  A single-character edit
distance is used when scoring candidate steps, because OCR errors on digits are
substitutions, not insertions.  Every extraction returns a confidence score so
the UI can fall back to manual entry when the image is unusual.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

try:  # pragma: no cover - exercised implicitly, guarded for import-time safety
    import pytesseract
    from pytesseract import Output

    _TESSERACT_AVAILABLE = True
except Exception:  # pragma: no cover
    _TESSERACT_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# HSV window for the ShinePhone bar green (#7ae67a-ish).  Wide enough to survive
# JPEG chroma subsampling, narrow enough to reject the teal/blue chrome.
GREEN_LO = (40, 60, 120)
GREEN_HI = (90, 255, 255)

# Axis steps that charting libraries actually choose.  ECharts (which the app
# uses) divides a padded max by the tick count, so 3x and 4x mantissas show up
# alongside the textbook 1/2/2.5/5.
NICE_MANTISSAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0)

DIGITS = "0123456789"


@dataclass
class AxisCalibration:
    """Linear map between pixel coordinates and data values."""

    scale: float  # data units per pixel
    origin_px: float  # pixel coordinate of value 0
    confidence: float  # 0-1
    detail: str = ""


@dataclass
class ChartExtraction:
    """Everything recovered from one screenshot."""

    kind: Literal["month", "year", "unknown"]
    year: int | None
    periods: list[int] = field(default_factory=list)  # month numbers or years
    values: list[float] = field(default_factory=list)  # kWh
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    y_axis: AxisCalibration | None = None
    debug: dict = field(default_factory=dict)

    def as_records(self) -> list[dict]:
        """Flatten into rows ready for a DataFrame."""
        rows = []
        for p, v in zip(self.periods, self.values):
            if self.kind == "month":
                rows.append({"year": self.year, "month": p, "solar_kwh": v})
            elif self.kind == "year":
                rows.append({"year": p, "month": None, "solar_kwh": v})
        return rows


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #


def _ocr_available() -> bool:
    return _TESSERACT_AVAILABLE


def _green_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    # Close 1-2px JPEG speckle inside bars without merging neighbours.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _contiguous_runs(indices: np.ndarray, max_gap: int = 3) -> list[tuple[int, int]]:
    """Group sorted indices into (start, end) runs, bridging gaps <= max_gap."""
    if len(indices) == 0:
        return []
    runs, start, prev = [], int(indices[0]), int(indices[0])
    for i in indices[1:]:
        i = int(i)
        if i - prev > max_gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


def _ocr_line(gray_crop: np.ndarray, whitelist: str = DIGITS + ".") -> str:
    """OCR a single short line of small UI text."""
    if not _ocr_available():
        return ""
    up = cv2.resize(gray_crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    up = cv2.copyMakeBorder(up, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    up = cv2.threshold(up, 170, 255, cv2.THRESH_BINARY)[1]
    cfg = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(up, config=cfg).strip()
    except Exception:
        return ""


def _ocr_words(gray_crop: np.ndarray, x_off: int, whitelist: str = DIGITS) -> list[tuple[float, str]]:
    """OCR a strip, returning (x_centre_in_original_pixels, text) per word."""
    if not _ocr_available():
        return []
    f = 5
    up = cv2.resize(gray_crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    pad = 30
    up = cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    up = cv2.threshold(up, 170, 255, cv2.THRESH_BINARY)[1]
    cfg = f"--psm 6 -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(up, config=cfg, output_type=Output.DICT)
    except Exception:
        return []
    out = []
    for i, text in enumerate(data["text"]):
        t = text.strip()
        if not t:
            continue
        cx = x_off + (data["left"][i] + data["width"][i] / 2 - pad) / f
        out.append((cx, t))
    return out


def _snap_to_nice(value: float) -> float:
    """Snap a rough axis step to the nearest 'nice' tick (1/2/2.5/5 x 10^k)."""
    if value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    best, best_err = None, np.inf
    for e in (exp - 1, exp, exp + 1):
        for m in NICE_MANTISSAS:
            cand = m * (10**e)
            err = abs(np.log(cand / value))
            if err < best_err:
                best, best_err = cand, err
    return float(best)


def _fmt_axis(value: float) -> str:
    """Reproduce the app's axis label formatting (always one decimal place)."""
    return f"{value:.1f}"


def _label_similarity(pred: str, read: str) -> float:
    """Score an OCR'd axis label against a predicted one.

    Exact match scores 1.0.  Same-length strings differing in a single
    character score 0.6 -- this is the signature of a digit substitution, which
    is by far the dominant OCR failure mode at this font size.
    """
    if not read:
        return 0.0
    if pred == read:
        return 1.0
    if len(pred) == len(read):
        diffs = sum(1 for a, b in zip(pred, read) if a != b)
        if diffs == 1:
            return 0.6
        if diffs == 2 and len(pred) >= 5:
            return 0.25
    return 0.0


def _ransac_line(xs: np.ndarray, ys: np.ndarray, tol: float) -> tuple[float, float, np.ndarray]:
    """Fit y = a + b*x keeping the largest consensus set. Returns (a, b, inliers)."""
    n = len(xs)
    best = (0.0, 0.0, np.zeros(n, dtype=bool))
    best_count = -1
    for i in range(n):
        for j in range(i + 1, n):
            if xs[j] == xs[i]:
                continue
            b = (ys[j] - ys[i]) / (xs[j] - xs[i])
            a = ys[i] - b * xs[i]
            resid = np.abs(a + b * xs - ys)
            inl = resid <= tol
            if inl.sum() > best_count:
                best_count, best = inl.sum(), (a, b, inl)
    a, b, inl = best
    if inl.sum() >= 2:  # least-squares refit on the consensus set
        b, a = np.polyfit(xs[inl], ys[inl], 1)
    return a, b, inl


# --------------------------------------------------------------------------- #
# Plot geometry
# --------------------------------------------------------------------------- #


def find_plot_band(mask: np.ndarray) -> tuple[int, int]:
    """Locate the vertical band containing the bars.

    A ShinePhone screenshot also contains green flow arrows and green status
    text.  The plot is the run of rows carrying by far the most green mass.
    """
    row_counts = (mask > 0).sum(axis=1)
    runs = _contiguous_runs(np.where(row_counts > 0)[0], max_gap=4)
    if not runs:
        raise ValueError("No green bars found - is this a ShinePhone chart screenshot?")
    best = max(runs, key=lambda r: row_counts[r[0] : r[1] + 1].sum())
    return best


def detect_bars(mask: np.ndarray, band: tuple[int, int]) -> list[tuple[float, float]]:
    """Return (x_centre, top_y) for every bar in the plot band."""
    top, bottom = band
    sub = mask[top : bottom + 1]
    height = bottom - top + 1
    col_counts = (sub > 0).sum(axis=0)

    # A real bar occupies a meaningful slice of the plot height at its column.
    live = np.where(col_counts > max(3, 0.02 * height))[0]
    # Bars are separated by wide gaps; bridge anti-aliasing/gridline breaks only.
    runs = _contiguous_runs(live, max_gap=6)
    if not runs:
        return []

    widths = np.array([r[1] - r[0] + 1 for r in runs])
    median_w = float(np.median(widths))
    bars = []
    for x0, x1 in runs:
        if (x1 - x0 + 1) < 0.45 * median_w:
            continue  # sliver from a gridline or JPEG artefact
        strip = sub[:, x0 : x1 + 1]
        rows = np.where((strip > 0).sum(axis=1) > 0.5 * (x1 - x0 + 1))[0]
        if len(rows) == 0:
            continue
        bars.append(((x0 + x1) / 2.0, top + float(rows.min())))
    return bars


# --------------------------------------------------------------------------- #
# Axis calibration
# --------------------------------------------------------------------------- #


def calibrate_y_axis(gray: np.ndarray, band: tuple[int, int], label_width: int = 160) -> AxisCalibration:
    """Recover kWh-per-pixel from the y-axis gridline labels."""
    top, bottom = band
    search_top = max(0, top - 60)
    search_bottom = min(gray.shape[0], bottom + 12)
    region = gray[search_top:search_bottom, 0:label_width]
    dark = (region < 180).astype(np.uint8)
    runs = _contiguous_runs(np.where(dark.sum(axis=1) > 0)[0], max_gap=2)
    runs = [(search_top + a, search_top + b) for a, b in runs if (b - a) >= 5]
    if len(runs) < 3:
        raise ValueError("Could not locate y-axis labels.")

    centres = np.array([(a + b) / 2 for a, b in runs])
    # Gridlines are evenly spaced; drop anything that breaks the rhythm.
    spacings = np.diff(centres)
    step_px = float(np.median(spacings))
    keep = [0]
    for i in range(1, len(centres)):
        if abs((centres[i] - centres[keep[-1]]) / step_px - round((centres[i] - centres[keep[-1]]) / step_px)) < 0.2:
            keep.append(i)
    centres = centres[keep]
    runs = [runs[i] for i in keep]
    step_px = float(np.median(np.diff(centres)))

    # The lowest label is the zero line; index upward from there.
    zero_y = centres[-1]
    texts = [_ocr_line(gray[max(0, a - 4) : b + 5, 0:label_width]) for a, b in runs]
    idx = [int(round((zero_y - c) / step_px)) for c in centres]

    # Candidate axis steps: every OCR reading implies one.  Keep both the raw
    # implied step and its "nice" snap -- a correctly-read label gives the exact
    # step, while a misread one is rescued by snapping.  Whichever candidate
    # explains the most labels wins.
    candidates: set[float] = set()
    raw_estimates: list[float] = []
    for k, t in zip(idx, texts):
        if k <= 0:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", t or ""):
            raw = float(t) / k
            if raw > 0:
                raw_estimates.append(raw)
                candidates.add(round(raw, 6))
                candidates.add(_snap_to_nice(raw))
    if not candidates:
        raise ValueError("No y-axis label could be read.")

    def score(step: float) -> float:
        return sum(_label_similarity(_fmt_axis(step * k), t) for k, t in zip(idx, texts) if k >= 0)

    # Ties break toward the step closest to the median of the raw estimates.
    median_raw = float(np.median(raw_estimates))
    best_step = max(candidates, key=lambda s: (round(score(s), 6), -abs(np.log(s / median_raw))))
    n_labels = sum(1 for k in idx if k >= 0)
    conf = score(best_step) / max(1, n_labels)

    return AxisCalibration(
        scale=best_step / step_px,
        origin_px=float(zero_y),
        confidence=float(min(1.0, conf)),
        detail=f"step={best_step:g} kWh over {step_px:.1f}px; labels read: {[t for t in texts]}",
    )


def calibrate_x_axis(gray: np.ndarray, band: tuple[int, int]) -> tuple[AxisCalibration, str]:
    """Recover the pixel-to-period map and whether this is a month or year chart."""
    _, bottom = band
    strip = gray[bottom + 8 : bottom + 40, 60 : gray.shape[1]]
    words = _ocr_words(strip, x_off=60)
    parsed = [(x, int(t)) for x, t in words if t.isdigit()]
    if len(parsed) < 3:
        raise ValueError("Could not read the x-axis tick labels.")

    kind = "year" if sum(1 for _, v in parsed if 1900 < v < 2200) >= max(2, len(parsed) // 2) else "month"
    xs = np.array([p[0] for p in parsed], dtype=float)
    vals = np.array([p[1] for p in parsed], dtype=float)

    # Fit x = a + b*period, keeping the largest consensus set (OCR misreads a
    # digit here and there; the surviving ticks are perfectly collinear).
    spacing_guess = np.median(np.diff(np.sort(xs)))
    a, b, inliers = _ransac_line(vals, xs, tol=max(6.0, 0.25 * spacing_guess))
    if inliers.sum() < 3 or b <= 0:
        raise ValueError("x-axis ticks are not consistent enough to calibrate.")

    conf = inliers.sum() / len(parsed)
    return (
        AxisCalibration(
            scale=1.0 / b,
            origin_px=a,
            confidence=float(conf),
            detail=f"{int(inliers.sum())}/{len(parsed)} ticks consistent; {b:.1f}px per period",
        ),
        kind,
    )


def read_header_year(gray: np.ndarray, band: tuple[int, int]) -> int | None:
    """Read the '< 2024 >' year selector that sits above the plot."""
    top = band[0]
    strip = gray[max(0, top - 220) : max(0, top - 120), 200 : gray.shape[1] - 200]
    for _, t in _ocr_words(strip, x_off=200):
        if re.fullmatch(r"(19|20)\d{2}", t):
            return int(t)
    return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def extract_chart(image: np.ndarray | str) -> ChartExtraction:
    """Extract monthly or yearly kWh from one ShinePhone screenshot."""
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        raise ValueError("Image could not be read.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _green_mask(img)
    band = find_plot_band(mask)
    warnings: list[str] = []

    bars = detect_bars(mask, band)
    if not bars:
        raise ValueError("No bars detected inside the plot area.")

    y_cal = calibrate_y_axis(gray, band)
    x_cal, kind = calibrate_x_axis(gray, band)

    periods, values = [], []
    for cx, top_y in bars:
        period = (cx - x_cal.origin_px) * x_cal.scale
        kwh = (y_cal.origin_px - top_y) * y_cal.scale
        periods.append(int(round(period)))
        values.append(round(max(0.0, float(kwh)), 1))

    year = None
    if kind == "month":
        year = read_header_year(gray, band)
        if year is None:
            warnings.append("Could not read the year above the chart - please confirm it.")
        bad = [p for p in periods if not 1 <= p <= 12]
        if bad:
            warnings.append(f"Month indices out of range were clipped: {bad}")
    if len(set(periods)) != len(periods):
        warnings.append("Two bars mapped to the same period; x-axis calibration may be off.")

    confidence = float(np.mean([y_cal.confidence, x_cal.confidence]))
    if confidence < 0.75:
        warnings.append("Low extraction confidence - review the values before continuing.")

    return ChartExtraction(
        kind=kind,
        year=year,
        periods=periods,
        values=values,
        confidence=confidence,
        warnings=warnings,
        y_axis=y_cal,
        debug={
            "plot_band": band,
            "n_bars": len(bars),
            "y_detail": y_cal.detail,
            "x_detail": x_cal.detail,
        },
    )


def extract_many(images: list) -> tuple[list[ChartExtraction], list[str]]:
    """Extract a batch, collecting per-file errors instead of raising."""
    results, errors = [], []
    for im in images:
        name = im if isinstance(im, str) else "uploaded image"
        try:
            results.append(extract_chart(im))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            errors.append(f"{name}: {exc}")
    return results, errors
