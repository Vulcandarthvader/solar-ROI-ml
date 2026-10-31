"""Read monthly solar generation (kWh) off a ShinePhone app screenshot.

The ShinePhone app shows generation as a green bar chart and has no export
button. This file turns that picture back into numbers in 5 steps:

    1. Find the green pixels (the bars).
    2. Find the chart area and each bar's top edge.
    3. Y-axis: read the labels (0, 40, 80 ...) to learn "kWh per pixel".
    4. X-axis: read the month numbers (2, 4, 6 ...) to learn "which month is where".
    5. Bar height in pixels x kWh-per-pixel = kWh for that month.

OCR (Tesseract) sometimes misreads a digit, e.g. "80.0" as "30.0". So we never
trust one label alone: on the y-axis all labels VOTE for the step size, and
on the x-axis RANSAC ignores ticks that do not fit a straight line.
"""

import os
import re #regular expressions (used to check "does this text look like a number / a year?").
import cv2
import numpy as np
import pytesseract

# On Windows, Tesseract is often installed here but not added to PATH.
WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT

# Range of "bar green" in HSV colour space (Hue, Saturation, Value).
# HSV separates colour from brightness, so shadows and JPEG noise matter less.
GREEN_LOW = (40, 60, 120)
GREEN_HIGH = (90, 255, 255)
"""
Hue 40–90 ≈ green (0 = red, 60 ≈ green, 120 ≈ blue in OpenCV's scale).
Saturation ≥ 60 → ignores grey/white background.
Value ≥ 120 → ignores dark pixels
"""
# Chart libraries use "nice" axis steps: 1, 2, 2.5, 4, 5 ... x 10^k.
NICE_STEPS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0)


# ---------------------------------------------------------------- helpers ---

def group_runs(indices, max_gap=3):
    """Group sorted positions into (start, end) runs, e.g. [1,2,3,9,10] -> [(1,3),(9,10)]."""
    runs = []
    for i in indices:
        i = int(i)
        if runs and i - runs[-1][1] <= max_gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [tuple(r) for r in runs]
#Input: sorted positions (rows or columns that contain something). Output: groups of
#neighbouring positions.

def ocr(gray_crop, psm=7, words=False):
    """Read digits from a small crop of the image with Tesseract.

    The crop is enlarged 5x and turned black-and-white first, because
    Tesseract reads small phone text badly.
    """
    big = cv2.resize(gray_crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    big = cv2.threshold(big, 170, 255, cv2.THRESH_BINARY)[1]
    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789." #only digits and a dot are allowed. Tesseract can't return "O" for "0" or "l" for "1".
    if not words:
        return pytesseract.image_to_string(big, config=config).strip()
    # words=True: also return where each word is, in original pixel units
    data = pytesseract.image_to_data(big, config=config, output_type=pytesseract.Output.DICT)
    found = []
    for text, left, width in zip(data["text"], data["left"], data["width"]):
        if text.strip():
            x_centre = (left + width / 2 - 30) / 5
            found.append((x_centre, text.strip()))
    return found


def snap_to_nice(step):
    """Round a rough axis step to the nearest 'nice' number, e.g. 38.7 -> 40."""
    power = 10 ** np.floor(np.log10(step))
    candidates = [m * power * f for m in NICE_STEPS for f in (0.1, 1, 10)]
    return min(candidates, key=lambda c: abs(np.log(c / step)))


def ransac_line(xs, ys, tolerance):
    """Fit y = a + b*x while ignoring outliers.

    Try a line through every pair of points; keep the one that most points
    agree with (within `tolerance`); then refit on just those points.
    """
    best_inliers = None
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] == xs[j]:
                continue
            b = (ys[j] - ys[i]) / (xs[j] - xs[i])
            a = ys[i] - b * xs[i]
            inliers = np.abs(a + b * xs - ys) <= tolerance
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
    b, a = np.polyfit(xs[best_inliers], ys[best_inliers], 1)
    return a, b, best_inliers
"""
there are only ~6 points:
1. Take every pair of points; draw the line through them.
2. Count how many points lie within tolerance of that line (inliers). inliers is a True/False
array.
3. Keep the line with the most inliers.
4. Refit properly on just those inliers with np.polyfit(..., 1) (degree-1 polynomial = straight
line; returns [slope, intercept] , hence b, a = ).
if xs[i] == xs[j]: continue avoids dividing by zero (two labels with the same month number).
Cost: O(n²) pairs × O(n) check = O(n³) — fine for 6 points.
"""

# -------------------------------------------------------- the 5 main steps ---

def find_bars(image):
    """Steps 1-2: return the chart's (top, bottom) rows and each bar's (x_centre, top_y)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH) > 0

    # The chart is the band of rows with the most green (ignores green icons).
    rows_with_green = np.where(mask.sum(axis=1) > 0)[0]
    bands = group_runs(rows_with_green, max_gap=4)
    if not bands:
        raise ValueError("No green bars found - is this a ShinePhone chart screenshot?")
    top, bottom = max(bands, key=lambda b: mask[b[0]:b[1] + 1].sum())

    # Inside that band, columns full of green are bars.
    chart = mask[top:bottom + 1]
    columns = np.where(chart.sum(axis=0) > max(3, 0.02 * chart.shape[0]))[0]
    runs = group_runs(columns, max_gap=6)
    if not runs:
        raise ValueError("No bars detected inside the chart.")
    typical_width = np.median([end - start + 1 for start, end in runs])
#A column is "part of a bar" if it has more than 3 green pixels and more than 2% of the
band height (so stray noise pixels don't count).
    bars = []
    for start, end in runs:
        if end - start + 1 < 0.45 * typical_width:
            continue  # a thin sliver, not a real bar
        strip = chart[:, start:end + 1]
        filled_rows = np.where(strip.sum(axis=1) > 0.5 * (end - start + 1))[0]
        if len(filled_rows):
            bars.append(((start + end) / 2, top + filled_rows.min()))
    return (top, bottom), bars


def read_y_axis(gray, band):
    """Step 3: return (kwh_per_pixel, zero_line_y)."""
    top, bottom = band
    y0 = max(0, top - 60)
    region = gray[y0:bottom + 12, 0:160]
    # Each label is a horizontal run of dark text rows.
    dark_rows = np.where((region < 180).sum(axis=1) > 0)[0]
    labels = [(y0 + a, y0 + b) for a, b in group_runs(dark_rows, max_gap=2) if b - a >= 5]
    if len(labels) < 3:
        raise ValueError("Could not find the y-axis labels.")

    centres = np.array([(a + b) / 2 for a, b in labels])
    pixel_step = np.median(np.diff(centres))   # gridlines are evenly spaced
    zero_y = centres[-1]                       # the lowest label is 0

    # Read every label. Each one gives a guess of the axis step: value / position.
    # e.g. "120.0" on the 3rd gridline -> step 40.
    positions, texts, guesses = [], [], []
    for (a, b), centre in zip(labels, centres):
        position = round((zero_y - centre) / pixel_step)
        text = ocr(gray[max(0, a - 4):b + 5, 0:160])
        positions.append(position)
        texts.append(text)
        if position > 0 and re.fullmatch(r"\d+(\.\d+)?", text):
            guesses.append(float(text) / position)
    if not guesses:
        raise ValueError("Could not read any y-axis label.")
#step = value ÷ position
    # Voting: for each candidate step, predict what every label SHOULD say
    # (step 40 -> "0.0", "40.0", "80.0" ...) and count how many labels match.
    # OCR usually gets one digit wrong ("80.0" read as "30.0"), so a label
    # that is off by exactly one character still earns 0.6 of a vote.
    def votes(step):
        total = 0.0
        for position, text in zip(positions, texts):
            expected = f"{step * position:.1f}"
            if text == expected:
                total += 1.0
            elif len(text) == len(expected) and sum(x != y for x, y in zip(text, expected)) == 1:
                total += 0.6
        return total

    candidates = set(guesses) | {snap_to_nice(g) for g in guesses}
    median = np.median(guesses)
    step_kwh = max(candidates, key=lambda s: (round(votes(s), 6), -abs(np.log(s / median))))
    confidence = min(1.0, votes(step_kwh) / len(texts))
    return step_kwh / pixel_step, zero_y, confidence


def read_x_axis(gray, band):
    """Step 4: return (a, b) so that x_pixel = a + b * month."""
    bottom = band[1]
    words = ocr(gray[bottom + 8:bottom + 40, 60:], psm=6, words=True)
    ticks = [(x + 60, int(t)) for x, t in words if t.isdigit()]
    if len(ticks) < 3:
        raise ValueError("Could not read the x-axis month numbers.")
    if sum(1900 < m < 2200 for _, m in ticks) >= 2:
        raise ValueError("This looks like a Year chart; please upload Month charts.")

    xs = np.array([x for x, _ in ticks], float)
    months = np.array([m for _, m in ticks], float)
    spacing = np.median(np.diff(np.sort(xs)))
    a, b, inliers = ransac_line(months, xs, tolerance=max(6, 0.25 * spacing))
    return a, b, inliers.mean()


def read_year(gray, band):
    """Read the '< 2024 >' year shown above the chart."""
    strip = gray[max(0, band[0] - 220):max(0, band[0] - 120), 200:gray.shape[1] - 200]
    for _, text in ocr(strip, psm=6, words=True):
        if re.fullmatch(r"20\d\d", text):
            return int(text)
    return None


def extract_chart(image):
    """Run all steps on one screenshot (a file path or an image array).

    Returns a dict: {"year", "months", "kwh", "confidence", "warnings"}.
    """
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        raise ValueError("Image could not be read.")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    band, bars = find_bars(img)
    kwh_per_px, zero_y, y_conf = read_y_axis(gray, band)
    a, b, x_conf = read_x_axis(gray, band)
    year = read_year(gray, band)

    months, kwh = [], []
    for x_centre, top_y in bars:                       # Step 5
        months.append(int(round((x_centre - a) / b)))
        kwh.append(round(max(0.0, (zero_y - top_y) * kwh_per_px), 1))

    warnings = []
    if year is None:
        warnings.append("Could not read the year above the chart - please type it in.")
    if any(not 1 <= m <= 12 for m in months):
        warnings.append("Some bars fell outside months 1-12.")
    confidence = float((y_conf + x_conf) / 2)
    if confidence < 0.75:
        warnings.append("Low confidence - check the values before using them.")
    return {"year": year, "months": months, "kwh": kwh,
            "confidence": confidence, "warnings": warnings}


def extract_many(paths):
    """Read several screenshots into one table: year, month, solar_kwh."""
    rows, errors = [], []
    for path in paths:
        try:
            r = extract_chart(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        for m, v in zip(r["months"], r["kwh"]):
            if r["year"] and 1 <= m <= 12:
                rows.append({"year": r["year"], "month": m, "solar_kwh": v})
    return rows, errors
