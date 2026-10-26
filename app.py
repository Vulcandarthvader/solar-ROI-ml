"""SolarROI - Streamlit web app.

Run with:   streamlit run app.py

Page layout:
    Sidebar  -> the money assumptions (cost, tariff rise, ...)
    Step 1   -> screenshots + bill CSV (or the bundled sample data)
    Step 2   -> check / edit the kWh read from the screenshots
    Step 3   -> results in 5 tabs
"""

import glob
import io

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import Assumptions
from src.pipeline import run_analysis
from src.vision import extract_chart

st.set_page_config(page_title="SolarROI", page_icon="☀️", layout="wide")


def rupees(x):
    return f"₹{x:,.0f}"


@st.cache_data(show_spinner=False)
def read_screenshot(image_bytes):
    """Cached, so the same screenshot is only read once."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return extract_chart(image)


# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.header("Assumptions")
    capex = st.number_input("System cost (₹)", 10_000, 5_000_000, 170_000, step=5_000)
    escalation = st.slider("Tariff rise per year (%)", 0.0, 12.0, 5.5, 0.5) / 100
    discount = st.slider("Discount rate for NPV (%)", 0.0, 15.0, 7.0, 0.5) / 100
    degradation = st.slider("Panel degradation per year (%)", 0.0, 2.0, 0.6, 0.1) / 100
    om = st.number_input("Upkeep per year (₹)", 0, 50_000, 2_000, step=500)
    inverter = st.number_input("Inverter replacement (₹)", 0, 200_000, 25_000, step=5_000)
    life = st.slider("System life (years)", 10, 30, 25)

assumptions = Assumptions(
    capex=capex, tariff_escalation=escalation, discount_rate=discount,
    degradation_rate=degradation, annual_om_cost=om, inverter_cost=inverter,
    system_life_years=life,
)

# ------------------------------------------------------------------- inputs --
st.title("☀️ SolarROI")
st.write("Reads your ShinePhone screenshots and electricity bills, estimates the bill "
         "you would have paid without panels, and tells you when the panels pay for themselves.")

use_sample = st.toggle("Use the sample data", value=True)
if use_sample:
    bill_bytes = open("data/billdata.csv", "rb").read()
    images = [(p, open(p, "rb").read()) for p in sorted(glob.glob("data/samples/*"))]
else:
    left, right = st.columns(2)
    shots = left.file_uploader("ShinePhone Month-chart screenshots",
                               type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    bill_file = right.file_uploader("Bill CSV (year, month, grid_kwh, bill_amount, solar_kwh)",
                                    type=["csv"])
    if bill_file is None:
        st.info("Upload your bill CSV to continue.")
        st.stop()
    bill_bytes = bill_file.getvalue()
    images = [(f.name, f.getvalue()) for f in (shots or [])]

# Read the screenshots and let the user fix any value.
rows = []
for name, data in images:
    try:
        r = read_screenshot(data)
    except Exception as exc:
        st.warning(f"{name}: {exc}")
        continue
    label = "✅" if r["confidence"] >= 0.85 else "⚠️"
    st.caption(f"{label} {name}: year {r['year']}, {len(r['kwh'])} bars, "
               f"confidence {r['confidence']:.0%} {' '.join(r['warnings'])}")
    rows += [{"year": r["year"], "month": m, "solar_kwh": v}
             for m, v in zip(r["months"], r["kwh"]) if r["year"] and 1 <= m <= 12]

if rows:
    st.subheader("Generation read from the screenshots (kWh) - edit if anything is wrong")
    rows = st.data_editor(pd.DataFrame(rows), num_rows="dynamic", height=220).to_dict("records")

# ------------------------------------------------------------------ results --
try:
    R = run_analysis(io.BytesIO(bill_bytes), rows, assumptions)
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

past = R["past"]
tab_result, tab_savings, tab_forecast, tab_models, tab_data = st.tabs(
    ["Result", "Savings so far", "25-year forecast", "Models", "Data quality"])

with tab_result:
    c = st.columns(4)
    payback = R["payback_years"]
    c[0].metric("Payback", f"{payback:.1f} years" if np.isfinite(payback) else f"over {life} years")
    c[1].metric("Saved so far", rupees(past["total_saving"]), f"{past['capex_recovered_pct']:.1f}% of cost")
    c[2].metric("NPV at discount rate", rupees(R["npv"]))
    c[3].metric(f"{life}-year ROI", f"{R['roi_pct']:.0f}%")

    c = st.columns(4)
    c[0].metric("Average saving per month", rupees(past["avg_monthly_saving"]))
    c[1].metric("Value of 1 solar unit today", f"₹{past['value_per_kwh']:.2f}")
    c[2].metric("Grid units per month", f"{R['post_solar_avg_grid_kwh']:.0f} kWh",
                f"{R['post_solar_avg_grid_kwh'] - R['pre_solar_avg_grid_kwh']:.0f} vs before solar",
                delta_color="inverse")
    c[3].metric("CO₂ avoided so far", f"{past['co2_avoided_kg'] / 1000:.1f} t")

    st.subheader("Scenarios")
    table = R["scenarios"].copy()
    table["payback (years)"] = table["payback (years)"].map(
        lambda v: f"{v:.1f}" if np.isfinite(v) else f"over {life}")
    st.table(table)

    cash = R["cash_table"]
    fig = go.Figure()
    fig.add_scatter(x=[0, past["years"]], y=[0, past["total_saving"]], name="Measured", mode="lines+markers")
    fig.add_scatter(x=[past["years"]] + cash["year_of_life"].tolist(),
                    y=[past["total_saving"]] + cash["cumulative"].tolist(), name="Forecast",
                    line=dict(dash="dash"))
    fig.add_hline(y=capex, line_dash="dot", annotation_text="System cost")
    fig.update_layout(title="Cumulative savings vs system cost",
                      xaxis_title="Years since installation", yaxis_title="₹")
    st.plotly_chart(fig, width="stretch")

with tab_savings:
    s = R["savings"]
    x = s["period"].astype(str)
    plot = s[~s["is_arrears"]]          # leave out the arrears bill so the chart stays readable
    px_ = plot["period"].astype(str)
    fig = go.Figure()
    fig.add_bar(x=px_, y=plot["bill_amount"], name="Real bill")
    fig.add_scatter(x=px_, y=plot["no_solar_bill_A"], name="Without solar - Model A")
    if "no_solar_bill_B" in plot:
        fig.add_scatter(x=px_, y=plot["no_solar_bill_B"], name="Without solar - Model B")
    fig.update_layout(title="Real bill vs estimated bill without panels", yaxis_title="₹ per month")
    st.plotly_chart(fig, width="stretch")
    if "agreement_pct" in past:
        st.write(f"Model A says ₹{past['saving_A_avg']:.0f}/month saved, Model B says "
                 f"₹{past['saving_B_avg']:.0f}/month: they agree to {past['agreement_pct']:.0f}%.")
    st.dataframe(s[["period", "grid_kwh", "solar_kwh", "bill_amount", "saving", "cumulative_saving"]]
                 .assign(period=x).round(0), width="stretch")

with tab_forecast:
    by_month = R["kwh_by_month"]
    fig = go.Figure(go.Bar(x=by_month.index, y=by_month.values))
    fig.update_layout(title=f"Typical year: {R['annual_kwh']:,.0f} kWh", xaxis_title="Month",
                      yaxis_title="Average kWh")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(R["cash_table"].round(1), width="stretch")

with tab_models:
    st.write("Each model is scored with TimeSeriesSplit (train on older months, test on newer). "
             "Lowest MAE wins.")
    st.subheader(f"Model A - tariff model (chosen: {R['model_a'].name})")
    st.dataframe(R["model_a"].leaderboard.round(2), width="stretch")
    if R["model_b"] is not None:
        st.subheader(f"Model B - pre-solar trend (chosen: {R['model_b'].name})")
        st.dataframe(R["model_b"].leaderboard.round(2), width="stretch")

with tab_data:
    rep, shot = R["report"], R["screenshot_stats"]
    st.write(f"**Bills:** {rep['rows_read']} rows → {rep['months_after_cleaning']} months "
             f"({rep['date_range']}). Panels installed: **{R['install_month']}**.")
    st.write(f"Duplicate months merged: {rep['duplicate_months']}")
    st.write(f"Missing months filled: {rep['imputed_months']}")
    st.write(f"Repeated readings flagged: {rep['repeat_readings']}")
    st.write(f"Credit (negative) bills flagged: {rep['credit_bills']}")
    st.write(f"Arrears bills flagged: {rep['arrears_bills']}")
    if shot["overlap_months"]:
        st.write(f"**Screenshots vs meter:** {shot['overlap_months']} months compared, average error "
                 f"{shot['screenshot_error_pct']:.2f}%. {shot['months_added']} months added after the last bill.")
