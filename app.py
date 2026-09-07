"""SolarROI - Streamlit front end.

Upload the ShinePhone screenshots and the utility bill CSV; get a defensible
answer to "when do these panels pay for themselves", with the model evidence
attached.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import FinancialAssumptions, MonteCarloAssumptions
from src.explain import compute_shap, explain_instance, narrate
from src.features import build_tariff_frame
from src.ingest import modelling_frame
from src.pipeline import extractions_to_frame, run_analysis
from src.theme import apply_layout, palette
from src.vision import extract_chart

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SolarROI", page_icon="☀", layout="wide", initial_sidebar_state="expanded")

DEMO_BILLS = "data/billdata.csv"
DEMO_SHOTS = [
    "data/samples/shinephone_month_2023.png",
    "data/samples/shinephone_month_2024.jpg",
    "data/samples/shinephone_month_2025.jpg",
    "data/samples/shinephone_month_2026.jpg",
    "data/samples/shinephone_year.jpg",
]


def theme_mode() -> str:
    try:
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:
        return "light"


MODE = theme_mode()
P = palette(MODE)


st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 2.2rem; max-width: 1280px;}}
      .hero {{font-size: 2.6rem; font-weight: 650; color: {P['text']}; letter-spacing: -0.02em;}}
      .sub {{color: {P['text_secondary']}; font-size: 0.95rem;}}
      .card {{background:{P['surface']}; border:1px solid {P['grid']}; border-radius:12px;
              padding:16px 18px; height:100%;}}
      .card .k {{color:{P['muted']}; font-size:0.78rem; text-transform:uppercase;
                 letter-spacing:0.06em;}}
      .card .v {{color:{P['text']}; font-size:1.75rem; font-weight:650; line-height:1.25;}}
      .card .n {{color:{P['text_secondary']}; font-size:0.82rem;}}
      .step {{border-left:3px solid {P['series'][0]}; padding-left:12px; margin: 6px 0 14px;}}
      div[data-testid="stMetricValue"] {{font-size:1.6rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def stat(label: str, value: str, note: str = "") -> str:
    return f'<div class="card"><div class="k">{label}</div><div class="v">{value}</div><div class="n">{note}</div></div>'


def inr(x: float, decimals: int = 0) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"₹{x:,.{decimals}f}"


# --------------------------------------------------------------------------- #
# Cached workers
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def _extract_cached(name: str, payload: bytes):
    import cv2

    arr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    res = extract_chart(arr)
    return {
        "name": name,
        "kind": res.kind,
        "year": res.year,
        "periods": res.periods,
        "values": res.values,
        "confidence": res.confidence,
        "warnings": res.warnings,
        "detail": res.debug,
    }


@st.cache_resource(show_spinner=False)
def _analyse_cached(bill_bytes: bytes, gen_records: str, assumptions: dict, mc: dict):
    gen = pd.DataFrame(json.loads(gen_records)) if gen_records else None
    return run_analysis(
        io.BytesIO(bill_bytes),
        gen if gen is not None and len(gen) else None,
        FinancialAssumptions(**assumptions),
        MonteCarloAssumptions(**mc),
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### Investment")
    capex = st.number_input("System cost (₹)", 10_000, 5_000_000, 170_000, step=5_000,
                            help="Everything you paid: panels, inverter, mounting, labour, wiring.")
    st.markdown("### Economics")
    escalation = st.slider("Tariff escalation (%/yr)", 0.0, 12.0, 5.5, 0.5,
                           help="How fast you expect grid electricity to get more expensive.") / 100
    discount = st.slider("Discount rate (%/yr)", 0.0, 15.0, 7.0, 0.5,
                         help="What the same money would have earned elsewhere. Used for NPV "
                              "and discounted payback, not for cash payback.") / 100
    om = st.number_input("Annual upkeep (₹/yr)", 0, 50_000, 2_000, step=500,
                         help="Cleaning, inspection, small repairs.")
    st.markdown("### Hardware")
    degradation = st.slider("Panel degradation (%/yr)", 0.0, 2.0, 0.6, 0.1) / 100
    inv_cost = st.number_input("Inverter replacement (₹)", 0, 200_000, 25_000, step=5_000)
    inv_year = st.slider("Replace inverter in year", 5, 25, 12)
    life = st.slider("System life (years)", 10, 30, 25)

    with st.expander("Simulation settings"):
        n_sims = st.select_slider("Simulations", [1_000, 5_000, 10_000, 25_000], value=10_000)
        gen_sigma = st.slider("Year-to-year output variability (%)", 0.0, 25.0, 8.0, 1.0) / 100
        esc_sigma = st.slider("Uncertainty on tariff escalation (pp)", 0.0, 5.0, 2.0, 0.5) / 100

    ASSUMPTIONS = dict(
        capex=float(capex), discount_rate=discount, tariff_escalation=escalation,
        degradation_rate=degradation, annual_om_cost=float(om),
        inverter_replacement_cost=float(inv_cost), inverter_replacement_year=int(inv_year),
        system_life_years=int(life),
    )
    MC = dict(n_simulations=int(n_sims), generation_sigma=gen_sigma,
              tariff_escalation_sigma=esc_sigma, degradation_mean=degradation)

    st.caption("Every number here is an assumption you can defend or change. "
               "The Risk tab shows which ones actually move the answer.")


# --------------------------------------------------------------------------- #
# Header + upload
# --------------------------------------------------------------------------- #

st.markdown('<div class="hero">SolarROI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Reads your ShinePhone charts and your electricity bills, reconstructs the '
    'bills you would have paid without the panels, and estimates when the system pays for itself.</div>',
    unsafe_allow_html=True,
)
st.write("")

# Once an analysis has run, the input steps fold away so the results own the
# screen; they stay one click from reach for changing a file or a value.
HAS_RUN = st.session_state.get("has_run", False)
inputs = st.expander("Inputs · files and extracted values", expanded=False) if HAS_RUN else st.container()

with inputs:
    use_demo = st.toggle("Use the bundled sample data", value=True,
                         help="Turn this off to analyse your own files.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="step"><b>1 · ShinePhone screenshots</b><br>'
                    '<span class="sub">Open the app → Plant → Power|Energy. Screenshot the '
                    '<b>Month</b> chart for each year, and the <b>Year</b> chart. '
                    'Upload as many as you have.</span></div>', unsafe_allow_html=True)
        shots = st.file_uploader("Screenshots", type=["png", "jpg", "jpeg"],
                                 accept_multiple_files=True, disabled=use_demo,
                                 label_visibility="collapsed")
    with col_b:
        st.markdown('<div class="step"><b>2 · Electricity bill history</b><br>'
                    '<span class="sub">A CSV with columns <code>year, month, grid_kwh, '
                    'bill_amount</code> (and <code>solar_kwh</code> if you have it). '
                    'Download it from your utility\'s portal.</span></div>', unsafe_allow_html=True)
        bill_file = st.file_uploader("billdata.csv", type=["csv"], disabled=use_demo,
                                     label_visibility="collapsed")

    # ---- resolve inputs ------------------------------------------------------- #
    if use_demo:
        try:
            bill_bytes = open(DEMO_BILLS, "rb").read()
            images = [(p.split("/")[-1], open(p, "rb").read()) for p in DEMO_SHOTS]
        except FileNotFoundError:
            st.error("Sample data not found. Turn the toggle off and upload your own files.")
            st.stop()
    else:
        if bill_file is None:
            st.info("Upload your bill CSV to continue. Screenshots are optional but they extend "
                    "the history past your last bill.")
            st.stop()
        bill_bytes = bill_file.getvalue()
        images = [(f.name, f.getvalue()) for f in (shots or [])]

    # --------------------------------------------------------------------------- #
    # Extraction review
    # --------------------------------------------------------------------------- #

    st.markdown('<div class="step"><b>3 · Check what was read off the screenshots</b><br>'
                '<span class="sub">Bar heights are measured in pixels and converted using the '
                'axis labels. Edit any value that looks wrong before running the analysis.'
                '</span></div>', unsafe_allow_html=True)

    records, ex_meta = [], []
    if images:
        prog = st.progress(0.0, "Reading charts…")
        for i, (name, payload) in enumerate(images):
            try:
                r = _extract_cached(name, payload)
                ex_meta.append(r)
                if r["kind"] == "month" and r["year"]:
                    for p, v in zip(r["periods"], r["values"]):
                        if 1 <= p <= 12:
                            records.append({"year": int(r["year"]), "month": int(p), "solar_kwh": float(v)})
            except Exception as exc:  # noqa: BLE001
                ex_meta.append({"name": name, "kind": "failed", "confidence": 0.0,
                                "warnings": [str(exc)], "periods": [], "values": [], "year": None})
            prog.progress((i + 1) / len(images), f"Reading charts… {i + 1}/{len(images)}")
        prog.empty()

    if ex_meta:
        cols = st.columns(min(len(ex_meta), 5))
        for c, m in zip(cols, ex_meta):
            conf = m["confidence"]
            colour = P["good"] if conf >= 0.85 else P["warning"] if conf >= 0.7 else P["critical"]
            label = "verified" if conf >= 0.85 else "check it" if conf >= 0.7 else "unreliable"
            title = f"{m['kind']} {m['year'] or ''}".strip()
            c.markdown(
                f'<div class="card"><div class="k" style="word-break:break-all">{m["name"][:22]}</div>'
                f'<div class="v" style="font-size:1.15rem">{title}</div>'
                f'<div class="n"><span style="color:{colour}">● {label}</span> · '
                f'{len(m["values"])} bars · {conf:.0%} confidence</div></div>',
                unsafe_allow_html=True,
            )
        st.write("")

    edited = pd.DataFrame(records)
    if len(edited):
        edited = edited.sort_values(["year", "month"]).reset_index(drop=True)
        st.markdown(f"**Extracted monthly generation — {len(edited)} months.** "
                    "Values are kWh per month; edit anything the reader got wrong. "
                    "Where these overlap the CSV, the metered CSV figure wins.")
        edited = st.data_editor(edited, num_rows="dynamic", use_container_width=True, height=240)
        yearly = [m for m in ex_meta if m["kind"] == "year"]
        if yearly:
            y = yearly[0]
            st.caption("Annual totals read from the Year chart: "
                       + " · ".join(f"{p}: {v:,.0f} kWh" for p, v in zip(y["periods"], y["values"])))
    elif images:
        st.warning("Nothing could be read from the screenshots. The analysis will use the CSV alone.")

    run = st.button("Run the analysis", type="primary", use_container_width=True)
    if run and not st.session_state.get("has_run"):
        # Re-run once so the input section renders in its collapsed form and the
        # results appear at the top of the page rather than below the uploaders.
        st.session_state["has_run"] = True
        st.rerun()
    if not st.session_state.get("has_run"):
        st.stop()

with st.spinner("Fitting models and simulating 25 years…"):
    try:
        R = _analyse_cached(bill_bytes, edited.to_json(orient="records") if len(edited) else "",
                            ASSUMPTIONS, MC)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")
        st.stop()

for w in R.warnings:
    st.warning(w)

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tabs = st.tabs(["Verdict", "Savings", "Forecast", "Risk", "Panel health", "Models", "Data quality"])
mc, pb, rl = R.monte_carlo, R.payback, R.realised

# ------------------------------- Verdict ----------------------------------- #
with tabs[0]:
    c = st.columns(4)
    c[0].markdown(stat("Payback", f"{mc.p50:.1f} yrs",
                       f"P10–P90 {mc.p10:.1f} – {mc.p90:.1f} · {mc.prob_within_life*100:.0f}% chance within {R.assumptions.system_life_years} yrs"),
                  unsafe_allow_html=True)
    c[1].markdown(stat("Recovered so far", f"{rl['capex_recovered_pct']:.1f}%",
                       f"{inr(rl['total_savings'])} of {inr(R.assumptions.capex)} in {rl['years_observed']:.1f} yrs"),
                  unsafe_allow_html=True)
    c[2].markdown(stat("Value per unit", f"₹{R.value_per_kwh:.2f}",
                       "what each generated kWh actually saves you"), unsafe_allow_html=True)
    c[3].markdown(stat("CO₂ avoided", f"{rl['co2_avoided_kg']/1000:.2f} t",
                       f"{rl['total_generation_kwh']:,.0f} kWh generated to date"), unsafe_allow_html=True)
    st.write("")

    c = st.columns(4)
    c[0].markdown(stat("NPV", inr(pb.npv), f"at a {R.assumptions.discount_rate:.0%} discount rate"),
                  unsafe_allow_html=True)
    c[1].markdown(stat("IRR", f"{pb.irr*100:.1f}%" if np.isfinite(pb.irr) else "n/a",
                       "annualised return over the system life"), unsafe_allow_html=True)
    c[2].markdown(stat("Your cost per unit", f"₹{pb.lcoe:.2f}",
                       "levelised; compare with your grid tariff"), unsafe_allow_html=True)
    c[3].markdown(stat("Lifetime savings", inr(pb.total_lifetime_savings),
                       f"over {R.assumptions.system_life_years} years, before capex"), unsafe_allow_html=True)
    st.write("")

    sched = pb.schedule
    fig = go.Figure()
    fig.add_hline(y=R.assumptions.capex, line=dict(color=P["axis"], width=1))
    fig.add_annotation(x=sched["year"].iloc[-1], y=R.assumptions.capex, text="system cost",
                       showarrow=False, yshift=12, xanchor="right",
                       font=dict(color=P["muted"], size=12))
    realised_n = int((sched["source"] == "realised").sum())
    fig.add_trace(go.Scatter(
        x=sched["year"], y=sched["cumulative"], name="Cumulative savings",
        mode="lines", line=dict(color=P["series"][0], width=2),
        fill="tozeroy", fillcolor=P["band"], hovertemplate="Year %{x}: ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=sched["year"], y=sched["cumulative_discounted"], name="In today's money",
        mode="lines", line=dict(color=P["series"][1], width=2, dash="dot"),
        hovertemplate="Year %{x}: ₹%{y:,.0f}<extra></extra>"))
    if realised_n:
        fig.add_vrect(x0=0.5, x1=realised_n + 0.5, fillcolor=P["muted"], opacity=0.07, line_width=0)
        fig.add_annotation(x=(realised_n + 1) / 2, y=sched["cumulative"].max(), text="measured",
                           showarrow=False, font=dict(color=P["muted"], size=12))
    if np.isfinite(pb.simple_payback_years):
        fig.add_trace(go.Scatter(
            x=[pb.simple_payback_years], y=[R.assumptions.capex], mode="markers+text",
            marker=dict(color=P["series"][0], size=11, line=dict(color=P["surface"], width=2)),
            text=[f"  breakeven, year {pb.simple_payback_years:.1f}"], textposition="middle right",
            textfont=dict(color=P["text"], size=12), name="Breakeven", showlegend=False,
            hoverinfo="skip"))
    fig.update_layout(title="When the system pays for itself")
    fig.update_xaxes(title="Year since installation")
    fig.update_yaxes(title="Cumulative savings (₹)")
    st.plotly_chart(apply_layout(fig, MODE, 420), use_container_width=True)

    beats_grid = pb.lcoe < R.value_per_kwh
    verdict = (
        f"On {rl['years_observed']:.1f} years of measured bills the array has returned "
        f"**{inr(rl['total_savings'])}** — {rl['capex_recovered_pct']:.0f}% of what it cost. "
        f"Extending that with a {R.assumptions.tariff_escalation:.1%} annual tariff rise and "
        f"{R.assumptions.degradation_rate:.1%} panel degradation, the median simulation breaks even "
        f"in **year {mc.p50:.1f}**, with a P10–P90 range of {mc.p10:.1f} to {mc.p90:.1f} years.\n\n"
        f"Your levelised cost of solar is **₹{pb.lcoe:.2f}/kWh** against a realised grid value of "
        f"**₹{R.value_per_kwh:.2f}/kWh**, so over the full life the panels "
        + ("**undercut** the grid." if beats_grid else
           "**cost more per unit** than grid power at today's tariff — the case rests on tariffs "
           "rising faster than assumed, or on the system lasting well past its warranty.")
    )
    st.info(verdict)
    if mc.prob_positive_npv < 0.5:
        st.caption(
            f"Only {mc.prob_positive_npv:.0%} of simulations clear a {R.assumptions.discount_rate:.0%} "
            "discount rate. The array is very likely to return its cost in cash, and unlikely to beat "
            "that benchmark as a pure investment — both statements can be true at once."
        )

# ------------------------------- Savings ----------------------------------- #
with tabs[1]:
    h = R.history
    st.markdown("**What you paid, against what you would have paid.** The counterfactual is "
                "estimated two independent ways; where they agree, the savings figure is solid.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["counterfactual_bill_tariff"],
                             name="Without solar — tariff model", mode="lines",
                             line=dict(color=P["series"][1], width=2),
                             hovertemplate="₹%{y:,.0f}<extra>tariff model</extra>"))
    if h["counterfactual_bill_synthetic"].notna().any():
        fig.add_trace(go.Scatter(x=h["date"], y=h["counterfactual_bill_synthetic"],
                                 name="Without solar — pre-solar trend", mode="lines",
                                 line=dict(color=P["series"][2], width=2, dash="dot"),
                                 hovertemplate="₹%{y:,.0f}<extra>synthetic control</extra>"))
    fig.add_trace(go.Scatter(x=h["date"], y=h["bill_amount"], name="What you actually paid",
                             mode="lines", line=dict(color=P["series"][0], width=2),
                             fill="tozeroy", fillcolor=P["band"],
                             hovertemplate="₹%{y:,.0f}<extra>actual</extra>"))
    fig.update_layout(title="Monthly bill: actual vs counterfactual")
    fig.update_yaxes(title="₹ per month")
    st.plotly_chart(apply_layout(fig, MODE, 400), use_container_width=True)

    c = st.columns(3)
    c[0].metric("Average monthly saving", inr(rl["avg_monthly_savings"]))
    c[1].metric("Last 12 months", inr(rl["last12_savings"]))
    agree = rl.get("counterfactual_agreement_pct", np.nan)
    c[2].metric("Agreement between methods", f"{agree:.0f}%" if np.isfinite(agree) else "n/a",
                help="100% would mean the two independent counterfactuals produce identical savings.")

    n_odd = rl.get("n_anomalous_months", 0)
    if n_odd:
        st.caption(
            f"As billed, the months since installation total {inr(rl['total_savings_as_billed'])}. "
            f"{n_odd} month(s) carry an arrears or true-up bill — money belonging to other months — "
            f"so the headline figure of {inr(rl['total_savings'])} is the mean saving over clean "
            "months scaled to the observed period, rather than letting one adjustment bill distort it."
        )

    # The arrears month is separated out rather than left to dominate the scale
    # as an ordinary bar: its colour means "billing artefact", not "series 2".
    odd = h["is_billing_anomaly"].fillna(False)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=h.loc[~odd, "date"], y=h.loc[~odd, "savings"], name="Monthly saving",
                          marker=dict(color=P["series"][0], line=dict(width=0)),
                          hovertemplate="₹%{y:,.0f}<extra></extra>"))
    if odd.any():
        fig2.add_trace(go.Bar(x=h.loc[odd, "date"], y=h.loc[odd, "savings"],
                              name="Arrears / true-up bill",
                              marker=dict(color=P["serious"], line=dict(width=0)),
                              hovertemplate="₹%{y:,.0f}<extra>billing artefact</extra>"))
    fig2.update_layout(title="Saving realised each month", bargap=0.35, barmode="relative")
    fig2.update_yaxes(title="₹ saved")
    st.plotly_chart(apply_layout(fig2, MODE, 320, legend=bool(odd.any())),
                    use_container_width=True)

    st.markdown("**Month-by-month detail**")
    show = h[["period", "grid_kwh", "solar_kwh", "bill_amount",
              "counterfactual_bill_tariff", "counterfactual_bill_synthetic",
              "savings", "value_per_kwh"]].copy()
    show["period"] = show["period"].astype(str)
    st.dataframe(
        show.rename(columns={
            "period": "Month", "grid_kwh": "Billed units", "solar_kwh": "Generated",
            "bill_amount": "Paid ₹", "counterfactual_bill_tariff": "Would have paid (tariff) ₹",
            "counterfactual_bill_synthetic": "Would have paid (trend) ₹",
            "savings": "Saved ₹", "value_per_kwh": "₹ per generated unit"}).round(1),
        hide_index=True, use_container_width=True, height=320)

# ------------------------------- Forecast ---------------------------------- #
with tabs[2]:
    st.markdown("**Generation to date, and the projection it feeds.** Seasonality is learned "
                "from your own history; the long decline is the assumed degradation rate.")
    hist = R.generation_fit.history
    proj = R.projection.copy()
    hist_dates = hist["period"].dt.to_timestamp()
    proj_dates = proj["period"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_dates, y=hist["solar_kwh"], name="Measured",
                             mode="lines", line=dict(color=P["series"][0], width=2),
                             hovertemplate="%{y:,.0f} kWh<extra>measured</extra>"))
    fig.add_trace(go.Scatter(x=proj_dates[:120], y=proj["solar_kwh"][:120], name="Projected",
                             mode="lines", line=dict(color=P["series"][1], width=2, dash="dot"),
                             hovertemplate="%{y:,.0f} kWh<extra>projected</extra>"))
    fig.update_layout(title="Monthly generation, measured and projected (next 10 years)")
    fig.update_yaxes(title="kWh per month")
    st.plotly_chart(apply_layout(fig, MODE, 380), use_container_width=True)

    c = st.columns(3)
    c[0].metric("Typical year", f"{R.generation_fit.annual_yield_kwh:,.0f} kWh")
    fitted_deg = R.generation_fit.fitted_degradation
    c[1].metric("Degradation measured in your data",
                f"{fitted_deg*100:.2f} %/yr" if np.isfinite(fitted_deg) else "n/a",
                help="Estimated from output against panel age. With only a few years of history "
                     "this is noisy, so the projection uses the assumption in the sidebar instead.")
    c[2].metric("Best / worst month",
                f"{hist.groupby('month')['solar_kwh'].mean().idxmax()} / "
                f"{hist.groupby('month')['solar_kwh'].mean().idxmin()}",
                help="Calendar months with the highest and lowest average output.")

    shape = hist.groupby("month")["solar_kwh"].mean().reindex(range(1, 13))
    figs = go.Figure(go.Bar(
        x=[pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)], y=shape.values,
        marker=dict(color=P["series"][0], line=dict(width=0)),
        hovertemplate="%{x}: %{y:,.0f} kWh<extra></extra>"))
    figs.update_layout(title="Seasonal shape — average output by calendar month", bargap=0.3)
    figs.update_yaxes(title="kWh")
    st.plotly_chart(apply_layout(figs, MODE, 300, legend=False), use_container_width=True)

    st.markdown("**Projected annual cash flow**")
    cf = R.cashflows.copy()
    cf["Year"] = cf["year_index"] + 1
    st.dataframe(
        cf[["Year", "generation_kwh", "value_per_kwh", "gross_savings", "om_cost",
            "capex_event", "net_cashflow"]].rename(columns={
                "generation_kwh": "kWh", "value_per_kwh": "₹/kWh", "gross_savings": "Gross ₹",
                "om_cost": "Upkeep ₹", "capex_event": "One-off ₹", "net_cashflow": "Net ₹"}).round(0),
        hide_index=True, use_container_width=True, height=300)

# --------------------------------- Risk ------------------------------------ #
with tabs[3]:
    st.markdown(f"**{MC['n_simulations']:,} simulated futures.** Each draws its own weather, "
                "tariff path, degradation rate, upkeep bill and chance of a fault. Years already "
                "measured are held fixed — they happened.")

    valid = mc.paybacks[~np.isnan(mc.paybacks)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=valid, nbinsx=45, name="Simulations",
                               marker=dict(color=P["series"][0], line=dict(width=0)),
                               hovertemplate="%{x:.0f} yrs: %{y} runs<extra></extra>"))
    for q, val, col in (("P10", mc.p10, P["good"]), ("P50", mc.p50, P["text"]), ("P90", mc.p90, P["serious"])):
        fig.add_vline(x=val, line=dict(color=col, width=2, dash="dot"))
        fig.add_annotation(x=val, y=1, yref="paper", text=f"{q}·{val:.1f}", showarrow=False,
                           yshift=-4, xshift=26, font=dict(color=col, size=12))
    fig.update_layout(title="Distribution of payback year", bargap=0.02)
    fig.update_xaxes(title="Years to break even")
    fig.update_yaxes(title="Simulations")
    st.plotly_chart(apply_layout(fig, MODE, 380, legend=False), use_container_width=True)

    c = st.columns(4)
    c[0].metric("Median payback", f"{mc.p50:.1f} yrs")
    c[1].metric("Optimistic (P10)", f"{mc.p10:.1f} yrs")
    c[2].metric("Pessimistic (P90)", f"{mc.p90:.1f} yrs")
    c[3].metric("Breaks even in system life", f"{mc.prob_within_life*100:.0f}%")

    st.markdown("**Which assumption moves the answer?** Each bar holds everything else fixed.")
    s = R.sensitivity.copy()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=s["factor"], x=s["worse"] - s["base"], base=s["base"],
                          orientation="h", name="Unfavourable",
                          marker=dict(color=P["serious"], line=dict(color=P["surface"], width=2)),
                          hovertemplate="%{y}: %{x:+.1f} yrs<extra>unfavourable</extra>"))
    fig2.add_trace(go.Bar(y=s["factor"], x=s["better"] - s["base"], base=s["base"],
                          orientation="h", name="Favourable",
                          marker=dict(color=P["good"], line=dict(color=P["surface"], width=2)),
                          hovertemplate="%{y}: %{x:+.1f} yrs<extra>favourable</extra>"))
    fig2.add_vline(x=s["base"].iloc[0], line=dict(color=P["axis"], width=1))
    fig2.update_layout(title="Sensitivity of payback to each assumption", barmode="overlay",
                       hovermode="closest")
    fig2.update_xaxes(title="Payback (years)")
    fig2.update_yaxes(autorange="reversed", showgrid=False)
    st.plotly_chart(apply_layout(fig2, MODE, 360), use_container_width=True)
    st.dataframe(s.round(2).rename(columns={
        "factor": "Assumption", "worse": "Unfavourable", "base": "Base",
        "better": "Favourable", "swing": "Swing (yrs)"}),
        hide_index=True, use_container_width=True)

# ----------------------------- Panel health -------------------------------- #
with tabs[4]:
    st.markdown("**Is the array still performing?** Each month is compared with what the model "
                "expects for that season and panel age. A month is flagged only when the "
                "residual test and an Isolation Forest agree *and* the shortfall is over 10%.")
    an = R.anomalies.copy()
    an_dates = an["period"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an_dates, y=an["predicted_kwh"], name="Expected",
                             mode="lines", line=dict(color=P["series"][1], width=2, dash="dot"),
                             hovertemplate="%{y:,.0f} kWh<extra>expected</extra>"))
    fig.add_trace(go.Scatter(x=an_dates, y=an["solar_kwh"], name="Measured",
                             mode="lines", line=dict(color=P["series"][0], width=2),
                             hovertemplate="%{y:,.0f} kWh<extra>measured</extra>"))
    flagged = an[an["severity"].isin(["high", "watch"])]
    if len(flagged):
        fig.add_trace(go.Scatter(
            x=flagged["period"].dt.to_timestamp(), y=flagged["solar_kwh"], mode="markers",
            name="Flagged", marker=dict(
                size=11, color=[P["critical"] if s == "high" else P["warning"] for s in flagged["severity"]],
                line=dict(color=P["surface"], width=2), symbol="circle"),
            hovertemplate="%{x|%b %Y}: %{y:,.0f} kWh<extra>flagged</extra>"))
    fig.update_layout(title="Measured output against expectation")
    fig.update_yaxes(title="kWh per month")
    st.plotly_chart(apply_layout(fig, MODE, 380), use_container_width=True)

    high = an[an["severity"] == "high"]
    if len(high):
        st.error(f"{len(high)} month(s) fell materially short of expectation: "
                 + ", ".join(f"{p:%b %Y} ({s:.0f}% low)" for p, s in
                             zip(high['period'].dt.to_timestamp(), high['shortfall_pct'])))
        st.caption("Usual causes, cheapest first: soiling on the panels, new shading (a grown tree, "
                   "a neighbour's construction), a tripped string, or inverter derating in heat.")
    else:
        st.success("No month shows a shortfall large enough to suggest a fault.")

    tbl = an[["period", "solar_kwh", "predicted_kwh", "shortfall_pct", "residual_z", "severity"]].copy()
    tbl["period"] = tbl["period"].astype(str)
    st.dataframe(tbl.rename(columns={
        "period": "Month", "solar_kwh": "Measured kWh", "predicted_kwh": "Expected kWh",
        "shortfall_pct": "Shortfall %", "residual_z": "Residual (σ)", "severity": "Status"}).round(2),
        hide_index=True, use_container_width=True, height=300)

# -------------------------------- Models ----------------------------------- #
with tabs[5]:
    st.markdown("**How the models were chosen.** Every candidate is scored with rolling-origin "
                "cross-validation, so no model is ever trained on months that come after the ones "
                "it is tested on.")
    for name, board in R.leaderboards.items():
        st.markdown(f"**{name}**")
        b = board.copy()
        if "selectable" in b.columns:
            b["note"] = np.where(b["selected"], "← selected",
                                 np.where(~b["selectable"], "excluded: cannot extrapolate", ""))
            b = b.drop(columns=["selectable", "selected"])
        st.dataframe(b.round(2), use_container_width=True)
    st.caption("Tree ensembles predict a constant outside the range they were trained on. For the "
               "model whose whole job is projecting a trend into unseen years, they are excluded on "
               "principle even when they win on cross-validated error — the leaderboard shows both.")

    st.divider()
    st.markdown("**What drives the tariff model**")
    with st.spinner("Computing SHAP values…"):
        Xt = build_tariff_frame(modelling_frame(R.bills), R.tariff_model.origin)
        expl = compute_shap(R.tariff_model.fitted, Xt)
    imp = expl.importance.head(10).iloc[::-1]
    figi = go.Figure(go.Bar(
        x=imp["mean_abs_shap"], y=imp["feature"], orientation="h",
        marker=dict(color=P["series"][0], line=dict(width=0)),
        hovertemplate="%{y}: %{x:,.1f}<extra></extra>"))
    figi.update_layout(title=f"Feature importance ({expl.method})")
    figi.update_xaxes(title="Mean absolute contribution to the predicted bill (₹)")
    figi.update_yaxes(showgrid=False)
    st.plotly_chart(apply_layout(figi, MODE, 360, legend=False), use_container_width=True)
    st.info(narrate(expl))

    if expl.shap_values is not None:
        idx = st.slider("Explain a single month", 0, len(Xt) - 1, len(Xt) - 1,
                        help="Shows how each feature pushed that month's predicted bill up or down.")
        contrib = explain_instance(expl, idx).head(8).iloc[::-1]
        month_label = str(Xt["period"].iloc[idx])
        figw = go.Figure(go.Bar(
            x=contrib["contribution"], y=contrib["feature"], orientation="h",
            marker=dict(color=[P["series"][1] if v < 0 else P["series"][0]
                               for v in contrib["contribution"]], line=dict(width=0)),
            hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
        figw.add_vline(x=0, line=dict(color=P["axis"], width=1))
        figw.update_layout(title=f"Why the model predicted this bill for {month_label}")
        figw.update_xaxes(title="Contribution (₹)")
        figw.update_yaxes(showgrid=False)
        st.plotly_chart(apply_layout(figw, MODE, 340, legend=False), use_container_width=True)

# ----------------------------- Data quality -------------------------------- #
with tabs[6]:
    st.markdown("**What the cleaner found.** Every repair is recorded rather than applied silently.")
    for line in R.quality.to_lines():
        st.markdown(f"- {line}")

    es = R.extraction_stats
    if es.get("overlap_n"):
        st.divider()
        st.markdown("**Screenshots checked against the metered CSV**")
        c = st.columns(3)
        c[0].metric("Overlapping months", es["overlap_n"])
        c[1].metric("Mean error vs meter", f"{es['overlap_mape']:.2f}%",
                    help="How far the pixel-measured values sit from the metered figures.")
        c[2].metric("Months added beyond the CSV", es.get("months_added", 0))
        st.caption("The screenshots are validated on the months both sources cover, then trusted "
                   "to extend the record past the last bill.")

    st.divider()
    st.markdown("**Download**")
    c = st.columns(3)
    c[0].download_button("Monthly savings (CSV)",
                         R.history.assign(period=R.history["period"].astype(str)).to_csv(index=False),
                         "monthly_savings.csv", use_container_width=True)
    c[1].download_button("Cash-flow schedule (CSV)", pb.schedule.to_csv(index=False),
                         "cashflow_schedule.csv", use_container_width=True)
    summary = {"headline": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                            for k, v in R.headline().items()},
               "assumptions": R.assumptions.to_dict(),
               "monte_carlo": mc.summary(),
               "value_per_kwh": R.value_per_kwh,
               "models": {k: v.fitted.name if hasattr(v, "fitted") else None
                          for k, v in {"tariff": R.tariff_model}.items()}}
    c[2].download_button("Full summary (JSON)", json.dumps(summary, indent=2, default=str),
                         "solar_roi_summary.json", use_container_width=True)

    with st.expander("Cleaned monthly dataset"):
        b = R.bills.copy()
        b["period"] = b["period"].astype(str)
        st.dataframe(b.round(2), hide_index=True, use_container_width=True, height=380)
