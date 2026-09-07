#!/usr/bin/env python3
"""Command-line entry point - the same pipeline the app runs, without the UI.

Useful for reproducing the numbers in the README, for CI, and for anyone who
wants the results as files rather than as a web page.

    python analyse.py --capex 170000 --outdir outputs
"""

from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import numpy as np

from src.config import FinancialAssumptions, MonteCarloAssumptions
from src.pipeline import extractions_to_frame, run_analysis
from src.vision import extract_many

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate the ROI of a rooftop solar array.")
    ap.add_argument("--bills", default="data/billdata.csv", help="utility bill CSV")
    ap.add_argument("--screenshots", default="data/samples",
                    help="folder of ShinePhone screenshots (use '' to skip)")
    ap.add_argument("--capex", type=float, default=170_000.0, help="total installed cost, INR")
    ap.add_argument("--escalation", type=float, default=5.5, help="annual tariff rise, %%")
    ap.add_argument("--discount", type=float, default=7.0, help="discount rate, %%")
    ap.add_argument("--simulations", type=int, default=10_000)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    extracted = None
    if args.screenshots:
        folder = pathlib.Path(args.screenshots)
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        results, errors = extract_many([str(p) for p in files])
        for e in errors:
            print(f"  ! {e}")
        if results:
            months = sum(len(r.periods) for r in results if r.kind == "month")
            mean_conf = float(np.mean([r.confidence for r in results]))
            print(f"Read {len(results)} chart(s): {months} monthly values, "
                  f"mean confidence {mean_conf:.0%}")
            extracted = extractions_to_frame(results)

    result = run_analysis(
        args.bills,
        extracted,
        FinancialAssumptions(
            capex=args.capex,
            tariff_escalation=args.escalation / 100,
            discount_rate=args.discount / 100,
        ),
        MonteCarloAssumptions(n_simulations=args.simulations),
    )

    print("\nData quality")
    for line in result.quality.to_lines():
        print(f"  {line}")

    if result.warnings:
        print("\nWarnings")
        for w in result.warnings:
            print(f"  ! {w}")

    print("\nSelected models")
    for name, board in result.leaderboards.items():
        chosen = board.index[board["selected"]][0] if "selected" in board else board.index[0]
        row = board.loc[chosen]
        print(f"  {name}: {chosen}  (CV MAE {row['MAE']:.1f}, MAPE {row['MAPE_%']:.1f}%)")

    print("\nResults")
    for key, value in result.headline().items():
        shown = value if isinstance(value, str) else f"{value:,.2f}"
        print(f"  {key:<34} {shown}")

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    result.history.assign(period=result.history["period"].astype(str)).to_csv(
        out / "monthly_savings.csv", index=False)
    result.payback.schedule.to_csv(out / "cashflow_schedule.csv", index=False)
    result.anomalies.assign(period=result.anomalies["period"].astype(str)).to_csv(
        out / "panel_health.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "headline": result.headline(),
        "monte_carlo": result.monte_carlo.summary(),
        "assumptions": result.assumptions.to_dict(),
        "value_per_kwh": result.value_per_kwh,
        "extraction": result.extraction_stats,
    }, indent=2, default=str))
    print(f"\nWrote CSVs and summary.json to {out}/")


if __name__ == "__main__":
    main()
