"""Model explainability.

An ROI number nobody can interrogate is worth very little, so every model in
the project can be asked two questions:

    "which features drive your predictions overall?"   -> global_importance
    "why this number for this month?"                  -> explain_instance

SHAP is used where it applies.  Because the model zoo mixes tree ensembles with
scaled linear pipelines, the explainer is chosen per model class, and there is
a permutation-importance fallback so the UI never breaks just because SHAP
cannot handle an estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import RANDOM_STATE

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAP_AVAILABLE = False


READABLE_NAMES = {
    "grid_kwh": "Units billed",
    "kwh_over_100": "Units above 100 (slab 2)",
    "kwh_over_200": "Units above 200 (slab 3)",
    "kwh_over_300": "Units above 300 (slab 4)",
    "kwh_over_400": "Units above 400 (slab 5)",
    "t_years": "Years elapsed (tariff drift)",
    "month_sin": "Season (annual cycle, sin)",
    "month_cos": "Season (annual cycle, cos)",
    "month_sin2": "Season (half-annual cycle, sin)",
    "month_cos2": "Season (half-annual cycle, cos)",
    "season_summer": "Summer (Mar-May)",
    "season_monsoon": "Monsoon (Jun-Sep)",
    "season_post_monsoon": "Post-monsoon (Oct-Nov)",
    "panel_age_years": "Panel age (degradation)",
}


def pretty(name: str) -> str:
    return READABLE_NAMES.get(name, name)


@dataclass
class Explanation:
    importance: pd.DataFrame
    method: str
    shap_values: np.ndarray | None = None
    base_value: float | None = None
    X: pd.DataFrame | None = None


def _unwrap(estimator):
    """Return (final_estimator, transform_fn) for a pipeline or bare model."""
    from sklearn.pipeline import Pipeline

    if isinstance(estimator, Pipeline):
        final = estimator.steps[-1][1]
        pre = estimator[:-1]
        return final, (lambda X: pre.transform(X))
    return estimator, (lambda X: X)


def compute_shap(fitted, X: pd.DataFrame, max_background: int = 100) -> Explanation:
    """SHAP values for a fitted model, with a graceful fallback."""
    Xf = X[fitted.features]
    if SHAP_AVAILABLE:
        try:
            model, transform = _unwrap(fitted.estimator)
            name = type(model).__name__
            if any(k in name for k in ("Forest", "Boosting", "Tree")):
                explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(transform(Xf))
                base = float(np.mean(explainer.expected_value))
            else:
                bg = shap.sample(Xf, min(max_background, len(Xf)), random_state=RANDOM_STATE)
                explainer = shap.Explainer(fitted.estimator.predict, bg)
                sv = explainer(Xf, silent=True)
                values, base = sv.values, float(np.mean(sv.base_values))
            imp = (
                pd.DataFrame(
                    {
                        "feature": [pretty(f) for f in fitted.features],
                        "raw_feature": fitted.features,
                        "mean_abs_shap": np.abs(values).mean(axis=0),
                    }
                )
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            return Explanation(importance=imp, method="SHAP", shap_values=values, base_value=base, X=Xf)
        except Exception:
            pass  # fall through to permutation importance
    return permutation_explanation(fitted, X)


def permutation_explanation(fitted, X: pd.DataFrame, y: pd.Series | None = None) -> Explanation:
    """Model-agnostic fallback: how much does shuffling each feature hurt?"""
    Xf = X[fitted.features]
    target = y if y is not None else pd.Series(fitted.estimator.predict(Xf), index=Xf.index)
    res = permutation_importance(
        fitted.estimator, Xf, target, n_repeats=20, random_state=RANDOM_STATE, scoring="r2"
    )
    imp = (
        pd.DataFrame(
            {
                "feature": [pretty(f) for f in fitted.features],
                "raw_feature": fitted.features,
                "mean_abs_shap": res.importances_mean,
                "std": res.importances_std,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return Explanation(importance=imp, method="Permutation importance", X=Xf)


def explain_instance(explanation: Explanation, row: int) -> pd.DataFrame:
    """Per-feature contributions for a single month (waterfall-ready)."""
    if explanation.shap_values is None:
        raise ValueError("Per-month explanations need SHAP values; the model fell back to permutation importance.")
    vals = explanation.shap_values[row]
    return (
        pd.DataFrame(
            {
                "feature": [pretty(c) for c in explanation.X.columns],
                "value": explanation.X.iloc[row].to_numpy(),
                "contribution": vals,
            }
        )
        .sort_values("contribution", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def narrate(explanation: Explanation, top_n: int = 3) -> str:
    """One plain-English sentence about what the model keyed on."""
    top = explanation.importance.head(top_n)["feature"].tolist()
    if not top:
        return "No feature attribution available."
    joined = ", ".join(top[:-1]) + (" and " + top[-1] if len(top) > 1 else "")
    return f"{explanation.method} attributes most of this model's output to {joined}."
