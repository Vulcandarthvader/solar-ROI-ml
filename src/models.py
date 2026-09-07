"""Supervised models: tariff surrogate, counterfactual bill, generation forecast.

The ROI question "how long until the array pays for itself" reduces to one
quantity that is never directly observed: **the bill that would have arrived
if the panels were not there**.  Two independent estimators are built for it,
and the app reports both plus their disagreement.

Estimator A - tariff surrogate (physical route)
    Learn the utility's price function from the bills themselves,
    bill = f(billed units, calendar time), then re-query it with gross
    household demand (billed units + generation).  Uses post-solar months too,
    since they supply the low-consumption end of the price curve.

Estimator B - synthetic counterfactual (statistical route)
    Fit bills against calendar features on the **pre-solar era only**, then
    extrapolate that trajectory across the solar era.  Knows nothing about
    generation, so it is a genuine independent check - at the cost of assuming
    household demand did not change for other reasons.

Model choice is not hard-coded.  Every model is scored with rolling-origin
(TimeSeriesSplit) cross-validation and the winner is picked on out-of-fold MAE,
so the leaderboard is part of the output rather than a design decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CV_SPLITS, RANDOM_STATE
from .features import (
    CONSUMPTION_FEATURES,
    GENERATION_FEATURES,
    TARIFF_FEATURES,
    build_counterfactual_input,
    build_generation_frame,
    build_tariff_frame,
    future_frame,
)


# --------------------------------------------------------------------------- #
# Model zoo and evaluation
# --------------------------------------------------------------------------- #


def build_model_zoo() -> dict[str, object]:
    """Candidate regressors, from a trivial baseline to gradient boosting.

    The baseline is not decoration: with ~60 monthly observations it is
    entirely possible for a boosted forest to lose to the mean, and a
    leaderboard that cannot show that is not telling the truth.
    """
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Huber (robust linear)": Pipeline(
            [("scale", StandardScaler()), ("m", HuberRegressor(epsilon=1.35, max_iter=500))]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
            random_state=RANDOM_STATE,
        ),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": mape,
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def evaluate_models(
    X: pd.DataFrame, y: pd.Series, n_splits: int = CV_SPLITS
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Rolling-origin CV over the model zoo.

    TimeSeriesSplit, never KFold: shuffling months would let the model peek at
    the future, which on a series with a rising tariff trend inflates every
    score.
    """
    n_splits = max(2, min(n_splits, max(2, len(X) // 8)))
    cv = TimeSeriesSplit(n_splits=n_splits)
    rows, oof_store = [], {}

    for name, model in build_model_zoo().items():
        oof = np.full(len(y), np.nan)
        fold_scores = []
        for train_idx, test_idx in cv.split(X):
            m = _clone(model)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = m.predict(X.iloc[test_idx])
            oof[test_idx] = pred
            fold_scores.append(_metrics(y.iloc[test_idx].to_numpy(), pred))
        agg = {k: float(np.mean([f[k] for f in fold_scores])) for k in fold_scores[0]}
        agg["model"] = name
        agg["MAE_std"] = float(np.std([f["MAE"] for f in fold_scores]))
        rows.append(agg)
        oof_store[name] = oof

    board = pd.DataFrame(rows).set_index("model").sort_values("MAE")
    return board[["MAE", "MAE_std", "RMSE", "MAPE_%", "R2"]], oof_store


def _clone(model):
    from sklearn.base import clone

    return clone(model)


@dataclass
class FittedModel:
    """A selected model plus the evidence for selecting it."""

    name: str
    estimator: object
    features: list[str]
    leaderboard: pd.DataFrame
    cv_metrics: dict[str, float]
    n_train: int
    target: str = ""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X[self.features])


# Tree ensembles predict a constant beyond the range of their training data:
# a forest asked about year 2030 having only seen up to 2025 returns the 2025
# leaf value forever.  For any model whose job is to extrapolate a trend, that
# is not a small loss of accuracy, it is a silent failure.  Those tasks are
# restricted to model classes with a global functional form.
EXTRAPOLATING_MODELS = {"Ridge", "Huber (robust linear)", "Baseline (mean)"}


def select_and_fit(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    target: str,
    require_extrapolation: bool = False,
) -> FittedModel:
    """Score every candidate, then select the best *admissible* one.

    The leaderboard always reports the full zoo -- including models excluded by
    ``require_extrapolation`` -- so the trade-off is visible rather than
    hidden.  A tree winning on cross-validated MAE and still being the wrong
    choice is a real result, not something to paper over.
    """
    board, _ = evaluate_models(X[features], y)
    ranked = list(board.index)
    if require_extrapolation:
        admissible = [m for m in ranked if m in EXTRAPOLATING_MODELS]
        best_name = admissible[0] if admissible else ranked[0]
    else:
        best_name = ranked[0]

    est = _clone(build_model_zoo()[best_name])
    est.fit(X[features], y)
    board = board.assign(
        selectable=[(not require_extrapolation) or (m in EXTRAPOLATING_MODELS) for m in board.index],
        selected=[m == best_name for m in board.index],
    )
    return FittedModel(
        name=best_name,
        estimator=est,
        features=features,
        leaderboard=board,
        cv_metrics=board.loc[best_name].to_dict(),
        n_train=len(y),
        target=target,
    )


# --------------------------------------------------------------------------- #
# Estimator A - tariff surrogate
# --------------------------------------------------------------------------- #


class TariffModel:
    """Learns the utility's price function and answers counterfactual queries."""

    def __init__(self, origin: pd.Period | None = None):
        self.origin = origin
        self.fitted: FittedModel | None = None

    def fit(self, model_df: pd.DataFrame) -> "TariffModel":
        frame = build_tariff_frame(model_df, self.origin)
        self.origin = self.origin or frame["period"].min()
        self.fitted = select_and_fit(frame, frame["bill_amount"], TARIFF_FEATURES, "bill_amount")
        self._frame = frame
        # The counterfactual query pushes billed units *up* (gross demand rather
        # than net import).  If that lands outside the range the model was fitted
        # on, a tree ensemble will silently return its top-leaf value, so the
        # bound is recorded and checked at query time.
        self.units_range = (float(frame["grid_kwh"].min()), float(frame["grid_kwh"].max()))
        return self

    def predict_bill(self, df: pd.DataFrame) -> np.ndarray:
        frame = build_tariff_frame(df, self.origin)
        return self.fitted.predict(frame)

    def counterfactual_bill(self, df: pd.DataFrame) -> np.ndarray:
        """The bill for gross demand: every unit the house used, none offset."""
        frame = build_counterfactual_input(df, self.origin)
        return np.maximum(self.fitted.predict(frame), 0.0)

    def out_of_range_months(self, df: pd.DataFrame) -> int:
        """How many counterfactual queries sit beyond the fitted unit range."""
        if not hasattr(self, "units_range"):
            return 0
        gross = df["total_consumption_kwh"].to_numpy()
        lo, hi = self.units_range
        return int(((gross > hi) | (gross < lo)).sum())

    def marginal_rate(self, df: pd.DataFrame, delta: float = 25.0) -> np.ndarray:
        """Numerical d(bill)/d(units) at each month's gross demand.

        This is the rate solar actually earns.  Because tariffs are block-rate,
        generation displaces the *top* slab, so the marginal rate is materially
        higher than the average rate printed on the bill - and using the
        average is the single most common way household solar ROI gets
        understated.
        """
        hi = df.copy()
        hi["total_consumption_kwh"] = hi["total_consumption_kwh"] + delta / 2
        lo = df.copy()
        lo["total_consumption_kwh"] = np.maximum(lo["total_consumption_kwh"] - delta / 2, 0)
        up = self.counterfactual_bill(hi)
        dn = self.counterfactual_bill(lo)
        return (up - dn) / delta


# --------------------------------------------------------------------------- #
# Estimator B - synthetic counterfactual
# --------------------------------------------------------------------------- #


class SyntheticCounterfactual:
    """Pre-solar bill trajectory, extrapolated across the solar era.

    Trained only on months before commissioning, so it never sees a single
    solar-affected bill.  Its residual spread on the pre-solar era is carried
    forward as the honest uncertainty band on the counterfactual.
    """

    def __init__(self, origin: pd.Period | None = None):
        self.origin = origin
        self.fitted: FittedModel | None = None
        self.residual_sigma: float = np.nan

    def fit(self, pre_solar_df: pd.DataFrame) -> "SyntheticCounterfactual":
        from .features import add_calendar_features

        frame = add_calendar_features(pre_solar_df, self.origin)
        self.origin = self.origin or frame["period"].min()
        # This model exists to predict years it has never seen, so it must be
        # able to extrapolate its own trend term.
        self.fitted = select_and_fit(
            frame, frame["bill_amount"], CONSUMPTION_FEATURES, "bill_amount", require_extrapolation=True
        )
        resid = frame["bill_amount"].to_numpy() - self.fitted.predict(frame)
        self.residual_sigma = float(np.std(resid))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        from .features import add_calendar_features

        frame = add_calendar_features(df, self.origin)
        return np.maximum(self.fitted.predict(frame), 0.0)


# --------------------------------------------------------------------------- #
# Generation model + anomaly detection
# --------------------------------------------------------------------------- #


@dataclass
class GenerationFit:
    model: FittedModel
    commissioning: pd.Period
    origin: pd.Period
    residual_sigma: float
    monthly_shape: pd.Series
    annual_yield_kwh: float
    fitted_degradation: float = np.nan
    history: pd.DataFrame = field(default_factory=pd.DataFrame)


class GenerationModel:
    """Monthly generation as seasonality x panel age.

    Age is a feature rather than a post-hoc multiplier so the model can
    *measure* degradation from the data instead of assuming a datasheet value -
    though with only a few years of history that estimate is reported with a
    health warning, and the long projection uses the assumption from config.
    """

    def __init__(self, commissioning: pd.Period, origin: pd.Period | None = None):
        self.commissioning = commissioning
        self.origin = origin
        self.fit_result: GenerationFit | None = None

    def fit(self, solar_df: pd.DataFrame) -> GenerationFit:
        frame = build_generation_frame(solar_df, self.commissioning, self.origin)
        self.origin = self.origin or frame["period"].min()
        fitted = select_and_fit(frame, frame["solar_kwh"], GENERATION_FEATURES, "solar_kwh")
        pred = fitted.predict(frame)
        resid = frame["solar_kwh"].to_numpy() - pred

        # Degradation read off the data: regress log(output / seasonal norm) on age.
        shape = frame.groupby("month")["solar_kwh"].mean()
        norm = frame["month"].map(shape).to_numpy()
        ok = (norm > 0) & (frame["solar_kwh"].to_numpy() > 0)
        deg = np.nan
        if ok.sum() > 12:
            slope = np.polyfit(
                frame["panel_age_years"].to_numpy()[ok],
                np.log(frame["solar_kwh"].to_numpy()[ok] / norm[ok]),
                1,
            )[0]
            deg = float(-slope)

        full_years = frame.groupby("year")["solar_kwh"].agg(["sum", "count"])
        complete = full_years[full_years["count"] == 12]["sum"]
        annual = float(complete.mean()) if len(complete) else float(frame["solar_kwh"].sum() / len(frame) * 12)

        frame = frame.assign(predicted_kwh=pred, residual=resid)
        self.fit_result = GenerationFit(
            model=fitted,
            commissioning=self.commissioning,
            origin=self.origin,
            residual_sigma=float(np.std(resid)),
            monthly_shape=shape,
            annual_yield_kwh=annual,
            fitted_degradation=deg,
            history=frame,
        )
        return self.fit_result

    def project(self, start: pd.Period, n_months: int, degradation: float) -> pd.DataFrame:
        """Project generation forward, damping the model's own age term.

        The model only ever saw two or three years of age, so its internal age
        response cannot be trusted decades out.  Predictions are made at the
        last observed age and then declined explicitly at the configured
        degradation rate - extrapolating a tree model past its training range
        would simply flat-line, which quietly overstates late-life output.
        """
        fr = self.fit_result
        fut = future_frame(start, n_months, self.commissioning, fr.origin)
        anchor_age = float(fr.history["panel_age_years"].max())
        held = fut.copy()
        held["panel_age_years"] = anchor_age
        base = np.maximum(fr.model.predict(held), 0.0)
        extra_age = fut["panel_age_years"].to_numpy() - anchor_age
        fut["solar_kwh"] = base * (1 - degradation) ** np.maximum(extra_age, 0)
        return fut[["period", "year", "month", "panel_age_years", "solar_kwh"]]


def detect_anomalies(gen_fit: GenerationFit, contamination: float = 0.12) -> pd.DataFrame:
    """Flag months where the array underperformed its own seasonal norm.

    Two detectors, deliberately different in kind:

    * a one-sided residual z-score, which is interpretable and directional -
      only *shortfalls* matter, a sunny month is not a fault;
    * an Isolation Forest over (residual, shortfall %, month, age), which is
      unsupervised and can catch odd combinations the z-score misses.

    A month flagged by both is worth walking onto the roof for: the usual
    physical causes are soiling, new shading, a tripped string or inverter
    derating.
    """
    hist = gen_fit.history.copy()
    resid = hist["residual"].to_numpy()
    sigma = np.std(resid) or 1.0
    hist["residual_z"] = resid / sigma
    hist["shortfall_pct"] = np.where(
        hist["predicted_kwh"] > 0, -hist["residual"] / hist["predicted_kwh"] * 100, 0.0
    )

    feats = hist[["residual", "shortfall_pct", "month", "panel_age_years"]].to_numpy()
    if len(hist) >= 12:
        iso = IsolationForest(
            contamination=min(0.4, contamination), random_state=RANDOM_STATE, n_estimators=300
        )
        hist["iso_flag"] = iso.fit_predict(feats) == -1
        hist["iso_score"] = -iso.score_samples(feats)
    else:
        hist["iso_flag"] = False
        hist["iso_score"] = 0.0

    # The commissioning month is a partial month by construction -- the array
    # was energised part-way through it -- so a shortfall there is an artefact
    # of the calendar, not a fault.
    hist["is_ramp_up"] = hist["period"] == hist["period"].min()

    hist["z_flag"] = (hist["residual_z"] < -1.5) & ~hist["is_ramp_up"]
    # A statistical flag is not yet an engineering problem: require a shortfall
    # large enough to be worth climbing onto the roof for.
    hist["is_anomaly"] = hist["z_flag"] & hist["iso_flag"] & (hist["shortfall_pct"] >= 10)
    # An unusually *good* month is still an outlier to the Isolation Forest, but
    # it is not a maintenance signal - label it separately so the UI does not
    # cry wolf over a bright dry spell.
    underperforming = hist["shortfall_pct"] > 0
    hist["severity"] = np.select(
        [
            hist["is_ramp_up"],
            hist["is_anomaly"],
            (hist["z_flag"] | hist["iso_flag"]) & underperforming,
            hist["iso_flag"] & ~underperforming,
        ],
        ["ramp-up", "high", "watch", "above normal"],
        default="ok",
    )
    return hist[
        [
            "period", "year", "month", "solar_kwh", "predicted_kwh", "residual",
            "residual_z", "shortfall_pct", "z_flag", "iso_flag", "is_anomaly",
            "is_ramp_up", "severity",
        ]
    ]
