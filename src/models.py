"""The machine-learning part: compare 4 simple models and keep the best.

The 4 candidates
----------------
Baseline (mean)    always predicts the average bill. The score to beat.
Linear Regression  bill = w1*x1 + w2*x2 + ... + b, weights chosen to make the
                   squared errors as small as possible.
Ridge (L2)         Linear Regression + a penalty on (weight)^2. Keeps all
                   weights small, so the model is steadier on new months.
Lasso (L1)         Linear Regression + a penalty on |weight|. Can push
                   useless weights to exactly 0 (it "selects" features).

How the winner is chosen
------------------------
TimeSeriesSplit cross-validation: train on older months, test on the next
block of newer months, repeat 5 times, average the error (MAE). The model
with the lowest average error wins. We never shuffle months, because then
the model could "see the future" while being tested on the past.

Why only linear models?  They can follow a trend into future years
(tree models cannot - they repeat their last value), they are easy to
explain, and with under 100 rows of data simple models are safer.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CV_SPLITS
from .features import CALENDAR_FEATURES, TARIFF_FEATURES, add_features


def candidate_models():
    """The 4 models. StandardScaler puts every feature on the same scale
    (mean 0, spread 1) so the Ridge/Lasso penalty treats them all fairly."""
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge (L2)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso (L1)": make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=10_000)),
    }


def compare_models(X, y):
    """Score every model with TimeSeriesSplit. Returns a table, best first."""
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS)
    rows = []
    for name, model in candidate_models().items():
        maes, mapes, r2s = [], [], []
        for train_idx, test_idx in splitter.split(X):
            m = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = m.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx]
            maes.append(mean_absolute_error(actual, pred))
            mapes.append(np.mean(np.abs((actual - pred) / actual)) * 100)
            r2s.append(r2_score(actual, pred))
        rows.append({"model": name, "MAE (Rs)": np.mean(maes),
                     "MAPE (%)": np.mean(mapes), "R2": np.mean(r2s)})
    return pd.DataFrame(rows).sort_values("MAE (Rs)").reset_index(drop=True)


def train_best(X, y):
    """Compare, pick the lowest-MAE model, retrain it on ALL rows."""
    board = compare_models(X, y)
    best_name = board.loc[0, "model"]
    model = clone(candidate_models()[best_name]).fit(X, y)
    board["chosen"] = board["model"] == best_name
    return best_name, model, board


class TariffModel:
    """Model A: learns the electricity price formula from the bills.

    Trained on: units bought + date  ->  bill.
    Then asked: "what would the bill be if the house had bought EVERY unit
    it used (bought + solar)?"  That is the bill without panels.
    """

    def fit(self, train_df, start_period):
        self.start = start_period
        X = add_features(train_df, start_period)[TARIFF_FEATURES]
        self.name, self.model, self.leaderboard = train_best(X, train_df["bill_amount"])
        return self

    def bill_without_solar(self, df):
        X = add_features(df, self.start, units_column="total_use_kwh")[TARIFF_FEATURES]
        return np.maximum(self.model.predict(X), 0)


class PreSolarModel:
    """Model B: learns the bill trend from months BEFORE the panels only.

    Trained on: date only  ->  bill  (pre-solar months).
    Then asked: "continue that trend into the solar years."
    It never sees any solar data, so it is an independent check on Model A.
    """

    def fit(self, pre_solar_df, start_period):
        self.start = start_period
        X = add_features(pre_solar_df, start_period)[CALENDAR_FEATURES]
        self.name, self.model, self.leaderboard = train_best(X, pre_solar_df["bill_amount"])
        return self

    def bill_without_solar(self, df):
        X = add_features(df, self.start)[CALENDAR_FEATURES]
        return np.maximum(self.model.predict(X), 0)
