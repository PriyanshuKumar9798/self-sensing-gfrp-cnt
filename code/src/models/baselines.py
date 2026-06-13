"""Phase 2 baseline models (PROJECT_PLAN §5 Phase 2 step 2-3).

Two regression baselines per target — a linear-regression floor on a hand-picked
feature subset, and a LightGBM ensemble whose hyper-parameters are tuned by inner
LOSO over the three training specimens with early stopping. Plus a stage
classifier (LightGBM multi-class) and a logistic-regression floor.

All models follow the ``Model`` protocol in ``src/evaluation/loso.py``:
``fit(X, y, groups)`` then ``predict(X)``.
"""

from __future__ import annotations

from itertools import product

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.loso import inner_loso_splits

SEED = 42

# Hand-picked, weakly-collinear subset for the linear floor.
LINEAR_FEATURES_IDX: list[int] | None = None  # set by the runner if subsetting

# Small inner-CV grid for LightGBM.
LGBM_GRID = {
    "num_leaves": [15, 31],
    "learning_rate": [0.05, 0.1],
    "min_data_in_leaf": [20, 50],
    "lambda_l2": [0.0, 1.0],
}
LGBM_FIXED = dict(
    n_estimators=500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=1,
)


def _grid_combos() -> list[dict]:
    keys = list(LGBM_GRID)
    return [dict(zip(keys, vals)) for vals in product(*LGBM_GRID.values())]


class LinearBaseline:
    """StandardScaler + ordinary least squares (regression floor)."""

    def __init__(self, feature_idx: list[int] | None = None):
        self.feature_idx = feature_idx
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def _sub(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.feature_idx] if self.feature_idx is not None else X

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "LinearBaseline":
        Xs = self.scaler.fit_transform(self._sub(X))
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(self._sub(X)))


class LogisticBaseline:
    """StandardScaler + multinomial logistic regression (stage floor)."""

    def __init__(self, feature_idx: list[int] | None = None):
        self.feature_idx = feature_idx
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, C=1.0)

    def _sub(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.feature_idx] if self.feature_idx is not None else X

    def fit(self, X, y, groups) -> "LogisticBaseline":
        self.model.fit(self.scaler.fit_transform(self._sub(X)), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(self._sub(X)))


class _LGBMTuned:
    """Shared inner-LOSO tuning logic for the LightGBM models."""

    estimator_cls = lgb.LGBMRegressor
    eval_metric = "l1"

    def __init__(self, **extra):
        self.extra = extra
        self.model = None

    def _tune(self, X, y, groups):
        splits = inner_loso_splits(groups)
        best, best_score, best_iter = None, np.inf, LGBM_FIXED["n_estimators"]
        for combo in _grid_combos():
            scores, iters = [], []
            for tr, va in splits:
                est = self.estimator_cls(**LGBM_FIXED, **combo, **self.extra)
                est.fit(
                    X[tr],
                    y[tr],
                    eval_set=[(X[va], y[va])],
                    eval_metric=self.eval_metric,
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                scores.append(est.best_score_["valid_0"][self.eval_metric])
                iters.append(est.best_iteration_ or LGBM_FIXED["n_estimators"])
            mean = float(np.mean(scores))
            if mean < best_score:
                best_score, best, best_iter = mean, combo, int(np.mean(iters))
        return best, max(best_iter, 20)

    def fit(self, X, y, groups):
        combo, n_iter = self._tune(X, y, groups)
        params = {**LGBM_FIXED, **combo, **self.extra}
        params["n_estimators"] = n_iter
        self.model = self.estimator_cls(**params)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LGBMRegressorBaseline(_LGBMTuned):
    estimator_cls = lgb.LGBMRegressor
    eval_metric = "l1"


class LGBMClassifierBaseline(_LGBMTuned):
    estimator_cls = lgb.LGBMClassifier
    eval_metric = "multi_logloss"

    def __init__(self, num_class: int = 4, **extra):
        super().__init__(objective="multiclass", num_class=num_class, **extra)
