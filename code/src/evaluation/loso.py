"""Leave-one-specimen-out (LOSO) harness (PROJECT_PLAN §5 Phase 2 step 1, D5).

Train on three specimens, test on the held-out fourth, rotate. Models that need
hyper-parameter tuning use an *inner* LOSO over the three training specimens
(never a random within-specimen split — adjacent samples are correlated and
would leak). The model object owns its own tuning; the harness only splits at
the specimen level and collects per-fold predictions.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    """A model fits on a feature matrix with specimen groups and predicts."""

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "Model": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


ModelFactory = Callable[[], Model]


def loso_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    factory: ModelFactory,
) -> pd.DataFrame:
    """Return per-sample predictions on each held-out specimen for one target.

    Output columns: ``specimen``, ``t``, ``y_true``, ``y_pred`` (concatenated
    across the four held-out folds).
    """
    specimens = sorted(df["specimen"].unique())
    parts = []
    for test in specimens:
        tr = df[df["specimen"] != test]
        te = df[df["specimen"] == test]
        model = factory()
        model.fit(
            tr[feature_cols].to_numpy(),
            tr[target_col].to_numpy(),
            tr["specimen"].to_numpy(),
        )
        pred = model.predict(te[feature_cols].to_numpy())
        parts.append(
            pd.DataFrame(
                {
                    "specimen": test,
                    "t": te["t"].to_numpy(),
                    "y_true": te[target_col].to_numpy(),
                    "y_pred": np.asarray(pred),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def inner_loso_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Index splits for inner LOSO over the training specimens."""
    specs = sorted(np.unique(groups))
    splits = []
    for val in specs:
        tr_idx = np.where(groups != val)[0]
        va_idx = np.where(groups == val)[0]
        splits.append((tr_idx, va_idx))
    return splits
