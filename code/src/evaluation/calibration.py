"""Phase 4 calibration (PROJECT_PLAN §5 Phase 4 steps 1-2).

DTF is re-cast as quantile regression: three LightGBM models with the pinball
(quantile) objective at q = 0.1 / 0.5 / 0.9, trained under LOSO. The 0.9 head is
the conservative upper bound used by the alarm. As a second uncertainty estimate
we also train a bootstrap ensemble (resamples of the training pool) and read its
spread. Both are evaluated for calibration (empirical coverage vs nominal).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.loso import loso_predict

QUANTILES = (0.1, 0.5, 0.9)
QCOLS = {0.1: "q10", 0.5: "q50", 0.9: "q90"}

# Fixed (untuned) LightGBM params for the quantile / bootstrap heads — kept
# modest to avoid over-fitting the tiny pool; documented in §10.
QPARAMS = dict(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=300,
    min_data_in_leaf=50,
    lambda_l2=1.0,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    n_jobs=1,
)


class LGBMQuantile:
    """LightGBM regressor with the pinball loss at a single quantile."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.model = None

    def fit(self, X, y, groups) -> "LGBMQuantile":
        self.model = lgb.LGBMRegressor(
            objective="quantile", alpha=self.alpha, **QPARAMS
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def quantile_loso(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """LOSO quantile predictions for DTF. Columns: specimen, t, dtf_true, q10/q50/q90."""
    preds = {
        q: loso_predict(df, feat_cols, "DTF_mm", lambda q=q: LGBMQuantile(q))
        for q in QUANTILES
    }
    m = preds[0.5][["specimen", "t", "y_true"]].rename(columns={"y_true": "dtf_true"})
    for q in QUANTILES:
        m = m.merge(
            preds[q][["specimen", "t", "y_pred"]].rename(columns={"y_pred": QCOLS[q]}),
            on=["specimen", "t"],
        )
    # Enforce non-crossing quantiles by sorting the three predictions per row.
    qc = list(QCOLS.values())
    m[qc] = np.sort(m[qc].to_numpy(), axis=1)
    return m


def bootstrap_loso(
    df: pd.DataFrame, feat_cols: list[str], n_boot: int = 10, seed: int = 42
) -> pd.DataFrame:
    """LOSO bootstrap ensemble (L1 LightGBM). Columns: specimen, t, boot_mean/lo/hi."""
    rng = np.random.default_rng(seed)
    parts = []
    for test in sorted(df["specimen"].unique()):
        tr = df[df["specimen"] != test]
        te = df[df["specimen"] == test]
        Xtr, ytr = tr[feat_cols].to_numpy(), tr["DTF_mm"].to_numpy()
        Xte = te[feat_cols].to_numpy()
        preds = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(Xtr), len(Xtr))
            mdl = lgb.LGBMRegressor(objective="regression_l1", **QPARAMS)
            mdl.fit(Xtr[idx], ytr[idx])
            preds.append(mdl.predict(Xte))
        p = np.asarray(preds)
        parts.append(
            pd.DataFrame(
                {
                    "specimen": test,
                    "t": te["t"].to_numpy(),
                    "boot_mean": p.mean(0),
                    "boot_lo": np.percentile(p, 10, axis=0),
                    "boot_hi": np.percentile(p, 90, axis=0),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def empirical_coverage(m: pd.DataFrame) -> dict[float, float]:
    """Fraction of held-out samples with true DTF ≤ the q-th predicted quantile."""
    return {q: float((m["dtf_true"] <= m[QCOLS[q]]).mean()) for q in QUANTILES}


def conformal_upper_loso(
    df: pd.DataFrame, feat_cols: list[str], alpha: float = 0.1
) -> pd.DataFrame:
    """Cross-conformal upper bound on DTF (PROJECT_PLAN §4.1.5).

    The raw quantile heads collapse under LOSO because DTF is nearly
    deterministic within a specimen; the dominant uncertainty is the
    cross-specimen D_peak. For each outer fold we estimate, by inner LOSO over
    the three training specimens, how far the (1-α) head is exceeded on an
    *unseen* specimen, and add that (1-α) exceedance quantile δ to the outer
    prediction. The result targets nominal coverage on the held-out specimen.
    Columns: specimen, t, q90_cal.
    """
    upper_q = 1.0 - alpha
    specs = sorted(df["specimen"].unique())
    parts = []
    for test in specs:
        train = [s for s in specs if s != test]
        scores = []
        for val in train:
            t2 = [s for s in train if s != val]
            tr = df[df["specimen"].isin(t2)]
            va = df[df["specimen"] == val]
            mdl = LGBMQuantile(upper_q).fit(
                tr[feat_cols].to_numpy(), tr["DTF_mm"].to_numpy(), None
            )
            scores.append(
                va["DTF_mm"].to_numpy() - mdl.predict(va[feat_cols].to_numpy())
            )
        delta = float(np.quantile(np.concatenate(scores), upper_q))

        trf = df[df["specimen"].isin(train)]
        te = df[df["specimen"] == test]
        full = LGBMQuantile(upper_q).fit(
            trf[feat_cols].to_numpy(), trf["DTF_mm"].to_numpy(), None
        )
        upper = np.clip(full.predict(te[feat_cols].to_numpy()) + delta, 0.0, None)
        parts.append(
            pd.DataFrame({"specimen": test, "t": te["t"].to_numpy(), "q90_cal": upper})
        )
    return pd.concat(parts, ignore_index=True)
