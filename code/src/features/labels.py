"""Phase 1 labels (PROJECT_PLAN §5 Phase 1 step 3).

DTF (primary) and TTF are deflection/time remaining until the load peak. The
four-class ``stage`` label uses deflection-based boundaries (fractions of
D_peak = deflection at the load peak), so it is rate-invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["DTF_mm", "TTF_s", "load_kN", "deflection_mm", "stage"]
STAGE_NAMES = {0: "early", 1: "elastic-rising", 2: "pre-failure", 3: "post-failure"}

# Stage boundaries as fractions of D_peak.
EARLY_FRAC = 0.1
PRE_FAILURE_FRAC = 0.8


def _stage(defl: np.ndarray, d_peak: float) -> np.ndarray:
    s = np.full(len(defl), 1, dtype=int)  # elastic-rising
    s[defl < EARLY_FRAC * d_peak] = 0
    s[(defl >= PRE_FAILURE_FRAC * d_peak) & (defl <= d_peak)] = 2
    s[defl > d_peak] = 3
    return s


def compute_specimen_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Labels for one specimen. D_peak is the deflection at the load peak."""
    df = df.reset_index(drop=True)
    i_peak = int(df["load_kN"].to_numpy().argmax())
    t_peak = float(df["t"].iloc[i_peak])
    d_peak = float(df["deflection_mm"].iloc[i_peak])

    out = pd.DataFrame(index=df.index)
    out["DTF_mm"] = (d_peak - df["deflection_mm"]).clip(lower=0.0).to_numpy()
    out["TTF_s"] = (t_peak - df["t"]).clip(lower=0.0).to_numpy()
    out["load_kN"] = df["load_kN"].to_numpy()
    out["deflection_mm"] = df["deflection_mm"].to_numpy()
    out["stage"] = _stage(df["deflection_mm"].to_numpy(), d_peak)
    return out


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Labels for every specimen, stacked."""
    parts = []
    for name, g in df.groupby("specimen", sort=True):
        lab = compute_specimen_labels(g)
        lab.insert(0, "specimen", name)
        lab.insert(1, "t", g["t"].to_numpy())
        parts.append(lab)
    return pd.concat(parts, ignore_index=True)
