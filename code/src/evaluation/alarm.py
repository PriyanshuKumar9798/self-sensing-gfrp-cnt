"""Phase 4 alarm policy (PROJECT_PLAN §5 Phase 4 steps 3-5).

The alarm fires when the conservative DTF estimate (the 0.9 upper bound) stays
below τ_alarm for N consecutive samples. We sweep (τ_alarm, N) and pick the
operating point that maximises median lead time subject to recall ≥ 0.75 and
precision ≥ 0.8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import metrics as M

TAU_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
N_SWEEP = [3, 6, 12]
RECALL_MIN = 0.75
PRECISION_MIN = 0.8


def sweep_operating(
    pred: pd.DataFrame,
    info: dict,
    signal_col: str,
    taus=TAU_SWEEP,
    ns=N_SWEEP,
) -> pd.DataFrame:
    """Operating curve over (τ, N) using ``signal_col`` as the alarm trigger."""
    n_spec = pred["specimen"].nunique()
    rows = []
    for tau in taus:
        for n in ns:
            alarms = [
                M.classify_alarm(
                    g["t"].to_numpy(),
                    g[signal_col].to_numpy(),
                    g["dtf_true"].to_numpy(),
                    info[s]["t_failure"],
                    tau,
                    n,
                )
                for s, g in pred.groupby("specimen")
            ]
            sc = M.alarm_scores(alarms)
            sc["false_alarms_per_test"] = sc["fp"] / n_spec
            rows.append({"tau": tau, "N": n, **sc})
    return pd.DataFrame(rows)


def pick_operating_point(
    curve: pd.DataFrame, recall_min=RECALL_MIN, precision_min=PRECISION_MIN
) -> pd.Series | None:
    """Max median lead time subject to the recall/precision floor; None if infeasible."""
    valid = curve[
        (curve["recall"] >= recall_min) & (curve["precision"] >= precision_min)
    ]
    valid = valid.dropna(subset=["median_lead_s"])
    if valid.empty:
        return None
    return valid.sort_values("median_lead_s", ascending=False).iloc[0]


def per_specimen_timing(
    pred: pd.DataFrame, info: dict, signal_col: str, tau: float, n: int
) -> pd.DataFrame:
    """Alarm outcome and timing for each specimen at one operating point."""
    rows = []
    for s, g in pred.groupby("specimen"):
        t = g["t"].to_numpy()
        idx = M.first_alarm_index(g[signal_col].to_numpy(), tau, n)
        tf = info[s]["t_failure"]
        if idx is None or t[idx] > tf:
            rows.append(
                {
                    "specimen": s,
                    "alarm_t_s": np.nan,
                    "t_failure_s": tf,
                    "lead_s": np.nan,
                    "outcome": "FN",
                }
            )
        else:
            true_dtf = g["dtf_true"].to_numpy()[idx]
            outcome = "TP" if true_dtf <= tau else "FP"
            rows.append(
                {
                    "specimen": s,
                    "alarm_t_s": float(t[idx]),
                    "t_failure_s": tf,
                    "lead_s": float(tf - t[idx]),
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)
