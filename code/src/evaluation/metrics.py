"""Phase 2 metrics (PROJECT_PLAN §5 Phase 2 step 4).

Regression error for DTF / load / deflection (overall and per stage), TTF error
derived from the loading rate, stage-classification quality, and the alarm
precision / recall / F1 with the true-positive lead-time distribution.

Alarm convention (recorded in §10): an alarm fires at the first sample where the
predicted DTF stays below τ_alarm for N consecutive samples. Per held-out
specimen it is classified as
  * true positive  — alarm fires before failure AND the *true* DTF at that
    instant is ≤ τ_alarm (the "within τ mm of failure" claim is actually true);
  * false positive — alarm fires before failure but the true DTF is still > τ
    (premature);
  * false negative — no alarm fires before failure.
Lead time of a true positive is (t_failure − t_alarm) in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

STAGES = (0, 1, 2, 3)


# --- regression -------------------------------------------------------------
def mae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)))


def medae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.median(np.abs(true - pred)))


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def ttf_mae(dtf_pred: np.ndarray, ttf_true: np.ndarray, loading_rate: float) -> float:
    """TTF error in seconds, converting predicted DTF via the known rate."""
    ttf_pred = dtf_pred / loading_rate
    return mae(ttf_true, ttf_pred)


def per_stage_rmse(
    true: np.ndarray, pred: np.ndarray, stage: np.ndarray
) -> dict[int, float]:
    out: dict[int, float] = {}
    for s in STAGES:
        m = stage == s
        out[s] = rmse(true[m], pred[m]) if m.any() else float("nan")
    return out


# --- stage classification ---------------------------------------------------
def stage_balanced_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(true, pred))


def stage_confusion(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(true, pred, labels=list(STAGES))


# --- alarm ------------------------------------------------------------------
def first_alarm_index(dtf_pred: np.ndarray, tau: float, n: int) -> int | None:
    """First sample index where predicted DTF stays < τ for n consecutive samples."""
    below = (dtf_pred < tau).astype(int)
    if n <= 1:
        hits = np.where(below == 1)[0]
        return int(hits[0]) if hits.size else None
    run = np.convolve(below, np.ones(n, dtype=int), mode="valid")
    hits = np.where(run == n)[0]
    return int(hits[0] + n - 1) if hits.size else None


@dataclass
class SpecimenAlarm:
    """Outcome of the alarm rule on one held-out specimen."""

    outcome: str  # "TP", "FP", or "FN"
    lead_time_s: float  # finite only for TP


def classify_alarm(
    t: np.ndarray,
    dtf_pred: np.ndarray,
    dtf_true: np.ndarray,
    t_failure: float,
    tau: float,
    n: int,
) -> SpecimenAlarm:
    idx = first_alarm_index(dtf_pred, tau, n)
    if idx is None or t[idx] > t_failure:
        return SpecimenAlarm("FN", float("nan"))
    if dtf_true[idx] <= tau:
        return SpecimenAlarm("TP", float(t_failure - t[idx]))
    return SpecimenAlarm("FP", float("nan"))


def alarm_scores(alarms: list[SpecimenAlarm]) -> dict[str, float]:
    """Precision / recall / F1 / median lead time over a set of specimens."""
    tp = sum(a.outcome == "TP" for a in alarms)
    fp = sum(a.outcome == "FP" for a in alarms)
    fn = sum(a.outcome == "FN" for a in alarms)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if precision and recall and not np.isnan(precision) and not np.isnan(recall):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    leads = [a.lead_time_s for a in alarms if a.outcome == "TP"]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_lead_s": float(np.median(leads)) if leads else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
