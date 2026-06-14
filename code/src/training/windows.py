"""Causal window construction for the sequence models.

For each end-sample i, the window is the trailing span of ~W mm of deflection,
decimated to a fixed length T of evenly-spaced samples (most recent sample
always included). Early samples are left-padded by clamping indices at 0. Only
past samples are ever used, so the windows are causal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FS = 12.0


@dataclass
class WindowSet:
    X: np.ndarray  # (N, T, F) raw feature windows
    dtf: np.ndarray
    load: np.ndarray
    defl: np.ndarray
    stage: np.ndarray
    t: np.ndarray  # end-sample time of each window
    specimen: np.ndarray


def _span_samples(rate: float, w_mm: float, t: int) -> int:
    spm = 1.0 / (rate / FS)
    return max(t, int(round(w_mm * spm)))


def build_specimen_windows(
    g: pd.DataFrame, feat_cols: list[str], w_mm: float, t_len: int, stride: int
) -> WindowSet:
    feats = g[feat_cols].to_numpy(dtype=np.float32)
    n = len(feats)
    rate = float(g["loading_rate"].iloc[0])
    span = _span_samples(rate, w_mm, t_len)

    ends = np.arange(0, n, stride)
    # source indices: T evenly-spaced points in [end-span+1, end], clamped >= 0.
    offs = np.linspace(-(span - 1), 0, t_len)
    src = np.clip(ends[:, None] + offs[None, :].round().astype(int), 0, n - 1)
    X = feats[src]  # (N, T, F)

    return WindowSet(
        X=X,
        dtf=g["DTF_mm"].to_numpy(np.float32)[ends],
        load=g["load_kN"].to_numpy(np.float32)[ends],
        defl=g["deflection_mm"].to_numpy(np.float32)[ends],
        stage=g["stage"].to_numpy(np.int64)[ends],
        t=g["t"].to_numpy(np.float64)[ends],
        specimen=g["specimen"].to_numpy()[ends],
    )


def build_windows(
    df: pd.DataFrame,
    specimens: list[str],
    feat_cols: list[str],
    w_mm: float,
    t_len: int,
    stride: int,
) -> WindowSet:
    parts = [
        build_specimen_windows(df[df["specimen"] == s], feat_cols, w_mm, t_len, stride)
        for s in specimens
    ]
    return WindowSet(
        X=np.concatenate([p.X for p in parts]),
        dtf=np.concatenate([p.dtf for p in parts]),
        load=np.concatenate([p.load for p in parts]),
        defl=np.concatenate([p.defl for p in parts]),
        stage=np.concatenate([p.stage for p in parts]),
        t=np.concatenate([p.t for p in parts]),
        specimen=np.concatenate([p.specimen for p in parts]),
    )
