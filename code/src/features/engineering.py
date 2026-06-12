"""Phase 1 causal feature engineering.

Every feature at time t uses only samples at or before t (PROJECT_PLAN §4.5).
The per-specimen scalars loading rate and R₀ are treated as *known calibration
constants* (the crosshead speed is a machine setting; R₀ is measured before
loading), so they are passed in rather than re-estimated from the live stream.
This keeps the streaming features strictly causal — verified in
``tests/test_causality.py``.

Derivatives use a one-sided (causal) Savitzky-Golay operator: a degree-2
polynomial is least-squares fit to the trailing window and its value/derivative
are evaluated at the most recent sample. The standard centred SavGol would peek
into the future, so it is not used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FS = 12.0
DT = 1.0 / FS

# Feature window sizes, expressed in mm of deflection (rate-invariant, §4.3).
SMOOTH_MM = 0.3  # causal-SavGol window for derivatives
ROLL_MM = (1.0, 3.0, 5.0)  # rolling mean/std windows
JUMP_MED_MM = 3.0  # window for the running median/MAD baseline
JUMP_K = 1.0  # threshold = running median + K * robust std
JUMP_REFRACTORY_MM = 0.5  # debounce between counted jump onsets
COUNT_MM = 3.0  # window for "step events in last 3 mm"

EPS = 1e-9

FEATURE_COLUMNS = [
    "fcr",
    "dfcr_dD",
    "d2fcr_dD2",
    "roll_mean_1mm",
    "roll_std_1mm",
    "roll_mean_3mm",
    "roll_std_3mm",
    "roll_mean_5mm",
    "roll_std_5mm",
    "cum_fcr",
    "defl_since_first_jump",
    "n_jumps_last_3mm",
    "loading_rate",
    "log_R0",
]


def _samples_per_mm(loading_rate: float) -> float:
    """Samples per mm of deflection from the known crosshead speed."""
    return 1.0 / max(loading_rate * DT, EPS)


def _win_samples(mm: float, spm: float) -> int:
    return max(2, int(round(mm * spm)))


def _causal_savgol_coeffs(m: int, deriv: int, poly: int = 2) -> np.ndarray:
    """Coefficients c so that c·y[t-m+1:t+1] = (value | d/dindex) at the last sample.

    Offsets are x = [-(m-1), ..., 0]; the current sample sits at x = 0.
    """
    poly = min(poly, m - 1)
    x = np.arange(-(m - 1), 1, dtype=float)
    A = np.vander(x, poly + 1, increasing=True)  # columns 1, x, x^2, ...
    pinv = np.linalg.pinv(A)  # (poly+1, m)
    return pinv[deriv]


def _apply_causal(y: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Trailing correlation; left-pad with y[0] so only past samples are used."""
    m = len(coeffs)
    yp = np.concatenate([np.full(m - 1, y[0]), y])
    return np.correlate(yp, coeffs, mode="valid")


def _causal_derivative(y: np.ndarray, dD_di: float, m: int) -> np.ndarray:
    """d y / d(deflection), causal.

    dy/d(deflection) = (dy/dindex) / (dD/dindex). The deflection-rate-per-index
    is the *known constant* crosshead rate (dD/di = loading_rate * DT), not the
    noisy measured value — dividing by the measured rate explodes wherever the
    stroke signal momentarily plateaus (§4.3).
    """
    dydi = _apply_causal(y, _causal_savgol_coeffs(m, deriv=1))
    return dydi / dD_di


def _detect_jumps(dfcr_dD: pd.Series, spm: float) -> np.ndarray:
    """Boolean onset mask of causal 'significant jump' events on the derivative.

    A jump is the rising edge where the derivative first exceeds its trailing
    running median + K robust-std, with a deflection refractory to avoid
    counting one event many times.
    """
    w_med = _win_samples(JUMP_MED_MM, spm)
    med = dfcr_dD.rolling(w_med, min_periods=5).median()
    mad = (dfcr_dD - med).abs().rolling(w_med, min_periods=5).median()
    robust_std = 1.4826 * mad
    over = (dfcr_dD > (med + JUMP_K * robust_std)).fillna(False).to_numpy()

    refractory = _win_samples(JUMP_REFRACTORY_MM, spm)
    onsets = np.zeros(len(over), dtype=bool)
    last = -refractory - 1
    prev = False
    for i, flag in enumerate(over):
        if flag and not prev and (i - last) > refractory:
            onsets[i] = True
            last = i
        prev = flag
    return onsets


def build_specimen_features(
    df: pd.DataFrame, loading_rate: float, r0_ohm: float
) -> pd.DataFrame:
    """Compute the causal feature columns for one specimen (sorted by t)."""
    df = df.reset_index(drop=True)
    fcr = df["FCR_pct"].to_numpy()
    defl = df["deflection_mm"].to_numpy()
    spm = _samples_per_mm(loading_rate)

    out = pd.DataFrame(index=df.index)
    out["fcr"] = fcr

    # Causal derivatives wrt deflection (normalised by the known crosshead rate).
    m = _win_samples(SMOOTH_MM, spm)
    dD_di = loading_rate * DT
    dfcr_dD = _causal_derivative(fcr, dD_di, m)
    out["dfcr_dD"] = dfcr_dD
    out["d2fcr_dD2"] = _causal_derivative(dfcr_dD, dD_di, m)

    # Trailing rolling mean/std over deflection windows.
    s_fcr = pd.Series(fcr)
    for mm in ROLL_MM:
        w = _win_samples(mm, spm)
        tag = f"{int(mm)}mm"
        out[f"roll_mean_{tag}"] = s_fcr.rolling(w, min_periods=1).mean().to_numpy()
        out[f"roll_std_{tag}"] = (
            s_fcr.rolling(w, min_periods=2).std().fillna(0.0).to_numpy()
        )

    # Cumulative ΔR/R₀ exposure: causal trapezoidal integral over deflection.
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (fcr[1:] + fcr[:-1]) * np.diff(defl))])
    out["cum_fcr"] = cum

    # Jump-based features (causal detector).
    onsets = _detect_jumps(pd.Series(dfcr_dD), spm)
    defl_since = np.zeros(len(defl))
    first = np.argmax(onsets) if onsets.any() else -1
    if first >= 0:
        defl_since[first:] = defl[first:] - defl[first]
    out["defl_since_first_jump"] = defl_since
    w_cnt = _win_samples(COUNT_MM, spm)
    out["n_jumps_last_3mm"] = (
        pd.Series(onsets.astype(float)).rolling(w_cnt, min_periods=1).sum().to_numpy()
    )

    # Static (known) features broadcast across the window.
    out["loading_rate"] = loading_rate
    out["log_R0"] = float(np.log(r0_ohm))
    return out


def known_constants(df: pd.DataFrame) -> tuple[float, float]:
    """Return (loading_rate, R₀) for a specimen — both known before/at t=0."""
    loading_rate = float(df["loading_rate_mm_per_s"].iloc[0])
    r0 = float(df.loc[df["t"] <= df["t"].iloc[0] + 1.0, "R_ohm"].median())
    return loading_rate, r0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal features for every specimen and return the stacked frame."""
    parts = []
    for name, g in df.groupby("specimen", sort=True):
        lr, r0 = known_constants(g)
        feats = build_specimen_features(g, lr, r0)
        feats.insert(0, "specimen", name)
        feats.insert(1, "t", g["t"].to_numpy())
        parts.append(feats)
    return pd.concat(parts, ignore_index=True)
