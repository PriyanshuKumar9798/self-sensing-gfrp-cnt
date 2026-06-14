"""Phase 3 augmentation operators (PROJECT_PLAN §4.2).

Each operator acts on a single causal window ``win`` of shape (T, F) in the
*original* feature units (augment first, standardise second). The five operators
are: random window crops (realised by the windowing itself, §4.2.1), mild
time-warping, additive ΔR/R₀ noise, baseline-R perturbation, and specimen mixup
(applied at batch level, late in training).

Channel groups: ``LEVEL`` are ΔR/R₀-level channels (shift with the baseline),
``SCALE`` are channels that scale like a resistance ratio (derivatives, rolling
std, cumulative). The remaining channels (jump distance/count, static) are left
untouched by the resistance-domain operators.
"""

from __future__ import annotations

import numpy as np

from src.features.engineering import FEATURE_COLUMNS

LEVEL = ["fcr", "roll_mean_1mm", "roll_mean_3mm", "roll_mean_5mm"]
SCALE = [
    "dfcr_dD",
    "d2fcr_dD2",
    "roll_std_1mm",
    "roll_std_3mm",
    "roll_std_5mm",
    "cum_fcr",
]
LEVEL_IDX = [FEATURE_COLUMNS.index(c) for c in LEVEL]
SCALE_IDX = [FEATURE_COLUMNS.index(c) for c in SCALE]


def time_warp(win: np.ndarray, rng: np.random.Generator, lo=0.9, hi=1.1) -> np.ndarray:
    """Resample the window onto a stretched/compressed time axis (±10%).

    Preserves the *shape* of the response and the most-recent sample (the label
    anchor); does not preserve absolute timing — the right inductive bias given
    that loading rates differ between specimens. Goes no further than ±10% so the
    warped curve stays physically plausible.
    """
    t = win.shape[0]
    factor = rng.uniform(lo, hi)
    src = np.arange(t, dtype=float)
    warped = np.clip(src * factor + (t - 1) * (1 - factor), 0, t - 1)  # end anchored
    out = np.empty_like(win)
    for f in range(win.shape[1]):
        out[:, f] = np.interp(warped, src, win[:, f])
    return out


def add_noise(win: np.ndarray, rng: np.random.Generator, lo=0.1, hi=0.3) -> np.ndarray:
    """Add zero-mean Gaussian noise (σ ∈ [0.1, 0.3] % ΔR/R₀) to the level channels.

    Simulates the source-meter measurement noise a deployed system would face
    (level calibrated from the flat early signal). Labels are never touched.
    """
    sigma = rng.uniform(lo, hi)
    out = win.copy()
    out[:, LEVEL_IDX] += rng.normal(0.0, sigma, size=(win.shape[0], len(LEVEL_IDX)))
    return out


def baseline_perturb(
    win: np.ndarray, rng: np.random.Generator, lo=-0.2, hi=0.2
) -> np.ndarray:
    """Recompute ΔR/R₀ as if R₀ were perturbed by δ ∈ [-0.2, 0.2].

    With R₀' = R₀(1+δ): level' = ((1 + level/100)/(1+δ) − 1)·100, and the
    ratio-like SCALE channels are multiplied by 1/(1+δ). Tells the model the
    *shape* of the curve is meaningful while a 20 % baseline shift is not —
    directly addressing generalisation across the 8× baseline spread. Preserves
    curve shape; does not preserve absolute ΔR/R₀ magnitude.
    """
    delta = rng.uniform(lo, hi)
    k = 1.0 / (1.0 + delta)
    out = win.copy()
    out[:, LEVEL_IDX] = ((1.0 + out[:, LEVEL_IDX] / 100.0) * k - 1.0) * 100.0
    out[:, SCALE_IDX] *= k
    return out


def augment(
    win: np.ndarray, rng: np.random.Generator, p_warp=0.5, p_baseline=0.3
) -> np.ndarray:
    """Per-sample training pipeline: (warp?) -> noise -> (baseline?) (§5 Phase 3 step 6)."""
    if rng.random() < p_warp:
        win = time_warp(win, rng)
    win = add_noise(win, rng)
    if rng.random() < p_baseline:
        win = baseline_perturb(win, rng)
    return win


def time_warp_batch(
    x: np.ndarray, rng: np.random.Generator, lo=0.9, hi=1.1
) -> np.ndarray:
    """Vectorised ±10% time-warp with one factor per batch (end anchored)."""
    t = x.shape[1]
    factor = rng.uniform(lo, hi)
    src = np.arange(t, dtype=float)
    warped = np.clip(src * factor + (t - 1) * (1 - factor), 0, t - 1)
    lo_i = np.floor(warped).astype(int)
    hi_i = np.minimum(lo_i + 1, t - 1)
    frac = (warped - lo_i)[None, :, None].astype(np.float32)
    return x[:, lo_i, :] * (1 - frac) + x[:, hi_i, :] * frac


def add_noise_batch(
    x: np.ndarray, rng: np.random.Generator, lo=0.1, hi=0.3
) -> np.ndarray:
    sigma = rng.uniform(lo, hi, size=(x.shape[0], 1, 1)).astype(np.float32)
    out = x.copy()
    noise = rng.normal(0.0, 1.0, size=(x.shape[0], x.shape[1], len(LEVEL_IDX)))
    out[:, :, LEVEL_IDX] += noise.astype(np.float32) * sigma
    return out


def baseline_perturb_batch(
    x: np.ndarray, rng: np.random.Generator, lo=-0.2, hi=0.2
) -> np.ndarray:
    delta = rng.uniform(lo, hi, size=(x.shape[0], 1, 1)).astype(np.float32)
    k = 1.0 / (1.0 + delta)
    out = x.copy()
    out[:, :, LEVEL_IDX] = ((1.0 + out[:, :, LEVEL_IDX] / 100.0) * k - 1.0) * 100.0
    out[:, :, SCALE_IDX] *= k
    return out


def augment_batch(
    x: np.ndarray, rng: np.random.Generator, p_warp=0.5, p_baseline=0.3
) -> np.ndarray:
    """Vectorised per-batch augmentation: (warp?) -> noise -> (baseline?)."""
    if rng.random() < p_warp:
        x = time_warp_batch(x, rng)
    x = add_noise_batch(x, rng)
    if rng.random() < p_baseline:
        x = baseline_perturb_batch(x, rng)
    return x


def mixup_batch(
    x: np.ndarray,
    y: dict[str, np.ndarray],
    stage: np.ndarray,
    rng: np.random.Generator,
    alpha=0.2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convex mixture of windows from the *same stage* (λ ~ Beta(α, α)).

    Enabled only late in training, regression targets only. Mixing within a
    stage avoids creating physically nonsensical pre/post-failure blends.
    Preserves the regression manifold locally; does not create new stage labels
    (the stage target is taken from the dominant partner).
    """
    b = x.shape[0]
    lam = rng.beta(alpha, alpha, size=b).astype(np.float32)
    partner = np.arange(b)
    for s in np.unique(stage):
        idx = np.where(stage == s)[0]
        partner[idx] = rng.permutation(idx)
    lam_x = lam[:, None, None]
    x_mix = lam_x * x + (1 - lam_x) * x[partner]
    y_mix = dict(y)
    for k in ("dtf", "load", "defl"):
        y_mix[k] = lam * y[k] + (1 - lam) * y[k][partner]
    y_mix["stage"] = np.where(lam >= 0.5, y["stage"], y["stage"][partner])
    return x_mix, y_mix
