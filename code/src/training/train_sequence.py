"""Phase 3 training: one LOSO fold of a multi-task sequence model.

Each outer fold trains an **inner-LOSO ensemble**: for each of the three
training specimens held out in turn, a model is trained on the other two and
early-stopped on the held-out one (an honest cross-specimen signal). The three
models' predictions on the outer test specimen are averaged. This uses all three
training specimens and sharply cuts the fold-to-fold variance that a single
model shows with n = 4.

AdamW (lr 3e-4, wd 1e-4), cosine schedule, augmentation per batch in original
units then standardised, mixup only in the final few epochs. CPU only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.sequence import count_params, make_model
from src.training.augmentations import augment_batch, mixup_batch
from src.training.windows import build_windows

SEED = 42
LOSS_W = {"dtf": 1.0, "load": 0.3, "defl": 0.3, "stage": 0.5}
TRAIN_STRIDE = 3
EVAL_STRIDE = 1
MAX_EPOCHS = 25
PATIENCE = 6
BATCH = 256
MIXUP_LAST = 5


@dataclass
class Config:
    model: str = "tcn"
    aug: bool = True
    w_mm: float = 2.0
    t_len: int = 32
    label: str = "tcn_aug_w2"
    mixup: bool = True
    extra: dict = field(default_factory=dict)


class _Scaler:
    def __init__(self, x: np.ndarray):
        flat = x.reshape(-1, x.shape[-1])
        self.mean = flat.mean(0)
        self.std = flat.std(0) + 1e-8

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Winsorise to ±5σ so out-of-distribution test specimens (the 8× baseline
        # spread) cannot push the network into wild extrapolation.
        z = (x - self.mean) / self.std
        return np.clip(z, -5.0, 5.0).astype(np.float32)


def _stats(y: np.ndarray) -> tuple[float, float]:
    return float(y.mean()), float(y.std() + 1e-8)


def _loss(model, xb, yb, w) -> torch.Tensor:
    out = model(xb)
    return (
        w["dtf"] * nn.functional.huber_loss(out["dtf"], yb["dtf"])
        + w["load"] * nn.functional.mse_loss(out["load"], yb["load"])
        + w["defl"] * nn.functional.mse_loss(out["defl"], yb["defl"])
        + w["stage"] * nn.functional.cross_entropy(out["stage"], yb["stage"])
    )


def _train_single(df, train2, val_spec, feat_cols, cfg, seed):
    """Train on the two specimens in ``train2``; early-stop on ``val_spec``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr = build_windows(df, train2, feat_cols, cfg.w_mm, cfg.t_len, TRAIN_STRIDE)
    va = build_windows(df, [val_spec], feat_cols, cfg.w_mm, cfg.t_len, TRAIN_STRIDE)

    scaler = _Scaler(tr.X)
    dm, ds = _stats(tr.dtf)
    lm, ls = _stats(tr.load)
    fm, fs = _stats(tr.defl)
    stats = (dm, ds, lm, ls, fm, fs)

    def targets(ws):
        return {
            "dtf": ((ws.dtf - dm) / ds).astype(np.float32),
            "load": ((ws.load - lm) / ls).astype(np.float32),
            "defl": ((ws.defl - fm) / fs).astype(np.float32),
            "stage": ws.stage,
        }

    ytr = targets(tr)
    Xva = torch.tensor(scaler(va.X))
    dtf_va_true = va.dtf

    model = make_model(cfg.model, len(feat_cols))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    n = len(tr.X)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(n)
        use_mixup = cfg.mixup and cfg.aug and epoch >= MAX_EPOCHS - MIXUP_LAST
        for b in range(0, n, BATCH):
            idx = order[b : b + BATCH]
            xb_raw = tr.X[idx]
            if cfg.aug:
                xb_raw = augment_batch(xb_raw, rng)
            xb = scaler(xb_raw)
            yb = {k: v[idx] for k, v in ytr.items()}
            if use_mixup:
                xb, yb = mixup_batch(xb, yb, yb["stage"], rng)
            yb_t = {
                "dtf": torch.tensor(yb["dtf"].astype(np.float32)),
                "load": torch.tensor(yb["load"].astype(np.float32)),
                "defl": torch.tensor(yb["defl"].astype(np.float32)),
                "stage": torch.tensor(yb["stage"]),
            }
            opt.zero_grad()
            _loss(model, torch.tensor(xb), yb_t, LOSS_W).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            dtf_va = model(Xva)["dtf"].numpy() * ds + dm
        vmae = float(np.mean(np.abs(dtf_va - dtf_va_true)))
        if vmae < best - 1e-4:
            best, best_state, bad = (
                vmae,
                {k: v.clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, stats


def _predict(df, test_spec, model, scaler, stats, feat_cols, cfg):
    dm, ds, lm, ls, fm, fs = stats
    te = build_windows(df, [test_spec], feat_cols, cfg.w_mm, cfg.t_len, EVAL_STRIDE)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(scaler(te.X)))
        probs = torch.softmax(out["stage"], dim=1).numpy()
    # DTF and deflection are physically non-negative; clamp to a sane upper bound.
    return te.t, {
        "dtf": np.clip(out["dtf"].numpy() * ds + dm, 0.0, 25.0),
        "load": out["load"].numpy() * ls + lm,
        "defl": np.clip(out["defl"].numpy() * fs + fm, 0.0, 25.0),
        "stage_probs": probs,
    }


def train_fold(df: pd.DataFrame, test_spec: str, feat_cols: list[str], cfg: Config):
    """Inner-LOSO ensemble of three models; averaged predictions on the held-out specimen."""
    specs = sorted(df["specimen"].unique())
    train_specs = [s for s in specs if s != test_spec]

    t_ref, acc = None, []
    nparams = 0
    for k, val_spec in enumerate(train_specs):
        train2 = [s for s in train_specs if s != val_spec]
        model, scaler, stats = _train_single(
            df, train2, val_spec, feat_cols, cfg, SEED + k
        )
        nparams = count_params(model)
        t_ref, preds = _predict(df, test_spec, model, scaler, stats, feat_cols, cfg)
        acc.append(preds)

    dtf = np.mean([p["dtf"] for p in acc], axis=0)
    load = np.mean([p["load"] for p in acc], axis=0)
    defl = np.mean([p["defl"] for p in acc], axis=0)
    stage = np.mean([p["stage_probs"] for p in acc], axis=0).argmax(1)
    return (
        pd.DataFrame(
            {
                "specimen": test_spec,
                "t": t_ref,
                "dtf_pred": dtf,
                "load_pred": load,
                "defl_pred": defl,
                "stage_pred": stage,
            }
        ),
        nparams,
    )


def run_config(df: pd.DataFrame, feat_cols: list[str], cfg: Config) -> pd.DataFrame:
    """LOSO over all four specimens for one configuration; merged predictions + truths."""
    specs = sorted(df["specimen"].unique())
    preds, nparams = [], 0
    for s in specs:
        p, nparams = train_fold(df, s, feat_cols, cfg)
        preds.append(p)
    pred = pd.concat(preds, ignore_index=True)
    truth = df[
        [
            "specimen",
            "t",
            "DTF_mm",
            "TTF_s",
            "load_kN",
            "deflection_mm",
            "stage",
            "loading_rate",
        ]
    ]
    out = truth.merge(pred, on=["specimen", "t"])
    out.attrs["nparams"] = nparams
    return out
