"""Phase 3 runner: sequence models + augmentation under LOSO.

Trains the main TCN model and the ablations, compares to the Phase 2 LightGBM
baseline, and writes ``reports/phase3_scorecard.md`` + figures. Run via
``make -C code phase3``. CPU only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.features.engineering import FEATURE_COLUMNS
from src.features.labels import STAGE_NAMES
from src.training.train_sequence import SEED, Config, run_config

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "processed" / "features.parquet"
FIGDIR = ROOT / "reports" / "figures" / "phase3"
SCORECARD = ROOT / "reports" / "phase3_scorecard.md"

DEFAULT_TAU, DEFAULT_N = 2.0, 6
plt.rcParams.update({"font.family": "serif", "figure.dpi": 140})

# Phase 2 LightGBM reference (from reports/phase2_scorecard.md, reproducible).
PHASE2_LGBM = {"S1": 1.269, "S2": 1.049, "S3": 2.101, "S4": 1.605, "mean": 1.506}

CONFIGS = [
    Config(model="tcn", aug=True, w_mm=2.0, label="tcn_aug_w2", mixup=True),
    Config(model="tcn", aug=False, w_mm=2.0, label="tcn_noaug_w2", mixup=False),
    Config(model="lstm", aug=True, w_mm=2.0, label="lstm_aug_w2", mixup=True),
    Config(model="tcn", aug=True, w_mm=1.0, label="tcn_aug_w1", mixup=True),
    Config(model="tcn", aug=True, w_mm=5.0, label="tcn_aug_w5", mixup=True),
]


def specimen_scalars(df: pd.DataFrame) -> dict:
    info = {}
    for s, g in df.groupby("specimen"):
        info[s] = {
            "t_failure": float(g.loc[g["DTF_mm"] <= 1e-9, "t"].min()),
            "rate": float(g["loading_rate"].iloc[0]),
        }
    return info


def summarize(m: pd.DataFrame, info: dict) -> dict:
    """Per-fold DTF MAE + alarm scores at the default operating point."""
    per_fold, alarms = {}, []
    for s, g in m.groupby("specimen"):
        per_fold[s] = M.mae(g["DTF_mm"].to_numpy(), g["dtf_pred"].to_numpy())
        alarms.append(
            M.classify_alarm(
                g["t"].to_numpy(),
                g["dtf_pred"].to_numpy(),
                g["DTF_mm"].to_numpy(),
                info[s]["t_failure"],
                DEFAULT_TAU,
                DEFAULT_N,
            )
        )
    per_fold["mean"] = float(np.mean(list(per_fold.values())))
    return {"dtf_mae": per_fold, "alarm": M.alarm_scores(alarms)}


def full_metrics(m: pd.DataFrame, info: dict) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    conf = np.zeros((4, 4), dtype=int)
    for s, g in m.groupby("specimen"):
        stage_t = g["stage"].to_numpy()
        ps = M.per_stage_rmse(
            g["load_kN"].to_numpy(), g["load_pred"].to_numpy(), stage_t
        )
        rows.append(
            {
                "specimen": s,
                "DTF_MAE": M.mae(g["DTF_mm"].to_numpy(), g["dtf_pred"].to_numpy()),
                "DTF_MedAE": M.medae(g["DTF_mm"].to_numpy(), g["dtf_pred"].to_numpy()),
                "TTF_MAE_s": M.ttf_mae(
                    g["dtf_pred"].to_numpy(), g["TTF_s"].to_numpy(), info[s]["rate"]
                ),
                "load_RMSE": M.rmse(g["load_kN"].to_numpy(), g["load_pred"].to_numpy()),
                "load_RMSE_I": ps[0],
                "load_RMSE_III": ps[2],
                "defl_RMSE": M.rmse(
                    g["deflection_mm"].to_numpy(), g["defl_pred"].to_numpy()
                ),
                "stage_bal_acc": M.stage_balanced_accuracy(
                    stage_t, g["stage_pred"].to_numpy()
                ),
            }
        )
        conf += M.stage_confusion(stage_t, g["stage_pred"].to_numpy())
    return pd.DataFrame(rows), conf


# --- figures ----------------------------------------------------------------
def fig_pred_vs_true(m: pd.DataFrame, name: str) -> None:
    specs = sorted(m["specimen"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, s in zip(axes.ravel(), specs):
        g = m[m.specimen == s]
        ax.plot(g["t"], g["DTF_mm"], "k", lw=1.2, label="true")
        ax.plot(g["t"], g["dtf_pred"], "C2", lw=1.0, label=name)
        ax.set_title(f"{s} (held out)")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("deflection to failure (mm)")
    axes[0, 0].legend()
    fig.suptitle(f"{name}: predicted vs true deflection to failure")
    fig.tight_layout()
    fig.savefig(FIGDIR / "dtf_pred_vs_true.png")
    plt.close(fig)


def fig_ablation(summ: dict) -> None:
    labels = [c.label for c in CONFIGS]
    means = [summ[lbl]["dtf_mae"]["mean"] for lbl in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, color="C2")
    ax.axhline(PHASE2_LGBM["mean"], color="C3", ls="--", label="Phase 2 LightGBM")
    ax.axhline(PHASE2_LGBM["mean"] * 0.85, color="C0", ls=":", label="15% target")
    for b, v in zip(bars, means):
        ax.annotate(
            f"{v:.2f}",
            (b.get_x() + b.get_width() / 2, v),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("mean error in deflection to failure (mm)")
    ax.set_title("Phase 3 ablations vs Phase 2")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "ablation_dtf_mae.png")
    plt.close(fig)


# --- scorecard --------------------------------------------------------------
def _fold_table(per_fold: pd.DataFrame) -> str:
    cols = [c for c in per_fold.columns if c != "specimen"]
    lines = [
        "| metric | " + " | ".join(per_fold["specimen"]) + " | mean |",
        "|" + "---|" * (len(per_fold) + 2),
    ]
    for c in cols:
        vals = " | ".join(f"{v:.3f}" for v in per_fold[c])
        lines.append(f"| {c} | {vals} | {per_fold[c].mean():.3f} |")
    return "\n".join(lines)


def write_scorecard(
    summ: dict, main_label: str, main_tbl: pd.DataFrame, conf: np.ndarray, nparams: int
) -> None:
    main = summ[main_label]
    improve = 100 * (1 - main["dtf_mae"]["mean"] / PHASE2_LGBM["mean"])
    verdict = "beats" if improve > 0 else "does not beat"
    parts = [
        "# Phase 3: Sequence-model scorecard\n",
        f"Seed {SEED}. CPU. Inner-LOSO ensemble (3 models/fold). Multi-task "
        "(DTF Huber + load/defl MSE + stage CE). Validation: leave-one-specimen-out.\n",
        f"**Selected model: `{main_label}`** ({nparams} params), chosen as the "
        "lowest LOSO DTF MAE across the ablation.\n",
        f"**Headline:** selected mean DTF MAE = {main['dtf_mae']['mean']:.3f} mm vs "
        f"Phase 2 LightGBM {PHASE2_LGBM['mean']:.3f} mm "
        f"(**{improve:+.1f}%**; it {verdict} the baseline, target was ≥15% reduction). "
        f"Alarm @ τ={DEFAULT_TAU},N={DEFAULT_N}: F1={main['alarm']['f1']:.2f}, "
        f"recall={main['alarm']['recall']:.2f}, precision={main['alarm']['precision']:.2f}, "
        f"median lead={main['alarm']['median_lead_s']:.1f} s.\n",
        "> **Finding.** No sequence model reaches the +15% target. The TCN "
        "underperforms LightGBM and augmentation does not help it here; only the "
        "LSTM with augmentation marginally edges the baseline. With n=4 and an 8× "
        "baseline spread, the gradient-boosted trees of Phase 2 remain very "
        "competitive (as anticipated in §3 D6).\n",
        "\n## Ablation: mean DTF MAE and alarm F1\n",
        "| config | S1 | S2 | S3 | S4 | mean | vs P2 | alarm F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in CONFIGS:
        d = summ[c.label]["dtf_mae"]
        a = summ[c.label]["alarm"]
        vs = 100 * (1 - d["mean"] / PHASE2_LGBM["mean"])
        parts.append(
            f"| {c.label} | {d['S1']:.2f} | {d['S2']:.2f} | {d['S3']:.2f} | "
            f"{d['S4']:.2f} | {d['mean']:.3f} | {vs:+.1f}% | {a['f1']:.2f} |"
        )
    parts.append(
        f"| _Phase 2 LightGBM_ | {PHASE2_LGBM['S1']:.2f} | {PHASE2_LGBM['S2']:.2f} "
        f"| {PHASE2_LGBM['S3']:.2f} | {PHASE2_LGBM['S4']:.2f} | "
        f"{PHASE2_LGBM['mean']:.3f} | - | - |"
    )

    parts += [
        f"\n## Selected model ({main_label}): full per-fold metrics\n",
        _fold_table(main_tbl),
    ]
    names = [STAGE_NAMES[i] for i in range(4)]
    parts += [
        "\n## Selected model stage confusion (rows=true, cols=pred)\n",
        "| true\\pred | " + " | ".join(names) + " |",
        "|" + "---|" * 5,
    ]
    for i, row in enumerate(conf):
        parts.append(f"| {names[i]} | " + " | ".join(str(int(v)) for v in row) + " |")
    parts += [
        "\n## Figures\n",
        "- `figures/phase3/dtf_pred_vs_true.png`",
        "- `figures/phase3/ablation_dtf_mae.png`\n",
    ]
    SCORECARD.write_text("\n".join(parts))


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEAT)
    info = specimen_scalars(df)

    summ, preds = {}, {}
    for cfg in CONFIGS:
        print(f"[{cfg.label}] training LOSO ...")
        m = run_config(df, FEATURE_COLUMNS, cfg)
        summ[cfg.label] = summarize(m, info)
        preds[cfg.label] = m
        print(f"  mean DTF MAE = {summ[cfg.label]['dtf_mae']['mean']:.3f} mm")

    # Select the model with the lowest LOSO DTF MAE (honest model selection).
    best_label = min(summ, key=lambda lbl: summ[lbl]["dtf_mae"]["mean"])
    main_m = preds[best_label]
    nparams = main_m.attrs.get("nparams", 0)

    main_tbl, conf = full_metrics(main_m, info)
    fig_pred_vs_true(main_m, best_label)
    fig_ablation(summ)
    write_scorecard(summ, best_label, main_tbl, conf, nparams)

    main = summ[best_label]
    improve = 100 * (1 - main["dtf_mae"]["mean"] / PHASE2_LGBM["mean"])
    print(f"\n=== HEADLINE (Phase 3, LOSO): selected: {best_label} ===")
    print(
        f"mean DTF MAE = {main['dtf_mae']['mean']:.3f} mm "
        f"({improve:+.1f}% vs Phase 2 {PHASE2_LGBM['mean']:.3f})"
    )
    print(
        f"Alarm F1={main['alarm']['f1']:.2f}, recall={main['alarm']['recall']:.2f}, "
        f"precision={main['alarm']['precision']:.2f}, "
        f"median lead={main['alarm']['median_lead_s']:.1f} s"
    )
    print(f"Scorecard: {SCORECARD}")


if __name__ == "__main__":
    main()
