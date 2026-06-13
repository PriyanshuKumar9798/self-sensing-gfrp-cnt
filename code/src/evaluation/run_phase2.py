"""Phase 2 runner: LOSO baselines -> scorecard + figures.

Produces ``reports/phase2_scorecard.md`` and the figures under
``reports/figures/phase2/``. Run via ``make -C code phase2``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation.loso import loso_predict
from src.features.engineering import FEATURE_COLUMNS
from src.features.labels import STAGE_NAMES
from src.models.baselines import (
    LGBMClassifierBaseline,
    LGBMRegressorBaseline,
    LinearBaseline,
    LogisticBaseline,
)

# LightGBM is fit on numpy arrays; sklearn's feature-name check is just noise here.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "processed" / "features.parquet"
FIGDIR = ROOT / "reports" / "figures" / "phase2"
SCORECARD = ROOT / "reports" / "phase2_scorecard.md"

SEED = 42
plt.rcParams.update({"font.family": "serif", "figure.dpi": 140})

LINEAR_FEATURES = [
    "fcr",
    "dfcr_dD",
    "roll_std_3mm",
    "cum_fcr",
    "defl_since_first_jump",
    "n_jumps_last_3mm",
    "loading_rate",
    "log_R0",
]
LINEAR_IDX = [FEATURE_COLUMNS.index(c) for c in LINEAR_FEATURES]

DEFAULT_TAU, DEFAULT_N = 2.0, 6  # initial alarm operating point (recorded in §10)
TAU_SWEEP = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
N_SWEEP = [3, 6, 12]

TARGETS = {"DTF_mm": "dtf", "load_kN": "load", "deflection_mm": "defl"}


def specimen_scalars(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """t_failure (= t at first DTF==0, the load peak) and loading rate per specimen."""
    info = {}
    for s, g in df.groupby("specimen"):
        t_fail = float(g.loc[g["DTF_mm"] <= 1e-9, "t"].min())
        info[s] = {"t_failure": t_fail, "rate": float(g["loading_rate"].iloc[0])}
    return info


def run_family(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """LOSO-predict every target for one model family; return merged predictions."""
    if name == "linear":
        reg = lambda: LinearBaseline(LINEAR_IDX)  # noqa: E731
        clf = lambda: LogisticBaseline(LINEAR_IDX)  # noqa: E731
    else:
        reg = LGBMRegressorBaseline
        clf = lambda: LGBMClassifierBaseline(num_class=4)  # noqa: E731

    m = df[
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
    ].copy()
    for target, short in TARGETS.items():
        p = loso_predict(df, FEATURE_COLUMNS, target, reg)
        m = m.merge(
            p[["specimen", "t", "y_pred"]].rename(columns={"y_pred": f"{short}_pred"}),
            on=["specimen", "t"],
        )
    ps = loso_predict(df, FEATURE_COLUMNS, "stage", clf)
    m = m.merge(
        ps[["specimen", "t", "y_pred"]].rename(columns={"y_pred": "stage_pred"}),
        on=["specimen", "t"],
    )
    return m


def fold_metrics(m: pd.DataFrame, info: dict) -> tuple[pd.DataFrame, list, np.ndarray]:
    """Per-specimen metric rows, alarm outcomes, and summed stage confusion."""
    rows, alarms = [], []
    conf = np.zeros((4, 4), dtype=int)
    for s, g in m.groupby("specimen"):
        t = g["t"].to_numpy()
        dtf_t, dtf_p = g["DTF_mm"].to_numpy(), g["dtf_pred"].to_numpy()
        stage_t, stage_p = g["stage"].to_numpy(), g["stage_pred"].to_numpy()
        rate = info[s]["rate"]
        ps_rmse = M.per_stage_rmse(
            g["load_kN"].to_numpy(), g["load_pred"].to_numpy(), stage_t
        )
        rows.append(
            {
                "specimen": s,
                "DTF_MAE": M.mae(dtf_t, dtf_p),
                "DTF_MedAE": M.medae(dtf_t, dtf_p),
                "TTF_MAE_s": M.ttf_mae(dtf_p, g["TTF_s"].to_numpy(), rate),
                "load_RMSE": M.rmse(g["load_kN"].to_numpy(), g["load_pred"].to_numpy()),
                "load_RMSE_I": ps_rmse[0],
                "load_RMSE_III": ps_rmse[2],
                "defl_RMSE": M.rmse(
                    g["deflection_mm"].to_numpy(), g["defl_pred"].to_numpy()
                ),
                "stage_bal_acc": M.stage_balanced_accuracy(stage_t, stage_p),
            }
        )
        conf += M.stage_confusion(stage_t, stage_p)
        alarms.append(
            M.classify_alarm(
                t, dtf_p, dtf_t, info[s]["t_failure"], DEFAULT_TAU, DEFAULT_N
            )
        )
    return pd.DataFrame(rows), alarms, conf


def alarm_curve(m: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Sweep (τ, N); precision/recall/F1/median-lead over the four specimens."""
    out = []
    for tau in TAU_SWEEP:
        for n in N_SWEEP:
            alarms = [
                M.classify_alarm(
                    g["t"].to_numpy(),
                    g["dtf_pred"].to_numpy(),
                    g["DTF_mm"].to_numpy(),
                    info[s]["t_failure"],
                    tau,
                    n,
                )
                for s, g in m.groupby("specimen")
            ]
            sc = M.alarm_scores(alarms)
            out.append({"tau": tau, "N": n, **sc})
    return pd.DataFrame(out)


# --- figures ----------------------------------------------------------------
def fig_pred_vs_true(lin: pd.DataFrame, lgb: pd.DataFrame) -> None:
    specs = sorted(lgb["specimen"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, s in zip(axes.ravel(), specs):
        g, gl = lgb[lgb.specimen == s], lin[lin.specimen == s]
        ax.plot(g["t"], g["DTF_mm"], "k", lw=1.2, label="true")
        ax.plot(g["t"], g["dtf_pred"], "C0", lw=1.0, label="LightGBM")
        ax.plot(gl["t"], gl["dtf_pred"], "C3", lw=0.8, alpha=0.7, label="Linear")
        ax.set_title(f"{s} (held out)")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("deflection to failure (mm)")
    axes[0, 0].legend()
    fig.suptitle("Predicted vs true deflection to failure (leave-one-out)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "dtf_pred_vs_true.png")
    plt.close(fig)


def fig_operating_curve(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    leads = curve["median_lead_s"].fillna(0.0)
    sc = ax.scatter(curve["recall"], curve["precision"], c=leads, cmap="viridis", s=80)
    for _, r in curve.iterrows():
        ax.annotate(
            f"τ{r.tau:g},N{int(r.N)}",
            (r["recall"], r["precision"]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    fig.colorbar(sc, label="median lead time [s]")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Alarm operating curve (LightGBM, LOSO)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "alarm_operating_curve.png")
    plt.close(fig)


# --- scorecard --------------------------------------------------------------
def _fmt_table(per_fold: pd.DataFrame) -> str:
    cols = [c for c in per_fold.columns if c != "specimen"]
    head = "| metric | " + " | ".join(per_fold["specimen"]) + " | mean |"
    sep = "|" + "---|" * (len(per_fold) + 2)
    lines = [head, sep]
    for c in cols:
        vals = [f"{v:.3f}" for v in per_fold[c]]
        lines.append(f"| {c} | " + " | ".join(vals) + f" | {per_fold[c].mean():.3f} |")
    return "\n".join(lines)


def _curve_table(curve: pd.DataFrame) -> str:
    cols = ["tau", "N", "precision", "recall", "f1", "median_lead_s", "tp", "fp", "fn"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in curve.iterrows():
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{r[c]:.2f}"
                    if c in ("precision", "recall", "f1", "median_lead_s")
                    else f"{r[c]:g}"
                )
                for c in cols
            )
            + " |"
        )
    return "\n".join(lines)


def write_scorecard(results: dict, curves: dict, confs: dict, alarm_def: dict) -> None:
    parts = [
        "# Phase 2: Classical LOSO baseline scorecard\n",
        f"Seed {SEED}. Features: {len(FEATURE_COLUMNS)} causal. "
        f"Models: Linear (floor, {len(LINEAR_FEATURES)} features) and LightGBM "
        "(inner-LOSO tuned). Validation: leave-one-specimen-out (4 folds).\n",
        f"Initial alarm operating point: τ_alarm={DEFAULT_TAU} mm, "
        f"N={DEFAULT_N} consecutive samples.\n",
    ]
    for fam in ("linear", "lgbm"):
        parts += [
            f"\n## {fam.upper()}: per-fold metrics\n",
            _fmt_table(results[fam]),
            "",
        ]
        sc = alarm_def[fam]
        parts.append(
            f"\n**Alarm @ default op point:** precision={sc['precision']:.2f}, "
            f"recall={sc['recall']:.2f}, F1={sc['f1']:.2f}, "
            f"median lead={sc['median_lead_s']:.1f} s "
            f"(TP={sc['tp']}, FP={sc['fp']}, FN={sc['fn']}).\n"
        )
    parts.append("\n## LightGBM stage confusion (rows=true, cols=pred)\n")
    names = [STAGE_NAMES[i] for i in range(4)]
    parts.append("| true\\pred | " + " | ".join(names) + " |")
    parts.append("|" + "---|" * 5)
    for i, row in enumerate(confs["lgbm"]):
        parts.append(f"| {names[i]} | " + " | ".join(str(int(v)) for v in row) + " |")
    parts.append("\n## Alarm operating curve (LightGBM)\n")
    parts.append(_curve_table(curves["lgbm"]))
    parts += [
        "\n## Figures\n",
        "- `figures/phase2/dtf_pred_vs_true.png`",
        "- `figures/phase2/alarm_operating_curve.png`\n",
    ]
    SCORECARD.write_text("\n".join(parts))


def main() -> None:
    np.random.seed(SEED)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEAT)
    info = specimen_scalars(df)

    preds, results, confs, alarm_def, curves = {}, {}, {}, {}, {}
    for fam in ("linear", "lgbm"):
        print(f"[{fam}] running LOSO ...")
        m = run_family(df, fam)
        preds[fam] = m
        rows, alarms, conf = fold_metrics(m, info)
        results[fam], confs[fam] = rows, conf
        alarm_def[fam] = M.alarm_scores(alarms)
        curves[fam] = alarm_curve(m, info)

    fig_pred_vs_true(preds["linear"], preds["lgbm"])
    fig_operating_curve(curves["lgbm"])
    write_scorecard(results, curves, confs, alarm_def)

    lg = results["lgbm"]
    sc = alarm_def["lgbm"]
    print("\n=== HEADLINE (LightGBM, LOSO) ===")
    print(
        "DTF MAE per fold:", {r.specimen: round(r.DTF_MAE, 2) for r in lg.itertuples()}
    )
    print(f"DTF MAE mean: {lg['DTF_MAE'].mean():.3f} mm")
    print(
        f"Alarm @ τ={DEFAULT_TAU},N={DEFAULT_N}: F1={sc['f1']:.2f}, "
        f"recall={sc['recall']:.2f}, precision={sc['precision']:.2f}, "
        f"median lead={sc['median_lead_s']:.1f} s"
    )
    print(f"Scorecard: {SCORECARD}")


if __name__ == "__main__":
    main()
