"""Phase 4 runner: calibrated DTF intervals + alarm operating point.

Trains the LightGBM quantile heads and the bootstrap ensemble under LOSO,
checks calibration, sweeps the alarm operating curve, picks the operating point,
and writes ``reports/phase4_operating_curve.md`` + figures. Run via
``make -C code phase4``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.alarm import (
    PRECISION_MIN,
    RECALL_MIN,
    per_specimen_timing,
    pick_operating_point,
    sweep_operating,
)
from src.evaluation.calibration import (
    QUANTILES,
    bootstrap_loso,
    conformal_upper_loso,
    empirical_coverage,
    quantile_loso,
)
from src.features.engineering import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "processed" / "features.parquet"
FIGDIR = ROOT / "reports" / "figures" / "phase4"
SCORECARD = ROOT / "reports" / "phase4_operating_curve.md"
SIGNAL = "q90_cal"  # conformally-calibrated conservative upper bound used by the alarm
plt.rcParams.update({"font.family": "serif", "figure.dpi": 140})


def specimen_scalars(df: pd.DataFrame) -> dict:
    info = {}
    for s, g in df.groupby("specimen"):
        info[s] = {
            "t_failure": float(g.loc[g["DTF_mm"] <= 1e-9, "t"].min()),
            "rate": float(g["loading_rate"].iloc[0]),
        }
    return info


# --- figures ----------------------------------------------------------------
def fig_reliability(cov: dict, boot_cov: float, cal_cov: float) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    xs = list(QUANTILES)
    ax.plot(xs, [cov[q] for q in xs], "o-", color="C0", label="raw model range")
    ax.scatter(
        [0.9], [cal_cov], color="C2", zorder=5, s=80, label="conformal 90% bound"
    )
    ax.scatter([0.9], [boot_cov], color="C3", zorder=5, label="bootstrap 90% bound")
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration of the deflection-to-failure range")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "reliability.png")
    plt.close(fig)


def fig_operating_curve(curve: pd.DataFrame, op) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    leads = curve["median_lead_s"].fillna(0.0)
    sc = ax.scatter(curve["recall"], curve["precision"], c=leads, cmap="viridis", s=70)
    for _, r in curve.iterrows():
        ax.annotate(
            f"τ{r.tau:g},N{int(r.N)}",
            (r["recall"], r["precision"]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    if op is not None:
        ax.scatter(
            [op["recall"]],
            [op["precision"]],
            s=240,
            facecolors="none",
            edgecolors="C3",
            linewidths=2,
            label="chosen",
        )
        ax.legend()
    ax.axhline(PRECISION_MIN, color="grey", ls=":", lw=0.8)
    ax.axvline(RECALL_MIN, color="grey", ls=":", lw=0.8)
    fig.colorbar(sc, label="median lead time [s]")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Alarm operating curve")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "operating_curve.png")
    plt.close(fig)


def fig_per_specimen(
    qm: pd.DataFrame, info: dict, timing: pd.DataFrame, tau: float
) -> None:
    specs = sorted(qm["specimen"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, s in zip(axes.ravel(), specs):
        g = qm[qm.specimen == s]
        ax.fill_between(
            g["t"], g["q10"], g["q90"], color="C0", alpha=0.25, label="raw range"
        )
        ax.plot(g["t"], g["q50"], "C0", lw=1.0, label="median estimate")
        ax.plot(
            g["t"],
            g["q90_cal"],
            "C1",
            lw=1.0,
            label="conservative estimate (alarm signal)",
        )
        ax.plot(g["t"], g["dtf_true"], "k", lw=1.0, label="true value")
        ax.axhline(tau, color="C2", ls=":", lw=0.8, label="alarm threshold")
        row = timing[timing.specimen == s].iloc[0]
        ax.axvline(info[s]["t_failure"], color="grey", ls="--", lw=0.8)
        if not np.isnan(row["alarm_t_s"]):
            ax.axvline(row["alarm_t_s"], color="C3", lw=1.2)
            ax.set_title(f"{s}: warned {row['lead_s']:.0f} s before failure")
        else:
            ax.set_title(f"{s}: no alarm")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("deflection to failure (mm)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Calibrated deflection to failure and alarm timing (each held-out bar)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "per_specimen_alarm.png")
    plt.close(fig)


# --- scorecard --------------------------------------------------------------
def _curve_table(curve: pd.DataFrame) -> str:
    cols = [
        "tau",
        "N",
        "precision",
        "recall",
        "f1",
        "median_lead_s",
        "false_alarms_per_test",
        "tp",
        "fp",
        "fn",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in curve.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(
                f"{v:.2f}"
                if c
                in (
                    "precision",
                    "recall",
                    "f1",
                    "median_lead_s",
                    "false_alarms_per_test",
                )
                else f"{v:g}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_scorecard(cov, boot_cov, cal_cov, curve, op, timing, median_rate) -> None:
    parts = [
        "# Phase 4: Calibration & alarm operating curve\n",
        "Model: LightGBM quantile heads (q=0.1/0.5/0.9, pinball loss) for DTF, "
        "LOSO. Alarm signal: the **conformally-calibrated** q=0.9 upper bound. "
        "Bootstrap ensemble (10 L1 models) as a second uncertainty estimate.\n",
        "> **Calibration finding.** The raw quantile heads collapse under LOSO: "
        "DTF is nearly deterministic within a specimen, so the dominant uncertainty "
        "is the cross-specimen D_peak, which pooled quantile regression cannot "
        "express (raw q90 covers only ~0.6 vs nominal 0.9). A cross-conformal "
        "correction (§4.1.5) inflates the upper bound to restore coverage.\n",
        "## Calibration (empirical coverage vs nominal)\n",
        "| estimator | nominal | empirical P(true ≤ pred) |",
        "|---|---|---|",
    ]
    for q in QUANTILES:
        parts.append(f"| raw q{int(q*100)} | {q:.2f} | {cov[q]:.3f} |")
    parts.append(f"| **conformal q90** | 0.90 | **{cal_cov:.3f}** |")
    parts.append(f"| bootstrap upper | 0.90 | {boot_cov:.3f} |")
    parts.append("\nQuantiles sorted per-sample to prevent crossing.\n")

    parts.append("\n## Alarm operating curve (signal = q90)\n")
    parts.append(_curve_table(curve))

    parts.append(
        f"\n## Chosen operating point "
        f"(max median lead s.t. recall ≥ {RECALL_MIN}, precision ≥ {PRECISION_MIN})\n"
    )
    if op is None:
        parts.append(
            "**No operating point satisfies the recall/precision floor** "
            "(n=4 makes precision/recall coarse). See the curve above; the "
            "best-F1 point is reported by the runner. The constraint should "
            "be revisited with the guide.\n"
        )
    else:
        lead_mm = op["median_lead_s"] * median_rate
        parts.append(
            f"- τ_alarm = **{op['tau']:g} mm**, N = **{int(op['N'])}** samples\n"
            f"- precision = {op['precision']:.2f}, recall = {op['recall']:.2f}, "
            f"F1 = {op['f1']:.2f}\n"
            f"- median lead time = **{op['median_lead_s']:.1f} s** "
            f"(~{lead_mm:.2f} mm of deflection at the median rate)\n"
            f"- false alarms per test = {op['false_alarms_per_test']:.2f}\n"
        )

    parts.append("\n## Per-specimen alarm timing at the chosen point\n")
    parts.append("| specimen | alarm t [s] | failure t [s] | lead [s] | outcome |")
    parts.append("|---|---|---|---|---|")
    for _, r in timing.iterrows():
        at = "-" if np.isnan(r["alarm_t_s"]) else f"{r['alarm_t_s']:.0f}"
        ld = "-" if np.isnan(r["lead_s"]) else f"{r['lead_s']:.0f}"
        parts.append(
            f"| {r['specimen']} | {at} | {r['t_failure_s']:.0f} | {ld} | {r['outcome']} |"
        )

    parts += [
        "\n## Figures\n",
        "- `figures/phase4/reliability.png`",
        "- `figures/phase4/operating_curve.png`",
        "- `figures/phase4/per_specimen_alarm.png`\n",
    ]
    SCORECARD.write_text("\n".join(parts))


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEAT)
    info = specimen_scalars(df)
    median_rate = float(np.median([info[s]["rate"] for s in info]))

    print("Training quantile heads (LOSO) ...")
    qm = quantile_loso(df, FEATURE_COLUMNS)
    print("Training bootstrap ensemble (LOSO) ...")
    bt = bootstrap_loso(df, FEATURE_COLUMNS)
    print("Conformal calibration of the upper bound (LOSO) ...")
    cal = conformal_upper_loso(df, FEATURE_COLUMNS)
    qm = qm.merge(bt, on=["specimen", "t"]).merge(cal, on=["specimen", "t"])

    cov = empirical_coverage(qm)
    boot_cov = float((qm["dtf_true"] <= qm["boot_hi"]).mean())
    cal_cov = float((qm["dtf_true"] <= qm["q90_cal"]).mean())

    curve = sweep_operating(qm, info, SIGNAL)
    op = pick_operating_point(curve)
    if op is None:
        op_report = curve.sort_values(["f1", "median_lead_s"], ascending=False).iloc[0]
        print("No point meets the recall/precision floor; reporting best-F1 point.")
    else:
        op_report = op
    timing = per_specimen_timing(
        qm, info, SIGNAL, op_report["tau"], int(op_report["N"])
    )

    fig_reliability(cov, boot_cov, cal_cov)
    fig_operating_curve(curve, op)
    fig_per_specimen(qm, info, timing, op_report["tau"])
    write_scorecard(cov, boot_cov, cal_cov, curve, op, timing, median_rate)

    print("\n=== HEADLINE (Phase 4) ===")
    print(
        "Raw quantile coverage:",
        {f"q{int(q*100)}": round(cov[q], 2) for q in QUANTILES},
    )
    print(f"Conformal q90 coverage: {cal_cov:.2f} (nominal 0.90)")
    print(
        f"Chosen op point: τ={op_report['tau']:g} mm, N={int(op_report['N'])}, "
        f"precision={op_report['precision']:.2f}, recall={op_report['recall']:.2f}, "
        f"median lead={op_report['median_lead_s']:.1f} s"
        + ("" if op is not None else "  [floor not met]")
    )
    print(f"Scorecard: {SCORECARD}")


if __name__ == "__main__":
    main()
