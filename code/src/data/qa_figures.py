"""Phase 0 QA figures.

Per-specimen load / deflection / resistance vs time, and a combined ΔR/R₀ vs
deflection panel. Saves PNGs under ``reports/figures/phase0/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "processed" / "all_specimens.parquet"
FIGDIR = ROOT / "reports" / "figures" / "phase0"

plt.rcParams.update({"font.family": "serif", "figure.dpi": 130})


def load() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


def per_specimen_traces(df: pd.DataFrame) -> plt.Figure:
    """Load, deflection, and ΔR/R₀ vs time, one row per specimen."""
    specimens = sorted(df["specimen"].unique())
    fig, axes = plt.subplots(
        len(specimens), 1, figsize=(8, 2.4 * len(specimens)), sharex=False
    )
    for ax, name in zip(axes, specimens):
        g = df[df["specimen"] == name]
        ax.plot(g["t"], g["load_kN"], color="C0", label="load [kN]")
        ax.set_ylabel("load [kN]", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2 = ax.twinx()
        ax2.plot(g["t"], g["FCR_pct"], color="C3", label="ΔR/R₀ [%]", lw=0.8)
        ax2.set_ylabel("ΔR/R₀ [%]", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        t_peak = g["t"].iloc[g["load_kN"].to_numpy().argmax()]
        ax.axvline(t_peak, color="k", ls="--", lw=0.8)
        ax.set_title(f"{name}  (load peak at t={t_peak:.0f} s)")
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    return fig


def combined_fcr_vs_deflection(df: pd.DataFrame) -> plt.Figure:
    """All four ΔR/R₀ vs deflection curves on one axis."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in sorted(df["specimen"].unique()):
        g = df[df["specimen"] == name]
        ax.plot(g["deflection_mm"], g["FCR_pct"], label=name, lw=1.0)
    ax.set_xlabel("deflection [mm]")
    ax.set_ylabel("ΔR/R₀ [%]")
    ax.set_title("Fractional resistance change vs deflection (all specimens)")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = load()
    per_specimen_traces(df).savefig(FIGDIR / "per_specimen_traces.png")
    combined_fcr_vs_deflection(df).savefig(FIGDIR / "fcr_vs_deflection.png")
    print(f"Saved 2 figures to {FIGDIR}")


if __name__ == "__main__":
    main()
