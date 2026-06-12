"""Phase 0 data pipeline.

Parse the four UTM `.xls` and four Sourcemeter `.csv` files, align them on a
common 12 Hz grid, compute ΔR/R₀ and the DTF/TTF labels, and write a single
tidy parquet to ``data/processed/all_specimens.parquet``.

All design decisions follow ``PROJECT_PLAN.md`` §5 Phase 0.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd

# --- file locations ---------------------------------------------------------
# ROOT is the code/ folder; the raw data lives one level up at the ugrp root.
ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
UTM_DIR = PROJECT_ROOT / "UTM"
SM_DIR = PROJECT_ROOT / "Sourcemeter"
OUT = ROOT / "data" / "processed" / "all_specimens.parquet"

# specimen -> (UTM file, Sourcemeter file)
FILES = {
    "S1": ("S1_Flex_23_Apr_26_12_21_05.xls", "s1_0423_124026.csv"),
    "S2": ("S2_Flex_23_Apr_26_12_21_26.xls", "s2_0423_124030.csv"),
    "S3": ("S3_Flex_23_Apr_26_12_21_42.xls", "s3_0423_124033.csv"),
    "S4": ("S4_Flex_23_Apr_26_12_22_03.xls", "s4_0423_124037.csv"),
}

FS = 12.0  # common resample grid (Hz)
UTM_HEADER_ROW = 9  # 0-indexed; data starts at row 10
SM_HEADER_ROW = 8  # 0-indexed; metadata is rows 0-7, header row 8, data from 9


@dataclass
class Specimen:
    """Per-specimen scalar summary produced by the pipeline."""

    name: str
    t_peak: float
    d_peak: float
    max_fcr_pct: float
    loading_rate: float
    r0_ohm: float


# --- parsers ----------------------------------------------------------------
def parse_utm(path: Path) -> pd.DataFrame:
    """Return UTM trace with columns ``t``, ``load_kN``, ``deflection_mm``.

    Signs are flipped so compressive load and downward deflection are positive.
    """
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)
    header = [sheet.cell_value(UTM_HEADER_ROW, c) for c in range(sheet.ncols)]
    col = {
        name: header.index(name) for name in ("Time sec", "Load kN", "Rel. Stroke mm")
    }
    rows = [
        [
            sheet.cell_value(r, col["Time sec"]),
            sheet.cell_value(r, col["Load kN"]),
            sheet.cell_value(r, col["Rel. Stroke mm"]),
        ]
        for r in range(UTM_HEADER_ROW + 1, sheet.nrows)
    ]
    df = (
        pd.DataFrame(rows, columns=["t", "load_kN", "deflection_mm"])
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    df["load_kN"] *= -1.0  # compressive positive
    df["deflection_mm"] *= -1.0  # downward positive
    return df.reset_index(drop=True)


def parse_sourcemeter(path: Path) -> pd.DataFrame:
    """Return resistance trace with columns ``t`` and ``R_ohm``."""
    df = pd.read_csv(path, skiprows=SM_HEADER_ROW)
    out = pd.DataFrame(
        {
            "t": pd.to_numeric(df["Relative Time"], errors="coerce"),
            "R_ohm": pd.to_numeric(df["Reading"], errors="coerce"),
        }
    ).dropna()
    return out.reset_index(drop=True)


# --- resampling and labels --------------------------------------------------
def resample_to_grid(utm: pd.DataFrame, sm: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate both traces onto a shared 12 Hz grid.

    The grid spans the overlap of the two records. Both instruments start near
    t=0; we assume no fixed offset (verified by the alignment check downstream).
    """
    t0 = max(utm["t"].iloc[0], sm["t"].iloc[0])
    t1 = min(utm["t"].iloc[-1], sm["t"].iloc[-1])
    grid = np.arange(t0, t1, 1.0 / FS)
    out = pd.DataFrame({"t": grid})
    out["load_kN"] = np.interp(grid, utm["t"], utm["load_kN"])
    out["deflection_mm"] = np.interp(grid, utm["t"], utm["deflection_mm"])
    out["R_ohm"] = np.interp(grid, sm["t"], sm["R_ohm"])
    return out


def alignment_offset(df: pd.DataFrame) -> float:
    """Estimate UTM-vs-sourcemeter offset (s) by matching first-rise events.

    Returns ``t_resistance_rise − t_deflection_rise``; ~0 means aligned. The
    raw traces are not shifted — this is a reported diagnostic only.
    """
    defl = df["deflection_mm"].to_numpy()
    fcr = (df["R_ohm"].to_numpy() / df["R_ohm"].to_numpy()[0]) - 1.0
    t = df["t"].to_numpy()
    d_event = _first_crossing(t, defl, 0.05 * np.nanmax(defl))
    r_event = _first_crossing(t, fcr, 0.05 * np.nanmax(fcr))
    if d_event is None or r_event is None:
        return float("nan")
    return r_event - d_event


def _first_crossing(t: np.ndarray, y: np.ndarray, thresh: float) -> float | None:
    idx = np.argmax(y > thresh)
    if y[idx] <= thresh:
        return None
    return float(t[idx])


def compute_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, Specimen, str]:
    """Add FCR, TTF, DTF, loading rate columns. Returns (df, summary, name set later)."""
    # ΔR/R₀: R₀ = median resistance over the first 1 s.
    r0 = float(df.loc[df["t"] <= df["t"].iloc[0] + 1.0, "R_ohm"].median())
    df["FCR_pct"] = (df["R_ohm"] / r0 - 1.0) * 100.0

    # Failure point = load peak.
    i_peak = int(df["load_kN"].to_numpy().argmax())
    t_peak = float(df["t"].iloc[i_peak])
    d_peak = float(df["deflection_mm"].iloc[i_peak])

    df["TTF_s"] = (t_peak - df["t"]).clip(lower=0.0)
    df["DTF_mm"] = (d_peak - df["deflection_mm"]).clip(lower=0.0)

    # Loading rate: linear fit of deflection vs time on the first half of the test.
    half = df["t"] <= (df["t"].iloc[0] + df["t"].iloc[-1]) / 2.0
    slope = float(np.polyfit(df.loc[half, "t"], df.loc[half, "deflection_mm"], 1)[0])
    df["loading_rate_mm_per_s"] = slope

    summary = Specimen(
        name="",
        t_peak=t_peak,
        d_peak=d_peak,
        max_fcr_pct=float(df["FCR_pct"].max()),
        loading_rate=slope,
        r0_ohm=r0,
    )
    return df, summary, ""


# --- driver -----------------------------------------------------------------
def build() -> tuple[pd.DataFrame, list[Specimen], dict[str, float]]:
    """Run the full pipeline and return (table, summaries, offsets)."""
    frames, summaries, offsets = [], [], {}
    for name, (utm_f, sm_f) in FILES.items():
        utm = parse_utm(UTM_DIR / utm_f)
        sm = parse_sourcemeter(SM_DIR / sm_f)
        df = resample_to_grid(utm, sm)
        offsets[name] = alignment_offset(df)
        df, summ, _ = compute_labels(df)
        summ.name = name
        df.insert(0, "specimen", name)
        frames.append(
            df[
                [
                    "specimen",
                    "t",
                    "load_kN",
                    "deflection_mm",
                    "R_ohm",
                    "FCR_pct",
                    "TTF_s",
                    "DTF_mm",
                    "loading_rate_mm_per_s",
                ]
            ]
        )
        summaries.append(summ)
    return pd.concat(frames, ignore_index=True), summaries, offsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all_specimens.parquet")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    table, summaries, offsets = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)

    print(f"Wrote {args.out}  ({len(table)} rows)\n")
    print(
        f"{'spec':4} {'t_peak[s]':>9} {'D_peak[mm]':>10} {'maxΔR/R₀[%]':>12} "
        f"{'rate[mm/s]':>11} {'R₀[MΩ]':>9} {'align[s]':>9}"
    )
    for s in summaries:
        print(
            f"{s.name:4} {s.t_peak:9.2f} {s.d_peak:10.3f} {s.max_fcr_pct:12.2f} "
            f"{s.loading_rate:11.4f} {s.r0_ohm / 1e6:9.2f} {offsets[s.name]:9.3f}"
        )


if __name__ == "__main__":
    main()
