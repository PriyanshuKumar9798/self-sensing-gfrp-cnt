"""Predict deflection/time to failure for a new specimen.

Trains a final LightGBM model on every labelled specimen in
``data/processed/features.parquet``, then applies it to a new specimen given its
raw UTM ``.xls`` and Sourcemeter ``.csv`` files. If the new bar was loaded to
failure (a clear post-peak load drop), the true deflection-to-failure is known
and the mean absolute error is reported.

Usage:
    python -m src.evaluation.predict --utm path/to.xls --sm path/to.csv --name S5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.pipeline import (
    compute_labels,
    parse_sourcemeter,
    parse_utm,
    resample_to_grid,
)
from src.features.engineering import FEATURE_COLUMNS, build_features
from src.features.labels import build_labels
from src.models.baselines import LGBMRegressorBaseline

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "processed" / "features.parquet"
FIGDIR = ROOT / "reports" / "figures" / "predict"
plt.rcParams.update({"font.family": "serif", "figure.dpi": 130})


def process_raw(utm_path: Path, sm_path: Path, name: str) -> pd.DataFrame:
    """Parse and align one specimen's raw files into the tidy per-sample table."""
    utm = parse_utm(utm_path)
    sm = parse_sourcemeter(sm_path)
    df = resample_to_grid(utm, sm)
    df, _, _ = compute_labels(df)
    df.insert(0, "specimen", name)
    return df


def has_failed(df: pd.DataFrame) -> bool:
    """True if the load shows a clear post-peak drop, so the DTF labels are valid."""
    load = df["load_kN"].to_numpy()
    i_peak = int(load.argmax())
    return i_peak < len(load) * 0.97 and load[i_peak:].min() < 0.8 * load[i_peak]


def predict_specimen(utm_path: Path, sm_path: Path, name: str):
    """Return (predictions DataFrame, failed flag, loading rate)."""
    feat = pd.read_parquet(FEAT)
    model = LGBMRegressorBaseline()
    model.fit(
        feat[FEATURE_COLUMNS].to_numpy(),
        feat["DTF_mm"].to_numpy(),
        feat["specimen"].to_numpy(),
    )

    raw = process_raw(utm_path, sm_path, name)
    feats = build_features(raw)
    labels = build_labels(raw)
    m = feats.merge(labels, on=["specimen", "t"])

    dtf_pred = np.clip(model.predict(m[FEATURE_COLUMNS].to_numpy()), 0.0, None)
    rate = float(raw["loading_rate_mm_per_s"].iloc[0])

    out = pd.DataFrame(
        {
            "specimen": name,
            "t": m["t"].to_numpy(),
            "deflection_mm": m["deflection_mm"].to_numpy(),
            "dtf_pred_mm": dtf_pred,
            "ttf_pred_s": dtf_pred / rate,
        }
    )
    failed = has_failed(raw)
    if failed:
        out["dtf_true_mm"] = m["DTF_mm"].to_numpy()
    return out, failed, rate


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict DTF/TTF for a new specimen")
    ap.add_argument("--utm", type=Path, required=True)
    ap.add_argument("--sm", type=Path, required=True)
    ap.add_argument("--name", default="NEW")
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    out, failed, rate = predict_specimen(args.utm, args.sm, args.name)
    print(f"Specimen {args.name}: {len(out)} samples, loading rate {rate:.4f} mm/s.")
    if failed:
        mae = float(np.mean(np.abs(out["dtf_pred_mm"] - out["dtf_true_mm"])))
        print(f"Bar reached failure. DTF mean absolute error = {mae:.2f} mm.")
    else:
        print(
            "No clear failure detected: predictions only "
            "(accuracy needs a bar loaded to failure)."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    csv = args.out / f"predict_{args.name}.csv"
    out.to_csv(csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(out["t"], out["dtf_pred_mm"], "C0", label="predicted DTF")
    if failed:
        ax.plot(out["t"], out["dtf_true_mm"], "k", lw=1.0, label="true DTF")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("deflection to failure [mm]")
    ax.set_title(f"DTF prediction: {args.name}")
    ax.legend()
    fig.tight_layout()
    figpath = FIGDIR / f"predict_{args.name}.png"
    fig.savefig(figpath)
    plt.close(fig)
    print(f"Wrote {csv}\nWrote {figpath}")


if __name__ == "__main__":
    main()
