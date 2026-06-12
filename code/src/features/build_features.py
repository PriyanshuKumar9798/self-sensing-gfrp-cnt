"""Build ``data/processed/features.parquet`` = causal features + labels.

Usage: ``python -m src.features.build_features``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.engineering import FEATURE_COLUMNS, build_features
from src.features.labels import LABEL_COLUMNS, build_labels

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "processed" / "all_specimens.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"


def build() -> pd.DataFrame:
    df = pd.read_parquet(IN)
    feats = build_features(df)
    labels = build_labels(df)
    merged = feats.merge(labels, on=["specimen", "t"], validate="one_to_one")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Build features.parquet")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    df = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"Wrote {args.out}  ({len(df)} rows, {df.shape[1]} cols)\n")
    print("Features:", ", ".join(FEATURE_COLUMNS))
    print("Labels:  ", ", ".join(LABEL_COLUMNS), "\n")
    print(df[FEATURE_COLUMNS].describe().T.to_string(float_format=lambda x: f"{x:.4g}"))
    print("\nNaN per column:")
    nz = df.isna().sum()
    print(nz[nz > 0].to_string() if nz.any() else "  none")
    print("\nStage counts:")
    print(df["stage"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
