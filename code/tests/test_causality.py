"""Causality test (PROJECT_PLAN hard rule 1).

Build features for a specimen, randomise every input column AFTER some time t0,
recompute, and assert the features at all times <= t0 are bit-identical. The
known constants (loading rate, R₀) are held fixed, as they would be in
deployment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import FEATURE_COLUMNS, build_specimen_features

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "processed" / "all_specimens.parquet"

RNG = np.random.default_rng(0)
INPUT_COLS = ["FCR_pct", "deflection_mm", "R_ohm", "load_kN"]


@pytest.fixture(scope="module")
def specimen() -> pd.DataFrame:
    if not PARQUET.exists():
        pytest.skip("run `make pipeline` first to create all_specimens.parquet")
    df = pd.read_parquet(PARQUET)
    return df[df["specimen"] == "S1"].reset_index(drop=True)


@pytest.mark.parametrize("t0_frac", [0.3, 0.6, 0.85])
def test_features_are_causal(specimen: pd.DataFrame, t0_frac: float) -> None:
    n = len(specimen)
    t0 = int(n * t0_frac)
    lr, r0 = 0.017, 5.0e7  # fixed known constants

    base = build_specimen_features(specimen, lr, r0)

    corrupted = specimen.copy()
    for c in INPUT_COLS:
        col = corrupted[c].to_numpy().copy()
        col[t0 + 1 :] = RNG.normal(size=n - t0 - 1) * col.std() + col.mean()
        corrupted[c] = col
    after = build_specimen_features(corrupted, lr, r0)

    for c in FEATURE_COLUMNS:
        np.testing.assert_array_equal(
            base[c].to_numpy()[: t0 + 1],
            after[c].to_numpy()[: t0 + 1],
            err_msg=f"feature '{c}' leaks future information at t0={t0}",
        )
