"""
scripts/dataset_generator.py

Creates synthetic drifted datasets from the Kaggle creditcard.csv reference.
Each scenario produces a CSV on disk that transaction_generator.py can load
and replay, alongside a metadata JSON file describing the scenario.

Scenarios:
    baseline            — original Kaggle CSV, no modification
    data_drift_amount   — Amount distribution shifted right (×3.5 for top 30%)
    data_drift_features — Selected V-features shifted by N(2.5, 0.5) noise
    concept_drift       — V14 and V4 swapped in fraud rows only (recall collapses)
    mixed_drift         — feature shift + label relationship swap
    gradual_drift       — N intermediate CSVs between baseline and full drift

Output layout:
    data/datasets/
        baseline.csv  /  baseline.json
        data_drift_amount.csv  /  data_drift_amount.json
        data_drift_features.csv  /  data_drift_features.json
        concept_drift.csv  /  concept_drift.json
        mixed_drift.csv  /  mixed_drift.json
        gradual_drift_step_1.csv  /  gradual_drift_step_1.json
        gradual_drift_step_2.csv  /  gradual_drift_step_2.json
        gradual_drift_step_3.csv  /  gradual_drift_step_3.json

Usage:
    python scripts/dataset_generator.py
    python scripts/dataset_generator.py --steps 5   # more gradual steps
    python scripts/dataset_generator.py --data-path ./data/creditcard.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PARENT_DIR = Path(__file__).parent.parent
DATA_PATH  = PARENT_DIR / "data" / "creditcard.csv"
OUT_DIR    = PARENT_DIR / "data" / "datasets"

GRADUAL_STEPS = 3   # default number of gradual-drift intermediates


# ── helpers ───────────────────────────────────────────────────────────────────

def _fraud_rate(df: pd.DataFrame) -> float:
    return float(df["Class"].mean())


def _save(df: pd.DataFrame, name: str, meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = OUT_DIR / f"{name}.csv"
    json_path = OUT_DIR / f"{name}.json"

    df.to_csv(csv_path, index=False)

    meta.update({
        "name":       name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source":     "creditcard.csv",
        "rows":       len(df),
        "fraud_rate": _fraud_rate(df),
    })
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ {name}.csv  ({len(df):,} rows, fraud_rate={_fraud_rate(df):.4f})")


# ── scenario functions ────────────────────────────────────────────────────────

def make_baseline(df: pd.DataFrame) -> None:
    """No change — original Kaggle CSV."""
    _save(df.copy(), "baseline", {
        "drift_type":        "none",
        "description":       "Original Kaggle creditcard.csv — no modification",
        "features_modified": [],
        "expected_severity": "none",
        "expected_action":   "none",
        "expected_strategy": "none",
    })


def make_data_drift_amount(df: pd.DataFrame) -> None:
    """
    Amount distribution shifted right: top 30% of Amount values multiplied by 3.5.
    The model sees unusually high-value transactions it wasn't trained on.
    """
    out = df.copy()
    threshold = out["Amount"].quantile(0.70)
    mask      = out["Amount"] > threshold
    out.loc[mask, "Amount"] = out.loc[mask, "Amount"] * 3.5

    _save(out, "data_drift_amount", {
        "drift_type":        "data_drift",
        "description":       "Amount distribution shifted right — top 30% multiplied by 3.5",
        "features_modified": ["Amount"],
        "magnitude":         3.5,
        "expected_severity": "minor",
        "expected_action":   "monitor",
        "expected_strategy": "recent_window",
    })


def make_data_drift_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
    mean_shift: float = 2.5,
    noise_std:  float = 0.5,
) -> None:
    """
    Selected V-features shifted by adding N(mean_shift, noise_std) noise.
    Model uncertainty rises; accuracy degrades.
    """
    if features is None:
        features = ["V14", "V17", "V12"]

    out  = df.copy()
    rng  = np.random.default_rng(42)
    for feat in features:
        if feat in out.columns:
            out[feat] = out[feat] + rng.normal(mean_shift, noise_std, size=len(out))

    _save(out, "data_drift_features", {
        "drift_type":        "data_drift",
        "description":       f"V-features {features} shifted by N({mean_shift},{noise_std})",
        "features_modified": features,
        "mean_shift":        mean_shift,
        "noise_std":         noise_std,
        "expected_severity": "major",
        "expected_action":   "retrain",
        "expected_strategy": "drift_period_only",
    })


def make_concept_drift(df: pd.DataFrame) -> None:
    """
    Swap V14 and V4 values across all fraud rows only.

    Important: this only swaps values within fraud rows — legitimate transactions
    look identical to training data. This correctly simulates concept drift:
    fraud now appears in a region of feature space the model never associated
    with fraud during training. Recall collapses. Precision stays high.
    """
    out        = df.copy()
    fraud_mask = out["Class"] == 1

    v14_vals = out.loc[fraud_mask, "V14"].values.copy()
    v4_vals  = out.loc[fraud_mask, "V4"].values.copy()
    out.loc[fraud_mask, "V14"] = v4_vals
    out.loc[fraud_mask, "V4"]  = v14_vals

    _save(out, "concept_drift", {
        "drift_type":        "concept_drift",
        "description":       "V14 and V4 swapped in fraud rows only — recall collapses",
        "features_modified": ["V14", "V4"],
        "expected_severity": "critical",
        "expected_action":   "retrain",
        "expected_strategy": "drift_period_only",
        "drift_dataset":     "concept_drift",
    })


def make_mixed_drift(
    df: pd.DataFrame,
    shift_features: list[str] | None = None,
    mean_shift: float = 2.0,
    noise_std:  float = 0.4,
    corruption_rate: float = 0.3,
) -> None:
    """
    Feature shift + label relationship swap in a fraction of fraud rows.
    All metrics degrade simultaneously.
    """
    if shift_features is None:
        shift_features = ["V14", "V17", "V12", "V4"]

    out = df.copy()
    rng = np.random.default_rng(42)

    # data drift component: add noise to selected features across all rows
    for feat in shift_features:
        if feat in out.columns:
            out[feat] = out[feat] + rng.normal(mean_shift, noise_std, size=len(out))

    # concept drift component: swap V14 and V4 in a fraction of fraud rows
    fraud_idx = out.index[out["Class"] == 1].tolist()
    n_corrupt = max(1, int(len(fraud_idx) * corruption_rate))
    corrupt_idx = rng.choice(fraud_idx, size=n_corrupt, replace=False)

    v14_vals = out.loc[corrupt_idx, "V14"].values.copy()
    v4_vals  = out.loc[corrupt_idx, "V4"].values.copy()
    out.loc[corrupt_idx, "V14"] = v4_vals
    out.loc[corrupt_idx, "V4"]  = v14_vals

    _save(out, "mixed_drift", {
        "drift_type":        "mixed",
        "description":       f"Feature shift ({shift_features}) + {int(corruption_rate*100)}% fraud rows concept-drifted",
        "features_modified": shift_features,
        "mean_shift":        mean_shift,
        "corruption_rate":   corruption_rate,
        "expected_severity": "critical",
        "expected_action":   "retrain",
        "expected_strategy": "drift_period_only",
        "drift_dataset":     "mixed_drift",
    })


def make_gradual_drift(df: pd.DataFrame, steps: int = GRADUAL_STEPS) -> None:
    """
    Produces `steps` intermediate CSVs between baseline and full concept drift.
    Each step swaps a larger fraction of fraud rows (1/steps, 2/steps, …, steps/steps).
    """
    fraud_idx = df.index[df["Class"] == 1].tolist()
    rng       = np.random.default_rng(42)

    for step in range(1, steps + 1):
        fraction    = step / steps
        n_swap      = max(1, int(len(fraud_idx) * fraction))
        swap_idx    = rng.choice(fraud_idx, size=n_swap, replace=False)

        out = df.copy()
        v14_vals = out.loc[swap_idx, "V14"].values.copy()
        v4_vals  = out.loc[swap_idx, "V4"].values.copy()
        out.loc[swap_idx, "V14"] = v4_vals
        out.loc[swap_idx, "V4"]  = v14_vals

        severity = "minor" if fraction < 0.35 else ("major" if fraction < 0.70 else "critical")
        name = f"gradual_drift_step_{step}"

        _save(out, name, {
            "drift_type":        "concept_drift",
            "description":       f"Gradual drift step {step}/{steps} — {int(fraction*100)}% fraud rows concept-drifted",
            "features_modified": ["V14", "V4"],
            "fraction":          fraction,
            "step":              step,
            "total_steps":       steps,
            "expected_severity": severity,
            "expected_action":   "retrain" if fraction >= 0.5 else "monitor",
            "expected_strategy": "drift_period_only",
        })


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(
        description="Generate drifted dataset CSVs from creditcard.csv"
    )
    parser.add_argument("--data-path", default=str(DATA_PATH),
                        help=f"Path to creditcard.csv (default: {DATA_PATH})")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--steps", type=int, default=GRADUAL_STEPS,
                        help=f"Number of gradual-drift steps (default: {GRADUAL_STEPS})")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir   = Path(args.out_dir)

    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        print("Download from: kaggle datasets download mlg-ulb/creditcardfraud -p ./data --unzip")
        sys.exit(1)

    OUT_DIR = out_dir

    print(f"Loading {data_path} …")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows  (fraud_rate={_fraud_rate(df):.4f})")
    print(f"Writing datasets to {out_dir}/\n")

    make_baseline(df)
    make_data_drift_amount(df)
    make_data_drift_features(df)
    make_concept_drift(df)
    make_mixed_drift(df)
    make_gradual_drift(df, steps=args.steps)

    print(f"\nDone — {len(list(out_dir.glob('*.csv')))} CSVs in {out_dir}/")


if __name__ == "__main__":
    main()