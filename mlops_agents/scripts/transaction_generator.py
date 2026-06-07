"""
scripts/transaction_generator.py

Replays rows from a named dataset CSV against the model server's /predict
endpoint in a continuous loop.  Supports hot-swapping the active dataset
at runtime via a shared data/active_dataset.json file so the Drift Lab
can switch scenarios without restarting the process.

Key changes from original:
  • --dataset argument selects which CSV to load from data/datasets/
  • polls data/active_dataset.json every 30 s; reloads rows when it changes
  • computes per-batch PSI against the baseline distribution after every
    100 predictions — real drift score, not confidence-variance proxy
  • --loop (default) and --once (--no-loop) modes

Usage:
    # continuous loop on concept_drift dataset, 2 req/s
    python scripts/transaction_generator.py --dataset concept_drift

    # faster seed, then exit
    python scripts/transaction_generator.py --dataset baseline --rate 10 --count 200 --no-loop

    # inject malformed requests to drive up error_rate
    python scripts/transaction_generator.py --dataset baseline --error-rate 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_SERVER     = os.getenv("FRAUD_MODEL_MCP_URL", "http://localhost:8080")
PARENT_DIR       = Path(__file__).parent.parent
DATASETS_DIR     = PARENT_DIR / "data" / "datasets"
DEFAULT_CSV      = PARENT_DIR / "data" / "creditcard.csv"
ACTIVE_FILE      = PARENT_DIR / "data" / "active_dataset.json"
BASELINE_CSV     = DATASETS_DIR / "baseline.csv"

HOT_SWAP_INTERVAL = 30  # seconds between polls of active_dataset.json
PSI_BATCH_SIZE    = 100  # compute PSI after this many predictions


# ── PSI helper ────────────────────────────────────────────────────────────────

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two 1-D arrays.
    PSI < 0.1  → no significant drift
    PSI < 0.25 → moderate drift
    PSI ≥ 0.25 → significant drift
    """
    eps = 1e-6
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    edges   = np.linspace(min_val, max_val, bins + 1)

    exp_counts = np.histogram(expected, bins=edges)[0].astype(float) + eps
    act_counts = np.histogram(actual,   bins=edges)[0].astype(float) + eps

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _compute_psi_for_batch(
    batch_rows: list[dict],
    baseline_df: pd.DataFrame,
    features: list[str],
) -> float:
    """Compute mean PSI across selected features for a batch of rows."""
    psi_values = []
    for feat in features:
        col = feat.lower()  # payload keys are lowercase (v1, v2, …)
        if col not in batch_rows[0]:
            continue
        actual_vals   = np.array([r[col] for r in batch_rows], dtype=float)
        expected_vals = baseline_df[feat].values.astype(float)
        psi_values.append(_psi(expected_vals, actual_vals))
    return float(np.mean(psi_values)) if psi_values else 0.0


# ── dataset loading ───────────────────────────────────────────────────────────

def _resolve_dataset_path(dataset: str) -> Path:
    """
    Resolve dataset name to a CSV path.
    Falls back to DEFAULT_CSV if the datasets directory doesn't exist yet.
    """
    candidate = DATASETS_DIR / f"{dataset}.csv"
    if candidate.exists():
        return candidate
    if DEFAULT_CSV.exists():
        print(f"  [WARN] {candidate} not found — falling back to {DEFAULT_CSV}")
        return DEFAULT_CSV
    print(f"ERROR: neither {candidate} nor {DEFAULT_CSV} exist.")
    print("Run POST /datasets/create or download creditcard.csv first.")
    sys.exit(1)


def load_rows(csv_path: Path, n: int = 500) -> tuple[list[dict], pd.DataFrame]:
    """
    Load up to n stratified rows from csv_path.
    Returns (rows_for_predict, raw_dataframe_for_psi_baseline).
    """
    df = pd.read_csv(csv_path)

    sample = (
    df.sample(n=n, random_state=42)
      .reset_index(drop=True)
    )

    rows = []
    for _, row in sample.iterrows():
        payload = {f"v{i}": float(row[f"V{i}"]) for i in range(1, 29)}
        payload["amount"]     = float(row["Amount"])
        payload["time"]       = float(row["Time"])
        payload["_true_label"] = int(row["Class"])
        rows.append(payload)

    return rows, sample


def _load_baseline_df() -> pd.DataFrame:
    """Load the baseline CSV for PSI reference. Uses creditcard.csv as fallback."""
    if BASELINE_CSV.exists():
        return pd.read_csv(BASELINE_CSV)
    if DEFAULT_CSV.exists():
        return pd.read_csv(DEFAULT_CSV)
    return pd.DataFrame()


def _read_active_dataset() -> str | None:
    try:
        if ACTIVE_FILE.exists():
            with open(ACTIVE_FILE) as f:
                return json.load(f).get("dataset")
    except Exception:
        pass
    return None


# ── malformed payload ─────────────────────────────────────────────────────────

def make_malformed_payload() -> dict:
    mode = random.choice(["missing_field", "nan_value", "negative_amount", "wrong_type"])
    base = {f"v{i}": random.gauss(0, 1) for i in range(1, 29)}
    base["amount"] = 100.0
    base["time"]   = 0.0

    if mode == "missing_field":
        del base["v28"]
    elif mode == "nan_value":
        base["v1"] = float("nan")
    elif mode == "negative_amount":
        base["amount"] = -999.0
    elif mode == "wrong_type":
        base["v1"] = "not_a_number"

    return base


# ── main loop ─────────────────────────────────────────────────────────────────

PSI_FEATURES = [f"V{i}" for i in range(1, 15)]  # first 14 PCA components


def run(
    dataset:    str   = "baseline",
    rate:       float = 2.0,
    count:      int   = 0,
    loop:       bool  = True,
    error_rate: float = 0.0,
    verbose:    bool  = True,
    seed_n:     int   = 500,
):
    global MODEL_SERVER

    csv_path = _resolve_dataset_path(dataset)
    current_dataset = dataset

    print(f"Loading {seed_n} rows from {csv_path} …")
    rows, sample_df = load_rows(csv_path, n=seed_n)
    print(f"Loaded {len(rows)} rows ({sum(1 for r in rows if r['_true_label']==1)} fraud)")

    baseline_df = _load_baseline_df()
    if baseline_df.empty:
        print("  [WARN] No baseline CSV found — PSI will be skipped")

    print(f"Sending to {MODEL_SERVER}/predict at {rate} req/s")
    if error_rate > 0:
        print(f"Injecting malformed requests at {error_rate:.0%} rate")
    print("─" * 60)

    interval      = 1.0 / rate
    sent          = 0
    errors        = 0
    fraud_hits    = 0
    batch_buffer: list[dict] = []   # accumulate rows for PSI
    last_swap_check = time.time()

    while True:
        for row in rows:
            # ── hot-swap check ──────────────────────────────────────────────
            if time.time() - last_swap_check >= HOT_SWAP_INTERVAL:
                active = _read_active_dataset()
                last_swap_check = time.time()
                if active and active != current_dataset:
                    new_path = _resolve_dataset_path(active)
                    print(f"\n  ↺ Hot-swap: {current_dataset} → {active}")
                    rows, sample_df  = load_rows(new_path, n=seed_n)
                    current_dataset  = active
                    batch_buffer     = []
                    print(f"  Loaded {len(rows)} rows from {new_path}\n")
                    break   # restart loop over new rows

            # ── decide whether to send a malformed request ──────────────────
            is_malformed = random.random() < error_rate

            if is_malformed:
                payload = make_malformed_payload()
            else:
                # Forward the row verbatim including _true_label so the model
                # server can compute real precision/recall/accuracy. The server
                # pops _true_label from the payload before feature extraction.
                payload = dict(row)

            try:
                r = requests.post(
                    f"{MODEL_SERVER}/predict",
                    json=payload,
                    timeout=5,
                )

                if is_malformed:
                    errors += 1
                    if verbose:
                        print(f"  [ERROR] malformed → {r.status_code}")
                else:
                    if r.ok:
                        result     = r.json()
                        fraud_prob = result.get("fraud_prob", 0)
                        prediction = result.get("prediction", 0)
                        true_label = row["_true_label"]
                        latency    = result.get("latency_ms", 0)

                        if prediction == 1:
                            fraud_hits += 1

                        batch_buffer.append(payload)

                        if verbose and sent % 20 == 0:
                            correct   = "✓" if prediction == true_label else "✗"
                            label_str = "FRAUD" if true_label == 1 else "legit"
                            print(
                                f"  [{sent:>4}] {label_str:<5} "
                                f"pred={prediction} prob={fraud_prob:.3f} "
                                f"lat={latency:.1f}ms {correct}"
                            )

                        # ── PSI batch logging ────────────────────────────────
                        if len(batch_buffer) >= PSI_BATCH_SIZE and not baseline_df.empty:
                            psi = _compute_psi_for_batch(
                                batch_buffer, baseline_df, PSI_FEATURES
                            )
                            label = (
                                "SIGNIFICANT" if psi >= 0.25
                                else "moderate"  if psi >= 0.1
                                else "stable"
                            )
                            print(
                                f"  [PSI]  batch={len(batch_buffer)} "
                                f"psi={psi:.4f} → {label} "
                                f"(dataset={current_dataset})"
                            )
                            batch_buffer = []

                    else:
                        errors += 1
                        if verbose:
                            print(f"  [ERROR] {r.status_code} — {r.text[:80]}")

            except requests.exceptions.ConnectionError:
                print(f"  [FATAL] Cannot reach {MODEL_SERVER} — is the model server running?")
                time.sleep(5)
                continue
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  [ERROR] {e}")

            sent += 1
            if count > 0 and sent >= count:
                print(f"\nDone. sent={sent} errors={errors} fraud_hits={fraud_hits}")
                return

            time.sleep(interval)

        else:
            # inner for-loop completed without a break (no hot-swap)
            if not loop:
                print(f"\nDone. sent={sent} errors={errors} fraud_hits={fraud_hits}")
                return
            if verbose:
                print(f"\n  ↻ replaying from start (sent={sent} errors={errors})\n")
            continue

        # Reached here only via break (hot-swap occurred) — continue outer while


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global MODEL_SERVER

    parser = argparse.ArgumentParser(
        description="Replay dataset rows against the fraud model server"
    )
    parser.add_argument(
        "--dataset", default="baseline",
        help="Dataset name to load from data/datasets/ (default: baseline). "
             "Falls back to creditcard.csv if the datasets dir doesn't exist yet.",
    )
    parser.add_argument("--rate",       type=float, default=2.0,
                        help="Requests per second (default: 2.0)")
    parser.add_argument("--count",      type=int,   default=0,
                        help="Stop after N requests; 0 = unlimited (default: 0)")
    parser.add_argument("--no-loop",    action="store_true",
                        help="Exit when all rows have been sent once")
    parser.add_argument("--error-rate", type=float, default=0.0,
                        help="Fraction of malformed requests to inject (default: 0.0)")
    parser.add_argument("--quiet",      action="store_true",
                        help="Suppress per-request output")
    parser.add_argument("--seed-n",     type=int,   default=500,
                        help="Number of rows to load from CSV (default: 500)")
    parser.add_argument("--server",     default=MODEL_SERVER,
                        help=f"Model server URL (default: {MODEL_SERVER})")

    args = parser.parse_args()
    MODEL_SERVER = args.server

    run(
        dataset    = args.dataset,
        rate       = args.rate,
        count      = args.count,
        loop       = not args.no_loop,
        error_rate = args.error_rate,
        verbose    = not args.quiet,
        seed_n     = args.seed_n,
    )


if __name__ == "__main__":
    main()