"""
scripts/transaction_generator.py

Replays rows from the Kaggle creditcard.csv against the model server's
/predict endpoint in a continuous loop. This is the missing piece that
makes Drift Lab metrics real — without it, prediction_history stays
empty and compute_current_metrics() falls back to stale training values.

Usage:
    # continuous loop, 2 requests/sec
    python scripts/transaction_generator.py

    # faster, for quickly building up prediction history
    python scripts/transaction_generator.py --rate 10 --count 200

    # run once and exit (useful for seeding before a demo)
    python scripts/transaction_generator.py --count 100 --no-loop

    # inject malformed requests to drive up error_rate
    python scripts/transaction_generator.py --error-rate 0.1
"""

from __future__ import annotations

import argparse
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

MODEL_SERVER = os.getenv("FRAUD_MODEL_MCP_URL", "http://localhost:8080")
PARENT_DIR     = Path(__file__).parent.parent
DATA_PATH    = PARENT_DIR / "data" / "creditcard.csv"


def load_test_rows(data_path: Path, n: int = 500) -> list[dict]:
    """
    Load N rows from creditcard.csv and convert to /predict payload format.
    Stratifies to include both fraud and legitimate transactions.
    """
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        print("Download from: kaggle datasets download mlg-ulb/creditcardfraud -p ./data --unzip")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # stratified sample — keep realistic fraud rate
    fraud  = df[df["Class"] == 1].sample(min(len(df[df["Class"]==1]), n // 6), random_state=42)
    legit  = df[df["Class"] == 0].sample(n - len(fraud), random_state=42)
    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)

    rows = []
    for _, row in sample.iterrows():
        payload = {f"v{i}": float(row[f"V{i}"]) for i in range(1, 29)}
        payload["amount"] = float(row["Amount"])
        payload["time"]   = float(row["Time"])
        payload["_true_label"] = int(row["Class"])  # for local tracking only, not sent
        rows.append(payload)

    return rows


def make_malformed_payload() -> dict:
    """Returns a payload that will fail Pydantic validation — drives up error_rate."""
    mode = random.choice(["missing_field", "nan_value", "negative_amount", "wrong_type"])
    base = {f"v{i}": random.gauss(0, 1) for i in range(1, 29)}
    base["amount"] = 100.0
    base["time"]   = 0.0

    if mode == "missing_field":
        del base["v28"]                  # missing required field
    elif mode == "nan_value":
        base["v1"] = float("nan")        # NaN in feature
    elif mode == "negative_amount":
        base["amount"] = -999.0          # violates ge=0 constraint
    elif mode == "wrong_type":
        base["v1"] = "not_a_number"      # wrong type

    return base


def run(
    rate: float       = 2.0,
    count: int        = 0,
    loop: bool        = True,
    error_rate: float = 0.0,
    verbose: bool     = True,
    data_path: Path   = DATA_PATH,
    seed_n: int       = 500,
):
    """
    Main generator loop.

    Args:
        rate:       requests per second
        count:      stop after N requests (0 = unlimited)
        loop:       restart from beginning when rows exhausted
        error_rate: fraction of requests that are intentionally malformed
        verbose:    print per-request results
        data_path:  path to creditcard.csv
        seed_n:     number of rows to load from CSV
    """
    print(f"Loading {seed_n} rows from {data_path}...")
    rows = load_test_rows(data_path, n=seed_n)
    print(f"Loaded {len(rows)} rows ({sum(1 for r in rows if r['_true_label']==1)} fraud)")
    print(f"Sending to {MODEL_SERVER}/predict at {rate} req/s")
    if error_rate > 0:
        print(f"Injecting malformed requests at {error_rate:.0%} rate")
    print("─" * 50)

    interval   = 1.0 / rate
    sent       = 0
    errors     = 0
    fraud_hits = 0

    while True:
        for row in rows:
            # decide whether to send a malformed request
            is_malformed = random.random() < error_rate

            if is_malformed:
                payload = make_malformed_payload()
            else:
                payload = {k: v for k, v in row.items() if k != "_true_label"}

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
                        result      = r.json()
                        fraud_prob  = result.get("fraud_prob", 0)
                        prediction  = result.get("prediction", 0)
                        true_label  = row["_true_label"]
                        latency     = result.get("latency_ms", 0)

                        if prediction == 1:
                            fraud_hits += 1

                        if verbose and sent % 20 == 0:
                            correct = "✓" if prediction == true_label else "✗"
                            label_str = "FRAUD" if true_label == 1 else "legit"
                            print(
                                f"  [{sent:>4}] {label_str:<5} "
                                f"pred={prediction} prob={fraud_prob:.3f} "
                                f"lat={latency:.1f}ms {correct}"
                            )
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

        if not loop:
            print(f"\nDone. sent={sent} errors={errors} fraud_hits={fraud_hits}")
            return

        if verbose:
            print(f"\n  ↻ replaying from start (sent={sent} errors={errors})\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global MODEL_SERVER
    
    parser = argparse.ArgumentParser(
        description="Replay Kaggle creditcard rows against the fraud model server"
    )
    parser.add_argument("--rate",       type=float, default=2.0,
                        help="Requests per second (default: 2.0)")
    parser.add_argument("--count",      type=int,   default=0,
                        help="Stop after N requests. 0 = unlimited (default: 0)")
    parser.add_argument("--no-loop",    action="store_true",
                        help="Exit when all rows have been sent once")
    parser.add_argument("--error-rate", type=float, default=0.0,
                        help="Fraction of malformed requests to send (default: 0.0)")
    parser.add_argument("--quiet",      action="store_true",
                        help="Suppress per-request output")
    parser.add_argument("--seed-n",     type=int,   default=500,
                        help="Number of rows to load from CSV (default: 500)")
    parser.add_argument("--data-path",  default=str(DATA_PATH),
                        help=f"Path to creditcard.csv (default: {DATA_PATH})")
    parser.add_argument("--server",     default=MODEL_SERVER,
                        help=f"Model server URL (default: {MODEL_SERVER})")

    args = parser.parse_args()

    MODEL_SERVER = args.server

    run(
        rate       = args.rate,
        count      = args.count,
        loop       = not args.no_loop,
        error_rate = args.error_rate,
        verbose    = not args.quiet,
        data_path  = Path(args.data_path),
        seed_n     = args.seed_n,
    )


if __name__ == "__main__":
    main()