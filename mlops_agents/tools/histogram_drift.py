"""
mlops_agents/tools/histogram_drift.py

Histogram drift detection exposed as a LangChain @tool.

The diagnosis agent gets reference (training) and production (live) histograms
per feature. Comparing them in prose with a small LLM is unreliable. This tool
does the maths in Python and returns a structured drift summary the LLM can
just read.

Metrics computed per feature:
  - PSI  (Population Stability Index) — the industry-standard drift metric
  - KS   (Kolmogorov–Smirnov)         — max |CDF_ref − CDF_prod|
  - mean_shift_z = (mu_prod − mu_ref) / sigma_ref — directional signal

PSI bands (per Siddiqi's standard cut-offs):
  < 0.10              → stable
  0.10 ≤ PSI < 0.25   → moderate drift
  ≥ 0.25              → significant drift

Reference and production rarely share bin_edges, so reference is re-binned
onto production's edges via empirical-CDF interpolation before PSI is computed.

Input shape (per feature):
    {"counts": [...], "bin_edges": [...], "mean": float, "std": float}

(Extra keys are tolerated and ignored.)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPSILON = 1e-6            # smoothing to avoid log(0) / div-by-zero in PSI
_PSI_MODERATE = 0.10
_PSI_SIGNIFICANT = 0.25
_TOP_N = 5                 # how many top-drifted features to highlight


# ---------------------------------------------------------------------------
# Core math (pure NumPy — no LangChain wrapper, easier to unit-test)
# ---------------------------------------------------------------------------

def _drift_level(psi: float) -> str:
    if psi >= _PSI_SIGNIFICANT:
        return "significant"
    if psi >= _PSI_MODERATE:
        return "moderate"
    return "stable"


def _empirical_pmf(counts: np.ndarray) -> np.ndarray:
    """Counts → probability mass function, with epsilon smoothing."""
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, _EPSILON, dtype=float)
    pmf = counts.astype(float) / total
    return np.where(pmf <= 0, _EPSILON, pmf)


def _cdf_at(edges: np.ndarray, cum_probs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Linear-interp the empirical CDF (defined at right edges of bins) at x.

    `cum_probs` is the cumulative probability AT each bin's right edge.
    We left-pad with (left_edge, 0) so the CDF is well-defined on the full
    range. x values outside the range are clipped to [0, 1].
    """
    xp = np.concatenate(([edges[0]], edges[1:]))
    fp = np.concatenate(([0.0], cum_probs))
    return np.interp(x, xp, fp, left=0.0, right=1.0)


def _align_reference_to_production(
    ref_counts: np.ndarray, ref_edges: np.ndarray,
    prod_edges: np.ndarray,
) -> np.ndarray:
    """
    Re-bin reference's PMF onto production's bin edges via CDF interpolation,
    so PSI can be computed bin-for-bin against production.
    """
    ref_pmf = _empirical_pmf(ref_counts)
    ref_cum = np.cumsum(ref_pmf)
    # CDF at production's right edges (i.e. edges[1:])
    cdf_at_prod_right = _cdf_at(ref_edges, ref_cum, prod_edges[1:])
    cdf_at_prod_left = _cdf_at(ref_edges, ref_cum, prod_edges[:-1])
    aligned_pmf = cdf_at_prod_right - cdf_at_prod_left
    aligned_pmf = np.clip(aligned_pmf, _EPSILON, None)
    aligned_pmf = aligned_pmf / aligned_pmf.sum()  # renormalise
    return aligned_pmf


def _psi(ref_aligned_pmf: np.ndarray, prod_pmf: np.ndarray) -> float:
    """Population Stability Index — symmetric KL-like divergence."""
    return float(np.sum((prod_pmf - ref_aligned_pmf) * np.log(prod_pmf / ref_aligned_pmf)))


def _ks(
    ref_edges: np.ndarray, ref_counts: np.ndarray,
    prod_edges: np.ndarray, prod_counts: np.ndarray,
) -> float:
    """Kolmogorov–Smirnov statistic on the empirical CDFs."""
    ref_cum = np.cumsum(_empirical_pmf(ref_counts))
    prod_cum = np.cumsum(_empirical_pmf(prod_counts))
    sample_points = np.unique(np.concatenate([ref_edges, prod_edges]))
    diff = np.abs(_cdf_at(ref_edges, ref_cum, sample_points) -
                  _cdf_at(prod_edges, prod_cum, sample_points))
    return float(diff.max())


def _feature_drift(ref: dict, prod: dict) -> dict:
    ref_counts = np.asarray(ref["counts"], dtype=float)
    ref_edges = np.asarray(ref["bin_edges"], dtype=float)
    prod_counts = np.asarray(prod["counts"], dtype=float)
    prod_edges = np.asarray(prod["bin_edges"], dtype=float)

    prod_pmf = _empirical_pmf(prod_counts)
    ref_aligned = _align_reference_to_production(ref_counts, ref_edges, prod_edges)
    psi_val = _psi(ref_aligned, prod_pmf)
    ks_val = _ks(ref_edges, ref_counts, prod_edges, prod_counts)

    ref_mean = float(ref.get("mean", 0.0))
    ref_std = float(ref.get("std", 0.0))
    prod_mean = float(prod.get("mean", 0.0))
    mean_shift_z = (prod_mean - ref_mean) / ref_std if ref_std > 0 else 0.0

    return {
        "psi": round(psi_val, 4),
        "ks": round(ks_val, 4),
        "mean_shift_z": round(mean_shift_z, 4),
        "drift_level": _drift_level(psi_val),
    }


def _compute(reference: dict, production: dict) -> dict:
    """Pure-Python entry point — also callable without the @tool wrapper."""
    per_feature: dict[str, dict] = {}
    skipped: list[str] = []

    common = sorted(set(reference) & set(production))
    for feat in common:
        ref_f, prod_f = reference[feat], production[feat]
        if not all(k in ref_f and k in prod_f for k in ("counts", "bin_edges")):
            skipped.append(feat)
            continue
        try:
            per_feature[feat] = _feature_drift(ref_f, prod_f)
        except Exception as exc:  # noqa: BLE001 — never let a bad bin kill the run
            skipped.append(f"{feat} ({exc})")

    drifted = sorted(
        [f for f, m in per_feature.items() if m["psi"] >= _PSI_MODERATE],
        key=lambda f: per_feature[f]["psi"],
        reverse=True,
    )
    top_drifted = [
        {"feature": f, **per_feature[f]}
        for f in drifted[:_TOP_N]
    ]

    sig_count = sum(1 for m in per_feature.values() if m["drift_level"] == "significant")
    mod_count = sum(1 for m in per_feature.values() if m["drift_level"] == "moderate")
    summary = (
        f"{len(per_feature)} features compared — "
        f"{sig_count} significant, {mod_count} moderate, "
        f"{len(per_feature) - sig_count - mod_count} stable."
    )
    if top_drifted:
        head = top_drifted[0]
        summary += f" Most-drifted: {head['feature']} (PSI={head['psi']:.3f}, KS={head['ks']:.3f})."

    return {
        "per_feature": per_feature,
        "drifted_features": drifted,
        "top_drifted": top_drifted,
        "summary": summary,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# LangChain tool wrapper
# ---------------------------------------------------------------------------

@tool
def compute_histogram_drift(
    reference: dict[str, dict[str, Any]],
    production: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Quantify drift between reference (training) and production histograms.

    For each feature present in BOTH inputs, computes:
      - PSI (Population Stability Index) — main drift signal
      - KS  (Kolmogorov–Smirnov)         — distribution-shape divergence
      - mean_shift_z                     — (prod.mean − ref.mean) / ref.std

    Args:
        reference:  {feature_name: {"counts": [...], "bin_edges": [...],
                                    "mean": float, "std": float}, ...}
        production: same shape as `reference`.

    Returns:
        {
            "per_feature":      {feat: {psi, ks, mean_shift_z, drift_level}, ...},
            "drifted_features": [features with PSI ≥ 0.10, ordered by PSI desc],
            "top_drifted":      [top 5 with their metrics],
            "summary":          short human-readable line,
            "skipped":          features that couldn't be compared,
        }
    """
    if not reference or not production:
        return {
            "per_feature": {}, "drifted_features": [], "top_drifted": [],
            "summary": "Drift comparison unavailable — missing reference or production histograms.",
            "skipped": [],
        }
    return _compute(reference, production)
