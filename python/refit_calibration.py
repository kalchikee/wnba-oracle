#!/usr/bin/env python3
"""Refit WNBA isotonic calibration using in-season graded picks.

The existing data/model/calibration.json was fit from training-set backtest
(2015-2024). After ~57 graded picks of 2026 play we have a real production
signal: the 80%+ bucket hit only 33% (n=6) while 60-65% hit 77% (n=13).
This script blends the training-fit calibration with the in-season
empirical hit rate using a Gaussian-kernel weighted shrinkage, so heavily-
sampled buckets pull toward the data and sparse buckets stay anchored
to the prior.

Usage:
    python python/refit_calibration.py [--dry-run] [--prior STRENGTH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).parent.parent
HISTORY = ROOT / "data" / "grading_history.json"
CALIB_PATH = ROOT / "data" / "model" / "calibration.json"


def load_pairs(history_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (home_prob_predicted, home_won) pairs from graded picks.

    Predictions JSON stores modelProb for the PICKED team; convert to a
    consistent home-side framing so the isotonic fit is symmetric over
    [0, 1] (matches the training-fit calibration's framing)."""
    with open(history_path) as f:
        h = json.load(f)
    xs, ys = [], []
    for r in h.get("graded", []):
        p_pick = float(r.get("modelProb") or 0)
        if p_pick == 0:
            continue
        won = 1 if r.get("correct") else 0
        picked_home = r.get("pickedTeam") == r.get("home")
        if picked_home:
            p_home, home_won = p_pick, won
        else:
            p_home, home_won = 1.0 - p_pick, 1 - won
        xs.append(p_home)
        ys.append(home_won)
    return np.array(xs), np.array(ys)


def refit(
    xs: np.ndarray,
    ys: np.ndarray,
    old_x: np.ndarray,
    old_y: np.ndarray,
    prior_strength: float = 12.0,
    kernel_width: float = 0.05,
) -> np.ndarray:
    """Blend an in-season isotonic fit with the training-fit calibration.

    At each x_threshold from the old calibration, compute the effective
    sample size (ESS) of nearby graded picks via a Gaussian kernel, then
    weighted-average the in-season isotonic prediction against the old
    y_threshold using ESS vs prior_strength. Returns a monotonic y array
    of the same length as old_x."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(xs, ys)
    data_y = ir.predict(old_x)
    ess = np.array([
        np.exp(-((xs - x) ** 2) / (2.0 * kernel_width ** 2)).sum()
        for x in old_x
    ])
    blended = (ess * data_y + prior_strength * old_y) / (ess + prior_strength)
    # Enforce monotonic non-decreasing
    for i in range(1, len(blended)):
        if blended[i] < blended[i - 1]:
            blended[i] = blended[i - 1]
    return blended


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print deltas without writing calibration.json")
    parser.add_argument("--prior", type=float, default=12.0,
                        help="Prior strength (higher = trust old fit more)")
    parser.add_argument("--kernel-width", type=float, default=0.05,
                        help="Gaussian kernel width for ESS calculation")
    args = parser.parse_args()

    xs, ys = load_pairs(HISTORY)
    if len(xs) == 0:
        print("[refit] No graded picks in history — nothing to refit.")
        return 1
    print(f"[refit] graded picks n={len(xs)}  mean p={xs.mean():.3f}  "
          f"home hit rate={ys.mean():.3f}")

    with open(CALIB_PATH) as f:
        old = json.load(f)
    old_x = np.array(old["x_thresholds"])
    old_y = np.array(old["y_thresholds"])

    new_y = refit(xs, ys, old_x, old_y,
                  prior_strength=args.prior,
                  kernel_width=args.kernel_width)

    print("[refit] deltas at key buckets:")
    print("  x      old_y    new_y    delta")
    for x in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        i = int(np.argmin(np.abs(old_x - x)))
        print(f"  {old_x[i]:.2f}   {old_y[i]:.3f}    {new_y[i]:.3f}    "
              f"{new_y[i]-old_y[i]:+.3f}")

    if args.dry_run:
        print("[refit] --dry-run: not writing calibration.json")
        return 0

    out = {
        "method": "isotonic",
        "x_thresholds": old["x_thresholds"],
        "y_thresholds": new_y.tolist(),
        "refit_meta": {
            "source": "in-season grading_history.json + Bayesian shrinkage",
            "n_picks": int(len(xs)),
            "prior_strength": args.prior,
            "kernel_width": args.kernel_width,
        },
    }
    CALIB_PATH.write_text(json.dumps(out, indent=2))
    print(f"[refit] wrote {CALIB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
