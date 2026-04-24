#!/usr/bin/env python3
"""
WNBA Model Trainer
Walk-forward cross-validation on WNBA historical data.
Exports model artifacts to data/model/.

Usage: python python/train_model.py
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HIST_CSV  = DATA_DIR / "training_data.csv"

FEATURE_NAMES = [
    "elo_diff", "win_pct_diff", "log5_prob",
    "pythagorean_diff", "is_home",
]


def main():
    print("WNBA Model Trainer")
    print("=" * 40)

    if not HIST_CSV.exists():
        print(f"No training data at {HIST_CSV}. Run: python python/build_dataset.py")
        sys.exit(1)

    df = pd.read_csv(HIST_CSV)
    df = df.dropna(subset=["home_win"])
    print(f"Loaded {len(df)} games, {df['season'].nunique()} seasons")

    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = 0.0

    seasons = sorted(df["season"].unique())
    print(f"Seasons: {seasons}")
    print()

    # Walk-forward cross-validation
    print("Walk-forward CV:")
    print(f"  {'Season':>6}  {'N':>4}  {'Acc':>6}  {'Brier':>6}")
    accs, briers, all_cal_raw, all_cal_true = [], [], [], []

    for i, test_season in enumerate(seasons[1:], 1):
        train_df = df[df["season"].isin(seasons[:i])]
        test_df  = df[df["season"] == test_season]
        if len(train_df) < 50 or len(test_df) < 10:
            continue

        X_tr = train_df[FEATURE_NAMES].fillna(0).values
        y_tr = train_df["home_win"].values
        X_te = test_df[FEATURE_NAMES].fillna(0).values
        y_te = test_df["home_win"].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_tr_s, y_tr)
        probs = lr.predict_proba(X_te_s)[:, 1]
        probs = np.clip(probs, 0.01, 0.99)

        acc   = accuracy_score(y_te, probs >= 0.5)
        brier = brier_score_loss(y_te, probs)
        accs.append(acc); briers.append(brier)
        all_cal_raw.extend(probs.tolist()); all_cal_true.extend(y_te.tolist())

        print(f"  {test_season:>6}  {len(test_df):>4}  {acc:.3f}  {brier:.3f}")

    print(f"\n  Avg accuracy: {np.mean(accs):.4f}")
    print(f"  Avg Brier:    {np.mean(briers):.4f}")

    # Final model on all data
    X_all = df[FEATURE_NAMES].fillna(0).values
    y_all = df["home_win"].values
    sc_f  = StandardScaler()
    X_s   = sc_f.fit_transform(X_all)
    lr_f  = LogisticRegression(C=1.0, max_iter=1000)
    lr_f.fit(X_s, y_all)

    # Isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip")
    sorted_raw = sorted(zip(all_cal_raw, all_cal_true))
    raw_sorted  = [x[0] for x in sorted_raw]
    true_sorted = [x[1] for x in sorted_raw]
    iso.fit(raw_sorted, true_sorted)
    thresholds = np.linspace(0.01, 0.99, 50).tolist()
    calibrated = iso.predict(thresholds).tolist()

    # Save model artifacts (dict-key format for predict.py compatibility)
    coeff_dict = {"_intercept": float(lr_f.intercept_[0])}
    for feat, w in zip(FEATURE_NAMES, lr_f.coef_[0]):
        coeff_dict[feat] = float(w)

    (MODEL_DIR / "coefficients.json").write_text(json.dumps(coeff_dict, indent=2))
    (MODEL_DIR / "scaler.json").write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "mean":  sc_f.mean_.tolist(),
        "scale": sc_f.scale_.tolist(),
    }, indent=2))
    (MODEL_DIR / "calibration.json").write_text(json.dumps({
        "method": "isotonic",
        "x_thresholds": thresholds,
        "y_thresholds": calibrated,
    }, indent=2))
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "version":       "4.1.0",
        "sport":         "WNBA",
        "feature_names": FEATURE_NAMES,
        "n_samples":     len(df),
        "avg_accuracy":  float(np.mean(accs)),
        "avg_brier":     float(np.mean(briers)),
    }, indent=2))

    print(f"\nModel saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
