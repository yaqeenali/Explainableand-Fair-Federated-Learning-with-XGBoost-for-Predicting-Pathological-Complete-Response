"""
Centralized and local XGBoost baselines for comparison with federated model.

Three paradigms (Section 2.3):
    1. Centralized — all data pooled; theoretical performance ceiling
    2. Local       — each site trains independently; no collaboration
    3. Federated   — see models/xgboost_federated.py

XGBoost config (paper): eta=0.1, max_depth=3, binary:logistic

Usage:
    # Centralized
    python models/xgboost_baselines.py --mode centralized \
        --features_csv /data/mama-mia/features_fused.csv --output_dir results/centralized

    # Local (all 4 sites)
    python models/xgboost_baselines.py --mode local \
        --features_csv /data/mama-mia/features_fused.csv --output_dir results/local

Reference:
    Ali et al., SPIE Medical Imaging 2026, Proc. SPIE Vol. 13926, 139260Q
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    recall_score, precision_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.xgboost_federated import SITES, XGBOOST_PARAMS


# --------------------------------------------------------------------------- #
#  Shared helpers                                                              #
# --------------------------------------------------------------------------- #

def compute_metrics(y_true, y_proba, threshold=0.5):
    """Return dict of all paper metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    tn     = int(((y_true == 0) & (y_pred == 0)).sum())
    tp     = int(((y_true == 1) & (y_pred == 1)).sum())
    fp     = int(((y_true == 0) & (y_pred == 1)).sum())
    fn     = int(((y_true == 1) & (y_pred == 0)).sum())

    auc    = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "auc":               round(auc, 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "sensitivity":       round(tp / max(tp + fn, 1), 4),
        "specificity":       round(tn / max(tn + fp, 1), 4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_test": len(y_true),
    }


def train_xgboost(X_train, y_train, X_test, y_test, params=None, num_rounds=100):
    """Train XGBoost and return (booster, metrics_dict)."""
    params   = params or XGBOOST_PARAMS
    dtrain   = xgb.DMatrix(X_train, label=y_train)
    dtest    = xgb.DMatrix(X_test,  label=y_test)

    # Scale pos_weight for class imbalance
    n_neg    = (y_train == 0).sum()
    n_pos    = (y_train == 1).sum()
    p        = {**params, "scale_pos_weight": n_neg / max(n_pos, 1)}

    model    = xgb.train(p, dtrain, num_boost_round=num_rounds,
                         evals=[(dtest, "test")], verbose_eval=False)
    proba    = model.predict(dtest)
    metrics  = compute_metrics(y_test, proba)
    return model, metrics


# --------------------------------------------------------------------------- #
#  Centralized baseline                                                        #
# --------------------------------------------------------------------------- #

def run_centralized(df, feature_cols, output_dir, test_size=0.2, random_state=42):
    """
    Pool all data from all 4 sites and train a single XGBoost model.
    Represents theoretical performance ceiling (non-privacy-preserving).
    """
    print("\n" + "=" * 60)
    print("CENTRALIZED BASELINE (all sites pooled)")
    print("=" * 60)

    X     = df[feature_cols].fillna(0).values
    y     = df["pcr"].values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train: {len(y_tr)}  Test: {len(y_te)}  pCR rate: {y_tr.mean():.2f}")

    model, metrics = train_xgboost(X_tr, y_tr, X_te, y_te)

    print(f"AUC = {metrics['auc']:.4f}  BalAcc = {metrics['balanced_accuracy']:.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "centralized_model.json"))
    pd.DataFrame([metrics]).to_csv(output_dir / "centralized_metrics.csv", index=False)

    return model, metrics


# --------------------------------------------------------------------------- #
#  Local baselines                                                             #
# --------------------------------------------------------------------------- #

def run_local(df, feature_cols, output_dir, test_size=0.2, random_state=42):
    """
    Train one independent XGBoost per site — no collaboration.
    Represents the lower-bound scenario.
    """
    print("\n" + "=" * 60)
    print("LOCAL BASELINES (per-site, no collaboration)")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for site in SITES:
        site_df = df[df["site"].str.lower().str.contains(site, case=False, na=False)]
        if len(site_df) < 10:
            print(f"  [{site}] too few samples ({len(site_df)}) — skipping")
            continue

        X = site_df[feature_cols].fillna(0).values
        y = site_df["pcr"].values.astype(float)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y if y.sum() > 1 else None,
            random_state=random_state,
        )
        print(f"\n  [{site}] train={len(y_tr)}  test={len(y_te)}  "
              f"pCR rate={y_tr.mean():.2f}")

        model, metrics = train_xgboost(X_tr, y_tr, X_te, y_te)
        all_metrics[site] = metrics

        print(f"  [{site}] AUC = {metrics['auc']:.4f}  "
              f"BalAcc = {metrics['balanced_accuracy']:.4f}")

        model.save_model(str(output_dir / f"local_model_{site}.json"))

    results_df = pd.DataFrame(all_metrics).T.reset_index().rename(columns={"index": "site"})
    results_df.to_csv(output_dir / "local_metrics.csv", index=False)
    print(f"\nLocal results saved: {output_dir / 'local_metrics.csv'}")

    return all_metrics


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Centralized / local XGBoost baselines")
    parser.add_argument("--mode",         choices=["centralized", "local", "both"],
                        default="both")
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--output_dir",   default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df   = pd.read_csv(args.features_csv)
    feature_cols = [c for c in df.columns if c not in ["patient_id", "pcr", "site"]]

    print(f"Dataset: {len(df)} patients, {len(feature_cols)} features")

    if args.mode in ("centralized", "both"):
        run_centralized(df, feature_cols, f"{args.output_dir}/centralized")

    if args.mode in ("local", "both"):
        run_local(df, feature_cols, f"{args.output_dir}/local")
