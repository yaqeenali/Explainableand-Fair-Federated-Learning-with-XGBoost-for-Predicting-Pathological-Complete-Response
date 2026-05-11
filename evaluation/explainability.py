"""
SHAP explainability for the global federated XGBoost model.
Reproduces Figure 2 from the paper.

Key findings:
    - High HR values → negative SHAP → non-pCR (HR+ tumors less chemosensitive)
    - High HER2 values → positive SHAP → pCR (HER2-targeted therapy efficacy)
    - Maximum 3D Tumour Diameter, Sphericity, Texture also ranked highly

Usage:
    python evaluation/explainability.py \
        --model_path results/federated/global_model.json \
        --features_csv /data/mama-mia/features_fused.csv \
        --output_dir results/shap

Reference:
    Ali et al., SPIE Medical Imaging 2026, Proc. SPIE Vol. 13926, 139260Q
    Lundberg & Lee, NeurIPS 2017 (SHAP)
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Human-readable feature name mapping                                        #
# --------------------------------------------------------------------------- #

FEATURE_DISPLAY_NAMES = {
    "hr":                           "Hormone Receptor (HR)",
    "her2":                         "HER2 Status",
    "er":                           "Estrogen Receptor (ER)",
    "pr":                           "Progesterone Receptor (PR)",
    "tumor_subtype":                "Molecular Subtype",
    # Radiomic features (mapped from PyRadiomics names, top SHAP features)
    "original_shape_Maximum3DDiameter":       "Maximum 3D Tumor Diameter",
    "original_shape_Sphericity":              "Tumor Sphericity",
    "original_glcm_Imc1":                     "Tumor Texture Complexity",
    "original_firstorder_Kurtosis":           "Voxel Intensity Kurtosis",
}


def friendly_name(col):
    for k, v in FEATURE_DISPLAY_NAMES.items():
        if k.lower() in col.lower():
            return v
    return col.replace("original_", "").replace("_", " ").title()


# --------------------------------------------------------------------------- #
#  SHAP computation                                                            #
# --------------------------------------------------------------------------- #

def compute_shap(model_path, df, feature_cols):
    """
    Load XGBoost model and compute SHAP values.

    Returns:
        shap_values: np.ndarray (n_samples, n_features)
        explainer:   shap.TreeExplainer
    """
    model    = xgb.Booster()
    model.load_model(str(model_path))

    X = df[feature_cols].fillna(0).values
    dmat = xgb.DMatrix(X, feature_names=feature_cols)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print(f"SHAP values computed for {X.shape[0]} samples, "
          f"{X.shape[1]} features")
    return shap_values, explainer, X


# --------------------------------------------------------------------------- #
#  Figure 2 reproduction — SHAP summary plot                                  #
# --------------------------------------------------------------------------- #

def plot_shap_summary(shap_values, X, feature_cols, output_dir, top_n=9):
    """
    Reproduce Figure 2: SHAP beeswarm summary plot.
    Ranks features by mean |SHAP value| and colours dots by feature value.

    Args:
        top_n: number of top features to display (paper shows 9)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rank features by mean |SHAP|
    mean_abs  = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:top_n]
    top_feats = [feature_cols[i] for i in top_idx]
    top_shap  = shap_values[:, top_idx]
    top_X     = X[:, top_idx]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for row_idx, (feat, col_idx) in enumerate(zip(top_feats, range(top_n))):
        feat_vals  = top_X[:, col_idx]
        shap_vals  = top_shap[:, col_idx]

        # Normalise feature values to [0,1] for colour
        f_min, f_max = feat_vals.min(), feat_vals.max()
        if f_max > f_min:
            feat_norm = (feat_vals - f_min) / (f_max - f_min)
        else:
            feat_norm = np.full_like(feat_vals, 0.5)

        # Jitter y for beeswarm effect
        y_pos  = (top_n - 1 - row_idx)
        jitter = np.random.uniform(-0.25, 0.25, size=len(shap_vals))

        colors = cmap(norm(feat_norm))
        ax.scatter(shap_vals, y_pos + jitter, c=colors, s=8,
                   alpha=0.7, linewidths=0, rasterized=True)

    # Y axis — feature names
    y_labels = [friendly_name(f) for f in top_feats]
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(y_labels[::-1], fontsize=10)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title("SHAP Summary Plot — Global Federated XGBoost Model",
                 fontsize=12, fontweight="bold", pad=12)

    # Colorbar
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.6)
    cbar.set_label("Feature value", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    fig.tight_layout()
    path = output_dir / "shap_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"SHAP summary plot saved: {path}")


def plot_shap_bar(shap_values, feature_cols, output_dir, top_n=15):
    """Mean |SHAP| bar chart — global feature importance."""
    output_dir = Path(output_dir)
    mean_abs   = np.abs(shap_values).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]

    names  = [friendly_name(feature_cols[i]) for i in top_idx]
    values = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.Blues(np.linspace(0.4, 0.85, top_n))[::-1]
    bars    = ax.barh(range(top_n), values[::-1], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title("Global Feature Importance — Federated XGBoost", fontsize=11)
    fig.tight_layout()

    path = output_dir / "shap_bar_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SHAP bar chart saved: {path}")


# --------------------------------------------------------------------------- #
#  Save SHAP values to CSV                                                     #
# --------------------------------------------------------------------------- #

def save_shap_csv(shap_values, feature_cols, df, output_dir):
    """Save SHAP values alongside patient IDs for downstream analysis."""
    output_dir = Path(output_dir)
    shap_df    = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    if "patient_id" in df.columns:
        shap_df.insert(0, "patient_id", df["patient_id"].values)
    if "pcr" in df.columns:
        shap_df.insert(1, "pcr", df["pcr"].values)
    shap_df.to_csv(output_dir / "shap_values.csv", index=False)
    print(f"SHAP values saved: {output_dir / 'shap_values.csv'}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def run(model_path, features_csv, output_dir, top_n=9):
    df           = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c not in ["patient_id", "pcr", "site"]]

    shap_values, explainer, X = compute_shap(model_path, df, feature_cols)

    plot_shap_summary(shap_values, X, feature_cols, output_dir, top_n=top_n)
    plot_shap_bar(shap_values, feature_cols, output_dir)
    save_shap_csv(shap_values, feature_cols, df, output_dir)

    # Print top features (matching paper Figure 2 description)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    print(f"\nTop {top_n} features by mean |SHAP|:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {friendly_name(feature_cols[idx]):<40}  "
              f"mean|SHAP|={mean_abs[idx]:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="SHAP explainability (Figure 2)")
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--features_csv",  required=True)
    parser.add_argument("--output_dir",    default="results/shap")
    parser.add_argument("--top_n",         type=int, default=9)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model_path, args.features_csv, args.output_dir, args.top_n)
