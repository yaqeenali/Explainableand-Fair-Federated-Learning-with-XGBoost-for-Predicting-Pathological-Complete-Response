"""
Feature selection and multimodal feature fusion.

Pipeline (Section 2.2):
    1. Spearman correlation filtering (remove highly correlated features)
    2. Mutual information ranking to identify top-k imaging features
    3. Fuse top-6 imaging features with 5 clinical variables:
       ER, PR, HER2, HR, tumour subtype

Reference:
    Ali et al., SPIE Medical Imaging 2026, Proc. SPIE Vol. 13926, 139260Q
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


# --------------------------------------------------------------------------- #
#  Clinical feature names (from paper Section 2.2)                            #
# --------------------------------------------------------------------------- #

CLINICAL_FEATURES = [
    "er",              # Estrogen Receptor
    "pr",              # Progesterone Receptor
    "her2",            # HER2 status
    "hr",              # Hormone Receptor
    "tumor_subtype",   # Molecular subtype (one-hot encoded downstream)
]


# --------------------------------------------------------------------------- #
#  Step 1 — Spearman correlation filtering                                    #
# --------------------------------------------------------------------------- #

def spearman_filter(df_features, threshold=0.90):
    """
    Remove one feature from each highly correlated pair (|rho| > threshold).
    Keeps the feature with higher variance.

    Args:
        df_features: DataFrame of radiomic features (no patient_id or label)
        threshold:   Spearman correlation threshold

    Returns:
        selected_cols: list of retained feature names
    """
    cols = df_features.columns.tolist()
    corr_matrix = df_features.corr(method="spearman").abs()

    to_drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the one with lower variance
                var_i = df_features[cols[i]].var()
                var_j = df_features[cols[j]].var()
                drop_col = cols[j] if var_i >= var_j else cols[i]
                to_drop.add(drop_col)

    selected = [c for c in cols if c not in to_drop]
    print(f"Spearman filter: {len(cols)} → {len(selected)} features "
          f"(removed {len(to_drop)} correlated)")
    return selected


# --------------------------------------------------------------------------- #
#  Step 2 — Mutual information ranking                                        #
# --------------------------------------------------------------------------- #

def mutual_info_ranking(df_features, labels, top_k=6, random_state=42):
    """
    Rank features by mutual information with the pCR label.
    Returns top_k feature names.

    Args:
        df_features: DataFrame of candidate radiomic features
        labels:      array-like binary pCR labels
        top_k:       number of top features to keep (paper uses 6)

    Returns:
        top_features: list of top_k feature names
    """
    mi_scores = mutual_info_classif(
        df_features.fillna(0).values,
        labels,
        discrete_features=False,
        random_state=random_state,
    )
    mi_series = pd.Series(mi_scores, index=df_features.columns)
    top_features = mi_series.nlargest(top_k).index.tolist()

    print(f"Top {top_k} imaging features by mutual information:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat:<60}  MI = {mi_series[feat]:.4f}")

    return top_features


# --------------------------------------------------------------------------- #
#  Step 3 — Multimodal feature fusion                                         #
# --------------------------------------------------------------------------- #

def fuse_features(radiomics_csv, clinical_csv, output_csv,
                  top_k=6, spearman_threshold=0.90):
    """
    Full pipeline: load → filter → rank → fuse → save.

    Args:
        radiomics_csv:      output of radiomic_extraction.py
        clinical_csv:       CSV with patient_id, pcr, and clinical variables
        output_csv:         path for the final fused feature CSV
        top_k:              number of top imaging features (paper: 6)
        spearman_threshold: correlation cutoff (paper: 0.90)
    """
    # Load data
    df_radio  = pd.read_csv(radiomics_csv)
    df_clin   = pd.read_csv(clinical_csv)

    # Merge on patient_id
    df = df_radio.merge(df_clin[["patient_id", "pcr", "site"] + CLINICAL_FEATURES],
                        on="patient_id", how="inner")
    print(f"Merged dataset: {len(df)} patients")

    labels        = df["pcr"].values
    imaging_cols  = [c for c in df_radio.columns if c != "patient_id"]
    df_imaging    = df[imaging_cols].fillna(0)

    # Step 1: Spearman correlation filter
    selected_cols = spearman_filter(df_imaging, threshold=spearman_threshold)
    df_imaging    = df_imaging[selected_cols]

    # Step 2: Mutual information — select top-k imaging features
    top_imaging   = mutual_info_ranking(df_imaging, labels, top_k=top_k)

    # Step 3: Build final multimodal feature vector
    # Clinical variables — one-hot encode tumour subtype
    df_clin_enc = pd.get_dummies(df[CLINICAL_FEATURES], columns=["tumor_subtype"],
                                 drop_first=False).astype(float)

    final_cols = ["patient_id", "pcr", "site"] + top_imaging + df_clin_enc.columns.tolist()
    df_final   = pd.concat([
        df[["patient_id", "pcr", "site"]],
        df[top_imaging],
        df_clin_enc,
    ], axis=1)

    # Standardise numeric features
    feature_cols = top_imaging + [c for c in df_clin_enc.columns if c not in ["patient_id"]]
    scaler       = StandardScaler()
    df_final[feature_cols] = scaler.fit_transform(df_final[feature_cols].fillna(0))

    df_final.to_csv(output_csv, index=False)
    print(f"\nFinal feature matrix: {df_final.shape[0]} patients × "
          f"{len(feature_cols)} features")
    print(f"Saved to: {output_csv}")
    return df_final, top_imaging


def parse_args():
    parser = argparse.ArgumentParser(description="Feature selection and fusion")
    parser.add_argument("--radiomics_csv",    required=True)
    parser.add_argument("--clinical_csv",     required=True)
    parser.add_argument("--output_csv",       required=True)
    parser.add_argument("--top_k",            type=int, default=6)
    parser.add_argument("--spearman_threshold", type=float, default=0.90)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fuse_features(
        args.radiomics_csv,
        args.clinical_csv,
        args.output_csv,
        args.top_k,
        args.spearman_threshold,
    )
