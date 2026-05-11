"""
Fairness analysis — implements Equations 1-3 from the paper.

Fairness Score (Equalized Odds-based):
    Disparity_v  = (max(TPR_v) - min(TPR_v)) + (max(FPR_v) - min(FPR_v))
    Disparity    = mean over V = {age, menopausal_status, scanner_vendor, site}
    Fairness Score = 1 - Disparity

Higher score → more equitable performance across subgroups.

Paper result:
    Federated model:   Fairness Score = 0.62
    Centralized model: Fairness Score = 0.53

Usage:
    python evaluation/fairness.py \
        --predictions_csv results/federated/predictions.csv \
        --subgroup_csv    /data/mama-mia/clinical.csv \
        --output_dir      results/fairness

Reference:
    Ali et al., SPIE Medical Imaging 2026, Proc. SPIE Vol. 13926, 139260Q
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Subgroup variables (from paper Section 2.4)                               #
# --------------------------------------------------------------------------- #

FAIRNESS_VARIABLES = ["age_group", "menopausal_status", "scanner_vendor", "site"]


def bin_age(age_series):
    """Bin continuous age into clinical groups."""
    bins   = [0, 40, 50, 60, 70, 200]
    labels = ["<40", "40-50", "50-60", "60-70", "71+"]
    return pd.cut(age_series, bins=bins, labels=labels, right=False)


# --------------------------------------------------------------------------- #
#  Core fairness metrics (Equations 1-3)                                      #
# --------------------------------------------------------------------------- #

def tpr_fpr_per_subgroup(df, subgroup_col, label_col="pcr", pred_col="prediction"):
    """
    Compute TPR and FPR for each subgroup within a variable.

    Returns:
        dict {subgroup_value: {"tpr": float, "fpr": float, "n": int}}
    """
    results = {}
    for group, gdf in df.groupby(subgroup_col):
        y_true = gdf[label_col].values
        y_pred = gdf[pred_col].values

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        results[str(group)] = {"tpr": tpr, "fpr": fpr, "n": len(gdf)}

    return results


def disparity_v(subgroup_stats):
    """
    Equation 1 — disparity for one variable.
    Disparity_v = (max(TPR_v) - min(TPR_v)) + (max(FPR_v) - min(FPR_v))
    """
    tprs = [v["tpr"] for v in subgroup_stats.values() if v["n"] >= 5]
    fprs = [v["fpr"] for v in subgroup_stats.values() if v["n"] >= 5]

    if len(tprs) < 2:
        return 0.0  # only one subgroup — no disparity measurable

    return (max(tprs) - min(tprs)) + (max(fprs) - min(fprs))


def fairness_score(df, variables=None):
    """
    Equations 1-3: compute overall fairness score.

    Args:
        df:        DataFrame with columns: pcr, prediction, and subgroup variables
        variables: list of subgroup variable names (default: FAIRNESS_VARIABLES)

    Returns:
        score:         float in [0, 1], higher = fairer
        per_var:       dict {variable: disparity_v}
        subgroup_data: dict {variable: subgroup_stats}
    """
    variables = variables or [v for v in FAIRNESS_VARIABLES if v in df.columns]

    per_var      = {}
    subgroup_data = {}

    for var in variables:
        stats        = tpr_fpr_per_subgroup(df, var)
        disp         = disparity_v(stats)
        per_var[var] = disp
        subgroup_data[var] = stats

    # Equation 2 — mean disparity
    mean_disp = np.mean(list(per_var.values()))

    # Equation 3 — fairness score
    score = 1.0 - mean_disp

    return score, per_var, subgroup_data


# --------------------------------------------------------------------------- #
#  Reporting and visualisation                                                 #
# --------------------------------------------------------------------------- #

def print_fairness_report(score, per_var, subgroup_data, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"  FAIRNESS REPORT — {model_name}")
    print(f"{'='*60}")
    print(f"  Overall Fairness Score: {score:.4f}  (paper: FL=0.62, CL=0.53)")
    print(f"\n  Per-variable disparities:")
    for var, disp in per_var.items():
        print(f"    {var:<22}: disparity = {disp:.4f}")
    print()
    for var, stats in subgroup_data.items():
        print(f"  [{var}]")
        for grp, m in stats.items():
            if m["n"] >= 5:
                print(f"    {grp:<12}: TPR={m['tpr']:.3f}  FPR={m['fpr']:.3f}  n={m['n']}")


def plot_subgroup_performance(df, variables, output_dir, model_name="model"):
    """Bar plots of balanced accuracy per subgroup, one panel per variable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(variables),
                             figsize=(5 * len(variables), 4), sharey=False)
    if len(variables) == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        groups, bal_accs, ns = [], [], []
        for grp, gdf in df.groupby(var):
            if len(gdf) < 5:
                continue
            y_true   = gdf["pcr"].values
            y_pred   = gdf["prediction"].values
            from sklearn.metrics import balanced_accuracy_score
            bal_acc  = balanced_accuracy_score(y_true, y_pred)
            groups.append(str(grp))
            bal_accs.append(bal_acc)
            ns.append(len(gdf))

        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(groups)))
        bars   = ax.bar(groups, bal_accs, color=colors, edgecolor="white")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
        ax.set_title(var.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)

        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"n={n}",
                    ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"Subgroup Performance — {model_name}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = output_dir / f"subgroup_performance_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Subgroup plot saved: {path}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def run(predictions_csv, subgroup_csv, output_dir, threshold=0.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    df_pred = pd.read_csv(predictions_csv)   # expects: patient_id, pcr, proba
    df_pred["prediction"] = (df_pred["proba"] >= threshold).astype(int)

    # Load subgroup info
    df_sub = pd.read_csv(subgroup_csv)

    # Bin age if raw age available
    if "age" in df_sub.columns:
        df_sub["age_group"] = bin_age(df_sub["age"]).astype(str)

    df = df_pred.merge(df_sub, on="patient_id", how="inner")
    print(f"Loaded {len(df)} patients with subgroup information")

    # Compute fairness
    score, per_var, subgroup_data = fairness_score(df)
    print_fairness_report(score, per_var, subgroup_data)

    # Save summary
    summary = {
        "fairness_score": round(score, 4),
        **{f"disparity_{k}": round(v, 4) for k, v in per_var.items()},
    }
    pd.DataFrame([summary]).to_csv(output_dir / "fairness_summary.csv", index=False)

    # Plot
    available_vars = [v for v in FAIRNESS_VARIABLES if v in df.columns]
    if available_vars:
        plot_subgroup_performance(df, available_vars, output_dir)

    return score, per_var


def parse_args():
    parser = argparse.ArgumentParser(description="Fairness analysis (Equations 1-3)")
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--subgroup_csv",    required=True)
    parser.add_argument("--output_dir",      default="results/fairness")
    parser.add_argument("--threshold",       type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.predictions_csv, args.subgroup_csv, args.output_dir, args.threshold)
