from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute AUROC and PR-AUC for a single binary classification task.
    Handles edge cases (e.g., if only one class is present in y_true).
    """
    # Ensure inputs are clean numpy arrays
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # If there are not at least two classes, AUROC/PR-AUC cannot be calculated
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {"auroc": float("nan"), "pr_auc": float("nan")}

    auroc = float(roc_auc_score(y_true, y_prob))

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))

    return {
        "auroc": round(auroc, 4),
        "pr_auc": round(pr_auc, 4),
    }


def evaluate_predictions(
    y_true_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    assays: List[str],
) -> pd.DataFrame:
    """
    Evaluate predicted probabilities across multiple assays.
    
    Args:
        y_true_df: DataFrame containing the true labels for each assay (can contain NaNs).
        predictions: Dictionary mapping assay_name -> 1D array of predicted probabilities.
        assays: List of assay names to evaluate.

    Returns:
        A pandas DataFrame with metrics per assay, plus an overall Macro Average row.
    """
    records = []

    for assay in assays:
        if assay not in y_true_df.columns or assay not in predictions:
            continue

        true_col = y_true_df[assay].values
        prob_col = predictions[assay]

        # Filter out NaN rows (molecules not tested on this specific assay)
        valid_mask = ~np.isnan(true_col)
        y_true_valid = true_col[valid_mask].astype(int)
        y_prob_valid = prob_col[valid_mask]

        if len(y_true_valid) == 0:
            continue

        n_pos = int(np.sum(y_true_valid == 1))
        n_neg = int(np.sum(y_true_valid == 0))

        metrics = compute_binary_metrics(y_true_valid, y_prob_valid)

        records.append({
            "Assay": assay,
            "Total Labeled": len(y_true_valid),
            "Positives (1)": n_pos,
            "AUROC": metrics["auroc"],
            "PR-AUC": metrics["pr_auc"],
        })

    results_df = pd.DataFrame(records)

    # Compute Macro Average summary row across valid assays
    if not results_df.empty:
        macro_row = pd.DataFrame([{
            "Assay": "MACRO AVERAGE",
            "Total Labeled": int(results_df["Total Labeled"].sum()),
            "Positives (1)": int(results_df["Positives (1)"].sum()),
            "AUROC": round(results_df["AUROC"].mean(), 4),
            "PR-AUC": round(results_df["PR-AUC"].mean(), 4),
        }])
        results_df = pd.concat([results_df, macro_row], ignore_index=True)

    return results_df
