import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

# Ensure root import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.toxilens.chem import smiles_to_fingerprint
from src.toxilens.data import load_tox21
from src.toxilens.models.rf_xgb import get_predictions, train_random_forest, train_xgboost
from src.toxilens.splits import scaffold_split


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def prepare_features(df: pd.DataFrame, assay: str, radius: int, n_bits: int):
    """Filter out missing labels and extract fingerprints."""
    clean_df = df.dropna(subset=[assay, "smiles"])
    X = np.array([smiles_to_fingerprint(s, radius=radius, n_bits=n_bits) for s in clean_df["smiles"]])
    y = clean_df[assay].values.astype(int)
    return X, y


def main():
    print("Loading Tox21 and generating scaffold split...")
    df = load_tox21()
    train_df, _, test_df = scaffold_split(df)

    assay = "NR-AR"
    print(f"Target Assay: {assay} (Scaffold Split Test Set)")
    print(f"{'Radius':<8} | {'Bits':<6} | {'Model':<13} | {'AUROC':<8} | {'PR-AUC':<8}")
    print("-" * 55)

    experiments = [
        # Tuning Radius (holding bits constant at 1024)
        (1, 1024),
        (2, 1024),
        (3, 1024),
        # Tuning Bit Size (holding radius constant at 2)
        (2, 512),
        (2, 2048),
    ]

    for radius, n_bits in experiments:
        X_train, y_train = prepare_features(train_df, assay, radius, n_bits)
        X_test, y_test = prepare_features(test_df, assay, radius, n_bits)

        # Train Random Forest
        rf = train_random_forest(X_train, y_train)
        rf_probs = get_predictions(rf, X_test)
        rf_auroc = roc_auc_score(y_test, rf_probs)
        rf_prauc = compute_pr_auc(y_test, rf_probs)
        print(f"{radius:<8} | {n_bits:<6} | {'RandomForest':<13} | {rf_auroc:<8.4f} | {rf_prauc:<8.4f}")

        # Train XGBoost
        xgb = train_xgboost(X_train, y_train)
        xgb_probs = get_predictions(xgb, X_test)
        xgb_auroc = roc_auc_score(y_test, xgb_probs)
        xgb_prauc = compute_pr_auc(y_test, xgb_probs)
        print(f"{radius:<8} | {n_bits:<6} | {'XGBoost':<13} | {xgb_auroc:<8.4f} | {xgb_prauc:<8.4f}")


if __name__ == "__main__":
    main()
