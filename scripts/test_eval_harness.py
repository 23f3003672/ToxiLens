import os
import sys
import numpy as np
import pandas as pd

# Ensure root import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.toxilens.chem import smiles_to_fingerprint
from src.toxilens.data import TOX21_ASSAYS, load_tox21
from src.toxilens.evaluate import evaluate_predictions
from src.toxilens.models.rf_xgb import get_predictions, train_random_forest
from src.toxilens.splits import scaffold_split


def main():
    print("Loading Tox21 dataset and scaffold split...")
    df = load_tox21()
    train_df, _, test_df = scaffold_split(df)

    print("Precomputing 1024-bit Morgan fingerprints (radius=2)...")
    X_train_all = np.array([smiles_to_fingerprint(s, radius=2, n_bits=1024) for s in train_df["smiles"]])
    X_test_all = np.array([smiles_to_fingerprint(s, radius=2, n_bits=1024) for s in test_df["smiles"]])

    rf_predictions = {}

    print(f"Training Random Forest across all {len(TOX21_ASSAYS)} assays...")
    for assay in TOX21_ASSAYS:
        # Drop missing values for this specific assay in train and test
        train_mask = ~train_df[assay].isna()
        test_mask = ~test_df[assay].isna()

        y_train = train_df[assay][train_mask].values.astype(int)

        # Train model
        model = train_random_forest(X_train_all[train_mask], y_train)

        # Predict for all test molecules (missing rows will be filtered out inside evaluate_predictions)
        test_probs = get_predictions(model, X_test_all)
        rf_predictions[assay] = test_probs

    print("\n--- Standardized Evaluation Harness Output ---")
    results_table = evaluate_predictions(test_df, rf_predictions, TOX21_ASSAYS)
    print(results_table.to_string(index=False))


if __name__ == "__main__":
    main()
