import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.toxilens.chem import smiles_to_fingerprint, smiles_to_descriptor_vector
from src.toxilens.conformal import BinaryConformalPredictor
from src.toxilens.data import TOX21_ASSAYS, load_tox21
from src.toxilens.evaluate import compute_binary_metrics, evaluate_predictions
from src.toxilens.models.gnn import GINNet, predict_gnn_proba, smiles_to_graph, train_gnn
from src.toxilens.models.rf_xgb import get_predictions, train_random_forest, train_xgboost
from src.toxilens.models.tabpfn_model import predict_tabpfn_proba, train_tabpfn
from src.toxilens.splits import scaffold_split


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Step 1: Loading Tox21 and performing scaffold split...")
    df = load_tox21()
    train_df, val_df, test_df = scaffold_split(df)
    print(f"Train size: {len(train_df)}, Val (Calibration) size: {len(val_df)}, Test size: {len(test_df)}")

    # Precompute features
    print("Precomputing Morgan fingerprints and physicochemical descriptors...")
    X_train_fp = np.array([smiles_to_fingerprint(s) for s in train_df["smiles"]])
    X_val_fp = np.array([smiles_to_fingerprint(s) for s in val_df["smiles"]])
    X_test_fp = np.array([smiles_to_fingerprint(s) for s in test_df["smiles"]])

    X_train_desc = np.array([smiles_to_descriptor_vector(s) for s in train_df["smiles"]])
    X_val_desc = np.array([smiles_to_descriptor_vector(s) for s in val_df["smiles"]])
    X_test_desc = np.array([smiles_to_descriptor_vector(s) for s in test_df["smiles"]])

    rf_models = {}
    xgb_models = {}
    conformal_predictors = {}

    rf_test_preds = {}
    xgb_test_preds = {}

    print("\nStep 2: Training Random Forest & XGBoost per assay + Conformal Calibration...")
    for assay in TOX21_ASSAYS:
        train_mask = ~train_df[assay].isna()
        val_mask = ~val_df[assay].isna()
        test_mask = ~test_df[assay].isna()

        y_train = train_df[assay][train_mask].values.astype(int)
        y_val = val_df[assay][val_mask].values.astype(int)

        # Train RF
        rf = train_random_forest(X_train_fp[train_mask], y_train)
        rf_models[assay] = rf
        rf_val_prob = get_predictions(rf, X_val_fp[val_mask])
        rf_test_prob = get_predictions(rf, X_test_fp)
        rf_test_preds[assay] = rf_test_prob

        # Train XGB
        xgb = train_xgboost(X_train_fp[train_mask], y_train)
        xgb_models[assay] = xgb
        xgb_test_preds[assay] = get_predictions(xgb, X_test_fp)

        # Conformal calibration on validation set
        cp = BinaryConformalPredictor(alpha=0.10)
        cp.fit(y_val, rf_val_prob)
        conformal_predictors[assay] = cp

    # Save models and conformal predictors to disk
    bundle_path = os.path.join("models", "toxilens_bundle.joblib")
    joblib.dump({
        "rf_models": rf_models,
        "xgb_models": xgb_models,
        "conformal_predictors": conformal_predictors,
        "assays": TOX21_ASSAYS,
    }, bundle_path)
    print(f"Successfully saved all 12 trained models and conformal predictors to {bundle_path}")

    # Evaluate RF and XGBoost
    print("\nStep 3: Evaluating models on scaffold test set...")
    rf_eval = evaluate_predictions(test_df, rf_test_preds, TOX21_ASSAYS)
    xgb_eval = evaluate_predictions(test_df, xgb_test_preds, TOX21_ASSAYS)

    comparison_df = pd.DataFrame({
        "Assay": rf_eval["Assay"],
        "RF_AUROC": rf_eval["AUROC"],
        "RF_PRAUC": rf_eval["PR-AUC"],
        "XGB_AUROC": xgb_eval["AUROC"],
        "XGB_PRAUC": xgb_eval["PR-AUC"],
    })
    comp_path = os.path.join("results", "model_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"Saved full model comparison table to {comp_path}")
    print(comparison_df.to_string(index=False))

    # Step 4: Advanced Model & Representation Ablation on target assay NR-AR
    print("\nStep 4: Running representation ablation (Fingerprints vs Descriptors vs Graph) on NR-AR...")
    target_assay = "NR-AR"
    tr_mask = ~train_df[target_assay].isna()
    te_mask = ~test_df[target_assay].isna()
    y_tr = train_df[target_assay][tr_mask].values.astype(int)
    y_te = test_df[target_assay][te_mask].values.astype(int)

    # 1. RF on Fingerprints
    rf_m = rf_models[target_assay]
    rf_probs = get_predictions(rf_m, X_test_fp[te_mask])
    rf_m_res = compute_binary_metrics(y_te, rf_probs)

    # 2. TabPFN on Descriptors
    print("Training TabPFN on RDKit descriptors...")
    tabpfn_m = train_tabpfn(X_train_desc[tr_mask], y_tr, max_train_samples=800)
    tabpfn_probs = predict_tabpfn_proba(tabpfn_m, X_test_desc[te_mask])
    tabpfn_res = compute_binary_metrics(y_te, tabpfn_probs)

    # 3. GNN (GIN) on Molecular Graphs
    print("Training GIN on molecular graphs...")
    train_graphs = [
        smiles_to_graph(s, float(y))
        for s, y in zip(train_df["smiles"][tr_mask], y_tr)
        if smiles_to_graph(s, float(y)) is not None
    ]
    test_graphs = [
        smiles_to_graph(s)
        for s in test_df["smiles"][te_mask]
        if smiles_to_graph(s) is not None
    ]

    gin_m = train_gnn(train_graphs, num_epochs=6, batch_size=64, pos_weight=10.0)
    gin_probs = predict_gnn_proba(gin_m, test_graphs)
    # Match length in case any failed graph parse
    gin_res = compute_binary_metrics(y_te[:len(gin_probs)], gin_probs)

    ablation_df = pd.DataFrame([
        {"Model": "Random Forest", "Representation": "Morgan Fingerprints (1024b)", "AUROC": rf_m_res["auroc"], "PR-AUC": rf_m_res["pr_auc"]},
        {"Model": "TabPFN", "Representation": "Physicochemical Descriptors", "AUROC": tabpfn_res["auroc"], "PR-AUC": tabpfn_res["pr_auc"]},
        {"Model": "GIN (GNN)", "Representation": "Molecular Graph (Atoms + Bonds)", "AUROC": gin_res["auroc"], "PR-AUC": gin_res["pr_auc"]},
    ])
    abl_path = os.path.join("results", "ablation_results.csv")
    ablation_df.to_csv(abl_path, index=False)
    print("\n--- Ablation Results ---")
    print(ablation_df.to_string(index=False))

    # Step 5: Conformal Calibration Evaluation & Plot
    print("\nStep 5: Evaluating Conformal Coverage across assays...")
    cal_metrics = []
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
    emp_coverages = []

    cp_target = conformal_predictors[target_assay]
    for a in alphas:
        cp_test = BinaryConformalPredictor(alpha=a)
        val_m = ~val_df[target_assay].isna()
        rf_val_p = get_predictions(rf_models[target_assay], X_val_fp[val_m])
        cp_test.fit(val_df[target_assay][val_m].values.astype(int), rf_val_p)
        stats = cp_test.evaluate_coverage(y_te, rf_probs)
        emp_coverages.append(stats["empirical_coverage"])

    plt.figure(figsize=(7, 5))
    target_coverages = [(1.0 - a) * 100 for a in alphas]
    plt.plot(target_coverages, target_coverages, "k--", label="Ideal Guarantee (y = x)")
    plt.plot(target_coverages, emp_coverages, "ro-", linewidth=2, label="Empirical Coverage")
    plt.title(f"Conformal Calibration Plot on Scaffold Split ({target_assay})")
    plt.xlabel("Nominal Target Coverage (%)")
    plt.ylabel("Empirical Coverage (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join("results", "conformal_calibration.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved conformal calibration curve to {plot_path}")


if __name__ == "__main__":
    main()
