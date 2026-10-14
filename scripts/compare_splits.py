import os
import sys
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Ensure root import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.toxilens.data import load_tox21
from src.toxilens.splits import get_scaffold, random_split, scaffold_split


def smiles_to_morgan_fingerprint(smiles: str, n_bits: int = 1024) -> np.ndarray:
    """Convert a SMILES string to a 1024-bit Morgan fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def measure_scaffold_overlap(train_df, test_df) -> float:
    """Calculate the percentage of test molecules whose scaffold appeared in the training set."""
    train_scaffolds = set(get_scaffold(str(s)) for s in train_df["smiles"])
    test_scaffolds = [get_scaffold(str(s)) for s in test_df["smiles"]]

    leaked = sum(1 for s in test_scaffolds if s in train_scaffolds)
    return (leaked / len(test_scaffolds)) * 100.0


def evaluate_assay(train_df, test_df, assay: str = "NR-AR") -> float:
    """Train a quick Random Forest on one assay and return test AUROC."""
    # Filter out missing rows for this specific assay
    train_clean = train_df.dropna(subset=[assay, "smiles"])
    test_clean = test_df.dropna(subset=[assay, "smiles"])

    X_train = np.array([smiles_to_morgan_fingerprint(s) for s in train_clean["smiles"]])
    y_train = train_clean[assay].values.astype(int)

    X_test = np.array([smiles_to_morgan_fingerprint(s) for s in test_clean["smiles"]])
    y_test = test_clean[assay].values.astype(int)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)


def main():
    print("Loading Tox21...")
    df = load_tox21()

    print("\n--- Running Random Split ---")
    r_train, r_val, r_test = random_split(df)
    r_leakage = measure_scaffold_overlap(r_train, r_test)
    r_auc = evaluate_assay(r_train, r_test, assay="NR-AR")
    print(f"Random Split Test Size: {len(r_test)}")
    print(f"Scaffold Leakage into Test: {r_leakage:.2f}%")
    print(f"NR-AR Test AUROC: {r_auc:.4f}")

    print("\n--- Running Scaffold Split ---")
    s_train, s_val, s_test = scaffold_split(df)
    s_leakage = measure_scaffold_overlap(s_train, s_test)
    s_auc = evaluate_assay(s_train, s_test, assay="NR-AR")
    print(f"Scaffold Split Test Size: {len(s_test)}")
    print(f"Scaffold Leakage into Test: {s_leakage:.2f}%")
    print(f"NR-AR Test AUROC: {s_auc:.4f}")

    print("\nSummary:")
    print(f"AUROC dropped by {((r_auc - s_auc) / r_auc) * 100:.1f}% under scaffold split.")
    print("This drop reveals the true difficulty of generalizing to unseen chemical families.")


if __name__ == "__main__":
    main()
