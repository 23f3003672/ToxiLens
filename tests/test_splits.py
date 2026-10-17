import pandas as pd
import pytest
from src.toxilens.splits import get_scaffold, random_split, scaffold_split


def test_get_scaffold_benzene():
    # Aspirin's core is a benzene ring
    scaffold = get_scaffold("CC(=O)Oc1ccccc1C(=O)O")
    assert scaffold == "c1ccccc1"


def test_get_scaffold_acyclic():
    # Ethanol has no rings, so scaffold must be empty string
    scaffold = get_scaffold("CCO")
    assert scaffold == ""


def test_scaffold_split_no_leakage():
    # Create sample DataFrame with 3 distinct scaffold families
    data = {
        "smiles": [
            "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC",  # Benzene family
            "c1ccncc1C", "c1ccncc1CC", "c1ccncc1CCC",  # Pyridine family
            "c1ncccc1N", "c1ncccc1O",                   # Another family
            "CCCO", "CCCCO",                            # Acyclic family
        ]
    }
    df = pd.DataFrame(data)
    train_df, val_df, test_df = scaffold_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    assert len(train_df) > 0
    assert len(test_df) > 0

    train_scaffolds = set(get_scaffold(s) for s in train_df["smiles"] if get_scaffold(s) != "")
    test_scaffolds = set(get_scaffold(s) for s in test_df["smiles"] if get_scaffold(s) != "")

    # Zero scaffold overlap between train and test
    overlap = train_scaffolds.intersection(test_scaffolds)
    assert len(overlap) == 0, f"Found leaked scaffolds: {overlap}"
