from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np 
import pandas as pd 
from rdkit import Chem 
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles: str) -> str: 
    """
    Extraxt the Bemis-Murcko scaffold string from smiles.
    If the molecule has no rings or is invalid, returns an empty string.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)

    except Exception:
        return ""

def random_split(df: pd.DataFrame, train_ratio: float=0.8, val_ratio: float=0.1, test_ratio: float=0.1, seed: int=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard random train/val/test split
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))

    n_total = len(df)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def scaffold_split(df: pd.DataFrame, smiles_col: str = "smiles", train_ratio: float=0.8, val_ratio:float=0.1,test_ratio:float=0.1) -> Tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame]:
    """
    Partition dataset by Bemis-Murcko scaffold into train, val and test sets.
    Molecules with the same core scaffold are paced in the same split.
    """

    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, smiles in enumerate(df[smiles_col]):
        scaffold = get_scaffold(str(smiles))
        scaffold_to_indices[scaffold].append(idx)

    scaffold_groups = sorted(
        scaffold_to_indices.values(),
        key=lambda group: len(group),
        reverse = True,
    )

    n_total = len(df)
    n_train_target = int(train_ratio * n_total)
    n_val_target = int(val_ratio * n_total)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for group in scaffold_groups:
        if len(train_indices) + len(group) <= n_train_target:
            train_indices.extend(group)
        elif len(val_indices) + len(group) <= n_val_target:
            val_indices.extend(group)
        else:
            test_indices.extend(group)

    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[val_indices].reset_index(drop=True),
        df.iloc[test_indices].reset_index(drop=True),
    )