import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from rdkit import Chem

from src.toxilens.chem import smiles_to_fingerprint
from src.toxilens.data import load_tox21, TOX21_ASSAYS


class MolecularSimilarityFinder:
    """
    Finds nearest chemical analogs in the Tox21 database using Morgan fingerprint Tanimoto similarity.
    """

    def __init__(self, data_path: Optional[str] = None):
        df = load_tox21()
        self.reference_data = []

        for idx, row in df.iterrows():
            smiles = str(row["smiles"])
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = smiles_to_fingerprint(smiles)
                active_assays = [
                    assay for assay in TOX21_ASSAYS
                    if assay in row and row[assay] == 1.0
                ]
                self.reference_data.append({
                    "mol_id": str(row.get("mol_id", f"MOL_{idx}")),
                    "smiles": smiles,
                    "fp": fp,
                    "active_toxic_assays": active_assays,
                })

    def find_nearest_neighbors(self, query_smiles: str, top_k: int = 5) -> List[Dict]:
        """
        Compute Tanimoto similarity across database and return top-K closest molecules.
        """
        mol = Chem.MolFromSmiles(query_smiles)
        if mol is None:
            return []

        query_fp = smiles_to_fingerprint(query_smiles)
        similarities = []

        q_sum = float(query_fp.sum())
        for item in self.reference_data:
            ref_fp = item["fp"]
            dot = float((query_fp * ref_fp).sum())
            denom = q_sum + float(ref_fp.sum()) - dot
            sim = (dot / denom) if denom > 0 else 0.0
            similarities.append((sim, item))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, item in similarities[:top_k]:
            results.append({
                "mol_id": item["mol_id"],
                "smiles": item["smiles"],
                "tanimoto_similarity": round(sim, 3),
                "known_toxic_endpoints": item["active_toxic_assays"],
            })

        return results
