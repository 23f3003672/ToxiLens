import os
from typing import Dict, List, Optional
import pandas as pd
from rdkit import Chem, DataStructs

from src.toxilens.chem import smiles_to_fingerprint


class AllergenChecker:
    """
    Checks molecules against curated lists of cosmetic, fragrance, and dermal allergens.
    """

    def __init__(self, csv_path: str = "data/allergens.csv"):
        if not os.path.exists(csv_path):
            alt_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "allergens.csv")
            if os.path.exists(alt_path):
                csv_path = alt_path

        self.allergens = []
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            for _, row in self.df.iterrows():
                smiles = str(row["smiles"])
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canon_smiles = Chem.MolToSmiles(mol)
                    fp = smiles_to_fingerprint(canon_smiles)
                    self.allergens.append({
                        "name": row["name"],
                        "smiles": canon_smiles,
                        "cas_number": row.get("cas_number", "N/A"),
                        "category": row.get("category", "Allergen"),
                        "fp": fp,
                    })

    def check_molecule(self, smiles: str, threshold: float = 0.80) -> Dict:
        """
        Check if a SMILES matches or is chemically very similar to a known allergen.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"is_allergen": False, "error": "Invalid SMILES"}

        query_canon = Chem.MolToSmiles(mol)
        query_fp = smiles_to_fingerprint(query_canon)

        exact_matches = []
        similar_matches = []

        for item in self.allergens:
            if query_canon == item["smiles"]:
                exact_matches.append({
                    "name": item["name"],
                    "cas_number": item["cas_number"],
                    "category": item["category"],
                    "match_type": "Exact Chemical Match",
                })
            else:
                # Cosine / Tanimoto similarity between binary numpy arrays
                dot = float(np_dot := (query_fp * item["fp"]).sum())
                denom = float(query_fp.sum() + item["fp"].sum() - np_dot)
                sim = (dot / denom) if denom > 0 else 0.0
                if sim >= threshold:
                    similar_matches.append({
                        "name": item["name"],
                        "cas_number": item["cas_number"],
                        "similarity": round(sim, 3),
                        "match_type": "Structural Analog",
                    })

        is_flagged = len(exact_matches) > 0 or len(similar_matches) > 0
        return {
            "smiles": query_canon,
            "is_allergen": is_flagged,
            "exact_matches": exact_matches,
            "similar_allergens": sorted(similar_matches, key=lambda x: x["similarity"], reverse=True),
        }
