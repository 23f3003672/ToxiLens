import os
import sys
from typing import Any, Dict

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from src.toxilens.allergens import AllergenChecker
from src.toxilens.chem import compute_descriptors, smiles_to_fingerprint, smiles_to_mol
from src.toxilens.explain import explain_molecule_substructures
from src.toxilens.similarity import MolecularSimilarityFinder

# Load models and services locally for immediate tool execution
bundle_path = os.path.join("models", "toxilens_bundle.joblib")
bundle = joblib.load(bundle_path) if os.path.exists(bundle_path) else None
allergen_checker = AllergenChecker()
similarity_finder = MolecularSimilarityFinder()


def predict_toxicity(smiles: str) -> Dict[str, Any]:
    """
    Predict toxicity risk across 12 Tox21 assays using trained Random Forest models
    and split conformal prediction (90% mathematical coverage guarantee).
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES string: '{smiles}'"}

    if bundle is None:
        return {"error": "Models bundle not found. Please train models first."}

    fp = smiles_to_fingerprint(smiles).reshape(1, -1)
    rf_models = bundle["rf_models"]
    conformal_predictors = bundle["conformal_predictors"]

    assay_results = {}
    flagged = 0
    uncertain = 0

    for assay in bundle["assays"]:
        prob = float(rf_models[assay].predict_proba(fp)[0, 1])
        cp = conformal_predictors[assay]
        p_set = cp.predict_set(prob)

        status = "Toxic" if p_set == {1} else ("Safe" if p_set == {0} else "Uncertain (I don't know)")
        if p_set == {1}:
            flagged += 1
        elif p_set == {0, 1}:
            uncertain += 1

        assay_results[assay] = {
            "probability": round(prob, 3),
            "status": status,
        }

    return {
        "smiles": smiles,
        "descriptors": compute_descriptors(mol),
        "overall_risk": "High" if flagged >= 3 else ("Moderate" if flagged >= 1 else "Low"),
        "flagged_toxic_assays": flagged,
        "uncertain_assays": uncertain,
        "assay_predictions": assay_results,
    }


def explain_substructure(smiles: str) -> Dict[str, Any]:
    """
    Identify key active fingerprint bits and scan for known chemical toxicophores
    (such as Nitroaromatics and Michael Acceptors).
    """
    return explain_molecule_substructures(smiles)


def find_similar_molecules(smiles: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Retrieve nearest molecular analogs in the Tox21 database based on Tanimoto similarity.
    """
    neighbors = similarity_finder.find_nearest_neighbors(smiles, top_k=top_k)
    return {
        "query_smiles": smiles,
        "neighbors": neighbors,
    }


def check_allergen_list(smiles: str) -> Dict[str, Any]:
    """
    Check if the candidate molecule is listed in the EU fragrance and cosmetic allergen registry.
    """
    return allergen_checker.check_molecule(smiles)


# Tool schemas for LLM function calling
TOOL_DEFINITIONS = [
    {
        "name": "predict_toxicity",
        "description": "Predict toxicity probability across 12 Tox21 endpoints with 90% conformal certainty.",
        "parameters": {
            "type": "object",
            "properties": {"smiles": {"type": "string", "description": "SMILES string"}},
            "required": ["smiles"],
        },
    },
    {
        "name": "explain_substructure",
        "description": "Explain which molecular substructures or toxicophore alerts trigger toxicity warnings.",
        "parameters": {
            "type": "object",
            "properties": {"smiles": {"type": "string", "description": "SMILES string"}},
            "required": ["smiles"],
        },
    },
    {
        "name": "find_similar_molecules",
        "description": "Search for nearest reference molecules in the Tox21 database to evaluate precedent.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string"},
                "top_k": {"type": "integer", "description": "Number of analogs (default 5)"},
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "check_allergen_list",
        "description": "Screen the molecule against the official EU cosmetics and fragrance allergen list.",
        "parameters": {
            "type": "object",
            "properties": {"smiles": {"type": "string", "description": "SMILES string"}},
            "required": ["smiles"],
        },
    },
]

TOOL_MAP = {
    "predict_toxicity": predict_toxicity,
    "explain_substructure": explain_substructure,
    "find_similar_molecules": find_similar_molecules,
    "check_allergen_list": check_allergen_list,
}
