import base64
from contextlib import asynccontextmanager
import io
import os
import sys
from typing import Dict, List, Optional
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Draw

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sentry_sdk
from src.toxilens.allergens import AllergenChecker
from src.toxilens.chem import compute_descriptors, smiles_to_fingerprint, smiles_to_mol
from src.toxilens.explain import explain_molecule_substructures
from src.toxilens.similarity import MolecularSimilarityFinder

sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )

# Global model state
bundle = None
allergen_checker = None
similarity_finder = None


def load_artifacts():
    global bundle, allergen_checker, similarity_finder
    bundle_path = os.path.join("models", "toxilens_bundle.joblib")
    if os.path.exists(bundle_path):
        bundle = joblib.load(bundle_path)
    if allergen_checker is None:
        allergen_checker = AllergenChecker()
    if similarity_finder is None:
        similarity_finder = MolecularSimilarityFinder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="ToxiLens API",
    description="Offline molecule safety screener powered by Conformal Prediction and RDKit",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local demo interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MoleculeRequest(BaseModel):
    smiles: str = Field(
        ...,
        description="SMILES string of the candidate molecule",
        json_schema_extra={"example": "c1ccccc1C=CC=O"},
    )


class SimilarMoleculesRequest(BaseModel):
    smiles: str = Field(
        ...,
        description="SMILES string",
        json_schema_extra={"example": "c1ccccc1C=CC=O"},
    )
    top_k: int = Field(5, description="Number of similar molecules to return", ge=1, le=20)


@app.get("/")
@app.get("/health")
def health_check():
    models_ready = bundle is not None
    return {
        "status": "healthy",
        "service": "ToxiLens Molecule Safety Screener",
        "models_loaded": models_ready,
        "assays_available": len(bundle["assays"]) if models_ready else 0,
    }


@app.post("/predict")
def predict_toxicity(req: MoleculeRequest):
    smiles = req.smiles.strip()
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SMILES string: '{smiles}'. Please provide a chemically valid SMILES.",
        )

    if bundle is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    fp = smiles_to_fingerprint(smiles, radius=2, n_bits=1024).reshape(1, -1)
    descriptors = compute_descriptors(mol)

    rf_models = bundle["rf_models"]
    conformal_predictors = bundle["conformal_predictors"]
    assays = bundle["assays"]

    results = {}
    toxic_count = 0
    uncertain_count = 0

    for assay in assays:
        model = rf_models[assay]
        prob = float(model.predict_proba(fp)[0, 1])

        cp = conformal_predictors[assay]
        p_set = cp.predict_set(prob)

        # Interpret conformal prediction set
        if p_set == {1}:
            conformal_label = "Toxic"
            toxic_count += 1
        elif p_set == {0}:
            conformal_label = "Safe"
        elif p_set == {0, 1}:
            conformal_label = "Uncertain (I don't know)"
            uncertain_count += 1
        else:
            conformal_label = "Out of distribution"

        results[assay] = {
            "predicted_probability": round(prob, 4),
            "point_prediction": "Toxic" if prob >= 0.50 else "Safe",
            "conformal_prediction_set": sorted(list(p_set)),
            "conformal_status": conformal_label,
            "confidence_guarantee": "90%",
        }

    overall_risk = "High" if toxic_count >= 3 else ("Moderate" if toxic_count >= 1 else "Low")

    return {
        "smiles": Chem.MolToSmiles(mol),
        "descriptors": descriptors,
        "overall_risk": overall_risk,
        "flagged_toxic_assays": toxic_count,
        "uncertain_assays": uncertain_count,
        "assay_predictions": results,
    }


@app.post("/explain")
def explain_toxicity(req: MoleculeRequest):
    smiles = req.smiles.strip()
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    importances = {}
    if bundle and "rf_models" in bundle:
        first_assay = bundle["assays"][0]
        rf = bundle["rf_models"][first_assay]
        for idx, imp in enumerate(rf.feature_importances_):
            if imp > 0.001:
                importances[idx] = float(imp)

    explanation = explain_molecule_substructures(smiles, feature_importances=importances, top_k=3)

    highlight_atoms = explanation.get("highlight_atoms", [])
    img = Draw.MolToImage(mol, size=(400, 400), highlightAtoms=highlight_atoms)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    explanation["image_base64"] = img_b64
    return explanation


@app.post("/similar-molecules")
def get_similar_molecules(req: SimilarMoleculesRequest):
    smiles = req.smiles.strip()
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    if similarity_finder is None:
        raise HTTPException(status_code=503, detail="Similarity finder not initialized.")

    neighbors = similarity_finder.find_nearest_neighbors(smiles, top_k=req.top_k)
    return {
        "query_smiles": Chem.MolToSmiles(mol),
        "nearest_neighbors": neighbors,
    }


@app.post("/check-allergens")
def check_allergens(req: MoleculeRequest):
    smiles = req.smiles.strip()
    if allergen_checker is None:
        raise HTTPException(status_code=503, detail="Allergen checker not initialized.")

    result = allergen_checker.check_molecule(smiles)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
