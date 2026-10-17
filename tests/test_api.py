from fastapi.testclient import TestClient
import pytest
from api.main import app, load_artifacts


@pytest.fixture(scope="module")
def client():
    load_artifacts()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_predict_valid_smiles(client):
    res = client.post("/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert res.status_code == 200
    data = res.json()
    assert "descriptors" in data
    assert "assay_predictions" in data
    assert len(data["assay_predictions"]) == 12
    assert "overall_risk" in data


def test_predict_invalid_smiles_returns_400(client):
    res = client.post("/predict", json={"smiles": "completely_invalid_smiles_xyz"})
    assert res.status_code == 400
    data = res.json()
    assert "detail" in data
    assert "Invalid SMILES" in data["detail"]


def test_check_allergens(client):
    # Cinnamaldehyde is a known EU fragrance allergen
    res = client.post("/check-allergens", json={"smiles": "c1ccccc1C=CC=O"})
    assert res.status_code == 200
    data = res.json()
    assert data["is_allergen"] is True


def test_similar_molecules(client):
    res = client.post("/similar-molecules", json={"smiles": "c1ccccc1C=CC=O", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert len(data["nearest_neighbors"]) <= 3
