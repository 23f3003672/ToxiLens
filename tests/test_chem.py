import numpy as np
import pytest
from src.toxilens.chem import (
    compute_descriptors,
    smiles_to_descriptor_vector,
    smiles_to_fingerprint,
    smiles_to_mol,
)


def test_smiles_to_mol_valid():
    mol = smiles_to_mol("c1ccccc1")  # Benzene
    assert mol is not None
    assert mol.GetNumAtoms() == 6


def test_smiles_to_mol_invalid():
    assert smiles_to_mol("invalid_smiles_string") is None
    assert smiles_to_mol("") is None
    assert smiles_to_mol(None) is None


def test_compute_descriptors():
    mol = smiles_to_mol("CCO")  # Ethanol
    desc = compute_descriptors(mol)
    assert "molecular_weight" in desc
    assert "logp" in desc
    assert "tpsa" in desc
    assert desc["molecular_weight"] > 40.0
    assert desc["tpsa"] > 15.0


def test_smiles_to_fingerprint():
    fp = smiles_to_fingerprint("c1ccccc1", radius=2, n_bits=1024)
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (1024,)
    assert fp.sum() > 0


def test_smiles_to_descriptor_vector():
    vec = smiles_to_descriptor_vector("c1ccccc1C(=O)O")  # Benzoic acid
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 10
