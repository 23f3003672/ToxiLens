from typing import Dict, List, Optional
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw, rdFingerprintGenerator

def smiles_to_mol(smiles:str) -> Optional[Chem.Mol]:
    """
    Parse a SMILES string into an RDKit molecule object.
    Returns none if the SMILES string is invalid
    """

    if not smiles or not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)

def compute_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute molecular weight, LogP, and TPSA for a valid molecule.
    """

    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
    }

def smiles_to_fingerprint(smiles: str, radius: int=2, n_bits: int=1024,) -> np.ndarray:

    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeroes(n_bits,dtype = np.float32)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=n_bits)
    fp = gen.GetFingerprint(mol)

    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr 

def draw_molecules_grid(
    mols: List[Chem.Mol],
    legends: List[str],
    output_path: str,
    mols_per_row: int = 4,
) -> None:
    """
    Draw a liist of molecules in a grid and save to an image file.
    """
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow= mols_per_row,
        subImgSize = (300, 300),
        legends = legends,
    )
    img.save(output_path)