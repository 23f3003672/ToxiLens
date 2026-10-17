from typing import Dict, List, Optional, Set, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Well-established chemical toxicophores (structural alerts)
STRUCTURAL_ALERTS: Dict[str, str] = {
    "Nitroaromatic": "c[N+](=O)[O-]",
    "Michael Acceptor": "[C;!R]=[C;!R]-[C;!R]=O",
    "Primary Aromatic Amine": "c[NH2]",
    "Epoxide": "C1OC1",
    "Aliphatic Halide": "[CX4][Cl,Br,I]",
    "Quinone": "O=C1C=CC(=O)C=C1",
    "Hydrazine": "[NX3][NX3]",
}


def get_atom_neighborhood(mol: Chem.Mol, center_idx: int, radius: int) -> List[int]:
    """
    Find all atom indices within graph topological distance `radius` from center_idx.
    """
    visited = {center_idx}
    layer = {center_idx}

    for _ in range(radius):
        next_layer = set()
        for idx in layer:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited:
                    visited.add(nbr_idx)
                    next_layer.add(nbr_idx)
        layer = next_layer

    return list(visited)


def explain_molecule_substructures(
    smiles: str,
    feature_importances: Optional[Dict[int, float]] = None,
    top_k: int = 3,
) -> Dict:
    """
    Map active Morgan fingerprint bits back to their originating atom indices,
    and check for classical structural alerts (toxicophores).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES string"}

    bit_info: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)

    # Find which structural alerts match this molecule
    detected_alerts = []
    for alert_name, smarts in STRUCTURAL_ALERTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            matches = mol.GetSubstructMatches(pattern)
            matched_atoms = list(set(atom for match in matches for atom in match))
            detected_alerts.append({
                "alert": alert_name,
                "smarts": smarts,
                "matched_atoms": matched_atoms,
            })

    # Identify most important fingerprint bits
    important_bits = []
    all_highlight_atoms = set()

    if feature_importances:
        # Sort active bits by model feature importance
        ranked_bits = sorted(
            [b for b in bit_info.keys() if b in feature_importances],
            key=lambda b: feature_importances.get(b, 0.0),
            reverse=True,
        )
    else:
        ranked_bits = list(bit_info.keys())

    for bit_id in ranked_bits[:top_k]:
        entries = bit_info[bit_id]
        bit_atoms = set()
        for center_idx, radius in entries:
            sub_atoms = get_atom_neighborhood(mol, center_idx, radius)
            bit_atoms.update(sub_atoms)
            all_highlight_atoms.update(sub_atoms)

        important_bits.append({
            "bit_id": int(bit_id),
            "atoms": sorted(list(bit_atoms)),
            "importance": round(float(feature_importances.get(bit_id, 1.0)), 4) if feature_importances else 1.0,
        })

    # Include detected alert atoms in highlights as well
    for alert in detected_alerts:
        all_highlight_atoms.update(alert["matched_atoms"])

    return {
        "smiles": smiles,
        "num_atoms": mol.GetNumAtoms(),
        "detected_alerts": detected_alerts,
        "top_bits": important_bits,
        "highlight_atoms": sorted(list(all_highlight_atoms)),
    }


def draw_explained_molecule(
    smiles: str,
    highlight_atoms: List[int],
    output_path: str,
    legend: str = "Highlighted Toxicity Substructures",
) -> bool:
    """
    Render a molecule with highlighted substructure atoms and save to PNG.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    img = Draw.MolToImage(
        mol,
        size=(400, 400),
        highlightAtoms=highlight_atoms,
        legend=legend,
    )
    img.save(output_path)
    return True
