import os
import sys

# Add the project root to the Python import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.toxilens.chem import smiles_to_mol, compute_descriptors, draw_molecules_grid

MOLECULES = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Vanillin", "O=Cc1ccc(O)c(OC)c1"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ("Limonene", "CC1=CCC(CC1)C(=C)C"),
    ("Linalool", "CC(=CCCC(C)(C=C)O)C"),
    ("Geraniol", "CC(=CCCC(=CCO)C)C"),
    ("Eugenol", "COc1cc(CC=C)ccc1O"),
    ("Citral", "CC(=CCCC(=CC=O)C)C"),
    ("Menthol", "CC1CCC(C(C1)O)C(C)C"),
    ("Camphor", "CC1(C)C2CCC1(C)C(=O)C2"),
    ("Salicylic acid", "c1ccc(c(c1)C(=O)O)O"),
    ("Coumarin", "c1ccc2c(c1)ccc(=O)o2"),
    ("Benzaldehyde", "c1ccccc1C=O"),
    ("Cinnamaldehyde", "c1ccccc1C=CC=O"),
    ("Phenoxyethanol", "c1ccccc1OCCO"),
    ("Benzyl alcohol", "c1ccccc1CO"),
    ("Ethanol", "CCO"),
    ("Glycerol", "C(C(CO)O)O"),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
]

def main():
    os.makedirs("results", exist_ok=True)

    valid_mols = []
    legends = []

    print(f"{'Name':<18} | {'MW':<8} | {'LogP':<6} | {'TPSA':<6}")
    print("-" * 46)

    for name, smiles in MOLECULES:
        mol = smiles_to_mol(smiles)
        if mol is None:
            print(f"Failed to parse {name}")
            continue

        desc = compute_descriptors(mol)
        print(f"{name:<18} | {desc['molecular_weight']:<8} | {desc['logp']:<6} | {desc['tpsa']:<6}")

        valid_mols.append(mol)
        legends.append(name)

    output_image = os.path.join("results", "molecules_grid.png")
    draw_molecules_grid(valid_mols, legends, output_image)
    print(f"\nSaved molecular drawing of {len(valid_mols)} molecules to {output_image}")


if __name__ == "__main__":
    main()
