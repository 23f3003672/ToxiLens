# ToxiLens: Learning Log

Daily reflections on chemistry representations, dataset quirks, ML architectures, and empirical findings.

---

## Day 1: Chemistry and Data Foundations

### What I Learned
- **Molecular Representation:** SMILES strings encode 2D/3D chemical graphs into single-line text notations (atoms, bonds, branches, and ring closures). RDKit allows us to parse these blueprints into molecular graphs and calculate physical descriptors like Molecular Weight, LogP (lipophilicity), and TPSA (polar surface area), which determine how molecules interact with cellular membranes.
- **Class Imbalance in Bioassays:** Tox21 is heavily imbalanced—on average, only ~7.75% of molecules test positive for toxicity across the 12 endpoints (some assays like `NR-PPAR-gamma` are as low as 2.88%). Standard accuracy is completely misleading here because a dummy model guessing "safe" on everything achieves >90% accuracy while failing 100% of toxic compounds. PR-AUC (Precision-Recall AUC) is the honest metric because it focuses specifically on the rare positive class without being inflated by true negatives.
- **Scaffold Splitting vs. Random Splitting:** In chemistry datasets, many molecules share the same core ring system (Bemis-Murcko scaffold). A standard random split scatters structural twins across training and test sets, causing massive data leakage (~50% in our test set). A hand-written Bemis-Murcko scaffold split groups entire chemical families together, ensuring the test set contains zero scaffolds seen during training (0.00% leakage).

### What Surprised Me
- **The Empirical Reality of Data Leakage:** On the `NR-AR` assay, moving from a random split to a scaffold split dropped the test AUROC from ~0.81 to ~0.72 (an 11.2% drop). This was a clear demonstration that random splits flatter the model by testing its memory rather than its ability to generalize to novel chemical families.
