import os
import sys
import joblib
import pandas as pd
from PIL import Image
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.toxilens.allergens import AllergenChecker
from src.toxilens.chem import compute_descriptors, smiles_to_fingerprint, smiles_to_mol
from src.toxilens.explain import explain_molecule_substructures
from src.toxilens.similarity import MolecularSimilarityFinder

st.set_page_config(
    page_title="ToxiLens | Molecule Safety Screener",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 ToxiLens: Offline Molecule Safety Screener")
st.markdown(
    "**Private, offline toxicity prediction with calibrated conformal uncertainty.** "
    "Designed for cosmetic, fragrance, and household product formulators."
)

# Load artifacts once
@st.cache_resource
def load_all_models():
    bundle_path = os.path.join("models", "toxilens_bundle.joblib")
    if not os.path.exists(bundle_path):
        return None, None, None
    bundle = joblib.load(bundle_path)
    allergen_checker = AllergenChecker()
    similarity_finder = MolecularSimilarityFinder()
    return bundle, allergen_checker, similarity_finder


bundle, allergen_checker, similarity_finder = load_all_models()

if bundle is None:
    st.error("Model artifacts not found in `models/toxilens_bundle.joblib`. Run `python scripts/train_all.py` first.")
    st.stop()

# Helper to render table safely across all pyarrow/streamlit configurations
def render_table_safely(df: pd.DataFrame):
    try:
        st.dataframe(df, hide_index=True)
    except BaseException:
        # Fallback to pure HTML table (100% independent of pyarrow)
        html_table = df.to_html(index=False, escape=False)
        style = """
        <style>
        .custom-table table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 14px;
        }
        .custom-table th, .custom-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        .custom-table th {
            background-color: #262730;
            color: #fafafa;
            font-weight: 600;
        }
        </style>
        """
        st.markdown(f"{style}<div class='custom-table'>{html_table}</div>", unsafe_allow_html=True)


# Sidebar: Preset Molecules
st.sidebar.header("Sample Compounds")
PRESETS = {
    "Select a preset...": "",
    "Cinnamaldehyde (Fragrance & Allergen)": "c1ccccc1C=CC=O",
    "Limonene (Citrus Fragrance)": "CC1=CCC(CC1)C(=C)C",
    "Aspirin (Analgesic)": "CC(=O)Oc1ccccc1C(=O)O",
    "Vanillin (Vanilla Aroma)": "O=Cc1ccc(O)c(OC)c1",
    "Coumarin (Fragrance Allergen)": "c1ccc2c(c1)ccc(=O)o2",
    "Benzo[a]pyrene (Carcinogen Benchmark)": "c1ccc2c(c1)c3ccc4cccc5c4c3c2cc5",
}

chosen_preset = st.sidebar.selectbox("Choose a known molecule:", list(PRESETS.keys()))
default_smiles = PRESETS[chosen_preset] if chosen_preset != "Select a preset..." else "c1ccccc1C=CC=O"

input_smiles = st.text_input("Enter Molecule SMILES:", value=default_smiles)

if not input_smiles.strip():
    st.warning("Please enter a valid SMILES string.")
    st.stop()

mol = smiles_to_mol(input_smiles.strip())
if mol is None:
    st.error(f"Invalid SMILES string: '{input_smiles}'. Please verify the chemical structure.")
    st.stop()

canon_smiles = Chem.MolToSmiles(mol)
descriptors = compute_descriptors(mol)

# Top Metrics Row
col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

with col1:
    st.subheader("2D Chemical Structure")
    img = Draw.MolToImage(mol, size=(320, 260))
    st.image(img)
    st.caption(f"Canonical SMILES: `{canon_smiles}`")

with col2:
    st.metric("Molecular Weight", f"{descriptors['molecular_weight']} g/mol")
    st.metric("LogP (Lipophilicity)", f"{descriptors['logp']}")

with col3:
    st.metric("TPSA (Polar Area)", f"{descriptors['tpsa']} Å²")
    st.metric("Heavy Atoms", mol.GetNumHeavyAtoms())

with col4:
    allergen_res = allergen_checker.check_molecule(canon_smiles)
    if allergen_res["is_allergen"]:
        st.error("⚠️ ALLERGEN ALERT")
        if allergen_res["exact_matches"]:
            st.write(f"**Exact match:** {allergen_res['exact_matches'][0]['name']}")
        elif allergen_res["similar_allergens"]:
            top_sim = allergen_res["similar_allergens"][0]
            st.write(f"**Analog of:** {top_sim['name']} ({top_sim['similarity']*100:.0f}% similarity)")
    else:
        st.success("✅ No Known Allergen Match")

st.divider()

# Tabbed Analysis
tab_pred, tab_explain, tab_similar = st.tabs([
    "📊 Toxicity & Conformal Prediction",
    "🔍 Substructure Attribution & Toxicophores",
    "🧬 Nearest Chemical Analogs",
])

with tab_pred:
    st.subheader("12 Tox21 Assay Predictions with 90% Conformal Coverage Guarantee")
    st.markdown(
        "Conformal prediction provides finite-sample statistical guarantees. "
        "When the model flags **'Uncertain (I don't know)'**, it means both Toxic and Safe classes "
        "remain mathematically plausible under the 90% confidence threshold."
    )

    fp = smiles_to_fingerprint(canon_smiles).reshape(1, -1)
    rf_models = bundle["rf_models"]
    conformal_predictors = bundle["conformal_predictors"]
    assays = bundle["assays"]

    rows = []
    toxic_count = 0
    uncertain_count = 0

    for assay in assays:
        prob = float(rf_models[assay].predict_proba(fp)[0, 1])
        cp = conformal_predictors[assay]
        p_set = cp.predict_set(prob)

        if p_set == {1}:
            badge = "🔴 Toxic"
            toxic_count += 1
        elif p_set == {0}:
            badge = "🟢 Safe"
        elif p_set == {0, 1}:
            badge = "🟡 Uncertain (I don't know)"
            uncertain_count += 1
        else:
            badge = "⚪ Outlier"

        rows.append({
            "Toxicity Endpoint": assay,
            "Toxicity Probability": f"{prob*100:.1f}%",
            "Point Prediction": "Toxic" if prob >= 0.50 else "Safe",
            "90% Conformal Guarantee": badge,
        })

    results_table = pd.DataFrame(rows)
    render_table_safely(results_table)

    c_risk1, c_risk2 = st.columns(2)
    with c_risk1:
        st.info(f"**Total Assays Flagged Toxic:** {toxic_count} of 12")
    with c_risk2:
        st.info(f"**Total Assays with High Uncertainty:** {uncertain_count} of 12")

with tab_explain:
    st.subheader("Substructure Attribution & Toxicophore Alert Detection")
    first_assay = bundle["assays"][0]
    importances = {
        idx: float(imp)
        for idx, imp in enumerate(bundle["rf_models"][first_assay].feature_importances_)
        if imp > 0.001
    }
    explanation = explain_molecule_substructures(canon_smiles, feature_importances=importances)

    col_e1, col_e2 = st.columns([1, 1.2])
    with col_e1:
        highlight_atoms = explanation.get("highlight_atoms", [])
        exp_img = Draw.MolToImage(
            mol,
            size=(380, 320),
            highlightAtoms=highlight_atoms,
            legend="Highlighted Active Features",
        )
        st.image(exp_img)

    with col_e2:
        alerts = explanation.get("detected_alerts", [])
        if alerts:
            st.warning(f"⚠️ Found {len(alerts)} Known Structural Toxicophore(s):")
            for a in alerts:
                st.write(f"- **{a['alert']}** (Pattern: `{a['smarts']}`)")
        else:
            st.success("✅ No classical SMARTS structural alerts (e.g. Nitroaromatic, Michael Acceptor) detected.")

        st.markdown("**Top Active Fingerprint Bits Identified:**")
        for bit in explanation.get("top_bits", []):
            st.write(f"- Bit `{bit['bit_id']}` (Atoms: {bit['atoms']}) — Feature Weight: `{bit['importance']}`")

with tab_similar:
    st.subheader("Nearest Chemical Neighbors in Tox21")
    neighbors = similarity_finder.find_nearest_neighbors(canon_smiles, top_k=5)
    if neighbors:
        sim_rows = []
        for n in neighbors:
            sim_rows.append({
                "Compound ID": n["mol_id"],
                "SMILES": n["smiles"],
                "Tanimoto Similarity": f"{n['tanimoto_similarity']*100:.1f}%",
                "Known Toxic Assays in Lab": ", ".join(n["known_toxic_endpoints"]) if n["known_toxic_endpoints"] else "None (Non-toxic)",
            })
        render_table_safely(pd.DataFrame(sim_rows))
    else:
        st.write("No similar molecules found.")
