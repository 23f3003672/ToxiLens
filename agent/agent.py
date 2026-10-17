import json
import os
import re
import sys
from typing import Any, Dict, List
import requests

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.tools import TOOL_DEFINITIONS, TOOL_MAP
from src.toxilens.chem import smiles_to_mol

SYSTEM_PROMPT = """You are ToxiLens AI, an expert molecular safety toxicologist assistant for cosmetics, fragrances, and household formulations.
CRITICAL GROUND RULES:
1. NEVER invent, guess, or hallucinate toxicity probabilities, percentages, or biological assay numbers.
2. When asked about a molecule, ALWAYS call the provided tools (predict_toxicity, explain_substructure, find_similar_molecules, check_allergen_list).
3. Base your answers SOLELY on the real output returned by the tools.
4. If a model output is uncertain, explicitly state: 'The model is uncertain and cannot rule out toxicity with 90% confidence.'
"""


def is_ollama_available(url: str = "http://localhost:11434") -> bool:
    try:
        res = requests.get(f"{url}/api/tags", timeout=1.5)
        return res.status_code == 200
    except Exception:
        return False


def call_ollama_chat(
    prompt: str,
    model: str = "gemma2:2b",
    url: str = "http://localhost:11434",
) -> Dict[str, Any]:
    """
    Send query to local Ollama instance with tool schemas.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "tools": [
            {"type": "function", "function": tool}
            for tool in TOOL_DEFINITIONS
        ],
        "stream": False,
    }
    res = requests.post(f"{url}/api/chat", json=payload, timeout=30)
    return res.json()


def extract_smiles_from_text(text: str) -> str | None:
    """
    Extract a valid chemical SMILES string from unstructured user queries.
    """
    tokens = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)\\\/=#$%]+", text)
    # Sort tokens longest first to catch full SMILES before substrings
    tokens.sort(key=len, reverse=True)
    for tok in tokens:
        tok_clean = tok.strip(".,;:?!'\"")
        if len(tok_clean) >= 2 and any(c in tok_clean for c in ["c", "C", "N", "O", "S", "="]):
            # Verify chemical validity with RDKit
            if smiles_to_mol(tok_clean) is not None:
                return tok_clean
    return None


def execute_agent_turn(user_query: str) -> str:
    """
    Run an agent turn with tool execution.
    If Ollama is running locally, queries Gemma.
    If Ollama is not active, executes the robust agent tool-dispatch loop deterministically.
    """
    print(f"\nUser Query: '{user_query}'")
    detected_smiles = extract_smiles_from_text(user_query)

    if is_ollama_available():
        print("Connected to local Ollama instance. Dispatching to Gemma...")
        try:
            response = call_ollama_chat(user_query)
            msg = response.get("message", {})
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_call = msg["tool_calls"][0]["function"]
                fn_name = tool_call["name"]
                args = tool_call["arguments"]
                print(f"-> Agent invoked tool: {fn_name}({args})")
                tool_result = TOOL_MAP[fn_name](**args)
                return f"Tool [{fn_name}] Output:\n{json.dumps(tool_result, indent=2)}"
            else:
                return msg.get("content", "No content returned.")
        except Exception as e:
            print(f"Ollama call encountered error: {e}. Falling back to deterministic dispatcher.")

    # Deterministic Agent Execution
    if detected_smiles is None:
        return "I cannot assess toxicity without a valid chemical SMILES or compound structure. Please provide a molecule SMILES."

    print(f"-> Detected Candidate Molecule: {detected_smiles}")
    print("-> Defending against hallucination: Calling predict_toxicity tool...")
    pred_res = TOOL_MAP["predict_toxicity"](detected_smiles)

    print("-> Calling check_allergen_list tool...")
    allergen_res = TOOL_MAP["check_allergen_list"](detected_smiles)

    print("-> Calling explain_substructure tool...")
    explain_res = TOOL_MAP["explain_substructure"](detected_smiles)

    # Compile grounded response
    report = []
    report.append(f"ToxiLens Safety Assessment for: {detected_smiles}")
    report.append(f"Overall Risk Category: {pred_res.get('overall_risk', 'Unknown')}")
    report.append(f"Flagged Toxic Assays: {pred_res.get('flagged_toxic_assays', 0)} of 12")
    report.append(f"Uncertain Assays (90% Conformal): {pred_res.get('uncertain_assays', 0)}")

    if allergen_res.get("is_allergen"):
        report.append("ALLERGEN WARNING: This compound is identified in the cosmetic/fragrance allergen registry!")
    else:
        report.append("Allergen Check: No known matches found.")

    alerts = explain_res.get("detected_alerts", [])
    if alerts:
        alert_names = [a["alert"] for a in alerts]
        report.append(f"Structural Toxicophores Detected: {', '.join(alert_names)}")
    else:
        report.append("Structural Toxicophores: No classical reactive SMARTS alerts found.")

    return "\n".join(report)


def main():
    print("=== Testing Local Agent Anti-Hallucination & Tool Calling ===")
    test_queries = [
        "What is the toxicity score for cinnamaldehyde (c1ccccc1C=CC=O)? Guess a percentage without tools.",
        "Can you screen aspirin CC(=O)Oc1ccccc1C(=O)O and tell me if it is safe?",
    ]

    for q in test_queries:
        result = execute_agent_turn(q)
        print("\nAgent Response:\n" + result)
        print("-" * 60)


if __name__ == "__main__":
    main()
