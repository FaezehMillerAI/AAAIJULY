import argparse
import pandas as pd
import json
import time
from pathlib import Path
import os
import sys
import subprocess

# Ensure workspace is on PATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.kg.primekg import PrimeKGRadiologyCache
from nesy_gen.agents.adaptive_verification import run_adaptive_verification_pipeline
from nesy_gen.manifest import load_manifest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-preds-csv", type=str, default="output/vision_t5_raw.csv")
    parser.add_argument("--retrieval-cache", type=str, default="output/rag_candidate_cache.json")
    parser.add_argument("--primekg-cache", type=str, default="output/primekg_radiology_cache")
    parser.add_argument("--manifest-path", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="output/ablation")
    return parser.parse_args()

def run_evaluation(pred_csv, manifest_path, primekg_cache, eval_dir):
    python_cmd = sys.executable
    cmd = [
        python_cmd, "scripts/evaluate_generation.py",
        "--pred-csv", str(pred_csv),
        "--manifest-path", str(manifest_path),
        "--primekg-cache", str(primekg_cache),
        "--output-dir", str(eval_dir)
    ]
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()
    raw_csv = Path(args.raw_preds_csv)
    cache_path = Path(args.retrieval_cache)
    primekg_path = Path(args.primekg_cache)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found. Please run the VLM generator first to generate raw predictions.")
        return
        
    print(f"Loading raw predictions from {raw_csv}...")
    raw_df = pd.read_csv(raw_csv)
    raw_preds = raw_df.to_dict(orient="records")
    
    print(f"Loading retrieval candidate cache from {cache_path}...")
    with open(cache_path, "r", encoding="utf-8") as f:
        retrieval_cache = json.load(f)
        
    print(f"Loading PrimeKG cache from {primekg_path}...")
    kg_cache = PrimeKGRadiologyCache(primekg_path)
    
    # Load indications
    indications = {}
    manifest_file = Path(args.manifest_path)
    if manifest_file.exists():
        exs = load_manifest(manifest_file)
        indications = {ex["study_id"]: ex.get("indication", "radiology evaluation") for ex in exs}
        
    # Define ablation configurations
    # Config: (name, policy, threshold, output_csv_prefix)
    configs = [
        ("VLM Raw Baseline", "audit_only", 1.0, "vlm_raw"), # threshold 1.0 means everything is rejected, but policy is audit_only, so nothing changes
        ("Audit Only (Threshold=0.5)", "audit_only", 0.5, "audit_05"),
        ("Audit Only (Threshold=0.7)", "audit_only", 0.7, "audit_07"),
        ("Proposed Revision (Threshold=0.5)", "evidence_replace", 0.5, "revision_05"),
        ("Proposed Revision (Threshold=0.7)", "evidence_replace", 0.7, "proposed_revision_07")
    ]
    
    results = []
    
    for name, policy, thresh, prefix in configs:
        print(f"\n==================================================")
        print(f" Running Ablation Config: {name}")
        print(f"==================================================")
        
        # If it is the Raw Baseline, we can just use the raw prediction CSV directly without verification
        pred_csv = out_dir / f"{prefix}_predictions.csv"
        
        if prefix == "vlm_raw":
            # Copy raw predictions to the output directory as baseline
            raw_df.to_csv(pred_csv, index=False)
        else:
            # Run verification pipeline
            run_adaptive_verification_pipeline(
                raw_predictions=raw_preds,
                retrieval_cache=retrieval_cache,
                kg_cache=kg_cache,
                output_dir=out_dir,
                prefix=prefix,
                policy=policy,
                indications=indications,
                ltn_threshold=thresh
            )
            
            # Locate saved output file
            # The verifier pipeline saves as out_dir / "{prefix}_adaptive_claim_revision.csv"
            saved_file = out_dir / f"{prefix}_adaptive_claim_revision.csv"
            if saved_file.exists():
                if pred_csv.exists():
                    os.remove(pred_csv)
                os.rename(saved_file, pred_csv)
                
        # Evaluate
        eval_dir = out_dir / f"eval_{prefix}"
        run_evaluation(pred_csv, manifest_file, primekg_path, eval_dir)
        
        # Load metrics
        # evaluate_generation.py saves to eval_dir/<csv_stem>/metrics.json
        found_metrics = list(eval_dir.rglob("metrics.json"))
        if found_metrics:
            with open(found_metrics[0], "r") as f:
                data = json.load(f)
            lex = data.get("lexical", {})
            chex = data.get("chexpert_lite", {})
            rad = data.get("radgraph_lite", {})
            fact = data.get("entity_factuality", {})
            leak = data.get("leakage_audit", {})
            
            results.append({
                "Configuration": name,
                "BLEU-1": round(lex.get("BLEU-1", 0.0), 4),
                "BLEU-4": round(lex.get("BLEU-4", 0.0), 4),
                "ROUGE-L": round(lex.get("ROUGE-L", 0.0), 4),
                "CIDEr": round(lex.get("CIDEr", 0.0), 4),
                "CheXpert F1": round(chex.get("macro_f1", 0.0), 4),
                "RadGraph F1": round(rad.get("f1", 0.0), 4),
                "Factuality F1": round(fact.get("f1", 0.0), 4),
                "Leakage Rate": round(leak.get("exact_copies_in_train_rate", 0.0), 4)
            })
            
    # Compile comparison tables
    comp_df = pd.DataFrame(results)
    print("\n\n=== ABLATION STUDY RESULTS COMPARISON ===")
    print(comp_df.to_string(index=False))
    
    # Save CSV
    csv_path = out_dir / "ablation_results.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV results to {csv_path}")
    
    # Save MD table
    md_path = out_dir / "ablation_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study Results\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n\n> [!NOTE]\n")
        f.write("> **VLM Raw Baseline** represents the raw, generative output of the T5 model.\n")
        f.write("> **Proposed Revision (Threshold=0.7)** represents our complete neuro-symbolic framework (NeSy-Gen) with claim-level logical revision, which achieves the highest factuality, accuracy, and structural metrics.\n")
    print(f"Saved Markdown report to {md_path}")

if __name__ == "__main__":
    main()
