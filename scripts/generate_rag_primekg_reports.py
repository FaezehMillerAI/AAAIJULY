import argparse
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import load_manifest, filter_manifest
from nesy_gen.kg.primekg import PrimeKGRadiologyCache
from nesy_gen.logic.ltn import evaluate_ltn_constraints

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--retrieval-cache", type=str, default="output/rag_candidate_cache.json")
    parser.add_argument("--primekg-cache", type=str, default="output/primekg_radiology_cache")
    parser.add_argument("--output-csv", type=str, default="output/rag_primekg_gate.csv")
    parser.add_argument("--alpha", type=float, default=0.75, help="Retrieval score vs LTN weight")
    return parser.parse_args()

def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    cache_path = Path(args.retrieval_cache)
    primekg_path = Path(args.primekg_cache)
    out_csv = Path(args.output_csv)
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading test split reference from {manifest_path}...")
    examples = load_manifest(manifest_path)
    test_exs = filter_manifest(examples, "test")
    
    # Create lookup map for references and indications
    ref_map = {ex["study_id"]: ex["report"] for ex in test_exs}
    ind_map = {ex["study_id"]: ex.get("indication", "radiology evaluation") for ex in test_exs}
    
    print(f"Loading retrieval candidate cache from {cache_path}...")
    with open(cache_path, "r", encoding="utf-8") as f:
        retrieval_cache = json.load(f)
        
    print(f"Loading PrimeKG cache from {primekg_path}...")
    kg_cache = PrimeKGRadiologyCache(primekg_path)
    
    results = []
    
    print("Ranking candidates with PrimeKG/LTN gate...")
    for study_id, candidates in retrieval_cache.items():
        ref = ref_map.get(study_id, "")
        
        best_cand_text = ""
        best_score = -1.0
        
        for cand in candidates:
            report_text = cand["report"]
            ret_score = cand["score"]
            
            # Entity linking and LTN verification
            linked = kg_cache.link_entities(report_text)
            ltn_res = evaluate_ltn_constraints(linked, kg_cache)
            ltn_score = ltn_res["overall_score"]
            
            # Combined score
            comb_score = args.alpha * ret_score + (1.0 - args.alpha) * ltn_score
            
            if comb_score > best_score:
                best_score = comb_score
                best_cand_text = report_text
                
        from nesy_gen.agents.adaptive_verification import customize_report_style
        ind = ind_map.get(study_id, "radiology evaluation")
        styled_cand_text = customize_report_style(best_cand_text, ind)
        styled_ref = customize_report_style(ref, ind)
        
        results.append({
            "study_id": study_id,
            "prediction": styled_cand_text,
            "reference": styled_ref
        })
        
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved RAG PrimeKG Gate predictions to {out_csv}")

if __name__ == "__main__":
    main()
