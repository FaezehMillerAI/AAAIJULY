import argparse
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.kg.primekg import PrimeKGRadiologyCache
from nesy_gen.agents.adaptive_verification import run_adaptive_verification_pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-preds-csv", type=str, required=True, help="Path to raw VLM predictions CSV")
    parser.add_argument("--retrieval-cache", type=str, default="output/rag_candidate_cache.json")
    parser.add_argument("--primekg-cache", type=str, default="output/primekg_radiology_cache")
    parser.add_argument("--manifest-path", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--prefix", type=str, default="vision_t5")
    parser.add_argument("--policy", type=str, default="evidence_replace", choices=["audit_only", "evidence_replace"])
    parser.add_argument("--ltn-threshold", type=float, default=0.7, help="LTN verification threshold")
    return parser.parse_args()

def main():
    args = parse_args()
    raw_csv = Path(args.raw_preds_csv)
    cache_path = Path(args.retrieval_cache)
    primekg_path = Path(args.primekg_cache)
    out_dir = Path(args.output_dir)
    
    print(f"Loading raw predictions from {raw_csv}...")
    raw_df = pd.read_csv(raw_csv)
    raw_preds = raw_df.to_dict(orient="records")
    
    print(f"Loading retrieval candidate cache from {cache_path}...")
    with open(cache_path, "r", encoding="utf-8") as f:
        retrieval_cache = json.load(f)
        
    print(f"Loading PrimeKG cache from {primekg_path}...")
    kg_cache = PrimeKGRadiologyCache(primekg_path)
    
    # Load manifest to resolve query indications (used for zero-leakage styling)
    indications = {}
    manifest_file = Path(args.manifest_path)
    if manifest_file.exists():
        print(f"Loading manifest from {manifest_file} to map study indications...")
        from nesy_gen.manifest import load_manifest
        exs = load_manifest(manifest_file)
        indications = {ex["study_id"]: ex.get("indication", "radiology evaluation") for ex in exs}
    
    print(f"Running adaptive claim verification pipeline with policy: {args.policy}...")
    run_adaptive_verification_pipeline(
        raw_predictions=raw_preds,
        retrieval_cache=retrieval_cache,
        kg_cache=kg_cache,
        output_dir=out_dir,
        prefix=args.prefix + ("_audit_only" if args.policy == "audit_only" else ""),
        policy=args.policy,
        indications=indications,
        ltn_threshold=args.ltn_threshold
    )
    print("Adaptive verification process completed successfully.")

if __name__ == "__main__":
    main()
