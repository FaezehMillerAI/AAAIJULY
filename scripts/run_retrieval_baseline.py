import argparse
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import load_manifest, filter_manifest
from nesy_gen.retrieval.visual import VisualRetrieval

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--output-csv", type=str, default="output/retrieval_tfidf.csv")
    parser.add_argument("--output-cache", type=str, default="output/rag_candidate_cache.json")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    out_csv = Path(args.output_csv)
    out_cache = Path(args.output_cache)
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading manifest from {manifest_path}...")
    examples = load_manifest(manifest_path)
    train_exs = filter_manifest(examples, "train")
    test_exs = filter_manifest(examples, "test")
    
    print(f"Train size: {len(train_exs)}, Test size: {len(test_exs)}")
    
    if not train_exs or not test_exs:
        print("Empty splits. Exiting.")
        return
        
    # Initialize Visual retriever
    print("Extracting train image features for Visual Retrieval...")
    retriever = VisualRetrieval(train_exs)
    
    results = []
    candidate_cache = {}
    
    print("Running visual retrieval for test split...")
    for item in test_exs:
        sid = item["study_id"]
        ref = item["report"]
        img_path = item["image_path"]
        
        # Retrieve visually similar reports from the training set
        candidates = retriever.retrieve(img_path, top_k=args.top_k)
        
        # Save cache
        candidate_cache[sid] = candidates
        
        # Top-1 is the baseline prediction
        top_1_pred = candidates[0]["report"] if candidates else ""
        
        # Format prediction with query indication to prevent exact training set leakage
        query = item.get("indication", "radiology evaluation")
        from nesy_gen.agents.adaptive_verification import customize_report_style
        top_1_pred = customize_report_style(top_1_pred, query)
        
        results.append({
            "study_id": sid,
            "prediction": top_1_pred,
            "reference": ref
        })
        
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved retrieval baseline CSV to {out_csv}")
    
    # Save JSON cache
    with open(out_cache, "w", encoding="utf-8") as f:
        json.dump(candidate_cache, f, indent=2)
    print(f"Saved retrieval candidate cache to {out_cache}")

if __name__ == "__main__":
    main()
