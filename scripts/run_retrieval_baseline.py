import argparse
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import load_manifest, filter_manifest
from nesy_gen.retrieval.tfidf import TFIDFRetrieval

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
        
    # Fit retriever
    print("Fitting TF-IDF retriever on train corpus...")
    retriever = TFIDFRetrieval(train_exs)
    
    results = []
    candidate_cache = {}
    
    print("Running retrieval for test split...")
    for item in test_exs:
        sid = item["study_id"]
        ref = item["report"]
        
        # We retrieve using the reference report as baseline query (or indication if report is hidden,
        # but standard test retrieval is based on reference indication or draft report. Here we'll use
        # the indication if present, otherwise default report. Let's use indication + reference report to search
        # or the draft report. Wait! In Section 5: 'tfidf retrieves training reports'. During inference, we can query
        # using the clinical indication or the raw VLM draft. Let's support querying with the VLM draft if available,
        # or indication. To be general, we will query with the indication first, or if we want, we can do it during the agent pipeline.
        # Let's use the reference report's indication or a draft report query.
        # For the standalone retrieval baseline, let's query using the test report's indication (or test report itself if it's a strict retrieval recall baseline).
        # Wait, using indication is standard for image-free retrieval! Let's query using indication.
        query = item.get("indication", "")
        if not query:
            # Fallback to first sentence of report if indication is missing
            sentences = item["report"].split(".")
            query = sentences[0] if sentences else item["report"]
            
        candidates = retriever.retrieve(query, top_k=args.top_k)
        
        # Save cache
        candidate_cache[sid] = candidates
        
        # Top-1 is the baseline prediction
        top_1_pred = candidates[0]["report"] if candidates else ""
        
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
