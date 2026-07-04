import argparse
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import load_manifest, filter_manifest
from nesy_gen.kg.primekg import PrimeKGRadiologyCache
from nesy_gen.evaluation.metrics import (
    compute_lexical_metrics,
    evaluate_chexpert_lite,
    evaluate_radgraph_lite,
    evaluate_entity_factuality,
    run_leakage_audit,
    generate_html_report,
    extract_radgraph_triplets
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--manifest-path", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--primekg-cache", type=str, default="output/primekg_radiology_cache")
    parser.add_argument("--traces-jsonl", type=str, default="", help="Path to claim traces JSONL if available")
    parser.add_argument("--output-dir", type=str, default="output/evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    pred_path = Path(args.pred_csv)
    manifest_path = Path(args.manifest_path)
    primekg_path = Path(args.primekg_cache)
    out_dir = Path(args.output_dir) / pred_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating predictions in {pred_path}...")
    pred_df = pd.read_csv(pred_path)
    
    # Load manifest splits to get training corpus for leakage auditing
    examples = load_manifest(manifest_path)
    train_exs = filter_manifest(examples, "train")
    train_corpus = [ex["report"] for ex in train_exs]
    
    # Load PrimeKG cache for factuality evaluation
    kg_cache = PrimeKGRadiologyCache(primekg_path)
    
    preds = pred_df["prediction"].fillna("").tolist()
    refs = pred_df["reference"].fillna("").tolist()
    study_ids = pred_df["study_id"].astype(str).tolist()
    
    # 1. Compute Lexical Metrics
    print("Computing lexical metrics (BLEU, ROUGE, CIDEr)...")
    lexical = compute_lexical_metrics(preds, refs)
    
    # 2. Compute CheXpert-Lite
    print("Computing clinical label proxy (CheXpert-lite)...")
    chexpert = evaluate_chexpert_lite(preds, refs)
    
    # 3. Compute RadGraph-Lite
    print("Computing relation proxy (RadGraph-lite)...")
    radgraph = evaluate_radgraph_lite(preds, refs)
    
    # 4. Compute Entity Factuality
    print("Computing entity factuality...")
    factuality = evaluate_entity_factuality(preds, refs, kg_cache)
    
    # 5. Run Leakage Audit
    print("Running leakage audit...")
    leakage = run_leakage_audit(preds, refs, train_corpus)
    
    # Compile final metrics dict
    metrics_summary = {
        "lexical": lexical,
        "chexpert_lite": {
            "macro_f1": chexpert["macro_f1"],
            "class_scores": chexpert["class_scores"]
        },
        "radgraph_lite": radgraph,
        "entity_factuality": factuality,
        "leakage_audit": leakage
    }
    
    # Save metrics JSON
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics to {out_dir / 'metrics.json'}")
    
    # Save CheXpert-Lite CSV
    chex_df = pd.DataFrame({
        "study_id": study_ids,
        "pred_labels": [str(l) for l in chexpert["raw_predictions"]],
        "ref_labels": [str(l) for l in chexpert["raw_references"]]
    })
    chex_df.to_csv(out_dir / "chexpert_lite.csv", index=False)
    
    # Save RadGraph-Lite CSV
    rad_triplets_pred = [list(extract_radgraph_triplets(p)) for p in preds]
    rad_triplets_ref = [list(extract_radgraph_triplets(r)) for r in refs]
    rad_df = pd.DataFrame({
        "study_id": study_ids,
        "pred_triplets": [str(t) for t in rad_triplets_pred],
        "ref_triplets": [str(t) for t in rad_triplets_ref]
    })
    rad_df.to_csv(out_dir / "radgraph_lite.csv", index=False)
    
    # 6. Save official input placeholders
    official_dir = out_dir / "official_inputs"
    official_dir.mkdir(parents=True, exist_ok=True)
    
    # CheXbert format: study_id, report
    pd.DataFrame({"study_id": study_ids, "report": preds}).to_csv(official_dir / "pred_reports_for_chexbert.csv", index=False)
    pd.DataFrame({"study_id": study_ids, "report": refs}).to_csv(official_dir / "ref_reports_for_chexbert.csv", index=False)
    
    # RadGraph format: JSON mapping file index or study ID to text
    radgraph_json = {sid: {"text": p} for sid, p in zip(study_ids, preds)}
    with open(official_dir / "reports_for_radgraph.json", "w", encoding="utf-8") as f:
        json.dump(radgraph_json, f, indent=2)
        
    # 7. Generate qualitative HTML report if traces exist
    traces_file = Path(args.traces_jsonl) if args.traces_jsonl else None
    if traces_file and traces_file.exists():
        print("Traces file found. Generating qualitative HTML report...")
        traces = []
        with open(traces_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        
        # We need a column for original draft in pred_df if not present
        if "original_draft" not in pred_df.columns:
            pred_df["original_draft"] = pred_df["prediction"]
            
        generate_html_report(pred_df, traces, out_dir / "qualitative_report.html")
        print(f"Generated qualitative report at {out_dir / 'qualitative_report.html'}")
        
    print(f"--- Evaluation Summary for {pred_path.name} ---")
    print(f"BLEU-4: {lexical['BLEU-4']:.4f}")
    print(f"ROUGE-L: {lexical['ROUGE-L']:.4f}")
    print(f"CIDEr: {lexical['CIDEr']:.4f}")
    print(f"CheXpert-Lite Macro F1: {chexpert['macro_f1']:.4f}")
    print(f"RadGraph-Lite F1: {radgraph['f1']:.4f}")
    print(f"Entity Factuality F1: {factuality['f1']:.4f}")
    print(f"Leakage copies: {leakage['exact_copies_in_train_count']}")

if __name__ == "__main__":
    main()
