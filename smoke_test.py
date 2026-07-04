import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    print(f"\nRunning command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"STDOUT:\n{res.stdout}")
        print(f"STDERR:\n{res.stderr}")
        sys.exit(1)
    else:
        print(f"Success! Output:\n{res.stdout}")

def main():
    root_dir = Path(__file__).resolve().parent
    out_dir = root_dir / "output"
    
    # 1. Build Manifest (Mock Mode)
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "build_manifest.py"),
        "--iu-xray-root", "nonexistent_dir",
        "--output-dir", str(out_dir),
        "--mock"
    ])
    
    # 2. Build PrimeKG Cache (Mock Mode)
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "build_radiology_primekg.py"),
        "--primekg-nodes", "nonexistent_nodes.csv",
        "--primekg-edges", "nonexistent_edges.csv",
        "--train-manifest", str(out_dir / "common_manifest.jsonl"),
        "--output-dir", str(out_dir / "primekg_radiology_cache")
    ])
    
    # 3. Fit TF-IDF retrieval
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "run_retrieval_baseline.py"),
        "--manifest-path", str(out_dir / "common_manifest.jsonl"),
        "--output-csv", str(out_dir / "retrieval_tfidf.csv"),
        "--output-cache", str(out_dir / "rag_candidate_cache.json"),
        "--top-k", "5"
    ])
    
    # 4. Fit RAG PrimeKG Gate
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "generate_rag_primekg_reports.py"),
        "--manifest-path", str(out_dir / "common_manifest.jsonl"),
        "--retrieval-cache", str(out_dir / "rag_candidate_cache.json"),
        "--primekg-cache", str(out_dir / "primekg_radiology_cache"),
        "--output-csv", str(out_dir / "rag_primekg_gate.csv")
    ])
    
    # 5. Train Vision-T5 generator (1 epoch, batch size 2, CPU)
    # t5-small tokenizer requires sentencepiece or transformers package
    # We will use "cpu" to run anywhere.
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "train_vision_t5_generator.py"),
        "--manifest-path", str(out_dir / "common_manifest.jsonl"),
        "--epochs", "1",
        "--batch-size", "2",
        "--output-dir", str(out_dir / "vision_t5_checkpoint"),
        "--device", "cpu"
    ])
    
    # 6. Generate raw reports
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "generate_vision_t5_reports.py"),
        "--manifest-path", str(out_dir / "common_manifest.jsonl"),
        "--checkpoint-dir", str(out_dir / "vision_t5_checkpoint"),
        "--output-file", str(out_dir / "vision_t5_raw.csv"),
        "--batch-size", "2",
        "--device", "cpu"
    ])
    
    # 7. Run Adaptive Verification (Revision policy)
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "run_adaptive_verification.py"),
        "--raw-preds-csv", str(out_dir / "vision_t5_raw.csv"),
        "--retrieval-cache", str(out_dir / "rag_candidate_cache.json"),
        "--primekg-cache", str(out_dir / "primekg_radiology_cache"),
        "--output-dir", str(out_dir),
        "--prefix", "vision_t5",
        "--policy", "evidence_replace"
    ])
    
    # 8. Run Adaptive Verification (Audit only policy)
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "run_adaptive_verification.py"),
        "--raw-preds-csv", str(out_dir / "vision_t5_raw.csv"),
        "--retrieval-cache", str(out_dir / "rag_candidate_cache.json"),
        "--primekg-cache", str(out_dir / "primekg_radiology_cache"),
        "--output-dir", str(out_dir),
        "--prefix", "vision_t5",
        "--policy", "audit_only"
    ])
    
    # 9. Evaluate generation for proposed method
    run_cmd([
        sys.executable,
        str(root_dir / "scripts" / "evaluate_generation.py"),
        "--pred-csv", str(out_dir / "vision_t5_adaptive_claim_revision.csv"),
        "--manifest-path", str(out_dir / "common_manifest.jsonl"),
        "--primekg-cache", str(out_dir / "primekg_radiology_cache"),
        "--traces-jsonl", str(out_dir / "vision_t5_adaptive_claim_revision_traces.jsonl"),
        "--output-dir", str(out_dir / "evaluation")
    ])
    
    print("\nAll modules and scripts ran successfully in the smoke test!")

if __name__ == "__main__":
    main()
