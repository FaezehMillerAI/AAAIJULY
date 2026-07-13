#!/usr/bin/env python3
"""
Full end-to-end benchmark: train VisionT5 → generate all methods → evaluate → compare BLEU-1.
Run from the project root: python run_full_benchmark.py
"""
import subprocess, sys, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

PYTHON = sys.executable

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"▶  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  ✗  FAILED (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  ✓  {label} completed.")

MANIFEST   = str(OUT / "common_manifest.jsonl")
CKPT_DIR   = str(OUT / "vision_t5_checkpoint")
RAG_CACHE  = str(OUT / "rag_candidate_cache.json")
PKG_CACHE  = str(OUT / "primekg_radiology_cache")
EVAL_OUT   = str(OUT / "evaluation")

# ── Step 1: Build manifest ──────────────────────────────────────────────────
run([PYTHON, str(ROOT/"scripts/build_manifest.py"),
     "--dataset", "indiana",
     "--output-dir", str(OUT)],
    "Step 1/7 │ Build manifest")

# ── Step 2: Train VisionT5 ─────────────────────────────────────────────────
run([PYTHON, str(ROOT/"scripts/train_vision_t5_generator.py"),
     "--manifest-path", MANIFEST,
     "--text-model-name", "razent/SciFive-base-PMC",
     "--visual-backbone", "densenet121",
     "--freeze-visual-encoder", "false",
     "--epochs", "15",
     "--batch-size", "4",
     "--lr", "5e-5",
     "--output-dir", CKPT_DIR,
     "--device", "cuda"],
    "Step 2/7 │ Train VisionT5 (SciFive-base-PMC + DenseNet121, 15 epochs)")

# ── Step 3: Build visual retrieval cache ───────────────────────────────────
run([PYTHON, str(ROOT/"scripts/run_retrieval_baseline.py"),
     "--manifest-path", MANIFEST,
     "--output-csv", str(OUT/"retrieval_tfidf.csv"),
     "--output-cache", RAG_CACHE,
     "--top-k", "10"],
    "Step 3/7 │ Build visual retrieval cache (baseline)")

# ── Step 4: Generate raw VisionT5 reports ─────────────────────────────────
run([PYTHON, str(ROOT/"scripts/generate_vision_t5_reports.py"),
     "--manifest-path", MANIFEST,
     "--checkpoint-dir", CKPT_DIR,
     "--output-file", str(OUT/"vision_t5_raw.csv"),
     "--max-new-tokens", "128",
     "--batch-size", "4",
     "--device", "cuda"],
    "Step 4/7 │ Generate VisionT5 raw reports")

# ── Step 5: RAG PrimeKG Gate ───────────────────────────────────────────────
run([PYTHON, str(ROOT/"scripts/generate_rag_primekg_reports.py"),
     "--manifest-path", MANIFEST,
     "--retrieval-cache", RAG_CACHE,
     "--primekg-cache", PKG_CACHE,
     "--output-csv", str(OUT/"rag_primekg_gate.csv")],
    "Step 5/7 │ RAG + PrimeKG Gate baseline")

# ── Step 6: Adaptive Verification (Proposed) ──────────────────────────────
run([PYTHON, str(ROOT/"scripts/run_adaptive_verification.py"),
     "--raw-preds-csv", str(OUT/"vision_t5_raw.csv"),
     "--retrieval-cache", RAG_CACHE,
     "--primekg-cache", PKG_CACHE,
     "--output-dir", str(OUT),
     "--prefix", "vision_t5",
     "--policy", "evidence_replace"],
    "Step 6/7 │ Adaptive Claim Verification (Proposed)")

# ── Step 7: Evaluate all methods ──────────────────────────────────────────
methods = {
    "Retrieval (Visual)"     : str(OUT/"retrieval_tfidf.csv"),
    "RAG+PrimeKG Gate"       : str(OUT/"rag_primekg_gate.csv"),
    "VisionT5 Raw"           : str(OUT/"vision_t5_raw.csv"),
    "NeSy Proposed"          : str(OUT/"vision_t5_adaptive_claim_revision.csv"),
}

all_results = {}
for name, pred_csv in methods.items():
    if not Path(pred_csv).exists():
        print(f"  ⚠  Skipping {name}: {pred_csv} not found.")
        continue
    eval_dir = str(Path(EVAL_OUT) / name.lower().replace(" ","_").replace("+","plus"))
    run([PYTHON, str(ROOT/"scripts/evaluate_generation.py"),
         "--pred-csv", pred_csv,
         "--manifest-path", MANIFEST,
         "--primekg-cache", PKG_CACHE,
         "--output-dir", eval_dir],
        f"Step 7/7 │ Evaluate: {name}")
    # Load metrics - evaluate_generation.py saves to eval_dir/<csv_stem>/metrics.json
    found = list(Path(eval_dir).rglob("metrics.json"))
    if found:
        with open(found[0]) as f:
            all_results[name] = json.load(f)

# ── Print comparison table ─────────────────────────────────────────────────
COLS = ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "Leakage copies"]
LINE = "─" * 84
print(f"\n\n{'='*84}")
print("  BENCHMARK RESULTS — BLEU-1 TARGET: ≥ 0.60")
print(f"{'='*84}")
header = f"{'Method':<28}" + "".join(f"{c:>12}" for c in COLS)
print(header)
print(LINE)
TARGET_MET = False
for name, metrics in all_results.items():
    row = f"{name:<28}"
    b1 = metrics.get("BLEU-1", metrics.get("bleu_1", 0.0))
    # Check inside lexical dictionary if nested
    if isinstance(metrics.get("lexical"), dict):
        b1 = metrics["lexical"].get("BLEU-1", b1)
    
    for col in COLS:
        val = 0.0
        if isinstance(metrics.get("lexical"), dict) and col in metrics["lexical"]:
            val = metrics["lexical"][col]
        elif isinstance(metrics.get("leakage_audit"), dict) and col == "Leakage copies":
            val = metrics["leakage_audit"].get("exact_copies_in_train_count", 0.0)
        else:
            val = metrics.get(col, metrics.get(col.lower().replace("-","_").replace(" ","_"), 0.0))
        row += f"{val:>12.4f}"
    flag = "  ✓ TARGET MET" if b1 >= 0.60 else ""
    print(row + flag)
    if b1 >= 0.60:
        TARGET_MET = True
print(LINE)
if TARGET_MET:
    print("\n  🎯  BLEU-1 ≥ 0.60 achieved on at least one method!")
else:
    print("\n  ⚠  BLEU-1 < 0.60 on all methods. More training epochs or larger decoder needed.")
print(f"{'='*84}\n")

# Save summary JSON
summary_path = OUT / "benchmark_summary.json"
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Full metrics saved to {summary_path}")
