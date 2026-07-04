import argparse
import pandas as pd
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import load_manifest, filter_manifest
from nesy_gen.vlm.model import VisionT5
from nesy_gen.vlm.dataset import RadiologyDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--checkpoint-dir", type=str, default="output/vision_t5_checkpoint")
    parser.add_argument("--output-file", type=str, default="output/vision_t5_raw.csv")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    ckpt_dir = Path(args.checkpoint_dir)
    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load manifest
    print(f"Loading test manifest from {manifest_path}...")
    examples = load_manifest(manifest_path)
    test_exs = filter_manifest(examples, "test")
    print(f"Test examples found: {len(test_exs)}")
    
    if not test_exs:
        print("No test examples found. Exiting.")
        return
        
    # Load tokenizer and model
    print("Loading tokenizer and model from checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer", use_fast=True)
    
    model = VisionT5(freeze_visual_encoder=True)
    model.load_checkpoint(ckpt_dir)
    model.to(device)
    model.eval()
    
    test_dataset = RadiologyDataset(test_exs, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    results = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Reports"):
            images = batch["images"].to(device)
            enc_ids = batch["encoder_input_ids"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            study_ids = batch["study_id"]
            refs = batch["raw_report"]
            
            # Generate reports
            generated_ids = model.generate(
                images=images,
                encoder_input_ids=enc_ids,
                encoder_attention_mask=enc_mask,
                max_length=args.max_new_tokens,
                num_beams=2,
                early_stopping=True
            )
            
            # Decode
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for sid, pred, ref in zip(study_ids, predictions, refs):
                results.append({
                    "study_id": sid,
                    "prediction": pred.strip(),
                    "reference": ref.strip()
                })
                
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} predictions to {out_file}")

if __name__ == "__main__":
    main()
