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
    
    # Load saved config to restore exact model arch that was trained
    import json
    config_path = ckpt_dir / "config.json"
    ckpt_text_model = "razent/SciFive-base-PMC"
    ckpt_visual_backbone = "densenet121"
    ckpt_freeze = False
    if config_path.exists():
        with open(config_path) as f:
            ckpt_cfg = json.load(f)
        ckpt_text_model = ckpt_cfg.get("text_model_name", ckpt_text_model)
        ckpt_visual_backbone = ckpt_cfg.get("visual_backbone", ckpt_visual_backbone)
        ckpt_freeze = ckpt_cfg.get("freeze_visual_encoder", ckpt_freeze)
    
    model = VisionT5(
        text_model_name=ckpt_text_model,
        visual_backbone=ckpt_visual_backbone,
        freeze_visual_encoder=ckpt_freeze
    )
    model.load_checkpoint(ckpt_dir)
    model.to(device)
    model.eval()
    
    test_dataset = RadiologyDataset(test_exs, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load manifest indications for styling
    indications = {ex["study_id"]: ex.get("indication", "radiology evaluation") for ex in test_exs}
    from nesy_gen.agents.adaptive_verification import customize_report_style
    
    results = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Reports"):
            images = batch["images"].to(device)
            enc_ids = batch["encoder_input_ids"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            study_ids = batch["study_id"]
            refs = batch["raw_report"]
            
            # Generate reports with 4-beam search, length + repetition penalties
            generated_ids = model.generate(
                images=images,
                encoder_input_ids=enc_ids,
                encoder_attention_mask=enc_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=4,
                length_penalty=1.5,    # encourage longer, more complete reports
                repetition_penalty=1.3, # mild penalty — SciFive sub-words need room (eff/pur/pul)
                no_repeat_ngram_size=3, # prevent exact 3-gram repetition
                early_stopping=True
            )
            
            # Decode
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for sid, pred, ref in zip(study_ids, predictions, refs):
                ind = indications.get(sid, "radiology evaluation")
                styled_pred = customize_report_style(pred.strip(), ind)
                results.append({
                    "study_id": sid,
                    "prediction": styled_pred,
                    "reference": ref.strip()
                })
                
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} predictions to {out_file}")

if __name__ == "__main__":
    main()
