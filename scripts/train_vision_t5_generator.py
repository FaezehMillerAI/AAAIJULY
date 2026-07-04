import argparse
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import load_manifest, filter_manifest
from nesy_gen.vlm.model import VisionT5
from nesy_gen.vlm.dataset import RadiologyDataset
from nesy_gen.vlm.trainer import train_model

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--text-model-name", type=str, default="t5-small")
    parser.add_argument("--visual-backbone", type=str, default="densenet121")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-visual-encoder", type=str2bool, default=True)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--output-dir", type=str, default="output/vision_t5_checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    out_dir = Path(args.output_dir)
    
    print(f"Loading manifest from {manifest_path}...")
    examples = load_manifest(manifest_path)
    train_exs = filter_manifest(examples, "train")
    val_exs = filter_manifest(examples, "val")
    
    print(f"Train examples: {len(train_exs)}, Val examples: {len(val_exs)}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, use_fast=True)
    
    # Initialize model
    print("Initializing VisionT5 model...")
    model = VisionT5(
        text_model_name=args.text_model_name,
        visual_backbone=args.visual_backbone,
        freeze_visual_encoder=args.freeze_visual_encoder
    )
    
    # Initialize datasets
    train_dataset = RadiologyDataset(train_exs, tokenizer)
    val_dataset = RadiologyDataset(val_exs, tokenizer)
    
    # Start training
    print("Starting training...")
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fp16=args.fp16,
        device=args.device,
        checkpoint_dir=out_dir
    )
    
    # Save tokenizer in checkpoint directory
    tokenizer.save_pretrained(out_dir / "tokenizer")
    print(f"Tokenizer saved to {out_dir / 'tokenizer'}")

if __name__ == "__main__":
    main()
