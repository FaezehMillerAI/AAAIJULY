import argparse
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


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
    parser.add_argument("--text-model-name", type=str, default="razent/SciFive-base-PMC")
    parser.add_argument("--visual-backbone", type=str, default="swin_tiny",
                        choices=["swin_tiny", "swin_base", "densenet121", "resnet50",
                                 "efficientnet_b0", "efficientnet_b4"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze-visual-encoder", type=str2bool, default=False)
    parser.add_argument("--use-diagnosis-prompts", type=str2bool, default=True,
                        help="Prepend CheXpert-14 diagnosis prefix to encoder prompt")
    parser.add_argument("--cls-lambda", type=float, default=0.5,
                        help="Weight of classification BCE loss")
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--output-dir", type=str, default="output/vision_t5_checkpoint")
    
    # Auto-select mps on Mac if no device is specified
    default_dev = "cuda"
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        default_dev = "mps"
    parser.add_argument("--device", type=str, default=default_dev)
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
    print("Initializing VisionT5 model (Swin-T + Diagnosis Classifier)...")
    model = VisionT5(
        text_model_name=args.text_model_name,
        visual_backbone=args.visual_backbone,
        freeze_visual_encoder=args.freeze_visual_encoder,
        use_diagnosis_prompts=args.use_diagnosis_prompts,
        cls_lambda=args.cls_lambda,
    )

    # NeSy-CARE visual template retrieval setup
    train_templates = {}
    val_templates = {}
    if args.use_diagnosis_prompts:
        print("Extracting train image features for visual template retrieval (NeSy-CARE)...")
        from nesy_gen.retrieval.visual import VisualRetrieval
        # Determine device for retrieval
        if args.device == "mps" and torch.backends.mps.is_available():
            retrieval_device = "mps"
        else:
            retrieval_device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        retriever = VisualRetrieval(train_exs, device=retrieval_device)


        print("Computing training templates (leave-one-out)...")
        for ex in train_exs:
            candidates = retriever.retrieve(ex["image_path"], top_k=2)
            # Avoid self-retrieval
            if candidates and candidates[0]["study_id"] == ex["study_id"] and len(candidates) > 1:
                template = candidates[1]["report"]
            elif candidates:
                template = candidates[0]["report"]
            else:
                template = ""
            train_templates[ex["study_id"]] = template

        print("Computing validation templates...")
        for ex in val_exs:
            candidates = retriever.retrieve(ex["image_path"], top_k=1)
            val_templates[ex["study_id"]] = candidates[0]["report"] if candidates else ""

    # Initialize datasets
    train_dataset = RadiologyDataset(
        train_exs, tokenizer,
        templates=train_templates,
        use_diagnosis_prompts=args.use_diagnosis_prompts,
    )
    val_dataset = RadiologyDataset(
        val_exs, tokenizer,
        templates=val_templates,
        use_diagnosis_prompts=args.use_diagnosis_prompts,
    )

    
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
