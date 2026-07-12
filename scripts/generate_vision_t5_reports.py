import argparse
import pandas as pd
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer
import ssl

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    
    # Auto-select mps on Mac if no device is specified
    default_dev = "cuda"
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        default_dev = "mps"
    parser.add_argument("--device", type=str, default=default_dev)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    ckpt_dir = Path(args.checkpoint_dir)
    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    
    # Load manifest splits
    print(f"Loading manifest from {manifest_path}...")
    examples = load_manifest(manifest_path)
    train_exs = filter_manifest(examples, "train")
    test_exs = filter_manifest(examples, "test")
    print(f"Train examples (template DB): {len(train_exs)}, Test examples: {len(test_exs)}")
    
    if not test_exs:
        print("No test examples found. Exiting.")
        return
        
    # Load saved config first to retrieve the exact text model name
    import json
    config_path = ckpt_dir / "config.json"
    ckpt_text_model        = "razent/SciFive-base-PMC"
    ckpt_visual_backbone   = "swin_tiny"
    ckpt_freeze            = False
    ckpt_diag_prompts      = True
    ckpt_cls_lambda        = 0.5
    if config_path.exists():
        with open(config_path) as f:
            ckpt_cfg = json.load(f)
        ckpt_text_model      = ckpt_cfg.get("text_model_name",       ckpt_text_model)
        ckpt_visual_backbone = ckpt_cfg.get("visual_backbone",       ckpt_visual_backbone)
        ckpt_freeze          = ckpt_cfg.get("freeze_visual_encoder", ckpt_freeze)
        ckpt_diag_prompts    = ckpt_cfg.get("use_diagnosis_prompts", ckpt_diag_prompts)
        ckpt_cls_lambda      = ckpt_cfg.get("cls_lambda",            ckpt_cls_lambda)

    # Load tokenizer and model
    print("Loading tokenizer and model from checkpoint...")
    
    local_tokenizer_path = ckpt_dir / "tokenizer"
    try:
        # Check if local directory exists, and pass as a string (safer for HF from_pretrained)
        if local_tokenizer_path.exists() and local_tokenizer_path.is_dir():
            tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer_path), use_fast=True)
        else:
            raise FileNotFoundError(f"Local tokenizer folder {local_tokenizer_path} not found.")
    except Exception as e:
        print(f"Warning: Failed to load local tokenizer ({e}). Falling back to HF Hub: {ckpt_text_model}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_text_model, use_fast=True)



    model = VisionT5(
        text_model_name=ckpt_text_model,
        visual_backbone=ckpt_visual_backbone,
        freeze_visual_encoder=ckpt_freeze,
        use_diagnosis_prompts=ckpt_diag_prompts,
        cls_lambda=ckpt_cls_lambda,
    )
    model.load_checkpoint(ckpt_dir)
    model.to(device)
    model.eval()

    # Pre-compute test templates
    test_templates = {}
    if ckpt_diag_prompts and train_exs:
        print("Extracting train image features for visual template retrieval (NeSy-CARE)...")
        from nesy_gen.retrieval.visual import VisualRetrieval
        if args.device == "mps" and torch.backends.mps.is_available():
            retrieval_device = "mps"
        else:
            retrieval_device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        retriever = VisualRetrieval(train_exs, device=retrieval_device)

        
        print("Computing test templates...")
        for ex in test_exs:
            candidates = retriever.retrieve(ex["image_path"], top_k=1)
            test_templates[ex["study_id"]] = candidates[0]["report"] if candidates else ""
    
    test_dataset = RadiologyDataset(
        test_exs, tokenizer,
        templates=test_templates,
        use_diagnosis_prompts=ckpt_diag_prompts,
    )
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
            
            # Generate: model.generate() runs classifier → injects diagnosis prefix → decodes
            generated_ids = model.generate(
                images=images,
                encoder_input_ids=enc_ids,
                encoder_attention_mask=enc_mask,
                tokenizer=tokenizer,               # enables diagnosis-prompt injection
                template_reports=batch.get("template_report"),  # NeSy-CARE templates
                max_new_tokens=args.max_new_tokens,
                num_beams=4,
                length_penalty=1.5,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True,
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
