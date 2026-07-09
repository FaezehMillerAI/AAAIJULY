import argparse
import pandas as pd
import json
from pathlib import Path
import sys
import torch
import os
import re
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Ensure workspace is on PATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import load_manifest, filter_manifest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--model-name", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--output-file", type=str, default="output/vision_t5_raw.csv")
    parser.add_argument("--quant", type=str, default="4bit", choices=["none", "4bit"])
    return parser.parse_args()

def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading test split from {manifest_path}...")
    examples = load_manifest(manifest_path)
    test_exs = filter_manifest(examples, "test")
    
    if not test_exs:
        print("No test examples found in manifest!")
        return
        
    print(f"Loading processor and VLM model: {args.model_name}...")
    
    # Load HF token if present in environment
    hf_token = os.environ.get("HF_TOKEN", None)
    
    proc = AutoProcessor.from_pretrained(args.model_name, token=hf_token, trust_remote_code=True)
    
    kw = dict(token=hf_token, device_map="cuda", trust_remote_code=True)
    if args.quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        kw["torch_dtype"] = torch.bfloat16
        
    model = AutoModelForImageTextToText.from_pretrained(args.model_name, **kw).eval()
    
    # Ensure pad token is set
    eos = model.generation_config.eos_token_id
    if eos is not None:
        eos0 = eos[0] if isinstance(eos, list) else eos
        model.generation_config.pad_token_id = eos0
        if getattr(proc, "tokenizer", None) is not None and proc.tokenizer.pad_token_id is None:
            proc.tokenizer.pad_token_id = eos0

    results = []
    print("Generating reports with VLM...")
    
    system_prompt = (
        "You are an expert radiologist. Analyze the provided frontal chest X-ray. "
        "Generate a concise, clinically accurate findings and impression report. "
        "Only describe findings that are visible in the image."
    )
    
    for ex in tqdm(test_exs, desc="VLM Generation"):
        sid = ex["study_id"]
        img_path = ex["image_path"]
        ind = ex.get("indication", "radiology evaluation")
        ref = ex.get("report", "")
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image at {img_path}: {e}")
            continue
            
        user_query = f"Generate report. Indication: {ind}."
        
        # Format prompt universally (supported by Qwen-VL, Gemma, LLaVA, etc.)
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{system_prompt}\n\n{user_query}"}
            ]}
        ]
        
        try:
            inp = proc.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to("cuda")
        except Exception:
            prompt = f"{system_prompt}\nUSER: <image>\n{user_query}\nASSISTANT:"
            inp = proc(images=image, text=prompt, return_tensors="pt").to("cuda")
            
        n = inp["input_ids"].shape[-1]
        
        with torch.inference_mode():
            out = model.generate(
                **inp,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                top_p=0.95
            )
            
        gen_text = proc.decode(out[0][n:], skip_special_tokens=True).strip()
        
        # Clean up any trailing system formatting
        gen_text = re.sub(r"<.*?|.*?>", "", gen_text).strip()
        
        results.append({
            "study_id": sid,
            "prediction": gen_text,
            "reference": ref
        })
        
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Saved VLM predictions to {out_file}")

if __name__ == "__main__":
    main()
