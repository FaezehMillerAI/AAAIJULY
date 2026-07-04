import json
import argparse
from pathlib import Path
import sys
import re

# Ensure workspace is on PATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import save_manifest, generate_mock_manifest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iu-xray-root", type=str, default="/kaggle/input/datasets/rezakurniawan27/iu-xray")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    iu_root = Path(args.iu_xray_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    annot_path = iu_root / "annotation.json"
    
    if args.mock or not annot_path.exists():
        print(f"IU-Xray annotation.json not found at {annot_path} or mock specified. Generating mock manifest...")
        manifest_path = generate_mock_manifest(
            output_dir=out_dir,
            image_dir=out_dir / "mock_images",
            num_train=16,
            num_val=4,
            num_test=8
        )
        print(f"Mock manifest created at {manifest_path}")
        return
        
    print(f"Found annotation.json at {annot_path}. Building common manifest...")
    
    # Load official JSON
    with open(annot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Standard format check
    # Usually it is a list of studies
    if isinstance(data, dict):
        # check if there's a key containing the list
        for k, v in data.items():
            if isinstance(v, list):
                studies = v
                break
        else:
            studies = []
    else:
        studies = data
        
    examples = []
    for study in studies:
        study_id = str(study.get("id", ""))
        report = study.get("report", "")
        split = study.get("split", "train")
        
        # Clinical indication if present
        indication = study.get("indication", "")
        if not indication:
            # try to extract from report or metadata
            match = re.search(r"indication:\s*(.*?)\.\s", report.lower())
            if match:
                indication = match.group(1)
                
        # Resolve images: study can have multiple images
        images = study.get("image_path", [])
        if isinstance(images, str):
            images = [images]
            
        for img in images:
            img_path = iu_root / img
            
            examples.append({
                "study_id": study_id,
                "image_path": str(img_path.absolute()),
                "report": report,
                "indication": indication,
                "split": split,
                "metadata": {"source": "iu-xray"}
            })
            
    manifest_path = out_dir / "common_manifest.jsonl"
    save_manifest(examples, manifest_path)
    print(f"Built common manifest at {manifest_path} with {len(examples)} entries.")

if __name__ == "__main__":
    main()
