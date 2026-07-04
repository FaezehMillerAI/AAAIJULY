import json
from pathlib import Path
from typing import List, Dict, Any, Optional

REQUIRED_FIELDS = {"study_id", "image_path", "report", "indication", "split"}

def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Loads a JSONL manifest file."""
    examples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Verify fields
                for fld in REQUIRED_FIELDS:
                    if fld not in item:
                        item[fld] = ""  # Default empty if missing
                if "metadata" not in item:
                    item["metadata"] = {}
                examples.append(item)
    return examples

def save_manifest(examples: List[Dict[str, Any]], manifest_path: Path) -> None:
    """Saves a list of examples to a JSONL manifest file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item) + "\n")

def filter_manifest(examples: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    """Filters examples by split name."""
    return [item for item in examples if item.get("split") == split]

def generate_mock_manifest(output_dir: Path, image_dir: Path, num_train: int = 16, num_val: int = 4, num_test: int = 8) -> Path:
    """
    Generates a mock manifest and synthetic images for testing and smoke runs.
    """
    import numpy as np
    from PIL import Image

    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    findings_pool = [
        "heart size is normal",
        "cardiomegaly is present",
        "no pleural effusion or pneumothorax is seen",
        "mild bilateral pleural effusion is noted",
        "lungs are clear",
        "patchy opacities in the base of lungs",
        "hilar congestion is identified",
        "the mediastinum is stable"
    ]
    
    indications_pool = [
        "cough and fever",
        "chest pain",
        "shortness of breath",
        "pre-operative evaluation",
        "history of heart failure"
    ]
    
    examples = []
    splits = [("train", num_train), ("val", num_val), ("test", num_test)]
    
    idx = 0
    for split_name, count in splits:
        for _ in range(count):
            study_id = f"s{10000 + idx}"
            image_filename = f"{study_id}.png"
            image_path = image_dir / image_filename
            
            # Create a mock grayscale chest X-ray image (224x224)
            img_arr = np.zeros((224, 224), dtype=np.uint8)
            # draw a mock rib cage / lung outline
            # Left and right lungs as dark regions, rib cages as lines
            img_arr[40:180, 40:100] = 50  # Left lung
            img_arr[40:180, 120:180] = 50 # Right lung
            # Heart in the middle (brighter)
            img_arr[110:170, 90:130] = 120
            # Ribs (horizontal lines)
            for r in range(50, 180, 20):
                img_arr[r:r+3, 30:190] = img_arr[r:r+3, 30:190] + 40
            
            img = Image.fromarray(img_arr, mode="L")
            img.save(image_path)
            
            # Choose a combination of findings
            import random
            random.seed(idx)
            f1 = random.choice(findings_pool)
            f2 = random.choice(findings_pool)
            while f2 == f1:
                f2 = random.choice(findings_pool)
            
            report = f"Chest X-ray. Indication: {random.choice(indications_pool)}. Findings: {f1}. {f2}. Impression: no acute cardiopulmonary process."
            indication = random.choice(indications_pool)
            
            examples.append({
                "study_id": study_id,
                "image_path": str(image_path.absolute()),
                "report": report,
                "indication": indication,
                "split": split_name,
                "metadata": {"mock": True}
            })
            idx += 1
            
    manifest_path = output_dir / "common_manifest.jsonl"
    save_manifest(examples, manifest_path)
    return manifest_path
