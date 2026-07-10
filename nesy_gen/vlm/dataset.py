import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from typing import List, Dict, Any

class RadiologyDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_target_len: int = 256, max_source_len: int = 96):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
        self.max_source_len = max_source_len
        
        # Standard torchvision preprocessing for DenseNet
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        item = self.examples[idx]
        image_path_str = item.get("image_path", "")
        img_path = Path(image_path_str)
        
        # Load image; fallback to synthesized placeholder if missing
        if img_path.exists() and img_path.is_file():
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=128)
        else:
            img = Image.new("RGB", (224, 224), color=128)
            
        image_tensor = self.transform(img)
        
        # Process input text (e.g. prompt + indication)
        indication = item.get("indication", "")
        prompt = f"generate report: {indication}".strip()
        
        source_encoding = self.tokenizer(
            prompt,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Process target text (report)
        report = item.get("report", "")
        target_encoding = self.tokenizer(
            report,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encoding["input_ids"].squeeze(0)
        # In PyTorch, labels with value -100 are ignored in cross-entropy loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "images": image_tensor,
            "encoder_input_ids": source_encoding["input_ids"].squeeze(0),
            "encoder_attention_mask": source_encoding["attention_mask"].squeeze(0),
            "decoder_input_ids": target_encoding["input_ids"].squeeze(0), # for T5 text inputs
            "labels": labels,
            "study_id": item.get("study_id", ""),
            "raw_report": report
        }
