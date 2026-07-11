import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from typing import List, Dict, Any

from .chexpert_labels import (
    extract_chexpert_labels,
    labels_to_prompt_prefix,
    get_edit_actions,
    CHEXPERT_LABELS,
)


class RadiologyDataset(Dataset):
    """
    Dataset for VisionT5 radiology report generation (NeSy-CARE edition).

    Improvements:
      - Loads pre-computed TF-IDF / Visual Retrieval template report.
      - Computes target edit actions (e.g. "remove pleural effusion, add cardiomegaly")
        by comparing template labels with ground truth labels.
      - Generates edit prompt: "template: [text] edit: [actions]. generate report: [indication]"
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer,
        templates: Dict[str, str] = None,   # study_id -> template_report
        max_target_len: int = 256,
        max_source_len: int = 384,          # Expanded for long template + edits
        use_diagnosis_prompts: bool = True,
    ):
        self.examples              = examples
        self.tokenizer             = tokenizer
        self.templates             = templates if templates is not None else {}
        self.max_target_len        = max_target_len
        self.max_source_len        = max_source_len
        self.use_diagnosis_prompts = use_diagnosis_prompts

        # ImageNet normalisation — works for Swin-T and CNN backbones
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        study_id = item.get("study_id", "")

        # ── Image ──────────────────────────────────────────────────────────
        img_path = Path(item.get("image_path", ""))
        if img_path.exists() and img_path.is_file():
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=128)
        else:
            img = Image.new("RGB", (224, 224), color=128)
        image_tensor = self.transform(img)

        # ── CheXpert-14 labels from findings text (weak supervision) ───────
        report      = item.get("report", "")
        lbl_floats  = extract_chexpert_labels(report)   # list[float], len=14
        chexpert_t  = torch.tensor(lbl_floats, dtype=torch.float32)

        # ── NeSy-CARE template-editing prompt ──────────────────────────────
        indication = item.get("indication", "radiology evaluation")
        template_report = self.templates.get(study_id, "").strip()

        if self.use_diagnosis_prompts and template_report:
            # Get template labels and compute edits
            tpl_floats = extract_chexpert_labels(template_report)
            edit_actions = get_edit_actions(tpl_floats, lbl_floats)
            prompt = (
                f"template: {template_report} "
                f"edit: {edit_actions}. "
                f"generate report: {indication}"
            )
        else:
            prompt = f"generate report: {indication}"

        source_enc = self.tokenizer(
            prompt,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Target report ──────────────────────────────────────────────────
        target_enc = self.tokenizer(
            report,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "images":                 image_tensor,
            "encoder_input_ids":      source_enc["input_ids"].squeeze(0),
            "encoder_attention_mask": source_enc["attention_mask"].squeeze(0),
            "decoder_input_ids":      target_enc["input_ids"].squeeze(0),
            "labels":                 labels,
            "chexpert_labels":        chexpert_t,
            "study_id":               study_id,
            "raw_report":             report,
            "template_report":        template_report,
        }

