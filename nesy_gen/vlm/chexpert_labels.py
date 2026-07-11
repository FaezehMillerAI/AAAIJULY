"""
CheXpert-14 keyword-based label extractor.
Used during training to derive weak supervision labels from findings text,
and to seed PrimeKG axioms for the NeSy verification layer.
"""
from typing import List, Dict
import re

# CheXpert-14 canonical labels (matches CheXbert / PromptMRG ordering)
CHEXPERT_LABELS: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Keyword triggers per label (case-insensitive)
_KEYWORDS: Dict[str, List[str]] = {
    "No Finding": [
        "no acute", "normal", "unremarkable", "clear", "no evidence",
        "no significant", "no abnormality", "no pathology"
    ],
    "Enlarged Cardiomediastinum": [
        "enlarged mediastinum", "widened mediastinum", "mediastinal widening",
        "prominent mediastinum"
    ],
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "enlarged heart",
        "increased cardiac silhouette", "cardiomegaly is present",
        "heart is enlarged"
    ],
    "Lung Opacity": [
        "opacity", "opacities", "haziness", "hazy", "airspace disease",
        "airspace opacity"
    ],
    "Lung Lesion": [
        "nodule", "nodules", "mass", "lesion", "lesions", "lung mass"
    ],
    "Edema": [
        "edema", "pulmonary edema", "interstitial edema", "alveolar edema",
        "vascular congestion", "hilar congestion", "perivascular cuffing"
    ],
    "Consolidation": [
        "consolidation", "consolidated", "airspace consolidation",
        "lobar consolidation", "segmental consolidation"
    ],
    "Pneumonia": [
        "pneumonia", "pneumonic", "infection", "infectious", "infiltrate",
        "infiltrates", "bronchopneumonia"
    ],
    "Atelectasis": [
        "atelectasis", "atelectatic", "subsegmental atelectasis",
        "discoid atelectasis", "linear atelectasis", "plate-like atelectasis",
        "volume loss"
    ],
    "Pneumothorax": [
        "pneumothorax", "pneumothoraces", "collapsed lung"
    ],
    "Pleural Effusion": [
        "pleural effusion", "effusion", "pleural fluid", "bilateral effusion",
        "small effusion", "moderate effusion", "large effusion"
    ],
    "Pleural Other": [
        "pleural thickening", "pleural plaque", "pleural calcification",
        "pleural scarring"
    ],
    "Fracture": [
        "fracture", "rib fracture", "fractured", "acute fracture",
        "vertebral fracture", "compression fracture"
    ],
    "Support Devices": [
        "tube", "tubes", "line", "lines", "catheter", "pacemaker",
        "defibrillator", "icd", "stent", "valve", "device", "support device",
        "endotracheal", "nasogastric", "central line", "chest tube"
    ],
}

_LABEL_INDEX: Dict[str, int] = {lbl: i for i, lbl in enumerate(CHEXPERT_LABELS)}


def extract_chexpert_labels(text: str) -> List[float]:
    """
    Keyword-based CheXpert-14 label extraction from free-form findings text.
    Returns a 14-dim float list (1.0 = present, 0.0 = absent).
    
    NOTE: This is weak-supervision labelling — not a replacement for
    CheXbert. Adequate for training signal; not for clinical use.
    """
    text_lower = text.lower()
    labels = [0.0] * len(CHEXPERT_LABELS)

    for label, keywords in _KEYWORDS.items():
        idx = _LABEL_INDEX[label]
        for kw in keywords:
            if kw in text_lower:
                labels[idx] = 1.0
                break

    # If nothing found and text is non-empty, tentatively mark "No Finding"
    if sum(labels) == 0 and text.strip():
        labels[_LABEL_INDEX["No Finding"]] = 1.0

    return labels


def labels_to_prompt_prefix(labels: List[float], threshold: float = 0.5) -> str:
    """
    Converts a 14-dim label vector to a natural-language diagnosis prefix
    that is prepended to the T5 encoder prompt.

    Example output: "diagnosis: cardiomegaly, pleural effusion."
    Returns empty string if no positive labels.
    """
    active = [
        CHEXPERT_LABELS[i].lower()
        for i, v in enumerate(labels)
        if v >= threshold and CHEXPERT_LABELS[i] != "No Finding"
    ]
    if not active:
        if labels[_LABEL_INDEX["No Finding"]] >= threshold:
            return "diagnosis: no acute finding."
        return ""
    return "diagnosis: " + ", ".join(active) + "."
