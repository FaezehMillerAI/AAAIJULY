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
    Keyword-based CheXpert-14 label extraction with clause-level negation detection.
    Returns a 14-dim float list (1.0 = present, 0.0 = absent).
    """
    labels = [0.0] * len(CHEXPERT_LABELS)
    if not text.strip():
        return labels

    text_lower = text.lower()
    # Split into clauses/sentences to avoid negation spillover
    clauses = re.split(r'[.,;!?]|\band\b|\bbut\b|\bhowever\b', text_lower)
    negation_words = {"no", "not", "without", "clear", "normal", "unremarkable", "free", "negative", "absent"}

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        
        for label, keywords in _KEYWORDS.items():
            if label == "No Finding":
                continue
            idx = _LABEL_INDEX[label]
            for kw in keywords:
                # Word boundary match for the keyword/phrase
                for match in re.finditer(r'\b' + re.escape(kw) + r'\b', clause):
                    # Check if a negation word occurs before the keyword in the same clause
                    words_before = clause[:match.start()].split()
                    is_neg = False
                    for w in words_before:
                        w_clean = re.sub(r'\W+', '', w)
                        if w_clean in negation_words:
                            is_neg = True
                            break
                    if not is_neg:
                        labels[idx] = 1.0
                        break

    # If no disease finding was positive, default to "No Finding"
    if sum(labels) == 0:
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


def apply_primekg_logic_rules(labels: List[float], probs: List[float] = None) -> List[float]:
    """
    Applies PrimeKG/LTN logical constraints to sanitize predictions before edit generation.
    - Rule 1: No Finding is mutually exclusive with any other finding.
    - Rule 2: Cardiomegaly implies Enlarged Cardiomediastinum.
    - Rule 3: Edema / Consolidation / Pneumonia / Atelectasis imply Lung Opacity.
    """
    labels = list(labels)  # copy
    
    # ── Rule 1: No Finding mutual exclusivity ──
    has_other = False
    max_other_prob = -1.0
    for idx in range(1, len(CHEXPERT_LABELS)):
        if labels[idx] > 0.5:
            has_other = True
            if probs is not None:
                max_other_prob = max(max_other_prob, probs[idx])
                
    if labels[0] > 0.5 and has_other:
        # Resolve conflict
        if probs is not None and probs[0] > max_other_prob:
            # Keep only "No Finding"
            labels = [0.0] * len(CHEXPERT_LABELS)
            labels[0] = 1.0
        else:
            # Clear "No Finding"
            labels[0] = 0.0

    # ── Rule 2: Cardiomegaly (2) -> Enlarged Cardiomediastinum (1) ──
    if labels[2] > 0.5:
        labels[1] = 1.0

    # ── Rule 3: Edema (5) / Consolidation (6) / Pneumonia (7) / Atelectasis (8) -> Lung Opacity (3) ──
    if labels[5] > 0.5 or labels[6] > 0.5 or labels[7] > 0.5 or labels[8] > 0.5:
        labels[3] = 1.0
        
    return labels


def get_edit_actions(
    tpl_labels: List[float],
    tgt_labels: List[float],
    tgt_probs: List[float] = None
) -> str:
    """
    Computes symbolic edit actions comparing the template labels to the target labels.
    Applies PrimeKG logic rules to filter out contradictory claims.
    Returns e.g. "remove pleural effusion, add cardiomegaly" or "none".
    """
    # Sanitize target labels with PrimeKG logic rules
    tgt_labels = apply_primekg_logic_rules(tgt_labels, tgt_probs)
    # Also sanitize template labels for consistency
    tpl_labels = apply_primekg_logic_rules(tpl_labels)

    actions = []
    for k, name in enumerate(CHEXPERT_LABELS):
        if name == "No Finding":
            continue
        if tpl_labels[k] > 0.5 and tgt_labels[k] <= 0.5:
            actions.append(f"remove {name.lower()}")
        elif tpl_labels[k] <= 0.5 and tgt_labels[k] > 0.5:
            actions.append(f"add {name.lower()}")
    
    if not actions:
        return "none"
    return ", ".join(actions)


