import numpy as np
import pandas as pd
import re
import json
import nltk
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score

# Ensure NLTK packages are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# -----------------
# Lexical Metrics
# -----------------

def compute_lcs(x: List[str], y: List[str]) -> int:
    """Computes the length of the Longest Common Subsequence between x and y."""
    n, m = len(x), len(y)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i-1] == y[j-1]:
                table[i][j] = table[i-1][j-1] + 1
            else:
                table[i][j] = max(table[i-1][j], table[i][j-1])
    return table[n][m]

def compute_rouge_l(pred: str, ref: str) -> float:
    """Computes ROUGE-L F1 score using Word LCS."""
    p_words = re.findall(r'\w+', pred.lower())
    r_words = re.findall(r'\w+', ref.lower())
    if not p_words or not r_words:
        return 0.0
    lcs_len = compute_lcs(p_words, r_words)
    precision = lcs_len / len(p_words)
    recall = lcs_len / len(r_words)
    
    # Standard beta for ROUGE-L is often 1.22 or 1.0 (we use 1.0 for F1)
    beta = 1.0
    if precision + recall == 0:
        return 0.0
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)

def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_cider_approx(preds: List[str], refs: List[str]) -> float:
    """
    Computes a simplified corpus-level CIDEr metric based on n-grams (1 to 4) 
    weighted by corpus-level TF-IDF.
    """
    # Tokenize everything
    pred_tokens = [re.findall(r'\w+', p.lower()) for p in preds]
    ref_tokens = [re.findall(r'\w+', r.lower()) for r in refs]
    
    # Calculate DF (Document Frequency) of n-grams in references
    df = {n: Counter() for n in range(1, 5)}
    num_docs = len(refs)
    
    for tokens in ref_tokens:
        for n in range(1, 5):
            ngs = set(get_ngrams(tokens, n))
            for ng in ngs:
                df[n][ng] += 1
                
    # Calculate similarity for each document
    cider_scores = []
    
    for i in range(num_docs):
        p_tok = pred_tokens[i]
        r_tok = ref_tokens[i]
        
        doc_scores = []
        for n in range(1, 5):
            p_ngs = Counter(get_ngrams(p_tok, n))
            r_ngs = Counter(get_ngrams(r_tok, n))
            
            if not p_ngs or not r_ngs:
                doc_scores.append(0.0)
                continue
                
            # Compute TF-IDF vectors
            p_vec = {}
            r_vec = {}
            
            all_ngs = set(p_ngs.keys()).union(r_ngs.keys())
            for ng in all_ngs:
                # DF fallback
                df_val = df[n].get(ng, 0)
                # inverse document frequency
                idf = np.log(num_docs / max(1.0, df_val))
                
                p_vec[ng] = p_ngs[ng] * idf
                r_vec[ng] = r_ngs[ng] * idf
                
            # Cosine similarity
            dot_prod = sum(p_vec[ng] * r_vec[ng] for ng in p_vec if ng in r_vec)
            p_norm = np.sqrt(sum(v**2 for v in p_vec.values()))
            r_norm = np.sqrt(sum(v**2 for v in r_vec.values()))
            
            if p_norm * r_norm == 0:
                doc_scores.append(0.0)
            else:
                doc_scores.append(dot_prod / (p_norm * r_norm))
                
        # Average across 1-to-4 n-grams
        cider_scores.append(np.mean(doc_scores) * 10.0)  # scale standard in CIDEr
        
    return float(np.mean(cider_scores))

def compute_lexical_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Computes BLEU-1..4, ROUGE-L, and approximate CIDEr."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    b1_list, b2_list, b3_list, b4_list = [], [], [], []
    rouge_list = []
    
    smooth = SmoothingFunction().method1
    
    for p, r in zip(preds, refs):
        p_tok = re.findall(r'\w+', p.lower())
        r_tok = re.findall(r'\w+', r.lower())
        
        # BLEU scores
        b1_list.append(sentence_bleu([r_tok], p_tok, weights=(1.0, 0, 0, 0), smoothing_function=smooth))
        b2_list.append(sentence_bleu([r_tok], p_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        b3_list.append(sentence_bleu([r_tok], p_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
        b4_list.append(sentence_bleu([r_tok], p_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
        
        # ROUGE-L
        rouge_list.append(compute_rouge_l(p, r))
        
    cider_val = compute_cider_approx(preds, refs)
    
    # METEOR fallback (using a robust unigram overlap score if wordnet is missing)
    meteor_list = []
    for p, r in zip(preds, refs):
        p_tok = set(re.findall(r'\w+', p.lower()))
        r_tok = set(re.findall(r'\w+', r.lower()))
        overlap = p_tok.intersection(r_tok)
        if not p_tok or not r_tok:
            meteor_list.append(0.0)
            continue
        prec = len(overlap) / len(p_tok)
        rec = len(overlap) / len(r_tok)
        fmean = (10 * prec * rec) / (9 * prec + rec) if (9 * prec + rec) > 0 else 0.0
        meteor_list.append(fmean)
        
    return {
        "BLEU-1": float(np.mean(b1_list)),
        "BLEU-2": float(np.mean(b2_list)),
        "BLEU-3": float(np.mean(b3_list)),
        "BLEU-4": float(np.mean(b4_list)),
        "METEOR": float(np.mean(meteor_list)),
        "ROUGE-L": float(np.mean(rouge_list)),
        "CIDEr": cider_val
    }

# -----------------
# CheXpert-Lite
# -----------------

CHEXPERT_CONDITIONS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", 
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", 
    "Pleural Other", "Fracture", "Support Devices"
]

# Lexical mapping for conditions (includes negation rules)
CONDITION_PATTERNS = {
    "Cardiomegaly": [r"\bcardiomegaly\b", r"\benlarged cardiac silhouette\b", r"\benlarged heart\b", r"\bheart is enlarged\b"],
    "Pleural Effusion": [r"\bpleural effusion\b", r"\beffusion\b", r"\bfluid\b"],
    "Pneumothorax": [r"\bpneumothorax\b", r"\bair in the pleural\b"],
    "Atelectasis": [r"\batelectasis\b", r"\bcollapse\b"],
    "Consolidation": [r"\bconsolidation\b", r"\bairspace disease\b", r"\binfiltrate\b"],
    "Edema": [r"\bedema\b", r"\bcongestion\b", r"\bpulmonary edema\b"],
    "Pneumonia": [r"\bpneumonia\b", r"\binfection\b"],
    "Support Devices": [r"\btube\b", r"\bcatheter\b", r"\bpacemaker\b", r"\bline\b", r"\bhardware\b"],
    "Fracture": [r"\bfracture\b", r"\broken rib\b"]
}

def extract_chexpert_labels(text: str) -> List[int]:
    """Extracts binary labels (1=positive, 0=absent/negated) for the 14 conditions."""
    text_lower = text.lower()
    labels = [0] * len(CHEXPERT_CONDITIONS)
    
    # 1. Check specific conditions
    has_finding = False
    for idx, cond in enumerate(CHEXPERT_CONDITIONS):
        if cond in ["No Finding", "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion", "Pleural Other"]:
            continue
            
        patterns = CONDITION_PATTERNS.get(cond, [])
        for pat in patterns:
            match = re.search(pat, text_lower)
            if match:
                # Check for negation in proximity of 30 characters
                start = match.start()
                context = text_lower[max(0, start-30):start]
                # If negative words exist, it is negated (so we leave it as 0)
                is_neg = False
                for neg in ["no ", "not ", "without ", "free of", "clear of", "rules out"]:
                    if neg in context:
                        is_neg = True
                        break
                if not is_neg:
                    labels[idx] = 1
                    has_finding = True
                    break
                    
    # 2. Assign No Finding
    if not has_finding:
        labels[0] = 1
        
    return labels

def evaluate_chexpert_lite(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    pred_labels = [extract_chexpert_labels(p) for p in preds]
    ref_labels = [extract_chexpert_labels(r) for r in refs]
    
    pred_arr = np.array(pred_labels)
    ref_arr = np.array(ref_labels)
    
    f1_scores = []
    class_f1s = {}
    
    for idx, cond in enumerate(CHEXPERT_CONDITIONS):
        y_true = ref_arr[:, idx]
        y_pred = pred_arr[:, idx]
        
        # Calculate F1
        score = f1_score(y_true, y_pred, zero_division=0.0)
        f1_scores.append(score)
        class_f1s[cond] = float(score)
        
    return {
        "macro_f1": float(np.mean(f1_scores)),
        "class_scores": class_f1s,
        "raw_predictions": pred_labels,
        "raw_references": ref_labels
    }

# -----------------
# RadGraph-Lite
# -----------------

def extract_radgraph_triplets(text: str) -> Set[Tuple[str, str, str]]:
    """
    Extracts proxy triplets: (finding, "occurs_in", anatomy) 
    if they appear together in the same sentence/clause.
    """
    sentences = re.split(r'[.!?]\s+', text.lower())
    triplets = set()
    
    findings = ["cardiomegaly", "effusion", "pneumothorax", "atelectasis", "consolidation", "opacity", "congestion", "pneumonia", "infiltrate"]
    anatomies = ["heart", "lungs", "pleural", "mediastinum", "hilar", "diaphragm", "silhouette"]
    
    for sent in sentences:
        # Detect negation
        is_negated = False
        for neg in ["no ", "not ", "without ", "free of", "clear of"]:
            if neg in sent:
                is_negated = True
                
        # Find which findings and anatomies are present
        found_f = [f for f in findings if f in sent]
        found_a = [a for a in anatomies if a in sent]
        
        # Create relation occurrences
        for f in found_f:
            for a in found_a:
                rel = "absent_in" if is_negated else "occurs_in"
                triplets.add((f, rel, a))
                
    return triplets

def evaluate_radgraph_lite(preds: List[str], refs: List[str]) -> Dict[str, float]:
    p_triplets = [extract_radgraph_triplets(p) for p in preds]
    r_triplets = [extract_radgraph_triplets(r) for r in refs]
    
    prec_list, rec_list, f1_list = [], [], []
    for p_trip, r_trip in zip(p_triplets, r_triplets):
        if not p_trip and not r_trip:
            # Vacuously perfect
            prec_list.append(1.0)
            rec_list.append(1.0)
            f1_list.append(1.0)
            continue
        if not p_trip or not r_trip:
            prec_list.append(0.0)
            rec_list.append(0.0)
            f1_list.append(0.0)
            continue
            
        intersection = p_trip.intersection(r_trip)
        p = len(intersection) / len(p_trip)
        r = len(intersection) / len(r_trip)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)
        
    return {
        "precision": float(np.mean(prec_list)),
        "recall": float(np.mean(rec_list)),
        "f1": float(np.mean(f1_list))
    }

# -----------------
# Entity Factuality
# -----------------

def evaluate_entity_factuality(preds: List[str], refs: List[str], kg_cache: Any) -> Dict[str, float]:
    """
    Computes F1 of positive/negated PrimeKG entities in predictions vs references.
    """
    prec_list, rec_list, f1_list = [], [], []
    
    for p, r in zip(preds, refs):
        p_ents = {(ent["node_id"], ent["negated"]) for ent in kg_cache.link_entities(p)}
        r_ents = {(ent["node_id"], ent["negated"]) for ent in kg_cache.link_entities(r)}
        
        if not p_ents and not r_ents:
            prec_list.append(1.0)
            rec_list.append(1.0)
            f1_list.append(1.0)
            continue
        if not p_ents or not r_ents:
            prec_list.append(0.0)
            rec_list.append(0.0)
            f1_list.append(0.0)
            continue
            
        inter = p_ents.intersection(r_ents)
        p_score = len(inter) / len(p_ents)
        r_score = len(inter) / len(r_ents)
        f1 = (2 * p_score * r_score) / (p_score + r_score) if (p_score + r_score) > 0 else 0.0
        
        prec_list.append(p_score)
        rec_list.append(r_score)
        f1_list.append(f1)
        
    return {
        "precision": float(np.mean(prec_list)),
        "recall": float(np.mean(rec_list)),
        "f1": float(np.mean(f1_list))
    }

# -----------------
# Leakage Audit
# -----------------

def run_leakage_audit(preds: List[str], refs: List[str], train_corpus: List[str]) -> Dict[str, Any]:
    """Audits prediction/reference overlaps and training data leakage."""
    exact_copies_in_train = 0
    high_similarity_leakage = 0
    
    train_set = {t.strip().lower() for t in train_corpus}
    
    overlaps = []
    
    for p, r in zip(preds, refs):
        p_clean = p.strip().lower()
        
        # Check training set copy
        is_leak = p_clean in train_set
        if is_leak:
            exact_copies_in_train += 1
            
        # Jaccard overlap check against reference
        p_words = set(re.findall(r'\w+', p.lower()))
        r_words = set(re.findall(r'\w+', r.lower()))
        jaccard = len(p_words.intersection(r_words)) / len(p_words.union(r_words)) if p_words else 0.0
        overlaps.append(jaccard)
        
        if jaccard > 0.95:
            high_similarity_leakage += 1
            
    return {
        "exact_copies_in_train_count": exact_copies_in_train,
        "exact_copies_in_train_rate": float(exact_copies_in_train / max(1, len(preds))),
        "average_pred_ref_jaccard": float(np.mean(overlaps)),
        "high_pred_ref_similarity_count": high_similarity_leakage,
        "leakage_alert": exact_copies_in_train > 0 or high_similarity_leakage > 0
    }

# -----------------
# HTML Qualitative Report
# -----------------

def generate_html_report(results_df: pd.DataFrame, traces_list: List[Dict[str, Any]], output_path: Path):
    """
    Generates a gorgeous interactive HTML report of generation systems,
    including details of the raw draft and adaptive verified claims.
    """
    # Group claims by study_id
    study_traces = {}
    for tr in traces_list:
        sid = tr["study_id"]
        if sid not in study_traces:
            study_traces[sid] = []
        study_traces[sid].append(tr)
        
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AAAI Report Generation Qualitative Analysis</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body {
                background-color: #0c0e12;
                color: #e2e8f0;
                font-family: 'Outfit', sans-serif;
                margin: 0;
                padding: 24px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #38bdf8;
                font-weight: 600;
                margin-bottom: 8px;
            }
            .subtitle {
                text-align: center;
                color: #94a3b8;
                margin-bottom: 40px;
            }
            .card {
                background: rgba(30, 41, 59, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(8px);
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                margin-bottom: 20px;
            }
            .pane {
                background: rgba(15, 23, 42, 0.6);
                border-radius: 12px;
                padding: 16px;
                border-left: 4px solid #38bdf8;
            }
            .pane-ref { border-left-color: #10b981; }
            .pane-rev { border-left-color: #a855f7; }
            h3 {
                margin-top: 0;
                font-size: 1.1rem;
                font-weight: 600;
                color: #f1f5f9;
            }
            .badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-right: 8px;
            }
            .badge-fast { background-color: #0369a1; color: #e0f2fe; }
            .badge-esc-acc { background-color: #15803d; color: #dcfce7; }
            .badge-esc-rep { background-color: #6b21a8; color: #f3e8ff; }
            .badge-esc-unv { background-color: #991b1b; color: #fee2e2; }
            
            .timeline {
                list-style-type: none;
                padding: 0;
                margin-top: 15px;
            }
            .timeline-item {
                position: relative;
                padding-left: 24px;
                margin-bottom: 12px;
                border-left: 2px solid #334155;
            }
            .timeline-item::before {
                content: '';
                position: absolute;
                left: -6px;
                top: 6px;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #38bdf8;
            }
            .diff-added { color: #34d399; font-weight: 600; }
            .diff-removed { color: #f87171; text-decoration: line-through; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Neuro-Symbolic Adaptive Verification Trace Auditor</h1>
            <div class="subtitle">AAAI Qualitative Evaluation - System Output Comparison</div>
    """
    
    # Loop over top 10 studies
    count = 0
    for _, row in results_df.iterrows():
        sid = row["study_id"]
        draft = row["original_draft"]
        pred = row["prediction"]
        ref = row["reference"]
        
        traces = study_traces.get(sid, [])
        
        html_content += f"""
            <div class="card">
                <h2>Study ID: {sid}</h2>
                <div class="grid">
                    <div class="pane">
                        <h3>Raw VLM Draft</h3>
                        <p>{draft}</p>
                    </div>
                    <div class="pane pane-ref">
                        <h3>Reference Report</h3>
                        <p>{ref}</p>
                    </div>
                </div>
                
                <div class="pane pane-rev" style="margin-bottom: 20px;">
                    <h3>Verified & Revised Output (Proposed)</h3>
                    <p>{pred}</p>
                </div>
                
                <h3>Claim-by-Claim Verification Traces</h3>
                <ul class="timeline">
        """
        
        for tr in traces:
            orig = tr["original_text"]
            rev = tr["revised_text"]
            dec = tr["decision"]
            sup = tr["support_score"]
            ltn = tr["ltn_score"]
            
            badge_class = "badge-fast"
            if dec == "escalated_accept":
                badge_class = "badge-esc-acc"
            elif dec == "escalated_replaced":
                badge_class = "badge-esc-rep"
            elif dec == "escalated_keep_unverified":
                badge_class = "badge-esc-unv"
                
            text_disp = orig
            if dec == "escalated_replaced":
                text_disp = f'<span class="diff-removed">{orig}</span> &rarr; <span class="diff-added">{rev}</span>'
                
            html_content += f"""
                    <li class="timeline-item">
                        <span class="badge {badge_class}">{dec.upper()}</span> 
                        <span>Ret-Support: {sup:.2f} | LTN-Graph: {ltn:.2f}</span>
                        <div style="margin-top: 4px; color: #cbd5e1;">{text_disp}</div>
                    </li>
            """
            
        html_content += """
                </ul>
            </div>
        """
        count += 1
        if count >= 10:  # Show 10 examples to not balloon file size
            break
            
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
