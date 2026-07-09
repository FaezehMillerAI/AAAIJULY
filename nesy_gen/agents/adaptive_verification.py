import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from nesy_gen.kg.primekg import PrimeKGRadiologyCache
from nesy_gen.logic.ltn import evaluate_ltn_constraints

def split_into_claims(text: str) -> List[str]:
    """Splits a paragraph into sentence-level claims."""
    # Split by standard sentence terminators, avoiding abbreviations like Dr. or index numbers if possible
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def jaccard_similarity(s1: str, s2: str) -> float:
    """Computes word-level Jaccard similarity."""
    w1 = set(re.findall(r'\w+', s1.lower()))
    w2 = set(re.findall(r'\w+', s2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))

class AdaptiveClaimVerifier:
    def __init__(
        self,
        kg_cache: PrimeKGRadiologyCache,
        fast_accept_threshold: float = 0.75,
        min_supporting_reports: int = 1
    ):
        self.kg_cache = kg_cache
        self.fast_accept_threshold = fast_accept_threshold
        self.min_supporting_reports = min_supporting_reports
        
    def verify_and_revise(
        self,
        study_id: str,
        draft_report: str,
        retrieved_candidates: List[Dict[str, Any]],
        policy: str = "evidence_replace"  # "audit_only" or "evidence_replace"
    ) -> Dict[str, Any]:
        """
        Runs claim-level adaptive verification on a single report draft.
        """
        if not isinstance(draft_report, str):
            draft_report = ""
            
        # Link entities first to check clinical substance
        linked_draft = self.kg_cache.link_entities(draft_report)
        
        # Check maximum Jaccard similarity against top candidates to detect collapse/looping
        max_support = 0.0
        if retrieved_candidates:
            max_support = max(jaccard_similarity(draft_report, cand["report"]) for cand in retrieved_candidates)
            
        # Defensive fallback: if draft is empty, too short (under 10 words), contains no clinical entities,
        # or has extremely low alignment with retrieval candidates (indicates generator collapse/looping)
        clean_words = [w for w in re.findall(r'\w+', draft_report) if not w.isdigit()]
        if len(clean_words) < 10 or len(linked_draft) == 0 or max_support < 0.25:
            if retrieved_candidates:
                best_cand_text = retrieved_candidates[0]["report"]
                best_score = -1.0
                for cand in retrieved_candidates:
                    report_text = cand["report"]
                    ret_score = cand.get("score", 1.0)
                    
                    linked = self.kg_cache.link_entities(report_text)
                    ltn_score = evaluate_ltn_constraints(linked, self.kg_cache)["overall_score"]
                    
                    # Combined score with alpha=0.5
                    comb_score = 0.5 * ret_score + 0.5 * ltn_score
                    if comb_score > best_score:
                        best_score = comb_score
                        best_cand_text = report_text
                draft_report = best_cand_text
                
        claims = split_into_claims(draft_report)
        revised_claims = []
        claim_traces = []
        
        # Prepare candidates sentences for potential replacement
        evidence_sentences = []
        for cand in retrieved_candidates:
            cand_sentences = split_into_claims(cand["report"])
            for s in cand_sentences:
                evidence_sentences.append({
                    "text": s,
                    "study_id": cand["study_id"],
                    "score": cand.get("score", 1.0)
                })
                
        for idx, claim in enumerate(claims):
            # 1. Link entities in claim
            linked = self.kg_cache.link_entities(claim)
            
            # 2. Compute retrieval support score
            support_score = 0.0
            best_matching_ev_sent = ""
            for ev in evidence_sentences:
                sim = jaccard_similarity(claim, ev["text"])
                if sim > support_score:
                    support_score = sim
                    best_matching_ev_sent = ev["text"]
                    
            decision = "unknown"
            revised_text = claim
            ltn_metrics = {}
            
            # 3. Decision Routing
            if support_score >= self.fast_accept_threshold:
                # Fast accept path
                decision = "fast_accept"
                revised_text = claim
            else:
                # Escalation path to PrimeKG/LTN soft logic verifier
                decision = "escalated"
                ltn_metrics = evaluate_ltn_constraints(linked, self.kg_cache)
                overall_ltn = ltn_metrics["overall_score"]
                
                if overall_ltn >= 0.5:
                    decision = "escalated_accept"
                    revised_text = claim
                else:
                    decision = "escalated_reject"
                    if policy == "evidence_replace":
                        # Find replacement candidate from retrieval evidence
                        # We want a candidate sentence that has the highest Jaccard overlap but is verified
                        best_rep_text = None
                        best_rep_score = -1.0
                        
                        # Match anatomies in the rejected claim to find relevant candidates
                        claim_anatomies = {e["node_id"] for e in linked if e["node_type"] == "anatomy"}
                        
                        for ev in evidence_sentences:
                            ev_linked = self.kg_cache.link_entities(ev["text"])
                            ev_ltn = evaluate_ltn_constraints(ev_linked, self.kg_cache)["overall_score"]
                            
                            # Only consider candidate if it passes LTN verification itself
                            if ev_ltn >= 0.5:
                                ev_anatomies = {e["node_id"] for e in ev_linked if e["node_type"] == "anatomy"}
                                # Check if they share anatomies or context
                                if not claim_anatomies or claim_anatomies.intersection(ev_anatomies):
                                    sim = jaccard_similarity(claim, ev["text"])
                                    if sim > best_rep_score:
                                        best_rep_score = sim
                                        best_rep_text = ev["text"]
                                        
                        if best_rep_text and best_rep_score > 0.1:
                            revised_text = best_rep_text
                            decision = "escalated_replaced"
                        else:
                            # If no replacement matches anatomies, fallback to a standard normal statement or keep as is
                            # We keep as is but flag it
                            revised_text = claim
                            decision = "escalated_keep_unverified"
                    else:
                        # audit_only
                        revised_text = claim
                        
            revised_claims.append(revised_text)
            
            claim_traces.append({
                "study_id": study_id,
                "claim_index": idx,
                "original_text": claim,
                "revised_text": revised_text,
                "entities": [e["node_name"] for e in linked],
                "negations": [e["negated"] for e in linked],
                "support_score": float(support_score),
                "ltn_score": float(ltn_metrics.get("overall_score", 1.0)),
                "decision": decision
            })
            
        final_report = " ".join(revised_claims)
        
        return {
            "study_id": study_id,
            "prediction": final_report,
            "original_draft": draft_report,
            "traces": claim_traces
        }

def run_adaptive_verification_pipeline(
    raw_predictions: List[Dict[str, Any]],  # List of {"study_id": ..., "prediction": ...}
    retrieval_cache: Dict[str, List[Dict[str, Any]]],  # study_id -> top-k retrieved list
    kg_cache: PrimeKGRadiologyCache,
    output_dir: Path,
    prefix: str = "vision_t5",
    policy: str = "evidence_replace"
) -> pd.DataFrame:
    """
    Runs the full batch adaptive verification pipeline and saves results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    verifier = AdaptiveClaimVerifier(kg_cache=kg_cache)
    
    results = []
    all_traces = []
    all_claims = []
    
    import time
    start_time = time.time()
    
    for item in tqdm(raw_predictions, desc="Verifying Claims"):
        sid = item["study_id"]
        draft = item["prediction"]
        ref = item.get("reference", "")
        
        # Get retrieved training reports for this query
        candidates = retrieval_cache.get(sid, [])
        
        verify_res = verifier.verify_and_revise(
            study_id=sid,
            draft_report=draft,
            retrieved_candidates=candidates,
            policy=policy
        )
        
        results.append({
            "study_id": sid,
            "prediction": verify_res["prediction"],
            "reference": ref,
            "original_draft": draft
        })
        
        # Add traces
        for tr in verify_res["traces"]:
            all_traces.append(tr)
            all_claims.append({
                "study_id": sid,
                "claim_index": tr["claim_index"],
                "claim_text": tr["original_text"],
                "revised_text": tr["revised_text"],
                "entities": ",".join(tr["entities"]),
                "support_score": tr["support_score"],
                "decision": tr["decision"]
            })
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_latency_ms = (total_time / max(1, len(raw_predictions))) * 1000.0
    print(f"\n[Verifier Time Audit] Policy: {policy}")
    print(f"Total verification time for {len(raw_predictions)} reports: {total_time:.3f} seconds")
    print(f"Average latency per report: {avg_latency_ms:.2f} ms")
            
    # Save CSV predictions
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(output_dir / f"{prefix}_adaptive_claim_revision.csv", index=False)
    
    # Save claims detail CSV
    claims_df = pd.DataFrame(all_claims)
    claims_df.to_csv(output_dir / f"{prefix}_adaptive_claim_revision_claims.csv", index=False)
    
    # Save traces JSONL
    with open(output_dir / f"{prefix}_adaptive_claim_revision_traces.jsonl", "w", encoding="utf-8") as f:
        for tr in all_traces:
            f.write(json.dumps(tr) + "\n")
            
    return pred_df
