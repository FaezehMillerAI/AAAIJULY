from typing import List, Dict, Any
from nesy_gen.kg.primekg import PrimeKGRadiologyCache

def evaluate_ltn_constraints(linked_entities: List[Dict[str, Any]], kg_cache: PrimeKGRadiologyCache) -> Dict[str, float]:
    """
    Computes soft logic scores inspired by Logic Tensor Networks (LTN).
    
    1. Connectivity: For every positive finding, does there exist an anatomy in the report
       that matches it in PrimeKG (within 1-2 hops)?
       Fuzzy formula: ALL f in Findings_pos ( SOME a in Anatomies_pos ( Connected(f, a) ) )
       Using product t-norm for ALL, and max t-conorm for SOME.
       
    2. Coherence: Are there contradictory mentions (e.g., cardiomegaly is both negated and positive)?
       Coherence = 1.0 if no overlap, 0.0 if overlap.
    """
    findings_pos = []
    anatomies_pos = []
    
    pos_ids = set()
    neg_ids = set()
    
    for ent in linked_entities:
        nid = ent["node_id"]
        ntype = ent["node_type"]
        negated = ent["negated"]
        
        if negated:
            neg_ids.add(nid)
        else:
            pos_ids.add(nid)
            if ntype == "finding":
                findings_pos.append(nid)
            elif ntype == "anatomy":
                anatomies_pos.append(nid)
                
    # 1. Coherence Score
    overlap = pos_ids.intersection(neg_ids)
    coherence_score = 1.0 - (len(overlap) / max(1, len(pos_ids.union(neg_ids))))
    
    # 2. Connectivity Score
    # If there are no positive findings, the connectivity is vacuously satisfied (1.0)
    # If there are positive findings but no anatomies, connectivity is low but we give a default floor
    if not findings_pos:
        connectivity_score = 1.0
    elif not anatomies_pos:
        connectivity_score = 0.2  # Penalty for having findings but no location context
    else:
        # For each finding, find the maximum connectivity with any anatomy
        finding_scores = []
        for f in findings_pos:
            max_conn = 0.0
            for a in anatomies_pos:
                score = kg_cache.get_path_score(f, a)
                if score > max_conn:
                    max_conn = score
            # We add a small constant (0.1) as a logical smooth relaxation
            finding_scores.append(max_conn)
            
        # Product t-norm to aggregate across all findings
        prod = 1.0
        for s in finding_scores:
            prod *= (0.8 * s + 0.2)  # relaxed product: maps [0, 1] to [0.2, 1.0]
        connectivity_score = prod
        
    overall_score = 0.6 * connectivity_score + 0.4 * coherence_score
    
    return {
        "connectivity_score": float(connectivity_score),
        "coherence_score": float(coherence_score),
        "overall_score": float(overall_score)
    }
