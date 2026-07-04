import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Any, Tuple, Set, Optional

# Standard default vocabulary for mock mode
MOCK_PRIMEKG_NODES = [
    # Findings
    {"node_id": "F_01", "node_name": "cardiomegaly", "node_type": "finding"},
    {"node_id": "F_02", "node_name": "pleural effusion", "node_type": "finding"},
    {"node_id": "F_03", "node_name": "pneumothorax", "node_type": "finding"},
    {"node_id": "F_04", "node_name": "atelectasis", "node_type": "finding"},
    {"node_id": "F_05", "node_name": "consolidation", "node_type": "finding"},
    {"node_id": "F_06", "node_name": "opacity", "node_type": "finding"},
    {"node_id": "F_07", "node_name": "hilar congestion", "node_type": "finding"},
    {"node_id": "F_08", "node_name": "normal", "node_type": "status"},
    # Anatomies
    {"node_id": "A_01", "node_name": "heart", "node_type": "anatomy"},
    {"node_id": "A_02", "node_name": "lungs", "node_type": "anatomy"},
    {"node_id": "A_03", "node_name": "mediastinum", "node_type": "anatomy"},
    {"node_id": "A_04", "node_name": "pleural space", "node_type": "anatomy"},
    {"node_id": "A_05", "node_name": "hilar region", "node_type": "anatomy"}
]

MOCK_PRIMEKG_EDGES = [
    {"x_id": "F_01", "y_id": "A_01", "relation": "occurs_in", "display_relation": "finding occurs in anatomy"},
    {"x_id": "F_02", "y_id": "A_04", "relation": "occurs_in", "display_relation": "finding occurs in anatomy"},
    {"x_id": "F_03", "y_id": "A_04", "relation": "occurs_in", "display_relation": "finding occurs in anatomy"},
    {"x_id": "F_04", "y_id": "A_02", "relation": "occurs_in", "display_relation": "finding occurs in anatomy"},
    {"x_id": "F_05", "y_id": "A_02", "relation": "occurs_in", "display_relation": "finding occurs in anatomy"},
    {"x_id": "F_06", "y_id": "A_02", "relation": "occurs_in", "display_relation": "finding occurs in anatomy"},
    {"x_id": "F_07", "y_id": "A_05", "relation": "occurs_in", "display_relation": "finding occurs in anatomy"},
    {"x_id": "A_01", "y_id": "A_03", "relation": "borders", "display_relation": "anatomy borders anatomy"},
    {"x_id": "A_05", "y_id": "A_02", "relation": "part_of", "display_relation": "anatomy is part of anatomy"}
]

# Negation terms
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bclear of\b",
    r"\bfree of\b",
    r"\brules out\b",
    r"\brule out\b",
    r"\bruled out\b",
    r"\bnegative for\b",
    r"\bdenies\b",
    r"\babsent\b",
    r"\bnormal\b"  # normal heart = negated finding of cardiomegaly (conceptually)
]

class PrimeKGRadiologyCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.nodes_df = None
        self.edges_df = None
        self.graph = {}  # Adjacency list for fast querying: node_id -> set(neighbor_id)
        self.node_lookup = {}  # node_name lowercase -> node details
        
        if cache_dir and cache_dir.exists():
            nodes_path = cache_dir / "nodes.csv"
            edges_path = cache_dir / "kg.csv"
            if nodes_path.exists() and edges_path.exists():
                try:
                    self.nodes_df = pd.read_csv(nodes_path)
                    self.edges_df = pd.read_csv(edges_path)
                except Exception:
                    pass
                    
        # Load mock if data is missing
        if self.nodes_df is None or self.edges_df is None:
            self.nodes_df = pd.DataFrame(MOCK_PRIMEKG_NODES)
            self.edges_df = pd.DataFrame(MOCK_PRIMEKG_EDGES)
            
        self._build_structures()
        
    def _build_structures(self):
        # Build node lookup by lowercase name
        for _, row in self.nodes_df.iterrows():
            nid = str(row["node_id"])
            name = str(row["node_name"]).lower()
            ntype = str(row["node_type"]) if "node_type" in row else "unknown"
            node_info = {"node_id": nid, "node_name": name, "node_type": ntype}
            self.node_lookup[name] = node_info
            
        # Build adjacency graph
        self.graph = {}
        for _, row in self.edges_df.iterrows():
            x = str(row["x_id"])
            y = str(row["y_id"])
            if x not in self.graph:
                self.graph[x] = set()
            if y not in self.graph:
                self.graph[y] = set()
            self.graph[x].add(y)
            self.graph[y].add(x)
            
    def get_path_score(self, node_id_1: str, node_id_2: str) -> float:
        """Returns 1.0 if direct edge, 0.5 if 2-hop, 0.0 otherwise."""
        if node_id_1 == node_id_2:
            return 1.0
        neighbors_1 = self.graph.get(node_id_1, set())
        if node_id_2 in neighbors_1:
            return 1.0
        neighbors_2 = self.graph.get(node_id_2, set())
        if neighbors_1.intersection(neighbors_2):
            return 0.5
        return 0.0

    def link_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Performs lexical entity matching against the cache vocabulary
        and identifies negation.
        """
        text_lower = text.lower()
        matched = []
        
        # Sort node vocabulary names by length descending to match longest terms first
        sorted_names = sorted(self.node_lookup.keys(), key=len, reverse=True)
        
        # Simple sliding window/regex match
        # To avoid double-matching sub-segments of an already matched term, track matched char indices
        matched_intervals: List[Tuple[int, int]] = []
        
        for name in sorted_names:
            # We match as word boundaries if possible
            pattern = r"\b" + re.escape(name) + r"\b"
            for m in re.finditer(pattern, text_lower):
                start, end = m.span()
                # Check if this overlaps with an already matched interval
                overlap = False
                for s, e in matched_intervals:
                    if not (end <= s or start >= e):
                        overlap = True
                        break
                if not overlap:
                    matched_intervals.append((start, end))
                    node_info = self.node_lookup[name]
                    
                    # Negation Check within window of 4 words before the matched entity
                    # or 2 words after (e.g. "pneumothorax is ruled out")
                    left_context = text_lower[max(0, start-40):start]
                    right_context = text_lower[end:min(len(text_lower), end+30)]
                    
                    negated = False
                    # Look for negation patterns in left and right context
                    for neg_pat in NEGATION_PATTERNS:
                        if re.search(neg_pat, left_context) or re.search(neg_pat, right_context):
                            negated = True
                            break
                            
                    matched.append({
                        "node_id": node_info["node_id"],
                        "node_name": node_info["node_name"],
                        "node_type": node_info["node_type"],
                        "negated": negated,
                        "span": (start, end)
                    })
                    
        return matched

def build_radiology_cache_from_raw(
    primekg_nodes_path: Path,
    primekg_edges_path: Path,
    train_manifest_path: Path,
    output_dir: Path,
    hops: int = 1
):
    """
    Reads the full PrimeKG, identifies seed nodes based on training report concepts,
    performs hop expansion, and writes nodes.csv and kg.csv to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse train manifest to collect vocabulary words
    from nesy_gen.manifest import load_manifest
    try:
        train_examples = load_manifest(train_manifest_path)
    except Exception:
        train_examples = []
        
    train_reports = " ".join([ex["report"].lower() for ex in train_examples])
    
    # 2. Read full PrimeKG nodes
    if not primekg_nodes_path.exists() or not primekg_edges_path.exists():
        # Write mock cache if source not available
        mock_nodes = pd.DataFrame(MOCK_PRIMEKG_NODES)
        mock_edges = pd.DataFrame(MOCK_PRIMEKG_EDGES)
        mock_nodes.to_csv(output_dir / "nodes.csv", index=False)
        mock_edges.to_csv(output_dir / "kg.csv", index=False)
        return
        
    nodes_df = pd.read_csv(primekg_nodes_path)
    
    # Find matching seed nodes
    # We look for nodes whose node_name matches terms in the train corpus
    # To be efficient, we can do substring matching on clinical finding/anatomy-related categories
    seed_node_ids = set()
    for _, row in nodes_df.iterrows():
        name = str(row["node_name"]).lower()
        # Check if the name appears as a full word in the training corpus
        pattern = r"\b" + re.escape(name) + r"\b"
        if len(name) > 3 and re.search(pattern, train_reports):
            seed_node_ids.add(row["node_id"])
            
    # 3. Read edges and perform expansion
    edges_df = pd.read_csv(primekg_edges_path)
    
    # Fast 1-hop expansion
    selected_node_ids = set(seed_node_ids)
    selected_edges = []
    
    # Map edges
    for _, row in edges_df.iterrows():
        x = row["x_id"]
        y = row["y_id"]
        if x in seed_node_ids or y in seed_node_ids:
            selected_node_ids.add(x)
            selected_node_ids.add(y)
            selected_edges.append(row)
            
    # Subsample nodes
    selected_nodes_df = nodes_df[nodes_df["node_id"].isin(selected_node_ids)]
    selected_edges_df = pd.DataFrame(selected_edges)
    
    selected_nodes_df.to_csv(output_dir / "nodes.csv", index=False)
    selected_edges_df.to_csv(output_dir / "kg.csv", index=False)
