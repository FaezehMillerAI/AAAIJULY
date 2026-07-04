from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any

class TFIDFRetrieval:
    def __init__(self, train_examples: List[Dict[str, Any]]):
        self.examples = train_examples
        self.corpus = [ex["report"] for ex in train_examples]
        self.study_ids = [ex["study_id"] for ex in train_examples]
        
        # Fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        if self.corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        else:
            self.tfidf_matrix = None
            
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves the top-k training examples closest to the query string."""
        if self.tfidf_matrix is None or not query:
            # Fallback if corpus is empty or query is empty
            return [
                {
                    "report": ex["report"],
                    "study_id": ex["study_id"],
                    "score": 0.0,
                    "rank": idx + 1
                }
                for idx, ex in enumerate(self.examples[:top_k])
            ]
            
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Sort and get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_k_indices):
            results.append({
                "report": self.corpus[idx],
                "study_id": self.study_ids[idx],
                "score": float(similarities[idx]),
                "rank": rank + 1
            })
            
        return results
