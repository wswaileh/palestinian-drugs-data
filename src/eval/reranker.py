"""Cross-encoder reranker — second-stage retriever that wraps any first-stage."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Two-stage retrieval: first_stage retrieves top_k_first; cross-encoder
    reranks all of them and returns the top k.

    Cross-encoders see (query, chunk) jointly with full bidirectional
    attention, so they catch fine-grained semantics that single-vector
    bi-encoders miss. Typical lift: +10-15pp Recall@5 in biomedical tasks.

    `aggregate_by_ingredient=True` max-pools chunk scores per ingredient
    before ranking — useful when the gold answers are at ingredient
    granularity (which is our case here).
    """

    def __init__(self, first_stage, ce_model_name, top_k_first=50,
                 device=None, aggregate_by_ingredient=False):
        from sentence_transformers import CrossEncoder
        self.first_stage = first_stage
        self.ce_model_name = ce_model_name
        self.top_k_first = top_k_first
        self.aggregate_by_ingredient = aggregate_by_ingredient
        self.ce = CrossEncoder(ce_model_name, device=device)

    def retrieve(self, query, k=10):
        candidates = self.first_stage.retrieve(query, k=self.top_k_first)
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.ce.predict(pairs, show_progress_bar=False,
                                 convert_to_numpy=True)

        if not self.aggregate_by_ingredient:
            order = np.argsort(-scores)[:k]
            return [candidates[int(i)] for i in order]

        # Max-pool per ingredient: keep the best-scoring chunk for each.
        best = {}  # ingredient -> (chunk, score)
        for c, s in zip(candidates, scores):
            ing = c["ingredient"]
            if ing not in best or float(s) > best[ing][1]:
                best[ing] = (c, float(s))
        ranked = sorted(best.values(), key=lambda x: -x[1])[:k]
        return [c for c, _ in ranked]
