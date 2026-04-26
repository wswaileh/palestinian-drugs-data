"""Retrievers used as baselines and as the eval interface for the FT model."""

import logging

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text):
    return [t for t in text.lower().split() if t]


class BM25Retriever:
    """Lexical baseline using rank-bm25."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

    def retrieve(self, query, k=10):
        scores = self._bm25.get_scores(_tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in idx]


class DenseRetriever:
    """Encoder + dense index. Wraps any HuggingFace model name or saved path."""

    def __init__(self, chunks, model_name, device=None):
        from sentence_transformers import SentenceTransformer, models
        self.chunks = list(chunks)
        self.model_name = model_name

        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception:
            # Fall back to mean-pooled raw HF model (off-the-shelf SciBERT case)
            word = models.Transformer(model_name, max_seq_length=256)
            pool = models.Pooling(word.get_word_embedding_dimension(),
                                   pooling_mode="mean")
            self.model = SentenceTransformer(modules=[word, pool], device=device)

        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )

    def retrieve(self, query, k=10):
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True)
        sims = (self.embeddings @ q[0])
        idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in idx]


class RRFRetriever:
    """Reciprocal Rank Fusion of two or more retrievers.

    For each retriever's ranked list, each chunk gets a score 1/(k0 + rank).
    Scores are summed across retrievers. The RRF constant k0=60 is the
    Cormack-Clarke-Buettcher 2009 default; results are insensitive to it.

    Operates at chunk-id granularity. Chunks not retrieved by a given
    backend contribute 0 from that backend.
    """

    K0 = 60

    def __init__(self, retrievers, fusion_k=100):
        """`retrievers`: list of objects with .retrieve(query, k). `fusion_k`
        is how deep we look in each backend before fusing."""
        self.retrievers = list(retrievers)
        self.fusion_k = fusion_k

    def retrieve(self, query, k=10):
        scores = {}
        chunk_by_id = {}
        for retriever in self.retrievers:
            hits = retriever.retrieve(query, k=self.fusion_k)
            for rank, hit in enumerate(hits, start=1):
                cid = hit["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.K0 + rank)
                chunk_by_id.setdefault(cid, hit)
        ordered_ids = sorted(scores, key=lambda c: -scores[c])[:k]
        return [chunk_by_id[c] for c in ordered_ids]
