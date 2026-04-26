import os
import pytest


@pytest.mark.skipif(
    not os.environ.get("RUN_CE_TEST"),
    reason="Skips without RUN_CE_TEST=1 (downloads cross-encoder, ~80MB)."
)
def test_reranker_promotes_clear_positive(tiny_chunks):
    """Confirm the cross-encoder ranks a topical chunk above an off-topic one."""
    from src.eval.baselines import BM25Retriever
    from src.eval.reranker import CrossEncoderReranker

    first = BM25Retriever(tiny_chunks)
    reranker = CrossEncoderReranker(
        first_stage=first,
        ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_first=6,
    )
    hits = reranker.retrieve("type 2 diabetes treatment", k=2)
    assert hits[0]["ingredient"] == "metformin"


def test_reranker_handles_empty_first_stage(monkeypatch):
    """If the first-stage returns nothing, the reranker shouldn't crash."""
    from src.eval.reranker import CrossEncoderReranker

    class EmptyRetriever:
        def retrieve(self, query, k=10):
            return []

    # Skip CE construction (network-dependent) by stubbing.
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.first_stage = EmptyRetriever()
    reranker.top_k_first = 10
    reranker.ce = None  # never called for empty candidates

    assert reranker.retrieve("anything", k=5) == []
