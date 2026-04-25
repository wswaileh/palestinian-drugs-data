from src.eval import baselines


def test_bm25_ranks_lexical_match_first(tiny_chunks):
    r = baselines.BM25Retriever(tiny_chunks)
    hits = r.retrieve("type 2 diabetes glucose", k=3)
    assert hits[0]["ingredient"] == "metformin"


def test_dense_retriever_returns_k_results(tiny_chunks):
    r = baselines.DenseRetriever(tiny_chunks, model_name="prajjwal1/bert-tiny")
    hits = r.retrieve("painful inflammation", k=2)
    assert len(hits) == 2
    assert {"ingredient", "section", "text"}.issubset(hits[0].keys())


def test_dense_retriever_caches_index(tiny_chunks):
    r = baselines.DenseRetriever(tiny_chunks, model_name="prajjwal1/bert-tiny")
    a = r.retrieve("pain", k=1)
    b = r.retrieve("pain", k=1)
    assert a == b


def test_run_eval_on_tiny(tiny_chunks_file, tmp_path):
    """Pipe-checking the harness wires together — no model accuracy assertion."""
    import json
    from src.eval import run_eval

    chunks = []
    with open(tiny_chunks_file) as fh:
        for line in fh:
            chunks.append(json.loads(line))
    queries = [{"query": "diabetes glucose", "gold": ["metformin"]}]
    r = run_eval.build_retriever("bm25", chunks)
    out = run_eval.evaluate_retriever(r, queries, k_max=3)
    assert out["recall@1"][0] == 1.0
