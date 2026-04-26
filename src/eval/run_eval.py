"""Run the full ablation table on Set 1 (auto held-out) and Set 2 (hand)."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.eval.baselines import BM25Retriever, DenseRetriever, RRFRetriever
from src.eval.reranker import CrossEncoderReranker

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BGE = "BAAI/bge-reranker-base"
CE_BGE_LARGE = "BAAI/bge-reranker-large"
from src.eval.metrics import (mrr_at_k, ndcg_at_k, paired_bootstrap,
                               recall_at_k)

logger = logging.getLogger(__name__)


def load_chunks():
    chunks = []
    with open(config.CHUNKS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_set1(test_ingredients, generated_queries_path):
    """Auto-mined held-out: queries from generated_queries.jsonl whose ingredient
    is in `test_ingredients`."""
    test_set = set(test_ingredients)
    queries = []
    with open(generated_queries_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["ingredient"] not in test_set:
                continue
            for q in rec["queries"]:
                queries.append({"query": q, "gold": [rec["ingredient"]]})
    return queries


def load_set2(path):
    """Hand-curated queries from JSONL. Each record: {query, gold: [ing,...]}."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                out.append(json.loads(line))
    return out


def _retrieved_ingredients_in_order(hits, max_k):
    """Collapse chunk-level hits to unique ingredient list, preserving rank."""
    seen = []
    for h in hits:
        ing = h["ingredient"]
        if ing not in seen:
            seen.append(ing)
            if len(seen) >= max_k:
                break
    return seen


def evaluate_retriever(retriever, queries, k_max=20):
    """Return dict mapping metric name → np.array of per-query scores."""
    rec_ks = {k: [] for k in config.RECALL_KS}
    mrr = []
    ndcg = []
    for q in queries:
        hits = retriever.retrieve(q["query"], k=k_max)
        ranked_ings = _retrieved_ingredients_in_order(hits, k_max)
        for k in config.RECALL_KS:
            rec_ks[k].append(recall_at_k(ranked_ings, q["gold"], k))
        mrr.append(mrr_at_k(ranked_ings, q["gold"], config.MRR_K))
        ndcg.append(ndcg_at_k(ranked_ings, q["gold"], config.NDCG_K))
    out = {f"recall@{k}": np.array(v) for k, v in rec_ks.items()}
    out[f"mrr@{config.MRR_K}"]   = np.array(mrr)
    out[f"ndcg@{config.NDCG_K}"] = np.array(ndcg)
    return out


_BUILT_RETRIEVERS = {}


def build_retriever(name, chunks):
    """Construct the retriever named in the ablation table.

    Cached by name to avoid re-encoding the corpus when an RRF variant
    references a backend that was already built standalone.
    """
    if name in _BUILT_RETRIEVERS:
        return _BUILT_RETRIEVERS[name]

    if name == "bm25":
        r = BM25Retriever(chunks)
    elif name == "scibert_offshelf":
        r = DenseRetriever(chunks, model_name=config.ENCODER_MODEL)
    elif name == "biobert_mnli":
        r = DenseRetriever(chunks, model_name="pritamdeka/BioBERT-mnli-snli-stsb")
    elif name == "minilm":
        r = DenseRetriever(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Phase 1.3: off-the-shelf alternative encoders (no fine-tuning).
    elif name == "bge_base_offshelf":
        r = DenseRetriever(chunks, model_name="BAAI/bge-base-en-v1.5")
    elif name == "pubmedbert_offshelf":
        r = DenseRetriever(chunks,
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    elif name == "sapbert_offshelf":
        r = DenseRetriever(chunks, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    elif name == "spubmedbert_msmarco":
        r = DenseRetriever(chunks, model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    elif name.startswith("scibert_ft_"):
        variant = name[len("scibert_ft_"):]
        r = DenseRetriever(chunks,
                            model_name=str(config.CHECKPOINT_DIR / "scibert_ft" / variant))
    elif name.startswith("rrf_"):
        # rrf_<a>__<b> fuses backends a and b (double underscore separator)
        backend_names = name[len("rrf_"):].split("__")
        backends = [build_retriever(b, chunks) for b in backend_names]
        r = RRFRetriever(backends)
    elif name.startswith("rerank_"):
        # rerank_<first_stage> wraps a first-stage retriever with a
        # cross-encoder. If a fine-tuned CE is on disk, prefer it; otherwise
        # fall back to the off-the-shelf MS-MARCO model.
        first_stage_name = name[len("rerank_"):]
        first_stage = build_retriever(first_stage_name, chunks)
        ft_ce_dir = config.CHECKPOINT_DIR / "cross_encoder" / "ab_atc"
        ce_model_to_use = str(ft_ce_dir) if ft_ce_dir.exists() else CE_MODEL
        r = CrossEncoderReranker(first_stage, ce_model_to_use, top_k_first=50)
    elif name.startswith("ftrerank_"):
        # ftrerank_* explicitly uses the fine-tuned CE (fails loud if missing).
        first_stage_name = name[len("ftrerank_"):]
        first_stage = build_retriever(first_stage_name, chunks)
        ft_ce_dir = config.CHECKPOINT_DIR / "cross_encoder" / "ab_atc"
        if not ft_ce_dir.exists():
            raise FileNotFoundError(
                f"fine-tuned CE not found at {ft_ce_dir}; train first")
        r = CrossEncoderReranker(first_stage, str(ft_ce_dir), top_k_first=50)
    elif name.startswith("bgererank_"):
        # bgererank_* uses BGE-reranker-base (stronger off-the-shelf CE).
        first_stage_name = name[len("bgererank_"):]
        first_stage = build_retriever(first_stage_name, chunks)
        r = CrossEncoderReranker(first_stage, CE_BGE, top_k_first=50)
    elif name.startswith("aggrerank_"):
        # aggrerank_* uses BGE-reranker with ingredient-level max-pooling.
        first_stage_name = name[len("aggrerank_"):]
        first_stage = build_retriever(first_stage_name, chunks)
        r = CrossEncoderReranker(first_stage, CE_BGE, top_k_first=50,
                                  aggregate_by_ingredient=True)
    else:
        raise ValueError(f"unknown retriever {name}")

    _BUILT_RETRIEVERS[name] = r
    return r


VARIANTS = ["bm25", "scibert_offshelf", "biobert_mnli", "minilm",
            # Phase 1.3: off-the-shelf alternative encoders
            "bge_base_offshelf", "pubmedbert_offshelf",
            "sapbert_offshelf", "spubmedbert_msmarco",
            "scibert_ft_a_only", "scibert_ft_b_only",
            "scibert_ft_ab_random", "scibert_ft_ab_atc",
            # BM25-mined hard negatives (Phase 1.2 — train on Colab when available)
            "scibert_ft_ab_bm25",
            "scibert_ft_ab_atc_bm25",
            "scibert_ft_b_only_bm25",
            "scibert_ft_b_only_atc_bm25",
            "rrf_bm25__scibert_ft_b_only",
            "rrf_bm25__scibert_ft_ab_atc_bm25",
            "rrf_bm25__minilm"]


def run_all(test_ingredients_path=None, queries_path=None, hand_path=None,
            output_dir=None):
    test_ingredients_path = test_ingredients_path or config.TEST_INGREDIENTS_PATH
    queries_path = queries_path or config.GENERATED_QUERIES_PATH
    hand_path = hand_path or config.HAND_CURATED_QUERIES_PATH
    output_dir = Path(output_dir or "eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    test_ings = [l.strip() for l in Path(test_ingredients_path).read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    set1 = load_set1(test_ings, queries_path)
    set2 = load_set2(hand_path) if Path(hand_path).exists() else []
    logger.info("Set 1: %d queries; Set 2: %d queries", len(set1), len(set2))

    results = defaultdict(dict)
    for name in VARIANTS:
        try:
            r = build_retriever(name, chunks)
        except Exception as e:
            logger.warning("skipping %s: %s", name, e)
            continue
        results[name]["set1"] = evaluate_retriever(r, set1)
        if set2:
            results[name]["set2"] = evaluate_retriever(r, set2)

    target = "scibert_offshelf"
    sig_rows = []
    for name in VARIANTS:
        if name == target or name not in results:
            continue
        for ds in ("set1", "set2"):
            if ds not in results.get(target, {}):
                continue
            if ds not in results.get(name, {}):
                continue
            for metric in (f"recall@{config.RECALL_KS[1]}", f"mrr@{config.MRR_K}"):
                a = results[target][ds][metric]
                b = results[name][ds][metric]
                bs = paired_bootstrap(a, b, n_resamples=config.BOOTSTRAP_RESAMPLES,
                                      seed=config.SEED)
                sig_rows.append({"variant": name, "set": ds, "metric": metric,
                                 **bs})

    table_rows = []
    for name in VARIANTS:
        if name not in results:
            continue
        row = {"variant": name}
        for ds in ("set1", "set2"):
            if ds not in results[name]:
                continue
            for metric, arr in results[name][ds].items():
                row[f"{ds}/{metric}"] = float(arr.mean())
        table_rows.append(row)
    df = pd.DataFrame(table_rows)
    df.to_csv(output_dir / "ablation_table.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(output_dir / "significance.csv", index=False)
    logger.info("Wrote %s/ablation_table.csv and significance.csv", output_dir)
    return df, pd.DataFrame(sig_rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df, _ = run_all()
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
