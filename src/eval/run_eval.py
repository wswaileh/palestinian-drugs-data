"""Run the full ablation table on Set 1 (auto held-out) and Set 2 (hand)."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.eval.baselines import BM25Retriever, DenseRetriever
from src.eval.metrics import (mrr_at_k, ndcg_at_k, paired_bootstrap,
                               recall_at_k)

logger = logging.getLogger(__name__)


def load_chunks():
    chunks = []
    with open(config.CHUNKS_PATH) as fh:
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
    with open(generated_queries_path) as fh:
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
    with open(path) as fh:
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


def build_retriever(name, chunks):
    """Construct the retriever named in the ablation table."""
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "scibert_offshelf":
        return DenseRetriever(chunks, model_name=config.ENCODER_MODEL)
    if name == "biobert_mnli":
        return DenseRetriever(chunks, model_name="pritamdeka/BioBERT-mnli-snli-stsb")
    if name == "minilm":
        return DenseRetriever(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")
    if name.startswith("scibert_ft_"):
        variant = name[len("scibert_ft_"):]
        return DenseRetriever(chunks,
                              model_name=str(config.CHECKPOINT_DIR / "scibert_ft" / variant))
    raise ValueError(f"unknown retriever {name}")


VARIANTS = ["bm25", "scibert_offshelf", "biobert_mnli", "minilm",
            "scibert_ft_a_only", "scibert_ft_b_only",
            "scibert_ft_ab_random", "scibert_ft_ab_atc"]


def run_all(test_ingredients_path=None, queries_path=None, hand_path=None,
            output_dir=None):
    test_ingredients_path = test_ingredients_path or config.TEST_INGREDIENTS_PATH
    queries_path = queries_path or config.GENERATED_QUERIES_PATH
    hand_path = hand_path or config.HAND_CURATED_QUERIES_PATH
    output_dir = Path(output_dir or "eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    test_ings = [l.strip() for l in Path(test_ingredients_path).read_text().splitlines()
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
