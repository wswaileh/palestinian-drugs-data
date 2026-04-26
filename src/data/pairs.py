"""Build contrastive (anchor, positive, negative) triples from the corpus.

Two complementary triple types:

* Type A — cross-section self-supervision. Anchor = indications chunk; positive
  = a chunk from another section of the same ingredient; hard negative =
  matching-section chunk from a same-ATC-class ingredient.

* Type B — LLM-generated query -> ingredient. Anchor = synthetic clinical query;
  positive = the ingredient's indications chunk; hard negative = indications
  chunk from a same-ATC-class ingredient.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from src import config
from src.data.atc import atc_level3

logger = logging.getLogger(__name__)


def load_chunks_by_ingredient(path=None):
    """Group `chunks.jsonl` records by ingredient. Returns {ing: [chunk, ...]}."""
    path = Path(path or config.CHUNKS_PATH)
    out = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["ingredient"]].append(rec)
    return dict(out)


def _atc_neighbors(atc_map, allowed):
    """Return {ingredient: [same-class ingredients in allowed set]}."""
    by_class = defaultdict(list)
    for ing in allowed:
        for code in atc_map.get(ing, []):
            by_class[atc_level3(code)].append(ing)
    out = {}
    for ing in allowed:
        peers = set()
        for code in atc_map.get(ing, []):
            peers.update(by_class[atc_level3(code)])
        peers.discard(ing)
        out[ing] = sorted(peers)
    return out


def _hard_negative(rng, anchor_ing, neighbors, allowed_pool, by_ing, section):
    """Pick a hard-negative chunk: prefer same-ATC peer's matching section.
    Falls back to a random other ingredient when no peer offers that section.
    """
    candidates = []
    for peer in neighbors.get(anchor_ing, []):
        for c in by_ing.get(peer, []):
            if c["section"] == section:
                candidates.append(c)
    if not candidates:
        pool = [c for ing in allowed_pool if ing != anchor_ing
                for c in by_ing.get(ing, []) if c["section"] == section]
        candidates = pool
    return rng.choice(candidates) if candidates else None


def _build_section_bm25_index(by_ing, allowed_pool, section):
    """Build a BM25 index over chunks of `section` belonging to `allowed_pool`.

    Returns (chunks_in_index, BM25Okapi). Caller pairs them by position.
    """
    from rank_bm25 import BM25Okapi
    chunks = [c for ing in allowed_pool
              for c in by_ing.get(ing, [])
              if c["section"] == section]
    if not chunks:
        return [], None
    tokenized = [c["text"].lower().split() for c in chunks]
    return chunks, BM25Okapi(tokenized)


def _bm25_hard_negative(anchor_text, anchor_ing, allowed_pool, by_ing,
                         section, bm25_cache):
    """Pick a hard-negative chunk via BM25: highest-scoring non-self chunk
    of the requested section. Cached per section across calls.

    `bm25_cache` is a dict the caller provides; this helper populates it
    on first use of each section so we only build the index once per
    training run rather than once per anchor.
    """
    if section not in bm25_cache:
        bm25_cache[section] = _build_section_bm25_index(by_ing, allowed_pool, section)
    chunks, bm25 = bm25_cache[section]
    if not chunks:
        return None
    scores = bm25.get_scores(anchor_text.lower().split())
    # Walk down the ranked list until we hit a chunk from a different ingredient.
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    for i in order:
        if chunks[i]["ingredient"] != anchor_ing:
            return chunks[i]
    return None


def _negatives_for(strategy, *, rng, anchor_ing, anchor_text, neighbors,
                    train_set, by_ingredient, section, bm25_cache):
    """Yield (negative_chunk, ...) tuples per the strategy.

    "atc"     → one ATC-class hard negative
    "bm25"    → one BM25 lexical hard negative
    "atc+bm25" → both negatives (yields TWO triples per anchor)
    """
    out = []
    if strategy in ("atc", "atc+bm25"):
        n = _hard_negative(rng, anchor_ing, neighbors, train_set,
                           by_ingredient, section)
        if n is not None:
            out.append(n)
    if strategy in ("bm25", "atc+bm25"):
        n = _bm25_hard_negative(anchor_text, anchor_ing, train_set,
                                 by_ingredient, section, bm25_cache)
        if n is not None:
            out.append(n)
    return out


def build_type_a_triples(by_ingredient, train_ingredients, atc_map, seed=None,
                          hard_neg_strategy="atc"):
    """Cross-section triples. One triple per (ingredient, non-indication section,
    negative). With hard_neg_strategy='atc+bm25' each anchor produces two
    triples (one per negative type)."""
    seed = config.SEED if seed is None else seed
    rng = random.Random(seed)
    train_set = set(train_ingredients)
    neighbors = _atc_neighbors(atc_map, train_set)
    bm25_cache = {}

    triples = []
    for ing in train_ingredients:
        chunks = by_ingredient.get(ing, [])
        ind_chunks = [c for c in chunks if c["section"] == "indications"]
        other_chunks = [c for c in chunks if c["section"] != "indications"]
        if not ind_chunks or not other_chunks:
            continue
        for pos in other_chunks:
            anchor = rng.choice(ind_chunks)
            negs = _negatives_for(hard_neg_strategy, rng=rng, anchor_ing=ing,
                                   anchor_text=anchor["text"],
                                   neighbors=neighbors, train_set=train_set,
                                   by_ingredient=by_ingredient,
                                   section=pos["section"],
                                   bm25_cache=bm25_cache)
            for neg in negs:
                triples.append({"anchor": anchor, "positive": pos,
                                "negative": neg, "kind": "A"})
    rng.shuffle(triples)
    logger.info("Type A (%s): built %d triples for %d ingredients",
                hard_neg_strategy, len(triples), len(train_ingredients))
    return triples


def build_type_b_triples(by_ingredient, train_ingredients, queries_by_ingredient,
                         atc_map, seed=None, hard_neg_strategy="atc"):
    """Query → indication triples. One triple per (query, ingredient, negative)."""
    seed = config.SEED if seed is None else seed
    rng = random.Random(seed)
    train_set = set(train_ingredients)
    neighbors = _atc_neighbors(atc_map, train_set)
    bm25_cache = {}

    triples = []
    for ing in train_ingredients:
        ind_chunks = [c for c in by_ingredient.get(ing, [])
                      if c["section"] == "indications"]
        if not ind_chunks:
            continue
        positive = ind_chunks[0]
        for q in queries_by_ingredient.get(ing, []):
            negs = _negatives_for(hard_neg_strategy, rng=rng, anchor_ing=ing,
                                   anchor_text=q, neighbors=neighbors,
                                   train_set=train_set,
                                   by_ingredient=by_ingredient,
                                   section="indications",
                                   bm25_cache=bm25_cache)
            for neg in negs:
                triples.append({"anchor": q, "positive": positive,
                                "negative": neg, "kind": "B"})
    rng.shuffle(triples)
    logger.info("Type B: built %d triples", len(triples))
    return triples


def build_combined_dataset(by_ingredient, train_ingredients, queries_by_ingredient,
                            atc_map, seed=None, hard_neg_strategy="atc"):
    """Concatenate Type A and Type B triples and shuffle for 50/50 mixing."""
    seed = config.SEED if seed is None else seed
    a = build_type_a_triples(by_ingredient, train_ingredients, atc_map,
                              seed=seed, hard_neg_strategy=hard_neg_strategy)
    b = build_type_b_triples(by_ingredient, train_ingredients,
                              queries_by_ingredient, atc_map,
                              seed=seed + 1,
                              hard_neg_strategy=hard_neg_strategy)
    combined = a + b
    random.Random(seed + 2).shuffle(combined)
    logger.info("Combined dataset (%s): %d triples (A=%d, B=%d)",
                hard_neg_strategy, len(combined), len(a), len(b))
    return combined
