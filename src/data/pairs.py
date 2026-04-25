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


def build_type_a_triples(by_ingredient, train_ingredients, atc_map, seed=None):
    """Cross-section triples. One triple per (ingredient, non-indication section)."""
    seed = config.SEED if seed is None else seed
    rng = random.Random(seed)
    train_set = set(train_ingredients)
    neighbors = _atc_neighbors(atc_map, train_set)

    triples = []
    for ing in train_ingredients:
        chunks = by_ingredient.get(ing, [])
        ind_chunks = [c for c in chunks if c["section"] == "indications"]
        other_chunks = [c for c in chunks if c["section"] != "indications"]
        if not ind_chunks or not other_chunks:
            continue
        for pos in other_chunks:
            anchor = rng.choice(ind_chunks)
            neg = _hard_negative(rng, ing, neighbors, train_set,
                                 by_ingredient, pos["section"])
            if neg is None:
                continue
            triples.append({"anchor": anchor, "positive": pos, "negative": neg,
                            "kind": "A"})
    rng.shuffle(triples)
    logger.info("Type A: built %d triples for %d ingredients",
                len(triples), len(train_ingredients))
    return triples


def build_type_b_triples(by_ingredient, train_ingredients, queries_by_ingredient,
                         atc_map, seed=None):
    """Query → indication triples. One triple per (query, ingredient)."""
    seed = config.SEED if seed is None else seed
    rng = random.Random(seed)
    train_set = set(train_ingredients)
    neighbors = _atc_neighbors(atc_map, train_set)

    triples = []
    for ing in train_ingredients:
        ind_chunks = [c for c in by_ingredient.get(ing, [])
                      if c["section"] == "indications"]
        if not ind_chunks:
            continue
        positive = ind_chunks[0]
        for q in queries_by_ingredient.get(ing, []):
            neg = _hard_negative(rng, ing, neighbors, train_set,
                                 by_ingredient, "indications")
            if neg is None:
                continue
            triples.append({"anchor": q, "positive": positive,
                            "negative": neg, "kind": "B"})
    rng.shuffle(triples)
    logger.info("Type B: built %d triples", len(triples))
    return triples


def build_combined_dataset(by_ingredient, train_ingredients, queries_by_ingredient,
                            atc_map, seed=None):
    """Concatenate Type A and Type B triples and shuffle for 50/50 mixing."""
    seed = config.SEED if seed is None else seed
    a = build_type_a_triples(by_ingredient, train_ingredients, atc_map, seed=seed)
    b = build_type_b_triples(by_ingredient, train_ingredients,
                              queries_by_ingredient, atc_map, seed=seed + 1)
    combined = a + b
    random.Random(seed + 2).shuffle(combined)
    logger.info("Combined dataset: %d triples (A=%d, B=%d)",
                len(combined), len(a), len(b))
    return combined
