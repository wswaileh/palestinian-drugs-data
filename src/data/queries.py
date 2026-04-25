"""Generate realistic clinical queries with Claude Haiku 4.5, with disk cache."""

import json
import logging
import os
import re
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical pharmacist. For the given drug ingredient, write 3 "
    "realistic short queries a clinician or patient would type into a "
    "drug-recommendation system that this drug should retrieve. "
    "Vary phrasing: one symptom-first, one patient-population-first, one "
    "mechanism- or condition-first. Each query under 25 words. "
    "Return strict JSON: {\"queries\": [\"...\", \"...\", \"...\"]}."
)


def _make_client():
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _parse_json(text):
    """Strip code fences then parse JSON. Returns {} on failure."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}


def generate_for_ingredient(client, ingredient, indications_text):
    """Call Claude Haiku 4.5 once for one ingredient. Returns a list of strings."""
    user = f"Drug ingredient: {ingredient}\nIndications: {indications_text[:1500]}"
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text
    parsed = _parse_json(text)
    out = parsed.get("queries", [])
    return [q for q in out if isinstance(q, str) and q.strip()]


def build_query_set(ingredients, indications_by_ingredient, cache_path=None):
    """Generate queries for each ingredient, caching results to JSONL.

    Each line in the cache file is {"ingredient": str, "queries": [str]}.
    Re-running with the same cache skips already-done ingredients.
    """
    cache_path = Path(cache_path or config.GENERATED_QUERIES_PATH)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["ingredient"]] = rec["queries"]

    todo = [i for i in ingredients if i not in done]
    logger.info("LLM queries: %d cached, %d to generate", len(done), len(todo))

    if not todo:
        return {i: done[i] for i in ingredients if i in done}

    client = _make_client()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a") as fh:
        for i, ing in enumerate(todo, 1):
            ind_text = indications_by_ingredient.get(ing, "")
            if not ind_text.strip():
                qs = []
            else:
                try:
                    qs = generate_for_ingredient(client, ing, ind_text)
                except Exception as e:
                    logger.warning("query gen failed for %s: %s", ing, e)
                    qs = []
            done[ing] = qs
            fh.write(json.dumps({"ingredient": ing, "queries": qs}) + "\n")
            fh.flush()
            if i % 50 == 0:
                logger.info("LLM queries progress: %d/%d", i, len(todo))

    return {i: done.get(i, []) for i in ingredients}


def main():
    """CLI: generate queries for all training ingredients."""
    import pandas as pd
    from dotenv import load_dotenv
    from src.data.splits import load_split

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_dotenv()

    ing = pd.read_parquet(config.INGREDIENTS_PATH)
    by_ing = dict(zip(ing["ingredient"], ing["indications"]))

    train, test = load_split()
    all_targets = train + test  # we need queries for both train and held-out eval
    build_query_set(all_targets, by_ing)


if __name__ == "__main__":
    main()
