"""ATC code lookup via NLM RxNav with on-disk caching."""

import json
import logging
import time
from pathlib import Path

import requests

from src import config

logger = logging.getLogger(__name__)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"


def _get(url, params=None):
    """HTTP GET with retries on transient failure. Returns parsed JSON or {}."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
            time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return {}


def _name_to_rxcui(name):
    payload = _get(f"{RXNAV_BASE}/rxcui.json", params={"name": name})
    ids = (payload.get("idGroup") or {}).get("rxnormId") or []
    return ids[0] if ids else None


def _rxcui_to_atc(rxcui):
    payload = _get(f"{RXNAV_BASE}/rxclass/class/byRxcui.json",
                   params={"rxcui": rxcui, "relaSource": "ATC"})
    items = (((payload.get("rxclassDrugInfoList") or {})
              .get("rxclassDrugInfo")) or [])
    return [it["rxclassMinConceptItem"]["classId"] for it in items
            if it.get("rxclassMinConceptItem", {}).get("classId")]


def lookup_atc(ingredient):
    """Resolve a single ingredient name to a list of ATC codes; [] if unknown."""
    rxcui = _name_to_rxcui(ingredient)
    if not rxcui:
        return []
    return _rxcui_to_atc(rxcui)


def build_mapping(ingredients, cache_path=None):
    """Resolve a list of ingredients to ATC codes, persisting a JSON cache.

    Cached entries are reused; only missing names hit the network.
    """
    cache_path = Path(cache_path or config.ATC_CACHE_PATH)
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    todo = [i for i in ingredients if i not in cache]
    logger.info("ATC: %d cached, %d to fetch", len(cache), len(todo))
    for i, name in enumerate(todo, 1):
        cache[name] = lookup_atc(name)
        if i % 25 == 0:
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            logger.info("ATC progress: %d/%d", i, len(todo))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return {k: cache[k] for k in ingredients}


def atc_level3(code):
    """First five chars of an ATC code (level 3 = therapeutic subgroup)."""
    return code[:5] if code else ""


def main():
    """CLI: build the ATC cache for all ingredients with monograph data."""
    import pandas as pd
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ing = pd.read_parquet(config.INGREDIENTS_PATH)
    found = ing[ing["found"]]["ingredient"].tolist()
    mapping = build_mapping(found)
    n = sum(1 for v in mapping.values() if v)
    logger.info("ATC coverage: %d/%d (%.1f%%)",
                n, len(mapping), 100 * n / max(len(mapping), 1))


if __name__ == "__main__":
    main()
