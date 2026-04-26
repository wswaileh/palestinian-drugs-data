"""Stratified ingredient-level train/test split keyed on ATC level-3 class."""

import logging
import random
from collections import defaultdict
from pathlib import Path

from src import config
from src.data.atc import atc_level3

logger = logging.getLogger(__name__)

_UNMAPPED = "__UNMAPPED__"


def _stratum(atc_codes):
    """Pick a single ATC level-3 stratum for an ingredient (first code wins)."""
    if not atc_codes:
        return _UNMAPPED
    return atc_level3(atc_codes[0])


def stratified_split(ingredients, atc_map, test_fraction=None, seed=None):
    """Return (train, test) ingredient lists, stratified by ATC level-3.

    Within each stratum, round(N * test_fraction) ingredients go to test.
    Strata with only one ingredient stay in train.
    """
    test_fraction = config.TEST_FRACTION if test_fraction is None else test_fraction
    seed = config.SEED if seed is None else seed

    rng = random.Random(seed)

    by_stratum = defaultdict(list)
    for ing in ingredients:
        by_stratum[_stratum(atc_map.get(ing, []))].append(ing)

    train, test = [], []
    for stratum, members in sorted(by_stratum.items()):
        members = sorted(members)
        rng.shuffle(members)
        n_test = round(len(members) * test_fraction)
        if len(members) == 1:
            n_test = 0
        test.extend(members[:n_test])
        train.extend(members[n_test:])

    train.sort()
    test.sort()
    logger.info("split: %d train / %d test (test_fraction=%.2f)",
                len(train), len(test), test_fraction)
    return train, test


def save_split(train, test, train_path=None, test_path=None):
    train_path = Path(train_path or config.TRAIN_INGREDIENTS_PATH)
    test_path  = Path(test_path  or config.TEST_INGREDIENTS_PATH)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(train) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(test) + "\n", encoding="utf-8")


def load_split(train_path=None, test_path=None):
    train_path = Path(train_path or config.TRAIN_INGREDIENTS_PATH)
    test_path  = Path(test_path  or config.TEST_INGREDIENTS_PATH)
    train = [line.strip() for line in train_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    test = [line.strip() for line in test_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return train, test


def main():
    """CLI: build the production split from ingredients.parquet + atc cache."""
    import json
    import pandas as pd
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ing = pd.read_parquet(config.INGREDIENTS_PATH)
    found = sorted(ing[ing["found"]]["ingredient"].tolist())
    atc_map = json.loads(config.ATC_CACHE_PATH.read_text(encoding="utf-8"))

    train, test = stratified_split(found, atc_map)
    save_split(train, test)
    logger.info("Wrote %s and %s", config.TRAIN_INGREDIENTS_PATH,
                config.TEST_INGREDIENTS_PATH)


if __name__ == "__main__":
    main()
