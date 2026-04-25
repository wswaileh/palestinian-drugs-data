# SciBERT Contrastive Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, and evaluate a SciBERT encoder fine-tuned with contrastive learning (Type A cross-section + Type B query→indication, with ATC-class hard negatives) for clinical retrieval over the Palestinian-pharmacy monograph corpus, with a four-baseline ablation table and statistical significance testing.

**Architecture:** Three layers — `src/data/` builds the training inputs (ATC mapping, train/test split, LLM-generated queries, contrastive triples); `src/training/` runs `sentence-transformers` MultipleNegativesRankingLoss with explicit hard negatives; `src/eval/` runs all baselines + the fine-tuned model on two query sets (auto-mined held-out, hand-curated) with paired-bootstrap CIs. Notebooks (`02_train_scibert.ipynb`, `03_eval_results.ipynb`) wrap the modules for Colab-runnable training and the final ablation table.

**Tech Stack:** Python 3.11, `sentence-transformers` 3.x, `transformers`, `torch` (CPU/MPS on Mac, CUDA on Colab T4), `rank-bm25`, `faiss-cpu`, `anthropic` (Claude Haiku 4.5), `pandas`, `pyarrow`, `pytest`, `requests-cache`. No type annotations (per repo style); `logging` over `print`.

**Spec:** [docs/superpowers/specs/2026-04-25-scibert-retrieval-design.md](../specs/2026-04-25-scibert-retrieval-design.md)

---

## File structure created by this plan

```
palestinian-drugs-data/
├── src/
│   ├── __init__.py
│   ├── config.py                     # paths, hyperparams, seeds
│   ├── data/
│   │   ├── __init__.py
│   │   ├── atc.py                    # NLM RxNav fetch + cache
│   │   ├── splits.py                 # stratified ingredient split
│   │   ├── queries.py                # Claude Haiku 4.5 query generation
│   │   └── pairs.py                  # Type A + B contrastive triples
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py                  # sentence-transformers training loop
│   └── eval/
│       ├── __init__.py
│       ├── baselines.py              # BM25 + dense retrievers
│       ├── metrics.py                # Recall@k, MRR, NDCG, bootstrap
│       └── run_eval.py               # full ablation table runner
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # shared fixtures
│   ├── data/
│   │   ├── __init__.py
│   │   ├── test_atc.py
│   │   ├── test_splits.py
│   │   ├── test_queries.py
│   │   └── test_pairs.py
│   ├── training/
│   │   └── test_train_smoke.py
│   └── eval/
│       ├── test_baselines.py
│       └── test_metrics.py
├── notebooks/
│   ├── 02_train_scibert.ipynb
│   └── 03_eval_results.ipynb
├── data/
│   ├── atc_mapping.json              # generated, committed
│   ├── generated_queries.jsonl       # generated, committed
│   └── splits/
│       ├── train_ingredients.txt
│       └── test_ingredients.txt
├── eval/
│   └── hand_curated_queries.jsonl    # author writes these
├── requirements.txt
├── .env.example
├── .gitignore
└── pytest.ini
```

---

## Task 1: Repo + dependency setup

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pytest.ini`, `.env.example`
- Create: `src/__init__.py`, `src/config.py`
- Create: `src/data/__init__.py`, `src/training/__init__.py`, `src/eval/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/data/__init__.py`

- [ ] **Step 1: Initialize git**

```bash
cd /Users/wswaileh/my_projects/palestinian-drugs-data
git init
git add corpus/ docs/ build_corpus.py fetch_monographs_v2.py pharmacy_all_drugs.csv fetch_checkpoint_v2.json
git commit -m "chore: snapshot finalized corpus before DL phase"
```

Expected: clean repo with one commit on `main`.

- [ ] **Step 2: Write `.gitignore`**

Create `.gitignore`:

```
__pycache__/
*.pyc
.pytest_cache/
.env
.venv/
venv/
.DS_Store
old-not-usefull-data/
fetch_v2*.log
drug_monographs_v2.csv
*.ipynb_checkpoints/
checkpoints/
.idea/
.vscode/
```

- [ ] **Step 3: Write `requirements.txt`**

Create `requirements.txt`:

```
sentence-transformers==3.3.1
transformers==4.46.2
torch==2.5.1
rank-bm25==0.2.2
faiss-cpu==1.9.0
anthropic==0.40.0
pandas==2.2.3
pyarrow==18.0.0
requests==2.32.3
requests-cache==1.2.1
tqdm==4.67.1
python-dotenv==1.0.1
pytest==8.3.4
pytest-mock==3.14.0
numpy==1.26.4
scikit-learn==1.5.2
```

- [ ] **Step 4: Write `pytest.ini`**

Create `pytest.ini`:

```
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra -q
```

- [ ] **Step 5: Write `.env.example`**

Create `.env.example`:

```
ANTHROPIC_API_KEY=sk-ant-replace-me
```

- [ ] **Step 6: Create empty package files**

Create empty files:

```bash
touch src/__init__.py src/data/__init__.py src/training/__init__.py src/eval/__init__.py
touch tests/__init__.py tests/data/__init__.py tests/training/__init__.py tests/eval/__init__.py
mkdir -p tests/training tests/eval
touch tests/training/__init__.py tests/eval/__init__.py
```

- [ ] **Step 7: Write `src/config.py`**

Create `src/config.py`:

```python
"""Project-wide constants. Single source of truth for paths and hyperparams."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINT_DIR = ROOT / "checkpoints"

CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"
INGREDIENTS_PATH = CORPUS_DIR / "ingredients.parquet"
CATALOG_PATH = CORPUS_DIR / "catalog.parquet"
CATALOG_MAP_PATH = CORPUS_DIR / "catalog_ingredient_map.parquet"

ATC_CACHE_PATH = DATA_DIR / "atc_mapping.json"
GENERATED_QUERIES_PATH = DATA_DIR / "generated_queries.jsonl"
TRAIN_INGREDIENTS_PATH = SPLITS_DIR / "train_ingredients.txt"
TEST_INGREDIENTS_PATH = SPLITS_DIR / "test_ingredients.txt"
HAND_CURATED_QUERIES_PATH = EVAL_DIR / "hand_curated_queries.jsonl"

# Reproducibility
SEED = 42

# Split
TEST_FRACTION = 0.15

# Encoder
ENCODER_MODEL = "allenai/scibert_scivocab_uncased"
MAX_SEQ_LENGTH = 256

# Training
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
EPOCHS = 3

# LLM query generation
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
QUERIES_PER_INGREDIENT = 3

# Eval
RECALL_KS = (1, 5, 10)
MRR_K = 10
NDCG_K = 10
BOOTSTRAP_RESAMPLES = 1000
ALPHA = 0.05  # significance level

DATA_DIR.mkdir(exist_ok=True)
SPLITS_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
```

- [ ] **Step 8: Write `tests/conftest.py`**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures."""

import json
import pytest


@pytest.fixture
def tiny_chunks():
    """Three ingredients × two sections, 6 chunks total."""
    return [
        {"chunk_id": "ibuprofen::indications::0", "ingredient": "ibuprofen",
         "section": "indications", "text": "pain fever inflammation",
         "n_drugs_using": 40, "source": "openfda"},
        {"chunk_id": "ibuprofen::mechanism::0", "ingredient": "ibuprofen",
         "section": "mechanism", "text": "inhibits cyclooxygenase prostaglandin",
         "n_drugs_using": 40, "source": "openfda"},
        {"chunk_id": "naproxen::indications::0", "ingredient": "naproxen",
         "section": "indications", "text": "pain inflammation arthritis",
         "n_drugs_using": 5, "source": "openfda"},
        {"chunk_id": "naproxen::mechanism::0", "ingredient": "naproxen",
         "section": "mechanism", "text": "inhibits cyclooxygenase",
         "n_drugs_using": 5, "source": "openfda"},
        {"chunk_id": "metformin::indications::0", "ingredient": "metformin",
         "section": "indications", "text": "type 2 diabetes glucose",
         "n_drugs_using": 12, "source": "openfda"},
        {"chunk_id": "metformin::mechanism::0", "ingredient": "metformin",
         "section": "mechanism", "text": "decreases hepatic glucose production",
         "n_drugs_using": 12, "source": "openfda"},
    ]


@pytest.fixture
def tiny_atc_map():
    """ibuprofen + naproxen share NSAID class M01AE; metformin is A10BA."""
    return {
        "ibuprofen": ["M01AE01"],
        "naproxen":  ["M01AE02"],
        "metformin": ["A10BA02"],
    }


@pytest.fixture
def tiny_chunks_file(tiny_chunks, tmp_path):
    p = tmp_path / "chunks.jsonl"
    with open(p, "w") as f:
        for c in tiny_chunks:
            f.write(json.dumps(c) + "\n")
    return p
```

- [ ] **Step 9: Install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: clean install, no errors. (On Mac M-series, torch will install MPS-aware build automatically.)

- [ ] **Step 10: Verify pytest discovery**

```bash
.venv/bin/pytest --collect-only
```

Expected: `0 tests collected` (no tests yet).

- [ ] **Step 11: Commit**

```bash
git add .gitignore requirements.txt pytest.ini .env.example src/ tests/
git commit -m "feat(setup): scaffolding, config, and dev dependencies"
```

---

## Task 2: ATC mapping module

**Files:**
- Create: `src/data/atc.py`
- Test: `tests/data/test_atc.py`

NLM RxNav exposes a free, no-key REST API. We map ingredient → ATC by querying the `class/byRxcui` endpoint after first resolving the name to an RxCUI. Responses are cached in `data/atc_mapping.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_atc.py`:

```python
import json
import pytest
from unittest.mock import patch, MagicMock
from src.data import atc


def _mock_rxnav(monkeypatch, name_to_rxcui, rxcui_to_atc):
    """Patch atc._get to return canned RxNav payloads."""
    def fake_get(url, params=None):
        if "/rxcui.json" in url:
            name = (params or {}).get("name", "")
            rxcui = name_to_rxcui.get(name.lower())
            if rxcui:
                return {"idGroup": {"rxnormId": [rxcui]}}
            return {"idGroup": {}}
        if "/class/byRxcui.json" in url:
            rxcui = (params or {}).get("rxcui", "")
            atc_codes = rxcui_to_atc.get(rxcui, [])
            return {"rxclassDrugInfoList": {"rxclassDrugInfo": [
                {"rxclassMinConceptItem": {"classId": code, "classType": "ATC1-4"}}
                for code in atc_codes
            ]}}
        return {}
    monkeypatch.setattr(atc, "_get", fake_get)


def test_lookup_returns_atc_codes(monkeypatch):
    _mock_rxnav(monkeypatch,
                name_to_rxcui={"ibuprofen": "5640"},
                rxcui_to_atc={"5640": ["M01AE01"]})
    assert atc.lookup_atc("ibuprofen") == ["M01AE01"]


def test_lookup_returns_empty_when_unknown(monkeypatch):
    _mock_rxnav(monkeypatch, name_to_rxcui={}, rxcui_to_atc={})
    assert atc.lookup_atc("nonexistent_drug_xyz") == []


def test_build_mapping_writes_cache(monkeypatch, tmp_path):
    _mock_rxnav(monkeypatch,
                name_to_rxcui={"ibuprofen": "5640", "naproxen": "5852"},
                rxcui_to_atc={"5640": ["M01AE01"], "5852": ["M01AE02"]})
    cache_path = tmp_path / "atc.json"
    mapping = atc.build_mapping(["ibuprofen", "naproxen"], cache_path=cache_path)
    assert mapping == {"ibuprofen": ["M01AE01"], "naproxen": ["M01AE02"]}
    assert json.loads(cache_path.read_text()) == mapping


def test_build_mapping_uses_cache_on_second_call(monkeypatch, tmp_path):
    cache_path = tmp_path / "atc.json"
    cache_path.write_text(json.dumps({"ibuprofen": ["M01AE01"]}))
    # If the cache is honored, _get should never be called.
    monkeypatch.setattr(atc, "_get",
                        lambda *a, **kw: pytest.fail("should not hit network"))
    mapping = atc.build_mapping(["ibuprofen"], cache_path=cache_path)
    assert mapping == {"ibuprofen": ["M01AE01"]}


def test_atc_level3():
    assert atc.atc_level3("M01AE01") == "M01AE"
    assert atc.atc_level3("A10BA02") == "A10BA"
    assert atc.atc_level3("") == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/data/test_atc.py -v
```

Expected: ImportError (`src.data.atc` doesn't exist yet).

- [ ] **Step 3: Implement `src/data/atc.py`**

Create `src/data/atc.py`:

```python
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
    """HTTP GET with one retry on transient failure. Returns parsed JSON or {}."""
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
        cache = json.loads(cache_path.read_text())

    todo = [i for i in ingredients if i not in cache]
    logger.info("ATC: %d cached, %d to fetch", len(cache), len(todo))
    for i, name in enumerate(todo, 1):
        cache[name] = lookup_atc(name)
        if i % 25 == 0:
            cache_path.write_text(json.dumps(cache, indent=2))
            logger.info("ATC progress: %d/%d", i, len(todo))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))
    return {k: cache[k] for k in ingredients}


def atc_level3(code):
    """First five chars of an ATC code (level 3 = therapeutic subgroup)."""
    return code[:5] if code else ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/data/test_atc.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Add CLI to build the production cache**

Append to `src/data/atc.py`:

```python
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
```

- [ ] **Step 6: Run the production cache build**

```bash
.venv/bin/python -m src.data.atc
```

Expected: prints progress every 25 ingredients; final coverage ≥85%; writes `data/atc_mapping.json`.

If coverage < 85%, list uncovered names with `python -c "import json; d=json.load(open('data/atc_mapping.json')); [print(k) for k,v in d.items() if not v][:30]"` and add aliases in `fetch_monographs_v2.py` if obvious typos. (Plan continues regardless — degraded entries fall back to random hard negatives.)

- [ ] **Step 7: Commit**

```bash
git add src/data/atc.py tests/data/test_atc.py data/atc_mapping.json
git commit -m "feat(data): ATC mapping via NLM RxNav with disk cache"
```

---

## Task 3: Stratified ingredient train/test split

**Files:**
- Create: `src/data/splits.py`
- Test: `tests/data/test_splits.py`
- Output: `data/splits/train_ingredients.txt`, `data/splits/test_ingredients.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_splits.py`:

```python
from src.data import splits


def test_split_disjoint_and_proportional(tiny_atc_map):
    ingredients = ["ibuprofen", "naproxen", "metformin"]
    train, test = splits.stratified_split(ingredients, tiny_atc_map,
                                          test_fraction=0.34, seed=42)
    assert set(train).isdisjoint(set(test))
    assert set(train) | set(test) == set(ingredients)
    assert len(test) == 1  # 34% of 3, rounded


def test_split_is_deterministic_across_seeds(tiny_atc_map):
    ingredients = ["ibuprofen", "naproxen", "metformin"]
    a = splits.stratified_split(ingredients, tiny_atc_map, 0.34, seed=42)
    b = splits.stratified_split(ingredients, tiny_atc_map, 0.34, seed=42)
    assert a == b


def test_split_unmapped_ingredients_go_to_a_random_bucket(tiny_atc_map):
    ingredients = ["ibuprofen", "naproxen", "metformin", "unknown_x", "unknown_y"]
    atc_map = {**tiny_atc_map, "unknown_x": [], "unknown_y": []}
    train, test = splits.stratified_split(ingredients, atc_map, 0.4, seed=42)
    assert set(train).isdisjoint(set(test))
    assert "unknown_x" in train + test
    assert "unknown_y" in train + test


def test_save_and_load_round_trip(tmp_path):
    train = ["ibuprofen", "metformin"]
    test = ["naproxen"]
    train_path = tmp_path / "train.txt"
    test_path  = tmp_path / "test.txt"
    splits.save_split(train, test, train_path, test_path)
    loaded_train, loaded_test = splits.load_split(train_path, test_path)
    assert loaded_train == train
    assert loaded_test == test
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/data/test_splits.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/data/splits.py`**

Create `src/data/splits.py`:

```python
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
    train_path.write_text("\n".join(train) + "\n")
    test_path.write_text("\n".join(test) + "\n")


def load_split(train_path=None, test_path=None):
    train_path = Path(train_path or config.TRAIN_INGREDIENTS_PATH)
    test_path  = Path(test_path  or config.TEST_INGREDIENTS_PATH)
    train = [l.strip() for l in train_path.read_text().splitlines() if l.strip()]
    test  = [l.strip() for l in test_path.read_text().splitlines()  if l.strip()]
    return train, test


def main():
    """CLI: build the production split from ingredients.parquet + atc cache."""
    import json
    import pandas as pd
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ing = pd.read_parquet(config.INGREDIENTS_PATH)
    found = sorted(ing[ing["found"]]["ingredient"].tolist())
    atc_map = json.loads(config.ATC_CACHE_PATH.read_text())

    train, test = stratified_split(found, atc_map)
    save_split(train, test)
    logger.info("Wrote %s and %s", config.TRAIN_INGREDIENTS_PATH,
                config.TEST_INGREDIENTS_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/data/test_splits.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Build the production split**

```bash
.venv/bin/python -m src.data.splits
wc -l data/splits/*.txt
```

Expected: ~559 train, ~98 test (totals depend on monograph coverage).

- [ ] **Step 6: Commit**

```bash
git add src/data/splits.py tests/data/test_splits.py data/splits/
git commit -m "feat(data): stratified ATC-level-3 ingredient split"
```

---

## Task 4: LLM query generation (Claude Haiku 4.5)

**Files:**
- Create: `src/data/queries.py`
- Test: `tests/data/test_queries.py`
- Output: `data/generated_queries.jsonl`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_queries.py`:

```python
import json
import pytest
from unittest.mock import MagicMock
from src.data import queries


def _mock_anthropic(text):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    client.messages.create.return_value = msg
    return client


def test_generate_for_ingredient_parses_json(monkeypatch):
    canned = json.dumps({"queries": [
        "treatment for pain and fever",
        "what helps inflammation in arthritis",
        "drug for headache and muscle ache",
    ]})
    client = _mock_anthropic(canned)
    out = queries.generate_for_ingredient(client, "ibuprofen",
                                           "Pain, fever, inflammation")
    assert out == [
        "treatment for pain and fever",
        "what helps inflammation in arthritis",
        "drug for headache and muscle ache",
    ]


def test_generate_for_ingredient_strips_code_fences(monkeypatch):
    canned = "```json\n" + json.dumps({"queries": ["a", "b", "c"]}) + "\n```"
    client = _mock_anthropic(canned)
    out = queries.generate_for_ingredient(client, "naproxen", "Pain.")
    assert out == ["a", "b", "c"]


def test_generate_for_ingredient_returns_empty_on_malformed(monkeypatch):
    client = _mock_anthropic("not json at all")
    out = queries.generate_for_ingredient(client, "x", "y")
    assert out == []


def test_build_query_set_is_idempotent(monkeypatch, tmp_path):
    cache_path = tmp_path / "queries.jsonl"
    cache_path.write_text(json.dumps(
        {"ingredient": "ibuprofen", "queries": ["existing query"]}) + "\n")

    client = _mock_anthropic(json.dumps({"queries": ["never called"]}))
    monkeypatch.setattr(queries, "_make_client", lambda: client)

    out = queries.build_query_set(["ibuprofen"], {"ibuprofen": "ind text"},
                                   cache_path=cache_path)
    assert out == {"ibuprofen": ["existing query"]}
    client.messages.create.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/data/test_queries.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/data/queries.py`**

Create `src/data/queries.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/data/test_queries.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Generate the production query set**

```bash
cp .env.example .env
# Edit .env to set ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python -m src.data.queries
wc -l data/generated_queries.jsonl
```

Expected: ~657 lines (one per ingredient with monograph data); ~$2 in API spend; resumable on Ctrl-C.

- [ ] **Step 6: Spot-check 5 generated queries**

```bash
.venv/bin/python -c "
import json
for line in open('data/generated_queries.jsonl').readlines()[:5]:
    rec = json.loads(line)
    print(rec['ingredient'])
    for q in rec['queries']:
        print('  -', q)
"
```

Expected: 3 distinct, clinically-phrased queries per ingredient. If queries look templated/generic, tighten the system prompt and regenerate (delete `data/generated_queries.jsonl` first).

- [ ] **Step 7: Commit**

```bash
git add src/data/queries.py tests/data/test_queries.py data/generated_queries.jsonl
git commit -m "feat(data): clinical query generation via Claude Haiku 4.5"
```

---

## Task 5: Type A pair construction (cross-section)

**Files:**
- Create: `src/data/pairs.py`
- Test: `tests/data/test_pairs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_pairs.py`:

```python
import json
import pytest
from src.data import pairs


def test_load_chunks_groups_by_ingredient(tiny_chunks_file):
    by_ing = pairs.load_chunks_by_ingredient(tiny_chunks_file)
    assert set(by_ing.keys()) == {"ibuprofen", "naproxen", "metformin"}
    assert len(by_ing["ibuprofen"]) == 2
    sections = {c["section"] for c in by_ing["ibuprofen"]}
    assert sections == {"indications", "mechanism"}


def test_type_a_anchor_and_positive_share_ingredient(tiny_chunks_file, tiny_atc_map):
    by_ing = pairs.load_chunks_by_ingredient(tiny_chunks_file)
    train = ["ibuprofen", "naproxen", "metformin"]
    triples = pairs.build_type_a_triples(by_ing, train, tiny_atc_map, seed=42)
    assert len(triples) > 0
    for t in triples:
        a, p, n = t["anchor"], t["positive"], t["negative"]
        assert a["ingredient"] == p["ingredient"]
        assert a["section"] == "indications"
        assert p["section"] != "indications"
        assert n["ingredient"] != a["ingredient"]


def test_type_a_hard_negative_prefers_same_atc(tiny_chunks_file, tiny_atc_map):
    by_ing = pairs.load_chunks_by_ingredient(tiny_chunks_file)
    train = ["ibuprofen", "naproxen", "metformin"]
    triples = pairs.build_type_a_triples(by_ing, train, tiny_atc_map, seed=42)
    # ibuprofen ↔ naproxen share M01AE; metformin is A10BA.
    # Every ibuprofen triple's negative should be naproxen.
    ibu = [t for t in triples if t["anchor"]["ingredient"] == "ibuprofen"]
    assert ibu, "expected ibuprofen anchors"
    assert all(t["negative"]["ingredient"] == "naproxen" for t in ibu)


def test_type_a_excludes_held_out_ingredients(tiny_chunks_file, tiny_atc_map):
    by_ing = pairs.load_chunks_by_ingredient(tiny_chunks_file)
    train = ["ibuprofen", "metformin"]   # naproxen held out
    triples = pairs.build_type_a_triples(by_ing, train, tiny_atc_map, seed=42)
    seen = {t["anchor"]["ingredient"] for t in triples} | {
            t["negative"]["ingredient"] for t in triples}
    assert "naproxen" not in seen
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/data/test_pairs.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/data/pairs.py` Type A**

Create `src/data/pairs.py`:

```python
"""Build contrastive (anchor, positive, negative) triples from the corpus.

Two complementary triple types:

* Type A — cross-section self-supervision. Anchor = indications chunk; positive
  = a chunk from another section of the same ingredient; hard negative =
  matching-section chunk from a same-ATC-class ingredient.

* Type B — LLM-generated query → ingredient. Anchor = synthetic clinical query;
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
        # Random fallback: pick any other ingredient with matching section.
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/data/test_pairs.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/pairs.py tests/data/test_pairs.py
git commit -m "feat(data): Type A cross-section contrastive triples"
```

---

## Task 6: Type B pair construction (query → indication)

**Files:**
- Modify: `src/data/pairs.py`, `tests/data/test_pairs.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/data/test_pairs.py`:

```python
def test_type_b_anchor_is_query_positive_is_indication(tiny_chunks_file, tiny_atc_map):
    by_ing = pairs.load_chunks_by_ingredient(tiny_chunks_file)
    train = ["ibuprofen", "naproxen", "metformin"]
    queries_map = {
        "ibuprofen": ["pain relief", "anti-inflammatory for arthritis"],
        "naproxen":  ["arthritis pain"],
        "metformin": ["type 2 diabetes management"],
    }
    triples = pairs.build_type_b_triples(by_ing, train, queries_map,
                                         tiny_atc_map, seed=42)
    assert len(triples) == 4  # 2 + 1 + 1
    for t in triples:
        assert isinstance(t["anchor"], str)
        assert t["positive"]["section"] == "indications"
        assert t["negative"]["section"] == "indications"
        assert t["positive"]["ingredient"] != t["negative"]["ingredient"]


def test_type_b_hard_negative_prefers_same_atc(tiny_chunks_file, tiny_atc_map):
    by_ing = pairs.load_chunks_by_ingredient(tiny_chunks_file)
    train = ["ibuprofen", "naproxen", "metformin"]
    queries_map = {"ibuprofen": ["pain"], "naproxen": [], "metformin": []}
    triples = pairs.build_type_b_triples(by_ing, train, queries_map,
                                         tiny_atc_map, seed=42)
    # ibuprofen's hard negative should be naproxen (same M01AE class).
    assert triples[0]["negative"]["ingredient"] == "naproxen"


def test_combined_dataset_mixes_a_and_b(tiny_chunks_file, tiny_atc_map):
    by_ing = pairs.load_chunks_by_ingredient(tiny_chunks_file)
    train = ["ibuprofen", "naproxen", "metformin"]
    queries_map = {"ibuprofen": ["pain"], "naproxen": ["pain"], "metformin": ["dm"]}
    combined = pairs.build_combined_dataset(by_ing, train, queries_map,
                                            tiny_atc_map, seed=42)
    kinds = {t["kind"] for t in combined}
    assert kinds == {"A", "B"}
    # Anchor format differs by kind.
    for t in combined:
        if t["kind"] == "A":
            assert isinstance(t["anchor"], dict)
        else:
            assert isinstance(t["anchor"], str)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/data/test_pairs.py -v
```

Expected: 3 new failures (`build_type_b_triples`, `build_combined_dataset` undefined).

- [ ] **Step 3: Append Type B + combined to `src/data/pairs.py`**

Append to `src/data/pairs.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/data/test_pairs.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/pairs.py tests/data/test_pairs.py
git commit -m "feat(data): Type B query→indication triples + combined dataset"
```

---

## Task 7: Training loop

**Files:**
- Create: `src/training/train.py`
- Test: `tests/training/test_train_smoke.py`

The training loop uses `sentence-transformers`' `MultipleNegativesRankingLoss`, which expects `InputExample(texts=[anchor, positive, negative])`. The hard negative gets concatenated to the in-batch easy negatives automatically, doubling the effective negative count.

- [ ] **Step 1: Write the smoke test**

Create `tests/training/test_train_smoke.py`:

```python
import os
import pytest
from src.training import train


@pytest.mark.skipif(
    not os.environ.get("RUN_SMOKE_TRAIN"),
    reason="Skips without RUN_SMOKE_TRAIN=1 (downloads SciBERT, ~440MB)."
)
def test_one_step_does_not_crash(tmp_path, tiny_chunks, tiny_atc_map):
    """Builds 3 triples and runs 1 epoch with batch_size=2 — smoke only."""
    from src.data import pairs
    by_ing = {"ibuprofen": [c for c in tiny_chunks if c["ingredient"]=="ibuprofen"],
              "naproxen":  [c for c in tiny_chunks if c["ingredient"]=="naproxen"],
              "metformin": [c for c in tiny_chunks if c["ingredient"]=="metformin"]}
    triples = pairs.build_combined_dataset(
        by_ing, ["ibuprofen", "naproxen", "metformin"],
        {"ibuprofen":["pain"], "naproxen":["pain"], "metformin":["dm"]},
        tiny_atc_map, seed=42)
    out_dir = tmp_path / "smoke_model"
    train.train(triples, output_dir=out_dir, epochs=1, batch_size=2,
                model_name="prajjwal1/bert-tiny")  # 4MB tiny model for CI
    assert (out_dir / "config.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/training/test_train_smoke.py -v
```

Expected: collection error (`src.training.train` missing).

- [ ] **Step 3: Implement `src/training/train.py`**

Create `src/training/train.py`:

```python
"""Train a sentence-transformer with InfoNCE + explicit hard negatives."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader

from src import config

logger = logging.getLogger(__name__)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _triple_to_example(t):
    """Convert a {anchor, positive, negative, kind} record to an InputExample.

    For Type A the anchor is a chunk dict; for Type B it is a string.
    """
    anchor = t["anchor"]["text"] if isinstance(t["anchor"], dict) else t["anchor"]
    return InputExample(texts=[anchor, t["positive"]["text"], t["negative"]["text"]])


def _build_model(model_name):
    word = models.Transformer(model_name, max_seq_length=config.MAX_SEQ_LENGTH)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[word, pool])


def train(triples, output_dir, epochs=None, batch_size=None,
          learning_rate=None, model_name=None, warmup_ratio=None):
    """Train and save a SentenceTransformer model.

    Each `triples` element must have `anchor` (str or {text}), `positive` ({text}),
    `negative` ({text}). Negatives are concatenated to the batch as hard negatives.
    """
    epochs = config.EPOCHS if epochs is None else epochs
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    learning_rate = config.LEARNING_RATE if learning_rate is None else learning_rate
    warmup_ratio = config.WARMUP_RATIO if warmup_ratio is None else warmup_ratio
    model_name = config.ENCODER_MODEL if model_name is None else model_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(config.SEED)

    examples = [_triple_to_example(t) for t in triples]
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)
    model = _build_model(model_name)
    loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_epoch = len(loader)
    warmup_steps = int(warmup_ratio * steps_per_epoch * epochs)
    logger.info("training: examples=%d batches/epoch=%d epochs=%d warmup=%d",
                len(examples), steps_per_epoch, epochs, warmup_steps)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        weight_decay=config.WEIGHT_DECAY,
        optimizer_params={"lr": learning_rate},
        use_amp=torch.cuda.is_available(),
        checkpoint_path=str(output_dir / "checkpoints"),
        checkpoint_save_steps=steps_per_epoch,
        checkpoint_save_total_limit=epochs,
        output_path=str(output_dir),
    )
    return model


def main():
    """CLI entrypoint: load corpus + split + queries, build triples, train."""
    import json
    import pandas as pd
    import argparse

    from src.data import pairs
    from src.data.splits import load_split

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="ab_atc",
                        choices=["a_only", "b_only", "ab_random", "ab_atc"])
    parser.add_argument("--output-dir", default="checkpoints/scibert_ft")
    args = parser.parse_args()

    by_ing = pairs.load_chunks_by_ingredient()
    train_ings, _ = load_split()

    queries_map = {}
    if config.GENERATED_QUERIES_PATH.exists():
        for line in config.GENERATED_QUERIES_PATH.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                queries_map[rec["ingredient"]] = rec["queries"]

    atc_map = json.loads(config.ATC_CACHE_PATH.read_text())

    if args.variant == "ab_random":
        atc_map = {ing: [] for ing in atc_map}  # disables ATC hard negatives

    if args.variant == "a_only":
        triples = pairs.build_type_a_triples(by_ing, train_ings, atc_map)
    elif args.variant == "b_only":
        triples = pairs.build_type_b_triples(by_ing, train_ings, queries_map, atc_map)
    else:
        triples = pairs.build_combined_dataset(by_ing, train_ings, queries_map, atc_map)

    train(triples, output_dir=Path(args.output_dir) / args.variant)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test**

```bash
RUN_SMOKE_TRAIN=1 .venv/bin/pytest tests/training/test_train_smoke.py -v
```

Expected: passes (downloads `prajjwal1/bert-tiny` once, ~10s training).

- [ ] **Step 5: Commit**

```bash
git add src/training/train.py tests/training/test_train_smoke.py
git commit -m "feat(training): SciBERT contrastive training loop"
```

---

## Task 8: BM25 + dense baselines

**Files:**
- Create: `src/eval/baselines.py`
- Test: `tests/eval/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_baselines.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/eval/test_baselines.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/eval/baselines.py`**

Create `src/eval/baselines.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/eval/test_baselines.py -v
```

Expected: 3 passed (downloads bert-tiny once, cached).

- [ ] **Step 5: Commit**

```bash
git add src/eval/baselines.py tests/eval/test_baselines.py
git commit -m "feat(eval): BM25 and dense-encoder retrievers"
```

---

## Task 9: Evaluation metrics + paired bootstrap

**Files:**
- Create: `src/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_metrics.py`:

```python
import math
import numpy as np
from src.eval import metrics


def test_recall_at_k_on_known_ranking():
    # Gold = "metformin"; retrieved ingredients in rank order:
    retrieved = ["ibuprofen", "metformin", "naproxen"]
    assert metrics.recall_at_k(retrieved, ["metformin"], k=1) == 0.0
    assert metrics.recall_at_k(retrieved, ["metformin"], k=2) == 1.0
    assert metrics.recall_at_k(retrieved, ["metformin"], k=3) == 1.0


def test_recall_with_multiple_gold():
    retrieved = ["a", "b", "c", "d"]
    # 2 of 3 gold present in top-3
    assert math.isclose(
        metrics.recall_at_k(retrieved, ["b", "c", "x"], k=3), 2/3)


def test_mrr_uses_first_relevant():
    retrieved = ["a", "b", "c"]
    assert metrics.mrr_at_k(retrieved, ["c"], k=10) == 1/3
    assert metrics.mrr_at_k(retrieved, ["x"], k=10) == 0.0


def test_ndcg_perfect_ranking():
    retrieved = ["a", "b", "c"]
    assert math.isclose(metrics.ndcg_at_k(retrieved, ["a"], k=3), 1.0)


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 0.4, size=200)
    b = rng.uniform(0.5, 0.9, size=200)  # b clearly larger
    res = metrics.paired_bootstrap(a, b, n_resamples=500, seed=0)
    assert res["mean_diff"] > 0
    assert res["ci_low"] > 0
    assert res["p_value"] < 0.05


def test_paired_bootstrap_no_difference():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.4, 0.6, size=200)
    b = rng.uniform(0.4, 0.6, size=200)
    res = metrics.paired_bootstrap(a, b, n_resamples=500, seed=0)
    assert res["p_value"] > 0.05
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/eval/test_metrics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/eval/metrics.py`**

Create `src/eval/metrics.py`:

```python
"""Retrieval metrics + paired bootstrap for significance testing.

Inputs are *retrieved ingredient names* (one list per query, in rank order)
and *gold ingredient names* (one set per query). Working at ingredient
granularity matches the spec — chunk-level retrieval is collapsed to its
ingredient before scoring.
"""

import math
import numpy as np


def recall_at_k(retrieved, gold, k):
    if not gold:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for g in gold if g in top)
    return hits / len(gold)


def mrr_at_k(retrieved, gold, k):
    for i, ing in enumerate(retrieved[:k], 1):
        if ing in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, gold, k):
    dcg = 0.0
    for i, ing in enumerate(retrieved[:k], 1):
        if ing in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def paired_bootstrap(a, b, n_resamples=1000, seed=0):
    """Two-sided paired bootstrap on per-query metric arrays.

    Returns dict with keys mean_diff (b-a), ci_low, ci_high, p_value (for null
    hypothesis "no difference").
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    diffs = b - a
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[r] = diffs[idx].mean()
    mean_diff = float(diffs.mean())
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    # Two-sided p-value: fraction of resamples on the wrong side of zero, ×2.
    if mean_diff >= 0:
        p = (means <= 0).mean() * 2
    else:
        p = (means >= 0).mean() * 2
    return {"mean_diff": mean_diff, "ci_low": float(ci_low),
            "ci_high": float(ci_high), "p_value": float(min(p, 1.0))}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/eval/test_metrics.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): retrieval metrics + paired bootstrap"
```

---

## Task 10: Eval harness (full ablation table)

**Files:**
- Create: `src/eval/run_eval.py`

This module loads each retriever (BM25, MiniLM, BioBERT-MNLI, off-the-shelf SciBERT, fine-tuned variants), runs them on Set 1 and Set 2, computes per-query metrics, and emits a results JSON + Markdown table.

- [ ] **Step 1: Implement `src/eval/run_eval.py`**

Create `src/eval/run_eval.py`:

```python
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

    # Significance vs. the strongest non-FT baseline (off-the-shelf SciBERT).
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
```

- [ ] **Step 2: Smoke-test the harness on a tiny corpus**

Append to `tests/eval/test_baselines.py`:

```python
def test_run_eval_on_tiny(tiny_chunks_file, tmp_path):
    """Pipe-checking the harness wires together — no model accuracy assertion."""
    from src.eval import run_eval

    chunks = []
    with open(tiny_chunks_file) as fh:
        for line in fh:
            chunks.append(__import__("json").loads(line))
    queries = [{"query": "diabetes glucose", "gold": ["metformin"]}]
    r = run_eval.build_retriever("bm25", chunks)
    out = run_eval.evaluate_retriever(r, queries, k_max=3)
    assert out["recall@1"][0] == 1.0
```

```bash
.venv/bin/pytest tests/eval/test_baselines.py -v
```

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add src/eval/run_eval.py tests/eval/test_baselines.py
git commit -m "feat(eval): full ablation runner with significance testing"
```

---

## Task 11: Hand-curated query template (you fill in 30)

**Files:**
- Create: `eval/hand_curated_queries.jsonl`

- [ ] **Step 1: Write the template with 3 example queries**

Create `eval/hand_curated_queries.jsonl`:

```jsonl
{"query": "first-line antihypertensive for a 65-year-old with type 2 diabetes and CKD stage 3", "gold": ["losartan", "valsartan"]}
{"query": "pediatric fever 38.5C in a 5-year-old, weight-based dose", "gold": ["acetaminophen", "ibuprofen"]}
{"query": "broad-spectrum antibiotic for community-acquired pneumonia in adult outpatient", "gold": ["amoxicillin", "azithromycin"]}
```

- [ ] **Step 2: Author 27 more queries (~2-3 hours of your time)**

Open `eval/hand_curated_queries.jsonl` and append 27 more lines. Diversity targets:
- ≥6 cardiovascular (HTN, AF, lipid)
- ≥5 endocrine (diabetes, thyroid)
- ≥4 infections (antibacterial / antiviral)
- ≥4 pain / inflammation
- ≥3 psychiatric
- ≥3 GI
- ≥3 women's health / pediatrics / geriatrics edge cases

For each, set `gold` to **only ingredients present in `corpus/ingredients.parquet` with `found=True`**. Verify with:

```bash
.venv/bin/python -c "
import pandas as pd, json
ings = set(pd.read_parquet('corpus/ingredients.parquet').query('found')['ingredient'])
for line in open('eval/hand_curated_queries.jsonl'):
    rec = json.loads(line)
    missing = [g for g in rec['gold'] if g not in ings]
    if missing: print('MISSING:', rec['query'], '→', missing)
"
```

Expected: zero `MISSING:` lines after authoring.

- [ ] **Step 3: Commit**

```bash
git add eval/hand_curated_queries.jsonl
git commit -m "eval: hand-curated 30 clinical queries"
```

---

## Task 12: Training notebook (Colab)

**Files:**
- Create: `notebooks/02_train_scibert.ipynb`

The notebook is a thin Colab driver around `src/training/train.py`. It mounts Drive, clones the repo, installs deps, runs all four FT variants, and uploads checkpoints to Drive.

- [ ] **Step 1: Generate the notebook from a JSON template**

Create `notebooks/02_train_scibert.ipynb`:

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": [
   "# Train SciBERT contrastive variants (Colab T4)\n",
   "Runs A-only, B-only, AB-random, AB-ATC, ~30-60 min total."
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "from google.colab import drive\n",
   "drive.mount('/content/drive')"
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "%%bash\n",
   "rm -rf /content/repo\n",
   "git clone https://github.com/<YOUR-USER>/palestinian-drugs-data.git /content/repo\n",
   "cd /content/repo && git checkout main\n",
   "pip install -q -r requirements.txt"
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "%cd /content/repo\n",
   "import os\n",
   "os.environ['ANTHROPIC_API_KEY'] = ''  # only needed if regenerating queries\n",
   "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "%%bash\n",
   "for variant in a_only b_only ab_random ab_atc; do\n",
   "  python -m src.training.train --variant $variant \\\n",
   "    --output-dir /content/drive/MyDrive/scibert_ft\n",
   "done"
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "%%bash\n",
   "ls -la /content/drive/MyDrive/scibert_ft/"
  ]}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Validate notebook JSON**

```bash
.venv/bin/python -c "import json; json.load(open('notebooks/02_train_scibert.ipynb')); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_train_scibert.ipynb
git commit -m "feat(notebooks): Colab training driver for all four FT variants"
```

---

## Task 13: Eval notebook (paper table generation)

**Files:**
- Create: `notebooks/03_eval_results.ipynb`

- [ ] **Step 1: Create the eval notebook**

Create `notebooks/03_eval_results.ipynb`:

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": [
   "# Ablation table (Set 1 + Set 2)\n",
   "Run after training. Loads each retriever, evaluates on both sets, prints the paper table and the significance file."
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "import sys; sys.path.insert(0, '..')\n",
   "import logging; logging.basicConfig(level=logging.INFO)\n",
   "from src.eval import run_eval\n",
   "df, sig = run_eval.run_all(\n",
   "    test_ingredients_path='../data/splits/test_ingredients.txt',\n",
   "    queries_path='../data/generated_queries.jsonl',\n",
   "    hand_path='../eval/hand_curated_queries.jsonl',\n",
   "    output_dir='../eval/results',\n",
   ")\n",
   "df"
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "sig.query('p_value < 0.05')"
  ]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "print(df.to_markdown(index=False))"
  ]}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Validate notebook**

```bash
.venv/bin/python -c "import json; json.load(open('notebooks/03_eval_results.ipynb')); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_eval_results.ipynb
git commit -m "feat(notebooks): ablation table + significance reporting"
```

---

## Task 14: End-to-end dry run on Mac

**Files:** none

This is the integration check before pushing to Colab. It runs everything except the actual full SciBERT training (which is reserved for Colab T4).

- [ ] **Step 1: Smoke-train one variant on Mac with a tiny model**

```bash
.venv/bin/python -m src.training.train --variant a_only \
    --output-dir checkpoints/scibert_ft_smoke \
    2>&1 | tail -20
```

Replace `ENCODER_MODEL` temporarily with `prajjwal1/bert-tiny` in `src/config.py` for this dry run, or pass `--model-name` if you've added that flag. **Revert afterward.**

Expected: training completes in ~5 min on Mac CPU/MPS.

- [ ] **Step 2: Run BM25 + off-the-shelf SciBERT eval only**

Comment out the four FT rows from `VARIANTS` in `src/eval/run_eval.py` temporarily, then:

```bash
.venv/bin/python -m src.eval.run_eval
```

Expected: ablation table prints with two rows (BM25, scibert_offshelf) on Set 1 and (if hand-curated authored) Set 2. **Revert VARIANTS afterward.**

- [ ] **Step 3: Commit any small fixes uncovered by dry run**

```bash
git status
git add -p   # stage selectively
git commit -m "fix: tweaks from end-to-end dry run"
```

---

## Task 15: Train on Colab + run final ablation

- [ ] **Step 1: Push the repo to GitHub**

```bash
gh repo create palestinian-drugs-data --private --source=. --push
```

- [ ] **Step 2: Open `notebooks/02_train_scibert.ipynb` in Colab**

Edit the `git clone` cell to use your new repo URL. Run all cells. Expected wall-clock: 30–60 min for all four variants on free T4.

- [ ] **Step 3: Sync checkpoints back to local**

Download `checkpoints/scibert_ft/{a_only,b_only,ab_random,ab_atc}/` from Drive into the repo (or `rclone`):

```bash
mkdir -p checkpoints/scibert_ft
rsync -av <drive-mount>/scibert_ft/ checkpoints/scibert_ft/
```

- [ ] **Step 4: Run the eval notebook locally**

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/03_eval_results.ipynb \
    --output 03_eval_results.executed.ipynb
```

Expected: `eval/results/ablation_table.csv` and `eval/results/significance.csv` written. The executed notebook contains the final paper table.

- [ ] **Step 5: Verify success criteria**

```bash
.venv/bin/python -c "
import pandas as pd
sig = pd.read_csv('eval/results/significance.csv')
ab_atc = sig[sig['variant'] == 'scibert_ft_ab_atc']
wins = ab_atc[(ab_atc['p_value'] < 0.05) & (ab_atc['ci_low'] > 0)]
print(f'Wins (p<0.05, CI lower > 0): {len(wins)} / {len(ab_atc)}')
print(wins[['set','metric','mean_diff','p_value']].to_string())
"
```

Expected: wins on at least Recall@5 and MRR@10 on Set 1; ideally Set 2 too. If not, report honestly in the paper.

- [ ] **Step 6: Final commit**

```bash
git add eval/results/ checkpoints/scibert_ft/
git commit -m "results: final ablation table and significance report"
git push
```

---

## Self-Review

**Spec coverage:**
- §1 Goal & success criteria → Task 9 (metrics), Task 10 (significance pipeline), Task 15 (verification step). ✅
- §2 Inputs → consumed throughout (Tasks 5–10). ✅
- §3 Train/test split → Task 3. ✅
- §4 ATC mapping → Task 2. ✅
- §5 Pair construction (Type A + B) → Tasks 5–6. ✅
- §6 Model/loss/optimization → Task 7. ✅
- §7 Baselines (BM25 + 3 dense) → Task 8. ✅
- §8 Eval (Set 1 auto + Set 2 hand) → Tasks 10, 11. ✅
- §9 Reproducibility (seeds, splits, queries, checkpoints) → Tasks 1, 3, 4, 7, 12. ✅
- §10 Code layout → matches plan's File-structure block. ✅
- §11 Out of scope → not built (correct). ✅
- §12 Risks (Colab disconnect, ATC coverage, query templating) → Task 7 (per-epoch checkpoints), Task 2 (coverage gate), Task 4 (spot-check step). ✅
- §13 Definition of done → Task 15 step 5 verifies. ✅

**Placeholder scan:** No TBD/TODO/"implement later" tokens. Test code, implementation code, and exact commands present in every step.

**Type/name consistency:**
- `ENCODER_MODEL`, `BATCH_SIZE`, `SEED` etc. defined once in `src/config.py`, referenced throughout.
- `build_type_a_triples`, `build_type_b_triples`, `build_combined_dataset` named identically in implementation, tests, and the train CLI.
- `BM25Retriever`, `DenseRetriever` consistent between baselines.py and run_eval.py.
- `paired_bootstrap` returns `{mean_diff, ci_low, ci_high, p_value}` and the run-eval consumer uses those exact keys.

**Risk noted:** `requirements.txt` pins specific versions; if Colab installs a different `torch` automatically, the notebook may need `pip install -q --force-reinstall torch==2.5.1`. Mitigated by Task 12's plain `pip install -q -r requirements.txt` letting Colab resolve compatibly.
