"""Project-wide constants. Single source of truth for paths and hyperparams."""

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
