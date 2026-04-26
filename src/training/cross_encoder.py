"""Fine-tune a cross-encoder on biomedical query-chunk pairs from src.data.pairs.

Cross-encoders see (query, chunk) jointly with full bidirectional attention,
catching fine-grained semantics single-vector bi-encoders miss. Fine-tuning
on domain-specific pairs is the standard architectural step that takes a
biomedical retriever from "decent" to "competitive."
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

# Disable third-party trackers BEFORE importing transformers (Trainer
# auto-detects wandb/codecarbon and crashes without keys).
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from src import config

logger = logging.getLogger(__name__)

DEFAULT_BASE_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _triple_to_examples(t):
    """One contrastive triple → two CE examples (positive=1, negative=0)."""
    anchor = t["anchor"]["text"] if isinstance(t["anchor"], dict) else t["anchor"]
    return [
        InputExample(texts=[anchor, t["positive"]["text"]], label=1.0),
        InputExample(texts=[anchor, t["negative"]["text"]], label=0.0),
    ]


def train(triples, output_dir, base_model=DEFAULT_BASE_CE, epochs=3,
          batch_size=32, learning_rate=2e-5, warmup_ratio=0.10):
    """Train and save a CrossEncoder. Each triple yields one positive + one
    negative training example."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.SEED)

    examples = []
    for t in triples:
        examples.extend(_triple_to_examples(t))
    logger.info("CE training: %d examples (%d triples × 2)",
                len(examples), len(triples))

    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)
    model = CrossEncoder(base_model, num_labels=1, max_length=256)
    warmup_steps = int(warmup_ratio * len(loader) * epochs)

    model.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        weight_decay=config.WEIGHT_DECAY,
        optimizer_params={"lr": learning_rate},
        use_amp=torch.cuda.is_available(),
        show_progress_bar=True,
    )
    # CrossEncoder.fit() only saves on evaluator-triggered improvement.
    # Without an evaluator we save explicitly at the end.
    model.save(str(output_dir))
    logger.info("CE saved to %s", output_dir)
    return model


def main():
    from src.data import pairs
    from src.data.splits import load_split

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="ab_atc",
                        choices=["a_only", "b_only", "ab_random", "ab_atc"])
    parser.add_argument("--output-dir", default="checkpoints/cross_encoder")
    parser.add_argument("--base-model", default=DEFAULT_BASE_CE)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
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
        atc_map = {ing: [] for ing in atc_map}

    if args.variant == "a_only":
        triples = pairs.build_type_a_triples(by_ing, train_ings, atc_map)
    elif args.variant == "b_only":
        triples = pairs.build_type_b_triples(by_ing, train_ings, queries_map, atc_map)
    else:
        triples = pairs.build_combined_dataset(by_ing, train_ings, queries_map, atc_map)

    if not triples:
        logger.warning("Variant %s produced 0 triples; skipping.", args.variant)
        return

    train(triples, output_dir=Path(args.output_dir) / args.variant,
          base_model=args.base_model, epochs=args.epochs,
          batch_size=args.batch_size)


if __name__ == "__main__":
    main()
