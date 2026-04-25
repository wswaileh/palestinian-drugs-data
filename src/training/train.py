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
