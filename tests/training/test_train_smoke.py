import os
import pytest
from src.training import train


@pytest.mark.skipif(
    not os.environ.get("RUN_SMOKE_TRAIN"),
    reason="Skips without RUN_SMOKE_TRAIN=1 (downloads model, takes ~10-30s).",
)
def test_one_step_does_not_crash(tmp_path, tiny_chunks, tiny_atc_map):
    """Builds 3 triples and runs 1 epoch with batch_size=2 — smoke only."""
    from src.data import pairs
    by_ing = {"ibuprofen": [c for c in tiny_chunks if c["ingredient"] == "ibuprofen"],
              "naproxen":  [c for c in tiny_chunks if c["ingredient"] == "naproxen"],
              "metformin": [c for c in tiny_chunks if c["ingredient"] == "metformin"]}
    triples = pairs.build_combined_dataset(
        by_ing, ["ibuprofen", "naproxen", "metformin"],
        {"ibuprofen": ["pain"], "naproxen": ["pain"], "metformin": ["dm"]},
        tiny_atc_map, seed=42)
    out_dir = tmp_path / "smoke_model"
    train.train(triples, output_dir=out_dir, epochs=1, batch_size=2,
                model_name="prajjwal1/bert-tiny")
    assert (out_dir / "config.json").exists()
