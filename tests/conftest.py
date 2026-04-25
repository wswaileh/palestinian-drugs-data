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
