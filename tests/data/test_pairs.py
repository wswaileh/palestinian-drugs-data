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
