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
