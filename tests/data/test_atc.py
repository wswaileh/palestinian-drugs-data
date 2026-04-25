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
    monkeypatch.setattr(atc, "_get",
                        lambda *a, **kw: pytest.fail("should not hit network"))
    mapping = atc.build_mapping(["ibuprofen"], cache_path=cache_path)
    assert mapping == {"ibuprofen": ["M01AE01"]}


def test_atc_level3():
    assert atc.atc_level3("M01AE01") == "M01AE"
    assert atc.atc_level3("A10BA02") == "A10BA"
    assert atc.atc_level3("") == ""
