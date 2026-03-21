from bespoke.teach.llm_client import _parse_json_response
import pytest


def test_parse_dict_response():
    text = '{"domain": "code", "quality_score": "high"}'
    result = _parse_json_response(text)
    assert isinstance(result, dict)
    assert result["domain"] == "code"


def test_parse_array_response():
    text = '[{"turn_number": 1, "domain": "code"}, {"turn_number": 3, "domain": "planning"}]'
    result = _parse_json_response(text)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["turn_number"] == 1


def test_parse_strips_markdown_fences():
    text = '```json\n[{"turn_number": 1}]\n```'
    result = _parse_json_response(text)
    assert isinstance(result, list)


def test_parse_none_raises():
    with pytest.raises(ValueError, match="LLM returned None"):
        _parse_json_response(None)


def test_parse_empty_raises():
    with pytest.raises(ValueError, match="LLM returned empty"):
        _parse_json_response("")


def test_parse_non_json_raises():
    with pytest.raises(ValueError, match="LLM returned non-JSON"):
        _parse_json_response("<thinking>some xml</thinking>")
