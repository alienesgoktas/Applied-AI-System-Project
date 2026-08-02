import sys
import types
from types import SimpleNamespace

import pytest

from src.backends import (
    AnthropicBackend,
    LocalServerBackend,
    _extract_json,
    select_backend,
)


# --- _extract_json: pull a JSON object out of a (possibly prose-wrapped) blob ---

def test_extract_json_bare_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_prose_wrapped():
    assert _extract_json('Sure! Here it is: {"a": 1, "b": 2} hope that helps') == {"a": 1, "b": 2}


def test_extract_json_code_fenced():
    assert _extract_json('```json\n{"x": true}\n```') == {"x": True}


def test_extract_json_ignores_braces_inside_strings():
    assert _extract_json('{"why": "a } b", "n": 2}') == {"why": "a } b", "n": 2}


def test_extract_json_no_object_raises():
    with pytest.raises(ValueError):
        _extract_json("there is no json here")


def test_extract_json_unbalanced_raises():
    with pytest.raises(ValueError):
        _extract_json('{"a": 1')


# --- AnthropicBackend: verify the request shape against a mocked SDK client ---

def test_anthropic_backend_request_shape(monkeypatch):
    calls = {}

    def fake_create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"summary": "ok", "picks": []}')]
        )

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = lambda api_key=None: fake_client
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    schema = {"type": "object"}
    result = AnthropicBackend("claude-haiku-4-5", "sk-test").complete_json("SYS", "USR", schema)

    assert result == {"summary": "ok", "picks": []}
    assert calls["model"] == "claude-haiku-4-5"
    assert calls["max_tokens"] == 1024
    assert calls["system"] == "SYS"
    assert calls["messages"] == [{"role": "user", "content": "USR"}]
    assert calls["output_config"] == {"format": {"type": "json_schema", "schema": schema}}


# --- select_backend env routing ---

def test_select_backend_off_returns_none(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "off")
    assert select_backend() is None


def test_select_backend_defaults_to_local(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setenv("LOCAL_LLM_MODEL", "gemma4-12b-says-v2")
    backend = select_backend()
    assert isinstance(backend, LocalServerBackend)
    assert backend.model == "gemma4-12b-says-v2"


def test_select_backend_anthropic_without_key_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert select_backend() is None
