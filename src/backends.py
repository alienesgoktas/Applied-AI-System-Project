"""Provider-neutral LLM backends for the recommender's RAG layer.

Three ways to obtain an LLM completion, selected by the ``LLM_BACKEND`` env var:

  - ``"local"``     -> :class:`LocalServerBackend` — the user's local server,
                       a custom ``POST /api/v1/chat`` API (the default).
  - ``"anthropic"`` -> :class:`AnthropicBackend` — Claude via the official SDK,
                       bring-your-own-key.
  - ``"off"``/none  -> ``None`` — callers fall back to the deterministic offline
                       path, so the app runs free with no LLM at all.

Every backend exposes one method, ``complete_json(system, user, schema) -> dict``,
and RAISES on any transport/parse failure so callers can degrade to the offline
path. Third-party imports are lazy (``anthropic`` only; ``urllib`` is stdlib), so
this module imports — and the whole test suite runs — with no SDK installed and
no server reachable.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Protocol
from urllib import request as _urlrequest

DEFAULT_LOCAL_BASE_URL = "http://localhost:1234"
DEFAULT_LOCAL_MODEL = "gemma4-12b-says-v2"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"

# Appended to the system prompt so free-text models emit a bare JSON object.
_JSON_ONLY = (
    "\n\nReturn ONLY a single valid JSON object and nothing else — "
    "no prose, no markdown, and no code fences."
)


class Backend(Protocol):
    """Minimal contract the pipeline depends on."""

    name: str

    def complete_json(self, system: str, user: str, schema: Dict) -> Dict:
        ...


def _extract_json(text: str) -> Dict:
    """Pull the first balanced ``{...}`` object out of a text blob and parse it.

    Local models return prose, so we cannot assume the whole response is JSON.
    Raises ``ValueError`` if no parseable object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in response")


class LocalServerBackend:
    """The user's local server: ``POST {base_url}/api/v1/chat``.

    Request body:  ``{"model", "system_prompt", "input"}``
    Response body: ``{"output": [{"type": "message", "content": "..."}], ...}``
    (contract verified live against ``gemma4-12b-says-v2``).
    """

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"local:{model}"

    def complete_json(self, system: str, user: str, schema: Dict) -> Dict:
        payload = json.dumps(
            {
                "model": self.model,
                "system_prompt": system + _JSON_ONLY,
                "input": user,
            }
        ).encode("utf-8")
        req = _urlrequest.Request(
            f"{self.base_url}/api/v1/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urlrequest.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = "".join(
            item.get("content", "")
            for item in body.get("output", [])
            if item.get("type") == "message"
        )
        return _extract_json(text)


class AnthropicBackend:
    """Claude via the official SDK (optional bring-your-own-key alternate)."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.name = f"anthropic:{model}"

    def complete_json(self, system: str, user: str, schema: Dict) -> Dict:
        import anthropic  # lazy: only imported on this path

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return _extract_json(text)


def select_backend() -> Optional[Backend]:
    """Choose a backend from the environment, or ``None`` (→ offline path).

    ``LLM_BACKEND``: ``"local"`` (default) | ``"anthropic"`` | ``"off"``.
    A chosen backend still returns ``None`` when its required config is missing,
    so a misconfigured environment degrades gracefully to offline rather than
    crashing.
    """
    choice = os.getenv("LLM_BACKEND", "local").strip().lower()
    if choice in ("off", "none", "offline"):
        return None
    if choice == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        model = os.getenv("RECOMMENDER_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return AnthropicBackend(model, key)
    # default: local
    base_url = os.getenv("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL)
    model = os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL)
    if not base_url or not model:
        return None
    return LocalServerBackend(base_url, model)
