import pytest


class FakeBackend:
    """A configurable fake LLM backend for tests — no network, no SDK.

    Modes:
      - FakeBackend(payload=X)          -> return X for every complete_json call.
      - FakeBackend(error=E)            -> raise E (transport/parse failure).
      - FakeBackend(profile=P, explain=Q) -> route by schema: the explanation call
        (schema has a 'picks' array) returns Q; any other call returns P.
    """

    name = "fake"

    def __init__(self, payload=None, profile=None, explain=None, error=None):
        self._payload = payload
        self._profile = profile
        self._explain = explain
        self._error = error

    def complete_json(self, system, user, schema):
        if self._error is not None:
            raise self._error
        if self._profile is not None or self._explain is not None:
            if "picks" in schema.get("properties", {}):
                return self._explain
            return self._profile
        return self._payload


@pytest.fixture
def fake_backend():
    """Factory fixture: call ``fake_backend(...)`` to build a FakeBackend."""
    return FakeBackend
