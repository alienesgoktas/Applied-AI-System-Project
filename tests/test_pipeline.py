from src.pipeline import recommend_from_query
from src.recommender import BALANCED, load_songs, recommend_songs


class RoutingBackend:
    """Fake backend that returns a profile or an explanation by schema shape."""

    name = "fake"

    def __init__(self, profile=None, explain=None, error=None):
        self._profile = profile
        self._explain = explain
        self._error = error

    def complete_json(self, system, user, schema):
        if self._error is not None:
            raise self._error
        if "picks" in schema.get("properties", {}):
            return self._explain
        return self._profile


def _songs():
    return load_songs("data/songs.csv")


def test_pipeline_offline_end_to_end():
    res = recommend_from_query("chill acoustic lofi to study", _songs(), k=3, backend=None)
    assert res["profile_source"] == "offline"
    assert res["used_llm"] is False
    assert res["backend"] == "offline"
    assert len(res["results"]) == 3
    assert 0.0 <= res["confidence"]["confidence"] <= 1.0


def test_pipeline_results_always_carry_deterministic_reasons():
    # The authoritative record is always present, independent of any LLM prose.
    res = recommend_from_query("upbeat happy pop", _songs(), k=3, backend=None)
    for _song, _score, reasons in res["results"]:
        assert isinstance(reasons, str) and reasons.strip()


def test_pipeline_guardrail_rejects_hallucinated_title():
    songs = _songs()
    profile = {"favorite_genre": "pop", "favorite_mood": "happy",
               "target_energy": 0.8, "target_valence": 0.7, "likes_acoustic": False}
    explain = {"summary": "x", "picks": [{"title": "Nonexistent Track 999", "why": "fake"}]}
    backend = RoutingBackend(profile=profile, explain=explain)
    res = recommend_from_query("upbeat happy pop", songs, k=3, backend=backend)
    assert res["used_llm"] is False
    assert "Nonexistent Track 999" not in res["explanation"]


def test_pipeline_uses_valid_llm_explanation():
    songs = _songs()
    profile = {"favorite_genre": "pop", "favorite_mood": "happy",
               "target_energy": 0.8, "target_valence": 0.7, "likes_acoustic": False}
    top_title = recommend_songs(profile, songs, k=3, strategy=BALANCED)[0][0]["title"]
    explain = {"summary": "Here you go.", "picks": [{"title": top_title, "why": "matches your vibe"}]}
    backend = RoutingBackend(profile=profile, explain=explain)
    res = recommend_from_query("upbeat happy pop", songs, k=3, backend=backend)
    assert res["used_llm"] is True
    assert top_title in res["explanation"]
    assert "Here you go." in res["explanation"]


def test_pipeline_empty_and_garbage_query_no_crash():
    songs = _songs()
    empty = recommend_from_query("", songs, k=3, backend=None)
    assert len(empty["results"]) == 3
    garbage = recommend_from_query("!!!###", songs, k=3, backend=None)
    assert len(garbage["results"]) == 3


def test_pipeline_empty_catalog_no_crash():
    res = recommend_from_query("pop", [], k=3, backend=None)
    assert res["results"] == []
    assert res["confidence"]["confidence"] == 0.0
    assert "No matches" in res["confidence"]["note"]
