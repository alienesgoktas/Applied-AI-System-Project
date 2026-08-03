from src.llm import _keyword_profile, refine_profile
from src.pipeline import recommend_for_profile, recommend_from_query
from src.recommender import load_songs


def _songs():
    return load_songs("data/songs.csv")


def _prof(**kw):
    base = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8,
            "target_valence": 0.7, "likes_acoustic": False, "blocked_genres": []}
    base.update(kw)
    return base


# --- recommend_for_profile parity ---

def test_recommend_for_profile_parity_with_from_query():
    songs = _songs()
    q = "chill acoustic lofi to study"
    a = recommend_from_query(q, songs, k=3, backend=None)
    b = recommend_for_profile(q, _keyword_profile(q, songs), songs, k=3, backend=None,
                              profile_source="offline")
    assert a["results"] == b["results"]
    assert a["confidence"] == b["confidence"]
    assert a["breakdowns"] == b["breakdowns"]
    assert a["explanation"] == b["explanation"]


# --- offline refinement deltas ---

def test_refine_calmer_lowers_energy():
    new = refine_profile(_prof(target_energy=0.8), "make it calmer", _songs(), backend=None)
    assert new["target_energy"] == 0.6


def test_refine_louder_raises_energy():
    new = refine_profile(_prof(target_energy=0.5), "louder and harder", _songs(), backend=None)
    assert new["target_energy"] == 0.7


def test_refine_no_genre_blocks_and_clears_favorite():
    new = refine_profile(_prof(favorite_genre="pop"), "no pop please", _songs(), backend=None)
    assert "pop" in new["blocked_genres"]
    assert new["favorite_genre"] != "pop"


def test_refine_more_genre_switches_favorite():
    new = refine_profile(_prof(favorite_genre="pop"), "more jazz", _songs(), backend=None)
    assert new["favorite_genre"] == "jazz"


def test_refine_more_like_n_adopts_that_song():
    songs = _songs()
    res = recommend_from_query("upbeat happy pop", songs, k=3, backend=None)
    target = res["results"][1][0]  # song #2
    new = refine_profile(res["profile"], "more like #2", songs, backend=None,
                         last_results=res["results"])
    assert new["favorite_genre"] == target["genre"].lower()
    assert new["target_energy"] == max(0.0, min(1.0, float(target["energy"])))


def test_refine_more_like_n_beats_backend(fake_backend):
    # A pure "#N" reference is deterministic even with a live backend: the 👍 button must
    # adopt song N's features, never whatever the LLM would return (regression: Warning 1).
    songs = _songs()
    res = recommend_from_query("upbeat happy pop", songs, k=3, backend=None)
    target = res["results"][1][0]
    payload = {"favorite_genre": "jazz", "favorite_mood": "x", "target_energy": 0.1,
               "target_valence": 0.1, "likes_acoustic": True, "blocked_genres": []}
    new = refine_profile(res["profile"], "more like #2", songs,
                         backend=fake_backend(payload=payload), last_results=res["results"])
    assert new["favorite_genre"] == target["genre"].lower()  # song #2, not the LLM's "jazz"


def test_refine_dislike_is_not_a_reference():
    # "dislike #1" must NOT be read as "like #1" (word-boundary guard).
    songs = _songs()
    res = recommend_from_query("upbeat happy pop", songs, k=3, backend=None)
    before = res["profile"]["favorite_genre"]
    new = refine_profile(res["profile"], "dislike #1", songs, backend=None,
                         last_results=res["results"])
    assert new["favorite_genre"] == before  # unchanged; not adopted from #1


def test_refine_bad_index_is_noop():
    songs = _songs()
    prof = _prof()
    new = refine_profile(prof, "more like #99", songs, backend=None, last_results=[])
    assert new["favorite_genre"] == prof["favorite_genre"]


# --- LLM path + fallback ---

def test_refine_uses_llm_backend(fake_backend):
    payload = {"favorite_genre": "jazz", "favorite_mood": "relaxed", "target_energy": 0.3,
               "target_valence": 0.4, "likes_acoustic": True, "blocked_genres": []}
    new = refine_profile(_prof(), "something jazzy and calm", _songs(),
                         backend=fake_backend(payload=payload))
    assert new["favorite_genre"] == "jazz"
    assert new["target_energy"] == 0.3


def test_refine_falls_back_offline_on_backend_error(fake_backend):
    new = refine_profile(_prof(target_energy=0.8), "calmer", _songs(),
                         backend=fake_backend(error=RuntimeError("x")))
    assert new["target_energy"] == 0.6
