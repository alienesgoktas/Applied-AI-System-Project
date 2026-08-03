from src.confidence import score_confidence
from src.llm import _keyword_profile
from src.pipeline import recommend_from_query
from src.recommender import (
    BALANCED,
    ScoringStrategy,
    load_songs,
    recommend_songs,
    score_song,
    strategy_from_name,
)


def _songs():
    return load_songs("data/songs.csv")


def _mk(sid, artist, energy=0.5):
    return {"id": sid, "title": f"S{sid}", "artist": artist, "genre": "pop", "mood": "happy",
            "energy": energy, "tempo_bpm": 120, "valence": 0.5, "danceability": 0.5, "acousticness": 0.3}


# --- strategy resolver (CLI env / UI presets) ---

def test_strategy_from_name_presets_case_insensitive():
    assert strategy_from_name("mood-blind").name == "mood-blind"
    assert strategy_from_name("GENRE-PURIST").name == "genre-purist"
    assert strategy_from_name("balanced") is BALANCED


def test_strategy_from_name_unknown_falls_back_to_balanced():
    assert strategy_from_name("nonsense") is BALANCED
    assert strategy_from_name(None) is BALANCED
    assert strategy_from_name("") is BALANCED


# --- dislikes / block-list ---

def test_blocked_genre_applies_penalty():
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7, "acousticness": 0.2}
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
    base, _ = score_song(prefs, song)
    blocked, reasons = score_song({**prefs, "blocked_genres": ["pop"]}, song)
    assert round(base - blocked, 2) == BALANCED.dislike
    assert any("blocked genre" in r for r in reasons)


def test_absent_or_empty_blocked_key_is_noop():
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7, "acousticness": 0.2}
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
    assert score_song(prefs, song) == score_song({**prefs, "blocked_genres": []}, song)


def test_unknown_blocked_genre_is_noop():
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7, "acousticness": 0.2}
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
    assert score_song(prefs, song) == score_song({**prefs, "blocked_genres": ["metal"]}, song)


def test_pipeline_blocked_genre_demotes_it():
    res = recommend_from_query("upbeat happy pop", _songs(), k=5, backend=None, blocked_genres=["pop"])
    assert "pop" in res["profile"]["blocked_genres"]
    assert res["results"][0][0]["genre"] != "pop"


def test_offline_parser_reads_no_genre_and_wont_favorite_it():
    prof = _keyword_profile("no pop please, something calm", _songs())
    assert "pop" in prof["blocked_genres"]
    assert prof["favorite_genre"] != "pop"


def test_offline_parser_ignores_substring_false_positive():
    # "piano house" must NOT read as "no house" (word-level, not substring).
    prof = _keyword_profile("piano house vibes", _songs())
    assert prof["blocked_genres"] == []


def test_offline_parser_blocks_multiword_genre():
    # "hip hop" is a two-word catalog genre and must still be blockable.
    prof = _keyword_profile("no hip hop please", _songs())
    assert "hip hop" in prof["blocked_genres"]


def test_confidence_stays_in_contract_when_everything_blocked():
    # Dislike penalties can drive the top score negative; confidence must not go < 0.
    songs = _songs()
    all_genres = sorted({s["genre"] for s in songs})
    res = recommend_from_query("anything", songs, k=3, backend=None, blocked_genres=all_genres)
    assert 0.0 <= res["confidence"]["confidence"] <= 1.0


# --- diversity cap + backfill ---

def test_diversity_cap_then_backfill_to_k():
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.9}
    songs = [_mk(1, "A", 0.95), _mk(2, "A", 0.90), _mk(3, "A", 0.85), _mk(4, "B", 0.80), _mk(5, "B", 0.75)]
    res = recommend_songs(prefs, songs, k=3, max_per_artist=1)
    artists = [s["artist"] for s, _, _ in res]
    assert len(res) == 3
    assert artists[0] == "A" and artists[1] == "B"   # cap respected first
    assert artists.count("A") == 2                    # then backfilled to k


def test_diversity_none_equals_plain_top_k():
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.9}
    songs = [_mk(i, f"Art{i}", 0.9 - i * 0.05) for i in range(6)]
    assert recommend_songs(prefs, songs, k=3) == recommend_songs(prefs, songs, k=3, max_per_artist=None)


def test_diversity_large_cap_equals_plain_top_k():
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.9}
    songs = [_mk(i, "A", 0.9 - i * 0.05) for i in range(6)]  # all same artist
    assert recommend_songs(prefs, songs, k=3, max_per_artist=99) == recommend_songs(prefs, songs, k=3)


# --- zero-weight custom strategy edge ---

def test_zero_weight_strategy_confidence_is_zero():
    zero = ScoringStrategy("zero", genre=0, genre_partial=0, mood=0, energy=0, valence=0, acoustic=0)
    scored = recommend_songs({"favorite_genre": "pop"}, _songs(), k=3, strategy=zero)
    assert score_confidence(scored, zero)["confidence"] == 0.0
