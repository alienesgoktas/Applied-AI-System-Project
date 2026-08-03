from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    score_song,
    recommend_songs,
    ScoringStrategy,
    BALANCED,
    ENERGY_FIRST,
    MOOD_BLIND,
    GENRE_PURIST,
)

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_recommend_respects_k():
    user = UserProfile("pop", "happy", 0.8, False)
    assert len(make_small_recommender().recommend(user, k=1)) == 1


def test_opposite_profile_flips_the_ranking():
    """A chill/lofi user should get the lofi track, not the pop one."""
    user = UserProfile("lofi", "chill", 0.4, True)
    assert make_small_recommender().recommend(user, k=1)[0].genre == "lofi"


def test_energy_is_scored_by_closeness_not_magnitude():
    """A low-energy user must not be handed the highest-energy song."""
    calm = {"favorite_genre": "", "favorite_mood": "", "target_energy": 0.3}
    quiet = {"genre": "x", "mood": "y", "energy": 0.3, "valence": 0.5, "acousticness": 0.5}
    loud = {"genre": "x", "mood": "y", "energy": 0.9, "valence": 0.5, "acousticness": 0.5}
    assert score_song(calm, quiet)[0] > score_song(calm, loud)[0]


def test_score_song_returns_reasons_with_points():
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7, "acousticness": 0.2}
    score, reasons = score_song(
        {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}, song
    )
    assert score > 0
    assert any("genre match" in r and "+2.0" in r for r in reasons)
    assert any("mood match" in r for r in reasons)


def test_exact_genre_beats_partial_beats_none():
    prefs = {"favorite_genre": "pop", "favorite_mood": "", "target_energy": 0.5}
    base = {"mood": "y", "energy": 0.5, "valence": 0.5, "acousticness": 0.5}
    exact = score_song(prefs, {**base, "genre": "pop"})[0]
    partial = score_song(prefs, {**base, "genre": "indie pop"})[0]
    none = score_song(prefs, {**base, "genre": "metal"})[0]
    assert exact > partial > none


def test_load_songs_parses_numbers():
    songs = load_songs("data/songs.csv")
    assert len(songs) == 70
    assert isinstance(songs[0]["id"], int)
    assert isinstance(songs[0]["energy"], float)


def test_recommend_songs_sorts_descending_and_explains():
    songs = load_songs("data/songs.csv")
    results = recommend_songs(
        {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4}, songs, k=3
    )

    assert len(results) == 3
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0]["genre"] == "lofi"
    assert results[0][2].strip() != ""


def test_default_strategy_is_balanced():
    """Callers that pass no strategy get the documented Algorithm Recipe."""
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7, "acousticness": 0.2}
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
    assert score_song(prefs, song) == score_song(prefs, song, BALANCED)


def test_mood_blind_strategy_drops_the_mood_points():
    """Swapping the strategy changes the score without touching score_song."""
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7, "acousticness": 0.2}
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}

    balanced, reasons = score_song(prefs, song, BALANCED)
    blind, blind_reasons = score_song(prefs, song, MOOD_BLIND)

    assert round(balanced - blind, 2) == BALANCED.mood
    assert any("mood match" in r for r in reasons)
    assert not any("mood match" in r for r in blind_reasons)


def test_strategies_can_produce_different_rankings():
    """Reweighting the strategy changes the ranking (re-derived for the 70-song catalog).

    A folk fan who wants high energy exposes the genre-vs-energy tension: the
    genre-weighted BALANCED keeps all-folk picks, while ENERGY_FIRST swaps a
    low-energy folk track for a loud non-folk one.
    """
    songs = load_songs("data/songs.csv")
    prefs = {"favorite_genre": "folk", "favorite_mood": "melancholy", "target_energy": 0.9}

    balanced = [s["title"] for s, _, _ in recommend_songs(prefs, songs, 3, BALANCED)]
    energy_first = [s["title"] for s, _, _ in recommend_songs(prefs, songs, 3, ENERGY_FIRST)]

    assert balanced != energy_first
    assert "Paper Boats" in balanced          # genre-weighted keeps a low-energy folk pick
    assert "Overdrive" in energy_first        # energy-weighted pulls in a high-energy track


def test_genre_purist_strategy_favors_genre_over_mood():
    songs = load_songs("data/songs.csv")
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
    top2 = [s["genre"] for s, _, _ in recommend_songs(prefs, songs, 2, GENRE_PURIST)]
    assert top2 == ["pop", "pop"]


def test_recommender_accepts_a_strategy():
    user = UserProfile("pop", "happy", 0.8, False)
    rec = Recommender(make_small_recommender().songs, strategy=MOOD_BLIND)
    assert rec.strategy is MOOD_BLIND
    assert rec.score(user, rec.songs[0])[0] < Recommender(rec.songs).score(user, rec.songs[0])[0]


def test_strategy_is_immutable():
    """Frozen dataclass, so an experiment cannot leak into other callers."""
    import pytest
    with pytest.raises(Exception):
        BALANCED.genre = 99.0


def test_custom_strategy_can_be_defined_inline():
    custom = ScoringStrategy("valence-only", genre=0.0, genre_partial=0.0,
                             mood=0.0, energy=0.0, acoustic=0.0, valence=1.0)
    song = {"genre": "pop", "mood": "happy", "energy": 0.9, "valence": 0.5, "acousticness": 0.2}
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy",
             "target_energy": 0.1, "target_valence": 0.5}
    score, reasons = score_song(prefs, song, custom)
    assert score == 1.0
    assert len(reasons) == 1


def test_recommend_songs_does_not_reorder_the_caller_catalog():
    """sorted() is used rather than .sort(), so the input list is untouched."""
    songs = load_songs("data/songs.csv")
    before = [s["id"] for s in songs]
    recommend_songs({"favorite_genre": "metal", "favorite_mood": "intense",
                     "target_energy": 0.9}, songs, k=5)
    assert [s["id"] for s in songs] == before
