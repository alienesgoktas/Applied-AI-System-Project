from src.pipeline import recommend_from_query
from src.recommender import BALANCED, load_songs, score_detail, score_song


def _songs():
    return load_songs("data/songs.csv")


_SONG = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.7, "acousticness": 0.2}
_PREFS = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8,
          "target_valence": 0.7, "likes_acoustic": False}


def test_score_detail_points_sum_to_score():
    score, terms = score_detail(_PREFS, _SONG, BALANCED)
    assert round(sum(p for _label, p in terms), 2) == score


def test_score_detail_matches_score_song():
    s1, reasons = score_song(_PREFS, _SONG, BALANCED)
    s2, terms = score_detail(_PREFS, _SONG, BALANCED)
    assert s1 == s2
    assert len(terms) == len(reasons)  # one term per reason (shared _score_terms)


def test_score_detail_labels_in_order():
    _score, terms = score_detail(_PREFS, _SONG, BALANCED)
    assert [label for label, _ in terms] == ["genre", "mood", "energy", "valence", "acoustic"]


def test_score_detail_includes_dislike_penalty():
    prefs = {**_PREFS, "blocked_genres": ["pop"]}
    _score, terms = score_detail(prefs, _SONG, BALANCED)
    labels = dict(terms)
    assert "blocked" in labels and labels["blocked"] < 0


def test_score_song_uses_naive_fold_not_compensated_sum():
    # Pins a real multi-term score to its exact value. "Storm Runner" under this
    # profile folds to 1.27; Python 3.12+ built-in sum() (compensated) would give
    # 1.26 - so this fails if the naive left-fold is ever replaced by sum().
    storm = next(s for s in _songs() if s["title"] == "Storm Runner")
    prof = {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.37,
            "target_valence": 0.53, "likes_acoustic": True}
    assert score_song(prof, storm, BALANCED)[0] == 1.27


def test_pipeline_includes_breakdowns_summing_to_scores():
    res = recommend_from_query("upbeat happy pop", _songs(), k=3, backend=None)
    assert "breakdowns" in res
    assert len(res["breakdowns"]) == len(res["results"]) == 3
    for (_song, score, _reasons), bd in zip(res["results"], res["breakdowns"]):
        assert round(sum(p for _label, p in bd), 2) == score
