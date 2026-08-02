from src.confidence import score_confidence
from src.recommender import BALANCED, GENRE_PURIST


def _scored(*scores):
    """Build a fake ranked list of (song, score, reasons) tuples."""
    return [({"title": f"S{i}"}, sc, "reason") for i, sc in enumerate(scores)]


def test_confidence_high_for_strong_top():
    conf = score_confidence(_scored(6.2, 5.0, 4.0))
    assert conf["confidence"] == round(6.2 / 6.5, 2)


def test_confidence_capped_at_one():
    # A score above the theoretical max must not exceed 1.0.
    assert score_confidence(_scored(9.0))["confidence"] == 1.0


def test_confidence_empty_no_crash():
    conf = score_confidence([])
    assert conf["confidence"] == 0.0
    assert conf["strong_matches"] == 0
    assert conf["note"] == "No matches found."


def test_strong_match_count_and_note():
    # BALANCED strong_min = genre(2.0) + mood(1.5) = 3.5
    conf = score_confidence(_scored(6.2, 4.0, 2.0))
    assert conf["strong_matches"] == 2
    assert "2 strong matches" in conf["note"]


def test_only_one_strong_match_fires_honesty_note():
    # The score-cliff case from model_card.md (6.41 then 2.71).
    conf = score_confidence(_scored(6.41, 2.71, 2.5))
    assert conf["strong_matches"] == 1
    assert "Only 1 strong match" in conf["note"]


def test_confidence_is_strategy_aware_and_does_not_saturate():
    # Denominator is the strategy's own max_score() (single source), so a
    # non-BALANCED strategy (GENRE_PURIST max > 6.5) must NOT peg at 1.0.
    scored = _scored(6.5)
    assert score_confidence(scored, BALANCED)["confidence"] == 1.0
    gp = score_confidence(scored, GENRE_PURIST)["confidence"]
    assert gp < 1.0
    assert gp == round(6.5 / GENRE_PURIST.max_score(), 2)
