"""Confidence + honesty-threshold scoring for a ranked recommendation list.

Turns the raw scores into (a) a 0-1 confidence in the top result and (b) a count
of "strong" matches, so the caller can warn when a list is padded with weak
filler — the score-cliff problem documented in model_card.md (a folk listener's
scores drop from 6.41 to 2.71 after the single real match). Both thresholds are
DERIVED FROM THE ACTIVE STRATEGY's weights, so they stay correct if the caller
swaps strategies (e.g. GENRE_PURIST, whose maximum exceeds BALANCED's 6.5) rather
than silently pegging confidence at 1.0 against the wrong denominator.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from src.recommender import BALANCED, ScoringStrategy

# A "scored" item is (song_dict, score, explanation), as returned by recommend_songs.
Scored = Sequence[Tuple[Dict, float, str]]


def score_confidence(scored: Scored, strategy: ScoringStrategy = BALANCED) -> Dict:
    """Return ``{confidence, strong_matches, note}`` for a ranked result list.

    - ``max_score``  = the strategy's maximum achievable score
      (``genre + mood + energy + valence + acoustic``; BALANCED -> 6.5).
    - ``strong_min`` = the strategy's categorical maximum, ``genre + mood``
      (BALANCED -> 3.5): a "strong" match cleared both categorical terms.
    - ``confidence`` = ``top_score / max_score``, capped at 1.0.
    - ``note`` fires the honesty warning when at most one match is strong.
    """
    max_score = strategy.max_score()
    strong_min = strategy.categorical_max()

    if not scored:
        return {"confidence": 0.0, "strong_matches": 0, "note": "No matches found."}

    top = scored[0][1]
    confidence = round(min(top / max_score, 1.0), 2) if max_score > 0 else 0.0
    strong_matches = sum(1 for _, score, _ in scored if score >= strong_min)

    if strong_matches == 0:
        note = "No strong matches - results are weak (chosen mostly by energy)."
    elif strong_matches == 1:
        note = (
            "Only 1 strong match found - everything below it is weak "
            "(chosen mostly by energy)."
        )
    else:
        note = f"{strong_matches} strong matches."

    return {
        "confidence": confidence,
        "strong_matches": strong_matches,
        "note": note,
    }
