"""Runtime agentic workflow: a confidence-driven relaxation chain over the recommender.

Unlike the single-shot RAG pipeline, this AGENT reasons across multiple steps. It uses the
existing pieces as tools -- parse_profile (understand), recommend_songs (retrieve), and
score_confidence (evaluate) -- and makes a decision each step: if the results are strong,
ACCEPT and stop; otherwise PLAN the next relaxation from a fixed ladder, apply it, and RETRY.
It keeps the best-confidence result it has seen and records every decision in a trace.

The decision logic is rule-based (over strong_matches / confidence) and profile parsing
degrades to the offline keyword parser, so the whole agent runs -- and is tested -- with no
LLM, key, or server.

Usage:
    python -m src.agent "upbeat pop, no pop"
    LLM_BACKEND=local python -m src.agent "melancholy bluegrass"
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

from src.confidence import score_confidence
from src.llm import parse_profile
from src.recommender import BALANCED, MOOD_BLIND, ScoringStrategy, recommend_songs

ACCEPT_STRONG = 2  # strong_matches >= this -> the agent accepts and stops


def _snapshot(step: int, action: str, reason: str, results, conf: Dict) -> Dict:
    return {
        "step": step,
        "action": action,
        "reason": reason,
        "confidence": conf["confidence"],
        "strong_matches": conf["strong_matches"],
        "top_pick": results[0][0]["title"] if results else None,
    }


# The relaxation ladder: each rung returns the next (profile, strategy) or None if it does
# not apply to the current state. Rungs are self-disabling (dropping blocks empties the list,
# so the same rung returns None next time), so iterating from the top each step advances.
def _drop_blocks(profile: Dict, strategy: ScoringStrategy):
    if profile.get("blocked_genres"):
        return {**profile, "blocked_genres": []}, strategy
    return None


def _ignore_mood(profile: Dict, strategy: ScoringStrategy):
    if profile.get("favorite_mood"):
        return {**profile, "favorite_mood": ""}, strategy
    return None


def _widen_strategy(profile: Dict, strategy: ScoringStrategy):
    if strategy is not MOOD_BLIND:
        return profile, MOOD_BLIND
    return None


_LADDER = [
    ("drop blocked genres", _drop_blocks),
    ("ignore mood", _ignore_mood),
    ("widen strategy (mood-blind)", _widen_strategy),
]


def agentic_recommend(query: str, songs: List[Dict], *, backend=None,
                      strategy: ScoringStrategy = BALANCED,
                      max_steps: int = 3) -> Tuple[Dict, List[Dict]]:
    """Multi-step: retrieve -> evaluate -> (accept | relax & retry). Returns (result, trace).

    ``result`` is the best-confidence pass seen: ``{profile, results, confidence, strategy,
    action}``. ``trace`` is one dict per step (step, action, reason, confidence,
    strong_matches, top_pick).
    """
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    profile, _src = parse_profile(query, songs, backend=backend)
    cur_profile, cur_strategy = profile, strategy
    action, reason = "initial retrieval", "first pass from the parsed profile"

    trace: List[Dict] = []
    best: Optional[Tuple[float, Dict]] = None

    for step in range(1, max_steps + 1):
        results = recommend_songs(cur_profile, songs, k=5, strategy=cur_strategy)
        conf = score_confidence(results, cur_strategy)
        trace.append(_snapshot(step, action, reason, results, conf))

        result = {"profile": cur_profile, "results": results, "confidence": conf,
                  "strategy": cur_strategy.name, "action": action}
        if best is None or conf["confidence"] > best[0]:
            best = (conf["confidence"], result)

        if conf["strong_matches"] >= ACCEPT_STRONG:
            break  # accepted: strong enough, stop reasoning

        nxt = None
        for name, rung in _LADDER:
            out = rung(cur_profile, cur_strategy)
            if out is not None:
                nxt = (name, out)
                break
        if nxt is None:
            break  # no relaxation left; keep the best result seen

        action, (cur_profile, cur_strategy) = nxt[0], nxt[1]
        reason = (f"weak (strong={conf['strong_matches']}, conf={conf['confidence']:.2f}); "
                  f"relaxing: {action}")

    assert best is not None  # loop runs at least once (max_steps >= 1)
    return best[1], trace


def _print_run(query: str, backend_name: str, result: Dict, trace: List[Dict]) -> None:
    print(f"Agent query: {query!r}   (backend {backend_name})\n")
    print("Reasoning trace:")
    for t in trace:
        print(f"  step {t['step']}: {t['action']}")
        print(f"          reason: {t['reason']}")
        print(f"          -> confidence {t['confidence']:.2f}, "
              f"strong {t['strong_matches']}, top {t['top_pick']!r}")
    conf = result["confidence"]
    print(f"\nChosen pass: strategy={result['strategy']}, "
          f"confidence={conf['confidence']:.2f} - {conf['note']}")
    for rank, (song, score, _reasons) in enumerate(result["results"], 1):
        print(f"  {rank}. {song['title']} - {song['artist']} "
              f"[{song['genre']}/{song['mood']}] ({score:.2f})")


def main() -> int:
    import logging
    import os

    from src.backends import select_backend
    from src.pipeline import get_logger
    from src.recommender import load_songs, strategy_from_name

    get_logger().setLevel(logging.WARNING)  # keep the console = the trace

    query = " ".join(sys.argv[1:]).strip() or "upbeat pop, no pop"
    songs = load_songs("data/songs.csv")
    backend = select_backend()
    strategy = strategy_from_name(os.getenv("RECOMMENDER_STRATEGY"))
    result, trace = agentic_recommend(query, songs, backend=backend, strategy=strategy)
    _print_run(query, getattr(backend, "name", None) or "offline", result, trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
