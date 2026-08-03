"""Evaluation harness — runs predefined queries through the pipeline and prints a
pass/fail + confidence summary.

Offline by default (``backend=None``), so it is fully reproducible with no LLM, key,
or server — the grader can run it and get identical results. Each case asserts only
deterministic, result-derivable properties (top genre, confidence bounds, whether the
honesty note fires, a blocked genre's absence). Exits 0 iff every case passes, else 1.

Reconnects the throwaway "five-profile evaluation harness" from the original build
(see ai_interactions.md, SF8) as a committed, runnable artifact.

Usage:
    python eval.py            # offline (deterministic, default)
    python eval.py --live     # use the configured LLM backend (LLM_BACKEND=...)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.pipeline import recommend_from_query
from src.recommender import load_songs

CATALOG = "data/songs.csv"


@dataclass(frozen=True)
class Expect:
    """Deterministic, result-derivable expectations for one query."""
    top_genre: Optional[str] = None       # genre of the #1 pick (unambiguous cases only)
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    expect_strong: bool = False           # strong_matches >= 2
    expect_weak: bool = False             # strong_matches == 0 (honesty note fires)
    absent_genre: Optional[str] = None    # this genre must NOT appear in the results


@dataclass(frozen=True)
class EvalCase:
    query: str
    expect: Expect


# Calibrated against the deterministic offline output over data/songs.csv.
CASES: List[EvalCase] = [
    EvalCase("upbeat happy pop for a workout",
             Expect(top_genre="pop", min_confidence=0.6, expect_strong=True)),
    EvalCase("chill acoustic lofi to study",
             Expect(top_genre="lofi", min_confidence=0.6, expect_strong=True)),
    EvalCase("aggressive loud metal",
             Expect(top_genre="metal", min_confidence=0.6, expect_strong=True)),
    EvalCase("smooth jazz for a rainy evening",
             Expect(top_genre="jazz", expect_strong=True)),
    EvalCase("energetic house dance music",
             Expect(top_genre="house", min_confidence=0.5, expect_strong=True)),
    # Niche taste — the honesty layer should flag weak results, not fake confidence.
    EvalCase("melancholy bluegrass",
             Expect(top_genre="bluegrass", max_confidence=0.7, expect_weak=True)),
    # Dislike/block — "no pop" must keep pop out of the results entirely.
    EvalCase("no pop, something calm and acoustic",
             Expect(absent_genre="pop", expect_weak=True)),
]


def evaluate_case(case: EvalCase, songs: List[Dict], backend) -> Dict:
    """Run one case and return a row: query, top pick, confidence, and any failures."""
    result = recommend_from_query(case.query, songs, k=5, backend=backend)
    results = result["results"]
    conf = result["confidence"]
    e = case.expect
    failures: List[str] = []

    if not results:
        failures.append("no results returned")
        return {"query": case.query, "top": "-", "genre": "-",
                "confidence": conf["confidence"], "strong": conf["strong_matches"],
                "passed": False, "failures": failures}

    top_song = results[0][0]
    genres = [s["genre"] for s, _score, _r in results]
    c = conf["confidence"]
    strong = conf["strong_matches"]

    if e.top_genre is not None and top_song["genre"] != e.top_genre:
        failures.append(f"top genre {top_song['genre']!r} != {e.top_genre!r}")
    if e.min_confidence is not None and c < e.min_confidence:
        failures.append(f"confidence {c:.2f} < min {e.min_confidence}")
    if e.max_confidence is not None and c > e.max_confidence:
        failures.append(f"confidence {c:.2f} > max {e.max_confidence}")
    if e.expect_strong and strong < 2:
        failures.append(f"expected strong matches (>=2), got {strong}")
    if e.expect_weak and strong != 0:
        failures.append(f"expected weak (0 strong), got {strong}")
    if e.absent_genre is not None and e.absent_genre in genres:
        failures.append(f"blocked genre {e.absent_genre!r} appeared in results")

    return {"query": case.query, "top": top_song["title"], "genre": top_song["genre"],
            "confidence": c, "strong": strong,
            "passed": not failures, "failures": failures}


def run_eval(songs: List[Dict], cases: List[EvalCase], backend=None) -> Dict:
    """Evaluate every case; return ``{total, passed, rows}`` (rows as from evaluate_case)."""
    rows = [evaluate_case(case, songs, backend) for case in cases]
    passed = sum(1 for r in rows if r["passed"])
    return {"total": len(rows), "passed": passed, "rows": rows}


def _print_summary(summary: Dict) -> None:
    print(f"{'#':>2}  {'query':<38} {'top pick':<22} {'genre':<10} {'conf':>5} {'strong':>6}  result")
    print("-" * 100)
    for i, r in enumerate(summary["rows"], 1):
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f"{i:>2}  {r['query'][:37]:<38} {r['top'][:21]:<22} {r['genre']:<10} "
              f"{r['confidence']:>5.2f} {r['strong']:>6}  {verdict}")
        for f in r["failures"]:
            print(f"      ! {f}")
    print("-" * 100)
    print(f"{summary['passed']}/{summary['total']} passed")


def main() -> int:
    import logging

    from src.pipeline import get_logger
    get_logger().setLevel(logging.WARNING)  # configure first, then quiet: console = the table

    live = "--live" in sys.argv[1:]
    backend = None
    if live:
        from src.backends import select_backend
        backend = select_backend()
    songs = load_songs(CATALOG)
    print(f"Music Recommender - evaluation harness "
          f"({'LIVE: ' + getattr(backend, 'name', 'offline') if live else 'offline'}, "
          f"{len(songs)} songs)\n")
    summary = run_eval(songs, CASES, backend)
    _print_summary(summary)
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
