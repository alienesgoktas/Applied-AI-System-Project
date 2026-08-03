"""Specialization demo (SF-C): the same query + same retrieved songs run through the
BASELINE vs the SPECIALIZED explanation prompt, side by side, with a measurable comparison.

The specialized prompt adds hard grounding, a one-sentence / no-marketing-adjective tone
constraint, and a one-shot exemplar (few-shot specialization). This script needs a live
backend to show a contrast; offline both styles fall back to the SAME deterministic text.

Usage:
    LLM_BACKEND=local python specialization_demo.py "upbeat pop for a workout"
    LLM_BACKEND=anthropic python specialization_demo.py "smooth jazz for a rainy evening"
"""

from __future__ import annotations

import re
import sys
from typing import Dict

from src.backends import select_backend
from src.llm import generate_explanation, parse_profile
from src.recommender import load_genre_notes, load_songs, recommend_songs, retrieve_notes

CATALOG = "data/songs.csv"
NOTES = "data/genre_notes.csv"
# Deliberately broader than the prompt's banned list (amazing/perfect/ultimate/best) so the
# metric also catches marketing tone the prompt didn't explicitly name.
_MARKETING = ("amazing", "perfect", "ultimate", "best", "incredible", "must-listen", "must listen")


def measure(text: str, used_llm: bool) -> Dict:
    """Measurable properties of one explanation: grounding, pick count, verbosity, tone."""
    picks = [ln for ln in text.splitlines() if ln.strip().startswith("- ")]
    words = [len(ln.split()) for ln in picks]
    low = text.lower()
    # Word-boundary match so "best" doesn't count inside "bestseller".
    marketing = sum(len(re.findall(rf"\b{re.escape(w)}\b", low)) for w in _MARKETING)
    return {
        "grounded (guardrail held)": used_llm,
        "picks": len(picks),
        "avg words/pick": round(sum(words) / len(words), 1) if words else 0.0,
        "marketing adjectives": marketing,
    }


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or "upbeat pop for a workout"
    backend = select_backend()
    songs = load_songs(CATALOG)
    notes_src = load_genre_notes(NOTES)

    profile, _src = parse_profile(query, songs, backend=backend)
    results = recommend_songs(profile, songs, k=5)
    notes = retrieve_notes(results, notes_src)

    name = getattr(backend, "name", None) or "offline"
    print(f"Query: {query!r}   backend: {name}\n")
    if backend is None:
        print("(No live backend - both styles fall back to the same deterministic text; "
              "set LLM_BACKEND=local|anthropic to see the A/B contrast.)\n")

    for style in ("baseline", "specialized"):
        text, used = generate_explanation(query, results, backend=backend, notes=notes, style=style)
        print(f"===== {style.upper()} =====")
        for key, val in measure(text, used).items():
            print(f"  {key}: {val}")
        print(text)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
