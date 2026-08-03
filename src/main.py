"""
Command line runner for the Music Recommender Simulation.

Takes a plain-English request (as command-line args or an interactive prompt),
runs the RAG pipeline, and prints the ranked songs with their deterministic
scoring reasons, a confidence/honesty note, and an AI-written summary. The LLM
backend is chosen from the environment (LLM_BACKEND=local|anthropic|off); with no
backend it runs free in deterministic offline mode.

Usage:
    python -m src.main "chill acoustic music to study to"
    python -m src.main            # prompts interactively
"""

import sys

from src.backends import select_backend
from src.pipeline import backend_label, recommend_from_query
from src.recommender import load_songs

CATALOG = "data/songs.csv"
DEMO_QUERY = "upbeat happy pop for a workout"


def _print_result(result: dict) -> None:
    """Print one pipeline result: header, ranked songs + reasons, AI summary."""
    badge = f"[{backend_label(result)}]"

    p = result["profile"]
    conf = result["confidence"]
    print("=" * 68)
    print(f"  You asked: {result['query']}   {badge}")
    print(
        f"  Understood as: {p['favorite_genre'] or '(any)'} / "
        f"{p['favorite_mood'] or '(any)'} | energy {p['target_energy']:.2f}"
        f" | valence {p['target_valence']:.2f}"
        f" | {'acoustic' if p['likes_acoustic'] else 'produced'}"
    )
    print(f"  Confidence: {conf['confidence']:.2f} - {conf['note']}")
    print("=" * 68)

    for rank, (song, score, reasons) in enumerate(result["results"], 1):
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Score: {score:.2f}   [{song['genre']} / {song['mood']}]")
        for reason in reasons.split("; "):
            print(f"     - {reason}")

    print("\n--- AI summary (advisory; the reasons above are the record) ---")
    print(result["explanation"])
    print()


def main() -> None:
    songs = load_songs(CATALOG)
    print(f"Loaded songs: {len(songs)}\n")

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        try:
            query = input("Describe the music you want (blank for a demo): ").strip()
        except EOFError:
            query = ""
    if not query:
        query = DEMO_QUERY
        print(f"(no request given — using demo: {query!r})\n")

    backend = select_backend()
    result = recommend_from_query(query, songs, k=5, backend=backend)
    _print_result(result)


if __name__ == "__main__":
    main()
