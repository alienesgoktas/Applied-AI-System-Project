"""
Command line runner for the Music Recommender Simulation.

Loads the catalog and prints the top-k recommendations for a taste profile,
each with a plain-language explanation of why it was chosen.
"""

from src.recommender import load_songs, recommend_songs

CATALOG = "data/songs.csv"


def print_recommendations(user_prefs: dict, songs: list, k: int = 5) -> None:
    """Prints the top k recommendations for one profile in a readable block."""
    print("=" * 68)
    print(f"  Profile: {user_prefs['favorite_genre']} / {user_prefs['favorite_mood']}"
          f" | energy {user_prefs['target_energy']:.2f}"
          f" | valence {user_prefs['target_valence']:.2f}"
          f" | {'acoustic' if user_prefs['likes_acoustic'] else 'produced'}")
    print("=" * 68)

    for rank, (song, score, explanation) in enumerate(recommend_songs(user_prefs, songs, k), 1):
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Score: {score:.2f}   [{song['genre']} / {song['mood']}]")
        for reason in explanation.split("; "):
            print(f"     - {reason}")
    print()


def main() -> None:
    songs = load_songs(CATALOG)
    print(f"Loaded songs: {len(songs)}\n")

    # Starter example profile
    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "target_valence": 0.7,
        "likes_acoustic": False,
    }

    print_recommendations(user_prefs, songs, k=5)


if __name__ == "__main__":
    main()
