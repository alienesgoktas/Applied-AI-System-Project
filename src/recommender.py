"""Content-based music recommender: scores songs against a user taste profile."""

import csv
from typing import List, Dict, Tuple, Callable, Optional, TypeVar
from dataclasses import dataclass

T = TypeVar("T")

NUMERIC_FIELDS = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")


@dataclass(frozen=True)
class ScoringStrategy:
    """A named set of Algorithm Recipe weights that can be swapped at runtime."""
    name: str = "balanced"
    genre: float = 2.0          # exact genre match
    genre_partial: float = 1.0  # shared word, e.g. "indie pop" for a "pop" fan
    mood: float = 1.5           # exact mood match
    energy: float = 1.5         # scaled by closeness to the user's target
    valence: float = 0.5        # scaled by closeness to the user's target
    acoustic: float = 1.0       # scaled by acousticness, or its inverse


# The recipe documented in README.md, plus the variants used in Phase 4.
BALANCED = ScoringStrategy()
ENERGY_FIRST = ScoringStrategy("energy-first", genre=1.0, genre_partial=0.5, energy=3.0)
MOOD_BLIND = ScoringStrategy("mood-blind", mood=0.0)
GENRE_PURIST = ScoringStrategy("genre-purist", genre=4.0, genre_partial=2.0)


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_valence: float = 0.5   # defaulted so existing callers keep working

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song], strategy: ScoringStrategy = BALANCED):
        self.songs = songs
        self.strategy = strategy

    def score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Scores one song for one user, returning (score, reasons)."""
        return score_song(vars(user), vars(song), self.strategy)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the k best-scoring songs for the user, highest score first."""
        return _top_k(self.songs, lambda s: self.score(user, s)[0], lambda s: s.id, k)

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explains in one sentence why a song was recommended to a user."""
        score, reasons = self.score(user, song)
        return f"Scored {score:.2f} - " + "; ".join(reasons or ["no strong matches"])


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            song = dict(row)
            song["id"] = int(song["id"])
            for field in NUMERIC_FIELDS:
                song[field] = float(song[field])
            songs.append(song)
    return songs


def _closeness(a: float, b: float) -> float:
    """Returns 1.0 when two 0-1 values match, falling to 0.0 as they diverge."""
    return 1.0 - abs(a - b)


def _closeness_term(user_prefs: Dict, song: Dict, feature: str, target_key: str,
                    weight: float) -> Tuple[float, Optional[str]]:
    """Scores one numeric feature by closeness of song[feature] to the user's target.

    Returns (points, reason). When the user gives no target for this feature or the
    strategy zeroes its weight, returns (0.0, None) and the caller adds nothing.
    Shared by the energy and valence terms so their formatting and guard logic
    cannot silently drift apart.
    """
    target = user_prefs.get(target_key)
    if target is None or not weight:
        return 0.0, None
    target = float(target)
    value = song[feature]
    points = weight * _closeness(value, target)
    return points, f"{feature} {value:.2f} vs target {target:.2f} (+{points:.2f})"


def _top_k(items: List[T], score_of: Callable[[T], float],
           id_of: Callable[[T], int], k: int) -> List[T]:
    """Returns the k highest-scoring items, best first, ties broken by id_of.

    Pure: builds a new list via sorted(), so the caller's input is never reordered.
    This is the single ranking rule shared by Recommender.recommend and recommend_songs.
    """
    return sorted(items, key=lambda item: (-score_of(item), id_of(item)))[:k]


def score_song(user_prefs: Dict, song: Dict,
               strategy: ScoringStrategy = BALANCED) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    score = 0.0
    reasons: List[str] = []

    # 1. Genre - the heaviest term. Exact match beats a shared-word match
    #    ("indie pop" for someone who asked for "pop").
    want_genre = str(user_prefs.get("favorite_genre", "")).lower()
    song_genre = str(song.get("genre", "")).lower()
    if want_genre and want_genre == song_genre and strategy.genre:
        score += strategy.genre
        reasons.append(f"genre match: {song_genre} (+{strategy.genre:.1f})")
    elif (want_genre and strategy.genre_partial
          and set(want_genre.split()) & set(song_genre.split())):
        score += strategy.genre_partial
        reasons.append(f"partial genre match: {song_genre} (+{strategy.genre_partial:.1f})")

    # 2. Mood.
    want_mood = str(user_prefs.get("favorite_mood", "")).lower()
    if want_mood and strategy.mood and want_mood == str(song.get("mood", "")).lower():
        score += strategy.mood
        reasons.append(f"mood match: {want_mood} (+{strategy.mood:.1f})")

    # 3-4. Numeric closeness terms. Each is scored by closeness to the user's
    #      target, not by magnitude, so a user wanting calm music is not handed
    #      the most intense track. Valence carries the smaller weight; it is the
    #      only numeric not strongly correlated with energy, so it separates dark
    #      from bright. Both go through one helper so they cannot drift apart.
    for feature, target_key, weight in (
        ("energy", "target_energy", strategy.energy),
        ("valence", "target_valence", strategy.valence),
    ):
        points, reason = _closeness_term(user_prefs, song, feature, target_key, weight)
        if reason is not None:
            score += points
            reasons.append(reason)

    # 5. Acoustic preference - read the same column in opposite directions.
    if "likes_acoustic" in user_prefs and strategy.acoustic:
        acousticness = float(song["acousticness"])
        if user_prefs["likes_acoustic"]:
            points = strategy.acoustic * acousticness
            label = "acoustic"
        else:
            points = strategy.acoustic * (1.0 - acousticness)
            label = "produced"
        score += points
        reasons.append(f"{label} sound (+{points:.2f})")

    return round(score, 2), reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5,
                    strategy: ScoringStrategy = BALANCED) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, strategy)
        scored.append((song, score, "; ".join(reasons or ["no strong matches"])))

    # _top_k builds a new list via sorted(), so the caller's `songs` catalog is
    # left untouched. Ties break on id so runs repeat.
    return _top_k(scored, lambda item: item[1], lambda item: item[0]["id"], k)
