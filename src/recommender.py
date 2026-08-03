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
    dislike: float = 3.0        # penalty subtracted when a song is a blocked genre

    def max_score(self) -> float:
        """Highest total a song can earn under this strategy (every term maxed)."""
        return self.genre + self.mood + self.energy + self.valence + self.acoustic

    def categorical_max(self) -> float:
        """Most the categorical terms alone can contribute (exact genre + mood)."""
        return self.genre + self.mood


# The recipe documented in README.md, plus the variants used in Phase 4.
BALANCED = ScoringStrategy()
ENERGY_FIRST = ScoringStrategy("energy-first", genre=1.0, genre_partial=0.5, energy=3.0)
MOOD_BLIND = ScoringStrategy("mood-blind", mood=0.0)
GENRE_PURIST = ScoringStrategy("genre-purist", genre=4.0, genre_partial=2.0)

# Presets addressable by name (e.g. from a UI dropdown or an env var).
STRATEGIES = {s.name: s for s in (BALANCED, ENERGY_FIRST, MOOD_BLIND, GENRE_PURIST)}


def strategy_from_name(name, default: ScoringStrategy = BALANCED) -> ScoringStrategy:
    """Resolve a preset name (case-insensitive) to a ScoringStrategy, else `default`."""
    return STRATEGIES.get(str(name).strip().lower(), default) if name else default


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


def load_genre_notes(csv_path: str) -> Dict[str, str]:
    """Load the genre-knowledge source (the RAG second source): ``{genre_lower: note}``.

    Mirrors ``load_songs`` (stdlib csv, fail-fast). Keys are lowercased so lookups against
    ``song["genre"]`` (already lowercase in the catalog) match multi-word / ``&`` genres
    like ``"hip hop"`` and ``"r&b"`` exactly.
    """
    notes: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            notes[row["genre"].strip().lower()] = row["note"].strip()
    return notes


def retrieve_notes(retrieved: List[Tuple[Dict, float, str]],
                   notes: Dict[str, str]) -> List[str]:
    """Second retrieval pass: the genre notes for the retrieved songs' distinct genres,
    in first-appearance order (so ``[0]`` is the top pick's genre). Returns ``"genre: note"``
    lines; genres with no note row are skipped."""
    out: List[str] = []
    seen = set()
    for song, _score, _reason in retrieved:
        g = str(song.get("genre", "")).strip().lower()
        if g and g not in seen and g in notes:
            seen.add(g)
            out.append(f"{g}: {notes[g]}")
    return out


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


def _diverse_top_k(scored: List[Tuple[Dict, float, str]], k: int,
                   max_per_artist: int) -> List[Tuple[Dict, float, str]]:
    """Top-k with at most `max_per_artist` songs per artist, best-first.

    Backfills from the deferred (over-cap) pool if the cap can't fill k, so the
    result still has k items whenever the catalog is large enough — de-bubbling
    the top slots without ever shrinking the list below the plain top-k would.
    """
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]["id"]))
    picked: List[Tuple[Dict, float, str]] = []
    deferred: List[Tuple[Dict, float, str]] = []
    counts: Dict[str, int] = {}
    for item in ranked:
        artist = str(item[0].get("artist", ""))
        if counts.get(artist, 0) < max_per_artist:
            picked.append(item)
            counts[artist] = counts.get(artist, 0) + 1
        else:
            deferred.append(item)
        if len(picked) == k:
            return picked
    for item in deferred:  # backfill in rank order
        if len(picked) >= k:
            break
        picked.append(item)
    return picked[:k]


def _score_terms(user_prefs: Dict, song: Dict,
                 strategy: ScoringStrategy) -> List[Tuple[str, float, str]]:
    """Every scoring term that fires, as (chart_label, points, reason_string).

    The single source for both score_song (the reason strings) and score_detail (the
    chart points), so the two can never drift. Reason-string formats are preserved
    exactly (genre/mood at .1f, numeric/acoustic at .2f).
    """
    terms: List[Tuple[str, float, str]] = []

    # 1. Genre - the heaviest term. Exact match beats a shared-word match
    #    ("indie pop" for someone who asked for "pop").
    want_genre = str(user_prefs.get("favorite_genre", "")).lower()
    song_genre = str(song.get("genre", "")).lower()
    if want_genre and want_genre == song_genre and strategy.genre:
        terms.append(("genre", strategy.genre,
                      f"genre match: {song_genre} (+{strategy.genre:.1f})"))
    elif (want_genre and strategy.genre_partial
          and set(want_genre.split()) & set(song_genre.split())):
        terms.append(("genre", strategy.genre_partial,
                      f"partial genre match: {song_genre} (+{strategy.genre_partial:.1f})"))

    # 2. Mood.
    want_mood = str(user_prefs.get("favorite_mood", "")).lower()
    if want_mood and strategy.mood and want_mood == str(song.get("mood", "")).lower():
        terms.append(("mood", strategy.mood, f"mood match: {want_mood} (+{strategy.mood:.1f})"))

    # 3-4. Numeric closeness terms (energy, valence), scored by closeness to the
    #      target - both through one helper so they cannot drift apart.
    for feature, target_key, weight in (
        ("energy", "target_energy", strategy.energy),
        ("valence", "target_valence", strategy.valence),
    ):
        points, reason = _closeness_term(user_prefs, song, feature, target_key, weight)
        if reason is not None:
            terms.append((feature, points, reason))

    # 5. Acoustic preference - read the acousticness column in the user's direction.
    if "likes_acoustic" in user_prefs and strategy.acoustic:
        acousticness = float(song["acousticness"])
        if user_prefs["likes_acoustic"]:
            points, label = strategy.acoustic * acousticness, "acoustic"
        else:
            points, label = strategy.acoustic * (1.0 - acousticness), "produced"
        terms.append(("acoustic", points, f"{label} sound (+{points:.2f})"))

    # 6. Dislikes - a penalty when the song is a user-blocked genre. Guarded on the
    #    key, so callers that never set `blocked_genres` are unaffected.
    blocked = user_prefs.get("blocked_genres") or []
    if blocked and strategy.dislike and song_genre in {str(g).lower() for g in blocked}:
        terms.append(("blocked", -strategy.dislike,
                      f"blocked genre: {song_genre} (-{strategy.dislike:.1f})"))

    return terms


def _total(terms: List[Tuple[str, float, str]]) -> float:
    """Naive left-fold sum of the term points, shared by score_song/score_detail.

    Deliberately NOT the built-in ``sum()``: on Python 3.12+ ``sum()`` uses
    compensated (Neumaier) summation, which differs from the original left-fold
    accumulation by up to a cent — this keeps scores byte-identical to that.
    """
    total = 0.0
    for _label, points, _reason in terms:
        total += points
    return total


def score_song(user_prefs: Dict, song: Dict,
               strategy: ScoringStrategy = BALANCED) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    terms = _score_terms(user_prefs, song, strategy)
    score = round(_total(terms), 2)
    return score, [reason for _label, _points, reason in terms]


def score_detail(user_prefs: Dict, song: Dict,
                 strategy: ScoringStrategy = BALANCED) -> Tuple[float, List[Tuple[str, float]]]:
    """Same score as score_song, plus the per-term (label, points) breakdown for charts.

    Shares `_score_terms` with score_song, so the charted points always equal the
    score the reasons explain.
    """
    terms = _score_terms(user_prefs, song, strategy)
    score = round(_total(terms), 2)
    return score, [(label, points) for label, points, _reason in terms]


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5,
                    strategy: ScoringStrategy = BALANCED,
                    max_per_artist: Optional[int] = None) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    `max_per_artist` (optional) caps how many songs one artist may fill in the
    top-k (a diversity control); `None` keeps the plain top-k behavior.
    """
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, strategy)
        scored.append((song, score, "; ".join(reasons or ["no strong matches"])))

    # Both ranking paths build a new list, so the caller's `songs` is left
    # untouched. Ties break on id so runs repeat.
    if max_per_artist is not None and max_per_artist > 0:
        return _diverse_top_k(scored, k, max_per_artist)
    return _top_k(scored, lambda item: item[1], lambda item: item[0]["id"], k)
