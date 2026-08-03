"""RAG intelligence for the recommender, over any backend, with offline fallbacks.

Two jobs, each usable with a live LLM backend OR a deterministic offline path:

  - ``parse_profile``        — natural-language request -> a user_prefs dict that
                               drives the existing retriever (``recommend_songs``).
  - ``generate_explanation`` — a grounded, guardrailed natural-language summary of
                               the retrieved songs.

Both degrade to deterministic logic on any backend/JSON failure, so the app runs
free with no LLM. The grounding guardrail is the load-bearing safety net: the LLM
may only recommend songs from the retrieved set (validated by title membership);
its free-text prose is a labeled "AI summary" shown *alongside* — never in place
of — the deterministic scoring reasons the caller renders from ``recommend_songs``.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("recommender.llm")

# --- Structured-output schemas (used by the Anthropic backend; local ignores) ---

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "favorite_genre": {"type": "string"},
        "favorite_mood": {"type": "string"},
        "target_energy": {"type": "number"},
        "target_valence": {"type": "number"},
        "likes_acoustic": {"type": "boolean"},
        "blocked_genres": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "favorite_genre",
        "favorite_mood",
        "target_energy",
        "target_valence",
        "likes_acoustic",
        "blocked_genres",
    ],
    "additionalProperties": False,
}

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["title", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "picks"],
    "additionalProperties": False,
}

PROFILE_SYSTEM = (
    "You convert a listener's plain-English request into a music taste profile. "
    "Known genres: {genres}. Known moods: {moods}. Respond as JSON with keys: "
    'favorite_genre (one of the known genres, or ""), favorite_mood (one of the '
    'known moods, or ""), target_energy (0.0-1.0, how energetic), target_valence '
    "(0.0-1.0, how upbeat/positive), likes_acoustic (true/false), "
    'blocked_genres (a list of known genres the listener wants to avoid, e.g. from '
    '"no rock", or [] if none).'
)

EXPLAIN_SYSTEM = (
    "You are a music recommendation assistant. Recommend songs to the user using "
    "ONLY the candidate songs provided — never mention a song that is not in the "
    "candidate list. Choose up to 3 and give a one-sentence reason for each, "
    "grounded in the song's genre, mood, and the scoring notes. Respond as JSON: "
    '{"summary": "...", "picks": [{"title": "...", "why": "..."}]}.'
)

# --- Offline keyword vocabulary ---------------------------------------------------

_LOW_ENERGY = {"calm", "chill", "relax", "relaxed", "study", "sleep", "slow",
               "mellow", "quiet", "soft", "lofi", "ambient"}
_HIGH_ENERGY = {"hype", "workout", "gym", "intense", "energetic", "party",
                "dance", "fast", "pump", "hard", "banger", "upbeat"}
_HAPPY = {"happy", "upbeat", "cheerful", "bright", "joy", "joyful", "positive",
          "sunny", "feel-good", "feelgood", "euphoric"}
_SAD = {"sad", "dark", "melancholy", "moody", "down", "gloomy", "somber",
        "depressing", "blue"}
_ACOUSTIC = {"acoustic", "unplugged", "organic", "folk", "stripped"}
_PRODUCED = {"produced", "electronic", "synth", "edm", "electro", "digital"}


def _vocab(songs: List[Dict]) -> Tuple[Set[str], Set[str]]:
    """Distinct genres and moods present in the catalog (keeps parsing in sync)."""
    genres = {str(s.get("genre", "")).lower() for s in songs if s.get("genre")}
    moods = {str(s.get("mood", "")).lower() for s in songs if s.get("mood")}
    return genres, moods


def _clamp01(value, default: float = 0.5) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def _norm(text: str) -> str:
    return str(text).strip().lower()


# --- Profile parsing --------------------------------------------------------------

def _as_str_list(value) -> List[str]:
    """Coerce an LLM value (list, single string, or junk) into a list of strings."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def _coerce_profile(raw: Dict) -> Dict:
    """Clamp/normalize an LLM-returned profile into the user_prefs dict shape."""
    if not isinstance(raw, dict):
        raise ValueError("profile is not an object")
    return {
        "favorite_genre": _norm(raw.get("favorite_genre", "")),
        "favorite_mood": _norm(raw.get("favorite_mood", "")),
        "target_energy": _clamp01(raw.get("target_energy")),
        "target_valence": _clamp01(raw.get("target_valence")),
        "likes_acoustic": bool(raw.get("likes_acoustic", False)),
        "blocked_genres": sorted({_norm(g) for g in _as_str_list(raw.get("blocked_genres"))}),
    }


def _keyword_profile(query: str, songs: List[Dict]) -> Dict:
    """Deterministic offline parser: match the query against catalog vocabulary."""
    q = " " + _norm(query) + " "
    genres, moods = _vocab(songs)

    def _first_present(vocab: Set[str]) -> str:
        # Longest first so "indie pop" beats "pop", "hip hop" beats "pop", etc.
        for term in sorted(vocab, key=len, reverse=True):
            if term and term in q:
                return term
        return ""

    tokens = set(q.split())
    if tokens & _HIGH_ENERGY:
        energy = 0.9
    elif tokens & _LOW_ENERGY:
        energy = 0.2
    else:
        energy = 0.5

    if tokens & _HAPPY:
        valence = 0.8
    elif tokens & _SAD:
        valence = 0.2
    else:
        valence = 0.5

    if tokens & _ACOUSTIC:
        likes_acoustic = True
    elif tokens & _PRODUCED:
        likes_acoustic = False
    else:
        likes_acoustic = False

    # Dislikes: a negation word immediately followed by a known genre PHRASE
    # ("no rock", "no hip hop"). Whitespace-boundary matching on a
    # punctuation-normalized query, so multi-word genres are blockable and
    # "piano house" does NOT read as "no house".
    neg = ("no", "not", "without", "avoid", "hate", "hates", "except", "skip")
    q_clean = " " + re.sub(r"[^a-z0-9&]+", " ", q).strip() + " "
    blocked = sorted(
        g for g in genres
        if any(f" {nw} {g} " in q_clean for nw in neg)
    )

    return {
        "favorite_genre": _first_present(genres - set(blocked)),
        "favorite_mood": _first_present(moods),
        "target_energy": energy,
        "target_valence": valence,
        "likes_acoustic": likes_acoustic,
        "blocked_genres": blocked,
    }


def parse_profile(query: str, songs: List[Dict], *, backend=None) -> Tuple[Dict, str]:
    """NL request -> (user_prefs dict, source) where source is "llm" or "offline"."""
    if backend is not None:
        try:
            genres, moods = _vocab(songs)
            system = PROFILE_SYSTEM.format(
                genres=", ".join(sorted(genres)) or "(none)",
                moods=", ".join(sorted(moods)) or "(none)",
            )
            raw = backend.complete_json(system, query, PROFILE_SCHEMA)
            profile = _coerce_profile(raw)
            logger.info("profile parsed via %s", getattr(backend, "name", "llm"))
            return profile, "llm"
        except Exception as exc:  # noqa: BLE001 - any failure degrades to offline
            logger.warning("LLM profile parse failed (%s); using offline parser", exc)
    return _keyword_profile(query, songs), "offline"


# --- Grounded explanation ---------------------------------------------------------

def _explain_prompt(query: str, retrieved) -> str:
    lines = [f'User request: "{query}"', "", "Candidate songs (recommend ONLY from these):"]
    for song, _score, reasons in retrieved:
        lines.append(
            f'- "{song["title"]}" by {song["artist"]} '
            f'[{song.get("genre", "?")}/{song.get("mood", "?")}] — {reasons}'
        )
    return "\n".join(lines)


def _deterministic_explanation(retrieved) -> str:
    """A grounded summary built directly from the top result's scoring reasons."""
    if not retrieved:
        return "No songs matched your request."
    song, score, reasons = retrieved[0]
    return (
        f"Top match: {song['title']} by {song['artist']} "
        f"(score {score:.2f}). {reasons}"
    )


def generate_explanation(query: str, retrieved, *, backend=None) -> Tuple[str, bool]:
    """Return (explanation_text, used_llm).

    Grounding guardrail (backend-agnostic): every recommended title must be in the
    retrieved set (normalized membership). On any backend/JSON failure or a title
    outside the set, fall back to the deterministic explanation. The returned text
    is the model-phrased "AI summary"; the caller always renders the deterministic
    scoring reasons from ``retrieved`` beside it.
    """
    if not retrieved:
        return "No songs matched your request.", False

    if backend is not None:
        try:
            # normalized title -> canonical title (we render the canonical form)
            allowed = {_norm(song["title"]): song["title"] for song, _s, _r in retrieved}
            raw = backend.complete_json(EXPLAIN_SYSTEM, _explain_prompt(query, retrieved), EXPLAIN_SCHEMA)
            picks = raw.get("picks") if isinstance(raw, dict) else None
            if not isinstance(picks, list) or not picks:
                raise ValueError("no picks returned")

            lines = []
            for pick in picks:
                key = _norm((pick or {}).get("title", ""))
                if key not in allowed:
                    raise ValueError(f"hallucinated title: {(pick or {}).get('title')!r}")
                why = str((pick or {}).get("why", "")).strip()
                lines.append(f"- {allowed[key]}: {why}" if why else f"- {allowed[key]}")

            summary = str(raw.get("summary", "")).strip()
            text = (summary + "\n" if summary else "") + "\n".join(lines)
            logger.info("explanation generated via %s", getattr(backend, "name", "llm"))
            return text.strip(), True
        except Exception as exc:  # noqa: BLE001 - guardrail/transport failure -> offline
            logger.warning(
                "LLM explanation rejected/failed (%s); using deterministic explanation", exc
            )
    return _deterministic_explanation(retrieved), False
