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
from typing import Dict, List, Optional, Set, Tuple

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

# Two explanation "styles" for the specialization A/B (SF-C). BASELINE is a generic,
# loosely-constrained prompt; SPECIALIZED adds hard grounding, a one-sentence/no-marketing
# tone constraint, and a one-shot exemplar (few-shot specialization).
EXPLAIN_SYSTEM_BASELINE = (
    "You are a music assistant. The user made a request and here are some candidate songs. "
    "Write a short recommendation of the songs you think fit best. Respond as JSON: "
    '{"summary": "...", "picks": [{"title": "...", "why": "..."}]}.'
)

EXPLAIN_SYSTEM_SPECIALIZED = (
    "You are a music recommendation assistant. Recommend songs to the user using "
    "ONLY the candidate songs provided — never mention a song that is not in the "
    "candidate list. Choose up to 3 and give a one-sentence reason for each, "
    "grounded in the song's genre, mood, and the scoring notes. You MAY use the "
    "genre context (factual background) to enrich a reason, but still recommend ONLY "
    "from the candidate songs. Keep each reason to ONE sentence and avoid marketing "
    "adjectives (no 'amazing', 'perfect', 'ultimate', 'best'). Respond as JSON: "
    '{"summary": "...", "picks": [{"title": "...", "why": "..."}]}.\n'
    'Example — for candidate "Library Rain" [lofi/chill], a good pick is '
    '{"title": "Library Rain", "why": "A lofi, chill track with a high acoustic score, '
    'matching your focus request."}'
)

# Back-compat alias: the specialized prompt is the production default.
EXPLAIN_SYSTEM = EXPLAIN_SYSTEM_SPECIALIZED


def _explain_system(style: str = "specialized") -> str:
    """Select the explanation system prompt for the A/B style ('baseline'|'specialized').

    A typo'd style is a loud error, not a silent fall-through to specialized — otherwise an
    A/B run could compare specialized against specialized without anyone noticing.
    """
    if style not in ("baseline", "specialized"):
        raise ValueError(f"unknown explanation style: {style!r}")
    return EXPLAIN_SYSTEM_BASELINE if style == "baseline" else EXPLAIN_SYSTEM_SPECIALIZED

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

# Refinement deltas ("make it calmer", "happier", ...) for the offline path.
_CALMER = {"calmer", "calm", "chill", "chiller", "slower", "softer", "mellow", "relax", "quieter"}
_LOUDER = {"louder", "harder", "faster", "hype", "energetic", "intense", "pump", "hyped"}
_HAPPIER = {"happier", "brighter", "cheerful", "positive", "upbeat"}
_SADDER = {"sadder", "darker", "moodier", "gloomier", "melancholy", "somber"}

REFINE_SYSTEM = (
    "You adjust a listener's music taste profile from a follow-up request. Known genres: "
    "{genres}. Known moods: {moods}. Given the CURRENT profile and the refinement, return the "
    "UPDATED full profile as JSON with the same keys (favorite_genre, favorite_mood, "
    "target_energy 0-1, target_valence 0-1, likes_acoustic true/false, blocked_genres list). "
    "Keep fields the refinement doesn't mention unchanged."
)


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


# --- Conversational refinement ----------------------------------------------------

# "\b" before the optional "more" keeps "dislike #2" from matching (no boundary inside
# "dislike"), so a 👎 phrase is never misread as a "more like #N" adoption.
_REF_RE = re.compile(r"\b(?:more\s+)?(?:like|number)\s*#?\s*(\d+)")


def _referenced_song(text: str, last_results) -> Optional[Dict]:
    """The song a "(more) like #N" / "number N" phrase points at, or None.

    Single source for reference resolution, used by both the LLM and offline paths so
    "#N" is honored regardless of backend (the LLM never sees the result list).
    """
    if not last_results:
        return None
    m = _REF_RE.search(_norm(text))
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < len(last_results):
        return last_results[idx][0]
    return None


def _refine_offline(prev: Dict, text: str, songs: List[Dict]) -> Dict:
    """Deterministic keyword deltas applied to a copy of the current profile.

    Reference resolution ("#N") is handled by the caller (`refine_profile`), so `text`
    here is the residual steering after any reference phrase is stripped.
    """
    prof = dict(prev)
    q_clean = " " + re.sub(r"[^a-z0-9&]+", " ", _norm(text)).strip() + " "
    tokens = set(q_clean.split())
    genres, _moods = _vocab(songs)

    if tokens & _LOUDER:
        prof["target_energy"] = _clamp01(round(prof.get("target_energy", 0.5) + 0.2, 2))
    elif tokens & _CALMER:
        prof["target_energy"] = _clamp01(round(prof.get("target_energy", 0.5) - 0.2, 2))
    if tokens & _HAPPIER:
        prof["target_valence"] = _clamp01(round(prof.get("target_valence", 0.5) + 0.2, 2))
    elif tokens & _SADDER:
        prof["target_valence"] = _clamp01(round(prof.get("target_valence", 0.5) - 0.2, 2))
    if tokens & _ACOUSTIC:
        prof["likes_acoustic"] = True
    elif tokens & _PRODUCED:
        prof["likes_acoustic"] = False

    # "no <genre>" adds a block (phrase-level, multi-word aware).
    neg = ("no", "not", "without", "avoid", "hate", "skip")
    new_blocks = {g for g in genres if any(f" {nw} {g} " in q_clean for nw in neg)}
    if new_blocks:
        prof["blocked_genres"] = sorted(set(prof.get("blocked_genres") or []) | new_blocks)
        if prof.get("favorite_genre") in new_blocks:
            prof["favorite_genre"] = ""

    # "more/add/some <genre>" switches the favorite genre.
    for g in sorted(genres, key=len, reverse=True):
        if g not in new_blocks and any(f" {w} {g} " in q_clean for w in ("more", "add", "some")):
            prof["favorite_genre"] = g
            break

    return prof


def refine_profile(prev_profile: Dict, text: str, songs: List[Dict], *,
                   backend=None, last_results=None) -> Dict:
    """Apply a natural-language refinement to an existing profile; return a new profile.

    A "(more) like #N" reference is resolved deterministically FIRST (the LLM never sees
    the result list): the referenced song's genre/energy/valence seed the profile, and a
    bare reference short-circuits without an LLM call. Residual steering then goes to the
    LLM (full updated profile) or, on any failure, deterministic keyword deltas
    (`_refine_offline`). `last_results` is the previous ``(song, score, reason)`` list.
    """
    prof = dict(prev_profile)
    text_norm = _norm(text)
    ref = _referenced_song(text, last_results)
    remaining = text_norm
    if ref is not None:
        prof["favorite_genre"] = _norm(ref.get("genre", prof.get("favorite_genre", "")))
        prof["target_energy"] = _clamp01(ref.get("energy"), prof.get("target_energy", 0.5))
        prof["target_valence"] = _clamp01(ref.get("valence"), prof.get("target_valence", 0.5))
        remaining = _REF_RE.sub(" ", text_norm).strip()
        if not remaining:  # pure "more like #N" — fully deterministic, no LLM needed
            return prof

    if backend is not None:
        try:
            genres, moods = _vocab(songs)
            system = REFINE_SYSTEM.format(
                genres=", ".join(sorted(genres)) or "(none)",
                moods=", ".join(sorted(moods)) or "(none)",
            )
            user = (
                f"Current profile - genre: {prof.get('favorite_genre') or 'any'}, "
                f"mood: {prof.get('favorite_mood') or 'any'}, "
                f"target_energy: {prof.get('target_energy')}, "
                f"target_valence: {prof.get('target_valence')}, "
                f"likes_acoustic: {prof.get('likes_acoustic')}, "
                f"blocked_genres: {prof.get('blocked_genres') or []}.\n"
                f"Refinement request: {remaining}"
            )
            profile = _coerce_profile(backend.complete_json(system, user, PROFILE_SCHEMA))
            logger.info("profile refined via %s", getattr(backend, "name", "llm"))
            return profile
        except Exception as exc:  # noqa: BLE001 - any failure degrades to offline
            logger.warning("LLM refine failed (%s); using offline delta", exc)
    return _refine_offline(prof, remaining, songs)


# --- Grounded explanation ---------------------------------------------------------

def _explain_prompt(query: str, retrieved, notes=None) -> str:
    lines = [f'User request: "{query}"', "", "Candidate songs (recommend ONLY from these):"]
    for song, _score, reasons in retrieved:
        lines.append(
            f'- "{song["title"]}" by {song["artist"]} '
            f'[{song.get("genre", "?")}/{song.get("mood", "?")}] — {reasons}'
        )
    if notes:  # second retrieval source: factual genre background (not song candidates)
        lines += ["", "Genre context (factual background you MAY use to enrich reasons):"]
        lines += [f"- {n}" for n in notes]
    return "\n".join(lines)


def _deterministic_explanation(retrieved, notes=None) -> str:
    """A grounded summary built directly from the top result's scoring reasons, optionally
    enriched with the top pick's genre note (the RAG second source)."""
    if not retrieved:
        return "No songs matched your request."
    song, score, reasons = retrieved[0]
    text = (
        f"Top match: {song['title']} by {song['artist']} "
        f"(score {score:.2f}). {reasons}"
    )
    if notes:  # notes[0] is the top pick's genre note (retrieve_notes preserves order)
        text += f" [{notes[0]}]"
    return text


def generate_explanation(query: str, retrieved, *, backend=None,
                         notes: Optional[List[str]] = None,
                         style: str = "specialized") -> Tuple[str, bool]:
    """Return (explanation_text, used_llm).

    Grounding guardrail (backend-agnostic): every recommended title must be in the
    retrieved set (normalized membership). On any backend/JSON failure or a title
    outside the set, fall back to the deterministic explanation. The returned text
    is the model-phrased "AI summary"; the caller always renders the deterministic
    scoring reasons from ``retrieved`` beside it.
    """
    _explain_system(style)  # validate style loudly, independent of backend (raises on typo)
    if not retrieved:
        return "No songs matched your request.", False

    if backend is not None:
        try:
            # normalized title -> canonical title (we render the canonical form)
            allowed = {_norm(song["title"]): song["title"] for song, _s, _r in retrieved}
            raw = backend.complete_json(_explain_system(style), _explain_prompt(query, retrieved, notes), EXPLAIN_SCHEMA)
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
    return _deterministic_explanation(retrieved, notes), False
