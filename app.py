"""Streamlit web UI for the Music Recommender (RAG + reliability layer).

Run from the repo root:

    streamlit run app.py

Pick a backend (your local server / Anthropic bring-your-own-key / free offline),
type a plain-English request, and get grounded recommendations with the
deterministic scoring reasons, a confidence meter, an honesty note, and an
advisory AI summary. Any API key you type stays in this browser session — it is
never stored or logged.
"""

import streamlit as st

from src.backends import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    AnthropicBackend,
    LocalServerBackend,
)
from src.pipeline import backend_label, recommend_from_query
from src.recommender import load_songs

CATALOG = "data/songs.csv"


@st.cache_data
def _load_catalog():
    return load_songs(CATALOG)


def _build_backend(choice: str, cfg: dict):
    """Construct the chosen backend from the sidebar config (or None -> offline)."""
    if choice == "Local server":
        return LocalServerBackend(cfg["base_url"], cfg["model"])
    if choice == "Anthropic (BYOK)":
        if not cfg.get("api_key"):
            return None  # no key -> offline
        return AnthropicBackend(cfg["model"], cfg["api_key"])
    return None  # Offline


st.set_page_config(page_title="Music Recommender - RAG", page_icon="🎵", layout="centered")
st.title("🎵 Music Recommender")
st.caption(
    "Describe what you want to hear. A local content-based recommender retrieves the "
    "matches; an LLM explains them - grounded to only the retrieved songs."
)

songs = _load_catalog()

with st.sidebar:
    st.header("LLM backend")
    choice = st.radio(
        "Where should the AI run?",
        ["Local server", "Anthropic (BYOK)", "Offline (no LLM)"],
        help="Offline uses the deterministic parser + scoring reasons - free, no key, always works.",
    )
    cfg: dict = {}
    if choice == "Local server":
        cfg["base_url"] = st.text_input("Base URL", value=DEFAULT_LOCAL_BASE_URL)
        cfg["model"] = st.text_input("Model", value=DEFAULT_LOCAL_MODEL)
    elif choice == "Anthropic (BYOK)":
        cfg["api_key"] = st.text_input(
            "Anthropic API key", type="password",
            help="Stays in this session; never stored or logged.",
        )
        cfg["model"] = st.text_input("Model", value=DEFAULT_ANTHROPIC_MODEL)
    st.divider()
    k = st.slider("How many recommendations?", 1, 10, 5)

query = st.text_input("Your request", placeholder="e.g. chill acoustic music to study to")
go = st.button("Recommend", type="primary")

if go and query.strip():
    backend = _build_backend(choice, cfg)
    with st.spinner("Thinking..."):
        result = recommend_from_query(query.strip(), songs, k=k, backend=backend)

    badge = backend_label(result)

    profile = result["profile"]
    conf = result["confidence"]

    st.subheader(f"Results  ·  `{badge}`")
    st.write(
        f"**Understood as:** {profile['favorite_genre'] or '(any)'} / "
        f"{profile['favorite_mood'] or '(any)'} · energy {profile['target_energy']:.2f} · "
        f"valence {profile['target_valence']:.2f} · "
        f"{'acoustic' if profile['likes_acoustic'] else 'produced'}"
    )
    st.progress(
        min(conf["confidence"], 1.0),
        text=f"Confidence {conf['confidence']:.2f} - {conf['note']}",
    )

    for rank, (song, score, reasons) in enumerate(result["results"], 1):
        with st.container(border=True):
            st.markdown(
                f"**{rank}. {song['title']}** - {song['artist']}  \n"
                f"`{song['genre']} / {song['mood']}` · score {score:.2f}"
            )
            for reason in reasons.split("; "):
                if reason.strip():
                    st.caption(f"• {reason}")

    st.divider()
    st.markdown("**AI summary** *(advisory - the scoring reasons above are the record)*")
    st.info(result["explanation"])
elif go:
    st.warning("Type a request first.")
