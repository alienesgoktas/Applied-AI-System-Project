import src.pipeline as pipe
from src.llm import _deterministic_explanation
from src.pipeline import recommend_from_query
from src.recommender import load_genre_notes, load_songs, retrieve_notes

NOTES_PATH = "data/genre_notes.csv"


def _songs():
    return load_songs("data/songs.csv")


def test_load_genre_notes_covers_every_catalog_genre():
    notes = load_genre_notes(NOTES_PATH)
    catalog_genres = {s["genre"] for s in _songs()}
    assert catalog_genres <= set(notes)  # no retrieved genre can be silently context-less
    for g in ("hip hop", "indie pop", "r&b"):  # multi-word / & keys must round-trip
        assert g in notes and notes[g]


def test_retrieve_notes_exact_key_roundtrip_and_order():
    notes = load_genre_notes(NOTES_PATH)
    retrieved = [
        ({"genre": "r&b"}, 5.0, ""),
        ({"genre": "hip hop"}, 4.0, ""),
        ({"genre": "r&b"}, 3.0, ""),        # duplicate genre -> deduped
        ({"genre": "indie pop"}, 2.0, ""),
    ]
    out = retrieve_notes(retrieved, notes)
    assert out[0].startswith("r&b: ")        # top pick's genre first (order-stable)
    assert any(n.startswith("hip hop: ") for n in out)
    assert any(n.startswith("indie pop: ") for n in out)
    assert len(out) == 3                      # r&b appears once


def test_retrieve_notes_skips_unknown_genre():
    assert retrieve_notes([({"genre": "polka"}, 1.0, "")], {"pop": "x"}) == []


def test_deterministic_explanation_enriched_with_note():
    retrieved = [({"title": "T", "artist": "A", "genre": "lofi", "mood": "chill"}, 6.0, "genre match")]
    text = _deterministic_explanation(retrieved, ["lofi: downtempo beats for focus"])
    assert "downtempo beats for focus" in text
    # without notes, the text is unchanged (existing callers pass none)
    assert "downtempo" not in _deterministic_explanation(retrieved)


def test_pipeline_adds_notes_key_and_grounds_offline():
    res = recommend_from_query("chill lofi to study", _songs(), k=3, backend=None)
    assert isinstance(res["notes"], list) and res["notes"]      # additive key present
    assert res["notes"][0].startswith("lofi: ")                 # matches the top genre
    assert "lofi:" in res["explanation"]                        # second source surfaced offline


def test_missing_notes_file_degrades(monkeypatch, tmp_path):
    # The second source is enrichment: an absent file must not crash the pipeline.
    monkeypatch.setattr(pipe, "_GENRE_NOTES_PATH", str(tmp_path / "nope.csv"))
    monkeypatch.setattr(pipe, "_genre_notes_cache", None)
    res = recommend_from_query("chill lofi to study", _songs(), k=3, backend=None)
    assert res["notes"] == [] and "lofi:" not in res["explanation"]


def test_malformed_notes_file_degrades(monkeypatch, tmp_path):
    # A file missing the 'note' column (KeyError) must degrade to {}, not crash.
    bad = tmp_path / "bad.csv"
    bad.write_text("genre\nlofi\n", encoding="utf-8")
    monkeypatch.setattr(pipe, "_GENRE_NOTES_PATH", str(bad))
    monkeypatch.setattr(pipe, "_genre_notes_cache", None)
    res = recommend_from_query("chill lofi to study", _songs(), k=3, backend=None)
    assert res["notes"] == []
