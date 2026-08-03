from src.llm import _keyword_profile, generate_explanation, parse_profile
from src.recommender import load_songs


def _songs():
    return load_songs("data/songs.csv")


def _retrieved():
    return [
        ({"title": "Sunrise City", "artist": "Neon Echo", "genre": "pop", "mood": "happy"},
         6.2, "genre match: pop (+2.0)"),
        ({"title": "Rooftop Lights", "artist": "Indigo Parade", "genre": "indie pop", "mood": "happy"},
         5.0, "mood match: happy (+1.5)"),
    ]


# --- offline keyword parser ---

def test_offline_keyword_profile_reads_the_request():
    prof = _keyword_profile("calm acoustic lofi to study", _songs())
    assert prof["favorite_genre"] == "lofi"
    assert prof["target_energy"] == 0.2
    assert prof["likes_acoustic"] is True


def test_offline_parse_is_deterministic():
    songs = _songs()
    a = _keyword_profile("intense metal workout", songs)
    b = _keyword_profile("intense metal workout", songs)
    assert a == b
    assert a["favorite_genre"] == "metal"
    assert a["target_energy"] == 0.9


# --- parse_profile over a backend ---

def test_parse_profile_uses_llm_payload(fake_backend):
    payload = {"favorite_genre": "jazz", "favorite_mood": "relaxed",
               "target_energy": 0.3, "target_valence": 0.4, "likes_acoustic": True}
    prof, source = parse_profile("anything", _songs(), backend=fake_backend(payload=payload))
    assert source == "llm"
    assert prof["favorite_genre"] == "jazz"
    assert prof["target_energy"] == 0.3


def test_parse_profile_clamps_and_normalizes(fake_backend):
    payload = {"favorite_genre": "POP", "favorite_mood": "Happy",
               "target_energy": 5, "target_valence": -2, "likes_acoustic": "yes"}
    prof, source = parse_profile("x", _songs(), backend=fake_backend(payload=payload))
    assert source == "llm"
    assert prof["favorite_genre"] == "pop"
    assert prof["target_energy"] == 1.0
    assert prof["target_valence"] == 0.0
    assert prof["likes_acoustic"] is True


def test_parse_profile_falls_back_on_backend_error(fake_backend):
    songs = _songs()
    prof, source = parse_profile("calm lofi", songs, backend=fake_backend(error=RuntimeError("boom")))
    assert source == "offline"
    assert prof == _keyword_profile("calm lofi", songs)


def test_parse_profile_falls_back_on_malformed_payload(fake_backend):
    prof, source = parse_profile("q", _songs(), backend=fake_backend(payload="not a dict"))
    assert source == "offline"


def test_parse_profile_falls_back_without_backend():
    _prof, source = parse_profile("calm lofi", _songs(), backend=None)
    assert source == "offline"


# --- generate_explanation grounding guardrail ---

def test_generate_explanation_uses_valid_picks(fake_backend):
    payload = {"summary": "Great upbeat set.", "picks": [{"title": "Sunrise City", "why": "bright pop"}]}
    text, used = generate_explanation("upbeat", _retrieved(), backend=fake_backend(payload=payload))
    assert used is True
    assert "Sunrise City" in text
    assert "Great upbeat set." in text


def test_generate_explanation_rejects_hallucinated_title(fake_backend):
    payload = {"summary": "x", "picks": [{"title": "Totally Made Up Song", "why": "nope"}]}
    text, used = generate_explanation("upbeat", _retrieved(), backend=fake_backend(payload=payload))
    assert used is False
    assert "Totally Made Up Song" not in text
    assert "Sunrise City" in text  # deterministic fallback names the real top song


def test_generate_explanation_falls_back_on_bad_json(fake_backend):
    text, used = generate_explanation("x", _retrieved(), backend=fake_backend(error=ValueError("no json")))
    assert used is False
    assert "Sunrise City" in text


def test_generate_explanation_offline_without_backend():
    text, used = generate_explanation("x", _retrieved(), backend=None)
    assert used is False
    assert "Sunrise City" in text
