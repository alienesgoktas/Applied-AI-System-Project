import pytest

from src.llm import (
    EXPLAIN_SYSTEM_BASELINE,
    EXPLAIN_SYSTEM_SPECIALIZED,
    _explain_system,
    generate_explanation,
)


def _retrieved():
    return [
        ({"title": "Sunrise City", "artist": "Neon Bloom", "genre": "pop", "mood": "happy"},
         6.1, "genre match: pop (+2.0)"),
        ({"title": "Paper Boats", "artist": "Kite", "genre": "folk", "mood": "calm"},
         3.2, "genre match: folk (+2.0)"),
    ]


def test_baseline_and_specialized_prompts_differ():
    assert EXPLAIN_SYSTEM_BASELINE != EXPLAIN_SYSTEM_SPECIALIZED


def test_specialized_has_grounding_tone_and_exemplar():
    s = EXPLAIN_SYSTEM_SPECIALIZED
    assert "ONLY" in s                       # hard grounding
    assert "marketing adjectives" in s       # tone constraint
    assert "Library Rain" in s               # one-shot exemplar (few-shot)
    assert "ONLY" not in EXPLAIN_SYSTEM_BASELINE   # baseline is loose by design


def test_style_routing():
    assert _explain_system("baseline") == EXPLAIN_SYSTEM_BASELINE
    assert _explain_system("specialized") == EXPLAIN_SYSTEM_SPECIALIZED
    assert _explain_system() == EXPLAIN_SYSTEM_SPECIALIZED   # default is specialized


def test_offline_both_styles_identical():
    # With no backend, style has no effect — both fall back to the same deterministic text.
    base, u1 = generate_explanation("x", _retrieved(), backend=None, style="baseline")
    spec, u2 = generate_explanation("x", _retrieved(), backend=None, style="specialized")
    assert base == spec
    assert u1 is False and u2 is False


def test_unknown_style_raises_loudly():
    # A typo'd style must fail, not silently compare specialized vs specialized.
    with pytest.raises(ValueError):
        _explain_system("baselnie")
    with pytest.raises(ValueError):
        generate_explanation("x", _retrieved(), backend=None, style="whoops")
