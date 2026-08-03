import pytest

from src.agent import ACCEPT_STRONG, agentic_recommend
from src.recommender import load_songs

TRACE_KEYS = {"step", "action", "reason", "confidence", "strong_matches", "top_pick"}


def _songs():
    return load_songs("data/songs.csv")


def test_strong_query_accepts_step_one():
    result, trace = agentic_recommend("chill acoustic lofi to study", _songs(), backend=None)
    assert len(trace) == 1                              # accepted immediately, no relaxation
    assert trace[0]["strong_matches"] >= ACCEPT_STRONG
    assert result["strategy"] == "balanced"


def test_weak_query_relaxes_and_improves():
    # A blocked query is weak at first; the agent relaxes and finds a better pass.
    result, trace = agentic_recommend("upbeat pop, no pop", _songs(), backend=None)
    assert len(trace) > 1                                                  # it kept reasoning
    assert result["confidence"]["confidence"] > trace[0]["confidence"]     # strictly improved
    assert result["confidence"]["confidence"] == max(t["confidence"] for t in trace)  # best chosen


def test_max_steps_respected():
    _result, trace = agentic_recommend("upbeat pop, no pop", _songs(), backend=None, max_steps=2)
    assert len(trace) <= 2


def test_trace_keys_present():
    _result, trace = agentic_recommend("melancholy bluegrass", _songs(), backend=None)
    assert trace and all(TRACE_KEYS <= set(t) for t in trace)


def test_result_is_best_confidence_seen():
    result, trace = agentic_recommend("melancholy bluegrass", _songs(), backend=None)
    assert result["confidence"]["confidence"] == max(t["confidence"] for t in trace)


def test_max_steps_below_one_rejected():
    # A clean error, not an AssertionError (stripped under -O) or a TypeError.
    with pytest.raises(ValueError):
        agentic_recommend("anything", _songs(), backend=None, max_steps=0)
