import eval as eval_mod
from eval import CASES, EvalCase, Expect, evaluate_case, run_eval
from src.recommender import load_songs


def _songs():
    return load_songs("data/songs.csv")


def test_run_eval_all_pass_offline():
    summary = run_eval(_songs(), CASES, backend=None)
    assert summary["total"] == len(CASES)
    assert summary["passed"] == summary["total"]  # every calibrated case passes offline
    assert all(r["passed"] for r in summary["rows"])
    row = summary["rows"][0]
    assert {"query", "top", "genre", "confidence", "strong", "passed", "failures"} <= set(row)


def test_failing_expectation_is_reported():
    # An impossible expectation must FAIL — proves the harness checks, not rubber-stamps.
    bad = EvalCase("upbeat happy pop for a workout", Expect(top_genre="metal"))
    summary = run_eval(_songs(), [bad], backend=None)
    assert summary["passed"] == 0
    assert summary["rows"][0]["passed"] is False
    assert any("top genre" in f for f in summary["rows"][0]["failures"])


def test_confidence_bound_failure():
    bad = EvalCase("melancholy bluegrass", Expect(min_confidence=0.99))
    row = evaluate_case(bad, _songs(), backend=None)
    assert row["passed"] is False
    assert any("min" in f for f in row["failures"])


def test_main_exit_code_contract(monkeypatch):
    # The CI contract: offline run of all calibrated cases exits 0.
    monkeypatch.setattr("sys.argv", ["eval.py"])
    assert eval_mod.main() == 0
