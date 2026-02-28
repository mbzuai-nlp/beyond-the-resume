from types import SimpleNamespace
from src.judge_testing import _belief_updated_expectedly, _is_close_to_uniform_prior


def _j(posteriors):
    return SimpleNamespace(posteriors=posteriors)


def test_invariant_examples():
    prior = _j({"dim": {"low": 0.30, "medium": 0.40, "high": 0.30}})
    post_pass = _j({"dim": {"low": 0.339, "medium": 0.37, "high": 0.30}})
    post_fail = _j({"dim": {"low": 0.341, "medium": 0.40, "high": 0.30}})

    assert _belief_updated_expectedly(prior, post_pass, None) is True
    assert _belief_updated_expectedly(prior, post_fail, None) is False


def test_increase_examples():
    expected = {"dimension": "dim", "level": "medium", "direction": "up"}

    prior = _j({"dim": {"low": 0.30, "medium": 0.30, "high": 0.40}})
    post_pass = _j({"dim": {"low": 0.30, "medium": 0.341, "high": 0.40}})
    post_fail = _j({"dim": {"low": 0.30, "medium": 0.339, "high": 0.40}})

    assert _belief_updated_expectedly(prior, post_pass, expected) is True
    assert _belief_updated_expectedly(prior, post_fail, expected) is False


def test_decrease_examples():
    expected = {"dimension": "dim", "level": "high", "direction": "down"}

    prior = _j({"dim": {"low": 0.20, "medium": 0.30, "high": 0.50}})
    post_pass = _j({"dim": {"low": 0.20, "medium": 0.30, "high": 0.459}})
    post_fail = _j({"dim": {"low": 0.20, "medium": 0.30, "high": 0.461}})

    assert _belief_updated_expectedly(prior, post_pass, expected) is True
    assert _belief_updated_expectedly(prior, post_fail, expected) is False


def test_uniform_examples():
    j_pass = _j({"dim": {"low": 1 / 3, "medium": 1 / 3, "high": 1 / 3}})
    j_fail = _j({"dim": {"low": 0.50, "medium": 0.25, "high": 0.25}})

    passed, _ = _is_close_to_uniform_prior(j_pass, "dim")
    assert passed is True

    passed, _ = _is_close_to_uniform_prior(j_fail, "dim")
    assert passed is False