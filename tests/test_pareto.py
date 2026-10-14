import pytest

from verge_lab.models import Candidate, Embedding, Objective, Score
from verge_lab.pareto import compare_candidates, mine_pareto_edges


def objective(identifier: str, direction: str = "maximize") -> Objective:
    return Objective(identifier, identifier.title(), "test objective", direction, "#000000")


def candidate(identifier: str, scores: dict[str, Score], prompt: str = "prompt") -> Candidate:
    return Candidate(identifier, prompt, identifier, 1, 10, scores, Embedding(0, 0, 0))


def test_direction_normalized_dominance_honors_minimization() -> None:
    objectives = (objective("quality"), objective("latency", "minimize"))
    fast_good = candidate(
        "fast-good", {"quality": Score(0.9, 1, "verified"), "latency": Score(20, 1, "measured")}
    )
    slow_weak = candidate(
        "slow-weak", {"quality": Score(0.7, 1, "verified"), "latency": Score(40, 1, "measured")}
    )

    pair = compare_candidates(fast_good, slow_weak, objectives)

    assert pair.verdict == "defended"
    assert pair.chosen_id == "fast-good"
    assert pair.margins["quality"] > 0
    assert pair.margins["latency"] > 0


def test_cross_objective_tradeoff_remains_ambiguous() -> None:
    objectives = (objective("quality"), objective("latency", "minimize"))
    accurate = candidate(
        "accurate", {"quality": Score(0.9, 1, "verified"), "latency": Score(50, 1, "measured")}
    )
    quick = candidate(
        "quick", {"quality": Score(0.7, 1, "verified"), "latency": Score(20, 1, "measured")}
    )

    pair = compare_candidates(accurate, quick, objectives)

    assert pair.verdict == "ambiguous"
    assert pair.reason == "cross-objective tradeoff"


def test_missing_and_low_confidence_reasons_are_deterministic() -> None:
    objectives = (objective("a"), objective("b"))
    complete = candidate("a", {"a": Score(0.9, 1, "a"), "b": Score(0.9, 1, "b")})
    missing = candidate("b", {"a": Score(0.8, 1, "a")})
    low = candidate("c", {"a": Score(0.8, 0.5, "a"), "b": Score(0.8, 1, "b")})

    assert compare_candidates(complete, missing, objectives).reason == "missing scores: b"
    assert compare_candidates(complete, low, objectives).reason == "confidence below threshold: a"


def test_mining_order_and_ids_do_not_depend_on_input_order() -> None:
    objectives = (objective("quality"),)
    candidates = [
        candidate("c", {"quality": Score(0.6, 1, "verified")}),
        candidate("a", {"quality": Score(0.9, 1, "verified")}),
        candidate("b", {"quality": Score(0.7, 1, "verified")}),
    ]

    forward = mine_pareto_edges(candidates, objectives)
    reverse = mine_pareto_edges(reversed(candidates), objectives)

    assert [pair.id for pair in forward] == [pair.id for pair in reverse]
    assert [pair.to_dict() for pair in forward] == [pair.to_dict() for pair in reverse]


def test_mining_accepts_a_one_shot_objective_iterator() -> None:
    objectives = (objective("quality"),)
    candidates = [
        candidate("a", {"quality": Score(0.9, 1, "verified")}),
        candidate("b", {"quality": Score(0.7, 1, "verified")}),
        candidate("c", {"quality": Score(0.6, 1, "verified")}),
    ]

    from_iterator = mine_pareto_edges(candidates, iter(objectives))

    assert from_iterator == mine_pareto_edges(candidates, objectives)
    assert {pair.verdict for pair in from_iterator} == {"defended"}


def test_comparison_requires_at_least_one_objective() -> None:
    left = candidate("a", {"quality": Score(0.9, 1, "verified")})
    right = candidate("b", {"quality": Score(0.7, 1, "verified")})

    with pytest.raises(ValueError, match="at least one objective"):
        compare_candidates(left, right, ())


def test_tie_on_one_objective_with_gain_on_another_is_defended() -> None:
    objectives = (objective("a"), objective("b"))
    better = candidate("better", {"a": Score(0.8, 1, "a"), "b": Score(0.9, 1, "b")})
    worse = candidate("worse", {"a": Score(0.8, 1, "a"), "b": Score(0.6, 1, "b")})

    pair = compare_candidates(worse, better, objectives)

    assert pair.verdict == "defended"
    assert pair.chosen_id == "better"
    assert pair.margins == {"a": 0.0, "b": 0.3}
    assert pair.reason == "lower-bound Pareto dominance: b"


def test_gain_no_larger_than_epsilon_is_equivalent() -> None:
    objectives = (objective("a"),)
    left = candidate("left", {"a": Score(0.805, 1, "a")})
    right = candidate("right", {"a": Score(0.8, 1, "a")})

    pair = compare_candidates(left, right, objectives, epsilon=0.01)

    assert pair.verdict == "ambiguous"
    assert pair.reason == "equivalent within epsilon"


def test_confidence_penalty_can_remove_dominance_on_a_tie() -> None:
    objectives = (objective("a"), objective("b"))
    better = candidate("better", {"a": Score(0.8, 0.9, "a"), "b": Score(0.9, 0.9, "b")})
    worse = candidate("worse", {"a": Score(0.8, 0.9, "a"), "b": Score(0.6, 0.9, "b")})

    penalized = compare_candidates(better, worse, objectives, uncertainty_scale=0.05)
    unpenalized = compare_candidates(better, worse, objectives, uncertainty_scale=0.0)

    assert penalized.verdict == "ambiguous"
    assert penalized.reason == "confidence bounds overlap"
    assert unpenalized.verdict == "defended"
    assert unpenalized.chosen_id == "better"


def test_comparison_does_not_depend_on_argument_order() -> None:
    objectives = (objective("quality"), objective("latency", "minimize"))
    first = candidate("x", {"quality": Score(0.9, 1, "q"), "latency": Score(20, 1, "l")})
    second = candidate("y", {"quality": Score(0.7, 1, "q"), "latency": Score(50, 1, "l")})
    tradeoff = candidate("z", {"quality": Score(0.95, 1, "q"), "latency": Score(90, 1, "l")})

    assert compare_candidates(first, second, objectives) == compare_candidates(
        second, first, objectives
    )
    assert compare_candidates(first, tradeoff, objectives) == compare_candidates(
        tradeoff, first, objectives
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"epsilon": -0.1}, "epsilon"),
        ({"min_confidence": 1.5}, "min_confidence"),
        ({"uncertainty_scale": -0.1}, "uncertainty_scale"),
    ],
)
def test_comparison_rejects_invalid_thresholds(options: dict[str, float], message: str) -> None:
    objectives = (objective("a"),)
    left = candidate("left", {"a": Score(0.9, 1, "a")})
    right = candidate("right", {"a": Score(0.7, 1, "a")})

    with pytest.raises(ValueError, match=message):
        compare_candidates(left, right, objectives, **options)
