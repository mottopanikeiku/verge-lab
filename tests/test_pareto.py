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
