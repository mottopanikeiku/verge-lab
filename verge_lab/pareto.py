"""Confidence-bound, direction-normalized Pareto preference mining."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations

from .ids import stable_id
from .models import Candidate, Objective, Pair, Score


def direction_normalized(value: float, objective: Objective) -> float:
    """Map every objective to a larger-is-better coordinate."""

    return value if objective.direction == "maximize" else -value


def point_margin(left: Score, right: Score, objective: Objective) -> float:
    return direction_normalized(left.value, objective) - direction_normalized(
        right.value, objective
    )


def lower_confidence_margin(
    left: Score,
    right: Score,
    objective: Objective,
    *,
    uncertainty_scale: float = 0.05,
) -> float:
    """Conservative lower bound for left's normalized advantage.

    Confidence represents verifier reliability, not a fabricated sample count.
    The configurable scale converts each score's untrusted fraction into a
    bounded uncertainty radius in normalized score units.
    """

    if uncertainty_scale < 0:
        raise ValueError("uncertainty_scale must be non-negative")
    uncertainty = uncertainty_scale * ((1.0 - left.confidence) + (1.0 - right.confidence))
    return point_margin(left, right, objective) - uncertainty


def _reason(prefix: str, objective_ids: Iterable[str]) -> str:
    return f"{prefix}: {', '.join(sorted(objective_ids))}"


def compare_candidates(
    left: Candidate,
    right: Candidate,
    objectives: Iterable[Objective],
    *,
    epsilon: float = 0.01,
    min_confidence: float = 0.8,
    uncertainty_scale: float = 0.05,
) -> Pair:
    """Compare two candidates without inventing a winner when evidence is weak."""

    if left.prompt_id != right.prompt_id:
        raise ValueError("candidates from different prompts cannot be compared")
    if left.id == right.id:
        raise ValueError("a candidate cannot be compared with itself")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")

    ordered_objectives = tuple(sorted(objectives, key=lambda item: item.id))
    objective_ids = [item.id for item in ordered_objectives]
    first, second = sorted((left, right), key=lambda item: item.id)
    pair_id = stable_id("pair", left.prompt_id, first.id, second.id)

    def conservative_margins(candidate: Candidate, other: Candidate) -> dict[str, float]:
        return {
            item.id: lower_confidence_margin(
                candidate.scores[item.id],
                other.scores[item.id],
                item,
                uncertainty_scale=uncertainty_scale,
            )
            for item in ordered_objectives
            if item.id in candidate.scores and item.id in other.scores
        }

    stable_margins = {
        key: round(value, 6) for key, value in sorted(conservative_margins(first, second).items())
    }
    missing = [
        item for item in objective_ids if item not in left.scores or item not in right.scores
    ]
    if missing:
        return Pair(
            pair_id,
            left.prompt_id,
            first.id,
            second.id,
            "ambiguous",
            0.0,
            stable_margins,
            _reason("missing scores", missing),
        )

    low_confidence = [
        item.id
        for item in ordered_objectives
        if left.scores[item.id].confidence < min_confidence
        or right.scores[item.id].confidence < min_confidence
    ]
    confidence = min(
        min(left.scores[item.id].confidence, right.scores[item.id].confidence)
        for item in ordered_objectives
    )
    if low_confidence:
        return Pair(
            pair_id,
            left.prompt_id,
            first.id,
            second.id,
            "ambiguous",
            confidence,
            stable_margins,
            _reason("confidence below threshold", low_confidence),
        )

    left_bounds = conservative_margins(left, right)
    right_bounds = conservative_margins(right, left)

    def defended(margins: Mapping[str, float]) -> bool:
        return all(value >= 0.0 for value in margins.values()) and any(
            value > epsilon for value in margins.values()
        )

    if defended(left_bounds):
        margins = {key: round(value, 6) for key, value in sorted(left_bounds.items())}
        strict = sorted(key for key, value in left_bounds.items() if value > epsilon)
        return Pair(
            pair_id,
            left.prompt_id,
            left.id,
            right.id,
            "defended",
            confidence,
            margins,
            _reason("lower-bound Pareto dominance", strict),
        )
    if defended(right_bounds):
        margins = {key: round(value, 6) for key, value in sorted(right_bounds.items())}
        strict = sorted(key for key, value in right_bounds.items() if value > epsilon)
        return Pair(
            pair_id,
            left.prompt_id,
            right.id,
            left.id,
            "defended",
            confidence,
            margins,
            _reason("lower-bound Pareto dominance", strict),
        )

    observed = [
        point_margin(left.scores[item.id], right.scores[item.id], item)
        for item in ordered_objectives
    ]
    has_positive = any(value > epsilon for value in observed)
    has_negative = any(value < -epsilon for value in observed)
    if has_positive and has_negative:
        reason = "cross-objective tradeoff"
    elif all(abs(value) <= epsilon for value in observed):
        reason = "equivalent within epsilon"
    else:
        reason = "confidence bounds overlap"
    return Pair(
        pair_id,
        left.prompt_id,
        first.id,
        second.id,
        "ambiguous",
        confidence,
        stable_margins,
        reason,
    )


def mine_pareto_edges(
    candidates: Iterable[Candidate],
    objectives: Iterable[Objective],
    *,
    epsilon: float = 0.01,
    min_confidence: float = 0.8,
    uncertainty_scale: float = 0.05,
) -> tuple[Pair, ...]:
    """Mine every within-prompt unordered pair in stable order."""

    objectives = tuple(objectives)
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.prompt_id].append(candidate)
    pairs = []
    for prompt_id in sorted(grouped):
        ordered = sorted(grouped[prompt_id], key=lambda item: item.id)
        for left, right in combinations(ordered, 2):
            pairs.append(
                compare_candidates(
                    left,
                    right,
                    objectives,
                    epsilon=epsilon,
                    min_confidence=min_confidence,
                    uncertainty_scale=uncertainty_scale,
                )
            )
    return tuple(sorted(pairs, key=lambda item: (item.prompt_id, item.id)))
