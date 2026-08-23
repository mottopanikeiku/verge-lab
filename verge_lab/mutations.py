"""Meaning-preserving output mutation and preference-flip audit utilities."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

from .ids import stable_id
from .models import Candidate, Mutation, Objective, Score
from .pareto import compare_candidates


def _collapse_whitespace(output: str) -> str:
    return re.sub(r"[ \t]+", " ", output).strip()


def _remove_markdown_emphasis(output: str) -> str:
    return output.replace("**", "").replace("__", "")


def _plain_headings(output: str) -> str:
    return "\n".join(re.sub(r"^#{1,6}\s+", "", line) for line in output.splitlines())


MUTATORS: Mapping[str, Callable[[str], str]] = {
    "collapse-whitespace": _collapse_whitespace,
    "plain-headings": _plain_headings,
    "remove-emphasis": _remove_markdown_emphasis,
}


def mutate_output(output: str, kind: str) -> str:
    """Apply one known textual mutation; arbitrary mutation code is unsupported."""

    try:
        mutation = MUTATORS[kind]
    except KeyError as error:
        raise ValueError(f"unknown mutation kind {kind!r}") from error
    result = mutation(output)
    if not result:
        raise ValueError("mutation produced empty output")
    return result


def audit_pair_mutation(
    source: Candidate,
    opponent: Candidate,
    *,
    kind: str,
    output: str,
    mutated_scores: Mapping[str, Score],
    objectives: Iterable[Objective],
    epsilon: float = 0.01,
    min_confidence: float = 0.8,
    uncertainty_scale: float = 0.05,
) -> Mutation:
    """Audit whether a presentation mutation reverses a defended preference."""

    if source.prompt_id != opponent.prompt_id:
        raise ValueError("mutation opponent must belong to the source prompt")
    objective_tuple = tuple(objectives)
    mutation_id = stable_id("mut", source.id, kind, output)
    mutated = Candidate(
        id=stable_id("candidate", source.prompt_id, output),
        prompt_id=source.prompt_id,
        output=output,
        tokens=len(output.split()),
        latency_ms=source.latency_ms,
        scores=dict(mutated_scores),
        embedding=source.embedding,
    )
    before = compare_candidates(
        source,
        opponent,
        objective_tuple,
        epsilon=epsilon,
        min_confidence=min_confidence,
        uncertainty_scale=uncertainty_scale,
    )
    after = compare_candidates(
        mutated,
        opponent,
        objective_tuple,
        epsilon=epsilon,
        min_confidence=min_confidence,
        uncertainty_scale=uncertainty_scale,
    )
    before_source_wins = before.verdict == "defended" and before.chosen_id == source.id
    before_opponent_wins = before.verdict == "defended" and before.chosen_id == opponent.id
    after_source_wins = after.verdict == "defended" and after.chosen_id == mutated.id
    after_opponent_wins = after.verdict == "defended" and after.chosen_id == opponent.id
    flipped = (before_source_wins and after_opponent_wins) or (
        before_opponent_wins and after_source_wins
    )
    if flipped:
        reason = f"defended preference reversed after {kind}"
    elif before.verdict == "defended" and after.verdict == "ambiguous":
        reason = f"defended preference became ambiguous after {kind}"
    elif before.verdict == "ambiguous" and after.verdict == "defended":
        reason = f"ambiguous preference became defended after {kind}"
    else:
        reason = f"preference verdict stable after {kind}"
    deltas = {
        item.id: round(mutated_scores[item.id].value - source.scores[item.id].value, 6)
        for item in sorted(objective_tuple, key=lambda value: value.id)
        if item.id in mutated_scores and item.id in source.scores
    }
    return Mutation(mutation_id, source.id, kind, output, deltas, flipped, reason)
