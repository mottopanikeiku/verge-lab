"""Deterministic analysis and demo artifact construction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .ids import stable_id
from .models import (
    COLLECTION_LIMITS,
    MAX_ARTIFACT_BYTES,
    MAX_CANDIDATES_PER_PROMPT,
    MAX_PAIRS,
    Candidate,
    Checkpoint,
    Embedding,
    ModelInfo,
    Objective,
    Prompt,
    PromptSpecification,
    RunArtifact,
    Score,
    Summary,
)
from .mutations import audit_pair_mutation, mutate_output
from .pareto import mine_pareto_edges

DEMO_CREATED_AT = "2026-08-23T12:00:00Z"


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be an object")
    return value


def build_artifact(payload: Mapping[str, Any]) -> RunArtifact:
    """Build the canonical artifact from deterministic scored candidate input."""

    objective_rows = payload.get("objectives")
    prompt_rows = payload.get("prompts")
    candidate_rows = payload.get("candidates")
    if (
        not isinstance(objective_rows, list)
        or not isinstance(prompt_rows, list)
        or not isinstance(candidate_rows, list)
    ):
        raise TypeError("input objectives, prompts and candidates must be arrays")
    for label, rows in (
        ("objectives", objective_rows),
        ("prompts", prompt_rows),
        ("candidates", candidate_rows),
    ):
        if len(rows) > COLLECTION_LIMITS[label]:
            raise ValueError(f"input {label} exceeds {COLLECTION_LIMITS[label]} items")

    preliminary_objectives = [
        Objective(
            id=str(row["id"]),
            label=str(row["label"]),
            description=str(row["description"]),
            direction=str(row["direction"]),
            color=str(row["color"]),
        )
        for raw in objective_rows
        for row in [_mapping(raw, "objective")]
    ]
    objective_ids = {item.id for item in preliminary_objectives}

    prompt_by_key: dict[str, Prompt] = {}
    for raw in prompt_rows:
        row = _mapping(raw, "prompt")
        key = str(row["key"])
        specification = PromptSpecification.from_dict(row["specification"])
        prompt = Prompt(
            id=stable_id("prompt", row["text"], row["domain"], specification.to_dict()),
            text=str(row["text"]),
            domain=str(row["domain"]),
            specification=specification,
        )
        if key in prompt_by_key:
            raise ValueError(f"duplicate prompt key: {key}")
        prompt_by_key[key] = prompt

    candidate_by_key: dict[str, Candidate] = {}
    for raw in candidate_rows:
        row = _mapping(raw, "candidate")
        key = str(row["key"])
        prompt_key = str(row["promptKey"])
        if prompt_key not in prompt_by_key:
            raise ValueError(f"unknown prompt key: {prompt_key}")
        scores_raw = _mapping(row["scores"], "candidate.scores")
        scores = {str(name): Score.from_dict(value) for name, value in sorted(scores_raw.items())}
        unknown = set(scores) - objective_ids
        if unknown:
            raise ValueError(
                f"candidate {key} has unknown objectives: {', '.join(sorted(unknown))}"
            )
        prompt_id = prompt_by_key[prompt_key].id
        candidate = Candidate(
            id=stable_id("candidate", prompt_id, row["output"]),
            prompt_id=prompt_id,
            output=str(row["output"]),
            tokens=int(row["tokens"]),
            latency_ms=float(row["latencyMs"]),
            scores=scores,
            embedding=Embedding.from_dict(row["embedding"]),
        )
        if key in candidate_by_key:
            raise ValueError(f"duplicate candidate key: {key}")
        candidate_by_key[key] = candidate

    candidates = tuple(candidate_by_key[key] for key in sorted(candidate_by_key))
    candidates_per_prompt: dict[str, int] = {}
    for candidate in candidates:
        candidates_per_prompt[candidate.prompt_id] = (
            candidates_per_prompt.get(candidate.prompt_id, 0) + 1
        )
    if any(count > MAX_CANDIDATES_PER_PROMPT for count in candidates_per_prompt.values()):
        raise ValueError(f"input exceeds {MAX_CANDIDATES_PER_PROMPT} candidates for one prompt")
    potential_pair_count = sum(count * (count - 1) // 2 for count in candidates_per_prompt.values())
    if potential_pair_count > MAX_PAIRS:
        raise ValueError(f"input would produce more than {MAX_PAIRS} candidate pairs")

    objective_list = []
    for index, item in enumerate(preliminary_objectives):
        observed_values = [
            candidate.scores[item.id].value
            for candidate in candidates
            if item.id in candidate.scores
        ]
        if not observed_values:
            raise ValueError(f"objective {item.id} has no observed scores")
        objective_list.append(
            Objective(
                id=item.id,
                label=item.label,
                description=item.description,
                direction=item.direction,
                color=item.color,
                mean=round(sum(observed_values) / len(observed_values), 4),
                delta=float(_mapping(objective_rows[index], "objective").get("delta", 0.0)),
            )
        )
    objectives = tuple(objective_list)
    pairs = mine_pareto_edges(candidates, objectives)

    mutations = []
    mutation_rows = payload.get("mutations", [])
    if not isinstance(mutation_rows, list):
        raise TypeError("input mutations must be an array")
    if len(mutation_rows) > COLLECTION_LIMITS["mutations"]:
        raise ValueError(f"input mutations exceeds {COLLECTION_LIMITS['mutations']} items")
    for raw in mutation_rows:
        row = _mapping(raw, "mutation")
        source = candidate_by_key[str(row["sourceKey"])]
        opponent = candidate_by_key[str(row["opponentKey"])]
        kind = str(row["kind"])
        output = mutate_output(source.output, kind)
        declared_output = row.get("output")
        if declared_output is not None and str(declared_output) != output:
            raise ValueError(f"mutation output does not match registered mutator {kind}")
        scores = {
            str(name): Score.from_dict(value)
            for name, value in _mapping(row["scores"], "mutation.scores").items()
        }
        mutations.append(
            audit_pair_mutation(
                source,
                opponent,
                kind=kind,
                output=output,
                mutated_scores=scores,
                objectives=objectives,
            )
        )

    checkpoint_rows = payload.get("checkpoints", [])
    if not isinstance(checkpoint_rows, list):
        raise TypeError("input checkpoints must be an array")
    if len(checkpoint_rows) > COLLECTION_LIMITS["checkpoints"]:
        raise ValueError(f"input checkpoints exceeds {COLLECTION_LIMITS['checkpoints']} items")
    checkpoints = tuple(
        sorted((Checkpoint.from_dict(item) for item in checkpoint_rows), key=lambda item: item.step)
    )
    model = ModelInfo.from_dict(payload["model"])
    prompts = tuple(prompt_by_key[key] for key in sorted(prompt_by_key))
    mutations_tuple = tuple(sorted(mutations, key=lambda item: item.id))
    estimates = _mapping(payload.get("estimates", {}), "estimates")
    summary = Summary(
        prompt_count=len(prompts),
        candidate_count=len(candidates),
        defended_pair_count=sum(item.verdict == "defended" for item in pairs),
        ambiguous_pair_count=sum(item.verdict == "ambiguous" for item in pairs),
        mutation_flip_count=sum(item.flipped for item in mutations_tuple),
        estimated_gpu_minutes=float(estimates.get("gpuMinutes", 0.0)),
        estimated_cost_usd=float(estimates.get("costUsd", 0.0)),
    )
    name = str(payload.get("name", "Verge Lab analysis"))
    artifact_id = stable_id(
        "run",
        name,
        model.to_dict(),
        [item.to_dict() for item in objectives],
        [item.to_dict() for item in prompts],
        [item.to_dict() for item in candidates],
    )
    return RunArtifact(
        schema_version=1,
        id=artifact_id,
        name=name,
        created_at=str(payload.get("createdAt", DEMO_CREATED_AT)),
        status=str(payload.get("status", "complete")),
        model=model,
        summary=summary,
        objectives=objectives,
        prompts=prompts,
        candidates=candidates,
        pairs=pairs,
        mutations=mutations_tuple,
        checkpoints=checkpoints,
    )


def analyze_file(path: str | Path) -> RunArtifact:
    source = Path(path)
    if source.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"input exceeds {MAX_ARTIFACT_BYTES} bytes")
    value = json.loads(source.read_text(encoding="utf-8"))
    return build_artifact(_mapping(value, "input"))


def build_demo_artifact() -> RunArtifact:
    path = Path(__file__).resolve().parent.parent / "examples" / "candidates.json"
    if not path.exists():
        raise FileNotFoundError("demo input not found; pass an input file to analyze instead")
    return analyze_file(path)
