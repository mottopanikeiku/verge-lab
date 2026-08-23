"""Dependency-free models for the canonical Verge Lab artifact schema."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_OBJECTIVES = 32
MAX_PROMPTS = 1_024
MAX_CANDIDATES = 8_192
MAX_CANDIDATES_PER_PROMPT = 64
MAX_PAIRS = 100_000
MAX_MUTATIONS = 8_192
MAX_CHECKPOINTS = 8_192
MAX_TEXT_CHARS = 131_072

COLLECTION_LIMITS = {
    "objectives": MAX_OBJECTIVES,
    "prompts": MAX_PROMPTS,
    "candidates": MAX_CANDIDATES,
    "pairs": MAX_PAIRS,
    "mutations": MAX_MUTATIONS,
    "checkpoints": MAX_CHECKPOINTS,
}


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{where} must be a JSON object with string keys")
    return value


def _keys(value: Mapping[str, Any], required: set[str], where: str) -> None:
    present = set(value)
    missing = sorted(required - present)
    unknown = sorted(present - required)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown {', '.join(unknown)}")
        raise ValueError(f"{where}: {'; '.join(parts)}")


def _string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TypeError(f"{where} must be a non-empty string")
    if len(value) > MAX_TEXT_CHARS:
        raise ValueError(f"{where} exceeds {MAX_TEXT_CHARS} characters")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where} must be an integer")
    return value


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{where} must be an array")
    return tuple(_string(item, f"{where}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class Score:
    value: float
    confidence: float
    evidence: str

    def __post_init__(self) -> None:
        _number(self.value, "score.value")
        confidence = _number(self.confidence, "score.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("score.confidence must be between 0 and 1")
        _string(self.evidence, "score.evidence")

    def to_dict(self) -> JsonObject:
        return {"value": self.value, "confidence": self.confidence, "evidence": self.evidence}

    @classmethod
    def from_dict(cls, raw: Any) -> Score:
        value = _object(raw, "score")
        _keys(value, {"value", "confidence", "evidence"}, "score")
        return cls(
            _number(value["value"], "score.value"),
            _number(value["confidence"], "score.confidence"),
            _string(value["evidence"], "score.evidence"),
        )


@dataclass(frozen=True, slots=True)
class Objective:
    id: str
    label: str
    description: str
    direction: str
    color: str
    mean: float = 0.0
    delta: float = 0.0

    def __post_init__(self) -> None:
        _string(self.id, "objective.id")
        _string(self.label, "objective.label")
        _string(self.description, "objective.description")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("objective.direction must be 'maximize' or 'minimize'")
        _string(self.color, "objective.color")
        _number(self.mean, "objective.mean")
        _number(self.delta, "objective.delta")

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "direction": self.direction,
            "color": self.color,
            "mean": self.mean,
            "delta": self.delta,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Objective:
        value = _object(raw, "objective")
        fields = {"id", "label", "description", "direction", "color", "mean", "delta"}
        _keys(value, fields, "objective")
        return cls(
            id=_string(value["id"], "objective.id"),
            label=_string(value["label"], "objective.label"),
            description=_string(value["description"], "objective.description"),
            direction=_string(value["direction"], "objective.direction"),
            color=_string(value["color"], "objective.color"),
            mean=_number(value["mean"], "objective.mean"),
            delta=_number(value["delta"], "objective.delta"),
        )


@dataclass(frozen=True, slots=True)
class PromptSpecification:
    rubrics: tuple[str, ...]
    constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.rubrics:
            raise ValueError("specification.rubrics must not be empty")
        for index, item in enumerate(self.rubrics):
            _string(item, f"specification.rubrics[{index}]")
        for index, item in enumerate(self.constraints):
            _string(item, f"specification.constraints[{index}]")

    def to_dict(self) -> JsonObject:
        return {"rubrics": list(self.rubrics), "constraints": list(self.constraints)}

    @classmethod
    def from_dict(cls, raw: Any) -> PromptSpecification:
        value = _object(raw, "specification")
        _keys(value, {"rubrics", "constraints"}, "specification")
        return cls(
            rubrics=_strings(value["rubrics"], "specification.rubrics"),
            constraints=_strings(value["constraints"], "specification.constraints"),
        )


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str
    text: str
    domain: str
    specification: PromptSpecification

    def __post_init__(self) -> None:
        _string(self.id, "prompt.id")
        _string(self.text, "prompt.text")
        _string(self.domain, "prompt.domain")

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "text": self.text,
            "domain": self.domain,
            "specification": self.specification.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Prompt:
        value = _object(raw, "prompt")
        _keys(value, {"id", "text", "domain", "specification"}, "prompt")
        return cls(
            id=_string(value["id"], "prompt.id"),
            text=_string(value["text"], "prompt.text"),
            domain=_string(value["domain"], "prompt.domain"),
            specification=PromptSpecification.from_dict(value["specification"]),
        )


@dataclass(frozen=True, slots=True)
class Embedding:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        _number(self.x, "embedding.x")
        _number(self.y, "embedding.y")
        _number(self.z, "embedding.z")

    def to_dict(self) -> JsonObject:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, raw: Any) -> Embedding:
        value = _object(raw, "embedding")
        _keys(value, {"x", "y", "z"}, "embedding")
        return cls(
            _number(value["x"], "embedding.x"),
            _number(value["y"], "embedding.y"),
            _number(value["z"], "embedding.z"),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    prompt_id: str
    output: str
    tokens: int
    latency_ms: float
    scores: Mapping[str, Score]
    embedding: Embedding

    def __post_init__(self) -> None:
        _string(self.id, "candidate.id")
        _string(self.prompt_id, "candidate.promptId")
        _string(self.output, "candidate.output")
        if _integer(self.tokens, "candidate.tokens") < 0:
            raise ValueError("candidate.tokens must be non-negative")
        if _number(self.latency_ms, "candidate.latencyMs") < 0:
            raise ValueError("candidate.latencyMs must be non-negative")
        if not isinstance(self.scores, Mapping) or not self.scores:
            raise TypeError("candidate.scores must be a non-empty mapping")
        if any(
            not isinstance(key, str) or not isinstance(score, Score)
            for key, score in self.scores.items()
        ):
            raise TypeError("candidate.scores must map objective IDs to Score values")

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "promptId": self.prompt_id,
            "output": self.output,
            "tokens": self.tokens,
            "latencyMs": self.latency_ms,
            "scores": {key: self.scores[key].to_dict() for key in sorted(self.scores)},
            "embedding": self.embedding.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Candidate:
        value = _object(raw, "candidate")
        fields = {"id", "promptId", "output", "tokens", "latencyMs", "scores", "embedding"}
        _keys(value, fields, "candidate")
        score_values = _object(value["scores"], "candidate.scores")
        return cls(
            id=_string(value["id"], "candidate.id"),
            prompt_id=_string(value["promptId"], "candidate.promptId"),
            output=_string(value["output"], "candidate.output"),
            tokens=_integer(value["tokens"], "candidate.tokens"),
            latency_ms=_number(value["latencyMs"], "candidate.latencyMs"),
            scores={key: Score.from_dict(item) for key, item in sorted(score_values.items())},
            embedding=Embedding.from_dict(value["embedding"]),
        )


@dataclass(frozen=True, slots=True)
class Pair:
    id: str
    prompt_id: str
    chosen_id: str
    rejected_id: str
    verdict: str
    confidence: float
    margins: Mapping[str, float]
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("id", self.id),
            ("promptId", self.prompt_id),
            ("chosenId", self.chosen_id),
            ("rejectedId", self.rejected_id),
        ):
            _string(value, f"pair.{label}")
        if self.chosen_id == self.rejected_id:
            raise ValueError("pair candidates must be distinct")
        if self.verdict not in {"defended", "ambiguous"}:
            raise ValueError("pair.verdict must be 'defended' or 'ambiguous'")
        confidence = _number(self.confidence, "pair.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("pair.confidence must be between 0 and 1")
        if not isinstance(self.margins, Mapping):
            raise TypeError("pair.margins must be a mapping")
        for key, margin in self.margins.items():
            _string(key, "pair.margins key")
            _number(margin, f"pair.margins.{key}")
        _string(self.reason, "pair.reason")

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "promptId": self.prompt_id,
            "chosenId": self.chosen_id,
            "rejectedId": self.rejected_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "margins": {key: self.margins[key] for key in sorted(self.margins)},
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Pair:
        value = _object(raw, "pair")
        fields = {
            "id",
            "promptId",
            "chosenId",
            "rejectedId",
            "verdict",
            "confidence",
            "margins",
            "reason",
        }
        _keys(value, fields, "pair")
        margins = _object(value["margins"], "pair.margins")
        return cls(
            id=_string(value["id"], "pair.id"),
            prompt_id=_string(value["promptId"], "pair.promptId"),
            chosen_id=_string(value["chosenId"], "pair.chosenId"),
            rejected_id=_string(value["rejectedId"], "pair.rejectedId"),
            verdict=_string(value["verdict"], "pair.verdict"),
            confidence=_number(value["confidence"], "pair.confidence"),
            margins={
                key: _number(item, f"pair.margins.{key}") for key, item in sorted(margins.items())
            },
            reason=_string(value["reason"], "pair.reason"),
        )


@dataclass(frozen=True, slots=True)
class Mutation:
    id: str
    source_candidate_id: str
    kind: str
    output: str
    score_delta: Mapping[str, float]
    flipped: bool
    reason: str

    def __post_init__(self) -> None:
        _string(self.id, "mutation.id")
        _string(self.source_candidate_id, "mutation.sourceCandidateId")
        _string(self.kind, "mutation.kind")
        _string(self.output, "mutation.output")
        if not isinstance(self.score_delta, Mapping):
            raise TypeError("mutation.scoreDelta must be a mapping")
        for key, delta in self.score_delta.items():
            _string(key, "mutation.scoreDelta key")
            _number(delta, f"mutation.scoreDelta.{key}")
        if not isinstance(self.flipped, bool):
            raise TypeError("mutation.flipped must be a boolean")
        _string(self.reason, "mutation.reason")

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "sourceCandidateId": self.source_candidate_id,
            "kind": self.kind,
            "output": self.output,
            "scoreDelta": {key: self.score_delta[key] for key in sorted(self.score_delta)},
            "flipped": self.flipped,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Mutation:
        value = _object(raw, "mutation")
        fields = {"id", "sourceCandidateId", "kind", "output", "scoreDelta", "flipped", "reason"}
        _keys(value, fields, "mutation")
        deltas = _object(value["scoreDelta"], "mutation.scoreDelta")
        flipped = value["flipped"]
        if not isinstance(flipped, bool):
            raise TypeError("mutation.flipped must be a boolean")
        return cls(
            id=_string(value["id"], "mutation.id"),
            source_candidate_id=_string(value["sourceCandidateId"], "mutation.sourceCandidateId"),
            kind=_string(value["kind"], "mutation.kind"),
            output=_string(value["output"], "mutation.output"),
            score_delta={
                key: _number(item, f"mutation.scoreDelta.{key}")
                for key, item in sorted(deltas.items())
            },
            flipped=flipped,
            reason=_string(value["reason"], "mutation.reason"),
        )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    step: int
    label: str
    train_loss: float
    eval_reward: float
    defended_win_rate: float
    gpu_minutes: float

    def __post_init__(self) -> None:
        if _integer(self.step, "checkpoint.step") < 0:
            raise ValueError("checkpoint.step must be non-negative")
        _string(self.label, "checkpoint.label")
        _number(self.train_loss, "checkpoint.trainLoss")
        _number(self.eval_reward, "checkpoint.evalReward")
        win_rate = _number(self.defended_win_rate, "checkpoint.defendedWinRate")
        if not 0.0 <= win_rate <= 1.0:
            raise ValueError("checkpoint.defendedWinRate must be between 0 and 1")
        if _number(self.gpu_minutes, "checkpoint.gpuMinutes") < 0:
            raise ValueError("checkpoint.gpuMinutes must be non-negative")

    def to_dict(self) -> JsonObject:
        return {
            "step": self.step,
            "label": self.label,
            "trainLoss": self.train_loss,
            "evalReward": self.eval_reward,
            "defendedWinRate": self.defended_win_rate,
            "gpuMinutes": self.gpu_minutes,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Checkpoint:
        value = _object(raw, "checkpoint")
        fields = {"step", "label", "trainLoss", "evalReward", "defendedWinRate", "gpuMinutes"}
        _keys(value, fields, "checkpoint")
        return cls(
            step=_integer(value["step"], "checkpoint.step"),
            label=_string(value["label"], "checkpoint.label"),
            train_loss=_number(value["trainLoss"], "checkpoint.trainLoss"),
            eval_reward=_number(value["evalReward"], "checkpoint.evalReward"),
            defended_win_rate=_number(value["defendedWinRate"], "checkpoint.defendedWinRate"),
            gpu_minutes=_number(value["gpuMinutes"], "checkpoint.gpuMinutes"),
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    base: str
    method: str
    adapter: str
    parameter_count: int
    quantization: str

    def __post_init__(self) -> None:
        _string(self.base, "model.base")
        _string(self.method, "model.method")
        _string(self.adapter, "model.adapter")
        if _integer(self.parameter_count, "model.parameterCount") <= 0:
            raise ValueError("model.parameterCount must be positive")
        _string(self.quantization, "model.quantization")

    def to_dict(self) -> JsonObject:
        return {
            "base": self.base,
            "method": self.method,
            "adapter": self.adapter,
            "parameterCount": self.parameter_count,
            "quantization": self.quantization,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> ModelInfo:
        value = _object(raw, "model")
        fields = {"base", "method", "adapter", "parameterCount", "quantization"}
        _keys(value, fields, "model")
        return cls(
            base=_string(value["base"], "model.base"),
            method=_string(value["method"], "model.method"),
            adapter=_string(value["adapter"], "model.adapter"),
            parameter_count=_integer(value["parameterCount"], "model.parameterCount"),
            quantization=_string(value["quantization"], "model.quantization"),
        )


@dataclass(frozen=True, slots=True)
class Summary:
    prompt_count: int
    candidate_count: int
    defended_pair_count: int
    ambiguous_pair_count: int
    mutation_flip_count: int
    estimated_gpu_minutes: float
    estimated_cost_usd: float

    def __post_init__(self) -> None:
        for name, value in (
            ("promptCount", self.prompt_count),
            ("candidateCount", self.candidate_count),
            ("defendedPairCount", self.defended_pair_count),
            ("ambiguousPairCount", self.ambiguous_pair_count),
            ("mutationFlipCount", self.mutation_flip_count),
        ):
            if _integer(value, f"summary.{name}") < 0:
                raise ValueError(f"summary.{name} must be non-negative")
        if _number(self.estimated_gpu_minutes, "summary.estimatedGpuMinutes") < 0:
            raise ValueError("summary.estimatedGpuMinutes must be non-negative")
        if _number(self.estimated_cost_usd, "summary.estimatedCostUsd") < 0:
            raise ValueError("summary.estimatedCostUsd must be non-negative")

    def to_dict(self) -> JsonObject:
        return {
            "promptCount": self.prompt_count,
            "candidateCount": self.candidate_count,
            "defendedPairCount": self.defended_pair_count,
            "ambiguousPairCount": self.ambiguous_pair_count,
            "mutationFlipCount": self.mutation_flip_count,
            "estimatedGpuMinutes": self.estimated_gpu_minutes,
            "estimatedCostUsd": self.estimated_cost_usd,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Summary:
        value = _object(raw, "summary")
        fields = {
            "promptCount",
            "candidateCount",
            "defendedPairCount",
            "ambiguousPairCount",
            "mutationFlipCount",
            "estimatedGpuMinutes",
            "estimatedCostUsd",
        }
        _keys(value, fields, "summary")
        return cls(
            prompt_count=_integer(value["promptCount"], "summary.promptCount"),
            candidate_count=_integer(value["candidateCount"], "summary.candidateCount"),
            defended_pair_count=_integer(value["defendedPairCount"], "summary.defendedPairCount"),
            ambiguous_pair_count=_integer(
                value["ambiguousPairCount"], "summary.ambiguousPairCount"
            ),
            mutation_flip_count=_integer(value["mutationFlipCount"], "summary.mutationFlipCount"),
            estimated_gpu_minutes=_number(
                value["estimatedGpuMinutes"], "summary.estimatedGpuMinutes"
            ),
            estimated_cost_usd=_number(value["estimatedCostUsd"], "summary.estimatedCostUsd"),
        )


@dataclass(frozen=True, slots=True)
class RunArtifact:
    schema_version: int
    id: str
    name: str
    created_at: str
    status: str
    model: ModelInfo
    summary: Summary
    objectives: tuple[Objective, ...]
    prompts: tuple[Prompt, ...]
    candidates: tuple[Candidate, ...]
    pairs: tuple[Pair, ...]
    mutations: tuple[Mutation, ...]
    checkpoints: tuple[Checkpoint, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schemaVersion must be 1")
        _string(self.id, "artifact.id")
        _string(self.name, "artifact.name")
        created_at = _string(self.created_at, "artifact.createdAt")
        if not created_at.endswith("Z"):
            raise ValueError("artifact.createdAt must be an RFC 3339 UTC timestamp")
        try:
            datetime.fromisoformat(created_at[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError("artifact.createdAt must be an RFC 3339 UTC timestamp") from error
        _string(self.status, "artifact.status")
        self.validate()

    def validate(self) -> None:
        for label, values in (
            ("objectives", self.objectives),
            ("prompts", self.prompts),
            ("candidates", self.candidates),
            ("pairs", self.pairs),
            ("mutations", self.mutations),
            ("checkpoints", self.checkpoints),
        ):
            if len(values) > COLLECTION_LIMITS[label]:
                raise ValueError(f"artifact.{label} exceeds {COLLECTION_LIMITS[label]} items")
        candidates_per_prompt: dict[str, int] = {}
        for candidate in self.candidates:
            candidates_per_prompt[candidate.prompt_id] = (
                candidates_per_prompt.get(candidate.prompt_id, 0) + 1
            )
        if any(count > MAX_CANDIDATES_PER_PROMPT for count in candidates_per_prompt.values()):
            raise ValueError(
                f"artifact exceeds {MAX_CANDIDATES_PER_PROMPT} candidates for one prompt"
            )
        objective_ids = [item.id for item in self.objectives]
        prompt_ids = [item.id for item in self.prompts]
        candidate_ids = [item.id for item in self.candidates]
        for label, values in (
            ("objective", objective_ids),
            ("prompt", prompt_ids),
            ("candidate", candidate_ids),
            ("pair", [item.id for item in self.pairs]),
            ("mutation", [item.id for item in self.mutations]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} ID")
        objective_set = set(objective_ids)
        prompt_set = set(prompt_ids)
        candidate_set = set(candidate_ids)
        if not objective_set or not prompt_set:
            raise ValueError("artifact must include objectives and prompts")
        for candidate in self.candidates:
            if candidate.prompt_id not in prompt_set:
                raise ValueError(f"candidate {candidate.id} references unknown prompt")
            unknown_scores = set(candidate.scores) - objective_set
            if unknown_scores:
                raise ValueError(f"candidate {candidate.id} has unknown score objectives")
        candidate_by_id = {item.id: item for item in self.candidates}
        for pair in self.pairs:
            if (
                pair.prompt_id not in prompt_set
                or pair.chosen_id not in candidate_set
                or pair.rejected_id not in candidate_set
            ):
                raise ValueError(f"pair {pair.id} has an unknown reference")
            if (
                candidate_by_id[pair.chosen_id].prompt_id != pair.prompt_id
                or candidate_by_id[pair.rejected_id].prompt_id != pair.prompt_id
            ):
                raise ValueError(f"pair {pair.id} crosses prompts")
            if set(pair.margins) - objective_set:
                raise ValueError(f"pair {pair.id} has an unknown margin objective")
        for mutation in self.mutations:
            if mutation.source_candidate_id not in candidate_set:
                raise ValueError(f"mutation {mutation.id} references unknown candidate")
            if set(mutation.score_delta) - objective_set:
                raise ValueError(f"mutation {mutation.id} has an unknown delta objective")
        expected = self.derived_summary(
            estimated_gpu_minutes=self.summary.estimated_gpu_minutes,
            estimated_cost_usd=self.summary.estimated_cost_usd,
        )
        if self.summary != expected:
            raise ValueError("artifact summary counts do not match artifact contents")
        if tuple(sorted(self.checkpoints, key=lambda item: item.step)) != self.checkpoints:
            raise ValueError("checkpoints must be ordered by step")

    def derived_summary(
        self, *, estimated_gpu_minutes: float, estimated_cost_usd: float
    ) -> Summary:
        return Summary(
            prompt_count=len(self.prompts),
            candidate_count=len(self.candidates),
            defended_pair_count=sum(pair.verdict == "defended" for pair in self.pairs),
            ambiguous_pair_count=sum(pair.verdict == "ambiguous" for pair in self.pairs),
            mutation_flip_count=sum(mutation.flipped for mutation in self.mutations),
            estimated_gpu_minutes=estimated_gpu_minutes,
            estimated_cost_usd=estimated_cost_usd,
        )

    def to_dict(self) -> JsonObject:
        return {
            "schemaVersion": self.schema_version,
            "id": self.id,
            "name": self.name,
            "createdAt": self.created_at,
            "status": self.status,
            "model": self.model.to_dict(),
            "summary": self.summary.to_dict(),
            "objectives": [item.to_dict() for item in self.objectives],
            "prompts": [item.to_dict() for item in self.prompts],
            "candidates": [item.to_dict() for item in self.candidates],
            "pairs": [item.to_dict() for item in self.pairs],
            "mutations": [item.to_dict() for item in self.mutations],
            "checkpoints": [item.to_dict() for item in self.checkpoints],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent) + (
            "\n" if indent is not None else ""
        )

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, raw: Any) -> RunArtifact:
        value = _object(raw, "artifact")
        fields = {
            "schemaVersion",
            "id",
            "name",
            "createdAt",
            "status",
            "model",
            "summary",
            "objectives",
            "prompts",
            "candidates",
            "pairs",
            "mutations",
            "checkpoints",
        }
        _keys(value, fields, "artifact")
        arrays: dict[str, list[Any]] = {}
        for key in ("objectives", "prompts", "candidates", "pairs", "mutations", "checkpoints"):
            item = value[key]
            if not isinstance(item, list):
                raise TypeError(f"artifact.{key} must be an array")
            if len(item) > COLLECTION_LIMITS[key]:
                raise ValueError(f"artifact.{key} exceeds {COLLECTION_LIMITS[key]} items")
            arrays[key] = item
        return cls(
            schema_version=_integer(value["schemaVersion"], "artifact.schemaVersion"),
            id=_string(value["id"], "artifact.id"),
            name=_string(value["name"], "artifact.name"),
            created_at=_string(value["createdAt"], "artifact.createdAt"),
            status=_string(value["status"], "artifact.status"),
            model=ModelInfo.from_dict(value["model"]),
            summary=Summary.from_dict(value["summary"]),
            objectives=tuple(Objective.from_dict(item) for item in arrays["objectives"]),
            prompts=tuple(Prompt.from_dict(item) for item in arrays["prompts"]),
            candidates=tuple(Candidate.from_dict(item) for item in arrays["candidates"]),
            pairs=tuple(Pair.from_dict(item) for item in arrays["pairs"]),
            mutations=tuple(Mutation.from_dict(item) for item in arrays["mutations"]),
            checkpoints=tuple(Checkpoint.from_dict(item) for item in arrays["checkpoints"]),
        )

    @classmethod
    def from_json(cls, text: str) -> RunArtifact:
        if len(text.encode("utf-8")) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON number {value} is not allowed")

        return cls.from_dict(json.loads(text, parse_constant=reject_constant))

    @classmethod
    def read_json(cls, path: str | Path) -> RunArtifact:
        source = Path(path)
        if source.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
        return cls.from_json(source.read_text(encoding="utf-8"))
