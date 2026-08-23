"""Reusable, declarative prompt-level reward specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .models import PromptSpecification

_ALLOWED_SCALAR_TYPES = (str, int, float, bool, type(None))


def _validate_json_value(value: Any, where: str) -> None:
    if isinstance(value, _ALLOWED_SCALAR_TYPES):
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{where}[{index}]")
        return
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _validate_json_value(item, f"{where}.{key}")
        return
    raise TypeError(f"{where} must contain only declarative JSON values")


@dataclass(frozen=True, slots=True)
class ScorerSpecification:
    """A safe declaration selecting a built-in scorer by name."""

    objective_id: str
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective_id or not self.kind:
            raise ValueError("objective_id and kind must be non-empty")
        _validate_json_value(self.parameters, "scorer.parameters")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class RewardSpecification:
    id: str
    rubrics: tuple[str, ...]
    constraints: tuple[str, ...]
    scorers: tuple[ScorerSpecification, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("specification ID must be non-empty")
        if not self.rubrics or any(not item for item in self.rubrics):
            raise ValueError("at least one non-empty rubric is required")
        if any(not item for item in self.constraints):
            raise ValueError("constraints must be non-empty strings")
        objective_ids = [item.objective_id for item in self.scorers]
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("a specification may define only one scorer per objective")

    def artifact_specification(self) -> PromptSpecification:
        return PromptSpecification(rubrics=self.rubrics, constraints=self.constraints)


class SpecificationLibrary:
    """An explicit registry for sharing specifications across prompts."""

    def __init__(self, specifications: tuple[RewardSpecification, ...] = ()) -> None:
        self._items: dict[str, RewardSpecification] = {}
        for specification in specifications:
            self.register(specification)

    def register(self, specification: RewardSpecification) -> None:
        if specification.id in self._items:
            raise ValueError(f"duplicate specification ID: {specification.id}")
        self._items[specification.id] = specification

    def resolve(self, specification_id: str) -> RewardSpecification:
        try:
            return self._items[specification_id]
        except KeyError as error:
            raise KeyError(f"unknown specification ID: {specification_id}") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
