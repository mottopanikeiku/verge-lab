"""Built-in scorers selected from declarative prompt specifications."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .checkers import ContainsAll, JsonObjectFields
from .models import Score
from .specs import ScorerSpecification


class Scorer(Protocol):
    def score(self, output: str) -> Score: ...


@dataclass(frozen=True, slots=True)
class KeywordCoverageScorer:
    keywords: tuple[str, ...]
    case_sensitive: bool = False

    def score(self, output: str) -> Score:
        if not self.keywords:
            raise ValueError("keyword coverage requires at least one keyword")
        haystack = output if self.case_sensitive else output.casefold()
        hits = sum(
            (item if self.case_sensitive else item.casefold()) in haystack for item in self.keywords
        )
        value = hits / len(self.keywords)
        return Score(value, 1.0, f"matched {hits}/{len(self.keywords)} declared keywords")


@dataclass(frozen=True, slots=True)
class RequiredPhrasesScorer:
    phrases: tuple[str, ...]
    case_sensitive: bool = False

    def score(self, output: str) -> Score:
        result = ContainsAll(self.phrases, self.case_sensitive).check(output)
        return Score(1.0 if result.passed else 0.0, 1.0, result.evidence)


@dataclass(frozen=True, slots=True)
class JsonFieldsScorer:
    fields: tuple[str, ...]

    def score(self, output: str) -> Score:
        result = JsonObjectFields(self.fields).check(output)
        return Score(1.0 if result.passed else 0.0, 1.0, result.evidence)


@dataclass(frozen=True, slots=True)
class LengthTargetScorer:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError("length target requires 0 <= minimum <= maximum")

    def score(self, output: str) -> Score:
        words = len(output.split())
        if self.minimum <= words <= self.maximum:
            return Score(1.0, 1.0, f"word count {words} is within [{self.minimum}, {self.maximum}]")
        distance = self.minimum - words if words < self.minimum else words - self.maximum
        scale = max(self.maximum - self.minimum, 1)
        value = max(0.0, 1.0 - distance / scale)
        return Score(value, 1.0, f"word count {words} is outside [{self.minimum}, {self.maximum}]")


def _strings(parameters: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = parameters.get(name)
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty array of strings")
    return tuple(value)


def _build_keyword(parameters: Mapping[str, Any]) -> Scorer:
    return KeywordCoverageScorer(
        _strings(parameters, "keywords"), bool(parameters.get("caseSensitive", False))
    )


def _build_phrases(parameters: Mapping[str, Any]) -> Scorer:
    return RequiredPhrasesScorer(
        _strings(parameters, "phrases"), bool(parameters.get("caseSensitive", False))
    )


def _build_json(parameters: Mapping[str, Any]) -> Scorer:
    return JsonFieldsScorer(_strings(parameters, "fields"))


def _build_length(parameters: Mapping[str, Any]) -> Scorer:
    minimum = parameters.get("minimum")
    maximum = parameters.get("maximum")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
    ):
        raise ValueError("minimum and maximum must be integers")
    return LengthTargetScorer(minimum, maximum)


_BUILDERS: dict[str, Callable[[Mapping[str, Any]], Scorer]] = {
    "json_fields": _build_json,
    "keyword_coverage": _build_keyword,
    "length_target": _build_length,
    "required_phrases": _build_phrases,
}


def build_scorer(specification: ScorerSpecification) -> Scorer:
    """Resolve only a fixed built-in scorer; never evaluate specification text."""

    try:
        builder = _BUILDERS[specification.kind]
    except KeyError as error:
        allowed = ", ".join(sorted(_BUILDERS))
        raise ValueError(
            f"unknown scorer kind {specification.kind!r}; expected one of {allowed}"
        ) from error
    return builder(specification.parameters)


def score_output(specifications: tuple[ScorerSpecification, ...], output: str) -> dict[str, Score]:
    return {item.objective_id: build_scorer(item).score(output) for item in specifications}
