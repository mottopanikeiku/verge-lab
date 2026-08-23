"""Deterministic checker primitives for executable verifier evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CheckResult:
    passed: bool
    evidence: str


class Checker(Protocol):
    def check(self, output: str) -> CheckResult: ...


@dataclass(frozen=True, slots=True)
class ContainsAll:
    phrases: tuple[str, ...]
    case_sensitive: bool = False

    def check(self, output: str) -> CheckResult:
        haystack = output if self.case_sensitive else output.casefold()
        missing = [
            phrase
            for phrase in self.phrases
            if (phrase if self.case_sensitive else phrase.casefold()) not in haystack
        ]
        if missing:
            return CheckResult(False, "missing required phrases: " + ", ".join(sorted(missing)))
        return CheckResult(True, f"found all {len(self.phrases)} required phrases")


@dataclass(frozen=True, slots=True)
class JsonObjectFields:
    required_fields: tuple[str, ...]

    def check(self, output: str) -> CheckResult:
        try:
            value = json.loads(output)
        except (json.JSONDecodeError, ValueError) as error:
            return CheckResult(
                False,
                f"invalid JSON: {error.msg if isinstance(error, json.JSONDecodeError) else error}",
            )
        if not isinstance(value, Mapping):
            return CheckResult(False, "JSON value is not an object")
        missing = sorted(set(self.required_fields) - set(value))
        if missing:
            return CheckResult(False, "missing JSON fields: " + ", ".join(missing))
        return CheckResult(
            True, f"valid JSON object with {len(self.required_fields)} required fields"
        )


@dataclass(frozen=True, slots=True)
class FullMatch:
    pattern: str
    flags: int = 0

    def __post_init__(self) -> None:
        re.compile(self.pattern, self.flags)

    def check(self, output: str) -> CheckResult:
        passed = re.fullmatch(self.pattern, output, self.flags) is not None
        return CheckResult(
            passed,
            "output fully matches pattern" if passed else "output does not fully match pattern",
        )
