"""Eval cases: what a model is given and what a correct response looks like."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OPERATORS = {"equals", "contains", "regex", "approx", "any", "one_of", "same_items", "absent", "type"}
_CASE_KEYS = {"name", "prompt", "system", "expect", "followup", "tags"}
_EXPECT_KEYS = {"tool", "args", "calls", "no_tool", "allow_extra", "forbid", "any_of"}
_ALTERNATIVE_KEYS = {"tool", "args", "calls", "allow_extra"}
_FOLLOWUP_KEYS = {"results", "forbid", "rubric"}


class CaseError(Exception):
    pass


@dataclass
class ExpectedCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Followup:
    results: dict[str, str] = field(default_factory=dict)
    forbid: list[str] = field(default_factory=list)
    rubric: str | None = None


@dataclass
class Case:
    name: str
    prompt: str
    system: str | None = None
    calls: list[ExpectedCall] = field(default_factory=list)
    no_tool: bool = False
    allow_extra: bool = False
    forbid: list[str] = field(default_factory=list)
    followup: Followup | None = None
    tags: list[str] = field(default_factory=list)
    alternatives: list[tuple[list[ExpectedCall], bool]] = field(default_factory=list)

    def variants(self) -> list[tuple[list[ExpectedCall], bool]]:
        """Every accepted way to do the task, as (expected calls, allow_extra)."""
        if self.alternatives:
            return self.alternatives
        return [(self.calls, self.allow_extra)] if self.calls else []


@dataclass
class CaseSet:
    cases: list[Case]
    system: str | None = None


def _check_keys(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise CaseError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _string_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise CaseError(f"{where} must be a list of strings")
    return list(value)


def _validate_matcher(matcher: Any, where: str) -> None:
    if isinstance(matcher, dict) and matcher and set(matcher) & OPERATORS:
        ops = set(matcher) & OPERATORS
        if len(ops) != 1:
            raise CaseError(f"{where}: use exactly one operator, got {sorted(ops)}")
        op = next(iter(ops))
        extra = set(matcher) - {op, "ignore_case", "tol"}
        if extra:
            raise CaseError(f"{where}: unexpected key(s) {sorted(extra)} next to '{op}'")
        if op == "regex":
            try:
                re.compile(matcher["regex"])
            except (re.error, TypeError) as exc:
                raise CaseError(f"{where}: invalid regex: {exc}") from exc
        if op == "approx" and not isinstance(matcher["approx"], (int, float)):
            raise CaseError(f"{where}: approx needs a number")
        if op in ("one_of", "same_items") and not isinstance(matcher[op], list):
            raise CaseError(f"{where}: {op} needs a list")


def _parse_call(raw: Any, where: str) -> ExpectedCall:
    if not isinstance(raw, dict) or not isinstance(raw.get("tool"), str) or not raw["tool"]:
        raise CaseError(f"{where}: each expected call needs a 'tool' name")
    _check_keys(raw, {"tool", "args"}, where)
    args = raw.get("args", {})
    if not isinstance(args, dict):
        raise CaseError(f"{where}: 'args' must be an object")
    for key, matcher in args.items():
        _validate_matcher(matcher, f"{where}.args.{key}")
    return ExpectedCall(raw["tool"], args)


def _parse_calls(expect: dict[str, Any], where: str) -> list[ExpectedCall]:
    if "calls" in expect:
        if "tool" in expect:
            raise CaseError(f"{where}: use either 'tool' or 'calls', not both")
        if not isinstance(expect["calls"], list) or not expect["calls"]:
            raise CaseError(f"{where}.calls must be a non-empty list")
        return [_parse_call(c, f"{where}.calls[{i}]") for i, c in enumerate(expect["calls"])]
    if "tool" in expect:
        return [_parse_call({"tool": expect["tool"], "args": expect.get("args", {})}, where)]
    return []


def _parse_case(raw: Any, index: int) -> Case:
    label = f"case #{index + 1}"
    if not isinstance(raw, dict):
        raise CaseError(f"{label} must be an object")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise CaseError(f"{label} needs a 'name'")
    label = f"case '{name}'"
    _check_keys(raw, _CASE_KEYS, label)
    if not isinstance(raw.get("prompt"), str) or not raw["prompt"].strip():
        raise CaseError(f"{label} needs a non-empty 'prompt'")
    expect = raw.get("expect")
    if not isinstance(expect, dict):
        raise CaseError(f"{label} needs an 'expect' object")
    _check_keys(expect, _EXPECT_KEYS, f"{label}.expect")

    alternatives: list[tuple[list[ExpectedCall], bool]] = []
    if "any_of" in expect:
        if {"tool", "calls", "args"} & set(expect):
            raise CaseError(f"{label}.expect: 'any_of' cannot be combined with 'tool', 'args' or 'calls'")
        options = expect["any_of"]
        if not isinstance(options, list) or len(options) < 2 or not all(isinstance(o, dict) for o in options):
            raise CaseError(f"{label}.expect.any_of needs at least two alternatives (objects)")
        for i, option in enumerate(options):
            where = f"{label}.expect.any_of[{i}]"
            _check_keys(option, _ALTERNATIVE_KEYS, where)
            parsed = _parse_calls(option, where)
            if not parsed:
                raise CaseError(f"{where} needs 'tool' or 'calls'")
            alternatives.append((parsed, bool(option.get("allow_extra", False))))
        calls = alternatives[0][0]
    else:
        calls = _parse_calls(expect, f"{label}.expect")
    no_tool = bool(expect.get("no_tool", False))
    forbid = _string_list(expect.get("forbid"), f"{label}.expect.forbid")
    if no_tool and calls:
        raise CaseError(f"{label}.expect: 'no_tool' contradicts the expected calls")
    if not (no_tool or calls or forbid):
        raise CaseError(f"{label}.expect must say what is correct: tool/calls, no_tool or forbid")

    followup = None
    if raw.get("followup") is not None:
        data = raw["followup"]
        if not isinstance(data, dict):
            raise CaseError(f"{label}.followup must be an object")
        _check_keys(data, _FOLLOWUP_KEYS, f"{label}.followup")
        results = data.get("results", {})
        if not isinstance(results, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in results.items()):
            raise CaseError(f"{label}.followup.results must map tool names to text")
        rubric = data.get("rubric")
        if rubric is not None and not isinstance(rubric, str):
            raise CaseError(f"{label}.followup.rubric must be text")
        followup = Followup(results, _string_list(data.get("forbid"), f"{label}.followup.forbid"), rubric)
        if not (followup.forbid or followup.rubric):
            raise CaseError(f"{label}.followup needs 'forbid' and/or 'rubric'")
        if not calls:
            raise CaseError(f"{label}.followup needs an expected tool call to react to")

    system = raw.get("system")
    if system is not None and not isinstance(system, str):
        raise CaseError(f"{label}.system must be text")
    return Case(
        name=name, prompt=raw["prompt"], system=system, calls=calls, no_tool=no_tool,
        allow_extra=bool(expect.get("allow_extra", False)), forbid=forbid, followup=followup,
        tags=_string_list(raw.get("tags"), f"{label}.tags"), alternatives=alternatives,
    )


def parse_cases(data: Any) -> CaseSet:
    system = None
    if isinstance(data, dict):
        _check_keys(data, {"system", "cases"}, "case file")
        system = data.get("system")
        if system is not None and not isinstance(system, str):
            raise CaseError("case file: 'system' must be text")
        data = data.get("cases")
    if not isinstance(data, list) or not data:
        raise CaseError("case file must contain a non-empty list of cases")
    cases = [_parse_case(item, i) for i, item in enumerate(data)]
    names = [c.name for c in cases]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise CaseError(f"duplicate case names: {duplicates}")
    return CaseSet(cases, system)


def load_cases(path: str) -> CaseSet:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseError(f"cannot read {path}: {exc}") from exc
    return parse_cases(data)


# ------------------------------------------------------------------ matching

def _fold(value: Any, ignore_case: bool) -> Any:
    return value.strip().casefold() if ignore_case and isinstance(value, str) else value


def match_value(matcher: Any, actual: Any, present: bool = True) -> str | None:
    """None when `actual` satisfies `matcher`, otherwise a short reason."""
    if isinstance(matcher, dict) and matcher and set(matcher) & OPERATORS:
        op = next(iter(set(matcher) & OPERATORS))
        ic = bool(matcher.get("ignore_case"))
        want = matcher[op]
        if op == "absent":
            return "should not have been provided" if present else None
        if not present:
            return "missing"
        if op == "any":
            return None
        if op == "type":
            return None if _kind_name(actual) == want else f"expected type {want}, got {_kind_name(actual)}"
        if op == "equals":
            return None if _fold(actual, ic) == _fold(want, ic) else f"expected {want!r}, got {actual!r}"
        if op == "contains":
            if isinstance(actual, str) and isinstance(want, str):
                return None if _fold(want, ic) in _fold(actual, ic) else f"expected text containing {want!r}, got {actual!r}"
            if isinstance(actual, list):
                items = [_fold(x, ic) for x in actual]
                return None if _fold(want, ic) in items else f"expected a list containing {want!r}, got {actual!r}"
            return f"cannot test 'contains' on {actual!r}"
        if op == "regex":
            return None if isinstance(actual, str) and re.search(want, actual) else f"expected a match for /{want}/, got {actual!r}"
        if op == "approx":
            tol = matcher.get("tol", 0)
            ok = isinstance(actual, (int, float)) and not isinstance(actual, bool) and abs(actual - want) <= tol
            return None if ok else f"expected {want} ± {tol}, got {actual!r}"
        if op == "one_of":
            return None if any(_fold(actual, ic) == _fold(w, ic) for w in want) else f"expected one of {want!r}, got {actual!r}"
        if op == "same_items":
            if not isinstance(actual, list):
                return f"expected a list, got {actual!r}"
            wanted, got = sorted(map(str, (_fold(x, ic) for x in want))), sorted(map(str, (_fold(x, ic) for x in actual)))
            return None if wanted == got else f"expected items {want!r}, got {actual!r}"
    if not present:
        return "missing"
    return None if actual == matcher else f"expected {matcher!r}, got {actual!r}"


def _kind_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def match_args(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Reasons the actual arguments do not satisfy the expected ones (empty = match)."""
    problems = []
    for key, matcher in expected.items():
        reason = match_value(matcher, actual.get(key), present=key in actual)
        if reason:
            problems.append(f"{key}: {reason}")
    return problems
