"""Run eval cases against models and score the responses.

The eval never executes MCP tools. It shows a model the tool *definitions*, then checks which
tool calls the model asks for. Tool results used in follow-up turns come from the case file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .evalcases import Case, CaseSet, ExpectedCall, match_args
from .jsonschema_lite import validate
from .llm import ChatClient, LlmError, Reply, ToolCall, ToolSet
from .models import clean_text

DEFAULT_SYSTEM = "You are a helpful assistant. Use the available tools when they help; otherwise answer directly."
_JUDGE_SYSTEM = (
    "You grade an AI assistant's answer against a rubric. Reply with only a JSON object: "
    '{"pass": true or false, "reason": "one short sentence"}.'
)


@dataclass
class Check:
    name: str          # selection | arguments | schema | extra_calls | forbidden | no_tool | followup | judge
    ok: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class RunResult:
    model: str
    case: str
    repeat: int
    checks: list[Check] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    error: str = ""
    latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    had_calls: bool = False

    @property
    def passed(self) -> bool:
        return not self.error and all(c.ok for c in self.checks if not c.skipped)

    def check(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    @property
    def failures(self) -> list[str]:
        if self.error:
            return [f"error: {self.error}"]
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.ok and not c.skipped]


def _describe_calls(calls: list[tuple[str, ToolCall]]) -> str:
    parts = []
    for name, call in calls:
        shown = json.dumps(call.arguments, ensure_ascii=False) if call.arguments is not None else call.raw_arguments
        parts.append(f"{name}({shown[:120]})")
    return ", ".join(parts)


def _said(content: str) -> str:
    text = " ".join(clean_text(content).split())
    return f'model said: "{text[:140]}{"…" if len(text) > 140 else ""}"' if text else "model gave an empty reply"


def _real_calls(reply: Reply, toolset: ToolSet) -> list[tuple[str, ToolCall]]:
    return [(toolset.real_name(c.name), c) for c in reply.tool_calls]


def _score_variant(expected_calls: list[ExpectedCall], allow_extra: bool,
                   calls: list[tuple[str, ToolCall]], content: str) -> list[Check]:
    checks: list[Check] = []
    names = [name for name, _ in calls]
    missing = [e.tool for e in expected_calls if e.tool not in names]
    checks.append(Check("selection", not missing, "" if not missing else (
        f"expected {missing}, model called {_describe_calls(calls)}" if calls
        else f"expected {missing}, model called nothing; {_said(content)}")))
    if not missing:
        problems: list[str] = []
        used: set[int] = set()
        for expected in expected_calls:
            reasons_per_call: list[tuple[int, list[str]]] = []
            for index, (name, call) in enumerate(calls):
                if name != expected.tool or index in used:
                    continue
                reasons = match_args(expected.args, call.arguments) if call.arguments is not None else [call.error]
                reasons_per_call.append((index, reasons))
            matched = next((i for i, r in reasons_per_call if not r), None)
            if matched is None:
                shown = reasons_per_call[0][1] if reasons_per_call else ["no unmatched call left for this expectation"]
                problems.append(f"{expected.tool}: " + "; ".join(shown))
            else:
                used.add(matched)
        checks.append(Check("arguments", not problems, " | ".join(problems)))
    if not allow_extra and len(calls) > len(expected_calls):
        checks.append(Check("extra_calls", False,
                            f"made {len(calls)} calls, expected {len(expected_calls)}: {_describe_calls(calls)}"))
    return checks


def _score_first_turn(case: Case, calls: list[tuple[str, ToolCall]], toolset: ToolSet, content: str = "") -> list[Check]:
    checks: list[Check] = []
    names = [name for name, _ in calls]

    if case.no_tool:
        checks.append(Check("no_tool", not calls, "" if not calls else f"called {_describe_calls(calls)} but no tool was needed"))

    variants = case.variants()
    if variants:
        scored = [_score_variant(expected, allow_extra, calls, content) for expected, allow_extra in variants]
        best = next((v for v in scored if all(c.ok for c in v)), None)
        if best is None:
            best = min(scored, key=lambda v: sum(not c.ok for c in v))
            if len(scored) > 1:
                first_bad = next(c for c in best if not c.ok)
                first_bad.detail += f" (closest of {len(scored)} accepted alternatives)"
        checks.extend(best)

    if calls:
        errors: list[str] = []
        for name, call in calls:
            if name not in toolset.schemas:
                errors.append(f"unknown tool '{clean_text(call.name)}'")
            elif call.arguments is None:
                errors.append(f"{name}: {call.error}")
            else:
                errors.extend(f"{name} {e}" for e in validate(call.arguments, toolset.schemas[name])[:3])
        checks.append(Check("schema", not errors, "; ".join(errors)))

    if case.forbid:
        hit = sorted({n for n in names if n in case.forbid})
        checks.append(Check("forbidden", not hit, "" if not hit else f"called forbidden tool(s): {_describe_calls([c for c in calls if c[0] in case.forbid])}"))
    return checks


def _judge(judge_client: ChatClient, model: str, rubric: str, case: Case, answer: str) -> Check:
    prompt = f"Rubric:\n{rubric}\n\nUser request:\n{case.prompt}\n\nAssistant answer:\n{answer or '(empty)'}"
    try:
        reply = judge_client.complete(model, [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": prompt}])
    except LlmError as exc:
        return Check("judge", False, f"judge error: {exc}")
    match = re.search(r"\{.*\}", reply.content, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        verdict = None
    if not isinstance(verdict, dict) or not isinstance(verdict.get("pass"), bool):
        return Check("judge", False, "judge did not return a verdict")
    return Check("judge", verdict["pass"], clean_text(str(verdict.get("reason", "")))[:200])


def run_case(
    client: ChatClient,
    model: str,
    toolset: ToolSet,
    case: Case,
    *,
    default_system: str | None = None,
    temperature: float = 0.0,
    repeat: int = 1,
    judge: tuple[ChatClient, str] | None = None,
) -> RunResult:
    result = RunResult(model=model, case=case.name, repeat=repeat, tags=list(case.tags))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": case.system or default_system or DEFAULT_SYSTEM},
        {"role": "user", "content": case.prompt},
    ]
    try:
        first = client.complete(model, messages, toolset.functions, temperature=temperature)
        result.latency += first.latency
        result.prompt_tokens += first.prompt_tokens
        result.completion_tokens += first.completion_tokens
        calls = _real_calls(first, toolset)
        result.had_calls = bool(calls)
        result.checks.extend(_score_first_turn(case, calls, toolset, first.content))

        if case.followup and calls:
            messages.append(first.message)
            for _name, call in calls:
                name = toolset.real_name(call.name)
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": case.followup.results.get(name, "ok"),
                })
            second = client.complete(model, messages, toolset.functions, temperature=temperature)
            result.latency += second.latency
            result.prompt_tokens += second.prompt_tokens
            result.completion_tokens += second.completion_tokens
            later_calls = _real_calls(second, toolset)
            if case.followup.forbid:
                hit = [c for c in later_calls if c[0] in case.followup.forbid]
                result.checks.append(Check("followup", not hit,
                                           "" if not hit else f"after reading the tool result the model called {_describe_calls(hit)}"))
            if case.followup.rubric:
                if judge is None:
                    result.checks.append(Check("judge", True, "no --judge-model given", skipped=True))
                else:
                    result.checks.append(_judge(judge[0], judge[1], case.followup.rubric, case, second.content))
        elif case.followup and not calls:
            result.checks.append(Check("followup", False, "no tool call to react to"))
    except LlmError as exc:
        result.error = str(exc)
    return result


@dataclass
class ModelSummary:
    model: str
    runs: int = 0
    passed: int = 0
    errors: int = 0
    selection_rate: float | None = None
    argument_rate: float | None = None
    schema_rate: float | None = None
    avg_latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_tag: dict[str, tuple[int, int]] = field(default_factory=dict)  # tag -> (passed, runs)
    flaky: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.runs if self.runs else 0.0


def _rate(results: list[RunResult], name: str) -> float | None:
    checks = [r.check(name) for r in results if not r.error]
    checks = [c for c in checks if c is not None and not c.skipped]
    return sum(c.ok for c in checks) / len(checks) if checks else None


def summarize_model(model: str, results: list[RunResult]) -> ModelSummary:
    mine = [r for r in results if r.model == model]
    summary = ModelSummary(model=model, runs=len(mine))
    summary.passed = sum(r.passed for r in mine)
    summary.errors = sum(bool(r.error) for r in mine)
    summary.selection_rate = _rate(mine, "selection")
    # argument accuracy is only defined where the right tool was chosen
    summary.argument_rate = _rate(mine, "arguments")
    summary.schema_rate = _rate(mine, "schema")
    ok_runs = [r for r in mine if not r.error]
    summary.avg_latency = sum(r.latency for r in ok_runs) / len(ok_runs) if ok_runs else 0.0
    summary.prompt_tokens = sum(r.prompt_tokens for r in mine)
    summary.completion_tokens = sum(r.completion_tokens for r in mine)
    for tag in sorted({t for r in mine for t in r.tags}):
        tagged = [r for r in mine if tag in r.tags]
        summary.by_tag[tag] = (sum(r.passed for r in tagged), len(tagged))
    for case in dict.fromkeys(r.case for r in mine):
        outcomes = {r.passed for r in mine if r.case == case}
        if len(outcomes) > 1:
            summary.flaky.append(case)
    return summary


def run_eval(
    client: ChatClient,
    models: list[str],
    toolset: ToolSet,
    cases: CaseSet,
    *,
    repeats: int = 1,
    temperature: float = 0.0,
    judge: tuple[ChatClient, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[RunResult]:
    results: list[RunResult] = []
    for model in models:
        for case in cases.cases:
            for repeat in range(1, repeats + 1):
                result = run_case(client, model, toolset, case, default_system=cases.system,
                                  temperature=temperature, repeat=repeat, judge=judge)
                results.append(result)
                if progress:
                    progress(f"{model}  {case.name}  run {repeat}/{repeats}: {'pass' if result.passed else 'FAIL'}")
    return results


def check_cases(cases: CaseSet, tools: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Dry run: compare the cases with the server's tools. Returns (errors, warnings)."""
    by_name = {t.get("name"): t for t in tools if isinstance(t.get("name"), str)}
    errors: list[str] = []
    warnings: list[str] = []
    for case in cases.cases:
        for expected in [c for calls, _ in case.variants() for c in calls]:
            tool = by_name.get(expected.tool)
            if tool is None:
                errors.append(f"{case.name}: expected tool '{expected.tool}' does not exist on the server")
                continue
            schema = tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else {}
            properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            for arg in expected.args:
                if properties and arg not in properties:
                    errors.append(f"{case.name}: '{expected.tool}' has no parameter '{arg}'")
        for name in case.forbid + (case.followup.forbid if case.followup else []):
            if name not in by_name:
                warnings.append(f"{case.name}: forbidden tool '{name}' does not exist, so the check can never fail")
        if case.followup:
            for name in case.followup.results:
                if name not in by_name:
                    warnings.append(f"{case.name}: follow-up result for unknown tool '{name}'")
    return errors, warnings
