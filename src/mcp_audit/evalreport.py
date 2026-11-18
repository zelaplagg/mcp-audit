from __future__ import annotations

import json
from typing import Any

from .evalrun import ModelSummary, RunResult, summarize_model
from .models import ScanTarget, clean_text


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _models(results: list[RunResult]) -> list[str]:
    return list(dict.fromkeys(r.model for r in results))


def _header(target: ScanTarget, case_count: int, repeats: int) -> str:
    return (f"mcp-audit eval: '{clean_text(target.server_name)}' {clean_text(target.server_version)}, "
            f"{len(target.tools)} tool(s), {case_count} case(s) x {repeats} run(s)")


def render_text(results: list[RunResult], target: ScanTarget, case_count: int, repeats: int) -> str:
    lines = [_header(target, case_count, repeats), ""]
    for model in _models(results):
        s = summarize_model(model, results)
        lines.append(f"Model: {clean_text(model)}")
        lines.append(f"  pass rate {_pct(s.pass_rate)} ({s.passed}/{s.runs})   selection {_pct(s.selection_rate)}   "
                     f"arguments {_pct(s.argument_rate)}   schema-valid {_pct(s.schema_rate)}")
        lines.append(f"  avg latency {s.avg_latency * 1000:.0f} ms   tokens {_tokens(s.prompt_tokens)} in / {_tokens(s.completion_tokens)} out"
                     + (f"   errors {s.errors}" if s.errors else ""))
        if len(s.by_tag) > 1:
            lines.append("  by tag: " + ", ".join(f"{t} {p}/{n}" for t, (p, n) in s.by_tag.items()))
        failed = [r for r in results if r.model == model and not r.passed]
        for r in failed:
            suffix = f" [run {r.repeat}]" if repeats > 1 else ""
            for failure in r.failures:
                lines.append(f"  FAIL {clean_text(r.case)}{suffix}: {clean_text(failure)}")
        if s.flaky:
            lines.append("  flaky (passed and failed across runs): " + ", ".join(s.flaky))
        lines.append("")
    if len(_models(results)) > 1:
        lines.append(_comparison_table(results))
    return "\n".join(lines).rstrip() + "\n"


def _comparison_table(results: list[RunResult]) -> str:
    rows = [["model", "pass", "selection", "arguments", "schema", "latency"]]
    for model in _models(results):
        s = summarize_model(model, results)
        rows.append([clean_text(model), _pct(s.pass_rate), _pct(s.selection_rate), _pct(s.argument_rate),
                     _pct(s.schema_rate), f"{s.avg_latency * 1000:.0f} ms"])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows) + "\n"


def render_markdown(results: list[RunResult], target: ScanTarget, case_count: int, repeats: int) -> str:
    out = [f"## {_header(target, case_count, repeats)}", ""]
    out.append("| Model | Pass rate | Selection | Arguments | Schema-valid | Avg latency | Errors |")
    out.append("|---|---|---|---|---|---|---|")
    for model in _models(results):
        s = summarize_model(model, results)
        out.append(f"| `{clean_text(model)}` | {_pct(s.pass_rate)} ({s.passed}/{s.runs}) | {_pct(s.selection_rate)} | "
                   f"{_pct(s.argument_rate)} | {_pct(s.schema_rate)} | {s.avg_latency * 1000:.0f} ms | {s.errors} |")
    failures = [r for r in results if not r.passed]
    if failures:
        out += ["", "### Failures", "", "| Model | Case | Run | Problem |", "|---|---|---|---|"]
        for r in failures:
            problem = "; ".join(r.failures).replace("|", "\\|")
            out.append(f"| `{clean_text(r.model)}` | {clean_text(r.case)} | {r.repeat} | {clean_text(problem)} |")
    return "\n".join(out) + "\n"


def render_json(results: list[RunResult], target: ScanTarget, case_count: int, repeats: int) -> str:
    models: list[dict[str, Any]] = []
    for model in _models(results):
        s: ModelSummary = summarize_model(model, results)
        models.append({
            "model": model, "runs": s.runs, "passed": s.passed, "pass_rate": round(s.pass_rate, 4),
            "selection_rate": s.selection_rate, "argument_rate": s.argument_rate, "schema_rate": s.schema_rate,
            "errors": s.errors, "avg_latency_s": round(s.avg_latency, 3),
            "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens,
            "by_tag": {t: {"passed": p, "runs": n} for t, (p, n) in s.by_tag.items()}, "flaky": s.flaky,
        })
    runs = [{
        "model": r.model, "case": r.case, "repeat": r.repeat, "passed": r.passed, "error": r.error or None,
        "checks": [{"name": c.name, "ok": c.ok, "skipped": c.skipped, "detail": c.detail} for c in r.checks],
    } for r in results]
    doc = {"tool": "mcp-audit", "server": {"name": target.server_name, "version": target.server_version,
                                           "tools": len(target.tools)},
           "cases": case_count, "repeats": repeats, "models": models, "runs": runs}
    return json.dumps(doc, indent=2, ensure_ascii=True) + "\n"
