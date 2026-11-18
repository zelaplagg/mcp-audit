"""Pinning tool definitions and detecting later changes ("rug pulls")."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Finding, ScanTarget, Severity, make_finding

LOCK_VERSION = 1
_TOOL_KEYS = ("name", "title", "description", "inputSchema", "outputSchema", "annotations")


class BaselineError(Exception):
    pass


def canonical_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {key: tool[key] for key in _TOOL_KEYS if key in tool}


def fingerprint(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def create_baseline(target: ScanTarget) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for tool in target.tools:
        canonical = canonical_tool(tool)
        tools[str(tool.get("name", ""))] = {"sha256": fingerprint(canonical), "definition": canonical}
    return {
        "version": LOCK_VERSION,
        "server": {"name": target.server_name, "version": target.server_version},
        "instructions": {"sha256": fingerprint(target.instructions), "text": target.instructions},
        "tools": tools,
    }


def save_baseline(path: str | Path, baseline: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(baseline, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BaselineError(f"baseline file not found: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read baseline {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != LOCK_VERSION or not isinstance(data.get("tools"), dict):
        raise BaselineError(f"{path} is not a valid mcp-audit baseline (version {LOCK_VERSION})")
    return data


def _diff(old: str, new: str) -> str:
    lines = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
    text = " ".join(line for line in lines if not line.startswith(("---", "+++", "@@")))
    return text[:600]


def compare(baseline: dict[str, Any], target: ScanTarget) -> list[Finding]:
    findings: list[Finding] = []
    pinned: dict[str, Any] = baseline["tools"]
    current = {str(t.get("name", "")): canonical_tool(t) for t in target.tools}

    for name, definition in current.items():
        loc = f"tool '{name}'"
        if name not in pinned:
            findings.append(make_finding(
                "MCP008", Severity.MEDIUM, "Tool added since pinning",
                f"{loc} was not in the pinned baseline", tool=name, location=loc, evidence=name,
                remediation="Review the new tool, then re-pin with 'mcp-audit pin'.",
            ))
            continue
        if fingerprint(definition) == pinned[name].get("sha256"):
            continue
        old = pinned[name].get("definition", {})
        changed = [k for k in _TOOL_KEYS if old.get(k) != definition.get(k)]
        severity = Severity.MEDIUM if changed == ["annotations"] else Severity.HIGH
        evidence = f"changed fields: {', '.join(changed) or 'unknown'}"
        if "description" in changed:
            evidence += f"; description diff: {_diff(str(old.get('description', '')), str(definition.get('description', '')))}"
        findings.append(make_finding(
            "MCP008", severity, "Tool definition changed since pinning",
            f"{loc} no longer matches the pinned definition", tool=name, location=loc, evidence=evidence,
            remediation="A silent change after approval is how rug-pull attacks work. Review the diff, then re-pin.",
        ))

    for name in pinned:
        if name not in current:
            findings.append(make_finding(
                "MCP008", Severity.LOW, "Tool removed since pinning",
                f"tool '{name}' is no longer offered", tool=name, location=f"tool '{name}'", evidence=name,
                remediation="Re-pin if the removal is expected.",
            ))

    pinned_instructions = baseline.get("instructions", {}).get("sha256")
    if pinned_instructions is not None and pinned_instructions != fingerprint(target.instructions):
        old_text = str(baseline.get("instructions", {}).get("text", ""))
        findings.append(make_finding(
            "MCP008", Severity.HIGH, "Server instructions changed since pinning",
            "the server's instructions differ from the pinned baseline", location="server instructions",
            evidence=_diff(old_text, target.instructions),
            remediation="Server instructions are injected into the model context. Review the change, then re-pin.",
        ))
    return findings
