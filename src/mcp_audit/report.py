from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable

from . import __version__
from .models import Finding, ScanTarget, Severity
from .rules import Rule

_SARIF_LEVEL = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}
_SECURITY_SEVERITY = {
    Severity.CRITICAL: "9.5",
    Severity.HIGH: "8.0",
    Severity.MEDIUM: "5.5",
    Severity.LOW: "3.0",
    Severity.INFO: "1.0",
}


def summarize(findings: Iterable[Finding]) -> dict[str, int]:
    counts = Counter(f.severity.label for f in findings)
    return {s.label: counts.get(s.label, 0) for s in sorted(Severity, reverse=True)}


def render_text(findings: list[Finding], target: ScanTarget) -> str:
    version = f" v{target.server_version}" if target.server_version else ""
    lines = [f"mcp-audit {__version__}: scanned '{target.server_name}'{version}, {len(target.tools)} tool(s)", ""]
    if not findings:
        lines.append("No findings.")
    for finding in findings:
        lines.append(f"[{finding.severity.label.upper()}] {finding.rule_id} {finding.title}")
        lines.append(f"  {finding.message}")
        if finding.tool is not None:
            lines.append(f"  tool:     {finding.tool}")
        if finding.location:
            lines.append(f"  where:    {finding.location}")
        if finding.evidence:
            lines.append(f"  evidence: {finding.evidence}")
        if finding.remediation:
            lines.append(f"  fix:      {finding.remediation}")
        lines.append("")
    counts = summarize(findings)
    lines.append("Summary: " + ", ".join(f"{n} {label}" for label, n in counts.items()))
    return "\n".join(lines) + "\n"


def render_json(findings: list[Finding], target: ScanTarget) -> str:
    document = {
        "tool": {"name": "mcp-audit", "version": __version__},
        "server": {
            "name": target.server_name,
            "version": target.server_version,
            "protocol_version": target.protocol_version,
            "tool_count": len(target.tools),
        },
        "summary": summarize(findings),
        "findings": [f.to_dict() for f in findings],
    }
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"


def render_sarif(findings: list[Finding], target: ScanTarget, rules: list[Rule], *, uri: str = "mcp-server.json") -> str:
    index = {rule.id: i for i, rule in enumerate(rules)}
    results = []
    for finding in findings:
        fingerprint = hashlib.sha256(
            "|".join([finding.rule_id, finding.tool or "", finding.location, finding.evidence]).encode("utf-8")
        ).hexdigest()
        text = f"{finding.title}: {finding.message}"
        if finding.evidence:
            text += f" (evidence: {finding.evidence})"
        results.append({
            "ruleId": finding.rule_id,
            "ruleIndex": index.get(finding.rule_id, -1),
            "level": _SARIF_LEVEL[finding.severity],
            "message": {"text": text},
            "locations": [{
                "physicalLocation": {"artifactLocation": {"uri": uri}, "region": {"startLine": 1}},
                "logicalLocations": [{"name": finding.tool or "server", "kind": "function"}],
            }],
            "partialFingerprints": {"mcpAudit/v1": fingerprint},
            "properties": {"severity": finding.severity.label, "remediation": finding.remediation},
        })
    document = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "mcp-audit",
                "version": __version__,
                "rules": [{
                    "id": rule.id,
                    "shortDescription": {"text": rule.title},
                    "fullDescription": {"text": rule.description},
                    "defaultConfiguration": {"level": _SARIF_LEVEL[rule.default_severity]},
                    "properties": {"security-severity": _SECURITY_SEVERITY[rule.default_severity]},
                } for rule in rules],
            }},
            "results": results,
        }],
    }
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"
