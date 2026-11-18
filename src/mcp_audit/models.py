from __future__ import annotations

import enum
import unicodedata
from dataclasses import dataclass, field
from typing import Any


class Severity(enum.IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: str) -> "Severity":
        try:
            return cls[value.strip().upper()]
        except KeyError:
            raise ValueError(f"unknown severity: {value!r}") from None


def clean_text(text: str) -> str:
    """Make untrusted text safe to print: control, format and unassigned characters become <U+XXXX>.

    This matters for a security tool: raw bidi or zero-width characters in a report could hide
    or reorder text in a terminal or a CI log.
    """
    out: list[str] = []
    for ch in text:
        if ch in "\n\r\t":
            out.append(" ")
        elif unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn", "Cs"):
            out.append(f"<U+{ord(ch):04X}>")
        else:
            out.append(ch)
    return "".join(out)


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: Severity
    title: str
    message: str
    tool: str | None = None
    location: str = ""
    evidence: str = ""
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.label,
            "title": self.title,
            "message": self.message,
            "tool": self.tool,
            "location": self.location,
            "evidence": self.evidence,
            "remediation": self.remediation,
        }


def make_finding(
    rule_id: str,
    severity: Severity,
    title: str,
    message: str,
    *,
    tool: str | None = None,
    location: str = "",
    evidence: str = "",
    remediation: str = "",
) -> Finding:
    """Build a Finding with every untrusted string sanitised for display."""
    return Finding(
        rule_id=rule_id,
        severity=severity,
        title=title,
        message=clean_text(message),
        tool=clean_text(tool) if tool is not None else None,
        location=clean_text(location),
        evidence=clean_text(evidence),
        remediation=remediation,
    )


@dataclass
class ScanTarget:
    """Everything the scanner knows about one MCP server, as reported by the server itself."""

    server_name: str = "unknown"
    server_version: str = ""
    protocol_version: str = ""
    instructions: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
