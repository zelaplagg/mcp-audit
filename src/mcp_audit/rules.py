"""Static rules over what an MCP server tells the model: tool definitions and server instructions."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

from .models import Finding, ScanTarget, Severity, make_finding

MAX_SCHEMA_DEPTH = 12
MAX_EVIDENCE = 140
LONG_DESCRIPTION = 1500


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    description: str
    default_severity: Severity
    check: Callable[[ScanTarget], Iterable[Finding]]


# --------------------------------------------------------------------------- helpers


def _snippet(text: str, start: int, end: int) -> str:
    snippet = text[max(0, start - 20) : min(len(text), end + 20)].replace("\n", " ").strip()
    return snippet if len(snippet) <= MAX_EVIDENCE else snippet[: MAX_EVIDENCE - 1] + "…"


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def name_tokens(name: str) -> set[str]:
    """Split snake_case, kebab-case and camelCase identifiers into lowercase words."""
    return {p.lower() for p in re.split(r"[^A-Za-z0-9]+", _CAMEL.sub("_", name)) if p}


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("name", ""))


def _schema_texts(schema: Any, where: str, depth: int = 0) -> Iterator[tuple[str, str]]:
    """Yield (location, text) for every string a model might read inside a JSON schema."""
    if not isinstance(schema, dict) or depth > MAX_SCHEMA_DEPTH:
        return
    for key in ("title", "description", "default", "const"):
        value = schema.get(key)
        if isinstance(value, str) and value:
            yield f"{where} {key}", value
    for key in ("enum", "examples"):
        values = schema.get(key)
        if isinstance(values, list):
            for item in values[:50]:
                if isinstance(item, str) and item:
                    yield f"{where} {key}", item
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for prop_name, sub in properties.items():
            yield f"{where} property name", str(prop_name)
            yield from _schema_texts(sub, f"{where}.{prop_name}", depth + 1)
    for key in ("items", "additionalProperties"):
        yield from _schema_texts(schema.get(key), f"{where}[]", depth + 1)
    for key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for sub in branches:
                yield from _schema_texts(sub, where, depth + 1)
    for key in ("$defs", "definitions"):
        definitions = schema.get(key)
        if isinstance(definitions, dict):
            for def_name, sub in definitions.items():
                yield from _schema_texts(sub, f"{where}.{def_name}", depth + 1)


def iter_text_surfaces(target: ScanTarget) -> Iterator[tuple[str | None, str, str]]:
    """Yield (tool name or None, location, text) for all text the model will see."""
    if target.instructions:
        yield None, "server instructions", target.instructions
    for tool in target.tools:
        name = _tool_name(tool)
        yield name, f"tool '{name}' name", name
        for key in ("title", "description"):
            value = tool.get(key)
            if isinstance(value, str) and value:
                yield name, f"tool '{name}' {key}", value
        annotations = tool.get("annotations")
        if isinstance(annotations, dict) and isinstance(annotations.get("title"), str):
            yield name, f"tool '{name}' annotation title", annotations["title"]
        for where, text in _schema_texts(tool.get("inputSchema"), f"tool '{name}' inputSchema"):
            yield name, where, text


def iter_params(schema: Any, prefix: str = "", depth: int = 0) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (dotted path, property name, sub-schema) for every property, recursively."""
    if not isinstance(schema, dict) or depth > MAX_SCHEMA_DEPTH:
        return
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for prop_name, sub in properties.items():
            if not isinstance(sub, dict):
                continue
            path = f"{prefix}.{prop_name}" if prefix else str(prop_name)
            yield path, str(prop_name), sub
            yield from iter_params(sub, path, depth + 1)
    items = schema.get("items")
    if isinstance(items, dict):
        yield from iter_params(items, f"{prefix}[]", depth + 1)
    for key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for sub in branches:
                yield from iter_params(sub, prefix, depth + 1)


def _is_string(schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    if kind is None:
        return "properties" not in schema and "items" not in schema
    return kind == "string" or (isinstance(kind, list) and "string" in kind)


def _constrained(schema: dict[str, Any]) -> bool:
    return any(key in schema for key in ("enum", "pattern", "const"))


# ------------------------------------------------------------------ MCP001 poisoning

_SENSITIVE_PATHS = (
    r"(~/|/etc/(passwd|shadow)|\.ssh\b|id_rsa|id_ed25519|\.env\b|\.aws\b|\.git-credentials|"
    r"\.npmrc|\bmcp\.json\b|claude_desktop_config|keychain|wallet\.dat)"
)
_SENSITIVE_ACCESS = "instructs access to sensitive files"

_POISON_PATTERNS: list[tuple[re.Pattern[str], Severity, str]] = [
    (
        re.compile(r"<\s*/?\s*(important|system|instructions?|secret|hidden|override)\b[^>]*>", re.I),
        Severity.HIGH,
        "hidden-instruction tag",
    ),
    (
        re.compile(
            r"\bignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|messages?|rules?)",
            re.I,
        ),
        Severity.HIGH,
        "instruction override",
    ),
    (
        re.compile(
            r"\b(do\s+not|don'?t|never)\s+(tell|inform|mention|reveal|show|notify|let)\b[^.]{0,40}\b(user|human|anyone)\b",
            re.I,
        ),
        Severity.HIGH,
        "concealment from the user",
    ),
    (
        re.compile(r"\bwithout\s+(telling|informing|notifying|alerting|asking)\s+(the\s+)?user\b", re.I),
        Severity.HIGH,
        "concealment from the user",
    ),
    (
        re.compile(
            r"\b(before|prior\s+to|first)\b[^.]{0,60}\b(read|open|access|cat|load|include|send)\b[^.]{0,60}"
            + _SENSITIVE_PATHS,
            re.I,
        ),
        Severity.CRITICAL,
        _SENSITIVE_ACCESS,
    ),
    (re.compile(_SENSITIVE_PATHS, re.I), Severity.HIGH, "references a sensitive file"),
    (
        re.compile(
            r"\b(send|post|forward|upload|exfiltrate|transmit|leak)\b[^.]{0,80}"
            r"(https?://|\b[\w.+-]+@[\w-]+\.[\w.]+)",
            re.I,
        ),
        Severity.MEDIUM,
        "mentions sending data to an external destination",
    ),
    (
        re.compile(r"\b(always|must|should)\s+(call|use|invoke)\s+this\s+tool\s+(first|before|instead)", re.I),
        Severity.MEDIUM,
        "tool-selection manipulation",
    ),
    (
        re.compile(
            r"\b(instead\s+of|rather\s+than|replaces?)\s+(using\s+|calling\s+)?(the\s+)?`?[\w\-]+`?\s+tool\b",
            re.I,
        ),
        Severity.MEDIUM,
        "tool-selection manipulation",
    ),
    (
        re.compile(
            r"\bwhen\s+(you\s+)?(use|using|call|calling|invoke|invoking)\s+(the\s+)?`?[\w\-]+`?\s+tool\b"
            r"[^.]{0,80}\b(also|first|must|should)\b",
            re.I,
        ),
        Severity.HIGH,
        "cross-tool instruction",
    ),
]


def check_poisoning(target: ScanTarget) -> Iterator[Finding]:
    for tool, where, text in iter_text_surfaces(target):
        hits: list[tuple[Severity, str, str]] = []
        for pattern, severity, label in _POISON_PATTERNS:
            match = pattern.search(text)
            if match:
                hits.append((severity, label, _snippet(text, match.start(), match.end())))
        if any(label == _SENSITIVE_ACCESS for _, label, _ in hits):
            hits = [h for h in hits if h[1] != "references a sensitive file"]
        for severity, label, evidence in hits:
            yield make_finding(
                "MCP001",
                severity,
                "Possible tool poisoning or hidden instructions",
                f"{label} in {where}",
                tool=tool,
                location=where,
                evidence=evidence,
                remediation=(
                    "Tool metadata should only describe what the tool does. Remove instructions "
                    "addressed to the model; do not install servers you cannot audit."
                ),
            )


# ------------------------------------------------------------- MCP002 hidden characters

_HIDDEN = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")
_TAG_CHARS = re.compile("[\U000e0020-\U000e007e]+")
_BLOB = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/=])")


def check_hidden_content(target: ScanTarget) -> Iterator[Finding]:
    for tool, where, text in iter_text_surfaces(target):
        found = _HIDDEN.findall(text)
        if found:
            codepoints = ", ".join(f"U+{ord(c):04X}" for c in sorted(set(found))[:8])
            tags = _TAG_CHARS.findall(text)
            if tags:
                decoded = "".join(chr(ord(c) - 0xE0000) for chunk in tags for c in chunk)
                yield make_finding(
                    "MCP002",
                    Severity.CRITICAL,
                    "Hidden text smuggled in invisible Unicode tag characters",
                    f"invisible text in {where}",
                    tool=tool,
                    location=where,
                    evidence=f"decoded hidden text: {decoded}",
                    remediation="Reject servers whose metadata contains invisible Unicode; it has no legitimate use here.",
                )
            else:
                yield make_finding(
                    "MCP002",
                    Severity.HIGH,
                    "Invisible or bidirectional Unicode characters in metadata",
                    f"{len(found)} invisible character(s) in {where}",
                    tool=tool,
                    location=where,
                    evidence=codepoints,
                    remediation="Strip zero-width and bidi control characters from all tool metadata.",
                )
        blob = _BLOB.search(text)
        if blob:
            yield make_finding(
                "MCP002",
                Severity.MEDIUM,
                "Long encoded blob in metadata",
                f"base64-like string of {len(blob.group(0))} characters in {where}",
                tool=tool,
                location=where,
                evidence=blob.group(0)[:60] + "…",
                remediation="Metadata should be human readable. Encoded payloads can hide instructions.",
            )


# ------------------------------------------------------------ MCP003 dangerous parameters

_EXEC_PARAMS = {"command", "cmd", "shell", "script", "eval", "exec", "bash", "powershell"}
_PATH_PARAMS = {"path", "file", "filename", "filepath", "directory", "dir", "folder"}
# "uri" is deliberately absent: in MCP it usually names a resource (test://..., file://...), not a URL the server fetches.
_URL_PARAMS = {"url", "endpoint", "webhook", "href", "link"}
_EXEC_TOOLS = {"exec", "execute", "shell", "bash", "eval", "command"}


def _grouped_finding(title: str, risk: str, hits: list[tuple[str, str]], remediation: str) -> Finding:
    """One finding for all parameters of the same kind: reference servers have dozens of `path` parameters."""
    tools = list(dict.fromkeys(tool for tool, _ in hits))
    shown = ", ".join(f"{tool}.{path}" for tool, path in hits[:8]) + (" …" if len(hits) > 8 else "")
    if len(tools) == 1:
        tool, path = hits[0]
        where = f"tool '{tool}' parameter '{path}'" if len(hits) == 1 else f"tool '{tool}' inputSchema"
        return make_finding(
            "MCP003", Severity.MEDIUM, title, f"{where} accepts any value ({risk})",
            tool=tool, location=where, evidence=shown, remediation=remediation,
        )
    return make_finding(
        "MCP003", Severity.MEDIUM, title + "s",
        f"{len(hits)} parameters in {len(tools)} tools accept any value ({risk})",
        tool=None, location=f"inputSchema of {len(tools)} tools", evidence=shown, remediation=remediation,
    )


def check_dangerous_capabilities(target: ScanTarget) -> Iterator[Finding]:
    path_hits: list[tuple[str, str]] = []
    url_hits: list[tuple[str, str]] = []
    for tool in target.tools:
        name = _tool_name(tool)
        if name_tokens(name) & _EXEC_TOOLS:
            yield make_finding(
                "MCP003",
                Severity.HIGH,
                "Tool exposes command execution",
                f"tool '{name}' looks like it runs commands or code",
                tool=name,
                location=f"tool '{name}' name",
                evidence=name,
                remediation="Prefer narrow, purpose-built tools. If execution is required, sandbox it and require user confirmation.",
            )
        for path, param, schema in iter_params(tool.get("inputSchema")):
            tokens = name_tokens(param)
            where = f"tool '{name}' parameter '{path}'"
            if not _is_string(schema):
                continue
            if tokens & _EXEC_PARAMS:
                yield make_finding(
                    "MCP003",
                    Severity.HIGH,
                    "Parameter accepts a command or script",
                    f"{where} is free text that is probably executed",
                    tool=name,
                    location=where,
                    evidence=f"{param}: string",
                    remediation="Never pass model-controlled strings to a shell or interpreter; expose fixed operations with enums.",
                )
            elif "sql" in tokens or param.lower() in {"raw_query", "statement"}:
                yield make_finding(
                    "MCP003",
                    Severity.HIGH,
                    "Parameter accepts raw SQL",
                    f"{where} accepts SQL text",
                    tool=name,
                    location=where,
                    evidence=f"{param}: string",
                    remediation="Use parameterised, read-only queries and a database account with minimal rights.",
                )
            elif "expression" in tokens:
                yield make_finding(
                    "MCP003",
                    Severity.MEDIUM,
                    "Parameter accepts an expression",
                    f"{where} may be evaluated; make sure it never reaches eval()",
                    tool=name,
                    location=where,
                    evidence=f"{param}: string",
                    remediation="Parse expressions with a restricted grammar instead of eval().",
                )
            elif tokens & _PATH_PARAMS and not _constrained(schema):
                path_hits.append((name, path))
            elif tokens & _URL_PARAMS and not _constrained(schema):
                url_hits.append((name, path))
    if path_hits:
        yield _grouped_finding(
            "Unconstrained file path parameter", "path traversal risk", path_hits,
            "A schema cannot express allowed directories, so enforce them in the server: resolve every path "
            "against an allow-listed root and reject '..' and symlink escapes. Try '../' and symlinks yourself.",
        )
    if url_hits:
        yield _grouped_finding(
            "Unconstrained URL parameter", "SSRF risk", url_hits,
            "Allow-list hosts and block private, loopback and link-local addresses on the server side.",
        )


# ----------------------------------------------------------- MCP004 secrets in inputs

_SECRET_TOKENS = {"password", "passwd", "passphrase", "secret", "token", "apikey", "credential", "credentials", "authorization", "bearer"}
_NOT_A_SECRET = {"max", "min", "num", "total", "page", "next", "prev", "cursor", "limit", "count", "completion", "prompt", "input", "output"}


def check_secrets_in_inputs(target: ScanTarget) -> Iterator[Finding]:
    for tool in target.tools:
        name = _tool_name(tool)
        for path, param, _schema in iter_params(tool.get("inputSchema")):
            tokens = name_tokens(param)
            secret = tokens & _SECRET_TOKENS
            pair = {"api", "key"} <= tokens or {"private", "key"} <= tokens or {"access", "key"} <= tokens
            if not secret and not pair:
                continue
            if secret == {"token"} and tokens & _NOT_A_SECRET:
                continue
            where = f"tool '{name}' parameter '{path}'"
            yield make_finding(
                "MCP004",
                Severity.MEDIUM,
                "Secret requested through tool arguments",
                f"{where} asks the model to supply a credential",
                tool=name,
                location=where,
                evidence=param,
                remediation="Authenticate on the server side (environment, OAuth). Secrets in arguments end up in model context and logs.",
            )


# ------------------------------------------------------------------- MCP005 hygiene


def check_schema_hygiene(target: ScanTarget) -> Iterator[Finding]:
    unbounded_hits: list[tuple[str, str]] = []
    for tool in target.tools:
        name = _tool_name(tool)
        description = tool.get("description")
        schema = tool.get("inputSchema")
        loc = f"tool '{name}'"
        if not isinstance(description, str) or len(description.strip()) < 10:
            yield make_finding(
                "MCP005", Severity.LOW, "Tool has no usable description",
                f"{loc} has a missing or very short description", tool=name, location=f"{loc} description",
                remediation="Describe what the tool does and when to use it; models rely on it.",
            )
        elif len(description) > LONG_DESCRIPTION:
            yield make_finding(
                "MCP005", Severity.LOW, "Unusually long tool description",
                f"{loc} description has {len(description)} characters", tool=name, location=f"{loc} description",
                remediation="Long descriptions hide instructions and waste context. Keep them short.",
            )
        if not isinstance(schema, dict) or schema.get("type") != "object":
            yield make_finding(
                "MCP005", Severity.MEDIUM, "Missing or invalid input schema",
                f"{loc} has no object inputSchema", tool=name, location=f"{loc} inputSchema",
                remediation="Declare inputSchema with type 'object' and typed properties.",
            )
            continue
        unbounded_hits.extend(
            (name, path)
            for path, _p, sub in iter_params(schema)
            if sub.get("type") == "string"
            and not any(key in sub for key in ("maxLength", "pattern", "enum", "const", "format"))
        )
        if "additionalProperties" not in schema:
            yield make_finding(
                "MCP005", Severity.INFO, "additionalProperties not set",
                f"{loc} schema does not say whether extra arguments are allowed",
                tool=name, location=f"{loc} inputSchema",
                remediation="Set additionalProperties to false so unexpected arguments are rejected.",
            )


    if unbounded_hits:
        tools = list(dict.fromkeys(t for t, _ in unbounded_hits))
        shown = ", ".join(f"{t}.{p}" for t, p in unbounded_hits[:8]) + (" …" if len(unbounded_hits) > 8 else "")
        yield make_finding(
            "MCP005", Severity.INFO, "Unbounded string parameters",
            f"{len(unbounded_hits)} string parameter(s) in {len(tools)} tool(s) have no maxLength, pattern or enum",
            tool=tools[0] if len(tools) == 1 else None, location=f"inputSchema of {len(tools)} tool(s)",
            evidence=shown,
            remediation="Bound strings where a sensible limit exists (maxLength, pattern or enum). Informational: almost every server has some.",
        )


# ---------------------------------------------------------------- MCP006 annotations

_STATE_CHANGING = {
    "delete", "remove", "drop", "destroy", "wipe", "truncate", "purge", "send", "write", "update", "create",
    "execute", "exec", "kill", "reset", "overwrite", "transfer", "pay", "publish", "post", "commit", "push",
    "deploy", "install", "uninstall", "grant", "revoke", "cancel", "approve", "modify", "edit",
}
_DESTRUCTIVE = {"delete", "remove", "drop", "destroy", "wipe", "truncate", "purge", "kill", "overwrite", "revoke", "uninstall", "reset"}


def check_annotations(target: ScanTarget) -> Iterator[Finding]:
    for tool in target.tools:
        name = _tool_name(tool)
        tokens = name_tokens(name)
        changing = tokens & _STATE_CHANGING
        if not changing:
            continue
        annotations = tool.get("annotations")
        if not isinstance(annotations, dict):
            annotations = {}
        loc = f"tool '{name}' annotations"
        if annotations.get("readOnlyHint") is True:
            yield make_finding(
                "MCP006", Severity.MEDIUM, "readOnlyHint contradicts the tool name",
                f"tool '{name}' is marked read-only but its name suggests it changes state",
                tool=name, location=loc, evidence=f"readOnlyHint=true, name contains {sorted(changing)}",
                remediation="Heuristic: a name cannot prove behaviour, so check the code. If the tool really changes state, the hint is wrong and clients will skip confirmation.",
            )
        elif "readOnlyHint" not in annotations and "destructiveHint" not in annotations:
            yield make_finding(
                "MCP006", Severity.MEDIUM, "State-changing tool without annotations",
                f"tool '{name}' looks state-changing but declares neither readOnlyHint nor destructiveHint",
                tool=name, location=loc, evidence=f"name contains {sorted(changing)}",
                remediation="Add annotations (readOnlyHint, destructiveHint, idempotentHint) so clients can ask for confirmation.",
            )
        if tokens & _DESTRUCTIVE and annotations.get("destructiveHint") is False:
            yield make_finding(
                "MCP006", Severity.MEDIUM, "Destructive-looking tool marked non-destructive",
                f"tool '{name}' sets destructiveHint=false", tool=name, location=loc,
                evidence=f"destructiveHint=false, name contains {sorted(tokens & _DESTRUCTIVE)}",
                remediation="Check that the tool really cannot destroy data, or set destructiveHint to true.",
            )


# ------------------------------------------------------------------ MCP007 names

_WELL_KNOWN = {
    "read_file", "write_file", "list_directory", "create_directory", "search_files", "get_file_info",
    "execute_command", "run_command", "send_email", "send_message", "web_search", "read_query",
    "write_query", "list_tables", "describe_table", "get_weather", "create_issue", "create_pull_request",
    "search_code", "take_screenshot",
}


def within_one_edit(a: str, b: str) -> bool:
    """True if the strings differ by exactly one insertion, deletion or substitution."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if len(a) == len(b):
        return a[i + 1 :] == b[i + 1 :]
    return a[i:] == b[i + 1 :]


def check_names(target: ScanTarget) -> Iterator[Finding]:
    names = [_tool_name(t) for t in target.tools]
    for name, count in Counter(names).items():
        if count > 1:
            yield make_finding(
                "MCP007", Severity.HIGH, "Duplicate tool name",
                f"tool name '{name}' is defined {count} times", tool=name, location=f"tool '{name}' name",
                evidence=name, remediation="Tool names must be unique; duplicates let one definition shadow another.",
            )
    for name in dict.fromkeys(names):
        if not name.isascii():
            yield make_finding(
                "MCP007", Severity.HIGH, "Non-ASCII tool name",
                f"tool name '{name}' contains non-ASCII characters (possible homoglyph lookalike)",
                tool=name, location=f"tool '{name}' name", evidence=name,
                remediation="Use plain ASCII names (letters, digits, underscore, hyphen).",
            )
            continue
        if len(name) >= 8 and name not in _WELL_KNOWN:
            for known in sorted(_WELL_KNOWN):
                if within_one_edit(name, known):
                    yield make_finding(
                        "MCP007", Severity.MEDIUM, "Name resembles a well-known tool",
                        f"tool '{name}' is one character away from the common tool name '{known}'",
                        tool=name, location=f"tool '{name}' name", evidence=f"{name} ~ {known}",
                        remediation="Typosquatted names can make the model pick the wrong tool. Verify the server's origin.",
                    )
                    break


def _noop(_target: ScanTarget) -> Iterable[Finding]:
    return ()


RULES: list[Rule] = [
    Rule("MCP001", "Tool poisoning or hidden instructions",
         "Tool descriptions, parameter descriptions or server instructions contain text that steers the model: hidden tags, instruction overrides, concealment, sensitive file access, data exfiltration, cross-tool instructions.",
         Severity.HIGH, check_poisoning),
    Rule("MCP002", "Invisible or encoded content",
         "Zero-width, bidirectional or Unicode tag characters (ASCII smuggling) and long encoded blobs in metadata.",
         Severity.HIGH, check_hidden_content),
    Rule("MCP003", "Dangerous capabilities",
         "Tools or parameters that accept commands, raw SQL, expressions, unconstrained file paths or URLs.",
         Severity.MEDIUM, check_dangerous_capabilities),
    Rule("MCP004", "Secrets in tool arguments",
         "Parameters that ask the model to supply passwords, tokens or API keys.",
         Severity.MEDIUM, check_secrets_in_inputs),
    Rule("MCP005", "Schema hygiene",
         "Missing descriptions, missing or invalid input schemas, unbounded strings, open additionalProperties.",
         Severity.LOW, check_schema_hygiene),
    Rule("MCP006", "Missing or misleading annotations",
         "State-changing tools without readOnlyHint/destructiveHint, or annotations that contradict the tool name.",
         Severity.MEDIUM, check_annotations),
    Rule("MCP007", "Tool name integrity",
         "Duplicate names, non-ASCII lookalikes and names one character away from well-known tools.",
         Severity.MEDIUM, check_names),
    Rule("MCP008", "Definition changed since pinned (rug pull)",
         "A tool was added, removed or changed compared to a pinned baseline. Needs --baseline; see 'mcp-audit pin'.",
         Severity.HIGH, _noop),
]


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    unique = list(dict.fromkeys(findings))
    return sorted(unique, key=lambda f: (-int(f.severity), f.rule_id, f.tool or "", f.location))


def run_rules(
    target: ScanTarget, *, only: set[str] | None = None, skip: set[str] | None = None
) -> list[Finding]:
    findings: list[Finding] = []
    for rule in RULES:
        if only and rule.id not in only:
            continue
        if skip and rule.id in skip:
            continue
        findings.extend(rule.check(target))
    return sort_findings(findings)
