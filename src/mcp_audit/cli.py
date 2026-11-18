from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .baseline import BaselineError, compare, create_baseline, load_baseline, save_baseline
from .client import HttpTransport, McpClient, McpError, StdioTransport
from .evalcases import CaseError, load_cases
from .evalreport import render_json as eval_json
from .evalreport import render_markdown as eval_markdown
from .evalreport import render_text as eval_text
from .evalrun import check_cases, run_eval, summarize_model
from .llm import ChatClient, LlmError, build_toolset
from .models import Finding, ScanTarget, Severity
from .report import render_json, render_sarif, render_text, summarize
from .rules import RULES, run_rules, sort_findings


class UsageError(Exception):
    pass


def _split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def _add_target_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "target", "Pick one: a command after '--' (local stdio server), --url, or --from-json."
    )
    group.add_argument("--url", help="Streamable HTTP endpoint of a remote server")
    group.add_argument("--from-json", metavar="FILE", help="read a saved tools/list result instead of connecting")
    parser.add_argument("--header", action="append", default=[], metavar="'Name: value'", help="extra HTTP header (repeatable)")
    parser.add_argument("--bearer-env", metavar="VAR", help="read a bearer token from this environment variable")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment variable for a stdio server (repeatable)")
    parser.add_argument("--cwd", help="working directory for a stdio server")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for each response (default 30)")
    parser.add_argument("--allow-insecure", action="store_true", help="allow sending credentials over plain http")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-audit",
        description="Security and regression audit for MCP servers. Only lists tools; never calls them.",
        epilog="Examples:\n  mcp-audit scan -- node dist/index.js\n  mcp-audit scan --url https://example.com/mcp --bearer-env MCP_TOKEN --format sarif -o audit.sarif",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"mcp-audit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="audit a server", description="Audit an MCP server.")
    _add_target_options(scan)
    scan.add_argument("--format", choices=("text", "json", "sarif"), default="text")
    scan.add_argument("-o", "--output", help="write the report to a file instead of stdout")
    scan.add_argument("--fail-on", choices=("none", "low", "medium", "high", "critical"), default="high",
                      help="exit with status 1 if a finding at or above this severity exists (default high)")
    scan.add_argument("--min-severity", choices=("info", "low", "medium", "high", "critical"), default="low",
                      help="hide findings below this severity (default low)")
    scan.add_argument("--baseline", metavar="FILE", help="also compare against a pinned baseline (rug-pull check)")
    scan.add_argument("--only-rule", action="append", default=[], metavar="ID", help="run only this rule (repeatable)")
    scan.add_argument("--skip-rule", action="append", default=[], metavar="ID", help="skip this rule (repeatable)")
    scan.add_argument("--sarif-uri", default="mcp-server.json", help="file path to attribute SARIF results to")

    pin = sub.add_parser("pin", help="record the current tool definitions as a baseline",
                         description="Write a baseline file; later scans with --baseline report any change.")
    _add_target_options(pin)
    pin.add_argument("-o", "--output", default="mcp-audit.lock.json", help="baseline file (default mcp-audit.lock.json)")

    ev = sub.add_parser(
        "eval", help="measure how well LLMs use a server's tools",
        description="Show a model the server's tool definitions, run golden test cases and score which tools "
                    "and arguments it picks. Tools are never executed. Needs an OpenAI-compatible chat API.",
    )
    _add_target_options(ev)
    ev.add_argument("--cases", required=True, metavar="FILE", help="JSON file with the test cases")
    ev.add_argument("--model", action="append", default=[], metavar="NAME", help="model to test (repeatable)")
    ev.add_argument("--base-url", default="https://api.openai.com/v1", help="OpenAI-compatible API (default OpenAI; Ollama: http://localhost:11434/v1)")
    ev.add_argument("--api-key-env", metavar="VAR", help="environment variable holding the API key")
    ev.add_argument("-n", "--repeats", type=int, default=1, help="runs per case, to expose flaky behaviour (default 1)")
    ev.add_argument("--temperature", type=float, default=0.0)
    ev.add_argument("--judge-model", metavar="NAME", help="model that grades follow-up answers against a rubric")
    ev.add_argument("--llm-timeout", type=float, default=60.0, help="seconds to wait for the model API (default 60)")
    ev.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    ev.add_argument("-o", "--output", help="write the report to a file instead of stdout")
    ev.add_argument("--min-pass-rate", type=float, default=0.0, metavar="0-1",
                    help="exit with status 1 if any model passes fewer cases than this (default: never)")
    ev.add_argument("--dry-run", action="store_true", help="only check the cases against the server's tools; no model calls")
    ev.add_argument("--quiet", action="store_true", help="no progress lines on stderr")

    sub.add_parser("rules", help="list the rules")
    return parser


def _parse_headers(args: argparse.Namespace) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw in args.header:
        name, sep, value = raw.partition(":")
        if not sep or not name.strip():
            raise UsageError(f"invalid --header {raw!r}; expected 'Name: value'")
        headers[name.strip()] = value.strip()
    if args.bearer_env:
        token = os.environ.get(args.bearer_env)
        if not token:
            raise UsageError(f"environment variable {args.bearer_env} is not set or empty")
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_env(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise UsageError(f"invalid --env {item!r}; expected KEY=VALUE")
        env[key] = value
    return env


def load_target_json(path: str) -> ScanTarget:
    try:
        data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"cannot read {path}: {exc}") from exc
    if isinstance(data, list):
        data = {"tools": data}
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        data = data["result"]
    if not isinstance(data, dict) or not isinstance(data.get("tools"), list):
        raise UsageError(f"{path} must contain a tools list or a tools/list result")
    info = data.get("serverInfo") if isinstance(data.get("serverInfo"), dict) else {}
    instructions = data.get("instructions")
    return ScanTarget(
        server_name=str(info.get("name", Path(path).name)),
        server_version=str(info.get("version", "")),
        instructions=instructions if isinstance(instructions, str) else "",
        tools=[t for t in data["tools"] if isinstance(t, dict)],
    )


def collect_target(args: argparse.Namespace, command: list[str]) -> ScanTarget:
    chosen = sum(bool(x) for x in (args.url, args.from_json, command))
    if chosen != 1:
        raise UsageError("choose exactly one target: a command after '--', --url, or --from-json")
    if args.from_json:
        return load_target_json(args.from_json)
    if args.url:
        transport: Any = HttpTransport(
            args.url, headers=_parse_headers(args), timeout=args.timeout, allow_insecure=args.allow_insecure
        )
    else:
        transport = StdioTransport(command, env=_parse_env(args.env), cwd=args.cwd, timeout=args.timeout)
    with McpClient(transport) as client:
        init = client.initialize()
        capabilities = init.get("capabilities")
        has_tools = isinstance(capabilities, dict) and "tools" in capabilities
        tools = client.list_tools() if has_tools else []
        if not has_tools:
            print("warning: server does not advertise the tools capability", file=sys.stderr)
    info = init.get("serverInfo") if isinstance(init.get("serverInfo"), dict) else {}
    instructions = init.get("instructions")
    return ScanTarget(
        server_name=str(info.get("name", "unknown")),
        server_version=str(info.get("version", "")),
        protocol_version=str(init.get("protocolVersion", "")),
        instructions=instructions if isinstance(instructions, str) else "",
        tools=tools,
    )


def _write(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _cmd_scan(args: argparse.Namespace, command: list[str]) -> int:
    target = collect_target(args, command)
    findings: list[Finding] = run_rules(target, only=set(args.only_rule) or None, skip=set(args.skip_rule) or None)
    if args.baseline:
        findings = sort_findings(findings + compare(load_baseline(args.baseline), target))
    minimum = Severity.parse(args.min_severity)
    findings = [f for f in findings if f.severity >= minimum]

    if args.format == "json":
        report = render_json(findings, target)
    elif args.format == "sarif":
        report = render_sarif(findings, target, RULES, uri=args.sarif_uri)
    else:
        report = render_text(findings, target)
    _write(report, args.output)

    if args.fail_on != "none":
        threshold = Severity.parse(args.fail_on)
        if any(f.severity >= threshold for f in findings):
            return 1
    return 0


def _cmd_pin(args: argparse.Namespace, command: list[str]) -> int:
    target = collect_target(args, command)
    save_baseline(args.output, create_baseline(target))
    print(f"pinned {len(target.tools)} tool(s) from '{target.server_name}' to {args.output}", file=sys.stderr)
    return 0


def _cmd_eval(args: argparse.Namespace, command: list[str]) -> int:
    cases = load_cases(args.cases)
    target = collect_target(args, command)
    errors, warnings = check_cases(cases, target.tools)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"ok: {len(cases.cases)} case(s) are consistent with {len(target.tools)} tool(s) of '{target.server_name}'")
        return 0
    if not args.model:
        raise UsageError("give at least one --model (or use --dry-run)")
    if args.repeats < 1:
        raise UsageError("--repeats must be at least 1")
    if not 0.0 <= args.min_pass_rate <= 1.0:
        raise UsageError("--min-pass-rate must be between 0 and 1")

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise UsageError(f"environment variable {args.api_key_env} is not set or empty")
    client = ChatClient(args.base_url, api_key=api_key, timeout=args.llm_timeout, allow_insecure=args.allow_insecure)
    judge = (client, args.judge_model) if args.judge_model else None
    toolset = build_toolset(target.tools)
    print(f"sending the definitions of {len(toolset.functions)} tool(s) to {args.base_url} "
          f"({len(cases.cases)} cases x {args.repeats} runs x {len(args.model)} model(s)); tools are not executed",
          file=sys.stderr)
    progress = None if args.quiet else (lambda line: print(line, file=sys.stderr))
    results = run_eval(client, args.model, toolset, cases, repeats=args.repeats,
                       temperature=args.temperature, judge=judge, progress=progress)

    render = {"text": eval_text, "json": eval_json, "markdown": eval_markdown}[args.format]
    _write(render(results, target, len(cases.cases), args.repeats), args.output)

    if results and all(r.error for r in results):
        print("error: every request to the model API failed", file=sys.stderr)
        return 2
    if any(summarize_model(m, results).pass_rate < args.min_pass_rate for m in args.model):
        return 1
    return 0


def _cmd_rules() -> int:
    for rule in RULES:
        print(f"{rule.id}  [{rule.default_severity.label:<8}] {rule.title}")
        print(f"          {rule.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    options, command = _split_command(raw)
    parser = build_parser()
    args = parser.parse_args(options)
    try:
        if args.command == "scan":
            return _cmd_scan(args, command)
        if args.command == "pin":
            return _cmd_pin(args, command)
        if args.command == "eval":
            return _cmd_eval(args, command)
        return _cmd_rules()
    except (UsageError, McpError, BaselineError, CaseError, LlmError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
