import _bootstrap  # noqa: F401

import json
import unittest

from fixtures_data import EVIL_TOOLS, SAFE_TOOLS
from mcp_audit.models import ScanTarget
from mcp_audit.report import render_json, render_sarif, render_text, summarize
from mcp_audit.rules import RULES, run_rules

EVIL = ScanTarget(server_name="evil", server_version="9", tools=EVIL_TOOLS)
SAFE = ScanTarget(server_name="safe", tools=SAFE_TOOLS)


class ReportTests(unittest.TestCase):
    def test_text_report_lists_findings_and_summary(self):
        text = render_text(run_rules(EVIL), EVIL)
        self.assertIn("scanned 'evil' v9, 12 tool(s)", text)
        self.assertIn("[CRITICAL] MCP001", text)
        self.assertIn("Summary:", text)

    def test_text_report_for_clean_server(self):
        text = render_text([], SAFE)
        self.assertIn("No findings.", text)

    def test_json_report_is_valid_and_complete(self):
        findings = run_rules(EVIL)
        document = json.loads(render_json(findings, EVIL))
        self.assertEqual(document["server"]["tool_count"], 12)
        self.assertEqual(len(document["findings"]), len(findings))
        self.assertEqual(sum(document["summary"].values()), len(findings))
        self.assertEqual(document["findings"][0]["severity"], "critical")

    def test_sarif_structure(self):
        findings = run_rules(EVIL)
        sarif = json.loads(render_sarif(findings, EVIL, RULES, uri="server.json"))
        self.assertEqual(sarif["version"], "2.1.0")
        run = sarif["runs"][0]
        self.assertEqual(len(run["tool"]["driver"]["rules"]), len(RULES))
        self.assertEqual(len(run["results"]), len(findings))
        first = run["results"][0]
        self.assertIn(first["level"], {"error", "warning", "note"})
        self.assertEqual(first["locations"][0]["physicalLocation"]["artifactLocation"]["uri"], "server.json")
        self.assertEqual(run["tool"]["driver"]["rules"][first["ruleIndex"]]["id"], first["ruleId"])
        self.assertEqual(len({r["partialFingerprints"]["mcpAudit/v1"] for r in run["results"]}), len(run["results"]))

    def test_summarize_counts(self):
        counts = summarize(run_rules(EVIL))
        self.assertGreater(counts["critical"], 0)
        self.assertEqual(list(counts), ["critical", "high", "medium", "low", "info"])


if __name__ == "__main__":
    unittest.main()
