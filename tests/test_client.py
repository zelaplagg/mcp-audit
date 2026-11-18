import _bootstrap  # noqa: F401

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from fixtures_data import SAFE_TOOLS, write_fixture
from mcp_audit.client import HttpTransport, McpClient, McpError, StdioTransport, parse_sse

FIXTURE = str(Path(__file__).parent / "mcp_stdio_server.py")


class StdioClientTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _client(self, tools=None, timeout=10.0, **extra):
        config = write_fixture(self._tmp.name, tools if tools is not None else SAFE_TOOLS, **extra)
        return McpClient(StdioTransport([sys.executable, FIXTURE, str(config)], timeout=timeout))

    def test_lists_tools_across_pages(self):
        with self._client(page_size=1) as client:
            init = client.initialize()
            tools = client.list_tools()
        self.assertEqual(init["serverInfo"]["name"], "fixture")
        self.assertEqual([t["name"] for t in tools], [t["name"] for t in SAFE_TOOLS])

    def test_returns_server_instructions(self):
        with self._client(instructions="Be nice.") as client:
            self.assertEqual(client.initialize()["instructions"], "Be nice.")

    def test_ignores_stdout_noise_and_answers_server_requests(self):
        with mock.patch.dict("os.environ", {"FIXTURE_MODE": "chatty"}):
            with self._client() as client:
                client.initialize()
                self.assertEqual(len(client.list_tools()), len(SAFE_TOOLS))

    def test_times_out_when_server_never_answers(self):
        with mock.patch.dict("os.environ", {"FIXTURE_MODE": "hang"}):
            with self._client(timeout=1.0) as client:
                with self.assertRaisesRegex(McpError, "timed out"):
                    client.initialize()

    def test_reports_missing_command(self):
        with self.assertRaisesRegex(McpError, "cannot start server"):
            StdioTransport(["/definitely/not/a/real/command"])

    def test_reports_server_that_exits_immediately(self):
        client = McpClient(StdioTransport([sys.executable, "-c", "pass"], timeout=5.0))
        with client:
            with self.assertRaisesRegex(McpError, "closed"):
                client.initialize()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        server = self.server
        if self.headers.get("Authorization") != "Bearer secret":
            self.send_response(401)
            self.end_headers()
            return
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.seen.append((message.get("method"), self.headers.get("Mcp-Session-Id")))
        if "id" not in message:
            self.send_response(202)
            self.end_headers()
            return
        if message["method"] == "initialize":
            body = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {
                "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                "serverInfo": {"name": "http-fixture", "version": "2"}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Mcp-Session-Id", "session-1")
        else:
            payload = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": SAFE_TOOLS}})
            body = f"event: message\ndata: {payload}\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self):
        self.send_response(200)
        self.end_headers()


class HttpClientTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.seen = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"

    def test_handshake_session_and_sse_response(self):
        transport = HttpTransport(self.url, headers={"Authorization": "Bearer secret"}, timeout=5)
        with McpClient(transport) as client:
            self.assertEqual(client.initialize()["serverInfo"]["name"], "http-fixture")
            tools = client.list_tools()
        self.assertEqual(len(tools), len(SAFE_TOOLS))
        methods = dict(self.server.seen)
        self.assertEqual(methods["tools/list"], "session-1")

    def test_wrong_token_is_reported_without_leaking_it(self):
        transport = HttpTransport(self.url, headers={"Authorization": "Bearer wrong"}, timeout=5)
        with self.assertRaises(McpError) as ctx:
            McpClient(transport).initialize()
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("wrong", str(ctx.exception))

    def test_refuses_credentials_over_plain_http_to_remote_hosts(self):
        with self.assertRaisesRegex(McpError, "plain http"):
            HttpTransport("http://example.com/mcp", headers={"Authorization": "Bearer x"})
        HttpTransport("https://example.com/mcp", headers={"Authorization": "Bearer x"})
        HttpTransport("http://example.com/mcp", headers={"Authorization": "Bearer x"}, allow_insecure=True)

    def test_rejects_non_http_urls(self):
        with self.assertRaises(McpError):
            HttpTransport("file:///etc/passwd")


class SseTests(unittest.TestCase):
    def test_parses_events_and_multiline_data(self):
        text = 'event: message\ndata: {"a":\ndata: 1}\n\ndata: {"b": 2}\n\n: comment\n\n'
        self.assertEqual(list(parse_sse(text)), [{"a": 1}, {"b": 2}])


if __name__ == "__main__":
    unittest.main()


class TlsHelperTests(unittest.TestCase):
    def test_context_verifies_certificates(self):
        import ssl
        from mcp_audit.client import make_ssl_context

        context = make_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_hint_only_for_certificate_errors(self):
        import ssl
        import urllib.error
        from mcp_audit.client import tls_hint

        cert = urllib.error.URLError(ssl.SSLCertVerificationError("unable to get local issuer certificate"))
        self.assertIn("Install Certificates", tls_hint(cert))
        self.assertEqual(tls_hint(urllib.error.URLError(ConnectionRefusedError())), "")
