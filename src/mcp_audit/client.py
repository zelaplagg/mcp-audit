"""A minimal, dependency-free MCP client.

It only ever performs the handshake (`initialize`) and `tools/list`. It never calls a tool.
Two transports are supported: a local server over stdio and a remote server over Streamable HTTP.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import ssl
from typing import Any, Iterator, Protocol

from . import __version__

PROTOCOL_VERSION = "2025-06-18"
MAX_PAGES = 50
MAX_BODY_BYTES = 10_000_000
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SYSTEM_CA_FILES = ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt")


def make_ssl_context() -> ssl.SSLContext:
    """A verifying TLS context that also works on Pythons without a CA bundle (python.org builds on macOS).

    Verification is never disabled. If the default context has no trusted roots and SSL_CERT_FILE is not
    set, fall back to certifi (when installed) or the operating system's bundle.
    """
    context = ssl.create_default_context()
    if os.environ.get("SSL_CERT_FILE") or context.cert_store_stats().get("x509_ca", 0) > 0:
        return context
    candidates: list[str] = []
    try:
        import certifi  # type: ignore[import-not-found]

        candidates.append(certifi.where())
    except ImportError:
        pass
    candidates.extend(_SYSTEM_CA_FILES)
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ssl.create_default_context(cafile=path)
            except (ssl.SSLError, OSError):
                continue
    return context


def tls_hint(exc: BaseException) -> str:
    """Extra advice for certificate errors, empty for anything else."""
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return (" (Python cannot verify the server certificate. On macOS run "
                "'/Applications/Python 3.x/Install Certificates.command' or set SSL_CERT_FILE=/etc/ssl/cert.pem)")
    return ""


class McpError(Exception):
    """Connection, protocol or server error."""


class Transport(Protocol):
    def request(self, method: str, params: dict[str, Any] | None = None) -> Any: ...

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None: ...

    def close(self) -> None: ...


class StdioTransport:
    """Runs the server as a subprocess and talks newline-delimited JSON-RPC over its pipes."""

    def __init__(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
        inherit_env: bool = True,
    ) -> None:
        if not command:
            raise McpError("no server command given")
        self._timeout = timeout
        self._next_id = 0
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
        if inherit_env:
            merged_env = dict(os.environ)
        else:  # an untrusted server must not see our secrets: pass only what a process needs to start
            keep = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "SYSTEMROOT", "USERPROFILE")
            merged_env = {k: v for k, v in os.environ.items() if k in keep}
        merged_env.update(env or {})
        try:
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
                cwd=cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise McpError(f"cannot start server: {exc}") from exc
        self._threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_tail.append(line.rstrip())

    def _send(self, message: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise McpError(f"server closed its input{self._stderr_hint()}") from exc

    def _stderr_hint(self) -> str:
        time.sleep(0.05)
        tail = " | ".join(self._stderr_tail)
        return f" (stderr: {tail[:300]})" if tail else ""

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._next_id += 1
        request_id = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpError(f"timed out waiting for response to '{method}'")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise McpError(f"timed out waiting for response to '{method}'") from None
            if line is None:
                raise McpError(f"server closed the connection{self._stderr_hint()}")
            try:
                incoming = json.loads(line)
            except json.JSONDecodeError:
                continue  # some servers print log lines to stdout; ignore them
            if not isinstance(incoming, dict):
                continue
            if "method" in incoming:
                if "id" in incoming:  # a request from the server: we support nothing
                    self._send(
                        {
                            "jsonrpc": "2.0",
                            "id": incoming["id"],
                            "error": {"code": -32601, "message": "Method not found"},
                        }
                    )
                continue  # notifications are ignored
            if incoming.get("id") != request_id:
                continue
            if "error" in incoming:
                raise McpError(_format_rpc_error(method, incoming["error"]))
            return incoming.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        for thread in self._threads:
            thread.join(timeout=1)
        for pipe in (self._proc.stdout, self._proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except (OSError, ValueError):
                pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects, so credentials are never forwarded to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


class HttpTransport:
    """Streamable HTTP transport (stateless or session based, JSON or SSE responses)."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        allow_insecure: bool = False,
    ) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise McpError("server URL must start with http:// or https://")
        self._headers = dict(headers or {})
        has_credentials = any(k.lower() == "authorization" for k in self._headers)
        if (
            parsed.scheme == "http"
            and has_credentials
            and parsed.hostname not in _LOCAL_HOSTS
            and not allow_insecure
        ):
            raise McpError(
                "refusing to send credentials over plain http; use https or pass --allow-insecure"
            )
        self._url = url
        self._timeout = timeout
        self._next_id = 0
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=make_ssl_context()))

    def _post(self, payload: dict[str, Any]) -> tuple[bytes, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": f"mcp-audit/{__version__}",
            **self._headers,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol_version:
            headers["MCP-Protocol-Version"] = self._protocol_version
        request = urllib.request.Request(
            self._url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                body = response.read(MAX_BODY_BYTES + 1)
                content_type = response.headers.get("Content-Type", "")
                session_id = response.headers.get("Mcp-Session-Id")
        except urllib.error.HTTPError as exc:
            exc.close()
            raise McpError(f"HTTP {exc.code} from server") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise McpError(f"cannot reach server: {exc}{tls_hint(exc)}") from exc
        except ValueError as exc:
            raise McpError(f"invalid request: {exc}") from exc
        if len(body) > MAX_BODY_BYTES:
            raise McpError("server response is too large")
        if session_id:
            self._session_id = session_id
        return body, content_type

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._next_id += 1
        request_id = self._next_id
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        body, content_type = self._post(payload)

        messages: list[Any]
        if "text/event-stream" in content_type:
            messages = list(parse_sse(body.decode("utf-8", errors="replace")))
        else:
            try:
                decoded = json.loads(body.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                raise McpError("server returned invalid JSON") from exc
            messages = decoded if isinstance(decoded, list) else [decoded]

        for message in messages:
            if isinstance(message, dict) and message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise McpError(_format_rpc_error(method, message["error"]))
                result = message.get("result")
                if method == "initialize" and isinstance(result, dict):
                    version = result.get("protocolVersion")
                    if isinstance(version, str):
                        self._protocol_version = version
                return result
        raise McpError(f"no response to '{method}' in server reply")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._post(payload)

    def close(self) -> None:
        if not self._session_id:
            return
        request = urllib.request.Request(
            self._url,
            headers={**self._headers, "Mcp-Session-Id": self._session_id},
            method="DELETE",
        )
        try:
            self._opener.open(request, timeout=5).close()
        except (urllib.error.URLError, OSError, ValueError):
            pass  # best effort: many servers do not support session termination


def parse_sse(text: str) -> Iterator[Any]:
    """Yield the JSON payload of every event in a Server-Sent Events body."""
    for block in text.replace("\r\n", "\n").split("\n\n"):
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
        if not data_lines:
            continue
        try:
            yield json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue


def _format_rpc_error(method: str, error: Any) -> str:
    if isinstance(error, dict):
        return f"'{method}' failed: {error.get('message', 'unknown error')} (code {error.get('code')})"
    return f"'{method}' failed: {error}"


class McpClient:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def __enter__(self) -> "McpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._transport.close()

    def initialize(self) -> dict[str, Any]:
        result = self._transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mcp-audit", "version": __version__},
            },
        )
        if not isinstance(result, dict):
            raise McpError("server returned an invalid initialize result")
        self._transport.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            result = self._transport.request("tools/list", {"cursor": cursor} if cursor else {})
            if not isinstance(result, dict):
                raise McpError("server returned an invalid tools/list result")
            page = result.get("tools", [])
            if isinstance(page, list):
                tools.extend(item for item in page if isinstance(item, dict))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
        raise McpError(f"server returned more than {MAX_PAGES} pages of tools")
