"""A tiny stdio MCP server for tests. Usage: mcp_stdio_server.py CONFIG.json

CONFIG.json may contain: tools, instructions, serverInfo, page_size.
FIXTURE_MODE=hang never answers; FIXTURE_MODE=chatty prints noise and sends a server request first.
"""
import json
import os
import sys
import time


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    with open(sys.argv[1], encoding="utf-8") as handle:
        config = json.load(handle)
    mode = os.environ.get("FIXTURE_MODE", "")
    if mode == "hang":
        time.sleep(60)
        return
    tools = config.get("tools", [])
    page_size = config.get("page_size", 0)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        message_id = message.get("id")
        method = message.get("method")
        if message_id is None:
            continue  # notification
        if mode == "chatty":
            sys.stdout.write("starting up, this is not JSON\n")
            send({"jsonrpc": "2.0", "id": 9000, "method": "ping"})
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": config.get("serverInfo", {"name": "fixture", "version": "1.0.0"}),
            }
            if "instructions" in config:
                result["instructions"] = config["instructions"]
        elif method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            start = int(cursor) if cursor else 0
            if page_size:
                result = {"tools": tools[start : start + page_size]}
                if start + page_size < len(tools):
                    result["nextCursor"] = str(start + page_size)
            else:
                result = {"tools": tools}
        elif method == "ping":
            result = {}
        else:
            send({"jsonrpc": "2.0", "id": message_id, "error": {"code": -32601, "message": "Method not found"}})
            continue
        send({"jsonrpc": "2.0", "id": message_id, "result": result})


if __name__ == "__main__":
    main()
