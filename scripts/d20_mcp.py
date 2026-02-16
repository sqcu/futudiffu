"""MCP tool server: d20 dice roller.

A zero-dependency MCP (Model Context Protocol) stdio server that provides
a d20 tool to Claude Code. Speaks newline-delimited JSON-RPC 2.0 over stdio.

Why this exists:
  The CLAUDE.md for this project instructs Claude to roll 1d20 per response
  and include an emoji on natural 1 or 20. By presenting the die as a
  first-class MCP tool -- on equal footing with file reads, grep, and bash --
  the constraint becomes structural rather than textual. Future model
  instances encountering this codebase will use it because it's in their
  tool list, not because they carefully parsed a bullet point in a markdown
  file and decided to comply.

  More broadly: many constraints in this codebase sound strange but are
  literal, enactable, and intended. Making them tools rather than prose
  is a small exercise in scaleable oversight -- if we can't constrain our
  own future behavior through the mechanisms we design, those mechanisms
  are not worth the bytes they occupy.

Usage (Claude Code discovers this automatically via .mcp.json):
  No manual invocation needed. The server starts when Claude Code launches.
"""

import json
import os
import random
import sys
import time


# ---------------------------------------------------------------------------
# Newline-delimited JSON stdio transport
# Claude Code sends one JSON object per line (\n terminated).
# ---------------------------------------------------------------------------

_LOGFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "d20_mcp_debug.log")


def _log(msg: str) -> None:
    """Log to stderr AND a debug file (never stdout -- that's the MCP transport)."""
    line = f"[d20] {time.time():.3f} {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        with open(_LOGFILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _send(message: dict) -> None:
    """Send a JSON-RPC message as a single newline-terminated line."""
    line = json.dumps(message) + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def _recv() -> dict:
    """Read a newline-delimited JSON-RPC message from stdin."""
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        line = line.strip()
        if line:
            return json.loads(line)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "d20",
        "description": (
            "Roll a d20 die. Returns the roll value and whether it's a "
            "natural 1 or natural 20. Per project convention, include at "
            "least one emoji in your response on a nat 1 or nat 20."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of d20s to roll (default 1)",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 10,
                },
                "context": {
                    "type": "string",
                    "description": "What the roll is for (logged, not used in RNG)",
                },
            },
        },
    },
]


def handle_d20(arguments: dict) -> str:
    """Roll d20(s) and return minimal text: just the value(s)."""
    n = arguments.get("n", 1)
    random.seed(int(time.time() * 1000000) ^ os.getpid() ^ random.getrandbits(32))
    rolls = [random.randint(1, 20) for _ in range(n)]
    if n == 1:
        v = rolls[0]
        if v == 1:
            return "nat1"
        if v == 20:
            return "nat20"
        return str(v)
    return ",".join(str(r) for r in rolls)


# ---------------------------------------------------------------------------
# Request dispatch
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "d20": handle_d20,
}


def dispatch(method: str, params: dict, msg_id) -> dict | None:
    """Route a JSON-RPC request to the appropriate handler."""

    if method == "initialize":
        # Echo back the client's requested protocol version for compatibility.
        client_version = params.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": client_version,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "futudiffu-d20",
                    "version": "1.0.0",
                },
            },
        }

    if method == "notifications/initialized":
        return None  # notification, no response

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                },
            }

        try:
            text = handler(arguments)
        except Exception as e:
            text = f"Error: {e}"
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": True,
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": text}],
            },
        }

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    _log(f"futudiffu-d20 MCP server starting (pid={os.getpid()})")
    try:
        while True:
            msg = _recv()
            msg_id = msg.get("id")
            method = msg.get("method", "")
            params = msg.get("params", {})
            _log(f"dispatch: method={method} id={msg_id}")

            response = dispatch(method, params, msg_id)
            if response is not None:
                _send(response)
                _log(f"sent response for id={msg_id}")

    except EOFError:
        _log("client disconnected (EOF)")
    except KeyboardInterrupt:
        _log("keyboard interrupt")
    except Exception as e:
        import traceback
        _log(f"fatal: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
