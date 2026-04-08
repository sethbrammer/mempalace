"""
mempalace.server — single long-lived HTTP MCP daemon for shared multi-client palaces.

Why this exists
---------------
Upstream `mempalace.mcp_server` is a stdio MCP server: each client spawns its own
subprocess that opens its own ChromaDB / KG handles against the same files.  That
breaks down when multiple machines want to share one palace:

  * Subprocesses can outlive their clients and hold DB locks (real incident:
    PID 2549 idle 33 minutes, blocking KG queries for everyone).
  * Concurrent writes from two machines race on ChromaDB / SQLite state.
  * There's no ergonomic way to register a remote palace with Claude Code or
    Claude Desktop other than wrapping `ssh ... docker exec ...` in a shell
    bridge.

This module replaces all of that with one process that:

  1. Owns the KnowledgeGraph + ChromaDB clients (single Python process, single
     in-memory state).
  2. Serves MCP over Streamable HTTP using the official `mcp` Python SDK
     (FastMCP), so clients connect via `claude mcp add --transport http ...`.
  3. Serializes every tool call through a single asyncio.Lock — the implicit
     FIFO "queue" that makes concurrent multi-machine use safe.
  4. Exposes a plain `/health` endpoint for monitoring + smoke tests.

The existing stdio entry point (`mempalace.mcp_server`) is left untouched so
upstream users are unaffected, and so a fallback bridge can still operate
during migration windows.

Run
---
    pip install mempalace[server]   # adds mcp[cli] and httpx
    mempalace-server                # listens on $MEMPALACE_HOST:$MEMPALACE_PORT (defaults 0.0.0.0:8765)

Then on each client:

    claude mcp add --transport http --scope user mempalace http://<host>:8765/mcp
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import signal
import sys
import time
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# Importing mcp_server triggers module-level init of KnowledgeGraph and
# MempalaceConfig — that's intentional, we want a single in-process owner.
from mempalace import mcp_server as _stdio
from mempalace.version import __version__

logger = logging.getLogger("mempalace.server")


# ---------------------------------------------------------------------------
# Concurrency: one global lock around every tool dispatch.
# ---------------------------------------------------------------------------
# All mempalace tool handlers are sync code that touches ChromaDB / KG SQLite.
# We funnel them through this lock so there is *never* more than one tool call
# executing against the data layer at any moment, regardless of how many
# clients are connected.  Reads are serialized too — we trade some throughput
# for absolute safety, which matches mempalace's "correctness over speed"
# stance.  If this becomes a bottleneck the lock can be split into a read /
# write pair (asyncio doesn't ship one, but `aiorwlock` is a one-line drop-in).

_dispatch_lock = asyncio.Lock()
_stats = {
    "started_at": time.time(),
    "requests_total": 0,
    "requests_in_flight": 0,
    "queue_high_water": 0,
    "last_error": None,
}


def _wrap_handler(name: str, handler):
    """Wrap a sync mempalace tool handler so FastMCP can register it.

    The wrapper:
      * preserves the original handler's signature so FastMCP can introspect
        argument names, types, and defaults to build the input schema;
      * acquires the global dispatch lock;
      * runs the sync handler in a worker thread (asyncio.to_thread) so the
        event loop stays responsive for /health and for queueing other
        requests;
      * tracks in-flight + total counters and the last error string.
    """

    sig = inspect.signature(handler)

    @functools.wraps(handler)
    async def wrapper(**kwargs: Any):
        _stats["requests_in_flight"] += 1
        if _stats["requests_in_flight"] > _stats["queue_high_water"]:
            _stats["queue_high_water"] = _stats["requests_in_flight"]
        try:
            async with _dispatch_lock:
                result = await asyncio.to_thread(handler, **kwargs)
            _stats["requests_total"] += 1
            return result
        except Exception as exc:  # noqa: BLE001 — propagate to MCP error envelope
            _stats["last_error"] = f"{name}: {type(exc).__name__}: {exc}"
            logger.exception("tool %s failed", name)
            raise
        finally:
            _stats["requests_in_flight"] -= 1

    # FastMCP introspects __signature__ to build the tool input schema.
    # Without this it would see wrapper's `**kwargs` and produce a schema
    # with no parameters at all.
    wrapper.__signature__ = sig  # type: ignore[attr-defined]
    wrapper.__name__ = name
    wrapper.__doc__ = handler.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Build the FastMCP app.
# ---------------------------------------------------------------------------

def _build_app() -> FastMCP:
    host = os.environ.get("MEMPALACE_HOST", "0.0.0.0")
    port = int(os.environ.get("MEMPALACE_PORT", "8765"))

    app = FastMCP(
        name="mempalace",
        instructions=(
            "MemPalace — shared multi-client memory palace served over HTTP. "
            "Tool calls are serialized through a single FIFO queue, so it is "
            "safe to invoke from any number of clients concurrently."
        ),
        host=host,
        port=port,
    )

    # Register every tool from the upstream stdio server's TOOLS dict.
    # We import the handlers from mempalace.mcp_server so we get the *exact*
    # same code paths the upstream server uses — no duplication, no schema
    # drift, no separate maintenance burden.
    for tool_name, entry in _stdio.TOOLS.items():
        wrapped = _wrap_handler(tool_name, entry["handler"])
        app.add_tool(
            wrapped,
            name=tool_name,
            description=entry["description"],
        )
        logger.debug("registered tool %s", tool_name)

    # ----- /health (plain HTTP, not MCP) -----
    # Used by smoke tests, monitoring, and the container's readiness probe.
    @app.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        uptime = int(time.time() - _stats["started_at"])
        try:
            tool_count = len(_stdio.TOOLS)
        except Exception:
            tool_count = -1
        body = {
            "status": "ok",
            "version": __version__,
            "uptime_s": uptime,
            "tools": tool_count,
            "requests_total": _stats["requests_total"],
            "requests_in_flight": _stats["requests_in_flight"],
            "queue_high_water": _stats["queue_high_water"],
            "last_error": _stats["last_error"],
        }
        return JSONResponse(body)

    return app


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def _install_signal_handlers() -> None:
    """Best-effort graceful shutdown logging on SIGTERM / SIGINT.

    FastMCP / uvicorn already install their own handlers; this just makes
    sure we leave a breadcrumb in the container logs explaining *why* the
    daemon went down, which is invaluable when debugging restart loops.
    """

    def _bye(signum, _frame):  # noqa: ANN001
        name = signal.Signals(signum).name
        logger.info("received %s — shutting down (uptime %ds, served %d requests)",
                    name,
                    int(time.time() - _stats["started_at"]),
                    _stats["requests_total"])
        # Re-raise default behavior — uvicorn's handler will run after this.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _bye)
        except (ValueError, OSError):
            pass  # not in main thread, or signal not supported on this OS


def main() -> None:
    log_level = os.environ.get("MEMPALACE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info(
        "mempalace-server %s starting on %s:%s (palace_path=%s)",
        __version__,
        os.environ.get("MEMPALACE_HOST", "0.0.0.0"),
        os.environ.get("MEMPALACE_PORT", "8765"),
        getattr(_stdio._config, "palace_path", "?"),
    )
    _install_signal_handlers()
    app = _build_app()
    # Streamable HTTP serves MCP at /mcp by default; /health is mounted via
    # the custom_route registered in _build_app().
    app.run(transport="streamable-http")


if __name__ == "__main__":
    main()
