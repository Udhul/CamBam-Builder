"""MCP stdio adapter for modern and selected legacy protocol eras.

The adapter intentionally uses the low-level server from the official Python
SDK.  ``Server.run`` owns era selection, initialization, cancellation and EOF;
the adapter's middleware limits legacy negotiation to the versions with tested
client support and adds workspace metadata to SDK-owned results.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import anyio
import mcp_types as types
from mcp.server.caching import CacheHint
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from mcp_types import (
    ErrorData,
    JSONRPCError,
    PROTOCOL_VERSION_META_KEY,
    UNSUPPORTED_PROTOCOL_VERSION,
    SERVER_INFO_META_KEY,
    jsonrpc_message_adapter,
)
from pydantic import ValidationError

from .schema import tool_definitions
from .service import DocumentService


PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSIONS = ("2025-06-18", "2025-11-25")
SUPPORTED_PROTOCOL_VERSIONS = (*LEGACY_PROTOCOL_VERSIONS, PROTOCOL_VERSION)
# A 10 MiB UTF-8 XML document can expand when represented as a JSON string.
# Keep framing bounded while leaving room for the documented XML payload plus
# JSON escaping and the surrounding request envelope.
MAX_INPUT_LINE_BYTES = 32 * 1024 * 1024
WORKSPACE_META_KEY = "cambam-builder/workspace"


class StrictJSONError(ValueError):
    """A stdio line rejected before the SDK's JSON-RPC decoder."""


class _DuplicateKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _strict_decode(line: bytes) -> str:
    """Validate strict JSON while retaining the original line for the SDK."""
    if len(line) > MAX_INPUT_LINE_BYTES:
        raise StrictJSONError("JSON-RPC input line exceeds 32 MiB")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJSONError("JSON-RPC input is not UTF-8") from exc
    try:
        def finite_float(value: str) -> float:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(value)
            return number

        json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=finite_float,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise StrictJSONError("JSON-RPC input is not strict JSON") from exc
    return text


class _StrictInput:
    """Bounded async line iterator for the SDK stdio transport.

    Reading in chunks keeps a malicious unterminated line from making the
    process allocate an unbounded buffer.  Decoder failures are reported by
    the framing context without echoing input data.
    """

    def __init__(self, source: Any):
        self._source = source
        self._buffer = bytearray()
        self._eof = False
        self._oversized = False

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[: newline + 1])
                del self._buffer[: newline + 1]
                oversized = self._oversized or len(line) > MAX_INPUT_LINE_BYTES
                self._oversized = False
                if oversized:
                    raise StrictJSONError("JSON-RPC input line exceeds 32 MiB")
                return _strict_decode(line)

            if self._eof:
                if self._oversized:
                    self._oversized = False
                    raise StrictJSONError("JSON-RPC input line exceeds 32 MiB")
                if not self._buffer:
                    raise StopAsyncIteration
                line = bytes(self._buffer)
                self._buffer.clear()
                oversized = self._oversized or len(line) > MAX_INPUT_LINE_BYTES
                self._oversized = False
                if oversized:
                    raise StrictJSONError("JSON-RPC input line exceeds 32 MiB")
                return _strict_decode(line)

            chunk = await self._source.read(8192)
            if not chunk:
                self._eof = True
                continue
            self._buffer.extend(chunk)
            if len(self._buffer) > MAX_INPUT_LINE_BYTES:
                # Keep at most one chunk while draining to the newline.
                self._oversized = True
                if b"\n" not in self._buffer:
                    self._buffer.clear()


def _serialized(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class _UTF8Output:
    """Async text sink that always writes the stdio wire as UTF-8 bytes."""

    def __init__(self, stream: Any):
        self._stream = stream

    @staticmethod
    def _write(stream: Any, data: bytes) -> None:
        view = memoryview(data)
        while len(view):
            written = stream.write(view)
            if not written:
                raise OSError("stdio output closed")
            view = view[written:]

    async def write(self, value: str) -> int:
        data = value.encode("utf-8")
        await anyio.to_thread.run_sync(self._write, self._stream, data)
        return len(value)

    async def flush(self) -> None:
        await anyio.to_thread.run_sync(self._stream.flush)


@asynccontextmanager
async def strict_stdio_server():
    """Provide bounded UTF-8 streams to the SDK's public ``Server.run``.

    The SDK's convenience ``stdio_server`` intentionally drops decoder
    exceptions.  This small framing layer keeps the same SessionMessage stream
    contract while returning bounded JSON-RPC parse/invalid-request errors and
    still leaves lifecycle, era selection, cancellation and EOF to ``Server``.
    """
    source = anyio.wrap_file(sys.stdin.buffer.raw)
    sink = _UTF8Output(sys.stdout.buffer.raw)
    read_send, read_stream = anyio.create_memory_object_stream(0)
    write_send, write_stream = anyio.create_memory_object_stream(0)

    async def emit_error(code: int, message: str, request_id: Any = None) -> None:
        error = JSONRPCError(
            jsonrpc="2.0", id=request_id, error=ErrorData(code=code, message=message)
        )
        try:
            await write_send.send(SessionMessage(error))
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            return

    async def reader() -> None:
        lines = _StrictInput(source)
        era_selected = False
        async with read_send:
            while True:
                try:
                    line = await lines.__anext__()
                except StopAsyncIteration:
                    return
                except StrictJSONError:
                    await emit_error(types.PARSE_ERROR, "Parse error")
                    continue
                try:
                    message = jsonrpc_message_adapter.validate_json(line, by_name=False)
                except ValidationError as exc:
                    kind = exc.errors()[0].get("type") if exc.errors() else ""
                    if kind == "json_invalid":
                        await emit_error(types.PARSE_ERROR, "Parse error")
                    else:
                        await emit_error(types.INVALID_REQUEST, "Invalid request")
                    continue
                except Exception:
                    await emit_error(types.INVALID_REQUEST, "Invalid request")
                    continue
                if isinstance(message, types.JSONRPCRequest) and not era_selected:
                    params = message.params
                    metadata = params.get("_meta") if isinstance(params, Mapping) else None
                    opens_modern = (
                        message.method != "initialize"
                        and isinstance(metadata, Mapping)
                        and PROTOCOL_VERSION_META_KEY in metadata
                    )
                    if message.method == "initialize" or opens_modern:
                        era_selected = True
                    else:
                        # The SDK treats every metadata-free first request as a
                        # legacy opener.  Refuse that implicit downgrade: only
                        # initialize may select legacy, and modern direct calls
                        # must carry their reserved protocol envelope.
                        await emit_error(
                            types.INVALID_PARAMS,
                            "Missing modern protocol metadata",
                            message.id,
                        )
                        continue
                await read_send.send(SessionMessage(message))

    async def writer() -> None:
        async with write_stream:
            async for session_message in write_stream:
                await sink.write(
                    session_message.message.model_dump_json(by_alias=True, exclude_unset=True)
                    + "\n"
                )
                await sink.flush()

    async with read_stream, write_send:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(reader)
            task_group.start_soon(writer)
            try:
                yield read_stream, write_send
            finally:
                task_group.cancel_scope.cancel()


def _tool_models() -> list[types.Tool]:
    models: list[types.Tool] = []
    for definition in tool_definitions():
        annotations = definition.get("annotations")
        models.append(
            types.Tool(
                name=definition["name"],
                description=definition.get("description"),
                inputSchema=definition["inputSchema"],
                outputSchema=definition.get("outputSchema"),
                annotations=types.ToolAnnotations.model_validate(annotations) if annotations else None,
            )
        )
    return models


def create_server(service: DocumentService) -> Server:
    """Build an SDK server exposing the implemented document and authoring tools."""
    known_tools = frozenset(definition["name"] for definition in tool_definitions())
    protocol_reported = False

    def report_protocol(selected: str) -> None:
        nonlocal protocol_reported
        if protocol_reported:
            return
        print(
            "CAMBAM_MCP_PROTOCOL "
            + json.dumps(
                {
                    "protocol_version": selected,
                    "mode": "legacy" if selected in LEGACY_PROTOCOL_VERSIONS else "modern",
                },
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        protocol_reported = True

    async def on_list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=_tool_models(), ttlMs=0, cacheScope="private", resultType="complete"
        )

    async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name not in known_tools:
            # Finding a tool is protocol dispatch.  The documented application
            # envelope is reserved for a known tool's argument/domain result.
            raise MCPError(code=types.INVALID_PARAMS, message="Unknown tool")
        arguments = params.arguments if params.arguments is not None else {}
        try:
            envelope = await service.call_tool(params.name, arguments)
        except Exception:
            # Tool exceptions are protocol failures only when the tool itself
            # cannot produce its documented application envelope.  Do not leak
            # exception text or traceback to a client.
            envelope = {
                "schema_version": 1,
                "ok": False,
                "workspace_id": service.workspace.id,
                "document": None,
                "revision": None,
                "request_id": None,
                "replayed": False,
                "data": None,
                "diagnostics": [],
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Operation failed without publication",
                    "field": None,
                    "current_revision": None,
                },
            }
        return types.CallToolResult(
            content=[types.TextContent(text=_serialized(envelope))],
            structuredContent=envelope,
            isError=not bool(envelope.get("ok")),
            resultType="complete",
        )

    server = Server(
        "cambam-builder",
        version="0.1.0",
        description="Local CamBam document and authoring tools",
        instructions=(
            "Use workspace_id="
            + service.workspace.id
            + " in every tool call. Client-local paths are not server workspace paths: "
            "use document_import with complete .cb XML, edit the returned handle, then "
            "use document_export and save its content unchanged with the returned hash. "
            "Legacy clients omit only modern per-request protocol metadata."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        cache_hints={
            "tools/list": CacheHint(0, "private"),
            "server/discover": CacheHint(0, "private"),
        },
    )

    async def protocol_middleware(ctx: Any, call_next: Any) -> Any:
        # The SDK negotiates legacy initialize inline.  Inspect the requested
        # version there because ctx.protocol_version is still its seed until
        # the initialize result commits.
        if ctx.method == "initialize":
            params = ctx.params if isinstance(ctx.params, Mapping) else {}
            requested = params.get("protocolVersion")
            if requested not in LEGACY_PROTOCOL_VERSIONS:
                raise MCPError(
                    code=UNSUPPORTED_PROTOCOL_VERSION,
                    message="Unsupported protocol version",
                    data={"supported": list(SUPPORTED_PROTOCOL_VERSIONS), "requested": requested},
                )
        elif ctx.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise MCPError(code=UNSUPPORTED_PROTOCOL_VERSION, message="Unsupported protocol version")

        # ``ping`` is present in the SDK's low-level registration for the
        # handshake era, while its 2026 method table omits it.  Short-circuit
        # this harmless capability probe in middleware and keep the SDK-owned
        # result metadata on modern responses.
        if ctx.method == "ping":
            report_protocol(ctx.protocol_version)
            if ctx.protocol_version == PROTOCOL_VERSION:
                return {
                    "resultType": "complete",
                    "_meta": {SERVER_INFO_META_KEY: server.server_info_stamp},
                }
            return {}

        result = await call_next(ctx)
        if ctx.method == "initialize" and isinstance(result, Mapping):
            result = dict(result)
            metadata = dict(result.get("_meta") or {})
            metadata[WORKSPACE_META_KEY] = dict(service.bootstrap)
            result["_meta"] = metadata
        if ctx.method == "server/discover" and isinstance(result, Mapping):
            # The SDK's default discovery handler and serializer own the
            # normative envelope and serverInfo stamp.  Add only our contract
            # metadata in middleware after that serialization step.
            result = dict(result)
            metadata = dict(result.get("_meta") or {})
            metadata[WORKSPACE_META_KEY] = dict(service.bootstrap)
            result["_meta"] = metadata
        selected = requested if ctx.method == "initialize" else ctx.protocol_version
        report_protocol(selected)
        return result

    server.middleware.append(protocol_middleware)
    return server


async def _run_server(service: DocumentService) -> None:
    server = create_server(service)
    async with strict_stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run_stdio(root: str) -> None:
    """Start the local server; called by the guarded launcher."""
    # Stdio diagnostics are deliberately limited to the bootstrap/protocol lines and
    # structured tool results; suppress SDK/framework logger output that could
    # disclose imported labels, XML details or local paths.
    logging.disable(logging.CRITICAL)
    service = DocumentService(root)
    print(
        "CAMBAM_MCP_WORKSPACE "
        + json.dumps(service.bootstrap, separators=(",", ":"), ensure_ascii=True, allow_nan=False),
        file=sys.stderr,
        flush=True,
    )
    try:
        anyio.run(_run_server, service)
    except KeyboardInterrupt:
        return


__all__ = [
    "LEGACY_PROTOCOL_VERSIONS",
    "MAX_INPUT_LINE_BYTES",
    "PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "create_server",
    "run_stdio",
]
