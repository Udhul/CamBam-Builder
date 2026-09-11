"""Focused subprocess checks for the dual-era MCP stdio boundary."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import threading
import unittest
from unittest.mock import patch
import uuid

if find_spec("mcp") is not None:
    from cambam_builder.mcp_adapter.server import (
        MAX_INPUT_LINE_BYTES,
        StrictJSONError,
        _strict_decode,
    )
else:
    MAX_INPUT_LINE_BYTES = 0
    StrictJSONError = ValueError
    _strict_decode = None


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


class MCPProcess:
    READ_TIMEOUT_SECONDS = 10

    def __init__(self, workspace: Path):
        self.process = subprocess.Popen(
            [str(PYTHON), "-m", "cambam_builder.mcp_adapter", "--workspace", str(workspace)],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stderr is not None
        startup = self._readline(self.process.stderr).decode("utf-8").strip()
        if not startup:
            self.close()
            raise AssertionError("MCP process exited before startup bootstrap")
        self.bootstrap_line = startup

    def _readline(self, stream) -> bytes:
        result: list[bytes] = []
        finished = threading.Event()

        def read() -> None:
            result.append(stream.readline())
            finished.set()

        threading.Thread(target=read, daemon=True).start()
        if not finished.wait(self.READ_TIMEOUT_SECONDS):
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
            raise AssertionError("timed out waiting for MCP subprocess response")
        return result[0]

    def request(self, message: dict) -> dict:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
        self.process.stdin.flush()
        line = self._readline(self.process.stdout)
        self.assert_running()
        return json.loads(line)

    def raw(self, line: bytes) -> dict:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(line + (b"" if line.endswith(b"\n") else b"\n"))
        self.process.stdin.flush()
        return json.loads(self._readline(self.process.stdout))

    def assert_running(self) -> None:
        self.process.poll()
        if self.process.returncode is not None:
            stderr = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
            raise AssertionError(f"MCP process exited {self.process.returncode}: {stderr}")

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()


class MCPProtocolTests(unittest.TestCase):
    def setUp(self):
        if not PYTHON.exists():
            self.skipTest("test interpreter is not available")
        try:
            if version("mcp") != "2.2.0":
                self.skipTest("mcp==2.2.0 is not installed in the test interpreter")
        except PackageNotFoundError:
            self.skipTest("mcp==2.2.0 is not installed in the test interpreter")
        self.workspace = (ROOT / "output" / f"mcp-protocol-{uuid.uuid4().hex}").resolve()
        self.workspace.mkdir()
        self.server = MCPProcess(self.workspace)

    def tearDown(self):
        self.server.close()
        shutil.rmtree(self.workspace, ignore_errors=True)

    def meta(self, protocol_version: str = "2026-07-28") -> dict:
        return {
            "io.modelcontextprotocol/protocolVersion": protocol_version,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "contract-test", "version": "0"},
        }

    def call(self, request_id: int, name: str, arguments: dict | None = None) -> dict:
        return self.server.request(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}, "_meta": self.meta()},
            }
        )

    def new_args(self, request_id: str | None = None) -> dict:
        return {
            "workspace_id": self.workspace_id,
            "request_id": request_id or str(uuid.uuid4()),
            "units": "mm",
            "name": "contract-test",
        }

    @property
    def workspace_id(self) -> str:
        canonical = os.path.normcase(str(self.workspace.resolve()))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def test_startup_discovery_listing_and_create_without_handshake(self):
        prefix, payload = self.server.bootstrap_line.split(" ", 1)
        self.assertEqual(prefix, "CAMBAM_MCP_WORKSPACE")
        bootstrap = json.loads(payload)
        self.assertEqual(bootstrap["workspace_id"], self.workspace_id)

        discovery = self.server.request(
            {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": self.meta()}}
        )
        self.assertEqual(discovery["result"]["supportedVersions"], ["2026-07-28"])
        self.assertEqual(discovery["result"]["_meta"]["cambam-builder/workspace"], bootstrap)

        listing = self.server.request(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": self.meta()}}
        )
        tools = listing["result"]["tools"]
        self.assertEqual([tool["name"] for tool in tools], sorted(tool["name"] for tool in tools))
        self.assertEqual([tool["name"] for tool in tools], [
            "document_close", "document_create", "document_export", "document_import",
            "document_inspect", "document_open", "document_save",
            "geometry_add_arc", "geometry_add_circle", "geometry_add_pline", "geometry_add_points",
            "geometry_add_rectangle", "geometry_add_region", "geometry_add_text", "geometry_bake",
            "geometry_mirror", "geometry_rotate", "geometry_scale", "geometry_translate",
            "geometry_translate_z", "machining_add_drill", "machining_add_engrave",
            "machining_add_pocket", "machining_add_profile", "machining_set_mop_targets",
            "relationship_add_to_group", "relationship_copy_tree",
            "relationship_copy_tree_between",
            "relationship_remove_from_group", "relationship_set_parent",
            "relationship_transfer_tree_between",
        ])
        expected_annotations = {
            "document_close": {"openWorldHint": False, "readOnlyHint": False,
                               "idempotentHint": True, "destructiveHint": True},
            "document_create": {"openWorldHint": False, "readOnlyHint": False,
                                "idempotentHint": True, "destructiveHint": False},
            "document_export": {"openWorldHint": False, "readOnlyHint": True,
                                "idempotentHint": True, "destructiveHint": False},
            "document_import": {"openWorldHint": False, "readOnlyHint": False,
                                "idempotentHint": True, "destructiveHint": False},
            "document_inspect": {"openWorldHint": False, "readOnlyHint": True,
                                 "idempotentHint": True, "destructiveHint": False},
            "document_open": {"openWorldHint": False, "readOnlyHint": False,
                              "idempotentHint": True, "destructiveHint": False},
            "document_save": {"openWorldHint": False, "readOnlyHint": False,
                              "idempotentHint": True, "destructiveHint": False},
            "relationship_copy_tree_between": {
                "openWorldHint": False, "readOnlyHint": False,
                "idempotentHint": True, "destructiveHint": True},
            "relationship_transfer_tree_between": {
                "openWorldHint": False, "readOnlyHint": False,
                "idempotentHint": True, "destructiveHint": True},
        }
        for tool in tools:
            self.assertIn("inputSchema", tool)
            self.assertIn("outputSchema", tool)
            expected_annotations.setdefault(
                tool["name"], {"openWorldHint": False, "readOnlyHint": False,
                                "idempotentHint": True, "destructiveHint": True}
            )
            self.assertEqual(tool["annotations"], expected_annotations[tool["name"]])
        self.assertEqual(listing["result"]["ttlMs"], 0)
        self.assertEqual(listing["result"]["cacheScope"], "private")

        result = self.call(3, "document_create", self.new_args())
        structured = result["result"]["structuredContent"]
        self.assertTrue(structured["ok"])
        self.assertEqual(structured["revision"], 0)
        self.assertEqual(result["result"]["resultType"], "complete")
        self.assertEqual(json.loads(result["result"]["content"][0]["text"]), structured)

    def test_missing_metadata_and_legacy_initialize_are_rejected(self):
        first_ping = self.server.request(
            {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
        )
        self.assertEqual(first_ping["error"]["code"], -32602)
        first_discovery = self.server.request(
            {"jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {}}
        )
        self.assertEqual(first_discovery["error"]["code"], -32602)
        self.server.request(
            {"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {"_meta": self.meta()}}
        )
        missing = self.server.request({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
        self.assertEqual(missing["error"]["code"], -32602)
        invalid = self.server.request(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/list",
             "params": {"_meta": {**self.meta(),
                                    "io.modelcontextprotocol/protocolVersion": 2026}}}
        )
        self.assertEqual(invalid["error"]["code"], -32602)
        legacy = self.server.request(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "legacy", "version": "0"},
                },
            }
        )
        self.assertEqual(legacy["error"]["code"], -32022)

    def test_legacy_initialize_and_tools_work_without_modern_metadata(self):
        initialized = self.server.request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "legacy", "version": "0"},
                },
            }
        )
        self.assertEqual(initialized["result"]["protocolVersion"], "2025-11-25")
        self.assertIn(self.workspace_id, initialized["result"]["instructions"])
        self.assertEqual(
            initialized["result"]["_meta"]["cambam-builder/workspace"]["workspace_id"],
            self.workspace_id,
        )
        listing = self.server.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual([tool["name"] for tool in listing["result"]["tools"]], [
            "document_close", "document_create", "document_export", "document_import",
            "document_inspect", "document_open", "document_save",
            "geometry_add_arc", "geometry_add_circle", "geometry_add_pline", "geometry_add_points",
            "geometry_add_rectangle", "geometry_add_region", "geometry_add_text", "geometry_bake",
            "geometry_mirror", "geometry_rotate", "geometry_scale", "geometry_translate",
            "geometry_translate_z", "machining_add_drill", "machining_add_engrave",
            "machining_add_pocket", "machining_add_profile", "machining_set_mop_targets",
            "relationship_add_to_group", "relationship_copy_tree",
            "relationship_copy_tree_between",
            "relationship_remove_from_group", "relationship_set_parent",
            "relationship_transfer_tree_between",
        ])
        result = self.server.request(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "document_create", "arguments": self.new_args()}}
        )
        created = result["result"]["structuredContent"]
        self.assertTrue(created["ok"])
        exported = self.server.request(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "document_export", "arguments": {
                 "workspace_id": self.workspace_id,
                 "document": created["document"],
                 "expected_revision": 0,
                 "suggested_filename": "legacy-export.cb",
             }}}
        )["result"]["structuredContent"]
        self.assertTrue(exported["ok"], exported)
        imported = self.server.request(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "document_import", "arguments": {
                 "workspace_id": self.workspace_id,
                 "request_id": str(uuid.uuid4()),
                 "units": "mm",
                 "source_name": "legacy-export.cb",
                 "content": exported["data"]["content"],
             }}}
        )["result"]["structuredContent"]
        self.assertTrue(imported["ok"], imported)

    def test_protocol_eras_cannot_be_mixed(self):
        self.server.request(
            {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"_meta": self.meta()}}
        )
        unsupported = self.server.request(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list",
             "params": {"_meta": self.meta("2025-03-26")}}
        )
        self.assertEqual(unsupported["error"]["code"], -32022)
        initialize = self.server.request(
            {"jsonrpc": "2.0", "id": 3, "method": "initialize",
             "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                         "clientInfo": {"name": "legacy", "version": "0"}}}
        )
        self.assertEqual(initialize["error"]["code"], -32022)

        legacy = MCPProcess(self.workspace)
        try:
            accepted = legacy.request(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                             "clientInfo": {"name": "legacy", "version": "0"}}}
            )
            self.assertEqual(accepted["result"]["protocolVersion"], "2025-06-18")
            modern = legacy.request(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": self.meta()}}
            )
            self.assertEqual(modern["error"]["code"], -32600)
        finally:
            legacy.close()

    def test_strict_json_rejects_duplicate_keys_and_nan(self):
        self.server.request(
            {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"_meta": self.meta()}}
        )
        duplicate = self.server.raw(
            b'{"jsonrpc":"2.0","jsonrpc":"2.0","id":2,"method":"ping","params":{}}'
        )
        self.assertEqual(duplicate["error"]["code"], -32700)
        nan = self.server.raw(
            b'{"jsonrpc":"2.0","id":2,"method":"ping","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{},"x":NaN}}}'
        )
        self.assertEqual(nan["error"]["code"], -32700)
        overflow = self.server.raw(
            b'{"jsonrpc":"2.0","id":4,"method":"ping","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{},"x":1e999}}}'
        )
        self.assertEqual(overflow["error"]["code"], -32700)
        malformed = self.server.raw(b'{"unterminated":')
        self.assertEqual(malformed["error"]["code"], -32700)
        recovered = self.server.request(
            {"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {"_meta": self.meta()}}
        )
        self.assertIn("result", recovered)

    def test_content_round_trip_above_previous_one_mib_framing_limit(self):
        self.assertGreater(MAX_INPUT_LINE_BYTES, 10 * 1024 * 1024)
        with patch("cambam_builder.mcp_adapter.server.MAX_INPUT_LINE_BYTES", 8):
            with self.assertRaises(StrictJSONError):
                _strict_decode(b'{"value":1}')

        prefix = '<?xml version="1.0" encoding="utf-8"?><CADFile><!--'
        suffix = '--><layers /></CADFile>'
        content = prefix + ("x" * (10 * 1024 * 1024 - len(prefix) - len(suffix))) + suffix
        imported = self.call(
            10,
            "document_import",
            {
                "workspace_id": self.workspace_id,
                "request_id": str(uuid.uuid4()),
                "units": "mm",
                "source_name": "large-client-input.cb",
                "content": content,
            },
        )["result"]["structuredContent"]
        self.assertTrue(imported["ok"], imported)
        exported_wire = self.call(
            11,
            "document_export",
            {
                "workspace_id": self.workspace_id,
                "document": imported["document"],
                "expected_revision": 0,
                "suggested_filename": "large-client-output.cb",
            },
        )
        exported = exported_wire["result"]["structuredContent"]
        self.assertTrue(exported["ok"], exported)
        self.assertEqual(
            json.loads(exported_wire["result"]["content"][0]["text"]), exported
        )
        artifact = exported["data"]
        encoded = artifact["content"].encode("utf-8")
        self.assertEqual(artifact["bytes"], len(encoded))
        self.assertEqual(artifact["sha256"], hashlib.sha256(encoded).hexdigest())

    def test_unknown_method_and_tool_are_protocol_errors(self):
        method = self.server.request(
            {"jsonrpc": "2.0", "id": 1, "method": "unknown/method", "params": {"_meta": self.meta()}}
        )
        self.assertEqual(method["error"]["code"], -32601)
        tool = self.server.request(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "unknown_tool", "arguments": {}, "_meta": self.meta()}}
        )
        self.assertEqual(tool["error"]["code"], -32602)

    def test_restart_expires_old_handles(self):
        created = self.call(1, "document_create", self.new_args())
        old_handle = created["result"]["structuredContent"]["document"]
        replacement = MCPProcess(self.workspace)
        try:
            expired = replacement.request(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "document_inspect",
                             "arguments": {"workspace_id": self.workspace_id, "document": old_handle},
                             "_meta": self.meta()}}
            )
            self.assertTrue(expired["result"]["structuredContent"]["error"])
            self.assertEqual(expired["result"]["structuredContent"]["error"]["code"], "DOCUMENT_EXPIRED")
        finally:
            replacement.close()

    def test_eof_shuts_down_server(self):
        self.server.process.stdin.close()
        self.server.process.wait(timeout=5)
        self.assertEqual(self.server.process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
