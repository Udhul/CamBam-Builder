"""Document/state/security acceptance independent of model behavior."""
import copy
import hashlib
from importlib.util import find_spec
import json
import os
import subprocess
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

if find_spec("mcp") is not None:
    import anyio
    from cambam_builder.mcp_adapter.service import DocumentService
    from cambam_builder.mcp_adapter.paths import DomainError, Workspace
    from cambam_builder.mcp_adapter.schema import CONTRACT, OUTPUTS, TOOLS, schema
else:
    DocumentService = None

from cambam_builder import CBProject


@unittest.skipIf(DocumentService is None, "Install .[mcp] for adapter checks")
class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_async(self, test):
        async def run():
            self.service = DocumentService(self.root)
            await test()
        anyio.run(run)

    def args(self, **kwargs):
        return {"workspace_id": self.service.workspace.id, "request_id": str(uuid4()), **kwargs}

    async def call(self, tool, args):
        result = await self.service.call_tool(tool, args)
        OUTPUTS[tool].validate(result)
        return result

    async def create(self):
        result = await self.call("document_create", self.args(name="test", units="mm"))
        self.assertTrue(result["ok"], result)
        return result["document"]

    def test_create_inspect_save_open_close_replay(self):
        async def test():
            handle = await self.create()
            args = self.args(document=handle, expected_revision=0, path="A.cb")
            saved = await self.call("document_save", args)
            self.assertTrue(saved["ok"], saved)
            content = (self.root / "A.cb").read_bytes()
            self.assertEqual(saved["data"]["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(saved["revision"], 0)
            replay = await self.call("document_save", args)
            self.assertEqual({**saved, "replayed": True}, replay)
            exists = await self.call("document_save", {**args, "request_id": str(uuid4())})
            self.assertEqual(exists["error"]["code"], "PATH_EXISTS")
            opened = await self.call("document_open", self.args(path="A.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            self.assertNotEqual(opened["document"], handle)
            read = {"workspace_id": self.service.workspace.id, "document": opened["document"]}
            inspected = await self.call("document_inspect", read)
            self.assertEqual(inspected["data"]["summary"]["source"]["sha256"], saved["data"]["sha256"])
            close = self.args(document=handle, expected_revision=0)
            closed = await self.call("document_close", close)
            self.assertEqual(await self.call("document_close", close), {**closed, "replayed": True})
            self.assertEqual((await self.call("document_save", args)), replay)
            self.assertEqual((self.root / "A.cb").read_bytes(), content)
        self.run_async(test)

    def test_identity_schema_and_cached_failures(self):
        async def test():
            handle = await self.create()
            base = self.args(document=handle, expected_revision=9)
            failed = await self.call("document_close", base)
            self.assertEqual(failed["error"]["code"], "STALE_REVISION")
            self.assertEqual(failed["error"]["current_revision"], 0)
            self.assertEqual(await self.call("document_close", base), {**failed, "replayed": True})
            changed = await self.call("document_close", {**base, "expected_revision": 0})
            self.assertEqual(changed["error"]["code"], "REQUEST_ID_CONFLICT")
            for change, code in [({"workspace_id": "0" * 64}, "WORKSPACE_MISMATCH"),
                                 ({"document": str(uuid4()) + ":" + str(uuid4())}, "DOCUMENT_EXPIRED"),
                                 ({"document": self.service.bootstrap["boot_id"] + ":" + str(uuid4())}, "DOCUMENT_NOT_FOUND"),
                                 ({"expected_revision": True}, "INVALID_ARGUMENT"),
                                 ({"expected_revision": 0.0}, "INVALID_ARGUMENT"),
                                 ({"extra": 1}, "INVALID_ARGUMENT")]:
                response = await self.call("document_close", {**base, "request_id": str(uuid4()), **change})
                self.assertEqual(response["error"]["code"], code, response)
            bad = await self.call("document_close", {"document": "bad", "request_id": "bad"})
            self.assertIsNone(bad["document"])
            self.assertIsNone(bad["request_id"])
            self.assertEqual(bad["workspace_id"], self.service.workspace.id)
        self.run_async(test)

    def test_ledger_keys_include_workspace_identity(self):
        async def test():
            args = self.args(name="workspace-key", units="mm")
            wrong_args = {**args, "workspace_id": "0" * 64}
            wrong = await self.call("document_create", wrong_args)
            self.assertEqual(wrong["error"]["code"], "WORKSPACE_MISMATCH")
            correct = await self.call("document_create", args)
            self.assertTrue(correct["ok"], correct)
            self.assertFalse(correct["replayed"])
            self.assertEqual(await self.call("document_create", wrong_args), {**wrong, "replayed": True})
            self.assertEqual(await self.call("document_create", args), {**correct, "replayed": True})
        self.run_async(test)

    def test_capacity_concurrent_open_and_ledger_drain(self):
        async def test():
            for _ in range(15):
                await self.create()
            CBProject("source").save(str(self.root / "source.cb"))
            results = []
            async def opening():
                results.append(await self.call("document_open", self.args(path="source.cb", units="mm")))
            async with anyio.create_task_group() as group:
                group.start_soon(opening)
                group.start_soon(opening)
            self.assertEqual(sum(result["ok"] for result in results), 1)
            self.assertEqual(self.service.reservations, 0)
            self.assertEqual(len(self.service.documents), 16)
            self.service.MAX_REGULAR_REQUESTS = self.service.regular_requests
            full = await self.call("document_create", self.args(name="full", units="mm"))
            self.assertEqual(full["error"]["code"], "LIMIT_EXCEEDED")
            handle = next(iter(self.service.documents))
            save = await self.call("document_save", self.args(document=handle, expected_revision=0, path="drain.cb"))
            self.assertTrue(save["ok"], save)
            closed = await self.call("document_close", self.args(document=handle, expected_revision=0))
            self.assertTrue(closed["ok"])
        self.run_async(test)

    def test_inflight_retry_conflict_and_close_serialization(self):
        async def test():
            args = self.args(name="joined", units="mm")
            original = self.service._stage_new
            entered, release = threading.Event(), threading.Event()
            def blocked(*arguments):
                entered.set()
                release.wait(5)
                return original(*arguments)
            results = []
            async def create():
                results.append(await self.call("document_create", args))
            with patch.object(self.service, "_stage_new", blocked):
                async with anyio.create_task_group() as group:
                    group.start_soon(create)
                    await anyio.to_thread.run_sync(entered.wait, 5)
                    group.start_soon(create)
                    conflict = await self.call("document_create", {**args, "name": "different"})
                    self.assertEqual(conflict["error"]["code"], "REQUEST_ID_CONFLICT")
                    release.set()
            self.assertEqual(len(self.service.documents), 1)
            self.assertEqual(sorted(r["replayed"] for r in results), [False, True])
            handle = results[0]["document"]
            results.clear()
            async def close():
                results.append(await self.call("document_close", self.args(document=handle, expected_revision=0)))
            async with anyio.create_task_group() as group:
                group.start_soon(close)
                group.start_soon(close)
            self.assertEqual(sum(r["ok"] for r in results), 1)
        self.run_async(test)

    def test_cancel_staging_and_retry_terminal_failure(self):
        async def test():
            args = self.args(name="cancel", units="mm")
            original = self.service._stage_new
            entered, release = threading.Event(), threading.Event()
            def blocked(*arguments):
                entered.set()
                release.wait(5)
                return original(*arguments)
            async def create():
                await self.call("document_create", args)
            with patch.object(self.service, "_stage_new", blocked):
                async with anyio.create_task_group() as group:
                    group.start_soon(create)
                    await anyio.to_thread.run_sync(entered.wait, 5)
                    group.cancel_scope.cancel()
                    release.set()
            self.assertFalse(self.service.documents)
            self.assertEqual(self.service.reservations, 0)
            replay = await self.call("document_create", args)
            self.assertEqual(replay["error"]["code"], "REQUEST_CANCELLED")
            self.assertTrue(replay["replayed"])
        self.run_async(test)

    def test_import_export_failures_and_destination_race(self):
        async def test():
            (self.root / "bad.cb").write_bytes(b'<!DOCTYPE x [<!ENTITY a "bad">]><x/>')
            bad = await self.call("document_open", self.args(path="bad.cb", units="mm"))
            self.assertEqual(bad["error"]["code"], "IMPORT_FAILED")
            self.assertFalse(self.service.documents)
            handle = await self.create()
            args = self.args(document=handle, expected_revision=0, path="race.cb")
            original = self.service.workspace.publish
            def race(temporary, relative):
                (self.root / relative).write_bytes(b"racing owner")
                return original(temporary, relative)
            with patch.object(self.service.workspace, "publish", race):
                result = await self.call("document_save", args)
            self.assertEqual(result["error"]["code"], "PATH_EXISTS")
            self.assertEqual((self.root / "race.cb").read_bytes(), b"racing owner")
            with patch.object(CBProject, "save", side_effect=ValueError("private detail")):
                result = await self.call("document_save", {**args, "request_id": str(uuid4()), "path": "fail.cb"})
            self.assertEqual(result["error"]["code"], "EXPORT_FAILED")
            self.assertFalse((self.root / "fail.cb").exists())
            self.assertFalse(list(self.root.glob(".cambam-mcp-*")))
        self.run_async(test)

    def test_schema_asset_and_strict_values(self):
        self.assertEqual(CONTRACT, json.loads((Path(__file__).parents[1] / "docs/mcp_contract_v1.schema.json").read_text()))
        async def test():
            for name in TOOLS:
                for value in [None, [], {}, {"request_id": str(uuid4()), "workspace_id": "bad"}]:
                    result = await self.call(name, value)
                    self.assertFalse(result["ok"])
            for name in ["a\x00b", "a\nb", "a\x7fb"]:
                result = await self.call("document_create", self.args(name=name, units="mm"))
                self.assertFalse(result["ok"])
        self.run_async(test)

    def test_path_policy_and_hardlinks(self):
        workspace = Workspace(self.root)
        for path in ["../x.cb", "/x.cb", "C:x.cb", "a\\x.cb", "a//x.cb", "./x.cb", "x.cb:ads", "NUL.cb",
                     "COM1.cb", "LPT¹.cb", "a./x.cb", "a /x.cb", "x.txt", "a/../x.cb", "x?.cb"]:
            with self.subTest(path=path), self.assertRaises(DomainError):
                workspace.path(path, destination=True)
        (self.root / "x.cb").write_bytes(b"x")
        os.link(self.root / "x.cb", self.root / "link.cb")
        with self.assertRaises(DomainError):
            workspace.read("link.cb")
        equivalent = Workspace(str(self.root) + os.sep)
        self.assertEqual(workspace.id, equivalent.id)
        if os.name == "nt":
            self.assertEqual(workspace.id, Workspace(str(self.root).upper().replace("\\", "/")).id)

    @unittest.skipUnless(os.name == "nt", "Windows junction containment")
    def test_junction_rejection_and_root_alias_equivalence(self):
        real = self.root / "real"
        real.mkdir()
        (real / "input.cb").write_bytes(b"private")
        alias = self.root / "alias"
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(real)],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Workspace(alias).id, Workspace(real).id)
        workspace = Workspace(self.root)
        with self.assertRaises(DomainError):
            workspace.read("alias/input.cb")
        with self.assertRaises(DomainError):
            workspace.path("alias/new.cb", destination=True)

    def test_symlink_rejection(self):
        target = self.root / "ordinary.cb"
        target.write_bytes(b"private")
        alias = self.root / "alias.cb"
        try:
            alias.symlink_to(target)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege unavailable; junction/reparse tests still run")
            raise
        with self.assertRaises(DomainError):
            Workspace(self.root).read("alias.cb")

    def test_two_creates_at_fifteen_documents(self):
        async def test():
            for _ in range(15):
                await self.create()
            results = []
            async def create():
                results.append(await self.call("document_create", self.args(name="capacity", units="mm")))
            async with anyio.create_task_group() as group:
                group.start_soon(create)
                group.start_soon(create)
            self.assertEqual(sum(r["ok"] for r in results), 1)
            self.assertEqual(self.service.reservations, 0)
        self.run_async(test)

    def test_save_lock_snapshot_identity_and_pagination(self):
        async def test():
            project = CBProject("inventory")
            layer = project.add_layer("Geometry")
            rect = project.add_rect(layer, identifier="outline", width=20, height=10)
            part = project.add_part("Part")
            mop = project.add_profile_mop(part, targets=[rect], identifier="profile")
            project.save(str(self.root / "input.cb"))
            opened = await self.call("document_open", self.args(path="input.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            handle = opened["document"]
            read = {"workspace_id": self.service.workspace.id, "document": handle, "limit": 1}
            records = []
            for offset in range(4):
                page = await self.call("document_inspect", {**read, "offset": offset})
                self.assertTrue(page["ok"], page)
                records.extend(page["data"]["entities"])
            self.assertEqual([r["kind"] for r in records], ["layer", "part", "primitive", "mop"])
            self.assertEqual(records[-2]["id"], str(rect.internal_id))
            self.assertEqual(records[-1]["id"], str(mop.internal_id))
            self.assertEqual(records[-1]["targets"], [str(rect.internal_id)])
            entered, release = threading.Event(), threading.Event()
            original = self.service.workspace.stage_save
            def blocked(*args):
                entered.set()
                release.wait(5)
                return original(*args)
            results = []
            save_args = self.args(document=handle, expected_revision=0, path="snapshot.cb")
            async def saving():
                results.append(await self.call("document_save", save_args))
            async def staged_edit():
                # Exercise the documented future-edit lock contract without adding
                # a premature public geometry tool to 4b.
                document = self.service.documents[handle]
                async with document.lock:
                    cloned = document.project.clone()
                    cloned.translate_primitive(rect.internal_id, 5, 2, bake=False)
                    document.project = cloned
                    document.revision += 1
            with patch.object(self.service.workspace, "stage_save", blocked):
                async with anyio.create_task_group() as group:
                    group.start_soon(saving)
                    await anyio.to_thread.run_sync(entered.wait, 5)
                    group.start_soon(staged_edit)
                    release.set()
            self.assertTrue(results[0]["ok"], results)
            self.assertEqual(results[0]["revision"], 0)
            self.assertEqual(self.service.documents[handle].revision, 1)
            stale = await self.call("document_save", {**save_args, "request_id": str(uuid4()), "path": "stale.cb"})
            self.assertEqual(stale["error"]["code"], "STALE_REVISION")
            reopened = await self.call("document_open", self.args(path="snapshot.cb", units="mm"))
            self.assertTrue(reopened["ok"], reopened)
            saved_project = self.service.documents[reopened["document"]].project
            self.assertEqual(str(saved_project.list_primitives()[0].internal_id), str(rect.internal_id))
            self.assertEqual(str(saved_project.list_mops()[0].internal_id), str(mop.internal_id))
            self.assertEqual(saved_project.get_mop_targets(mop.internal_id), [rect.internal_id])
        self.run_async(test)

    def test_cancel_save_before_and_after_publication(self):
        async def test():
            handle = await self.create()
            args = self.args(document=handle, expected_revision=0, path="cancel.cb")
            original = self.service.workspace.stage_save
            entered, release = threading.Event(), threading.Event()
            def blocked(*arguments):
                staged = original(*arguments)
                entered.set()
                release.wait(5)
                return staged
            async def saving():
                await self.call("document_save", args)
            with patch.object(self.service.workspace, "stage_save", blocked):
                async with anyio.create_task_group() as group:
                    group.start_soon(saving)
                    await anyio.to_thread.run_sync(entered.wait, 5)
                    group.cancel_scope.cancel()
                    release.set()
            self.assertFalse((self.root / "cancel.cb").exists())
            self.assertFalse(list(self.root.glob(".cambam-mcp-*")))
            retry = await self.call("document_save", args)
            self.assertEqual(retry["error"]["code"], "REQUEST_CANCELLED")
            args = {**args, "request_id": str(uuid4())}
            publish = self.service.workspace.publish
            with anyio.CancelScope() as scope:
                def cancel_after_publish(*arguments):
                    publish(*arguments)
                    scope.cancel()
                with patch.object(self.service.workspace, "publish", cancel_after_publish):
                    await saving()
                    await anyio.lowlevel.checkpoint()
            retry = await self.call("document_save", args)
            self.assertTrue(retry["ok"], retry)
            self.assertTrue(retry["replayed"])
            self.assertTrue((self.root / "cancel.cb").exists())
        self.run_async(test)

    def test_cleanup_failure_is_success_and_size_limit_is_atomic(self):
        async def test():
            handle = await self.create()
            args = self.args(document=handle, expected_revision=0, path="cleanup.cb")
            with patch.object(self.service.workspace, "cleanup", return_value=False):
                result = await self.call("document_save", args)
            self.assertTrue(result["ok"], result)
            self.assertIn("CLEANUP_PENDING", [d["code"] for d in result["diagnostics"]])
            self.assertEqual(await self.call("document_save", args), {**result, "replayed": True})
            (self.root / "huge.cb").write_bytes(b" " * (10 * 1024 * 1024 + 1))
            result = await self.call("document_open", self.args(path="huge.cb", units="mm"))
            self.assertEqual(result["error"]["code"], "LIMIT_EXCEEDED")
            def huge_save(project, path):
                Path(path).write_bytes(b" " * (10 * 1024 * 1024 + 1))
            with patch.object(CBProject, "save", huge_save):
                result = await self.call("document_save", {**args, "request_id": str(uuid4()), "path": "too-big.cb"})
            self.assertEqual(result["error"]["code"], "LIMIT_EXCEEDED")
            self.assertFalse((self.root / "too-big.cb").exists())
            (self.root / "too-many.cb").write_bytes(
                b'<CADFile><layers><layer name="Geometry"><objects>' + b'<rect/>' * 10001
                + b'</objects></layer></layers></CADFile>')
            result = await self.call("document_open", self.args(path="too-many.cb", units="mm"))
            self.assertEqual(result["error"]["code"], "LIMIT_EXCEEDED")
            self.assertEqual(len(self.service.documents), 1)
        self.run_async(test)


if __name__ == "__main__":
    unittest.main()
