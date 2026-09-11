"""4d batch 5 MCP cross-document copy/transfer acceptance.

The two-document tools are exercised through imported document handles
(plus a created handle in limit/concurrency cases), compared with
independently authored public-framework projects, and probed for per-document
revision semantics, failure addressing, staged-target limits, ledger replay,
cancellation and both same- and opposing-direction serialization.
"""

import anyio
from importlib.util import find_spec
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from uuid import uuid4

if find_spec("mcp") is not None:
    from cambam_builder.mcp_adapter.service import DocumentService
    from cambam_builder.mcp_adapter.schema import OUTPUTS, TOOLS
else:  # pragma: no cover - exercised only in a base-library environment
    DocumentService = None
    OUTPUTS = {}
    TOOLS = ()

from cambam_builder import CBProject


UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?(?![A-Za-z])"
)


@unittest.skipIf(DocumentService is None, "Install .[mcp] for adapter checks")
class CrossDocumentTests(unittest.TestCase):
    """DocumentService contract tests for the cross-document family."""

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
        result = {
            "workspace_id": self.service.workspace.id,
            "request_id": str(uuid4()),
        }
        result.update(kwargs)
        return result

    async def call(self, name, arguments):
        result = await self.service.call_tool(name, arguments)
        OUTPUTS[name].validate(result)
        return result

    async def import_document(self, project):
        destination = self.root / f"{project.project_name}-{uuid4().hex[:8]}.cb"
        project.save(str(destination))
        content = destination.read_bytes().decode("utf-8")
        destination.unlink()
        result = await self.call("document_import", self.args(
            source_name=f"{project.project_name}.cb", units="mm", content=content))
        self.assertTrue(result["ok"], result)
        return result["document"]

    async def create(self, name="moved"):
        result = await self.call("document_create", self.args(name=name, units="mm"))
        self.assertTrue(result["ok"], result)
        return result["document"]

    async def inspect(self, handle, *, revision=None):
        arguments = {"workspace_id": self.service.workspace.id, "document": handle}
        if revision is not None:
            arguments["expected_revision"] = revision
        result = await self.call("document_inspect", arguments)
        self.assertTrue(result["ok"], result)
        return result

    async def inspect_records(self, handle, revision=None):
        result = await self.inspect(handle, revision=revision)
        data = result["data"]
        records = list(data["entities"])
        while data["next_offset"] is not None:
            result = await self.call("document_inspect", {
                "workspace_id": self.service.workspace.id,
                "document": handle,
                "offset": data["next_offset"],
            })
            self.assertTrue(result["ok"], result)
            data = result["data"]
            records.extend(data["entities"])
        return records

    async def save(self, handle, revision, path):
        result = await self.call("document_save", self.args(
            document=handle, expected_revision=revision, path=path))
        self.assertTrue(result["ok"], result)
        return result

    @staticmethod
    def source_project():
        project = CBProject("source-doc")
        layer = project.add_layer("Geometry")
        rectangle = project.add_rect(layer, corner=(0, 0), width=20, height=10,
                                     identifier="outline")
        circle = project.add_circle(layer, center=(5, 5), diameter=4,
                                    identifier="circle")
        project.add_primitive_to_group(circle, "targets")
        project.link_primitive_parent(circle, rectangle)
        island = project.add_circle(layer, center=(40, 0), diameter=2,
                                    identifier="island")
        part = project.add_part("P", enabled=True, stock_width=0, stock_height=0,
                                stock_thickness=0, stock_material="")
        project.add_profile_mop(
            part, targets=[rectangle], identifier="prof", name="prof",
            profile_side="Inside", target_depth=-1, depth_increment=0.5,
            tool_diameter=3, cut_feedrate=300, plunge_feedrate=100,
            spindle_speed=12000, stock_surface=0, clearance_plane=5,
            lead_in_type="None")
        return project, rectangle, island

    @staticmethod
    def target_project():
        project = CBProject("target-doc")
        layer = project.add_layer("Board")
        project.add_rect(layer, corner=(100, 100), width=30, height=20,
                         identifier="anchor")
        return project

    @staticmethod
    def colliding_project():
        project = CBProject("colliding")
        layer = project.add_layer("Geometry")
        project.add_rect(layer, corner=(0, 0), width=5, height=5,
                         identifier="keep")
        return project

    def cross_arguments(self, source, source_revision, target, target_revision,
                        root, **extra):
        return self.args(
            source_document=source,
            source_expected_revision=source_revision,
            target_document=target,
            target_expected_revision=target_revision,
            root=str(root),
            **extra,
        )

    @staticmethod
    def diagnostics(result):
        return {item["code"] for item in result["diagnostics"]}

    @staticmethod
    def xml_semantics(path):
        """Return parsed XML semantics while ignoring run-specific UUIDs."""
        root = ET.parse(path).getroot()
        id_to_user = {}
        for element in root.iter():
            raw_id = element.attrib.get("id")
            if raw_id is None:
                continue
            try:
                tag = json.loads(element.findtext("Tag") or "null")
            except ValueError:
                tag = None
            user_id = tag.get("user_id") if isinstance(tag, dict) else None
            if user_id:
                id_to_user[raw_id] = user_id
        for element in root.iter():
            raw_id = element.attrib.get("id")
            if raw_id in id_to_user:
                element.set("id", id_to_user[raw_id])
            if (element.tag.rsplit("}", 1)[-1] == "prim"
                    and (element.text or "") in id_to_user):
                element.text = id_to_user[element.text]
        for container in root.iter("primitive"):
            children = list(container)
            for child in children:
                container.remove(child)
            for child in sorted(children, key=lambda item: item.text or ""):
                container.append(child)
        for objects in root.iter("objects"):
            children = list(objects)

            def sort_key(element):
                try:
                    tag = json.loads(element.findtext("Tag") or "null")
                except ValueError:
                    tag = None
                user_id = tag.get("user_id") if isinstance(tag, dict) else None
                return (element.tag.rsplit("}", 1)[-1], user_id or "")

            for child in children:
                objects.remove(child)
            for child in sorted(children, key=sort_key):
                objects.append(child)

        def normalized_numbers(value):
            def replace(match):
                try:
                    number = Decimal(match.group(0))
                except InvalidOperation:
                    return match.group(0)
                if number == 0:
                    return "0"
                return format(number.normalize(), "f")

            return NUMBER_PATTERN.sub(replace, value)

        def normalized_text(element):
            value = (element.text or "").strip()
            if element.tag.rsplit("}", 1)[-1] == "Tag":
                try:
                    tag = json.loads(value)
                except (TypeError, ValueError):
                    return UUID_PATTERN.sub("<UUID>", value)
                if isinstance(tag, dict):
                    tag = dict(tag)
                    tag.pop("internal_id", None)
                    value = json.dumps(tag, sort_keys=True, separators=(",", ":"))
            return normalized_numbers(UUID_PATTERN.sub("<UUID>", value))

        def node(element):
            attributes = tuple(sorted(
                (key, normalized_numbers(UUID_PATTERN.sub("<UUID>", value)))
                for key, value in element.attrib.items()
            ))
            return (
                element.tag.rsplit("}", 1)[-1],
                attributes,
                normalized_text(element),
                tuple(node(child) for child in element),
            )

        return node(root)

    def project_semantics(self, project, name):
        """Serialize a project clone for a stable before/after comparison."""
        path = self.root / name
        project.clone().save(str(path))
        return self.xml_semantics(path)

    def test_tools_and_copy_parity_between_two_imported_documents(self):
        async def test():
            self.assertIn("relationship_copy_tree_between", TOOLS)
            self.assertIn("relationship_transfer_tree_between", TOOLS)
            source_direct, rectangle, island = self.source_project()
            target_direct = self.target_project()
            source = await self.import_document(source_direct)
            target = await self.import_document(target_direct)
            self.assertNotEqual(source, target)

            copy_arguments = self.cross_arguments(
                source, 0, target, 0, rectangle.internal_id, include_mops=True,
                identifier_map={"outline": "outline-copy", "circle": "circle-copy",
                                "Geometry": "Geometry-copy", "P": "P-copy",
                                "prof": "prof-copy"},
                group_map={"targets": "targets-copy"})
            copied = await self.call("relationship_copy_tree_between", copy_arguments)
            self.assertTrue(copied["ok"], copied)
            mapping = copied["data"]["mapping"]
            self.assertEqual(copied["data"], {
                "mapping": mapping,
                "source_document": source,
                "target_document": target,
                "source_revision": 0,
                "target_revision": 1,
            })
            self.assertEqual(len(mapping), 5)
            new_rect = mapping[str(rectangle.internal_id)]
            self.assertEqual(copied["revision"], 0)
            self.assertEqual(copied["document"], source)

            replay = await self.call("relationship_copy_tree_between", copy_arguments)
            # An identical retry replays the original completed result before
            # any revision check, so the target is not copied into twice.
            self.assertEqual(replay, {**copied, "replayed": True})
            self.assertEqual((await self.inspect(target))["data"]["summary"]["counts"],
                             {"layers": 2, "parts": 1, "primitives": 3, "mops": 1})

            stale_retry = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 0, target, 0, rectangle.internal_id, include_mops=True,
                identifier_map={"outline": "outline-copy", "circle": "circle-copy",
                                "Geometry": "Geometry-copy", "P": "P-copy",
                                "prof": "prof-copy"},
                group_map={"targets": "targets-copy"}))
            self.assertEqual(stale_retry["error"]["code"], "STALE_REVISION")
            self.assertEqual(stale_retry["error"]["field"], "target_expected_revision")
            self.assertEqual(stale_retry["document"], target)

            source_records = await self.inspect_records(source, revision=0)
            target_records = await self.inspect_records(target, revision=1)
            self.assertEqual(
                (await self.inspect(target))["data"]["summary"]["counts"],
                {"layers": 2, "parts": 1, "primitives": 3, "mops": 1})
            # The copied root still has its copied child, so both stay
            # diagnostic while the copy itself is complete.
            self.assertIn("INSPECTION_UNSUPPORTED", self.diagnostics(await self.inspect(target)))
            by_identifier = {record["identifier"]: record for record in target_records
                             if record["kind"] in ("primitive", "mop")}
            self.assertEqual(by_identifier["outline-copy"]["id"], new_rect)
            self.assertIsNone(by_identifier["outline-copy"]["parent"])
            self.assertIsNone(by_identifier["outline-copy"]["geometry"])
            self.assertEqual(by_identifier["circle-copy"]["parent"], new_rect)
            self.assertEqual(by_identifier["circle-copy"]["groups"], ["targets-copy"])
            self.assertIsNone(by_identifier["circle-copy"]["geometry"])
            self.assertEqual(by_identifier["prof-copy"]["part"], "P-copy")
            self.assertEqual(by_identifier["prof-copy"]["targets"], [new_rect])

            # A second cross-document copy of the childless island returns to
            # the typed similarity slice in the target.
            island_copy = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 0, target, 1, island.internal_id,
                identifier_map={"island": "island-copy",
                                "Geometry": "Geometry-copy-2"}))
            self.assertTrue(island_copy["ok"], island_copy)
            self.assertEqual(island_copy["data"]["source_revision"], 0)
            self.assertEqual(island_copy["data"]["target_revision"], 2)
            new_island = island_copy["data"]["mapping"][str(island.internal_id)]
            island_records = await self.inspect_records(target, revision=2)
            self.assertEqual(
                (await self.inspect(target))["data"]["summary"]["counts"],
                {"layers": 3, "parts": 1, "primitives": 4, "mops": 1})
            by_identifier = {record["identifier"]: record for record in island_records
                             if record["kind"] == "primitive"}
            self.assertEqual(by_identifier["island-copy"]["id"], new_island)
            self.assertIsNone(by_identifier["island-copy"]["parent"])
            self.assertEqual(by_identifier["island-copy"]["groups"], [])
            self.assertEqual(by_identifier["island-copy"]["geometry"]["kind"], "circle")
            self.assertEqual(
                [round(value, 9) for value in by_identifier["island-copy"]["geometry"]["center"]],
                [40.0, 0.0, 0.0])

            await self.save(source, 0, "S1.cb")
            await self.save(target, 2, "T1.cb")
            source_direct.copy_primitive_tree(
                rectangle, target_direct, preserve_ids=False,
                identifier_map={"outline": "outline-copy", "circle": "circle-copy",
                                "Geometry": "Geometry-copy", "P": "P-copy",
                                "prof": "prof-copy"},
                group_map={"targets": "targets-copy"}, include_mops=True)
            source_direct.copy_primitive_tree(
                island, target_direct, preserve_ids=False,
                identifier_map={"island": "island-copy",
                                "Geometry": "Geometry-copy-2"})
            source_direct.save(str(self.root / "direct-S1.cb"))
            target_direct.save(str(self.root / "direct-T1.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "S1.cb"),
                self.xml_semantics(self.root / "direct-S1.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "T1.cb"),
                self.xml_semantics(self.root / "direct-T1.cb"))

        self.run_async(test)

    def test_transfer_moves_subtree_and_advances_both_revisions(self):
        async def test():
            source_direct, rectangle, island = self.source_project()
            moved_direct = self.target_project()
            source = await self.import_document(source_direct)
            moved = await self.import_document(moved_direct)

            transferred = await self.call("relationship_transfer_tree_between", self.cross_arguments(
                source, 0, moved, 0, rectangle.internal_id, include_mops=True))
            self.assertTrue(transferred["ok"], transferred)
            mapping = transferred["data"]["mapping"]
            self.assertEqual(transferred["data"], {
                "mapping": mapping,
                "source_document": source,
                "target_document": moved,
                "source_revision": 1,
                "target_revision": 1,
            })
            self.assertEqual(len(mapping), 5)
            new_rect = mapping[str(rectangle.internal_id)]

            source_counts = (await self.inspect(source))["data"]["summary"]["counts"]
            moved_counts = (await self.inspect(moved))["data"]["summary"]["counts"]
            # Removal empties the moved subtree but keeps the source layer,
            # part and the untouched island primitive.  The imported target's
            # existing Board/anchor content must remain alongside the transfer.
            self.assertEqual(source_counts,
                             {"layers": 1, "parts": 1, "primitives": 1, "mops": 0})
            self.assertEqual(moved_counts,
                             {"layers": 2, "parts": 1, "primitives": 3, "mops": 1})
            moved_records = await self.inspect_records(moved, revision=1)
            by_identifier = {record["identifier"]: record for record in moved_records
                             if record["kind"] in ("primitive", "mop")}
            self.assertEqual(by_identifier["outline"]["id"], new_rect)
            self.assertEqual(by_identifier["circle"]["parent"], new_rect)
            self.assertEqual(by_identifier["circle"]["groups"], ["targets"])
            self.assertEqual(by_identifier["prof"]["part"], "P")
            self.assertEqual(by_identifier["prof"]["targets"], [new_rect])
            self.assertEqual(by_identifier["anchor"]["parent"], None)

            await self.save(source, 1, "S2.cb")
            await self.save(moved, 1, "M2.cb")
            source_direct.transfer_primitive_tree(
                rectangle, moved_direct, preserve_ids=False, include_mops=True)
            source_direct.save(str(self.root / "direct-S2.cb"))
            moved_direct.save(str(self.root / "direct-M2.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "S2.cb"),
                self.xml_semantics(self.root / "direct-S2.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "M2.cb"),
                self.xml_semantics(self.root / "direct-M2.cb"))

        self.run_async(test)

    def test_transfer_cancellation_is_atomic_and_replayable(self):
        async def test():
            source_direct, rectangle, island = self.source_project()
            target_direct = self.target_project()
            source = await self.import_document(source_direct)
            target = await self.import_document(target_direct)
            arguments = self.cross_arguments(
                source, 0, target, 0, rectangle.internal_id, include_mops=True)

            source_document = self.service.documents[source]
            target_document = self.service.documents[target]
            source_project = source_document.project
            target_project = target_document.project
            source_before = self.project_semantics(source_project, "cancel-before-source.cb")
            target_before = self.project_semantics(target_project, "cancel-before-target.cb")

            original_stage = self.service._stage_cross_edit
            entered, release = threading.Event(), threading.Event()

            def blocked_stage(*stage_arguments):
                staged = original_stage(*stage_arguments)
                entered.set()
                release.wait(5)
                return staged

            async def transfer():
                await self.call("relationship_transfer_tree_between", arguments)

            # Cancel after both clones have been staged, but before the
            # publication checkpoint.  The ledger must retain a terminal
            # REQUEST_CANCELLED result while neither live document changes.
            with patch.object(self.service, "_stage_cross_edit", blocked_stage):
                async with anyio.create_task_group() as group:
                    group.start_soon(transfer)
                    self.assertTrue(await anyio.to_thread.run_sync(entered.wait, 5))
                    group.cancel_scope.cancel()
                    release.set()

            self.assertIs(self.service.documents[source].project, source_project)
            self.assertIs(self.service.documents[target].project, target_project)
            self.assertEqual(source_document.revision, 0)
            self.assertEqual(target_document.revision, 0)
            self.assertEqual(self.project_semantics(source_project, "cancel-after-source.cb"),
                             source_before)
            self.assertEqual(self.project_semantics(target_project, "cancel-after-target.cb"),
                             target_before)

            replay = await self.call("relationship_transfer_tree_between", arguments)
            self.assertEqual(replay["error"]["code"], "REQUEST_CANCELLED")
            self.assertTrue(replay["replayed"])

            # A cancellation raised from inside the shielded publication block
            # must still publish both documents and complete the ledger entry.
            publication_arguments = self.cross_arguments(
                source, 0, target, 0, rectangle.internal_id, include_mops=True,
                request_id=str(uuid4()))
            original_complete = self.service._complete
            with anyio.CancelScope() as scope:
                def cancel_during_publication(name, entry, result):
                    scope.cancel()
                    return original_complete(name, entry, result)

                with patch.object(self.service, "_complete", cancel_during_publication):
                    completed = await self.call(
                        "relationship_transfer_tree_between", publication_arguments)
                    await anyio.lowlevel.checkpoint()

            self.assertTrue(completed["ok"], completed)
            self.assertEqual(source_document.revision, 1)
            self.assertEqual(target_document.revision, 1)
            self.assertEqual(
                (await self.inspect(source))["data"]["summary"]["counts"],
                {"layers": 1, "parts": 1, "primitives": 1, "mops": 0})
            self.assertEqual(
                (await self.inspect(target))["data"]["summary"]["counts"],
                {"layers": 2, "parts": 1, "primitives": 3, "mops": 1})
            replay = await self.call(
                "relationship_transfer_tree_between", publication_arguments)
            self.assertEqual(replay, {**completed, "replayed": True})

        self.run_async(test)

    def test_failures_address_the_right_document_and_preserve_state(self):
        async def test():
            source_direct, rectangle, island = self.source_project()
            source = await self.import_document(source_direct)
            target_direct = self.colliding_project()
            target = await self.import_document(target_direct)
            expired = f"00000000-0000-0000-0000-000000000000:{uuid4()}"

            arguments = self.cross_arguments(source, 0, target, 0, rectangle.internal_id)
            collision = await self.call("relationship_copy_tree_between", arguments)
            self.assertEqual(collision["error"]["code"], "INVALID_ARGUMENT")
            self.assertEqual(collision["error"]["field"], "root")
            self.assertIn("Geometry", collision["error"]["message"])
            self.assertEqual(collision["document"], source)
            self.assertEqual(collision["error"]["current_revision"], 0)

            mapped = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 0, target, 0, rectangle.internal_id, include_mops=True,
                identifier_map={"outline": "outline-copy", "circle": "circle-copy",
                                "Geometry": "Geometry-copy", "P": "P-copy",
                                "prof": "prof-copy"},
                group_map={"targets": "targets-copy"}))
            self.assertTrue(mapped["ok"], mapped)
            target_revision = 1

            stale_source = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 5, target, target_revision, rectangle.internal_id))
            self.assertEqual(stale_source["error"]["code"], "STALE_REVISION")
            self.assertEqual(stale_source["error"]["field"], "source_expected_revision")
            self.assertEqual(stale_source["error"]["current_revision"], 0)
            self.assertEqual(stale_source["document"], source)

            stale_target = await self.call("relationship_transfer_tree_between", self.cross_arguments(
                source, 0, target, 9, rectangle.internal_id))
            self.assertEqual(stale_target["error"]["code"], "STALE_REVISION")
            self.assertEqual(stale_target["error"]["field"], "target_expected_revision")
            self.assertEqual(stale_target["error"]["current_revision"], target_revision)
            self.assertEqual(stale_target["document"], target)

            missing = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 0, target, target_revision, uuid4()))
            self.assertEqual(missing["error"]["code"], "ENTITY_NOT_FOUND")
            self.assertEqual(missing["document"], source)

            mop = next(record["id"] for record in await self.inspect_records(source)
                       if record["kind"] == "mop")
            not_primitive = await self.call("relationship_transfer_tree_between", self.cross_arguments(
                source, 0, target, target_revision, mop))
            self.assertEqual(not_primitive["error"]["code"], "UNSUPPORTED_OPERATION")
            self.assertEqual(not_primitive["document"], source)

            same = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 0, source, 0, rectangle.internal_id))
            self.assertEqual(same["error"]["code"], "INVALID_ARGUMENT")
            self.assertEqual(same["error"]["field"], "target_document")

            expired_result = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 0, expired, 0, rectangle.internal_id))
            self.assertEqual(expired_result["error"]["code"], "DOCUMENT_EXPIRED")
            self.assertEqual(expired_result["document"], expired)

            closed = await self.call("document_close", self.args(
                document=target, expected_revision=target_revision))
            self.assertTrue(closed["ok"], closed)
            not_found = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 0, target, target_revision, rectangle.internal_id))
            self.assertEqual(not_found["error"]["code"], "DOCUMENT_NOT_FOUND")
            self.assertEqual(not_found["document"], target)
            self.assertEqual(not_found["error"]["current_revision"], None)

            conflict = await self.call("relationship_copy_tree_between", self.cross_arguments(
                source, 0, expired, 0, rectangle.internal_id,
                request_id=arguments["request_id"]))
            self.assertEqual(conflict["error"]["code"], "REQUEST_ID_CONFLICT")

            failed_args = self.cross_arguments(source, 0, expired, 0, str(rectangle.internal_id))
            failed = await self.call("relationship_copy_tree_between", failed_args)
            self.assertEqual(failed["error"]["code"], "DOCUMENT_EXPIRED")
            self.assertEqual(
                await self.call("relationship_copy_tree_between", failed_args),
                {**failed, "replayed": True})

            source_counts = (await self.inspect(source))["data"]["summary"]["counts"]
            self.assertEqual(source_counts,
                             {"layers": 1, "parts": 1, "primitives": 3, "mops": 1})

        self.run_async(test)

    def test_limits_and_concurrent_transfers_serialize_atomically(self):
        async def test():
            source_direct, rectangle, island = self.source_project()
            source = await self.import_document(source_direct)
            target = await self.create("spare")
            root = rectangle.internal_id

            with patch.object(DocumentService, "MAX_PRIMITIVES", 1):
                rejected = await self.call("relationship_copy_tree_between", self.cross_arguments(
                    source, 0, target, 0, root))
            self.assertEqual(rejected["error"]["code"], "LIMIT_EXCEEDED")
            self.assertEqual(rejected["document"], source)
            self.assertEqual((await self.inspect(target))["data"]["summary"]["counts"],
                             {"layers": 0, "parts": 0, "primitives": 0, "mops": 0})

            with patch.object(DocumentService, "MAX_MOPS", 0):
                rejected = await self.call("relationship_copy_tree_between", self.cross_arguments(
                    source, 0, target, 0, root, include_mops=True,
                    identifier_map={"outline": "outline-copy", "circle": "circle-copy",
                                    "Geometry": "Geometry-copy", "P": "P-copy",
                                    "prof": "prof-copy"},
                    group_map={"targets": "targets-copy"}))
            self.assertEqual(rejected["error"]["code"], "LIMIT_EXCEEDED")
            self.assertEqual((await self.inspect(target))["data"]["summary"]["counts"],
                             {"layers": 0, "parts": 0, "primitives": 0, "mops": 0})
            self.assertEqual((await self.inspect(source))["data"]["summary"]["counts"],
                             {"layers": 1, "parts": 1, "primitives": 3, "mops": 1})

            first = self.cross_arguments(source, 0, target, 0, root, include_mops=True)
            second = self.cross_arguments(source, 0, target, 0, root, include_mops=True)
            results = []

            async def transfer(arguments):
                results.append(await self.call("relationship_transfer_tree_between", arguments))

            async with anyio.create_task_group() as group:
                group.start_soon(transfer, first)
                group.start_soon(transfer, second)
            self.assertEqual(sum(result["ok"] for result in results), 1)
            self.assertEqual(
                sum(not result["ok"] and result["error"]["code"] == "STALE_REVISION"
                    for result in results),
                1)
            self.assertEqual((await self.inspect(source))["data"]["summary"]["counts"],
                             {"layers": 1, "parts": 1, "primitives": 1, "mops": 0})
            self.assertEqual((await self.inspect(target))["data"]["summary"]["counts"],
                             {"layers": 1, "parts": 1, "primitives": 2, "mops": 1})

        self.run_async(test)

    def test_opposing_direction_transfers_do_not_deadlock(self):
        async def test():
            source_direct, rectangle, island = self.source_project()
            target_direct = self.target_project()
            anchor = target_direct.list_primitives()[0]
            source = await self.import_document(source_direct)
            target = await self.import_document(target_direct)

            forward = self.cross_arguments(
                source, 0, target, 0, rectangle.internal_id, include_mops=True)
            reverse = self.cross_arguments(
                target, 0, source, 0, anchor.internal_id)

            class FirstAcquireBarrier:
                def __init__(self):
                    self.entered = anyio.Event()
                    self.release = anyio.Event()
                    self.seen = False

            class BlockingFirstLock:
                def __init__(self, lock, barrier):
                    self.lock = lock
                    self.barrier = barrier

                async def __aenter__(self):
                    await self.lock.acquire()
                    if not self.barrier.seen:
                        self.barrier.seen = True
                        self.barrier.entered.set()
                        try:
                            await self.barrier.release.wait()
                        except BaseException:
                            self.lock.release()
                            raise
                    return None

                async def __aexit__(self, exc_type, exc_value, traceback):
                    self.lock.release()

            barrier = FirstAcquireBarrier()
            source_document = self.service.documents[source]
            target_document = self.service.documents[target]
            source_document.lock = BlockingFirstLock(source_document.lock, barrier)
            target_document.lock = BlockingFirstLock(target_document.lock, barrier)

            results = {}
            reverse_started = anyio.Event()

            async def transfer(label, arguments, started=None):
                if started is not None:
                    started.set()
                results[label] = await self.call(
                    "relationship_transfer_tree_between", arguments)

            # Hold the first operation after its first document lock.  With
            # source-then-target locking, the opposing operation would hold
            # the other lock and both would wait forever.  Canonical ordering
            # makes both operations contend for the same first lock instead.
            try:
                with anyio.fail_after(5):
                    async with anyio.create_task_group() as group:
                        group.start_soon(transfer, "forward", forward)
                        await barrier.entered.wait()
                        group.start_soon(transfer, "reverse", reverse, reverse_started)
                        await reverse_started.wait()
                        await anyio.sleep(0)
                        barrier.release.set()
            finally:
                barrier.release.set()

            self.assertEqual(set(results), {"forward", "reverse"})
            self.assertEqual(sum(result["ok"] for result in results.values()), 1)
            self.assertEqual(
                sum(not result["ok"] and result["error"]["code"] == "STALE_REVISION"
                    for result in results.values()),
                1)
            self.assertEqual(source_document.revision, 1)
            self.assertEqual(target_document.revision, 1)
            successful = next(result for result in results.values() if result["ok"])
            self.assertEqual(successful["data"]["source_revision"], 1)
            self.assertEqual(successful["data"]["target_revision"], 1)

        self.run_async(test)


if __name__ == "__main__":
    unittest.main()
