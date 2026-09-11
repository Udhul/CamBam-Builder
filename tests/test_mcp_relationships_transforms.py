"""4d fourth-batch MCP relationship and transform breadth acceptance.

Parent/group/copy relationships and similarity transforms through the adapter
are compared with independently authored public-framework projects, including
typed geometry after transforms, bake semantics, cycle rejection and the
negative/resilience cases required before advertisement.
"""

import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from uuid import uuid4

if find_spec("mcp") is not None:
    import anyio
    from cambam_builder.mcp_adapter.service import DocumentService
    from cambam_builder.mcp_adapter.schema import OUTPUTS
else:  # pragma: no cover - exercised only in a base-library environment
    DocumentService = None
    OUTPUTS = {}

from cambam_builder import CBProject
from cambam_builder.cambam_entities import Vertex


UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?(?![A-Za-z])"
)


@unittest.skipIf(DocumentService is None, "Install .[mcp] for adapter checks")
class RelationshipTransformTests(unittest.TestCase):
    """DocumentService contract tests for the relationship/transform family."""

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
        validator = OUTPUTS.get(name)
        if validator is not None:
            validator.validate(result)
        return result

    async def create(self, name="relations"):
        result = await self.call("document_create", self.args(name=name, units="mm"))
        self.assertTrue(result["ok"], result)
        return result["document"]

    async def inspect(self, handle, *, revision=None):
        arguments = {
            "workspace_id": self.service.workspace.id,
            "document": handle,
        }
        if revision is not None:
            arguments["expected_revision"] = revision
        result = await self.call("document_inspect", arguments)
        self.assertTrue(result["ok"], result)
        return result

    async def inspect_records(self, handle, revision=None):
        arguments = {
            "workspace_id": self.service.workspace.id,
            "document": handle,
        }
        if revision is not None:
            arguments["expected_revision"] = revision
        result = await self.call("document_inspect", arguments)
        self.assertTrue(result["ok"], result)
        data = result["data"]
        records = list(data["entities"])
        while data["next_offset"] is not None:
            result = await self.call("document_inspect", {
                **arguments, "offset": data["next_offset"],
            })
            self.assertTrue(result["ok"], result)
            data = result["data"]
            records.extend(data["entities"])
        return records

    @staticmethod
    def diagnostics(result):
        return {item["code"] for item in result["diagnostics"]}

    @staticmethod
    def assert_records_close(testcase, got, want, places=9):
        if isinstance(got, float) or isinstance(want, float):
            testcase.assertAlmostEqual(float(got), float(want), places=places)
        elif isinstance(got, list) and isinstance(want, list):
            testcase.assertEqual(len(got), len(want))
            for first, second in zip(got, want):
                RelationshipTransformTests.assert_records_close(
                    testcase, first, second, places)
        elif isinstance(got, dict) and isinstance(want, dict):
            testcase.assertEqual(set(got), set(want))
            for key in want:
                RelationshipTransformTests.assert_records_close(
                    testcase, got[key], want[key], places)
        else:
            testcase.assertEqual(got, want)

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

    @staticmethod
    def direct_transform_project():
        project = CBProject("relations")
        layer = project.add_layer("Geometry")
        circle = project.add_circle(layer, center=(5, 5), diameter=4,
                                    identifier="circle", elevation=1)
        pline = project.add_pline(layer, identifier="shape", closed=False,
                                  points=[Vertex(0, 0), Vertex(10, 0, 0, bulge=0.5),
                                          Vertex(10, 10, 1)])
        rectangle = project.add_rect(layer, corner=(0, 0), width=20, height=10,
                                     identifier="outline")
        return project, circle, pline, rectangle

    def test_similarity_transforms_keep_typed_geometry_with_parity(self):
        async def test():
            handle = await self.create()
            revision = 0
            ids = {}

            async def add(tool, **arguments):
                nonlocal revision
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                return result

            for tool, arguments in (
                ("geometry_add_circle", {
                    "identifier": "circle", "layer": "Geometry",
                    "x": 5, "y": 5, "diameter": 4, "z": 1}),
                ("geometry_add_pline", {
                    "identifier": "shape", "layer": "Geometry", "closed": False,
                    "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0, "bulge": 0.5},
                               {"x": 10, "y": 10, "z": 1}]}),
                ("geometry_add_rectangle", {
                    "identifier": "outline", "layer": "Geometry",
                    "x": 0, "y": 0, "width": 20, "height": 10}),
            ):
                result = await add(tool, **arguments)
                ids[arguments["identifier"]] = result["data"]["entity_id"]

            rotated = await add("geometry_rotate", entity_id=ids["circle"],
                                angle_deg=40, cx=0, cy=0)
            self.assertEqual(rotated["data"], {"entity_id": ids["circle"]})
            await add("geometry_rotate", entity_id=ids["circle"], angle_deg=90)
            await add("geometry_scale", entity_id=ids["shape"], factor=2,
                      cx=0, cy=0)
            await add("geometry_mirror", entity_id=ids["shape"], axis="x", position=0)
            await add("geometry_mirror", entity_id=ids["outline"], axis="x", position=0)
            await add("geometry_translate_z", entity_id=ids["outline"], dz=2)
            baked = await add("geometry_bake", entity_id=ids["circle"])
            self.assertEqual(baked["data"]["type"], "Circle")

            records = await self.inspect_records(handle, revision=revision)
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED", self.diagnostics(await self.inspect(handle)))
            by_identifier = {record["identifier"]: record for record in records
                             if record["kind"] == "primitive"}
            self.assertEqual(by_identifier["shape"]["geometry"]["world_xyz"],
                             [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, -20.0, 1.0]])
            self.assertEqual(by_identifier["shape"]["geometry"]["bulges"],
                             [0.0, -0.5, 0.0])
            self.assertEqual(by_identifier["outline"]["geometry"]["world_xyz"],
                             [[0.0, 0.0, 2.0], [20.0, 0.0, 2.0],
                              [20.0, -10.0, 2.0], [0.0, -10.0, 2.0]])
            self.assertEqual(by_identifier["outline"]["geometry"]["bounds"],
                             [0.0, -10.0, 20.0, 0.0])
            self.assertEqual(
                round(by_identifier["circle"]["geometry"]["diameter"], 9), 4.0)

            direct, circle, pline, rectangle = self.direct_transform_project()
            self.assertTrue(direct.rotate_primitive_deg(circle, 40, 0, 0, bake=False))
            self.assertTrue(direct.rotate_primitive_deg(circle, 90, bake=False))
            self.assertTrue(direct.scale_primitive(pline, 2, 2, 0, 0, bake=False))
            self.assertTrue(direct.mirror_primitive_x(pline, 0, bake=False))
            self.assertTrue(direct.mirror_primitive_x(rectangle, 0, bake=False))
            self.assertTrue(direct.translate_primitive_z(rectangle, 2, bake=True))
            circle_world = circle.get_absolute_coordinates_xyz()
            self.assertEqual([round(float(value), 9) for value in circle_world["center"]],
                             [round(float(value), 9)
                              for value in by_identifier["circle"]["geometry"]["center"]])
            self.assertEqual(
                pline.get_absolute_coordinates_xyz()[1][3],
                by_identifier["shape"]["geometry"]["bulges"][1])
            circle.bake_geometry()

            saved_a = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="A.cb"))
            self.assertTrue(saved_a["ok"], saved_a)
            opened = await self.call("document_open", self.args(path="A.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            reopened_records = await self.inspect_records(opened["document"], revision=0)
            self.assert_records_close(self, reopened_records, records)
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED",
                self.diagnostics(await self.inspect(opened["document"])))

            direct.save(str(self.root / "direct-A.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "A.cb"),
                self.xml_semantics(self.root / "direct-A.cb"),
            )

        self.run_async(test)

    def test_relationship_tools_copy_parity_and_cycle_rejection(self):
        async def test():
            handle = await self.create()
            revision = 0
            ids = {}

            async def add(tool, **arguments):
                nonlocal revision
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                return result

            rectangle = (await add("geometry_add_rectangle", identifier="outline",
                                   layer="Geometry", x=0, y=0, width=20, height=10))
            circle = (await add("geometry_add_circle", identifier="circle",
                                layer="Geometry", x=5, y=5, diameter=4))
            ids = {"outline": rectangle["data"]["entity_id"],
                   "circle": circle["data"]["entity_id"]}

            grouped = await add("relationship_add_to_group",
                                entity_id=ids["circle"], group="targets")
            self.assertEqual(grouped["data"]["groups"], ["targets"])
            linked = await add("relationship_set_parent",
                               entity_id=ids["circle"], parent_id=ids["outline"])
            self.assertEqual(linked["data"],
                             {"entity_id": ids["circle"], "parent": ids["outline"]})

            records = await self.inspect_records(handle, revision=revision)
            self.assertIn("INSPECTION_UNSUPPORTED",
                          self.diagnostics(await self.inspect(handle)))
            by_identifier = {record["identifier"]: record for record in records
                             if record["kind"] == "primitive"}
            self.assertEqual(by_identifier["circle"]["parent"], ids["outline"])
            self.assertEqual(by_identifier["outline"]["children"], [ids["circle"]])
            self.assertEqual(by_identifier["circle"]["groups"], ["targets"])
            self.assertIsNone(by_identifier["circle"]["geometry"])
            self.assertIsNone(by_identifier["outline"]["geometry"])

            copied = await add("relationship_copy_tree", root=ids["outline"],
                               identifier_map={"outline": "outline-copy",
                                               "circle": "circle-copy",
                                               "Geometry": "Geometry-copy"},
                               group_map={"targets": "targets-copy"})
            mapping = copied["data"]["mapping"]
            self.assertIn(ids["outline"], mapping)
            self.assertIn(ids["circle"], mapping)
            new_rect = mapping[ids["outline"]]
            new_circle = mapping[ids["circle"]]

            records = await self.inspect_records(handle, revision=revision)
            by_identifier = {record["identifier"]: record for record in records
                             if record["kind"] == "primitive"}
            self.assertEqual(by_identifier["outline-copy"]["id"], new_rect)
            self.assertEqual(by_identifier["outline-copy"]["children"], [new_circle])
            self.assertIsNone(by_identifier["outline-copy"]["geometry"])
            self.assertEqual(by_identifier["circle-copy"]["parent"], new_rect)
            self.assertEqual(by_identifier["circle-copy"]["groups"], ["targets-copy"])

            detached = await add("relationship_set_parent",
                                 entity_id=ids["circle"], parent_id=None)
            self.assertEqual(detached["data"],
                             {"entity_id": ids["circle"], "parent": None})
            removed = await add("relationship_remove_from_group",
                                entity_id=ids["circle"], group="targets")
            self.assertEqual(removed["data"], {"entity_id": ids["circle"], "groups": []})

            cycle = await self.call("relationship_set_parent", self.args(
                document=handle, expected_revision=revision,
                entity_id=new_rect, parent_id=new_circle))
            self.assertEqual(cycle["error"]["code"], "INVALID_ARGUMENT")
            self_loop = await self.call("relationship_set_parent", self.args(
                document=handle, expected_revision=revision,
                entity_id=ids["circle"], parent_id=ids["circle"]))
            self.assertEqual(self_loop["error"]["code"], "INVALID_ARGUMENT")
            missing_parent = await self.call("relationship_set_parent", self.args(
                document=handle, expected_revision=revision,
                entity_id=ids["circle"], parent_id=str(uuid4())))
            self.assertEqual(missing_parent["error"]["code"], "ENTITY_NOT_FOUND")
            missing_copy_root = await self.call("relationship_copy_tree", self.args(
                document=handle, expected_revision=revision, root=str(uuid4())))
            self.assertEqual(missing_copy_root["error"]["code"], "ENTITY_NOT_FOUND")
            conflict_copy = await self.call("relationship_copy_tree", self.args(
                document=handle, expected_revision=revision, root=ids["circle"]))
            self.assertEqual(conflict_copy["error"]["code"], "INVALID_ARGUMENT")
            missing_layer_map = await self.call("relationship_copy_tree", self.args(
                document=handle, expected_revision=revision, root=ids["circle"],
                identifier_map={"circle": "circle-2"}))
            self.assertEqual(missing_layer_map["error"]["code"], "INVALID_ARGUMENT")
            counts = (await self.inspect(handle))["data"]["summary"]["counts"]
            self.assertEqual(counts, {"layers": 2, "parts": 0, "primitives": 4, "mops": 0})

            saved_a = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="A.cb"))
            self.assertTrue(saved_a["ok"], saved_a)

            direct = CBProject("relations")
            layer = direct.add_layer("Geometry")
            direct_rect = direct.add_rect(layer, corner=(0, 0), width=20, height=10,
                                          identifier="outline")
            direct_circle = direct.add_circle(layer, center=(5, 5), diameter=4,
                                              identifier="circle")
            direct.add_primitive_to_group(direct_circle, "targets")
            direct.link_primitive_parent(direct_circle, direct_rect)
            direct.copy_primitive_tree(
                direct_rect, direct, preserve_ids=False,
                identifier_map={"outline": "outline-copy", "circle": "circle-copy",
                                "Geometry": "Geometry-copy"},
                group_map={"targets": "targets-copy"})
            direct.link_primitive_parent(direct_circle, None)
            direct.remove_primitive_from_group(direct_circle, "targets")
            direct.save(str(self.root / "direct-A.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "A.cb"),
                self.xml_semantics(self.root / "direct-A.cb"),
            )

        self.run_async(test)

    def test_copy_tree_with_mops_keeps_slice_valid_copies(self):
        async def test():
            handle = await self.create()
            revision = 0
            ids = {}

            async def add(tool, **arguments):
                nonlocal revision
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                return result

            rectangle = (await add("geometry_add_rectangle", identifier="base",
                                   layer="Geometry", x=0, y=0, width=20, height=10))
            profile = (await add("machining_add_profile", identifier="prof",
                                 part="P", targets=[rectangle["data"]["entity_id"]],
                                 side="Inside", target_depth=-1, depth_increment=0.5,
                                 tool_diameter=3, cut_feedrate=300,
                                 plunge_feedrate=100, spindle_speed=12000,
                                 clearance_plane=5))
            ids = {"base": rectangle["data"]["entity_id"],
                   "prof": profile["data"]["mop_id"]}

            copied = await add("relationship_copy_tree", root=ids["base"],
                               include_mops=True,
                               identifier_map={"base": "base-copy",
                                               "prof": "prof-copy", "P": "P-copy",
                                               "Geometry": "Geometry-copy"})
            mapping = copied["data"]["mapping"]
            self.assertIn(ids["base"], mapping)
            self.assertIn(ids["prof"], mapping)
            new_rect, new_mop = mapping[ids["base"]], mapping[ids["prof"]]

            records = await self.inspect_records(handle, revision=revision)
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED", self.diagnostics(await self.inspect(handle)))
            by_identifier = {record["identifier"]: record for record in records
                             if record["kind"] in ("primitive", "mop")}
            self.assertEqual(by_identifier["base-copy"]["id"], new_rect)
            self.assertIsNone(by_identifier["base-copy"]["parent"])
            self.assertEqual(by_identifier["base-copy"]["children"], [])
            self.assertEqual(by_identifier["base-copy"]["geometry"]["kind"], "rect")
            self.assertEqual(by_identifier["prof-copy"]["id"], new_mop)
            self.assertEqual(by_identifier["prof-copy"]["part"], "P-copy")
            self.assertEqual(by_identifier["prof-copy"]["targets"], [new_rect])
            self.assertNotEqual(by_identifier["prof-copy"]["parameters"], {})

            saved_a = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="A.cb"))
            self.assertTrue(saved_a["ok"], saved_a)

            direct = CBProject("relations")
            layer = direct.add_layer("Geometry")
            direct_rect = direct.add_rect(layer, corner=(0, 0), width=20, height=10,
                                          identifier="base")
            part = direct.add_part(
                "P", enabled=True, stock_width=0, stock_height=0,
                stock_thickness=0, stock_material="")
            direct.add_profile_mop(
                part, targets=[direct_rect], identifier="prof", name="prof",
                profile_side="Inside", target_depth=-1, depth_increment=0.5,
                tool_diameter=3, cut_feedrate=300, plunge_feedrate=100,
                spindle_speed=12000, stock_surface=0, clearance_plane=5,
                lead_in_type="None")
            direct.copy_primitive_tree(
                direct_rect, direct, preserve_ids=False,
                identifier_map={"base": "base-copy", "prof": "prof-copy",
                                "P": "P-copy", "Geometry": "Geometry-copy"},
                include_mops=True)
            direct.save(str(self.root / "direct-A.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "A.cb"),
                self.xml_semantics(self.root / "direct-A.cb"),
            )

        self.run_async(test)

    def test_copy_tree_enforces_primitive_and_mop_document_limits_atomically(self):
        async def test():
            handle = await self.create()
            rectangle = await self.call("geometry_add_rectangle", self.args(
                document=handle, expected_revision=0, identifier="base",
                layer="Geometry", x=0, y=0, width=20, height=10,
            ))
            self.assertTrue(rectangle["ok"], rectangle)
            rectangle_id = rectangle["data"]["entity_id"]

            copy_arguments = {
                "document": handle,
                "expected_revision": 1,
                "root": rectangle_id,
                "identifier_map": {
                    "base": "base-copy", "Geometry": "Geometry-copy",
                },
            }
            with patch.object(DocumentService, "MAX_PRIMITIVES", 1):
                rejected = await self.call(
                    "relationship_copy_tree", self.args(**copy_arguments))
            self.assertEqual(rejected["error"]["code"], "LIMIT_EXCEEDED")
            self.assertEqual(rejected["revision"], 1)
            self.assertEqual(
                (await self.inspect(handle, revision=1))["data"]["summary"]["counts"],
                {"layers": 1, "parts": 0, "primitives": 1, "mops": 0},
            )

            profile = await self.call("machining_add_profile", self.args(
                document=handle, expected_revision=1, identifier="prof", part="P",
                targets=[rectangle_id], side="Inside", target_depth=-1,
                depth_increment=0.5, tool_diameter=3, cut_feedrate=300,
                plunge_feedrate=100, spindle_speed=12000, clearance_plane=5,
            ))
            self.assertTrue(profile["ok"], profile)
            with patch.object(DocumentService, "MAX_MOPS", 1):
                rejected = await self.call("relationship_copy_tree", self.args(
                    document=handle, expected_revision=2, root=rectangle_id,
                    include_mops=True,
                    identifier_map={
                        "base": "base-copy", "Geometry": "Geometry-copy",
                        "prof": "prof-copy", "P": "P-copy",
                    },
                ))
            self.assertEqual(rejected["error"]["code"], "LIMIT_EXCEEDED")
            self.assertEqual(rejected["revision"], 2)
            self.assertEqual(
                (await self.inspect(handle, revision=2))["data"]["summary"]["counts"],
                {"layers": 1, "parts": 1, "primitives": 1, "mops": 1},
            )

        self.run_async(test)

    def test_transform_negatives_retries_and_atomic_failure(self):
        async def test():
            handle = await self.create()
            revision = 0

            async def add(tool, **arguments):
                nonlocal revision
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                return result

            circle = (await add("geometry_add_circle", identifier="circle",
                                layer="Geometry", x=5, y=5, diameter=4))
            circle_id = circle["data"]["entity_id"]
            note = (await add("geometry_add_text", identifier="note",
                              layer="Annotation", text="note", x=0, y=0))
            note_id = note["data"]["entity_id"]
            await add("geometry_rotate", entity_id=note_id, angle_deg=25)
            await add("geometry_scale", entity_id=circle_id, factor=3)
            grouped = await add("relationship_add_to_group",
                                entity_id=note_id, group="labels")
            self.assertEqual(grouped["data"], {"entity_id": note_id,
                                               "groups": ["labels"]})

            for tool, arguments, expected in (
                ("geometry_rotate", {"entity_id": str(uuid4()), "angle_deg": 5},
                 "ENTITY_NOT_FOUND"),
                ("geometry_rotate", {"entity_id": note_id, "angle_deg": 5},
                 "UNSUPPORTED_OPERATION"),
                ("geometry_scale", {"entity_id": circle_id, "factor": 0},
                 "INVALID_ARGUMENT"),
                ("geometry_scale", {"entity_id": circle_id, "factor": -1},
                 "INVALID_ARGUMENT"),
                ("geometry_rotate", {"entity_id": circle_id, "angle_deg": 5, "cx": 1},
                 "INVALID_ARGUMENT"),
                ("geometry_mirror", {"entity_id": circle_id, "axis": "z"},
                 "INVALID_ARGUMENT"),
                ("geometry_translate_z", {"entity_id": circle_id, "dz": "up"},
                 "INVALID_ARGUMENT"),
                ("geometry_bake", {"entity_id": note_id}, "UNSUPPORTED_OPERATION"),
                ("relationship_add_to_group",
                 {"entity_id": circle_id, "group": "bad\x01group"},
                 "INVALID_ARGUMENT"),
            ):
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertEqual(result["error"]["code"], expected, (tool, arguments))

            rotate = self.args(document=handle, expected_revision=revision,
                               entity_id=circle_id, angle_deg=10)
            rotated = await self.call("geometry_rotate", rotate)
            self.assertTrue(rotated["ok"], rotated)
            revision += 1
            self.assertEqual(await self.call("geometry_rotate", rotate), {
                **rotated, "replayed": True,
            })
            before = (await self.inspect(handle, revision=revision))["data"]
            stale = await self.call("geometry_scale", self.args(
                document=handle, expected_revision=revision - 1,
                entity_id=circle_id, factor=2))
            self.assertEqual(stale["error"]["code"], "STALE_REVISION")
            self.assertEqual((await self.inspect(handle, revision=revision))["data"],
                             before)

            requests = [
                self.args(document=handle, expected_revision=revision,
                          entity_id=circle_id, angle_deg=1),
                self.args(document=handle, expected_revision=revision,
                          entity_id=circle_id, angle_deg=2),
            ]
            results = []

            async def rotate_call(arguments):
                results.append(await self.call("geometry_rotate", arguments))

            async with anyio.create_task_group() as group:
                for arguments in requests:
                    group.start_soon(rotate_call, arguments)
            self.assertEqual(sum(result["ok"] for result in results), 1)
            self.assertEqual(
                sum(not result["ok"] and result["error"]["code"] == "STALE_REVISION"
                    for result in results),
                1,
            )

        self.run_async(test)


if __name__ == "__main__":
    unittest.main()
