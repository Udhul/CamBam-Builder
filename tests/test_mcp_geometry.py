"""4d first-batch MCP geometry breadth acceptance.

Circle, Arc, Pline and Points authoring through the adapter is compared with
independently authored public-framework projects, including typed world
geometry, bulge-aware bounds, save/reopen identity retention and the
negative/resilience cases required before advertising a family.
"""

import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import re
import tempfile
import unittest
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


def bounds_of(entity):
    box = entity.get_bounding_box()
    return [box.min_x, box.min_y, box.max_x, box.max_y]


@unittest.skipIf(DocumentService is None, "Install .[mcp] for adapter checks")
class GeometryBreadthTests(unittest.TestCase):
    """DocumentService contract tests for the curve/point geometry family."""

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

    async def create(self, name="geometry"):
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
        """Compare inspection records; public-query floats may differ in noise."""
        if isinstance(got, float) or isinstance(want, float):
            testcase.assertAlmostEqual(float(got), float(want), places=places)
        elif isinstance(got, list) and isinstance(want, list):
            testcase.assertEqual(len(got), len(want))
            for first, second in zip(got, want):
                GeometryBreadthTests.assert_records_close(testcase, first, second, places)
        elif isinstance(got, dict) and isinstance(want, dict):
            testcase.assertEqual(set(got), set(want))
            for key in want:
                GeometryBreadthTests.assert_records_close(testcase, got[key], want[key], places)
        else:
            testcase.assertEqual(got, want)

    @staticmethod
    def xml_semantics(path):
        """Return parsed XML semantics while ignoring run-specific UUIDs.

        Primitive document order and the transient XML ``id`` numbering follow
        the framework's UUID-sorted listing, not authoring order, so both are
        canonicalized by the stable ``user_id`` identity before comparison.
        """
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
    def direct_project():
        """Build the same family through the public framework API."""
        project = CBProject("geometry")
        layer = project.add_layer("Geometry")
        circle = project.add_circle(
            layer, center=(5, 5), diameter=4, identifier="circle", elevation=1,
        )
        arc = project.add_arc(
            layer, center=(0, 0), radius=10, start_angle=30, extent_angle=120,
            identifier="arc", elevation=-2,
        )
        pline = project.add_pline(
            layer, identifier="shape", closed=False,
            points=[Vertex(0, 0, 0), Vertex(10, 0, 0, bulge=0.5), Vertex(10, 10, 1)],
        )
        marks = project.add_points(
            layer, identifier="marks",
            points=[Vertex(1, 2, 3), Vertex(4, 5, 6)],
        )
        return project, circle, arc, pline, marks

    @staticmethod
    def expected_geometries(circle, arc, pline, marks):
        """Expected typed payloads from independent public-framework queries."""
        circle_world = circle.get_absolute_coordinates_xyz()
        arc_world = arc.get_absolute_coordinates_xyz()
        pline_world = pline.get_absolute_coordinates_xyz()
        marks_world = marks.get_absolute_coordinates_xyz()
        return {
            "circle": {
                "kind": "circle",
                "center": [float(value) for value in circle_world["center"]],
                "diameter": float(circle_world["diameter"]),
                "bounds": bounds_of(circle),
            },
            "arc": {
                "kind": "arc",
                "center": [float(value) for value in arc_world["center"]],
                "radius": float(arc_world["radius"]),
                "start_angle": float(arc_world["start_angle"]),
                "extent_angle": float(arc_world["extent_angle"]),
                "bounds": bounds_of(arc),
            },
            "pline": {
                "kind": "pline",
                "world_xyz": [[float(value) for value in point[:3]] for point in pline_world],
                "bulges": [float(point[3]) for point in pline_world],
                "closed": False,
                "bounds": bounds_of(pline),
            },
            "points": {
                "kind": "points",
                "world_xyz": [[float(value) for value in point] for point in marks_world],
                "bounds": bounds_of(marks),
            },
        }

    def test_expected_payloads_match_analytic_values(self):
        """Guard the direct-query parity source against hand-derived geometry."""
        _, circle, arc, pline, marks = self.direct_project()
        expected = self.expected_geometries(circle, arc, pline, marks)
        self.assertEqual(expected["circle"]["center"], [5.0, 5.0, 1.0])
        self.assertEqual(expected["circle"]["diameter"], 4.0)
        self.assertEqual(expected["circle"]["bounds"], [3.0, 3.0, 7.0, 7.0])
        self.assertEqual(expected["arc"]["center"], [0.0, 0.0, -2.0])
        self.assertEqual(expected["arc"]["radius"], 10.0)
        self.assertAlmostEqual(expected["arc"]["start_angle"], 30.0, places=9)
        self.assertEqual(expected["arc"]["extent_angle"], 120.0)
        for index, value in zip(range(4), (-8.660254037844387, 5.0, 8.660254037844387, 10.0)):
            self.assertAlmostEqual(expected["arc"]["bounds"][index], value, places=9)
        self.assertEqual(expected["pline"]["world_xyz"],
                         [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 1.0]])
        self.assertEqual(expected["pline"]["bulges"], [0.0, 0.5, 0.0])
        # The bulge stored on the second vertex curves the (10,0)->(10,10)
        # segment to x=12.5; endpoint-only bounds would wrongly stop at 10.
        self.assertEqual(expected["pline"]["bounds"], [0.0, 0.0, 12.5, 10.0])
        self.assertEqual(expected["points"]["world_xyz"],
                         [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(expected["points"]["bounds"], [1.0, 2.0, 4.0, 5.0])

    def test_add_curves_points_roundtrip_and_direct_parity(self):
        async def test():
            handle = await self.create()
            layer = "Geometry"
            revision = 0
            ids = {}
            for tool, arguments in (
                ("geometry_add_circle", {
                    "identifier": "circle", "layer": layer,
                    "x": 5, "y": 5, "diameter": 4, "z": 1,
                }),
                ("geometry_add_arc", {
                    "identifier": "arc", "layer": layer,
                    "x": 0, "y": 0, "radius": 10, "start_angle": 30,
                    "extent_angle": 120, "z": -2,
                }),
                ("geometry_add_pline", {
                    "identifier": "shape", "layer": layer, "closed": False,
                    "points": [
                        {"x": 0, "y": 0}, {"x": 10, "y": 0, "bulge": 0.5},
                        {"x": 10, "y": 10, "z": 1},
                    ],
                }),
                ("geometry_add_points", {
                    "identifier": "marks", "layer": layer,
                    "points": [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}],
                }),
            ):
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                self.assertEqual(result["revision"], revision)
                self.assertEqual(result["data"]["layer"], layer)
                ids[arguments["identifier"]] = result["data"]["entity_id"]

            records = await self.inspect_records(handle, revision=revision)
            self.assertEqual(
                [record["kind"] for record in records],
                ["layer", "primitive", "primitive", "primitive", "primitive"],
            )
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED", self.diagnostics(await self.inspect(handle)))
            by_identifier = {record["identifier"]: record for record in records[1:]}
            expected = self.expected_geometries(*self.direct_project()[1:])
            for name, identifier in (("circle", "circle"), ("arc", "arc"),
                                     ("pline", "shape"), ("points", "marks")):
                record = by_identifier[identifier]
                self.assertEqual(record["id"], ids[identifier])
                self.assertEqual(record["geometry"], expected[name])
                self.assertIsNone(record["parent"])
                self.assertEqual(record["children"], [])

            saved_a = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="A.cb"))
            self.assertTrue(saved_a["ok"], saved_a)
            bytes_a = (self.root / "A.cb").read_bytes()
            self.assertEqual(saved_a["data"]["bytes"], len(bytes_a))
            self.assertEqual(saved_a["data"]["sha256"], hashlib.sha256(bytes_a).hexdigest())

            opened = await self.call("document_open", self.args(path="A.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            reopened = opened["document"]
            reopened_records = await self.inspect_records(reopened, revision=0)
            self.assert_records_close(self, reopened_records, records)

            direct, _, _, _, _ = self.direct_project()
            direct.save(str(self.root / "direct-A.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "A.cb"),
                self.xml_semantics(self.root / "direct-A.cb"),
            )

        self.run_async(test)

    def test_translate_supported_families_with_reopen_parity(self):
        async def test():
            handle = await self.create()
            ids = {}
            revision = 0
            for tool, arguments in (
                ("geometry_add_rectangle", {
                    "identifier": "outline", "layer": "Geometry",
                    "x": 0, "y": 0, "width": 20, "height": 10,
                }),
                ("geometry_add_circle", {
                    "identifier": "circle", "layer": "Geometry",
                    "x": 5, "y": 5, "diameter": 4, "z": 1,
                }),
                ("geometry_add_pline", {
                    "identifier": "shape", "layer": "Geometry", "closed": False,
                    "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0, "bulge": 0.5},
                               {"x": 10, "y": 10, "z": 1}],
                }),
                ("geometry_add_arc", {
                    "identifier": "arc", "layer": "Geometry",
                    "x": 0, "y": 0, "radius": 10, "start_angle": 30,
                    "extent_angle": 120, "z": -2,
                }),
                ("geometry_add_points", {
                    "identifier": "marks", "layer": "Geometry",
                    "points": [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}],
                }),
            ):
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                ids[arguments["identifier"]] = result["data"]["entity_id"]

            moves = {
                "outline": (5, 2), "circle": (2, 3), "shape": (-1, 0),
                "arc": (10, -1), "marks": (0.5, 0.5),
            }
            for identifier, (dx, dy) in moves.items():
                result = await self.call("geometry_translate", self.args(
                    document=handle, expected_revision=revision,
                    entity_id=ids[identifier], dx=dx, dy=dy))
                self.assertTrue(result["ok"], result)
                revision += 1

            records = await self.inspect_records(handle, revision=revision)
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED", self.diagnostics(await self.inspect(handle)))
            by_identifier = {record["identifier"]: record for record in records
                             if record["kind"] == "primitive"}
            self.assertEqual(by_identifier["outline"]["geometry"]["world_xyz"],
                             [[5.0, 2.0, 0.0], [25.0, 2.0, 0.0],
                              [25.0, 12.0, 0.0], [5.0, 12.0, 0.0]])
            self.assertEqual(by_identifier["circle"]["geometry"]["center"], [7.0, 8.0, 1.0])
            self.assertEqual(by_identifier["circle"]["geometry"]["bounds"],
                             [5.0, 6.0, 9.0, 10.0])
            self.assertEqual(by_identifier["shape"]["geometry"]["world_xyz"],
                             [[-1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [9.0, 10.0, 1.0]])
            self.assertEqual(by_identifier["shape"]["geometry"]["bulges"], [0.0, 0.5, 0.0])
            self.assertEqual(by_identifier["shape"]["geometry"]["bounds"],
                             [-1.0, 0.0, 11.5, 10.0])
            self.assertEqual(by_identifier["arc"]["geometry"]["center"], [10.0, -1.0, -2.0])
            self.assertEqual(by_identifier["marks"]["geometry"]["world_xyz"],
                             [[1.5, 2.5, 3.0], [4.5, 5.5, 6.0]])

            saved_a = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="A.cb"))
            self.assertTrue(saved_a["ok"], saved_a)
            opened = await self.call("document_open", self.args(path="A.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            reopened = opened["document"]
            reopened_records = await self.inspect_records(reopened, revision=0)
            self.assert_records_close(self, reopened_records, records)

            translate = self.args(document=reopened, expected_revision=0,
                                  entity_id=ids["shape"], dx=4, dy=0)
            translated = await self.call("geometry_translate", translate)
            self.assertTrue(translated["ok"], translated)
            self.assertEqual(translated["revision"], 1)
            self.assertEqual(await self.call("geometry_translate", translate), {
                **translated, "replayed": True,
            })
            after = await self.inspect_records(reopened, revision=1)
            moved = next(record for record in after
                         if record["kind"] == "primitive"
                         and record["identifier"] == "shape")
            self.assertEqual(moved["geometry"]["world_xyz"],
                             [[3.0, 0.0, 0.0], [13.0, 0.0, 0.0], [13.0, 10.0, 1.0]])
            self.assertEqual(moved["geometry"]["bounds"], [3.0, 0.0, 15.5, 10.0])
            self.assertEqual(moved["id"], ids["shape"])

            direct = CBProject("geometry")
            layer = direct.add_layer("Geometry")
            rectangle = direct.add_rect(
                layer, corner=(0, 0), width=20, height=10, identifier="outline")
            circle = direct.add_circle(
                layer, center=(5, 5), diameter=4, identifier="circle", elevation=1)
            pline = direct.add_pline(
                layer, identifier="shape", closed=False,
                points=[Vertex(0, 0, 0), Vertex(10, 0, 0, bulge=0.5), Vertex(10, 10, 1)])
            arc = direct.add_arc(
                layer, center=(0, 0), radius=10, start_angle=30, extent_angle=120,
                identifier="arc", elevation=-2)
            marks = direct.add_points(
                layer, identifier="marks",
                points=[Vertex(1, 2, 3), Vertex(4, 5, 6)])
            for entity, (dx, dy) in (
                (rectangle, (5, 2)), (circle, (2, 3)), (pline, (-1, 0)),
                (arc, (10, -1)), (marks, (0.5, 0.5)),
            ):
                self.assertTrue(direct.translate_primitive(entity, dx, dy, bake=False))
            direct.save(str(self.root / "direct-A.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "A.cb"),
                self.xml_semantics(self.root / "direct-A.cb"),
            )
            self.assertTrue(direct.translate_primitive(pline, 4, 0, bake=False))
            direct.save(str(self.root / "direct-B.cb"))
            await self.call("document_save", self.args(
                document=reopened, expected_revision=1, path="B.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "B.cb"),
                self.xml_semantics(self.root / "direct-B.cb"),
            )

        self.run_async(test)

    def test_negative_inputs_conflicts_and_atomic_failures(self):
        async def test():
            handle = await self.create()
            complete_inputs = (
                ("geometry_add_circle", {
                    "identifier": "circle", "layer": "Geometry",
                    "x": 5, "y": 5, "diameter": 4,
                }),
                ("geometry_add_arc", {
                    "identifier": "arc", "layer": "Geometry",
                    "x": 0, "y": 0, "radius": 10, "start_angle": 30,
                    "extent_angle": 120,
                }),
                ("geometry_add_pline", {
                    "identifier": "shape", "layer": "Geometry",
                    "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
                }),
                ("geometry_add_points", {
                    "identifier": "marks", "layer": "Geometry",
                    "points": [{"x": 1, "y": 2}],
                }),
            )
            for tool, complete in complete_inputs:
                malformed = await self.call(tool, {})
                self.assertEqual(malformed["error"]["code"], "INVALID_ARGUMENT")
                for field in complete:
                    incomplete = {key: value for key, value in complete.items()
                                  if key != field}
                    result = await self.call(tool, self.args(
                        document=handle, expected_revision=0, **incomplete))
                    self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT",
                                     (tool, field, result))

            for tool, bad in (
                ("geometry_add_circle", {"diameter": 0}),
                ("geometry_add_circle", {"diameter": -2}),
                ("geometry_add_arc", {"radius": 0}),
                ("geometry_add_arc", {"extent_angle": "90"}),
                ("geometry_add_arc", {"start_angle": True}),
                ("geometry_add_pline", {"points": [{"x": 0, "y": 0}]}),
                ("geometry_add_pline", {"points": [{"x": 0, "y": 0, "bulge": True}]}),
                ("geometry_add_pline", {
                    "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}], "closed": "yes"}),
                ("geometry_add_points", {"points": [{"x": 1, "y": 2, "bulge": 0}]}),
                ("geometry_add_points", {"points": []}),
            ):
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=0,
                    identifier="broken", layer="Geometry", **bad))
                self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT",
                                 (tool, bad, result))
            self.assertEqual(
                (await self.inspect(handle))["data"]["summary"]["counts"],
                {"layers": 0, "parts": 0, "primitives": 0, "mops": 0},
            )

            rectangle = await self.call("geometry_add_rectangle", self.args(
                document=handle, expected_revision=0, identifier="outline",
                layer="Geometry", x=0, y=0, width=20, height=10))
            self.assertTrue(rectangle["ok"], rectangle)
            rectangle_id = rectangle["data"]["entity_id"]

            duplicate = await self.call("geometry_add_circle", self.args(
                document=handle, expected_revision=1, identifier="outline",
                layer="Geometry", x=5, y=5, diameter=4))
            self.assertEqual(duplicate["error"]["code"], "IDENTIFIER_CONFLICT")
            same_as_layer = await self.call("geometry_add_circle", self.args(
                document=handle, expected_revision=1, identifier="Rims",
                layer="Rims", x=5, y=5, diameter=4))
            self.assertEqual(same_as_layer["error"]["code"], "IDENTIFIER_CONFLICT")
            layer_taken = await self.call("geometry_add_circle", self.args(
                document=handle, expected_revision=1, identifier="rims",
                layer="outline", x=5, y=5, diameter=4))
            self.assertEqual(layer_taken["error"]["code"], "IDENTIFIER_CONFLICT")
            self.assertEqual(
                (await self.inspect(handle))["data"]["summary"]["counts"],
                {"layers": 1, "parts": 0, "primitives": 1, "mops": 0},
            )

            circle = await self.call("geometry_add_circle", self.args(
                document=handle, expected_revision=1, identifier="circle",
                layer="Geometry", x=5, y=5, diameter=4))
            self.assertTrue(circle["ok"], circle)
            circle_id = circle["data"]["entity_id"]

            stale = await self.call("geometry_add_points", self.args(
                document=handle, expected_revision=1, identifier="marks",
                layer="Geometry", points=[{"x": 1, "y": 2}]))
            self.assertEqual(stale["error"]["code"], "STALE_REVISION")

            profile = await self.call("machining_add_profile", self.args(
                document=handle, expected_revision=2, identifier="profile",
                part="Part", targets=[rectangle_id], side="Outside",
                target_depth=-1, depth_increment=0.5, tool_diameter=3,
                cut_feedrate=300, plunge_feedrate=100, spindle_speed=12000,
                clearance_plane=5))
            self.assertTrue(profile["ok"], profile)
            mop_id = profile["data"]["mop_id"]

            wrong_kind_target = await self.call("machining_add_profile", self.args(
                document=handle, expected_revision=3, identifier="circle-profile",
                part="Part", targets=[circle_id], side="Outside",
                target_depth=-1, depth_increment=0.5, tool_diameter=3,
                cut_feedrate=300, plunge_feedrate=100, spindle_speed=12000,
                clearance_plane=5))
            self.assertEqual(wrong_kind_target["error"]["code"], "UNSUPPORTED_OPERATION")

            missing_translate = await self.call("geometry_translate", self.args(
                document=handle, expected_revision=3, entity_id=str(uuid4()),
                dx=1, dy=0))
            self.assertEqual(missing_translate["error"]["code"], "ENTITY_NOT_FOUND")
            mop_translate = await self.call("geometry_translate", self.args(
                document=handle, expected_revision=3, entity_id=mop_id, dx=1, dy=0))
            self.assertEqual(mop_translate["error"]["code"], "UNSUPPORTED_OPERATION")
            self.assertEqual(
                (await self.inspect(handle, revision=3))["data"]["summary"]["counts"],
                {"layers": 1, "parts": 1, "primitives": 2, "mops": 1},
            )

            requests = [
                self.args(document=handle, expected_revision=3, identifier="first",
                          layer="Geometry", x=0, y=0, diameter=2),
                self.args(document=handle, expected_revision=3, identifier="second",
                          layer="Geometry", x=0, y=0, diameter=2),
            ]
            results = []

            async def add(arguments):
                results.append(await self.call("geometry_add_circle", arguments))

            async with anyio.create_task_group() as group:
                for arguments in requests:
                    group.start_soon(add, arguments)
            self.assertEqual(sum(result["ok"] for result in results), 1)
            self.assertEqual(
                sum(not result["ok"] and result["error"]["code"] == "STALE_REVISION"
                    for result in results),
                1,
            )

        self.run_async(test)

    def test_imported_out_of_slice_shapes_stay_diagnostic(self):
        async def test():
            source = CBProject("mixed")
            layer = source.add_layer("Geometry")
            plain = source.add_circle(layer, center=(5, 5), diameter=4,
                                      identifier="plain")
            stretched = source.add_circle(layer, center=(5, 5), diameter=4,
                                          identifier="stretched")
            self.assertTrue(source.scale_primitive(stretched, 2, 1, bake=False))
            source.add_circle(layer, center=(1, 1), diameter=2, identifier="nested",
                              parent=stretched)
            source.add_pline(
                layer, identifier="oversized-pline", closed=False,
                points=[Vertex(index, 0) for index in range(10001)],
            )
            note = source.add_text(layer, text="note", position=(0, 0), height=3,
                                   identifier="note")
            self.assertTrue(source.scale_primitive(note, 2, 1, bake=False))
            source.save(str(self.root / "mixed.cb"))

            opened = await self.call("document_open", self.args(path="mixed.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            records = await self.inspect_records(opened["document"], revision=0)
            primitives = {record["identifier"]: record for record in records
                          if record["kind"] == "primitive"}
            self.assertEqual(primitives["plain"]["geometry"], {
                "kind": "circle", "center": [5.0, 5.0, 0.0], "diameter": 4.0,
                "bounds": [3.0, 3.0, 7.0, 7.0],
            })
            self.assertIsNone(primitives["stretched"]["geometry"])
            self.assertIsNone(primitives["nested"]["geometry"])
            self.assertIsNone(primitives["oversized-pline"]["geometry"])
            self.assertIsNone(primitives["note"]["geometry"])
            self.assertEqual(primitives["nested"]["parent"],
                             primitives["stretched"]["id"])
            self.assertIn(
                "INSPECTION_UNSUPPORTED",
                self.diagnostics(await self.inspect(opened["document"])))

            rejected = await self.call("machining_add_profile", self.args(
                document=opened["document"], expected_revision=0,
                identifier="stretched-profile", part="Part",
                targets=[primitives["stretched"]["id"]], side="Outside",
                target_depth=-1, depth_increment=0.5, tool_diameter=3,
                cut_feedrate=300, plunge_feedrate=100, spindle_speed=12000,
                clearance_plane=5))
            self.assertEqual(rejected["error"]["code"], "UNSUPPORTED_OPERATION")

        self.run_async(test)


if __name__ == "__main__":
    unittest.main()
