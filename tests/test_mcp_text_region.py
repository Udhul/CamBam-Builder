"""4d second-batch MCP Text and Region breadth acceptance.

Text and Region authoring through the adapter is compared with independently
authored public-framework projects, including typed world geometry, exact
bulge-aware Region bounds, XY topology rejection, save/reopen identity
retention and the negative/resilience cases required before advertisement.
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
from cambam_builder.cambam_entities import Pline, Vertex


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


def square_outer(bulge=0.0):
    return [
        Vertex(0, 0, 0, bulge=bulge), Vertex(10, 0), Vertex(10, 10), Vertex(0, 10),
    ]


def triangle_hole():
    return [Vertex(4, 4), Vertex(6, 4), Vertex(5, 6)]


@unittest.skipIf(DocumentService is None, "Install .[mcp] for adapter checks")
class TextRegionTests(unittest.TestCase):
    """DocumentService contract tests for the Text/Region family."""

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

    async def create(self, name="shapes"):
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
                TextRegionTests.assert_records_close(testcase, first, second, places)
        elif isinstance(got, dict) and isinstance(want, dict):
            testcase.assertEqual(set(got), set(want))
            for key in want:
                TextRegionTests.assert_records_close(testcase, got[key], want[key], places)
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
        project = CBProject("shapes")
        annotation = project.add_layer("Annotation")
        note = project.add_text(
            annotation, text="Hello\nCamBam", position=(10, 20), height=4,
            font="Consolas", style="bold", line_spacing=1.5,
            align_horizontal="left", align_vertical="top",
            identifier="note", elevation=2,
        )
        regions = project.add_layer("Regions")
        plate = project.add_region(
            regions,
            outer_curve=Pline(vertices=square_outer(), closed=True),
            hole_curves=[Pline(vertices=triangle_hole(), closed=True)],
            identifier="plate",
        )
        curved = project.add_region(
            regions,
            outer_curve=Pline(vertices=square_outer(bulge=0.2), closed=True),
            identifier="curved_plate",
        )
        return project, note, plate, curved

    @staticmethod
    def expected_geometries(note, plate, curved):
        """Expected typed payloads from independent public-framework queries."""
        note_world = note.get_absolute_coordinates_xyz()

        def contour_payload(points):
            return {
                "world_xyz": [[float(value) for value in point[:3]] for point in points],
                "bulges": [float(point[3]) for point in points],
            }

        plate_world = plate.get_absolute_coordinates_xyz()
        curved_world = curved.get_absolute_coordinates_xyz()
        return {
            "note": {
                "kind": "text",
                "text": "Hello\nCamBam",
                "anchor": [float(value) for value in note_world["position"]],
                "height": float(note_world["height"]),
                "font": "Consolas",
                "style": "bold",
                "line_spacing": 1.5,
                "align_horizontal": "left",
                "align_vertical": "top",
                "p2": None,
            },
            "plate": {
                "kind": "region",
                "outer_curve": contour_payload(plate_world["outer_curve"]),
                "hole_curves": [contour_payload(contour)
                                for contour in plate_world["hole_curves"]],
                "bounds": bounds_of(plate),
            },
            "curved_plate": {
                "kind": "region",
                "outer_curve": contour_payload(curved_world["outer_curve"]),
                "hole_curves": [contour_payload(contour)
                                for contour in curved_world["hole_curves"]],
                "bounds": bounds_of(curved),
            },
        }

    def test_expected_payloads_match_analytic_values(self):
        """Guard the direct-query parity source against hand-derived geometry."""
        _, note, plate, curved = self.direct_project()
        expected = self.expected_geometries(note, plate, curved)
        self.assertEqual(expected["note"]["anchor"], [10.0, 20.0, 2.0])
        self.assertEqual(expected["note"]["height"], 4.0)
        self.assertEqual(expected["note"]["text"], "Hello\nCamBam")
        self.assertEqual(expected["plate"]["outer_curve"]["world_xyz"],
                         [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                          [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]])
        self.assertEqual(expected["plate"]["outer_curve"]["bulges"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(expected["plate"]["hole_curves"], [{
            "world_xyz": [[4.0, 4.0, 0.0], [6.0, 4.0, 0.0], [5.0, 6.0, 0.0]],
            "bulges": [0.0, 0.0, 0.0],
        }])
        self.assertEqual(expected["plate"]["bounds"], [0.0, 0.0, 10.0, 10.0])
        self.assertEqual(expected["curved_plate"]["outer_curve"]["bulges"], [0.2, 0.0, 0.0, 0.0])
        # The 0.2 bulge on the bottom segment sweeps 4*atan(0.2) degrees around
        # center (5,12) with radius 13, dipping exactly to y=-1; endpoint-only
        # bounds would wrongly stop at 0.
        self.assertEqual(expected["curved_plate"]["bounds"], [0.0, -1.0, 10.0, 10.0])

    def test_add_text_region_roundtrip_and_direct_parity(self):
        async def test():
            handle = await self.create()
            revision = 0
            ids = {}
            for tool, arguments in (
                ("geometry_add_text", {
                    "identifier": "note", "layer": "Annotation",
                    "text": "Hello\nCamBam", "x": 10, "y": 20,
                    "height": 4, "font": "Consolas", "style": "bold",
                    "line_spacing": 1.5, "align_horizontal": "left",
                    "align_vertical": "top", "z": 2,
                }),
                ("geometry_add_region", {
                    "identifier": "plate", "layer": "Regions",
                    "outer": {"points": [{"x": 0, "y": 0}, {"x": 10, "y": 0},
                                         {"x": 10, "y": 10}, {"x": 0, "y": 10}]},
                    "holes": [{"points": [{"x": 4, "y": 4}, {"x": 6, "y": 4},
                                          {"x": 5, "y": 6}]}],
                }),
                ("geometry_add_region", {
                    "identifier": "curved_plate", "layer": "Regions",
                    "outer": {"points": [{"x": 0, "y": 0, "bulge": 0.2},
                                         {"x": 10, "y": 0}, {"x": 10, "y": 10},
                                         {"x": 0, "y": 10}]},
                }),
            ):
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                self.assertEqual(result["revision"], revision)
                self.assertEqual(result["data"]["layer"], arguments["layer"])
                ids[arguments["identifier"]] = result["data"]["entity_id"]

            records = await self.inspect_records(handle, revision=revision)
            self.assertEqual(
                [record["kind"] for record in records],
                ["layer", "layer", "primitive", "primitive", "primitive"],
            )
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED", self.diagnostics(await self.inspect(handle)))
            by_identifier = {record["identifier"]: record for record in records[2:]}
            expected = self.expected_geometries(*self.direct_project()[1:])
            for name, identifier in (("note", "note"), ("plate", "plate"),
                                     ("curved_plate", "curved_plate")):
                record = by_identifier[identifier]
                self.assertEqual(record["id"], ids[identifier])
                self.assertEqual(record["geometry"], expected[name])
                self.assertIsNone(record["parent"])
                self.assertEqual(record["children"], [])

            saved_a = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="A.cb"))
            self.assertTrue(saved_a["ok"], saved_a)
            bytes_a = (self.root / "A.cb").read_bytes()
            self.assertEqual(saved_a["data"]["sha256"], hashlib.sha256(bytes_a).hexdigest())

            opened = await self.call("document_open", self.args(path="A.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            reopened_records = await self.inspect_records(opened["document"], revision=0)
            self.assert_records_close(self, reopened_records, records)

            direct, _, _, _ = self.direct_project()
            direct.save(str(self.root / "direct-A.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "A.cb"),
                self.xml_semantics(self.root / "direct-A.cb"),
            )

        self.run_async(test)

    def test_translate_text_region_with_reopen_parity(self):
        async def test():
            handle = await self.create()
            ids = {}
            revision = 0
            for tool, arguments in (
                ("geometry_add_text", {
                    "identifier": "note", "layer": "Annotation",
                    "text": "Hello\nCamBam", "x": 10, "y": 20,
                    "height": 4, "font": "Consolas", "style": "bold",
                    "line_spacing": 1.5, "align_horizontal": "left",
                    "align_vertical": "top", "z": 2,
                }),
                ("geometry_add_region", {
                    "identifier": "plate", "layer": "Regions",
                    "outer": {"points": [{"x": 0, "y": 0}, {"x": 10, "y": 0},
                                         {"x": 10, "y": 10}, {"x": 0, "y": 10}]},
                    "holes": [{"points": [{"x": 4, "y": 4}, {"x": 6, "y": 4},
                                          {"x": 5, "y": 6}]}],
                }),
                ("geometry_add_rectangle", {
                    "identifier": "outline", "layer": "Geometry",
                    "x": 0, "y": 0, "width": 20, "height": 10,
                }),
            ):
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                ids[arguments["identifier"]] = result["data"]["entity_id"]

            for identifier, (dx, dy) in (("note", (2, 3)), ("plate", (1, 1)),
                                         ("outline", (-1, -2))):
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
            self.assertEqual(by_identifier["note"]["geometry"]["anchor"], [12.0, 23.0, 2.0])
            self.assertEqual(by_identifier["plate"]["geometry"]["outer_curve"]["world_xyz"],
                             [[1.0, 1.0, 0.0], [11.0, 1.0, 0.0],
                              [11.0, 11.0, 0.0], [1.0, 11.0, 0.0]])
            self.assertEqual(by_identifier["plate"]["geometry"]["hole_curves"], [{
                "world_xyz": [[5.0, 5.0, 0.0], [7.0, 5.0, 0.0], [6.0, 7.0, 0.0]],
                "bulges": [0.0, 0.0, 0.0],
            }])
            self.assertEqual(by_identifier["plate"]["geometry"]["bounds"],
                             [1.0, 1.0, 11.0, 11.0])

            saved_a = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="A.cb"))
            self.assertTrue(saved_a["ok"], saved_a)
            opened = await self.call("document_open", self.args(path="A.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            reopened = opened["document"]
            reopened_records = await self.inspect_records(reopened, revision=0)
            self.assert_records_close(self, reopened_records, records)

            translate = self.args(document=reopened, expected_revision=0,
                                  entity_id=ids["plate"], dx=5, dy=0)
            translated = await self.call("geometry_translate", translate)
            self.assertTrue(translated["ok"], translated)
            self.assertEqual(translated["revision"], 1)
            self.assertEqual(await self.call("geometry_translate", translate), {
                **translated, "replayed": True,
            })
            after = await self.inspect_records(reopened, revision=1)
            moved = next(record for record in after
                         if record["kind"] == "primitive"
                         and record["identifier"] == "plate")
            self.assertEqual(moved["geometry"]["outer_curve"]["world_xyz"],
                             [[6.0, 1.0, 0.0], [16.0, 1.0, 0.0],
                              [16.0, 11.0, 0.0], [6.0, 11.0, 0.0]])
            self.assertEqual(moved["id"], ids["plate"])

            direct = CBProject("shapes")
            annotation = direct.add_layer("Annotation")
            note = direct.add_text(
                annotation, text="Hello\nCamBam", position=(10, 20), height=4,
                font="Consolas", style="bold", line_spacing=1.5,
                align_horizontal="left", align_vertical="top",
                identifier="note", elevation=2)
            regions = direct.add_layer("Regions")
            plate = direct.add_region(
                regions,
                outer_curve=Pline(vertices=square_outer(), closed=True),
                hole_curves=[Pline(vertices=triangle_hole(), closed=True)],
                identifier="plate")
            direct.add_rect("Geometry", corner=(0, 0), width=20, height=10,
                            identifier="outline")
            for entity, (dx, dy) in (
                (note, (2, 3)), (plate, (1, 1)),
                (direct.get_entity("outline"), (-1, -2)),
            ):
                self.assertTrue(direct.translate_primitive(entity, dx, dy, bake=False))
            direct.save(str(self.root / "direct-A.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "A.cb"),
                self.xml_semantics(self.root / "direct-A.cb"),
            )
            self.assertTrue(direct.translate_primitive(plate, 5, 0, bake=False))
            direct.save(str(self.root / "direct-B.cb"))
            await self.call("document_save", self.args(
                document=reopened, expected_revision=1, path="B.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "B.cb"),
                self.xml_semantics(self.root / "direct-B.cb"),
            )

        self.run_async(test)

    def test_negative_inputs_topology_conflicts_and_atomic_failures(self):
        async def test():
            handle = await self.create()

            text_cases = (
                {"text": ""},
                {"text": "   "},
                {"text": "bad\x01text"},
                {"text": "x" * 1025},
                {"height": 0},
                {"height": -1},
                {"line_spacing": 0},
                {"font": 4},
                {"font": ""},
                {"style": "bold\nitalic"},
                {"align_horizontal": "middle"},
                {"align_vertical": "middle"},
                {"align_horizontal": True},
                {"text": "note", "x": 0, "y": 0, "extra": 1},
            )
            complete_text = {
                "identifier": "note", "layer": "Annotation", "text": "note",
                "x": 0, "y": 0,
            }
            malformed = await self.call("geometry_add_text", {})
            self.assertEqual(malformed["error"]["code"], "INVALID_ARGUMENT")
            for override in text_cases:
                broken = dict(complete_text)
                broken.update(override)
                result = await self.call("geometry_add_text", self.args(
                    document=handle, expected_revision=0, **broken))
                self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT",
                                 (override, result))
            for field in complete_text:
                incomplete = {key: value for key, value in complete_text.items()
                              if key != field}
                result = await self.call("geometry_add_text", self.args(
                    document=handle, expected_revision=0, **incomplete))
                self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT",
                                 (field, result))

            good_contour = [{"x": 0, "y": 0}, {"x": 10, "y": 0},
                            {"x": 10, "y": 10}, {"x": 0, "y": 10}]
            region_cases = (
                ({"outer": {"points": good_contour[:1]}}, None),
                ({"outer": {"points": good_contour}}, {"points": good_contour[:1]}),
                ({"outer": {"points": [{"x": 0, "y": 0}, {"x": 10, "y": 0, "bulge": True}]}}, None),
                ({"outer": {"points": [
                    {"x": 0, "y": 0}, {"x": 10, "y": 10},
                    {"x": 10, "y": 0}, {"x": 0, "y": 10}]}}, None),
                ({"outer": {"points": good_contour},
                  "holes": [{"points": [{"x": 20, "y": 20}, {"x": 25, "y": 20},
                                        {"x": 22, "y": 25}]}]}, None),
                ({"outer": {"points": good_contour},
                  "holes": [{"points": [{"x": 2, "y": 2}, {"x": 4, "y": 2},
                                        {"x": 3, "y": 4}]},
                            {"points": [{"x": 3, "y": 3}, {"x": 5, "y": 3},
                                        {"x": 4, "y": 5}]}]}, None),
                ({"outer": {"points": good_contour},
                  "holes": [{"points": [{"x": 2, "y": 2}, {"x": 4, "y": 2},
                                        {"x": 3, "y": 4}]},
                            {"points": [{"x": 1, "y": 1}, {"x": 9, "y": 1},
                                        {"x": 5, "y": 9}]}]}, None),
                ({"outer": {"points": [{"x": 0, "y": 0}, {"x": 0, "y": 0},
                                       {"x": 10, "y": 10}, {"x": 0, "y": 10}]}}, None),
            )
            for override, holes_override in region_cases:
                broken = {
                    "identifier": "broken-region", "layer": "Regions",
                    "outer": {"points": good_contour},
                }
                broken.update(override)
                if holes_override is not None:
                    broken["holes"] = [{"points": holes_override}]
                result = await self.call("geometry_add_region", self.args(
                    document=handle, expected_revision=0, **broken))
                self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT",
                                 (override, holes_override, result))
            self.assertEqual(
                (await self.inspect(handle))["data"]["summary"]["counts"],
                {"layers": 0, "parts": 0, "primitives": 0, "mops": 0},
            )

            text = await self.call("geometry_add_text", self.args(
                document=handle, expected_revision=0, identifier="note",
                layer="Annotation", text="note", x=0, y=0))
            self.assertTrue(text["ok"], text)

            duplicate = await self.call("geometry_add_region", self.args(
                document=handle, expected_revision=1, identifier="note",
                layer="Regions", outer={"points": good_contour}))
            self.assertEqual(duplicate["error"]["code"], "IDENTIFIER_CONFLICT")
            same_as_layer = await self.call("geometry_add_region", self.args(
                document=handle, expected_revision=1, identifier="Plates",
                layer="Plates", outer={"points": good_contour}))
            self.assertEqual(same_as_layer["error"]["code"], "IDENTIFIER_CONFLICT")
            self.assertEqual(
                (await self.inspect(handle))["data"]["summary"]["counts"],
                {"layers": 1, "parts": 0, "primitives": 1, "mops": 0},
            )

            stale = await self.call("geometry_add_text", self.args(
                document=handle, expected_revision=0, identifier="second",
                layer="Annotation", text="second", x=1, y=1))
            self.assertEqual(stale["error"]["code"], "STALE_REVISION")

            missing_translate = await self.call("geometry_translate", self.args(
                document=handle, expected_revision=1, entity_id=str(uuid4()),
                dx=1, dy=0))
            self.assertEqual(missing_translate["error"]["code"], "ENTITY_NOT_FOUND")

        self.run_async(test)

    def test_imported_out_of_slice_text_region_stay_diagnostic(self):
        async def test():
            source = CBProject("mixed")
            annotation = source.add_layer("Annotation")
            plain_note = source.add_text(annotation, text="plain", position=(5, 5),
                                         height=3, identifier="plain_note")
            rotated_note = source.add_text(annotation, text="rotated", position=(5, 5),
                                           height=3, identifier="rotated_note")
            self.assertTrue(source.scale_primitive(rotated_note, 2, 1, bake=False))
            source.add_text(annotation, text="x" * 1025, position=(0, 0),
                            height=3, identifier="oversized_note")
            regions = source.add_layer("Regions")
            plate = source.add_region(
                regions, outer_curve=Pline(vertices=square_outer(), closed=True),
                identifier="plate")
            source.add_region(
                regions, outer_curve=Pline(vertices=square_outer(), closed=True),
                identifier="nested", parent=rotated_note)
            source.save(str(self.root / "mixed.cb"))

            opened = await self.call("document_open", self.args(path="mixed.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            records = await self.inspect_records(opened["document"], revision=0)
            primitives = {record["identifier"]: record for record in records
                          if record["kind"] == "primitive"}
            self.assertEqual(primitives["plain_note"]["geometry"]["kind"], "text")
            self.assertEqual(primitives["plain_note"]["geometry"]["text"], "plain")
            self.assertIsNone(primitives["plain_note"]["geometry"]["p2"])
            self.assertIsNone(primitives["rotated_note"]["geometry"])
            self.assertIsNone(primitives["oversized_note"]["geometry"])
            self.assertEqual(primitives["plate"]["geometry"]["kind"], "region")
            self.assertIsNone(primitives["nested"]["geometry"])
            self.assertEqual(primitives["nested"]["parent"],
                             primitives["rotated_note"]["id"])
            self.assertIn(
                "INSPECTION_UNSUPPORTED",
                self.diagnostics(await self.inspect(opened["document"])))

        self.run_async(test)


if __name__ == "__main__":
    unittest.main()
