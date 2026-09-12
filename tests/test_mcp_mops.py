"""4d third-batch MCP machining breadth acceptance.

Pocket, Engrave and Drill authoring plus explicit MOP target replacement
through the adapter is compared with independently authored public-framework
projects, including closed parameter records, per-MOP-kind target rules,
save/reopen retention and the negative/resilience cases required before
advertisement.
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


@unittest.skipIf(DocumentService is None, "Install .[mcp] for adapter checks")
class MopBreadthTests(unittest.TestCase):
    """DocumentService contract tests for the Pocket/Engrave/Drill family."""

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

    async def create(self, name="mops"):
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
                MopBreadthTests.assert_records_close(testcase, first, second, places)
        elif isinstance(got, dict) and isinstance(want, dict):
            testcase.assertEqual(set(got), set(want))
            for key in want:
                MopBreadthTests.assert_records_close(testcase, got[key], want[key], places)
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
    def mop_arguments(**overrides):
        base = {
            "target_depth": -1,
            "depth_increment": 0.5,
            "tool_diameter": 3,
            "cut_feedrate": 300,
            "plunge_feedrate": 100,
            "spindle_speed": 12000,
            "clearance_plane": 5,
        }
        base.update(overrides)
        return base

    @staticmethod
    def direct_project():
        """Build the same machining slice through the public framework API."""
        project = CBProject("mops")
        layer = project.add_layer("Geometry")
        rectangle = project.add_rect(layer, corner=(0, 0), width=20, height=10,
                                     identifier="outline")
        circle = project.add_circle(layer, center=(30, 5), diameter=6,
                                    identifier="circle")
        arc = project.add_arc(layer, center=(45, 0), radius=8, start_angle=20,
                              extent_angle=140, identifier="arc")
        pline = project.add_pline(layer, identifier="path", closed=False,
                                  points=[Vertex(0, 20), Vertex(15, 20, 1),
                                          Vertex(25, 30, 1, bulge=0.3)])
        marks = project.add_points(layer, identifier="marks",
                                   points=[Vertex(3, 3, 1), Vertex(9, 4)])
        part = project.add_part(
            "Part", enabled=True, stock_width=0, stock_height=0,
            stock_thickness=0, stock_material="")
        pocket = project.add_pocket_mop(
            part, targets=[rectangle, circle], identifier="pocket", name="pocket",
            enabled=True, target_depth=-1, depth_increment=0.5, stock_surface=0,
            roughing_clearance=0.0, clearance_plane=5, spindle_direction="CW",
            spindle_speed=12000, velocity_mode="ExactStop", work_plane="XY",
            optimisation_mode="Standard", tool_diameter=3, tool_number=0,
            tool_profile="EndMill", plunge_feedrate=100, cut_feedrate=300,
            max_crossover_distance=0.7, custom_mop_header="", custom_mop_footer="",
            stepover=0.4, stepover_feedrate="Plunge Feedrate",
            milling_direction="Conventional", collision_detection=True,
            lead_in_type="Spiral", lead_in_spiral_angle=30.0,
            final_depth_increment=0.0, cut_ordering="DepthFirst",
            region_fill_style="InsideOutsideOffsets", finish_stepover=0.0,
            finish_stepover_at_target_depth=False, roughing_finishing="Roughing")
        engrave = project.add_engrave_mop(
            part, targets=[arc, pline], identifier="engrave", name="engrave",
            enabled=True, target_depth=-1, depth_increment=0.5, stock_surface=0,
            roughing_clearance=0.0, clearance_plane=5, spindle_direction="CW",
            spindle_speed=12000, velocity_mode="ExactStop", work_plane="XY",
            optimisation_mode="Standard", tool_diameter=3, tool_number=0,
            tool_profile="EndMill", plunge_feedrate=100, cut_feedrate=300,
            max_crossover_distance=0.7, custom_mop_header="", custom_mop_footer="",
            roughing_finishing="Roughing", final_depth_increment=0.0,
            cut_ordering="DepthFirst")
        drill = project.add_drill_mop(
            part, targets=[marks, circle], identifier="drill", name="drill",
            enabled=True, target_depth=-2, depth_increment=1, stock_surface=0,
            roughing_clearance=0.0, clearance_plane=5, spindle_direction="CW",
            spindle_speed=9000, velocity_mode="ExactStop", work_plane="XY",
            optimisation_mode="Standard", tool_diameter=3.175, tool_number=0,
            tool_profile="Drill", plunge_feedrate=80, cut_feedrate=250,
            max_crossover_distance=0.7, custom_mop_header="", custom_mop_footer="",
            drilling_method="CannedCycle", peck_distance=2, retract_height=6,
            dwell=50, hole_diameter=None, drill_lead_out=False,
            spiral_flat_base=True, lead_out_length=0.0, custom_script="")
        return project, pocket, engrave, drill

    @staticmethod
    def expected_pocket_parameters():
        return {
            "target_depth": -1, "depth_increment": 0.5, "tool_diameter": 3,
            "cut_feedrate": 300, "plunge_feedrate": 100, "spindle_speed": 12000,
            "stock_surface": 0, "clearance_plane": 5, "enabled": True,
            "work_plane": "XY", "tool_profile": "EndMill",
            "spindle_direction": "CW", "velocity_mode": "ExactStop",
            "roughing_clearance": 0, "tool_number": 0,
            "final_depth_increment": 0, "cut_ordering": "DepthFirst",
            "custom_mop_header": "", "custom_mop_footer": "",
            "stepover": 0.4, "stepover_feedrate": "Plunge Feedrate",
            "milling_direction": "Conventional", "collision_detection": True,
            "lead_in_type": "Spiral",
            "region_fill_style": "InsideOutsideOffsets",
            "finish_stepover": 0, "finish_stepover_at_target_depth": False,
            "roughing_finishing": "Roughing",
        }

    @staticmethod
    def expected_engrave_parameters():
        return {
            "target_depth": -1, "depth_increment": 0.5, "tool_diameter": 3,
            "cut_feedrate": 300, "plunge_feedrate": 100, "spindle_speed": 12000,
            "stock_surface": 0, "clearance_plane": 5, "enabled": True,
            "work_plane": "XY", "tool_profile": "EndMill",
            "spindle_direction": "CW", "velocity_mode": "ExactStop",
            "roughing_clearance": 0, "tool_number": 0,
            "custom_mop_header": "", "custom_mop_footer": "",
            "roughing_finishing": "Roughing", "final_depth_increment": 0,
            "cut_ordering": "DepthFirst",
        }

    @staticmethod
    def expected_drill_parameters():
        return {
            "target_depth": -2, "depth_increment": 1, "tool_diameter": 3.175,
            "cut_feedrate": 250, "plunge_feedrate": 80, "spindle_speed": 9000,
            "stock_surface": 0, "clearance_plane": 5, "enabled": True,
            "peck_distance": 2, "retract_height": 6, "dwell": 50,
            "drilling_method": "CannedCycle", "tool_profile": "Drill",
            "work_plane": "XY", "spindle_direction": "CW",
            "velocity_mode": "ExactStop", "roughing_clearance": 0,
            "tool_number": 0, "custom_mop_header": "", "custom_mop_footer": "",
        }

    def test_add_pocket_engrave_drill_targets_roundtrip_and_parity(self):
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
                self.assertEqual(result["revision"], revision)
                return result

            for tool, arguments in (
                ("geometry_add_rectangle", {
                    "identifier": "outline", "layer": "Geometry",
                    "x": 0, "y": 0, "width": 20, "height": 10}),
                ("geometry_add_circle", {
                    "identifier": "circle", "layer": "Geometry",
                    "x": 30, "y": 5, "diameter": 6}),
                ("geometry_add_arc", {
                    "identifier": "arc", "layer": "Geometry", "x": 45, "y": 0,
                    "radius": 8, "start_angle": 20, "extent_angle": 140}),
                ("geometry_add_pline", {
                    "identifier": "path", "layer": "Geometry", "closed": False,
                    "points": [{"x": 0, "y": 20}, {"x": 15, "y": 20, "z": 1},
                               {"x": 25, "y": 30, "z": 1, "bulge": 0.3}]}),
                ("geometry_add_points", {
                    "identifier": "marks", "layer": "Geometry",
                    "points": [{"x": 3, "y": 3, "z": 1}, {"x": 9, "y": 4}]}),
            ):
                result = await add(tool, **arguments)
                ids[arguments["identifier"]] = result["data"]["entity_id"]

            pocket = await add("machining_add_pocket", **self.mop_arguments(
                identifier="pocket", part="Part",
                targets=[ids["outline"], ids["circle"]]))
            engrave = await add("machining_add_engrave", **self.mop_arguments(
                identifier="engrave", part="Part",
                targets=[ids["arc"], ids["path"]]))
            drill = await add("machining_add_drill", **self.mop_arguments(
                identifier="drill", part="Part",
                targets=[ids["marks"], ids["circle"]],
                target_depth=-2, depth_increment=1, tool_diameter=3.175,
                cut_feedrate=250, plunge_feedrate=80, spindle_speed=9000,
                peck_distance=2, retract_height=6, dwell=50))
            mop_ids = {"pocket": pocket["data"]["mop_id"],
                       "engrave": engrave["data"]["mop_id"],
                       "drill": drill["data"]["mop_id"]}
            for result in (pocket, engrave, drill):
                self.assertEqual(result["data"]["part"], "Part")
            self.assertEqual(set(pocket["data"]["targets"]),
                             {ids["outline"], ids["circle"]})
            self.assertEqual(set(engrave["data"]["targets"]),
                             {ids["arc"], ids["path"]})
            self.assertEqual(set(drill["data"]["targets"]),
                             {ids["marks"], ids["circle"]})

            retarget = await add("machining_set_mop_targets",
                                 mop_id=mop_ids["pocket"],
                                 targets=[ids["circle"], ids["outline"]])
            self.assertEqual(set(retarget["data"]["targets"]),
                             {ids["outline"], ids["circle"]})
            retarget_engrave = await add("machining_set_mop_targets",
                                         mop_id=mop_ids["engrave"],
                                         targets=[ids["path"]])
            self.assertEqual(retarget_engrave["data"]["targets"], [ids["path"]])

            records = await self.inspect_records(handle, revision=revision)
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED", self.diagnostics(await self.inspect(handle)))
            mops = {record["identifier"]: record for record in records
                    if record["kind"] == "mop"}
            self.assertEqual(mops["pocket"]["parameters"],
                             self.expected_pocket_parameters())
            self.assertEqual(mops["engrave"]["parameters"],
                             self.expected_engrave_parameters())
            self.assertEqual(mops["drill"]["parameters"],
                             self.expected_drill_parameters())

            saved_a = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="A.cb"))
            self.assertTrue(saved_a["ok"], saved_a)
            bytes_a = (self.root / "A.cb").read_bytes()
            self.assertEqual(saved_a["data"]["sha256"], hashlib.sha256(bytes_a).hexdigest())

            opened = await self.call("document_open", self.args(path="A.cb", units="mm"))
            self.assertTrue(opened["ok"], opened)
            reopened_records = await self.inspect_records(opened["document"], revision=0)
            self.assert_records_close(self, reopened_records, records)
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED",
                self.diagnostics(await self.inspect(opened["document"])))

            direct, _, _, _ = self.direct_project()
            direct.set_mop_targets("pocket", ["circle", "outline"])
            direct.set_mop_targets("engrave", ["path"])
            direct.save(str(self.root / "direct-A.cb"))
            self.assertEqual(
                self.xml_semantics(self.root / "A.cb"),
                self.xml_semantics(self.root / "direct-A.cb"),
            )

        self.run_async(test)

    def test_target_kind_rules_negatives_retries_and_atomic_failure(self):
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

            rectangle = (await add("geometry_add_rectangle", identifier="outline",
                                   layer="Geometry", x=0, y=0, width=20, height=10))
            circle = (await add("geometry_add_circle", identifier="circle",
                                layer="Geometry", x=30, y=5, diameter=6))
            open_path = (await add("geometry_add_pline", identifier="path",
                                   layer="Geometry", closed=False,
                                   points=[{"x": 0, "y": 20}, {"x": 15, "y": 20}]))
            closed_path = (await add("geometry_add_pline", identifier="closed-path",
                                     layer="Geometry", closed=True,
                                     points=[{"x": 0, "y": 30}, {"x": 15, "y": 30},
                                             {"x": 10, "y": 40}]))
            region = (await add(
                "geometry_add_region", identifier="region", layer="Geometry",
                outer={"points": [{"x": 40, "y": 20}, {"x": 50, "y": 20},
                                   {"x": 50, "y": 30}, {"x": 40, "y": 30}]},
            ))
            marks = (await add("geometry_add_points", identifier="marks",
                               layer="Geometry",
                               points=[{"x": 3, "y": 3}, {"x": 9, "y": 4}]))
            rectangle_id = rectangle["data"]["entity_id"]
            circle_id = circle["data"]["entity_id"]
            open_path_id = open_path["data"]["entity_id"]
            closed_path_id = closed_path["data"]["entity_id"]
            region_id = region["data"]["entity_id"]
            marks_id = marks["data"]["entity_id"]

            pocket = (await add("machining_add_pocket", **self.mop_arguments(
                identifier="pocket", part="Part", targets=[rectangle_id])))
            mop_id = pocket["data"]["mop_id"]

            profile = await add(
                "machining_add_profile", identifier="profile", part="Part",
                targets=[circle_id, closed_path_id, region_id], side="Inside",
                **self.mop_arguments(),
            )
            self.assertEqual(
                set(profile["data"]["targets"]),
                {circle_id, closed_path_id, region_id},
            )

            for tool, target_id, expected_code in (
                ("machining_add_pocket", open_path_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_pocket", marks_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_engrave", marks_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_drill", rectangle_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_drill", open_path_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_profile", open_path_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_profile", marks_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_pocket", str(uuid4()), "ENTITY_NOT_FOUND"),
            ):
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision,
                    identifier="probe", part="Part",
                    targets=[target_id],
                    **self.mop_arguments(),
                    **({"side": "Outside"} if tool == "machining_add_profile" else {})))
                self.assertEqual(result["error"]["code"], expected_code, (tool, result))

            engrave = (await add("machining_add_engrave", **self.mop_arguments(
                identifier="engrave", part="Part", targets=[rectangle_id])))
            engrave_id = engrave["data"]["mop_id"]

            for tool, bad in (
                ("machining_add_pocket", {"target_depth": 0}),
                ("machining_add_engrave", {"target_depth": 1}),
                ("machining_add_drill", {"target_depth": 0.5}),
                ("machining_add_drill", {"peck_distance": -1}),
                ("machining_add_drill", {"dwell": -0.5}),
                ("machining_add_drill", {"retract_height": "high"}),
                ("machining_add_drill", {"peck_distance": True}),
                ("machining_add_engrave", {"spindle_speed": 0}),
                ("machining_add_pocket", {"depth_increment": 0}),
            ):
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision,
                    identifier="broken", part="Part",
                    targets=[rectangle_id],
                    **self.mop_arguments(**bad)))
                self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT",
                                 (tool, bad, result))

            for tool, arguments in (
                ("machining_set_mop_targets", {
                    "mop_id": str(uuid4()), "targets": [rectangle_id]}),
                ("machining_set_mop_targets", {
                    "mop_id": rectangle_id, "targets": [rectangle_id]}),
                ("machining_set_mop_targets", {
                    "mop_id": mop_id, "targets": [marks_id]}),
                ("machining_set_mop_targets", {
                    "mop_id": engrave_id, "targets": [str(uuid4())]}),
            ):
                result = await self.call("machining_set_mop_targets", self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertIn(result["error"]["code"],
                              ("ENTITY_NOT_FOUND", "UNSUPPORTED_OPERATION"),
                              (arguments, result))
            duplicates = await self.call("machining_set_mop_targets", self.args(
                document=handle, expected_revision=revision, mop_id=mop_id,
                targets=[rectangle_id] * 2))
            self.assertEqual(duplicates["error"]["code"], "INVALID_ARGUMENT")

            set_targets = self.args(document=handle, expected_revision=revision,
                                    mop_id=engrave_id, targets=[rectangle_id])
            retargeted = await self.call("machining_set_mop_targets", set_targets)
            self.assertTrue(retargeted["ok"], retargeted)
            revision += 1
            self.assertEqual(await self.call("machining_set_mop_targets", set_targets), {
                **retargeted, "replayed": True,
            })

            stale = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=0,
                identifier="stale", part="Part",
                targets=[marks_id], **self.mop_arguments()))
            self.assertEqual(stale["error"]["code"], "STALE_REVISION")

            requests = [
                self.args(document=handle, expected_revision=revision,
                          identifier="first", part="Part",
                          targets=[circle_id], **self.mop_arguments()),
                self.args(document=handle, expected_revision=revision,
                          identifier="second", part="Part",
                          targets=[circle_id], **self.mop_arguments()),
            ]
            results = []

            async def pocket_add(arguments):
                results.append(await self.call("machining_add_pocket", arguments))

            async with anyio.create_task_group() as group:
                for arguments in requests:
                    group.start_soon(pocket_add, arguments)
            self.assertEqual(sum(result["ok"] for result in results), 1)
            self.assertEqual(
                sum(not result["ok"] and result["error"]["code"] == "STALE_REVISION"
                    for result in results),
                1,
            )
            final = await self.inspect(handle)
            self.assertEqual(
                final["data"]["summary"]["counts"],
                {"layers": 1, "parts": 1, "primitives": 6, "mops": 4},
            )
            self.assertNotIn("INSPECTION_UNSUPPORTED", self.diagnostics(final))
            profile_record = next(
                record for record in await self.inspect_records(handle)
                if record["kind"] == "mop" and record["identifier"] == "profile"
            )
            self.assertEqual(
                set(profile_record["targets"]),
                {circle_id, closed_path_id, region_id},
            )

            saved = await self.call("document_save", self.args(
                document=handle, expected_revision=final["revision"],
                path="profile-closed-contours.cb",
            ))
            self.assertTrue(saved["ok"], saved)
            self.assertIn("SERVER_WORKSPACE_ONLY", self.diagnostics(saved))
            reopened = await self.call("document_open", self.args(
                path="profile-closed-contours.cb", units="mm",
            ))
            self.assertTrue(reopened["ok"], reopened)
            reopened_profile = next(
                record for record in await self.inspect_records(reopened["document"])
                if record["kind"] == "mop" and record["identifier"] == "profile"
            )
            self.assertEqual(
                set(reopened_profile["targets"]),
                {circle_id, closed_path_id, region_id},
            )

        self.run_async(test)


if __name__ == "__main__":
    unittest.main()
