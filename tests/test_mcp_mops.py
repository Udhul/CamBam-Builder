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
            "optimisation_mode": "Standard", "max_crossover_distance": 0.7,
            "roughing_clearance": 0, "tool_number": 0,
            "final_depth_increment": 0, "cut_ordering": "DepthFirst",
            "stepover": 0.4, "stepover_feedrate": "Plunge Feedrate",
            "milling_direction": "Conventional", "collision_detection": True,
            "lead_in_type": "Spiral", "lead_in_spiral_angle": 30,
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
            "optimisation_mode": "Standard", "max_crossover_distance": 0.7,
            "roughing_clearance": 0, "tool_number": 0,
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
            "optimisation_mode": "Standard", "max_crossover_distance": 0.7,
            "tool_number": 0,
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
            canned_xml = ET.parse(self.root / "A.cb").getroot().find(
                "./parts/part/machineops/drill")
            self.assertIsNotNone(canned_xml)
            self.assertIsNone(canned_xml.find("CustomScript"))
            for tag in ("PeckDistance", "RetractHeight", "Dwell"):
                self.assertEqual("Value", canned_xml.find(tag).get("state"), tag)

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

    def test_missing_revision_reports_field_and_does_not_implicate_region_target(self):
        async def test():
            handle = await self.create("region-pocket-validation")
            region = await self.call("geometry_add_region", self.args(
                document=handle, expected_revision=0, identifier="region",
                layer="Geometry",
                outer={"points": [
                    {"x": 0, "y": 0}, {"x": 40, "y": 0},
                    {"x": 40, "y": 20}, {"x": 0, "y": 20},
                ]},
                holes=[{"points": [
                    {"x": 20, "y": 10}, {"x": 31, "y": 10},
                    {"x": 31, "y": 14}, {"x": 20, "y": 14},
                ]}],
            ))
            self.assertTrue(region["ok"], region)
            pocket_arguments = {
                "workspace_id": self.service.workspace.id,
                "document": handle,
                "request_id": str(uuid4()),
                "identifier": "pocket",
                "part": "Part",
                "targets": [region["data"]["entity_id"]],
                **self.mop_arguments(target_depth=-2, depth_increment=1,
                                     clearance_plane=3),
            }

            missing_revision = await self.call(
                "machining_add_pocket", pocket_arguments)
            self.assertFalse(missing_revision["ok"], missing_revision)
            self.assertEqual(missing_revision["error"]["code"], "INVALID_ARGUMENT")
            self.assertEqual(missing_revision["error"]["field"], "expected_revision")
            self.assertEqual(missing_revision["error"]["message"],
                             "Missing required field: expected_revision")
            self.assertEqual(missing_revision["revision"], 1)

            pocket = await self.call("machining_add_pocket", {
                **pocket_arguments,
                "expected_revision": 1,
                "request_id": str(uuid4()),
            })
            self.assertTrue(pocket["ok"], pocket)
            self.assertEqual(pocket["data"]["targets"],
                             [region["data"]["entity_id"]])
            inspected = await self.inspect(handle, revision=2)
            self.assertEqual(inspected["data"]["summary"]["counts"]["mops"], 1)

        self.run_async(test)

    def test_depth_increment_planner_balances_and_warns_on_heuristic_divergence(self):
        async def test():
            base = {
                "workspace_id": self.service.workspace.id,
                "units": "mm",
                "stock_thickness": 9,
                "cut_through": 0.5,
            }
            exact = await self.call("machining_calculate_depth_increment", {
                **base, "pass_count": 3,
            })
            self.assertTrue(exact["ok"], exact)
            self.assertIsNone(exact["document"])
            self.assertEqual(exact["data"]["mode"], "pass_count")
            self.assertEqual(exact["data"]["depth_increment"], 3.2)
            self.assertEqual(exact["data"]["pass_depths"], [3.2, 6.4, 9.5])
            self.assertAlmostEqual(exact["data"]["nominal_overshoot"], 0.1)
            self.assertAlmostEqual(exact["data"]["final_pass_depth"], 3.1)
            self.assertAlmostEqual(exact["data"]["final_pass_stock"], 2.6)
            self.assertEqual(exact["data"]["final_pass_cut_through"], 0.5)
            self.assertGreater(exact["data"]["final_stock_fraction"], 1 / 3)
            self.assertTrue(exact["data"]["recommendation_met"])
            self.assertFalse(exact["diagnostics"])

            bounded = await self.call("machining_calculate_depth_increment", {
                **base, "max_depth_increment": 3,
            })
            self.assertTrue(bounded["ok"], bounded)
            self.assertEqual(bounded["data"]["mode"], "max_depth_increment")
            self.assertEqual(bounded["data"]["pass_count"], 4)
            self.assertEqual(bounded["data"]["depth_increment"], 2.4)
            self.assertEqual(bounded["data"]["pass_depths"], [2.4, 4.8, 7.2, 9.5])
            self.assertAlmostEqual(bounded["data"]["final_pass_stock"], 1.8)
            self.assertTrue(bounded["data"]["recommendation_met"])

            inches = await self.call("machining_calculate_depth_increment", {
                **base, "units": "in", "stock_thickness": 0.354,
                "cut_through": 0.02, "pass_count": 3,
            })
            self.assertTrue(inches["ok"], inches)
            self.assertEqual(inches["data"]["rounding_increment"], 0.001)
            self.assertEqual(inches["data"]["depth_increment"], 0.125)
            self.assertEqual(inches["data"]["pass_depths"], [0.125, 0.25, 0.374])

            low_engagement = await self.call("machining_calculate_depth_increment", {
                **base, "pass_count": 19,
            })
            self.assertTrue(low_engagement["ok"], low_engagement)
            self.assertEqual(low_engagement["data"]["depth_increment"], 0.5)
            self.assertEqual(low_engagement["data"]["final_pass_stock"], 0)
            self.assertFalse(low_engagement["data"]["recommendation_met"])
            self.assertEqual(
                {item["code"] for item in low_engagement["diagnostics"]},
                {"DEPTH_ROUNDING_RELAXED", "FINAL_STOCK_ENGAGEMENT_LOW"},
            )

            shallow_maximum = await self.call("machining_calculate_depth_increment", {
                **base, "max_depth_increment": 0.4,
            })
            self.assertTrue(shallow_maximum["ok"], shallow_maximum)
            self.assertEqual(shallow_maximum["data"]["depth_increment"], 0.4)
            self.assertFalse(shallow_maximum["data"]["recommendation_met"])
            self.assertIn("FINAL_STOCK_ENGAGEMENT_LOW",
                          {item["code"] for item in shallow_maximum["diagnostics"]})

            coarse_rounding = await self.call("machining_calculate_depth_increment", {
                **base, "pass_count": 3, "rounding_increment": 5,
            })
            self.assertTrue(coarse_rounding["ok"], coarse_rounding)
            self.assertAlmostEqual(coarse_rounding["data"]["depth_increment"], 9.5 / 3)
            self.assertIn("DEPTH_ROUNDING_RELAXED",
                          {item["code"] for item in coarse_rounding["diagnostics"]})

            for arguments, field in (
                ({**base, "pass_count": 3, "max_depth_increment": 3.5}, None),
                ({**base}, None),
                ({**base, "stock_thickness": 1000000000,
                  "cut_through": 1, "pass_count": 1}, "stock_thickness"),
                ({**base, "pass_count": 3, "workspace_id": "0" * 64}, "workspace_id"),
            ):
                result = await self.call("machining_calculate_depth_increment", arguments)
                self.assertFalse(result["ok"], (arguments, result))
                self.assertEqual(result["error"]["code"],
                                 "WORKSPACE_MISMATCH" if field == "workspace_id"
                                 else "INVALID_ARGUMENT")
                self.assertEqual(result["error"]["field"], field)

            self.assertFalse(self.service.documents)
            self.assertFalse(self.service.ledger)

        self.run_async(test)

    def test_group_membership_preserves_geometry_and_mop_target_eligibility(self):
        async def test():
            handle = await self.create("grouped-targets")
            revision = 0

            async def add(tool, **arguments):
                nonlocal revision
                result = await self.call(tool, self.args(
                    document=handle, expected_revision=revision, **arguments))
                self.assertTrue(result["ok"], result)
                revision += 1
                return result

            additions = (
                ("geometry_add_rectangle", "rect", {
                    "layer": "Geometry", "x": 0, "y": 0, "width": 10, "height": 8}),
                ("geometry_add_circle", "circle", {
                    "layer": "Geometry", "x": 20, "y": 4, "diameter": 6}),
                ("geometry_add_arc", "arc", {
                    "layer": "Geometry", "x": 30, "y": 4, "radius": 3,
                    "start_angle": 0, "extent_angle": 180}),
                ("geometry_add_pline", "open", {
                    "layer": "Geometry", "closed": False,
                    "points": [{"x": 0, "y": 15}, {"x": 10, "y": 15}]}),
                ("geometry_add_pline", "closed", {
                    "layer": "Geometry", "closed": True,
                    "points": [{"x": 15, "y": 12}, {"x": 25, "y": 12},
                               {"x": 20, "y": 20}]}),
                ("geometry_add_points", "points", {
                    "layer": "Geometry", "points": [{"x": 35, "y": 15}]}),
                ("geometry_add_text", "text", {
                    "layer": "Geometry", "text": "CUT", "x": 0, "y": 25,
                    "height": 4}),
                ("geometry_add_region", "region", {
                    "layer": "Geometry",
                    "outer": {"points": [
                        {"x": 15, "y": 25}, {"x": 25, "y": 25},
                        {"x": 25, "y": 35}, {"x": 15, "y": 35},
                    ]}}),
            )
            ids = {}
            for tool, identifier, arguments in additions:
                result = await add(tool, identifier=identifier, **arguments)
                ids[identifier] = result["data"]["entity_id"]

            def primitive_geometry(records):
                return {
                    record["identifier"]: record["geometry"]
                    for record in records if record["kind"] == "primitive"
                }

            baseline_geometry = primitive_geometry(
                await self.inspect_records(handle, revision=revision))
            self.assertTrue(all(baseline_geometry.values()))

            for entity_id in ids.values():
                await add("relationship_add_to_group", entity_id=entity_id,
                          group="machining-targets")
            grouped_records = await self.inspect_records(handle, revision=revision)
            self.assertEqual(primitive_geometry(grouped_records), baseline_geometry)
            self.assertNotIn(
                "INSPECTION_UNSUPPORTED", self.diagnostics(await self.inspect(handle)))

            target_sets = {
                "profile": [ids[name] for name in
                            ("rect", "circle", "open", "closed", "text", "region")],
                "pocket": [ids[name] for name in
                           ("rect", "circle", "closed", "text", "region")],
                "engrave": [ids[name] for name in
                            ("rect", "circle", "arc", "open", "closed", "text")],
                "drill": [ids[name] for name in ("points", "circle")],
            }
            mop_ids = {}
            for family, targets in target_sets.items():
                arguments = self.mop_arguments(
                    identifier=family, part="Part", targets=[targets[0]])
                if family == "profile":
                    arguments["side"] = "Outside"
                result = await add(f"machining_add_{family}", **arguments)
                mop_ids[family] = result["data"]["mop_id"]

            for family, targets in target_sets.items():
                result = await add(
                    "machining_set_mop_targets", mop_id=mop_ids[family], targets=targets)
                self.assertEqual(set(result["data"]["targets"]), set(targets))

            grouped_records = await self.inspect_records(handle, revision=revision)

            def mop_state(records):
                return {
                    record["identifier"]: {
                        "targets": record["targets"],
                        "parameters": record["parameters"],
                        "parameter_metadata": record["parameter_metadata"],
                        "unsupported_fields": record["unsupported_fields"],
                    }
                    for record in records if record["kind"] == "mop"
                }

            grouped_mops = mop_state(grouped_records)
            self.assertEqual(set(grouped_mops), set(target_sets))
            for family, targets in target_sets.items():
                self.assertEqual(set(grouped_mops[family]["targets"]), set(targets))

            saved = await self.call("document_save", self.args(
                document=handle, expected_revision=revision, path="grouped-targets.cb"))
            self.assertTrue(saved["ok"], saved)
            reopened = await self.call("document_open", self.args(
                path="grouped-targets.cb", units="mm"))
            self.assertTrue(reopened["ok"], reopened)
            reopened_records = await self.inspect_records(reopened["document"], revision=0)
            self.assertEqual(primitive_geometry(reopened_records), baseline_geometry)
            self.assertEqual(mop_state(reopened_records), grouped_mops)

            for entity_id in ids.values():
                await add("relationship_remove_from_group", entity_id=entity_id,
                          group="machining-targets")
            ungrouped_records = await self.inspect_records(handle, revision=revision)
            self.assertEqual(primitive_geometry(ungrouped_records), baseline_geometry)
            self.assertEqual(mop_state(ungrouped_records), grouped_mops)

            await add("relationship_set_parent", entity_id=ids["points"],
                      parent_id=ids["rect"])
            for family, target in (("profile", ids["rect"]),
                                   ("drill", ids["points"])):
                rejected = await self.call("machining_set_mop_targets", self.args(
                    document=handle, expected_revision=revision,
                    mop_id=mop_ids[family], targets=[target]))
                self.assertEqual(rejected["error"]["code"], "UNSUPPORTED_OPERATION")
            await add("relationship_set_parent", entity_id=ids["points"],
                      parent_id=None)

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
                targets=[circle_id, open_path_id, closed_path_id, region_id], side="Inside",
                **self.mop_arguments(),
            )
            self.assertEqual(
                set(profile["data"]["targets"]),
                {circle_id, open_path_id, closed_path_id, region_id},
            )
            self.assertEqual(profile["data"]["side_semantics"], "VertexOrderRelative")
            self.assertIn("OPEN_PROFILE_SIDE_DIRECTIONAL", self.diagnostics(profile))

            for tool, target_id, expected_code in (
                ("machining_add_pocket", open_path_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_pocket", marks_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_engrave", marks_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_drill", rectangle_id, "UNSUPPORTED_OPERATION"),
                ("machining_add_drill", open_path_id, "UNSUPPORTED_OPERATION"),
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
                {circle_id, open_path_id, closed_path_id, region_id},
            )

            saved = await self.call("document_save", self.args(
                document=handle, expected_revision=final["revision"],
                path="profile-closed-contours.cb",
            ))
            self.assertTrue(saved["ok"], saved)
            self.assertIn("SERVER_WORKSPACE_ARTIFACT", self.diagnostics(saved))
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
                {circle_id, open_path_id, closed_path_id, region_id},
            )

        self.run_async(test)

    def test_spiral_drill_methods_states_auto_diameter_and_fit_validation(self):
        async def test():
            handle = await self.create("spiral-drill")
            points = await self.call("geometry_add_points", self.args(
                document=handle, expected_revision=0, identifier="point-hole",
                layer="Geometry", points=[{"x": 0, "y": 0}],
            ))
            self.assertTrue(points["ok"], points)
            circle = await self.call("geometry_add_circle", self.args(
                document=handle, expected_revision=1, identifier="circle-hole",
                layer="Geometry", x=20, y=0, diameter=12,
            ))
            self.assertTrue(circle["ok"], circle)

            common = self.mop_arguments(
                target_depth=-10, depth_increment=2, tool_diameter=6,
                cut_feedrate=800, plunge_feedrate=300, spindle_speed=1000,
                clearance_plane=3,
            )
            clockwise = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=2, identifier="spiral-cw",
                part="Part", targets=[points["data"]["entity_id"]],
                drilling_method="SpiralMill_CW", hole_diameter=12,
                roughing_clearance=0.5, drill_lead_out=True,
                spiral_flat_base=True, lead_out_length=0,
                **common,
            ))
            self.assertTrue(clockwise["ok"], clockwise)
            counterclockwise = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=3, identifier="spiral-ccw-auto",
                part="Part", targets=[circle["data"]["entity_id"]],
                drilling_method="SpiralMill_CCW", roughing_clearance=-0.5,
                drill_lead_out=True, spiral_flat_base=False,
                lead_out_length=-1.25, tool_profile="EndMill", **common,
            ))
            self.assertTrue(counterclockwise["ok"], counterclockwise)

            records = await self.inspect_records(handle, revision=4)
            mops = {record["identifier"]: record["parameters"] for record in records
                    if record["kind"] == "mop"}
            self.assertEqual(mops["spiral-cw"]["drilling_method"], "SpiralMill_CW")
            self.assertEqual(mops["spiral-cw"]["hole_diameter"], 12)
            self.assertEqual(mops["spiral-cw"]["roughing_clearance"], 0.5)
            self.assertTrue(mops["spiral-cw"]["drill_lead_out"])
            self.assertEqual(mops["spiral-cw"]["tool_profile"], "Unspecified")
            self.assertEqual(mops["spiral-ccw-auto"]["drilling_method"],
                             "SpiralMill_CCW")
            self.assertIsNone(mops["spiral-ccw-auto"]["hole_diameter"])
            self.assertFalse(mops["spiral-ccw-auto"]["spiral_flat_base"])
            self.assertEqual(mops["spiral-ccw-auto"]["lead_out_length"], -1.25)
            self.assertEqual(mops["spiral-ccw-auto"]["tool_profile"], "EndMill")

            saved = await self.call("document_save", self.args(
                document=handle, expected_revision=4, path="spiral.cb"))
            self.assertTrue(saved["ok"], saved)
            tree = ET.parse(self.root / "spiral.cb")
            native = {element.findtext("Name"): element
                      for element in tree.getroot().findall("./parts/part/machineops/drill")}
            cw = native["spiral-cw"]
            for tag in ("DrillingMethod", "HoleDiameter", "DrillLeadOut",
                        "SpiralFlatBase", "LeadOutLength", "RoughingClearance",
                        "ToolDiameter", "ToolProfile"):
                self.assertEqual(cw.find(tag).get("state"), "Value", tag)
            for tag in ("PeckDistance", "RetractHeight", "Dwell", "CustomScript"):
                self.assertIsNone(cw.find(tag), tag)
            auto = native["spiral-ccw-auto"]
            self.assertEqual(auto.find("HoleDiameter").get("state"), "Default")
            self.assertIn(auto.findtext("HoleDiameter"), (None, ""))

            reopened = await self.call("document_open", self.args(
                path="spiral.cb", units="mm"))
            self.assertTrue(reopened["ok"], reopened)
            reopened_records = await self.inspect_records(reopened["document"], revision=0)
            self.assert_records_close(self, reopened_records, records)

            point_auto = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=4, identifier="bad-auto",
                part="Part", targets=[points["data"]["entity_id"]],
                drilling_method="SpiralMill_CW", **common,
            ))
            self.assertEqual(point_auto["error"]["field"], "hole_diameter")
            impossible = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=4, identifier="bad-fit",
                part="Part", targets=[points["data"]["entity_id"]],
                drilling_method="SpiralMill_CCW", hole_diameter=7,
                roughing_clearance=0.5, **common,
            ))
            self.assertEqual(impossible["error"]["field"], "hole_diameter")
            impossible_auto = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=4, identifier="bad-auto-fit",
                part="Part", targets=[circle["data"]["entity_id"]],
                drilling_method="SpiralMill_CW", tool_diameter=12,
                roughing_clearance=0.5,
                **{key: value for key, value in common.items()
                   if key != "tool_diameter"},
            ))
            self.assertEqual(impossible_auto["error"]["field"], "hole_diameter")
            inactive_lead_out = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=4, identifier="bad-lead-out",
                part="Part", targets=[points["data"]["entity_id"]],
                drilling_method="SpiralMill_CW", hole_diameter=12,
                drill_lead_out=False, lead_out_length=1, **common,
            ))
            self.assertEqual(inactive_lead_out["error"]["field"], "lead_out_length")
            excessive_lead_out = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=4,
                identifier="bad-lead-out-radius", part="Part",
                targets=[points["data"]["entity_id"]],
                drilling_method="SpiralMill_CW", hole_diameter=12,
                roughing_clearance=0.5, drill_lead_out=True,
                lead_out_length=5.6, **common,
            ))
            self.assertEqual(excessive_lead_out["error"]["field"],
                             "lead_out_length")
            wrong_family = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=4, identifier="bad-peck",
                part="Part", targets=[points["data"]["entity_id"]],
                drilling_method="SpiralMill_CW", hole_diameter=12,
                peck_distance=1, **common,
            ))
            self.assertEqual(wrong_family["error"]["field"], "drilling_method")
            canned_clearance = await self.call("machining_add_drill", self.args(
                document=handle, expected_revision=4, identifier="bad-canned",
                part="Part", targets=[points["data"]["entity_id"]],
                drilling_method="CannedCycle", roughing_clearance=0.1,
                **common,
            ))
            self.assertEqual(canned_clearance["error"]["field"],
                             "roughing_clearance")

        self.run_async(test)

    def test_text_vcutter_engrave_and_automatic_profile_tabs_round_trip(self):
        async def test():
            handle = await self.create("engrave-tabs")
            text_result = await self.call("geometry_add_text", self.args(
                document=handle, expected_revision=0, identifier="label",
                layer="Geometry", text="CAM", x=2, y=3,
            ))
            self.assertTrue(text_result["ok"], text_result)
            text_id = text_result["data"]["entity_id"]

            rectangle = await self.call("geometry_add_rectangle", self.args(
                document=handle, expected_revision=1, identifier="outline",
                layer="Geometry", x=0, y=0, width=40, height=20,
            ))
            self.assertTrue(rectangle["ok"], rectangle)
            rectangle_id = rectangle["data"]["entity_id"]

            engrave = await self.call("machining_add_engrave", self.args(
                document=handle, expected_revision=2, identifier="v-label",
                part="Part", targets=[text_id], tool_profile="VCutter",
                **self.mop_arguments(),
            ))
            self.assertTrue(engrave["ok"], engrave)

            inactive_tab_leadin = await self.call("machining_add_profile", self.args(
                document=handle, expected_revision=3, identifier="inactive-tab-leadin",
                part="Part", targets=[rectangle_id], side="Outside",
                tab_method="Automatic", tab_use_leadins=True,
                **self.mop_arguments(target_depth=-3),
            ))
            self.assertEqual(inactive_tab_leadin["error"]["code"], "INVALID_ARGUMENT")
            self.assertEqual(inactive_tab_leadin["error"]["field"], "tab_use_leadins")

            profile = await self.call("machining_add_profile", self.args(
                document=handle, expected_revision=3, identifier="tabbed-outline",
                part="Part", targets=[rectangle_id], side="Outside",
                tab_method="Automatic", tab_width=5, tab_height=1,
                tab_min_tabs=2, tab_max_tabs=4, tab_distance=0,
                tab_size_threshold=0, tab_use_leadins=False, tab_style="Triangle",
                **self.mop_arguments(target_depth=-3),
            ))
            self.assertTrue(profile["ok"], profile)
            self.assertEqual(profile["data"]["side_semantics"], "ClosedBoundary")

            records = await self.inspect_records(handle, revision=4)
            engrave_record = next(record for record in records
                                  if record.get("identifier") == "v-label")
            profile_record = next(record for record in records
                                  if record.get("identifier") == "tabbed-outline")
            self.assertEqual(engrave_record["parameters"]["tool_profile"], "VCutter")
            self.assertEqual(engrave_record["targets"], [text_id])
            self.assertEqual(profile_record["parameters"]["tab_method"], "Automatic")
            self.assertEqual(profile_record["parameters"]["tab_distance"], 0)
            self.assertEqual(profile_record["parameters"]["tab_size_threshold"], 0)
            self.assertEqual(profile_record["parameters"]["tab_style"], "Triangle")
            self.assertFalse(profile_record["parameters"]["tab_use_leadins"])

            invalid = await self.call("machining_add_profile", self.args(
                document=handle, expected_revision=4, identifier="bad-tabs",
                part="Part", targets=[rectangle_id], side="Outside",
                tab_method="Automatic", tab_min_tabs=5, tab_max_tabs=2,
                **self.mop_arguments(),
            ))
            self.assertEqual(invalid["error"]["code"], "INVALID_ARGUMENT")
            self.assertEqual((await self.inspect(handle))["revision"], 4)

            saved = await self.call("document_save", self.args(
                document=handle, expected_revision=4, path="engrave-tabs.cb",
            ))
            self.assertTrue(saved["ok"], saved)
            root = ET.parse(self.root / "engrave-tabs.cb").getroot()
            self.assertEqual(root.findtext(".//engrave[Name='v-label']/ToolProfile"), "VCutter")
            tabs = root.find(".//profile[Name='tabbed-outline']/HoldingTabs")
            self.assertIsNotNone(tabs)
            self.assertEqual(tabs.findtext("TabMethod"), "Automatic")
            self.assertEqual(tabs.findtext("TabDistance"), "0")
            self.assertEqual(tabs.findtext("SizeThreshold"), "0")
            self.assertEqual(tabs.findtext("TabStyle"), "Triangle")

            reopened = await self.call("document_open", self.args(
                path="engrave-tabs.cb", units="mm",
            ))
            self.assertTrue(reopened["ok"], reopened)
            reopened_records = await self.inspect_records(reopened["document"])
            reopened_engrave = next(record for record in reopened_records
                                    if record.get("identifier") == "v-label")
            reopened_profile = next(record for record in reopened_records
                                    if record.get("identifier") == "tabbed-outline")
            self.assertEqual(reopened_engrave["parameters"]["tool_profile"], "VCutter")
            self.assertEqual(reopened_profile["parameters"]["tab_style"], "Triangle")
            self.assertEqual(reopened_profile["parameters"]["tab_distance"], 0.0)

        self.run_async(test)

    def test_text_profile_pocket_and_signed_roughing_clearance_round_trip(self):
        async def test():
            handle = await self.create("text-machining")
            text_result = await self.call("geometry_add_text", self.args(
                document=handle, expected_revision=0, identifier="label",
                layer="Geometry", text="CAM", x=2, y=3,
            ))
            self.assertTrue(text_result["ok"], text_result)
            text_id = text_result["data"]["entity_id"]

            profile = await self.call("machining_add_profile", self.args(
                document=handle, expected_revision=1, identifier="text-profile",
                part="Part", targets=[text_id], side="Inside",
                roughing_clearance=0.2, **self.mop_arguments(),
            ))
            self.assertTrue(profile["ok"], profile)
            pocket = await self.call("machining_add_pocket", self.args(
                document=handle, expected_revision=2, identifier="text-pocket",
                part="Part", targets=[text_id], roughing_clearance=-0.1,
                **self.mop_arguments(),
            ))
            self.assertTrue(pocket["ok"], pocket)
            engrave = await self.call("machining_add_engrave", self.args(
                document=handle, expected_revision=3, identifier="text-engrave",
                part="Part", targets=[text_id], roughing_clearance=0.35,
                **self.mop_arguments(),
            ))
            self.assertTrue(engrave["ok"], engrave)

            records = await self.inspect_records(handle, revision=4)
            parameters = {
                record["identifier"]: record["parameters"]
                for record in records if record["kind"] == "mop"
            }
            self.assertEqual(parameters["text-profile"]["roughing_clearance"], 0.2)
            self.assertEqual(parameters["text-pocket"]["roughing_clearance"], -0.1)
            self.assertEqual(parameters["text-engrave"]["roughing_clearance"], 0.35)

            saved = await self.call("document_save", self.args(
                document=handle, expected_revision=4, path="text-machining.cb",
            ))
            self.assertTrue(saved["ok"], saved)
            root = ET.parse(self.root / "text-machining.cb").getroot()
            self.assertEqual(root.findtext(".//profile[Name='text-profile']/RoughingClearance"), "0.2")
            self.assertEqual(root.findtext(".//pocket[Name='text-pocket']/RoughingClearance"), "-0.1")
            self.assertEqual(root.findtext(".//engrave[Name='text-engrave']/RoughingClearance"), "0.35")

            reopened = await self.call("document_open", self.args(
                path="text-machining.cb", units="mm",
            ))
            self.assertTrue(reopened["ok"], reopened)
            reopened_records = await self.inspect_records(reopened["document"])
            reopened_parameters = {
                record["identifier"]: record["parameters"]
                for record in reopened_records if record["kind"] == "mop"
            }
            self.assertEqual(reopened_parameters, parameters)

        self.run_async(test)

    def test_imported_canned_drill_nonzero_roughing_clearance_is_inspectable(self):
        async def test():
            source, _, _, drill = self.direct_project()
            drill.roughing_clearance = 0.25
            source.save(str(self.root / "native-drill-clearance.cb"))

            opened = await self.call("document_open", self.args(
                path="native-drill-clearance.cb", units="mm",
            ))
            self.assertTrue(opened["ok"], opened)
            records = await self.inspect_records(opened["document"])
            drill_record = next(record for record in records
                                if record.get("identifier") == "drill")
            self.assertEqual(drill_record["parameters"]["roughing_clearance"], 0.25)

        self.run_async(test)


if __name__ == "__main__":
    unittest.main()
