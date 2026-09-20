"""Preservation-aware structured MOP inspection regressions."""

from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from uuid import uuid4

if find_spec("mcp") is not None:
    import anyio
    from cambam_builder.mcp_adapter.schema import OUTPUTS
    from cambam_builder.mcp_adapter.service import Document, DocumentService
else:  # pragma: no cover - base-only environment
    Document = DocumentService = None
    OUTPUTS = {}

from cambam_builder import CBProject


@unittest.skipIf(DocumentService is None, "Install .[mcp] for adapter checks")
class PreservationAwareMopInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_async(self, body):
        async def run():
            self.service = DocumentService(self.root)
            await body()

        anyio.run(run)

    def args(self, **values):
        result = {
            "workspace_id": self.service.workspace.id,
            "request_id": str(uuid4()),
        }
        result.update(values)
        return result

    async def call(self, name, arguments):
        result = await self.service.call_tool(name, arguments)
        OUTPUTS[name].validate(result)
        return result

    async def inspect(self, handle):
        result = await self.call("document_inspect", self.args(document=handle))
        self.assertTrue(result["ok"], result)
        return result

    @staticmethod
    def common(**overrides):
        values = {
            "target_depth": -2.0,
            "depth_increment": 0.5,
            "tool_diameter": 3.0,
            "cut_feedrate": 300.0,
            "plunge_feedrate": 100.0,
            "spindle_speed": 12000,
            "stock_surface": 0.0,
            "clearance_plane": 5.0,
        }
        values.update(overrides)
        return values

    def make_native_fixture(self):
        project = CBProject("preserved-inspection")
        project.add_layer("Geometry")
        outline = project.add_rect(
            "Geometry", identifier="outline", width=20, height=10
        )
        hole = project.add_circle(
            "Geometry", identifier="hole", center=(5, 5), diameter=8
        )
        part = project.add_part("Part")
        project.add_profile_mop(
            part, targets=[outline], identifier="profile", name="profile",
            profile_side="Outside", lead_in_type="None", **self.common()
        )
        project.add_pocket_mop(
            part, targets=[outline], identifier="pocket", name="pocket",
            **self.common()
        )
        project.add_engrave_mop(
            part, targets=[], identifier="empty-engrave", name="empty-engrave",
            **self.common()
        )
        project.add_drill_mop(
            part, targets=[hole], identifier="spiral", name="spiral",
            drilling_method="SpiralMill_CW", hole_diameter=8.0,
            tool_profile="EndMill", **self.common()
        )
        project.add_drill_mop(
            part, targets=[hole], identifier="script", name="script",
            drilling_method="CustomScript", custom_script="G1 X$x Y$y",
            tool_profile="Drill", **self.common()
        )
        path = self.root / "native.cb"
        project.save(str(path))

        tree = ET.parse(path)
        root = tree.getroot()
        profile = root.find("./parts/part/machineops/profile")
        profile.find("TargetDepth").set("state", "Default")
        profile.find("TargetDepth").text = "-2.5"
        profile.find("SpindleDirection").set("state", "Default")
        profile.find("SpindleDirection").text = "CCW"
        profile.find("OptimisationMode").text = "Experimental"
        tabs = profile.find("HoldingTabs")
        tabs.find("TabMethod").text = "Manual"
        ET.SubElement(tabs, "ManualTabPoints").text = "1,2;3,4"
        profile.find("MaxCrossoverDistance").set("plugin", "keep")

        pocket = root.find("./parts/part/machineops/pocket")
        lead = pocket.find("LeadInMove")
        lead.set("state", "Default")
        lead.find("LeadInType").set("state", "Value")
        ET.SubElement(lead, "NativeLeadField").text = "keep"

        engrave = root.find("./parts/part/machineops/engrave")
        engrave.find("CutOrdering").text = "LevelFirst"
        engrave.find("MaxCrossoverDistance").text = "0.55"

        spiral = root.findall("./parts/part/machineops/drill")[0]
        ET.SubElement(spiral, "PeckDistance", {"state": "Default"}).text = "3"
        ET.SubElement(spiral, "PluginField").text = "opaque"
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return path

    def test_imported_values_states_applicability_and_opaque_fields_are_separate(self):
        async def body():
            path = self.make_native_fixture()
            opened = await self.call(
                "document_open", self.args(path=path.name, units="mm")
            )
            self.assertTrue(opened["ok"], opened)
            inspected = await self.inspect(opened["document"])
            records = {
                record["identifier"]: record
                for record in inspected["data"]["entities"]
                if record["kind"] == "mop"
            }

            profile = records["profile"]
            self.assertEqual(profile["parameters"]["target_depth"], -2.5)
            self.assertEqual(profile["parameters"]["spindle_direction"], "CCW")
            self.assertEqual(
                profile["parameters"]["optimisation_mode"], "Experimental"
            )
            self.assertEqual(
                profile["parameter_metadata"]["target_depth"]["native_state"],
                "Default",
            )
            self.assertEqual(profile["parameters"]["tab_method"], "Manual")
            self.assertFalse(
                profile["parameter_metadata"]["tab_width"]["applicable"]
            )
            self.assertEqual(
                profile["unsupported_fields"],
                ["HoldingTabs/ManualTabPoints", "MaxCrossoverDistance/@plugin"],
            )

            pocket = records["pocket"]
            self.assertEqual(
                pocket["parameter_metadata"]["lead_in_type"]["native_state"],
                "Default",
            )
            self.assertEqual(
                pocket["parameter_metadata"]["lead_in_spiral_angle"]["native_state"],
                "Default",
            )
            self.assertEqual(
                pocket["unsupported_fields"], ["LeadInMove/NativeLeadField"]
            )

            empty = records["empty-engrave"]
            self.assertEqual(empty["targets"], [])
            self.assertEqual(empty["parameters"]["cut_ordering"], "LevelFirst")
            self.assertEqual(empty["parameters"]["max_crossover_distance"], 0.55)

            spiral = records["spiral"]
            self.assertEqual(spiral["parameters"]["peck_distance"], 3.0)
            self.assertEqual(
                spiral["parameter_metadata"]["peck_distance"],
                {"native_state": "Default", "applicable": False},
            )
            self.assertEqual(spiral["unsupported_fields"], ["PluginField"])

            script = records["script"]
            self.assertEqual(script["parameters"]["drilling_method"], "CustomScript")
            self.assertNotIn("custom_script", script["parameters"])
            self.assertEqual(script["unsupported_fields"], ["CustomScript"])

            messages = [item["message"] for item in inspected["diagnostics"]]
            self.assertTrue(any("ManualTabPoints" in message for message in messages))
            self.assertTrue(any("CustomScript" in message for message in messages))

            saved = await self.call(
                "document_save",
                self.args(
                    document=opened["document"], expected_revision=0,
                    path="roundtrip.cb",
                ),
            )
            self.assertTrue(saved["ok"], saved)
            roundtrip = ET.parse(self.root / "roundtrip.cb").getroot()
            saved_profile = roundtrip.find("./parts/part/machineops/profile")
            self.assertEqual(saved_profile.find("TargetDepth").get("state"), "Default")
            self.assertEqual(
                saved_profile.findtext("HoldingTabs/ManualTabPoints"), "1,2;3,4"
            )
            self.assertEqual(
                saved_profile.find("MaxCrossoverDistance").get("plugin"), "keep"
            )
            self.assertEqual(
                roundtrip.findtext("./parts/part/machineops/drill/PluginField"),
                "opaque",
            )

        self.run_async(body)

    def test_live_group_source_retains_parameters_and_is_identified(self):
        async def body():
            project = CBProject("live-group")
            project.add_layer("Geometry")
            project.add_rect(
                "Geometry", identifier="outline", width=10, height=10,
                groups=["targets"],
            )
            part = project.add_part("Part")
            project.add_profile_mop(
                part, target_group="targets", identifier="profile", name="profile",
                profile_side="Outside", lead_in_type="None", **self.common()
            )
            handle = f"{self.service.bootstrap['boot_id']}:{uuid4()}"
            self.service.documents[handle] = Document(project=project, units="mm")
            inspected = await self.inspect(handle)
            mop = next(
                record for record in inspected["data"]["entities"]
                if record["kind"] == "mop"
            )
            self.assertEqual(mop["target_group"], "targets")
            self.assertEqual(mop["parameters"]["target_depth"], -2.0)
            self.assertFalse(mop["unsupported_fields"])

        self.run_async(body)


if __name__ == "__main__":
    unittest.main()
