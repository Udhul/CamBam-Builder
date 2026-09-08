"""Native MOP parameter state and template fidelity regressions."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class MopParameterTests(unittest.TestCase):
    def setUp(self):
        self.output = Path("output").resolve()
        self.output.mkdir(exist_ok=True)

    def make_project(self):
        project = CBProject("parameter-regression")
        layer = project.add_layer("Geometry")
        part = project.add_part("Machining")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        mops = [
            project.add_profile_mop(part, targets=[target], identifier="profile",
                                    target_depth=-1.25, stepover=0.2,
                                    custom_mop_header="  G0 X0\n",
                                    tab_method="Automatic", tab_style="Triangle"),
            project.add_pocket_mop(part, targets=[target], identifier="pocket",
                                   target_depth=-2.0, region_fill_style="HorizontalScanline"),
            project.add_engrave_mop(part, targets=[target], identifier="engrave",
                                    target_depth=-0.4, cut_ordering="LevelFirst"),
            project.add_drill_mop(part, targets=[target], identifier="drill",
                                  target_depth=-3.0, drilling_method="SpiralMill_CW",
                                  hole_diameter=2.2, lead_out_length=1.5),
        ]
        return project, mops

    @staticmethod
    def mop(tree, identifier):
        for element in tree.getroot().findall("./parts/part/machineops/*"):
            if identifier in (element.findtext("Tag") or ""):
                return element
        raise AssertionError(identifier)

    def save(self, project, directory, name):
        path = Path(directory) / f"{name}.cb"
        save_cambam_file(project, str(path))
        return path

    def test_supported_fields_and_unknown_native_children_round_trip(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            tree = ET.parse(first)
            profile = self.mop(tree, "profile")
            ET.SubElement(profile, "NativeExtension").text = "retained"
            ET.SubElement(profile.find("LeadOutMove"), "NativeLeadOutField").text = "keep"
            profile.find("TargetDepth").set("state", "Default")
            profile.find("TargetDepth").text = "-99.5"
            tree.write(first, encoding="utf-8", xml_declaration=True)

            loaded = read_cambam_file(str(first))
            self.assertIsNotNone(loaded)
            self.assertEqual(-99.5, loaded.get_mop("profile").target_depth)
            self.assertEqual("Default", loaded.get_mop("profile")._xml_parameter_states["target_depth"])
            self.assertEqual("Triangle", loaded.get_mop("profile").tab_style)
            self.assertEqual("  G0 X0\n", loaded.get_mop("profile").custom_mop_header)
            self.assertEqual("HorizontalScanline", loaded.get_mop("pocket").region_fill_style)
            self.assertEqual("LevelFirst", loaded.get_mop("engrave").cut_ordering)
            self.assertEqual(2.2, loaded.get_mop("drill").hole_diameter)

            second = self.save(loaded, directory, "second")
            round_trip = ET.parse(second)
            profile = self.mop(round_trip, "profile")
            self.assertEqual("retained", profile.findtext("NativeExtension"))
            self.assertEqual("keep", profile.findtext("LeadOutMove/NativeLeadOutField"))
            self.assertEqual("Default", profile.find("TargetDepth").get("state"))
            self.assertEqual("-99.5", profile.find("TargetDepth").text)

    def test_direct_edit_overrides_imported_state_and_explicit_state_api(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            loaded = read_cambam_file(str(first))
            mop = loaded.get_mop("profile")
            mop.target_depth = -7.0
            second = self.save(loaded, directory, "edited")
            element = self.mop(ET.parse(second), "profile")
            self.assertEqual("Value", element.find("TargetDepth").get("state"))
            self.assertEqual("-7.0", element.find("TargetDepth").text)

            mop.set_parameter_state("target_depth", "Default")
            third = self.save(loaded, directory, "default")
            element = self.mop(ET.parse(third), "profile")
            self.assertEqual("Default", element.find("TargetDepth").get("state"))
            mop.target_depth = -7.0  # Repeating a cached value intentionally makes it explicit.
            element = mop.to_xml_element(loaded, [1])
            self.assertEqual("Value", element.find("TargetDepth").get("state"))
            with self.assertRaises(ValueError):
                mop.set_parameter_state("no_such_parameter", "Value")
            with self.assertRaises(ValueError):
                mop.set_parameter_state("target_depth", "Inherited")
            with self.assertRaises(ValueError):
                mop.set_parameter_state("tab_width", "Default")

    def test_explicit_state_api_applies_to_new_mops(self):
        project, mops = self.make_project()
        mop = mops[0]
        mop.set_parameter_state("target_depth", "Default")
        element = mop.to_xml_element(project, [1])
        self.assertEqual("Default", element.find("TargetDepth").get("state"))

    def test_missing_optional_elements_remain_absent_until_edited(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            tree = ET.parse(first)
            drill = self.mop(tree, "drill")
            drill.remove(drill.find("HoleDiameter"))
            tree.write(first, encoding="utf-8", xml_declaration=True)
            loaded = read_cambam_file(str(first))
            self.assertIsNone(loaded.get_mop("drill").hole_diameter)
            second = self.save(loaded, directory, "second")
            self.assertIsNone(self.mop(ET.parse(second), "drill").find("HoleDiameter"))
            loaded.get_mop("drill").hole_diameter = 3.0
            third = self.save(loaded, directory, "third")
            hole = self.mop(ET.parse(third), "drill").find("HoleDiameter")
            self.assertIsNotNone(hole)
            self.assertEqual("Value", hole.get("state"))
            self.assertEqual("3.0", hole.text)

    def test_nested_edits_activate_container_without_changing_plain_scalar_format(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            tree = ET.parse(first)
            profile = self.mop(tree, "profile")
            profile.find("HoldingTabs").set("state", "Default")
            profile.find("LeadInMove").set("state", "Default")
            tree.write(first, encoding="utf-8")
            loaded = read_cambam_file(str(first))
            mop = loaded.get_mop("profile")
            self.assertIsNone(mop._xml_template.find("primitive"))
            mop.tab_width = 9.0
            mop.lead_in_spiral_angle = 22.0
            element = mop.to_xml_element(loaded, [1])
            self.assertEqual("Value", element.find("HoldingTabs").get("state"))
            self.assertEqual("9.0", element.findtext("HoldingTabs/Width"))
            self.assertNotIn("state", element.find("HoldingTabs/Width").attrib)
            self.assertEqual("Value", element.find("LeadInMove").get("state"))
            self.assertEqual("22.0", element.findtext("LeadInMove/SpiralAngle"))


if __name__ == "__main__":
    unittest.main()
