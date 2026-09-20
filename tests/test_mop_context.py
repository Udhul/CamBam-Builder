"""Preserve native context used to resolve Default MOP parameters."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_bytes, read_cambam_file
from cambam_builder.cambam_writer import build_xml_tree


def signature(element):
    if element is None:
        return None
    return (element.tag, element.attrib, (element.text or "").strip(),
            [signature(child) for child in element])


class MopContextTests(unittest.TestCase):
    def test_point_list_nesting_is_preserved_and_not_mixed_into_grid(self):
        project = CBProject("native-nesting")
        layer = project.add_layer("Geometry")
        square = project.add_rect(layer, identifier="square")
        project.add_points(layer, identifier="locations", points=[(3, 4), (20, 8)])
        part = project.add_part("Part1", nesting_method="PointList")
        project.add_profile_mop(part, [square], identifier="profile")

        source_tree = build_xml_tree(project)
        source_part = source_tree.getroot().find("./parts/part")
        native_nesting = source_part.find("Nesting")
        point_xml_id = source_tree.getroot().find(".//points").get("id")
        ET.SubElement(native_nesting, "PointListID").text = point_xml_id
        ET.SubElement(native_nesting, "GCodeOrder").text = "Auto"
        expected_native = signature(native_nesting)

        loaded = read_cambam_bytes(
            ET.tostring(source_tree.getroot(), encoding="utf-8"),
            source_name="native-nesting.cb",
        )
        unchanged_tree = build_xml_tree(loaded)
        self.assertEqual(
            expected_native,
            signature(unchanged_tree.getroot().find("./parts/part/Nesting")),
        )

        loaded_part = loaded.get_part("Part1")
        updated = loaded.add_part(
            "Part1",
            enabled=loaded_part.enabled,
            stock_thickness=loaded_part.stock_thickness,
            stock_width=loaded_part.stock_width,
            stock_height=loaded_part.stock_height,
            stock_material=loaded_part.stock_material,
            stock_color=loaded_part.stock_color,
            machining_origin=loaded_part.machining_origin,
            default_tool_diameter=loaded_part.default_tool_diameter,
            default_spindle_speed=loaded_part.default_spindle_speed,
            nesting_method="Grid",
            nesting_rows=2,
            nesting_columns=3,
            nesting_spacing=5.5,
            nesting_grid_order="LeftDown",
            nesting_grid_alternate=True,
        )
        self.assertIs(updated, loaded_part)

        changed_tree = build_xml_tree(loaded)
        changed_nesting = changed_tree.getroot().find("./parts/part/Nesting")
        self.assertEqual("Grid", changed_nesting.findtext("NestMethod"))
        self.assertEqual("2", changed_nesting.findtext("Rows"))
        self.assertEqual("3", changed_nesting.findtext("Columns"))
        self.assertEqual("5.5", changed_nesting.findtext("Spacing"))
        self.assertEqual("LeftDown", changed_nesting.findtext("GridOrder"))
        self.assertEqual("true", changed_nesting.findtext("GridDirectionAlternate"))
        self.assertIsNone(changed_nesting.find("PointListID"))
        self.assertIsNone(changed_nesting.find("GCodeOrder"))

        reloaded = read_cambam_bytes(
            ET.tostring(changed_tree.getroot(), encoding="utf-8"),
            source_name="updated-nesting.cb",
        )
        self.assertEqual("Grid", reloaded.get_part("Part1").nesting_method)
        self.assertEqual(2, reloaded.get_part("Part1").nesting_rows)
        self.assertEqual(3, reloaded.get_part("Part1").nesting_columns)

    def test_same_primitive_can_be_targeted_by_mops_in_multiple_parts(self):
        project = CBProject("shared-geometry")
        layer = project.add_layer("Geometry")
        outline = project.add_rect(layer, identifier="shared-outline")
        first = project.add_part("First")
        second = project.add_part("Second")
        project.add_profile_mop(first, [outline], identifier="first-profile")
        project.add_profile_mop(second, [outline], identifier="second-profile")

        tree = build_xml_tree(project)
        references = tree.getroot().findall("./parts/part/machineops/profile/primitive/prim")
        self.assertEqual(2, len(references))
        self.assertEqual(references[0].text, references[1].text)
        self.assertEqual(tree.getroot().find(".//rect").get("id"), references[0].text)

    def test_direct_api_normalizes_documentation_vcutter_spelling(self):
        project = CBProject("vcutter-spelling")
        layer = project.add_layer("Geometry")
        line = project.add_pline(layer, [(0, 0), (10, 0)], identifier="line")
        part = project.add_part("Part")
        mop = project.add_engrave_mop(
            part, [line], identifier="engrave", tool_profile="Vcutter"
        )

        self.assertEqual("VCutter", mop.tool_profile)
        self.assertEqual(
            "VCutter",
            build_xml_tree(project).getroot().findtext(
                "./parts/part/machineops/engrave/ToolProfile"
            ),
        )

    def test_fresh_manual_tabs_fail_instead_of_emitting_incomplete_xml(self):
        project = CBProject("manual-tabs")
        layer = project.add_layer("Geometry")
        outline = project.add_rect(layer, identifier="outline")
        part = project.add_part("Part")
        project.add_profile_mop(
            part, [outline], identifier="profile", tab_method="Manual"
        )

        with self.assertRaisesRegex(ValueError, "explicit native tab points"):
            build_xml_tree(project)

    def test_native_machining_and_part_style_context_survives_two_roundtrips(self):
        project = CBProject("native-context")
        layer = project.add_layer("Geometry")
        square = project.add_rect(layer, identifier="square")
        part = project.add_part("Part1")
        project.add_profile_mop(part, [square], identifier="profile")
        tree = build_xml_tree(project)
        options = tree.getroot().find("MachiningOptions")
        ET.SubElement(options, "Style").text = "Global style"
        ET.SubElement(options, "StyleLibrary").text = "Example-mm"
        ET.SubElement(options, "ClearancePlane", state="Value").text = "7.5"
        part_xml = tree.getroot().find("./parts/part")
        ET.SubElement(part_xml, "ToolDiameter", state="Default").text = "4.2"
        ET.SubElement(part_xml, "Style").text = "Part style"
        ET.SubElement(part_xml, "StyleLibrary").text = "Part-mm"
        part_xml.find("ToolProfile").text = "BallNose"
        part_xml.find("Nesting/NestMethod").text = "Grid"
        expected_options = signature(options)
        expected_parameters = [signature(child) for child in part_xml
                               if child.tag not in {"Stock", "MachiningOrigin", "ToolDiameter", "machineops"}]
        output = Path("output").resolve()
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mop-context-", dir=output) as directory:
            for index in range(2):
                path = Path(directory) / f"native-{index}.cb"
                tree.write(path, encoding="utf-8")
                loaded = read_cambam_file(str(path))
                self.assertIsNotNone(loaded)
                tree = build_xml_tree(loaded)
                self.assertEqual(expected_options, signature(tree.getroot().find("MachiningOptions")))
                actual = tree.getroot().find("./parts/part")
                self.assertEqual("Default", actual.find("ToolDiameter").get("state"))
                self.assertEqual("4.2", actual.findtext("ToolDiameter"))
                self.assertEqual(expected_parameters, [signature(child) for child in actual
                                 if child.tag not in {"Stock", "MachiningOrigin", "ToolDiameter", "machineops"}])
                self.assertEqual([square.internal_id], loaded.get_mop_targets("profile"))
            loaded.get_part("Part1").default_tool_diameter = 6.0
            self.assertEqual("6.0", build_xml_tree(loaded).findtext("./parts/part/ToolDiameter"))


if __name__ == "__main__":
    unittest.main()
