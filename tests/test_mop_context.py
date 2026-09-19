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
    def test_imported_part_nesting_preserves_native_subtree_until_model_edit(self):
        project = CBProject("native-nesting")
        layer = project.add_layer("Geometry")
        square = project.add_rect(layer, identifier="square")
        part = project.add_part("Part1")
        project.add_profile_mop(part, [square], identifier="profile")

        source_tree = build_xml_tree(project)
        source_part = source_tree.getroot().find("./parts/part")
        native_nesting = source_part.find("Nesting")
        native_nesting.set("vendor-attribute", "preserve-me")
        ET.SubElement(native_nesting, "VendorSetting").text = "native-value"
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
        self.assertEqual("preserve-me", changed_nesting.get("vendor-attribute"))
        self.assertEqual("native-value", changed_nesting.findtext("VendorSetting"))

        reloaded = read_cambam_bytes(
            ET.tostring(changed_tree.getroot(), encoding="utf-8"),
            source_name="updated-nesting.cb",
        )
        self.assertEqual("Grid", reloaded.get_part("Part1").nesting_method)
        self.assertEqual(2, reloaded.get_part("Part1").nesting_rows)
        self.assertEqual(3, reloaded.get_part("Part1").nesting_columns)

    def test_add_part_keeps_legacy_ordering_arguments_positional(self):
        project = CBProject("part-ordering")
        first = project.add_part("First")
        second = project.add_part(
            "Second", True, 12.5, 1220.0, 2440.0, "MDF", "210,180,140",
            (0.0, 0.0), None, None, "First", False,
        )
        self.assertIsNotNone(second)
        self.assertEqual([second.internal_id, first.internal_id], project._part_order)

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
