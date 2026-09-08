"""Preserve native context used to resolve Default MOP parameters."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import build_xml_tree


def signature(element):
    if element is None:
        return None
    return (element.tag, element.attrib, (element.text or "").strip(),
            [signature(child) for child in element])


class MopContextTests(unittest.TestCase):
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
