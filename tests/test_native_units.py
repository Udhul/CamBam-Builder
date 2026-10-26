"""Drawing-unit labels survive editing without converting document values."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.native.reader import read_cambam_bytes, read_cambam_file
from cambam_builder.native.writer import build_xml_tree


class NativeUnitsTests(unittest.TestCase):
    def test_explicit_and_unspecified_native_units_survive_edit_and_clone(self) -> None:
        for units in ("Millimeters", "Inches", None):
            with self.subTest(units=units):
                attribute = f' units="{units}"' if units is not None else ""
                data = (
                    f'<CADFile{attribute}><layers><layer name="Geometry"><objects>'
                    '<rect id="47" p="1.25,2.5,0" w="12.5" h="3.75"/>'
                    '</objects></layer></layers></CADFile>'
                ).encode()
                loaded = read_cambam_bytes(data)
                self.assertEqual(loaded.units, units)
                outline = loaded.get_primitive("rect_47")
                self.assertEqual(outline.width, 12.5)
                outline.width = 13.25
                edited = loaded.clone()
                root = build_xml_tree(edited).getroot()
                self.assertEqual(root.get("units"), units)
                reopened = read_cambam_bytes(ET.tostring(root))
                self.assertEqual(reopened.units, units)
                self.assertEqual(reopened.get_primitive("rect_47").width, 13.25)
                self.assertEqual(reopened.get_primitive("rect_47").relative_corner, (1.25, 2.5))

    def test_default_and_explicit_assignment_change_only_the_label(self) -> None:
        project = CBProject("metric-default")
        self.assertEqual(project.units, "Millimeters")
        layer = project.add_layer("Geometry")
        outline = project.add_rect(layer, identifier="outline", width=25.4, height=10)
        before = outline.get_absolute_coordinates_xyz()
        for units in ("Inches", None, "Millimeters"):
            project.units = units
            self.assertEqual(build_xml_tree(project).getroot().get("units"), units)
            self.assertEqual(outline.get_absolute_coordinates_xyz(), before)
        self.assertEqual(CBProject("imperial", units="Inches").units, "Inches")

    def test_invalid_unit_labels_reject_without_changing_existing_value(self) -> None:
        project = CBProject("valid")
        for invalid in ("mm", "inches", "", "Unspecified", " Millimeters ", 25.4):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "units"):
                    project.units = invalid
                self.assertEqual(project.units, "Millimeters")
                with self.assertRaisesRegex(ValueError, "units"):
                    CBProject("invalid", units=invalid)
                data = f'<CADFile units="{invalid}"><layers/></CADFile>'.encode()
                with self.assertRaisesRegex(ValueError, "units"):
                    read_cambam_bytes(data)

    def test_legacy_file_reader_keeps_its_none_on_invalid_input_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.cb"
            path.write_text('<CADFile units="invalid"><layers/></CADFile>', encoding="utf-8")
            self.assertIsNone(read_cambam_file(str(path)))


if __name__ == "__main__":
    unittest.main()
