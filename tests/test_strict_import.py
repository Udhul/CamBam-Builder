"""Regression tests for bounded, strict byte-oriented CamBam import."""

import unittest
import xml.etree.ElementTree as ET
import json

from cambam_builder import CBProject
from cambam_builder.cambam_reader import CamBamImportLimitError, read_cambam_bytes
from cambam_builder.cambam_writer import build_xml_tree


class StrictImportTests(unittest.TestCase):
    def project_bytes(self):
        project = CBProject("strict-input")
        layer = project.add_layer("Geometry")
        project.add_rect(layer, width=2, height=3, identifier="outline")
        return ET.tostring(
            build_xml_tree(project).getroot(), encoding="utf-8", xml_declaration=True
        )

    def test_bytes_round_trip_returns_project(self):
        loaded = read_cambam_bytes(self.project_bytes(), source_name="example.cb")
        self.assertEqual(loaded.project_name, "strict-input")
        self.assertIsNotNone(loaded.get_primitive("outline"))

    def test_rejects_non_bytes_and_oversized_input(self):
        with self.assertRaises(ValueError):
            read_cambam_bytes("<CADFile/>")
        with self.assertRaises(CamBamImportLimitError):
            read_cambam_bytes(b" " * (10 * 1024 * 1024 + 1))

    def test_rejects_dtd_and_entity_declarations_before_parsing(self):
        payloads = (
            b'<!DOCTYPE CADFile [<!ENTITY x "unsafe">]><CADFile><layers/></CADFile>',
            b'<CADFile><!ENTITY x "unsafe"><layers/></CADFile>',
        )
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                read_cambam_bytes(payload)

    def test_rejects_non_utf8_and_utf16_xml(self):
        xml = self.project_bytes()
        for payload in (xml.decode("utf-8").encode("utf-16"), b"\xff" + xml):
            with self.subTest(payload=payload[:8]), self.assertRaises(ValueError):
                read_cambam_bytes(payload)

    def test_rejects_entity_collection_limits_before_reconstruction(self):
        primitives = (
            b'<CADFile><layers><layer name="Geometry"><objects>'
            + (b"<rect/>" * 10_001)
            + b"</objects></layer></layers></CADFile>"
        )
        with self.assertRaises(CamBamImportLimitError):
            read_cambam_bytes(primitives)

        mops = (
            b'<CADFile><layers><layer name="Geometry"/></layers><parts>'
            b'<part Name="Part"><machineops>'
            + (b"<profile/>" * 1_001)
            + b"</machineops></part></parts></CADFile>"
        )
        with self.assertRaises(CamBamImportLimitError):
            read_cambam_bytes(mops)

    def test_rejects_malformed_known_primitive_scalar(self):
        tree = ET.fromstring(self.project_bytes())
        tree.find("./layers/layer/objects/rect").set("w", "not-a-number")
        with self.assertRaises(ValueError):
            read_cambam_bytes(ET.tostring(tree, encoding="utf-8"))

    def test_rejects_failed_registry_add_from_duplicate_primitive_identity(self):
        project = CBProject("duplicate-input")
        layer = project.add_layer("Geometry")
        project.add_rect(layer, width=2, height=3, identifier="first")
        project.add_rect(layer, width=4, height=5, identifier="second")
        tree = build_xml_tree(project).getroot()
        primitives = tree.findall("./layers/layer/objects/*")
        first_tag = json.loads(primitives[0].findtext("Tag"))
        second_tag = json.loads(primitives[1].findtext("Tag"))
        second_tag["user_id"] = first_tag["user_id"]
        primitives[1].find("Tag").text = json.dumps(second_tag)
        with self.assertRaises(ValueError):
            read_cambam_bytes(ET.tostring(tree, encoding="utf-8"))

    def test_rejects_unresolved_mop_target(self):
        project = CBProject("unresolved-input")
        layer = project.add_layer("Geometry")
        rect = project.add_rect(layer, width=2, height=3, identifier="outline")
        part = project.add_part("Part")
        project.add_profile_mop(part, targets=[rect], identifier="profile", name="profile")
        tree = build_xml_tree(project).getroot()
        target = tree.find("./parts/part/machineops/profile/primitive/prim")
        target.text = "999999"
        with self.assertRaises(ValueError):
            read_cambam_bytes(ET.tostring(tree, encoding="utf-8"))

    def test_strict_rejects_unsupported_mop_legacy_file_reader_skips_it(self):
        payloads = (
            b'<CADFile Name="legacy"><layers><layer name="Geometry"/></layers>'
            b'<parts><part Name="Part"><machineops><laser/></machineops>'
            b'</part></parts></CADFile>',
            b'<CADFile Name="nameless"><layers><layer name="Geometry"/></layers>'
            b'<parts><part><machineops><laser/></machineops></part></parts></CADFile>',
        )
        for payload in payloads:
            with self.assertRaises(ValueError):
                read_cambam_bytes(payload)
        # The compatibility API remains warning-and-skip for the named case.
        payload = payloads[0]
        loaded = read_cambam_bytes(payload, strict=False)
        self.assertEqual(loaded.project_name, "legacy")
        self.assertEqual(loaded.list_mops(), [])


if __name__ == "__main__":
    unittest.main()
