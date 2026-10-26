"""Self-contained NR03 mesh, native fidelity and independent posted-stock tests."""

from dataclasses import replace
import unittest
import xml.etree.ElementTree as ET

from cambam_builder.cam_core.relief import HeightField
from cambam_builder.integrations.cambam.native_relief import (
    ReliefSettings, audit_post, create_document, mesh_relief,
)
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


def flat() -> HeightField:
    return HeightField((0.0, 1.0), (0.0, 1.0), ((-0.5, -0.5), (-0.5, -0.5)), 1.2)


def synthetic_post() -> bytes:
    """Synthetic decoder/replay test input, never native-generation evidence."""
    lines = ["( Made using CamBam - http://www.cambam.co.uk )", "( synthetic test )",
             "( Post processor: Default )", "G21 G90", "G0 Z2", "( T1 : 0.5 )",
             "T1 M6", "G17", "( Relief finish )", "G61", "S10000 M3"]
    for i in range(11):
        lines.extend((f"G0 X0 Y{i / 10}", "G1 Z-0.48 F120", "G1 X1 F240", "G0 Z2"))
    lines.extend(("M5", "M30"))
    return "\n".join(lines).encode()


class NativeReliefTests(unittest.TestCase):
    def test_bilinear_triangulation_error_and_winding(self):
        target = HeightField((0.0, 2.0), (0.0, 3.0), ((-0.4, -0.8), (-0.9, -0.3)), 1.2)
        coarse, fine = mesh_relief(target), mesh_relief(target, subdivisions=2)
        self.assertAlmostEqual(coarse.maximum_error_mm, 0.25, places=8)
        self.assertAlmostEqual(fine.maximum_error_mm, 0.0625, places=8)
        self.assertEqual((len(fine.vertices), len(fine.faces)), (9, 8))
        for mesh in (coarse, fine):
            for ia, ib, ic in mesh.faces:
                a, b, c = (mesh.vertices[index] for index in (ia, ib, ic))
                self.assertGreater((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]), 0)
                for u in range(11):
                    for v in range(11-u):
                        wa, wb, wc = u/10, v/10, 1-(u+v)/10
                        p = tuple(wa*a[k]+wb*b[k]+wc*c[k] for k in range(3))
                        self.assertLessEqual(abs(p[2]-target.height_at(*p[:2])), mesh.maximum_error_mm)

    def test_mesh_validation_and_allowance(self):
        for value in (0, -1, True, 1.2, 1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mesh_relief(flat(), subdivisions=value)
        with self.assertRaisesRegex(ValueError, "allowance"):
            create_document(flat(), settings=ReliefSettings(allowance_mm=0.0001))
        for xs in ((1e16, 1e16 + 2), (0., 1e-10)):
            with self.subTest(xs=xs), self.assertRaisesRegex(ValueError, "coordinate resolution"):
                mesh_relief(HeightField(xs, (0., 1.), ((-.5, -.5), (-.5, -.5)), 1), subdivisions=2)

    def test_mesh_error_includes_large_coordinate_arithmetic(self):
        height = -100000000.12345679
        target = HeightField((0., 1.), (0., 1.), ((height, height), (height, height)), 100000001.)
        mesh = mesh_relief(target, subdivisions=3)
        self.assertTrue(all(abs(p[2] - height) <= mesh.maximum_error_mm for p in mesh.vertices))

    def test_document_roundtrip_edit_and_freshness(self):
        document = create_document(flat())
        project = read_cambam_bytes(document.xml_bytes)
        document.check_document(serialize_cambam_bytes(project))
        project.get_mop("relief-finish").cut_feedrate = 180.0
        changed = serialize_cambam_bytes(project)
        self.assertEqual(read_cambam_bytes(changed).get_mop("relief-finish").cut_feedrate, 180)
        with self.assertRaisesRegex(ValueError, "changed"):
            document.check_document(changed)
        target = HeightField((0.0, 1.0), (0.0, 1.0), ((-0.4, -0.4), (-0.4, -0.4)), 1.2)
        with self.assertRaisesRegex(ValueError, "stale"):
            replace(document, target=target).check_document()
        for forged in (
            replace(document, settings=replace(document.settings, radius_mm=0.5)),
            replace(document, settings=replace(document.settings, cut_feed_mm_min=180)),
            replace(document, mesh=replace(document.mesh, maximum_error_mm=0)),
            replace(document, xml_bytes=changed),
            replace(document, target=target, mesh=mesh_relief(target, subdivisions=2)),
            replace(document, initial_tip=(0, 0, 2)),
        ):
            with self.subTest(forged=forged.settings), self.assertRaises(ValueError):
                forged.check_document()

    def test_complete_post_stock_and_native_retracts(self):
        document = create_document(flat())
        result = audit_post(document, synthetic_post(), max_residual_volume_mm3=0.2,
                            max_residual_thickness_mm=0.2)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["rapid_retract_count"], 11)
        prefix = result["stock"]["prefixes"][-1]
        self.assertEqual(prefix["protected_overcut_upper_mm3"], 0)
        self.assertLess(prefix["maximum_residual_thickness_mm"][1], 0.2)
        self.assertGreater(prefix["newly_removed_volume_lower_mm3"], 0.3)

    def test_native_save_equivalences_keep_unknown_process_state_closed(self):
        document = create_document(flat())
        root = ET.fromstring(document.xml_bytes)
        root.attrib.pop("units")  # Observed native XML metric default.
        root.set("Name", "saved-file")
        layer = root.find("./layers/layer")
        for name in ("alpha", "visible", "locked"):
            layer.attrib.pop(name)
        layer.set("color", "0,128,0")
        surface = layer.find("./objects/surface")
        surface.find("ModificationCount").text = "2"
        surface.find("mat").set("m", "Identity")
        ET.SubElement(layer, "ModificationCount").text = "0"
        part = root.find("./parts/part")
        part.attrib.pop("Enabled")
        part.find("./Stock/PMax").text = "1,1"
        ET.SubElement(part.find("Nesting"), "BasePoint")
        options = root.find("MachiningOptions")
        ET.SubElement(options, "ToolProfile").text = "Unspecified"
        stock = ET.SubElement(options, "Stock")
        for tag, value in (("PMin", "0,0"), ("PMax", "0,0"), ("Color", "255,165,0")):
            ET.SubElement(stock, tag).text = value
        mop = part.find("./machineops/surface3d")
        for child in mop:
            if child.text in ("true", "false"):
                child.text = child.text.title()
        mop[:] = list(reversed(mop))
        candidate = ET.tostring(root)
        document.check_document(candidate)
        document.check_document(serialize_cambam_bytes(read_cambam_bytes(candidate)))
        ET.SubElement(mop, "UnmodeledMachiningEffect").text = "1"
        with self.assertRaises(ValueError):
            document.check_document(ET.tostring(root))
        mop.remove(mop.find("UnmodeledMachiningEffect"))
        mop.find("RoughingClearance").set("state", "Default")
        with self.assertRaises(ValueError):
            document.check_document(ET.tostring(root))

    def test_gouge_low_travel_wrong_state_and_incomplete_post_reject(self):
        document = create_document(flat())
        source = synthetic_post()
        variants = {
            "gouge": source.replace(b"Z-0.48", b"Z-0.6"),
            "low rapid": source.replace(b"G1 X1 F240", b"G0 X1"),
            "feed": source.replace(b"F240", b"F241"),
            "spindle": source.replace(b"S10000", b"S9000"),
            "tool": source.replace(b"T1 M6", b"T2 M6"),
            "blend": source.replace(b"G61", b"G64"),
            "implicit blend": source.replace(b"G61\n", b""),
            "curved interior": source.replace(b"G1 X1 F240", b"G2 X1 I0.5 J0 F240"),
            "truncated": source.replace(b"M30", b""),
            "extra section": source.replace(b"M5", b"( other operation )\nM5"),
            "command after end": source + b"\nG0 X1",
        }
        for name, post in variants.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit_post(document, post, max_residual_volume_mm3=1,
                           max_residual_thickness_mm=1)
        with self.assertRaisesRegex(ValueError, "residual"):
            audit_post(document, source, max_residual_volume_mm3=0,
                       max_residual_thickness_mm=0)


if __name__ == "__main__":
    unittest.main()
