"""Native Part stock presence is independent of dimensions and MOP Z values."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import build_xml_tree
from cambam_builder.cam_core import replay
from cambam_builder.integrations.cambam.native_series import _source_semantic


class StocklessNativeTests(unittest.TestCase):
    def make_stockless(self):
        project = CBProject("stockless")
        layer = project.add_layer("Geometry")
        part = project.add_part("Part", stock_width=0, stock_height=0,
                                stock_thickness=0, stock_material="")
        rect = project.add_rect(layer, identifier="shape", width=10, height=5)
        project.add_profile_mop(
            part, targets=[rect], name="Explicit", target_depth=3.5,
            depth_increment=0.5, stock_surface=4.5, clearance_plane=7,
            tool_diameter=2, cut_feedrate=100)
        root = build_xml_tree(project).getroot()
        root.find("./parts/part").remove(root.find("./parts/part/Stock"))
        root.find("./MachiningOptions").remove(
            root.find("./MachiningOptions/Stock"))
        root.find("./parts/part/machineops/profile/ClearancePlane").set(
            "state", "Default")
        return ET.tostring(root, encoding="utf-8")

    def test_stockless_import_save_reopen_preserves_explicit_mop_and_raw_state(self):
        source = self.make_stockless()
        project = read_cambam_bytes(source)
        self.assertFalse(project.get_part("Part").stock_present)
        mop = project.list_mops()[0]
        identity = (mop.internal_id, project.get_primitive("shape").internal_id)
        self.assertEqual((4.5, 3.5), (mop.stock_surface, mop.target_depth))
        self.assertEqual("Default", mop._xml_parameter_states["clearance_plane"])
        # The stock model belongs to the caller; supplying it cannot create a
        # Stock node or translate explicit native MOP Z coordinates.
        caller_stock = replay.Target("caller-stock", (0, 0, 10, 5), 1)
        self.assertEqual(1, caller_stock.depth)
        with tempfile.TemporaryDirectory() as folder:
            for index in range(2):
                path = Path(folder) / f"round-{index}.cb"
                project.save(str(path))
                root = ET.parse(path).getroot()
                self.assertIsNone(root.find("./parts/part/Stock"))
                self.assertIsNone(root.find("./MachiningOptions/Stock"))
                native_mop = root.find("./parts/part/machineops/profile")
                self.assertEqual("Value", native_mop.find("StockSurface").get("state"))
                self.assertEqual("4.5", native_mop.findtext("StockSurface"))
                self.assertEqual("3.5", native_mop.findtext("TargetDepth"))
                self.assertEqual("Default", native_mop.find("ClearancePlane").get("state"))
                project = read_cambam_bytes(path.read_bytes())
                self.assertFalse(project.get_part("Part").stock_present)
                mop = project.list_mops()[0]
                self.assertEqual(identity, (mop.internal_id,
                                            project.get_primitive("shape").internal_id))
                self.assertEqual((4.5, 3.5), (mop.stock_surface, mop.target_depth))

    def test_explicit_zero_and_new_default_stock_remain_present(self):
        project = CBProject("present")
        zero = project.add_part("Zero", stock_width=0, stock_height=0,
                                stock_thickness=0, stock_material="")
        default = project.add_part("Default")
        root = build_xml_tree(project).getroot()
        self.assertIsNotNone(root.find("./parts/part[@Name='Zero']/Stock"))
        self.assertIsNotNone(root.find("./parts/part[@Name='Default']/Stock"))
        loaded = read_cambam_bytes(ET.tostring(root, encoding="utf-8"))
        self.assertTrue(loaded.get_part(zero.user_identifier).stock_present)
        self.assertTrue(loaded.get_part(default.user_identifier).stock_present)
        self.assertEqual(0, loaded.get_part("Zero").stock_thickness)

    def test_stockless_auto_states_stay_unresolved_and_authored(self):
        root = ET.fromstring(self.make_stockless())
        mop = root.find("./parts/part/machineops/profile")
        mop.find("StockSurface").set("state", "Auto")
        mop.find("TargetDepth").set("state", "Auto")
        source = ET.tostring(root, encoding="utf-8")
        project = read_cambam_bytes(source)
        states = project.list_mops()[0]._xml_parameter_states
        self.assertEqual(states["stock_surface"], "Auto")
        self.assertEqual(states["target_depth"], "Auto")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "auto.cb"
            project.save(str(path))
            saved = ET.parse(path).getroot()
            self.assertIsNone(saved.find("./parts/part/Stock"))
            self.assertEqual(saved.find(
                "./parts/part/machineops/profile/StockSurface").get("state"),
                "Auto")
            self.assertEqual(saved.find(
                "./parts/part/machineops/profile/TargetDepth").get("state"),
                "Auto")

    def test_equal_placeholder_dimensions_do_not_hide_stock_presence(self):
        project = CBProject("same-dimensions")
        layer = project.add_layer("Geometry")
        project.add_part("Part", stock_width=100, stock_height=100,
                         stock_thickness=12.5, stock_material="Default")
        project.add_rect(layer, identifier="shape", width=10, height=5)
        root = build_xml_tree(project).getroot()
        present_data = ET.tostring(root, encoding="utf-8")
        part = root.find("./parts/part")
        part.remove(part.find("Stock"))
        absent_data = ET.tostring(root, encoding="utf-8")
        present = read_cambam_bytes(present_data)
        absent = read_cambam_bytes(absent_data)
        for field in ("stock_width", "stock_height", "stock_thickness",
                      "stock_material", "stock_surface", "stock_offset"):
            self.assertEqual(getattr(present.get_part("Part"), field),
                             getattr(absent.get_part("Part"), field))
        self.assertNotEqual(_source_semantic(present, present_data)[0],
                            _source_semantic(absent, absent_data)[0])

    def test_clone_copy_and_transfer_preserve_stock_absence(self):
        for transfer in (False, True):
            with self.subTest(transfer=transfer):
                source = read_cambam_bytes(self.make_stockless())
                self.assertFalse(source.clone().get_part("Part").stock_present)
                target = CBProject("target")
                method = (source.transfer_primitive_tree if transfer
                          else source.copy_primitive_tree)
                method("shape", target, include_mops=True)
                self.assertFalse(target.get_part("Part").stock_present)
                self.assertIsNone(build_xml_tree(target).find("./parts/part/Stock"))
                if not transfer:
                    self.assertFalse(source.get_part("Part").stock_present)


if __name__ == "__main__":
    unittest.main()
