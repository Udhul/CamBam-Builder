"""Native owner imports preserve the document interchange contract."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_bytes as legacy_read
from cambam_builder.cambam_writer import serialize_cambam_bytes as legacy_write
from cambam_builder.native.cad import Pline
from cambam_builder.native.cam import PocketMop
from cambam_builder.native.core import Primitive
from cambam_builder.native.project import CamBamProject
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.region import Region
from cambam_builder.native.writer import serialize_cambam_bytes


class NativeOwnerMigrationTests(unittest.TestCase):
    def test_region_mop_reference_and_stock_offset_survive_two_roundtrips_and_snapshot(self):
        self.assertIs(CBProject, CamBamProject)
        self.assertIs(legacy_read, read_cambam_bytes)
        self.assertIs(legacy_write, serialize_cambam_bytes)
        self.assertTrue(issubclass(Region, Primitive))

        project = CBProject("native-owner")
        layer = project.add_layer("Geometry")
        outer = Pline(vertices=[(0, 0), (30, 0), (30, 30), (0, 30)], closed=True)
        hole = Pline(vertices=[(10, 10), (20, 10), (20, 20), (10, 20)], closed=True)
        region = project.add_region(layer, outer, [hole], identifier="target-region")
        part = project.add_part(
            "Part", stock_width=40, stock_height=35, stock_thickness=6,
            machining_origin=(100, 200), stock_offset=(2.5, -3.0),
            stock_surface=1.0,
        )
        mop = project.add_pocket_mop(
            part, [region], identifier="rest-pocket", target_depth=-2.0,
            tool_diameter=3.0,
        )
        self.assertIsInstance(mop, PocketMop)

        current = project
        for _ in range(2):
            content = serialize_cambam_bytes(current)
            root = ET.fromstring(content)
            stock = root.find("./parts/part/Stock")
            self.assertEqual("2.5,-3.0,-5.0", stock.findtext("PMin"))
            self.assertEqual("42.5,32.0,1.0", stock.findtext("PMax"))
            self.assertEqual(
                (100.0, 200.0),
                tuple(map(float, root.findtext("./parts/part/MachiningOrigin").split(","))),
            )
            region_id = root.find("./layers/layer/objects/entity").get("id")
            self.assertEqual(
                region_id,
                root.findtext("./parts/part/machineops/pocket/primitive/prim"),
            )
            current = read_cambam_bytes(content, source_name="native-owner.cb")
            self.assertIsInstance(current.get_primitive("target-region"), Region)
            self.assertEqual(region.internal_id, current.get_primitive("target-region").internal_id)
            self.assertEqual(mop.internal_id, current.get_mop("rest-pocket").internal_id)
            self.assertEqual([region.internal_id], current.get_mop_targets("rest-pocket"))
            self.assertEqual((2.5, -3.0), current.get_part("Part").stock_offset)
            self.assertEqual((102.5, 197.0, 1.0), current.get_part("Part").stock_drawing_origin)

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "native-owner.pkl"
            current.save_state(str(snapshot))
            restored = CBProject.load_state(str(snapshot))
        self.assertIsInstance(restored, CamBamProject)
        self.assertIsInstance(restored.get_primitive("target-region"), Region)
        self.assertIs(restored.get_primitive("target-region").get_project(), restored)
        self.assertEqual([region.internal_id], restored.get_mop_targets("rest-pocket"))


if __name__ == "__main__":
    unittest.main()
