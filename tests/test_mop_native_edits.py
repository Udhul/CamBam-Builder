"""Synthetic native edits must outrank framework identity/group metadata."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import build_xml_tree


class NativeMopEditTests(unittest.TestCase):
    def test_metadata_free_and_malformed_native_edits_and_new_operation(self):
        output = Path("output").resolve()
        output.mkdir(exist_ok=True)
        for metadata in (None, "not JSON", '{"internal_id": 12, "user_id": []}'):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory(
                    prefix="mop-native-", dir=output) as directory:
                source = CBProject("native-edits")
                layer = source.add_layer("Geometry")
                part = source.add_part("Part1")
                first = source.add_rect(layer, identifier="first", groups=["live"])
                second = source.add_rect(layer, identifier="second", corner=(20, 0))
                source.add_profile_mop(part, target_group="live", name="Original")
                source.add_pocket_mop(part, [first], name="Pocket")
                tree = build_xml_tree(source)
                root = tree.getroot()
                shapes = root.findall("./layers/layer/objects/*")
                # Native documents have no framework primitive metadata either.
                first_id = next(e.get("id") for e in shapes if json.loads(e.findtext("Tag"))["user_id"] == "first")
                second_id = next(e.get("id") for e in shapes if json.loads(e.findtext("Tag"))["user_id"] == "second")
                for shape in shapes:
                    shape.remove(shape.find("Tag"))
                ops = root.find("./parts/part/machineops")
                profile, pocket = list(ops)
                for mop in (profile, pocket):
                    tag = mop.find("Tag")
                    if metadata is None:
                        mop.remove(tag)
                    else:
                        tag.text = metadata
                profile.find("primitive/prim").text = second_id
                profile.find("CutFeedrate").text = "450"
                profile.find("ClearancePlane").set("state", "Default")
                profile.find("ClearancePlane").text = "8.5"
                ops.remove(pocket)
                ops.insert(0, pocket)
                native = ET.SubElement(ops, "engrave", Enabled="true")
                ET.SubElement(native, "Name").text = "Native"
                ET.SubElement(ET.SubElement(native, "primitive"), "prim").text = first_id
                for index in range(2):
                    path = Path(directory) / f"round-{index}.cb"
                    tree.write(path, encoding="utf-8")
                    loaded = read_cambam_file(str(path))
                    self.assertIsNotNone(loaded)
                    mops = loaded.list_mops()
                    self.assertEqual(["Pocket", "Original", "Native"], [m.name for m in mops])
                    self.assertEqual(450, mops[1].cut_feedrate)
                    target = loaded.get_primitive(loaded.get_mop_targets(mops[1])[0])
                    self.assertEqual((20.0, 0.0), target.relative_corner)
                    native_target = loaded.get_primitive(loaded.get_mop_targets(mops[2])[0])
                    self.assertEqual((0.0, 0.0), native_target.relative_corner)
                    self.assertTrue(all(loaded.get_mop_target_group(m) is None for m in mops))
                    tree = build_xml_tree(loaded)
                    self.assertEqual("Default", tree.find("./parts/part/machineops/profile/ClearancePlane").get("state"))


if __name__ == "__main__":
    unittest.main()
