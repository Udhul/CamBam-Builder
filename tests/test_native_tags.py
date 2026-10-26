"""Native operator Tags coexist with framework identity and classification."""

import json
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder import CBProject
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


class NativeTagTests(unittest.TestCase):
    def make_project(self):
        project = CBProject("notes")
        layer = project.add_layer("Geometry")
        shape = project.add_rect(layer, width=30, height=20, identifier="outline",
                                 groups=["board"], description="framework description")
        part = project.add_part("Board")
        mop = project.add_profile_mop(part, [shape], identifier="cut")
        return project, shape, mop

    def test_native_plain_and_json_notes_survive_import_edit_and_reopen(self):
        texts = ["Inspect grain →\n  Face A & B <top>  ", "", ' ["setup", 2] ',
                 '{"user_id":"operator label","tool":"6mm"}',
                 '{"user_id":17}', '{"user_id":"cut","internal_id":"bad"}',
                 '{"user_id":"valid","internal_id":"00112233-4455-6677-8899-aabbccddeeff","operator":"keep"}',
                 '{"_cambam_builder":2,"native_tag":"future"}']
        for note in texts:
            with self.subTest(note=note):
                project, _, _ = self.make_project()
                root = ET.fromstring(serialize_cambam_bytes(project))
                root.find("./layers/layer/objects/*/Tag").text = note
                root.find("./parts/part/machineops/*/Tag").text = note
                loaded = read_cambam_bytes(ET.tostring(root))
                shape, mop = loaded.list_primitives()[0], loaded.list_mops()[0]
                self.assertEqual(note, shape.native_tag)
                self.assertEqual(note, mop.native_tag)
                shape.width = 45
                mop.target_depth = -4
                for _ in range(2):
                    data = serialize_cambam_bytes(loaded)
                    tree = ET.fromstring(data)
                    self.assertEqual(1, len(tree.findall("./layers/layer/objects/*/Tag")))
                    self.assertEqual(1, len(tree.findall("./parts/part/machineops/*/Tag")))
                    loaded = read_cambam_bytes(data)
                    self.assertEqual(note, loaded.list_primitives()[0].native_tag)
                    self.assertEqual(note, loaded.list_mops()[0].native_tag)
                    self.assertEqual(45, loaded.list_primitives()[0].width)
                    self.assertEqual(-4, loaded.list_mops()[0].target_depth)

    def test_authored_notes_metadata_edit_copy_and_rect_fallback(self):
        project, shape, mop = self.make_project()
        # Deliberately resembles the reserved metadata envelope: property values
        # remain opaque, so callers can still author every possible text value.
        note = json.dumps({"user_id": "collision", "internal_id": str(shape.internal_id),
                           "_cambam_builder": 1, "native_tag": "nested"})
        shape.native_tag = note
        mop.native_tag = "Cut after groove"
        shape.effective_transform = np.array([[1., .2, 0.], [0., 1., 0.], [0., 0., 1.]])
        loaded = read_cambam_bytes(serialize_cambam_bytes(project))
        self.assertEqual(note, loaded.get_primitive("outline").native_tag)
        self.assertEqual(["board"], loaded.get_primitive("outline").groups)
        self.assertEqual("framework description", loaded.get_primitive("outline").description)
        self.assertEqual(shape.internal_id, loaded.get_primitive("outline").internal_id)
        copied = CBProject("copied")
        loaded.copy_primitive_tree("outline", copied, preserve_ids=False)
        copied = read_cambam_bytes(serialize_cambam_bytes(copied))
        self.assertEqual(note, copied.get_primitive("outline").native_tag)
        self.assertEqual("Cut after groove", copied.get_mop("cut").native_tag)
        self.assertNotEqual(shape.internal_id, copied.get_primitive("outline").internal_id)
        copied.get_primitive("outline").native_tag = "edited"
        copied.get_mop("cut").native_tag = None
        again = read_cambam_bytes(serialize_cambam_bytes(copied))
        self.assertEqual("edited", again.get_primitive("outline").native_tag)
        self.assertIsNone(again.get_mop("cut").native_tag)

    def test_legacy_identity_and_user_id_only_remain_metadata(self):
        project, shape, mop = self.make_project()
        root = ET.fromstring(serialize_cambam_bytes(project))
        loaded = read_cambam_bytes(ET.tostring(root))
        self.assertEqual(shape.internal_id, loaded.get_primitive("outline").internal_id)
        self.assertEqual(mop.internal_id, loaded.get_mop("cut").internal_id)
        self.assertIsNone(loaded.get_primitive("outline").native_tag)
        root.find("./layers/layer/objects/*/Tag").text = '{"user_id":"old-label"}'
        loaded = read_cambam_bytes(ET.tostring(root))
        self.assertIsNotNone(loaded.get_primitive("old-label"))
        self.assertIsNone(loaded.get_primitive("old-label").native_tag)

    def test_invalid_operator_note_type_is_rejected_on_export(self):
        for target in ("primitive", "mop"):
            with self.subTest(target=target):
                project, shape, mop = self.make_project()
                (shape if target == "primitive" else mop).native_tag = {"not": "text"}
                with self.assertRaisesRegex(ValueError, "native_tag must be a string or None"):
                    serialize_cambam_bytes(project)

    def test_root_layer_and_part_tags_survive_as_native_context(self):
        project, _, _ = self.make_project()
        root = ET.fromstring(serialize_cambam_bytes(project))
        for path in (".", "./layers/layer", "./parts/part"):
            ET.SubElement(root.find(path), "Tag", {"source": "native"}).text = "note " + path
        loaded = read_cambam_bytes(ET.tostring(root))
        loaded.get_layer("Geometry").color = "Blue"
        loaded.get_part("Board").stock_material = "birch"
        for _ in range(2):
            root = ET.fromstring(serialize_cambam_bytes(loaded))
            for path in (".", "./layers/layer", "./parts/part"):
                tags = root.find(path).findall("Tag")
                self.assertEqual(1, len(tags))
                self.assertEqual("note " + path, tags[0].text)
                self.assertEqual({"source": "native"}, tags[0].attrib)
            loaded = read_cambam_bytes(ET.tostring(root))


if __name__ == "__main__":
    unittest.main()
