"""Independent native bitmap schema/placement and no-implicit-image-I/O witnesses."""

from copy import deepcopy
import unittest
from unittest.mock import patch
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder import CBProject
from cambam_builder.native.bitmap import CADBitmap, bitmap_geometry_from_xml
from cambam_builder.native.reader import CamBamReaderError, read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


NATIVE = '''<bitmap id="7" p="3,5,0" w="40" h="20">
<ModificationCount>0</ModificationCount><Tag>operator artwork</Tag><mat m="Identity"/>
<SourceFile>artwork.png</SourceFile><DisplayBitmap>false</DisplayBitmap>
<DisplayVectors>false</DisplayVectors><Invert>true</Invert><Threshold>0.6</Threshold>
<Smoothing>true</Smoothing><SmoothPixels>2</SmoothPixels></bitmap>'''


class NativeBitmapTests(unittest.TestCase):
    def bitmap(self, **changes):
        values = dict(user_identifier="reference", source_file="missing image.png",
                      position=(3, 5, 0), width=40, height=20)
        values.update(changes)
        return CADBitmap(**values)

    def test_independent_native_schema_preserves_all_modeled_fields(self):
        definition = bitmap_geometry_from_xml(ET.fromstring(NATIVE))
        bitmap = CADBitmap(user_identifier="native", **definition)
        self.assertEqual(definition, bitmap_geometry_from_xml(bitmap.to_xml_element(1, uuid.uuid4())))
        # Native GetExtents observed these corners for an 8x4 PNG at 40x20.
        self.assertEqual([(3, 5, 0), (43, 5, 0), (43, 25, 0), (3, 25, 0)],
                         bitmap.get_absolute_coordinates_xyz())
        self.assertEqual((23, 15), bitmap.get_geometric_center())

    def test_save_load_clone_parent_identity_and_tags_need_no_image(self):
        project = CBProject("artwork-job")
        layer = project.add_layer("Artwork")
        parent = project.add_rect(layer, identifier="frame", width=40, height=20)
        parent.effective_transform = np.array([[1., 0., 10.], [0., 1., 12.], [0., 0., 1.]])
        parent.local_z_offset = 2
        bitmap = project.add_bitmap(layer, "nonexistent/path.png", identifier="reference",
                                    parent=parent, position=(3, 5, 1), width=40, height=20)
        bitmap.native_tag = "Operator reference"
        bitmap.effective_transform = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        expected = [tuple(v) for v in ((5, 15, 3), (5, 55, 3), (-15, 55, 3), (-15, 15, 3))]
        with patch("pathlib.Path.read_bytes", side_effect=AssertionError("implicit image read")):
            data = serialize_cambam_bytes(project)
            reopened = read_cambam_bytes(data, strict=True)
            cloned = reopened.clone()
            for candidate in (reopened, cloned):
                reference = candidate.get_primitive("reference")
                self.assertIsInstance(reference, CADBitmap)
                self.assertEqual(reference.internal_id, bitmap.internal_id)
                self.assertEqual(reference.native_tag, "Operator reference")
                self.assertEqual(reference.source_file, "nonexistent/path.png")
                self.assertEqual(reference.get_absolute_coordinates_xyz(), expected)
                self.assertEqual(candidate.get_parent_of_primitive(reference).internal_id, parent.internal_id)

    def test_affine_retained_but_lossy_orientation_bakes_reject_atomically(self):
        bitmap = self.bitmap()
        bitmap.effective_transform = np.array([[-1., 2., 9.], [0., 1., 7.], [0., 0., 1.]])
        before = deepcopy(bitmap.__dict__)
        bitmap.to_xml_element(1, None)
        with self.assertRaises(ValueError):
            bitmap.bake_geometry()
        self.assertEqual(bitmap.position, before["position"])
        np.testing.assert_array_equal(bitmap.effective_transform, before["effective_transform"])
        bitmap.effective_transform = np.array([[2., 0., 3.], [0., 3., 4.], [0., 0., 1.]])
        expected = bitmap.get_absolute_coordinates_xyz()
        bitmap.bake_geometry()
        self.assertEqual(bitmap.get_absolute_coordinates_xyz(), expected)
        bitmap.shift_geometry_z(2)
        self.assertEqual(bitmap.position, (9, 19, 2))

        overflowing = self.bitmap(position=(-9e307, 0, 0), width=1.7e308, height=1)
        overflowing.effective_transform = np.diag((1.1, 1., 1.))
        before = deepcopy(overflowing.__dict__)
        with self.assertRaises(ValueError):
            overflowing.bake_geometry()
        self.assertEqual(overflowing.width, before["width"])
        self.assertEqual(overflowing.position, before["position"])
        np.testing.assert_array_equal(overflowing.effective_transform, before["effective_transform"])

    def test_invalid_geometry_and_lossy_encodings_reject(self):
        for change in (dict(source_file=""), dict(source_file="a\n.png"), dict(position=(1, 2)),
                       dict(width=0), dict(height=-1), dict(width=float("nan")),
                       dict(display_bitmap=1), dict(threshold=1.1), dict(smooth_pixels=-1),
                       dict(smooth_pixels=1e300)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.bitmap(**change)
        invalid = [NATIVE.replace(' w="40"', ' w="0"'),
                   NATIVE.replace('<SourceFile>', '<SourceFile bad="1">'),
                   NATIVE.replace('</bitmap>', '<Unknown>1</Unknown></bitmap>'),
                   NATIVE.replace('</bitmap>', '<SourceFile>x.png</SourceFile></bitmap>'),
                   NATIVE.replace('>false</DisplayVectors>', '>maybe</DisplayVectors>')]
        for xml in invalid:
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                CADBitmap(user_identifier="invalid", **bitmap_geometry_from_xml(ET.fromstring(xml)))
        bitmap = self.bitmap(width=1e-12)
        with self.assertRaises(ValueError):
            bitmap.to_xml_element(1, None)


if __name__ == "__main__":
    unittest.main()
