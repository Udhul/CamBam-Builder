"""Unsupported curved bakes must not publish partial subtree changes."""
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cambam_entities import Pline, Rect, Vertex
from cambam_builder.cambam_writer import build_xml_tree


class ParityBakeFailureTests(unittest.TestCase):
    def make_project(self):
        project = CBProject("atomic-bake")
        root = project.add_rect("Geometry", width=4, height=2, identifier="root",
                                elevation=2, local_z_offset=3)
        contour = Pline(vertices=[(0, 0, 4), Vertex(10, 0, 4, bulge=1),
                                  (10, 10, 4), (0, 10, 4)], closed=True)
        region = project.add_region("Geometry", contour, identifier="region", parent=root)
        self.assertIsNotNone(region)
        return project, root, region

    def assert_unchanged(self, project, root, before):
        self.assertIsInstance(root, Rect)
        self.assertIs(project.get_primitive("root"), root)
        self.assertEqual(ET.tostring(build_xml_tree(project).getroot()), before)

    def test_global_curved_bake_failure_preserves_whole_subtree(self):
        project, root, region = self.make_project()
        before = ET.tostring(build_xml_tree(project).getroot())
        with self.assertLogs("cambam_builder.cambam_project", level="ERROR"):
            self.assertFalse(project.transform_primitive(root, np.diag([2., 1., 1.]), bake=True))
        self.assert_unchanged(project, root, before)
        self.assertIs(project.get_primitive("region"), region)

    def test_full_curved_bake_failure_preserves_geometry_and_z_offsets(self):
        project, root, _ = self.make_project()
        root.effective_transform = np.array([[1., .4, 2.], [0., 1., 3.], [0., 0., 1.]])
        before = ET.tostring(build_xml_tree(project).getroot())
        with self.assertLogs("cambam_builder.cambam_project", level="ERROR"):
            self.assertFalse(project.bake_primitive_transform(root))
        self.assert_unchanged(project, root, before)
        self.assertEqual(root.local_z_offset, 3.)


if __name__ == "__main__":
    unittest.main()
