"""Native mesh/MOP ownership across edit, placement and document boundaries."""

import unittest
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder import CBProject, Vertex
from cambam_builder.native.cam import Surface3DMop
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.surface import Surface
from cambam_builder.native.writer import serialize_cambam_bytes


class NativeReliefInterchangeTests(unittest.TestCase):
    def test_edit_roundtrip_copy_and_bake_preserve_mesh_mop_and_relationships(self):
        project = CBProject("relief component")
        layer = project.add_layer("Design")
        parent = project.add_line(layer, [(0, 0), (4, 0)], identifier="frame")
        mesh = project.add_surface(
            layer, [(0, 0, -1), (4, 0, -0.5), (4, 3, -0.25), (0, 3, -0.75)],
            [(0, 1, 2), (0, 2, 3)], identifier="mesh", parent=parent,
            groups=["relief"], display_edges=True)
        mesh.native_tag = "Preserve relief detail"
        part = project.add_part("Front", stock_width=10, stock_height=10,
                                stock_thickness=2, stock_offset=(10, 20))
        finish = project.add_surface3d_mop(
            part, identifier="finish", target_group="relief", tool_diameter=1,
            target_depth=-1, roughing_clearance=0.02, cut_feedrate=200)
        self.assertIsInstance(mesh, Surface)
        self.assertIsInstance(finish, Surface3DMop)
        self.assertTrue(project.rotate_primitive_deg(parent, 90, cx=0, cy=0))
        self.assertTrue(project.translate_primitive(parent, 10, 20))
        self.assertTrue(project.translate_primitive_z(parent, 2))

        edited = read_cambam_bytes(serialize_cambam_bytes(project))
        edited.get_primitive("mesh").vertices[0] = Vertex(0, 0, -1.25)
        edited.get_mop("finish").stepover = 0.15
        expected = {"vertices": [(10, 20, 0.75), (10, 24, 1.5),
                                 (7, 24, 1.75), (7, 20, 1.25)],
                    "faces": [(0, 1, 2), (0, 2, 3)]}
        for _ in range(2):
            data = serialize_cambam_bytes(edited)
            xml = ET.fromstring(data)
            mesh_id = xml.find("./layers/layer/objects/surface").get("id")
            self.assertEqual(mesh_id, xml.findtext("./parts/part/machineops/surface3d/primitive/prim"))
            edited = read_cambam_bytes(data)
            current = edited.get_primitive("mesh")
            np.testing.assert_allclose(current.get_absolute_coordinates_xyz()["vertices"],
                                       expected["vertices"], atol=1e-9)
            self.assertEqual(current.get_absolute_coordinates_xyz()["faces"], expected["faces"])
            self.assertEqual(current.internal_id, mesh.internal_id)
            self.assertEqual(current.native_tag, "Preserve relief detail")
            self.assertTrue(current.display_edges)
            self.assertIs(edited.get_parent_of_primitive(current), edited.get_primitive("frame"))
            self.assertEqual(edited.get_groups_of_primitive(current), ["relief"])
            self.assertEqual(edited.get_mop_targets("finish"), [mesh.internal_id])
            self.assertEqual(edited.get_mop("finish").internal_id, finish.internal_id)
            self.assertEqual(edited.get_mop("finish").stepover, 0.15)
            self.assertEqual(edited.get_part("Front").stock_offset, (10, 20))

        clone = edited.clone()
        clone.get_primitive("mesh").vertices[0].z = 99
        self.assertEqual(edited.get_primitive("mesh").vertices[0].z, -1.25)
        copied = CBProject("copied relief")
        mapping = edited.copy_primitive_tree("frame", copied, preserve_ids=False)
        copied_mesh = copied.get_primitive(mapping[mesh.internal_id])
        self.assertIsNot(copied_mesh, edited.get_primitive("mesh"))
        self.assertEqual(copied.get_mop_targets("finish"), [copied_mesh.internal_id])
        self.assertTrue(copied.bake_primitive_transform("frame"))
        reopened = read_cambam_bytes(serialize_cambam_bytes(copied))
        np.testing.assert_allclose(reopened.get_primitive("mesh").get_absolute_coordinates_xyz()["vertices"],
                                   expected["vertices"], atol=1e-9)

    def test_invalid_mesh_adder_does_not_register_partial_entity(self):
        project = CBProject("invalid relief")
        layer = project.add_layer("Geometry")
        result = project.add_surface(layer, [(0, 0), (1, 0), (0, 1)],
                                     [(0, 1, 3)], identifier="invalid")
        self.assertIsNone(result)
        self.assertEqual(project.list_primitives(), [])
        self.assertIsNone(project.get_entity("invalid"))


if __name__ == "__main__":
    unittest.main()
