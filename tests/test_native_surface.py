"""Independent native mesh schema, geometry/state and rejection witnesses."""

from copy import deepcopy
from fractions import Fraction
import unittest
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder.native.core import Vertex
from cambam_builder.native.surface import Surface, surface_geometry_from_xml
from cambam_builder.native.tags import decode_tag
from cambam_builder.native.transformations import identity_matrix


_NATIVE_XML = """<surface id="41"><ModificationCount>17</ModificationCount>
<Tag>native surface witness</Tag><mat m="Identity" />
<verts><v>0,0,-3</v><v>8,0,-2</v><v>8,6,-1</v><v>0,6,-2</v></verts>
<faces><f>0,1,2</f><f>0,2,3</f></faces><DisplayEdges>true</DisplayEdges></surface>"""


class NativeSurfaceTests(unittest.TestCase):
    def surface(self, **changes):
        values = dict(user_identifier="relief", vertices=[(0, 0, -3), (8, 0, -2),
                                                         (8, 6, -1), (0, 6, -2)],
                      faces=[(0, 1, 2), (0, 2, 3)])
        values.update(changes)
        return Surface(**values)

    def test_independent_native_schema_preserves_mesh_display_and_common_identity(self):
        definition = surface_geometry_from_xml(ET.fromstring(_NATIVE_XML))
        surface = Surface(user_identifier="native", native_tag="operator note", **definition)
        parent = uuid.uuid4()
        element = surface.to_xml_element(71, parent)
        self.assertEqual(["0.0,0.0,-3.0", "8.0,0.0,-2.0", "8.0,6.0,-1.0", "0.0,6.0,-2.0"],
                         [v.text for v in element.findall("verts/v")])
        self.assertEqual(["0,1,2", "0,2,3"], [f.text for f in element.findall("faces/f")])
        self.assertEqual("true", element.findtext("DisplayEdges"))
        self.assertEqual("0", element.findtext("ModificationCount"))
        self.assertEqual("71", element.get("id"))
        metadata, note = decode_tag(element.findtext("Tag"), primitive=True)
        self.assertEqual(str(parent), metadata["parent"])
        self.assertEqual(str(surface.internal_id), metadata["internal_id"])
        self.assertEqual("operator note", note)
        self.assertEqual(definition, surface_geometry_from_xml(element))

    def test_world_affine_reflection_and_bake_match_native_winding(self):
        surface = self.surface()
        surface.effective_transform = np.array([[-2., 1., 10.], [0., 3., -4.], [0., 0., 1.]])
        surface.local_z_offset = 7
        expected = {"vertices": [(10, -4, 4), (-6, -4, 5), (0, 14, 6), (16, 14, 5)],
                    "faces": [(0, 2, 1), (0, 3, 2)]}
        self.assertEqual(expected, surface.get_absolute_coordinates_xyz())
        self.assertEqual(expected, surface.get_absolute_coordinates())
        bounds = surface.get_bounding_box()
        self.assertEqual((-6, -4, 16, 14), (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y))
        self.assertEqual((5, 5), surface.get_geometric_center())
        surface.bake_geometry()
        self.assertEqual(expected, surface.get_absolute_coordinates_xyz())
        np.testing.assert_array_equal(identity_matrix(), surface.effective_transform)
        surface.shift_geometry_z(7)
        surface.local_z_offset = 0
        self.assertEqual(expected, surface.get_absolute_coordinates_xyz())

    def test_entity_is_general_mesh_not_only_relief(self):
        # A vertical triangle is a valid native mesh, even though it is not a heightfield.
        surface = self.surface(vertices=[(0, 0, 0), (0, 2, 0), (0, 0, 3)], faces=[(0, 1, 2)])
        self.assertEqual([(0, 1, 2)], surface.get_absolute_coordinates_xyz()["faces"])
        surface.to_xml_element(1, None)

    def test_invalid_meshes_reject(self):
        changes = [dict(vertices=[]), dict(vertices=[(0, 0), (1, 1)]), dict(faces=[]),
                   dict(faces=[(0, 1)]), dict(faces=[(0, 1, 4)]), dict(faces=[(-1, 1, 2)]),
                   dict(faces=[(0, 0, 2)]), dict(faces=[(0, True, 2)]), dict(faces=[(0, 1.0, 2)]),
                   dict(vertices=[(0, 0), (1, 1), (2, 2)], faces=[(0, 1, 2)]),
                   dict(vertices=[(0, 0), (1, 0), (float("nan"), 1)], faces=[(0, 1, 2)]),
                   dict(vertices=[Vertex(0, 0, bulge=1), (1, 0), (0, 1)], faces=[(0, 1, 2)]),
                   dict(display_edges=1)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.surface(**change)

    def test_mutation_revalidates_and_failed_bake_is_atomic(self):
        surface = self.surface()
        before = deepcopy(surface)
        with self.assertRaises(ValueError):
            surface.bake_geometry(np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]))
        self.assertEqual(before.vertices, surface.vertices)
        self.assertEqual(before.faces, surface.faces)
        np.testing.assert_array_equal(before.effective_transform, surface.effective_transform)
        surface.faces[1] = (0, 2, 999)
        for action in (lambda: surface.to_xml_element(1, None), surface.get_absolute_coordinates_xyz,
                       surface.bake_geometry):
            with self.assertRaises(ValueError):
                action()

    def test_mutated_numeric_values_normalize_without_float32_loss(self):
        surface = self.surface()
        surface.output_decimals = None
        surface.vertices[0].x = Fraction(1, 3)
        surface.vertices[0].y = "0.12345678901234566"
        surface.vertices[0].z = False
        result = surface.get_absolute_coordinates_xyz()["vertices"][0]
        self.assertEqual((1 / 3, 0.12345678901234566, 0.), result)
        element = surface.to_xml_element(1, None)
        self.assertEqual("0.3333333333333333,0.12345678901234566,0.0", element.findtext("verts/v"))
        self.assertEqual(result, tuple(getattr(surface_geometry_from_xml(element)["vertices"][0], name)
                                       for name in ("x", "y", "z")))

    def test_export_rejects_rounding_that_collapses_a_triangle(self):
        surface = self.surface(vertices=[(0, 0), (1e-12, 0), (0, 1e-12)], faces=[(0, 1, 2)])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            surface.to_xml_element(1, None)
        surface.output_decimals = None
        surface.to_xml_element(1, None)

    def test_parser_rejects_unsupported_and_ambiguous_encodings(self):
        def set_text(path, value):
            return lambda e: setattr(e.find(path), "text", value)

        mutations = [lambda e: e.append(deepcopy(e.find("verts"))),
                     lambda e: e.remove(e.find("faces")),
                     lambda e: e.set("encoding", "binary"),
                     lambda e: ET.SubElement(e, "Normals"),
                     lambda e: e.find("verts/v").set("b", "1"),
                     lambda e: ET.SubElement(e.find("verts/v"), "x"),
                     lambda e: e.find("faces").set("encoding", "base64"),
                     set_text("verts/v", "0,0"), set_text("verts/v", "0,0,NaN"),
                     set_text("faces/f", "0,1.0,2"), set_text("faces/f", "0,1,99"),
                     set_text("DisplayEdges", "True"), set_text("ModificationCount", "1.5"),
                     set_text("ModificationCount", str(2**63)),
                     lambda e: setattr(e.find("faces/f"), "tail", "unmodeled")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                element = ET.fromstring(_NATIVE_XML)
                mutate(element)
                surface_geometry_from_xml(element)

    def test_optional_display_defaults_and_native_boolean_variants(self):
        element = ET.fromstring(_NATIVE_XML)
        element.remove(element.find("DisplayEdges"))
        element.remove(element.find("ModificationCount"))
        self.assertFalse(surface_geometry_from_xml(element)["display_edges"])
        state = ET.SubElement(element, "DisplayEdges")
        for text, expected in (("true", True), ("1", True), ("false", False), ("0", False)):
            state.text = text
            self.assertEqual(expected, surface_geometry_from_xml(element)["display_edges"])


if __name__ == "__main__":
    unittest.main()
