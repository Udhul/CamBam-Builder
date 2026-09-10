"""Focused entity-level contracts for Z coordinates on existing shapes."""

import math
import pickle
import unittest

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cad_transformations import (
    from_cambam_matrix_str,
    mirror_y_matrix,
    rotation_matrix_deg,
    skew_matrix,
    translation_matrix,
)
from cambam_builder.cambam_entities import Arc, Circle, Pline, Points, Rect, Text, Vertex


def xyz(text):
    return tuple(float(value) for value in text.split(','))


class ExistingShapeElevationTests(unittest.TestCase):
    def test_small_xy_bakes_are_not_discarded_as_identity(self):
        for shape in (Circle(), Arc(), Points(vertices=[(0, 0)]), Text()):
            with self.subTest(shape=type(shape).__name__):
                shape.bake_geometry(translation_matrix(5e-9, -5e-9))
                result = shape.get_absolute_coordinates_xyz()
                point = (result[0] if isinstance(result, list) else
                         result.get("center", result.get("position")))
                np.testing.assert_allclose(point[:2], (5e-9, -5e-9), rtol=0, atol=1e-15)

    def test_xyz_queries_report_reflected_curves_and_reject_elliptical_forms(self):
        arc = Arc(start_angle=30, extent_angle=60, elevation=4,
                  effective_transform=mirror_y_matrix())
        result = arc.get_absolute_coordinates_xyz()
        self.assertAlmostEqual(result["start_angle"], 150)
        self.assertEqual(result["extent_angle"], -60)
        arc.bake_geometry()
        self.assertEqual(arc.get_absolute_coordinates_xyz(), result)
        pline = Pline(vertices=[Vertex(0, 0, 3, bulge=.5), (1, 0, 3)],
                      effective_transform=mirror_y_matrix())
        self.assertEqual(pline.get_absolute_coordinates()[0][2], .5)
        self.assertEqual(pline.get_absolute_coordinates_xyz()[0][3], -.5)
        before = pline.get_absolute_coordinates_xyz()
        pline.bake_geometry()
        self.assertEqual(pline.get_absolute_coordinates_xyz(), before)
        for matrix in (np.diag([2., 1., 1.]), np.diag([2e-8, 1e-8, 1.])):
            for shape in (Circle(), Arc(), pline):
                with self.subTest(shape=type(shape).__name__, matrix=matrix):
                    shape.effective_transform = matrix
                    with self.assertRaisesRegex(ValueError, "similarity"):
                        shape.get_absolute_coordinates_xyz()

    def test_pline_vertex_records_preserve_query_contracts(self):
        pline = Pline(
            vertices=[Vertex(0, 0, -2, bulge=0.25), (2, 0, -2),
                      Vertex(3, 4, 7, bulge=-0.5)],
            local_z_offset=3,
        )

        self.assertEqual(
            pline.get_absolute_coordinates(),
            [(0.0, 0.0, 0.25), (2.0, 0.0, 0.0), (3.0, 4.0, -0.5)],
        )
        self.assertEqual(
            pline.get_absolute_coordinates_xyz(),
            [(0.0, 0.0, 1.0, 0.25), (2.0, 0.0, 1.0, 0.0),
             (3.0, 4.0, 10.0, -0.5)],
        )
        self.assertEqual(
            Pline(vertices=[(1, 2, 8)]).get_absolute_coordinates_xyz(),
            [(1.0, 2.0, 8.0, 0.0)],
        )

    def test_points_and_analytic_shapes_return_shape_analog_xyz(self):
        matrix = translation_matrix(10, -4) @ rotation_matrix_deg(90)
        points = Points(
            vertices=[(1, 0, -3), (0, 2, 8)],
            effective_transform=matrix, local_z_offset=0.5,
        )
        np.testing.assert_allclose(
            points.get_absolute_coordinates_xyz(),
            [(10, -3, -2.5), (8, -4, 8.5)], rtol=0, atol=1e-12,
        )
        self.assertEqual(points.get_absolute_coordinates(), [(10.0, -3.0), (8.0, -4.0)])

        circle = Circle(relative_center=(2, 1), diameter=6, elevation=-2,
                        effective_transform=matrix, local_z_offset=0.5)
        circle_xyz = circle.get_absolute_coordinates_xyz()
        np.testing.assert_allclose(circle_xyz["center"], (9, -2, -1.5), rtol=0, atol=1e-12)
        self.assertEqual(circle_xyz["diameter"], 6)
        self.assertEqual(len(circle.get_absolute_coordinates()["center"]), 2)

        arc = Arc(relative_center=(2, 1), radius=4, start_angle=10,
                  extent_angle=-70, elevation=3, effective_transform=matrix,
                  local_z_offset=-1)
        arc_xyz = arc.get_absolute_coordinates_xyz()
        np.testing.assert_allclose(arc_xyz["center"], (9, -2, 2), rtol=0, atol=1e-12)
        self.assertEqual((arc_xyz["radius"], arc_xyz["start_angle"], arc_xyz["extent_angle"]),
                         (4, 100.0, -70))

        rect = Rect(relative_corner=(1, 2), width=3, height=4, elevation=-6,
                    effective_transform=matrix, local_z_offset=2)
        rect_xyz = rect.get_absolute_coordinates_xyz()
        self.assertTrue(all(len(point) == 3 and point[2] == -4 for point in rect_xyz))
        np.testing.assert_allclose(
            [point[:2] for point in rect_xyz], rect.get_absolute_coordinates(),
            rtol=0, atol=1e-12,
        )

    def test_text_preserves_anchor_and_uninterpreted_xml_p2_coordinates(self):
        text = Text(
            text_content="Z", relative_position=(1, 2), elevation=-4,
            xml_p2_position=(5, 6), xml_p2_elevation=9,
            effective_transform=translation_matrix(10, 20), local_z_offset=0.25,
        )
        old_geometry = text.get_absolute_coordinates()
        self.assertEqual(old_geometry["position"], (11.0, 22.0))
        geometry = text.get_absolute_coordinates_xyz()
        self.assertEqual(geometry["position"], (11.0, 22.0, -3.75))
        self.assertEqual(geometry["xml_p2"], (15.0, 26.0, 9.25))

        no_p2 = Text(relative_position=(3, 4), elevation=2)
        default_geometry = no_p2.get_absolute_coordinates_xyz()
        self.assertIsNone(default_geometry["xml_p2"])
        self.assertIsNone(no_p2.to_xml_element(1, None).get("p2"))

    def test_hierarchy_sums_local_z_offsets_and_exposes_4x4_world_transform(self):
        project = CBProject("z-hierarchy")
        layer = project.add_layer("Geometry")
        parent = project.add_points(layer, [(0, 0)], identifier="parent")
        child = project.add_circle(layer, (1, 2), 4, identifier="child", parent=parent)
        parent.local_z_offset = 5.25
        child.local_z_offset = -1.5
        child.elevation = 2
        child.effective_transform = translation_matrix(7, 8)

        self.assertEqual(child.get_total_z_offset(), 3.75)
        self.assertEqual(child.get_absolute_coordinates_xyz()["center"], (8.0, 10.0, 5.75))
        expected = np.identity(4)
        expected[0, 3], expected[1, 3], expected[2, 3] = 7, 8, 3.75
        np.testing.assert_allclose(child.get_total_transform_xyz(), expected, rtol=0, atol=0)

    def test_xml_writes_intrinsic_z_and_total_transform_z_for_every_shape(self):
        shapes_and_fields = [
            (Pline(vertices=[Vertex(1, 2, -3.25, bulge=0.125)],
                   local_z_offset=7.5, output_decimals=12), "pts/p", (-3.25,)),
            (Points(vertices=[(1, 2, 4.5)],
                    local_z_offset=7.5, output_decimals=12), "pts/p", (4.5,)),
            (Circle(relative_center=(1, 2), elevation=-6.75,
                    local_z_offset=7.5, output_decimals=12), None, (-6.75,)),
            (Arc(relative_center=(1, 2), elevation=8.125,
                 local_z_offset=7.5, output_decimals=12), None, (8.125,)),
            (Rect(relative_corner=(1, 2), elevation=-9.875,
                  local_z_offset=7.5, output_decimals=12), None, (-9.875,)),
            (Text(relative_position=(1, 2), elevation=3.25,
                  xml_p2_position=(4, 5), xml_p2_elevation=-2.5,
                  local_z_offset=7.5, output_decimals=12), None, (3.25, -2.5)),
        ]

        for index, (shape, point_path, expected_z) in enumerate(shapes_and_fields, 1):
            with self.subTest(shape=type(shape).__name__):
                element = shape.to_xml_element(index, None)
                if isinstance(shape, (Pline, Points)):
                    actual_z = (xyz(element.find(point_path).text)[2],)
                elif isinstance(shape, Circle):
                    actual_z = (xyz(element.get("c"))[2],)
                elif isinstance(shape, Arc):
                    actual_z = (xyz(element.get("p"))[2],)
                elif isinstance(shape, Rect):
                    actual_z = (xyz(element.get("p"))[2],)
                else:
                    actual_z = (xyz(element.get("p1"))[2], xyz(element.get("p2"))[2])
                    self.assertEqual(xyz(element.get("p2"))[:2], (4.0, 5.0))
                self.assertEqual(actual_z, expected_z)
                _, z_offset = from_cambam_matrix_str(
                    element.find("mat").get("m"), return_z=True
                )
                self.assertEqual(z_offset, 7.5)

    def test_rect_to_pline_paths_preserve_elevation(self):
        rect = Rect(
            relative_corner=(0, 0), width=2, height=1, elevation=6.5,
            local_z_offset=-2, effective_transform=skew_matrix(angle_x_deg=20),
        )
        representation = rect.to_pline_representation()
        self.assertEqual([vertex.z for vertex in representation.vertices], [6.5] * 4)
        self.assertEqual(representation.local_z_offset, -2)
        self.assertTrue(all(point[2] == 4.5 for point in
                            representation.get_absolute_coordinates_xyz()))

        rect.bake_geometry()
        self.assertIsInstance(rect, Pline)
        self.assertEqual([vertex.z for vertex in rect.vertices], [6.5] * 4)
        self.assertEqual(rect.local_z_offset, -2)
        self.assertTrue(all(point[2] == 4.5 for point in rect.get_absolute_coordinates_xyz()))

    def test_intrinsic_z_shift_is_atomic_and_xy_bakes_preserve_z(self):
        shapes = [
            Pline(vertices=[(0, 0, -2), (1, 1, 4)]),
            Points(vertices=[(0, 0, -2), (1, 1, 4)]),
            Circle(elevation=-2), Rect(elevation=-2), Arc(elevation=-2),
            Text(elevation=-2, xml_p2_elevation=4),
        ]
        for shape in shapes:
            with self.subTest(shape=type(shape).__name__):
                before_xy = shape.get_absolute_coordinates()
                shape.bake_geometry(translation_matrix(3, 5))
                if isinstance(shape, (Pline, Points)):
                    before_z = [vertex.z for vertex in shape.vertices]
                else:
                    before_z = (shape.elevation, getattr(shape, "xml_p2_elevation", None))
                shape.shift_geometry_z(2.5)
                if isinstance(shape, (Pline, Points)):
                    self.assertEqual(
                        [vertex.z for vertex in shape.vertices],
                        [z + 2.5 for z in before_z],
                    )
                else:
                    self.assertEqual(shape.elevation, before_z[0] + 2.5)
                    if isinstance(shape, Text):
                        self.assertEqual(shape.xml_p2_elevation, before_z[1] + 2.5)
                with self.assertRaises(ValueError):
                    shape.shift_geometry_z(math.inf)
                self.assertNotEqual(shape.get_absolute_coordinates(), before_xy)

        overflow = Text(elevation=0, xml_p2_elevation=1e308)
        with self.assertRaises(ValueError):
            overflow.shift_geometry_z(1e308)
        self.assertEqual((overflow.elevation, overflow.xml_p2_elevation), (0, 1e308))

    def test_invalid_z_and_unsupported_bakes_fail_cleanly(self):
        for constructor in (
            lambda: Pline(vertices=[(0, 0, 0, 0)]),
            lambda: Points(vertices=[(0, 0, math.nan)]),
            lambda: Points(vertices=[Vertex(0, 0, bulge=0.5)]),
            lambda: Circle(elevation=math.inf),
            lambda: Arc(local_z_offset=math.nan),
            lambda: Text(xml_p2_elevation=-math.inf),
            lambda: Rect(effective_transform=np.full((3, 3), math.nan)),
        ):
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor()

        with self.assertRaisesRegex(ValueError, "equal endpoint Z"):
            Pline(vertices=[Vertex(0, 0, 0, bulge=0.5), (1, 0, 1)])
        with self.assertRaisesRegex(ValueError, "segment 1 to 0"):
            Pline(vertices=[(0, 0, 0), Vertex(1, 0, 1, bulge=0.5)], closed=True)

        bulged = Pline(vertices=[Vertex(0, 0, 2, bulge=0.5), (1, 0, 2)])
        with self.assertRaisesRegex(ValueError, "similarity"):
            bulged.bake_geometry(skew_matrix(angle_x_deg=10))
        self.assertEqual(bulged.vertices,
                         [Vertex(0, 0, 2, bulge=0.5), Vertex(1, 0, 2)])
        bulged.bake_geometry(mirror_y_matrix())
        self.assertEqual(bulged.vertices[0].bulge, -0.5)
        self.assertEqual([vertex.z for vertex in bulged.vertices], [2.0, 2.0])

        for analytic in (Circle(), Arc()):
            with self.subTest(shape=type(analytic).__name__), self.assertRaisesRegex(
                    ValueError, "similarity"):
                analytic.bake_geometry(skew_matrix(angle_x_deg=10))
        with self.assertRaisesRegex(ValueError, "Text geometry"):
            Text().bake_geometry(rotation_matrix_deg(20))

    def test_current_pickle_retains_canonical_vertex_records(self):
        originals = (
            Pline(vertices=[Vertex(1, 2, 3, bulge=0.5), (4, 5, 3)]),
            Points(vertices=[(1, 2, -3), (4, 5, 6)]),
        )
        for original in originals:
            with self.subTest(shape=type(original).__name__):
                restored = pickle.loads(pickle.dumps(original))
                self.assertEqual(restored.vertices, original.vertices)
                self.assertTrue(all(isinstance(vertex, Vertex) for vertex in restored.vertices))
                self.assertFalse(hasattr(restored, "vertex_z"))


if __name__ == "__main__":
    unittest.main()
