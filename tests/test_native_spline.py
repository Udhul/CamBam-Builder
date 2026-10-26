"""Native Line/Spline definition preservation and unsupported-input boundaries."""

from copy import deepcopy
from fractions import Fraction
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder import CBProject
from cambam_builder.native.cad import Line, Spline
from cambam_builder.native.core import Vertex
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.transformations import translation_matrix
from cambam_builder.native.writer import serialize_cambam_bytes


class NativeSplineTests(unittest.TestCase):
    def spline(self, **changes):
        values = dict(user_identifier="curve", degree=2,
                      control_points=[(0, 0, 1), (1, 2, 3), (2, 0, 5)],
                      knots=[2, 2, 2, 5, 5, 5])
        values.update(changes)
        return Spline(**values)

    def project_with_spline(self):
        project = CBProject("spline")
        spline = project.add_spline(
            "Geometry", [(0, 0, 1), (1, 2, 3), (2, 0, 5)], [2, 2, 2, 5, 5, 5],
            degree=2, identifier="curve")
        return project, spline

    def test_line_is_distinct_and_retains_ordered_xyz_segments(self):
        line = Line(user_identifier="line", vertices=[(1, 2, 3), (4, 5, 6), (7, 9, 11)])
        line.effective_transform = translation_matrix(10, -4)
        line.local_z_offset = 7
        self.assertEqual([(11, -2, 10), (14, 1, 13), (17, 5, 18)],
                         line.get_absolute_coordinates_xyz())
        self.assertEqual((11, -2, 17, 5), tuple(
            getattr(line.get_bounding_box(), name) for name in ("min_x", "min_y", "max_x", "max_y")))
        xml = line.to_xml_element(17, None)
        self.assertEqual("line", xml.tag)
        self.assertEqual(["1.0,2.0,3.0", "4.0,5.0,6.0", "7.0,9.0,11.0"],
                         [point.text for point in xml.findall("pts/p")])
        line.bake_geometry()
        self.assertEqual([(11, -2, 10), (14, 1, 13), (17, 5, 18)],
                         line.get_absolute_coordinates_xyz())

    def test_line_rejects_empty_single_point_and_bulge(self):
        for vertices in ([], [(0, 0)], [Vertex(0, 0, bulge=1), (1, 1)]):
            with self.subTest(vertices=vertices), self.assertRaises(ValueError):
                Line(user_identifier="bad", vertices=vertices)

    def test_spline_world_definition_and_bake_preserve_native_auxiliary_data(self):
        spline = self.spline(weights=[1, 2, 1], fit_points=[(1, 1, 2)], flags=31,
                             start_tangent=(1, 2, 3), end_tangent=(0, 0, 0),
                             knot_tolerance=1e-12, control_point_tolerance=2e-11,
                             fit_tolerance=3e-13)
        spline.effective_transform = np.array([[2., 1., 10.], [0., -3., 20.], [0., 0., 1.]])
        spline.local_z_offset = 4
        before = spline.get_absolute_coordinates_xyz()
        self.assertEqual([(10, 20, 5), (14, 14, 7), (14, 20, 9)], before["control_points"])
        self.assertEqual([(13, 17, 6)], before["fit_points"])
        self.assertEqual((1, 2, 3), before["start_tangent"])
        self.assertEqual((0, 0, 0), before["end_tangent"])
        spline.bake_geometry()
        self.assertEqual(before, spline.get_absolute_coordinates_xyz())
        spline.shift_geometry_z(4)
        spline.local_z_offset = 0
        self.assertEqual(before, spline.get_absolute_coordinates_xyz())
        self.assertEqual([1, 2, 1], spline.weights)
        self.assertEqual(31, spline.flags)

    def test_native_scalar_precision_and_absent_tangent_survive(self):
        spline = self.spline(knots=[0, 0, 0, 0.123456789012345, 0.123456789012345, 0.123456789012345],
                             weights=[1, 1.123456789012345, 1], start_tangent=(0, 0.123456789012345, 0),
                             knot_tolerance=1.23456789012345e-15)
        spline.output_decimals = 2
        xml = spline.to_xml_element(9, None)
        self.assertEqual("0.123456789012345", xml.findall("knots/k")[-1].text)
        self.assertEqual("1.123456789012345", xml.findall("weights/w")[1].text)
        self.assertEqual("1.23456789012345e-15", xml.get("ktol"))
        self.assertEqual("0.0,0.123456789012345,0.0", xml.get("starttan"))
        self.assertNotIn("endtan", xml.attrib)

    def test_invalid_spline_definitions_reject(self):
        changes = [dict(degree=0), dict(degree=True), dict(degree=2.5),
                   dict(control_points=[]), dict(knots=[0, 0, 0, 1, 1]),
                   dict(knots=[0, 0, 0, -1, 1, 1]), dict(knots=[0] * 6),
                   dict(knots=[0, 0, 0, float("inf"), 1, 1]), dict(weights=[]),
                   dict(weights=[1, 1]), dict(weights=[1, 0, 1]), dict(weights=[1, -1, 1]),
                   dict(flags=-1), dict(flags=2**31), dict(start_tangent=(1, 2)),
                   dict(end_tangent=(1, 2, float("nan"))), dict(knot_tolerance=-1),
                   dict(fit_tolerance=float("nan"))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.spline(**change)

    def test_mutated_numeric_values_normalize_in_queries_and_native_xml(self):
        project, spline = self.project_with_spline()
        spline.knots = [False, 0, 0, Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)]
        spline.weights = [True, Fraction(2, 3), 1]
        spline.knot_tolerance = Fraction(1, 10)
        spline.control_point_tolerance = True
        spline.fit_tolerance = "0.002"
        spline.start_tangent = (Fraction(1, 3), True, "0")
        definition = spline.get_absolute_coordinates_xyz()
        self.assertEqual([0.0, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 3], definition["knots"])
        self.assertEqual([1.0, 2 / 3, 1.0], definition["weights"])
        self.assertEqual((1 / 3, 1.0, 0.0), definition["start_tangent"])
        for name in ("knots", "weights"):
            self.assertTrue(all(type(value) is float for value in definition[name]))
        for name in ("knot_tolerance", "control_point_tolerance", "fit_tolerance"):
            self.assertIs(type(definition[name]), float)
        reopened = read_cambam_bytes(serialize_cambam_bytes(project)).get_primitive("curve")
        self.assertEqual(definition, reopened.get_absolute_coordinates_xyz())

    def test_mutated_invalid_data_rejects_export_and_atomic_bake(self):
        spline = self.spline()
        spline.control_points[1].x = float("nan")
        original = deepcopy(spline)
        with self.assertRaises(ValueError):
            spline.to_xml_element(1, None)
        with self.assertRaises(ValueError):
            spline.bake_geometry(translation_matrix(3, 4))
        self.assertEqual(original.control_points[0], spline.control_points[0])
        np.testing.assert_array_equal(original.effective_transform, spline.effective_transform)

    def test_unavailable_bounds_and_implicit_pivots_never_drop_or_move_spline(self):
        project, spline = self.project_with_spline()
        project.add_line("Geometry", [(0, 0), (100, 100)])
        for query in (spline.get_bounding_box, spline.get_geometric_center, project.get_bounding_box):
            with self.assertRaises(NotImplementedError):
                query()
        before = spline.get_absolute_coordinates_xyz()
        actions = [lambda: project.rotate_primitive_deg(spline, 30),
                   lambda: project.rotate_primitive_rad(spline, 1),
                   lambda: project.scale_primitive(spline, 2),
                   lambda: project.mirror_primitive_x(spline),
                   lambda: project.mirror_primitive_y(spline),
                   lambda: project.align_primitive(spline, (0, 0), align_x="left")]
        for action in actions:
            self.assertFalse(action())
            self.assertEqual(before, spline.get_absolute_coordinates_xyz())
        self.assertTrue(project.rotate_primitive_deg(spline, 90, cx=0, cy=0))
        np.testing.assert_allclose([(0, 0, 1), (-2, 1, 3), (0, 2, 5)],
                                   spline.get_absolute_coordinates_xyz()["control_points"], atol=1e-12)

    def test_reader_rejects_missing_weights_and_unmodeled_point_data(self):
        project, _ = self.project_with_spline()
        original = ET.fromstring(serialize_cambam_bytes(project))
        for mutation in ("missing_weights", "empty_weights", "bulge", "bad_knots", "fit_only",
                         "duplicate_ctrl", "duplicate_knots", "unknown_point", "unknown_knot"):
            with self.subTest(mutation=mutation):
                root = deepcopy(original)
                element = root.find("./layers/layer/objects/spline")
                if mutation == "missing_weights":
                    element.remove(element.find("weights"))
                elif mutation == "empty_weights":
                    element.find("weights").clear()
                elif mutation == "bulge":
                    element.find("ctrl/p").set("b", "1")
                elif mutation == "bad_knots":
                    element.find("knots/k").text = "nan"
                elif mutation == "duplicate_ctrl":
                    element.append(deepcopy(element.find("ctrl")))
                elif mutation == "duplicate_knots":
                    element.append(deepcopy(element.find("knots")))
                elif mutation == "unknown_point":
                    ET.SubElement(element.find("ctrl"), "point").text = "1,2,3"
                elif mutation == "unknown_knot":
                    ET.SubElement(element.find("knots"), "knot").text = "4"
                else:
                    element.find("ctrl").clear()
                with self.assertLogs("cambam_builder.native.reader", level="ERROR"), self.assertRaises(ValueError):
                    read_cambam_bytes(ET.tostring(root))

    def test_reader_distinguishes_empty_and_zero_native_tangents(self):
        project, _ = self.project_with_spline()
        root = ET.fromstring(serialize_cambam_bytes(project))
        element = root.find("./layers/layer/objects/spline")
        element.set("starttan", "")
        element.set("endtan", "0,0,0")
        loaded = read_cambam_bytes(ET.tostring(root)).list_primitives()[0]
        self.assertIsNone(loaded.start_tangent)
        self.assertEqual((0, 0, 0), loaded.end_tangent)

    def test_adders_return_none_on_invalid_input_and_author_unit_weights(self):
        project, spline = self.project_with_spline()
        self.assertEqual([1, 1, 1], spline.weights)
        self.assertIsNone(project.add_line("Geometry", [(0, 0)]))
        self.assertIsNone(project.add_spline("Geometry", [(0, 0)], [], degree=0))
        self.assertEqual(1, len(project.list_primitives()))


if __name__ == "__main__":
    unittest.main()
