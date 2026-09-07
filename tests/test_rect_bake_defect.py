"""Characterization tests for the remaining Rect transform bake defect."""

import tempfile
import logging
import unittest
from pathlib import Path

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cad_transformations import (
    apply_transform,
    identity_matrix,
    mirror_y_matrix,
    rotation_matrix_deg,
    skew_matrix,
)
from cambam_builder.cambam_entities import Pline, Rect
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class RectBakeDefectTests(unittest.TestCase):
    """Synthetic root Rect cases that bound current geometry behavior."""

    @staticmethod
    def make_rect(matrix=None):
        project = CBProject("rect-bake-defect")
        layer = project.add_layer("Geometry")
        rect = project.add_rect(
            layer, corner=(0.0, 0.0), width=4.0, height=2.0, identifier="rect"
        )
        if matrix is not None:
            rect.effective_transform = np.asarray(matrix, dtype=float)
        return project, rect

    @staticmethod
    def corners(rect):
        return [(point[0], point[1]) for point in rect.get_absolute_coordinates()]

    @staticmethod
    def polygon_area(points):
        values = np.asarray(points, dtype=float)
        return 0.5 * abs(
            np.dot(values[:, 0], np.roll(values[:, 1], -1))
            - np.dot(values[:, 1], np.roll(values[:, 0], -1))
        )

    @classmethod
    def expected_corners(cls, matrix):
        return apply_transform(
            [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)], matrix
        )

    @classmethod
    def expected_aabb(cls, points):
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_x = max(point[0] for point in points)
        max_y = max(point[1] for point in points)
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    @staticmethod
    def assert_same_point_set(test_case, expected, actual):
        np.testing.assert_allclose(
            np.asarray(sorted(expected)), np.asarray(sorted(actual)), rtol=0, atol=2e-9
        )

    @staticmethod
    def save(project, directory, name):
        path = Path(directory) / (name + ".cb")
        save_cambam_file(project, str(path))
        return path

    def run_without_entity_warnings(self, operation):
        logger = logging.getLogger("cambam_builder.cambam_entities")
        records = []

        class WarningRecorder(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    records.append(record)

        handler = WarningRecorder()
        previous_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            operation()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        self.assertEqual(records, [])

    def test_known_defect_effective_45_degree_rotation_bake_uses_aabb(self):
        matrix = rotation_matrix_deg(45.0)
        project, rect = self.make_rect(matrix)
        expected = self.expected_corners(matrix)

        self.run_without_entity_warnings(
            lambda: self.assertTrue(project.bake_primitive_transform("rect"))
        )
        actual = self.corners(rect)
        expected_aabb = self.expected_aabb(expected)

        self.assert_same_point_set(self, expected_aabb, actual)
        self.assertAlmostEqual(self.polygon_area(expected), 8.0, places=10)
        self.assertAlmostEqual(self.polygon_area(actual), 18.0, places=10)
        self.assertGreater(self.polygon_area(actual), self.polygon_area(expected))
        np.testing.assert_array_equal(rect.effective_transform, identity_matrix())

    def test_known_defect_effective_x_shear_bake_warns_and_uses_aabb(self):
        # x += y, with unit shear; the 4x2 source polygon has area 8.
        matrix = skew_matrix(angle_y_deg=45.0)
        project, rect = self.make_rect(matrix)
        expected = self.expected_corners(matrix)

        with self.assertLogs("cambam_builder.cambam_entities", level="WARNING") as logs:
            self.assertTrue(project.bake_primitive_transform("rect"))
        self.assertTrue(any("rotation/shear will be lost" in entry.lower() for entry in logs.output))
        actual = self.corners(rect)
        self.assert_same_point_set(self, self.expected_aabb(expected), actual)
        self.assertAlmostEqual(self.polygon_area(expected), 8.0, places=10)
        self.assertAlmostEqual(self.polygon_area(actual), 12.0, places=10)
        self.assertGreater(self.polygon_area(actual), self.polygon_area(expected))

    def test_axis_preserving_transforms_keep_rect_outline(self):
        for name, matrix in (
            ("rotation", rotation_matrix_deg(90.0)),
            ("reflection", mirror_y_matrix()),
            ("translation", np.array([[1., 0., 3.], [0., 1., -2.], [0., 0., 1.]])),
            ("scale", np.diag([2., 3., 1.])),
        ):
            with self.subTest(transform=name):
                project, rect = self.make_rect(matrix)
                expected = self.expected_corners(matrix)
                self.assertTrue(project.bake_primitive_transform("rect"))
                self.assert_same_point_set(self, expected, self.corners(rect))
                self.assertAlmostEqual(self.polygon_area(self.corners(rect)), self.polygon_area(expected))
                np.testing.assert_array_equal(rect.effective_transform, identity_matrix())

    def test_known_defect_public_global_bake_shear_is_silent_and_uses_aabb(self):
        matrix = skew_matrix(angle_y_deg=45.0)
        project, rect = self.make_rect()
        expected = self.expected_corners(matrix)

        self.run_without_entity_warnings(
            lambda: self.assertTrue(project.transform_primitive("rect", matrix, bake=True))
        )
        self.assert_same_point_set(self, self.expected_aabb(expected), self.corners(rect))
        self.assertAlmostEqual(self.polygon_area(self.corners(rect)), 12.0, places=10)
        np.testing.assert_array_equal(rect.effective_transform, identity_matrix())

    def test_known_defect_explicit_matrix_bake_uses_aabb(self):
        matrix = rotation_matrix_deg(45.0)
        project, rect = self.make_rect()
        expected = self.expected_corners(matrix)

        self.assertTrue(project.bake_primitive_transform("rect", transform_to_bake=matrix))
        self.assert_same_point_set(self, self.expected_aabb(expected), self.corners(rect))
        self.assertAlmostEqual(self.polygon_area(self.corners(rect)), 18.0, places=10)
        np.testing.assert_array_equal(rect.effective_transform, identity_matrix())

    def test_known_defect_rotation_component_bake_uses_aabb(self):
        matrix = rotation_matrix_deg(45.0)
        project, rect = self.make_rect(matrix)
        expected = self.expected_corners(matrix)

        self.assertTrue(project.bake_primitive_transform_component("rect", "rotation"))
        self.assert_same_point_set(self, self.expected_aabb(expected), self.corners(rect))
        self.assertAlmostEqual(self.polygon_area(self.corners(rect)), 18.0, places=10)
        np.testing.assert_array_equal(rect.effective_transform, identity_matrix())

    def test_known_defect_baked_rect_damage_persists_through_two_xml_round_trips(self):
        matrix = rotation_matrix_deg(45.0)
        project, rect = self.make_rect(matrix)
        expected = self.expected_aabb(self.expected_corners(matrix))
        self.assertTrue(project.bake_primitive_transform("rect"))

        with tempfile.TemporaryDirectory() as directory:
            first = self.save(project, directory, "baked-first")
            loaded_once = read_cambam_file(str(first))
            second = self.save(loaded_once, directory, "baked-second")
            loaded_twice = read_cambam_file(str(second))
            for loaded in (loaded_once, loaded_twice):
                self.assertEqual(len(loaded.list_primitives()), 1)
                self.assertEqual(loaded.get_primitive("rect").internal_id, rect.internal_id)
                self.assertIsInstance(loaded.get_primitive("rect"), Rect)
                self.assert_same_point_set(self, expected, self.corners(loaded.get_primitive("rect")))
                np.testing.assert_array_equal(
                    loaded.get_primitive("rect").effective_transform, identity_matrix()
                )

    def test_unbaked_root_rotation_and_shear_preserve_geometry_through_two_xml_round_trips(self):
        for name, matrix in (
            ("rotation", rotation_matrix_deg(45.0)),
            ("shear", skew_matrix(angle_y_deg=45.0)),
        ):
            with self.subTest(transform=name), tempfile.TemporaryDirectory() as directory:
                project, rect = self.make_rect(matrix)
                expected = self.expected_corners(matrix)
                first = self.save(project, directory, "unbaked-first")
                loaded_once = read_cambam_file(str(first))
                second = self.save(loaded_once, directory, "unbaked-second")
                loaded_twice = read_cambam_file(str(second))
                for loaded in (loaded_once, loaded_twice):
                    self.assertEqual(len(loaded.list_primitives()), 1)
                    self.assertEqual(loaded.get_primitive("rect").internal_id, rect.internal_id)
                    actual = self.corners(loaded.get_primitive("rect"))
                    self.assert_same_point_set(self, expected, actual)
                    self.assertAlmostEqual(self.polygon_area(actual), 8.0, places=8)
                self.assertIsInstance(loaded_twice.get_primitive("rect"), Rect if name == "rotation" else Pline)
                if name == "shear":
                    self.assertTrue(loaded_twice.get_primitive("rect").closed)


if __name__ == "__main__":
    unittest.main()
