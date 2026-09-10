"""Focused regression tests for exact Arc and bulged-Pline bounds."""

import math
import unittest

import numpy as np

from cambam_builder.cambam_entities import (
    ARC_SWEEP_TOLERANCE_DEGREES,
    PLINE_BULGE_TOLERANCE,
    Arc,
    Pline,
    Vertex,
)
from cambam_builder.cad_transformations import mirror_x_matrix, rotation_matrix_deg, skew_matrix


class CurvedBoundsTests(unittest.TestCase):
    def assert_box(self, box, expected, places=10):
        self.assertTrue(box.is_valid())
        for actual, wanted in zip((box.min_x, box.min_y, box.max_x, box.max_y), expected):
            self.assertAlmostEqual(actual, wanted, places=places)

    def test_arc_quadrant_and_wraparound_sweeps(self):
        self.assert_box(Arc(radius=2, start_angle=0, extent_angle=90).get_bounding_box(),
                        (0, 0, 2, 2))
        self.assert_box(Arc(radius=1, start_angle=350, extent_angle=40).get_bounding_box(),
                        (math.cos(math.radians(30)), math.sin(math.radians(-10)), 1, .5))
        self.assert_box(Arc(radius=1, start_angle=10, extent_angle=-40).get_bounding_box(),
                        (math.cos(math.radians(30)), -.5, 1, math.sin(math.radians(10))))

    def test_arc_full_negative_and_tolerance_boundary(self):
        for sweep in (360, -360, 720, -720, 360 - ARC_SWEEP_TOLERANCE_DEGREES / 2):
            self.assert_box(Arc(relative_center=(3, -4), radius=2,
                                start_angle=17, extent_angle=sweep).get_bounding_box(),
                            (1, -6, 5, -2))
        almost_full = 360 - 2 * ARC_SWEEP_TOLERANCE_DEGREES
        box = Arc(radius=1, start_angle=45, extent_angle=almost_full).get_bounding_box()
        self.assertLess(box.min_x, -0.70)
        self.assertGreater(box.max_x, 0.99)
        self.assertGreater(box.max_y, 0.99)

    def test_arc_affine_extrema_support_reflection_and_shear(self):
        self.assert_box(Arc(radius=2, start_angle=0, extent_angle=90,
                            effective_transform=rotation_matrix_deg(90)).get_bounding_box(),
                        (-2, 0, 0, 2))
        self.assert_box(Arc(radius=2, start_angle=0, extent_angle=90,
                            effective_transform=mirror_x_matrix()).get_bounding_box(),
                        (0, -2, 2, 0))
        partial = np.array(((2, 1, 0), (0, 3, 0), (0, 0, 1)), dtype=float)
        self.assert_box(Arc(radius=1, start_angle=0, extent_angle=90,
                            effective_transform=partial).get_bounding_box(),
                        (1, 0, math.sqrt(5), 3))
        singular = np.array(((1, 0, 0), (0, 0, 2), (0, 0, 1)), dtype=float)
        self.assert_box(Arc(radius=1, start_angle=0, extent_angle=90,
                            effective_transform=singular).get_bounding_box(),
                        (0, 2, 1, 2))
        matrix = skew_matrix(angle_x_deg=0, angle_y_deg=0)
        matrix[:2, :2] = ((2, .5), (.25, 1))
        matrix[0, 2], matrix[1, 2] = (3, -2)
        # Full-circle affine bounds are row norms, not an average scale.
        box = Arc(relative_center=(1, -2), radius=2, extent_angle=360,
                  effective_transform=matrix).get_bounding_box()
        self.assert_box(box, (4 - math.sqrt(17), -3.75 - math.sqrt(17) / 2,
                              4 + math.sqrt(17), -3.75 + math.sqrt(17) / 2))

    def test_pline_positive_negative_and_large_bulges(self):
        self.assert_box(Pline(vertices=[Vertex(0, 0, bulge=1), (2, 0)]).get_bounding_box(),
                        (0, -1, 2, 0))
        self.assert_box(Pline(vertices=[Vertex(0, 0, bulge=-1), (2, 0)]).get_bounding_box(),
                        (0, 0, 2, 1))
        self.assert_box(Pline(vertices=[Vertex(0, 0, bulge=2), (2, 0)]).get_bounding_box(),
                        (-.25, -2, 2.25, 0))

    def test_pline_open_closed_final_bulge_and_degenerate_segments(self):
        points = [Vertex(0, 0), Vertex(2, 0), Vertex(2, 2, bulge=1)]
        self.assert_box(Pline(vertices=points, closed=False).get_bounding_box(),
                        (0, 0, 2, 2))
        self.assert_box(Pline(vertices=points, closed=True).get_bounding_box(),
                        (1 - math.sqrt(2), 0, 2, 1 + math.sqrt(2)))
        self.assert_box(Pline(vertices=[Vertex(0, 0, bulge=1), (0, 0), (2, 0)]).get_bounding_box(),
                        (0, 0, 2, 0))
        self.assert_box(Pline(vertices=[Vertex(0, 0, bulge=PLINE_BULGE_TOLERANCE / 2), (2, 0)]).get_bounding_box(),
                        (0, 0, 2, 0))

    def test_pline_affine_and_invalid_transform_boundary(self):
        matrix = np.array(((1.5, .5, 2), (-.25, 2, -3), (0, 0, 1)), dtype=float)
        box = Pline(vertices=[Vertex(0, 0, bulge=1), (2, 0)], effective_transform=matrix).get_bounding_box()
        # Independent endpoint/ellipse extrema check for this semicircle.
        self.assert_box(box, (3.5 - math.sqrt(2.5), -3.25 - math.sqrt(4.0625), 5, -3))
        invalid = matrix.copy()
        invalid[2, 0] = .01
        corrupted = Pline(vertices=[Vertex(0, 0, bulge=1), (2, 0)])
        corrupted.effective_transform = invalid
        self.assertFalse(corrupted.get_bounding_box().is_valid())

        for invalid in (
            np.eye(3, dtype=complex),
            np.array(((1, 0, 0), (0, 1, 0), (1e-13, 0, 1)), dtype=float),
            np.array(((1, 0, 0), (0, math.nan, 0), (0, 0, 1)), dtype=float),
        ):
            for corrupted in (Arc(), Pline(vertices=[(0, 0), (1, 0)])):
                # Constructors now reject invalid transforms. Bounds must also
                # remain defensive against a subsequently corrupted attribute.
                corrupted.effective_transform = invalid
                self.assertFalse(corrupted.get_bounding_box().is_valid())
        enormous = 10 ** 10000
        self.assertFalse(Arc(radius=enormous).get_bounding_box().is_valid())
        with self.assertRaisesRegex(ValueError, "finite"):
            Pline(vertices=[(enormous, 0), (1, 0)])

    def test_large_finite_values_avoid_intermediate_overflow(self):
        matrix = np.array(((1e200, 0, 0), (0, 1e200, 0), (0, 0, 1)), dtype=float)
        self.assert_box(Arc(radius=1, extent_angle=360,
                            effective_transform=matrix).get_bounding_box(),
                        (-1e200, -1e200, 1e200, 1e200))

        box = Pline(vertices=[Vertex(-1e308, 0, bulge=1), (1e308, 0)]).get_bounding_box()
        self.assertTrue(box.is_valid())
        self.assertAlmostEqual(box.min_x, -1e308, delta=1e294)
        self.assertAlmostEqual(box.min_y, -1e308, delta=1e294)
        self.assertAlmostEqual(box.max_x, 1e308, delta=1e294)
        self.assertLessEqual(abs(box.max_y), 1e294)


if __name__ == "__main__":
    unittest.main()
