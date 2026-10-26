"""Independent analytic and sampled references for bounded relief contact."""

from dataclasses import replace
import math
import unittest

from cambam_builder.cam_core.relief import CubicRelief, HeightField, contact


class ReliefGeometryTests(unittest.TestCase):
    def test_cubic_valley_ridge_extrema_normals_and_curvature(self):
        surface = CubicRelief((-2, -1, 2, 1), 2, (-0.7, -0.3, 0, 0.1))
        bounds = surface.height_bounds(surface.stock_xy)
        self.assertAlmostEqual(bounds[0], -0.9)
        self.assertAlmostEqual(bounds[1], -0.5)
        # h'=0 at +/-1; h''=0.6*x.  These values are independent of
        # the contact subdivision and catch lost interior extrema.
        for x, expected_h, expected_k in ((-1, -0.5, -0.6), (1, -0.9, 0.6)):
            sample = surface.sample(x, 0.3)
            self.assertAlmostEqual(sample.height_mm, expected_h)
            self.assertAlmostEqual(sample.normal_xyz[2], 1)
            self.assertAlmostEqual(sum(sample.principal_curvatures_per_mm), expected_k)
        self.assertAlmostEqual(surface.gradient_bounds[0], 0.9)
        self.assertEqual(surface.gradient_bounds[1], 0)
        self.assertEqual(surface.height_at(-10, 4), surface.height_at(-2, 1))
        self.assertEqual(surface.fingerprint, replace(surface).fingerprint)
        self.assertNotEqual(surface.fingerprint, replace(surface, coefficients=(-0.71, -0.3, 0, 0.1)).fingerprint)

    def test_flat_and_affine_contact_exact_oracles(self):
        for slope in (0, -0.4, 0.25, 1):
            surface = CubicRelief((-2, -2, 2, 2), 5, (-2.1, slope, 0, 0))
            radius, center_x = 0.3, 0.2
            expected = -2.1 + slope * center_x + radius * (math.sqrt(1 + slope * slope) - 1)
            result = contact(surface, radius, center_x, 0.7, tolerance_mm=0.00002)
            self.assertLessEqual(result.lower_tip_z_mm, expected)
            self.assertGreaterEqual(result.upper_tip_z_mm, expected)
            self.assertLessEqual(result.upper_tip_z_mm - result.lower_tip_z_mm, 0.00002)
        # At a stock edge the clamped extension remains conservative and flat.
        flat = CubicRelief((0, 0, 1, 1), 1, (-0.5, 0, 0, 0))
        edge = contact(flat, 0.2, -0.1, 0)
        self.assertLessEqual(edge.lower_tip_z_mm, -0.5)
        self.assertGreaterEqual(edge.upper_tip_z_mm, -0.5)

    def test_cubic_dense_oracle_is_enclosed_and_refinement_tightens(self):
        surface = CubicRelief((-2, -1, 2, 1), 2, (-0.7, -0.3, 0, 0.1))
        radius, center_x = 0.4, 0.85
        values = []
        for i in range(20001):
            offset = -radius + 2 * radius * i / 20000
            x = center_x + offset
            # Direct polynomial and sphere equations form a separate oracle.
            h = -0.7 - 0.3 * x + 0.1 * x ** 3
            values.append(h - radius + math.sqrt(max(0, radius * radius - offset * offset)))
        expected = max(values)
        rough = contact(surface, radius, center_x, 0, tolerance_mm=0.01)
        fine = contact(surface, radius, center_x, 0, tolerance_mm=0.0001)
        self.assertLessEqual(fine.lower_tip_z_mm - 1e-8, expected)
        self.assertGreaterEqual(fine.upper_tip_z_mm, expected)
        self.assertLess(fine.upper_tip_z_mm - fine.lower_tip_z_mm,
                        rough.upper_tip_z_mm - rough.lower_tip_z_mm)
        self.assertGreater(fine.evaluated_cells, rough.evaluated_cells)

    def test_bilinear_saddle_differentials_bounds_and_seams(self):
        # z=-1+0.2*x*y on [-1,1]^2: a saddle with curvatures +/-0.2.
        surface = HeightField((-1, 1), (-1, 1), ((-0.8, -1.2), (-1.2, -0.8)), 2)
        sample = surface.sample(0, 0)
        self.assertAlmostEqual(sample.height_mm, -1)
        self.assertEqual(sample.normal_xyz, (0, 0, 1))
        self.assertAlmostEqual(sample.principal_curvatures_per_mm[0], -0.2)
        self.assertAlmostEqual(sample.principal_curvatures_per_mm[1], 0.2)
        self.assertEqual(surface.height_bounds(surface.stock_xy), (-1.2, -0.8))
        self.assertAlmostEqual(surface.gradient_bounds[0], 0.2)
        self.assertAlmostEqual(surface.gradient_bounds[1], 0.2)
        multi = HeightField((-1, 0, 1), (-1, 0.2, 1),
                            ((-1, -0.1, -1), (-0.5, -0.8, -0.4), (-1, -0.4, -1)), 2)
        self.assertEqual(multi.height_bounds((-0.5, -0.5, 0.5, 0.5))[1],
                         max(multi.height_at(x, y) for x in (-0.5, 0, 0.5) for y in (-0.5, 0.2, 0.5)))
        with self.assertRaisesRegex(ValueError, "seam"):
            multi.sample(0, 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            multi.sample(-2, 0)

    def test_general_heightfield_contact_against_dense_disk(self):
        surface = HeightField((-1, 0, 1), (-1, 0, 1),
                              ((-1.2, -0.8, -1.1), (-0.8, -1.3, -0.9), (-1, -0.6, -1.2)), 2)
        center_x, center_y, radius = 0.2, 0.15, 0.25
        result = contact(surface, radius, center_x, center_y, tolerance_mm=0.0003)
        dense = -math.inf
        for i in range(301):
            dx = -radius + 2 * radius * i / 300
            for j in range(301):
                dy = -radius + 2 * radius * j / 300
                distance2 = dx * dx + dy * dy
                if distance2 <= radius * radius:
                    value = surface.height_at(center_x + dx, center_y + dy) - radius + math.sqrt(radius * radius - distance2)
                    dense = max(dense, value)
        self.assertGreaterEqual(result.upper_tip_z_mm, dense)
        self.assertLessEqual(result.lower_tip_z_mm - dense, 0.00003)
        self.assertLessEqual(result.upper_tip_z_mm - result.lower_tip_z_mm, 0.0003)

    def test_oblique_affine_heightfield_contact_has_independent_plane_oracle(self):
        # The exact offset of z=-2+0.3*x-0.4*y has ||gradient||=0.5.
        surface = HeightField((-2, 2), (-2, 2),
                              ((-1.8, -0.6), (-3.4, -2.2)), 4)
        radius, x, y = 0.35, 0.25, -0.15
        expected = -2 + 0.3 * x - 0.4 * y + radius * (math.sqrt(1.25) - 1)
        result = contact(surface, radius, x, y, tolerance_mm=0.0001)
        self.assertLessEqual(result.lower_tip_z_mm, expected)
        self.assertGreaterEqual(result.upper_tip_z_mm, expected)
        normal = surface.sample(x, y).normal_xyz
        norm = math.sqrt(1.25)
        for actual, expected_component in zip(normal, (-0.3 / norm, 0.4 / norm, 1 / norm)):
            self.assertAlmostEqual(actual, expected_component)
        for curvature in surface.sample(x, y).principal_curvatures_per_mm:
            self.assertAlmostEqual(curvature, 0)

    def test_translation_of_heightfield_preserves_contacts(self):
        original = HeightField((-1, 1), (-1, 1), ((-0.8, -1.2), (-1.2, -0.8)), 2)
        shifted = HeightField((2, 4), (-3, -1), original.heights, 2)
        first = contact(original, 0.3, 0.2, -0.1, tolerance_mm=0.0001)
        second = contact(shifted, 0.3, 3.2, -2.1, tolerance_mm=0.0001)
        self.assertAlmostEqual(first.lower_tip_z_mm, second.lower_tip_z_mm, places=10)
        self.assertAlmostEqual(first.upper_tip_z_mm, second.upper_tip_z_mm, places=10)
        self.assertNotEqual(original.fingerprint, shifted.fingerprint)

    def test_rejects_unsupported_geometry_arithmetic_and_contact_budget(self):
        surface = CubicRelief((-2, -1, 2, 1), 2, (-0.7, -0.3, 0, 0.1))
        for kwargs in ({"radius_mm": 0}, {"radius_mm": True}, {"tolerance_mm": 0},
                       {"tolerance_mm": 1e-20}, {"max_cells": True}, {"x": math.nan}):
            params = dict(radius_mm=0.4, x=0.8, y=0)
            params.update(kwargs)
            with self.assertRaises(ValueError):
                contact(surface, **params)
        with self.assertRaisesRegex(ValueError, "budget"):
            contact(surface, 0.4, 0.8, 0, tolerance_mm=1e-6, max_cells=1)
        invalid = (
            lambda: CubicRelief((0, 0, 0, 1), 2, (-1, 0, 0, 0)),
            lambda: CubicRelief((-2, -1, 2, 1), 0.8, (-0.7, -0.3, 0, 0.1)),
            lambda: CubicRelief((0, 0, 1, 1), 2, (-1, math.inf, 0, 0)),
            lambda: CubicRelief((1e200, 0, 2e200, 1), 2, (-1, 0, 0, 1)),
            lambda: HeightField((0, 0), (0, 1), ((-1, -1), (-1, -1)), 2),
            lambda: HeightField((False, 1), (0, 1), ((-1, -1), (-1, -1)), 2),
            lambda: HeightField((0, 1), (0, 1), ((-1,), (-1,)), 2),
            lambda: HeightField((0, 1), (0, 1), ((-1, 1), (-1, -1)), 2),
        )
        for constructor in invalid:
            with self.assertRaises(ValueError):
                constructor()


if __name__ == "__main__":
    unittest.main()
