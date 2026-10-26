"""Independent fidelity, interval and edit checks for compact construction."""

from dataclasses import replace
import json
import math
import unittest

from cambam_builder.cam_core.constructive import (
    ConstructiveSource, SweptRidge, reconstruct_spine,
)


def frieze() -> ConstructiveSource:
    return ConstructiveSource("frieze", 4, 2, 1, -0.8, (
        SweptRidge("upper", (0, 4), (0.65, 1.35), (0.25, 0.4), (0.45, 0.55),
                   wave_amplitude_mm=0.2),
        SweptRidge("lower", (0, 4), (1.35, 0.65), (0.3, 0.25), (0.5, 0.4),
                   wave_amplitude_mm=-0.2, profile="tent"),
    ), repeat_count=2, provenance=("authored crossing ribbons",))


class ConstructiveSourceTests(unittest.TestCase):
    def test_closed_form_profile_caps_sharp_ridge_and_repeat_seams(self):
        ridge = SweptRidge("petal", (0, 4), (1, 1), (0.5, 0.5), (0.6, 0.6))
        source = ConstructiveSource("petal", 4, 2, 1, -0.8, (ridge,), repeat_count=3)
        # Independent closed form at quarter and middle stations.
        for x, expected in ((0, -0.8), (1, -0.5), (2, -0.2), (3, -0.5), (4, -0.8)):
            self.assertAlmostEqual(source.height_at(x, 1), expected)
        self.assertAlmostEqual(source.height_at(2, 1.25), -0.5)
        for seam in (4, 8):
            self.assertAlmostEqual(source.height_at(seam - 1e-6, 1), -0.8, places=10)
            self.assertAlmostEqual(source.height_at(seam + 1e-6, 1), -0.8, places=10)
        tent = replace(source, ridges=(replace(ridge, profile="tent"),))
        eps = 1e-6
        left = (tent.height_at(2, 1) - tent.height_at(2, 1 - eps)) / eps
        right = (tent.height_at(2, 1 + eps) - tent.height_at(2, 1)) / eps
        self.assertAlmostEqual(left, 1.2, places=7)
        self.assertAlmostEqual(right, -1.2, places=7)

    def test_max_union_crossing_preserves_branch_crease(self):
        source = ConstructiveSource("cross", 4, 2, 1, -0.8, (
            SweptRidge("a", (0, 4), (0.5, 1.5), (0.2, 0.2), (0.5, 0.5)),
            SweptRidge("b", (0, 4), (1.5, 0.5), (0.2, 0.2), (0.5, 0.5)),
        ))
        # The two authored branches meet at a single common ridge at x=2.
        self.assertAlmostEqual(source.height_at(2, 1), -0.3)
        for x in (1, 3):
            cap = math.sin(math.pi * x / 4) ** 2
            self.assertAlmostEqual(source.height_at(x, 0.5 + x / 4), -0.8 + 0.5 * cap)
            self.assertAlmostEqual(source.height_at(x, 1.5 - x / 4), -0.8 + 0.5 * cap)
        self.assertEqual(source.height_at(1, 1), -0.8)

    def test_local_interval_bounds_dense_independent_formula(self):
        source = frieze()
        # Separate equations, including a repeated join and clipped stock edge.
        def oracle(x, y):
            x = max(0, min(8, x))
            y = max(0, min(2, y))
            u = (x % 4 if x < 8 else 4) / 4
            cap = math.sin(math.pi * u) ** 2
            wave = 0.2 * math.sin(2 * math.pi * u)
            qa = (y - (0.65 + 0.7 * u + wave)) / (0.25 + 0.15 * u)
            qb = (y - (1.35 - 0.7 * u - wave)) / (0.3 - 0.05 * u)
            pa = (1 + math.cos(math.pi * qa)) / 2 if abs(qa) < 1 else 0
            pb = max(0, 1 - abs(qb))
            return -0.8 + cap * max((0.45 + 0.1 * u) * pa, (0.5 - 0.1 * u) * pb)
        rects = ((0.2, 0.4, 0.6, 0.9), (1.7, 0.8, 2.2, 1.3),
                 (3.8, 0.6, 4.3, 1.4), (-1, -1, 0.2, 0.2), (8.2, 2.2, 8.5, 2.5))
        for rect in rects:
            low, high = source.height_bounds(rect)
            for i in range(21):
                for j in range(21):
                    x = rect[0] + (rect[2] - rect[0]) * i / 20
                    y = rect[1] + (rect[3] - rect[1]) * j / 20
                    expected = oracle(x, y)
                    self.assertAlmostEqual(source.height_at(x, y), expected, places=12)
                    self.assertLessEqual(low, expected)
                    self.assertGreaterEqual(high, expected)
        center = source.height_at(1.7, 0.8)
        low, high = source.height_bounds((1.7 - 1e-7, 0.8 - 1e-7, 1.7 + 1e-7, 0.8 + 1e-7))
        self.assertLess(high - low, 1e-5)
        self.assertLessEqual(low, center)
        self.assertGreaterEqual(high, center)

    def test_derivative_bounds_enclose_finite_differences_with_modifiers(self):
        source = replace(frieze(), scale_xyz=(1.7, 0.8, 0.6), origin_xy=(-3, 2))
        gx, gy = source.gradient_bounds
        x0, y0, x1, y1 = source.stock_xy
        h = 1e-6
        for i in range(1, 60):
            for j in range(1, 12):
                x, y = x0 + (x1 - x0) * i / 60, y0 + (y1 - y0) * j / 12
                dx = abs(source.height_at(x + h, y) - source.height_at(x - h, y)) / (2 * h)
                dy = abs(source.height_at(x, y + h) - source.height_at(x, y - h)) / (2 * h)
                self.assertLessEqual(dx, gx)
                self.assertLessEqual(dy, gy)

    def test_changing_section_scaling_edit_identity_and_json(self):
        ridge = SweptRidge("moulding", (0, 5), (0.8, 1.1), (0.4, 0.7), (0.3, 0.6),
                           wave_amplitude_mm=0.08, cap_ends=False)
        source = ConstructiveSource("moulding", 5, 2, 1, -0.8, (ridge,),
                                    scale_xyz=(2, 0.5, 1.5), origin_xy=(-1, 3),
                                    provenance=("section changes along spine",))
        self.assertEqual(source.stock_xy, (-1, 3, 9, 4))
        self.assertEqual(source.stock_depth_mm, 1.5)
        self.assertAlmostEqual(source.height_at(-1, 3.4), -0.75)
        self.assertAlmostEqual(source.height_at(9, 3.55), -0.3)
        loaded = ConstructiveSource.from_json(source.canonical_json())
        self.assertEqual(loaded, source)
        self.assertEqual(loaded.fingerprint, source.fingerprint)
        changed = replace(source, ridges=(replace(ridge, widths_mm=(0.5, 0.7)),))
        self.assertEqual(changed.source_id, source.source_id)
        self.assertNotEqual(changed.fingerprint, source.fingerprint)
        self.assertNotEqual(replace(source, scale_xyz=(1, 1, 1)).fingerprint, source.fingerprint)

    def test_sampling_bounds_bilinear_and_triangle_error_across_seams(self):
        source = frieze()
        coarse = source.sample_heightfield(x_samples=18, y_samples=12)
        fine = source.sample_heightfield(x_samples=35, y_samples=23)
        self.assertLess(fine.deviation_bound_mm, coarse.deviation_bound_mm)
        for artifact in (coarse, fine):
            grid = artifact.target
            for i in range(len(grid.xs) - 1):
                for j in range(len(grid.ys) - 1):
                    u, v = 0.23, 0.67
                    x = grid.xs[i] * (1-u) + grid.xs[i+1] * u
                    y = grid.ys[j] * (1-v) + grid.ys[j+1] * v
                    actual = source.height_at(x, y)
                    self.assertLessEqual(abs(actual - grid.height_at(x, y)), artifact.deviation_bound_mm)
                    z00, z10 = grid.heights[j][i:i+2]
                    z01, z11 = grid.heights[j+1][i:i+2]
                    # Independent barycentric interpolation for each diagonal.
                    diagonal_a = ((1-v)*z00 + u*z11 + (v-u)*z01 if v >= u else
                                  (1-u)*z00 + v*z11 + (u-v)*z10)
                    diagonal_b = ((1-u-v)*z00 + u*z10 + v*z01 if u+v <= 1 else
                                  (u+v-1)*z11 + (1-v)*z10 + (1-u)*z01)
                    for triangle in (diagonal_a, diagonal_b):
                        self.assertLessEqual(abs(actual - triangle), artifact.deviation_bound_mm)
            artifact.require_current(source)
            with self.assertRaisesRegex(ValueError, "stale"):
                artifact.require_current(replace(source, repeat_count=3))

    def test_two_trace_families_retain_source_features_inside_stock(self):
        source = frieze()
        x0, y0, x1, y1 = source.stock_xy
        families = []
        for family in ("spine", "section"):
            paths = source.trace_xy(family, spacing_mm=0.25, sample_step_mm=0.15)
            self.assertTrue(paths)
            for path in paths:
                self.assertGreaterEqual(len(path), 2)
                for x, y in path:
                    self.assertTrue(x0 <= x <= x1 and y0 <= y <= y1)
            families.append(paths)
        self.assertNotEqual(*families)
        # There is an explicit center trace along each authored ridge.
        expected = (2.0, 1.0)
        self.assertTrue(any(any(math.dist(p, expected) < 1e-10 for p in path)
                            for path in source.trace_xy("spine", spacing_mm=0.2, sample_step_mm=0.1)))

    def test_translated_grid_bound_uses_emitted_cell_sizes(self):
        source = ConstructiveSource("translated", 1, 2, 1, -0.75, (
            SweptRidge("tent", (0, 1), (1, 1), (1, 1), (0.5, 0.5),
                       profile="tent", cap_ends=False),), origin_xy=(0, 1e12))
        artifact = source.sample_heightfield(x_samples=2, y_samples=4)
        actual_error = abs(source.height_at(0.5, 1e12 + 1) -
                           artifact.target.height_at(0.5, 1e12 + 1))
        self.assertGreater(actual_error, 1/6)
        self.assertGreaterEqual(artifact.deviation_bound_mm, actual_error)
        self.assertGreater(max(b-a for a, b in zip(artifact.target.ys, artifact.target.ys[1:])), 2/3)

    def test_base_only_source_is_exact_and_has_no_feature_paths(self):
        source = ConstructiveSource("plane", 2, 0.5, 1, -0.5, ())
        self.assertEqual(source.gradient_bounds, (0, 0))
        self.assertEqual(source.height_at(1, 0.2), -0.5)
        low, high = source.height_bounds(source.stock_xy)
        self.assertLessEqual(low, -0.5)
        self.assertGreaterEqual(high, -0.5)
        self.assertEqual(source.trace_xy("spine", spacing_mm=0.2, sample_step_mm=0.1), ())
        self.assertLess(source.sample_heightfield(x_samples=2, y_samples=2).deviation_bound_mm, 1e-12)

    def test_rejects_cancellation_that_quantizes_spine_above_width_budget(self):
        # These finite controls cancel near X=.5, jumping the computed spine
        # from Y=0 to Y=2 despite a small claimed analytic derivative.
        with self.assertRaisesRegex(ValueError, "ill-conditioned spine"):
            SweptRidge("cancellation", (0, 1), (-1e16, -1e16+2), (0.5, 0.5), (0.2, 0.2),
                       wave_amplitude_mm=1e16, wave_cycles=0, wave_phase_rad=math.pi/2,
                       cap_ends=False)

    def test_rejects_discontinuous_repeats_invalid_sources_and_sample_budgets(self):
        ridge = SweptRidge("bad", (0, 4), (0.5, 1.5), (0.2, 0.3), (0.3, 0.4), cap_ends=False)
        with self.assertRaisesRegex(ValueError, "periodic"):
            ConstructiveSource("bad", 4, 2, 1, -0.8, (ridge,), repeat_count=2)
        with self.assertRaisesRegex(ValueError, "entire template"):
            ConstructiveSource("bad", 5, 2, 1, -0.8, (ridge,))
        with self.assertRaisesRegex(ValueError, "stock top"):
            replace(frieze(), base_height_mm=-0.1)
        with self.assertRaises(ValueError):
            replace(frieze(), scale_xyz=(1, -1, 1))
        for counts in ((1, 2), (True, 3), (1000, 1000)):
            with self.assertRaises(ValueError):
                frieze().sample_heightfield(x_samples=counts[0], y_samples=counts[1])
        with self.assertRaisesRegex(ValueError, "budget"):
            frieze().trace_xy("spine", spacing_mm=1e-6, sample_step_mm=1e-6)
        for bad in ('{}', '{"schema":"future"}', 'null', '{'):
            with self.assertRaises(ValueError):
                ConstructiveSource.from_json(bad)
        document = json.loads(frieze().canonical_json())
        document["unmodeled"] = 1
        with self.assertRaises(ValueError):
            ConstructiveSource.from_json(json.dumps(document))


class ConstructiveReconstructionTests(unittest.TestCase):
    def test_degraded_spine_recovers_declared_model_with_located_residuals(self):
        # This independent known source is deliberately quantized and perturbed.
        truth = lambda x: 0.7 + 0.08*x + 0.24*math.sin(2*math.pi*x/4 + 0.3)
        points = tuple((4*i/24, round(truth(4*i/24), 2) + (0.002 if i % 2 else -0.002))
                       for i in range(25))
        fit = reconstruct_spine(points, x_interval_mm=(0, 4), residual_limit_mm=0.012,
                                ambiguity_margin_mm=0.001)
        self.assertFalse(fit.ambiguous)
        self.assertFalse(fit.recovered_author_intent)
        self.assertEqual(fit.observations_xy, points)
        chosen = fit.candidates[fit.selected_index]
        self.assertEqual(chosen.cycles, 1)
        self.assertLessEqual(chosen.maximum_residual_mm, 0.008)
        self.assertEqual(len(chosen.residuals_mm), len(points))
        for i in range(101):
            x = 4*i/100
            predicted = (chosen.centers_mm[0] + (chosen.centers_mm[1]-chosen.centers_mm[0])*x/4 +
                         chosen.wave_amplitude_mm*math.sin(2*math.pi*chosen.cycles*x/4 + chosen.wave_phase_rad))
            self.assertLess(abs(predicted-truth(x)), 0.005)
        self.assertNotEqual(fit.input_fingerprint,
                            reconstruct_spine(points[:-1], x_interval_mm=(0, 4)).input_fingerprint)

    def test_ambiguous_and_rank_deficient_inputs_do_not_assert_intent(self):
        flat = tuple((i/8, 1.0) for i in range(9))
        ambiguous = reconstruct_spine(flat, x_interval_mm=(0, 1))
        self.assertTrue(ambiguous.ambiguous)
        self.assertIsNone(ambiguous.selected_index)
        self.assertFalse(ambiguous.recovered_author_intent)
        deficient = reconstruct_spine(((0.0, 1.0), (1.0, 1.0)), x_interval_mm=(0, 1))
        self.assertTrue(deficient.ambiguous)
        self.assertIsNone(deficient.selected_index)
        self.assertTrue(any("rank deficient" in item for item in deficient.diagnostics))
        for kwargs in ({"candidate_cycles": (1, 1)}, {"ambiguity_margin_mm": -1},
                       {"candidate_cycles": (math.nan,)}, {"residual_limit_mm": 0}):
            with self.assertRaises(ValueError):
                reconstruct_spine(flat, x_interval_mm=(0, 1), **kwargs)


if __name__ == "__main__":
    unittest.main()
