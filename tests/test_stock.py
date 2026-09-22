"""Independent analytic acceptance for conditional directional sweep bounds."""

from fractions import Fraction as Q
from dataclasses import replace
import math
import unittest

from cambam_builder.planar import Rectangle
from cambam_builder.stock import (
    Capsule, CapsuleUnion, HorizontalSweep, RemainingSection, SectionRectangle,
    SectionTarget, TargetRestSection, bound_horizontal_sweep, compose_sweep_bounds,
    compose_target_rest_bounds,
)


class StockBoundsTests(unittest.TestCase):
    def bounds(self, **kwargs):
        args = dict(stock=SectionRectangle(0, 0, 20, 10),
                    sweep=HorizontalSweep(4, 16, 5), frame_id="fixture",
                    section_z_mm=-2, radius_min_mm=2, radius_max_mm=2,
                    position_error_mm=0)
        args.update(kwargs)
        return bound_horizontal_sweep(**args)

    def test_exact_sweep_and_rest_area(self):
        b = self.bounds()
        self.assertEqual(b.removal_lower, b.removal_upper)
        # Rectangle strip plus two semicircles, independent scalar reference.
        removed = 48 + 4 * math.pi
        lo, hi = b.removal_upper.area_interval
        self.assertLess(float(lo), removed)
        self.assertGreater(float(hi), removed)
        lo, hi = b.remaining_upper.area_interval
        self.assertLess(float(lo), 200 - removed)
        self.assertGreater(float(hi), 200 - removed)
        self.assertLess(float(hi - lo), 0.00034)
        self.assertTrue(b.remaining_lower.contains(1, 1))
        self.assertFalse(b.remaining_upper.contains(10, 5))
        # Feasible center elsewhere is not a swept/removal certificate.
        self.assertTrue(b.remaining_lower.contains(10, 2))

    def test_uncertainty_envelopes_and_independent_actual_sweeps(self):
        b = self.bounds(radius_min_mm=Q(7, 4), radius_max_mm=Q(9, 4),
                        position_error_mm=Q(1, 4))
        self.assertEqual(b.removal_lower.radius, Q(3, 2))
        self.assertEqual(b.removal_upper.radius, Q(5, 2))
        # Exact grid membership in independently translated disks/strip.
        for dx, dy in ((Q(1, 4), 0), (0, Q(-1, 4)),
                       (Q(3, 20), Q(1, 5)), (0, 0)):
            for radius in (Q(7, 4), 2, Q(9, 4)):
                for i in range(81):
                    for j in range(41):
                        x, y = Q(i, 4), Q(j, 4)
                        left = (x - 4 - dx) ** 2 + (y - 5 - dy) ** 2 <= radius ** 2
                        right = (x - 16 - dx) ** 2 + (y - 5 - dy) ** 2 <= radius ** 2
                        strip = 4 + dx <= x <= 16 + dx and abs(y - 5 - dy) <= radius
                        actual = left or right or strip
                        self.assertFalse(b.removal_lower.contains(x, y) and not actual)
                        self.assertFalse(actual and not b.removal_upper.contains(x, y))
                        self.assertFalse(b.remaining_lower.contains(x, y) and actual)
                        self.assertFalse(not actual and not b.remaining_upper.contains(x, y))

    def test_contact_and_exact_rejection_beyond_boundary(self):
        self.bounds(sweep=HorizontalSweep(2, 18, 2))
        for change in (dict(sweep=HorizontalSweep(math.nextafter(2, 0), 18, 2)),
                       dict(position_error_mm=Q(1, 10**30),
                            sweep=HorizontalSweep(2, 18, 2))):
            with self.assertRaisesRegex(ValueError, "protected"):
                self.bounds(**change)

    def test_empty_and_lower_dimensional_guarantees(self):
        b = self.bounds(radius_min_mm=1, position_error_mm=1)
        self.assertEqual(b.removal_lower.radius, 0)
        self.assertTrue(b.removal_lower.contains(10, 5))
        self.assertFalse(b.removal_lower.contains(10, Q(5001, 1000)))
        b = self.bounds(radius_min_mm=1, position_error_mm=Q(3, 2))
        self.assertIsNone(b.removal_lower)
        self.assertEqual(b.remaining_upper.area_interval, (200, 200))
        b = self.bounds(sweep=HorizontalSweep(10, 10, 5))
        self.assertTrue(b.removal_lower.contains(12, 5))
        self.assertFalse(b.remaining_lower.contains(12, 5))

    def test_translation_and_fraction_precision(self):
        n = 10**30
        b = self.bounds(stock=SectionRectangle(n, n, n+20, n+10),
                        sweep=HorizontalSweep(n+4, n+16, n+5))
        self.assertEqual(b.removal_upper.area_interval,
                         self.bounds().removal_upper.area_interval)
        self.assertTrue(b.removal_upper.contains(n+2, n+5))

    def test_invalid_and_uncertified_inputs(self):
        with self.assertRaisesRegex(ValueError, "wholly inside"):
            RemainingSection(SectionRectangle(0, 0, 1, 1),
                             Capsule(HorizontalSweep(5, 6, 5), 2))
        for change in (dict(stock=Rectangle((10, 5), 20, 10)),
                       dict(sweep=((4, 5), (16, 5))), dict(frame_id=""),
                       dict(section_z_mm=None), dict(radius_min_mm=0),
                       dict(radius_min_mm=3), dict(radius_max_mm=math.inf),
                       dict(position_error_mm=-1), dict(radius_min_mm=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.bounds(**change)


class ComposedStockTests(unittest.TestCase):
    stock = SectionRectangle(0, 0, 20, 10)

    def source(self, xmin=4, xmax=10, y=5, **kwargs):
        args = dict(frame_id="fixture", section_z_mm=-2,
                    radius_min_mm=2, radius_max_mm=2)
        args.update(kwargs)
        return bound_horizontal_sweep(self.stock, HorizontalSweep(xmin, xmax, y), **args)

    def compose(self, sources, **kwargs):
        args = dict(frame_id="fixture", section_z_mm=-2, grid_size=16)
        args.update(kwargs)
        return compose_sweep_bounds(self.stock, sources, **args)

    def test_overlap_disjoint_and_identical_area_references(self):
        cases = (
            # Collinear overlap is a single longer capsule.
            ([self.source(), self.source(8, 16)], 48 + 4 * math.pi),
            # Two disjoint disks.
            ([self.source(4, 4), self.source(16, 16)], 8 * math.pi),
            ([self.source(), self.source()], 24 + 4 * math.pi),
            # Perpendicular row separation 2: independent equal-disk lens formula.
            ([self.source(10, 10, 4), self.source(10, 10, 6)],
             8 * math.pi - (8 * math.acos(0.5) - math.sqrt(12))),
        )
        for sources, reference in cases:
            with self.subTest(reference=reference):
                b = self.compose(sources, grid_size=32)
                lo, hi = b.removal_upper.area_interval
                self.assertLessEqual(float(lo), reference)
                self.assertGreaterEqual(float(hi), reference)
                self.assertLess(float(hi - lo), 25)
                self.assertEqual(b.remaining_lower.area_interval,
                                 (200 - hi, 200 - lo))
        one = self.compose([self.source()])
        duplicate = self.compose([self.source()] * 3)
        self.assertEqual(one.removal_upper.area_interval,
                         duplicate.removal_upper.area_interval)
        self.assertEqual(len(duplicate.sources), 3)

    def test_prefix_monotonicity_order_and_empty(self):
        sources = [self.source(), self.source(8, 16), self.source(4, 4, 2)]
        previous = self.compose([])
        self.assertEqual(previous.remaining_lower.area_interval, (200, 200))
        self.assertFalse(previous.removal_upper.contains(4, 5))
        for i in range(1, 4):
            current = self.compose(sources[:i])
            for name in ("remaining_lower", "remaining_upper"):
                old, new = getattr(previous, name), getattr(current, name)
                self.assertTrue(all(a <= b for a, b in zip(new.area_interval,
                                                          old.area_interval)))
                for x in range(21):
                    for y in range(11):
                        self.assertFalse(new.contains(x, y) and not old.contains(x, y))
            previous = current
        reverse = self.compose(reversed(sources))
        self.assertEqual(reverse.removal_upper.area_interval,
                         previous.removal_upper.area_interval)
        self.assertEqual(reverse.sources, tuple(reversed(sources)))

    def test_uncertain_composition_against_independent_actual_union(self):
        sources = [self.source(4, 10, radius_min_mm=Q(7, 4),
                               radius_max_mm=Q(9, 4), position_error_mm=Q(1, 4)),
                   self.source(8, 16, radius_min_mm=1,
                               position_error_mm=Q(3, 2))]
        b = self.compose(sources)
        # Different actual shifts/radii per pass, within supplied uncertainty.
        actual = [(Q(17, 4), Q(41, 4), 5, 2), (8, 16, 6, Q(3, 2))]
        for i in range(41):
            for j in range(21):
                x, y = Q(i, 2), Q(j, 2)
                removed = any(
                    ((left <= x <= right and abs(y-cy) <= r)
                     or (x-left)**2 + (y-cy)**2 <= r*r
                     or (x-right)**2 + (y-cy)**2 <= r*r)
                    for left, right, cy, r in actual)
                self.assertFalse(b.removal_lower.contains(x, y) and not removed)
                self.assertFalse(removed and not b.removal_upper.contains(x, y))
                self.assertFalse(b.remaining_lower.contains(x, y) and removed)
                self.assertFalse(not removed and not b.remaining_upper.contains(x, y))
        self.assertEqual(b.sources, tuple(sources))

    def test_collapsed_guarantees_contact_and_refinement(self):
        b = self.compose([self.source(radius_min_mm=1, position_error_mm=1)])
        self.assertTrue(b.removal_lower.contains(6, 5))
        self.assertEqual(b.removal_lower.area_interval, (0, 0))
        self.assertFalse(b.remaining_upper.contains(6, 5))
        b = self.compose([self.source(radius_min_mm=1, position_error_mm=Q(3, 2))])
        self.assertEqual(b.remaining_upper.area_interval, (200, 200))
        sources = [self.source(2, 18, 2), self.source(2, 18, 6)]
        coarse, fine = self.compose(sources), self.compose(sources, grid_size=32)
        lo, hi = coarse.removal_upper.area_interval
        flo, fhi = fine.removal_upper.area_interval
        self.assertLessEqual(lo, flo)
        self.assertLessEqual(fhi, hi)
        self.assertTrue(fine.removal_lower.contains(10, 4))
        self.assertFalse(fine.remaining_upper.contains(10, 0))

    def test_mismatches_invalid_and_forged_certificates(self):
        source = self.source()
        for sources in ([self.source(frame_id="other")],
                        [self.source(section_z_mm=-3)],
                        [replace(source, stock=SectionRectangle(0, 0, 30, 10))],
                        [replace(source, removal_lower=None)],
                        [replace(source, evidence_class="nominal")], [object()], None):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                self.compose(sources)
        for change in (dict(grid_size=0), dict(grid_size=True), dict(grid_size=1.5),
                       dict(frame_id=""), dict(section_z_mm=math.nan)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compose([], **change)


class HoledTargetStockTests(unittest.TestCase):
    stock = SectionRectangle(0, 0, 30, 20)
    target = SectionTarget(stock, SectionRectangle(2, 2, 28, 18),
                           SectionRectangle(13, 7, 17, 13))

    def source(self, xmin, xmax, y, radius, error=0, radius_max=None):
        return bound_horizontal_sweep(
            self.stock, HorizontalSweep(xmin, xmax, y), frame_id="fixture",
            section_z_mm=-2, radius_min_mm=radius,
            radius_max_mm=radius if radius_max is None else radius_max,
            position_error_mm=error)

    def compose(self, sources, **kwargs):
        return compose_target_rest_bounds(
            self.target, sources, frame_id="fixture", section_z_mm=-2,
            grid_size=32, **kwargs)

    @staticmethod
    def actual_contains(sweep, radius, x, y, dx=0, dy=0):
        left, right, center_y = sweep.xmin + dx, sweep.xmax + dx, sweep.y + dy
        nearest_x = min(max(x, left), right)
        return (x - nearest_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    @staticmethod
    def target_contains(x, y):
        return (2 <= x <= 28 and 2 <= y <= 18 and
                not (13 < x < 17 and 7 < y < 13))

    def test_original_target_rest_and_safe_crossing_of_cleared_interface(self):
        large = self.source(6, 24, 5, 2)
        small = self.source(3, 10, 5, 1)
        near_island = self.source(8, 12, 8, 1)
        initial = self.compose([])
        self.assertEqual(initial.rest_lower.area_interval, (392, 392))
        self.assertEqual(initial.stock_bounds.remaining_lower.area_interval, (600, 600))
        self.assertTrue(initial.rest_upper.contains(13, 10))
        self.assertFalse(initial.rest_upper.contains(15, 10))
        before = self.compose([large])
        after = self.compose([large, small, near_island])
        self.assertTrue(before.rest_lower.contains(3, 5))
        self.assertFalse(after.rest_upper.contains(3, 5))
        self.assertTrue(large.removal_upper.contains(8, 5))
        self.assertTrue(small.removal_upper.contains(8, 5))
        self.assertFalse(after.rest_lower.contains(15, 10))
        self.assertTrue(after.stock_bounds.remaining_lower.contains(15, 10))
        self.assertTrue(after.rest_upper.contains(12, 10))
        self.assertFalse(after.rest_upper.contains(12, 8))
        self.assertEqual(after.stock_bounds.sources, (large, small, near_island))
        reference = 320 - 4 * math.pi
        lo, hi = before.rest_lower.area_interval
        self.assertLessEqual(float(lo), reference)
        self.assertGreaterEqual(float(hi), reference)
        for ix in range(121):
            for iy in range(81):
                x, y = Q(ix, 4), Q(iy, 4)
                removed = any(self.actual_contains(s.sweep, s.radius_min_mm, x, y)
                              for s in (large, small, near_island))
                expected_rest = self.target_contains(x, y) and not removed
                self.assertEqual(after.rest_lower.contains(x, y), expected_rest)
                self.assertEqual(after.rest_upper.contains(x, y), expected_rest)
                self.assertEqual(after.stock_bounds.remaining_lower.contains(x, y),
                                 not removed)
                self.assertFalse(removed and not self.target_contains(x, y))

    def test_uncertain_sources_bound_independent_actual_removal(self):
        sources = (
            self.source(6, 24, Q(9, 2), Q(7, 4), Q(1, 4), 2),
            self.source(8, 11, 8, Q(3, 4), Q(1, 4), 1),
        )
        result = self.compose(sources)
        actual = ((sources[0].sweep, 2, Q(1, 4), 0),
                  (sources[1].sweep, Q(3, 4), 0, Q(-1, 4)))
        for ix in range(121):
            for iy in range(81):
                x, y = Q(ix, 4), Q(iy, 4)
                removed = any(self.actual_contains(s, r, x, y, dx, dy)
                              for s, r, dx, dy in actual)
                required = self.target_contains(x, y)
                self.assertFalse(result.stock_bounds.removal_lower.contains(x, y)
                                 and not removed)
                self.assertFalse(removed and not result.stock_bounds.removal_upper.contains(x, y))
                self.assertFalse(result.rest_lower.contains(x, y) and
                                 (not required or removed))
                self.assertFalse(required and not removed and
                                 not result.rest_upper.contains(x, y))
                self.assertFalse(removed and not required)

    def test_protected_walls_tangency_and_overrun(self):
        self.compose([self.source(6, 24, 5, 2)])  # island top tangent
        self.compose([self.source(4, 10, 4, 2)])  # target exterior tangent
        epsilon = Q(1, 10**30)
        for unsafe in (self.source(6, 24, 5, 2 + epsilon),
                       self.source(4, 10, 4 - epsilon, 2)):
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(ValueError, "protected"):
                self.compose([unsafe])

    def test_invalid_target_forged_source_and_direct_rest(self):
        for args in ((self.stock, SectionRectangle(0, 0, 31, 18),
                      SectionRectangle(13, 7, 17, 13)),
                     (self.stock, SectionRectangle(2, 2, 28, 18),
                      SectionRectangle(2, 7, 17, 13)),
                     (self.stock, SectionRectangle(2, 2, 28, 18), object())):
            with self.subTest(args=args), self.assertRaises(ValueError):
                SectionTarget(*args)
        with self.assertRaises(ValueError):
            self.compose([replace(self.source(6, 24, 5, 2), removal_upper=None)])
        with self.assertRaisesRegex(ValueError, "protected"):
            TargetRestSection(self.target, CapsuleUnion(
                self.stock, (Capsule(HorizontalSweep(10, 15, 9), 1),)))
        with self.assertRaisesRegex(ValueError, "protected"):
            TargetRestSection(self.target, CapsuleUnion(
                self.stock, (Capsule(HorizontalSweep(14, 16, 9), 0),)))
        TargetRestSection(self.target, CapsuleUnion(
            self.stock, (Capsule(HorizontalSweep(14, 16, 7), 0),)))


if __name__ == "__main__":
    unittest.main()
