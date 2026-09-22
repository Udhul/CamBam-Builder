"""Independent analytic acceptance for conditional directional sweep bounds."""

from fractions import Fraction as Q
import math
import unittest

from cambam_builder.planar import Rectangle
from cambam_builder.stock import (
    Capsule, HorizontalSweep, RemainingSection, SectionRectangle, bound_horizontal_sweep,
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


if __name__ == "__main__":
    unittest.main()
