"""Bounded variable-depth cone and independent row-area reference."""

from dataclasses import replace
import math
import unittest

from cambam_builder.cam_core import replay, tapered_vcarve as groove
from cambam_builder.cam_core.vcarve import Motion, PointedCone


def _independent_row_area(spine, depth, rows=8000):
    """Midpoint-integrate horizontal disk-union widths, separate from arc formula."""
    x0, x1, _, d0, d1 = spine
    if depth >= d1:
        return 0.0
    m = (d1 - d0) / (x1 - x0)
    r0, r1 = d0 - depth, d1 - depth
    dy = r1 / rows
    total = 0.0
    for i in range(rows):
        y = (i + 0.5) * dy
        first = x0 + max(0.0, (y - r0) / m)
        tangent = x0 + (y / math.sqrt(1 - m * m) - r0) / m
        candidates = (first, x1, min(x1, max(first, tangent)))
        left = min(s - math.sqrt(max(0.0,
                    (r0 + m * (s - x0)) ** 2 - y * y))
                   for s in candidates)
        right = x1 + math.sqrt(max(0.0, r1 * r1 - y * y))
        total += (right - left) * 2 * dy
    return total


class VariableVCarveTests(unittest.TestCase):
    def test_sloped_cut_replays_and_residual_matches_independent_rows(self):
        plan = groove.generate()
        result = groove.verify(plan)
        self.assertEqual(plan.cut_spine, (2, 10, 2, 1.25, 2.25))
        self.assertEqual([m.role for m in plan.motions],
                         ["plunge", "cut", "retract"])
        self.assertEqual(plan.motions[1].start, (2, 2, -1.25))
        self.assertEqual(plan.motions[1].end, (10, 2, -2.25))
        self.assertEqual(result.stock.prefixes, (("variable-v", 2),))
        self.assertEqual(result.completion, "partial_target_completion")
        expected = {0: 15.091265791880026, 1: 7.028684027518187,
                    1.5: 4.21460291488107, 2: 1.8062583920918873,
                    2.5: 0.0}
        for depth, area in expected.items():
            self.assertAlmostEqual(result.residual_area(depth), area, places=9)
            reference = (_independent_row_area(plan.target_spine, depth) -
                         _independent_row_area(plan.cut_spine, depth))
            self.assertAlmostEqual(result.residual_area(depth), reference,
                                   delta=0.0005)
        self.assertTrue(result.residual_contains(-0.5, 2, 0))
        self.assertFalse(result.residual_contains(6, 2, 1))
        self.assertTrue(result.residual_contains(12, 2, 1))
        self.assertFalse(result.residual_contains(-1.1, 2, 0))
        self.assertFalse(result.residual_contains(-1.0001, 2, 0))

    def test_edited_straight_groove_and_tool_have_independent_section_oracle(self):
        request = groove.TaperedRequest(
            target_spine=(3, 17, 2, 0.8, 2.4),
            stock_bounds=(-2, -2, 20, 6), stock_bottom=-3,
            tool=PointedCone(4, 4), cut_interval=(5, 15))
        plan = groove.generate(request)
        result = groove.verify(plan)
        self.assertEqual(plan.cut_spine[:3], (5.0, 15.0, 2.0))
        self.assertAlmostEqual(plan.cut_spine[3], 0.8 + 1.6 * 2 / 14)
        self.assertAlmostEqual(plan.cut_spine[4], 0.8 + 1.6 * 12 / 14)
        self.assertNotEqual(plan.fingerprint, groove.generate().fingerprint)
        wider_stock = groove.generate(replace(
            request, stock_bounds=(-2, -2, 21, 6)))
        self.assertNotEqual(plan.fingerprint, wider_stock.fingerprint)
        self.assertEqual(result.stock.prefixes, (("variable-v", 2),))
        for depth in (0, 0.8, 1.2, 2, 2.4):
            reference = (_independent_row_area(plan.target_spine, depth) -
                         _independent_row_area(plan.cut_spine, depth))
            self.assertAlmostEqual(result.residual_area(depth), reference,
                                   delta=0.0005)
        self.assertTrue(result.residual_contains(3, 2, 0))
        self.assertFalse(result.residual_contains(10, 2, 1))
        self.assertTrue(result.residual_contains(17, 2, 1))
        self.assertFalse(result.residual_contains(0, 2, 0))

    def test_family_rejects_invalid_tool_stock_and_spines(self):
        base = groove.TaperedRequest(
            target_spine=(3, 17, 2, 0.8, 2.4),
            stock_bounds=(-2, -2, 20, 6), stock_bottom=-3,
            tool=PointedCone(4, 4), cut_interval=(5, 15))
        invalid = (
            replace(base, target_spine=(3, 17, 2, 0.8, 0.8)),
            replace(base, target_spine=(3, 4, 2, 0.8, 2.4)),
            replace(base, target_spine=(3, 17, 2, float("nan"), 2.4)),
            replace(base, tool=PointedCone(2, 2)),
            replace(base, stock_bounds=(-2, -2, 18, 6)),
            replace(base, stock_bottom=-2),
            replace(base, cut_interval=(3, 15)),
            replace(base, cut_interval=(5, 17)),
            replace(base, safe_z=0),
        )
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                groove.generate(request)

        plan = groove.generate(base)
        trace = groove.trace_for(plan)
        items = list(trace.items)
        items[3] = replace(items[3], end=(15, 2, -2.6))
        items[4] = replace(items[4], start=(15, 2, -2.6))
        with self.assertRaisesRegex(ValueError, "crosses tapered cone target"):
            replay.replay(replace(trace, items=tuple(items)),
                          expected_source=plan.fingerprint)

    def test_stale_and_overdeep_cut_are_rejected(self):
        plan = groove.generate()
        trace = groove.trace_for(plan)
        with self.assertRaisesRegex(ValueError, "stale"):
            replay.replay(trace, expected_source="changed")
        changed = (Motion("plunge", (2, 2, 1), (2, 2, -1.25)),
                   Motion("cut", (2, 2, -1.25), (10, 2, -2.35)),
                   Motion("retract", (10, 2, -2.35), (10, 2, 1)))
        bad = replace(plan, cut_spine=(2, 10, 2, 1.25, 2.35),
                      motions=changed)
        with self.assertRaisesRegex(ValueError, "crosses target"):
            groove.verify(bad)
        items = list(trace.items)
        items[3] = replace(items[3], end=(10, 2, -2.35))
        items[4] = replace(items[4], start=(10, 2, -2.35))
        with self.assertRaisesRegex(ValueError, "crosses tapered cone target"):
            replay.replay(replace(trace, items=tuple(items)),
                          expected_source=plan.fingerprint)
        items = list(trace.items)
        items[2] = replace(items[2], end=(2, 2, -1.3))
        with self.assertRaises(ValueError):
            replay.replay(replace(trace, items=tuple(items)),
                          expected_source=plan.fingerprint)


if __name__ == "__main__":
    unittest.main()
