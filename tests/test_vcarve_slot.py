"""Independent section references for the bounded pointed-cone slot."""

import math
import unittest

from cambam_builder.cam_core.vcarve import (
    Motion, Pass, PointedCone, Slot, SlotPlan, generate_slot, verify_slot,
)


class PointedConeSlotTests(unittest.TestCase):
    def test_full_depth_v_slot_has_finite_motion_and_corner_rest(self):
        plan = generate_slot()
        result = verify_slot(plan)
        self.assertEqual(len(plan.passes), 1)
        self.assertEqual([m.role for m in plan.motions],
                         ["plunge", "cut", "retract"])
        self.assertEqual(plan.passes[0], Pass(2, 10, 2, 2))
        self.assertEqual(plan.motions[1].start, (2, 2, -2))
        self.assertEqual(plan.motions[1].end, (10, 2, -2))
        self.assertAlmostEqual(result.residual_area(0), 16 - 4 * math.pi, places=12)
        self.assertAlmostEqual(result.residual_area(1), 4 - math.pi, places=12)
        self.assertEqual(result.residual_area(2), 0)
        lower, upper = result.residual_volume_bounds()
        self.assertLessEqual(lower, (4 - math.pi) * 8 / 3)
        self.assertGreaterEqual(upper, (4 - math.pi) * 8 / 3)
        self.assertLess(upper - lower, 0.05)
        self.assertTrue(result.residual_contains(0.1, 0.1, 0))
        self.assertFalse(result.residual_contains(6, 2, 1))
        self.assertFalse(result.residual_contains(0, 0, 1))
        self.assertEqual(plan.tool.radius_at_height(2), 2)
        self.assertEqual(result.evidence_class, "conditional_analytic_cone_slot")
        self.assertEqual(result.completion, "partial_target_completion")

    def test_capped_slot_has_extra_paths_and_genuine_floor_rest(self):
        plan = generate_slot(Slot(depth_cap=1))
        result = verify_slot(plan)
        self.assertEqual([(p.x0, p.x1, p.y, p.tip_depth) for p in plan.passes],
                         [(1, 11, 1, 1), (1, 11, 2, 1), (1, 11, 3, 1)])
        self.assertEqual([m.role for m in plan.motions],
                         ["plunge", "cut", "retract", "rapid",
                          "plunge", "cut", "retract", "rapid",
                          "plunge", "cut", "retract"])
        self.assertEqual(plan.motions[3].start[1:], (1, 1))
        self.assertEqual(plan.motions[3].end[1:], (2, 1))
        self.assertAlmostEqual(result.residual_area(0.5),
                               3 - 3 * math.pi / 4, places=12)
        self.assertEqual(result.residual_area(1), 20)
        self.assertEqual(result.completion, "partial_target_completion")
        self.assertTrue(result.residual_contains(6, 1.5, 1))
        self.assertFalse(result.residual_contains(6, 2, 1))
        self.assertGreater(result.residual_area(0), 0)
        self.assertLess(result.residual_area(0), 16 - math.pi)
        center_only = verify_slot(SlotPlan(plan.slot, plan.tool,
                                           plan.passes[:1], plan.motions[:3]))
        self.assertAlmostEqual(center_only.residual_area(0),
                               28 - math.pi, places=12)
        self.assertGreater(center_only.residual_area(0), result.residual_area(0))
        lower, upper = result.residual_volume_bounds()
        self.assertGreater(lower, 0)
        self.assertLess(upper - lower, 0.05)

    def test_independent_row_integration_agrees_between_sections(self):
        # Integrate direct disk membership along each horizontal row, separately
        # from the production Voronoi/antiderivative area calculation.
        for plan in (generate_slot(), generate_slot(Slot(depth_cap=1))):
            result = verify_slot(plan)
            for depth in (0, plan.slot.depth_cap / 4,
                          plan.slot.depth_cap / 2, 3 * plan.slot.depth_cap / 4):
                low, high = depth, plan.slot.width - depth
                steps = 20000
                dy = (high - low) / steps
                removed = 0.0
                for i in range(steps):
                    y = low + (i + 0.5) * dy
                    intervals = []
                    for path in plan.passes:
                        radius = path.tip_depth - depth
                        if radius >= abs(y - path.y):
                            reach = math.sqrt(max(0, radius ** 2 - (y - path.y) ** 2))
                            intervals.append((path.x0 - reach, path.x1 + reach))
                    intervals.sort()
                    end = low
                    for left, right in intervals:
                        removed += max(0, right - max(left, end)) * dy
                        end = max(end, right)
                approximate = plan.slot.section_area(depth) - removed
                self.assertAlmostEqual(result.residual_area(depth), approximate,
                                       delta=0.002)

    def test_tool_depth_and_all_height_wall_guards(self):
        with self.assertRaisesRegex(ValueError, "exceeds cone"):
            generate_slot(tool=PointedCone(1.5, 1.5))
        with self.assertRaisesRegex(ValueError, "requires radius"):
            PointedCone(3, 2)
        with self.assertRaisesRegex(ValueError, "unsupported rectangular"):
            Slot(depth_cap=2.1)
        good = generate_slot()
        for bad_pass in (Pass(1.99, 10, 2, 2), Pass(2, 10, 1.99, 2),
                         Pass(2, 10, 2, 2.01)):
            paths = (bad_pass,)
            plan = SlotPlan(good.slot, good.tool, paths,
                            (Motion("plunge", (bad_pass.x0, bad_pass.y, 1),
                                    (bad_pass.x0, bad_pass.y, -bad_pass.tip_depth)),
                             Motion("cut", (bad_pass.x0, bad_pass.y, -bad_pass.tip_depth),
                                    (bad_pass.x1, bad_pass.y, -bad_pass.tip_depth)),
                             Motion("retract", (bad_pass.x1, bad_pass.y, -bad_pass.tip_depth),
                                    (bad_pass.x1, bad_pass.y, 1))))
            with self.assertRaisesRegex(ValueError, "crosses target"):
                verify_slot(plan)

    def test_missing_or_changed_access_motion_is_rejected(self):
        good = generate_slot(Slot(depth_cap=1))
        for moves in (good.motions[:-1],
                      good.motions[:3] + good.motions[4:],
                      good.motions[:3] + (Motion("rapid", (11, 1, -0.1),
                                                        (1, 2, -0.1)),) +
                      good.motions[4:]):
            with self.assertRaisesRegex(ValueError, "complete ordered"):
                verify_slot(SlotPlan(good.slot, good.tool, good.passes, moves))


if __name__ == "__main__":
    unittest.main()
