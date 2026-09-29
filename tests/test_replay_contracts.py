"""Public replay value boundaries, independent of generated reference jobs."""

from dataclasses import replace
import math
import unittest

from cambam_builder.cam_core import replay


class ReplayTargetContractTests(unittest.TestCase):
    def targets(self):
        base = replay.Target("target", (0, 0, 20, 20), 3)
        triangle = ((2, 2), (18, 2), (10, 18))
        return (
            ("bounds", base),
            ("island", replace(base, island=(8, 8, 12, 12))),
            ("cone_spine", replace(base, cone_spine=(5, 10, 10, 1, 2))),
            ("polygon", replace(base, polygon=triangle)),
            ("region_shell", replace(base, region_shell=triangle)),
            ("region_holes", replace(base, region_shell=triangle,
                region_holes=(((9, 5), (11, 5), (10, 7)),))),
        )

    def test_tuple_geometry_is_retained_and_hashable(self):
        for field, target in self.targets():
            with self.subTest(field=field):
                value = getattr(target, field)
                twin = replace(target, **{field: value})
                self.assertEqual(twin, target)
                self.assertEqual(hash(twin), hash(target))
                self.assertIs(getattr(twin, field), value)

    def test_mutable_geometry_rejected_at_construction(self):
        for field, target in self.targets():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "target|polygon|ring"):
                    replace(target, **{field: list(getattr(target, field))})

    def test_empty_mutable_optional_geometry_rejected(self):
        target = replay.Target("target", (0, 0, 20, 20), 3)
        for field in ("island", "cone_spine", "polygon", "region_shell",
                      "region_holes"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "replay target"):
                    replace(target, **{field: []})

    def test_protected_island_has_finite_ordered_in_bounds_geometry(self):
        target = replay.Target("target", (0, 0, 20, 20), 3,
                               island=(8, 8, 12, 12))
        tool = replay.ToolProfile("T1", "cylinder", 1, 3)
        with self.assertRaisesRegex(ValueError, "protected island"):
            replay._safe_cut(replay.Sweep("cut", tool, (10, 10),
                                          (10, 10), -1), target)
        for island in ((12, 8, 8, 12), (8, 12, 12, 8),
                       (-1, 8, 12, 12), (8, 8, 21, 12),
                       (8, 8, 12, math.nan), (8, 8, 12, math.inf)):
            with self.subTest(island=island):
                with self.assertRaisesRegex(ValueError, "protected island"):
                    replace(target, island=island)

    def test_shared_names_require_identical_tool_and_target_values(self):
        target = replay.Target("field", (0, 0, 10, 10), 2)
        tool = replay.ToolProfile("T1", "cylinder", 1, 3)
        op = replay.Operation("first", tool, target)
        at = (5, 5, 2)
        items = (replay.Event("tool_change", "T1", at),
                 replay.Event("spindle_start", "T1", at),
                 replay.Motion("entry", "T1", "first", at, (5, 5, -1), 60),
                 replay.Motion("retract", "T1", "first", (5, 5, -1), at, 60),
                 replay.Event("spindle_stop", "T1", at))
        trace = replay.Trace("source", "XY", at,
                             (op, replay.Operation("second", tool, target)),
                             items)
        self.assertEqual(replay.replay(trace, expected_source="source").targets,
                         (target,))
        for second in (
                replay.Operation("second", tool,
                                 replace(target, bounds=(0, 0, 20, 20))),
                replay.Operation("second", replace(tool, radius=1.5), target)):
            with self.subTest(second=second):
                with self.assertRaisesRegex(ValueError, "conflicting replay"):
                    replace(trace, operations=(op, second))

    def test_numeric_fields_reject_nonfinite_or_nonreal_values(self):
        for value in (math.nan, math.inf, True, "1", 10 ** 400):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "tool profile"):
                    replay.ToolProfile("T1", "cylinder", value, 3)
                with self.assertRaisesRegex(ValueError, "replay target"):
                    replay.Target("target", (value, 0, 20, 20), 3)
                with self.assertRaisesRegex(ValueError, "finite XYZ"):
                    replay.Motion("cut", "T1", "cut", (value, 5, -1),
                                  (10, 5, -1), 100)
                with self.assertRaisesRegex(ValueError, "replay motion"):
                    replay.Motion("cut", "T1", "cut", (5, 5, -1),
                                  (10, 5, -1), value)

    def test_affine_cone_contains_common_tangent_envelope_and_closed_tip(self):
        tool = replay.ToolProfile("V", "pointed_cone", 5, 5)
        # The 3-4-5 spine has depth slope 3/5. Its common tangent at
        # along-coordinate 2 and section depth 1/2 has offset
        # (1 - 1/2 + (3/5)*2) / (4/5) = 17/8 mm.
        cut = replay.Sweep("cut", tool, (0, 0), (3, 4), -4, -1)
        for offset, expected in ((17 / 8 - 1e-6, True),
                                 (17 / 8 + 1e-6, False)):
            x, y = 2 * 3 / 5 - offset * 4 / 5, 2 * 4 / 5 + offset * 3 / 5
            self.assertIs(cut.contains(x, y, 0.5), expected)
        self.assertTrue(cut.contains(3, 4, 4))
        self.assertFalse(cut.contains(3.000001, 4, 4))
        flat = replay.Sweep("flat", tool, (0, 0), (3, 4), -4)
        self.assertTrue(flat.contains(1.5, 2, 4))
        self.assertFalse(flat.contains(1.5, 2.000001, 4))

    def test_descending_cylinder_section_is_clipped_at_reached_depth(self):
        cut = replay.Sweep("cut", replay.ToolProfile("T1", "cylinder", 0.25, 3),
                           (0, 0), (4, 0), -3, -1)
        self.assertEqual(cut.section_segment(2), ((2.0, 0.0), (4, 0)))
        self.assertTrue(cut.contains(1.75, 0, 2))
        self.assertFalse(cut.contains(1.75 - 1e-6, 0, 2))
        self.assertTrue(cut.contains(4, 0, 3))
        self.assertFalse(cut.contains(0, 0, 3))
        self.assertIsNone(cut.section_segment(3.000001))

    def test_tuple_target_replays_supplied_motion_and_retains_fingerprint(self):
        target = replay.Target("target", (0, 0, 20, 20), 3)
        tool = replay.ToolProfile("T1", "cylinder", 1, 3)
        operation = replay.Operation("cut", tool, target)
        top, low, end, safe = (5, 5, 2), (5, 5, -1), (10, 5, -1), (10, 5, 2)
        trace = replay.Trace("source", "XY", top, (operation,), (
            replay.Event("tool_change", "T1", top),
            replay.Event("spindle_start", "T1", top),
            replay.Motion("entry", "T1", "cut", top, low, 60),
            replay.Motion("cut", "T1", "cut", low, end, 100),
            replay.Motion("retract", "T1", "cut", end, safe, 60),
            replay.Event("spindle_stop", "T1", safe),
        ))
        fingerprint = trace.motion_fingerprint
        result = replay.replay(trace, expected_source="source")
        self.assertEqual(result.motion_fingerprint, fingerprint)
        self.assertEqual(trace.motion_fingerprint, fingerprint)
        self.assertEqual(result.targets, (target,))
        self.assertTrue(result.removed_contains(7, 5, 1))
        self.assertFalse(result.residual_contains("target", 7, 5, 1))
        self.assertTrue(result.residual_contains("target", 7, 8, 1))


if __name__ == "__main__":
    unittest.main()
