"""Public replay value boundaries, independent of generated reference jobs."""

from dataclasses import replace
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
