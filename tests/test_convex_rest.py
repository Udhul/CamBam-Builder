"""Independent triangle and row-integration checks for supplied-motion rest."""

from dataclasses import replace
import math
import unittest

from cambam_builder.cam_core import convex_rest, replay


SOURCE = "synthetic-triangle-source-v1"
POLYGON = ((0, 0), (12, 0), (6, 8))


def prior_trace():
    target = replay.Target("triangle", (0, 0, 12, 8), 3, polygon=POLYGON)
    tool = replay.ToolProfile("T3", "pointed_cone", 3, 3)
    operation = replay.Operation("prior", tool, target)
    high, low = (3, 2, 1), (3, 2, -1.2)
    return replay.Trace(SOURCE, "synthetic-mm", high, (operation,), (
        replay.Event("tool_change", tool.name, high),
        replay.Event("spindle_start", tool.name, high),
        replay.Motion("entry", tool.name, operation.name, high, low),
        replay.Motion("retract", tool.name, operation.name, low, high),
        replay.Event("spindle_stop", tool.name, high),
    ))


def independent_sweep_area(depth, rows=16000):
    """Midpoint integrate disk-union row widths, without the core area formula."""
    start_radius, end_radius = 1.2 - depth, 2.0 - depth
    if end_radius <= 0:
        return 0.0
    slope = 0.8
    total = 0.0
    for i in range(rows):
        offset = (i + 0.5) * end_radius / rows
        first = max(0.0, (offset - start_radius) / slope)
        tangent = (offset / math.sqrt(1 - slope * slope) -
                   start_radius) / slope
        positions = (first, 1.0, min(1.0, max(first, tangent)))
        left = min(3 + u - math.sqrt(max(0.0,
                   (start_radius + slope * u) ** 2 - offset * offset))
                   for u in positions)
        right = 4 + math.sqrt(max(0.0, end_radius ** 2 - offset ** 2))
        total += 2 * (right - left) * end_radius / rows
    return total


class ConvexRestTests(unittest.TestCase):
    def setUp(self):
        self.prior = prior_trace()
        self.result = convex_rest.generate(
            self.prior, (4, 2), expected_source=SOURCE,
            expected_motion=self.prior.motion_fingerprint)

    def test_pure_rest_and_ordered_cleanup_against_independent_sections(self):
        result = self.result
        self.assertEqual(result.stock.prefixes, (("prior", 1), ("cleanup", 2)))
        self.assertEqual([m.role for m in result.trace.items if type(m) is replay.Motion],
                         ["entry", "retract", "cleared_descent", "cut", "retract"])
        cut = result.stock.cuts[-1]
        self.assertEqual((cut.a, cut.b), ((3, 2), (4, 2)))
        self.assertAlmostEqual(-cut.bottom_start, 1.2)
        self.assertAlmostEqual(-cut.bottom, 2)
        for depth, target_area, prior_area in (
                (0, 48, 48 - 1.44 * math.pi),
                (1, 64 / 3, 64 / 3 - .04 * math.pi),
                (2, 16 / 3, 16 / 3)):
            with self.subTest(depth=depth):
                self.assertAlmostEqual(result.target_area(depth), target_area,
                                       delta=1e-9)
                self.assertAlmostEqual(result.pure_rest_area(depth), prior_area,
                                       delta=1e-9)
                expected_final = target_area - independent_sweep_area(depth)
                self.assertAlmostEqual(result.residual_area(depth), expected_final,
                                       delta=.002)
                self.assertLessEqual(result.residual_area(depth), prior_area + 1e-9)
        self.assertFalse(result.pure_rest_contains(3, 2, 0))
        self.assertTrue(result.pure_rest_contains(4, 2, 1))
        self.assertFalse(result.residual_contains(4, 2, 1))
        self.assertTrue(result.residual_contains(6, 2, 1))
        self.assertEqual(result.completion, "partial_target_completion")

    def test_freshness_and_unsupported_prior_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale replay source"):
            convex_rest.generate(self.prior, (4, 2), expected_source="edited",
                                 expected_motion=self.prior.motion_fingerprint)
        with self.assertRaisesRegex(ValueError, "stale prior motion"):
            convex_rest.generate(self.prior, (4, 2), expected_source=SOURCE,
                                 expected_motion="edited")
        changed = list(self.prior.items)
        changed[2] = replace(changed[2], end=(3, 2, -1.1))
        changed[3] = replace(changed[3], start=(3, 2, -1.1))
        altered = replace(self.prior, items=tuple(changed))
        with self.assertRaisesRegex(ValueError, "stale prior motion"):
            convex_rest.generate(altered, (4, 2), expected_source=SOURCE,
                                 expected_motion=self.prior.motion_fingerprint)
        with self.assertRaisesRegex(ValueError, "original clearance"):
            convex_rest.generate(altered, (4, 2), expected_source=SOURCE,
                                 expected_motion=altered.motion_fingerprint)
        low_start = (3, 2, -0.1)
        low_items = list(self.prior.items)
        low_items[0] = replace(low_items[0], position=low_start)
        low_items[1] = replace(low_items[1], position=low_start)
        low_items[2] = replace(low_items[2], start=low_start)
        invalid_setup = replace(self.prior, initial_position=low_start,
                                items=tuple(low_items))
        with self.assertRaisesRegex(ValueError, "unsupported prior"):
            convex_rest.generate(invalid_setup, (4, 2),
                                 expected_source=SOURCE,
                                 expected_motion=invalid_setup.motion_fingerprint)

    def test_protected_target_tool_and_low_link_rejections(self):
        with self.assertRaises(ValueError):
            convex_rest.generate(self.prior, (4, 0.5), expected_source=SOURCE,
                                 expected_motion=self.prior.motion_fingerprint)
        # A shallower side-clearance endpoint is not a rising-depth cleanup.
        with self.assertRaises(ValueError):
            convex_rest.generate(self.prior, (3.1, 2.5), expected_source=SOURCE,
                                 expected_motion=self.prior.motion_fingerprint)
        target = self.prior.operations[0].target
        with self.assertRaisesRegex(ValueError, "convex"):
            replay.Target("bad", target.bounds, 3,
                          polygon=((0, 0), (12, 0), (6, 8), (6, 3)))
        with self.assertRaisesRegex(ValueError, "arithmetic is unresolved"):
            replay.Target("huge", (0, 0, 1e200, 1e200), 3,
                          polygon=((0, 0), (1e200, 0), (0, 1e200)))
        with self.assertRaisesRegex(ValueError, "arithmetic is unresolved"):
            convex_rest.generate(self.prior, (1e308, 1e308),
                                 expected_source=SOURCE,
                                 expected_motion=self.prior.motion_fingerprint)
        bad_tool = replace(self.prior.operations[0].tool,
                           radius=1, cutting_length=1)
        with self.assertRaises(ValueError):
            replay.replay(replace(self.prior,
                operations=(replace(self.prior.operations[0], tool=bad_tool),)),
                expected_source=SOURCE)
        shallow_target = replace(target, depth=1)
        with self.assertRaisesRegex(ValueError, "protected convex target"):
            replay.replay(replace(self.prior,
                operations=(replace(self.prior.operations[0],
                                    target=shallow_target),)),
                expected_source=SOURCE)
        items = list(self.result.trace.items)
        cut = items[5]
        items[5] = replace(cut, end=(4, 2, -2.1))
        items[6] = replace(items[6], start=(4, 2, -2.1))
        with self.assertRaisesRegex(ValueError, "protected convex target"):
            replay.replay(replace(self.result.trace, items=tuple(items)),
                          expected_source=self.result.trace.source_fingerprint)
        items = list(self.result.trace.items)
        items[5] = replace(items[5], end=(4, 2, -1.0))
        items[6] = replace(items[6], start=(4, 2, -1.0))
        with self.assertRaisesRegex(ValueError, "must grow in depth"):
            replay.replay(replace(self.result.trace, items=tuple(items)),
                          expected_source=self.result.trace.source_fingerprint)
        items = list(self.result.trace.items)
        items[6] = replace(items[6], role="rapid")
        with self.assertRaisesRegex(ValueError, "low rapid"):
            replay.replay(replace(self.result.trace, items=tuple(items)),
                          expected_source=self.result.trace.source_fingerprint)
        # The deep endpoint of a sloped cut does not clear a deeper column at
        # its shallow start, even though that point lies on the XY segment.
        items = list(self.result.trace.items[:-1])
        items.extend((
            replay.Motion("rapid", "T3", "cleanup", (4, 2, 1), (3, 2, 1)),
            replay.Motion("cleared_descent", "T3", "cleanup",
                          (3, 2, 1), (3, 2, -1.5)),
            replay.Event("spindle_stop", "T3", (3, 2, -1.5)),
        ))
        with self.assertRaisesRegex(ValueError, "uncleared travel/access"):
            replay.replay(replace(self.result.trace, items=tuple(items)),
                          expected_source=self.result.trace.source_fingerprint)


if __name__ == "__main__":
    unittest.main()
