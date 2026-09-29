"""One continuous RC01/cone trace and shared stock-prefix rejection cases."""

from dataclasses import replace
import unittest

from cambam_builder.cam_core import mixed, replay


class MixedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = mixed.verify_mixed()

    def test_both_tools_share_source_order_and_stock_prefixes(self):
        result = self.result
        self.assertEqual([name for name, _ in result.stock.prefixes],
                         ["rough", "cleanup", "slot"])
        rough_end, cleanup_end, slot_end = (count for _, count in
                                           result.stock.prefixes)
        self.assertLess(rough_end, cleanup_end)
        self.assertLess(cleanup_end, slot_end)
        self.assertEqual(result.stock.motion_fingerprint,
                         result.trace.motion_fingerprint)
        self.assertEqual(result.stock.source_fingerprint,
                         result.trace.source_fingerprint)
        self.assertTrue(result.stock.residual_contains(
            "rc01-pocket", 0.1, 0.1, 1, through=cleanup_end))
        self.assertFalse(result.stock.residual_contains(
            "cone-slot", 56, 2, 1, through=slot_end))
        self.assertTrue(result.stock.residual_contains(
            "cone-slot", 50.1, 0.1, 0, through=slot_end))
        self.assertFalse(result.stock.removed_contains(56, 2, 1,
                                                        through=cleanup_end))
        self.assertTrue(result.stock.removed_contains(56, 2, 1,
                                                       through=slot_end))
        self.assertEqual(result.rc_certificate.status,
                         "partial_target_completion")
        self.assertEqual(result.slot_result.completion,
                         "partial_target_completion")

    def test_changed_source_or_reordered_motion_fails_closed(self):
        trace = self.result.trace
        with self.assertRaisesRegex(ValueError, "stale replay source"):
            replay.replay(trace, expected_source="old source")
        items = list(trace.items)
        first_slot = next(i for i, item in enumerate(items)
                          if isinstance(item, replay.Motion) and
                          item.operation == "slot" and item.role == "entry")
        items.pop(first_slot)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            replay.replay(replace(trace, items=tuple(items)),
                          expected_source=trace.source_fingerprint)

    def test_low_link_and_missing_tool_event_fail_closed(self):
        trace = self.result.trace
        items = list(trace.items)
        link = next(i for i, item in enumerate(items)
                    if isinstance(item, replay.Motion) and
                    item.operation == "slot" and item.role == "rapid")
        items[link] = replace(items[link], end=(52, 2, -0.1))
        with self.assertRaisesRegex(ValueError, "low rapid"):
            replay.replay(replace(trace, items=tuple(items)),
                          expected_source=trace.source_fingerprint)
        items = tuple(item for item in trace.items if not (
            isinstance(item, replay.Event) and item.kind == "tool_change" and
            item.tool == "cone"))
        with self.assertRaisesRegex(ValueError, "invalid spindle start"):
            replay.replay(replace(trace, items=items),
                          expected_source=trace.source_fingerprint)

    def test_shared_cone_target_guard_rejects_wall_overcut(self):
        trace = self.result.trace
        ops = list(trace.operations)
        cone = ops[-1]
        target = replace(cone.target, bounds=(50.1, 0, 62, 4))
        ops[-1] = replace(cone, target=target)
        with self.assertRaisesRegex(ValueError, "protected target"):
            replay.replay(replace(trace, operations=tuple(ops)),
                          expected_source=trace.source_fingerprint)


if __name__ == "__main__":
    unittest.main()
