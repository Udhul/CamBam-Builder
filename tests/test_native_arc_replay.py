"""Independent analytic check for bounded continuous native arc sweeps."""

import math
import unittest

from cambam_builder.cam_core import polygon_rest, replay


class NativeArcReplayTests(unittest.TestCase):
    def test_semicircular_cutter_tube_encloses_analytic_area(self):
        target = replay.Target("field", (0, 0, 16, 16), 1,
                               region_shell=((0, 0), (16, 0),
                                             (16, 16), (0, 16)))
        tool = replay.ToolProfile("T1", "cylinder", 0.5, 2)
        operation = replay.Operation("native-arc", tool, target)
        top, low = (5, 8, 5), (5, 8, -1)
        end, safe = (11, 8, -1), (11, 8, 5)
        trace = replay.Trace("analytic-semicircle", "XY", top, (operation,), (
            replay.Event("tool_change", "T1", top),
            replay.Event("spindle_start", "T1", top),
            replay.Motion("entry", "T1", operation.name, top, low, 60),
            replay.ArcMotion("cut", "T1", operation.name, low, end,
                             240, 2, (8, 8)),
            replay.Motion("retract", "T1", operation.name, end, safe, 60),
            replay.Event("spindle_stop", "T1", safe),
        ))
        stock = replay.replay(trace, expected_source="analytic-semicircle")
        self.assertGreater(len(stock.cuts), 100)
        inner = polygon_rest._cut_polygon(stock.cuts, 1,
                                           radial_error=-1e-6).area
        outer = polygon_rest._cut_polygon(stock.cuts, 1,
                                           radial_error=1e-6).area
        exact = math.pi * (2 * 3 * 0.5 + 0.5 ** 2)
        self.assertLessEqual(inner, exact)
        self.assertGreaterEqual(outer, exact)
        self.assertLess(outer - inner, 0.02)


if __name__ == "__main__":
    unittest.main()
