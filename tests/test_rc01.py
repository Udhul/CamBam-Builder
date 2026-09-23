"""Independent acceptance for RC01's ordered, full-height synthetic sequence."""

from dataclasses import replace
import math
import unittest

from cambam_builder.rc01 import Event, Job, Move, _allowed_cut, generate, verify


class RC01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = generate()

    def altered(self, index, **changes):
        items = list(self.program.items)
        items[index] = replace(items[index], **changes)
        return replace(self.program, items=tuple(items))

    def test_standalone_sequence_and_independent_corner_oracle(self):
        program = self.program
        self.assertEqual(program, generate())
        self.assertEqual(program.items[0].kind, "tool_change")
        self.assertEqual(program.items[1].rpm, 12000)
        self.assertEqual(program.items[1].direction, "CW")
        self.assertEqual(program.items[-1].kind, "spindle_stop")
        self.assertEqual([i.tool for i in program.items if isinstance(i, Event)
                          and i.kind == "tool_change"], ["T1", "T2"])
        first_entry = next(i for i in program.items if isinstance(i, Move)
                           and i.role == "entry")
        self.assertEqual(first_entry.end, (5, 5, -1))
        self.assertEqual(first_entry.feed, 60)
        cert = verify(program)
        self.assertEqual(cert.status, "partial_target_completion")
        self.assertEqual(len(cert.rough_rest_by_depth), 3)
        self.assertEqual(len(cert.final_rest_by_depth), 3)
        rough_ideal = (4 - math.pi) * 9
        final_ideal = 4 - math.pi
        for rough, final in zip(cert.rough_rest_by_depth, cert.final_rest_by_depth):
            self.assertLessEqual(rough_ideal - 1e-5, rough[0])
            self.assertLess(rough[1], rough_ideal + 0.5)
            self.assertLessEqual(final_ideal - 1e-5, final[0])
            self.assertLess(final[1], final_ideal + 0.5)
            self.assertGreater(rough[0] - final[1],
                               rough_ideal - final_ideal - 0.5)
        self.assertLess(cert.rough_rest_volume[1], 3 * (rough_ideal + 0.5))
        self.assertLess(cert.final_rest_volume[1], 3 * (final_ideal + 0.5))
        self.assertEqual(cert.area_coordinate_enclosure_mm, 1e-9)
        self.assertLess(cert.location_polygon_sagitta_mm, 1e-6)
        self.assertIsNone(cert.location_numeric_enclosure_mm)

    def test_stale_and_uncertain_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            verify(self.program, replace(Job(), floor=-4), measure_rest=False)
        uncertain = replace(Job(), position_error=0.005, radius_error=0.005)
        with self.assertRaisesRegex(ValueError, "uncertainty"):
            verify(replace(self.program, job_fingerprint=uncertain.fingerprint),
                   uncertain, measure_rest=False)

    def test_continuous_island_crossing_and_low_rapid_rejected(self):
        cut_index = next(n for n, i in enumerate(self.program.items)
                         if isinstance(i, Move) and i.role == "cut" and
                         i.tool == "T1" and i.end[2] == -1 and i.end[1] == 15)
        cut = self.program.items[cut_index]
        # End disks are tangent to the island; the interior of this segment
        # crosses it.  Verification must evaluate the continuous sweep.
        self.assertFalse(_allowed_cut(Job(), (13, 15), (27, 15), 3))
        crossing = replace(cut, start=(13, 15, -1), end=(27, 15, -1))
        items = list(self.program.items)
        items[cut_index - 3] = replace(items[cut_index - 3], end=(13, 15, 5))
        items[cut_index - 2] = replace(items[cut_index - 2],
                                       start=(13, 15, 5), end=(13, 15, 1))
        items[cut_index - 1] = replace(items[cut_index - 1],
                                       start=(13, 15, 1), end=(13, 15, -1))
        items[cut_index] = crossing
        with self.assertRaisesRegex(ValueError, "protected target"):
            verify(replace(self.program, items=tuple(items)), measure_rest=False)
        items = list(self.program.items)
        items.insert(cut_index + 1, Move("rapid", "T1", cut.end,
                                         (27, 15, -1), 0, "rough"))
        with self.assertRaisesRegex(ValueError, "rapid below clearance"):
            verify(replace(self.program, items=tuple(items)), measure_rest=False)

    def test_component_reach_and_floor_rejected(self):
        for changes in ({"cutting_length": 2}, {"holder_start": 2}):
            tool = replace(Job().tools[1], **changes)
            job = replace(Job(), tools=(Job().tools[0], tool))
            with self.subTest(changes=changes), self.assertRaisesRegex(
                    ValueError, "non-cutting component stock collision"):
                verify(replace(self.program, job_fingerprint=job.fingerprint),
                       job, measure_rest=False)
        cut_index = next(n for n, i in enumerate(self.program.items)
                         if isinstance(i, Move) and i.role == "cut")
        cut = self.program.items[cut_index]
        with self.assertRaises(ValueError):
            verify(self.altered(cut_index, end=(cut.end[0], cut.end[1], -4)),
                   measure_rest=False)

    def test_missing_rough_predecessor_rejected(self):
        items = tuple(i for i in self.program.items if not (
            isinstance(i, Move) and i.tool == "T1" and i.role in ("entry", "cut")))
        with self.assertRaises(ValueError):
            verify(replace(self.program, items=items), measure_rest=False)

    def test_bottom_layer_cannot_borrow_upper_layer_coverage(self):
        items = list(self.program.items)
        for n, item in enumerate(items):
            if (isinstance(item, Move) and item.role == "cut" and
                    item.tool == "T1" and item.end[2] == -3 and
                    item.start[0] == 3 and 10 <= item.end[1] <= 15 and
                    item.end[0] > 4):
                # Keep all row spacings and full upper-layer coverage, but
                # shorten bottom-layer left passes near the island.
                y = item.end[1]
                items[n] = replace(item, end=(4, y, -3))
                items[n + 1] = replace(items[n + 1], start=(4, y, -3),
                                       end=(4, y, 5))
                items[n + 2] = replace(items[n + 2], start=(4, y, 5))
        with self.assertRaisesRegex(ValueError, "residual at Z=-3"):
            verify(replace(self.program, items=tuple(items)))


if __name__ == "__main__":
    unittest.main()
