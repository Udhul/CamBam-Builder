"""Actual CamBam Pocket helix and dependent, decoded circular cleanup."""

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from cambam_builder.cam_core import ordered_job, polygon_rest, replay
from cambam_builder.integrations.cambam.native_ordered_job import (
    NativeBinding, from_native_circle_cleanup,
)
from cambam_builder.integrations.cambam.native_series import (
    PostedMove, normalize_native_series,
)
from cambam_builder.integrations.cambam.native_series_audit import circle_target
from cambam_builder.integrations.ordered_dialects import decode
from cambam_builder.integrations.ordered_output import audit_bundle, write_bundle


FIXTURE = Path(__file__).parent / "fixtures" / "helical_pocket"
SOURCE = FIXTURE / "packet2-helical-pocket.cb"
POST = FIXTURE / "packet2-helical-pocket.nc"
SOURCE_HASH = "f53d0221a0a841af9ccb0d2bc1af1607f8b5161535466a7cce7e0a59393f0e03"
POST_HASH = "af2208d1019af5684d082dd62ce5f148b1a63667f71c151a720cfbcb88fb0978"


class NativeHelixPocketTests(unittest.TestCase):
    def test_actual_post_and_decoded_cleanup(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_HASH)
        self.assertEqual(hashlib.sha256(POST.read_bytes()).hexdigest(), POST_HASH)
        series = normalize_native_series(SOURCE, SOURCE, POST,
                                         initial_position=(-5, -5, 5))
        helices = [move for move in series.items
                   if type(move) is PostedMove and move.g == 2 and
                   move.end[2] < move.start[2]]
        self.assertEqual(len(helices), 10)
        self.assertEqual(series.stages[0].move_count, 67)
        target = circle_target(SOURCE, "pocket-circle", 2)
        job = from_native_circle_cleanup(series, target)
        binding = NativeBinding(series, SOURCE, SOURCE, POST)
        self.assertTrue(binding.check(job))
        self.assertEqual(tuple(len(stage.motions) for stage in job.stages), (67, 9))

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "uccnc"
            report = write_bundle(output, job, "uccnc", source_binding=binding)
            self.assertEqual(report["status"], "ordered_output_pass")
            self.assertEqual(report["stock_access_residual"]["status"], "pass")
            self.assertEqual(report["motion_equivalence"]["status"], "pass")
            self.assertEqual(report["stock_access_residual"]["cuts_by_prefix"],
                             (4004, 4745))
            self.assertEqual(audit_bundle(output / "handoff.json", job,
                                          source_binding=binding), report)
            files = tuple((output / f"stage-{i}.nc").read_bytes()
                          for i in (1, 2))
            decoded = decode(files, "uccnc", initial_work_tip=job.initial_tip)
            observed = tuple(ordered_job._observed_stage(stage, stage_read,
                                                         job.translation_xyz_mm)
                             for stage, stage_read in zip(job.stages,
                                                          decoded.stages))
            stocks = ordered_job._replay_endmills(job, job.stages, observed)
            circle_area = math.pi * 12 ** 2
            polygon_error = circle_area - polygon_rest._geometry(target).area
            self.assertGreaterEqual(polygon_error, 0)
            self.assertLess(polygon_error, 0.05)
            ideal_native_ring = math.pi * (12 ** 2 - 11 ** 2)
            for depth in (1, 2):
                with self.subTest(depth=depth):
                    prior = polygon_rest._areas(target, stocks[0][1].cuts, depth)
                    final = polygon_rest._areas(target, stocks[1][1].cuts, depth)
                    self.assertLess(abs(prior[0] - ideal_native_ring), 0.1)
                    self.assertLess(prior[1] - prior[0], 0.05)
                    self.assertLess(final[1], 1.0)
                    self.assertGreater(prior[0] - final[1], 70)
                    self.assertLess(final[1] - final[0], 0.05)
                    protected = polygon_rest._cut_polygon(
                        stocks[1][1].cuts, depth, radial_error=1e-6
                    ).difference(polygon_rest._geometry(target)).area
                    self.assertLess(protected, 0.01)

            changed_moves = list(job.stages[1].motions)
            changed_moves[2] = replace(changed_moves[2], feed=241)
            tampered = replace(job, stages=(job.stages[0], replace(
                job.stages[1], motions=tuple(changed_moves))))
            with self.assertRaisesRegex(ValueError, "generated circle cleanup differs"):
                binding.check(tampered)
            first = output / "stage-1.nc"
            original = first.read_bytes()
            first.write_bytes(original.replace(b"I0.766", b"I0.7661", 1))
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                audit_bundle(output / "handoff.json", job,
                             source_binding=binding)

    def test_helix_section_clip_and_unsupported_forms(self):
        target = replay.Target("square", (0, 0, 10, 10), 1,
                               region_shell=((0, 0), (10, 0),
                                             (10, 10), (0, 10)))
        tool = replay.ToolProfile("T1", "cylinder", 0.2, 2)
        operation = replay.Operation("cut", tool, target)
        start = (5, 5, 0)
        end = (6, 5, -1)
        at = (5, 5, 5)
        trace = replay.Trace("helix-section", "XY", at, (operation,), (
            replay.Event("tool_change", "T1", at),
            replay.Event("spindle_start", "T1", at),
            replay.Motion("approach", "T1", "cut", at, start, 60),
            replay.ArcMotion("cut", "T1", "cut", start, end,
                             240, 2, (5.5, 5)),
            replay.Motion("retract", "T1", "cut", end, (6, 5, 5), 60),
            replay.Event("spindle_stop", "T1", (6, 5, 5)),
        ))
        stock = replay.replay(trace, expected_source="helix-section")
        self.assertTrue(stock.removed_contains(6, 5, 0.75))
        self.assertFalse(stock.removed_contains(5, 5, 0.75))
        self.assertGreater(polygon_rest._areas(target, stock.cuts, 0.75)[0], 99)
        outside = replace(trace, items=trace.items[:3] + (
            replay.ArcMotion("cut", "T1", "cut", start, end,
                             240, 2, (5.5, 9.5)),
        ) + trace.items[4:])
        with self.assertRaisesRegex(ValueError, "boundary"):
            replay.replay(outside, expected_source="helix-section")
        with self.assertRaisesRegex(ValueError, "unsupported planar arc geometry"):
            replay.arc_segments(start, end, (5.5, 5), 2)
        with self.assertRaisesRegex(ValueError, "unsupported planar cutting arc"):
            replay.ArcMotion("cut", "T1", "cut", end, start,
                             240, 3, (5.5, 5))
        with self.assertRaisesRegex(ValueError, "helical volume"):
            polygon_rest._volume(target, stock.cuts)

    def test_source_and_post_edits_invalidate_series(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / SOURCE.name
            post = Path(folder) / POST.name
            source.write_bytes(SOURCE.read_bytes())
            post.write_bytes(POST.read_bytes())
            series = normalize_native_series(source, source, post,
                                             initial_position=(-5, -5, 5))
            self.assertTrue(series.check_freshness(source, source, post))
            source.write_bytes(source.read_bytes().replace(b'd="24"', b'd="25"'))
            with self.assertRaisesRegex(ValueError, "changed"):
                series.check_freshness(source, source, post)
            source.write_bytes(SOURCE.read_bytes())
            post.write_bytes(post.read_bytes().replace(b"G17", b"G18", 1))
            with self.assertRaisesRegex(ValueError, "unsupported G/M command"):
                normalize_native_series(source, source, post,
                                        initial_position=(-5, -5, 5))


if __name__ == "__main__":
    unittest.main()
