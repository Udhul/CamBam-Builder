"""Sloped surface and ball cutter evidence through decoded ordered output."""

from dataclasses import replace
import math
import tempfile
from pathlib import Path
import unittest

from cambam_builder.cam_core.ordered_job import Job, JobMove, Stage, Transition
from cambam_builder.cam_core.surface3d import (
    SlopedTarget, SurfaceOperation, compare_representations, contact_tip_z,
    contact_x, straight_pass_volume_mm3,
)
from cambam_builder.integrations.ordered_output import audit_bundle, emit, write_bundle


def synthetic_job(boundary="split"):
    target = SlopedTarget((0, 0, 4, 2), 3, 1, 0.25)
    radius = 0.5
    safe = (0, 0, 3)
    at = (1, 0, 3)
    tip1 = contact_tip_z(target, radius, 1, clearance_mm=0.001)
    tip3 = contact_tip_z(target, radius, 3, clearance_mm=0.001)
    low1 = (1, 0, tip1)
    low3 = (3, 0, tip3)
    first = (
        JobMove("rapid", safe, at),
        JobMove("entry", at, low1, 60),
        JobMove("cut", low1, (1, 2, tip1), 300),
        JobMove("retract", (1, 2, tip1), (1, 2, 3), 120),
        JobMove("rapid", (1, 2, 3), safe),
    )
    second = (
        JobMove("rapid", safe, at),
        JobMove("cleared_descent", at, low1, 60),
        JobMove("cut", low1, low3, 300),
        JobMove("cut", low3, (3, 2, tip3), 300),
        JobMove("cut", (3, 2, tip3), (1, 2, tip1), 300),
        JobMove("retract", (1, 2, tip1), (1, 2, 3), 120),
        JobMove("rapid", (1, 2, 3), safe),
    )
    stages = (
        Stage("follow", "T1", first, 10000,
              operation=SurfaceOperation("follow", "T1", radius, 2.5,
                                         target, "surface_follow"),
              source_revision="slope-v1"),
        Stage("dependent", "T1", second, 10000,
              operation=SurfaceOperation("dependent", "T1", radius, 2.5,
                                         target, "rest"),
              source_revision="slope-v1",
              transition=Transition("operator", boundary, "T1", safe,
                                    "verified-stage-boundary")),
    )
    return Job("slope-v1", stages, safe)


class Surface3DTests(unittest.TestCase):
    def test_analytic_contact_target_and_straight_pass(self):
        job = synthetic_job()
        target = job.stages[0].surface_operation.target
        self.assertEqual(target.target_volume_mm3, 12)
        self.assertEqual(target.section_area_mm2(0.5), 8)
        self.assertEqual(target.section_area_mm2(1.5), 4)
        contact = contact_x(target, 0.5, 1)
        tip = job.stages[0].motions[1].end[2]
        center_z = tip + 0.5
        distance = math.hypot(contact - 1, target.depth_at(contact) + center_z)
        self.assertAlmostEqual(distance, 0.5 + 0.001 / math.sqrt(1 + 0.25 ** 2),
                               places=6)
        exact_first = straight_pass_volume_mm3(target, 0.5, 1, tip)
        self.assertAlmostEqual(exact_first, 2.252621756993033)
        comparison = compare_representations(target)
        self.assertEqual(comparison[0]["volume_interval_mm3"], (12, 12))
        for row in comparison[1:]:
            self.assertLessEqual(row["volume_interval_mm3"][0], 12)
            self.assertGreaterEqual(row["volume_interval_mm3"][1], 12)
        self.assertLess(
            comparison[2]["volume_interval_mm3"][1] -
            comparison[2]["volume_interval_mm3"][0],
            comparison[1]["volume_interval_mm3"][1] -
            comparison[1]["volume_interval_mm3"][0])

    def test_decoded_ball_stock_and_dependent_pass(self):
        job = synthetic_job()
        target = job.stages[0].surface_operation.target
        tip = job.stages[0].motions[1].end[2]
        exact_first = straight_pass_volume_mm3(target, 0.5, 1, tip)
        for dialect, boundary in (("uccnc", "split"), ("grbl", "pause")):
            with self.subTest(dialect=dialect):
                files, report = emit(synthetic_job(boundary), dialect)
                self.assertEqual(len(files), 2 if dialect == "uccnc" else 1)
                self.assertEqual(report["motion_equivalence"]["status"], "pass")
                stock = report["stock_access_residual"]
                self.assertEqual(stock["status"], "pass")
                self.assertEqual(stock["model"], "sloped-ball-v1")
                first, final = stock["prefixes"]
                self.assertLessEqual(first["residual_volume_mm3"][0],
                                     12 - exact_first)
                self.assertGreaterEqual(first["residual_volume_mm3"][1],
                                        12 - exact_first)
                self.assertGreater(first["residual_volume_mm3"][0],
                                   final["residual_volume_mm3"][1])
                self.assertEqual(final["protected_overcut_upper_mm3"], 0)
                self.assertGreater(final["minimum_holder_clearance_mm"], 0.5)
        only_rest = replace(job, stages=(replace(job.stages[1], transition=None),))
        with self.assertRaisesRegex(ValueError, "uncleared descent"):
            emit(only_rest, "uccnc")

    def test_protected_slope_cutting_length_and_stale_output(self):
        job = synthetic_job()
        stage = job.stages[0]
        motions = list(stage.motions)
        low = motions[1].end
        bad_low = (low[0], low[1], low[2] - 0.02)
        motions[1] = replace(motions[1], end=bad_low)
        motions[2] = replace(motions[2], start=bad_low)
        with self.assertRaisesRegex(ValueError, "protected slope"):
            emit(replace(job, stages=(replace(stage, motions=tuple(motions)),
                                      job.stages[1])), "uccnc")
        short = replace(stage.surface_operation, cutting_length_mm=1.0)
        with self.assertRaisesRegex(ValueError, "cutting length"):
            emit(replace(job, stages=(replace(stage, operation=short),
                                      job.stages[1])), "uccnc")
        _, report = emit(replace(job, stock_present=False), "uccnc")
        self.assertEqual(report["stock_access_residual"]["status"],
                         "not_evaluated")
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "surface-output"
            report = write_bundle(bundle, job, "uccnc")
            self.assertEqual(audit_bundle(bundle / "handoff.json", job), report)
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_bundle(bundle / "handoff.json",
                             replace(job, source_fingerprint="edited"))
            program = bundle / "stage-2.nc"
            program.write_bytes(program.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                audit_bundle(bundle / "handoff.json", job)


if __name__ == "__main__":
    unittest.main()
