"""Synthetic holder and fixture occupancy on decoded ordered 3D output."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

from cambam_builder.cam_core.occupancy import (
    Box, OccupancySetup, ToolBand, ToolBody,
)
from cambam_builder.integrations.ordered_output import audit_bundle, emit, write_bundle
from tests.test_surface3d import synthetic_job


def with_setup(job, *, fixture_y=-0.9, holder_radius=0.8):
    target = job.stages[0].surface_operation.target
    x0, y0, x1, y1 = target.stock_xy
    setup = OccupancySetup(
        job.program_frame,
        Box("initial stock", (x0, y0, -target.stock_depth_mm, x1, y1, 0)),
        (Box("side clamp", (1.8, fixture_y - 0.3, 2, 2.2,
                             fixture_y, 2.5)),),
        (ToolBody("T1", (
            ToolBand("cutter", 0, 2.5, 0.5),
            ToolBand("shank", 2.5, 3.5, 0.5),
            ToolBand("holder", 3.5, 5.5, holder_radius),
        )),))
    return replace(job, occupancy_setup=setup)


class OccupancyTests(unittest.TestCase):
    def test_decoded_continuous_holder_and_fixture_clearance(self):
        for dialect, boundary in (("uccnc", "split"), ("grbl", "pause")):
            with self.subTest(dialect=dialect):
                job = with_setup(synthetic_job(boundary))
                files, report = emit(job, dialect)
                self.assertEqual(len(files), 2 if dialect == "uccnc" else 1)
                self.assertEqual(report["motion_equivalence"]["status"], "pass")
                self.assertEqual(report["stock_access_residual"]["status"], "pass")
                occupancy = report["tool_fixture_occupancy"]
                self.assertEqual(occupancy["status"], "pass")
                self.assertEqual(occupancy["model"], "bounded-tool-occupancy-v1")
                self.assertEqual(occupancy["checked_moves"], 12)
                self.assertGreater(occupancy["minimum_fixture_clearance_mm"], 0)
                self.assertLess(occupancy["minimum_fixture_clearance_mm"], 0.2)

    def test_holder_hits_fixture_between_clear_endpoints(self):
        job = synthetic_job()
        safe = with_setup(job)
        collision = with_setup(job, fixture_y=-0.55)
        self.assertNotEqual(safe.fingerprint, collision.fingerprint)
        self.assertEqual(safe.stages, collision.stages)
        clamp = collision.occupancy_setup.fixtures[0].bounds
        self.assertGreater(-clamp[4], 0.5)  # cutter clears the near Y face
        self.assertGreater(math.hypot(1 - clamp[0], -clamp[4]), 0.8)
        self.assertGreater(math.hypot(3 - clamp[3], -clamp[4]), 0.8)
        with self.assertRaisesRegex(ValueError,
                                    "holder collision with fixture side clamp"):
            emit(collision, "uccnc")
        # The same cutter and decoded path clear the clamp when the holder
        # radius is narrowed. The collision is from the modeled holder.
        _, report = emit(with_setup(job, fixture_y=-0.55,
                                    holder_radius=0.4), "uccnc")
        self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")

    def test_nonidentity_work_translation_uses_program_frame_setup(self):
        job = with_setup(replace(synthetic_job(),
                                 translation_xyz_mm=(10, -7, 1)))
        _, report = emit(job, "uccnc")
        self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")
        self.assertEqual(report["stock_access_residual"]["status"], "pass")

    def test_changed_setup_invalidates_written_evidence(self):
        job = with_setup(synthetic_job())
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "holder-case"
            report = write_bundle(bundle, job, "uccnc")
            self.assertEqual(audit_bundle(bundle / "handoff.json", job), report)
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_bundle(bundle / "handoff.json",
                             with_setup(synthetic_job(), fixture_y=-0.55))

    def test_setup_must_cover_operation_and_frame(self):
        job = with_setup(synthetic_job())
        with self.assertRaisesRegex(ValueError, "matching frame"):
            replace(job, program_frame="edited")
        with self.assertRaisesRegex(ValueError, "occupancy cutter differs"):
            bad = ToolBody("T1", (
                ToolBand("cutter", 0, 2.5, 0.4),
                ToolBand("shank", 2.5, 3.5, 0.5),
                ToolBand("holder", 3.5, 5.5, 0.8)))
            emit(replace(job, occupancy_setup=replace(
                job.occupancy_setup, tools=(bad,))), "uccnc")
        with self.assertRaisesRegex(ValueError, "stock differs"):
            emit(replace(job, occupancy_setup=replace(
                job.occupancy_setup,
                stock=Box("initial stock", (0, 0, -2, 4, 2, 0)))), "uccnc")


if __name__ == "__main__":
    unittest.main()
