"""Synthetic holder and fixture occupancy on decoded ordered 3D output."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from cambam_builder.cam_core import ordered_job, replay, v_region
from cambam_builder.cam_core.ordered_job import JobMove
from cambam_builder.cam_core.occupancy import (
    Box, OccupancySetup, ToolBand, ToolBody, verify,
)
from cambam_builder.integrations.ordered_output import audit_bundle, emit, write_bundle
from tests.test_surface3d import synthetic_job
from tests.test_volume3d import synthetic_job as volume_job


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


def _planar_job(kind):
    """One supplied protected removal segment, independent of physical stock."""
    initial, high, low = (-1, -1, 3), (3, 4, 3), (3, 4, -.5)
    end, retract = (7, 4, -.5), (7, 4, 3)
    motions = (JobMove("rapid", initial, high),
               JobMove("entry", high, low, 60),
               JobMove("cut", low, end, 300),
               JobMove("retract", end, retract, 120),
               JobMove("rapid", retract, initial))
    source, shell = "physical-stock-enclosure", ((0, 0), (10, 0), (10, 8), (0, 8))
    if kind == "v":
        target = v_region.VTarget.polygon(source, shell, (), 1,
            design_angle_degrees=90, frame="program")
        tool = v_region.VProfile("pointed", 90, 0, 2, 2)
        paths = (v_region.VPath("fill", ((3, 4, .5), (7, 4, .5))),)
        plan = v_region.VPlan(target, tool, paths, v_region._motions(paths, 3),
                              3, .02, 1, "partial", "stock enclosure witness")
        stage = ordered_job.Stage("supplied", "T1", motions, 10000,
            v_plan=plan, source_revision=plan.fingerprint)
        radius = tool.radius(tool.cutting_length)
    else:
        target = replay.Target("opening", (0, 0, 10, 8), 1, region_shell=shell)
        operation = replay.Operation("supplied", replay.ToolProfile("T1", "cylinder", .5, 2), target)
        stage = ordered_job.Stage("supplied", "T1", motions, 10000,
            operation=operation, source_revision=source)
        radius = .5
    setup = OccupancySetup("program", Box("physical stock", (0, 0, -1, 10, 8, 0)), (),
        (ToolBody("T1", (ToolBand("cutter", 0, 2, radius),
                         ToolBand("shank", 2, 3, radius),
                         ToolBand("holder", 3, 5, 3))),))
    return ordered_job.Job(source, (stage,), initial, occupancy_setup=setup)


class OccupancyTests(unittest.TestCase):
    def test_planar_physical_stock_can_enclose_removal_without_changing_stock_credit(self):
        for kind in ("v", "cylinder"):
            for dialect in ("uccnc", "grbl"):
                with self.subTest(kind=kind, dialect=dialect):
                    job = _planar_job(kind)
                    _, original = emit(job, dialect)
                    full_stock = replace(job, occupancy_setup=replace(
                        job.occupancy_setup,
                        stock=Box("larger physical stock", (-2, -3, -4, 12, 11, 0))))
                    _, enlarged = emit(full_stock, dialect)
                    self.assertNotEqual(job.fingerprint, full_stock.fingerprint)
                    self.assertEqual(enlarged["tool_fixture_occupancy"]["status"], "pass")
                    self.assertEqual(enlarged["stock_access_residual"], original["stock_access_residual"])

    def test_planar_physical_stock_cannot_crop_or_shift_away_from_target(self):
        invalid = ((.01, 0, -2, 10, 8, 0), (0, .01, -2, 10, 8, 0),
                   (0, 0, -2, 9.99, 8, 0), (0, 0, -2, 10, 7.99, 0),
                   (0, 0, -.99, 10, 8, 0), (1, 0, -2, 11, 8, 0))
        for kind in ("v", "cylinder"):
            job = _planar_job(kind)
            for bounds in invalid:
                with self.subTest(kind=kind, bounds=bounds), self.assertRaisesRegex(
                        ValueError, "occupancy stock differs.*enclose planar stage target"):
                    emit(replace(job, occupancy_setup=replace(job.occupancy_setup,
                        stock=Box("cropped or shifted physical stock", bounds))), "uccnc")

    def test_enclosing_planar_stock_does_not_allow_cutting_outside_removal_target(self):
        for kind in ("v", "cylinder"):
            job = _planar_job(kind)
            stage = job.stages[0]
            moves = list(stage.motions)
            moves[2] = replace(moves[2], end=(11, 4, -.5))
            moves[3] = replace(moves[3], start=moves[2].end, end=(11, 4, 3))
            moves[4] = replace(moves[4], start=moves[3].end)
            changes = {"motions": tuple(moves)}
            if kind == "v":
                paths = (v_region.VPath("fill", ((3, 4, .5), (11, 4, .5))),)
                plan = replace(stage.v_plan, paths=paths, motions=v_region._motions(paths, 3))
                changes.update(v_plan=plan, source_revision=plan.fingerprint)
            full_stock = replace(job, stages=(replace(stage, **changes),),
                occupancy_setup=replace(job.occupancy_setup,
                    stock=Box("larger physical stock", (-2, -3, -4, 14, 11, 0))))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "crosses"):
                emit(full_stock, "uccnc")

    def test_noncutting_bands_use_entire_physical_stock_box(self):
        # Isolate the occupancy owner's geometry: a vertical sweep outside the
        # selected target's box must still see physical stock there. This is
        # not a claim that the separate cutting/access gate permits the sweep.
        body = ToolBody("T1", (ToolBand("cutter", 0, 1, .2),
                               ToolBand("shank", 1, 1.5, .2),
                               ToolBand("holder", 1.5, 3, .5)))
        stage = SimpleNamespace(tool_id="T1")
        move = JobMove("rapid_retract", (-2, -2, -2), (-2, -2, 3))
        target_box = OccupancySetup("program", Box("removal bounds", (0, 0, -1, 10, 8, 0)),
                                    (), (body,))
        self.assertEqual(verify(target_box, (stage,), ((move,),))["status"], "pass")
        physical = replace(target_box,
                           stock=Box("full physical stock", (-4, -4, -3, 12, 11, 0)))
        with self.assertRaisesRegex(ValueError, "shank collision with stock"):
            verify(physical, (stage,), ((move,),))

    def test_surface_and_volume_stock_still_require_exact_target_stock(self):
        surface = with_setup(synthetic_job())
        volume = volume_job()
        target = volume.stages[0].volume_operation.target
        x0, y0, x1, y1 = target.stock_xy
        bodies = tuple(ToolBody(stage.tool_id, (
            ToolBand("cutter", 0, 2.5, stage.volume_operation.radius_mm),
            ToolBand("shank", 2.5, 3.5, .5),
            ToolBand("holder", 3.5, 5.5, .8))) for stage in volume.stages)
        volume = replace(volume, occupancy_setup=OccupancySetup("program",
            Box("volume stock", (x0, y0, -target.stock_depth, x1, y1, 0)), (), bodies))
        for job in (surface, volume):
            emit(job, "uccnc")
            bounds = job.occupancy_setup.stock.bounds
            larger = (bounds[0] - 1, bounds[1] - 1, bounds[2] - 1,
                      bounds[3] + 1, bounds[4] + 1, 0)
            with self.subTest(operation=type(job.stages[0].operation).__name__), self.assertRaisesRegex(
                    ValueError, "occupancy stock differs from stage target"):
                emit(replace(job, occupancy_setup=replace(job.occupancy_setup,
                    stock=Box("different physical stock", larger))), "uccnc")

    def test_straight_holder_contact_and_numeric_admission(self):
        body = ToolBody("T1", (
            ToolBand("cutter", 0, 1, 0.2),
            ToolBand("shank", 1, 1.5, 0.2),
            ToolBand("holder", 1.5, 3, 0.5)))
        stage = SimpleNamespace(tool_id="T1")
        move = JobMove("cut", (0, 0, 0), (10, 0, 0), 100)
        stock = Box("stock", (-1, -1, -2, 11, 1, 0))
        for gap, collision in ((0.0, True), (0.5e-9, True),
                               (2e-9, False)):
            with self.subTest(gap=gap):
                setup = OccupancySetup("program", stock,
                    (Box("clamp", (4, 0.5 + gap, 1.6,
                                    6, 1.0 + gap, 2.5)),), (body,))
                if collision:
                    with self.assertRaisesRegex(ValueError,
                                                "holder collision with fixture clamp"):
                        verify(setup, (stage,), ((move,),))
                else:
                    result = verify(setup, (stage,), ((move,),))
                    self.assertAlmostEqual(result["minimum_fixture_clearance_mm"],
                                           gap, delta=1e-15)

        setup = OccupancySetup("program", stock, (), (body,))
        extreme = JobMove("cut", (-1e308, 0, -1e200),
                          (1e308, 0, 1e200), 100)
        with self.assertRaisesRegex(ValueError, "arithmetic is unresolved"):
            verify(setup, (stage,), ((extreme,),))

    def test_planar_arc_holder_hits_between_clear_endpoints(self):
        body = ToolBody("T1", (
            ToolBand("cutter", 0, 3, 0.5),
            ToolBand("shank", 3, 3.5, 0.5),
            ToolBand("holder", 3.5, 6, 0.8)))
        stage = SimpleNamespace(tool_id="T1")
        for direction, side in ((3, 1), (2, -1)):
            with self.subTest(direction=direction):
                move = JobMove("cut", (3, 0, -2), (-3, 0, -2), 100,
                               direction, (0, 0))
                safe_y = (4.1, 4.4) if side > 0 else (-4.4, -4.1)
                collision_y = (3.7, 4.0) if side > 0 else (-4.0, -3.7)
                safe = OccupancySetup("program", Box("stock", (-5, -5, -2, 5, 5, 0)),
                    (Box("clamp", (-0.1, safe_y[0], 1.6, 0.1,
                                   safe_y[1], 2.5)),),
                    (body,))
                result = verify(safe, (stage,), ((move,),))
                self.assertEqual(result["checked_arcs"], 1)
                self.assertGreater(result["minimum_fixture_clearance_mm"], 0.2)
                clamp = safe.fixtures[0]
                collided = replace(safe, fixtures=(Box("clamp", (
                    clamp.bounds[0], collision_y[0],
                    clamp.bounds[2], clamp.bounds[3],
                    collision_y[1], clamp.bounds[5])),))
                for x in (3, -3):
                    self.assertGreater(math.hypot(abs(x) - 0.1, 3.7), 0.8)
                with self.assertRaisesRegex(ValueError,
                                            "holder collision with fixture clamp"):
                    verify(collided, (stage,), ((move,),))

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
