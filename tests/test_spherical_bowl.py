"""Analytic curved ball finish/rest through independently decoded output."""

from dataclasses import replace
import math
import tempfile
from pathlib import Path
import unittest

from cambam_builder.cam_core.occupancy import (
    Box, OccupancySetup, ToolBand, ToolBody,
)
from cambam_builder.cam_core.ordered_job import Job, JobMove, Stage, Transition
from cambam_builder.cam_core.surface3d import (
    SphericalBowlTarget, SurfaceOperation, bowl_contact_tip_z,
    compare_representations,
)
from cambam_builder.integrations.ordered_output import audit_bundle, emit, write_bundle


def synthetic_job(*, depth_mm=0.8, small_radius_mm=0.25, boundary="split"):
    target = SphericalBowlTarget((-2.5, -2.5, 2.5, 2.5), 2,
                                  (0, 0), 2, depth_mm)
    safe = (0, 0, 3)
    large = 0.8
    allowance = 0.001

    def tip(radius, x, y):
        return (x, y, bowl_contact_tip_z(target, radius, x, y,
                                          clearance_mm=allowance))

    first = []
    at = safe

    def add(moves, role, end, feed=0):
        nonlocal at
        moves.append(JobMove(role, at, end, feed))
        at = end

    # A center entry witnesses the smaller tool's dependent descent. The
    # first finish has three sampled chords within the large ball's rim limit.
    add(first, "entry", tip(large, 0, 0), 60)
    for x in (i * 1.15 / 12 for i in range(1, 13)):
        add(first, "cut", tip(large, x, 0), 240)
    for x in (1.15 - i * 2.3 / 24 for i in range(1, 25)):
        add(first, "cut", tip(large, x, 0), 240)
    add(first, "retract", (at[0], at[1], 3), 120)
    for y in (-0.7, 0.7):
        extent = math.sqrt(1.18 ** 2 - y ** 2)
        add(first, "rapid", (-extent, y, 3))
        add(first, "entry", tip(large, -extent, y), 60)
        for i in range(1, 17):
            x = -extent + i * 2 * extent / 16
            add(first, "cut", tip(large, x, y), 240)
        add(first, "retract", (at[0], at[1], 3), 120)
    add(first, "rapid", safe)

    second = []
    at = safe
    add(second, "cleared_descent", tip(small_radius_mm, 0, 0), 60)
    # Three-turn radial rest path samples exact offset contact. The disk of
    # admissible centers is convex, so each chord stays inside the rim guard.
    for i in range(1, 73):
        t = i / 72
        radial = 1.65 * t
        angle = 6 * math.pi * t
        add(second, "cut", tip(small_radius_mm,
                                radial * math.cos(angle),
                                radial * math.sin(angle)), 240)
    add(second, "retract", (at[0], at[1], 3), 120)
    add(second, "rapid", safe)

    setup = OccupancySetup(
        "program", Box("stock", (-2.5, -2.5, -2, 2.5, 2.5, 0)),
        (Box("side-clamp", (3.25, -0.5, 0.5, 3.75, 0.5, 1.0)),),
        tuple(ToolBody(name, (
            ToolBand("cutter", 0, 2, radius),
            ToolBand("shank", 2, 2.5, radius),
            ToolBand("holder", 2.5, 3.5, holder),
        )) for name, radius, holder in
              (("T1", large, 0.9), ("T2", small_radius_mm, 0.5))))
    revision = f"bowl-R2-h{depth_mm}-r{small_radius_mm}"
    stages = (
        Stage("finish", "T1", tuple(first), 10000,
              operation=SurfaceOperation("finish", "T1", large, 2,
                                         target, "surface_follow"),
              source_revision=revision),
        Stage("rest", "T2", tuple(second), 10000,
              operation=SurfaceOperation("rest", "T2", small_radius_mm, 2,
                                         target, "rest"),
              source_revision=revision,
              transition=Transition("operator", boundary, "T2", safe,
                                    "verified-stage-boundary")),
    )
    return Job(revision, stages, safe, occupancy_setup=setup)


class SphericalBowlTests(unittest.TestCase):
    def test_independent_bowl_contact_section_and_volume_references(self):
        target = synthetic_job().stages[0].surface_operation.target
        radius, depth = 2, 0.8
        sphere_radius = (radius * radius + depth * depth) / (2 * depth)
        rim_plane = sphere_radius - depth
        self.assertAlmostEqual(target.target_volume_mm3,
                               math.pi * depth ** 2 * (sphere_radius - depth / 3))
        self.assertAlmostEqual(target.section_area_mm2(0), math.pi * radius ** 2)
        self.assertAlmostEqual(target.section_area_mm2(0.4),
                               math.pi * (sphere_radius ** 2 -
                                          (rim_plane + 0.4) ** 2))
        self.assertEqual(target.section_area_mm2(depth), 0)
        self.assertEqual(target.depth_at(radius, 0), 0)
        for ball_radius in (0.8, 0.25):
            rho = 0.8
            tip = bowl_contact_tip_z(target, ball_radius, rho, 0)
            # The ball center lies on the concentric sphere of radius Rs-r.
            self.assertAlmostEqual(math.hypot(rho, tip + ball_radius - rim_plane),
                                   sphere_radius - ball_radius)
            contact_rho = rho * sphere_radius / (sphere_radius - ball_radius)
            surface_z = rim_plane - math.sqrt(sphere_radius ** 2 -
                                              contact_rho ** 2)
            self.assertAlmostEqual(target.depth_at(contact_rho, 0), -surface_z)
            self.assertAlmostEqual(math.hypot(rho - contact_rho,
                                               tip + ball_radius - surface_z),
                                   ball_radius)
        comparison = compare_representations(target)
        self.assertEqual(comparison[0]["representation"], "exact_spherical_cap")
        for row in comparison[1:]:
            self.assertLessEqual(row["volume_interval_mm3"][0],
                                 target.target_volume_mm3)
            self.assertGreaterEqual(row["volume_interval_mm3"][1],
                                    target.target_volume_mm3)
        self.assertLess(comparison[2]["volume_interval_mm3"][1] -
                        comparison[2]["volume_interval_mm3"][0],
                        comparison[1]["volume_interval_mm3"][1] -
                        comparison[1]["volume_interval_mm3"][0])

    def test_decoded_finish_rest_rim_and_body(self):
        for dialect, boundary in (("uccnc", "split"), ("grbl", "pause")):
            with self.subTest(dialect=dialect):
                job = synthetic_job(boundary=boundary)
                files, report = emit(job, dialect)
                self.assertEqual(len(files), 2 if dialect == "uccnc" else 1)
                self.assertEqual(report["motion_equivalence"]["status"], "pass")
                stock = report["stock_access_residual"]
                self.assertEqual(stock["model"], "spherical-bowl-ball-v1")
                first, final = stock["prefixes"]
                self.assertGreater(final["newly_removed_volume_lower_mm3"], 0)
                self.assertLess(final["residual_volume_mm3"][1],
                                stock["target_volume_mm3"])
                self.assertEqual(final["protected_overcut_upper_mm3"], 0)
                self.assertGreater(final["minimum_holder_clearance_mm"], 0)
                self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")
                self.assertGreater(report["tool_fixture_occupancy"]
                                   ["minimum_fixture_clearance_mm"], 0)

    def test_protected_rim_contact_prior_stock_and_stale_curvature(self):
        job = synthetic_job()
        target = job.stages[0].surface_operation.target
        with self.assertRaisesRegex(ValueError, "protected rim"):
            bowl_contact_tip_z(target, 0.8, 1.3, 0)
        only_rest = replace(job, stages=(replace(job.stages[1], transition=None),),
                            occupancy_setup=None)
        with self.assertRaisesRegex(ValueError, "uncleared descent"):
            emit(only_rest, "uccnc")
        first = job.stages[0]
        moves = list(first.motions)
        low = moves[0].end
        unsafe = (low[0], low[1], low[2] - 0.02)
        moves[0] = replace(moves[0], end=unsafe)
        moves[1] = replace(moves[1], start=unsafe)
        with self.assertRaisesRegex(ValueError, "protected bowl"):
            emit(replace(job, stages=(replace(first, motions=tuple(moves)),
                                      job.stages[1])), "uccnc")
        varied = synthetic_job(depth_mm=0.7)
        self.assertNotEqual(job.fingerprint, varied.fingerprint)
        self.assertEqual(emit(varied, "uccnc")[1]
                         ["stock_access_residual"]["status"], "pass")
        changed_target = replace(target, depth_mm=0.7)
        changed_only_curvature = replace(job, stages=tuple(
            replace(stage, operation=replace(stage.operation,
                                             target=changed_target))
            for stage in job.stages))
        self.assertEqual(job.source_fingerprint,
                         changed_only_curvature.source_fingerprint)
        self.assertNotEqual(job.prefixes, changed_only_curvature.prefixes)
        Path("output").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="packet3-bowl-test-",
                                         dir="output") as temp:
            bundle = Path(temp) / "bowl-output"
            report = write_bundle(bundle, job, "uccnc")
            self.assertEqual(audit_bundle(bundle / "handoff.json", job), report)
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_bundle(bundle / "handoff.json", varied)
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_bundle(bundle / "handoff.json", changed_only_curvature)
            program = bundle / "stage-2.nc"
            program.write_bytes(program.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                audit_bundle(bundle / "handoff.json", job)


if __name__ == "__main__":
    unittest.main()
