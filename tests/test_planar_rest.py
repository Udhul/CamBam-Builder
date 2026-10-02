"""RP01 whole-height stock proofs, feature containment and decoded motion."""
from dataclasses import replace
import math
import re
import unittest

try:
    from cambam_builder.cam_core import ordered_job, planar_rest, v_region as v
    from cambam_builder.cam_core.occupancy import (
        Box, OccupancySetup, ToolBand, ToolBody,
    )
    from cambam_builder.integrations.ordered_output import audit_files, emit
    from tests.test_mixed_v_composition import _cylinder, _trace
    from tests.test_multistage_v import _for_dialect, _job, _supplied
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _rectangle():
    return v.VTarget.polygon("RP01-union-proof",
        ((0, 0), (16, 0), (16, 12), (0, 12)), (), 1.5,
        design_angle_degrees=90)


def _stock(target, segments, *, radius=1, depths=(.5, 1, 1.5)):
    return v.VComposition(target, tuple(_trace(target, _cylinder(
        target, f"prior-{i}", f"T{i+1}", radius, a, b, depths=depths))
        for i, (a, b) in enumerate(segments)))


def _island():
    target = v.VTarget.polygon("RP01-island-different-profile",
        ((0, 0), (12, 0), (12, 10), (0, 10)),
        (((5, 5), (7, 5), (7, 7), (5, 7)),), 1,
        design_angle_degrees=90)
    tool = v.VProfile("pointed", 120, 0, 4, 2)
    prior = _stock(target, (((2.5, 2.5), (4, 2.5)),),
                   radius=.4, depths=(.5, 1))
    return target, tool, prior


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class PlanarRestTests(unittest.TestCase):
    def test_union_clears_whole_capsule_when_neither_stage_does(self):
        target = _rectangle()
        tool = v.VProfile("pointed", 90, 0, 4, 3)
        segments = (((4, 5), (7.2, 5)), ((6.8, 5), (10, 5)))
        prior = _stock(target, segments)
        a, b = (4, 5, .8), (10, 5, .8)
        # Analytic witness: overlapping collinear unit-radius cylinders give
        # one length-six capsule, enclosing the candidate's radius <= .8 at
        # every height. Each isolated stage misses the opposite endpoint.
        self.assertLess(tool.radius(a[2]), 1)
        self.assertLess(segments[1][0][0], segments[0][1][0])
        self.assertTrue(planar_rest.cutting_sweep_clear(prior, tool, a, b))
        self.assertTrue(planar_rest.cutting_sweep_clear(
            prior, tool, (5, 5, .3), (9, 5, .8)))
        self.assertTrue(planar_rest.cutting_sweep_clear(
            prior, tool, (7, 5, .8), (7, 5, .8)))
        for trace in prior.stages:
            self.assertFalse(planar_rest.cutting_sweep_clear(
                v.VComposition(target, (trace,)), tool, a, b))

    def test_gap_and_stock_below_sampled_opening_are_not_clear(self):
        target = _rectangle()
        tool = v.VProfile("pointed", 90, 0, 4, 3)
        gap = _stock(target, (((4, 5), (6, 5)), ((8, 5), (10, 5))))
        self.assertFalse(planar_rest.cutting_sweep_clear(
            gap, tool, (4, 5, .8), (10, 5, .8)))
        # At x=7, the prior disks only touch at y=5; y=5.4 is uncut.
        self.assertGreater(math.hypot(7-6, .4), 1)
        shallow = _stock(target, (((4, 5), (10, 5)),),
                         radius=.6, depths=(.25,))
        # A Z=0-only check would accept the .5-radius candidate, but its tip
        # reaches .5 mm while the prior cylinder stopped at .25 mm.
        self.assertGreater(.6, tool.radius(.5))
        for slabs in (1, 4, 16):
            self.assertFalse(planar_rest.cutting_sweep_clear(
                shallow, tool, (4, 5, .5), (10, 5, .5), slabs=slabs))
        deep = _stock(target, (((4, 5), (10, 5)),), radius=.6)
        self.assertTrue(planar_rest.cutting_sweep_clear(
            deep, tool, (4, 5, .5), (10, 5, .5)))

    def test_same_profile_reversed_variable_depth_retrace(self):
        target = _rectangle()
        tool = v.VProfile("pointed", 90, 0, 4, 3)
        a, b = (4, 5, 1), (8, 5, 1.2)
        prior = v.VComposition(target, (_supplied(target, tool, a, b),))
        self.assertTrue(planar_rest.cutting_sweep_clear(prior, tool, b, a))
        self.assertTrue(planar_rest.cutting_sweep_clear(
            prior, tool, (8, 5, 1.1), (4, 5, .9)))
        self.assertFalse(planar_rest.cutting_sweep_clear(
            prior, tool, (8, 5, 1.3), a))
        different = v.VProfile("pointed", 120, 0, 6, 3)
        self.assertFalse(planar_rest.cutting_sweep_clear(prior, different, b, a))

    def test_clearance_rejects_stale_design_and_invalid_controls(self):
        target = _rectangle()
        tool = v.VProfile("pointed", 90, 0, 4, 3)
        prior = _stock(target, (((4, 5), (10, 5)),))
        with self.assertRaises(ValueError):
            v.VComposition(replace(target, source_id="edited-source"), prior.stages)
        changed = v.VTarget.polygon(target.source_id,
            tuple(target.safe.exterior.coords),
            (((6, 4), (8, 4), (8, 6), (6, 6)),), target.cap_depth,
            design_angle_degrees=90)
        with self.assertRaises(ValueError):
            v.VComposition(changed, prior.stages)
        for kwargs in ({"slabs": 0}, {"slabs": True}, {"slabs": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                planar_rest.cutting_sweep_clear(
                    prior, tool, (4, 5, .5), (10, 5, .5), **kwargs)
        for point in ((4, 5, 0), (4, 5, 2), (4, 5, float("nan")),
                      (4, float("inf"), .5), (4, 5)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                planar_rest.cutting_sweep_clear(prior, tool, point, (10, 5, .5))

    def test_island_candidates_keep_full_profile_and_complete_high_motion(self):
        target, tool, prior = _island()
        candidate = planar_rest.generate(prior, tool, stepover_mm=1,
            max_cusp_mm=.5, xy_step_mm=2)
        self.assertTrue(candidate.plan.paths)
        self.assertEqual(candidate.plan.target.fingerprint, target.fingerprint)
        self.assertIs(v.verify(candidate.plan), candidate.plan)
        boundary = target.safe.boundary
        from shapely.geometry import LineString
        for path in candidate.plan.paths:
            for a, b in zip(path.points, path.points[1:]):
                segment = LineString((a[:2], b[:2]))
                self.assertTrue(target.safe.covers(segment))
                self.assertGreaterEqual(segment.distance(boundary)+1e-8,
                    v.contact_radius(target, tool, max(a[2], b[2]))+.005)
        for motion in candidate.plan.motions:
            if motion.role == "rapid":
                self.assertEqual(motion.start[2], 3)
                self.assertEqual(motion.end[2], 3)
            elif motion.role == "entry":
                self.assertEqual(motion.start[:2], motion.end[:2])
                self.assertEqual(motion.start[2], 3)
                self.assertLess(motion.end[2], 0)
        length = sum(math.dist(a[:2], b[:2]) for path in candidate.plan.paths
                     for a, b in zip(path.points, path.points[1:]))
        self.assertAlmostEqual(candidate.proposed_length_mm,
            length+candidate.omitted_air_length_mm, places=8)
        for depth in (0, .5, 1):
            section = candidate.section(depth)
            for key in ("new_removal_mm2", "overlap_mm2"):
                low, high = section[key]
                self.assertGreaterEqual(low, 0)
                self.assertGreaterEqual(high, low)
            self.assertEqual(section["residual"].design_fingerprint, target.fingerprint)
        repeated = planar_rest.generate(candidate.composition, tool,
            stepover_mm=1, max_cusp_mm=.5, xy_step_mm=2)
        self.assertFalse(repeated.plan.paths)
        self.assertEqual(repeated.plan.status, "infeasible")
        self.assertAlmostEqual(repeated.proposed_length_mm,
                               repeated.omitted_air_length_mm, places=8)

    def test_generation_rejects_malformed_controls_and_exhausted_budgets(self):
        _, tool, prior = _island()
        cases = ({"stepover_mm": 0}, {"xy_step_mm": float("nan")},
            {"margin_mm": 1e-6}, {"safe_z": -1}, {"max_cusp_mm": 3},
            {"clearance_slabs": True}, {"max_paths": 0}, {"max_sites": 0},
            {"max_paths": 1}, {"max_sites": 4})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                planar_rest.generate(prior, tool, **kwargs)

    def test_depth_passes_round_trip_both_dialects_and_reject_low_rapid(self):
        target, tool, prior = _island()
        candidate = planar_rest.generate(prior, tool, stepover_mm=1,
            max_cusp_mm=.5, xy_step_mm=2)
        plans = v.depth_passes(candidate.plan, .5)
        job = _job(plans)
        rough = _cylinder(target, "prior-0", "T1", .4,
                          (2.5, 2.5), (4, 2.5), depths=(.5, 1))
        self.assertEqual(prior.stages, (_trace(target, rough),))
        stages = (rough, replace(job.stages[0],
            transition=ordered_job.Transition("operator", "split",
                job.stages[0].tool_id, job.initial_tip, "offline-install"))) + job.stages[1:]
        job = replace(job, stages=stages)
        xmin, ymin, xmax, ymax = candidate.plan.target.safe.bounds
        bodies = []
        for stage in job.stages:
            length = stage.operation.tool.cutting_length if stage.operation else tool.cutting_length
            radius = stage.operation.tool.radius if stage.operation else tool.radius(length)
            bodies.append(ToolBody(stage.tool_id, (
                ToolBand("cutter", 0, length, radius),
                ToolBand("shank", length, length+1, radius),
                ToolBand("holder", length+1, length+3, 4),
            )))
        setup = OccupancySetup("program", Box("original-stock",
            (xmin, ymin, -candidate.plan.target.cap_depth, xmax, ymax, 0)),
            (), tuple(bodies))
        job = replace(job, stages=tuple(replace(stage,
            axial_limits=ordered_job.AxialLimits(.5, 1)) for stage in job.stages),
            occupancy_setup=setup)
        for dialect in ("uccnc", "grbl"):
            with self.subTest(dialect=dialect):
                selected = _for_dialect(job, dialect)
                files, report = emit(selected, dialect,
                    coordinate_decimals=6 if dialect == "uccnc" else 4)
                self.assertEqual(report["motion_equivalence"]["status"], "pass")
                self.assertEqual(report["stock_access_residual"]["status"], "pass")
                self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")
                stock = report["stock_access_residual"]
                self.assertEqual(stock["scope"],
                    "decoded cumulative cylindrical/Region V from virgin stock")
                self.assertEqual(len(stock["prefixes"]), len(plans)+1)
                self.assertLess(stock["volume_mm3"][1],
                                stock["prefixes"][0]["volume_mm3"][1])
                # Actual bytes, rather than intended geometry, are the safety
                # boundary: turn an emitted high rapid into a below-stock move.
                changed, count = re.subn(rb"(G0[^\r\n]*\bZ)3(?=\r?$)",
                    rb"\g<1>-0.1", files[0], count=1, flags=re.MULTILINE)
                self.assertEqual(count, 1)
                with self.assertRaises(ValueError):
                    audit_files(selected, dialect, (changed,)+files[1:])


if __name__ == "__main__":
    unittest.main()
