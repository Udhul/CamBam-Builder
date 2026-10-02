"""MX01 mixed decoded stock, independent capsule witnesses and access limits."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

try:
    from cambam_builder.cam_core import ordered_job, replay, v_region
    from cambam_builder.cam_core.occupancy import (
        Box, OccupancySetup, ToolBand, ToolBody,
    )
    from cambam_builder.integrations import ordered_dialects
    from cambam_builder.integrations.ordered_output import (
        audit_bundle, audit_files, emit, write_bundle,
    )
    from tests.test_multistage_v import _for_dialect, _job, _supplied
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _cylinder(target, name, tool_id, radius, a, b, *, depths=(.5, 1, 1.5)):
    """Actual entry/cut/retract depth passes; no supplied cleared opening."""
    start = (-2, -2, 3)
    tool = replay.ToolProfile(tool_id, "cylinder", radius, 3)
    shell = tuple(target.safe.exterior.coords)[:-1]
    holes = tuple(tuple(hole.coords)[:-1] for hole in target.safe.interiors)
    op = replay.Operation(name, tool, replay.Target(
        "original-design", target.safe.bounds, target.cap_depth,
        region_shell=shell, region_holes=holes))
    moves = []
    at = start
    for depth in depths:
        high = (*a, 3)
        if at != high:
            moves.append(ordered_job.JobMove("rapid", at, high))
        low, end = (*a, -depth), (*b, -depth)
        moves.extend((ordered_job.JobMove("entry", high, low, 60),
                      ordered_job.JobMove("cut", low, end, 180),
                      ordered_job.JobMove("retract", end, (*b, 3), 100)))
        at = (*b, 3)
    moves.append(ordered_job.JobMove("rapid", at, start))
    return ordered_job.Stage(name, tool_id, tuple(moves), 11000,
        operation=op, source_revision="synthetic-original-design-motion",
        axial_limits=ordered_job.AxialLimits(.5, 1.5))


def _trace(target, stage):
    at = stage.motions[0].start
    items = [replay.Event("tool_change", stage.tool_id, at),
             replay.Event("spindle_start", stage.tool_id, at)]
    items.extend(replay.Motion(move.role, stage.tool_id, stage.id,
                               move.start, move.end, move.feed)
                 for move in stage.motions)
    items.append(replay.Event("spindle_stop", stage.tool_id, stage.motions[-1].end))
    return replay.Trace(target.source_id, "program", at, (stage.operation,),
                        tuple(items))


def _mixed_job(*, island=False, angle=90):
    shell = ((0, 0), (16, 0), (16, 12), (0, 12))
    holes = (((7, 8), (9, 8), (9, 10), (7, 10)),) if island else ()
    target = v_region.VTarget.polygon("MX01-island" if island else "MX01-capsules",
        shell, holes, 1.5, design_angle_degrees=90)
    tool = v_region.VProfile("pointed", angle, 0, 4, 3)
    plans = v_region.depth_passes(
        _supplied(target, tool, (7, 5, 1.5), (9, 5, 1.5)), .5)
    stages = [_cylinder(target, "rough", "T1", .5, (4, 5), (6, 5))]
    start = stages[0].motions[0].start
    for i, plan in enumerate(plans):
        stages.append(ordered_job.Stage(f"v-pass-{i+1}", "T3", tuple(
            ordered_job.JobMove(m.role, m.start, m.end,
                               0 if m.role == "rapid" else 100)
            for m in v_region.complete_motion(plan, start)), 11000,
            v_plan=plan, source_revision=plan.fingerprint,
            axial_limits=ordered_job.AxialLimits(.5, 1.5)))
    stages.append(_cylinder(target, "cleanup", "T2", .4, (9, 5), (11, 5)))
    stages = tuple(stage if i == 0 else replace(stage,
        transition=ordered_job.Transition("operator", "split", stage.tool_id,
            start, "offline-declared-install")) for i, stage in enumerate(stages))
    bodies = tuple(ToolBody(tool_id, (
        ToolBand("cutter", 0, 3, radius),
        ToolBand("shank", 3, 4, radius),
        ToolBand("holder", 4, 6, .8),
    )) for tool_id, radius in (("T1", .5), ("T3", tool.radius(3)), ("T2", .4)))
    setup = OccupancySetup("program", Box("original-stock", (0, 0, -1.5, 16, 12, 0)),
                           (), bodies)
    return target, ordered_job.Job(target.source_id, stages, start,
                                   occupancy_setup=setup)


def _composition(target, stages):
    return v_region.VComposition(target, tuple(stage.v_plan if stage.v_plan is not None
        else _trace(target, stage) for stage in stages))


def _decoded_composition(target, job, files, dialect):
    """Recover this straight fixture's sweeps directly from decoded XYZ."""
    decoded = ordered_dialects.decode(files, dialect, initial_work_tip=job.initial_tip)
    sweeps = []
    for stage, observed in zip(job.stages, decoded.stages):
        moves = tuple(ordered_job.JobMove(expected.role, move.start, move.end,
            0 if expected.role == "rapid" else move.feed)
            for expected, move in zip(stage.motions, observed.moves))
        if stage.operation is not None:
            sweeps.append(_trace(target, replace(stage, motions=moves)))
        else:
            cuts = tuple(move for move in moves if move.role == "cut")
            if len(cuts) != 1:
                raise AssertionError("independent straight witness expects one V segment")
            cut = cuts[0]
            sweeps.append(_supplied(target, stage.v_plan.tool,
                (*cut.start[:2], -cut.start[2]), (*cut.end[:2], -cut.end[2])))
    return v_region.VComposition(target, tuple(sweeps))


def _capsule_area(length, radius):
    return 2 * length * radius + math.pi * radius * radius


def _decoded_removed(stage, observed, point, depth):
    """Independent disk/capsule membership from this fixture's decoded cuts."""
    for expected, move in zip(stage.motions, observed.moves):
        if expected.role != "cut":
            continue
        # The fixture intentionally supplies constant-depth straight cuts.
        if move.start[2] != move.end[2] or -move.end[2] < depth:
            continue
        radius = (stage.operation.tool.radius if stage.operation is not None else
                  (-move.end[2]-depth)*math.tan(
                      math.radians(stage.v_plan.tool.angle_degrees)/2))
        ax, ay = move.start[:2]
        dx, dy = move.end[0]-ax, move.end[1]-ay
        fraction = max(0, min(1, ((point[0]-ax)*dx+(point[1]-ay)*dy)/(dx*dx+dy*dy)))
        if math.hypot(point[0]-ax-fraction*dx, point[1]-ay-fraction*dy) <= radius:
            return True
    return False


def _independent_removed_area(depth, intervals=800):
    """Simpson-integrate the union's analytic transverse intervals.

    All centerlines share Y=5, so the union at X is one symmetric interval.
    This uses no Shapely, production sections, reconstructed plans or replay.
    """
    capsules = ((4, 6, .5), (7, 9, 1.5-depth), (9, 11, .4))
    left, right = 3.5, 11.4
    dx = (right-left) / intervals

    def height(x):
        return 2 * max(math.sqrt(max(0, r*r-max(a-x, 0, x-b)**2))
                       for a, b, r in capsules)

    return dx / 3 * (height(left)+height(right)+sum(
        (4 if i % 2 else 2) * height(left+i*dx)
        for i in range(1, intervals)))


def _independent_volume(slabs):
    dz = 1.5 / slabs
    values = [(16-2*i*dz)*(12-2*i*dz)-_independent_removed_area(i*dz)
              for i in range(slabs+1)]
    return dz / 3 * (values[0]+values[-1]+sum(
        (4 if i % 2 else 2)*values[i] for i in range(1, slabs)))


def _output_tempdir():
    output = Path(__file__).resolve().parents[1] / "output"
    output.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="mixed-v-regression-", dir=output)


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class MixedVCompositionTests(unittest.TestCase):
    def test_decoded_prefixes_enclose_exact_overlap_and_positive_cleanup(self):
        target, job = _mixed_job()
        files, report = emit(job, "uccnc")
        stock = report["stock_access_residual"]
        self.assertEqual(stock["scope"],
                         "decoded cumulative cylindrical/Region V from virgin stock")
        self.assertEqual(stock["design_fingerprint"], target.fingerprint)
        self.assertEqual(stock["status"], "pass")
        self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")
        prefixes = stock["prefixes"]
        self.assertEqual(len(prefixes), len(job.stages))
        previous_section = (16-2)*(12-2)
        for stage, identity, prefix in zip(job.stages, job.prefixes, prefixes):
            self.assertEqual(prefix["stage_id"], stage.id)
            self.assertEqual(prefix["prefix_fingerprint"], identity)
            self.assertEqual(len(prefix["decoded_sequence_fingerprint"]), 64)
            bounds = prefix["section_1_mm2"]
            self.assertLessEqual(bounds[0], bounds[1])
            self.assertLessEqual(bounds[1], previous_section+1e-7)
            previous_section = bounds[1]
        # At section Z=-1, T1 and V have radius .5; cleanup has radius .4.
        # T1 and V touch without overlapping. V/cleanup overlap positively.
        r, q = .5, .4
        cross = math.sqrt(r*r-q*q)
        primitive = lambda x: .5*(x*math.sqrt(max(0, r*r-x*x)) +
                                  r*r*math.asin(x/r))
        overlap = math.pi*q*q/2 + 2*(q*cross+primitive(r)-primitive(cross))
        first_area = _capsule_area(2, r)
        final_area = 2*first_area + _capsule_area(2, q)-overlap
        self.assertGreater(overlap, .5)
        self.assertGreater(_capsule_area(2, q)-overlap, 1.49)
        self.assertAlmostEqual(_independent_removed_area(1), final_area, delta=.0002)
        expected = (16-2)*(12-2)-final_area
        final_bounds = stock["section_1_mm2"]
        self.assertLessEqual(final_bounds[0], expected)
        self.assertGreaterEqual(final_bounds[1], expected)
        self.assertGreater(prefixes[-2]["section_1_mm2"][0]-final_bounds[1], 1.47)
        last_only = _composition(target, job.stages[-1:])
        self.assertGreater(v_region.section_report(last_only, 1)[0]-final_bounds[1], 4.9)
        # Decode directly for a distinct stage-only membership witness.
        decoded = ordered_dialects.decode(files, "uccnc", initial_work_tip=job.initial_tip)
        self.assertEqual(tuple(len(stage.moves) for stage in decoded.stages),
                         tuple(len(stage.motions) for stage in job.stages))
        from shapely.geometry import Point
        witnesses = ((4, 5.2), (8, 5.2), (11, 5.2))
        expected_by_prefix = ((True, False, False), (True, False, False),
                              (True, False, False), (True, True, False),
                              (True, True, True))
        for count, expected in enumerate(expected_by_prefix, 1):
            evidence = v_region.section_evidence(_composition(target, job.stages[:count]), 1)
            for point, removed in zip(witnesses, expected):
                independently_removed = any(_decoded_removed(stage, observed, point, 1)
                    for stage, observed in zip(job.stages[:count], decoded.stages[:count]))
                self.assertEqual(independently_removed, removed)
                if removed:
                    self.assertTrue(evidence.known_free_inner.covers(Point(point)))
                else:
                    self.assertFalse(evidence.known_free_outer.covers(Point(point)))
        union = v_region.section_evidence(_composition(target, job.stages), 1)
        self.assertFalse(union.known_free_outer.covers(Point(8, 7)))

    def test_volume_union_encloses_independent_quadrature(self):
        target, job = _mixed_job()
        files, _ = emit(job, "uccnc")
        decoded = _decoded_composition(target, job, files, "uccnc")
        coarse, fine = _independent_volume(120), _independent_volume(240)
        self.assertAlmostEqual(coarse, fine, delta=.003)
        bounds = v_region.volume_bounds(decoded, slabs=128)
        self.assertLessEqual(bounds[0], fine-.003)
        self.assertGreaterEqual(bounds[1], fine+.003)
        prior = v_region.volume_bounds(v_region.VComposition(target, decoded.stages[:-1]),
                                       slabs=128)
        self.assertGreater(prior[0]-bounds[1], .8)

    def test_island_different_angle_and_both_dialects(self):
        target, job = _mixed_job(island=True, angle=60)
        sections = []
        for dialect in ("uccnc", "grbl"):
            with self.subTest(dialect=dialect):
                candidate = _for_dialect(job, dialect)
                files, report = emit(candidate, dialect)
                stock = report["stock_access_residual"]
                self.assertEqual(stock["status"], "pass")
                self.assertEqual(stock["design_angle_degrees"], 90)
                self.assertEqual(stock["design_fingerprint"], target.fingerprint)
                self.assertEqual(len(files), len(job.stages) if dialect == "uccnc" else 1)
                self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")
                sections.append(stock["section_1_mm2"])
        self.assertEqual(sections[0], sections[1])
        self.assertEqual(job.stages[1].v_plan.tool.angle_degrees, 60)
        from shapely.geometry import LineString, Point
        evidence = v_region.section_evidence(_composition(target, job.stages), 1)
        self.assertFalse(evidence.required_outer.covers(Point(8, 9)))
        self.assertFalse(evidence.known_free_outer.covers(Point(8, 9)))
        self.assertTrue(evidence.residual_inner.covers(Point(3, 3)))
        self.assertEqual(evidence.possible_overcut.area, 0)
        crossed = _cylinder(target, "island-crossing", "T9", .4, (6, 7.4), (10, 7.4))
        line = LineString(((6, 7.4), (10, 7.4)))
        self.assertTrue(target.safe.covers(line))
        self.assertGreater(line.distance(target.safe.boundary), .4)
        with self.assertRaisesRegex(ValueError, "crosses capped finish target"):
            _composition(target, (crossed,))

    def test_omitted_depth_pass_and_entry_limits_reject(self):
        _, job = _mixed_job()
        emit(job, "uccnc")
        skipped = replace(job, stages=(job.stages[0],)+job.stages[3:])
        with self.assertRaisesRegex(ValueError, "axial|stepdown"):
            emit(skipped, "uccnc")
        stage = replace(job.stages[-1], axial_limits=ordered_job.AxialLimits(.5, .4))
        with self.assertRaisesRegex(ValueError, "entry|plunge"):
            emit(replace(job, stages=job.stages[:-1]+(stage,)), "uccnc")
        with self.assertRaises(ValueError):
            emit(replace(job, occupancy_setup=None), "uccnc")
        with self.assertRaises(ValueError):
            emit(replace(job, stages=(replace(job.stages[0], axial_limits=None),)
                         +job.stages[1:]), "uccnc")

    def test_pass_clipping_preserves_design_and_allowance(self):
        target, job = _mixed_job()
        final = job.stages[-2].v_plan
        passes = v_region.depth_passes(final, .4, depth_cap_mm=1.1)
        depths = tuple(max(p[2] for path in plan.paths for p in path.points)
                       for plan in passes)
        self.assertEqual(len(passes), 3)
        for actual, expected in zip(depths, (.4, .8, 1.1)):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(plan.target.fingerprint == target.fingerprint for plan in passes))
        self.assertGreater(v_region.volume_bounds(passes[-1])[0], 0)
        for invalid in (0, -1, math.nan, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                v_region.depth_passes(final, invalid)

    def test_crossing_depth_witnesses_cannot_invent_interior_clearance(self):
        target = v_region.VTarget.polygon("MX01-crossing-depths",
            ((0, 0), (16, 0), (16, 12), (0, 12)), (), 2,
            design_angle_degrees=90)
        tool = v_region.VProfile("pointed", 90, 0, 4, 3)
        plans = tuple(_supplied(target, tool, (5, 5, a), (9, 5, b))
                      for a, b in ((.5, 1.5), (1.5, .5), (1.6, 1.6)))
        job = _job(plans)
        job = replace(job, stages=tuple(replace(stage,
            axial_limits=ordered_job.AxialLimits(1.5 if i < 2 else .25, 2))
            for i, stage in enumerate(job.stages)))
        # Earlier depths at the midpoint reach only 1.0, despite each endpoint
        # separately reaching 1.5. A constant 1.6 pass advances .6 there.
        self.assertGreater(1.6-max((.5+1.5)/2, (1.5+.5)/2), .25)
        with self.assertRaisesRegex(ValueError, "stepdown"):
            emit(job, "uccnc")

    def test_reverse_retrace_and_invalid_axial_inputs(self):
        target, job = _mixed_job()
        tool = job.stages[1].v_plan.tool
        plans = (_supplied(target, tool, (7, 5, .5), (9, 5, .5)),
                 _supplied(target, tool, (9, 5, 1), (7, 5, 1)),
                 _supplied(target, tool, (7, 5, 1.5), (9, 5, 1.5)))
        candidate = _job(plans)
        candidate = replace(candidate, stages=tuple(replace(stage,
            axial_limits=ordered_job.AxialLimits(.5, 1.5)) for stage in candidate.stages))
        _, report = emit(candidate, "uccnc")
        self.assertEqual(tuple(stage["maximum_observed_advance_mm"]
                               for stage in report["axial_process_limits"]["stages"]),
                         (.5, .5, .5))
        for invalid in (0, -1, math.nan, math.inf, True, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ordered_job.AxialLimits(invalid, 1)
                with self.assertRaises(ValueError):
                    ordered_job.AxialLimits(1, invalid)

    def test_holder_collision_between_clear_endpoints(self):
        _, job = _mixed_job()
        clamp = Box("middle-clamp", (4.9, 5.7, 3.8, 5.1, 5.8, 4.1))
        for x in (4, 6):
            self.assertGreater(math.hypot(abs(x-5)-.1, .7), .8)
        self.assertGreater(clamp.bounds[1]-5, .5)  # T1 cutter clears.
        collision = replace(job, occupancy_setup=replace(job.occupancy_setup,
                                                         fixtures=(clamp,)))
        with self.assertRaisesRegex(ValueError, "holder collision with fixture middle-clamp"):
            emit(collision, "uccnc")

    def test_full_cylinder_and_original_boundary_are_required(self):
        target, job = _mixed_job()
        # Centerline lies in the deepest design section; radius .5 does not.
        crossed = _cylinder(target, "crossed", "T9", .5, (1.7, 4), (1.7, 6))
        from shapely.geometry import LineString
        self.assertTrue(target.section(1.5).covers(LineString(((1.7, 4), (1.7, 6)))))
        with self.assertRaisesRegex(ValueError, "crosses capped finish target"):
            _composition(target, (crossed,))
        altered = replace(target, design_angle_degrees=60)
        with self.assertRaisesRegex(ValueError, "design targets differ"):
            v_region.VComposition(altered, (job.stages[1].v_plan,))
        unbound = replace(target, design_angle_degrees=None)
        with self.assertRaisesRegex(ValueError, "fixed design"):
            v_region.VComposition(unbound, (_trace(target, job.stages[0]),))

    def test_bundle_freshness_binds_order_limits_and_protected_boundary(self):
        target, job = _mixed_job()
        with _output_tempdir() as temp:
            bundle = Path(temp) / "mixed-job"
            report = write_bundle(bundle, job, "uccnc")
            manifest = bundle / "handoff.json"
            self.assertEqual(audit_bundle(manifest, job), report)
            changed = replace(job.stages[-1], axial_limits=ordered_job.AxialLimits(.6, 1.5))
            edited = replace(job, stages=job.stages[:-1]+(changed,))
            self.assertNotEqual(edited.fingerprint, job.fingerprint)
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_bundle(manifest, edited)
            files = tuple((bundle / f"stage-{i+1}.nc").read_bytes()
                          for i in range(len(job.stages)))
            with self.assertRaises(ValueError):
                audit_files(job, "uccnc", tuple(reversed(files)))
        changed_plan = replace(job.stages[1].v_plan,
                               target=replace(target, frame="changed-frame"))
        with self.assertRaisesRegex(ValueError, "design targets differ"):
            v_region.VComposition(target, (changed_plan,))


if __name__ == "__main__":
    unittest.main()
