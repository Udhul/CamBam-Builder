"""DT01 fixed designs, independent contact witnesses and decoded consumers."""
from dataclasses import replace
import math
import unittest

try:
    from shapely.geometry import Point, box
    from cambam_builder.cam_core import curved_region, ordered_job, replay, v_region
    from cambam_builder.integrations.ordered_output import emit
    from tests.test_standalone_v import _circle, _job
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class FixedVDesignTests(unittest.TestCase):
    def target(self, **kwargs):
        return v_region.VTarget.polygon("design-source",
            ((0, 0), (20, 0), (20, 16), (0, 16)),
            (((2, 6), (12, 6), (12, 10), (2, 10)),), 2,
            design_angle_degrees=90, **kwargs)

    def tools(self):
        return (v_region.VProfile("pointed", 90, 0, 5, 2),
                v_region.VProfile("flat", 90, .6, 5, 2),
                v_region.VProfile("pointed", 60, 0, 5, 2),
                v_region.VProfile("pointed", 120, 0, 5, 2),
                v_region.VProfile("rounded", 60, .5, 5, 2))

    def supplied(self, target, tool, points):
        paths = (v_region.VPath("fill", points),)
        return v_region.VPlan(target, tool, paths, v_region._motions(paths, 3),
                              3, .001, 1, "partial", "independent witness")

    def test_holed_ornament_tools_share_sections_volume_and_decoded_design(self):
        target = self.target()
        # At depth .5, outer rectangle is 19x15. The 10x4 island
        # expands by perimeter*.5 + pi*.5^2. No offset boundaries meet.
        expected = 19*15 - (40 + 28*.5 + math.pi*.25)
        self.assertLessEqual(target.section(.5).area, expected)
        self.assertGreaterEqual(target.section(.5, outer=True).area, expected)
        self.assertLess(target.section(.5, outer=True).area -
                        target.section(.5).area, .02)
        volume = target.volume_bounds()
        fingerprints = set()
        for tool in self.tools():
            with self.subTest(tool=tool):
                plan = v_region.plan(target, tool, stepover_mm=3, xy_step_mm=2)
                self.assertIs(plan.target, target)
                self.assertEqual(plan.status, "partial")
                self.assertEqual(plan.target.volume_bounds(), volume)
                self.assertEqual(plan.target.section(.5).wkb, target.section(.5).wkb)
                evidence = v_region.section_evidence(plan, .75)
                self.assertEqual(evidence.design_fingerprint, target.fingerprint)
                self.assertGreater(evidence.residual_inner.area, 0)
                self.assertTrue(evidence.residual_outer.covers(evidence.residual_inner))
                self.assertTrue(evidence.known_free_outer.covers(evidence.known_free_inner))
                self.assertLess(evidence.possible_overcut.area, 1e-6)
                job, _ = _job(plan)
                _, report = emit(job, "uccnc")
                stock = report["stock_access_residual"]
                self.assertEqual(stock["status"], "pass")
                self.assertEqual(stock["design_fingerprint"], target.fingerprint)
                self.assertEqual(stock["design_angle_degrees"], 90)
                fingerprints.add(plan.fingerprint)
        self.assertEqual(len(fingerprints), len(self.tools()))

    def test_curved_annulus_uses_same_contract_and_independent_volume(self):
        target = v_region.VTarget.curved("ring", curved_region.approximate(
            _circle(8), (_circle(2, clockwise=True),)), 1,
            design_angle_degrees=90)
        # pi*((8-z)^2-(2+z)^2) = pi*(60-20z).
        inner, outer = target.volume_bounds(slabs=32)
        self.assertLessEqual(inner, 50*math.pi)
        self.assertGreaterEqual(outer, 50*math.pi)
        self.assertLess(outer-inner, 3)
        for tool in (self.tools()[2], self.tools()[3], self.tools()[4]):
            plan = v_region.plan(target, tool, fill_pattern="offset",
                                 stepover_mm=3, xy_step_mm=2)
            self.assertIs(plan.target, target)
            self.assertLess(v_region.section_report(plan, .5)[2], 1e-6)
            _, report = emit(_job(plan)[0], "grbl")
            self.assertEqual(report["stock_access_residual"]["design_fingerprint"],
                             target.fingerprint)

    def test_sharp_broad_and_rounded_all_height_overcut_witnesses(self):
        target = v_region.VTarget.polygon("square", ((0, 0), (10, 0),
            (10, 10), (0, 10)), (), 2, design_angle_degrees=90)
        cases = ((self.tools()[2], 1, .8),   # tip/floor, top fits
                 (self.tools()[3], 1, 1.3), # top, tip/floor fits
                 (self.tools()[4], .5, .65)) # rounded interior, both endpoints fit
        for tool, depth, clearance in cases:
            with self.subTest(tool=tool):
                plan = self.supplied(target, tool,
                    ((clearance, 4, depth), (clearance, 6, depth)))
                with self.assertRaisesRegex(ValueError, "protected Region"):
                    v_region.verify(plan)
        rounded = self.tools()[4]
        self.assertLess(rounded.radius(.5), .65)
        self.assertLess(.5*target.tangent, .65)
        self.assertAlmostEqual(v_region.contact_radius(target, rounded, .5),
                               math.sqrt(.5), places=12)
        # Independent profile sampling verifies the analytic maximum enclosure.
        for tool in self.tools():
            for depth in (.1, .5, 1, 2):
                maximum = v_region.contact_radius(target, tool, depth)
                for i in range(101):
                    t = depth*i/100
                    self.assertLessEqual(t + tool.radius(depth-t), maximum+1e-12)

    def test_continuous_hole_crossing_and_varying_z(self):
        target = self.target()
        tool = self.tools()[2]
        # Safe endpoints do not authorize a chord across the protected island.
        bad = self.supplied(target, tool, ((1, 8, .2), (16, 8, .3)))
        with self.assertRaises(ValueError):
            v_region.verify(bad)
        good = self.supplied(target, tool, ((16, 5, .5), (16, 10, 1.5)))
        self.assertIs(v_region.verify(good), good)

    def test_legacy_mapping_freezes_before_cutter_comparison(self):
        legacy = v_region.VTarget.polygon("old", ((0, 0), (10, 0),
            (10, 10), (0, 10)), (), 2)
        with self.assertRaises(ValueError):
            legacy.section(1)
        first = v_region.plan(legacy, self.tools()[0])
        second = v_region.plan(first.target, self.tools()[2])
        mutated = replace(first, tool=self.tools()[2])
        self.assertEqual(first.target.fingerprint, second.target.fingerprint)
        self.assertEqual(first.target.fingerprint, mutated.target.fingerprint)
        self.assertNotEqual(first.fingerprint, mutated.fingerprint)
        self.assertEqual(first.target.section(1).wkb, mutated.target.section(1).wkb)
        with self.assertRaisesRegex(ValueError, "differs from fixed"):
            first.target.section(1, self.tools()[2].tangent)
        v_region.verify(mutated)

    def test_design_mutations_and_frame_binding_invalidate_output(self):
        target = self.target()
        plan = v_region.plan(target, self.tools()[0], stepover_mm=4, xy_step_mm=3)
        job, stage = _job(plan)
        for changed in (replace(target, source_id="other"),
                        replace(target, design_angle_degrees=100),
                        replace(target, cap_depth=2.1),
                        replace(target, safe=box(0, 0, 1, 1)),
                        replace(target, frame="part")):
            with self.subTest(changed=changed.fingerprint):
                self.assertNotEqual(changed.fingerprint, target.fingerprint)
                altered = replace(plan, target=changed)
                with self.assertRaises(ValueError):
                    altered_job = replace(job, stages=(replace(stage, v_plan=altered),))
                    emit(altered_job, "uccnc")
        altered = replace(plan, tool=self.tools()[2])
        with self.assertRaises(ValueError):
            emit(replace(job, stages=(replace(stage, v_plan=altered),)), "uccnc")

    def test_short_and_infeasible_tools_keep_cap_and_located_required_stock(self):
        target = self.target()
        short = v_region.VProfile("pointed", 60, 0, 2, .5)
        plan = v_region.plan(target, short)
        self.assertEqual(plan.target.cap_depth, 2)
        self.assertEqual(plan.status, "partial")
        section = v_region.section_evidence(plan, 1)
        self.assertTrue(section.known_free_outer.is_empty)
        self.assertTrue(section.residual_inner.equals(target.section(1)))
        tiny = v_region.VTarget.polygon("tiny", ((0, 0), (.4, 0),
            (.4, .4), (0, .4)), (), 1, design_angle_degrees=90)
        plan = v_region.plan(tiny, self.tools()[1])
        self.assertEqual(plan.status, "infeasible")
        self.assertEqual(plan.target.cap_depth, 1)
        self.assertGreater(v_region.section_evidence(plan, .05).residual_inner.area, 0)

    def test_depth_dependent_endmill_roughing_preserves_original_design(self):
        from tests.test_ordered_job import _rectangle
        # Existing caller composition is a supplied cylinder prefix, then V.
        base, original_prior, start = _rectangle()
        job = ordered_job.from_prior_v(base, original_prior, start)
        target = replace(base.target, design_angle_degrees=90, frame=job.program_frame)
        plan = v_region.plan(target, self.tools()[2], stepover_mm=2, safe_z=3)
        self.assertLess(target.roughing_centers(.8, .2).area,
                        target.roughing_centers(.2, .2).area)
        self.assertTrue(target.section(.8).covers(target.roughing_centers(.8, .2)))
        # Source/plan revision binds design separately from derived boundaries.
        revised = ordered_job.from_prior_v(plan, original_prior, start)
        _, report = emit(revised, "uccnc")
        self.assertEqual(report["stock_access_residual"]["design_fingerprint"],
                         target.fingerprint)
        prior = job.stages[0]
        # Put a floor cut next to the outer opening: cylinder fits opening but
        # erases fixed V wall. Ordinary replay must not establish V protection.
        # Use lower-level trace for an independently authored complete prefix.
        op = prior.operation
        start = (1, 1, 3)
        a, b = (.6, 2, -1), (.6, 4, -1)
        high_a, high_b = (a[0], a[1], 3), (b[0], b[1], 3)
        trace = replay.Trace(target.source_id, target.frame, start, (op,), (
            replay.Event("tool_change", op.tool.name, start),
            replay.Event("spindle_start", op.tool.name, start),
            replay.Motion("rapid", op.tool.name, op.name, start, high_a),
            replay.Motion("entry", op.tool.name, op.name, high_a, a, 60),
            replay.Motion("cut", op.tool.name, op.name, a, b, 100),
            replay.Motion("retract", op.tool.name, op.name, b, high_b, 60),
            replay.Event("spindle_stop", op.tool.name, high_b)))
        with self.assertRaises(ValueError):
            v_region.with_prior(plan, trace)

    def test_cached_rest_cannot_bypass_source_frame_design_or_stock_checks(self):
        from tests.test_ordered_job import _rectangle
        plan, prior, _ = _rectangle()
        rest = v_region.with_prior(plan, prior)
        for changed in (replace(plan.target, source_id="stale"),
                        replace(plan.target, frame="other"),
                        replace(plan.target, design_angle_degrees=160)):
            with self.subTest(changed=changed.fingerprint):
                mutated = replace(rest, plan=replace(plan, target=changed))
                with self.assertRaises(ValueError):
                    v_region.section_evidence(mutated, 1)
        with self.assertRaisesRegex(ValueError, "prior stock evidence"):
            v_region.section_evidence(replace(rest,
                prior_stock=replace(rest.prior_stock, cuts=())), 1)

    def test_derived_centers_drive_two_depth_roughing_against_original_target(self):
        from tests.test_ordered_job import _rectangle
        old, prior, start = _rectangle()
        target = replace(old.target, design_angle_degrees=90, frame=prior.frame)
        plan = v_region.plan(target, self.tools()[2], safe_z=start[2], stepover_mm=3)
        op = prior.operations[0]
        tool = op.tool.name
        items = [replay.Event("tool_change", tool, start),
                 replay.Event("spindle_start", tool, start)]
        at = start
        xs = []
        for depth in (.2, .8):
            centers = target.roughing_centers(depth, op.tool.radius)
            x = centers.bounds[0] + .02
            xs.append(x)
            high, low, end, up = (x, 2, 3), (x, 2, -depth), (x, 4, -depth), (x, 4, 3)
            items.extend((replay.Motion("rapid", tool, op.name, at, high),
                          replay.Motion("entry", tool, op.name, high, low, 60),
                          replay.Motion("cut", tool, op.name, low, end, 100),
                          replay.Motion("retract", tool, op.name, end, up, 60)))
            at = up
        items.extend((replay.Motion("rapid", tool, op.name, at, start),
                      replay.Event("spindle_stop", tool, start)))
        trace = replay.Trace(target.source_id, target.frame, start, (op,), tuple(items))
        rest = v_region.with_prior(plan, trace)
        deep = v_region.section_evidence(rest, .8, final=False)
        self.assertTrue(deep.known_free_inner.covers(Point(xs[1], 3)))
        self.assertFalse(deep.known_free_outer.covers(Point(xs[0], 3)))
        self.assertLess(deep.possible_overcut.area, 1e-6)
        self.assertEqual(deep.design_fingerprint, target.fingerprint)
        # The original opening remains the replay target. Replacing it with
        # derived centers is not generalized native-boundary binding.
        shape = target.roughing_centers(.8, op.tool.radius)
        derived = replace(op.target, bounds=shape.bounds,
                          region_shell=tuple(shape.exterior.coords[:-1]))
        with self.assertRaises(ValueError):
            v_region.with_prior(plan, replace(trace, operations=(replace(op, target=derived),)))
        _, report = emit(ordered_job.from_prior_v(plan, trace, start), "uccnc")
        self.assertEqual(report["stock_access_residual"]["design_fingerprint"],
                         target.fingerprint)

    def test_nonbinary_caps_and_invalid_design_controls(self):
        for cap in (.1, .2):
            target = v_region.VTarget.polygon("decimal-cap", ((0, 0), (10, 0),
                (10, 10), (0, 10)), (), cap, design_angle_degrees=90)
            expected = 100*cap - 20*cap**2 + 4/3*cap**3
            inner, outer = target.volume_bounds(slabs=11)
            self.assertLessEqual(inner, expected)
            self.assertGreaterEqual(outer, expected)
        for angle in (0, 180, float("nan"), True):
            with self.assertRaises(ValueError):
                replace(self.target(), design_angle_degrees=angle)
        for frame in ("", 5):
            with self.assertRaises(ValueError):
                self.target(frame=frame)
