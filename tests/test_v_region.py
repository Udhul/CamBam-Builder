"""M3 V profile, continuous occupancy and Region path regressions."""

import math
import json
import unittest
from dataclasses import replace
from pathlib import Path

try:
    from cambam_builder.cam_core import curved_region, replay, v_region
    from cambam_builder.integrations.cambam.native_polygon_rest import SHELL, HOLE
    from shapely.affinity import affine_transform
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _circle(radius, clockwise=False, center=(0, 0)):
    direction = -1 if clockwise else 1
    bulge = direction * math.tan(math.pi / 8)
    return tuple((center[0] + radius * math.cos(direction * i * math.pi / 2),
                  center[1] + radius * math.sin(direction * i * math.pi / 2),
                  bulge) for i in range(4))


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class VRegionTests(unittest.TestCase):
    def test_tool_angle_changes_the_defined_v_target_section(self):
        target = v_region.VTarget.polygon("opening", ((0, 0), (10, 0),
            (10, 10), (0, 10)), (), 2)
        for angle in (60, 90):
            with self.subTest(angle=angle):
                tool = v_region.VProfile("pointed", angle, 0, 3, 2)
                inner = target.section(1, tool.tangent).area
                outer = target.section(1, tool.tangent, outer=True).area
                expected = (10 - 2 * math.tan(math.radians(angle / 2))) ** 2
                self.assertLessEqual(inner, expected)
                self.assertGreaterEqual(outer, expected)
                self.assertLess(outer - inner, .01)
        self.assertGreater(target.section(1, math.tan(math.pi / 6)).area,
                           target.section(1, 1).area + 10)

    def test_primary_raster_reaches_short_and_disconnected_components(self):
        from shapely.geometry import box

        tool = v_region.VProfile("pointed", 90, 0, 2, 1)
        for width in (0.4, 0.08):
            for pattern in ("raster", "offset"):
                with self.subTest(width=width, pattern=pattern):
                    target = v_region.VTarget.polygon("small", ((0, 0),
                        (width, 0), (width, width), (0, width)), (), 1)
                    result = v_region.plan(target, tool, fill_pattern=pattern)
                    self.assertEqual(result.status, "partial")
                    self.assertTrue(result.paths)
                    self.assertGreater(max(p[2] for path in result.paths
                                           for p in path.points), 0)
                    self.assertEqual(v_region.verify(result), result)

        # The thin bridge vanishes under cutter-center erosion.  The large
        # component gets ordinary grid rows; the short component still needs
        # its own row to avoid silently losing that part of the target.
        shape = box(0, 0, 10, 10).union(box(9.9, .245, 12.1, .255)).union(
            box(12, .05, 12.4, .45))
        self.assertEqual(shape.geom_type, "Polygon")
        target = v_region.VTarget("satellite", shape, shape, 1)
        for pattern in ("raster", "offset"):
            with self.subTest(pattern=pattern):
                result = v_region.plan(target, tool, fill_pattern=pattern)
                self.assertEqual(result.status, "partial")
                self.assertTrue(any(path.points[0][0] > 12
                                    for path in result.paths))
                self.assertTrue(any(path.points[0][0] < 10
                                    for path in result.paths))

    def test_short_closed_contour_sampling_preserves_traversal_and_closure(self):
        from shapely.geometry import LineString
        ring = LineString(((0, 0), (.01, 0), (.01, .01), (0, .01), (0, 0)))
        points = v_region._sample(ring, .6)
        self.assertGreaterEqual(len(set(points)), 3)
        self.assertEqual(points[0], points[-1])
        self.assertTrue(all(a != b for a, b in zip(points, points[1:])))
        # Rounding can collapse an open sub-resolution segment; it must not
        # become a zero-length cutting motion with an invented positive depth.
        tiny = LineString(((0, 0), (1e-9, 0)))
        self.assertEqual(v_region._sample(tiny, .6), ((0, 0),))

    def test_generated_tapered_inlay_offset_keeps_short_closed_paths(self):
        from shapely.geometry import box
        from cambam_builder.cam_core import ornamental_inlay, tapered_inlay
        design = tapered_inlay.Design(ornamental_inlay.Design(
            "short-offset-rings", box(0, 0, 3, 3), box(-1, -1, 4, 4),
            .35, .05, .05, 1, 3, .18, .3, .2, 1), 30, .35, .35)
        tool = v_region.VProfile("flat", 30, .03, 2, 2)
        for side in ("receiver", "plug"):
            for target in design.targets(side):
                with self.subTest(side=side, target=target.source_id):
                    plan = v_region.plan(target, tool, stepover_mm=.10,
                        xy_step_mm=.6, margin_mm=.002, safe_z=3,
                        fill_pattern="offset")
                    self.assertTrue(plan.paths)
                    self.assertIs(v_region.verify(plan), plan)
                    self.assertEqual(plan.target.fingerprint, target.fingerprint)
                    closed = []
                    for path in plan.paths:
                        self.assertTrue(all(a[:2] != b[:2]
                            for a, b in zip(path.points, path.points[1:])))
                        if path.points[0][:2] == path.points[-1][:2]:
                            closed.append(path)
                            self.assertGreaterEqual(
                                len({p[:2] for p in path.points}), 3)
                    self.assertTrue(closed)
                    # Preserve the formerly collapsed terminal contours as
                    # positive-length loops, rather than silently dropping them.
                    self.assertTrue(any(sum(math.dist(a[:2], b[:2])
                        for a, b in zip(path.points, path.points[1:])) < .6
                        for path in closed))

    def test_short_flute_makes_partial_cut_and_keeps_deep_target_rest(self):
        target = v_region.VTarget.polygon("deep", ((0, 0), (10, 0),
            (10, 10), (0, 10)), (), 2)
        tool = v_region.VProfile("pointed", 90, 0, 1, .5)
        result = v_region.plan(target, tool)
        self.assertEqual(result.status, "partial")
        self.assertTrue(result.paths)
        self.assertLessEqual(max(p[2] for path in result.paths
                                 for p in path.points), .5)
        self.assertEqual(v_region.verify(result), result)
        # At depth 1 the short flute cannot have removed any material.  The
        # independent square erosion has side 10 - 2*1 = 8 mm.
        lower, upper, _ = v_region.section_report(result, 1)
        self.assertLessEqual(lower, 64)
        self.assertGreaterEqual(upper, 64)
        self.assertLess(upper - lower, .01)

    def test_helical_prior_section_clips_at_reached_depth(self):
        shell = ((0, 0), (10, 0), (10, 10), (0, 10))
        target = v_region.VTarget.polygon("helix", shell, (), 1)
        tool = v_region.VProfile("pointed", 90, 0, 2, 1)
        plan = v_region.VPlan(target, tool, (), (), 5, 0.001, 1,
                              "infeasible", "supplied stock analysis")
        cylinder = replay.ToolProfile("T1", "cylinder", 0.2, 2)
        op = replay.Operation("prior", cylinder,
            replay.Target("square", (0, 0, 10, 10), 1, region_shell=shell))
        at, start, end, high = (5, 5, 5), (5, 5, 0), (6, 5, -1), (6, 5, 5)
        trace = replay.Trace("helix", "XY", at, (op,), (
            replay.Event("tool_change", "T1", at),
            replay.Event("spindle_start", "T1", at),
            replay.Motion("approach", "T1", "prior", at, start, 60),
            replay.ArcMotion("cut", "T1", "prior", start, end, 240, 2, (5.5, 5)),
            replay.Motion("retract", "T1", "prior", end, high, 60),
            replay.Event("spindle_stop", "T1", high)))
        rest = v_region.with_prior(plan, trace)
        lower, upper, _ = v_region.section_report(rest, 0.75, final=False)
        # Only the last quarter of the semicircle has reached this section.
        exact = 8.5**2 - (2*0.2*0.5*math.pi/4 + math.pi*0.2**2)
        self.assertLessEqual(lower, exact)
        self.assertGreaterEqual(upper, exact)
        self.assertLess(upper - lower, 0.03)

    def test_capsule_residual_encloses_independent_area(self):
        target = v_region.VTarget.polygon("capsule", ((0, 0), (10, 0),
            (10, 10), (0, 10)), (), 2)
        tool = v_region.VProfile("pointed", 90, 0, 3, 2)
        path = v_region.VPath("fill", ((4, 5, 1), (6, 5, 1)))
        plan = v_region.VPlan(target, tool, (path,),
            v_region._motions((path,), 5), 5, 0.001, 1, "partial", "analytic test")
        # A length-two unit-radius capsule has area 4 + pi.
        lower, upper, overcut = v_region.section_report(plan, 0)
        exact = 100 - 4 - math.pi
        self.assertLessEqual(lower, exact)
        self.assertGreaterEqual(upper, exact)
        self.assertLess(upper - lower, 0.01)
        self.assertEqual(overcut, 0)

    def test_volume_slabs_enclose_integrated_capsule_and_refine(self):
        target = v_region.VTarget.polygon("volume", ((0, 0), (10, 0),
            (10, 10), (0, 10)), (), 1)
        tool = v_region.VProfile("pointed", 90, 0, 3, 2)
        path = v_region.VPath("fill", ((4, 5, 1), (6, 5, 1)))
        plan = v_region.VPlan(target, tool, (path,),
            v_region._motions((path,), 5), 5, 0.001, 1, "partial", "analytic test")
        # Integral 0..1 of (10-2z)^2 - 4(1-z) - pi(1-z)^2.
        exact = 100 - 20 + 4/3 - 2 - math.pi/3
        coarse = v_region.volume_bounds(plan, slabs=4)
        fine = v_region.volume_bounds(plan, slabs=16)
        for lower, upper in (coarse, fine):
            self.assertLessEqual(lower, exact)
            self.assertGreaterEqual(upper, exact)
        self.assertGreaterEqual(fine[0], coarse[0])
        self.assertLessEqual(fine[1], coarse[1])

    def test_holed_target_section_encloses_true_corner_offsets(self):
        from shapely.geometry import Point
        target = v_region.VTarget.polygon("hole", ((0, 0), (20, 0),
            (20, 20), (0, 20)), (((8, 8), (12, 8), (12, 12), (8, 12)),), 2)
        inner, outer = (target.section(1, 1, outer=flag) for flag in (False, True))
        # Square shell erodes to 18^2; hole dilates by four strips and a disk.
        exact_area = 18**2 - (16 + 16 + math.pi)
        self.assertLessEqual(inner.area, exact_area)
        self.assertGreaterEqual(outer.area, exact_area)
        # Just inside the true unit-radius exclusion at a chord midpoint.
        theta = math.pi / 256
        point = Point(12 + (1 - 1e-5)*math.cos(theta),
                      12 + (1 - 1e-5)*math.sin(theta))
        self.assertFalse(inner.covers(point))
        self.assertEqual(len(inner.interiors), 1)
        self.assertEqual(len(outer.interiors), 1)

    def test_short_rounded_profile_inverse_and_tiny_clearance(self):
        tool = v_region.VProfile("rounded", 60, 0.5, 1, 0.1)
        # From r^2 = 2 R h - h^2, independently choose h and derive r.
        for height in (1e-20, 0.025, 0.1):
            radius = math.sqrt(height - height**2)
            self.assertAlmostEqual(tool.depth_for_radius(radius) / height,
                                   1, delta=1e-12)
        self.assertIsNone(tool.depth_for_radius(-1e-12))
        flat = v_region.VProfile("flat", 90, 0.25, 2, 1)
        self.assertIsNone(flat.depth_for_radius(math.nextafter(0.25, 0)))

    def profiles(self):
        return (
            v_region.VProfile("pointed", 90, 0, 4, 3),
            v_region.VProfile("flat", 90, 0.25, 4, 3),
            v_region.VProfile("rounded", 60, 0.5, 4, 3),
        )

    def test_profile_join_inverse_and_full_height_occupancy(self):
        pointed, flat, rounded = self.profiles()
        self.assertAlmostEqual(pointed.depth_for_radius(2), 2)
        self.assertAlmostEqual(flat.depth_for_radius(2), 1.75)
        self.assertIsNone(flat.depth_for_radius(0.2))
        self.assertAlmostEqual(rounded.join_height, 0.25)
        self.assertAlmostEqual(rounded.radius(rounded.join_height),
                               math.sqrt(3) / 4)
        self.assertAlmostEqual((rounded.radius(rounded.join_height + 1e-5) -
                                rounded.radius(rounded.join_height - 1e-5)) /
                               2e-5, math.tan(math.pi / 6), delta=1e-5)
        for tool in self.profiles():
            for penetration in (0.1, 0.5, 2):
                for section in (0, penetration / 3, penetration):
                    self.assertLessEqual(tool.occupancy_radius(penetration, section),
                                         tool.radius(penetration) + 1e-12)

    def test_accepted_letter_and_curved_annulus_all_profiles(self):
        corpus = json.loads((Path(__file__).parent / "fixtures" /
                             "rest_vcarve_acceptance.json").read_text(encoding="utf-8"))
        limits = next(row["expected"] for row in corpus["cases"]
                      if row["id"] == "M3_polygonal_and_curved_V_paths")
        letter = v_region.VTarget.polygon("A01", SHELL, (HOLE,), 2)
        annulus = v_region.VTarget.curved("M2-annulus", curved_region.approximate(
            _circle(9), (_circle(2, clockwise=True),)), 2)
        for target in (letter, annulus):
            for tool in self.profiles():
                with self.subTest(target=target.source_id, tool=tool.kind):
                    result = v_region.plan(target, tool)
                    self.assertEqual(result.status, "partial")
                    self.assertTrue(any(p.role == "edge" for p in result.paths))
                    self.assertTrue(any(p.role == "fill" for p in result.paths))
                    self.assertTrue(any(len({p[2] for p in path.points}) > 1
                                        for path in result.paths if path.role == "fill"))
                    self.assertEqual(v_region.verify(result), result)
                    lower, upper, overcut = v_region.section_report(result, 1)
                    self.assertLessEqual(lower, upper)
                    self.assertGreater(upper, 0)
                    budget = (limits["A01_residual_upper_mm2"] if target is letter
                              else limits["annulus_residual_upper_mm2"])
                    self.assertLess(upper, budget[tool.kind])
                    self.assertLess(overcut,
                                    limits["protected_overcut_upper_mm2"])
                    if target is annulus and tool.kind == "rounded":
                        volume = v_region.volume_bounds(result, slabs=8)
                        self.assertLessEqual(volume[0], volume[1])
                        self.assertGreater(volume[0], 0)
                        self.assertLess(volume[1], limits[
                            "annulus_rounded_residual_volume_upper_mm3"])
                    self.assertTrue(all(m.start[2] > 0 and m.end[2] > 0
                                        for m in result.motions if m.role == "rapid"))

    def test_mixed_arc_concavity_hole_and_reflection(self):
        shell = ((-16, -10, 0.17), (16, -10, 0), (16, 10, 0),
                 (5, 10, 0), (5, 4, 0), (-5, 4, 0), (-5, 10, 0),
                 (-16, 10, 0))
        hole = _circle(1.5, clockwise=True, center=(0, -3))
        original = curved_region.approximate(shell, (hole,))
        reflected = replace(original,
            nominal=affine_transform(original.nominal, [-1, 0, 0, 1, 40, 7]),
            safe=affine_transform(original.safe, [-1, 0, 0, 1, 40, 7]),
            outer=affine_transform(original.outer, [-1, 0, 0, 1, 40, 7]))
        for name, shape in (("M2-mixed", original),
                            ("M2-mixed-reflected", reflected)):
            target = v_region.VTarget.curved(name, shape, 2)
            result = v_region.plan(target, self.profiles()[2])
            self.assertGreaterEqual(sum(p.role == "edge" for p in result.paths), 2)
            self.assertGreater(sum(p.role == "fill" for p in result.paths), 5)
            lower, upper, overcut = v_region.section_report(result, 1)
            self.assertLess(upper, 5)
            self.assertLess(overcut, 1e-7)

    def test_between_vertex_gouge_and_tampered_motion_fail(self):
        target = v_region.VTarget.polygon("square", ((0, 0), (10, 0),
            (10, 10), (0, 10)), (), 2)
        result = v_region.plan(target, self.profiles()[0], stepover_mm=2)
        bad = v_region.VPath("fill", ((2, 2, 2), (8, 8, 2), (2, 8, 2)))
        with self.assertRaisesRegex(ValueError, "protected Region"):
            v_region.verify(replace(result, paths=(bad,),
                                    motions=v_region._motions((bad,), result.safe_z)))
        with self.assertRaisesRegex(ValueError, "motion differs"):
            v_region.verify(replace(result, motions=result.motions[:-1]))
        steep = v_region.VPath("fill", ((5, 5, 0.1), (5.1, 5, 2)))
        with self.assertRaisesRegex(ValueError, "slope"):
            v_region.verify(replace(result, paths=(steep,),
                                    motions=v_region._motions((steep,), result.safe_z)))
        annulus = v_region.VTarget.curved("annulus", curved_region.approximate(
            _circle(9), (_circle(2, clockwise=True),)), 2)
        curved = v_region.plan(annulus, self.profiles()[0])
        bridge = v_region.VPath("fill", ((-5, 0, 1), (5, 0, 1)))
        with self.assertRaisesRegex(ValueError, "protected Region"):
            v_region.verify(replace(curved, paths=(bridge,),
                                    motions=v_region._motions((bridge,), curved.safe_z)))

    def test_narrow_curved_access_reports_infeasible(self):
        narrow = curved_region.approximate(_circle(3),
                                           (_circle(2.2, clockwise=True),))
        target = v_region.VTarget.curved("M2-narrow", narrow, 1)
        flat = v_region.VProfile("flat", 90, 0.5, 2, 1)
        result = v_region.plan(target, flat)
        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.paths, ())
        self.assertEqual(result.motions, ())

    def test_source_bound_prior_stock_and_v_finish_gain(self):
        shell = ((0, 0), (10, 0), (10, 10), (0, 10))
        target = v_region.VTarget.polygon("source-hash", shell, (), 2)
        plan = v_region.plan(target, self.profiles()[0])
        tool = replay.ToolProfile("T1", "cylinder", 0.5, 3)
        region = replay.Target("opening", (0, 0, 10, 10), 2,
                               region_shell=shell)
        op = replay.Operation("prior", tool, region)
        high, a, b = (4, 5, 5), (4, 5, -2), (6, 5, -2)
        items = (replay.Event("tool_change", "T1", high),
                 replay.Event("spindle_start", "T1", high),
                 replay.Motion("entry", "T1", "prior", high, a),
                 replay.Motion("cut", "T1", "prior", a, b),
                 replay.Motion("retract", "T1", "prior", b, (6, 5, 5)),
                 replay.Event("spindle_stop", "T1", (6, 5, 5)))
        trace = replay.Trace(target.source_id, "square", high, (op,), items)
        rest = v_region.with_prior(plan, trace)
        self.assertGreater(v_region.section_report(rest, 1, final=False)[0],
                           v_region.section_report(rest, 1)[1])
        with self.assertRaisesRegex(ValueError, "stale"):
            v_region.with_prior(plan, replace(trace,
                source_fingerprint="changed"))
        near = (1, 5, -2)
        bad_items = items[:2] + (
            replay.Motion("entry", "T1", "prior", (1, 5, 5), near),
            replay.Motion("cut", "T1", "prior", near, (2, 5, -2)),
            replay.Motion("retract", "T1", "prior", (2, 5, -2), (2, 5, 5)),
            replay.Event("spindle_stop", "T1", (2, 5, 5)))
        bad_items = (replay.Event("tool_change", "T1", (1, 5, 5)),
                     replay.Event("spindle_start", "T1", (1, 5, 5))) + bad_items[2:]
        with self.assertRaisesRegex(ValueError, "capped finish target"):
            v_region.with_prior(plan, replay.Trace(target.source_id, "square",
                (1, 5, 5), (op,), bad_items))


if __name__ == "__main__":
    unittest.main()
