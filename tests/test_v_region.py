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
