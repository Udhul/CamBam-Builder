"""RP01 synthetic ornamental consumer and measured baseline comparison."""
from dataclasses import replace
import math
import unittest

try:
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    from cambam_builder.cam_core import planar_rest, v_region as v
    from tests.test_multistage_v import _membership_margins
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _frieze():
    # Polygonal lobes are authored design, not an uncertified circle adapter.
    shape = unary_union([box(0, 0, 20, 8)] + [
        Point(x, 8).buffer(3, quad_segs=8) for x in (3, 10, 17)] + [
        box(9.5, 10, 10.5, 13), Point(10, 13).buffer(1.3, quad_segs=8)
    ]).difference(box(8, 3, 10, 4))
    target = v.VTarget("RP01-frieze", shape, shape, 1,
                      design_angle_degrees=90, frame="program")
    rough = v.plan(target, v.VProfile("flat", 90, .3, 3, 2),
                   stepover_mm=.7, xy_step_mm=1, margin_mm=.02)
    stock = v.VComposition(target, (rough,))
    tool = v.VProfile("pointed", 90, 0, 3, 2)
    candidate = planar_rest.generate(stock, tool, stepover_mm=.6,
        max_cusp_mm=.3, xy_step_mm=1, margin_mm=.02)
    baselines = tuple(v.plan(target, tool, stepover_mm=.6, xy_step_mm=1,
        margin_mm=.02, fill_pattern=pattern) for pattern in ("raster", "offset"))
    return candidate, baselines


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class PlanarFriezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate, cls.baselines = _frieze()

    def test_located_detail_gain_over_both_baselines_at_equal_controls(self):
        candidate = self.candidate
        self.assertEqual(len(candidate.prior.target.safe.interiors), 1)
        self.assertEqual(candidate.plan.status, "partial")
        self.assertAlmostEqual(candidate.plan.stepover_mm, .6)
        self.assertLess(candidate.guide_deviation_mm, 1e-6)
        self.assertGreater(candidate.omitted_air_length_mm, 40)
        for depth in (.25, .5, .75):
            result = candidate.section(depth)
            self.assertGreater(result["new_removal_mm2"][0], 0)
            self.assertGreater(result["overlap_mm2"][0], 0)
            self.assertEqual(result["residual"].possible_overcut.area, 0)
            for baseline in self.baselines:
                stock = v.VComposition(candidate.prior.target,
                                        candidate.prior.stages+(baseline,))
                reference = v.section_evidence(stock, depth)
                self.assertLess(result["residual"].residual_outer.area,
                                reference.residual_inner.area)
        # No speed advantage is assumed; feature coverage costs more travel.
        travel = lambda plan: sum(math.dist(m.start, m.end) for m in plan.motions)
        self.assertTrue(all(travel(candidate.plan) > travel(b)
                            for b in self.baselines))

    def test_medial_valley_witness_uses_independent_disk_membership(self):
        candidate = self.candidate
        detail = box(9.4, 10.5, 10.6, 12.5)
        cut = v.section_evidence(candidate.plan, .5)
        for baseline in self.baselines:
            prior = v.VComposition(candidate.prior.target,
                                  candidate.prior.stages+(baseline,))
            old = v.section_evidence(prior, .5)
            gain = cut.known_free_inner.difference(old.known_free_outer).intersection(detail)
            self.assertGreater(gain.area, 0)
            point = tuple(gain.representative_point().coords[0])
            # Analytic segment distances and endpoint radii, with no stock
            # Boolean, verify the newly cut point and absent prior removal.
            self.assertGreater(_membership_margins(candidate.plan, point, .5)[0], 0)
            for plan in candidate.prior.stages+(baseline,):
                self.assertLess(_membership_margins(plan, point, .5)[1], 0)

    def test_capped_floor_cusp_and_protected_island(self):
        candidate = self.candidate
        report = candidate.floor_cusp()
        self.assertEqual(report["status"], "bounded")
        self.assertTrue(report["unproved"].is_empty)
        self.assertAlmostEqual(report["depth_mm"], .7)
        self.assertGreater(report["floor"].area, 128)
        # Independent analytic disks witness axial coverage on a grid of the
        # capped floor; the continuous inner-union gate covers between points.
        checked = 0
        for i in range(40):
            for j in range(28):
                point = (i*.5+.25, j*.5+.25)
                if report["floor"].contains(Point(point)):
                    checked += 1
                    self.assertGreater(max(_membership_margins(plan, point, .7)[0]
                        for plan in candidate.prior.stages+(candidate.plan,)), 0)
        self.assertGreater(checked, 400)
        for depth in (0, .5, 1):
            evidence = v.section_evidence(candidate.composition, depth)
            self.assertFalse(evidence.known_free_outer.covers(Point(9, 3.5)))
        # Broad cap retains finite-tip residual, never reported complete.
        self.assertGreater(v.section_report(candidate.composition, 1)[0], 0)

    def test_mutated_cusp_report_cannot_manufacture_a_bound(self):
        for cusp in (float("nan"), float("inf"), 0, -1, True):
            with self.subTest(cusp=cusp), self.assertRaises(ValueError):
                replace(self.candidate, max_cusp_mm=cusp).floor_cusp()


if __name__ == "__main__":
    unittest.main()
