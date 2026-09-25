"""M1 supplied-stock, original-boundary and emitted-output regression checks."""

import json
import math
from dataclasses import replace
from pathlib import Path
import re
import tempfile
import unittest

try:
    from shapely.geometry import Polygon
    from cambam_builder.cam_core import polygon_rest, replay
    from cambam_builder.integrations.cambam import native_polygon_rest as m1
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class PolygonRestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.manifest_path = cls.directory / "expected-motion.json"
        cls.manifest = m1.build_workflow(cls.directory)
        source = (cls.directory / "source.cb").read_bytes()
        prior = json.loads((cls.directory / "prior.json").read_text(encoding="utf-8"))
        cls.result = m1.plan(source, prior)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_independent_area_limits_and_original_boundary(self):
        result = self.result
        fixture = json.loads(Path("tests/fixtures/rest_vcarve_acceptance.json").read_text(
            encoding="utf-8"))
        case = next(row for row in fixture["cases"]
                    if row["id"] == "A01_general_region_composition")
        limits = case["expected"]["m1"]
        region = Polygon(m1.SHELL, (m1.HOLE,))
        self.assertEqual(region.area, 1532)
        signed = sum(a[0] * b[1] - a[1] * b[0]
                     for a, b in zip(m1.SHELL, m1.SHELL[1:] + m1.SHELL[:1]))
        winding = 1 if signed > 0 else -1
        analytic_corner_rest = 0.0
        for before, at, after in zip(m1.SHELL[-1:] + m1.SHELL[:-1],
                                     m1.SHELL, m1.SHELL[1:] + m1.SHELL[:1]):
            incoming = (at[0] - before[0], at[1] - before[1])
            outgoing = (after[0] - at[0], after[1] - at[1])
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            if cross * winding <= 0:
                continue
            u = (before[0] - at[0], before[1] - at[1])
            v = (after[0] - at[0], after[1] - at[1])
            angle = math.acos((u[0] * v[0] + u[1] * v[1]) /
                              (math.hypot(*u) * math.hypot(*v)))
            analytic_corner_rest += 1 / math.tan(angle / 2) - (math.pi - angle) / 2
        self.assertAlmostEqual(analytic_corner_rest,
                               limits["analytic_finite_cleanup_rest_lower_mm2"],
                               delta=1e-9)
        self.assertEqual(result.stock.prefixes[0][0], "prior")
        self.assertEqual(result.stock.prefixes[1][0], "cleanup")
        self.assertEqual(len(m1._preview_paths(result.trace)), 2)
        self.assertGreater(len(result.stock.cuts), len(result.prior_stock.cuts))
        self.assertTrue(result.pure_rest_contains(0, 59.75, 1))
        self.assertFalse(result.residual_contains(0, 59.75, 1))
        self.assertFalse(result.target.contains(0, 35, 1))
        for depth in limits["section_depths_mm"]:
            rough_lo, rough_hi = result.pure_rest_area(depth)
            final_lo, final_hi = result.residual_area(depth)
            self.assertGreater(rough_lo, limits["original_boundary_allowance_band_mm2_approx"])
            self.assertLess(rough_hi, limits["rough_rest_upper_mm2"])
            self.assertGreaterEqual(final_lo + 1e-5,
                                    limits["analytic_finite_cleanup_rest_lower_mm2"])
            self.assertLess(final_hi, limits["final_rest_upper_mm2"])
            self.assertGreater(rough_lo - final_hi,
                               limits["minimum_cleanup_gain_mm2"])
            self.assertLess(final_hi - final_lo,
                            limits["area_interval_width_upper_mm2"])
        for cut in result.stock.cuts:
            line = polygon_rest._line(cut.a, cut.b)
            self.assertTrue(region.covers(line))
            self.assertGreaterEqual(line.distance(region.boundary) + 1e-8,
                                    cut.tool.radius + (0.5 if cut.operation == "prior" else 0))
        self.assertLessEqual(result.protected_overcut_upper_area(1), 1e-9)
        self.assertLessEqual(result.residual_outside_ideal_envelope_area(1), 1e-9)
        rough_volume = result.pure_rest_volume()
        final_volume = result.residual_volume()
        for volume, section in ((rough_volume, result.pure_rest_area(1)),
                                (final_volume, result.residual_area(1))):
            self.assertAlmostEqual(volume[0], 8 * section[0], delta=1e-7)
            self.assertAlmostEqual(volume[1], 8 * section[1], delta=1e-7)
        self.assertGreater(rough_volume[0] - final_volume[1], 800)

    def test_two_ordered_prior_operations_contribute_to_rest_stock(self):
        first = self.result.prior_trace
        second = replace(first.operations[0], name="prior_second")
        halfway = len(first.items) // 2
        items = tuple(replace(item, operation="prior_second")
                      if type(item) is replay.Motion and index >= halfway else item
                      for index, item in enumerate(first.items))
        supplied = replace(first, operations=first.operations + (second,), items=items)
        result = polygon_rest.generate(
            supplied, replay.ToolProfile("T2", "cylinder", 1, 10),
            expected_source=supplied.source_fingerprint,
            expected_motion=supplied.motion_fingerprint)
        self.assertEqual(tuple(name for name, _ in result.stock.prefixes),
                         ("prior", "prior_second", "cleanup"))
        self.assertEqual(result.pure_rest_area(1), self.result.pure_rest_area(1))
        self.assertEqual(result.pure_rest_volume(), self.result.pure_rest_volume())

    def test_source_motion_and_narrow_access_rejections(self):
        result = self.result
        with self.assertRaisesRegex(ValueError, "stale"):
            polygon_rest.generate(result.prior_trace,
                                  replay.ToolProfile("T2", "cylinder", 1, 10),
                                  expected_source="edited",
                                  expected_motion=result.prior_trace.motion_fingerprint)
        fixture = json.loads(Path("tests/fixtures/rest_vcarve_acceptance.json").read_text(
            encoding="utf-8"))
        case = next(row for row in fixture["cases"]
                    if row["id"] == "M1_narrow_access_rejection")
        shell = tuple(tuple(p) for p in case["input"]["shell"])
        self.assertAlmostEqual(Polygon(shell).area, case["expected"]["area_mm2"])
        target = replay.Target("narrow", (0, 0, 24, 10), 3,
                               region_shell=shell)
        tool = replay.ToolProfile("T2", "cylinder", 1, 6)
        crossing = replay.Sweep("cleanup", tool, (5, 5), (19, 5), -3)
        with self.assertRaisesRegex(ValueError, "original polygonal Region"):
            replay._safe_cut(crossing, target)
        self.assertAlmostEqual(case["expected"]["low_crossing_protected_overcut_mm2"],
                               4 * (2 - case["expected"]["throat_width_mm"]))

    def test_constructed_post_checks_full_sequence_and_deviation(self):
        # This synthetic file exercises the audit only; it is not CamBam output.
        candidate = self.directory / "explicit" / "m1-explicit.cb"
        lines = ["( Made using CamBam - constructed test only )",
                 f"( {candidate.stem} constructed )", "( Post processor: Default )",
                 "G21 G90 G61 G40", "G0 Z5", "T1 M6",
                 "( M1 T1 rough plus T2 cleanup literal motion )",
                 "G17", "M3 S12000"]
        lines.extend(m1._script(self.result.trace).splitlines())
        lines.extend(("M5", "M30"))
        posted = self.directory / "constructed-explicit.nc"
        posted.write_text("\n".join(lines) + "\n", encoding="utf-8")
        audit = m1.audit_explicit_post(self.manifest_path, posted)
        self.assertEqual(audit["status"], "bounded_m1_explicit_post_pass")
        self.assertEqual(audit["item_count"], len(self.result.trace.items))
        text = posted.read_text(encoding="utf-8")
        first = m1._script(self.result.trace).splitlines()[0]
        changed = re.sub(r"X[+-]?\d+(?:\.\d+)?", "X0", first, count=1)
        self.assertNotEqual(changed, first)
        posted.write_text(text.replace(first, changed, 1), encoding="utf-8")
        bad = m1.audit_explicit_post(self.manifest_path, posted)
        self.assertEqual(bad["status"], "deviation")

    def test_native_gate_rejects_low_rapid_from_constructed_post(self):
        # Synthetic failure only; the actual Pocket candidate still needs posting.
        lines = ["( m1-native constructed )", "( Post processor: Default )",
                 "G21 G90 G61 G40", "G0 Z5",
                 "( NATIVE T1 letter Pocket )", "T1 M6", "G17", "M3 S12000",
                 "G0 X0 Y0 Z-1", "M5", "G0 Z5", "G0 X-30 Y-10",
                 "( NATIVE T2 letter Pocket )", "T2 M6", "M3 S12000",
                 "G0 X0 Y0 Z-1", "M5", "M30"]
        posted = self.directory / "constructed-native-failure.nc"
        posted.write_text("\n".join(lines) + "\n", encoding="utf-8")
        audit = m1.audit_native_post(self.manifest_path, posted)
        self.assertEqual(audit["status"], "native_motion_gate_failed")
        self.assertTrue(any("low rapid" in finding
                            for finding in audit["findings"]))


if __name__ == "__main__":
    unittest.main()
