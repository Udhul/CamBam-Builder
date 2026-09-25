"""M2 curved Region source, conservative cleanup and literal post checks."""

import json
import math
from pathlib import Path
import tempfile
import unittest

try:
    from cambam_builder.cam_core import curved_region, replay
    from cambam_builder.integrations.cambam import native_curved_rest as m2
    from cambam_builder.native.reader import read_cambam_bytes
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class CurvedApproximationTests(unittest.TestCase):
    def test_subresolution_bulge_fails_instead_of_losing_arc(self):
        shell = ((0, 0, 1e-10), (10, 0, 0), (10, 10, 0), (0, 10, 0))
        with self.assertRaisesRegex(ValueError, "bulge below supported resolution"):
            curved_region.approximate(shell)


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class CurvedRestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name) / "annulus"
        cls.manifest = m2.build_workflow(cls.directory)
        source = (cls.directory / "source.cb").read_bytes()
        prior = json.loads((cls.directory / "prior.json").read_text(encoding="utf-8"))
        cls.result = m2.plan(source, prior)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_annulus_analytic_source_and_bounded_stock(self):
        result = self.result
        bounds = self.manifest
        self.assertAlmostEqual(bounds["analytic_area_mm2"], 77 * math.pi,
                               delta=1e-7)
        self.assertLess(bounds["chord_area_error_mm2"], 0.05)
        self.assertLessEqual(bounds["sagitta_mm"], 0.001)
        self.assertEqual(tuple(name for name, _ in result.planned.stock.prefixes),
                         ("prior", "cleanup"))
        self.assertEqual(len(m2._preview_paths(result.planned.trace)), 2)
        self.assertGreater(len(result.planned.stock.cuts),
                           len(result.planned.prior_stock.cuts))
        for depth in (1, 3):
            rough = result.rest_area(depth, final=False)
            final = result.rest_area(depth, final=True)
            self.assertLess(rough[1], 18.2)
            self.assertLess(final[1], 0.2)
            self.assertGreater(rough[0] - final[1], 17.7)
            self.assertLess(final[1] - final[0], 0.15)
            self.assertLess(result.protected_overcut_upper_area(depth), 1e-8)
        self.assertLess(result.rest_volume(final=True)[1], 0.8)

    def test_original_arc_identity_and_source_freshness(self):
        source = (self.directory / "source.cb").read_bytes()
        original = read_cambam_bytes(source)
        region = original.get_primitive("curved-finish")
        self.assertTrue(any(abs(row[3]) > 0 for row in
                            region.get_absolute_coordinates_xyz()["outer_curve"]))
        preview = read_cambam_bytes(
            (self.directory / "preview/m2-preview.cb").read_bytes())
        self.assertEqual(preview.get_primitive("curved-finish").internal_id,
                         region.internal_id)
        supplied = json.loads((self.directory / "prior.json").read_text(
            encoding="utf-8"))
        supplied["source_sha256"] = "edited"
        with self.assertRaisesRegex(ValueError, "stale"):
            m2.plan(source, supplied)
        with self.assertRaisesRegex(ValueError, "requires supplied prior"):
            m2.build_workflow(Path(self.temp.name) / "missing-prior",
                              source_path=self.directory / "source.cb")

    def test_curved_narrow_access_rejected(self):
        bulge = math.tan(math.pi / 8)
        outer = tuple((3 * math.cos(a), 3 * math.sin(a), bulge)
                      for a in (0, math.pi/2, math.pi, 3*math.pi/2))
        hole = tuple((2.2 * math.cos(a), 2.2 * math.sin(a), -bulge)
                     for a in (0, -math.pi/2, -math.pi, -3*math.pi/2))
        approx = curved_region.approximate(outer, (hole,))
        target = approx.target("narrow-annulus", 4)
        tool = replay.ToolProfile("T2", "cylinder", 0.5, 8)
        with self.assertRaisesRegex(ValueError, "original polygonal Region"):
            replay._safe_cut(replay.Sweep("cleanup", tool, (2.6, 0),
                                          (2.6, 0.1), -2), target)

    def test_mixed_and_reflected_frame(self):
        angle = 4 * math.atan(0.17)
        radius = 32 / (2 * math.sin(angle / 2))
        analytic_area = (32 * 20 - 10 * 6 +
                         radius * radius * (angle - math.sin(angle)) / 2 -
                         math.pi * 1.5 ** 2)
        for case in ("mixed", "mixed_reflected"):
            source_project = m2.synthetic_source(case)
            with tempfile.TemporaryDirectory() as folder:
                source_path = Path(folder) / "source.cb"
                source_project.save(str(source_path))
                source = source_path.read_bytes()
                result = m2.plan(source, m2.synthetic_prior(source))
            self.assertAlmostEqual(result.approximation.analytic_area_mm2,
                                   analytic_area, delta=1e-6)
            self.assertLess(result.rest_area(1, final=False)[1], 35.5)
            self.assertLess(result.rest_area(1, final=True)[1], 1)
            self.assertLess(result.protected_overcut_upper_area(1), 1e-8)
            if case == "mixed_reflected":
                self.assertGreater(result.approximation.safe.bounds[0], 20)

    def test_constructed_post_and_tamper(self):
        # Parser regression only; actual CamBam output remains a separate gate.
        candidate = self.directory / "explicit/m2-explicit.cb"
        lines = ["( Made using CamBam - constructed test only )",
                 f"( {candidate.stem} constructed )", "( Post processor: Default )",
                 "G21 G90 G61 G40", "G0 Z5", "T1 M6",
                 "( M2 T1 rough plus T2 cleanup literal motion )",
                 "G17", "M3 S12000"]
        lines.extend(m2._script(self.result.planned.trace).splitlines())
        lines.extend(("M5", "M30"))
        posted = self.directory / "constructed.nc"
        posted.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = m2.audit_post(self.directory / "expected-motion.json", posted)
        self.assertEqual(report["status"], "bounded_m2_curved_post_pass")
        self.assertEqual(report["item_count"], len(self.result.planned.trace.items))
        posted.write_text(posted.read_text(encoding="utf-8").replace(
            "G1 F60", "G1 F61", 1), encoding="utf-8")
        self.assertEqual(m2.audit_post(self.directory / "expected-motion.json",
                                       posted)["status"], "deviation")


if __name__ == "__main__":
    unittest.main()
