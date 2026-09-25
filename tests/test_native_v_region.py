"""M3 native source, candidate and complete posted-motion gates."""

from pathlib import Path
import json
import tempfile
import unittest

try:
    from cambam_builder.cam_core.v_region import VProfile
    from cambam_builder.integrations.cambam import native_v_region as m3
    from cambam_builder.native.reader import read_cambam_bytes
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class NativeVRegionTests(unittest.TestCase):
    def tools(self):
        return (
            VProfile("pointed", 90, 0, 4, 3),
            VProfile("flat", 90, 0.25, 4, 3),
            VProfile("rounded", 60, 0.5, 4, 3),
        )

    def _synthetic_post(self, directory):
        candidate = directory / "explicit/m3-explicit.cb"
        project = read_cambam_bytes(candidate.read_bytes())
        script = project.list_mops()[0].custom_script
        posted = directory / "synthetic.nc"
        posted.write_text(
            "( m3-explicit synthetic )\n( Post processor: Default )\n"
            "G21 G90 G61 G40\nG0 Z5\nT1 M6\nG17\nM3 S12000\n" +
            script + "\nM5\nM30\n", encoding="utf-8")
        return posted

    def test_polygonal_and_curved_candidates_cover_all_profiles(self):
        corpus = json.loads((Path(__file__).parent / "fixtures" /
                             "rest_vcarve_acceptance.json").read_text(encoding="utf-8"))
        limits = next(row["expected"] for row in corpus["cases"]
                      if row["id"] == "M3_polygonal_and_curved_V_paths")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case in ("letter", "annulus"):
                for tool in self.tools():
                    with self.subTest(case=case, kind=tool.kind):
                        folder = root / f"{case}-{tool.kind}"
                        manifest = m3.build_workflow(folder, case=case, tool=tool)
                        self.assertEqual(manifest["post_status"],
                                         "pending_actual_CamBam_post")
                        self.assertGreater(manifest["move_count"], 100)
                        self.assertGreater(manifest["prior_section_1_mm2"][0],
                                           manifest["section_1_mm2"][1])
                        budget = limits["minimum_prior_to_v_gain_mm2"][
                            "A01" if case == "letter" else "M2_annulus"]
                        self.assertGreater(manifest["prior_section_1_mm2"][0] -
                                           manifest["section_1_mm2"][1], budget)
                        post = self._synthetic_post(folder)
                        report = m3.audit_post(folder / "expected-motion.json", post)
                        self.assertEqual(report["status"],
                                         "bounded_m3_v_post_pass")
                        self.assertGreater(report["path_count"], 2)

    def test_post_deviation_and_stale_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "letter"
            m3.build_workflow(folder)
            post = self._synthetic_post(folder)
            body = post.read_text(encoding="utf-8")
            post.write_text(body.replace("M3 S12000", "M3 S9000", 1),
                            encoding="utf-8")
            self.assertEqual(m3.audit_post(folder / "expected-motion.json", post)
                             ["status"], "deviation")
            prior = folder / "prior.json"
            original_prior = prior.read_bytes()
            prior.write_bytes(original_prior + b"\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                m3.audit_post(folder / "expected-motion.json", post)
            prior.write_bytes(original_prior)
            candidate = folder / "explicit/m3-explicit.cb"
            original_candidate = candidate.read_bytes()
            candidate.write_bytes(original_candidate + b"\n")
            with self.assertRaisesRegex(ValueError, "candidate changed"):
                m3.audit_post(folder / "expected-motion.json", post)
            candidate.write_bytes(original_candidate)
            source = folder / "source.cb"
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                m3.audit_post(folder / "expected-motion.json", post)

    def test_mixed_arc_concave_prior_and_rounded_post(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "mixed-rounded"
            manifest = m3.build_workflow(folder, case="mixed",
                                         tool=self.tools()[2])
            self.assertGreater(manifest["prior_section_1_mm2"][0] -
                               manifest["section_1_mm2"][1], 200)
            report = m3.audit_post(folder / "expected-motion.json",
                                   self._synthetic_post(folder))
            self.assertEqual(report["status"], "bounded_m3_v_post_pass")
            self.assertGreater(report["prior_cut_count"], 0)

    def test_two_xml_round_trips_preserve_curved_paths_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "annulus-rounded"
            m3.build_workflow(folder, case="annulus", tool=self.tools()[2])
            source = (folder / "source.cb").read_bytes()
            plan, start = m3._plan(source, "annulus", self.tools()[2], 2, 1)
            prior = m3._prior_trace(source, plan, start,
                json.loads((folder / "prior.json").read_text(encoding="utf-8")))
            moves = m3._motion_sequence(plan, start)
            for kind in ("preview", "explicit"):
                project = read_cambam_bytes(
                    (folder / kind / f"m3-{kind}.cb").read_bytes())
                first = folder / f"{kind}-round1.cb"
                second = folder / f"{kind}-round2.cb"
                project.save(str(first))
                read_cambam_bytes(first.read_bytes()).save(str(second))
                m3._check_candidate(second.read_bytes(), source, "annulus", kind,
                                    plan, moves, start, self.tools()[2], prior)

    def test_preview_centerline_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "letter"
            m3.build_workflow(folder)
            source = (folder / "source.cb").read_bytes()
            plan, _ = m3._plan(source, "letter", self.tools()[0], 2, 1)
            lines = ["( m3-preview synthetic )", "( Post processor: Default )",
                     "G21 G90 G61 G40", "G0 Z5", "T3 M6", "G17", "M3 S12000"]
            for path in plan.paths:
                x, y, depth = path.points[0]
                lines.append(f"G0 X{x} Y{y} Z5")
                lines.append(f"G1 F60 Z{-depth}")
                for x, y, depth in path.points[1:]:
                    lines.append(f"G1 F300 X{x} Y{y} Z{-depth}")
                lines.append("G0 Z5")
            lines.extend(("M5", "M30"))
            posted = folder / "synthetic-preview.nc"
            posted.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = m3.audit_preview(folder / "expected-motion.json", posted)
            self.assertEqual(report["status"],
                             "m3_v_preview_centerlines_match")
            posted.write_text(posted.read_text(encoding="utf-8").replace(
                "G1 F300", "G1 F300 X0", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                m3.audit_preview(folder / "expected-motion.json", posted)


if __name__ == "__main__":
    unittest.main()
