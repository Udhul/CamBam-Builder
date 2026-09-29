"""M4 edited annulus: offset/raster and complete output evidence."""

import tempfile
import unittest
from pathlib import Path

try:
    from cambam_builder.cam_core import v_region
    from cambam_builder.integrations import m4_curved_workflow as m4
    from cambam_builder.integrations.cambam import native_curved_rest as m2
    from cambam_builder.native.core import Vertex
    from cambam_builder.native.reader import read_cambam_bytes
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class M4CurvedWorkflowTests(unittest.TestCase):
    def _edited_source(self, root):
        original = root / "original.cb"
        m2.synthetic_source("annulus").save(str(original))
        project = read_cambam_bytes(original.read_bytes())
        region = project.get_primitive("curved-finish")
        hole = region.hole_curves[0]
        hole.vertices = [Vertex(v.x * 1.05, v.y * 1.05, v.z,
                                bulge=v.bulge) for v in hole.vertices]
        region.validate_geometry()
        edited = root / "edited.cb"
        project.save(str(edited))
        self.assertNotEqual(m4._semantic(original.read_bytes()),
                            m4._semantic(edited.read_bytes()))
        return edited

    def _post(self, folder):
        candidate = folder / "explicit/m3-explicit.cb"
        script = read_cambam_bytes(candidate.read_bytes()).list_mops()[
            0].custom_script
        posted = folder / "synthetic.nc"
        posted.write_text(
            "( m3-explicit synthetic )\n( Post processor: Default )\n"
            "G21 G90 G61 G40\nG0 Z5\nT1 M6\nG17\nM3 S12000\n" +
            script + "\nM5\nM30\n", encoding="utf-8")
        return posted

    def _preview(self, folder, fill):
        source = (folder / "source.cb").read_bytes()
        plan, _ = m4.m3._plan(source, "annulus", m4.ROUNDED,
                              2, 1, fill)
        lines = ["( m3-preview synthetic )", "( Post processor: Default )",
                 "G21 G90 G61 G40", "G0 Z5", "T3 M6", "G17", "M3 S12000"]
        for path in plan.paths:
            x, y, depth = path.points[0]
            lines.extend((f"G0 X{x} Y{y} Z5", f"G1 F60 Z{-depth}"))
            lines.extend(f"G1 F300 X{x} Y{y} Z{-depth}"
                         for x, y, depth in path.points[1:])
            lines.append("G0 Z5")
        lines.extend(("M5", "M30"))
        posted = folder / "synthetic-preview.nc"
        posted.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return posted

    def test_edited_source_two_fills_direct_and_native_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            edited = self._edited_source(root)
            folder = root / "comparison"
            manifest = m4.build_comparison(folder, source_path=edited,
                                           synthetic_prior=True)
            self.assertTrue(manifest["synthetic_prior"])
            self.assertEqual(m4.audit_endmill_direct(
                folder / "comparison.json")["status"],
                "bounded_m4_endmill_direct_pass")
            self.assertNotEqual(manifest["jobs"]["raster"]["plan_fingerprint"],
                                manifest["jobs"]["offset"]["plan_fingerprint"])
            for fill in ("raster", "offset"):
                report = m4.audit_direct(folder / "comparison.json", fill)
                self.assertEqual(report["status"], "bounded_m4_direct_pass")
                self.assertLess(report["section_1_mm2"][1], 2)
                self.assertLess(report["volume_mm3"][1], 80)
                source = (folder / fill / "source.cb").read_bytes()
                plan, _ = m4.m3._plan(source, "annulus", m4.ROUNDED,
                                      2, 1, fill)
                self.assertEqual(plan.fill_pattern, fill)
                self.assertEqual(v_region.verify(plan), plan)
                self.assertTrue(any(path.role == "fill" for path in plan.paths))
            missing = m4.audit_comparison(folder / "comparison.json")
            self.assertEqual((missing["status"], missing["chosen"]),
                             ("partial", "endmill_only"))
            raster = self._post(folder / "raster")
            offset = self._post(folder / "offset")
            raster_preview = self._preview(folder / "raster", "raster")
            offset_preview = self._preview(folder / "offset", "offset")
            result = m4.audit_comparison(folder / "comparison.json",
                raster_post=raster, offset_post=offset,
                raster_preview=raster_preview, offset_preview=offset_preview)
            self.assertEqual(result["status"], "selected")
            self.assertIn(result["chosen"], ("rounded_raster",
                                              "rounded_offset"))
            self.assertEqual(tuple(row["status"] for row in result[
                "assessments"]), ("partial", "feasible", "feasible"))
            manual = m4.audit_comparison(folder / "comparison.json",
                raster_post=raster, offset_post=offset,
                raster_preview=raster_preview, offset_preview=offset_preview,
                manual_choice="endmill_only")
            self.assertEqual((manual["status"], manual["chosen"]),
                             ("partial", "endmill_only"))
            project = read_cambam_bytes(edited.read_bytes())
            project.project_name += " cosmetic title"
            cosmetic = root / "cosmetic.cb"
            project.save(str(cosmetic))
            self.assertEqual(m4.audit_direct(folder / "comparison.json",
                "offset", current_source_path=cosmetic)["status"],
                "bounded_m4_direct_pass")
            region = project.get_primitive("curved-finish")
            region.hole_curves[0].vertices[0].x += 0.1
            region.validate_geometry()
            changed = root / "changed.cb"
            project.save(str(changed))
            with self.assertRaisesRegex(ValueError, "invalidated"):
                m4.audit_direct(folder / "comparison.json", "offset",
                                current_source_path=changed)
            direct = folder / "offset/direct-reference.nc"
            direct.write_bytes(direct.read_bytes().replace(b"S12000", b"S9000", 1))
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                m4.audit_direct(folder / "comparison.json", "offset")
            fallback = m4.audit_comparison(folder / "comparison.json",
                raster_post=raster, offset_post=offset,
                raster_preview=raster_preview, offset_preview=offset_preview)
            self.assertEqual(fallback["chosen"], "rounded_raster")
            self.assertEqual(fallback["reports"]["offset"]["direct"][
                "status"], "unverified")


if __name__ == "__main__":
    unittest.main()
