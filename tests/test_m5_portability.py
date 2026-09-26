"""Grbl v1.1 second dialect and mixed manual/automatic transition gate."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from cambam_builder.integrations import m4_curved_workflow as m4
    from cambam_builder.integrations import m5_portability as m5
    from cambam_builder.integrations.cambam import native_curved_rest as m2
    from cambam_builder.integrations.grbl_m5_reader import decode_job
    from cambam_builder.native.core import Vertex
    from cambam_builder.native.reader import read_cambam_bytes
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class M5PortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        source = root / "source.cb"
        m2.synthetic_source("annulus").save(str(source))
        project = read_cambam_bytes(source.read_bytes())
        hole = project.get_primitive("curved-finish").hole_curves[0]
        hole.vertices = [Vertex(v.x * 1.05, v.y * 1.05, v.z,
                                bulge=v.bulge) for v in hole.vertices]
        edited = root / "edited.cb"
        project.save(str(edited))
        cls.comparison = root / "comparison"
        m4.build_comparison(cls.comparison, source_path=edited,
                            synthetic_prior=True)
        cls.bundle = root / "portability"
        cls.result = m5.build_bundle(
            cls.bundle, m4_manifest=cls.comparison / "comparison.json")
        cls.plan, cls.prior, cls.start = m4.load_selected_plan(
            cls.comparison / "comparison.json", "raster")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_manual_and_mixed_job_replay(self):
        self.assertEqual(self.result["status"], "bounded_m5_portability_pass")
        for role in ("manual", "mixed"):
            report = self.result[role]
            self.assertEqual((report["motion_equivalence"]["t1_moves"],
                              report["motion_equivalence"]["t3_moves"]),
                             (61, 383))
            self.assertEqual(report["stock_access_residual"]["t1_cuts"], 30)
            self.assertAlmostEqual(
                report["stock_access_residual"]["final_section_1_mm2"][1],
                1.526323, places=5)
            self.assertAlmostEqual(
                report["stock_access_residual"]["final_volume_mm3"][1],
                42.675341, places=5)
            self.assertEqual(report["runtime_parity"]["status"],
                             "not_evaluated")
        self.assertEqual(self.result["mixed"]["motion_equivalence"]["return_t1_moves"], 2)
        self.assertEqual(self.result["mixed"]["transition_evidence"]["changer_travel_segments"], 3)
        self.assertEqual(m5.audit_bundle(self.bundle / "handoff.json"), self.result)
        manual = (self.bundle / "manual.nc").read_bytes()
        mixed = (self.bundle / "mixed.nc").read_bytes()
        self.assertEqual(manual.count(b"M0\n"), 1)
        self.assertEqual(mixed.count(b"M0\n"), 2)
        self.assertNotIn(b"M6", manual + mixed)
        self.assertEqual(decode_job(mixed, initial_tip=self.start).length_offsets_mm,
                         (2, 3, 2))

    def test_unknown_stop_macro_offset_and_tool_are_rejected(self):
        manual = (self.bundle / "manual.nc").read_bytes()
        mixed = (self.bundle / "mixed.nc").read_bytes()
        cases = (
            (manual.replace(b"M0\n", b"M6\n", 1), False),
            (manual.replace(b"M0\n", b"M98 P1\n", 1), False),
            (manual.replace(b"M0\n", b"", 1), False),
            (mixed.replace(b"G43.1 Z3\n", b"G43.1 Z4\n", 1), True),
            (mixed.replace(b"( STAGE T3 )", b"( STAGE T1 )", 1), True),
        )
        for data, is_mixed in cases:
            with self.subTest(data=data[:35]):
                with self.assertRaises(ValueError):
                    m5._audit_program(data, self.plan, self.prior, self.start,
                                      mixed=is_mixed,
                                      effect=(self.bundle / "mixed-changer.json").read_bytes()
                                      if is_mixed else None)

    def test_nonidentity_datum_rejects_unshifted_or_double_shifted_path(self):
        correct = (self.bundle / "manual.nc").read_bytes()
        # The source CAM path is at +4.5; changing an emitted endpoint to
        # either CAM Z or an extra -4.5 work shift violates the declared map.
        for replacement in (b"Z2.5\n", b"Z-6.5\n"):
            bad = correct.replace(b"Z-2\n", replacement, 1)
            self.assertTrue(bad != correct)
            with self.assertRaisesRegex(ValueError, "differs"):
                m5._audit_program(bad, self.plan, self.prior, self.start,
                                  mixed=False)

    def test_changer_effect_and_handoff_are_fail_closed(self):
        mixed = (self.bundle / "mixed.nc").read_bytes()
        effect = (self.bundle / "mixed-changer.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "missing"):
            m5._audit_program(mixed, self.plan, self.prior, self.start,
                              mixed=True)
        changed = json.loads(effect)
        changed["travel_work_tip_xyz_mm"][1][2] = -1
        with self.assertRaisesRegex(ValueError, "effect"):
            m5._audit_program(mixed, self.plan, self.prior, self.start,
                              mixed=True, effect=json.dumps(changed).encode())
        changed = json.loads(effect)
        changed["length_offset_after_mm"] = 0
        with self.assertRaisesRegex(ValueError, "effect"):
            m5._audit_program(mixed, self.plan, self.prior, self.start,
                              mixed=True, effect=json.dumps(changed).encode())
        with self.assertRaisesRegex(ValueError, "encoding"):
            m5._audit_program(mixed, self.plan, self.prior, self.start,
                              mixed=True, effect=json.dumps(json.loads(effect)).encode())
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "bundle"
            shutil.copytree(self.bundle, copy)
            manifest_path = copy / "handoff.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["transitions"]["mixed"][1]["offset_method"] = "unknown"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "transitions"):
                m5.audit_bundle(manifest_path)
            manifest["transitions"]["mixed"][1]["offset_method"] = "fixed_origin_table"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (copy / "mixed-changer.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                m5.audit_bundle(manifest_path)
            (copy / "mixed-changer.json").write_bytes(effect)
            program = copy / "mixed.nc"
            altered = program.read_bytes().replace(b"G43.1 Z3\n",
                                                   b"G43.1 Z4\n", 1)
            program.write_bytes(altered)
            manifest["programs"][1]["sha256"] = m5._sha(altered)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "length state"):
                m5.audit_bundle(manifest_path)


if __name__ == "__main__":
    unittest.main()
