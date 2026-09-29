"""Grbl v1.1 second dialect and mixed manual/automatic transition gate."""

import json
from dataclasses import replace
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
            stock = report["stock_access_residual"]
            # Portability preserves material evidence for the same physical
            # paths; copied GEOS output digits are not a geometric oracle.
            # Analytic enclosure correctness is exercised in test_v_region.
            for before, after, budget in (
                    ("prior_section_1_mm2", "final_section_1_mm2", 2),
                    ("prior_volume_mm3", "final_volume_mm3", 80)):
                self.assertGreaterEqual(stock[after][0], 0)
                self.assertLessEqual(stock[after][0], stock[after][1])
                self.assertLess(stock[after][1], stock[before][0])
                self.assertLess(stock[after][1], budget)
                self.assertEqual(stock[after], self.result["manual"][
                    "stock_access_residual"][after])
            self.assertLess(stock["final_section_1_mm2"][2], 1e-7)
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
        decoded = decode_job(mixed, initial_tip=self.start)
        self.assertEqual([s.transition_moves[0].start[2] for s in decoded.stages],
                         [self.start[2] - 2, self.start[2] - 1, self.start[2] + 1])
        self.assertTrue(all(s.transition_moves[0].end == self.start
                            for s in decoded.stages))
        self.assertEqual(self.result["mixed"]["transition_evidence"][
            "offset_compensation_segments"], 3)

    def test_compensation_is_mandatory_decoded_motion_with_stock_clearance(self):
        program = (self.bundle / "mixed.nc").read_bytes()
        lines = program.split(b"\n")
        index = lines.index(b"G43.1 Z2") + 1
        for altered in (lines[:index] + lines[index + 1:],
                        lines[:index] + [lines[index].replace(b"G0", b"G1 F60")]
                        + lines[index + 1:],
                        lines[:index] + [lines[index].replace(b"Z5", b"Z4")]
                        + lines[index + 1:]):
            with self.assertRaisesRegex(ValueError, "compensation"):
                decode_job(b"\n".join(altered), initial_tip=self.start)
        decoded = decode_job(program, initial_tip=self.start)
        self.assertEqual(m5._audit_offset_travel(decoded, self.start), 3)
        # A syntactically valid compensated return still fails the stock gate
        # if the changed work coordinate reaches the stock top.
        lowered = decode_job(program.replace(b"G43.1 Z2", b"G43.1 Z5", 1),
                             initial_tip=self.start)
        with self.assertRaisesRegex(ValueError, "into stock"):
            m5._audit_offset_travel(lowered, self.start)
        first = decoded.stages[0]
        with self.assertRaisesRegex(ValueError, "differs"):
            m5._audit_offset_travel(replace(decoded, stages=(
                replace(first, transition_moves=()), *decoded.stages[1:])),
                self.start)

    def test_reader_retains_actual_boundary_position_and_requires_lf(self):
        program = (self.bundle / "manual.nc").read_bytes()
        decoded = decode_job(program, initial_tip=self.start)
        self.assertEqual(decoded.stages[1].moves[0].start,
                         decoded.stages[0].end_position)
        boundary = program.index(b"M5\nM0")
        prefix = program[:boundary]
        move_start = prefix.rfind(b"G0 ")
        changed = (prefix[:move_start] + prefix[move_start:].replace(b"Z5", b"Z4")
                   + program[boundary:])
        shifted = decode_job(changed, initial_tip=self.start)
        self.assertEqual(shifted.stages[1].moves[0].start[2], 4)
        with self.assertRaisesRegex(ValueError, "safe return"):
            m5._audit_program(changed, self.plan, self.prior, self.start,
                              mixed=False)
        for separator in (b"\x0b", b"\x0c", b"\x1c", b"\x1d", b"\x1e"):
            for altered in (program.replace(b"G21\n", b"G21" + separator),
                            program.replace(b"M5\n", b"M5" + separator)):
                with self.assertRaises(ValueError):
                    decode_job(altered, initial_tip=self.start)

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
