"""M5 split UCCNC output, strict decoding and conditional stock evidence."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    from cambam_builder.cam_core import v_region
    from cambam_builder.integrations import m4_curved_workflow as m4
    from cambam_builder.integrations import uccnc_m5 as m5
    from cambam_builder.integrations.uccnc_reader import decode_program
    from cambam_builder.integrations.cambam import native_curved_rest as m2
    from cambam_builder.native.core import Vertex
    from cambam_builder.native.reader import read_cambam_bytes
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class UCCNCM5Tests(unittest.TestCase):
    def _bundle(self, root):
        original = root / "original.cb"
        m2.synthetic_source("annulus").save(str(original))
        project = read_cambam_bytes(original.read_bytes())
        hole = project.get_primitive("curved-finish").hole_curves[0]
        hole.vertices = [Vertex(v.x * 1.05, v.y * 1.05, v.z,
                                bulge=v.bulge) for v in hole.vertices]
        edited = root / "edited.cb"
        project.save(str(edited))
        comparison = root / "comparison"
        m4.build_comparison(comparison, source_path=edited,
                            synthetic_prior=True)
        bundle = root / "uccnc"
        result = m5.build_bundle(bundle,
                                 m4_manifest=comparison / "comparison.json")
        return bundle, result

    def test_split_files_decode_and_replay_both_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, result = self._bundle(Path(temp))
            self.assertEqual(result["status"], "bounded_m5_uccnc_split_pass")
            self.assertEqual(result["motion_equivalence"]["t1_moves"] +
                             result["motion_equivalence"]["t3_moves"], 444)
            self.assertEqual(result["stock_access_residual"]["t1_cuts"], 30)
            self.assertLess(
                result["stock_access_residual"]["final_section_1_mm2"][1], 2)
            self.assertLess(
                result["stock_access_residual"]["final_volume_mm3"][1], 80)
            self.assertEqual(result["runtime_parity"]["status"],
                             "not_evaluated")
            self.assertEqual(m5.audit_bundle(bundle / "handoff.json"), result)
            self.assertNotIn(b"M6", (bundle / "T1.nc").read_bytes())
            self.assertNotIn(b"M6", (bundle / "T3.nc").read_bytes())

    def test_altered_second_file_and_stale_handoff_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, _ = self._bundle(Path(temp))
            manifest_path = bundle / "handoff.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["handoff"]["order"] = ["T3", "T1"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "handoff"):
                m5.audit_bundle(manifest_path)
            manifest["handoff"]["order"] = ["T1", "T3"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            program = bundle / "T3.nc"
            original = program.read_bytes()
            program.write_bytes(original.replace(b"X", b"X0", 1))
            with self.assertRaisesRegex(ValueError, "altered"):
                m5.audit_bundle(manifest_path)
            # Rehashing edited bytes cannot promote a changed endpoint into
            # valid evidence; the independent decode still compares the plan.
            data = original.replace(b"Z5\n", b"Z4.99\n", 1)
            self.assertNotEqual(data, original)
            program.write_bytes(data)
            manifest["programs"][1]["sha256"] = m5._sha(data)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                m5.audit_bundle(manifest_path)
            program.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                m5.audit_bundle(manifest_path)

    def test_unknown_commands_and_bare_tool_selection_fail(self):
        moves = (
            v_region.VMotion("rapid", (-1, -1, 5), (0, 0, 5)),
            v_region.VMotion("entry", (0, 0, 5), (0, 0, -1)),
            v_region.VMotion("retract", (0, 0, -1), (0, 0, 5)),
            v_region.VMotion("rapid", (0, 0, 5), (-1, -1, 5)),
        )
        program = m5.render_program(moves)
        decoded = decode_program(program, initial_tip=(-1, -1, 5))
        self.assertEqual(len(decoded.moves), 4)
        for command in (b"T3", b"M6", b"M0", b"G43 H3", b"G53 G0 Z5"):
            with self.subTest(command=command):
                modified = program.replace(b"M5\n", command + b"\nM5\n")
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    decode_program(modified, initial_tip=(-1, -1, 5))

    def test_explicit_datum_map_and_rounding(self):
        # Intended surface is programmed +4.5; work surface is touched to 0.
        moves = (
            v_region.VMotion("rapid", (-1, -1, 9.5),
                             (0.00004, 0, 9.5)),
            v_region.VMotion("entry", (0.00004, 0, 9.5),
                             (0.00004, 0, 4.5)),
            v_region.VMotion("retract", (0.00004, 0, 4.5),
                             (0.00004, 0, 9.5)),
            v_region.VMotion("rapid", (0.00004, 0, 9.5),
                             (-1, -1, 9.5)),
        )
        translated = decode_program(
            m5.render_program(moves, translation_xyz_mm=(0, 0, -4.5)),
            initial_tip=(-1, -1, 5))
        actual = m5.compare_motion(
            translated, moves, initial_cam_tip=(-1, -1, 9.5),
            translation_xyz_mm=(0, 0, -4.5))
        self.assertEqual(actual[1].end[2], 4.5)
        self.assertEqual(actual[0].end[0], 0)
        unchanged = decode_program(m5.render_program(moves),
                                   initial_tip=(-1, -1, 9.5))
        with self.assertRaisesRegex(ValueError, "initial tip"):
            m5.compare_motion(unchanged, moves,
                              initial_cam_tip=(-1, -1, 9.5),
                              translation_xyz_mm=(0, 0, -4.5))
        changed = m5.render_program(
            moves, translation_xyz_mm=(0, 0, -4.5)).replace(b"Z0\n", b"Z0.01\n", 1)
        with self.assertRaisesRegex(ValueError, "differs"):
            m5.compare_motion(
                decode_program(changed, initial_tip=(-1, -1, 5)),
                moves, initial_cam_tip=(-1, -1, 9.5),
                translation_xyz_mm=(0, 0, -4.5))


if __name__ == "__main__":
    unittest.main()
