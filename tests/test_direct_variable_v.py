"""Direct reference G-code from the detached bounded V request."""

import json
import tempfile
import unittest
from pathlib import Path

from cambam_builder.integrations.cambam.native_variable_v import (
    synthetic_setup, synthetic_source,
)
from cambam_builder.integrations.direct_variable_v import (
    audit_program, build_program,
)


CAMBAM_REFERENCE = """( V-variable synthetic reference )
( Post processor: Default )
G21 G90 G61 G40
G0 Z5
T3 M6
G17
M3 S12000
G0 X-10 Y-10
G98
G0 X2 Y2 Z5
G1 F120 X2 Y2 Z1
G1 F60 X2 Y2 Z-1.25
G1 F300 X10 Y2 Z-2.25
G1 F300 X10 Y2 Z1
G0 X-10 Y-10 Z5
G80
G0 Z5
M5
M30
"""


class DirectVariableVTests(unittest.TestCase):
    def test_standalone_and_native_source_emit_same_verified_program(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "native.cb"
            synthetic_source().save(str(source))
            comparison = directory / "CamBam.nc"
            comparison.write_text(CAMBAM_REFERENCE, encoding="utf-8")
            standalone = build_program(directory / "standalone",
                                       comparison_post=comparison)
            native = build_program(directory / "native",
                                   source_path=source, setup=synthetic_setup(),
                                   comparison_post=comparison)
            self.assertEqual(standalone["status"], "bounded_direct_variable_v_pass")
            self.assertEqual(native["program_sha256"],
                             standalone["program_sha256"])
            self.assertEqual(native["item_count"], 9)
            self.assertEqual(native["stock_prefixes"], [["variable-v", 2]])
            self.assertEqual(native["completion"], "partial_target_completion")
            self.assertAlmostEqual(native["section_rest_mm2"]["depth_1.5"],
                                   4.21460291488107)
            text = Path(native["program"]).read_text(encoding="ascii")
            self.assertIn("G1 F120 X2 Y2 Z1\n", text)
            self.assertIn("G1 F300 X10 Y2 Z-2.25\n", text)
            self.assertEqual(text.count("G1 F300 X10"), 2)
            audited = audit_program(directory / "native" / "direct-evidence.json",
                                    comparison_post=comparison)
            self.assertEqual(audited["status"], "bounded_direct_variable_v_pass")

    def test_changed_program_source_or_comparison_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "native.cb"
            synthetic_source().save(str(source))
            comparison = directory / "CamBam.nc"
            comparison.write_text(CAMBAM_REFERENCE, encoding="utf-8")
            folder = directory / "direct"
            build_program(folder, source_path=source, setup=synthetic_setup(),
                          comparison_post=comparison)
            manifest_path = folder / "direct-evidence.json"
            program = folder / "direct-V-variable.nc"
            original = program.read_text(encoding="ascii")
            program.write_text(original.replace("F120", "F200"), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "program changed"):
                audit_program(manifest_path, comparison_post=comparison)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            from hashlib import sha256
            manifest["program_sha256"] = sha256(program.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified rendering"):
                audit_program(manifest_path, comparison_post=comparison)
            program.write_text(original, encoding="ascii")
            manifest["program_sha256"] = sha256(program.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            source_copy = folder / "source.cb"
            source_copy.write_bytes(source_copy.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "source or setup changed"):
                audit_program(manifest_path, comparison_post=comparison)
            source_copy.write_bytes(source.read_bytes())
            comparison.write_text(CAMBAM_REFERENCE.replace("F300 X10", "F200 X10"),
                                  encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "comparison CamBam post changed"):
                audit_program(manifest_path, comparison_post=comparison)


if __name__ == "__main__":
    unittest.main()
