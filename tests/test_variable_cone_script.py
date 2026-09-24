"""Bounded variable-depth CustomScript and emitted-motion boundary."""

import json
import tempfile
import unittest
from pathlib import Path

from cambam_builder.cambam_reader import read_cambam_bytes
from cambam_builder.integrations.cambam.variable_cone_script import (
    audit_variable_post, build_variable_carrier,
)


class VariableConeScriptTests(unittest.TestCase):
    def test_candidate_and_synthetic_default_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "variable"
            expected = build_variable_carrier(directory)
            project = read_cambam_bytes((directory / "V-variable.cb").read_bytes())
            mop = next(m for m in project.list_mops() if m.enabled)
            self.assertEqual(mop.tool_profile, "VCutter")
            self.assertEqual(mop.custom_script.splitlines(), [
                "G0 X2 Y2 Z5", "G1 F120 X2 Y2 Z1",
                "G1 F60 X2 Y2 Z-1.25", "G1 F300 X10 Y2 Z-2.25",
                "G1 F300 X10 Y2 Z1", "G0 X-10 Y-10 Z5",
            ])
            self.assertEqual(expected["item_count"], 9)
            reference = (
                "( V-variable synthetic reference )\n"
                "( Post processor: Default )\n"
                "G21 G90 G61 G40\nG0 Z5\nT3 M6\nG17\n"
                "M3 S12000\nG0 X-10 Y-10\nG98\n" +
                mop.custom_script + "\nG80\nG0 Z5\nM5\nM30\n"
            )
            posted = directory / "V-variable.nc"
            posted.write_text(reference, encoding="utf-8")
            result = audit_variable_post(directory / "expected-motion.json", posted)
            self.assertEqual(result["status"],
                             "bounded_emitted_variable_v_motion_pass")
            self.assertEqual(result["stock_prefixes"], (("variable-v", 2),))
            self.assertAlmostEqual(result["section_rest_mm2"]["depth_1.5"],
                                   4.21460291488107)
            for changed in (
                reference.replace("Z-2.25", "Z-2.35"),
                reference.replace("G1 F300 X10", "G0 X10"),
                reference.replace("M3 S12000", "M3 S9000"),
                reference.replace("G1 F120 X2 Y2 Z1\n", ""),
            ):
                posted.write_text(changed, encoding="utf-8")
                self.assertNotEqual(
                    audit_variable_post(directory / "expected-motion.json",
                                        posted)["status"],
                    "bounded_emitted_variable_v_motion_pass")

    def test_changed_candidate_and_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "variable"
            build_variable_carrier(directory)
            expected = directory / "expected-motion.json"
            posted = directory / "V-variable.nc"
            posted.write_text("", encoding="utf-8")
            data = json.loads(expected.read_text(encoding="utf-8"))
            data["motion_fingerprint"] = "stale"
            expected.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_variable_post(expected, posted)
            candidate = directory / "V-variable.cb"
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit_variable_post(expected, posted)


if __name__ == "__main__":
    unittest.main()
