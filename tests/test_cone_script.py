"""Bounded cone carrier and posted-motion boundary regressions."""

import json
import tempfile
import unittest
from pathlib import Path

from cambam_builder.cambam_reader import read_cambam_bytes
from cambam_builder.integrations.cambam.cone_script import (
    audit_cone_post, build_cone_carrier,
)


class ConeScriptTests(unittest.TestCase):
    def test_carrier_reimports_and_synthetic_default_wrapper_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "cone"
            expected = build_cone_carrier(directory)
            project = read_cambam_bytes((directory / "V-cone.cb").read_bytes())
            mop = next(m for m in project.list_mops() if m.enabled)
            self.assertEqual(mop.tool_profile, "VCutter")
            self.assertEqual(mop.custom_script.splitlines(), [
                "G0 X2 Y2 Z5", "G1 F120 X2 Y2 Z1",
                "G1 F60 X2 Y2 Z-2", "G1 F300 X10 Y2 Z-2",
                "G1 F300 X10 Y2 Z1", "G0 X-10 Y-10 Z5",
            ])
            self.assertEqual(expected["item_count"], 9)
            reference = (
                "( V-cone synthetic reference )\n"
                "( Post processor: Default )\n"
                "G21 G90 G61 G40\nG0 Z5\nT3 M6\nG17\n"
                "M3 S12000\nG0 X-10 Y-10\nG98\n" +
                mop.custom_script + "\nG80\nG0 Z5\nM5\nM30\n"
            )
            posted = directory / "V-cone.nc"
            posted.write_text(reference, encoding="utf-8")
            result = audit_cone_post(directory / "expected-motion.json", posted)
            self.assertEqual(result["status"], "bounded_emitted_cone_motion_pass")
            self.assertEqual(result["stock_prefixes"], (("slot", 2),))
            self.assertAlmostEqual(result["section_rest_mm2"]["surface"],
                                   3.4336293856408275)
            self.assertAlmostEqual(result["section_rest_mm2"]["depth_1"],
                                   0.8584073464102069)
            self.assertEqual(result["section_rest_mm2"]["depth_2"], 0)

            for changed in (
                reference.replace("G1 F60 X2 Y2 Z-2", "G0 X2 Y2 Z-2"),
                reference.replace("G1 F300 X10 Y2 Z-2", "G1 F300 X10 Y2 Z-2.1"),
                reference.replace("M3 S12000", "M3 S9000"),
                reference.replace("G1 F120 X2 Y2 Z1\n", ""),
            ):
                posted.write_text(changed, encoding="utf-8")
                self.assertNotEqual(
                    audit_cone_post(directory / "expected-motion.json", posted)["status"],
                    "bounded_emitted_cone_motion_pass")

    def test_changed_candidate_and_stale_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "cone"
            build_cone_carrier(directory)
            expected_path = directory / "expected-motion.json"
            post = directory / "V-cone.nc"
            post.write_text("", encoding="utf-8")
            data = json.loads(expected_path.read_text(encoding="utf-8"))
            data["motion_fingerprint"] = "stale"
            expected_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_cone_post(expected_path, post)
            candidate = directory / "V-cone.cb"
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit_cone_post(expected_path, post)


if __name__ == "__main__":
    unittest.main()
