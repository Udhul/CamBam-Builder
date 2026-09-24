"""Headless RC01 reference output and exact parsed-motion acceptance."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cambam_builder.cam_core import rc01
from cambam_builder.integrations.direct_rc01 import (
    _parsed_program, audit_program, build_program, render,
)


class DirectRC01Tests(unittest.TestCase):
    def test_nominal_file_reparses_and_replays_both_tool_prefixes(self):
        with tempfile.TemporaryDirectory(prefix="rc01-direct-test-") as temporary:
            folder = Path(temporary) / "direct"
            result = build_program(folder)
            self.assertEqual(result["status"], "bounded_direct_rc01_pass")
            self.assertEqual(result["item_count"], 2945)
            self.assertEqual(result["move_count"], 2939)
            self.assertEqual(result["completion"], "partial_target_completion")
            self.assertLess(result["rough_rest_by_depth_mm2"][0][1], 7.8)
            self.assertLess(result["final_rest_by_depth_mm2"][0][1], 0.95)
            self.assertGreater(result["rough_rest_by_depth_mm2"][0][0],
                               result["final_rest_by_depth_mm2"][0][1])
            self.assertEqual(result["area_coordinate_enclosure_mm"], 1e-9)
            self.assertIsNone(result["location_numeric_enclosure_mm"])
            text = (folder / "direct-RC01.nc").read_text(encoding="ascii")
            self.assertIn("T1 M6\nM3 S12000\n", text)
            self.assertIn("M5\nT2 M6\nM3 S12000\n", text)
            self.assertTrue(text.endswith("M5\nM30\n"))
            audited = audit_program(folder / "direct-evidence.json")
            self.assertEqual(audited["program_sha256"], result["program_sha256"])
            self.assertEqual(audited["final_rest_by_depth_mm2"],
                             result["final_rest_by_depth_mm2"])

    def test_changed_feed_and_forged_hash_cannot_inherit_acceptance(self):
        expected = rc01.generate()
        original = render(expected)
        changed = original.replace("G1 F120 X5 Y5 Z1", "G1 F200 X5 Y5 Z1", 1)
        with self.assertRaisesRegex(ValueError, "move .* differs"):
            _parsed_program(changed, expected)
        with tempfile.TemporaryDirectory(prefix="rc01-direct-test-") as temporary:
            folder = Path(temporary) / "direct"
            folder.mkdir()
            (folder / "direct-RC01.nc").write_bytes(changed.encode("ascii"))
            manifest = {
                "format": "direct-rc01-v1", "program": "direct-RC01.nc",
                "program_sha256": hashlib.sha256(original.encode("ascii")).hexdigest(),
                "job_fingerprint": rc01.Job().fingerprint,
                "motion_fingerprint": expected.motion_fingerprint,
                "result": {},
            }
            path = folder / "direct-evidence.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "program changed"):
                audit_program(path)
            manifest["program_sha256"] = hashlib.sha256(changed.encode("ascii")).hexdigest()
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified rendering"):
                audit_program(path)


if __name__ == "__main__":
    unittest.main()
