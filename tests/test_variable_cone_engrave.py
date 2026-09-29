"""Bounded XYZ Engrave candidate and whole-post comparison."""

import json
import tempfile
import unittest
from pathlib import Path

from cambam_builder.cambam_reader import read_cambam_bytes
from cambam_builder.integrations.cambam.variable_cone_engrave import (
    audit_engrave_post, build_engrave_candidate,
)


class VariableConeEngraveTests(unittest.TestCase):
    def test_candidate_and_whole_motion_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "engrave"
            build_engrave_candidate(directory)
            project = read_cambam_bytes(
                (directory / "V-variable-engrave.cb").read_bytes())
            mop = next(m for m in project.list_mops() if m.enabled)
            target = project.get_primitive("generated-variable-v-cut")
            guide = project.get_primitive("tapered-target-spine")
            self.assertEqual(project.get_mop_targets(mop), [target.internal_id])
            self.assertNotEqual(guide.internal_id, target.internal_id)
            self.assertEqual((mop.target_depth, mop.optimisation_mode), (0, "None"))
            self.assertEqual([(v.x, v.y, v.z) for v in target.vertices],
                             [(2, 2, -1.25), (10, 2, -2.25)])

            header = ("( V-variable-engrave synthetic reference )\n"
                      "( Post processor: Default )\n"
                      "G21 G90 G61 G40\nG0 Z5\nT3 M6\nG17\n"
                      "M3 S12000\nG0 X-10 Y-10\n")
            moves = ("G0 X2 Y2 Z5\nG1 F120 X2 Y2 Z1\n"
                     "G1 F60 X2 Y2 Z-1.25\n"
                     "G1 F300 X10 Y2 Z-2.25\n"
                     "G1 F300 X10 Y2 Z1\nG0 X-10 Y-10 Z5\n")
            posted = directory / "V-variable-engrave.nc"
            posted.write_text(header + moves + "M5\nM30\n", encoding="utf-8")
            result = audit_engrave_post(directory / "expected-motion.json", posted)
            self.assertEqual(result["status"],
                             "bounded_emitted_variable_v_engrave_pass")
            self.assertEqual(result["sloped_cut_count"], 1)
            self.assertEqual(result["stock_prefixes"], (("variable-v", 2),))

            # A visible sloped cut alone cannot establish the execution roles.
            native_like = ("G0 X2 Y2 Z5\nG1 F60 X2 Y2 Z-1.25\n"
                           "G1 F300 X10 Y2 Z-2.25\nG0 X10 Y2 Z5\n")
            posted.write_text(header + native_like + "M5\nM30\n",
                              encoding="utf-8")
            result = audit_engrave_post(directory / "expected-motion.json", posted)
            self.assertEqual(result["status"], "engrave_emitted_motion_deviation")
            self.assertEqual(result["sloped_cut_count"], 1)
            self.assertIn("approach", result["missing_expected_roles"])
            self.assertIn("retract", result["missing_expected_roles"])

            posted.write_text(header + "G81 X2 Y2 Z-1.25\nM30\n",
                              encoding="utf-8")
            result = audit_engrave_post(directory / "expected-motion.json", posted)
            self.assertEqual(result["status"], "unverified_engrave_post")

    def test_changed_candidate_and_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "engrave"
            build_engrave_candidate(directory)
            expected_path = directory / "expected-motion.json"
            post = directory / "V-variable-engrave.nc"
            post.write_text("", encoding="utf-8")
            manifest = json.loads(expected_path.read_text(encoding="utf-8"))
            manifest["motion_fingerprint"] = "stale"
            expected_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_engrave_post(expected_path, post)
            candidate = directory / "V-variable-engrave.cb"
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit_engrave_post(expected_path, post)


if __name__ == "__main__":
    unittest.main()
