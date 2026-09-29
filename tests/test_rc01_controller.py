"""Full generated RC01 output through the bounded UCCNC profile."""

import json
from pathlib import Path
import tempfile
import unittest

from cambam_builder.integrations import ordered_output
from cambam_builder.integrations.rc01_controller import (
    _source, audit_controller_bundle, build_controller_bundle,
)


class RC01ControllerTests(unittest.TestCase):
    def test_nominal_decoded_two_stage_stock_and_handoff(self):
        Path("output").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="packet5-", dir="output") as temp:
            folder = Path(temp) / "controller"
            result = build_controller_bundle(folder)
            self.assertEqual(result["status"], "rc01_uccnc_offline_pass")
            self.assertEqual(result["move_count"], 2939)
            self.assertEqual(result["completion"], "partial_target_completion")
            self.assertEqual(result["transition_evidence"]["status"],
                             "pass_with_assumptions")
            self.assertEqual(len(result["program_sha256"]), 2)
            self.assertEqual(len(result["rough_rest_by_depth_mm2"]), 3)
            self.assertEqual(len(result["final_rest_by_depth_mm2"]), 3)
            self.assertGreater(result["rough_rest_by_depth_mm2"][0][0],
                               result["final_rest_by_depth_mm2"][0][1])
            self.assertIn(b"X14.341687", (folder / "stage-1.nc").read_bytes())
            self.assertEqual(audit_controller_bundle(folder / "rc01-evidence.json"),
                             result)
            with self.assertRaisesRegex(ValueError, "stale RC01"):
                audit_controller_bundle(folder / "rc01-evidence.json",
                                        transition_token="changed-setup")

            nc = folder / "stage-2.nc"
            original = nc.read_bytes()
            unsupported = original.replace(b"G1 F300", b"G91 F300", 1)
            _, _, job = _source("offline-synthetic-operator-state")
            with self.assertRaisesRegex(ValueError, "unsupported NC motion"):
                ordered_output.audit_files(
                    job, "uccnc",
                    ((folder / "stage-1.nc").read_bytes(), unsupported))
            nc.write_bytes(unsupported)
            with self.assertRaises(ValueError):
                audit_controller_bundle(folder / "rc01-evidence.json")
            nc.write_bytes(original)
            handoff_path = folder / "handoff.json"
            original_handoff = handoff_path.read_bytes()
            handoff = json.loads(original_handoff)
            handoff["numerical"]["coordinate_decimals"] = 4
            handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "precision"):
                ordered_output.audit_bundle(handoff_path, job)
            handoff_path.write_bytes(original_handoff)
            evidence_path = folder / "rc01-evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["source_motion_fingerprint"] = "changed"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale RC01"):
                audit_controller_bundle(evidence_path)

    def test_four_decimal_writer_is_insufficient_for_nominal_island(self):
        _, _, job = _source("offline-synthetic-operator-state")
        with self.assertRaisesRegex(ValueError, "protected island"):
            ordered_output.emit(job, "uccnc")


if __name__ == "__main__":
    unittest.main()
