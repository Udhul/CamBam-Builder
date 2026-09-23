"""Recorded native roughing stock and independent RC01 authority selection."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from cambam_builder.cam_core.rc01 import generate
from cambam_builder.integrations.cambam.rc01_stock_authority import (
    analyze_rc01_stock, check_native_freshness,
)


FIXTURE = Path(__file__).parent / "fixtures" / "rc01_native_stock"


class RC01StockAuthorityTests(unittest.TestCase):
    def test_recorded_native_post_is_a_distinct_bounded_stock_source(self):
        evidence = FIXTURE / "evidence.json"
        native = analyze_rc01_stock("native_posted", evidence_path=evidence)
        self.assertEqual(native["status"], "bounded_posted_stock_observation")
        self.assertTrue(check_native_freshness(evidence, native))
        self.assertEqual(native["input_sha256"]["post"],
                         "cde87d91d6f4c746cdabe7a80b04808d65551bc4d15db47e724550c6b0d444a2")
        self.assertEqual([row["bottom_z"] for row in
                          native["rough_rest_by_depth"]], [-1.0, -2.0, -3.0])
        for row in native["rough_rest_by_depth"]:
            self.assertAlmostEqual(row["rest_area_mm2"][0], 7.72557766758, places=8)
            self.assertAlmostEqual(row["rest_area_mm2"][1], 7.72584353670, places=8)
            self.assertEqual(row["residual_outside_ideal_or_boundary_0_05mm_mm2"], 0)
        self.assertIsNone(native["final_rest_by_depth"])
        self.assertTrue(native["rough_rest_budget_met"])
        self.assertEqual(native["stock_dependent_use"], "blocked_by_motion_or_rest")
        self.assertTrue(any("rapid below clearance" in finding
                            for finding in native["motion_role_findings"]))

        generated = analyze_rc01_stock("framework_generated", program=generate())
        self.assertEqual(generated["status"], "partial_target_completion")
        self.assertEqual(generated["stock_dependent_use"], "bounded_core_certificate")
        self.assertNotEqual(generated["motion_fingerprint"],
                            native["motion_fingerprint"])
        self.assertGreater(generated["rough_rest_by_depth_mm2"][0][0],
                           native["rough_rest_by_depth"][0]["rest_area_mm2"][1])
        with self.assertRaisesRegex(ValueError, "requires only"):
            analyze_rc01_stock("native_posted", evidence_path=evidence,
                               program=generate())
        with self.assertRaisesRegex(ValueError, "requires only"):
            analyze_rc01_stock("framework_generated", evidence_path=evidence)

    def test_source_setup_manifest_candidate_and_post_edits_invalidate(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder)
            for name in ("evidence.json", "comparison.json", "setup.json", "source.cb",
                         "N-rough.cb", "N-rough.nc"):
                shutil.copyfile(FIXTURE / name, copy / name)
            evidence = copy / "evidence.json"
            original = analyze_rc01_stock("native_posted", evidence_path=evidence)
            for name in ("comparison.json", "setup.json", "source.cb",
                         "N-rough.cb", "N-rough.nc"):
                path = copy / name
                saved = path.read_bytes()
                try:
                    path.write_bytes(saved + b" ")
                    with self.assertRaisesRegex(ValueError, "stale native"):
                        analyze_rc01_stock("native_posted", evidence_path=evidence)
                    with self.assertRaisesRegex(ValueError, "stale native"):
                        check_native_freshness(evidence, original)
                finally:
                    path.write_bytes(saved)
            self.assertTrue(check_native_freshness(evidence, original))

            saved_record = evidence.read_bytes()
            evidence.write_bytes(saved_record + b" ")
            with self.assertRaisesRegex(ValueError, "different evidence bytes"):
                check_native_freshness(evidence, original)
            evidence.write_bytes(saved_record)

            # An explicitly repinned post still needs the reviewed motion
            # interpretation; changing a cutting command cannot reuse it.
            post = copy / "N-rough.nc"
            post.write_text(post.read_text(encoding="utf-8").replace(
                "G1 F60.0 Z0.0", "G0 Z0.0", 1), encoding="utf-8")
            record = json.loads(evidence.read_text(encoding="utf-8"))
            import hashlib
            record["post_sha256"] = hashlib.sha256(post.read_bytes()).hexdigest()
            evidence.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "interpretation changed"):
                analyze_rc01_stock("native_posted", evidence_path=evidence)


if __name__ == "__main__":
    unittest.main()
