"""Analytic circular inlay and independent decoded output acceptance."""

from dataclasses import replace
import hashlib
import math
import unittest

from cambam_builder.cam_core.inlay import (
    InlayRequest, assembly, audit_pair, generate,
)
from cambam_builder.cam_core.v_region import VProfile
from cambam_builder.integrations.ordered_output import emit


def synthetic_pair(clearance=0.0, *, angle_tangent=0.5, offset=(0.0, 0.0)):
    tool = VProfile("pointed", 2 * math.degrees(math.atan(angle_tangent)),
                    0, 0.6, 1.1)
    return generate(InlayRequest(4, 7, 1, 0.9, clearance, tool,
                                 assembly_xy_mm=offset))


class PairedInlayTests(unittest.TestCase):
    def test_exact_cross_section_and_insertion_oracle(self):
        for clearance in (0.0, 0.1):
            with self.subTest(clearance=clearance):
                pair = synthetic_pair(clearance)
                q = pair.request
                female = pair.female.stages[0].operation
                male = pair.male.stages[0].operation
                self.assertIsNot(pair.female, pair.male)
                self.assertEqual(pair.female.source_fingerprint,
                                 pair.male.source_fingerprint)
                self.assertAlmostEqual(female.target_interval(0.4)[1], 3.8)
                self.assertAlmostEqual(male.target_interval(0.5)[0],
                                       3.8-clearance)
                female_report = emit(pair.female, "uccnc")[1]
                male_report = emit(pair.male, "uccnc")[1]
                self.assertAlmostEqual(
                    female_report["stock_access_residual"]["sections"][1]
                    ["required_area_mm2"], math.pi * 3.775 ** 2)
                self.assertAlmostEqual(
                    male_report["stock_access_residual"]["sections"][1]
                    ["required_area_mm2"],
                    math.pi * (7 ** 2 - (3.775-clearance) ** 2))
                oracle = assembly(q)
                self.assertAlmostEqual(oracle["side_gap_mm"], clearance)
                self.assertAlmostEqual(oracle["bottom_gap_mm"], 0.1)
                self.assertEqual(oracle["side_contact"], clearance == 0)
                for insertion, gap in oracle["insertion_gap_samples_mm"]:
                    self.assertAlmostEqual(gap,
                                           clearance + (0.9-insertion)*0.5)

    def test_both_stock_states_from_complete_decoded_programs(self):
        for clearance in (0.0, 0.1):
            for dialect in ("uccnc", "grbl"):
                with self.subTest(clearance=clearance, dialect=dialect):
                    pair = synthetic_pair(clearance)
                    female_files, female_report = emit(pair.female, dialect)
                    male_files, male_report = emit(pair.male, dialect)
                    certificate = audit_pair(pair, female_files, male_files,
                                             dialect)
                    self.assertEqual(certificate["status"],
                                     "paired_inlay_pass")
                    for side, report in (("female", female_report),
                                         ("male", male_report)):
                        stock = report["stock_access_residual"]
                        self.assertEqual(stock["side"], side)
                        self.assertTrue(stock["independent_stock"])
                        self.assertGreater(stock["decoded_ring_count"], 30)
                        for row in stock["sections"][:3]:
                            self.assertLess(row["residual_area_mm2"], 0.002)
                            self.assertLess(row["protected_overcut_mm2"], 0.002)
                        self.assertGreater(stock["sections"][-2]
                                           ["residual_area_mm2"], 0)
                    for gap in certificate["decoded_side_gap_samples_mm"]:
                        self.assertAlmostEqual(gap, clearance, delta=0.00011)
                    self.assertEqual(
                        certificate["program_sha256"][0][0],
                        hashlib.sha256(female_files[0]).hexdigest())
                    self.assertEqual(
                        certificate["program_sha256"][1][0],
                        hashlib.sha256(male_files[0]).hexdigest())

    def test_reject_registration_clearance_and_changed_tool(self):
        with self.assertRaisesRegex(ValueError, "registration"):
            synthetic_pair(0.0, offset=(0.01, 0))
        registered = synthetic_pair(0.1, offset=(0.05, 0))
        self.assertAlmostEqual(assembly(registered.request)["side_gap_mm"],
                               0.05)
        with self.assertRaisesRegex(ValueError, "registration"):
            synthetic_pair(0.1, offset=(0.101, 0))
        with self.assertRaisesRegex(ValueError, "impossible"):
            synthetic_pair(4)
        with self.assertRaisesRegex(ValueError, "impossible"):
            generate(replace(synthetic_pair().request, flipped=False))
        pair = synthetic_pair()
        changed = synthetic_pair(angle_tangent=0.45)
        self.assertNotEqual(pair.fingerprint, changed.fingerprint)
        female_files = emit(pair.female, "uccnc")[0]
        male_files = emit(pair.male, "uccnc")[0]
        with self.assertRaisesRegex(ValueError, "stale"):
            audit_pair(changed, female_files, male_files, "uccnc",
                       expected_fingerprint=pair.fingerprint)
        with self.assertRaisesRegex(ValueError, "stale"):
            audit_pair(replace(pair, request=changed.request),
                       female_files, male_files, "uccnc")

    def test_reject_changed_program_and_stale_hashes(self):
        pair = synthetic_pair(0.1)
        female = emit(pair.female, "uccnc")[0]
        male = emit(pair.male, "uccnc")[0]
        hashes = (hashlib.sha256(female[0]).hexdigest(),)
        certificate = audit_pair(pair, female, male, "uccnc",
                                 female_hashes=hashes)
        self.assertEqual(certificate["status"], "paired_inlay_pass")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            audit_pair(pair, (female[0]+b"\n",), male, "uccnc",
                       female_hashes=hashes)
        with self.assertRaises(ValueError):
            audit_pair(pair, (female[0].replace(b"G3", b"G2", 1),),
                       male, "uccnc")
        with self.assertRaises(ValueError):
            audit_pair(pair, female, (male[0].replace(b"G3", b"G2", 1),),
                       "uccnc")


if __name__ == "__main__":
    unittest.main()
