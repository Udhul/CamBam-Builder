"""Regression checks for the native mapping fixture and intake boundary."""

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from cambam_builder.integrations.cambam.optimizer_corpus import (
    MODES, build, inspect_post, parse_default_motion,
)


class OptimizerCorpusTests(unittest.TestCase):
    def test_pair_is_same_case_except_mode_and_name(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "corpus"
            manifest = build(directory)
            self.assertEqual(set(manifest["cases"]),
                             {"atlas-legacy", "atlas-new",
                              "links-legacy", "links-new"})
            for family, count in (("atlas", 6), ("links", 3)):
                pair = [manifest["cases"][f"{family}-{mode}"] for mode in MODES]
                self.assertEqual([len(case["operations"]) for case in pair],
                                 [count, count])
                self.assertEqual([[op["name"], op["kind"],
                                   op["targets_xml_order"], op["tool"]]
                                  for op in pair[0]["operations"]],
                                 [[op["name"], op["kind"],
                                   op["targets_xml_order"], op["tool"]]
                                  for op in pair[1]["operations"]])
                roots = [ET.parse(directory / case["file"]).getroot()
                         for case in pair]
                for root in roots:
                    root.attrib.pop("Name")
                    for mode_element in root.findall(
                            "./parts/part/machineops/*/OptimisationMode"):
                        mode_element.text = "MODE"
                self.assertEqual(ET.tostring(roots[0]), ET.tostring(roots[1]))

    def test_native_post_parser_retains_modal_arcs_and_unknown_cycles(self):
        posted = """( Made using CamBam )
( Post processor: Default )
G21 G90 G61 G40
G0 Z5
T1 M6
( CASE_A )
G17
M3 S12000
G0 X10 Y10
G1 F60 Z-1
G2 F240 X12 Y10 I1 J0
( CASE_B )
G81 X20 Y10 Z-2 R2 F60
G80
M5
M30
"""
        result = parse_default_motion(posted, {"CASE_A", "CASE_B"})
        self.assertEqual([item["name"] for item in result["sections"]],
                         ["CASE_A", "CASE_B"])
        self.assertEqual(result["moves"][0]["start"], [None, None, None])
        self.assertEqual(result["moves"][-1]["center"], [11.0, 10.0])
        self.assertEqual(result["moves"][-1]["feed"], 240.0)
        self.assertEqual(result["unsupported"][0]["reason"], "unsupported_G81")
        self.assertEqual(result["events"][0]["m"], 6)

    def test_intake_guards_source_hash_and_mop_comments(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "corpus"
            manifest = build(directory)
            key = "links-legacy"
            case = manifest["cases"][key]
            posted = directory / case["post_file"]
            text = (f"( Made using CamBam )\n( {key} synthetic )\n"
                    "( Post processor: Default )\n")
            text += "G21 G90 G61 G40\nG0 Z5\nT1 M6\nG17\n"
            for op in case["operations"]:
                text += f"( {op['name']} )\nM3 S12000\nG0 X10 Y10\nG1 F240 Z-1\n"
            posted.write_text(text + "M5\nM30\n", encoding="utf-8")
            result = inspect_post(directory / "manifest.json", key, posted)
            self.assertEqual(result["status"], "posted_unreviewed")
            self.assertEqual(result["stock_authority"], "none")
            self.assertEqual(len(result["sections"]), 3)
            posted.write_text(text.replace(f"( {key} synthetic )",
                                           "( links-new synthetic )") + "M5\nM30\n",
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "title does not identify"):
                inspect_post(directory / "manifest.json", key, posted)
            posted.write_text(text.replace(f"( {case['operations'][0]['name']} )",
                                           "( omitted )") + "M30\n",
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "MOP comments incomplete"):
                inspect_post(directory / "manifest.json", key, posted)
            source = directory / case["file"]
            source.write_bytes(source.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                inspect_post(directory / "manifest.json", key, posted)


if __name__ == "__main__":
    unittest.main()
