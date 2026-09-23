"""RC01 native input, artifact, and posted-motion boundary regressions."""

import json
import tempfile
import unittest
from pathlib import Path

from cambam_builder.cam_core.rc01 import Job, Move, generate
from cambam_builder.cambam_reader import read_cambam_bytes
from cambam_builder.integrations.cambam.rc01_post import compare_file, compare_posted, read_default_post
from cambam_builder.integrations.cambam.rc01_adapter import build_artifacts, build_native_variant, normalize, synthetic_setup, synthetic_source
from cambam_builder.integrations.cambam.rc01_native_post import _area_by_depth, _budget, _path_for_move, _t2_vertical_access, audit_native_posts
from cambam_builder.integrations.cambam.rc01_script import audit_script_post, build_script_carrier


class NativeRC01Tests(unittest.TestCase):
    def test_full_literal_motion_carrier_and_post_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "script"
            manifest = build_script_carrier(folder)
            self.assertEqual(manifest["item_count"], 2945)
            candidate = folder / "S-combined.cb"
            project = read_cambam_bytes(candidate.read_bytes())
            self.assertEqual(normalize(project, synthetic_setup(),
                                       allow_attachments=True), Job())
            self.assertEqual([m.name for m in project.list_mops() if m.enabled],
                             ["RC01 T1 rough plus T2 cleanup literal motion"])
            script = next(m for m in project.list_mops() if m.enabled).custom_script
            post = folder / "S-combined.nc"
            header = ("( S-combined synthetic post )\n"
                      "( Post processor: Default )\n"
                      "G21 G90 G61 G40\nG0 Z5\nT1 M6\nG17\n"
                      "M3 S12000\nG0 Z5\nG0 X-10 Y-10\nG98\n")
            footer = "\nG80\nG0 Z5\nM5\nM30\n"
            post.write_text(header + script.replace("|", "\n") + footer,
                            encoding="utf-8")
            result = audit_script_post(folder / "comparison.json", post)
            self.assertEqual(result["status"], "bounded_emitted_motion_pass")
            self.assertEqual(result["certificate_status"], "partial_target_completion")
            self.assertEqual(result["item_count"], manifest["item_count"])
            self.assertTrue(all(lo <= hi < 8.3 for lo, hi in
                                result["rough_rest_by_depth_mm2"]))
            self.assertTrue(all(lo <= hi < 1.4 for lo, hi in
                                result["final_rest_by_depth_mm2"]))
            post.write_text(header + script.replace("|", "\n").replace(
                "G1 F120 X5 Y5 Z1", "G0 X5 Y5 Z1", 1) + footer,
                encoding="utf-8")
            self.assertEqual(audit_script_post(folder / "comparison.json", post)
                             ["status"], "deviation")

    def test_native_arc_reader_preserves_center_and_bounded_curve(self):
        prefix = ("G21 G90 G61 G40\nG17\nT1 M6\nM3 S12000\n"
                  "G0 X5 Y5\nG0 Z1\nG1 F60 Z-1\n")
        arc = "G3 F300 X6 Y6 I0 J1\n"
        post = prefix + arc + "G0 Z5\nM5\nM30\n"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            read_default_post(post)
        items, warnings = read_default_post(post, allow_arcs=True)
        self.assertFalse(warnings)
        curve = next(item for item in items if item.get("g") == 3)
        self.assertEqual(curve["center"], [5, 6])
        points, error = _path_for_move(curve, -1)
        self.assertGreater(len(points), 2)
        self.assertLess(error, 0.001)
        with self.assertRaisesRegex(ValueError, "inconsistent arc radii"):
            read_default_post(prefix + arc.replace("J1", "J2") +
                              "G0 Z5\nM5\nM30\n", allow_arcs=True)

    def test_t2_column_has_exact_posted_t1_boundary_witness(self):
        rough = [{"type": "move", "g": 1, "tool": "T1", "line": 42,
                  "start": [3, 3, -3], "end": [3, 27, -3]}]
        t2 = [{"type": "move", "g": 0, "tool": "T2", "line": 43,
               "start": [1, 3.4, 5], "end": [1, 3.4, -1]}]
        access = _t2_vertical_access(rough, t2)
        self.assertEqual(access[0]["exact_single_t1_cut_witness_line"], 42)
        t2[0]["start"][0] = t2[0]["end"][0] = 0.9
        access = _t2_vertical_access(rough, t2)
        self.assertIsNone(access[0]["exact_single_t1_cut_witness_line"])

    def test_posted_rest_oracle_accepts_known_full_coverage(self):
        job = Job()
        moves = [{"type": "move", "g": 1, "tool": move.tool,
                  "start": list(map(float, move.start)),
                  "end": list(map(float, move.end))}
                 for move in generate(job).items
                 if isinstance(move, Move) and move.role in ("entry", "cut")]
        rough = _area_by_depth([m for m in moves if m["tool"] == "T1"], job, 3)
        final = _area_by_depth(moves, job, 1)
        self.assertEqual(_budget(rough, final), (True, True))
        self.assertTrue(all(7.7 < row["rest_area_mm2"][0] < 7.9
                            for row in rough))
        self.assertTrue(all(0.8 < row["rest_area_mm2"][0] < 1.0
                            for row in final))

    def test_native_pocket_variant_and_post_replay_reject_missing_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "native"
            manifest = build_native_variant(folder)
            self.assertEqual(manifest["format"], "rc01-native-v1")
            self.assertEqual([len(manifest["variants"][key]["enabled_mops"])
                              for key in ("rough", "combined")], [1, 5])
            for key in ("rough", "combined"):
                project = read_cambam_bytes((folder / manifest["variants"][key]["file"])
                                            .read_bytes())
                self.assertEqual(normalize(project, synthetic_setup(),
                                           allow_attachments=True), Job())
                self.assertTrue(all(not m.enabled for m in project.list_mops()[:2]))
                self.assertTrue(all(m.enabled for m in project.list_mops()[2:]))
            preamble = ("( Post processor: Default )\nG21 G90 G61 G40\n"
                        "G0 Z5\nT1 M6\nG17\nM3 S12000\nG0 X5 Y5\n"
                        "G0 Z0\nG1 F60 Z-1\nG0 Z5\n")
            rough = folder / "N-rough.nc"
            combined = folder / "N-native-cleanup.nc"
            rough.write_text("( N-rough synthetic )\n" + preamble + "M5\nM30\n",
                             encoding="utf-8")
            combined.write_text("( N-native-cleanup synthetic )\n" + preamble
                                + "M5\nT2 M6\nM3 S12000\n"
                                "G0 X5 Y5\nG0 Z0\nG1 F60 Z-1\n"
                                "G0 Z5\nM5\nM30\n", encoding="utf-8")
            result = audit_native_posts(folder / "comparison.json", rough, combined)
            self.assertTrue(result["rough_prefix_identical"])
            self.assertFalse(result["rough_rest_budget_met"])
            self.assertFalse(result["final_rest_budget_met"])
            self.assertEqual(result["status"], "fails_RC01")
            self.assertTrue(any("rapid below clearance" in issue
                                for issue in result["issues"]["rough"]))

    def test_role_trial_reimports_repaired_full_pocket_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "trial"
            manifest = build_native_variant(folder, role_trial=True)
            self.assertEqual(manifest["format"], "rc01-native-v1")
            self.assertTrue(manifest["role_trial"])
            self.assertEqual(manifest["pocket_repairs"], {
                "lead_in_type": "None", "optimisation_mode": "None",
                "stepover_feedrate": "Cut Feedrate",
                "max_crossover_distance": 0,
            })
            self.assertEqual([len(manifest["variants"][key]["enabled_mops"])
                              for key in ("rough", "combined")], [1, 5])
            for key in ("rough", "combined"):
                candidate = folder / manifest["variants"][key]["file"]
                project = read_cambam_bytes(candidate.read_bytes())
                self.assertEqual(normalize(project, synthetic_setup(),
                                           allow_attachments=True), Job())
                self.assertTrue(all(not mop.enabled for mop in project.list_mops()[:2]))
                enabled = [mop for mop in project.list_mops() if mop.enabled]
                self.assertTrue(all(mop.lead_in_type == "None" and
                                    mop.optimisation_mode == "None" and
                                    mop.stepover_feedrate == "Cut Feedrate" and
                                    mop.max_crossover_distance == 0
                                    for mop in enabled))
            with self.assertRaisesRegex(ValueError, "new or empty"):
                build_native_variant(folder, role_trial=True)

    def test_native_input_normalizes_and_rejects_relevant_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.cb"
            synthetic_source().save(str(path))
            project = read_cambam_bytes(path.read_bytes())
            self.assertEqual(normalize(project, synthetic_setup()), Job())
            source = project.list_mops()[0]
            source.name = "renamed intent"
            source.user_identifier = "renamed-identity"
            project.save(str(path))
            project = read_cambam_bytes(path.read_bytes())
            self.assertEqual(normalize(project, synthetic_setup()), Job())
            source = project.list_mops()[0]
            source.stepover = 0.5
            with self.assertRaisesRegex(ValueError, "stepover"):
                normalize(project, synthetic_setup())
            source.stepover = 0.4
            source.set_parameter_state("tool_diameter", "Default")
            project.save(str(path))
            project = read_cambam_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "inherited"):
                normalize(project, synthetic_setup())
            synthetic_source().save(str(path))
            project = read_cambam_bytes(path.read_bytes())
            target = next(p for p in project.list_primitives() if p.user_identifier == "rc01-target")
            target.outer_curve.vertices[1].x = 41
            with self.assertRaisesRegex(ValueError, "rectangular contours"):
                normalize(project, synthetic_setup())
            setup = synthetic_setup()
            setup["tools"][1]["holder_start"] = 2
            with self.assertRaisesRegex(ValueError, "tool change"):
                normalize(read_cambam_bytes(path.read_bytes()), setup)

    def test_artifacts_keep_source_targets_and_separate_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_artifacts(Path(directory) / "rc01")
            self.assertEqual(manifest["certificate_status"], "partial_target_completion")
            self.assertEqual(manifest["full_item_count"], 2945)
            self.assertEqual(set(manifest["variants"]), {"A", "B", "C"})
            self.assertEqual([len(manifest["variants"][x]["enabled_mops"])
                              for x in "ABC"], [3, 6, 7])
            source = read_cambam_bytes((Path(directory) / "rc01" / "source.cb").read_bytes())
            source_region_id = source.list_primitives()[0].internal_id
            source_mop_ids = [m.internal_id for m in source.list_mops()]
            for letter in "ABC":
                item = manifest["variants"][letter]
                file = Path(directory) / "rc01" / item["file"]
                project = read_cambam_bytes(file.read_bytes())
                self.assertEqual(normalize(project, synthetic_setup(), allow_attachments=True), Job())
                self.assertTrue(item["source_mops_disabled"])
                self.assertEqual(next(p.internal_id for p in project.list_primitives()
                                      if p.user_identifier == "rc01-target"), source_region_id)
                self.assertEqual([m.internal_id for m in project.list_mops()[:2]],
                                 source_mop_ids)
                self.assertEqual([len(project.get_mop_targets(m)) for m in project.list_mops()[:2]],
                                 [1, 1])
                candidates = [m for m in project.list_mops() if m.enabled
                              and m.name.startswith("CANDIDATE")]
                for mop in candidates:
                    self.assertEqual(mop.stock_surface, mop.target_depth + 1)
                    self.assertEqual(mop.depth_increment, 1)
                    self.assertEqual(mop.optimisation_mode, "None")
                    targets = set(project.get_mop_targets(mop))
                    paths = [p for p in project.list_primitives()
                             if p.internal_id in targets]
                    self.assertTrue(paths)
                    self.assertTrue(all(vertex.z == 0 for path in paths
                                        for vertex in path.vertices))
            self.assertEqual(json.loads((Path(directory) / "rc01" / "comparison.json").read_text())
                             ["motion_fingerprint"], manifest["motion_fingerprint"])
            candidate = Path(directory) / "rc01" / manifest["variants"]["A"]["file"]
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            posted = Path(directory) / "rc01" / "A-rough.nc"
            posted.write_text("M30\n")
            self.assertEqual(compare_file(Path(directory) / "rc01" / "comparison.json",
                                          "A", posted)["status"], "unverified")

    def test_reader_rejects_unmodeled_motion_and_reports_deviation(self):
        prefix = "G21 G90 G61 G40\nG17\nT1 M6\nM3 S12000\n"
        end = "G0 X-10 Y-10 Z5\nM5\nM30\n"
        expected = [
            {"type": "event", "kind": "tool_change", "tool": "T1",
             "position": [-10, -10, 5], "rpm": 0},
            {"type": "event", "kind": "spindle_start", "tool": "T1",
             "position": [-10, -10, 5], "rpm": 12000},
            {"type": "move", "tool": "T1", "start": [-10, -10, 5],
             "end": [5, 5, 5], "feed": 0},
        ]
        manifest = {"format": "rc01-comparison-v1", "rough_item_count": 3,
                    "expected_items": expected}
        good = prefix + "G0 X5 Y5\n" + end
        actual, warnings = read_default_post(good)
        self.assertFalse(warnings)
        self.assertEqual(actual[2]["end"], [5, 5, 5])
        # This deliberately short manifest makes the remaining retract/stop extra.
        self.assertEqual(compare_posted(manifest, "A", good)["status"], "deviation")
        bad = prefix + "G0 X6 Y5\n" + end
        result = compare_posted(manifest, "A", bad)
        self.assertEqual((result["status"], result["field"]), ("deviation", "end"))
        self.assertEqual(compare_posted(manifest, "A", prefix + "G2 X5 Y5\nM30\n")
                         ["status"], "unverified")
        manifest["expected_items"] = actual
        manifest["rough_item_count"] = len(actual)
        self.assertEqual(compare_posted(manifest, "A", good)["status"], "sequence_matches")
        self.assertEqual(compare_posted(manifest, "A", good.replace("G61", "G64"))
                         ["status"], "unverified")
        native_preamble = "G21 G90 G61 G40\nG0 Z5.0\nT1 M6\nG17\nM3 S12000\n"
        first = compare_posted(manifest, "A", native_preamble + "G0 X6 Y5\n" + end)
        self.assertEqual((first["status"], first["field"]), ("deviation", "end"))
        actual, warnings = read_default_post(native_preamble + "G0 X5 Y5\n" + end)
        self.assertIn("initial machine position", warnings[0])
        displaced_change = (native_preamble + "G0 X5 Y5\nT2 M6\nM3 S12000\n"
                            + end)
        displaced, warnings = read_default_post(displaced_change)
        self.assertEqual(displaced[3]["position"], [5, 5, 5])
        self.assertEqual(displaced[3]["rpm"], 12000)
        self.assertTrue(any("without explicit spindle stop" in warning
                            for warning in warnings))


if __name__ == "__main__":
    unittest.main()
