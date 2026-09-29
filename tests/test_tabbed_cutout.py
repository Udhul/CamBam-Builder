"""Synthetic retained-bridge regressions plus byte-bound CamBam observations."""

import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from cambam_builder.integrations.cambam.tabbed_cutout import (
    Case, audit_bundle, write_bundle,
)
from cambam_builder import CBProject
from cambam_builder.native.writer import serialize_cambam_bytes


FIXTURES = (Path(__file__).resolve().parents[1] / "output" /
            "tabbed-cutout-20260927-04" / "fixtures")
FIXTURE_NAMES = ("B-fresh-manual.cb", "B-fresh-manual.nc",
                 "C-fresh-manual.cb", "C-fresh-manual.nc")
ACCEPTED_HASHES = {
    "B": ("1d5a6d7a984cca8ef5d7539ec68fbcbfa17477adbe31738663a2924d90490d27",
          "8ebb6d30c3cae613fd0cca9a3ef2ada99db3535b1537224e40bf72727dc8d3c9"),
    "C": ("35b36611ab4cf9ab98322406b98e24d2642ef71e0c7bcbf3bb550df688a6ae98",
          "f51d810d97227094df42177566ce87e3061b1641ed0d5bc1e5b4a7816a9b7a1f"),
}


class TabbedCutoutTests(unittest.TestCase):
    def _case(self, root, name, count):
        source = root / f"{name}-fresh-manual.cb"
        post = root / f"{name}-fresh-manual.nc"
        project = CBProject(f"manual-tabs-fresh-{name}")
        outline = project.add_pline(
            "Geometry", [(10, 10), (70, 10), (70, 40), (10, 40)],
            closed=True, identifier="outline")
        part = project.add_part(
            "Part", stock_width=80.0, stock_height=50.0,
            stock_thickness=3.0, stock_surface=0.0)
        left = [25] if count == 4 else [17, 33]
        project.add_profile_mop(
            part, [outline], identifier="manual-tab-fixture", name="manual-tab-fixture",
            profile_side="Outside",
            target_depth=-3, stock_surface=0, tool_diameter=3, tool_number=1,
            spindle_speed=12000, cut_feedrate=250, plunge_feedrate=80,
            clearance_plane=5, roughing_clearance=0, lead_in_type="None",
            tab_method="Manual", tab_style="Square", tab_width=6, tab_height=1,
            tab_min_tabs=count, tab_max_tabs=count,
            manual_tab_points=[(40, 10), (70, 25), (40, 40)] + [(10, y) for y in left])
        source.write_bytes(serialize_cambam_bytes(project))
        # Independent hand-derived 1.5 mm offset rectangle, with 9 mm
        # tool-center gaps retaining 6 mm stock bridges. This is synthetic
        # parser input, not a claim of CamBam output or optimizer behavior.
        lines = [f"( {source.stem} synthetic test )", "( Post processor: Default )",
                 "G21 G90 G17", "G0 Z5", "T1 M6", "( manual-tab-fixture )",
                 "S12000 M3", "G0 X8.5 Y8.5", "G1 F80 Z-2",
                 "G1 F250 X71.5", "G1 Y41.5", "G1 X8.5", "G1 Y8.5",
                 "G1 F80 Z-3", "G1 F250 X35.5", "G0 Z-2.0",
                 "G1 X44.5", "G1 F80.0 Z-3.0", "G1 F250 X71.5",
                 "G1 Y20.5", "G0 Z-2", "G1 Y29.5", "G1 F80 Z-3",
                 "G1 F250 Y41.5", "G1 X44.5", "G0 Z-2", "G1 X35.5",
                 "G1 F80 Z-3", "G1 F250 X8.5"]
        for y in reversed(left):
            lines.extend([f"G1 Y{y + 4.5}", "G0 Z-2", f"G1 Y{y - 4.5}",
                          "G1 F80 Z-3", "G1 F250"])
        lines.extend(["G1 Y8.5", "G0 Z5", "M5", "M30"])
        post.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return Case(source, post, count)

    def test_synthetic_four_and_five_tabs_leave_retained_bridges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, count in (("B", 4), ("C", 5)):
                with self.subTest(name=name):
                    case = self._case(root, name, count)
                    report = write_bundle(root / name, case)
                    self.assertEqual(report, audit_bundle(root / name / "handoff.json", case))
                    self.assertEqual(report["bridges"]["count"], count)
                    self.assertTrue(report["bridges"]["retained_part_connected"])
                    self.assertEqual(report["bridges"]["minimum_centerline_stock_width_mm"], 6)
                    self.assertEqual(report["generated_roles"],
                                     ("rapid", "entry", "cut", "retract", "rapid"))
                    self.assertLess(report["groove_section_0_25_mm2"][1], 1)
                    self.assertEqual(report["groove_section_0_25_mm2"][2], 0)
                    for row in report["remaining_stock_prefixes_mm2"].values():
                        self.assertGreater(row[0][1], row[1][1])
                        self.assertGreater(row[1][1], 0)

    def test_source_post_and_generated_bytes_invalidate_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self._case(root, "B", 4)
            manifest = root / "job" / "handoff.json"
            write_bundle(manifest.parent, case)
            original = case.source.read_bytes()
            case.source.write_bytes(original.replace(b"manual-tabs-fresh-B",
                                                     b"manual-tabs-fresh-B-edit"))
            with self.assertRaises(ValueError):
                audit_bundle(manifest, case)
            case.source.write_bytes(original)
            case.source.write_bytes(original.replace(b"<Width>6</Width>",
                                                     b"<Width>7</Width>"))
            with self.assertRaises(ValueError):
                write_bundle(root / "wrong-width", case)
            case.source.write_bytes(original)
            original_post = case.post.read_bytes()
            case.post.write_bytes(original_post + b"\n")
            with self.assertRaises(ValueError):
                audit_bundle(manifest, case)
            case.post.write_bytes(original_post)
            program = manifest.parent / "interior-v.nc"
            original_program = program.read_bytes()
            for old, new in ((b"G1 F60", b"G1 F61"),
                             (b"G0 X25", b"G0 X24"),
                             (b"G1 F250 X55", b"G1 F250 X54")):
                with self.subTest(old=old):
                    self.assertIn(old, original_program)
                    program.write_bytes(original_program.replace(old, new, 1))
                    with self.assertRaises(ValueError):
                        audit_bundle(manifest, case)
            program.write_bytes(original_program)
            self.assertEqual(audit_bundle(manifest, case)["status"], "pass")

    def test_removed_and_misplaced_posted_tab_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self._case(root, "B", 4)
            original = case.post.read_text(encoding="utf-8")
            old = "G0 Z-2.0\nG1 X44.5\nG1 F80.0 Z-3.0"
            self.assertIn(old, original)
            for name, replacement in (
                ("removed", "G1 X44.5"),
                ("misplaced", "G0 Z-2.0\nG1 X45.5\nG1 F80.0 Z-3.0"),
            ):
                with self.subTest(name=name):
                    case.post.write_text(original.replace(old, replacement, 1),
                                         encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "posted final-depth bridge count"):
                        write_bundle(root / name, case)

    def test_reordering_needs_new_evidence_and_fresh_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self._case(root, "B", 4)
            first = write_bundle(root / "first", case)
            manifest_path = root / "first" / "handoff.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["order"] = ["native-profile", "interior-v"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_bundle(manifest_path, case)
            reverse = write_bundle(root / "reverse", case,
                                   order=("native-profile", "interior-v"))
            self.assertEqual(reverse["status"], "pass")
            self.assertNotEqual(first["certificate"], reverse["certificate"])
            self.assertNotEqual(first["remaining_stock_prefixes_mm2"]["0.25"][0][1],
                                reverse["remaining_stock_prefixes_mm2"]["0.25"][0][1])


@unittest.skipUnless(all((FIXTURES / name).is_file() for name in FIXTURE_NAMES),
                     "accepted B/C one-off inputs are absent from ignored output")
class NativeTabbedCutoutObservationTests(unittest.TestCase):
    def test_actual_b_and_c_posts_leave_retained_bridges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, count, moves in (("B", 4, 46), ("C", 5, 50)):
                with self.subTest(name=name):
                    paths = [root / f"{name}-fresh-manual.{ext}" for ext in ("cb", "nc")]
                    for path in paths:
                        shutil.copyfile(FIXTURES / path.name, path)
                    self.assertEqual(tuple(hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in paths), ACCEPTED_HASHES[name])
                    case = Case(*paths, count)
                    report = write_bundle(root / name, case)
                    self.assertEqual(report, audit_bundle(root / name / "handoff.json", case))
                    self.assertEqual(report["native_posted_moves"], moves)
                    self.assertEqual(report["bridges"]["count"], count)
                    self.assertTrue(report["bridges"]["retained_part_connected"])
                    self.assertEqual(report["bridges"]["minimum_centerline_stock_width_mm"], 6)


if __name__ == "__main__":
    unittest.main()
