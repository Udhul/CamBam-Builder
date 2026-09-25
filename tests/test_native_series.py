"""Bounded native Default-post MOP-series normalization and replay bridge."""

import hashlib
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cam_core import replay
from cambam_builder.integrations.cambam.native_series import (
    normalize_native_series,
)
from cambam_builder.native.reader import read_cambam_bytes


POST = """( Made using CamBam )
( candidate test )
( Post processor: Default )
G21 G90 G61 G40
G0 Z5
T1 M6
( FIRST )
G17
M3 S12000
G1 F60 Z-1
G1 F240 X7
G1 F60 Z5
M5
T2 M6
( SECOND )
M3 S12000
G0 X5 Y5
G1 F60 Z-1
G1 F240 X6
G1 F60 Z5
M5
M30
"""


class NativeSeriesTests(unittest.TestCase):
    def make_case(self, directory):
        project = CBProject("series")
        layer = project.add_layer("Geometry")
        part = project.add_part("Part", stock_thickness=1, stock_width=10,
                                stock_height=10, stock_surface=0,
                                nesting_method="None")
        rect = project.add_rect(layer, (0, 0), 10, 10, identifier="opening")
        common = dict(target_depth=-1, depth_increment=1, stock_surface=0,
                      clearance_plane=5, spindle_speed=12000,
                      plunge_feedrate=60, cut_feedrate=240,
                      tool_profile="EndMill")
        project.add_pocket_mop(part, targets=[rect], name="FIRST",
                               tool_number=1, tool_diameter=4, **common)
        project.add_pocket_mop(part, targets=[rect], name="SECOND",
                               tool_number=2, tool_diameter=2, **common)
        candidate = directory / "candidate.cb"
        post = directory / "candidate.nc"
        project.save(str(candidate))
        content = candidate.read_text(encoding="utf-8")
        content = content.replace("<CADFile ", '<CADFile units="Millimeters" ', 1)
        content = content.replace("<MachiningOptions>",
                                  "<MachiningOptions><PostProcessor>Default</PostProcessor>", 1)
        candidate.write_text(content, encoding="utf-8")
        post.write_text(POST, encoding="utf-8")
        return candidate, post

    def test_two_stages_bind_post_and_replay_stock_prefixes(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, post = self.make_case(Path(folder))
            series = normalize_native_series(candidate, candidate, post,
                                             initial_position=(5, 5, 5))
            self.assertEqual([stage.name for stage in series.stages], ["FIRST", "SECOND"])
            self.assertEqual([stage.tool for stage in series.stages], ["T1", "T2"])
            self.assertEqual([stage.move_count for stage in series.stages], [3, 4])
            self.assertEqual(series.source_sha256,
                             hashlib.sha256(candidate.read_bytes()).hexdigest())
            self.assertEqual(series.post_sha256,
                             hashlib.sha256(post.read_bytes()).hexdigest())
            self.assertTrue(series.check_freshness(candidate, candidate, post))
            target = replay.Target("opening", (0, 0, 10, 10), 1)
            trace = series.to_trace({"FIRST": target, "SECOND": target},
                                    {"T1": 2, "T2": 2},
                                    {"FIRST": "virgin", "SECOND": "cleared"})
            stock = replay.replay(trace, expected_source=series.evidence_fingerprint)
            self.assertEqual(stock.prefixes, (("FIRST", 2), ("SECOND", 3)))
            self.assertTrue(stock.removed_contains(5, 5, 1, through=2))
            post.write_text(POST + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                series.check_freshness(candidate, candidate, post)

    def test_mop_free_cosmetic_source_edit_retains_semantic_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            candidate, post = self.make_case(directory)
            source = directory / "source.cb"
            project = read_cambam_bytes(candidate.read_bytes())
            for mop in tuple(project.list_mops()):
                self.assertTrue(project.remove_mop(mop.internal_id))
            project.save(str(source))
            setup = {"units": "mm", "postprocessor": "Default"}
            series = normalize_native_series(source, candidate, post,
                                             initial_position=(5, 5, 5),
                                             setup=setup)
            self.assertFalse(series.source_has_mops)
            text = source.read_text(encoding="utf-8")
            source.write_text(text.replace('Name="series"',
                                           'Name="series cosmetic"', 1),
                              encoding="utf-8")
            self.assertTrue(series.check_freshness(source, candidate, post,
                                                   setup=setup))
            edited = read_cambam_bytes(source.read_bytes())
            edited.get_primitive("opening").width = 11
            edited.save(str(source))
            with self.assertRaisesRegex(ValueError, "geometry or stock changed"):
                series.check_freshness(source, candidate, post, setup=setup)

    def test_sections_modal_words_and_tool_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, post = self.make_case(Path(folder))
            original = post.read_text(encoding="utf-8")
            variants = (
                (original.replace("( SECOND )", "( OTHER )"), "sections"),
                (original.replace("( SECOND )", "( FIRST )"), "sections"),
                (original.replace("G1 F240 X6", "G81 X6 Z-1 R2"), "unsupported"),
                (original.replace("G1 F240 X6", "T3 M6\nG1 F240 X6"), "tool"),
                (original.replace("( candidate test )", "( another test )"), "header"),
            )
            for changed, reason in variants:
                with self.subTest(reason=reason, changed=changed[-60:]):
                    post.write_text(changed, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, reason):
                        normalize_native_series(candidate, candidate, post,
                                                initial_position=(5, 5, 5))

    def test_lowering_rejects_unproved_arc_and_low_xy_rapid(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, post = self.make_case(Path(folder))
            target = replay.Target("opening", (0, 0, 10, 10), 1)
            args = ({"FIRST": target, "SECOND": target},
                    {"T1": 2, "T2": 2},
                    {"FIRST": "virgin", "SECOND": "cleared"})
            text = post.read_text(encoding="utf-8")
            post.write_text(text.replace("G1 F240 X7", "G2 F240 X7 Y5 I1 J0"),
                            encoding="utf-8")
            series = normalize_native_series(candidate, candidate, post,
                                             initial_position=(5, 5, 5))
            with self.assertRaisesRegex(ValueError, "arc needs"):
                series.to_trace(*args)
            post.write_text(text.replace("G1 F240 X7", "G0 X7"), encoding="utf-8")
            series = normalize_native_series(candidate, candidate, post,
                                             initial_position=(5, 5, 5))
            with self.assertRaisesRegex(ValueError, "low XY rapid"):
                series.to_trace(*args)

    def test_nonpositive_feed_or_spindle_never_lowers_to_stock_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, post = self.make_case(Path(folder))
            target = replay.Target("opening", (0, 0, 10, 10), 1)
            args = ({"FIRST": target, "SECOND": target},
                    {"T1": 2, "T2": 2},
                    {"FIRST": "virgin", "SECOND": "cleared"})
            for changed, reason in ((POST.replace("G1 F240 X7", "G1 F0 X7"),
                                     "nonpositive cutting feed"),
                                    (POST.replace("M3 S12000", "M3 S0", 1),
                                     "nonpositive spindle speed")):
                with self.subTest(reason=reason):
                    post.write_text(changed, encoding="utf-8")
                    series = normalize_native_series(candidate, candidate, post,
                                                     initial_position=(5, 5, 5))
                    with self.assertRaisesRegex(ValueError, reason):
                        series.to_trace(*args)
            altered = replace(series, stages=(replace(series.stages[0], kind="DrillMop"),
                                              series.stages[1]))
            with self.assertRaisesRegex(ValueError, "unsupported native stage"):
                altered.to_trace(*args)

    def test_source_primitive_and_stock_mutations_reject_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            candidate, post = self.make_case(directory)
            source = directory / "source.cb"
            original = candidate.read_text(encoding="utf-8")
            source.write_text(original, encoding="utf-8")
            self.assertIn('w="10"', original)
            candidate.write_text(original.replace('w="10"', 'w="11"', 1),
                                 encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "analytic geometry"):
                normalize_native_series(source, candidate, post,
                                        initial_position=(5, 5, 5))
            marker = '"user_id":"opening","internal_id":"'
            start = original.index(marker) + len(marker)
            end = original.index('"', start)
            candidate.write_text(original[:start] + str(uuid.uuid4()) + original[end:],
                                 encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "primitive identity"):
                normalize_native_series(source, candidate, post,
                                        initial_position=(5, 5, 5))
            self.assertIn("<PMax>10.0,10.0,0</PMax>", original)
            candidate.write_text(original.replace("<PMax>10.0,10.0,0</PMax>",
                                                  "<PMax>11.0,10.0,0</PMax>", 1),
                                 encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Part stock"):
                normalize_native_series(source, candidate, post,
                                        initial_position=(5, 5, 5))

    def test_retained_m1_actual_native_post_normalizes_without_stock_authority(self):
        directory = Path(__file__).resolve().parents[1] / "output" / "m1-polygon-20260924-04"
        source = directory / "source.cb"
        candidate = directory / "native" / "m1-native.cb"
        post = directory / "native" / "m1-native.nc"
        if not all(path.is_file() for path in (source, candidate, post)):
            self.skipTest("retained user-posted M1 output is unavailable")
        setup = {"units": "mm", "postprocessor": "Default"}
        with self.assertRaisesRegex(ValueError, "bound millimetre"):
            normalize_native_series(source, candidate, post,
                                    initial_position=(-30, -10, 5))
        series = normalize_native_series(source, candidate, post,
                                         initial_position=(-30, -10, 5), setup=setup)
        self.assertEqual([stage.name for stage in series.stages],
                         ["NATIVE T1 letter Pocket", "NATIVE T2 letter Pocket"])
        self.assertTrue(series.check_freshness(source, candidate, post, setup=setup))
        self.assertTrue(any(getattr(item, "g", None) in (2, 3) for item in series.items))
        with self.assertRaisesRegex(ValueError, "bound setup"):
            series.check_freshness(source, candidate, post)


if __name__ == "__main__":
    unittest.main()
