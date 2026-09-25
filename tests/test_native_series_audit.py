"""One native-post-to-stock-to-strategy end-to-end linear slice."""

import tempfile
import unittest
from pathlib import Path

from cambam_builder.cam_core import replay
from cambam_builder.cam_extensions.strategy import select_strategy
from cambam_builder.integrations.cambam.native_series import normalize_native_series
from cambam_builder.integrations.cambam.native_series_audit import (
    audit_linear_native_series,
)
from cambam_builder.native.reader import read_cambam_bytes
from tests import test_native_series as fixture


class NativeSeriesAuditTests(unittest.TestCase):
    def make_case(self, directory, posted):
        candidate, post = fixture.NativeSeriesTests().make_case(directory)
        post.write_text(posted, encoding="utf-8")
        series = normalize_native_series(candidate, candidate, post,
                                         initial_position=(5, 5, 5))
        target = replay.Target("opening", (0, 0, 10, 10), 1,
                               region_shell=((0, 0), (10, 0), (10, 10), (0, 10)))
        args = dict(targets={"FIRST": target, "SECOND": target},
                    cutting_lengths_mm={"T1": 2, "T2": 2},
                    entry_modes={"FIRST": "virgin", "SECOND": "cleared"},
                    section_depths_mm=(0.5, 1.0),
                    max_protected_overcut_mm2=0.001)
        return candidate, post, series, args

    def test_actual_linear_post_replays_each_prefix_and_selects_route(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, post, series, args = self.make_case(
                Path(folder), fixture.POST.replace("G1 F240 X6", "G1 F240 X9"))
            audit = audit_linear_native_series(series, candidate, candidate, post,
                                               **args)
            self.assertEqual(audit.stock.prefixes, (("FIRST", 2), ("SECOND", 3)))
            self.assertEqual(len(audit.stages), 2)
            first, second = audit.stages
            self.assertEqual([row.depth_mm for row in first.sections], [0.5, 1.0])
            self.assertGreater(first.evidence.residual.area_upper_mm2,
                               second.evidence.residual.area_upper_mm2)
            self.assertGreater(first.evidence.residual.volume_upper_mm3,
                               second.evidence.residual.volume_upper_mm3)
            self.assertEqual(second.evidence.predecessor_fingerprint,
                             first.evidence.chain_fingerprint)
            self.assertTrue(all(section.protected_overcut_upper_mm2 < 0.001
                                for stage in audit.stages for section in stage.sections))
            final = second.evidence.residual
            selected = select_strategy(series.source_sha256,
                                       (audit.as_alternative("native"),),
                                       max_area_mm2=final.area_upper_mm2 + 0.01,
                                       max_volume_mm3=final.volume_upper_mm3 + 0.01,
                                       tie_order=("native",))
            self.assertEqual((selected.status, selected.chosen), ("selected", "native"))
            self.assertEqual(len(selected.assessments[0].stage_residuals), 2)
            post.write_text(post.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit_linear_native_series(series, candidate, candidate, post,
                                           **args)

    def test_unsafe_access_and_arcs_never_acquire_stage_certificates(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            candidate, post, series, args = self.make_case(
                directory, fixture.POST.replace("G0 X5 Y5", "G0 X9 Y9"))
            with self.assertRaisesRegex(ValueError, "uncleared travel/access"):
                audit_linear_native_series(series, candidate, candidate, post,
                                           **args)
            post.write_text(fixture.POST.replace("G1 F240 X7", "G2 F240 X7 Y5 I1 J0"),
                            encoding="utf-8")
            series = normalize_native_series(candidate, candidate, post,
                                             initial_position=(5, 5, 5))
            with self.assertRaisesRegex(ValueError, "arc needs continuous sweep proof"):
                audit_linear_native_series(series, candidate, candidate, post,
                                           **args)

    def test_substituted_larger_target_cannot_receive_stock_certificate(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, post, series, args = self.make_case(Path(folder), fixture.POST)
            larger = replay.Target("opening", (-1, -1, 11, 11), 1,
                                   region_shell=((-1, -1), (11, -1),
                                                 (11, 11), (-1, 11)))
            args["targets"] = {"FIRST": larger, "SECOND": larger}
            with self.assertRaisesRegex(ValueError, "differs from original native source"):
                audit_linear_native_series(series, candidate, candidate, post,
                                           **args)

    def test_omitting_floor_section_cannot_understate_final_rest(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate, post, series, args = self.make_case(Path(folder), fixture.POST)
            args["section_depths_mm"] = (0.5,)
            with self.assertRaisesRegex(ValueError, "include the target floor"):
                audit_linear_native_series(series, candidate, candidate, post,
                                           **args)

    def test_cosmetic_source_edit_keeps_audited_strategy(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            candidate, post, _, args = self.make_case(directory, fixture.POST)
            source = directory / "source.cb"
            project = read_cambam_bytes(candidate.read_bytes())
            for mop in tuple(project.list_mops()):
                project.remove_mop(mop.internal_id)
            project.save(str(source))
            setup = {"units": "mm", "postprocessor": "Default"}
            series = normalize_native_series(source, candidate, post,
                                             initial_position=(5, 5, 5), setup=setup)
            source.write_text(source.read_text(encoding="utf-8").replace(
                'Name="series"', 'Name="series renamed"', 1), encoding="utf-8")
            audit = audit_linear_native_series(series, source, candidate, post,
                                               setup=setup, **args)
            final = audit.stages[-1].evidence.residual
            choice = select_strategy(series.source_semantic_sha256,
                                     (audit.as_alternative("native"),),
                                     max_area_mm2=final.area_upper_mm2 + 0.01,
                                     max_volume_mm3=final.volume_upper_mm3 + 0.01,
                                     tie_order=("native",))
            self.assertEqual((choice.status, choice.chosen), ("selected", "native"))


if __name__ == "__main__":
    unittest.main()
