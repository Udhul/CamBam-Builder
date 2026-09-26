"""One native-post-to-stock-to-strategy end-to-end linear slice."""

import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

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

    def make_three_cut_case(self, directory):
        candidate, post = fixture.NativeSeriesTests().make_case(directory)
        project = read_cambam_bytes(candidate.read_bytes())
        project.add_pocket_mop(
            project.get_part("Part"), targets=[project.get_primitive("opening")],
            name="THIRD", tool_number=1, tool_diameter=4,
            target_depth=-1, depth_increment=1, stock_surface=0,
            clearance_plane=5, spindle_speed=12000,
            plunge_feedrate=60, cut_feedrate=240, tool_profile="EndMill")
        project.save(str(candidate))
        tree = ET.parse(candidate)
        tree.getroot().set("units", "Millimeters")
        tree.write(candidate, encoding="utf-8", xml_declaration=True)
        return candidate, post

    @staticmethod
    def write_three_cut_post(post, order):
        # These are separate stock cuts, including the return to T1. Coordinates
        # are intentionally fixed independently of the native MOP tree order.
        paths = {"FIRST": (1, 5, 7), "SECOND": (2, 5, 3),
                 "THIRD": (1, 7, 7)}
        lines = ["( Made using CamBam )", "( candidate test )",
                 "( Post processor: Default )", "G21 G90 G61 G40", "G0 Z5"]
        for name in order:
            tool, y, end_x = paths[name]
            lines.extend((f"T{tool} M6", f"( {name} )", "G17", "M3 S12000",
                          f"G0 X5 Y{y}", "G1 F60 Z-1", f"G1 F240 X{end_x}",
                          "G1 F60 Z5", "M5"))
        post.write_text("\n".join((*lines, "M30", "")), encoding="utf-8")

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

    def test_native_predecessor_disable_and_reorder_require_fresh_stock_replay(self):
        target = replay.Target("opening", (0, 0, 10, 10), 1,
                               region_shell=((0, 0), (10, 0), (10, 10), (0, 10)))
        exclusive_points = {"FIRST": (8.5, 5), "SECOND": (2.5, 5),
                            "THIRD": (5, 8.5)}

        def assert_each_stage_removes_new_stock(audit):
            previous = 0
            for name, through in audit.stock.prefixes:
                x, y = exclusive_points[name]
                self.assertFalse(audit.stock.removed_contains(
                    x, y, 1, through=previous), name)
                self.assertTrue(audit.stock.removed_contains(
                    x, y, 1, through=through), name)
                previous = through

        for edit, expected_order, expected_prefixes in (
                ("disable", ("SECOND", "THIRD"),
                 (("SECOND", 2), ("THIRD", 4))),
                ("reorder", ("THIRD", "SECOND", "FIRST"),
                 (("THIRD", 2), ("SECOND", 4), ("FIRST", 6)))):
            with self.subTest(edit=edit), tempfile.TemporaryDirectory() as folder:
                candidate, post = self.make_three_cut_case(Path(folder))
                self.write_three_cut_post(post, ("FIRST", "SECOND", "THIRD"))
                old = normalize_native_series(candidate, candidate, post,
                                              initial_position=(5, 5, 5))
                old_args = dict(targets={name: target for name in
                                         ("FIRST", "SECOND", "THIRD")},
                                cutting_lengths_mm={"T1": 2, "T2": 2},
                                entry_modes={"FIRST": "virgin", "SECOND": "cleared",
                                             "THIRD": "virgin"},
                                section_depths_mm=(0.5, 1.0),
                                max_protected_overcut_mm2=0.001)
                old_audit = audit_linear_native_series(
                    old, candidate, candidate, post, **old_args)
                self.assertEqual(old_audit.stock.prefixes,
                                 (("FIRST", 2), ("SECOND", 3), ("THIRD", 5)))
                self.assertEqual([cut.operation for cut in old_audit.stock.cuts],
                                 ["FIRST", "FIRST", "SECOND", "THIRD", "THIRD"])
                assert_each_stage_removes_new_stock(old_audit)
                old_remaining = [stage.evidence.residual.area_upper_mm2
                                 for stage in old_audit.stages]
                self.assertTrue(all(later < earlier for earlier, later in
                                    zip(old_remaining, old_remaining[1:])))

                tree = ET.parse(candidate)
                machineops = tree.find("./parts/part/machineops")
                operations = list(machineops)
                if edit == "disable":
                    operations[0].set("Enabled", "false")
                else:
                    machineops[:] = operations[::-1]
                tree.write(candidate, encoding="utf-8", xml_declaration=True)
                with self.assertRaisesRegex(ValueError, "changed"):
                    old.check_freshness(candidate, candidate, post)
                with self.assertRaisesRegex(ValueError, "out of order|disabled MOP"):
                    normalize_native_series(candidate, candidate, post,
                                            initial_position=(5, 5, 5))

                self.write_three_cut_post(post, expected_order)
                fresh = normalize_native_series(candidate, candidate, post,
                                                initial_position=(5, 5, 5))
                fresh_args = dict(old_args,
                                  targets={name: target for name in expected_order},
                                  entry_modes={name: "virgin" for name in expected_order})
                fresh_audit = audit_linear_native_series(
                    fresh, candidate, candidate, post, **fresh_args)
                self.assertEqual(tuple(stage.name for stage in fresh.stages),
                                 expected_order)
                self.assertEqual(fresh_audit.stock.prefixes, expected_prefixes)
                assert_each_stage_removes_new_stock(fresh_audit)
                fresh_remaining = [stage.evidence.residual.area_upper_mm2
                                   for stage in fresh_audit.stages]
                self.assertTrue(all(later < earlier for earlier, later in
                                    zip(fresh_remaining, fresh_remaining[1:])))
                self.assertNotEqual(old.source_semantic_sha256,
                                    fresh.source_semantic_sha256)
                self.assertEqual(fresh_audit.stages[0].evidence.predecessor_fingerprint,
                                 fresh.source_semantic_sha256)
                self.assertTrue(all(
                    stage.evidence.predecessor_fingerprint ==
                    fresh_audit.stages[index - 1].evidence.chain_fingerprint
                    for index, stage in enumerate(fresh_audit.stages) if index))

                residual = fresh_audit.stages[-1].evidence.residual
                budget = dict(max_area_mm2=residual.area_upper_mm2 + 0.01,
                              max_volume_mm3=residual.volume_upper_mm3 + 0.01,
                              tie_order=("native",))
                stale = select_strategy(
                    fresh.source_semantic_sha256,
                    (old_audit.as_alternative("native"),), **budget)
                self.assertEqual(stale.status, "infeasible")
                self.assertIn("stale source fingerprint",
                              stale.assessments[0].reasons[0])
                replanned = select_strategy(
                    fresh.source_semantic_sha256,
                    (fresh_audit.as_alternative("native"),), **budget)
                self.assertEqual((replanned.status, replanned.chosen),
                                 ("selected", "native"))

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
