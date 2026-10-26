"""Relief policy consumers through complete decoded output and body gates."""

from dataclasses import replace
import hashlib
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from cambam_builder.cam_core import relief, relief_stock, surface3d
from cambam_builder.cam_core.occupancy import Box, ToolBand, ToolBody
from cambam_builder.cam_core.ordered_job import AxialLimits
from cambam_builder.cam_extensions import relief_finishing
from cambam_builder.integrations import ordered_output


def tool_body():
    return ToolBody("T1", (ToolBand("cutter", 0, 1, 0.25),
                           ToolBand("shank", 1, 1.5, 0.25),
                           ToolBand("holder", 1.5, 2, 0.4)))


def analytic_target():
    return relief.CubicRelief((-1.5, 0, 1.5, 1), 1,
                              (-0.5, -0.12, 0, 0.04))


def grid_target():
    return relief.HeightField((-0.5, 0, 0.5), (0, 0.5, 1),
        ((-0.4, -0.5, -0.4), (-0.5, -0.35, -0.5),
         (-0.4, -0.5, -0.4)), 1)


def proposal(target=None, strategy="guided", **options):
    settings = {"tool": tool_body(), "stepover_mm": 0.5,
                "sample_step_mm": 0.4, "safe_z_mm": 1,
                "feed_mm_min": 180, "plunge_mm_min": 60, "rpm": 9000,
                "axial_limits": AxialLimits(0.8, 0.8), "pitch_mm": 0.1,
                "contact_tolerance_mm": 0.003, "allowance_mm": 0.010037,
                "waterline_step_mm": 0.04, **options}
    return relief_finishing.generate(analytic_target() if target is None else target,
                                     strategy, **settings)


class ReliefFinishingTests(unittest.TestCase):
    def test_relief_output_does_not_import_optional_backends(self):
        # A fresh process prevents earlier planar tests from hiding an eager
        # import. The same test runs in source-free installed snapshots.
        script = textwrap.dedent('''
            import importlib.abc
            import sys
            class WithoutOptional(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in ('shapely', 'mcp'):
                        raise ImportError('optional backend imported: ' + fullname)
            sys.meta_path.insert(0, WithoutOptional())
            from tests.test_relief_finishing import proposal
            from cambam_builder.integrations.ordered_output import emit
            for dialect in ('uccnc', 'grbl'):
                files, report = emit(proposal(), dialect)
                assert report['stock_access_residual']['status'] == 'pass'
                assert report['tool_fixture_occupancy']['status'] == 'pass'
        ''')
        result = subprocess.run([sys.executable, '-c', script],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_analytic_and_two_dimensional_consumers_decode_both_dialects(self):
        for target in (analytic_target(), grid_target()):
            job = proposal(target)
            for dialect in ("uccnc", "grbl"):
                with self.subTest(target=type(target).__name__, dialect=dialect):
                    files, report = ordered_output.emit(job, dialect)
                    self.assertTrue(files)
                    self.assertEqual(report["motion_equivalence"]["status"], "pass")
                    self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")
                    stock = report["stock_access_residual"]
                    self.assertEqual(stock["status"], "pass")
                    self.assertEqual(stock["design_fingerprint"], target.fingerprint)
                    self.assertGreater(stock["prefixes"][0]["newly_removed_volume_lower_mm3"], 0)
                    self.assertEqual(report["runtime_parity"]["status"], "not_evaluated")

    def test_parallel_guided_and_waterline_are_distinct_complete_candidates(self):
        jobs = {strategy: proposal(strategy=strategy)
                for strategy in ("parallel", "guided", "waterline")}
        cuts = {strategy: [move for move in job.stages[0].motions if move.role == "cut"]
                for strategy, job in jobs.items()}
        self.assertTrue(all(move.start[1] == move.end[1] for move in cuts["parallel"]))
        self.assertTrue(all(move.start[0] == move.end[0] for move in cuts["guided"]))
        self.assertTrue(all(move.start[2] == move.end[2] for move in cuts["waterline"]))
        self.assertTrue(all(cuts.values()))
        for job in jobs.values():
            _, report = ordered_output.emit(job, "uccnc")
            self.assertEqual(report["stock_access_residual"]["status"], "pass")
            cost = relief_finishing.motion_cost(job, rapid_mm_min=1000)
            self.assertGreater(cost["estimated_minutes"], 0)
            self.assertGreater(cost["feed_distance_mm"], 0)
        flat = relief.CubicRelief((0, 0, 1, 1), 1, (-0.5, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "no paths"):
            proposal(flat, strategy="waterline")

    def test_grid_creases_are_explicit_guided_stations(self):
        job = proposal(grid_target())
        cuts = [move for move in job.stages[0].motions if move.role == "cut"]
        along_x = any(move.start[0] != move.end[0] for move in cuts)
        stations = {move.start[1 if along_x else 0] for move in cuts}
        self.assertIn(0.5 if along_x else 0, stations)

    def test_stock_replay_receives_decoded_rounded_coordinates(self):
        job = proposal()
        # Supplied proposals may retain finer coordinates than the strategy's
        # own four-decimal paths. Match tolerance must not replace actual stock.
        def raised(point):
            return (*point[:2], point[2] + 0.00003) if point[2] < 0 else point

        stage = job.stages[0]
        supplied = tuple(replace(move, start=raised(move.start), end=raised(move.end))
                         for move in stage.motions)
        job = replace(job, stages=(replace(stage, motions=supplied),))
        captured = []
        original = relief_stock.replay_stages

        def observe(stages, motions):
            captured.extend(motions[0])
            return original(stages, motions)

        with patch.object(relief_stock, "replay_stages", side_effect=observe):
            ordered_output.emit(job, "uccnc")
        intended = job.stages[0].motions
        self.assertEqual(len(captured), len(intended))
        self.assertTrue(any(actual.end != wanted.end
                            for actual, wanted in zip(captured, intended)))
        for actual, wanted in zip(captured, intended):
            self.assertEqual(actual.end, tuple(round(value, 4) for value in wanted.end))

    def test_repeated_stage_replays_union_without_double_count(self):
        job = proposal()
        for dialect, boundary in (("uccnc", "split"), ("grbl", "pause")):
            combined = relief_finishing.sequence(job, job, boundary=boundary)
            _, report = ordered_output.emit(combined, dialect)
            first, second = report["stock_access_residual"]["prefixes"]
            self.assertEqual(first["residual_volume_mm3"], second["residual_volume_mm3"])
            self.assertEqual(second["newly_removed_volume_lower_mm3"], 0)
            self.assertEqual(len(report["transition_evidence"]["completion_assumed_not_observed"]), 1)

    def test_stale_design_source_and_stockless_revision_are_independent_gates(self):
        job = proposal()
        stale_stage = replace(job.stages[0], source_revision="stale")
        for candidate in (replace(job, source_fingerprint="other"),
                          replace(job, stages=(stale_stage,)),
                          replace(job, stages=(stale_stage,), occupancy_setup=None,
                                  stock_present=False)):
            with self.subTest(candidate=candidate.source_fingerprint), self.assertRaisesRegex(
                    ValueError, "stale relief"):
                ordered_output.emit(candidate, "uccnc")
        _, report = ordered_output.emit(replace(job, occupancy_setup=None,
                                                stock_present=False), "uccnc")
        self.assertEqual(report["stock_access_residual"]["status"], "not_evaluated")

    def test_complete_byte_corruption_rejects(self):
        job = proposal()
        files, _ = ordered_output.emit(job, "uccnc")
        altered = (files[0] + b"G0 X9 Y9 Z9\n",)
        with self.assertRaises(ValueError):
            ordered_output.audit_files(job, "uccnc", altered)
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            ordered_output.audit_files(job, "uccnc", altered,
                expected_hashes=(hashlib.sha256(files[0]).hexdigest(),))

    def test_physical_stock_cutter_fixture_and_process_mutations_reject(self):
        job = proposal()
        setup = job.occupancy_setup
        small_cutter = replace(tool_body(), bands=(
            replace(tool_body().bands[0], radius_mm=0.1),
            *tool_body().bands[1:]))
        wrong_stock = replace(setup.stock, bounds=(-2, 0, -1, 2, 1, 0))
        clamp = Box("clamp", (-0.2, 0.2, 0.1, 0.2, 0.8, 0.5))
        for changed, message in (
                (replace(setup, tools=(small_cutter,)), "cutter differs"),
                (replace(setup, stock=wrong_stock), "stock differs"),
                (replace(setup, fixtures=(clamp,)), "fixture")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                ordered_output.emit(replace(job, occupancy_setup=changed), "uccnc")
        limited = replace(job.stages[0], axial_limits=AxialLimits(0.2, 0.2))
        with self.assertRaisesRegex(ValueError, "depth limit"):
            ordered_output.emit(replace(job, stages=(limited,)), "uccnc")

    def test_mixed_relief_evaluator_returns_explicit_unsupported(self):
        job = proposal()
        combined = relief_finishing.sequence(job, job)
        second = combined.stages[1]
        op = surface3d.SurfaceOperation(second.id, second.tool_id, 0.25, 1,
            surface3d.SlopedTarget((-1.5, 0, 1.5, 1), 1, 0.5, 0.1), "surface_follow")
        mixed = replace(combined, stages=(combined.stages[0],
                                         replace(second, operation=op)))
        _, report = ordered_output.emit(mixed, "uccnc")
        self.assertEqual(report["stock_access_residual"]["status"], "unsupported")

    def test_sequence_rejects_incompatible_metadata_and_tool_bodies(self):
        job = proposal()
        other_frame = replace(job, program_frame="other",
            occupancy_setup=replace(job.occupancy_setup, frame="other"))
        different_body = replace(tool_body(), bands=(
            *tool_body().bands[:2], replace(tool_body().bands[2], radius_mm=0.5)))
        body_job = replace(job, occupancy_setup=replace(job.occupancy_setup,
                                                       tools=(different_body,)))
        candidates = (
            replace(job, translation_xyz_mm=(0, 0, 1)), other_frame,
            replace(job, source_kind="native"),
            replace(job, source_fingerprint="other"),
            replace(job, stock_present=False, occupancy_setup=None), body_job,
            replace(job, occupancy_setup=replace(job.occupancy_setup,
                stock=Box("other", (-2, 0, -1, 2, 1, 0)))),
            replace(job, occupancy_setup=replace(job.occupancy_setup,
                fixtures=(Box("far", (4, 4, 0, 5, 5, 1)),))),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                relief_finishing.sequence(job, candidate)
        with self.assertRaises(ValueError):
            relief_finishing.sequence(job, boundary="unmodeled")

    def test_generation_refuses_missing_process_and_unusable_clearance(self):
        for options, message in (({"axial_limits": None}, "axial"),
                                 ({"allowance_mm": 1e-6}, "rounding"),
                                 ({"safe_z_mm": 0}, "safe Z"),
                                 ({"feed_mm_min": True}, "feed")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                proposal(**options)


if __name__ == "__main__":
    unittest.main()
