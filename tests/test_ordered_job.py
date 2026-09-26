"""Caller-supplied ordered job, decoded output and stock-prefix regressions."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from cambam_builder.cam_core import replay, v_region
from cambam_builder.cam_core.ordered_job import (
    Job, JobMove, Stage, Transition, audit, from_prior_v,
)
from cambam_builder.integrations import ordered_dialects
from cambam_builder.integrations import m4_curved_workflow as m4
from cambam_builder.integrations.ordered_output import (
    audit_bundle, audit_files, emit, write_bundle,
)
from cambam_builder.integrations.cambam.native_ordered_job import (
    NativeBinding, from_native_series,
)
from cambam_builder.integrations.cambam.native_series import normalize_native_series
from cambam_builder.integrations.cambam import native_curved_rest as m2
from cambam_builder.native.core import Vertex
from cambam_builder.native.reader import read_cambam_bytes
from tests import test_native_series_audit as native_fixture


def _rectangle(fill="raster"):
    shell = ((0, 0), (10, 0), (10, 10), (0, 10))
    target = v_region.VTarget.polygon("rectangle-v-source", shell, (), 1)
    plan = v_region.plan(target, v_region.VProfile("rounded", 60, 0.5, 4, 3),
                         stepover_mm=2, xy_step_mm=1, safe_z=3,
                         fill_pattern=fill)
    start = (-2, -2, 3)
    op = replay.Operation("rough", replay.ToolProfile("T7", "cylinder", 0.5, 3),
                          replay.Target("rectangle", (0, 0, 10, 10), 1,
                                        region_shell=shell))
    items = (
        replay.Event("tool_change", "T7", start),
        replay.Event("spindle_start", "T7", start),
        replay.Motion("rapid", "T7", "rough", start, (5, 5, 3)),
        replay.Motion("entry", "T7", "rough", (5, 5, 3), (5, 5, -1), 45),
        replay.Motion("cut", "T7", "rough", (5, 5, -1), (6, 5, -1), 275),
        replay.Motion("retract", "T7", "rough", (6, 5, -1), (6, 5, 3), 175),
        replay.Motion("rapid", "T7", "rough", (6, 5, 3), start),
        replay.Event("spindle_stop", "T7", start),
    )
    prior = replay.Trace(target.source_id, "program", start, (op,), items)
    return plan, prior, start


def _native(root, order=("FIRST", "SECOND", "THIRD"), boundary="split"):
    fixture = native_fixture.NativeSeriesAuditTests()
    candidate, post = fixture.make_three_cut_case(root)
    fixture.write_three_cut_post(post, order)
    series = normalize_native_series(candidate, candidate, post,
                                     initial_position=(5, 5, 5))
    target = replay.Target("opening", (0, 0, 10, 10), 1,
                           region_shell=((0, 0), (10, 0), (10, 10), (0, 10)))
    job = from_native_series(
        series, targets={name: target for name in order},
        cutting_lengths_mm={"T1": 2, "T2": 2},
        entry_modes={name: ("cleared" if name == "SECOND" else "virgin")
                     for name in order}, boundary=boundary)
    return candidate, post, series, job


class OrderedJobTests(unittest.TestCase):
    def test_edited_annulus_selected_strategies_share_output_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "original.cb"
            m2.synthetic_source("annulus").save(str(original))
            project = read_cambam_bytes(original.read_bytes())
            hole = project.get_primitive("curved-finish").hole_curves[0]
            hole.vertices = [Vertex(v.x * 1.05, v.y * 1.05, v.z,
                                    bulge=v.bulge) for v in hole.vertices]
            edited = root / "edited.cb"
            project.save(str(edited))
            comparison = root / "comparison"
            m4.build_comparison(comparison, source_path=edited,
                                synthetic_prior=True)
            source = comparison / "comparison.json"
            for fill in ("raster", "offset"):
                plan, prior, start = m4.load_selected_plan(source, fill)
                for dialect, boundary in (("uccnc", "split"),
                                          ("grbl", "pause")):
                    with self.subTest(fill=fill, dialect=dialect):
                        job = from_prior_v(plan, prior, start,
                                           tool_id="T23", rpm=12000,
                                           boundary=boundary)
                        files, report = emit(job, dialect)
                        self.assertEqual(report["stock_access_residual"]["status"],
                                         "pass")
                        self.assertEqual(report["motion_equivalence"]["moves"][0],
                                         61)
                        self.assertGreater(len(files[0]), 1000)

    def test_two_existing_v_strategies_two_dialects_and_different_target(self):
        results = {}
        for fill in ("raster", "offset"):
            plan, prior, start = _rectangle(fill)
            for dialect, boundary in (("uccnc", "split"), ("grbl", "pause")):
                with self.subTest(fill=fill, dialect=dialect):
                    job = from_prior_v(plan, prior, start, tool_id="T19",
                                       rpm=11750, entry_feed=47.5,
                                       cut_feed=282.5,
                                       translation_xyz_mm=(2, -3, 1),
                                       boundary=boundary)
                    files, report = emit(job, dialect)
                    self.assertEqual(report["motion_equivalence"]["status"], "pass")
                    self.assertEqual(report["stock_access_residual"]["status"], "pass")
                    self.assertEqual(len(files), 2 if dialect == "uccnc" else 1)
                    self.assertEqual(report["stock_access_residual"]["cuts_by_prefix"],
                                     (2,))
                    results[(fill, dialect)] = report["stock_access_residual"][
                        "section_1_mm2"]
        self.assertEqual(results[("raster", "uccnc")],
                         results[("raster", "grbl")])
        self.assertEqual(results[("offset", "uccnc")],
                         results[("offset", "grbl")])
        self.assertNotEqual(results[("raster", "grbl")],
                            results[("offset", "grbl")])

    def test_repeated_tool_native_source_through_both_outputs_and_freshness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, post, series, job = _native(root)
            binding = NativeBinding(series, candidate, candidate, post)
            split = root / "ordered-split"
            result = write_bundle(split, job, "uccnc",
                                  source_binding=binding)
            self.assertEqual(result["stock_access_residual"]["cuts_by_prefix"],
                             (2, 3, 5))
            self.assertEqual(audit_bundle(split / "handoff.json", job,
                                          source_binding=binding), result)
            self.assertEqual([stage.tool_id for stage in job.stages],
                             ["T1", "T2", "T1"])
            grbl = replace(job, stages=(job.stages[0],) + tuple(
                replace(stage, transition=replace(stage.transition,
                                                  boundary="pause"))
                for stage in job.stages[1:]))
            _, report = emit(grbl, "grbl", source_binding=binding)
            self.assertEqual(report["stock_access_residual"]["cuts_by_prefix"],
                             (2, 3, 5))
            bad_operation = replace(job.stages[0].operation,
                tool=replace(job.stages[0].operation.tool, radius=1.5))
            changed_stage = replace(job.stages[0], operation=bad_operation)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                emit(replace(job, stages=(changed_stage,) + job.stages[1:]),
                     "uccnc", source_binding=binding)
            bad_file = (split / "stage-2.nc")
            original = bad_file.read_bytes()
            data = original.replace(b"X3", b"X4", 1)
            bad_file.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                audit_bundle(split / "handoff.json", job,
                             source_binding=binding)
            with self.assertRaisesRegex(ValueError, "source|setup"):
                audit_bundle(split / "handoff.json",
                             replace(job, source_fingerprint="changed"),
                             source_binding=binding)
            bad_file.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "freshness binding"):
                audit_bundle(split / "handoff.json", job)
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit_bundle(split / "handoff.json", job,
                             source_binding=binding)

    def test_offset_physical_tip_and_decoded_extra_travel(self):
        plan, prior, start = _rectangle()
        job = from_prior_v(plan, prior, start, boundary="pause")
        first = replace(job.stages[0], offset_mm=1.5, tool_length_mm=1.5)
        second = replace(job.stages[1], offset_mm=2.5, tool_length_mm=2.5)
        job = replace(job, stages=(first, second))
        files, report = emit(job, "grbl")
        self.assertEqual(report["stock_access_residual"]["status"], "pass")
        _, zero_offset = emit(from_prior_v(plan, prior, start,
                                           boundary="pause"), "grbl")
        self.assertEqual(report["stock_access_residual"]["section_1_mm2"],
                         zero_offset["stock_access_residual"]["section_1_mm2"])
        self.assertEqual(report["stock_access_residual"]["volume_mm3"],
                         zero_offset["stock_access_residual"]["volume_mm3"])
        decoded = ordered_dialects.decode(files, "grbl",
                                           initial_work_tip=job.initial_tip)
        self.assertEqual([stage.offset_mm for stage in decoded.stages],
                         [1.5, 2.5])
        self.assertEqual([move.start[2] for stage in decoded.stages
                          for move in stage.transition_moves], [1.5, 2])
        self.assertEqual([move.end[2] for stage in decoded.stages
                          for move in stage.transition_moves], [3, 3])
        with self.assertRaisesRegex(ValueError, "effective tool-tip"):
            audit(replace(job, stages=(first, replace(second,
                  tool_length_mm=2))), decoded, dialect="grbl")
        changed = files[0].replace(b"G43.1 Z2.5", b"G43.1 Z2")
        with self.assertRaises(ValueError):
            audit_files(job, "grbl", (changed,))

    def test_stockless_and_unresolved_motion_capability(self):
        move = (JobMove("rapid", (0, 0, 2), (1, 0, 2)),
                JobMove("rapid", (1, 0, 2), (0, 0, 2)))
        stage = Stage("unknown-path", "T21", move, 1000,
                      source_revision="native-source")
        job = Job("native-source", (stage,), (0, 0, 2), stock_present=False)
        _, report = emit(job, "uccnc")
        self.assertEqual(report["motion_equivalence"]["status"], "pass")
        self.assertEqual(report["stock_access_residual"]["status"],
                         "not_evaluated")
        _, report = emit(replace(job, stock_present=True), "uccnc")
        self.assertEqual(report["stock_access_residual"]["status"],
                         "unsupported")

    def test_stockless_native_series_keeps_motion_without_stock_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            candidate, post, _, _ = _native(Path(temp))
            tree = ET.parse(candidate)
            part = tree.find("./parts/part")
            part.remove(part.find("Stock"))
            options = tree.find("./MachiningOptions")
            if options is not None and options.find("Stock") is not None:
                options.remove(options.find("Stock"))
            tree.write(candidate, encoding="utf-8", xml_declaration=True)
            series = normalize_native_series(candidate, candidate, post,
                                             initial_position=(5, 5, 5))
            target = replay.Target("opening", (0, 0, 10, 10), 1,
                                   region_shell=((0, 0), (10, 0),
                                                 (10, 10), (0, 10)))
            job = from_native_series(
                series, targets={name: target for name in
                                 ("FIRST", "SECOND", "THIRD")},
                cutting_lengths_mm={"T1": 2, "T2": 2},
                entry_modes={"FIRST": "virgin", "SECOND": "cleared",
                             "THIRD": "virgin"}, stock_present=False)
            _, report = emit(job, "uccnc", source_binding=NativeBinding(
                series, candidate, candidate, post))
            self.assertEqual(report["document_fidelity"]["status"], "pass")
            self.assertEqual(report["motion_equivalence"]["status"], "pass")
            self.assertEqual(report["stock_access_residual"]["status"],
                             "not_evaluated")

    def test_synthetic_external_effect_must_match_and_clear_fixture(self):
        plan, prior, start = _rectangle()
        job = from_prior_v(plan, prior, start, boundary="pause")
        points = (start, (-3, -2, 3), start)
        effect = {"version": "ordered-effect-v1", "stage_id": "v-finish",
                  "tool_id": "T3", "model": "synthetic-host-v1",
                  "travel_program_tip_xyz_mm": [list(point) for point in points]}
        stage = replace(job.stages[1], transition=Transition(
            "synthetic_host", "pause", "T3", start,
            "host-completion-assertion", points, "synthetic-host-v1"))
        job = replace(job, stages=(job.stages[0], stage))
        data = (json.dumps(effect) + "\n").encode("utf-8")
        files, report = emit(job, "grbl", effects={stage.id: data})
        self.assertEqual(report["transition_evidence"]["status"],
                         "pass_with_assumptions")
        with self.assertRaisesRegex(ValueError, "missing"):
            audit_files(job, "grbl", files)
        effect["travel_program_tip_xyz_mm"][1][2] = -1
        with self.assertRaisesRegex(ValueError, "differs"):
            audit_files(job, "grbl", files,
                        effects={stage.id: json.dumps(effect).encode()})
        unsafe_points = (start, (-3, -2, -0.1), start)
        unsafe_stage = replace(stage, transition=replace(
            stage.transition, travel=unsafe_points))
        unsafe_job = replace(job, stages=(job.stages[0], unsafe_stage))
        effect["travel_program_tip_xyz_mm"] = [list(point)
                                                 for point in unsafe_points]
        with self.assertRaisesRegex(ValueError, "contacts"):
            emit(unsafe_job, "grbl", effects={stage.id:
                 json.dumps(effect).encode()})
        with self.assertRaisesRegex(ValueError, "transition"):
            replace(job, stages=(job.stages[0], replace(
                stage, transition=replace(stage.transition,
                                          resume_tip=(0, 0, 3)))))


if __name__ == "__main__":
    unittest.main()
