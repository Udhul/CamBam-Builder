"""Evidence cannot be promoted by bypassing the byte/source/occupancy gates."""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from cambam_builder.cam_core import ordered_job, replay
from cambam_builder.cam_core.occupancy import Box, OccupancySetup, ToolBand, ToolBody
from cambam_builder.integrations import ordered_dialects, ordered_output
from cambam_builder.integrations.cambam.native_ordered_job import NativeBinding
from tests.test_ordered_job import _native, _rectangle
from tests.test_occupancy import with_setup
from tests.test_paired_inlay import synthetic_pair
from tests.test_surface3d import synthetic_job


class ExecutionEvidenceTests(unittest.TestCase):
    def test_decoded_values_do_not_establish_native_document_fidelity(self):
        with tempfile.TemporaryDirectory() as temp:
            candidate, post, series, job = _native(Path(temp))
            binding = NativeBinding(series, candidate, candidate, post)
            files, verified = ordered_output.emit(job, "uccnc", source_binding=binding)
            self.assertEqual(verified["document_fidelity"]["status"], "pass")
            decoded = ordered_dialects.decode(files, "uccnc",
                                               initial_work_tip=job.initial_tip)
            raw = ordered_job.audit(job, decoded, dialect="uccnc")
            self.assertEqual(raw["document_fidelity"]["status"], "not_evaluated")
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                ordered_output.audit_files(job, "uccnc", files, source_binding=binding)
            self.assertEqual(ordered_job.audit(job, decoded, dialect="uccnc")
                             ["document_fidelity"]["status"], "not_evaluated")
            with self.assertRaisesRegex(ValueError, "freshness binding"):
                ordered_output.audit_files(job, "uccnc", files,
                    source_binding=SimpleNamespace(check=lambda _: True))

    def test_native_frame_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as temp:
            candidate, post, series, job = _native(Path(temp))
            binding = NativeBinding(series, candidate, candidate, post)
            self.assertTrue(binding.check(job))
            with self.assertRaisesRegex(ValueError, "frame"):
                binding.check(replace(job, program_frame="unrelated-frame"))

    def test_prefix_identity_includes_program_frame(self):
        job = synthetic_job()
        changed = replace(job, program_frame="other-frame")
        self.assertNotEqual(job.fingerprint, changed.fingerprint)
        self.assertNotEqual(job.prefixes, changed.prefixes)

    def test_operator_transition_cannot_silently_discard_declared_motion(self):
        plan, prior, start = _rectangle()
        job = ordered_job.from_prior_v(plan, prior, start)
        transition = job.stages[1].transition
        self.assertEqual(transition.travel, ())
        for values in ({"travel": (start, (0, 0, -1), start)},
                       {"effect_model": "unverified-macro"}):
            with self.subTest(values=values), self.assertRaisesRegex(
                    ValueError, "operator.*effect"):
                replace(transition, **values)

    def test_low_level_audit_does_not_certify_external_effect_bytes(self):
        plan, prior, start = _rectangle()
        job = ordered_job.from_prior_v(plan, prior, start, boundary="pause")
        points = (start, (-3, -2, 3), start)
        second = replace(job.stages[1], transition=ordered_job.Transition(
            "synthetic_host", "pause", "T3", start, "asserted", points, "host-v1"))
        job = replace(job, stages=(job.stages[0], second))
        files = ordered_dialects.render(job, "grbl")
        decoded = ordered_dialects.decode(files, "grbl", initial_work_tip=start)
        result = ordered_job.audit(job, decoded, dialect="grbl")
        self.assertEqual(result["external_effects"]["status"], "not_evaluated")
        with self.assertRaisesRegex(ValueError, "missing or unmodeled external effect"):
            ordered_output.audit_files(job, "grbl", files)

    def test_mixed_inlay_stock_evaluator_is_explicitly_unsupported(self):
        job = synthetic_pair().female
        first = job.stages[0]
        op = replay.Operation("cylinder", replay.ToolProfile("T1", "cylinder", 0.1, 2),
                              replay.Target("stock", (-7, -7, 7, 7), 1))
        second = replace(first, id="cylinder", operation=op,
            transition=ordered_job.Transition("operator", "split", "T1",
                                              job.initial_tip, "asserted"))
        _, report = ordered_output.emit(replace(job, stages=(first, second)), "uccnc")
        self.assertEqual(report["stock_access_residual"], {
            "status": "unsupported", "reason": "unsupported stock evaluator sequence"})

    def test_inlay_cannot_silently_skip_supplied_occupancy(self):
        job = synthetic_pair().female
        _, valid = ordered_output.emit(job, "uccnc")
        self.assertEqual(valid["stock_access_residual"]["status"], "pass")
        body = ToolBody("T1", (ToolBand("cutter", 0, 1.1, 0.6),
                               ToolBand("shank", 1.1, 2, 0.6),
                               ToolBand("holder", 2, 3, 1)))
        setup = OccupancySetup(job.program_frame,
            Box("stock", (-7, -7, -1, 7, 7, 0)),
            (Box("blocking fixture", (-10, -10, -2, 10, 10, 6)),), (body,))
        with self.assertRaisesRegex(ValueError, "inlay.*occupancy.*unsupported"):
            ordered_output.emit(replace(job, occupancy_setup=setup), "uccnc")

    def test_body_cannot_extend_cutting_length_to_hide_shank(self):
        job = with_setup(synthetic_job())
        _, valid = ordered_output.emit(job, "uccnc")
        self.assertEqual(valid["tool_fixture_occupancy"]["status"], "pass")
        body = ToolBody("T1", (ToolBand("cutter", 0, 3, 0.5),
                               ToolBand("shank", 3, 3.5, 0.5),
                               ToolBand("holder", 3.5, 5.5, 0.8)))
        with self.assertRaisesRegex(ValueError, "occupancy cutter differs"):
            ordered_output.emit(replace(job, occupancy_setup=replace(
                job.occupancy_setup, tools=(body,))), "uccnc")

    def test_inlay_access_roles_cannot_hide_diagonal_stock_motion(self):
        job = synthetic_pair().female
        stage = job.stages[0]
        self.assertEqual(ordered_output.emit(job, "uccnc")[1]
                         ["stock_access_residual"]["status"], "pass")
        # At X=4 the diagonal still lies below Z=0, beyond the protected wall.
        high = (30, 0, 2)
        motions = list(stage.motions)
        self.assertEqual(motions[0].role, "entry")
        entry = (ordered_job.JobMove("rapid", job.initial_tip, high),
                 replace(motions[0], start=high), *motions[1:])
        self.assertEqual(motions[1].role, "retract")
        retract = (motions[0], replace(motions[1], end=high),
                   replace(motions[2], start=high), *motions[3:])
        for moves in (entry, retract):
            with self.subTest(role=moves[1].role), self.assertRaisesRegex(
                    ValueError, "vertical"):
                ordered_output.emit(replace(job, stages=(replace(
                    stage, motions=moves),)), "uccnc")


if __name__ == "__main__":
    unittest.main()
