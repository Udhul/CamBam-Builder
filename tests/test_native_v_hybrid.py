"""One posted native cylinder followed by generated, decoded V cleanup."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from cambam_builder.cam_core import replay, v_region
from cambam_builder.integrations.cambam.native_ordered_job import (
    NativeBinding, from_native_v,
)
from cambam_builder.integrations.cambam.native_series import normalize_native_series
from cambam_builder.integrations.ordered_output import audit_bundle, write_bundle
from cambam_builder.native.reader import read_cambam_bytes
from tests import test_native_series as native_fixture


POST = """( Made using CamBam )
( candidate test )
( Post processor: Default )
G21 G90 G61 G40
G0 Z5
T1 M6
( FIRST )
G17
M3 S12000
G1 F60 Z-2
G1 F240 X6
G0 Z5
M5
M30
"""


class NativeVHybridTests(unittest.TestCase):
    def test_native_prefix_generated_v_and_edit_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate, post = native_fixture.NativeSeriesTests().make_case(root)
            project = read_cambam_bytes(candidate.read_bytes())
            self.assertTrue(project.remove_mop(project.list_mops()[1].internal_id))
            project.get_part("Part").stock_thickness = 2
            project.list_mops()[0].target_depth = -2
            project.list_mops()[0].depth_increment = 2
            project.save(str(candidate))
            source = root / "source.cb"
            original = read_cambam_bytes(candidate.read_bytes())
            self.assertTrue(original.remove_mop(original.list_mops()[0].internal_id))
            original.save(str(source))
            post.write_text(POST, encoding="utf-8")
            setup = {"units": "mm", "postprocessor": "Default"}
            series = normalize_native_series(source, candidate, post,
                                             initial_position=(5, 5, 5),
                                             setup=setup)
            target = replay.Target(
                "opening", (0, 0, 10, 10), 2,
                region_shell=((0, 0), (10, 0), (10, 10), (0, 10)))
            v_target = v_region.VTarget.polygon(
                series.evidence_fingerprint, target.region_shell, (), 2)
            plan = v_region.plan(
                v_target, v_region.VProfile("rounded", 60, 0.5, 4, 3),
                stepover_mm=2, xy_step_mm=1, safe_z=5)
            job = from_native_v(series, plan, target=target,
                                cutting_length_mm=2, tool_id="T3")
            binding = NativeBinding(series, source, candidate, post, setup)
            report = write_bundle(root / "hybrid", job, "uccnc",
                                  source_binding=binding)
            self.assertEqual(report["stock_access_residual"]["status"], "pass")
            self.assertEqual(report["motion_equivalence"]["status"], "pass")
            self.assertEqual(report["document_fidelity"]["status"], "pass")
            self.assertEqual(report["stock_access_residual"]["cuts_by_prefix"], (2,))
            self.assertEqual(job.stages[0].motions[-1].role, "rapid_retract")
            stock = report["stock_access_residual"]
            self.assertLess(stock["section_1_mm2"][1],
                            stock["prior_section_1_mm2"][0])
            self.assertLess(stock["section_1_mm2"][2], 0.001)
            self.assertEqual(audit_bundle(root / "hybrid" / "handoff.json", job,
                                          source_binding=binding), report)
            self.assertNotEqual(job.prefixes[0], job.prefixes[1])
            with self.assertRaisesRegex(ValueError, "identity differs"):
                write_bundle(root / "rpm-forged", replace(
                    job, stages=(replace(job.stages[0], rpm=12500),
                                 job.stages[1])),
                    "uccnc", source_binding=binding)
            with self.assertRaisesRegex(ValueError, "stage count"):
                write_bundle(root / "forged", replace(job, stages=job.stages +
                             (replace(job.stages[-1], id="v-extra"),)),
                             "uccnc", source_binding=binding)
            with self.assertRaisesRegex(ValueError, "source-bound"):
                from_native_v(series, replace(plan, target=replace(
                    v_target, source_id="unrelated")), target=target,
                    cutting_length_mm=2)
            post.write_text(POST + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit_bundle(root / "hybrid" / "handoff.json", job,
                             source_binding=binding)
            post.write_text(POST, encoding="utf-8")
            original.get_primitive("opening").width = 11
            original.save(str(source))
            with self.assertRaisesRegex(ValueError, "geometry or stock changed"):
                audit_bundle(root / "hybrid" / "handoff.json", job,
                             source_binding=binding)


if __name__ == "__main__":
    unittest.main()
