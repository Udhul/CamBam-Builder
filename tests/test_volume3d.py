"""Stepped 3D stock, waterline and dependent rest through decoded output."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

from cambam_builder.cam_core.ordered_job import Job, JobMove, Stage, Transition
from cambam_builder.cam_core.volume3d import (
    LayeredTarget, VolumeOperation, _sweep, compare_representations,
)
from cambam_builder.integrations.ordered_output import (
    audit_bundle, emit, write_bundle,
)


def synthetic_job(boundary="split"):
    target = LayeredTarget((0, 0, 8, 6), 3,
                           ((1, 1, 3, 5, 1), (3.4, 1, 6, 5, 2)),
                           (3, 1, 3.4, 5))
    start = (0, 0, 3)

    def path(motions, at, points, depth, *, cleared=False):
        first = (points[0][0], points[0][1], 3)
        if at != first:
            motions.append(JobMove("rapid", at, first))
        low = (first[0], first[1], -depth)
        motions.append(JobMove("cleared_descent" if cleared else "entry",
                               first, low, 60))
        at = low
        for x, y in points[1:]:
            nxt = (x, y, -depth)
            motions.append(JobMove("cut", at, nxt, 300))
            at = nxt
        high = (at[0], at[1], 3)
        motions.append(JobMove("retract", at, high, 120))
        return high

    first = []
    at = start
    for x0, x1, depth in ((1, 3, 1), (3.4, 6, 1),
                          (3.4, 6, 2)):
        x0 += 0.21
        x1 -= 0.21
        at = path(first, at, ((x0, 1.21), (x1, 1.21),
                              (x1, 4.79), (x0, 4.79), (x0, 1.21)), depth)
    first.append(JobMove("rapid", at, start))
    second = []
    at = start
    for x0, x1, depth in ((1.21, 2.79, 1), (3.61, 5.79, 2)):
        count = round((x1 - x0) / 0.35)
        for i in range(count + 1):
            x = round(x0 + (x1 - x0) * i / count, 4)
            ends = ((x, 1.21), (x, 4.79)) if i % 2 == 0 else (
                (x, 4.79), (x, 1.21))
            at = path(second, at, ends, depth, cleared=i == 0)
    second.append(JobMove("rapid", at, start))
    stages = (
        Stage("contours", "T1", tuple(first), 10000,
              operation=VolumeOperation("contours", "T1", 0.2, 2.5,
                                        target, "waterline"),
              source_revision="stepped-prisms-v1"),
        Stage("rest", "T2", tuple(second), 10000,
              operation=VolumeOperation("rest", "T2", 0.18, 2.5,
                                        target, "rest"),
              source_revision="stepped-prisms-v1",
              transition=Transition("operator", boundary, "T2", start,
                                    "verified-stage-boundary")),
    )
    return Job("stepped-prisms-v1", stages, start)


class Volume3DTests(unittest.TestCase):
    def test_tiny_prisms_keep_disjoint_volume_and_column_enclosure(self):
        # The exact union cannot use an area tolerance to accept overlap:
        # even a tiny shared rectangle would be counted twice in volume.
        stock = (0, 0, 1, 1)
        protected = (0.8, 0.8, 0.9, 0.9)
        narrow = (0.1, 0.1, 0.10000001, 0.2, 1)
        target = LayeredTarget(stock, 1, (narrow,), protected)
        exact = 1e-9
        self.assertAlmostEqual(target.target_volume_mm3, exact, delta=1e-17)
        low, high = compare_representations(target, (0.5,))[1][
            "volume_interval_mm3"]
        self.assertLessEqual(low, exact)
        self.assertGreaterEqual(high, exact)
        self.assertGreater(high, 0)

        first = (0.1, 0.1, 0.5, 0.5, 1)
        second = (0.49999996, 0.1, 0.7, 0.5, 1)
        with self.assertRaisesRegex(ValueError, "overlap"):
            LayeredTarget(stock, 1, (first, second), protected)
        touching_protected = (0.49999996, 0.2, 0.6, 0.3)
        with self.assertRaisesRegex(ValueError, "protected"):
            LayeredTarget(stock, 1, (first,), touching_protected)
        for invalid_depth in (float("nan"), float("inf"), True):
            with self.assertRaisesRegex(ValueError, "section depth"):
                target.section(invalid_depth)

    def test_cutter_sweep_brackets_independent_capsule_formula(self):
        a, b, radius = (2, 2, -1), (4, 2, -1), 0.2
        analytic = 2 * radius * 2 + math.pi * radius ** 2
        inner = _sweep(a, b, radius, outer=False)
        outer = _sweep(a, b, radius, outer=True)
        self.assertLess(inner.area, analytic)
        self.assertGreater(outer.area, analytic)
        self.assertLess(outer.area - inner.area, 0.001)

    def test_analytic_geometry_representation_and_decoded_rest(self):
        job = synthetic_job()
        target = job.stages[0].volume_operation.target
        self.assertAlmostEqual(target.target_volume_mm3, 28.8)
        self.assertAlmostEqual(target.section(0.5).area, 18.4)
        self.assertAlmostEqual(target.section(1.5).area, 10.4)
        comparison = compare_representations(target)
        self.assertEqual(comparison[0]["volume_interval_mm3"], (28.8, 28.8))
        for row in comparison[1:]:
            lo, hi = row["volume_interval_mm3"]
            self.assertLessEqual(lo, 28.8 + 1e-7)
            self.assertGreaterEqual(hi, 28.8 - 1e-7)
            self.assertGreater(row["elapsed_ms"], 0)
            self.assertGreater(row["python_peak_bytes"], 0)
        self.assertLess(
            comparison[2]["volume_interval_mm3"][1] -
            comparison[2]["volume_interval_mm3"][0],
            comparison[1]["volume_interval_mm3"][1] -
            comparison[1]["volume_interval_mm3"][0])
        for dialect, boundary in (("uccnc", "split"), ("grbl", "pause")):
            with self.subTest(dialect=dialect):
                files, report = emit(synthetic_job(boundary), dialect)
                self.assertEqual(len(files), 2 if dialect == "uccnc" else 1)
                self.assertEqual(report["motion_equivalence"]["status"], "pass")
                stock = report["stock_access_residual"]
                self.assertEqual(stock["status"], "pass")
                self.assertEqual(stock["model"], "layered-volume-v1")
                prior, final = stock["prefixes"]
                self.assertGreater(prior["residual_volume_mm3"][0],
                                   final["residual_volume_mm3"][1])
                self.assertGreater(final["residual_volume_mm3"][0], 0)
                self.assertEqual(prior["protected_overcut_upper_mm3"], 0)
                self.assertEqual(final["protected_overcut_upper_mm3"], 0)
                self.assertGreaterEqual(prior["sections"]["0:1"][2], 2)
        only_rest = replace(job, stages=(replace(job.stages[1],
                            transition=None),))
        with self.assertRaisesRegex(ValueError, "uncleared descent"):
            emit(only_rest, "uccnc")

    def test_protected_rib_stale_source_and_missing_stock(self):
        job = synthetic_job()
        stage = job.stages[0]
        motions = list(stage.motions)
        cut = motions[2]
        motions[2] = replace(cut, end=(3.2, 1.21, -1))
        motions[3] = replace(motions[3], start=motions[2].end)
        bad = replace(job, stages=(replace(stage, motions=tuple(motions)),
                                   job.stages[1]))
        with self.assertRaisesRegex(ValueError, "protected"):
            emit(bad, "uccnc")
        _, report = emit(replace(job, stock_present=False), "uccnc")
        self.assertEqual(report["stock_access_residual"]["status"],
                         "not_evaluated")
        with self.assertRaisesRegex(ValueError, "identity"):
            replace(stage, operation=replace(stage.volume_operation,
                                             tool_id="T2"))
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "volume-output"
            report = write_bundle(bundle, job, "uccnc")
            self.assertEqual(report["stock_access_residual"]["status"],
                             "pass")
            self.assertEqual(audit_bundle(bundle / "handoff.json", job), report)
            with self.assertRaisesRegex(ValueError, "stale"):
                audit_bundle(bundle / "handoff.json",
                             replace(job, source_fingerprint="edited"))
            program = bundle / "stage-2.nc"
            program.write_bytes(program.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                audit_bundle(bundle / "handoff.json", job)


if __name__ == "__main__":
    unittest.main()
