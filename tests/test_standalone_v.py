"""Decoded ordered-job evidence for a standalone Region-V stage."""

from dataclasses import replace
import json
import math
from pathlib import Path
import re
import tempfile
import unittest

try:
    from shapely.geometry import box
    from cambam_builder.cam_core import curved_region, ordered_job, v_region
    from cambam_builder.integrations import ordered_dialects
    from cambam_builder.integrations.ordered_output import (
        audit_bundle, audit_files, emit, write_bundle,
    )
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _circle(radius, clockwise=False, center=(0, 0)):
    direction = -1 if clockwise else 1
    bulge = direction * math.tan(math.pi / 8)
    return tuple((center[0] + radius * math.cos(direction * i * math.pi / 2),
                  center[1] + radius * math.sin(direction * i * math.pi / 2),
                  bulge) for i in range(4))


def _profiles():
    return (
        v_region.VProfile("pointed", 90, 0, 4, 3),
        v_region.VProfile("flat", 90, 0.25, 4, 3),
        v_region.VProfile("rounded", 60, 0.5, 4, 3),
    )


def _job(plan, *, tool_id="T41", feed=100):
    start = (-2, -2, plan.safe_z)
    motion = v_region.complete_motion(plan, start)
    stage = ordered_job.Stage(
        "standalone-v", tool_id,
        tuple(ordered_job.JobMove(
            move.role, move.start, move.end,
            0 if move.role == "rapid" else feed)
              for move in motion),
        11000, v_plan=plan, source_revision=plan.fingerprint)
    job = ordered_job.Job(plan.target.source_id, (stage,), start,
                          stock_present=True)
    return job, stage


def _output_tempdir():
    output = Path(__file__).resolve().parents[1] / "output"
    output.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="standalone-v-regression-",
                                       dir=output)


def _one_changed_line(data, pattern, change):
    result, count = pattern.subn(change, data, count=1)
    if count != 1:
        raise AssertionError("expected one controller motion line")
    return result


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class StandaloneVTests(unittest.TestCase):
    def test_curved_annulus_decodes_for_profiles_strategies_and_dialects(self):
        target = v_region.VTarget.curved(
            "M2-annulus",
            curved_region.approximate(
                _circle(9), (_circle(2, clockwise=True),)),
            2)
        reports = {}
        for fill in ("raster", "offset"):
            for tool in _profiles():
                with self.subTest(fill=fill, tool=tool.kind):
                    if fill == "raster":
                        # Match the original annulus consumer's default
                        # stepover, XY sampling and safe height.
                        plan = v_region.plan(target, tool)
                    else:
                        # Offset contours are more vertex-dense; coarsen only
                        # this additional strategy probe to bound test cost.
                        plan = v_region.plan(
                            target, tool, stepover_mm=2.5, xy_step_mm=2,
                            safe_z=2, fill_pattern="offset")
                    self.assertEqual(plan.status, "partial")
                    self.assertTrue(plan.paths)
                    job, _ = _job(plan)
                    for dialect in ("uccnc", "grbl"):
                        with self.subTest(dialect=dialect):
                            _, report = emit(job, dialect)
                            stock = report["stock_access_residual"]
                            self.assertEqual(
                                report["motion_equivalence"]["status"],
                                "pass")
                            self.assertEqual(stock["status"], "pass")
                            self.assertEqual(
                                stock["scope"],
                                "decoded standalone Region V from virgin stock")
                            self.assertEqual(stock["initial_stock"], "virgin")
                            self.assertEqual(stock["plan_status"], plan.status)
                            self.assertEqual(stock["section_depth_mm"], 1)
                            prior = stock["prior_section_1_mm2"]
                            final = stock["section_1_mm2"]
                            self.assertLessEqual(prior[0], prior[1])
                            self.assertLessEqual(final[0], final[1])
                            # Independent analytic section for this included
                            # angle: pi((9-d*tan)^2-(2+d*tan)^2), d=1 mm.
                            inset = tool.tangent
                            analytic = math.pi * ((9 - inset) ** 2 -
                                                  (2 + inset) ** 2)
                            self.assertLessEqual(prior[0], analytic)
                            self.assertGreaterEqual(prior[1], analytic)
                            self.assertLess(final[1], prior[0])
                            self.assertGreater(final[1], 0)
                            self.assertTrue(math.isfinite(final[1]))
                            self.assertLess(final[2], 0.1)
                            self.assertLessEqual(stock["prior_volume_mm3"][0],
                                                 stock["prior_volume_mm3"][1])
                            self.assertLessEqual(stock["volume_mm3"][0],
                                                 stock["volume_mm3"][1])
                            self.assertEqual(
                                len(stock["decoded_v_plan_fingerprint"]), 64)
                            reports[(fill, tool.kind, dialect)] = stock

                    # Both decoders must replay identical motion/effects.
                    for field in ("prior_section_1_mm2", "section_1_mm2",
                                  "prior_volume_mm3", "volume_mm3"):
                        self.assertEqual(
                            reports[(fill, tool.kind, "uccnc")][field],
                            reports[(fill, tool.kind, "grbl")][field])

    def test_analytic_capsule_witness_bounds_decoded_stock_and_volume(self):
        target = v_region.VTarget.polygon(
            "capsule-witness", ((0, 0), (10, 0), (10, 10), (0, 10)), (), 1)
        tool = v_region.VProfile("pointed", 90, 0, 4, 3)
        path = v_region.VPath("fill", ((4, 5, 1), (6, 5, 1)))
        plan = v_region.VPlan(
            target, tool, (path,), v_region._motions((path,), 3),
            3, 0.001, 1, "partial", "hand-authored capsule witness")
        job, _ = _job(plan)
        _, report = emit(job, "uccnc")
        stock = report["stock_access_residual"]

        # Independently integrate the square section and the length-two
        # pointed-cutter capsule: area(z)=(10-2z)^2 - 4(1-z) - pi(1-z)^2.
        initial_volume = 100 - 20 + 4 / 3
        residual_volume = initial_volume - 2 - math.pi / 3
        prior_section = stock["prior_section_1_mm2"]
        final_section = stock["section_1_mm2"]
        self.assertLessEqual(prior_section[0], 64)
        self.assertGreaterEqual(prior_section[1], 64)
        self.assertLessEqual(final_section[0], 64)
        self.assertGreaterEqual(final_section[1], 64)
        self.assertLessEqual(stock["prior_volume_mm3"][0], initial_volume)
        self.assertGreaterEqual(stock["prior_volume_mm3"][1], initial_volume)
        self.assertLessEqual(stock["volume_mm3"][0], residual_volume)
        self.assertGreaterEqual(stock["volume_mm3"][1], residual_volume)

    def test_section_depth_field_uses_cap_below_one_millimetre(self):
        target = v_region.VTarget.polygon(
            "shallow-cap", ((0, 0), (10, 0), (10, 10), (0, 10)), (), 0.5)
        plan = v_region.plan(target, v_region.VProfile("pointed", 90, 0, 4, 3))
        job, _ = _job(plan)
        _, report = emit(job, "uccnc")
        stock = report["stock_access_residual"]
        self.assertEqual(stock["section_depth_mm"], 0.5)
        # The legacy report field names retain their `_1` spelling while the
        # reported section follows min(1 mm, target cap), here 0.5 mm.
        self.assertLessEqual(stock["prior_section_1_mm2"][0], 81)
        self.assertGreaterEqual(stock["prior_section_1_mm2"][1], 81)
        self.assertLessEqual(stock["section_1_mm2"][0], 81)
        self.assertGreaterEqual(stock["section_1_mm2"][1], 81)
        with self.assertRaisesRegex(ValueError, "section outside V target"):
            v_region.section_report(plan, 1)

    def test_within_tolerance_decoded_edit_changes_v_fingerprint_and_residual(self):
        target = v_region.VTarget.polygon(
            "decoded-edit", ((0, 0), (10, 0), (10, 10), (0, 10)), (), 2)
        tool = v_region.VProfile("pointed", 90, 0, 4, 3)
        path = v_region.VPath("fill", ((4, 5, 1.5), (6, 5, 1.5)))
        plan = v_region.VPlan(
            target, tool, (path,), v_region._motions((path,), 3),
            3, 0.001, 1, "partial", "decoded coordinate sensitivity")
        job, _ = _job(plan)
        files, planned = emit(job, "uccnc", coordinate_decimals=6)
        data = files[0]
        endpoint = re.compile(
            rb"^(G1 F[^\r\n]*? X)(6)(?= Y)", re.MULTILINE)
        edited, changed = endpoint.subn(
            lambda match: match.group(1) + b"6.00001", data)
        self.assertEqual(changed, 2)
        observed = audit_files(job, "uccnc", (edited,))
        self.assertEqual(observed["motion_equivalence"]["status"], "pass")
        self.assertNotEqual(
            observed["stock_access_residual"]["decoded_v_plan_fingerprint"],
            planned["stock_access_residual"]["decoded_v_plan_fingerprint"])
        self.assertNotEqual(observed["stock_access_residual"]["section_1_mm2"],
                            planned["stock_access_residual"]["section_1_mm2"])

    def test_disconnected_satellite_short_flute_and_no_fit(self):
        tool = v_region.VProfile("pointed", 90, 0, 2, 1)
        shape = box(0, 0, 10, 10).union(
            box(9.9, 0.245, 12.1, 0.255)).union(
            box(12, 0.05, 12.4, 0.45))
        target = v_region.VTarget("satellite", shape, shape, 1)
        for fill in ("raster", "offset"):
            with self.subTest(fill=fill):
                plan = v_region.plan(target, tool, fill_pattern=fill)
                self.assertEqual(plan.status, "partial")
                self.assertTrue(any(path.points[0][0] > 12
                                    for path in plan.paths))
                self.assertTrue(any(path.points[0][0] < 10
                                    for path in plan.paths))
                job, _ = _job(plan)
                _, report = emit(job, "uccnc")
                self.assertEqual(report["stock_access_residual"]["status"],
                                 "pass")

        deep = v_region.VTarget.polygon(
            "short-flute", ((0, 0), (10, 0), (10, 10), (0, 10)), (), 2)
        short_tool = v_region.VProfile("pointed", 90, 0, 1, 0.5)
        short_plan = v_region.plan(deep, short_tool)
        self.assertEqual(short_plan.status, "partial")
        self.assertLessEqual(max(point[2] for path in short_plan.paths
                                 for point in path.points), 0.5)
        short_job, _ = _job(short_plan)
        _, report = emit(short_job, "uccnc")
        stock = report["stock_access_residual"]
        self.assertEqual(stock["plan_status"], "partial")
        self.assertEqual(stock["section_depth_mm"], 1)
        # The flute ends above this section, so the decoded residual encloses
        # the independent 8-by-8 mm square section without claiming progress.
        self.assertLessEqual(stock["section_1_mm2"][0], 64)
        self.assertGreaterEqual(stock["section_1_mm2"][1], 64)

        tiny = v_region.VTarget.polygon(
            "no-fit", ((0, 0), (0.005, 0), (0.005, 0.005), (0, 0.005)),
            (), 1)
        rejected = v_region.plan(tiny, tool)
        self.assertEqual(rejected.status, "infeasible")
        self.assertFalse(rejected.paths)
        with self.assertRaisesRegex(ValueError, "no executable motion"):
            v_region.complete_motion(rejected, (0, 0, 3))

    def test_source_revision_stockless_and_unsupported_mixes(self):
        target = v_region.VTarget.polygon(
            "identity", ((0, 0), (6, 0), (6, 6), (0, 6)), (), 1)
        plan = v_region.plan(target, v_region.VProfile("pointed", 90, 0, 3, 2),
                             stepover_mm=2, xy_step_mm=1, safe_z=3)
        job, stage = _job(plan)
        _, report = emit(replace(job, stock_present=False), "uccnc")
        self.assertEqual(report["stock_access_residual"]["status"],
                         "not_evaluated")

        stale_revision = replace(
            job, stock_present=False,
            stages=(replace(stage, source_revision="old-plan"),))
        with self.assertRaisesRegex(ValueError, "stale V stage revision or source"):
            emit(stale_revision, "uccnc")
        stale_source = replace(job, source_fingerprint="old-source")
        with self.assertRaisesRegex(ValueError, "stale V stage revision or source"):
            emit(stale_source, "uccnc")

        second_motion = stage.motions
        second = replace(stage, id="second-v", tool_id="T42",
                         motions=second_motion,
                         transition=ordered_job.Transition(
                             "operator", "split", "T42", job.initial_tip,
                             "manual-install"))
        multiple = replace(job, stages=(stage, second))
        _, report = emit(multiple, "uccnc")
        stock = report["stock_access_residual"]
        self.assertEqual(stock["status"], "pass")
        for first, repeated in zip(stock["prefixes"][0]["section_1_mm2"],
                                   stock["prefixes"][1]["section_1_mm2"]):
            self.assertAlmostEqual(first, repeated, delta=1e-10)

        unknown = ordered_job.Stage(
            "unknown-stage", "T42", second_motion, 11000,
            source_revision="unknown-evaluator",
            transition=ordered_job.Transition(
                "operator", "split", "T42", job.initial_tip,
                "manual-install"))
        mixed = replace(job, stages=(stage, unknown))
        _, report = emit(mixed, "uccnc")
        self.assertEqual(report["stock_access_residual"]["status"],
                         "unsupported")

    def test_output_byte_tampering_and_bundle_freshness(self):
        target = v_region.VTarget.polygon(
            "bundle", ((0, 0), (10, 0), (10, 10), (0, 10)), (), 1)
        tool = v_region.VProfile("pointed", 90, 0, 4, 3)
        path = v_region.VPath("fill", ((4, 5, 1), (6, 5, 1)))
        plan = v_region.VPlan(
            target, tool, (path,), v_region._motions((path,), 3),
            3, 0.001, 1, "partial", "hand-authored capsule witness")
        job, _ = _job(plan)
        files, report = emit(job, "uccnc")
        data = files[0]
        mutations = (
            (re.compile(rb"^(G1 F)([0-9.]+)(?= )", re.MULTILINE),
             lambda match: match.group(1) +
             str(float(match.group(2)) + 1).encode("ascii")),
            (re.compile(rb"^(G1 F[^\r\n]*? Z)(-?(?:\d+(?:\.\d*)?|\.\d+))$",
                        re.MULTILINE),
             lambda match: match.group(1) +
             str(float(match.group(2)) - 0.01).encode("ascii")),
            (re.compile(rb"^(G1 F[^\r\n]*? X)(-?(?:\d+(?:\.\d*)?|\.\d+))(?= Y)",
                        re.MULTILINE),
             lambda match: match.group(1) +
             str(float(match.group(2)) + 0.01).encode("ascii")),
        )
        for pattern, change in mutations:
            with self.subTest(pattern=pattern.pattern):
                edited = _one_changed_line(data, pattern, change)
                with self.assertRaises(ValueError):
                    audit_files(job, "uccnc", (edited,))

        with _output_tempdir() as temp:
            root = Path(temp)
            bundle = root / "capsule"
            built = write_bundle(bundle, job, "uccnc")
            self.assertEqual(built["stock_access_residual"]["status"],
                             "pass")
            with self.assertRaisesRegex(ValueError, "stale ordered output"):
                audit_bundle(bundle / "handoff.json",
                             replace(job, source_fingerprint="old-source"))
            with self.assertRaisesRegex(ValueError, "stale ordered output"):
                audit_bundle(bundle / "handoff.json",
                             replace(job, translation_xyz_mm=(1, 0, 0)))

            manifest_path = bundle / "handoff.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["evidence"]["stock_access_residual"]["volume_mm3"] = [0, 0]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                audit_bundle(manifest_path, job)

    def test_motion_equivalence_cannot_admit_boundary_crossing_plan(self):
        target = v_region.VTarget.polygon(
            "boundary-bypass", ((0, 0), (10, 0), (10, 10), (0, 10)), (), 1)
        tool = v_region.VProfile("pointed", 90, 0, 4, 3)
        path = v_region.VPath("fill", ((0.25, 5, 1), (2.25, 5, 1)))
        start = (-2, -2, 3)
        path_motion = v_region._motions((path,), 3)
        bad_plan = v_region.VPlan(target, tool, (path,), path_motion,
                                  3, 0.001, 1, "partial", "forged path")
        job_motion = (
            v_region.VMotion("rapid", start, path_motion[0].start),
            *path_motion,
            v_region.VMotion("rapid", path_motion[-1].end, start),
        )
        stage = ordered_job.Stage(
            "boundary-bypass", "T41",
            tuple(ordered_job.JobMove(
                move.role, move.start, move.end,
                0 if move.role == "rapid" else 100)
                  for move in job_motion),
            11000, v_plan=bad_plan, source_revision=bad_plan.fingerprint)
        job = ordered_job.Job(target.source_id, (stage,), start,
                              stock_present=True)
        files = ordered_dialects.render(job, "uccnc")
        decoded = ordered_dialects.decode(files, "uccnc",
                                           initial_work_tip=start)
        # Every decoded move agrees with the deliberately forged stage; only
        # the independent continuous cutter-to-boundary check can reject it.
        self.assertEqual(ordered_job._observed_stage(
            stage, decoded.stages[0], (0, 0, 0)), stage.motions)
        with self.assertRaisesRegex(ValueError, "full V cutter crosses protected Region"):
            ordered_job.audit(job, decoded, dialect="uccnc")

    def test_matching_tolerance_does_not_relax_decoded_protected_boundary(self):
        target = v_region.VTarget.polygon(
            "decoded-boundary", ((0, 0), (10, 0), (10, 10), (0, 10)), (), 2)
        tool = v_region.VProfile("pointed", 90, 0, 4, 3)
        path = v_region.VPath("fill", ((1.00001, 5, 1), (2, 5, 1)))
        plan = v_region.VPlan(target, tool, (path,),
            v_region._motions((path,), 3), 3, 0.000011, 1, "partial",
            "analytic boundary witness")
        self.assertEqual(v_region.verify(plan), plan)
        job, stage = _job(plan)
        files, _ = emit(job, "uccnc", coordinate_decimals=6)
        # A 90-degree pointed cutter at 1 mm penetration has radius 1 mm.
        # Moving the first column to x=.99999 crosses x=0 by .00001 mm,
        # although its .00002 mm change is within motion matching tolerance.
        edited = files[0].replace(b"X1.00001 ", b"X0.99999 ")
        self.assertNotEqual(edited, files[0])
        decoded = ordered_dialects.decode((edited,), "uccnc",
                                           initial_work_tip=job.initial_tip)
        observed = ordered_job._observed_stage(
            stage, decoded.stages[0], (0, 0, 0))
        self.assertEqual(len(observed), len(stage.motions))
        with self.assertRaisesRegex(ValueError, "full V cutter crosses protected Region"):
            ordered_job.audit(job, decoded, dialect="uccnc")
        with self.assertRaisesRegex(ValueError, "full V cutter crosses protected Region"):
            audit_files(job, "uccnc", (edited,))

    def test_forged_plan_controls_cannot_disable_protected_sweep(self):
        target = v_region.VTarget.polygon(
            "forged-margin", ((0, 0), (10, 0), (10, 10), (0, 10)), (), 1)
        tool = v_region.VProfile("pointed", 90, 0, 4, 3)
        path = v_region.VPath("fill", ((0.25, 5, 1), (2.25, 5, 1)))
        motions = v_region._motions((path,), 3)
        # A negative margin made the old boundary predicate vacuous; NaN made
        # its inequality false. Stage, source and decoded motion still agree.
        base = v_region.VPlan(target, tool, (path,), motions, 3, .001, 1,
                              "partial", "forged controls")
        start = (-2, -2, 3)
        complete = (v_region.VMotion("rapid", start, motions[0].start),
                    *motions,
                    v_region.VMotion("rapid", motions[-1].end, start))
        moves = tuple(ordered_job.JobMove(m.role, m.start, m.end,
                     0 if m.role == "rapid" else 100) for m in complete)
        for controls in ({"margin_mm": -10}, {"margin_mm": float("nan")},
                         {"margin_mm": 0}, {"safe_z": float("inf")},
                         {"stepover_mm": -1}):
            with self.subTest(controls=controls):
                forged = replace(base, **controls)
                with self.assertRaisesRegex(ValueError, "path controls"):
                    v_region.verify(forged)
                stage = ordered_job.Stage("forged-margin", "T41", moves,
                    11000, v_plan=forged, source_revision=forged.fingerprint)
                job = ordered_job.Job(target.source_id, (stage,), start)
                files = ordered_dialects.render(job, "uccnc")
                with self.assertRaisesRegex(ValueError, "path controls"):
                    audit_files(job, "uccnc", files)

    def test_v_cut_arc_cannot_be_replayed_as_its_endpoint_chord(self):
        target = v_region.VTarget.polygon(
            "arc-bypass", ((0, 0), (6, 0), (6, 6), (0, 6)), (), 1)
        tool = v_region.VProfile("pointed", 90, 0, 4, 3)
        path = v_region.VPath("fill", ((1.1, 1.1, 1), (4.9, 1.1, 1)))
        plan = v_region.VPlan(target, tool, (path,),
            v_region._motions((path,), 3), 3, .001, 1, "partial",
            "safe straight chord")
        job, stage = _job(plan)
        # The chord fits. The lower G3 semicircle through the same endpoints
        # has center (3,1.1), radius 1.9, and minimum Y=-.8, outside the Region.
        motions = tuple(replace(m, arc_g=3, center=(3, 1.1))
                        if m.role == "cut" else m for m in stage.motions)
        stage = replace(stage, motions=motions)
        job = replace(job, stages=(stage,))
        for dialect in ("uccnc", "grbl"):
            with self.subTest(dialect=dialect):
                files = ordered_dialects.render(job, dialect)
                decoded = ordered_dialects.decode(files, dialect,
                                                  initial_work_tip=job.initial_tip)
                observed = ordered_job._observed_stage(
                    stage, decoded.stages[0], (0, 0, 0))
                self.assertEqual(observed, motions)
                with self.assertRaisesRegex(ValueError, "linear motion only"):
                    ordered_job.audit(job, decoded, dialect=dialect)
                with self.assertRaisesRegex(ValueError, "linear motion only"):
                    audit_files(job, dialect, files)

    def test_complete_decoded_v_return_cannot_enter_protected_stock(self):
        target = v_region.VTarget.polygon("return-bypass",
            ((0, 0), (10, 0), (10, 10), (0, 10)),
            (((4, 4), (6, 4), (6, 6), (4, 6)),), 1)
        tool = v_region.VProfile("flat", 90, .25, 4, 3)
        plan = v_region.plan(target, tool, stepover_mm=2, safe_z=3)
        start = (4, 5, .00004)
        stage = ordered_job.Stage("return-bypass", "T41",
            tuple(ordered_job.JobMove(m.role, m.start, m.end,
                  0 if m.role == "rapid" else 100)
                  for m in v_region.complete_motion(plan, start)),
            11000, v_plan=plan, source_revision=plan.fingerprint)
        job = ordered_job.Job(target.source_id, (stage,), start)
        for dialect in ("uccnc", "grbl"):
            with self.subTest(dialect=dialect):
                if dialect == "uccnc":
                    files, report = emit(job, dialect, coordinate_decimals=6)
                    self.assertEqual(report["stock_access_residual"]["status"], "pass")
                    endpoint = b"X4 Y5 Z0.00004"
                else:
                    # The Grbl writer supports four decimals only, so this
                    # nominal return already rounds onto the stock plane.
                    files = ordered_dialects.render(job, dialect)
                    endpoint = b"X4 Y5 Z0"
                # Only the safe return ends at this column. A .00005 mm
                # change satisfies matching tolerance but plunges into the
                # protected hole boundary with the flat tool's .25 mm tip.
                edited = files[0].replace(endpoint, b"X4 Y5 Z-0.00001")
                self.assertNotEqual(edited, files[0])
                decoded = ordered_dialects.decode((edited,), dialect,
                                                   initial_work_tip=start)
                observed = ordered_job._observed_stage(
                    stage, decoded.stages[0], (0, 0, 0))
                self.assertEqual(observed[-1].end[2], -.00001)
                with self.assertRaisesRegex(ValueError, "travel contacts"):
                    ordered_job.audit(job, decoded, dialect=dialect)
                with self.assertRaisesRegex(ValueError, "travel contacts"):
                    audit_files(job, dialect, (edited,))
                # Four-decimal rounding alone places this return on Z=0.
                with self.assertRaisesRegex(ValueError, "travel contacts"):
                    emit(job, dialect)

    def test_v_rapid_links_obey_declared_flat_fixture_top(self):
        target = v_region.VTarget.polygon("fixture-links",
            ((0, 0), (6, 0), (6, 6), (0, 6)), (), 1)
        plan = v_region.plan(target, v_region.VProfile("pointed", 90, 0, 3, 2))
        job, _ = _job(plan)
        emit(job, "uccnc", fixture_top_z_mm=1)
        with self.assertRaisesRegex(ValueError, "travel contacts"):
            emit(job, "uccnc", fixture_top_z_mm=plan.safe_z)

    def test_infeasible_status_cannot_smuggle_executable_paths(self):
        target = v_region.VTarget.polygon("infeasible-status",
            ((0, 0), (6, 0), (6, 6), (0, 6)), (), 1)
        plan = v_region.plan(target, v_region.VProfile("pointed", 90, 0, 3, 2))
        job, stage = _job(plan)
        forged = replace(plan, status="infeasible")
        with self.assertRaisesRegex(ValueError, "infeasible.*executable paths"):
            v_region.verify(forged)
        stage = replace(stage, v_plan=forged, source_revision=forged.fingerprint)
        with self.assertRaisesRegex(ValueError, "infeasible.*executable paths"):
            emit(replace(job, stages=(stage,)), "uccnc")


if __name__ == "__main__":
    unittest.main()
