"""MV01 cumulative decoded stock, independent overlaps and protected membership."""
from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from shapely.geometry import Point
    from cambam_builder.cam_core import curved_region, ordered_job, v_region
    from cambam_builder.integrations import ordered_dialects
    from cambam_builder.integrations.ordered_output import (
        audit_bundle, audit_files, emit, write_bundle,
    )
    from tests.test_standalone_v import _circle
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _job(plans, *, tool_ids=None, start=(-2, -2, 3)):
    tool_ids = tool_ids or tuple(f"T{41+i}" for i in range(len(plans)))
    stages = []
    for i, (plan, tool_id) in enumerate(zip(plans, tool_ids)):
        motion = tuple(ordered_job.JobMove(
            m.role, m.start, m.end, 0 if m.role == "rapid" else 100)
            for m in v_region.complete_motion(plan, start))
        transition = None if i == 0 else ordered_job.Transition(
            "operator", "split", tool_id, start, "synthetic-offline-install")
        stages.append(ordered_job.Stage(f"v-{i+1}", tool_id, motion, 11000,
            v_plan=plan, source_revision=plan.fingerprint,
            transition=transition))
    return ordered_job.Job(plans[0].target.source_id, tuple(stages), start)


def _supplied(target, tool, a, b):
    paths = (v_region.VPath("fill", (a, b)),)
    return v_region.VPlan(target, tool, paths, v_region._motions(paths, 3),
                          3, .001, 1, "partial", "independent capsule witness")


def _for_dialect(job, dialect):
    boundary = "pause" if dialect == "grbl" else "split"
    return replace(job, stages=tuple(stage if i == 0 else replace(stage,
        transition=replace(stage.transition, boundary=boundary))
        for i, stage in enumerate(job.stages)))


def _overlap_plans():
    target = v_region.VTarget.polygon("overlap-oracle",
        ((0, 0), (10, 0), (10, 10), (0, 10)), (), 1.5,
        design_angle_degrees=90)
    tool = v_region.VProfile("pointed", 90, 0, 4, 3)
    return (_supplied(target, tool, (4, 5, 1.5), (6, 5, 1.5)),
            _supplied(target, tool, (5, 5, 1.5), (7, 5, 1.5)))


def _distance_to_segment(x, y, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    denominator = dx*dx+dy*dy
    fraction = (0 if denominator == 0 else
                max(0, min(1, ((x-a[0])*dx+(y-a[1])*dy)/denominator)))
    return math.hypot(x-a[0]-fraction*dx, y-a[1]-fraction*dy)


def _membership_margins(plan, point, depth):
    """Independent 90-degree flat/pointed disk membership on linear sweeps.

    Endpoint penetration extrema bound every interpolated section radius.
    Positive inner margin proves a disk removed; negative outer margin proves
    a disk untouched. Neither planning nor stock geometry supplies this oracle.
    """
    inner = outer = -math.inf
    tip = plan.tool.tip_radius if plan.tool.kind == "flat" else 0
    for path in plan.paths:
        for a, b in zip(path.points, path.points[1:]):
            distance = _distance_to_segment(*point, a, b)
            low, high = min(a[2], b[2]), max(a[2], b[2])
            if high >= depth:
                outer = max(outer, tip+high-depth-distance)
            if low >= depth:
                inner = max(inner, tip+low-depth-distance)
    return inner, outer


def _output_tempdir():
    # This remains correct for the package verifier's source-free snapshot.
    output = Path(__file__).resolve().parents[1] / "output"
    output.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="multistage-v-regression-", dir=output)


def _independent_decoded_paths(job, files, dialect):
    """Extract cutting line/depth tuples directly from the dialect decoder.

    No cumulative evaluator or production decoded-plan reconstruction is used.
    Complete-motion equivalence separately checks the expected role alignment.
    """
    decoded = ordered_dialects.decode(files, dialect, initial_work_tip=job.initial_tip)
    result = []
    for stage, observed in zip(job.stages, decoded.stages):
        paths = tuple(SimpleNamespace(points=(
            (move.start[0], move.start[1], -move.start[2]),
            (move.end[0], move.end[1], -move.end[2])))
            for expected, move in zip(stage.motions, observed.moves)
            if expected.role == "cut")
        result.append(SimpleNamespace(tool=stage.v_plan.tool, paths=paths))
    return tuple(result)


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class MultistageVTests(unittest.TestCase):
    def assert_encloses(self, bounds, analytic):
        self.assertLessEqual(bounds[0], analytic)
        self.assertGreaterEqual(bounds[1], analytic)

    def assert_prefixes(self, stock, job):
        self.assertEqual(stock["status"], "pass")
        self.assertEqual(stock["scope"], "decoded cumulative Region V from virgin stock")
        self.assertEqual(stock["initial_stock"], "virgin")
        self.assertEqual(stock["design_fingerprint"], job.stages[0].v_plan.target.fingerprint)
        self.assertEqual(stock["design_angle_degrees"], 90)
        prefixes = stock["prefixes"]
        self.assertEqual(len(prefixes), len(job.stages))
        previous_section = stock["prior_section_1_mm2"]
        previous_volume = stock["prior_volume_mm3"]
        for stage, identity, prefix in zip(job.stages, job.prefixes, prefixes):
            self.assertEqual(prefix["stage_id"], stage.id)
            self.assertEqual(prefix["prefix_fingerprint"], identity)
            self.assertEqual(prefix["plan_status"], stage.v_plan.status)
            for field in ("decoded_v_plan_fingerprint", "decoded_sequence_fingerprint"):
                self.assertEqual(len(prefix[field]), 64)
            section, volume = prefix["section_1_mm2"], prefix["volume_mm3"]
            self.assertLessEqual(section[0], section[1])
            self.assertLessEqual(volume[0], volume[1])
            # Union removal retains both interval endpoints monotonically.
            for i in (0, 1):
                self.assertLessEqual(section[i], previous_section[i]+1e-8)
                self.assertLessEqual(volume[i], previous_volume[i]+1e-8)
            self.assertLess(section[2], .1)
            previous_section, previous_volume = section, volume
        self.assertEqual(stock["section_1_mm2"], prefixes[-1]["section_1_mm2"])
        self.assertEqual(stock["volume_mm3"], prefixes[-1]["volume_mm3"])

    def test_flat_then_pointed_rectangular_island_has_positive_independent_witness(self):
        target = v_region.VTarget.polygon("MV01-rectangular-island",
            ((0, 0), (20, 0), (20, 16), (0, 16)),
            (((8, 6), (12, 6), (12, 10), (8, 10)),), 1.5,
            design_angle_degrees=90)
        plans = tuple(v_region.plan(target, tool, stepover_mm=.6,
            xy_step_mm=.4, safe_z=3) for tool in (
                v_region.VProfile("flat", 90, .6, 4, 2),
                v_region.VProfile("pointed", 90, 0, 4, 2)))
        job = _job(plans)
        self.assertTrue(all(plan.status == "partial" for plan in plans))
        depth = 1
        # Search a fixed corner grid with independent distance/radius arithmetic.
        witness = None
        for iy in range(21, 61):
            for ix in range(21, 61):
                point = (ix*.05, iy*.05)
                _, primary_outer = _membership_margins(plans[0], point, depth)
                finish_inner, _ = _membership_margins(plans[1], point, depth)
                radius = min(-primary_outer, finish_inner)-.0001
                if radius > .002:
                    witness = (point, radius)
                    break
            if witness:
                break
        self.assertIsNotNone(witness, "pointed finish must clear a positive primary residual disk")
        point, radius = witness
        disk = Point(point).buffer(radius, quad_segs=32)
        first = v_region.section_evidence(v_region.VSequence(plans[:1]), depth)
        final = v_region.section_evidence(v_region.VSequence(plans), depth)
        self.assertTrue(first.residual_inner.covers(disk))
        self.assertTrue(final.known_free_inner.covers(disk))
        self.assertTrue(final.residual_inner.intersection(disk).is_empty)
        expected_section = 18*14-(16+16*depth+math.pi*depth**2)
        expected_volume = (304*1.5-44*1.5**2+(4-math.pi)*1.5**3/3)
        reports = []
        for dialect in ("uccnc", "grbl"):
            job = _for_dialect(job, dialect)
            files, report = emit(job, dialect)
            self.assertEqual(report["motion_equivalence"]["status"], "pass")
            decoded_paths = _independent_decoded_paths(job, files, dialect)
            _, primary_outer = _membership_margins(decoded_paths[0], point, depth)
            finish_inner, _ = _membership_margins(decoded_paths[1], point, depth)
            self.assertLess(primary_outer, -radius)
            self.assertGreater(finish_inner, radius)
            stock = report["stock_access_residual"]
            self.assert_prefixes(stock, job)
            self.assert_encloses(stock["prior_section_1_mm2"], expected_section)
            self.assert_encloses(stock["prior_volume_mm3"], expected_volume)
            self.assertLess(stock["section_1_mm2"][1], stock["prefixes"][0]["section_1_mm2"][1])
            reports.append(stock)
        for field in ("prior_section_1_mm2", "section_1_mm2", "prior_volume_mm3", "volume_mm3"):
            self.assertEqual(reports[0][field], reports[1][field])

    def test_overlapping_capsules_have_independent_section_and_integral_oracles(self):
        plans = _overlap_plans()
        job = _job(plans, tool_ids=("T41", "T41"))
        # At z=1, r=.5: each capsule centerline length is 2 mm; their
        # union centerline length is 3 mm. Overlap must be counted once.
        virgin_section = 64
        first_section = virgin_section-2-math.pi/4
        final_section = virgin_section-3-math.pi/4
        # Integrate (10-2z)^2 and 2*L*(1.5-z)+pi*(1.5-z)^2.
        virgin_volume = 109.5
        final_volume = virgin_volume-3*1.5**2-math.pi*1.5**3/3
        for dialect in ("uccnc", "grbl"):
            job = _for_dialect(job, dialect)
            _, report = emit(job, dialect)
            stock = report["stock_access_residual"]
            self.assert_prefixes(stock, job)
            self.assert_encloses(stock["prior_section_1_mm2"], virgin_section)
            self.assert_encloses(stock["prefixes"][0]["section_1_mm2"], first_section)
            self.assert_encloses(stock["section_1_mm2"], final_section)
            self.assertLess(stock["section_1_mm2"][1]-stock["section_1_mm2"][0], .02)
            self.assert_encloses(stock["prior_volume_mm3"], virgin_volume)
            self.assert_encloses(stock["volume_mm3"], final_volume)
        self.assert_encloses(v_region.volume_bounds(v_region.VSequence(plans), slabs=64), final_volume)

    def test_variable_depth_flat_pointed_keep_both_independent_decoded_disks(self):
        target = v_region.VTarget.polygon("variable-depth-pair",
            ((0, 0), (10, 0), (10, 10), (0, 10)), (), 1.5,
            design_angle_degrees=90)
        flat = v_region.VProfile("flat", 90, .4, 4, 2)
        pointed = v_region.VProfile("pointed", 90, 0, 4, 2)
        plans = (_supplied(target, flat, (3, 5, .5), (6, 5, 1.5)),
                 _supplied(target, pointed, (7, 6, .8), (7, 8, 1.2)))
        depth, radius = .4, .2
        points = ((4, 5), (7, 7))
        section = v_region.section_evidence(v_region.VSequence(plans), depth)
        for point in points:
            self.assertTrue(section.known_free_inner.covers(Point(point).buffer(radius)))
        self.assertLess(_membership_margins(plans[0], points[1], depth)[1], -radius)
        self.assertLess(_membership_margins(plans[1], points[0], depth)[1], -radius)
        for dialect in ("uccnc", "grbl"):
            job = _for_dialect(_job(plans), dialect)
            files, report = emit(job, dialect)
            self.assert_prefixes(report["stock_access_residual"], job)
            observed = _independent_decoded_paths(job, files, dialect)
            for index, point in enumerate(points):
                inner, _ = _membership_margins(observed[index], point, depth)
                _, other_outer = _membership_margins(observed[1-index], point, depth)
                self.assertGreater(inner, radius+.0001)
                self.assertLess(other_outer, -radius-.0001)

    def test_duplicate_repeat_cuts_preserve_inclusion_and_omitted_prefix_mutation_fails(self):
        first, second = _overlap_plans()
        sequences = tuple(v_region.VSequence(plans) for plans in (
            (first,), (first, second), (first, second, first), (first, second, second)))
        sections = [v_region.section_evidence(sequence, 1) for sequence in sequences]
        self.assertTrue(sections[1].known_free_inner.covers(sections[0].known_free_inner))
        self.assertTrue(sections[0].residual_inner.covers(sections[1].residual_inner))
        self.assertTrue(sections[0].residual_outer.covers(sections[1].residual_outer))
        for i in (2, 3):
            self.assertEqual(sections[1].report, sections[i].report)
            self.assertTrue(sections[1].known_free_inner.equals(sections[i].known_free_inner))
            self.assertEqual(v_region.volume_bounds(sequences[1]), v_region.volume_bounds(sequences[i]))
        virgin = v_region.section_evidence(sequences[1], 1, final=False)
        self.assertTrue(virgin.known_free_outer.is_empty)
        self.assert_encloses(virgin.report, 64)
        for plans, ids in (((first, second, first), ("T41", "T42", "T41")),
                            ((first, second, second), ("T41", "T41", "T41"))):
            for dialect in ("uccnc", "grbl"):
                job = _for_dialect(_job(plans, tool_ids=ids), dialect)
                self.assert_prefixes(emit(job, dialect)[1]["stock_access_residual"], job)
        original = v_region.section_evidence
        def last_only(result, depth, *, final=True):
            if type(result) is v_region.VSequence:
                result = result.plans[-1]
            return original(result, depth, final=final)
        # Controlled mutation is rejected by a numeric oracle independent of
        # prefix count/fingerprints and independent of production union code.
        with patch.object(v_region, "section_evidence", side_effect=last_only):
            mutated = emit(_job((first, second)), "uccnc")[1]["stock_access_residual"]
            with self.assertRaises(AssertionError):
                self.assert_encloses(mutated["section_1_mm2"], 64-3-math.pi/4)

    def test_three_profiles_curved_annulus_raster_offset_and_both_dialects(self):
        target = v_region.VTarget.curved("MV01-curved-annulus",
            curved_region.approximate(_circle(8), (_circle(2, clockwise=True),)),
            1.5, design_angle_degrees=90)
        tools = (v_region.VProfile("flat", 90, .3, 4, 2),
                 v_region.VProfile("rounded", 90, .5, 4, 2),
                 v_region.VProfile("pointed", 90, 0, 4, 2))
        for fill in ("raster", "offset"):
            plans = tuple(v_region.plan(target, tool, stepover_mm=2.5,
                xy_step_mm=2, safe_z=3, fill_pattern=fill) for tool in tools)
            job = _job(plans)
            reports = []
            for dialect in ("uccnc", "grbl"):
                with self.subTest(fill=fill, dialect=dialect):
                    job = _for_dialect(job, dialect)
                    _, report = emit(job, dialect)
                    stock = report["stock_access_residual"]
                    self.assert_prefixes(stock, job)
                    self.assert_encloses(stock["prior_section_1_mm2"], 40*math.pi)
                    self.assert_encloses(stock["prior_volume_mm3"], 67.5*math.pi)
                    self.assertLess(stock["section_1_mm2"][1], stock["prior_section_1_mm2"][0])
                    reports.append(stock)
            self.assertEqual(reports[0]["section_1_mm2"], reports[1]["section_1_mm2"])
            self.assertEqual(reports[0]["volume_mm3"], reports[1]["volume_mm3"])

    def test_sequence_is_immutable_ordered_and_requires_one_complete_design_identity(self):
        first, second = _overlap_plans()
        sequence = v_region.VSequence((first, second))
        self.assertIs(sequence.target, first.target)
        self.assertEqual(sequence.fingerprint, v_region.VSequence((first, second)).fingerprint)
        self.assertNotEqual(sequence.fingerprint, v_region.VSequence((second, first)).fingerprint)
        with self.assertRaises(FrozenInstanceError):
            sequence.plans = (first,)
        for invalid in ((), [first], (first, object())):
            with self.assertRaises(ValueError):
                v_region.VSequence(invalid)
        for changed in (replace(second.target, source_id="stale"),
                        replace(second.target, cap_depth=1.6),
                        replace(second.target, design_angle_degrees=80),
                        replace(second.target, frame="different")):
            altered = replace(second, target=changed)
            with self.assertRaises(ValueError):
                v_region.VSequence((first, altered))
            with self.assertRaises(ValueError):
                emit(_job((first, altered)), "uccnc")

    def test_decoded_feed_motion_and_small_edit_change_prefix_evidence(self):
        job = _job(_overlap_plans())
        files, original = emit(job, "uccnc", coordinate_decimals=6)
        feed_changed = files[-1].replace(b"F100", b"F101", 1)
        self.assertNotEqual(files[-1], feed_changed)
        with self.assertRaisesRegex(ValueError, "motion.*differs"):
            audit_files(job, "uccnc", files[:-1]+(feed_changed,))
        pattern = re.compile(rb"^(G1 F[^\r\n]*? X)(7)(?= Y)", re.MULTILINE)
        changed, count = pattern.subn(lambda m: m.group(1)+b"7.01", files[-1])
        self.assertGreater(count, 0)
        with self.assertRaisesRegex(ValueError, "motion.*differs"):
            audit_files(job, "uccnc", files[:-1]+(changed,))
        small, count = pattern.subn(lambda m: m.group(1)+b"7.00001", files[-1])
        self.assertGreater(count, 0)
        observed = audit_files(job, "uccnc", files[:-1]+(small,))["stock_access_residual"]
        planned = original["stock_access_residual"]
        self.assertEqual(observed["prefixes"][0], planned["prefixes"][0])
        self.assertNotEqual(observed["prefixes"][1]["decoded_v_plan_fingerprint"],
                            planned["prefixes"][1]["decoded_v_plan_fingerprint"])
        self.assertNotEqual(observed["prefixes"][1]["decoded_sequence_fingerprint"],
                            planned["prefixes"][1]["decoded_sequence_fingerprint"])
        self.assertNotEqual(observed["section_1_mm2"], planned["section_1_mm2"])

    def test_protected_sweep_stale_revision_stockless_and_unsupported_mix(self):
        plans = _overlap_plans()
        job = _job(plans)
        for dialect in ("uccnc", "grbl"):
            job = _for_dialect(job, dialect)
            stock = emit(replace(job, stock_present=False), dialect)[1]["stock_access_residual"]
            self.assertEqual(stock["status"], "not_evaluated")
            stale = replace(job.stages[1], source_revision="old")
            with self.assertRaisesRegex(ValueError, "stale V stage revision or source"):
                emit(replace(job, stages=(job.stages[0], stale)), dialect)
            unknown = replace(job.stages[1], v_plan=None, source_revision="no-evaluator")
            unsupported = emit(replace(job, stages=(job.stages[0], unknown)), dialect)[1]
            self.assertEqual(unsupported["stock_access_residual"]["status"], "unsupported")
            self.assertEqual(unsupported["stock_access_residual"]["reason"],
                             "unsupported stock evaluator sequence")
        bad = _supplied(plans[0].target, plans[0].tool,
                        (.5, 4, 1.5), (.5, 6, 1.5))
        with self.assertRaisesRegex(ValueError, "protected Region"):
            v_region.VSequence((plans[0], bad))
        with self.assertRaisesRegex(ValueError, "protected Region"):
            emit(_job((plans[0], bad)), "uccnc")
        holed = v_region.VTarget.polygon("island-crossing",
            ((0, 0), (10, 0), (10, 10), (0, 10)),
            (((4, 4), (6, 4), (6, 6), (4, 6)),), 1.5,
            design_angle_degrees=90)
        good = _supplied(holed, plans[0].tool, (3, 2, .5), (7, 2, .5))
        # Both endpoints fit; the continuous chord crosses the protected island.
        bad = _supplied(holed, plans[0].tool, (3, 5, .2), (7, 5, .2))
        with self.assertRaisesRegex(ValueError, "protected Region"):
            v_region.VSequence((good, bad))
        with self.assertRaisesRegex(ValueError, "protected Region"):
            emit(_job((good, bad)), "uccnc")

    def test_complete_rapid_protection_and_fixture_height(self):
        plans = _overlap_plans()
        job = _job(plans)
        for dialect in ("uccnc", "grbl"):
            job = _for_dialect(job, dialect)
            emit(job, dialect, fixture_top_z_mm=1)
            with self.assertRaisesRegex(ValueError, "travel contacts"):
                emit(job, dialect, fixture_top_z_mm=3)
        low_job = _job(plans, start=(5, 5, .00004))
        files, _ = emit(low_job, "uccnc", coordinate_decimals=6)
        changed = files[-1].replace(b"X5 Y5 Z0.00004", b"X5 Y5 Z-0.00001")
        self.assertNotEqual(changed, files[-1])
        with self.assertRaisesRegex(ValueError, "travel contacts"):
            audit_files(low_job, "uccnc", files[:-1]+(changed,))

    def test_bundle_rejects_stale_reordered_and_tampered_prefix_reports(self):
        plans = _overlap_plans()
        job = _job(plans)
        # A reordered job is safe and receives fresh replay and fresh identities.
        reordered = _job(tuple(reversed(plans)))
        for dialect in ("uccnc", "grbl"):
            job = _for_dialect(job, dialect)
            reordered = _for_dialect(reordered, dialect)
            self.assert_prefixes(emit(reordered, dialect)[1]["stock_access_residual"], reordered)
            with _output_tempdir() as root:
                directory = Path(root)/dialect
                write_bundle(directory, job, dialect)
                manifest = directory/"handoff.json"
                audit_bundle(manifest, job)
                with self.assertRaisesRegex(ValueError, "stale ordered output"):
                    audit_bundle(manifest, reordered)
                revised = replace(job, stages=(job.stages[0], replace(
                    job.stages[1], rpm=12000)))
                with self.assertRaisesRegex(ValueError, "stale ordered output"):
                    audit_bundle(manifest, revised)
                import json
                data = json.loads(manifest.read_text(encoding="utf-8"))
                data["evidence"]["stock_access_residual"]["prefixes"][0]["section_1_mm2"][0] = 0
                manifest.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit_bundle(manifest, job)


if __name__ == "__main__":
    unittest.main()
