"""Whole native Default-post evidence from small, in-memory bound documents."""

from dataclasses import replace
import hashlib
import math
from types import SimpleNamespace
import unittest

try:
    from cambam_builder import CBProject
    from cambam_builder.cam_core import occupancy, ordered_job as oj, replay, v_region as v
    from cambam_builder.cam_extensions import bundle_search as b
    from cambam_builder.integrations.ordered_output import emit
    from cambam_builder.integrations.cambam import bundle_document as native
    from cambam_builder.integrations.cambam.bundle_post import audit_post, _edge_multiset
    from cambam_builder.native.reader import read_cambam_bytes
    from cambam_builder.native.writer import serialize_cambam_bytes
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _document(specs=(("T1", .5, "v"),), *, fixtures=(), cap_depth=1,
              supplied_paths=None, axial_limits=None, safe_z=3):
    project = CBProject("synthetic source")
    layer = project.add_layer("design")
    project.add_rect(layer=layer, identifier="opening", width=10, height=8)
    project.add_part("Part", stock_width=10, stock_height=8, stock_thickness=cap_depth)
    source = serialize_cambam_bytes(project)
    design = native.bind_design(source, "opening", "Part", cap_depth_mm=cap_depth,
                                design_angle_degrees=90)
    start, stages, sweeps = (-1, -1, safe_z), [], []
    profile = v.VProfile("pointed", 90, 0, 2, 2)
    for index, (tool, depth, kind) in enumerate(specs):
        paths = supplied_paths or (
            v.VPath("fill", ((3, 3, depth), (5, 3, depth), (6, 3, depth))),
            v.VPath("fill", ((3, 5, depth), (6, 5, depth))))
        plan = v.VPlan(design.target, profile, paths, v._motions(paths, safe_z),
                       safe_z, .02, 1, "partial", "small supplied witness")
        motions = tuple(oj.JobMove(m.role, m.start, m.end,
            0 if m.role == "rapid" else 120 if m.role == "entry" else 300)
            for m in v.complete_motion(plan, start))
        transition = None if not index else oj.Transition(
            "operator", "split", tool, start, "declared synthetic install")
        operation = None
        if kind == "cylinder":
            target = replay.Target("opening", design.target.safe.bounds, cap_depth,
                region_shell=tuple(design.target.safe.exterior.coords)[:-1])
            operation = replay.Operation(f"pass-{index}",
                replay.ToolProfile(tool, "cylinder", .25, 2), target)
        stage = oj.Stage(f"pass-{index}", tool, motions, 10000,
            v_plan=plan if operation is None else None, operation=operation,
            source_revision=plan.fingerprint if operation is None else design.source_sha256,
            transition=transition,
            axial_limits=axial_limits or oj.AxialLimits(.5, 1))
        stages.append(stage)
        if operation is None:
            sweeps.append(plan)
        else:
            items = (replay.Event("tool_change", tool, start),
                     replay.Event("spindle_start", tool, start))
            items += tuple(replay.Motion(m.role, tool, stage.id, m.start, m.end, m.feed)
                           for m in motions)
            items += (replay.Event("spindle_stop", tool, start),)
            sweeps.append(replay.Trace(design.source_sha256, "program", start,
                                      (operation,), items))
    tools = []
    for tool in dict.fromkeys(spec[0] for spec in specs):
        tools.append(occupancy.ToolBody(tool, (
            occupancy.ToolBand("cutter", 0, 2, 2),
            occupancy.ToolBand("shank", 2, 3, 2),
            occupancy.ToolBand("holder", 3, 5, 3))))
    setup = occupancy.OccupancySetup("program",
        occupancy.Box("declared Part stock", (0, 0, -cap_depth, 10, 8, 0)), fixtures, tuple(tools))
    job = oj.Job(design.source_sha256, tuple(stages), start, occupancy_setup=setup)
    files, _ = emit(job, "uccnc", coordinate_decimals=6)
    composition = v.VComposition(design.target, tuple(sweeps))
    bundle = b.Bundle(job, files, "uccnc", v.section_report(composition, .25),
        v.volume_bounds(composition), .25, None,
        b.Cost(0, 0, 0, 0, 0), tuple(hashlib.sha256(f).hexdigest() for f in files),
        design.target.fingerprint)
    return native.author(design, bundle)


def _post(document, *, reverse=False):
    """Independent native-like writer; this is never production adapter input generation."""
    project = read_cambam_bytes(document.xml_bytes)
    safe_z = f"{document.bundle.job.initial_tip[2]:.9f}".rstrip("0").rstrip(".")
    lines = ["( Made using CamBam - http://www.cambam.co.uk )",
             "( synthetic 10/26/2026 12:00:00 PM )", "( Post processor: Default )",
             "G21 G90 G64 G40", f"G0 Z{safe_z}"]
    previous = None
    for stage, identifier in zip(document.bundle.job.stages, document.stage_mop_identifiers):
        if stage.tool_id != previous:
            if previous is not None:
                lines.append("M5")
            diameter = (2 * stage.v_plan.tool.maximum_radius if stage.v_plan
                        else 2 * stage.operation.tool.radius)
            lines.extend((f"( {stage.tool_id} : {diameter} )", f"{stage.tool_id} M6"))
        lines.append(f"( {project.get_mop(identifier).name} )")
        if stage.tool_id != previous:
            lines.extend(("G61 G17", "M3 S10000"))
        else:
            lines.append("S10000")
        paths, current = [], []
        for move in stage.motions:
            if move.role == "cut":
                if not current:
                    current.append(move.start)
                current.append(move.end)
            elif current:
                paths.append(tuple(current))
                current = []
        if reverse:
            paths = [tuple(reversed(path)) for path in reversed(paths)]
        for path in paths:
            x, y, z = path[0]
            lines.extend((f"G0 X{x} Y{y}", f"G1 F120 Z{z}"))
            for x, y, z in path[1:]:
                lines.append(f"G1 F300 X{x} Y{y} Z{z}")
            lines.append(f"G0 Z{safe_z}")
        previous = stage.tool_id
    lines.extend(("M5", "M30"))
    return ("\n".join(lines) + "\n").encode()


def _shared_endpoint_paths():
    return (v.VPath("fill", ((3, 3, .25), (5, 3, .25))),
            v.VPath("fill", ((5, 3, .5), (7, 3, .5))))


def _partial_retract_paths():
    return (v.VPath("fill", ((3, 3, .5), (5, 3, .5))),
            v.VPath("fill", ((5, 3, .25), (7, 3, .25))))


def _joined_post(document):
    posted = _post(document)
    boundary = b"G0 Z3\nG0 X5 Y3\n"
    assert posted.count(boundary) == 1
    return posted.replace(boundary, b"")


def _partial_retract_post(document):
    posted = _joined_post(document)
    entry = b"G1 F120 Z-0.25"
    assert posted.count(entry) == 1
    return posted.replace(entry, b"G0 Z-0.25")


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class BundlePostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single = _document()
        cls.passes = _document((("T1", .5, "v"), ("T1", 1, "v")))
        cls.mixed = _document((("T2", .5, "cylinder"), ("T1", .5, "v")))

    def audit(self, document, post=None, **kwargs):
        return audit_post(document, _post(document) if post is None else post,
            max_residual_area_mm2=1000, max_residual_volume_mm3=1000, **kwargs)

    def test_whole_single_same_tool_depth_passes_and_changed_tool_mixed(self):
        for document in (self.single, self.passes, self.mixed):
            with self.subTest(stages=len(document.bundle.job.stages)):
                report = self.audit(document)
                self.assertEqual(report["status"], "pass")
                self.assertEqual(report["tool_fixture_occupancy"]["checked_moves"],
                    sum(r["checked_moves"] for r in report["cutting_path_equivalence"]["stages"]))
                self.assertEqual(report["post_sha256"], hashlib.sha256(_post(document)).hexdigest())
                self.assertEqual(report["operator_assumptions"]["initial_tip_xyz_mm"], (-1, -1, 3))
                self.assertEqual(report["operator_assumptions"]["stock_box"], (0, 0, -1, 10, 8, 0))
                self.assertEqual(report["selected_time_cost_and_optimum"], "not_transferred")

    def test_path_order_and_direction_may_change_with_stage_history_preserved(self):
        for document in (self.single, self.passes, self.mixed):
            report = self.audit(document, _post(document, reverse=True))
            self.assertEqual(report["axial_process_limits"]["status"], "pass")
            self.assertEqual(report["cutting_path_equivalence"]["within_stage_order_and_direction"], "may differ")

    def test_native_shared_endpoint_descending_reentry_is_fully_checked(self):
        for kind in ("v", "cylinder"):
            document = _document((("T1", .5, kind),), supplied_paths=_shared_endpoint_paths())
            unjoined = self.audit(document)
            joined = self.audit(document, _joined_post(document))
            with self.subTest(kind=kind):
                self.assertEqual(joined["whole_motion_safety"]["same_xy_descending_reentries"], 1)
                self.assertEqual(joined["cutting_path_equivalence"]["stages"][0]["paths"], 2)
                self.assertEqual(joined["stock_access_residual"]["section_mm2"],
                                 unjoined["stock_access_residual"]["section_mm2"])
                self.assertEqual(joined["stock_access_residual"]["volume_mm3"],
                                 unjoined["stock_access_residual"]["volume_mm3"])
                self.assertEqual(joined["axial_process_limits"], unjoined["axial_process_limits"])
                self.assertEqual(joined["tool_fixture_occupancy"]["checked_moves"],
                                 unjoined["tool_fixture_occupancy"]["checked_moves"] - 1)

    def test_shared_endpoint_support_rejects_other_low_actions_and_extra_edges(self):
        document = _document(supplied_paths=_shared_endpoint_paths())
        posted = _joined_post(document)
        entry = b"G1 F120 Z-0.5"
        cases = (posted.replace(entry, b"G1 F120 Z-0.2"),
                 posted.replace(entry, b"G1 F120 X5.00001 Z-0.5"),
                 posted.replace(entry, b"G0 Z-0.5"),
                 posted.replace(entry, b"G1 F119 Z-0.5"),
                 posted.replace(b"G1 F300 X5 Y3 Z-0.25\n", b""),
                 posted.replace(entry, entry + b"\nG1 F120 Z-0.6"),
                 posted.replace(entry, entry + b"\nG1 F300 X8 Y3 Z-0.5"))
        for candidate in cases:
            with self.subTest(post=candidate), self.assertRaises(ValueError):
                self.audit(document, candidate)

    def test_native_shared_endpoint_partial_retract_is_verified_subset(self):
        for kind in ("v", "cylinder"):
            document = _document((("T1", .5, kind),), supplied_paths=_partial_retract_paths())
            unjoined = self.audit(document)
            joined = self.audit(document, _partial_retract_post(document))
            with self.subTest(kind=kind):
                self.assertEqual(joined["whole_motion_safety"]["same_xy_partial_rapid_retracts"], 1)
                self.assertEqual(joined["whole_motion_safety"]["same_xy_descending_reentries"], 0)
                self.assertEqual(joined["cutting_path_equivalence"]["stages"][0]["paths"], 2)
                self.assertEqual(joined["stock_access_residual"]["section_mm2"],
                                 unjoined["stock_access_residual"]["section_mm2"])
                self.assertEqual(joined["stock_access_residual"]["volume_mm3"],
                                 unjoined["stock_access_residual"]["volume_mm3"])
                self.assertEqual(joined["tool_fixture_occupancy"]["checked_moves"],
                                 unjoined["tool_fixture_occupancy"]["checked_moves"] - 1)

    def test_partial_retract_support_rejects_other_motion_and_unmatched_endpoint(self):
        document = _document(supplied_paths=_partial_retract_paths())
        posted = _partial_retract_post(document)
        retract = b"G0 Z-0.25"
        cases = (posted.replace(retract, b"G1 F120 Z-0.25"),
                 posted.replace(retract, b"G0 X5.00001 Z-0.25"),
                 posted.replace(retract, b"G0 Z-0.6"),
                 posted.replace(retract, b"G0 Z-0.2"),
                 posted.replace(retract, retract + b"\nG0 Z-0.2"),
                 posted.replace(b"G1 F300 X7 Y3 Z-0.25\n", b""))
        for candidate in cases:
            with self.subTest(post=candidate), self.assertRaises(ValueError):
                self.audit(document, candidate)
        fixture = occupancy.Box("retract-side clamp", (4.99, 5.00002, -.4, 5.01, 5.1, -.3))
        body = _document(supplied_paths=_partial_retract_paths(), fixtures=(fixture,))
        self.audit(body, _partial_retract_post(body))
        with self.assertRaisesRegex(ValueError, "cutter collision with fixture retract-side clamp"):
            self.audit(body, _partial_retract_post(body).replace(b"Y3", b"Y3.00004"))

    def test_joined_reentry_retains_actual_axial_protected_depth_and_body_gates(self):
        paths = _shared_endpoint_paths()
        retrace_paths = (paths[0], v.VPath("fill", ((5, 3, .5), (3, 3, .5))))
        stepdown = _document(supplied_paths=retrace_paths, axial_limits=oj.AxialLimits(.25, 1))
        self.audit(stepdown, _joined_post(stepdown))
        with self.assertRaisesRegex(ValueError, "stepdown"):
            self.audit(stepdown, _joined_post(stepdown).replace(b"Z-0.5", b"Z-0.50004"))
        axial = _document(supplied_paths=paths, axial_limits=oj.AxialLimits(.5, .5))
        with self.assertRaisesRegex(ValueError, "entry exceeds declared depth limit"):
            self.audit(axial, _joined_post(axial).replace(b"Z-0.5", b"Z-0.50004"))
        protected = _document(supplied_paths=paths, cap_depth=.5)
        with self.assertRaisesRegex(ValueError, "depth/profile limits"):
            self.audit(protected, _joined_post(protected).replace(b"Z-0.5", b"Z-0.50004"))
        fixture = occupancy.Box("entry-side clamp", (4.99, 5.00002, -.4, 5.01, 5.1, -.3))
        body = _document(supplied_paths=paths, fixtures=(fixture,))
        self.audit(body, _joined_post(body))
        with self.assertRaisesRegex(ValueError, "cutter collision with fixture entry-side clamp"):
            self.audit(body, _joined_post(body).replace(b"Y3", b"Y3.00004"))

    def test_actual_coordinates_recompute_stock_within_edge_tolerance(self):
        post = _post(self.single).replace(b"X6 Y3", b"X6.00004 Y3")
        report = self.audit(self.single, post)
        self.assertNotEqual(report["stock_access_residual"]["actual_composition_fingerprint"],
                            self.audit(self.single)["stock_access_residual"]["actual_composition_fingerprint"])
        self.assertNotEqual(report["stock_access_residual"]["volume_mm3"], self.single.bundle.volume_mm3)

    def test_cached_section_depth_cannot_change_canonical_area_acceptance(self):
        document = _document(cap_depth=2)
        changed = replace(document, bundle=replace(document.bundle, section_depth_mm=2))
        target = document.design.target
        # Both depths are below every .5-deep cut, so these target areas are
        # independent exact upper residual witnesses. The deeper section is
        # smaller and would bypass this threshold if its cached scalar guided
        # the native gate.
        canonical_upper = target.section(1, outer=True).area
        deeper_upper = target.section(2, outer=True).area
        threshold = (canonical_upper + deeper_upper) / 2
        self.assertGreater(canonical_upper, threshold)
        self.assertLess(deeper_upper, threshold)
        for candidate in (document, changed):
            with self.subTest(cached_depth=candidate.bundle.section_depth_mm):
                with self.assertRaisesRegex(ValueError, "residual upper bound"):
                    audit_post(candidate, _post(candidate),
                               max_residual_area_mm2=threshold,
                               max_residual_volume_mm3=1000)
                report = self.audit(candidate)["stock_access_residual"]
                self.assertEqual(report["section_depth_mm"], 1.0)
                self.assertEqual(report["section_mm2"][1], canonical_upper)
        shallow = _document((("T1", .25, "v"),), cap_depth=.4)
        self.assertEqual(self.audit(shallow)["stock_access_residual"]["section_depth_mm"], .4)

    def test_actual_tolerated_edges_still_obey_axial_limits(self):
        # Cutting-edge equivalence tolerates this coordinate rounding, while
        # the independently applied declared .5 stepdown does not.
        post = _post(self.single).replace(b"Z-0.5", b"Z-0.50004")
        with self.assertRaisesRegex(ValueError, "stepdown"):
            self.audit(self.single, post)

    def test_multiset_retains_duplicates_and_rejects_ambiguous_tolerance(self):
        edge = oj.JobMove("cut", (3, 3, -.5), (5, 3, -.5), 300)
        duplicate = SimpleNamespace(id="duplicates", motions=(edge, edge))
        self.assertEqual(_edge_multiset(duplicate, (edge, edge)), 2)
        nearby = oj.JobMove("cut", (3, 3.00004, -.5), (5, 3.00004, -.5), 300)
        ambiguous = SimpleNamespace(id="nearby", motions=(edge, nearby))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _edge_multiset(ambiguous, (edge, nearby))

    def test_actual_safe_travel_is_checked_against_fixture_occupancy(self):
        document = _document(fixtures=(occupancy.Box("tall clamp", (20, 20, 0, 21, 21, 4)),))
        self.audit(document)
        post = _post(document).replace(b"G0 X3 Y3", b"G0 X20 Y20\nG0 X3 Y3", 1)
        with self.assertRaisesRegex(ValueError, "fixture tall clamp"):
            self.audit(document, post)

    def test_clearance_tolerance_never_allows_stock_contact_or_negative_xy_rapid(self):
        document = _document(safe_z=.00001)
        posted = _post(document)
        boundary = b"G0 Z0.00001\nG0 X3 Y5"
        self.assertEqual(posted.count(boundary), 1)
        self.audit(document, posted)
        # Positive numeric rounding remains allowed, but positional tolerance
        # cannot turn the independently required Z>0 clearance into Z<=0.
        self.audit(document, posted.replace(boundary, b"G0 Z0.000005\nG0 X3 Y5"))
        for z in (b"0", b"-0.00001"):
            with self.subTest(z=z), self.assertRaises(ValueError):
                self.audit(document, posted.replace(boundary, b"G0 Z" + z + b"\nG0 X3 Y5"))
            with self.subTest(final_z=z), self.assertRaises(ValueError):
                self.audit(document, posted.replace(b"G0 Z0.00001\nM5", b"G0 Z" + z + b"\nM5"))

    def test_missing_extra_or_modified_cut_edges_fail(self):
        original = _post(self.single)
        cases = (original.replace(b"X5 Y3 Z-0.5", b"X5.01 Y3 Z-0.5"),
                 original.replace(b"G1 F300 X5 Y3 Z-0.5\n", b""),
                 original.replace(b"G1 F300 X6 Y3 Z-0.5", b"G1 F300 X3 Y3 Z-0.5"),
                 original.replace(b"G0 Z3\nM5", b"G1 F300 X3 Y5 Z-0.5\nG0 Z3\nM5"))
        for post in cases:
            with self.subTest(post=post), self.assertRaises(ValueError):
                self.audit(self.single, post)

    def test_unplanned_rapids_entries_links_and_missing_retract_fail(self):
        original = _post(self.single)
        cases = (original.replace(b"G1 F300 X5", b"G0 X5", 1),
                 original.replace(b"G1 F120 Z-0.5", b"G1 F120 X3.1 Z-0.5", 1),
                 original.replace(b"G1 F120 Z-0.5", b"G1 F120 Z-0.6", 1),
                 original.replace(b"G0 X3 Y5", b"G1 F300 X3 Y5", 1),
                 original.replace(b"G0 Z3\nM5", b"M5"),
                 original.replace(b"G0 X3 Y3", b"G0 X3 Y3\nG0 Z1", 1),
                 original.replace(b"G0 X3 Y3", b"G0 X3 Y3\nG1 F120 Z1\nG0 Z3", 1))
        for post in cases:
            with self.subTest(post=post), self.assertRaises(ValueError):
                self.audit(self.single, post)

    def test_wrong_feed_tool_rpm_and_unmodeled_blending_fail(self):
        original = _post(self.single)
        cases = (original.replace(b"F120", b"F121", 1),
                 original.replace(b"F300", b"F301", 1),
                 original.replace(b"T1 M6", b"T2 M6"),
                 original.replace(b"S10000", b"S9999"),
                 original.replace(b"G61 G17", b"G64 G17"),
                 original.replace(b"G1 F300 X5", b"G64\nG1 F300 X5", 1),
                 original.replace(b"M3 S10000", b"M3 S10000\nM5"))
        for post in cases:
            with self.subTest(post=post), self.assertRaises(ValueError):
                self.audit(self.single, post)

    def test_sections_and_whole_trailing_bytes_fail_closed(self):
        original = _post(self.passes)
        project = read_cambam_bytes(self.passes.xml_bytes)
        names = [project.get_mop(i).name.encode() for i in self.passes.stage_mop_identifiers]
        cases = (original.replace(names[1], names[0]),
                 original.replace(b"( " + names[1] + b" )\n", b""),
                 original + b"( extra MOP )\n",
                 original + b"G0 X1\n", original.replace(b"M30", b"G2 X3 Y3 I1 J0\nM30"),
                 original.replace(b"Post processor: Default", b"Post processor: Other"),
                 original.replace(b"( " + names[1] + b" )", b"( unknown MOP )"))
        for post in cases:
            with self.subTest(post=post), self.assertRaises(ValueError):
                self.audit(self.passes, post)

    def test_source_candidate_and_selected_bundle_must_be_fresh(self):
        with self.assertRaises(ValueError):
            self.audit(self.single, source_bytes=self.single.design.source_bytes + b" ")
        with self.assertRaises(ValueError):
            self.audit(self.single, candidate_bytes=self.single.xml_bytes.replace(
                b'<TargetDepth state="Value">0</TargetDepth>',
                b'<TargetDepth state="Value">-1</TargetDepth>'))
        forged = replace(self.single, bundle=replace(self.single.bundle,
            files=(self.single.bundle.files[0] + b"\nG0 X999",)))
        with self.assertRaises(ValueError):
            self.audit(forged)

    def test_explicit_residual_limits_are_checked_against_actual_output(self):
        with self.assertRaisesRegex(ValueError, "residual upper bound"):
            audit_post(self.single, _post(self.single),
                       max_residual_area_mm2=0, max_residual_volume_mm3=1000)
        for limit in (-1, math.nan, math.inf, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                audit_post(self.single, _post(self.single),
                           max_residual_area_mm2=1000, max_residual_volume_mm3=limit)


if __name__ == "__main__":
    unittest.main()
