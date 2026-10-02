"""Synthetic posts test certification; they never assert actual native planning."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from cambam_builder import CBProject
from cambam_builder.cam_core import polygon_rest, replay, rest_boundaries
from cambam_builder.integrations.cambam import native_rest
from cambam_builder.integrations.cambam.native_ordered_job import NativeBinding, from_native_series
from cambam_builder.integrations.cambam.native_series import normalize_native_series
from cambam_builder.integrations.cambam.native_series_audit import audit_linear_native_series
from cambam_builder.integrations.ordered_output import write_bundle
from cambam_builder.native.cad import Pline
from cambam_builder.native.reader import read_cambam_bytes


SETUP = {"units": "mm", "postprocessor": "Default"}
START = (0, 0, 5)


def make_case(root, *, island=True):
    shell = ((0, 0), (40, 0), (40, 30), (0, 30))
    holes = (((16, 10), (24, 10), (24, 20), (16, 20)),) if island else ()
    target = replay.Target("design", (0, 0, 40, 30), 2,
                           region_shell=shell, region_holes=holes)
    project = CBProject("NR01 native rest")
    layer = project.add_layer("Original CAD")
    region = project.add_region(layer, Pline(vertices=shell, closed=True),
                                [Pline(vertices=h, closed=True) for h in holes],
                                identifier="design")
    part = project.add_part("Part", stock_thickness=2, stock_width=40,
                            stock_height=30, stock_surface=0, nesting_method="None")
    source = root / "source.cb"
    project.save(str(source))
    project.add_pocket_mop(part, targets=[region], name="ROUGH", tool_number=1,
                           tool_diameter=6, tool_profile="EndMill", target_depth=-2,
                           depth_increment=2, stock_surface=0, clearance_plane=5,
                           spindle_speed=12000, plunge_feedrate=60, cut_feedrate=240,
                           roughing_clearance=0.005,
                           lead_in_type="None", max_crossover_distance=0,
                           optimisation_mode="None")
    candidate = root / "rough.cb"
    project.save(str(candidate))
    feasible = Polygon(shell, holes).buffer(-3.005, quad_segs=32)
    paths = [tuple(feasible.exterior.coords)] + [tuple(r.coords) for r in feasible.interiors]
    for index in range(13):
        y = 3.005 + index * 2
        row = feasible.intersection(LineString(((-1, y), (41, y))))
        paths.extend(tuple(p.coords) for p in
                     (tuple(row.geoms) if hasattr(row, "geoms") else (row,))
                     if p.geom_type == "LineString" and not p.is_empty)
    post = root / "rough.nc"
    lines = ["( Made using CamBam )", "( rough test )", "( Post processor: Default )",
             "G21 G90 G61 G40", "G0 Z5", "T1 M6", "( ROUGH )", "G17", "M3 S12000"]
    for path in paths:
        x, y = path[0]
        lines.extend((f"G0 X{x} Y{y}", "G1 F60 Z-2"))
        for x, y in path[1:]:
            lines.append(f"G1 F240 X{x} Y{y}")
        lines.append("G0 Z5")
    lines.extend(("G0 X0 Y0", "M5", "M30"))
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")
    series = normalize_native_series(source, candidate, post, initial_position=START, setup=SETUP)
    binding = native_rest.prepare(
        series, source, candidate, post, target=target,
        cutting_lengths_mm={"T1": 5}, entry_modes={"ROUGH": "virgin"},
        radius_mm=1, overlap_mm=0.5, setup=SETUP)
    return binding


def add_cleanup_post(binding, candidate, post, *, changed_prior=False):
    # Independently supplied simple cuts at feasible window centers. Certification
    # proves motion safety/useful stock change, not Pocket planner equivalence.
    lines = Path(binding.prior_post_path).read_text().splitlines()[:-1]
    lines[1] = f"( {candidate.stem} test )"
    if changed_prior:
        lines[8] = "M3 S11000"
    lines.extend(("T2 M6", "( REST )", "M3 S12000"))
    for boundary in binding.boundaries.windows:
        feasible = boundary.geometry.buffer(-1.01)
        corner = min(binding.boundaries.target.region_shell,
                     key=lambda p: Point(p).distance(boundary.geometry.centroid))
        center = nearest_points(feasible, Point(corner))[0]
        x, y = center.x, center.y
        toward = feasible.representative_point()
        dx, dy = toward.x - x, toward.y - y
        scale = 0.01 / max(0.01, (dx * dx + dy * dy) ** 0.5)
        lines.extend((f"G0 X{x} Y{y}", "G1 F60 Z-2",
                      f"G1 F240 X{x + dx * scale} Y{y + dy * scale}", "G0 Z5"))
    lines.extend(("G0 X0 Y0", "M5", "M30"))
    post.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return normalize_native_series(binding.source_path, candidate, post,
                                   initial_position=START, setup=SETUP)


class NativeRestTests(unittest.TestCase):
    def author(self, root, binding):
        candidate = root / "rest.cb"
        native_rest.author(binding, candidate, tool_number=2, depth_increment_mm=1,
                           clearance_mm=5, spindle_rpm=12000,
                           plunge_feed_mm_min=60, cut_feed_mm_min=240)
        return candidate

    def test_island_disconnected_rest_compensated_windows_and_strict_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binding = make_case(root)
            boundaries = binding.boundaries
            candidate = self.author(root, binding)
            self.assertGreater(len(boundaries.windows), 1)
            self.assertEqual(len(boundaries.windows), 4)
            self.assertGreater(boundaries.reachable_rest_area_mm2, 0)
            design = Polygon(boundaries.target.region_shell, boundaries.target.region_holes)
            prior = replay.replay(binding.check_derivation(),
                                  expected_source=binding.prior_series.evidence_fingerprint)
            pure = design.difference(polygon_rest._cut_polygon(prior.cuts, 2, radial_error=-1e-6))
            for window in boundaries.windows:
                self.assertTrue(design.covers(window.geometry))
                self.assertFalse(window.geometry.buffer(-1).is_empty)
                self.assertGreater(window.geometry.difference(pure).area, 0)
            reopened = binding.check_candidate(candidate)
            self.assertEqual(len(reopened.list_primitives()), 1 + len(boundaries.windows))
            original = read_cambam_bytes(Path(binding.source_path).read_bytes()).get_primitive("design")
            self.assertEqual(reopened.get_primitive("design").get_absolute_coordinates_xyz(),
                             original.get_absolute_coordinates_xyz())
            self.assertEqual([m.name for m in reopened.list_mops()], ["ROUGH", "REST"])
            self.assertEqual(reopened.list_mops()[0].custom_mop_footer, "G0 Z5.0\nM5")
            self.assertEqual(reopened.list_mops()[-1].custom_mop_footer, "")

    def test_complete_synthetic_post_audit_and_both_ordered_consumers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binding = make_case(root)
            candidate = self.author(root, binding)
            post = root / "rest.nc"
            series = add_cleanup_post(binding, candidate, post)
            target = binding.boundaries.target
            args = dict(targets={s.name: target for s in series.stages},
                        cutting_lengths_mm={"T1": 5, "T2": 5},
                        entry_modes={"ROUGH": "virgin", "REST": "virgin"})
            with self.assertRaisesRegex(ValueError, "targets differ"):
                audit_linear_native_series(series, binding.source_path, candidate, post,
                                           **args, section_depths_mm=(1, 2),
                                           max_protected_overcut_mm2=0.001, setup=SETUP)
            audit = audit_linear_native_series(
                series, binding.source_path, candidate, post, **args,
                section_depths_mm=(1, 2), max_protected_overcut_mm2=0.001,
                setup=SETUP, derived_binding=binding)
            self.assertLess(audit.stages[-1].evidence.residual.area_upper_mm2,
                            audit.stages[0].evidence.residual.area_lower_mm2)
            checked_series, checked = native_rest.audit(
                binding, candidate, post, cutting_length_mm=5, entry_mode="virgin",
                section_depths_mm=(1, 2), max_protected_overcut_mm2=0.001,
                min_new_floor_area_mm2=1)
            self.assertEqual((checked_series, checked), (series, audit))
            with self.assertRaisesRegex(ValueError, "useful floor removal"):
                native_rest.audit(binding, candidate, post, cutting_length_mm=5,
                                  entry_mode="virgin", section_depths_mm=(2,),
                                  max_protected_overcut_mm2=0.001,
                                  min_new_floor_area_mm2=100)
            job = from_native_series(series, **args)
            source_binding = NativeBinding(series, binding.source_path, candidate, post,
                                           SETUP, binding)
            self.assertTrue(source_binding.check(job))
            for dialect in ("uccnc", "grbl"):
                output_job = from_native_series(series, **args,
                    boundary="split" if dialect == "uccnc" else "pause")
                write_bundle(root / dialect, output_job, dialect, source_binding=source_binding)
            with self.assertRaises(ValueError):
                audit_linear_native_series(
                    series, binding.source_path, candidate, post,
                    **dict(args, entry_modes={"ROUGH": "virgin", "REST": "cleared"}),
                    section_depths_mm=(2,), max_protected_overcut_mm2=0.001,
                    setup=SETUP, derived_binding=binding)

    def test_mutations_and_changed_predecessor_cannot_reuse_certification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binding = make_case(root, island=False)
            candidate = self.author(root, binding)
            post = root / "rest.nc"
            series = add_cleanup_post(binding, candidate, post, changed_prior=True)
            with self.assertRaisesRegex(ValueError, "predecessor changed"):
                binding.check(binding.source_path, candidate, series, binding.boundaries.target)
            forged = replace(binding, boundaries=replace(binding.boundaries, windows=binding.boundaries.windows[:-1]))
            with self.assertRaisesRegex(ValueError, "derivation changed"):
                forged.check_derivation()
            project = read_cambam_bytes(candidate.read_bytes())
            project.get_primitive(binding.names[0]).description = "forged"
            project.save(str(candidate))
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                binding.check_candidate(candidate)

    def test_unrepresentable_or_invalid_controls_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            binding = make_case(Path(folder))
            trace = binding.check_derivation()
            for radius, overlap in ((3, 0.5), (0, 0.5), (1, -1), (float("nan"), 1)):
                with self.subTest(radius=radius), self.assertRaises(ValueError):
                    rest_boundaries.derive(trace, radius_mm=radius, overlap_mm=overlap)
            for protected in (binding.source_path, binding.prior_candidate_path,
                              binding.prior_post_path):
                before = Path(protected).read_bytes()
                with self.assertRaisesRegex(ValueError, "preserve original"):
                    native_rest.author(binding, protected, tool_number=2,
                        depth_increment_mm=1, clearance_mm=5, spindle_rpm=12000,
                        plunge_feed_mm_min=60, cut_feed_mm_min=240)
                self.assertEqual(Path(protected).read_bytes(), before)

    def test_fresh_candidate_mutations_reject_in_audit_and_ordered_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binding = make_case(root)
            candidate = self.author(root, binding)
            post = root / "rest.nc"
            add_cleanup_post(binding, candidate, post)
            original = candidate.read_bytes()
            for edit in ("geometry", "floor", "diameter", "header", "selection", "setup"):
                with self.subTest(edit=edit):
                    project = read_cambam_bytes(original)
                    mop = project.list_mops()[-1]
                    if edit == "geometry":
                        project.get_primitive(binding.names[0]).local_z_offset = 0.001
                    elif edit == "floor":
                        mop.target_depth = -1
                    elif edit == "diameter":
                        mop.tool_diameter = 1.5
                    elif edit == "header":
                        mop.custom_mop_header = "G0 Z5"
                    elif edit == "selection":
                        project.set_mop_targets(mop, [project.get_primitive(binding.names[0])])
                    else:
                        project.list_mops()[0].custom_mop_footer = ""
                    project.save(str(candidate))
                    fresh = normalize_native_series(binding.source_path, candidate, post,
                        initial_position=START, setup=SETUP)
                    args = dict(targets={s.name: binding.boundaries.target for s in fresh.stages},
                        cutting_lengths_mm={"T1": 5, "T2": 5},
                        entry_modes={"ROUGH": "virgin", "REST": "virgin"})
                    with self.assertRaises(ValueError):
                        audit_linear_native_series(fresh, binding.source_path, candidate,
                            post, **args, section_depths_mm=(2,),
                            max_protected_overcut_mm2=0.001, setup=SETUP, derived_binding=binding)
                    job = from_native_series(fresh, **args)
                    source_binding = NativeBinding(fresh, binding.source_path, candidate,
                                                   post, SETUP, binding)
                    with self.assertRaises(ValueError):
                        write_bundle(root / edit, job, "uccnc", source_binding=source_binding)

    def test_rect_source_accepts_bare_rectangular_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binding = make_case(root, island=False)
            # Change the original and its retained predecessor to a native Rect,
            # preserving that new Rect's identity across the two documents.
            project = read_cambam_bytes(Path(binding.source_path).read_bytes())
            project.remove_primitive("design")
            rect = project.add_rect(project.list_layers()[0], (0, 0), 40, 30,
                                    identifier="design")
            project.save(str(binding.source_path))
            project.add_pocket_mop(project.list_parts()[0], targets=[rect],
                name="ROUGH", tool_number=1, tool_diameter=6, tool_profile="EndMill",
                target_depth=-2, depth_increment=2, stock_surface=0)
            project.save(str(binding.prior_candidate_path))
            prior = normalize_native_series(binding.source_path,
                binding.prior_candidate_path, binding.prior_post_path,
                initial_position=START, setup=SETUP)
            bare = replay.Target("design", (0, 0, 40, 30), 2)
            current = native_rest.prepare(prior, binding.source_path,
                binding.prior_candidate_path, binding.prior_post_path, target=bare,
                cutting_lengths_mm={"T1": 5}, entry_modes={"ROUGH": "virgin"},
                radius_mm=1, overlap_mm=0.5, setup=SETUP)
            self.assertTrue(current.boundaries.target.region_shell)
            self.assertEqual(len(current.boundaries.windows), 4)
            current.check_derivation()


if __name__ == "__main__":
    unittest.main()
