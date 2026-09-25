"""M2 curved Region rest: source-bound stock, visual preview and literal post gate."""

import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import LineString

from ...cam_core import curved_region, replay
from ...native.cad import Pline, Points, Vertex
from ...native.cam import DrillMop, EngraveMop
from ...native.project import CamBamProject as CBProject
from ...native.reader import read_cambam_bytes
from ...native.region import Region
from ...cad_transformations import scale_matrix, translation_matrix
from .native_polygon_rest import _record, _script
from .rc01_post import read_default_post


DEPTHS = (2, 4)
T1 = replay.ToolProfile("T1", "cylinder", 1.5, 8)
T2 = replay.ToolProfile("T2", "cylinder", 0.75, 8)
ALLOWANCE = 0.25
RPM = 12000


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _circle(cx, cy, radius, *, clockwise=False):
    angles = (0, -math.pi / 2, -math.pi, -3 * math.pi / 2) if clockwise else (
        0, math.pi / 2, math.pi, 3 * math.pi / 2)
    bulge = math.tan((-1 if clockwise else 1) * math.pi / 8)
    return Pline(vertices=[Vertex(cx + radius * math.cos(a),
                                  cy + radius * math.sin(a), 0, bulge=bulge)
                           for a in angles], closed=True)


def synthetic_source(case):
    project = CBProject("M2 " + case + " curved Region")
    layer = project.add_layer("Original curved Region")
    if case == "annulus":
        outer, hole = _circle(0, 0, 9), _circle(0, 0, 2, clockwise=True)
        bounds = (-12, -12, 12, 12)
    elif case in ("mixed", "mixed_reflected"):
        shell = [(-16, -10, 0.17), (16, -10, 0), (16, 10, 0),
                 (5, 10, 0), (5, 4, 0), (-5, 4, 0), (-5, 10, 0),
                 (-16, 10, 0)]
        outer = Pline(vertices=[Vertex(x, y, 0, bulge=b)
                                for x, y, b in shell], closed=True)
        hole = _circle(0, -3, 1.5, clockwise=True)
        bounds = (-20, -15, 20, 14) if case == "mixed" else (20, -8, 60, 21)
    else:
        raise ValueError("unknown M2 synthetic case")
    region = project.add_region(layer, outer, [hole], identifier="curved-finish")
    if case == "mixed_reflected" and region is not None:
        region.effective_transform = (translation_matrix(40, 7) @
                                      scale_matrix(-1, 1))
        region.validate_geometry()
    part = project.add_part("Curved part", stock_thickness=4,
                            stock_width=bounds[2] - bounds[0],
                            stock_height=bounds[3] - bounds[1],
                            stock_offset=bounds[:2], stock_surface=0,
                            machining_origin=(0, 0), nesting_method="None")
    if region is None or part is None:
        raise RuntimeError("could not build curved source")
    return project


def _source(data):
    project = read_cambam_bytes(data, source_name="M2 curved source")
    shapes, parts = project.list_primitives(), project.list_parts()
    if (len(shapes) != 1 or type(shapes[0]) is not Region or
            len(parts) != 1 or project.list_mops()):
        raise ValueError("M2 source needs one original Region and one Part")
    region, part = shapes[0], parts[0]
    if (not part.enabled or part.nesting_method != "None" or
            part.stock_surface != 0 or part.stock_thickness != 4 or
            not region.user_identifier):
        raise ValueError("unsupported M2 Part or Region")
    xyz = region.get_absolute_coordinates_xyz()
    rings = []
    for coordinates in (xyz["outer_curve"],) + tuple(xyz["hole_curves"]):
        if any(abs(z) > 1e-9 for _, _, z, _ in coordinates):
            raise ValueError("M2 requires planar zero-Z contours")
        rings.append(tuple((x, y, bulge) for x, y, _, bulge in coordinates))
    if not any(abs(b) > 1e-14 for ring in rings for _, _, b in ring):
        raise ValueError("M2 source has no curved boundary")
    approximation = curved_region.approximate(rings[0], rings[1:])
    xmin, ymin, xmax, ymax = part.stock_drawing_origin[:2] + (
        part.stock_drawing_origin[0] + part.stock_width,
        part.stock_drawing_origin[1] + part.stock_height)
    if (approximation.outer.bounds[0] < xmin or
            approximation.outer.bounds[1] < ymin or
            approximation.outer.bounds[2] > xmax or
            approximation.outer.bounds[3] > ymax):
        raise ValueError("curved target outside Part stock")
    start = (xmin - 5.0, ymin - 5.0, 5.0)
    return project, region, approximation, start


def _append_path(items, path, depth, start):
    first = path[0]
    high = (first[0], first[1], start[2])
    at = items[-1].position if type(items[-1]) is replay.Event else items[-1].end
    if at != high:
        items.append(replay.Motion("rapid", "T1", "prior", at, high))
    low = (first[0], first[1], -depth)
    items.append(replay.Motion("entry", "T1", "prior", high, low, 60))
    at = low
    for point in path[1:]:
        end = (point[0], point[1], -depth)
        if at != end:
            items.append(replay.Motion("cut", "T1", "prior", at, end, 300))
        at = end
    items.append(replay.Motion("retract", "T1", "prior", at,
                               (at[0], at[1], start[2]), 300))


def _synthetic_prior_trace(source_bytes):
    _, region, approximation, start = _source(source_bytes)
    target = approximation.target(region.user_identifier, 4)
    # This is supplied test motion, not a recommendation for production roughing.
    centers = approximation.safe.buffer(-(T1.radius + ALLOWANCE + 0.01),
                                         quad_segs=32)
    if centers.geom_type != "Polygon" or centers.is_empty:
        raise ValueError("synthetic curved rough centers disconnected")
    paths = []
    for ring in (centers.exterior,) + tuple(centers.interiors):
        points = [tuple(round(v, 6) for v in point) for point in ring.coords]
        if len(points) > 2:
            paths.append(points)
    y = math.floor(centers.bounds[1]) + 1
    while y < centers.bounds[3]:
        section = centers.intersection(LineString(((centers.bounds[0] - 1, y),
                                                    (centers.bounds[2] + 1, y))))
        segments = ([section] if section.geom_type == "LineString" else
                    list(section.geoms) if section.geom_type == "MultiLineString" else [])
        for segment in segments:
            if segment.length > 0.1:
                paths.append([tuple(round(v, 6) for v in point)
                              for point in (segment.coords[0], segment.coords[-1])])
        y += 2
    items = [replay.Event("tool_change", "T1", start),
             replay.Event("spindle_start", "T1", start)]
    for depth in DEPTHS:
        for path in paths:
            _append_path(items, path, depth, start)
    at = items[-1].end
    if at != start:
        items.append(replay.Motion("rapid", "T1", "prior", at, start))
    items.append(replay.Event("spindle_stop", "T1", start))
    trace = replay.Trace(_sha(source_bytes), "curved-drawing", start,
                         (replay.Operation("prior", T1, target),), tuple(items))
    replay.replay(trace, expected_source=_sha(source_bytes))
    return trace


def synthetic_prior(source_bytes):
    trace = _synthetic_prior_trace(source_bytes)
    return {"format": "m2-curved-prior-v1", "source_sha256": _sha(source_bytes),
            "units": "mm", "frame": trace.frame, "start": list(trace.initial_position),
            "rough_allowance_mm": ALLOWANCE,
            "motion_fingerprint": trace.motion_fingerprint,
            "items": [_record(item) for item in trace.items]}


def plan(source_bytes, supplied):
    _, region, approximation, start = _source(source_bytes)
    if (type(supplied) is not dict or set(supplied) != {
            "format", "source_sha256", "units", "frame", "start",
            "rough_allowance_mm", "motion_fingerprint", "items"} or
            supplied["format"] != "m2-curved-prior-v1" or
            supplied["source_sha256"] != _sha(source_bytes) or
            supplied["units"] != "mm" or supplied["frame"] != "curved-drawing" or
            tuple(supplied["start"]) != start or
            supplied["rough_allowance_mm"] != ALLOWANCE or
            type(supplied["items"]) is not list):
        raise ValueError("stale or unsupported M2 supplied prior")
    items = []
    for row in supplied["items"]:
        if set(row) == {"kind", "tool", "position"}:
            items.append(replay.Event(row["kind"], row["tool"],
                                      tuple(row["position"])))
        elif set(row) == {"role", "tool", "operation", "start", "end", "feed"}:
            items.append(replay.Motion(row["role"], row["tool"],
                                       row["operation"], tuple(row["start"]),
                                       tuple(row["end"]), row["feed"]))
        else:
            raise ValueError("invalid M2 supplied item")
    target = approximation.target(region.user_identifier, 4)
    trace = replay.Trace(_sha(source_bytes), "curved-drawing", start,
                         (replay.Operation("prior", T1, target),), tuple(items))
    if trace.motion_fingerprint != supplied["motion_fingerprint"]:
        # The v1 fingerprint includes the replay target's repr. Earlier
        # prepared mixed cases have the same source and every T1 item, but a
        # changed target repr. Admit only the exact source-derived fixture
        # motion; arbitrary supplied traces still require their fingerprint.
        fixture = _synthetic_prior_trace(source_bytes)
        if (tuple(items) != fixture.items or
                tuple(supplied["start"]) != fixture.initial_position):
            raise ValueError("M2 supplied motion changed")
    result = curved_region.generate(
        approximation, trace, T2, expected_source=_sha(source_bytes),
        expected_motion=trace.motion_fingerprint,
        rough_allowance_mm=ALLOWANCE)
    return result


def _preview_paths(trace):
    paths, pending = [], []
    for item in trace.items:
        if (type(item) is replay.Motion and item.tool == "T2" and
                item.role == "cut" and item.end[2] == -4):
            if not pending:
                pending = [item.start]
            pending.append(item.end)
        elif pending:
            paths.append(tuple(pending))
            pending = []
    if pending:
        raise ValueError("unfinished M2 preview path")
    return tuple(paths)


def _make_preview(project, trace):
    layer = project.add_layer("Generated T2 preview only")
    paths = []
    for index, points in enumerate(_preview_paths(trace)):
        path = project.add_pline(layer, points,
                                 identifier=f"m2-preview-path-{index}")
        if path is None:
            raise RuntimeError("could not add M2 preview path")
        paths.append(path)
    mop = project.add_engrave_mop(
        project.list_parts()[0], targets=paths,
        name="PREVIEW T2 curved cleanup only", identifier="m2-preview",
        enabled=True, tool_number=2, tool_diameter=1.5,
        tool_profile="EndMill", stock_surface=0, target_depth=0,
        depth_increment=4, roughing_clearance=0, clearance_plane=5,
        plunge_feedrate=60, cut_feedrate=300, spindle_direction="CW",
        spindle_speed=RPM, work_plane="XY", velocity_mode="ExactStop",
        optimisation_mode="None")
    if mop is None or not paths:
        raise RuntimeError("could not add M2 preview")


def _make_explicit(project, trace):
    start = trace.initial_position
    layer = project.add_layer("M2 literal motion anchor")
    anchor = project.add_points(layer, [(start[0], start[1], 0)],
                                identifier="m2-script-anchor")
    mop = project.add_drill_mop(
        project.list_parts()[0], targets=[anchor],
        name="M2 T1 rough plus T2 cleanup literal motion",
        identifier="m2-literal", enabled=True,
        drilling_method="CustomScript", custom_script=_script(trace),
        tool_number=1, tool_diameter=3, tool_profile="EndMill",
        stock_surface=0, target_depth=-4, depth_increment=2,
        clearance_plane=5, plunge_feedrate=60, cut_feedrate=300,
        spindle_direction="CW", spindle_speed=RPM, work_plane="XY",
        velocity_mode="ExactStop")
    if anchor is None or mop is None:
        raise RuntimeError("could not add M2 literal carrier")


def _check_candidate(data, source_bytes, kind, trace):
    project = read_cambam_bytes(data, source_name="M2 candidate")
    original, source_region, _, _ = _source(source_bytes)
    region = project.get_primitive(source_region.user_identifier)
    if (type(region) is not Region or
            region.internal_id != source_region.internal_id or
            region.get_absolute_coordinates_xyz() !=
            source_region.get_absolute_coordinates_xyz()):
        raise ValueError("M2 candidate changed analytic source Region")
    a, b = project.list_parts()[0], original.list_parts()[0]
    if (tuple(a.stock_drawing_origin) != tuple(b.stock_drawing_origin) or
            (a.stock_width, a.stock_height, a.stock_thickness, a.stock_surface) !=
            (b.stock_width, b.stock_height, b.stock_thickness, b.stock_surface)):
        raise ValueError("M2 candidate changed stock")
    enabled = [m for m in project.list_mops() if m.enabled]
    if len(enabled) != 1 or len(project.list_mops()) != 1:
        raise ValueError("M2 candidate changed enabled operation")
    if kind == "explicit":
        anchor = project.get_primitive("m2-script-anchor")
        mop = enabled[0]
        if (len(project.list_primitives()) != 2 or type(anchor) is not Points or
                type(mop) is not DrillMop or
                mop.drilling_method != "CustomScript" or
                mop.custom_script != _script(trace) or
                project.get_mop_targets(mop) != [anchor.internal_id]):
            raise ValueError("M2 literal candidate changed")
    elif kind == "preview":
        paths = _preview_paths(trace)
        stored = [project.get_primitive(f"m2-preview-path-{i}")
                  for i in range(len(paths))]
        if (len(project.list_primitives()) != len(paths) + 1 or
                any(type(path) is not Pline or path.closed or
                    tuple(path.get_absolute_coordinates_xyz()) !=
                    tuple((x, y, z, 0.0) for x, y, z in points)
                    for path, points in zip(stored, paths)) or
                type(enabled[0]) is not EngraveMop or
                set(project.get_mop_targets(enabled[0])) !=
                {path.internal_id for path in stored}):
            raise ValueError("M2 preview candidate changed")
    else:
        raise ValueError("unknown M2 candidate")


def build_workflow(directory, *, case="annulus", source_path=None,
                   prior_path=None):
    if source_path is not None and prior_path is None:
        raise ValueError("existing curved source requires supplied prior motion")
    directory = Path(directory)
    if directory.exists():
        raise ValueError("M2 output directory must be new")
    directory.mkdir(parents=True)
    source = directory / "source.cb"
    if source_path is None:
        synthetic_source(case).save(str(source))
    else:
        source.write_bytes(Path(source_path).read_bytes())
    source_bytes = source.read_bytes()
    supplied = (synthetic_prior(source_bytes) if prior_path is None else
                json.loads(Path(prior_path).read_text(encoding="utf-8")))
    prior_file = directory / "prior.json"
    prior_file.write_text(json.dumps(supplied, indent=2) + "\n", encoding="utf-8")
    result = plan(source_bytes, supplied)
    trace = result.planned.trace
    candidates = {}
    for kind in ("preview", "explicit"):
        project = read_cambam_bytes(source_bytes, source_name="M2 source")
        (_make_preview if kind == "preview" else _make_explicit)(project, trace)
        path = directory / kind / f"m2-{kind}.cb"
        path.parent.mkdir()
        project.save(str(path))
        _check_candidate(path.read_bytes(), source_bytes, kind, trace)
        candidates[kind] = {"file": str(path.relative_to(directory)),
                            "sha256": _sha(path.read_bytes())}
    depths = (1, 3)
    manifest = {
        "format": "m2-curved-rest-v1", "source_sha256": _sha(source_bytes),
        "prior_sha256": _sha(prior_file.read_bytes()),
        "prior_motion_fingerprint": result.planned.prior_trace.motion_fingerprint,
        "motion_fingerprint": trace.motion_fingerprint,
        "candidates": candidates, "expected_items": [_record(i) for i in trace.items],
        "script_lines": _script(trace).splitlines(),
        "analytic_area_mm2": result.approximation.analytic_area_mm2,
        "chord_area_error_mm2": result.approximation.area_error_mm2,
        "sagitta_mm": result.approximation.sagitta_mm,
        "chord_segments": result.approximation.segment_count,
        "rough_rest_mm2": {str(d): result.rest_area(d, final=False) for d in depths},
        "final_rest_mm2": {str(d): result.rest_area(d, final=True) for d in depths},
        "rough_rest_mm3": result.rest_volume(final=False),
        "final_rest_mm3": result.rest_volume(final=True),
        "protected_overcut_upper_mm2": {
            str(d): result.protected_overcut_upper_area(d) for d in depths},
        "setup": {"units": "mm", "initial_tip_xyz_mm": list(trace.initial_position),
                  "stock_top_z_mm": 0, "stock_bottom_z_mm": -4,
                  "tool_diameters_mm": [3, 1.5], "cutting_lengths_mm": [8, 8],
                  "spindle_rpm": RPM, "entry_feed_mm_min": 60,
                  "cut_feed_mm_min": 300},
        "postprocessor": "Default", "profile": "Default mm",
        "post_status": "pending_actual_CamBam_post",
    }
    (directory / "expected-motion.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in manifest.items()
            if k not in ("expected_items", "script_lines")}


def audit_post(manifest_path, posted_path):
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("format") != "m2-curved-rest-v1":
        raise ValueError("not an M2 curved rest manifest")
    source = directory / "source.cb"
    prior = directory / "prior.json"
    if (_sha(source.read_bytes()) != expected["source_sha256"] or
            _sha(prior.read_bytes()) != expected["prior_sha256"]):
        raise ValueError("M2 source or supplied prior changed")
    result = plan(source.read_bytes(), json.loads(prior.read_text(encoding="utf-8")))
    trace = result.planned.trace
    fingerprints_match = (
        trace.motion_fingerprint == expected["motion_fingerprint"] and
        result.planned.prior_trace.motion_fingerprint ==
        expected["prior_motion_fingerprint"])
    if (not fingerprints_match and
            (expected["prior_motion_fingerprint"] !=
             json.loads(prior.read_text(encoding="utf-8"))["motion_fingerprint"] or
             result.planned.prior_trace.items !=
             _synthetic_prior_trace(source.read_bytes()).items)):
        raise ValueError("M2 expected motion fingerprint changed")
    if ([_record(i) for i in trace.items] != expected["expected_items"] or
            _script(trace).splitlines() != expected["script_lines"]):
        raise ValueError("M2 expected motion changed")
    setup = {"units": "mm", "initial_tip_xyz_mm": list(trace.initial_position),
             "stock_top_z_mm": 0, "stock_bottom_z_mm": -4,
             "tool_diameters_mm": [3, 1.5], "cutting_lengths_mm": [8, 8],
             "spindle_rpm": RPM, "entry_feed_mm_min": 60,
             "cut_feed_mm_min": 300}
    if (expected["setup"] != setup or expected["postprocessor"] != "Default" or
            expected["profile"] != "Default mm" or
            expected["analytic_area_mm2"] !=
            result.approximation.analytic_area_mm2 or
            expected["chord_area_error_mm2"] !=
            result.approximation.area_error_mm2 or
            expected["sagitta_mm"] != result.approximation.sagitta_mm or
            expected["chord_segments"] != result.approximation.segment_count or
            expected["rough_rest_mm3"] != list(result.rest_volume(final=False)) or
            expected["final_rest_mm3"] != list(result.rest_volume(final=True)) or
            any(expected["rough_rest_mm2"][str(d)] !=
                list(result.rest_area(d, final=False)) or
                expected["final_rest_mm2"][str(d)] !=
                list(result.rest_area(d, final=True)) or
                expected["protected_overcut_upper_mm2"][str(d)] !=
                result.protected_overcut_upper_area(d) for d in (1, 3))):
        raise ValueError("M2 curved source, setup or residual manifest changed")
    for kind, entry in expected["candidates"].items():
        candidate = directory / entry["file"]
        if _sha(candidate.read_bytes()) != entry["sha256"]:
            raise ValueError("M2 candidate changed")
        _check_candidate(candidate.read_bytes(), source.read_bytes(), kind, trace)
    candidate = directory / expected["candidates"]["explicit"]["file"]
    post = Path(posted_path).read_text(encoding="utf-8-sig")
    lines = post.splitlines()
    if ("( Post processor: Default )" not in lines[:12] or
            not any(line.startswith(f"( {candidate.stem} ") for line in lines[:12])):
        raise ValueError("M2 post header does not match explicit candidate")
    normalized = "\n".join("" if line.strip() in ("G98", "G80") else line
                           for line in lines)
    actual, warnings = read_default_post(
        normalized, initial_position=trace.initial_position)
    warnings = [w for w in warnings if
                "initial machine position is not encoded" not in w]
    if warnings:
        return {"status": "unverified", "reason": warnings[0]}
    if (len(actual) == len(trace.items) + 1 and
            actual[-1]["type"] == "event" and
            actual[-1]["kind"] == "spindle_stop" and
            actual[-1]["tool"] == "T2" and
            tuple(actual[-1]["position"]) == trace.initial_position):
        actual = actual[:-1]
    if len(actual) != len(trace.items):
        return {"status": "deviation", "reason": "extra or missing post item",
                "expected_items": len(trace.items), "actual_items": len(actual)}
    observed = []
    for index, (want, got) in enumerate(zip(trace.items, actual)):
        if type(want) is replay.Motion:
            if (got["type"] != "move" or got["tool"] != want.tool or
                    got["g"] != (0 if want.role == "rapid" else 1) or
                    got["feed"] != want.feed or
                    tuple(got["start"]) != want.start or
                    tuple(got["end"]) != want.end):
                return {"status": "deviation", "item": index, "actual": got}
            observed.append(replay.Motion(want.role, want.tool, want.operation,
                                          tuple(got["start"]), tuple(got["end"]),
                                          got["feed"]))
        else:
            if (got["type"] != "event" or got["kind"] != want.kind or
                    got["tool"] != want.tool or
                    tuple(got["position"]) != want.position or
                    (want.kind == "spindle_start" and got["rpm"] != RPM)):
                return {"status": "deviation", "item": index, "actual": got}
            observed.append(replay.Event(want.kind, want.tool,
                                         tuple(got["position"])))
    posted_trace = replay.Trace(trace.source_fingerprint, trace.frame,
                                trace.initial_position, trace.operations,
                                tuple(observed))
    stock = replay.replay(posted_trace, expected_source=posted_trace.source_fingerprint)
    posted_result = curved_region.RestResult(result.approximation,
        result.planned.__class__(result.planned.prior_trace, posted_trace,
                                 result.planned.prior_stock, stock, ALLOWANCE))
    areas = {str(d): posted_result.rest_area(d, final=True) for d in (1, 3)}
    if (any(abs(areas[str(d)][i] - expected["final_rest_mm2"][str(d)][i]) > 1e-8
            for d in (1, 3) for i in (0, 1)) or
            any(posted_result.protected_overcut_upper_area(d) > 1e-5
                for d in (1, 3))):
        return {"status": "deviation", "reason": "posted curved stock changed"}
    return {"status": "bounded_m2_curved_post_pass",
            "post_sha256": _sha(Path(posted_path).read_bytes()),
            "candidate_sha256": expected["candidates"]["explicit"]["sha256"],
            "item_count": len(actual), "stock_prefixes": stock.prefixes,
            "rough_rest_mm2": expected["rough_rest_mm2"],
            "final_rest_mm2": areas, "final_rest_mm3": posted_result.rest_volume(final=True),
            "numeric_limit": "GEOS topology conditional; analytic source area and chord error bounded"}


def _preview_xy_paths(items):
    paths, pending = [], []
    for item in items:
        if (item.get("role") == "cut" and item.get("tool") == "T2" and
                item["start"][2] == -4 and item["end"][2] == -4):
            if not pending:
                pending = [item["start"]]
            pending.append(item["end"])
        elif pending:
            paths.append(pending)
            pending = []
    if pending:
        paths.append(pending)
    return paths


def _posted_preview_xy_paths(items):
    paths, pending = [], []
    for item in items:
        if item["type"] != "move":
            continue
        if (item["tool"] != "T2" or min(item["start"][2], item["end"][2]) < -4 or
                (item["g"] == 0 and
                 min(item["start"][2], item["end"][2]) < 0 and
                 (item["start"][:2] != item["end"][:2] or
                  item["end"][2] < item["start"][2]))):
            raise ValueError("unsafe M2 preview motion")
        if item["g"] == 1 and item["start"][2] == item["end"][2] == -4:
            if item["feed"] != 300:
                raise ValueError("M2 preview cut feed changed")
            if not pending:
                pending = [item["start"]]
            pending.append(item["end"])
        else:
            if pending:
                paths.append(pending)
                pending = []
            if (item["g"] == 1 and item["end"][2] == -4 and
                    (item["start"][:2] != item["end"][:2] or
                     item["feed"] != 60)):
                raise ValueError("M2 preview entry changed")
    if pending:
        paths.append(pending)
    return paths


def _preview_coordinates(points):
    answer = []
    rounded = []
    for x, y, _ in points:
        xy = (round(x, 4), round(y, 4))
        if not rounded or rounded[-1] != xy:
            rounded.append(xy)
            answer.append((x, y))
    return tuple(answer)


def audit_preview(manifest_path, posted_path):
    """Compare CamBam's Engrave centerlines, not its entire execution plan."""
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = directory / "source.cb"
    prior = directory / "prior.json"
    candidate = directory / expected["candidates"]["preview"]["file"]
    if (expected["format"] != "m2-curved-rest-v1" or
            _sha(source.read_bytes()) != expected["source_sha256"] or
            _sha(prior.read_bytes()) != expected["prior_sha256"] or
            _sha(candidate.read_bytes()) !=
            expected["candidates"]["preview"]["sha256"]):
        raise ValueError("M2 preview source or candidate changed")
    _, region, approximation, start = _source(source.read_bytes())
    target = approximation.target(region.user_identifier, 4)
    recorded = []
    for row in expected["expected_items"]:
        if set(row) == {"kind", "tool", "position"}:
            recorded.append(replay.Event(row["kind"], row["tool"],
                                         tuple(row["position"])))
        elif set(row) == {"role", "tool", "operation", "start", "end", "feed"}:
            recorded.append(replay.Motion(row["role"], row["tool"],
                                          row["operation"], tuple(row["start"]),
                                          tuple(row["end"]), row["feed"]))
        else:
            raise ValueError("M2 preview expected item changed")
    trace = replay.Trace(_sha(source.read_bytes()), "curved-drawing", start,
                         (replay.Operation("prior", T1, target),
                          replay.Operation("cleanup", T2, target)), tuple(recorded))
    _check_candidate(candidate.read_bytes(), source.read_bytes(), "preview", trace)
    lines = Path(posted_path).read_text(encoding="utf-8-sig").splitlines()
    if ("( Post processor: Default )" not in lines[:12] or
            not any(line.startswith(f"( {candidate.stem} ") for line in lines[:12]) or
            not any(line.startswith("G21 ") for line in lines[:12])):
        raise ValueError("M2 preview post header or units changed")
    actual, warnings = read_default_post(
        "\n".join(lines),
        initial_position=tuple(expected["setup"]["initial_tip_xyz_mm"]))
    warnings = [w for w in warnings if
                "initial machine position is not encoded" not in w]
    if warnings:
        return {"status": "unverified", "reason": warnings[0]}
    wanted_paths = tuple(_preview_coordinates(path) for path in
                         _preview_xy_paths(expected["expected_items"]))
    observed_paths = tuple(_preview_coordinates(path) for path in
                           _posted_preview_xy_paths(actual))
    if (len({len(path) for path in wanted_paths}) != len(wanted_paths) or
            len(wanted_paths) != len(observed_paths)):
        raise ValueError("unsupported M2 preview path count or ambiguous lengths")
    wanted_paths = sorted(wanted_paths, key=len)
    observed_paths = sorted(observed_paths, key=len)
    if (any(len(want) != len(got) or
            not any(all(abs(a - b) <= 0.000051 for p, q in zip(want, ordered)
                        for a, b in zip(p, q))
                    for ordered in (got, got[::-1]))
            for want, got in zip(wanted_paths, observed_paths))):
        return {"status": "deviation", "reason": "preview centerlines differ",
                "expected_lengths": [len(path) for path in wanted_paths],
                "actual_lengths": [len(path) for path in observed_paths]}
    return {"status": "m2_native_preview_centerlines_pass",
            "post_sha256": _sha(Path(posted_path).read_bytes()),
            "path_count": len(observed_paths),
            "vertex_counts": [len(path) for path in observed_paths],
            "scope": "Engrave centerlines at Z=-4 only"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("posted", nargs="?")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--case", choices=("annulus", "mixed", "mixed_reflected"),
                        default="annulus")
    parser.add_argument("--source")
    parser.add_argument("--prior")
    args = parser.parse_args()
    answer = (audit_preview(args.path, args.posted) if args.preview else
              audit_post(args.path, args.posted) if args.posted else
              build_workflow(args.path, case=args.case,
                             source_path=args.source, prior_path=args.prior))
    print(json.dumps(answer, indent=2))
