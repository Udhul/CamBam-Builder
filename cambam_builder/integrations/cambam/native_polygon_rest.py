"""M1 letter-like Region: supplied rough stock, smaller endmill, native carriers."""

import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import GeometryCollection, LineString, Polygon as ShapelyPolygon

from ...cam_core import polygon_rest, replay
from ...native.cad import Pline, Points
from ...native.cam import DrillMop, EngraveMop, PocketMop
from ...native.project import CamBamProject as CBProject
from ...native.reader import read_cambam_bytes
from ...native.region import Region
from .rc01_post import read_default_post
from .rc01_native_post import _path_for_move


SHELL = ((-24, 0), (-8, 60), (8, 60), (24, 0),
         (12, 0), (6, 18), (-6, 18), (-12, 0))
HOLE = ((-4, 28), (4, 28), (0, 44))
START = (-30.0, -10.0, 5.0)
DEPTHS = (2, 4, 6, 8)
RPM = 12000
NATIVE_FOOTER = "G0 Z5\nG0 X-30 Y-10\nM5"
SETUP = {
    "units": "mm", "frame": "letter-drawing",
    "tip_datum": "flat_endmill_bottom", "initial_tip_xyz_mm": list(START),
    "stock_top_z_mm": 0, "stock_bottom_z_mm": -8,
    "max_new_axial_depth_mm": 2, "center_cutting": True,
    "synthetic_full_width_slotting": True,
    "spindle_rpm": RPM, "entry_feed_mm_min": 60,
    "cut_retract_feed_mm_min": 300, "coolant": "off",
    "tools": [
        {"name": "T1", "diameter_mm": 5, "cutting_length_mm": 10,
         "shank_start_above_tip_mm": 10, "holder_start_above_tip_mm": 20},
        {"name": "T2", "diameter_mm": 2, "cutting_length_mm": 10,
         "shank_start_above_tip_mm": 10, "holder_start_above_tip_mm": 20},
    ],
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _target():
    return replay.Target("letter-opening", (-24, 0, 24, 60), 8,
                         region_shell=SHELL, region_holes=(HOLE,))


def synthetic_source():
    project = CBProject("M1 letter Region rest source")
    layer = project.add_layer("Original letter Region")
    region = project.add_region(layer, Pline(vertices=SHELL, closed=True),
                                [Pline(vertices=HOLE, closed=True)],
                                identifier="letter-finish")
    part = project.add_part("Letter part", stock_thickness=8,
                            stock_width=52, stock_height=64,
                            stock_offset=(-26, -2), stock_surface=0,
                            machining_origin=(0, 0), nesting_method="None")
    if None in (layer, region, part):
        raise RuntimeError("could not create M1 source")
    return project


def _source(data):
    project = read_cambam_bytes(data, source_name="M1 source")
    shapes, parts = project.list_primitives(), project.list_parts()
    if (len(shapes) != 1 or type(shapes[0]) is not Region or
            len(parts) != 1 or project.list_mops()):
        raise ValueError("M1 source requires one original Region and one Part")
    region, part = shapes[0], parts[0]
    xyz = region.get_absolute_coordinates_xyz()
    def ring(points):
        if any(z != 0 or bulge != 0 for _, _, z, bulge in points):
            raise ValueError("M1 source requires planar straight rings")
        return tuple((x, y) for x, y, _, _ in points)
    if (region.user_identifier != "letter-finish" or
            ring(xyz["outer_curve"]) != SHELL or
            len(xyz["hole_curves"]) != 1 or
            ring(xyz["hole_curves"][0]) != HOLE or
            not part.enabled or part.nesting_method != "None" or
            tuple(part.machining_origin) != (0, 0) or
            tuple(part.stock_drawing_origin) != (-26, -2, 0) or
            (part.stock_width, part.stock_height, part.stock_thickness,
             part.stock_surface) != (52, 64, 8, 0)):
        raise ValueError("M1 original Region or stock changed")
    return project, region


def _append_path(items, path, depth):
    """Full retreat between every prior path; no unproved low travel."""
    first = path[0]
    high = (first[0], first[1], START[2])
    at = items[-1].position if type(items[-1]) is replay.Event else items[-1].end
    if at != high:
        items.append(replay.Motion("rapid", "T1", "prior", at, high))
    low = (first[0], first[1], -depth)
    items.append(replay.Motion("entry", "T1", "prior", high, low, 60))
    at = low
    for point in path[1:]:
        end = (point[0], point[1], -depth)
        if end != at:
            items.append(replay.Motion("cut", "T1", "prior", at, end, 300))
        at = end
    items.append(replay.Motion("retract", "T1", "prior", at,
                               (at[0], at[1], START[2]), 300))


def _synthetic_prior_trace(source_sha):
    """Test fixture only: make ordered T1 motion for the supplied trace file."""
    region = ShapelyPolygon(SHELL, (HOLE,))
    centers = region.buffer(-3.01, quad_segs=16)
    if centers.geom_type != "Polygon":
        raise ValueError("synthetic rough centers disconnected")
    paths = []
    for ring in (centers.exterior,) + tuple(centers.interiors):
        points = [tuple(round(v, 6) for v in p) for p in ring.coords]
        if len(points) > 2:
            paths.append(points)
    for y in range(4, 58, 2):
        cross = centers.intersection(LineString(((-25, y), (25, y))))
        segments = ([cross] if cross.geom_type == "LineString" else
                    list(cross.geoms) if cross.geom_type == "MultiLineString" else [])
        for segment in segments:
            if segment.length > 0.05:
                a, b = (tuple(round(v, 6) for v in point)
                        for point in (segment.coords[0], segment.coords[-1]))
                paths.append([a, b])
    tool = replay.ToolProfile("T1", "cylinder", 2.5, 10)
    op = replay.Operation("prior", tool, _target())
    items = [replay.Event("tool_change", "T1", START),
             replay.Event("spindle_start", "T1", START)]
    for depth in DEPTHS:
        for path in paths:
            _append_path(items, path, depth)
    at = items[-1].end
    if at != START:
        items.append(replay.Motion("rapid", "T1", "prior", at, START))
    items.append(replay.Event("spindle_stop", "T1", START))
    trace = replay.Trace(source_sha, "letter-drawing", START, (op,), tuple(items))
    replay.replay(trace, expected_source=source_sha)
    return trace


def _record(item):
    if type(item) is replay.Event:
        return {"kind": item.kind, "tool": item.tool,
                "position": list(item.position)}
    return {"role": item.role, "tool": item.tool,
            "operation": item.operation, "start": list(item.start),
            "end": list(item.end), "feed": item.feed}


def synthetic_prior(source_bytes):
    trace = _synthetic_prior_trace(_sha(source_bytes))
    return {"format": "m1-prior-v1", "source_sha256": _sha(source_bytes),
            "units": "mm", "frame": "letter-drawing", "target_depth_mm": 8,
            "rough_allowance_mm": 0.5, "motion_fingerprint": trace.motion_fingerprint,
            "items": [_record(item) for item in trace.items]}


def _prior_trace(source_bytes, supplied):
    _source(source_bytes)
    if (type(supplied) is not dict or set(supplied) != {
            "format", "source_sha256", "units", "frame", "target_depth_mm",
            "rough_allowance_mm", "motion_fingerprint", "items"} or
            supplied["format"] != "m1-prior-v1" or
            supplied["source_sha256"] != _sha(source_bytes) or
            supplied["units"] != "mm" or supplied["frame"] != "letter-drawing" or
            supplied["target_depth_mm"] != 8 or
            supplied["rough_allowance_mm"] != 0.5 or
            type(supplied["items"]) is not list):
        raise ValueError("stale or unsupported M1 supplied prior")
    items = []
    for row in supplied["items"]:
        if type(row) is not dict:
            raise ValueError("invalid M1 supplied item")
        if set(row) == {"kind", "tool", "position"}:
            items.append(replay.Event(row["kind"], row["tool"],
                                      tuple(row["position"])))
        elif set(row) == {"role", "tool", "operation", "start", "end", "feed"}:
            items.append(replay.Motion(row["role"], row["tool"], row["operation"],
                                       tuple(row["start"]), tuple(row["end"]),
                                       row["feed"]))
        else:
            raise ValueError("invalid M1 supplied item")
    op = replay.Operation("prior", replay.ToolProfile("T1", "cylinder", 2.5, 10),
                          _target())
    trace = replay.Trace(_sha(source_bytes), "letter-drawing", START,
                         (op,), tuple(items))
    if trace.motion_fingerprint != supplied["motion_fingerprint"]:
        raise ValueError("M1 supplied motion changed")
    replay.replay(trace, expected_source=_sha(source_bytes))
    return trace


def plan(source_bytes, supplied):
    prior = _prior_trace(source_bytes, supplied)
    result = polygon_rest.generate(
        prior, replay.ToolProfile("T2", "cylinder", 1, 10),
        expected_source=_sha(source_bytes), expected_motion=prior.motion_fingerprint,
        rough_allowance_mm=0.5)
    return result


def _script(trace):
    def decimal(value):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value else "0"
    lines = []
    for item in trace.items[2:-1]:
        if type(item) is replay.Motion:
            x, y, z = item.end
            code = "G0" if item.role == "rapid" else "G1"
            feed = f" F{item.feed}" if item.feed else ""
            lines.append(f"{code}{feed} X{decimal(x)} Y{decimal(y)} Z{decimal(z)}")
        elif item.kind == "spindle_stop":
            lines.append("M5")
        elif item.kind == "tool_change":
            lines.append("T2 M6")
        elif item.kind == "spindle_start":
            lines.append(f"M3 S{RPM}")
        else:
            raise ValueError("unsupported M1 literal script event")
    return "\n".join(lines)


def _source_unchanged(project, source_bytes):
    source, original = _source(source_bytes)
    region = project.get_primitive("letter-finish")
    if (type(region) is not Region or
            region.internal_id != original.internal_id or
            region.get_absolute_coordinates_xyz() !=
            original.get_absolute_coordinates_xyz() or
            len(project.list_parts()) != 1):
        raise ValueError("M1 candidate changed the original Region")
    a, b = project.list_parts()[0], source.list_parts()[0]
    if (a.user_identifier != b.user_identifier or a.enabled != b.enabled or
            a.nesting_method != b.nesting_method or
            tuple(a.machining_origin) != tuple(b.machining_origin) or
            tuple(a.stock_drawing_origin) != tuple(b.stock_drawing_origin) or
            (a.stock_width, a.stock_height, a.stock_thickness, a.stock_surface) !=
            (b.stock_width, b.stock_height, b.stock_thickness, b.stock_surface)):
        raise ValueError("M1 candidate changed original stock")


def _check_candidate(data, source_bytes, kind, trace):
    project = read_cambam_bytes(data, source_name=f"M1 {kind} candidate")
    _source_unchanged(project, source_bytes)
    enabled = [m for m in project.list_mops() if m.enabled]
    if len(enabled) != len(project.list_mops()):
        raise ValueError("M1 candidate has unexpected disabled MOPs")
    if kind == "explicit":
        anchor = project.get_primitive("m1-script-anchor")
        if (len(project.list_primitives()) != 2 or type(anchor) is not Points or
                tuple(anchor.get_absolute_coordinates_xyz()) !=
                ((START[0], START[1], 0.0),) or
                len(enabled) != 1 or type(enabled[0]) is not DrillMop or
                enabled[0].drilling_method != "CustomScript" or
                enabled[0].custom_script != _script(trace) or
                (enabled[0].tool_number, enabled[0].tool_diameter,
                 enabled[0].tool_profile, enabled[0].target_depth,
                 enabled[0].depth_increment, enabled[0].clearance_plane,
                 enabled[0].plunge_feedrate, enabled[0].cut_feedrate,
                 enabled[0].spindle_speed) !=
                (1, 5, "EndMill", -8, 2, 5, 60, 300, RPM) or
                project.get_mop_targets(enabled[0]) != [anchor.internal_id]):
            raise ValueError("M1 literal carrier changed on strict reimport")
    elif kind == "native":
        region = project.get_primitive("letter-finish")
        if (len(project.list_primitives()) != 1 or len(enabled) != 2 or
                any(type(m) is not PocketMop for m in enabled) or
                any(project.get_mop_targets(m) != [region.internal_id]
                    for m in enabled) or
                [(m.tool_number, m.tool_diameter, m.roughing_clearance)
                 for m in enabled] != [(1, 5, 0.5), (2, 2, 0)] or
                any((m.lead_in_type, m.optimisation_mode, m.stepover_feedrate,
                     m.max_crossover_distance) !=
                    ("None", "None", "Cut Feedrate", 0) for m in enabled) or
                any((m.custom_mop_footer, m.target_depth, m.depth_increment,
                     m.clearance_plane, m.plunge_feedrate, m.cut_feedrate,
                     m.spindle_speed) !=
                    (NATIVE_FOOTER, -8, 2, 5, 60, 300, RPM) for m in enabled)):
            raise ValueError("M1 native Pocket selection changed on reimport")
    elif kind == "preview":
        paths = _preview_paths(trace)
        stored = [project.get_primitive(f"m1-preview-path-{i}")
                  for i in range(len(paths))]
        if (len(project.list_primitives()) != len(paths) + 1 or
                any(type(path) is not Pline or path.closed or
                    tuple(path.get_absolute_coordinates_xyz()) !=
                    tuple((x, y, z, 0.0) for x, y, z in points)
                    for path, points in zip(stored, paths)) or
                len(enabled) != 1 or type(enabled[0]) is not EngraveMop or
                set(project.get_mop_targets(enabled[0])) !=
                {path.internal_id for path in stored}):
            raise ValueError("M1 cleanup preview changed on strict reimport")
    else:
        raise ValueError("unknown M1 candidate kind")
    return project


def _preview_paths(trace):
    paths, pending = [], []
    for item in trace.items:
        if (type(item) is replay.Motion and item.tool == "T2" and
                item.role == "cut" and item.end[2] == -8):
            if not pending:
                pending = [item.start]
            pending.append(item.end)
        elif pending:
            paths.append(tuple(pending))
            pending = []
    if pending:
        raise ValueError("M1 preview has unfinished path")
    return tuple(paths)


def _make_preview(project, trace):
    layer = project.add_layer("Generated T2 preview only")
    part = project.list_parts()[0]
    paths = []
    for index, points in enumerate(_preview_paths(trace)):
        path = project.add_pline(layer, points,
                                 identifier=f"m1-preview-path-{index}")
        if path is None:
            raise RuntimeError("could not add M1 preview path")
        paths.append(path)
    mop = project.add_engrave_mop(
        part, targets=paths, name="PREVIEW T2 generated cleanup only",
        identifier="m1-preview", enabled=True, tool_number=2,
        tool_diameter=2, tool_profile="EndMill", stock_surface=0,
        target_depth=0, depth_increment=8, roughing_clearance=0,
        clearance_plane=5, plunge_feedrate=60, cut_feedrate=300,
        spindle_direction="CW", spindle_speed=RPM, work_plane="XY",
        velocity_mode="ExactStop", optimisation_mode="None")
    if mop is None or not paths:
        raise RuntimeError("could not add M1 cleanup preview")


def _make_explicit(project, trace):
    layer = project.add_layer("M1 literal motion anchor")
    anchor = project.add_points(layer, [(START[0], START[1], 0)],
                                identifier="m1-script-anchor")
    mop = project.add_drill_mop(
        project.list_parts()[0], targets=[anchor],
        name="M1 T1 rough plus T2 cleanup literal motion",
        identifier="m1-literal", enabled=True,
        drilling_method="CustomScript", custom_script=_script(trace),
        tool_number=1, tool_diameter=5, tool_profile="EndMill",
        stock_surface=0, target_depth=-8, depth_increment=2,
        clearance_plane=5, plunge_feedrate=60, cut_feedrate=300,
        spindle_direction="CW", spindle_speed=RPM, work_plane="XY",
        velocity_mode="ExactStop")
    if anchor is None or mop is None:
        raise RuntimeError("could not add M1 literal carrier")


def _make_native(project):
    """One bounded Pocket/Default role hypothesis, separate from the literal path."""
    part = project.list_parts()[0]
    region = project.get_primitive("letter-finish")
    fields = dict(target_depth=-8, depth_increment=2, stock_surface=0,
                  clearance_plane=5, spindle_direction="CW", spindle_speed=RPM,
                  tool_profile="EndMill", plunge_feedrate=60, cut_feedrate=300,
                  stepover=0.4, work_plane="XY", lead_in_type="None",
                  optimisation_mode="None", stepover_feedrate="Cut Feedrate",
                  max_crossover_distance=0, velocity_mode="ExactStop")
    # Footer travels above stock and stops at the shared setup point before
    # CamBam's next tool-change block. Its actual output must still be audited.
    for number, diameter, clearance in ((1, 5, 0.5), (2, 2, 0)):
        mop = project.add_pocket_mop(
            part, targets=[region], name=f"NATIVE T{number} letter Pocket",
            identifier=f"m1-native-t{number}", enabled=True,
            tool_number=number, tool_diameter=diameter,
            roughing_clearance=clearance,
            custom_mop_footer=NATIVE_FOOTER, **fields)
        if mop is None:
            raise RuntimeError("could not add M1 native Pocket")


def build_workflow(directory, *, source_path=None, prior_path=None,
                   native_candidate_path=None):
    """Prepare a pinned source, supplied prior, preview and two output routes."""
    if source_path is not None and prior_path is None:
        raise ValueError("existing M1 source requires source-bound supplied prior")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("M1 output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source.cb"
    if source_path is None:
        synthetic_source().save(str(source))
    else:
        source.write_bytes(Path(source_path).read_bytes())
    source_bytes = source.read_bytes()
    supplied = (synthetic_prior(source_bytes) if prior_path is None else
                json.loads(Path(prior_path).read_text(encoding="utf-8")))
    prior_file = directory / "prior.json"
    prior_file.write_text(json.dumps(supplied, indent=2) + "\n", encoding="utf-8")
    result = plan(source_bytes, supplied)
    trace = result.trace
    candidates = {}
    for kind in ("preview", "explicit", "native"):
        project = read_cambam_bytes(source_bytes, source_name=f"M1 {kind} source")
        if kind == "preview":
            _make_preview(project, trace)
        elif kind == "explicit":
            _make_explicit(project, trace)
        elif native_candidate_path is None:
            _make_native(project)
        candidate = directory / kind / f"m1-{kind}.cb"
        candidate.parent.mkdir()
        if kind == "native" and native_candidate_path is not None:
            candidate.write_bytes(Path(native_candidate_path).read_bytes())
        else:
            project.save(str(candidate))
        _check_candidate(candidate.read_bytes(), source_bytes, kind, trace)
        candidates[kind] = {"file": str(candidate.relative_to(directory)),
                            "sha256": _sha(candidate.read_bytes())}
    manifest = {
        "format": "m1-polygon-rest-v1", "source_sha256": _sha(source_bytes),
        "prior_sha256": _sha(prior_file.read_bytes()),
        "prior_motion_fingerprint": result.prior_trace.motion_fingerprint,
        "motion_fingerprint": trace.motion_fingerprint,
        "candidates": candidates,
        "expected_items": [_record(item) for item in trace.items],
        "script_lines": _script(trace).splitlines(),
        "rough_rest_mm2": {str(d): result.pure_rest_area(d)
                           for d in (1, 3, 5, 7)},
        "final_rest_mm2": {str(d): result.residual_area(d)
                           for d in (1, 3, 5, 7)},
        "finite_tool_ideal_mm2": 1.190659933126879,
        "protected_overcut_upper_mm2": {
            str(d): result.protected_overcut_upper_area(d) for d in (1, 3, 5, 7)},
        "residual_outside_0_05mm_ideal_envelope_mm2": {
            str(d): result.residual_outside_ideal_envelope_area(d)
            for d in (1, 3, 5, 7)},
        "setup": SETUP, "completion": result.completion,
        "postprocessor": "Default", "profile": "Default mm",
        "explicit_status": "pending_actual_CamBam_post",
        "native_status": "pending_actual_CamBam_post_and_whole_motion_audit",
    }
    (directory / "expected-motion.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in manifest.items()
            if key not in ("expected_items", "script_lines")}


def _reload(manifest_path):
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("format") != "m1-polygon-rest-v1":
        raise ValueError("not an M1 polygonal-rest manifest")
    source = directory / "source.cb"
    prior = directory / "prior.json"
    if (_sha(source.read_bytes()) != expected["source_sha256"] or
            _sha(prior.read_bytes()) != expected["prior_sha256"]):
        raise ValueError("M1 source or supplied prior changed")
    source_bytes = source.read_bytes()
    result = plan(source_bytes, json.loads(prior.read_text(encoding="utf-8")))
    if (result.prior_trace.motion_fingerprint !=
            expected["prior_motion_fingerprint"] or
            result.trace.motion_fingerprint != expected["motion_fingerprint"] or
            [_record(item) for item in result.trace.items] !=
            expected["expected_items"] or
            _script(result.trace).splitlines() != expected["script_lines"]):
        raise ValueError("M1 expected motion manifest is stale")
    if expected["setup"] != SETUP or expected["completion"] != result.completion:
        raise ValueError("M1 output setup or completion changed")
    for kind, entry in expected["candidates"].items():
        candidate = directory / entry["file"]
        if _sha(candidate.read_bytes()) != entry["sha256"]:
            raise ValueError(f"M1 {kind} candidate changed")
        _check_candidate(candidate.read_bytes(), source_bytes, kind, result.trace)
    for d in (1, 3, 5, 7):
        key = str(d)
        if (list(result.pure_rest_area(d)) != expected["rough_rest_mm2"][key] or
                list(result.residual_area(d)) != expected["final_rest_mm2"][key] or
                result.protected_overcut_upper_area(d) !=
                expected["protected_overcut_upper_mm2"][key] or
                result.residual_outside_ideal_envelope_area(d) !=
                expected["residual_outside_0_05mm_ideal_envelope_mm2"][key]):
            raise ValueError("M1 rest manifest is stale")
    return directory, expected, result


def _reload_native(manifest_path):
    """Native Pocket evidence is independent of the generated T2 path revision."""
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("format") != "m1-polygon-rest-v1" or expected.get("setup") != SETUP:
        raise ValueError("not a current M1 native setup manifest")
    source = directory / "source.cb"
    prior = directory / "prior.json"
    if (_sha(source.read_bytes()) != expected["source_sha256"] or
            _sha(prior.read_bytes()) != expected["prior_sha256"]):
        raise ValueError("M1 native source or prior file changed")
    entry = expected["candidates"]["native"]
    candidate = directory / entry["file"]
    if _sha(candidate.read_bytes()) != entry["sha256"]:
        raise ValueError("M1 native Pocket candidate changed")
    _check_candidate(candidate.read_bytes(), source.read_bytes(), "native", None)
    return directory, expected


def _read_post(candidate, posted_path, *, allow_arcs=False):
    post = Path(posted_path).read_text(encoding="utf-8-sig")
    lines = post.splitlines()
    if ("( Post processor: Default )" not in lines[:12] or
            not any(line.startswith(f"( {candidate.stem} ") for line in lines[:12])):
        raise ValueError("CamBam post header does not match M1 candidate/Default")
    normalized = "\n".join("" if line.strip() in ("G98", "G80") else line
                           for line in lines)
    actual, warnings = read_default_post(normalized, allow_arcs=allow_arcs,
                                         initial_position=START)
    warnings = [warning for warning in warnings
                if "initial machine position is not encoded" not in warning]
    return actual, warnings


def audit_explicit_post(manifest_path, posted_path):
    """Exact Default-post item match followed by shared stock replay."""
    directory, expected, result = _reload(manifest_path)
    candidate = directory / expected["candidates"]["explicit"]["file"]
    actual, warnings = _read_post(candidate, posted_path)
    if warnings:
        return {"status": "unverified", "reason": warnings[0]}
    if (len(actual) == len(result.trace.items) + 1 and
            actual[-1]["type"] == "event" and
            actual[-1]["kind"] == "spindle_stop" and
            actual[-1]["tool"] == "T2" and
            tuple(actual[-1]["position"]) == START):
        actual = actual[:-1]
    if len(actual) != len(result.trace.items):
        return {"status": "deviation", "reason": "extra or missing emitted item",
                "expected_items": len(result.trace.items), "actual_items": len(actual)}
    observed = []
    for index, (want, got) in enumerate(zip(result.trace.items, actual)):
        if type(want) is replay.Motion:
            if (got["type"] != "move" or got["tool"] != want.tool or
                    got["g"] != (0 if want.role == "rapid" else 1) or
                    got["feed"] != want.feed or
                    tuple(got["start"]) != want.start or
                    tuple(got["end"]) != want.end):
                return {"status": "deviation", "item": index,
                        "line": got["line"], "expected": _record(want),
                        "actual": got}
            observed.append(replay.Motion(want.role, want.tool, want.operation,
                                          tuple(got["start"]), tuple(got["end"]),
                                          got["feed"]))
        else:
            if (got["type"] != "event" or got["kind"] != want.kind or
                    got["tool"] != want.tool or
                    tuple(got["position"]) != want.position or
                    (want.kind == "spindle_start" and got["rpm"] != RPM)):
                return {"status": "deviation", "item": index,
                        "line": got["line"], "expected": _record(want),
                        "actual": got}
            observed.append(replay.Event(want.kind, want.tool,
                                         tuple(got["position"])))
    trace = replay.Trace(result.trace.source_fingerprint, result.trace.frame,
                         START, result.trace.operations, tuple(observed))
    stock = replay.replay(trace, expected_source=trace.source_fingerprint)
    if stock.prefixes != result.stock.prefixes:
        return {"status": "deviation", "reason": "posted stock prefix changed"}
    posted = polygon_rest.RestResult(result.prior_trace, trace,
                                     result.prior_stock, stock, 0.5)
    areas = {str(d): posted.residual_area(d) for d in (1, 3, 5, 7)}
    if any(abs(areas[key][i] - expected["final_rest_mm2"][key][i]) > 1e-8
           for key in areas for i in (0, 1)):
        return {"status": "deviation", "reason": "posted residual changed",
                "posted_final_rest_mm2": areas}
    return {"status": "bounded_m1_explicit_post_pass",
            "post_sha256": _sha(Path(posted_path).read_bytes()),
            "candidate_sha256": expected["candidates"]["explicit"]["sha256"],
            "item_count": len(actual), "stock_prefixes": stock.prefixes,
            "rough_rest_mm2": expected["rough_rest_mm2"],
            "final_rest_mm2": areas,
            "numeric_limit": "GEOS floating topology lacks a formal interval proof"}


def audit_native_post(manifest_path, posted_path):
    """Audit the independent Pocket route; never borrow the literal certificate."""
    directory, expected = _reload_native(manifest_path)
    candidate = directory / expected["candidates"]["native"]["file"]
    post = Path(posted_path).read_text(encoding="utf-8-sig")
    if any(post.count(f"( NATIVE T{n} letter Pocket )") != 1 for n in (1, 2)):
        raise ValueError("M1 native post lacks both selected Pocket sections")
    actual, warnings = _read_post(candidate, posted_path, allow_arcs=True)
    findings = list(warnings)
    if not any(row["type"] == "move" and row["tool"] == "T1" for row in actual):
        findings.append("missing T1 roughing motion")
    if not any(row["type"] == "move" and row["tool"] == "T2" for row in actual):
        findings.append("missing T2 cleanup motion")
    if [row["tool"] for row in actual if row["type"] == "event" and
        row["kind"] == "tool_change"] != ["T1", "T2"]:
        findings.append("native tool-change sequence is not T1 then T2")
    target = _target()
    region = ShapelyPolygon(SHELL, (HOLE,))
    paths = []
    active = None
    running = False
    cleared_low_rapids = 0
    boundary_shortfall = 0.0
    for row in actual:
        if row["type"] == "event":
            if row["kind"] == "spindle_stop":
                running = False
            elif row["kind"] == "tool_change":
                if running or tuple(row["position"]) != START:
                    findings.append(f"line {row['line']}: displaced or running tool change")
                active = row["tool"]
            elif row["kind"] == "spindle_start":
                if row["rpm"] != RPM:
                    findings.append(f"line {row['line']}: spindle speed changed")
                running = True
            continue
        a, b = tuple(row["start"]), tuple(row["end"])
        low = min(a[2], b[2])
        if row["tool"] not in ("T1", "T2"):
            findings.append(f"line {row['line']}: unexpected tool")
            continue
        if row["tool"] != active or not running:
            findings.append(f"line {row['line']}: inactive or wrong tool motion")
        if row["g"] == 0 and low < 0:
            if a[:2] != b[:2]:
                findings.append(f"line {row['line']}: low rapid changes XY")
            else:
                point = polygon_rest._line(a[:2], a[:2])
                radius = 2.5 if row["tool"] == "T1" else 1.0
                if any(bottom <= low and
                       point.distance(prior_path) + prior_error <=
                       (2.5 if name == "T1" else 1.0) - radius + 0.001
                       for name, bottom, prior_path, prior_error in paths):
                    cleared_low_rapids += 1
                else:
                    findings.append(f"line {row['line']}: low rapid lacks cleared column")
        if row["g"] not in (1, 2, 3) or low >= 0:
            continue
        if a[2] != b[2] and a[:2] != b[:2]:
            findings.append(f"line {row['line']}: ramp lacks all-height sweep proof")
            continue
        if a[:2] != b[:2] and row["feed"] != 300:
            findings.append(f"line {row['line']}: unsupported low-level XY feed")
        if low < -target.depth:
            findings.append(f"line {row['line']}: cut exceeds target floor")
        radius = 2.5 if row["tool"] == "T1" else 1.0
        coordinates, deviation = _path_for_move(row)
        line = (polygon_rest._line(*coordinates) if len(coordinates) == 2 else
                LineString(coordinates))
        clearance = line.distance(region.boundary)
        shortfall = radius + deviation - clearance
        boundary_shortfall = max(boundary_shortfall, shortfall)
        if not region.covers(line) or shortfall > 0.001:
            findings.append(f"line {row['line']}: original Region overcut")
        if row["tool"] == "T1" and 3 + deviation - clearance > 0.001:
            findings.append(f"line {row['line']}: T1 allowance violated")
        if (row["tool"] == "T2" and a[:2] == b[:2] and
                not any(name == "T1" and bottom <= low and
                        polygon_rest._line(a[:2], a[:2]).distance(prior_path) +
                        prior_error <= 1.5 + 0.001
                        for name, bottom, prior_path, prior_error in paths)):
            findings.append(f"line {row['line']}: T2 descent lacks T1-cleared column")
        paths.append((row["tool"], low, line, deviation))
    areas = None
    if not any("ramp" in item for item in findings):
        from shapely.ops import unary_union
        def bounds(depth, selected):
            inner, outer = [], []
            inflation = 1 / math.cos(math.pi / (4 * polygon_rest.QUAD_SEGS))
            unique = {}
            for name, bottom, line, deviation in paths:
                if bottom <= -depth and name in selected:
                    unique[(name, line.wkb)] = (name, line, deviation)
            for name, line, deviation in unique.values():
                radius = 2.5 if name == "T1" else 1.0
                inner.append(line.buffer(radius - deviation - 1e-6,
                                         quad_segs=polygon_rest.QUAD_SEGS))
                outer.append(line.buffer((radius + deviation + 1e-6) * inflation,
                                         quad_segs=polygon_rest.QUAD_SEGS))
            inner_union = unary_union(inner) if inner else GeometryCollection()
            outer_union = unary_union(outer) if outer else GeometryCollection()
            return (max(0.0, region.area - region.intersection(outer_union).area),
                    max(0.0, region.area - region.intersection(inner_union).area),
                    outer_union.difference(region).area)
        areas = {str(d): {"rough": bounds(d, {"T1"}),
                          "final": bounds(d, {"T1", "T2"})}
                 for d in (1, 3, 5, 7)}
        if any(row["rough"][1] > 139 or
               row["rough"][0] < 127.265122238 - 0.01 or
               row["final"][1] > 1.190659933126879 + 0.5 or
               row["rough"][0] - row["final"][1] <= 100 or
               row["final"][2] > 0.01
               for row in areas.values()):
            findings.append("native rough/final rest or protected-overcut budget failed")
    return {"status": "bounded_m1_native_post_pass" if not findings else
            "native_motion_gate_failed",
            "candidate_sha256": expected["candidates"]["native"]["sha256"],
            "post_sha256": _sha(Path(posted_path).read_bytes()),
            "parsed_items": len(actual), "findings": findings,
            "cleared_low_rapid_count": cleared_low_rapids,
            "maximum_boundary_shortfall_mm": boundary_shortfall,
            "section_area_mm2": areas,
            "independent_route": "native_Pocket_Default"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="new output directory or expected-motion.json")
    parser.add_argument("posted", nargs="?", help="actual CamBam Default-post NC")
    parser.add_argument("--route", choices=("explicit", "native"),
                        default="explicit")
    parser.add_argument("--source", help="existing exact M1 source.cb")
    parser.add_argument("--prior", help="matching supplied prior.json")
    parser.add_argument("--native-candidate",
                        help="reuse an unchanged native Pocket candidate")
    args = parser.parse_args()
    answer = ((audit_explicit_post(args.path, args.posted) if
               args.route == "explicit" else
               audit_native_post(args.path, args.posted)) if args.posted else
              build_workflow(args.path, source_path=args.source,
                             prior_path=args.prior,
                             native_candidate_path=args.native_candidate))
    print(json.dumps(answer, indent=2))
    if args.posted and not answer["status"].endswith("_pass"):
        raise SystemExit(1)
