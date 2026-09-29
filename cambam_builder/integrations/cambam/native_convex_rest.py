"""Native source, visual preview, and audited literal output for one triangle."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from ...cam_core import convex_rest, replay
from ...native.cad import Pline, Points
from ...native.cam import DrillMop, EngraveMop
from ...native.project import CamBamProject as CBProject
from ...native.reader import read_cambam_bytes
from ...native.region import Region
from .cone_script import _compare_post, _expected_items, _script


POLYGON = ((0.0, 0.0), (12.0, 0.0), (6.0, 8.0))
START = (-10.0, -10.0, 5.0)
HIGH = (3.0, 2.0, 1.0)
END = (4.0, 2.0)
FEEDS = {"entry": 60, "retract": 300, "cleared_descent": 60, "cut": 300}
RPM = 12000


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def synthetic_source():
    project = CBProject("Convex triangle rest source")
    layer = project.add_layer("Original finish Region")
    region = project.add_region(
        layer, Pline(vertices=POLYGON, closed=True), identifier="triangle-finish")
    part = project.add_part(
        "Triangle part", stock_thickness=3, stock_width=14,
        stock_height=10, stock_offset=(-1, -1), stock_surface=0,
        machining_origin=(0, 0), nesting_method="None")
    if None in (layer, region, part):
        raise RuntimeError("could not create triangle source")
    return project


def synthetic_prior(source_bytes):
    """Explicit supplied-motion fixture, bound to these exact native bytes."""
    return {
        "format": "convex-prior-v1", "source_sha256": _sha(source_bytes),
        "units": "mm", "frame": "triangle-drawing", "target_depth_mm": 3,
        "tool": {"name": "T3", "kind": "pointed_cone",
                 "radius_mm": 3, "cutting_length_mm": 3},
        "initial_xyz_mm": list(HIGH), "cleanup_end_xy_mm": list(END),
        "items": [
            {"kind": "tool_change", "position": list(HIGH)},
            {"kind": "spindle_start", "position": list(HIGH)},
            {"role": "entry", "start": list(HIGH), "end": [3, 2, -1.2]},
            {"role": "retract", "start": [3, 2, -1.2], "end": list(HIGH)},
            {"kind": "spindle_stop", "position": list(HIGH)},
        ],
    }


def _source(data):
    project = read_cambam_bytes(data, source_name="triangle native source")
    parts, shapes = project.list_parts(), project.list_primitives()
    if (len(parts) != 1 or len(shapes) != 1 or type(shapes[0]) is not Region or
            project.list_mops()):
        raise ValueError("triangle source requires one Region, one Part and no MOPs")
    part, region = parts[0], shapes[0]
    xyz = region.get_absolute_coordinates_xyz()
    outer = xyz["outer_curve"]
    if (region.user_identifier != "triangle-finish" or xyz["hole_curves"] or
            len(outer) != 3 or any(z != 0 or bulge != 0
                                   for _, _, z, bulge in outer) or
            tuple((x, y) for x, y, _, _ in outer) != POLYGON):
        raise ValueError("unsupported triangle Region geometry or placement")
    if (not part.enabled or part.nesting_method != "None" or
            tuple(part.machining_origin) != (0, 0) or
            tuple(part.stock_drawing_origin) != (-1, -1, 0) or
            (part.stock_width, part.stock_height, part.stock_thickness,
             part.stock_surface) != (14, 10, 3, 0)):
        raise ValueError("unsupported triangle Part stock or placement")
    target = replay.Target("triangle", (0, 0, 12, 8), 3, polygon=POLYGON)
    return project, region, part, target


def _plan(source_bytes, prior):
    project, region, part, target = _source(source_bytes)
    if (type(prior) is not dict or set(prior) != {
            "format", "source_sha256", "units", "frame", "target_depth_mm",
            "tool", "initial_xyz_mm", "cleanup_end_xy_mm", "items"} or
            prior["format"] != "convex-prior-v1" or
            prior["source_sha256"] != _sha(source_bytes) or
            prior["units"] != "mm" or prior["frame"] != "triangle-drawing" or
            prior["target_depth_mm"] != 3 or
            prior["tool"] != {"name": "T3", "kind": "pointed_cone",
                              "radius_mm": 3, "cutting_length_mm": 3} or
            prior["initial_xyz_mm"] != list(HIGH) or
            prior["cleanup_end_xy_mm"] != list(END) or
            type(prior["items"]) is not list or len(prior["items"]) != 5):
        raise ValueError("stale or unsupported triangle prior trace/setup")
    tool = replay.ToolProfile("T3", "pointed_cone", 3, 3)
    operation = replay.Operation("prior", tool, target)
    items = []
    for raw in prior["items"]:
        if type(raw) is not dict:
            raise ValueError("invalid supplied prior item")
        if "kind" in raw and set(raw) == {"kind", "position"}:
            items.append(replay.Event(raw["kind"], "T3", tuple(raw["position"])))
        elif set(raw) == {"role", "start", "end"}:
            items.append(replay.Motion(raw["role"], "T3", "prior",
                                       tuple(raw["start"]), tuple(raw["end"])))
        else:
            raise ValueError("invalid supplied prior item")
    trace = replay.Trace(_sha(source_bytes), "triangle-drawing", HIGH,
                         (operation,), tuple(items))
    result = convex_rest.generate(trace, END, expected_source=_sha(source_bytes),
                                  expected_motion=trace.motion_fingerprint)
    return project, region, part, trace, result


def _output_trace(result):
    """Add the external setup and concrete process to the verified core trace."""
    core = result.trace
    items = (replay.Event("tool_change", "T3", START),
             replay.Event("spindle_start", "T3", START),
             replay.Motion("rapid", "T3", "prior", START, HIGH))
    items += tuple(replace(item, feed=FEEDS[item.role])
                   for item in core.items[2:-1] if type(item) is replay.Motion)
    last = items[-1].end
    items += (replay.Motion("rapid", "T3", "cleanup", last, START),
              replay.Event("spindle_stop", "T3", START))
    trace = replay.Trace(core.source_fingerprint, core.frame, START,
                         core.operations, items)
    replay.replay(trace, expected_source=core.source_fingerprint)
    return trace


def _check_candidate(data, source_bytes, kind, trace):
    project = read_cambam_bytes(data, source_name=f"triangle {kind} candidate")
    source = read_cambam_bytes(source_bytes, source_name="triangle source comparison")
    original = source.get_primitive("triangle-finish")
    region = project.get_primitive("triangle-finish")
    if (type(region) is not Region or region.internal_id != original.internal_id or
            region.get_absolute_coordinates_xyz() != original.get_absolute_coordinates_xyz()):
        raise ValueError("candidate changed original triangle Region")
    if len(project.list_parts()) != 1:
        raise ValueError("candidate changed source Part count")
    part, original_part = project.list_parts()[0], source.list_parts()[0]
    if (part.user_identifier != original_part.user_identifier or
            part.enabled != original_part.enabled or
            part.nesting_method != original_part.nesting_method or
            tuple(part.machining_origin) != tuple(original_part.machining_origin) or
            tuple(part.stock_drawing_origin) !=
            tuple(original_part.stock_drawing_origin) or
            (part.stock_width, part.stock_height, part.stock_thickness,
             part.stock_surface) !=
            (original_part.stock_width, original_part.stock_height,
             original_part.stock_thickness, original_part.stock_surface)):
        raise ValueError("candidate changed source Part stock")
    mops = project.list_mops()
    if len(mops) != 1 or not mops[0].enabled:
        raise ValueError("candidate operation/geometry count changed")
    # The preview adds only a generated cut Pline; the execution file adds one anchor.
    if kind == "preview":
        path = project.get_primitive("generated-triangle-cleanup")
        if (len(project.list_primitives()) != 2 or type(path) is not Pline or
                path.closed or tuple(path.get_absolute_coordinates_xyz()) !=
                ((3.0, 2.0, -1.2, 0.0), (4.0, 2.0, -2.0, 0.0)) or
                type(mops[0]) is not EngraveMop or
                project.get_mop_targets(mops[0]) != [path.internal_id]):
            raise ValueError("preview is not isolated to generated cleanup path")
        mop = mops[0]
        if (mop.user_identifier != "triangle-preview" or
                (mop.tool_number, mop.tool_diameter, mop.tool_profile,
                 mop.stock_surface, mop.target_depth, mop.depth_increment,
                 mop.roughing_clearance, mop.clearance_plane,
                 mop.plunge_feedrate, mop.cut_feedrate,
                 mop.spindle_direction, mop.spindle_speed, mop.work_plane,
                 mop.velocity_mode, mop.optimisation_mode) !=
                (3, 6, "VCutter", 0, 0, 3, 0, 5, 60, 300,
                 "CW", RPM, "XY", "ExactStop", "None")):
            raise ValueError("preview process fields changed")
    else:
        anchor = project.get_primitive("triangle-script-anchor")
        if (len(project.list_primitives()) != 2 or type(anchor) is not Points or
                tuple(anchor.get_absolute_coordinates_xyz()) !=
                ((-10.0, -10.0, 0.0),) or
                type(mops[0]) is not DrillMop or
                mops[0].drilling_method != "CustomScript" or
                mops[0].custom_script != _script(trace) or
                project.get_mop_targets(mops[0]) != [anchor.internal_id]):
            raise ValueError("literal candidate script/target changed")
        mop = mops[0]
        if (mop.user_identifier != "triangle-literal" or
                (mop.tool_number, mop.tool_diameter, mop.tool_profile,
                 mop.stock_surface, mop.target_depth, mop.depth_increment,
                 mop.clearance_plane, mop.plunge_feedrate, mop.cut_feedrate,
                 mop.spindle_direction, mop.spindle_speed, mop.work_plane,
                 mop.velocity_mode) !=
                (3, 6, "VCutter", 0, -2, 2, 5, 60, 300,
                 "CW", RPM, "XY", "ExactStop")):
            raise ValueError("literal candidate process fields changed")
    _source(source_bytes)
    return project


def build_workflow(directory, *, source_path=None, prior_path=None):
    """Prepare fresh source, preview and literal execution candidate files."""
    if source_path is not None and prior_path is None:
        raise ValueError("existing native source requires its supplied prior trace")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("triangle output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source.cb"
    if source_path is None:
        synthetic_source().save(str(source))
    else:
        source.write_bytes(Path(source_path).read_bytes())
    source_bytes = source.read_bytes()
    prior = (synthetic_prior(source_bytes) if prior_path is None else
             json.loads(Path(prior_path).read_text(encoding="utf-8")))
    (directory / "prior.json").write_text(json.dumps(prior, indent=2) + "\n",
                                           encoding="utf-8")
    _, _, _, prior_trace, result = _plan(source_bytes, prior)
    trace = _output_trace(result)

    preview = read_cambam_bytes(source_bytes, source_name="triangle preview source")
    layer = preview.add_layer("Generated cleanup preview")
    path = preview.add_pline(layer, [(3, 2, -1.2), (4, 2, -2)],
                             identifier="generated-triangle-cleanup")
    preview_mop = preview.add_engrave_mop(
        preview.list_parts()[0], targets=[path],
        name="PREVIEW triangle cleanup only", identifier="triangle-preview",
        enabled=True, tool_number=3, tool_diameter=6, tool_profile="VCutter",
        stock_surface=0, target_depth=0, depth_increment=3,
        roughing_clearance=0, clearance_plane=5, plunge_feedrate=60,
        cut_feedrate=300, spindle_direction="CW", spindle_speed=RPM,
        work_plane="XY", velocity_mode="ExactStop", optimisation_mode="None")
    if path is None or preview_mop is None:
        raise RuntimeError("could not attach triangle preview")
    preview_path = directory / "preview" / "triangle-preview.cb"
    preview_path.parent.mkdir()
    preview.save(str(preview_path))

    explicit = read_cambam_bytes(source_bytes, source_name="triangle literal source")
    anchor_layer = explicit.add_layer("Literal motion anchor")
    anchor = explicit.add_points(anchor_layer, [(-10, -10, 0)],
                                 identifier="triangle-script-anchor")
    mop = explicit.add_drill_mop(
        explicit.list_parts()[0], targets=[anchor],
        name="Triangle prior plus cleanup literal motion",
        identifier="triangle-literal", enabled=True,
        drilling_method="CustomScript", custom_script=_script(trace),
        tool_number=3, tool_diameter=6, tool_profile="VCutter",
        stock_surface=0, target_depth=-2, depth_increment=2,
        clearance_plane=5, plunge_feedrate=60, cut_feedrate=300,
        spindle_direction="CW", spindle_speed=RPM,
        work_plane="XY", velocity_mode="ExactStop")
    if anchor is None or mop is None:
        raise RuntimeError("could not attach literal triangle motion")
    explicit_path = directory / "explicit" / "triangle-explicit.cb"
    explicit_path.parent.mkdir()
    explicit.save(str(explicit_path))
    for kind, path in (("preview", preview_path), ("explicit", explicit_path)):
        _check_candidate(path.read_bytes(), source_bytes, kind, trace)
    manifest = {
        "format": "native-convex-rest-v1", "source_sha256": _sha(source_bytes),
        "prior_sha256": _sha((directory / "prior.json").read_bytes()),
        "prior_motion_fingerprint": prior_trace.motion_fingerprint,
        "preview_sha256": _sha(preview_path.read_bytes()),
        "candidate_sha256": _sha(explicit_path.read_bytes()),
        "motion_fingerprint": trace.motion_fingerprint,
        "expected_items": _expected_items(trace),
        "script_lines": _script(trace).splitlines(),
        "pure_rest_mm2": {str(d): result.pure_rest_area(d) for d in (0, 1, 2)},
        "final_rest_mm2": {str(d): result.residual_area(d) for d in (0, 1, 2)},
        "completion": result.completion,
        "postprocessor": "Default", "profile": "Default mm",
        "execution_status": "pending_actual_CamBam_post",
    }
    (directory / "expected-motion.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in manifest.items() if k not in ("expected_items", "script_lines")}


def audit_post(manifest_path, posted_path):
    """Compare a real Default post item by item, then replay its stock removal."""
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("format") != "native-convex-rest-v1":
        raise ValueError("not a native triangle motion manifest")
    paths = {"source_sha256": directory / "source.cb",
             "prior_sha256": directory / "prior.json",
             "preview_sha256": directory / "preview" / "triangle-preview.cb",
             "candidate_sha256": directory / "explicit" / "triangle-explicit.cb"}
    if any(_sha(path.read_bytes()) != expected[key] for key, path in paths.items()):
        raise ValueError("triangle source, prior, preview or candidate changed")
    source_bytes = paths["source_sha256"].read_bytes()
    prior = json.loads(paths["prior_sha256"].read_text(encoding="utf-8"))
    _, _, _, prior_trace, result = _plan(source_bytes, prior)
    trace = _output_trace(result)
    if (expected["prior_motion_fingerprint"] != prior_trace.motion_fingerprint or
            expected["motion_fingerprint"] != trace.motion_fingerprint or
            expected["expected_items"] != _expected_items(trace) or
            expected["script_lines"] != _script(trace).splitlines() or
            expected["pure_rest_mm2"] !=
            {str(d): result.pure_rest_area(d) for d in (0, 1, 2)} or
            expected["final_rest_mm2"] !=
            {str(d): result.residual_area(d) for d in (0, 1, 2)}):
        raise ValueError("triangle motion manifest is stale")
    _check_candidate(paths["preview_sha256"].read_bytes(), source_bytes,
                     "preview", trace)
    _check_candidate(paths["candidate_sha256"].read_bytes(), source_bytes,
                     "explicit", trace)
    posted_trace, failure = _compare_post(
        trace, paths["candidate_sha256"], posted_path)
    if failure:
        return failure
    stock = replay.replay(posted_trace, expected_source=trace.source_fingerprint)
    if stock.prefixes != (("prior", 1), ("cleanup", 2)):
        return {"status": "deviation", "reason": "posted stock order changed"}
    posted_result = replace(result, trace=posted_trace, stock=stock)
    posted_rest = {str(d): posted_result.residual_area(d) for d in (0, 1, 2)}
    if any(abs(posted_rest[key] - value) > 1e-9
           for key, value in expected["final_rest_mm2"].items()):
        return {"status": "deviation", "reason": "posted section rest changed",
                "posted_final_rest_mm2": posted_rest}
    return {"status": "bounded_triangle_post_pass",
            "candidate_sha256": expected["candidate_sha256"],
            "post_sha256": _sha(Path(posted_path).read_bytes()),
            "item_count": len(posted_trace.items), "stock_prefixes": stock.prefixes,
            "pure_rest_mm2": expected["pure_rest_mm2"],
            "final_rest_mm2": posted_rest,
            "completion": result.completion}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="new output directory or expected-motion.json")
    parser.add_argument("posted", nargs="?", help="actual CamBam Default-post NC")
    parser.add_argument("--source", help="existing native triangle source.cb")
    parser.add_argument("--prior", help="matching supplied prior.json")
    args = parser.parse_args()
    answer = (audit_post(args.path, args.posted) if args.posted else
              build_workflow(args.path, source_path=args.source,
                             prior_path=args.prior))
    print(json.dumps(answer, indent=2))
    if args.posted and answer["status"] != "bounded_triangle_post_pass":
        raise SystemExit(1)
