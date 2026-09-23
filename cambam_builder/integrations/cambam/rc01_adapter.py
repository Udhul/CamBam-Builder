"""RC01 native input boundary and reproducible comparison candidates.

The .cb files carry authored intent and candidate CamBam operations.  The motion
manifest is the framework authority until posted motion has been read and replayed.
"""

from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from ... import CBProject
from ...cad_entities import Pline
from ...cam_entities import MOP_XML_FIELD_PATHS, PocketMop
from ...cam_core.rc01 import Event, Job, Move, Tool, generate, verify
from ...cambam_reader import read_cambam_bytes
from ...region import Region
from ...stock import SectionRectangle


SOURCE_FIELDS = {
    "target_depth": -3, "depth_increment": 1, "stock_surface": 0,
    "roughing_clearance": 0, "clearance_plane": 5,
    "spindle_direction": "CW", "spindle_speed": 12000,
    "tool_profile": "EndMill", "plunge_feedrate": 60,
    "cut_feedrate": 300, "stepover": 0.4, "work_plane": "XY",
}
TOOL_COMPONENTS = (
    {"name": "T1", "radius": 3, "cutting_length": 6, "shank_radius": 3,
     "shank_top": 20, "holder_radius": 10, "holder_start": 20,
     "holder_top": 40, "rpm": 12000},
    {"name": "T2", "radius": 1, "cutting_length": 6, "shank_radius": 3,
     "shank_top": 20, "holder_radius": 10, "holder_start": 20,
     "holder_top": 40, "rpm": 12000},
)


def synthetic_setup():
    """Explicit values with no native field or safe inheritance policy."""
    return {
        "units": "mm", "frame": "rc01-drawing", "tip_datum": "flat_endmill_bottom",
        "fixture_xy": [-5, -5, 45, 35], "fixture_top": -10,
        "fixture_bottom": -15, "position_error": 0, "radius_error": 0,
        "start_toolchange_end": [-10, -10, 5], "approach_plane": 1,
        "cleared_vertical_feed": 120, "retract_feed": 300,
        "center_cutting": True, "full_width_slotting": True,
        "max_new_axial_depth": 1, "rough_stepover_max": 2.4,
        "cleanup_stepover_max": 0.8,
        "travel_x": [-15, 55], "travel_y": [-15, 45],
        "travel_z": [-3, 10], "coolant": "off",
        "tools": [dict(component) for component in TOOL_COMPONENTS],
    }


def _rectangle(x0, y0, x1, y1):
    return Pline(vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                 closed=True)


def synthetic_source():
    """Build the editable native design with disabled source machining intent."""
    project = CBProject("RC01 synthetic input")
    layer = project.add_layer("RC01 target")
    part = project.add_part("RC01 part", stock_thickness=10, stock_width=50,
                            stock_height=40, stock_offset=(-5, -5),
                            machining_origin=(0, 0), stock_surface=0,
                            nesting_method="None")
    target = project.add_region(
        layer, _rectangle(0, 0, 40, 30),
        [_rectangle(16, 11, 24, 19)], identifier="rc01-target")
    if None in (layer, part, target):
        raise RuntimeError("could not construct RC01 native source")
    for number, diameter in ((1, 6), (2, 2)):
        mop = project.add_pocket_mop(
            part, targets=[target], name=f"SOURCE T{number} Pocket intent (disabled)",
            identifier=f"rc01-source-t{number}", enabled=False,
            tool_number=number, tool_diameter=diameter, **SOURCE_FIELDS)
        if mop is None:
            raise RuntimeError("could not construct RC01 source MOP")
    return project


def _box(contour):
    points = contour
    if len(points) != 4 or any(z != 0 or bulge != 0 for _, _, z, bulge in points):
        raise ValueError("RC01 requires four straight zero-Z Region vertices")
    xy = [(Fraction(str(x)), Fraction(str(y))) for x, y, _, _ in points]
    xs, ys = sorted({x for x, _ in xy}), sorted({y for _, y in xy})
    if len(xs) != 2 or len(ys) != 2 or set(xy) != {
            (x, y) for x in xs for y in ys}:
        raise ValueError("RC01 requires axis-aligned rectangular contours")
    return SectionRectangle(xs[0], ys[0], xs[1], ys[1])


def normalize(project, setup, *, allow_attachments=False):
    """Read one exact native RC01 intent; fail closed on unmodeled changes."""
    if not isinstance(setup, dict) or set(setup) != set(synthetic_setup()):
        raise ValueError("RC01 setup fields are missing or unsupported")
    if {k: v for k, v in setup.items() if k != "tools"} != {
            k: v for k, v in synthetic_setup().items() if k != "tools"}:
        raise ValueError("unsupported RC01 setup or unit change")
    if len(setup["tools"]) != 2 or any(set(t) != set(TOOL_COMPONENTS[0])
                                        for t in setup["tools"]):
        raise ValueError("RC01 needs complete explicit tool components")
    tools = tuple(Tool(**component) for component in setup["tools"])
    if len(project.list_parts()) != 1:
        raise ValueError("RC01 requires exactly one Part")
    part = project.list_parts()[0]
    if not part.enabled or part.nesting_method != "None" or tuple(
            part.machining_origin) != (0, 0) or part.stock_surface != 0:
        raise ValueError("unsupported RC01 Part placement or state")
    x, y, _ = part.stock_drawing_origin
    stock = SectionRectangle(x, y, x + part.stock_width, y + part.stock_height)
    regions = [p for p in project.list_primitives() if isinstance(p, Region)
               and len(p.hole_curves) == 1]
    if len(regions) != 1:
        raise ValueError("RC01 requires one target Region with one island")
    target = regions[0]
    if not allow_attachments and (len(project.list_primitives()) != 1 or
                                  len(project.list_mops()) != 2):
        raise ValueError("unsupported RC01 extra geometry or machining operation")
    xyz = target.get_absolute_coordinates_xyz()
    outer, island = _box(xyz["outer_curve"]), _box(xyz["hole_curves"][0])
    mops = [m for m in project.get_mops_in_part(part) if
            type(m) is PocketMop and
            project.get_mop_targets(m) == [target.internal_id]]
    if len(mops) != 2 or [m.tool_number for m in mops] != [1, 2]:
        raise ValueError("RC01 source Pocket order/targets/tool numbers changed")
    baseline_mops = synthetic_source().list_mops()
    for mop, baseline in zip(mops, baseline_mops):
        if mop.enabled:
            raise ValueError("RC01 source Pocket must stay disabled")
        for field in MOP_XML_FIELD_PATHS:
            if not hasattr(mop, field):
                continue
            state = getattr(mop, "_xml_parameter_states", {}).get(field)
            if state == "Default" or (state is None and
                                      getattr(baseline, field) not in (None, "")):
                raise ValueError(f"RC01 {field} is inherited or absent")
            if getattr(mop, field) != getattr(baseline, field):
                raise ValueError(f"unsupported RC01 {field} change")
    job = Job(stock=stock, outer=outer, island=island,
              stock_bottom=part.stock_surface - part.stock_thickness,
              fixture_bottom=setup["fixture_bottom"], tools=tools,
              position_error=setup["position_error"],
              radius_error=setup["radius_error"])
    if job != Job():
        raise ValueError("unsupported RC01 geometry, stock or tool change")
    return job


def normalize_bytes(data, setup):
    return normalize(read_cambam_bytes(data, source_name="RC01 input"), setup)


def _candidate_mop(project, part, layer, moves, tool, depth, label):
    paths = []
    for index, move in enumerate(moves):
        if not isinstance(move, Move) or move.role != "cut" or move.tool != tool or move.end[2] != depth:
            continue
        path = project.add_pline(layer, [tuple(map(float, move.start)),
                                         tuple(map(float, move.end))],
                                 identifier=f"{label}-path-{index}")
        if path is None:
            raise RuntimeError("could not attach RC01 candidate path")
        paths.append(path)
    mop = project.add_engrave_mop(
        part, targets=paths, name=f"CANDIDATE {label} Z{depth}",
        identifier=f"{label}-z{abs(depth)}", enabled=True,
        tool_number=1 if tool == "T1" else 2,
        tool_diameter=6 if tool == "T1" else 2,
        target_depth=depth, depth_increment=1, stock_surface=0,
        roughing_clearance=0, clearance_plane=5, spindle_direction="CW",
        spindle_speed=12000, plunge_feedrate=60, cut_feedrate=300,
        work_plane="XY")
    if mop is None:
        raise RuntimeError("could not attach RC01 candidate Engrave")
    return len(paths)


def _attach_candidate(project, items, tools):
    layer = project.add_layer("RC01 candidate centerlines")
    part = project.list_parts()[0]
    return {tool: [_candidate_mop(project, part, layer, items, tool, depth,
                                  f"generated-{tool.lower()}")
                   for depth in (-1, -2, -3)] for tool in tools}


def _attach_native_cleanup(project):
    layer = project.add_layer("RC01 native cleanup windows")
    part = project.list_parts()[0]
    windows = ((0, 0, 7, 7), (33, 0, 40, 7),
               (33, 23, 40, 30), (0, 23, 7, 30))
    for index, bounds in enumerate(windows, 1):
        shape = project.add_region(layer, _rectangle(*bounds),
                                   identifier=f"cleanup-window-{index}")
        mop = project.add_pocket_mop(
            part, targets=[shape], name=f"NATIVE T2 window {index}",
            identifier=f"native-t2-window-{index}", enabled=True,
            tool_number=2, tool_diameter=2, **SOURCE_FIELDS)
        if shape is None or mop is None:
            raise RuntimeError("could not attach native cleanup window")


def _motion_record(item):
    record = {k: v for k, v in asdict(item).items()}
    for key in ("position", "start", "end"):
        if key in record:
            record[key] = [float(v) for v in record[key]]
    record["type"] = "event" if isinstance(item, Event) else "move"
    return record


def build_artifacts(directory):
    """Create A/B/C candidates, authoritative motion and pending comparison."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("RC01 output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    setup = synthetic_setup()
    source = synthetic_source()
    source_path = directory / "source.cb"
    source.save(str(source_path))
    reloaded = read_cambam_bytes(source_path.read_bytes(), source_name=str(source_path))
    job = normalize(reloaded, setup)
    program = generate(job)
    certificate = verify(program, job)
    items = program.items
    split = next(i for i, item in enumerate(items)
                 if isinstance(item, Event) and item.kind == "tool_change"
                 and item.tool == "T2")
    variants = {}
    for letter, name, included, native in (
            ("A", "rough", ("T1",), False),
            ("B", "explicit", ("T1", "T2"), False),
            ("C", "native-cleanup", ("T1",), True)):
        project = reloaded.clone()
        counts = _attach_candidate(project, items, included)
        if native:
            _attach_native_cleanup(project)
        path = directory / f"{letter}-{name}.cb"
        project.save(str(path))
        reopened = read_cambam_bytes(path.read_bytes(), source_name=str(path))
        if normalize(reopened, setup, allow_attachments=True) != job:
            raise ValueError(f"{letter} source input changed during attachment")
        variants[letter] = {
            "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "candidate_cut_path_counts_by_depth": counts,
            "enabled_mops": [m.name for m in reopened.list_mops() if m.enabled],
            "source_mops_disabled": all(not m.enabled for m in reopened.list_mops()[:2]),
            "expected_motion_prefix": "rough" if letter != "B" else "rough_then_cleanup",
            "rough_rest_by_depth_mm2": certificate.rough_rest_by_depth,
            "final_rest_by_depth_mm2": certificate.final_rest_by_depth if letter == "B" else None,
            "required_native_final_upper_mm2": 4 - math.pi + 0.5 if letter == "C" else None,
            "emitted_motion_status": "pending_CamBam_regeneration_and_post",
        }
    manifest = {
        "format": "rc01-comparison-v1", "job_fingerprint": job.fingerprint,
        "motion_fingerprint": program.motion_fingerprint,
        "source_file": source_path.name,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "rough_item_count": split, "full_item_count": len(items),
        "certificate_status": certificate.status,
        "rough_rest_by_depth_mm2": certificate.rough_rest_by_depth,
        "final_rest_by_depth_mm2": certificate.final_rest_by_depth,
        "rough_rest_volume_mm3": certificate.rough_rest_volume,
        "final_rest_volume_mm3": certificate.final_rest_volume,
        "area_coordinate_enclosure_mm": certificate.area_coordinate_enclosure_mm,
        "location_polygon_sagitta_mm": certificate.location_polygon_sagitta_mm,
        "location_numeric_enclosure_mm": certificate.location_numeric_enclosure_mm,
        "variants": variants, "expected_items": [_motion_record(item) for item in items],
    }
    (directory / "setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    (directory / "comparison.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RC01 native A/B/C candidates")
    parser.add_argument("directory", help="new or empty output directory")
    args = parser.parse_args()
    result = build_artifacts(args.directory)
    print(json.dumps({"directory": args.directory,
                      "job_fingerprint": result["job_fingerprint"],
                      "motion_fingerprint": result["motion_fingerprint"]}, indent=2))
