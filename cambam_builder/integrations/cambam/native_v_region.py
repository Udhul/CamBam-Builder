"""M3 Region V preview and exact-motion CamBam candidates.

Engrave previews the generated XYZ paths.  Drill/CustomScript transports their
literal motion.  Neither candidate is credited until its actual Default post
passes the whole-stream audit below.
"""

from dataclasses import replace
from collections import Counter
import hashlib
import json
from pathlib import Path

from ...cam_core import replay, v_region
from shapely.geometry import LineString
from ...native.cad import Pline, Points
from ...native.cam import DrillMop, EngraveMop
from ...native.reader import read_cambam_bytes
from ...native.region import Region
from . import native_curved_rest as m2
from . import native_polygon_rest as m1
from .rc01_post import read_default_post


RPM = 12000
TOOL_NUMBER = 3
ENTRY_FEED = 60
CUT_FEED = 300


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _source(source_bytes, case, cap_depth):
    if case == "letter":
        project, region = m1._source(source_bytes)
        target = v_region.VTarget.polygon(_sha(source_bytes), m1.SHELL,
                                          (m1.HOLE,), cap_depth)
        start = m1.START
    elif case in ("annulus", "mixed", "mixed_reflected"):
        project, region, approximation, start = m2._source(source_bytes)
        target = v_region.VTarget.curved(_sha(source_bytes), approximation,
                                         cap_depth)
    else:
        raise ValueError("unsupported M3 source case")
    return project, region, target, start


def _plan(source_bytes, case, tool, cap_depth, stepover_mm,
          fill_pattern="raster"):
    _, _, target, start = _source(source_bytes, case, cap_depth)
    raw = v_region.plan(target, tool, stepover_mm=stepover_mm,
                        safe_z=start[2], fill_pattern=fill_pattern)
    if raw.status != "partial":
        raise ValueError("M3 source has no admissible V path")
    # The accepted CamBam Default post uses four decimal places for coordinates.
    # Reverify that exact emitted coordinate grid before making candidates.
    paths = tuple(v_region.VPath(path.role, tuple(
        tuple(round(v, 4) for v in point) for point in path.points))
                  for path in raw.paths)
    plan = replace(raw, paths=paths,
                   motions=v_region._motions(paths, start[2]))
    v_region.verify(plan)
    return plan, start


def _prior_target(plan):
    shape = plan.target.safe
    shell = tuple(tuple(p) for p in shape.exterior.coords[:-1])
    holes = tuple(tuple(tuple(p) for p in ring.coords[:-1])
                  for ring in shape.interiors)
    return replay.Target("m3-v-opening", shape.bounds, plan.target.cap_depth,
                         region_shell=shell, region_holes=holes)


def _prior_operation(plan):
    return replay.Operation("prior", replay.ToolProfile("T1", "cylinder", 0.5, 3),
                            _prior_target(plan))


def _record(item):
    if type(item) is replay.Event:
        return {"kind": item.kind, "tool": item.tool,
                "position": list(item.position)}
    return {"role": item.role, "tool": item.tool,
            "operation": item.operation, "start": list(item.start),
            "end": list(item.end), "feed": item.feed}


def _prior_trace(source_bytes, plan, start, supplied=None):
    op = _prior_operation(plan)
    if supplied is None:
        # This source-bound proof fixture leaves a conservative V wall band.
        # It is a supplied-motion authority for tests, not a production
        # roughing recommendation.
        clearance = (op.tool.radius + plan.target.cap_depth *
                     plan.tool.tangent + 0.02)
        centers = plan.target.safe.buffer(-clearance, quad_segs=32)
        if centers.is_empty:
            raise ValueError("no V prior cylindrical center region")
        items = [replay.Event("tool_change", "T1", start),
                 replay.Event("spindle_start", "T1", start)]
        ymin, ymax = centers.bounds[1], centers.bounds[3]
        y = ymin + 0.5
        while y < ymax:
            cross = centers.intersection(LineString(((centers.bounds[0] - 1, y),
                                                     (centers.bounds[2] + 1, y))))
            for segment in v_region._segments(cross):
                if segment.length < 0.2:
                    continue
                a, b = (tuple(round(v, 4) for v in point)
                        for point in (segment.coords[0], segment.coords[-1]))
                if a == b:
                    continue
                at = items[-1].position if type(items[-1]) is replay.Event else items[-1].end
                high = (a[0], a[1], start[2])
                if at != high:
                    items.append(replay.Motion("rapid", "T1", "prior", at, high))
                low = (a[0], a[1], -plan.target.cap_depth)
                end = (b[0], b[1], -plan.target.cap_depth)
                items.extend((replay.Motion("entry", "T1", "prior", high, low,
                                            ENTRY_FEED),
                              replay.Motion("cut", "T1", "prior", low, end,
                                            CUT_FEED),
                              replay.Motion("retract", "T1", "prior", end,
                                            (b[0], b[1], start[2]), CUT_FEED)))
            y += 1.5
        at = items[-1].position if type(items[-1]) is replay.Event else items[-1].end
        if at != start:
            items.append(replay.Motion("rapid", "T1", "prior", at, start))
        items.append(replay.Event("spindle_stop", "T1", start))
        trace = replay.Trace(_sha(source_bytes), "m3-" + plan.target.source_id,
                             start, (op,), tuple(items))
        v_region.with_prior(plan, trace)
        return trace
    if (type(supplied) is not dict or set(supplied) != {
            "format", "source_sha256", "target_fingerprint", "tool",
            "initial_xyz_mm", "motion_fingerprint", "items"} or
            supplied["format"] != "m3-v-prior-v1" or
            supplied["source_sha256"] != _sha(source_bytes) or
            supplied["target_fingerprint"] != plan.target.fingerprint or
            supplied["tool"] != {"name": "T1", "radius_mm": 0.5,
                                 "cutting_length_mm": 3} or
            supplied["initial_xyz_mm"] != list(start) or
            type(supplied["items"]) is not list):
        raise ValueError("stale or unsupported M3 supplied prior")
    items = []
    for row in supplied["items"]:
        if type(row) is not dict:
            raise ValueError("invalid M3 supplied prior item")
        if set(row) == {"kind", "tool", "position"}:
            items.append(replay.Event(row["kind"], row["tool"],
                                      tuple(row["position"])))
        elif set(row) == {"role", "tool", "operation", "start", "end", "feed"}:
            items.append(replay.Motion(row["role"], row["tool"],
                                       row["operation"], tuple(row["start"]),
                                       tuple(row["end"]), row["feed"]))
        else:
            raise ValueError("invalid M3 supplied prior item")
    trace = replay.Trace(_sha(source_bytes), "m3-" + plan.target.source_id,
                         start, (op,), tuple(items))
    if trace.motion_fingerprint != supplied["motion_fingerprint"]:
        raise ValueError("M3 supplied prior motion changed")
    v_region.with_prior(plan, trace)
    return trace


def _supplied_prior(source_bytes, plan, trace, start):
    return {"format": "m3-v-prior-v1", "source_sha256": _sha(source_bytes),
            "target_fingerprint": plan.target.fingerprint,
            "tool": {"name": "T1", "radius_mm": 0.5,
                     "cutting_length_mm": 3},
            "initial_xyz_mm": list(start),
            "motion_fingerprint": trace.motion_fingerprint,
            "items": [_record(item) for item in trace.items]}


def _decimal(value):
    return f"{value:.4f}".rstrip("0").rstrip(".") if value else "0"


def _motion_sequence(plan, start):
    first = plan.motions[0].start
    moves = []
    if first != start:
        moves.append(v_region.VMotion("rapid", start, first))
    moves.extend(plan.motions)
    last = moves[-1].end
    if last != start:
        moves.append(v_region.VMotion("rapid", last, start))
    return tuple(moves)


def _script(moves):
    lines = []
    for move in moves:
        x, y, z = move.end
        code = "G0" if move.role == "rapid" else "G1"
        feed = "" if code == "G0" else (
            f" F{ENTRY_FEED}" if move.role == "entry" else f" F{CUT_FEED}")
        lines.append(f"{code}{feed} X{_decimal(x)} Y{_decimal(y)} Z{_decimal(z)}")
    return "\n".join(lines)


def _combined_script(prior_trace, moves):
    lines = []
    for item in prior_trace.items[2:-1]:
        if type(item) is not replay.Motion:
            raise ValueError("unsupported M3 prior script event")
        x, y, z = item.end
        code = "G0" if item.role == "rapid" else "G1"
        feed = ("" if code == "G0" else
                f" F{ENTRY_FEED}" if item.role == "entry" else
                f" F{CUT_FEED}")
        lines.append(f"{code}{feed} X{_decimal(x)} Y{_decimal(y)} Z{_decimal(z)}")
    lines.extend(("M5", f"T{TOOL_NUMBER} M6", f"M3 S{RPM}"))
    lines.extend(_script(moves).splitlines())
    return "\n".join(lines)


def _source_unchanged(candidate, source_bytes, case, cap_depth):
    original, region, _, _ = _source(source_bytes, case, cap_depth)
    compare = candidate.get_primitive(region.user_identifier)
    if (type(compare) is not Region or
            compare.internal_id != region.internal_id or
            compare.get_absolute_coordinates_xyz() !=
            region.get_absolute_coordinates_xyz()):
        raise ValueError("M3 analytic source Region changed")
    a, b = candidate.list_parts(), original.list_parts()
    if len(a) != 1 or len(b) != 1:
        raise ValueError("M3 Part count changed")
    attributes = ("stock_drawing_origin", "stock_width", "stock_height",
                  "stock_thickness", "stock_surface", "machining_origin",
                  "enabled", "nesting_method")
    if any(getattr(a[0], name) != getattr(b[0], name) for name in attributes):
        raise ValueError("M3 original stock changed")


def _candidate(project, kind, plan, moves, start, tool, prior_trace):
    part = project.list_parts()[0]
    if kind == "preview":
        layer = project.add_layer("Generated M3 V preview only")
        paths = []
        for index, path in enumerate(plan.paths):
            points = [(x, y, -d) for x, y, d in path.points]
            shape = project.add_pline(layer, points,
                                      identifier=f"m3-preview-path-{index}")
            if shape is None:
                raise RuntimeError("could not create M3 XYZ preview")
            paths.append(shape)
        mop = project.add_engrave_mop(
            part, targets=paths, name="PREVIEW M3 V paths only",
            identifier="m3-preview", enabled=True, tool_number=TOOL_NUMBER,
            tool_diameter=2 * tool.maximum_radius, tool_profile="VCutter",
            stock_surface=0, target_depth=0, depth_increment=plan.target.cap_depth,
            roughing_clearance=0, clearance_plane=start[2],
            plunge_feedrate=ENTRY_FEED, cut_feedrate=CUT_FEED,
            spindle_direction="CW", spindle_speed=RPM, work_plane="XY",
            velocity_mode="ExactStop", optimisation_mode="None",
            max_crossover_distance=0)
    elif kind == "explicit":
        layer = project.add_layer("M3 literal motion anchor")
        anchor = project.add_points(layer, [(start[0], start[1], 0)],
                                    identifier="m3-script-anchor")
        if anchor is None:
            raise RuntimeError("could not create M3 script anchor")
        mop = project.add_drill_mop(
            part, targets=[anchor], name="M3 V literal motion",
            identifier="m3-literal", enabled=True, drilling_method="CustomScript",
            custom_script=_combined_script(prior_trace, moves), tool_number=1,
            tool_diameter=1, tool_profile="EndMill",
            stock_surface=0, target_depth=-plan.target.cap_depth,
            depth_increment=plan.target.cap_depth, clearance_plane=start[2],
            plunge_feedrate=ENTRY_FEED, cut_feedrate=CUT_FEED,
            spindle_direction="CW", spindle_speed=RPM, work_plane="XY",
            velocity_mode="ExactStop")
    else:
        raise ValueError("unknown M3 candidate kind")
    if mop is None:
        raise RuntimeError("could not create M3 MOP")


def _check_candidate(data, source_bytes, case, kind, plan, moves, start, tool,
                     prior_trace):
    project = read_cambam_bytes(data, source_name="M3 candidate")
    _source_unchanged(project, source_bytes, case, plan.target.cap_depth)
    mops = project.list_mops()
    if len(mops) != 1 or not mops[0].enabled:
        raise ValueError("M3 candidate operation changed")
    mop = mops[0]
    expected_tool = ((TOOL_NUMBER, 2 * tool.maximum_radius, "VCutter")
                     if kind == "preview" else (1, 1, "EndMill"))
    if ((mop.tool_number, mop.tool_diameter, mop.tool_profile,
         mop.clearance_plane, mop.plunge_feedrate, mop.cut_feedrate,
         mop.spindle_speed) !=
            (*expected_tool, start[2],
             ENTRY_FEED, CUT_FEED, RPM)):
        raise ValueError("M3 candidate process changed")
    if kind == "preview":
        shapes = [project.get_primitive(f"m3-preview-path-{i}")
                  for i in range(len(plan.paths))]
        if (len(project.list_primitives()) != len(plan.paths) + 1 or
                any(type(shape) is not Pline or shape.closed or
                    tuple(shape.get_absolute_coordinates_xyz()) !=
                    tuple((x, y, -d, 0.0) for x, y, d in path.points)
                    for shape, path in zip(shapes, plan.paths)) or
                type(mop) is not EngraveMop or
                mop.target_depth != 0 or mop.optimisation_mode != "None" or
                mop.max_crossover_distance != 0 or
                set(project.get_mop_targets(mop)) !=
                {shape.internal_id for shape in shapes}):
            raise ValueError("M3 preview path or MOP changed")
    else:
        anchor = project.get_primitive("m3-script-anchor")
        if (len(project.list_primitives()) != 2 or type(anchor) is not Points or
                type(mop) is not DrillMop or
                mop.drilling_method != "CustomScript" or
                mop.custom_script != _combined_script(prior_trace, moves) or
                project.get_mop_targets(mop) != [anchor.internal_id]):
            raise ValueError("M3 literal candidate changed")


def build_workflow(directory, *, case="letter", tool=None, cap_depth=2,
                   stepover_mm=1, source_path=None, prior_path=None,
                   fill_pattern="raster"):
    """Create one profile/source pair in a new ignored output directory."""
    tool = (v_region.VProfile("pointed", 90, 0, 4, 3)
            if tool is None else tool)
    if type(tool) is not v_region.VProfile:
        raise ValueError("M3 V tool required")
    if source_path is not None and prior_path is None:
        raise ValueError("existing M3 source requires supplied prior motion")
    directory = Path(directory)
    if directory.exists():
        raise ValueError("M3 output directory must be new")
    directory.mkdir(parents=True)
    source = directory / "source.cb"
    if source_path is not None:
        source.write_bytes(Path(source_path).read_bytes())
    elif case == "letter":
        m1.synthetic_source().save(str(source))
    else:
        m2.synthetic_source(case).save(str(source))
    source_bytes = source.read_bytes()
    plan, start = _plan(source_bytes, case, tool, cap_depth, stepover_mm,
                        fill_pattern)
    supplied = (None if prior_path is None else
                json.loads(Path(prior_path).read_text(encoding="utf-8")))
    prior_trace = _prior_trace(source_bytes, plan, start, supplied)
    rest = v_region.with_prior(plan, prior_trace)
    prior_file = directory / "prior.json"
    prior_file.write_text(json.dumps(_supplied_prior(
        source_bytes, plan, prior_trace, start), indent=2) + "\n",
                          encoding="utf-8")
    moves = _motion_sequence(plan, start)
    candidates = {}
    for kind in ("preview", "explicit"):
        project = read_cambam_bytes(source_bytes, source_name="M3 source")
        _candidate(project, kind, plan, moves, start, tool, prior_trace)
        path = directory / kind / f"m3-{kind}.cb"
        path.parent.mkdir()
        project.save(str(path))
        _check_candidate(path.read_bytes(), source_bytes, case, kind,
                         plan, moves, start, tool, prior_trace)
        candidates[kind] = {"file": str(path.relative_to(directory)),
                            "sha256": _sha(path.read_bytes())}
    manifest = {
        "format": "m3-v-region-v1", "case": case, "tool": vars(tool),
        "cap_depth": cap_depth, "stepover_mm": stepover_mm,
        "source_sha256": _sha(source_bytes), "target_fingerprint": plan.target.fingerprint,
        "prior_sha256": _sha(prior_file.read_bytes()),
        "prior_motion_fingerprint": prior_trace.motion_fingerprint,
        "plan_fingerprint": plan.fingerprint, "candidates": candidates,
        "initial_xyz_mm": start,
        "move_count": len(prior_trace.items) - 3 + len(moves),
        "script_sha256": _sha(_combined_script(prior_trace, moves).encode("utf-8")),
        "prior_section_1_mm2": v_region.section_report(rest, 1, final=False),
        "section_1_mm2": v_region.section_report(rest, 1),
        "postprocessor": "Default", "profile": "Default mm",
        "post_status": "pending_actual_CamBam_post",
    }
    if fill_pattern != "raster":
        manifest["fill_pattern"] = fill_pattern
    (directory / "expected-motion.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit_post(manifest_path, posted_path):
    """Audit both posted tools, source-bound stock and the final V paths."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "m3-v-region-v1":
        raise ValueError("not an M3 V Region manifest")
    source_bytes = (root / "source.cb").read_bytes()
    prior_file = root / "prior.json"
    if (_sha(source_bytes) != manifest["source_sha256"] or
            _sha(prior_file.read_bytes()) != manifest["prior_sha256"]):
        raise ValueError("M3 source changed")
    tool = v_region.VProfile(**manifest["tool"])
    plan, start = _plan(source_bytes, manifest["case"], tool,
                        manifest["cap_depth"], manifest["stepover_mm"],
                        manifest.get("fill_pattern", "raster"))
    prior_trace = _prior_trace(source_bytes, plan, start,
        json.loads(prior_file.read_text(encoding="utf-8")))
    rest = v_region.with_prior(plan, prior_trace)
    moves = _motion_sequence(plan, start)
    if (plan.target.fingerprint != manifest["target_fingerprint"] or
            plan.fingerprint != manifest["plan_fingerprint"] or
            prior_trace.motion_fingerprint != manifest["prior_motion_fingerprint"] or
            len(prior_trace.items) - 3 + len(moves) != manifest["move_count"] or
            _sha(_combined_script(prior_trace, moves).encode("utf-8")) !=
            manifest["script_sha256"] or
            list(v_region.section_report(rest, 1, final=False)) !=
            manifest["prior_section_1_mm2"] or
            list(v_region.section_report(rest, 1)) != manifest["section_1_mm2"]):
        raise ValueError("M3 target, plan or stock report changed")
    entry = manifest["candidates"]["explicit"]
    candidate = root / entry["file"]
    data = candidate.read_bytes()
    if _sha(data) != entry["sha256"]:
        raise ValueError("M3 candidate changed")
    _check_candidate(data, source_bytes, manifest["case"], "explicit",
                     plan, moves, start, tool, prior_trace)
    posted = Path(posted_path).read_text(encoding="utf-8-sig")
    lines = posted.splitlines()
    if ("( Post processor: Default )" not in lines[:12] or
            not any(line.startswith(f"( {candidate.stem} ") for line in lines[:12])):
        raise ValueError("M3 post header does not match candidate")
    normalized = "\n".join("" if line.strip() in ("G98", "G80") else line
                           for line in lines)
    actual, warnings = read_default_post(normalized, initial_position=start)
    warnings = [w for w in warnings if
                "initial machine position is not encoded" not in w]
    if warnings:
        return {"status": "unverified", "reason": warnings[0]}
    expected = list(prior_trace.items) + [
        replay.Event("tool_change", f"T{TOOL_NUMBER}", start),
        replay.Event("spindle_start", f"T{TOOL_NUMBER}", start)]
    expected.extend(moves)
    expected.append(replay.Event("spindle_stop", f"T{TOOL_NUMBER}", start))
    if (len(actual) == len(expected) + 1 and
            actual[-1]["type"] == "event" and
            actual[-1]["kind"] == "spindle_stop"):
        actual = actual[:-1]
    if len(actual) != len(expected):
        return {"status": "deviation", "reason": "extra or missing post item",
                "expected_items": len(expected), "actual_items": len(actual)}
    observed_prior, observed_v = [], []
    for index, (want, got) in enumerate(zip(expected, actual)):
        if type(want) is replay.Event:
            if (got["type"] != "event" or got["kind"] != want.kind or
                    got["tool"] != want.tool or
                    tuple(got["position"]) != want.position or
                    (want.kind == "spindle_start" and got["rpm"] != RPM)):
                return {"status": "deviation", "item_index": index,
                        "reason": "tool or spindle event changed"}
            if index < len(prior_trace.items):
                observed_prior.append(replay.Event(want.kind, want.tool,
                                                   tuple(got["position"])))
            continue
        tool_name = (want.tool if type(want) is replay.Motion else
                     f"T{TOOL_NUMBER}")
        feed = (want.feed if type(want) is replay.Motion else
                0 if want.role == "rapid" else
                ENTRY_FEED if want.role == "entry" else CUT_FEED)
        if (got["type"] != "move" or got["tool"] != tool_name or
                got["g"] != (0 if want.role == "rapid" else 1) or
                got["feed"] != feed or tuple(got["start"]) != want.start or
                tuple(got["end"]) != want.end):
            return {"status": "deviation", "item_index": index,
                    "line": got.get("line")}
        if type(want) is replay.Motion:
            observed_prior.append(replay.Motion(want.role, want.tool,
                want.operation, tuple(got["start"]), tuple(got["end"]), got["feed"]))
        else:
            observed_v.append(v_region.VMotion(want.role, tuple(got["start"]),
                                                tuple(got["end"])))
    posted_prior = replay.Trace(prior_trace.source_fingerprint,
        prior_trace.frame, start, prior_trace.operations, tuple(observed_prior))
    posted_rest = v_region.with_prior(plan, posted_prior)
    if tuple(observed_v[1:-1] if moves[0].role == "rapid" else observed_v[:-1]) != plan.motions:
        return {"status": "deviation", "reason": "posted V path changed"}
    if (v_region.section_report(posted_rest, 1, final=False) !=
            v_region.section_report(rest, 1, final=False) or
            v_region.section_report(posted_rest, 1) !=
            v_region.section_report(rest, 1)):
        return {"status": "deviation", "reason": "posted V stock changed"}
    return {"status": "bounded_m3_v_post_pass",
            "post_sha256": _sha(Path(posted_path).read_bytes()),
            "candidate_sha256": manifest["candidates"]["explicit"]["sha256"],
            "item_count": len(actual), "path_count": len(plan.paths),
            "prior_cut_count": len(posted_rest.prior_stock.cuts),
            "v_cut_segment_count": sum(len(path.points) - 1
                                       for path in plan.paths),
            "prior_section_1_mm2": v_region.section_report(posted_rest, 1,
                                                             final=False),
            "section_1_mm2": v_region.section_report(posted_rest, 1),
            "completion": plan.status}


def audit_preview(manifest_path, posted_path):
    """Check Engrave's visible XYZ centerlines; no execution credit follows."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "m3-v-region-v1":
        raise ValueError("not an M3 V Region manifest")
    source_bytes = (root / "source.cb").read_bytes()
    prior_file = root / "prior.json"
    entry = manifest["candidates"]["preview"]
    candidate = root / entry["file"]
    if (_sha(source_bytes) != manifest["source_sha256"] or
            _sha(prior_file.read_bytes()) != manifest["prior_sha256"] or
            _sha(candidate.read_bytes()) != entry["sha256"]):
        raise ValueError("M3 preview source or candidate changed")
    tool = v_region.VProfile(**manifest["tool"])
    plan, start = _plan(source_bytes, manifest["case"], tool,
                        manifest["cap_depth"], manifest["stepover_mm"],
                        manifest.get("fill_pattern", "raster"))
    prior_trace = _prior_trace(source_bytes, plan, start,
        json.loads(prior_file.read_text(encoding="utf-8")))
    if plan.fingerprint != manifest["plan_fingerprint"]:
        raise ValueError("M3 preview plan changed")
    _check_candidate(candidate.read_bytes(), source_bytes, manifest["case"],
                     "preview", plan, _motion_sequence(plan, start), start, tool,
                     prior_trace)
    lines = Path(posted_path).read_text(encoding="utf-8-sig").splitlines()
    if ("( Post processor: Default )" not in lines[:12] or
            not any(line.startswith(f"( {candidate.stem} ") for line in lines[:12])):
        raise ValueError("M3 preview post header changed")
    actual, warnings = read_default_post("\n".join(lines), initial_position=start)
    warnings = [w for w in warnings if
                "initial machine position is not encoded" not in w]
    if warnings:
        return {"status": "unverified", "reason": warnings[0]}
    def key(a, b):
        return tuple(sorted((tuple(a), tuple(b))))
    expected = Counter(key(a, b) for path in plan.paths
                       for a, b in zip(
                           ((x, y, -d) for x, y, d in path.points),
                           ((x, y, -d) for x, y, d in path.points[1:])))
    observed = Counter()
    first_line = {}
    for item in actual:
        if item["type"] != "move":
            continue
        a, b = item["start"], item["end"]
        if (item["tool"] != f"T{TOOL_NUMBER}" or
                min(a[2], b[2]) < -plan.target.cap_depth - 1e-7 or
                (item["g"] == 0 and min(a[2], b[2]) < 0 and a[:2] != b[:2])):
            return {"status": "deviation", "reason": "unsafe preview motion",
                    "line": item["line"]}
        if item["g"] == 1 and a[:2] != b[:2]:
            segment = key(a, b)
            observed[segment] += 1
            first_line.setdefault(segment, item["line"])
    if observed != expected:
        extra = observed - expected
        return {"status": "deviation", "reason": "preview centerlines differ",
                "expected_segments": sum(expected.values()),
                "observed_segments": sum(observed.values()),
                "first_extra_line": (first_line[next(iter(extra))]
                                     if extra else None)}
    return {"status": "m3_v_preview_centerlines_match",
            "post_sha256": _sha(Path(posted_path).read_bytes()),
            "centerline_segments": sum(expected.values()),
            "scope": "inspection only; whole Engrave post not stock certified"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M3 Region V candidate and post gate")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("directory")
    build.add_argument("--case", choices=("letter", "annulus", "mixed",
                                           "mixed_reflected"), default="letter")
    build.add_argument("--profile", choices=("pointed", "flat", "rounded"),
                       default="pointed")
    build.add_argument("--source")
    build.add_argument("--prior")
    build.add_argument("--fill", choices=("raster", "offset"),
                       default="raster")
    for name in ("audit", "preview"):
        check = sub.add_parser(name)
        check.add_argument("manifest")
        check.add_argument("posted")
    args = parser.parse_args()
    if args.command == "build":
        profiles = {
            "pointed": v_region.VProfile("pointed", 90, 0, 4, 3),
            "flat": v_region.VProfile("flat", 90, 0.25, 4, 3),
            "rounded": v_region.VProfile("rounded", 60, 0.5, 4, 3),
        }
        report = build_workflow(args.directory, case=args.case,
                                tool=profiles[args.profile],
                                source_path=args.source, prior_path=args.prior,
                                fill_pattern=args.fill)
    elif args.command == "audit":
        report = audit_post(args.manifest, args.posted)
    else:
        report = audit_preview(args.manifest, args.posted)
    print(json.dumps(report, indent=2))
