"""Bounded pointed-cone CustomScript carrier and Default-post stock replay.

The .cb carries one full-depth synthetic V slot. Only a CamBam-produced post
can establish output acceptance; the expected motion is a comparison reference.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from ... import CBProject
from ...cam_core import replay, vcarve
from ...cam_entities import DrillMop
from ...cambam_reader import read_cambam_bytes
from .rc01_post import read_default_post


START = (-10.0, -10.0, 5.0)
TOOL = "T3"
RPM = 12000
FEEDS = {"approach": 120, "entry": 60, "cut": 300, "retract": 300}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _trace(plan):
    """Add a concrete setup and process to the detached cone geometry."""
    original = vcarve.replay_trace(plan)
    first = plan.motions[0].start
    at_clearance = (first[0], first[1], 5.0)
    items = [replay.Event("tool_change", TOOL, START),
             replay.Event("spindle_start", TOOL, START),
             replay.Motion("rapid", TOOL, "slot", START, at_clearance),
             replay.Motion("approach", TOOL, "slot", at_clearance,
                           first, FEEDS["approach"])]
    for motion in plan.motions:
        role = "entry" if motion.role == "plunge" else motion.role
        items.append(replay.Motion(role, TOOL, "slot", motion.start,
                                   motion.end, FEEDS.get(role, 0)))
    last = items[-1].end
    items.extend((replay.Motion("rapid", TOOL, "slot", last, START),
                  replay.Event("spindle_stop", TOOL, START)))
    operation = replace(original.operations[0],
                        tool=replace(original.operations[0].tool, name=TOOL))
    return replay.Trace(plan.fingerprint, plan.slot.frame, START,
                        (operation,), tuple(items))


def _line(item):
    if type(item) is replay.Motion:
        x, y, z = item.end
        command = "G0" if item.role == "rapid" else "G1"
        feed = f" F{item.feed}" if item.feed else ""
        return f"{command}{feed} X{x:g} Y{y:g} Z{z:g}"
    raise ValueError("script contains an unsupported event")


def _script(trace):
    # CamBam supplies the first change/start and the terminal stop. Its
    # CustomScript property needs real newlines, as proven by the RC01 post.
    return "\n".join(_line(item) for item in trace.items[2:-1])


def _expected_items(trace):
    items = []
    for item in trace.items:
        if type(item) is replay.Motion:
            items.append({"type": "move", "role": item.role, "tool": item.tool,
                          "start": list(item.start), "end": list(item.end),
                          "feed_mm_min": item.feed})
        else:
            items.append({"type": "event", "kind": item.kind,
                          "tool": item.tool, "position": list(item.position),
                          "rpm": RPM if item.kind == "spindle_start" else 0})
    return items


def build_cone_carrier(directory):
    """Write one synthetic .cb and its exact, hash-guarded motion reference."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("cone output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    plan = vcarve.generate_slot()
    trace = _trace(plan)
    replay.replay(trace, expected_source=plan.fingerprint)
    source = CBProject("Cone V-slot synthetic input")
    layer = source.add_layer("V-slot target")
    target = source.add_rect(layer, (0, 0), 12, 4,
                             identifier="cone-slot-target")
    part = source.add_part("Cone V-slot part", stock_thickness=3,
                           stock_width=14, stock_height=6,
                           stock_offset=(-1, -1), stock_surface=0)
    if target is None or part is None:
        raise RuntimeError("could not create cone source")
    source_path = directory / "source.cb"
    source.save(str(source_path))
    project = read_cambam_bytes(source_path.read_bytes(),
                                source_name=str(source_path))
    anchor_layer = project.add_layer("Literal motion anchor")
    anchor = project.add_points(anchor_layer, [(-10, -10, 0)],
                                identifier="cone-script-anchor")
    part = project.list_parts()[0]
    script = _script(trace)
    mop = project.add_drill_mop(
        part, targets=[anchor], name="Cone full-depth V-slot literal motion",
        identifier="cone-literal-motion", enabled=True,
        drilling_method="CustomScript", custom_script=script,
        tool_number=3, tool_diameter=6, tool_profile="VCutter",
        stock_surface=0, target_depth=-2, depth_increment=2,
        clearance_plane=5, plunge_feedrate=60, cut_feedrate=300,
        spindle_direction="CW", spindle_speed=RPM,
        work_plane="XY", velocity_mode="ExactStop")
    if anchor is None or mop is None:
        raise RuntimeError("could not attach cone literal-motion carrier")
    candidate = directory / "V-cone.cb"
    project.save(str(candidate))
    reopened = read_cambam_bytes(candidate.read_bytes(), source_name=str(candidate))
    enabled = [m for m in reopened.list_mops() if m.enabled]
    if (len(enabled) != 1 or type(enabled[0]) is not DrillMop or
            enabled[0].drilling_method != "CustomScript" or
            enabled[0].custom_script != script or
            enabled[0].tool_profile != "VCutter" or
            reopened.get_mop_targets(enabled[0]) != [anchor.internal_id]):
        raise ValueError("cone carrier changed on strict reimport")
    expected = {
        "format": "cone-script-v1",
        "candidate": candidate.name,
        "candidate_sha256": _sha(candidate),
        "source_sha256": _sha(source_path),
        "plan_fingerprint": plan.fingerprint,
        "motion_fingerprint": trace.motion_fingerprint,
        "postprocessor": "Default",
        "profile": "Default mm",
        "setup_tip_xyz_mm": START,
        "tool": {"number": 3, "profile": "90-degree pointed cone",
                 "maximum_radius_mm": 3, "conical_length_mm": 3},
        "spindle_rpm": RPM,
        "feeds_mm_min": FEEDS,
        "script_lines": script.splitlines(),
        "expected_items": _expected_items(trace),
        "item_count": len(trace.items),
        "expected_section_rest_mm2": {
            "surface": vcarve.verify_slot(plan).residual_area(0),
            "depth_1": vcarve.verify_slot(plan).residual_area(1),
            "depth_2": vcarve.verify_slot(plan).residual_area(2),
        },
        "emitted_motion_status": "pending_CamBam_post",
    }
    (directory / "expected-motion.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in expected.items()
            if key not in ("script_lines", "expected_items")}


def audit_cone_post(expected_path, posted_path):
    """Read actual emitted items, compare every role, then replay their sweeps."""
    expected_path = Path(expected_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if expected.get("format") != "cone-script-v1":
        raise ValueError("not a bounded cone manifest")
    directory = expected_path.parent
    candidate = directory / expected["candidate"]
    if (_sha(candidate) != expected["candidate_sha256"] or
            _sha(directory / "source.cb") != expected["source_sha256"]):
        raise ValueError("cone source or candidate changed after preparation")
    plan = vcarve.generate_slot()
    trace = _trace(plan)
    if (plan.fingerprint != expected["plan_fingerprint"] or
            trace.motion_fingerprint != expected["motion_fingerprint"] or
            _script(trace).splitlines() != expected["script_lines"] or
            _expected_items(trace) != expected["expected_items"] or
            len(trace.items) != expected["item_count"]):
        raise ValueError("cone motion manifest is stale")
    reopened = read_cambam_bytes(candidate.read_bytes(), source_name=str(candidate))
    enabled = [m for m in reopened.list_mops() if m.enabled]
    if (len(enabled) != 1 or type(enabled[0]) is not DrillMop or
            enabled[0].custom_script != _script(trace)):
        raise ValueError("cone candidate script differs from manifest")
    posted_path = Path(posted_path)
    post = posted_path.read_text(encoding="utf-8-sig")
    if ("( Post processor: Default )" not in post or
            not any(line.startswith(f"( {candidate.stem} ")
                    for line in post.splitlines()[:6])):
        raise ValueError("CamBam post header does not match candidate/Default")
    normalized = "\n".join("" if line.strip() in ("G98", "G80") else line
                           for line in post.splitlines())
    actual, warnings = read_default_post(normalized)
    warnings = [w for w in warnings
                if "initial machine position is not encoded" not in w]
    if warnings:
        return {"status": "unverified", "reason": warnings[0]}
    if (len(actual) == len(trace.items) + 1 and
            actual[-1]["type"] == "event" and
            actual[-1]["kind"] == "spindle_stop" and
            actual[-1]["tool"] == TOOL and
            tuple(actual[-1]["position"]) == START):
        actual = actual[:-1]
    if len(actual) != len(trace.items):
        return {"status": "deviation", "reason": "extra or missing emitted item",
                "expected_items": len(trace.items), "actual_items": len(actual)}
    observed = []
    for index, (want, got) in enumerate(zip(trace.items, actual)):
        if type(want) is replay.Motion:
            if got["type"] != "move":
                return {"status": "deviation", "item": index,
                        "line": got["line"], "expected_role": want.role,
                        "actual": got}
            fields = (got["tool"] == want.tool,
                      got["g"] == (0 if want.role == "rapid" else 1),
                      got["feed"] == want.feed,
                      tuple(got["start"]) == want.start,
                      tuple(got["end"]) == want.end)
            if not all(fields):
                return {"status": "deviation", "item": index,
                        "line": got["line"], "expected_role": want.role,
                        "actual": got}
            observed.append(replace(want, start=tuple(got["start"]),
                                    end=tuple(got["end"]), feed=got["feed"]))
        else:
            if got["type"] != "event":
                return {"status": "deviation", "item": index,
                        "line": got["line"], "expected_event": want.kind,
                        "actual": got}
            fields = (got["kind"] == want.kind,
                      got["tool"] == want.tool,
                      tuple(got["position"]) == want.position)
            if want.kind == "spindle_start":
                fields += (got["rpm"] == RPM,)
            if not all(fields):
                return {"status": "deviation", "item": index,
                        "line": got["line"], "expected_event": want.kind,
                        "actual": got}
            observed.append(replace(want, position=tuple(got["position"])))
    posted_trace = replace(trace, items=tuple(observed))
    stock = replay.replay(posted_trace, expected_source=plan.fingerprint)
    # Build the specialized cone result from the actual pass coordinates, not
    # from the original plan. The exact comparison above preserves its roles.
    slot_motions = tuple(vcarve.Motion(
        "plunge" if move.role == "entry" else move.role,
        move.start, move.end) for move in observed[4:-2])
    posted_plan = replace(plan, motions=slot_motions)
    result = vcarve.verify_slot(posted_plan)
    if stock.motion_fingerprint != posted_trace.motion_fingerprint:
        raise ValueError("posted replay fingerprint mismatch")
    return {
        "status": "bounded_emitted_cone_motion_pass",
        "candidate_sha256": expected["candidate_sha256"],
        "post_sha256": _sha(posted_path),
        "item_count": len(actual),
        "stock_prefixes": stock.prefixes,
        "completion": result.completion,
        "section_rest_mm2": {"surface": result.residual_area(0),
                             "depth_1": result.residual_area(1),
                             "depth_2": result.residual_area(2)},
        "initial_position_assumption": "tip begins at (-10,-10,+5)",
        "scope": "ideal pointed cone, zero physical error; no holder or fixture proof",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit cone script carrier")
    parser.add_argument("path", help="new output directory or expected-motion.json")
    parser.add_argument("post", nargs="?", help="CamBam-posted V-cone.nc")
    args = parser.parse_args()
    result = (audit_cone_post(args.path, args.post) if args.post else
              build_cone_carrier(args.path))
    print(json.dumps(result, indent=2))
    if args.post and result["status"] != "bounded_emitted_cone_motion_pass":
        raise SystemExit(1)
