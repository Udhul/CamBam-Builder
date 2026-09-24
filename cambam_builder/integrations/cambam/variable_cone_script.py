"""Bounded variable-depth cone carrier and exact Default-post replay."""

import json
from pathlib import Path

from ... import CBProject
from ...cam_core import replay, tapered_vcarve
from ...cam_entities import DrillMop
from ...cambam_reader import read_cambam_bytes
from .cone_script import _compare_post, _expected_items, _script, _sha


FEEDS = tapered_vcarve.OUTPUT_FEEDS
RPM = tapered_vcarve.OUTPUT_RPM
START = tapered_vcarve.OUTPUT_SETUP
TOOL = tapered_vcarve.OUTPUT_TOOL


def _trace(plan):
    return tapered_vcarve.output_trace(plan)


def _sections(result):
    return {f"depth_{depth:g}": result.residual_area(depth)
            for depth in (0, 1, 1.5, 2, 2.5)}


def build_variable_carrier(directory, *, native_source_bytes=None, native_setup=None):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("variable V output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    if native_source_bytes is not None:
        from .native_variable_v import normalize_bytes
        request = normalize_bytes(native_source_bytes, native_setup)
    else:
        request = tapered_vcarve.standalone_request()
    plan = tapered_vcarve.generate(request)
    result = tapered_vcarve.verify(plan)
    trace = _trace(plan)
    replay.replay(trace, expected_source=plan.fingerprint)

    source_path = directory / "source.cb"
    if native_source_bytes is None:
        source = CBProject("Tapered variable-depth V groove synthetic input")
        layer = source.add_layer("Exact target spine guide")
        x0, x1, y, d0, d1 = plan.target_spine
        guide = source.add_pline(layer, [(x0, y, -d0), (x1, y, -d1)],
                                 identifier="tapered-target-spine")
        part = source.add_part("Tapered V part", stock_thickness=3,
                               stock_width=18, stock_height=8,
                               stock_offset=(-2, -2), stock_surface=0)
        if guide is None or part is None:
            raise RuntimeError("could not create variable V source")
        source.save(str(source_path))
    else:
        source_path.write_bytes(native_source_bytes)

    project = read_cambam_bytes(source_path.read_bytes(),
                                source_name=str(source_path))
    source_ids = {p.internal_id for p in project.list_primitives()}
    anchor_layer = project.add_layer("Literal motion anchor")
    anchor = project.add_points(anchor_layer, [(-10, -10, 0)],
                                identifier="variable-script-anchor")
    script = _script(trace)
    mop = project.add_drill_mop(
        project.list_parts()[0], targets=[anchor],
        name="Variable-depth V groove literal motion",
        identifier="variable-v-literal-motion", enabled=True,
        drilling_method="CustomScript", custom_script=script,
        tool_number=3, tool_diameter=6, tool_profile="VCutter",
        stock_surface=0, target_depth=-2.25, depth_increment=2.25,
        clearance_plane=5, plunge_feedrate=60, cut_feedrate=300,
        spindle_direction="CW", spindle_speed=RPM,
        work_plane="XY", velocity_mode="ExactStop")
    if anchor is None or mop is None:
        raise RuntimeError("could not attach variable V carrier")
    candidate = directory / "V-variable.cb"
    project.save(str(candidate))
    reopened = read_cambam_bytes(candidate.read_bytes(),
                                source_name=str(candidate))
    enabled = [m for m in reopened.list_mops() if m.enabled]
    if (len(enabled) != 1 or type(enabled[0]) is not DrillMop or
            enabled[0].drilling_method != "CustomScript" or
            enabled[0].custom_script != script or
            enabled[0].tool_profile != "VCutter" or
            reopened.get_mop_targets(enabled[0]) != [anchor.internal_id]):
        raise ValueError("variable V carrier changed on strict reimport")
    if native_source_bytes is not None and (
            not source_ids.issubset({p.internal_id for p in reopened.list_primitives()}) or
            len([m for m in reopened.list_mops() if not m.enabled]) != 1):
        raise ValueError("native V source changed during explicit attachment")
    if native_source_bytes is not None:
        from .native_variable_v import normalize
        if normalize(reopened, native_setup, allow_attachments=True) != request:
            raise ValueError("native V request changed during explicit attachment")
    expected = {
        "format": "variable-cone-script-v1",
        "candidate": candidate.name,
        "candidate_sha256": _sha(candidate),
        "source_sha256": _sha(source_path),
        "plan_fingerprint": plan.fingerprint,
        "motion_fingerprint": trace.motion_fingerprint,
        "postprocessor": "Default",
        "profile": "Default mm",
        "setup_tip_xyz_mm": START,
        "target_spine_x0_x1_y_d0_d1_mm": plan.target_spine,
        "cut_spine_x0_x1_y_d0_d1_mm": plan.cut_spine,
        "tool": {"number": 3, "profile": "90-degree pointed cone",
                 "maximum_radius_mm": 3, "conical_length_mm": 3},
        "spindle_rpm": RPM,
        "feeds_mm_min": FEEDS,
        "script_lines": script.splitlines(),
        "expected_items": _expected_items(trace),
        "item_count": len(trace.items),
        "expected_section_rest_mm2": _sections(result),
        "emitted_motion_status": "pending_CamBam_post",
    }
    (directory / "expected-motion.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in expected.items()
            if key not in ("script_lines", "expected_items")}


def audit_variable_post(expected_path, posted_path):
    expected_path = Path(expected_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if expected.get("format") != "variable-cone-script-v1":
        raise ValueError("not a bounded variable V manifest")
    directory = expected_path.parent
    candidate = directory / expected["candidate"]
    if (_sha(candidate) != expected["candidate_sha256"] or
            _sha(directory / "source.cb") != expected["source_sha256"]):
        raise ValueError("variable V source or candidate changed")
    plan = tapered_vcarve.generate()
    trace = _trace(plan)
    if (plan.fingerprint != expected["plan_fingerprint"] or
            trace.motion_fingerprint != expected["motion_fingerprint"] or
            plan.target_spine != tuple(expected["target_spine_x0_x1_y_d0_d1_mm"]) or
            plan.cut_spine != tuple(expected["cut_spine_x0_x1_y_d0_d1_mm"]) or
            _script(trace).splitlines() != expected["script_lines"] or
            _expected_items(trace) != expected["expected_items"] or
            len(trace.items) != expected["item_count"] or
            _sections(tapered_vcarve.verify(plan)) !=
            expected["expected_section_rest_mm2"]):
        raise ValueError("variable V motion manifest is stale")
    reopened = read_cambam_bytes(candidate.read_bytes(), source_name=str(candidate))
    enabled = [m for m in reopened.list_mops() if m.enabled]
    if (len(enabled) != 1 or type(enabled[0]) is not DrillMop or
            enabled[0].custom_script != _script(trace)):
        raise ValueError("variable V candidate script differs from manifest")
    posted_trace, failure = _compare_post(trace, candidate, posted_path)
    if failure:
        return failure
    stock = replay.replay(posted_trace, expected_source=plan.fingerprint)
    result = tapered_vcarve.TaperedResult(plan, stock)
    areas = _sections(result)
    if any(abs(areas[k] - v) > 1e-9
           for k, v in expected["expected_section_rest_mm2"].items()):
        return {"status": "deviation", "reason": "posted section residual changed"}
    return {
        "status": "bounded_emitted_variable_v_motion_pass",
        "candidate_sha256": expected["candidate_sha256"],
        "post_sha256": _sha(posted_path),
        "item_count": len(posted_trace.items),
        "stock_prefixes": stock.prefixes,
        "completion": result.completion,
        "section_rest_mm2": areas,
        "initial_position_assumption": "tip begins at (-10,-10,+5)",
        "scope": "ideal pointed cone, zero physical error; no holder or fixture proof",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit variable V carrier")
    parser.add_argument("path", help="new output directory or expected-motion.json")
    parser.add_argument("post", nargs="?", help="CamBam-posted V-variable.nc")
    args = parser.parse_args()
    result = (audit_variable_post(args.path, args.post) if args.post else
              build_variable_carrier(args.path))
    print(json.dumps(result, indent=2))
    if args.post and result["status"] != "bounded_emitted_variable_v_motion_pass":
        raise SystemExit(1)
