"""One XYZ-Pline/Engrave probe for the bounded tapered V cut."""

import json
from pathlib import Path

from ... import CBProject
from ...cad_entities import Pline
from ...cam_core import replay, tapered_vcarve
from ...cam_entities import EngraveMop
from ...cambam_reader import read_cambam_bytes
from .cone_script import _compare_post, _expected_items, _sha
from .rc01_post import read_default_post
from .variable_cone_script import _sections, _trace


def _vertices(shape):
    return tuple((vertex.x, vertex.y, vertex.z, vertex.bulge)
                 for vertex in shape.vertices)


def _check_candidate(path, plan):
    project = read_cambam_bytes(path.read_bytes(), source_name=str(path))
    guide = project.get_primitive("tapered-target-spine")
    cut = project.get_primitive("generated-variable-v-cut")
    x0, x1, y, d0, d1 = plan.target_spine
    cx0, cx1, cy, cd0, cd1 = plan.cut_spine
    if (type(guide) is not Pline or type(cut) is not Pline or
            _vertices(guide) != ((x0, y, -d0, 0.0), (x1, y, -d1, 0.0)) or
            _vertices(cut) != ((cx0, cy, -cd0, 0.0), (cx1, cy, -cd1, 0.0))):
        raise ValueError("XYZ target or cut path changed on strict reimport")
    enabled = [m for m in project.list_mops() if m.enabled]
    if (len(enabled) != 1 or type(enabled[0]) is not EngraveMop or
            project.get_mop_targets(enabled[0]) != [cut.internal_id]):
        raise ValueError("Engrave must target only the generated XYZ cut")
    mop = enabled[0]
    fields = (mop.tool_number, mop.tool_diameter, mop.tool_profile,
              mop.stock_surface, mop.target_depth, mop.depth_increment,
              mop.roughing_clearance, mop.clearance_plane,
              mop.plunge_feedrate, mop.cut_feedrate, mop.spindle_direction,
              mop.spindle_speed, mop.work_plane, mop.velocity_mode,
              mop.optimisation_mode)
    if fields != (3, 6, "VCutter", 0, 0, 3, 0, 5, 60, 300,
                  "CW", 12000, "XY", "ExactStop", "None"):
        raise ValueError("Engrave process fields changed on strict reimport")
    return project


def build_engrave_candidate(directory, *, native_source_bytes=None, native_setup=None):
    """Create a separate finish guide and active generated path; pin exact bytes."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("XYZ Engrave output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    if native_source_bytes is not None:
        from .native_variable_v import normalize_bytes
        request = normalize_bytes(native_source_bytes, native_setup)
    else:
        request = tapered_vcarve.standalone_request()
    plan = tapered_vcarve.generate(request)
    trace = _trace(plan)
    replay.replay(trace, expected_source=plan.fingerprint)

    source_path = directory / "source.cb"
    if native_source_bytes is None:
        source = CBProject("Tapered variable-depth V groove synthetic input")
        guide_layer = source.add_layer("Exact target spine guide")
        x0, x1, y, d0, d1 = plan.target_spine
        guide = source.add_pline(guide_layer, [(x0, y, -d0), (x1, y, -d1)],
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
    path_layer = project.add_layer("Generated XYZ cut path")
    cx0, cx1, cy, cd0, cd1 = plan.cut_spine
    cut = project.add_pline(path_layer, [(cx0, cy, -cd0),
                                         (cx1, cy, -cd1)],
                            identifier="generated-variable-v-cut")
    mop = project.add_engrave_mop(
        project.list_parts()[0], targets=[cut],
        name="CANDIDATE variable-depth XYZ Engrave",
        identifier="variable-v-xyz-engrave", enabled=True,
        tool_number=3, tool_diameter=6, tool_profile="VCutter",
        stock_surface=0, target_depth=0, depth_increment=3,
        roughing_clearance=0, clearance_plane=5,
        plunge_feedrate=60, cut_feedrate=300,
        spindle_direction="CW", spindle_speed=12000,
        work_plane="XY", velocity_mode="ExactStop",
        optimisation_mode="None")
    if cut is None or mop is None:
        raise RuntimeError("could not attach XYZ Engrave candidate")
    candidate = directory / "V-variable-engrave.cb"
    project.save(str(candidate))
    reopened = _check_candidate(candidate, plan)
    if native_source_bytes is not None and (
            not source_ids.issubset({p.internal_id for p in reopened.list_primitives()}) or
            len([m for m in reopened.list_mops() if not m.enabled]) != 1):
        raise ValueError("native V source changed during preview attachment")
    if native_source_bytes is not None:
        from .native_variable_v import normalize
        if normalize(reopened, native_setup, allow_attachments=True) != request:
            raise ValueError("native V request changed during preview attachment")
    expected = {
        "format": "variable-cone-engrave-v1",
        "candidate": candidate.name,
        "candidate_sha256": _sha(candidate),
        "source_sha256": _sha(source_path),
        "plan_fingerprint": plan.fingerprint,
        "motion_fingerprint": trace.motion_fingerprint,
        "postprocessor": "Default",
        "profile": "Default mm",
        "target_spine_x0_x1_y_d0_d1_mm": plan.target_spine,
        "cut_spine_x0_x1_y_d0_d1_mm": plan.cut_spine,
        "setup_tip_xyz_mm": trace.initial_position,
        "expected_items": _expected_items(trace),
        "expected_section_rest_mm2": _sections(tapered_vcarve.verify(plan)),
        "status": "pending_CamBam_preview_and_post",
    }
    (directory / "expected-motion.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in expected.items()
            if key != "expected_items"}


def _observations(posted_path, plan, trace):
    posted = Path(posted_path).read_text(encoding="utf-8-sig")
    normalized = "\n".join("" if line.strip() in ("G98", "G80") else line
                           for line in posted.splitlines())
    items, warnings = read_default_post(normalized)
    cut_start = (plan.cut_spine[0], plan.cut_spine[2], -plan.cut_spine[3])
    cut_end = (plan.cut_spine[1], plan.cut_spine[2], -plan.cut_spine[4])
    sloped = [item for item in items if item["type"] == "move" and
              item["g"] == 1 and tuple(item["start"]) == cut_start and
              tuple(item["end"]) == cut_end and item["feed"] == 300]
    other_xy_cuts = [item for item in items if item["type"] == "move" and
                     item["g"] == 1 and item["start"][:2] != item["end"][:2]
                     and item not in sloped]
    missing = []
    for want in trace.items:
        if type(want) is replay.Motion:
            found = any(got["type"] == "move" and
                        got["tool"] == want.tool and
                        got["g"] == (0 if want.role == "rapid" else 1) and
                        got["feed"] == want.feed and
                        tuple(got["start"]) == want.start and
                        tuple(got["end"]) == want.end for got in items)
            label = want.role
        else:
            found = any(got["type"] == "event" and
                        got["kind"] == want.kind and
                        got["tool"] == want.tool and
                        tuple(got["position"]) == want.position and
                        (want.kind != "spindle_start" or got["rpm"] == 12000)
                        for got in items)
            label = want.kind
        if not found:
            missing.append(label)
    return {"item_count": len(items), "sloped_cut_count": len(sloped),
            "other_xy_feed_moves": other_xy_cuts,
            "missing_expected_roles": missing,
            "warnings": warnings, "actual_items": items}


def audit_engrave_post(expected_path, posted_path):
    """Compare every native emitted role; report cut visibility separately."""
    expected_path = Path(expected_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if expected.get("format") != "variable-cone-engrave-v1":
        raise ValueError("not a bounded XYZ Engrave manifest")
    directory = expected_path.parent
    candidate = directory / expected["candidate"]
    if (_sha(candidate) != expected["candidate_sha256"] or
            _sha(directory / "source.cb") != expected["source_sha256"]):
        raise ValueError("XYZ Engrave source or candidate changed")
    plan = tapered_vcarve.generate()
    trace = _trace(plan)
    if (expected["plan_fingerprint"] != plan.fingerprint or
            expected["motion_fingerprint"] != trace.motion_fingerprint or
            expected["target_spine_x0_x1_y_d0_d1_mm"] != list(plan.target_spine) or
            expected["cut_spine_x0_x1_y_d0_d1_mm"] != list(plan.cut_spine) or
            expected["setup_tip_xyz_mm"] != list(trace.initial_position) or
            expected["expected_items"] != _expected_items(trace) or
            expected["expected_section_rest_mm2"] !=
            _sections(tapered_vcarve.verify(plan))):
        raise ValueError("XYZ Engrave manifest is stale")
    _check_candidate(candidate, plan)
    posted_path = Path(posted_path)
    posted = posted_path.read_text(encoding="utf-8-sig")
    if ("( Post processor: Default )" not in posted or
            not any(line.startswith(f"( {candidate.stem} ")
                    for line in posted.splitlines()[:6])):
        raise ValueError("CamBam post header does not match candidate/Default")
    try:
        observations = _observations(posted_path, plan, trace)
        posted_trace, failure = _compare_post(trace, candidate, posted_path)
    except ValueError as error:
        return {"status": "unverified_engrave_post",
                "candidate_sha256": expected["candidate_sha256"],
                "post_sha256": _sha(posted_path), "reason": str(error)}
    if failure:
        return {"status": "engrave_emitted_motion_deviation",
                "candidate_sha256": expected["candidate_sha256"],
                "post_sha256": _sha(posted_path),
                "first_deviation": failure, **observations}
    stock = replay.replay(posted_trace, expected_source=plan.fingerprint)
    result = tapered_vcarve.TaperedResult(plan, stock)
    areas = _sections(result)
    if any(abs(areas[k] - v) > 1e-9 for k, v in
           expected["expected_section_rest_mm2"].items()):
        return {"status": "engrave_stock_deviation",
                "section_rest_mm2": areas, **observations}
    return {"status": "bounded_emitted_variable_v_engrave_pass",
            "candidate_sha256": expected["candidate_sha256"],
            "post_sha256": _sha(posted_path),
            "stock_prefixes": stock.prefixes,
            "section_rest_mm2": areas, **observations}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit XYZ Engrave probe")
    parser.add_argument("path", help="new output directory or expected-motion.json")
    parser.add_argument("post", nargs="?", help="CamBam-posted .nc")
    args = parser.parse_args()
    result = (audit_engrave_post(args.path, args.post) if args.post else
              build_engrave_candidate(args.path))
    print(json.dumps(result, indent=2))
    if args.post and result["status"] != "bounded_emitted_variable_v_engrave_pass":
        raise SystemExit(1)
