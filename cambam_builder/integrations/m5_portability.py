"""Bounded Grbl v1.1 portability fixtures and source-bound offline audit.

The automatic changer is a synthetic external-host effect stream. Its travel,
tool and offset state are declared assumptions, not observed machine behavior.
"""

import hashlib
import json
import math
from pathlib import Path

from ..cam_core import replay, v_region
from . import m4_curved_workflow as m4
from . import uccnc_m5 as shared
from .grbl_m5_reader import decode_job


FORMAT = "m5-portability-v1"
PROFILE = "grbl-v1.1-absolute-mm-g0g1-m0-v1"
STARTUP = ("G21", "G90", "G17", "G94", "G61", "G40", "G49", "G54")
CAM_SHIFT = (0, 0, 4.5)
WORK_MAP = (0, 0, -4.5)
TABLE = {"T1": 2.0, "T3": 3.0}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _shifted(moves, delta):
    return tuple(v_region.VMotion(m.role, shared._shift(m.start, delta),
                                   shared._shift(m.end, delta)) for m in moves)


def _tail(start):
    away = (start[0] - 1, start[1], start[2])
    return (v_region.VMotion("rapid", start, away),
            v_region.VMotion("rapid", away, start))


def _stage_lines(moves, tool, *, offset, translation=(0, 0, 0)):
    if not moves or tool not in TABLE or moves[-1].end != moves[0].start:
        raise ValueError("invalid Grbl stage or safe handoff")
    lines = [f"( STAGE {tool} )",
             "G49" if offset == 0 else f"G43.1 Z{shared._number(offset)}",
             f"M3 S{shared.RPM}"]
    at = moves[0].start
    for move in moves:
        if move.start != at or move.role not in ("rapid", "entry", "cut", "retract"):
            raise ValueError("noncontinuous Grbl stage")
        g = "G0" if move.role == "rapid" else "G1"
        feed = "" if g == "G0" else f" F{shared.ENTRY_FEED if move.role == 'entry' else shared.CUT_FEED}"
        end = shared._shift(move.end, translation)
        lines.append(g + feed + " " + " ".join(
            axis + shared._number(value) for axis, value in zip("XYZ", end)))
        at = move.end
    lines.append("M5")
    return lines


def render_job(prior, plan, start, *, mixed=False):
    """Emit one Grbl program with explicit pause and offset state per stage."""
    t1 = shared._prior_as_vmotion(prior)
    t3 = v_region.complete_motion(plan, start)
    lines = ["( M5 GRBL v1.1 PORTABILITY 1 )", *STARTUP]
    if mixed:
        lines += _stage_lines(t1, "T1", offset=TABLE["T1"])
        lines += ["M0"]
        lines += _stage_lines(t3, "T3", offset=TABLE["T3"])
        lines += ["M0"]
        lines += _stage_lines(_tail(start), "T1", offset=TABLE["T1"])
    else:
        # A distinct resolved CAM datum has its surface at +4.5 mm; the
        # declared work datum puts that same physical surface at zero.
        lines += _stage_lines(_shifted(t1, CAM_SHIFT), "T1", offset=0,
                              translation=WORK_MAP)
        lines += ["M0"]
        lines += _stage_lines(_shifted(t3, CAM_SHIFT), "T3", offset=0,
                              translation=WORK_MAP)
    return ("\n".join((*lines, "M30")) + "\n").encode("ascii")


def _changer_model(start):
    return {
        "format": "m5-synthetic-changer-effect-v1",
        "old_tool": "T3", "new_tool": "T1",
        "spindle_stopped": True,
        "completion_assumed_not_observed": True,
        "length_offset_before_mm": TABLE["T3"],
        "length_offset_after_mm": TABLE["T1"],
        "travel_work_tip_xyz_mm": [list(start),
                                    [start[0] - 3, start[1], start[2]],
                                    [start[0] - 3, start[1] - 2, start[2]],
                                    list(start)],
    }


def decode_changer(data, *, start):
    """Read all synthetic host-effect bytes and check no stock is contacted."""
    try:
        effect = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid changer effect stream") from exc
    if effect != _changer_model(start):
        raise ValueError("unknown or changed changer effect")
    if data != (json.dumps(_changer_model(start), indent=2) + "\n").encode("utf-8"):
        raise ValueError("changed changer effect encoding")
    points = effect["travel_work_tip_xyz_mm"]
    if (len(points) < 3 or points[0] != list(start) or
            points[-1] != list(start) or
            any(len(p) != 3 or any(type(v) not in (int, float) or
                not math.isfinite(v) for v in p) or p[2] < 5 for p in points)
            or any(a == b for a, b in zip(points, points[1:]))):
        raise ValueError("changer travel or safe return invalid")
    return effect


def _audit_safe_tail(prior, decoded, start):
    expected = _tail(start)
    actual = shared.compare_motion(decoded, expected, initial_cam_tip=start)
    operation = prior.operations[0]
    motions = tuple(replay.Motion("rapid", "T1", operation.name,
                                  move.start, move.end) for move in actual)
    trace = replay.Trace(prior.source_fingerprint, prior.frame, start,
                         prior.operations,
                         (replay.Event("tool_change", "T1", start),
                          replay.Event("spindle_start", "T1", start),
                          *motions, replay.Event("spindle_stop", "T1", start)))
    if replay.replay(trace, expected_source=prior.source_fingerprint).cuts:
        raise ValueError("changer return stage unexpectedly cuts stock")
    return len(actual)


def _audit_program(data, plan, prior, start, *, mixed, effect=None):
    decoded = decode_job(data, initial_tip=start)
    tools = tuple(stage.tool_label for stage in decoded.stages)
    expected_tools = ("T1", "T3", "T1") if mixed else ("T1", "T3")
    expected_offsets = (2.0, 3.0, 2.0) if mixed else (0.0, 0.0)
    if (tools != expected_tools or decoded.length_offsets_mm != expected_offsets
            or decoded.stops != tuple(range(len(expected_tools) - 1))):
        raise ValueError("Grbl tool order, pause or length state changed")
    if any(not shared._near(stage.end_position, start)
           for stage in decoded.stages):
        raise ValueError("Grbl pause or end lacks safe return")
    if mixed:
        if effect is None:
            raise ValueError("automatic changer effect missing")
        changer = decode_changer(effect, start=start)
        tail_moves = _audit_safe_tail(prior, decoded.stages[2], start)
        report = shared.audit_decoded_pair(
            plan, prior, start, decoded.stages[0], decoded.stages[1])
        report["motion_equivalence"]["return_t1_moves"] = tail_moves
        report["transition_evidence"] = {
            "status": "pass_with_asserted_external_effects",
            "manual": "operator pause at first M0; install T3, fixed G54, table Z=3",
            "automatic": "second M0; synthetic host T3-to-T1 effect decoded",
            "changer_travel_segments": len(changer["travel_work_tip_xyz_mm"]) - 1,
            "length_offsets_mm": list(expected_offsets),
            "completion": "assumed_not_observed"}
    else:
        if effect is not None:
            raise ValueError("manual fixture has unexpected changer effect")
        # Check the explicitly resolved +4.5 CAM path against emitted work
        # coordinates, then normalize to the accepted M4 frame for stock replay.
        shared.compare_motion(decoded.stages[0],
            _shifted(shared._prior_as_vmotion(prior), CAM_SHIFT),
            initial_cam_tip=shared._shift(start, CAM_SHIFT),
            translation_xyz_mm=WORK_MAP)
        shared.compare_motion(decoded.stages[1],
            _shifted(v_region.complete_motion(plan, start), CAM_SHIFT),
            initial_cam_tip=shared._shift(start, CAM_SHIFT),
            translation_xyz_mm=WORK_MAP)
        report = shared.audit_decoded_pair(
            plan, prior, start, decoded.stages[0], decoded.stages[1])
        report["transition_evidence"] = {
            "status": "pass_with_operator_preconditions",
            "manual": "first M0; operator installs and registers T3 tip",
            "cam_surface_z_mm": 4.5, "work_surface_z_mm": 0,
            "cam_to_work_z_mm": -4.5,
            "completion": "assumed_not_observed"}
    report["runtime_parity"] = {
        "status": "not_evaluated", "reason": "no Grbl interpreter trace"}
    report["physical_setup"] = {
        "status": "not_evaluated", "reason": "tool and host effects are setup assertions"}
    report["motion_equivalence"]["scope"] = "complete decoded Grbl program and transitions"
    return report


def _manifest(m4_path, plan, prior, start, programs):
    root = Path(m4_path).resolve().parent
    return {
        "format": FORMAT,
        "profile": {"name": PROFILE, "controller": "Grbl v1.1",
                    "subset": "G21 G90 G17 G94 G61 G40 G49 G54 G43.1 G0 G1 M0 M3 M5 M30",
                    "settings": {"$32_laser_mode": 0, "work_offset": "fixed G54",
                                 "spindle_rpm_range_contains": 12000},
                    "transport": "pause at M0; sender must wait for declared completion before resume"},
        "source": {"m4_manifest": str(Path(m4_path).resolve()),
                   "m4_manifest_sha256": _sha(Path(m4_path).read_bytes()),
                   "native_sha256": _sha((root / "source.cb").read_bytes()),
                   "prior_sha256": _sha((root / "prior.json").read_bytes()),
                   "plan_fingerprint": plan.fingerprint,
                   "prior_motion_fingerprint": prior.motion_fingerprint},
        "setup": {"initial_tip_work_xyz_mm": list(start),
                  "stock": {"bounds_xy_mm": [-12, -12, 12, 12],
                            "top_z_mm": 0, "thickness_mm": 4},
                  "manual": {"cam_surface_z_mm": 4.5, "work_surface_z_mm": 0,
                             "cam_to_work_translation_xyz_mm": list(WORK_MAP),
                             "registration": "each tool touched to surface; G49"},
                  "mixed": {"work_origin": "fixed G54 for all tools",
                            "tool_table_lengths_mm": TABLE,
                            "registration": "external table values applied with G43.1; no per-tool G54 touch-off"}},
        "transitions": {
            "manual": [{"old_tool": "T1", "new_tool": "T3",
                        "actor": "operator", "boundary": "in_program_M0",
                        "offset_method": "surface_touch_off", "resume": "declared tip and spindle restart"}],
            "mixed": [{"old_tool": "T1", "new_tool": "T3",
                       "actor": "operator", "boundary": "in_program_M0",
                       "offset_method": "fixed_origin_table", "resume": "declared tip and spindle restart"},
                      {"old_tool": "T3", "new_tool": "T1",
                       "actor": "synthetic_host_changer", "boundary": "in_program_M0",
                       "offset_method": "fixed_origin_table", "resume": "declared tip and spindle restart",
                       "effect_file": "mixed-changer.json"}],
            "abort": "do not resume after any missing tool, offset, safe-return or completion condition"},
        "programs": programs,
    }


def _read_files(manifest, root):
    expected = (("manual.nc", "manual"), ("mixed.nc", "mixed"),
                ("mixed-changer.json", "changer_effect"))
    if tuple((r.get("file"), r.get("role")) for r in manifest["programs"]) != expected:
        raise ValueError("M5 portability file order changed")
    data = []
    for row in manifest["programs"]:
        try:
            contents = (root / row["file"]).read_bytes()
        except FileNotFoundError as exc:
            raise ValueError("M5 portability file missing") from exc
        if _sha(contents) != row["sha256"]:
            raise ValueError("M5 portability file altered")
        data.append(contents)
    return data


def _evidence(manifest, root, plan, prior, start):
    manual, mixed, effect = _read_files(manifest, root)
    return {"manual": _audit_program(manual, plan, prior, start, mixed=False),
            "mixed": _audit_program(mixed, plan, prior, start, mixed=True,
                                    effect=effect)}


def build_bundle(directory, *, m4_manifest):
    directory = Path(directory)
    if directory.exists():
        raise ValueError("M5 portability output directory must be new")
    m4.audit_direct(m4_manifest, "raster")
    plan, prior, start = m4.load_selected_plan(m4_manifest, "raster")
    files = {"manual.nc": render_job(prior, plan, start),
             "mixed.nc": render_job(prior, plan, start, mixed=True),
             "mixed-changer.json": (json.dumps(_changer_model(start), indent=2) + "\n").encode("utf-8")}
    directory.mkdir(parents=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    programs = [{"file": name, "role": role, "sha256": _sha(files[name])}
                for name, role in (("manual.nc", "manual"),
                                   ("mixed.nc", "mixed"),
                                   ("mixed-changer.json", "changer_effect"))]
    manifest = _manifest(m4_manifest, plan, prior, start, programs)
    manifest["evidence"] = json.loads(json.dumps(
        _evidence(manifest, directory, plan, prior, start)))
    path = directory / "handoff.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return audit_bundle(path)


def audit_bundle(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("not an M5 portability handoff")
    source = manifest["source"]
    m4_path = Path(source["m4_manifest"])
    if (_sha(m4_path.read_bytes()) != source["m4_manifest_sha256"] or
            _sha((m4_path.parent / "source.cb").read_bytes()) != source["native_sha256"] or
            _sha((m4_path.parent / "prior.json").read_bytes()) != source["prior_sha256"]):
        raise ValueError("M5 portability source or prior changed")
    m4.audit_direct(m4_path, "raster")
    plan, prior, start = m4.load_selected_plan(m4_path, "raster")
    expected = _manifest(m4_path, plan, prior, start, manifest["programs"])
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"M5 portability {key} setup or lineage changed")
    evidence = json.loads(json.dumps(_evidence(manifest, path.parent, plan,
                                                prior, start)))
    if manifest.get("evidence") != evidence:
        raise ValueError("M5 portability decoded evidence changed")
    return {"status": "bounded_m5_portability_pass",
            "manifest_sha256": _sha(path.read_bytes()), **evidence}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit Grbl M5 portability fixtures")
    parser.add_argument("path", help="new output directory or handoff.json")
    parser.add_argument("--m4-manifest", help="accepted M4 comparison.json when building")
    args = parser.parse_args()
    if args.path.lower().endswith(".json"):
        result = audit_bundle(args.path)
    else:
        if args.m4_manifest is None:
            parser.error("--m4-manifest is required to build")
        result = build_bundle(args.path, m4_manifest=args.m4_manifest)
    print(json.dumps(result, indent=2))
