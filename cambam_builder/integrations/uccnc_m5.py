"""Bounded M5 UCCNC split-file output and decoded whole-job audit.

The files assume each tool is installed and its tip registered before that
file starts. They contain no M6 or pause macro. Physical completion remains
an external condition, never an observation inferred from NC text.
"""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import Point

from ..cam_core import replay, v_region
from . import m4_curved_workflow as m4
from .uccnc_reader import decode_program


FORMAT = "m5-uccnc-split-v1"
PROFILE = "uccnc-absolute-mm-g0g1-v1"
TOLERANCE_MM = 0.000051
ROUND_DECIMALS = 4
RPM = 12000
ENTRY_FEED = 60
CUT_FEED = 300


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _number(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("finite UCCNC number required")
    value = round(float(value), ROUND_DECIMALS)
    if abs(value) > 1_000_000:
        raise ValueError("UCCNC coordinate exceeds bounded profile")
    return format(value, ".4f").rstrip("0").rstrip(".") if value else "0"


def _shift(point, translation):
    return tuple(a + b for a, b in zip(point, translation))


def _unshift(point, translation):
    return tuple(round(a - b, 7) for a, b in zip(point, translation))


def _near(a, b):
    return all(abs(x - y) <= TOLERANCE_MM for x, y in zip(a, b))


def render_program(moves, *, tool="T1", translation_xyz_mm=(0, 0, 0)):
    """Lower one installed-tool motion list to the declared UCCNC subset."""
    if not moves:
        raise ValueError("empty controller stage")
    if tool not in ("T1", "T3"):
        raise ValueError("unsupported split-file tool label")
    lines = [f"( M5 UCCNC split-file v1 TOOL {tool} G54 )", "G21", "G90", "G17",
             "G61", "G40", "G49", "G54", f"M3 S{RPM}"]
    at = moves[0].start
    for move in moves:
        if move.start != at or move.role not in (
                "rapid", "entry", "cut", "retract"):
            raise ValueError("noncontinuous or unsupported controller motion")
        g = "G0" if move.role == "rapid" else "G1"
        feed = ("" if g == "G0" else
                f" F{ENTRY_FEED if move.role == 'entry' else CUT_FEED}")
        xyz = _shift(move.end, translation_xyz_mm)
        lines.append(g + feed + " " + " ".join(
            axis + _number(value) for axis, value in zip("XYZ", xyz)))
        at = move.end
    lines.extend(("M5", "M30"))
    return ("\n".join(lines) + "\n").encode("ascii")


def compare_motion(decoded, intended, *, initial_cam_tip,
                   translation_xyz_mm=(0, 0, 0)):
    """Check every decoded endpoint and role, then return actual CAM motion."""
    if len(decoded.moves) != len(intended):
        raise ValueError("UCCNC missing or extra motion")
    if decoded.rpm != RPM or not _near(
            _unshift(decoded.moves[0].start, translation_xyz_mm),
            initial_cam_tip):
        raise ValueError("UCCNC spindle or initial tip differs")
    actual = []
    for index, (got, want) in enumerate(zip(decoded.moves, intended)):
        feed = 0 if want.role == "rapid" else (
            ENTRY_FEED if want.role == "entry" else CUT_FEED)
        start = _unshift(got.start, translation_xyz_mm)
        end = _unshift(got.end, translation_xyz_mm)
        if (got.g != (0 if want.role == "rapid" else 1) or
                got.feed != feed or not _near(start, want.start) or
                not _near(end, want.end)):
            raise ValueError(f"UCCNC motion {index} differs from intended path")
        actual.append(v_region.VMotion(want.role, start, end))
    if not _near(actual[-1].end, initial_cam_tip):
        raise ValueError("UCCNC end tip differs from handoff position")
    return tuple(actual)


def _prior_moves(prior):
    if (tuple(type(item) for item in prior.items[:2]) !=
            (replay.Event, replay.Event) or
            type(prior.items[-1]) is not replay.Event or
            prior.items[0].kind != "tool_change" or
            prior.items[1].kind != "spindle_start" or
            prior.items[-1].kind != "spindle_stop" or
            any(type(item) is not replay.Motion for item in prior.items[2:-1])):
        raise ValueError("unsupported T1 prior sequence")
    return tuple(prior.items[2:-1])


def _prior_as_vmotion(prior):
    return tuple(v_region.VMotion(item.role, item.start, item.end)
                 for item in _prior_moves(prior))


def _decoded_prior(prior, actual):
    wanted = _prior_moves(prior)
    if len(actual) != len(wanted):
        raise ValueError("T1 decoded length changed")
    observed = tuple(replace(item, start=motion.start, end=motion.end,
                             feed=0 if item.role == "rapid" else
                             ENTRY_FEED if item.role == "entry" else CUT_FEED)
                     for item, motion in zip(wanted, actual))
    ending = actual[-1].end
    trace = replace(prior, items=(
        replay.Event("tool_change", "T1", prior.initial_position),
        replay.Event("spindle_start", "T1", prior.initial_position),
        *observed, replay.Event("spindle_stop", "T1", ending)))
    replay.replay(trace, expected_source=prior.source_fingerprint)
    return trace


def _decoded_v_plan(plan, actual, start):
    intended = v_region.complete_motion(plan, start)
    if len(actual) != len(intended):
        raise ValueError("T3 decoded length changed")
    prefix = int(intended[0].role == "rapid")
    middle = actual[prefix:prefix + len(plan.motions)]
    if len(middle) != len(plan.motions):
        raise ValueError("T3 decoded path incomplete")
    paths, points = [], None
    for move in middle:
        if move.role == "rapid":
            if points is not None:
                raise ValueError("T3 rapid inside low path")
        elif move.role == "entry":
            if points is not None:
                raise ValueError("T3 entry before retract")
            points = [(move.end[0], move.end[1], -move.end[2])]
        elif move.role == "cut":
            if points is None:
                raise ValueError("T3 cut without entry")
            points.append((move.end[0], move.end[1], -move.end[2]))
        elif move.role == "retract":
            if points is None or len(points) < 2:
                raise ValueError("T3 retract without cut")
            if len(paths) >= len(plan.paths):
                raise ValueError("T3 extra path")
            paths.append(v_region.VPath(plan.paths[len(paths)].role,
                                        tuple(points)))
            points = None
    if points is not None or len(paths) != len(plan.paths):
        raise ValueError("T3 path count changed")
    decoded_plan = replace(plan, paths=tuple(paths), motions=tuple(middle))
    v_region.verify(decoded_plan)
    if v_region._motions(decoded_plan.paths, decoded_plan.safe_z) != middle:
        raise ValueError("T3 decoded path has an unmodeled link")
    for path in decoded_plan.paths:
        x, y, depth = path.points[0]
        point = Point(x, y)
        if (not decoded_plan.target.safe.covers(point) or
                point.distance(decoded_plan.target.safe.boundary) +
                1e-8 < decoded_plan.tool.radius(depth) +
                decoded_plan.margin_mm * 0.5):
            raise ValueError("T3 entry cutter crosses protected boundary")
    return decoded_plan


def _result(plan, prior, start, t1_data, t3_data, translation):
    initial_work = _shift(start, translation)
    t1 = decode_program(t1_data, initial_tip=initial_work)
    t3 = decode_program(t3_data, initial_tip=initial_work)
    if (t1.tool_label, t3.tool_label) != ("T1", "T3"):
        raise ValueError("UCCNC installed-tool file identity or order changed")
    actual_t1 = compare_motion(t1, _prior_as_vmotion(prior),
                               initial_cam_tip=start,
                               translation_xyz_mm=translation)
    actual_t3 = compare_motion(t3, v_region.complete_motion(plan, start),
                               initial_cam_tip=start,
                               translation_xyz_mm=translation)
    decoded_prior = _decoded_prior(prior, actual_t1)
    decoded_plan = _decoded_v_plan(plan, actual_t3, start)
    rest = v_region.with_prior(decoded_plan, decoded_prior)
    prior_area = v_region.section_report(rest, 1, final=False)
    final_area = v_region.section_report(rest, 1)
    prior_volume = v_region.volume_bounds(rest, final=False)
    final_volume = v_region.volume_bounds(rest)
    if (final_area[2] >= 1e-7 or final_area[1] > 2 or
            final_volume[1] > 80):
        raise ValueError("decoded UCCNC stock or residual budget failed")
    return {
        "document_fidelity": {"status": "pass",
                              "scope": "M4 source hash and semantic plan"},
        "input_resolution": {"status": "pass",
                             "scope": "accepted M4 rounded raster and supplied T1"},
        "motion_equivalence": {"status": "pass",
                               "scope": "both complete UCCNC files",
                               "tolerance_mm": TOLERANCE_MM,
                               "t1_moves": len(actual_t1),
                               "t3_moves": len(actual_t3)},
        "stock_access_residual": {
            "status": "pass",
            "scope": "decoded T1 stock carried into decoded T3 rounded V paths",
            "t1_cuts": len(rest.prior_stock.cuts),
            "prior_section_1_mm2": prior_area,
            "final_section_1_mm2": final_area,
            "prior_volume_mm3": prior_volume,
            "final_volume_mm3": final_volume,
            "decoded_t1_fingerprint": decoded_prior.motion_fingerprint,
            "decoded_t3_plan_fingerprint": decoded_plan.fingerprint},
        "runtime_parity": {"status": "not_evaluated",
                           "reason": "no complete UCCNC interpreter trace"},
        "physical_setup": {"status": "not_evaluated",
                           "reason": "tool installation and touch-off are operator assertions"},
    }


def _expected_manifest(m4_path, plan, prior, start, programs):
    root = Path(m4_path).resolve().parent
    return {
        "format": FORMAT,
        "profile": {"name": PROFILE, "controller": "UCCNC",
                    "subset": "G21 G90 G17 G61 G40 G49 G54 G0 G1 M3 M5 M30",
                    "version": 1},
        "source": {"m4_manifest": str(Path(m4_path).resolve()),
                   "m4_manifest_sha256": _sha(Path(m4_path).read_bytes()),
                   "native_sha256": _sha((root / "source.cb").read_bytes()),
                   "prior_sha256": _sha((root / "prior.json").read_bytes()),
                   "plan_fingerprint": plan.fingerprint,
                   "prior_motion_fingerprint": prior.motion_fingerprint},
        "setup": {
            "program_frame": "accepted M4 CAM tip mm",
            "work_frame": "G54 tip mm",
            "cam_to_work_translation_xyz_mm": [0, 0, 0],
            "initial_tip_cam_xyz_mm": list(start),
            "initial_tip_work_xyz_mm": list(start),
            "work_xy_datum": "unchanged between T1 and T3",
            "work_z_registration": "operator touches each installed tool to physical stock top and sets work Z=0",
            "tool_length_compensation": "G49 inactive",
            "stock": {"bounds_xy_mm": [-12, -12, 12, 12],
                      "top_z_mm": 0, "thickness_mm": 4},
            "fixture": "synthetic M4 annulus; no fixture geometry certified",
            "tools": {
                "T1": {"kind": "cylinder", "radius_mm": 0.5,
                       "cutting_length_mm": 3},
                "T3": {"kind": "rounded V", "angle_degrees": 60,
                       "tip_radius_mm": 0.5, "maximum_radius_mm": 4,
                       "cutting_length_mm": 3}},
            "spindle_rpm": RPM, "entry_feed_mm_min": ENTRY_FEED,
            "cut_retract_feed_mm_min": CUT_FEED,
        },
        "handoff": {
            "order": ["T1", "T3"], "actor": "operator",
            "boundary": "separate programs", "stock_lineage": "decoded T1 to T3",
            "preconditions": [
                "T1 installed and tip registered to work Z=0 before T1.nc",
                "T3 installed and tip registered to work Z=0 before T3.nc",
                "same G54 XY datum, stock and fixture across both files",
                "tip at declared (-17,-17,+5) work position before each file"],
            "postconditions_assumed_not_observed": [
                "spindle stopped and tip returned to declared safe position",
                "operator confirms T1 completion before T3 starts"],
            "abort": "do not start next file if any setup or preceding file differs",
        },
        "numerical": {"coordinate_decimals": ROUND_DECIMALS,
                      "match_tolerance_mm": TOLERANCE_MM,
                      "section_depth_mm": 1,
                      "final_area_upper_limit_mm2": 2,
                      "final_volume_upper_limit_mm3": 80},
        "programs": programs,
    }


def _audit_files(manifest, root, plan, prior, start):
    programs = manifest["programs"]
    if ([row["tool"] for row in programs] != ["T1", "T3"] or
            [row["file"] for row in programs] != ["T1.nc", "T3.nc"]):
        raise ValueError("UCCNC file order or handoff changed")
    data = []
    for row in programs:
        path = root / row["file"]
        try:
            contents = path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError("UCCNC program missing or altered") from exc
        if _sha(contents) != row["sha256"]:
            raise ValueError("UCCNC program missing or altered")
        data.append(contents)
    return _result(plan, prior, start, *data, (0, 0, 0))


def build_bundle(directory, *, m4_manifest):
    """Emit and independently audit the accepted rounded-raster split job."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError("M5 output directory must be new")
    m4.audit_direct(m4_manifest, "raster")
    plan, prior, start = m4.load_selected_plan(m4_manifest, "raster")
    t1 = render_program(_prior_as_vmotion(prior))
    t3 = render_program(v_region.complete_motion(plan, start), tool="T3")
    directory.mkdir(parents=True)
    (directory / "T1.nc").write_bytes(t1)
    (directory / "T3.nc").write_bytes(t3)
    programs = [{"tool": "T1", "file": "T1.nc", "sha256": _sha(t1)},
                {"tool": "T3", "file": "T3.nc", "sha256": _sha(t3)}]
    manifest = _expected_manifest(m4_manifest, plan, prior, start, programs)
    manifest["evidence"] = json.loads(json.dumps(
        _audit_files(manifest, directory, plan, prior, start)))
    path = directory / "handoff.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return audit_bundle(path)


def audit_bundle(manifest_path):
    """Revalidate provenance, both final files and decoded chained stock."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("not an M5 UCCNC split handoff")
    source = manifest["source"]
    m4_path = Path(source["m4_manifest"])
    if (_sha(m4_path.read_bytes()) != source["m4_manifest_sha256"] or
            _sha((m4_path.parent / "source.cb").read_bytes()) !=
            source["native_sha256"] or
            _sha((m4_path.parent / "prior.json").read_bytes()) !=
            source["prior_sha256"]):
        raise ValueError("M5 source or prior changed")
    m4.audit_direct(m4_path, "raster")
    plan, prior, start = m4.load_selected_plan(m4_path, "raster")
    expected = _expected_manifest(m4_path, plan, prior, start,
                                  manifest["programs"])
    for key in expected:
        if manifest.get(key) != expected[key]:
            raise ValueError(f"M5 {key} setup or lineage changed")
    evidence = json.loads(json.dumps(
        _audit_files(manifest, path.parent, plan, prior, start)))
    if evidence != manifest.get("evidence"):
        raise ValueError("M5 decoded evidence differs from manifest")
    return {"status": "bounded_m5_uccnc_split_pass",
            "manifest_sha256": _sha(path.read_bytes()), **evidence}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit bounded UCCNC split output")
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
