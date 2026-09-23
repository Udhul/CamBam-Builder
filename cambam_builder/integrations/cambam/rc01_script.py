"""One RC01 CustomScript carrier; CamBam remains the native G-code emitter.

The script carries the already verified framework motion as literal G-code
inside a .cb Drill MOP.  Only the user's actual CamBam post is output evidence.
"""

from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

from ...cam_core.rc01 import Event, Job, Move, Program, generate, verify
from ...cam_entities import DrillMop
from ...cambam_reader import read_cambam_bytes
from .rc01_adapter import normalize, synthetic_setup, synthetic_source
from .rc01_post import read_default_post


def _decimal(value):
    """Render an exact terminating decimal; fail on a rounded cutter center."""
    value = Q(value)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("RC01 coordinate has no exact decimal representation")
    places = max(twos, fives)
    scaled = abs(value.numerator) * (10 ** places // value.denominator)
    whole, fraction = divmod(scaled, 10 ** places)
    sign = "-" if value < 0 else ""
    if not fraction:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{fraction:0{places}d}".rstrip("0")


def _script(program):
    """Use CamBam's T1 start and terminal M5; carry every intervening role."""
    lines = []
    for index, item in enumerate(program.items):
        if index in (0, 1, len(program.items) - 1):
            continue
        if isinstance(item, Move):
            x, y, z = (_decimal(v) for v in item.end)
            line = "G0" if item.role == "rapid" else "G1"
            if item.feed:
                line += f" F{item.feed}"
            lines.append(f"{line} X{x} Y{y} Z{z}")
        elif item.kind == "spindle_stop":
            lines.append("M5")
        elif item.kind == "tool_change":
            lines.append("T2 M6")
        elif item.kind == "spindle_start":
            lines.append(f"M3 S{item.rpm}")
        else:
            raise ValueError("unsupported RC01 event")
    # CamBam Plus 1.0 preserved literal '|' in a posted CustomScript. The
    # property's XML must contain real line breaks for separate NC blocks.
    return "\n".join(lines)


def build_script_carrier(directory):
    """Prepare the full T1/T2 .cb with one bounded literal-motion MOP."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("RC01 output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "source.cb"
    synthetic_source().save(str(source_path))
    setup = synthetic_setup()
    source = read_cambam_bytes(source_path.read_bytes(), source_name=str(source_path))
    job = normalize(source, setup)
    program = generate(job)
    certificate = verify(program, job)
    project = source.clone()
    layer = project.add_layer("RC01 literal motion carrier")
    part = project.list_parts()[0]
    point = project.add_points(layer, [(-10, -10, 0)],
                               identifier="rc01-script-anchor")
    script = _script(program)
    mop = project.add_drill_mop(
        part, targets=[point], name="RC01 T1 rough plus T2 cleanup literal motion",
        identifier="rc01-literal-motion", enabled=True,
        drilling_method="CustomScript", custom_script=script,
        tool_number=1, tool_diameter=6, tool_profile="EndMill",
        stock_surface=0, target_depth=-3, depth_increment=1,
        clearance_plane=5, plunge_feedrate=60, cut_feedrate=300,
        spindle_direction="CW", spindle_speed=12000,
        work_plane="XY", velocity_mode="ExactStop")
    if point is None or mop is None:
        raise RuntimeError("could not attach RC01 literal-motion carrier")
    candidate = directory / "S-combined.cb"
    project.save(str(candidate))
    reopened = read_cambam_bytes(candidate.read_bytes(), source_name=str(candidate))
    if normalize(reopened, setup, allow_attachments=True) != job:
        raise ValueError("RC01 source changed during literal-motion attachment")
    enabled = [m for m in reopened.list_mops() if m.enabled]
    if (len(enabled) != 1 or type(enabled[0]) is not DrillMop or
            enabled[0].drilling_method != "CustomScript" or
            enabled[0].custom_script != script or
            reopened.get_mop_targets(enabled[0]) != [point.internal_id] or
            any(m.enabled for m in reopened.list_mops()[:2])):
        raise ValueError("RC01 literal-motion carrier changed on reimport")
    manifest = {
        "format": "rc01-script-v1",
        "candidate": candidate.name,
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "job_fingerprint": job.fingerprint,
        "motion_fingerprint": program.motion_fingerprint,
        "item_count": len(program.items),
        "script_lines": script.count("\n") + 1,
        "postprocessor": "Default",
        "profile": "Default mm",
        "certificate_status": certificate.status,
        "emitted_motion_status": "pending_user_CamBam_post",
    }
    (directory / "setup.json").write_text(json.dumps(setup, indent=2) + "\n",
                                            encoding="utf-8")
    (directory / "comparison.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                 encoding="utf-8")
    return manifest


def audit_script_post(manifest_path, posted_path):
    """Compare every emitted item, then replay actual coordinates and stock."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "rc01-script-v1":
        raise ValueError("not an RC01 literal-motion manifest")
    directory = manifest_path.parent
    candidate = directory / manifest["candidate"]
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != manifest["candidate_sha256"]:
        raise ValueError("RC01 candidate changed before posting")
    if hashlib.sha256((directory / "source.cb").read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("RC01 source changed before posting")
    setup = json.loads((directory / "setup.json").read_text(encoding="utf-8"))
    job = normalize(read_cambam_bytes(candidate.read_bytes()), setup,
                    allow_attachments=True)
    expected = generate(job)
    if (job.fingerprint != manifest["job_fingerprint"] or
            expected.motion_fingerprint != manifest["motion_fingerprint"]):
        raise ValueError("RC01 motion manifest is stale")
    post = Path(posted_path).read_text(encoding="utf-8-sig")
    if ("( Post processor: Default )" not in post or
            not any(line.startswith(f"( {candidate.stem} ")
                    for line in post.splitlines()[:6])):
        raise ValueError("CamBam post header does not match candidate/Default")
    # CustomScript Drill contributes G98/G80 around its literal text. They
    # have no movement without a canned cycle; reject combined/other forms.
    normalized = "\n".join("" if line.strip() in ("G98", "G80") else line
                           for line in post.splitlines())
    actual, warnings = read_default_post(normalized)
    warnings = [w for w in warnings if "initial machine position is not encoded" not in w]
    if warnings:
        return {"status": "unverified", "reason": warnings[0]}
    # CamBam's footer can emit one redundant M5 after the script's T2 stop.
    if (len(actual) == len(expected.items) + 1 and
            actual[-1]["type"] == "event" and actual[-1]["kind"] == "spindle_stop" and
            actual[-1]["tool"] == "T2" and tuple(actual[-1]["position"]) == (-10, -10, 5)):
        actual = actual[:-1]
    if len(actual) != len(expected.items):
        return {"status": "deviation", "reason": "extra or missing emitted item",
                "expected_items": len(expected.items), "actual_items": len(actual)}
    replay = []
    for index, (want, got) in enumerate(zip(expected.items, actual)):
        if isinstance(want, Move):
            if (got["type"] != "move" or got["tool"] != want.tool or
                    got["g"] != (0 if want.role == "rapid" else 1) or
                    got["feed"] != want.feed or
                    tuple(Q(str(v)) for v in got["start"]) != want.start or
                    tuple(Q(str(v)) for v in got["end"]) != want.end):
                return {"status": "deviation", "item": index,
                        "line": got["line"], "expected_role": want.role,
                        "actual": got}
            replay.append(Move(want.role, got["tool"],
                               tuple(Q(str(v)) for v in got["start"]),
                               tuple(Q(str(v)) for v in got["end"]),
                               got["feed"], want.operation))
        else:
            if (got["type"] != "event" or got["kind"] != want.kind or
                    got["tool"] != want.tool or got["rpm"] != want.rpm or
                    tuple(Q(str(v)) for v in got["position"]) != want.position):
                return {"status": "deviation", "item": index,
                        "line": got["line"], "expected_event": want.kind,
                        "actual": got}
            replay.append(Event(want.kind, got["tool"],
                                tuple(Q(str(v)) for v in got["position"]),
                                want.operation, want.rpm, want.direction))
    certificate = verify(Program(job.fingerprint, tuple(replay), job.frame), job)
    return {
        "status": "bounded_emitted_motion_pass",
        "candidate_sha256": manifest["candidate_sha256"],
        "post_sha256": hashlib.sha256(Path(posted_path).read_bytes()).hexdigest(),
        "item_count": len(actual),
        "certificate_status": certificate.status,
        "rough_rest_by_depth_mm2": certificate.rough_rest_by_depth,
        "final_rest_by_depth_mm2": certificate.final_rest_by_depth,
        "numeric_limit": "RC01 residual-location GEOS topology lacks formal interval proof",
        "initial_position_assumption": "tip begins at (-10,-10,+5)",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit RC01 script carrier")
    parser.add_argument("path", help="new output directory or comparison.json")
    parser.add_argument("post", nargs="?", help="CamBam-posted S-combined.nc")
    args = parser.parse_args()
    result = (audit_script_post(args.path, args.post) if args.post else
              build_script_carrier(args.path))
    print(json.dumps(result, indent=2))
