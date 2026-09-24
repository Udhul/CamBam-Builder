"""Headless reference-dialect output for the nominal two-tool RC01 job.

The file assumes the declared initial tip position. It is not a controller
profile or evidence of a physical machine setup.
"""

from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

from ..cam_core import rc01
from .cambam.rc01_post import read_default_post


PROGRAM_NAME = "direct-RC01.nc"
MANIFEST_NAME = "direct-evidence.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _decimal(value):
    """Render a rational coordinate exactly or reject it."""
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


def render(program):
    """Emit the verified nominal RC01 trace in a strict absolute-mm dialect."""
    rc01.verify(program, measure_rest=False)
    return _render_verified(program)


def _render_verified(program):
    lines = [
        "( bounded RC01 direct reference )",
        "( initial tip assumed X-10 Y-10 Z5 )",
        "G21 G90 G61 G40 G17",
    ]
    for item in program.items:
        if isinstance(item, rc01.Move):
            word = "G0" if item.role == "rapid" else "G1"
            feed = "" if item.role == "rapid" else f" F{item.feed}"
            xyz = " ".join(f"{axis}{_decimal(value)}"
                           for axis, value in zip("XYZ", item.end))
            lines.append(f"{word}{feed} {xyz}")
        elif item.kind == "tool_change":
            lines.append(f"{item.tool} M6")
        elif item.kind == "spindle_start":
            lines.append(f"M3 S{item.rpm}")
        elif item.kind == "spindle_stop":
            lines.append("M5")
        else:
            raise ValueError("unsupported RC01 process event")
    lines.append("M30")
    return "\n".join(lines) + "\n"


def _parsed_program(text, expected):
    actual, warnings = read_default_post(text)
    if warnings:
        raise ValueError(f"RC01 reference parser warning: {warnings[0]}")
    if len(actual) != len(expected.items):
        raise ValueError("RC01 reference has extra or missing items")
    observed = []
    for index, (want, got) in enumerate(zip(expected.items, actual)):
        if isinstance(want, rc01.Move):
            if got["type"] != "move":
                raise ValueError(f"RC01 reference move {index} differs")
            start = tuple(Q(str(value)) for value in got["start"])
            end = tuple(Q(str(value)) for value in got["end"])
            if (got["tool"] != want.tool or
                    got["g"] != (0 if want.role == "rapid" else 1) or
                    got["feed"] != want.feed or start != want.start or
                    end != want.end):
                raise ValueError(f"RC01 reference move {index} differs")
            observed.append(rc01.Move(want.role, got["tool"], start, end,
                                      want.feed, want.operation))
        else:
            if got["type"] != "event":
                raise ValueError(f"RC01 reference event {index} differs")
            position = tuple(Q(str(value)) for value in got["position"])
            if (got["kind"] != want.kind or
                    got["tool"] != want.tool or got["rpm"] != want.rpm or
                    position != want.position):
                raise ValueError(f"RC01 reference event {index} differs")
            observed.append(rc01.Event(want.kind, got["tool"], position,
                                       want.operation, want.rpm, want.direction))
    return rc01.Program(expected.job_fingerprint, tuple(observed), expected.frame)


def _audit_text(text, job, expected):
    if text != _render_verified(expected):
        raise ValueError("RC01 reference bytes differ from verified rendering")
    parsed = _parsed_program(text, expected)
    certificate = rc01.verify(parsed, job)
    if parsed.motion_fingerprint != expected.motion_fingerprint:
        raise ValueError("RC01 parsed motion fingerprint differs")
    return {
        "status": "bounded_direct_rc01_pass",
        "item_count": len(parsed.items),
        "move_count": certificate.moves,
        "completion": certificate.status,
        "rough_rest_by_depth_mm2": [list(area) for area in certificate.rough_rest_by_depth],
        "final_rest_by_depth_mm2": [list(area) for area in certificate.final_rest_by_depth],
        "rough_rest_volume_mm3": list(certificate.rough_rest_volume),
        "final_rest_volume_mm3": list(certificate.final_rest_volume),
        "area_coordinate_enclosure_mm": certificate.area_coordinate_enclosure_mm,
        "location_polygon_sagitta_mm": certificate.location_polygon_sagitta_mm,
        "location_numeric_enclosure_mm": certificate.location_numeric_enclosure_mm,
        "initial_position_assumption": "tip begins at (-10,-10,+5)",
        "scope": "strict mm reference dialect; no controller or physical acceptance",
    }


def build_program(directory):
    """Write, reparse and verify one nominal RC01 reference program."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("RC01 reference directory must be new or empty")
    job = rc01.Job()
    expected = rc01.generate(job)
    text = render(expected)
    directory.mkdir(parents=True, exist_ok=True)
    program_path = directory / PROGRAM_NAME
    program_path.write_bytes(text.encode("ascii"))
    result = _audit_text(program_path.read_text(encoding="ascii"), job, expected)
    manifest = {
        "format": "direct-rc01-v1",
        "program": PROGRAM_NAME,
        "program_sha256": _sha(program_path),
        "job_fingerprint": job.fingerprint,
        "motion_fingerprint": expected.motion_fingerprint,
        "result": result,
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    return {"program": str(program_path),
            "program_sha256": manifest["program_sha256"], **result}


def audit_program(manifest_path):
    """Recheck exact bytes and replay the parsed output against nominal RC01."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "direct-rc01-v1" or
            manifest.get("program") != PROGRAM_NAME):
        raise ValueError("not a bounded RC01 reference manifest")
    program_path = manifest_path.parent / PROGRAM_NAME
    if _sha(program_path) != manifest["program_sha256"]:
        raise ValueError("RC01 reference program changed")
    job = rc01.Job()
    expected = rc01.generate(job)
    if (manifest["job_fingerprint"] != job.fingerprint or
            manifest["motion_fingerprint"] != expected.motion_fingerprint):
        raise ValueError("RC01 reference manifest is stale")
    result = _audit_text(program_path.read_text(encoding="ascii"), job, expected)
    if result != manifest["result"]:
        raise ValueError("RC01 reference evidence differs from manifest")
    return {"program_sha256": manifest["program_sha256"], **result}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit bounded direct RC01")
    parser.add_argument("path", help="new directory or direct-evidence.json")
    args = parser.parse_args()
    result = (audit_program(args.path) if args.path.endswith(".json") else
              build_program(args.path))
    print(json.dumps(result, indent=2))
