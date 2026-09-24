"""Bounded headless G-code output for the verified variable-depth V trace.

This is a strict millimetre G0/G1 reference dialect, not a controller profile.
The caller still owns the declared initial tip position and physical setup.
"""

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

from ..cam_core import replay, tapered_vcarve
from .cambam.rc01_post import read_default_post


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("direct V output requires a finite numeric value")
    number = Decimal(str(value))
    if number.as_tuple().exponent < -17 or abs(number) > 1_000_000:
        raise ValueError("direct V value exceeds reference-dialect precision")
    if number == 0:
        return "0"
    rendered = format(number, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def render(plan):
    """Lower a verified straight variable-depth V plan to reference G-code."""
    trace = tapered_vcarve.output_trace(plan)
    replay.replay(trace, expected_source=plan.fingerprint)
    lines = [
        "( bounded variable-depth V direct reference )",
        "( initial tip assumed X-10 Y-10 Z5 )",
        "G21 G90 G61 G40 G17",
        "T3 M6",
        f"M3 S{tapered_vcarve.OUTPUT_RPM}",
    ]
    for item in trace.items[2:-1]:
        if type(item) is not replay.Motion:
            raise ValueError("unsupported direct V event sequence")
        word = "G0" if item.role == "rapid" else "G1"
        feed = "" if item.role == "rapid" else f" F{_number(item.feed)}"
        xyz = " ".join(f"{axis}{_number(value)}"
                       for axis, value in zip("XYZ", item.end))
        lines.append(f"{word}{feed} {xyz}")
    lines.extend(("M5", "M30"))
    return "\n".join(lines) + "\n"


def _parsed_items(text):
    items, warnings = read_default_post(text)
    if warnings:
        raise ValueError(f"direct V post warning: {warnings[0]}")
    return items


def _compare(trace, actual):
    if len(actual) != len(trace.items):
        raise ValueError("direct V output has extra or missing events/moves")
    observed = []
    for index, (want, got) in enumerate(zip(trace.items, actual)):
        if type(want) is replay.Motion:
            if (got["type"] != "move" or got["tool"] != want.tool or
                    got["g"] != (0 if want.role == "rapid" else 1) or
                    got["feed"] != want.feed or
                    tuple(got["start"]) != want.start or
                    tuple(got["end"]) != want.end):
                raise ValueError(f"direct V motion item {index} differs")
            observed.append(replace(want, start=tuple(got["start"]),
                                    end=tuple(got["end"]), feed=got["feed"]))
        else:
            if (got["type"] != "event" or got["kind"] != want.kind or
                    got["tool"] != want.tool or
                    tuple(got["position"]) != want.position or
                    (want.kind == "spindle_start" and
                     got["rpm"] != tapered_vcarve.OUTPUT_RPM)):
                raise ValueError(f"direct V event item {index} differs")
            observed.append(replace(want, position=tuple(got["position"])))
    return replace(trace, items=tuple(observed))


def _semantic_items(items):
    result = []
    for item in items:
        if item["type"] == "move":
            result.append(("move", item["tool"], item["g"], item["feed"],
                           tuple(item["start"]), tuple(item["end"])))
        else:
            result.append(("event", item["kind"], item["tool"],
                           tuple(item["position"]), item["rpm"]))
    return tuple(result)


def _cambam_items(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if "( Post processor: Default )" not in text:
        raise ValueError("comparison post is not CamBam Default")
    text = "\n".join("" if line.strip() in ("G98", "G80") else line
                     for line in text.splitlines())
    items, warnings = read_default_post(text)
    warnings = [warning for warning in warnings
                if "initial machine position is not encoded" not in warning]
    if warnings:
        raise ValueError(f"comparison post warning: {warnings[0]}")
    return items


def _audit_text(text, plan, *, comparison_post=None):
    if text != render(plan):
        raise ValueError("direct V output bytes differ from verified rendering")
    trace = tapered_vcarve.output_trace(plan)
    items = _parsed_items(text)
    observed = _compare(trace, items)
    stock = replay.replay(observed, expected_source=plan.fingerprint)
    result = tapered_vcarve.TaperedResult(plan, stock)
    first_depth, last_depth = plan.target_spine[3:5]
    span = last_depth - first_depth
    depths = (0, first_depth, first_depth + span / 3,
              first_depth + 2 * span / 3, last_depth)
    if len(set(depths)) != 5:
        raise ValueError("direct V target cannot resolve five section depths")
    areas = {f"depth_{_number(depth)}": result.residual_area(depth)
             for depth in depths}
    if comparison_post is not None and _semantic_items(items) != _semantic_items(
            _cambam_items(comparison_post)):
        raise ValueError("direct V semantics differ from accepted CamBam post")
    return {"status": "bounded_direct_variable_v_pass",
            "item_count": len(items),
            "stock_prefixes": [list(prefix) for prefix in stock.prefixes],
            "completion": result.completion, "section_rest_mm2": areas,
            "matches_cambam_post": comparison_post is not None,
            "initial_position_assumption": "tip begins at (-10,-10,+5)",
            "scope": "strict mm reference dialect; no controller or physical acceptance"}


def build_program(directory, *, source_path=None, setup=None,
                  comparison_post=None):
    """Generate, reparse and replay one direct program from a detached request."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("direct V output directory must be new or empty")
    if source_path is None:
        if setup is not None:
            raise ValueError("setup requires a native source")
        request = tapered_vcarve.standalone_request()
        source_bytes = None
    else:
        if setup is None:
            raise ValueError("native direct V input requires explicit setup")
        from .cambam.native_variable_v import normalize_bytes
        source_bytes = Path(source_path).read_bytes()
        request = normalize_bytes(source_bytes, setup)
    plan = tapered_vcarve.generate(request)
    text = render(plan)
    result = _audit_text(text, plan, comparison_post=comparison_post)
    directory.mkdir(parents=True, exist_ok=True)
    if source_bytes is not None:
        (directory / "source.cb").write_bytes(source_bytes)
        (directory / "setup.json").write_text(
            json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    program = directory / "direct-V-variable.nc"
    program.write_bytes(text.encode("ascii"))
    manifest = {
        "format": "direct-variable-v-v1",
        "program": program.name, "program_sha256": _sha(program),
        "source_sha256": _sha(directory / "source.cb") if source_bytes else None,
        "setup_sha256": _sha(directory / "setup.json") if source_bytes else None,
        "comparison_post_sha256": _sha(comparison_post) if comparison_post else None,
        "plan_fingerprint": plan.fingerprint,
        "motion_fingerprint": tapered_vcarve.output_trace(plan).motion_fingerprint,
        "result": result,
    }
    (directory / "direct-evidence.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"program": str(program), "program_sha256": manifest["program_sha256"],
            **result}


def audit_program(manifest_path, *, comparison_post=None):
    """Recheck exact program/source bytes, parsed roles and posted-coordinate rest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "direct-variable-v-v1":
        raise ValueError("not a bounded direct V manifest")
    directory = manifest_path.parent
    program = directory / manifest["program"]
    if _sha(program) != manifest["program_sha256"]:
        raise ValueError("direct V program changed")
    if manifest["source_sha256"] is not None:
        if (_sha(directory / "source.cb") != manifest["source_sha256"] or
                _sha(directory / "setup.json") != manifest["setup_sha256"]):
            raise ValueError("direct V source or setup changed")
        from .cambam.native_variable_v import normalize_bytes
        setup = json.loads((directory / "setup.json").read_text(encoding="utf-8"))
        request = normalize_bytes((directory / "source.cb").read_bytes(), setup)
    else:
        request = tapered_vcarve.standalone_request()
    plan = tapered_vcarve.generate(request)
    if (plan.fingerprint != manifest["plan_fingerprint"] or
            tapered_vcarve.output_trace(plan).motion_fingerprint !=
            manifest["motion_fingerprint"]):
        raise ValueError("direct V manifest is stale")
    if comparison_post is not None and (
            _sha(comparison_post) != manifest["comparison_post_sha256"]):
        raise ValueError("comparison CamBam post changed")
    if comparison_post is None and manifest["comparison_post_sha256"] is not None:
        raise ValueError("accepted comparison post is required for this manifest")
    result = _audit_text(program.read_text(encoding="ascii"), plan,
                         comparison_post=comparison_post)
    if result != manifest["result"]:
        raise ValueError("direct V evidence differs from manifest")
    return {"program_sha256": manifest["program_sha256"], **result}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit direct bounded V")
    parser.add_argument("path", help="new directory or direct-evidence.json")
    parser.add_argument("--source", help="native source.cb to normalize")
    parser.add_argument("--setup", help="explicit native setup.json")
    parser.add_argument("--compare", help="accepted CamBam Default NC")
    args = parser.parse_args()
    if args.path.endswith(".json"):
        result = audit_program(args.path, comparison_post=args.compare)
    else:
        setup = json.loads(Path(args.setup).read_text(encoding="utf-8")) if args.setup else None
        result = build_program(args.path, source_path=args.source, setup=setup,
                               comparison_post=args.compare)
    print(json.dumps(result, indent=2))
