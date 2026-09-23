"""Fail-closed reader for RC01 CamBam Default-post A/B/C comparison.

This checks whether posted motion exactly carries the ordered framework prefix.
It cannot certify CamBam's native T2 Pocket or controller execution.
"""

import json
import hashlib
import re
from pathlib import Path


_WORD = re.compile(r"([A-Za-z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_COMMENT = re.compile(r"\([^()]*\)")
_ALLOWED_G = {0, 1, 17, 21, 40, 61, 64, 90}
_ALLOWED_M = {3, 5, 6, 30}


def read_default_post(data):
    """Decode known absolute-mm G0/G1 and T/M/F/S tokens; reject everything else."""
    if not isinstance(data, str) or len(data) > 20_000_000:
        raise ValueError("posted text must be bounded Unicode")
    at = (-10.0, -10.0, 5.0)
    tool, feed, rpm, motion = None, None, None, None
    units, absolute, plane = False, False, False
    items, warnings = [], []
    stopped = False
    for number, raw in enumerate(data.splitlines(), 1):
        line = _COMMENT.sub("", raw).strip()
        if not line or line == "%":
            continue
        if stopped:
            raise ValueError(f"line {number}: commands after M30")
        words = _WORD.findall(line)
        if "".join(a + b for a, b in words).replace(" ", "") != line.replace(" ", ""):
            raise ValueError(f"line {number}: unsupported command text")
        codes = [(a.upper(), float(b)) for a, b in words]
        for axis in "XYZFST":
            if sum(a == axis for a, _ in codes) > 1:
                raise ValueError(f"line {number}: repeated {axis} word")
        if any(a == "N" for a, _ in codes):
            codes = [(a, b) for a, b in codes if a != "N"]
        if any(a not in "GMXYZFST" for a, _ in codes):
            raise ValueError(f"line {number}: unsupported word")
        gs = [int(v) for a, v in codes if a == "G" and v.is_integer()]
        ms = [int(v) for a, v in codes if a == "M" and v.is_integer()]
        if (len(gs) != sum(a == "G" for a, _ in codes) or
                len(ms) != sum(a == "M" for a, _ in codes) or
                any(g not in _ALLOWED_G for g in gs) or
                any(m not in _ALLOWED_M for m in ms) or
                sum(g in (0, 1) for g in gs) > 1):
            raise ValueError(f"line {number}: unsupported G/M command")
        for g in gs:
            if g == 21:
                units = True
            elif g == 90:
                absolute = True
            elif g == 17:
                plane = True
            elif g in (0, 1):
                motion = g
            elif g == 64:
                warnings.append(f"line {number}: G64 blending needs trajectory evidence")
        values = {a: v for a, v in codes if a in "XYZFST"}
        if any(a in values for a in "XYZ"):
            if not (units and absolute and plane and motion is not None):
                raise ValueError(f"line {number}: motion before explicit G21/G90/G17")
            end = tuple(values.get(axis, at[i]) for i, axis in enumerate("XYZ"))
            if end != at:
                if tool is None:
                    raise ValueError(f"line {number}: motion before tool event")
                if motion == 1 and "F" not in values and feed is None:
                    raise ValueError(f"line {number}: feed move without feed")
                items.append({"type": "move", "tool": f"T{tool}",
                              "start": list(at), "end": list(end),
                              "feed": 0 if motion == 0 else values.get("F", feed),
                              "g": motion, "line": number})
            at = end
        if "F" in values:
            feed = values["F"]
        if "S" in values:
            rpm = values["S"]
        if "T" in values:
            if not values["T"].is_integer():
                raise ValueError(f"line {number}: noninteger tool")
            tool = int(values["T"])
        for m in ms:
            if m == 6:
                if "T" not in values or at != (-10.0, -10.0, 5.0):
                    raise ValueError(f"line {number}: tool change state")
                items.append({"type": "event", "kind": "tool_change",
                              "tool": f"T{tool}", "position": list(at),
                              "rpm": 0, "line": number})
            elif m == 3:
                if tool is None or rpm is None or at != (-10.0, -10.0, 5.0):
                    raise ValueError(f"line {number}: spindle start state")
                items.append({"type": "event", "kind": "spindle_start",
                              "tool": f"T{tool}", "position": list(at),
                              "rpm": rpm, "line": number})
            elif m == 5:
                if tool is None:
                    raise ValueError(f"line {number}: spindle stop without tool")
                items.append({"type": "event", "kind": "spindle_stop",
                              "tool": f"T{tool}", "position": list(at),
                              "rpm": 0, "line": number})
            else:
                stopped = True
    if not stopped:
        raise ValueError("posted file lacks M30 end")
    return items, warnings


def compare_posted(manifest, variant, posted_text):
    """Report first exact-sequence deviation; a clean C prefix still leaves N open."""
    if variant not in ("A", "B", "C") or manifest.get("format") != "rc01-comparison-v1":
        raise ValueError("unknown RC01 comparison variant/manifest")
    expected = manifest["expected_items"]
    if variant != "B":
        expected = expected[:manifest["rough_item_count"]]
    try:
        actual, warnings = read_default_post(posted_text)
    except ValueError as exc:
        return {"variant": variant, "status": "unverified",
                "reason": str(exc), "matched_items": 0}
    for index, (want, got) in enumerate(zip(expected, actual)):
        keys = ("type", "tool")
        if want["type"] == "event":
            keys += ("kind", "rpm", "position")
        else:
            keys += ("start", "end", "feed")
        for key in keys:
            a, b = want[key], got[key]
            if key in ("start", "end", "position"):
                equal = all(abs(x - y) <= 0.001 for x, y in zip(a, b))
            else:
                equal = a == b
            if not equal:
                return {"variant": variant, "status": "deviation",
                        "matched_items": index, "line": got["line"],
                        "field": key, "expected": a, "actual": b}
    if len(actual) < len(expected):
        return {"variant": variant, "status": "deviation",
                "matched_items": len(actual), "reason": "posted prefix incomplete"}
    if variant != "C" and len(actual) != len(expected):
        return {"variant": variant, "status": "deviation",
                "matched_items": len(expected), "reason": "extra posted motion/events"}
    if warnings:
        return {"variant": variant, "status": "unverified",
                "matched_items": len(expected), "reason": warnings[0]}
    return {"variant": variant, "status": "prefix_matches" if variant == "C" else "sequence_matches",
            "matched_items": len(expected),
            "remaining_items": len(actual) - len(expected)}


def compare_file(manifest_path, variant, posted_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if variant not in ("A", "B", "C") or manifest.get("format") != "rc01-comparison-v1":
        raise ValueError("unknown RC01 comparison variant/manifest")
    candidate = Path(manifest_path).parent / manifest["variants"][variant]["file"]
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != manifest["variants"][variant]["sha256"]:
        return {"variant": variant, "status": "unverified",
                "reason": "candidate .cb changed after comparison manifest"}
    posted = Path(posted_path).read_text(encoding="utf-8-sig")
    return compare_posted(manifest, variant, posted)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare RC01 Default-post motion")
    parser.add_argument("manifest", help="comparison.json from RC01 adapter")
    parser.add_argument("variant", choices=("A", "B", "C"))
    parser.add_argument("posted", help="CamBam-posted .nc file")
    args = parser.parse_args()
    result = compare_file(args.manifest, args.variant, args.posted)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] in ("sequence_matches", "prefix_matches") else 1)
