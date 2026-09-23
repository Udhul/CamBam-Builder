"""Bounded CamBam Plus 1.0 native output mapping fixtures and post intake.

The XML is an input to CamBam, never a certificate of its emitted motion.
Only an unchanged candidate and a native Default post can enter the corpus.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from ... import CBProject
from ...cambam_reader import read_cambam_bytes
from ...cad_entities import Pline
from ...cam_entities import MOP_XML_FIELD_PATHS


MODES = {"legacy": "Standard", "new": "Experimental"}
BUILD = "CamBam Plus 1.0 / CamBam.CAD 1.0.7364.41819 / CamBam 1.0.7364.41821"
SYSTEM_FINGERPRINTS = {
    "Default.cbpp": "adc3034633f0f6ad6ddc4c9eabc63844cb59173be8ee780ac47e198a7145c263",
    "Standard-mm.xml": "d3887ee96277a1a8e8d452b223f2d02d34b341a9de7b7e3b3e7d7080b224e4c4",
    "Default-mm.xml": "afe4b2c7abcfe17a44a9fd4752a1d40847ce0e399fde9b105bd8da66160d08db",
}
_WORD = re.compile(r"([A-Za-z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_COMMENT = re.compile(r"^\(\s*(.*?)\s*\)$")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rect(x0, y0, x1, y1):
    return Pline(vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                 closed=True)


def _common(mode, *, tool=1, diameter=2):
    return dict(target_depth=-2, depth_increment=1, stock_surface=0,
                roughing_clearance=0, clearance_plane=5,
                spindle_direction="CW", spindle_speed=12000,
                velocity_mode="ExactStop", work_plane="XY",
                optimisation_mode=MODES[mode], tool_number=tool,
                tool_diameter=diameter, tool_profile="EndMill",
                plunge_feedrate=60, cut_feedrate=240,
                max_crossover_distance=0.7)


def _add(project, method, part, targets, name, mode, **changes):
    tool = changes.pop("tool", 1)
    diameter = changes.pop("diameter", 2)
    fields = _common(mode, tool=tool, diameter=diameter)
    fields.update(changes)
    mop = getattr(project, method)(part, targets=targets, name=name,
                                   identifier=name.lower(), **fields)
    if mop is None:
        raise ValueError(f"could not author {name}")
    return mop


def _base(name):
    project = CBProject(name)
    layer = project.add_layer("Mapping shapes")
    part = project.add_part("Mapping part", stock_thickness=8,
                            stock_width=185, stock_height=45,
                            stock_offset=(0, 0), stock_surface=0,
                            nesting_method="None")
    if layer is None or part is None:
        raise ValueError("could not author mapping document")
    return project, layer, part


def _atlas(mode):
    p, layer, part = _base(f"Native shape atlas {mode}")
    rect = p.add_rect(layer, (5, 5), 10, 8, identifier="rect")
    circle = p.add_circle(layer, (30, 10), 8, identifier="circle")
    opened = p.add_pline(layer, [(48, 5), (56, 12), (64, 5)],
                          identifier="open-pline")
    closed = p.add_pline(layer, [(71, 5), (83, 5), (83, 15), (71, 15)],
                          closed=True, identifier="closed-pline")
    points = p.add_points(layer, [(94, 8), (102, 8)], identifier="points")
    txt = p.add_text(layer, "A", (113, 6), height=8, identifier="text")
    arc = p.add_arc(layer, (139, 11), 5, 25, 220, identifier="arc")
    region = p.add_region(layer, _rect(153, 5, 177, 27),
                          [_rect(161, 13, 169, 20)], identifier="region-island")
    if any(value is None for value in (rect, circle, opened, closed,
                                       points, txt, arc, region)):
        raise ValueError("could not author atlas geometry")
    _add(p, "add_profile_mop", part, [rect, circle],
         "ATLAS_PROFILE_RECT_CIRCLE", mode, profile_side="Outside",
         lead_in_type="None")
    _add(p, "add_profile_mop", part, [opened],
         "ATLAS_PROFILE_OPEN_PLINE", mode, profile_side="Outside",
         lead_in_type="None")
    _add(p, "add_pocket_mop", part, [closed],
         "ATLAS_POCKET_CLOSED_PLINE", mode, lead_in_type="None")
    _add(p, "add_pocket_mop", part, [region],
         "ATLAS_POCKET_REGION_ISLAND", mode, lead_in_type="None")
    _add(p, "add_engrave_mop", part, [arc, txt],
         "ATLAS_ENGRAVE_ARC_TEXT", mode)
    _add(p, "add_drill_mop", part, [points],
         "ATLAS_DRILL_POINTS", mode, tool=2, tool_profile="Drill",
         drilling_method="CannedCycle", peck_distance=0,
         retract_height=2, dwell=0)
    return p


def _links(mode):
    p, layer, part = _base(f"Native link interactions {mode}")
    circles = [p.add_circle(layer, xy, 8, identifier=f"link-circle-{i}")
               for i, xy in enumerate(((15, 10), (38, 10), (62, 10)), 1)]
    arc = p.add_arc(layer, (93, 11), 7, 45, 255, identifier="link-arc")
    finish = p.add_rect(layer, (122, 6), 19, 13, identifier="finish-rect")
    if any(value is None for value in (*circles, arc, finish)):
        raise ValueError("could not author link geometry")
    _add(p, "add_profile_mop", part, circles,
         "LINKS_MULTI_TARGET_DEPTH_LEAD", mode, profile_side="Outside",
         lead_in_type="Spiral", lead_in_spiral_angle=20,
         target_depth=-3, depth_increment=1, max_crossover_distance=0.25)
    _add(p, "add_engrave_mop", part, [arc],
         "LINKS_ARC_SECOND_TOOL", mode, tool=3, diameter=1)
    _add(p, "add_profile_mop", part, [finish],
         "LINKS_RETURN_FIRST_TOOL", mode, profile_side="Inside",
         lead_in_type="None", tool=1)
    return p


def _snapshot(path, mode):
    data = path.read_bytes()
    root = ET.fromstring(data)
    if root.get("Version") != "0.9.8.0":
        raise ValueError("unexpected CamBam XML version marker")
    if root.get("units") != "Millimeters":
        raise ValueError("candidate does not pin millimeter drawing units")
    options = root.find("MachiningOptions")
    if options is None or {name: options.findtext(name) for name in
                           ("PostProcessor", "StyleLibrary", "ToolLibrary")} != {
                               "PostProcessor": "Default", "StyleLibrary": "Standard-mm",
                               "ToolLibrary": "Default-mm"}:
        raise ValueError("candidate does not pin Default mm machining options")
    loaded = read_cambam_bytes(data, source_name=str(path))
    operations = []
    for mop, node in zip(loaded.list_mops(),
                         root.findall("./parts/part/machineops/*")):
        if mop.optimisation_mode != MODES[mode] or not mop.enabled:
            raise ValueError("optimisation mode or enabled state changed on strict import")
        targets = [loaded.get_primitive(uid).user_identifier
                   for uid in loaded.get_mop_targets(mop)]
        style = node.findtext("Style") or "library_default"
        if style != ("cutout" if mop.name == "LINKS_RETURN_FIRST_TOOL"
                     else "library_default"):
            raise ValueError("unexpected per-MOP style selection")
        operations.append({"name": mop.name, "kind": type(mop).__name__,
                           "targets_xml_order": targets,
                           "tool": mop.tool_number, "style": style,
                           "parameters": {field: getattr(mop, field)
                                          for field in MOP_XML_FIELD_PATHS
                                          if hasattr(mop, field)}})
    xml_mops = root.findall("./parts/part/machineops/*")
    if len(xml_mops) != len(operations) or not operations:
        raise ValueError("MOP count changed on strict import")
    for node in xml_mops:
        value = node.find("OptimisationMode")
        if value is None or value.get("state") != "Value" or value.text != MODES[mode]:
            raise ValueError("MOP mode is not explicitly pinned in XML")
    return operations


def _pin_document_settings(path):
    """Add native 1.0 document settings absent from the framework writer model."""
    content = path.read_text(encoding="utf-8")
    if content.count("<CADFile ") != 1 or content.count("<MachiningOptions>") != 1:
        raise ValueError("unexpected writer XML structure")
    content = content.replace("<CADFile ", '<CADFile units="Millimeters" ', 1)
    content = content.replace(
        "<MachiningOptions>",
        "<MachiningOptions>\n    <PostProcessor>Default</PostProcessor>"
        "\n    <StyleLibrary>Standard-mm</StyleLibrary>"
        "\n    <ToolLibrary>Default-mm</ToolLibrary>", 1)
    if "<Name>LINKS_RETURN_FIRST_TOOL</Name>" in content:
        content = content.replace("<Name>LINKS_RETURN_FIRST_TOOL</Name>",
                                  "<Name>LINKS_RETURN_FIRST_TOOL</Name>\n"
                                  "          <Style>cutout</Style>", 1)
    path.write_text(content, encoding="utf-8")


def build(directory):
    """Write four self-contained candidates and a hash-bound intake manifest."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("corpus output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    cases = {}
    for family, factory in (("atlas", _atlas), ("links", _links)):
        source = factory("legacy")
        for mode in MODES:
            key = f"{family}-{mode}"
            path = directory / f"{key}.cb"
            project = source.clone()
            project.project_name = f"Native {family} {mode}"
            for mop in project.list_mops():
                mop.optimisation_mode = MODES[mode]
            project.save(str(path))
            _pin_document_settings(path)
            cases[key] = {"file": path.name, "sha256": _digest(path),
                          "mode_label": mode, "mode_xml": MODES[mode],
                          "operations": _snapshot(path, mode),
                          "post_file": f"{key}.nc",
                          "status": "pending_native_post"}
    manifest = {"format": "cambam-optimizer-corpus-v1",
                "baseline": BUILD, "postprocessor": "Default",
                "profile": "Default mm", "units": "mm",
                "style_library": "Standard-mm", "tool_library": "Default-mm",
                "expected_system_sha256": SYSTEM_FINGERPRINTS,
                "xml_version_marker": "0.9.8.0",
                "mode_mapping_status": "requires_CamBam_UI_confirmation",
                "cases": cases}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_default_motion(content, names):
    """Read posted modal motion conservatively; retain unknown cycles as raw words.

    The incoming machine position and controller trajectory are unavailable.
    The first move can therefore have a partial or entirely unknown start.
    """
    known = set(names)
    section = None
    sections, moves, events, unsupported = [], [], [], []
    position = [None, None, None]
    motion = None
    feed = None
    tool = None
    units = absolute = plane = False
    for number, raw in enumerate(content.splitlines(), 1):
        stripped = raw.strip()
        comment = _COMMENT.fullmatch(stripped)
        if comment and comment.group(1) in known:
            section = comment.group(1)
            sections.append({"name": section, "line": number})
            continue
        line = re.sub(r"\([^()]*\)", "", stripped).strip()
        if not line or line == "%":
            continue
        words = _WORD.findall(line)
        if "".join(a + b for a, b in words).replace(" ", "") != line.replace(" ", ""):
            unsupported.append({"line": number, "words": line, "reason": "unparsed_text"})
            continue
        values = [(a.upper(), value) for a, value in words if a.upper() != "N"]
        gs = [int(float(value)) for a, value in values
              if a == "G" and float(value).is_integer()]
        ms = [int(float(value)) for a, value in values
              if a == "M" and float(value).is_integer()]
        if len(gs) != sum(a == "G" for a, _ in values):
            unsupported.append({"line": number, "words": line, "reason": "fractional_G"})
            continue
        if len(ms) != sum(a == "M" for a, _ in values):
            unsupported.append({"line": number, "words": line, "reason": "fractional_M"})
            continue
        for g in gs:
            if g == 21:
                units = True
            elif g == 90:
                absolute = True
            elif g == 17:
                plane = True
            elif g in (0, 1, 2, 3):
                motion = g
            elif g == 80:
                motion = None
            elif g not in (40, 61, 64):
                unsupported.append({"line": number, "words": line,
                                    "reason": f"unsupported_G{g}"})
                motion = None
        value_map = {a: float(value) for a, value in values if a in "XYZFIJST"}
        if "F" in value_map:
            feed = value_map["F"]
        if "T" in value_map:
            if not value_map["T"].is_integer():
                unsupported.append({"line": number, "words": line,
                                    "reason": "fractional_tool"})
            else:
                tool = int(value_map["T"])
        for m in ms:
            events.append({"line": number, "m": m, "tool": tool,
                           "operation_at_line": section, "words": line,
                           "position": list(position)})
            if m not in (3, 4, 5, 6, 30):
                unsupported.append({"line": number, "words": line,
                                    "reason": f"unsupported_M{m}"})
        if not any(a in value_map for a in "XYZ"):
            continue
        if not (units and absolute and motion in (0, 1, 2, 3)):
            if not any(item["line"] == number for item in unsupported):
                unsupported.append({"line": number, "words": line,
                                    "reason": "motion_state_unknown"})
            continue
        if motion in (2, 3) and not plane:
            unsupported.append({"line": number, "words": line,
                                "reason": "arc_plane_unknown"})
            continue
        start = list(position)
        for index, axis in enumerate("XYZ"):
            if axis in value_map:
                position[index] = value_map[axis]
        move = {"line": number, "operation": section, "g": motion,
                "start": start, "end": list(position), "tool": tool,
                "feed": 0 if motion == 0 else feed, "words": line}
        if motion in (2, 3):
            if (position[0] is None or position[1] is None or
                    start[0] is None or start[1] is None or
                    "I" not in value_map or "J" not in value_map):
                unsupported.append({"line": number, "words": line,
                                    "reason": "arc_center_unknown"})
            else:
                move["center"] = [start[0] + value_map["I"],
                                  start[1] + value_map["J"]]
                r0 = math.hypot(start[0] - move["center"][0],
                                start[1] - move["center"][1])
                r1 = math.hypot(position[0] - move["center"][0],
                                position[1] - move["center"][1])
                move["radius_mismatch_mm"] = abs(r0 - r1)
                if r0 <= 0 or move["radius_mismatch_mm"] > 0.001:
                    unsupported.append({"line": number, "words": line,
                                        "reason": "inconsistent_arc_radius"})
        moves.append(move)
    return {"sections": sections, "moves": moves, "events": events,
            "unsupported": unsupported,
            "incoming_machine_position": "unknown"}


def inspect_post(manifest_path, key, post_path):
    """Hash-guard input and record native sections without guessing stock removal."""
    manifest_path, post_path = Path(manifest_path), Path(post_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cambam-optimizer-corpus-v1":
        raise ValueError("unknown mapping manifest")
    case = manifest["cases"][key]
    source = manifest_path.parent / case["file"]
    if _digest(source) != case["sha256"]:
        raise ValueError("candidate .cb hash changed")
    if _snapshot(source, case["mode_label"]) != case["operations"]:
        raise ValueError("candidate .cb semantics changed")
    content = post_path.read_text(encoding="utf-8-sig")
    if len(content) > 20_000_000:
        raise ValueError("post exceeds 20 MB")
    lines = content.splitlines()
    title = re.compile(r"^\(\s*" + re.escape(source.stem) + r"(?:\s|\))")
    if not any(title.match(line.strip()) for line in lines[:6]):
        raise ValueError("post title does not identify the candidate .cb")
    if not any("( Post processor: Default )" in line for line in lines[:12]):
        raise ValueError("post lacks Default header")
    if not any("G21" in line and "G90" in line for line in lines[:20]):
        raise ValueError("post lacks absolute-mm preamble")
    names = {op["name"] for op in case["operations"]}
    parsed = parse_default_motion(content, names)
    found = [section["name"] for section in parsed["sections"]]
    if len(found) != len(names) or set(found) != names:
        raise ValueError(f"MOP comments incomplete or repeated: {found}")
    if not any(re.search(r"\bM30\b", line) for line in lines[-10:]):
        raise ValueError("post lacks M30")
    return {"case": key, "candidate_sha256": case["sha256"],
            "post_sha256": _digest(post_path), "post_file": post_path.name,
            "observed_mop_order": found, **parsed,
            "status": "posted_unreviewed", "stock_authority": "none"}


def _stable_fingerprint(items):
    stable = [{key: value for key, value in item.items() if key != "line"}
              for item in items]
    data = json.dumps(stable, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def observe_corpus(manifest_path, *, post_directory=None):
    """Summarize four exact native posts without accepting stock removal.

    Fingerprints exclude line numbers/timestamp headers and retain ordered
    posted moves, events and unresolved words. Raw .nc remains the authority.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cambam-optimizer-corpus-v1":
        raise ValueError("unknown mapping manifest")
    post_directory = (Path(post_directory) if post_directory is not None
                      else manifest_path.parent)
    cases = {}
    for key, case in manifest["cases"].items():
        audit = inspect_post(manifest_path, key,
                             post_directory / case["post_file"])
        sections = []
        for index, marker in enumerate(audit["sections"]):
            name = marker["name"]
            last_line = (audit["sections"][index + 1]["line"]
                         if index + 1 < len(audit["sections"]) else float("inf"))
            op = next(op for op in case["operations"] if op["name"] == name)
            moves = [move for move in audit["moves"]
                     if move["operation"] == name]
            cuts = [move for move in moves if move["g"] in (1, 2, 3)]
            approaches = []
            for move in moves:
                if (move["g"] == 0 and move["end"][2] == 5.0
                        and move["end"][:2] != move["start"][:2]
                        and None not in move["end"][:2]):
                    xy = move["end"][:2]
                    if not approaches or approaches[-1] != xy:
                        approaches.append(xy)
            depths = []
            for move in cuts:
                z = move["end"][2]
                if z is not None and z < 0 and z not in depths:
                    depths.append(z)
            unresolved = [item for item in audit["unsupported"]
                          if marker["line"] < item["line"] < last_line]
            sections.append({
                "name": name, "kind": op["kind"],
                "targets_xml_order": op["targets_xml_order"],
                "style": op["style"], "tool": op["tool"],
                "motion_sha256": _stable_fingerprint(moves),
                "motion_word_count": len(moves),
                "g_counts": {f"G{g}": sum(move["g"] == g for move in moves)
                             for g in (0, 1, 2, 3)},
                "feed_values": sorted({move["feed"] for move in cuts
                                       if move["feed"] is not None}),
                "cut_depth_endpoints_ordered_mm": depths,
                "approach_xy_at_z5_ordered_mm": approaches,
                "first_cut_move": ({field: cuts[0].get(field) for field in
                                    ("g", "start", "end", "center", "feed")}
                                   if cuts else None),
                "last_cut_end": cuts[-1]["end"] if cuts else None,
                "max_arc_radius_mismatch_mm": max(
                    (move.get("radius_mismatch_mm", 0.0) for move in moves),
                    default=0.0),
                "unresolved_words": unresolved,
            })
        cases[key] = {
            "candidate_sha256": case["sha256"],
            "post_sha256": audit["post_sha256"],
            "program_motion_sha256": _stable_fingerprint(
                audit["moves"] + audit["events"] + audit["unsupported"]),
            "observed_mop_order": audit["observed_mop_order"],
            "sections": sections, "events": audit["events"],
            "unsupported": audit["unsupported"],
            "status": "posted_observation_unreviewed",
            "stock_authority": "none",
        }
    comparisons = {}
    for family in ("atlas", "links"):
        legacy, new = (cases[f"{family}-{mode}"] for mode in MODES)
        new_sections = {section["name"]: section for section in new["sections"]}
        comparisons[family] = {
            "same_program_motion": (legacy["program_motion_sha256"] ==
                                    new["program_motion_sha256"]),
            "same_section_motion": {
                left["name"]: (left["motion_sha256"] ==
                               new_sections[left["name"]]["motion_sha256"])
                for left in legacy["sections"]
            },
        }
    return {"format": "cambam-optimizer-observations-v1",
            "manifest_sha256": _digest(manifest_path),
            "cases": cases, "mode_comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--post", nargs=2, metavar=("CASE", "NC_FILE"))
    parser.add_argument("--observe", action="store_true",
                        help="summarize all native posts beside the manifest")
    parser.add_argument("--record", help="write UTF-8 JSON to a new file")
    args = parser.parse_args()
    if args.post and args.observe:
        parser.error("--post and --observe are mutually exclusive")
    result = (inspect_post(Path(args.directory) / "manifest.json", *args.post)
              if args.post else
              observe_corpus(Path(args.directory) / "manifest.json")
              if args.observe else build(args.directory))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.record:
        target = Path(args.record)
        if target.exists():
            parser.error("--record target must be new")
        target.write_text(rendered, encoding="utf-8")
        print(f"wrote {target}")
    else:
        print(rendered, end="")
