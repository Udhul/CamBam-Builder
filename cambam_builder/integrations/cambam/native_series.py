"""Bounded normalization of an actual CamBam Default MOP-series post.

This adapter admits one enabled, non-nested millimetre Part with uniquely named
enabled MOPs and explicitly numbered cylindrical endmills. It retains posted
G0/G1/G2/G3 motion in file order. Only planar G0/G1 motion can be lowered to
the current core replay Trace; arcs and ramps need a separate continuous-sweep
proof. Parsing alone never grants stock or access authority: callers must replay
the lowered trace against their independently normalized target and tool lengths.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Optional
from xml.etree import ElementTree as ET

from ...cam_core import replay
from ...native.reader import read_cambam_bytes
from .rc01_post import read_default_post


_SECTION = re.compile(r"^\(\s*(.*?)\s*\)$")
_TOOL_COMMENT = re.compile(r"T\d+\s*:\s*[+\-]?(?:\d+(?:\.\d*)?|\.\d+)")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _stable(value):
    if isinstance(value, dict):
        return tuple((key, _stable(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_stable(item) for item in value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite source geometry")
        return number
    if value is None or isinstance(value, (str, bool)):
        return value
    raise ValueError("unsupported source geometry value")


def _source_primitives(project, required_names=None):
    found = {}
    for primitive in project.list_primitives():
        name = primitive.user_identifier
        if required_names is not None and name not in required_names:
            continue
        if not name or name in found or type(primitive).__name__ == "Text":
            raise ValueError("source needs uniquely identified supported primitives")
        # World XYZ includes directed Region/Pline bulges and analytic Circle/Arc
        # parameters; it also detects pose or parent changes without relying on
        # CamBam's numeric XML IDs.
        found[name] = (str(primitive.internal_id), type(primitive).__name__,
                       _stable(primitive.get_absolute_coordinates_xyz()))
    if not found:
        raise ValueError("source has no supported primitives")
    return found


def _part_stock(part):
    fields = ("user_identifier", "enabled", "stock_thickness", "stock_width",
              "stock_height", "stock_surface", "stock_offset", "machining_origin",
              "default_tool_diameter", "default_spindle_speed", "nesting_method",
              "nesting_rows", "nesting_columns", "nesting_spacing",
              "nesting_grid_order", "nesting_grid_alternate")
    snapshot = tuple((field, _stable(getattr(part, field))) for field in fields)
    # Preserve the accepted explicit-stock semantic fingerprint while giving
    # stockless imports their own identity, even with identical dimensions.
    return snapshot if part.stock_present else snapshot + (("stock_present", False),)


def _source_semantic(project, data):
    """Stable source identity for MOP-free source documents only.

    Source MOP intent has no bounded semantic normalizer here, so documents
    containing MOPs retain exact-byte freshness. Document/layer presentation
    edits on a MOP-free source do not invalidate geometry and stock evidence.
    """
    root = ET.fromstring(data)
    if root.get("units") not in (None, "Millimeters"):
        raise ValueError("source drawing units changed")
    if project.list_mops():
        return _sha(data), True
    parts = project.list_parts()
    if len(parts) != 1:
        raise ValueError("semantic source requires one Part")
    snapshot = (_part_stock(parts[0]), _source_primitives(project), "mm")
    return _sha(repr(_stable(snapshot)).encode("utf-8")), False


@dataclass(frozen=True)
class NativeStage:
    name: str
    kind: str
    tool: str
    diameter_mm: float
    work_plane: str
    stock_surface_mm: float
    target_ids: tuple
    first_line: int
    move_count: int


@dataclass(frozen=True)
class PostedMove:
    operation: str
    tool: str
    g: int
    start: tuple
    end: tuple
    feed: float
    center: Optional[tuple]
    line: int


@dataclass(frozen=True)
class PostedEvent:
    kind: str
    tool: str
    position: tuple
    rpm: float
    line: int


@dataclass(frozen=True)
class NativeSeries:
    source_sha256: str
    source_semantic_sha256: str
    source_has_mops: bool
    candidate_sha256: str
    post_sha256: str
    motion_sha256: str
    setup_bound_externally: bool
    initial_position: tuple
    stages: tuple
    items: tuple

    def parsed_evidence(self):
        """Report parser evidence without promoting it to a stock certificate."""
        return {
            "document_fidelity": {
                "status": "pass",
                "scope": "bound native source/candidate geometry, Part and MOP order",
                "source_sha256": self.source_sha256,
                "candidate_sha256": self.candidate_sha256,
            },
            "posted_motion": {
                "status": "pass", "scope": "strict parsed CamBam Default post",
                "post_sha256": self.post_sha256,
                "motion_sha256": self.motion_sha256,
                "initial_position_assumption_xyz_mm": self.initial_position,
            },
            "stock_access_residual": {
                "status": "not_evaluated",
                "reason": "no caller-supplied initial stock and replay certificate",
            },
        }

    @property
    def evidence_fingerprint(self):
        return _sha(json.dumps((self.source_semantic_sha256, self.candidate_sha256,
                                self.post_sha256, self.motion_sha256,
                                self.setup_bound_externally, "mm", "Default"),
                               separators=(",", ":")).encode("ascii"))

    def check_freshness(self, source_path, candidate_path, post_path, *, setup=None):
        """Retain harmless MOP-free source edits; pin candidate/post exactly."""
        source_data = Path(source_path).read_bytes()
        if (_sha(Path(candidate_path).read_bytes()) != self.candidate_sha256 or
                _sha(Path(post_path).read_bytes()) != self.post_sha256):
            raise ValueError("native series candidate or post changed")
        if _sha(source_data) != self.source_sha256:
            if self.source_has_mops:
                raise ValueError("native series source with MOP intent changed")
            current = read_cambam_bytes(source_data, source_name=str(source_path))
            semantic, has_mops = _source_semantic(current, source_data)
            if has_mops or semantic != self.source_semantic_sha256:
                raise ValueError("native series source geometry or stock changed")
        if self.setup_bound_externally and setup != {"units": "mm", "postprocessor": "Default"}:
            raise ValueError("native series bound setup missing or changed")
        return True

    def to_trace(self, targets, cutting_lengths_mm, entry_modes):
        """Lower a linear series to core replay with explicit entry intent.

        ``targets`` maps each MOP name to an independently normalized core
        Target. ``entry_modes`` maps each name to ``virgin`` or ``cleared``.
        Every low travel and cleared descent is then checked by core replay.
        This method does not discretize arcs or infer a rounded/pointed cutter.
        """
        names = {stage.name for stage in self.stages}
        if (set(targets) != names or set(entry_modes) != names or
                {stage.tool for stage in self.stages} != set(cutting_lengths_mm)):
            raise ValueError("series replay requires exact stage targets, entry modes and tools")
        if any(mode not in ("virgin", "cleared") for mode in entry_modes.values()):
            raise ValueError("unsupported stage entry mode")
        if any(stage.kind not in ("PocketMop", "ProfileMop", "EngraveMop") or
               stage.work_plane != "XY" or stage.stock_surface_mm != 0
               for stage in self.stages):
            raise ValueError("unsupported native stage type, plane or stock surface for replay")
        if any(type(target) is not replay.Target for target in targets.values()):
            raise ValueError("series replay needs core Target values")
        tools = {}
        for stage in self.stages:
            length = cutting_lengths_mm[stage.tool]
            if not isinstance(length, (int, float)) or isinstance(length, bool) or not math.isfinite(length) or length <= 0:
                raise ValueError("finite positive cutting length required")
            profile = replay.ToolProfile(stage.tool, "cylinder", stage.diameter_mm / 2, length)
            if stage.tool in tools and tools[stage.tool] != profile:
                raise ValueError("one tool number has inconsistent diameters")
            tools[stage.tool] = profile
        operations = tuple(replay.Operation(stage.name, tools[stage.tool], targets[stage.name])
                           for stage in self.stages)
        lowered = []
        for item in self.items:
            if isinstance(item, PostedEvent):
                if item.kind == "spindle_start" and item.rpm <= 0:
                    raise ValueError(f"line {item.line}: nonpositive spindle speed")
                lowered.append(replay.Event(item.kind, item.tool, item.position))
                continue
            if item.g not in (0, 1):
                raise ValueError(f"line {item.line}: posted arc needs continuous sweep proof")
            a, b = item.start, item.end
            low = min(a[2], b[2])
            if item.g == 1 and low < 0 and item.feed <= 0:
                raise ValueError(f"line {item.line}: nonpositive cutting feed")
            if item.g == 0:
                if low < 0 and a[:2] != b[:2]:
                    raise ValueError(f"line {item.line}: low XY rapid is unsupported")
                role = ("rapid" if low >= 0 else
                        "retract" if b[2] > a[2] else "cleared_descent")
            elif low >= 0:
                role = "approach"
            elif a[:2] != b[:2] and a[2] != b[2]:
                raise ValueError(f"line {item.line}: posted ramp needs all-height proof")
            elif a[:2] != b[:2]:
                role = "cut"
            elif b[2] < a[2]:
                role = "entry" if entry_modes[item.operation] == "virgin" else "cleared_descent"
            elif b[2] > a[2]:
                role = "retract"
            else:
                raise ValueError(f"line {item.line}: unresolved stationary feed")
            lowered.append(replay.Motion(role, item.tool, item.operation, a, b, item.feed))
        return replay.Trace(self.evidence_fingerprint, "native-default-mm", self.initial_position,
                            operations, tuple(lowered))


def normalize_native_series(source_path, candidate_path, post_path, *, initial_position,
                            setup=None):
    """Bind actual posted motion to exact source/candidate/post bytes.

    Source/candidate reimport checks source primitive identity and analytic world
    geometry plus Part stock/nesting, while allowing derived attachments. The
    post is accepted only with a matching candidate title, Default header, complete
    enabled MOP comments in document order and no unsupported modal words.
    """
    source_path, candidate_path, post_path = map(Path, (source_path, candidate_path, post_path))
    source_data, candidate_data, post_data = (path.read_bytes() for path in
                                             (source_path, candidate_path, post_path))
    source = read_cambam_bytes(source_data, source_name=str(source_path))
    project = read_cambam_bytes(candidate_data, source_name=str(candidate_path))
    root = ET.fromstring(candidate_data)
    source_root = ET.fromstring(source_data)
    if setup is not None and (not isinstance(setup, dict) or
                              setup != {"units": "mm", "postprocessor": "Default"}):
        raise ValueError("series setup must explicitly select mm and Default")
    xml_units = root.get("units")
    xml_post = root.findtext("./MachiningOptions/PostProcessor")
    if (source_root.get("units") not in (None, "Millimeters") or
            xml_units not in (None, "Millimeters") or xml_post not in (None, "Default") or
            (xml_units is None or xml_post is None) and setup is None):
        raise ValueError("series needs bound millimetre units and Default postprocessor")
    parts = project.list_parts()
    if len(parts) != 1 or not parts[0].enabled or parts[0].nesting_method != "None":
        raise ValueError("series supports one enabled, non-nested Part")
    source_parts = source.list_parts()
    if (len(source_parts) != 1 or _part_stock(source_parts[0]) != _part_stock(parts[0])):
        raise ValueError("candidate changed source Part stock or nesting")
    original = _source_primitives(source)
    semantic_source, source_has_mops = _source_semantic(source, source_data)
    candidate_primitives = _source_primitives(project, original.keys())
    if any(candidate_primitives.get(name) != geometry for name, geometry in original.items()):
        raise ValueError("candidate changed source primitive identity or analytic geometry")
    mops = [mop for mop in project.get_mops_in_part(parts[0]) if mop.enabled]
    if not mops or len({mop.name for mop in mops}) != len(mops):
        raise ValueError("series needs unique enabled MOP names")
    for mop in mops:
        states = getattr(mop, "_xml_parameter_states", {})
        if (mop.tool_profile != "EndMill" or not isinstance(mop.tool_number, int) or
                mop.tool_number <= 0 or states.get("tool_number") != "Value" or
                not isinstance(mop.tool_diameter, (int, float)) or
                isinstance(mop.tool_diameter, bool) or
                not math.isfinite(mop.tool_diameter) or mop.tool_diameter <= 0 or
                states.get("tool_diameter") != "Value" or
                states.get("tool_profile") != "Value" or
                not project.get_mop_targets(mop)):
            raise ValueError(f"MOP {mop.name!r} lacks explicit cylindrical tool or targets")
    try:
        post = post_data.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("post must be UTF-8") from exc
    lines = post.splitlines()
    if ("( Post processor: Default )" not in lines[:12] or
            not any(re.match(r"^\(\s*" + re.escape(candidate_path.stem) + r"(?:\s|\))", line)
                    for line in lines[:12])):
        raise ValueError("post header does not identify candidate and Default")
    expected = [mop.name for mop in mops]
    markers = [(number, match.group(1)) for number, line in enumerate(lines, 1)
               if (match := _SECTION.fullmatch(line.strip())) and match.group(1) in expected]
    if [name for _, name in markers] != expected:
        raise ValueError("post MOP sections are missing, repeated or out of order")
    all_names = {mop.name for mop in project.get_mops_in_part(parts[0])}
    for number, line in enumerate(lines, 1):
        match = _SECTION.fullmatch(line.strip())
        if not match or match.group(1) in expected or _TOOL_COMMENT.fullmatch(match.group(1)):
            continue
        if match.group(1) in all_names or number > markers[0][0]:
            raise ValueError(f"line {number}: ambiguous or disabled MOP section")
    raw_items, warnings = read_default_post(post, allow_arcs=True,
                                            initial_position=initial_position)
    warnings = [warning for warning in warnings
                if "initial machine position is not encoded" not in warning]
    if warnings:
        raise ValueError(f"posted motion unresolved: {warnings[0]}")
    stages = []
    items = []
    move_counts = {name: 0 for name in expected}
    cut_counts = {name: 0 for name in expected}
    section_index = -1
    for raw in raw_items:
        while section_index + 1 < len(markers) and raw["line"] > markers[section_index + 1][0]:
            section_index += 1
        if raw["type"] == "event":
            items.append(PostedEvent(raw["kind"], raw["tool"], tuple(raw["position"]),
                                     raw["rpm"], raw["line"]))
            continue
        if section_index < 0:
            raise ValueError(f"line {raw['line']}: motion before first MOP section")
        mop = mops[section_index]
        if raw["tool"] != f"T{mop.tool_number}":
            raise ValueError(f"line {raw['line']}: posted tool differs from MOP")
        name = mop.name
        move_counts[name] += 1
        # A posted cut can be entirely above program Z=0 when the explicit
        # MOP stock surface is positive. Native stock presence does not shift
        # these authored or posted coordinates.
        if (raw["g"] in (1, 2, 3) and
                min(raw["start"][2], raw["end"][2]) <
                float(mop.stock_surface) - 1e-9):
            cut_counts[name] += 1
        items.append(PostedMove(name, raw["tool"], raw["g"],
                                tuple(raw["start"]), tuple(raw["end"]), raw["feed"],
                                tuple(raw["center"]) if "center" in raw else None,
                                raw["line"]))
    for mop, (line, name) in zip(mops, markers):
        if not cut_counts[name]:
            raise ValueError(f"MOP {name!r} emitted no stock cutting motion")
        targets = tuple(sorted(project.get_primitive(uid).user_identifier
                               for uid in project.get_mop_targets(mop)))
        stages.append(NativeStage(name, type(mop).__name__, f"T{mop.tool_number}",
                                  float(mop.tool_diameter), mop.work_plane,
                                  float(mop.stock_surface), targets, line,
                                  move_counts[name]))
    stable = [(item.__class__.__name__, *tuple(getattr(item, field) for field in
               ("operation", "tool", "g", "start", "end", "feed", "center")
               if hasattr(item, field))) if isinstance(item, PostedMove) else
              ("event", item.kind, item.tool, item.position, item.rpm) for item in items]
    motion_sha = _sha(json.dumps(stable, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
    return NativeSeries(_sha(source_data), semantic_source, source_has_mops,
                        _sha(candidate_data), _sha(post_data),
                        motion_sha, xml_units is None or xml_post is None,
                        tuple(initial_position), tuple(stages), tuple(items))
