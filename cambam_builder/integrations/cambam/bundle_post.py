"""Audit complete straight CamBam Default posts of source-bound BO01 documents.

Native path order and direction may differ within a stage. Every cutting edge
must still occur with its original multiplicity, and every supported posted
movement is checked independently before actual-coordinate stock evaluation.
This adapter grants no controller, physical machining or search-cost evidence.
"""

from dataclasses import replace
import hashlib
from itertools import product
import math
import re

from ...cam_core import occupancy, ordered_job as oj, replay, v_region as v
from ...native.reader import read_cambam_bytes
from .bundle_document import NativeDocument
from .rc01_post import read_default_post


_COMMENT = re.compile(r"\(\s*([^()]*)\s*\)")
_TOOL_COMMENT = re.compile(r"T([1-9][0-9]*)\s*:\s*([+]?(?:\d+(?:\.\d*)?|\.\d+))")
_INITIAL_WARNING = re.compile(r"line [0-9]+: initial machine position is not encoded")
_TOL = oj.MATCH_TOLERANCE_MM


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _near(a, b):
    return all(abs(x - y) <= _TOL for x, y in zip(a, b))


def _cell(point):
    scaled = tuple(value / _TOL for value in point)
    if any(not math.isfinite(value) for value in scaled):
        raise ValueError("native edge coordinates exceed matching resolution")
    return tuple(math.floor(value) for value in scaled)


def _same_edge(a, b):
    return ((_near(a[0], b[0]) and _near(a[1], b[1])) or
            (_near(a[0], b[1]) and _near(a[1], b[0])))


def _edge_multiset(stage, moves):
    """Conservatively consume all expected edges; ambiguous tolerance fails."""
    expected = [(move.start, move.end) for move in stage.motions if move.role == "cut"]
    actual = [(move.start, move.end) for move in moves if move.role == "cut"]
    if len(actual) != len(expected):
        raise ValueError(f"stage {stage.id}: missing or extra native cutting edge")
    buckets = {}
    for index, edge in enumerate(expected):
        for point in edge:
            buckets.setdefault(_cell(point), set()).add(index)
    remaining = set(range(len(expected)))
    for edge in actual:
        base = _cell(edge[0])
        possible = set()
        for delta in product((-1, 0, 1), repeat=3):
            key = tuple(a + b for a, b in zip(base, delta))
            possible.update(buckets.get(key, ()))
        matches = [index for index in possible & remaining
                   if _same_edge(edge, expected[index])]
        if not matches:
            raise ValueError(f"stage {stage.id}: native cutting edge differs")
        # Identical original duplicates have meaningful multiplicity. Distinct
        # originals competing within tolerance cannot be assigned confidently.
        canonical = {tuple(sorted(expected[index])) for index in matches}
        if len(canonical) != 1:
            raise ValueError(f"stage {stage.id}: ambiguous native edge tolerance")
        remaining.remove(matches[0])
    if remaining:
        raise ValueError(f"stage {stage.id}: missing native cutting edge")
    return len(actual)


def _sections(text, document, candidate):
    project = read_cambam_bytes(candidate, source_name="audited native bundle")
    names = tuple(project.get_mop(identifier).name
                  for identifier in document.stage_mop_identifiers)
    if (len(set(names)) != len(names) or
            any(not name or any(c in name for c in "()\r\n") for name in names)):
        raise ValueError("unambiguous single-line native MOP names required")
    nonempty = [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1)
                if line.strip()]
    if (len(nonempty) < 3 or
            nonempty[0][1] != "( Made using CamBam - http://www.cambam.co.uk )" or
            _COMMENT.fullmatch(nonempty[1][1]) is None or
            nonempty[2][1] != "( Post processor: Default )"):
        raise ValueError("complete CamBam Default header required")
    header_lines = {number for number, _ in nonempty[:3]}
    diameters = {}
    for stage in document.bundle.job.stages:
        diameter = (2 * stage.v_plan.tool.maximum_radius if stage.v_plan is not None
                    else 2 * stage.operation.tool.radius)
        diameters.setdefault(stage.tool_id, set()).add(diameter)
    markers = []
    for number, raw in enumerate(text.splitlines(), 1):
        comments = list(_COMMENT.finditer(raw))
        if not comments:
            continue
        if len(comments) != 1 or comments[0].group(0).strip() != raw.strip():
            raise ValueError(f"line {number}: ambiguous native section comment")
        comment = comments[0].group(1).strip()
        if number in header_lines:
            continue
        if len(markers) < len(names) and comment == names[len(markers)]:
            markers.append(number)
            continue
        tool = _TOOL_COMMENT.fullmatch(comment)
        if tool:
            tool_id, diameter = f"T{tool[1]}", float(tool[2])
            if any(abs(diameter - expected) <= _TOL
                   for expected in diameters.get(tool_id, ())):
                continue
        raise ValueError(f"line {number}: extra, repeated or out-of-order native MOP section")
    if len(markers) != len(names):
        raise ValueError("missing native MOP section")
    return names, tuple(markers)


def _classify(stage, items, safe_z):
    moves, paths, points = [], [], None
    cut_feed = {m.feed for m in stage.motions if m.role == "cut"}
    entry_feed = {m.feed for m in stage.motions if m.role == "entry"}
    for item in items:
        start, end = tuple(item["start"]), tuple(item["end"])
        vertical = start[:2] == end[:2]
        g, feed = item["g"], item["feed"]
        if g == 0 and min(start[2], end[2]) >= safe_z - _TOL:
            if points is not None:
                raise ValueError(f"stage {stage.id}: rapid before path retract")
            role = "rapid"
        elif (vertical and end[2] > start[2] and
              abs(end[2] - safe_z) <= _TOL and points is not None and
              len(points) >= 2):
            if g == 1 and not any(abs(feed - f) <= _TOL for f in cut_feed | entry_feed):
                raise ValueError(f"stage {stage.id}: unbound native retract feed")
            role = "rapid_retract" if g == 0 else "retract"
            paths.append(tuple(points))
            points = None
        elif (g == 1 and vertical and points is None and
              abs(start[2] - safe_z) <= _TOL and end[2] < 0):
            if not any(abs(feed - f) <= _TOL for f in entry_feed):
                raise ValueError(f"stage {stage.id}: native entry feed differs")
            role = "entry"
            points = [end]
        elif (g == 1 and not vertical and points is not None and
              start[2] < 0 and end[2] < 0):
            if not any(abs(feed - f) <= _TOL for f in cut_feed):
                raise ValueError(f"stage {stage.id}: native cut feed differs")
            role = "cut"
            points.append(end)
        else:
            raise ValueError(f"stage {stage.id}: unsafe or unplanned native movement")
        moves.append(oj.JobMove(role, start, end, feed))
    if points is not None or not paths:
        raise ValueError(f"stage {stage.id}: missing complete native entry/cut/retract path")
    return tuple(moves), tuple(paths)


def _stock_stage(job, stage, moves, paths):
    if stage.v_plan is not None:
        # Synthetic canonical links are solely the stock owner's required VPlan
        # representation. Actual links/entries/retracts were all checked above
        # and are passed unchanged to occupancy and axial verification below.
        actual_paths = tuple(v.VPath("fill", tuple((x, y, -z) for x, y, z in path))
                             for path in paths)
        plan = replace(stage.v_plan, paths=actual_paths,
                       motions=v._motions(actual_paths, stage.v_plan.safe_z))
        return v.verify(plan)
    at = moves[0].start
    items = [replay.Event("tool_change", stage.tool_id, at),
             replay.Event("spindle_start", stage.tool_id, at)]
    for move in moves:
        role = "retract" if move.role == "rapid_retract" else move.role
        items.append(replay.Motion(role, stage.tool_id, stage.id,
                                   move.start, move.end, move.feed))
    items.append(replay.Event("spindle_stop", stage.tool_id, moves[-1].end))
    trace = replay.Trace(job.source_fingerprint, job.program_frame, at,
                         (stage.operation,), tuple(items))
    replay.replay(trace, expected_source=job.source_fingerprint)
    return trace


def audit_post(document: NativeDocument, post_bytes: bytes, *,
               candidate_bytes: bytes | None = None,
               source_bytes: bytes | None = None,
               max_residual_area_mm2: float,
               max_residual_volume_mm3: float) -> dict:
    """Verify a whole native straight Default post, raising on any failed gate.

    Requires explicit residual upper limits and a declared initial tip, stock,
    fixtures, tool bodies and axial controls in the bound bundle. Stage order is
    fixed; within-stage path order/reversal is accepted by a complete undirected
    edge multiset at ``MATCH_TOLERANCE_MM``. Actual geometry, not cached selected
    metrics, supplies the residual bounds. The result describes conditional
    geometric evidence and operator setup assumptions, never machine telemetry.
    """
    if type(document) is not NativeDocument or type(post_bytes) is not bytes:
        raise ValueError("bound native document and complete posted bytes required")
    for limit in (max_residual_area_mm2, max_residual_volume_mm3):
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit < 0:
            raise ValueError("finite nonnegative native residual limits required")
    document_report = document.check_document(candidate_bytes, source_bytes=source_bytes)
    candidate = document.xml_bytes if candidate_bytes is None else candidate_bytes
    try:
        text = post_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("native post must be UTF-8 text") from exc
    names, markers = _sections(text, document, candidate)
    job = document.bundle.job
    safe_z = job.initial_tip[2]
    items, warnings = read_default_post(text, initial_position=job.initial_tip)
    if (len(warnings) > 1 or any(_INITIAL_WARNING.fullmatch(warning) is None
                                  for warning in warnings)):
        raise ValueError(f"unsupported native post warnings: {warnings}")
    groups = [[] for _ in job.stages]
    at, active_tool, running, rpm = job.initial_tip, None, False, None
    observed_events = []
    marker_index = 0
    stage_index = -1
    for item in items:
        while marker_index < len(markers) and markers[marker_index] < item["line"]:
            if abs(at[2] - safe_z) > _TOL:
                raise ValueError("native MOP boundary is below declared clearance")
            stage_index = marker_index
            marker_index += 1
        if item["type"] == "event":
            if abs(at[2] - safe_z) > _TOL or tuple(item["position"]) != at:
                raise ValueError("native spindle/tool event outside declared safe plane")
            kind, tool = item["kind"], item["tool"]
            if kind == "tool_change":
                if running:
                    raise ValueError("native tool change with running spindle")
                active_tool = tool
            elif kind == "spindle_start":
                if running or active_tool != tool:
                    raise ValueError("invalid native spindle start")
                running, rpm = True, item["rpm"]
            else:
                if not running or active_tool != tool:
                    raise ValueError("invalid native spindle stop")
                running = False
            observed_events.append((kind, tool, item["rpm"]))
            continue
        if stage_index < 0 or tuple(item["start"]) != at:
            raise ValueError("native motion outside bound MOP or discontinuous stream")
        stage = job.stages[stage_index]
        if item["tool"] != stage.tool_id or active_tool != stage.tool_id:
            raise ValueError("native motion uses wrong stage tool")
        if min(item["start"][2], item["end"][2]) <= 0 and (
                not running or rpm != stage.rpm):
            raise ValueError("native stock movement requires bound running spindle/rpm")
        groups[stage_index].append(item)
        at = tuple(item["end"])
    if marker_index != len(markers) or abs(at[2] - safe_z) > _TOL or running:
        raise ValueError("native post lacks complete stages and stopped safe return")
    expected_events = []
    previous = None
    for stage in job.stages:
        if stage.tool_id != previous:
            if previous is not None:
                expected_events.append(("spindle_stop", previous, 0))
            expected_events.extend((("tool_change", stage.tool_id, 0),
                                    ("spindle_start", stage.tool_id, stage.rpm)))
            previous = stage.tool_id
    expected_events.append(("spindle_stop", previous, 0))
    if observed_events != expected_events:
        raise ValueError("native tool-change/spindle event sequence differs")
    actual, sweeps, records = [], [], []
    for stage, group, name in zip(job.stages, groups, names):
        moves, paths = _classify(stage, group, safe_z)
        count = _edge_multiset(stage, moves)
        actual.append(moves)
        sweeps.append(_stock_stage(job, stage, moves, paths))
        records.append({"stage_id": stage.id, "mop_name": name,
                        "cutting_edges": count, "paths": len(paths),
                        "checked_moves": len(moves)})
    axial = oj._axial_evidence(job.stages, tuple(actual))
    body = occupancy.verify(job.occupancy_setup, job.stages, tuple(actual))
    composition = v.VComposition(document.design.target, tuple(sweeps))
    # BO01 defines its area metric at min(1 mm, cap). The selected bundle's
    # cached reporting scalar has no authority over this independent gate.
    depth = min(1.0, document.design.target.cap_depth)
    area = v.section_report(composition, depth)
    volume = v.volume_bounds(composition)
    if area[1] > max_residual_area_mm2 or volume[1] > max_residual_volume_mm3:
        raise ValueError("native actual residual upper bound exceeds explicit limit")
    return {"status": "pass", "document_fidelity": document_report,
        "source_sha256": document.design.source_sha256,
        "candidate_sha256": _sha(candidate), "post_sha256": _sha(post_bytes),
        "job_fingerprint": job.fingerprint,
        "cutting_path_equivalence": {"status": "pass",
            "scope": "complete per-stage undirected cutting-edge multiset",
            "tolerance_mm": _TOL, "stage_order": "preserved",
            "within_stage_order_and_direction": "may differ", "stages": tuple(records)},
        "whole_motion_safety": {"status": "pass",
            "scope": "complete supported straight native stream, including entry, rapid, link and retract",
            "recorded_initial_position_warnings": tuple(warnings)},
        "stock_access_residual": {"status": "pass",
            "scope": "actual-coordinate cylindrical/V composition from virgin stock",
            "design_fingerprint": document.design.target.fingerprint,
            "actual_composition_fingerprint": composition.fingerprint,
            "section_depth_mm": depth, "section_mm2": area, "volume_mm3": volume,
            "max_residual_area_mm2": max_residual_area_mm2,
            "max_residual_volume_mm3": max_residual_volume_mm3},
        "tool_fixture_occupancy": body, "axial_process_limits": axial,
        "operator_assumptions": {"initial_tip_xyz_mm": job.initial_tip,
            "program_frame": job.program_frame, "safe_z_mm": safe_z,
            "stock_box": job.occupancy_setup.stock.bounds,
            "fixtures": tuple((box.name, box.bounds) for box in job.occupancy_setup.fixtures),
            "tool_bodies": repr(job.occupancy_setup.tools),
            "tool_changes": "installation at posted safe tip; unencoded external travel is not inferred",
            "initial_stock": "virgin; declared setup only, no inferred wider stock"},
        "controller_and_physical_execution": "not_evaluated",
        "selected_time_cost_and_optimum": "not_transferred"}
