"""Bounded native Manual-tab documents and independently sectioned posted stock.

This consumer supports one untransformed rectangular Outside Profile. It does
not enable general straight-ramp replay or certify physical holding strength.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from xml.etree import ElementTree as ET

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ... import CBProject
from ...cam_core.replay import arc_segments
from ...native.reader import read_cambam_bytes
from ...native.tags import decode_tag
from ...native.writer import serialize_cambam_bytes
from .rc01_post import read_default_post


OUTLINE = ((10.0, 10.0), (70.0, 10.0), (70.0, 40.0), (10.0, 40.0))
DEFAULT_POINTS = ((40.0, 10.0), (70.0, 25.0), (40.0, 40.0), (10.0, 25.0))
INITIAL_TIP = (0.0, 0.0, 5.0)
RADIUS = 1.5
HALF_GAP = 4.5
TOLERANCE = 1e-6
SECTION_DEPTHS = (2.25, 2.5, 2.75, 3.0)
STOCK = Polygon(((0, 0), (80, 0), (80, 50), (0, 50)))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class NativeTabDocument:
    """A fresh Manual-tab source binding; recreate it after intended edits.

    Dimensions, depth and cutter are fixed by this consumer. ``points`` are
    drawing XY positions on the counterclockwise source perimeter, in mm.
    """

    style: str
    points: tuple[tuple[float, float], ...]
    xml_bytes: bytes

    def check_document(self, candidate_bytes: bytes | None = None) -> dict:
        """Check a complete source or library round trip against bound intent."""
        data = self.xml_bytes if candidate_bytes is None else candidate_bytes
        if type(data) is not bytes or type(self.xml_bytes) is not bytes:
            raise ValueError("complete native tab document bytes required")
        expected = _author_bytes(self.style, self.points)
        if _signature(self.xml_bytes, identities=False) != _signature(expected, identities=False):
            raise ValueError("bound native tab intent changed; recreate after edits")
        project = read_cambam_bytes(data, strict=True)
        if (len(project.list_primitives()) != 1 or len(project.list_parts()) != 1
                or len(project.list_mops()) != 1
                or _signature(data) != _signature(self.xml_bytes)):
            raise ValueError("native tab document changed; recreate after edits")
        return {"status": "pass", "document_sha256": _sha(data),
                "style": self.style, "tab_count": len(self.points)}


def create_document(*, style: str = "Triangle",
                    points: tuple[tuple[float, float], ...] = DEFAULT_POINTS) -> NativeTabDocument:
    """Author one editable 60 x 30 mm cutout, with four or five Manual tabs.

    The 3 mm endmill cuts 3 mm stock in 1 mm levels, with 6 x 1 mm tab
    parameters. Points must satisfy the native writer's corner/overlap limits;
    its 1e-6 mm edge-distance tolerance projects points onto the straight edge.
    Only Square and Triangle are admitted. This creates no toolpaths or G-code.
    """
    if style not in ("Square", "Triangle"):
        raise ValueError("bounded native tabs require Square or Triangle style")
    try:
        points = tuple(tuple(float(v) for v in point) for point in points)
    except (TypeError, ValueError) as exc:
        raise ValueError("finite drawing XY tab points required") from exc
    if (len(points) not in (4, 5) or any(len(p) != 2 or
            any(not math.isfinite(v) for v in p) for p in points)):
        raise ValueError("four or five finite drawing XY tab points required")
    document = NativeTabDocument(style, points, _author_bytes(style, points))
    document.check_document()
    return document


def _author_bytes(style: str, points: tuple[tuple[float, float], ...]) -> bytes:
    if style not in ("Square", "Triangle") or len(points) not in (4, 5):
        raise ValueError("unsupported bounded native tab intent")
    project = CBProject(f"manual-{style.lower()}-tabs")
    outline = project.add_pline("Geometry", OUTLINE, closed=True, identifier="outline")
    part = project.add_part("Part", stock_width=80.0, stock_height=50.0,
                            stock_thickness=3.0, stock_surface=0.0)
    project.add_profile_mop(part, [outline], identifier="manual-tab-fixture",
        name="manual-tab-fixture", profile_side="Outside", target_depth=-3,
        depth_increment=1, stock_surface=0, tool_diameter=3, tool_number=1,
        spindle_speed=12000, cut_feedrate=250, plunge_feedrate=80,
        clearance_plane=5, roughing_clearance=0, lead_in_type="None",
        tab_method="Manual", tab_style=style, tab_width=6, tab_height=1,
        tab_min_tabs=len(points), tab_max_tabs=len(points),
        manual_tab_points=points)
    return serialize_cambam_bytes(project)


def _signature(data: bytes, *, identities: bool = True) -> tuple:
    """Keep every XML field, normalizing numbers and framework UUIDs only."""
    def value(text: str | None) -> str:
        text = (text or "").strip()
        if re.fullmatch(r"[+\-0-9.eE ,]+", text):
            try:
                return " ".join(str(Decimal(v).normalize()) if Decimal(v) else "0"
                                for v in re.split(r"[ ,]+", text))
            except InvalidOperation:
                pass
        return text

    def visit(node: ET.Element, parent: str = "") -> tuple:
        text = value(node.text)
        if node.tag == "Tag" and not identities:
            metadata, note = decode_tag(node.text, primitive=parent == "pline")
            if metadata:
                metadata.pop("internal_id", None)
                text = json.dumps((metadata, note), sort_keys=True)
        attributes = tuple(sorted((k, value(v)) for k, v in node.attrib.items()))
        return (node.tag, attributes, text,
                tuple(visit(child, node.tag) for child in node))
    return visit(ET.fromstring(data))


def _tabs_by_edge(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, ...], ...]:
    result: list[list[float]] = [[], [], [], []]
    for x, y in points:
        if abs(y - 10) <= TOLERANCE:
            edge, q = 0, x - 10
        elif abs(x - 70) <= TOLERANCE:
            edge, q = 1, y - 10
        elif abs(y - 40) <= TOLERANCE:
            edge, q = 2, 70 - x
        elif abs(x - 10) <= TOLERANCE:
            edge, q = 3, 40 - y
        else:
            raise ValueError("tab point is not on the bounded outline")
        result[edge].append(q)
    return tuple(tuple(sorted(items)) for items in result)


def _edge_point(edge: int, q: float) -> tuple[float, float]:
    return ((10 + q, 8.5), (71.5, 10 + q),
            (70 - q, 41.5), (8.5, 40 - q))[edge]


def _edge_move(a: tuple, b: tuple) -> tuple[int, float, float] | None:
    for edge, length in enumerate((60, 30, 60, 30)):
        start, end = _edge_point(edge, 0), _edge_point(edge, length)
        axis = 0 if edge in (0, 2) else 1
        fixed = 1 - axis
        if abs(a[fixed] - start[fixed]) > TOLERANCE or abs(b[fixed] - start[fixed]) > TOLERANCE:
            continue
        sign = 1 if edge < 2 else -1
        qa, qb = sign * (a[axis] - start[axis]), sign * (b[axis] - start[axis])
        if min(qa, qb) >= -TOLERANCE and max(qa, qb) <= length + TOLERANCE:
            return edge, max(0.0, qa), min(float(length), qb)
    return None


def _depth_function(style: str, centers: tuple[float, ...], q: float) -> float:
    if style == "Square":
        return -2.0 if any(abs(q - c) < HALF_GAP - TOLERANCE for c in centers) else -3.0
    return -3.0 + max((max(0.0, 1 - abs(q - c) / HALF_GAP) for c in centers), default=0.0)


def _check_edge_depth(style: str, centers: tuple[float, ...], qa: float, qb: float,
                      za: float, zb: float, level: int) -> None:
    if level < 3:
        if abs(za + level) > TOLERANCE or abs(zb + level) > TOLERANCE:
            raise ValueError("posted preliminary level differs from native depth increment")
        return
    samples = [0.0, 1.0, 0.5]
    # Endpoint checks alone cannot distinguish a missing tab or a chord through
    # its apex. Every depth-function breakpoint inside this move is checked.
    for center in centers:
        for breakpoint in (center - HALF_GAP, center, center + HALF_GAP):
            if qa < breakpoint < qb:
                samples.append((breakpoint - qa) / (qb - qa))
    for t in samples:
        q, z = qa + t * (qb - qa), za + t * (zb - za)
        expected = _depth_function(style, centers, q)
        if style == "Square" and t in (0.0, 1.0):
            if any(abs(abs(q - c) - HALF_GAP) <= TOLERANCE for c in centers):
                if min(abs(z + 2), abs(z + 3)) <= TOLERANCE:
                    continue
        if abs(z - expected) > TOLERANCE:
            raise ValueError("posted tab ramp, height or location differs from source")


def _add_interval(coverage: dict, level: int, component: int,
                  start: float, end: float) -> None:
    if end <= start + TOLERANCE:
        raise ValueError("posted cut reverses or repeats the bounded contour")
    intervals = coverage[level][component]
    if any(min(end, hi) - max(start, lo) > TOLERANCE for lo, hi in intervals):
        raise ValueError("extra posted contour traversal")
    intervals.append((start, end))


def _complete(coverage: dict, level: int) -> bool:
    for intervals, length in zip(coverage[level], (60, 30, 60, 30, *([math.pi / 2] * 4))):
        cursor = 0.0
        for lo, hi in sorted(intervals):
            if abs(lo - cursor) > TOLERANCE:
                return False
            cursor = hi
        if abs(cursor - length) > TOLERANCE:
            return False
    return True


def _corner_move(item: dict) -> tuple[int, float, float]:
    a, b, center = item["start"], item["end"], item["center"]
    for corner, expected in enumerate(((70, 10), (70, 40), (10, 40), (10, 10))):
        if math.dist(center, expected) > TOLERANCE:
            continue
        angles = []
        for point in (a, b):
            if abs(math.dist(point[:2], expected) - RADIUS) > TOLERANCE:
                raise ValueError("posted corner radius differs from cutter offset")
            angle = (math.atan2(point[1] - center[1], point[0] - center[0])
                     - (-math.pi / 2 + corner * math.pi / 2)) % (2 * math.pi)
            if abs(angle - 2 * math.pi) < TOLERANCE:
                angle = 0.0
            if angle > math.pi / 2 + TOLERANCE:
                raise ValueError("posted arc leaves bounded offset corner")
            angles.append(angle)
        return corner + 4, angles[0], angles[1]
    raise ValueError("posted arc has an unsupported corner center")


def _section_segment(a: tuple, b: tuple, depth: float) -> tuple[tuple, tuple] | None:
    """Clip a straight cylindrical sweep exactly at a horizontal stock plane."""
    threshold = -depth
    if min(a[2], b[2]) > threshold:
        return None
    if max(a[2], b[2]) <= threshold:
        return a[:2], b[:2]
    t = (threshold - a[2]) / (b[2] - a[2])
    crossing = tuple(x + t * (y - x) for x, y in zip(a[:2], b[:2]))
    return (a[:2], crossing) if a[2] <= threshold else (crossing, b[:2])


def _section_removal(moves: list[dict], depth: float) -> Polygon:
    covers = []
    for item in moves:
        a, b = tuple(item["start"]), tuple(item["end"])
        error = 0.0
        if item["g"] in (2, 3):
            if a[2] > -depth:
                continue
            segments, error = arc_segments(a, b, tuple(item["center"]), item["g"])
        else:
            clipped = _section_segment(a, b, depth)
            if clipped is None:
                continue
            segments = (clipped,)
        # Circumscribed polygonal buffers cover each continuous cutter disk.
        # Arc chord error is added before the circular-cover inflation.
        radius = (RADIUS + error + 1e-7) / math.cos(math.pi / 96)
        for start, end in segments:
            path = Point(start) if start == end else LineString((start, end))
            covers.append(path.buffer(radius, quad_segs=24))
    return unary_union(covers)


def _stock_sections(document: NativeTabDocument, moves: list[dict]) -> list[dict]:
    sections = []
    for depth in SECTION_DEPTHS:
        removed = _section_removal(moves, depth)
        remaining = STOCK.difference(removed)
        components = ([remaining] if remaining.geom_type == "Polygon" else
                      list(remaining.geoms) if remaining.geom_type == "MultiPolygon" else [])
        connected = any(component.covers(Point(40, 25)) and
                        component.covers(Point(2, 2)) for component in components)
        overcut = removed.intersection(Polygon(OUTLINE)).area
        if overcut > 0.2:
            raise ValueError("posted cutter removes protected cutout stock")
        expected_width = (6.0 if document.style == "Square" else
                          max(0.0, 9 * (depth - 2) - 3))
        widths = []
        for edge, centers in enumerate(_tabs_by_edge(document.points)):
            for q in centers:
                center = Point(_edge_point(edge, q))
                tangent = LineString((_edge_point(edge, q - HALF_GAP),
                                      _edge_point(edge, q + HALF_GAP)))
                retained = tangent.difference(removed)
                pieces = ([retained] if retained.geom_type == "LineString" else
                          list(retained.geoms) if retained.geom_type == "MultiLineString" else [])
                width = next((piece.length for piece in pieces if piece.distance(center) < 1e-8), 0.0)
                if abs(width - expected_width) > 0.01:
                    raise ValueError("posted retained bridge width differs from independent section reference")
                widths.append(width)
        if connected != (expected_width > 0):
            raise ValueError("posted section connectivity differs from tab stock reference")
        sections.append({"depth_mm": depth,
            "minimum_bridge_width_mm_lower": math.floor(min(widths) * 1e6) / 1e6,
            "ideal_minimum_bridge_width_mm": expected_width,
            "retained_part_connected": connected,
            "protected_part_overcut_mm2_upper": math.ceil(overcut * 1e6) / 1e6})
    return sections


def _post_marker(text: str) -> int:
    lines = [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1)
             if line.strip()]
    if (len(lines) < 3 or lines[0][1] != "( Made using CamBam - http://www.cambam.co.uk )"
            or not re.fullmatch(r"\([^()]*\)", lines[1][1])
            or lines[2][1] != "( Post processor: Default )"):
        raise ValueError("complete CamBam Default post header required")
    marker = None
    for number, line in lines[3:]:
        if "(" not in line and ")" not in line:
            continue
        if line == "( manual-tab-fixture )" and marker is None:
            marker = number
        elif re.fullmatch(r"\(\s*T1\s*:\s*3(?:\.0*)?\s*\)", line):
            continue
        else:
            raise ValueError("unknown or repeated native tab post section")
    if marker is None:
        raise ValueError("native Manual-tab Profile section missing")
    return marker


def _post_motion(document: NativeTabDocument, text: str) -> tuple[list[dict], list[dict]]:
    marker = _post_marker(text)
    items, warnings = read_default_post(text, allow_arcs=True, initial_position=INITIAL_TIP)
    if len(warnings) > 1 or any(re.fullmatch(
            r"line [0-9]+: initial machine position is not encoded", warning) is None
            for warning in warnings):
        raise ValueError(f"unsupported posted execution state: {warnings}")
    first_move = min((item["line"] for item in items if item["type"] == "move"), default=0)
    prefix = re.sub(r"\([^()]*\)", "", "\n".join(text.splitlines()[:first_move - 1]))
    if not re.search(r"(?i)G\s*61(?![0-9.])", prefix):
        raise ValueError("explicit exact-stop G61 required before native motion")
    coverage = {level: [[] for _ in range(8)] for level in (1, 2, 3)}
    centers = _tabs_by_edge(document.points)
    at, active, running, level = INITIAL_TIP, False, False, 0
    events, moves, cuts = [], [], []
    for item in items:
        if item["type"] == "event":
            if tuple(item["position"]) != at or at[2] != 5 or item["tool"] != "T1":
                raise ValueError("native tab process event differs from declared clearance or tool")
            kind = item["kind"]
            if kind == "tool_change":
                if active or running:
                    raise ValueError("unexpected native tab tool change")
                active = True
            elif kind == "spindle_start":
                if not active or running or item["rpm"] != 12000:
                    raise ValueError("native tab spindle differs from bound process")
                running = True
            elif not running:
                raise ValueError("native tab spindle stopped without running")
            else:
                running = False
            events.append(kind)
            continue
        a, b = tuple(item["start"]), tuple(item["end"])
        if a != at or item["tool"] != "T1" or not active or item["line"] <= marker:
            raise ValueError("native tab motion is discontinuous or outside its Profile")
        if not running:
            raise ValueError("native tab motion requires its running spindle")
        g, vertical = item["g"], a[:2] == b[:2]
        if g == 0:
            if min(a[2], b[2]) >= 5:
                pass
            elif vertical and min(a[2], b[2]) >= 0:
                pass
            elif vertical and b[2] == 5 and b[2] > a[2] and _complete(coverage, 3):
                pass
            elif vertical and document.style == "Square" and level == 3 and a[2] == -3 and b[2] == -2:
                edge = _edge_move(a, a)
                if edge is None or not any(abs(edge[1] - (c - HALF_GAP)) < TOLERANCE for c in centers[edge[0]]):
                    raise ValueError("Square tab rapid lift differs from source")
            else:
                raise ValueError("unsupported native tab rapid below clearance")
        elif g in (1, 2, 3):
            expected_feed = 80 if vertical and b[2] < a[2] else 250
            if item["feed"] != expected_feed:
                raise ValueError("native tab feed differs from bound process")
            if vertical:
                if g != 1 or b[2] >= a[2]:
                    raise ValueError("unsupported native tab vertical feed")
                edge = _edge_move(a, a)
                if edge is None:
                    raise ValueError("native entry is outside bounded contour")
                if level < 3 and abs(b[2] + level + 1) <= TOLERANCE:
                    if level and not _complete(coverage, level):
                        raise ValueError("native level ends before completing its contour")
                    level += 1
                elif document.style == "Square" and level == 3 and a[2] == -2 and b[2] == -3:
                    if not any(abs(edge[1] - (c + HALF_GAP)) < TOLERANCE for c in centers[edge[0]]):
                        raise ValueError("Square tab descent differs from source")
                else:
                    raise ValueError("native tab depth increment or entry differs")
            elif g == 1:
                edge = _edge_move(a, b)
                if edge is None or level == 0:
                    raise ValueError("posted straight cut leaves bounded offset contour")
                index, qa, qb = edge
                if qb <= qa + TOLERANCE:
                    raise ValueError("posted cut reverses bounded contour")
                _check_edge_depth(document.style, centers[index], qa, qb, a[2], b[2], level)
                _add_interval(coverage, level, index, qa, qb)
            else:
                if g != 3 or level == 0 or a[2] != b[2] or abs(a[2] + level) > TOLERANCE:
                    raise ValueError("unsupported native tab corner direction or depth")
                index, qa, qb = _corner_move(item)
                _add_interval(coverage, level, index, qa, qb)
            cuts.append(item)
        else:
            raise ValueError("unsupported native tab interpolation")
        moves.append(item)
        at = b
    if (events != ["tool_change", "spindle_start", "spindle_stop"] or running
            or at[2] != 5 or not all(_complete(coverage, d) for d in (1, 2, 3))):
        raise ValueError("native tab post lacks complete levels or stopped clearance return")
    return cuts, moves


def audit_post(document: NativeTabDocument, post_bytes: bytes, *,
               candidate_bytes: bytes | None = None) -> dict:
    """Audit a complete actual Default post against this bounded tab source.

    Every cut must cover the prescribed three-level offset contour exactly once,
    with the requested tab shape. All process events, feeds and travel are
    checked. Continuous cylindrical ramp sweeps are clipped independently at
    four stock depths. Returned hashes bind the current complete input bytes;
    call again after file changes. Initial positioning, fixtures, controller
    behavior, cutting loads and physical holding strength remain unassessed.
    """
    if type(document) is not NativeTabDocument or type(post_bytes) is not bytes:
        raise ValueError("bound native tab document and complete post bytes required")
    checked = document.check_document(candidate_bytes)
    try:
        text = post_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("native tab post must be UTF-8") from exc
    cuts, moves = _post_motion(document, text)
    sections = _stock_sections(document, cuts)
    certificate = _sha(json.dumps((checked["document_sha256"], _sha(post_bytes),
                                   sections, INITIAL_TIP), sort_keys=True).encode("utf-8"))
    return {"status": "pass", "scope": "bounded Manual rectangular native tab post and section stock",
        "certificate": certificate, "source_sha256": _sha(document.xml_bytes),
        "candidate_sha256": checked["document_sha256"], "post_sha256": _sha(post_bytes),
        "style": document.style, "tab_count": len(document.points),
        "initial_tip_assumption_mm": INITIAL_TIP, "native_posted_moves": len(moves),
        "native_ramp_moves": sum(m["start"][:2] != m["end"][:2] and
                                 m["start"][2] != m["end"][2] for m in cuts),
        "sections": sections, "retained_part_connected_at_bottom": sections[-1]["retained_part_connected"],
        "section_cover_tolerance_mm": 0.002,
        "physical_holding_strength_and_runtime": "not_evaluated"}
