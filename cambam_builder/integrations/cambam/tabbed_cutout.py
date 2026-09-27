"""Offline, source-bound evidence for the accepted rectangular Manual-tab cutout.

This deliberately bounded job has one generated interior V groove and one actual
CamBam Default Profile post. It checks retained stock at the final depth; a
Profile's skipped tool-center motion alone is not a retained-part certificate.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ...cam_core import ordered_job, replay, v_region
from .. import ordered_dialects
from .native_series import PostedMove, normalize_native_series


FORMAT = "tabbed-cutout-v1"
SETUP = {"units": "mm", "postprocessor": "Default"}
INITIAL_TIP = (0, 0, 5)
STOCK = Polygon(((0, 0), (80, 0), (80, 50), (0, 50)))
OUTLINE = ((10, 10), (70, 10), (70, 40), (10, 40))
GROOVE = ((24.7, 24.7), (55.3, 24.7), (55.3, 25.3), (24.7, 25.3))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Case:
    source: Path
    post: Path
    tab_count: int


def _source_tabs(source, expected_count):
    root = ET.fromstring(Path(source).read_bytes())
    points = root.findall("./layers/layer/objects/pline/pts/p")
    coords = tuple(tuple(float(v) for v in point.text.split(",")) for point in points)
    if (root.get("Version") != "0.9.8.0" or len(coords) != 4 or
            any(point.get("b") not in ("0", "0.0") for point in points) or
            coords != tuple((x, y, 0.0) for x, y in OUTLINE)):
        raise ValueError("cutout source is not the bounded 60 x 30 mm outline")
    part = root.find("./parts/part")
    mop = part.find("./machineops/profile") if part is not None else None
    if (part is None or mop is None or
            part.findtext("./Stock/PMin") != "0.0,0.0,-3.0" or
            part.findtext("./Stock/PMax") != "80.0,50.0,0.0" or
            mop.findtext("./InsideOutside") != "Outside" or
            mop.findtext("./TargetDepth") != "-3" or
            mop.findtext("./ToolDiameter") != "3" or
            mop.findtext("./HoldingTabs/TabMethod") != "Manual" or
            mop.findtext("./HoldingTabs/TabStyle") != "Square" or
            mop.findtext("./HoldingTabs/Width") != "6" or
            mop.findtext("./HoldingTabs/Height") != "1" or
            mop.findtext("./HoldingTabs/MinimumTabs") != str(expected_count) or
            mop.findtext("./HoldingTabs/MaximumTabs") != str(expected_count)):
        raise ValueError("cutout stock, Profile or Manual-tab setup differs")
    tabs = mop.findall("./Tabs/HoldingTab")
    if len(tabs) != expected_count or expected_count not in (4, 5):
        raise ValueError("cutout source tab count differs")
    centers = []
    for tab in tabs:
        t = float(tab.findtext("ParametricPoint"))
        nx = float(tab.findtext("./Normal/X"))
        ny = float(tab.findtext("./Normal/Y"))
        if (not 0 <= t < 1 or tab.findtext("ParentEntityID") != "1" or
                tab.findtext("NormalInverted") != "false"):
            raise ValueError("unsupported native Manual-tab record")
        distance = t * 180
        if distance < 60:
            point, normal = (10 + distance, 10), (0, -1)
        elif distance < 90:
            point, normal = (70, 10 + distance - 60), (1, 0)
        elif distance < 150:
            point, normal = (70 - (distance - 90), 40), (0, 1)
        else:
            point, normal = (10, 40 - (distance - 150)), (-1, 0)
        if (nx, ny) != normal:
            raise ValueError("Manual-tab normal differs from contour")
        centers.append((point[0] + 1.5 * nx, point[1] + 1.5 * ny,
                        nx, ny))
    if len(set(centers)) != expected_count:
        raise ValueError("duplicate Manual-tab location")
    return tuple(centers)


def _plan(source_id):
    target = v_region.VTarget.polygon(source_id, GROOVE, (), 0.5)
    tool = v_region.VProfile("pointed", 60, 0, 0.5, 0.5)
    path = v_region.VPath("fill", ((25, 25, 0.5), (55, 25, 0.5)))
    motions = (v_region.VMotion("entry", (25, 25, 5), (25, 25, -0.5)),
               v_region.VMotion("cut", (25, 25, -0.5), (55, 25, -0.5)),
               v_region.VMotion("retract", (55, 25, -0.5), (55, 25, 5)))
    plan = v_region.VPlan(target, tool, (path,), motions, 5, 0.01, 1,
                          "partial", "single bounded interior groove")
    return v_region.verify(plan)


def _generated(plan, start):
    moves = v_region.complete_motion(plan, start)
    stage = ordered_job.Stage(
        "interior-v", "T3",
        tuple(ordered_job.JobMove(m.role, m.start, m.end,
                                  0 if m.role == "rapid" else
                                  60 if m.role == "entry" else 250)
              for m in moves), 12000, v_plan=plan,
        source_revision=plan.fingerprint)
    job = ordered_job.Job(plan.target.source_id, (stage,), start)
    return job, ordered_dialects.render(job, "uccnc")[0]


def _native(case):
    source = Path(case.source)
    series = normalize_native_series(source, source, case.post,
                                     initial_position=INITIAL_TIP, setup=SETUP)
    stage = series.stages[0]
    if (len(series.stages) != 1 or stage.name != "manual-tab-fixture" or
            stage.kind != "ProfileMop" or stage.target_ids != ("outline",) or
            stage.tool != "T1" or stage.diameter_mm != 3 or
            stage.work_plane != "XY" or stage.stock_surface_mm != 0):
        raise ValueError("actual post differs from bounded native Profile")
    target = replay.Target("declared-stock", (0, 0, 80, 50), 3)
    trace = series.to_trace({stage.name: target}, {"T1": 3},
                            {stage.name: "virgin"})
    stock = replay.replay(trace, expected_source=series.evidence_fingerprint)
    return series, trace, stock


def _native_section(stock, depth):
    discs = []
    for cut in stock.cuts:
        if -cut.bottom >= depth:
            segment = (Point(cut.a) if cut.a == cut.b else
                       LineString((cut.a, cut.b)))
            # GEOS circular buffers are inscribed polygons. Inflate the
            # radius so every continuous cutter disk lies inside this cover.
            discs.append(segment.buffer((cut.tool.radius +
                                         cut.path_error_mm + 1e-6) /
                                        math.cos(math.pi / 96),
                                        quad_segs=24))
    return unary_union(discs)


def _bridges(series, stock, centers):
    links = [m for m in series.items if type(m) is PostedMove and m.g == 1 and
             m.start[2] == m.end[2] == -2 and m.start[:2] != m.end[:2] and
             abs(math.dist(m.start[:2], m.end[:2]) - 9) < 1e-6]
    if len(links) != len(centers):
        raise ValueError("posted final-depth bridge count differs from source")
    unmatched = list(centers)
    for link in links:
        center = ((link.start[0] + link.end[0]) / 2,
                  (link.start[1] + link.end[1]) / 2)
        matches = [item for item in unmatched if math.dist(item[:2], center) < 1e-5]
        if len(matches) != 1:
            raise ValueError("posted gap differs from source Manual-tab position")
        unmatched.remove(matches[0])
    # The bottom section is the relevant retained-part test: shallower passes
    # legitimately cut across all tab positions at Z=-2.
    removed = _native_section(stock, 2.5)
    remaining = STOCK.difference(removed)
    components = ([remaining] if remaining.geom_type == "Polygon" else
                  list(remaining.geoms) if remaining.geom_type == "MultiPolygon" else [])
    if not any(component.covers(Point(40, 25)) and
               component.covers(Point(2, 2)) for component in components):
        raise ValueError("cutout interior is detached from surrounding stock")
    widths = []
    for x, y, nx, ny in centers:
        cross = LineString(((x - 4 * nx, y - 4 * ny),
                            (x + 4 * nx, y + 4 * ny)))
        if removed.intersects(cross) or not STOCK.covers(cross):
            raise ValueError("bottom-depth stock bridge is missing or misplaced")
        widths.append(round(9 - 2 * 1.5, 6))
    return {"count": len(centers), "minimum_centerline_stock_width_mm":
            min(widths), "retained_part_connected": True,
            "interior_stock_area_mm2_lower": round(
                remaining.intersection(Polygon(OUTLINE)).area, 6)}


def _section_prefixes(stock, plan, order):
    """Replay both swept removals on the same declared 80 x 50 stock section."""
    prefixes, protected = {}, {}
    for depth in (0.25, 2.5):
        native = _native_section(stock, depth)
        protected_upper = native.intersection(Polygon(OUTLINE)).area
        if protected_upper > 0.2:
            raise ValueError("native Profile removes protected part stock")
        protected[str(depth)] = round(protected_upper, 6)
        if depth <= 0.5:
            a, b = plan.paths[0].points
            radius = (plan.tool.radius(a[2] - depth) + 1e-6) / math.cos(math.pi / 256)
            groove = LineString((a[:2], b[:2])).buffer(radius, quad_segs=64)
            if not plan.target.section(depth, plan.tool.tangent).covers(groove):
                raise ValueError("generated V cutter crosses interior target")
        else:
            groove = Polygon()
        if native.intersects(groove):
            raise ValueError("native and generated removals overlap unexpectedly")
        removed = {"interior-v": groove, "native-profile": native}
        cumulative = Polygon()
        rows = []
        for stage in order:
            cumulative = cumulative.union(removed[stage])
            rows.append((stage, round(STOCK.difference(cumulative).area, 6)))
        if (not 0 < rows[-1][1] < STOCK.area or
                rows[-1][1] > rows[0][1] or
                depth == 0.25 and not rows[0][1] > rows[-1][1]):
            raise ValueError("ordered stock prefixes do not retain a part")
        prefixes[str(depth)] = rows
    return prefixes, protected


def audit(case, generated_bytes, *, order=("interior-v", "native-profile")):
    """Recheck current source/post, decode every generated move and replay stock."""
    if type(case) is not Case or order not in (("interior-v", "native-profile"),
                                                ("native-profile", "interior-v")):
        raise ValueError("unsupported tabbed-cutout job or order")
    centers = _source_tabs(case.source, case.tab_count)
    series, trace, stock = _native(case)
    series.check_freshness(case.source, case.source, case.post, setup=SETUP)
    plan = _plan(series.evidence_fingerprint)
    start = INITIAL_TIP if order[0] == "interior-v" else trace.items[-2].end
    job, expected = _generated(plan, start)
    if generated_bytes != expected:
        raise ValueError("generated operation bytes differ from resolved job")
    decoded = ordered_dialects.decode((generated_bytes,), "uccnc",
                                      initial_work_tip=start)
    motion = ordered_job.audit(job, decoded, dialect="uccnc")
    if motion["motion_equivalence"]["status"] != "pass":
        raise ValueError("generated motion did not decode exactly")
    observed = ordered_job._observed_stage(job.stages[0], decoded.stages[0],
                                           (0, 0, 0))
    decoded_plan = ordered_job._decoded_v_plan(plan, observed)
    if decoded_plan.paths != plan.paths or decoded_plan.motions != plan.motions:
        raise ValueError("decoded generated V path differs")
    bridges = _bridges(series, stock, centers)
    prefixes, protected = _section_prefixes(stock, decoded_plan, order)
    section = v_region.section_report(decoded_plan, 0.25)
    if section[0] >= decoded_plan.target.section(
            0.25, decoded_plan.tool.tangent).area or section[2] > 1e-6:
        raise ValueError("generated V section lacks bounded stock removal")
    if any(m.role not in ("rapid", "entry", "cut", "retract") for m in observed):
        raise ValueError("unexpected generated entry, link or retract role")
    if order[0] == "interior-v" and observed[-1].end != INITIAL_TIP:
        raise ValueError("generated stage does not hand off at native start")
    certificate = _sha(repr((order, series.evidence_fingerprint,
                             decoded_plan.fingerprint, _sha(generated_bytes),
                             tuple((m.role, m.start, m.end) for m in observed),
                             trace.motion_fingerprint)).encode("utf-8"))
    return {"status": "pass", "order": order,
            "certificate": certificate,
            "initial_tip_assumption_xyz_mm": INITIAL_TIP,
            "source_sha256": series.source_sha256,
            "post_sha256": series.post_sha256,
            "generated_sha256": _sha(generated_bytes),
            "native_motion_fingerprint": trace.motion_fingerprint,
            "native_posted_moves": sum(type(m) is PostedMove for m in series.items),
            "native_replayed_sweeps": len(stock.cuts),
            "generated_decoded_moves": len(observed),
            "generated_roles": tuple(m.role for m in observed),
            "groove_section_0_25_mm2": section,
            "remaining_stock_prefixes_mm2": prefixes,
            "protected_part_overcut_upper_mm2": protected,
            "bridges": bridges,
            "assumption": "operator positions and installs T1/T3 between split files"}


def write_bundle(directory, case, *, order=("interior-v", "native-profile")):
    directory = Path(directory)
    if directory.exists():
        raise ValueError("tabbed-cutout output directory must be new")
    series, trace, _ = _native(case)
    start = INITIAL_TIP if order[0] == "interior-v" else trace.items[-2].end
    _, program = _generated(_plan(series.evidence_fingerprint), start)
    report = audit(case, program, order=order)
    directory.mkdir(parents=True)
    (directory / "interior-v.nc").write_bytes(program)
    manifest = {"format": FORMAT, "order": order,
                "source": str(Path(case.source).resolve()),
                "post": str(Path(case.post).resolve()),
                "tab_count": case.tab_count,
                "evidence": report}
    path = directory / "handoff.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return audit_bundle(path, case)


def audit_bundle(path, case):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("format") != FORMAT or
            manifest.get("source") != str(Path(case.source).resolve()) or
            manifest.get("post") != str(Path(case.post).resolve()) or
            manifest.get("tab_count") != case.tab_count):
        raise ValueError("tabbed-cutout source, post or setup changed")
    report = audit(case, (path.parent / "interior-v.nc").read_bytes(),
                   order=tuple(manifest.get("order", ())))
    if json.loads(json.dumps(report)) != manifest.get("evidence"):
        raise ValueError("tabbed-cutout stage order or evidence changed")
    return report
