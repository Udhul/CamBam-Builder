"""Editable native mesh/3D finishing for one bounded bilinear relief.

CamBam owns path generation. The independent SF01 relief remains the protected
design: mesh approximation and native sampling settings never permit gouging it.
Complete Default posts are decoded and replayed, including links and retracts.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

from ...cam_core import ordered_job as ordered, relief, relief_stock
from ...native.project import CamBamProject
from ...native.reader import read_cambam_bytes
from ...native.writer import serialize_cambam_bytes
from ...native.tags import decode_tag
from .rc01_post import read_default_post


def _positive(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"positive finite {name} required")


@dataclass(frozen=True)
class ReliefMesh:
    """Upward triangles and a uniform vertical approximation bound in mm."""

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    maximum_error_mm: float
    source_fingerprint: str


def mesh_relief(target: relief.HeightField, *, subdivisions: int = 1) -> ReliefMesh:
    """Triangulate each bilinear cell on its bottom-left/top-right diagonal.

    The difference on either triangle is the cell's mixed height difference
    times a product bounded by 1/4. Subdivision reduces that bound by n squared.
    Coordinates are sampled from the exact bilinear design, not a source recipe.
    """
    if type(target) is not relief.HeightField or type(subdivisions) is not int or subdivisions < 1:
        raise ValueError("bilinear relief and positive integer subdivisions required")
    nx = (len(target.xs) - 1) * subdivisions
    ny = (len(target.ys) - 1) * subdivisions
    if (nx + 1) * (ny + 1) > 250_000:
        raise ValueError("native relief mesh vertex budget exceeded")
    xs = tuple(a + (b - a) * k / subdivisions
               for a, b in zip(target.xs, target.xs[1:])
               for k in range(subdivisions)) + (target.xs[-1],)
    ys = tuple(a + (b - a) * k / subdivisions
               for a, b in zip(target.ys, target.ys[1:])
               for k in range(subdivisions)) + (target.ys[-1],)
    for axis in (xs, ys):
        if (not all(math.isfinite(v) for v in axis) or
                any(a >= b or round(a, 9) >= round(b, 9) for a, b in zip(axis, axis[1:]))):
            raise ValueError("native relief subdivisions exceed coordinate resolution")
    vertices = tuple((x, y, target.height_at(x, y)) for y in ys for x in xs)
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, d = a + 1, a + nx + 1
            c = d + 1
            faces.extend(((a, b, c), (a, c, d)))
    error = max(abs(target.heights[j][i] - target.heights[j][i + 1]
                    - target.heights[j + 1][i] + target.heights[j + 1][i + 1])
                for j in range(len(target.ys) - 1)
                for i in range(len(target.xs) - 1)) / (4 * subdivisions ** 2)
    # XML uses nine decimal places; include its coordinate rounding and the
    # design slope converting XY rounding to a conservative vertical bound.
    slope = sum(target.gradient_bounds)
    xy_scale = max(1.0, *(abs(v) for v in (*target.xs, *target.ys)))
    z_scale = max(1.0, *(abs(v) for row in target.heights for v in row))
    # A scale-aware 64-ulp envelope covers the bounded chains of coordinate,
    # bilinear interpolation and mixed-difference arithmetic. A fixed decimal
    # padding alone would be false for large-coordinate shallow reliefs.
    error += (0.5e-9 * (1 + slope) + 64 * math.ulp(z_scale)
              + 64 * math.ulp(xy_scale) * slope)
    if not math.isfinite(error):
        raise ValueError("native relief approximation bound is not representable")
    return ReliefMesh(vertices, tuple(faces), error, target.fingerprint)


@dataclass(frozen=True)
class ReliefSettings:
    """Explicit native finish and independent replay parameters, all in mm.

    StepOver and Resolution are native fractions of the tool diameter. Neither
    is a guaranteed contact error; every actual cut still needs independent proof.
    """

    radius_mm: float = 0.25
    cutting_length_mm: float = 2.0
    stepover: float = 0.2
    resolution: float = 0.04
    allowance_mm: float = 0.02
    clearance_mm: float = 2.0
    cut_feed_mm_min: float = 240.0
    plunge_feed_mm_min: float = 120.0
    rpm: float = 10000.0
    pitch_mm: float = 0.1
    contact_tolerance_mm: float = 0.001
    max_stepdown_mm: float = 1.0
    max_entry_depth_mm: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _positive(getattr(self, name), name)
        if (self.cutting_length_mm < self.radius_mm or self.stepover > 1 or
                self.resolution > 1 or int(self.rpm) != self.rpm):
            raise ValueError("unsupported relief tool or native sampling fraction")


@dataclass(frozen=True)
class NativeReliefDocument:
    """One source-bound editable Surface and native 3D finishing operation."""

    target: relief.HeightField
    mesh: ReliefMesh
    settings: ReliefSettings
    xml_bytes: bytes
    initial_tip: tuple[float, float, float]
    subdivisions: int

    def check_document(self, candidate_bytes: bytes | None = None) -> dict:
        """Reject changed geometry, setup or process state before post replay."""
        data = self.xml_bytes if candidate_bytes is None else candidate_bytes
        if type(data) is not bytes:
            raise ValueError("complete native document bytes required")
        _check_document(self, data)
        return {"status": "pass", "source_fingerprint": self.target.fingerprint,
                "document_sha256": hashlib.sha256(data).hexdigest(),
                "mesh_error_upper_mm": self.mesh.maximum_error_mm}


def create_document(target: relief.HeightField, *, subdivisions: int = 2,
                    settings: ReliefSettings | None = None) -> NativeReliefDocument:
    """Create an editable native job; no toolpaths or post evidence are generated.

    The sole native operation uses T1, a ball nose, exact-stop straight motion
    and bounding-box finishing. Recreate/rebind after geometry or process edits.
    """
    settings = ReliefSettings() if settings is None else settings
    if type(settings) is not ReliefSettings:
        raise ValueError("relief settings required")
    mesh = mesh_relief(target, subdivisions=subdivisions)
    if settings.allowance_mm <= mesh.maximum_error_mm + settings.contact_tolerance_mm:
        raise ValueError("native allowance must exceed mesh error and contact tolerance")
    data = _author_bytes(target, mesh, settings)
    x0, y0, _, _ = target.stock_xy
    document = NativeReliefDocument(target, mesh, settings, data,
                                    (x0 - 1, y0 - 1, settings.clearance_mm), subdivisions)
    document.check_document()
    return document


def _author_bytes(target: relief.HeightField, mesh: ReliefMesh,
                  settings: ReliefSettings) -> bytes:
    project = CamBamProject("NR03 bounded relief")
    layer = project.add_layer("Relief")
    surface = project.add_surface(layer, vertices=mesh.vertices, faces=mesh.faces,
                                  identifier="relief-surface")
    x0, y0, x1, y1 = target.stock_xy
    part = project.add_part("relief-stock", stock_width=x1 - x0, stock_height=y1 - y0,
                            stock_thickness=target.stock_depth_mm, stock_offset=(x0, y0))
    mop = project.add_surface3d_mop(
        part, [surface], identifier="relief-finish", name="Relief finish",
        profile_3d_method="Horizontal", roughing_finishing="Finishing",
        stepover=settings.stepover, resolution=settings.resolution,
        tool_profile="BallNose", tool_number=1, tool_diameter=2 * settings.radius_mm,
        roughing_clearance=settings.allowance_mm, stock_surface=0,
        target_depth=-target.stock_depth_mm, depth_increment=settings.max_stepdown_mm,
        clearance_plane=settings.clearance_mm,
        cut_feedrate=settings.cut_feed_mm_min, plunge_feedrate=settings.plunge_feed_mm_min,
        spindle_speed=int(settings.rpm), spindle_direction="CW", velocity_mode="ExactStop",
        work_plane="XY", max_crossover_distance=0,
        boundary_method="BoundingBox", boundary_margin=-settings.radius_mm, backface_culling=True,
        start_corner="BottomLeft", milling_direction="Mixed", cut_ordering="LevelFirst",
        auto_arc_fitting=False)
    if surface is None or part is None or mop is None:
        raise ValueError("native relief construction failed")
    # Force feed descent from clearance rather than inheriting an external style.
    options = ET.Element("MachiningOptions")
    ET.SubElement(options, "FastPlungeHeight").text = "0"
    project._xml_machining_options = options
    return serialize_cambam_bytes(project)


def _check_document(document: NativeReliefDocument, data: bytes) -> None:
    # The native reader enforces mesh topology and supported 3D MOP schema.
    project = read_cambam_bytes(data, source_name="bound native relief")
    if project.units not in (None, "Millimeters") or len(project.list_primitives()) != 1 or len(project.list_mops()) != 1 or len(project.list_parts()) != 1:
        raise ValueError("native relief needs one mm Surface, Part and 3D MOP")
    # Compare complete native records, including preserved unmodeled settings;
    # equivalent decimal spellings are normalized, but no setting is discarded.
    if _signature(data) != _signature(document.xml_bytes):
        raise ValueError("native relief document changed; recreate after edits")
    if (type(document.settings) is not ReliefSettings or
            document.mesh != mesh_relief(document.target, subdivisions=document.subdivisions)):
        raise ValueError("native relief source revision is stale")
    if document.settings.allowance_mm <= document.mesh.maximum_error_mm + document.settings.contact_tolerance_mm:
        raise ValueError("native allowance must exceed mesh error and contact tolerance")
    x0, y0, _, _ = document.target.stock_xy
    if document.initial_tip != (x0 - 1, y0 - 1, document.settings.clearance_mm):
        raise ValueError("native relief declared initial tip changed")
    expected = _author_bytes(document.target, document.mesh, document.settings)
    if _signature(document.xml_bytes, omit_identity=True) != _signature(expected, omit_identity=True):
        raise ValueError("native relief bound design or process differs from document")


def _signature(data: bytes, *, omit_identity: bool = False) -> tuple:
    """Normalize only observed native-save defaults and display bookkeeping.

    CamBam's CADFile.DrawingUnits has an XML default of Millimeters, so native
    saves omit that attribute. This local equivalence does not alter the general
    reader's distinction between explicit and unspecified units. Actual posts
    independently require G21. Ordered vertices/faces and all machining state
    remain in the signature, including unsupported additions.
    """
    def text_value(value: str | None) -> str:
        value = (value or "").strip()
        if value.lower() in ("true", "false"):
            return value.lower()
        if re.fullmatch(r"[+\-0-9.eE ,]+", value):
            try:
                from decimal import Decimal
                return " ".join("0" if Decimal(x) == 0 else str(Decimal(x).normalize())
                                for x in re.split(r"[ ,]+", value))
            except ArithmeticError:
                pass
        return value

    def visit(node: ET.Element, parent: str = "") -> tuple:
        attributes = dict(node.attrib)
        if node.tag == "CADFile":
            attributes.pop("Name", None)  # Native Save uses the file basename.
            attributes.setdefault("units", "Millimeters")
        elif node.tag == "layer":
            attributes.setdefault("alpha", "1")
            attributes.setdefault("visible", "true")
            attributes.setdefault("locked", "false")
            if attributes.get("color") == "Green":
                attributes["color"] = "0,128,0"
        elif node.tag == "part":
            attributes.setdefault("Enabled", "true")
        elif node.tag == "mat" and attributes.get("m", "").lower() == "identity":
            attributes["m"] = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
        value = text_value(node.text)
        if node.tag in ("PMin", "PMax") and parent == "Stock" and len(value.split()) == 2:
            value += " 0"  # Native Point3F omits a zero third coordinate.
        if node.tag == "Tag":
            metadata, note = decode_tag(node.text, primitive=parent == "surface")
            if metadata:
                if omit_identity:
                    metadata.pop("internal_id", None)
                value = json.dumps((metadata, note), sort_keys=True)
        children = []
        for child in node:
            if child.tag == "ModificationCount" and node.tag in ("surface", "layer"):
                continue
            if (node.tag == "Nesting" and node.findtext("NestMethod") == "None"
                    and child.tag == "BasePoint" and not child.attrib and not len(child)
                    and not (child.text or "").strip()):
                continue
            if node.tag == "MachiningOptions":
                if child.tag == "ToolProfile" and not child.attrib and not len(child) and child.text == "Unspecified":
                    continue
                if child.tag == "Stock" and not child.attrib:
                    defaults = {"PMin": "0 0", "PMax": "0 0", "Color": "255 165 0"}
                    if (len(child) == 3 and {c.tag for c in child} == defaults.keys()
                            and all(not c.attrib and not len(c) and text_value(c.text) == defaults[c.tag] for c in child)):
                        continue
            children.append(visit(child, node.tag))
        if node.tag not in ("verts", "faces", "layers", "objects", "parts", "machineops"):
            children.sort(key=repr)
        return (node.tag, tuple(sorted((key, text_value(value)) for key, value in attributes.items())),
                value, tuple(children), text_value(node.tail))

    return visit(ET.fromstring(data))


def audit_post(document: NativeReliefDocument, post_bytes: bytes, *,
               candidate_bytes: bytes | None = None,
               max_residual_volume_mm3: float,
               max_residual_thickness_mm: float) -> dict:
    """Audit the entire native Default post against original bilinear stock.

    Raises ValueError for unsupported motion, stale documents, gouging, missing
    completion, incorrect process state or residual upper bounds above limits.
    The declared initial position is an operator assumption, not post telemetry.
    Fixtures, holders, dynamics and cutting load are outside this stock proof.
    """
    if type(document) is not NativeReliefDocument or type(post_bytes) is not bytes:
        raise ValueError("bound native relief and complete post bytes required")
    for limit in (max_residual_volume_mm3, max_residual_thickness_mm):
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit < 0:
            raise ValueError("nonnegative finite residual limits required")
    document_report = document.check_document(candidate_bytes)
    try:
        text = post_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("native post must be UTF-8") from exc
    marker = _post_section(text, document.settings)
    items, warnings = read_default_post(text, initial_position=document.initial_tip)
    if len(warnings) > 1 or any(re.fullmatch(r"line [0-9]+: initial machine position is not encoded", warning) is None for warning in warnings):
        raise ValueError(f"unsupported native post warnings: {warnings}")
    # The decoder does not assume an unknown controller's incoming G61/G64
    # state. Require an explicit G61 before *every* admitted movement; G64
    # executed movement has already produced a fatal decoder warning above.
    first_move_line = min((item["line"] for item in items if item["type"] == "move"), default=0)
    prefix = "\n".join(text.splitlines()[:max(0, first_move_line - 1)])
    prefix = re.sub(r"\([^()]*\)", "", prefix)
    if not re.search(r"(?i)G\s*61(?![0-9.])", prefix):
        raise ValueError("explicit exact-stop G61 required before native motion")
    moves = _post_moves(document, items, marker)
    s = document.settings
    op = relief_stock.ReliefOperation("relief-finish", "T1", s.radius_mm,
        s.cutting_length_mm, document.target, "parallel", s.pitch_mm, s.contact_tolerance_mm)
    stage = ordered.Stage(op.name, "T1", moves, s.rpm, operation=op,
        source_revision=document.target.fingerprint,
        axial_limits=ordered.AxialLimits(s.max_stepdown_mm, s.max_entry_depth_mm))
    stock = relief_stock.replay_stages((stage,), (moves,))
    final = stock["prefixes"][-1]
    if final["residual_volume_mm3"][1] > max_residual_volume_mm3 or final["maximum_residual_thickness_mm"][1] > max_residual_thickness_mm:
        raise ValueError("native relief residual upper bound exceeds limit")
    return {"status": "pass", "scope": "complete decoded native straight ball-relief post",
        "document": document_report, "post_sha256": hashlib.sha256(post_bytes).hexdigest(),
        "initial_tip_assumption_mm": document.initial_tip, "motion_count": len(moves),
        "rapid_retract_count": sum(m.role == "rapid_retract" for m in moves),
        "native_resolution_mm": s.resolution * 2 * s.radius_mm,
        "native_allowance_mm": s.allowance_mm, "stock": stock,
        "independent_proof_assumptions": {
            "cutting_length_mm": s.cutting_length_mm, "stock_pitch_mm": s.pitch_mm,
            "contact_tolerance_mm": s.contact_tolerance_mm,
            "max_virgin_stepdown_mm": s.max_stepdown_mm,
            "max_entry_depth_mm": s.max_entry_depth_mm},
        "limits": {"residual_volume_mm3": max_residual_volume_mm3,
                   "residual_thickness_mm": max_residual_thickness_mm},
        "fixtures_holders_dynamics_load": "not_evaluated"}


def _post_section(text: str, settings: ReliefSettings) -> int:
    nonempty = [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1) if line.strip()]
    if (len(nonempty) < 3 or nonempty[0][1] != "( Made using CamBam - http://www.cambam.co.uk )"
            or not re.fullmatch(r"\([^()]*\)", nonempty[1][1])
            or nonempty[2][1] != "( Post processor: Default )"):
        raise ValueError("complete CamBam Default header required")
    header = {i for i, _ in nonempty[:3]}
    marker = None
    for i, line in nonempty:
        if "(" not in line and ")" not in line:
            continue
        if i in header:
            continue
        if line == "( Relief finish )" and marker is None:
            marker = i
            continue
        tool = re.fullmatch(r"\(\s*T1\s*:\s*([0-9.]+)\s*\)", line)
        if tool and abs(float(tool[1]) - 2 * settings.radius_mm) <= 1e-9:
            continue
        raise ValueError(f"line {i}: unknown or duplicate native section comment")
    if marker is None:
        raise ValueError("native relief MOP section missing")
    return marker


def _post_moves(document: NativeReliefDocument, items: list, marker: int) -> tuple:
    s = document.settings
    at, running, active = document.initial_tip, False, False
    events, moves = [], []
    for item in items:
        if item["type"] == "event":
            if tuple(item["position"]) != at or abs(at[2] - s.clearance_mm) > 1e-9:
                raise ValueError("native tool/spindle event outside declared clearance")
            kind = item["kind"]
            if item["tool"] != "T1":
                raise ValueError("native relief uses wrong tool")
            if kind == "tool_change":
                if active or running:
                    raise ValueError("unexpected native relief tool change")
                active = True
            elif kind == "spindle_start":
                if not active or running or item["rpm"] != s.rpm:
                    raise ValueError("native relief spindle state differs")
                running = True
            else:
                if not running:
                    raise ValueError("native relief spindle stop without running spindle")
                running = False
            events.append(kind)
            continue
        a, b = tuple(item["start"]), tuple(item["end"])
        if item["g"] not in (0, 1):
            raise ValueError("native relief requires straight G0/G1 motion")
        if item["line"] <= marker or a != at or item["tool"] != "T1" or not active:
            raise ValueError("native motion outside bound relief section")
        vertical = a[:2] == b[:2]
        if min(a[2], b[2]) <= 0 and not running:
            raise ValueError("native stock movement without running spindle")
        if item["g"] == 0:
            if min(a[2], b[2]) >= s.clearance_mm - 1e-9:
                role = "rapid"
            elif vertical and b[2] > a[2] and abs(b[2] - s.clearance_mm) <= 1e-9:
                role = "rapid_retract"
            else:
                raise ValueError("unsafe native relief rapid below clearance")
        else:
            role = "entry" if vertical and b[2] < a[2] else "retract" if vertical else "cut"
            expected_feed = s.plunge_feed_mm_min if role == "entry" else s.cut_feed_mm_min
            if abs(item["feed"] - expected_feed) > 1e-9:
                raise ValueError("native relief feed differs")
        moves.append(ordered.JobMove(role, a, b, item["feed"]))
        at = b
    if events != ["tool_change", "spindle_start", "spindle_stop"] or running or not moves or abs(at[2] - s.clearance_mm) > 1e-9:
        raise ValueError("native relief requires complete motion and stopped clearance return")
    return tuple(moves)
