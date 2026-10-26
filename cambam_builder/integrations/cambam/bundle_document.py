"""Bind a planar native design and author an editable selected BO01 bundle.

Only straight, world-Z-zero outlines and linear millimetre jobs are supported.
The generated Engrave paths retain world XYZ without CAD parenting. Moving or
composing the document therefore requires rebinding and regenerating evidence.
Document fidelity is separate from auditing an actual CamBam post.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
import uuid
import xml.etree.ElementTree as ET

from shapely.geometry import Polygon, box

from ...cam_core import ordered_job, replay, v_region
from ...cam_extensions import bundle_search
from ...native.cad import Layer, Pline, Rect
from ...native.cam import EngraveMop, Part
from ...native.project import CamBamProject
from ...native.reader import read_cambam_bytes
from ...native.region import Region
from ...native.tags import decode_tag
from ...native.transformations import from_cambam_matrix_str
from .. import ordered_output


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _one(parent: ET.Element, name: str, *, required: bool = True) -> ET.Element | None:
    children = parent.findall(name)
    if len(children) > 1 or (required and not children):
        raise ValueError(f"bound source requires {'exactly' if required else 'at most'} one {name}")
    return children[0] if children else None


_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _point_value(value: str | None, size: int, name: str) -> tuple[float, ...]:
    fields = value.split(",") if value is not None else []
    if len(fields) != size or any(not _NUMBER.fullmatch(field.strip()) for field in fields):
        raise ValueError(f"bound source {name} requires {size} explicit numeric coordinates")
    result = tuple(float(field) for field in fields)
    if not all(math.isfinite(number) for number in result):
        raise ValueError(f"bound source {name} requires finite coordinates")
    return result


def _point_element(parent: ET.Element, name: str, size: int) -> tuple[float, ...]:
    element = _one(parent, name)
    if element.attrib or len(element):
        raise ValueError(f"bound source {name} requires a plain coordinate record")
    return _point_value(element.text, size, name)


def _explicit_stock(root: ET.Element, part: Part) -> None:
    """Verify consumed raw stock instead of accepting reader fallback values."""
    if root.tag != "CADFile":
        raise ValueError("bound native source requires a CADFile root")
    _one(root, "layers")
    parts = _one(root, "parts")
    _one(root, "MachiningOptions", required=False)
    selected = [element for element in parts if element.get("Name") == part.user_identifier]
    if len(selected) != 1:
        raise ValueError("bound source selected Part must be unambiguous")
    # A second collection could hide operations or geometry from the reader's
    # first-match reconstruction, invalidating the whole-source no-enabled-MOPs claim.
    for element in parts.findall("part"):
        _one(element, "machineops", required=False)
    for layer in root.findall("./layers/layer"):
        _one(layer, "objects")
    selected = selected[0]
    stock = _one(selected, "Stock")
    if stock.attrib or any(child.tag not in {"PMin", "PMax", "Material", "Color"}
                           for child in stock):
        raise ValueError("bound source Stock has unsupported geometry schema")
    for name in ("Material", "Color"):
        _one(stock, name, required=False)
    low = _point_element(stock, "PMin", 3)
    high = _point_element(stock, "PMax", 3)
    origin = _point_element(selected, "MachiningOrigin", 2)
    if any(a >= b for a, b in zip(low, high)):
        raise ValueError("bound source Stock requires ordered nonempty PMin/PMax bounds")
    modeled = (part.stock_offset, part.stock_width, part.stock_height,
               part.stock_surface, part.stock_thickness, part.machining_origin)
    explicit = (low[:2], high[0] - low[0], high[1] - low[1],
                high[2], high[2] - low[2], origin)
    if modeled != explicit:
        raise ValueError("bound source explicit Stock/origin differs from parsed Part")
    nesting = _one(selected, "Nesting", required=False)
    if nesting is not None:
        method = _one(nesting, "NestMethod")
        if method.attrib or len(method) or method.text != "None":
            raise ValueError("bound source selected Part requires unambiguous no nesting")


def _matrix_record(element: ET.Element) -> None:
    matrix = _one(element, "mat", required=False)
    if matrix is not None:
        if set(matrix.attrib) != {"m"} or len(matrix) or (matrix.text or "").strip():
            raise ValueError("bound source requires an unambiguous native matrix record")
        value = matrix.get("m")
        if value.strip().lower() != "identity" and any(
                not _NUMBER.fullmatch(number) for number in value.split()):
            raise ValueError("bound source matrix requires native numeric values")
        from_cambam_matrix_str(matrix.get("m"), return_z=True)


def _selected_geometry(root: ET.Element, project: CamBamProject,
                       primitive: Rect | Pline | Region) -> None:
    """Reject structural geometry/matrix records the native reader can skip."""
    elements = root.findall("./layers/layer/objects/*")

    def raw_shape(shape):
        matches = []
        for element in elements:
            metadata, _ = decode_tag(element.findtext("Tag"), primitive=True)
            identifier = metadata.get("user_id", f"{element.tag}_{element.get('id')}")
            if identifier == shape.user_identifier:
                matches.append(element)
        if len(matches) != 1:
            raise ValueError("bound source selected geometry must be unambiguous")
        return matches[0]

    shape = primitive
    while shape is not None:
        element = raw_shape(shape)
        _one(element, "Tag", required=False)
        _matrix_record(element)
        shape = project.get_parent_of_primitive(shape)
    element = raw_shape(primitive)
    if type(primitive) is Rect:
        # Native CamBam omits p at the default origin; dimensions are required
        # here so the bounded design never depends on reader scalar fallbacks.
        corner = element.get("p", "0,0,0")
        _point_value(corner, 2 if len(corner.split(",")) == 2 else 3, "Rect position")
        for name in ("w", "h"):
            _point_value(element.get(name), 1, f"Rect {name}")
        if element.get("Closed", "true").strip().lower() != "true":
            raise ValueError("bound source Rect must be closed")
    else:
        contours = [element] if type(primitive) is Pline else [
            element.find("OuterCurve"), *element.findall("./HoleCurves/Polyline")]
        for contour in contours:
            _matrix_record(contour)
            points = _one(contour, "pts")
            if points.attrib or any(point.tag != "p" or len(point) or
                                    set(point.attrib) - {"b"} for point in points):
                raise ValueError("bound source outline requires plain native point records")
            for point in points:
                count = len((point.text or "").split(","))
                _point_value(point.text, 2 if count == 2 and type(primitive) is Pline else 3,
                             "outline point")
                _point_value(point.get("b", "0"), 1, "outline bulge")


def _ring(points) -> tuple[tuple[float, float], ...]:
    if len(points) < 3:
        raise ValueError("selected outline needs a nonempty closed ring")
    result = []
    for point in points:
        if (len(point) not in (3, 4) or
                any(not math.isfinite(value) for value in point) or
                point[2] != 0 or (len(point) == 4 and point[3] != 0)):
            raise ValueError("selected outline must be straight and at world Z=0")
        result.append((point[0], point[1]))
    return tuple(result)


def _source_target(source_bytes: bytes, primitive_identifier: str,
                   part_identifier: str, cap_depth_mm: float,
                   design_angle_degrees: float, frame: str
                   ) -> tuple[CamBamProject, v_region.VTarget]:
    if (type(primitive_identifier) is not str or not primitive_identifier or
            type(part_identifier) is not str or not part_identifier or
            type(frame) is not str or not frame):
        raise ValueError("explicit primitive, Part and frame identifiers required")
    project = read_cambam_bytes(source_bytes, source_name="bound bundle source")
    if project.units != "Millimeters":
        raise ValueError("native bundle source requires explicit millimetre units")
    primitive = project.get_primitive(primitive_identifier)
    if (type(primitive) not in (Rect, Pline, Region) or
            primitive.user_identifier != primitive_identifier):
        raise ValueError("selected primitive must be a straight closed Rect, Pline or Region")
    root = ET.fromstring(source_bytes)
    _selected_geometry(root, project, primitive)
    if type(primitive) is Pline and not primitive.closed:
        raise ValueError("selected Pline must be closed")
    coordinates = primitive.get_absolute_coordinates_xyz()
    if type(primitive) is Region:
        shell = _ring(coordinates["outer_curve"])
        holes = tuple(_ring(ring) for ring in coordinates["hole_curves"])
    else:
        shell, holes = _ring(coordinates), ()
    target = v_region.VTarget.polygon(
        _sha(source_bytes), shell, holes, cap_depth_mm,
        design_angle_degrees=design_angle_degrees, frame=frame)
    if target.design_angle_degrees is None:
        raise ValueError("explicit fixed design angle required")
    part = project.get_part(part_identifier)
    if part is not None:
        _explicit_stock(root, part)
    if (part is None or part.user_identifier != part_identifier or
            not part.enabled or not part.stock_present or
            part.nesting_method != "None" or part.machining_origin != (0, 0) or
            part.stock_surface != 0 or
            part.stock_thickness < target.cap_depth):
        raise ValueError("selected enabled Part needs explicit stock, top/origin zero and no nesting")
    x, y, _ = part.stock_drawing_origin
    if (not all(math.isfinite(v) for v in (x, y, part.stock_width,
                                          part.stock_height, part.stock_thickness)) or
            part.stock_width <= 0 or part.stock_height <= 0 or
            not box(x, y, x + part.stock_width, y + part.stock_height).covers(target.safe)):
        raise ValueError("selected Part stock must contain the entire target cap")
    if any(mop.enabled for mop in project.list_mops()):
        raise ValueError("native bundle source may contain only disabled pre-existing MOPs")
    return project, target


@dataclass(frozen=True)
class NativeDesign:
    """Exact native byte snapshot and its independent fixed planar V design.

    Use :func:`bind_design` to construct this value. Dimensions are millimetres
    in ``target.frame``; the angle describes design intent, not the cutter.
    """

    source_bytes: bytes
    primitive_identifier: str
    part_identifier: str
    target: v_region.VTarget

    @property
    def source_sha256(self) -> str:
        """SHA-256 of the complete original native source bytes."""
        return _sha(self.source_bytes)

    def check_source(self, source_bytes: bytes) -> None:
        """Reject changed bytes or a design inconsistent with the source geometry."""
        if type(source_bytes) is not bytes or source_bytes != self.source_bytes:
            raise ValueError("bound native source bytes changed")
        if type(self.target) is not v_region.VTarget:
            raise ValueError("bound native design target required")
        _, target = _source_target(
            source_bytes, self.primitive_identifier, self.part_identifier,
            self.target.cap_depth, self.target.design_angle_degrees, self.target.frame)
        if target.fingerprint != self.target.fingerprint:
            raise ValueError("bound native design geometry differs from source")


def bind_design(source_bytes: bytes, primitive_identifier: str,
                part_identifier: str, *, cap_depth_mm: float,
                design_angle_degrees: float, frame: str = "program") -> NativeDesign:
    """Freeze a closed straight native outline at world Z=0 as a V design.

    Reads strictly without modifying the source. Requires explicit metric stock
    enclosing the outline and cap, zero Part origin/top, no selected-Part nesting
    and no enabled existing operations anywhere. Stock PMin/PMax, Part origin and
    selected Rect dimensions must be explicit; malformed or ambiguous native
    coordinate records and curved approximations raise ``ValueError``.
    """
    _, target = _source_target(source_bytes, primitive_identifier, part_identifier,
                               cap_depth_mm, design_angle_degrees, frame)
    return NativeDesign(source_bytes, primitive_identifier, part_identifier, target)


def _paths(stage: ordered_job.Stage) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    paths = []
    current = []
    for move in stage.motions:
        if move.arc_g:
            raise ValueError("native bundle authoring supports linear motion only")
        if move.role == "cut":
            if not current:
                current.append(move.start)
            current.append(move.end)
        elif current:
            paths.append(tuple(current))
            current = []
    if current:
        paths.append(tuple(current))
    if not paths:
        raise ValueError("each native bundle stage needs nonempty cutting paths")
    return tuple(paths)


def _feeds(stage: ordered_job.Stage) -> tuple[float, float]:
    cuts = {move.feed for move in stage.motions if move.role == "cut"}
    entries = {move.feed for move in stage.motions if move.role == "entry"}
    if len(cuts) != 1 or len(entries) != 1:
        raise ValueError("each stage requires uniform explicit cut and entry feeds")
    return next(iter(cuts)), next(iter(entries))


def _cylinder_shape(target: replay.Target) -> Polygon:
    if target.inset_per_depth or target.island or target.cone_spine:
        raise ValueError("native cylindrical stage requires the same planar source target")
    if target.region_shell:
        return Polygon(target.region_shell, target.region_holes)
    if target.polygon:
        return Polygon(target.polygon)
    return box(*target.bounds)


def _check_bundle(design: NativeDesign, bundle: bundle_search.Bundle) -> None:
    if type(design) is not NativeDesign or type(bundle) is not bundle_search.Bundle:
        raise ValueError("bound native design and selected BO01 bundle required")
    design.check_source(design.source_bytes)
    if bundle.design_fingerprint != design.target.fingerprint:
        raise ValueError("selected bundle design fingerprint differs from frozen native design")
    job = bundle.job
    if (type(job) is not ordered_job.Job or job.source_kind != "direct" or
            job.units != "mm" or job.translation_xyz_mm != (0, 0, 0) or
            job.source_fingerprint != design.source_sha256 or
            job.program_frame != design.target.frame or not job.stock_present or
            job.initial_tip[2] <= 0 or job.occupancy_setup is None):
        raise ValueError("native bundle needs a source/frame-bound direct mm job with zero translation and occupancy")
    project = read_cambam_bytes(design.source_bytes, source_name="native bundle stock")
    part = project.get_part(design.part_identifier)
    x, y, top = part.stock_drawing_origin
    native_stock = (x, y, top - part.stock_thickness,
                    x + part.stock_width, y + part.stock_height, top)
    if job.occupancy_setup.stock.bounds != native_stock:
        raise ValueError("native bundle occupancy stock must exactly match the complete source Part stock")
    for stage in job.stages:
        if (stage.offset_mm != 0 or stage.tool_length_mm != 0 or
                stage.axial_limits is None or
                stage.motions[0].start[2] != job.initial_tip[2] or
                stage.motions[-1].end[2] != job.initial_tip[2]):
            raise ValueError("native stages require zero offsets, common safe Z and explicit axial limits")
        if not re.fullmatch(r"T[1-9][0-9]*", stage.tool_id):
            raise ValueError("native stage tool must have a positive numeric Tn identifier")
        _paths(stage)
        _feeds(stage)
        if stage.v_plan is not None:
            if stage.v_plan.target.fingerprint != design.target.fingerprint:
                raise ValueError("native stage V target differs from frozen source design")
        elif (type(stage.operation) is not replay.Operation or
                stage.operation.tool.kind != "cylinder" or
                stage.operation.target.depth != design.target.cap_depth or
                stage.operation.target.bounds != design.target.safe.bounds or
                not _cylinder_shape(stage.operation.target).equals(design.target.safe)):
            raise ValueError("native cylindrical stage target differs from frozen source design")
    report = ordered_output.audit_files(
        job, bundle.dialect, bundle.files, expected_hashes=bundle.program_sha256)
    for gate in ("motion_equivalence", "stock_access_residual",
                 "tool_fixture_occupancy", "axial_process_limits"):
        if report.get(gate, {}).get("status") != "pass":
            raise ValueError(f"native bundle requires passing {gate} evidence")
    # A cylindrical replay target has no design angle. Prove those decoded
    # sweeps against the entire bound tapered design, including cylinder-only
    # bundles; a matching source hash or planar shell alone is insufficient.
    bundle_search._composition(job, bundle.files, bundle.dialect, design.target)


def _document_xml(design: NativeDesign, bundle: bundle_search.Bundle,
                  name: str) -> tuple[bytes, tuple[str, ...]]:
    if type(name) is not str or not name.strip():
        raise ValueError("nonempty generated document name required")
    project = read_cambam_bytes(design.source_bytes, source_name="native bundle source")
    root = ET.fromstring(design.source_bytes)
    layers = root.find("layers")
    parts = root.find("parts")
    if layers is None or parts is None:
        raise ValueError("native source requires layers and Parts")
    selected = [part for part in parts if part.get("Name") == design.part_identifier]
    if len(selected) != 1:
        raise ValueError("native source selected Part must be unambiguous")
    machining = root.find("MachiningOptions")
    if machining is None:
        machining = ET.Element("MachiningOptions")
        root.insert(0, machining)
    fast_plunge = machining.findall("FastPlungeHeight")
    if len(fast_plunge) > 1:
        raise ValueError("ambiguous source FastPlungeHeight")
    if fast_plunge:
        machining.remove(fast_plunge[0])
    # This is a global native property, not a MOP state/Value property. Zero
    # feeds the entry from the clearance plane rather than a lower rapid point.
    ET.SubElement(machining, "FastPlungeHeight").text = "0"
    layer_name = f"{name} bundle paths"
    if project.get_layer(layer_name) is not None:
        raise ValueError("generated native layer name conflicts with source")
    layer = Layer(user_identifier=layer_name).to_xml_element()
    objects = ET.SubElement(layer, "objects")
    layers.append(layer)
    machineops = selected[0].find("machineops")
    if machineops is None:
        machineops = ET.SubElement(selected[0], "machineops")
    ids = [int(node.get("id")) for node in root.findall("./layers/layer/objects/*")
           if node.get("id") is not None]
    next_id = max(ids, default=0) + 1
    namespace = uuid.uuid5(uuid.NAMESPACE_URL,
        f"cambam-builder:bundle:{design.source_sha256}:{bundle.job.fingerprint}:{name}")
    provenance = (f"source_sha256={design.source_sha256}; "
                  f"target_fingerprint={design.target.fingerprint}; "
                  f"job_fingerprint={bundle.job.fingerprint}; "
                  "world XYZ; repositioning/composition requires rebind")
    identifiers = []
    for index, stage in enumerate(bundle.job.stages):
        identifier = f"{name}:stage:{index + 1}:{stage.id}"
        if project.get_mop(identifier) is not None:
            raise ValueError("generated native MOP identifier conflicts with source")
        identifiers.append(identifier)
        path_ids = []
        for path_index, path in enumerate(_paths(stage), 1):
            path_identifier = f"{identifier}:path:{path_index}"
            if project.get_primitive(path_identifier) is not None:
                raise ValueError("generated native path identifier conflicts with source")
            primitive = Pline(user_identifier=path_identifier, vertices=list(path),
                              closed=False, groups=[name],
                              description=f"{provenance}; stage_id={stage.id}")
            primitive.internal_id = uuid.uuid5(namespace, path_identifier)
            primitive.output_decimals = None
            objects.append(primitive.to_xml_element(next_id, None))
            path_ids.append(next_id)
            next_id += 1
        cut_feed, entry_feed = _feeds(stage)
        if stage.v_plan is not None:
            diameter, profile = 2 * stage.v_plan.tool.maximum_radius, "VCutter"
        else:
            diameter, profile = 2 * stage.operation.tool.radius, "EndMill"
        safe_z = bundle.job.initial_tip[2]
        footer = f"G0 Z{format(Decimal(str(safe_z)), 'f')}"
        if (index + 1 < len(bundle.job.stages) and
                bundle.job.stages[index + 1].tool_id != stage.tool_id):
            footer += "\nM5"
        mop = EngraveMop(
            user_identifier=identifier, name=f"{name} {index + 1}: {stage.id}",
            enabled=True, tool_number=int(stage.tool_id[1:]),
            tool_diameter=diameter, tool_profile=profile,
            stock_surface=0, target_depth=0,
            depth_increment=stage.axial_limits.max_stepdown_mm,
            roughing_clearance=0, clearance_plane=safe_z,
            plunge_feedrate=entry_feed, cut_feedrate=cut_feed,
            spindle_direction="CW", spindle_speed=stage.rpm,
            work_plane="XY", velocity_mode="ExactStop",
            optimisation_mode="None", max_crossover_distance=0,
            final_depth_increment=0, cut_ordering="DepthFirst",
            custom_mop_footer=footer,
            native_tag=f"{provenance}; stage_id={stage.id}")
        mop.internal_id = uuid.uuid5(namespace, identifier)
        machineops.append(mop.to_xml_element(project, path_ids))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), tuple(identifiers)


_NUMERIC_TEXT = {
    "p", "PMin", "PMax", "MachiningOrigin", "BasePoint", "FastPlungeHeight",
    "TargetDepth", "DepthIncrement", "StockSurface", "RoughingClearance",
    "ClearancePlane", "SpindleSpeed", "ToolDiameter", "ToolNumber",
    "PlungeFeedrate", "CutFeedrate", "MaxCrossoverDistance", "FinalDepthIncrement",
}
_NUMERIC_ATTRIBUTES = {"p", "w", "h", "b", "m", "alpha", "pen"}


def _numbers(value: str) -> str:
    try:
        return " ".join("0" if Decimal(token) == 0 else str(Decimal(token).normalize())
                        for token in re.split(r"[ ,]+", value.strip()))
    except InvalidOperation:
        return value


def _xml_signature(data: bytes):
    # Validate safety and native relationships first, but compare the original
    # XML below: serializing this read would discard unsupported additions.
    read_cambam_bytes(data, source_name="native bundle document")
    root = ET.fromstring(data)
    id_map = {}
    for primitive in root.findall("./layers/layer/objects/*"):
        metadata, _ = decode_tag(primitive.findtext("Tag"), primitive=True)
        xml_id = primitive.get("id")
        if xml_id in id_map:
            raise ValueError("duplicate native primitive XML ID")
        id_map[xml_id] = metadata.get("internal_id", f"xml:{xml_id}")

    def signature(node: ET.Element, parent: str = ""):
        attributes = []
        for key, value in sorted(node.attrib.items()):
            if key == "id" and parent == "objects":
                value = id_map[value]
            elif key in _NUMERIC_ATTRIBUTES:
                value = _numbers(value)
            attributes.append((key, value))
        value = node.text or ""
        if list(node) and not value.strip():
            value = ""
        elif node.tag == "Tag":
            metadata, native_tag = decode_tag(value, primitive=parent not in ("engrave", "pocket", "drill", "profile"))
            if metadata:
                value = json.dumps((metadata, native_tag), sort_keys=True)
        elif node.tag == "prim" and parent == "primitive":
            value = id_map.get(value.strip(), f"unresolved:{value}")
        elif node.tag in _NUMERIC_TEXT:
            value = _numbers(value)
        children = [signature(child, node.tag) for child in node]
        if node.tag in ("objects", "primitive"):
            children.sort(key=repr)
        # Non-whitespace mixed tails are data too; unknown XML never disappears.
        tail = node.tail if node.tail and node.tail.strip() else ""
        return node.tag, tuple(attributes), value, tuple(children), tail

    return signature(root)


@dataclass(frozen=True)
class NativeDocument:
    """Editable native source plus generated Engrave stages, without post credit."""

    design: NativeDesign
    bundle: bundle_search.Bundle
    xml_bytes: bytes
    stage_mop_identifiers: tuple[str, ...]
    name: str = "Generated"

    def check_document(self, candidate_bytes: bytes | None = None, *,
                       source_bytes: bytes | None = None) -> dict[str, str]:
        """Check complete XML fidelity, source freshness and selected bundle bytes.

        Equivalent library save/reopen numeric spellings, XML-ID renumbering and
        MOP target-set order are accepted. Extra or changed XML, stage order,
        source geometry, paths and process settings fail. This grants document
        fidelity only; actual CamBam cutting/entry motion needs a separate audit.
        """
        self.design.check_source(self.design.source_bytes if source_bytes is None else source_bytes)
        _check_bundle(self.design, self.bundle)
        expected, identifiers = _document_xml(self.design, self.bundle, self.name)
        expected_signature = _xml_signature(expected)
        if (identifiers != self.stage_mop_identifiers or
                _xml_signature(self.xml_bytes) != expected_signature or
                (candidate_bytes is not None and
                 _xml_signature(candidate_bytes) != expected_signature)):
            raise ValueError("native bundle document fidelity changed")
        return {"status": "pass", "scope": "source-bound editable document fidelity",
                "source_sha256": self.design.source_sha256,
                "job_fingerprint": self.bundle.job.fingerprint,
                "actual_post": "not_evaluated"}


def author(design: NativeDesign, bundle: bundle_search.Bundle, *,
           name: str = "Generated") -> NativeDocument:
    """Author ordinary editable Engrave stages from a freshly audited BO01 bundle.

    Source CAD, Tags, layers, stock and disabled MOP XML are retained. The global
    native FastPlungeHeight is explicitly zero so entries feed from clearance.
    Occupancy must declare the complete selected Part stock box, including
    material outside the design and below its cap. No original bytes are mutated.
    Cuts remain unparented world XYZ; rebind after repositioning/composition.
    Unsupported setup, motion, targets or feeds raise ``ValueError``. Actual
    native post fidelity remains a separate gate.
    """
    _check_bundle(design, bundle)
    data, identifiers = _document_xml(design, bundle, name)
    read_cambam_bytes(data, source_name="authored bundle document")
    return NativeDocument(design, bundle, data, identifiers, name)
