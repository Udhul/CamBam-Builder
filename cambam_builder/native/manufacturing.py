"""Compose editable native setups from caller-designed components.

This module owns document composition, not cabinet design, packing or toolpaths.
Each setup has one explicit stock batch and coordinate system. Face selection is
explicit: author separate native Parts for front/back operations, then select the
desired Parts when placing a component. All dimensions use the setup's units.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
import xml.etree.ElementTree as ET

import numpy as np

from .cam import Part, ProfileMop
from .core import Primitive
from .project import CamBamProject, Identifiable
from .writer import build_xml_tree


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


@dataclass(frozen=True)
class SetupStock:
    """One caller-chosen rectangular stock batch, expressed in drawing units.

    ``origin`` is its lower-left XY corner; ``surface`` is its top Z. The
    material label, thickness and surface must match every selected source Part.
    Width, height and origin replace the source component's blank dimensions.
    This declaration does not check packing, overlap, clamps or cutting access.
    """

    width: float
    height: float
    thickness: float
    material: str
    origin: tuple[float, float] = (0.0, 0.0)
    surface: float = 0.0

    def __post_init__(self) -> None:
        for name in ("width", "height", "thickness"):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        _name(self.material, "material")
        if not isinstance(self.origin, (tuple, list)) or len(self.origin) != 2:
            raise ValueError("origin must contain two finite coordinates")
        object.__setattr__(self, "origin", tuple(_finite(v, "origin") for v in self.origin))
        object.__setattr__(self, "surface", _finite(self.surface, "surface"))


@dataclass(frozen=True)
class SetupFrame:
    """Map source drawing XY into a setup: reflect X, rotate, then translate.

    Reflection negates X about source X=0. For a board of width W, the common
    back-face mapping ``(x, y) -> (W-x, y)`` is ``SetupFrame(x=W, reflect_x=True)``.
    Rotation is counterclockwise in degrees. This planar placement leaves Z and
    all MOP values unchanged: depths and coordinate-bearing MOP parameters must
    already describe the destination setup. It is not a 3D workpiece flip or a
    transformation of custom G-code, toolpaths or stock-removal evidence.
    """

    x: float = 0.0
    y: float = 0.0
    rotation_degrees: float = 0.0
    reflect_x: bool = False

    def __post_init__(self) -> None:
        for name in ("x", "y", "rotation_degrees"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if type(self.reflect_x) is not bool:
            raise ValueError("reflect_x must be a boolean")

    def matrix(self) -> np.ndarray:
        """Return a fresh 3x3 planar rigid/reflected affine matrix."""
        angle = math.radians(self.rotation_degrees)
        c, s = math.cos(angle), math.sin(angle)
        sign = -1.0 if self.reflect_x else 1.0
        return np.array([[sign * c, -s, self.x],
                         [sign * s, c, self.y], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class ComponentPlacement:
    """Select a component subtree and ordered machining Parts for one placement.

    ``name`` scopes every copied entity and group as ``name/original``. Selected
    Parts must have nonempty operations targeting only this subtree. All subtree
    geometry is retained as editable design context, including unmachined faces.
    Part order follows ``parts``; operation order within each Part is preserved.
    The source remains caller-owned and is read afresh on each composition.
    """

    source: CamBamProject
    root: Identifiable
    name: str
    parts: tuple[Identifiable, ...]
    frame: SetupFrame = SetupFrame()

    def __post_init__(self) -> None:
        if not isinstance(self.source, CamBamProject):
            raise TypeError("source must be a CamBamProject")
        _name(self.name, "placement name")
        if not isinstance(self.frame, SetupFrame):
            raise TypeError("frame must be a SetupFrame")
        if isinstance(self.parts, (str, bytes)) or not isinstance(self.parts, Sequence):
            raise TypeError("parts must be a sequence of Part identifiers")
        if not self.parts:
            raise ValueError("a placement must select at least one Part")
        object.__setattr__(self, "parts", tuple(self.parts))


def _component_tree(source: CamBamProject, root: Identifiable) -> list[Primitive]:
    primitive = source.get_primitive(root)
    if primitive is None:
        raise ValueError("component root must belong to the source project")
    pending = [primitive]
    seen = set()
    result = []
    while pending:
        current = pending.pop()
        if current.internal_id in seen:
            raise ValueError("component hierarchy contains a cycle")
        seen.add(current.internal_id)
        result.append(current)
        pending.extend(source.get_children_of_primitive(current))
    return result


def _selected_parts(placement: ComponentPlacement, stock: SetupStock,
                    primitives: list[Primitive]) -> list[Part]:
    source = placement.source
    primitive_ids = {p.internal_id for p in primitives}
    selected = []
    selected_ids = set()
    for identifier in placement.parts:
        part = source.get_part(identifier)
        if part is None or part.internal_id in selected_ids:
            raise ValueError("selected Parts must exist and be distinct")
        selected_ids.add(part.internal_id)
        if (not part.stock_present or part.stock_material != stock.material
                or part.stock_thickness != stock.thickness or part.stock_surface != stock.surface):
            raise ValueError("selected Part stock must match setup material, thickness and surface")
        if tuple(part.machining_origin) != (0.0, 0.0) or part.nesting_method != "None":
            raise ValueError("components require zero machining origin and no native nesting")
        mops = source.get_mops_in_part(part)
        if not mops:
            raise ValueError("selected Parts must contain machining operations")
        for mop in mops:
            if isinstance(mop, ProfileMop) and mop.tab_method == "Manual":
                raise ValueError("manual holding tabs require placement-aware authoring outside compose_setup")
            targets = set(source.get_mop_targets(mop))
            if not targets or not targets <= primitive_ids:
                raise ValueError("every selected operation must target only the component subtree")
        selected.append(part)
    return selected


def compose_setup(name: str, stock: SetupStock,
                  placements: Sequence[ComponentPlacement], *,
                  units: str = "Millimeters") -> CamBamProject:
    """Build an independent native document in caller-specified placement order.

    Sources and their entities are never mutated, including on failure. Copies
    have fresh UUIDs and scoped identifiers/groups; inherited geometry transforms
    and MOP bindings remain intact. Each copied Part retains its machining defaults
    and receives the common setup stock at zero machining origin. Source document
    default tool diameters and native MachiningOptions XML must agree; the common
    context is preserved so inherited MOP values retain their meaning. Units must
    be explicit and equal; nothing
    is converted. Missing selections, collisions and incompatible stock/setup
    context raise ``ValueError``. Manual holding tabs are unsupported here because
    their native records require placement-aware reconstruction. No partial
    document is returned on failure.

    Recompose after source edits, or edit the returned native project directly.
    Existing derived paths/evidence are not transferred or certified by this API.
    Stock placement, face registration, tools and manufacturing order remain
    caller decisions. CamBam generates final toolpaths from the native document.
    """
    _name(name, "setup name")
    if not isinstance(stock, SetupStock):
        raise TypeError("stock must be a SetupStock")
    if units not in ("Millimeters", "Inches"):
        raise ValueError("setup units must be explicit Millimeters or Inches")
    placements = tuple(placements)
    if not placements:
        raise ValueError("a setup must contain at least one placement")
    names = set()
    for placement in placements:
        if not isinstance(placement, ComponentPlacement):
            raise TypeError("placements must contain ComponentPlacement values")
        if placement.name in names:
            raise ValueError("placement names must be distinct within a setup")
        names.add(placement.name)

    default_tool = placements[0].source.default_tool_diameter
    result = CamBamProject(name, default_tool_diameter=default_tool, units=units)
    result.output_decimals = max(p.source.output_decimals for p in placements)
    default_context = build_xml_tree(result).getroot().find("MachiningOptions")
    context = getattr(placements[0].source, "_xml_machining_options", default_context)
    context_bytes = ET.tostring(context) if context is not None else None
    result._xml_machining_options = deepcopy(context)
    for placement in placements:
        source = placement.source
        if source.units != units:
            raise ValueError("source and setup units must match explicitly")
        if source.default_tool_diameter != default_tool:
            raise ValueError("source document default tool diameters must agree")
        source_context = getattr(source, "_xml_machining_options", default_context)
        source_context_bytes = ET.tostring(source_context) if source_context is not None else None
        if source_context_bytes != context_bytes:
            raise ValueError("source document MachiningOptions contexts must agree")
        primitives = _component_tree(source, placement.root)
        parts = _selected_parts(placement, stock, primitives)
        entities = {p.internal_id: p for p in primitives}
        groups = set()
        for primitive in primitives:
            layer = source.get_layer_of_primitive(primitive)
            if layer is None:
                raise ValueError("component primitives must belong to layers")
            entities[layer.internal_id] = layer
            groups.update(source.get_groups_of_primitive(primitive))
        for part in parts:
            entities[part.internal_id] = part
            for mop in source.get_mops_in_part(part):
                entities[mop.internal_id] = mop

        # Select on an independent graph so the existing transactional tree-copy
        # owner can enforce identity, closure and naming invariants unchanged.
        selected = source.clone()
        part_ids = [part.internal_id for part in parts]
        for part in selected.list_parts():
            if part.internal_id not in part_ids:
                selected.remove_part(part, remove_contained_mops=True)
        selected._part_order = part_ids
        mapping = selected.copy_primitive_tree(
            primitives[0].internal_id, result, preserve_ids=False,
            identifier_map={e.user_identifier: f"{placement.name}/{e.user_identifier}"
                            for e in entities.values()},
            group_map={group: f"{placement.name}/{group}" for group in groups},
        )
        if not result.transform_primitive(mapping[primitives[0].internal_id], placement.frame.matrix()):
            raise ValueError("component frame could not be applied")
        for primitive in primitives:
            copied = result.get_primitive(mapping[primitive.internal_id])
            if not np.isfinite(copied.get_total_transform()).all():
                raise ValueError("component frame produced non-finite world coordinates")
        for part in parts:
            copied_part = result.get_part(mapping[part.internal_id])
            copied_part.stock_width = stock.width
            copied_part.stock_height = stock.height
            copied_part.stock_offset = stock.origin
    return result
