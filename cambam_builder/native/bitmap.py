"""Editable native image references; importing/exporting never opens image files."""

from dataclasses import dataclass
import re
from typing import TypedDict
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from .core import BoundingBox, Primitive, _affine_matrix_or_none, _finite_float
from .transformations import identity_matrix


class BitmapDefinition(TypedDict):
    source_file: str
    position: tuple[float, float, float]
    width: float
    height: float
    display_bitmap: bool
    display_vectors: bool
    invert: bool
    threshold: float
    smoothing: bool
    smooth_pixels: float


@dataclass
class CADBitmap(Primitive):
    """Placed external artwork, independent of machining geometry.

    ``position`` is the lower-left XYZ corner; positive width/height are drawing
    units, independent of pixel dimensions. The literal source path is retained.
    Relative paths resolve from the native document directory. Display/vectorizer
    controls are native state, not a promise of library image tracing. XY affine
    placement and Z translation are retained; baking supports positive axis scale
    and translation only, so it cannot discard image orientation.
    """

    source_file: str = ""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    width: float = 1.0
    height: float = 1.0
    display_bitmap: bool = True
    display_vectors: bool = False
    invert: bool = False
    threshold: float = 0.4
    smoothing: bool = False
    smooth_pixels: float = 3.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate()

    def _validate(self) -> None:
        if (not isinstance(self.source_file, str) or not self.source_file.strip() or
                any(ord(c) < 32 for c in self.source_file)):
            raise ValueError("Bitmap source_file must be a nonempty plain path")
        if not isinstance(self.position, (tuple, list)) or len(self.position) != 3:
            raise ValueError("Bitmap position must contain XYZ")
        self.position = tuple(_finite_float(v, "Bitmap position") for v in self.position)
        self.width = _finite_float(self.width, "Bitmap width")
        self.height = _finite_float(self.height, "Bitmap height")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Bitmap width and height must be positive")
        for name in ("display_bitmap", "display_vectors", "invert", "smoothing"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Bitmap {name} must be a bool")
        self.threshold = _finite_float(self.threshold, "Bitmap threshold")
        self.smooth_pixels = _finite_float(self.smooth_pixels, "Bitmap smooth_pixels")
        # Installed CADBitmap.SmoothPixels is System.Single, unlike its double
        # placement fields. Refuse finite Python values that overflow native IO.
        if not 0 <= self.threshold <= 1 or not 0 <= self.smooth_pixels <= 3.4028234663852886e38:
            raise ValueError("Bitmap threshold must be in [0,1] and smoothing a nonnegative finite Single")

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> list[tuple[float, float, float]]:
        return self._calculate_absolute_geometry_xyz(total_transform, self.get_total_z_offset())

    def _calculate_absolute_geometry_xyz(self, total_transform: np.ndarray,
                                         total_z_offset: float) -> list[tuple[float, float, float]]:
        self._validate()
        matrix = _affine_matrix_or_none(total_transform)
        if matrix is None:
            raise ValueError("Expected a finite affine 3x3 matrix")
        x, y, z = self.position
        result = []
        for px, py in ((x, y), (x + self.width, y),
                       (x + self.width, y + self.height), (x, y + self.height)):
            transformed = matrix @ np.array((px, py, 1.0))
            result.append((float(transformed[0]), float(transformed[1]),
                           z + _finite_float(total_z_offset, "Bitmap Z offset")))
        if not np.isfinite(result).all():
            raise ValueError("Bitmap placement overflows")
        return result

    def _calculate_bounding_box(self, absolute_geometry: list[tuple[float, float, float]]) -> BoundingBox:
        return BoundingBox.from_points([(x, y) for x, y, _ in absolute_geometry])

    def get_geometric_center(self) -> tuple[float, float]:
        points = self.get_absolute_coordinates_xyz()
        return (points[0][0] / 2 + points[2][0] / 2,
                points[0][1] / 2 + points[2][1] / 2)

    def shift_geometry_z(self, dz: float) -> None:
        self._validate()
        x, y, z = self.position
        self.position = (x, y, _finite_float(z + _finite_float(dz, "dz"), "Bitmap Z"))

    def bake_geometry(self, transform_to_bake: np.ndarray | None = None) -> None:
        self._validate()
        matrix = _affine_matrix_or_none(
            self.effective_transform if transform_to_bake is None else transform_to_bake)
        if (matrix is None or matrix[0, 1] != 0 or matrix[1, 0] != 0 or
                matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
            raise ValueError("Bitmap bake supports positive axis scale and translation only")
        corners = self._calculate_absolute_geometry_xyz(matrix, 0)
        width = _finite_float(corners[1][0] - corners[0][0], "Baked Bitmap width")
        height = _finite_float(corners[3][1] - corners[0][1], "Baked Bitmap height")
        if width <= 0 or height <= 0:
            raise ValueError("Bitmap bake collapses its dimensions")
        self.position, self.width, self.height = corners[0], width, height
        if transform_to_bake is None:
            self.effective_transform = identity_matrix()

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: uuid.UUID | None) -> ET.Element:
        self._validate()
        self.get_absolute_coordinates_xyz()
        def number(value: float) -> str:
            return str(value if self.output_decimals is None else round(value, self.output_decimals))
        if float(number(self.width)) <= 0 or float(number(self.height)) <= 0:
            raise ValueError("Bitmap output precision collapses its dimensions")
        element = ET.Element("bitmap", {"p": ",".join(number(v) for v in self.position),
                                        "w": number(self.width), "h": number(self.height)})
        ET.SubElement(element, "ModificationCount").text = "0"
        self._add_common_xml_attributes(element, xml_primitive_id, parent_uuid)
        ET.SubElement(element, "SourceFile").text = self.source_file
        defaults = {"DisplayBitmap": (self.display_bitmap, True),
                    "DisplayVectors": (self.display_vectors, True),
                    "Invert": (self.invert, False), "Threshold": (self.threshold, 0.4),
                    "Smoothing": (self.smoothing, False), "SmoothPixels": (self.smooth_pixels, 3.0)}
        for name, (value, default) in defaults.items():
            if value != default:
                ET.SubElement(element, name).text = str(value).lower() if type(value) is bool else str(value)
        return element


def bitmap_geometry_from_xml(element: ET.Element) -> BitmapDefinition:
    """Decode the observed native bitmap schema without loading its dependency."""
    allowed = {"ModificationCount", "Tag", "mat", "SourceFile", "DisplayBitmap",
               "DisplayVectors", "Invert", "Threshold", "Smoothing", "SmoothPixels"}
    if element.tag != "bitmap" or set(element.attrib) - {"id", "p", "w", "h"}:
        raise ValueError("Unsupported Bitmap element or attributes")
    if element.text and element.text.strip():
        raise ValueError("Unsupported Bitmap text")
    seen = set()
    for child in element:
        if (child.tag not in allowed or child.tag in seen or
                (child.tail and child.tail.strip())):
            raise ValueError("Unsupported or duplicate Bitmap child")
        seen.add(child.tag)
        if child.tag not in {"Tag", "mat"} and (child.attrib or len(child)):
            raise ValueError("Unsupported Bitmap scalar encoding")
    def boolean(name: str, default: bool) -> bool:
        value = element.findtext(name)
        if value is None:
            return default
        if value.strip() not in {"true", "false", "1", "0"}:
            raise ValueError(f"Invalid Bitmap {name}")
        return value.strip() in {"true", "1"}
    counter = element.findtext("ModificationCount")
    if counter is not None:
        try:
            if re.fullmatch(r"[+-]?\d+", counter.strip()) is None or not -(2**63) <= int(counter) < 2**63:
                raise ValueError("Invalid Bitmap ModificationCount")
        except ValueError as exc:
            raise ValueError("Invalid Bitmap ModificationCount") from exc
    position = tuple(_finite_float(v, "Bitmap position") for v in element.get("p", "0,0,0").split(","))
    return dict(source_file=element.findtext("SourceFile", ""), position=position,
                width=_finite_float(element.get("w", "0"), "Bitmap width"),
                height=_finite_float(element.get("h", "0"), "Bitmap height"),
                display_bitmap=boolean("DisplayBitmap", True),
                display_vectors=boolean("DisplayVectors", True), invert=boolean("Invert", False),
                threshold=_finite_float(element.findtext("Threshold", "0.4"), "Bitmap threshold"),
                smoothing=boolean("Smoothing", False),
                smooth_pixels=_finite_float(element.findtext("SmoothPixels", "3"), "Bitmap smooth_pixels"))
