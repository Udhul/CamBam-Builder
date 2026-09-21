"""Ordinary CAD containers and primitive entities."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import json
import logging
import math
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .cad_transformations import (
    identity_matrix, apply_transform, get_transformed_point,
    to_cambam_matrix_str, from_cambam_matrix_str,
)
from .entity_core import (
    CURVE_POINT_TOLERANCE, PLINE_BULGE_TOLERANCE,
    BoundingBox, CamBamEntity, Primitive, Vertex, VertexInput,
    _affine_matrix_or_none, _finite_float, _local_arc_bounding_box,
    _normalize_vertices, _validate_stored_vertices, _xy_similarity_scale,
)

if TYPE_CHECKING:
    from .cambam_project import CamBamProject

logger = logging.getLogger(__name__)

@dataclass
class Layer(CamBamEntity):
    """Represents a drawing layer with visual properties."""
    color: str = 'Green'
    alpha: float = 1.0
    pen_width: float = 1.0
    visible: bool = True
    locked: bool = False
    # Note: No primitive_ids or _xml_objects_element here. Managed by Project.

    def to_xml_element(self) -> ET.Element:
        """Creates the <layer> XML element (without the <objects> container)."""
        layer_elem = ET.Element("layer", {
            "name": self.user_identifier, # Use user_identifier as CamBam's layer name
            "color": self.color,
            "alpha": str(self.alpha),
            "pen": str(self.pen_width),
            "visible": str(self.visible).lower(),
            "locked": str(self.locked).lower()
        })
        # The <objects> sub-element will be added by the writer based on project registry
        return layer_elem
@dataclass
class Pline(Primitive):
    # Intrinsic geometry
    vertices: List[Vertex] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.vertices = _normalize_vertices(self.vertices, allow_bulge=True)
        self._validated_vertices()

    def _validated_vertices(self) -> List[Vertex]:
        return _validate_stored_vertices(self.vertices, allow_bulge=True)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float, float]]:
        # Extract XY for transformation
        vertices = self._validated_vertices()
        rel_pts_xy = [(vertex.x, vertex.y) for vertex in vertices]
        abs_pts_xy = apply_transform(rel_pts_xy, total_transform)

        # Re-attach bulge values (bulge is not transformed by standard matrix)
        abs_pts_with_bulge = []
        for i, (x, y) in enumerate(abs_pts_xy):
            # Handle potential index out of bounds if apply_transform skipped points
            if i < len(vertices):
                abs_pts_with_bulge.append((x, y, vertices[i].bulge))
            else:
                logger.warning(f"Point mismatch after transformation for Pline {self.user_identifier}. Skipping bulge.")

        return abs_pts_with_bulge

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> List[Tuple[float, float, float, float]]:
        vertices = self._validated_vertices()
        absolute_xy_bulge = self._calculate_absolute_geometry(total_transform)
        if (any(abs(p[2]) > PLINE_BULGE_TOLERANCE for p in absolute_xy_bulge)
                and _xy_similarity_scale(total_transform) is None):
            raise ValueError("Bulged Pline XYZ queries require an XY similarity transform")
        orientation = -1.0 if np.linalg.det(total_transform[:2, :2]) < 0.0 else 1.0
        return [
            (x, y, _finite_float(vertices[index].z + total_z_offset, "world Z"),
             bulge * orientation)
            for index, (x, y, bulge) in enumerate(absolute_xy_bulge)
        ]

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        vertices = self._validated_vertices()
        self.vertices = [
            Vertex(vertex.x, vertex.y, _finite_float(vertex.z + delta, "shifted vertex Z"),
                   bulge=vertex.bulge)
            for vertex in vertices
        ]

    def _calculate_bounding_box(self, absolute_geometry: List[Tuple[float, float, float]]) -> BoundingBox:
        # Bounding box ignores bulge, uses only XY coordinates
        points_xy = [(p[0], p[1]) for p in absolute_geometry]
        if not points_xy:
            return BoundingBox()
        # TODO: Bulges affect the bounding box! This is an approximation.
        # A proper calculation would need to find the extrema of the arcs.
        if any(abs(p[2]) > 1e-6 for p in absolute_geometry):
            logger.debug(f"Bounding box for Pline {self.user_identifier} with bulges is approximate.")
        return BoundingBox.from_points(points_xy)

    def get_bounding_box(self) -> BoundingBox:
        """Return exact bounds for line and bulge segments in world space.

        Bulge arcs are kept in local coordinates and transformed as affine
        parametric curves.  This is necessary because a general affine map
        turns a circular bulge into an ellipse; transforming a sampled or
        endpoint-only absolute representation cannot recover its extrema.
        """
        try:
            matrix = _affine_matrix_or_none(self.get_total_transform())
        except (TypeError, ValueError):
            matrix = None
        if matrix is None or not self.vertices:
            return BoundingBox()
        try:
            vertices = self._validated_vertices()
        except (TypeError, ValueError):
            return BoundingBox()

        local_points = []
        for vertex in vertices:
            try:
                x, y = float(vertex.x), float(vertex.y)
            except (TypeError, ValueError, OverflowError):
                return BoundingBox()
            if not (math.isfinite(x) and math.isfinite(y)):
                return BoundingBox()
            local_points.append((x, y))

        # A point-only/open one-point Pline still has a useful point bound.
        if len(local_points) == 1:
            transformed = matrix[:2, :2] @ np.asarray(local_points[0]) + matrix[:2, 2]
            if not np.isfinite(transformed).all():
                return BoundingBox()
            return BoundingBox.from_points([(float(transformed[0]), float(transformed[1]))])

        segment_count = len(local_points) - 1 + (1 if self.closed else 0)
        bounds = BoundingBox()
        for index in range(segment_count):
            next_index = (index + 1) % len(local_points)
            p0 = local_points[index]
            p1 = local_points[next_index]
            raw_bulge = vertices[index].bulge
            try:
                bulge = float(raw_bulge)
            except (TypeError, ValueError, OverflowError):
                return BoundingBox()
            if not math.isfinite(bulge):
                return BoundingBox()

            point0 = np.asarray(p0, dtype=float)
            point1 = np.asarray(p1, dtype=float)
            # Half-deltas and half-sums avoid overflowing when opposite finite
            # endpoints span more than the largest representable float.
            half_chord = point1 * 0.5 - point0 * 0.5
            half_chord_length = math.hypot(float(half_chord[0]), float(half_chord[1]))
            if (half_chord_length <= CURVE_POINT_TOLERANCE * 0.5
                    or abs(bulge) <= PLINE_BULGE_TOLERANCE):
                segment_points = apply_transform((p0, p1), matrix)
                if len(segment_points) != 2:
                    return BoundingBox()
                segment_box = BoundingBox.from_points(segment_points)
            else:
                # CamBam's bulge is tan(included_angle / 4), with positive
                # values sweeping counter-clockwise from p0 to p1.
                sweep = 4.0 * math.atan(bulge)
                midpoint = point0 * 0.5 + point1 * 0.5
                left_normal = np.asarray(
                    (-half_chord[1], half_chord[0])
                ) / half_chord_length
                inverse_bulge = 1.0 / bulge
                offset = half_chord_length * ((inverse_bulge - bulge) / 2.0)
                center = midpoint + left_normal * offset
                radius = half_chord_length * (
                    (abs(bulge) + abs(inverse_bulge)) / 2.0
                )
                start = math.atan2(p0[1] - center[1], p0[0] - center[0])
                segment_box = _local_arc_bounding_box(
                    (float(center[0]), float(center[1])), radius, start, sweep, matrix
                )
            if not segment_box.is_valid():
                return BoundingBox()
            bounds = bounds.union(segment_box)
        return bounds

    def get_geometric_center(self) -> Tuple[float, float]:
        # Use bounding box center as geometric center
        bbox = self.get_bounding_box()
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        elif self.vertices:
            # Fallback to first point if bbox is invalid (e.g., single point pline)
            abs_coords = self.get_absolute_coordinates()
            if abs_coords:
                return abs_coords[0][0], abs_coords[0][1]
        return (0.0, 0.0) # Default fallback

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to vertices.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        affine = _affine_matrix_or_none(transform_to_apply)
        if affine is None:
            raise ValueError("Expected a finite affine 3x3 matrix")
        transform_to_apply = affine
        vertices = self._validated_vertices()
        segment_count = max(0, len(vertices) - 1)
        if self.closed and vertices:
            segment_count += 1
        has_bulged_segment = any(
            abs(_finite_float(
                vertices[index].bulge,
                f"vertices[{index}].bulge",
            )) > PLINE_BULGE_TOLERANCE
            for index in range(segment_count)
        )
        if has_bulged_segment and _xy_similarity_scale(transform_to_apply) is None:
            raise ValueError(
                "Bulged Pline geometry can only bake XY similarity transforms"
            )
        reflected = np.linalg.det(transform_to_apply[0:2, 0:2]) < 0.0
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Transform XY coordinates
            rel_pts_xy = [(vertex.x, vertex.y) for vertex in vertices]
            baked_pts_xy = apply_transform(rel_pts_xy, transform_to_apply)

            # Rebuild relative points with original bulge values
            new_vertices = []
            for i, (x, y) in enumerate(baked_pts_xy):
                if i < len(vertices):
                    bulge = vertices[i].bulge
                    if reflected:
                        bulge = -bulge
                    new_vertices.append(Vertex(x, y, vertices[i].z, bulge=bulge))
                else:
                    logger.warning(f"Point mismatch during baking for Pline {self.user_identifier}. Skipping point.")

            self.vertices = new_vertices
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Pline {self.user_identifier}: {e}")


    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <pline> XML element."""
        pline_elem = ET.Element("pline", {"Closed": str(self.closed).lower()})

        # Add points WITHOUT applying the effective_transform here
        # The matrix added later by _add_common_xml_attributes handles the total transform
        pts_elem = ET.SubElement(pline_elem, "pts")
        vertices = self._validated_vertices()
        for vertex in vertices:
            # CamBam point format: x,y,z (z is usually 0 for 2D)
            x = round(vertex.x, self.output_decimals) if self.output_decimals is not None else vertex.x
            y = round(vertex.y, self.output_decimals) if self.output_decimals is not None else vertex.y
            z = vertex.z
            z = round(z, self.output_decimals) if self.output_decimals is not None else z
            bulge = vertex.bulge
            bulge = round(bulge, self.output_decimals) if self.output_decimals is not None else bulge
            
            ET.SubElement(pts_elem, "p", {"b": str(bulge)}).text = f"{x},{y},{z}"

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(pline_elem, xml_primitive_id, parent_uuid)
        return pline_elem


@dataclass
class Circle(Primitive):
    # Intrinsic geometry
    relative_center: Tuple[float, float] = (0.0, 0.0)
    diameter: float = 1.0
    elevation: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.elevation = _finite_float(self.elevation, "elevation")

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        """Returns absolute center and scaled diameter."""
        abs_center = get_transformed_point(self.relative_center, total_transform)

        # Calculate scaled diameter - average scaling factor from matrix columns
        # Assumes uniform scaling or takes an average for non-uniform
        sx = np.linalg.norm(total_transform[:, 0]) # Length of transformed x-axis vector
        sy = np.linalg.norm(total_transform[:, 1]) # Length of transformed y-axis vector
        avg_scale = (sx + sy) / 2.0
        abs_diameter = self.diameter * avg_scale
        if not math.isclose(sx, sy):
            logger.warning(f"Circle {self.user_identifier} transformed with non-uniform scale ({sx:.3f}, {sy:.3f}). Using average scale for diameter calculation.")

        return {"center": abs_center, "diameter": abs_diameter}

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> Dict[str, Any]:
        if _xy_similarity_scale(total_transform) is None:
            raise ValueError("Circle XYZ queries require an XY similarity transform")
        geometry = self._calculate_absolute_geometry(total_transform)
        center_x, center_y = geometry["center"]
        geometry["center"] = (
            center_x, center_y,
            _finite_float(_finite_float(self.elevation, "elevation") + total_z_offset, "world Z"),
        )
        return geometry

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        shifted = _finite_float(
            _finite_float(self.elevation, "elevation") + delta,
            "shifted elevation",
        )
        self.elevation = shifted

    def _calculate_bounding_box(self, absolute_geometry: Dict[str, Any]) -> BoundingBox:
        cx, cy = absolute_geometry["center"]
        radius = absolute_geometry["diameter"] / 2.0
        return BoundingBox(cx - radius, cy - radius, cx + radius, cy + radius)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Center is simply the transformed relative center
        return get_transformed_point(self.relative_center, self.get_total_transform())

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to center and diameter.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        similarity_scale = _xy_similarity_scale(transform_to_apply)
        if similarity_scale is None:
            raise ValueError("Circle geometry can only bake XY similarity transforms")
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake center position
            self.relative_center = get_transformed_point(self.relative_center, transform_to_apply)

            # Bake diameter (using average scale factor)
            self.diameter *= similarity_scale
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Circle {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <circle> XML element."""
        cx = round(self.relative_center[0], self.output_decimals) if self.output_decimals is not None else self.relative_center[0]
        cy = round(self.relative_center[1], self.output_decimals) if self.output_decimals is not None else self.relative_center[1]
        cz = _finite_float(self.elevation, "elevation")
        cz = round(cz, self.output_decimals) if self.output_decimals is not None else cz
        c_diam = round(self.diameter, self.output_decimals) if self.output_decimals is not None else self.diameter
        
        circle_elem = ET.Element("circle", {
            "c": f"{cx},{cy},{cz}", # Center (x,y,z)
            "d": str(c_diam) # Diameter
         })
        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(circle_elem, xml_primitive_id, parent_uuid)
        return circle_elem


@dataclass
class Rect(Primitive):
    # Intrinsic geometry defined by corner, width, height
    # Note: CamBam rect XML uses 'p' (corner), 'w', 'h' but seems to store it as a Pline internally.
    # We'll keep w/h definition for ease of use but might need adjustments based on CamBam behavior.
    relative_corner: Tuple[float, float] = (0.0, 0.0) # Bottom-left
    width: float = 1.0
    height: float = 1.0
    elevation: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.elevation = _finite_float(self.elevation, "elevation")

    def _get_relative_corners(self) -> List[Tuple[float, float]]:
        """Returns the four corners in relative coordinates."""
        x0, y0 = self.relative_corner
        return [(x0, y0), (x0 + self.width, y0), (x0 + self.width, y0 + self.height), (x0, y0 + self.height)]

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        """Returns the four corners in absolute coordinates."""
        return apply_transform(self._get_relative_corners(), total_transform)

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> List[Tuple[float, float, float]]:
        z = _finite_float(_finite_float(self.elevation, "elevation") + total_z_offset, "world Z")
        return [
            (x, y, z)
            for x, y in self._calculate_absolute_geometry(total_transform)
        ]

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        shifted = _finite_float(
            _finite_float(self.elevation, "elevation") + delta,
            "shifted elevation",
        )
        self.elevation = shifted

    def _calculate_bounding_box(self, absolute_geometry: List[Tuple[float, float]]) -> BoundingBox:
        if not absolute_geometry:
            return BoundingBox()
        return BoundingBox.from_points(absolute_geometry)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Calculate relative center
        rel_cx = self.relative_corner[0] + self.width / 2.0
        rel_cy = self.relative_corner[1] + self.height / 2.0
        # Transform relative center to absolute
        return get_transformed_point((rel_cx, rel_cy), self.get_total_transform())

    def is_rectangular_after_transform(self) -> bool:
        """
        Determines if the rectangle's geometry is still rectangular after applying
        the effective transformation.
        
        Returns:
            True if the transformed corners still form a rectangle, False otherwise
        """
        if np.allclose(self.effective_transform, identity_matrix()):
            return True  # With identity transform, it's definitely rectangular
        
        # Get the rectangle corners
        corners = self._get_relative_corners()
        
        # Transform the corners
        transformed_corners = apply_transform(corners, self.effective_transform)
        
        # Check if the transformed corners still form a rectangle
        # This requires adjacent sides to be perpendicular
        if len(transformed_corners) == 4:
            # Calculate vectors for adjacent sides
            v1 = (transformed_corners[1][0] - transformed_corners[0][0], 
                  transformed_corners[1][1] - transformed_corners[0][1])
            v2 = (transformed_corners[3][0] - transformed_corners[0][0], 
                  transformed_corners[3][1] - transformed_corners[0][1])
            v3 = (transformed_corners[2][0] - transformed_corners[1][0],
                  transformed_corners[2][1] - transformed_corners[1][1])
            v4 = (transformed_corners[2][0] - transformed_corners[3][0],
                  transformed_corners[2][1] - transformed_corners[3][1])
            
            # Calculate dot products to check perpendicularity
            dot1 = v1[0]*v2[0] + v1[1]*v2[1]
            dot2 = v2[0]*v4[0] + v2[1]*v4[1]
            dot3 = v4[0]*v3[0] + v4[1]*v3[1]
            dot4 = v3[0]*v1[0] + v3[1]*v1[1]
            
            # All dot products should be close to 0 for a rectangle
            return (math.isclose(dot1, 0, abs_tol=1e-10) and
                    math.isclose(dot2, 0, abs_tol=1e-10) and
                    math.isclose(dot3, 0, abs_tol=1e-10) and
                    math.isclose(dot4, 0, abs_tol=1e-10))
        
        return False
    
    def to_pline_representation(self) -> Pline:
        """
        Creates a Pline representation of this rectangle, applying any transformations.
        
        Returns:
            A new Pline object representing the same geometry with the transformation applied
        """
        # Get the rectangle corners
        corners = self._get_relative_corners()
        
        # Transform the corners
        transformed_corners = apply_transform(corners, self.effective_transform)
        
        # Create points for Pline (adding the first point again to close the loop if needed)
        pline_points = [Vertex(p[0], p[1], self.elevation) for p in transformed_corners]
        
        # Create a new Pline
        pline = Pline(
            user_identifier=f"{self.user_identifier}_as_pline",
            groups=self.groups.copy() if self.groups else [],
            description=f"Converted from Rect: {self.description}",
            effective_transform=identity_matrix(),  # Use identity since we've already applied the transform
            vertices=pline_points,
            closed=True,
            local_z_offset=self.local_z_offset,
        )
        
        return pline

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """Bake the outline, converting this object to a closed Pline if needed.

        Existing references and all common metadata survive conversion. Rect-only
        geometry attributes cease to exist. An explicit matrix leaves the local
        effective transform intact; an implicit bake resets it to identity.
        """
        matrix = self.effective_transform if transform_to_bake is None else transform_to_bake
        if np.iscomplexobj(matrix):
            raise ValueError("Expected a finite affine 3x3 matrix")
        matrix = np.asarray(matrix, dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.array_equal(matrix[2], [0., 0., 1.])):
            raise ValueError("Expected a finite affine 3x3 matrix")

        corners = np.asarray(apply_transform(self._get_relative_corners(), matrix))
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("Rect baking produced invalid corners")
        lower = corners.min(axis=0)
        upper = corners.max(axis=0)
        bounds = np.array([lower, (upper[0], lower[1]), upper, (lower[0], upper[1])])
        # Compare the complete closed outline, not perpendicularity or bounds.
        # Absolute tolerance only: large coordinates must not hide a shear.
        axis_aligned = any(
            np.allclose(corners, np.roll(order, shift, axis=0), rtol=0, atol=1e-12)
            for order in (bounds, bounds[::-1]) for shift in range(4)
        )
        if axis_aligned:
            self.relative_corner = tuple(lower)
            self.width, self.height = upper - lower
        else:
            elevation = _finite_float(self.elevation, "elevation")
            vertices = [Vertex(x, y, elevation) for x, y in corners]
            # Both dataclasses have the same ordinary Python object layout.
            # Assign the class before geometry edits so an unsupported subclass
            # layout fails without discarding the original Rect geometry.
            self.__class__ = Pline
            self.vertices = vertices
            self.closed = True
            del self.relative_corner, self.width, self.height, self.elevation
        if transform_to_bake is None:
            self.effective_transform = identity_matrix()

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """
        Creates an XML element for this Rect.
        If the geometry is not rectangular after transformation, converts to a Pline representation.
        """
        # Check if we need to convert to Pline for XML output
        if not self.is_rectangular_after_transform():
            # Serialize the complete world outline with an identity matrix.
            # ``to_pline_representation`` intentionally bakes only this Rect's
            # local matrix; using it here loses ancestor transforms because the
            # temporary Pline is not linked to a project.
            pline_repr = Pline(
                user_identifier=self.user_identifier,
                groups=self.groups.copy() if self.groups else [],
                description=self.description,
                output_decimals=self.output_decimals,
                effective_transform=identity_matrix(),
                local_z_offset=self.get_total_z_offset(),
                vertices=[Vertex(x, y, self.elevation) for x, y in apply_transform(
                    self._get_relative_corners(), self.get_total_transform())],
                closed=True,
            )
            
            # Get the Pline's XML element, but with our metadata
            pline_elem = pline_repr.to_xml_element(xml_primitive_id, parent_uuid)
            
            # Update the Tag element to maintain our identity (but no rect-specific data)
            tag_data = {
                "user_id": self.user_identifier,
                "internal_id": str(self.internal_id),
                "groups": self.groups,
                "description": self.description
            }
            
            # Add parent reference if provided
            if parent_uuid:
                tag_data["parent"] = str(parent_uuid)
            
            # Update or add the Tag element
            tag_node = pline_elem.find("Tag")
            if tag_node is None:
                tag_node = ET.SubElement(pline_elem, "Tag")
            tag_node.text = json.dumps(tag_data)

            # Log that this Rect was converted to Pline
            logger.info(f"Rect '{self.user_identifier}' converted to Pline for XML output due to non-rectangular geometry.")

            return pline_elem
        
        # Otherwise, create a normal Rect XML element
        x = round(self.relative_corner[0], self.output_decimals) if self.output_decimals is not None else self.relative_corner[0]
        y = round(self.relative_corner[1], self.output_decimals) if self.output_decimals is not None else self.relative_corner[1]
        z = _finite_float(self.elevation, "elevation")
        z = round(z, self.output_decimals) if self.output_decimals is not None else z
        w = round(self.width, self.output_decimals) if self.output_decimals is not None else self.width
        h = round(self.height, self.output_decimals) if self.output_decimals is not None else self.height
        
        rect_elem = ET.Element("rect", {
            "Closed": "true", # Rectangles are implicitly closed
            "p": f"{x},{y},{z}", # Corner (x,y,z)
            "w": str(w),
            "h": str(h)
        })
        
        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(rect_elem, xml_primitive_id, parent_uuid)
        
        return rect_elem


@dataclass
class Arc(Primitive):
    # Intrinsic geometry
    relative_center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    start_angle: float = 0.0 # Degrees
    extent_angle: float = 90.0 # Degrees (sweep angle)
    elevation: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.elevation = _finite_float(self.elevation, "elevation")

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        """Returns absolute center, scaled radius, and transformed angles."""
        abs_center = get_transformed_point(self.relative_center, total_transform)

        # Calculate scaled radius (similar to Circle)
        sx = np.linalg.norm(total_transform[:, 0])
        sy = np.linalg.norm(total_transform[:, 1])
        avg_scale = (sx + sy) / 2.0
        abs_radius = self.radius * avg_scale

        # Calculate transformed start angle
        # Get the transformation's rotation component in degrees
        # atan2(m10, m00) gives the rotation angle
        rotation_rad = math.atan2(total_transform[1, 0], total_transform[0, 0])
        rotation_deg = math.degrees(rotation_rad)
        abs_start_angle = (self.start_angle + rotation_deg) % 360

        # Extent angle remains unchanged by rotation/translation/uniform scale
        # Non-uniform scale would distort the arc into an elliptical arc, not handled here.
        abs_extent_angle = self.extent_angle
        if not math.isclose(sx, sy):
            logger.warning(f"Arc {self.user_identifier} transformed with non-uniform scale ({sx:.3f}, {sy:.3f}). Extent angle might be inaccurate.")

        return {
            "center": abs_center,
            "radius": abs_radius,
            "start_angle": abs_start_angle,
            "extent_angle": abs_extent_angle
        }

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> Dict[str, Any]:
        if _xy_similarity_scale(total_transform) is None:
            raise ValueError("Arc XYZ queries require an XY similarity transform")
        geometry = self._calculate_absolute_geometry(total_transform)
        center_x, center_y = geometry["center"]
        geometry["center"] = (
            center_x, center_y,
            _finite_float(_finite_float(self.elevation, "elevation") + total_z_offset, "world Z"),
        )
        start = math.radians(self.start_angle)
        direction = total_transform[:2, :2] @ np.array([math.cos(start), math.sin(start)])
        geometry["start_angle"] = math.degrees(math.atan2(direction[1], direction[0])) % 360
        geometry["extent_angle"] = self.extent_angle * (
            -1.0 if np.linalg.det(total_transform[:2, :2]) < 0.0 else 1.0
        )
        return geometry

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        shifted = _finite_float(
            _finite_float(self.elevation, "elevation") + delta,
            "shifted elevation",
        )
        self.elevation = shifted

    def _calculate_bounding_box(self, absolute_geometry: Dict[str, Any]) -> BoundingBox:
        # Approximate bounding box using center and radius
        # TODO: A precise bounding box needs to consider the actual arc sweep.
        cx, cy = absolute_geometry["center"]
        radius = absolute_geometry["radius"]
        logger.debug(f"Bounding box for Arc {self.user_identifier} is approximate (using full circle).")
        return BoundingBox(cx - radius, cy - radius, cx + radius, cy + radius)

    def get_bounding_box(self) -> BoundingBox:
        """Return exact world bounds of the directed circular sweep."""
        try:
            matrix = _affine_matrix_or_none(self.get_total_transform())
        except (TypeError, ValueError):
            matrix = None
        if matrix is None:
            return BoundingBox()
        try:
            center = (float(self.relative_center[0]), float(self.relative_center[1]))
            radius = float(self.radius)
            start = math.radians(float(self.start_angle))
            sweep = math.radians(float(self.extent_angle))
        except (TypeError, ValueError, OverflowError, IndexError):
            return BoundingBox()
        return _local_arc_bounding_box(center, radius, start, sweep, matrix)

    def get_geometric_center(self) -> Tuple[float, float]:
        # For simplicity, use the transformed center of the arc's circle
        # TODO: A more accurate center would be the midpoint of the arc chord or centroid.
        return get_transformed_point(self.relative_center, self.get_total_transform())

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to center, radius, and angles.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        similarity_scale = _xy_similarity_scale(transform_to_apply)
        if similarity_scale is None:
            raise ValueError("Arc geometry can only bake XY similarity transforms")
        transform_to_apply = _affine_matrix_or_none(transform_to_apply)
        assert transform_to_apply is not None
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake center position
            self.relative_center = get_transformed_point(self.relative_center, transform_to_apply)
            
            self.radius *= similarity_scale

            # Transform the actual start direction. This also handles a
            # reflected similarity without guessing its mirror axis.
            start_radians = math.radians(self.start_angle)
            start_vector = np.asarray(
                (math.cos(start_radians), math.sin(start_radians)), dtype=float
            )
            transformed_start = transform_to_apply[0:2, 0:2] @ start_vector
            self.start_angle = math.degrees(math.atan2(
                transformed_start[1], transformed_start[0]
            )) % 360
            if np.linalg.det(transform_to_apply[0:2, 0:2]) < 0:
                self.extent_angle = -self.extent_angle
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Arc {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <arc> XML element."""
        cx = round(self.relative_center[0], self.output_decimals) if self.output_decimals is not None else self.relative_center[0]
        cy = round(self.relative_center[1], self.output_decimals) if self.output_decimals is not None else self.relative_center[1]
        cz = _finite_float(self.elevation, "elevation")
        cz = round(cz, self.output_decimals) if self.output_decimals is not None else cz
        radius = round(self.radius, self.output_decimals) if self.output_decimals is not None else self.radius
        start_angle = round(self.start_angle % 360, self.output_decimals) if self.output_decimals is not None else self.start_angle % 360
        extent_angle = round(self.extent_angle, self.output_decimals) if self.output_decimals is not None else self.extent_angle
        
        arc_elem = ET.Element("arc", {
            "p": f"{cx},{cy},{cz}", # Center (x,y,z)
            "r": str(radius),       # Radius
            "s": str(start_angle),  # Start Angle (degrees)
            "w": str(extent_angle)  # Sweep Angle (degrees)
        })

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(arc_elem, xml_primitive_id, parent_uuid)
        return arc_elem


@dataclass
class Points(Primitive):
    # Intrinsic geometry
    vertices: List[Vertex] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.vertices = _normalize_vertices(self.vertices, allow_bulge=False)

    def _validated_vertices(self) -> List[Vertex]:
        return _validate_stored_vertices(self.vertices, allow_bulge=False)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        return apply_transform(
            [(vertex.x, vertex.y) for vertex in self._validated_vertices()],
            total_transform,
        )

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> List[Tuple[float, float, float]]:
        vertices = self._validated_vertices()
        return [
            (x, y, _finite_float(vertices[index].z + total_z_offset, "world Z"))
            for index, (x, y) in enumerate(
                self._calculate_absolute_geometry(total_transform)
            )
        ]

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        vertices = self._validated_vertices()
        self.vertices = [
            Vertex(vertex.x, vertex.y, _finite_float(vertex.z + delta, "shifted vertex Z"))
            for vertex in vertices
        ]

    def _calculate_bounding_box(self, absolute_geometry: List[Tuple[float, float]]) -> BoundingBox:
        if not absolute_geometry:
            return BoundingBox()
        return BoundingBox.from_points(absolute_geometry)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Use bounding box center
        bbox = self.get_bounding_box()
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        elif self.vertices:
            abs_coords = self.get_absolute_coordinates()
            if abs_coords:
                return abs_coords[0] # Fallback to first point
        return (0.0, 0.0)

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to vertices.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        affine = _affine_matrix_or_none(transform_to_apply)
        if affine is None:
            raise ValueError("Expected a finite affine 3x3 matrix")
        transform_to_apply = affine
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Transform the points
            vertices = self._validated_vertices()
            transformed = apply_transform(
                [(vertex.x, vertex.y) for vertex in vertices], transform_to_apply
            )
            self.vertices = [
                Vertex(x, y, vertices[index].z)
                for index, (x, y) in enumerate(transformed)
            ]
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Points {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <points> XML element."""
        points_elem = ET.Element("points")
        pts_elem = ET.SubElement(points_elem, "pts")
        for vertex in self._validated_vertices():
            x, y, pz = vertex.x, vertex.y, vertex.z
            px = round(x, self.output_decimals) if self.output_decimals is not None else x
            py = round(y, self.output_decimals) if self.output_decimals is not None else y
            pz = round(pz, self.output_decimals) if self.output_decimals is not None else pz
            # CamBam point format: x,y,z (z is usually 0)
            ET.SubElement(pts_elem, "p").text = f"{px},{py},{pz}"

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(points_elem, xml_primitive_id, parent_uuid)
        return points_elem


@dataclass
class Text(Primitive):
    # Intrinsic properties
    text_content: str = "Text"
    relative_position: Tuple[float, float] = (0.0, 0.0) # Anchor point
    # CamBam's MText API documents p2 as currently unused. Preserve the optional
    # serialized value without assigning unverified drawing semantics.
    xml_p2_position: Optional[Tuple[float, float]] = None
    elevation: float = 0.0
    xml_p2_elevation: Optional[float] = None
    height: float = 10.0 # Font height in drawing units
    font: str = 'Arial'
    style: str = '' # e.g., 'bold', 'italic', 'bold,italic'
    line_spacing: float = 1.0 # Multiplier
    align_horizontal: str = 'center' # 'left', 'center', 'right'
    align_vertical: str = 'center' # 'top', 'center', 'bottom'

    def __post_init__(self):
        super().__post_init__()
        self.elevation = _finite_float(self.elevation, "elevation")
        if self.xml_p2_elevation is not None:
            self.xml_p2_elevation = _finite_float(
                self.xml_p2_elevation, "xml_p2_elevation"
            )
        if self.xml_p2_position is not None:
            try:
                if len(self.xml_p2_position) != 2:
                    raise ValueError
                self.xml_p2_position = (
                    _finite_float(self.xml_p2_position[0], "xml_p2_position x"),
                    _finite_float(self.xml_p2_position[1], "xml_p2_position y"),
                )
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    "xml_p2_position must contain exactly two finite coordinates"
                ) from exc

    def _effective_xml_p2_position(self) -> Tuple[float, float]:
        return self.relative_position if self.xml_p2_position is None else self.xml_p2_position

    def _effective_xml_p2_elevation(self) -> float:
        if self.xml_p2_elevation is None:
            return _finite_float(self.elevation, "elevation")
        return _finite_float(self.xml_p2_elevation, "xml_p2_elevation")

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        """Returns absolute anchor position and scaled height."""
        abs_position = get_transformed_point(self.relative_position, total_transform)

        # Calculate scaled height (using average scale factor)
        sx = np.linalg.norm(total_transform[:, 0])
        sy = np.linalg.norm(total_transform[:, 1])
        avg_scale = (sx + sy) / 2.0
        abs_height = self.height * avg_scale

        # Font, style, alignment etc are not directly transformed, but scale affects appearance
        return {
            "position": abs_position,
            "height": abs_height,
            "text": self.text_content,
            # Include other properties if needed for bounding box calculation
        }

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> Dict[str, Any]:
        geometry = self._calculate_absolute_geometry(total_transform)
        position_x, position_y = geometry["position"]
        geometry["position"] = (
            position_x, position_y,
            _finite_float(_finite_float(self.elevation, "elevation") + total_z_offset, "world Z"),
        )
        if self.xml_p2_position is None and self.xml_p2_elevation is None:
            geometry["xml_p2"] = None
        else:
            p2_x, p2_y = get_transformed_point(
                self._effective_xml_p2_position(), total_transform
            )
            geometry["xml_p2"] = (
                p2_x, p2_y,
                _finite_float(self._effective_xml_p2_elevation() + total_z_offset,
                              "world XML p2 Z"),
            )
        return geometry

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        elevation = _finite_float(
            _finite_float(self.elevation, "elevation") + delta,
            "shifted elevation",
        )
        if self.xml_p2_elevation is None:
            p2_elevation = None
        else:
            p2_elevation = _finite_float(
                self._effective_xml_p2_elevation() + delta,
                "shifted xml_p2_elevation",
            )
        self.elevation = elevation
        self.xml_p2_elevation = p2_elevation

    def _calculate_bounding_box(self, absolute_geometry: Dict[str, Any]) -> BoundingBox:
        # Bounding box for text is complex and font-dependent.
        # Provide a rough estimate based on height and text length.
        logger.warning(f"Bounding box for Text primitive '{self.user_identifier}' is approximate.")
        px, py = absolute_geometry["position"]
        h = absolute_geometry["height"]
        # Estimate width: number of chars in longest line * height * aspect ratio (e.g., 0.6)
        longest_line_len = max(len(line) for line in self.text_content.splitlines()) if self.text_content else 0
        est_width = longest_line_len * h * 0.6
        num_lines = self.text_content.count('\n') + 1
        est_total_height = h * (1 + (num_lines - 1) * self.line_spacing)

        # Adjust position based on alignment (relative to estimated box)
        # Horizontal
        if self.align_horizontal == 'left':
            min_x, max_x = px, px + est_width
        elif self.align_horizontal == 'right':
            min_x, max_x = px - est_width, px
        else: # center
            min_x, max_x = px - est_width / 2, px + est_width / 2
        # Vertical
        if self.align_vertical == 'top':
            min_y, max_y = py - est_total_height, py
        elif self.align_vertical == 'bottom':
            min_y, max_y = py, py + est_total_height
        else: # center
            min_y, max_y = py - est_total_height / 2, py + est_total_height / 2

        return BoundingBox(min_x, min_y, max_x, max_y)


    def get_geometric_center(self) -> Tuple[float, float]:
        # Use the transformed anchor position as the 'center'
        # A better center might be the center of the approximate bounding box.
        return get_transformed_point(self.relative_position, self.get_total_transform())

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to position and height.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        affine = _affine_matrix_or_none(transform_to_apply)
        if affine is None:
            raise ValueError("Expected a finite affine 3x3 matrix")
        linear = affine[0:2, 0:2]
        scale = float((linear[0, 0] + linear[1, 1]) / 2.0)
        if (scale <= 0.0 or not np.allclose(
                linear, np.identity(2) * scale, rtol=0.0, atol=1e-12)):
            raise ValueError(
                "Text geometry can only bake XY translation and positive uniform scale"
            )
        transform_to_apply = affine
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake position
            self.relative_position = get_transformed_point(self.relative_position, transform_to_apply)
            if self.xml_p2_position is not None:
                self.xml_p2_position = get_transformed_point(
                    self.xml_p2_position, transform_to_apply
                )
            
            # Bake height (using average scale factor)
            self.height *= scale
            
            # Handle text mirroring - affects alignment
            det = np.linalg.det(transform_to_apply[0:2, 0:2])
            if det < 0:  # Mirroring detected
                # Determine axis of mirroring
                sx_sign = np.sign(transform_to_apply[0, 0]) 
                sy_sign = np.sign(transform_to_apply[1, 1])
                
                # For x-mirroring (negative x scale)
                if sx_sign < 0:
                    if self.align_horizontal == 'left':
                        self.align_horizontal = 'right'
                    elif self.align_horizontal == 'right':
                        self.align_horizontal = 'left'
                
                # For y-mirroring (negative y scale)
                if sy_sign < 0:
                    if self.align_vertical == 'top':
                        self.align_vertical = 'bottom'
                    elif self.align_vertical == 'bottom':
                        self.align_vertical = 'top'
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Text {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <text> XML element."""
        # CamBam omits p1 at the default origin and may preserve an optional,
        # currently-unused p2. Align is serialized independently.
        cb_v_align = self.align_vertical
        cb_h_align = self.align_horizontal
        p1x = round(self.relative_position[0], self.output_decimals) if self.output_decimals is not None else self.relative_position[0]
        p1y = round(self.relative_position[1], self.output_decimals) if self.output_decimals is not None else self.relative_position[1]
        p1z = _finite_float(self.elevation, "elevation")
        p2_position = self._effective_xml_p2_position()
        p2x = round(p2_position[0], self.output_decimals) if self.output_decimals is not None else p2_position[0]
        p2y = round(p2_position[1], self.output_decimals) if self.output_decimals is not None else p2_position[1]
        p2z = self._effective_xml_p2_elevation()
        if self.output_decimals is not None:
            p1z = round(p1z, self.output_decimals)
            p2z = round(p2z, self.output_decimals)

        attributes = {
            "Height": str(self.height),
            "Font": self.font,
            "linespace": str(self.line_spacing),
            "align": f"{cb_v_align},{cb_h_align}",
            "style": self.style
        }
        if (p1x, p1y, p1z) != (0, 0, 0):
            attributes["p1"] = f"{p1x},{p1y},{p1z}"
        if self.xml_p2_position is not None or self.xml_p2_elevation is not None:
            attributes["p2"] = f"{p2x},{p2y},{p2z}"
        text_elem = ET.Element("text", attributes)
        # Text content goes inside the element
        text_elem.text = self.text_content

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(text_elem, xml_primitive_id, parent_uuid)
        return text_elem
