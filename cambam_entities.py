"""
cambam_entities.py

Defines the core CamBam entity classes (Layer, Part, Primitive, Mop).
Entities primarily hold their intrinsic attributes (geometry, parameters, visual style).
Relationships between entities (layer assignment, parent/child links, MOP assignments)
are managed centrally by the CamBamProject class.
"""

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import weakref
import uuid
import logging
import json
import math
from typing import List, Dict, Tuple, Union, Optional, Set, Any, Sequence, Type, TypeVar, TYPE_CHECKING

import numpy as np

# Assuming cad_transformations provides these:
from cad_transformations import (
    identity_matrix, apply_transform, get_transformed_point, to_cambam_matrix_str, from_cambam_matrix_str
)

# Type hint for the project class without circular import
if TYPE_CHECKING:
    from cambam_project import CamBamProject

logger = logging.getLogger(__name__)

# --- Helper Classes ---

@dataclass(frozen=True)
class BoundingBox:
    """Represents a 2D bounding box."""
    min_x: float = float('inf')
    min_y: float = float('inf')
    max_x: float = float('-inf')
    max_y: float = float('-inf')

    def is_valid(self) -> bool:
        return self.min_x <= self.max_x and self.min_y <= self.max_y

    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        if not other.is_valid():
            return self
        if not self.is_valid():
            return other
        return BoundingBox(
            min_x=min(self.min_x, other.min_x),
            min_y=min(self.min_y, other.min_y),
            max_x=max(self.max_x, other.max_x),
            max_y=max(self.max_y, other.max_y)
        )

    @staticmethod
    def from_points(points: Sequence[Tuple[float, float]]) -> 'BoundingBox':
        if not points:
            return BoundingBox() # Invalid box
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        return BoundingBox(min_x, min_y, max_x, max_y)

# --- Base Entity Class ---

@dataclass
class CamBamEntity:
    """Base class for all CamBam entities."""
    internal_id: uuid.UUID = field(default_factory=uuid.uuid4, init=False) # Primary key, set on creation
    user_identifier: str = "" # User-friendly ID, must be unique within project

    def __post_init__(self):
        # Ensure user_identifier is set if not provided, using UUID initially
        if not self.user_identifier:
            self.user_identifier = str(self.internal_id)

# --- Layer Entity ---

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

# --- Part Entity ---

@dataclass
class Part(CamBamEntity):
    """Represents a machining part with stock and default parameters."""
    enabled: bool = True
    stock_thickness: float = 12.5
    stock_width: float = 1220.0
    stock_height: float = 2440.0
    stock_material: str = "MDF"
    stock_color: str = "210,180,140" # RGB string
    machining_origin: Tuple[float, float] = (0.0, 0.0) # XY offset
    default_tool_diameter: Optional[float] = None
    default_spindle_speed: Optional[int] = None
    # Note: No mop_ids or _xml_machineops_element here. Managed by Project.

    def to_xml_element(self) -> ET.Element:
        """Creates the <part> XML element (without the <machineops> container)."""
        part_elem = ET.Element("part", {
            "Name": self.user_identifier, # Use user_identifier as CamBam's part name
            "Enabled": str(self.enabled).lower()
        })
        # The <machineops> sub-element will be added by the writer based on project registry

        # Add stock and other part-level settings directly here
        stock = ET.SubElement(part_elem, "Stock")
        # CamBam stock is defined by PMin(x,y,z) and PMax(x,y,z)
        # We define the stock offset, PMin, as (0,0,-thickness) since it will be aligned to the machine origin, and thickness is negative Z (material surface at Z=0).
        # Then we use the machine origin to offset the stock and machine origin out on the canvas, where primitives pertaining to this stock will be drawn.
        # PMax is the stock width, height, surface (Z=0)
        ET.SubElement(stock, "PMin").text = f"0,0,{-self.stock_thickness}"
        ET.SubElement(stock, "PMax").text = f"{self.stock_width},{self.stock_height},0"
        ET.SubElement(stock, "Material").text = self.stock_material
        ET.SubElement(stock, "Color").text = self.stock_color

        ET.SubElement(part_elem, "MachiningOrigin").text = f"{self.machining_origin[0]},{self.machining_origin[1]}"

        if self.default_tool_diameter is not None:
            ET.SubElement(part_elem, "ToolDiameter").text = str(self.default_tool_diameter)
        if self.default_spindle_speed is not None:
            # CamBam doesn't seem to have a direct default spindle speed at part level in XML?
            # MOPs inherit if not set, but storing it here is useful for the framework.
            # We won't write it to XML unless a specific field is found later.
            pass

        # Add other common Part elements expected by CamBam
        ET.SubElement(part_elem, "ToolProfile").text = "EndMill" # Default, can be overridden by MOPs
        nesting = ET.SubElement(part_elem, "Nesting")
        ET.SubElement(nesting, "BasePoint").text = "0,0"
        ET.SubElement(nesting, "NestMethod").text = "None"

        return part_elem

# --- Primitive Base Class and Concrete Classes ---

@dataclass
class Primitive(CamBamEntity, ABC):
    """
    Abstract Base Class for geometric primitives.
    Holds intrinsic geometry and transformation state.
    Classification attributes (groups, description) are stored for the XML <Tag>.
    Layer assignment and parent/child links are managed externally by the project.
    """
    # Intrinsic Attributes
    effective_transform: np.ndarray = field(default_factory=identity_matrix)
    groups: List[str] = field(default_factory=list) # Classification
    description: str = ""                           # Classification
    output_decimals: Optional[int] = 9              # For XML serialization

    # Reference back to the project (transient, for context)
    _project_ref: Optional[weakref.ReferenceType] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        # Ensure transform is always a valid matrix
        if not isinstance(self.effective_transform, np.ndarray) or self.effective_transform.shape != (3, 3):
            self.effective_transform = identity_matrix()
        # Ensure groups is a list
        if self.groups is None:
            self.groups = []

    # --- State Management (for pickling) ---
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle weak reference
        if '_project_ref' in state:
            del state['_project_ref']
        # Convert numpy array to list for potentially better pickle compatibility?
        # Or keep as array if pickle handles it reliably. Let's keep it for now.
        # state['effective_transform'] = self.effective_transform.tolist()
        return state

    def __setstate__(self, state):
        # Restore numpy array if it was converted to list
        # if isinstance(state.get('effective_transform'), list):
        #     state['effective_transform'] = np.array(state['effective_transform'])
        self.__dict__.update(state)
        # Re-initialize transient fields
        self._project_ref = None

    # --- Project Context ---
    def get_project(self) -> Optional["CamBamProject"]:
        """Returns the owning project, if linked."""
        if self._project_ref:
            project = self._project_ref()
            if project:
                return project
        # logger.warning(f"Primitive {self.user_identifier} ({self.internal_id}) is not linked to a project.")
        return None

    def set_project_link(self, project: "CamBamProject"):
        """Sets the weak reference to the owning project."""
        self._project_ref = weakref.ref(project)

    # --- Transformations ---
    def get_total_transform(self) -> np.ndarray:
        """Calculates the total transformation matrix by cascading from the root parent."""
        total_tf = self.effective_transform.copy()
        project = self.get_project()
        if project:
            parent_id = project.get_parent_of_primitive(self.internal_id)
            while parent_id:
                parent = project.get_primitive(parent_id)
                if parent:
                    total_tf = parent.effective_transform @ total_tf # Apply parent's transform first
                    parent_id = project.get_parent_of_primitive(parent.internal_id)
                else:
                    logger.warning(f"Parent primitive {parent_id} not found for {self.user_identifier} during transform calculation.")
                    break # Stop climbing if parent is missing
        return total_tf

    # --- Geometry Calculations ---
    def get_absolute_coordinates(self) -> Any:
        """Calculates the primitive's geometry in absolute world coordinates."""
        total_tf = self.get_total_transform()
        return self._calculate_absolute_geometry(total_tf)

    def get_bounding_box(self) -> BoundingBox:
        """Calculates the 2D bounding box in absolute world coordinates."""
        # Avoid recalculating absolute geometry if possible
        # This implementation recalculates each time. Caching could be added if needed.
        try:
            abs_coords = self.get_absolute_coordinates()
            return self._calculate_bounding_box(abs_coords)
        except Exception as e:
            logger.error(f"Error calculating bounding box for {self.user_identifier}: {e}")
            return BoundingBox() # Return invalid box on error

    @abstractmethod
    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Any:
        """Subclasses implement this to return absolute geometry based on total transform."""
        pass

    @abstractmethod
    def _calculate_bounding_box(self, absolute_geometry: Any) -> BoundingBox:
        """Subclasses implement this to calculate BB from their absolute geometry."""
        pass

    @abstractmethod
    def get_geometric_center(self) -> Tuple[float, float]:
        """Subclasses implement this to return the geometric center in absolute coordinates."""
        pass

    # --- XML Generation ---
    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """
        Generates the base XML element for the primitive.
        Subclasses will create the specific element type (e.g., 'pline', 'circle')
        and call this method to add common attributes and the matrix.
        """
        # This method now expects the parent UUID to be passed in by the writer
        # It also doesn't return the element directly, but rather configures a passed element.
        # Let's change the pattern: subclasses create their element, then call a helper
        # from this base class to add common stuff.

        # This method won't be called directly. See _add_common_xml_attributes
        raise NotImplementedError("Use specific to_xml_element in subclasses.")

    def _add_common_xml_attributes(self, element: ET.Element, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]):
        """Adds common attributes (ID, Tag, Matrix) to the primitive's XML element."""
        element.set("id", str(xml_primitive_id))

        # Create Tag data
        tag_data = {
            "user_id": self.user_identifier,
            "internal_id": str(self.internal_id),
            "groups": self.groups or [],
            "parent": str(parent_uuid) if parent_uuid else None,
            "description": self.description or ""
        }
        # Remove None values for cleaner JSON
        tag_data = {k: v for k, v in tag_data.items() if v is not None}
        ET.SubElement(element, "Tag").text = json.dumps(tag_data, separators=(',', ':')) # Compact JSON

        # Add Transformation Matrix
        # The matrix stored in XML is the *total* transformation relative to world origin
        total_tf = self.get_total_transform()
        mat_str = to_cambam_matrix_str(total_tf, output_decimals=self.output_decimals)
        ET.SubElement(element, "mat", {"m": mat_str})

    @abstractmethod
    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix directly to the primitive's relative geometry.
        
        Args:
            transform_to_bake: The transformation matrix to bake into the geometry.
                            If None, the primitive's current effective_transform is used
                            and then reset to identity.
        
        When a specific matrix is provided, that transformation is baked into the actual
        geometry points without modifying the primitive's effective_transform.
        
        When no matrix is provided, the primitive's effective_transform is applied to
        its geometry and then reset to identity.
        """
        pass

# --- Concrete Primitive Classes ---

@dataclass
class Pline(Primitive):
    # Intrinsic geometry
    relative_points: List[Union[Tuple[float, float], Tuple[float, float, float]]] = field(default_factory=list)
    closed: bool = False

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float, float]]:
        # Extract XY for transformation
        rel_pts_xy = [(p[0], p[1]) for p in self.relative_points]
        abs_pts_xy = apply_transform(rel_pts_xy, total_transform)

        # Re-attach bulge values (bulge is not transformed by standard matrix)
        abs_pts_with_bulge = []
        for i, (x, y) in enumerate(abs_pts_xy):
            # Handle potential index out of bounds if apply_transform skipped points
            if i < len(self.relative_points):
                bulge = self.relative_points[i][2] if len(self.relative_points[i]) > 2 else 0.0
                abs_pts_with_bulge.append((x, y, bulge))
            else:
                logger.warning(f"Point mismatch after transformation for Pline {self.user_identifier}. Skipping bulge.")

        return abs_pts_with_bulge

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

    def get_geometric_center(self) -> Tuple[float, float]:
        # Use bounding box center as geometric center
        bbox = self.get_bounding_box()
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        elif self.relative_points:
            # Fallback to first point if bbox is invalid (e.g., single point pline)
            abs_coords = self.get_absolute_coordinates()
            if abs_coords:
                return abs_coords[0][0], abs_coords[0][1]
        return (0.0, 0.0) # Default fallback

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to relative_points.
        
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
            
        # Skip if identity matrix (nothing to bake)
        if np.allclose(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Transform XY coordinates
            rel_pts_xy = [(p[0], p[1]) for p in self.relative_points]
            baked_pts_xy = apply_transform(rel_pts_xy, transform_to_apply)

            # Rebuild relative points with original bulge values
            new_relative_points = []
            for i, (x, y) in enumerate(baked_pts_xy):
                if i < len(self.relative_points):
                    bulge = self.relative_points[i][2] if len(self.relative_points[i]) > 2 else 0.0
                    new_relative_points.append((x, y, bulge))
                else:
                    logger.warning(f"Point mismatch during baking for Pline {self.user_identifier}. Skipping point.")

            self.relative_points = new_relative_points
            
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
        for pt in self.relative_points:
            # CamBam point format: x,y,z (z is usually 0 for 2D)
            x = round(pt[0], self.output_decimals) if self.output_decimals is not None else pt[0]
            y = round(pt[1], self.output_decimals) if self.output_decimals is not None else pt[1]
            z = 0.0
            bulge = pt[2] if len(pt) > 2 else 0.0
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
            
        # Skip if identity matrix (nothing to bake)
        if np.allclose(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake center position
            self.relative_center = get_transformed_point(self.relative_center, transform_to_apply)

            # Bake diameter (using average scale factor)
            sx = np.linalg.norm(transform_to_apply[:, 0])
            sy = np.linalg.norm(transform_to_apply[:, 1])
            avg_scale = (sx + sy) / 2.0
            self.diameter *= avg_scale
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Circle {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <circle> XML element."""
        cx = round(self.relative_center[0], self.output_decimals) if self.output_decimals is not None else self.relative_center[0]
        cy = round(self.relative_center[1], self.output_decimals) if self.output_decimals is not None else self.relative_center[1]
        cz = 0.0
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

    def _get_relative_corners(self) -> List[Tuple[float, float]]:
        """Returns the four corners in relative coordinates."""
        x0, y0 = self.relative_corner
        return [(x0, y0), (x0 + self.width, y0), (x0 + self.width, y0 + self.height), (x0, y0 + self.height)]

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        """Returns the four corners in absolute coordinates."""
        return apply_transform(self._get_relative_corners(), total_transform)

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
        pline_points = [(p[0], p[1], 0.0) for p in transformed_corners]
        
        # Create a new Pline
        pline = Pline(
            user_identifier=f"{self.user_identifier}_as_pline",
            groups=self.groups.copy() if self.groups else [],
            description=f"Converted from Rect: {self.description}",
            effective_transform=identity_matrix(),  # Use identity since we've already applied the transform
            relative_points=pline_points,
            closed=True
        )
        
        return pline

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """Applies transformation to relative_corner, width, and height."""
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False
            
        # Skip if identity matrix (nothing to bake)
        if np.allclose(transform_to_apply, identity_matrix()):
            return

        try:
            # Get the rectangle corners
            corners = self._get_relative_corners()
            
            # Transform the corners
            transformed_corners = apply_transform(corners, transform_to_apply)
            
            # Find new axis-aligned bounding box
            min_x = min(p[0] for p in transformed_corners)
            min_y = min(p[1] for p in transformed_corners)
            max_x = max(p[0] for p in transformed_corners)
            max_y = max(p[1] for p in transformed_corners)
            
            # Update rectangle properties
            self.relative_corner = (min_x, min_y)
            self.width = max_x - min_x
            self.height = max_y - min_y
            
            # Check if geometry is not rectangular
            if not self.is_rectangular_after_transform() and transform_to_apply is self.effective_transform:
                logger.warning(f"Baking Rect {self.user_identifier}: Applied transform approximately. Rotation/shear will be lost.")
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Rect {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """
        Creates an XML element for this Rect.
        If the geometry is not rectangular after transformation, converts to a Pline representation.
        """
        # Check if we need to convert to Pline for XML output
        if not self.is_rectangular_after_transform():
            # Create a Pline representation
            pline_repr = self.to_pline_representation()
            
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
        z = 0.0
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

    def _calculate_bounding_box(self, absolute_geometry: Dict[str, Any]) -> BoundingBox:
        # Approximate bounding box using center and radius
        # TODO: A precise bounding box needs to consider the actual arc sweep.
        cx, cy = absolute_geometry["center"]
        radius = absolute_geometry["radius"]
        logger.debug(f"Bounding box for Arc {self.user_identifier} is approximate (using full circle).")
        return BoundingBox(cx - radius, cy - radius, cx + radius, cy + radius)

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
            
        # Skip if identity matrix (nothing to bake)
        if np.allclose(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake center position
            self.relative_center = get_transformed_point(self.relative_center, transform_to_apply)
            
            # Bake radius (using average scale factor)
            sx = np.linalg.norm(transform_to_apply[:, 0])
            sy = np.linalg.norm(transform_to_apply[:, 1])
            avg_scale = (sx + sy) / 2.0
            self.radius *= avg_scale
            
            # Bake start angle
            rotation_rad = math.atan2(transform_to_apply[1, 0], transform_to_apply[0, 0])
            rotation_deg = math.degrees(rotation_rad)
            self.start_angle = (self.start_angle + rotation_deg) % 360
            
            # Handle mirroring which can affect angle direction
            det = np.linalg.det(transform_to_apply[0:2, 0:2])
            if det < 0:  # Mirroring detected
                # Flip the direction of the arc
                self.start_angle = (self.start_angle + self.extent_angle) % 360
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
        cz = 0.0
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
    relative_points: List[Tuple[float, float]] = field(default_factory=list)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        return apply_transform(self.relative_points, total_transform)

    def _calculate_bounding_box(self, absolute_geometry: List[Tuple[float, float]]) -> BoundingBox:
        if not absolute_geometry:
            return BoundingBox()
        return BoundingBox.from_points(absolute_geometry)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Use bounding box center
        bbox = self.get_bounding_box()
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        elif self.relative_points:
            abs_coords = self.get_absolute_coordinates()
            if abs_coords:
                return abs_coords[0] # Fallback to first point
        return (0.0, 0.0)

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to relative_points.
        
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
            
        # Skip if identity matrix (nothing to bake)
        if np.allclose(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Transform the points
            self.relative_points = apply_transform(self.relative_points, transform_to_apply)
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Points {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <points> XML element."""
        points_elem = ET.Element("points")
        pts_elem = ET.SubElement(points_elem, "pts")
        for x, y in self.relative_points:
            px = round(x, self.output_decimals) if self.output_decimals is not None else x
            py = round(y, self.output_decimals) if self.output_decimals is not None else y
            pz = 0.0
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
    height: float = 10.0 # Font height in drawing units
    font: str = 'Arial'
    style: str = '' # e.g., 'bold', 'italic', 'bold,italic'
    line_spacing: float = 1.0 # Multiplier
    align_horizontal: str = 'center' # 'left', 'center', 'right'
    align_vertical: str = 'center' # 'top', 'center', 'bottom'

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
            
        # Skip if identity matrix (nothing to bake)
        if np.allclose(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake position
            self.relative_position = get_transformed_point(self.relative_position, transform_to_apply)
            
            # Bake height (using average scale factor)
            sx = np.linalg.norm(transform_to_apply[:, 0])
            sy = np.linalg.norm(transform_to_apply[:, 1])
            avg_scale = (sx + sy) / 2.0
            self.height *= avg_scale
            
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
        # CamBam text uses p1, p2 (often same), Height, Font, align etc.
        # Align format seems to be "Vertical,Horizontal" e.g., "center,center"
        cb_v_align = self.align_vertical
        cb_h_align = self.align_horizontal
        p1x = round(self.relative_position[0], self.output_decimals) if self.output_decimals is not None else self.relative_position[0]
        p1y = round(self.relative_position[1], self.output_decimals) if self.output_decimals is not None else self.relative_position[1]
        p1z = 0.0
        p2x, p2y, p2z = p1x, p1y, p1z

        text_elem = ET.Element("text", {
            "p1": f"{p1x},{p1y},{p1z}",
            "p2": f"{p2x},{p2y},{p2z}", # p2 seems unused?
            "Height": str(self.height),
            "Font": self.font,
            "linespace": str(self.line_spacing),
            "align": f"{cb_v_align},{cb_h_align}",
            "style": self.style
        })
        # Text content goes inside the element
        text_elem.text = self.text_content

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(text_elem, xml_primitive_id, parent_uuid)
        return text_elem


# --- MOP Base and Concrete Classes ---

MopType = TypeVar('MopType', bound='Mop')

@dataclass
class Mop(CamBamEntity, ABC):
    """
    Abstract Base Class for Machine Operations (MOPs).
    Holds intrinsic machining parameters and the definition of what primitives to target.
    Part assignment is managed externally by the project.
    """
    # Intrinsic Attributes
    name: str = "MOP" # User-visible name in CamBam UI MOP tree
    pid_source: Union[str, List[uuid.UUID]] = field(default_factory=list) # Group name or list of Primitive UUIDs
    enabled: bool = True
    target_depth: Optional[float] = None # If None, uses Part/Project default (or fails if none set)
    depth_increment: Optional[float] = None # If None, uses TargetDepth (single pass)
    stock_surface: float = 0.0
    roughing_clearance: float = 0.0
    clearance_plane: float = 15.0
    spindle_direction: str = 'CW' # 'CW', 'CCW', 'Off'
    spindle_speed: Optional[int] = None # If None, uses Part/Project default
    velocity_mode: str = 'ExactStop' # 'ExactStop', 'ConstantVelocity'
    work_plane: str = 'XY' # 'XY', 'XZ', 'YZ'
    optimisation_mode: str = 'Standard' # 'Standard', 'Experimental', 'Legacy'
    tool_diameter: Optional[float] = None # If None, uses Part/Project default
    tool_number: int = 0 # If 0, uses current tool
    tool_profile: str = 'EndMill' # 'EndMill', 'Vcutter', 'BallNose', 'Engrave', 'Drill'
    plunge_feedrate: float = 1000.0
    cut_feedrate: Optional[float] = None # If None, calculated based on TargetDepth or uses default
    max_crossover_distance: float = 0.7 # Multiplier of tool diameter
    custom_mop_header: str = ""
    custom_mop_footer: str = ""
    # Note: No part_id or _resolved_xml_primitive_ids here. Managed by Project/Writer.

    @abstractmethod
    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        """
        Generates the MOP's specific XML element (e.g., <profile>, <pocket>).
        Requires the project context to resolve default parameters and the list of
        resolved primitive XML IDs (calculated by the writer).
        """
        pass

    def _get_effective_param(self, param_name: str, project: "CamBamProject") -> Any:
        """Helper to get a parameter value, checking MOP, then Part, then Project defaults."""
        mop_value = getattr(self, param_name, None)
        if mop_value is not None:
            return mop_value

        # Check Part default
        part = project.get_part_of_mop(self.internal_id) if project else None
        if part:
            part_param_map = {
                'tool_diameter': 'default_tool_diameter',
                'spindle_speed': 'default_spindle_speed'
                # Add other mappings if Part gets more defaults
            }
            part_attr = part_param_map.get(param_name)
            if part_attr:
                part_value = getattr(part, part_attr, None)
                if part_value is not None:
                    return part_value

        # Check Project default
        project_param_map = {
            'tool_diameter': 'default_tool_diameter'
            # Add other mappings if Project gets more defaults
        }
        project_attr = project_param_map.get(param_name)
        if project_attr and project:
            project_value = getattr(project, project_attr, None)
            if project_value is not None:
                return project_value

        # No value found anywhere
        # logger.debug(f"Parameter '{param_name}' not resolved for MOP '{self.name}'. Returning None.")
        return None

    def _calculate_effective_cut_feedrate(self, project: "CamBamProject") -> float:
        """Calculates the cut feedrate to use, applying logic if not explicitly set."""
        if self.cut_feedrate is not None:
            return self.cut_feedrate

        # Try simple fallback logic (e.g., based on depth, or just a default)
        # Original logic: 350 * abs(TargetDepth) + 6500
        eff_target_depth = self._get_effective_param('target_depth', project)
        if eff_target_depth is not None and eff_target_depth != 0:
            # Apply the formula (ensure it makes sense, maybe cap it)
            calculated_feedrate = round(350 * abs(eff_target_depth) + 6500, 0)
            feedrate = max(calculated_feedrate, 1000.0) # Ensure minimum feedrate
            logger.debug(f"MOP '{self.name}': Calculated CutFeedrate={feedrate} based on TargetDepth={eff_target_depth}")
            return feedrate
        else:
            # Fallback if TargetDepth isn't available or zero
            default_feedrate = 3000.0
            logger.warning(f"MOP '{self.name}': TargetDepth not set or zero. Using fallback CutFeedrate={default_feedrate}.")
            return default_feedrate

    def _add_common_mop_elements(self, mop_root_elem: ET.Element, project: "CamBamProject", resolved_primitive_xml_ids: List[int]):
        """Adds common XML sub-elements shared by all MOP types."""
        ET.SubElement(mop_root_elem, "Name").text = self.name # Use MOP's intrinsic name

        # Resolve parameters using defaults if needed
        eff_target_depth = self._get_effective_param('target_depth', project)
        eff_depth_inc = self.depth_increment if self.depth_increment is not None else abs(eff_target_depth) if eff_target_depth is not None else None
        eff_spindle_speed = self._get_effective_param('spindle_speed', project)
        eff_tool_dia = self._get_effective_param('tool_diameter', project)
        eff_cut_feedrate = self._calculate_effective_cut_feedrate(project)

        # Helper to create elements with state attribute
        def add_param(parent, tag, value, mop_attr):
            state = "Value" if getattr(self, mop_attr, None) is not None else "Default"
            # Handle optional values that might resolve to None
            text_value = str(value) if value is not None else ""
            # Special case: TargetDepth needs a value even if default? Check CamBam output.
            # Assuming empty text is okay for unresolved optional defaults.
            if value is None and tag in ["TargetDepth", "DepthIncrement", "SpindleSpeed", "ToolDiameter"]:
                logger.warning(f"MOP '{self.name}': Parameter '{tag}' resolved to None. Writing empty element.")
                # CamBam might require a default value here, e.g., "0" or "-1"
                # text_value = "0" # Or handle based on specific parameter

            ET.SubElement(parent, tag, {"state": state}).text = text_value

        add_param(mop_root_elem, "TargetDepth", eff_target_depth, 'target_depth')
        add_param(mop_root_elem, "DepthIncrement", eff_depth_inc, 'depth_increment')
        add_param(mop_root_elem, "StockSurface", self.stock_surface, 'stock_surface')
        add_param(mop_root_elem, "RoughingClearance", self.roughing_clearance, 'roughing_clearance')
        add_param(mop_root_elem, "ClearancePlane", self.clearance_plane, 'clearance_plane')
        add_param(mop_root_elem, "SpindleDirection", self.spindle_direction, 'spindle_direction')
        add_param(mop_root_elem, "SpindleSpeed", eff_spindle_speed, 'spindle_speed')
        ET.SubElement(mop_root_elem, "SpindleRange", {"state": "Value"}).text = "0" # Default value
        add_param(mop_root_elem, "VelocityMode", self.velocity_mode, 'velocity_mode')
        add_param(mop_root_elem, "WorkPlane", self.work_plane, 'work_plane')
        add_param(mop_root_elem, "OptimisationMode", self.optimisation_mode, 'optimisation_mode')
        add_param(mop_root_elem, "ToolDiameter", eff_tool_dia, 'tool_diameter')
        add_param(mop_root_elem, "ToolNumber", self.tool_number, 'tool_number')
        add_param(mop_root_elem, "ToolProfile", self.tool_profile, 'tool_profile')
        add_param(mop_root_elem, "PlungeFeedrate", self.plunge_feedrate, 'plunge_feedrate')
        add_param(mop_root_elem, "CutFeedrate", eff_cut_feedrate, 'cut_feedrate')
        add_param(mop_root_elem, "MaxCrossoverDistance", self.max_crossover_distance, 'max_crossover_distance')
        add_param(mop_root_elem, "CustomMOPHeader", self.custom_mop_header, 'custom_mop_header')
        add_param(mop_root_elem, "CustomMOPFooter", self.custom_mop_footer, 'custom_mop_footer')

        # Add primitive references (using the resolved XML IDs passed by the writer)
        primitive_container = ET.SubElement(mop_root_elem, "primitive")
        if resolved_primitive_xml_ids:
            for pid in sorted(resolved_primitive_xml_ids): # Sort for consistency
                ET.SubElement(primitive_container, "prim").text = str(pid)
        # else: # CamBam seems to omit the <primitive> tag entirely if empty
            # pass

    def _add_lead_in_out_elements(self, parent_elem: ET.Element, lead_type: str = "Spiral", spiral_angle: float = 30.0, tangent_radius: float = 0.0, feedrate: float = 0.0):
        """Adds LeadInMove and LeadOutMove elements (common pattern)."""
        # Determine state (assume "Value" if explicitly called, could be refined)
        state = "Value"

        lead_in = ET.SubElement(parent_elem, "LeadInMove", {"state": state})
        ET.SubElement(lead_in, "LeadInType").text = lead_type
        ET.SubElement(lead_in, "SpiralAngle").text = str(spiral_angle)
        ET.SubElement(lead_in, "TangentRadius").text = str(tangent_radius)
        ET.SubElement(lead_in, "LeadInFeedrate").text = str(feedrate) # 0 usually means use CutFeedrate

        # Lead out often mirrors lead in settings in CamBam defaults
        lead_out = ET.SubElement(parent_elem, "LeadOutMove", {"state": state})
        ET.SubElement(lead_out, "LeadInType").text = lead_type # Yes, uses "LeadInType" tag name
        ET.SubElement(lead_out, "SpiralAngle").text = str(spiral_angle)
        ET.SubElement(lead_out, "TangentRadius").text = str(tangent_radius)
        ET.SubElement(lead_out, "LeadInFeedrate").text = str(feedrate)

# --- Concrete MOP Classes ---
# (Minimal changes: Update to_xml_element signature and call _add_common_mop_elements)

@dataclass
class ProfileMop(Mop):
    # Profile specific parameters
    stepover: float = 0.4 # Tool diameter fraction
    profile_side: str = 'Inside' # 'Inside', 'Outside'
    milling_direction: str = 'Conventional' # 'Conventional', 'Climb'
    collision_detection: bool = True
    corner_overcut: bool = False
    lead_in_type: str = 'Spiral' # 'None', 'Spiral', 'Tangent', 'Ramp'
    lead_in_spiral_angle: float = 30.0
    final_depth_increment: Optional[float] = 0.0 # If > 0, amount for final pass
    cut_ordering: str = 'DepthFirst' # 'DepthFirst', 'LevelFirst'
    # Holding Tabs parameters
    tab_method: str = 'None' # 'None', 'Automatic', 'Manual' (Manual needs points)
    tab_width: float = 6.0
    tab_height: float = 1.5
    tab_min_tabs: int = 3
    tab_max_tabs: int = 3
    tab_distance: float = 40.0 # Approx distance between auto tabs
    tab_size_threshold: float = 4.0 # Min shape size for tabs
    tab_use_leadins: bool = False
    tab_style: str = 'Square' # 'Square', 'Triangle', 'Ramp'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        mop_elem = ET.Element("profile", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)

        # Add profile-specific elements
        state = "Value" # Assume explicit value for these for now
        ET.SubElement(mop_elem, "StepOver", {"state": state}).text = str(self.stepover)
        ET.SubElement(mop_elem, "InsideOutside", {"state": state}).text = self.profile_side
        ET.SubElement(mop_elem, "MillingDirection", {"state": state}).text = self.milling_direction
        ET.SubElement(mop_elem, "CollisionDetection", {"state": state}).text = str(self.collision_detection).lower()
        ET.SubElement(mop_elem, "CornerOvercut", {"state": state}).text = str(self.corner_overcut).lower()

        self._add_lead_in_out_elements(mop_elem, lead_type=self.lead_in_type, spiral_angle=self.lead_in_spiral_angle)

        fdi_state = "Value" if self.final_depth_increment is not None else "Default" # Can be optional
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": fdi_state}).text = str(self.final_depth_increment if self.final_depth_increment is not None else 0.0)

        ET.SubElement(mop_elem, "CutOrdering", {"state": state}).text = self.cut_ordering

        # Holding Tabs
        tabs = ET.SubElement(mop_elem, "HoldingTabs", {"state": state})
        ET.SubElement(tabs, "TabMethod").text = self.tab_method
        if self.tab_method != 'None':
            ET.SubElement(tabs, "Width").text = str(self.tab_width)
            ET.SubElement(tabs, "Height").text = str(self.tab_height)
            ET.SubElement(tabs, "MinimumTabs").text = str(self.tab_min_tabs)
            ET.SubElement(tabs, "MaximumTabs").text = str(self.tab_max_tabs)
            ET.SubElement(tabs, "TabDistance").text = str(self.tab_distance)
            ET.SubElement(tabs, "SizeThreshold").text = str(self.tab_size_threshold)
            ET.SubElement(tabs, "UseLeadIns").text = str(self.tab_use_leadins).lower()
            ET.SubElement(tabs, "TabStyle").text = self.tab_style
            # Manual tabs would need a <points> sub-element here if TabMethod='Manual'

        return mop_elem


@dataclass
class PocketMop(Mop):
    # Pocket specific parameters
    stepover: float = 0.4
    stepover_feedrate: str = 'Plunge Feedrate' # Name of feedrate to use for stepover moves
    milling_direction: str = 'Conventional'
    collision_detection: bool = True
    lead_in_type: str = 'Spiral'
    lead_in_spiral_angle: float = 30.0
    final_depth_increment: Optional[float] = 0.0
    cut_ordering: str = 'DepthFirst'
    region_fill_style: str = 'InsideOutsideOffsets' # 'InsideOutsideOffsets', 'HorizontalScanline', 'VerticalScanline'
    finish_stepover: float = 0.0 # If > 0, performs a finishing pass this far from edge
    finish_stepover_at_target_depth: bool = False # Apply finish pass only at final depth
    roughing_finishing: str = 'Roughing' # 'Roughing', 'Finishing', 'RoughFinish'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        mop_elem = ET.Element("pocket", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)

        state = "Value"
        ET.SubElement(mop_elem, "StepOver", {"state": state}).text = str(self.stepover)
        ET.SubElement(mop_elem, "StepoverFeedrate", {"state": state}).text = self.stepover_feedrate
        ET.SubElement(mop_elem, "MillingDirection", {"state": state}).text = self.milling_direction
        ET.SubElement(mop_elem, "CollisionDetection", {"state": state}).text = str(self.collision_detection).lower()

        self._add_lead_in_out_elements(mop_elem, lead_type=self.lead_in_type, spiral_angle=self.lead_in_spiral_angle)

        fdi_state = "Value" if self.final_depth_increment is not None else "Default"
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": fdi_state}).text = str(self.final_depth_increment if self.final_depth_increment is not None else 0.0)

        ET.SubElement(mop_elem, "CutOrdering", {"state": state}).text = self.cut_ordering
        ET.SubElement(mop_elem, "RegionFillStyle", {"state": state}).text = self.region_fill_style
        ET.SubElement(mop_elem, "FinishStepover", {"state": state}).text = str(self.finish_stepover)
        ET.SubElement(mop_elem, "FinishStepoverAtTargetDepth", {"state": state}).text = str(self.finish_stepover_at_target_depth).lower()
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": state}).text = self.roughing_finishing
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"}) # Usually calculated unless specified

        return mop_elem

@dataclass
class EngraveMop(Mop):
    # Engrave specific parameters
    roughing_finishing: str = 'Roughing' # Seems less relevant for Engrave? Default='Roughing'
    final_depth_increment: Optional[float] = 0.0 # Depth for final pass
    cut_ordering: str = 'DepthFirst'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        mop_elem = ET.Element("engrave", {"Enabled": str(self.enabled).lower()})
        # Engrave uses ToolDiameter differently (often for simulation only)
        # We still add common params, including ToolDiameter resolution
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)

        state = "Value"
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": state}).text = self.roughing_finishing

        fdi_state = "Value" if self.final_depth_increment is not None else "Default"
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": fdi_state}).text = str(self.final_depth_increment if self.final_depth_increment is not None else 0.0)

        ET.SubElement(mop_elem, "CutOrdering", {"state": state}).text = self.cut_ordering
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"}) # Usually follows shape order

        return mop_elem


@dataclass
class DrillMop(Mop):
    # Drill specific parameters
    drilling_method: str = 'CannedCycle' # 'CannedCycle', 'SpiralMill_CW', 'SpiralMill_CCW', 'CustomScript'
    # Parameters for CannedCycle
    peck_distance: float = 0.0 # If > 0, enables pecking (G83)
    retract_height: float = 5.0 # R plane for canned cycles
    dwell: float = 0.0 # Dwell time at bottom (ms)
    # Parameters for SpiralMill
    hole_diameter: Optional[float] = None # Required for SpiralMill if not using points
    drill_lead_out: bool = False
    spiral_flat_base: bool = True
    lead_out_length: float = 0.0
    # Parameter for CustomScript
    custom_script: str = ""

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        mop_elem = ET.Element("drill", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)

        state = "Value"
        ET.SubElement(mop_elem, "DrillingMethod", {"state": state}).text = self.drilling_method

        # Canned Cycle Params
        ET.SubElement(mop_elem, "PeckDistance", {"state": state}).text = str(self.peck_distance)
        ET.SubElement(mop_elem, "RetractHeight", {"state": state}).text = str(self.retract_height)
        ET.SubElement(mop_elem, "Dwell", {"state": state}).text = str(self.dwell)

        # Spiral Mill Params (conditionally add based on method)
        if self.drilling_method.startswith("SpiralMill"):
            hd_state = "Value" if self.hole_diameter is not None else "Default"
            # HoleDiameter is crucial for spiral milling if source isn't points
            if self.hole_diameter is None:
                logger.warning(f"Drill MOP '{self.name}' uses SpiralMill but HoleDiameter is not set. ToolDiameter will likely be used by CamBam.")
            # Write element even if None, CamBam might use ToolDiameter as fallback
            ET.SubElement(mop_elem, "HoleDiameter", {"state": hd_state}).text = str(self.hole_diameter if self.hole_diameter is not None else "")

            ET.SubElement(mop_elem, "DrillLeadOut", {"state": state}).text = str(self.drill_lead_out).lower()
            ET.SubElement(mop_elem, "SpiralFlatBase", {"state": state}).text = str(self.spiral_flat_base).lower()
            ET.SubElement(mop_elem, "LeadOutLength", {"state": state}).text = str(self.lead_out_length)

        # Custom Script Param
        cs_state = "Value" if self.custom_script else "Default"
        ET.SubElement(mop_elem, "CustomScript", {"state": cs_state}).text = self.custom_script

        # Other common Drill elements
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"})
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": "Value"}).text = "Roughing" # Drill is typically roughing

        return mop_elem