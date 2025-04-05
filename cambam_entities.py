# cambam_entities.py
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import weakref
import uuid
import logging
import math
import numpy as np
from copy import deepcopy
from typing import (
    List, Dict, Tuple, Union, Optional, Set, Any, Sequence, cast, Type, TypeVar
)

# Module imports
from cad_common import CamBamEntity, BoundingBox, CamBamError
from cad_transformations import (
    identity_matrix, apply_transform, get_transformed_point,
    matrix_to_cambam_string, has_rotation, get_scale, get_translation,
    rotation_matrix_deg, translation_matrix
)

# Forward declaration for type hinting
class CamBamProject: pass
class Primitive: pass # Needed for Pline conversion in Rect bake

logger = logging.getLogger(__name__)

# --- Project Structure Entities ---

@dataclass
class Layer(CamBamEntity):
    """Represents a drawing layer."""
    color: str = 'Green'
    alpha: float = 1.0
    pen_width: float = 1.0
    visible: bool = True
    locked: bool = False
    # Managed by CamBamProject
    primitive_ids: Set[uuid.UUID] = field(default_factory=set, init=False, repr=False)
    # Transient field for XML writer
    _xml_objects_element: Optional[ET.Element] = field(default=None, repr=False, init=False)

    def to_xml_element(self) -> ET.Element:
        """Creates the XML element for this Layer."""
        layer_elem = ET.Element("layer", {
            "name": self.user_identifier,
            "color": self.color,
            "alpha": str(self.alpha),
            "pen": str(self.pen_width),
            "visible": str(self.visible).lower(),
            "locked": str(self.locked).lower()
        })
        # The <objects> sub-element is added and stored here by the writer
        self._xml_objects_element = ET.SubElement(layer_elem, "objects")
        return layer_elem

@dataclass
class Part(CamBamEntity):
    """Represents a machining part (container for MOPs and stock definition)."""
    enabled: bool = True
    stock_thickness: float = 12.5
    stock_width: float = 1220.0
    stock_height: float = 2440.0
    stock_material: str = "MDF"
    stock_color: str = "210,180,140" # RGB string
    machining_origin: Tuple[float, float] = (0.0, 0.0)
    default_tool_diameter: Optional[float] = None
    default_spindle_speed: Optional[int] = None
    # Managed by CamBamProject
    mop_ids: List[uuid.UUID] = field(default_factory=list, init=False, repr=False)
    # Transient field for XML writer
    _xml_machineops_element: Optional[ET.Element] = field(default=None, repr=False, init=False)

    def to_xml_element(self) -> ET.Element:
        """Creates the XML element for this Part."""
        part_elem = ET.Element("part", {
            "Name": self.user_identifier,
            "Enabled": str(self.enabled).lower()
        })
        # The <machineops> sub-element is added and stored here by the writer
        self._xml_machineops_element = ET.SubElement(part_elem, "machineops")

        # Stock definition
        stock = ET.SubElement(part_elem, "Stock")
        # CamBam typically defines stock relative to machining origin
        # PMin Z is -thickness, PMax Z is 0 (Stock Surface)
        ET.SubElement(stock, "PMin").text = f"0,0,{-self.stock_thickness}"
        ET.SubElement(stock, "PMax").text = f"{self.stock_width},{self.stock_height},0"
        ET.SubElement(stock, "Material").text = self.stock_material
        ET.SubElement(stock, "Color").text = self.stock_color

        # Machining Origin
        ET.SubElement(part_elem, "MachiningOrigin").text = f"{self.machining_origin[0]},{self.machining_origin[1]}"

        # Default Part Params (Optional)
        if self.default_tool_diameter is not None:
            ET.SubElement(part_elem, "ToolDiameter").text = str(self.default_tool_diameter)
        # CamBam seems to always write a ToolProfile even if empty?
        ET.SubElement(part_elem, "ToolProfile").text = "EndMill" # Or get from project default?

        # Nesting Params (Defaults)
        nesting = ET.SubElement(part_elem, "Nesting")
        ET.SubElement(nesting, "BasePoint").text = "0,0" # Usually top-left of shape bbox
        ET.SubElement(nesting, "NestMethod").text = "None" # Or TopLeft etc.

        return part_elem

# --- Primitive Base Class ---

@dataclass
class Primitive(CamBamEntity, ABC):
    """
    Abstract Base Class for geometric primitives (Points, Pline, Circle, etc.).
    Manages local transformation relative to parent and calculates total transform.
    """
    layer_id: uuid.UUID # Must be provided
    groups: List[str] = field(default_factory=list)
    tags: str = "" # Additional user data string

    # Transformation relative to parent (or global origin if no parent)
    effective_transform: np.ndarray = field(default_factory=identity_matrix)
    parent_primitive_id: Optional[uuid.UUID] = None # Managed by CamBamProject

    # References managed by CamBamProject
    _project_ref: Optional[weakref.ReferenceType['CamBamProject']] = field(default=None, init=False, repr=False)
    _child_primitive_ids: Set[uuid.UUID] = field(default_factory=set, init=False, repr=False)

    # Type checking hint for concrete geometry type
    GEOM_TYPE = TypeVar('GEOM_TYPE')

    def __post_init__(self):
        super().__post_init__()
        # Ensure transform is a NumPy array (e.g., after pickling)
        if not isinstance(self.effective_transform, np.ndarray):
            self.effective_transform = identity_matrix()
        # Ensure groups is a list
        if self.groups is None:
            self.groups = []

    # Pickling support (weakref cannot be pickled)
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_project_ref'] = None # Don't pickle weakref
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._project_ref = None # Needs to be re-established by project on load

    def get_project(self) -> 'CamBamProject':
        """Retrieves the associated CamBamProject instance."""
        if self._project_ref is None:
            raise CamBamError(f"Primitive '{self.user_identifier}' is not registered with a project.")
        project = self._project_ref()
        if project is None:
            # This can happen if the project was deleted but the primitive somehow survived
            raise CamBamError(f"Project reference for primitive '{self.user_identifier}' is lost (weakref expired).")
        return project

    # --- Transformation Calculation ---

    def _get_total_transform(self) -> np.ndarray:
        """
        Calculates the total absolute transformation matrix by recursing up the parent chain.
        Result: M_total = M_parent_total @ M_effective
        """
        parent_tf = identity_matrix()
        if self.parent_primitive_id:
            try:
                project = self.get_project()
                parent = project.get_primitive(self.parent_primitive_id)
                if parent:
                    parent_tf = parent._get_total_transform() # Recursive call
                else:
                    # This suggests an inconsistency in project state
                    logger.warning(f"Parent primitive {self.parent_primitive_id} not found for '{self.user_identifier}' during transform calculation. Assuming no parent transform.")
                    # Should we clear parent_primitive_id here? Maybe not safe during calculation.
            except CamBamError as e:
                 logger.error(f"Error getting project or parent for '{self.user_identifier}': {e}")

        # Combine parent's total transform with this primitive's effective transform
        return parent_tf @ self.effective_transform

    # --- Geometry Calculation ---

    def get_absolute_geometry(self) -> Any:
        """
        Calculates the primitive's geometry in absolute coordinates based on its
        total transformation matrix.
        The return type depends on the specific primitive subclass.
        """
        total_transform = self._get_total_transform()
        return self._calculate_absolute_geometry(total_transform)

    @abstractmethod
    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Any:
        """
        Subclasses implement this to calculate their specific geometry
        (e.g., list of points, center point) after applying the total transform.
        """
        pass

    def get_original_geometry(self) -> Any:
         """
         Calculates the primitive's geometry based *only* on its relative definition,
         ignoring all transformations (equivalent to applying identity matrix).
         """
         return self._calculate_absolute_geometry(identity_matrix())

    # --- Bounding Box Calculation ---

    def get_bounding_box(self) -> BoundingBox:
        """
        Calculates the bounding box of the primitive's geometry
        AFTER applying the total transformation.
        """
        abs_geometry = self.get_absolute_geometry()
        return self._calculate_bounding_box(abs_geometry, self._get_total_transform())

    def get_original_bounding_box(self) -> BoundingBox:
        """
        Calculates the bounding box of the primitive's geometry based
        *only* on its relative definition, ignoring transformations.
        """
        original_geometry = self.get_original_geometry()
        # Pass identity matrix as transform info might be needed (e.g., for Circle diameter)
        return self._calculate_bounding_box(original_geometry, identity_matrix())

    @abstractmethod
    def _calculate_bounding_box(self, geometry: Any, transform: np.ndarray) -> BoundingBox:
        """
        Subclasses implement this to calculate the bounding box based on
        the provided geometry (which could be absolute or original) and the
        transform matrix that generated it (useful for scaling factors etc.).
        """
        pass

    # --- Geometric Center ---

    def get_geometric_center(self, use_original: bool = False) -> Tuple[float, float]:
        """
        Calculates the geometric center of the primitive.
        By default, uses the transformed geometry. If use_original=True, uses
        the original, untransformed geometry.
        """
        if use_original:
            bbox = self.get_original_bounding_box()
        else:
            bbox = self.get_bounding_box()

        if not bbox.is_valid():
             logger.warning(f"Cannot calculate geometric center for '{self.user_identifier}': Bounding box is invalid.")
             # Fallback: Use a known point if possible, otherwise origin
             try:
                 geom = self.get_original_geometry() if use_original else self.get_absolute_geometry()
                 if isinstance(geom, (tuple, list)) and geom:
                     # Take first point of a list, or the tuple itself if it's a point
                     first_point = geom[0] if isinstance(geom, list) else geom
                     if isinstance(first_point, tuple) and len(first_point) >= 2:
                         return (float(first_point[0]), float(first_point[1]))
             except Exception:
                 pass # Ignore errors trying fallback
             return (0.0, 0.0)

        return bbox.center

    # --- Transformation Application (Internal) ---

    def _set_effective_transform(self, matrix: np.ndarray) -> None:
        """ Directly sets the effective transform matrix. """
        if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
            raise ValueError("Effective transform must be a 3x3 NumPy array.")
        self.effective_transform = matrix.copy()
        # No cache invalidation needed anymore

    def _apply_transform_locally(self, local_transform_matrix: np.ndarray) -> None:
        """
        Applies a transformation matrix LOCALLY to the primitive's
        effective_transform (post-multiplication).
        M_eff_new = M_eff_old @ local_transform_matrix
        """
        if not isinstance(local_transform_matrix, np.ndarray) or local_transform_matrix.shape != (3, 3):
            raise ValueError("Local transform must be a 3x3 NumPy array.")
        new_eff_tf = self.effective_transform @ local_transform_matrix
        self._set_effective_transform(new_eff_tf)

    def _apply_transform_globally(self, global_transform_matrix: np.ndarray) -> None:
        """
        Applies a GLOBAL transformation matrix to the primitive, modifying its
        effective_transform relative to its parent.
        M_total_new = global_transform_matrix @ M_total_old
        Requires calculating how M_effective needs to change:
        M_eff_new = inv(M_parent_total) @ global_transform_matrix @ M_parent_total @ M_eff_old
        """
        if not isinstance(global_transform_matrix, np.ndarray) or global_transform_matrix.shape != (3, 3):
            raise ValueError("Global transform must be a 3x3 NumPy array.")

        # 1. Get parent's total transform
        parent_tf = identity_matrix()
        if self.parent_primitive_id:
            try:
                project = self.get_project()
                parent = project.get_primitive(self.parent_primitive_id)
                if parent:
                    parent_tf = parent._get_total_transform()
            except Exception as e: # Catch potential project/primitive lookup errors
                 logger.error(f"Error getting parent transform for global application on '{self.user_identifier}': {e}")
                 # Fallback: Apply as if no parent (effectively local relative to origin)
                 self._apply_transform_locally(global_transform_matrix)
                 return

        # 2. Calculate the change needed in the effective transform
        try:
            # Need inverse of parent's total transform
            inv_parent_tf = np.linalg.inv(parent_tf)
        except np.linalg.LinAlgError:
            logger.error(f"Cannot apply global transform to '{self.user_identifier}': Parent transform is singular (non-invertible). Applying locally instead.")
            # Fallback to local application if parent matrix is degenerate
            self._apply_transform_locally(global_transform_matrix)
            return

        # Calculate the delta matrix: transform from parent's space to global, apply global, transform back to parent's space
        transform_delta = inv_parent_tf @ global_transform_matrix @ parent_tf

        # 3. Apply this delta locally (post-multiply) to the current effective transform
        new_eff_tf = self.effective_transform @ transform_delta
        self._set_effective_transform(new_eff_tf)

    # --- Baking Transformations ---

    @abstractmethod
    def bake_geometry(self, total_transform: np.ndarray) -> Optional[List['Primitive']]:
        """
        Applies the given *total* transformation matrix directly to the
        primitive's defining geometric data (e.g., points, center, radius).
        This *modifies* the primitive's definition.
        The effective_transform should be reset to identity *after* calling this.

        Returns:
            Optional[List[Primitive]]: If baking changes the primitive type
            (e.g., rotated Rect becomes Pline), returns a list containing
            the *new* primitive object(s) to replace the original. The caller
            (CamBamProject) is responsible for handling the replacement.
            Returns None if the primitive type remains the same and was modified in-place.
        """
        pass

    def _propagate_bake(self, bake_transform: np.ndarray) -> Optional[List['Primitive']]:
        """
        Internal helper to apply bake_geometry and recursively update children.
        This is called by CamBamProject when a bake operation is requested.

        Args:
            bake_transform: The *total* transformation matrix to bake into this primitive.

        Returns:
            Optional[List[Primitive]]: Result from self.bake_geometry.
        """
        logger.debug(f"Baking '{self.user_identifier}' with transform:\n{bake_transform}")

        # 1. Bake self
        replacement_primitives = self.bake_geometry(bake_transform) # Modify geometry

        if replacement_primitives is not None:
            # If the type changed, the original is invalid. Baking stops here for this branch.
            # Children will be handled by the project when re-parenting to the replacement.
            logger.debug(f"'{self.user_identifier}' was replaced by {len(replacement_primitives)} primitive(s) during bake.")
            # The original effective transform is irrelevant now.
            return replacement_primitives # Signal replacement to project

        # 2. Reset own effective transform (if not replaced)
        self.effective_transform = identity_matrix()

        # 3. Update children and recurse
        try:
            project = self.get_project()
            # Iterate over a copy of child IDs in case children get replaced
            child_ids_copy = list(self._child_primitive_ids)

            for child_id in child_ids_copy:
                child = project.get_primitive(child_id)
                if child:
                    # The child's new total transform *would* be:
                    # bake_transform @ child.effective_transform
                    # We bake this total transform into the child recursively.
                    child_bake_transform = bake_transform @ child.effective_transform
                    # Project handles potential replacement of the child
                    project._bake_primitive_and_handle_replacement(child, child_bake_transform)
                else:
                     logger.warning(f"Child primitive {child_id} not found during bake propagation from '{self.user_identifier}'.")
        except CamBamError as e:
             logger.error(f"Error during bake propagation for '{self.user_identifier}': {e}")

        return None # No replacement happened for *this* primitive


    # --- Copying ---
    def copy_recursive(self, project: 'CamBamProject',
                       new_identifier_map: Dict[uuid.UUID, uuid.UUID],
                       new_parent_id: Optional[uuid.UUID] = None) -> 'Primitive':
        """
        Creates a deep copy of this primitive and recursively copies its children,
        assigning new unique IDs and updating parent links.

        Args:
            project: The CamBamProject instance (needed to find children).
            new_identifier_map: A dictionary to track original UUID -> new UUID mapping
                                 to ensure consistent re-linking within the copied structure.
            new_parent_id: The UUID of the new parent for this copied primitive.

        Returns:
            The newly created copy of this primitive.
        """
        if self.internal_id in new_identifier_map:
             # Avoid infinite recursion if graph has cycles (shouldn't happen with parent links)
             # Or if called incorrectly on an already processed node
             logger.warning(f"Primitive '{self.user_identifier}' already encountered in this copy operation. Skipping duplicate copy.")
             # Need to return the *existing* copy
             return project.get_primitive(new_identifier_map[self.internal_id])


        logger.debug(f"Copying primitive '{self.user_identifier}' ({self.internal_id})")
        # 1. Deep copy self
        new_primitive = deepcopy(self)

        # 2. Assign new ID and update map
        new_primitive.internal_id = uuid.uuid4()
        new_identifier_map[self.internal_id] = new_primitive.internal_id

        # 3. Update user identifier (avoid conflicts)
        # Check if the original name exists in the project - requires project access
        # For now, just append a copy marker. Project can refine later.
        new_primitive.user_identifier = f"{self.user_identifier}_copy_{new_primitive.internal_id.hex[:4]}"

        # 4. Set new parent link
        new_primitive.parent_primitive_id = new_parent_id

        # 5. Clear children (they will be added back as copies)
        new_primitive._child_primitive_ids = set()

        # 6. Reset transient fields
        new_primitive._project_ref = None # Will be set by project upon registration

        # 7. Recursively copy children
        original_child_ids = list(self._child_primitive_ids) # Copy before modification
        for child_id in original_child_ids:
            original_child = project.get_primitive(child_id)
            if original_child:
                # Recursive call, passing the NEW parent ID
                copied_child = original_child.copy_recursive(project, new_identifier_map, new_primitive.internal_id)
                # Add the copied child's NEW ID to the NEW parent's children set
                new_primitive._child_primitive_ids.add(copied_child.internal_id)
            else:
                logger.warning(f"Child primitive {child_id} not found during recursive copy of '{self.user_identifier}'.")

        logger.debug(f"Finished copying primitive '{self.user_identifier}'. New ID: {new_primitive.internal_id}")
        return new_primitive


    # --- XML Generation ---

    def _add_common_xml_attributes(self, element: ET.Element, xml_primitive_id: int):
        """Adds common ID and Tag sub-element."""
        element.set("id", str(xml_primitive_id))
        # Format groups as comma-separated list within square brackets for Tag
        groups_str = f"[{','.join(self.groups)}]" if self.groups else "[]"
        # Combine groups and tags into the Tag element content
        tag_text = f"{groups_str}\n{self.tags}".strip()
        if tag_text: # Only add Tag element if there's content
             ET.SubElement(element, "Tag").text = tag_text

    def _add_matrix_xml(self, element: ET.Element):
        """Adds the <mat> sub-element with the total transformation matrix."""
        total_tf = self._get_total_transform()
        # Only add matrix if it's not identity
        if not np.allclose(total_tf, identity_matrix()):
            matrix_str = matrix_to_cambam_string(total_tf)
            ET.SubElement(element, "mat", {"m": matrix_str})

    @abstractmethod
    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        """Generates the XML element for the primitive."""
        pass


# --- Concrete Primitive Classes ---

@dataclass
class Pline(Primitive):
    relative_points: List[Union[Tuple[float, float], Tuple[float, float, float]]] = field(default_factory=list)
    closed: bool = False

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float, float]]:
        """Returns transformed points as (x, y, bulge). Bulge is NOT transformed."""
        if not self.relative_points:
            return []
        # Extract XY points for transformation
        rel_pts_xy = [(p[0], p[1]) for p in self.relative_points]
        abs_pts_xy = apply_transform(rel_pts_xy, total_transform)

        # Combine transformed XY with original bulge
        abs_pts_with_bulge = []
        for i, (x, y) in enumerate(abs_pts_xy):
            # Get bulge from original point tuple if it exists
            bulge = self.relative_points[i][2] if len(self.relative_points[i]) > 2 else 0.0
            abs_pts_with_bulge.append((x, y, bulge))
        return abs_pts_with_bulge

    def _calculate_bounding_box(self, geometry: List[Tuple[float, float, float]], transform: np.ndarray) -> BoundingBox:
        """Calculates bbox from vertex points. Ignores bulge extent for simplicity."""
        points_xy = [(p[0], p[1]) for p in geometry]
        if not points_xy:
            return BoundingBox()
        # TODO: Accurate bbox for transformed bulges (elliptical arcs) is complex.
        # Current implementation uses vertices only.
        if any(p[2] != 0 for p in geometry):
             logger.warning(f"Bounding box for Pline '{self.user_identifier}' with bulges may be inaccurate.")
        return BoundingBox.from_points(points_xy)

    def bake_geometry(self, total_transform: np.ndarray) -> None:
        """Applies transform to relative_points XY coordinates."""
        abs_geometry = self._calculate_absolute_geometry(total_transform)
        # Update relative points: Use transformed XY, keep original bulge
        self.relative_points = [(pt[0], pt[1], pt[2]) for pt in abs_geometry]
        return None # Modified in-place

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        pline_elem = ET.Element("pline", {"Closed": str(self.closed).lower()})
        self._add_common_xml_attributes(pline_elem, xml_primitive_id)
        pts_elem = ET.SubElement(pline_elem, "pts")
        # Write RELATIVE points to XML
        for pt in self.relative_points:
            x, y = pt[0], pt[1]
            bulge = pt[2] if len(pt) > 2 else 0.0
            ET.SubElement(pts_elem, "p", {"b": str(bulge)}).text = f"{x},{y},0" # Z is always 0 in CamBam 2D
        # Add the TOTAL transform matrix
        self._add_matrix_xml(pline_elem)
        return pline_elem

@dataclass
class Circle(Primitive):
    relative_center: Tuple[float, float] = (0.0, 0.0)
    diameter: float = 1.0

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Tuple[float, float]:
        """Returns the transformed center point."""
        return get_transformed_point(self.relative_center, total_transform)

    def _calculate_bounding_box(self, geometry: Tuple[float, float], transform: np.ndarray) -> BoundingBox:
        """Calculates bbox considering scaling."""
        cx, cy = geometry # Transformed center
        # Estimate scaled radius - use average scale factor for non-uniform scaling
        sx, sy = get_scale(transform)
        avg_scale = (abs(sx) + abs(sy)) / 2.0
        scaled_radius = (self.diameter / 2.0) * avg_scale
        if not np.isclose(sx, sy):
            logger.warning(f"Bounding box for non-uniformly scaled Circle '{self.user_identifier}' is approximate.")
        return BoundingBox(cx - scaled_radius, cy - scaled_radius, cx + scaled_radius, cy + scaled_radius)

    def bake_geometry(self, total_transform: np.ndarray) -> None:
        """Applies transform to relative_center and scales diameter."""
        self.relative_center = get_transformed_point(self.relative_center, total_transform)
        # Scale diameter by average scale factor
        sx, sy = get_scale(total_transform)
        avg_scale = (abs(sx) + abs(sy)) / 2.0
        self.diameter *= avg_scale
        if not np.isclose(sx, sy):
             logger.warning(f"Baked diameter for non-uniformly scaled Circle '{self.user_identifier}' uses average scale factor.")
        return None # Modified in-place

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
         circle_elem = ET.Element("circle", {
             "c": f"{self.relative_center[0]},{self.relative_center[1]},0", # Use relative center
             "d": str(self.diameter)
         })
         self._add_common_xml_attributes(circle_elem, xml_primitive_id)
         self._add_matrix_xml(circle_elem) # Add total transform
         return circle_elem


@dataclass
class Rect(Primitive):
    # Note: CamBam XML stores 'p' (corner), w, h. We use the same convention.
    relative_corner: Tuple[float, float] = (0.0, 0.0) # Lower-left corner
    width: float = 1.0
    height: float = 1.0

    def _get_relative_corners(self) -> List[Tuple[float, float]]:
        """Returns the four corners based on relative_corner, width, height."""
        x0, y0 = self.relative_corner
        x1 = x0 + self.width
        y1 = y0 + self.height
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        """Returns the four transformed corner points."""
        return apply_transform(self._get_relative_corners(), total_transform)

    def _calculate_bounding_box(self, geometry: List[Tuple[float, float]], transform: np.ndarray) -> BoundingBox:
        """Calculates bbox from the four transformed corners."""
        if not geometry:
            return BoundingBox()
        return BoundingBox.from_points(geometry)

    def bake_geometry(self, total_transform: np.ndarray) -> Optional[List[Primitive]]:
        """
        Applies transform. If rotation is present, converts to a Pline.
        Otherwise, updates relative_corner, width, height (if only scaled/translated).
        """
        # Check for rotation in the transformation
        if has_rotation(total_transform):
            logger.info(f"Baking rotated Rect '{self.user_identifier}' into a Pline.")
            # Convert to Pline
            abs_corners = self.get_absolute_geometry() # Get corners after full transform
            # Create new Pline with these absolute corners as its relative points
            new_pline = Pline(
                layer_id=self.layer_id,
                user_identifier=f"{self.user_identifier}_baked_pline",
                groups=deepcopy(self.groups),
                tags=deepcopy(self.tags),
                # Relative points are the absolute corners of the transformed rect
                relative_points=[(p[0], p[1]) for p in abs_corners],
                closed=True,
                # Baked primitive has identity effective transform
                effective_transform=identity_matrix(),
                # Parent link will be handled by project during replacement
                parent_primitive_id=self.parent_primitive_id
            )
            # Return the new Pline to replace this Rect
            return [new_pline]
        else:
            # No rotation, just translation and/or scaling
            abs_corners = self._calculate_absolute_geometry(total_transform)
            if not abs_corners: return None # Should not happen if width/height > 0

            # Find new min/max coords to define the baked rectangle properties
            min_x = min(p[0] for p in abs_corners)
            min_y = min(p[1] for p in abs_corners)
            max_x = max(p[0] for p in abs_corners)
            max_y = max(p[1] for p in abs_corners)

            self.relative_corner = (min_x, min_y)
            self.width = max_x - min_x
            self.height = max_y - min_y
            logger.debug(f"Baked Rect '{self.user_identifier}' in-place (no rotation). New corner: {self.relative_corner}, W: {self.width}, H: {self.height}")
            return None # Modified in-place

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        # CamBam Rect seems to be implicitly closed Pline in structure
        rect_elem = ET.Element("rect", {
            "Closed": "true", # Explicitly set, though perhaps redundant for <rect>
            "p": f"{self.relative_corner[0]},{self.relative_corner[1]},0", # Relative corner
            "w": str(self.width),
            "h": str(self.height)
        })
        self._add_common_xml_attributes(rect_elem, xml_primitive_id)
        self._add_matrix_xml(rect_elem) # Add total transform
        return rect_elem


@dataclass
class Arc(Primitive):
    relative_center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    start_angle: float = 0.0 # Degrees
    extent_angle: float = 90.0 # Degrees (sweep angle)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        """ Returns transformed center, radius, start/extent angles. """
        abs_center = get_transformed_point(self.relative_center, total_transform)

        # Calculate scaled radius and effective rotation from the transform matrix
        sx, sy = get_scale(total_transform)
        avg_scale = (abs(sx) + abs(sy)) / 2.0
        scaled_radius = self.radius * avg_scale

        # Rotation angle of the transform affects the start angle
        transform_rotation = math.degrees(math.atan2(total_transform[1, 0], total_transform[0, 0]))
        abs_start_angle = (self.start_angle + transform_rotation) % 360

        # Extent angle is generally preserved under affine transforms,
        # but mirroring flips its sign.
        det = total_transform[0, 0] * total_transform[1, 1] - total_transform[0, 1] * total_transform[1, 0]
        abs_extent_angle = self.extent_angle * np.sign(det)

        if not np.isclose(sx, sy):
             logger.warning(f"Transformed Arc '{self.user_identifier}' with non-uniform scale will be elliptical, but represented as circular Arc. Radius uses average scale.")

        return {
            "center": abs_center,
            "radius": scaled_radius,
            "start_angle": abs_start_angle,
            "extent_angle": abs_extent_angle
        }

    def _get_arc_points_for_bbox(self, geometry: Dict[str, Any]) -> List[Tuple[float, float]]:
        """ Helper to get key points of the arc for bounding box calculation. """
        cx, cy = geometry["center"]
        r = geometry["radius"]
        start_rad = math.radians(geometry["start_angle"])
        extent_rad = math.radians(geometry["extent_angle"])
        end_rad = start_rad + extent_rad

        points = []
        # Start and End points
        points.append((cx + r * math.cos(start_rad), cy + r * math.sin(start_rad)))
        points.append((cx + r * math.cos(end_rad), cy + r * math.sin(end_rad)))

        # Check for intersections with cardinal axes within the arc's sweep
        angles_to_check = [0, math.pi/2, math.pi, 3*math.pi/2] # 0, 90, 180, 270 deg
        start_norm = start_rad % (2 * math.pi)
        end_norm = end_rad % (2 * math.pi)

        for angle in angles_to_check:
            # Normalize angle check relative to start/end
            angle_norm = angle % (2 * math.pi)
            # Check if angle lies within the sweep range (handling wrap-around)
            in_range = False
            if start_norm <= end_norm: # Simple case, no wrap
                if start_norm <= angle_norm <= end_norm:
                    in_range = True
            else: # Wraps around 0/360
                 if start_norm <= angle_norm or angle_norm <= end_norm:
                     in_range = True

            # More robust check for extent > 360? Assume extent <= 360 for now.
            # Need to handle extent < 0 as well? Assume positive extent.

            if in_range:
                 points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

        return points


    def _calculate_bounding_box(self, geometry: Dict[str, Any], transform: np.ndarray) -> BoundingBox:
        """ Calculates bbox considering the arc's curve. """
        arc_points = self._get_arc_points_for_bbox(geometry)
        if not arc_points:
             # Use center point if arc is zero-length?
             return BoundingBox.from_points([geometry["center"]])
        return BoundingBox.from_points(arc_points)


    def bake_geometry(self, total_transform: np.ndarray) -> None:
        """ Applies transform to center, radius, and angles. """
        # Get the fully transformed state
        abs_geometry = self._calculate_absolute_geometry(total_transform)

        # Update relative definition based on the transformed state
        self.relative_center = abs_geometry["center"]
        self.radius = abs_geometry["radius"]
        self.start_angle = abs_geometry["start_angle"]
        self.extent_angle = abs_geometry["extent_angle"]
        return None # Modified in-place

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
         arc_elem = ET.Element("arc", {
             "p": f"{self.relative_center[0]},{self.relative_center[1]},0", # Relative center
             "r": str(self.radius),
             "s": str(self.start_angle % 360), # Normalize start angle
             "w": str(self.extent_angle)
         })
         self._add_common_xml_attributes(arc_elem, xml_primitive_id)
         self._add_matrix_xml(arc_elem) # Add total transform
         return arc_elem


@dataclass
class Points(Primitive):
    relative_points: List[Tuple[float, float]] = field(default_factory=list)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        """Returns transformed points."""
        return apply_transform(self.relative_points, total_transform)

    def _calculate_bounding_box(self, geometry: List[Tuple[float, float]], transform: np.ndarray) -> BoundingBox:
        """Calculates bbox from the transformed points."""
        if not geometry:
            return BoundingBox()
        return BoundingBox.from_points(geometry)

    def bake_geometry(self, total_transform: np.ndarray) -> None:
        """Applies transform to relative_points."""
        self.relative_points = apply_transform(self.relative_points, total_transform)
        return None # Modified in-place

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        points_elem = ET.Element("points")
        self._add_common_xml_attributes(points_elem, xml_primitive_id)
        pts_elem = ET.SubElement(points_elem, "pts")
        for x, y in self.relative_points: # Use relative points
            ET.SubElement(pts_elem, "p").text = f"{x},{y}" # CamBam uses Z=0 implicitly? Check format. Seems so.
        self._add_matrix_xml(points_elem) # Add total transform
        return points_elem


@dataclass
class Text(Primitive):
    text_content: str = "Text"
    relative_position: Tuple[float, float] = (0.0, 0.0) # Anchor point
    height: float = 10.0 # Text height in drawing units
    font: str = 'Arial'
    style: str = '' # e.g., 'Bold', 'Italic' - Comma separated? Check CamBam format. Assume single string.
    line_spacing: float = 1.0 # Multiplier
    align_horizontal: str = 'Center' # Cambam uses: Left, Center, Right
    align_vertical: str = 'Middle' # Cambam uses: Top, Middle, Bottom

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        """ Returns transformed position and scaled height. """
        abs_position = get_transformed_point(self.relative_position, total_transform)
        sx, sy = get_scale(total_transform)
        # Use Y scale factor for height, consistent with CamBam behavior
        scaled_height = self.height * abs(sy)
        # Rotation doesn't change the definition, it's handled by the matrix
        # We could store effective rotation here if needed, but matrix handles it.
        return {
            "position": abs_position,
            "height": scaled_height
        }

    def _calculate_bounding_box(self, geometry: Dict[str, Any], transform: np.ndarray) -> BoundingBox:
        """ Calculates an *approximate* bounding box for text. """
        logger.warning(f"Bounding box for Text '{self.user_identifier}' is approximate and ignores rotation.")
        px, py = geometry["position"]
        scaled_height = geometry["height"]

        # Very rough estimate based on character count and height
        # This doesn't account for font metrics, alignment, or rotation.
        num_lines = len(self.text_content.split('\n'))
        max_line_len = max(len(line) for line in self.text_content.split('\n')) if self.text_content else 0
        # Aspect ratio guess (depends heavily on font)
        char_aspect_ratio = 0.6
        est_width = scaled_height * max_line_len * char_aspect_ratio
        est_height = scaled_height * num_lines * self.line_spacing

        # Adjust based on alignment (relative to the estimated block)
        dx, dy = 0, 0
        if self.align_horizontal == 'Left': dx = est_width / 2
        elif self.align_horizontal == 'Right': dx = -est_width / 2
        if self.align_vertical == 'Top': dy = -est_height / 2
        elif self.align_vertical == 'Bottom': dy = est_height / 2

        center_x, center_y = px + dx, py + dy
        min_x = center_x - est_width / 2
        min_y = center_y - est_height / 2
        max_x = center_x + est_width / 2
        max_y = center_y + est_height / 2

        return BoundingBox(min_x, min_y, max_x, max_y)

    def bake_geometry(self, total_transform: np.ndarray) -> None:
        """ Text geometry is not baked. Transformations are handled by the matrix in CamBam. """
        logger.debug(f"Skipping bake_geometry for Text primitive '{self.user_identifier}'. Transformations applied via matrix.")
        # Optionally, we could bake position and height scale if needed,
        # but standard CamBam behavior relies on the matrix for text transforms.
        # Let's update position and height based on scale/translation only.
        abs_pos = get_transformed_point(self.relative_position, total_transform)
        _ , translation_y = get_translation(total_transform) # Unused dx?
        sx, sy = get_scale(total_transform)

        self.relative_position = abs_pos
        self.height *= abs(sy) # Scale height by Y scale factor

        if has_rotation(total_transform):
             logger.warning(f"Rotation component ignored during bake for Text '{self.user_identifier}'. Matrix will still apply it.")

        return None # Modified in-place (partially)

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        # CamBam text alignment mapping
        cb_align_map_h = {"Left": "left", "Center": "center", "Right": "right"}
        cb_align_map_v = {"Top": "top", "Middle": "middle", "Bottom": "bottom"}
        align_str = f"{cb_align_map_v.get(self.align_vertical, 'middle')},{cb_align_map_h.get(self.align_horizontal, 'center')}"

        text_elem = ET.Element("text", {
            # CamBam uses p1 and p2, often the same for single point anchor
            "p1": f"{self.relative_position[0]},{self.relative_position[1]},0",
            "p2": f"{self.relative_position[0]},{self.relative_position[1]},0",
            "Height": str(self.height),
            "Font": self.font,
            "linespace": str(self.line_spacing),
            "align": align_str,
            "style": self.style
        })
        self._add_common_xml_attributes(text_elem, xml_primitive_id)
        # Text content goes directly into the element
        text_elem.text = self.text_content
        self._add_matrix_xml(text_elem) # Add total transform matrix
        return text_elem


# --- MOP Base and Concrete Classes ---

MopType = TypeVar('MopType', bound='Mop')
Identifiable = Union[str, uuid.UUID, CamBamEntity] # For project method hints

@dataclass
class Mop(CamBamEntity, ABC):
    """Abstract Base Class for Machine Operations (MOPs)."""
    part_id: uuid.UUID # Must belong to a Part
    name: str # User-defined name for the MOP
    # Source can be a group name or list of primitive UUIDs
    pid_source: Union[str, List[uuid.UUID]]
    enabled: bool = True

    # Common MOP parameters (with typical CamBam defaults where applicable)
    target_depth: Optional[float] = None # If None, uses Part/Project default? Or error? Let's make it optional.
    depth_increment: Optional[float] = None # Step down. If None, full depth in one pass.
    stock_surface: float = 0.0
    roughing_clearance: float = 0.0 # Material to leave (negative for overcut)
    clearance_plane: float = 15.0 # Z height for rapid moves
    spindle_direction: str = 'CW' # Clockwise
    spindle_speed: Optional[int] = None # If None, uses Part/Project default
    velocity_mode: str = 'ExactStop' # Or ConstantVelocity
    work_plane: str = 'XY' # Typically XY
    optimisation_mode: str = 'Standard' # Or Basic, Experimental
    tool_diameter: Optional[float] = None # If None, uses Part/Project default
    tool_number: int = 0 # Tool index in controller
    tool_profile: str = 'EndMill' # Cylinder, BallNose, Vcutter etc.
    plunge_feedrate: float = 1000.0 # Feedrate for Z moves down
    cut_feedrate: Optional[float] = None # Feedrate for XY cutting moves. If None, calculated.
    max_crossover_distance: float = 0.7 # Max distance for rapids vs feed moves (as fraction of tool dia)
    custom_mop_header: str = "" # GCode to insert before MOP
    custom_mop_footer: str = "" # GCode to insert after MOP

    # Internal field for resolved XML IDs (set by writer)
    _resolved_xml_primitive_ids: List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not self.name:
             self.name = f"{self.__class__.__name__}_{self.internal_id.hex[:6]}"


    def _get_effective_param(self, param_name: str, project: 'CamBamProject') -> Any:
        """ Gets parameter value, falling back to Part and then Project defaults. """
        mop_value = getattr(self, param_name, None)
        if mop_value is not None:
            return mop_value

        part = project.get_part(self.part_id)
        if part:
            # Map MOP param name to Part param name if different
            part_param_map = {'tool_diameter': 'default_tool_diameter', 'spindle_speed': 'default_spindle_speed'}
            part_attr = part_param_map.get(param_name)
            if part_attr:
                part_value = getattr(part, part_attr, None)
                if part_value is not None:
                    return part_value

        # Fallback to project defaults (add if needed)
        project_param_map = {'tool_diameter': 'default_tool_diameter'}
        project_attr = project_param_map.get(param_name)
        if project_attr:
             project_value = getattr(project, project_attr, None)
             if project_value is not None:
                 return project_value

        # If still None, return None (caller must handle)
        return None

    def _calculate_cut_feedrate(self, project: 'CamBamProject') -> float:
        """ Calculates a default cut feedrate if not explicitly set. """
        if self.cut_feedrate is not None:
            return self.cut_feedrate

        # Example calculation (simple heuristic based on depth)
        td = self._get_effective_param('target_depth', project)
        # Requires target depth to be set somewhere (MOP or Part/Project defaults)
        if td is None:
             logger.warning(f"MOP '{self.name}': TargetDepth not set. Using fallback cut feedrate 3000.")
             return 3000.0 # Default fallback

        # Heuristic: Feedrate increases with shallower depth
        # Adjust coefficients based on typical materials/machines
        # Example: 6500 at depth 0, increases by 350 per mm of depth
        calculated_feedrate = round(-350 * td + 6500, 0) # td is usually negative

        # Ensure a minimum feedrate
        min_feedrate = 1000.0
        final_feedrate = max(calculated_feedrate, min_feedrate)
        logger.debug(f"MOP '{self.name}': Calculated cut feedrate = {final_feedrate} (based on depth {td})")
        return final_feedrate

    def _add_common_mop_elements(self, mop_root_elem: ET.Element, project: 'CamBamProject'):
        """Adds common XML elements shared by most MOP types."""

        # Helper to add element with state attribute
        def add_elem(parent, name, value, state="Value"):
             # Only add if value is not None
             if value is not None:
                 ET.SubElement(parent, name, {"state": state}).text = str(value)
             # Else CamBam might use default based on parent Part/global options

        add_elem(mop_root_elem, "Name", self.name)

        # Resolve parameters considering fallbacks
        eff_td = self._get_effective_param('target_depth', project)
        add_elem(mop_root_elem, "TargetDepth", eff_td, state="Value" if self.target_depth is not None else "Default")

        # Default DepthIncrement is full TargetDepth if not specified
        eff_di = self.depth_increment
        if eff_di is None and eff_td is not None:
             eff_di = abs(eff_td) # Use absolute value for increment
        add_elem(mop_root_elem, "DepthIncrement", eff_di, state="Value" if self.depth_increment is not None else "Default")

        add_elem(mop_root_elem, "StockSurface", self.stock_surface)
        add_elem(mop_root_elem, "RoughingClearance", self.roughing_clearance)
        add_elem(mop_root_elem, "ClearancePlane", self.clearance_plane)
        add_elem(mop_root_elem, "SpindleDirection", self.spindle_direction)

        eff_ss = self._get_effective_param('spindle_speed', project)
        add_elem(mop_root_elem, "SpindleSpeed", eff_ss, state="Value" if self.spindle_speed is not None else "Default")

        # SpindleRange seems constant in examples?
        add_elem(mop_root_elem, "SpindleRange", "0")
        add_elem(mop_root_elem, "VelocityMode", self.velocity_mode)
        add_elem(mop_root_elem, "WorkPlane", self.work_plane)
        add_elem(mop_root_elem, "OptimisationMode", self.optimisation_mode)

        eff_tool_dia = self._get_effective_param('tool_diameter', project)
        if eff_tool_dia is None:
             logger.error(f"MOP '{self.name}' has no effective ToolDiameter set (MOP, Part, or Project). XML may be invalid.")
             # Add element anyway, CamBam might complain or use a last resort default
             add_elem(mop_root_elem, "ToolDiameter", None, state="Default")
        else:
             add_elem(mop_root_elem, "ToolDiameter", eff_tool_dia, state="Value" if self.tool_diameter is not None else "Default")

        add_elem(mop_root_elem, "ToolNumber", self.tool_number)
        add_elem(mop_root_elem, "ToolProfile", self.tool_profile)
        add_elem(mop_root_elem, "PlungeFeedrate", self.plunge_feedrate)

        eff_cf = self._calculate_cut_feedrate(project)
        add_elem(mop_root_elem, "CutFeedrate", eff_cf, state="Value" if self.cut_feedrate is not None else "Calculated") # Use 'Value' state for consistency

        add_elem(mop_root_elem, "MaxCrossoverDistance", self.max_crossover_distance)
        if self.custom_mop_header:
            add_elem(mop_root_elem, "CustomMOPHeader", self.custom_mop_header)
        if self.custom_mop_footer:
            add_elem(mop_root_elem, "CustomMOPFooter", self.custom_mop_footer)

        # Add primitive references (using resolved integer IDs)
        primitive_container = ET.SubElement(mop_root_elem, "primitive")
        if self._resolved_xml_primitive_ids:
            for pid in self._resolved_xml_primitive_ids:
                ET.SubElement(primitive_container, "prim").text = str(pid)
        elif isinstance(self.pid_source, str): # Group name specified but empty?
             logger.warning(f"MOP '{self.name}' references group '{self.pid_source}' which resolved to zero primitives.")
             # Add empty container anyway
        elif isinstance(self.pid_source, list) and not self.pid_source: # Empty list
             logger.warning(f"MOP '{self.name}' has an empty list as pid_source.")
             # Add empty container


    def _add_lead_in_out_elements(self, parent_elem: ET.Element,
                                  lead_type: Optional[str] = "Spiral", # None, Spiral, Tangent, Ramp
                                  spiral_angle: float = 30.0, # Degrees
                                  tangent_radius: float = 0.0, # Radius for tangent leadin
                                  feedrate: float = 0.0): # 0 = Use Plunge Feedrate
        """Adds LeadInMove and LeadOutMove elements (often identical)."""
        if lead_type is None or lead_type == "None":
             # Add element with type None if needed? Check CamBam behavior.
             # Let's assume omitting the element means no leadin.
             # If specific 'None' type is required, uncomment below:
             # lead_in = ET.SubElement(parent_elem, "LeadInMove", {"state": "Value"})
             # ET.SubElement(lead_in, "LeadInType").text = "None"
             # lead_out = ET.SubElement(parent_elem, "LeadOutMove", {"state": "Value"})
             # ET.SubElement(lead_out, "LeadInType").text = "None"
             return

        # Lead In
        lead_in = ET.SubElement(parent_elem, "LeadInMove", {"state": "Value"})
        ET.SubElement(lead_in, "LeadInType").text = lead_type
        if lead_type == "Spiral":
            ET.SubElement(lead_in, "SpiralAngle").text = str(spiral_angle)
        elif lead_type == "Tangent":
             ET.SubElement(lead_in, "TangentRadius").text = str(tangent_radius)
        # Add Ramp options if needed
        ET.SubElement(lead_in, "LeadInFeedrate").text = str(feedrate)

        # Lead Out (typically mirrors Lead In settings)
        lead_out = ET.SubElement(parent_elem, "LeadOutMove", {"state": "Value"})
        ET.SubElement(lead_out, "LeadInType").text = lead_type # Note: XML uses 'LeadInType' for both
        if lead_type == "Spiral":
             ET.SubElement(lead_out, "SpiralAngle").text = str(spiral_angle)
        elif lead_type == "Tangent":
             ET.SubElement(lead_out, "TangentRadius").text = str(tangent_radius)
        ET.SubElement(lead_out, "LeadInFeedrate").text = str(feedrate) # Note: XML uses 'LeadInFeedrate'

    @abstractmethod
    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        """Generates the specific XML element for the MOP subclass."""
        pass


@dataclass
class ProfileMop(Mop):
    # Profile specific parameters
    stepover: float = 0.4 # Fraction of tool diameter for roughing passes if needed
    profile_side: str = 'Inside' # Inside, Outside, On
    milling_direction: str = 'Conventional' # Conventional, Climb
    collision_detection: bool = True
    corner_overcut: bool = False # Extends path at sharp corners
    lead_in_type: Optional[str] = 'Spiral' # See _add_lead_in_out_elements
    lead_in_spiral_angle: float = 30.0
    lead_in_tangent_radius: float = 0.0 # Used if lead_in_type is Tangent
    final_depth_increment: Optional[float] = 0.0 # Finishing pass depth, 0 = no final pass
    cut_ordering: str = 'DepthFirst' # DepthFirst, LevelFirst
    # Holding Tabs
    tab_method: str = 'None' # None, Automatic, Manual (Manual needs Points list)
    tab_width: float = 6.0
    tab_height: float = 1.5 # Height relative to bottom of cut
    tab_min_tabs: int = 3
    tab_max_tabs: int = 3
    tab_distance: float = 40.0 # Approx distance between tabs for Automatic
    tab_size_threshold: float = 4.0 # Min shape size for tabs
    tab_use_leadins: bool = False # Use lead moves for tabs
    tab_style: str = 'Square' # Square, Triangular, Ramp

    XML_TAG = "profile"

    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        mop_elem = ET.Element(self.XML_TAG, {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project)

        # Profile specific elements
        ET.SubElement(mop_elem, "StepOver", {"state": "Value"}).text = str(self.stepover)
        ET.SubElement(mop_elem, "InsideOutside", {"state": "Value"}).text = self.profile_side
        ET.SubElement(mop_elem, "MillingDirection", {"state": "Value"}).text = self.milling_direction
        ET.SubElement(mop_elem, "CollisionDetection", {"state": "Value"}).text = str(self.collision_detection).lower()
        ET.SubElement(mop_elem, "CornerOvercut", {"state": "Value"}).text = str(self.corner_overcut).lower()

        self._add_lead_in_out_elements(mop_elem,
                                         lead_type=self.lead_in_type,
                                         spiral_angle=self.lead_in_spiral_angle,
                                         tangent_radius=self.lead_in_tangent_radius)

        # Use 0.0 if None for XML
        final_di = self.final_depth_increment if self.final_depth_increment is not None else 0.0
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": "Value"}).text = str(final_di)
        ET.SubElement(mop_elem, "CutOrdering", {"state": "Value"}).text = self.cut_ordering

        # Holding Tabs
        tabs = ET.SubElement(mop_elem, "HoldingTabs", {"state": "Value"})
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
            # Manual tab points would go here if implemented (<TabPoints><Point3F>...)

        return mop_elem


@dataclass
class PocketMop(Mop):
    # Pocket specific parameters
    stepover: float = 0.4 # Fraction of tool diameter
    stepover_feedrate: str = 'Plunge Feedrate' # Or 'Cut Feedrate'
    milling_direction: str = 'Conventional' # Conventional, Climb
    collision_detection: bool = True
    lead_in_type: Optional[str] = 'Spiral'
    lead_in_spiral_angle: float = 30.0
    lead_in_tangent_radius: float = 0.0
    final_depth_increment: Optional[float] = 0.0
    cut_ordering: str = 'DepthFirst'
    region_fill_style: str = 'InsideOutsideOffsets' # HorizontalScan, VerticalScan
    finish_stepover: float = 0.0 # Stepover for final finishing pass (0 = no finish pass)
    finish_stepover_at_target_depth: bool = False # Apply finish pass only at final depth
    roughing_finishing: str = 'Roughing' # Roughing, Finishing, RoughingThenFinishing

    XML_TAG = "pocket"

    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        mop_elem = ET.Element(self.XML_TAG, {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project)

        # Pocket specific elements
        ET.SubElement(mop_elem, "StepOver", {"state": "Value"}).text = str(self.stepover)
        ET.SubElement(mop_elem, "StepoverFeedrate", {"state": "Value"}).text = self.stepover_feedrate
        ET.SubElement(mop_elem, "MillingDirection", {"state": "Value"}).text = self.milling_direction
        ET.SubElement(mop_elem, "CollisionDetection", {"state": "Value"}).text = str(self.collision_detection).lower()

        self._add_lead_in_out_elements(mop_elem,
                                         lead_type=self.lead_in_type,
                                         spiral_angle=self.lead_in_spiral_angle,
                                         tangent_radius=self.lead_in_tangent_radius)

        final_di = self.final_depth_increment if self.final_depth_increment is not None else 0.0
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": "Value"}).text = str(final_di)
        ET.SubElement(mop_elem, "CutOrdering", {"state": "Value"}).text = self.cut_ordering
        ET.SubElement(mop_elem, "RegionFillStyle", {"state": "Value"}).text = self.region_fill_style
        ET.SubElement(mop_elem, "FinishStepover", {"state": "Value"}).text = str(self.finish_stepover)
        ET.SubElement(mop_elem, "FinishStepoverAtTargetDepth", {"state": "Value"}).text = str(self.finish_stepover_at_target_depth).lower()
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": "Value"}).text = self.roughing_finishing
        # Pocket StartPoint - Usually Default (center) or user defined Point List ID
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"}) # TODO: Allow setting start point

        return mop_elem

@dataclass
class EngraveMop(Mop):
    # Engrave specific parameters
    roughing_finishing: str = 'Roughing' # Roughing, Finishing
    final_depth_increment: Optional[float] = 0.0
    cut_ordering: str = 'DepthFirst'

    XML_TAG = "engrave"

    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        mop_elem = ET.Element(self.XML_TAG, {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project)

        # Engrave specific elements
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": "Value"}).text = self.roughing_finishing
        final_di = self.final_depth_increment if self.final_depth_increment is not None else 0.0
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": "Value"}).text = str(final_di)
        ET.SubElement(mop_elem, "CutOrdering", {"state": "Value"}).text = self.cut_ordering
        # Engrave StartPoint - Often irrelevant as it follows the line
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"}) # Or "Value"? Check CamBam

        return mop_elem

@dataclass
class DrillMop(Mop):
    # Drill specific parameters
    drilling_method: str = 'CannedCycle' # CannedCycle, SpiralMill, SpiralMill_CW, CustomScript
    # HoleDiameter needed for SpiralMill, optional for CannedCycle (uses point size?)
    hole_diameter: Optional[float] = None
    # Spiral Mill specific
    drill_lead_out: bool = False
    spiral_flat_base: bool = True
    lead_out_length: float = 0.0 # Z lift before retract for spiral
    # Canned Cycle specific
    peck_distance: float = 0.0 # Depth per peck (0 = no peck)
    retract_height: float = 5.0 # Z height for retract between pecks/holes (relative to stock?)
    dwell: float = 0.0 # Dwell time at bottom (seconds)
    # Custom Script
    custom_script: str = ""

    XML_TAG = "drill"

    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        mop_elem = ET.Element(self.XML_TAG, {"Enabled": str(self.enabled).lower()})
        # Drill MOPs have slightly different common defaults/params
        # Override some common settings? E.g. ToolProfile might be 'Drill'
        # Let's assume user sets ToolProfile correctly via common params
        self._add_common_mop_elements(mop_elem, project)

        # Drill specific elements
        ET.SubElement(mop_elem, "DrillingMethod", {"state": "Value"}).text = self.drilling_method

        # Conditional parameters based on method
        if self.drilling_method.startswith("SpiralMill"):
            ET.SubElement(mop_elem, "DrillLeadOut", {"state": "Value"}).text = str(self.drill_lead_out).lower()
            ET.SubElement(mop_elem, "SpiralFlatBase", {"state": "Value"}).text = str(self.spiral_flat_base).lower()
            ET.SubElement(mop_elem, "LeadOutLength", {"state": "Value"}).text = str(self.lead_out_length)
            # HoleDiameter is crucial for spiral
            hd = self.hole_diameter
            state = "Value" if hd is not None else "Default" # Should probably be required
            if hd is None:
                logger.warning(f"Drill MOP '{self.name}' uses SpiralMill but HoleDiameter is not set.")
                # Add element as Default, CamBam might error or use tool diameter
                ET.SubElement(mop_elem, "HoleDiameter", {"state": "Default"}).text = str(self._get_effective_param('tool_diameter', project)) # Guess
            else:
                 ET.SubElement(mop_elem, "HoleDiameter", {"state": state}).text = str(hd)

        elif self.drilling_method == "CannedCycle":
            ET.SubElement(mop_elem, "PeckDistance", {"state": "Value"}).text = str(self.peck_distance)
            ET.SubElement(mop_elem, "RetractHeight", {"state": "Value"}).text = str(self.retract_height)
            ET.SubElement(mop_elem, "Dwell", {"state": "Value"}).text = str(self.dwell)
            # HoleDiameter is often optional for canned cycles (can use point size)
            if self.hole_diameter is not None:
                 ET.SubElement(mop_elem, "HoleDiameter", {"state": "Value"}).text = str(self.hole_diameter)


        elif self.drilling_method == "CustomScript":
             state = "Value" if self.custom_script else "Default"
             ET.SubElement(mop_elem, "CustomScript", {"state": state}).text = self.custom_script


        # Other common Drill params (often default)
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"})
        # RoughingFinishing usually just Roughing for drill
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": "Value"}).text = "Roughing"

        return mop_elem


# --- Type mapping for XML Reader ---
PRIMITIVE_TAG_MAP: Dict[str, Type[Primitive]] = {
    "pline": Pline,
    "circle": Circle,
    "rect": Rect,
    "arc": Arc,
    "points": Points,
    "text": Text,
}

MOP_TAG_MAP: Dict[str, Type[Mop]] = {
    "profile": ProfileMop,
    "pocket": PocketMop,
    "engrave": EngraveMop,
    "drill": DrillMop,
    # Add other MOP types like VEngrave, 3DSurface etc. if needed
}