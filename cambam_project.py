"""
cambam_project.py

Defines the CamBamProject class which is responsible for managing layers, parts,
primitives and machine operations (MOPs). It provides methods for adding entities,
applying transformations, transferring entities between projects, and copying
entities recursively.
"""

import os
import pickle
import uuid
import weakref
import logging
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Optional, Union, Type, TypeVar, Any

from cad_transformations import (translation_matrix, identity_matrix, rotation_matrix_deg, 
scale_matrix, mirror_x_matrix, mirror_y_matrix, skew_matrix, rotation_matrix_rad)
from cambam_entities import (
    CamBamEntity, Layer, Part, Primitive, XmlPrimitiveIdResolver,
    Pline, Circle, Rect, Arc, Points, Text, Mop, ProfileMop, PocketMop, EngraveMop, DrillMop, BoundingBox
)

logger = logging.getLogger(__name__)

Identifiable = Union[str, uuid.UUID, CamBamEntity]
T = TypeVar('T', bound=CamBamEntity)
M = TypeVar('M', bound=Mop)

class CamBamProject:
    """
    Represents a complete CamBam project.
    Manages layers, parts, primitives and MOPs.
    """
    def __init__(self, project_name: str, default_tool_diameter: float = 6.0):
        self.project_name: str = project_name
        self.default_tool_diameter: float = default_tool_diameter
        self._primitives: Dict[uuid.UUID, Primitive] = {}
        self._layers: Dict[uuid.UUID, Layer] = {}
        self._parts: Dict[uuid.UUID, Part] = {}
        self._mops: Dict[uuid.UUID, Mop] = {}
        self._layer_order: List[uuid.UUID] = []
        self._part_order: List[uuid.UUID] = []
        self._identifier_registry: Dict[str, uuid.UUID] = {}
        self._primitive_groups: Dict[str, Set[uuid.UUID]] = {"unassigned": set()}
        self._transform_stack: List[Any] = [identity_matrix()]
        self._cursor: Tuple[float, float] = (0.0, 0.0)
        logger.info(f"Initialized CamBamProject: {self.project_name}")

    def _resolve_identifier(self, identifier: Identifiable, entity_type: Optional[Type[T]] = None) -> Optional[uuid.UUID]:
        target_uuid: Optional[uuid.UUID] = None
        if isinstance(identifier, uuid.UUID):
            target_uuid = identifier
        elif isinstance(identifier, str):
            target_uuid = self._identifier_registry.get(identifier)
        elif isinstance(identifier, CamBamEntity):
            target_uuid = identifier.internal_id
        else:
            logger.error(f"Invalid identifier type: {type(identifier)}")
            return None
        if target_uuid and entity_type:
            entity = (self._primitives.get(target_uuid) or self._layers.get(target_uuid) or
                      self._parts.get(target_uuid) or self._mops.get(target_uuid))
            if not isinstance(entity, entity_type):
                logger.debug(f"Identifier '{identifier}' resolved but type mismatch.")
                return None
        return target_uuid

    def _register_entity(self, entity: T, registry: Dict[uuid.UUID, T]) -> None:
        if entity.internal_id in registry:
            return
        registry[entity.internal_id] = entity
        if entity.user_identifier:
            if entity.user_identifier not in self._identifier_registry or self._identifier_registry[entity.user_identifier] == entity.internal_id:
                self._identifier_registry[entity.user_identifier] = entity.internal_id
            else:
                logger.warning(f"Identifier '{entity.user_identifier}' conflict; not re-registering.")

    def _unregister_entity(self, entity_id: uuid.UUID, registry: Dict[uuid.UUID, T]) -> Optional[T]:
        entity = registry.pop(entity_id, None)
        if entity and entity.user_identifier in self._identifier_registry:
            if self._identifier_registry[entity.user_identifier] == entity_id:
                self._identifier_registry.pop(entity.user_identifier, None)
        return entity

    def _register_primitive(self, primitive: Primitive):
        self._register_entity(primitive, self._primitives)
        primitive._project_ref = weakref.ref(self)
        layer = self.get_layer(primitive.layer_id)
        if layer:
            layer.primitive_ids.add(primitive.internal_id)
        else:
            logger.error(f"Primitive {primitive.user_identifier} references non-existent layer {primitive.layer_id}.")
        if primitive.groups:
            for group_name in primitive.groups:
                self._primitive_groups.setdefault(group_name, set()).add(primitive.internal_id)
                self._primitive_groups.setdefault("unassigned", set()).discard(primitive.internal_id)
        else:
            self._primitive_groups.setdefault("unassigned", set()).add(primitive.internal_id)
        if primitive.parent_primitive_id:
            parent = self.get_primitive(primitive.parent_primitive_id)
            if parent:
                parent._child_primitive_ids.add(primitive.internal_id)
            else:
                logger.warning(f"Invalid parent for {primitive.user_identifier}; clearing link.")
                primitive.parent_primitive_id = None

    def _unregister_primitive(self, primitive_id: uuid.UUID) -> Optional[Primitive]:
        primitive = self.get_primitive(primitive_id)
        if not primitive:
            return None
        for group in self._primitive_groups.values():
            group.discard(primitive_id)
        layer = self.get_layer(primitive.layer_id)
        if layer:
            layer.primitive_ids.discard(primitive_id)
        return self._unregister_entity(primitive_id, self._primitives)

    def _find_insert_position(self, target_identifier: Optional[Identifiable],
                              order_list: List[uuid.UUID],
                              place_last: bool) -> int:
        if target_identifier is None:
            return len(order_list) if place_last else 0
        target_uuid = self._resolve_identifier(target_identifier)
        if target_uuid is None or target_uuid not in order_list:
            logger.warning(f"Target identifier '{target_identifier}' not found for ordering. Appending.")
            return len(order_list)
        index = order_list.index(target_uuid)
        return index + 1 if place_last else index

    @property
    def current_transform(self) -> Any:
        return self._transform_stack[-1]

    def set_transform(self, matrix: Any) -> None:
        self._transform_stack[-1] = matrix.copy()

    def push_transform(self, matrix: Optional[Any] = None) -> None:
        current_top = self.current_transform
        new_top = current_top.copy() if matrix is None else current_top @ matrix
        self._transform_stack.append(new_top)

    def pop_transform(self) -> Any:
        if len(self._transform_stack) > 1:
            return self._transform_stack.pop()
        else:
            logger.warning("Cannot pop the base transform.")
            return self._transform_stack[0]

    def get_entity_by_identifier(self, identifier: str) -> Optional[CamBamEntity]:
        uid = self._resolve_identifier(identifier)
        if not uid:
            return None
        return (self._primitives.get(uid) or self._layers.get(uid) or
                self._parts.get(uid) or self._mops.get(uid))

    def set_cursor(self, x: float, y: float) -> None:
        self._cursor = (x, y)
        logger.debug(f"Cursor set to: {self._cursor}")

    def reset_cursor(self) -> None:
        self.set_cursor(0.0, 0.0)

    def get_primitive(self, primitive_id: uuid.UUID) -> Optional[Primitive]:
        return self._primitives.get(primitive_id)

    def get_layer(self, layer_id: uuid.UUID) -> Optional[Layer]:
        return self._layers.get(layer_id)

    def get_part(self, part_id: uuid.UUID) -> Optional[Part]:
        return self._parts.get(part_id)

    def get_mop(self, mop_id: uuid.UUID) -> Optional[Mop]:
        return self._mops.get(mop_id)

    def get_layer_by_identifier(self, identifier: Identifiable) -> Optional[Layer]:
        if isinstance(identifier, Layer):
            if identifier.internal_id not in self._layers:
                self._register_entity(identifier, self._layers)
                self._layer_order.append(identifier.internal_id)
            return identifier
        uid = self._resolve_identifier(identifier, Layer)
        result = self._layers.get(uid) if uid else None
        if result and uid not in self._layer_order:
            self._layer_order.append(uid)
        return result

    def get_part_by_identifier(self, identifier: Identifiable) -> Optional[Part]:
        if isinstance(identifier, Part):
            if identifier.internal_id not in self._parts:
                self._register_entity(identifier, self._parts)
                self._part_order.append(identifier.internal_id)
            return identifier
        uid = self._resolve_identifier(identifier, Part)
        result = self._parts.get(uid) if uid else None
        if result and uid not in self._part_order:
            self._part_order.append(uid)
        return result

    def get_mop_by_identifier(self, identifier: Identifiable, part_context: Optional[Identifiable] = None) -> Optional[Mop]:
        uid = self._resolve_identifier(identifier, Mop)
        if not uid:
            return None
        mop = self._mops.get(uid)
        if mop and part_context:
            part_uid = self._resolve_identifier(part_context, Part)
            if mop.part_id != part_uid:
                return None
        return mop

    def list_primitives(self) -> List[Primitive]:
        return list(self._primitives.values())

    def list_layers(self) -> List[Layer]:
        return [self._layers[uid] for uid in self._layer_order if uid in self._layers]

    def list_parts(self) -> List[Part]:
        return [self._parts[uid] for uid in self._part_order if uid in self._parts]

    def list_mops(self, part_identifier: Optional[Identifiable] = None) -> List[Mop]:
        if part_identifier:
            part = self.get_part_by_identifier(part_identifier)
            if not part:
                return []
            return [self._mops[uid] for uid in part.mop_ids if uid in self._mops]
        return list(self._mops.values())

    def list_groups(self) -> List[str]:
        return list(self._primitive_groups.keys())

    def get_primitives_in_group(self, group_name: str) -> List[Primitive]:
        uuids = self._primitive_groups.get(group_name, set())
        return [self._primitives[uid] for uid in uuids if uid in self._primitives]

    # --- Entity Creation Methods ---

    def add_layer(self, identifier: str, color: str = 'Green', alpha: float = 1.0,
                  pen_width: float = 1.0, visible: bool = True, locked: bool = False,
                  target_identifier: Optional[Identifiable] = None, place_last: bool = True) -> Layer:
        existing = self.get_layer_by_identifier(identifier)
        if existing:
            existing.color = color
            existing.alpha = alpha
            existing.pen_width = pen_width
            existing.visible = visible
            existing.locked = locked
            self._register_entity(existing, self._layers)
            if existing.internal_id not in self._layer_order:
                self._layer_order.append(existing.internal_id)
            return existing
        new_layer = Layer(user_identifier=identifier, color=color, alpha=alpha, pen_width=pen_width, visible=visible, locked=locked)
        self._register_entity(new_layer, self._layers)
        pos = self._find_insert_position(target_identifier, self._layer_order, place_last)
        self._layer_order.insert(pos, new_layer.internal_id)
        return new_layer

    def add_part(self, identifier: str, enabled: bool = True, stock_thickness: float = 12.5,
                 stock_width: float = 1220.0, stock_height: float = 2440.0,
                 stock_material: str = "MDF", stock_color: str = "210,180,140",
                 machining_origin: Tuple[float, float] = (0.0, 0.0),
                 default_tool_diameter: Optional[float] = None,
                 default_spindle_speed: Optional[int] = None,
                 target_identifier: Optional[Identifiable] = None, place_last: bool = True) -> Part:
        existing = self.get_part_by_identifier(identifier)
        if existing:
            existing.enabled = enabled
            existing.stock_thickness = stock_thickness
            existing.stock_width = stock_width
            existing.stock_height = stock_height
            existing.stock_material = stock_material
            existing.stock_color = stock_color
            existing.machining_origin = machining_origin
            existing.default_tool_diameter = default_tool_diameter
            existing.default_spindle_speed = default_spindle_speed
            self._register_entity(existing, self._parts)
            if existing.internal_id not in self._part_order:
                self._part_order.append(existing.internal_id)
            return existing
        new_part = Part(user_identifier=identifier, enabled=enabled, stock_thickness=stock_thickness,
                        stock_width=stock_width, stock_height=stock_height, stock_material=stock_material,
                        stock_color=stock_color, machining_origin=machining_origin,
                        default_tool_diameter=default_tool_diameter, default_spindle_speed=default_spindle_speed)
        self._register_entity(new_part, self._parts)
        pos = self._find_insert_position(target_identifier, self._part_order, place_last)
        self._part_order.insert(pos, new_part.internal_id)
        return new_part

    def _add_primitive_internal(self, PrimitiveClass: Type[Primitive],
                                layer_identifier: Identifiable,
                                identifier: Optional[str] = None,
                                groups: Optional[List[str]] = None,
                                description: str = "",
                                parent_identifier: Optional[Identifiable] = None,
                                **kwargs) -> Primitive:
        layer = self.get_layer_by_identifier(layer_identifier)
        if not layer:
            if isinstance(layer_identifier, str):
                layer = self.add_layer(layer_identifier)
            else:
                raise ValueError(f"Layer identifier '{layer_identifier}' not found.")
        parent_uuid = self._resolve_identifier(parent_identifier, Primitive) if parent_identifier else None
        if parent_identifier and not parent_uuid:
            raise ValueError(f"Parent identifier '{parent_identifier}' not found.")
        prim_id = identifier if identifier else str(uuid.uuid4())
        groups = groups if groups is not None else []
        cursor_tf = translation_matrix(self._cursor[0], self._cursor[1])
        initial_eff_tf = self.current_transform @ cursor_tf
        primitive = PrimitiveClass(user_identifier=prim_id, layer_id=layer.internal_id,
                                     groups=groups, description=description,
                                     effective_transform=initial_eff_tf,
                                     parent_primitive_id=parent_uuid,
                                     **kwargs)
        self._register_primitive(primitive)
        return primitive

    def add_pline(self, layer: Identifiable, points: List[Union[Tuple[float, float], Tuple[float, float, float]]],
                  closed: bool = False, identifier: Optional[str] = None, groups: Optional[List[str]] = None,
                  description: str = "", parent: Optional[Identifiable] = None) -> Pline:
        prim = self._add_primitive_internal(Pline, layer, identifier, groups, description, parent, relative_points=points, closed=closed)
        return prim

    def add_circle(self, layer: Identifiable, center: Tuple[float, float], diameter: float,
                   identifier: Optional[str] = None, groups: Optional[List[str]] = None, description: str = "",
                   parent: Optional[Identifiable] = None) -> Circle:
        prim = self._add_primitive_internal(Circle, layer, identifier, groups, description, parent, relative_center=center, diameter=diameter)
        return prim

    def add_rect(self, layer: Identifiable, corner: Tuple[float, float] = (0,0), width: float = 0, height: float = 0,
                 identifier: Optional[str] = None, groups: Optional[List[str]] = None, description: str = "",
                 parent: Optional[Identifiable] = None) -> Rect:
        prim = self._add_primitive_internal(Rect, layer, identifier, groups, description, parent, relative_corner=corner, width=width, height=height)
        return prim

    def add_arc(self, layer: Identifiable, center: Tuple[float, float], radius: float, start_angle: float, extent_angle: float,
                identifier: Optional[str] = None, groups: Optional[List[str]] = None, description: str = "",
                parent: Optional[Identifiable] = None) -> Arc:
        prim = self._add_primitive_internal(Arc, layer, identifier, groups, description, parent, relative_center=center, radius=radius, start_angle=start_angle, extent_angle=extent_angle)
        return prim

    def add_points(self, layer: Identifiable, points: List[Tuple[float, float]],
                   identifier: Optional[str] = None, groups: Optional[List[str]] = None, description: str = "",
                   parent: Optional[Identifiable] = None) -> Points:
        prim = self._add_primitive_internal(Points, layer, identifier, groups, description, parent, relative_points=points)
        return prim

    def add_text(self, layer: Identifiable, text: str, position: Tuple[float, float], height: float = 100.0,
                 font: str = 'Arial', style: str = '', line_spacing: float = 1.0, align_horizontal: str = 'center',
                 align_vertical: str = 'center', identifier: Optional[str] = None, groups: Optional[List[str]] = None,
                 description: str = "", parent: Optional[Identifiable] = None) -> Text:
        prim = self._add_primitive_internal(Text, layer, identifier, groups, description, parent,
                                            text_content=text, relative_position=position,
                                            height=height, font=font, style=style, line_spacing=line_spacing,
                                            align_horizontal=align_horizontal, align_vertical=align_vertical)
        return prim

    # --- MOP Creation Methods ---

    def _add_mop_internal(self, MopClass: Type[M], part_identifier: Identifiable,
                          pid_source: Union[str, List[Identifiable]], name: str,
                          identifier: Optional[str] = None,
                          target_mop_identifier: Optional[Identifiable] = None, place_last: bool = True,
                          **kwargs) -> M:
        part = self.get_part_by_identifier(part_identifier)
        if not part:
            raise ValueError(f"Part identifier '{part_identifier}' not found.")
        if isinstance(pid_source, str):
            resolved_pid_source = pid_source
        elif isinstance(pid_source, list):
            resolved_pid_list: List[uuid.UUID] = []
            for item in pid_source:
                uid = self._resolve_identifier(item, Primitive)
                if uid:
                    resolved_pid_list.append(uid)
                else:
                    logger.warning(f"Could not resolve primitive identifier '{item}' for MOP '{name}'.")
            resolved_pid_source = resolved_pid_list
        else:
            raise TypeError(f"Invalid pid_source type for MOP '{name}': {type(pid_source)}")
        mop_id = identifier if identifier else f"{name}_{uuid.uuid4().hex[:6]}"
        mop = MopClass(user_identifier=mop_id, part_id=part.internal_id,
                       pid_source=resolved_pid_source, name=name, **kwargs)
        self._register_entity(mop, self._mops)
        pos = self._find_insert_position(target_mop_identifier, part.mop_ids, place_last)
        part.mop_ids.insert(pos, mop.internal_id)
        return mop

    def add_profile_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'new profile',
                        identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> ProfileMop:
        mop = self._add_mop_internal(ProfileMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs)
        return mop

    def add_pocket_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'new pocket',
                       identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> PocketMop:
        mop = self._add_mop_internal(PocketMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs)
        return mop

    def add_engrave_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'new engrave',
                        identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> EngraveMop:
        mop = self._add_mop_internal(EngraveMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs)
        return mop

    def add_drill_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'new drill',
                      identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> DrillMop:
        mop = self._add_mop_internal(DrillMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs)
        return mop

    # --- Transformation Methods for Primitives ---

    def transform_primitive_locally(self, primitive_identifier: Identifiable, matrix: Any, bake: bool = False) -> bool:
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
        if bake:
            primitive.apply_bake(matrix)
        else:
            primitive._apply_transform_locally(matrix)
        return True

    def transform_primitive_globally(self, primitive_identifier: Identifiable, matrix: Any, bake: bool = False) -> bool:
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
        if bake:
            primitive.apply_bake(matrix)
        else:
            primitive._apply_transform_globally(matrix)
        return True

    def translate_primitive(self, primitive_identifier: Identifiable, dx: float, dy: float, bake: bool = False) -> bool:
        return self.transform_primitive_globally(primitive_identifier, translation_matrix(dx, dy), bake=bake)

    def rotate_primitive_deg(self, primitive_identifier: Identifiable, angle_deg: float, cx: Optional[float] = None,
                               cy: Optional[float] = None, bake: bool = False) -> bool:
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
        if cx is None or cy is None:
            try:
                center_x, center_y = primitive.get_geometric_center()
            except Exception as e:
                logger.error(f"Could not calculate geometric center for {primitive.user_identifier}: {e}")
                return False
        else:
            center_x, center_y = cx, cy
        rot_mat = rotation_matrix_deg(angle_deg, center_x, center_y)
        return self.transform_primitive_globally(primitive_identifier, rot_mat, bake=bake)

    # --- Additional Transformation Methods for Primitives ---
    
    def scale_primitive(self, primitive_identifier: Identifiable, sx: float, sy: Optional[float] = None, 
                         cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """
        Scale a primitive by the given factors.
        
        Args:
            primitive_identifier: Identifier of the primitive to scale
            sx: Scale factor in X direction
            sy: Scale factor in Y direction (if None, uses sx for uniform scaling)
            cx: X coordinate of scaling center (if None, uses primitive's geometric center)
            cy: Y coordinate of scaling center (if None, uses primitive's geometric center)
            bake: Whether to bake the transformation into the geometry
            
        Returns:
            Success status
        """
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
            
        if cx is None or cy is None:
            try:
                center_x, center_y = primitive.get_geometric_center()
            except Exception as e:
                logger.error(f"Could not calculate geometric center for {primitive.user_identifier}: {e}")
                return False
        else:
            center_x, center_y = cx, cy
            
        scale_mat = scale_matrix(sx, sy, center_x, center_y)
        return self.transform_primitive_globally(primitive_identifier, scale_mat, bake=bake)
    
    def mirror_primitive_x(self, primitive_identifier: Identifiable, cy: Optional[float] = None, 
                            bake: bool = False) -> bool:
        """
        Mirror a primitive across a horizontal line.
        
        Args:
            primitive_identifier: Identifier of the primitive to mirror
            cy: Y coordinate of the mirror line (if None, uses primitive's center Y)
            bake: Whether to bake the transformation into the geometry
            
        Returns:
            Success status
        """
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
            
        if cy is None:
            try:
                _, center_y = primitive.get_geometric_center()
                cy = center_y
            except Exception as e:
                logger.error(f"Could not calculate geometric center for {primitive.user_identifier}: {e}")
                return False
                
        mirror_mat = mirror_x_matrix(cy)
        return self.transform_primitive_globally(primitive_identifier, mirror_mat, bake=bake)
    
    def mirror_primitive_y(self, primitive_identifier: Identifiable, cx: Optional[float] = None, 
                            bake: bool = False) -> bool:
        """
        Mirror a primitive across a vertical line.
        
        Args:
            primitive_identifier: Identifier of the primitive to mirror
            cx: X coordinate of the mirror line (if None, uses primitive's center X)
            bake: Whether to bake the transformation into the geometry
            
        Returns:
            Success status
        """
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
            
        if cx is None:
            try:
                center_x, _ = primitive.get_geometric_center()
                cx = center_x
            except Exception as e:
                logger.error(f"Could not calculate geometric center for {primitive.user_identifier}: {e}")
                return False
                
        mirror_mat = mirror_y_matrix(cx)
        return self.transform_primitive_globally(primitive_identifier, mirror_mat, bake=bake)
    
    def skew_primitive(self, primitive_identifier: Identifiable, angle_x_deg: float = 0.0, 
                         angle_y_deg: float = 0.0, bake: bool = False) -> bool:
        """
        Apply skew (shear) transformation to a primitive.
        
        Args:
            primitive_identifier: Identifier of the primitive to skew
            angle_x_deg: Skew angle in X direction (degrees)
            angle_y_deg: Skew angle in Y direction (degrees)
            bake: Whether to bake the transformation into the geometry
            
        Returns:
            Success status
        """
        skew_mat = skew_matrix(angle_x_deg, angle_y_deg)
        return self.transform_primitive_globally(primitive_identifier, skew_mat, bake=bake)
    
    def rotate_primitive_rad(self, primitive_identifier: Identifiable, angle_rad: float, 
                               cx: Optional[float] = None, cy: Optional[float] = None, 
                               bake: bool = False) -> bool:
        """
        Rotate a primitive by the given angle in radians.
        
        Args:
            primitive_identifier: Identifier of the primitive to rotate
            angle_rad: Rotation angle in radians (positive is counterclockwise)
            cx: X coordinate of rotation center (if None, uses primitive's geometric center)
            cy: Y coordinate of rotation center (if None, uses primitive's geometric center)
            bake: Whether to bake the transformation into the geometry
            
        Returns:
            Success status
        """
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
            
        if cx is None or cy is None:
            try:
                center_x, center_y = primitive.get_geometric_center()
            except Exception as e:
                logger.error(f"Could not calculate geometric center for {primitive.user_identifier}: {e}")
                return False
        else:
            center_x, center_y = cx, cy
            
        rot_mat = rotation_matrix_rad(angle_rad, center_x, center_y)
        return self.transform_primitive_globally(primitive_identifier, rot_mat, bake=bake)
    
    def align_primitive(self, primitive_identifier: Identifiable, 
                         reference_identifier: Identifiable,
                         alignment: str = "center", 
                         offset_x: float = 0.0, offset_y: float = 0.0) -> bool:
        """
        Align a primitive to a reference primitive.
        
        Args:
            primitive_identifier: Identifier of the primitive to align
            reference_identifier: Identifier of the reference primitive
            alignment: Alignment type ("center", "top", "bottom", "left", "right",
                      "top-left", "top-right", "bottom-left", "bottom-right")
            offset_x: Additional X offset after alignment
            offset_y: Additional Y offset after alignment
            
        Returns:
            Success status
        """
        primitive = self._resolve_primitive(primitive_identifier)
        reference = self._resolve_primitive(reference_identifier)
        
        if not primitive or not reference:
            return False
            
        try:
            primitive_bbox = primitive.get_bounding_box()
            reference_bbox = reference.get_bounding_box()
            
            if not primitive_bbox.is_valid() or not reference_bbox.is_valid():
                logger.error("Cannot align primitives with invalid bounding boxes")
                return False
                
            tx, ty = 0.0, 0.0
            
            # Horizontal alignment
            if alignment in ["left", "top-left", "bottom-left"]:
                tx = reference_bbox.min_x - primitive_bbox.min_x
            elif alignment in ["right", "top-right", "bottom-right"]:
                tx = reference_bbox.max_x - primitive_bbox.max_x
            elif "center" in alignment or alignment in ["top", "bottom"]:
                tx = (reference_bbox.min_x + reference_bbox.max_x)/2 - (primitive_bbox.min_x + primitive_bbox.max_x)/2
                
            # Vertical alignment
            if alignment in ["top", "top-left", "top-right"]:
                ty = reference_bbox.max_y - primitive_bbox.max_y
            elif alignment in ["bottom", "bottom-left", "bottom-right"]:
                ty = reference_bbox.min_y - primitive_bbox.min_y
            elif "center" in alignment or alignment in ["left", "right"]:
                ty = (reference_bbox.min_y + reference_bbox.max_y)/2 - (primitive_bbox.min_y + primitive_bbox.max_y)/2
                
            # Apply additional offset
            tx += offset_x
            ty += offset_y
            
            # Translate the primitive
            return self.translate_primitive(primitive_identifier, tx, ty)
                
        except Exception as e:
            logger.error(f"Error aligning primitives: {e}")
            return False
    
    def distribute_primitives(self, primitive_identifiers: List[Identifiable], 
                               direction: str = "horizontal",
                               spacing: Optional[float] = None,
                               equal_spacing: bool = True) -> bool:
        """
        Distribute primitives evenly along a direction.
        
        Args:
            primitive_identifiers: List of primitives to distribute
            direction: "horizontal" or "vertical"
            spacing: Fixed spacing between primitives (if equal_spacing is False)
            equal_spacing: Whether to distribute with equal spacing
            
        Returns:
            Success status
        """
        if len(primitive_identifiers) < 2:
            return True  # Nothing to distribute
            
        primitives = []
        for pid in primitive_identifiers:
            prim = self._resolve_primitive(pid)
            if prim:
                primitives.append(prim)
                
        if len(primitives) < 2:
            return False
            
        try:
            # Sort primitives by position
            if direction == "horizontal":
                primitives.sort(key=lambda p: p.get_bounding_box().min_x)
            else:  # vertical
                primitives.sort(key=lambda p: p.get_bounding_box().min_y)
                
            first = primitives[0]
            last = primitives[-1]
            first_bbox = first.get_bounding_box()
            last_bbox = last.get_bounding_box()
            
            if equal_spacing:
                # Calculate total available space
                if direction == "horizontal":
                    total_space = last_bbox.min_x - first_bbox.max_x
                    for p in primitives[1:-1]:
                        total_space -= (p.get_bounding_box().max_x - p.get_bounding_box().min_x)
                else:  # vertical
                    total_space = last_bbox.min_y - first_bbox.max_y
                    for p in primitives[1:-1]:
                        total_space -= (p.get_bounding_box().max_y - p.get_bounding_box().min_y)
                
                # Calculate equal spacing
                equal_gap = total_space / (len(primitives) - 1)
                
                # Position each primitive
                current_pos = first_bbox.max_x if direction == "horizontal" else first_bbox.max_y
                for i, p in enumerate(primitives[1:-1], 1):
                    p_bbox = p.get_bounding_box()
                    if direction == "horizontal":
                        target_pos = current_pos + equal_gap
                        delta_x = target_pos - p_bbox.min_x
                        self.translate_primitive(p, delta_x, 0)
                        current_pos = p_bbox.min_x + delta_x + (p_bbox.max_x - p_bbox.min_x)
                    else:  # vertical
                        target_pos = current_pos + equal_gap
                        delta_y = target_pos - p_bbox.min_y
                        self.translate_primitive(p, 0, delta_y)
                        current_pos = p_bbox.min_y + delta_y + (p_bbox.max_y - p_bbox.min_y)
            else:
                # Use fixed spacing
                if spacing is None:
                    spacing = 10.0  # Default spacing
                    
                current_pos = first_bbox.max_x if direction == "horizontal" else first_bbox.max_y
                for p in primitives[1:]:
                    p_bbox = p.get_bounding_box()
                    if direction == "horizontal":
                        target_pos = current_pos + spacing
                        delta_x = target_pos - p_bbox.min_x
                        self.translate_primitive(p, delta_x, 0)
                        current_pos = p_bbox.min_x + delta_x + (p_bbox.max_x - p_bbox.min_x)
                    else:  # vertical
                        target_pos = current_pos + spacing
                        delta_y = target_pos - p_bbox.min_y
                        self.translate_primitive(p, 0, delta_y)
                        current_pos = p_bbox.min_y + delta_y + (p_bbox.max_y - p_bbox.min_y)
                        
            return True
            
        except Exception as e:
            logger.error(f"Error distributing primitives: {e}")
            return False

    # --- Copy/Transfer Methods ---

    def copy_primitive_recursive(self, primitive_identifier: Identifiable, new_name: Optional[str] = None) -> Optional[Primitive]:
        orig = self._resolve_primitive(primitive_identifier)
        if not orig:
            logger.error(f"Primitive '{primitive_identifier}' not found for copying.")
            return None
        copied = deepcopy(orig)
        def reassign_ids(prim: Primitive):
            prim.internal_id = uuid.uuid4()
            if new_name:
                prim.user_identifier = new_name
            else:
                prim.user_identifier = str(prim.internal_id)
            prim.parent_primitive_id = None
            for child_id in list(prim._child_primitive_ids):
                child = self.get_primitive(child_id)
                if child:
                    reassign_ids(child)
            prim._child_primitive_ids = set()
        reassign_ids(copied)
        self._register_primitive(copied)
        return copied

    def _resolve_primitive(self, identifier: Identifiable) -> Optional[Primitive]:
        uid = self._resolve_identifier(identifier, Primitive)
        if not uid:
            logger.error(f"Primitive identifier '{identifier}' not found.")
            return None
        prim = self.get_primitive(uid)
        if not prim:
            logger.error(f"Primitive with UUID {uid} not found.")
            return None
        return prim

    # --- Removal Methods ---

    def remove_primitive(self, primitive_identifier: Identifiable) -> bool:
        uid = self._resolve_identifier(primitive_identifier, Primitive)
        if not uid:
            logger.error(f"Primitive '{primitive_identifier}' not found for removal.")
            return False
        prim = self.get_primitive(uid)
        if not prim:
            return False
        self._unregister_primitive(uid)
        logger.info(f"Removed primitive '{prim.user_identifier}'.")
        return True

    def remove_mop(self, mop_identifier: Identifiable) -> bool:
        uid = self._resolve_identifier(mop_identifier, Mop)
        if not uid:
            logger.error(f"Cannot remove: MOP '{mop_identifier}' not found.")
            return False
        mop = self._mops.pop(uid, None)
        if mop:
            part = self.get_part(mop.part_id)
            if part and uid in part.mop_ids:
                part.mop_ids.remove(uid)
            logger.info(f"Removed MOP '{mop.user_identifier}'.")
            return True
        return False

    def remove_layer(self, layer_identifier: Identifiable, transfer_primitives_to: Optional[Identifiable] = None) -> bool:
        uid = self._resolve_identifier(layer_identifier, Layer)
        if not uid:
            logger.error(f"Layer '{layer_identifier}' not found for removal.")
            return False
        layer = self.get_layer(uid)
        if not layer:
            return False
        target_layer: Optional[Layer] = None
        if transfer_primitives_to:
            target_layer = self.get_layer_by_identifier(transfer_primitives_to)
            if not target_layer:
                logger.error(f"Target layer '{transfer_primitives_to}' not found.")
                return False
        prim_ids = list(layer.primitive_ids)
        if target_layer:
            for pid in prim_ids:
                prim = self.get_primitive(pid)
                if prim:
                    prim.layer_id = target_layer.internal_id
                    target_layer.primitive_ids.add(pid)
            layer.primitive_ids.clear()
        else:
            for pid in prim_ids:
                self.remove_primitive(pid)
        self._unregister_entity(uid, self._layers)
        if uid in self._layer_order:
            self._layer_order.remove(uid)
        logger.info(f"Removed layer '{layer.user_identifier}'.")
        return True

    def remove_part(self, part_identifier: Identifiable, remove_contained_mops: bool = True) -> bool:
        uid = self._resolve_identifier(part_identifier, Part)
        if not uid:
            logger.error(f"Part '{part_identifier}' not found for removal.")
            return False
        part = self.get_part(uid)
        if not part:
            return False
        mop_ids = list(part.mop_ids)
        if remove_contained_mops:
            for mid in mop_ids:
                self.remove_mop(mid)
        elif mop_ids:
            logger.error(f"Cannot remove part '{part.user_identifier}' because it contains MOPs.")
            return False
        self._unregister_entity(uid, self._parts)
        if uid in self._part_order:
            self._part_order.remove(uid)
        logger.info(f"Removed part '{part.user_identifier}'.")
        return True

    # --- Bounding Box ---

    def get_bounding_box(self) -> BoundingBox:
        overall = BoundingBox()
        for prim in self._primitives.values():
            try:
                bb = prim.get_bounding_box()
                if bb.is_valid():
                    overall = overall.union(bb)
            except Exception as e:
                logger.error(f"Error computing bounding box for {prim.user_identifier}: {e}")
        return overall

    # --- Persistence ---

    def save_state(self, file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            logger.info(f"Project state saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            raise

    @staticmethod
    def load_state(file_path: str) -> "CamBamProject":
        try:
            with open(file_path, 'rb') as f:
                project = pickle.load(f)
            for prim in project._primitives.values():
                prim._project_ref = weakref.ref(project)
            logger.info(f"Loaded project state from {file_path}")
            return project
        except Exception as e:
            logger.error(f"Error loading project state: {e}")
            raise
