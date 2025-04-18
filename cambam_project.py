"""
cambam_project.py

Defines the CamBamProject class, the central manager for a CamBam project.
It holds registries for all entities (Layers, Parts, Primitives, MOPs) and,
crucially, manages the relationships between them (layer assignment, parent/child links,
part-MOP links) in dedicated internal registries.

Provides methods for adding, removing, querying, and manipulating entities and their
relationships, including transformation propagation and serialization support.
"""

import os
import pickle
import uuid
import weakref
import logging
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Optional, Union, Type, TypeVar, Any, Sequence

import numpy as np

from cad_transformations import (
    identity_matrix, translation_matrix, rotation_matrix_deg, scale_matrix,
    mirror_x_matrix, mirror_y_matrix, skew_matrix, rotation_matrix_rad,
    apply_transform, get_transformed_point, extract_transform_component, remove_transform_component
)
# Import entity types
from cambam_entities import (
    CamBamEntity, Layer, Part, Primitive, Mop, MopType, BoundingBox,
    Pline, Circle, Rect, Arc, Points, Text,
    ProfileMop, PocketMop, EngraveMop, DrillMop
)

logger = logging.getLogger(__name__)

# --- Type Hinting ---
# Generic type for CamBamEntity subclasses
EntityType = TypeVar('EntityType', bound=CamBamEntity)
# Type for identifiers (UUID, user identifier string, or entity object itself)
Identifiable = Union[str, uuid.UUID, CamBamEntity]


class CamBamProject:
    """
    Manages all entities and their relationships within a CamBam project.
    Acts as the central registry and source of truth for the project structure.
    """
    def __init__(self, project_name: str, default_tool_diameter: float = 6.0):
        self.project_name: str = project_name
        self.default_tool_diameter: float = default_tool_diameter
        # Add other project-level defaults as needed (e.g., default_spindle_speed)

        # --- Entity Registries ---
        # Store entities keyed by their internal UUID
        self._primitives: Dict[uuid.UUID, Primitive] = {}
        self._layers: Dict[uuid.UUID, Layer] = {}
        self._parts: Dict[uuid.UUID, Part] = {}
        self._mops: Dict[uuid.UUID, Mop] = {}

        # --- Ordering (for consistent output/UI) ---
        self._layer_order: List[uuid.UUID] = []
        self._part_order: List[uuid.UUID] = []
        self._mop_order_in_part: Dict[uuid.UUID, List[uuid.UUID]] = {} # Part UUID -> List of MOP UUIDs

        # --- Relationship Registries (Centralized Management) ---
        # Primitive <-> Layer
        self._primitive_layer_assignment: Dict[uuid.UUID, uuid.UUID] = {} # Primitive UUID -> Layer UUID
        self._layer_primitive_membership: Dict[uuid.UUID, Set[uuid.UUID]] = {} # Layer UUID -> Set of Primitive UUIDs

        # Primitive <-> Parent/Child
        self._primitive_parent_link: Dict[uuid.UUID, uuid.UUID] = {} # Child Primitive UUID -> Parent Primitive UUID
        self._primitive_children_link: Dict[uuid.UUID, Set[uuid.UUID]] = {} # Parent Primitive UUID -> Set of Child Primitive UUIDs

        # MOP <-> Part
        self._mop_part_assignment: Dict[uuid.UUID, uuid.UUID] = {} # MOP UUID -> Part UUID
        # Note: MOP order within a part is handled by _mop_order_in_part

        # Primitive <-> Group (Classification / MOP Targeting)
        self._primitive_groups: Dict[str, Set[uuid.UUID]] = {} # Group Name -> Set of Primitive UUIDs
        self._primitive_group_membership: Dict[uuid.UUID, Set[str]] = {} # Primitive UUID -> Set of Group Names

        # --- Lookup and State ---
        # Map user identifiers (must be unique) back to internal UUIDs
        self._identifier_registry: Dict[str, uuid.UUID] = {}

        # Transformation context stack (primarily for interactive/programmatic building)
        self._transform_stack: List[np.ndarray] = [identity_matrix()]
        # Cursor position (primarily for interactive/programmatic building)
        self._cursor: Tuple[float, float] = (0.0, 0.0)

        logger.info(f"Initialized CamBamProject: {self.project_name}")

    # --- Internal Helper: Identifier Resolution ---
    def _resolve_identifier(self, identifier: Optional[Identifiable],
                             expected_type: Optional[Type[EntityType]] = None) -> Optional[uuid.UUID]:
        """Resolves a string, UUID, or entity object to its internal UUID."""
        if identifier is None:
            return None

        target_uuid: Optional[uuid.UUID] = None
        if isinstance(identifier, uuid.UUID):
            target_uuid = identifier
        elif isinstance(identifier, str):
            target_uuid = self._identifier_registry.get(identifier)
            if target_uuid is None:
                logger.debug(f"Identifier string '{identifier}' not found in registry.")
                return None
        elif isinstance(identifier, CamBamEntity):
            target_uuid = identifier.internal_id
        else:
            logger.error(f"Invalid identifier type: {type(identifier)}")
            return None

        # Validate existence and optionally type
        if target_uuid:
            entity = self._get_entity_by_uuid(target_uuid)
            if entity is None:
                 logger.debug(f"Identifier '{identifier}' resolved to UUID {target_uuid}, but entity not found in registries.")
                 return None
            if expected_type and not isinstance(entity, expected_type):
                logger.debug(f"Identifier '{identifier}' resolved to entity of type {type(entity)}, but expected {expected_type}.")
                return None
        return target_uuid

    def _get_entity_by_uuid(self, entity_uuid: uuid.UUID) -> Optional[CamBamEntity]:
        """Retrieves an entity directly by its UUID from any registry."""
        return (self._primitives.get(entity_uuid) or
                self._layers.get(entity_uuid) or
                self._parts.get(entity_uuid) or
                self._mops.get(entity_uuid))

    # --- Internal Helper: Entity Registration ---
    def _register_entity(self, entity: EntityType, registry: Dict[uuid.UUID, EntityType]) -> bool:
        """Adds an entity to its specific registry and the identifier lookup."""
        if not isinstance(entity, CamBamEntity):
             logger.error(f"Attempted to register non-CamBamEntity object: {entity}")
             return False
        if entity.internal_id in self._primitives or \
           entity.internal_id in self._layers or \
           entity.internal_id in self._parts or \
           entity.internal_id in self._mops:
             logger.warning(f"Entity with UUID {entity.internal_id} ('{entity.user_identifier}') already registered.")
             # Should we update or reject? Let's update for now.
             # return False # If rejecting duplicates

        # Check user identifier uniqueness
        if entity.user_identifier:
            existing_uuid = self._identifier_registry.get(entity.user_identifier)
            if existing_uuid and existing_uuid != entity.internal_id:
                logger.error(f"User identifier '{entity.user_identifier}' is already used by entity {existing_uuid}. Cannot register {entity.internal_id}.")
                # Optionally, auto-rename:
                # entity.user_identifier = f"{entity.user_identifier}_{entity.internal_id.hex[:4]}"
                # logger.warning(f"Renamed conflicting identifier to '{entity.user_identifier}'")
                return False # Reject conflicting user identifiers

            self._identifier_registry[entity.user_identifier] = entity.internal_id

        # Add to the specific registry
        registry[entity.internal_id] = entity

        # Link primitive back to project
        if isinstance(entity, Primitive):
            entity.set_project_link(self)

        logger.debug(f"Registered {type(entity).__name__} '{entity.user_identifier}' ({entity.internal_id})")
        return True

    def _unregister_entity(self, entity_uuid: uuid.UUID) -> Optional[CamBamEntity]:
        """Removes an entity from all registries and lookups."""
        entity = self._get_entity_by_uuid(entity_uuid)
        if not entity:
            logger.warning(f"Attempted to unregister non-existent entity UUID {entity_uuid}")
            return None

        # Remove from specific registry
        if isinstance(entity, Primitive):
            self._primitives.pop(entity_uuid, None)
            # Clean up all relationships involving this primitive
            self._cleanup_primitive_relationships(entity_uuid)
        elif isinstance(entity, Layer):
            self._layers.pop(entity_uuid, None)
            self._cleanup_layer_relationships(entity_uuid)
            if entity_uuid in self._layer_order: self._layer_order.remove(entity_uuid)
        elif isinstance(entity, Part):
            self._parts.pop(entity_uuid, None)
            self._cleanup_part_relationships(entity_uuid)
            if entity_uuid in self._part_order: self._part_order.remove(entity_uuid)
        elif isinstance(entity, Mop):
            self._mops.pop(entity_uuid, None)
            self._cleanup_mop_relationships(entity_uuid)

        # Remove from identifier registry
        if entity.user_identifier and self._identifier_registry.get(entity.user_identifier) == entity_uuid:
            self._identifier_registry.pop(entity.user_identifier, None)

        logger.debug(f"Unregistered {type(entity).__name__} '{entity.user_identifier}' ({entity_uuid})")
        return entity

    # --- Internal Helper: Relationship Cleanup ---
    def _cleanup_primitive_relationships(self, primitive_uuid: uuid.UUID):
        """Removes a primitive from all relationship registries."""
        # Layer assignment
        layer_id = self._primitive_layer_assignment.pop(primitive_uuid, None)
        if layer_id and layer_id in self._layer_primitive_membership:
            self._layer_primitive_membership[layer_id].discard(primitive_uuid)

        # Parent/Child links
        parent_id = self._primitive_parent_link.pop(primitive_uuid, None)
        if parent_id and parent_id in self._primitive_children_link:
            self._primitive_children_link[parent_id].discard(primitive_uuid)
        child_ids = self._primitive_children_link.pop(primitive_uuid, set())
        for child_id in child_ids:
            self._primitive_parent_link.pop(child_id, None) # Remove link from child back to this parent

        # Group membership
        group_names = self._primitive_group_membership.pop(primitive_uuid, set())
        for group_name in group_names:
            if group_name in self._primitive_groups:
                self._primitive_groups[group_name].discard(primitive_uuid)
                if not self._primitive_groups[group_name]: # Remove empty group
                     del self._primitive_groups[group_name]

        # MOP References (MOPs store pid_source, no direct registry here, handled by resolution)
        # If MOPs had a resolved list, we'd clean it here.

    def _cleanup_layer_relationships(self, layer_uuid: uuid.UUID):
        """Handles layer removal: unassigns its primitives."""
        primitive_ids_on_layer = list(self._layer_primitive_membership.pop(layer_uuid, set()))
        if primitive_ids_on_layer:
             logger.warning(f"Layer {layer_uuid} removed. Contained primitives {primitive_ids_on_layer} are now unassigned. Consider removing or reassigning them.")
             for prim_id in primitive_ids_on_layer:
                 self._primitive_layer_assignment.pop(prim_id, None)
                 # Note: Primitives themselves are NOT deleted here.

    def _cleanup_part_relationships(self, part_uuid: uuid.UUID):
        """Handles part removal: unassigns its MOPs."""
        mop_ids_in_part = list(self._mop_order_in_part.pop(part_uuid, []))
        if mop_ids_in_part:
            logger.warning(f"Part {part_uuid} removed. Contained MOPs {mop_ids_in_part} are now unassigned. Consider removing or reassigning them.")
            for mop_id in mop_ids_in_part:
                 self._mop_part_assignment.pop(mop_id, None)
                 # Note: MOPs themselves are NOT deleted here.

    def _cleanup_mop_relationships(self, mop_uuid: uuid.UUID):
        """Handles MOP removal: remove from part assignment and order."""
        part_id = self._mop_part_assignment.pop(mop_uuid, None)
        if part_id and part_id in self._mop_order_in_part:
            if mop_uuid in self._mop_order_in_part[part_id]:
                self._mop_order_in_part[part_id].remove(mop_uuid)

    # --- Internal Helper: Ordering ---
    def _find_insert_position(self, target_identifier: Optional[Identifiable],
                              order_list: List[uuid.UUID],
                              place_last: bool) -> int:
        """Finds the index where a new entity should be inserted in an order list."""
        if target_identifier is None:
            return len(order_list) if place_last else 0

        target_uuid = self._resolve_identifier(target_identifier)
        if target_uuid is None or target_uuid not in order_list:
            logger.warning(f"Target identifier '{target_identifier}' not found for ordering. Appending/Prepending.")
            return len(order_list) if place_last else 0

        try:
            index = order_list.index(target_uuid)
            return index + 1 if place_last else index
        except ValueError: # Should not happen if target_uuid is in order_list
             logger.error(f"UUID {target_uuid} not found in order list {order_list} despite check.")
             return len(order_list)


    # --- Public API: Getters ---

    def get_entity(self, identifier: Identifiable) -> Optional[CamBamEntity]:
        """Gets any entity by its identifier (UUID, user ID string, or object)."""
        uuid = self._resolve_identifier(identifier)
        return self._get_entity_by_uuid(uuid) if uuid else None

    def get_primitive(self, identifier: Identifiable) -> Optional[Primitive]:
        uuid = self._resolve_identifier(identifier, Primitive)
        return self._primitives.get(uuid) if uuid else None

    def get_layer(self, identifier: Identifiable) -> Optional[Layer]:
        uuid = self._resolve_identifier(identifier, Layer)
        return self._layers.get(uuid) if uuid else None

    def get_part(self, identifier: Identifiable) -> Optional[Part]:
        uuid = self._resolve_identifier(identifier, Part)
        return self._parts.get(uuid) if uuid else None

    def get_mop(self, identifier: Identifiable) -> Optional[Mop]:
        uuid = self._resolve_identifier(identifier, Mop)
        return self._mops.get(uuid) if uuid else None

    def list_primitives(self) -> List[Primitive]:
        """Returns a list of all primitives in the project."""
        # Return in a somewhat consistent order (e.g., sorted by UUID)
        return [self._primitives[uid] for uid in sorted(self._primitives.keys())]

    def list_layers(self) -> List[Layer]:
        """Returns a list of layers in their defined order."""
        return [self._layers[uid] for uid in self._layer_order if uid in self._layers]

    def list_parts(self) -> List[Part]:
        """Returns a list of parts in their defined order."""
        return [self._parts[uid] for uid in self._part_order if uid in self._parts]

    def list_mops(self, part_identifier: Optional[Identifiable] = None) -> List[Mop]:
        """Returns a list of MOPs, optionally filtered by part."""
        if part_identifier:
            part_uuid = self._resolve_identifier(part_identifier, Part)
            if not part_uuid:
                logger.warning(f"Part '{part_identifier}' not found for listing MOPs.")
                return []
            mop_order = self._mop_order_in_part.get(part_uuid, [])
            return [self._mops[mid] for mid in mop_order if mid in self._mops]
        else:
            # Return all MOPs, perhaps ordered by part then by order within part
            all_mops = []
            for part_uuid in self._part_order:
                 mop_order = self._mop_order_in_part.get(part_uuid, [])
                 all_mops.extend(self._mops[mid] for mid in mop_order if mid in self._mops)
            # Add any MOPs not assigned to a part (shouldn't happen with proper management)
            assigned_mop_ids = set(m for order in self._mop_order_in_part.values() for m in order)
            unassigned_mops = [m for mid, m in self._mops.items() if mid not in assigned_mop_ids]
            if unassigned_mops:
                 logger.warning(f"Found {len(unassigned_mops)} unassigned MOPs.")
                 all_mops.extend(unassigned_mops)
            return all_mops

    # --- Public API: Relationship Queries ---

    def get_layer_of_primitive(self, primitive_identifier: Identifiable) -> Optional[Layer]:
        """Finds the layer a primitive belongs to."""
        prim_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        if not prim_uuid: return None
        layer_uuid = self._primitive_layer_assignment.get(prim_uuid)
        return self._layers.get(layer_uuid) if layer_uuid else None

    def get_primitives_on_layer(self, layer_identifier: Identifiable) -> List[Primitive]:
        """Gets all primitives assigned to a specific layer."""
        layer_uuid = self._resolve_identifier(layer_identifier, Layer)
        if not layer_uuid: return []
        primitive_ids = self._layer_primitive_membership.get(layer_uuid, set())
        # Return in consistent order
        return [self._primitives[pid] for pid in sorted(primitive_ids) if pid in self._primitives]

    def get_parent_of_primitive(self, primitive_identifier: Identifiable) -> Optional[Primitive]:
        """Finds the parent of a primitive."""
        prim_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        if not prim_uuid: return None
        parent_uuid = self._primitive_parent_link.get(prim_uuid)
        return self._primitives.get(parent_uuid) if parent_uuid else None

    def get_children_of_primitive(self, primitive_identifier: Identifiable) -> List[Primitive]:
        """Gets the direct children of a primitive."""
        parent_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        if not parent_uuid: return []
        child_ids = self._primitive_children_link.get(parent_uuid, set())
        # Return in consistent order
        return [self._primitives[cid] for cid in sorted(child_ids) if cid in self._primitives]

    def get_part_of_mop(self, mop_identifier: Identifiable) -> Optional[Part]:
        """Finds the part a MOP belongs to."""
        mop_uuid = self._resolve_identifier(mop_identifier, Mop)
        if not mop_uuid: return None
        part_uuid = self._mop_part_assignment.get(mop_uuid)
        return self._parts.get(part_uuid) if part_uuid else None

    def get_mops_in_part(self, part_identifier: Identifiable) -> List[Mop]:
        """Gets the MOPs assigned to a specific part, in order."""
        part_uuid = self._resolve_identifier(part_identifier, Part)
        if not part_uuid: return []
        mop_ids = self._mop_order_in_part.get(part_uuid, [])
        return [self._mops[mid] for mid in mop_ids if mid in self._mops]

    def get_groups_of_primitive(self, primitive_identifier: Identifiable) -> List[str]:
        """Gets the list of groups a primitive belongs to."""
        prim_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        if not prim_uuid: return []
        return sorted(list(self._primitive_group_membership.get(prim_uuid, set())))

    def get_primitives_in_group(self, group_name: str) -> List[Primitive]:
        """Gets all primitives belonging to a specific group."""
        primitive_ids = self._primitive_groups.get(group_name, set())
        # Return in consistent order
        return [self._primitives[pid] for pid in sorted(primitive_ids) if pid in self._primitives]

    def list_groups(self) -> List[str]:
        """Returns a sorted list of all defined group names."""
        return sorted(self._primitive_groups.keys())

    # --- Public API: Relationship Management ---

    def assign_primitive_to_layer(self, primitive_identifier: Identifiable, layer_identifier: Identifiable) -> bool:
        """Assigns a primitive to a layer, updating registries."""
        prim_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        layer_uuid = self._resolve_identifier(layer_identifier, Layer)

        if not prim_uuid or not layer_uuid:
            logger.error(f"Cannot assign primitive '{primitive_identifier}' to layer '{layer_identifier}': One or both not found.")
            return False

        # Remove from old layer membership if already assigned
        old_layer_uuid = self._primitive_layer_assignment.get(prim_uuid)
        if old_layer_uuid and old_layer_uuid != layer_uuid:
            if old_layer_uuid in self._layer_primitive_membership:
                self._layer_primitive_membership[old_layer_uuid].discard(prim_uuid)

        # Update assignment
        self._primitive_layer_assignment[prim_uuid] = layer_uuid
        self._layer_primitive_membership.setdefault(layer_uuid, set()).add(prim_uuid)
        logger.debug(f"Assigned primitive {prim_uuid} to layer {layer_uuid}")
        return True

    def link_primitive_parent(self, child_identifier: Identifiable, parent_identifier: Optional[Identifiable]) -> bool:
        """Links a child primitive to a parent primitive, updating registries."""
        child_uuid = self._resolve_identifier(child_identifier, Primitive)
        parent_uuid = self._resolve_identifier(parent_identifier, Primitive) if parent_identifier else None

        if not child_uuid:
            logger.error(f"Cannot link child '{child_identifier}': Not found.")
            return False
        if parent_identifier and not parent_uuid:
             logger.error(f"Cannot link child '{child_identifier}' to parent '{parent_identifier}': Parent not found.")
             return False
        if child_uuid == parent_uuid:
            logger.error(f"Cannot link primitive {child_uuid} to itself.")
            return False
        # TODO: Add check for circular dependencies?

        # Remove old parent link if exists
        old_parent_uuid = self._primitive_parent_link.pop(child_uuid, None)
        if old_parent_uuid:
            if old_parent_uuid in self._primitive_children_link:
                self._primitive_children_link[old_parent_uuid].discard(child_uuid)

        # Add new link
        if parent_uuid:
            self._primitive_parent_link[child_uuid] = parent_uuid
            self._primitive_children_link.setdefault(parent_uuid, set()).add(child_uuid)
            logger.debug(f"Linked primitive {child_uuid} to parent {parent_uuid}")
        else:
             logger.debug(f"Unlinked primitive {child_uuid} from parent.")

        return True

    def assign_mop_to_part(self, mop_identifier: Identifiable, part_identifier: Identifiable,
                           target_mop_identifier: Optional[Identifiable] = None, place_last: bool = True) -> bool:
        """Assigns a MOP to a part and sets its order, updating registries."""
        mop_uuid = self._resolve_identifier(mop_identifier, Mop)
        part_uuid = self._resolve_identifier(part_identifier, Part)

        if not mop_uuid or not part_uuid:
            logger.error(f"Cannot assign MOP '{mop_identifier}' to Part '{part_identifier}': One or both not found.")
            return False

        # Remove from old part assignment and order if exists
        old_part_uuid = self._mop_part_assignment.pop(mop_uuid, None)
        if old_part_uuid and old_part_uuid != part_uuid:
            if old_part_uuid in self._mop_order_in_part:
                 if mop_uuid in self._mop_order_in_part[old_part_uuid]:
                     self._mop_order_in_part[old_part_uuid].remove(mop_uuid)

        # Update assignment
        self._mop_part_assignment[mop_uuid] = part_uuid

        # Update order within the new part
        order_list = self._mop_order_in_part.setdefault(part_uuid, [])
        # Remove if already in the list (e.g., changing order within same part)
        if mop_uuid in order_list: order_list.remove(mop_uuid)
        # Find insert position relative to target MOP
        insert_pos = self._find_insert_position(target_mop_identifier, order_list, place_last)
        order_list.insert(insert_pos, mop_uuid)

        logger.debug(f"Assigned MOP {mop_uuid} to part {part_uuid} at position {insert_pos}")
        return True

    def set_primitive_groups(self, primitive_identifier: Identifiable, group_names: List[str]) -> bool:
        """Sets the groups for a primitive, replacing any existing groups."""
        prim_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        primitive = self.get_primitive(prim_uuid) if prim_uuid else None
        if not primitive:
            logger.error(f"Cannot set groups for primitive '{primitive_identifier}': Not found.")
            return False

        # Remove from old groups
        old_group_names = self._primitive_group_membership.pop(prim_uuid, set())
        for old_group in old_group_names:
            if old_group in self._primitive_groups:
                self._primitive_groups[old_group].discard(prim_uuid)
                if not self._primitive_groups[old_group]:
                    del self._primitive_groups[old_group]

        # Add to new groups
        valid_group_names = set(g for g in group_names if isinstance(g, str) and g) # Ensure non-empty strings
        for group_name in valid_group_names:
            self._primitive_groups.setdefault(group_name, set()).add(prim_uuid)
        self._primitive_group_membership[prim_uuid] = valid_group_names
        # Update the primitive object itself as well (spec says primitive stores this)
        primitive.groups = sorted(list(valid_group_names))

        logger.debug(f"Set groups for primitive {prim_uuid} to {primitive.groups}")
        return True

    def add_primitive_to_group(self, primitive_identifier: Identifiable, group_name: str) -> bool:
        """Adds a primitive to a specific group."""
        prim_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        primitive = self.get_primitive(prim_uuid) if prim_uuid else None
        if not primitive or not isinstance(group_name, str) or not group_name:
            logger.error(f"Cannot add primitive '{primitive_identifier}' to group '{group_name}': Invalid input.")
            return False

        self._primitive_groups.setdefault(group_name, set()).add(prim_uuid)
        self._primitive_group_membership.setdefault(prim_uuid, set()).add(group_name)
        primitive.groups = sorted(list(self._primitive_group_membership[prim_uuid]))
        logger.debug(f"Added primitive {prim_uuid} to group '{group_name}'")
        return True

    def remove_primitive_from_group(self, primitive_identifier: Identifiable, group_name: str) -> bool:
        """Removes a primitive from a specific group."""
        prim_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        primitive = self.get_primitive(prim_uuid) if prim_uuid else None
        if not primitive or not isinstance(group_name, str) or not group_name:
            logger.error(f"Cannot remove primitive '{primitive_identifier}' from group '{group_name}': Invalid input.")
            return False

        if group_name in self._primitive_groups:
            self._primitive_groups[group_name].discard(prim_uuid)
            if not self._primitive_groups[group_name]:
                del self._primitive_groups[group_name]
        if prim_uuid in self._primitive_group_membership:
            self._primitive_group_membership[prim_uuid].discard(group_name)
        primitive.groups = sorted(list(self._primitive_group_membership.get(prim_uuid, set())))
        logger.debug(f"Removed primitive {prim_uuid} from group '{group_name}'")
        return True


    # --- Public API: Entity Creation ---

    def add_layer(self, identifier: str, color: str = 'Green', alpha: float = 1.0,
                  pen_width: float = 1.0, visible: bool = True, locked: bool = False,
                  target_identifier: Optional[Identifiable] = None, place_last: bool = True) -> Optional[Layer]:
        """Creates or updates a layer and adds it to the project."""
        existing_uuid = self._resolve_identifier(identifier, Layer)
        if existing_uuid:
            layer = self._layers[existing_uuid]
            logger.info(f"Updating existing layer '{identifier}' ({existing_uuid})")
            layer.color = color
            layer.alpha = alpha
            layer.pen_width = pen_width
            layer.visible = visible
            layer.locked = locked
            # Ensure it's in the order list if it somehow got removed
            if layer.internal_id not in self._layer_order:
                 pos = self._find_insert_position(target_identifier, self._layer_order, place_last)
                 self._layer_order.insert(pos, layer.internal_id)
            return layer
        else:
            # Create new layer
            new_layer = Layer(user_identifier=identifier, color=color, alpha=alpha,
                              pen_width=pen_width, visible=visible, locked=locked)
            if not self._register_entity(new_layer, self._layers):
                return None # Registration failed (e.g., identifier conflict)
            # Add to order list
            pos = self._find_insert_position(target_identifier, self._layer_order, place_last)
            self._layer_order.insert(pos, new_layer.internal_id)
            # Initialize membership registry
            self._layer_primitive_membership[new_layer.internal_id] = set()
            logger.info(f"Added new layer '{identifier}' ({new_layer.internal_id})")
            return new_layer

    def add_part(self, identifier: str, enabled: bool = True, stock_thickness: float = 12.5,
                 stock_width: float = 1220.0, stock_height: float = 2440.0,
                 stock_material: str = "MDF", stock_color: str = "210,180,140",
                 machining_origin: Tuple[float, float] = (0.0, 0.0),
                 default_tool_diameter: Optional[float] = None,
                 default_spindle_speed: Optional[int] = None,
                 target_identifier: Optional[Identifiable] = None, place_last: bool = True) -> Optional[Part]:
        """Creates or updates a part and adds it to the project."""
        existing_uuid = self._resolve_identifier(identifier, Part)
        if existing_uuid:
            part = self._parts[existing_uuid]
            logger.info(f"Updating existing part '{identifier}' ({existing_uuid})")
            part.enabled = enabled
            part.stock_thickness = stock_thickness
            part.stock_width = stock_width
            part.stock_height = stock_height
            part.stock_material = stock_material
            part.stock_color = stock_color
            part.machining_origin = machining_origin
            part.default_tool_diameter = default_tool_diameter
            part.default_spindle_speed = default_spindle_speed
            if part.internal_id not in self._part_order:
                 pos = self._find_insert_position(target_identifier, self._part_order, place_last)
                 self._part_order.insert(pos, part.internal_id)
            return part
        else:
            new_part = Part(user_identifier=identifier, enabled=enabled, stock_thickness=stock_thickness,
                            stock_width=stock_width, stock_height=stock_height, stock_material=stock_material,
                            stock_color=stock_color, machining_origin=machining_origin,
                            default_tool_diameter=default_tool_diameter, default_spindle_speed=default_spindle_speed)
            if not self._register_entity(new_part, self._parts):
                return None
            pos = self._find_insert_position(target_identifier, self._part_order, place_last)
            self._part_order.insert(pos, new_part.internal_id)
            # Initialize MOP order registry
            self._mop_order_in_part[new_part.internal_id] = []
            logger.info(f"Added new part '{identifier}' ({new_part.internal_id})")
            return new_part

    def _add_primitive_internal(self, PrimitiveClass: Type[Primitive],
                                layer_identifier: Identifiable,
                                identifier: Optional[str] = None,
                                groups: Optional[List[str]] = None,
                                description: str = "",
                                parent_identifier: Optional[Identifiable] = None,
                                **kwargs) -> Optional[Primitive]:
        """Internal helper to create, register, and link a primitive."""
        # 1. Resolve Layer
        layer = self.get_layer(layer_identifier)
        if not layer:
            # Option: Auto-create layer if identifier is a string?
            if isinstance(layer_identifier, str):
                logger.info(f"Layer '{layer_identifier}' not found, creating it.")
                layer = self.add_layer(layer_identifier)
                if not layer: return None # Failed to create layer
            else:
                logger.error(f"Layer identifier '{layer_identifier}' not found or invalid.")
                return None

        # 2. Resolve Parent (optional)
        parent_uuid = self._resolve_identifier(parent_identifier, Primitive) if parent_identifier else None
        if parent_identifier and not parent_uuid:
            logger.error(f"Parent identifier '{parent_identifier}' not found or invalid.")
            return None

        # 3. Create Primitive instance
        prim_id_str = identifier if identifier else f"{PrimitiveClass.__name__}_{uuid.uuid4().hex[:6]}"
        groups_list = groups if groups is not None else []

        # Apply current context transform (stack + cursor) to initial effective transform
        cursor_tf = translation_matrix(self._cursor[0], self._cursor[1])
        initial_eff_tf = self.current_transform @ cursor_tf

        try:
            primitive = PrimitiveClass(user_identifier=prim_id_str,
                                       groups=groups_list, # Store groups on primitive
                                       description=description,
                                       effective_transform=initial_eff_tf,
                                       **kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate {PrimitiveClass.__name__} with identifier '{prim_id_str}': {e}")
            return None

        # 4. Register Entity
        if not self._register_entity(primitive, self._primitives):
            # Registration failed (e.g., identifier conflict)
            return None

        # 5. Establish Relationships using project methods
        self.assign_primitive_to_layer(primitive.internal_id, layer.internal_id)
        if parent_uuid:
            self.link_primitive_parent(primitive.internal_id, parent_uuid)
        # Set groups via relationship manager (also updates primitive.groups)
        self.set_primitive_groups(primitive.internal_id, groups_list)

        logger.info(f"Added {PrimitiveClass.__name__} '{prim_id_str}' ({primitive.internal_id}) to layer '{layer.user_identifier}'")
        return primitive

    # --- Public API: Concrete Primitive Adders ---
    # These now just call _add_primitive_internal

    def add_pline(self, layer: Identifiable, points: List[Union[Tuple[float, float], Tuple[float, float, float]]],
                  closed: bool = False, identifier: Optional[str] = None, groups: Optional[List[str]] = None,
                  description: str = "", parent: Optional[Identifiable] = None) -> Optional[Pline]:
        return self._add_primitive_internal(Pline, layer, identifier, groups, description, parent, relative_points=points, closed=closed) # type: ignore

    def add_circle(self, layer: Identifiable, center: Tuple[float, float], diameter: float,
                   identifier: Optional[str] = None, groups: Optional[List[str]] = None, description: str = "",
                   parent: Optional[Identifiable] = None) -> Optional[Circle]:
        return self._add_primitive_internal(Circle, layer, identifier, groups, description, parent, relative_center=center, diameter=diameter) # type: ignore

    def add_rect(self, layer: Identifiable, corner: Tuple[float, float] = (0,0), width: float = 1.0, height: float = 1.0,
                 identifier: Optional[str] = None, groups: Optional[List[str]] = None, description: str = "",
                 parent: Optional[Identifiable] = None) -> Optional[Rect]:
        return self._add_primitive_internal(Rect, layer, identifier, groups, description, parent, relative_corner=corner, width=width, height=height) # type: ignore

    def add_arc(self, layer: Identifiable, center: Tuple[float, float], radius: float, start_angle: float, extent_angle: float,
                identifier: Optional[str] = None, groups: Optional[List[str]] = None, description: str = "",
                parent: Optional[Identifiable] = None) -> Optional[Arc]:
        return self._add_primitive_internal(Arc, layer, identifier, groups, description, parent, relative_center=center, radius=radius, start_angle=start_angle, extent_angle=extent_angle) # type: ignore

    def add_points(self, layer: Identifiable, points: List[Tuple[float, float]],
                   identifier: Optional[str] = None, groups: Optional[List[str]] = None, description: str = "",
                   parent: Optional[Identifiable] = None) -> Optional[Points]:
        return self._add_primitive_internal(Points, layer, identifier, groups, description, parent, relative_points=points) # type: ignore

    def add_text(self, layer: Identifiable, text: str, position: Tuple[float, float], height: float = 10.0,
                 font: str = 'Arial', style: str = '', line_spacing: float = 1.0, align_horizontal: str = 'center',
                 align_vertical: str = 'center', identifier: Optional[str] = None, groups: Optional[List[str]] = None,
                 description: str = "", parent: Optional[Identifiable] = None) -> Optional[Text]:
        return self._add_primitive_internal(Text, layer, identifier, groups, description, parent,
                                            text_content=text, relative_position=position,
                                            height=height, font=font, style=style, line_spacing=line_spacing,
                                            align_horizontal=align_horizontal, align_vertical=align_vertical) # type: ignore

    def _add_mop_internal(self, MopClass: Type[MopType], part_identifier: Identifiable,
                          pid_source: Union[str, List[Identifiable]], name: str,
                          identifier: Optional[str] = None,
                          target_mop_identifier: Optional[Identifiable] = None, place_last: bool = True,
                          **kwargs) -> Optional[MopType]:
        """Internal helper to create, register, and link a MOP."""
        # 1. Resolve Part
        part = self.get_part(part_identifier)
        if not part:
            logger.error(f"Part identifier '{part_identifier}' not found.")
            return None

        # 2. Process pid_source (resolve primitive identifiers to UUIDs if needed)
        resolved_pid_source: Union[str, List[uuid.UUID]]
        if isinstance(pid_source, str):
            # Keep group name as string
            resolved_pid_source = pid_source
            # Optionally check if group exists?
            # if pid_source not in self._primitive_groups:
            #     logger.warning(f"MOP '{name}' references potentially non-existent group '{pid_source}'.")
        elif isinstance(pid_source, list):
            resolved_uuids: List[uuid.UUID] = []
            all_resolved = True
            for item in pid_source:
                prim_uuid = self._resolve_identifier(item, Primitive)
                if prim_uuid:
                    resolved_uuids.append(prim_uuid)
                else:
                    logger.warning(f"Could not resolve primitive identifier '{item}' for MOP '{name}'. Skipping.")
                    all_resolved = False
            resolved_pid_source = resolved_uuids
            if not resolved_uuids and pid_source: # Original list was not empty but resolved list is
                 logger.error(f"MOP '{name}' pid_source list {pid_source} resolved to no valid primitives.")
                 # return None # Optionally fail if no primitives found
            elif not all_resolved:
                 logger.warning(f"MOP '{name}' pid_source list resolved partially: {resolved_uuids}")

        else:
            logger.error(f"Invalid pid_source type for MOP '{name}': {type(pid_source)}. Must be group name (str) or list of primitive identifiers.")
            return None

        # 3. Create MOP instance
        mop_id_str = identifier if identifier else f"{name.replace(' ','_')}_{uuid.uuid4().hex[:6]}"
        try:
            mop = MopClass(user_identifier=mop_id_str,
                           name=name,
                           pid_source=resolved_pid_source, # Store resolved UUID list or group name
                           **kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate {MopClass.__name__} with identifier '{mop_id_str}': {e}")
            return None

        # 4. Register Entity
        if not self._register_entity(mop, self._mops):
            return None

        # 5. Establish Relationship using project method
        self.assign_mop_to_part(mop.internal_id, part.internal_id, target_mop_identifier, place_last)

        logger.info(f"Added {MopClass.__name__} '{name}' ({mop.internal_id}) to part '{part.user_identifier}'")
        return mop


    # --- Public API: Concrete MOP Adders ---

    def add_profile_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'Profile',
                        identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> Optional[ProfileMop]:
        return self._add_mop_internal(ProfileMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs) # type: ignore

    def add_pocket_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'Pocket',
                       identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> Optional[PocketMop]:
        return self._add_mop_internal(PocketMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs) # type: ignore

    def add_engrave_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'Engrave',
                        identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> Optional[EngraveMop]:
        return self._add_mop_internal(EngraveMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs) # type: ignore

    def add_drill_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'Drill',
                      identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> Optional[DrillMop]:
        return self._add_mop_internal(DrillMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs) # type: ignore


    # --- Public API: Entity Removal ---

    def remove_entity(self, identifier: Identifiable) -> bool:
        """Removes any entity (Layer, Part, Primitive, MOP) by its identifier."""
        entity_uuid = self._resolve_identifier(identifier)
        if not entity_uuid:
            logger.error(f"Entity '{identifier}' not found for removal.")
            return False
        entity = self._unregister_entity(entity_uuid)
        return entity is not None

    def remove_primitive(self, identifier: Identifiable) -> bool:
        """Removes a primitive and cleans up its relationships."""
        # Just call the generic remove
        return self.remove_entity(identifier)

    def remove_layer(self, identifier: Identifiable, remove_contained_primitives: bool = False) -> bool:
        """Removes a layer. Optionally removes contained primitives."""
        layer_uuid = self._resolve_identifier(identifier, Layer)
        if not layer_uuid:
            logger.error(f"Layer '{identifier}' not found for removal.")
            return False

        if remove_contained_primitives:
            # Get primitives *before* unregistering the layer cleans relationships
            prims_on_layer = list(self._layer_primitive_membership.get(layer_uuid, set()))
            logger.info(f"Removing {len(prims_on_layer)} primitives contained in layer '{identifier}'.")
            for prim_uuid in prims_on_layer:
                self.remove_primitive(prim_uuid) # Recursive removal handled here if prims have children

        # Unregister the layer itself (this also logs warning about unassigned prims if not removed above)
        return self.remove_entity(layer_uuid)

    def remove_part(self, identifier: Identifiable, remove_contained_mops: bool = False) -> bool:
        """Removes a part. Optionally removes contained MOPs."""
        part_uuid = self._resolve_identifier(identifier, Part)
        if not part_uuid:
            logger.error(f"Part '{identifier}' not found for removal.")
            return False

        if remove_contained_mops:
            mops_in_part = list(self._mop_order_in_part.get(part_uuid, []))
            logger.info(f"Removing {len(mops_in_part)} MOPs contained in part '{identifier}'.")
            for mop_uuid in mops_in_part:
                 self.remove_mop(mop_uuid)

        # Check if MOPs still assigned (if remove_contained_mops=False)
        if not remove_contained_mops and self._mop_order_in_part.get(part_uuid):
            logger.error(f"Cannot remove part '{identifier}' because it still contains MOPs and remove_contained_mops=False.")
            return False

        return self.remove_entity(part_uuid)

    def remove_mop(self, identifier: Identifiable) -> bool:
        """Removes a MOP and cleans up its relationships."""
        return self.remove_entity(identifier)


    # --- Public API: Transformations ---
    # Transformations are applied globally and propagated down the hierarchy

    def transform_primitive(self, primitive_identifier: Identifiable, matrix: np.ndarray, bake: bool = False) -> bool:
        """
        Applies a transformation matrix globally to a primitive and its descendants.
        
        Args:
            primitive_identifier: The primitive to transform
            matrix: The transformation matrix to apply (3x3)
            bake: If True, bakes the matrix directly into geometry; if False, updates effective_transform
            
        Returns:
            True if successful, False otherwise
        """
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
            logger.error(f"Primitive '{primitive_identifier}' not found for transformation.")
            return False
        if matrix.shape != (3, 3):
            logger.error("Transformation matrix must be 3x3.")
            return False

        # Apply transform to the target primitive
        if bake:
            # Directly bake the provided matrix into the geometry
            try:
                primitive.bake_geometry(matrix)
                logger.debug(f"Baked transformation matrix directly into primitive {primitive.user_identifier}")
            except Exception as e:
                logger.error(f"Error baking matrix into primitive {primitive.user_identifier}: {e}")
                return False
        else:
            # Non-baking just multiplies the effective transform
            # Apply the new matrix AFTER the existing one
            primitive.effective_transform = primitive.effective_transform @ matrix

        # # Recursively apply the SAME matrix to children for consistency
        # child_ids = self.get_children_of_primitive(primitive.internal_id)
        # for child_id in child_ids:
        #     # Recursive call - pass the exact same matrix and bake setting
        #     self.transform_primitive(child_id, matrix, bake=bake)

        # return True
        
        # Only for baking mode, we need to apply the transform to each child's geometry, else they gets applied twice: Now, and during output when they receive the parent's effective transform
        if bake:
            child_ids = self.get_children_of_primitive(primitive.internal_id)
            for child_id in child_ids:
                self.transform_primitive(child_id, matrix, bake=bake)

        return True

    # Convenience transformation methods
    def translate_primitive(self, primitive_identifier: Identifiable, dx: float, dy: float, bake: bool = False) -> bool:
        """Translates a primitive and its descendants."""
        return self.transform_primitive(primitive_identifier, translation_matrix(dx, dy), bake=bake)

    def rotate_primitive_deg(self, primitive_identifier: Identifiable, angle_deg: float,
                               cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Rotates a primitive and descendants around a center point (degrees)."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False
        # Use primitive's geometric center if cx, cy not provided
        if cx is None or cy is None:
            try:
                center_x, center_y = primitive.get_geometric_center()
            except Exception as e:
                logger.error(f"Could not get geometric center for '{primitive.user_identifier}' to rotate: {e}")
                center_x, center_y = 0.0, 0.0 # Fallback to origin? Or fail? Let's use origin.
        else:
            center_x, center_y = cx, cy

        return self.transform_primitive(primitive_identifier, rotation_matrix_deg(angle_deg, center_x, center_y), bake=bake)

    def scale_primitive(self, primitive_identifier: Identifiable, sx: float, sy: Optional[float] = None,
                         cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Scales a primitive and descendants around a center point."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False
        if cx is None or cy is None:
             try: center_x, center_y = primitive.get_geometric_center()
             except Exception: center_x, center_y = 0.0, 0.0
        else: center_x, center_y = cx, cy
        return self.transform_primitive(primitive_identifier, scale_matrix(sx, sy, center_x, center_y), bake=bake)

    def mirror_primitive_x(self, primitive_identifier: Identifiable, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Mirrors a primitive and descendants across a horizontal line y=cy.
        If cy is None, the center of the primitive is used."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False
        if cy is None:
             try: _, center_y = primitive.get_geometric_center(); cy = center_y
             except Exception: cy = 0.0
        return self.transform_primitive(primitive_identifier, mirror_x_matrix(cy), bake=bake)

    def mirror_primitive_y(self, primitive_identifier: Identifiable, cx: Optional[float] = None, bake: bool = False) -> bool:
        """Mirrors a primitive and descendants across a vertical line x=cx.
        If cx is None, the center of the primitive is used."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False
        if cx is None:
             try: center_x, _ = primitive.get_geometric_center(); cx = center_x
             except Exception: cx = 0.0
        return self.transform_primitive(primitive_identifier, mirror_y_matrix(cx), bake=bake)

    def skew_primitive(self, primitive_identifier: Identifiable, angle_x_deg: float = 0.0, angle_y_deg: float = 0.0, bake: bool = False) -> bool:
        """Applies skew (shear) to a primitive and descendants."""
        return self.transform_primitive(primitive_identifier, skew_matrix(angle_x_deg, angle_y_deg), bake=bake)

    def rotate_primitive_rad(self, primitive_identifier: Identifiable, angle_rad: float,
                               cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Rotates a primitive and descendants around a center point (radians)."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False
        if cx is None or cy is None:
             try: center_x, center_y = primitive.get_geometric_center()
             except Exception: center_x, center_y = 0.0, 0.0
        else: center_x, center_y = cx, cy
        return self.transform_primitive(primitive_identifier, rotation_matrix_rad(angle_rad, center_x, center_y), bake=bake)

    def align_primitive(self, primitive_identifier: Identifiable, 
                    alignment_point: Union[Tuple[float, float], float],
                    align_x: Optional[str] = None,
                    align_y: Optional[str] = None,
                    bake: bool = False) -> bool:
        """
        Aligns a primitive to a global point based on its transformed bounding box.
        
        The alignment respects the primitive's current transformation. For example,
        if a rectangle is rotated 90 degrees, aligning its "left" side will align
        what was originally the bottom side before rotation.
        
        Args:
            primitive_identifier: The primitive to align
            alignment_point: The global point to align to. Can be:
                            - A tuple (x, y) for both axes alignment
                            - A single float value when aligning only one axis
            align_x: Horizontal alignment - 'left', 'right', 'center', or None for no horizontal alignment
            align_y: Vertical alignment - 'top', 'bottom', 'center', or None for no vertical alignment
            bake: If True, bakes the translation into geometry; if False, updates effective_transform
            
        Returns:
            True if successful, False otherwise
        """
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
            logger.error(f"Primitive '{primitive_identifier}' not found for alignment.")
            return False
        
        # Get the bounding box in its transformed state
        try:
            bbox = primitive.get_bounding_box()
        except Exception as e:
            logger.error(f"Error getting bounding box for primitive '{primitive.user_identifier}': {e}")
            return False
        
        if not bbox.is_valid():
            logger.error(f"Invalid bounding box for primitive '{primitive.user_identifier}'")
            return False
        
        # Handle the alignment_point parameter
        x_point = None
        y_point = None
        
        if isinstance(alignment_point, (int, float)):
            # Single value - use it for the specified axis only
            if align_x is not None:
                x_point = float(alignment_point)
            if align_y is not None:
                y_point = float(alignment_point)
            if align_x is not None and align_y is not None:
                logger.warning(f"Single value {alignment_point} provided but both align_x and align_y specified. "
                            "Consider using a tuple (x,y) for clarity.")
        elif isinstance(alignment_point, (tuple, list)) and len(alignment_point) >= 2:
            # Tuple - extract x and y components
            x_point = alignment_point[0] if align_x is not None else None
            y_point = alignment_point[1] if align_y is not None else None
        else:
            logger.error(f"Invalid alignment_point: {alignment_point}. Must be a number or tuple (x,y).")
            return False
        
        # Calculate required translation for each specified alignment
        dx, dy = 0.0, 0.0
        
        # Ensure lower case alignment string
        if isinstance(align_x, str):
            align_x = align_x.lower()
        if isinstance(align_y, str):
            align_y = align_y.lower()

        # X-axis alignment
        if align_x is not None and x_point is not None:
            if align_x == 'left':
                dx = x_point - bbox.min_x
            elif align_x == 'right':
                dx = x_point - bbox.max_x
            elif align_x == 'center' or align_x == 'middle' or align_x == 'c':
                dx = x_point - (bbox.min_x + bbox.max_x) / 2
            else:
                logger.warning(f"Invalid x alignment '{align_x}'. Use 'left', 'right', or 'center'")
        
        # Y-axis alignment
        if align_y is not None and y_point is not None:
            if align_y == 'top' or align_y == 'upper':
                dy = y_point - bbox.max_y
            elif align_y == 'bottom' or align_y == 'lower':
                dy = y_point - bbox.min_y
            elif align_y == 'center' or align_y == 'middle' or align_y == 'c':
                dy = y_point - (bbox.min_y + bbox.max_y) / 2
            else:
                logger.warning(f"Invalid y alignment '{align_y}'. Use 'top'/'upper', 'bottom'/'lower', or 'center'")
        
        # If no movement needed, return early
        if (dx == 0 and dy == 0):
            return True
        
        # Create the translation matrix for the alignment
        translation = translation_matrix(dx, dy)
        
        if bake:
            # Bake the translation directly into the geometry
            try:
                # Store the original transform
                orig_transform = primitive.effective_transform.copy()
                # For baking, temporarily set the effective transform to just the translation
                primitive.effective_transform = translation
                primitive.bake_geometry()
                # Restore the original transform
                primitive.effective_transform = orig_transform
                logger.debug(f"Baked alignment translation ({dx}, {dy}) into primitive '{primitive.user_identifier}'")
            except Exception as e:
                logger.error(f"Error baking alignment for {primitive.user_identifier}: {e}")
                return False
        else:
            # For non-baking mode, we need to apply the translation in global coordinates,
            # so we need to apply it BEFORE the existing transforms in the chain
            primitive.effective_transform = translation @ primitive.effective_transform
            logger.debug(f"Applied alignment translation ({dx}, {dy}) to primitive '{primitive.user_identifier}'")
        
        return True

    # --- Recursive Bake ---
    def bake_primitive_transform(self, primitive_identifier: Identifiable, recursive: bool = True) -> bool:
        """
        Applies the effective transform of a primitive (and optionally its children)
        directly to its geometry, resetting the effective transform to identity.
        
        This is an improved version that correctly handles parent-child transform propagation.
        """
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
            logger.error(f"Primitive '{primitive_identifier}' not found for baking.")
            return False

        # Get the transform to bake (the primitive's own effective transform)
        transform_to_bake = primitive.effective_transform.copy()

        # Bake the primitive's geometry using its own transform
        if not np.allclose(transform_to_bake, identity_matrix()):
            try:
                # Apply baking to the primitive's geometry
                primitive.bake_geometry()
                logger.debug(f"Baked transform for primitive {primitive.user_identifier}")
                
                # Reset the primitive's transform to identity after baking
                primitive.effective_transform = identity_matrix()
            except Exception as e:
                logger.error(f"Error baking geometry for {primitive.user_identifier}: {e}")
                return False

        # Handle children recursively
        if recursive:
            child_ids = self.get_children_of_primitive(primitive.internal_id)
            for child_id in child_ids:
                child = self.get_primitive(child_id)
                if child:
                    # Apply the parent's baked transform to the child's transform
                    # This maintains the child's position relative to the parent
                    # ChildNew = ParentBaked * ChildOldEffective * ChildRelativeGeom
                    child.effective_transform = transform_to_bake @ child.effective_transform
                    # Then recursively bake the child with its updated transform
                    self.bake_primitive_transform(child_id, recursive=True)

        return True

    def bake_primitive_transform_component(self, primitive_identifier: Identifiable, 
                                        component_type: str, recursive: bool = True) -> bool:
        """
        Bakes a specific transformation component (translation, rotation, scale, mirror)
        into a primitive's geometry, while preserving other transformation components.
        
        Args:
            primitive_identifier: The primitive to bake
            component_type: The type of transformation to bake ('translation', 'rotation', 'scale', 'mirror_x', 'mirror_y', 'mirror')
            recursive: Whether to recursively apply to children
            
        Returns:
            True if successful, False otherwise
        """
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
            logger.error(f"Primitive '{primitive_identifier}' not found for component baking.")
            return False
            
        # Get the primitive's current transform
        current_transform = primitive.effective_transform.copy()
        
        # If identity matrix, nothing to bake
        if np.allclose(current_transform, identity_matrix()):
            logger.debug(f"Primitive '{primitive.user_identifier}' has identity transform, nothing to bake.")
            return True
        
        try:
            # Extract the component to bake
            component_to_bake = extract_transform_component(current_transform, component_type)
            
            # Skip if component is identity (nothing to bake)
            if np.allclose(component_to_bake, identity_matrix()):
                logger.debug(f"No {component_type} component to bake for '{primitive.user_identifier}'")
                return True
                
            # Apply just this component to the geometry
            orig_transform = primitive.effective_transform
            primitive.effective_transform = component_to_bake
            primitive.bake_geometry()
            
            # Update effective transform to contain everything except the baked component
            remaining_transform = remove_transform_component(current_transform, component_type)
            primitive.effective_transform = remaining_transform
            
            logger.debug(f"Baked {component_type} for primitive '{primitive.user_identifier}'")
            
            # Handle children if recursive
            if recursive:
                child_ids = self.get_children_of_primitive(primitive.internal_id)
                for child_id in child_ids:
                    child = self.get_primitive(child_id)
                    if child:
                        # For children, we need to apply the parent's baked component to their transforms
                        # This maintains their relative position to the parent
                        child.effective_transform = component_to_bake @ child.effective_transform
                        # Then recursively bake the same component
                        self.bake_primitive_transform_component(child_id, component_type, recursive=True)
            
            return True
        except Exception as e:
            logger.error(f"Error baking {component_type} for primitive '{primitive.user_identifier}': {e}")
            # Restore original transform in case of error
            primitive.effective_transform = current_transform
            return False

    def bake_all_primitives(self, component_type: Optional[str] = None) -> bool:
        """
        Bakes transformations for all primitives in the project.
        
        Args:
            component_type: Optional specific component to bake. If None, bakes entire transform.
            
        Returns:
            True if all primitives were successfully baked, False if any failed
        """
        # First identify root primitives (those without parents)
        root_primitives = []
        for prim_id in self._primitives:
            if prim_id not in self._primitive_parent_link:
                root_primitives.append(prim_id)
        
        # Bake each root primitive (and its descendants)
        all_success = True
        for root_id in root_primitives:
            if component_type:
                success = self.bake_primitive_transform_component(root_id, component_type, recursive=True)
            else:
                success = self.bake_primitive_transform(root_id, recursive=True)
            all_success = all_success and success
        
        return all_success

# , transform_to_bake: Optional[np.ndarray] = None

    # --- Public API: MOP Primitive Resolution ---

    def resolve_pid_source_to_uuids(self, pid_source: Union[str, List[uuid.UUID]]) -> List[uuid.UUID]:
        """Resolves a MOP's pid_source (group name or list of UUIDs) to a list of actual primitive UUIDs."""
        resolved_uuids: Set[uuid.UUID] = set()
        if isinstance(pid_source, str):
            # Resolve group name
            group_uuids = self._primitive_groups.get(pid_source, set())
            resolved_uuids.update(p_uuid for p_uuid in group_uuids if p_uuid in self._primitives)
            if not resolved_uuids and pid_source in self._primitive_groups:
                 logger.warning(f"Group '{pid_source}' exists but contains no primitives currently in the project.")
            elif pid_source not in self._primitive_groups:
                 logger.warning(f"Group '{pid_source}' referenced by MOP not found in project.")
        elif isinstance(pid_source, list):
            # Assume list of UUIDs (already resolved during MOP creation ideally)
            resolved_uuids.update(p_uuid for p_uuid in pid_source if p_uuid in self._primitives)
            if len(resolved_uuids) != len(pid_source):
                 missing = set(pid_source) - resolved_uuids
                 logger.warning(f"MOP references primitive UUIDs that are not in the project: {missing}")
        else:
            logger.error(f"Invalid pid_source type for resolution: {type(pid_source)}")

        # Return consistently sorted list
        return sorted(list(resolved_uuids))


    # --- Public API: Bounding Box ---

    def get_bounding_box(self, include_layers: Optional[Sequence[Identifiable]] = None,
                        include_primitives: Optional[Sequence[Identifiable]] = None) -> BoundingBox:
        """
        Calculates the overall bounding box of specified entities or all primitives.
        Args:
            include_layers: Optional list of layer identifiers. Only primitives on these layers are included.
            include_primitives: Optional list of primitive identifiers to include.
        Returns:
            A BoundingBox object. Returns an invalid BoundingBox if no valid primitives are found.
        """
        overall_bb = BoundingBox()
        target_primitive_ids: Optional[Set[uuid.UUID]] = None

        if include_primitives:
            target_primitive_ids = set()
            for p_id in include_primitives:
                p_uuid = self._resolve_identifier(p_id, Primitive)
                if p_uuid:
                    target_primitive_ids.add(p_uuid)

        if include_layers:
            layer_primitive_ids = set()
            for l_id in include_layers:
                l_uuid = self._resolve_identifier(l_id, Layer)
                if l_uuid:
                    layer_primitive_ids.update(self._layer_primitive_membership.get(l_uuid, set()))
            # Intersect with specific primitives if both are provided
            if target_primitive_ids is not None:
                target_primitive_ids &= layer_primitive_ids
            else:
                target_primitive_ids = layer_primitive_ids

        # If no specific targets, use all primitives
        primitives_to_scan = target_primitive_ids if target_primitive_ids is not None else self._primitives.keys()

        for prim_uuid in primitives_to_scan:
            primitive = self._primitives.get(prim_uuid)
            if primitive:
                try:
                    prim_bb = primitive.get_bounding_box()
                    if prim_bb.is_valid():
                        overall_bb = overall_bb.union(prim_bb)
                except Exception as e:
                    logger.error(f"Error getting bounding box for primitive {primitive.user_identifier}: {e}")

        return overall_bb

    # --- Public API: Context Management (for building) ---

    @property
    def current_transform(self) -> np.ndarray:
        """Returns the transformation matrix at the top of the stack."""
        return self._transform_stack[-1]

    def set_transform(self, matrix: np.ndarray):
        """Sets the current transformation matrix."""
        if matrix.shape == (3,3):
            self._transform_stack[-1] = matrix.copy()
        else:
            logger.error("Invalid matrix shape for set_transform. Must be 3x3.")

    def push_transform(self, matrix: Optional[np.ndarray] = None):
        """Pushes a new transform onto the stack. If matrix is provided, it's combined."""
        current_top = self.current_transform
        # New matrix applies *after* the current top
        new_top = current_top @ matrix if matrix is not None else current_top.copy()
        if new_top.shape == (3,3):
            self._transform_stack.append(new_top)
        else:
             logger.error("Invalid matrix shape for push_transform. Must be 3x3.")


    def pop_transform(self) -> Optional[np.ndarray]:
        """Pops the top transform matrix from the stack. Cannot pop the base identity."""
        if len(self._transform_stack) > 1:
            return self._transform_stack.pop()
        else:
            logger.warning("Cannot pop the base transformation matrix.")
            return None

    def reset_transform(self):
        """Resets the transform stack to only the base identity matrix."""
        self._transform_stack = [identity_matrix()]

    def set_cursor(self, x: float, y: float):
        """Sets the current cursor position for primitive placement."""
        self._cursor = (x, y)
        logger.debug(f"Cursor set to: {self._cursor}")

    def reset_cursor(self):
        """Resets the cursor to (0, 0)."""
        self.set_cursor(0.0, 0.0)

    # --- Persistence ---

    def save_state(self, file_path: str) -> None:
        """Saves the entire project state (including registries) to a pickle file."""
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        except OSError as e:
             logger.error(f"Could not create directory for state file {file_path}: {e}")
             return # Or raise

        # Prepare for pickling: remove weakrefs or other unpickleable things if any
        # (Primitives already handle _project_ref in __getstate__)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            logger.info(f"Project state saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving project state to {file_path}: {e}")
            raise # Re-raise the exception

    @staticmethod
    def load_state(file_path: str) -> Optional["CamBamProject"]:
        """Loads a project state from a pickle file."""
        if not os.path.exists(file_path):
            logger.error(f"Project state file not found: {file_path}")
            return None
        try:
            with open(file_path, 'rb') as f:
                project = pickle.load(f)

            # Post-load restoration:
            # - Relink primitives to the loaded project instance
            if isinstance(project, CamBamProject):
                for primitive in project._primitives.values():
                    primitive.set_project_link(project)
                logger.info(f"Loaded project state '{project.project_name}' from {file_path}")
                return project
            else:
                 logger.error(f"File {file_path} did not contain a valid CamBamProject object.")
                 return None
        except pickle.UnpicklingError as e:
            logger.error(f"Error unpickling project state from {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred loading project state from {file_path}: {e}")
            raise # Re-raise other exceptions