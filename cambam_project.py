# cambam_project.py
import uuid
import logging
import pickle
import os
from copy import deepcopy
import weakref
from collections import defaultdict
from typing import (
    List, Dict, Tuple, Union, Optional, Set, Any, Sequence, cast, Type, TypeVar
)

import numpy as np

# Module imports
from cad_common import CamBamEntity, BoundingBox, CamBamError, ProjectConfigurationError, TransformationError
from cad_transformations import (
    identity_matrix, translation_matrix, rotation_matrix_deg,
    scaling_matrix, mirror_x_matrix, mirror_y_matrix
)
from cambam_entities import (
    Layer, Part, Primitive, Mop, Pline, Circle, Rect, Arc, Points, Text,
    ProfileMop, PocketMop, EngraveMop, DrillMop
)

logger = logging.getLogger(__name__)

# Type alias for user-facing identifiers
Identifiable = Union[str, uuid.UUID, CamBamEntity]
PrimitiveType = TypeVar('PrimitiveType', bound=Primitive)
MopType = TypeVar('MopType', bound=Mop)


class CamBamProject:
    """
    Manages the collections of Layers, Parts, Primitives, and MOPs that
    constitute a CamBam project. Handles entity relationships (parent/child, groups)
    and provides methods for transformations and project operations.
    """
    def __init__(self, project_name: str, default_tool_diameter: Optional[float] = None):
        self.project_name: str = project_name
        # Project-level defaults (can be overridden by Part/MOP)
        self.default_tool_diameter: Optional[float] = default_tool_diameter

        # Core entity storage
        self._primitives: Dict[uuid.UUID, Primitive] = {}
        self._layers: Dict[uuid.UUID, Layer] = {}
        self._parts: Dict[uuid.UUID, Part] = {}
        self._mops: Dict[uuid.UUID, Mop] = {}

        # Ordering for XML output
        self._layer_order: List[uuid.UUID] = []
        self._part_order: List[uuid.UUID] = []
        # MOP order is stored within Part.mop_ids

        # Lookup for user identifiers
        self._identifier_registry: Dict[str, uuid.UUID] = {}

        # Primitive group management: group_name -> set(primitive_uuid)
        self._primitive_groups: Dict[str, Set[uuid.UUID]] = defaultdict(set)

        logger.info(f"Initialized CamBamProject: '{self.project_name}'")

    # --- Entity Registration / Unregistration ---

    def _register_entity(self, entity: CamBamEntity):
        """Internal helper to register any entity type."""
        registry: Optional[Dict[uuid.UUID, CamBamEntity]] = None
        order_list: Optional[List[uuid.UUID]] = None

        if isinstance(entity, Primitive): registry = self._primitives
        elif isinstance(entity, Layer): registry = self._layers; order_list = self._layer_order
        elif isinstance(entity, Part): registry = self._parts; order_list = self._part_order
        elif isinstance(entity, Mop): registry = self._mops
        else:
            raise TypeError(f"Unsupported entity type for registration: {type(entity)}")

        if entity.internal_id in registry:
            logger.warning(f"Entity {entity.user_identifier} ({entity.internal_id}) already registered. Overwriting.")

        registry[entity.internal_id] = entity

        # Update identifier registry if user_identifier is set
        if entity.user_identifier:
            if entity.user_identifier in self._identifier_registry and \
               self._identifier_registry[entity.user_identifier] != entity.internal_id:
                logger.warning(f"User identifier '{entity.user_identifier}' is already used by entity {self._identifier_registry[entity.user_identifier]}. "
                               f"Cannot map it to {entity.internal_id}. Access via UUID or different identifier.")
            else:
                 self._identifier_registry[entity.user_identifier] = entity.internal_id

        # Add to order list if applicable and not already present
        if order_list is not None and entity.internal_id not in order_list:
             # Default add to end, specific add methods handle insertion point
             order_list.append(entity.internal_id)

        # --- Specific registration logic ---
        if isinstance(entity, Primitive):
            entity._project_ref = weakref.ref(self) # Set back-reference
            self._update_primitive_layer_link(entity)
            self._update_primitive_group_links(entity)
            self._update_primitive_parent_child_links(entity)

        elif isinstance(entity, Mop):
            # Ensure MOP is linked to its Part's order list
            part = self.get_part(entity.part_id)
            if part and entity.internal_id not in part.mop_ids:
                 # Default add to end, specific add_mop methods handle insertion
                 part.mop_ids.append(entity.internal_id)
            elif not part:
                 logger.error(f"MOP '{entity.user_identifier}' registered but its Part ID {entity.part_id} not found.")

    def _unregister_entity(self, entity_id: uuid.UUID) -> Optional[CamBamEntity]:
        """Internal helper to unregister any entity type and clean up references."""
        entity: Optional[CamBamEntity] = None
        registry: Optional[Dict[uuid.UUID, CamBamEntity]] = None
        order_list: Optional[List[uuid.UUID]] = None

        if entity_id in self._primitives:
            entity = self._primitives.pop(entity_id)
            registry = self._primitives
        elif entity_id in self._layers:
            entity = self._layers.pop(entity_id)
            registry = self._layers
            order_list = self._layer_order
        elif entity_id in self._parts:
            entity = self._parts.pop(entity_id)
            registry = self._parts
            order_list = self._part_order
        elif entity_id in self._mops:
            entity = self._mops.pop(entity_id)
            registry = self._mops
        else:
            logger.warning(f"Entity ID {entity_id} not found for unregistration.")
            return None

        if not entity: # Should not happen if ID was found initially
             return None

        # Remove from identifier registry
        if entity.user_identifier and entity.user_identifier in self._identifier_registry:
            if self._identifier_registry[entity.user_identifier] == entity_id:
                del self._identifier_registry[entity.user_identifier]

        # Remove from order list
        if order_list is not None:
             try:
                 order_list.remove(entity_id)
             except ValueError:
                 pass # Not in order list, ignore

        # --- Specific unregistration logic ---
        if isinstance(entity, Primitive):
            primitive = cast(Primitive, entity)
            # Remove from layer
            layer = self.get_layer(primitive.layer_id)
            if layer:
                 layer.primitive_ids.discard(primitive.internal_id)
            # Remove from groups
            self._remove_primitive_from_all_groups(primitive)
            # Remove from parent's child list
            if primitive.parent_primitive_id:
                parent = self.get_primitive(primitive.parent_primitive_id)
                if parent:
                    parent._child_primitive_ids.discard(primitive.internal_id)
            # Clear parent link for direct children and remove self as child from them (shouldn't be needed)
            for child_id in list(primitive._child_primitive_ids): # Iterate copy
                 child = self.get_primitive(child_id)
                 if child:
                     child.parent_primitive_id = None # Child loses its parent
                     # Child's _child_primitive_ids remains unchanged
            primitive._project_ref = None # Clear back reference

        elif isinstance(entity, Mop):
            mop = cast(Mop, entity)
            # Remove from part's MOP list
            part = self.get_part(mop.part_id)
            if part:
                 try:
                     part.mop_ids.remove(mop.internal_id)
                 except ValueError:
                     pass

        elif isinstance(entity, Layer):
            layer = cast(Layer, entity)
            if layer.primitive_ids:
                 logger.warning(f"Layer '{layer.user_identifier}' still contains {len(layer.primitive_ids)} primitives upon removal. Primitives are now orphaned from layer.")
                 # Primitives remain in the project but lose layer association. Consider adding remove_layer method to handle this.

        elif isinstance(entity, Part):
            part = cast(Part, entity)
            if part.mop_ids:
                 logger.warning(f"Part '{part.user_identifier}' still contains {len(part.mop_ids)} MOPs upon removal. MOPs are now orphaned from part.")
                 # MOPs remain in project but lose part association.

        logger.debug(f"Unregistered {type(entity).__name__}: {entity.user_identifier} ({entity.internal_id})")
        return entity

    # --- Link Management Helpers ---

    def _update_primitive_layer_link(self, primitive: Primitive):
        """Ensures primitive is added/removed from layer's primitive_ids set."""
        # Remove from old layer if necessary (though layer_id change should be handled elsewhere)
        # Add to new layer
        layer = self.get_layer(primitive.layer_id)
        if layer:
            layer.primitive_ids.add(primitive.internal_id)
        else:
            # This indicates an issue - primitive added with invalid layer ID
            logger.error(f"Primitive '{primitive.user_identifier}' references non-existent layer {primitive.layer_id}.")
            # Assign to a default layer? Or raise error? For now, just log.

    def _update_primitive_group_links(self, primitive: Primitive, old_groups: Optional[Set[str]] = None):
        """Updates group memberships based on primitive.groups list."""
        current_groups = set(primitive.groups)
        old_groups = old_groups if old_groups is not None else set()

        groups_to_add = current_groups - old_groups
        groups_to_remove = old_groups - current_groups

        for group_name in groups_to_remove:
            if group_name in self._primitive_groups:
                self._primitive_groups[group_name].discard(primitive.internal_id)
                if not self._primitive_groups[group_name]: # Remove empty group set
                     del self._primitive_groups[group_name]

        for group_name in groups_to_add:
             self._primitive_groups[group_name].add(primitive.internal_id)

    def _remove_primitive_from_all_groups(self, primitive: Primitive):
        """Removes a primitive from all groups it might be in."""
        current_groups = set(primitive.groups)
        for group_name in current_groups:
             if group_name in self._primitive_groups:
                 self._primitive_groups[group_name].discard(primitive.internal_id)
                 if not self._primitive_groups[group_name]:
                     del self._primitive_groups[group_name]

    def _update_primitive_parent_child_links(self, primitive: Primitive, old_parent_id: Optional[uuid.UUID] = None):
        """Updates parent's child list and primitive's parent link."""
        new_parent_id = primitive.parent_primitive_id

        # Remove from old parent's children if necessary
        if old_parent_id and old_parent_id != new_parent_id:
            old_parent = self.get_primitive(old_parent_id)
            if old_parent:
                old_parent._child_primitive_ids.discard(primitive.internal_id)

        # Add to new parent's children
        if new_parent_id:
            new_parent = self.get_primitive(new_parent_id)
            if new_parent:
                new_parent._child_primitive_ids.add(primitive.internal_id)
            else:
                # Invalid parent ID set on primitive
                logger.warning(f"Primitive '{primitive.user_identifier}' has invalid parent ID {new_parent_id}. Clearing parent link.")
                primitive.parent_primitive_id = None # Correct the link

    # --- Entity Accessors ---

    def _resolve_identifier(self, identifier: Identifiable,
                            expected_type: Optional[Type[CamBamEntity]] = None) -> Optional[uuid.UUID]:
        """Resolves various identifier forms to a UUID, optionally checking type."""
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
            logger.error(f"Invalid identifier type provided: {type(identifier)}")
            return None

        # Optional type check
        if expected_type and target_uuid:
            entity = self.get_entity(target_uuid) # Use generic getter
            if not entity:
                # UUID known but object missing? Should not happen if registry is consistent.
                logger.error(f"Identifier '{identifier}' resolved to UUID {target_uuid}, but the entity object is missing.")
                return None
            if not isinstance(entity, expected_type):
                logger.debug(f"Identifier '{identifier}' resolved to UUID {target_uuid}, but type mismatch (Expected {expected_type.__name__}, Got {type(entity).__name__}).")
                return None # Found, but wrong type

        return target_uuid

    def get_entity(self, identifier: Identifiable) -> Optional[CamBamEntity]:
         """Gets any entity by its identifier."""
         entity_uuid = self._resolve_identifier(identifier)
         if not entity_uuid: return None
         return (self._primitives.get(entity_uuid) or
                 self._layers.get(entity_uuid) or
                 self._parts.get(entity_uuid) or
                 self._mops.get(entity_uuid))

    def get_primitive(self, identifier: Identifiable) -> Optional[Primitive]:
        prim_uuid = self._resolve_identifier(identifier, Primitive)
        return self._primitives.get(prim_uuid) if prim_uuid else None

    def get_layer(self, identifier: Identifiable) -> Optional[Layer]:
        layer_uuid = self._resolve_identifier(identifier, Layer)
        return self._layers.get(layer_uuid) if layer_uuid else None

    def get_part(self, identifier: Identifiable) -> Optional[Part]:
        part_uuid = self._resolve_identifier(identifier, Part)
        return self._parts.get(part_uuid) if part_uuid else None

    def get_mop(self, identifier: Identifiable) -> Optional[Mop]:
        mop_uuid = self._resolve_identifier(identifier, Mop)
        return self._mops.get(mop_uuid) if mop_uuid else None

    def list_primitives(self, layer: Optional[Identifiable] = None) -> List[Primitive]:
        prims = list(self._primitives.values())
        if layer:
            layer_obj = self.get_layer(layer)
            if layer_obj:
                return [p for p in prims if p.layer_id == layer_obj.internal_id]
            else:
                logger.warning(f"Layer '{layer}' not found for filtering primitives.")
                return []
        return prims

    def list_layers(self) -> List[Layer]:
        return [self._layers[uid] for uid in self._layer_order if uid in self._layers]

    def list_parts(self) -> List[Part]:
        return [self._parts[uid] for uid in self._part_order if uid in self._parts]

    def list_mops(self, part: Optional[Identifiable] = None) -> List[Mop]:
        if part:
            part_obj = self.get_part(part)
            if part_obj:
                return [self._mops[uid] for uid in part_obj.mop_ids if uid in self._mops]
            else:
                logger.warning(f"Part '{part}' not found for filtering MOPs.")
                return []
        # Return all MOPs if no part specified
        all_mops = []
        for p in self.list_parts():
             all_mops.extend(self.list_mops(part=p))
        return all_mops


    def list_groups(self) -> List[str]:
        return sorted(list(self._primitive_groups.keys()))

    def get_primitives_in_group(self, group_name: str) -> List[Primitive]:
        uuids = self._primitive_groups.get(group_name, set())
        return [self._primitives[uid] for uid in uuids if uid in self._primitives]


    # --- Entity Creation Methods ---

    def add_layer(self, layer: Layer,
                  target_layer: Optional[Identifiable] = None,
                  place_after: bool = True) -> Layer:
        """Adds a Layer object to the project, handling order."""
        if not isinstance(layer, Layer):
             raise TypeError("Must add a Layer instance.")

        self._register_entity(layer) # Handles registry and adds to end of order list by default

        # Adjust order if target is specified
        if target_layer:
            target_uuid = self._resolve_identifier(target_layer, Layer)
            if target_uuid and target_uuid in self._layer_order:
                 current_index = self._layer_order.index(layer.internal_id)
                 self._layer_order.pop(current_index) # Remove from end

                 target_index = self._layer_order.index(target_uuid)
                 insert_pos = target_index + 1 if place_after else target_index
                 self._layer_order.insert(insert_pos, layer.internal_id)
            else:
                 logger.warning(f"Target layer '{target_layer}' for ordering not found. Layer '{layer.user_identifier}' added at the end.")
        logger.debug(f"Added Layer: {layer.user_identifier}")
        return layer

    def add_part(self, part: Part,
                 target_part: Optional[Identifiable] = None,
                 place_after: bool = True) -> Part:
        """Adds a Part object to the project, handling order."""
        if not isinstance(part, Part):
            raise TypeError("Must add a Part instance.")

        self._register_entity(part)

        if target_part:
            target_uuid = self._resolve_identifier(target_part, Part)
            if target_uuid and target_uuid in self._part_order:
                 current_index = self._part_order.index(part.internal_id)
                 self._part_order.pop(current_index)
                 target_index = self._part_order.index(target_uuid)
                 insert_pos = target_index + 1 if place_after else target_index
                 self._part_order.insert(insert_pos, part.internal_id)
            else:
                 logger.warning(f"Target part '{target_part}' for ordering not found. Part '{part.user_identifier}' added at the end.")
        logger.debug(f"Added Part: {part.user_identifier}")
        return part

    def add_primitive(self, primitive: Primitive) -> Primitive:
        """Adds a Primitive object to the project."""
        if not isinstance(primitive, Primitive):
            raise TypeError("Must add a Primitive instance.")
        if not self.get_layer(primitive.layer_id):
             raise ProjectConfigurationError(f"Cannot add primitive '{primitive.user_identifier}': Layer {primitive.layer_id} does not exist.")
        if primitive.parent_primitive_id and not self.get_primitive(primitive.parent_primitive_id):
             raise ProjectConfigurationError(f"Cannot add primitive '{primitive.user_identifier}': Parent {primitive.parent_primitive_id} does not exist.")

        self._register_entity(primitive) # Handles registry, layer, group, parent/child links
        logger.debug(f"Added Primitive: {primitive.user_identifier} to Layer {primitive.layer_id}")
        return primitive

    def add_mop(self, mop: MopType,
                target_mop: Optional[Identifiable] = None,
                place_after: bool = True) -> MopType:
        """Adds a Mop object to the project, handling order within its Part."""
        if not isinstance(mop, Mop):
            raise TypeError("Must add a Mop instance.")
        part = self.get_part(mop.part_id)
        if not part:
             raise ProjectConfigurationError(f"Cannot add MOP '{mop.user_identifier}': Part {mop.part_id} does not exist.")

        self._register_entity(mop) # Handles registry and adds to end of part.mop_ids by default

        # Adjust order within the part's list
        if target_mop:
             target_uuid = self._resolve_identifier(target_mop, Mop)
             if target_uuid and target_uuid in part.mop_ids:
                 try:
                     current_index = part.mop_ids.index(mop.internal_id)
                     part.mop_ids.pop(current_index) # Remove from current position
                     target_index = part.mop_ids.index(target_uuid)
                     insert_pos = target_index + 1 if place_after else target_index
                     part.mop_ids.insert(insert_pos, mop.internal_id)
                 except ValueError:
                      logger.warning(f"MOP '{mop.user_identifier}' or target '{target_mop}' index issue during ordering. Added at end.")
                      if mop.internal_id not in part.mop_ids: part.mop_ids.append(mop.internal_id) # Ensure it's added
             else:
                 logger.warning(f"Target MOP '{target_mop}' for ordering not found in part '{part.user_identifier}'. MOP '{mop.user_identifier}' added at the end.")
                 if mop.internal_id not in part.mop_ids: part.mop_ids.append(mop.internal_id) # Ensure it's added

        logger.debug(f"Added MOP: {mop.user_identifier} to Part {part.user_identifier}")
        return mop


    # --- Entity Modification Methods ---

    def set_primitive_layer(self, primitive_identifier: Identifiable, layer_identifier: Identifiable) -> bool:
        """Moves a primitive to a different layer."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
             logger.error(f"Primitive '{primitive_identifier}' not found.")
             return False
        new_layer = self.get_layer(layer_identifier)
        if not new_layer:
             logger.error(f"Target layer '{layer_identifier}' not found.")
             return False

        if primitive.layer_id != new_layer.internal_id:
             old_layer = self.get_layer(primitive.layer_id)
             if old_layer:
                 old_layer.primitive_ids.discard(primitive.internal_id)
             primitive.layer_id = new_layer.internal_id
             new_layer.primitive_ids.add(primitive.internal_id)
             logger.info(f"Moved primitive '{primitive.user_identifier}' to layer '{new_layer.user_identifier}'.")
        return True

    def set_primitive_groups(self, primitive_identifier: Identifiable, groups: List[str]) -> bool:
        """Sets the groups for a primitive, updating project group registry."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
             logger.error(f"Primitive '{primitive_identifier}' not found.")
             return False

        old_groups = set(primitive.groups)
        primitive.groups = sorted(list(set(groups))) # Store unique, sorted groups
        self._update_primitive_group_links(primitive, old_groups)
        logger.debug(f"Set groups for primitive '{primitive.user_identifier}' to: {primitive.groups}")
        return True

    def set_primitive_parent(self, primitive_identifier: Identifiable, parent_identifier: Optional[Identifiable]) -> bool:
        """Sets or clears the parent of a primitive."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
            logger.error(f"Primitive '{primitive_identifier}' not found.")
            return False

        new_parent: Optional[Primitive] = None
        if parent_identifier:
            new_parent = self.get_primitive(parent_identifier)
            if not new_parent:
                logger.error(f"Parent primitive '{parent_identifier}' not found.")
                return False
            # Prevent self-parenting
            if new_parent.internal_id == primitive.internal_id:
                 logger.error(f"Cannot set primitive '{primitive.user_identifier}' as its own parent.")
                 return False
            # Prevent cyclic parenting (basic check: new parent cannot be child of primitive)
            if self._is_descendant(new_parent, primitive):
                 logger.error(f"Cannot set parent for '{primitive.user_identifier}' to '{new_parent.user_identifier}': Creates cyclic dependency.")
                 return False

        old_parent_id = primitive.parent_primitive_id
        new_parent_id = new_parent.internal_id if new_parent else None

        if old_parent_id != new_parent_id:
            primitive.parent_primitive_id = new_parent_id
            self._update_primitive_parent_child_links(primitive, old_parent_id)
            parent_name = f"'{new_parent.user_identifier}'" if new_parent else "None"
            logger.info(f"Set parent for primitive '{primitive.user_identifier}' to: {parent_name}")
        return True

    def _is_descendant(self, potential_descendant: Primitive, ancestor: Primitive) -> bool:
        """Checks if potential_descendant is a child or further descendant of ancestor."""
        current = potential_descendant
        while current.parent_primitive_id:
             if current.parent_primitive_id == ancestor.internal_id:
                 return True
             parent = self.get_primitive(current.parent_primitive_id)
             if not parent: return False # Broken chain
             current = parent
        return False


    # --- Removal Methods ---

    def remove_primitive(self, primitive_identifier: Identifiable, remove_children: bool = True) -> bool:
        """Removes a primitive and optionally its descendants."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
            logger.error(f"Cannot remove: Primitive '{primitive_identifier}' not found.")
            return False

        prim_id = primitive.internal_id
        prim_name = primitive.user_identifier

        # Handle children first
        children_to_process = list(primitive._child_primitive_ids) # Copy IDs
        if remove_children:
            logger.debug(f"Recursively removing children of '{prim_name}'.")
            for child_id in children_to_process:
                self.remove_primitive(child_id, remove_children=True) # Recursive call
        else:
             # Re-parent children to the primitive's parent (or make them top-level)
             new_parent_id = primitive.parent_primitive_id
             logger.debug(f"Re-parenting children of '{prim_name}' to {new_parent_id or 'top-level'}.")
             for child_id in children_to_process:
                  child = self.get_primitive(child_id)
                  if child:
                      # Use set_primitive_parent for proper link updates
                      self.set_primitive_parent(child, new_parent_id)

        # Now remove the primitive itself
        removed_entity = self._unregister_entity(prim_id)
        if removed_entity:
            logger.info(f"Removed primitive: {prim_name}")

            # Check MOPs using this primitive (by ID or potentially by group if group becomes empty)
            affected_mops = []
            primitive_groups = set(primitive.groups)
            for mop in self.list_mops():
                 is_affected = False
                 if isinstance(mop.pid_source, list) and prim_id in mop.pid_source:
                     is_affected = True
                     # Optionally remove ID from MOP source?
                     # mop.pid_source.remove(prim_id)
                 elif isinstance(mop.pid_source, str) and mop.pid_source in primitive_groups:
                     # If the group is now empty after removing this primitive, MOP is affected
                     if not self.get_primitives_in_group(mop.pid_source):
                          is_affected = True

                 if is_affected:
                      affected_mops.append(mop.user_identifier)

            if affected_mops:
                 logger.warning(f"Removed primitive '{prim_name}' was potentially used by MOPs: {', '.join(affected_mops)}. Review MOP source.")

            return True
        return False


    def remove_mop(self, mop_identifier: Identifiable) -> bool:
        """Removes a MOP from the project."""
        mop = self.get_mop(mop_identifier)
        if not mop:
            logger.error(f"Cannot remove: MOP '{mop_identifier}' not found.")
            return False
        mop_name = mop.user_identifier
        removed_entity = self._unregister_entity(mop.internal_id)
        if removed_entity:
            logger.info(f"Removed MOP: {mop_name}")
            return True
        return False

    def remove_layer(self, layer_identifier: Identifiable, transfer_primitives_to: Optional[Identifiable] = None) -> bool:
        """Removes a layer, optionally transferring its primitives."""
        layer_to_remove = self.get_layer(layer_identifier)
        if not layer_to_remove:
            logger.error(f"Cannot remove: Layer '{layer_identifier}' not found.")
            return False

        target_layer: Optional[Layer] = None
        if transfer_primitives_to:
            target_layer = self.get_layer(transfer_primitives_to)
            if not target_layer:
                logger.error(f"Target layer '{transfer_primitives_to}' for primitive transfer not found. Cannot remove layer.")
                return False
            if target_layer.internal_id == layer_to_remove.internal_id:
                 logger.error(f"Cannot transfer primitives to the layer being removed ('{layer_identifier}').")
                 return False

        primitive_ids_to_process = list(layer_to_remove.primitive_ids) # Copy IDs

        if target_layer:
            logger.info(f"Transferring {len(primitive_ids_to_process)} primitives from layer '{layer_to_remove.user_identifier}' to '{target_layer.user_identifier}'.")
            for prim_id in primitive_ids_to_process:
                 # Use set_primitive_layer for proper updates
                 self.set_primitive_layer(prim_id, target_layer.internal_id)
        else:
             logger.warning(f"Removing layer '{layer_to_remove.user_identifier}' and its {len(primitive_ids_to_process)} contained primitives.")
             for prim_id in primitive_ids_to_process:
                 self.remove_primitive(prim_id, remove_children=True) # Remove children as well

        # Now remove the layer itself
        layer_name = layer_to_remove.user_identifier
        removed_entity = self._unregister_entity(layer_to_remove.internal_id)
        if removed_entity:
            logger.info(f"Removed layer: {layer_name}")
            return True
        return False


    def remove_part(self, part_identifier: Identifiable, remove_contained_mops: bool = True) -> bool:
        """Removes a part, optionally removing its MOPs."""
        part_to_remove = self.get_part(part_identifier)
        if not part_to_remove:
            logger.error(f"Cannot remove: Part '{part_identifier}' not found.")
            return False

        mop_ids_to_process = list(part_to_remove.mop_ids) # Copy IDs

        if remove_contained_mops:
            logger.info(f"Removing part '{part_to_remove.user_identifier}' and its {len(mop_ids_to_process)} MOPs.")
            for mop_id in mop_ids_to_process:
                self.remove_mop(mop_id)
        elif mop_ids_to_process:
            logger.error(f"Cannot remove part '{part_to_remove.user_identifier}': Contains MOPs and 'remove_contained_mops' is False.")
            return False

        # Now remove the part itself
        part_name = part_to_remove.user_identifier
        removed_entity = self._unregister_entity(part_to_remove.internal_id)
        if removed_entity:
            logger.info(f"Removed part: {part_name}")
            return True
        return False

    # --- Transformation Methods ---

    def transform_primitive(self,
                            primitive_identifier: Identifiable,
                            matrix: np.ndarray,
                            transform_type: str = 'global', # 'global' or 'local'
                            bake: bool = False) -> bool:
        """
        Applies a transformation matrix to a primitive and its children (if applicable).

        Args:
            primitive_identifier: Identifier of the primitive to transform.
            matrix: The 3x3 transformation matrix.
            transform_type: 'global' (pre-multiply effective total) or 'local' (post-multiply effective local).
            bake: If True, apply the transform directly to the geometry and reset
                  effective transforms recursively. If False, update effective transform.

        Returns:
            True if successful, False otherwise.
        """
        primitive = self.get_primitive(primitive_identifier)
        if not primitive:
            logger.error(f"Primitive '{primitive_identifier}' not found for transformation.")
            return False

        if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
            logger.error(f"Invalid transform matrix shape for '{primitive.user_identifier}'. Must be 3x3.")
            return False

        if np.isclose(np.linalg.det(matrix), 0):
             logger.warning(f"Applying singular (degenerate) transformation matrix to '{primitive.user_identifier}'. Geometry may collapse.")

        if bake:
            # Calculate the final total transform after applying the new matrix globally
            current_total_tf = primitive._get_total_transform()
            if transform_type == 'global':
                # Global bake means the final state is matrix @ current_total
                final_bake_transform = matrix @ current_total_tf
            elif transform_type == 'local':
                 # Local bake means the final state is current_total @ matrix
                 # But we need to bake relative to the *final* state after local transform.
                 # The local matrix is applied to the effective transform.
                 # Final total = ParentTotal @ (Effective @ Local)
                 # Bake transform should be ParentTotal @ Effective @ Local = CurrentTotal @ Local
                 final_bake_transform = current_total_tf @ matrix
            else:
                 logger.error(f"Invalid transform_type '{transform_type}' for baking.")
                 return False

            logger.debug(f"Baking primitive '{primitive.user_identifier}' with final transform.")
            # Delegate recursive baking and potential replacement handling to internal method
            self._bake_primitive_and_handle_replacement(primitive, final_bake_transform)

        else:
            # Apply transform without baking (modify effective_transform)
            if transform_type == 'global':
                primitive._apply_transform_globally(matrix)
            elif transform_type == 'local':
                primitive._apply_transform_locally(matrix)
            else:
                logger.error(f"Invalid transform_type '{transform_type}'.")
                return False
            logger.debug(f"Applied {transform_type} transform to '{primitive.user_identifier}'.")

        return True


    def _bake_primitive_and_handle_replacement(self, primitive: Primitive, bake_transform: np.ndarray):
        """ Internal method to call _propagate_bake and handle potential primitive replacement. """
        original_id = primitive.internal_id
        original_parent_id = primitive.parent_primitive_id
        original_children_ids = list(primitive._child_primitive_ids) # Copy before modification

        replacement_primitives = primitive._propagate_bake(bake_transform)

        if replacement_primitives:
            logger.info(f"Replacing primitive '{primitive.user_identifier}' ({original_id}) with {len(replacement_primitives)} new primitive(s) due to baking.")
            # Unregister the original primitive *without* messing with children yet
            # (children were handled/reparented by the replacement logic if needed?)
            # No, _propagate_bake stops if self is replaced. Children links are lost on original.
            # We need to link original children to the *first* replacement primitive.

            # Simple strategy: Link all original children to the first replacement primitive.
            # More complex strategies could distribute children or duplicate them.
            first_replacement = replacement_primitives[0]
            new_parent_id_for_children = first_replacement.internal_id

            # Unregister original primitive - use internal method carefully
            registry = self._primitives
            entity = registry.pop(original_id, None)
            if entity:
                # Clean up registry, layer, groups
                 if entity.user_identifier and entity.user_identifier in self._identifier_registry:
                     if self._identifier_registry[entity.user_identifier] == original_id: del self._identifier_registry[entity.user_identifier]
                 layer = self.get_layer(entity.layer_id)
                 if layer: layer.primitive_ids.discard(original_id)
                 self._remove_primitive_from_all_groups(entity)
                 # Detach from original parent
                 if original_parent_id:
                     parent = self.get_primitive(original_parent_id)
                     if parent: parent._child_primitive_ids.discard(original_id)

            # Register replacement primitive(s)
            for i, new_prim in enumerate(replacement_primitives):
                # Ensure parent link is set (might be None if original was top-level)
                new_prim.parent_primitive_id = original_parent_id
                # Register the new primitive - this sets project ref, adds to layer/groups/parent
                self.add_primitive(new_prim)

                # Link original children to the first replacement
                if i == 0:
                     for child_id in original_children_ids:
                         child = self.get_primitive(child_id)
                         if child:
                              self.set_primitive_parent(child, new_prim.internal_id)


    # --- Transformation Helper Functions ---

    def translate_primitive(self, primitive_identifier: Identifiable, dx: float, dy: float, bake: bool = False) -> bool:
        """Translates a primitive globally."""
        matrix = translation_matrix(dx, dy)
        return self.transform_primitive(primitive_identifier, matrix, transform_type='global', bake=bake)

    def rotate_primitive_deg(self, primitive_identifier: Identifiable, angle_deg: float,
                             cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Rotates a primitive globally around a center point (defaults to its geometric center)."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False # Error logged by transform_primitive

        center_x, center_y = cx, cy
        if center_x is None or center_y is None:
            try:
                # Use center of current transformed shape for rotation axis
                center_x, center_y = primitive.get_geometric_center(use_original=False)
            except Exception as e:
                logger.error(f"Could not calculate geometric center for rotation of '{primitive.user_identifier}': {e}. Rotation failed.")
                return False

        matrix = rotation_matrix_deg(angle_deg, center_x, center_y)
        return self.transform_primitive(primitive_identifier, matrix, transform_type='global', bake=bake)

    def scale_primitive(self, primitive_identifier: Identifiable, sx: float, sy: Optional[float] = None,
                          cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Scales a primitive globally around a center point (defaults to its geometric center)."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False

        center_x, center_y = cx, cy
        if center_x is None or center_y is None:
            try:
                center_x, center_y = primitive.get_geometric_center(use_original=False)
            except Exception as e:
                logger.error(f"Could not calculate geometric center for scaling '{primitive.user_identifier}': {e}. Scaling failed.")
                return False

        sy_val = sy if sy is not None else sx
        matrix = scaling_matrix(sx, sy_val, center_x, center_y)
        return self.transform_primitive(primitive_identifier, matrix, transform_type='global', bake=bake)

    def mirror_primitive(self, primitive_identifier: Identifiable, axis: str = 'x',
                           cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Mirrors a primitive globally across a line (defaults to axis through its geometric center)."""
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False

        center_x, center_y = cx, cy
        # Determine default mirror line coordinate based on axis
        if axis.lower() == 'x': # Mirror across horizontal line y=center_y
             if center_y is None: center_y = primitive.get_geometric_center(use_original=False)[1]
             matrix = mirror_x_matrix(cy=center_y)
        elif axis.lower() == 'y': # Mirror across vertical line x=center_x
             if center_x is None: center_x = primitive.get_geometric_center(use_original=False)[0]
             matrix = mirror_y_matrix(cx=center_x)
        else:
             logger.error("Invalid mirror axis. Use 'x' or 'y'.")
             return False

        return self.transform_primitive(primitive_identifier, matrix, transform_type='global', bake=bake)


    def align_primitive(self, primitive_identifier: Identifiable,
                        align_point: str, # e.g., 'center', 'lower_left', 'upper_right'
                        target_coord: Tuple[float, float],
                        use_original_bbox: bool = False,
                        bake: bool = False) -> bool:
        """
        Aligns a primitive's bounding box point to a target coordinate using global translation.

        Args:
            primitive_identifier: The primitive to align.
            align_point: Name of the point on the bbox to align ('center',
                         'lower_left', 'lower_center', 'lower_right',
                         'middle_left', 'middle_right',
                         'upper_left', 'upper_center', 'upper_right').
            target_coord: The (x, y) coordinate to move the alignment point to.
            use_original_bbox: If True, align based on the untransformed geometry.
                               If False (default), align based on the current transformed geometry.
            bake: If True, bake the resulting translation into the geometry.
        """
        primitive = self.get_primitive(primitive_identifier)
        if not primitive: return False

        try:
            if use_original_bbox:
                bbox = primitive.get_original_bounding_box()
            else:
                bbox = primitive.get_bounding_box()

            if not bbox.is_valid():
                 logger.error(f"Cannot align '{primitive.user_identifier}': Bounding box is invalid.")
                 return False

            # Determine current coordinates of the alignment point
            current_x, current_y = 0.0, 0.0
            lc = (bbox.min_x + bbox.max_x) / 2 # center x
            mc = (bbox.min_y + bbox.max_y) / 2 # center y

            align_map = {
                'lower_left': (bbox.min_x, bbox.min_y), 'll': (bbox.min_x, bbox.min_y),
                'lower_center': (lc, bbox.min_y), 'lc': (lc, bbox.min_y),
                'lower_right': (bbox.max_x, bbox.min_y), 'lr': (bbox.max_x, bbox.min_y),
                'middle_left': (bbox.min_x, mc), 'ml': (bbox.min_x, mc),
                'center': (lc, mc), 'c': (lc, mc),
                'middle_right': (bbox.max_x, mc), 'mr': (bbox.max_x, mc),
                'upper_left': (bbox.min_x, bbox.max_y), 'ul': (bbox.min_x, bbox.max_y),
                'upper_center': (lc, bbox.max_y), 'uc': (lc, bbox.max_y),
                'upper_right': (bbox.max_x, bbox.max_y), 'ur': (bbox.max_x, bbox.max_y),
            }

            align_key = align_point.lower().replace("_", "")
            if align_key in align_map:
                 current_x, current_y = align_map[align_key]
            else:
                 logger.error(f"Invalid alignment point specified: '{align_point}'. Use one of {list(align_map.keys())}.")
                 return False

            # Calculate required translation
            dx = target_coord[0] - current_x
            dy = target_coord[1] - current_y

            if np.isclose(dx, 0) and np.isclose(dy, 0):
                 logger.debug(f"Primitive '{primitive.user_identifier}' already aligned. No translation needed.")
                 return True

            # Apply the translation globally
            return self.translate_primitive(primitive_identifier, dx, dy, bake=bake)

        except Exception as e:
            logger.error(f"Error during alignment of '{primitive.user_identifier}': {e}", exc_info=True)
            return False

    # --- Copying ---
    def copy_primitive(self, primitive_identifier: Identifiable,
                         new_identifier_suffix: str = "_copy") -> Optional[Primitive]:
        """
        Creates a deep copy of a primitive and its descendants, assigning new IDs
        and registering them with the project.

        Args:
            primitive_identifier: Identifier of the primitive to copy.
            new_identifier_suffix: Suffix to append to the user identifiers of copied items.

        Returns:
            The top-level copied primitive, or None if the original was not found.
        """
        original_primitive = self.get_primitive(primitive_identifier)
        if not original_primitive:
            logger.error(f"Primitive '{primitive_identifier}' not found for copying.")
            return None

        # Dictionary to track old_id -> new_id mapping during recursion
        new_id_map: Dict[uuid.UUID, uuid.UUID] = {}

        try:
            # Start recursive copy using the primitive's own method
            copied_primitive_top = original_primitive.copy_recursive(self, new_id_map, original_primitive.parent_primitive_id)

            # Register the copied primitives (recursively)
            primitives_to_register = [self.get_primitive(new_id) for new_id in new_id_map.values() if self.get_primitive(new_id) is None] # Find copied items not yet known by project? This seems wrong.
            # Let's iterate the map and register each one, copy_recursive creates the objects.
            all_copied_primitives : List[Primitive] = []
            for old_id, new_id in new_id_map.items():
                # Need to retrieve the *actual* copied object. The copy_recursive should return the top one.
                # How to get the others? Maybe copy_recursive should return the whole map of new objects?
                # Alternative: traverse the copied structure starting from copied_primitive_top.
                # Let's try traversal.
                pass # Need a way to get all copied objects.

            # Simpler: Assume copy_recursive created the objects. Find them by new ID and register.
            # We need to get the objects associated with new_id_map.values()
            # This requires searching or a temporary holding structure.

            # Revised approach: copy_recursive returns the top node.
            # The project iterates through the new_id_map and registers each object found
            # via the map values (new IDs), setting project refs etc.

            registered_copies = []
            for old_id, new_id in new_id_map.items():
                # How to get the object corresponding to new_id?
                # Assume it exists in memory but isn't registered yet.
                # This is tricky without passing the objects back.

                # Let's modify copy_recursive to populate a dict: new_id -> new_object
                new_objects_map : Dict[uuid.UUID, Primitive] = {}

                def _copy_recursive_populate(prim: Primitive, proj: 'CamBamProject',
                                            id_map: Dict[uuid.UUID, uuid.UUID],
                                            obj_map: Dict[uuid.UUID, Primitive],
                                            new_parent_id: Optional[uuid.UUID]) -> Primitive:
                    if prim.internal_id in id_map: return obj_map[id_map[prim.internal_id]] # Already processed

                    new_prim = deepcopy(prim)
                    new_prim.internal_id = uuid.uuid4()
                    id_map[prim.internal_id] = new_prim.internal_id
                    obj_map[new_prim.internal_id] = new_prim # Store object by NEW id

                    new_prim.user_identifier = f"{prim.user_identifier}{new_identifier_suffix}" # Apply suffix
                    new_prim.parent_primitive_id = new_parent_id
                    new_prim._child_primitive_ids = set()
                    new_prim._project_ref = None

                    orig_child_ids = list(prim._child_primitive_ids)
                    for child_id in orig_child_ids:
                        orig_child = proj.get_primitive(child_id)
                        if orig_child:
                            copied_child = _copy_recursive_populate(orig_child, proj, id_map, obj_map, new_prim.internal_id)
                            new_prim._child_primitive_ids.add(copied_child.internal_id)
                    return new_prim

                # Run the revised copy function
                copied_primitive_top = _copy_recursive_populate(original_primitive, self, new_id_map, new_objects_map, original_primitive.parent_primitive_id)

                # Now register all collected objects
                for new_id, new_obj in new_objects_map.items():
                    # Check for identifier conflicts before registering
                    if new_obj.user_identifier in self._identifier_registry:
                        conflict_id = self._identifier_registry[new_obj.user_identifier]
                        if conflict_id != new_obj.internal_id: # Ensure it's not just re-registering self if map was reused
                             new_obj.user_identifier = f"{new_obj.user_identifier}_{new_obj.internal_id.hex[:4]}"
                             logger.warning(f"Copied primitive identifier '{new_obj.user_identifier}' conflicted. Renamed.")

                    self.add_primitive(new_obj) # Registers and links layers, groups, parent/child
                    registered_copies.append(new_obj)

            logger.info(f"Copied primitive '{original_primitive.user_identifier}' and {len(registered_copies)-1} children as '{copied_primitive_top.user_identifier}'.")
            return copied_primitive_top

        except Exception as e:
            logger.error(f"Error during recursive copy of '{original_primitive.user_identifier}': {e}", exc_info=True)
            # Clean up any partially registered copies? Difficult state to revert.
            return None


    # --- Bounding Box ---
    def get_bounding_box(self, include_invisible: bool = False) -> BoundingBox:
        """Calculates the overall bounding box of all primitives in the project."""
        overall_bbox = BoundingBox()
        for primitive in self.list_primitives():
            try:
                # Check visibility if required
                if not include_invisible:
                     layer = self.get_layer(primitive.layer_id)
                     if not layer or not layer.visible:
                         continue # Skip primitives on invisible layers

                prim_bbox = primitive.get_bounding_box()
                if prim_bbox.is_valid():
                    overall_bbox = overall_bbox.union(prim_bbox)
            except Exception as e:
                logger.error(f"Could not get bounding box for primitive '{primitive.user_identifier}': {e}", exc_info=True)
        return overall_bbox

    # --- Persistence (Pickle - preserves full state including parent/child) ---
    def save_state(self, file_path: str) -> None:
        """Saves the current project state to a pickle file."""
        logger.info(f"Saving project state to {file_path}...")
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.isdir(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            # Before pickling, clear transient refs like _project_ref?
            # The __getstate__ in Primitive handles this.
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            logger.info(f"Project state saved successfully.")
        except Exception as e:
            logger.error(f"Error saving project state to {file_path}: {e}", exc_info=True)
            raise CamBamError(f"Failed to save state: {e}")

    @staticmethod
    def load_state(file_path: str) -> 'CamBamProject':
        """Loads a project state from a pickle file."""
        logger.info(f"Loading project state from {file_path}...")
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Pickle file not found: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                loaded_project = pickle.load(f)
            if not isinstance(loaded_project, CamBamProject):
                raise TypeError("Loaded object is not a CamBamProject instance.")

            # Post-load restoration: Re-establish weakrefs and potentially validate links
            loaded_project._restore_transient_state()
            logger.info(f"Project state loaded successfully: '{loaded_project.project_name}'")
            return loaded_project
        except Exception as e:
            logger.error(f"Error loading project state from {file_path}: {e}", exc_info=True)
            raise CamBamError(f"Failed to load state: {e}")

    def _restore_transient_state(self):
        """ Internal method to restore state after unpickling. """
        logger.debug("Restoring transient state after loading...")
        # Rebuild identifier registry (might be slightly redundant if pickled, but safer)
        self._identifier_registry = {}
        all_entities = list(self._layers.values()) + list(self._parts.values()) + \
                       list(self._primitives.values()) + list(self._mops.values())
        for entity in all_entities:
             if entity.user_identifier:
                  if entity.user_identifier in self._identifier_registry:
                      logger.warning(f"Duplicate user identifier '{entity.user_identifier}' detected during state restoration.")
                  else:
                      self._identifier_registry[entity.user_identifier] = entity.internal_id

        # Restore primitive project references and validate links
        all_child_links = set()
        primitives_to_remove = []
        for prim_id, primitive in self._primitives.items():
             primitive._project_ref = weakref.ref(self)
             # Ensure transform is numpy array
             if not isinstance(primitive.effective_transform, np.ndarray):
                  primitive.effective_transform = identity_matrix()

             # Validate parent link
             if primitive.parent_primitive_id:
                 parent = self._primitives.get(primitive.parent_primitive_id)
                 if parent:
                     parent._child_primitive_ids.add(prim_id) # Ensure child link exists on parent
                     all_child_links.add(prim_id)
                 else:
                      logger.warning(f"Primitive '{primitive.user_identifier}' has orphaned parent ID {primitive.parent_primitive_id}. Clearing.")
                      primitive.parent_primitive_id = None

             # Clear and rebuild child links based on validated parent links later?
             # For now, just ensure parent's set contains the child. We trust pickled child sets mostly.
             # Validate children links
             invalid_child_ids = set()
             for child_id in primitive._child_primitive_ids:
                  if child_id not in self._primitives:
                       logger.warning(f"Primitive '{primitive.user_identifier}' has invalid child ID {child_id}. Removing link.")
                       invalid_child_ids.add(child_id)
                  else:
                       all_child_links.add(child_id) # Track valid children
             primitive._child_primitive_ids -= invalid_child_ids


        # Validate layer primitive IDs
        for layer in self._layers.values():
             invalid_prim_ids = set()
             for prim_id in layer.primitive_ids:
                  if prim_id not in self._primitives:
                       invalid_prim_ids.add(prim_id)
             if invalid_prim_ids:
                 logger.warning(f"Layer '{layer.user_identifier}' contains invalid primitive IDs: {invalid_prim_ids}. Removing.")
                 layer.primitive_ids -= invalid_prim_ids

        # Validate part MOP IDs
        for part in self._parts.values():
             invalid_mop_ids = set()
             for mop_id in part.mop_ids:
                 if mop_id not in self._mops:
                      invalid_mop_ids.add(mop_id)
             if invalid_mop_ids:
                  logger.warning(f"Part '{part.user_identifier}' contains invalid MOP IDs: {invalid_mop_ids}. Removing.")
                  part.mop_ids = [mid for mid in part.mop_ids if mid not in invalid_mop_ids] # Rebuild list

        # Validate MOP Part ID and pid_source
        for mop in self._mops.values():
             if mop.part_id not in self._parts:
                  logger.warning(f"MOP '{mop.user_identifier}' has invalid Part ID {mop.part_id}. MOP is orphaned.")
             if isinstance(mop.pid_source, list):
                 invalid_pids = set()
                 for pid in mop.pid_source:
                      if pid not in self._primitives:
                          invalid_pids.add(pid)
                 if invalid_pids:
                      logger.warning(f"MOP '{mop.user_identifier}' references invalid primitive IDs: {invalid_pids}. Removing.")
                      mop.pid_source = [pid for pid in mop.pid_source if pid not in invalid_pids]
             elif isinstance(mop.pid_source, str): # Group name
                  # Group validity checked dynamically by resolver
                  pass


        logger.debug("Transient state restoration finished.")


    # --- XML Saving/Loading (Delegation) ---

    def save_to_xml(self, file_path: str, pretty_print: bool = True):
        """ Saves the project to a CamBam (.cb) XML file. """
        from cambam_writer import CamBamWriter # Local import to avoid circular dependency
        writer = CamBamWriter(self)
        writer.save(file_path, pretty_print)

    @staticmethod
    def load_from_xml(file_path: str) -> 'CamBamProject':
        """ Loads a project from a CamBam (.cb) XML file. """
        from cambam_reader import CamBamReader # Local import
        reader = CamBamReader()
        return reader.load(file_path)