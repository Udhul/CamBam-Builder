"""Transactional primitive-tree copy and transfer helpers."""

from copy import deepcopy
import uuid
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from .cambam_entities import CamBamEntity, Layer, Mop, Part, Primitive

if TYPE_CHECKING:  # pragma: no cover
    from .cambam_project import CamBamProject, Identifiable


_REGISTRY_NAMES = ("_primitives", "_layers", "_parts", "_mops")


def _snapshot(project: "CamBamProject") -> Dict[str, object]:
    """Copy project containers while retaining existing entity objects."""
    return {
        "_primitives": project._primitives.copy(),
        "_layers": project._layers.copy(),
        "_parts": project._parts.copy(),
        "_mops": project._mops.copy(),
        "_layer_order": list(project._layer_order),
        "_part_order": list(project._part_order),
        "_mop_order_in_part": {k: list(v) for k, v in project._mop_order_in_part.items()},
        "_primitive_layer_assignment": project._primitive_layer_assignment.copy(),
        "_layer_primitive_membership": {k: set(v) for k, v in project._layer_primitive_membership.items()},
        "_primitive_parent_link": project._primitive_parent_link.copy(),
        "_primitive_children_link": {k: set(v) for k, v in project._primitive_children_link.items()},
        "_mop_part_assignment": project._mop_part_assignment.copy(),
        "_mop_targets": project._mop_targets.copy(),
        "_primitive_groups": {k: set(v) for k, v in project._primitive_groups.items()},
        "_primitive_group_membership": {k: set(v) for k, v in project._primitive_group_membership.items()},
        "_identifier_registry": project._identifier_registry.copy(),
    }


def _publish(project: "CamBamProject", state: Dict[str, object]) -> None:
    for name, value in state.items():
        setattr(project, name, value)


def _valid_affine(matrix: object) -> bool:
    try:
        array = np.asarray(matrix)
        if np.iscomplexobj(array) or array.shape != (3, 3):
            return False
        array = np.asarray(array, dtype=float)
        return bool(np.isfinite(array).all() and np.array_equal(array[2], [0.0, 0.0, 1.0]))
    except (TypeError, ValueError, OverflowError):
        return False


def _world_matrix(project: "CamBamProject", primitive_id: uuid.UUID) -> np.ndarray:
    chain: List[Primitive] = []
    seen: Set[uuid.UUID] = set()
    current = primitive_id
    while current is not None:
        if current in seen:
            raise ValueError("Primitive parent cycle detected")
        seen.add(current)
        primitive = project._primitives.get(current)
        if primitive is None:
            raise ValueError("Primitive parent link points to a missing primitive")
        if not _valid_affine(primitive.effective_transform):
            raise ValueError("Primitive transform must be a finite affine 3x3 matrix")
        chain.append(primitive)
        current = project._primitive_parent_link.get(current)
    result = np.identity(3, dtype=float)
    for primitive in reversed(chain):
        result = result @ np.asarray(primitive.effective_transform, dtype=float)
    if not _valid_affine(result):
        raise ValueError("Primitive world transform must be a finite affine 3x3 matrix")
    return result


def _descendants(project: "CamBamProject", root_id: uuid.UUID) -> List[uuid.UUID]:
    """Return a deterministic complete descendant closure, rejecting cycles."""
    children: Dict[uuid.UUID, List[uuid.UUID]] = {}
    for child_id, parent_id in project._primitive_parent_link.items():
        if child_id not in project._primitives or parent_id not in project._primitives:
            raise ValueError("Primitive parent link points to a missing primitive")
        children.setdefault(parent_id, []).append(child_id)
    for values in children.values():
        values.sort()

    result: List[uuid.UUID] = []
    active: Set[uuid.UUID] = set()
    finished: Set[uuid.UUID] = set()

    def visit(node: uuid.UUID) -> None:
        if node in active:
            raise ValueError("Primitive parent cycle detected")
        if node in finished:
            return
        active.add(node)
        result.append(node)
        for child in children.get(node, ()):
            visit(child)
        active.remove(node)
        finished.add(node)

    visit(root_id)
    return result


def _all_entity_ids(state: Dict[str, object]) -> Set[uuid.UUID]:
    result: Set[uuid.UUID] = set()
    for name in _REGISTRY_NAMES:
        result.update(getattr(state[name], "keys")())  # type: ignore[union-attr]
    return result


def _entity_name(entity: CamBamEntity) -> str:
    name = entity.user_identifier
    if not isinstance(name, str) or not name:
        raise ValueError("Included entities must have nonempty user identifiers")
    return name


def _mapping(mapping: Optional[Dict[str, str]], allowed: Set[str], kind: str) -> Dict[str, str]:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise TypeError(f"{kind}_map must be a dictionary")
    result: Dict[str, str] = {}
    values: Set[str] = set()
    for key, value in mapping.items():
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"Unknown {kind}_map key: {key!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{kind}_map values must be nonempty strings")
        if value in values:
            raise ValueError(f"Duplicate mapped {kind} name: {value!r}")
        values.add(value)
        result[key] = value
    return result


def _fresh_id(used: Set[uuid.UUID]) -> uuid.UUID:
    candidate = uuid.uuid4()
    while candidate in used:
        candidate = uuid.uuid4()
    used.add(candidate)
    return candidate


def _copy_tree(source: "CamBamProject", root: "Identifiable", target: "CamBamProject",
               *, preserve_ids: bool, identifier_map: Optional[Dict[str, str]],
               group_map: Optional[Dict[str, str]], include_mops: bool,
               remove_source: bool) -> Dict[uuid.UUID, uuid.UUID]:
    from .cambam_project import CamBamProject
    if not isinstance(source, CamBamProject) or not isinstance(target, CamBamProject):
        raise TypeError("source and target_project must be CamBamProject instances")
    if type(preserve_ids) is not bool or type(include_mops) is not bool:
        raise TypeError("preserve_ids and include_mops must be booleans")
    if remove_source and source is target:
        raise ValueError("Same-project transfer is not supported")
    if source is target and preserve_ids:
        raise ValueError("Same-project copying requires preserve_ids=False")

    root_id = source._resolve_identifier(root, Primitive)
    if root_id is None or root_id not in source._primitives:
        raise ValueError("root must resolve to a primitive in the source project")
    primitive_ids = _descendants(source, root_id)
    # Validate the root's external ancestors and every included local matrix.
    root_world = _world_matrix(source, root_id)
    for primitive_id in primitive_ids:
        # Validate every composed frame, including descendants whose finite
        # local matrices could still overflow during parent composition.
        _world_matrix(source, primitive_id)

    layer_ids: Set[uuid.UUID] = set()
    primitive_groups: Dict[uuid.UUID, Set[str]] = {}
    group_names: Set[str] = set()
    for primitive_id in primitive_ids:
        layer_id = source._primitive_layer_assignment.get(primitive_id)
        if layer_id is None or layer_id not in source._layers:
            raise ValueError("Every included primitive must have a valid layer")
        layer_ids.add(layer_id)
        names = set(source._primitive_group_membership.get(primitive_id, set()))
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Primitive group names must be nonempty strings")
        primitive_groups[primitive_id] = names
        group_names.update(names)

    source_names: Set[str] = set()
    entities: List[CamBamEntity] = []
    for primitive_id in primitive_ids:
        entities.append(source._primitives[primitive_id])
    ordered_layers = [uid for uid in source._layer_order if uid in layer_ids]
    if set(ordered_layers) != layer_ids:
        raise ValueError("Included layers must have valid source order entries")
    entities.extend(source._layers[uid] for uid in ordered_layers)

    mop_ids: List[uuid.UUID] = []
    part_ids: Set[uuid.UUID] = set()
    if include_mops:
        included_set = set(primitive_ids)
        for mop_id, mop in source._mops.items():
            if mop_id not in source._mop_targets:
                raise ValueError("MOP has no target selection")
            selection = source._mop_targets[mop_id]
            if isinstance(selection, str):
                resolved = set(source._primitive_groups.get(selection, set()))
            else:
                try:
                    resolved = set(selection)
                except (TypeError, ValueError):
                    raise ValueError("MOP target selection is invalid")
            if any(pid not in source._primitives for pid in resolved):
                raise ValueError("MOP selection targets a missing or non-primitive entity")
            if resolved & included_set and not resolved <= included_set:
                raise ValueError("Included MOP selection targets outside the primitive subtree")
            if not (resolved & included_set):
                continue
            part_id = source._mop_part_assignment.get(mop_id)
            if part_id is None or part_id not in source._parts:
                raise ValueError("Every included MOP must have a valid owning part")
            mop_ids.append(mop_id)
            part_ids.add(part_id)

        for mop_id in mop_ids:
            selection = source._mop_targets[mop_id]
            if isinstance(selection, str) and selection not in group_names:
                raise ValueError("Included live MOP group has no copied primitive membership")

        ordered_parts = [uid for uid in source._part_order if uid in part_ids]
        if set(ordered_parts) != part_ids:
            raise ValueError("Included parts must have valid source order entries")
        entities.extend(source._parts[uid] for uid in ordered_parts)
        ordered_mops: List[uuid.UUID] = []
        for part_id in ordered_parts:
            order = source._mop_order_in_part.get(part_id)
            if order is None:
                raise ValueError("Included part has no MOP order registry")
            for mop_id in order:
                if mop_id in mop_ids:
                    ordered_mops.append(mop_id)
        if set(ordered_mops) != set(mop_ids) or len(ordered_mops) != len(mop_ids):
            raise ValueError("Included MOPs must have valid source order entries")
        mop_ids = ordered_mops
        entities.extend(source._mops[uid] for uid in mop_ids)

    for entity in entities:
        name = _entity_name(entity)
        if name in source_names:
            raise ValueError(f"Duplicate source user identifier: {name!r}")
        source_names.add(name)
    id_map_names = _mapping(identifier_map, source_names, "identifier")
    group_names_map = _mapping(group_map, group_names, "group")
    mapped_group_names = [group_names_map.get(name, name) for name in group_names]
    if len(mapped_group_names) != len(set(mapped_group_names)):
        raise ValueError("Duplicate destination group names")

    destination = _snapshot(target)
    target_ids = _all_entity_ids(destination)
    target_names = set(destination["_identifier_registry"].keys())  # type: ignore[arg-type]
    for name in _REGISTRY_NAMES:
        target_names.update(entity.user_identifier for entity in destination[name].values())  # type: ignore[union-attr]
    for name in target_names:
        if not isinstance(name, str):
            raise ValueError("Destination identifier registry is invalid")
    for name in group_names:
        destination_name = group_names_map.get(name, name)
        if destination_name in target._primitive_groups:
            raise ValueError(f"Destination group name is already in use: {destination_name!r}")
    reserved_live_groups = {
        value for value in destination["_mop_targets"].values() if isinstance(value, str)  # type: ignore[union-attr]
    }
    for name in group_names:
        if group_names_map.get(name, name) in reserved_live_groups:
            raise ValueError("Destination group name is reserved by a live MOP selector")

    all_ids = target_ids | _all_entity_ids(_snapshot(source))
    source_id_values = [entity.internal_id for entity in entities]
    if len(source_id_values) != len(set(source_id_values)):
        raise ValueError("Included entity UUIDs collide across entity types")
    uuid_map: Dict[uuid.UUID, uuid.UUID] = {}
    for entity in entities:
        old_id = entity.internal_id
        new_id = old_id if preserve_ids else _fresh_id(all_ids)
        if new_id in target_ids:
            raise ValueError(f"Destination UUID collision: {new_id}")
        uuid_map[old_id] = new_id

    used_names = set(target_names)
    for entity in entities:
        new_name = id_map_names.get(entity.user_identifier, entity.user_identifier)
        if new_name in used_names:
            raise ValueError(f"Destination identifier collision: {new_name!r}")
        used_names.add(new_name)

    clones: Dict[uuid.UUID, CamBamEntity] = {}
    for entity in entities:
        clone = deepcopy(entity)
        clone.internal_id = uuid_map[entity.internal_id]
        clone.user_identifier = id_map_names.get(entity.user_identifier, entity.user_identifier)
        if isinstance(clone, Primitive):
            clone.groups = sorted(group_names_map.get(name, name) for name in primitive_groups[entity.internal_id])
            if entity.internal_id == root_id:
                clone.effective_transform = root_world.copy()
            else:
                clone.effective_transform = np.asarray(entity.effective_transform, dtype=float).copy()
            # Do this while still staging so an overridden hook cannot leave a
            # published target partially updated.
            clone.set_project_link(target)
        clones[entity.internal_id] = clone

    # Stage all relationship containers on independent copies.
    for entity in entities:
        new_id = uuid_map[entity.internal_id]
        if isinstance(entity, Primitive):
            destination["_primitives"][new_id] = clones[entity.internal_id]  # type: ignore[index]
            layer_id = source._primitive_layer_assignment[entity.internal_id]
            new_layer_id = uuid_map[layer_id]
            destination["_primitive_layer_assignment"][new_id] = new_layer_id  # type: ignore[index]
            destination["_layer_primitive_membership"].setdefault(new_layer_id, set()).add(new_id)  # type: ignore[index]
            for group_name in primitive_groups[entity.internal_id]:
                mapped_group = group_names_map.get(group_name, group_name)
                destination["_primitive_groups"].setdefault(mapped_group, set()).add(new_id)  # type: ignore[index]
                destination["_primitive_group_membership"].setdefault(new_id, set()).add(mapped_group)  # type: ignore[index]
            parent_id = source._primitive_parent_link.get(entity.internal_id)
            if entity.internal_id != root_id:
                if parent_id is None or parent_id not in uuid_map:
                    raise ValueError("Included primitive has a parent outside the copied subtree")
                destination["_primitive_parent_link"][new_id] = uuid_map[parent_id]  # type: ignore[index]
                destination["_primitive_children_link"].setdefault(uuid_map[parent_id], set()).add(new_id)  # type: ignore[index]
            destination["_primitive_children_link"].setdefault(new_id, set())  # type: ignore[index]
        elif isinstance(entity, Layer):
            destination["_layers"][new_id] = clones[entity.internal_id]  # type: ignore[index]
            destination["_layer_order"].append(new_id)  # type: ignore[union-attr]
            destination["_layer_primitive_membership"].setdefault(new_id, set())  # type: ignore[index]
        elif isinstance(entity, Part):
            destination["_parts"][new_id] = clones[entity.internal_id]  # type: ignore[index]
            destination["_part_order"].append(new_id)  # type: ignore[union-attr]
            destination["_mop_order_in_part"].setdefault(new_id, [])  # type: ignore[index]
        elif isinstance(entity, Mop):
            destination["_mops"][new_id] = clones[entity.internal_id]  # type: ignore[index]
            part_id = source._mop_part_assignment[entity.internal_id]
            new_part_id = uuid_map[part_id]
            destination["_mop_part_assignment"][new_id] = new_part_id  # type: ignore[index]
            destination["_mop_order_in_part"].setdefault(new_part_id, []).append(new_id)  # type: ignore[index]
            selection = source._mop_targets[entity.internal_id]
            if isinstance(selection, str):
                mapped_selection = group_names_map.get(selection, selection)
                destination["_mop_targets"][new_id] = mapped_selection  # type: ignore[index]
            else:
                destination["_mop_targets"][new_id] = frozenset(uuid_map[pid] for pid in selection)  # type: ignore[index]
        destination["_identifier_registry"][clones[entity.internal_id].user_identifier] = new_id  # type: ignore[index]

    if remove_source:
        source_state = _snapshot(source)
        source_primitives = {uid: source._primitives[uid] for uid in primitive_ids}
        copied_primitives = set(primitive_ids)
        copied_mops = set(mop_ids)
        for primitive_id in copied_primitives:
            source_state["_primitives"].pop(primitive_id, None)  # type: ignore[union-attr]
            layer_id = source_state["_primitive_layer_assignment"].pop(primitive_id, None)  # type: ignore[union-attr]
            if layer_id is not None:
                source_state["_layer_primitive_membership"].get(layer_id, set()).discard(primitive_id)  # type: ignore[union-attr]
            parent_id = source_state["_primitive_parent_link"].pop(primitive_id, None)  # type: ignore[union-attr]
            if parent_id is not None:
                source_state["_primitive_children_link"].get(parent_id, set()).discard(primitive_id)  # type: ignore[union-attr]
            children = source_state["_primitive_children_link"].pop(primitive_id, set())  # type: ignore[union-attr]
            for child_id in children:
                source_state["_primitive_parent_link"].pop(child_id, None)  # type: ignore[union-attr]
            for group_name in source_state["_primitive_group_membership"].pop(primitive_id, set()):  # type: ignore[union-attr]
                members = source_state["_primitive_groups"].get(group_name, set())  # type: ignore[union-attr]
                members.discard(primitive_id)
                if not members:
                    source_state["_primitive_groups"].pop(group_name, None)  # type: ignore[union-attr]
            source_state["_identifier_registry"].pop(source._primitives[primitive_id].user_identifier, None)  # type: ignore[union-attr]
        for mop_id in copied_mops:
            source_state["_mops"].pop(mop_id, None)  # type: ignore[union-attr]
            source_state["_mop_targets"].pop(mop_id, None)  # type: ignore[union-attr]
            part_id = source_state["_mop_part_assignment"].pop(mop_id, None)  # type: ignore[union-attr]
            if part_id is not None:
                source_state["_mop_order_in_part"].get(part_id, []).remove(mop_id)  # type: ignore[union-attr]
            source_state["_identifier_registry"].pop(source._mops[mop_id].user_identifier, None)  # type: ignore[union-attr]
        for mop_id, selection in list(source_state["_mop_targets"].items()):  # type: ignore[union-attr]
            if not isinstance(selection, str):
                source_state["_mop_targets"][mop_id] = frozenset(selection - copied_primitives)  # type: ignore[index,operator]

        # This is the final commit point; all validation and deepcopy work is done.
        _publish(target, destination)
        _publish(source, source_state)
        for primitive in source_primitives.values():
            primitive._project_ref = None
    else:
        _publish(target, destination)

    return uuid_map


def copy_primitive_tree(source: "CamBamProject", root: "Identifiable", target_project: "CamBamProject", *,
                        preserve_ids: bool = True, identifier_map: Optional[Dict[str, str]] = None,
                        group_map: Optional[Dict[str, str]] = None, include_mops: bool = True) -> Dict[uuid.UUID, uuid.UUID]:
    """Copy a primitive subtree and all required relationship entities."""
    return _copy_tree(source, root, target_project, preserve_ids=preserve_ids,
                      identifier_map=identifier_map, group_map=group_map,
                      include_mops=include_mops, remove_source=False)


def transfer_primitive_tree(source: "CamBamProject", root: "Identifiable", target_project: "CamBamProject", *,
                            preserve_ids: bool = True, identifier_map: Optional[Dict[str, str]] = None,
                            group_map: Optional[Dict[str, str]] = None, include_mops: bool = True) -> Dict[uuid.UUID, uuid.UUID]:
    """Transfer a primitive subtree transactionally, removing source entities after staging."""
    return _copy_tree(source, root, target_project, preserve_ids=preserve_ids,
                      identifier_map=identifier_map, group_map=group_map,
                      include_mops=include_mops, remove_source=True)
