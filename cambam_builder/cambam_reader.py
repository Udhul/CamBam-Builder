"""
cambam_reader.py

Provides functionality to read a CamBam XML file (.cb) and reconstruct a
CamBamProject instance, including its entities and their relationships.
Follows the specified reconstruction order to ensure dependencies are met.
"""

import xml.etree.ElementTree as ET
import logging
import uuid
import json
import os
from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Union, Any

import numpy as np # For matrix conversion

from .cambam_project import CamBamProject
from .cambam_entities import ( # Import concrete entity types
    Layer, Part, Mop, Primitive,
    Pline, Circle, Rect, Arc, Points, Text,
    ProfileMop, PocketMop, EngraveMop, DrillMop,
    MOP_XML_PATH_TO_FIELD,
)
from .cad_transformations import identity_matrix, from_cambam_matrix_str # For parsing matrix
from .region import Region, parse_region_geometry

logger = logging.getLogger(__name__)

# Mapping from CamBam MOP XML tag names to our Mop classes
MOP_TAG_TO_CLASS = {
    "profile": ProfileMop,
    "pocket": PocketMop,
    "engrave": EngraveMop,
    "drill": DrillMop,
    # Add mappings for other MOP types if implemented (e.g., "lathe", "script")
}

# Mapping from CamBam Primitive XML tag names to our Primitive classes
PRIMITIVE_TAG_TO_CLASS = {
    "pline": Pline,
    "circle": Circle,
    "rect": Rect,
    "arc": Arc,
    "points": Points,
    "text": Text,
    "Region": Region,
    # Add mappings for other primitive types ("surface", "region", etc.)
}

class CamBamReaderError(Exception):
    """Custom exception for errors during CamBam file reading."""
    pass

def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    """Safely parse boolean strings."""
    if value is None: return default
    return value.strip().lower() == 'true'

def _parse_float(value: Optional[str], default: float = 0.0) -> float:
    """Safely parse float strings."""
    if value is None: return default
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return default

def _parse_int(value: Optional[str], default: int = 0) -> int:
    """Safely parse integer strings."""
    if value is None: return default
    try:
        # Allow parsing floats then converting, as CamBam sometimes uses "0.0" for int fields
        return int(float(value.strip()))
    except (ValueError, TypeError):
        return default


def _read_mop_parameter(
    mop_elem: ET.Element,
    tag: str,
    parser,
    default: Any,
) -> Tuple[Any, Optional[str], Optional[str]]:
    """Read one CamBam MOP parameter without allowing bad metadata to abort import.

    The value is parsed for the in-memory model.  The state and original text are
    returned separately so entity encoders can retain CamBam's Default/Value
    distinction while still allowing later field edits to win.
    """
    element = mop_elem.find(tag)
    if element is None:
        return default, None, None
    raw = element.text
    state = element.get("state")
    if state not in ("Default", "Value"):
        state = None
    if raw is None or not raw.strip():
        return default, state, raw
    try:
        value = parser(raw, default)
    except (TypeError, ValueError, AttributeError):
        value = default
    return value, state, raw


def _read_nested_mop_parameter(
    parent: Optional[ET.Element],
    tag: str,
    parser,
    default: Any,
) -> Tuple[Any, Optional[str], Optional[str]]:
    """Variant of :func:`_read_mop_parameter` for nested MOP settings."""
    if parent is None:
        return default, None, None
    element = parent.find(tag)
    if element is None:
        return default, None, None
    raw = element.text
    state = element.get("state")
    if state not in ("Default", "Value"):
        state = None
    if raw is None or not raw.strip():
        return default, state, raw
    try:
        value = parser(raw, default)
    except (TypeError, ValueError, AttributeError):
        value = default
    return value, state, raw

def _parse_point_2d(value: Optional[str]) -> Optional[Tuple[float, float]]:
    """Safely parse 'x,y' or 'x,y,z' strings into (x, y)."""
    if value is None: return None
    try:
        parts = [float(p.strip()) for p in value.split(',')]
        if len(parts) >= 2:
            return parts[0], parts[1]
    except (ValueError, TypeError, AttributeError):
        pass
    return None

def _parse_point_3d(value: Optional[str]) -> Optional[Tuple[float, float, float]]:
    """Safely parse 'x,y,z' strings into (x, y, z)."""
    if value is None: return None
    try:
        parts = [float(p.strip()) for p in value.split(',')]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def read_cambam_file(file_path: str) -> Optional[CamBamProject]:
    """
    Reads a CamBam XML file (.cb) and reconstructs a CamBamProject object.

    Args:
        file_path: Path to the .cb file.

    Returns:
        A reconstructed CamBamProject instance, or None if reading fails.
    """
    if not os.path.exists(file_path):
        logger.error(f"CamBam file not found: {file_path}")
        return None

    logger.info(f"Attempting to read CamBam file: {file_path}")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file {file_path}: {e}", exc_info=True)
        return None
    except Exception as e:
         logger.error(f"Unexpected error opening or parsing {file_path}: {e}", exc_info=True)
         return None

    project_name = root.get("Name", os.path.basename(file_path))
    # TODO: Read project-level defaults if they exist in XML?
    project = CamBamProject(project_name)
    # Preserve native project context (styles and other unknown children) for
    # the writer; modeled settings remain owned by the project entity fields.
    project._xml_machining_options = deepcopy(root.find("MachiningOptions"))
    logger.info(f"Reconstructing project: {project.project_name}")

    # --- Temporary storage during reconstruction ---
    # Map XML integer ID back to Primitive UUID (filled during primitive parsing)
    xml_id_to_primitive_uuid: Dict[int, uuid.UUID] = {}
    # Store MOP info temporarily with XML IDs before linking primitives
    # MOP UUID -> List of primitive XML IDs it references
    mop_primitive_xml_id_refs: Dict[uuid.UUID, List[int]] = {}

    try:
        # --- Reconstruction Order ---

        # 1. Parts
        logger.debug("Reading Parts...")
        parts_node = root.find("parts")
        if parts_node is not None:
            for part_elem in parts_node.findall("part"):
                _reconstruct_part(project, part_elem)

        # 3. Layers
        logger.debug("Reading Layers...")
        layers_node = root.find("layers")
        if layers_node is None:
            raise CamBamReaderError("Unsupported drawing schema: missing layers container")
        if layers_node is not None:
            for layer_elem in layers_node:
                if layer_elem.tag != "layer" or not layer_elem.get("name"):
                    raise CamBamReaderError("Unsupported layer schema: expected layer with a name")
                _reconstruct_layer(project, layer_elem)

        # 4. Primitives (read structure, link to layers, store parent XML ID refs, populate xml_id_to_primitive_uuid)
        logger.debug("Reading Primitives...")
        # Store parent info temporarily: Child UUID -> Parent XML ID (or UUID if resolved from Tag)
        primitive_parent_ref: Dict[uuid.UUID, Union[int, uuid.UUID, str]] = {}
        if layers_node is not None:
            for layer_elem in layers_node.findall("layer"):
                layer_uuid = project._resolve_identifier(layer_elem.get("name"), Layer)
                if not layer_uuid:
                    logger.warning(f"Skipping primitives for layer '{layer_elem.get('name')}' as layer was not reconstructed.")
                    continue
                objects_node = layer_elem.find("objects")
                if objects_node is not None:
                     for prim_elem in objects_node: # Iterate over actual primitive tags (<pline>, <circle> etc)
                         prim_type_tag = _primitive_type(prim_elem)
                         if prim_type_tag in PRIMITIVE_TAG_TO_CLASS:
                            _reconstruct_primitive(project, prim_elem, layer_uuid,
                                                   xml_id_to_primitive_uuid, primitive_parent_ref)
                         else:
                            raise CamBamReaderError(f"Unsupported primitive type '{prim_type_tag}' in layer '{layer_elem.get('name')}'")

        # 5. Link Relationships using stored temporary data
        logger.debug("Linking relationships...")
        # 5a. Link Primitive Parents
        # XML matrices are world poses. Snapshot before changing any local matrix
        # so reconstruction is independent of XML and parent-link iteration order.
        world_transforms = {
            prim_uuid: primitive.effective_transform.copy()
            for prim_uuid, primitive in project._primitives.items()
        }
        world_z_offsets = {uid: p.local_z_offset for uid, p in project._primitives.items()}
        for child_uuid, parent_ref in primitive_parent_ref.items():
            parent_uuid: Optional[uuid.UUID] = None
            if isinstance(parent_ref, uuid.UUID): # Resolved directly from Tag's internal_id
                parent_uuid = parent_ref
            elif isinstance(parent_ref, int): # XML ID needs mapping
                parent_uuid = xml_id_to_primitive_uuid.get(parent_ref)
                if parent_uuid is None:
                     logger.warning(f"Could not link child primitive {child_uuid}: Parent XML ID '{parent_ref}' not found in reconstructed primitives.")
            elif isinstance(parent_ref, str): # UUID string from Tag
                 try: parent_uuid = uuid.UUID(parent_ref)
                 except ValueError: logger.warning(f"Invalid parent UUID string '{parent_ref}' in Tag for child {child_uuid}.")
            else:
                 # Should not happen if tag parsing is correct
                 logger.warning(f"Invalid parent reference type '{type(parent_ref)}' for child {child_uuid}.")

            if parent_uuid:
                 # Check parent actually exists in project before linking
                 if parent_uuid in project._primitives:
                     if parent_uuid == child_uuid:
                         # Preserve the existing rejected-link behavior and world pose.
                         project.link_primitive_parent(child_uuid, parent_uuid)
                         continue
                     try:
                         local_transform = np.linalg.solve(
                             world_transforms[parent_uuid], world_transforms[child_uuid]
                         )
                     except np.linalg.LinAlgError as e:
                         raise CamBamReaderError(
                             f"Cannot reconstruct child {child_uuid}: parent {parent_uuid} "
                             "has a singular world transform."
                         ) from e
                     if project.link_primitive_parent(child_uuid, parent_uuid):
                         local_z = world_z_offsets[child_uuid] - world_z_offsets[parent_uuid]
                         if not np.isfinite(local_z):
                             raise CamBamReaderError(f"Parent-relative Z offset overflows for primitive {child_uuid}")
                         project._primitives[child_uuid].effective_transform = local_transform
                         project._primitives[child_uuid].local_z_offset = local_z
                 else:
                     logger.warning(f"Could not link child primitive {child_uuid}: Resolved parent UUID {parent_uuid} not found in project primitives registry.")

        # Reconstruct MOPs after other entities so identity collisions are detectable
        logger.debug("Reading MOPs...")
        if parts_node is not None:
            for part_elem in parts_node.findall("part"):
                part_uuid = project._resolve_identifier(part_elem.get("Name"), Part) # Assume Name is unique ID here
                if not part_uuid:
                     logger.warning(f"Skipping MOPs for part '{part_elem.get('Name')}' as part was not reconstructed.")
                     continue
                mops_node = part_elem.find("machineops")
                if mops_node is not None:
                    for mop_elem in mops_node: # Iterate over actual MOP tags (<profile>, <pocket> etc)
                         mop_type_tag = mop_elem.tag
                         if mop_type_tag in MOP_TAG_TO_CLASS:
                             _reconstruct_mop(project, mop_elem, part_uuid, mop_primitive_xml_id_refs)
                         else:
                              logger.warning(f"Unsupported MOP type tag '{mop_type_tag}' encountered in part '{part_elem.get('Name')}'. Skipping.")

        # 5b. Link MOPs to Primitives through the project's target registry.
        for mop_uuid, xml_ids in mop_primitive_xml_id_refs.items():
            mop = project.get_mop(mop_uuid)
            if not mop: continue # Should not happen

            resolved_primitive_uuids: List[uuid.UUID] = []
            all_resolved = True
            for xml_id in xml_ids:
                prim_uuid = xml_id_to_primitive_uuid.get(xml_id)
                if prim_uuid:
                    # Check primitive actually exists in registry
                    if prim_uuid in project._primitives:
                        resolved_primitive_uuids.append(prim_uuid)
                    else:
                        logger.warning(f"MOP '{mop.name}' references primitive XML ID {xml_id}, which resolved to UUID {prim_uuid}, but primitive not in registry.")
                        all_resolved = False
                else:
                    logger.warning(f"MOP '{mop.name}' references primitive XML ID {xml_id}, which could not be mapped back to a UUID.")
                    all_resolved = False

            # Native CamBam primitive references are authoritative.  Framework
            # identity metadata never supplies or overrides this relationship.
            try:
                project.set_mop_targets(mop.internal_id, resolved_primitive_uuids)
            except (TypeError, ValueError) as exc:
                raise CamBamReaderError(
                    f"Could not assign targets for MOP '{mop.name}': {exc}"
                ) from exc
            if not all_resolved:
                logger.warning(f"MOP '{mop.name}' primitive target list was partially resolved.")
            elif not resolved_primitive_uuids and xml_ids:
                logger.warning(f"MOP '{mop.name}' primitive target list {xml_ids} resolved to empty UUID list.")

        logger.info(f"Project '{project.project_name}' reconstruction complete. "
                    f"Layers: {len(project.list_layers())}, "
                    f"Parts: {len(project.list_parts())}, "
                    f"Primitives: {len(project.list_primitives())}, "
                    f"MOPs: {len(project.list_mops())}")
        return project

    except CamBamReaderError as e:
        logger.error(f"Failed to reconstruct project from {file_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during project reconstruction: {e}", exc_info=True)
        return None


def _reconstruct_layer(project: CamBamProject, layer_elem: ET.Element):
    """Parses a <layer> element and adds/updates the layer in the project."""
    name = layer_elem.get("name")
    if not name:
        logger.warning("Skipping layer with missing 'name' attribute.")
        return

    # Use project's add_layer which handles creation or update and registration
    project.add_layer(
        identifier=name,
        color=layer_elem.get("color", "Green"),
        alpha=_parse_float(layer_elem.get("alpha"), 1.0),
        pen_width=_parse_float(layer_elem.get("pen"), 1.0),
        visible=_parse_bool(layer_elem.get("visible"), True),
        locked=_parse_bool(layer_elem.get("locked"), False)
        # Order is determined by XML sequence, add_layer doesn't reorder existing
    )

def _reconstruct_part(project: CamBamProject, part_elem: ET.Element):
    """Parses a <part> element and adds/updates the part in the project."""
    name = part_elem.get("Name")
    if not name:
        logger.warning("Skipping part with missing 'Name' attribute.")
        return

    # Parse attributes
    enabled = _parse_bool(part_elem.get("Enabled"), True)

    # Parse Stock (requires more detail)
    stock_thickness = 12.5 # Default
    stock_width = 100.0    # Default
    stock_height = 100.0   # Default
    stock_material = "Default"
    stock_color = "210,180,140" # Default
    stock_node = part_elem.find("Stock")
    if stock_node is not None:
        stock_material = stock_node.findtext("Material", stock_material)
        stock_color = stock_node.findtext("Color", stock_color)
        pmin_str = stock_node.findtext("PMin")
        pmax_str = stock_node.findtext("PMax")
        pmin = _parse_point_3d(pmin_str)
        pmax = _parse_point_3d(pmax_str)
        if pmin and pmax:
            stock_width = abs(pmax[0] - pmin[0])
            stock_height = abs(pmax[1] - pmin[1])
            stock_thickness = abs(pmax[2] - pmin[2]) # Assumes surface at Z=0 or Z=thickness

    # Parse Machining Origin
    origin_str = part_elem.findtext("MachiningOrigin")
    machining_origin = _parse_point_2d(origin_str) or (0.0, 0.0)

    # Parse Defaults (if they exist in XML - check typical CamBam file structure)
    default_tool_dia = _parse_float(part_elem.findtext("ToolDiameter"), None) # Use None if not present
    # SpindleSpeed default seems not stored at Part level in standard XML?
    default_spindle_speed = None # Assume None

    # Use project's add_part
    part = project.add_part(
        identifier=name,
        enabled=enabled,
        stock_thickness=stock_thickness,
        stock_width=stock_width,
        stock_height=stock_height,
        stock_material=stock_material,
        stock_color=stock_color,
        machining_origin=machining_origin,
        default_tool_diameter=default_tool_dia,
        default_spindle_speed=default_spindle_speed
        # Order handled by XML sequence
    )
    if part is not None:
        part._xml_tool_diameter = deepcopy(part_elem.find("ToolDiameter"))
        part._xml_tool_diameter_value = part.default_tool_diameter
        part._xml_machining_parameters = tuple(
            deepcopy(child) for child in part_elem
            if child.tag not in {"Stock", "MachiningOrigin", "ToolDiameter", "machineops"}
        )


def _reconstruct_mop(project: CamBamProject, mop_elem: ET.Element, part_uuid: uuid.UUID,
                      mop_primitive_xml_id_refs: Dict[uuid.UUID, List[int]]):
    """Parses a MOP element (<profile>, <pocket>...) and adds it to the project."""
    mop_class = MOP_TAG_TO_CLASS.get(mop_elem.tag)
    if not mop_class: return # Should have been checked by caller

    # Display names are not registry keys. Legacy or malformed metadata gets
    # a fresh identity; complete valid metadata is preserved before registration.
    internal_id = uuid.uuid4()
    mop_identifier = str(internal_id)
    tag_text = mop_elem.findtext("Tag")
    if tag_text:
        try:
            metadata = json.loads(tag_text)
            if not isinstance(metadata, dict):
                raise ValueError("MOP Tag must be an object")
            candidate_id = metadata.get("user_id")
            candidate_uuid = metadata.get("internal_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("MOP user_id must be a nonempty string")
            if not isinstance(candidate_uuid, str):
                raise ValueError("MOP internal_id must be a UUID string")
            internal_id = uuid.UUID(candidate_uuid)
            mop_identifier = candidate_id
        except (ValueError, TypeError):
            logger.warning("Invalid MOP identity metadata; allocating a fresh identity.")
    if (project._get_entity_by_uuid(internal_id) is not None
            or mop_identifier in project._identifier_registry):
        raise CamBamReaderError(f"Conflicting MOP identity: {mop_identifier} ({internal_id})")

    # Native Default text is a cached value, not an evaluated CAM style value.
    parameter_states: Dict[str, str] = {}
    parameter_baseline: Dict[str, Any] = {}

    def parameter(tag, parser, default, optional_default=False):
        value, state, raw = _read_mop_parameter(mop_elem, tag, parser, default)
        field = MOP_XML_PATH_TO_FIELD.get((tag,), tag)
        if state:
            parameter_states[field] = state
        if mop_elem.find(tag) is not None:
            parameter_baseline[field] = value
        return value

    def nested_parameter(parent, tag, parser, default, optional_default=False):
        value, state, raw = _read_nested_mop_parameter(parent, tag, parser, default)
        key = f"{parent.tag}/{tag}" if parent is not None else tag
        field = MOP_XML_PATH_TO_FIELD.get(tuple(key.split("/")), key)
        if state:
            parameter_states[field] = state
        if parent is not None and parent.find(tag) is not None:
            parameter_baseline[field] = value
        return value

    parse_string = lambda value, default: value
    mop_name = mop_elem.findtext("Name", f"Unnamed_{mop_elem.tag}")
    enabled = _parse_bool(mop_elem.get("Enabled"), True)
    target_depth = parameter("TargetDepth", _parse_float, None, True)
    depth_increment = parameter("DepthIncrement", _parse_float, None, True)
    stock_surface = parameter("StockSurface", _parse_float, 0.0)
    roughing_clearance = parameter("RoughingClearance", _parse_float, 0.0)
    clearance_plane = parameter("ClearancePlane", _parse_float, 15.0)
    spindle_dir = parameter("SpindleDirection", parse_string, "CW")
    spindle_speed = parameter("SpindleSpeed", _parse_int, None, True)
    velocity_mode = parameter("VelocityMode", parse_string, "ExactStop")
    work_plane = parameter("WorkPlane", parse_string, "XY")
    optimisation_mode = parameter("OptimisationMode", parse_string, "Standard")
    tool_diameter = parameter("ToolDiameter", _parse_float, None, True)
    tool_number = parameter("ToolNumber", _parse_int, 0)
    tool_profile = parameter("ToolProfile", parse_string, "EndMill")
    plunge_feed = parameter("PlungeFeedrate", _parse_float, 1000.0)
    cut_feedrate = parameter("CutFeedrate", _parse_float, None, True)
    max_crossover = parameter("MaxCrossoverDistance", _parse_float, 0.7)
    custom_header = parameter("CustomMOPHeader", parse_string, "")
    custom_footer = parameter("CustomMOPFooter", parse_string, "")

    # Native primitive references are the only import target authority.
    primitive_xml_ids: List[int] = []
    primitive_container = mop_elem.find("primitive")
    if primitive_container is not None:
        for prim_ref in primitive_container.findall("prim"):
            xml_id = _parse_int(prim_ref.text, -1)
            if xml_id > 0:
                primitive_xml_ids.append(xml_id)

    # Parse all fields emitted by each supported MOP encoder. Missing fields use
    # dataclass defaults; malformed values are handled by the safe parsers.
    mop_specific_kwargs: Dict[str, Any] = {}
    if mop_class is ProfileMop:
        mop_specific_kwargs["stepover"] = parameter("StepOver", _parse_float, 0.4)
        mop_specific_kwargs["profile_side"] = parameter("InsideOutside", parse_string, "Inside")
        mop_specific_kwargs["milling_direction"] = parameter("MillingDirection", parse_string, "Conventional")
        mop_specific_kwargs["collision_detection"] = parameter("CollisionDetection", _parse_bool, True)
        mop_specific_kwargs["corner_overcut"] = parameter("CornerOvercut", _parse_bool, False)
        lead_in = mop_elem.find("LeadInMove")
        mop_specific_kwargs["lead_in_type"] = nested_parameter(lead_in, "LeadInType", parse_string, "Spiral")
        mop_specific_kwargs["lead_in_spiral_angle"] = nested_parameter(lead_in, "SpiralAngle", _parse_float, 30.0)
        mop_specific_kwargs["final_depth_increment"] = parameter("FinalDepthIncrement", _parse_float, 0.0, True)
        mop_specific_kwargs["cut_ordering"] = parameter("CutOrdering", parse_string, "DepthFirst")
        tabs = mop_elem.find("HoldingTabs")
        mop_specific_kwargs["tab_method"] = nested_parameter(tabs, "TabMethod", parse_string, "None")
        mop_specific_kwargs["tab_width"] = nested_parameter(tabs, "Width", _parse_float, 6.0)
        mop_specific_kwargs["tab_height"] = nested_parameter(tabs, "Height", _parse_float, 1.5)
        mop_specific_kwargs["tab_min_tabs"] = nested_parameter(tabs, "MinimumTabs", _parse_int, 3)
        mop_specific_kwargs["tab_max_tabs"] = nested_parameter(tabs, "MaximumTabs", _parse_int, 3)
        mop_specific_kwargs["tab_distance"] = nested_parameter(tabs, "TabDistance", _parse_float, 40.0)
        mop_specific_kwargs["tab_size_threshold"] = nested_parameter(tabs, "SizeThreshold", _parse_float, 4.0)
        mop_specific_kwargs["tab_use_leadins"] = nested_parameter(tabs, "UseLeadIns", _parse_bool, False)
        mop_specific_kwargs["tab_style"] = nested_parameter(tabs, "TabStyle", parse_string, "Square")
    elif mop_class is PocketMop:
        mop_specific_kwargs["stepover"] = parameter("StepOver", _parse_float, 0.4)
        mop_specific_kwargs["stepover_feedrate"] = parameter("StepoverFeedrate", parse_string, "Plunge Feedrate")
        mop_specific_kwargs["milling_direction"] = parameter("MillingDirection", parse_string, "Conventional")
        mop_specific_kwargs["collision_detection"] = parameter("CollisionDetection", _parse_bool, True)
        lead_in = mop_elem.find("LeadInMove")
        mop_specific_kwargs["lead_in_type"] = nested_parameter(lead_in, "LeadInType", parse_string, "Spiral")
        mop_specific_kwargs["lead_in_spiral_angle"] = nested_parameter(lead_in, "SpiralAngle", _parse_float, 30.0)
        mop_specific_kwargs["final_depth_increment"] = parameter("FinalDepthIncrement", _parse_float, 0.0, True)
        mop_specific_kwargs["cut_ordering"] = parameter("CutOrdering", parse_string, "DepthFirst")
        mop_specific_kwargs["region_fill_style"] = parameter("RegionFillStyle", parse_string, "InsideOutsideOffsets")
        mop_specific_kwargs["finish_stepover"] = parameter("FinishStepover", _parse_float, 0.0)
        mop_specific_kwargs["finish_stepover_at_target_depth"] = parameter("FinishStepoverAtTargetDepth", _parse_bool, False)
        mop_specific_kwargs["roughing_finishing"] = parameter("RoughingFinishing", parse_string, "Roughing")
    elif mop_class is EngraveMop:
        mop_specific_kwargs["roughing_finishing"] = parameter("RoughingFinishing", parse_string, "Roughing")
        mop_specific_kwargs["final_depth_increment"] = parameter("FinalDepthIncrement", _parse_float, 0.0, True)
        mop_specific_kwargs["cut_ordering"] = parameter("CutOrdering", parse_string, "DepthFirst")
    elif mop_class is DrillMop:
        mop_specific_kwargs["drilling_method"] = parameter("DrillingMethod", parse_string, "CannedCycle")
        mop_specific_kwargs["peck_distance"] = parameter("PeckDistance", _parse_float, 0.0)
        mop_specific_kwargs["retract_height"] = parameter("RetractHeight", _parse_float, 5.0)
        mop_specific_kwargs["dwell"] = parameter("Dwell", _parse_float, 0.0)
        mop_specific_kwargs["hole_diameter"] = parameter("HoleDiameter", _parse_float, None, True)
        mop_specific_kwargs["drill_lead_out"] = parameter("DrillLeadOut", _parse_bool, False)
        mop_specific_kwargs["spiral_flat_base"] = parameter("SpiralFlatBase", _parse_bool, True)
        mop_specific_kwargs["lead_out_length"] = parameter("LeadOutLength", _parse_float, 0.0)
        mop_specific_kwargs["custom_script"] = parameter("CustomScript", parse_string, "")

    # --- Create and Register MOP ---
    try:
        mop = mop_class(
            user_identifier=mop_identifier,
            name=mop_name,
            enabled=enabled,
            target_depth=target_depth,
            depth_increment=depth_increment,
            stock_surface=stock_surface,
            roughing_clearance=roughing_clearance,
            clearance_plane=clearance_plane,
            spindle_direction=spindle_dir,
            spindle_speed=spindle_speed,
            velocity_mode=velocity_mode,
            work_plane=work_plane,
            optimisation_mode=optimisation_mode,
            tool_diameter=tool_diameter,
            tool_number=tool_number,
            tool_profile=tool_profile,
            plunge_feedrate=plunge_feed,
            cut_feedrate=cut_feedrate,
            max_crossover_distance=max_crossover,
            custom_mop_header=custom_header,
            custom_mop_footer=custom_footer,
            **mop_specific_kwargs
        )
        mop.internal_id = internal_id
        for field_name in MOP_XML_PATH_TO_FIELD.values():
            if hasattr(mop, field_name):
                parameter_baseline.setdefault(field_name, getattr(mop, field_name))
        mop._xml_parameter_states = parameter_states
        mop._xml_parameter_baseline = parameter_baseline
        mop._xml_template = deepcopy(mop_elem)
        mop._xml_primitive_index = 0
        for child in mop_elem:
            if child.tag == "primitive":
                break
            if child.tag not in {"Name", "Tag"}:
                mop._xml_primitive_index += 1
        for child in list(mop._xml_template):
            if child.tag in {"Name", "Tag", "primitive"}:
                mop._xml_template.remove(child)
        if not project._register_entity(mop, project._mops):
            raise CamBamReaderError(f"Failed to register MOP '{mop_name}'")
        project.assign_mop_to_part(mop.internal_id, part_uuid)
        mop_primitive_xml_id_refs[mop.internal_id] = primitive_xml_ids
    except Exception as e:
        raise CamBamReaderError(f"Error reconstructing MOP '{mop_name}': {e}") from e


def _primitive_type(element):
    typed = element.get("{http://www.w3.org/2001/XMLSchema-instance}type")
    if typed:
        return {"Polyline": "pline", "Circle": "circle", "Rectangle": "rect",
                "Arc": "arc", "PointList": "points", "Text": "text"}.get(typed, typed)
    return element.tag


def _geometry_point(value):
    """Read supported XY/XYZ coordinates without silently substituting geometry."""
    parts = tuple(float(p.strip()) for p in value.split(',')) if value is not None else ()
    if len(parts) not in (2, 3) or not np.isfinite(parts).all():
        raise ValueError(f"Expected a finite XY or XYZ coordinate, got {value!r}")
    return parts if len(parts) == 3 else (*parts, 0.0)


def _text_element_content(element: ET.Element) -> str:
    """Read Text mixed content from framework or CamBam child ordering.

    Framework XML places content before ``Tag``; CamBam 1.0 places it after
    ``mat``. Formatting-only indentation is ignored. Multiple non-whitespace
    direct chunks are ambiguous and rejected instead of being reordered.
    """
    chunks = [element.text, *(child.tail for child in element)]
    content = [chunk for chunk in chunks if chunk is not None and chunk.strip()]
    if len(content) > 1:
        raise ValueError("Text element contains multiple direct content chunks")
    return content[0] if content else ""


def _reconstruct_primitive(project: CamBamProject, prim_elem: ET.Element, layer_uuid: uuid.UUID,
                           xml_id_to_primitive_uuid: Dict[int, uuid.UUID],
                           primitive_parent_ref: Dict[uuid.UUID, Union[int, uuid.UUID, str]]):
    """Parses a primitive element (<pline>, <circle>...) and adds it to the project."""
    prim_class = PRIMITIVE_TAG_TO_CLASS.get(_primitive_type(prim_elem))
    if not prim_class: return

    xml_id_str = prim_elem.get("id")
    xml_id = _parse_int(xml_id_str, -1)
    if xml_id <= 0:
        raise CamBamReaderError(f"Primitive <{prim_elem.tag}> has missing or invalid id")
    if xml_id in xml_id_to_primitive_uuid:
        raise CamBamReaderError(f"Duplicate primitive XML id {xml_id}")

    # --- Parse Tag Data (User ID, Internal UUID, Groups, Parent, Description) ---
    tag_node = prim_elem.find("Tag")
    user_identifier = f"{prim_elem.tag}_{xml_id}" # Default identifier
    internal_uuid: Optional[uuid.UUID] = None
    groups: List[str] = []
    parent_ref: Optional[Union[int, uuid.UUID, str]] = None # Store XML ID or UUID string
    description = ""

    if tag_node is not None and tag_node.text:
        try:
            tag_data = json.loads(tag_node.text)
            if isinstance(tag_data, dict):
                 user_identifier = tag_data.get("user_id", user_identifier)
                 # Try to recover original internal UUID
                 internal_uuid_str = tag_data.get("internal_id")
                 if internal_uuid_str:
                     try: internal_uuid = uuid.UUID(internal_uuid_str)
                     except ValueError: logger.warning(f"Invalid internal_id format in Tag for XML ID {xml_id}: '{internal_uuid_str}'")
                 groups = tag_data.get("groups", [])
                 if not isinstance(groups, list): groups = []
                 description = tag_data.get("description", "")
                 # Get parent reference (could be our internal UUID string, or potentially an XML ID if written differently)
                 parent_ref_str = tag_data.get("parent")
                 if parent_ref_str:
                     # Try parsing as UUID first, then as int (XML ID)
                     try: parent_ref = uuid.UUID(str(parent_ref_str))
                     except ValueError:
                          try: parent_ref = int(str(parent_ref_str))
                          except ValueError: logger.warning(f"Invalid parent reference '{parent_ref_str}' in Tag for XML ID {xml_id}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from Tag for primitive XML ID {xml_id}.")
        except Exception as e:
             logger.warning(f"Error processing Tag data for primitive XML ID {xml_id}: {e}")

    # If internal UUID wasn't in Tag, generate a new one
    if internal_uuid is None:
        internal_uuid = uuid.uuid4()
        logger.debug(f"Generated new internal UUID {internal_uuid} for primitive XML ID {xml_id}")

    # --- Parse Transformation Matrix ---
    matrix_str = prim_elem.find("mat")
    effective_transform = identity_matrix() # Default if no matrix found
    world_z_offset = 0.0
    if matrix_str is not None and "m" in matrix_str.attrib:
        try:
            # Keep the XML world pose until the deferred parent-linking pass
            # reconstructs local matrices from snapshots of all imported poses.
            total_transform, world_z_offset = from_cambam_matrix_str(matrix_str.attrib["m"], return_z=True)
            effective_transform = total_transform # Store total as effective initially
        except ValueError as e:
            raise CamBamReaderError(f"Unsupported matrix for primitive XML ID {xml_id}: {e}") from e
        except Exception as e:
             raise CamBamReaderError(f"Invalid matrix for primitive XML ID {xml_id}: {e}") from e


    # --- Parse Primitive-Specific Geometry ---
    prim_specific_kwargs = {}
    try:
        if prim_class is Pline:
            points = []
            elevations = []
            pts_node = prim_elem.find("pts")
            if pts_node is not None:
                for p_elem in pts_node.findall("p"):
                    pt = _geometry_point(p_elem.text)
                    if pt:
                        bulge = float(p_elem.get("b", "0"))
                        points.append((pt[0], pt[1], bulge)) # Store x, y, bulge
                        elevations.append(pt[2])
            prim_specific_kwargs["relative_points"] = points
            prim_specific_kwargs["vertex_z"] = elevations
            prim_specific_kwargs["closed"] = _parse_bool(prim_elem.get("Closed"), False)
        elif prim_class is Circle:
             center = _geometry_point(prim_elem.get("c"))
             diameter = _parse_float(prim_elem.get("d"), 1.0)
             prim_specific_kwargs.update(relative_center=center[:2], elevation=center[2])
             prim_specific_kwargs["diameter"] = diameter
        elif prim_class is Rect:
             corner = _geometry_point(prim_elem.get("p"))
             width = _parse_float(prim_elem.get("w"), 1.0)
             height = _parse_float(prim_elem.get("h"), 1.0)
             prim_specific_kwargs.update(relative_corner=corner[:2], elevation=corner[2])
             prim_specific_kwargs["width"] = width
             prim_specific_kwargs["height"] = height
        elif prim_class is Arc:
             center = _geometry_point(prim_elem.get("p"))
             radius = _parse_float(prim_elem.get("r"), 1.0)
             start = _parse_float(prim_elem.get("s"), 0.0)
             sweep = _parse_float(prim_elem.get("w"), 90.0)
             prim_specific_kwargs.update(relative_center=center[:2], elevation=center[2])
             prim_specific_kwargs["radius"] = radius
             prim_specific_kwargs["start_angle"] = start
             prim_specific_kwargs["extent_angle"] = sweep
        elif prim_class is Points:
             points = []
             elevations = []
             pts_node = prim_elem.find("pts")
             if pts_node is not None:
                 for p_elem in pts_node.findall("p"):
                     pt = _geometry_point(p_elem.text)
                     points.append(pt[:2])
                     elevations.append(pt[2])
             prim_specific_kwargs["relative_points"] = points
             prim_specific_kwargs["vertex_z"] = elevations
        elif prim_class is Text:
             # Native CamBam suppresses p1 for the default origin.
             pos1 = _geometry_point(prim_elem.get("p1", "0,0,0"))
             pos2_text = prim_elem.get("p2")
             pos2 = _geometry_point(pos2_text) if pos2_text is not None else None
             height = _parse_float(prim_elem.get("Height"), 10.0)
             font = prim_elem.get("Font", "Arial")
             style = prim_elem.get("style", "")
             linespace = _parse_float(prim_elem.get("linespace"), 1.0)
             align_str = prim_elem.get("align", "center,center")
             align_parts = align_str.split(',')
             v_align = align_parts[0].strip() if len(align_parts) > 0 else "center"
             h_align = align_parts[1].strip() if len(align_parts) > 1 else "center"
             text_content = _text_element_content(prim_elem)

             prim_specific_kwargs.update(relative_position=pos1[:2], elevation=pos1[2])
             if pos2 is not None:
                 prim_specific_kwargs.update(
                     xml_p2_position=pos2[:2], xml_p2_elevation=pos2[2]
                 )
             prim_specific_kwargs["height"] = height
             prim_specific_kwargs["font"] = font
             prim_specific_kwargs["style"] = style
             prim_specific_kwargs["line_spacing"] = linespace
             prim_specific_kwargs["align_vertical"] = v_align
             prim_specific_kwargs["align_horizontal"] = h_align
             prim_specific_kwargs["text_content"] = text_content
        elif prim_class is Region:
             prim_specific_kwargs = parse_region_geometry(prim_elem)

    except Exception as e:
        raise CamBamReaderError(f"Invalid geometry for primitive XML ID {xml_id}: {e}") from e


    # --- Create and Register Primitive ---
    try:
        # Instantiate the primitive class
        primitive = prim_class(
            user_identifier=user_identifier,
            groups=groups, # Store groups read from Tag
            description=description,
            effective_transform=effective_transform, # Store total transform initially
            local_z_offset=world_z_offset,
            **prim_specific_kwargs
        )
        # Manually set the internal UUID we recovered or generated
        primitive.internal_id = internal_uuid

        # Register with project
        if project._register_entity(primitive, project._primitives):
             # Assign layer relationship
             project.assign_primitive_to_layer(primitive.internal_id, layer_uuid)
             # Set groups via relationship manager (redundant if primitive stores it?)
             project.set_primitive_groups(primitive.internal_id, groups)

             # Store mapping from XML ID to this primitive's UUID
             xml_id_to_primitive_uuid[xml_id] = primitive.internal_id

             # Store parent reference for later linking pass
             if parent_ref is not None:
                 primitive_parent_ref[primitive.internal_id] = parent_ref
        else:
            raise CamBamReaderError(f"Failed to register primitive XML ID {xml_id} (User ID '{user_identifier}')")

    except Exception as e:
        raise CamBamReaderError(f"Error reconstructing primitive XML ID {xml_id}: {e}") from e
