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
from typing import Optional, Dict, List, Tuple, Union, Any

import numpy as np # For matrix conversion

from .cambam_project import CamBamProject
from .cambam_entities import ( # Import concrete entity types
    Layer, Part, Mop, Primitive,
    Pline, Circle, Rect, Arc, Points, Text,
    ProfileMop, PocketMop, EngraveMop, DrillMop
)
from .cad_transformations import identity_matrix, from_cambam_matrix_str # For parsing matrix

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

        # 2. MOPs (read structure, store primitive XML ID refs)
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

        # 3. Layers
        logger.debug("Reading Layers...")
        layers_node = root.find("layers")
        if layers_node is not None:
            for layer_elem in layers_node.findall("layer"):
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
                         prim_type_tag = prim_elem.tag
                         if prim_type_tag in PRIMITIVE_TAG_TO_CLASS:
                            _reconstruct_primitive(project, prim_elem, layer_uuid,
                                                   xml_id_to_primitive_uuid, primitive_parent_ref)
                         else:
                            logger.warning(f"Unsupported primitive type tag '{prim_type_tag}' encountered in layer '{layer_elem.get('name')}'. Skipping.")

        # 5. Link Relationships using stored temporary data
        logger.debug("Linking relationships...")
        # 5a. Link Primitive Parents
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
                     project.link_primitive_parent(child_uuid, parent_uuid)
                 else:
                     logger.warning(f"Could not link child primitive {child_uuid}: Resolved parent UUID {parent_uuid} not found in project primitives registry.")

        # 5b. Link MOPs to Primitives (resolve pid_source if it was XML IDs)
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

            # Update the MOP's pid_source if it was originally defined by XML IDs
            # Assumes _reconstruct_mop stored the XML IDs in pid_source if that's how it was defined
            if isinstance(mop.pid_source, list): # Check if it needs updating
                 mop.pid_source = resolved_primitive_uuids # Replace list of XML IDs/placeholders with UUIDs
                 if not all_resolved:
                     logger.warning(f"MOP '{mop.name}' pid_source list only partially resolved.")
                 elif not resolved_primitive_uuids and xml_ids:
                      logger.warning(f"MOP '{mop.name}' pid_source list {xml_ids} resolved to empty UUID list.")
            # If pid_source was a string (group name), it remains unchanged.

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
    project.add_part(
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


def _reconstruct_mop(project: CamBamProject, mop_elem: ET.Element, part_uuid: uuid.UUID,
                      mop_primitive_xml_id_refs: Dict[uuid.UUID, List[int]]):
    """Parses a MOP element (<profile>, <pocket>...) and adds it to the project."""
    mop_class = MOP_TAG_TO_CLASS.get(mop_elem.tag)
    if not mop_class: return # Should have been checked by caller

    # --- Parse Common MOP Parameters ---
    mop_name = mop_elem.findtext("Name", f"Unnamed_{mop_elem.tag}")
    enabled = _parse_bool(mop_elem.get("Enabled"), True) # Enabled is attribute on root MOP tag

    target_depth_str = mop_elem.findtext("TargetDepth")
    target_depth = _parse_float(target_depth_str, None) if target_depth_str is not None else None

    depth_inc_str = mop_elem.findtext("DepthIncrement")
    depth_increment = _parse_float(depth_inc_str, None) if depth_inc_str is not None else None

    stock_surface = _parse_float(mop_elem.findtext("StockSurface"), 0.0)
    roughing_clearance = _parse_float(mop_elem.findtext("RoughingClearance"), 0.0)
    clearance_plane = _parse_float(mop_elem.findtext("ClearancePlane"), 15.0)
    spindle_dir = mop_elem.findtext("SpindleDirection", "CW")

    spindle_speed_str = mop_elem.findtext("SpindleSpeed")
    spindle_speed = _parse_int(spindle_speed_str, None) if spindle_speed_str is not None else None

    velocity_mode = mop_elem.findtext("VelocityMode", "ExactStop")
    work_plane = mop_elem.findtext("WorkPlane", "XY")
    optimisation_mode = mop_elem.findtext("OptimisationMode", "Standard")

    tool_dia_str = mop_elem.findtext("ToolDiameter")
    tool_diameter = _parse_float(tool_dia_str, None) if tool_dia_str is not None else None

    tool_number = _parse_int(mop_elem.findtext("ToolNumber"), 0)
    tool_profile = mop_elem.findtext("ToolProfile", "EndMill")
    plunge_feed = _parse_float(mop_elem.findtext("PlungeFeedrate"), 1000.0)

    cut_feed_str = mop_elem.findtext("CutFeedrate")
    cut_feedrate = _parse_float(cut_feed_str, None) if cut_feed_str is not None else None

    max_crossover = _parse_float(mop_elem.findtext("MaxCrossoverDistance"), 0.7)
    custom_header = mop_elem.findtext("CustomMOPHeader", "")
    custom_footer = mop_elem.findtext("CustomMOPFooter", "")

    # Determine pid_source: Check for <primitive> tag content
    # CamBam seems to store EITHER a list of primitive IDs OR rely on implicit context (e.g. selected items).
    # If the <primitive> tag is present and non-empty, we use its content.
    # If it's missing or empty, the MOP targets nothing explicitly via XML refs.
    # We cannot easily reconstruct group membership from standard CamBam XML alone,
    # unless our writer puts group info into a Tag (which it doesn't for MOPs).
    # So, pid_source reconstruction will primarily be a list of XML IDs found, or empty list.
    primitive_xml_ids: List[int] = []
    primitive_container = mop_elem.find("primitive")
    if primitive_container is not None:
        for prim_ref in primitive_container.findall("prim"):
            xml_id = _parse_int(prim_ref.text, -1)
            if xml_id > 0:
                primitive_xml_ids.append(xml_id)

    # Store the list of XML IDs temporarily. This will be resolved to UUIDs later.
    # We store it directly as the pid_source for now, assuming list type.
    pid_source_value: Union[str, List[int]] = primitive_xml_ids # Treat as list of XML IDs for now


    # --- Parse MOP-Specific Parameters ---
    # This requires adding parsing logic for each MOP type based on its unique tags
    mop_specific_kwargs = {}
    if mop_class is ProfileMop:
        mop_specific_kwargs["stepover"] = _parse_float(mop_elem.findtext("StepOver"), 0.4)
        mop_specific_kwargs["profile_side"] = mop_elem.findtext("InsideOutside", "Inside")
        mop_specific_kwargs["milling_direction"] = mop_elem.findtext("MillingDirection", "Conventional")
        # ... parse LeadIn, Tabs etc.
    elif mop_class is PocketMop:
         mop_specific_kwargs["stepover"] = _parse_float(mop_elem.findtext("StepOver"), 0.4)
         mop_specific_kwargs["region_fill_style"] = mop_elem.findtext("RegionFillStyle", "InsideOutsideOffsets")
         # ... parse FinishStepover etc.
    elif mop_class is EngraveMop:
         # ... parse Engrave specific if any
         pass
    elif mop_class is DrillMop:
         mop_specific_kwargs["drilling_method"] = mop_elem.findtext("DrillingMethod", "CannedCycle")
         mop_specific_kwargs["peck_distance"] = _parse_float(mop_elem.findtext("PeckDistance"), 0.0)
         mop_specific_kwargs["dwell"] = _parse_float(mop_elem.findtext("Dwell"), 0.0)
         # ... parse HoleDiameter etc.

    # --- Create and Register MOP ---
    try:
        # Use MOP name as identifier for add_mop_internal (might need adjusting if names aren't unique)
        # Or perhaps generate a UUID based on hash of XML element? For now, use name.
        mop_identifier = mop_name # Potential uniqueness issue here!
        mop = project._add_mop_internal(
            MopClass=mop_class,
            part_identifier=part_uuid,
            pid_source=pid_source_value, # Store XML IDs for now
            name=mop_name,
            identifier=mop_identifier, # Use name as identifier during reconstruction
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
        if mop:
            # Store the XML IDs referenced by this MOP for later resolution
            mop_primitive_xml_id_refs[mop.internal_id] = primitive_xml_ids
        else:
             logger.error(f"Failed to reconstruct MOP '{mop_name}' of type {mop_class.__name__}.")

    except Exception as e:
        logger.error(f"Error reconstructing MOP '{mop_name}': {e}", exc_info=True)


def _reconstruct_primitive(project: CamBamProject, prim_elem: ET.Element, layer_uuid: uuid.UUID,
                           xml_id_to_primitive_uuid: Dict[int, uuid.UUID],
                           primitive_parent_ref: Dict[uuid.UUID, Union[int, uuid.UUID, str]]):
    """Parses a primitive element (<pline>, <circle>...) and adds it to the project."""
    prim_class = PRIMITIVE_TAG_TO_CLASS.get(prim_elem.tag)
    if not prim_class: return

    xml_id_str = prim_elem.get("id")
    xml_id = _parse_int(xml_id_str, -1)
    if xml_id <= 0:
        logger.warning(f"Skipping primitive element <{prim_elem.tag}> with missing or invalid 'id' attribute.")
        return

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
                     try: parent_ref = uuid.UUID(parent_ref_str)
                     except ValueError:
                          try: parent_ref = int(parent_ref_str)
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
    if matrix_str is not None and "m" in matrix_str.attrib:
        try:
            # Note: CamBam matrix is the TOTAL transform. We store it as effective_transform for now.
            # If parent links exist, this might need adjustment post-linking, or assume baked state.
            # Let's assume the matrix represents the final state relative to world origin.
            # If we reconstruct parent links, the child's effective transform should become Identity?
            # Or T_total = T_parent_total * T_child_effective. => T_child_effective = inv(T_parent_total) * T_total
            # This is complex to reconstruct perfectly. Let's load T_total into effective_transform first.
            total_transform = from_cambam_matrix_str(matrix_str.attrib["m"])
            effective_transform = total_transform # Store total as effective initially
        except ValueError as e:
            logger.warning(f"Could not parse transformation matrix for primitive XML ID {xml_id}: {e}")
        except Exception as e:
             logger.error(f"Unexpected error parsing matrix for primitive XML ID {xml_id}: {e}")


    # --- Parse Primitive-Specific Geometry ---
    prim_specific_kwargs = {}
    try:
        if prim_class is Pline:
            points = []
            pts_node = prim_elem.find("pts")
            if pts_node is not None:
                for p_elem in pts_node.findall("p"):
                    pt = _parse_point_3d(p_elem.text)
                    if pt:
                        bulge = _parse_float(p_elem.get("b"), 0.0)
                        points.append((pt[0], pt[1], bulge)) # Store x, y, bulge
            prim_specific_kwargs["relative_points"] = points
            prim_specific_kwargs["closed"] = _parse_bool(prim_elem.get("Closed"), False)
        elif prim_class is Circle:
             center = _parse_point_2d(prim_elem.get("c"))
             diameter = _parse_float(prim_elem.get("d"), 1.0)
             if center: prim_specific_kwargs["relative_center"] = center
             prim_specific_kwargs["diameter"] = diameter
        elif prim_class is Rect:
             corner = _parse_point_2d(prim_elem.get("p"))
             width = _parse_float(prim_elem.get("w"), 1.0)
             height = _parse_float(prim_elem.get("h"), 1.0)
             if corner: prim_specific_kwargs["relative_corner"] = corner
             prim_specific_kwargs["width"] = width
             prim_specific_kwargs["height"] = height
        elif prim_class is Arc:
             center = _parse_point_2d(prim_elem.get("p"))
             radius = _parse_float(prim_elem.get("r"), 1.0)
             start = _parse_float(prim_elem.get("s"), 0.0)
             sweep = _parse_float(prim_elem.get("w"), 90.0)
             if center: prim_specific_kwargs["relative_center"] = center
             prim_specific_kwargs["radius"] = radius
             prim_specific_kwargs["start_angle"] = start
             prim_specific_kwargs["extent_angle"] = sweep
        elif prim_class is Points:
             points = []
             pts_node = prim_elem.find("pts")
             if pts_node is not None:
                 for p_elem in pts_node.findall("p"):
                     pt = _parse_point_2d(p_elem.text) # Points are usually 2D
                     if pt: points.append(pt)
             prim_specific_kwargs["relative_points"] = points
        elif prim_class is Text:
             pos1 = _parse_point_2d(prim_elem.get("p1"))
             height = _parse_float(prim_elem.get("Height"), 10.0)
             font = prim_elem.get("Font", "Arial")
             style = prim_elem.get("style", "")
             linespace = _parse_float(prim_elem.get("linespace"), 1.0)
             align_str = prim_elem.get("align", "center,center")
             align_parts = align_str.split(',')
             v_align = align_parts[0].strip() if len(align_parts) > 0 else "center"
             h_align = align_parts[1].strip() if len(align_parts) > 1 else "center"
             text_content = prim_elem.text or ""

             if pos1: prim_specific_kwargs["relative_position"] = pos1
             prim_specific_kwargs["height"] = height
             prim_specific_kwargs["font"] = font
             prim_specific_kwargs["style"] = style
             prim_specific_kwargs["line_spacing"] = linespace
             prim_specific_kwargs["align_vertical"] = v_align
             prim_specific_kwargs["align_horizontal"] = h_align
             prim_specific_kwargs["text_content"] = text_content

    except Exception as e:
        logger.error(f"Error parsing geometry for primitive <{prim_elem.tag}> XML ID {xml_id}: {e}", exc_info=True)
        return # Skip this primitive if geometry parsing fails


    # --- Create and Register Primitive ---
    try:
        # Instantiate the primitive class
        primitive = prim_class(
            user_identifier=user_identifier,
            groups=groups, # Store groups read from Tag
            description=description,
            effective_transform=effective_transform, # Store total transform initially
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
            logger.error(f"Failed to register reconstructed primitive XML ID {xml_id} (User ID '{user_identifier}')")

    except Exception as e:
        logger.error(f"Error instantiating or registering primitive XML ID {xml_id}: {e}", exc_info=True)