"""
cambam_writer.py

Provides functions to serialize a CamBamProject instance into the CamBam XML file format.
It queries the project's registries to correctly structure layers, parts, primitives,
MOPs, and their relationships according to the CamBam schema.
"""

import xml.etree.ElementTree as ET
import os
import logging
import uuid
from typing import Dict, List

from cambam_project import CamBamProject # Use Type Hinting
from cambam_entities import Primitive, Layer, Part, Mop # For type checking if needed

logger = logging.getLogger(__name__)

def build_xml_tree(project: CamBamProject) -> ET.ElementTree:
    """Constructs the XML ElementTree for the CamBam project."""

    # 1. Assign XML integer IDs to primitives (consistent ordering)
    # We need a map from Primitive UUID -> XML int ID
    uuid_to_xml_id: Dict[uuid.UUID, int] = {}
    xml_id_counter = 1
    # Sort primitives by UUID for deterministic ID assignment
    sorted_primitive_uuids = sorted(project._primitives.keys())
    for prim_uuid in sorted_primitive_uuids:
        uuid_to_xml_id[prim_uuid] = xml_id_counter
        xml_id_counter += 1

    # 2. Create root <CADFile> element
    root = ET.Element("CADFile", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xmlns:xsd": "http://www.w3.org/2001/XMLSchema",
        "Version": "0.9.8.0", # Or make this configurable?
        "Name": project.project_name
    })

    # 3. Global Machining Options (Defaults for now, could be project attributes)
    machining_options = ET.SubElement(root, "MachiningOptions")
    stock = ET.SubElement(machining_options, "Stock")
    ET.SubElement(stock, "Material") # Empty Material tag
    ET.SubElement(stock, "PMin").text = "0,0,0" # Default values
    ET.SubElement(stock, "PMax").text = "0,0,0"
    ET.SubElement(stock, "Color").text = "255,165,0" # Default orange

    # 4. Build <layers> container
    layers_container = ET.SubElement(root, "layers")
    # Iterate through layers *in order* defined by the project
    for layer_uuid in project._layer_order:
        layer = project.get_layer(layer_uuid)
        if not layer:
             logger.warning(f"Layer UUID {layer_uuid} found in order list but not in registry. Skipping.")
             continue

        # Create the <layer> element itself
        layer_elem = layer.to_xml_element()
        layers_container.append(layer_elem)

        # Create the <objects> container within this layer
        objects_container = ET.SubElement(layer_elem, "objects")

        # Get primitives assigned to this layer from the project registry
        primitives_on_layer = project.get_primitives_on_layer(layer_uuid)

        # Add primitive XML elements to this layer's <objects> container
        for primitive in sorted(primitives_on_layer, key=lambda p: uuid_to_xml_id[p.internal_id]): # Sort by XML ID for consistency
             xml_id = uuid_to_xml_id.get(primitive.internal_id)
             if xml_id is None:
                 logger.error(f"Primitive {primitive.user_identifier} ({primitive.internal_id}) on layer {layer.user_identifier} has no assigned XML ID. Skipping.")
                 continue

             # Get parent UUID from project registry to inject into the Tag
             parent_uuid = project.get_parent_of_primitive(primitive.internal_id)

             try:
                 # Generate the primitive's specific XML element (<pline>, <circle>, etc.)
                 # Pass the required XML ID and parent UUID
                 prim_elem = primitive.to_xml_element(xml_id, parent_uuid)
                 objects_container.append(prim_elem)
             except Exception as e:
                 logger.error(f"Error building XML for primitive {primitive.user_identifier} ({primitive.internal_id}): {e}", exc_info=True)


    # 5. Build <parts> container
    parts_container = ET.SubElement(root, "parts")
    # Iterate through parts *in order* defined by the project
    for part_uuid in project._part_order:
        part = project.get_part(part_uuid)
        if not part:
             logger.warning(f"Part UUID {part_uuid} found in order list but not in registry. Skipping.")
             continue

        # Create the <part> element itself
        part_elem = part.to_xml_element()
        parts_container.append(part_elem)

        # Create the <machineops> container within this part
        machineops_container = ET.SubElement(part_elem, "machineops")

        # Get MOPs assigned to this part from the project registry, in order
        mops_in_part = project.get_mops_in_part(part_uuid)

        # Add MOP XML elements to this part's <machineops> container
        for mop in mops_in_part:
            # Resolve the MOP's pid_source (group or UUID list) to a list of primitive *XML IDs*
            try:
                primitive_uuids = project.resolve_pid_source_to_uuids(mop.pid_source)
                resolved_primitive_xml_ids: List[int] = []
                for prim_uuid in primitive_uuids:
                    xml_id = uuid_to_xml_id.get(prim_uuid)
                    if xml_id:
                        resolved_primitive_xml_ids.append(xml_id)
                    else:
                        # This case should be rare if resolve_pid_source_to_uuids filters correctly
                        logger.warning(f"MOP {mop.name} references primitive {prim_uuid} which has no XML ID assigned.")

                if not resolved_primitive_xml_ids and mop.pid_source:
                     # Log if the source wasn't empty but resolution yielded nothing valid
                     logger.warning(f"MOP '{mop.name}' ({mop.user_identifier}) resolved to zero primitives for source: {mop.pid_source}")

                # Generate the MOP's specific XML element (<profile>, <pocket>, etc.)
                # Pass the project context and the resolved XML IDs
                mop_elem = mop.to_xml_element(project, resolved_primitive_xml_ids)
                machineops_container.append(mop_elem)

            except Exception as e:
                logger.error(f"Error building XML for MOP {mop.name} ({mop.user_identifier}): {e}", exc_info=True)


    # 6. Return the complete ElementTree
    return ET.ElementTree(root)


def save_cambam_file(project: CamBamProject, file_path: str, pretty_print: bool = True) -> None:
    """
    Builds the XML tree for the project and saves it to a .cb file.

    Args:
        project: The CamBamProject instance to save.
        file_path: The desired output file path (extension will be forced to .cb).
        pretty_print: If True, attempts to indent the XML for readability (requires Python 3.9+).
    """
    try:
        # Ensure the output path has a .cb extension
        base, _ = os.path.splitext(file_path)
        output_path = base + '.cb'

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Handle case where path is just filename in current dir
             os.makedirs(output_dir, exist_ok=True)

        # Build the XML tree
        logger.info(f"Building XML tree for project '{project.project_name}'...")
        tree = build_xml_tree(project)
        logger.info("XML tree built.")

        # Apply pretty printing if requested and supported
        if pretty_print:
            try:
                # ET.indent is available in Python 3.9+
                ET.indent(tree, space="  ", level=0)
                logger.debug("XML pretty-printing applied.")
            except AttributeError:
                logger.warning("XML pretty-printing (indentation) requires Python 3.9 or later.")
            except Exception as e_indent:
                 logger.warning(f"XML indentation failed: {e_indent}")


        # Write the XML file
        tree.write(output_path, encoding='utf-8', xml_declaration=True, short_empty_elements=False)
        logger.info(f"CamBam file successfully saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save CamBam file to {file_path}: {e}", exc_info=True)
        raise # Re-raise the exception