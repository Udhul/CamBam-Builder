"""
cambam_writer.py

Provides functions to assemble the final CamBam XML file from a CamBamProject instance.
This module separates the XML generation logic from the project management.
"""

import xml.etree.ElementTree as ET
import os
import logging
from typing import Dict
import uuid

from cambam_entities import Primitive, Layer, Part, XmlPrimitiveIdResolver
from cambam_project import CamBamProject

logger = logging.getLogger(__name__)

def build_xml_tree(project: CamBamProject) -> ET.ElementTree:
    # 1. Assign XML primitive IDs
    uuid_to_xml_id: Dict[uuid.UUID, int] = {}
    xml_id_counter = 1
    sorted_ids = sorted(project._primitives.keys(), key=lambda u: u.int)
    for uid in sorted_ids:
        uuid_to_xml_id[uid] = xml_id_counter
        xml_id_counter += 1

    # 2. Create root element
    root = ET.Element("CADFile", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xmlns:xsd": "http://www.w3.org/2001/XMLSchema",
        "Version": "0.9.8.0",
        "Name": project.project_name
    })

    # 3. Global machining options (static for now)
    machining_options = ET.SubElement(root, "MachiningOptions")
    stock = ET.SubElement(machining_options, "Stock")
    ET.SubElement(stock, "Material")
    ET.SubElement(stock, "PMin").text = "0,0,0"
    ET.SubElement(stock, "PMax").text = "0,0,0"
    ET.SubElement(stock, "Color").text = "255,165,0"

    # 4. Build Layers structure by iterating over all registered layers.
    layers_container = ET.SubElement(root, "layers")
    layer_objects_map: Dict[uuid.UUID, ET.Element] = {}
    for lid, layer in project._layers.items():
        layer_elem = layer.to_xml_element()
        layers_container.append(layer_elem)
        if layer._xml_objects_element is not None:
            layer_objects_map[lid] = layer._xml_objects_element
        else:
            logger.error(f"Layer {layer.user_identifier} missing objects element.")

    # 5. Build Primitives (place primitives in the correct layer container)
    for uid in sorted_ids:
        primitive = project.get_primitive(uid)
        if not primitive:
            continue
        xml_id = uuid_to_xml_id[uid]
        try:
            prim_elem = primitive.to_xml_element(xml_id)
        except Exception as e:
            logger.error(f"Error building XML for primitive {primitive.user_identifier}: {e}")
            continue
        layer_container = layer_objects_map.get(primitive.layer_id)
        if layer_container:
            layer_container.append(prim_elem)
        else:
            logger.error(f"Layer container not found for primitive {primitive.user_identifier}.")
            if layer_objects_map:
                next(iter(layer_objects_map.values())).append(prim_elem)

    # 6. Build Parts structure
    parts_container = ET.SubElement(root, "parts")
    part_machineops_map: Dict[uuid.UUID, ET.Element] = {}
    for pid, part in project._parts.items():
        part_elem = part.to_xml_element()
        parts_container.append(part_elem)
        if part._xml_machineops_element is not None:
            part_machineops_map[pid] = part._xml_machineops_element
        else:
            logger.error(f"Part {part.user_identifier} missing machineops element.")

    # 7. Build MOPs (placed in correct part)
    resolver = XmlPrimitiveIdResolver(uuid_to_xml_id, project._primitive_groups, project._primitives)
    for pid, part in project._parts.items():
        machineops_container = part_machineops_map.get(pid)
        if machineops_container is None:
            logger.error(f"Machineops container not found for part {part.user_identifier}.")
            continue
        for mop_id in part.mop_ids:
            mop = project.get_mop(mop_id)
            if not mop:
                logger.warning(f"MOP {mop_id} not found in part {part.user_identifier}.")
                continue
            try:
                mop.resolve_xml_primitive_ids(resolver)
                mop_elem = mop.to_xml_element(project)
                machineops_container.append(mop_elem)
            except Exception as e:
                logger.error(f"Error building XML for MOP {mop.user_identifier}: {e}")
    return ET.ElementTree(root)

def save_cambam_file(project: CamBamProject, file_path: str, pretty_print: bool = True) -> None:
    try:
        base, _ = os.path.splitext(file_path)
        output_path = base + '.cb'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tree = build_xml_tree(project)
        if pretty_print:
            try:
                ET.indent(tree, space="  ", level=0)
            except AttributeError:
                logger.warning("XML pretty-printing requires Python 3.9+.")
        tree.write(output_path, encoding='utf-8', xml_declaration=True, short_empty_elements=False)
        logger.info(f"CamBam file saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving CamBam file: {e}")
        raise
