# INCOMPLETE: TODO: COMPLETE FULL FUNCTIONING SERIAL RECONSTRUCTION FROM CB FILE WITHOUT PICKLING.
"""
cambam_reader.py

Provides a function to read a CamBam XML file and reconstruct a CamBamProject.
This is a basic implementation; real-world use may require additional error checking
and support for all features.
"""

import xml.etree.ElementTree as ET
import logging
from typing import Optional
from cambam_project import CamBamProject
from cambam_entities import Layer, Part

logger = logging.getLogger(__name__)

def read_cambam_file(file_path: str) -> Optional[CamBamProject]:
    """
    Read a CamBam XML file and return a CamBamProject instance.
    This implementation reads basic project, layers, and part data.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        project_name = root.get("Name", "Unnamed Project")
        project = CamBamProject(project_name)
        # Read layers
        layers_node = root.find("layers")
        if layers_node is not None:
            for layer_elem in layers_node.findall("layer"):
                name = layer_elem.get("name")
                color = layer_elem.get("color", "Green")
                alpha = float(layer_elem.get("alpha", "1.0"))
                pen = float(layer_elem.get("pen", "1.0"))
                visible = layer_elem.get("visible", "true").lower() == "true"
                locked = layer_elem.get("locked", "false").lower() == "true"
                layer = project.add_layer(name, color, alpha, pen, visible, locked)
        # Read parts
        parts_node = root.find("parts")
        if parts_node is not None:
            for part_elem in parts_node.findall("part"):
                name = part_elem.get("Name")
                enabled = part_elem.get("Enabled", "true").lower() == "true"
                # For simplicity we use default stock values; a real reader would parse these.
                part = project.add_part(name, enabled)
        # Note: Detailed reconstruction of primitives and MOPs would follow similar parsing.
        logger.info(f"Read project '{project.project_name}' from {file_path}")
        return project
    except Exception as e:
        logger.error(f"Error reading CamBam file: {e}")
        return None
