# cambam_writer.py
import xml.etree.ElementTree as ET
import os
import uuid
import logging
from typing import Dict, Set, List, Union

# Module imports
from cambam_project import CamBamProject
from cambam_entities import Primitive, Layer, Part, Mop
from cad_common import CamBamError, XmlWritingError

logger = logging.getLogger(__name__)


class XmlPrimitiveIdResolver:
    """ Helper class passed to MOPs to resolve UUIDs/groups to XML integer IDs. """
    def __init__(self,
                 uuid_to_xml_id: Dict[uuid.UUID, int],
                 primitive_groups: Dict[str, Set[uuid.UUID]],
                 all_primitives: Dict[uuid.UUID, Primitive]):
        self._uuid_to_xml_id = uuid_to_xml_id
        self._primitive_groups = primitive_groups
        self._all_primitives = all_primitives # Keep reference for checking existence

    def resolve(self, pid_source: Union[str, List[uuid.UUID]]) -> List[int]:
        """ Resolves a group name or list of UUIDs to sorted XML integer IDs. """
        primitive_uuids: Set[uuid.UUID] = set()

        if isinstance(pid_source, str): # Group name
            group_uuids = self._primitive_groups.get(pid_source)
            if group_uuids is not None:
                # Verify UUIDs still exist in the project
                valid_uuids = {uid for uid in group_uuids if uid in self._all_primitives}
                if len(valid_uuids) != len(group_uuids):
                     logger.warning(f"Group '{pid_source}' contains {len(group_uuids) - len(valid_uuids)} stale primitive UUIDs.")
                primitive_uuids.update(valid_uuids)
            else:
                logger.warning(f"MOP references non-existent group '{pid_source}'.")

        elif isinstance(pid_source, list): # List of UUIDs
            for uid in pid_source:
                if uid in self._all_primitives:
                    primitive_uuids.add(uid)
                else:
                    logger.warning(f"MOP references non-existent primitive UUID '{uid}'.")
        else:
            logger.error(f"Invalid pid_source type for MOP resolution: {type(pid_source)}. Expected str or List[uuid.UUID].")
            return []

        # Convert valid UUIDs to XML IDs
        resolved_ids: List[int] = []
        for uid in primitive_uuids:
            xml_id = self._uuid_to_xml_id.get(uid)
            if xml_id is not None:
                resolved_ids.append(xml_id)
            else:
                # This should not happen if uuid_to_xml_id covers all primitives
                logger.error(f"Primitive UUID '{uid}' exists but missing from XML ID map during MOP resolution.")

        return sorted(resolved_ids)


class CamBamWriter:
    """ Handles writing a CamBamProject object to a CamBam XML file. """
    def __init__(self, project: CamBamProject):
        self.project = project
        self._uuid_to_xml_id: Dict[uuid.UUID, int] = {}
        self._xml_id_counter: int = 1 # CamBam IDs usually start from 1

    def _assign_xml_ids(self):
        """Assigns sequential integer IDs to all primitives for XML output."""
        self._uuid_to_xml_id = {}
        self._xml_id_counter = 1
        # Sort primitives for consistent ID assignment (optional, but good practice)
        # Sorting by creation time or UUID might be better if available/needed
        sorted_primitive_uuids = sorted(self.project._primitives.keys(), key=lambda u: u.int)

        for prim_uuid in sorted_primitive_uuids:
            self._uuid_to_xml_id[prim_uuid] = self._xml_id_counter
            self._xml_id_counter += 1
        logger.debug(f"Assigned XML IDs to {len(self._uuid_to_xml_id)} primitives.")


    def _build_xml_tree(self) -> ET.ElementTree:
        """Constructs the full XML ElementTree for the project."""
        # 1. Assign XML Primitive IDs first
        self._assign_xml_ids()

        # 2. Create Root Element
        # CamBam root element includes namespaces
        root = ET.Element("CADFile", {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xmlns:xsd": "http://www.w3.org/2001/XMLSchema",
            "Version": "0.9.8.0", # Use a known compatible version string
            "Name": self.project.project_name
        })

        # 3. Add Global Machining Options (minimal, often overridden by Parts)
        machining_options = ET.SubElement(root, "MachiningOptions")
        stock = ET.SubElement(machining_options, "Stock")
        ET.SubElement(stock, "Material") # Empty by default?
        ET.SubElement(stock, "PMin").text = "0,0,0" # Default global stock min
        ET.SubElement(stock, "PMax").text = "0,0,0" # Default global stock max
        ET.SubElement(stock, "Color").text = "255,165,0" # Default orange

        # 4. Build Layers structure
        layers_container = ET.SubElement(root, "layers")
        layer_objects_map: Dict[uuid.UUID, ET.Element] = {} # Map layer ID to its <objects> element
        for layer_uuid in self.project._layer_order:
            layer = self.project.get_layer(layer_uuid)
            if layer:
                try:
                    layer_elem = layer.to_xml_element()
                    layers_container.append(layer_elem)
                    # Store the reference to the <objects> sub-element created by to_xml_element
                    if layer._xml_objects_element is not None:
                         layer_objects_map[layer_uuid] = layer._xml_objects_element
                    else:
                         # Should not happen if to_xml_element works correctly
                         logger.error(f"Layer '{layer.user_identifier}' failed to create its <objects> XML reference.")
                except Exception as e:
                    logger.error(f"Error generating XML for Layer '{layer.user_identifier}': {e}", exc_info=True)
                    # Continue with other layers

        # 5. Build Primitives (placed in correct layer's <objects> tag)
        # Iterate in the order IDs were assigned for consistency
        for primitive_uuid, xml_primitive_id in self._uuid_to_xml_id.items():
            primitive = self.project.get_primitive(primitive_uuid)
            if not primitive:
                logger.warning(f"Primitive UUID {primitive_uuid} in ID map but not found in project during XML build.")
                continue

            try:
                # Primitive's method generates its specific XML element
                primitive_elem = primitive.to_xml_element(xml_primitive_id)

                # Find the correct layer's <objects> container
                layer_objects_container = layer_objects_map.get(primitive.layer_id)
                if layer_objects_container is not None:
                    layer_objects_container.append(primitive_elem)
                else:
                    logger.error(f"Layer {primitive.layer_id} <objects> container not found for primitive '{primitive.user_identifier}'. Appending to first layer as fallback.")
                    # Fallback: Append to the first layer found (if any)
                    if layer_objects_map:
                         first_layer_objects = next(iter(layer_objects_map.values()))
                         first_layer_objects.append(primitive_elem)
                    else:
                         logger.error("No valid layer containers found at all. Primitive XML cannot be placed.")

            except Exception as e:
                logger.error(f"Error generating XML for Primitive '{primitive.user_identifier}': {e}", exc_info=True)
                # Continue with other primitives

        # 6. Build Parts structure
        parts_container = ET.SubElement(root, "parts")
        part_machineops_map: Dict[uuid.UUID, ET.Element] = {} # Map part ID to its <machineops> element
        for part_uuid in self.project._part_order:
             part = self.project.get_part(part_uuid)
             if part:
                 try:
                     part_elem = part.to_xml_element()
                     parts_container.append(part_elem)
                     # Store the reference to the <machineops> sub-element
                     if part._xml_machineops_element is not None:
                         part_machineops_map[part_uuid] = part._xml_machineops_element
                     else:
                         logger.error(f"Part '{part.user_identifier}' failed to create its <machineops> XML reference.")
                 except Exception as e:
                     logger.error(f"Error generating XML for Part '{part.user_identifier}': {e}", exc_info=True)


        # 7. Build MOPs (placed in correct part's <machineops> tag)
        resolver = XmlPrimitiveIdResolver(self._uuid_to_xml_id, self.project._primitive_groups, self.project._primitives)
        for part_uuid in self.project._part_order:
            part = self.project.get_part(part_uuid)
            if not part: continue # Should not happen if order list is correct

            machineops_container = part_machineops_map.get(part_uuid)
            if machineops_container is None:
                logger.error(f"<machineops> container not found for part '{part.user_identifier}'. Cannot add MOPs.")
                continue

            # Iterate MOPs in the order defined within the part
            for mop_uuid in part.mop_ids:
                mop = self.project.get_mop(mop_uuid)
                if not mop:
                    logger.warning(f"MOP {mop_uuid} listed in part '{part.user_identifier}' order not found in project.")
                    continue

                try:
                    # Resolve primitive UUIDs/group to XML integer IDs
                    mop._resolved_xml_primitive_ids = resolver.resolve(mop.pid_source)
                    if not mop._resolved_xml_primitive_ids and mop.pid_source:
                         logger.warning(f"MOP '{mop.name}' ({mop.user_identifier}) resolved to zero primitives for source: {mop.pid_source}")

                    # Generate MOP XML using its specific method
                    mop_elem = mop.to_xml_element(self.project)
                    machineops_container.append(mop_elem)

                except Exception as e:
                    logger.error(f"Error generating XML for MOP '{mop.user_identifier}': {e}", exc_info=True)

        return ET.ElementTree(root)


    def save(self, file_path: str, pretty_print: bool = True) -> None:
        """Builds the XML tree and saves it to the specified file path."""
        logger.info(f"Building CamBam XML for project '{self.project.project_name}'...")
        try:
            # Ensure filename ends with .cb
            base, _ = os.path.splitext(file_path)
            output_path = base + '.cb'

            dir_name = os.path.dirname(output_path)
            if dir_name and not os.path.isdir(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                logger.info(f"Created directory: {dir_name}")

            # Build the XML structure
            tree = self._build_xml_tree()
            logger.info("XML tree built successfully.")

            # Optional pretty printing (requires Python 3.9+)
            if pretty_print:
                try:
                    ET.indent(tree, space="  ", level=0)
                    logger.debug("XML pretty printing applied.")
                except AttributeError:
                    logger.warning("XML pretty-printing (indent) requires Python 3.9+. Skipping.")
                except Exception as e:
                    # Catch other potential indent errors
                    logger.warning(f"XML indenting failed: {e}. Saving without indentation.")

            # Write the XML file
            tree.write(output_path, encoding='utf-8', xml_declaration=True, short_empty_elements=False)
            logger.info(f"CamBam file saved successfully: {output_path}")

        except Exception as e:
            logger.error(f"Error saving CamBam file to {output_path}: {e}", exc_info=True)
            raise XmlWritingError(f"Failed to write XML file: {e}")