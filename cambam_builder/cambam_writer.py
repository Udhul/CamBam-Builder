"""
cambam_writer.py

Provides functions to serialize a CamBamProject instance into the CamBam XML file format.
It queries the project's registries to correctly structure layers, parts, primitives,
MOPs, and their relationships according to the CamBam schema.
"""

import xml.etree.ElementTree as ET
import io
import os
import logging
import uuid
import tempfile
from copy import deepcopy
from typing import Dict, List

from .cambam_project import CamBamProject # Use Type Hinting
from .entity_core import Primitive
from .cad_entities import Layer
from .cam_entities import Part, Mop

logger = logging.getLogger(__name__)


def _parse_native_float(value, default):
    """Parse the numeric spelling used by the reader for imported nesting."""
    if value is None:
        return default
    try:
        return float(value.strip())
    except (AttributeError, TypeError, ValueError):
        return default


def _parse_native_int(value, default):
    """Parse integer nesting values, including CamBam's occasional ``1.0``."""
    if value is None:
        return default
    try:
        return int(float(value.strip()))
    except (AttributeError, TypeError, ValueError):
        return default


def _parse_native_bool(value, default=False):
    if value is None:
        return default
    return value.strip().lower() == "true"


def _nesting_matches_model(part: Part, nesting: ET.Element) -> bool:
    """Return whether an imported native subtree still represents the model.

    Imported parts retain the complete native ``Nesting`` element so unknown
    attributes/children and its position can round-trip.  Once a modeled
    nesting field is changed, the generated element must replace that native
    subtree; otherwise a later export would silently discard the requested
    edit.  Compare parsed values using the same defaults/conversions as the
    reader, rather than comparing XML text formatting.
    """
    method = nesting.findtext("NestMethod", "None")
    native = (method,)
    modeled = (part.nesting_method,)
    if method in ("Grid", "IsoGrid"):
        alternate = nesting.findtext(
            "GridDirectionAlternate", nesting.findtext("GridAlternate")
        )
        native += (
            _parse_native_int(nesting.findtext("Rows"), 1),
            _parse_native_int(nesting.findtext("Columns"), 1),
            _parse_native_float(nesting.findtext("Spacing"), 0.0),
            nesting.findtext("GridOrder", "RightUp"),
            _parse_native_bool(alternate, False),
        )
        modeled += (
            part.nesting_rows,
            part.nesting_columns,
            part.nesting_spacing,
            part.nesting_grid_order,
            part.nesting_grid_alternate,
        )
    return native == modeled


def _nesting_is_default(part: Part) -> bool:
    """Whether an imported part with no native Nesting remains untouched."""
    return (
        part.nesting_method == "None"
        and part.nesting_rows == 1
        and part.nesting_columns == 1
        and part.nesting_spacing == 0.0
        and part.nesting_grid_order == "RightUp"
        and part.nesting_grid_alternate is False
    )


def _merge_native_nesting(part: Part, native: ET.Element,
                          generated: ET.Element) -> ET.Element:
    """Update modeled nesting values without dropping native-only settings."""
    if _nesting_matches_model(part, native):
        return deepcopy(native)
    if native.findtext("NestMethod", "None") != part.nesting_method:
        # Placement data belongs to its method. In particular, PointListID and
        # ManualItems must not leak into a newly selected Grid/None method.
        return deepcopy(generated)
    merged = deepcopy(native)
    for tag in ("NestMethod", "Rows", "Columns", "Spacing", "GridOrder",
                "GridDirectionAlternate"):
        source = generated.find(tag)
        if source is None:
            continue
        target = merged.find(tag)
        if target is None:
            target = ET.SubElement(merged, tag)
        target.text = source.text
    # The old alias must not override the explicitly updated canonical field.
    for alias in list(merged.findall("GridAlternate")):
        merged.remove(alias)
    return merged


def _stock_matches_model(part: Part, stock: ET.Element) -> bool:
    pmin = stock.findtext("PMin")
    pmax = stock.findtext("PMax")
    try:
        min_values = tuple(float(value.strip()) for value in pmin.split(","))
        max_values = tuple(float(value.strip()) for value in pmax.split(","))
    except (AttributeError, TypeError, ValueError):
        return False
    if len(min_values) != 3 or len(max_values) != 3:
        return False
    native = (
        abs(max_values[0] - min_values[0]),
        abs(max_values[1] - min_values[1]),
        abs(max_values[2] - min_values[2]),
        min(min_values[0], max_values[0]),
        min(min_values[1], max_values[1]),
        max(min_values[2], max_values[2]),
        stock.findtext("Material", "Default"),
        stock.findtext("Color", "210,180,140"),
    )
    modeled = (
        part.stock_width, part.stock_height, part.stock_thickness,
        part.stock_offset[0], part.stock_offset[1], part.stock_surface,
        part.stock_material, part.stock_color,
    )
    return native == modeled


def build_xml_tree(project: CamBamProject) -> ET.ElementTree:
    """Construct the XML tree, propagating encoding errors without omitting entities."""

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
        # CamBam Plus 1.0 itself continues to emit this legacy file marker.
        # It does not identify the installed application version.
        "Version": "0.9.8.0",
        "Name": project.project_name
    })

    # Retain imported machining context, including CAM style inheritance.
    # The framework does not evaluate external CamBam style libraries.
    if hasattr(project, "_xml_machining_options"):
        if project._xml_machining_options is not None:
            root.append(deepcopy(project._xml_machining_options))
    else:
        machining_options = ET.SubElement(root, "MachiningOptions")
        stock = ET.SubElement(machining_options, "Stock")
        ET.SubElement(stock, "Material")
        ET.SubElement(stock, "PMin").text = "0,0,0"
        ET.SubElement(stock, "PMax").text = "0,0,0"
        ET.SubElement(stock, "Color").text = "255,165,0"

    # 4. Build <layers> container
    layers_container = ET.SubElement(root, "layers")
    # Iterate through layers *in order* defined by the project
    for layer_uuid in project._layer_order:
        layer = project.get_layer(layer_uuid)
        if not layer:
            raise ValueError(f"Layer UUID {layer_uuid} found in order list but not in registry.")

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
                raise ValueError(f"Primitive {primitive.user_identifier} ({primitive.internal_id}) has no assigned XML ID.")

            # Get parent UUID from project registry to inject into the Tag
            parent = project.get_parent_of_primitive(primitive.internal_id)
            parent_uuid = parent.internal_id if parent is not None else None

            try:
                # Generate the primitive's specific XML element (<pline>, <circle>, etc.)
                # Pass the required XML ID and parent UUID
                prim_elem = primitive.to_xml_element(xml_id, parent_uuid)
                objects_container.append(prim_elem)


            except Exception as e:
                logger.error(f"Error building XML for primitive {primitive.user_identifier} ({primitive.internal_id}): {e}", exc_info=True)
                raise


    # 5. Build <parts> container
    parts_container = ET.SubElement(root, "parts")
    # Iterate through parts *in order* defined by the project
    for part_uuid in project._part_order:
        part = project.get_part(part_uuid)
        if not part:
            raise ValueError(f"Part UUID {part_uuid} found in order list but not in registry.")

        # Create the <part> element itself
        part_elem = part.to_xml_element()
        native_stock = getattr(part, "_xml_stock", None)
        if native_stock is not None and _stock_matches_model(part, native_stock):
            generated_stock = part_elem.find("Stock")
            stock_index = list(part_elem).index(generated_stock)
            part_elem.remove(generated_stock)
            part_elem.insert(stock_index, deepcopy(native_stock))
        if hasattr(part, "_xml_machining_parameters"):
            # ``to_xml_element`` creates a fresh, model-backed Nesting node.
            # Keep it available while restoring native children: the native
            # subtree is preferred only while its modeled values are unchanged.
            generated_nesting = part_elem.find("Nesting")
            native_nesting = getattr(part, "_xml_nesting", None)
            preserve_native_nesting = (
                native_nesting is not None
                and _nesting_matches_model(part, native_nesting)
            )
            emitted_nesting = False
            for child in list(part_elem):
                if child.tag not in {"Stock", "MachiningOrigin", "ToolDiameter"}:
                    part_elem.remove(child)
            for child in part._xml_machining_parameters:
                if child.tag == "Nesting":
                    nesting = (
                        deepcopy(native_nesting) if preserve_native_nesting
                        else _merge_native_nesting(part, native_nesting, generated_nesting)
                        if native_nesting is not None and generated_nesting is not None
                        else generated_nesting
                    )
                    if nesting is not None:
                        part_elem.append(deepcopy(nesting))
                    emitted_nesting = True
                else:
                    part_elem.append(deepcopy(child))
            # A malformed/older imported file may not contain a Nesting node.
            # Preserve that shape until a modeled nesting field is changed; a
            # subsequent explicit edit then gets a usable generated node.
            if not emitted_nesting and not _nesting_is_default(part):
                if generated_nesting is not None:
                    part_elem.append(deepcopy(generated_nesting))
        if (hasattr(part, "_xml_tool_diameter_value")
                and part.default_tool_diameter == part._xml_tool_diameter_value):
            insertion_index = 2  # After the modeled Stock and MachiningOrigin.
            for child in list(part_elem.findall("ToolDiameter")):
                insertion_index = list(part_elem).index(child)
                part_elem.remove(child)
            if part._xml_tool_diameter is not None:
                part_elem.insert(insertion_index, deepcopy(part._xml_tool_diameter))
        parts_container.append(part_elem)

        # Create the <machineops> container within this part
        machineops_container = ET.SubElement(part_elem, "machineops")

        # Get MOPs assigned to this part from the project registry, in order
        mops_in_part = project.get_mops_in_part(part_uuid)

        # Add MOP XML elements to this part's <machineops> container
        for mop in mops_in_part:
            # Resolve project-owned targets to primitive XML IDs
            try:
                primitive_uuids = project.get_mop_targets(mop)
                resolved_primitive_xml_ids: List[int] = []
                for prim_uuid in primitive_uuids:
                    xml_id = uuid_to_xml_id.get(prim_uuid)
                    if xml_id:
                        resolved_primitive_xml_ids.append(xml_id)
                    else:
                        # This case should be rare if target ownership is consistent
                        raise ValueError(f"MOP {mop.name} references primitive {prim_uuid} which has no XML ID assigned.")

                # Generate the MOP's specific XML element (<profile>, <pocket>, etc.)
                # Pass the project context and the resolved XML IDs
                mop_elem = mop.to_xml_element(project, resolved_primitive_xml_ids)
                machineops_container.append(mop_elem)

            except Exception as e:
                logger.error(f"Error building XML for MOP {mop.name} ({mop.user_identifier}): {e}", exc_info=True)
                raise


    # 6. Return the complete ElementTree
    return ET.ElementTree(root)


def serialize_cambam_bytes(project: CamBamProject, pretty_print: bool = True) -> bytes:
    """Serialize a project as complete UTF-8 CamBam XML bytes."""
    output_decimals = project.output_decimals
    for primitive in project._primitives.values():
        if isinstance(primitive, Primitive):
            primitive.output_decimals = output_decimals

    logger.info("Building XML tree for project '%s'...", project.project_name)
    tree = build_xml_tree(project)
    logger.info("XML tree built.")
    if pretty_print:
        indent = getattr(ET, "indent", None)
        if indent is not None:
            indent(tree, space="  ", level=0)
            logger.debug("XML pretty-printing applied.")
        else:
            logger.warning("XML pretty-printing (indentation) requires Python 3.9 or later.")

    stream = io.BytesIO()
    tree.write(
        stream, encoding="utf-8", xml_declaration=True, short_empty_elements=False
    )
    return stream.getvalue()


def save_cambam_file(project: CamBamProject, file_path: str, pretty_print: bool = True) -> None:
    """
    Builds the XML tree for the project and saves it to a .cb file.

    Errors propagate to the caller. A completed temporary file replaces the
    destination only after serialization and close succeed, preserving an
    existing destination on failure. This does not guarantee crash durability.

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

        data = serialize_cambam_bytes(project, pretty_print=pretty_print)

        # Keep the temporary file on the destination filesystem. Close it before
        # replacement, including on Windows where open files cannot be replaced.
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=output_dir or '.', prefix='.cambam-', suffix='.tmp', delete=False
            ) as temporary_file:
                temporary_path = temporary_file.name
                temporary_file.write(data)
            os.replace(temporary_path, output_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    # Preserve the original failure if cleanup also fails.
                    logger.warning("Could not remove temporary export file %s", temporary_path, exc_info=True)
        logger.info(f"CamBam file successfully saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save CamBam file to {file_path}: {e}", exc_info=True)
        raise # Re-raise the exception
