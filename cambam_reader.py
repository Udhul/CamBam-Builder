# cambam_reader.py
import xml.etree.ElementTree as ET
import os
import uuid
import logging
from typing import Dict, List, Tuple, Optional, Union, Type, Any

# Module imports
from cambam_project import CamBamProject
from cambam_entities import (
    CamBamEntity, Layer, Primitive, Pline, Circle, Rect, Arc, Points, Text, 
    Part, Mop, ProfileMop, PocketMop, EngraveMop, DrillMop,
    PRIMITIVE_TAG_MAP, MOP_TAG_MAP # Maps XML tag name to class
)
from cad_transformations import cambam_string_to_matrix, identity_matrix
from cad_common import CamBamError, XmlParsingError

logger = logging.getLogger(__name__)

class CamBamReader:
    """ Handles reading a CamBam XML file and constructing a CamBamProject object. """

    def __init__(self):
        self._project: Optional[CamBamProject] = None
        self._xml_id_to_primitive: Dict[int, Primitive] = {} # Map XML int ID -> Primitive object
        self._parsed_layers: Dict[str, Layer] = {} # name -> Layer
        self._parsed_parts: Dict[str, Part] = {} # name -> Part
        self._mop_primitive_refs: Dict[Mop, List[int]] = {} # Mop -> List[XML Prim ID]

    def _parse_float(self, value_str: Optional[str], default: float = 0.0) -> float:
        """ Safely parse a float from string. """
        if value_str is None: return default
        try:
            return float(value_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse float from '{value_str}', using default {default}.")
            return default

    def _parse_int(self, value_str: Optional[str], default: int = 0) -> int:
        """ Safely parse an integer from string. """
        if value_str is None: return default
        try:
            # Handle potential floats like "1.0"
            return int(float(value_str))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse int from '{value_str}', using default {default}.")
            return default

    def _parse_bool(self, value_str: Optional[str], default: bool = False) -> bool:
        """ Safely parse a boolean from string ('true'/'false'). """
        if value_str is None: return default
        return value_str.strip().lower() == 'true'

    def _parse_point_2d(self, value_str: Optional[str], default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
        """ Safely parse a 2D point "x,y" or "x,y,z". """
        if value_str is None: return default
        try:
            parts = [p.strip() for p in value_str.split(',')]
            if len(parts) >= 2:
                return (float(parts[0]), float(parts[1]))
        except (ValueError, TypeError, IndexError):
             pass # Fall through to warning
        logger.warning(f"Could not parse 2D point from '{value_str}', using default {default}.")
        return default

    def _parse_color(self, value_str: Optional[str], default: str = "0,0,0") -> str:
         """ Parses color string, expecting 'R,G,B'. """
         if value_str is None: return default
         parts = [p.strip() for p in value_str.split(',')]
         if len(parts) == 3:
             try:
                 # Validate they are numbers, but return the string format
                 int(parts[0]), int(parts[1]), int(parts[2])
                 return value_str
             except (ValueError, TypeError):
                 pass
         logger.warning(f"Could not parse color 'R,G,B' from '{value_str}', using default {default}.")
         return default

    def _get_sub_element_text(self, parent_element: ET.Element, tag_name: str, default: Optional[str] = None) -> Optional[str]:
        """ Finds a sub-element and returns its text content. """
        sub_element = parent_element.find(tag_name)
        if sub_element is not None and sub_element.text is not None:
             return sub_element.text.strip()
        return default


    def _parse_layers(self, root: ET.Element):
        """ Parses <layers> section and creates Layer objects. """
        layers_container = root.find("layers")
        if layers_container is None:
            logger.warning("No <layers> section found in XML.")
            return

        for layer_elem in layers_container.findall("layer"):
            name = layer_elem.get("name", f"Layer_{uuid.uuid4().hex[:6]}")
            layer = Layer(
                user_identifier=name,
                color=self._parse_color(layer_elem.get("color"), default="0,255,0"), # Default Green
                alpha=self._parse_float(layer_elem.get("alpha"), default=1.0),
                pen_width=self._parse_float(layer_elem.get("pen"), default=1.0),
                visible=self._parse_bool(layer_elem.get("visible"), default=True),
                locked=self._parse_bool(layer_elem.get("locked"), default=False),
            )
            self._project.add_layer(layer) # Adds to project and order list
            self._parsed_layers[name] = layer # Store for primitive assignment

            # Now parse primitives within this layer's <objects> tag
            objects_container = layer_elem.find("objects")
            if objects_container is not None:
                self._parse_primitives(objects_container, layer)


    def _parse_primitives(self, objects_container: ET.Element, layer: Layer):
        """ Parses primitive elements within a layer's <objects> tag. """
        for elem in objects_container: # Iterate direct children
            tag_name = elem.tag
            if tag_name in PRIMITIVE_TAG_MAP:
                PrimitiveClass = PRIMITIVE_TAG_MAP[tag_name]
                try:
                    primitive = self._parse_single_primitive(elem, PrimitiveClass, layer)
                    if primitive:
                         # add_primitive handles linking primitive to layer etc.
                         self._project.add_primitive(primitive)
                         xml_id = self._parse_int(elem.get("id"), default=-1)
                         if xml_id != -1:
                             self._xml_id_to_primitive[xml_id] = primitive
                         else:
                             logger.warning(f"Primitive element <{tag_name}> missing 'id' attribute.")
                except Exception as e:
                    logger.error(f"Failed to parse primitive element <{tag_name}>: {e}", exc_info=True)
            # else: # Handle other object types if necessary (e.g., groups?)


    def _parse_single_primitive(self, elem: ET.Element, PrimitiveClass: Type[Primitive], layer: Layer) -> Optional[Primitive]:
        """ Parses common attributes and delegates to specific primitive parsers. """
        # Common attributes
        xml_id_str = elem.get("id")
        if not xml_id_str: return None # Skip primitives without ID? Or assign one? Skip for now.

        # Parse Tag for groups and user tags
        groups = []
        user_tags = ""
        tag_elem = elem.find("Tag")
        if tag_elem is not None and tag_elem.text:
             lines = tag_elem.text.strip().split('\n', 1)
             group_line = lines[0].strip()
             if group_line.startswith('[') and group_line.endswith(']'):
                 group_content = group_line[1:-1].strip()
                 if group_content: # Avoid empty string group
                      groups = [g.strip() for g in group_content.split(',')]
             if len(lines) > 1:
                 user_tags = lines[1].strip()

        # Parse Matrix
        effective_transform = identity_matrix()
        mat_elem = elem.find("mat")
        if mat_elem is not None:
             matrix_str = mat_elem.get("m")
             if matrix_str:
                 try:
                     effective_transform = cambam_string_to_matrix(matrix_str)
                 except Exception as e:
                      logger.warning(f"Failed to parse matrix for primitive id {xml_id_str}: {e}. Using identity.")

        # Primitive specific parsing
        kwargs: Dict[str, Any] = {
            "layer_id": layer.internal_id,
            "groups": groups,
            "tags": user_tags,
            "effective_transform": effective_transform,
            # parent_primitive_id cannot be determined from standard CamBam XML
            "parent_primitive_id": None,
        }

        # --- Delegate to specific parsers based on class ---
        if PrimitiveClass is Pline:
            pts = []
            pts_elem = elem.find("pts")
            if pts_elem is not None:
                for p_elem in pts_elem.findall("p"):
                     coord = self._parse_point_2d(p_elem.text)
                     bulge = self._parse_float(p_elem.get("b"), default=0.0)
                     pts.append((coord[0], coord[1], bulge))
            kwargs["relative_points"] = pts
            kwargs["closed"] = self._parse_bool(elem.get("Closed"), default=False)

        elif PrimitiveClass is Circle:
            kwargs["relative_center"] = self._parse_point_2d(elem.get("c"))
            kwargs["diameter"] = self._parse_float(elem.get("d"), default=1.0)

        elif PrimitiveClass is Rect:
            kwargs["relative_corner"] = self._parse_point_2d(elem.get("p"))
            kwargs["width"] = self._parse_float(elem.get("w"), default=1.0)
            kwargs["height"] = self._parse_float(elem.get("h"), default=1.0)
            # Note: Ignores "Closed" attribute as Rect is implicitly closed

        elif PrimitiveClass is Arc:
             kwargs["relative_center"] = self._parse_point_2d(elem.get("p"))
             kwargs["radius"] = self._parse_float(elem.get("r"), default=1.0)
             kwargs["start_angle"] = self._parse_float(elem.get("s"), default=0.0)
             kwargs["extent_angle"] = self._parse_float(elem.get("w"), default=90.0)

        elif PrimitiveClass is Points:
            pts = []
            pts_elem = elem.find("pts")
            if pts_elem is not None:
                 for p_elem in pts_elem.findall("p"):
                     pts.append(self._parse_point_2d(p_elem.text))
            kwargs["relative_points"] = pts

        elif PrimitiveClass is Text:
            # CamBam uses p1, p2 - often same. Use p1 as position.
            kwargs["relative_position"] = self._parse_point_2d(elem.get("p1"))
            kwargs["text_content"] = elem.text.strip() if elem.text else ""
            kwargs["height"] = self._parse_float(elem.get("Height"), default=10.0)
            kwargs["font"] = elem.get("Font", "Arial")
            kwargs["style"] = elem.get("style", "")
            kwargs["line_spacing"] = self._parse_float(elem.get("linespace"), default=1.0)
            align_str = elem.get("align", "middle,center")
            try:
                 v_align, h_align = align_str.split(',')
                 # Map Cambam align names to our names if different (seems same here)
                 h_align_map = {"left": "Left", "center": "Center", "right": "Right"}
                 v_align_map = {"top": "Top", "middle": "Middle", "bottom": "Bottom"}
                 kwargs["align_horizontal"] = h_align_map.get(h_align.lower(), "Center")
                 kwargs["align_vertical"] = v_align_map.get(v_align.lower(), "Middle")
            except Exception:
                 logger.warning(f"Could not parse text align string '{align_str}'. Using defaults.")
                 kwargs["align_horizontal"] = "Center"
                 kwargs["align_vertical"] = "Middle"

        else:
            logger.error(f"No specific XML parser implemented for primitive type: {PrimitiveClass.__name__}")
            return None

        # Use user_identifier from Tag if available, otherwise generate default
        # This requires parsing the Tag first. For now, let post_init generate default.
        primitive = PrimitiveClass(**kwargs)
        # We could try to extract a user_identifier from the Tag here if needed

        return primitive


    def _parse_parts(self, root: ET.Element):
        """ Parses <parts> section and creates Part objects. """
        parts_container = root.find("parts")
        if parts_container is None:
             logger.warning("No <parts> section found in XML.")
             return

        for part_elem in parts_container.findall("part"):
            name = self._get_sub_element_text(part_elem, "Name", f"Part_{uuid.uuid4().hex[:6]}")
            # Parse stock info (usually within <Stock> sub-element)
            stock_elem = part_elem.find("Stock")
            stock_thickness = 0.0
            stock_width = 0.0
            stock_height = 0.0
            stock_material = ""
            stock_color = "210,180,140" # Default Brown
            if stock_elem is not None:
                 pmin_str = self._get_sub_element_text(stock_elem, "PMin", "0,0,0")
                 pmax_str = self._get_sub_element_text(stock_elem, "PMax", "0,0,0")
                 stock_material = self._get_sub_element_text(stock_elem, "Material", "")
                 stock_color = self._parse_color(self._get_sub_element_text(stock_elem, "Color"), default=stock_color)
                 try:
                     # Thickness is abs(Z_min), Width is X_max, Height is Y_max (assuming min is 0,0,-T)
                     pmin_parts = [float(p) for p in pmin_str.split(',')]
                     pmax_parts = [float(p) for p in pmax_str.split(',')]
                     if len(pmin_parts) == 3: stock_thickness = abs(pmin_parts[2])
                     if len(pmax_parts) >= 1: stock_width = pmax_parts[0]
                     if len(pmax_parts) >= 2: stock_height = pmax_parts[1]
                 except Exception:
                     logger.warning(f"Could not parse stock dimensions for part '{name}'. Using defaults.")


            part = Part(
                user_identifier=name,
                enabled=self._parse_bool(part_elem.get("Enabled"), default=True),
                stock_thickness=stock_thickness,
                stock_width=stock_width,
                stock_height=stock_height,
                stock_material=stock_material,
                stock_color=stock_color,
                machining_origin=self._parse_point_2d(self._get_sub_element_text(part_elem, "MachiningOrigin"), default=(0.0,0.0)),
                default_tool_diameter=self._parse_float(self._get_sub_element_text(part_elem, "ToolDiameter"), default=None), # Allow None
                default_spindle_speed=self._parse_int(self._get_sub_element_text(part_elem, "SpindleSpeed"), default=None) # Allow None
            )
            self._project.add_part(part)
            self._parsed_parts[name] = part # Store for MOP assignment? MOPs link by ID usually.

            # Parse MOPs within this part's <machineops> tag
            mops_container = part_elem.find("machineops")
            if mops_container is not None:
                self._parse_mops(mops_container, part)


    def _parse_mops(self, mops_container: ET.Element, part: Part):
        """ Parses MOP elements within a part's <machineops> tag. """
        for elem in mops_container:
            tag_name = elem.tag
            if tag_name in MOP_TAG_MAP:
                MopClass = MOP_TAG_MAP[tag_name]
                try:
                    mop = self._parse_single_mop(elem, MopClass, part)
                    if mop:
                         self._project.add_mop(mop) # Adds to project and part's order list
                except Exception as e:
                    logger.error(f"Failed to parse MOP element <{tag_name}>: {e}", exc_info=True)

    def _parse_single_mop(self, elem: ET.Element, MopClass: Type[Mop], part: Part) -> Optional[Mop]:
        """ Parses common MOP attributes and delegates to specific MOP parsers. """

        # Common parameters from sub-elements
        name = self._get_sub_element_text(elem, "Name", f"{MopClass.__name__}_{uuid.uuid4().hex[:6]}")
        enabled = self._parse_bool(elem.get("Enabled"), default=True)

        # Helper to get value respecting 'state' attribute (Value, Default, Calculated...)
        def get_mop_param(tag: str, parse_func, default: Any = None, required: bool = False) -> Any:
            sub_elem = elem.find(tag)
            if sub_elem is not None:
                 # state = sub_elem.get("state", "Value") # Could use state if needed
                 return parse_func(sub_elem.text, default=default)
            if required:
                 logger.warning(f"Required MOP parameter '{tag}' not found for MOP '{name}'. Using default {default}.")
            return default

        # Parse primitive source IDs (integers) - store temporarily
        primitive_ids_xml: List[int] = []
        primitive_container = elem.find("primitive")
        if primitive_container is not None:
            for prim_elem in primitive_container.findall("prim"):
                try:
                    primitive_ids_xml.append(int(prim_elem.text))
                except (ValueError, TypeError):
                     logger.warning(f"Invalid primitive ID '{prim_elem.text}' found in MOP '{name}'.")

        # Store primitive IDs for later resolution to UUIDs
        # pid_source will be set after all primitives are parsed
        temp_pid_source_placeholder = primitive_ids_xml # Store XML IDs for now

        kwargs: Dict[str, Any] = {
            "part_id": part.internal_id,
            "name": name,
            "pid_source": temp_pid_source_placeholder, # Store XML IDs temporarily
            "enabled": enabled,
            # Parse common params
            "target_depth": get_mop_param("TargetDepth", self._parse_float, default=None),
            "depth_increment": get_mop_param("DepthIncrement", self._parse_float, default=None),
            "stock_surface": get_mop_param("StockSurface", self._parse_float, default=0.0),
            "roughing_clearance": get_mop_param("RoughingClearance", self._parse_float, default=0.0),
            "clearance_plane": get_mop_param("ClearancePlane", self._parse_float, default=15.0),
            "spindle_direction": get_mop_param("SpindleDirection", str, default='CW'),
            "spindle_speed": get_mop_param("SpindleSpeed", self._parse_int, default=None),
            "velocity_mode": get_mop_param("VelocityMode", str, default='ExactStop'),
            "work_plane": get_mop_param("WorkPlane", str, default='XY'),
            "optimisation_mode": get_mop_param("OptimisationMode", str, default='Standard'),
            "tool_diameter": get_mop_param("ToolDiameter", self._parse_float, default=None, required=True),
            "tool_number": get_mop_param("ToolNumber", self._parse_int, default=0),
            "tool_profile": get_mop_param("ToolProfile", str, default='EndMill'),
            "plunge_feedrate": get_mop_param("PlungeFeedrate", self._parse_float, default=1000.0),
            "cut_feedrate": get_mop_param("CutFeedrate", self._parse_float, default=None),
            "max_crossover_distance": get_mop_param("MaxCrossoverDistance", self._parse_float, default=0.7),
            "custom_mop_header": get_mop_param("CustomMOPHeader", str, default=""),
            "custom_mop_footer": get_mop_param("CustomMOPFooter", str, default=""),
        }

        # --- Delegate to specific MOP parsers ---
        if MopClass is ProfileMop:
            kwargs["stepover"] = get_mop_param("StepOver", self._parse_float, default=0.4)
            kwargs["profile_side"] = get_mop_param("InsideOutside", str, default='Inside')
            kwargs["milling_direction"] = get_mop_param("MillingDirection", str, default='Conventional')
            kwargs["collision_detection"] = get_mop_param("CollisionDetection", self._parse_bool, default=True)
            kwargs["corner_overcut"] = get_mop_param("CornerOvercut", self._parse_bool, default=False)
            # Parse lead-in (assuming LeadInMove holds relevant info)
            lead_in_elem = elem.find("LeadInMove")
            if lead_in_elem:
                 kwargs["lead_in_type"] = self._get_sub_element_text(lead_in_elem, "LeadInType", "Spiral")
                 kwargs["lead_in_spiral_angle"] = self._parse_float(self._get_sub_element_text(lead_in_elem, "SpiralAngle"), 30.0)
                 kwargs["lead_in_tangent_radius"] = self._parse_float(self._get_sub_element_text(lead_in_elem, "TangentRadius"), 0.0)
            else: kwargs["lead_in_type"] = None

            kwargs["final_depth_increment"] = get_mop_param("FinalDepthIncrement", self._parse_float, default=None) # 0.0 means no pass
            kwargs["cut_ordering"] = get_mop_param("CutOrdering", str, default='DepthFirst')
            # Parse Tabs
            tabs_elem = elem.find("HoldingTabs")
            if tabs_elem:
                 kwargs["tab_method"] = self._get_sub_element_text(tabs_elem, "TabMethod", "None")
                 kwargs["tab_width"] = self._parse_float(self._get_sub_element_text(tabs_elem, "Width"), 6.0)
                 kwargs["tab_height"] = self._parse_float(self._get_sub_element_text(tabs_elem, "Height"), 1.5)
                 kwargs["tab_min_tabs"] = self._parse_int(self._get_sub_element_text(tabs_elem, "MinimumTabs"), 3)
                 kwargs["tab_max_tabs"] = self._parse_int(self._get_sub_element_text(tabs_elem, "MaximumTabs"), 3)
                 kwargs["tab_distance"] = self._parse_float(self._get_sub_element_text(tabs_elem, "TabDistance"), 40.0)
                 kwargs["tab_size_threshold"] = self._parse_float(self._get_sub_element_text(tabs_elem, "SizeThreshold"), 4.0)
                 kwargs["tab_use_leadins"] = self._parse_bool(self._get_sub_element_text(tabs_elem, "UseLeadIns"), False)
                 kwargs["tab_style"] = self._get_sub_element_text(tabs_elem, "TabStyle", "Square")

        elif MopClass is PocketMop:
            kwargs["stepover"] = get_mop_param("StepOver", self._parse_float, default=0.4)
            kwargs["stepover_feedrate"] = get_mop_param("StepoverFeedrate", str, default='Plunge Feedrate')
            kwargs["milling_direction"] = get_mop_param("MillingDirection", str, default='Conventional')
            kwargs["collision_detection"] = get_mop_param("CollisionDetection", self._parse_bool, default=True)
            lead_in_elem = elem.find("LeadInMove")
            if lead_in_elem:
                 kwargs["lead_in_type"] = self._get_sub_element_text(lead_in_elem, "LeadInType", "Spiral")
                 kwargs["lead_in_spiral_angle"] = self._parse_float(self._get_sub_element_text(lead_in_elem, "SpiralAngle"), 30.0)
                 kwargs["lead_in_tangent_radius"] = self._parse_float(self._get_sub_element_text(lead_in_elem, "TangentRadius"), 0.0)
            else: kwargs["lead_in_type"] = None
            kwargs["final_depth_increment"] = get_mop_param("FinalDepthIncrement", self._parse_float, default=None)
            kwargs["cut_ordering"] = get_mop_param("CutOrdering", str, default='DepthFirst')
            kwargs["region_fill_style"] = get_mop_param("RegionFillStyle", str, default='InsideOutsideOffsets')
            kwargs["finish_stepover"] = get_mop_param("FinishStepover", self._parse_float, default=0.0)
            kwargs["finish_stepover_at_target_depth"] = get_mop_param("FinishStepoverAtTargetDepth", self._parse_bool, default=False)
            kwargs["roughing_finishing"] = get_mop_param("RoughingFinishing", str, default='Roughing')

        elif MopClass is EngraveMop:
            kwargs["roughing_finishing"] = get_mop_param("RoughingFinishing", str, default='Roughing')
            kwargs["final_depth_increment"] = get_mop_param("FinalDepthIncrement", self._parse_float, default=None)
            kwargs["cut_ordering"] = get_mop_param("CutOrdering", str, default='DepthFirst')

        elif MopClass is DrillMop:
            kwargs["drilling_method"] = get_mop_param("DrillingMethod", str, default='CannedCycle')
            kwargs["hole_diameter"] = get_mop_param("HoleDiameter", self._parse_float, default=None)
            kwargs["drill_lead_out"] = get_mop_param("DrillLeadOut", self._parse_bool, default=False)
            kwargs["spiral_flat_base"] = get_mop_param("SpiralFlatBase", self._parse_bool, default=True)
            kwargs["lead_out_length"] = get_mop_param("LeadOutLength", self._parse_float, default=0.0)
            kwargs["peck_distance"] = get_mop_param("PeckDistance", self._parse_float, default=0.0)
            kwargs["retract_height"] = get_mop_param("RetractHeight", self._parse_float, default=5.0)
            kwargs["dwell"] = get_mop_param("Dwell", self._parse_float, default=0.0)
            kwargs["custom_script"] = get_mop_param("CustomScript", str, default="")

        else:
            logger.error(f"No specific XML parser implemented for MOP type: {MopClass.__name__}")
            return None

        mop = MopClass(**kwargs)
        # Store the XML primitive IDs temporarily on the mop object
        self._mop_primitive_refs[mop] = primitive_ids_xml
        return mop


    def _resolve_mop_primitive_refs(self):
        """ Converts stored XML primitive IDs on MOPs to UUID lists. """
        logger.debug(f"Resolving MOP primitive references using {len(self._xml_id_to_primitive)} parsed primitives.")
        for mop, xml_ids in self._mop_primitive_refs.items():
            resolved_uuids: List[uuid.UUID] = []
            for xml_id in xml_ids:
                primitive = self._xml_id_to_primitive.get(xml_id)
                if primitive:
                    resolved_uuids.append(primitive.internal_id)
                else:
                    logger.warning(f"MOP '{mop.user_identifier}' references unknown primitive XML ID {xml_id}. Skipping.")
            # Set the final pid_source on the MOP object
            # Note: This assumes all MOPs loaded from XML used explicit IDs, not group names.
            # If group names need to be reconstructed from Tags, that logic would go here.
            mop.pid_source = resolved_uuids
            logger.debug(f"Resolved {len(resolved_uuids)} primitive refs for MOP '{mop.user_identifier}'.")


    def load(self, file_path: str) -> CamBamProject:
        """Loads the CamBam project from an XML file."""
        logger.info(f"Loading CamBam project from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CamBam file not found: {file_path}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            if root.tag != "CADFile":
                 raise XmlParsingError("Root element is not <CADFile>.")

            project_name = root.get("Name", "Loaded Project")
            # Read global defaults if needed (e.g., ToolDiameter)
            global_tool_dia = None
            mach_opts = root.find("MachiningOptions")
            if mach_opts is not None:
                 td_elem = mach_opts.find("ToolDiameter")
                 if td_elem is not None: global_tool_dia = self._parse_float(td_elem.text)


            # Initialize project
            self._project = CamBamProject(project_name, default_tool_diameter=global_tool_dia)
            self._xml_id_to_primitive = {}
            self._parsed_layers = {}
            self._parsed_parts = {}
            self._mop_primitive_refs = {}

            # Parse in order: Layers (contains Primitives), then Parts (contains MOPs)
            self._parse_layers(root)
            self._parse_parts(root)

            # After parsing everything, resolve MOP primitive ID references
            self._resolve_mop_primitive_refs()

            logger.info(f"Successfully loaded project '{project_name}' with "
                        f"{len(self._project.list_layers())} layers, "
                        f"{len(self._project.list_primitives())} primitives, "
                        f"{len(self._project.list_parts())} parts, "
                        f"{len(self._project.list_mops())} MOPs.")
            return self._project

        except ET.ParseError as e:
            logger.error(f"XML parsing error in {file_path}: {e}", exc_info=True)
            raise XmlParsingError(f"Invalid XML structure: {e}")
        except Exception as e:
            logger.error(f"Error loading CamBam file {file_path}: {e}", exc_info=True)
            raise CamBamError(f"Failed to load CamBam file: {e}")