"""Part containers, machining operations, and their XML policies."""

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, TYPE_CHECKING

from .core import CamBamEntity

if TYPE_CHECKING:
    from .project import CamBamProject

logger = logging.getLogger(__name__)

@dataclass
class Part(CamBamEntity):
    """Represents a machining part with stock and default parameters.

    ``stock_offset`` is local to the Part's ``machining_origin``.  CamBam
    therefore draws the stock's lower-left top corner at
    ``machining_origin + stock_offset`` in XY, with ``stock_surface`` as Z.
    The XML PMin/PMax coordinates encode the local stock box, not that derived
    drawing position.
    ``stock_present`` records whether a Part Stock element exists. Imported
    stockless Parts retain their parsed placeholder dimensions without turning
    those values into stock on export.
    """
    enabled: bool = True
    stock_thickness: float = 12.5
    stock_width: float = 1220.0
    stock_height: float = 2440.0
    stock_material: str = "MDF"
    stock_color: str = "210,180,140" # RGB string
    machining_origin: Tuple[float, float] = (0.0, 0.0) # XY offset
    default_tool_diameter: Optional[float] = None
    default_spindle_speed: Optional[int] = None
    nesting_method: str = "None"
    nesting_rows: int = 1
    nesting_columns: int = 1
    nesting_spacing: float = 0.0
    nesting_grid_order: str = "RightUp"
    nesting_grid_alternate: bool = False
    stock_offset: Tuple[float, float] = (0.0, 0.0)
    stock_surface: float = 0.0
    stock_present: bool = True
    # Note: No mop_ids or _xml_machineops_element here. Managed by Project.

    @property
    def stock_drawing_origin(self) -> Tuple[float, float, float]:
        """Return the stock lower-left top corner in drawing coordinates."""
        return (
            self.machining_origin[0] + self.stock_offset[0],
            self.machining_origin[1] + self.stock_offset[1],
            self.stock_surface,
        )

    def to_xml_element(self) -> ET.Element:
        """Creates the <part> XML element (without the <machineops> container)."""
        part_elem = ET.Element("part", {
            "Name": self.user_identifier, # Use user_identifier as CamBam's part name
            "Enabled": str(self.enabled).lower()
        })
        # The <machineops> sub-element will be added by the writer based on project registry

        # Add stock and other part-level settings directly here
        if self.stock_present:
            stock = ET.SubElement(part_elem, "Stock")
            # PMin/PMax are the Part-local stock box. The effective drawing XY
            # is machining_origin + stock_offset; stock_surface is the top Z.
            offset_x, offset_y = self.stock_offset
            ET.SubElement(stock, "PMin").text = (
                f"{offset_x},{offset_y},{self.stock_surface - self.stock_thickness}"
            )
            ET.SubElement(stock, "PMax").text = (
                f"{offset_x + self.stock_width},{offset_y + self.stock_height},{self.stock_surface}"
            )
            ET.SubElement(stock, "Material").text = self.stock_material
            ET.SubElement(stock, "Color").text = self.stock_color

        ET.SubElement(part_elem, "MachiningOrigin").text = f"{self.machining_origin[0]},{self.machining_origin[1]}"

        if self.default_tool_diameter is not None:
            ET.SubElement(part_elem, "ToolDiameter").text = str(self.default_tool_diameter)
        if self.default_spindle_speed is not None:
            # CamBam doesn't seem to have a direct default spindle speed at part level in XML?
            # MOPs inherit if not set, but storing it here is useful for the framework.
            # We won't write it to XML unless a specific field is found later.
            pass

        # Add other common Part elements expected by CamBam
        ET.SubElement(part_elem, "ToolProfile").text = "EndMill" # Default, can be overridden by MOPs
        nesting = ET.SubElement(part_elem, "Nesting")
        ET.SubElement(nesting, "NestMethod").text = self.nesting_method
        # CamBam nesting records are method-specific. Grid controls are not
        # valid placement data for None, Manual, or PointList nesting.
        if self.nesting_method in ("Grid", "IsoGrid"):
            ET.SubElement(nesting, "BasePoint").text = "0,0"
            ET.SubElement(nesting, "Rows").text = str(self.nesting_rows)
            ET.SubElement(nesting, "Columns").text = str(self.nesting_columns)
            ET.SubElement(nesting, "Spacing").text = str(self.nesting_spacing)
            ET.SubElement(nesting, "GridOrder").text = self.nesting_grid_order
            ET.SubElement(nesting, "GridDirectionAlternate").text = str(self.nesting_grid_alternate).lower()

        return part_elem

# --- MOP Base and Concrete Classes ---

MopType = TypeVar('MopType', bound='Mop')

# Native XML paths for fields represented by the four supported MOP classes.
# Keeping this table beside the entity model gives the reader and writer one
# vocabulary for state tracking while the native XML template retains fields
# that this model does not represent.
MOP_XML_FIELD_PATHS: Dict[str, Tuple[str, ...]] = {
    'target_depth': ('TargetDepth',), 'depth_increment': ('DepthIncrement',),
    'stock_surface': ('StockSurface',), 'roughing_clearance': ('RoughingClearance',),
    'clearance_plane': ('ClearancePlane',), 'spindle_direction': ('SpindleDirection',),
    'spindle_speed': ('SpindleSpeed',), 'velocity_mode': ('VelocityMode',),
    'work_plane': ('WorkPlane',), 'optimisation_mode': ('OptimisationMode',),
    'tool_diameter': ('ToolDiameter',), 'tool_number': ('ToolNumber',),
    'tool_profile': ('ToolProfile',), 'plunge_feedrate': ('PlungeFeedrate',),
    'cut_feedrate': ('CutFeedrate',), 'max_crossover_distance': ('MaxCrossoverDistance',),
    'custom_mop_header': ('CustomMOPHeader',), 'custom_mop_footer': ('CustomMOPFooter',),
    'stepover': ('StepOver',), 'profile_side': ('InsideOutside',),
    'milling_direction': ('MillingDirection',), 'collision_detection': ('CollisionDetection',),
    'corner_overcut': ('CornerOvercut',), 'lead_in_type': ('LeadInMove', 'LeadInType'),
    'lead_in_spiral_angle': ('LeadInMove', 'SpiralAngle'),
    'final_depth_increment': ('FinalDepthIncrement',), 'cut_ordering': ('CutOrdering',),
    'tab_method': ('HoldingTabs', 'TabMethod'), 'tab_width': ('HoldingTabs', 'Width'),
    'tab_height': ('HoldingTabs', 'Height'), 'tab_min_tabs': ('HoldingTabs', 'MinimumTabs'),
    'tab_max_tabs': ('HoldingTabs', 'MaximumTabs'), 'tab_distance': ('HoldingTabs', 'TabDistance'),
    'tab_size_threshold': ('HoldingTabs', 'SizeThreshold'),
    'tab_use_leadins': ('HoldingTabs', 'UseLeadIns'), 'tab_style': ('HoldingTabs', 'TabStyle'),
    'stepover_feedrate': ('StepoverFeedrate',), 'region_fill_style': ('RegionFillStyle',),
    'finish_stepover': ('FinishStepover',),
    'finish_stepover_at_target_depth': ('FinishStepoverAtTargetDepth',),
    'roughing_finishing': ('RoughingFinishing',), 'drilling_method': ('DrillingMethod',),
    'peck_distance': ('PeckDistance',), 'retract_height': ('RetractHeight',),
    'dwell': ('Dwell',), 'hole_diameter': ('HoleDiameter',),
    'drill_lead_out': ('DrillLeadOut',), 'spiral_flat_base': ('SpiralFlatBase',),
    'lead_out_length': ('LeadOutLength',), 'custom_script': ('CustomScript',),
}
MOP_XML_PATH_TO_FIELD = {path: field for field, path in MOP_XML_FIELD_PATHS.items()}


@dataclass(frozen=True)
class MopFieldEncodingPolicy:
    """Fresh-export policy for one modeled common MOP field."""

    xml_tag: str
    resolution: str = "direct"
    omit_when_none: bool = False
    omit_when_empty: bool = False


@dataclass(frozen=True)
class MopSubtypeFieldEncodingPolicy:
    """Fresh-export policy for one modeled MOP subtype field.

    Nested containers are explicit ``Value`` records.  ``leaf_state`` reflects
    CamBam's native convention: lead-move children carry their own state while
    holding-tab children are plain values governed by the container state.
    """

    xml_path: Tuple[str, ...]
    omit_when_none: bool = False
    omit_when_empty: bool = False
    default_when_none: bool = False
    requirements: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()
    leaf_state: bool = True


# Ordered to match native CamBam MOP XML.  Fresh authoring emits every retained
# field as an explicit Value; inheritance is represented by omission unless a
# caller deliberately requests Default through ``set_parameter_state``.  Imported
# templates follow the separate preservation path in ``_native_mop_element``.
MOP_COMMON_FIELD_POLICIES: Dict[str, MopFieldEncodingPolicy] = {
    'target_depth': MopFieldEncodingPolicy('TargetDepth', omit_when_none=True),
    'depth_increment': MopFieldEncodingPolicy('DepthIncrement', omit_when_none=True),
    'stock_surface': MopFieldEncodingPolicy('StockSurface'),
    'roughing_clearance': MopFieldEncodingPolicy('RoughingClearance'),
    'clearance_plane': MopFieldEncodingPolicy('ClearancePlane'),
    'spindle_direction': MopFieldEncodingPolicy('SpindleDirection'),
    'spindle_speed': MopFieldEncodingPolicy(
        'SpindleSpeed', resolution='part', omit_when_none=True),
    'velocity_mode': MopFieldEncodingPolicy('VelocityMode'),
    'work_plane': MopFieldEncodingPolicy('WorkPlane'),
    'optimisation_mode': MopFieldEncodingPolicy('OptimisationMode'),
    'tool_diameter': MopFieldEncodingPolicy(
        'ToolDiameter', resolution='part_or_project', omit_when_none=True),
    'tool_number': MopFieldEncodingPolicy('ToolNumber'),
    'tool_profile': MopFieldEncodingPolicy('ToolProfile'),
    'plunge_feedrate': MopFieldEncodingPolicy('PlungeFeedrate'),
    'cut_feedrate': MopFieldEncodingPolicy('CutFeedrate', omit_when_none=True),
    'max_crossover_distance': MopFieldEncodingPolicy('MaxCrossoverDistance'),
    'custom_mop_header': MopFieldEncodingPolicy(
        'CustomMOPHeader', omit_when_empty=True),
    'custom_mop_footer': MopFieldEncodingPolicy(
        'CustomMOPFooter', omit_when_empty=True),
}


MOP_PROFILE_FIELD_POLICIES: Dict[str, MopSubtypeFieldEncodingPolicy] = {
    'stepover': MopSubtypeFieldEncodingPolicy(('StepOver',)),
    'profile_side': MopSubtypeFieldEncodingPolicy(('InsideOutside',)),
    'milling_direction': MopSubtypeFieldEncodingPolicy(('MillingDirection',)),
    'collision_detection': MopSubtypeFieldEncodingPolicy(('CollisionDetection',)),
    'corner_overcut': MopSubtypeFieldEncodingPolicy(('CornerOvercut',)),
    'lead_in_type': MopSubtypeFieldEncodingPolicy(('LeadInMove', 'LeadInType')),
    'lead_in_spiral_angle': MopSubtypeFieldEncodingPolicy(
        ('LeadInMove', 'SpiralAngle'),
        requirements=(('lead_in_type', ('Spiral',)),),
    ),
    'final_depth_increment': MopSubtypeFieldEncodingPolicy(
        ('FinalDepthIncrement',), omit_when_none=True),
    'cut_ordering': MopSubtypeFieldEncodingPolicy(('CutOrdering',)),
    'tab_method': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'TabMethod'), leaf_state=False),
    'tab_width': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'Width'), requirements=(('tab_method', ('Automatic',)),),
        leaf_state=False),
    'tab_height': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'Height'), requirements=(('tab_method', ('Automatic',)),),
        leaf_state=False),
    'tab_min_tabs': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'MinimumTabs'), requirements=(('tab_method', ('Automatic',)),),
        leaf_state=False),
    'tab_max_tabs': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'MaximumTabs'), requirements=(('tab_method', ('Automatic',)),),
        leaf_state=False),
    'tab_distance': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'TabDistance'), requirements=(('tab_method', ('Automatic',)),),
        leaf_state=False),
    'tab_size_threshold': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'SizeThreshold'), requirements=(('tab_method', ('Automatic',)),),
        leaf_state=False),
    'tab_use_leadins': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'UseLeadIns'), requirements=(('tab_method', ('Automatic',)),),
        leaf_state=False),
    'tab_style': MopSubtypeFieldEncodingPolicy(
        ('HoldingTabs', 'TabStyle'), requirements=(('tab_method', ('Automatic',)),),
        leaf_state=False),
}


MOP_POCKET_FIELD_POLICIES: Dict[str, MopSubtypeFieldEncodingPolicy] = {
    'stepover': MopSubtypeFieldEncodingPolicy(('StepOver',)),
    'stepover_feedrate': MopSubtypeFieldEncodingPolicy(('StepoverFeedrate',)),
    'milling_direction': MopSubtypeFieldEncodingPolicy(('MillingDirection',)),
    'collision_detection': MopSubtypeFieldEncodingPolicy(('CollisionDetection',)),
    'lead_in_type': MopSubtypeFieldEncodingPolicy(('LeadInMove', 'LeadInType')),
    'lead_in_spiral_angle': MopSubtypeFieldEncodingPolicy(
        ('LeadInMove', 'SpiralAngle'),
        requirements=(('lead_in_type', ('Spiral',)),),
    ),
    'final_depth_increment': MopSubtypeFieldEncodingPolicy(
        ('FinalDepthIncrement',), omit_when_none=True),
    'cut_ordering': MopSubtypeFieldEncodingPolicy(('CutOrdering',)),
    'region_fill_style': MopSubtypeFieldEncodingPolicy(('RegionFillStyle',)),
    'finish_stepover': MopSubtypeFieldEncodingPolicy(('FinishStepover',)),
    'finish_stepover_at_target_depth': MopSubtypeFieldEncodingPolicy(
        ('FinishStepoverAtTargetDepth',)),
    'roughing_finishing': MopSubtypeFieldEncodingPolicy(('RoughingFinishing',)),
}


MOP_ENGRAVE_FIELD_POLICIES: Dict[str, MopSubtypeFieldEncodingPolicy] = {
    'roughing_finishing': MopSubtypeFieldEncodingPolicy(('RoughingFinishing',)),
    'final_depth_increment': MopSubtypeFieldEncodingPolicy(
        ('FinalDepthIncrement',), omit_when_none=True),
    'cut_ordering': MopSubtypeFieldEncodingPolicy(('CutOrdering',)),
}


_SPIRAL_DRILL_METHODS = ('SpiralMill_CW', 'SpiralMill_CCW')
MOP_DRILL_FIELD_POLICIES: Dict[str, MopSubtypeFieldEncodingPolicy] = {
    'drilling_method': MopSubtypeFieldEncodingPolicy(('DrillingMethod',)),
    'peck_distance': MopSubtypeFieldEncodingPolicy(
        ('PeckDistance',), requirements=(('drilling_method', ('CannedCycle',)),)),
    'retract_height': MopSubtypeFieldEncodingPolicy(
        ('RetractHeight',), requirements=(('drilling_method', ('CannedCycle',)),)),
    'dwell': MopSubtypeFieldEncodingPolicy(
        ('Dwell',), requirements=(('drilling_method', ('CannedCycle',)),)),
    'hole_diameter': MopSubtypeFieldEncodingPolicy(
        ('HoleDiameter',), default_when_none=True,
        requirements=(('drilling_method', _SPIRAL_DRILL_METHODS),)),
    'drill_lead_out': MopSubtypeFieldEncodingPolicy(
        ('DrillLeadOut',),
        requirements=(('drilling_method', _SPIRAL_DRILL_METHODS),)),
    'spiral_flat_base': MopSubtypeFieldEncodingPolicy(
        ('SpiralFlatBase',),
        requirements=(('drilling_method', _SPIRAL_DRILL_METHODS),)),
    'lead_out_length': MopSubtypeFieldEncodingPolicy(
        ('LeadOutLength',),
        requirements=(('drilling_method', _SPIRAL_DRILL_METHODS),)),
    'custom_script': MopSubtypeFieldEncodingPolicy(
        ('CustomScript',), omit_when_empty=True,
        requirements=(('drilling_method', ('CustomScript',)),)),
}

@dataclass
class Mop(CamBamEntity, ABC):
    """
    Abstract Base Class for Machine Operations (MOPs).
    Holds intrinsic machining parameters.
    Targets and part assignment are owned by the project.
    """
    # Intrinsic Attributes
    name: str = "MOP" # User-visible name in CamBam UI MOP tree
    enabled: bool = True
    target_depth: Optional[float] = None # None leaves the field to CamBam inheritance.
    depth_increment: Optional[float] = None # None leaves the field to CamBam inheritance.
    stock_surface: float = 0.0
    roughing_clearance: float = 0.0
    clearance_plane: float = 15.0
    spindle_direction: str = 'CW' # 'CW', 'CCW', 'Off'
    spindle_speed: Optional[int] = None # If None, uses Part/Project default
    velocity_mode: str = 'ExactStop' # 'ExactStop', 'ConstantVelocity'
    work_plane: str = 'XY' # 'XY', 'XZ', 'YZ'
    optimisation_mode: str = 'Standard' # Standard=0.9.7 Legacy; Experimental=0.9.8 New; None
    tool_diameter: Optional[float] = None # If None, uses Part/Project default
    tool_number: int = 0 # If 0, uses current tool
    tool_profile: str = 'EndMill' # CamBam enum: EndMill, VCutter, BullNose, BallNose, Drill, Lathe
    plunge_feedrate: float = 1000.0
    cut_feedrate: Optional[float] = None # None leaves the field to CamBam inheritance.
    max_crossover_distance: float = 0.7 # Multiplier of tool diameter
    custom_mop_header: str = ""
    custom_mop_footer: str = ""
    # Note: No part_id or _resolved_xml_primitive_ids here. Managed by Project/Writer.

    def __setattr__(self, name, value):
        if name == "tool_profile" and value in {"Vcutter", "V-Cutter"}:
            # CamBam's API enum is ToolProfiles.VCutter. Its UI and older prose
            # use other spellings, but those strings deserialize as Unspecified.
            value = "VCutter"
        # Once imported or explicitly state-edited, an assignment is intentional,
        # even if it repeats a cached Default value or restores an earlier value.
        if name in MOP_XML_FIELD_PATHS and (
                "_xml_template" in self.__dict__
                or "_xml_explicit_parameter_states" in self.__dict__):
            self.__dict__.setdefault("_xml_dirty_parameters", set()).add(name)
            self.__dict__.get("_xml_explicit_parameter_states", set()).discard(name)
        super().__setattr__(name, value)

    def set_parameter_state(self, field_name: str, state: str) -> None:
        """Set a top-level scalar parameter's CamBam inheritance state.

        ``Default`` leaves resolution to CamBam's CAM styles.  The state is
        metadata separate from the Python value so callers may explicitly
        restore inheritance after assigning a value.
        """
        if field_name not in MOP_XML_FIELD_PATHS or not hasattr(self, field_name):
            raise ValueError(f"Unsupported MOP parameter: {field_name}")
        if len(MOP_XML_FIELD_PATHS[field_name]) != 1:
            raise ValueError("Nested parameter inheritance belongs to its native container")
        if state not in ("Default", "Value"):
            raise ValueError("MOP parameter state must be 'Default' or 'Value'")
        states = getattr(self, "_xml_parameter_states", None)
        if states is None:
            states = {}
            self._xml_parameter_states = states
        states[field_name] = state
        explicit = getattr(self, "_xml_explicit_parameter_states", None)
        if explicit is None:
            explicit = set()
            self._xml_explicit_parameter_states = explicit
        explicit.add(field_name)

    def _native_mop_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> Optional[ET.Element]:
        """Clone an imported MOP and patch only fields explicitly changed."""
        template = getattr(self, "_xml_template", None)
        if template is None:
            return None
        root = deepcopy(template)
        root.set("Enabled", str(self.enabled).lower())
        name = root.find("Name")
        if name is None:
            name = ET.Element("Name")
            root.insert(0, name)
        name.text = self.name
        tag = root.find("Tag")
        if tag is None:
            tag = ET.Element("Tag")
            root.insert(1, tag)
        tag.text = json.dumps({"user_id": self.user_identifier, "internal_id": str(self.internal_id)}, separators=(",", ":"))

        primitive = root.find("primitive")
        if primitive is None:
            primitive = ET.Element("primitive")
            root.insert(2 + self._xml_primitive_index, primitive)
        for child in list(primitive):
            primitive.remove(child)
        for pid in sorted(resolved_primitive_xml_ids):
            ET.SubElement(primitive, "prim").text = str(pid)

        baseline = getattr(self, "_xml_parameter_baseline", {})
        states = getattr(self, "_xml_parameter_states", {})
        explicit_states = getattr(self, "_xml_explicit_parameter_states", set())
        for field_name, path in MOP_XML_FIELD_PATHS.items():
            if not hasattr(self, field_name):
                continue
            current = getattr(self, field_name)
            changed = (field_name in getattr(self, "_xml_dirty_parameters", set())
                       or field_name not in baseline or current != baseline[field_name])
            explicit_state = field_name in explicit_states
            if not changed and not explicit_state:
                continue
            parent = root
            for part in path[:-1]:
                child = parent.find(part)
                if child is None:
                    child = ET.SubElement(parent, part)
                parent = child
            element = parent.find(path[-1])
            if element is None:
                element = ET.SubElement(parent, path[-1])
            state = states.get(field_name) if field_name in explicit_states else None
            if state is None:
                state = "Default" if current is None else "Value"
            if len(path) == 1 or "state" in element.attrib or path[0] == "LeadInMove":
                element.set("state", state)
            element.text = self._format_mop_parameter(current)
            # A nested child edit must opt the native container into Value as
            # well, otherwise CamBam can continue inheriting the whole group.
            for depth in range(1, len(path)):
                parent_node = root.find("/".join(path[:depth]))
                if parent_node is not None:
                    parent_node.set("state", "Value")
        return root

    @staticmethod
    def _format_mop_parameter(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    def _apply_explicit_parameter_states(self, root: ET.Element) -> None:
        """Apply state edits to a newly-created (non-imported) MOP tree."""
        for field_name in getattr(self, "_xml_explicit_parameter_states", set()):
            path = MOP_XML_FIELD_PATHS[field_name]
            parent = root
            for part in path[:-1]:
                child = parent.find(part)
                if child is None:
                    child = ET.SubElement(parent, part)
                parent = child
            element = parent.find(path[-1])
            if element is None:
                element = ET.SubElement(parent, path[-1])
            state = self._xml_parameter_states[field_name]
            element.set("state", state)
            element.text = self._format_mop_parameter(getattr(self, field_name))
            if path[:-1]:
                ancestor = root.find(path[0])
                if ancestor is not None:
                    ancestor.set("state", state)

    @abstractmethod
    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        """
        Generates the MOP's specific XML element (e.g., <profile>, <pocket>).
        Requires the project context to resolve default parameters and the list of
        resolved primitive XML IDs (calculated by the writer).
        """
        pass

    def _get_effective_param(self, param_name: str, project: "CamBamProject") -> Any:
        """Helper to get a parameter value, checking MOP, then Part, then Project defaults."""
        mop_value = getattr(self, param_name, None)
        if mop_value is not None:
            return mop_value

        # Check Part default
        part = project.get_part_of_mop(self.internal_id) if project else None
        if part:
            part_param_map = {
                'tool_diameter': 'default_tool_diameter',
                'spindle_speed': 'default_spindle_speed'
                # Add other mappings if Part gets more defaults
            }
            part_attr = part_param_map.get(param_name)
            if part_attr:
                part_value = getattr(part, part_attr, None)
                if part_value is not None:
                    return part_value

        # Check Project default
        project_param_map = {
            'tool_diameter': 'default_tool_diameter'
            # Add other mappings if Project gets more defaults
        }
        project_attr = project_param_map.get(param_name)
        if project_attr and project:
            project_value = getattr(project, project_attr, None)
            if project_value is not None:
                return project_value

        # No value found anywhere
        # logger.debug(f"Parameter '{param_name}' not resolved for MOP '{self.name}'. Returning None.")
        return None

    def _add_common_mop_elements(self, mop_root_elem: ET.Element, project: "CamBamProject", resolved_primitive_xml_ids: List[int]):
        """Adds common XML sub-elements shared by all MOP types."""
        ET.SubElement(mop_root_elem, "Name").text = self.name # Use MOP's intrinsic name
        ET.SubElement(mop_root_elem, "Tag").text = json.dumps({
            "user_id": self.user_identifier, "internal_id": str(self.internal_id),
        })

        for field_name, policy in MOP_COMMON_FIELD_POLICIES.items():
            value = getattr(self, field_name)
            if policy.resolution in {'part', 'part_or_project'}:
                value = self._get_effective_param(field_name, project)
            if policy.omit_when_none and value is None:
                continue
            if policy.omit_when_empty and value == "":
                continue
            ET.SubElement(
                mop_root_elem, policy.xml_tag, {"state": "Value"}
            ).text = self._format_mop_parameter(value)

        # Add primitive references (using the resolved XML IDs passed by the writer)
        primitive_container = ET.SubElement(mop_root_elem, "primitive")
        if resolved_primitive_xml_ids:
            for pid in sorted(resolved_primitive_xml_ids): # Sort for consistency
                ET.SubElement(primitive_container, "prim").text = str(pid)
        # else: # CamBam seems to omit the <primitive> tag entirely if empty
            # pass

    def _policy_applies(self, policy: MopSubtypeFieldEncodingPolicy) -> bool:
        return all(getattr(self, field_name) in allowed
                   for field_name, allowed in policy.requirements)

    @staticmethod
    def _policy_omits(policy: MopSubtypeFieldEncodingPolicy, value: Any) -> bool:
        return ((policy.omit_when_none and value is None)
                or (policy.omit_when_empty and value == ""))

    @staticmethod
    def _policy_state(policy: MopSubtypeFieldEncodingPolicy, value: Any) -> str:
        if policy.default_when_none and value is None:
            return "Default"
        return "Value"

    def _add_subtype_mop_elements(
        self,
        parent_elem: ET.Element,
        policies: Dict[str, MopSubtypeFieldEncodingPolicy],
    ) -> None:
        """Emit one declarative fresh Profile/Pocket subtype policy."""
        for field_name, policy in policies.items():
            value = getattr(self, field_name)
            if not self._policy_applies(policy):
                continue
            if self._policy_omits(policy, value):
                continue
            state = self._policy_state(policy, value)
            parent = parent_elem
            for tag in policy.xml_path[:-1]:
                child = parent.find(tag)
                if child is None:
                    child = ET.SubElement(parent, tag, {"state": "Value"})
                else:
                    child.set("state", "Value")
                parent = child
            attributes = {"state": state} if policy.leaf_state else {}
            ET.SubElement(parent, policy.xml_path[-1], attributes).text = (
                self._format_mop_parameter(value)
            )

    def _reconcile_native_mode_group(
        self,
        root: ET.Element,
        policies: Dict[str, MopSubtypeFieldEncodingPolicy],
        controller: str,
    ) -> None:
        """Make an edited native mode switch internally coherent.

        Only modeled children are added or removed.  Unknown native children and
        independent lead-out records remain preservation-owned.
        """
        if controller not in getattr(self, "_xml_dirty_parameters", set()):
            return
        container_tag = policies[controller].xml_path[0]
        container = root.find(container_tag)
        if container is None:
            container = ET.SubElement(root, container_tag)
        container.set("state", "Value")
        for field_name, policy in policies.items():
            if len(policy.xml_path) != 2 or policy.xml_path[0] != container_tag:
                continue
            leaf = container.find(policy.xml_path[1])
            value = getattr(self, field_name)
            applies = (self._policy_applies(policy)
                       and not self._policy_omits(policy, value))
            if not applies:
                if leaf is not None:
                    container.remove(leaf)
                continue
            if leaf is None:
                leaf = ET.SubElement(container, policy.xml_path[1])
            if policy.leaf_state:
                leaf.set("state", "Value")
            else:
                leaf.attrib.pop("state", None)
            leaf.text = self._format_mop_parameter(value)

    def _reconcile_native_flat_mode(
        self,
        root: ET.Element,
        policies: Dict[str, MopSubtypeFieldEncodingPolicy],
        controller: str,
    ) -> None:
        """Rebuild modeled top-level dependents after an imported mode switch."""
        if controller not in getattr(self, "_xml_dirty_parameters", set()):
            return
        for field_name, policy in policies.items():
            if field_name == controller or len(policy.xml_path) != 1:
                continue
            element = root.find(policy.xml_path[0])
            value = getattr(self, field_name)
            applies = (self._policy_applies(policy)
                       and not self._policy_omits(policy, value))
            if not applies:
                if element is not None:
                    root.remove(element)
                continue
            if element is None:
                element = ET.SubElement(root, policy.xml_path[0])
            element.set("state", self._policy_state(policy, value))
            element.text = self._format_mop_parameter(value)

    def _validate_lead_encoding(self) -> None:
        supported = {"None", "Spiral"}
        dirty = getattr(self, "_xml_dirty_parameters", set())
        baseline = getattr(self, "_xml_parameter_baseline", {})
        if self.lead_in_type not in supported and (
                not hasattr(self, "_xml_template") or "lead_in_type" in dirty):
            raise ValueError(
                "Fresh lead-in authoring supports only None and Spiral; other "
                "native lead modes are preserve-only"
            )
        if ("lead_in_type" in dirty
                and baseline.get("lead_in_type") not in supported):
            raise ValueError(
                "Switching an imported unsupported native lead mode is not supported"
            )
        if "lead_in_spiral_angle" in dirty and self.lead_in_type != "Spiral":
            raise ValueError("lead_in_spiral_angle requires lead_in_type='Spiral'")

# --- Concrete MOP Classes ---
# (Minimal changes: Update to_xml_element signature and call _add_common_mop_elements)

@dataclass
class ProfileMop(Mop):
    # Profile specific parameters
    stepover: float = 0.4 # Tool diameter fraction
    profile_side: str = 'Inside' # 'Inside', 'Outside'
    milling_direction: str = 'Conventional' # 'Conventional', 'Climb'
    collision_detection: bool = True
    # Allow CamBam to overcut inside corners for round tools; may remove
    # additional material along adjacent sides.
    corner_overcut: bool = False
    lead_in_type: str = 'Spiral' # Fresh authoring: 'None', 'Spiral'; other native modes preserve-only
    lead_in_spiral_angle: float = 30.0
    final_depth_increment: Optional[float] = 0.0 # If > 0, amount for final pass
    cut_ordering: str = 'DepthFirst' # 'DepthFirst', 'LevelFirst'
    # Holding Tabs parameters
    tab_method: str = 'None' # 'None', 'Automatic'; imported native Manual is preserve-only
    tab_width: float = 6.0
    tab_height: float = 1.5
    tab_min_tabs: int = 3
    tab_max_tabs: int = 3
    tab_distance: float = 40.0 # Approx distance between auto tabs
    tab_size_threshold: float = 4.0 # Min shape size for tabs
    tab_use_leadins: bool = False
    tab_style: str = 'Square' # 'Square', 'Triangle', 'Skip'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        self._validate_lead_encoding()
        dirty = getattr(self, "_xml_dirty_parameters", set())
        baseline = getattr(self, "_xml_parameter_baseline", {})
        if self.tab_method == 'Manual' and (
                not hasattr(self, "_xml_template") or 'tab_method' in dirty):
            raise ValueError(
                "Manual holding-tab authoring requires explicit native tab points and "
                "is not supported; imported native Manual tabs are preserve-only"
            )
        if ('tab_method' in dirty and baseline.get('tab_method') == 'Manual'):
            raise ValueError(
                "Switching an imported Manual holding-tab record is not supported"
            )
        tab_dependency_edited = not hasattr(self, "_xml_template") or bool(
            dirty.intersection(
                {'tab_method', 'tab_use_leadins', 'tab_style', 'lead_in_type'}))
        if self.tab_use_leadins and tab_dependency_edited and not (
                self.tab_method == 'Automatic'
                and self.tab_style == 'Square'
                and self.lead_in_type != 'None'):
            raise ValueError(
                "tab_use_leadins requires Automatic Square tabs and an active lead-in"
            )
        native = self._native_mop_element(project, resolved_primitive_xml_ids)
        if native is not None:
            self._reconcile_native_mode_group(
                native, MOP_PROFILE_FIELD_POLICIES, 'lead_in_type')
            self._reconcile_native_mode_group(
                native, MOP_PROFILE_FIELD_POLICIES, 'tab_method')
            return native
        mop_elem = ET.Element("profile", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)
        self._add_subtype_mop_elements(mop_elem, MOP_PROFILE_FIELD_POLICIES)
        self._apply_explicit_parameter_states(mop_elem)
        return mop_elem


@dataclass
class PocketMop(Mop):
    # Pocket specific parameters
    stepover: float = 0.4
    stepover_feedrate: str = 'Plunge Feedrate' # Name of feedrate to use for stepover moves
    milling_direction: str = 'Conventional'
    collision_detection: bool = True
    lead_in_type: str = 'Spiral'
    lead_in_spiral_angle: float = 30.0
    final_depth_increment: Optional[float] = 0.0
    cut_ordering: str = 'DepthFirst'
    region_fill_style: str = 'InsideOutsideOffsets' # 'InsideOutsideOffsets', 'HorizontalScanline', 'VerticalScanline'
    finish_stepover: float = 0.0 # If > 0, performs a finishing pass this far from edge
    finish_stepover_at_target_depth: bool = False # Apply finish pass only at final depth
    roughing_finishing: str = 'Roughing' # 'Roughing', 'Finishing', 'RoughFinish'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        self._validate_lead_encoding()
        native = self._native_mop_element(project, resolved_primitive_xml_ids)
        if native is not None:
            self._reconcile_native_mode_group(
                native, MOP_POCKET_FIELD_POLICIES, 'lead_in_type')
            return native
        mop_elem = ET.Element("pocket", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)
        self._add_subtype_mop_elements(mop_elem, MOP_POCKET_FIELD_POLICIES)
        self._apply_explicit_parameter_states(mop_elem)
        return mop_elem

@dataclass
class EngraveMop(Mop):
    # Engrave specific parameters
    # CamBam publishes this property on Engrave but documents it as effective
    # only for Lathe and 3D Profile. Retained as explicit compatibility metadata.
    roughing_finishing: str = 'Roughing'
    final_depth_increment: Optional[float] = 0.0 # Depth for final pass
    cut_ordering: str = 'DepthFirst'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        native = self._native_mop_element(project, resolved_primitive_xml_ids)
        if native is not None:
            return native
        mop_elem = ET.Element("engrave", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)
        self._add_subtype_mop_elements(mop_elem, MOP_ENGRAVE_FIELD_POLICIES)
        self._apply_explicit_parameter_states(mop_elem)
        return mop_elem


@dataclass
class DrillMop(Mop):
    # Drill specific parameters
    drilling_method: str = 'CannedCycle' # 'CannedCycle', 'SpiralMill_CW', 'SpiralMill_CCW', 'CustomScript'
    # Parameters for CannedCycle
    peck_distance: float = 0.0 # If > 0, enables pecking (G83)
    retract_height: float = 5.0 # R plane for canned cycles
    dwell: float = 0.0 # Bottom dwell; controller/postprocessor determines time units.
    # Parameters for SpiralMill
    # Desired hole-boundary diameter. ``None`` writes CamBam's Default/Auto
    # state, which derives the diameter from supported selected geometry.
    hole_diameter: Optional[float] = None
    drill_lead_out: bool = False
    spiral_flat_base: bool = True
    lead_out_length: float = 0.0
    # Parameter for CustomScript
    custom_script: str = ""

    def effective_spiral_hole_diameter(self) -> Optional[float]:
        """Return the cut diameter after applying signed radial clearance."""
        if self.hole_diameter is None:
            return None
        return self.hole_diameter - 2.0 * self.roughing_clearance

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        supported_methods = {'CannedCycle', *_SPIRAL_DRILL_METHODS, 'CustomScript'}
        dirty = getattr(self, "_xml_dirty_parameters", set())
        baseline = getattr(self, "_xml_parameter_baseline", {})
        if self.drilling_method not in supported_methods and (
                not hasattr(self, "_xml_template") or 'drilling_method' in dirty):
            raise ValueError(
                "Fresh Drill authoring supports CannedCycle, SpiralMill_CW, "
                "SpiralMill_CCW and CustomScript; other native methods are preserve-only"
            )
        if ('drilling_method' in dirty
                and baseline.get('drilling_method') not in supported_methods):
            raise ValueError(
                "Switching an unsupported imported Drill method is not supported"
            )
        if self.drilling_method == 'CustomScript' and not self.custom_script and (
                not hasattr(self, "_xml_template")
                or 'drilling_method' in dirty or 'custom_script' in dirty):
            raise ValueError("CustomScript drilling requires a nonempty custom_script")
        validate_spiral = (self.drilling_method in _SPIRAL_DRILL_METHODS and (
            not hasattr(self, "_xml_template")
            or bool(dirty.intersection({
                'drilling_method', 'hole_diameter', 'roughing_clearance',
                'tool_diameter', 'drill_lead_out', 'lead_out_length',
            }))
        ))
        if validate_spiral:
            effective_hole_diameter = self.effective_spiral_hole_diameter()
            effective_tool_diameter = self._get_effective_param('tool_diameter', project)
            if (effective_hole_diameter is not None
                    and effective_tool_diameter is not None
                    and effective_hole_diameter <= effective_tool_diameter):
                raise ValueError(
                    "SpiralMill requires hole_diameter - 2 * roughing_clearance "
                    "to be greater than tool_diameter"
                )
            if not self.drill_lead_out and self.lead_out_length != 0:
                raise ValueError(
                    "SpiralMill lead_out_length must be zero when drill_lead_out is false"
                )
            if (self.drill_lead_out and effective_hole_diameter is not None
                    and self.lead_out_length > effective_hole_diameter / 2.0):
                raise ValueError(
                    "SpiralMill positive lead_out_length must not exceed the "
                    "effective hole radius"
                )
        native = self._native_mop_element(project, resolved_primitive_xml_ids)
        if native is not None:
            self._reconcile_native_flat_mode(
                native, MOP_DRILL_FIELD_POLICIES, 'drilling_method')
            return native
        mop_elem = ET.Element("drill", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)
        if self.drilling_method in _SPIRAL_DRILL_METHODS and self.hole_diameter is None:
            logger.warning(
                f"Drill MOP '{self.name}' uses SpiralMill with Auto HoleDiameter; "
                "CamBam must derive it from the selected target geometry."
            )
        self._add_subtype_mop_elements(mop_elem, MOP_DRILL_FIELD_POLICIES)
        self._apply_explicit_parameter_states(mop_elem)
        return mop_elem
