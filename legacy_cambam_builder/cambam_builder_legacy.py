import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import pickle
from datetime import datetime
import logging

# Set up logger for this module
logger = logging.getLogger('main.cambam_builder')

# Fix for the whitespace issue in the text elements.
TEXT_IDENTIFIER_PREFIX = "THIS_IS_A_CUSTOM_PREFIX_THAT_WONT_BE_FOUND_IN_TEXT_ELEMENTS_BUT_WILL_BE_USED_TO_IDENTIFY_THE_BEGINNING_OF_A_TEXT_ELEMENT"
PROBLEMATIC_WHITESPACE = "\n          " # The whitespace that needs to be removed.

class CamBam:
    """Create a CamBam file with CAD and CAM elements.
    
    Attributes:
        name (str): Name of the file.
        endmill (float): Endmill diameter to use in the file.
        stock_thickness (float): Board thickness of the material.
        stock_width (int): Width of the stock, drawn in the file.
        stock_height (int): Height of the stock, drawn in the file.
        spindle_speed (int): Fixed spindle rpm for this file.
    """

    def __init__(self, name: str, endmill: float = 5, stock_thickness: float = 12.5, 
                 stock_width: int = 1220, stock_height: int = 2440, spindle_speed: int = 24000):
        self.name = name
        self.pid = 1
        self.endmill = endmill
        self.stock_thickness = stock_thickness
        self.stock_width = stock_width
        self.stock_height = stock_height
        self.spindle_speed = spindle_speed
        self.groups = {'unassigned': []}

        self.root = ET.Element("CADFile", {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xmlns:xsd": "http://www.w3.org/2001/XMLSchema",
            "Version": "0.9.8.0",
            "Name": name
        })
        self.layers = ET.SubElement(self.root, "layers")
        self.machining_options = ET.SubElement(self.root, "MachiningOptions")
        self.parts = ET.SubElement(self.root, "parts")

        stock = ET.SubElement(self.machining_options, "Stock")
        ET.SubElement(stock, "Material").text = "wood"
        ET.SubElement(stock, "PMin").text = f"0,0,{-self.stock_thickness}"
        ET.SubElement(stock, "PMax").text = f"{self.stock_width},{self.stock_height},0"
        ET.SubElement(stock, "Color").text = "255,165,0"
        ET.SubElement(self.machining_options, "ToolDiameter").text = str(self.endmill)
        ET.SubElement(self.machining_options, "ToolProfile").text = "EndMill"

        logger.info(f"Initialized CamBam object: {self.name}")
        logger.debug(f"CamBam settings: endmill={endmill}, stock_thickness={stock_thickness}, stock_width={stock_width}, stock_height={stock_height}, spindle_speed={spindle_speed}")
    
    # Alternative pretty-print method.
    def _prettify(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        content = reparsed.toprettyxml(indent="  ")

        # Remove the whitespaces, by locating the sequence of whitespace and the text identifier prefix, and replacing the them with nothing.
        content = content.replace(PROBLEMATIC_WHITESPACE+TEXT_IDENTIFIER_PREFIX, "")
        content = content.replace(TEXT_IDENTIFIER_PREFIX, "") # Remove any remaining text identifier prefixes that were not coupled with whitespace
        return content

    def _find_insert_position(self, elements: ET.Element, target: str, place_last: bool) -> int:
        if target == '':
            return -1 if not place_last else len(elements)
        
        target_element = None
        for i, elem in enumerate(elements):
            if elem.get('name') == target or elem.get('Name') == target:
                target_element = i
                break
        
        if target_element is None:
            return -1
        
        return target_element + 1 if place_last else target_element

    def layer(self, name: str = 'new layer', color: str = 'Green', alpha: float = 1.0, pen: float = 1.0,
              visible: bool = True, locked: bool = False, target: str = '', place_last: bool = True) -> None:
        '''Place a new layer in the range of layers\n
        name:       Name of the new layer
        color:      Format 'Red' or '255,0,0'
        alpha:      Layer transparence (0-1)
        pen:        Layer stroke width
        visible:    Show/hide layer
        locked:     Lock layer
        target:     Name of the existing layer to place this new layer at
        place_last: Place last/first relative to the target. If no target given, place last/first in the range of all layers'''

        # Create the new layer element
        new_layer = ET.Element("layer", {
            "name": name,
            "color": color,
            "alpha": str(alpha),
            "pen": str(pen),
            "visible": str(visible).lower(),
            "locked": str(locked).lower()
        })
        ET.SubElement(new_layer, "objects")
        pos = self._find_insert_position(self.layers, target, place_last)
        if pos == -1:
            self.layers.append(new_layer)
        else:
            self.layers.insert(pos, new_layer)

        logger.debug(f"Added new layer: {name} with color {color}")

    def part(self, name: str = 'new part', enabled: bool = True, target: str = '', place_last: bool = True) -> None:
        '''Place a new part in the range of parts\n
        name:       Name of the new part
        enabled:    Enable or disable the part
        target:     Name of the existing part to place this new part at
        place_last: Place last/first relative to the target. If no target given, place last/first in the range of all layers'''

        # Create the new part element
        new_part = ET.Element("part", {"Name": name, "Enabled": str(enabled).lower()})
        ET.SubElement(new_part, "machineops")
        stock = ET.SubElement(new_part, "Stock")
        ET.SubElement(stock, "PMin").text = "0,0"
        ET.SubElement(stock, "PMax").text = "0,0"
        ET.SubElement(stock, "Color").text = "255,165,0"
        ET.SubElement(new_part, "ToolProfile").text = "EndMill"
        nesting = ET.SubElement(new_part, "Nesting")
        ET.SubElement(nesting, "BasePoint")
        ET.SubElement(nesting, "NestMethod").text = "None"
        pos = self._find_insert_position(self.parts, target, place_last)
        if pos == -1:
            self.parts.append(new_part)
        else:
            self.parts.insert(pos, new_part)

    def _group_object(self, pid: int, target_groups: list) -> None:
        '''Internal: Assign an object to a cam group.\n
        Used to assign objects into one or more machine operations. One group should represent one MOP\n
        pid: The primitive ID of the object to be assigned
        groups: One or more groups to assign to. list[str] or str with comma for each group. None or '' => put in group 'unassigned'
        '''
        if target_groups is None or target_groups == '':
            self.groups['unassigned'].append(pid)
            return

        if isinstance(target_groups, str):
            target_groups = [x.strip() for x in target_groups.split(',')]

        for group in target_groups:
            if group == '':
                if pid not in self.groups['unassigned']:
                    self.groups['unassigned'].append(pid)
            else:
                if group not in self.groups:
                    self.groups[group] = []
                if pid not in self.groups[group]:
                    self.groups[group].append(pid)

    def _place_object(self, object_elem: ET.Element, target: str, place_last: bool = True) -> None:
        '''Internal: Place object in target layer\n
        object_elem: the XML element making up the object to be injected
        target: str. Layer to place in. Empty=first layer
        place_last: Place as last object in layer. False: place first in layer
        '''

        target_layer = None
        if target == '':
            layers = self.layers.findall('layer')
            if not layers:
                self.layer()
                layers = self.layers.findall('layer')
            target_layer = layers[0]
        else:
            for layer in self.layers.findall('layer'):
                if layer.get('name') == target:
                    target_layer = layer
                    break
            if not target_layer:
                self.layer(name=target)
                for layer in self.layers.findall('layer'):
                    if layer.get('name') == target:
                        target_layer = layer
                        break

        objects = target_layer.find('objects')
        if place_last:
            objects.append(object_elem)
        else:
            objects.insert(0, object_elem)

        self.pid += 1

    def obj_pline(self, points: list[tuple], closed: bool = False, target: str = '', place_last: bool = True, tags: str = '', groups: list[str] = None) -> None:
        '''Create a polyline in target layer\n
        points:     (p1x,p1y,(pb)),...,(pnx,pny,(pb)). List of tuples with point x, point y, and optionally bulge
        closed:     Whether to close the polyline (join ends)
        target:     str. Layer to place in. Empty=first layer
        place_last: Place as last object in layer. False: place first in layer
        tags:       tags associated with the object
        groups:     Assign the object to MOP groups. List of strings; or string with commas as separation
        '''
        pline = ET.Element("pline", {"id": str(self.pid), "Closed": str(closed).lower()})
        pts = ET.SubElement(pline, "pts")
        for point in points:
            b = point[2] if len(point) > 2 else 0
            ET.SubElement(pts, "p", {"b": str(b)}).text = f"{point[0]},{point[1]},0"

        tag_elem = ET.SubElement(pline, "Tag")
        tag_elem.text = f"{groups}\n{tags}"

        self._group_object(self.pid, groups)
        self._place_object(pline, target, place_last)

        logger.debug(f"Added pline to layer {target}")

    def obj_circle(self, center_x: float, center_y: float, diameter: float, target: str = '', place_last: bool = True, tags: str = '', groups: list[str] = None) -> None:
        '''Create a circle in target layer
        center_x:   Circle center x coordinate
        center_y:   Circle center y coordinate
        diameter:   Circle diameter
        target:     str. Layer to place in. Empty=first layer
        place_last: Place as last object in layer. False: place first in layer
        tags:       tags associated with the object
        groups:     Assign the object to MOP groups. List of strings; or string with commas as separation
        '''
        circle = ET.Element("circle", {"id": str(self.pid), "c": f"{center_x},{center_y},0", "d": str(diameter)})
        tag_elem = ET.SubElement(circle, "Tag")
        tag_elem.text = f"{groups}\n{tags}"

        self._group_object(self.pid, groups)
        self._place_object(circle, target, place_last)

        logger.debug(f"Added circle to layer {target}")

    def obj_points(self, points: list[tuple], target: str = '', place_last: bool = True, tags: str = '', groups: list[str] = None) -> None:
        '''Create a pointlist in target layer\n
        points:     (p1x,p1y),...,(pnx,pny). List of tuples with point x, point y
        target:     str. Layer to place in. Empty=first layer
        place_last: Place as last object in layer. False: place first in layer
        tags:       tags associated with the object
        groups:     Assign the object to MOP groups. List of strings; or string with commas as separation
        '''
        points_elem = ET.Element("points", {"id": str(self.pid)})
        tag_elem = ET.SubElement(points_elem, "Tag")
        tag_elem.text = f"{groups}\n{tags}"
        pts = ET.SubElement(points_elem, "pts")
        for point in points:
            ET.SubElement(pts, "p").text = f"{point[0]},{point[1]}"

        self._group_object(self.pid, groups)
        self._place_object(points_elem, target, place_last)

        logger.debug(f"Added points to layer {target}")

    def obj_rect(self, rect_x: float, rect_y: float, width: float, height: float, target: str = '', place_last: bool = True, tags: str = '', groups: list[str] = None) -> None:
        '''Create a rectangle in target layer
        rect_x: Rectangle lower left corner x coordinate
        rect_y: Rectangle lower left corner y coordinate
        width: Rectangle width
        height: Rectangle height
        target: str. Layer to place in. Empty=first layer
        place_last: Place as last object in layer. False: place first in layer
        tags: tags associated with the object
        groups:     Assign the object to MOP groups. List of strings; or string with commas as separation
        '''
        rect = ET.Element("rect", {"id": str(self.pid), "Closed": "true", "p": f"{rect_x},{rect_y},0", "w": str(width), "h": str(height)})
        tag_elem = ET.SubElement(rect, "Tag")
        tag_elem.text = f"{groups}\n{tags}"

        self._group_object(self.pid, groups)
        self._place_object(rect, target, place_last)

        logger.debug(f"Added rect to layer {target}")

    def obj_text(self, text: str, text_x: float, text_y: float, target: str = '', height: float = 100, align_hori: str = 'center', align_vert: str = 'center',
                 font: str = 'Arial', style: str = '', linespace: float = 1, place_last: bool = True, tags: str = '', groups: list[str] = None) -> None:
        '''Create a text element in target layer
        text:       The text to write
        text_x:     Text x alignment coordinate
        text_y:     Text y alignment coordinate
        target:     str. Layer to place in. Empty=first layer
        height:     Text height (size)
        align_hori: Horizontal alignment (right, center, left)
        align_vert: Vertical Alignment (top, center, bottom)
        font:       Font to use
        style:      ('', 'bold', 'italic', 'bold+italic')
        linespace:  Spacing between lines
        place_last: Place as last object in layer. False: place first in layer
        tags:       tags associated with the object
        groups:     Assign the object to MOP groups. List of strings; or string with commas as separation
        '''
        text_elem = ET.Element("text", {
            "id": str(self.pid), "p1": f"{text_x},{text_y},0", "p2": f"{text_x},{text_y},0", "Height": str(height),
            "Font": font, "linespace": str(linespace), "align": f"{align_vert},{align_hori}", "style": style
        })
        tag_elem = ET.SubElement(text_elem, "Tag")
        tag_elem.text = f"{groups}\n{tags}"
        text_elem.text = TEXT_IDENTIFIER_PREFIX + text # Add text identifier prefix to all text elements, so we can identify them before saving, and remove the undesired whitespace.

        self._group_object(self.pid, groups)
        self._place_object(text_elem, target, place_last)

        logger.debug(f"Added text to layer {target}")

    def obj_arc(self, center_x: float, center_y: float, radius: float, start: float, rotation: float, target: str = '', place_last: bool = True, tags: str = '', groups: list[str] = None) -> None:
        '''Create a arc in target layer
        center_x:   The text to write
        center_y:   Text x alignment coordinate
        radius:     Radius of the full circle
        start:      Angle (deg) for arc start. 0=flat right from center
        rotation:   Rotation (deg) from start point. 360=full circle
        target:     str. Layer to place in. Empty=first layer
        place_last: Place as last object in layer. False: place first in layer
        tags:       tags associated with the object
        groups:     Assign the object to MOP groups. List of strings; or string with commas as separation
        '''
        arc = ET.Element("arc", {"id": str(self.pid), "p": f"{center_x},{center_y},0", "r": str(radius), "s": str(start), "w": str(rotation)})
        tag_elem = ET.SubElement(arc, "Tag")
        tag_elem.text = f"{groups}\n{tags}"

        self._group_object(self.pid, groups)
        self._place_object(arc, target, place_last)

        logger.debug(f"Added arc to layer {target}")

    def _place_mop(self, mop_elem: ET.Element, target_part: str, target_mop: str, place_last: bool) -> None:
        '''Internal function that places mop in target_part, at target_mop\n
        mop_elem:     the XML element making up the object to be injected
        target_part: part to place in
        target_mop:  mop to place before
        place_last:  Place as last/first in chosen part or relative chosen mop.
        '''
        if target_part == '' or target_part is None:
            parts = self.parts.findall('part')
            if not parts:
                self.part()
                parts = self.parts.findall('part')
            target_part_elem = parts[0]
        else:
            target_part_elem = None
            for part in self.parts.findall('part'):
                if part.get('Name') == target_part:
                    target_part_elem = part
                    break
            if not target_part_elem:
                self.part(name=target_part)
                for part in self.parts.findall('part'):
                    if part.get('Name') == target_part:
                        target_part_elem = part
                        break

        machineops = target_part_elem.find('machineops')
        if target_mop == '' or target_mop is None:
            if place_last:
                machineops.append(mop_elem)
            else:
                machineops.insert(0, mop_elem)
        else:
            target_mop_elem = None
            for mop in machineops:
                if mop.find('Name').text == target_mop:
                    target_mop_elem = mop
                    break
            if not target_mop_elem:
                machineops.append(mop_elem)
            else:
                index = list(machineops).index(target_mop_elem)
                if place_last:
                    machineops.insert(index + 1, mop_elem)
                else:
                    machineops.insert(index, mop_elem)

        logger.debug(f"Added MOP to part {target_part}")

    def _pid_list_to_txt(self, pid_list) -> str:
        '''Internal: Convert a list of primitive IDs to a string for MOP XML.\n
        Called by mop creator functions. Returns a str with xml structure for <prim>...</prim>\n
        pid_list: list of primitives to use in machine operation. Can alternatively give the group name to get all PIDs in the self.groups['group name']
        '''
        if isinstance(pid_list, str):
            try:
                pid_list = self.groups[pid_list]
            except KeyError:
                return ''

        pid_list_txt = ""
        if len(pid_list) != 0:
            for pid in pid_list:
                pid_list_txt += f"\n            <prim>{pid}</prim>"
        return pid_list_txt

    def mop_profile(self, name: str = 'new profile', pid_list: list = [], target_depth: float = 0, profile_side: str = 'Inside', tabs: int = 0,
                    roughing_clearance: float = 0, stepover: float = 0.4, header: str = '', footer: str = '', target_part: str = '', target_mop: str = '', place_last: bool = True) -> None:
        '''Create a profile MOP.'''
        if tabs > 0:
            tab_txt = f'''        <TabMethod>Automatic</TabMethod>
        <Width>6</Width>
        <Height>1.5</Height>
        <MinimumTabs>{tabs}</MinimumTabs>
        <MaximumTabs>{tabs}</MaximumTabs> 
        <TabDistance>40</TabDistance>
        <SizeThreshold>4</SizeThreshold>
        <UseLeadIns>False</UseLeadIns>
        <TabStyle>Square</TabStyle>'''
        else:
            tab_txt = '        <TabMethod>None</TabMethod>'

        mop_txt = f'''        <profile Enabled="true">
          <Name>{name}</Name>
          <TargetDepth state="Value">{target_depth}</TargetDepth>
          <DepthIncrement state="Value">{abs(target_depth)}</DepthIncrement>
          <FinalDepthIncrement state="Value">0</FinalDepthIncrement>
          <CutOrdering state="Value">DepthFirst</CutOrdering>
          <StepOver state="Value">{stepover}</StepOver>
          <InsideOutside state="Value">{profile_side}</InsideOutside>
          <MillingDirection state="Value">Conventional</MillingDirection>
          <CollisionDetection state="Value">True</CollisionDetection>
          <CornerOvercut state="Value">False</CornerOvercut>
          <LeadInMove state="Value">
            <LeadInType>Spiral</LeadInType>
            <SpiralAngle>30</SpiralAngle>
            <TangentRadius>0</TangentRadius>
            <LeadInFeedrate>0</LeadInFeedrate>
          </LeadInMove>
          <LeadOutMove state="Value">
            <LeadInType>Spiral</LeadInType>
            <SpiralAngle>30</SpiralAngle>
            <TangentRadius>0</TangentRadius>
            <LeadInFeedrate>0</LeadInFeedrate>
          </LeadOutMove>
          <RoughingClearance state="Value">{roughing_clearance}</RoughingClearance>
          <ClearancePlane state="Value">15</ClearancePlane>
          <SpindleDirection state="Value">CW</SpindleDirection>
          <SpindleSpeed state="Value">{self.spindle_speed}</SpindleSpeed>
          <SpindleRange state="Value">0</SpindleRange>
          <VelocityMode state="Value">ExactStop</VelocityMode>
          <WorkPlane state="Value">XY</WorkPlane>
          <OptimisationMode state="Value">Standard</OptimisationMode>
          <ToolDiameter state="Value">{self.endmill}</ToolDiameter>
          <ToolNumber state="Value">0</ToolNumber>
          <ToolProfile state="Value">EndMill</ToolProfile>
          <PlungeFeedrate state="Value">1000</PlungeFeedrate>
          <CutFeedrate state="Value">{round(350*target_depth + 6500, 0)}</CutFeedrate>
          <MaxCrossoverDistance state="Value">0.7</MaxCrossoverDistance>
          <HoldingTabs state="Value">
            {tab_txt}
          </HoldingTabs>
          <CustomMOPHeader state="Value">{header}</CustomMOPHeader>
          <CustomMOPFooter state="Value">{footer}</CustomMOPFooter>
          <primitive>{self._pid_list_to_txt(pid_list)}
          </primitive>
        </profile>'''

        mop_elem = ET.fromstring(mop_txt)
        self._place_mop(mop_elem, target_part, target_mop, place_last)

        logger.debug(f"Added profile MOP to part {target_part}")

    def mop_pocket(self, name: str = 'new pocket', pid_list: list = [], target_depth: float = 0, stepover: float = 0.4,
                   roughing_clearance: float = 0, header: str = '', footer: str = '', target_part: str = '', target_mop: str = '', place_last: bool = True) -> None:
        '''Create a pocket MOP.'''
        mop_txt = f'''        <pocket Enabled="true">
          <Name>{name}</Name>
          <TargetDepth state="Value">{target_depth}</TargetDepth>
          <DepthIncrement state="Value">{abs(target_depth)}</DepthIncrement>
          <FinalDepthIncrement state="Value">0</FinalDepthIncrement>
          <CutOrdering state="Value">DepthFirst</CutOrdering>
          <StepOver state="Value">{stepover}</StepOver>
          <StepoverFeedrate state="Value">Plunge Feedrate</StepoverFeedrate>
          <MillingDirection state="Value">Conventional</MillingDirection>
          <CollisionDetection state="Value">True</CollisionDetection>
          <LeadInMove state="Value">
            <LeadInType>Spiral</LeadInType>
            <SpiralAngle>30</SpiralAngle>
            <TangentRadius>0</TangentRadius>
            <LeadInFeedrate>0</LeadInFeedrate>
          </LeadInMove>
          <LeadOutMove state="Value">
            <LeadInType>Spiral</LeadInType>
            <SpiralAngle>30</SpiralAngle>
            <TangentRadius>0</TangentRadius>
            <LeadInFeedrate>0</LeadInFeedrate>
          </LeadOutMove>
          <RegionFillStyle state="Value">InsideOutsideOffsets</RegionFillStyle>
          <FinishStepover state="Value">0</FinishStepover>
          <FinishStepoverAtTargetDepth state="Value">False</FinishStepoverAtTargetDepth>
          <StockSurface state="Value">0</StockSurface>
          <RoughingClearance state="Value">{roughing_clearance}</RoughingClearance>
          <ClearancePlane state="Value">15</ClearancePlane>
          <SpindleDirection state="Value">CW</SpindleDirection>
          <SpindleSpeed state="Value">{self.spindle_speed}</SpindleSpeed>
          <SpindleRange state="Value">0</SpindleRange>
          <VelocityMode state="Value">ExactStop</VelocityMode>
          <WorkPlane state="Value">XY</WorkPlane>
          <OptimisationMode state="Value">Experimental</OptimisationMode>
          <RoughingFinishing state="Value">Roughing</RoughingFinishing>
          <ToolDiameter state="Value">{self.endmill}</ToolDiameter>
          <ToolNumber state="Value">0</ToolNumber>
          <ToolProfile state="Value">EndMill</ToolProfile>
          <PlungeFeedrate state="Value">1000</PlungeFeedrate>
          <CutFeedrate state="Value">{round(350*target_depth + 6500, 0)}</CutFeedrate>
          <MaxCrossoverDistance state="Value">0.7</MaxCrossoverDistance>
          <StartPoint state="Default" />
          <CustomMOPHeader state="Value">{header}</CustomMOPHeader>
          <CustomMOPFooter state="Value">{footer}</CustomMOPFooter>
          <primitive>{self._pid_list_to_txt(pid_list)}
          </primitive>
        </pocket>'''

        mop_elem = ET.fromstring(mop_txt)
        self._place_mop(mop_elem, target_part, target_mop, place_last)

        logger.debug(f"Added pocket MOP to part {target_part}")

    def mop_engrave(self, name: str = 'new engrave', pid_list: list = [], target_depth: float = 0,
                    roughing_clearance: float = 0, header: str = '', footer: str = '', target_part: str = '', target_mop: str = '', place_last: bool = True) -> None:
        '''Create an engrave MOP.'''
        mop_txt = f'''        <engrave Enabled="true">
          <Name>{name}</Name>
          <TargetDepth state="Value">{target_depth}</TargetDepth>
          <DepthIncrement state="Value">{abs(target_depth)}</DepthIncrement>
          <FinalDepthIncrement state="Value">0</FinalDepthIncrement>
          <CutOrdering state="Value">DepthFirst</CutOrdering>
          <StockSurface state="Value">0</StockSurface>
          <RoughingClearance state="Value">{roughing_clearance}</RoughingClearance>
          <ClearancePlane state="Value">15</ClearancePlane>
          <SpindleDirection state="Value">CW</SpindleDirection>
          <SpindleSpeed state="Value">{self.spindle_speed}</SpindleSpeed>
          <SpindleRange state="Value">0</SpindleRange>
          <VelocityMode state="Value">ExactStop</VelocityMode>
          <WorkPlane state="Value">XY</WorkPlane>
          <OptimisationMode state="Value">Standard</OptimisationMode>
          <RoughingFinishing state="Value">Roughing</RoughingFinishing>
          <ToolDiameter state="Value">{self.endmill}</ToolDiameter>
          <ToolNumber state="Value">0</ToolNumber>
          <ToolProfile state="Value">EndMill</ToolProfile>
          <PlungeFeedrate state="Value">1000</PlungeFeedrate>
          <CutFeedrate state="Value">{round(350*target_depth + 6500, 0)}</CutFeedrate>
          <MaxCrossoverDistance state="Value">0.7</MaxCrossoverDistance>
          <StartPoint state="Value" />
          <CustomMOPHeader state="Value">{header}</CustomMOPHeader>
          <CustomMOPFooter state="Value">{footer}</CustomMOPFooter>
          <primitive>{self._pid_list_to_txt(pid_list)}
          </primitive>
        </engrave>'''

        mop_elem = ET.fromstring(mop_txt)
        self._place_mop(mop_elem, target_part, target_mop, place_last)

    def mop_drill(self, name: str = 'new drill', pid_list: list = [], target_depth: float = 0, hole_diameter: float = 0,
                  roughing_clearance: float = 0, peck_distance: float = 0, header: str = '', footer: str = '', target_part: str = '', target_mop: str = '', place_last: bool = True) -> None:
        '''Create a drill MOP.'''
        drill_method_txt = '<DrillingMethod state="Value">CannedCycle</DrillingMethod>'
        if hole_diameter > 0:
            drill_method_txt += f'\n          <DrillingMethod state="Value">SpiralMill_CW</DrillingMethod>'
            drill_method_txt += f'\n          <HoleDiameter state="Value">{hole_diameter}</HoleDiameter>'

        mop_txt = f'''        <drill Enabled="true">
          <Name>{name}</Name>
          <TargetDepth state="Value">{target_depth}</TargetDepth>
          <DepthIncrement state="Value">{abs(target_depth)}</DepthIncrement>
          <DrillLeadOut state="Value">False</DrillLeadOut>
          <SpiralFlatBase state="Value">True</SpiralFlatBase>
          <LeadOutLength state="Value">0</LeadOutLength>
          {drill_method_txt}
          <PeckDistance state="Value">{peck_distance}</PeckDistance>
          <RetractHeight state="Value">5</RetractHeight>
          <Dwell state="Value">0</Dwell>
          <CustomScript state="Default" />
          <StockSurface state="Value">0</StockSurface>
          <RoughingClearance state="Value">{roughing_clearance}</RoughingClearance>
          <ClearancePlane state="Value">15</ClearancePlane>
          <SpindleDirection state="Value">CW</SpindleDirection>
          <SpindleSpeed state="Value">{self.spindle_speed}</SpindleSpeed>
          <SpindleRange state="Value">0</SpindleRange>
          <VelocityMode state="Value">ExactStop</VelocityMode>
          <WorkPlane state="Value">XY</WorkPlane>
          <OptimisationMode state="Value">Standard</OptimisationMode>
          <RoughingFinishing state="Value">Roughing</RoughingFinishing>
          <ToolDiameter state="Value">{self.endmill}</ToolDiameter>
          <ToolNumber state="Value">0</ToolNumber>
          <ToolProfile state="Value">EndMill</ToolProfile>
          <PlungeFeedrate state="Value">1000</PlungeFeedrate>
          <CutFeedrate state="Value">{round(350*target_depth + 6500, 0)}</CutFeedrate>
          <MaxCrossoverDistance state="Value">0.7</MaxCrossoverDistance>
          <StartPoint state="Default" />
          <CustomMOPHeader state="Value">{header}</CustomMOPHeader>
          <CustomMOPFooter state="Value">{footer}</CustomMOPFooter>
          <primitive>{self._pid_list_to_txt(pid_list)}
          </primitive>
        </drill>'''

        mop_elem = ET.fromstring(mop_txt)
        self._place_mop(mop_elem, target_part, target_mop, place_last)

    def save_state(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # New save_file method.
    def save_file(self, file_path: str) -> None:
        """Save the CamBam file to the specified path."""
        try:
            # Process the file path
            base, ext = os.path.splitext(file_path)
            if not base:
                base = self.name + datetime.now().strftime('(%d-%m-%y_%H.%M.%S)')
            file_path = base + '.cb'  # Ensure .cb ending

            # Create directory if it doesn't exist
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.isdir(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                logger.debug(f"Created directory: {dir_name}")

            # Generate the XML content and remove the artificially placed text identifiers and undesired whitespace before texts
            xml_content = self._prettify(self.root)

            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)

            logger.info(f"CamBam file saved successfully: {file_path}")

        except IOError as e:
            logger.error(f"Error saving CamBam file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while saving CamBam file: {e}")


# Main loop for testing
if __name__ == '__main__':
    file_name = 'My Test cb file'
    dir_name = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    file_path = os.path.join(dir_name, file_name)

    cb = CamBam(file_name)

    # Create new parts
    cb.part(name='main part')
    cb.part(name='final part')
    cb.part(name='middle part', target='final part', place_last=False)

    # Create layers
    cb.layer(name='Layer A', color='green')
    cb.layer(name='Layer B', color='red')

    # Insert layer before Layer B
    cb.layer(name='Middle Layer', color='Blue', target='Layer B', place_last=False)

    # Create objects in specified layers
    cb.obj_pline([(0, 100), (500, 700, -0.35), (800, 600), (450, 0, -0.5)], True, 'Layer A', False, groups='groove')

    cb.obj_circle(300, 500, 425, '', place_last=True, tags='tag1, tag2\nnewlinetag', groups='groove, slit')

    cb.obj_arc(500, 500, 710, -75, 245, 'Layer B', groups=['text'])

    cb.obj_points([(100, 50), (120, 25), (150, 60), (70, 40)], target='Layer B', groups='screwprimer')

    cb.obj_rect(200, 50, 600, 300, 'Layer A', groups='cutout')

    cb.obj_text('test text...\nnew line', 200, 400, 'Layer A', height=80, linespace=2, groups='text')

    print('groups', cb.groups)

    # Create MOP with PIDs from a list
    cb.mop_pocket(name='groove pocket', pid_list=cb.groups['groove'], target_depth=-4, roughing_clearance=-0.1)

    cb.mop_engrave(name='engrave text', pid_list='text', target_depth=-1, place_last=False)

    cb.mop_drill(name='screwprimers', target_depth=-2, pid_list='screwprimer', hole_diameter=6)

    cb.mop_profile(name='mop1', pid_list=cb.groups['cutout'], target_depth=-5, profile_side='Outside', tabs=0, roughing_clearance=0.5,
                  target_part='', target_mop='', place_last=True)

    cb.save_file(file_path=file_path)