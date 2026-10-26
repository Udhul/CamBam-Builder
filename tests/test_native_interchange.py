"""Direct consumers retain native board geometry, relationships, and notes."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from cambam_builder import CBProject, Vertex
from cambam_builder.native.reader import read_cambam_bytes, read_cambam_file
from cambam_builder.native.writer import serialize_cambam_bytes


class NativeInterchangeTests(unittest.TestCase):
    def test_cabinet_board_edit_save_reopen(self):
        project = CBProject("cabinet-side")
        project.units = "Millimeters"
        layer = project.add_layer("Board")
        outline = project.add_rect(layer, width=600, height=300,
                                   identifier="outline", groups=["board"])
        outline.native_tag = "Cabinet side A\nKeep marked face up."
        grooves = [project.add_rect(
            layer, corner=(x, 10), width=6, height=280,
            identifier=f"groove-{i}", parent=outline, groups=["board", "grooves"],
        ) for i, x in enumerate((80, 500), start=1)]
        holes = project.add_points(layer, [(32, 32), (568, 268)],
                                    identifier="holes", parent=outline, groups=["board"])
        layout_layer = project.add_layer("Layout")
        layout = project.add_points(layout_layer, [(0, 0), (640, 0)], identifier="layout")
        part = project.add_part("Board machining", stock_width=1250, stock_height=350,
                                stock_thickness=18, nesting_method="PointList",
                                nesting_point_list=layout)
        pocket = project.add_pocket_mop(part, grooves, identifier="grooves",
                                        target_depth=-6, tool_diameter=6)
        project.add_drill_mop(part, [holes], identifier="drill",
                              target_depth=-12, tool_diameter=5)
        cutout = project.add_profile_mop(part, [outline], identifier="cutout",
                                         profile_side="Outside", target_depth=-18,
                                         tool_diameter=6)
        cutout.native_tag = '{"instruction":"Cut last","user_id":"operator note"}'
        self.assertTrue(project.translate_primitive(outline, 20, 25))
        self.assertTrue(project.translate_primitive_z(outline, 2))
        original_ids = {p.user_identifier: p.internal_id for p in project.list_primitives()}
        output = Path("output")
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-interchange-", dir=output) as directory:
            first = Path(directory) / "original.cb"
            project.save(str(first))
            edited = read_cambam_file(str(first))
            self.assertIsNotNone(edited)
            edited.get_primitive("outline").width = 620
            edited.get_primitive("layout").vertices = [Vertex(0, 0), Vertex(660, 0)]
            edited.get_mop("grooves").target_depth = -7
            edited.get_primitive("outline").native_tag += "\nRevision B."
            self.assertTrue(edited.translate_primitive("outline", 5, -5))
            second = Path(directory) / "edited.cb"
            edited.save(str(second))
            reopened = read_cambam_file(str(second))
            self.assertIsNotNone(reopened)
            root = ET.parse(second).getroot()

        self.assertEqual("Millimeters", reopened.units)
        self.assertEqual("Millimeters", root.get("units"))
        self.assertEqual(original_ids,
                         {p.user_identifier: p.internal_id for p in reopened.list_primitives()})
        board = reopened.get_primitive("outline")
        self.assertEqual((620, 300), (board.width, board.height))
        self.assertEqual([(25, 20, 2), (645, 20, 2), (645, 320, 2), (25, 320, 2)],
                         board.get_absolute_coordinates_xyz())
        groove = reopened.get_primitive("groove-1")
        self.assertEqual((6, 280), (groove.width, groove.height))
        self.assertEqual([(105, 30, 2), (111, 30, 2), (111, 310, 2), (105, 310, 2)],
                         groove.get_absolute_coordinates_xyz())
        self.assertIs(board, reopened.get_parent_of_primitive(groove))
        self.assertEqual(["board", "grooves"], reopened.get_groups_of_primitive(groove))
        self.assertEqual({"groove-1", "groove-2"},
                         {p.user_identifier for p in reopened.get_primitives_in_group("grooves")})
        self.assertEqual([(57, 52, 2), (593, 288, 2)],
                         reopened.get_primitive("holes").get_absolute_coordinates_xyz())
        placement = reopened.get_part_nesting_points("Board machining")
        self.assertIs(placement, reopened.get_primitive("layout"))
        self.assertEqual([(0, 0, 0), (660, 0, 0)], placement.get_absolute_coordinates_xyz())
        self.assertIsNone(reopened.get_parent_of_primitive(placement))
        self.assertEqual(["grooves", "drill", "cutout"],
                         [m.user_identifier for m in reopened.get_mops_in_part("Board machining")])
        self.assertEqual({p.internal_id for p in grooves}, set(reopened.get_mop_targets("grooves")))
        self.assertEqual([holes.internal_id], reopened.get_mop_targets("drill"))
        self.assertEqual([board.internal_id], reopened.get_mop_targets("cutout"))
        self.assertEqual(pocket.internal_id, reopened.get_mop("grooves").internal_id)
        self.assertEqual(-7, reopened.get_mop("grooves").target_depth)
        self.assertEqual(-18, reopened.get_mop("cutout").target_depth)
        self.assertEqual("Cabinet side A\nKeep marked face up.\nRevision B.", board.native_tag)
        self.assertEqual(cutout.native_tag, reopened.get_mop("cutout").native_tag)
        point_id = root.find("./layers/layer[@name='Layout']/objects/points").get("id")
        self.assertEqual(point_id, root.findtext("./parts/part/Nesting/PointListID"))

    def test_independent_inches_input_remaps_native_references_without_conversion(self):
        # Deliberately native XML, without framework-generated metadata or IDs.
        source = b'''<CADFile Version="0.9.8.0" Name="inch-panel" units="Inches">
          <layers><layer name="Imported"><objects>
            <points id="205"><pts><p>0,0,0</p><p>25.5,0,0</p></pts>
              <Tag>Two panel placements</Tag></points>
            <rect id="81" p="1.25,2.5,0" w="24" h="12" Closed="true">
              <Tag>Grain along the 24 inch edge</Tag></rect>
            <rect id="7" p="2,3,0" w="0.25" h="11" Closed="true"/>
          </objects></layer></layers>
          <parts><part Name="Inch board" Enabled="true">
            <Stock><PMin>0,0,-0.75</PMin><PMax>50,15,0</PMax></Stock>
            <Nesting><NestMethod>PointList</NestMethod><PointListID>205</PointListID></Nesting>
            <machineops>
              <pocket><Name>Dado</Name><TargetDepth state="Value">-0.25</TargetDepth>
                <ToolDiameter state="Value">0.25</ToolDiameter>
                <primitive><prim>7</prim></primitive><Tag>Check groove fit</Tag></pocket>
              <profile><Name>Cutout</Name><TargetDepth state="Value">-0.75</TargetDepth>
                <ToolDiameter state="Value">0.25</ToolDiameter>
                <primitive><prim>81</prim></primitive><InsideOutside state="Value">Outside</InsideOutside>
              </profile>
            </machineops>
          </part></parts>
        </CADFile>'''
        imported = read_cambam_bytes(source, source_name="inch-panel.cb")
        self.assertEqual("Inches", imported.units)
        outline = next(p for p in imported.list_primitives() if getattr(p, "width", None) == 24)
        layout = imported.get_part_nesting_points("Inch board")
        self.assertEqual([(0, 0, 0), (25.5, 0, 0)], layout.get_absolute_coordinates_xyz())
        self.assertEqual("Two panel placements", layout.native_tag)
        outline.width = 23.5
        # Add geometry before export; source IDs must resolve by identity after remapping.
        first_layer = imported.add_layer("Registration", place_last=False)
        imported.add_circle(first_layer, (0, 0), 0.125, identifier="registration")
        content = serialize_cambam_bytes(imported)
        root = ET.fromstring(content)
        points = root.find("./layers/layer[@name='Imported']/objects/points")
        self.assertNotEqual("205", points.get("id"))
        self.assertEqual(points.get("id"), root.findtext("./parts/part/Nesting/PointListID"))
        reopened = read_cambam_bytes(content, source_name="edited-inch-panel.cb")
        self.assertEqual("Inches", reopened.units)
        self.assertEqual("Inches", root.get("units"))
        self.assertEqual((50, 15, 0.75),
                         (reopened.get_part("Inch board").stock_width,
                          reopened.get_part("Inch board").stock_height,
                          reopened.get_part("Inch board").stock_thickness))
        self.assertEqual(layout.internal_id, reopened.get_part_nesting_points("Inch board").internal_id)
        result = reopened.get_primitive(outline.internal_id)
        self.assertEqual((23.5, 12), (result.width, result.height))
        self.assertEqual([(1.25, 2.5, 0), (24.75, 2.5, 0), (24.75, 14.5, 0), (1.25, 14.5, 0)],
                         result.get_absolute_coordinates_xyz())
        self.assertEqual("Grain along the 24 inch edge", result.native_tag)
        mops = reopened.get_mops_in_part("Inch board")
        self.assertEqual(["Dado", "Cutout"], [m.name for m in mops])
        self.assertEqual([-0.25, -0.75], [m.target_depth for m in mops])
        self.assertEqual([0.25, 0.25], [m.tool_diameter for m in mops])
        self.assertEqual("Check groove fit", mops[0].native_tag)
        self.assertEqual([outline.internal_id], reopened.get_mop_targets(mops[1]))
        dado = reopened.get_primitive(reopened.get_mop_targets(mops[0])[0])
        self.assertEqual((0.25, 11), (dado.width, dado.height))


if __name__ == "__main__":
    unittest.main()
