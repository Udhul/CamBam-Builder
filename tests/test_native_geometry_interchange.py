"""Native Line/Spline consumers protect shape data, world pose and ownership."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder import CBProject, Vertex
from cambam_builder.native.cad import Line, Spline
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


class NativeGeometryInterchangeTests(unittest.TestCase):
    def test_authored_component_edits_two_file_cycles_and_copy(self):
        project = CBProject("arched-component")
        layer = project.add_layer("Design")
        base = project.add_line(layer, [(0, 0, 0), (4, 0, 1)], identifier="base",
                                groups=["component"])
        curve = project.add_spline(
            layer, [(0, 0, 1), (0, 3, 2), (4, 3, 3), (4, 0, 4)],
            knots=[0, 0, 0, 0, 1, 1, 1, 1], degree=3,
            identifier="curve", parent=base, groups=["component", "curves"],
            weights=[1, 2, 2, 1], fit_points=[(2, 18 / 7, 2.5)],
            flags=12, start_tangent=(1, 2, 3), end_tangent=(4, 5, 6),
            knot_tolerance=1e-7, control_point_tolerance=2e-7, fit_tolerance=1e-10,
        )
        base.native_tag = "Registration edge"
        curve.native_tag = "Keep cubic intent"
        part = project.add_part("Detail")
        curved_mop = project.add_engrave_mop(part, [curve], identifier="curve-pass",
                                            target_depth=-0.5, tool_diameter=1)
        line_mop = project.add_engrave_mop(part, [base], identifier="line-pass",
                                          target_depth=-0.25, tool_diameter=1)
        self.assertTrue(project.rotate_primitive_deg(base, 90, cx=0, cy=0))
        self.assertTrue(project.translate_primitive(base, 10, 20))
        self.assertTrue(project.translate_primitive_z(base, 2))
        ids = {entity.user_identifier: entity.internal_id
               for entity in [base, curve, curved_mop, line_mop]}
        Path("output").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-geometry-", dir="output") as directory:
            first, second = Path(directory) / "first.cb", Path(directory) / "second.cb"
            project.save(str(first))
            edited = read_cambam_bytes(first.read_bytes())
            edited.get_primitive("base").vertices[1] = Vertex(5, 0, 1)
            edited.get_primitive("curve").control_points[1] = Vertex(0, 4, 2)
            edited.get_primitive("curve").native_tag += "; revised"
            edited.get_mop("curve-pass").target_depth = -0.75
            self.assertTrue(edited.translate_primitive(base.internal_id, 5, -5))
            edited.save(str(second))
            result = read_cambam_bytes(second.read_bytes())
            xml = ET.parse(second).getroot()

        self.assertEqual(result.units, "Millimeters")
        for name, identity in ids.items():
            self.assertEqual(result.get_entity(name).internal_id, identity)
        base, curve = result.get_primitive("base"), result.get_primitive("curve")
        self.assertIsInstance(base, Line)
        self.assertIsInstance(curve, Spline)
        np.testing.assert_allclose(base.get_absolute_coordinates_xyz(),
                                   [(15, 15, 2), (15, 20, 3)], atol=1e-9)
        world = curve.get_absolute_coordinates_xyz()
        np.testing.assert_allclose(world["control_points"],
                                   [(15, 15, 3), (11, 15, 4), (12, 19, 5), (15, 19, 6)],
                                   atol=1e-9)
        np.testing.assert_allclose(world["fit_points"], [(15 - 18 / 7, 17, 4.5)], atol=1e-8)
        self.assertEqual(curve.knots, [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(curve.weights, [1, 2, 2, 1])
        self.assertEqual((curve.degree, curve.flags), (3, 12))
        self.assertEqual(world["start_tangent"], (1, 2, 3))
        self.assertEqual(world["end_tangent"], (4, 5, 6))
        self.assertEqual((curve.knot_tolerance, curve.control_point_tolerance,
                          curve.fit_tolerance), (1e-7, 2e-7, 1e-10))
        self.assertEqual(curve.native_tag, "Keep cubic intent; revised")
        self.assertIs(result.get_parent_of_primitive(curve), base)
        self.assertIs(result.get_layer_of_primitive(curve), result.get_layer("Design"))
        self.assertEqual(result.get_groups_of_primitive(curve), ["component", "curves"])
        self.assertEqual([m.user_identifier for m in result.get_mops_in_part("Detail")],
                         ["curve-pass", "line-pass"])
        self.assertEqual(result.get_mop_targets("curve-pass"), [curve.internal_id])
        self.assertEqual(result.get_mop_targets("line-pass"), [base.internal_id])
        self.assertEqual(result.get_mop("curve-pass").target_depth, -0.75)
        for tag, name in [("spline", "curve-pass"), ("line", "line-pass")]:
            native_id = xml.find(f"./layers/layer/objects/{tag}").get("id")
            mop = next(m for m in xml.findall("./parts/part/machineops/engrave")
                       if m.findtext("Name") == result.get_mop(name).name
                       and native_id == m.findtext("primitive/prim"))
            self.assertEqual(mop.findtext("primitive/prim"), native_id)

        clone = result.clone()
        self.assertIs(clone.get_primitive("curve").get_project(), clone)
        clone.get_primitive("curve").control_points[0].x = 99
        self.assertEqual(curve.control_points[0].x, 0)
        copied = CBProject("copied")
        mapping = result.copy_primitive_tree(base, copied, preserve_ids=False)
        copied_curve = copied.get_primitive(mapping[curve.internal_id])
        self.assertIsNot(copied_curve, curve)
        self.assertNotEqual(copied_curve.internal_id, curve.internal_id)
        np.testing.assert_allclose(copied_curve.get_absolute_coordinates_xyz()["control_points"],
                                   world["control_points"])
        self.assertEqual(copied.get_mop_targets("curve-pass"), [copied_curve.internal_id])
        self.assertTrue(copied.bake_primitive_transform("base"))
        self.assertEqual(copied_curve.start_tangent, (1, 2, 3))
        np.testing.assert_allclose(copied_curve.effective_transform, np.identity(3))
        np.testing.assert_allclose(copied_curve.get_absolute_coordinates_xyz()["control_points"],
                                   world["control_points"])
        reopened = read_cambam_bytes(serialize_cambam_bytes(copied))
        np.testing.assert_allclose(reopened.get_primitive("curve").get_absolute_coordinates_xyz()["control_points"],
                                   world["control_points"], atol=1e-9)

    def test_independent_inches_xml_nonconsecutive_ids_and_live_group(self):
        # Independently specified native fields, with no builder Tags or writer.
        source = b'''<CADFile Version="0.9.8.0" units="Inches">
          <layers><layer name="Imported"><objects>
            <spline id="93" deg="3" flags="12" ktol="1e-7" cptol="2e-7" ftol="1e-10"
                    starttan="1,2,3" endtan="4,5,6">
              <Tag>Native curved edge</Tag><mat m="Identity"/>
              <knots><k>0</k><k>0</k><k>0</k><k>0</k><k>1</k><k>1</k><k>1</k><k>1</k></knots>
              <weights><w>1</w><w>2</w><w>2</w><w>1</w></weights>
              <ctrl><p>0,0,1</p><p>0,3,2</p><p>4,3,3</p><p>4,0,4</p></ctrl>
              <fit><p>2,2.5,2.5</p></fit>
            </spline>
            <line id="7"><Tag>Native straight edge</Tag><pts>
              <p>0,0,1</p><p>4,0,4</p><p>5,1,4</p></pts></line>
          </objects></layer></layers>
          <parts><part Name="Native detail"><machineops>
            <engrave><Name>Curved first</Name><primitive><prim>93</prim></primitive>
              <ToolDiameter state="Value">0.125</ToolDiameter>
              <TargetDepth state="Value">-0.0625</TargetDepth></engrave>
            <profile><Name>Straight second</Name><primitive><prim>7</prim></primitive>
              <TargetDepth state="Value">-0.25</TargetDepth></profile>
          </machineops></part></parts>
        </CADFile>'''
        project = read_cambam_bytes(source)
        curve, line = project.get_primitive("spline_93"), project.get_primitive("line_7")
        self.assertEqual(project.units, "Inches")
        self.assertEqual(line.get_absolute_coordinates_xyz(), [(0, 0, 1), (4, 0, 4), (5, 1, 4)])
        curve.control_points[1] = Vertex(0, 3.25, 2)
        line.vertices[-1] = Vertex(5.5, 1, 4)
        project.add_primitive_to_group(curve, "details")
        project.add_primitive_to_group(line, "details")
        part = project.get_part("Native detail")
        group_mop = project.add_engrave_mop(part, identifier="group-pass", target_group="details")
        registration = project.add_layer("Registration", place_last=False)
        project.add_points(registration, [(0, 0)], identifier="origin")
        for _ in range(2):
            content = serialize_cambam_bytes(project)
            xml = ET.fromstring(content)
            native_curve = xml.find("./layers/layer/objects/spline")
            self.assertNotEqual(native_curve.get("id"), "93")
            self.assertEqual(xml.findtext("./parts/part/machineops/engrave/primitive/prim"),
                             native_curve.get("id"))
            project = read_cambam_bytes(content)
            current_curve = project.get_primitive(curve.internal_id)
            self.assertEqual(project.units, "Inches")
            self.assertEqual(current_curve.control_points[1], Vertex(0, 3.25, 2))
            self.assertEqual(current_curve.weights, [1, 2, 2, 1])
            self.assertEqual(current_curve.native_tag, "Native curved edge")
            self.assertEqual(project.get_primitive(line.internal_id).vertices[-1], Vertex(5.5, 1, 4))
            mops = project.get_mops_in_part("Native detail")
            self.assertEqual([m.name for m in mops], ["Curved first", "Straight second", "Engrave"])
            self.assertEqual(project.get_mop_targets(mops[0]), [curve.internal_id])
            self.assertEqual(project.get_mop_targets(mops[1]), [line.internal_id])
            self.assertEqual(set(project.get_mop_targets(group_mop.internal_id)),
                             {curve.internal_id, line.internal_id})
            self.assertEqual(mops[0].tool_diameter, 0.125)
            self.assertEqual(mops[0].target_depth, -0.0625)


if __name__ == "__main__":
    unittest.main()
