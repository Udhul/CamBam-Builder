import copy
import json
import math
import unittest
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder.cad_transformations import (
    from_cambam_matrix_str,
    rotation_matrix_deg,
    scale_matrix,
    translation_matrix,
)
from cambam_builder.cambam_entities import Pline
from cambam_builder import CBProject
from cambam_builder.region import Region, parse_region_geometry


def square(x0, y0, x1, y1, z=0.0, *, transform=None, z_offset=0.0):
    kwargs = {}
    if transform is not None:
        kwargs["effective_transform"] = transform
    return Pline(
        relative_points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        vertex_z=[z] * 4,
        closed=True,
        local_z_offset=z_offset,
        **kwargs,
    )


def bulged_outer(z=3.0):
    # The bulge on the right edge is a clockwise-facing semicircle with its
    # extreme at X=90; its value is deliberately unrelated to Z.
    return Pline(
        relative_points=[(60, 0), (80, 0, 1), (80, 20), (60, 20)],
        vertex_z=[z] * 4,
        closed=True,
    )


class RegionConstructionTests(unittest.TestCase):
    def test_registered_contour_copy_captures_parent_world_pose_before_detaching(self):
        project = CBProject("registered-contour")
        parent = project.add_points("Geometry", [(0, 0)], local_z_offset=4)
        parent.effective_transform = translation_matrix(10, 20)
        selected = project.add_pline("Geometry", [(0, 0), (10, 0), (10, 10), (0, 10)],
                                     closed=True, vertex_z=[2] * 4, parent=parent,
                                     local_z_offset=3)
        before = selected.get_absolute_coordinates_xyz()
        region = project.add_region("Geometry", selected)
        self.assertEqual(region.get_absolute_coordinates_xyz()["outer_curve"], before)
        self.assertIsNone(region.outer_curve.get_project())
        self.assertIs(project.get_parent_of_primitive(selected), parent)

    def test_owns_deep_copied_unregistered_contours_and_preserves_winding(self):
        outer = square(0, 0, 20, 20, z=2)
        hole = Pline(
            relative_points=[(5, 5), (5, 10), (10, 10), (10, 5)],
            vertex_z=[2] * 4,
            closed=True,
            user_identifier="input-hole",
        )
        original_outer_points = list(outer.relative_points)
        region = Region(
            outer_curve=outer,
            hole_curves=[hole],
            user_identifier="region",
        )

        self.assertIsNot(region.outer_curve, outer)
        self.assertIsNot(region.hole_curves[0], hole)
        self.assertIsNone(region.outer_curve.get_project())
        self.assertIsNone(region.hole_curves[0].get_project())
        self.assertEqual(region.hole_curves[0].relative_points, hole.relative_points)

        outer.relative_points[0] = (-100, -100)
        hole.vertex_z[0] = 99
        self.assertEqual(region.outer_curve.relative_points, original_outer_points)
        self.assertEqual(region.hole_curves[0].vertex_z, [2.0] * 4)

    def test_accepts_bulged_outer_and_analytic_bounds_include_arc_extreme(self):
        region = Region(
            outer_curve=bulged_outer(),
            hole_curves=[square(64, 4, 68, 8, z=3), square(70, 12, 74, 16, z=3)],
        )
        bounds = region.get_bounding_box()
        self.assertEqual(
            (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y),
            (60.0, 0.0, 90.0, 20.0),
        )

    def test_accepts_two_semicircle_circle_without_false_self_overlap(self):
        circle = Pline(
            relative_points=[(-10, 0, 1), (10, 0, 1)],
            vertex_z=[-2, -2],
            closed=True,
        )
        region = Region(outer_curve=circle, hole_curves=[square(-2, -2, 2, 2, z=-2)])
        bounds = region.get_bounding_box()
        self.assertAlmostEqual(bounds.min_y, -10.0)
        self.assertAlmostEqual(bounds.max_y, 10.0)

    def test_outside_hole_at_arc_endpoint_height_is_rejected(self):
        outer = Pline(relative_points=[(0, 0, -1), (10, 0), (10, 10), (0, 10)],
                      closed=True)
        for dy in (0, 1e-7, -1e-7):
            with self.subTest(dy=dy), self.assertRaisesRegex(ValueError, "strictly inside"):
                Region(outer_curve=outer, hole_curves=[square(-1, dy, -.5, 1 + dy)])
        # The same exact endpoint height must remain usable for valid holes.
        circular = Pline(relative_points=[(-10, 0, 1), (10, 0, 1)], closed=True)
        self.assertIsInstance(Region(outer_curve=circular, hole_curves=[square(-1, 0, 1, 1)]), Region)

    def test_small_well_conditioned_contour_scale_preserves_useful_geometry(self):
        outer = square(0, 0, 1e9, 1e9, transform=scale_matrix(1e-8))
        region = Region(outer_curve=outer, hole_curves=[square(2, 2, 3, 3)])
        bounds = region.get_bounding_box()
        np.testing.assert_allclose((bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y),
                                   (0, 0, 10, 10), rtol=0, atol=1e-10)
        region.bake_geometry()
        self.assertEqual(region.outer_curve.relative_points[2], (10., 10., 0.))

    def test_rejects_coincident_overlapping_arcs(self):
        doubled_arc = Pline(
            relative_points=[(-10, 0, 1), (10, 0, -1)],
            closed=True,
        )
        with self.assertRaisesRegex(ValueError, "overlapping segments"):
            Region(outer_curve=doubled_arc)

    def test_topology_tolerance_is_translation_independent(self):
        offset = 1.0e12
        translated = Region(
            outer_curve=square(offset, offset, offset + 20, offset + 20),
            hole_curves=[square(offset + 3, offset + 3, offset + 6, offset + 6)],
        )
        self.assertEqual(translated.get_geometric_center(), (offset + 10, offset + 10))

    def test_rejects_invalid_contours_and_topology(self):
        cases = {
            "open outer": lambda: Region(
                outer_curve=Pline(relative_points=[(0, 0), (10, 0), (0, 10)], closed=False)
            ),
            "nonfinite": lambda: Region(
                outer_curve=Pline(
                    relative_points=[(0, 0), (math.nan, 0), (0, 10)], closed=True
                )
            ),
            "zero segment": lambda: Region(
                outer_curve=Pline(
                    relative_points=[(0, 0), (10, 0), (10, 0), (0, 10)], closed=True
                )
            ),
            "self crossing": lambda: Region(
                outer_curve=Pline(
                    relative_points=[(0, 0), (10, 10), (0, 10), (10, 0)], closed=True
                )
            ),
            "outside hole": lambda: Region(
                outer_curve=square(0, 0, 10, 10),
                hole_curves=[square(20, 20, 21, 21)],
            ),
            "touching hole": lambda: Region(
                outer_curve=square(0, 0, 10, 10),
                hole_curves=[square(0, 2, 2, 4)],
            ),
            "crossing hole": lambda: Region(
                outer_curve=square(0, 0, 10, 10),
                hole_curves=[square(8, 2, 12, 4)],
            ),
            "nested holes": lambda: Region(
                outer_curve=square(0, 0, 20, 20),
                hole_curves=[square(2, 2, 15, 15), square(4, 4, 6, 6)],
            ),
            "touching holes": lambda: Region(
                outer_curve=square(0, 0, 20, 20),
                hole_curves=[square(2, 2, 6, 6), square(6, 2, 10, 6)],
            ),
            "noncoplanar": lambda: Region(
                outer_curve=square(0, 0, 20, 20, z=1),
                hole_curves=[square(2, 2, 6, 6, z=2)],
            ),
        }
        for name, build in cases.items():
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                build()

    def test_rejects_curved_contour_under_nonsimilarity_matrix(self):
        shear = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        outer = bulged_outer()
        outer.effective_transform = shear
        with self.assertRaisesRegex(ValueError, "elliptical Region topology is unsupported"):
            Region(outer_curve=outer)


class RegionGeometryTests(unittest.TestCase):
    def test_xy_xyz_queries_compose_region_and_contour_poses(self):
        contour_transform = translation_matrix(1, 2)
        outer = square(0, 0, 10, 10, z=3, transform=contour_transform, z_offset=4)
        region = Region(
            outer_curve=outer,
            effective_transform=translation_matrix(10, 20),
            local_z_offset=-2,
        )

        legacy = region.get_absolute_coordinates()
        xyz = region.get_absolute_coordinates_xyz()
        self.assertEqual(legacy["outer_curve"][0], (11.0, 22.0, 0.0))
        self.assertEqual(xyz["outer_curve"][0], (11.0, 22.0, 5.0, 0.0))

    def test_similarity_bake_preserves_world_geometry_z_and_reflected_arc(self):
        region = Region(
            outer_curve=bulged_outer(z=3),
            effective_transform=translation_matrix(5, -2) @ rotation_matrix_deg(90),
            local_z_offset=2,
        )
        before = region.get_absolute_coordinates_xyz()
        region.bake_geometry()

        np.testing.assert_array_equal(region.effective_transform, np.identity(3))
        self.assertEqual(region.local_z_offset, 2.0)
        np.testing.assert_allclose(
            region.get_absolute_coordinates_xyz()["outer_curve"],
            before["outer_curve"],
            rtol=0,
            atol=1e-10,
        )

        reflected = Region(outer_curve=bulged_outer())
        reflected.bake_geometry(scale_matrix(-1, 1))
        self.assertEqual(reflected.outer_curve.relative_points[1][2], -1)
        bounds = reflected.get_bounding_box()
        self.assertAlmostEqual(bounds.min_x, -90.0)
        self.assertAlmostEqual(bounds.max_x, -60.0)

    def test_curved_nonsimilarity_bake_fails_atomically(self):
        region = Region(outer_curve=bulged_outer())
        before_points = copy.deepcopy(region.outer_curve.relative_points)
        before_matrix = region.outer_curve.effective_transform.copy()
        shear = np.array([[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        with self.assertRaisesRegex(ValueError, "non-similarity"):
            region.bake_geometry(shear)
        self.assertEqual(region.outer_curve.relative_points, before_points)
        np.testing.assert_array_equal(region.outer_curve.effective_transform, before_matrix)

    def test_straight_contours_can_bake_general_affine_transform(self):
        shear = np.array([[1.0, 0.5, 3.0], [0.25, 1.0, -2.0], [0.0, 0.0, 1.0]])
        region = Region(
            outer_curve=square(0, 0, 20, 20),
            hole_curves=[square(3, 3, 6, 6)],
        )
        region.bake_geometry(shear)
        np.testing.assert_allclose(region.outer_curve.relative_points[0][:2], (3, -2))
        np.testing.assert_array_equal(region.outer_curve.effective_transform, np.identity(3))

    def test_bake_normalizes_owned_xy_and_z_poses(self):
        region = Region(
            outer_curve=square(
                0, 0, 20, 20, z=3,
                transform=translation_matrix(2, 3), z_offset=4,
            ),
            hole_curves=[square(
                4, 4, 8, 8, z=5,
                transform=translation_matrix(2, 3), z_offset=2,
            )],
            effective_transform=translation_matrix(10, 20),
            local_z_offset=2,
        )
        before = region.get_absolute_coordinates_xyz()
        region.bake_geometry()

        after = region.get_absolute_coordinates_xyz()
        np.testing.assert_allclose(after["outer_curve"], before["outer_curve"], atol=1e-10)
        np.testing.assert_allclose(after["hole_curves"][0], before["hole_curves"][0], atol=1e-10)
        np.testing.assert_array_equal(region.effective_transform, np.identity(3))
        self.assertEqual(region.local_z_offset, 2.0)
        for contour in region.contours:
            np.testing.assert_array_equal(contour.effective_transform, np.identity(3))
            self.assertEqual(contour.local_z_offset, 0.0)
            self.assertEqual(contour.vertex_z, [7.0] * 4)

    def test_shift_geometry_z_rejects_overflow_atomically(self):
        region = Region(outer_curve=square(0, 0, 10, 10, z=1e308))
        before = list(region.outer_curve.vertex_z)
        with self.assertRaisesRegex(ValueError, "finite"):
            region.shift_geometry_z(1e308)
        self.assertEqual(region.outer_curve.vertex_z, before)


class RegionXmlTests(unittest.TestCase):
    XML = """\
<entity xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" id="12" xsi:type="Region">
  <ModificationCount>0</ModificationCount>
  <mat m="Identity" />
  <OuterCurve Closed="true">
    <ModificationCount>0</ModificationCount>
    <mat m="1 0 0 0 0 1 0 0 0 0 1 0 0 0 2 1" />
    <pts>
      <p b="0">0,0,3</p><p b="1">20,0,3</p>
      <p b="0">20,20,3</p><p b="0">0,20,3</p>
    </pts>
  </OuterCurve>
  <HoleCurves>
    <Polyline Closed="true">
      <ModificationCount>0</ModificationCount><mat m="Identity" />
      <pts><p b="0">3,3,5</p><p b="0">6,3,5</p><p b="0">6,6,5</p><p b="0">3,6,5</p></pts>
    </Polyline>
  </HoleCurves>
</entity>
"""

    def test_parse_typed_schema_retains_xyz_bulge_and_contour_matrix(self):
        kwargs = parse_region_geometry(ET.fromstring(self.XML))
        region = Region(**kwargs)

        self.assertEqual(region.outer_curve.relative_points[1], (20.0, 0.0, 1.0))
        self.assertEqual(region.outer_curve.vertex_z, [3.0] * 4)
        self.assertEqual(region.outer_curve.local_z_offset, 2.0)
        self.assertEqual(region.hole_curves[0].vertex_z, [5.0] * 4)
        self.assertEqual(
            [point[2] for point in region.get_absolute_coordinates_xyz()["outer_curve"]],
            [5.0] * 4,
        )

    def test_xml_is_typed_entity_with_identity_only_on_region(self):
        parent_id = uuid.uuid4()
        region = Region(
            outer_curve=bulged_outer(),
            hole_curves=[square(64, 4, 68, 8, z=3)],
            user_identifier="typed-region",
            description="owned contours",
            groups=["cuttable"],
            local_z_offset=7,
        )
        element = region.to_xml_element(42, parent_id)

        self.assertEqual(element.tag, "entity")
        self.assertEqual(element.get("xsi:type"), "Region")
        self.assertEqual(element.get("id"), "42")
        tag = json.loads(element.findtext("Tag"))
        self.assertEqual(tag["user_id"], "typed-region")
        self.assertEqual(tag["parent"], str(parent_id))
        _, region_z = from_cambam_matrix_str(element.find("mat").get("m"), return_z=True)
        self.assertEqual(region_z, 7.0)

        outer = element.find("OuterCurve")
        holes = element.find("HoleCurves")
        self.assertIsNotNone(outer)
        self.assertEqual(outer.get("Closed"), "true")
        self.assertEqual(len(holes.findall("Polyline")), 1)
        for contour in [outer, *holes.findall("Polyline")]:
            self.assertIsNone(contour.get("id"))
            self.assertIsNone(contour.find("Tag"))
        self.assertEqual(float(outer.findall("pts/p")[1].get("b")), 1.0)
        self.assertEqual(tuple(map(float, outer.findall("pts/p")[1].text.split(','))),
                         (80.0, 0.0, 3.0))

        wrapper = ET.Element(
            "objects", {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance"}
        )
        wrapper.append(element)
        reparsed = ET.fromstring(ET.tostring(wrapper, encoding="unicode"))[0]
        reconstructed = Region(**parse_region_geometry(reparsed))
        self.assertEqual(reconstructed.outer_curve.get_absolute_coordinates_xyz(),
                         region.outer_curve.get_absolute_coordinates_xyz())
        self.assertEqual(reconstructed.outer_curve.vertex_z, region.outer_curve.vertex_z)

    def test_parser_rejects_malformed_or_lossy_contour_data(self):
        malformed = [
            self.XML.replace('Closed="true"', 'Closed="false"', 1),
            self.XML.replace("0,0,3", "0,0"),
            self.XML.replace("0,0,3", "nan,0,3"),
            self.XML.replace('b="0"', 'b="inf"', 1),
            self.XML.replace("</HoleCurves>", "<Circle /></HoleCurves>"),
            self.XML.replace("</pts>", "<q /></pts>", 1),
            self.XML.replace("<OuterCurve", "<WrongCurve", 1).replace(
                "</OuterCurve>", "</WrongCurve>", 1
            ),
        ]
        for xml in malformed:
            with self.subTest(xml=xml[:80]), self.assertRaises(ValueError):
                parse_region_geometry(ET.fromstring(xml))

    def test_export_revalidates_mutable_geometry_and_world_transform(self):
        region = Region(
            outer_curve=square(0, 0, 20, 20),
            hole_curves=[square(3, 3, 6, 6)],
        )
        region.hole_curves[0].relative_points = square(30, 30, 32, 32).relative_points
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            region.to_xml_element(1, None)

        region = Region(outer_curve=square(0, 0, 20, 20))
        region.effective_transform = scale_matrix(0, 1)
        with self.assertRaisesRegex(ValueError, "nonsingular"):
            region.to_xml_element(1, None)

    def test_export_rejects_rounding_that_makes_hole_touch_outer(self):
        region = Region(outer_curve=square(0, 0, 1, 1),
                        hole_curves=[square(1.5e-10, .2, .2, .3)])
        with self.assertRaisesRegex(ValueError, "touches or crosses"):
            region.to_xml_element(1, None)
        region.output_decimals = 12
        element = region.to_xml_element(1, None)
        restored = Region(**parse_region_geometry(element))
        self.assertEqual(restored.hole_curves[0].relative_points[0][0], 1.5e-10)


if __name__ == "__main__":
    unittest.main()
