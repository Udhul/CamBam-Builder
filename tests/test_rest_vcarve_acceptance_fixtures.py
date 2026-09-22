"""Check independent acceptance references, not an unimplemented CAM engine.

The JSON corpus is intended for future backend/engine runners. These checks catch
corrupt inputs or golden values using analytic geometry and existing CAD validation.
They do not certify the future runner's Boolean, offset, sweep or path behavior.
"""

import json
import math
from pathlib import Path
import unittest

from cambam_builder.cad_entities import Pline
from cambam_builder.region import Region


CORPUS = Path(__file__).parent / "fixtures" / "rest_vcarve_acceptance.json"


def rectangle_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def polygon_area(points):
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) / 2


def simpson(function, start, end, steps=2000):
    width = (end - start) / steps
    return width / 3 * (function(start) + function(end) + sum(
        (4 if index % 2 else 2) * function(start + index * width)
        for index in range(1, steps)))


class RestVcarveReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        cls.cases = {case["kind"]: case for case in cls.corpus["cases"]}

    def values(self, kind):
        case = self.cases[kind]
        return case["input"], case["expected"]

    def close(self, actual, expected):
        self.assertAlmostEqual(actual, expected,
                               delta=self.corpus["reference_numeric_tolerance"])

    def test_corpus_identity_and_explicit_units(self):
        cases = self.corpus["cases"]
        self.assertEqual(self.corpus["version"], 1)
        self.assertEqual(self.corpus["units"], "mm")
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertEqual(len(self.cases), len(cases))
        for value in self.corpus["backend_limits"].values():
            self.assertGreater(value, 0)

    def test_rectangle_overlay_reference(self):
        source, expected = self.values("rectangle_overlay")
        a, b = source["a"], source["b"]
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1]))
        self.close(overlap, expected["intersection_area"])
        self.close(rectangle_area(a) + rectangle_area(b) - overlap,
                   expected["union_area"])
        self.close(rectangle_area(a) - overlap, expected["difference_area"])
        self.assertGreater(overlap, 0)
        self.assertEqual(expected["union_components"], 1)

    def test_square_corner_reachability_reference(self):
        source, expected = self.values("square_reachability")
        radii = source["tool_radii"]
        for radius, rest in zip(radii, expected["rest_areas"]):
            self.assertGreater(source["side"], 2 * radius)
            # Four corner squares minus four quarter-disks.
            self.close(4 * radius ** 2 - math.pi * radius ** 2, rest)
        self.close(expected["rest_areas"][0] - expected["rest_areas"][1],
                   expected["second_tool_gain"])
        self.assertEqual(expected["rest_components"], 4)

    def test_profile_and_ideal_pocket_have_different_removal(self):
        source, expected = self.values("profile_versus_pocket")
        side, radius, allowance = (source[key] for key in ("side", "radius", "allowance"))
        center_side = side - 2 * (radius + allowance)
        outer_sweep = center_side ** 2 + 4 * center_side * radius + math.pi * radius ** 2
        inner_side = center_side - 2 * radius
        profile = outer_sweep - inner_side ** 2
        pocket = (side - 2 * allowance) ** 2 - (4 - math.pi) * radius ** 2
        self.close(profile, expected["profile_sweep_area"])
        self.close(side ** 2 - profile, expected["profile_rest_area"])
        self.close(inner_side, expected["profile_unswept_center_side"])
        self.close(pocket, expected["ideal_pocket_sweep_area"])
        self.close(side ** 2 - pocket, expected["ideal_pocket_rest_area"])
        self.close(side ** 2 - (side - 2 * allowance) ** 2,
                   expected["intentional_allowance_area"])
        lower, upper = radius + allowance, side - radius - allowance
        self.assertEqual(source["profile_centers"],
                         [[lower, lower], [upper, lower], [upper, upper], [lower, upper]])
        self.assertTrue(source["closed"])

    def test_annulus_section_reference(self):
        source, expected = self.values("annulus_inset")
        outer = source["outer_radius"] - source["inset"]
        inner = source["inner_radius"] + source["inset"]
        self.close(outer, expected["outer_radius"])
        self.close(inner, expected["inner_radius"])
        self.close(math.pi * (outer ** 2 - inner ** 2), expected["area"])
        self.assertGreater(outer, inner)
        self.assertGreater(inner, 0)
        self.assertEqual((expected["components"], expected["holes"]), (1, 1))

    def test_v_targets_sections_and_independent_volume_integration(self):
        for kind in ("square_v_target", "circle_v_target"):
            with self.subTest(kind=kind):
                source, expected = self.values(kind)
                slope = math.tan(math.radians(source["included_angle_degrees"] / 2))
                if kind == "square_v_target":
                    area = lambda depth: max(0, source["side"] - 2 * slope * depth) ** 2
                    for depth, section in zip(source["section_depths"], expected["section_areas"]):
                        self.close(area(depth), section)
                else:
                    area = lambda depth: math.pi * max(0, source["radius"] - slope * depth) ** 2
                    for depth, radius in zip(source["section_depths"], expected["section_radii"]):
                        self.close(source["radius"] - slope * depth, radius)
                self.close(simpson(area, 0, source["cap_depth"]), expected["volume"])
                self.close(area(source["cap_depth"]), expected["flat_floor_area"])

    def test_cone_profile_limits(self):
        source, expected = self.values("cone_profiles")
        slope = math.tan(math.radians(source["included_angle_degrees"] / 2))
        for flat, depth, end in zip(source["flat_radii"], expected["penetrations"],
                                    expected["cone_end_heights"]):
            self.close(flat + depth * slope, source["available_radius"])
            self.close(flat + end * slope, source["maximum_radius"])
        self.assertLess(expected["flat_tip_unreachable_clearance"], max(source["flat_radii"]))

    def test_spherical_join_and_finite_cone_height(self):
        source, expected = self.values("spherical_cone_join")
        alpha = math.radians(source["included_angle_degrees"] / 2)
        ball = source["ball_radius"]
        height = ball * (1 - math.sin(alpha))
        radius = math.sqrt(2 * ball * height - height ** 2)
        self.close(height, expected["join_height"])
        self.close(radius, expected["join_radius"])
        self.close((ball - height) / radius, expected["join_slope"])
        self.close(radius + (expected["cone_end_height"] - height) * math.tan(alpha),
                   source["maximum_radius"])

    def test_capsule_sweep_reference(self):
        source, expected = self.values("capsule_sweep")
        length = math.dist(source["start"], source["end"])
        radius = source["radius"]
        self.close(2 * radius * length + math.pi * radius ** 2, expected["area"])
        self.assertEqual([-radius, -radius, length + radius, radius], expected["bounds"])

    def test_affine_radius_sweep_by_independent_support_integration(self):
        source, expected = self.values("affine_radius_sweep")
        length = math.dist(source["start"], source["end"])
        r0, r1 = source["start_radius"], source["end_radius"]
        switch = math.acos((r0 - r1) / length)
        # Area = 1/2 integral(h^2 - h'^2); symmetry removes the 1/2.
        # Integrate each smooth support-function branch separately.
        area = simpson(lambda angle: (length * math.cos(angle) + r1) ** 2
                       - (length * math.sin(angle)) ** 2, 0, switch)
        area += r0 ** 2 * (math.pi - switch)
        self.close(area, expected["area"])
        self.assertEqual([-r0, -r1, length + r1, r1], expected["bounds"])

    def test_exact_fit_has_lower_dimensional_centers(self):
        source, expected = self.values("exact_fit_centers")
        x0, y0, x1, y1 = source["rectangle"]
        radius = source["tool_radius"]
        self.assertEqual(y1 - y0, 2 * radius)
        self.assertEqual([[x0 + radius, y0 + radius], [x1 - radius, y1 - radius]],
                         expected["rectangle_center_segment"])
        self.assertEqual(source["circle_radius"], radius)
        self.assertEqual(source["circle_center"], expected["circle_center_point"])
        self.assertEqual(expected["center_area"], 0)
        self.assertTrue(expected["nominal_feasible"])

    def test_near_contact_reference_and_bridge_bottleneck(self):
        source, expected = self.values("near_contact_overlay")
        for gap, area, components in zip(source["horizontal_gaps"], expected["union_areas"],
                                         expected["interior_components"]):
            self.close(2 * source["side"] ** 2 + min(0, gap) * source["side"], area)
            self.assertEqual(2 if gap > 0 else 1, components)
        source, expected = self.values("narrow_bridge")
        self.close(sum(rectangle_area(source[key]) for key in ("left_box", "right_box", "bridge")),
                   expected["initial_area"])
        self.assertLess(source["bridge"][3] - source["bridge"][1], 2 * source["inset"])
        # A separating section x=5 disappears, but each box retains an interior.
        self.assertGreater(source["left_box"][2] - source["left_box"][0], 2 * source["inset"])
        self.assertEqual((expected["initial_components"], expected["inset_components"],
                          expected["inset_holes"]), (1, 2, 0))

    def test_general_region_input_topology_and_area(self):
        source, expected = self.values("region_composition")
        def contour(points):
            return Pline(vertices=[(x, y, 0) for x, y in points], closed=True)
        region = Region(outer_curve=contour(source["outer"]),
                        hole_curves=[contour(hole) for hole in source["holes"]])
        region.validate_geometry()
        self.close(polygon_area(source["outer"]) - sum(polygon_area(hole) for hole in source["holes"]),
                   expected["opening_area"])
        self.assertEqual(len(region.hole_curves), expected["holes"])
        self.assertEqual(expected["components"], 1)


if __name__ == "__main__":
    unittest.main()
