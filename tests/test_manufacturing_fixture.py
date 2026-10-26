"""End-to-end consumer checks for a distinct non-cabinet fixture job."""

from pathlib import Path
import tempfile
import unittest

from cambam_builder import CBProject
from cambam_builder.native.manufacturing import (
    ComponentPlacement,
    SetupFrame,
    SetupStock,
    compose_setup,
)
from cambam_builder.native.reader import read_cambam_file


def build_fixture_setups():
    """Build a compact independent sensor-fixture consumer."""
    design = CBProject("test-sensor-fixture")
    design.units = "Millimeters"
    design.default_tool_diameter = 6
    layer = design.add_layer("Fixture")
    blank = design.add_rect(layer, width=100, height=70, identifier="blank",
                            groups=["fixture", "outline"])
    recess = design.add_rect(layer, corner=(12, 11), width=22, height=15,
                             identifier="recess", parent=blank,
                             groups=["front", "pocket"])
    holes = design.add_points(layer, [(20, 19), (79, 27), (31, 58)],
                              identifier="holes", parent=blank,
                              groups=["front", "drilling"])
    relief = design.add_rect(layer, corner=(61, 43), width=25, height=8,
                             identifier="relief", parent=blank,
                             groups=["back", "relief"])
    front_part = design.add_part("front", stock_width=100, stock_height=70,
                                 stock_thickness=10, stock_material="Aluminum")
    back_part = design.add_part("back", stock_width=100, stock_height=70,
                                stock_thickness=10, stock_material="Aluminum")
    design.add_pocket_mop(front_part, [recess], identifier="front-pocket",
                          target_depth=-2.5, depth_increment=1, tool_diameter=6)
    design.add_drill_mop(front_part, [holes], identifier="front-drill",
                         target_depth=-10, depth_increment=3, tool_diameter=5)
    design.add_pocket_mop(back_part, [relief], identifier="back-relief",
                          target_depth=-1.5, depth_increment=.75, tool_diameter=4)
    design.add_profile_mop(back_part, [blank], identifier="back-profile",
                            target_depth=-10, depth_increment=2, tool_diameter=6)
    stock = SetupStock(300, 180, 10, "Aluminum")
    front = compose_setup(
        "test-fixture-front", stock,
        [ComponentPlacement(design, blank, "left", (front_part,), SetupFrame(15, 20)),
         ComponentPlacement(design, blank, "turned", (front_part,),
                            SetupFrame(185, 20, rotation_degrees=90))],
    )
    back = compose_setup(
        "test-fixture-back", stock,
        [ComponentPlacement(design, blank, "left", (back_part,),
                            SetupFrame(285, 20, reflect_x=True)),
         ComponentPlacement(design, blank, "turned", (back_part,),
                            SetupFrame(115, 20, rotation_degrees=-90, reflect_x=True))],
    )
    return {"front": front, "back": back}


class SensorFixtureConsumerTests(unittest.TestCase):
    def test_front_and_back_jobs_round_trip_with_manual_placements(self):
        setups = build_fixture_setups()
        self.assertEqual({"front", "back"}, set(setups))
        front, back = setups["front"], setups["back"]

        self.assertEqual(
            ["left/front-pocket", "left/front-drill",
             "turned/front-pocket", "turned/front-drill"],
            [m.user_identifier for m in front.list_mops()],
        )
        self.assertEqual(["left/back-relief", "left/back-profile",
                          "turned/back-relief", "turned/back-profile"],
                         [m.user_identifier for m in back.list_mops()])
        self.assertEqual(2, len(front.list_parts()))
        self.assertEqual(2, len(back.list_parts()))
        self.assertEqual([], [m for m in front.list_mops()
                              if "back-" in m.user_identifier])
        self.assertEqual([], [m for m in back.list_mops()
                              if "front-" in m.user_identifier])

        for project in setups.values():
            self.assertEqual("Millimeters", project.units)
            self.assertEqual(6.0, project.default_tool_diameter)
            for part in project.list_parts():
                self.assertEqual((300, 180, 10),
                                 (part.stock_width, part.stock_height, part.stock_thickness))
                self.assertEqual("Aluminum", part.stock_material)
                self.assertEqual((0, 0), part.machining_origin)

        left_holes = front.get_primitive("left/holes")
        turned_holes = front.get_primitive("turned/holes")
        self.assertEqual([(35, 39), (94, 47), (46, 78)],
                         left_holes.get_absolute_coordinates())
        self.assertEqual([(166, 40), (158, 99), (127, 51)],
                         turned_holes.get_absolute_coordinates())
        self.assertEqual([left_holes.internal_id],
                         front.get_mop_targets("left/front-drill"))
        self.assertEqual(-10, front.get_mop("left/front-drill").target_depth)
        self.assertEqual(["left/drilling", "left/front"],
                         front.get_groups_of_primitive("left/holes"))
        self.assertNotEqual(left_holes.internal_id, turned_holes.internal_id)

        back_left_holes = back.get_primitive("left/holes").get_absolute_coordinates()
        back_turned_holes = back.get_primitive("turned/holes").get_absolute_coordinates()
        self.assertEqual([(265, 39), (206, 47), (254, 78)], back_left_holes)
        self.assertEqual([(134, 40), (142, 99), (173, 51)], back_turned_holes)
        for front_points, back_points in ((left_holes.get_absolute_coordinates(), back_left_holes),
                                         (turned_holes.get_absolute_coordinates(), back_turned_holes)):
            self.assertEqual([(300 - x, y) for x, y in front_points], back_points)

        # Manually allocated rectangles meet at an edge and do not overlap.
        for project, names in ((front, ("left/blank", "turned/blank")),
                               (back, ("turned/blank", "left/blank"))):
            first = project.get_primitive(names[0]).get_absolute_coordinates()
            second = project.get_primitive(names[1]).get_absolute_coordinates()
            self.assertLessEqual(max(x for x, _ in first), min(x for x, _ in second))

        reflected_relief = back.get_primitive("left/relief")
        self.assertEqual([(224, 63), (199, 63), (199, 71), (224, 71)],
                         reflected_relief.get_absolute_coordinates())
        self.assertEqual([reflected_relief.internal_id],
                         back.get_mop_targets("left/back-relief"))
        self.assertEqual(-1.5, back.get_mop("left/back-relief").target_depth)
        self.assertEqual(["left/back", "left/relief"],
                         back.get_groups_of_primitive("left/relief"))

        Path("output").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cf02-fixture-", dir="output") as temporary:
            for side, source in setups.items():
                path = Path(temporary) / f"fixture-{side}.cb"
                source.save(str(path))
                reopened = read_cambam_file(str(path))
                self.assertIsNotNone(reopened)
                self.assertEqual([m.user_identifier for m in source.list_mops()],
                                 [m.user_identifier for m in reopened.list_mops()])
                self.assertEqual([p.stock_material for p in source.list_parts()],
                                 [p.stock_material for p in reopened.list_parts()])

            # A caller edit remains attached to the same geometry after export.
            reopened_front = read_cambam_file(str(Path(temporary) / "fixture-front.cb"))
            edited_mop = reopened_front.get_mop("left/front-pocket")
            edited_mop.target_depth = -3
            edited_path = Path(temporary) / "fixture-front-edited.cb"
            reopened_front.save(str(edited_path))
            final = read_cambam_file(str(edited_path))
            self.assertEqual(-3, final.get_mop("left/front-pocket").target_depth)
            self.assertEqual([final.get_primitive("left/recess").internal_id],
                             final.get_mop_targets("left/front-pocket"))


if __name__ == "__main__":
    unittest.main()
