"""End-to-end tests for multi-board, multi-face setup composition."""

from pathlib import Path
import tempfile
import unittest
import math

from cambam_builder import CBProject
from cambam_builder.native.manufacturing import (
    ComponentPlacement,
    SetupFrame,
    SetupStock,
    compose_setup,
)
from cambam_builder.native.reader import read_cambam_bytes


def board(name: str, width: float, height: float, thickness: float,
          material: str) -> CBProject:
    """Create a small caller-owned design with front and back Part selections."""
    project = CBProject(name, units="Millimeters")
    layer = project.add_layer("Design")
    outline = project.add_rect(layer, width=width, height=height,
                               identifier="outline", groups=["board"])
    front_shape = project.add_rect(
        layer, corner=(width * .12, height * .1), width=8, height=height * .7,
        identifier="front-groove", groups=["front-machining", "grooves"], parent=outline)
    back_shape = project.add_rect(
        layer, corner=(width * .28, height * .18), width=width * .24, height=7,
        identifier="back-slot", groups=["back-machining", "slots"], parent=outline)
    front = project.add_part("front", stock_width=width + 30, stock_height=height + 30,
                             stock_thickness=thickness, stock_material=material)
    back = project.add_part("back", stock_width=width + 30, stock_height=height + 30,
                            stock_thickness=thickness, stock_material=material)
    project.add_pocket_mop(front, target_group="front-machining", identifier="front-cut",
                           target_depth=-4, depth_increment=2, tool_diameter=4)
    project.add_pocket_mop(back, target_group="back-machining", identifier="back-cut",
                           target_depth=-2, depth_increment=1, tool_diameter=4)
    project.add_profile_mop(back, targets=[outline], identifier="edge-profile",
                            profile_side="Outside", target_depth=-thickness,
                            depth_increment=3, tool_diameter=6)
    if outline is None or front_shape is None or back_shape is None:
        raise RuntimeError("Could not create synthetic component")
    return project


class ManufacturingCompositionTests(unittest.TestCase):
    def _check_layout(self, project: CBProject, part_names: list[str],
                      stock_width: float, stock_height: float) -> None:
        bounds = []
        for name in part_names:
            component = name.rsplit("/", 1)[0]
            outline = project.get_primitive(component + "/outline")
            points = outline.get_absolute_coordinates()
            box = (min(p[0] for p in points), min(p[1] for p in points),
                   max(p[0] for p in points), max(p[1] for p in points))
            self.assertGreaterEqual(box[0], 0)
            self.assertGreaterEqual(box[1], 0)
            self.assertLessEqual(box[2], stock_width)
            self.assertLessEqual(box[3], stock_height)
            bounds.append(box)
        for index, first in enumerate(bounds):
            for second in bounds[index + 1:]:
                overlap_x = min(first[2], second[2]) - max(first[0], second[0])
                overlap_y = min(first[3], second[3]) - max(first[1], second[1])
                self.assertFalse(overlap_x > 0 and overlap_y > 0,
                                 f"Placed boards overlap: {first}, {second}")

    def _check_sheet_mirror(self, front: CBProject, back: CBProject,
                            components: list[str], stock_width: float) -> None:
        for component in components:
            front_entities = {p.user_identifier: p for p in front.list_primitives()
                              if p.user_identifier.startswith(component + "/")}
            back_entities = {p.user_identifier: p for p in back.list_primitives()
                             if p.user_identifier.startswith(component + "/")}
            self.assertEqual(set(front_entities), set(back_entities))
            for identifier, source in front_entities.items():
                front_points = source.get_absolute_coordinates()
                back_points = back_entities[identifier].get_absolute_coordinates()
                for x, y in front_points:
                    self.assertTrue(any(math.isclose(stock_width - x, bx, abs_tol=1e-9)
                                        and math.isclose(y, by, abs_tol=1e-9)
                                        for bx, by in back_points), identifier)

    def test_multi_board_face_batches_reflection_and_round_trip(self):
        side = board("side", 600, 700, 18, "Plywood")
        shelf = board("shelf", 560, 160, 18, "Plywood")
        panel = board("panel", 1200, 700, 6, "MDF")

        front = compose_setup(
            "cabinet-front", SetupStock(2000, 800, 18, "Plywood"),
            (ComponentPlacement(side, "outline", "left-side", ("front",),
                                SetupFrame(x=30, y=40)),
             ComponentPlacement(shelf, "outline", "shelf", ("front",),
                                SetupFrame(x=700, y=40)),
             ComponentPlacement(side, "outline", "right-side", ("front",),
                                SetupFrame(x=1370, y=40))),
        )
        back = compose_setup(
            "cabinet-back", SetupStock(2000, 800, 18, "Plywood"),
            (ComponentPlacement(side, "outline", "left-side", ("back",),
                                SetupFrame(x=1970, y=40, reflect_x=True)),
             ComponentPlacement(shelf, "outline", "shelf", ("back",),
                                SetupFrame(x=1300, y=40, reflect_x=True)),
             ComponentPlacement(side, "outline", "right-side", ("back",),
                                SetupFrame(x=630, y=40, reflect_x=True))),
        )
        mdf_front = compose_setup(
            "panel-front", SetupStock(1280, 760, 6, "MDF"),
            (ComponentPlacement(panel, "outline", "back-panel", ("front",),
                                SetupFrame(x=40, y=30)),),
        )
        mdf_back = compose_setup(
            "panel-back", SetupStock(1280, 760, 6, "MDF"),
            (ComponentPlacement(panel, "outline", "back-panel", ("back",),
                                SetupFrame(x=1240, y=30, reflect_x=True)),),
        )

        expected = {
            front: ["left-side/front", "shelf/front", "right-side/front"],
            back: ["left-side/back", "shelf/back", "right-side/back"],
            mdf_front: ["back-panel/front"],
            mdf_back: ["back-panel/back"],
        }
        for project, names in expected.items():
            self.assertEqual(names, [part.user_identifier for part in project.list_parts()])
            is_front = project in (front, mdf_front)
            for part in project.list_parts():
                self.assertEqual((18, "Plywood") if project in (front, back)
                                 else (6, "MDF"),
                                 (part.stock_thickness, part.stock_material))
                self.assertEqual((0.0, 0.0), part.machining_origin)
                mops = project.get_mops_in_part(part)
                expected_mops = ["front-cut"] if is_front else ["back-cut", "edge-profile"]
                self.assertEqual(expected_mops, [mop.user_identifier.rsplit("/", 1)[-1]
                                                 for mop in mops])
                mop = mops[0]
                targets = project.get_mop_targets(mop)
                self.assertEqual(1, len(targets))
                target = next(p for p in project.list_primitives() if p.internal_id in targets)
                component_name = part.user_identifier.rsplit("/", 1)[0]
                self.assertEqual(component_name + ("/front-groove" if is_front else "/back-slot"),
                                 target.user_identifier)
                self.assertIn(component_name + ("/front-machining" if is_front
                                                else "/back-machining"),
                              project.get_groups_of_primitive(target))
                # Each Part gets the shared setup stock dimensions, while source
                # design dimensions and source projects remain caller-owned.
                self.assertEqual((2000, 800) if project in (front, back)
                                 else (1280, 760),
                                 (part.stock_width, part.stock_height))

        # Back is registered against the full sheet width: x_back = 2000-x_front.
        # Left side therefore moves from x=30..630 to x=1370..1970. The slot's
        # 144 mm span and Z/MOP depth are preserved.
        slot = back.get_primitive("left-side/back-slot")
        coordinates = slot.get_absolute_coordinates()
        self.assertEqual((1802.0, 166.0), coordinates[0])
        self.assertEqual((1658.0, 166.0), coordinates[1])
        self.assertEqual(-2, back.get_mop("left-side/back-cut").target_depth)
        source_point = side.get_primitive("back-slot").get_absolute_coordinates()[0]
        self.assertAlmostEqual(168.0, source_point[0])
        self.assertAlmostEqual(126.0, source_point[1])

        self._check_layout(front, [p.user_identifier for p in front.list_parts()],
                           2000, 800)
        self._check_layout(back, [p.user_identifier for p in back.list_parts()],
                           2000, 800)
        self._check_layout(mdf_front, ["back-panel/front"], 1280, 760)
        self._check_layout(mdf_back, ["back-panel/back"], 1280, 760)
        self._check_sheet_mirror(front, back,
                                 ["left-side", "shelf", "right-side"], 2000)
        self._check_sheet_mirror(mdf_front, mdf_back, ["back-panel"], 1280)

        output_root = Path(__file__).resolve().parents[1] / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root, prefix="cf02-compose-test-") as temporary:
            for project in (front, back, mdf_front, mdf_back):
                path = Path(temporary) / f"{project.project_name}.cb"
                project.save(str(path))
                reopened = read_cambam_bytes(path.read_bytes())
                self.assertEqual([p.user_identifier for p in project.list_parts()],
                                 [p.user_identifier for p in reopened.list_parts()])
                for part in reopened.list_parts():
                    self.assertEqual((part.stock_material, part.stock_thickness,
                                      part.stock_width, part.stock_height),
                                     ("Plywood", 18, 2000, 800)
                                     if project in (front, back)
                                     else ("MDF", 6, 1280, 760))
                    operations = reopened.get_mops_in_part(part)
                    self.assertEqual([m.user_identifier.rsplit("/", 1)[-1]
                                      for m in project.get_mops_in_part(part.user_identifier)],
                                     [m.user_identifier.rsplit("/", 1)[-1] for m in operations])
                    for mop in operations:
                        targets = reopened.get_mop_targets(mop)
                        self.assertTrue(targets)
                        self.assertTrue(all(reopened.get_primitive(uid) is not None
                                            for uid in targets))
                self._check_layout(reopened, [p.user_identifier for p in reopened.list_parts()],
                                   reopened.list_parts()[0].stock_width,
                                   reopened.list_parts()[0].stock_height)

            front.get_mop("left-side/front-cut").target_depth = -5
            edited_path = Path(temporary) / "cabinet-front-edited.cb"
            front.save(str(edited_path))
            edited = read_cambam_bytes(edited_path.read_bytes())
            mop = edited.get_mop("left-side/front-cut")
            self.assertEqual(-5, mop.target_depth)
            self.assertEqual("left-side/front", edited.get_part_of_mop(mop).user_identifier)
            self.assertEqual(1, len(edited.get_mop_targets(mop)))


if __name__ == "__main__":
    unittest.main()
