"""Compose editable cabinet manufacturing setups with the base library.

Run from any directory with an installed cambam-builder:
    python manufacturing_cabinet.py /path/to/output-directory

Dimensions are illustrative millimeters. Board grouping and placement are
caller decisions. The files retain native CamBam MOPs; they do not generate
toolpaths or machine-ready G-code.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.native.manufacturing import (
    ComponentPlacement,
    SetupFrame,
    SetupStock,
    compose_setup,
)
from cambam_builder.native.reader import read_cambam_bytes


def _board(name: str, width: float, height: float, thickness: float,
           material: str) -> CBProject:
    """Create one board design with distinct front grooves and back slots."""
    project = CBProject(name, units="Millimeters")
    layer = project.add_layer("Board design")
    outline = project.add_rect(layer, width=width, height=height,
                               identifier="outline", groups=["board"])
    front_features = [
        project.add_rect(layer, corner=(width * .15, height * .08),
                         width=8, height=height * .78, identifier="front-groove-a",
                         groups=["front-machining", "grooves"], parent=outline),
        project.add_rect(layer, corner=(width * .72, height * .21),
                         width=6, height=height * .62, identifier="front-groove-b",
                         groups=["front-machining", "grooves"], parent=outline),
    ]
    back_features = [
        project.add_rect(layer, corner=(width * .28, height * .18),
                         width=width * .24, height=7, identifier="back-slot-a",
                         groups=["back-machining", "slots"], parent=outline),
        project.add_rect(layer, corner=(width * .54, height * .64),
                         width=width * .18, height=5, identifier="back-slot-b",
                         groups=["back-machining", "slots"], parent=outline),
    ]
    front = project.add_part(
        "front", stock_width=width + 40, stock_height=height + 40,
        stock_thickness=thickness, stock_material=material,
    )
    back = project.add_part(
        "back", stock_width=width + 40, stock_height=height + 40,
        stock_thickness=thickness, stock_material=material,
    )
    project.add_pocket_mop(front, target_group="front-machining",
                           identifier="front-grooves", name="Front grooves",
                           target_depth=-5 if thickness > 10 else -2,
                           depth_increment=2, tool_diameter=6, tool_number=1)
    project.add_pocket_mop(back, target_group="back-machining",
                           identifier="back-slots", name="Back slots",
                           target_depth=-4 if thickness > 10 else -1.5,
                           depth_increment=1.5, tool_diameter=4, tool_number=2)
    project.add_profile_mop(back, targets=[outline], identifier="edge-profile",
                            name="Board outline", profile_side="Outside",
                            target_depth=-thickness, depth_increment=3,
                            tool_diameter=6, tool_number=1)
    if outline is None or any(feature is None for feature in front_features + back_features):
        raise RuntimeError(f"Could not author {name}")
    return project


def build_cabinet_setups() -> dict[str, CBProject]:
    """Build manually placed plywood sides/shelf and an MDF back panel."""
    side = _board("cabinet-side", 600, 700, 18, "Birch plywood")
    shelf = _board("cabinet-shelf", 560, 160, 18, "Birch plywood")
    back_panel = _board("cabinet-back", 1200, 700, 6, "White melamine MDF")

    plywood_stock = SetupStock(2000, 800, 18, "Birch plywood")
    mdf_stock = SetupStock(1280, 780, 6, "White melamine MDF")

    def pair(source: CBProject, name: str, x_front: float, stock_width: float,
             y: float = 40,
             parts: tuple[str, str] = ("front", "back")) -> tuple[ComponentPlacement, ...]:
        """Register front and back faces across their full stock sheet."""
        return (
            ComponentPlacement(source, "outline", name, (parts[0],),
                               SetupFrame(x=x_front, y=y)),
            ComponentPlacement(source, "outline", name, (parts[1],),
                               SetupFrame(x=stock_width - x_front,
                                          y=y, reflect_x=True)),
        )

    side_left = pair(side, "left-side", 30, plywood_stock.width)
    shelf_middle = pair(shelf, "middle-shelf", 700, plywood_stock.width)
    side_right = pair(side, "right-side", 1370, plywood_stock.width)
    plywood = compose_setup(
        "cabinet-plywood-front", plywood_stock,
        (side_left[0], shelf_middle[0], side_right[0]),
    )
    plywood_back = compose_setup(
        "cabinet-plywood-back", plywood_stock,
        (side_left[1], shelf_middle[1], side_right[1]),
    )
    panel = pair(back_panel, "back-panel", 40, mdf_stock.width, y=30)
    mdf_front = compose_setup("cabinet-mdf-front", mdf_stock, (panel[0],))
    mdf_back = compose_setup("cabinet-mdf-back", mdf_stock, (panel[1],))
    return {
        "cabinet-plywood-front": plywood,
        "cabinet-plywood-back": plywood_back,
        "cabinet-mdf-front": mdf_front,
        "cabinet-mdf-back": mdf_back,
    }


def _bounds(primitive) -> tuple[float, float, float, float]:
    points = primitive.get_absolute_coordinates()
    return (min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points))


def _verify_output(project: CBProject, expected_parts: list[str],
                   expected_ops: dict[str, list[str]], stock: SetupStock,
                   expected_bounds: dict[str, tuple[float, float, float, float]]) -> None:
    """Check operations, stock and board layout after the native XML round trip."""
    actual_parts = [part.user_identifier for part in project.list_parts()]
    if actual_parts != expected_parts:
        raise RuntimeError(f"Unexpected parts in {project.project_name}: {actual_parts}")
    boxes = []
    for part_name in actual_parts:
        component_name = part_name.rsplit("/", 1)[0]
        part = project.get_part(part_name)
        if (part.stock_material, part.stock_thickness, part.stock_width,
                part.stock_height, part.machining_origin) != (
                stock.material, stock.thickness, stock.width, stock.height, (0.0, 0.0)):
            raise RuntimeError(f"Stock metadata changed on {part_name}")
        operations = project.get_mops_in_part(part)
        suffixes = [mop.user_identifier.rsplit("/", 1)[-1] for mop in operations]
        if suffixes != expected_ops[part_name]:
            raise RuntimeError(f"Unexpected operation order on {part_name}: {suffixes}")
        for mop, suffix in zip(operations, suffixes, strict=True):
            expected_tool = 2 if suffix == "back-slots" else 1
            if mop.tool_number != expected_tool:
                raise RuntimeError(f"Tool identity changed on {mop.user_identifier}")
            targets = project.get_mop_targets(mop)
            target_names = {project.get_primitive(uid).user_identifier for uid in targets}
            expected_count = 1 if suffix == "edge-profile" else 2
            if (len(target_names) != expected_count
                    or not all(name.startswith(component_name + "/") for name in target_names)):
                raise RuntimeError(f"MOP targets are not bound to {part_name}: {target_names}")
            if suffix == "edge-profile":
                if target_names != {component_name + "/outline"}:
                    raise RuntimeError(f"Profile does not target the board outline: {target_names}")
            else:
                group = "front-machining" if suffix == "front-grooves" else "back-machining"
                groups = {group for uid in targets
                          for group in project.get_groups_of_primitive(uid)}
                if not any(value.endswith("/" + group) for value in groups):
                    raise RuntimeError(f"The {group} group was not retained")

        outline = project.get_primitive(component_name + "/outline")
        actual_bounds = _bounds(outline)
        if any(not math.isclose(a, b, abs_tol=1e-9)
               for a, b in zip(actual_bounds, expected_bounds[part_name], strict=True)):
            raise RuntimeError(f"Unexpected placement for {part_name}: {actual_bounds}")
        if (actual_bounds[0] < stock.origin[0] or actual_bounds[1] < stock.origin[1]
                or actual_bounds[2] > stock.origin[0] + stock.width
                or actual_bounds[3] > stock.origin[1] + stock.height):
            raise RuntimeError(f"{part_name} falls outside declared stock: {actual_bounds}")
        boxes.append(actual_bounds)
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            overlap_x = min(first[2], second[2]) - max(first[0], second[0])
            overlap_y = min(first[3], second[3]) - max(first[1], second[1])
            if overlap_x > 0 and overlap_y > 0:
                raise RuntimeError(f"Placed boards overlap: {first}, {second}")


def _verify_sheet_pair(front: CBProject, back: CBProject,
                       components: list[str], stock_width: float) -> None:
    """Check every matching outline/feature point against whole-sheet reflection."""
    for component in components:
        front_shapes = {p.user_identifier: p for p in front.list_primitives()
                        if p.user_identifier.startswith(component + "/")}
        back_shapes = {p.user_identifier: p for p in back.list_primitives()
                       if p.user_identifier.startswith(component + "/")}
        if set(front_shapes) != set(back_shapes):
            raise RuntimeError(f"Front/back design entities differ for {component}")
        for name, front_shape in front_shapes.items():
            front_points = front_shape.get_absolute_coordinates()
            back_points = back_shapes[name].get_absolute_coordinates()
            for x, y in front_points:
                if not any(math.isclose(stock_width - x, back_x, abs_tol=1e-9)
                           and math.isclose(y, back_y, abs_tol=1e-9)
                           for back_x, back_y in back_points):
                    raise RuntimeError(f"Whole-sheet registration failed for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    setups = build_cabinet_setups()
    plywood_stock = SetupStock(2000, 800, 18, "Birch plywood")
    mdf_stock = SetupStock(1280, 780, 6, "White melamine MDF")
    layouts = {
        "cabinet-plywood-front": (
            ["left-side/front", "middle-shelf/front", "right-side/front"],
            {"left-side/front": ["front-grooves"],
             "middle-shelf/front": ["front-grooves"],
             "right-side/front": ["front-grooves"]},
            plywood_stock,
            {"left-side/front": (30, 40, 630, 740),
             "middle-shelf/front": (700, 40, 1260, 200),
             "right-side/front": (1370, 40, 1970, 740)}),
        "cabinet-plywood-back": (
            ["left-side/back", "middle-shelf/back", "right-side/back"],
            {"left-side/back": ["back-slots", "edge-profile"],
             "middle-shelf/back": ["back-slots", "edge-profile"],
             "right-side/back": ["back-slots", "edge-profile"]},
            plywood_stock,
            {"left-side/back": (1370, 40, 1970, 740),
             "middle-shelf/back": (740, 40, 1300, 200),
             "right-side/back": (30, 40, 630, 740)}),
        "cabinet-mdf-front": (
            ["back-panel/front"], {"back-panel/front": ["front-grooves"]},
            mdf_stock, {"back-panel/front": (40, 30, 1240, 730)}),
        "cabinet-mdf-back": (
            ["back-panel/back"],
            {"back-panel/back": ["back-slots", "edge-profile"]},
            mdf_stock, {"back-panel/back": (40, 30, 1240, 730)}),
    }
    reopened_setups = {}
    for name, project in setups.items():
        path = args.output_directory / f"{name}-original.cb"
        project.save(str(path))
        reopened = read_cambam_bytes(path.read_bytes())
        _verify_output(reopened, *layouts[name])
        reopened_setups[name] = reopened

    _verify_sheet_pair(reopened_setups["cabinet-plywood-front"],
                       reopened_setups["cabinet-plywood-back"],
                       ["left-side", "middle-shelf", "right-side"], 2000)
    _verify_sheet_pair(reopened_setups["cabinet-mdf-front"],
                       reopened_setups["cabinet-mdf-back"],
                       ["back-panel"], 1280)

    edited = read_cambam_bytes(
        (args.output_directory / "cabinet-plywood-front-original.cb").read_bytes())
    mop = edited.get_mop("left-side/front-grooves")
    if mop is None:
        raise RuntimeError("Could not find the left-side groove operation")
    mop.target_depth = -6
    edited_path = args.output_directory / "cabinet-plywood-front-edited.cb"
    edited.save(str(edited_path))
    reopened_edit = read_cambam_bytes(edited_path.read_bytes())
    if reopened_edit.get_mop("left-side/front-grooves").target_depth != -6:
        raise RuntimeError("The edited operation depth did not survive reopening")
    _verify_output(reopened_edit, *layouts["cabinet-plywood-front"])
    print("Verified four registered front/back stock setups, back profiles and an edited round trip")
    print(args.output_directory.resolve())


if __name__ == "__main__":
    main()
