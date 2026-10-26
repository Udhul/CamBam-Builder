"""Author, edit, and reopen a native cabinet board using only the base library.

Run this copied script from any directory with an installed cambam-builder:
    python native_board.py /path/to/output-directory

Dimensions are illustrative millimeters. CamBam remains responsible for final
toolpath generation; this example does not produce machine-ready G-code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cambam_builder import CBProject, Vertex
from cambam_builder.native.reader import read_cambam_file


def build_board() -> CBProject:
    """Create a 600 x 300 x 18 board with grooves, holes, and two placements."""
    project = CBProject("cabinet-board")
    project.units = "Millimeters"
    geometry = project.add_layer("Board geometry")
    layout_layer = project.add_layer("Layout", color="Blue")
    outline = project.add_rect(geometry, width=600, height=300,
                               identifier="outline", groups=["board"])
    outline.native_tag = "Cabinet side A\nKeep the marked face up."
    grooves = [project.add_rect(
        geometry, corner=(x, 10), width=6, height=280,
        identifier=f"groove-{index}", parent=outline, groups=["board", "grooves"],
    ) for index, x in enumerate((80, 500), start=1)]
    holes = project.add_points(
        geometry, [(32, 32), (568, 32), (568, 268), (32, 268)],
        identifier="holes", parent=outline, groups=["board"],
    )
    placements = project.add_points(layout_layer, [(0, 0), (640, 0)],
                                    identifier="placements")
    part = project.add_part(
        "Board", stock_width=1250, stock_height=350, stock_thickness=18,
        stock_material="Plywood", nesting_method="PointList",
        nesting_point_list=placements,
    )
    project.add_pocket_mop(part, grooves, identifier="grooves", name="Grooves",
                           target_depth=-6, depth_increment=3, tool_diameter=6)
    project.add_drill_mop(part, [holes], identifier="drilling", name="Holes",
                          target_depth=-12, depth_increment=4, tool_diameter=5)
    cutout = project.add_profile_mop(
        part, [outline], identifier="cutout", name="Cutout", profile_side="Outside",
        target_depth=-18, depth_increment=3, tool_diameter=6,
    )
    cutout.native_tag = "Cut last; inspect workholding before generating paths."
    project.translate_primitive(outline, 20, 25)
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    original_path = args.output_directory / "board-original.cb"
    edited_path = args.output_directory / "board-edited.cb"
    original = build_board()
    original.save(str(original_path))
    edited = read_cambam_file(str(original_path))
    if edited is None:
        raise RuntimeError("Could not reopen the authored board")
    edited.get_primitive("outline").width = 620
    edited.get_primitive("placements").vertices = [Vertex(0, 0), Vertex(660, 0)]
    edited.get_mop("grooves").target_depth = -7
    edited.get_primitive("outline").native_tag += "\nRevision B: width 620 mm."
    edited.save(str(edited_path))
    reopened = read_cambam_file(str(edited_path))
    if reopened is None:
        raise RuntimeError("Could not reopen the edited board")
    outline = reopened.get_primitive("outline")
    checks = {
        "units": reopened.units == "Millimeters",
        "width": outline.width == 620,
        "world pose": outline.get_absolute_coordinates()[0] == (20, 25),
        "hierarchy": reopened.get_parent_of_primitive("groove-1") is outline,
        "layout identity": reopened.get_part_nesting_points("Board") is
                           reopened.get_primitive("placements"),
        "layout edit": reopened.get_primitive("placements").vertices[1].x == 660,
        "groove depth": reopened.get_mop("grooves").target_depth == -7,
        "operation order": [m.user_identifier for m in reopened.get_mops_in_part("Board")]
                           == ["grooves", "drilling", "cutout"],
        "cutout target": reopened.get_mop_targets("cutout") == [outline.internal_id],
        "notes": outline.native_tag.endswith("Revision B: width 620 mm."),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"Native board round trip failed: {', '.join(failures)}")
    print(f"Verified {len(checks)} native board checks")
    print(original_path.resolve())
    print(edited_path.resolve())


if __name__ == "__main__":
    main()
