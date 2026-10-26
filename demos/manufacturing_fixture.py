"""Compose a small front/back sensor fixture from reusable native parts.

Run from any directory with the base library installed:
    python manufacturing_fixture.py /path/to/output-directory

The example authors the fixture recipe in the caller. Manual stock positions
are explicit; CamBam remains responsible for final toolpaths and machine output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.native.manufacturing import (
    ComponentPlacement,
    SetupFrame,
    SetupStock,
    compose_setup,
)
from cambam_builder.native.reader import read_cambam_file


def _build_fixture_design() -> CBProject:
    """Author one 160 x 100 x 12 mm aluminum sensor fixture blank."""
    design = CBProject("sensor-fixture-design")
    design.units = "Millimeters"
    design.default_tool_diameter = 6.0
    machining = design.add_layer("Fixture geometry")
    blank = design.add_rect(
        machining, width=160, height=100, identifier="fixture-blank",
        groups=["fixture", "outline"],
    )
    recess = design.add_rect(
        machining, corner=(18, 16), width=48, height=31,
        identifier="sensor-recess", parent=blank,
        groups=["fixture", "front", "recess"],
    )
    # Deliberately asymmetric so a reflected setup is observable in XY.
    holes = design.add_points(
        machining, [(28, 24), (130, 31), (39, 78)], identifier="mounting-holes",
        parent=blank, groups=["fixture", "front", "drilling"],
    )
    relief = design.add_rect(
        machining, corner=(94, 57), width=37, height=11,
        identifier="cable-relief", parent=blank,
        groups=["fixture", "back", "relief"],
    )

    # The two Parts are setup choices over the same design. Their depths are
    # authored relative to the stock top for the selected setup.
    front = design.add_part(
        "fixture-front", stock_width=160, stock_height=100,
        stock_thickness=12, stock_material="Aluminum 6061",
    )
    back = design.add_part(
        "fixture-back", stock_width=160, stock_height=100,
        stock_thickness=12, stock_material="Aluminum 6061",
    )
    design.add_pocket_mop(
        front, [recess], identifier="front-recess", name="Sensor recess",
        target_depth=-3.5, depth_increment=1.5, tool_diameter=6, tool_number=1,
    )
    design.add_drill_mop(
        front, [holes], identifier="front-holes", name="Mounting holes",
        target_depth=-12, depth_increment=3, tool_diameter=5, tool_number=2,
        drilling_method="CannedCycle", retract_height=2,
    )
    design.add_pocket_mop(
        back, [relief], identifier="back-relief", name="Cable relief",
        target_depth=-2, depth_increment=1, tool_diameter=4, tool_number=3,
    )
    design.add_profile_mop(
        back, [blank], identifier="back-profile", name="Back profile",
        profile_side="Outside", target_depth=-12, depth_increment=3,
        tool_diameter=6, tool_number=1,
    )
    return design


def build_fixture_setups() -> dict[str, CBProject]:
    """Return manually placed front and back jobs for the sensor fixture."""
    design = _build_fixture_design()
    stock = SetupStock(
        width=400, height=250, thickness=12, material="Aluminum 6061",
    )
    front = compose_setup(
        "sensor-fixture-front", stock,
        [
            ComponentPlacement(
                design, "fixture-blank", "fixture-left", ("fixture-front",),
                SetupFrame(x=25, y=35),
            ),
            ComponentPlacement(
                design, "fixture-blank", "fixture-rotated", ("fixture-front",),
                SetupFrame(x=365, y=35, rotation_degrees=90),
            ),
        ],
        units="Millimeters",
    )
    back = compose_setup(
        "sensor-fixture-back", stock,
        [
            ComponentPlacement(
                design, "fixture-blank", "fixture-left", ("fixture-back",),
                # Mirror across the whole 400 mm sheet to register the opposite
                # face of the first front component.
                SetupFrame(x=375, y=35, reflect_x=True),
            ),
            ComponentPlacement(
                design, "fixture-blank", "fixture-rotated", ("fixture-back",),
                SetupFrame(x=35, y=35, rotation_degrees=-90, reflect_x=True),
            ),
        ],
        units="Millimeters",
    )
    return {"front": front, "back": back}


def _check_roundtrip(project: CBProject, path: Path, side: str) -> None:
    project.save(str(path))
    reopened = read_cambam_file(str(path))
    if reopened is None:
        raise RuntimeError(f"Could not reopen {side} setup")

    expected = (
        ["fixture-left/front-recess", "fixture-left/front-holes",
         "fixture-rotated/front-recess", "fixture-rotated/front-holes"]
        if side == "front" else
        ["fixture-left/back-relief", "fixture-left/back-profile",
         "fixture-rotated/back-relief", "fixture-rotated/back-profile"]
    )
    checks = {
        "stock material": all(p.stock_material == "Aluminum 6061" for p in reopened.list_parts()),
        "stock thickness": all(p.stock_thickness == 12 for p in reopened.list_parts()),
        "stock width": all(p.stock_width == 400 for p in reopened.list_parts()),
        "operation count/order": [m.user_identifier for m in reopened.list_mops()] == expected,
        "tool identities": [m.tool_number for m in reopened.list_mops()] == (
            [1, 2, 1, 2] if side == "front" else [3, 1, 3, 1]),
    }
    if side == "front":
        hole_targets = reopened.get_mop_targets("fixture-left/front-holes")
        checks["drill target"] = hole_targets == [
            reopened.get_primitive("fixture-left/mounting-holes").internal_id,
        ]
        points = reopened.get_primitive("fixture-left/mounting-holes").get_absolute_coordinates()
        checks["front placement"] = points == [(53, 59), (155, 66), (64, 113)]
        rotated = reopened.get_primitive("fixture-rotated/mounting-holes").get_absolute_coordinates()
        checks["rotated placement"] = rotated == [(341, 63), (334, 165), (287, 74)]
    else:
        back_target = reopened.get_mop_targets("fixture-left/back-relief")
        checks["relief target"] = back_target == [
            reopened.get_primitive("fixture-left/cable-relief").internal_id,
        ]
        relief = reopened.get_primitive("fixture-left/cable-relief").get_absolute_coordinates()
        checks["reflected relief"] = relief == [(281, 92), (244, 92), (244, 103), (281, 103)]
        checks["back depth"] = reopened.get_mop("fixture-left/back-relief").target_depth == -2
        rotated = reopened.get_primitive("fixture-rotated/mounting-holes").get_absolute_coordinates()
        checks["registered back placement"] = rotated == [(59, 63), (66, 165), (113, 74)]

    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"{side} setup round trip failed: {', '.join(failures)}")

    # Exercise a normal edit and a second interchange cycle. The edited depth
    # remains a caller choice for the stock-top datum of this setup.
    edited_mop = reopened.get_mop(expected[0])
    edited_mop.target_depth -= 0.5
    edited_path = path.with_name(path.stem + "-edited" + path.suffix)
    reopened.save(str(edited_path))
    final = read_cambam_file(str(edited_path))
    if final is None or final.get_mop(expected[0]).target_depth != edited_mop.target_depth:
        raise RuntimeError(f"Could not preserve edited {side} operation")
    print(f"Verified {side}: {len(checks)} setup checks and edited round trip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    setups = build_fixture_setups()
    _verify_registered_layout(setups)
    for side, project in setups.items():
        _check_roundtrip(project, args.output_directory / f"sensor-fixture-{side}.cb", side)


def _verify_registered_layout(setups: dict[str, CBProject]) -> None:
    """Check manually placed board envelopes and corresponding face datums."""
    front, back = setups["front"], setups["back"]
    left_front = front.get_primitive("fixture-left/fixture-blank").get_absolute_coordinates()
    turned_front = front.get_primitive("fixture-rotated/fixture-blank").get_absolute_coordinates()
    left_back = back.get_primitive("fixture-left/fixture-blank").get_absolute_coordinates()
    turned_back = back.get_primitive("fixture-rotated/fixture-blank").get_absolute_coordinates()
    # Inclusive bounds touch at edges only; neither pair overlaps on its stock.
    for first, second in ((left_front, turned_front), (left_back, turned_back)):
        first_min_x, first_max_x = min(p[0] for p in first), max(p[0] for p in first)
        second_min_x, second_max_x = min(p[0] for p in second), max(p[0] for p in second)
        if first_max_x > second_min_x and second_max_x > first_min_x:
            raise RuntimeError("Manual component placements overlap")

    for placement in ("fixture-left", "fixture-rotated"):
        front_points = front.get_primitive(f"{placement}/mounting-holes").get_absolute_coordinates()
        back_points = back.get_primitive(f"{placement}/mounting-holes").get_absolute_coordinates()
        expected = [(400 - x, y) for x, y in front_points]
        if back_points != expected:
            raise RuntimeError(f"{placement} front/back setup datums do not register")


if __name__ == "__main__":
    main()
