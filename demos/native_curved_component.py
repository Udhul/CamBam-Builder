"""Edit a native Line/Spline component through two save/reopen cycles.

Copy this script outside the checkout and run with a base-only installation:
    python native_curved_component.py /path/to/output-directory

Coordinates are illustrative millimeters. Native Engrave targets retain editable
CAD intent; this example does not generate toolpaths or G-code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cambam_builder import CBProject, Vertex
from cambam_builder.native.reader import read_cambam_bytes


def build_component() -> CBProject:
    """Author a straight base and a cubic curved edge with native operations."""
    project = CBProject("curved-component")
    layer = project.add_layer("Component")
    base = project.add_line(
        layer, [(0, 0, 0), (40, 0, 0)], identifier="base",
        groups=["component"],
    )
    edge = project.add_spline(
        layer, [(40, 0, 0), (40, 30, 2), (0, 30, 2), (0, 0, 0)],
        knots=[0, 0, 0, 0, 1, 1, 1, 1], degree=3,
        identifier="curved-edge", parent=base, groups=["component"],
    )
    if base is None or edge is None:
        raise RuntimeError("Component geometry could not be created")
    base.native_tag = "Straight registration edge"
    edge.native_tag = "Retain the editable cubic curve"
    part = project.add_part("Detail", stock_width=60, stock_height=60,
                            stock_thickness=6)
    project.add_engrave_mop(part, [edge], identifier="curve-pass",
                            target_depth=-0.5, tool_diameter=2)
    project.add_engrave_mop(part, [base], identifier="base-pass",
                            target_depth=-0.25, tool_diameter=2)
    if not project.rotate_primitive_deg(base, 90, cx=0, cy=0):
        raise RuntimeError("Component rotation failed")
    if not project.translate_primitive(base, 10, 20):
        raise RuntimeError("Component placement failed")
    if not project.translate_primitive_z(base, 2):
        raise RuntimeError("Component elevation failed")
    return project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    original_path = args.output_directory / "component-original.cb"
    edited_path = args.output_directory / "component-edited.cb"

    project = build_component()
    original_ids = {p.user_identifier: p.internal_id for p in project.list_primitives()}
    project.save(str(original_path))
    edited = read_cambam_bytes(original_path.read_bytes())
    edited.get_primitive("base").vertices[1] = Vertex(42, 0, 0)
    edge = edited.get_primitive("curved-edge")
    edge.control_points[0] = Vertex(42, 0, 0)
    edge.control_points[2] = Vertex(0, 36, 2)
    edge.native_tag += "; revision B"
    edited.get_mop("curve-pass").target_depth = -0.75
    if not edited.translate_primitive("base", 5, -5):
        raise RuntimeError("Edited placement failed")
    edited.save(str(edited_path))
    reopened = read_cambam_bytes(edited_path.read_bytes())
    edge = reopened.get_primitive("curved-edge")
    actual = edge.get_absolute_coordinates_xyz()["control_points"]
    expected = [(15, 57, 2), (-15, 55, 4), (-21, 15, 4), (15, 15, 2)]
    checks = {
        "units": reopened.units == "Millimeters",
        "identity": original_ids == {
            p.user_identifier: p.internal_id for p in reopened.list_primitives()},
        "world control points": all(abs(a - b) < 1e-8
                                    for p, q in zip(actual, expected, strict=True)
                                    for a, b in zip(p, q, strict=True)),
        "spline definition": edge.degree == 3 and edge.weights == [1, 1, 1, 1]
                             and edge.knots == [0, 0, 0, 0, 1, 1, 1, 1],
        "hierarchy": reopened.get_parent_of_primitive(edge)
                     is reopened.get_primitive("base"),
        "operation order": [m.user_identifier for m in reopened.get_mops_in_part("Detail")]
                           == ["curve-pass", "base-pass"],
        "curve target": reopened.get_mop_targets("curve-pass") == [edge.internal_id],
        "edited depth": reopened.get_mop("curve-pass").target_depth == -0.75,
        "operator note": edge.native_tag.endswith("revision B"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"Component round trip failed: {', '.join(failures)}")
    print(f"Verified {len(checks)} native curved component checks")
    print(original_path.resolve())
    print(edited_path.resolve())


if __name__ == "__main__":
    main()
