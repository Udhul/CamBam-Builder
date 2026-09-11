"""Verify the two artifacts produced by the MCP 4e named-client workflow.

This checks the portable files independently of the MCP process and prints the
exact facts needed before manual CamBam inspection.  It does not claim client,
units, rendering, or generated-toolpath acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from cambam_builder.cambam_entities import ProfileMop, Rect
from cambam_builder.cambam_reader import read_cambam_bytes


EXPECTED_A = ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0),
              (20.0, 10.0, 0.0), (0.0, 10.0, 0.0))
EXPECTED_B = ((5.0, 2.0, 0.0), (25.0, 2.0, 0.0),
              (25.0, 12.0, 0.0), (5.0, 12.0, 0.0))


def _cyclic_equal(actual: Iterable[Iterable[float]], expected, tolerance=1e-9) -> bool:
    got = tuple(tuple(float(value) for value in point) for point in actual)
    if len(got) != len(expected):
        return False
    for offset in range(len(expected)):
        rotated = expected[offset:] + expected[:offset]
        if all(
            all(abs(left - right) <= tolerance for left, right in zip(got_point, want_point))
            for got_point, want_point in zip(got, rotated)
        ):
            return True
    return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path):
    data = path.read_bytes()
    return data, read_cambam_bytes(data, source_name=path.name, strict=True)


def verify_pair(a_path: Path, b_path: Path) -> dict:
    """Validate the contract's Rect/Profile A-to-B acceptance pair."""
    a_path = a_path.resolve()
    b_path = b_path.resolve()
    a_bytes, project_a = _load(a_path)
    b_bytes, project_b = _load(b_path)
    _require(a_bytes != b_bytes, "A.cb and B.cb must differ after translation")

    def records(project, label, expected_corners):
        layers = project.list_layers()
        parts = project.list_parts()
        primitives = project.list_primitives()
        mops = project.list_mops()
        _require(project.project_name == "slice", f"{label}: project name must be 'slice'")
        _require(len(layers) == len(parts) == len(primitives) == len(mops) == 1,
                 f"{label}: expected one layer, part, primitive, and MOP")
        rect, profile = primitives[0], mops[0]
        _require(layers[0].user_identifier == "Geometry", f"{label}: layer must be Geometry")
        _require(parts[0].user_identifier == "Part", f"{label}: part must be Part")
        _require(isinstance(rect, Rect) and rect.user_identifier == "outline",
                 f"{label}: primitive must be Rect 'outline'")
        _require(isinstance(profile, ProfileMop) and profile.user_identifier == "profile"
                 and profile.name == "profile",
                 f"{label}: MOP must be Profile 'profile'")
        _require((parts[0].stock_width, parts[0].stock_height, parts[0].stock_thickness,
                  parts[0].stock_material) == (0.0, 0.0, 0.0, ""),
                 f"{label}: Part must retain the MCP slice's unspecified zero stock")
        _require(_cyclic_equal(rect.get_absolute_coordinates_xyz(), expected_corners),
                 f"{label}: rectangle world corners do not match the acceptance contract")
        _require(project.get_layer_of_primitive(rect) is layers[0],
                 f"{label}: outline is not assigned to Geometry")
        _require(project.get_part_of_mop(profile) is parts[0],
                 f"{label}: profile is not assigned to Part")
        _require(project.get_mop_targets(profile) == [rect.internal_id],
                 f"{label}: profile does not target outline")
        expected_parameters = {
            "enabled": True,
            "profile_side": "Outside", "target_depth": -1.0, "depth_increment": 0.5,
            "tool_diameter": 3.0, "cut_feedrate": 300.0, "plunge_feedrate": 100.0,
            "spindle_speed": 12000, "stock_surface": 0.0, "clearance_plane": 5.0,
        }
        for name, expected in expected_parameters.items():
            _require(getattr(profile, name) == expected,
                     f"{label}: profile {name} must be {expected!r}")
        return rect, profile

    rect_a, profile_a = records(project_a, "A", EXPECTED_A)
    rect_b, profile_b = records(project_b, "B", EXPECTED_B)
    _require(rect_a.internal_id == rect_b.internal_id,
             "rectangle identity changed between A.cb and B.cb")
    _require(profile_a.internal_id == profile_b.internal_id,
             "Profile identity changed between A.cb and B.cb")

    return {
        "status": "artifact contract passed; CamBam acceptance is still manual",
        "A": {"path": str(a_path), "bytes": len(a_bytes),
              "sha256": hashlib.sha256(a_bytes).hexdigest(), "world_xyz": EXPECTED_A},
        "B": {"path": str(b_path), "bytes": len(b_bytes),
              "sha256": hashlib.sha256(b_bytes).hexdigest(), "world_xyz": EXPECTED_B},
        "rectangle_id": str(rect_a.internal_id),
        "profile_id": str(profile_a.internal_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="unmodified A.cb from the client workflow")
    parser.add_argument("b", type=Path, help="translated B.cb from the client workflow")
    args = parser.parse_args()
    try:
        result = verify_pair(args.a, args.b)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"acceptance artifact verification failed: {exc}") from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
