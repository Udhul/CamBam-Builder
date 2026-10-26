"""Build an editable mixed native/external-program job with portable dependencies.

Run with the base installed library, from any working directory:
    python native_ncfile_workflow.py /path/to/new-output-directory

The NC fragment is a synthetic interchange example. This script verifies file
integrity and edit/reopen behavior; it does not certify machine execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.integrations.cambam import ncfile
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


PROGRAM = (
    "(NC02 supplied accent begin)\r\n"
    "G21 G90 G17\r\n"
    "G0 Z5.000\r\n"
    "G0 X0.000 Y2.000\r\n"
    "G1 Z-0.200 F100.0\r\n"
    "G1 X5.000 Y2.000 F250.0\r\n"
    "G0 Z5.000\r\n"
    "(NC02 supplied accent end)\r\n"
).encode("ascii")


def build_project() -> CBProject:
    """Place a supplied accent between two native engraving operations."""
    project = CBProject("mixed-native-ncfile", units="Millimeters")
    layer = project.add_layer("Design")
    part = project.add_part("Panel", stock_width=8, stock_height=6,
                            stock_thickness=1, stock_offset=(-1, -1))
    for name, y in (("before", 0), ("after", 4)):
        line = project.add_pline(layer, [(0, y), (5, y)], identifier=f"line-{name}")
        project.add_engrave_mop(
            part, [line], identifier=name, name=f"Native {name}",
            target_depth=-0.2, depth_increment=0.2, stock_surface=0,
            clearance_plane=5, tool_number=1, tool_diameter=1,
            cut_feedrate=250, plunge_feedrate=100, spindle_speed=10000,
        )
    external = project.add_ncfile_mop(
        part, "accent.nc", identifier="external-accent", name="Supplied accent",
        target_mop="after", place_last=False,
    )
    if external is None:
        raise RuntimeError("Could not create external-program operation")
    external.native_tag = "Caller-owned program; regenerate evidence after editing."
    return project


def run(directory: Path) -> dict:
    """Write original, portable and edited bundles into a fresh directory."""
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / "inputs"
    inputs.mkdir()
    (inputs / "accent.nc").write_bytes(PROGRAM)
    original = build_project()
    original_bytes = serialize_cambam_bytes(original)
    (inputs / "source.cb").write_bytes(original_bytes)
    snapshot = ncfile.capture(original, source_directory=inputs)
    snapshot.check_current(original, source_directory=inputs)
    job = snapshot.write_bundle(directory / "portable")
    original_report = ncfile.verify_bundle(job.parent)

    edited = read_cambam_bytes(job.read_bytes(), strict=True)
    external = edited.get_mop("external-accent")
    external.name = "Accent revision B"
    external.native_tag += "\nRevision B: native lines extended."
    edited.get_primitive("line-after").vertices[1].x = 6
    stale_rejected = False
    try:
        snapshot.check_current(edited, source_directory=job.parent)
    except ValueError:
        stale_rejected = True
    if not stale_rejected:
        raise RuntimeError("Edited document retained stale dependency binding")

    rebound = ncfile.capture(edited, source_directory=job.parent)
    edited_job = rebound.write_bundle(directory / "edited")
    edited_report = ncfile.verify_bundle(edited_job.parent)
    reopened = read_cambam_bytes(edited_job.read_bytes(), strict=True)
    second = read_cambam_bytes(serialize_cambam_bytes(reopened), strict=True)
    checks = {
        "order": [m.user_identifier for m in second.get_mops_in_part("Panel")]
                 == ["before", "external-accent", "after"],
        "identity": second.get_mop("external-accent").internal_id
                    == original.get_mop("external-accent").internal_id,
        "targetless": second.get_mop_targets("external-accent") == [],
        "native target": second.get_mop_targets("after")
                         == [second.get_primitive("line-after").internal_id],
        "geometry edit": second.get_primitive("line-after").vertices[1].x == 6,
        "name edit": second.get_mop("external-accent").name == "Accent revision B",
        "notes": second.get_mop("external-accent").native_tag.endswith("lines extended."),
        "source unchanged": (edited_job.parent / second.get_mop("external-accent").source_file)
                            .read_bytes() == PROGRAM,
        "caller unchanged": serialize_cambam_bytes(original) == original_bytes,
        "stale rejected": stale_rejected,
    }
    if not all(checks.values()):
        raise RuntimeError(f"NCFile consumer checks failed: {checks}")
    report = {
        "checks": checks, "original": original_report, "edited": edited_report,
        "program_sha256": hashlib.sha256(PROGRAM).hexdigest(),
        "acceptance": "document/dependency integrity; no posted-motion safety claim",
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    report = run(args.output_directory)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
