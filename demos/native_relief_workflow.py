"""Author editable native relief; audit only posts actually generated in CamBam.

Run from the repository root with its declared interpreter. The exact design is
the sampled bilinear field, not the analytic sampling recipe. No toolpaths or
G-code are synthesized by this consumer.
"""

import argparse
import json
import math
from pathlib import Path
import uuid

from cambam_builder.cam_core.relief import HeightField
from cambam_builder.integrations.cambam.native_relief import (
    audit_post, create_document,
)
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


def design() -> HeightField:
    """A small asymmetric ridge with nonzero bilinear mixed curvature."""
    xs = (0.0, 0.5, 1.0, 1.5, 2.0)
    ys = (0.0, 0.5, 1.0, 1.5)
    heights = tuple(tuple(-0.5 + 0.12 * math.cos(math.pi * x)
                          + 0.04 * math.sin(math.pi * y) * (x - 1)
                          for x in xs) for y in ys)
    return HeightField(xs, ys, heights, 1.2)


def run(directory: Path, *, post: Path | None = None) -> dict:
    """Write source, library reopen and edited export; optionally audit a post."""
    directory.mkdir(parents=True, exist_ok=False)
    document = create_document(design())
    source = directory / "native-relief.cb"
    source.write_bytes(document.xml_bytes)
    reopened = read_cambam_bytes(document.xml_bytes)
    roundtrip = serialize_cambam_bytes(reopened)
    document.check_document(roundtrip)
    (directory / "native-relief-reopened.cb").write_bytes(roundtrip)
    # Demonstrate an ordinary native process edit and its second round trip.
    mop = reopened.get_mop("relief-finish")
    mop.cut_feedrate = 180.0
    edited = serialize_cambam_bytes(reopened)
    if read_cambam_bytes(edited).get_mop("relief-finish").cut_feedrate != 180.0:
        raise RuntimeError("native process edit did not survive export")
    (directory / "native-relief-edited.cb").write_bytes(edited)
    summary = {"source_fingerprint": document.target.fingerprint,
        "mesh_vertices": len(document.mesh.vertices), "mesh_faces": len(document.mesh.faces),
        "mesh_error_upper_mm": document.mesh.maximum_error_mm,
        "native_resolution_mm": document.settings.resolution * 2 * document.settings.radius_mm,
        "native_allowance_mm": document.settings.allowance_mm,
        "document": document.check_document(), "edit_roundtrip": "pass",
        "actual_post": "not_evaluated",
        "scope": "native document and independent geometry; no physical cutting acceptance"}
    if post is not None:
        data = post.read_bytes()
        (directory / "actual-native-post.nc").write_bytes(data)
        summary["actual_post"] = audit_post(document, data,
            max_residual_volume_mm3=0.5, max_residual_thickness_mm=0.25)
    (directory / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path("output") / f"nr03-relief-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--post", type=Path, help="complete actual Default post to audit")
    args = parser.parse_args()
    print(json.dumps(run(args.directory, post=args.post), indent=2))
