"""Synthetic artwork registration and exact-byte, relocatable image dependencies.

Run with an installed base library. The PNG is reference context; the editable
rectangle and registration marks remain independent CAD geometry.
"""

import argparse
import binascii
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import zlib

from cambam_builder import CBProject
from cambam_builder.integrations.cambam.bitmap_bundle import capture, verify_bundle
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


def synthetic_png() -> bytes:
    """Eight by four pixels: red left half, blue right half, no external assets."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xffffffff)
    scanline = b"\x00" + b"\xff\x00\x00" * 4 + b"\x00\x00\xff" * 4
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 4, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(scanline * 4)) + chunk(b"IEND", b""))


def run(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / "inputs"
    inputs.mkdir()
    image = synthetic_png()
    image_path = inputs / "artwork.png"
    image_path.write_bytes(image)
    project = CBProject("bitmap-registration")
    artwork = project.add_layer("Artwork")
    geometry = project.add_layer("Editable geometry")
    bitmap = project.add_bitmap(artwork, "artwork.png", identifier="artwork-reference",
                                position=(3, 5, 0), width=40, height=20)
    bitmap.native_tag = "Synthetic image, red left / blue right. Reference only."
    project.add_rect(geometry, identifier="registration-frame", corner=(3, 5), width=40, height=20)
    for name, x in (("left-mark", 3), ("middle-mark", 23), ("right-mark", 43)):
        project.add_line(geometry, [(x, 3), (x, 7)], identifier=name)
    source = serialize_cambam_bytes(project)
    (inputs / "source.cb").write_bytes(source)
    snapshot = capture(project, source_directory=inputs)
    snapshot.check_current(project, source_directory=inputs)
    snapshot.write_bundle(destination / "portable")
    shutil.copytree(destination / "portable", destination / "relocated")
    portable_report = verify_bundle(destination / "relocated")
    reopened = read_cambam_bytes((destination / "relocated/job.cb").read_bytes(), strict=True)
    edited = reopened.get_primitive("artwork-reference")
    same_id = edited.internal_id == bitmap.internal_id
    same_geometry = edited.get_absolute_coordinates_xyz() == bitmap.get_absolute_coordinates_xyz()
    edited.position = (10, 11, 2)
    edited.width, edited.height = 24, 12
    edited_data = serialize_cambam_bytes(reopened)
    (destination / "relocated/library-edited.cb").write_bytes(edited_data)
    edited_reopen = read_cambam_bytes(edited_data, strict=True).get_primitive("artwork-reference")
    changed_rejected = False
    bitmap.width += 1
    try:
        snapshot.check_current(project, source_directory=inputs)
    except ValueError:
        changed_rejected = True
    bitmap.width -= 1
    image_path.write_bytes(image + b"changed")
    changed_image_rejected = False
    try:
        snapshot.check_current(project, source_directory=inputs)
    except ValueError:
        changed_image_rejected = True
    image_path.unlink()
    missing_rejected = False
    try:
        snapshot.check_current(project, source_directory=inputs)
    except ValueError:
        missing_rejected = True
    # Restore the pending native-validation source exactly after the negative checks.
    image_path.write_bytes(image)
    checks = dict(identity=same_id, placement=same_geometry,
                  editable_geometry=reopened.get_primitive("registration-frame") is not None,
                  edit_roundtrip=edited_reopen.position == (10, 11, 2) and edited_reopen.width == 24,
                  document_change_rejected=changed_rejected,
                  image_change_rejected=changed_image_rejected,
                  missing_image_rejected=missing_rejected,
                  exact_image=(destination / "relocated" / edited.source_file).read_bytes() == image)
    if not all(checks.values()):
        raise AssertionError(checks)
    report = dict(checks=checks, portable=portable_report,
                  image_sha256=hashlib.sha256(image).hexdigest(),
                  source_document_sha256=hashlib.sha256(source).hexdigest(),
                  bitmap_uuid=str(bitmap.internal_id),
                  source_corners=bitmap.get_absolute_coordinates_xyz(),
                  edited_corners=edited_reopen.get_absolute_coordinates_xyz(),
                  limits="Bitmap dependencies only; no image tracing, native display or machining claim.")
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    args = parser.parse_args()
    if args.directory is None:
        Path("output").mkdir(exist_ok=True)
        # mkdtemp reserves a unique name; run owns the new consumer subdirectory.
        args.directory = Path(tempfile.mkdtemp(prefix="bitmap-consumer-", dir="output")) / "artifacts"
    result = run(args.directory)
    print(json.dumps(result, indent=2))
