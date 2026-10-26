"""Exact bitmap image dependencies, relocation and malformed-input rejection."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from cambam_builder import CBProject
from cambam_builder.integrations.cambam.bitmap_bundle import BitmapBundle, capture, verify_bundle
from cambam_builder.native.reader import read_cambam_bytes


IMAGE = b"synthetic image bytes: integrity, no image-decoding claim\r\n"


class BitmapBundleTests(unittest.TestCase):
    def setUp(self):
        Path("output").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="bitmap-tests-", dir="output")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        (self.inputs / "reference.png").write_bytes(IMAGE)
        self.project = CBProject("artwork")
        self.layer = self.project.add_layer("Artwork")
        self.bitmap = self.project.add_bitmap(self.layer, "reference.png", identifier="reference",
                                              position=(3, 5, 0), width=40, height=20)
        self.project.add_rect(self.layer, width=40, height=20, identifier="outline")

    def capture(self):
        return capture(self.project, source_directory=self.inputs)

    def write(self):
        self.capture().write_bundle(self.root / "portable")
        return self.root / "portable"

    def test_shared_dependency_relocates_with_identity_geometry_and_exact_bytes(self):
        second = self.project.add_bitmap(self.layer, "reference.png", identifier="second",
                                         position=(45, 5, 0), width=20, height=10,
                                         display_bitmap=False)
        snapshot = self.capture()
        destination = self.root / "portable"
        snapshot.check_current(self.project, source_directory=self.inputs)
        snapshot.write_bundle(destination)
        moved = self.root / "moved"
        shutil.copytree(destination, moved)
        (self.inputs / "reference.png").unlink()
        report = verify_bundle(moved)
        self.assertEqual((report["status"], report["source_count"], report["asset_count"]),
                         ("verified", 2, 1))
        reopened = read_cambam_bytes((moved / "job.cb").read_bytes(), strict=True)
        for original in (self.bitmap, second):
            bitmap = reopened.get_primitive(original.user_identifier)
            self.assertEqual(bitmap.internal_id, original.internal_id)
            self.assertEqual(bitmap.get_absolute_coordinates_xyz(), original.get_absolute_coordinates_xyz())
            self.assertEqual((moved / bitmap.source_file).read_bytes(), IMAGE)
        self.assertEqual(self.bitmap.source_file, "reference.png")
        with self.assertRaises(FileExistsError):
            snapshot.write_bundle(destination)

    def test_freshness_rejects_image_document_reference_and_missing_dependencies(self):
        snapshot = self.capture()
        self.bitmap.width += 1
        with self.assertRaisesRegex(ValueError, "changed"):
            snapshot.check_current(self.project, source_directory=self.inputs)
        self.bitmap.width -= 1
        (self.inputs / "reference.png").write_bytes(IMAGE + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            snapshot.check_current(self.project, source_directory=self.inputs)
        # A captured snapshot remains a reproducible exact-byte copy.
        snapshot.write_bundle(self.root / "captured")
        self.assertEqual(verify_bundle(self.root / "captured")["status"], "verified")
        (self.inputs / "reference.png").unlink()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            snapshot.check_current(self.project, source_directory=self.inputs)

    def test_path_boundaries_and_incomplete_raw_collections_reject(self):
        (self.root / "outside.png").write_bytes(IMAGE)
        for reference in ("../outside.png", "C:reference.png", "https://example.org/a.png"):
            self.bitmap.source_file = reference
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.capture()
        self.bitmap.source_file = str(self.inputs / "reference.png")
        self.capture()
        self.bitmap.source_file = "reference.png"
        snapshot = self.capture()
        root = ET.fromstring(snapshot.document_bytes)
        ET.SubElement(root.find("layers/layer"), "objects")
        bad = replace(snapshot, document_bytes=ET.tostring(root))
        with self.assertRaises(ValueError):
            bad.write_bundle(self.root / "bad")
        self.assertFalse((self.root / "bad").exists())
        with self.assertRaises(ValueError):
            replace(snapshot, sources=()).write_bundle(self.root / "empty")

    def test_written_integrity_rejects_document_manifest_and_dependency_tampering(self):
        changes = ("document", "image", "missing", "identity", "duplicate", "path", "hash")
        for change in changes:
            with self.subTest(change=change):
                destination = self.root / change
                self.capture().write_bundle(destination)
                manifest_path = destination / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                record = manifest["images"][0]
                if change == "document":
                    with (destination / "job.cb").open("ab") as stream:
                        stream.write(b" ")
                elif change == "image":
                    (destination / record["path"]).write_bytes(b"different")
                elif change == "missing":
                    (destination / record["path"]).unlink()
                elif change == "identity":
                    record["primitive_id"] = "unknown"
                elif change == "duplicate":
                    manifest["images"].append(record)
                elif change == "path":
                    record["path"] = "../reference.png"
                else:
                    record["sha256"] = "x" * 64
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    verify_bundle(destination)


if __name__ == "__main__":
    unittest.main()
