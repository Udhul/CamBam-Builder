"""Explicit NCFile dependency handling, portable copies and integrity failures."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from cambam_builder import CBProject
from cambam_builder.integrations.cambam.ncfile import (
    NCFileBundle, capture, verify_bundle,
)
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


PROGRAM = b"(fixture program)\r\nG90 G21\r\nG0 X1.250 Y2.500 Z5\r\nM7\r\nG1 X3.750 F120\r\n"


class NCFileBundleTests(unittest.TestCase):
    def setUp(self):
        Path("output").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="nc02-tests-", dir="output")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        (self.inputs / "supplied.nc").write_bytes(PROGRAM)
        self.project = CBProject("mixed-job")
        layer = self.project.add_layer("Geometry")
        outline = self.project.add_rect(layer, width=8, height=6, identifier="outline")
        part = self.project.add_part("Part", stock_width=8, stock_height=6, stock_thickness=2)
        self.native = self.project.add_profile_mop(
            part, [outline], identifier="cut", tool_diameter=1, target_depth=-1)
        self.external = self.project.add_ncfile_mop(
            part, "supplied.nc", identifier="external", name="Supplied operation")
        self.external.native_tag = "Caller-owned source"

    def snapshot(self):
        return capture(self.project, source_directory=self.inputs)

    def write(self):
        snapshot = self.snapshot()
        destination = self.root / "bundle"
        self.assertEqual(snapshot.write_bundle(destination), destination / "job.cb")
        return destination

    def change_manifest(self, destination, change):
        path = destination / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        change(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_complete_mixed_document_relocates_and_source_bytes_are_exact(self):
        # Two operation identities sharing one source must remain distinct,
        # including a disabled operation that the caller could enable later.
        spare = self.project.add_ncfile_mop(
            "Part", "supplied.nc", identifier="spare", enabled=False)
        before = serialize_cambam_bytes(self.project)
        snapshot = self.snapshot()
        self.assertEqual(snapshot.document_bytes, before)
        self.assertEqual(len(snapshot.sources), 2)
        self.assertEqual(snapshot.sources[0].data, PROGRAM)
        with self.assertRaises(FrozenInstanceError):
            snapshot.sources = ()
        with self.assertRaises(FrozenInstanceError):
            snapshot.sources[0].data = b"changed"
        destination = self.root / "portable"
        snapshot.check_current(self.project, source_directory=self.inputs)
        snapshot.write_bundle(destination)
        relocated = self.root / "another-directory"
        shutil.copytree(destination, relocated)
        # A consumer must need only the relocated directory, not input paths.
        (self.inputs / "supplied.nc").unlink()
        result = verify_bundle(relocated)
        self.assertEqual((result["status"], result["source_count"], result["asset_count"]),
                         ("verified", 2, 1))
        reopened = read_cambam_bytes((relocated / "job.cb").read_bytes())
        self.assertEqual([m.user_identifier for m in reopened.get_mops_in_part("Part")],
                         ["cut", "external", "spare"])
        for identifier in ("external", "spare"):
            mop = reopened.get_mop(identifier)
            self.assertEqual((relocated / mop.source_file).read_bytes(), PROGRAM)
            self.assertEqual(reopened.get_mop_targets(mop), [])
        self.assertEqual(reopened.get_mop("external").internal_id, self.external.internal_id)
        self.assertEqual(reopened.get_mop("external").native_tag, "Caller-owned source")
        self.assertEqual(reopened.get_mop("spare").internal_id, spare.internal_id)
        self.assertFalse(reopened.get_mop("spare").enabled)
        self.assertEqual(serialize_cambam_bytes(self.project), before)
        self.assertEqual(self.external.source_file, "supplied.nc")
        (relocated / "native-saved.cb").write_bytes(b"unrelated native result")
        self.assertEqual(verify_bundle(relocated)["status"], "verified")

    def test_capture_accepts_relative_windows_separators_and_contained_absolute_path(self):
        subdir = self.inputs / "nested"
        subdir.mkdir()
        (subdir / "piece.nc").write_bytes(PROGRAM)
        for reference in ("nested\\piece.nc", str(subdir / "piece.nc")):
            with self.subTest(reference=reference):
                self.external.source_file = reference
                self.assertEqual(self.snapshot().sources[0].data, PROGRAM)

    def test_capture_rejects_missing_directory_and_missing_nonfile_or_escaping_sources(self):
        outside = self.root / "outside.nc"
        outside.write_bytes(PROGRAM)
        for reference in ("missing.nc", "nested", "../outside.nc", str(outside),
                          "C:relative.nc", "https://example.invalid/source.nc"):
            with self.subTest(reference=reference):
                (self.inputs / "nested").mkdir(exist_ok=True)
                self.external.source_file = reference
                with self.assertRaises(ValueError):
                    self.snapshot()
        self.external.source_file = "supplied.nc"
        with self.assertRaises(ValueError):
            capture(self.project, source_directory=self.root / "missing")
        with self.assertRaises(ValueError):
            capture(self.project, source_directory=outside)

    def test_link_escape_is_rejected_after_resolving_source(self):
        outside = self.root / "outside.nc"
        outside.write_bytes(PROGRAM)
        # Deterministically model a filesystem resolution escape, including
        # Windows reparse points without depending on symlink privilege.
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == self.inputs / "supplied.nc":
                return outside
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.snapshot()

    @unittest.skipUnless(os.name == "nt", "Windows junction containment")
    def test_actual_junction_cannot_escape_capture_or_bundle_verification(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "program.nc").write_bytes(PROGRAM)
        alias = self.inputs / "linked"
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(outside)],
            capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        try:
            self.external.source_file = "linked/program.nc"
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.snapshot()
        finally:
            # Remove only the newly created junction, never its target.
            os.rmdir(alias)
        self.external.source_file = "supplied.nc"
        destination = self.write()
        asset = next((destination / "sources").iterdir())
        (outside / asset.name).write_bytes(PROGRAM)
        asset.unlink()
        (destination / "sources").rmdir()
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(destination / "sources"), str(outside)],
            capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        try:
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_bundle(destination)
        finally:
            os.rmdir(destination / "sources")

    def test_current_check_detects_each_document_intent_change(self):
        snapshot = self.snapshot()
        for mutate in (
                lambda p: setattr(p, "units", "Inches"),
                lambda p: setattr(p.get_mop("external"), "name", "Changed display"),
                lambda p: setattr(p.get_mop("external"), "enabled", False),
                lambda p: setattr(p.get_mop("external"), "distance_mode", "Incremental"),
                lambda p: setattr(p.get_mop("external"), "arc_center_mode", "Absolute"),
                lambda p: p.assign_mop_to_part("external", "Part", "cut", False),
                lambda p: setattr(p.get_primitive("outline"), "width", 9)):
            changed = self.project.clone()
            mutate(changed)
            with self.assertRaisesRegex(ValueError, "changed"):
                snapshot.check_current(changed, source_directory=self.inputs)
        snapshot.check_current(self.project, source_directory=self.inputs)

    def test_current_check_detects_source_edits_and_missing_file_but_snapshot_remains_usable(self):
        snapshot = self.snapshot()
        (self.inputs / "supplied.nc").write_bytes(PROGRAM + b"M8\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            snapshot.check_current(self.project, source_directory=self.inputs)
        (self.inputs / "supplied.nc").unlink()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            snapshot.check_current(self.project, source_directory=self.inputs)
        destination = self.root / "captured-copy"
        snapshot.write_bundle(destination)
        self.assertEqual(verify_bundle(destination)["status"], "verified")
        asset = next((destination / "sources").iterdir())
        self.assertEqual(asset.read_bytes(), PROGRAM)

    def test_empty_or_inconsistent_snapshot_rejects_before_creating_directory(self):
        no_sources = CBProject("no-dependencies")
        with self.assertRaisesRegex(ValueError, "every NCFile"):
            capture(no_sources, source_directory=self.inputs)
        snapshot = self.snapshot()
        for bad in (replace(snapshot, sources=()),
                    replace(snapshot, sources=snapshot.sources * 2),
                    replace(snapshot, sources=(replace(snapshot.sources[0], original_reference="other.nc"),)),
                    NCFileBundle(snapshot.document_bytes, list(snapshot.sources))):
            with self.subTest(snapshot=bad):
                destination = self.root / "must-not-exist"
                with self.assertRaises(ValueError):
                    bad.write_bundle(destination)
                self.assertFalse(destination.exists())

    def test_write_refuses_existing_destination(self):
        destination = self.write()
        original = (destination / "job.cb").read_bytes()
        with self.assertRaises(FileExistsError):
            self.snapshot().write_bundle(destination)
        self.assertEqual((destination / "job.cb").read_bytes(), original)

    def test_verify_rejects_changed_document_and_changed_missing_asset(self):
        destination = self.write()
        document = destination / "job.cb"
        original = document.read_bytes()
        document.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "document hash"):
            verify_bundle(destination)
        document.write_bytes(original)
        asset = next((destination / "sources").iterdir())
        asset.write_bytes(PROGRAM + b"\n")
        with self.assertRaisesRegex(ValueError, "hash/size"):
            verify_bundle(destination)
        asset.unlink()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            verify_bundle(destination)

    def test_verify_rejects_manifest_path_escape_duplicate_missing_and_extra_records(self):
        destination = self.write()
        manifest_path = destination / "manifest.json"
        original = manifest_path.read_bytes()
        changes = (
            lambda m: m["sources"][0].update(path="../outside.nc"),
            lambda m: m["sources"].append(dict(m["sources"][0])),
            lambda m: m.update(sources=[]),
            lambda m: m["sources"][0].update(mop_id="unknown"),
            lambda m: m["sources"][0].update(identifier="other-operation"),
            lambda m: m["sources"][0].update(size_bytes=True),
            lambda m: m["sources"][0].update(extra="unknown-field"),
            lambda m: m.update(format="future-format"),
            lambda m: m.update(extra="unknown-field"),
        )
        for change in changes:
            manifest_path.write_bytes(original)
            self.change_manifest(destination, change)
            with self.assertRaises(ValueError):
                verify_bundle(destination)
        manifest_path.write_bytes(b'{"format":"one","format":"two"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_bundle(destination)
        manifest_path.write_bytes(b"invalid-json")
        with self.assertRaisesRegex(ValueError, "JSON"):
            verify_bundle(destination)

    def test_verify_checks_document_identity_references_even_with_recomputed_hash(self):
        destination = self.write()
        path = destination / "job.cb"
        project = read_cambam_bytes(path.read_bytes())
        project.get_mop("external").source_file = "sources/other.nc"
        changed = serialize_cambam_bytes(project)
        path.write_bytes(changed)
        self.change_manifest(destination, lambda m: m.update(
            document_sha256=hashlib.sha256(changed).hexdigest()))
        with self.assertRaisesRegex(ValueError, "identity/reference"):
            verify_bundle(destination)

    def test_snapshot_and_verify_reject_all_hidden_raw_ncfile_dependencies(self):
        snapshot = self.snapshot()
        destination = self.write()
        portable = (destination / "job.cb").read_bytes()

        def hidden_operation():
            return ET.fromstring(
                '<ncfile><Name>Hidden</Name><SourceFile>../not-bundled.nc</SourceFile></ncfile>')

        def duplicate_parts(root):
            extra = ET.SubElement(root, "parts")
            part = ET.SubElement(extra, "part", {"Name": "Hidden"})
            ET.SubElement(part, "machineops").append(hidden_operation())

        def duplicate_machineops(root):
            ET.SubElement(root.find("./parts/part"), "machineops").append(hidden_operation())

        def misplaced_root_operation(root):
            root.append(hidden_operation())

        def misplaced_nested_operation(root):
            ET.SubElement(root.find("./parts/part"), "Extension").append(hidden_operation())

        def misplaced_collection(root):
            ET.SubElement(root, "machineops").append(hidden_operation())

        for mutate in (duplicate_parts, duplicate_machineops, misplaced_root_operation,
                       misplaced_nested_operation, misplaced_collection):
            with self.subTest(mutation=mutate.__name__):
                raw = ET.fromstring(snapshot.document_bytes)
                mutate(raw)
                bad = replace(snapshot, document_bytes=ET.tostring(raw, encoding="utf-8"))
                unopened = self.root / "rejected-before-write"
                with self.assertRaisesRegex(ValueError, "NCFile document"):
                    bad.write_bundle(unopened)
                self.assertFalse(unopened.exists())

                raw = ET.fromstring(portable)
                mutate(raw)
                altered = ET.tostring(raw, encoding="utf-8")
                (destination / "job.cb").write_bytes(altered)
                self.change_manifest(destination, lambda m: m.update(
                    document_sha256=hashlib.sha256(altered).hexdigest()))
                with self.assertRaisesRegex(ValueError, "NCFile document"):
                    verify_bundle(destination)


if __name__ == "__main__":
    unittest.main()
