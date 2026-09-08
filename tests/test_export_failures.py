"""Export must report failures and preserve the destination until completion."""

import tempfile
import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import build_xml_tree


class ExportFailureTests(unittest.TestCase):
    def setUp(self):
        output = Path("output")
        output.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="export-tests-", dir=output)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name).resolve() / "project.cb"
        self.project = CBProject("export-regression")
        self.layer = self.project.add_layer("Geometry")
        self.primitive = self.project.add_rect(self.layer, width=10, height=5, identifier="outline")
        self.part = self.project.add_part("Part")
        self.mop = self.project.add_profile_mop(
            self.part, [self.primitive], identifier="profile", target_depth=-1,
            tool_diameter=3, cut_feedrate=300,
        )

    def assert_destination(self, existing):
        if existing:
            self.assertEqual(self.path.read_bytes(), b"previous complete document")
        else:
            self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [self.path] if existing else [])

    def test_entity_encoding_errors_abort_tree_and_both_public_save_methods(self):
        for entity in (self.layer, self.primitive, self.part, self.mop):
            for existing in (False, True):
                for method in (self.project.save, self.project.export):
                    with self.subTest(entity=type(entity).__name__, existing=existing, method=method.__name__):
                        if existing:
                            self.path.write_bytes(b"previous complete document")
                        failure = ValueError("synthetic encoder failure")
                        with patch.object(entity, "to_xml_element", side_effect=failure):
                            with self.assertRaises(ValueError) as caught:
                                build_xml_tree(self.project)
                            self.assertIs(caught.exception, failure)
                            with self.assertRaises(ValueError):
                                method(str(self.path))
                        self.assert_destination(existing)
                        if existing:
                            self.path.unlink()

    def test_partial_serialization_and_replace_errors_preserve_destination(self):
        def partial_write(tree, destination, **kwargs):
            if hasattr(destination, "write"):
                destination.write(b"partial XML")
            else:
                Path(destination).write_bytes(b"partial XML")
            raise OSError("synthetic write failure")

        for stage in ("write", "replace"):
            for existing in (False, True):
                with self.subTest(stage=stage, existing=existing):
                    if existing:
                        self.path.write_bytes(b"previous complete document")
                    target = "xml.etree.ElementTree.ElementTree.write" if stage == "write" else "cambam_builder.cambam_writer.os.replace"
                    effect = partial_write if stage == "write" else OSError("synthetic replace failure")
                    with patch(target, autospec=True, side_effect=effect):
                        with self.assertRaises(OSError):
                            self.project.export(str(self.path))
                    self.assert_destination(existing)
                    if existing:
                        self.path.unlink()

    def test_destination_directory_error_propagates(self):
        self.path.write_bytes(b"previous complete document")
        with self.assertRaises(OSError):
            self.project.export(str(self.path / "child.cb"))
        self.assert_destination(True)

    def test_prewrite_failures_preserve_destination(self):
        for target in (
            "cambam_builder.cambam_writer.ET.indent",
            "cambam_builder.cambam_writer.tempfile.NamedTemporaryFile",
        ):
            with self.subTest(target=target):
                self.path.write_bytes(b"previous complete document")
                with patch(target, side_effect=OSError("synthetic prewrite failure"), create=True):
                    with self.assertRaises(OSError):
                        self.project.export(str(self.path))
                self.assert_destination(True)

    def test_mop_resolution_error_aborts_export(self):
        with patch.object(self.project, "get_mop_targets", side_effect=ValueError("bad source")):
            with self.assertRaises(ValueError):
                self.project.export(str(self.path))
        self.assert_destination(False)

    def test_missing_ordered_entities_are_not_silently_skipped(self):
        for registry, entity in ((self.project._layers, self.layer), (self.project._parts, self.part)):
            with self.subTest(entity=type(entity).__name__):
                registry.pop(entity.internal_id)
                try:
                    with self.assertRaises(ValueError):
                        self.project.export(str(self.path))
                    self.assert_destination(False)
                finally:
                    registry[entity.internal_id] = entity

    def test_bare_filename_and_missing_indent_support(self):
        previous_directory = Path.cwd()
        try:
            os.chdir(self.path.parent)
            with patch("cambam_builder.cambam_writer.ET.indent", None, create=True):
                self.assertIsNone(self.project.save("project"))
            self.assertEqual(ET.parse(self.path).getroot().tag, "CADFile")
            self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        finally:
            os.chdir(previous_directory)

    def test_success_replaces_complete_file_and_preserves_roundtrip_content(self):
        for pretty in (False, True):
            with self.subTest(pretty=pretty):
                self.path.write_bytes(b"previous complete document")
                self.assertIsNone(self.project.export(str(self.path.with_suffix(".xml")), pretty_print=pretty))
                tree = ET.parse(self.path)
                primitives = tree.findall("./layers/layer/objects/*")
                mops = tree.findall("./parts/part/machineops/*")
                self.assertEqual(len(primitives), 1)
                self.assertEqual(len(mops), 1)
                loaded = read_cambam_file(str(self.path))
                self.assertIsNotNone(loaded)
                self.assertEqual(set(loaded._primitives), set(self.project._primitives))
                self.assertEqual(set(loaded._mops), set(self.project._mops))
                self.assertEqual(loaded.get_mop_targets("profile"), [self.primitive.internal_id])
                self.assertEqual(loaded.get_mop("profile").target_depth, -1)
                self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
