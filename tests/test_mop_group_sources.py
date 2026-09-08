"""Regression coverage for core-owned MOP target selections."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class MopGroupSourceTests(unittest.TestCase):
    def make_project(self):
        project = CBProject("mop-target-regression")
        layer = project.add_layer("Geometry")
        part = project.add_part("Machining")
        first = project.add_rect(layer, width=4.0, height=2.0,
                                 identifier="first", groups=["targets", "first"])
        second = project.add_circle(layer, center=(10.0, 2.0), diameter=3.0,
                                    identifier="second", groups=["targets"])
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        return project, layer, part, first, second

    @staticmethod
    def save(project, directory, name):
        path = Path(directory) / (name + ".cb")
        save_cambam_file(project, str(path))
        return path

    @staticmethod
    def mop_elements(tree):
        return list(tree.getroot().findall("./parts/part/machineops/*"))

    @staticmethod
    def primitive_xml_ids(tree):
        result = {}
        for element in tree.getroot().findall("./layers/layer/objects/*"):
            metadata = json.loads(element.findtext("Tag"))
            result[metadata["user_id"]] = int(element.get("id"))
        return result

    def test_group_selection_is_live_and_recreation_retargets_it(self):
        project, layer, part, first, second = self.make_project()
        mop = project.add_profile_mop(part, target_group="targets", identifier="live")
        self.assertFalse(hasattr(mop, "pid_source"))
        self.assertEqual("targets", project.get_mop_target_group(mop))
        self.assertEqual(sorted((first.internal_id, second.internal_id)),
                         project.get_mop_targets(mop))

        self.assertTrue(project.remove_primitive_from_group(first, "targets"))
        self.assertEqual([second.internal_id], project.get_mop_targets(mop))
        self.assertTrue(project.remove_entity(second))
        self.assertEqual([], project.get_mop_targets(mop))

        replacement = project.add_rect(layer, corner=(20.0, 0.0), width=2.0,
                                       height=2.0, identifier="replacement")
        self.assertTrue(project.add_primitive_to_group(replacement, "targets"))
        self.assertEqual([replacement.internal_id], project.get_mop_targets(mop))

    def test_explicit_targets_are_snapshot_and_identifiers_are_strict(self):
        project, _, part, first, second = self.make_project()
        mop = project.add_profile_mop(part, targets=(first, first.user_identifier),
                                      identifier="snapshot")
        self.assertFalse(hasattr(mop, "pid_source"))
        expected = [first.internal_id]
        self.assertEqual(expected, project.get_mop_targets(mop))
        returned = project.get_mop_targets(mop)
        returned.append(second.internal_id)
        self.assertEqual(expected, project.get_mop_targets(mop))
        self.assertIsNone(project.get_mop_target_group(mop))

        with self.assertRaises(TypeError):
            project.add_profile_mop(part, targets="first", identifier="bare-string")
        with self.assertRaises(ValueError):
            project.add_profile_mop(part, targets=["missing"], identifier="missing")
        with self.assertRaises(ValueError):
            project.add_profile_mop(part, targets=[first], target_group="targets",
                                    identifier="mixed")

        old = project.get_mop_targets(mop)
        with self.assertRaises(TypeError):
            project.set_mop_targets(mop, "first")
        self.assertEqual(old, project.get_mop_targets(mop))
        with self.assertRaises(ValueError):
            project.set_mop_targets(mop, ["missing"])
        self.assertEqual(old, project.get_mop_targets(mop))

    def test_setters_replace_selection_atomically_and_group_missing_is_empty(self):
        project, _, part, first, second = self.make_project()
        mop = project.add_profile_mop(part, targets=[first], identifier="setters")
        project.set_mop_targets(mop, [second])
        self.assertEqual([second.internal_id], project.get_mop_targets(mop))
        self.assertIsNone(project.get_mop_target_group(mop))
        project.set_mop_target_group(mop, "does-not-exist")
        self.assertEqual([], project.get_mop_targets(mop))
        self.assertEqual("does-not-exist", project.get_mop_target_group(mop))
        for invalid in ("", None, [], 12):
            with self.subTest(group=invalid), self.assertRaises(ValueError):
                project.set_mop_target_group(mop, invalid)
            self.assertEqual("does-not-exist", project.get_mop_target_group(mop))

    def test_explicit_selection_cleans_on_deletion_and_does_not_reuse_identifier(self):
        project, layer, part, first, _ = self.make_project()
        mop = project.add_profile_mop(part, targets=[first], identifier="snapshot")
        old_uuid = first.internal_id
        self.assertTrue(project.remove_entity(first))
        self.assertEqual([], project.get_mop_targets(mop))
        replacement = project.add_rect(layer, identifier="first", width=2.0, height=2.0)
        self.assertNotEqual(old_uuid, replacement.internal_id)
        self.assertEqual([], project.get_mop_targets(mop))

    def test_all_mop_types_export_snapshot_refs_and_roundtrip_twice(self):
        project, _, part, first, second = self.make_project()
        common = dict(target_depth=-1.0, depth_increment=0.5, spindle_speed=1000,
                      tool_diameter=1.0, cut_feedrate=100.0)
        mops = [
            project.add_profile_mop(part, target_group="targets", identifier="profile", **common),
            project.add_pocket_mop(part, target_group="targets", identifier="pocket", **common),
            project.add_engrave_mop(part, target_group="targets", identifier="engrave", **common),
            project.add_drill_mop(part, target_group="targets", identifier="drill", **common),
        ]
        expected = {first.internal_id, second.internal_id}
        self.assertTrue(all(project.get_mop_targets(mop) == sorted(expected) for mop in mops))
        self.assertTrue(all(project.get_mop_target_group(mop) == "targets" for mop in mops))

        output = Path("output").resolve()
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mop-targets-", dir=output) as directory:
            first_path = self.save(project, directory, "first")
            tree = ET.parse(first_path)
            xml_ids = self.primitive_xml_ids(tree)
            for element in self.mop_elements(tree):
                refs = {int(value.text) for value in element.findall("./primitive/prim")}
                self.assertEqual({xml_ids["first"], xml_ids["second"]}, refs)
                self.assertNotIn("target_group", (element.findtext("Tag") or ""))
                self.assertNotIn("targets", ET.tostring(element, encoding="unicode"))

            loaded_once = read_cambam_file(str(first_path))
            self.assertIsNotNone(loaded_once)
            for identifier in ("profile", "pocket", "engrave", "drill"):
                imported = loaded_once.get_mop(identifier)
                self.assertEqual(sorted(expected), loaded_once.get_mop_targets(imported))
                self.assertIsNone(loaded_once.get_mop_target_group(imported))

            second_path = self.save(loaded_once, directory, "second")
            loaded_twice = read_cambam_file(str(second_path))
            self.assertEqual(["profile", "pocket", "engrave", "drill"],
                             [m.user_identifier for m in loaded_twice.list_mops()])
            loaded_twice.remove_primitive_from_group("first", "targets")
            loaded_twice.add_rect("Geometry", identifier="new-member", groups=["targets"])
            for identifier in ("profile", "pocket", "engrave", "drill"):
                imported = loaded_twice.get_mop(identifier)
                self.assertEqual(sorted(expected), loaded_twice.get_mop_targets(imported))
                self.assertIsNone(loaded_twice.get_mop_target_group(imported))

    def test_removing_mop_clears_registry(self):
        project, _, part, first, _ = self.make_project()
        mop = project.add_profile_mop(part, targets=[first], identifier="remove-me")
        mop_id = mop.internal_id
        self.assertTrue(project.remove_mop(mop))
        self.assertIsNone(project.get_mop(mop_id))
        self.assertNotIn(mop, project.list_mops())
        self.assertNotIn(mop_id, project._mop_targets)
        with self.assertRaises(ValueError):
            project.get_mop_targets(mop)


if __name__ == "__main__":
    unittest.main()
