"""Regression coverage for live group and snapshot MOP sources."""

import json
import tempfile
import unittest
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class MopGroupSourceTests(unittest.TestCase):
    """Synthetic primitive groups used to characterize MOP source modes."""

    def make_project(self):
        project = CBProject("mop-group-source-regression")
        layer = project.add_layer("Geometry")
        part = project.add_part("Machining")
        first = project.add_rect(
            layer, corner=(0.0, 0.0), width=4.0, height=2.0,
            identifier="first", groups=["targets", "first"],
        )
        second = project.add_circle(
            layer, center=(10.0, 2.0), diameter=3.0,
            identifier="second", groups=["targets"],
        )
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

    def test_group_string_tracks_membership_and_group_recreation(self):
        project, layer, _, first, second = self.make_project()
        mop = project.add_profile_mop(project.list_parts()[0], "targets", identifier="live")
        self.assertIsNotNone(mop)
        self.assertEqual("targets", mop.pid_source)
        self.assertEqual(
            {first.internal_id, second.internal_id},
            set(project.resolve_pid_source_to_uuids(mop.pid_source)),
        )

        self.assertTrue(project.remove_primitive_from_group(first, "targets"))
        self.assertEqual(
            [second.internal_id],
            project.resolve_pid_source_to_uuids(mop.pid_source),
        )
        self.assertTrue(project.remove_entity(second))
        self.assertNotIn("targets", project.list_groups())
        self.assertEqual([], project.resolve_pid_source_to_uuids(mop.pid_source))

        replacement = project.add_rect(
            layer, corner=(20.0, 0.0), width=2.0, height=2.0,
            identifier="replacement",
        )
        self.assertTrue(project.add_primitive_to_group(replacement, "targets"))
        self.assertEqual(
            [replacement.internal_id],
            project.resolve_pid_source_to_uuids(mop.pid_source),
        )
        self.assertEqual("targets", mop.pid_source)

    def test_list_source_is_uuid_snapshot_and_does_not_confuse_group_string(self):
        project, _, part, first, second = self.make_project()
        # The same text is a group name when passed bare, but an identifier is
        # resolved as an explicit snapshot when passed inside a list.
        group_mop = project.add_profile_mop(part, "first", identifier="group")
        snapshot_input = [first, first.user_identifier, first.internal_id, "missing"]
        snapshot_mop = project.add_profile_mop(part, snapshot_input, identifier="snapshot")
        self.assertEqual("first", group_mop.pid_source)
        self.assertTrue(all(isinstance(value, uuid.UUID) for value in snapshot_mop.pid_source))
        self.assertEqual({first.internal_id}, set(snapshot_mop.pid_source))
        snapshot_input.append(second)
        self.assertEqual({first.internal_id}, set(snapshot_mop.pid_source))
        self.assertEqual([first.internal_id], project.resolve_pid_source_to_uuids(snapshot_mop.pid_source))

        self.assertTrue(project.add_primitive_to_group(second, "first"))
        self.assertEqual(
            {first.internal_id, second.internal_id},
            set(project.resolve_pid_source_to_uuids(group_mop.pid_source)),
        )
        self.assertEqual([first.internal_id], project.resolve_pid_source_to_uuids(snapshot_mop.pid_source))

    def test_missing_group_resolves_to_empty_without_rejecting_mop(self):
        project, _, part, _, _ = self.make_project()
        mop = project.add_profile_mop(part, "does-not-exist", identifier="missing-group")
        empty = project.add_profile_mop(part, "", identifier="empty-group")
        invalid = project.add_profile_mop(part, ["missing", uuid.uuid4()], identifier="all-invalid")
        self.assertIsNotNone(mop)
        self.assertIsNotNone(empty)
        self.assertIsNotNone(invalid)
        self.assertEqual("does-not-exist", mop.pid_source)
        self.assertEqual("", empty.pid_source)
        self.assertEqual([], invalid.pid_source)
        self.assertEqual([], project.resolve_pid_source_to_uuids(mop.pid_source))
        self.assertEqual([], project.resolve_pid_source_to_uuids(empty.pid_source))
        self.assertEqual([], project.resolve_pid_source_to_uuids(invalid.pid_source))

    def test_direct_pid_source_mode_switch_keeps_string_and_uuid_modes(self):
        project, _, part, first, _ = self.make_project()
        mop = project.add_profile_mop(part, [], identifier="mode-switch")
        self.assertIsNotNone(mop)

        mop.pid_source = [first.internal_id]
        self.assertEqual([first.internal_id], project.resolve_pid_source_to_uuids(mop.pid_source))
        mop.pid_source = "targets"
        self.assertEqual(
            {first.internal_id, project.get_primitive("second").internal_id},
            set(project.resolve_pid_source_to_uuids(mop.pid_source)),
        )
        mop.pid_source = [first.internal_id]
        self.assertEqual([first.internal_id], project.resolve_pid_source_to_uuids(mop.pid_source))

    def test_uuid_snapshot_does_not_retarget_after_deletion_and_identifier_reuse(self):
        project, layer, part, first, _ = self.make_project()
        mop = project.add_profile_mop(part, [first], identifier="snapshot")
        old_uuid = first.internal_id
        self.assertTrue(project.remove_entity(first))
        replacement = project.add_rect(
            layer, corner=(30.0, 0.0), width=2.0, height=2.0,
            identifier="first", groups=["targets"],
        )
        self.assertNotEqual(old_uuid, replacement.internal_id)
        self.assertEqual([], project.resolve_pid_source_to_uuids(mop.pid_source))

    def test_all_mop_types_export_refs_and_import_uuid_snapshots_twice(self):
        project, layer, part, first, second = self.make_project()
        common = dict(target_depth=-1.0, depth_increment=0.5, spindle_speed=1000,
                      tool_diameter=1.0, cut_feedrate=100.0)
        mops = [
            project.add_profile_mop(part, "targets", identifier="profile", **common),
            project.add_pocket_mop(part, "targets", identifier="pocket", **common),
            project.add_engrave_mop(part, "targets", identifier="engrave", **common),
            project.add_drill_mop(part, "targets", identifier="drill", **common),
        ]
        self.assertTrue(all(mop is not None for mop in mops))
        self.assertTrue(all(mop.pid_source == "targets" for mop in mops))

        output = Path("output").resolve()
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mop-group-sources-", dir=output) as directory:
            first_path = self.save(project, directory, "first")
            tree = ET.parse(first_path)
            xml_ids = self.primitive_xml_ids(tree)
            expected_xml_ids = {xml_ids["first"], xml_ids["second"]}
            for element in self.mop_elements(tree):
                refs = [int(value.text) for value in element.findall("./primitive/prim")]
                self.assertEqual(expected_xml_ids, set(refs))
            self.assertTrue(all(mop.pid_source == "targets" for mop in mops))

            loaded_once = read_cambam_file(str(first_path))
            self.assertIsNotNone(loaded_once)
            loaded_mops = {mop.user_identifier: mop for mop in loaded_once.list_mops()}
            expected_uuids = {first.internal_id, second.internal_id}
            expected_groups = {"first": ["first", "targets"], "second": ["targets"]}
            self.assertEqual(
                ["profile", "pocket", "engrave", "drill"],
                [mop.user_identifier for mop in loaded_once.get_mops_in_part("Machining")],
            )
            for identifier in ("profile", "pocket", "engrave", "drill"):
                self.assertIsInstance(loaded_mops[identifier].pid_source, list)
                self.assertTrue(all(isinstance(value, uuid.UUID) for value in loaded_mops[identifier].pid_source))
                self.assertEqual(expected_uuids, set(loaded_mops[identifier].pid_source))
            for identifier, groups in expected_groups.items():
                self.assertEqual(groups, loaded_once.get_primitive(identifier).groups)
            self.assertEqual((0.0, 0.0), loaded_once.get_primitive("first").relative_corner)
            self.assertEqual(4.0, loaded_once.get_primitive("first").width)
            self.assertEqual((10.0, 2.0), loaded_once.get_primitive("second").relative_center)
            self.assertEqual(3.0, loaded_once.get_primitive("second").diameter)

            second_path = self.save(loaded_once, directory, "second")
            loaded_twice = read_cambam_file(str(second_path))
            self.assertIsNotNone(loaded_twice)
            self.assertEqual(
                ["profile", "pocket", "engrave", "drill"],
                [mop.user_identifier for mop in loaded_twice.get_mops_in_part("Machining")],
            )
            for identifier in ("profile", "pocket", "engrave", "drill"):
                imported = loaded_twice.get_mop(identifier)
                self.assertIsInstance(imported.pid_source, list)
                self.assertEqual(expected_uuids, set(imported.pid_source))
            for identifier, groups in expected_groups.items():
                self.assertEqual(groups, loaded_twice.get_primitive(identifier).groups)

            imported_first = loaded_twice.get_primitive("first")
            imported_second = loaded_twice.get_primitive("second")
            imported_layer = loaded_twice.get_layer_of_primitive(imported_first)
            self.assertTrue(loaded_twice.remove_primitive_from_group(imported_first, "targets"))
            self.assertTrue(loaded_twice.add_primitive_to_group(imported_second, "other"))
            added = loaded_twice.add_rect(
                imported_layer, corner=(40.0, 0.0), width=2.0, height=2.0,
                identifier="added", groups=["targets"],
            )
            self.assertIsNotNone(added)
            for identifier in ("profile", "pocket", "engrave", "drill"):
                imported = loaded_twice.get_mop(identifier)
                self.assertEqual(expected_uuids, set(imported.pid_source))
                self.assertEqual(expected_uuids, set(loaded_twice.resolve_pid_source_to_uuids(imported.pid_source)))
                self.assertNotIn(added.internal_id, imported.pid_source)


if __name__ == "__main__":
    unittest.main()
