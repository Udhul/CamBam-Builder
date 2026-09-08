"""Regression coverage for MOP identity and relationship XML round trips."""

import json
import tempfile
import unittest
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class MopRoundTripTests(unittest.TestCase):
    """Synthetic MOP fixture with duplicate display names and two parts."""

    def make_project(self):
        project = CBProject("mop-identity-regression")
        layer = project.add_layer("Layer Name")
        part_a = project.add_part("Part Name")
        part_b = project.add_part("Part Two")
        target_a = project.add_rect(layer, width=10.0, height=5.0, identifier="target-a")
        target_b = project.add_circle(layer, center=(20.0, 4.0), diameter=6.0, identifier="target-b")

        operations = [
            (part_a, project.add_profile_mop(
                part_a, targets=[target_a], name="Layer Name", identifier="profile-a",
                target_depth=-1.25, depth_increment=0.25, spindle_speed=1200,
                tool_diameter=3.175, stepover=0.2, profile_side="Outside",
                milling_direction="Climb", cut_feedrate=321.0,
            )),
            (part_a, project.add_pocket_mop(
                part_a, targets=[target_a], name="Layer Name", identifier="pocket-a",
                target_depth=-2.0, depth_increment=0.5, spindle_speed=1300,
                tool_diameter=4.0, stepover=0.3,
                region_fill_style="HorizontalScanline", cut_feedrate=654.0,
            )),
            (part_b, project.add_engrave_mop(
                part_b, targets=[target_b], name="Layer Name", identifier="engrave-b",
                target_depth=-0.4, depth_increment=0.1, spindle_speed=1400,
                tool_diameter=1.5, stock_surface=0.2, cut_feedrate=777.0,
            )),
            (part_b, project.add_drill_mop(
                part_b, targets=[target_b], name="Part Name", identifier="drill-b",
                target_depth=-3.0, depth_increment=1.0, spindle_speed=1500,
                tool_diameter=2.5, drilling_method="CannedCycle",
                peck_distance=0.75, dwell=25.0, cut_feedrate=888.0,
            )),
        ]
        self.assertTrue(all(part is not None and mop is not None for part, mop in operations))
        return project, operations

    @staticmethod
    def mop_elements(tree):
        return list(tree.getroot().findall("./parts/part/machineops/*"))

    @staticmethod
    def mop_tags(path):
        tree = ET.parse(path)
        return [json.loads(element.findtext("Tag")) for element in MopRoundTripTests.mop_elements(tree)]

    def save(self, project, directory, name):
        path = Path(directory) / (name + ".cb")
        save_cambam_file(project, str(path))
        return path

    def assert_mop_state(self, source, loaded):
        self.assertEqual(len(source.list_mops()), len(loaded.list_mops()))
        expected_by_id = {mop.user_identifier: mop for mop in source.list_mops()}
        actual_by_id = {mop.user_identifier: mop for mop in loaded.list_mops()}
        self.assertEqual(set(expected_by_id), set(actual_by_id))
        for identifier, expected in expected_by_id.items():
            actual = actual_by_id[identifier]
            self.assertEqual(expected.internal_id, actual.internal_id)
            self.assertEqual(type(expected), type(actual))
            self.assertIs(actual, loaded.get_mop(expected.internal_id))
            self.assertEqual(expected.name, actual.name)
            self.assertFalse(hasattr(expected, "pid_source"))
            self.assertFalse(hasattr(actual, "pid_source"))
            self.assertEqual(source.get_mop_targets(expected), loaded.get_mop_targets(actual))
            self.assertIsNone(source.get_mop_target_group(expected))
            self.assertIsNone(loaded.get_mop_target_group(actual))
            for field in ("target_depth", "depth_increment", "spindle_speed", "tool_diameter", "cut_feedrate"):
                self.assertEqual(getattr(expected, field), getattr(actual, field))

        for part in source.list_parts():
            self.assertEqual(
                [m.user_identifier for m in source.get_mops_in_part(part)],
                [m.user_identifier for m in loaded.get_mops_in_part(part.user_identifier)],
            )

        self.assertEqual(source.get_mop("profile-a").stepover, loaded.get_mop("profile-a").stepover)
        self.assertEqual(source.get_mop("profile-a").profile_side, loaded.get_mop("profile-a").profile_side)
        self.assertEqual(source.get_mop("pocket-a").region_fill_style, loaded.get_mop("pocket-a").region_fill_style)
        self.assertEqual(source.get_mop("drill-b").drilling_method, loaded.get_mop("drill-b").drilling_method)
        self.assertEqual(source.get_mop("drill-b").peck_distance, loaded.get_mop("drill-b").peck_distance)
        self.assertEqual(source.get_mop("drill-b").dwell, loaded.get_mop("drill-b").dwell)

    def test_writer_emits_identity_and_two_round_trips_preserve_mops(self):
        source, operations = self.make_project()
        with tempfile.TemporaryDirectory() as directory:
            first = self.save(source, directory, "first")
            tags = self.mop_tags(first)
            self.assertEqual(
                {(tag["user_id"], uuid.UUID(tag["internal_id"])) for tag in tags},
                {(mop.user_identifier, mop.internal_id) for _, mop in operations},
            )
            loaded_once = read_cambam_file(str(first))
            self.assertIsNotNone(loaded_once)
            self.assert_mop_state(source, loaded_once)
            second = self.save(loaded_once, directory, "second")
            loaded_twice = read_cambam_file(str(second))
            self.assertIsNotNone(loaded_twice)
            self.assert_mop_state(source, loaded_twice)

    def test_missing_tags_use_fresh_distinct_identities_for_colliding_names(self):
        source, _ = self.make_project()
        source_ids = {mop.internal_id for mop in source.list_mops()}
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(source, directory, "legacy")
            tree = ET.parse(path)
            for element in self.mop_elements(tree):
                tag = element.find("Tag")
                if tag is not None:
                    element.remove(tag)
            tree.write(path, encoding="utf-8", xml_declaration=True)
            loaded = read_cambam_file(str(path))
            self.assertIsNotNone(loaded)
            mops = loaded.list_mops()
            self.assertEqual(4, len(mops))
            self.assertEqual(4, len({mop.internal_id for mop in mops}))
            self.assertTrue(source_ids.isdisjoint({mop.internal_id for mop in mops}))
            self.assertEqual({"Layer Name", "Part Name"}, {mop.name for mop in mops})
            again = read_cambam_file(str(self.save(loaded, directory, "legacy-again")))
            self.assertIsNotNone(again)
            self.assertEqual([m.internal_id for m in mops], [m.internal_id for m in again.list_mops()])
            self.assertEqual([m.user_identifier for m in mops], [m.user_identifier for m in again.list_mops()])

    def test_invalid_identity_metadata_uses_fresh_identity(self):
        source, _ = self.make_project()
        source_by_id = {mop.user_identifier: mop for mop in source.list_mops()}
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(source, directory, "invalid")
            tree = ET.parse(path)
            elements = self.mop_elements(tree)
            elements[0].find("Tag").text = "not-json"
            elements[1].find("Tag").text = json.dumps(["not", "an", "object"])
            elements[2].find("Tag").text = json.dumps({"user_id": "engrave-b", "internal_id": "invalid-uuid"})
            tree.write(path, encoding="utf-8", xml_declaration=True)
            loaded = read_cambam_file(str(path))
            self.assertIsNotNone(loaded)
            self.assertEqual(4, len(loaded.list_mops()))
            self.assertEqual(source_by_id["drill-b"].internal_id, loaded.get_mop("drill-b").internal_id)
            loaded_ids = {mop.internal_id for mop in loaded.list_mops()}
            self.assertEqual(4, len(loaded_ids))
            self.assertTrue({source_by_id[key].internal_id for key in ("profile-a", "pocket-a", "engrave-b")}.isdisjoint(loaded_ids))

    def test_duplicate_explicit_identity_aborts_import(self):
        for duplicate_kind in ("internal_id", "user_id"):
            with self.subTest(duplicate_kind=duplicate_kind), tempfile.TemporaryDirectory() as directory:
                source, _ = self.make_project()
                path = self.save(source, directory, "duplicate")
                tree = ET.parse(path)
                first_tag = json.loads(self.mop_elements(tree)[0].findtext("Tag"))
                second_tag = json.loads(self.mop_elements(tree)[1].findtext("Tag"))
                second_tag[duplicate_kind] = first_tag[duplicate_kind]
                self.mop_elements(tree)[1].find("Tag").text = json.dumps(second_tag)
                tree.write(path, encoding="utf-8", xml_declaration=True)
                self.assertIsNone(read_cambam_file(str(path)))

    def test_identity_collision_with_geometry_aborts_import(self):
        for field in ("internal_id", "user_id"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                source, _ = self.make_project()
                path = self.save(source, directory, "geometry-collision")
                tree = ET.parse(path)
                element = self.mop_elements(tree)[0]
                metadata = json.loads(element.findtext("Tag"))
                primitive = source.get_primitive("target-a")
                metadata[field] = str(primitive.internal_id) if field == "internal_id" else primitive.user_identifier
                element.find("Tag").text = json.dumps(metadata)
                tree.write(path, encoding="utf-8", xml_declaration=True)
                self.assertIsNone(read_cambam_file(str(path)))


if __name__ == "__main__":
    unittest.main()
