"""Regression coverage for pickle state persistence paths and project links."""

import os
import tempfile
import unittest
from pathlib import Path

from cambam_builder import CBProject


class StatePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.output = Path("output").resolve()
        self.output.mkdir(exist_ok=True)

    def make_project(self):
        project = CBProject("state-regression")
        layer = project.add_layer("Geometry")
        primitive = project.add_rect(
            layer, corner=(1.0, 2.0), width=3.0, height=4.0, identifier="outline"
        )
        return project, primitive

    def test_bare_filename_saves_and_round_trips_with_primitive_project_link(self):
        project, primitive = self.make_project()
        with tempfile.TemporaryDirectory(prefix="state-tests-", dir=self.output) as directory:
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                self.assertIsNone(project.save_state("project.pkl"))
                state_path = Path(directory) / "project.pkl"
                self.assertTrue(state_path.is_file())

                loaded = CBProject.load_state("project.pkl")
            finally:
                os.chdir(previous_directory)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.project_name, project.project_name)
        loaded_primitive = loaded.get_primitive(primitive.user_identifier)
        self.assertIsNotNone(loaded_primitive)
        self.assertIs(loaded_primitive.get_project(), loaded)

    def test_save_state_creates_nested_parent_directories(self):
        project, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="state-tests-", dir=self.output) as directory:
            state_path = Path(directory) / "one" / "two" / "project.pkl"
            self.assertIsNone(project.save_state(str(state_path)))
            self.assertTrue(state_path.is_file())

    def test_save_state_raises_when_parent_cannot_be_created(self):
        project, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="state-tests-", dir=self.output) as directory:
            blocker = Path(directory) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            state_path = blocker / "project.pkl"
            with self.assertRaises(OSError):
                project.save_state(str(state_path))


if __name__ == "__main__":
    unittest.main()
