"""Fault-sensitive checks for package archive and subprocess provenance gates."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from tools.verification_package import (
    _archive_members, _assert_archive_bytes, _write_import_guard,
)


class VerificationPackageTests(unittest.TestCase):
    def test_archive_rejects_disposable_and_traversal_members(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "sample.whl"
            for name in ("output/job.nc", "cambam_builder/__pycache__/core.pyc",
                         "../outside.py"):
                with self.subTest(name=name):
                    with zipfile.ZipFile(wheel, "w") as opened:
                        opened.writestr(name, "unexpected")
                    with self.assertRaises(ValueError):
                        _archive_members(wheel, source=False)

    def test_archive_byte_check_rejects_stale_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.py").write_bytes(b"current")
            wheel = root / "sample.whl"
            with zipfile.ZipFile(wheel, "w") as opened:
                opened.writestr("example.py", b"stale")
            with self.assertRaisesRegex(ValueError, "stale or changed bytes"):
                _assert_archive_bytes(root, wheel, {"example.py": "example.py"},
                                      ["example.py"], source=False)

    def test_child_import_guard_rejects_checkout_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            _write_import_guard(snapshot)
            checkout = root / "checkout"
            package = checkout / "cambam_builder"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("SOURCE = True\n", encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONNOUSERSITE="1")
            result = subprocess.run(
                [sys.executable, "-c", "import cambam_builder"], cwd=checkout,
                env=env, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside installed site-packages", result.stderr)


if __name__ == "__main__":
    unittest.main()
