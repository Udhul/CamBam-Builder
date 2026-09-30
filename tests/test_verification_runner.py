"""Fault-sensitive checks for the development runner; no CAM fixture inputs."""
import importlib.util
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[1] / "tools" / "verify.py"
spec = importlib.util.spec_from_file_location("verification_runner", RUNNER)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class VerificationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "tests").mkdir()

    def execute(self, body):
        (self.root / "tests" / "test_sample.py").write_text(body, encoding="utf-8")
        result = self.root / "result.json"
        command = [sys.executable, str(RUNNER), "_worker", str(self.root), "test_sample", str(result)]
        completed = subprocess.run(command, capture_output=True, text=True)
        return completed, json.loads(result.read_text(encoding="utf-8"))

    def test_success_and_named_skip_have_serializable_test_ids(self):
        completed, result = self.execute("""import unittest
class Sample(unittest.TestCase):
    def test_pass(self): self.assertEqual(2 + 2, 4)
    @unittest.skip('explicit observation boundary')
    def test_skip(self): pass
""")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["tests_run"], 2)
        self.assertEqual(len(result["test_ids"]), 2)
        self.assertEqual(result["tests"][1]["detail"], "explicit observation boundary")
        self.assertGreater(result["wall_seconds"], 0)
        self.assertGreaterEqual(result["cpu_seconds"], 0)

    def test_subtest_failure_and_unexpected_success_are_failures(self):
        completed, result = self.execute("""import unittest
class Sample(unittest.TestCase):
    def test_subtest(self):
        with self.subTest(value=1): self.assertEqual(1, 2)
    @unittest.expectedFailure
    def test_unexpected(self): pass
""")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(result["status"], "fail")
        self.assertEqual({t["status"] for t in result["tests"]}, {"fail", "unexpected_success"})

    def test_collection_error_is_not_optional_skip(self):
        completed, result = self.execute("import missing_dependency_for_runner_test\n")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["collection_errors"])

    def test_empty_module_cannot_pass(self):
        completed, result = self.execute("# no tests\n")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(result["status"], "fail")

    def test_terminated_worker_retains_incomplete_record(self):
        (self.root / "tests" / "test_sample.py").write_text(
            "import time\ntime.sleep(30)\n", encoding="utf-8")
        result = self.root / "result.json"
        process = subprocess.Popen([sys.executable, str(RUNNER), "_worker", str(self.root),
                                    "test_sample", str(result)], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while not result.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(result.exists())
        finally:
            process.terminate()
            process.wait(timeout=10)
        self.assertEqual(json.loads(result.read_text())["status"], "incomplete")

    def test_report_write_failure_is_nonzero_and_keeps_test_log(self):
        (self.root / "tests" / "test_sample.py").write_text("""import unittest
from pathlib import Path
class Sample(unittest.TestCase):
    def test_break_report(self):
        Path(__file__).parents[1].joinpath('result.json.tmp').mkdir()
""", encoding="utf-8")
        commands = verify.Commands(self.root)
        record = commands("report failure", [sys.executable, str(RUNNER), "_worker", str(self.root),
                          "test_sample", str(self.root / "result.json")], self.root, check=False)
        self.assertNotEqual(record["returncode"], 0)
        self.assertEqual(json.loads((self.root / "result.json").read_text())["status"], "incomplete")
        self.assertIn("test_break_report", (self.root / record["log"]).read_text())

    def test_setup_failure_records_incomplete_command(self):
        commands = verify.Commands(self.root)
        with self.assertRaises(OSError):
            commands("missing executable", [str(self.root / "missing.exe")], self.root)
        record = json.loads((self.root / "command-0000.json").read_text())
        self.assertEqual(record["status"], "incomplete")
        self.assertTrue((self.root / record["log"]).exists())

    def test_reconciliation_requires_all_identity_boundaries_and_inventory(self):
        prior = {"identity": {"source": "a"}, "target_identities": {"deps": "b"},
                 "inventory": {"checkout": ["test_a", "test_b"]}}
        verify.reusable(prior, prior["identity"], prior["target_identities"], prior["inventory"])
        for field in ("identity", "target_identities", "inventory"):
            inputs = {key: dict(value) for key, value in prior.items()}
            inputs[field]["changed"] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "refused"):
                verify.reusable(prior, inputs["identity"], inputs["target_identities"], inputs["inventory"])

    def test_module_selection_deduplicates_and_includes_all_matching_modules(self):
        for name in ("test_a.py", "test_b.py", "helper.py"):
            (self.root / "tests" / name).touch()
        self.assertEqual(verify.modules_in(self.root, ["test_*.py", "test_a.py"]), ["test_a", "test_b"])

    def test_resume_reuses_pass_and_reruns_failure_then_rejects_changed_source(self):
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
            (self.root / name).touch()
        (self.root / "demos").mkdir()
        (self.root / "demos/mcp_client_acceptance_verify.py").touch()
        (self.root / "tests/test_pass.py").write_text(
            "import unittest\nclass Passing(unittest.TestCase):\n def test_ok(self): pass\n")
        (self.root / "tests/test_retry.py").write_text("""import unittest
from pathlib import Path
class Retry(unittest.TestCase):
    def test_retry(self):
        marker = Path(__file__).parents[1] / 'output' / 'marker'
        previous = marker.exists()
        marker.touch()
        self.assertTrue(previous, 'intentional first-attempt failure')
""")
        args = argparse.Namespace(package=False, resume=None, pattern=None, rerun=[])
        with patch.object(verify, "ROOT", self.root), patch.object(
                verify.subprocess, "check_output", return_value="test revision"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(verify.run(args), 1)
            first = next((self.root / "output").glob("verification-*/report.json"))
            original = json.loads(first.read_text())
            self.assertEqual(original["status"], "fail")
            args.resume = str(first)
            self.assertEqual(verify.run(args), 0)
            reports = list((self.root / "output").glob("verification-*/report.json"))
            second = next(path for path in reports if path != first)
            reconciled = json.loads(second.read_text())
            self.assertEqual(reconciled["status"], "pass")
            self.assertIn("reused_from", reconciled["modules"]["checkout/test_pass"])
            self.assertNotIn("reused_from", reconciled["modules"]["checkout/test_retry"])
            (self.root / "tests/test_pass.py").write_text("# changed\n")
            self.assertEqual(verify.run(args), 1)
            third = next(path for path in (self.root / "output").glob("verification-*/report.json")
                         if path not in reports)
            rejected = json.loads(third.read_text())
            self.assertEqual(rejected["status"], "incomplete")
            self.assertIn("identity changed", rejected["error"])


if __name__ == "__main__":
    unittest.main()
