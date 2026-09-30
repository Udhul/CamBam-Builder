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
        output = RUNNER.parents[1] / "output"
        output.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="verification-runner-tests-", dir=output)
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

    def test_stopped_suite_is_incomplete_even_when_executed_test_passes(self):
        completed, result = self.execute("""import unittest
class Sample(unittest.TestCase):
    def test_a_stop(self): self._outcome.result.stop()
    def test_b_unexecuted(self): self.fail('must not disappear from evidence')
""")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["tests_run"], 1)
        self.assertEqual(result["completion"]["missing_ids"],
                         ["test_sample.Sample.test_b_unexecuted"])
        self.assertTrue(result["completion"]["stopped"])

    def test_suite_silently_omitting_selected_test_is_incomplete(self):
        completed, result = self.execute("""import unittest
class Sample(unittest.TestCase):
    def test_a(self): pass
    def test_b(self): self.fail('unexecuted')
class PartialSuite(unittest.TestSuite):
    def run(self, result, debug=False):
        self._tests[0](result)
        return result
def load_tests(loader, tests, pattern):
    return PartialSuite(loader.loadTestsFromTestCase(Sample))
""")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["completion"]["stopped"])

    def test_test_lifecycle_without_an_outcome_is_incomplete(self):
        completed, result = self.execute("""import unittest
class Sample(unittest.TestCase):
    def run(self, result):
        result.startTest(self)
        result.stopTest(self)
        return result
    def test_unexecuted(self): self.fail('body must not disappear')
""")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["tests"], [])
        self.assertEqual(result["completion"]["missing_ids"],
                         ["test_sample.Sample.test_unexecuted"])

    def test_class_and_module_fixture_skips_are_complete_named_skips(self):
        bodies = ("""import unittest
def setUpModule(): raise unittest.SkipTest('module observation absent')
class Sample(unittest.TestCase):
    def test_a(self): self.fail('skipped')
    def test_b(self): self.fail('skipped')
""", """import unittest
class Sample(unittest.TestCase):
    @classmethod
    def setUpClass(cls): raise unittest.SkipTest('class observation absent')
    def test_a(self): self.fail('skipped')
    def test_b(self): self.fail('skipped')
class Other(unittest.TestCase):
    def test_ok(self): pass
""")
        for body in bodies:
            with self.subTest(body=body):
                completed, result = self.execute(body)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(result["status"], "pass")
                self.assertTrue(result["completion"]["complete"])
                self.assertEqual(result["completion"]["missing_ids"], [])
                self.assertEqual(len(result["completion"]["fixture_skips"]), 1)
                self.assertTrue(any(t["status"] == "skip" and 'observation absent' in t["detail"]
                                    for t in result["tests"]))

    def test_repeated_id_requires_an_outcome_for_each_execution(self):
        completed, result = self.execute("""import unittest
class Sample(unittest.TestCase):
    calls = 0
    def run(self, result):
        self.calls += 1
        if self.calls == 1: return super().run(result)
        result.startTest(self)
        result.stopTest(self)
        return result
    def test_ok(self): pass
def load_tests(loader, tests, pattern):
    test = Sample('test_ok')
    return unittest.TestSuite([test, test])
""")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["tests_run"], 2)
        self.assertEqual(len(result["completion"]["completed_ids"]), 1)
        self.assertEqual(result["completion"]["missing_ids"], ["test_sample.Sample.test_ok"])

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

    def prepare_checkout(self):
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
            (self.root / name).touch()
        (self.root / "demos").mkdir()
        (self.root / "demos/mcp_client_acceptance_verify.py").touch()

    def run_checkout(self, resume=None):
        before = set((self.root / "output").glob("verification-*/report.json"))
        args = argparse.Namespace(package=False, resume=resume, pattern=None, rerun=[])
        class SyntheticCommands(verify.Commands):
            def __call__(self, label, command, cwd, env=None, *, check=True):
                if "_probe" in command:
                    # These fault witnesses use one fixed environment. Real
                    # dependency/provenance probes run in checkout/package gates;
                    # repeatedly hashing the entire environment obscures this test.
                    verify.write_json(command[-1], {"executable": sys.executable})
                    return {"returncode": 0, "status": "pass"}
                return super().__call__(label, command, cwd, env, check=check)
        with patch.object(verify, "ROOT", self.root), patch.object(
                verify.subprocess, "check_output", return_value="test revision"), patch.object(
                verify, "Commands", SyntheticCommands), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = verify.run(args)
        path, = set((self.root / "output").glob("verification-*/report.json")) - before
        return code, path, json.loads(path.read_text())

    def test_stopped_suite_cannot_pass_or_be_reused_on_resume(self):
        self.prepare_checkout()
        (self.root / "tests/test_sample.py").write_text("""import unittest
class Sample(unittest.TestCase):
    def test_a_stop(self): self._outcome.result.stop()
    def test_b_unexecuted(self): self.fail('unexecuted')
""", encoding="utf-8")
        code, path, first = self.run_checkout()
        self.assertEqual(code, 1)
        self.assertEqual(first["status"], "incomplete")
        code, _, second = self.run_checkout(str(path))
        self.assertEqual(code, 1)
        self.assertEqual(second["status"], "incomplete")
        self.assertNotIn("reused_from", second["modules"]["checkout/test_sample"])

    def test_resume_observation_changes_rerun_and_unrelated_history_does_not(self):
        self.prepare_checkout()
        witness = self.root / "output/witness.txt"
        witness.parent.mkdir()
        (self.root / "tests/test_sample.py").write_text("""import unittest
from pathlib import Path
class Sample(unittest.TestCase):
    def test_observation(self):
        witness = Path(__file__).parents[1] / 'output/witness.txt'
        if not witness.is_file(): self.skipTest('observation absent')
        self.assertEqual(witness.read_text(), 'good')
""", encoding="utf-8")
        with patch.dict(verify.OBSERVATION_INPUTS, {"test_sample": ("output/witness.txt",)}):
            code, absent, first = self.run_checkout()
            self.assertEqual(code, 0)
            self.assertEqual(len(first["summary"]["named_skips"]), 1)
            witness.write_text('good')
            code, present, second = self.run_checkout(str(absent))
            self.assertEqual(code, 0)
            self.assertEqual(second["summary"]["named_skips"], [])
            self.assertNotIn("reused_from", second["modules"]["checkout/test_sample"])
            (self.root / "output/unrelated.txt").write_text('history')
            code, _, unchanged = self.run_checkout(str(present))
            self.assertEqual(code, 0)
            self.assertIn("reused_from", unchanged["modules"]["checkout/test_sample"])
            witness.write_text('changed')
            code, _, changed = self.run_checkout(str(present))
            self.assertEqual(code, 1)
            self.assertEqual(changed["status"], "fail")
            self.assertNotIn("reused_from", changed["modules"]["checkout/test_sample"])
            witness.unlink()
            code, _, removed = self.run_checkout(str(present))
            self.assertEqual(code, 0)
            self.assertEqual(len(removed["summary"]["named_skips"]), 1)
            self.assertNotIn("reused_from", removed["modules"]["checkout/test_sample"])

    def test_observation_mutation_during_execution_is_incomplete(self):
        self.prepare_checkout()
        (self.root / "output").mkdir()
        (self.root / "output/witness.txt").write_text('good')
        (self.root / "tests/test_sample.py").write_text("""import unittest
from pathlib import Path
class Sample(unittest.TestCase):
    def test_mutate(self):
        (Path(__file__).parents[1] / 'output/witness.txt').write_text('changed')
""", encoding="utf-8")
        with patch.dict(verify.OBSERVATION_INPUTS, {"test_sample": ("output/witness.txt",)}):
            code, _, report = self.run_checkout()
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["modules"]["checkout/test_sample"]["status"], "incomplete")

    def test_resume_reuses_pass_and_reruns_failure_then_rejects_changed_source(self):
        self.prepare_checkout()
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
        code, first, original = self.run_checkout()
        self.assertEqual(code, 1)
        self.assertEqual(original["status"], "fail")
        code, _, reconciled = self.run_checkout(str(first))
        self.assertEqual(code, 0)
        self.assertEqual(reconciled["status"], "pass")
        self.assertIn("reused_from", reconciled["modules"]["checkout/test_pass"])
        self.assertNotIn("reused_from", reconciled["modules"]["checkout/test_retry"])
        (self.root / "tests/test_pass.py").write_text("# changed\n")
        code, _, rejected = self.run_checkout(str(first))
        self.assertEqual(code, 1)
        self.assertEqual(rejected["status"], "incomplete")
        self.assertIn("identity changed", rejected["error"])


if __name__ == "__main__":
    unittest.main()
