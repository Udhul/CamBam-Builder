"""Auditable unittest and installed-package verification. See docs/DEVELOPMENT.md."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 2

# Retained observations are outside source_identity and fresh package snapshots.
# Keep this bounded list aligned with the two observation tests, not output history.
OBSERVATION_INPUTS = {
    "test_tabbed_cutout": tuple("output/tabbed-cutout-20260927-04/fixtures/" + name
        for name in ("B-fresh-manual.cb", "B-fresh-manual.nc",
                     "C-fresh-manual.cb", "C-fresh-manual.nc")),
    "test_native_series": ("output/m1-polygon-20260924-04/source.cb",
        "output/m1-polygon-20260924-04/native/m1-native.cb",
        "output/m1-polygon-20260924-04/native/m1-native.nc"),
}


def observation_identity(snapshot, module):
    return {name: digest(snapshot / name) if (snapshot / name).is_file() else None
            for name in OBSERVATION_INPUTS.get(module, ())}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_identity(root):
    """Conservative identity: all runtime/tests/tools, including untracked inputs."""
    files = {}
    for directory in ("cambam_builder", "legacy_cambam_builder", "tests", "tools"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[path.relative_to(root).as_posix()] = digest(path)
    for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE",
                 "demos/mcp_client_acceptance_verify.py"):
        files[name] = digest(root / name)
    return files


def environment_identity():
    distributions = {}
    dependency_hashes = {}
    prefix = Path(sys.prefix).resolve()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        distributions[name] = dist.version
        files = {}
        for relative in dist.files or []:
            # Resolve the environment once, not every file: on Windows repeated
            # realpath traversal dominates hashing the small dependency files.
            path = Path(os.path.abspath(dist.locate_file(relative)))
            if path.is_relative_to(prefix) and path.is_file() and path.suffix != ".pyc":
                files[str(relative)] = digest(path)
        dependency_hashes[name] = hashlib.sha256(
            json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {"executable": str(Path(sys.executable).resolve()), "python": sys.version,
            "executable_sha256": digest(sys.executable), "platform": platform.platform(),
            "dependencies": distributions, "dependency_hashes": dependency_hashes}


def target_fingerprint(target, probe):
    identity = json.loads(probe.read_text(encoding="utf-8"))
    if target["installed"]:
        snapshot = Path(target["snapshot"])
        identity["snapshot"] = {p.relative_to(snapshot).as_posix(): digest(p)
            for p in sorted(snapshot.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and "output" not in p.relative_to(snapshot).parts
            and p.suffix != ".pyc"}
        identity["artifacts"] = {str(p): digest(p) for p in sorted(
            Path(target["dist"]).glob("*")) if p.is_file()}
        if not identity["artifacts"]:
            raise ValueError("package artifacts are missing")
    return identity


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.completed_ids = []
        self.fixture_skips = []
        self.outcome_ids = set()

    def startTest(self, test):
        super().startTest(test)
        self.outcome_ids.discard(test.id())

    def stopTest(self, test):
        super().stopTest(test)
        if test.id() in self.outcome_ids:
            self.completed_ids.append(test.id())
        self.outcome_ids.discard(test.id())

    def record(self, test, status, detail=None):
        self.records.append({"id": test.id(), "status": status, "detail": detail})
        self.outcome_ids.add(test.id())

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, "pass")

    def addError(self, test, err):
        super().addError(test, err)
        self.record(test, "error", self._exc_info_to_string(err, test))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record(test, "fail", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, "skip", reason)
        # unittest represents setUpModule/setUpClass skips as _ErrorHolder,
        # without startTest/stopTest. Ordinary per-test skips complete normally.
        if not isinstance(test, unittest.TestCase):
            self.fixture_skips.append(test.id())

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.record(test, "expected_failure", self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.record(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.record(subtest, "fail", self._exc_info_to_string(err, test))
            self.outcome_ids.add(test.id())


def worker(snapshot, module, destination):
    start, cpu = time.perf_counter(), time.process_time()
    report = {"status": "incomplete", "module": module, "tests": []}
    write_json(destination, report)
    try:
        sys.path.insert(0, str(snapshot / "tests"))
        sys.path.insert(0, str(snapshot))
        loader = unittest.TestLoader()
        suite = loader.discover(str(snapshot / "tests"), pattern=module + ".py")
        selected = list(flatten(suite))
        ids = [test.id() for test in selected]
        scopes = {test.id(): (f"setUpModule ({type(test).__module__})",
                             f"setUpClass ({type(test).__module__}.{type(test).__qualname__})")
                  for test in selected}
        result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
        accounted = Counter(result.completed_ids)
        for test_id in ids:
            if any(scope in result.fixture_skips for scope in scopes[test_id]):
                accounted[test_id] += 1
        missing = list((Counter(ids) - accounted).elements())
        complete = bool(ids) and not result.shouldStop and not missing
        status = "fail" if not result.wasSuccessful() or loader.errors or not ids else (
            "pass" if complete else "incomplete")
        report.update(test_ids=ids, tests=result.records, tests_run=result.testsRun,
                      collection_errors=loader.errors,
                      completion={"complete": complete, "stopped": result.shouldStop,
                                  "completed_ids": result.completed_ids,
                                  "fixture_skips": result.fixture_skips, "missing_ids": missing},
                      status=status)
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report.update(wall_seconds=time.perf_counter() - start,
                      cpu_seconds=time.process_time() - cpu)
        write_json(destination, report)
    return 0 if report["status"] == "pass" else 1


class Commands:
    def __init__(self, output):
        self.output = output
        self.records = []

    def __call__(self, label, command, cwd, env=None, *, check=True):
        index = len(self.records)
        stem = f"command-{index:04d}"
        record = {"label": label, "command": list(map(str, command)), "cwd": str(cwd),
                  "log": stem + ".log", "status": "incomplete", "returncode": None}
        record["environment"] = {key: env[key] for key in
            ("PYTHONPATH", "PYTHONNOUSERSITE", "TMP", "TEMP", "TMPDIR") if env and key in env}
        self.records.append(record)
        record_path = self.output / (stem + ".json")
        write_json(record_path, record)
        start = time.perf_counter()
        try:
            with (self.output / record["log"]).open("wb") as log:
                process = subprocess.Popen(record["command"], cwd=cwd, env=env, stdout=log,
                                           stderr=subprocess.STDOUT)
                try:
                    record["returncode"] = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            record["status"] = "pass" if record["returncode"] == 0 else "fail"
        finally:
            record["wall_seconds"] = time.perf_counter() - start
            # Portable subprocess CPU accounting is unavailable here. Worker reports
            # its own CPU; never label parent CPU as child execution time.
            record["cpu_seconds"] = None
            write_json(record_path, record)
        if check and record["status"] != "pass":
            raise RuntimeError(f"{label} failed: {self.output / record['log']}")
        return record


def clean_env(snapshot, temporary):
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONPATH"] = str(snapshot)
    env["PYTHONNOUSERSITE"] = "1"
    env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(temporary)
    return env


def modules_in(snapshot, patterns):
    return sorted({p.stem for pattern in patterns for p in (snapshot / "tests").glob(pattern)
                   if p.is_file() and p.suffix == ".py"})


def load_report(path):
    path = Path(path).resolve()
    if path.is_dir():
        path /= "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA:
        raise ValueError("unsupported report schema")
    return path, report


def reusable(prior, identity, target_identities, inventory):
    if prior["identity"] != identity:
        raise ValueError("reconciliation refused: source/configuration identity changed")
    if prior["target_identities"] != target_identities:
        raise ValueError("reconciliation refused: interpreter/dependency/artifact identity changed")
    if prior["inventory"] != inventory:
        raise ValueError("reconciliation refused: test inventory changed")


def summarize(modules):
    return {
        "modules": len(modules),
        "reused_modules": sum("reused_from" in value for value in modules.values()),
        "tests_run": sum(value.get("tests_run", 0) for value in modules.values()),
        "named_skips": [{"module": key, **test} for key, value in modules.items()
                        for test in value.get("tests", []) if test["status"] == "skip"],
        "slowest_modules": sorted(
            [{"module": key, "wall_seconds": value.get("wall_seconds"),
              "cpu_seconds": value.get("cpu_seconds"), "reused": "reused_from" in value}
             for key, value in modules.items() if value.get("wall_seconds") is not None],
            key=lambda row: row["wall_seconds"], reverse=True)[:10],
    }


def run(args):
    root = ROOT
    output = root / "output" / ("verification-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    temporary = output / "tmp"
    temporary.mkdir()
    print(f"Report: {output / 'report.json'}", flush=True)
    report = {"schema": SCHEMA, "status": "incomplete", "modules": {}, "output": str(output),
              "command": sys.argv, "started": time.time(), "mode": "package" if args.package else "checkout"}
    commands = Commands(output)
    start = time.perf_counter()
    prior = None
    try:
        report["identity"] = source_identity(root)
        report["git"] = {key: subprocess.check_output(command, cwd=root, text=True).strip()
                         for key, command in {"commit": ["git", "rev-parse", "HEAD"],
                                              "worktree": ["git", "status", "--short"]}.items()}
        write_json(output / "report.json", report)
        if args.resume:
            prior_path, prior = load_report(args.resume)
            if prior["mode"] != report["mode"]:
                raise ValueError("resume mode must match original run")
            report["reconciled_from"] = str(prior_path)
        if args.package:
            import verification_package
            if prior:
                targets = prior["targets"]
            else:
                targets = verification_package.prepare(root, output, commands)
        else:
            targets = [{"name": "checkout", "python": sys.executable, "snapshot": root,
                        "cwd": root, "installed": False, "base_only": False}]
        targets = [{k: str(v) if isinstance(v, Path) else v for k, v in target.items()} for target in targets]
        report["targets"] = targets
        report["target_identities"] = {}
        report["inventory"] = {}
        patterns = args.pattern or (prior.get("patterns") if prior else None) or ["test_*.py"]
        report["patterns"] = patterns
        for target in targets:
            name = target["name"]
            snapshot = Path(target["snapshot"])
            env = clean_env(snapshot, temporary)
            probe = output / (name + "-environment.json")
            commands(name + " environment", [target["python"], str(Path(__file__).resolve()),
                     "_probe", str(probe)], Path(target["cwd"]), env)
            target_identity = target_fingerprint(target, probe)
            if target["installed"]:
                commands(name + " installed smoke", [target["python"], str(root / "tools/verification_package.py"),
                         "smoke", "--kind", "base" if target["base_only"] else "full"],
                         Path(target["cwd"]), env)
            report["target_identities"][name] = target_identity
            report["inventory"][name] = [] if target["base_only"] else modules_in(snapshot, patterns)
            if not target["base_only"] and not report["inventory"][name]:
                raise ValueError(f"no tests selected for {name}")
        if prior:
            reusable(prior, report["identity"], report["target_identities"], report["inventory"])
        unknown_reruns = set(args.rerun) - {m for modules in report["inventory"].values() for m in modules}
        if unknown_reruns:
            raise ValueError(f"unknown rerun modules: {sorted(unknown_reruns)}")
        # Persist every planned unit before starting any expensive test.
        for target, modules in report["inventory"].items():
            for module in modules:
                key = target + "/" + module
                report["modules"][key] = {"status": "incomplete"}
        write_json(output / "report.json", report)
        for target in targets:
            name = target["name"]
            for module in report["inventory"][name]:
                key = name + "/" + module
                old = prior.get("modules", {}).get(key, {}) if prior else {}
                inputs = observation_identity(Path(target["snapshot"]), module)
                if (old.get("status") == "pass" and module not in args.rerun
                        and old.get("observation_inputs") == inputs):
                    result_path = Path(old["result"])
                    if digest(result_path) != old["result_sha256"]:
                        raise ValueError(f"reused evidence was modified: {key}")
                    retained = json.loads(result_path.read_text(encoding="utf-8"))
                    if retained.get("status") != "pass" or not retained.get("completion", {}).get("complete"):
                        raise ValueError(f"reused evidence lacks suite completion: {key}")
                    report["modules"][key] = {**old, "reused_from": str(prior_path)}
                else:
                    result_path = output / (name + "-" + module + ".json")
                    print(f"Running {key}", flush=True)
                    command = commands(key, [target["python"], str(Path(__file__).resolve()), "_worker",
                        target["snapshot"], module, str(result_path)], Path(target["cwd"]),
                        clean_env(Path(target["snapshot"]), temporary), check=False)
                    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
                    status = result.get("status", "incomplete")
                    if status == "pass" and (command["returncode"] != 0
                            or not result.get("completion", {}).get("complete")):
                        status = "incomplete"
                    if inputs != observation_identity(Path(target["snapshot"]), module):
                        status = "incomplete"
                    report["modules"][key] = {"status": status, "result": str(result_path),
                        "observation_inputs": inputs, "completion": result.get("completion"),
                        "result_sha256": digest(result_path) if result_path.exists() else None,
                        "command": command, "wall_seconds": result.get("wall_seconds"),
                        "cpu_seconds": result.get("cpu_seconds"), "tests_run": result.get("tests_run", 0),
                        "tests": result.get("tests", []), "test_ids": result.get("test_ids", [])}
                write_json(output / "report.json", report)
        if report["identity"] != source_identity(root):
            raise ValueError("source/configuration changed during execution; rerun required")
        for target in targets:
            for module in report["inventory"][target["name"]]:
                key = target["name"] + "/" + module
                if report["modules"][key]["observation_inputs"] != observation_identity(
                        Path(target["snapshot"]), module):
                    report["modules"][key]["status"] = "incomplete"
                    raise ValueError(f"observation inputs changed during execution: {key}")
            probe = output / (target["name"] + "-environment-final.json")
            commands(target["name"] + " final environment", [target["python"],
                     str(Path(__file__).resolve()), "_probe", str(probe)], Path(target["cwd"]),
                     clean_env(Path(target["snapshot"]), temporary))
            if target_fingerprint(target, probe) != report["target_identities"][target["name"]]:
                raise ValueError("interpreter/dependencies/snapshot/artifacts changed during execution")
        statuses = [r["status"] for r in report["modules"].values()]
        report["status"] = "incomplete" if "incomplete" in statuses else "fail" if "fail" in statuses else "pass"
    except BaseException:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
        report["status"] = "incomplete"
    finally:
        report["wall_seconds"] = time.perf_counter() - start
        report["commands"] = commands.records
        report["summary"] = summarize(report["modules"])
        write_json(output / "report.json", report)
    print(f"{report['status'].upper()}: {output / 'report.json'}", flush=True)
    return 0 if report["status"] == "pass" else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_probe":
        result = environment_identity()
        # Runtime provenance hashes exclude editable checkout sources, covered by
        # source_identity. Installed artifacts are independently fingerprinted.
        dist = importlib.metadata.distribution("cambam-builder")
        result["installed_files"] = {str(p): digest(dist.locate_file(p)) for p in dist.files or []
            if str(p).startswith(("cambam_builder/", "legacy_cambam_builder/"))
            and str(p).endswith((".py", ".json", ".md"))}
        write_json(sys.argv[2], result)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        return worker(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true", help="clean wheel/sdist matrix and base-only checks")
    parser.add_argument("--pattern", action="append", help="unittest module glob (repeatable)")
    parser.add_argument("--resume", help="explicitly reconcile a report; rerun failed/incomplete modules")
    parser.add_argument("--rerun", action="append", default=[], help="also rerun this passed module on resume")
    args = parser.parse_args()
    if args.rerun and not args.resume:
        parser.error("--rerun requires --resume")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
