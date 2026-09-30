"""Build and inspect clean package targets for the repository verification runner.

The module has no third-party dependencies so it can also run from a test snapshot
against a freshly installed wheel or sdist, outside the source checkout.
"""

from __future__ import annotations

import argparse
from importlib import metadata, resources
import importlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile


PACKAGES = ("cambam_builder", "legacy_cambam_builder")
ASSETS = (
    "cambam_builder/mcp_adapter/contract_v1.schema.json",
    "cambam_builder/mcp_adapter/consumer_AGENTS.template.md",
)
SDIST_ROOT_FILES = ("LICENSE", "README.md", "pyproject.toml", "MANIFEST.in")
DEMO = "demos/mcp_client_acceptance_verify.py"
RUNNERS = ("tools/verify.py", "tools/verification_package.py")
FORBIDDEN_PARTS = {"__pycache__", "output", ".venv", ".git", ".pytest_cache"}
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".cb.bak")


def _tracked(root: Path, *paths: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "--", *paths], cwd=root, check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(p.decode("utf-8") for p in result.stdout.split(b"\0") if p)


def _matrix(root: Path) -> tuple[str, ...]:
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    versions = config["tool"]["cambam-verification"]["python"]
    if not isinstance(versions, list) or len(versions) < 2:
        raise ValueError("tool.cambam-verification.python must declare the package matrix")
    if any(not isinstance(v, str) or not v.startswith("3.") for v in versions):
        raise ValueError("invalid package matrix interpreter")
    if len(versions) != len(set(versions)):
        raise ValueError("duplicate package matrix interpreter")
    return tuple(versions)


def _archive_members(archive: Path, *, source: bool) -> dict[str, str]:
    """Map logical paths to archive members, rejecting unsafe or disposable entries."""
    if source:
        with tarfile.open(archive, "r:gz") as opened:
            names = [member.name for member in opened.getmembers() if member.isfile()]
    else:
        with zipfile.ZipFile(archive) as opened:
            names = [member.filename for member in opened.infolist() if not member.is_dir()]
    result: dict[str, str] = {}
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {name}")
        logical = PurePosixPath(*path.parts[1:]) if source else path
        if not logical.parts:
            continue
        if FORBIDDEN_PARTS.intersection(logical.parts) or logical.name.endswith(FORBIDDEN_SUFFIXES):
            raise ValueError(f"disposable file included in package: {name}")
        key = logical.as_posix()
        if key in result:
            raise ValueError(f"duplicate archive member: {key}")
        result[key] = name
    return result


def inspect_archives(root: Path, wheel: Path, sdist: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Check declared modules, assets, distributed tests and source metadata."""
    expected_runtime = [
        p for p in _tracked(root, *PACKAGES)
        if p.endswith(".py") or p in ASSETS
    ]
    if not expected_runtime or not all(a in expected_runtime for a in ASSETS):
        raise ValueError("missing tracked runtime modules or package assets")
    expected_tests = _tracked(root, "tests")
    if not any(p.startswith("tests/test_") and p.endswith(".py") for p in expected_tests):
        raise ValueError("no tracked regression tests")
    if not all((root / p).is_file() for p in (*expected_runtime, *expected_tests,
                                            *RUNNERS, DEMO)):
        raise ValueError("tracked package or regression input missing from checkout")
    wheel_members = _archive_members(wheel, source=False)
    source_members = _archive_members(sdist, source=True)
    wheel_missing = sorted(set(expected_runtime) - wheel_members.keys())
    source_missing = sorted(
        (set(expected_runtime) | set(expected_tests) | set(SDIST_ROOT_FILES)
         | set(RUNNERS) | {DEMO})
        - source_members.keys()
    )
    if wheel_missing or source_missing:
        raise ValueError(
            f"incomplete package archives: wheel={wheel_missing}, sdist={source_missing}"
        )
    if any(p.startswith("tests/") for p in wheel_members):
        raise ValueError("wheel unexpectedly contains regression tests")
    if not any(p.endswith(".dist-info/METADATA") for p in wheel_members):
        raise ValueError("wheel metadata missing")
    if not any(p.endswith(".dist-info/licenses/LICENSE") or p.endswith(".dist-info/LICENSE")
               for p in wheel_members):
        raise ValueError("wheel license missing")
    if not any(p.endswith(".egg-info/PKG-INFO") for p in source_members):
        raise ValueError("sdist metadata missing")
    tracked_tests = set(expected_tests)
    for label, members in (("wheel", wheel_members), ("sdist", source_members)):
        unexpected_cam = sorted(
            path for path in members
            if path.lower().endswith((".cb", ".nc")) and path not in tracked_tests
        )
        if unexpected_cam:
            raise ValueError(f"{label} contains untracked CAM files: {unexpected_cam}")
    unexpected_test_data = sorted(
        path for path in source_members
        if path.startswith("tests/") and path not in tracked_tests
    )
    if unexpected_test_data:
        raise ValueError(f"sdist contains untracked test files: {unexpected_test_data}")
    _assert_archive_bytes(root, wheel, wheel_members, expected_runtime, source=False)
    _assert_archive_bytes(
        root, sdist, source_members,
        [*expected_runtime, *expected_tests, *SDIST_ROOT_FILES, DEMO, *RUNNERS],
        source=True,
    )
    return wheel_members, source_members


def _assert_archive_bytes(root: Path, archive: Path, members: dict[str, str],
                          paths: list[str] | tuple[str, ...], *, source: bool) -> None:
    if source:
        with tarfile.open(archive, "r:gz") as opened:
            for relative in paths:
                stream = opened.extractfile(members[relative])
                if stream is None or stream.read() != (root / relative).read_bytes():
                    raise ValueError(f"sdist has stale or changed bytes: {relative}")
    else:
        with zipfile.ZipFile(archive) as opened:
            for relative in paths:
                if opened.read(members[relative]) != (root / relative).read_bytes():
                    raise ValueError(f"wheel has stale or changed bytes: {relative}")


def _copy_source_tests(root: Path, snapshot: Path) -> None:
    for relative in _tracked(root, "tests", DEMO):
        if not (relative.startswith("tests/") or relative == DEMO):
            continue
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)


def _copy_distributed_tests(sdist: Path, members: dict[str, str], snapshot: Path) -> None:
    with tarfile.open(sdist, "r:gz") as opened:
        for logical, archive_name in members.items():
            if not (logical.startswith("tests/") or logical == DEMO or logical in RUNNERS):
                continue
            target = snapshot / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            source = opened.extractfile(archive_name)
            if source is None:
                raise ValueError(f"cannot extract distributed test file: {logical}")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _write_import_guard(snapshot: Path) -> None:
    # Python loads sitecustomize in child processes when PYTHONPATH contains this
    # snapshot. Check the package resolver before every first package import.
    (snapshot / "sitecustomize.py").write_text(
        """from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
from pathlib import Path
import site
import sys

_roots = tuple(Path(p).resolve() for p in site.getsitepackages())
_packages = {'cambam_builder', 'legacy_cambam_builder'}

class _InstalledPackageGuard(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] not in _packages:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        paths = ([spec.origin] if spec.origin and spec.origin != 'namespace' else
                 list(spec.submodule_search_locations or ()))
        for candidate in paths:
            resolved = Path(candidate).resolve()
            if not any(resolved.is_relative_to(root) for root in _roots):
                raise ImportError(f'package import outside installed site-packages: {fullname}: {resolved}')
        return None

sys.meta_path.insert(0, _InstalledPackageGuard())
""", encoding="utf-8",
    )


def _snapshot(root: Path, directory: Path, *, sdist: Path | None = None,
              members: dict[str, str] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    if sdist is None:
        _copy_source_tests(root, directory)
    else:
        assert members is not None
        _copy_distributed_tests(sdist, members, directory)
    if sdist is None:
        for relative in RUNNERS:
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, destination)
    (directory / "output").mkdir()
    _write_import_guard(directory)
    return directory


def _python(venv: Path) -> Path:
    windows = venv / "Scripts" / "python.exe"
    return windows if windows.exists() else venv / "bin" / "python"


def _external_cwd(name: str) -> Path:
    cwd = Path(tempfile.mkdtemp(prefix=f"cambam-{name}-")).resolve()
    (cwd / "output").mkdir()
    return cwd


def prepare(root: Path, output: Path, run_command) -> list[dict]:
    """Build archives and return clean wheel, source and base-only test targets.

    ``run_command(label, command, cwd, env=None)`` must retain its own logs and
    raise on failure. The caller runs smoke and unittest against returned targets.
    """
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    versions = _matrix(root)
    dist = output / "dist"
    dist.mkdir(exist_ok=False)
    run_command("package-build", ["uv", "build", "--out-dir", str(dist)], root)
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"expected one wheel and one sdist, got {wheels}, {sdists}")
    wheel, sdist = wheels[0].resolve(), sdists[0].resolve()
    _, source_members = inspect_archives(root, wheel, sdist)
    targets: list[dict] = []
    for version in versions:
        name = f"wheel-py{version}"
        env_path = output / "envs" / name
        run_command(f"{name}-venv", ["uv", "venv", "--python", version, str(env_path)], root)
        python = _python(env_path).resolve()
        run_command(
            f"{name}-install",
            ["uv", "pip", "install", "--python", str(python), f"{wheel}[planar,mcp]"], root,
        )
        snapshot = _snapshot(root, output / "snapshots" / name)
        cwd = _external_cwd(name)
        targets.append(dict(name=name, python=python, snapshot=snapshot, cwd=cwd,
                            installed=True, base_only=False, artifact=wheel,
                            environment=env_path, dist=dist, smoke_kind="full"))
    source_version = versions[0]
    name = f"sdist-py{source_version}"
    env_path = output / "envs" / name
    run_command(f"{name}-venv", ["uv", "venv", "--python", source_version, str(env_path)], root)
    python = _python(env_path).resolve()
    run_command(
        f"{name}-install", ["uv", "pip", "install", "--python", str(python),
                             f"{sdist}[planar,mcp]"], root,
    )
    snapshot = _snapshot(root, output / "snapshots" / name,
                         sdist=sdist, members=source_members)
    cwd = _external_cwd(name)
    targets.append(dict(name=name, python=python, snapshot=snapshot, cwd=cwd,
                        installed=True, base_only=False, artifact=sdist,
                        environment=env_path, dist=dist, smoke_kind="full"))
    name = f"base-py{source_version}"
    env_path = output / "envs" / name
    run_command(f"{name}-venv", ["uv", "venv", "--python", source_version, str(env_path)], root)
    python = _python(env_path).resolve()
    run_command(
        f"{name}-install", ["uv", "pip", "install", "--python", str(python), str(wheel)], root,
    )
    snapshot = _snapshot(root, output / "snapshots" / name)
    cwd = _external_cwd(name)
    targets.append(dict(name=name, python=python, snapshot=snapshot, cwd=cwd,
                        installed=True, base_only=True, artifact=wheel,
                        environment=env_path, dist=dist, smoke_kind="base"))
    return targets


def _assert_installed(module) -> str:
    origin = getattr(module, "__file__", None)
    if origin is None:
        search = getattr(module, "__path__", ())
        if len(search) != 1:
            raise AssertionError(f"ambiguous installed package path: {module.__name__}: {search}")
        origin = next(iter(search))
    path = Path(origin).resolve()
    roots = [Path(p).resolve() for p in site.getsitepackages()]
    if not any(path.is_relative_to(p) for p in roots):
        raise AssertionError(f"module imported outside installed site-packages: {module.__name__}: {path}")
    return str(path)


def smoke(kind: str) -> dict:
    """Exercise package imports, metadata, native XML and optional boundaries."""
    import cambam_builder
    import legacy_cambam_builder
    import numpy
    from cambam_builder.native.reader import read_cambam_bytes
    from cambam_builder.native.writer import serialize_cambam_bytes
    from cambam_builder.planar import (Circle, ErrorBudget, PlanarFrame,
                                      RegionSet, feasible_centers, normalize)

    imports = {}
    if kind == "base":
        module_names = (
            "cambam_builder", "cambam_builder.native", "cambam_builder.native.project",
            "cambam_builder.native.reader", "cambam_builder.native.writer",
            "cambam_builder.planar", "legacy_cambam_builder",
            "legacy_cambam_builder.legacy_cambam_project",
        )
    else:
        names = []
        for package in PACKAGES:
            module = importlib.import_module(package)
            names.append(package)
            for path in sorted(Path(module.__file__).parent.rglob("*.py")):
                relative = path.relative_to(Path(module.__file__).parent)
                if relative.name == "__init__.py":
                    relative = relative.parent
                else:
                    relative = relative.with_suffix("")
                name = ".".join((package, *relative.parts)).rstrip(".")
                if name != "legacy_cambam_builder.cambam_builder_cli":
                    names.append(name)
        module_names = tuple(dict.fromkeys(names))
    for name in module_names:
        imports[name] = _assert_installed(importlib.import_module(name))

    project = cambam_builder.CBProject("package-smoke")
    layer = project.add_layer("Geometry")
    project.add_rect(layer, identifier="outline", width=10, height=5)
    xml = serialize_cambam_bytes(project)
    assert read_cambam_bytes(xml).get_primitive("outline") is not None
    assert legacy_cambam_builder.LegacyCBProject("legacy-package-smoke").name == "legacy-package-smoke"
    frame = PlanarFrame("mm", "package-smoke", (0, 0), 0)
    budget = ErrorBudget(.001, .01)
    assert feasible_centers(Circle((0, 0), 2), frame, 1, "mm", budget).status == "ok"
    versions = {"python": sys.version.split()[0], "numpy": numpy.__version__,
                "cambam-builder": metadata.version("cambam-builder")}
    assert versions["cambam-builder"] == cambam_builder.__version__
    distribution = metadata.distribution("cambam-builder")
    assert distribution.metadata["Requires-Python"].startswith(">=3.12")
    requirements = tuple(distribution.requires or ())
    assert any(requirement.startswith("numpy") for requirement in requirements)
    assert any(requirement.startswith("shapely") for requirement in requirements)
    assert any(requirement.startswith("mcp") for requirement in requirements)
    entry_points = {entry.name: entry for entry in distribution.entry_points}
    assert "cambam-mcp" in entry_points
    assert callable(entry_points["cambam-mcp"].load())
    for relative in ASSETS:
        package, *parts = relative.split("/")
        resource = resources.files("cambam_builder.mcp_adapter").joinpath(*parts[1:])
        assert resource.read_bytes(), relative
    if kind == "full":
        import mcp
        import shapely
        imports["mcp"] = _assert_installed(mcp)
        imports["shapely"] = _assert_installed(shapely)
        versions["mcp"] = metadata.version("mcp")
        versions["shapely"] = shapely.__version__
        versions["GEOS"] = shapely.geos_version_string
        assert normalize(RegionSet(frame, (Circle((0, 0), 2),)), budget).status == "ok"
    else:
        assert importlib.util.find_spec("shapely") is None
        assert importlib.util.find_spec("mcp") is None
        assert normalize(RegionSet(frame, (Circle((0, 0), 2),)), budget).status == "unsupported"
        try:
            importlib.import_module("cambam_builder.cam_core.v_region")
        except ModuleNotFoundError as exc:
            assert exc.name == "shapely", str(exc)
        else:
            raise AssertionError("backend-dependent CAM import unexpectedly succeeded")
    return {"kind": kind, "versions": versions, "imports": imports,
            "xml_bytes": len(xml), "resources": list(ASSETS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke",))
    parser.add_argument("--kind", choices=("full", "base"), default="full")
    args = parser.parse_args()
    print(json.dumps(smoke(args.kind), sort_keys=True))


if __name__ == "__main__":
    main()
