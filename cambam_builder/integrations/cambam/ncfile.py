"""Explicit source snapshots and portable native NCFile document bundles.

Document import/export never reads NCFile dependencies. ``capture`` is the
opt-in boundary: it reads every referenced program inside a caller-selected
directory, including disabled operations. These are byte-integrity checks;
they do not parse G-code or establish machining or controller safety.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import uuid
import xml.etree.ElementTree as ET

from ...native.cam import NCFileMop
from ...native.project import CamBamProject
from ...native.reader import read_cambam_bytes
from ...native.writer import serialize_cambam_bytes


_FORMAT = "cambam-ncfile-bundle-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = {"format", "original_document_sha256", "document_sha256", "sources"}
_SOURCE_FIELDS = {"mop_id", "identifier", "original_reference", "path", "sha256", "size_bytes"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _directory(value: str | Path) -> Path:
    try:
        root = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Source directory is unavailable: {value}") from exc
    if not root.is_dir():
        raise ValueError(f"Source directory must be a directory: {value}")
    return root


def _read_within(root: Path, path: Path) -> bytes:
    """Resolve before opening so traversal and link/reparse escapes reject."""
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"NCFile source escapes the selected directory: {path}")
        if not resolved.is_file():
            raise ValueError(f"NCFile source must be a file: {path}")
        return resolved.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"NCFile source is unavailable: {path}") from exc


def _source_path(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference.strip() or "\x00" in reference:
        raise ValueError("NCFile capture requires a nonempty SourceFile")
    windows = PureWindowsPath(reference)
    if windows.drive and not windows.root:
        raise ValueError("Drive-relative NCFile source paths are unsupported")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", reference) and not windows.drive:
        raise ValueError("NCFile source must be a file path, not a URI")
    # A reference is native Windows data even on another host. Backslashes in
    # relative references are separators, never literal filename characters.
    path = Path(reference.replace("\\", "/"))
    if (windows.drive or path.root) and not path.is_absolute():
        raise ValueError("NCFile source requires a supported fully absolute or relative path")
    return path if path.is_absolute() else root / path


def _read_document(data: bytes) -> CamBamProject:
    """Admit a complete, unambiguous raw NCFile dependency collection.

    The general reader performs its XML safety/size checks first. Its legacy
    first-collection reconstruction is insufficient for dependency completeness,
    so this boundary also checks every raw occurrence before trusting the model.
    """
    project = read_cambam_bytes(data, strict=True)
    root = ET.fromstring(data)
    parts = root.findall("parts")
    if (root.tag != "CADFile" or len(parts) != 1 or
            list(root.iter("parts")) != parts):
        raise ValueError("NCFile document requires exactly one standard parts collection")
    part_elements = parts[0].findall("part")
    if list(root.iter("part")) != part_elements:
        raise ValueError("NCFile document has a nonstandard Part placement")
    collections = []
    raw_ncfiles = []
    for part in part_elements:
        machineops = part.findall("machineops")
        if len(machineops) > 1:
            raise ValueError("NCFile document has duplicate machineops collections")
        collections.extend(machineops)
        if machineops:
            raw_ncfiles.extend(machineops[0].findall("ncfile"))
    if list(root.iter("machineops")) != collections:
        raise ValueError("NCFile document has a nonstandard machineops placement")
    if list(root.iter("ncfile")) != raw_ncfiles:
        raise ValueError("NCFile document has a nonstandard NCFile placement")
    modeled_count = sum(isinstance(mop, NCFileMop) for mop in project.list_mops())
    if len(raw_ncfiles) != modeled_count:
        raise ValueError("NCFile document has unmodeled external-program references")
    return project


@dataclass(frozen=True)
class NCFileSource:
    """One operation's original source spelling and exact captured program bytes."""

    mop_id: str
    identifier: str
    original_reference: str
    data: bytes


@dataclass(frozen=True)
class NCFileBundle:
    """Immutable document/dependency snapshot produced by :func:`capture`.

    ``write_bundle`` uses the captured bytes even if originals later change.
    Call ``check_current`` immediately before writing when current-source
    freshness is required. This snapshot is not an NC execution certificate.
    """

    document_bytes: bytes
    sources: tuple[NCFileSource, ...]

    def _project(self) -> CamBamProject:
        if type(self.document_bytes) is not bytes or type(self.sources) is not tuple:
            raise ValueError("NCFile bundle requires immutable document and source values")
        project = _read_document(self.document_bytes)
        mops = {str(mop.internal_id): mop for mop in project.list_mops()
                if isinstance(mop, NCFileMop)}
        if not mops or len(mops) != len(self.sources):
            raise ValueError("NCFile bundle must contain every NCFile source exactly once")
        seen = set()
        for source in self.sources:
            if (not isinstance(source, NCFileSource) or type(source.data) is not bytes or
                    not isinstance(source.mop_id, str)):
                raise ValueError("NCFile bundle requires immutable source records")
            mop = mops.get(source.mop_id)
            if (mop is None or source.mop_id in seen or
                    source.identifier != mop.user_identifier or
                    source.original_reference != mop.source_file or
                    not isinstance(source.original_reference, str) or
                    not source.original_reference.strip()):
                raise ValueError("NCFile bundle source does not match its document operation")
            seen.add(source.mop_id)
        return project

    def check_current(self, project: CamBamProject, *, source_directory: str | Path) -> None:
        """Reject changed document intent or missing/changed source bytes.

        The comparison covers the complete serialized document, including
        disabled operations, names, modes, identities, order and units.
        """
        self._project()
        current = capture(project, source_directory=source_directory)
        if current != self:
            raise ValueError("NCFile document or source dependency changed after capture")

    def write_bundle(self, directory: str | Path) -> Path:
        """Write ``job.cb``, exact assets and ``manifest.json`` to a new directory.

        Programs use content-addressed relative paths, so the complete directory
        can be relocated. Existing destinations are never overwritten. All
        semantic validation precedes directory creation; an OS failure can leave
        an incomplete directory, with the manifest written last.
        """
        project = self._project()
        records = []
        assets: dict[str, bytes] = {}
        for source in self.sources:
            digest = _sha(source.data)
            relative = f"sources/{digest}.nc"
            project.get_mop(uuid.UUID(source.mop_id)).source_file = relative
            assets[relative] = source.data
            records.append({
                "mop_id": source.mop_id, "identifier": source.identifier,
                "original_reference": source.original_reference, "path": relative,
                "sha256": digest, "size_bytes": len(source.data),
            })
        document = serialize_cambam_bytes(project)
        manifest = {
            "format": _FORMAT, "original_document_sha256": _sha(self.document_bytes),
            "document_sha256": _sha(document), "sources": records,
        }
        encoded_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "sources").mkdir()
        for relative, data in assets.items():
            (destination / relative).write_bytes(data)
        document_path = destination / "job.cb"
        document_path.write_bytes(document)
        (destination / "manifest.json").write_bytes(encoded_manifest)
        return document_path


def capture(project: CamBamProject, *, source_directory: str | Path) -> NCFileBundle:
    """Snapshot all NCFile dependencies under an explicit existing directory.

    Relative ``SourceFile`` paths resolve from this directory, matching CamBam's
    document-directory rule. Absolute references are admitted only inside it.
    Traversal and symlink/reparse escapes reject. This operation reads files,
    but neither mutates the project nor parses or executes their contents.
    """
    if not isinstance(project, CamBamProject):
        raise TypeError("project must be a CamBamProject")
    root = _directory(source_directory)
    snapshot = project.clone()
    document = serialize_cambam_bytes(snapshot)
    sources = tuple(
        NCFileSource(str(mop.internal_id), mop.user_identifier, mop.source_file,
                     _read_within(root, _source_path(root, mop.source_file)))
        for mop in snapshot.list_mops() if isinstance(mop, NCFileMop)
    )
    bundle = NCFileBundle(document, sources)
    bundle._project()
    return bundle


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate NCFile manifest key: {key}")
        result[key] = value
    return result


def verify_bundle(directory: str | Path) -> dict[str, str | int]:
    """Check a written bundle's exact document, sources and operation references.

    Returns byte-integrity counts and hashes. A mutable manifest cannot prove
    provenance authenticity, motion correctness or controller compatibility.
    Unrelated extra files are ignored, allowing native saves/posts beside it.
    """
    root = _directory(directory)
    try:
        manifest = json.loads(_read_within(root, root / "manifest.json"),
                              object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid NCFile manifest JSON") from exc
    if (not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS or
            manifest.get("format") != _FORMAT or
            not isinstance(manifest.get("sources"), list)):
        raise ValueError("Unsupported NCFile manifest schema")
    for field in ("document_sha256", "original_document_sha256"):
        if not isinstance(manifest[field], str) or not _SHA256.fullmatch(manifest[field]):
            raise ValueError("Invalid NCFile manifest document hash")
    document = _read_within(root, root / "job.cb")
    if _sha(document) != manifest["document_sha256"]:
        raise ValueError("NCFile bundle document hash mismatch")
    project = _read_document(document)
    mops = {str(mop.internal_id): mop for mop in project.list_mops()
            if isinstance(mop, NCFileMop)}
    records = manifest["sources"]
    if not mops or len(records) != len(mops):
        raise ValueError("NCFile manifest must cover every NCFile operation exactly once")
    seen: set[str] = set()
    paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != _SOURCE_FIELDS:
            raise ValueError("Invalid NCFile manifest source record")
        if (any(not isinstance(record[field], str) for field in _SOURCE_FIELDS - {"size_bytes"}) or
                type(record["size_bytes"]) is not int or record["size_bytes"] < 0 or
                not _SHA256.fullmatch(record["sha256"]) or
                record["path"] != f"sources/{record['sha256']}.nc" or
                not record["original_reference"].strip()):
            raise ValueError("Invalid NCFile manifest source values or path")
        mop = mops.get(record["mop_id"])
        if (mop is None or record["mop_id"] in seen or
                mop.user_identifier != record["identifier"] or
                mop.source_file != record["path"]):
            raise ValueError("NCFile manifest operation identity/reference mismatch")
        data = _read_within(root, root / record["path"])
        if len(data) != record["size_bytes"] or _sha(data) != record["sha256"]:
            raise ValueError("NCFile bundle source hash/size mismatch")
        seen.add(record["mop_id"])
        paths.add(record["path"])
    return {"status": "verified", "document_path": str(root / "job.cb"),
            "document_sha256": _sha(document), "source_count": len(seen),
            "asset_count": len(paths)}
