"""Opt-in, exact-byte image snapshots for native CADBitmap reference artwork.

Ordinary document operations retain paths without accessing them. These bundles
cover bitmap dependencies only; they neither vectorize images nor certify CAM.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import uuid
import xml.etree.ElementTree as ET

from ...native.bitmap import CADBitmap
from ...native.project import CamBamProject
from ...native.reader import read_cambam_bytes
from ...native.writer import serialize_cambam_bytes


_FORMAT = "cambam-bitmap-bundle-v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SUFFIX = re.compile(r"\.[a-z0-9]{1,10}\Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _directory(value: str | Path) -> Path:
    try:
        root = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Bitmap directory is unavailable: {value}") from exc
    if not root.is_dir():
        raise ValueError("Bitmap source_directory must be a directory")
    return root


def _read_within(root: Path, path: Path) -> bytes:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("Bitmap dependency escapes its directory or is not a file")
        return resolved.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Bitmap dependency is unavailable: {path}") from exc


def _source_path(root: Path, reference: str) -> Path:
    windows = PureWindowsPath(reference)
    if windows.drive and not windows.root:
        raise ValueError("Drive-relative bitmap sources are unsupported")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", reference) and not windows.drive:
        raise ValueError("Bitmap source must be a file path, not a URI")
    path = Path(reference.replace("\\", "/"))
    if (windows.drive or path.root) and not path.is_absolute():
        raise ValueError("Unsupported absolute bitmap source path")
    return path if path.is_absolute() else root / path


def _read_document(data: bytes) -> CamBamProject:
    project = read_cambam_bytes(data, strict=True)
    root = ET.fromstring(data)
    layers = root.findall("layers")
    if len(layers) != 1 or list(root.iter("layers")) != layers:
        raise ValueError("Bitmap document requires one standard layers collection")
    layer_elements = layers[0].findall("layer")
    if list(root.iter("layer")) != layer_elements:
        raise ValueError("Nonstandard Bitmap layer placement")
    objects, bitmaps = [], []
    for layer in layer_elements:
        collections = layer.findall("objects")
        if len(collections) > 1:
            raise ValueError("Duplicate Bitmap objects collection")
        objects.extend(collections)
        if collections:
            bitmaps.extend(collections[0].findall("bitmap"))
    if list(root.iter("objects")) != objects or list(root.iter("bitmap")) != bitmaps:
        raise ValueError("Nonstandard Bitmap dependency placement")
    if len(bitmaps) != sum(isinstance(p, CADBitmap) for p in project.list_primitives()):
        raise ValueError("Unmodeled Bitmap dependency")
    return project


@dataclass(frozen=True)
class BitmapSource:
    """One primitive's identity, literal original reference and immutable bytes."""

    primitive_id: str
    identifier: str
    original_reference: str
    data: bytes


@dataclass(frozen=True)
class BitmapBundle:
    """Immutable source/document snapshot; captured images are never re-encoded."""

    document_bytes: bytes
    sources: tuple[BitmapSource, ...]

    def _project(self) -> CamBamProject:
        if type(self.document_bytes) is not bytes or type(self.sources) is not tuple:
            raise ValueError("Bitmap bundle requires immutable bytes and source tuple")
        project = _read_document(self.document_bytes)
        bitmaps = {str(p.internal_id): p for p in project.list_primitives() if isinstance(p, CADBitmap)}
        if not bitmaps or len(bitmaps) != len(self.sources):
            raise ValueError("Bitmap bundle must contain every bitmap exactly once")
        seen = set()
        for source in self.sources:
            if (not isinstance(source, BitmapSource) or type(source.data) is not bytes or
                    any(not isinstance(value, str) for value in
                        (source.primitive_id, source.identifier, source.original_reference))):
                raise ValueError("Bitmap bundle requires immutable source records")
            bitmap = bitmaps.get(source.primitive_id)
            if (bitmap is None or source.primitive_id in seen or
                    source.identifier != bitmap.user_identifier or
                    source.original_reference != bitmap.source_file):
                raise ValueError("Bitmap source does not match its document primitive")
            suffix = PureWindowsPath(source.original_reference).suffix.lower()
            if not _SUFFIX.fullmatch(suffix):
                raise ValueError("Bitmap capture requires a simple image filename extension")
            seen.add(source.primitive_id)
        return project

    def check_current(self, project: CamBamProject, *, source_directory: str | Path) -> None:
        """Reject changed document intent or changed/missing image dependencies."""
        self._project()
        if capture(project, source_directory=source_directory) != self:
            raise ValueError("Bitmap document or dependency changed after capture")

    def write_bundle(self, directory: str | Path) -> Path:
        """Write job.cb, images and manifest into a new relocatable directory.

        Content-addressed image paths retain the original extension. Existing
        destinations are never overwritten. The manifest is written last.
        """
        project = self._project()
        records, assets = [], {}
        for source in self.sources:
            digest = _sha(source.data)
            suffix = PureWindowsPath(source.original_reference).suffix.lower()
            relative = f"images/{digest}{suffix}"
            project.get_primitive(uuid.UUID(source.primitive_id)).source_file = relative
            assets[relative] = source.data
            records.append(dict(primitive_id=source.primitive_id, identifier=source.identifier,
                                original_reference=source.original_reference, path=relative,
                                sha256=digest, size_bytes=len(source.data)))
        document = serialize_cambam_bytes(project)
        manifest = dict(format=_FORMAT, original_document_sha256=_sha(self.document_bytes),
                        document_sha256=_sha(document), images=records)
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "images").mkdir()
        for relative, data in assets.items():
            (destination / relative).write_bytes(data)
        document_path = destination / "job.cb"
        document_path.write_bytes(document)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return document_path


def capture(project: CamBamProject, *, source_directory: str | Path) -> BitmapBundle:
    """Read every bitmap dependency within an explicitly selected directory.

    Relative native Windows references resolve from this directory. Absolute
    references are admitted only inside it; traversal/link escapes reject.
    Capturing does not mutate the project, decode images or trace geometry.
    Other external dependency kinds require their own explicit capture.
    """
    if not isinstance(project, CamBamProject):
        raise TypeError("project must be a CamBamProject")
    root = _directory(source_directory)
    snapshot = project.clone()
    sources = tuple(BitmapSource(str(p.internal_id), p.user_identifier, p.source_file,
                                 _read_within(root, _source_path(root, p.source_file)))
                    for p in snapshot.list_primitives() if isinstance(p, CADBitmap))
    bundle = BitmapBundle(serialize_cambam_bytes(snapshot), sources)
    bundle._project()
    return bundle


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate Bitmap manifest key")
        result[key] = value
    return result


def verify_bundle(directory: str | Path) -> dict[str, str | int]:
    """Check exact document/images and all bitmap identities/references.

    A mutable manifest is an integrity aid, not a provenance signature. This
    check does not establish image validity, tracing or machining correctness.
    """
    root = _directory(directory)
    try:
        manifest = json.loads(_read_within(root, root / "manifest.json"), object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Bitmap manifest JSON") from exc
    if (not isinstance(manifest, dict) or
            set(manifest) != {"format", "original_document_sha256", "document_sha256", "images"} or
            manifest["format"] != _FORMAT or not isinstance(manifest["images"], list)):
        raise ValueError("Unsupported Bitmap manifest schema")
    for field in ("original_document_sha256", "document_sha256"):
        if not isinstance(manifest[field], str) or not _HASH.fullmatch(manifest[field]):
            raise ValueError("Invalid Bitmap manifest document hash")
    document = _read_within(root, root / "job.cb")
    if _sha(document) != manifest["document_sha256"]:
        raise ValueError("Bitmap bundle document hash mismatch")
    project = _read_document(document)
    bitmaps = {str(p.internal_id): p for p in project.list_primitives() if isinstance(p, CADBitmap)}
    if not bitmaps or len(manifest["images"]) != len(bitmaps):
        raise ValueError("Bitmap manifest must cover every bitmap exactly once")
    seen, paths = set(), set()
    for record in manifest["images"]:
        if (not isinstance(record, dict) or set(record) != {
                "primitive_id", "identifier", "original_reference", "path", "sha256", "size_bytes"}):
            raise ValueError("Invalid Bitmap manifest image record")
        if (any(not isinstance(record[k], str) for k in record if k != "size_bytes") or
                type(record["size_bytes"]) is not int or record["size_bytes"] < 0 or
                not _HASH.fullmatch(record["sha256"])):
            raise ValueError("Invalid Bitmap manifest image values")
        suffix = PureWindowsPath(record["original_reference"]).suffix.lower()
        if not _SUFFIX.fullmatch(suffix) or record["path"] != f"images/{record['sha256']}{suffix}":
            raise ValueError("Invalid Bitmap manifest image path")
        bitmap = bitmaps.get(record["primitive_id"])
        if (bitmap is None or record["primitive_id"] in seen or
                bitmap.user_identifier != record["identifier"] or bitmap.source_file != record["path"]):
            raise ValueError("Bitmap manifest primitive identity/reference mismatch")
        data = _read_within(root, root / record["path"])
        if len(data) != record["size_bytes"] or _sha(data) != record["sha256"]:
            raise ValueError("Bitmap image hash/size mismatch")
        seen.add(record["primitive_id"])
        paths.add(record["path"])
    return dict(status="verified", document_path=str(root / "job.cb"),
                document_sha256=_sha(document), source_count=len(seen), asset_count=len(paths))
