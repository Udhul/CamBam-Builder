"""Source-bound native Surface interchange for compact constructive designs.

The native Surface is derived geometry for CamBam display/edit/handoff. The
compact JSON construction remains the design authority in its native Tag.
"""

from dataclasses import dataclass
import hashlib
import json
import math

from ...cam_core.constructive import ConstructiveSource
from ...native.project import CamBamProject
from ...native.reader import read_cambam_bytes
from ...native.surface import Surface
from ...native.writer import serialize_cambam_bytes


_ENVELOPE = "cambam-builder.constructive-surface/v1"


@dataclass(frozen=True)
class NativeConstructiveDocument:
    """Compact source and its bounded editable native triangle projection."""

    source: ConstructiveSource
    x_samples: int
    y_samples: int
    surface_identifier: str
    surface_internal_id: str
    xml_bytes: bytes
    source_deviation_bound_mm: float
    geometry_sha256: str

    def check_document(self, candidate_bytes: bytes | None = None, *,
                       current_source: ConstructiveSource | None = None) -> dict:
        """Check source provenance and native geometry against its derivation."""
        data = self.xml_bytes if candidate_bytes is None else candidate_bytes
        if type(data) is not bytes:
            raise ValueError("complete native document bytes required")
        if current_source is not None and (
                type(current_source) is not ConstructiveSource or
                current_source.fingerprint != self.source.fingerprint):
            raise ValueError("constructive source changed; explicitly regenerate")
        checked = _check(data)
        if checked["source"].fingerprint != self.source.fingerprint:
            raise ValueError("native constructive source changed; explicitly regenerate")
        if (checked["surface_identifier"] != self.surface_identifier or
                checked["surface_internal_id"] != self.surface_internal_id):
            raise ValueError("native constructive Surface identity changed")
        if (checked["x_samples"], checked["y_samples"]) != (
                self.x_samples, self.y_samples):
            raise ValueError("native constructive sampling parameters changed")
        if checked["geometry_sha256"] != self.geometry_sha256:
            raise ValueError("native constructive Surface changed; explicitly regenerate")
        if not math.isclose(checked["source_deviation_bound_mm"],
                            self.source_deviation_bound_mm, rel_tol=0, abs_tol=1e-15):
            raise ValueError("native constructive fidelity bound changed")
        return {"status": "pass", "source_id": self.source.source_id,
                "source_fingerprint": self.source.fingerprint,
                "source_deviation_bound_mm": self.source_deviation_bound_mm,
                "surface_identifier": self.surface_identifier,
                "geometry_sha256": checked["geometry_sha256"]}


def create_document(source: ConstructiveSource, *, x_samples: int,
                    y_samples: int,
                    surface_identifier: str | None = None
                    ) -> NativeConstructiveDocument:
    """Create an editable native Surface with a bound, compact source recipe.

    ``x_samples``/``y_samples`` include both endpoints. The source sampler's
    bound covers source-to-triangle deviation, including either cell diagonal;
    the native Surface rounds coordinates to nine decimal places, which is
    included in the reported bound.
    """
    if type(source) is not ConstructiveSource:
        raise ValueError("constructive source required")
    if (type(x_samples) is not int or type(y_samples) is not int or
            x_samples < 2 or y_samples < 2):
        raise ValueError("at least two samples per axis required")
    source_id = surface_identifier or f"{source.source_id}-surface"
    if type(source_id) is not str or not source_id:
        raise ValueError("nonempty Surface identifier required")
    sampled = source.sample_heightfield(x_samples=x_samples,
                                        y_samples=y_samples)
    mesh = _mesh(sampled.target)
    # NR03's mesh bound includes independent XYZ rounding and scale-aware
    # floating arithmetic. Adding its bilinear triangulation term is redundant
    # but conservative because the source bound already covers triangles.
    report_bound = sampled.deviation_bound_mm + mesh.maximum_error_mm
    if not math.isfinite(report_bound):
        raise ValueError("constructive native deviation bound is not representable")
    envelope = {
        "format": _ENVELOPE,
        "source_json": source.canonical_json(),
        "source_fingerprint": source.fingerprint,
        "source_id": source.source_id,
        "x_samples": x_samples,
        "y_samples": y_samples,
        "source_deviation_bound_mm": report_bound,
        "geometry_sha256": _geometry_fingerprint(mesh.vertices, mesh.faces),
    }
    project = CamBamProject(f"Constructive {source.source_id}")
    layer = project.add_layer("Constructive Surface")
    surface = project.add_surface(
        layer, mesh.vertices, mesh.faces, identifier=source_id,
        description="Derived from source-bound constructive geometry",
        display_edges=True)
    if surface is None:
        raise ValueError("native constructive Surface construction failed")
    envelope["surface_internal_id"] = str(surface.internal_id)
    surface.native_tag = json.dumps(envelope, sort_keys=True,
                                    separators=(",", ":"))
    data = serialize_cambam_bytes(project)
    document = NativeConstructiveDocument(
        source, x_samples, y_samples, source_id, str(surface.internal_id), data, report_bound,
        envelope["geometry_sha256"])
    document.check_document()
    return document


def read_document(data: bytes) -> NativeConstructiveDocument:
    """Recover a source-bound document from a supported native `.cb` file."""
    if type(data) is not bytes:
        raise ValueError("complete native document bytes required")
    checked = _check(data)
    source = checked["source"]
    document = NativeConstructiveDocument(
        source, checked["x_samples"], checked["y_samples"],
        checked["surface_identifier"], checked["surface_internal_id"], data,
        checked["reported_deviation_bound_mm"], checked["geometry_sha256"])
    document.check_document()
    return document


def regenerate(source: ConstructiveSource, *, x_samples: int, y_samples: int,
               surface_identifier: str | None = None
               ) -> NativeConstructiveDocument:
    """Explicitly derive a fresh native Surface after a source/modifier edit."""
    return create_document(source, x_samples=x_samples, y_samples=y_samples,
                           surface_identifier=surface_identifier)


def _mesh(target):
    from .native_relief import mesh_relief
    return mesh_relief(target, subdivisions=1)


def _check(data: bytes) -> dict:
    project = read_cambam_bytes(data, source_name="source-bound constructive Surface")
    # Installed CADFile.DrawingUnits has XML DefaultValue(0)=Millimeters;
    # its serializer omits that attribute. Normalize only this bound adapter,
    # preserving the ordinary reader's explicit/unspecified distinction.
    if project.units not in (None, "Millimeters"):
        raise ValueError("native constructive source requires Millimeters units")
    candidates = []
    for primitive in project.list_primitives():
        if not isinstance(primitive, Surface) or not primitive.native_tag:
            continue
        try:
            envelope = json.loads(primitive.native_tag)
        except (ValueError, TypeError):
            continue
        if isinstance(envelope, dict) and envelope.get("format") == _ENVELOPE:
            candidates.append((primitive, envelope))
    if len(candidates) != 1:
        raise ValueError("native document requires exactly one bound constructive Surface")
    surface, envelope = candidates[0]
    required = {"format", "source_json", "source_fingerprint", "source_id",
                "x_samples", "y_samples", "source_deviation_bound_mm",
                "geometry_sha256", "surface_internal_id"}
    if set(envelope) != required or type(envelope["source_json"]) is not str:
        raise ValueError("invalid constructive source envelope")
    if str(surface.internal_id) != envelope["surface_internal_id"]:
        raise ValueError("native constructive Surface identity changed")
    source = ConstructiveSource.from_json(envelope["source_json"])
    if (source.fingerprint != envelope["source_fingerprint"] or
            source.source_id != envelope["source_id"]):
        raise ValueError("constructive source provenance does not match its JSON")
    nx, ny = envelope["x_samples"], envelope["y_samples"]
    if (type(nx) is not int or type(ny) is not int or nx < 2 or ny < 2):
        raise ValueError("invalid constructive sample dimensions")
    sampled = source.sample_heightfield(x_samples=nx, y_samples=ny)
    mesh = _mesh(sampled.target)
    expected_hash = _geometry_fingerprint(mesh.vertices, mesh.faces)
    if expected_hash != envelope["geometry_sha256"]:
        raise ValueError("constructive source no longer reproduces its native Surface")
    # Compare effective world geometry so native placement edits cannot silently
    # detach the sampled mesh from its construction frame.
    geometry = surface.get_absolute_coordinates_xyz()
    actual_hash = _geometry_fingerprint(geometry["vertices"], geometry["faces"])
    canonical_vertices = tuple(tuple(round(value, 9) for value in point)
                               for point in mesh.vertices)
    if (tuple(geometry["vertices"]) != canonical_vertices or
            tuple(geometry["faces"]) != tuple(mesh.faces) or actual_hash != expected_hash):
        raise ValueError("native constructive Surface geometry changed")
    bound = envelope["source_deviation_bound_mm"]
    if type(bound) not in (int, float) or not math.isfinite(bound) or bound < 0:
        raise ValueError("invalid constructive source deviation bound")
    required_bound = sampled.deviation_bound_mm + mesh.maximum_error_mm
    if bound != required_bound:
        raise ValueError("constructive source deviation bound is inconsistent")
    return {"source": source, "surface_identifier": surface.user_identifier,
            "x_samples": nx, "y_samples": ny,
            "surface_internal_id": str(surface.internal_id),
            "source_deviation_bound_mm": bound,
            "reported_deviation_bound_mm": bound,
            "geometry_sha256": expected_hash}


def _geometry_fingerprint(vertices, faces) -> str:
    payload = json.dumps({
        "vertices": [[round(float(value), 9) for value in point]
                     for point in vertices],
        "faces": [list(face) for face in faces],
    }, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
