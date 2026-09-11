"""Volatile documents, revision serialization and process-lifetime retry ledger."""
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from uuid import UUID, uuid4

import anyio
import numpy as np
from jsonschema import ValidationError

from cambam_builder import CBProject
from cambam_builder.cambam_entities import (
    Arc, Circle, DrillMop, EngraveMop, Mop, Pline, PocketMop, Points,
    Primitive, ProfileMop, Rect, Text, Vertex,
)
from cambam_builder.cambam_reader import CamBamImportLimitError, read_cambam_bytes
from cambam_builder.region import Region
from .paths import DomainError, MAX_XML_BYTES, Workspace
from .schema import CONTRACT, OUTPUTS, StrictValidator, TOOLS, validated_arguments

_PARAMETER_VALIDATORS = {
    name: StrictValidator({"$ref": f"#/$defs/{name}", "$defs": CONTRACT["$defs"]})
    for name in ("ProfileParameters", "PocketParameters", "EngraveParameters",
                 "DrillParameters")
}
_GEOMETRY_VALIDATORS = {
    kind: StrictValidator({"$ref": f"#/$defs/{definition}", "$defs": CONTRACT["$defs"]})
    for kind, definition in {
        "rect": "RectGeometry", "circle": "CircleGeometry", "arc": "ArcGeometry",
        "pline": "PlineGeometry", "points": "PointsGeometry", "text": "TextGeometry",
        "region": "RegionGeometry",
    }.items()
}


@dataclass
class Document:
    project: object
    units: str
    source: object = None
    revision: int = 0
    lock: object = field(default_factory=anyio.Lock)


@dataclass
class Entry:
    signature: str
    done: object = field(default_factory=anyio.Event)
    result: object = None


def _typed(value, name):
    return value if StrictValidator(CONTRACT["$defs"][name]).is_valid(value) else None


class DocumentService:
    MAX_DOCUMENTS = 16
    MAX_REGULAR_REQUESTS = 10000
    MAX_PRIMITIVES = 10000
    MAX_MOPS = 1000
    SLICE_TYPES = (Rect, Circle, Arc, Pline, Points, Text, Region)
    EDIT_TOOLS = frozenset((
        "geometry_add_rectangle", "geometry_add_circle", "geometry_add_arc",
        "geometry_add_pline", "geometry_add_points", "geometry_add_text",
        "geometry_add_region", "geometry_translate", "geometry_translate_z",
        "geometry_rotate", "geometry_scale", "geometry_mirror", "geometry_bake",
        "machining_add_profile", "machining_add_pocket", "machining_add_engrave",
        "machining_add_drill", "machining_set_mop_targets",
        "relationship_set_parent", "relationship_add_to_group",
        "relationship_remove_from_group", "relationship_copy_tree",
        "relationship_copy_tree_between", "relationship_transfer_tree_between",
    ))
    CROSS_EDIT_TOOLS = frozenset((
        "relationship_copy_tree_between", "relationship_transfer_tree_between",
    ))
    MOP_TARGET_RULES = {
        "profile": ("supported root rectangles", lambda e: isinstance(e, Rect)),
        "pocket": ("supported root Rect/Circle/closed-Pline/Region shapes",
                   lambda e: isinstance(e, (Rect, Circle, Region))
                   or (isinstance(e, Pline) and bool(e.closed))),
        "engrave": ("supported root Rect/Circle/Arc/Pline curves",
                    lambda e: isinstance(e, (Rect, Circle, Arc, Pline))),
        "drill": ("supported root Points/Circle primitives",
                  lambda e: isinstance(e, (Points, Circle))),
    }
    MOP_KIND_BY_CLASS = {
        ProfileMop: "profile", PocketMop: "pocket",
        EngraveMop: "engrave", DrillMop: "drill",
    }

    def __init__(self, root):
        self.workspace = Workspace(root)
        self.bootstrap = {"schema_version": 1, "workspace_id": self.workspace.id, "boot_id": str(uuid4())}
        self.documents = {}
        self.ledger = {}
        self.registry_lock = anyio.Lock()
        self.ledger_lock = anyio.Lock()
        self.reservations = 0
        self.regular_requests = 0

    def _envelope(self, args, *, data=None, error=None, diagnostics=(), document=None, revision=None):
        handle = document or _typed(args.get("document"), "Handle")
        current = self.documents.get(handle)
        if revision is None and current is not None:
            revision = current.revision
        return {"schema_version": 1, "ok": error is None, "workspace_id": self.workspace.id,
                "document": handle, "revision": revision,
                "request_id": _typed(args.get("request_id"), "UUID"), "replayed": False,
                "data": data if error is None else None, "diagnostics": list(diagnostics),
                "error": None if error is None else {"code": error.code, "message": str(error)[:1024],
                    "field": error.field, "current_revision": revision}}

    @staticmethod
    def _diagnostics(units, interchange=False):
        result = [{"code": "UNITS_ASSUMED", "message": f"Coordinates are asserted as {units}; CamBam units must match. No unit conversion or verified units setting is written."}]
        if interchange:
            result.append({"code": "INTERCHANGE_LIMITED", "message": "Unsupported XML metadata has no preservation guarantee."})
        return result

    async def call_tool(self, name, arguments):
        if name not in TOOLS:
            raise ValueError("Unknown tool")
        args = arguments if type(arguments) is dict else {}
        entry = None
        owns_entry = False
        try:
            try:
                validated = validated_arguments(name, arguments)
                validation_error = None
            except (ValidationError, TypeError, ValueError, OverflowError):
                validated = copy.deepcopy(args)
                validation_error = DomainError("INVALID_ARGUMENT", "Arguments do not match the tool schema")
            # Invalid identity cannot reserve ledger state. Valid identity with invalid
            # remaining arguments does reserve and caches its terminal failure.
            request_id = _typed(args.get("request_id"), "UUID")
            workspace_id = _typed(args.get("workspace_id"), "Workspace")
            if name not in ("document_export", "document_inspect") and request_id is not None and workspace_id is not None:
                ledger_key = (workspace_id, request_id)
                try:
                    signature_args = {k: v for k, v in validated.items() if k != "request_id"}
                    if (
                        name == "document_import"
                        and isinstance(signature_args.get("content"), str)
                    ):
                        content = signature_args.pop("content")
                        encoded = content.encode("utf-8")
                        signature_args["content_sha256"] = hashlib.sha256(encoded).hexdigest()
                        signature_args["content_bytes"] = len(encoded)
                    signature = json.dumps([name, signature_args],
                                           sort_keys=True, separators=(",", ":"), allow_nan=False)
                except (TypeError, ValueError):
                    raise DomainError("INVALID_ARGUMENT", "Arguments must be finite JSON values")
                async with self.ledger_lock:
                    entry = self.ledger.get(ledger_key)
                    if entry is not None:
                        if entry.signature != signature:
                            raise DomainError("REQUEST_ID_CONFLICT", "Request ID already identifies different arguments", "request_id")
                        replay = True
                    else:
                        regular = name not in ("document_save", "document_close")
                        if regular and self.regular_requests >= self.MAX_REGULAR_REQUESTS:
                            raise DomainError("LIMIT_EXCEEDED", "Regular request ledger is full; save and close remain available")
                        entry = Entry(signature)
                        self.ledger[ledger_key] = entry
                        owns_entry = True
                        self.regular_requests += int(regular)
                        replay = False
                if replay:
                    # A cancelled waiter never cancels or completes the owner entry.
                    await entry.done.wait()
                    result = copy.deepcopy(entry.result)
                    result["replayed"] = True
                    return result
            if validation_error:
                raise validation_error
            args = validated
            if name in self.CROSS_EDIT_TOOLS:
                # Cross-document envelopes address the source handle by
                # default; target-side failures carry their own handle on
                # the error.
                args["document"] = args["source_document"]
            if args["workspace_id"] != self.workspace.id:
                raise DomainError("WORKSPACE_MISMATCH", "Workspace ID does not match this server", "workspace_id")
            if name in ("document_create", "document_import", "document_open"):
                return await self._new(name, args, entry)
            if name in self.CROSS_EDIT_TOOLS:
                return await self._cross_edit(name, args, entry)
            handle = args["document"]
            if handle.split(":")[0] != self.bootstrap["boot_id"]:
                raise DomainError("DOCUMENT_EXPIRED", "Document belongs to a previous server process", "document")
            document = self.documents.get(handle)
            if document is None:
                raise DomainError("DOCUMENT_NOT_FOUND", "Document is not open", "document")
            async with document.lock:
                if self.documents.get(handle) is not document:
                    raise DomainError("DOCUMENT_NOT_FOUND", "Document is not open", "document")
                if args.get("expected_revision", document.revision) != document.revision:
                    raise DomainError("STALE_REVISION", "Expected revision does not match", "expected_revision")
                if name == "document_inspect":
                    data, diagnostics = self._inspect(document, args)
                    return self._complete(name, entry, self._envelope(args, data=data, diagnostics=diagnostics))
                if name == "document_export":
                    return await self._export(name, args, document)
                if name == "document_close":
                    with anyio.CancelScope(shield=True):
                        async with self.registry_lock:
                            del self.documents[handle]
                        return self._complete(name, entry, self._envelope(args, data={"closed": True}, revision=document.revision))
                if name in self.EDIT_TOOLS:
                    return await self._edit(name, args, document, entry)
                return await self._save(name, args, document, entry)
        except anyio.get_cancelled_exc_class():
            # Do not overwrite a completed commit (or another request's entry).
            if owns_entry and entry.result is None:
                self._complete(name, entry, self._envelope(args, error=DomainError("REQUEST_CANCELLED", "Request canceled before publication")))
            raise
        except DomainError as exc:
            result = self._envelope(args, error=exc, document=getattr(exc, "document", None))
        except OSError:
            result = self._envelope(args, error=DomainError("IO_ERROR", "Workspace I/O failed"))
        except Exception:
            result = self._envelope(args, error=DomainError("INTERNAL_ERROR", "Operation failed without publication"))
        # Conflicting reuses must never replace the existing ledger result.
        if entry is not None and not owns_entry:
            return result
        return self._complete(name, entry, result)

    def _complete(self, name, entry, result):
        OUTPUTS[name].validate(result)
        if entry is not None:
            entry.result = copy.deepcopy(result)
            entry.done.set()
        return result

    @staticmethod
    def _summary(document):
        project = document.project
        return {"name": project.project_name, "units": document.units, "source": copy.deepcopy(document.source),
                "counts": {"layers": len(project.list_layers()), "parts": len(project.list_parts()),
                           "primitives": len(project.list_primitives()), "mops": len(project.list_mops())}}

    @classmethod
    def _check_limits(cls, project):
        if (len(project.list_primitives()) > cls.MAX_PRIMITIVES
                or len(project.list_mops()) > cls.MAX_MOPS):
            raise DomainError("LIMIT_EXCEEDED", "Document exceeds 10000 primitives or 1000 MOPs")

    def _stage_new(self, name, args):
        if name == "document_create":
            return Document(CBProject(args["name"]), args["units"])
        if name == "document_import":
            try:
                data = args["content"].encode("utf-8")
            except UnicodeEncodeError:
                raise DomainError(
                    "IMPORT_FAILED", "Document content must be valid UTF-8", "content"
                ) from None
            if len(data) > MAX_XML_BYTES:
                raise DomainError("LIMIT_EXCEEDED", "XML exceeds 10 MiB", "content")
            source_name = args["source_name"]
        else:
            data = self.workspace.read(args["path"])
            source_name = args["path"]
        try:
            project = read_cambam_bytes(data, source_name=source_name, strict=True)
        except CamBamImportLimitError:
            field = "content" if name == "document_import" else "path"
            raise DomainError("LIMIT_EXCEEDED", "Document exceeds import resource limits", field) from None
        except ValueError:
            field = "content" if name == "document_import" else "path"
            raise DomainError("IMPORT_FAILED", "Strict XML import failed", field) from None
        self._check_limits(project)
        source = {"sha256": hashlib.sha256(data).hexdigest()}
        if name == "document_import":
            source.update({"name": source_name, "bytes": len(data)})
        else:
            source["path"] = source_name
        return Document(project, args["units"], source)

    async def _new(self, name, args, entry):
        async with self.registry_lock:
            if len(self.documents) + self.reservations >= self.MAX_DOCUMENTS:
                raise DomainError("LIMIT_EXCEEDED", "At most 16 documents may be open")
            self.reservations += 1
        reserved = True
        try:
            document = await anyio.to_thread.run_sync(self._stage_new, name, args)
            await anyio.lowlevel.checkpoint()
            handle = self.bootstrap["boot_id"] + ":" + str(uuid4())
            result = self._envelope(args, data=self._summary(document), document=handle, revision=0,
                                    diagnostics=self._diagnostics(
                                        document.units, name in ("document_import", "document_open")
                                    ))
            OUTPUTS[name].validate(result)
            with anyio.CancelScope(shield=True):
                async with self.registry_lock:
                    self.documents[handle] = document
                    self.reservations -= 1
                    reserved = False
                return self._complete(name, entry, result)
        finally:
            if reserved:
                with anyio.CancelScope(shield=True):
                    async with self.registry_lock:
                        self.reservations -= 1

    async def _save(self, name, args, document, entry):
        temporary = None
        try:
            def stage():
                try:
                    return self.workspace.stage_save(document.project.clone(), args["path"])
                except (DomainError, OSError):
                    raise
                except Exception:
                    raise DomainError("EXPORT_FAILED", "XML export failed") from None
            temporary, artifact = await anyio.to_thread.run_sync(stage)
            await anyio.lowlevel.checkpoint()
            diagnostics = self._diagnostics(document.units, True)
            result = self._envelope(args, data=artifact, diagnostics=diagnostics)
            OUTPUTS[name].validate(result)
            with anyio.CancelScope(shield=True):
                # No await between publication and completion: cancellation cannot
                # make a published artifact appear to be a failed operation.
                self.workspace.publish(temporary, args["path"])
                if not self.workspace.cleanup(temporary):
                    result["diagnostics"].append({"code": "CLEANUP_PENDING", "message": "Saved successfully; an adapter temporary remains."})
                temporary = None
                return self._complete(name, entry, result)
        finally:
            if temporary is not None:
                self.workspace.cleanup(temporary)

    async def _export(self, name, args, document):
        def serialize():
            try:
                return self.workspace.serialize(document.project.clone())
            except (DomainError, OSError):
                raise
            except Exception:
                raise DomainError("EXPORT_FAILED", "XML export failed") from None

        data = await anyio.to_thread.run_sync(serialize)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            raise DomainError("EXPORT_FAILED", "XML export was not valid UTF-8") from None
        artifact = {
            "kind": "cambam_document",
            "mime_type": "application/xml",
            "encoding": "utf-8",
            "suggested_filename": args["suggested_filename"],
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "content": content,
        }
        return self._complete(
            name,
            None,
            self._envelope(
                args,
                data=artifact,
                diagnostics=self._diagnostics(document.units, True),
            ),
        )

    @staticmethod
    def _identifier_available(project, identifier, field):
        if project.get_entity(identifier) is not None:
            raise DomainError("IDENTIFIER_CONFLICT", "Identifier is already in use", field)

    @staticmethod
    def _similarity_scale(matrix):
        """Return the non-degenerate XY similarity scale of a 3x3 affine, or None."""
        if not np.isfinite(matrix).all():
            return None
        linear = matrix[:2, :2]
        magnitude = max(abs(float(value)) for value in linear.flat)
        if magnitude == 0.0:
            return None
        normalized = linear / magnitude
        gram = normalized.T @ normalized
        scale_squared = float((gram[0, 0] + gram[1, 1]) / 2.0)
        if (not math.isfinite(scale_squared) or scale_squared <= 0.0
                or not np.allclose(gram, np.identity(2) * scale_squared,
                                   rtol=0.0, atol=1e-12 * max(1.0, scale_squared))):
            return None
        scale = magnitude * math.sqrt(scale_squared)
        return scale if math.isfinite(scale) else None

    @staticmethod
    def _slice_supported(project, entity):
        """Return whether a primitive stays inside the root similarity slice."""
        if not isinstance(entity, DocumentService.SLICE_TYPES):
            return False
        if project.get_parent_of_primitive(entity) is not None:
            return False
        if project.get_children_of_primitive(entity) or project.get_groups_of_primitive(entity):
            return False
        matrix = entity.effective_transform
        try:
            supported = (
                matrix.shape == (3, 3)
                and np.array_equal(matrix[2], (0.0, 0.0, 1.0))
                and float(entity.local_z_offset) == 0.0
                and DocumentService._similarity_scale(matrix) is not None
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        if not supported:
            return False
        try:
            if isinstance(entity, Rect):
                return float(entity.width) > 0.0 and float(entity.height) > 0.0
            if isinstance(entity, Circle):
                return float(entity.diameter) > 0.0
            if isinstance(entity, Arc):
                return float(entity.radius) > 0.0
            if isinstance(entity, Text):
                return float(entity.height) > 0.0
            if isinstance(entity, Region):
                return all(
                    bool(contour.vertices) and all(
                        math.isfinite(float(vertex.x)) and math.isfinite(float(vertex.y))
                        and math.isfinite(float(vertex.z))
                        and math.isfinite(float(vertex.bulge))
                        for vertex in contour.vertices)
                    for contour in entity.contours)
            vertices = entity.vertices
            return bool(vertices) and all(
                math.isfinite(float(vertex.x)) and math.isfinite(float(vertex.y))
                and math.isfinite(float(vertex.z)) and math.isfinite(float(vertex.bulge))
                for vertex in vertices
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    def _slice_rect(self, project, entity):
        return isinstance(entity, Rect) and self._slice_supported(project, entity)

    @staticmethod
    def _checked_values(values):
        if any(
                not math.isfinite(float(value)) or abs(float(value)) > 1_000_000_000
                for value in values):
            raise DomainError("INVALID_ARGUMENT", "Resulting geometry exceeds the supported coordinate range")
        return [float(value) for value in values]

    @classmethod
    def _world_bounds(cls, entity):
        bounds = entity.get_bounding_box()
        if not bounds.is_valid():
            raise DomainError("INVALID_ARGUMENT", "Resulting geometry exceeds the supported coordinate range")
        return cls._checked_values(
            (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y))

    def _rect_geometry(self, entity):
        try:
            world = [[float(value) for value in point]
                     for point in entity.get_absolute_coordinates_xyz()]
            bounds = self._world_bounds(entity)
        except DomainError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError):
            raise DomainError("UNSUPPORTED_OPERATION", "Rectangle geometry is outside the supported slice") from None
        if len(world) != 4 or any(len(point) != 3 for point in world):
            raise DomainError("UNSUPPORTED_OPERATION", "Rectangle geometry is outside the supported slice")
        self._checked_values([value for point in world for value in point])
        return {"kind": "rect", "world_xyz": world, "bounds": bounds}

    def _circle_geometry(self, entity):
        try:
            geometry = entity.get_absolute_coordinates_xyz()
            center = tuple(geometry["center"])
            if len(center) != 3:
                raise TypeError("circle center must be XYZ")
            center = self._checked_values(center)
            diameter = self._checked_values((geometry["diameter"],))[0]
            bounds = self._world_bounds(entity)
        except DomainError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError, KeyError):
            raise DomainError("UNSUPPORTED_OPERATION", "Circle geometry is outside the supported slice") from None
        if diameter <= 0.0:
            raise DomainError("UNSUPPORTED_OPERATION", "Circle geometry is outside the supported slice")
        return {"kind": "circle", "center": center, "diameter": diameter, "bounds": bounds}

    def _arc_geometry(self, entity):
        try:
            geometry = entity.get_absolute_coordinates_xyz()
            center = tuple(geometry["center"])
            if len(center) != 3:
                raise TypeError("arc center must be XYZ")
            center = self._checked_values(center)
            radius = self._checked_values((geometry["radius"],))[0]
            angles = self._checked_values((geometry["start_angle"], geometry["extent_angle"]))
            bounds = self._world_bounds(entity)
        except DomainError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError, KeyError):
            raise DomainError("UNSUPPORTED_OPERATION", "Arc geometry is outside the supported slice") from None
        if radius <= 0.0:
            raise DomainError("UNSUPPORTED_OPERATION", "Arc geometry is outside the supported slice")
        return {"kind": "arc", "center": center, "radius": radius,
                "start_angle": angles[0], "extent_angle": angles[1], "bounds": bounds}

    def _pline_geometry(self, entity):
        try:
            points = entity.get_absolute_coordinates_xyz()
            bounds = self._world_bounds(entity)
        except DomainError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError):
            raise DomainError("UNSUPPORTED_OPERATION", "Polyline geometry is outside the supported slice") from None
        if not points or any(len(point) != 4 for point in points):
            raise DomainError("UNSUPPORTED_OPERATION", "Polyline geometry is outside the supported slice")
        world_xyz = self._checked_values(
            [value for point in points for value in point[:3]])
        bulges = self._checked_values([point[3] for point in points])
        return {"kind": "pline",
                "world_xyz": [world_xyz[index:index + 3] for index in range(0, len(world_xyz), 3)],
                "bulges": bulges, "closed": bool(entity.closed), "bounds": bounds}

    def _points_geometry(self, entity):
        try:
            points = entity.get_absolute_coordinates_xyz()
            bounds = self._world_bounds(entity)
        except DomainError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError):
            raise DomainError("UNSUPPORTED_OPERATION", "Point-list geometry is outside the supported slice") from None
        if not points or any(len(point) != 3 for point in points):
            raise DomainError("UNSUPPORTED_OPERATION", "Point-list geometry is outside the supported slice")
        world_xyz = self._checked_values(
            [value for point in points for value in point])
        return {"kind": "points",
                "world_xyz": [world_xyz[index:index + 3] for index in range(0, len(world_xyz), 3)],
                "bounds": bounds}

    def _text_geometry(self, entity):
        try:
            geometry = entity.get_absolute_coordinates_xyz()
            anchor = tuple(geometry["position"])
            if len(anchor) != 3:
                raise TypeError("text anchor must be XYZ")
            anchor = self._checked_values(anchor)
            height = self._checked_values((geometry["height"],))[0]
            line_spacing = self._checked_values((entity.line_spacing,))[0]
            p2 = geometry.get("xml_p2")
            if p2 is not None:
                p2 = tuple(p2)
                if len(p2) != 3:
                    raise TypeError("text p2 must be XYZ")
                p2 = self._checked_values(p2)
        except DomainError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError, KeyError):
            raise DomainError("UNSUPPORTED_OPERATION", "Text geometry is outside the supported slice") from None
        if height <= 0.0:
            raise DomainError("UNSUPPORTED_OPERATION", "Text geometry is outside the supported slice")
        return {"kind": "text", "text": str(entity.text_content), "anchor": anchor,
                "height": height, "font": str(entity.font), "style": str(entity.style),
                "line_spacing": line_spacing, "align_horizontal": str(entity.align_horizontal),
                "align_vertical": str(entity.align_vertical), "p2": p2}

    def _contour_payload(self, points, name):
        if not points or any(len(point) != 4 for point in points):
            raise DomainError("UNSUPPORTED_OPERATION", f"{name} geometry is outside the supported slice")
        world_xyz = self._checked_values(
            [value for point in points for value in point[:3]])
        bulges = self._checked_values([point[3] for point in points])
        return {"world_xyz": [world_xyz[index:index + 3] for index in range(0, len(world_xyz), 3)],
                "bulges": bulges}

    def _region_geometry(self, entity):
        try:
            geometry = entity.get_absolute_coordinates_xyz()
            bounds = self._world_bounds(entity)
            outer = self._contour_payload(geometry["outer_curve"], "Region outer curve")
            holes = [self._contour_payload(contour, "Region hole curve")
                     for contour in geometry["hole_curves"]]
        except DomainError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError, KeyError, IndexError):
            raise DomainError("UNSUPPORTED_OPERATION", "Region geometry is outside the supported slice") from None
        return {"kind": "region", "outer_curve": outer, "hole_curves": holes, "bounds": bounds}

    def _geometry_payload(self, entity):
        if isinstance(entity, Rect):
            result = self._rect_geometry(entity)
        elif isinstance(entity, Circle):
            result = self._circle_geometry(entity)
        elif isinstance(entity, Arc):
            result = self._arc_geometry(entity)
        elif isinstance(entity, Pline):
            result = self._pline_geometry(entity)
        elif isinstance(entity, Points):
            result = self._points_geometry(entity)
        elif isinstance(entity, Text):
            result = self._text_geometry(entity)
        elif isinstance(entity, Region):
            result = self._region_geometry(entity)
        else:
            raise DomainError(
                "UNSUPPORTED_OPERATION", "Primitive geometry is outside the supported slice")
        if not _GEOMETRY_VALIDATORS[result["kind"]].is_valid(result):
            raise DomainError(
                "UNSUPPORTED_OPERATION",
                "Primitive geometry exceeds the supported inspection schema",
            )
        return result

    def _require_slice_rect(self, project, entity_id, field="entity_id"):
        entity = project.get_entity(UUID(entity_id))
        if entity is None:
            raise DomainError("ENTITY_NOT_FOUND", "Entity was not found", field)
        if not self._slice_rect(project, entity):
            raise DomainError("UNSUPPORTED_OPERATION", "Entity is not a supported root rectangle", field)
        self._rect_geometry(entity)
        return entity

    def _require_slice_primitive(self, project, entity_id, field="entity_id"):
        entity = project.get_entity(UUID(entity_id))
        if entity is None:
            raise DomainError("ENTITY_NOT_FOUND", "Entity was not found", field)
        if not self._slice_supported(project, entity):
            raise DomainError("UNSUPPORTED_OPERATION", "Entity is not a supported root primitive", field)
        self._geometry_payload(entity)
        return entity

    @classmethod
    def _target_allowed(cls, kind, entity):
        return bool(cls.MOP_TARGET_RULES[kind][1](entity))

    def _require_mop_target(self, project, entity_id, kind):
        entity = self._require_slice_primitive(project, entity_id, "targets")
        message = self.MOP_TARGET_RULES[kind][0]
        if not self._target_allowed(kind, entity):
            raise DomainError("UNSUPPORTED_OPERATION",
                              f"{kind} targets must be {message}", "targets")
        return entity

    def _stage_mop_prelude(self, staged, args, kind):
        self._identifier_available(staged, args["identifier"], "identifier")
        if args["target_depth"] >= args["stock_surface"]:
            raise DomainError("INVALID_ARGUMENT", "Target depth must be below stock surface", "target_depth")
        if args["clearance_plane"] <= args["stock_surface"]:
            raise DomainError("INVALID_ARGUMENT", "Clearance plane must be above stock surface", "clearance_plane")
        targets = [self._require_mop_target(staged, target, kind)
                   for target in args["targets"]]
        part_entity = staged.get_entity(args["part"])
        part = staged.get_part(args["part"])
        if part_entity is None and args["identifier"] == args["part"]:
            raise DomainError("IDENTIFIER_CONFLICT", "MOP and new part identifiers must differ", "identifier")
        if part_entity is not None and part is None:
            raise DomainError("IDENTIFIER_CONFLICT", "Part name is used by another entity", "part")
        if part is None:
            part = staged.add_part(
                args["part"], enabled=True, stock_thickness=0.0,
                stock_width=0.0, stock_height=0.0, stock_material="",
                machining_origin=(0.0, 0.0), default_tool_diameter=None,
                default_spindle_speed=None,
            )
        if part is None:
            raise DomainError("INTERNAL_ERROR", "Framework rejected part creation")
        return part, targets

    @staticmethod
    def _check_new_layer_name(project, args):
        layer_entity = project.get_entity(args["layer"])
        if layer_entity is None and args["identifier"] == args["layer"]:
            raise DomainError("IDENTIFIER_CONFLICT", "Primitive and new layer identifiers must differ", "identifier")
        if layer_entity is not None and project.get_layer(args["layer"]) is None:
            raise DomainError("IDENTIFIER_CONFLICT", "Layer name is used by another entity", "layer")

    @staticmethod
    def _vertex_records(points, allow_bulge):
        return [
            Vertex(point["x"], point["y"], point["z"], bulge=point.get("bulge", 0.0))
            if allow_bulge else Vertex(point["x"], point["y"], point["z"])
            for point in points
        ]

    def _stage_edit(self, name, args, project):
        staged = project.clone()
        if name == "geometry_add_rectangle":
            self._identifier_available(staged, args["identifier"], "identifier")
            self._check_new_layer_name(staged, args)
            rectangle = staged.add_rect(
                layer=args["layer"], identifier=args["identifier"],
                corner=(args["x"], args["y"]), width=args["width"],
                height=args["height"], elevation=args["z"],
            )
            if rectangle is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected rectangle creation")
            self._rect_geometry(rectangle)
            self._check_limits(staged)
            data = {"entity_id": str(rectangle.internal_id), "layer": args["layer"]}
        elif name == "geometry_add_circle":
            self._identifier_available(staged, args["identifier"], "identifier")
            self._check_new_layer_name(staged, args)
            circle = staged.add_circle(
                layer=args["layer"], identifier=args["identifier"],
                center=(args["x"], args["y"]), diameter=args["diameter"],
                elevation=args["z"],
            )
            if circle is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected circle creation")
            self._circle_geometry(circle)
            self._check_limits(staged)
            data = {"entity_id": str(circle.internal_id), "layer": args["layer"]}
        elif name == "geometry_add_arc":
            self._identifier_available(staged, args["identifier"], "identifier")
            self._check_new_layer_name(staged, args)
            arc = staged.add_arc(
                layer=args["layer"], identifier=args["identifier"],
                center=(args["x"], args["y"]), radius=args["radius"],
                start_angle=args["start_angle"], extent_angle=args["extent_angle"],
                elevation=args["z"],
            )
            if arc is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected arc creation")
            self._arc_geometry(arc)
            self._check_limits(staged)
            data = {"entity_id": str(arc.internal_id), "layer": args["layer"]}
        elif name == "geometry_add_pline":
            self._identifier_available(staged, args["identifier"], "identifier")
            self._check_new_layer_name(staged, args)
            pline = staged.add_pline(
                layer=args["layer"], identifier=args["identifier"],
                points=self._vertex_records(args["points"], allow_bulge=True),
                closed=args["closed"],
            )
            if pline is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected polyline creation")
            self._pline_geometry(pline)
            self._check_limits(staged)
            data = {"entity_id": str(pline.internal_id), "layer": args["layer"]}
        elif name == "geometry_add_points":
            self._identifier_available(staged, args["identifier"], "identifier")
            self._check_new_layer_name(staged, args)
            points = staged.add_points(
                layer=args["layer"], identifier=args["identifier"],
                points=self._vertex_records(args["points"], allow_bulge=False),
            )
            if points is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected point-list creation")
            self._points_geometry(points)
            self._check_limits(staged)
            data = {"entity_id": str(points.internal_id), "layer": args["layer"]}
        elif name == "geometry_add_text":
            self._identifier_available(staged, args["identifier"], "identifier")
            self._check_new_layer_name(staged, args)
            text = staged.add_text(
                layer=args["layer"], identifier=args["identifier"],
                text=args["text"], position=(args["x"], args["y"]),
                height=args["height"], font=args["font"], style=args["style"],
                line_spacing=args["line_spacing"],
                align_horizontal=args["align_horizontal"],
                align_vertical=args["align_vertical"], elevation=args["z"],
            )
            if text is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected text creation")
            self._text_geometry(text)
            self._check_limits(staged)
            data = {"entity_id": str(text.internal_id), "layer": args["layer"]}
        elif name == "geometry_add_region":
            self._identifier_available(staged, args["identifier"], "identifier")
            self._check_new_layer_name(staged, args)
            outer = Pline(vertices=self._vertex_records(
                args["outer"]["points"], allow_bulge=True), closed=True)
            holes = [Pline(vertices=self._vertex_records(
                contour["points"], allow_bulge=True), closed=True)
                for contour in args["holes"]]
            region = staged.add_region(
                layer=args["layer"], identifier=args["identifier"],
                outer_curve=outer, hole_curves=holes,
            )
            if region is None:
                try:
                    Region(outer_curve=outer, hole_curves=holes)
                except (TypeError, ValueError) as exc:
                    raise DomainError("INVALID_ARGUMENT", str(exc)[:512], "outer") from None
                raise DomainError("INTERNAL_ERROR", "Framework rejected Region creation")
            self._region_geometry(region)
            self._check_limits(staged)
            data = {"entity_id": str(region.internal_id), "layer": args["layer"]}
        elif name == "geometry_translate":
            entity = self._require_slice_primitive(staged, args["entity_id"])
            if not staged.translate_primitive(entity.internal_id, args["dx"], args["dy"], bake=False):
                raise DomainError("UNSUPPORTED_OPERATION", "Translation was rejected", "entity_id")
            self._geometry_payload(entity)
            data = {"entity_id": str(entity.internal_id)}
        elif name == "geometry_translate_z":
            entity = self._require_slice_primitive(staged, args["entity_id"])
            if not staged.translate_primitive_z(entity.internal_id, args["dz"], bake=True):
                raise DomainError("UNSUPPORTED_OPERATION", "Z translation was rejected", "entity_id")
            self._geometry_payload(entity)
            data = {"entity_id": str(entity.internal_id)}
        elif name == "geometry_rotate":
            entity = self._require_slice_primitive(staged, args["entity_id"])
            if not staged.rotate_primitive_deg(
                    entity.internal_id, args["angle_deg"],
                    args.get("cx"), args.get("cy"), bake=False):
                raise DomainError("UNSUPPORTED_OPERATION", "Rotation was rejected", "entity_id")
            self._geometry_payload(entity)
            data = {"entity_id": str(entity.internal_id)}
        elif name == "geometry_scale":
            entity = self._require_slice_primitive(staged, args["entity_id"])
            if not staged.scale_primitive(
                    entity.internal_id, args["factor"], args["factor"],
                    args.get("cx"), args.get("cy"), bake=False):
                raise DomainError("UNSUPPORTED_OPERATION", "Scaling was rejected", "entity_id")
            self._geometry_payload(entity)
            data = {"entity_id": str(entity.internal_id)}
        elif name == "geometry_mirror":
            entity = self._require_slice_primitive(staged, args["entity_id"])
            mirror = (staged.mirror_primitive_x if args["axis"] == "x"
                      else staged.mirror_primitive_y)
            if not mirror(entity.internal_id, args.get("position"), bake=False):
                raise DomainError("UNSUPPORTED_OPERATION", "Mirroring was rejected", "entity_id")
            self._geometry_payload(entity)
            data = {"entity_id": str(entity.internal_id)}
        elif name == "geometry_bake":
            entity = self._require_slice_primitive(staged, args["entity_id"])
            try:
                entity.bake_geometry()
            except (TypeError, ValueError) as exc:
                raise DomainError("UNSUPPORTED_OPERATION",
                                  f"Bake was rejected for this geometry: {exc}"[:512],
                                  "entity_id") from None
            self._geometry_payload(entity)
            data = {"entity_id": str(entity.internal_id), "type": type(entity).__name__}
        elif name == "relationship_set_parent":
            child = staged.get_entity(UUID(args["entity_id"]))
            if child is None:
                raise DomainError("ENTITY_NOT_FOUND", "Entity was not found", "entity_id")
            if not isinstance(child, Primitive):
                raise DomainError("UNSUPPORTED_OPERATION", "Entity is not a primitive", "entity_id")
            parent = None
            if args["parent_id"] is not None:
                parent = staged.get_entity(UUID(args["parent_id"]))
                if parent is None:
                    raise DomainError("ENTITY_NOT_FOUND", "Parent was not found", "parent_id")
                if not isinstance(parent, Primitive):
                    raise DomainError("UNSUPPORTED_OPERATION", "Parent is not a primitive", "parent_id")
            if parent is not None and parent.internal_id == child.internal_id:
                raise DomainError("INVALID_ARGUMENT", "A primitive cannot be its own parent", "parent_id")
            if not staged.link_primitive_parent(
                    child.internal_id, parent.internal_id if parent else None):
                raise DomainError("INVALID_ARGUMENT",
                                  "Parent link was rejected (cycle or invalid link)", "parent_id")
            data = {"entity_id": str(child.internal_id),
                    "parent": str(parent.internal_id) if parent else None}
        elif name == "relationship_add_to_group":
            entity = staged.get_entity(UUID(args["entity_id"]))
            if entity is None:
                raise DomainError("ENTITY_NOT_FOUND", "Entity was not found", "entity_id")
            if not isinstance(entity, Primitive):
                raise DomainError("UNSUPPORTED_OPERATION", "Entity is not a primitive", "entity_id")
            if not staged.add_primitive_to_group(entity.internal_id, args["group"]):
                raise DomainError("INVALID_ARGUMENT", "Group membership was rejected", "group")
            data = {"entity_id": str(entity.internal_id),
                    "groups": sorted(staged.get_groups_of_primitive(entity))}
        elif name == "relationship_remove_from_group":
            entity = staged.get_entity(UUID(args["entity_id"]))
            if entity is None:
                raise DomainError("ENTITY_NOT_FOUND", "Entity was not found", "entity_id")
            if not isinstance(entity, Primitive):
                raise DomainError("UNSUPPORTED_OPERATION", "Entity is not a primitive", "entity_id")
            if not staged.remove_primitive_from_group(entity.internal_id, args["group"]):
                raise DomainError("INVALID_ARGUMENT", "Group membership was rejected", "group")
            data = {"entity_id": str(entity.internal_id),
                    "groups": sorted(staged.get_groups_of_primitive(entity))}
        elif name == "relationship_copy_tree":
            root = staged.get_entity(UUID(args["root"]))
            if root is None:
                raise DomainError("ENTITY_NOT_FOUND", "Root was not found", "root")
            if not isinstance(root, Primitive):
                raise DomainError("UNSUPPORTED_OPERATION", "Root is not a primitive", "root")
            identifier_map = args.get("identifier_map") or None
            group_map = args.get("group_map") or None
            try:
                mapping = staged.copy_primitive_tree(
                    root, staged, preserve_ids=False,
                    identifier_map=identifier_map, group_map=group_map,
                    include_mops=args["include_mops"])
            except (TypeError, ValueError) as exc:
                raise DomainError("INVALID_ARGUMENT", str(exc)[:512], "root") from None
            self._check_limits(staged)
            data = {"mapping": {str(source): str(target)
                                for source, target in mapping.items()}}
        elif name == "machining_add_profile":
            part, targets = self._stage_mop_prelude(staged, args, "profile")
            mop = staged.add_profile_mop(
                part, targets=targets, identifier=args["identifier"], name=args["identifier"],
                enabled=args["enabled"], target_depth=args["target_depth"],
                depth_increment=args["depth_increment"], stock_surface=args["stock_surface"],
                roughing_clearance=0.0, clearance_plane=args["clearance_plane"],
                spindle_direction="CW", spindle_speed=args["spindle_speed"],
                velocity_mode="ExactStop", work_plane="XY", optimisation_mode="Standard",
                tool_diameter=args["tool_diameter"], tool_number=0, tool_profile="EndMill",
                plunge_feedrate=args["plunge_feedrate"], cut_feedrate=args["cut_feedrate"],
                max_crossover_distance=0.7, custom_mop_header="", custom_mop_footer="",
                stepover=0.4, profile_side=args["side"], milling_direction="Conventional",
                collision_detection=True, corner_overcut=False, lead_in_type="None",
                lead_in_spiral_angle=30.0, final_depth_increment=0.0,
                cut_ordering="DepthFirst", tab_method="None", tab_width=6.0,
                tab_height=1.5, tab_min_tabs=3, tab_max_tabs=3, tab_distance=40.0,
                tab_size_threshold=4.0, tab_use_leadins=False, tab_style="Square",
            )
            if mop is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected Profile creation")
            self._check_limits(staged)
            data = {"mop_id": str(mop.internal_id), "part": args["part"],
                    "targets": [str(value) for value in staged.get_mop_targets(mop)]}
        elif name == "machining_add_pocket":
            part, targets = self._stage_mop_prelude(staged, args, "pocket")
            mop = staged.add_pocket_mop(
                part, targets=targets, identifier=args["identifier"], name=args["identifier"],
                enabled=args["enabled"], target_depth=args["target_depth"],
                depth_increment=args["depth_increment"], stock_surface=args["stock_surface"],
                roughing_clearance=0.0, clearance_plane=args["clearance_plane"],
                spindle_direction="CW", spindle_speed=args["spindle_speed"],
                velocity_mode="ExactStop", work_plane="XY", optimisation_mode="Standard",
                tool_diameter=args["tool_diameter"], tool_number=0, tool_profile="EndMill",
                plunge_feedrate=args["plunge_feedrate"], cut_feedrate=args["cut_feedrate"],
                max_crossover_distance=0.7, custom_mop_header="", custom_mop_footer="",
                stepover=0.4, stepover_feedrate="Plunge Feedrate",
                milling_direction="Conventional", collision_detection=True,
                lead_in_type="Spiral", lead_in_spiral_angle=30.0,
                final_depth_increment=0.0, cut_ordering="DepthFirst",
                region_fill_style="InsideOutsideOffsets", finish_stepover=0.0,
                finish_stepover_at_target_depth=False, roughing_finishing="Roughing",
            )
            if mop is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected Pocket creation")
            self._check_limits(staged)
            data = {"mop_id": str(mop.internal_id), "part": args["part"],
                    "targets": [str(value) for value in staged.get_mop_targets(mop)]}
        elif name == "machining_add_engrave":
            part, targets = self._stage_mop_prelude(staged, args, "engrave")
            mop = staged.add_engrave_mop(
                part, targets=targets, identifier=args["identifier"], name=args["identifier"],
                enabled=args["enabled"], target_depth=args["target_depth"],
                depth_increment=args["depth_increment"], stock_surface=args["stock_surface"],
                roughing_clearance=0.0, clearance_plane=args["clearance_plane"],
                spindle_direction="CW", spindle_speed=args["spindle_speed"],
                velocity_mode="ExactStop", work_plane="XY", optimisation_mode="Standard",
                tool_diameter=args["tool_diameter"], tool_number=0, tool_profile="EndMill",
                plunge_feedrate=args["plunge_feedrate"], cut_feedrate=args["cut_feedrate"],
                max_crossover_distance=0.7, custom_mop_header="", custom_mop_footer="",
                roughing_finishing="Roughing", final_depth_increment=0.0,
                cut_ordering="DepthFirst",
            )
            if mop is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected Engrave creation")
            self._check_limits(staged)
            data = {"mop_id": str(mop.internal_id), "part": args["part"],
                    "targets": [str(value) for value in staged.get_mop_targets(mop)]}
        elif name == "machining_add_drill":
            part, targets = self._stage_mop_prelude(staged, args, "drill")
            mop = staged.add_drill_mop(
                part, targets=targets, identifier=args["identifier"], name=args["identifier"],
                enabled=args["enabled"], target_depth=args["target_depth"],
                depth_increment=args["depth_increment"], stock_surface=args["stock_surface"],
                roughing_clearance=0.0, clearance_plane=args["clearance_plane"],
                spindle_direction="CW", spindle_speed=args["spindle_speed"],
                velocity_mode="ExactStop", work_plane="XY", optimisation_mode="Standard",
                tool_diameter=args["tool_diameter"], tool_number=0, tool_profile="Drill",
                plunge_feedrate=args["plunge_feedrate"], cut_feedrate=args["cut_feedrate"],
                max_crossover_distance=0.7, custom_mop_header="", custom_mop_footer="",
                drilling_method="CannedCycle", peck_distance=args["peck_distance"],
                retract_height=args["retract_height"], dwell=args["dwell"],
                hole_diameter=None, drill_lead_out=False, spiral_flat_base=True,
                lead_out_length=0.0, custom_script="",
            )
            if mop is None:
                raise DomainError("INTERNAL_ERROR", "Framework rejected Drill creation")
            self._check_limits(staged)
            data = {"mop_id": str(mop.internal_id), "part": args["part"],
                    "targets": [str(value) for value in staged.get_mop_targets(mop)]}
        elif name == "machining_set_mop_targets":
            mop_entity = staged.get_entity(UUID(args["mop_id"]))
            if mop_entity is None:
                raise DomainError("ENTITY_NOT_FOUND", "MOP was not found", "mop_id")
            if not isinstance(mop_entity, Mop):
                raise DomainError("UNSUPPORTED_OPERATION", "Entity is not a supported MOP", "mop_id")
            kind = self.MOP_KIND_BY_CLASS[type(mop_entity)]
            targets = [self._require_mop_target(staged, target, kind)
                       for target in args["targets"]]
            staged.set_mop_targets(mop_entity.internal_id,
                                   [target.internal_id for target in targets])
            data = {"mop_id": args["mop_id"],
                    "targets": [str(value) for value in staged.get_mop_targets(mop_entity)]}
        return staged, data

    async def _edit(self, name, args, document, entry):
        staged, data = await anyio.to_thread.run_sync(
            self._stage_edit, name, args, document.project
        )
        await anyio.lowlevel.checkpoint()
        revision = document.revision + 1
        if revision > 9007199254740991:
            raise DomainError("LIMIT_EXCEEDED", "Document revision limit reached")
        result = self._envelope(args, data=data, revision=revision)
        OUTPUTS[name].validate(result)
        with anyio.CancelScope(shield=True):
            document.project = staged
            document.revision = revision
            return self._complete(name, entry, result)

    def _stage_cross_edit(self, name, args, source_project, target_project):
        """Validate and stage one two-document subtree operation on clones.

        Copy publishes into a staged clone of the target while reading the live
        source (the public copy operation never mutates it).  Transfer stages
        both documents and runs the public transfer between the clones, so
        source removal and target insertion commit as one framework operation.
        The live documents are replaced only by the caller after all checks.
        """
        root = source_project.get_entity(UUID(args["root"]))
        if root is None:
            raise DomainError("ENTITY_NOT_FOUND", "Root was not found", "root")
        if not isinstance(root, Primitive):
            raise DomainError("UNSUPPORTED_OPERATION", "Root is not a primitive", "root")
        transfer = name == "relationship_transfer_tree_between"
        staged_source = source_project.clone() if transfer else None
        staged_target = target_project.clone()
        try:
            if transfer:
                mapping = staged_source.transfer_primitive_tree(
                    root, staged_target, preserve_ids=False,
                    identifier_map=args.get("identifier_map") or None,
                    group_map=args.get("group_map") or None,
                    include_mops=args["include_mops"])
            else:
                mapping = source_project.copy_primitive_tree(
                    root, staged_target, preserve_ids=False,
                    identifier_map=args.get("identifier_map") or None,
                    group_map=args.get("group_map") or None,
                    include_mops=args["include_mops"])
        except (TypeError, ValueError) as exc:
            raise DomainError("INVALID_ARGUMENT", str(exc)[:512], "root") from None
        self._check_limits(staged_target)
        data = {"mapping": {str(source): str(target)
                            for source, target in mapping.items()}}
        return staged_source, staged_target, data

    async def _cross_edit(self, name, args, entry):
        source_handle = args["source_document"]
        target_handle = args["target_document"]
        if source_handle == target_handle:
            raise DomainError("INVALID_ARGUMENT",
                              "Source and target documents must be different",
                              "target_document")
        for handle, field in ((source_handle, "source_document"),
                              (target_handle, "target_document")):
            if handle.split(":")[0] != self.bootstrap["boot_id"]:
                raise DomainError("DOCUMENT_EXPIRED",
                                  "Document belongs to a previous server process",
                                  field, document=handle)
        source = self.documents.get(source_handle)
        if source is None:
            raise DomainError("DOCUMENT_NOT_FOUND", "Document is not open",
                              "source_document", document=source_handle)
        target = self.documents.get(target_handle)
        if target is None:
            raise DomainError("DOCUMENT_NOT_FOUND", "Document is not open",
                              "target_document", document=target_handle)
        transfer = name == "relationship_transfer_tree_between"
        # Canonical lock order prevents deadlock; no other path holds two
        # document locks, and nothing waits on a document lock while holding
        # the registry or ledger lock.
        handles = {source_handle: source, target_handle: target}
        first, second = sorted((source_handle, target_handle))
        async with handles[first].lock, handles[second].lock:
            for handle, document, field in (
                    (source_handle, source, "source_document"),
                    (target_handle, target, "target_document")):
                if self.documents.get(handle) is not document:
                    raise DomainError("DOCUMENT_NOT_FOUND", "Document is not open",
                                      field, document=handle)
            if args["source_expected_revision"] != source.revision:
                raise DomainError("STALE_REVISION", "Expected source revision does not match",
                                  "source_expected_revision", document=source_handle)
            if args["target_expected_revision"] != target.revision:
                raise DomainError("STALE_REVISION", "Expected target revision does not match",
                                  "target_expected_revision", document=target_handle)
            staged_source, staged_target, data = await anyio.to_thread.run_sync(
                self._stage_cross_edit, name, args, source.project, target.project
            )
            await anyio.lowlevel.checkpoint()
            source_revision = source.revision + int(transfer)
            target_revision = target.revision + 1
            if max(source_revision, target_revision) > 9007199254740991:
                raise DomainError("LIMIT_EXCEEDED", "Document revision limit reached")
            data.update({"source_document": source_handle,
                         "target_document": target_handle,
                         "source_revision": source_revision,
                         "target_revision": target_revision})
            result = self._envelope(args, data=data, document=source_handle,
                                    revision=source_revision)
            OUTPUTS[name].validate(result)
            with anyio.CancelScope(shield=True):
                # No await between these assignments and completion: the
                # two-document publication cannot be split by cancellation.
                if staged_source is not None:
                    source.project = staged_source
                    source.revision = source_revision
                target.project = staged_target
                target.revision = target_revision
                return self._complete(name, entry, result)

    PROFILE_XML_PATHS = {
        "target_depth": ("TargetDepth",), "depth_increment": ("DepthIncrement",),
        "tool_diameter": ("ToolDiameter",), "cut_feedrate": ("CutFeedrate",),
        "plunge_feedrate": ("PlungeFeedrate",), "spindle_speed": ("SpindleSpeed",),
        "stock_surface": ("StockSurface",), "clearance_plane": ("ClearancePlane",),
        "profile_side": ("InsideOutside",), "work_plane": ("WorkPlane",),
        "tool_profile": ("ToolProfile",), "spindle_direction": ("SpindleDirection",),
        "velocity_mode": ("VelocityMode",), "milling_direction": ("MillingDirection",),
        "roughing_clearance": ("RoughingClearance",), "stepover": ("StepOver",),
        "tool_number": ("ToolNumber",), "collision_detection": ("CollisionDetection",),
        "corner_overcut": ("CornerOvercut",),
        "final_depth_increment": ("FinalDepthIncrement",),
        "cut_ordering": ("CutOrdering",),
        "lead_in_type": ("LeadInMove", "LeadInType"),
        "tab_method": ("HoldingTabs", "TabMethod"),
        "custom_mop_header": ("CustomMOPHeader",),
        "custom_mop_footer": ("CustomMOPFooter",),
    }
    POCKET_XML_PATHS = {
        "target_depth": ("TargetDepth",), "depth_increment": ("DepthIncrement",),
        "tool_diameter": ("ToolDiameter",), "cut_feedrate": ("CutFeedrate",),
        "plunge_feedrate": ("PlungeFeedrate",), "spindle_speed": ("SpindleSpeed",),
        "stock_surface": ("StockSurface",), "clearance_plane": ("ClearancePlane",),
        "work_plane": ("WorkPlane",), "tool_profile": ("ToolProfile",),
        "spindle_direction": ("SpindleDirection",), "velocity_mode": ("VelocityMode",),
        "roughing_clearance": ("RoughingClearance",), "tool_number": ("ToolNumber",),
        "custom_mop_header": ("CustomMOPHeader",), "custom_mop_footer": ("CustomMOPFooter",),
        "stepover": ("StepOver",), "stepover_feedrate": ("StepoverFeedrate",),
        "milling_direction": ("MillingDirection",),
        "collision_detection": ("CollisionDetection",),
        "lead_in_type": ("LeadInMove", "LeadInType"),
        "final_depth_increment": ("FinalDepthIncrement",),
        "cut_ordering": ("CutOrdering",), "region_fill_style": ("RegionFillStyle",),
        "finish_stepover": ("FinishStepover",),
        "finish_stepover_at_target_depth": ("FinishStepoverAtTargetDepth",),
        "roughing_finishing": ("RoughingFinishing",),
    }
    ENGRAVE_XML_PATHS = {
        "target_depth": ("TargetDepth",), "depth_increment": ("DepthIncrement",),
        "tool_diameter": ("ToolDiameter",), "cut_feedrate": ("CutFeedrate",),
        "plunge_feedrate": ("PlungeFeedrate",), "spindle_speed": ("SpindleSpeed",),
        "stock_surface": ("StockSurface",), "clearance_plane": ("ClearancePlane",),
        "work_plane": ("WorkPlane",), "tool_profile": ("ToolProfile",),
        "spindle_direction": ("SpindleDirection",), "velocity_mode": ("VelocityMode",),
        "roughing_clearance": ("RoughingClearance",), "tool_number": ("ToolNumber",),
        "custom_mop_header": ("CustomMOPHeader",), "custom_mop_footer": ("CustomMOPFooter",),
        "roughing_finishing": ("RoughingFinishing",),
        "final_depth_increment": ("FinalDepthIncrement",),
        "cut_ordering": ("CutOrdering",),
    }
    DRILL_XML_PATHS = {
        "target_depth": ("TargetDepth",), "depth_increment": ("DepthIncrement",),
        "tool_diameter": ("ToolDiameter",), "cut_feedrate": ("CutFeedrate",),
        "plunge_feedrate": ("PlungeFeedrate",), "spindle_speed": ("SpindleSpeed",),
        "stock_surface": ("StockSurface",), "clearance_plane": ("ClearancePlane",),
        "work_plane": ("WorkPlane",), "tool_profile": ("ToolProfile",),
        "spindle_direction": ("SpindleDirection",), "velocity_mode": ("VelocityMode",),
        "roughing_clearance": ("RoughingClearance",), "tool_number": ("ToolNumber",),
        "custom_mop_header": ("CustomMOPHeader",), "custom_mop_footer": ("CustomMOPFooter",),
        "drilling_method": ("DrillingMethod",), "peck_distance": ("PeckDistance",),
        "retract_height": ("RetractHeight",), "dwell": ("Dwell",),
    }

    @staticmethod
    def _common_mop_scalars_ok(values):
        return (
            type(values["enabled"]) is bool
            and all(type(values[key]) in (int, float) and math.isfinite(values[key])
                    for key in ("target_depth", "depth_increment", "tool_diameter",
                                "cut_feedrate", "plunge_feedrate", "stock_surface",
                                "clearance_plane"))
            and type(values["spindle_speed"]) is int
            and values["target_depth"] < values["stock_surface"]
            and values["clearance_plane"] > values["stock_surface"]
            and all(values[key] > 0 for key in ("depth_increment", "tool_diameter",
                                                "cut_feedrate", "plunge_feedrate"))
            and 1 <= values["spindle_speed"] <= 1_000_000
        )

    @staticmethod
    def _explicit_mop_xml_states(mop, paths):
        template = getattr(mop, "_xml_template", None)
        if template is None:
            return True
        for path in paths.values():
            current = template
            saw_value_state = False
            for tag in path:
                current = current.find(tag)
                if current is None or current.get("state") == "Default":
                    return False
                saw_value_state = saw_value_state or current.get("state") == "Value"
            if not saw_value_state:
                return False
        return True

    @classmethod
    def _mop_record_values(cls, mop, mop_class, fields, fixed, unexposed_fixed, extra=None):
        if not isinstance(mop, mop_class):
            return None
        values = {field: getattr(mop, field) for field in fields}
        if (any(values[key] != expected for key, expected in fixed.items())
                or any(getattr(mop, key) != expected
                       for key, expected in unexposed_fixed.items())):
            return None
        if not cls._common_mop_scalars_ok(values):
            return None
        if extra is not None and not extra(values):
            return None
        return values

    def _profile_parameters(self, mop):
        values = self._mop_record_values(
            mop, ProfileMop,
            (
                "target_depth", "depth_increment", "tool_diameter", "cut_feedrate",
                "plunge_feedrate", "spindle_speed", "stock_surface", "clearance_plane",
                "enabled", "profile_side", "work_plane", "tool_profile",
                "spindle_direction", "velocity_mode", "milling_direction",
                "roughing_clearance", "stepover", "tool_number", "collision_detection",
                "corner_overcut", "final_depth_increment", "cut_ordering", "lead_in_type",
                "tab_method", "custom_mop_header", "custom_mop_footer",
            ),
            {
                "work_plane": "XY", "tool_profile": "EndMill", "spindle_direction": "CW",
                "velocity_mode": "ExactStop", "milling_direction": "Conventional",
                "roughing_clearance": 0.0, "stepover": 0.4, "tool_number": 0,
                "collision_detection": True, "corner_overcut": False,
                "final_depth_increment": 0.0, "cut_ordering": "DepthFirst",
                "lead_in_type": "None", "tab_method": "None",
                "custom_mop_header": "", "custom_mop_footer": "",
            },
            {
                "optimisation_mode": "Standard", "max_crossover_distance": 0.7,
                "lead_in_spiral_angle": 30.0, "tab_width": 6.0, "tab_height": 1.5,
                "tab_min_tabs": 3, "tab_max_tabs": 3, "tab_distance": 40.0,
                "tab_size_threshold": 4.0, "tab_use_leadins": False,
                "tab_style": "Square",
            },
        )
        if values is None or not self._explicit_mop_xml_states(mop, self.PROFILE_XML_PATHS):
            return None
        return values if _PARAMETER_VALIDATORS["ProfileParameters"].is_valid(values) else None

    def _pocket_parameters(self, mop):
        values = self._mop_record_values(
            mop, PocketMop,
            (
                "target_depth", "depth_increment", "tool_diameter", "cut_feedrate",
                "plunge_feedrate", "spindle_speed", "stock_surface", "clearance_plane",
                "enabled", "work_plane", "tool_profile", "spindle_direction",
                "velocity_mode", "roughing_clearance", "tool_number",
                "final_depth_increment", "cut_ordering", "custom_mop_header",
                "custom_mop_footer", "stepover", "stepover_feedrate",
                "milling_direction", "collision_detection", "lead_in_type",
                "region_fill_style", "finish_stepover",
                "finish_stepover_at_target_depth", "roughing_finishing",
            ),
            {
                "work_plane": "XY", "tool_profile": "EndMill", "spindle_direction": "CW",
                "velocity_mode": "ExactStop", "roughing_clearance": 0.0,
                "tool_number": 0, "final_depth_increment": 0.0,
                "cut_ordering": "DepthFirst", "custom_mop_header": "",
                "custom_mop_footer": "", "stepover": 0.4,
                "stepover_feedrate": "Plunge Feedrate",
                "milling_direction": "Conventional", "collision_detection": True,
                "lead_in_type": "Spiral",
                "region_fill_style": "InsideOutsideOffsets",
                "finish_stepover": 0.0, "finish_stepover_at_target_depth": False,
                "roughing_finishing": "Roughing",
            },
            {
                "optimisation_mode": "Standard", "max_crossover_distance": 0.7,
                "lead_in_spiral_angle": 30.0,
            },
        )
        if values is None or not self._explicit_mop_xml_states(mop, self.POCKET_XML_PATHS):
            return None
        return values if _PARAMETER_VALIDATORS["PocketParameters"].is_valid(values) else None

    def _engrave_parameters(self, mop):
        values = self._mop_record_values(
            mop, EngraveMop,
            (
                "target_depth", "depth_increment", "tool_diameter", "cut_feedrate",
                "plunge_feedrate", "spindle_speed", "stock_surface", "clearance_plane",
                "enabled", "work_plane", "tool_profile", "spindle_direction",
                "velocity_mode", "roughing_clearance", "tool_number",
                "custom_mop_header", "custom_mop_footer", "roughing_finishing",
                "final_depth_increment", "cut_ordering",
            ),
            {
                "work_plane": "XY", "tool_profile": "EndMill", "spindle_direction": "CW",
                "velocity_mode": "ExactStop", "roughing_clearance": 0.0,
                "tool_number": 0, "custom_mop_header": "", "custom_mop_footer": "",
                "roughing_finishing": "Roughing", "final_depth_increment": 0.0,
                "cut_ordering": "DepthFirst",
            },
            {
                "optimisation_mode": "Standard", "max_crossover_distance": 0.7,
            },
        )
        if values is None or not self._explicit_mop_xml_states(mop, self.ENGRAVE_XML_PATHS):
            return None
        return values if _PARAMETER_VALIDATORS["EngraveParameters"].is_valid(values) else None

    @staticmethod
    def _drill_scalars_ok(values):
        return (
            all(type(values[key]) in (int, float) and math.isfinite(values[key])
                and values[key] >= 0 for key in ("peck_distance", "dwell"))
            and type(values["retract_height"]) in (int, float)
            and math.isfinite(values["retract_height"])
        )

    def _drill_parameters(self, mop):
        values = self._mop_record_values(
            mop, DrillMop,
            (
                "target_depth", "depth_increment", "tool_diameter", "cut_feedrate",
                "plunge_feedrate", "spindle_speed", "stock_surface", "clearance_plane",
                "enabled", "peck_distance", "retract_height", "dwell",
                "drilling_method", "tool_profile", "work_plane", "spindle_direction",
                "velocity_mode", "roughing_clearance", "tool_number",
                "custom_mop_header", "custom_mop_footer",
            ),
            {
                "drilling_method": "CannedCycle", "tool_profile": "Drill",
                "work_plane": "XY", "spindle_direction": "CW",
                "velocity_mode": "ExactStop", "roughing_clearance": 0.0,
                "tool_number": 0, "custom_mop_header": "", "custom_mop_footer": "",
            },
            {
                "optimisation_mode": "Standard", "max_crossover_distance": 0.7,
                "hole_diameter": None, "drill_lead_out": False,
                "spiral_flat_base": True, "lead_out_length": 0.0,
                "custom_script": "",
            },
            extra=self._drill_scalars_ok,
        )
        if values is None or not self._explicit_mop_xml_states(mop, self.DRILL_XML_PATHS):
            return None
        return values if _PARAMETER_VALIDATORS["DrillParameters"].is_valid(values) else None

    def _mop_parameters(self, project, mop):
        kind = self.MOP_KIND_BY_CLASS.get(type(mop))
        if kind == "profile":
            parameters = self._profile_parameters(mop)
        elif kind == "pocket":
            parameters = self._pocket_parameters(mop)
        elif kind == "engrave":
            parameters = self._engrave_parameters(mop)
        elif kind == "drill":
            parameters = self._drill_parameters(mop)
        else:
            return None
        if parameters is None:
            return None
        try:
            targets = project.get_mop_targets(mop)
            if (project.get_mop_target_group(mop) is not None or not targets
                    or any(not (entity is not None
                                and self._slice_supported(project, entity)
                                and self._target_allowed(kind, entity))
                           for target in targets
                           for entity in (project.get_primitive(target),))):
                return None
        except (KeyError, TypeError, ValueError):
            return None
        return parameters

    def _inspect(self, document, args):
        project = document.project
        records = [{"kind": "layer", "name": layer.user_identifier} for layer in project.list_layers()]
        records.extend({"kind": "part", "name": part.user_identifier, "enabled": part.enabled,
                        "stock_width": part.stock_width, "stock_height": part.stock_height,
                        "stock_thickness": part.stock_thickness, "stock_material": part.stock_material}
                       for part in project.list_parts())
        for primitive in project.list_primitives():
            parent = project.get_parent_of_primitive(primitive)
            layer = project.get_layer_of_primitive(primitive)
            geometry = None
            if self._slice_supported(project, primitive):
                try:
                    geometry = self._geometry_payload(primitive)
                except DomainError:
                    geometry = None
            records.append({"kind": "primitive", "id": str(primitive.internal_id), "identifier": primitive.user_identifier,
                            "type": type(primitive).__name__, "layer": layer.user_identifier,
                            "parent": str(parent.internal_id) if parent else None,
                            "children": [str(child.internal_id) for child in project.get_children_of_primitive(primitive)],
                            "groups": sorted(primitive.groups),
                            "geometry": geometry})
        for mop in project.list_mops():
            part = project.get_part_of_mop(mop)
            parameters = self._mop_parameters(project, mop)
            records.append({"kind": "mop", "id": str(mop.internal_id), "identifier": mop.user_identifier,
                            "type": type(mop).__name__, "part": part.user_identifier,
                            "targets": [str(uid) for uid in project.get_mop_targets(mop)],
                            "parameters": parameters or {}})
        offset, limit = args["offset"], args["limit"]
        page = records[offset:offset + limit]
        diagnostics = []
        if any((record["kind"] == "primitive" and record["geometry"] is None)
               or (record["kind"] == "mop" and not record["parameters"])
               for record in page):
            diagnostics.append({"code": "INSPECTION_UNSUPPORTED", "message": "Some entity details are outside the supported geometry/MOP inspection slice."})
        return {"summary": self._summary(document), "offset": offset,
                "next_offset": offset + limit if offset + limit < len(records) else None, "entities": page}, diagnostics
