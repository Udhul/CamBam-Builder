"""Volatile documents, revision serialization and process-lifetime retry ledger."""
import copy
from dataclasses import dataclass, field
import hashlib
import json
from uuid import uuid4

import anyio
from jsonschema import ValidationError

from cambam_builder import CBProject
from cambam_builder.cambam_reader import CamBamImportLimitError, read_cambam_bytes
from .paths import DomainError, Workspace
from .schema import CONTRACT, OUTPUTS, StrictValidator, TOOLS, validated_arguments


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
            if name != "document_inspect" and request_id is not None and workspace_id is not None:
                ledger_key = (workspace_id, request_id)
                try:
                    signature = json.dumps([name, {k: v for k, v in validated.items() if k != "request_id"}],
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
            if args["workspace_id"] != self.workspace.id:
                raise DomainError("WORKSPACE_MISMATCH", "Workspace ID does not match this server", "workspace_id")
            if name in ("document_create", "document_open"):
                return await self._new(name, args, entry)
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
                if name == "document_close":
                    with anyio.CancelScope(shield=True):
                        async with self.registry_lock:
                            del self.documents[handle]
                        return self._complete(name, entry, self._envelope(args, data={"closed": True}, revision=document.revision))
                return await self._save(name, args, document, entry)
        except anyio.get_cancelled_exc_class():
            # Do not overwrite a completed commit (or another request's entry).
            if owns_entry and entry.result is None:
                self._complete(name, entry, self._envelope(args, error=DomainError("REQUEST_CANCELLED", "Request canceled before publication")))
            raise
        except DomainError as exc:
            result = self._envelope(args, error=exc)
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

    @staticmethod
    def _check_limits(project):
        if len(project.list_primitives()) > 10000 or len(project.list_mops()) > 1000:
            raise DomainError("LIMIT_EXCEEDED", "Document exceeds 10000 primitives or 1000 MOPs")

    def _stage_new(self, name, args):
        if name == "document_create":
            return Document(CBProject(args["name"]), args["units"])
        data = self.workspace.read(args["path"])
        try:
            project = read_cambam_bytes(data, source_name=args["path"], strict=True)
        except CamBamImportLimitError:
            raise DomainError("LIMIT_EXCEEDED", "Document exceeds import resource limits", "path") from None
        except ValueError:
            raise DomainError("IMPORT_FAILED", "Strict XML import failed", "path") from None
        self._check_limits(project)
        return Document(project, args["units"], {"path": args["path"], "sha256": hashlib.sha256(data).hexdigest()})

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
                                    diagnostics=self._diagnostics(document.units, name == "document_open"))
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
            records.append({"kind": "primitive", "id": str(primitive.internal_id), "identifier": primitive.user_identifier,
                            "type": type(primitive).__name__, "layer": layer.user_identifier,
                            "parent": str(parent.internal_id) if parent else None,
                            "children": [str(child.internal_id) for child in project.get_children_of_primitive(primitive)],
                            "world_xyz": None, "bounds": None})
        for mop in project.list_mops():
            part = project.get_part_of_mop(mop)
            records.append({"kind": "mop", "id": str(mop.internal_id), "identifier": mop.user_identifier,
                            "type": type(mop).__name__, "part": part.user_identifier,
                            "targets": [str(uid) for uid in project.get_mop_targets(mop)], "parameters": {}})
        offset, limit = args["offset"], args["limit"]
        page = records[offset:offset + limit]
        diagnostics = []
        if any(record["kind"] in ("primitive", "mop") for record in page):
            diagnostics.append({"code": "INSPECTION_UNSUPPORTED", "message": "4b returns identity/relationships only; typed geometry and MOP parameters are deferred to 4c/4d."})
        return {"summary": self._summary(document), "offset": offset,
                "next_offset": offset + limit if offset + limit < len(records) else None, "entities": page}, diagnostics
