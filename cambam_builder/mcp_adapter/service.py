"""Volatile documents, revision serialization and process-lifetime retry ledger."""
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from uuid import UUID, uuid4

import anyio
from jsonschema import ValidationError

from cambam_builder import CBProject
from cambam_builder.cambam_entities import ProfileMop, Rect
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
    EDIT_TOOLS = frozenset((
        "geometry_add_rectangle", "geometry_translate", "machining_add_profile"
    ))

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
                if name in self.EDIT_TOOLS:
                    return await self._edit(name, args, document, entry)
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

    @staticmethod
    def _identifier_available(project, identifier, field):
        if project.get_entity(identifier) is not None:
            raise DomainError("IDENTIFIER_CONFLICT", "Identifier is already in use", field)

    @staticmethod
    def _slice_rect(project, entity):
        """Return whether a primitive stays inside the root Rect vertical slice."""
        if not isinstance(entity, Rect):
            return False
        if project.get_parent_of_primitive(entity) is not None:
            return False
        if project.get_children_of_primitive(entity) or project.get_groups_of_primitive(entity):
            return False
        matrix = entity.effective_transform
        try:
            translation_only = (
                matrix.shape == (3, 3)
                and all(math.isfinite(float(value)) for row in matrix for value in row)
                and float(matrix[0, 0]) == 1.0 and float(matrix[0, 1]) == 0.0
                and float(matrix[1, 0]) == 0.0 and float(matrix[1, 1]) == 1.0
                and float(matrix[2, 0]) == 0.0 and float(matrix[2, 1]) == 0.0
                and float(matrix[2, 2]) == 1.0
                and float(entity.local_z_offset) == 0.0
                and float(entity.width) > 0.0 and float(entity.height) > 0.0
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return translation_only

    @staticmethod
    def _rect_geometry(entity):
        try:
            world = [[float(value) for value in point]
                     for point in entity.get_absolute_coordinates_xyz()]
            bounds = entity.get_bounding_box()
            result_bounds = [float(bounds.min_x), float(bounds.min_y),
                             float(bounds.max_x), float(bounds.max_y)]
        except (AttributeError, TypeError, ValueError, OverflowError):
            raise DomainError("UNSUPPORTED_OPERATION", "Rectangle geometry is outside the supported slice") from None
        values = [value for point in world for value in point] + result_bounds
        if (len(world) != 4 or any(len(point) != 3 for point in world)
                or not bounds.is_valid()
                or any(not math.isfinite(value) or abs(value) > 1_000_000_000 for value in values)):
            raise DomainError("INVALID_ARGUMENT", "Resulting geometry exceeds the supported coordinate range")
        return world, result_bounds

    def _require_slice_rect(self, project, entity_id, field="entity_id"):
        entity = project.get_entity(UUID(entity_id))
        if entity is None:
            raise DomainError("ENTITY_NOT_FOUND", "Entity was not found", field)
        if not self._slice_rect(project, entity):
            raise DomainError("UNSUPPORTED_OPERATION", "Entity is not a supported root rectangle", field)
        self._rect_geometry(entity)
        return entity

    def _stage_edit(self, name, args, project):
        staged = project.clone()
        if name == "geometry_add_rectangle":
            self._identifier_available(staged, args["identifier"], "identifier")
            layer_entity = staged.get_entity(args["layer"])
            if layer_entity is None and args["identifier"] == args["layer"]:
                raise DomainError("IDENTIFIER_CONFLICT", "Rectangle and new layer identifiers must differ", "identifier")
            if layer_entity is not None and staged.get_layer(args["layer"]) is None:
                raise DomainError("IDENTIFIER_CONFLICT", "Layer name is used by another entity", "layer")
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
        elif name == "geometry_translate":
            rectangle = self._require_slice_rect(staged, args["entity_id"])
            if not staged.translate_primitive(rectangle.internal_id, args["dx"], args["dy"], bake=False):
                raise DomainError("UNSUPPORTED_OPERATION", "Rectangle translation was rejected", "entity_id")
            self._rect_geometry(rectangle)
            data = {"entity_id": str(rectangle.internal_id)}
        else:
            self._identifier_available(staged, args["identifier"], "identifier")
            if args["target_depth"] >= args["stock_surface"]:
                raise DomainError("INVALID_ARGUMENT", "Target depth must be below stock surface", "target_depth")
            if args["clearance_plane"] <= args["stock_surface"]:
                raise DomainError("INVALID_ARGUMENT", "Clearance plane must be above stock surface", "clearance_plane")
            targets = [self._require_slice_rect(staged, target, "targets")
                       for target in args["targets"]]
            part_entity = staged.get_entity(args["part"])
            part = staged.get_part(args["part"])
            if part_entity is None and args["identifier"] == args["part"]:
                raise DomainError("IDENTIFIER_CONFLICT", "Profile and new part identifiers must differ", "identifier")
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

    @staticmethod
    def _profile_parameters(mop):
        fields = (
            "target_depth", "depth_increment", "tool_diameter", "cut_feedrate",
            "plunge_feedrate", "spindle_speed", "stock_surface", "clearance_plane",
            "enabled", "profile_side", "work_plane", "tool_profile",
            "spindle_direction", "velocity_mode", "milling_direction",
            "roughing_clearance", "stepover", "tool_number", "collision_detection",
            "corner_overcut", "final_depth_increment", "cut_ordering", "lead_in_type",
            "tab_method", "custom_mop_header", "custom_mop_footer",
        )
        if not isinstance(mop, ProfileMop):
            return None
        values = {field: getattr(mop, field) for field in fields}
        fixed = {
            "work_plane": "XY", "tool_profile": "EndMill", "spindle_direction": "CW",
            "velocity_mode": "ExactStop", "milling_direction": "Conventional",
            "roughing_clearance": 0.0, "stepover": 0.4, "tool_number": 0,
            "collision_detection": True, "corner_overcut": False,
            "final_depth_increment": 0.0, "cut_ordering": "DepthFirst",
            "lead_in_type": "None", "tab_method": "None",
            "custom_mop_header": "", "custom_mop_footer": "",
        }
        unexposed_fixed = {
            "optimisation_mode": "Standard", "max_crossover_distance": 0.7,
            "lead_in_spiral_angle": 30.0, "tab_width": 6.0, "tab_height": 1.5,
            "tab_min_tabs": 3, "tab_max_tabs": 3, "tab_distance": 40.0,
            "tab_size_threshold": 4.0, "tab_use_leadins": False,
            "tab_style": "Square",
        }
        if (any(values[key] != expected for key, expected in fixed.items())
                or any(getattr(mop, key) != expected
                       for key, expected in unexposed_fixed.items())):
            return None
        if (type(values["enabled"]) is not bool
                or any(type(values[key]) not in (int, float) or not math.isfinite(values[key])
                       for key in ("target_depth", "depth_increment", "tool_diameter",
                                   "cut_feedrate", "plunge_feedrate", "stock_surface",
                                   "clearance_plane"))
                or type(values["spindle_speed"]) is not int
                or values["target_depth"] >= values["stock_surface"]
                or values["clearance_plane"] <= values["stock_surface"]
                or any(values[key] <= 0 for key in ("depth_increment", "tool_diameter",
                                                     "cut_feedrate", "plunge_feedrate"))
                or not 1 <= values["spindle_speed"] <= 1_000_000):
            return None
        if hasattr(mop, "_xml_template"):
            paths = {
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
            template = mop._xml_template
            for path in paths.values():
                current = template
                saw_value_state = False
                for tag in path:
                    current = current.find(tag)
                    if current is None or current.get("state") == "Default":
                        return None
                    saw_value_state = saw_value_state or current.get("state") == "Value"
                if not saw_value_state:
                    return None
        validator = StrictValidator({
            "$ref": "#/$defs/ProfileParameters", "$defs": CONTRACT["$defs"]
        })
        return values if validator.is_valid(values) else None

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
            world_xyz = bounds = None
            if self._slice_rect(project, primitive):
                try:
                    world_xyz, bounds = self._rect_geometry(primitive)
                except DomainError:
                    pass
            records.append({"kind": "primitive", "id": str(primitive.internal_id), "identifier": primitive.user_identifier,
                            "type": type(primitive).__name__, "layer": layer.user_identifier,
                            "parent": str(parent.internal_id) if parent else None,
                            "children": [str(child.internal_id) for child in project.get_children_of_primitive(primitive)],
                            "world_xyz": world_xyz, "bounds": bounds})
        for mop in project.list_mops():
            part = project.get_part_of_mop(mop)
            parameters = self._profile_parameters(mop)
            if parameters is not None:
                try:
                    targets = project.get_mop_targets(mop)
                    if (project.get_mop_target_group(mop) is not None or not targets
                            or any(not self._slice_rect(project, project.get_primitive(target))
                                   for target in targets)):
                        parameters = None
                except (KeyError, TypeError, ValueError):
                    parameters = None
            records.append({"kind": "mop", "id": str(mop.internal_id), "identifier": mop.user_identifier,
                            "type": type(mop).__name__, "part": part.user_identifier,
                            "targets": [str(uid) for uid in project.get_mop_targets(mop)],
                            "parameters": parameters or {}})
        offset, limit = args["offset"], args["limit"]
        page = records[offset:offset + limit]
        diagnostics = []
        if any((record["kind"] == "primitive" and record["world_xyz"] is None)
               or (record["kind"] == "mop" and not record["parameters"])
               for record in page):
            diagnostics.append({"code": "INSPECTION_UNSUPPORTED", "message": "Some entity details are outside the supported Rect/Profile inspection slice."})
        return {"summary": self._summary(document), "offset": offset,
                "next_offset": offset + limit if offset + limit < len(records) else None, "entities": page}, diagnostics
