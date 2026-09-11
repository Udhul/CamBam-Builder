"""Packaged v1 schemas for the implemented document and authoring tools."""
import copy
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, validators

CONTRACT = json.loads(Path(__file__).with_name("contract_v1.schema.json").read_text(encoding="utf-8"))
TOOLS = tuple(sorted((
    "document_close",
    "document_create",
    "document_inspect",
    "document_open",
    "document_save",
    "geometry_add_arc",
    "geometry_add_circle",
    "geometry_add_pline",
    "geometry_add_points",
    "geometry_add_rectangle",
    "geometry_add_region",
    "geometry_add_text",
    "geometry_bake",
    "geometry_mirror",
    "geometry_rotate",
    "geometry_scale",
    "geometry_translate",
    "geometry_translate_z",
    "machining_add_drill",
    "machining_add_engrave",
    "machining_add_pocket",
    "machining_add_profile",
    "machining_set_mop_targets",
    "relationship_add_to_group",
    "relationship_copy_tree",
    "relationship_remove_from_group",
    "relationship_set_parent",
)))


def _expand(value):
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        if "$ref" in value:
            result = _expand(CONTRACT["$defs"][value["$ref"].split("/")[-1]])
            result.update({key: _expand(item) for key, item in value.items() if key != "$ref"})
            return result
        return {key: _expand(item) for key, item in value.items()}
    return value


def schema(name, direction):
    result = _expand(CONTRACT["$defs"][name + "_" + direction])
    result["type"] = "object"
    return result


_checker = Draft202012Validator.TYPE_CHECKER.redefine(
    "integer", lambda checker, value: type(value) is int
).redefine("number", lambda checker, value: type(value) in (int, float) and math.isfinite(value))
StrictValidator = validators.extend(Draft202012Validator, type_checker=_checker)
INPUTS = {name: StrictValidator(schema(name, "input")) for name in TOOLS}
OUTPUTS = {name: StrictValidator(schema(name, "output")) for name in TOOLS}


def _apply_defaults(value, field):
    """Expand defaults recursively so retries canonicalize identically."""
    if isinstance(value, list):
        items = field.get("items")
        return [_apply_defaults(item, items or {}) for item in value]
    if isinstance(value, dict):
        result = dict(value)
        properties = field.get("properties") or {}
        for key, sub in properties.items():
            if key in result:
                result[key] = _apply_defaults(result[key], sub)
            elif "default" in sub:
                result[key] = copy.deepcopy(sub["default"])
        return result
    return value


def validated_arguments(name, arguments):
    INPUTS[name].validate(arguments)
    return _apply_defaults(copy.deepcopy(arguments), schema(name, "input"))


def tool_definitions():
    descriptions = {
        "document_create": "Create an empty volatile document with explicitly asserted units.",
        "document_open": "Open a bounded workspace .cb snapshot; unsupported interchange data may be lost on save.",
        "document_inspect": "Inspect a revision-consistent paginated inventory, including typed geometry for supported primitives and explicit Profile/Pocket/Engrave/Drill parameters.",
        "document_save": "Save the current revision to a new workspace .cb file. Never overwrites. Units are asserted, not converted.",
        "document_close": "Close a volatile document, explicitly discarding unsaved edits.",
        "geometry_add_rectangle": "Add one axis-aligned root rectangle, creating its named layer when absent.",
        "geometry_add_circle": "Add one axis-parallel root circle from a center, diameter and elevation, creating its named layer when absent.",
        "geometry_add_arc": "Add one root circular arc from center, radius, CCW-positive degree angles and elevation, creating its named layer when absent.",
        "geometry_add_pline": "Add one root polyline from explicit vertices with optional per-vertex Z and bulge, creating its named layer when absent.",
        "geometry_add_points": "Add one root point list from explicit XYZ points, creating its named layer when absent.",
        "geometry_add_text": "Add one root text annotation with explicit anchor, font and alignment, creating its named layer when absent.",
        "geometry_add_region": "Add one root Region from one closed outer contour plus optional closed hole contours, creating its named layer when absent; XY topology is validated.",
        "geometry_translate": "Translate one supported root primitive in XY without baking its geometry.",
        "geometry_translate_z": "Shift one supported root primitive's stored Z geometry once, keeping world matrices.",
        "geometry_rotate": "Rotate one supported root primitive around an explicit or geometric center; the world pose stays a similarity.",
        "geometry_scale": "Uniformly scale one supported root primitive around an explicit or geometric center; non-uniform scale is unsupported.",
        "geometry_mirror": "Mirror one supported root primitive across an x- or y-parallel line through an explicit or geometric center.",
        "geometry_bake": "Fold one supported root primitive's world transform into its stored geometry and reset its matrix to identity; non-axis-aligned Rects become closed Plines.",
        "machining_add_drill": "Add one explicit canned-cycle Drill operation on supported root Points/Circle primitives, creating its named part when absent.",
        "machining_add_engrave": "Add one explicit Engrave operation tracing supported root Rect/Circle/Arc/Pline curves, creating its named part when absent.",
        "machining_add_pocket": "Add one explicit Pocket operation for supported root Rect/Circle/closed-Pline/Region shapes, creating its named part when absent.",
        "machining_add_profile": "Add one explicit Profile operation for supported root rectangles, creating its named part when absent.",
        "machining_set_mop_targets": "Atomically replace one supported MOP's target selection with explicit primitive targets.",
        "relationship_set_parent": "Link a primitive under a parent, or detach it with a null parent; local transforms are kept, so the world pose follows the new frame.",
        "relationship_add_to_group": "Add a primitive to a named group.",
        "relationship_remove_from_group": "Remove a primitive from a named group.",
        "relationship_copy_tree": "Copy a primitive subtree inside the same document with fresh identities; identifier collisions require an explicit identifier map.",
    }
    nondestructive = {"document_create", "document_inspect", "document_open", "document_save"}
    return [{"name": name, "description": descriptions[name],
             "inputSchema": schema(name, "input"), "outputSchema": schema(name, "output"),
             "annotations": {"openWorldHint": False, "readOnlyHint": name == "document_inspect",
                             "idempotentHint": True,
                             "destructiveHint": name not in nondestructive}}
            for name in TOOLS]
