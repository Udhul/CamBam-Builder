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
    "document_export",
    "document_import",
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
    "relationship_copy_tree_between",
    "relationship_remove_from_group",
    "relationship_set_parent",
    "relationship_transfer_tree_between",
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
        "document_create": "Create an empty volatile document with explicitly asserted units. For a client-local file, finish with document_export and have the client write the returned content in its own project workspace.",
        "document_export": "Default delivery for every new or modified client-local document: return the current revision as complete UTF-8 CamBam .cb XML. The client must write content unchanged to its own project workspace and may verify sha256; no server file is created and absolute server paths are unnecessary.",
        "document_import": "Default input for an existing client-local file: import the complete UTF-8 CamBam .cb XML content read by the client into a new volatile document. source_name is a leaf name, not a server path.",
        "document_open": "SERVER WORKSPACE ONLY: open a bounded .cb snapshot already inside the configured MCP workspace. Do not use for a file in the client's project workspace; have the client read it and use document_import instead.",
        "document_inspect": "Inspect a revision-consistent paginated inventory, including typed geometry for supported primitives and explicit Profile/Pocket/Engrave/Drill parameters.",
        "document_save": "SERVER WORKSPACE ONLY: save to a new .cb file inside the MCP server's configured workspace. Never use this for normal client-local delivery, and never ask the client to access the returned absolute_path; use document_export and the client's own file-write tool instead. Never overwrites. Units are asserted, not converted.",
        "document_close": "Close a volatile document, explicitly discarding unsaved edits.",
        "geometry_add_rectangle": "Add one axis-aligned root rectangle, creating its named layer when absent.",
        "geometry_add_circle": "Add one axis-parallel root circle from an absolute center, diameter and elevation, creating its named layer when absent. The result echoes typed center/diameter/bounds; compare them with the requested placement and containing geometry before machining or export.",
        "geometry_add_arc": "Add one root circular arc from center, radius, CCW-positive degree angles and elevation, creating its named layer when absent.",
        "geometry_add_pline": "Add one root polyline from absolute vertices in listed traversal order. A vertex bulge curves the segment from that vertex to the next (last to first when closed): positive sweeps CCW and lies to the right of the directed chord; negative lies to its left; abs(bulge)=tan(abs(sweep)/4). The result echoes world vertices, bulges and exact curved bounds; verify requested bounds, symmetry and placement before adding dependent geometry, machining or export.",
        "geometry_add_points": "Add one root point list from explicit XYZ points, creating its named layer when absent.",
        "geometry_add_text": "Add one root text annotation with explicit anchor, font and alignment, creating its named layer when absent.",
        "geometry_add_region": "Add one root Region from one closed outer contour plus optional closed hole contours, creating its named layer when absent; XY topology is validated.",
        "geometry_translate": "Translate one supported root primitive in XY without baking its geometry.",
        "geometry_translate_z": "Shift one supported root primitive's stored Z geometry once, keeping world matrices.",
        "geometry_rotate": "Rotate one supported root primitive around an explicit or geometric center; the world pose stays a similarity.",
        "geometry_scale": "Uniformly scale one supported root primitive around an explicit or geometric center; non-uniform scale is unsupported.",
        "geometry_mirror": "Mirror one supported root primitive across an x- or y-parallel line through an explicit or geometric center.",
        "geometry_bake": "Fold one supported root primitive's world transform into its stored geometry and reset its matrix to identity; non-axis-aligned Rects become closed Plines.",
        "machining_add_drill": "Add a canned-cycle Drill operation at Points entries or Circle centers. Choose Drill for drilling center locations, not for tracing a circle boundary. MOPs append in call order; drill enclosed features before an Outside Profile that cuts their containing part loose. The part name must be project-unique and is created when absent.",
        "machining_add_engrave": "Add an Engrave operation whose tool center follows the selected Rect/Circle/Arc/Pline curves with no cutter-radius compensation. Choose it only when centerline tracing or engraving is intended, not as a substitute for dimensionally accurate inside/outside contour cutting. MOPs append in call order; engrave a contained part before cutting that part loose. The part name must be project-unique and is created when absent.",
        "machining_add_pocket": "Add a Pocket operation that clears the entire area inside selected root Rect/Circle/closed-Pline/Region boundaries into chips. Use Pocket for a cavity or when no loose slug should remain; for a through-opening whose interior may be released as a slug, normally use Profile Inside. MOPs append in call order; pocket enclosed features before an Outside Profile that cuts their containing part loose. The part name must be project-unique and is created when absent. Never invent missing depth, feeds or spindle speed; obtain them from the user or project.",
        "machining_add_profile": "Add a cutter-radius-compensated Profile around root Rect/Circle/closed-Pline/Region boundaries. Outside keeps the selected boundary as the finished exterior part edge and cuts in surrounding stock; Inside keeps it as the finished opening edge and cuts on the removable interior side. Use Inside for a through-opening when releasing a slug is intended; use Pocket only to clear its whole area. MOPs append in call order: create enclosed/detail operations first, and if an Outside Profile cuts their containing part loose, create that cutout last. The result echoes side and target IDs: verify them against inspected geometry. The part name must be project-unique and is created when absent. Never invent missing depth, feeds or spindle speed; obtain them from the user or project.",
        "machining_set_mop_targets": "Atomically replace one supported MOP's target selection with explicit primitive targets.",
        "relationship_set_parent": "Link a primitive under a parent, or detach it with a null parent; local transforms are kept, so the world pose follows the new frame.",
        "relationship_add_to_group": "Add a primitive to a named group.",
        "relationship_remove_from_group": "Remove a primitive from a named group.",
        "relationship_copy_tree": "Copy a primitive subtree inside the same document with fresh identities; identifier collisions require an explicit identifier map.",
        "relationship_copy_tree_between": "Copy a primitive subtree from one open document into another in this workspace with fresh identities; both documents keep their handles and the source is unchanged.",
        "relationship_transfer_tree_between": "Move a primitive subtree from one open document into another in this workspace, removing it from the source; both revisions advance together.",
    }
    nondestructive = {"document_create", "document_export", "document_import",
                      "document_inspect", "document_open", "document_save"}
    return [{"name": name, "description": descriptions[name],
             "inputSchema": schema(name, "input"), "outputSchema": schema(name, "output"),
             "annotations": {"openWorldHint": False,
                             "readOnlyHint": name in {"document_export", "document_inspect"},
                             "idempotentHint": True,
                             "destructiveHint": name not in nondestructive}}
            for name in TOOLS]
