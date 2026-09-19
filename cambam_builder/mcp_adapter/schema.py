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
    "document_list",
    "document_open",
    "document_save",
    "geometry_add_arc",
    "geometry_add_circle",
    "geometry_add_pline",
    "geometry_add_points",
    "geometry_add_rectangle",
    "geometry_add_region",
    "geometry_replace_with_region",
    "geometry_update_region",
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
    "machining_calculate_depth_increment",
    "machining_configure_part",
    "machining_set_mop_targets",
    "document_set_layer_properties",
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
        "document_create": "Create an empty volatile document with explicitly asserted units. For client-local delivery, finish with a document_save exact-byte handoff when client and server share a filesystem; otherwise use document_export and write its returned content.",
        "document_export": "INLINE CONTENT ONLY; THIS TOOL CREATES NO FILE IN ANY WORKSPACE. Portable cross-host delivery returns the current revision as complete UTF-8 CamBam .cb XML with delivery=inline_content_only and file_created=false. Never claim the suggested filename exists until the client writes content unchanged and verifies sha256. For a same-host file in the shared MCP workspace, call document_save instead. A hash mismatch means delivery failed and must be corrected; do not accept newline or encoding changes. A valid request_id is accepted for client compatibility but ignored because this read-only call has no request ledger.",
        "document_import": "CROSS-HOST input for an existing client-local file: import the complete current UTF-8 CamBam .cb XML into a new volatile revision-0 document, then use the returned handle and source sha256. On a shared filesystem, avoid model-reconstructed XML: use document_list.workspace_path, copy the source bytes to a fresh workspace-relative .cb name, verify the copy hash, and call document_open with expected_sha256. A CONTENT_MISMATCH means the supplied string is not the file; never fall back to an older workspace file or infer parser incompatibility. source_name is a leaf name, not a server path; importing never updates an older handle. If an older handle also has MCP edits newer than the last delivered artifact, preserve both versions and ask which changes to keep or merge.",
        "document_open": "SERVER WORKSPACE SNAPSHOT: open a bounded .cb file already inside the configured MCP workspace as a new revision-0 handle. On a shared filesystem, stage an existing or manually edited client file by exact binary copy to a fresh name under document_list.workspace_path, verify its hash, and pass that hash as expected_sha256. A mismatch rejects a stale/wrong workspace artifact before parsing; never fall back to a similarly named older file.",
        "document_inspect": "Inspect a revision-consistent paginated inventory, including stable entity IDs, typed geometry for supported primitives and explicit Profile/Pocket/Engrave/Drill parameters. To determine or repeat a manual edit, inspect both a trustworthy baseline and the current snapshot, match stable IDs, and propagate only the measured common delta; absolute placement alone does not prove translation, rotation, alignment or layout intent. Omit expected_revision to discover the current revision after a stale follow-up, then retry a mutation with that revision and a new request_id. A valid request_id is accepted but ignored.",
        "document_list": "Discover this server boot, the absolute shared workspace path, and currently open volatile documents, including each handle, revision, source hash and summary. On a shared filesystem, use workspace_path only to stage an exact client .cb copy under a fresh safe filename before document_open(expected_sha256=...). Use the listing after context loss or reconnect; an empty list or changed boot means a durable file must be staged/opened or imported again. A valid request_id is accepted but ignored.",
        "document_save": "INTERMEDIATE SAME-HOST HANDOFF, NOT CLIENT DELIVERY: atomically save to a new .cb file inside the configured MCP workspace. The result says delivery=server_workspace_handoff, workspace_file_created=true and client_file_created=false. For a client-local task, do not report completion until the artifact is binary-copied to the intended project path and its sha256 is verified. The current handle remains authoritative until that destination is manually changed. To reload a changed client file later, copy it to a fresh workspace name and call document_open(expected_sha256=...). Otherwise use document_export inline. Never overwrites a workspace path and never accepts an arbitrary client path. Units are asserted, not converted.",
        "document_close": "Close a volatile document, explicitly discarding unsaved edits.",
        "geometry_add_rectangle": "Add one axis-aligned root rectangle, creating its named layer when absent. When a layer is created, identifier and layer must be different project-unique names.",
        "geometry_add_circle": "Add one axis-parallel root circle from an absolute center, diameter and elevation, creating its named layer when absent. When a layer is created, identifier and layer must be different project-unique names. The result echoes typed center/diameter/bounds; compare them with the requested placement and containing geometry before machining or export.",
        "geometry_add_arc": "Add one root circular arc from center, radius, CCW-positive degree angles and elevation, creating its named layer when absent. When a layer is created, identifier and layer must be different project-unique names.",
        "geometry_add_pline": "Add one root polyline from absolute vertices in listed traversal order. When a layer is created, identifier and layer must be different project-unique names. A vertex bulge curves the segment from that vertex to the next (last to first when closed): positive sweeps CCW and lies to the right of the directed chord; negative lies to its left; abs(bulge)=tan(abs(sweep)/4). The result echoes world vertices, bulges and exact curved bounds; verify requested bounds, symmetry and placement before adding dependent geometry, machining or export.",
        "geometry_add_points": "Add one root point list from explicit XYZ points, creating its named layer when absent. When a layer is created, identifier and layer must be different project-unique names.",
        "geometry_add_text": "Add one root text annotation with explicit anchor, font and alignment, creating its named layer when absent. When a layer is created, identifier and layer must be different project-unique names.",
        "geometry_add_region": "Add one root Region from one closed outer contour plus optional closed hole contours, creating its named layer when absent. Use a Region when one bounded area has excluded holes/islands; use a simpler Rect, Circle or closed Pline when there are no exclusions. A Pocket targeting the Region clears inside the outer boundary while preserving its hole contours as islands. identifier and layer must be different project-unique names when the layer is new; XY topology is validated.",
        "geometry_replace_with_region": "Atomically replace existing root Rect, Circle or closed-Pline contours on one layer with one Region. outer_id identifies the exterior and hole_ids identify its holes. Sources are removed only after the staged Region and compatible Profile/Pocket target replacements validate; other relationships are rejected.",
        "geometry_update_region": "Atomically replace the outer and hole contours of one existing supported root Region from absolute XYZ/bulge points. Use this when the user asks to change a Region rather than adding a duplicate. The Region UUID, identifier, layer and existing Profile/Pocket targets are preserved; topology is validated before publication. A Pocket targeting the Region preserves the supplied holes as islands.",
        "geometry_translate": "Translate one supported root primitive in XY without baking its geometry.",
        "geometry_translate_z": "Shift one supported root primitive's stored Z geometry once, keeping world matrices.",
        "geometry_rotate": "Rotate one supported root primitive around an explicit or geometric center; the world pose stays a similarity.",
        "geometry_scale": "Uniformly scale one supported root primitive around an explicit or geometric center; non-uniform scale is unsupported.",
        "geometry_mirror": "Mirror one supported root primitive across an x- or y-parallel line through an explicit or geometric center.",
        "geometry_bake": "Fold one supported root primitive's world transform into its stored geometry and reset its matrix to identity; non-axis-aligned Rects become closed Plines.",
        "machining_add_drill": "Add a canned-cycle Drill operation at Points entries or Circle centers. This mutation requires expected_revision set to the current document revision and a fresh request_id. Choose Drill for drilling center locations, not for tracing a circle boundary. MOPs append in call order; drill enclosed features before an Outside Profile that cuts their containing part loose. The part name must be project-unique and is created when absent.",
        "machining_add_engrave": "Add an Engrave operation whose tool center follows the selected Rect/Circle/Arc/Pline curves with no cutter-radius compensation. This mutation requires expected_revision set to the current document revision and a fresh request_id. Choose it only when centerline tracing or engraving is intended, not as a substitute for dimensionally accurate inside/outside contour cutting. MOPs append in call order; engrave a contained part before cutting that part loose. The part name must be project-unique and is created when absent.",
        "machining_add_pocket": "Add a Pocket operation that clears the entire area inside selected root Rect/Circle/closed-Pline/Region boundaries into chips. This mutation requires expected_revision set to the current document revision and a fresh request_id. Use Pocket for a cavity or when no loose slug should remain; for a through-opening whose interior may be released as a slug, normally use Profile Inside. MOPs append in call order; pocket enclosed features before an Outside Profile that cuts their containing part loose. The part name must be project-unique and is created when absent. Do not silently invent machining parameters; elicit confirmation of a reasoned proposal based on known stock, material and tool limits.",
        "machining_add_profile": "Add a cutter-radius-compensated Profile around root Rect/Circle/closed-Pline/Region boundaries. This mutation requires expected_revision set to the current document revision and a fresh request_id. Outside keeps the selected boundary as the finished exterior part edge and cuts in surrounding stock; Inside keeps it as the finished opening edge and cuts on the removable interior side. Use Inside for a through-opening when releasing a slug is intended; use Pocket only to clear its whole area. Set corner_overcut=true only when the user wants CamBam to overcut inside corners for a round endmill; it can remove extra material along adjacent sides. MOPs append in call order: create enclosed/detail operations first, and if an Outside Profile cuts their containing part loose, create that cutout last. For through-cuts, choose a material/tool-safe depth increment whose multiples slightly exceed total depth and whose final pass still cuts meaningful stock before crossing the stock bottom; confirm the proposal with the user. The result echoes side and target IDs: verify them against inspected geometry. The part name must be project-unique and is created when absent. Do not silently invent machining parameters.",
        "machining_calculate_depth_increment": "Calculate a rounded through-cut depth increment from stock thickness, cut-through allowance and either an exact pass count or a maximum material/tool-safe depth increment. Returns every depth below stock surface and reports whether at least one third of the final pass remains in stock. A valid request_id is accepted but ignored. It is not a material/tool safety calculator or a veto: preserve valid user-requested constraints, surface warnings, and confirm the result before adding a MOP.",
        "machining_configure_part": "Create or update a Part's explicit stock definition and native CamBam nesting settings. Set stock dimensions/material before adding MOPs; choose Grid or IsoGrid with rows, columns and spacing to repeat the entire Part's operations, rather than copying geometry. A Part with no MOPs has nothing to nest, so inspect each Part's MOP membership before export. default_spindle_speed is session-only framework context and does not survive XML export/re-import, so set spindle_speed explicitly on every durable MOP. The operation preserves explicit user choices and returns the resolved settings.",
        "machining_set_mop_targets": "Atomically replace one supported MOP's target selection with explicit primitive targets.",
        "document_set_layer_properties": "Create or update a drawing layer's display properties, including color, visibility, alpha, pen width and lock state. This affects drawing presentation only, not machining semantics.",
        "relationship_set_parent": "Link a primitive under a parent, or detach it with a null parent; local transforms are kept, so the world pose follows the new frame.",
        "relationship_add_to_group": "Add a primitive to a named group.",
        "relationship_remove_from_group": "Remove a primitive from a named group.",
        "relationship_copy_tree": "Copy a primitive subtree inside the same document with fresh identities; identifier collisions require an explicit identifier map.",
        "relationship_copy_tree_between": "Copy a primitive subtree from one open document into another in this workspace with fresh identities; both documents keep their handles and the source is unchanged.",
        "relationship_transfer_tree_between": "Move a primitive subtree from one open document into another in this workspace, removing it from the source; both revisions advance together.",
    }
    nondestructive = {"document_create", "document_export", "document_import",
                      "document_inspect", "document_list", "document_open", "document_save",
                      "machining_calculate_depth_increment"}
    revision_mutations = {
        "document_set_layer_properties",
        "geometry_add_arc", "geometry_add_circle", "geometry_add_pline",
        "geometry_add_points", "geometry_add_rectangle", "geometry_add_region",
        "geometry_add_text", "geometry_bake", "geometry_mirror",
        "geometry_replace_with_region", "geometry_rotate", "geometry_scale",
        "geometry_translate", "geometry_translate_z", "geometry_update_region",
        "machining_add_drill", "machining_add_engrave", "machining_add_pocket",
        "machining_add_profile", "machining_configure_part",
        "machining_set_mop_targets", "relationship_add_to_group",
        "relationship_copy_tree", "relationship_remove_from_group",
        "relationship_set_parent",
    }
    sequential_rule = (
        " SAME-DOCUMENT MUTATIONS MUST BE SEQUENTIAL: never call this concurrently "
        "with another mutation on the same document. Each success increments revision; "
        "use that returned revision as the next expected_revision and use a fresh request_id."
    )
    return [{"name": name,
             "description": descriptions[name] + (
                 sequential_rule if name in revision_mutations else ""),
             "inputSchema": schema(name, "input"), "outputSchema": schema(name, "output"),
             "annotations": {"openWorldHint": False,
                             "readOnlyHint": name in {"document_export", "document_inspect", "document_list",
                                                       "machining_calculate_depth_increment"},
                             "idempotentHint": True,
                             "destructiveHint": name not in nondestructive}}
            for name in TOOLS]
