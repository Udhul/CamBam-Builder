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
    "geometry_add_rectangle",
    "geometry_translate",
    "machining_add_profile",
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


def validated_arguments(name, arguments):
    INPUTS[name].validate(arguments)
    result = copy.deepcopy(arguments)
    for key, field in schema(name, "input")["properties"].items():
        if "default" in field:
            result.setdefault(key, field["default"])
    return result


def tool_definitions():
    descriptions = {
        "document_create": "Create an empty volatile document with explicitly asserted units.",
        "document_open": "Open a bounded workspace .cb snapshot; unsupported interchange data may be lost on save.",
        "document_inspect": "Inspect a revision-consistent paginated inventory, including typed Rect geometry and explicit Profile parameters for the supported slice.",
        "document_save": "Save the current revision to a new workspace .cb file. Never overwrites. Units are asserted, not converted.",
        "document_close": "Close a volatile document, explicitly discarding unsaved edits.",
        "geometry_add_rectangle": "Add one axis-aligned root rectangle, creating its named layer when absent.",
        "geometry_translate": "Translate one supported root rectangle in XY without baking its geometry.",
        "machining_add_profile": "Add one explicit Profile operation for supported root rectangles, creating its named part when absent.",
    }
    return [{"name": name, "description": descriptions[name],
             "inputSchema": schema(name, "input"), "outputSchema": schema(name, "output"),
             "annotations": {"openWorldHint": False, "readOnlyHint": name == "document_inspect",
                             "idempotentHint": True,
                             "destructiveHint": name in (
                                 "document_close", "geometry_add_rectangle", "geometry_translate"
                             )}}
            for name in TOOLS]
