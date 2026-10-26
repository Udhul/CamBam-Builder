"""Encode operator text and framework metadata in CamBam's single string Tag.

Legacy objects with only recognized keys and a nonempty ``user_id`` are reserved
for primitive metadata; MOP metadata also requires a valid ``internal_id`` UUID.
An exact legacy signature is inherently ambiguous and wins over operator text.
An invalid optional parent value is ignored without dropping the remaining
metadata, preserving the legacy parentless world-pose recovery behavior.
All other text, including unknown/malformed JSON objects, stays opaque. Callers
can author any text, even a reserved signature, through ``native_tag``: it is
stored inside a versioned envelope alongside the existing identity fields.
"""

import json
import uuid


type TagValue = str | int | list[str] | None
type TagMetadata = dict[str, TagValue]


def decode_tag(text: str | None, *, primitive: bool) -> tuple[TagMetadata, str | None]:
    """Separate a recognized metadata object from unmodified operator text."""
    if text is None:
        return {}, None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    keys = {"user_id", "internal_id"}
    if primitive:
        keys.update({"groups", "description", "parent"})
    versioned = "_cambam_builder" in data
    if versioned:
        keys.update({"_cambam_builder", "native_tag"})
        if (type(data["_cambam_builder"]) is not int or data["_cambam_builder"] != 1
                or not isinstance(data.get("native_tag"), str)):
            return {}, text
    if data.keys() - keys or not isinstance(data.get("user_id"), str) or not data["user_id"]:
        return {}, text
    if "internal_id" in data:
        try:
            if not isinstance(data["internal_id"], str):
                return {}, text
            uuid.UUID(data["internal_id"])
        except ValueError:
            return {}, text
    elif not primitive or versioned:
        return {}, text
    if "groups" in data and (not isinstance(data["groups"], list)
                              or not all(isinstance(item, str) for item in data["groups"])):
        return {}, text
    if "description" in data and not isinstance(data["description"], str):
        return {}, text
    if "parent" in data and data["parent"] is not None:
        parent = data["parent"]
        if not isinstance(parent, (str, int)) or isinstance(parent, bool):
            data["parent"] = None
    note = data.pop("native_tag") if versioned else None
    data.pop("_cambam_builder", None)
    return data, note


def encode_tag(metadata: TagMetadata, native_tag: str | None) -> str:
    """Keep legacy metadata spelling and add an envelope only for operator text."""
    if native_tag is not None and not isinstance(native_tag, str):
        raise ValueError("native_tag must be a string or None")
    data = dict(metadata)
    if native_tag is not None:
        data.update({"_cambam_builder": 1, "native_tag": native_tag})
    return json.dumps(data, separators=(",", ":"))
