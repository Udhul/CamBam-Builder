"""Run the MCP 4c Rect/Profile/save/reopen/translate vertical slice.

Run from the repository root with the MCP extra installed. By default artifacts
are written to a unique ignored ``output/mcp-authoring-demo-*`` workspace.
"""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import anyio

from cambam_builder.mcp_adapter.service import DocumentService


def _request(service, **values):
    return {"workspace_id": service.workspace.id, "request_id": str(uuid4()), **values}


async def _call(service, name, arguments):
    result = await service.call_tool(name, arguments)
    if not result["ok"]:
        raise RuntimeError(f"{name} failed: {result['error']['code']}: {result['error']['message']}")
    return result


async def _run(workspace):
    service = DocumentService(workspace)
    created = await _call(service, "document_create", _request(service, name="slice", units="mm"))
    first = created["document"]
    rectangle = await _call(service, "geometry_add_rectangle", _request(
        service, document=first, expected_revision=0, identifier="outline",
        layer="Geometry", x=0, y=0, width=20, height=10, z=0,
    ))
    entity_id = rectangle["data"]["entity_id"]
    await _call(service, "machining_add_profile", _request(
        service, document=first, expected_revision=1, identifier="profile", part="Part",
        targets=[entity_id], side="Outside", target_depth=-1, depth_increment=0.5,
        tool_diameter=3, cut_feedrate=300, plunge_feedrate=100,
        spindle_speed=12000, stock_surface=0, clearance_plane=5, enabled=True,
    ))
    await _call(service, "document_inspect", {
        "workspace_id": service.workspace.id, "document": first, "expected_revision": 2,
    })
    saved_a = await _call(service, "document_save", _request(
        service, document=first, expected_revision=2, path="A.cb",
    ))
    a_bytes = (workspace / "A.cb").read_bytes()

    opened = await _call(service, "document_open", _request(service, path="A.cb", units="mm"))
    second = opened["document"]
    translation_args = _request(
        service, document=second, expected_revision=0, entity_id=entity_id, dx=5, dy=2,
    )
    translated = await _call(service, "geometry_translate", translation_args)
    replayed = await _call(service, "geometry_translate", translation_args)
    if not replayed["replayed"] or replayed["revision"] != translated["revision"]:
        raise RuntimeError("translation retry did not replay its original result")
    inspected = await _call(service, "document_inspect", {
        "workspace_id": service.workspace.id, "document": second, "expected_revision": 1,
    })
    saved_b = await _call(service, "document_save", _request(
        service, document=second, expected_revision=1, path="B.cb",
    ))

    primitive = next(item for item in inspected["data"]["entities"] if item["kind"] == "primitive")
    expected = [[5.0, 2.0, 0.0], [25.0, 2.0, 0.0],
                [25.0, 12.0, 0.0], [5.0, 12.0, 0.0]]
    if primitive["world_xyz"] != expected:
        raise RuntimeError(f"unexpected translated corners: {primitive['world_xyz']!r}")
    if (workspace / "A.cb").read_bytes() != a_bytes:
        raise RuntimeError("A.cb changed after the reopened document was edited")
    if hashlib.sha256(a_bytes).hexdigest() != saved_a["data"]["sha256"]:
        raise RuntimeError("A.cb hash does not match its save result")

    return {
        "workspace": str(workspace.resolve()),
        "rectangle_id": entity_id,
        "revisions": {"created": 0, "rectangle": 1, "profile_and_A": 2,
                      "reopened": 0, "translated_and_B": 1},
        "A": saved_a["data"], "B": saved_b["data"],
        "world_xyz": primitive["world_xyz"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path,
                        help="new workspace path that does not exist (default: unique path under output)")
    args = parser.parse_args()
    workspace = args.workspace
    if workspace is None:
        output = Path("output")
        output.mkdir(exist_ok=True)
        workspace = output / f"mcp-authoring-demo-{uuid4().hex[:12]}"
    workspace.mkdir()
    workspace = workspace.resolve()
    print(json.dumps(anyio.run(_run, workspace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
