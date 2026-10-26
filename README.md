# CamBam Builder

An installable Python CAD/CAM library for applications that create geometry,
organize machining operations and export editable CamBam files. The main entry
point is `CamBamProject`, also exported as `CBProject`.

The priority is a faithful, simple native foundation, followed by reusable rest
machining, V-carving and inlay engines. Full CamBam feature coverage is the target;
the current pre-1.0 library supports a bounded subset. General independent G-code
output remains future work. See the [product direction and consumer requirements](docs/structure_spec.md#product-priority-and-library-consumers).

## Start with the library

Python 3.12 or newer is required; development checks cover 3.12 and 3.13.
From a checkout, install into your application's environment:

```powershell
python -m pip install .
```

Install `".[planar]"` when using the Shapely-backed planar engines. The base library
needs neither CamBam installed nor an MCP server for document authoring. This is a
source installation instruction, not a claim that a release is published on PyPI.
For development use [the environment and verification runbook](docs/DEVELOPMENT.md#environment-and-setup).

```python
from cambam_builder import CBProject

project = CBProject("board")
layer = project.add_layer("Geometry")
outline = project.add_rect(layer=layer, identifier="outline", width=100, height=50)
part = project.add_part(
    "Board", stock_width=100, stock_height=50, stock_thickness=12,
    stock_material="example material",
)
project.add_profile_mop(
    part, targets=[outline], identifier="cutout", profile_side="Outside",
    target_depth=-12, depth_increment=3, tool_diameter=6,
)
project.save("board.cb")
```

These are illustrative geometry and machining values; configure the actual job's
parameters. Open the exported file in CamBam for inspection, final toolpath
generation and G-code export. Operator layout edits are part of this supported
intermediate workflow; recompute affected derived paths/stock results after edits.

New documents default to millimeters; use `CBProject("board", units="Inches")`
for imperial dimensions. Import preserves explicit or unspecified units without
numeric conversion. PointList placement follows owned geometry through export,
and operator Tags coexist with framework metadata under the
[native interchange contract](docs/structure_spec.md#native-interchange-integrity-cf01).
The [board example](demos/native_board.py) demonstrates editing and reopening a
complete direct-library job. Unsupported native classes and arbitrary unknown
XML still have limits; the [native inventory](docs/REST_MACHINING_PLAN.md#inventoried-baseline-and-prepared-slice-2026-09-23)
distinguishes current support from the full-coverage goal.

## Capabilities and project documentation

- [Current work, delivered branches and ordered backlog](docs/PROGRESS.md)
- [Public capability map and architecture](docs/structure_spec.md#capability-and-public-api-boundary-map)
- [Documentation ownership and navigation](docs/README.md)
- [Development and verification](docs/DEVELOPMENT.md)
- [Agent operating agreement](AGENTS.md)

Native entity ownership lives in `cambam_builder.native`; detached calculations
live in `cam_core`, compositions in `cam_extensions`, and native integration in
`integrations.cambam`. Existing rest/V, ordered-stock, bounded surface and inlay
capabilities have explicit limits. Their tests and recorded acceptance do not
establish universal 3D, controller-runtime or physical-machining support.

## Optional MCP interface

MCP offers a bounded native-document authoring/inspection interface and is lower
priority than the direct library. It does not expose the advanced engines; its
coverage and maintenance policy live in the [MCP contract](docs/MCP_CONTRACT.md#maintenance-priority-and-coverage).
Install it only for that workflow with `uv sync --python 3.13 --extra mcp`, then
follow [client setup](docs/DEVELOPMENT.md#local-mcp-setup-and-verification).
The packaged [consumer agent template](cambam_builder/mcp_adapter/consumer_AGENTS.template.md)
applies only to projects deliberately choosing MCP, not ordinary library consumers.

Licensed under the [MIT license](LICENSE).
