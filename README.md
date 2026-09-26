# CamBam Builder

A Python framework for building CamBam CAD/CAM files. The modern package exposes
`CamBamProject` (also `CBProject`) for entities, relationships, transforms and XML
I/O. It is work in progress; round-trip fidelity and CamBam acceptance are not
established for all supported entities.

The intended framework combines faithful, editable CamBam interchange with an
independent CAM core for toolpaths, evolving stock and rest analysis. Native and
framework-generated machining can be composed through explicit motion evidence.
Path strategies, geometry/stock backends and controller adapters have separate
contracts so future surface/volume methods can extend the same foundation. See
the [framework direction](docs/structure_spec.md#framework-direction-and-extension-principles)
for the design and its current implementation limits.

Start with the [documentation map](docs/README.md), then the relevant topic:

- [Development and verification](docs/DEVELOPMENT.md)
- [Current status and priority](docs/PROGRESS.md)
- [Implemented architecture and target design](docs/structure_spec.md)
- [Agent operating agreement](AGENTS.md)

```python
from cambam_builder import CBProject

project = CBProject('example')
layer = project.add_layer('Geometry')
project.add_rect(layer=layer, identifier='outline', width=100, height=50)
```

See `demos/` for larger examples and the runbook for their limitations.
Licensed under the [MIT license](LICENSE).

Document-independent CAM calculations and planning live under
`cambam_builder.cam_core`; native CamBam entities/XML and the MCP adapter remain
separate. The first generated roughing/cleanup slice is `cam_core.rc01`.
A validated family of straight variable-depth V grooves supports detached
planning and edited native inputs. The accepted baseline and one edited member
also have verified headless G-code reference output. The nominal RC01
roughing/cleanup job also has a parsed and stock-replayed headless reference
file. Their strict millimetre dialect is not a machine controller profile.
The first bounded UCCNC profile emits an independently decoded, stock-replayed
synthetic T1/T3 pair. A Grbl v1.1 portability fixture now checks in-program
manual and modeled automatic transitions through the same stock verifier;
the reusable ordered-job API now accepts caller-supplied stages, raster or
offset V plans, and supported native-normalized linear motion for UCCNC or
Grbl output with decoded stock evidence. Runtime and physical setup are
unassessed. See the
[current status](docs/PROGRESS.md#active-work-and-next-priority).

For development, install the environment with `uv sync`, then run the
project through `.venv\Scripts\python.exe`. Python 3.9 through 3.13 are verified;
see the [development runbook](docs/DEVELOPMENT.md#environment-and-setup).

The optional local MCP server exposes thirty-seven document, planning and authoring
tools for AI clients, including verified geometry authoring (Rect, Circle,
Arc, Pline, Points, Text, Region), Profile/Pocket/Engrave/Drill MOPs with
target replacement, open-Pline Profile side diagnostics, automatic Profile holding
tabs, Text-targeted Profile/Pocket/Engrave, signed Profile/Pocket/Engrave roughing
clearance, CannedCycle and SpiralMill CW/CCW Drill authoring, and VCutter-path
engraving, similarity transforms with baking, and parent/group/copy
relationships including cross-document subtree copy and transfer between two
open documents. Part stock coordinates and native Grid/IsoGrid nesting are explicitly
configurable; imported valid Manual/PointList nesting is inspectable and preserved. Stock
inspection reports both Part-local offsets and the derived drawing-space stock origin. Layer
display settings are also configurable. Complete UTF-8 `.cb` XML can be imported across hosts. On the normal
same-host path, `document_list` exposes the shared staging directory so a client can
binary-copy an existing or manually edited `.cb` there and hash-guard `document_open`;
this avoids model-mediated XML. Export returns inline content and explicitly creates
no file; same-host clients can instead copy an exact server-generated `document_save`
handoff artifact and verify its SHA-256. Install
with `uv sync --extra mcp` on Python 3.10+;
see [MCP setup and client configuration](docs/DEVELOPMENT.md#local-mcp-setup-and-verification).
Clean Windows installation, an agentic OpenCode stdio workflow and CamBam Plus 1.0
domain acceptance are verified for the bounded Rect/Outside-Profile A/B slice.

For a new AI-assisted CamBam project, copy the packaged
[`consumer_AGENTS.template.md`](cambam_builder/mcp_adapter/consumer_AGENTS.template.md)
to the project root as `AGENTS.md`. Its one-time section elicits project defaults and
then replaces itself with concise daily-work instructions. The explicit pending gate
uses one separately answerable prompt per durable policy and defers document-specific
inputs until the held first task resumes; it does not alter this repository's
development-agent policy. Its daily workflow permits deterministic helper scripts for
complex calculations while keeping `.cb` authoring and state in the MCP adapter.

## Add to OpenCode (Windows)

Run these commands from the cloned repository. The MCP workspace is a private
server-side working directory; choose any existing absolute directory you control
and keep it separate from the OpenCode project where client files are created.

```powershell
uv sync --extra mcp
$python = (Resolve-Path ".venv\Scripts\python.exe").Path
$workspace = Join-Path $env:LOCALAPPDATA "CamBam-Builder\McpWorkspace"
New-Item -ItemType Directory -Force $workspace | Out-Null
opencode.cmd mcp add cambam -- "$python" -m cambam_builder.mcp_adapter --workspace "$workspace"
opencode.cmd mcp list
```

Expect `cambam connected`. OpenCode now starts and stops the local stdio server
automatically. The server publishes its operating guidance and individual tool
contracts to the connected MCP client.
