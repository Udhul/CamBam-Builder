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

Document-independent CAM planning lives under `cambam_builder.cam_core`, with
native CamBam interchange and output adapters in separate owners. Bounded
rest/V, ordered stock verification, surface and inlay capabilities are described
in the [implemented capability map](docs/structure_spec.md#capability-and-public-api-boundary-map).
The [ordered-job contract](docs/structure_spec.md#reusable-ordered-job-output-and-verification)
defines decoded UCCNC/Grbl evidence and its limits; controller runtime and physical
machining have separate acceptance gates. See [current status and next priority](docs/PROGRESS.md#active-work-and-next-priority)
for delivery state and planned extensions.

Python 3.12 or newer is required. For development, install the environment with
`uv sync --python 3.13`, then run the project through `.venv\Scripts\python.exe`.
The verification matrix covers Python 3.12 and 3.13;
see the [development runbook](docs/DEVELOPMENT.md#environment-and-setup).

The optional local MCP server provides document inspection, geometry/MOP authoring,
transforms and cross-document operations for AI clients. The [MCP contract](docs/MCP_CONTRACT.md)
owns supported tools, native field semantics and file handoff behavior. Install
with `uv sync --python 3.13 --extra mcp`; see
[MCP setup and client configuration](docs/DEVELOPMENT.md#local-mcp-setup-and-verification)
and the [bounded installation/client acceptance record](docs/REVIEW.md#mcp-4e-acceptance-completion---2026-09-21).

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
