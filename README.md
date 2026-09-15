# CamBam Builder

A Python framework for building CamBam CAD/CAM files. The modern package exposes
`CamBamProject` (also `CBProject`) for entities, relationships, transforms and XML
I/O. It is work in progress; round-trip fidelity and CamBam acceptance are not
established for all supported entities.

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

For development, install the environment with `uv sync`, then run the
project through `.venv\Scripts\python.exe`. Python 3.9 through 3.13 are verified;
see the [development runbook](docs/DEVELOPMENT.md#environment-and-setup).

The optional local MCP server exposes thirty-four document, planning and authoring
tools for AI clients, including verified geometry authoring (Rect, Circle,
Arc, Pline, Points, Text, Region), Profile/Pocket/Engrave/Drill MOPs with
target replacement, similarity transforms with baking, and parent/group/copy
relationships including cross-document subtree copy and transfer between two
open documents. Part stock/nesting and layer display settings are explicitly
configurable. Complete UTF-8 `.cb` XML can be imported from a client and
exported back as a hashed artifact without using server workspace paths. Install
with `uv sync --extra mcp` on Python 3.10+;
see [MCP setup and client configuration](docs/DEVELOPMENT.md#local-mcp-setup-and-verification).
Clean Windows installation and an OpenCode stdio connection are verified;
agentic OpenCode and CamBam domain acceptance remain pending.

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
automatically; use `cambam` tools in an agent prompt. Files in the OpenCode
project are exchanged as content: the agent should use `document_export` and its
normal local write tool, not access or copy files from the MCP workspace.
