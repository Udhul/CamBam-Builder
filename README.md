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

The optional local MCP server exposes thirty-one document and authoring
tools for AI clients, including verified geometry authoring (Rect, Circle,
Arc, Pline, Points, Text, Region), Profile/Pocket/Engrave/Drill MOPs with
target replacement, similarity transforms with baking, and parent/group/copy
relationships including cross-document subtree copy and transfer between two
open documents. Complete UTF-8 `.cb` XML can be imported from a client and
exported back as a hashed artifact without using server workspace paths. Install
with `uv sync --extra mcp` on Python 3.10+;
see [MCP setup and client configuration](docs/DEVELOPMENT.md#local-mcp-setup-and-verification).
Clean Windows installation and an OpenCode stdio connection are verified;
agentic OpenCode and CamBam domain acceptance remain pending.
