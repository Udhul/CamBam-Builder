# CamBam Builder

A Python framework for building CamBam CAD/CAM files. The modern package exposes
`CamBamProject` (also `CBProject`) for entities, relationships, transforms and XML
I/O. It is work in progress; round-trip fidelity and CamBam acceptance are not
established for all supported entities.

Start with the [documentation map](docs/README.md), then the relevant topic:

- [Development and verification](docs/DEVELOPMENT.md)
- [Current status and priority](PROGRESS.md)
- [Implemented architecture and target design](structure_spec.md)
- [Agent operating agreement](AGENTS.md)

```python
from cambam_builder import CBProject

project = CBProject('example')
layer = project.add_layer('Geometry')
project.add_rect(layer=layer, identifier='outline', width=100, height=50)
```

See `demos/` for larger examples and the runbook for their limitations.
Licensed under the [MIT license](LICENSE).
