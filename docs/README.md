# Documentation and topic ownership

Read this map after the root README; follow only the relevant owner.

| Topic | Authoritative home | Boundary |
| --- | --- | --- |
| Agent operational rules | [AGENTS.md](../AGENTS.md) | Compact mandatory context |
| Architecture and intended domain relationships | [structure_spec.md](../structure_spec.md) | Target design, not proof of implemented behavior |
| Current baseline, priority, backlog, blockers | [PROGRESS.md](../PROGRESS.md) | Single status surface; pending items live here |
| Development commands and troubleshooting | [DEVELOPMENT.md](DEVELOPMENT.md) | Setup, checks and artifact handling |
| Delegation, lifecycle and handoff | [WORKFLOW.md](WORKFLOW.md) | Reusable working procedures |
| Review evidence and rejected approaches | [REVIEW.md](REVIEW.md) | Dated findings, acceptance evidence and reopening conditions; not a second backlog |
| Package metadata and dependency declarations | `pyproject.toml`, `setup.py`, `requirements.txt` | Metadata, dependency loading adapter, dependency list respectively |
| Executable API behavior | `cambam_builder/` and future regression tests | Actual implementation; document divergences from target explicitly |
| License | [LICENSE](../LICENSE) | MIT terms; does not authorize external processing of user data |

## Code and artifact boundaries

- `cambam_project.py`: entity registries, relationship updates, project transforms,
  persistence and public orchestration.
- `cambam_entities.py`: dataclasses, primitive geometry and entity XML encoding.
- `cad_transformations.py` and `cad_common.py`: matrix helpers and shared values.
- `cambam_writer.py` / `cambam_reader.py`: XML orchestration and reconstruction.
- `__init__.py`: public project alias and version.
- `legacy_cambam_builder/`: separately packaged legacy code; changes require an
  explicit legacy scope. `inactive/`: historical implementation, not runtime owner.
- `demos/`: authored examples, not a regression suite. `output/`, `__pycache__/`
  and `*.egg-info` are generated/disposable; do not treat them as source or erase
  existing contents without authorization. User input files are private data.

No issue tracker, CI configuration, decision log or dedicated test suite was found
in the tracked repository at this review. An external tracker may exist; if one is
designated, move task ownership there and retain only baseline/active links in
`PROGRESS.md`. Do not mirror issue descriptions.

Completed plans must move lasting contracts to the specification or runbook and
retain only useful evidence in the review record. Search direct repository content
first; indexing/RAG requires measured discovery failures and a maintenance owner.
