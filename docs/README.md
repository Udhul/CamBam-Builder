# Documentation and topic ownership

Read this map after the root README; follow only the relevant owner.

| Topic | Authoritative home | Boundary |
| --- | --- | --- |
| Agent operational rules | [AGENTS.md](../AGENTS.md) | Compact mandatory context |
| Current architecture, code ownership and intended domain relationships | [structure_spec.md](../structure_spec.md) | Section 0: implemented structure; remaining sections: target design |
| Current baseline, priority, backlog, blockers | [PROGRESS.md](PROGRESS.md) | Single status surface; pending items live here |
| Development commands and troubleshooting | [DEVELOPMENT.md](DEVELOPMENT.md) | Setup, checks and artifact handling |
| Delegation, lifecycle and handoff | [WORKFLOW.md](WORKFLOW.md) | Reusable working procedures |
| Review evidence and rejected approaches | [REVIEW.md](REVIEW.md) | Dated findings, acceptance evidence and reopening conditions; not a second backlog |
| Future local MCP requirements and implementation outline | [MCP_PLAN.md](MCP_PLAN.md) | Pending design/acceptance; priority stays in PROGRESS |
| Region and all-shape Z-coordinate parity | [SHAPE_PARITY_PLAN.md](SHAPE_PARITY_PLAN.md) | Independent upstream feature support; priority stays in PROGRESS |
| Future rest machining and V-cutter planning | [REST_MACHINING_PLAN.md](REST_MACHINING_PLAN.md) | Optional feature proposal and acceptance; low priority stays in PROGRESS |
| Package metadata and dependency declarations | `pyproject.toml`, `setup.py`, `requirements.txt` | Metadata, dependency loading adapter, dependency list respectively |
| Executable API behavior | `cambam_builder/` and future regression tests | Actual implementation; document divergences from target explicitly |
| License | [LICENSE](../LICENSE) | MIT terms; does not authorize external processing of user data |

## Code and artifact boundaries

Runtime module ownership and data flow live in
[the implemented architecture](../structure_spec.md#0-implemented-architecture-and-change-ownership).

- `legacy_cambam_builder/`: separately packaged legacy code; changes require an
  explicit legacy scope. `inactive/`: historical implementation, not runtime owner.
- `demos/`: authored examples, not a regression suite. `output/`, `__pycache__/`
  and `*.egg-info` are generated/disposable; do not treat them as source or erase
  existing contents without authorization. User input files are private data.

No issue tracker, CI configuration, pre-existing decision log or dedicated test suite
was found during initial discovery. This documentation now provides a review record.
An external tracker may exist; if one is
designated, move task ownership there and retain only baseline/active links in
[PROGRESS.md](PROGRESS.md). Do not mirror issue descriptions.

Completed plans must move lasting contracts to the specification or runbook and
retain only useful evidence in the review record. Search direct repository content
first; indexing/RAG requires measured discovery failures and a maintenance owner.

## Maintenance rules

The author of a change updates its owning guide in the same increment; the lead
reviews consistency with code and status. Link to facts elsewhere rather than
copying them. Keep current contracts in the specification, dated evidence in the
review, and live priority in status. The MCP plan captures the requested future
integration; add further linked plans only if acceptance/dependencies outgrow a bounded status item.
The plan owns execution detail, while status retains the single priority order.

One root agent file covers this small, coupled library. Add nested instructions
only when a subtree gains distinct operational requirements that meaningfully
reduce irrelevant context; ordinary module knowledge stays in the specification.
