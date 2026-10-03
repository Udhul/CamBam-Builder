# Documentation and topic ownership

Read this map after the root README; follow only the relevant owner.

| Topic | Authoritative home | Boundary |
| --- | --- | --- |
| Agent operational rules | [AGENTS.md](../AGENTS.md) | Compact mandatory context |
| Framework direction, current architecture, package migration, code ownership and intended domain relationships | [structure_spec.md](structure_spec.md#framework-direction-and-extension-principles) | Section 0 distinguishes implemented owners, extension principles and staged target; remaining sections: domain design |
| Current baseline, priority, backlog, blockers | [PROGRESS.md](PROGRESS.md) | Single status surface; pending items live here |
| Development commands and troubleshooting | [DEVELOPMENT.md](DEVELOPMENT.md) | Setup, checks and artifact handling |
| Delegation, lifecycle, acceptance ownership and handoff | [WORKFLOW.md](WORKFLOW.md) | Engineering closure versus concrete user observations; reusable working procedures |
| Model selection and cost-aware routing | [MODEL_ROUTING.md](MODEL_ROUTING.md) | Reusable risk model for main-thread and worker selection |
| Review evidence and rejected approaches | [REVIEW.md](REVIEW.md) | Dated findings, acceptance evidence and reopening conditions; not a second backlog |
| Historical rest/V branch review programme | [BRANCH_REVIEW_PLAN.md](BRANCH_REVIEW_PLAN.md) | Closed session scopes and routes to dated REVIEW evidence; current procedures stay in WORKFLOW |
| Historical local MCP delivery outline | [MCP_PLAN.md](MCP_PLAN.md) | Closed increments and routes to current contract, commands and acceptance evidence |
| Local MCP protocol, state, tools and compatibility contract | [MCP_CONTRACT.md](MCP_CONTRACT.md) and [tool schemas](../cambam_builder/mcp_adapter/contract_v1.schema.json) | Document state, modern/legacy protocols, native authoring and transport boundaries |
| Reusable consuming-project agent policy | [consumer_AGENTS.template.md](../cambam_builder/mcp_adapter/consumer_AGENTS.template.md) | Copy-and-customize bootstrap plus daily natural-language CamBam collaboration workflow; not repository development policy |
| Historical Region and all-shape Z-coordinate parity plan | [SHAPE_PARITY_PLAN.md](SHAPE_PARITY_PLAN.md) | Closed scope and routes to implemented shape contracts and native acceptance evidence |
| Rest machining, V-cutter and shared CAM execution design | [REST_MACHINING_PLAN.md](REST_MACHINING_PLAN.md) | Product intent, proposed extensions, research and acceptance criteria; closed packets link to current contracts/evidence and priority stays in PROGRESS |
| Package metadata, dependencies and source-archive contents | `pyproject.toml`, `MANIFEST.in` | `pyproject.toml` owns published metadata/direct dependencies; `MANIFEST.in` includes the existing regression fixture data in the sdist |
| Executable API behavior | `cambam_builder/` and `tests/` | Actual implementation and regressions; document divergences from target explicitly |
| License | [LICENSE](../LICENSE) | MIT terms; does not authorize external processing of user data |

## Code and artifact boundaries

Runtime module ownership and data flow live in
[the implemented architecture](structure_spec.md#0-implemented-architecture-and-change-ownership).

- `legacy_cambam_builder/`: separately packaged legacy code; changes require an
  explicit legacy scope. `inactive/`: historical implementation, not runtime owner.
- `demos/`: authored examples, not a regression suite. `output/`, `__pycache__/`
  and `*.egg-info` are generated/disposable; do not treat them as source or erase
  existing contents without authorization. In particular, `output/` must never own
  durable project documentation, history, audit results, contracts or backlog state;
  record those in the tracked topic owner under `docs/` (or reusable tests/source)
  even when ignored artifacts provide supporting evidence. Session `.cb`/`.nc`
  inputs and posts stay under ignored `output/`, including exact validation copies;
  tracked tests generate synthetic inputs at run time. User input files are private
  data.

The repository uses `tests/` for regressions and REVIEW for dated acceptance evidence.
If an external issue tracker is designated, move task ownership there and retain
only baseline/active links in
[PROGRESS.md](PROGRESS.md). Do not mirror issue descriptions.

Completed plans must move lasting contracts to the specification or runbook and
retain only useful evidence in the review record. Search direct repository content
first; indexing/RAG requires measured discovery failures and a maintenance owner.

## Maintenance rules

This section owns documentation maintenance policy. Give each fact or rule one
authoritative home from the topic map; update that owner when the fact changes.
Other documents may give a short orientation sentence and a direct link, but must
not mirror rules, test results, acceptance narratives or backlog descriptions.
An ordinary change can require only one document update; document count is not
evidence of completion.

- Keep `AGENTS.md` compact: mandatory boundaries and routes to relevant procedures.
  WORKFLOW owns working procedures; DEVELOPMENT owns executable commands and
  troubleshooting. A policy edit does not need completion entries in status/review.
- PROGRESS owns current task state, priority, blockers and next action. Update it
  only when those change; summarize a completed capability briefly and link to its
  evidence. REVIEW owns unique dated acceptance evidence, consequential decisions
  and useful failed investigations, not a log of every exchange or document edit.
- Keep user intent, current contracts and related constraints together in the
  domain owner. Plans own proposed scope and acceptance detail, not another priority
  order. On closure, transfer lasting contracts to their owner and replace proposal
  text with a closure link; retain unique evidence and rejected approaches in REVIEW.
- Link directly to the authoritative file/heading with a label that names the
  needed information. Co-locate details normally used together; avoid chains of
  pointer-only pages or splitting small topics. A short stable orientation summary
  is useful when it saves a read, but mutable values and detailed claims stay in
  their owner. Read that owner's relevant section, not every linked document.
- Distinguish dated history from current instructions. Consolidation must preserve
  unique user observations, parameters/hashes, numeric witnesses, acceptance scope,
  failures and reopening criteria. Check incoming links before moving sections;
  repair references or retain a useful redirect. When a duplication/conflict is
  discovered, fix its owner and the affected copies within scope; backlog wider
  cleanup rather than reviewing the whole documentation set on every change.

This uses progressive disclosure and task-specific retrieval, consistent with
[OpenAI's guidance on agent instructions](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra).

One root agent file covers this small, coupled library. Add nested instructions
only when a subtree gains distinct operational requirements that meaningfully
reduce irrelevant context; ordinary module knowledge stays in the specification.
