# CamBam project agent instructions

Use the configured `cambam_*` MCP tools for CamBam document, geometry and machining
state. Use ordinary local filesystem tools only for exact file staging, delivery and
hash verification.

## First-run setup gate

`FIRST_RUN_SETUP: PENDING`

While that exact marker remains, project onboarding takes priority over executing the
first CamBam request. Preserve that request so it can resume after setup. In the first
response:

1. Say briefly that the project needs one-time setup before document work.
2. Inspect the project folder read-only and use evident facts instead of asking for
   them again. Keep discovery shallow and relevant: check the project instructions,
   top-level context/configuration and existing `.cb` files. Do not recursively scan
   nested repositories or treat their source tests as part of CamBam artifact
   validation unless the user explicitly makes software development the task.
3. Use the client's interactive question facility when available. Each unresolved
   item below must be its own clearly labeled prompt with its own answer field:
   - default units;
   - default output directory and naming convention for new files;
   - one file-history policy covering edits: update in place and any backup rule, or
     always Save As; for Save As, also establish the versioned filename pattern.
     Recommend versioned Save As unless evident project conventions favor another
     choice;
   - source for job-specific material, stock, tool and machining inputs;
   - preferred post-save review handoff. Phrase this as: "After I save and self-check
     each CAM revision, how should I hand it off for review?" Recommend saving and
     reporting the file ready for optional human CamBam/toolpath review, followed by
     iterative corrections when requested. Alternative policies may require explicit
     human confirmation before labeling a revision production-ready, or omit a review
     prompt. Human review must never be a prerequisite for saving the reviewable file.
   Offer a small set of context-based choices and a recommended option where useful.
   Never collapse several items into one free-text question. Present as many separate
   questions in one dialog as the client supports; if some remain, continue with
   another setup dialog before beginning document work.
4. Do not ask for the current document's filename, stock dimensions, material, tools,
   feeds, speeds, depths, geometry or nesting choices during onboarding. Those are
   task inputs, not project setup, unless the user explicitly says a value is a durable
   project default.
5. Do not create, import, edit or save a CamBam document until the setup answers are
   received.

After the user answers, edit this copied `AGENTS.md` before resuming document work:

- replace this entire section with `## Project defaults` and concise, concrete rules;
- replace the pending marker with `FIRST_RUN_SETUP: COMPLETE`;
- omit unanswered optional preferences instead of retaining setup prompts; and
- acknowledge the saved defaults, then resume the preserved original request from its
  beginning. Ask any necessary task-specific questions separately and only when the
  request or safe machining requires them.

Do not hard-code an MCP workspace ID in project defaults. Discover the active value
and workspace path from the connected server each run.

## Daily workflow

- Keep a short task checklist for multi-step work.
- Treat the user's current project file as the durable source of truth and an MCP
  handle as a volatile working snapshot.
- At the start of file work, call `document_list`. Copy workspace IDs, document
  handles, revisions and entity IDs exactly from tool results; never reconstruct,
  shorten or retype opaque IDs.
- On a shared filesystem, load an existing or manually edited client file by binary-
  copying it under a fresh `.cb` name inside `document_list.workspace_path`, verifying
  source and staged SHA-256 equality, and calling `document_open` with
  `expected_sha256`. Use `document_import` only when client and server filesystems are
  separate.
- Issue at most one mutation at a time for a document. Wait for success, then use its
  returned revision as the next `expected_revision` and a fresh UUID as `request_id`.
  A stale failure is a rejected edit, not partial success: inspect current state and
  retry only if the intent still applies.
- Use IDs from the latest successful snapshot. Inspect at meaningful checkpoints and
  before dependent machining or final delivery; also verify geometry returned directly
  by creation tools. Do not add repetitive inspections that provide no new evidence.
- Use deterministic calculations and advisory tools for derived values. Never invent
  stock, material, tool, depth, feed, speed, clearance, cut-through or holding choices.
  Propose context-based options and ask when a missing choice affects geometry,
  machining or safety.
- Calling a document a test, demo or example does not by itself authorize invented
  machining values. For a clearly non-production exercise, propose a coherent set of
  visibly labeled fixture values and obtain task-level confirmation before creating
  enabled MOPs. Otherwise use established project/job inputs or ask for them. Do not
  add a fixture-value exception to project defaults unless the user explicitly chooses
  it as durable policy.
- For advanced geometry, layout, nesting, repeated features or constraint math, create
  and run a small deterministic local helper script when that improves reliability.
  Make units and assumptions explicit, inspect its calculated points/transforms and
  invariants, then pass the derived values through the MCP authoring tools. Do not use
  a helper script to generate or patch `.cb` XML or bypass MCP document state.
- Before delivery, self-check a produced `.cb` using MCP mutation results and
  `document_inspect` geometry/relationships, then verify the delivered file hash. The
  MCP exposes these inspection and calculation facilities, not this repository's
  development test suite or a CamBam toolpath engine. Do not invent Python test scripts
  merely to validate a produced file; computational helper scripts are for solving the
  user's geometry or planning problem.
- Agent completion, artifact delivery and human acceptance are separate. Normally save
  and self-check each requested revision, report it ready for review, and continue from
  any user corrections. If project policy requires human CamBam/toolpath confirmation,
  that confirmation gates only claims such as "accepted" or "production-ready"; it
  never gates saving or handing off the reviewable file. Do not claim human acceptance
  merely because the agent completed its requested edit.

## Understanding and repeating manual edits

- Compare a trustworthy baseline snapshot with the current file using stable entity
  IDs and typed geometry. A current snapshot alone cannot establish what changed.
- Separate user edits from CamBam serialization changes such as formatting, element
  order, generated defaults and native state fields.
- Derive the smallest delta demonstrated by all selected examples (for example, one
  translation vector, one rotation about a known center, or one parameter change).
  Apply only that delta to the requested counterparts. Do not add alignment,
  normalization, rotation, spacing or layout changes merely because they appear
  aesthetically plausible.
- If no trustworthy baseline exists, entity correspondence is uncertain, or several
  different edits explain the observations, explain the ambiguity and ask before
  mutating.

## Saving and completion

- A task that creates or changes a CamBam document includes durable project-local
  delivery unless the user explicitly requests a preview or analysis only.
- For a new document, save to the configured project destination, normally the current
  project directory plus the document name. For an edit, write back to the source path
  unless the user or project defaults request Save As.
- On a shared filesystem, call `document_save` with a fresh workspace-relative name.
  Its result is only a server-workspace handoff (`client_file_created=false`), not task
  completion. Binary-copy the returned `absolute_path` to the intended project path
  and verify the destination SHA-256.
- Before replacing an existing client file, verify that its current hash still equals
  the hash opened for this edit. If it changed again, do not overwrite it; preserve the
  generated candidate and ask how to reconcile the versions.
- Deliver the saved, self-checked revision before requesting any configured human
  CamBam review. Describe it as ready for review rather than production-ready until the
  user confirms acceptance; subsequent feedback starts another edit/save/review cycle.
- Report the final project-local path, hash, document revision and concise verification
  results. Never present an MCP workspace artifact or an inline export suggestion as
  the user's saved project file.
