# Current status

Baseline reviewed 2026-09-07: package version 0.1.0. The modern package provides
entity registries, layer/group/parent and part-MOP relationships, transforms,
XML reader/writer and pickle persistence. These are implemented capabilities,
not a guarantee of complete round-trip fidelity. MOP sources remain on instances.

## Active work and next priority

**Next priority: correct MOP core-model ownership and CamBam interchange**
(backlog item 1). User clarification, 2026-09-08: breaking framework API/storage
changes are allowed; no legacy adapters or old-pickle migration are required.
[Development compatibility policy](structure_spec.md#development-compatibility-policy).

MOP group-source characterization is complete: six tests, all 57 suite tests
passed in that increment. It records existing behavior, not a requirement to
preserve the old `pid_source` API or live-group implementation. Update those tests
with the chosen model. Runtime is unchanged by this clarification. See
[decision update](REVIEW.md#development-compatibility-priority).

Rect rotation/shear baking repair: **implemented; automated checks complete; synthetic display accepted**
(2026-09-08). Owners:
`cambam_entities.py`, the explicit bake helper in `cambam_project.py`,
`tests/test_rect_bake_defect.py`, and the implemented specification.
Acceptance: preserve exact closed outlines, identities and relationships through
full/explicit/global baking and two XML round trips; retain axis-aligned Rects
and preserve descendant world geometry. Non-axis-aligned results become Plines
in place so existing Python references remain valid. No implementation blocker.
All 51 suite tests pass, including nine Rect regressions. Synthetic A/B files
were accepted by the user on 2026-09-08: all conditions met, full outline match
and identity transforms. CamBam version was not supplied. See
[acceptance evidence](REVIEW.md#rect-baking-display-acceptance).
The following MOP group-source compatibility increment is recorded above;
registry migration remains backlog item 1. General component ordering,
curved geometry and alignment remain outside this increment.
Phases 1 and 2 (project discovery and working
agreement) and [phase 3 (initial engineering review)](REVIEW.md#phase-3-completion-audit)
are complete; [workflow completion evidence](REVIEW.md#phase-1-and-2-completion-audit)
records the foundations. Product acceptance remains separate.

**Parent round-trip fidelity: implemented; automated checks complete.** Preserve parent
identity and world geometry through XML export/import. Owners: `cambam_writer.py`,
`cambam_reader.py`, new focused `tests/`, and the specification for any clarified
local/world transform contract. [Defect evidence](REVIEW.md#parent-identity-and-transform-reconstruction).

Acceptance:

- Preserve UUIDs, identifiers, layers and parent edges regardless of XML order.
- Preserve root/child/grandchild world matrices and geometry through two round
  trips, including non-identity parent transforms and child offsets.
- Document and test world-pose behavior for parentless/missing/invalid parents
  and explicit failure/fallback behavior for singular parent transforms.
- Add regression tests under `tests/` that fail before the fix and pass afterward; assert
  counts/relationships and numeric tolerances. Record the test command in the
  runbook and pass relevant syntax/import checks; inspect synthetic XML.
- Record CamBam geometry validation separately; do not claim it from local tests.

Verification: nine authored unittest tests pass on Python 3.10.9 / NumPy 1.23.5,
with syntax/import checks complete. See [repair evidence](REVIEW.md#parent-round-trip-repair-verification).
Synthetic parent A/B geometry/placement **accepted by the user on 2026-09-07**:
both files match the expected world coordinates; resetting B's matrices restores
local placement. CamBam version was not supplied. See
[acceptance evidence](REVIEW.md#parent-ab-user-acceptance).
Packaging and broader CamBam compatibility remain unverified.

**Parent-cycle rejection: implemented; automated checks complete.**
Owner: `cambam_project.py`; focused API and synthetic XML regression tests.
Acceptance: reject self/descendant cycles before mutation; preserve both parent
indexes, memberships and transforms; valid reparent/detach remain usable; cyclic
XML produces an acyclic hierarchy with preserved world geometry.
[Verification evidence](REVIEW.md#parent-cycle-rejection-verification) records
the regression and checks. Related parent A/B display validation is accepted.
General transform refactoring and MOP migration remain outside this slice. Rollback: revert only the slice's
patch; do not rewrite existing CAD files or saved pickle state.

## Completed line: hierarchy and global transform fidelity

State: **implemented; automated checks complete; synthetic display accepted**. Owner: `transform_primitive` in `cambam_project.py`,
`tests/test_global_transforms.py`, and the implemented specification. Acceptance:
world points in the selected subtree receive `M @ world_point` exactly once in
both matrix and baked modes; ancestors/siblings/registries remain unchanged.
Matrix mode retains descendant local matrices; baked mode retains every effective
matrix. Reject invalid affine inputs and unsolvable coordinate frames before
mutation. Verify wrappers and repeated XML round trips on straight polylines.
Curved/component baking and alignment's separate implementation remain outside
this slice. Eight new regressions and all 26 suite tests pass.
[Before/reference/result files and criteria](DEVELOPMENT.md#manual-global-transform-acceptance)
were accepted by the user: A/B coordinates match, B retains matrices, and child/leaf
move (+5,-3) from Before while root stays fixed. CamBam version was not supplied.
[Verification evidence](REVIEW.md#global-transform-ordering-verification)
records the failure and coordinate-frame policy.

Full-bake hierarchy fidelity: **implemented; automated checks complete; synthetic
A/B user acceptance complete**. The user confirmed matching final coordinates,
identity matrices in both files and the intended layer difference. See
[repair and acceptance evidence](REVIEW.md#full-bake-hierarchy-verification).

## Completed implementation: MOP round-trip identity

**Implemented; automated checks complete; synthetic CamBam acceptance complete.** Entity
encoders store UUID/identifier metadata separately from display names. The reader
restores identity before registration, retains XML operation order and targets,
allocates fresh identities for legacy/malformed Tags, and fails import on explicit
identity collisions. No MOP registry migration or live group semantic change.
Five new regressions and all 31 suite tests pass. The original reader fails the
new suite. See [evidence](REVIEW.md#mop-identity-round-trip-verification) and
[prepared A/B acceptance](DEVELOPMENT.md#manual-mop-identity-acceptance).
The stopping condition for implementation is met; complete parameter coverage,
Default/Value fidelity and production toolpaths remain unverified.

## Completed implementation: export failures and state paths

**Implemented; automated checks complete.** Owners: `cambam_writer.py`,
`CamBamProject.save_state`, `tests/test_export_failures.py`, and
`tests/test_state_persistence.py`. Encoder failures propagate; export replaces
the destination only after complete serialization and close. Bare state filenames
work, and directory errors propagate. Eight export and three persistence tests
plus all prior regressions pass (42 total). Acceptance covers existing/new
destinations, failure propagation/cleanup, XML round trips and restored pickle
project links. No blocker or manual validation remains for this scope.
See [contract](structure_spec.md#export-failure-and-state-saving-contract) and
[verification](REVIEW.md#export-failure-and-state-path-verification).

## Remaining backlog, in order

1. Redesign MOP target ownership around a coherent project-owned core model.
   Choose explicit-target and group-selection semantics for domain needs, with
   one authoritative relationship owner. Change the Python API and update callers
   and tests as needed; no old-version adapters or pickle migration. Validate the
   `.cb` boundary: output for CamBam and input created/edited in CamBam, including
   absent/malformed framework metadata, native target edits, operation order and
   parameter/default reconstruction. Establish focused synthetic checks and
   prepare CamBam validation where local tests cannot establish interoperability.
   [Governing policy](structure_spec.md#development-compatibility-policy).
2. Improve curved-geometry bounds; implement copy/transfer utilities against the
   specification, with collision and relationship tests.
3. Implement core CamBam shape parity: Region shapes and Z coordinates across
   Pline, Circle, Rect, Arc, Points, Text and Region. Extend existing entity/API/XML
   patterns with backward-compatible coordinate semantics, identity preservation
   and synthetic round-trip/display checks. This is independent upstream feature
   support, after core design/correctness and before downstream integrations.
   [Scope, evidence, delivery and acceptance plan](SHAPE_PARITY_PLAN.md).
4. Validate packaging and supported Python versions; expand examples and tests as
   each capability is verified. Add CLI/distribution work only for an actual need.
5. Build and maintain a local stateless MCP adapter for AI-assisted CamBam generation
   and load/modify/save workflows on another PC. User-requested future work;
   not active and does not displace correctness fixes.
   [Requirements, acceptance and maintenance plan](MCP_PLAN.md).
6. **Planned future feature:** rest-area calculation and rest
   machining helpers for pocket and inside/outside profile MOPs. Five outcomes:
   pure rest regions; safe expansion for a smaller endmill; pointed/flat-tip
   V-cutter preparation; a general bounded XYZ V-carving path calculator; and
   V-cutter edge tracing with corner cleanup. Region topology and shape Z fidelity
   are independently prioritized upstream dependencies owned by the
   [shape parity plan](SHAPE_PARITY_PLAN.md), not part of rest implementation.
   Calculations must run programmatically in this framework without CamBam;
   `.cb` files provide interchange, not access to a headless CAM engine.
   [Problem, reasoning, support gaps and acceptance plan](REST_MACHINING_PLAN.md).
   Promote for a concrete workflow after higher-priority work; no implementation
   or machining acceptance is claimed by this planning entry.

This supersedes the former five broad increments; their pending scope is retained
above. Detailed contracts remain in `docs/structure_spec.md`, and review evidence in
`docs/REVIEW.md`. No repository-linked issue tracker was found.

## Blockers and decisions

No implementation blocker or pending validation for the completed MOP synthetic case. Full source-system
compatibility requires CamBam validation and representative authorized fixtures.
The old framework API and pickle format do not constrain redesign. Group semantics
and the new API remain design work in backlog item 1; CamBam `.cb` interoperability
is the required compatibility boundary. Centralized ownership is not yet implemented.

## Completion and verification

- MOP group-source compatibility: defined and characterized; six new tests and
  all 57 suite tests pass; runtime/XML behavior unchanged.
- Working agreement: implemented; documentation checks recorded in the review.
- Parent XML slice: implemented and automated checks complete; nine regression tests.
- Parent-cycle rejection: implemented and automated checks complete.
- Broader product baseline: limited local checks only.
- User/CamBam acceptance: parent, full-bake, global-transform and Rect-bake synthetic display and MOP load/properties accepted; broader production acceptance not performed.

[Verification commands](DEVELOPMENT.md) and [work lifecycle](WORKFLOW.md)
are authoritative. Existing MOP characterization informs the redesign without
freezing historical API choices. Correct ownership and CamBam interoperability
remain the next priority; broader geometry and integration work stays ordered
in the backlog above.

## Session breakpoint

This clarification is persisted in the specification and review. A fresh session
is appropriate for the distinct core-model implementation scope. Design the MOP
relationships and public API without old-version compatibility overhead, then
validate the `.cb` boundary. No implementation or CamBam acceptance is claimed
by this documentation update. No staging or commit was performed.
