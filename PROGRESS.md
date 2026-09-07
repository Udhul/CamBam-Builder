# Current status

Baseline reviewed 2026-09-07: package version 0.1.0. The modern package provides
entity registries, layer/group/parent and part-MOP relationships, transforms,
XML reader/writer and pickle persistence. These are implemented capabilities,
not a guarantee of complete round-trip fidelity. MOP sources remain on instances.

## Active work and next priority

Rect rotation/shear baking repair: **implemented; automated checks complete**
(2026-09-08). Owners:
`cambam_entities.py`, the explicit bake helper in `cambam_project.py`,
`tests/test_rect_bake_defect.py`, and the implemented specification.
Acceptance: preserve exact closed outlines, identities and relationships through
full/explicit/global baking and two XML round trips; retain axis-aligned Rects
and preserve descendant world geometry. Non-axis-aligned results become Plines
in place so existing Python references remain valid. No implementation blocker.
All 51 suite tests pass, including nine Rect regressions. Synthetic A/B files
are generated and XML-inspected; **user display acceptance is pending**. Follow
[the prepared criteria](docs/DEVELOPMENT.md#manual-rect-bake-acceptance) and record
the result in [repair evidence](docs/REVIEW.md#rect-baking-repair-verification). General component ordering,
curved geometry and alignment remain outside this increment.
Phases 1 and 2 (project discovery and working
agreement) and [phase 3 (initial engineering review)](docs/REVIEW.md#phase-3-completion-audit)
are complete; [workflow completion evidence](docs/REVIEW.md#phase-1-and-2-completion-audit)
records the foundations. Product acceptance remains separate.

**Parent round-trip fidelity: implemented; automated checks complete.** Preserve parent
identity and world geometry through XML export/import. Owners: `cambam_writer.py`,
`cambam_reader.py`, new focused `tests/`, and the specification for any clarified
local/world transform contract. [Defect evidence](docs/REVIEW.md#parent-identity-and-transform-reconstruction).

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
with syntax/import checks complete. See [repair evidence](docs/REVIEW.md#parent-round-trip-repair-verification).
Synthetic parent A/B geometry/placement **accepted by the user on 2026-09-07**:
both files match the expected world coordinates; resetting B's matrices restores
local placement. CamBam version was not supplied. See
[acceptance evidence](docs/REVIEW.md#parent-ab-user-acceptance).
Packaging and broader CamBam compatibility remain unverified.

**Parent-cycle rejection: implemented; automated checks complete.**
Owner: `cambam_project.py`; focused API and synthetic XML regression tests.
Acceptance: reject self/descendant cycles before mutation; preserve both parent
indexes, memberships and transforms; valid reparent/detach remain usable; cyclic
XML produces an acyclic hierarchy with preserved world geometry.
[Verification evidence](docs/REVIEW.md#parent-cycle-rejection-verification) records
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
[Before/reference/result files and criteria](docs/DEVELOPMENT.md#manual-global-transform-acceptance)
were accepted by the user: A/B coordinates match, B retains matrices, and child/leaf
move (+5,-3) from Before while root stays fixed. CamBam version was not supplied.
[Verification evidence](docs/REVIEW.md#global-transform-ordering-verification)
records the failure and coordinate-frame policy.

Full-bake hierarchy fidelity: **implemented; automated checks complete; synthetic
A/B user acceptance complete**. The user confirmed matching final coordinates,
identity matrices in both files and the intended layer difference. See
[repair and acceptance evidence](docs/REVIEW.md#full-bake-hierarchy-verification).

## Completed implementation: MOP round-trip identity

**Implemented; automated checks complete; synthetic CamBam acceptance complete.** Entity
encoders store UUID/identifier metadata separately from display names. The reader
restores identity before registration, retains XML operation order and targets,
allocates fresh identities for legacy/malformed Tags, and fails import on explicit
identity collisions. No MOP registry migration or live group semantic change.
Five new regressions and all 31 suite tests pass. The original reader fails the
new suite. See [evidence](docs/REVIEW.md#mop-identity-round-trip-verification) and
[prepared A/B acceptance](docs/DEVELOPMENT.md#manual-mop-identity-acceptance).
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
[verification](docs/REVIEW.md#export-failure-and-state-path-verification).

## Remaining backlog, in order

1. Align MOP registry ownership with the specification after defining group-source
   compatibility; test defaults and malformed XML metadata reconstruction.
2. Improve curved-geometry bounds; implement copy/transfer utilities against the
   specification, with collision and relationship tests.
3. Validate packaging and supported Python versions; expand examples and tests as
   each capability is verified. Add CLI/distribution work only for an actual need.
4. Build and maintain a local stateless MCP adapter for AI-assisted CamBam generation
   and load/modify/save workflows on another PC. User-requested future work;
   not active and does not displace correctness fixes.
   [Requirements, acceptance and maintenance plan](docs/MCP_PLAN.md).

This supersedes the former five broad increments; their pending scope is retained
above. Detailed contracts remain in `structure_spec.md`, and review evidence in
`docs/REVIEW.md`. No repository-linked issue tracker was found.

## Blockers and decisions

No implementation blocker or pending validation for the completed MOP synthetic case. Full source-system
compatibility requires CamBam validation and representative authorized fixtures.
MOP live group versus snapshot semantics need resolution before registry migration.
The architecture specification is a target; its PID-source wording and centralized
MOP ownership must be reconciled before implementing that migration.

## Completion and verification

- Working agreement: implemented; documentation checks recorded in the review.
- Parent XML slice: implemented and automated checks complete; nine regression tests.
- Parent-cycle rejection: implemented and automated checks complete.
- Broader product baseline: limited local checks only.
- User/CamBam acceptance: parent, full-bake and global-transform synthetic display and MOP load/properties accepted; broader production acceptance not performed.

[Verification commands](docs/DEVELOPMENT.md) and [work lifecycle](docs/WORKFLOW.md)
are authoritative. Promote one bounded item with acceptance criteria, implement
and verify it, then separately record user acceptance before claiming that level
of completion. The active Rect repair matters because successful bake/export
calls previously persisted enlarged outlines. Once its display acceptance is
recorded, define MOP live-group versus snapshot compatibility before migrating
ownership (backlog item 1). That resolves the next architectural dependency;
general component ordering, curved baking and alignment remain deferred until a
failing fixture or workflow dependency justifies them.

## Session breakpoint

The Rect implementation, automated evidence and acceptance preparation are
complete. This is a good fresh-session breakpoint because the contract, files
and remaining acceptance are persisted; no worker results or decisions remain
unsaved. Continue this session for the short display acceptance result, then
start a fresh session for the distinct MOP compatibility scope. CamBam display acceptance is pending;
retain its generated artifacts and record the user's result before closing that
acceptance. MOP source compatibility is a distinct next scope with its priority
and unresolved contract captured above. No staging or commit was performed.
