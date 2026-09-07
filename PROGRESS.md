# Current status

Baseline reviewed 2026-09-07: package version 0.1.0. The modern package provides
entity registries, layer/group/parent and part-MOP relationships, transforms,
XML reader/writer and pickle persistence. These are implemented capabilities,
not a guarantee of complete round-trip fidelity. MOP sources remain on instances.

## Active work and next priority

Full-bake hierarchy fidelity is implemented; manual acceptance is pending below. Phases 1 and 2 (project discovery and working
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

## Active increment: full-bake hierarchy fidelity

State: **implemented; automated checks complete; user acceptance pending**. Owner: `bake_primitive_transform` in `cambam_project.py`,
focused synthetic tests and the implemented specification. First prove full
local-transform baking on straight polylines: recursive and nonrecursive baking
must preserve every descendant's world geometry, preserve relationship/identity
registries, reset selected matrices, and survive XML round trips. Include
noncommuting affine transforms and singular scales. Explicit-matrix/component
baking, global transform application and shape conversion remain subsequent
bounded work. Five full-bake regressions and all 18 suite tests pass. The two nonrecursive
cases failed before the fix. [Manual A/B files and criteria](docs/DEVELOPMENT.md#manual-full-bake-acceptance)
are prepared; [repair evidence](docs/REVIEW.md#full-bake-hierarchy-verification)
records scope and limitations.

## Remaining backlog, in order

1. Complete remaining transform fidelity after the active full-bake slice:
   global transform application, component baking, curved geometry and Rect conversion.
2. Preserve MOP identities across XML round trips with duplicate display names.
3. Define export failure behavior and prevent silent incomplete output; fix bare
   filename state saving as a separate small persistence slice.
4. Align MOP registry ownership with the specification after defining group-source
   compatibility; test defaults and malformed XML metadata reconstruction.
5. Improve curved-geometry bounds; implement copy/transfer utilities against the
   specification, with collision and relationship tests.
6. Validate packaging and supported Python versions; expand examples and tests as
   each capability is verified. Add CLI/distribution work only for an actual need.
7. Build and maintain a local stateless MCP adapter for AI-assisted CamBam generation
   and load/modify/save workflows on another PC. User-requested future work;
   not active and does not displace correctness fixes.
   [Requirements, acceptance and maintenance plan](docs/MCP_PLAN.md).

This supersedes the former five broad increments; their pending scope is retained
above. Detailed contracts remain in `structure_spec.md`, and review evidence in
`docs/REVIEW.md`. No repository-linked issue tracker was found.

## Blockers and decisions

No blocker for the recommended synthetic regression/fix. Full source-system
compatibility requires CamBam validation and representative authorized fixtures.
MOP live group versus snapshot semantics need resolution before registry migration.
The architecture specification is a target; its PID-source wording and centralized
MOP ownership must be reconciled before implementing that migration.

## Completion and verification

- Working agreement: implemented; documentation checks recorded in the review.
- Parent XML slice: implemented and automated checks complete; nine regression tests.
- Parent-cycle rejection: implemented and automated checks complete.
- Broader product baseline: limited local checks only.
- User/CamBam acceptance: parent A/B synthetic display accepted; broader production acceptance not performed.

[Verification commands](docs/DEVELOPMENT.md) and [work lifecycle](docs/WORKFLOW.md)
are authoritative. Promote one bounded item with acceptance criteria, implement
and verify it, then separately record user acceptance before claiming that level
of completion. Next action: user full-bake A/B acceptance; next implementation slice is global transform application ordering.
