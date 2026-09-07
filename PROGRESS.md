# Current status

Baseline reviewed 2026-09-07: package version 0.1.0. The modern package provides
entity registries, layer/group/parent and part-MOP relationships, transforms,
XML reader/writer and pickle persistence. These are implemented capabilities,
not a guarantee of complete round-trip fidelity. MOP sources remain on instances.

## Active work and next priority

No implementation slice is active. Phases 1 and 2 (project discovery and working
agreement) are complete; [completion evidence](docs/REVIEW.md#phase-1-and-2-completion-audit)
records their scope and verification. Product acceptance remains separate.

**Next: parent round-trip fidelity — backlog, not started.** Preserve parent
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

Next action: promote this slice to active and write reproducing tests. Exclude
general transform refactoring and MOP migration. Rollback: revert only the slice's
patch; do not rewrite existing CAD files or saved pickle state.

## Remaining backlog, in order

1. Reject parent cycles atomically; test traversal and registry integrity.
2. Establish transform matrix and baking fidelity with synthetic end-to-end tests.
3. Define export failure behavior and prevent silent incomplete output.
4. Align MOP registry ownership with the specification after defining group-source
   compatibility; test defaults and malformed XML metadata reconstruction.
5. Improve curved-geometry bounds; implement copy/transfer utilities against the
   specification, with collision and relationship tests.
6. Validate packaging and supported Python versions; expand examples and tests as
   each capability is verified. Add CLI/distribution work only for an actual need.

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
- Product baseline: limited local checks only; no dedicated automated test suite.
- User/CamBam/production acceptance: not performed.

[Verification commands](docs/DEVELOPMENT.md) and [work lifecycle](docs/WORKFLOW.md)
are authoritative. Promote one bounded item with acceptance criteria, implement
and verify it, then separately record user acceptance before claiming that level
of completion. Recommended next step: execute the parent metadata slice.
