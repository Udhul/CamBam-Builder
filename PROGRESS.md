# Current status

Baseline reviewed 2026-09-07: package version 0.1.0. The modern package provides
entity registries, layer/group/parent and part-MOP relationships, transforms,
XML reader/writer and pickle persistence. These are implemented capabilities,
not a guarantee of complete round-trip fidelity. MOP sources remain on instances.

## Active priority

Recommended next slice: preserve parent identity and world geometry through XML export/import.
Scope: writer/reader parent metadata and local/world transform boundary and focused regression tests.
Acceptance and evidence: [initial review](docs/REVIEW.md). Runtime work has not
started; this review establishes the working agreement and selects the next slice.

## Ordered backlog

1. Parent metadata round-trip regression and fix (active recommendation).
2. Reject parent cycles atomically; test traversal and registry integrity.
3. Establish transform matrix and baking fidelity with synthetic end-to-end tests.
4. Define export failure behavior and prevent silent incomplete output.
5. Align MOP registry ownership with the specification after defining group-source
   compatibility; test defaults and malformed XML metadata reconstruction.
6. Improve curved-geometry bounds; implement copy/transfer utilities against the
   specification, with collision and relationship tests.
7. Validate packaging and supported Python versions; expand examples and tests as
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
