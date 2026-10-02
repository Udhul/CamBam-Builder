# Region and Z-coordinate shape parity

Historical completed plan. [The implemented shape contract](structure_spec.md#shape-elevation-and-region-contract)
owns current APIs, topology and transform limits. [The review record](REVIEW.md#region-and-shape-elevation-implementation)
owns automated and user acceptance; [the runbook](DEVELOPMENT.md#manual-shape-parity-acceptance)
owns reusable checks. Later [canonical vertex records](REVIEW.md#canonical-vertex-record-refactor)
and [varying-Z interchange evidence](REVIEW.md#varying-z-bulged-pline-and-region-interchange)
supersede the original storage assumptions. Current work belongs in [PROGRESS](PROGRESS.md).

## Objective and scope

Region and elevation interchange are native CAD capabilities, independent of
the downstream rest-machining design. Use the [seven-shape scope and compatibility
boundary](structure_spec.md#shape-elevation-and-region-contract).

### Repository evidence

See [the native fixture finding](REVIEW.md#native-region-fixture-finding) and
[the later varying-Z native observations](REVIEW.md#varying-z-bulged-pline-and-region-interchange).

### Native example boundary discovered during implementation

[The native Region finding](REVIEW.md#native-region-fixture-finding) preserves
the crossing pairs, independent chord witness and reopening criterion for invalid
contours. Schema support does not imply acceptance of that self-intersecting source.

## Shape coverage and compatibility contract

Use the [implemented elevation, vertex and matrix contract](structure_spec.md#shape-elevation-and-region-contract).
The historical parallel storage and tuple proposals are superseded.

## Region ownership and XML support

Use the [registered Region and owned-contour contract](structure_spec.md#shape-elevation-and-region-contract)
for identity, MOP targeting, schema, winding, topology and queries.

## Bounded delivery sequence

The [implementation record](REVIEW.md#region-and-shape-elevation-implementation),
[vertex-record follow-up](REVIEW.md#canonical-vertex-record-refactor) and
[varying-Z follow-up](REVIEW.md#varying-z-bulged-pline-and-region-interchange)
preserve the delivered sequence and consequential compatibility changes.

## Acceptance and stopping condition

[Recorded verification and user acceptance](REVIEW.md#verification-and-acceptance-state)
own observations, tolerances, repaired defects and accepted display/interchange scope.
[The manual procedure](DEVELOPMENT.md#manual-shape-parity-acceptance)
defines reusable checks; the completed feature adds no production-toolpath claim.
