# Current status

Baseline reviewed 2026-09-07: package version 0.1.0. The modern package provides
entity registries, layer/group/parent and part-MOP relationships, transforms,
XML reader/writer and pickle persistence. These are implemented capabilities,
not a guarantee of complete round-trip fidelity. MOP target selections are project-owned.

## Active work and next priority

**Secure MCP server and document foundation: complete** (2026-09-10; backlog
4b). The optional Python >=3.10 adapter provides a strict stdio boundary, an
explicit volatile workspace/document model and the five document tools needed to
create, open, inspect, save and close `.cb` documents. It contains all workspace
paths, rejects links and unsafe XML, bounds input/state growth, keeps revisioned
idempotency ledgers and publishes new files atomically. Public project cloning and
strict byte import provide the staged-document prerequisites.

The server targets MCP 2026-07-28 while also negotiating the deployed 2025-06-18
and 2025-11-25 client versions. Codex CLI 0.154.0 advertised 2025-06-18 even with
its `mcp_2026_07_28` feature enabled, so backward protocol compatibility is now a
required adapter boundary rather than deferred work. A deterministic Codex
app-server session completed create/inspect/save/open/inspect/close/close against
the installed adapter and independently verified the saved file hash. Detailed
security, client and packaging evidence is in the
[review](REVIEW.md#mcp-foundation-and-client-compatibility---2026-09-10).
The final suite completes successfully: 174 tests pass and one Windows
symlink-privilege case skips; the equivalent junction/reparse-point rejection
is exercised and passes.
The next increment is **4c: one authoring/MOP/inspect/save/reload parity slice**;
its boundary remains the existing eight-tool application contract.

**MCP protocol/client/adapter contract: complete** (2026-09-10; backlog 4a).
The [contract guide](MCP_CONTRACT.md) and [eight tool schemas](mcp_contract_v1.schema.json)
settle stdio MCP 2026-07-28, optional SDK 2.2.0/Python >=3.10, volatile explicit
handles, revision/retry rules, workspace containment and new-file-only saves.
SDK stdio direct-call/discovery probes passed. OpenCode 1.18.29's installed backend
uses the 2025-11-25 handshake and failed the original modern-only probe; 4b now
supports that deployed protocol era. Desktop GUI and second-PC acceptance remain
for 4e. Scope,
schema checks, compatibility evidence and reopening criteria are persisted in the
[contract](MCP_CONTRACT.md#acceptance-and-handoff-to-implementation) and
[review](REVIEW.md#mcp-protocol-and-adapter-contract---2026-09-10).
No adapter runtime or project dependency was added in 4a; 4b subsequently added
the SDK as an optional Python-version-gated extra.

**Packaging and supported Python validation: complete** (2026-09-10; backlog
item 3). Published metadata now declares Python >=3.9 and NumPy >=1.23.5 directly
in `pyproject.toml`, the sole dependency source;
`uv sync --python 3.13` created the local `.venv`. The generated lockfile is
intentionally ignored so supported-version checks continue resolving currently
compatible NumPy releases independently. Isolated wheel and sdist builds
contain both package roots and all active modern modules, with version 0.1.0,
`Requires-Python: >=3.9`, and the bounded NumPy dependency. Clean wheel installs from
outside the repository pass imports, metadata/source-path checks, representative
construction/XML behavior and all 142 tests on Python 3.9, 3.10, 3.11, 3.12 and
3.13. The sdist independently installs and passes all 142 tests on the 3.9
minimum. Exact interpreter/NumPy versions and reproduction guidance live in the
[development runbook](DEVELOPMENT.md#packaging-and-supported-python-validation),
with [review evidence](REVIEW.md#packaging-and-supported-python-verification).
Python 3.8, CLI entry points and publishing remain unsupported/out of scope.

**Varying-Z bulged Pline and Region interchange: implemented and verified**
(2026-09-10). User-supplied CamBam-generated XML establishes that Pline and
Region contours may combine bulge with unequal endpoint Z and that Region
topology is consumed as an XY projection. The framework now retains and emits
those vertex records; Region no longer imposes contour coplanarity, while its
existing XY simplicity, containment, disjointness and curve-transform rules
remain intact. Geometry queries still report stored endpoints, and bounds remain
XY projections; no intermediate spatial-curve or MOP toolpath calculation was
added. The native-derived Region regression covers an outer curve and hole with
mixed Z/bulge through two XML cycles at `1e-9` tolerance. Four focused suites pass
49 tests and the full suite passes all 142 tests. Owners: entities, Region,
focused elevation/vertex/Region tests, specification and
[review evidence](REVIEW.md#varying-z-bulged-pline-and-region-interchange).

**Canonical Pline/Points vertex records: implemented; automated verification
complete** (2026-09-09). `Vertex(x, y, z=0, *, bulge=0)` now keeps coordinates,
elevation and segment bulge together. Pline, Points and Region-owned contours use
only `vertices`; old `vertex_z`, `relative_points` and `(x, y, bulge)` inputs are
removed. XY/XYZ tuple shorthand is unambiguous, Points rejects nonzero bulge, and
existing geometry-query outputs remain unchanged. Repository callers and tests
were migrated explicitly. Current-version pickle and repeated XML interchange
retain the new representation; old pickle migration is unsupported as authorized.
Owners: entities, project adders, reader, Region, focused tests, specification and
[review evidence](REVIEW.md#canonical-vertex-record-refactor).
Six focused vertex tests and all 141 suite tests pass. Two new framework round
trips compare Pline, Points and Region XML numerically with the accepted
pre-refactor B fixture at `1e-8`; maximum observed difference is zero, and hashes
confirm the accepted A/B/C files were unchanged. Output semantics did not change,
so the existing CamBam acceptance remains applicable and was not repeated.

**Region and all-shape Z parity: implemented; automated verification, review
and CamBam display/property acceptance complete** (2026-09-09; backlog item 2).
Pline/Points vertex elevations, Circle/Arc/Rect/Text elevation fields, independent
parent/local Z offsets and one owned-contour Region primitive now extend the
existing project/XML/persistence/transfer model. Full/global subtree bakes stage
geometry changes before publication. Unsupported spatial matrices and invalid
Region topology fail explicitly. User confirmed the unreleased API may change.

Owners: entities, Region module, project transform/API orchestration, matrix
helpers, reader, transfer helper and focused regressions.
[Implemented contract](structure_spec.md#shape-elevation-and-region-contract) and
[implementation/review evidence](REVIEW.md#region-and-shape-elevation-implementation).
All 135 tests pass, along with compile/import and independent XML inspection
checks. The final topology review found and verified repairs for arc endpoint
containment, topology-changing XML rounding and small uniform scales.
The prepared A/B generator verifies eight primitives covering seven types,
XYZ/identity preservation, two Region holes and one MOP source through repeated
round trips. The user accepted the A/B display and property checks in CamBam Plus
1.0. Native C and fresh/moved Text evidence establish that serialized Text `p2`
is an optional, currently unused interchange field and omitted `p1` means the
default origin; a C-to-D framework round trip
retains it and all other declared data. See the recorded
[CamBam validation](DEVELOPMENT.md#manual-shape-parity-acceptance). The private native
Region sample contains independently confirmed self-intersections and is rejected
by the declared strict topology contract; it remains unchanged.

**Copy/transfer utilities implemented; automated checks complete** (2026-09-08; backlog item 1b).
Contract recorded in
[copy and transfer contract](structure_spec.md#copy-and-transfer-contract).
Provides complete subtree copy/move with preserved world pose, explicit identity
remapping, collision rejection, closed MOP targets and staged registry updates.
Owners: project API/transfer helper, focused copy-transfer tests and specification.
Acceptance: atomic failure and relationship/collision regressions plus two XML
round trips checking identity, references, geometry and parameters. Sixteen focused
tests and the full 88-test suite pass; compile/import checks also pass. No blocker
is recorded and no manual CamBam validation is required for this registry/XML scope.

**Curved-geometry bounds: implemented; automated checks complete** (2026-09-08).
Arc and bulged-Pline bounds now use directed analytic sweep extrema under the
complete finite affine transform, including reflection, nonuniform scale, shear
and singular projection. Seven focused regressions cover wrap-around, full and
negative sweeps, open/closed and zero-bulge segments, tolerances, malformed
inputs and overflow-resistant large finite cases. All 72 tests pass. Bounds are
not serialized, so no manual CamBam validation adds evidence for this numerical
slice. See the [contract](structure_spec.md#curved-geometry-bounds-contract) and
[verification](REVIEW.md#curved-geometry-bounds-verification).

**MOP core-model ownership and CamBam interchange: implemented; automated checks
complete; CamBam 1.0 display/property interchange accepted** (2026-09-08).
Owners: project target registry/API, entity MOP parameter encoding, reader/writer,
focused MOP tests and updated callers. Breaking API/storage changes were authorized;
no legacy adapters or old-pickle migration were added.

Acceptance: one project-owned target selection per MOP; explicit targets and live
named groups with validated atomic mutation and deletion cleanup; native XML target
edits/order remain authoritative; supported parameter and Default/Value reconstruction;
synthetic round trips plus prepared CamBam-edited input acceptance.
Implementation and automated verification are complete. The user accepted the
prepared A/B and native-edited C/framework-roundtripped D display/property cases
on CamBam Plus 1.0. The C-to-D automated comparison preserved operation order,
targets, exact parameter/state sets and geometry; C and D loaded without error
and all stated visual/property checks matched. This establishes the supported
interchange case, not production toolpaths or complete `.cb` coverage.
[Target contract](structure_spec.md#mop-target-ownership-contract).

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
and identity transforms. Later clarification establishes CamBam Plus 1.0 as the
validation environment. See
[acceptance evidence](REVIEW.md#rect-baking-display-acceptance).
The following MOP group-source compatibility increment is recorded above;
core ownership/interchange is active above. General component ordering,
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
local placement. Later clarification establishes CamBam Plus 1.0 as the validation
environment. See
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
move (+5,-3) from Before while root stays fixed. Later clarification establishes
CamBam Plus 1.0 as the validation environment.
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

1a. **Completed 2026-09-08.** Exact Arc sweep and bulged-Pline extrema, the
    finite-affine transform/tolerance policy and focused regressions are recorded
    in the implemented specification and review.
1b. **Completed 2026-09-08:** copy/transfer utilities; scope and
    acceptance are recorded above.
2. **Completed 2026-09-09:** Core CamBam shape parity: Region shapes and Z coordinates across
   Pline, Circle, Rect, Arc, Points, Text and Region. Extend existing entity/API/XML
   patterns with backward-compatible coordinate semantics, identity preservation
   and synthetic round-trip/display checks. This is independent upstream feature
   support, after core design/correctness and before downstream integrations.
   [Scope, evidence, delivery and acceptance plan](SHAPE_PARITY_PLAN.md).
   The completed canonical vertex-record refactor subsequently replaced the
   temporary parallel Pline/Points coordinate/elevation storage without changing
   the accepted XML geometry or query contracts.
3. **Completed 2026-09-10:** packaging and supported Python 3.9-3.13 validation.
   Add CLI/publishing work only for an actual need.
4. Build and maintain a local stateless MCP adapter for AI-assisted CamBam generation
   and load/modify/save workflows on another PC. User-requested future work, split
   into fresh-session increments with independent stopping conditions:
   - **4a — completed 2026-09-10:** [protocol/client and adapter contract](MCP_CONTRACT.md),
     eight tool schemas and compatibility probes. OpenCode 1.18.29 is incompatible
     with a modern-only server; backward negotiation is implemented in 4b and
     desktop acceptance remains pending in 4e.
   - **4b — completed 2026-09-10:** secure server/document foundation, required
     public clone/strict-import contracts and backward protocol negotiation.
   - **4c — next:** prove one authoring/MOP/inspect/save/reload parity slice.
     Sol/xhigh.
   - **4d:** expand the explicitly supported API and resilience coverage.
     Sol/high, with Terra/xhigh suitable for settled coverage batches.
   - **4e:** validate installation, named desktop-client interoperability and
     second-PC/user acceptance. Terra/high; escalate interoperability defects to
     Sol/high.
   Each increment must persist its decisions, evidence and remaining limits before
   handoff; later breadth must not leak into earlier scopes. Full acceptance and
   maintenance details: [MCP delivery plan](MCP_PLAN.md#delivery-increments-and-session-boundaries).
5. **Planned future feature:** rest-area calculation and rest
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

No blocker for the completed MOP ownership/interchange scope. Production
toolpaths and complete `.cb` format coverage remain outside that acceptance.
The old framework API and pickle format did not constrain the redesign.
MCP 4e still requires named desktop-client and second-PC acceptance. The adapter
retains 2026-07-28 as its target and also supports 2025-06-18 and 2025-11-25;
reprobe the clients during 4e and retire an older version only after deployed
clients no longer need it. SDK-backed 4c implementation can proceed.

## Completion and verification

- MOP group-source compatibility: historically characterized with six tests;
  the redesign supersedes its old runtime ownership contract.
- Working agreement: implemented; documentation checks recorded in the review.
- Parent XML slice: implemented and automated checks complete; nine regression tests.
- Parent-cycle rejection: implemented and automated checks complete.
- Broader product baseline: limited local checks only.
- User/CamBam acceptance: parent, full-bake, global-transform and Rect-bake
  synthetic display cases were accepted. MOP A/B and native C/D display/property
  interchange were accepted in CamBam Plus 1.0. Production toolpaths and broader
  `.cb` coverage were not validated.

The completed MOP implementation and automated verification include a final
suite reports 65 passed tests in
`output/mop-core-checks-ncsbkpgz/suite.log`; compileall for
`cambam_builder`, `legacy_cambam_builder`, `tests` and `demos` exits 0. The A/B
MOP generator exits 0 and verifies counts, operation targets, depths and feeds.
Lead XML inspection also confirmed rectangle coordinates, triangle vertices
and drill points. The
context regression also reproduces and guards the former lost global
`Style`/`StyleLibrary` state. See
[MOP review evidence](REVIEW.md#mop-core-ownership-and-interchange-redesign).

[Verification commands](DEVELOPMENT.md) and [work lifecycle](WORKFLOW.md)
are authoritative. MOP ownership, supported interchange, curved geometry bounds
(1a), copy/transfer (1b) and shape parity (2) are recorded above with their
verification and acceptance state.

This is a good fresh-session breakpoint: 4b's server, document lifecycle,
backward negotiation and client evidence are persisted, while 4c has a distinct
authoring scope. No pending decision depends on chat; desktop and second-PC
interoperability remain the explicit 4e gate.
Suggested commit: `feat: add secure MCP document foundation`.
