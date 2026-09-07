# Initial workflow and engineering review — 2026-09-07

## Scope and workflow assessment

The initial working tree was clean. This pass changed documentation and ignored
local-environment configuration only; no runtime implementation was changed.
The modern package is the review scope. Legacy and inactive code were preserved.

Useful existing practices: modular project/entity/XML/matrix boundaries, a central
relationship design, dependency declarations, examples and a progress tracker.
Main gaps were an almost empty README, no agent entry point or topic ownership,
overconfident round-trip status, broad overlapping increments, and no dedicated
assertion-based test suite or CI. The specification repeats transformation concepts
in sections 3.3 and 6 and summaries; those remain in one design owner rather than
being copied into new guides. Its MOP PID-source wording is ambiguous relative to
central registry ownership. The progress tracker is now the sole priority owner.

No separate decision/failure records or tracker links were found in the existing
tracked documentation. In-code TODOs and comments are evidence of unresolved work,
not proof that experiments were run. No historical logs or user CAD assets were
loaded. This record starts durable negative knowledge without inventing history.

Session capabilities: local PowerShell/Python, patching, repository search and
native subagents were callable. One native read-only retrieval worker reviewed
matrix code; the lead checked its actionable finding locally. External-provider
credentials/routing and price telemetry were not inspected or claimed available.
No external provider, web lookup, account connector or added service was needed.
The MIT license does not relax private-input or external-processing boundaries.

## Ranked engineering findings

Order balances impact, risk, leverage, confidence and cost; the active priority
and remaining backlog live only in [PROGRESS.md](PROGRESS.md).

| Rank | Finding and classification | Impact / confidence / approximate cost | Leverage |
| --- | --- | --- | --- |
| 1 | **Verified:** parent links are lost on XML round trip; a UUID-only repair also changes world geometry | High impact/confidence; small-to-medium coherent I/O slice | Establishes hierarchy/geometry regression foundation |
| 2 | **Verified:** two-node parent cycles are accepted | High: traversal can fail to terminate; high confidence; small validation slice | Protects every hierarchy consumer |
| 3 | **Verified:** non-baked global transform order contradicts the method contract | High impact; high confidence for root case; medium cost across nested/baked cases | Restores predictable editing operations |
| 4 | **Verified:** duplicate MOP display names lose operations during XML import | High: machining operation omitted; high confidence; small-to-medium cost | Preserves machining intent through interchange |
| 5 | **Verified:** export tree construction catches entity serialization exceptions and returns incomplete output | High: incomplete output appears successful; high confidence; medium cost for failure/atomic-write contract | Reliable success/error boundary for Python and future MCP |
| 6 | **Verified:** bare-filename state save returns without creating a file | Medium impact; high confidence; low cost | Small persistence regression and clearer I/O behavior |
| 7 | **Source-confirmed maintenance risk:** converted Rect duplicates primitive Tag construction | Low-to-medium impact; high confidence in duplication, future drift risk; low cost | One metadata owner when serialization next changes |
| 8 | **Known gaps / enhancements:** MOP registry migration, project-default import, curved bounds, transfer APIs, packaging/test coverage | Variable impact; source/TODO/spec evidence; separate bounded increments | Extend capability after fidelity and tests |

### Parent identity and transform reconstruction

`cambam_writer.build_xml_tree` obtains a `Primitive` from
`CamBamProject.get_parent_of_primitive` and passes it as `parent_uuid`.
`Primitive._add_common_xml_attributes` converts it with `str()`, producing an entity
representation instead of a UUID. `cambam_reader._reconstruct_primitive` cannot
parse that parent reference. A synthetic two-rectangle round trip kept both
primitives but lost the child's parent.

The entity writer exports `get_total_transform()`. The reader stores that world
matrix as `effective_transform` and later links parents without converting to
local coordinates. Editing only the synthetic XML parent field to the correct
UUID changed the child's world X translation from 10 to 20 after import.

**Negative knowledge:** a writer-only UUID substitution is insufficient. Reopen
that approach only alongside a tested world-to-local reconstruction rule. Likewise,
do not recursively apply a non-baked parent transform to descendants already
inheriting it; `transform_primitive` comments identify this double-application risk.

### Parent round-trip repair verification

2026-09-07: implemented the bounded parent XML repair. The writer now supplies
the parent's UUID. The reader snapshots XML world matrices and solves for each
child's local matrix before assigning it with the resolved parent link. Numeric
XML ID references accept strings and JSON integers; malformed/unresolved parents
retain the imported world pose without a parent. The current contract lives in
`structure_spec.md`, section 0, “XML parent identity and world-pose contract.”

Decision: a singular resolved parent fails the whole import through the existing
logged-error/`None` boundary, including when the child world matrix is compatible.
World-only XML cannot uniquely recover the local transform. Pseudoinverse recovery
and silent detachment were rejected because they introduce arbitrary local state
or lose hierarchy. Reopen if an explicit local-matrix metadata format or product
requirement supplies a reconstruction rule. Singular roots without children load.

Authored reproduction: `python -m unittest discover -s tests -v`. The initial
eight tests ran before runtime edits. A subsequent isolated temporary copy of the
original HEAD reader/writer with the expanded nine-test suite returned exit 1:
four failures (UUID emission, world reconstruction, reversed order, singular
parent rejection) and one error (numeric parent reference remained unlinked).
No working-tree runtime files were reverted for that comparison.

After the repair, all nine tests pass on the available Python 3.10.9 / NumPy
1.23.5 interpreter without installing dependencies. Tests inspect synthetic XML
IDs, parent tags and world matrices, and assert hierarchy in both directions,
primitive identities, layer membership and world geometry through two round
trips, including reordered XML. Numeric comparisons use `rtol=0, atol=2e-8`.
Missing/malformed/self references, singular parents and singular roots are covered.
`python -m compileall -q cambam_builder legacy_cambam_builder tests`, the runbook's
import/construct smoke command, and `git diff --check` pass.

This is automated acceptance of the bounded contract, not CamBam or packaging
acceptance. CamBam placement/geometry checks remain required. General transform
order, baking/Rect conversion, cycles, MOP migration and legacy behavior were not
repaired or established by these tests.

### Parent cycles

`CamBamProject.link_primitive_parent` rejects self-links but not ancestor cycles.
Create A, create B with parent A, then link A to B: it returns `True` and both
relationships exist. `Primitive.get_total_transform` walks parents in an unguarded
`while` loop. An initial traversal probe did not finish and was interrupted;
the bounded reproduction verifies the cycle without traversing it.

### Global transform order

`CamBamProject.transform_primitive` promises a global operation but computes
`effective_transform @ matrix`. On a root rectangle, translate by (10, 0), then
rotate 90 degrees about explicit origin (0, 0): the local origin remains at
(10, 0), where a global rotation yields (0, 10). The lead reproduced this through
the public convenience methods. A nested target requires conversion through its
parent frame; simply changing multiplication order everywhere is not an accepted
fix. Baked/non-baked equivalence needs independent tests.

### Other bounded observations and uncertainties

- `build_xml_tree` catches primitive/MOP exceptions and returns the remaining tree;
  artifact completeness must be checked in addition to successful file creation.
- `read_cambam_file` leaves project defaults as a TODO. `Mop.pid_source` remains
  the implemented MOP relationship source; central migration is an enhancement.
- Pline bulge bounds, arc bounds and text bounds are marked approximate in
  `cambam_entities.py`; rotated/sheared rectangle baking warns of approximation.
- The primary matrix encoder/decoder are internally paired. Their comments and
  alternative v2 functions do not establish CamBam compatibility or prove a
  layout defect. Do not replace them based solely on row/column terminology;
  use a known CamBam fixture and geometry checks. Invalid matrix lengths also need
  a controlled parsing contract; this pass did not broaden into parser hardening.
- No exhaustive source-system support matrix, supported-Python build validation,
  legacy compatibility audit or real machining acceptance was performed.

## Recommendation from this review

The parent round-trip finding warrants the smallest coherent repair across writer
and reader. Current scope, acceptance criteria and priority are owned exclusively
by [PROGRESS.md](PROGRESS.md). This dated review owns the defect evidence and
the reasons a writer-only repair is insufficient.

## Validation performed

Environment: available Python 3.10.9, NumPy 1.23.5 on Windows. No dependencies were
installed. No project-managed environment or lockfile existed. Results below are
local checks, not claims about every declared Python version.

| Exact command | Result |
| --- | --- |
| `git status --short` (before editing) | Clean |
| `python -m compileall -q cambam_builder legacy_cambam_builder` | Exit 0 |
| `python -c "from cambam_builder import CBProject; p = CBProject('smoke'); assert p.project_name == 'smoke'; print('import/construct OK')"` | Exit 0; import/construct OK |
| `python -m output.review_baseline` | Exit 0; parent loss and accepted two-node cycle confirmed |
| `python -m output.review_parent_transform` | Exit 0; corrected synthetic UUID alone gives X=10 -> 20 |
| `python -m output.review_transform_order` | Exit 0; global-origin rotation discrepancy confirmed |
| `python -m output.review_doc_links` | Exit 0; local Markdown file targets exist |
| `git diff --check` | Exit 0; only line-ending conversion notices |

The three synthetic diagnostic scripts and documentation-link checker are retained
locally under ignored `output/`; they are not a committed regression suite and
assert current defects, not desired behavior. They use synthetic data only and
temporary XML files. The initial unbounded cycle probe was interrupted and replaced
with bounded registry assertions. Future regression tests must assert desired
behavior using the scenarios above, not preserve those diagnostic assertions.

Working-agreement implementation and local documentation verification are complete.
Runtime repairs, package installation/build, automated regression coverage and user
acceptance remain unperformed. No staging or commits were made.

Suggested commit message: `docs: establish agent workflow and evidence-based project status`

## Phase 1 and 2 completion audit

Follow-up on 2026-09-07: the working tree was clean at the start. A native read-only
documentation worker independently checked gaps; the lead validated ownership and
data flow against targeted implementation symbols. No runtime changes, private
assets, dependency installation or external-provider processing were needed.

The initial pass covered the foundations. This follow-up closes gaps in current
architecture, interpreter selection, lifecycle/evidence recording and single-owner
acceptance criteria. The user requested completion of these phases; product repair
remains a subsequent increment.

| Requirement | Authoritative result / completion evidence |
| --- | --- |
| Inspect working tree; preserve changes | Clean initial status; documentation-only final diff |
| Locate entry points, architecture, status, plans and checks | README -> topic map -> named owners; existing tracker/test/CI limitations recorded |
| Current architecture and ownership | Specification section 0, grounded in project/entity/reader/writer symbols; target design remains explicitly separate |
| Authored/generated/runtime/legacy boundaries | Topic map artifact rules and specification module ownership |
| Toolchain, commands and troubleshooting | Development runbook with explicit interpreter selection and checks by change type |
| Status, backlog, decisions and rejected approaches | One PROGRESS surface; workflow lifecycle; dated review evidence with reopening conditions |
| Competing documents | Live acceptance moved from this review into PROGRESS; module overview moved from the map to the specification |
| Capabilities, privacy, security and licensing | Session observations above; capability discovery/delegation rules; private-data boundaries, trusted-pickle rule and MIT owner |
| Compact agent contract and judgment | Root AGENTS; minor reversible choices autonomous, material missing decisions escalated |
| Progressive context and no competing wiki | Workflow search examples, evidence-triggered expansion/stopping rules and documentation maintenance ownership |
| Agile lifecycle and acceptance separation | Explicit backlog/active/blocked/completion states and reusable completion record |
| Cost-aware delegation | Capability matrix, bounded packet, total coordination cost, exclusive ownership, evidence escalation and local lead validation |
| Verification and handoff | Per-change minimum evidence and exact-results template; future test entry point required before runtime closure |
| Avoid unnecessary infrastructure | Existing documents reused; no nested agent files, new dependencies, services or indexing systems |

Implementation: complete for phases 1 and 2. Documentation validation: complete.
User/production acceptance is not required for these repository workflow documents;
no source-system behavior changed. This does not claim user approval of runtime
behavior or CamBam acceptance. No unresolved user decision blocks use of the
working agreement. External tracker existence and full product compatibility remain
explicit uncertainties rather than assumptions presented as facts.

Follow-up checks (available Python 3.10.9, without installing dependencies):

- `git diff --check`: passed.
- `python -m output.review_doc_links`: passed; checked current local file targets.
  This pre-existing local helper is supplementary, not required infrastructure.
- README Python code block executed with `python -`: passed, including an assertion
  that the returned project contains the `outline` primitive.
- Inspected the final changed-file diff, ownership links and referenced heading
  targets. No runtime file changed; no staging, commits or publication performed.

Next increment: the selected parent round-trip slice in PROGRESS. Suggested commit:
`docs: complete project overview and agent working agreement`

## Phase 3 completion audit

Completed 2026-09-07 after the working agreement was established. The original
review supplied the main correctness findings and next-slice recommendation. This
follow-up adds bounded I/O and modularity coverage, local verification of three
additional failure cases, and explicit leverage in the ranking above. A native
read-only worker supplied focused findings; the lead checked their source evidence
and reproduced the two additional I/O defects. No product fixes were made.

### Additional evidence

- **Duplicate MOP names:** `cambam_reader._reconstruct_mop` assigns `mop_name` as
  the identifier, while `_register_entity` rejects duplicate identifiers. Creating
  two profiles with identifiers `first`/`second` but display name `Same`, targeting
  the same rectangle, is accepted by the public API. Write/read of that synthetic
  project returns one MOP from an original two. Fix identity reconstruction at the
  I/O boundary; do not silently require globally unique display names in callers.
- **Bare state path:** `save_state('state.pkl')` calls `os.makedirs('')`, catches
  the error and returns before writing. The lead reproduced this in a fresh
  temporary working directory and asserted no file exists. Other write errors in
  that method raise, while `read_cambam_file` generally returns `None` on failure.
  A future adapter needs a deliberate error mapping; wrapping these return values
  without checking outcomes is insufficient.
- **Incomplete XML:** with one rectangle, patch its class's `to_xml_element` to
  raise `ValueError('synthetic serialization failure')`; `build_xml_tree` returns
  normally, with zero `./layers/layer/objects/*` nodes. This upgrades the earlier
  source-level risk to an injected-failure reproduction. It does not establish
  filesystem atomicity or exhaustively test MOP failures.
- **Metadata duplication:** `Rect.to_xml_element` replaces the Tag after conversion
  to Pline, separately from `Primitive._add_common_xml_attributes`. The duplicated
  fields and filtering/formatting differ. This is a maintenance risk, not proof
  that every converted rectangle is currently wrong. Consolidate at the owning
  serialization abstraction when a covered change requires it.

### Architecture and coverage judgment

The project/entity/matrix/reader/writer split is adequate for the next increments.
The reader/writer directly depend on private registries and entities encode XML;
these are coupling points worth testing, not sufficient grounds for a broad layer
rewrite. Bidirectional indexes and duplicate entity/group metadata need invariant
tests before expanding mutation or transfer APIs. Do not start by migrating MOP
ownership: its live-group semantics are unresolved and would broaden the repair.

Source-system coverage is bounded by the reader tag maps and corresponding entity
encoders. Unsupported tags are skipped; project defaults and approximate curved
bounds remain documented gaps. Full fidelity needs authorized CamBam fixtures and
domain validation. This initial review is complete without pretending to be an
exhaustive feature, security or machining certification.

| Original phase 3 requirement | Result |
| --- | --- |
| Review correctness, architecture, duplication, I/O, validation, tests, modularity and source behavior | Ranked findings, additional evidence and architecture judgment above; earlier transform/hierarchy probes retained |
| Rank by impact, risk, leverage, confidence and cost | Updated ranking; PROGRESS owns execution order |
| Separate verified defects, hypotheses and enhancements | Classification explicit; source compatibility and future metadata drift not claimed proven defects |
| Check decisions/failures before recommending | Reviewed existing record; UUID-only and double-transform approaches remain rejected without new evidence |
| Recommend smallest high-impact slice with acceptance, owners, tests, user validation and rollback | Parent round-trip slice in PROGRESS remains the recommendation; newly found I/O defects do not force a wider first patch |
| Avoid broad speculative refactoring | No runtime edits; retain current module boundaries and repair demonstrated contracts |

Checks performed this follow-up: two synthetic programs run with PowerShell
here-strings piped to `python -`, both exit 0. The first used
`unittest.mock.patch.object(type(rect), 'to_xml_element', side_effect=ValueError(...))`
and asserted the original count was 1 and output count 0. The second used
`tempfile.TemporaryDirectory`, `save_state('state.pkl')`, and `build_xml_tree` /
`read_cambam_file`; assertions confirmed missing state output and MOP count 2 -> 1.
Inputs and assertion scenarios are specified above; no user fixtures were loaded.
`python -m output.review_doc_links` and `git diff --check` passed after documentation
updates. No packaging, server/client interoperability or CamBam acceptance was run.

All three initial phases are complete at their requested discovery/agreement/review
scope. Runtime implementation and user/domain acceptance remain pending. The new
MCP plan records user-authorized future direction, not a reason to defer the
selected correctness fix or claim an implemented integration.

Suggested commit: `docs: backlog stateless MCP integration and complete initial review`


### Parent-cycle rejection verification

2026-09-07: `link_primitive_parent` now walks the proposed parent's ancestor
chain before removing the child's old edge. Self/descendant links and already
cyclic candidate chains return `False` without mutation. Iteration with a visited
set avoids introducing a recursion limit or hanging on malformed prior state.
The mutation API owns this invariant; no transform or MOP refactor was needed.

The new cyclic-XML regression failed before the fix in both forward/reversed
XML order: three parent edges were retained instead of two. With the guard,
import rejects the closing edge and preserves all three world matrices and
geometries across another export/import. Which edge is rejected depends on XML
order; malformed cycles have no uniquely intended root. Existing singular-parent
import failure behavior is retained. Direct dictionary mutation/old pickle repair
and general traversal hardening remain outside this API guarantee.

The run also exposed an existing UUID-order-dependent singular-parent test:
either root or child can be the first singular parent encountered. Its assertion
now checks the reported child/parent pair against either valid failing edge.

Verification: `python -m unittest discover -s tests -v`,
`python -m compileall -q cambam_builder legacy_cambam_builder`, import/construction
smoke check, and `git diff --check` pass with the available Python 3.10.9 / NumPy
1.23.5 (no project environment or dependency changes). API tests check atomic
registry and transform preservation plus valid hierarchy mutations; XML tests
inspect parent edge counts, traversal, world poses and repeated round trips.
CamBam geometry/placement acceptance and packaging compatibility remain unverified.
Reopen if another supported relationship mutation bypasses the guard, or cyclic
XML requires an order-independent rejection policy.


### Parent A/B user acceptance

2026-09-07: the user confirmed all listed coordinates in both
`A_reference.cb` and `B_parent_roundtrip.cb` from the prepared parent validation
case. B achieves placement through transform properties; resetting them to
identity restores original local placement, as expected. This accepts the
synthetic parent XML/display criteria in the development runbook. CamBam version
was not supplied; no machining, curved geometry or general baking acceptance is
inferred. The one-off generator was archived under the same ignored local output
directory and removed from the tracked demos; original source remains in git
history, with reusable regression coverage retained in `tests/`.


### Full-bake hierarchy verification

2026-09-07: `bake_primitive_transform(..., recursive=False)` removed the target
matrix without compensating children, moving their world coordinates. Two new
straight-polyline regressions failed before repair (root bake and child bake under
a transformed parent). The fix always transfers the removed local matrix to direct
children; recursion controls whether their geometry is baked. Reported recursive
failure results now propagate. This is not rollback/transaction support.

Five new tests cover both nonrecursive cases, recursive root/child baking, and
recursive singular scaling. They assert world geometry, selected identity matrices,
untouched descendant local geometry when nonrecursive, UUID/object/relationship,
layer and group preservation, idempotence, and two XML round trips. All 18 tests
pass with `python -m unittest discover -s tests -v` on Python 3.10.9 / NumPy 1.23.5.
Syntax compilation, import/construction and `git diff --check` also pass.

A one-off ignored generator produced an independent explicit-coordinate A and
fully baked B under `output/full-bake-validation-n132g8ro/`, checking the loaded
coordinates/counts. Manual criteria live in the development runbook; user
acceptance is pending. Prior parent display acceptance does not establish baked
coordinate fidelity. General transform, component and curved-shape baking remain
unverified. In particular, `transform_primitive` currently postmultiplies despite
claiming global application; that is the next bounded slice. Pline bulges are
preserved under reflections/nonuniform scales and need a separate curved-geometry
contract. Reopen full-bake hierarchy repair if descendant pose changes under the
tested straight-polyline affine conditions.


### Full-bake A/B user acceptance

The user confirmed all full-bake A/B endpoints, identity matrices and final stored
coordinates in both files, and the intentional separate-versus-shared layers.
They subsequently confirmed understanding that A was constructed directly and B
was baked before export, and authorized the next increment after committing.
This completes the synthetic full-bake display acceptance; CamBam version was not
supplied. The original verification entry's pending acceptance is superseded by
this report. Broader curved/component baking and production machining remain
unverified. Future before/after cases should include the pre-operation file when
it helps the user inspect what changed.


### Global transform ordering verification

2026-09-07: the pre-fix root probe used local endpoints (1,0)/(2,0), an existing
translation (10,0), and a requested global 90-degree rotation. It produced
(10,1)/(10,2), failing the expected (0,11)/(0,12). `transform_primitive` previously
postmultiplied the effective matrix and baked the same world matrix into every
node's local geometry. Both violate the public global-coordinate contract.

Matrix mode now solves in the parent frame; baked mode precomputes a conjugated
operation in every affected original world frame, preserving effective matrices.
Invalid affine input and singular required frames fail before mutation. Even a
compatible singular-frame operation is rejected because the local reconstruction
is not unique. No pseudoinverse or reparenting fallback is used. Geometry-baker
failures are not transactional; entity-specific approximation/error handling is
unchanged. `combine_transformations` documentation now describes its existing
rightmost-first point application; its implementation is unchanged.

`python -m unittest discover -s tests -v`: all 26 tests pass (eight new ordering,
wrapper, singular/invalid-input and repeated round-trip regressions).
`python -m compileall -q cambam_builder legacy_cambam_builder tests`, import/construct
smoke check and `git diff --check` pass on Python 3.10.9 / NumPy 1.23.5.
The tests verify affected world geometry, unchanged frames/geometry where required,
relationship integrity and XML counts/UUIDs/layers across two round trips.

Prepared ignored Before/A/B files were parsed and reloaded numerically. A has
identity matrices; B retains the root world matrix and changes both descendant
world translations by (+5,-3). The development runbook owns precise manual
criteria; user acceptance remains pending. Existing full-bake acceptance is
recorded above and need not be repeated. Curved/component baking and alignment
remain outside this increment. Reopen if supported callers relied on the former
local-coordinate ordering, or a singular-frame policy needs expansion.


Final bounded review reproduced two additional failures, now covered by tests:
a baked 1e-9 translation was discarded by Pline's approximate identity check,
and a nested-list matrix containing `10**1000` raised `OverflowError` instead of
returning `False`. Pline geometry baking now skips only exact identity; matrix
conversion catches overflow. The final `python -m unittest discover -s tests -q`
run passes all 26 tests. No curved-bulge fidelity claim follows from this change.


### Global-transform acceptance and workstream checkpoint

The user confirmed A/B endpoints and placement, transform properties in B versus
explicit coordinates in A, and the correct (+5,-3) child/leaf movement from Before.
They reported committing the implementation. This supersedes pending manual
acceptance in the earlier verification record; CamBam version was not supplied.

The parent/hierarchy, full-bake and global-transform straight-polyline line has
reached its scoped stopping condition. Component baking is distinct (fold one
existing component into geometry while retaining the rest), but adjacency alone
is insufficient to prioritize it. The documented duplicate-name MOP import loss
(two operations become one; `_reconstruct_mop` uses the display name as identifier)
has greater immediate impact on preserving machining intent. It is now the next
recommended outcome, without expanding into registry migration. Remaining
transform gaps stay explicit in PROGRESS with reopening criteria. A fresh session
can start from repository documentation without retaining this conversation.

## MOP identity round-trip verification

2026-09-07: fixed the I/O identity boundary in `Mop._add_common_mop_elements`
and `_reconstruct_mop`. Previously display Name became the unique identifier,
merging/rejecting duplicate names and replacing UUIDs. JSON Tag now carries
`user_id`/`internal_id`; restoration precedes registration. Reconstruction follows
geometry to detect cross-entity collisions. Native test delegation was limited to
the synthetic test module; lead reviewed and extended it and verified locally.

Complete valid colliding identity metadata fails the entire import, preserving no
partial result. Missing/incomplete/malformed metadata gets a new UUID/identifier;
this keeps ordinary metadata-free CamBam files usable without unique names.
No registry migration, group semantics change or broad parameter parser expansion.
Reopen metadata policy for an authorized real interoperability fixture showing
CamBam strips/rejects Tags or requires different identity handling.

Verification on existing Python 3.10.9 / NumPy 1.23.5 (no project venv):

- `python -m unittest discover -s tests -p test_mop_roundtrip.py -q`: 5 tests pass.
  Four MOP types, same-part/across-part duplicate names, name collisions with
  layers/parts, UUID/identifier lookups, per-part order, target UUIDs, representative
  supported explicit parameters and two round trips. Also legacy fallback and
  subsequent identity stability, malformed Tags and conflicting MOP/geometry IDs.
- Loading `git show HEAD:cambam_builder/cambam_reader.py` into an isolated Python
  module and substituting its reader in the new suite gives 7 assertion failures
  across 5 tests, zero errors. Working files were not reverted. The fixed writer
  was retained, proving the old reader loses operations even with identity Tags.
- `python -m unittest discover -s tests -q`: all 31 tests pass.
- `python -m compileall -q cambam_builder legacy_cambam_builder`, import/construct
  smoke and `git diff --check`: pass.
- Generated and inspected `output/mop-validation-kpk5hxop/` A/B XML: two Profiles
  named Duplicate, separate left/right square references, ordered depths -1/-2
  and feeds 300/600 retained after two round trips. Reference omits new MOP Tags.
  Initial generator incorrectly assumed UUID-assigned primitive IDs followed
  creation order; corrected its assertion to resolve IDs through primitive Tags.

Implementation and automated verification are complete. CamBam acceptance is
pending under [the prepared criteria](DEVELOPMENT.md#manual-mop-identity-acceptance);
no machine execution is required. Existing parameter coverage and Default/Value
limitations remain; this is not production machining acceptance. The next useful
increment is export failure handling, whose silent omissions remain a data-loss
risk. Suggested commit: `fix: preserve MOP identities across XML round trips`.

## MOP A/B user acceptance

2026-09-07: the user confirmed that all described expected outcomes were validated
and true. This accepts both prepared files loading successfully, two enabled
Duplicate Profiles under Part1 in the expected order, left/right square targets,
depths -1/-2, feeds 300/600, Outside, diameter 3, depth increment 0.5 and spindle
12000. CamBam version was not supplied. This supersedes the pending acceptance
in the preceding verification record; no repeated check is required without a
relevant behavior change.

Scope is the synthetic load/display/property case, not machine execution or
complete parameter compatibility. Identity/registry preservation remains covered
by automated tests. No runtime changes were made for this acceptance update;
document consistency and `git diff --check` were checked. The local A/B artifacts
are retained. Export failure behavior remains the next implementation priority;
this accepted slice is a good fresh-session breakpoint.

## Export failure and state path verification

2026-09-08: completed the export error boundary and separate bare-filename
persistence slice. `build_xml_tree` re-raises encoder/resolution errors, retaining
their original types and logged entity context. Ordered missing layers/parts and
resolved targets without XML IDs now fail instead of being skipped. XML saves
serialize into a unique sibling temporary file, close it, then use `os.replace`.
Failure cleanup preserves the original exception. Formatting failures propagate,
except unavailable Python 3.8 indentation retains the existing unindented fallback.
`save_state` creates the absolute parent directory, so bare names work, and raises
directory errors. The lasting contract and limits live in specification section 0.

Evidence:

- The initial four `test_export_failures.py` tests against the old writer produced
  14 failing subtests: swallowed entity errors, partial destination writes and
  missing replacement behavior. The fixed writer passed those tests. Expanded
  coverage checks preparation/resolution failures, missing ordered entities,
  bare filenames and unavailable indentation (eight tests total).
- Loading only HEAD's `save_state` method into the current class in a separate
  Python process produced two failures across the three state tests: no bare-name
  file and no raised directory error. No working files were reverted.
- `python -m unittest discover -s tests -v`: 42 tests passed on the available
  Python 3.10.9 / NumPy 1.23.5; no repository virtual environment was present.
  Tests assert unchanged bytes or absent destinations after failures, no leftover
  temporary files under normal cleanup, successful XML counts/UUIDs/MOP targets
  and depth, plus restored pickle primitive project links. Existing geometry and
  repeated round-trip regressions remain green.
- `python -m compileall -q cambam_builder legacy_cambam_builder`, the runbook
  import/construct smoke command and `git diff --check` passed.
- Supplementary local logs are in
  `output/export-persistence-89fec11716684f3db4282876edf95cb8/`;
  reusable failure fixtures are in the two new test modules.

Direct destination writes were rejected because late serialization/I/O failure can
truncate a previous good document. Best-effort encoder continuation was rejected
because it falsely signals successful export. No custom exception hierarchy or
dependency was needed. Atomic pickle writing, crash durability, comprehensive
corrupt-registry validation, PID filtering/group semantics and full schema
validation remain outside this bounded contract. Reopen those areas for a
reproduced failure or an explicit workflow requirement.

Implementation and automated verification are complete. No manual CamBam check
adds evidence for this filesystem/exception change; successful XML structure is
unchanged and existing synthetic product acceptance remains scoped as recorded.
This does not establish broader production machining acceptance. Overall priority
now returns to remaining transform fidelity; bound it with a reproduced case
before choosing the repair. This is a good fresh-session breakpoint: required
evidence and limits are saved, with no pending acceptance or decision.
Suggested commit: `fix: propagate export failures and support bare state filenames`.

## Rect baking loss investigation

2026-09-08. Scope: reproduce and bound the next transform defect, without changing
runtime behavior. The candidate is confirmed in `Rect.bake_geometry`: after
transforming its four corners, it saves only their minima/maxima as a new local
axis-aligned Rect. The original outline is irrecoverable after this assignment.
The current representation cannot encode a general rotated rectangle or sheared
parallelogram with an identity matrix. Bounds-only comparisons hide the loss.

The warning is insufficient: `is_rectangular_after_transform` checks
perpendicularity, not axis alignment, so pure rotation is accepted. The additional
matrix object-identity condition suppresses warnings for separate explicit bake
matrices. Project bake methods may return `True` despite the changed outline;
the writer's error propagation cannot detect successful but lossy geometry edits.

### Reproduction and verification

Reusable fixture: `tests/test_rect_bake_defect.py`, a root Rect at `(0,0)` with
width 4 and height 2. Commands run from the repository root:

- `python -m unittest tests.test_rect_bake_defect -v`: eight characterization
  tests passed. Known-defect tests assert the current wrong result explicitly;
  they are not claims of repaired fidelity.
- `python -m unittest discover -s tests -v`: all 50 tests passed after adding
  translation/scale controls and XML count/UUID/type checks. Python 3.10.9,
  NumPy 1.23.5; no repository virtual environment was present.
- `python -m compileall -q cambam_builder legacy_cambam_builder` and the runbook
  import/construct smoke command passed. `git diff --check` passed.

| Operation on the fixture | Required outline area | Actual baked area | Observed boundary |
| --- | --- | --- | --- |
| Full bake of 45-degree rotation about origin | 8 | 18 (+125%) | Success, identity matrix, no entity warning |
| Full bake of unit X shear (`x += y`) | 8 | 12 (+50%) | Success with loss warning |
| Global bake of the same shear from identity | 8 | 12 | Success without entity warning |
| Explicit-matrix 45-degree rotation bake | 8 | 18 | Success, original identity matrix retained |
| Rotation-component bake of pure 45-degree rotation | 8 | 18 | Success, identity matrix |

The shear matrix is `[[1,1,0],[0,1,0],[0,0,1]]`; the existing helper spells this
`skew_matrix(angle_y_deg=45)`. Correct shear corners are `(0,0), (4,0), (6,2),
(2,2)`; baking produces `(0,0), (6,0), (6,2), (0,2)`. At 45 degrees the rotated
outline fits in a square of side `3*sqrt(2)`, but is not that square.

Translation, diagonal scaling, reflection and 90-degree rotation preserve the
tested corner sets and areas. Two XML round trips preserve the already-damaged
rotated Rect, its UUID, count and identity matrix. Unbaked root rotation and shear
instead preserve their expected corners/area and UUID/count through two trips:
rotation stays a Rect with a matrix, shear is serialized as a closed Pline.
These root controls do not establish conversion fidelity under a parent.

Investigation and automated checks are complete; runtime repair remains backlog.
No manual acceptance is needed for this numerical reproduction. Hierarchies,
nonidentity frames in explicit/global bake mode, mixed component decomposition,
degenerate shapes and general curved geometry were not validated by this slice.
The fixture deliberately isolates representation loss from those other contracts.
This is a good fresh-session breakpoint: evidence, limits and next priority are
persisted. Suggested commit: `test: reproduce and bound Rect baking geometry loss`.

### Bounded repair recommendation

Prioritize exact Rect baking over MOP registry migration: this is reproduced
geometry corruption through existing public operations, whereas registry migration
still needs group-source semantics. Start with full local-transform baking of a
Rect and its hierarchy, using a closed straight polyline when the transformed
outline cannot be represented by an axis-aligned Rect. Preserve the outline for
axis-preserving cases without unnecessary conversion. Extend the same geometry
policy to explicit/global bake entry points; do not couple the repair to general
component decomposition, curved entities or alignment ordering.

Acceptance for that repair:

- Compare all four transformed vertices/closed edges and area, allowing cyclic
  reordering or reversed winding, at absolute tolerance `1e-10` in memory and
  `1e-8` after two XML round trips with adequate output precision. Equal bounds
  alone do not pass. Include rotation, shear and axis-preserving controls.
- Preserve UUID, identifier, description, layer/group membership, parent/child
  links and MOP target resolution through any representation change. Specify
  behavior of existing Python object references before implementing conversion;
  `to_pline_representation` currently creates a new identity and is not a safe
  registry replacement operation by itself.
- Preserve descendant world geometry for recursive and nonrecursive full baking,
  reset only the matrices required by that API, and retain effective matrices for
  explicit/global geometry baking. Test a transformed parent and unaffected sibling.
- Never report success after approximating the outline. If a supported conversion
  cannot be completed, surface failure before destructive geometry/registry edits;
  do not substitute a warning for the geometry contract.
- Replace known-loss characterization assertions with preservation regressions
  during repair. Prepare and inspect synthetic CamBam A/B files for any changed
  representation, then record user display acceptance separately.

An export-only workaround cannot recover discarded corners. A blanket rejection
of all rotation would unnecessarily reject quarter turns. General transform
refactoring is deferred: reopen component ordering, alignment, curved geometry
or degenerate/nonfinite edge cases only for a failing workflow fixture or an
explicit dependency. No manual CamBam check is needed to establish this numerical
defect; no runtime behavior or previously accepted display fixture changed.


## Rect baking repair verification

2026-09-08. The repair replaces bounding-box baking with a closed-outline
comparison and in-place conversion to a four-vertex, zero-bulge Pline when needed.
Axis-aligned results remain Rects. The owning contract is
[Rect baking representation](../structure_spec.md#rect-baking-representation-contract).
The same Python object survives conversion, including UUID, project link and
metadata, so registries, MOP source resolution and external references continue
to designate it. Callers must accept the runtime type change; Rect-specific
corner/width/height attributes are removed. Creating a separate replacement
would leave external references stale and require registry coordination, while
retaining an AABB or rejecting all rotation would violate the repair criteria.

The explicit project helper now calls the entity's explicit-matrix API directly,
so validation failure does not leave a temporary effective matrix installed.
Full-bake identity detection uses exact equality to avoid dropping small but
observable transforms. Rect input/corner validation occurs before mutation and
raises on failure. This does not establish transactional subtree baking or
repair general component ordering, curved geometry or alignment.

Verification on system Python 3.10.9 / NumPy 1.23.5 (no project venv):

- `python -m unittest tests.test_rect_bake_defect -q`: nine tests pass, checking
  closed edges/area, axis-preserving controls, recursive/nonrecursive full bake
  under a transformed ancestor, explicit/global matrices and descendants,
  repeated baking, pickle and two XML round trips, identity/metadata/MOP targets,
  small shear and invalid explicit input without mutation.
- The worker substituted the original HEAD Rect method in memory: eight
  regression failures confirmed detection of the former loss (before the two
  additional small-shear/error tests were added).
- `python -m unittest discover -s tests -q`: all 51 tests pass. Expected error
  logs come from negative-path tests; no test failures.
- `python -m compileall -q cambam_builder legacy_cambam_builder`, the runbook
  import/construct smoke command and `git diff --check` pass.
- `python output/rect-bake-validation-20260908-a/generate.py`: generated and
  inspected reference/result XML, three closed four-point zero-bulge Plines,
  identity matrices, metadata/parent linkage and vertices within `1e-8` after
  two round trips. [Manual criteria and files](DEVELOPMENT.md#manual-rect-bake-acceptance)
  use display tolerance `0.01`. These ignored artifacts remain available.
- Native review found no additional runtime defect; its in-progress test shape
  finding was resolved by normalizing Pline triples to XY in the final tests.

Implementation and automated verification are complete. Manual validation adds
CamBam renderer/load evidence beyond library XML reconstruction; report version,
A/B pass/fail and any differing outline or matrix against the prepared criteria.
CamBam user display acceptance is now complete (see below); previous accepted fixtures are
unchanged and need no repetition. Reopen this scope for a failing supported
outline/relationship fixture, object-layout compatibility need, or reported
CamBam display discrepancy.

Suggested commit: `fix: preserve Rect outlines when baking rotation and shear`.
The next priority is defining MOP group-source
compatibility before registry migration, as recorded in [PROGRESS.md](PROGRESS.md). This is a
coherent accepted breakpoint; use a fresh session for that distinct design scope. Nothing was staged or committed.


## Rect baking display acceptance

2026-09-08. The user confirmed all validation conditions were met, a full match
and identity transforms for the prepared Rect A/B case. This accepts loading and
displaying the three closed outlines with the documented vertices within `0.01`
drawing units and identity matrices in A/B. CamBam version was not supplied.
[Files and retained criteria](DEVELOPMENT.md#manual-rect-bake-acceptance) remain
available; no regeneration or repeat acceptance is required for unchanged code.

Implementation, automated verification and synthetic display acceptance are
complete. This acceptance does not extend to production toolpaths, other curved
entities or general component ordering. No runtime code changed in this round;
`git diff --check` passed and the documentation links/status were reviewed.
No automated tests were rerun because only acceptance documentation changed.
Suggested commit: `docs: record Rect baking display acceptance`.
