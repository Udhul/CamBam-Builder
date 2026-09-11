# Initial workflow and engineering review — 2026-09-07

## Delegation routing correction - 2026-09-09

The initial revision mixed project policy, role selection, provider setup and worker
execution behavior. It also omitted the Sol role registration, duplicated stale role
descriptions, used a false-precision score and recorded availability before an
invocation passed. The corrected ownership is: `AGENTS.md` states the short project
rule, [MODEL_ROUTING.md](MODEL_ROUTING.md) is the sole role selector,
[WORKFLOW.md](WORKFLOW.md#delegation-execution) owns packet/integration mechanics, and
user-level role TOMLs govern only the selected worker's execution.

Provider evidence: a direct read-only `codex exec` using `model_provider=openrouter`
and `z-ai/glm-5.3-flash` returned the exact requested marker. A native-parent custom
GLM spawn failed with HTTP 400 because Codex 0.150.1 reapplied the ChatGPT account
transport to the OpenRouter child. This is a mixed-provider limitation, not an API-key
or model-name failure. The corrected user-level `openrouter-glm` profile selects
OpenRouter/GLM and overrides the inherited subagent model/effort defaults. A fresh
profile-backed process spawned `glm_retriever`; the consolidated-profile retest returned
`GLM_SUBAGENT_ROUTE_OK Model routing`. A separate fresh native process spawned the newly
registered `sol_scoped_worker` and returned `SOL_SUBAGENT_OK`. All tests were read-only.

Reopen routing if reviewed worker output shows regressions or coordination cost
erases the expected benefit. Reopen provider setup if a profile-backed GLM subagent
fails or a later Codex release claims cross-provider children are supported; retain
exact errors rather than inferring model quality or access.

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
`docs/structure_spec.md`, section 0, “XML parent identity and world-pose contract.”

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
acceptance in the earlier verification record. Later user clarification establishes
CamBam Plus 1.0 as the validation environment.

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
12000. Later user clarification establishes CamBam Plus 1.0 as the validation
environment. This supersedes the pending acceptance
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
[Rect baking representation](structure_spec.md#rect-baking-representation-contract).
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
drawing units and identity matrices in A/B. Later user clarification establishes
CamBam Plus 1.0 as the validation environment.
[Files and retained criteria](DEVELOPMENT.md#manual-rect-bake-acceptance) remain
available; no regeneration or repeat acceptance is required for unchanged code.

Implementation, automated verification and synthetic display acceptance are
complete. This acceptance does not extend to production toolpaths, other curved
entities or general component ordering. No runtime code changed in this round;
`git diff --check` passed and the documentation links/status were reviewed.
No automated tests were rerun because only acceptance documentation changed.
Suggested commit: `docs: record Rect baking display acceptance`.

## Rest machining future-feature planning

2026-09-08. Recorded the user's five requested outcomes as a low-priority optional
feature in [PROGRESS.md](PROGRESS.md#remaining-backlog-in-order), with one
[plan owner](REST_MACHINING_PLAN.md) for problem, methods, dependencies and future
acceptance. Existing priorities and runtime behavior are unchanged.

Reasoning separates ideal tool reachability from trajectory-derived remaining
stock, pure rest from access expansion, and surface V-corner sharpening from
complete volumetric removal. Corner overcut is modeled from its sweep rather
than presumed to eliminate all rest. Native CamBam V-corner support and the
XYZ-polyline/Engrave proxy are attributed to official documentation in the plan.
Region and Z are feature prerequisites, not newly promoted foundation work.

The user-authorized Region example was inspected locally only and left unchanged.
Its typed XML outer/hole curves and the current reader's Region/Z gaps are
recorded in the plan. No private fixture content was sent to external providers.
The proposal includes numerical references, workflow helpers, provenance and
invalidation, five separate acceptance outcomes and promotion/stopping criteria.
Native technical review prompted explicit gates on lossless XYZ proxy export
and arc-extrema-aware contour bounds before using bulged Regions in analysis.
The user clarified that standalone programmatic calculation is essential: CamBam
cannot supply a headless engine/API to this workflow. Removed the proposed
toolpath-acquisition/scripting dependency in favor of local supported MOP and
geometry algorithms. Optional CamBam exports are validation fixtures only. The
retained XYZ/Engrave proxy is a file encoding for paths calculated by the framework.

Documentation verification: local links/anchors and analytic example values
checked; `git diff --check` passed. No runtime tests or manual machining validation
are needed for this documentation-only increment. Implementation, CamBam path
acceptance and production machining remain future work. Planning is a coherent
fresh-session breakpoint; the next task remains MOP group-source compatibility.
Suggested commit: `docs: plan optional rest machining and V-cutter workflows`.


## Shape parity separated from rest machining

The user clarified that Region and all-shape Z support are missing core CamBam
feature parity, independently useful and upstream of rest machining. Moved their
repository evidence and implementation contract to [SHAPE_PARITY_PLAN.md](SHAPE_PARITY_PLAN.md).
The [rest plan](REST_MACHINING_PLAN.md#upstream-dependencies) now consumes those
contracts instead of owning their implementation. This supersedes the earlier
planning decision that treated them as support inside the optional rest workstream.

Backlog placement is after MOP design/ownership and geometry/relationship
correctness, before packaging, MCP and rest machining. Existing patterns remain
the implementation approach; tuple/bulge compatibility, shape elevation and matrix
Z semantics require focused evidence. No runtime behavior changed. Local links,
plan ownership and `git diff --check` were checked; runtime tests and manual
validation are not needed for this documentation change. The next immediate task
remains MOP group-source compatibility; the distinct parity scope is ready for a
fresh session when its priority is reached.
Suggested commit: `docs: separate and prioritize Region and Z shape parity`.


## MOP group-source compatibility

**Policy superseded by [user clarification](#development-compatibility-priority);
observed behavior and test results below remain historical evidence.**

2026-09-08: compatibility definition and characterization before backlog item 1's
registry migration. Runtime behavior and XML schema are unchanged.

Evidence: `CamBamProject._add_mop_internal` retains string sources and resolves
list entries once to primitive UUIDs; `resolve_pid_source_to_uuids` performs live
group lookup, filters missing primitives and returns sorted unique UUIDs.
`Mop._add_common_mop_elements` writes concrete XML references and identity-only
Tags. `_reconstruct_mop` and the reader's deferred linking pass reconstruct UUID
lists, regardless of primitive group metadata.

Decision: preserve live groups in memory and snapshots after XML import. The
then-current contract constrained migration to retain source intent and the public
`pid_source` surface; this is superseded by the development compatibility priority
and [current target contract](structure_spec.md#mop-target-ownership-contract).
Freezing all sources at MOP creation would change existing live behavior. Inferring
groups from matching targets is ambiguous (multiple groups may match), and adding
live-group Tag metadata would change how reimported jobs respond to membership
edits. Neither is necessary for ownership migration. Reopen persistent live XML
sources only for an explicit workflow with a defined opt-in/schema and external
edit conflict policy; do not silently retarget existing files.

Scope excludes registry implementation, old-pickle compatibility changes,
Default/Value parameter fidelity and broader malformed metadata handling. Existing
MOP identity tests already cover malformed JSON/nonobject/invalid UUID Tags and
identity collisions; they do not establish exhaustive metadata/default coverage.

Verification (available system Python 3.10.9 / NumPy 1.23.5; no project venv):

- `python -m unittest discover -s tests -p test_mop_group_sources.py -v`: six pass.
- `python -m unittest discover -s tests -v`: all 57 pass.
- `python -m compileall -q cambam_builder legacy_cambam_builder`: pass.
- `python -c "from cambam_builder import CBProject; p = CBProject('smoke'); assert p.project_name == 'smoke'; print('import/construct OK')"`: pass.
- `git diff --check`: pass; changed documentation links/headings reviewed.

The new tests cover membership/deletion/recreation, group-versus-list selection,
normalization and caller-list isolation, empty/missing sources, source mode
switching, UUID non-rebinding, and all four MOP types through two XML round trips.
Synthetic XML assertions inspect concrete refs; reimports check target snapshots,
part order, group metadata and sample geometry before testing membership edits.
These are characterization tests against unchanged runtime, not a runtime defect
repair, so no before-fix failure is claimed. Manual validation is
not required for this slice because no emitted schema or runtime behavior changes.
Existing synthetic CamBam acceptance is not expanded to production machining.


## Development compatibility priority

2026-09-08 user clarification: no important files depend on older framework
versions; the framework has not been used in production. Correct architecture
and the core model take priority over backward compatibility with those versions.
The required compatibility boundary is CamBam `.cb` files, including framework
output and files created or modified directly in CamBam.

This supersedes the prior obligation to retain `pid_source`, its source modes
and old-pickle handling during registry migration. Breaking APIs/storage are
allowed; update repository callers/tests together and avoid legacy adapters or
duplicate ownership. The six characterization tests remain evidence of current
behavior, not a veto on redesign. Native CamBam edits must not be silently undone
by stale framework metadata. Domain semantics and external-format fidelity still
need design and verification; this clarification does not claim new format support.

Updated the specification policy, backlog and test runbook. Documentation-only
change: references reviewed and `git diff --check` passed; no runtime tests or
manual CamBam validation needed. Next: implement the MOP core-model increment
against this policy. Suggested commit: `docs: prioritize core model and CamBam interchange`.


## MOP core ownership and interchange redesign

2026-09-08 (continued after a usage-limit interruption). Target ownership moved
from MOP instances to one project registry. Explicit UUID sets support stable
selection; a separate live group-name mode supports incremental construction.
Bare-string ambiguity and silently skipped invalid API targets were removed.
Setters validate before mutation, deletion cleans references, and XML snapshots
remain authoritative on import. No old API adapter or pickle migration was added.

Rejected restoring live-group mode from framework Tags: CamBam can edit native
primitive IDs independently, so group restoration could silently undo a native
edit. The native list is now the import authority even when metadata survives.
CamBam's [Machining Basics](https://www.cambam.info/doc/plus/cam/Basics.htm)
documents target editing, operation reordering and Default inheritance.
Its [CAM Styles guide](https://www.cambam.info/doc/fr/cam/CAMStyles.htm)
describes Style/StyleLibrary resolution through the hierarchy and external style
libraries. This supports retaining native state/context, not evaluating CAM styles
inside this library.

The interrupted parameter work had parsing without state-aware encoding. A focused
native-context regression reproduced dropped global Style/StyleLibrary and
ClearancePlane values before capture/serialization was integrated. The replacement
must retain native parameter content and explicit Default/Value state while allowing
Python edits; synthetic tests do not establish native application acceptance.

Implementation and automated verification are complete for this slice. The user
accepted the prepared A/B CamBam display/property case on 2026-09-08; CamBam
version was not supplied. The final
checks were:

- `python -m unittest discover -s tests -v`: 65 tests passed; the retained log is
  `output/mop-core-checks-ncsbkpgz/suite.log`.
- `python -m compileall -q cambam_builder legacy_cambam_builder tests demos`:
  exit 0.
- `python output/mop-core-validation-1roowlbtpza/generate.py`: exit 0; counts,
  Profile/Pocket/Engrave/Drill targets, depths and feeds verified. Lead XML
  inspection confirmed rectangle coordinates, triangle vertices and drill points.
- The tightened native comparison helper passed synthetic C-to-D checks for
  operation names/order, targets and exact parameter/state sets. The clearly
  named Synthetic_C/D files remain in the checks directory; these are not
  native CamBam acceptance.
- Import/construct smoke, `git diff --check`, and local Markdown path/anchor
  validation passed. Independent read-only reviews found no material target or
  parameter issue; the lead integrated subsequent assignment/state fixes.
- `test_mop_context` reproduced the lost global `Style`/`StyleLibrary` context
  before reader capture and now passes after the fix.

The native edit is retained as
[`C_native_edited.cb`](../output/mop-core-validation-1roowlbtpza/C_native_edited.cb).
The C-to-D import/export comparison exited 0 and preserved five enabled
operations in the order Drill, Engrave, Pocket, Profile, Profile. It preserved
the operation names/order/targets and exact parameter/state sets: the fourth
Profile targets `pocket-square`, has CutFeedrate 450 and ClearancePlane Default,
and the last Profile targets `profile-square`. Independent XML comparison also
matched all four shapes' coordinates, closure flags and identity transforms; the
native check logs report no warnings or errors. See
[`C_framework_roundtrip.cb`](../output/mop-core-validation-1roowlbtpza/C_framework_roundtrip.cb)
and the adjacent inspection JSON, `native-check.log` and
`native-geometry-check.log`.

Final native acceptance was reported on 2026-09-08 using CamBam Plus 1.0:
`CamBam.CAD` 1.0.7364.41819, `CamBam` 1.0.7364.41821, build 2020-02-29
23:13:58. C and D loaded without error and every stated display/property check
matched. This closes the supported MOP interchange slice. No full `.cb` fidelity,
production toolpath or broader production acceptance claim is made. Default
fields are cached but CAM styles are not evaluated; nested state setters are
intentionally unavailable while imported nested state is preserved. Unknown MOP
types remain skipped. Existing Python 3.10.9 and NumPy 1.23.5 were used; no
virtual environment or dependency changes were made.

The user clarified that CamBam Plus 1.0 itself saves `.cb` files with XML
`Version="0.9.8.0"`. That marker is therefore a legacy file-format value and is
not evidence that CamBam 0.9.8 created the file. Use the CamBam Plus 1.0 baseline
above for the user's past and future validation reports unless they explicitly
state a different application version. Keep the framework marker at `0.9.8.0`:
using `1.0` would depart from the native producer without evidence that the field
is an application label or that the alternate value improves compatibility.
Reopen this decision for a documented format revision or controlled native test.

The next backlog work is split into curved-geometry bounds (1a) followed by
copy/transfer utilities (1b). Starting with 1a is a coherent fresh-session
breakpoint with implementation, automated evidence, and accepted A/B and native
C/D results persisted. No staging or commit was performed. Suggested commit:
`refactor: centralize MOP targets and preserve native parameters`.

## Model routing and curved-workstream split

2026-09-08: the next backlog work was split after assessing its reasoning and
coordination risk. The combined bounds plus copy/transfer item scores about 11/13
because transfer semantics are registry-coupled and its UUID/collision contract
is not yet settled. Treating it as one undivided implementation would justify
Astra. Splitting it makes curved bounds (1a) a bounded Sol task and leaves
copy/transfer (1b) as a contract-first follow-up.

Decision: start the next session with GPT-5.6 Sol at high or xhigh reasoning. Use
native Luna `project_explorer`, `focused_implementer` and `deep_reviewer` roles
for bounded retrieval, implementation/tests and review. Escalate the main thread
to GPT-6 Astra if 1b must be designed and implemented as one change or if the
UUID/collision contract remains unresolved. The reusable factors, thresholds,
role mapping and privacy constraints are in [MODEL_ROUTING.md](MODEL_ROUTING.md).

Acceptance for the documentation decision: the topic map links the routing
guide; `PROGRESS.md` owns the 1a/1b ordering and next-session handoff; and
`WORKFLOW.md` points to the reusable model-selection rules. No runtime behavior
changed, so runtime tests and CamBam validation are not required for this slice.

## Curved geometry bounds verification

2026-09-08: endpoint-only Pline bounds and full-circle Arc bounds reproduced the
documented curved-geometry gap. The bounded repair adds analytic directed-sweep
extrema for Arcs and for every active Pline bulge segment. It evaluates extrema
after the complete local-plus-ancestor affine map, so rigid transforms,
reflections, nonuniform scales, shears and singular affine projections share one
exact parametric contract. Open Plines ignore the final vertex's bulge; closed
Plines use it for the closing segment. The implemented tolerance and invalid-box
boundaries are owned by the
[specification](structure_spec.md#curved-geometry-bounds-contract).

The implementation avoids sampling and the former average-radius approximation.
It also rejects complex, perspective and nonfinite transforms consistently with
the project transform boundary. Deep review found no ordinary-range underbound
in the sweep or bulge derivation. Its findings led to strict homogeneous-row
validation, overflow-safe full-sweep norms and half-chord bulge construction,
and exact partial-affine and singular-transform assertions. A second review found
no remaining blocker; 1,000 randomly sampled transformed bulge arcs produced no
underbounds.

Verification on Python 3.10.9:

- Pre-change baseline: `python -m unittest discover -s tests -v` passed 65 tests.
- `python -m unittest discover -s tests -p test_curved_bounds.py -v` passed all
  seven focused tests.
- `python -m unittest discover -s tests -v` passed all 72 tests.
- `python -m compileall -q cambam_builder legacy_cambam_builder tests demos`, the
  import/construct smoke check and `git diff --check` passed.

No XML fields or stored geometry changed, so a CamBam display round trip would
not test this runtime calculation and is not required. Curved geometry baking,
the approximate curved shape returned by `get_absolute_coordinates()`, Region
support and copy/transfer APIs remain outside this result. Item 1b is ready as a
fresh contract-first increment; reopen 1a for a demonstrated numerical underbound
or a new non-affine geometry policy. Suggested commit:
`fix: calculate exact curved geometry bounds`.

## Copy and transfer contract decision

2026-09-08: backlog 1b began with no modern copy/transfer API. Section 5's
conceptual UUID-overwrite proposal did not define ownership of existing target
children, group selections or operation order. The project registration helper
also permits UUID overwrite while separately rejecting identifier collisions;
using it incrementally would not establish atomic transfer.

Chosen contract: complete descendant subtree, detached root preserving world
pose, cloned containers, UUID preservation or explicit full remapping, and
rejection of destination collisions. Associated MOPs must have targets wholly
inside the subtree; callers may explicitly omit MOPs. Group names cannot merge
implicitly because that would expand existing live MOP targets. Stage container
and relationship changes before publication, preserving unrelated object
identities. The [current contract](structure_spec.md#copy-and-transfer-contract)
owns API details and limitations.

Overwrite/merge and automatic expansion to unrelated MOP targets were rejected
for this increment: both need additional destination-ownership decisions and
could change unrelated geometry or machining selections. Reopen for a concrete
synchronization or assembly workflow with explicit merge acceptance criteria.
Destination global machining/style context stays authoritative; parameter
preservation does not establish identical inherited settings or toolpaths.

Lead owns the shared contract and integration; native bounded workers handle
retrieval, implementation and independent tests. External authorization and cost
telemetry were unavailable. No budget saving is asserted from measured usage.
Automated evidence is recorded below. Manual
CamBam checking adds no distinct evidence for registry cloning once numerical
and serialized relationship regressions pass. Existing display acceptance is
not extended to production machining by this work.

Integration exposed a pre-existing Rect XML defect: the conversion branch used
`to_pline_representation()`, which bakes only the local matrix into an unlinked
Pline. A transformed parent was therefore lost on export. The new two-round-trip
test first failed its matrix assertion, then its world-outline assertion;
checking geometry confirmed this was a real defect, not merely an alternate
representation. The bounded repair applies the complete world matrix to Rect
corners and emits a temporary Pline with an identity XML matrix, retaining the
existing world-baked representation. Keeping corners local with a world matrix
was also geometrically valid but broke the existing root-Rect identity-matrix
control; retaining that representation avoids an unnecessary compatibility change.
It does not change the public conversion helper or bake behavior. Required
evidence includes the complete outline, parent links and descendant world poses
through both round trips. The existing Pline XML representation needs no new
schema/display acceptance; no production machining acceptance is inferred.

### Copy and transfer verification

2026-09-08: the implemented `copy_primitive_tree` and
`transfer_primitive_tree` APIs pass the focused copy/transfer suite (16 tests)
and the full suite (88 tests) on global Python 3.10.9 with NumPy 1.23.5; no
project virtual environment is present. Coverage includes atomic collision and
late-clone failures, subtree relationships, UUID/name/group remapping, source
preservation, transfer cleanup, MOP closure, world pose, and two XML round trips
covering outlines, root/leaf poses, primitive/MOP UUIDs, parent/layer assignments,
MOP parameters and targets.
The retained logs are [focused.log](../output/copy-transfer-checks-final-7c34ab/focused.log)
and [full.log](../output/copy-transfer-checks-final-7c34ab/full.log).

The bounded `Rect.to_xml_element` repair applies the complete ancestor world
transform when emitting the world-baked Pline representation, preserving its
identity XML matrix and outline through the transfer round trips. Compileall and
the import/construct smoke check also pass; [compile.log](../output/copy-transfer-checks-final-7c34ab/compile.log)
and [import.log](../output/copy-transfer-checks-final-7c34ab/import.log) are retained.
`git diff --check` exits 0 with only line-ending normalization warnings. No manual
CamBam validation adds evidence for these registry and known XML-encoding checks;
production toolpaths remain outside scope. Packaging is unverified.

This is a good fresh-session breakpoint with implementation, evidence and limits
persisted. The next priority is Region/all-shape Z parity under
[`SHAPE_PARITY_PLAN.md`](SHAPE_PARITY_PLAN.md). Suggested commit:
`feat: add atomic primitive tree copy and transfer`.

## Region and shape-elevation implementation

2026-09-09. Backlog item 2 adds geometry elevation and Region interchange to the modern
package. The user reaffirmed that the API is unreleased and may change. The
[specification](structure_spec.md#shape-elevation-and-region-contract) owns the
implemented API, topology, matrix and query/bake boundaries; the
[runbook](DEVELOPMENT.md#manual-shape-parity-acceptance) owns prepared acceptance.

The chosen representation keeps intrinsic XY/bulge data, with separate explicit
per-vertex Z or analytic elevation and independent local Z offsets. This avoids
reinterpreting the third Pline tuple element or replacing all affine/relationship
machinery with a general 3D framework. XYZ queries are explicit; reflected arc
sweeps and bulges are corrected, while representations that cannot express an
ellipse reject a nonsimilarity query. XML can retain supported XY affine matrices.
Spatial matrices, varying-Z bulged segments and lossy analytic/Text bakes were
rejected pending native evidence instead of flattened or approximated. The later
varying-Z interchange evidence is recorded below. Existing 2D constructor/query
usage remains valid except that invalid affine constructor matrices now raise.

Region is one registered primitive with owned contours and a single MOP identity.
It supports the observed typed `entity xsi:type="Region"` schema, `OuterCurve/pts`
and `HoleCurves/Polyline`, including contour poses and elevation. A copied
registered contour captures its complete world pose before detachment. The
validator handles lines/arcs analytically, either winding, two complementary
semicircles, holes and nondegenerate planar topology. Direct shifts/bakes and
project full/global subtree bakes stage changes before publication. Review added
coverage for invalid topology, coincident arcs, near-identity bakes, reflected
queries, finite Z, contour ownership and export revalidation.

### Native Region fixture finding

Local import of the private `output/region_example.cb` exercises the expected
schema but fails strict topology validation: the first hole self-intersects
between zero-based contour segments 2/4 and 3/4. A separate local script sampled
each circular segment into 2,048 chords and independently found an interior
crossing in each pair. This diagnostic does not call the Region intersection
validator. Hash comparison before/after the check confirms no source change.
No GLM/OpenRouter processing was used; delegated edit ownership was restricted
to source code and authored synthetic tests.

The declared model rejects self-intersection rather than silently repairing the
hole, filling it or importing an empty/partial project. Therefore the native
sample itself is **not accepted**. Supported typed XML is covered with authored
valid synthetic cases. Reopen this boundary only for an explicit requirement to
retain invalid contours for interchange, or evidence that CamBam interprets the
same supported bulge data differently. Such a requirement needs a separate
invalid-topology representation, not bypassing validation in the current Region.

### Verification and acceptance state

Environment: existing Python 3.10.9 / NumPy 1.23.5; no project `.venv`, dependency
installation, legacy-package changes or CamBam runtime dependency. Checks run from
the repository root:

- `python -m unittest discover -s tests -v`: 135 tests passed after final review. The new tests cover all seven types, two XML round trips, actual
  XML geometry/matrix Z, parent/world poses, bulges, bounds, holes, identities,
  groups/MOP targeting, full/nonrecursive/Z bakes, pickle and detached copy/transfer.
- `python -m compileall -q cambam_builder legacy_cambam_builder tests demos`:
  exit 0; `CBProject('smoke')` import/construction assertion passed.
- `python output/shape-parity-kvr0bdc2/generate.py`: passed; A/B retain eight
  primitives across all seven types, XYZ, identities, two Region holes and one
  Region pocket target through repeated round trips.
- `python output/shape-parity-kvr0bdc2/verify_acceptance_xml.py`: passed; an
  independent hardcoded expected-value inspection checks every runbook coordinate,
  dimension, bulge, hole, identity matrix and exact Region MOP reference in B.
- `python output/shape-parity-kvr0bdc2/verify_native_roundtrip.py`: passed; native
  C and framework D retain all eight identities, XYZ geometry, hierarchy, two
  Region holes, the Region-only pocket target and Text content/optional `p2`.
- `git diff --check`: passed after final review.

The final independent topology review found three concrete issues, all repaired
and reverified with the original synthetic reproducers and 21 focused Region
tests. Arc ray containment now uses exact stored endpoints, preventing a wholly
outside hole at an arc endpoint height from passing validation. XML export
validates the rounded geometry, rejecting a hole gap erased by output precision;
increasing precision to preserve the gap permits the round trip. Matrix
conditioning uses normalized linear entries, accepting a well-conditioned
`1e-8` scale that yields a useful 10-by-10 contour. These checks complete the
review scope; unrestricted numerical geometry and arbitrary 3D remain excluded.

The first integration run exposed test assumptions about Rect XYZ triples versus
converted Pline XYZ/bulge tuples, and XML precision amplification in computed arc
angles. Tests now compare geometry projections across representation changes and
explicitly choose 12-decimal XML precision for repeated transformed fixtures;
they retain `1e-10` in-memory / `1e-8` XML tolerances. XML tests compare numeric
values, allowing equivalent lexical forms such as `1` and `1.0`. The acceptance
A/B fixture uses 10 decimal places and exact translation matrices.

The user accepted every specified A/B display and property check in CamBam Plus
1.0, including matching geometry, all declared coordinates, Text at its `p1`
anchor, Region holes and the Region-only MOP target. CamBam saved B as native C
while retaining the framework's optional `p2="35,35,7"`. A Text created in a
fresh CamBam session uses `align="bottom,left"` and only `p1`; alignment therefore
does not require `p2`. This agrees with the
[official CamBam MText API](https://www.cambam.info/doc/api/MText.htm), which
documents `P1` as the current alignment point and `P2` as currently unused.

Further native inspection showed that CamBam suppresses both points for Text at
the default origin and may materialize equal `p1`/`p2` after it is moved. The
reader now defaults absent `p1` to `(0,0,0)`, and the writer suppresses the same
default. Together with the earlier non-origin `p1`-only file, this establishes
that optional `p2` reflects native serialization history rather than alignment.

Native C placed Text content after its `mat` child. That exposed a reader defect
that only read leading element text; the reader now retains one non-whitespace
direct mixed-content chunk and rejects ambiguous multiple chunks. Focused native
child-order and absent-`p2` regressions pass. The C-to-D check verifies the repair
without modifying C. This acceptance establishes the declared display/property
interchange case, not generated production toolpaths. Item 2 is complete and the
next priority is packaging and supported Python-version validation. This is a
good fresh-session breakpoint because the next scope does not depend on unresolved
item-2 decisions.

Suggested commit: `feat: add Region and all-shape elevation parity`.

## Canonical vertex-record refactor

2026-09-09. The bounded follow-up replaces Pline and Points parallel coordinate,
bulge and elevation storage with one public `Vertex(x, y, z=0, *, bulge=0)`
record and a sole `vertices` collection. Region-owned Pline contours use the same
records. Constructors and project adders normalize only named records, `(x, y)`
and `(x, y, z)` tuples. Thus three-tuples now unambiguously mean XYZ, four-tuples
are rejected, nonzero bulge requires a named record, and Points rejects nonzero
bulge at API and XML boundaries. The API is unreleased and the user authorized
these breaking changes; no compatibility property or old-pickle migration was
added.

Geometry consumers, bounds, reflected/similarity bakes, intrinsic Z shifts,
Rect-to-Pline conversion, Region topology/ownership/bakes, XML read/write and
current persistence were migrated. Existing XY and XYZ geometry-query return
shapes remain unchanged. Mutable collection validation requires records after
construction, so insertion and reordering carry X/Y/Z/bulge together instead of
depending on an index-aligned elevation list. Repository callers and authored
tests preserve every former bulge explicitly with `Vertex(..., bulge=value)`.
Analytic elevation fields, Text `xml_p2`, MOP ownership and Region topology rules
are unchanged.

Verification on Python 3.10.9 / NumPy 1.23.5:

- `python -m unittest discover -s tests -p test_vertex_records.py -v`: six tests
  passed, covering constructor defaults, keyword-only bulge, tuple meanings,
  malformed/four-value rejection, Points rejection, collection edits, adders,
  current pickle and two XML round trips.
- `python -m unittest discover -s tests -q`: all 141 tests passed. Existing
  Region holes/topology, curved bounds, transforms/bakes, identities,
  relationships, MOP references, copy/transfer and persistence remain covered.
- `python -c "import runpy; runpy.run_path('output/vertex-record-parity-20260909-a/compare.py', run_name='__main__')"`:
  passed. It read the accepted pre-refactor `B_baked_elevations.cb`, wrote two
  results only under the new ignored task directory, and compared all serialized
  Pline, Points and Region point/bulge/matrix numbers at absolute tolerance
  `1e-8`. Maximum observed difference was `0.0`. SHA-256 checks before/after
  confirmed accepted A/B/C inputs were unchanged.

No CamBam acceptance was repeated because serialized geometry and query/output
semantics are unchanged. At completion, remaining limits included the then-declared
rejection of varying-Z bulged segments, general spatial matrices and invalid Region
topology, plus the explicit absence of old-pickle compatibility. The varying-Z
restriction was subsequently superseded by native evidence and the increment
below. This is a good
fresh-session breakpoint once final compile/import/diff checks pass; packaging
and supported-Python validation remain the separate next priority.

Suggested commit: `refactor: consolidate pline and points vertices`.

## Varying-Z bulged Pline and Region interchange

2026-09-10. The user supplied CamBam-generated XML demonstrating both a Pline and
a Region whose bulged segments have unequal endpoint Z values. The Region includes
mixed-Z outer and hole contours. The user additionally verified that CamBam renders
the sloping curves, while Pocket and Profile operations use their machining depth
parameters and ignore the Region contour elevations. This is direct evidence for
native storage/interchange and projected Region topology, but not for an exact
intermediate Z parameterization or this framework's future MOP calculations.

The owning Pline validation now accepts every finite Z/bulge combination. Region
validation no longer requires coplanar contours and continues to perform simplicity,
intersection, containment, nesting, degeneracy and bounds work entirely in XY.
Reader/writer, copy, Z shift, endpoint queries and similarity/reflection bakes
already operated on canonical Vertex records, so no schema or output-contract
change was needed. Non-similarity transforms of bulged geometry remain rejected
because the XY circular projection becomes elliptical. Points still reject bulge,
and tilted matrices and intermediate spatial-curve evaluation remain unsupported.

Verification on Python 3.10.9 / NumPy 1.23.5:

- `python -m unittest discover -s tests -p test_shape_elevation.py -v`: 11 passed.
- `python -m unittest discover -s tests -p test_vertex_records.py -v`: 6 passed;
  its two project XML cycles now include unequal-Z endpoints on a bulged segment.
- `python -m unittest discover -s tests -p test_region.py -v`: 22 passed; the
  native-derived mixed-Z/bulge outer and hole survive two XML cycles within the
  default ten-decimal serialization tolerance (`1e-9` asserted).
- `python -m unittest discover -s tests -p test_shape_parity.py -v`: 10 passed;
  a mixed-Z/bulge Pline and Region retain world geometry, identity and the Region's
  Profile MOP reference across two complete project XML round trips.
- `python -m unittest discover -s tests -q`: all 142 tests passed, including
  relationships, Region holes, MOP references, transforms, bounds and persistence.

No additional CamBam acceptance is required: the new supported forms are derived
from XML the user created and exercised in CamBam. Production toolpath generation
and exact interpolation between varying-Z arc endpoints remain separate work.

Suggested commit: `feat: support varying-z bulges in plines and regions`.

# Packaging and supported Python verification

Date: 2026-09-10.

The former Python >=3.8 declaration was not credible because the distributed
legacy implementation evaluates built-in generic annotations such as
`list[tuple]`; the validation host also had no Python 3.8 interpreter. The
user authorized Python 3.9 as the minimum, so published metadata now matches the
lowest interpreter actually exercised rather than claiming an untested repair.
NumPy moved from setuptools' dynamic `setup.py` adapter into standard static
project metadata, allowing pip, build frontends and `uv` to consume one contract.
The redundant `setup.py` and `requirements.txt` files were removed. The lower
bound is NumPy 1.23.5, the oldest version with recorded full-suite evidence in
this repository; compatible newer releases remain independently resolvable.
The generated `uv.lock` is intentionally local and ignored at the user's request;
the tradeoff is that future environment resolution is current rather than exactly
reproducible, which matches the requirement to exercise unpinned NumPy separately.

`uv build` produced an sdist and a `py3-none-any` wheel in the ignored
`output/packaging-validation-20260910-c/` directory. Archive inspection confirmed
both package roots, all active modern modules, the MIT license and metadata for
version 0.1.0, Python >=3.9 and NumPy. Separate clean wheel environments used
Python/NumPy 3.9.0/2.0.2, 3.10.9/2.2.6, 3.11.0/2.4.6, 3.12.10/2.5.3 and
3.13.9/2.5.3. Each asserted that the imported modern package lived outside the
repository, imported both package roots, checked installed metadata, completed a
representative construction and XML write/read, and passed all 142 tests. A
separate sdist install on Python 3.9/NumPy 2.0.2 passed the same installed-path
and import checks plus all 142 tests.

After removing the compatibility `setup.py` and duplicate `requirements.txt`, a
fresh sdist and wheel were rebuilt. Neither removed file appears in the sdist;
the wheel imports both package roots on Python 3.9-3.13, and the setup-free sdist
installs on Python 3.9 and passes all 142 tests from outside the repository.

No CLI entry point or publishing flow was added. The shipped historical CLI
module is not imported by the legacy package root and remains outside the
supported surface. Reopen Python 3.8 only if a concrete consumer requires it and
the entire distribution plus dependency resolution can be tested there; reopen
CLI/distribution work only for an actual user workflow.

# MCP protocol and adapter contract - 2026-09-10

Backlog 4a is a contract/probe outcome, not an implemented adapter. The lasting
decisions and first-slice acceptance are in [MCP_CONTRACT.md](MCP_CONTRACT.md),
with [machine-readable input/output schemas](mcp_contract_v1.schema.json).
No runtime source, base dependency, installed client configuration or launcher
was changed. Disposable scripts, environments, public-source snapshots and
synthetic artifacts are under `output/mcp-contract-haugv1gu/`.

Primary evidence checked on 2026-09-10:

- The [normative core](https://modelcontextprotocol.io/specification/2026-07-28/basic/index)
  requires per-request protocol version and client capabilities; client identity
  is optional. The [stdio binding](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
  retains local subprocess transport without an initialization requirement.
- The [SDK v2 migration guide](https://py.sdk.modelcontextprotocol.io/migration/)
  describes `MCPServer`, split `mcp_types` and modern support. Actual PyPI metadata
  selected `mcp==2.2.0`, `mcp-types==2.2.0`, Python >=3.10. An isolated environment
  used Python 3.13.9, Pydantic 2.13.5 and jsonschema 4.26.0. No SDK was installed
  into the project `.venv`.
- [OpenCode MCP source at 1.18.29](https://github.com/anomalyco/opencode/blob/v1.18.29/packages/opencode/src/mcp/index.ts)
  has stdio, Streamable HTTP and legacy SSE transports. Its
  [package declaration](https://github.com/anomalyco/opencode/blob/v1.18.29/packages/opencode/package.json)
  pins TS SDK 1.29.0. The installed backend binary reports 1.18.29.

The Python SDK subprocess successfully answered `tools/call` as the first request,
then `tools/list`, then `server/discover`, with `2026-07-28` metadata on each.
The synthetic typed `echo(value: int) -> EchoResult` returned structured
`{"value":7}` and `resultType="complete"`; closing stdin exited 0. A bare `dict`
return annotation did not produce `structuredContent`, so adapter handlers must use
explicit typed result models/output schemas. The contract requires a middleware
version gate because SDK v2 also supports legacy clients; the probe did not claim
that the SDK's default server rejects legacy initialization.

The installed OpenCode backend was run through `mcp list --pure` with separate
XDG config/data/state/cache directories, project config/default plugins/model fetch
and auto-update disabled, and only one synthetic local server. It sent:

```json
{"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{"roots":{}},"clientInfo":{"name":"opencode","version":"1.18.29"}},"jsonrpc":"2.0","id":0}
```

A mock returning method-not-found to initialization made OpenCode report `failed`.
A control mock answering initialization at `2025-11-25`, advertising tools and
returning an empty catalog made it report `connected`, followed by
`notifications/initialized` and `tools/list`. Both CLI commands exited **0**;
the wire transcript and displayed status, not exit status, establish the result.
The modern mock is a negative compatibility detector, not a complete conforming
MCP server. This proves the installed backend's handshake behavior; it does not
prove the version or behavior of an installed Desktop GUI. Modern Desktop
compatibility remains unaccepted and gates 4e. No model was invoked or user CAD
data sent to a service. Reopen with a client release that claims modern support,
then require modern wire evidence rather than mere successful connection.

Commands actually run from the repository root:

```powershell
uv venv --python .venv/Scripts/python.exe output/mcp-contract-haugv1gu/sdk-env
uv pip install --python output/mcp-contract-haugv1gu/sdk-env/Scripts/python.exe mcp==2.2.0
.venv/Scripts/python.exe output/mcp-contract-haugv1gu/client_probe.py
output/mcp-contract-haugv1gu/sdk-env/Scripts/python.exe output/mcp-contract-haugv1gu/sdk_probe.py
output/mcp-contract-haugv1gu/sdk-env/Scripts/python.exe output/mcp-contract-haugv1gu/validate_schemas.py
.venv/Scripts/python.exe output/mcp-contract-haugv1gu/framework_probe.py
git diff --check
```

The initial default-sandbox external fetch and uv cache access failed; authorized
escalated commands completed the disposable environment/probes. PowerShell's
`opencode.ps1` shim was blocked by execution policy, so the installed `.exe` was
used without changing policy. One probe print initially hit cp1252 console encoding;
UTF-8 output corrected it. These were harness/environment failures, not protocol
support. Final schema synchronization was interrupted when automatic approval
review hit the session usage limit. After the user resumed the task, the approved
verification completed; no approval rejection remains unresolved.

Reproduction in a fresh checkout does not require the ignored scripts:

1. Create a unique `output/mcp-contract-<suffix>/` environment with the two uv
   commands above, substituting that directory. Run a Python subprocess server
   using `from mcp.server import MCPServer`, a Pydantic model `EchoResult` with
   `value: int`, and an `echo` tool returning `EchoResult(value=value)`. Call
   `server.run()` for stdio. A parent process writes each of the following JSON-RPC
   requests as one line, reads and checks the matching response before continuing,
   then closes stdin and waits for exit:

   ```text
   params._meta = {"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                   "io.modelcontextprotocol/clientCapabilities":{},
                   "io.modelcontextprotocol/clientInfo":{"name":"cambam-probe","version":"0"}}
   id 1: tools/call, params.name="echo", params.arguments={"value":7}
   id 2: tools/list
   id 3: server/discover
   ```

2. For OpenCode, configure `mcp.probe` as `type="local"`, a command array for
   environment Python plus a synthetic mock script, and `timeout=3000`. Use a
   copied subprocess environment with `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
   `XDG_STATE_HOME`, `XDG_CACHE_HOME` pointing inside the task directory;
   `OPENCODE_CONFIG_DIR` points to its config directory and
   `OPENCODE_CONFIG_CONTENT` contains only that MCP configuration. Set
   `OPENCODE_DISABLE_PROJECT_CONFIG`, `OPENCODE_DISABLE_DEFAULT_PLUGINS`,
   `OPENCODE_DISABLE_MODELS_FETCH`, `OPENCODE_DISABLE_AUTOUPDATE` to `true`.
   Run the installed executable's `mcp list --pure` with that directory as cwd
   and a 40-second process timeout. The mock records each inbound JSON line;
   for any request with `id`, echo the ID and return JSON-RPC error `-32601`.
   Repeat with a control that returns initialization result
   `{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"synthetic-probe","version":"0"}}`
   and `{"tools":[]}` to `tools/list`; ignore notifications. Expected transcripts
   and interpretation are above. Do not read real client credentials/config.
3. Validate the tracked schema with `jsonschema.Draft202012Validator.check_schema`.
   To validate a tool, construct `{"$ref":"#/$defs/<tool>_input","$defs":...}`
   (or `_output`) using its definitions. Exercise every tool's example inputs,
   missing required fields, unknown properties, zero width, boolean width, string
   translation, oversized page, absolute/wrong-extension paths, duplicate targets,
   and every success/error envelope. Schema validation alone does not prove
   canonical filesystem containment or state/retry behavior.
4. Author the exact direct-framework A/B slice in the contract's acceptance section
   using `CBProject`, `add_rect`, `add_part`, `add_profile_mop`, `save`,
   `read_cambam_file`, `translate_primitive`, then `save`/reload. This independently
   confirms that 4c's target values are supported by the current framework.

Results: schema meta-validation plus **98 assertions** across eight tool inputs
and success/error outputs passed. After contract review, **7 additional assertions**
passed for the workspace bootstrap schema and terminal cancellation error, for
**105 assertions total**. The direct A/B workflow retained primitive/MOP
UUIDs, targets, six explicit machining values and the original A bytes. B corners
matched (5,2,0), (25,2,0), (25,12,0), (5,12,0), absolute tolerance `1e-9`.
This is a direct API reference probe, not MCP parity implementation.

The API review also reproduced that `copy.deepcopy(project)` leaves primitive
project links detached: a cloned child loses its translated parent's world pose.
`Primitive.__getstate__/__setstate__` deliberately remove the link; only pickle
loading currently repairs it. 4b needs public `CamBamProject.clone()` that rebinds
cloned primitives, plus independent-mutation/parent-transform/MOP-template tests.
The reader currently skips unknown MOP types; 4b needs the strict byte-snapshot
reader contract rather than duplicate XML policy inside tool handlers. These
prerequisites are intentionally small owning-framework changes, not adapter access
to private registries or a broader geometry refactor.

Independent contract review identified bootstrap/error-envelope ambiguity,
concurrent document-capacity reservation, canceled ledger entries, exhaustion
recovery and saved-artifact revision ambiguity. The final contract specifies the
discovery metadata key and stderr grammar, canonical root hash, configured workspace
ID in error envelopes, a registry reservation lock, terminal `REQUEST_CANCELLED`,
save/close access after regular ledger exhaustion, and a document lock held through
save publication and ledger completion. The schemas include the bootstrap record
and cancellation error. These were reviewed design rules at the 4a boundary;
the 4b implementation and verification are recorded in the next section. Final
local documentation link/anchor and schema-reference checks passed, as did
`git diff --check`.

No manual validation was needed for 4a. Runtime/security implementation and full
framework regression checks are now recorded under 4b; actual desktop/second-PC
use and CamBam production acceptance remain separate future evidence. Volatile handles
deliberately lose unsaved work at restart; `.cb` interchange still has limited
unknown-field preservation and no verified persisted unit setting. These limits,
new-file-only save policy and reopening criteria are in the contract.

## MCP foundation and client compatibility - 2026-09-10

The client survey distinguishes transport support, advertised protocol support,
observed wire behavior and user acceptance. The earlier modern-only 4a decision
was explicitly expanded by the user to require backward protocol compatibility.
2025-06-18 and 2025-11-25 are the initial compatibility targets because they match
the observed Codex and OpenCode clients. The modern protocol remains supported;
legacy success does not count as 2026-07-28 conformance.

| Client | Evidence | Scope of conclusion |
| --- | --- | --- |
| Codex CLI 0.154.0, Windows | `codex features list` exposes `mcp_2026_07_28`, under development and false by default. Both `codex exec` with the flag and CLI app-server with explicit runtime enablement send `initialize` at 2025-06-18. Modern-only adapter returns -32601. Legacy control accepts handshake and receives `tools/list`. | Actual negative modern-stdio test, not an inference from generic MCP documentation. The [official MCP guide](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) confirms stdio support but does not promise this modern protocol version. |
| OpenCode 1.18.29 | 4a captured 2025-11-25 initialization and successful legacy control. | Actual backend probe; Desktop GUI not tested. |
| Claude Code | [Official runtime documentation](https://code.claude.com/docs/en/mcp#mcp-client-runtimes) states v2.1.232+ has SDK v2 support, subject to provider/feature conditions; stdio modern negotiation needs `MCP_PROTOCOL_NEGOTIATION=auto`, and runtime can be selected with `MCP_SDK_GENERATION=v2`. | Documented conditional support, no local test. |
| Gemini CLI | Retrieved [core dependency declaration](https://raw.githubusercontent.com/google-gemini/gemini-cli/main/packages/core/package.json) pins `@modelcontextprotocol/sdk` 1.23.0. | Evidence suggesting a legacy client path in retrieved source; no installed-version or modern wire acceptance. |
| Cursor | [Official MCP documentation](https://cursor.com/docs/mcp) lists stdio/HTTP/SSE and features but no explicit 2026-07-28 claim. | Modern compatibility unverified; lack of a claim is not proof of incompatibility. |

This is a small targeted survey, not a statistical claim about all clients.
Compatibility friction is sufficient evidence to require legacy acceptance rather
than block delivery on modern client adoption. Recheck exact installed versions
and wire traffic when expanding the client matrix.

Disposable evidence is retained in `output/mcp-foundation-20260910/`:
`modern-wire.jsonl`/`modern-codex.stderr`, `runtime-wire.jsonl`/
`runtime-host.jsonl`, and `control-wire.jsonl`. The runtime feature-enable response
confirms the flag was accepted before the negative probe; the legacy control
removes a launch/stdio failure as the explanation. Only synthetic task data was
used; no global MCP configuration was edited. The CLI app-server probe calls the
installed client directly without an LLM request. Early harness issues (buffered
stdio read, SDK snake_case result access) were corrected before acceptance runs.

The foundation introduces the optional adapter package, independent public
project clone and strict bounded UTF-8 XML snapshot reader. Document operations
use explicit workspace/boot handles, locks, capacity reservations, revision checks
and a process-lifetime terminal-result ledger. Saving stages a clone, checks and
flushes bytes, and publishes with no-replace hard-link creation. Failed staging or
destination races preserve existing bytes; successful publication survives retry
and post-publication cancellation. Typed CAD detail and authoring tools remain
4c scope. Framework regressions are separate from protocol/client acceptance.

After adding SDK-backed legacy support, `codex exec` completed create, inspect,
save, open and reopened inspection. Its default noninteractive `never` approval
policy refused close; this was a client approval configuration issue, not a server
error. A second synthetic workflow through `codex app-server` used a process-local
`tools.document_close.approval_mode="auto"` override and completed all seven calls,
including both closes. The lead independently checked every application result,
both closed results and the saved file's SHA-256 against the returned artifact.
`compat-host-wire.jsonl` records actual 2025-06-18 negotiation;
`compat-host-host.jsonl` records client results. `compat-codex.jsonl` retains the
first agentic workflow and its approval limitation. The saved empty document is
`client-compat-host/Codex.cb`, 410 bytes, SHA-256
`36e3244907051031ec52e5acdd2d411eb9b0d5af0ace9488144b7ab02adfe688`.
This selects **Codex 0.154.0 on the legacy compatibility path** for subsequent
work. Desktop GUI, second-PC and authoring/CAM acceptance remain separate.

The final full run completed 175 tests in 15.676 seconds: 174 passed and one
symlink-creation case skipped because this Windows account lacks that privilege.
The separate junction/reparse-point rejection test passed. `compileall` and
`git diff --check` also pass. A fresh wheel and sdist in
`output/mcp-foundation-final-20260910-3/dist/` contain the packaged contract.
Clean wheel installs on Python 3.10.9 and 3.13.9 load MCP 2.2.0 and the five
document schemas from site-packages. A clean Python 3.9.0 base install imports
and clones projects without installing MCP, while `cambam-mcp` exits with the
documented Python-version message. The installed Python 3.13 executable completed
discovery/listing at 2026-07-28 and initialize/listing at both legacy versions.
Independent final review found that an initial metadata-free direct request could
otherwise make the SDK select its legacy loop implicitly. The stdio reader now
rejects that request with `INVALID_PARAMS` before era selection; only an explicit
legacy `initialize` can select legacy. The subprocess regression and installed-wheel
probe confirm rejection followed by a valid modern discovery on the same process.

## MCP first authoring and round-trip slice - 2026-09-10

Backlog 4c implements the three reserved authoring tools and advertises the complete
eight-tool version 1 catalog. Rectangle, Profile and translation mutations clone
the project under the document lock, validate all slice-specific preconditions and
publish the staged project with one revision increment. Identifier and target
resolution uses public framework getters; absent layers/parts are created through
the public adders, with the contract's zero-stock Part values supplied explicitly.
Root Rect inspection uses public world-coordinate and bounding-box queries. Profile
inspection returns the closed explicit parameter record only when type, target,
parameter values and imported inheritance state remain inside 4c; other details
stay identity/relationship-only with `INSPECTION_UNSUPPORTED`.

The focused regression in `tests/test_mcp_authoring.py` runs the exact contract
workflow and an independently authored `CBProject` reference. It checks revision
0/1/2 and reopened 0/1 transitions; primitive/MOP UUIDs, target and ordering
preservation; all exposed Profile values and fixed settings; cyclic world corners
within `1e-9`; semantic XML equality after normalizing UUIDs and equivalent numeric
spellings; translation retry; concurrent same-revision serialization; and unchanged
A bytes. Negative cases cover malformed arguments, stale revisions, missing,
duplicate, wrong-kind and transformed targets, cross-kind identifier conflicts,
out-of-slice inspection and complete inspection preservation after failed edits.
The regression exposed that schema reference expansion had discarded sibling
defaults such as rectangle `z`; expansion now merges those keywords and the tests
exercise omitted optional defaults.

Independent runtime review found two pre-closure inspection gaps: fixed public
Profile fields omitted from the returned record were not yet part of slice
classification, and native parent-container `Default` states could look explicit.
The final classifier checks every fixed public field and requires the expected XML
leaf/ancestor path to be present, never `Default`, and explicitly `Value` at some
level. Regressions mutate `optimisation_mode` and `HoldingTabs` inheritance to keep
both cases identity-only with `INSPECTION_UNSUPPORTED`. The same review found no
other high/medium issue in atomicity, cancellation, revisions, ledger behavior,
error mapping or schema paths.

Verification on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_authoring.py -v`:
  5 passed.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  27 passed and one Windows symlink-privilege test skipped; junction/reparse
  rejection passes.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 179 passed and
  the same one skip.
- `.venv/Scripts/python.exe -m compileall -q cambam_builder legacy_cambam_builder tests demos`,
  contract meta-validation/tool-catalog smoke and `git diff --check`: pass.
- `.venv/Scripts/python.exe demos/mcp_authoring_slice.py`: passed and wrote
  inspected A/B files to
  `output/mcp-authoring-demo-bba34075e1d0/`. A is 3,821 bytes with SHA-256
  `3b04f2abe882de5d7105c9729e20a3025d5f531d1893042ea3470d0c71639896`; B is
  3,826 bytes with SHA-256
  `ca08657e667b508cb6d0043245c0581dbc1771ca2250b6794f4490bf55ee78c6`.
  The script verified B corners `(5,2,0)`, `(25,2,0)`, `(25,12,0)`, `(5,12,0)`,
  exact revision milestones, retry replay, returned hashes and unchanged A bytes.

The standard managed sandbox cannot reopen Python-created OS temporary directories;
the focused and full temporary-workspace tests therefore used the approved local
filesystem path, matching the existing MCP runbook. No manual check adds evidence
to this framework/adapter parity slice. Named desktop/second-PC connection and
CamBam units, property and toolpath acceptance remain backlog 4e. Broader entity,
relationship and operation coverage remains 4d and must not be inferred from this
Rect/Profile proof.

## MCP geometry breadth batch 1 - 2026-09-10

Backlog 4d's reopening criterion asked for one concrete high-value family from
the contract's documented mappings. The curve/point geometry family
(Circle/Arc/Pline/Points adders with their public world queries) was selected:
Rect-only authoring blocked most drawing work, and later Drill/Pocket MOP
breadth needs Points/Circle targets to be expressible at all. Text, Region,
Pocket/Engrave/Drill, parenting/groups/copy-transfer and bake transforms stay
explicitly unsupported with reopening criteria in
[PROGRESS.md](PROGRESS.md#active-work-and-next-priority).

Implemented contract surface (schema `docs/mcp_contract_v1.schema.json`, byte
identical packaged copy, contract sections updated in the same increment):

- Four new tools map only to public framework adders:
  `geometry_add_circle` (center/diameter/elevation), `geometry_add_arc`
  (center/radius/CCW-positive degree sweep/elevation), `geometry_add_pline`
  (2..10000 vertex records with per-vertex Z and bulge, optional `closed`) and
  `geometry_add_points` (1..10000 plain XYZ points; bulge input is rejected by
  the schema). All create root primitives on a possibly new default layer and
  return the primitive UUID plus layer name.
- `geometry_translate` accepts root Rect/Circle/Arc/Pline/Points primitives in
  the translation-only slice (identity or pure-translation world matrix, zero
  local Z offset, no parent/children/groups, valid positive geometry) and still
  never bakes. Profile targets deliberately remain root Rects only.
- Inspection's primitive record now carries one closed typed `geometry` payload
  instead of top-level `world_xyz`/`bounds`: `rect` corners, `circle`
  center/diameter, `arc` center/radius/degree angles, `pline` vertices with a
  parallel `bulges` array plus `closed`, and `points` vertices, each with
  directed analytic world bounds. The bulge convention is documented: the bulge
  stored on vertex i curves the segment that starts at vertex i. Unsupported
  primitives (Text, Region, transformed or related shapes) keep
  `geometry: null` with `INSPECTION_UNSUPPORTED`; no geometry is invented.
- Retry identity canonicalizes nested defaults, so omitting `z`/`bulge` and
  sending explicit zeros replay as the same request instead of conflicting.
- Verification notes: the Arc world `start_angle` comes from the public
  direction-based query and may differ from the authored angle by float noise
  (30 degrees reports 29.999999999999996 before the 9-decimal XML round trip),
  and primitive document order is the framework's UUID-sorted
  `list_primitives()` order, not insertion order. Parity fixtures therefore
  compare inspection records with a tolerance-aware deep comparison and XML
  with order/id-canonicalized semantics; the saved artifacts themselves are
  bit-compared only within one run (A bytes unchanged across reopen).

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_geometry.py -v`:
  5 passed (analytic guard, round-trip parity, translate parity/replay,
  negative/conflict/atomicity, imported out-of-slice diagnostics).
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  33 tests, one Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 185 tests with
  the same one skip (184 passed).
- `.venv/Scripts/python.exe -m compileall -q cambam_builder legacy_cambam_builder tests demos`,
  `git diff --check`, and `.venv/Scripts/python.exe demos/mcp_authoring_slice.py`
  (updated for the `geometry` payload): pass.
- Direct-framework parity confirmed for all four families: adapter inspection
  payloads equal independently computed public query values (`1e-9` for the
  noise-bearing arc angle), XML semantics match after UUID/order/id
  normalization, save/reopen retains primitive identities and typed geometry,
  and failed edits preserve complete inspection.

Remaining limits: no Text/Region detail, no Pocket/Engrave/Drill, no
parenting/groups/copy-transfer, no bake transforms, Profile targets still
Rect-only, and inspection of a very large single primitive is not separately
bounded. The next batch reopens with Text/Region authoring or another
family if user needs change the priority.

## MCP Text and Region breadth batch 2 - 2026-09-10

4d batch 2 adds the annotation/composite authoring family: `geometry_add_text`
and `geometry_add_region`, with typed inspection detail for both and
`geometry_translate` extended to root Text/Region primitives. The reopening
criterion (curve/point family complete in batch 1) made Text/Region the next
mapped family; Pocket/Engrave/Drill MOP breadth remains next.

Implemented contract surface (schema copies updated and re-verified identical):

- `geometry_add_text` maps only to the public `add_text` adder: required
  identifier/layer/content/anchor, optional height (default 10), font (default
  Arial), style (default empty), line spacing (default 1) and alignment enums,
  plus elevation. Content is bounded 1..1024 characters, rejects control
  characters except newlines, and must contain at least one non-whitespace
  character because whitespace-only content does not survive the XML round
  trip. The optional unused `xml_p2_*` interchange fields are not authorable.
- `geometry_add_region` maps only to the public `add_region` adder: one closed
  outer contour plus 0..100 closed hole contours, each 2..10000 vertex records
  with per-vertex Z/bulge. The framework validates XY topology (closed, simple,
  finite nonzero area, holes contained and disjoint, no nesting) and topology
  failures are surfaced as `INVALID_ARGUMENT` with the bounded framework
  message by re-running the public `Region` validation on the rejected inputs;
  staged clones keep failure atomic.
- Inspection adds closed `text` payloads (anchor/height/font/style/line
  spacing/alignment plus the optional unused `p2`, no bounds because text
  extents are a font-dependent framework estimate) and `region` payloads
  (outer/hole contour `world_xyz`+`bulges` plus directed analytic bounds).
  Out-of-slice Text/Region (transformed or related primitives) stay
  `geometry: null` with `INSPECTION_UNSUPPORTED`.
- Verification notes: the batch-1 parity findings carry over (arc/text angle
  noise is absent here, but primitive document order remains the UUID-sorted
  listing), and the `objects`/id-canonicalized XML comparison is reused. A
  bulged outer Region (bulge 0.2 on the bottom segment, center (5,12), radius
  13) dips exactly to y=-1 and both the in-memory payload and the reloaded XML
  agree.

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_text_region.py -v`:
  5 passed (analytic guard, round-trip parity, translate parity/replay,
  negative/topology/conflict/atomicity, imported out-of-slice diagnostics).
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  38 tests, one Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 190 tests with
  the same one skip (189 passed).
- `compileall`, contract schema-copy identity, `git diff --check` and
  `demos/mcp_authoring_slice.py`: pass.

Remaining limits: no Pocket/Engrave/Drill MOPs, no parenting/groups/
copy-transfer, no bake transforms, Profile targets still Rect-only, text bounds
deliberately not reported. The next batch reopens with Pocket/Engrave/Drill
machining breadth or another family if user needs change the priority.

## MCP Pocket, Engrave, Drill breadth batch 3 - 2026-09-10

4d batch 3 adds the remaining basic machining MOPs and the public target-state
operation: `machining_add_pocket`, `machining_add_engrave`,
`machining_add_drill` and `machining_set_mop_targets`. This completes the four
basic CamBam MOP kinds behind closed, explicit parameter records.

Implemented contract surface (schema copies updated and re-verified identical):

- The three adders map only to the public `add_pocket_mop`/`add_engrave_mop`/
  `add_drill_mop` adders. Inputs are the shared closed machining record
  (targets, depth/increment, tool, feeds, spindle, stock surface, clearance,
  enabled); Pocket additionally pins stepover 0.4, `Plunge Feedrate` stepover
  feed, Conventional milling, collision detection, Spiral lead-in,
  `InsideOutsideOffsets` fill, finish stepover 0 and Roughing; Engrave pins
  Roughing, final increment 0 and DepthFirst with an EndMill; Drill pins the
  CannedCycle method, a `Drill` tool profile and exposes peck distance,
  retract height and dwell (defaults 0/5/0). All non-input settings match the
  current public framework defaults; Profile keeps its 4c record unchanged.
- `machining_set_mop_targets` maps to the public `set_mop_targets` and enforces
  the same per-kind target rules as creation before replacing the
  project-owned selection atomically on the staged clone. Targets are returned
  in the project's UUID-sorted snapshot order in results and inspection.
- Per-kind target rules are explicit: Profile root Rects; Pocket root
  Rect/Circle/closed-Pline/Region; Engrave root Rect/Circle/Arc/Pline; Drill
  root Points/Circle. CamBam resolves drill positions (point locations, circle
  centers) at toolpath time; that production semantics acceptance stays with 4e.
- Inspection generalizes the Profile parameter guard to all four kinds: a
  record is returned only when every exposed value matches the pinned fixed
  settings, every unexposed public field keeps its default, all scalar/range
  checks pass, every returned field's XML path is explicitly `Value` (never
  inherited) and every target resolves slice-valid under the kind rule.
  Imported or edited MOPs outside the slice stay diagnostic (`{}` parameters
  with `INSPECTION_UNSUPPORTED`).
- The shared prelude also centralizes depth/clearance relations, identifier
  and part-name conflicts, part auto-creation and limit checks for all four
  MOP adders; behavior for Profile is unchanged.

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_mops.py -v`:
  2 passed (pocket/engrave/drill creation + target replacement round trip with
  direct-framework XML parity; target-kind rules, negatives, replay,
  stale-revision and concurrency serialization).
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  40 tests, one Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 192 tests with
  the same one skip (191 passed).
- `compileall`, contract schema-copy identity, `git diff --check` and
  `demos/mcp_authoring_slice.py`: pass.

Remaining limits: no parenting/groups/copy-transfer tools, no bake/rotate/scale/
mirror/translate-Z tools, CamBam production toolpath semantics unverified
(4e), and MOP parameter editing beyond target replacement remains excluded.
The next batch reopens with the relationship/transform family.

## MCP relationship and transform breadth batch 4 - 2026-09-10

4d batch 4 adds the relationship and transform families and generalizes the
inspection slice from translation-only to similarity transforms, completing
the documented 4d mapping except cross-document copy/transfer.

Implemented contract surface (schema copies updated and re-verified identical):

- `relationship_set_parent` maps to the public `link_primitive_parent` (null
  detaches). Local transforms are preserved, so the world pose follows the new
  frame; self links and framework-rejected cycles return `INVALID_ARGUMENT`
  with complete inspection preservation. After linking, both endpoints leave
  the similarity slice by contract (relationships are diagnostic), which is
  asserted in inspection.
- `relationship_add_to_group`/`relationship_remove_from_group` map to the
  public group membership methods and return the sorted group names. Primitive
  inspection records now carry their sorted `groups` field.
- `relationship_copy_tree` maps to the public `copy_primitive_tree` into the
  same document with `preserve_ids=False`: copies get fresh UUIDs, and the
  framework's collision rejection (layer, primitive, group, part, MOP names)
  surfaces as `INVALID_ARGUMENT` unless the caller supplies the explicit
  identifier/group maps. `include_mops` (default false) copies MOPs whose
  selections lie inside the subtree. The result mapping includes copied
  layers/parts as well as primitives. Copied childless primitives return to
  the supported slice with typed geometry, and copied MOPs keep valid
  parameter records.
- Transform tools map to public methods only: `geometry_translate_z` uses
  `translate_primitive_z(..., bake=True)` (stored-geometry Z shift, world
  matrices retained), `geometry_rotate`/`geometry_scale`/`geometry_mirror`
  compose similarity world transforms (`bake=False`; scale is uniform-only
  because non-uniform scale leaves the slice), and `geometry_bake` calls the
  public `bake_geometry()` (non-axis-aligned Rects become closed Plines and
  the result type is returned; Text bakes only translation/positive uniform
  scale and otherwise fails `UNSUPPORTED_OPERATION`).
- The similarity slice: `_slice_supported` now accepts any root primitive
  whose world matrix is a finite non-degenerate XY similarity (translation,
  rotation, uniform scale, reflection) with zero local Z offset and no
  relationships. All public world queries and analytic bounds already support
  similarities, so rotated/scaled/mirrored primitives keep typed payloads and
  reflection flips bulge signs via the framework's orientation rule. The
  previously rotated-out-of-slice expectations in earlier suites were updated
  to non-uniform scale/shear cases, which remain diagnostic.
- Deliberately deferred: cross-document copy/transfer. The framework supports
  it (`transfer_primitive_tree` to another project), but an adapter tool
  spans two document handles, two revisions and one request ledger key, and
  cannot publish atomically across documents. A two-document staging,
  revision and failure-semantics decision must be recorded in the contract
  before such tools are advertised; the reopening criterion is in
  [PROGRESS.md](PROGRESS.md#active-work-and-next-priority).

Post-review corrections close three resilience/contract gaps before commit:
same-document copies now recheck the 10,000-primitive/1,000-MOP hard limits
before publication; typed geometry payloads are checked against their advertised
output schema so oversized imported Pline/Text detail returns `geometry: null`
with `INSPECTION_UNSUPPORTED`; and create/open/new-file save annotations are
nondestructive as required. Boundary regressions verify atomic state/revision
preservation for both copy limits, strict reopen of a 10,001-vertex Pline and a
1,025-character Text, and exact annotations for all advertised tools.

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_relationships_transforms.py -v`:
  5 passed (similarity transform parity incl. bake; parent/group/copy parity
  with cycle/self/missing rejection and copy collisions; copy-with-MOPs slice
  validity and limit atomicity; transform negatives, replay, stale and concurrency).
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  45 tests, one Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 197 tests with
  the same one skip (196 passed).
- `compileall`, contract schema-copy identity, `git diff --check` and
  `demos/mcp_authoring_slice.py`: pass.

Remaining limits: cross-document copy/transfer (deferred above), no generic
field setters or deletion tools, and CamBam production acceptance stays with
4e. Same-document copy requires explicit identifier/group maps for every
included non-unique name by framework contract.

## MCP portable document interchange - 2026-09-11

Client-local `.cb` files no longer need to be copied into the server workspace.
The adapter adds `document_import`, which accepts a complete UTF-8 XML snapshot
and publishes a revision-0 volatile handle through the existing strict reader,
and `document_export`, which serializes the locked revision to an in-memory byte
snapshot and returns a typed client artifact with filename, media type, encoding,
byte count, SHA-256 and complete XML content. Import retains only the content
digest and byte count in its process-lifetime retry signature; export is read-only
and is not retained in that ledger. Neither operation interprets a client name as
a server path, and export creates no server filesystem artifact.

The established 10 MiB XML limit remains unchanged and is enforced on encoded
UTF-8 bytes in both directions. Stdio framing increased from 1 MiB to 32 MiB only
to accommodate a valid 10 MiB payload after JSON string escaping and envelope
overhead. Inline content is the compatibility baseline because named-client
resource-result behavior is not yet accepted; 4e may add resource delivery if
it demonstrably reduces context cost without hiding the artifact from agents.
The standard tool result currently includes both structured content and the
backward-compatible serialized text block, so a large export may cross the client
boundary twice. This is a known context/network cost, not a correctness gap.

Automated coverage verifies import/edit/export/reimport geometry and identity,
source and artifact hashes, revisions, import retry/conflict behavior, malformed
and declaration-bearing XML rejection, exact 10 MiB acceptance, over-limit
rejection, invalid content types, stale export, an export/edit lock race and
bounded export failure. A real stdio subprocess moves an exact-10-MiB XML request
through the modern protocol path and recovers the exported artifact; a legacy
initialized connection also exports and reimports a document. Existing save
failure/atomic replacement regressions pass after extracting the shared in-memory
byte serializer.

Verification on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_documents.py -v`:
  18 tests, one existing Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_protocol.py -v`:
  9 passed, including the above-1-MiB real stdio round trip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_export_failures.py -v`:
  8 passed.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  49 tests, the same one skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 201 tests with
  the same one skip (200 passed).
- `compileall`, schema-copy identity (inside the document suite),
  `demos/mcp_authoring_slice.py` and `git diff --check`: pass.

No manual CamBam validation adds evidence for this transport and serialization
slice because it uses the same XML builder and strict reader already covered by
the existing interchange acceptance. Named desktop-client handling of a local
file and returned inline artifact remains 4e user acceptance. The next coherent
increment remains 4d batch 5 cross-document copy, now testable with two documents
imported from client content.

## MCP cross-document copy/transfer - 2026-09-11

4d batch 5 completes the documented 4d surface with two-document subtree
operations, advertised as `relationship_copy_tree_between` and
`relationship_transfer_tree_between` (thirty-one version 1 tools).

Recorded contract (now in the
[MCP contract](MCP_CONTRACT.md#cross-document-copy-and-transfer)): the tools
require two distinct live handles in this workspace and boot, each with its own
required revision assertion. Both document locks are acquired in canonical
sorted-handle order and held from the revision checks through publication;
concurrent edits/saves wait and concurrent closes cannot remove a locked
document. Copy stages on an independent full-project clone of the target and
reads the live source (the public copy never mutates it); transfer stages
independent clones of both documents and runs the public
`transfer_primitive_tree` between the clones, so source removal and target
insertion are one transactional framework operation. Publication replaces the
staged projects and increments revisions inside a cancellation shield with no
await between assignments: copy increments only the target revision, transfer
increments both by exactly one, and no partial two-document publication is
observable. Per-call failures (root resolution, framework collision/topology
rejection as `INVALID_ARGUMENT` with the bounded message, staged-target
limit overflow, per-document handle/revision mismatch) leave both documents,
both revisions and all files unchanged. `DomainError` gained an optional
addressed handle: a failing target document check (expired, not found, stale
target revision) references the target handle in the result envelope with its
current revision, and every other outcome references the source. Success data
always carries `{mapping, source_document, target_document, source_revision,
target_revision}` while the envelope's `document`/`revision` refer to the
source. Ledger rules are unchanged: regular edit entry per
`(workspace_id, request_id)`, replay before stale/closed checks, and
`REQUEST_ID_CONFLICT` on argument reuse.

Coverage: six tests in `tests/test_mcp_cross_document.py` — copy parity
between two imported documents (parented/grouped subtree with layer, part and
Profile MOP closure; a second childless copy returns to the typed similarity
slice; both saved sides match independently authored public-framework projects
by XML semantics); transfer into a populated imported document with both-revision
advancement, preservation of its existing content, source removal keeping the
untouched primitive/layer/part, and two-sided XML parity; failure addressing and
state preservation (same-handle
rejection, expired and closed targets addressed to the target, stale
source/target fields each reporting that document's current revision, missing
and non-primitive roots, framework collision message, failure replay with the
original key, and `REQUEST_ID_CONFLICT`); and limit/concurrency atomicity
(patched primitive and MOP limits reject without publication; two concurrent
same-revision transfers produce exactly one success and one `STALE_REVISION`
with correct final counts). Cancellation before publication preserves both live
project objects and revisions and replays `REQUEST_CANCELLED`; cancellation
inside the shielded publication completes both revisions and is recoverable as
a successful replay. Opposing A-to-B and B-to-A transfers complete under a
five-second timeout, proving both directions contend in canonical lock order.

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_cross_document.py -v`:
  6 passed.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  55 tests, one Windows symlink-privilege skip (tool listing/annotation
  expectations updated to the thirty-one tools).
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 207 tests with
  the same one skip (206 passed).
- `compileall`, contract schema-copy identity (inside the document suite),
  `demos/mcp_authoring_slice.py` and `git diff --check`: pass.

Remaining limits: no deletion or batch tools, no cross-restart recovery of
volatile state, and CamBam production acceptance stays with 4e. 4d is
complete; the next increment is 4e desktop/second-PC acceptance.
