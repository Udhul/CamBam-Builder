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
and remaining backlog live only in [PROGRESS.md](../PROGRESS.md).

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
by [PROGRESS.md](../PROGRESS.md). This dated review owns the defect evidence and
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
