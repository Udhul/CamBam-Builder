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

| Rank | Finding and classification | Impact / confidence / approximate cost |
| --- | --- | --- |
| 1 | **Verified:** parent links are lost on XML round trip; a UUID-only repair also changes world geometry | High: hierarchy and geometry fidelity; high confidence; small-to-medium coherent I/O slice |
| 2 | **Verified:** two-node parent cycles are accepted | High: transform traversal can fail to terminate; high confidence; small validation slice |
| 3 | **Verified:** non-baked global transform order contradicts the method contract | High: rotation after translation yields wrong world position; high confidence for root case; medium cost across nested/baked cases |
| 4 | **Source-confirmed risk:** export catches entity/MOP serialization exceptions and continues | High: incomplete output may appear successful; high confidence in control flow, no injected failure run; medium cost to define failure/atomic-write behavior |
| 5 | **Known gaps / enhancements:** MOP registry migration, project-default import, curved bounds, transfer APIs, packaging/test coverage | Variable impact; code/TODO/spec evidence, not a complete compatibility audit; separate bounded increments required |

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
