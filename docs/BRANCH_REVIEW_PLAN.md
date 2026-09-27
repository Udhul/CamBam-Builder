# Rest/V branch engineering and delivery review

## Mandate and baseline

Requested 2026-09-27: review `feat/rest-machining-and-vcarving` against `main`,
including native versus extended capabilities, reusable foundations, mathematical
correctness, foreseeable misuse and API relationships. This is a five-session
review programme, followed by the user's integration decision. Planning is not
review acceptance. Live session state and priority belong only in
[PROGRESS](PROGRESS.md#branch-review-session-queue); dated findings and coverage
belong in [REVIEW](REVIEW.md). This document owns execution scope and gates.

Planning snapshot (refresh at every session start):

- Branch HEAD: `69988f76cd3ce2627ffd17be1b83183052c7fa98`.
- Local `main` and merge base: `18dbb9950f3065e95df0ee645a25361d45e63b30`.
- `main..HEAD`: 88 commits; `main...HEAD`: 162 files, 47,417 insertions and
  7,282 deletions. Initial worktree clean. These are inventory facts, not a
  content review or a statement about a remote branch.

Review the complete committed diff, including migrations, deletions, compatibility
facades, tests, fixtures, metadata and documentation. Follow unchanged callers and
dependencies when needed to establish the changed contract. Accepted M0-M5 and the
five job packets are evidence to interrogate, not substitutes for branch review.
Preserve their bounded acceptance unless a counterexample invalidates it.

The five sessions are meaningful review units, not deadlines that waive evidence.
If a unit cannot close, retain it as active with the exact remaining coverage or
repair; the next session resumes it. Do not skip work to reach session 5.

## Review method and completion rules

Start with the repository reading order, this plan, the active queue and latest
review evidence. Record base/HEAD/worktree and compare with the reviewed revision.
Read implementations, callers and tests; directory names, docstrings, test counts
and passing examples do not establish a contract. Prioritize false clearance,
unintended removal and stale evidence, then compatibility and composability.
All changed files still need disposition.

Maintain a durable coverage ledger in REVIEW with exact paths (group only when
every member is named), reviewed revision, session, relevant symbols, caller/test
evidence and disposition. Distinguish inspected implementation, independently
checked behavior, unresolved concern and deferred capability. Session inventories
and logs may live in unique ignored `output/branch-review-<unique>/` directories;
the durable conclusion must survive without them. Reconcile the ledger against
the final diff; new edits reopen affected contracts and consumers.

For each finding record a stable `BR-<session>-<number>` ID, severity, code location,
contract, triggering inputs, expected/actual behavior, evidence or reproducer,
impact, acceptance scope invalidated, repair and regression criterion. Classify:

| Class | Disposition |
| --- | --- |
| Delivery blocker | Supported behavior is incorrect, unsafe within its claimed model, stale evidence is accepted, compatibility is broken, or a required gate fails. Repair and verify before delivery. |
| Contract/documentation defect | A caller could reasonably misuse an ambiguous or overstated API. Correct the owning contract and implementation guard where necessary; safety-relevant ambiguity blocks delivery. |
| Capability/architecture gap | A supported contract lacks a required foundation, or coupling prevents an intended caller. Establish impact with a concrete consumer; decide whether delivery is blocked. |
| Future extension | Outside current claims and correctly rejected or explicitly limited. Record rationale, owner and reopening condition in the existing backlog; do not implement speculatively. |

Severity reflects consequence and reachability, not file size or elegance.
Unresolved technical uncertainty is not a pass. Repair demonstrated defects in
the owning contract with focused counterexamples; broaden checks for shared
changes. Do not refactor packages for naming alone, add a backend/registry without
evidence, or conflate an absent feature with a mathematical bug. First principles
means explicit models, derivations, invariants and independent evidence; it does
not require reimplementing established geometry kernels.

Keep current API/architecture facts in [structure_spec](structure_spec.md), CAM
semantics in [REST_MACHINING_PLAN](REST_MACHINING_PLAN.md), procedures in
[DEVELOPMENT](DEVELOPMENT.md), and MCP behavior in its existing contract/schema.
For unfamiliar external algorithms or controller semantics, verify against primary
documentation and record version/relevance. Analogy is not compatibility proof.

## Session 1: Capability boundaries and public API contracts

**Outcome:** an evidence-backed capability/dependency map, complete diff coverage
allocation, and explicit distinctions between implemented public contracts,
bounded reference jobs and intended extensions.

Inspect exports, native migration/facades, import directions, shared values,
core/extension/integration entry points and consumer docs. Use `native/`,
`cam_core/`, `cam_extensions/`, root planar/stock/math modules, `integrations/`,
MCP and `pyproject.toml` as discovery boundaries, not proof of correct ownership.
Allocate every changed file to a session; session 5 reconciles final coverage.

Build the capability matrix in the architecture owner. Each row names entry
points, owner/dependencies, supported domain, rejected cases, evidence level,
public/internal/reference status and extension seam. Cover these groups:

| Group | Required distinction |
| --- | --- |
| Native CamBam | CAD/MOP authoring and XML fidelity, normalized intent, actual posted motion and versioned native algorithm parity are different claims. |
| Reusable foundations | Geometry/topology, units/frames, cutter profiles, contact/sweep, evolving stock, access/collision and uncertainty predicates. |
| Machining capabilities | Toolpath calculation, primary V-carving without a required roughing predecessor, rest analysis and rest path generation; distinguish kernels from strategy choices. |
| Policy | Tool/bundle selection, order, costs, recommendations and optional smoothing; candidate ranking is not necessarily search or global optimization. |
| Integration/workflow | Native import/attachment, output lowering, verification, MCP and convenience jobs; workflow composition must not own machining truth. |

Trace one detached generated path and one native-plus-generated path from caller
inputs through replay and output, locating dependencies and state authority.
Check analysis-only, supplied-motion, alternate-tool/order and primary-V callers
can use supported capabilities without fixture identifiers, mandatory file
sequences, global sessions or GUI. Restrictions must not masquerade as generality.

For public entry points document units/frames/signs, target versus stock meaning,
tool domains, mutability, outputs, failure/partial/unsupported states, freshness
and required call order. Identify undocumented interrelations, hidden preconditions,
fixture constants and shared mutable state. Check old imports/native identity and
whether MCP mirrors changes to native APIs.

**Exit evidence:** capability/API and dependency map linked to symbols/tests; all
diff paths allocated; boundary defects reproduced or explicitly unresolved;
architecture docs corrected. Use focused boundary, migration, round-trip and
relevant MCP checks for repairs. This defines what sessions 2-4 will challenge;
it does not certify algorithms.

**Next session task:** audit geometry, topology, stock mathematics and numerical
guarantees against the session 1 contracts.

## Session 2: Geometry, topology and numerical foundations

**Outcome:** an assumption/guarantee inventory and independent checks of the
mathematical foundation, with a justified reusable-core gap assessment.

Start with `planar.py`, `_planar_shapely.py`, `stock.py`, core replay,
curved-region, volume, surface and occupancy primitives plus tests/callers.
Inspect native transforms/geometry where normalization depends on them. Read full
contracts around changed equations, not isolated expressions.

For each material calculation record equation/algorithm, symbol units, domain,
orientation/signs, exact versus approximate status, error direction, degeneracies,
implementation symbol and independent oracle. Challenge:

- Offset curves/polylines versus set erosion/dilation; open/closed contours,
  winding, holes/islands, tangencies, self-intersections and disconnected sets.
- Area and volume offsets, clearance/contact, edge/boundary queries and path
  access: what exists, is delegated or absent, and what callers can safely
  compose. A 2D buffer operation does not establish a 3D capability.
- Swept cutting volume versus non-cutting occupancy, complete segments/arcs and
  changing Z/radius; remaining stock, desired removal and protected material stay
  distinct. Later operations must not erase earlier overcut reports.
- Cone half-angle/flat-tip/cutting-length limits, rounded-tip joins/continuity,
  ball contact, sections, surface/volume integrals and frame transformations.
- Enclosure directions through union/intersection/difference, tessellation,
  between-slice guarantees, conservative cell bounds, accumulated tolerances,
  finite precision, unit scaling and output rounding.
- Empty/full stock, zero/negative/non-finite inputs, nearly coincident edges,
  thin features, extreme tool ratios, narrow V angles and conditioning.
  Distinguish proof, conservative enclosure and heuristic sampling.

Use analytic references, dimensional checks and metamorphic properties
(translation/rotation where supported, consistent scaling, stock monotonicity and
refinement enclosure). Reference checks must not repeat the implementation's
same assumption or algorithm. Challenge topology separately from area accuracy:
a small area error can hide a lost bridge or hole. Inspect backend conversion
validity and repair semantics, not just success status.

**Exit evidence:** foundational claims have explicit assumptions and test/proof
disposition; counterexamples have regression coverage; reusable gaps have concrete
consumers/owners. Run focused planar, stock, replay, surface/volume and occupancy
suites as applicable. A missing formal proof is not automatically a defect when
the exposed contract accurately bounds what is established.

**Next session task:** evaluate strategies, primary V-carving, tool selection and
rest smoothing using the audited primitives and limits.

## Session 3: Machining strategies, rest behavior and reuse

**Outcome:** establish whether generated capabilities honor their targets and
compose beyond recipes without weakening coverage, edge fidelity or feasibility.

Inspect RC01, convex/polygon/curved rest, V-region, pointed/rounded/variable V,
inlay and `cam_extensions/strategy.py`; follow native/direct consumers. Use
accepted jobs as references, then vary dimensions, topology, tools, supplied
stock and ordering within the claimed domain.

- Separate target construction, cutter feasibility, candidate passes, entry/link
  planning, verification and selection. Identify reusable calculations embedded
  in recipes; extract only when independent callers justify it.
- Treat V-carving as primary machining as well as cleanup. Check wide areas,
  capped depth, unreachable corners, floor cusps/scallops and honest finite-tool
  residual/partial completion; paired inlay is a consumer, not its definition.
- Trace pure rest to allowed overlap, feasible centers, passes and removal.
  Evaluate conditional smoothing into verified cleared space separately from
  changing original edges. Check cutter-diameter fit, facing fidelity, holes/tabs,
  thin bridges, access and final swept coverage after fitting. If smoothing is
  absent, record the gap; a buffer or contour offset is not automatically equivalent.
- Separate geometric reachability/path finding from pass ordering and controller
  dynamics. Check continuous approach, plunge, retract, links and intermediate
  stock, including travel across partially cleared regions.
- Audit bundle objectives, feasibility filters, residual/engagement measures,
  tool-change/cutting/air costs, deterministic ties, dominated candidates and
  partial/infeasible results. Identify supplied-bundle ranking versus bundle
  generation/optimization; state search scope without unsupported optimality
  claims. A cheaper route cannot override failed safety gates.
- Examine hidden fixture/tool labels and duplicate verifiers; judge extension
  points using a second supported consumer, not hypothetical universal interfaces.

**Exit evidence:** strategy guarantees/omissions mapped to reusable owners;
non-default supported variations check composability and independent material
results; counterexamples reject or repairs pass focused suites. Prioritize gaps
in the existing backlog with reopening criteria. Do not create another job packet
merely to increase example count.

**Next session task:** test execution adapters, state/evidence invalidation and
foreseeable API misuse against the audited machining contracts.

## Session 4: Execution safety, evidence and misuse resistance

**Outcome:** determine whether actual emitted/imported motion and caller state
receive only the evidence they earned, with actionable failures.

Inspect ordered jobs, replay integration, native normalization/series audits,
stock authority, scripts/carriers, controller writers/readers, handoffs and MCP
changes. Reuse native observations only for their unchanged source/post scope.

- Trace source intent -> normalized geometry/motion -> ordered stock -> output
  bytes -> independently decoded motion -> final audit. Check generator/verifier
  common-mode assumptions and complete-byte parsing, including added commands.
- Challenge stale geometry, tools, units/frames, fixtures, tolerance, source/post
  pairing, order, skipped/disabled/repeated stages and mutated results. A hash
  establishes identity, not physical correctness or provenance by itself.
- Check modal units/distance/arc-center modes, G2/G3/helix interpretation,
  effective tip/offset composition, tool changes, split-program restart, safe
  points, pauses/macros/external effects and unknown commands. Unsupported effects
  must not disappear or silently become assertions of zero motion.
- Separate cutting clearance, shank/holder/fixture clearance, declared transition
  effects and physical setup. Test low rapids, between-endpoint collisions,
  between-height intrusions and rounding into protected material.
- Audit preview/intent/actual motion confusion, missing/stockless inputs, export
  versus verified output, absent optional backends and callers bypassing convenience
  workflows. Diagnostics must expose evidence scope and partial/unsupported/stale
  states without implying machine-ready execution.
- Check native/MCP round-trip/mutation contracts, schema agreement and relevant
  path/artifact handling. Identify exact behavior needing external observation;
  technical uncertainty cannot be resolved by user assent.

**Exit evidence:** misuse matrix with calls, expected rejection or qualified result
and tested outcome; source-to-output evidence trace; no unresolved false acceptance
within advertised scope. Run focused native, ordered output, controller, freshness,
occupancy and affected MCP suites. For necessary external evidence, first prepare
and inspect synthetic artifacts and precise pass/fail steps under
[acceptance ownership](WORKFLOW.md#acceptance-ownership). Do not repeat unchanged
accepted posts or request a general machining trial for offline closure.

**Next session task:** resolve delivery findings, reconcile complete diff coverage
and run final branch/package gates against `main`.

## Session 5: Integrated regression and delivery decision

**Outcome:** final engineering review and exact delivery state, including a
prioritized judgment of framework foundations and remaining limits.

Reconcile coverage with final `main...HEAD` and all review repairs. Inspect remaining
diffs, including test quality, deleted behavior, docs, dependency/package metadata
and historical byte fixtures. Confirm explicit authorization/provenance and reusable
value for tracked `.cb`/`.nc` fixtures; do not automatically delete historical
evidence. Inspect untracked candidates and ignored CAM files outside `output/`,
preserving unrelated work.

Execute the declared [development gates](DEVELOPMENT.md#verification-entry-points):
full unittest discovery, compile/import smoke, applicable MCP checks, optional
backend present/absent behavior and required supported-Python/package verification.
Use clean wheel/sdist installs and imports/tests outside the checkout as the
runbook prescribes. Record interpreter/dependency versions, test/skip counts,
commands, artifact inspection and unavailable gates. Earlier package tests do not
prove this branch's new modules install/run. Skips cannot replace required evidence.

Summarize findings by consequence, verified repairs, remaining limits and capability
gaps. Give an explicit judgment on native/extended separation, API usability,
foundational reuse and mathematical guarantees with supporting symbols/checks.
Compare the next increment with the overall backlog and user needs, not simply
the next nearby edge case.

Apply [delivery rules](WORKFLOW.md#verification-and-handoff-checklist): name `main`
and final HEAD, require clean worktree/branch commits, inspect the complete diff,
run `git diff --check main...HEAD` plus ancestry/status/commit-range gates. If
`main` advanced, assess that relationship and required integration work; do not
substitute a remembered base. Uncommitted repairs/docs can be **ready to commit**,
never **merge-ready**. After a user commit, rerun branch gates and confirm checks
cover final content. No staging, commits, publication or merge is authorized here.

**Stop:** all engineering/final committed-branch gates pass, or a specific failing
or unavailable gate is recorded with its remedy. Only on merge-ready closure
provide the user's `git merge --no-ff` commands. Runtime/physical machining stay
separate unless a claim/change requires their evidence.

**Next session task:** the highest-impact backlog item selected from the review;
if delivery is blocked, resolve the named blocker or verify the user's final
commit first. Persist that selection in PROGRESS before handoff.

## Required session handoff

Every session updates owning contracts, dated REVIEW evidence/coverage and the
single PROGRESS queue. State changed areas, decisions/assumptions, exact checks,
engineering acceptance, invalidated claims, remaining risks, specific required
user observations, delivery state and suggested commit message. End with one
actionable next task linked to the queue and a separate continue/fresh-session
recommendation. A breakpoint is valid when evidence/outstanding work are durable;
name pending outputs or decisions rather than losing them. The user should never
need to repeat the review mandate.
