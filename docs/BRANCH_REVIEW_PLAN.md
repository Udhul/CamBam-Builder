# Rest/V branch engineering and delivery review

Historical completed review mandate. [Final policy, verification and delivery](REVIEW.md#python-312-minimum-and-session-5-closure---2026-09-29)
and [merge closure](REVIEW.md#merge-closure-and-next-task-planning---2026-09-29)
own the outcome; [PROGRESS](PROGRESS.md) owns subsequent priorities. The retained
method and test-adequacy criteria describe the original review's evidence standard.

## Mandate and baseline

Requested 2026-09-27: review `feat/rest-machining-and-vcarving` against `main`,
including native versus extended capabilities, reusable foundations, mathematical
correctness, foreseeable misuse and API relationships. This is a five-session
review programme, followed by the user's integration decision. Planning is not
review acceptance. Current state and priority belong only in [PROGRESS](PROGRESS.md); dated findings
and coverage belong in [REVIEW](REVIEW.md). This document retains the historical
mandate and routes completed session scopes to their evidence.

Historical planning snapshot (2026-09-27):

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

The original review required the following method; it is historical scope rather
than a pending procedure for the completed branch.

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

## Test adequacy across all five sessions

The review must answer: **would these tests detect a plausible implementation
that violates intended behavior, or do they merely preserve today's output?**
Passing tests establish only what their inputs, oracles and assertions can detect.
Review existing tests and identify missing tests alongside each capability; do not
defer this judgment to the final full-suite run. Include unchanged tests where a
changed contract depends on them. Test counts and line/branch coverage are discovery
tools, not acceptance criteria for behavioral correctness.

Maintain a behavior-to-test matrix in REVIEW alongside the coverage ledger. Each
row records the intended behavior and authoritative source, owning API/capability,
valid domain and boundary partitions, forbidden outcome, existing test IDs and
assertions, oracle provenance/independence, missing cases, plausible defect the
test should detect, and adequacy disposition. Live repairs/priorities stay in
PROGRESS. Derive expectations from product intent, accepted contracts, mathematical
invariants or independently observed native behavior before comparing code output.
If sources conflict, resolve the contract explicitly; current behavior is not its
own specification. Separate characterization of existing behavior from correctness
and compatibility tests, and identify intentional compatibility requirements.

| Review dimension | Required evidence |
| --- | --- |
| Core and native contracts | Direct tests of geometry/math/topology, stock, transforms, identity and interchange, including supported-domain boundaries and unsupported-input diagnostics. Higher-level happy paths cannot replace these invariants. |
| Extended capabilities | Observable target fidelity, residual/coverage, feasible tool selection, primary V behavior, rest overlap/smoothing and partial/infeasible outcomes. Check results rather than private call sequences or incidental path ordering unless order is contractual. |
| Outliers and degeneracy | Nominal, just-inside/on/just-outside limits, empty and disconnected geometry, thin features, numerical extremes and malformed/non-finite inputs where relevant. Include both valid unusual inputs and invalid ones. |
| Combinations and state | Native/generated stages, tool profiles, stock/predecessors, order/repetition, units/frames, topology, output dialects and backend availability. Select interactions by failure mechanisms, use pairwise coverage where useful, and add higher-order cases for coupled hazards; explain omitted combinations. |
| End-to-end and isolation | Public calls through actual serialization/decoding/replay, plus focused owner tests; deterministic generated fixtures, fresh state, no dependency on test order, hidden local artifacts or mock acceptance. |

Audit expected-value construction and assertion strength. Look for production
helpers reused to compute expected answers, writer/reader errors that cancel in a
round trip, generator and verifier sharing the same faulty assumption, and golden
outputs copied from an unvalidated run. Independent analytic references, separately
derived invariants and source-bound native observations can provide stronger
oracles; each still needs a stated domain. A snapshot is useful for a justified
byte-compatibility contract, but is not by itself a machining-correctness oracle.

Challenge weak assertions (nonempty output, no exception, success flags, broad
exception catches, only total area when topology matters), tolerances broad enough
to hide defects, missing lower/upper bounds, vacuous loops, skips/expected failures,
over-mocking and tests that never reach their intended rejection condition. Verify
negative tests fail for the intended cause and leave state unchanged where required;
pair them with a nearby valid case to detect implementations that reject everything.
Never loosen a tolerance or regenerate a golden result solely to make code pass;
require an independent error budget or an explicitly corrected contract.

For high-consequence invariants and suspected weak tests, demonstrate sensitivity
with a bounded counterexample or controlled fault injection: wrong sign/units,
dropped hole, omitted predecessor, bypassed freshness/clearance check, changed arc
direction or rounding intrusion, as applicable. Record which assertion detects it.
An existing negative case counts only if it exercises that defect. Where injection
adds evidence, use isolated task-owned copies under `output/`; do not mutate the
working source concurrently or introduce a mutation-testing dependency by default.
Surviving faults require a stronger test or an explained equivalent/unreachable
case. Add deterministic regression tests for demonstrated reusable gaps, showing
failure on the defective behavior and success after repair where feasible.

**Adequacy gate:** classify each reviewed behavior as adequately controlled,
partially controlled, untested or asserted against a disputed expectation, with
specific evidence. Required claims with inadequate evidence remain open even when
the full suite is green. Address material gaps during the owning session; defer
only with a justified scope/claim limit and reopening criterion. The final review
must say which behaviors the suite protects, what it could still miss, and which
tests were added, strengthened or retired and why. Do not imply exhaustive coverage
of every input or composition.

## Session 1: Capability boundaries and public API contracts

Capability ownership, public/reference boundaries, full-diff allocation and API
relationships are recorded with [session 1 findings and coverage](REVIEW.md#branch-review-session-1---2026-09-27).
The resulting [capability map](structure_spec.md#capability-and-public-api-boundary-map)
owns the current architecture contract.

## Session 2: Geometry, topology and numerical foundations

Equations/domains, topology, numerical direction and independent oracle review
are recorded with [session 2 findings and coverage](REVIEW.md#branch-review-session-2---2026-09-28).
Current [foundation assumptions and guarantees](structure_spec.md#foundation-assumptions-and-numerical-guarantees)
belong in the specification.

## Session 3: Machining strategies, rest behavior and reuse

Strategy composition, primary/rest V behavior, policy and reuse assessment are
recorded with [session 3 findings and coverage](REVIEW.md#branch-review-session-3---2026-09-28).
The [strategy guarantees and limits](structure_spec.md#strategy-guarantees-and-composition-limits)
own the implemented boundary.

## Session 4: Execution safety, evidence and misuse resistance

Source-to-output identity, decoded motion, stock dependence, transitions and
misuse resistance are recorded with [session 4 findings and coverage](REVIEW.md#branch-review-session-4---2026-09-29).
The [mediation evidence contract](structure_spec.md#mediation-invariants-and-evidence-contract)
owns current rules.

## Session 5: Integrated regression and delivery decision

Final coverage reconciliation, fixture disposition and package/delivery findings
are recorded with [session 5 evidence](REVIEW.md#branch-review-session-5---2026-09-29).
The [Python minimum and final closure](REVIEW.md#python-312-minimum-and-session-5-closure---2026-09-29)
own replacement checks and the final delivery judgment;
[current development gates](DEVELOPMENT.md#verification-entry-points) own commands.

## Required session handoff

See the [completion and handoff procedure](WORKFLOW.md#completion-record--handoff-template).
The completed review's coverage, decisions and evidence remain in the dated
REVIEW sections above; further work is selected from [PROGRESS](PROGRESS.md).
