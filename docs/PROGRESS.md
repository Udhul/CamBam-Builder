# Current status

Status updated 2026-10-26 for CF01 on `feat/cf01-native-interchange`, based on
clean local `main` at `78fe421`. The remote inventory below remains the earlier
audit snapshot at `b400610`; it is not a new remote delivery check. This page owns
live priority and delivery state. The
[product direction](structure_spec.md#product-priority-and-library-consumers),
[implemented contracts](structure_spec.md#0-implemented-architecture-and-change-ownership)
and [dated audit evidence](REVIEW.md#project-direction-and-foundation-audit---2026-10-26)
own requirements, behavior and findings respectively.

## Active work and next priority

<a id="doc01-documentation-consolidation-backlog"></a>

**CF01 implementation and engineering acceptance are complete.** The native
integrity repairs, direct board consumer and verification are recorded in
[REVIEW](REVIEW.md#cf01-native-interchange-integrity---2026-10-26). The branch is
committed separately from `main`; merge remains user-owned. **Next recommended
implementation: NC01's first native geometry coverage increment.** Establish the
maintained native coverage map and advance Line/Spline interchange from its
explicit unsupported state. This addresses the next native compatibility gap
for consuming applications. BO01 remains separate and unchanged.

### CF01: native interchange integrity

**State:** complete within the [native interchange contract](structure_spec.md#native-interchange-integrity-cf01).
Explicit and unspecified units, owned PointList identity and native operator Tags
survive edit/export/reopen. Metric board and independent inches/native-input
workflows establish the direct-library outcome; the base-wheel example works
outside the checkout. The [acceptance record](REVIEW.md#cf01-native-interchange-integrity---2026-10-26)
owns regression counts, source evidence, package provenance and limitations.

The stopping condition is satisfied without changing its scope. No additional
user observation is required for these serialization claims. General unknown XML,
full native class coverage, arbitrary 3D setups, bin packing, G-code generation
and MCP parity remain outside CF01. No BO01 search changes were integrated.

### Current branch closure

| Remote branch | Tip at audit | Delivery state relative to `origin/main` |
| --- | --- | --- |
| `main` | `b400610` | Delivered baseline; includes feature-aware rest, native-rest repairs and ornamental inlay work. |
| `feat/tool-path-bundle-search` | `06509e4` | Only unmerged feature branch: 3 commits ahead, 0 behind. BO01-A/B implementation and recorded offline acceptance are committed; BO01-C closure remains outstanding. |
| `feat/feature-aware-v-rest` | `9b4ba03` | Merged at `b400610`; earlier ready-to-commit statements are historical. |
| `feat/composed-stock-verification` | `e44186f` | Merged; ancestor of main. |
| `feat/cam-verification-hardening` | `86ec581` | Merged; ancestor of main. |
| `feat/rest-machining-and-vcarving` | `199f1f4` | Merged; ancestor of main. |
| `mcp-fixes` | `82ae0ea` | Merged; ancestor of main. |
| `mop-export-policy-audit` | `cb72be7` | Merged; ancestor of main. |
| `dev/create-progress.md-plan-for-framework-development` | `284a947` | Merged historical tracker branch. |

BO01 commits are `7ab21cc` (finite planar search), `6686adb` (routing docs) and
`06509e4` (mixed-tool/paired-inlay search and offset repair). Its tree changes 16
files; it is not part of the installed mainline baseline. Branch-local PROGRESS
still calls B ready to commit, but B is committed. The feature branch's
`REST_MACHINING_PLAN.md` BO01 epic, contracts and dated REVIEW entries own its
implementation details until integration; inspect them with `git show` or on that
branch. This audit did not certify BO01 merge readiness or rerun its behavior tests.

When BO01-C is selected, reconcile the final branch diff and existing evidence,
fix material defects only, and reconcile this newer direction with its changes to
AGENTS, PROGRESS, the specification and model routing. Do not overwrite the new
priority or reintroduce stale status during integration. No model refresh is
required. Close the bounded scope; continuous optimization and further search
features stay deferred. Commit/merge are user-owned unless explicitly delegated;
use the existing readiness gate and visible `--no-ff` integration preference.

At that earlier audit, local `main` was left at `853f749` (205 commits behind fetched main); the audit
branch was created from `origin/main` to avoid modifying that older branch. No tags
or stashes were present and there was one worktree. The public GitHub API reports
no open issues or pull requests. Its complete returned history contains only
[PR #1](https://github.com/Udhul/CamBam-Builder/pull/1), merged 2025-07-02,
and no standalone issues. BO01 has no open PR at audit time. Repository docs and
branches therefore remain the available work-tracking source; CI/check-run state
and any work outside this repository were not audited.

### Post-merge task queue

This is the implementation order after the audit; reassess at each coherent
outcome using the [strategy checkpoint](WORKFLOW.md#strategy-checkpoint).

| Order | Increment | Useful outcome / scope |
| --- | --- | --- |
| 1 | **NC01: native coverage and preservation** | Promote the existing native inventory into a maintained manual/API-to-code/test coverage map and select the first Line/Spline interchange increment. Stage Surface/3D MOP support plus safe preservation of unmodeled document context separately. Surface/3D interchange is **NR03**; it does not require independent 3D path generation. Select one useful native slice at a time, with explicit import/author/preserve/unsupported and evidence levels. Include NCFile, Lathe, scripts and external bitmap assets in the map with explicit dependencies, not implicit full-support claims. |
| 2 | **CF02: reusable manufacturing composition** | A simple installed-library cabinet consumer with multiple boards, groups/inherited transforms, grooves/slots, materials/thicknesses and separate front/back setup outputs. Establish missing reusable setup/frame APIs from this consumer and a distinct non-cabinet job. Keep cabinet design rules in the caller and stock placement manual initially. |
| 3 | **PK01: consumer release checkpoint** | Build/install wheel and sdist outside the checkout, verify documented public imports and simple CAD/MOP export on supported Python versions, version the accepted scope and publish compatibility notes. No tag or published-release claim until actually delivered; package registry publishing remains separately authorized. |
| 4 | **BO01-C** | Close/reconcile the existing committed search branch; preserve useful work without expanding the search epic. May be a bounded maintenance interlude sooner if independent and cheap, not a competing active priority. |
| 5 | **Native extended-engine output** | Improve reusable rest/V/inlay attachment through ordinary Regions/Plines and MOPs for a named consumer. Retain protected boundaries, signed allowances and specified overlap. Existing XYZ/Engrave whole-motion gaps remain explicit; inspectable path geometry alone is not a complete execution certificate. |
| 6 | **SF01** | Generalize bounded relief contact/stock and shape-aware finishing when the native/library foundation and a concrete consumer justify it. NR03 remains a separate native interchange capability. |

This order does not require completing all possible NC01 classes before CF02 or a
useful extension. Close a coherent native slice and re-evaluate. Detailed engine
proposals stay in the [rest/V design](REST_MACHINING_PLAN.md#follow-up-contracts-and-representative-consumers).

## Remaining backlog, in order

These are conditional/deferred items, not additional active work.

| Scope | Promotion trigger and owner |
| --- | --- |
| Existing engine edge cases | Tapered plug offset candidate can emit a zero-length V segment on main; BO01-B reports a repair, so inspect/reuse it before duplicating work. Hard overlap caps, fitted paths and low cleared links need a named consumer and continuous containment/access evidence. [RP01 limits](structure_spec.md#feature-aware-planar-vrest-candidates-rp01). |
| Further native parameter/shape variants | Broader tab geometry and fresh Triangle output need named native observations; retain accepted Square-tab evidence. [Manual tab evidence](REVIEW.md#manual-profile-tab-native-fixture--2026-09-27). Native field/state coverage belongs to NC01, not MCP parity. |
| Full independent G-code / optional native automation bridge | Future capability with explicit controller/post semantics and actual runtime acceptance where claimed. Existing bounded adapters stay supported; native `.cb` export and core engine development must not wait. [Execution boundary](REST_MACHINING_PLAN.md#programmatic-execution-requirement). |
| Controller runtime and physical validation | Promote for a real cutting job with actual tooling/material/setup. [Runtime boundary](REVIEW.md#m5-controller-runtime-evidence-scope---2026-09-26). No pending physical observation blocks the accepted offline baseline. |
| Stock logistics and bin packing | Inventory, sheet/offcut selection, allocation across materials/thicknesses, packing and scheduling are later support features, promoted when integration cost is low and a caller needs them. Preserve native stock fields and engine removal-state semantics now. |
| Broader geometric/fixture backends | Non-box fixtures, arbitrary 3D frames, general surfaces/volumes, transition/holder envelopes and numerical refinement need a consumer and independent error/coverage evidence. [Capability map](structure_spec.md#capability-and-public-api-boundary-map). |
| Optional MCP evolution | Remote transport, sequence repair, new geometry/engine tools and parity/performance work stay deferred. Record known discrepancies under the [MCP policy](MCP_CONTRACT.md#maintenance-priority-and-coverage); do not expand core work into adapter maintenance. |
| Package/core orchestration migration | Promote only when a real consumer or change-locality problem requires it; avoid a speculative framework rewrite. [Staged organization](structure_spec.md#package-organization-decision-and-migration-plan). |
| Planning extensions / CLI | New catalogs, derating, profile persistence or a CLI need concrete caller input and benefit. [Planning contract](structure_spec.md#milling-pass-and-candidate-planning). |
| Documentation simplification | Consolidate touched historical/current sections when they cause stale instructions or repeated retrieval. Preserve unique observations and repair links; no blanket rewrite or new tracking surface. [Ownership rules](README.md#maintenance-rules). |

## Delivered capabilities

These remain closed for their recorded scopes, not general production certification.

### IN01: ornamental paired-stock and assembly verification

Straight/tapered paired stock, assembly and composite facing are merged in
`b400610`, including repair `9b4ba03`. [Contract](structure_spec.md#composite-stock-facing-and-final-inlay-verification-in01)
and [acceptance](REVIEW.md#in01-composite-facing-and-final-verification---2026-10-03).

### NR01: editable native rest boundaries and MOPs

Implementation `3122803`, native observation `010829a` and binding repair
`9b4ba03` are merged in `b400610`. [Native acceptance and limits](REVIEW.md#nr01-editable-native-rest-preparation---2026-10-02).

### RP01: feature-aware planar V/rest candidates

Implementation `096ba97` is merged in `b400610`.
[Contract](structure_spec.md#feature-aware-planar-vrest-candidates-rp01) and
[offline acceptance](REVIEW.md#rp01-feature-aware-planar-rest-candidates---2026-10-02).

### RQ01: reliable and proportionate merge readiness

Delivered with the repaired feature branch; no uncommitted RQ01 work remains on
main. [Procedure](WORKFLOW.md#correctness-evidence-before-a-readiness-decision) and
[demonstration](REVIEW.md#pr-blocker-repairs-and-rq01-readiness-evidence---2026-10-03).

### Branch review session queue

All five rest/V review sessions are closed for their recorded scopes.
[Historical programme](BRANCH_REVIEW_PLAN.md); [rest/V merge](REVIEW.md#merge-closure-and-next-task-planning---2026-09-29),
[verification merge](REVIEW.md#cam-verification-epic-merge-closure-and-priority-assessment---2026-09-30),
and [DT01/MV01/MX01 delivery](REVIEW.md#mv01mx01-committed-review-and-merged-delivery---2026-10-02).
DOC01 was delivered as `bbd4f97`; current documentation policy is in the topic map.

#### Next detached stock/rest increment

Historical anchor: the earlier section-motion/RC01 work is closed. Follow the
[current queue](#post-merge-task-queue), not an old session handoff.

### 9d: Machine and user operating ranges (completed 2026-09-21)

Closed arithmetic scope; [evidence and reopening conditions](REVIEW.md#r1-r3-repair-and-caller-defined-operating-ranges---2026-09-21).

## Blockers and decisions

CF01 has no outstanding acceptance blocker. Full native parity and production readiness
are not established. Required native/controller/physical observations
are specific to each advertised capability; preserve previous acceptance and do
not re-request it just because a branch merged.

## Completed line: hierarchy and global transform fidelity

Closed for the [global-transform](REVIEW.md#global-transform-acceptance-and-workstream-checkpoint)
and [full-bake](REVIEW.md#full-bake-ab-user-acceptance) scopes; general 3D setup
transforms are still outside those guarantees.

## Completed implementation: MOP round-trip identity

Closed for [identity](REVIEW.md#mop-identity-round-trip-verification),
[A/B observation](REVIEW.md#mop-ab-user-acceptance) and later
[MOP ownership/interchange](REVIEW.md#mop-core-ownership-and-interchange-redesign).
CF01 separately closes PointList nesting identity under its bounded contract.

## Completed implementation: export failures and state paths

Closed for the [export contract](structure_spec.md#export-failure-and-state-saving-contract)
and [verification](REVIEW.md#export-failure-and-state-path-verification).

## Completion and verification

CF01's implementation, automated checks and engineering acceptance are complete;
[REVIEW](REVIEW.md#cf01-native-interchange-integrity---2026-10-26) records the
evidence and limits. A fresh session is appropriate for NC01: CF01 has a coherent
verified outcome and no unresolved product decision or pending native observation.
The next task has distinct class-coverage scope and does not depend on this chat.
