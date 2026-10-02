# Current status

Status reviewed 2026-10-02. The modern package implements native CAD/CAM
interchange, detached bounded CAM calculation, ordered stock verification and
local MCP authoring. The [implemented architecture and capability boundaries](structure_spec.md#0-implemented-architecture-and-change-ownership)
own those contracts; [REVIEW](REVIEW.md) owns dated acceptance and limitations.

## Active work and next priority

<a id="doc01-documentation-consolidation-backlog"></a>

### RP01: feature-aware planar V/rest candidates

**Accepted by engineering 2026-10-02; ready to commit.** Contact and
sampled-medial candidates consume composed stock, omit union-proved air sweeps,
retain safe high links and report located residual/overlap and floor cusp bounds.
The synthetic frieze demonstrates better detail coverage than raster/offset at
the same requested controls, with higher travel cost. The
[RP01 contract](structure_spec.md#feature-aware-planar-vrest-candidates-rp01)
and [measured evidence](REVIEW.md#rp01-feature-aware-planar-rest-candidates---2026-10-02)
own scope, verification and limits. No manual observation is needed for this
offline gate; native/runtime and physical claims remain separate.

DOC01 documentation consolidation is delivered as `bbd4f97`; its ownership rules
remain in the [documentation map](README.md#maintenance-rules).

### Post-merge task queue

**Delivered baseline:** the rest/V branch review, CAM verification runner and
DT01/MV01/MX01 capability work are accepted for their recorded offline scopes.
The latest MV01/MX01 merge is `2f10b07`; [committed review and delivery evidence](REVIEW.md#mv01mx01-committed-review-and-merged-delivery---2026-10-02)
own the final checks and acceptance. The [rest/V merge record](REVIEW.md#merge-closure-and-next-task-planning---2026-09-29)
and [verification epic merge record](REVIEW.md#cam-verification-epic-merge-closure-and-priority-assessment---2026-09-30)
own earlier delivery. Historical pending/priority statements in those dated
records do not reopen closed work.

The user's expanded ornamental, inlay, native-rest and relief requirements live
in the [product expectation map](REST_MACHINING_PLAN.md#product-expectations-and-capability-follow-ups-2026-09-30).
Continue with shared capabilities and distinct synthetic consumers under the
[framework direction](structure_spec.md#framework-direction-and-extension-principles).
The order below is a development priority, not a requirement to finish all native
work before detached inlay or relief work.

| Order | Backlog increment | Useful next outcome |
| --- | --- | --- |
| 1 | **NR01** | Editable derived rest Regions/Plines and fully native MOPs bound to the original design; accept complete actual CamBam posts. |
| 2 | **IN01** | Ornamental receiver/plug targets with independent fit, glue, seating and backing allowances; insertion and assembled-stock finishing/final motif checks. |
| 3 | **BO01** | Generate and search finite tool/path bundles under explicit finish, cost and setup constraints; independently verify candidates and report search quality. |
| 4 | **SF01** | Bounded relief contact/stock foundation and shape-aware 3D tracing/finishing. **NR03** separately establishes native Surface/3D MOP/post interoperability. |

The [follow-up contracts and representative consumers](REST_MACHINING_PLAN.md#follow-up-contracts-and-representative-consumers)
own scope, dependencies and acceptance. RP01's bounded output establishes useful
detached ornamental planning; NR01 now matters because editable native rest output
is the next distinct workflow gap. Conditional fitting/overlap/low-access work is
listed below; finite tool search belongs to BO01; broader surfaces
belong to SF01. Reassess order after each coherent outcome. Pre-v6 ordered
bundles require regeneration under the [MX01 contract](structure_spec.md#mixed-cylindricalv-composition-and-whole-tool-access-mx01).

**Next agent task:** implement NR01 editable derived rest boundaries and fully
native MOPs bound to the original design under its
[follow-up contract](REST_MACHINING_PLAN.md#follow-up-contracts-and-representative-consumers).
Start a fresh session for NR01: it has distinct source binding and actual
native-post acceptance needs; its requirements are durable.

### Branch review session queue

**All five sessions are accepted by engineering and delivered.**
[Execution scopes and gates](BRANCH_REVIEW_PLAN.md) remain the review programme;
[session 1 findings](REVIEW.md#branch-review-session-1---2026-09-27),
[session 2 findings](REVIEW.md#branch-review-session-2---2026-09-28),
[session 3 findings](REVIEW.md#branch-review-session-3---2026-09-28),
[session 4 findings](REVIEW.md#branch-review-session-4---2026-09-29) and
[session 5 closure](REVIEW.md#python-312-minimum-and-session-5-closure---2026-09-29)
own findings, coverage and acceptance. Current implementation order is the
[post-merge queue](#post-merge-task-queue).

## Remaining backlog, in order

The queue above owns unconditional implementation priority. These conditional
items remain visible without reopening the accepted bounded scopes.

| Pending scope | Reopening need and detail owner |
| --- | --- |
| Planar fitted paths, hard overlap caps and low cleared links | Reopen for an actual consumer requiring a declared fit/overlap limit or low-access route; require continuous containment, topology and coverage proof. RP01's high-link candidate needs none of these. [RP01 limits](structure_spec.md#feature-aware-planar-vrest-candidates-rp01). |
| Controller runtime and supervised physical validation | Prioritize when the next goal is cutting a real part. Select actual controller/machine, tooling, stock and setup; prepare exact observations. [Runtime evidence boundary](REVIEW.md#m5-controller-runtime-evidence-scope---2026-09-26) and [job acceptance packets](REST_MACHINING_PLAN.md#ordered-next-session-job-packets-selected-2026-09-27). |
| Native Pocket and XYZ/Engrave whole-motion role parity | Reopen for a carrier/post control that can encode missing entry, retract and setup roles, then audit a fresh complete post. NR01 covers useful native rest output. [Pocket route failure](REVIEW.md#rc01-pocketdefault-role-carrier-assessment---2026-09-23), [M1 rejected native entries](REVIEW.md#m1-first-actual-cambam-posts-and-contour-only-revision---2026-09-24) and [XYZ Engrave failure](REVIEW.md#bounded-xyz-engrave-cambam-post-finding---2026-09-24). |
| Fresh Triangle-tab output and broader tab geometry | Bounded Manual authoring and B/C Square output are accepted. Fresh Triangle output plus curved, reversed, transformed or multi-target variants require their named native evidence. [Manual native fixture and limits](REVIEW.md#manual-profile-tab-native-fixture--2026-09-27), [fresh writer acceptance](REVIEW.md#fresh-writer-cambam-default-post-acceptance--2026-09-27). |
| Remote MCP transport (historical backlog 5) | Non-urgent; promote only for an actual remote-PC or multi-client need. Define authenticated Streamable HTTP modern/legacy behavior, bind/origin/TLS policy, document lifecycle and cancellation before exposing the write-capable server. [MCP boundary](MCP_PLAN.md#boundary-resolved-by-the-mcp-contract). |
| Broader geometry, precision and setup occupancy | Reopen for a named target/fixture or tolerance/workload blocked by bounded methods; require independent coverage oracles and decoded motion/setup binding. Non-box fixtures, inlay occupancy and helix/transition body checks remain outside current guarantees. [Strategy guarantees](structure_spec.md#strategy-guarantees-and-composition-limits), [body/fixture contract](structure_spec.md#bounded-tool-body-and-fixture-occupancy) and [stock numerical limits](REVIEW.md#cumulative-directional-stockrest-bounds---2026-09-22). |
| Package/core orchestration migration | Promote when a second paired-output consumer, independently packaged core or obstructed dialect proves the need. [Staged organization owner](structure_spec.md#package-organization-decision-and-migration-plan) and [BR-1-004 finding](REVIEW.md#findings-and-dispositions). |
| MCP sequence repair and geometry affordances | A reorder tool reopens for a named-client misordering/repair need; general containment/parametric helpers require demonstrated authoring benefit. [Existing operation-order observation](REVIEW.md#mcp-agent-geometry-feedback-correction---2026-09-11). |
| CLI/publishing | Promote only for an actual consuming workflow. [Original packaging scope and reopening criteria](REVIEW.md#packaging-and-supported-python-verification). |
| Planning extensions | New constraint/derating models, sourced catalogs, profile persistence or MCP profile composition require concrete supplied inputs and acceptance data. [Planning evidence](REVIEW.md#milling-pass-planning-and-bounded-mcp-exposure---2026-09-21) and [operating-range closure](REVIEW.md#r1-r3-repair-and-caller-defined-operating-ranges---2026-09-21). |

Historical backlog numbering remains meaningful in older reviews: 1a/1b were
curved bounds and copy/transfer, 2 shape parity, 3 packaging, 4a-4e local MCP,
5 remote transport, 6 rest/V work, 7 Manual tabs, 8a-8e MOP audit, 9a-9d milling
planning and 10 entity ownership. Items 1-4 and 7-10 are complete for their
accepted scopes; item 6 continues through the named capability queue. The old
Python 3.9-3.13 packaging checkpoint is historical; the
[development toolchain](DEVELOPMENT.md#environment-and-setup) owns current support.

#### Next detached stock/rest increment

This stable historical anchor now leads to the [current capability queue](#post-merge-task-queue).
The earlier directional, holed-target, section-motion and RC01 increments are
closed; their [stock/replay evidence](REVIEW.md#bounded-section-motion-verification---2026-09-22)
and [RC01 evidence](REVIEW.md#rc01-standalone-generated-sequence---2026-09-23)
preserve numeric bounds and reopening conditions.

### 9d: Machine and user operating ranges (completed 2026-09-21)

Closed arithmetic scope; [operating-range evidence and reopening conditions](REVIEW.md#r1-r3-repair-and-caller-defined-operating-ranges---2026-09-21)
own the completed request. Further planning needs remain conditional above.

## Blockers and decisions

No blocker or missing observation remains for the delivered offline baseline.
Same-machine MCP stdio is accepted; second-PC hardware was unavailable and remote
transport remains conditional. [Local client/CamBam acceptance](REVIEW.md#mcp-4e-acceptance-completion---2026-09-21)
owns that boundary. Actual controller/machine execution and physical machining
remain separate from engineering acceptance; promote them for the real-cutting
need described above.

## Completed line: hierarchy and global transform fidelity

Closed within the [global-transform acceptance](REVIEW.md#global-transform-acceptance-and-workstream-checkpoint)
and [full-bake acceptance](REVIEW.md#full-bake-ab-user-acceptance) scopes.

## Completed implementation: MOP round-trip identity

Closed within the [MOP identity evidence](REVIEW.md#mop-identity-round-trip-verification)
and [A/B acceptance](REVIEW.md#mop-ab-user-acceptance) scopes; later
[MOP ownership/interchange evidence](REVIEW.md#mop-core-ownership-and-interchange-redesign)
owns the redesign.

## Completed implementation: export failures and state paths

Closed within the [export/persistence contract](structure_spec.md#export-failure-and-state-saving-contract)
and [verification evidence](REVIEW.md#export-failure-and-state-path-verification).

## Completion and verification

Current baseline acceptance/delivery links are in the [post-merge queue](#post-merge-task-queue).
Dated evidence stays in REVIEW; current contracts stay in their topic owners;
[development commands](DEVELOPMENT.md) and [acceptance/handoff procedures](WORKFLOW.md)
own verification practice. RP01's bounded offline gate is closed; contracts,
measured limits and next priority are saved. This coherent detached planning
outcome is a good fresh-session breakpoint for NR01's distinct native geometry
and source-binding scope. No commit or merge was performed in this work round.
