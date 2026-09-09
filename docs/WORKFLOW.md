# Working procedures

## Work lifecycle

[PROGRESS.md](PROGRESS.md) owns the ordered backlog and one active bounded slice. Promote an
item by recording objective, affected owners and executable acceptance criteria.
Record blockers there; implementation detail belongs in the domain owner.
Close a slice by separately recording implementation, automated verification and
user acceptance. Pending user acceptance must remain visible. Retire completed
plans after transferring durable facts and failure evidence to their owners.

Use these states explicitly: **backlog**, **active**, **blocked**, **implemented**,
**automated checks complete**, and **accepted**. A recommendation is still backlog
until work begins. Blocked work records the missing decision/dependency and a
concrete unblocking action; do not open a second implicit active priority.
For documentation-only work, user/production acceptance may be marked not required
with a reason. For product behavior, identify the specific validation still owed.

An active item needs only: objective, scope/owners, acceptance criteria, current
state, blocker (or none), and next action. Links can supply detail. Do not duplicate
the active item in the remaining backlog. If an external issue tracker is designated,
it owns task state/acceptance and the repository status links to the active issue;
the repository continues to own contracts and verification instructions.

## Progressive retrieval and decisions

Start at the README/topic map, then inspect status and the one relevant owner.
Use searches such as `rg -n '^##' docs/structure_spec.md` and
`rg -n 'get_total_transform' cambam_builder demos -g '*.py'`; read bounded sections,
then callers/tests where a contract crosses modules. Do not load every guide,
historical record, asset or generated report by default. Return a short conclusion
with file/symbol references, first failures and local log paths. Measure repeated
missed symbols or discovery effort before proposing indexing infrastructure.
Broaden retrieval only when an unresolved symbol, caller, failing check or contract
dependency supplies a concrete reason. Stop once the slice and its verification
are supported; unrelated findings become brief evidence for backlog triage.

Use safe, reversible assumptions for minor matters; record material ones with the
work. The lead resolves implementation choices within the authorized contract.
Ask the user for missing decisions only when they materially change architecture,
behavior, acceptance, destructive actions or expensive work. Existing authorization
continues to apply; a routine technical review is not a new permission gate.

For a consequential decision or failed investigation, append a compact dated record
to `docs/REVIEW.md`: question/context, evidence, chosen approach, rejected alternatives
and why, reopening criteria, and verification/acceptance state. Put the resulting
current contract in its normal owner and link to this evidence. Do not manufacture
rejected alternatives for trivial edits. Add a domain-specific record only when
volume warrants it and update the topic map rather than creating a parallel wiki.

## Delegation execution

[MODEL_ROUTING.md](MODEL_ROUTING.md) is the sole role-selection guide. Keep the
immediate blocking step with the lead and continue useful non-overlapping work while
agents run. Give editing workers disjoint ownership, preserve concurrent changes and
wait for required results before integration. Give one focused correction for a
misunderstood task; return unresolved semantics to the lead. Validate worker claims
in proportion to their impact and use the same correctness and privacy standard for
every route.

### Worker packet

```text
Objective: one bounded result
Scope/boundary: what is included and excluded
Permitted files/excerpts: explicit list (no private data)
Necessary contract context: owner links and relevant requirements only
Acceptance: executable checks or evidence criteria
Constraints: privacy, compatibility, dependencies, shared temporary resources
Edit permission/exclusive ownership: read-only or named files
Output shape/limit: concise result appropriate to the task
Result form: findings with file/symbol evidence, patch, or tests; no transcript
```

## Verification and handoff checklist

- Inspect the initial diff; preserve unrelated edits and define acceptance.
- Use [the runbook](DEVELOPMENT.md); record interpreter/environment limitations.
- Run focused checks; broaden when shared contracts change. Inspect counts,
  references, geometry and parameters in serialized output, not just exit status.
- Review the final diff and documentation links; keep detailed logs local.
- Report changed areas, assumptions, exact commands/results and unverified risks.
- Identify user/domain validation, next increment and suggested commit message.
- Do not conflate passing synthetic checks with CamBam or machining acceptance.

### Completion record / handoff template

```text
Slice and date:
Changed areas and owning files:
Decisions/assumptions and evidence links:
Implementation: pending / complete
Automated verification: exact commands, exit/results, reproducible fixtures/artifacts
User/production acceptance: pending / accepted / not required (reason)
Acceptance authority/evidence: user or domain reviewer; never infer acceptance
Remaining risks, blockers and reopening criteria:
Recommended next increment:
Suggested commit message:
```

Keep the live state in [PROGRESS.md](PROGRESS.md) (or its designated issue). The review record
owns detailed completed evidence; handoffs link to it and do not create another
status file. A failed/unperformed required check keeps technical closure pending.
