# Working procedures

## Work lifecycle

`PROGRESS.md` owns the ordered backlog and one active bounded slice. Promote an
item by recording objective, affected owners and executable acceptance criteria.
Record blockers there; implementation detail belongs in the domain owner.
Close a slice by separately recording implementation, automated verification and
user acceptance. Pending user acceptance must remain visible. Retire completed
plans after transferring durable facts and failure evidence to their owners.

## Capability-based delegation

| Capability | Suitable work | Keep with lead |
| --- | --- | --- |
| Fast retrieval/classification | Narrow searches, authorized extraction, consistency checks, synthetic test ideas | Cross-layer interpretation |
| Focused implementation | Exclusive small code/test/docs slice with stable contract and executable checks | Overlapping edits or undefined behavior |
| Deep review | Bounded difficult logic, regression/security concern or defined design | Final integration and acceptance decisions |

Choose the least costly capable route that satisfies tool, privacy and integration
requirements. Inspect actual available tools and routing/telemetry when cost
distinctions matter; availability and model names are session facts, not contracts.
If external processing is not authorized, use a native worker. Coordinate temporary
resources. Give one focused correction for a misunderstood task, then return
unresolved semantics to the lead. Duplicate reviews only when independence is
worth the added cost. Workers are not alone: preserve other workers' edits.

### Worker packet

```text
Objective: one bounded result
Scope/boundary: what is included and excluded
Permitted files/excerpts: explicit list (no private data)
Necessary contract context: owner links and relevant requirements only
Acceptance: executable checks or evidence criteria
Constraints: privacy, compatibility, dependencies, shared temporary resources
Edit permission/exclusive ownership: read-only or named files
Output budget: short limit
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
