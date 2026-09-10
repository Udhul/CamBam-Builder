# Future local MCP integration

User-requested direction, captured 2026-09-07. This is a lightweight pending plan,
not an implemented API or installation guide. Priority/state belong exclusively
to [PROGRESS.md](PROGRESS.md); existing correctness repairs remain ahead of it.

## Outcome and requirements

Install the project on another PC, start a local MCP server easily, and connect a
compatible AI agent (for example OpenCode Desktop). The user can then request CAD/CAM
work in natural language and the agent can create, inspect, load, modify and save
CamBam documents through supported framework operations. Include geometry,
relationships, transforms and machining operations as their contracts become
reliable. Server maintenance is part of framework maintenance, not a separate
one-off demonstration. Client compatibility must be tested, not assumed.

Target the stateless **MCP 2026-07-28** contract requested by the user. The
[official release announcement](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
describes self-describing requests without the initialization handshake or protocol
session ID, optional discovery, and explicit application handles when state is
needed. Stateless transport does not require discarding document state. Verify
the normative specification and SDK/client conformance again when work starts;
do not silently substitute an older session-based protocol for client convenience.

## Proposed boundary (subject to implementation design)

- Keep a thin adapter in this repository, calling public framework operations.
  Geometry, relationships and XML rules stay in the framework. Do not duplicate
  them in tool handlers or expose arbitrary Python execution/private registries.
- Use documented, typed tool inputs and structured results: stable entity IDs,
  document references, actual saved artifact paths, diagnostics and explicit
  failure/partial-result information. Expose useful task-level operations, not
  an automatic tool for every internal method. State units, coordinate conventions,
  defaults and unsupported behavior so agents need not infer machining semantics.
- Each call identifies its document/workspace explicitly. Choose file-backed
  references or explicit document handles during design; define persistence,
  restart behavior, revisions/concurrent edits and retry behavior. Never rely on
  one process-global active document or transport session for correctness.
- Default to local use and a configured workspace boundary. Define path access,
  overwrite/save-as behavior and failure atomicity. Imported file content is data;
  do not expose untrusted pickle loading as an agent-facing interchange path.
- Provide a documented install/start/stop procedure and client configuration
  example with a verified SDK/protocol/framework version combination. Choose
  packaging (for example an optional extra and launcher) and transport after
  checking target-client support; avoid dependencies for library-only users where
  practical. Remote hosting and machine control are outside this initial aim.

## Delivery increments and session boundaries

Execute the work in order. Each increment is a complete project outcome with its
own verification and documentation handoff; do not pull later API breadth into an
earlier increment. Model recommendations identify the appropriate main-session
driver, not a requirement to delegate or consume the entire model tier.

### 4a. Protocol, client and adapter contract

**Recommended main session:** GPT-6 Astra, high reasoning; use xhigh only if the
normative stateless protocol or target-client behavior remains ambiguous after
direct evidence. **Relative usage:** high for one architecture-focused session.

- Verify the normative MCP 2026-07-28 specification, current Python SDK support
  and at least one named desktop client's actual transport/protocol capabilities.
- Decide transport, file-backed references versus explicit handles, restart and
  persistence behavior, revision/concurrent-edit rules, retries/idempotency,
  workspace/path policy, overwrite/save-as behavior and error/result envelopes.
- Select the smallest initial tool set and map each tool to public framework APIs,
  units, identifiers, defaults and explicit exclusions. Decide optional dependency
  and launcher packaging without adding them yet unless a disposable protocol
  probe is required to settle compatibility.
- Create the authoritative MCP contract guide and update the topic map/status.
  Record primary-source and client-probe evidence plus reopening criteria.

**Stopping condition:** every material implementation decision above is persisted,
the first vertical slice has testable schemas and acceptance criteria, and no
implementation depends on conversational context. This is the intended Astra-to-
Sol fresh-session handoff.

### 4b. Secure server and document foundation

**Recommended main session:** GPT-5.6 Sol, xhigh reasoning. **Relative usage:**
medium-high for one bounded implementation session.

- Add the minimal adapter package, optional dependency boundary and documented
  local start/stop command selected in 4a.
- Implement protocol/server wiring, explicit document references or handles,
  revision checks, structured diagnostics and the configured workspace boundary.
  Enforce canonical path containment, save/overwrite policy and restart behavior;
  expose no arbitrary Python, private registries, pickle loading or machine control.
- Add focused contract tests for startup/discovery, invalid inputs/references,
  traversal/escape attempts, stale revisions, repeated requests and atomic errors.

**Stopping condition:** the server foundation is installable and deterministic,
security/state contracts pass without CAD feature breadth, and exact checks plus
known limits are recorded. Start 4c in a fresh session.

### 4c. First authoring and round-trip vertical slice

**Recommended main session:** GPT-5.6 Sol, xhigh reasoning. **Relative usage:**
high for one feature-and-parity session.

- Implement only the tools needed to create a document, add representative
  geometry and one supported MOP, inspect it, save it, reload and modify it, then
  save under a new name.
- Compare MCP results with direct public-framework calls for IDs, relationships,
  world geometry, machining parameters and serialized XML. Cover malformed
  arguments, invalid references, repeated calls and revision behavior.
- Provide a runnable synthetic demonstration, but defer broader entity/operation
  coverage and desktop-client acceptance.

**Stopping condition:** the complete slice passes adapter contract tests and
direct-framework parity checks, with artifacts inspected and durable evidence
recorded. This proves the abstraction before API expansion and is a fresh-session
breakpoint.

### 4d. Supported API breadth and resilience

**Recommended main session:** GPT-5.6 Sol, high reasoning; Terra xhigh is suitable
for clearly mapped, lower-risk coverage batches. **Relative usage:** high overall,
prefer multiple outcome-sized sessions if coverage does not fit one coherent unit.

- Expand only to explicitly supported geometry, relationships, transforms and
  machining operations, following the 4a mapping and proven 4c patterns.
- Add schema/API parity fixtures and negative tests for every added family. Test
  concurrent/stale edits, restart recovery, retry/idempotency and failure atomicity
  across the supported surface. Keep unsupported operations explicit.
- Update the adapter impact check in the workflow and the contract/runbook as
  each public family becomes supported; do not claim complete CamBam coverage.

**Stopping condition:** the declared supported surface and exclusions match code,
schemas and parity tests, all relevant framework regressions pass, and no unresolved
contract decision is carried only in chat. Begin distribution acceptance fresh.

### 4e. Installation, client and user acceptance

**Recommended main session:** GPT-5.6 Terra, high reasoning for packaging/runbook
execution; move to Sol high only for substantive interoperability defects.
**Relative usage:** medium model usage plus external user time.

- Validate clean installation and uninstall/rollback without affecting direct
  library users; verify the supported Python/OS and adapter dependency matrix.
- Publish exact local start/stop and named-client configuration instructions.
  Connect the selected desktop client on a clean second-PC environment and repeat
  the 4c workflow through the negotiated stateless protocol.
- Record agent usability and required CamBam geometry/property/toolpath inspection
  as user/domain acceptance, keeping automated, protocol and production evidence
  separate.

**Stopping condition:** clean-machine install, connection, protocol conformance,
end-to-end client behavior and required user acceptance are recorded; rollback is
demonstrated and the live backlog advances. This completes the initial adapter.

## First implementation proof and acceptance

After the relevant framework defects are fixed, prove one end-to-end slice:
create a small document, add geometry and a supported MOP, inspect it, save it,
reload and modify it, then save under a new name through MCP. Compare output
semantics with direct framework calls, including IDs, relationships, world geometry
and machining parameters. Test malformed arguments, invalid references, file-access
limits, repeated calls and explicit document/revision behavior.

Demonstrate installation and connection on a clean second-PC environment with a
named compatible desktop client. Test the requested stateless protocol separately
from tool behavior. Record agent usability and CamBam geometry/toolpath inspection
as user/domain acceptance; a successful tool response alone is insufficient.
Expand API coverage only after that representative slice passes. Rollback should
disable/uninstall the adapter without changing existing user documents or the
framework's direct Python API.

## Maintenance obligations when implementation starts

The framework change author owns assessing adapter impact; the lead owns integration.
In the implementation increment:

1. Establish an authoritative MCP contract guide mapping supported public operations
   to tool schemas, results/errors, compatibility policy and intentional exclusions.
2. Add adapter contract tests and direct-framework/MCP parity fixtures. Require
   them for changes affecting exposed signatures, defaults, units, identifiers,
   transformations, serialization or errors; update schemas/docs in the same slice.
3. Add the exact commands to the development runbook and the impact check to the
   agent workflow. Add subtree instructions only if distinct adapter rules justify
   them. Maintain version compatibility and test protocol/SDK upgrades explicitly.
4. Move lasting contracts and launch instructions into their owners, update the
   topic map, and reduce this plan to evidence/links when complete.

Increment 4a owns the previously deferred initial coverage, SDK/transport/client,
document persistence/revision and packaging-boundary decisions. Increment 4e owns
the final supported OS installation matrix. No server, dependency or advertised
tool exists yet.
