# Future local MCP integration

User-requested direction, captured 2026-09-07. This is a delivery plan,
not an implemented API or installation guide. Priority/state belong exclusively
to [PROGRESS.md](PROGRESS.md). The decided 4a design lives in the
[MCP contract](MCP_CONTRACT.md), with [machine-readable schemas](mcp_contract_v1.schema.json).

## Outcome and requirements

Install the project on another PC, start a local MCP server easily, and connect a
compatible AI agent (Codex is the preferred client). The user can then request CAD/CAM
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
the normative specification and SDK/client conformance again when work starts.
The user expanded the requirements on 2026-09-10 after wire probes showed current
clients still using older protocols: **backward compatibility is required scope**,
beginning with stdio MCP 2025-06-18 (installed Codex) and 2025-11-25 (OpenCode).
Use the pinned SDK's protocol negotiation and retain one application contract
across versions. Legacy initialization must never select an active document or
weaken workspace, revision, retry, validation or save rules. Report the actually
negotiated protocol; a successful legacy connection is not modern conformance.

## Boundary (resolved by the MCP contract)

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

The resulting [authoritative contract](MCP_CONTRACT.md) and
[probe evidence](REVIEW.md#mcp-protocol-and-adapter-contract---2026-09-10)
resolve this scope. OpenCode 1.18.29 fails modern-only interoperability; 4b/4c
use SDK 2.2.0 with the legacy compatibility boundary implemented in 4b, while
desktop distribution acceptance remains gated.

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
- Include SDK-backed compatibility with MCP 2025-06-18 and 2025-11-25 alongside
  modern direct requests. Verify Codex's real client and both legacy versions;
  reject mixed-version traffic and unsupported versions without state changes.
- Implement protocol/server wiring, explicit document references or handles,
  revision checks, structured diagnostics and the configured workspace boundary.
  Enforce canonical path containment, save/overwrite policy and restart behavior;
  expose no arbitrary Python, private registries, pickle loading or machine control.
- Add the narrowly required public project clone and strict byte-snapshot XML
  reader contracts identified in [4a](MCP_CONTRACT.md#documents-revisions-and-atomicity),
  with focused framework regressions. Keep those rules out of tool handlers.
- Add focused contract tests for startup/discovery, invalid inputs/references,
  traversal/escape attempts, stale revisions, repeated requests and atomic errors.

**Stopping condition:** the server foundation is installable and deterministic,
security/state contracts pass without CAD feature breadth, and exact checks plus
known limits are recorded. Start 4c in a fresh session.

### 4c. First authoring and round-trip vertical slice

**Completed 2026-09-10.** Durable behavior remains in the
[adapter contract](MCP_CONTRACT.md#tools-and-public-api-mapping), current
state in [PROGRESS.md](PROGRESS.md), and verification evidence in
[REVIEW.md](REVIEW.md#mcp-first-authoring-and-round-trip-slice---2026-09-10).

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
- Support portable client-content import/export before second-PC acceptance so a
  client can edit local `.cb` content without sharing the server filesystem.
- Update the adapter impact check in the workflow and the contract/runbook as
  each public family becomes supported; do not claim complete CamBam coverage.

**Stopping condition:** the declared supported surface and exclusions match code,
schemas and parity tests, all relevant framework regressions pass, and no unresolved
contract decision is carried only in chat. Begin distribution acceptance fresh.

### 4e. Installation, local client and user acceptance

**Recommended main session:** GPT-5.6 Terra, high reasoning for packaging/runbook
execution; move to Sol high only for substantive interoperability defects.
**Relative usage:** medium model usage plus external user time.

- Validate clean installation and uninstall/rollback without affecting direct
  library users; verify the supported Python/OS and adapter dependency matrix.
- Publish exact local start/stop and named-client configuration instructions.
  Connect the selected desktop client to its own local stdio subprocess and repeat
  the 4c workflow through the actually negotiated supported protocol. Record
  modern/legacy conformance separately; include backward compatibility regressions.
- Record agent usability and required CamBam geometry/property/toolpath inspection
  as user/domain acceptance, keeping automated, protocol and production evidence
  separate.
- Treat portable import/export content as the default for the client's own project
  files. Server-workspace open/save is explicit scratch or artifact scope and must
  never cause a client agent to request filesystem access to the server workspace.
- Require agents to verify returned/inspected geometry against requested bounds,
  centers, dimensions and containment before CAM or export, and to ask rather than
  fabricate unspecified machining depths, feeds or spindle settings.

**Stopping condition:** isolated clean install, local stdio connection, protocol
conformance, end-to-end client behavior and required user acceptance are recorded;
rollback is demonstrated and the live backlog advances. This completes the initial
adapter. The user accepted same-machine stdio as the initial deployment boundary on
2026-09-11 because second-PC hardware is unavailable; Streamable HTTP and remote-PC
deployment are a separate non-urgent backlog item. Named-client CAM acceptance must
also verify that enclosed/detail MOPs precede any final through-cut Outside Profile.

## First implementation proof and acceptance

After the relevant framework defects are fixed, prove one end-to-end slice:
create a small document, add geometry and a supported MOP, inspect it, save it,
reload and modify it, then save under a new name through MCP. Compare output
semantics with direct framework calls, including IDs, relationships, world geometry
and machining parameters. Test malformed arguments, invalid references, file-access
limits, repeated calls and explicit document/revision behavior.

Demonstrate installation in an isolated environment and connection from a named
compatible client that launches the server locally over stdio. Test the requested protocol separately
from tool behavior. Record agent usability and CamBam geometry/toolpath inspection
as user/domain acceptance; a successful tool response alone is insufficient.
Expand API coverage only after that representative slice passes. Rollback should
disable/uninstall the adapter without changing existing user documents or the
framework's direct Python API.

## Maintenance obligations during implementation

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

The [MCP contract](MCP_CONTRACT.md) owns initial coverage, SDK/transport/client,
document persistence/revision and packaging-boundary decisions. Increment 4e owns
the supported OS installation matrix. Increment 4b now provides the optional
server and five document tools; 4c adds the first three authoring tools after its
direct-framework parity slice passed.
