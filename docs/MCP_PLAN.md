# Future local MCP integration

User-requested direction, captured 2026-09-07. This is a lightweight pending plan,
not an implemented API or installation guide. Priority/state belong exclusively
to [PROGRESS.md](../PROGRESS.md); existing correctness repairs remain ahead of it.

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

Decisions deferred until activation: initial coverage, SDK/transport and client
versions, document persistence/revision model, packaging and supported OS matrix.
None blocks backlog capture. No server, dependency or advertised tool exists yet.
