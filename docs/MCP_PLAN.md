# Future local MCP integration

Historical delivery outline for the completed local adapter. The [MCP contract](MCP_CONTRACT.md)
owns current requirements, protocols, state, supported tools and exclusions;
[the runbook](DEVELOPMENT.md#local-mcp-setup-and-verification) owns installation
and verification. [PROGRESS](PROGRESS.md) owns current priorities. The original
increment anchors below route to their final contracts and unique evidence.

## Outcome and requirements

The requested natural-language CAD/CAM collaboration and deployment boundary are
implemented through the [local adapter contract](MCP_CONTRACT.md#protocol-and-compatibility-decision)
and [supported public operations](MCP_CONTRACT.md#tools-and-public-api-mapping).
The [consuming-project agent template](../cambam_builder/mcp_adapter/consumer_AGENTS.template.md)
owns the reusable collaboration workflow. Second-PC and remote transport support
remain bounded by the contract and current backlog.

## Boundary (resolved by the MCP contract)

Use the [document and revision contract](MCP_CONTRACT.md#documents-revisions-and-atomicity),
[workspace/interchange policy](MCP_CONTRACT.md#workspace-and-document-interchange-policy)
and [process ownership](MCP_CONTRACT.md#packaging-and-process-ownership).

## Delivery increments and session boundaries

These historical increments establish the evidence trail, not a new execution queue.

### 4a. Protocol, client and adapter contract

[Protocol decisions and compatibility](MCP_CONTRACT.md#protocol-and-compatibility-decision)
are authoritative. [Wire probes and requirement changes](REVIEW.md#mcp-protocol-and-adapter-contract---2026-09-10)
preserve the modern-only failure and backward-compatibility decision.

### 4b. Secure server and document foundation

[Document/revision rules](MCP_CONTRACT.md#documents-revisions-and-atomicity) and
[workspace policy](MCP_CONTRACT.md#workspace-and-document-interchange-policy)
own the implementation boundary; [foundation evidence](REVIEW.md#mcp-foundation-and-client-compatibility---2026-09-10)
owns state, path and actual-client checks.

### 4c. First authoring and round-trip vertical slice

[Supported tools](MCP_CONTRACT.md#tools-and-public-api-mapping) own the delivered API.
[The first slice evidence](REVIEW.md#mcp-first-authoring-and-round-trip-slice---2026-09-10)
records direct-framework parity and round-trip acceptance.

### 4d. Supported API breadth and resilience

Use [the supported operation mapping](MCP_CONTRACT.md#tools-and-public-api-mapping),
[MOP parity matrix](MCP_CONTRACT.md#frameworkmcp-mop-parity-matrix) and
[portable interchange policy](MCP_CONTRACT.md#workspace-and-document-interchange-policy).
Batch evidence begins with [geometry breadth](REVIEW.md#mcp-geometry-breadth-batch-1---2026-09-10)
and includes [portable documents](REVIEW.md#mcp-portable-document-interchange---2026-09-11)
and [cross-document copy/transfer](REVIEW.md#mcp-cross-document-copytransfer---2026-09-11).

### 4e. Installation, local client and user acceptance

[The completed acceptance record](REVIEW.md#mcp-4e-acceptance-completion---2026-09-21)
owns the named-agent and CamBam observations and their same-machine scope.
[Clean installation and rollback](DEVELOPMENT.md#clean-wheel-installation-and-rollback)
and [client/CamBam acceptance procedures](DEVELOPMENT.md#4e-opencode-and-cambam-acceptance)
own reusable instructions.

## First implementation proof and acceptance

See [the first complete parity slice](REVIEW.md#mcp-first-authoring-and-round-trip-slice---2026-09-10)
and [completed external acceptance](REVIEW.md#mcp-4e-acceptance-completion---2026-09-21).
Their distinct evidence boundaries must not be inferred from successful tool responses.

## Maintenance obligations during implementation

The [adapter contract](MCP_CONTRACT.md), [development checks](DEVELOPMENT.md#required-checks-by-change)
and [verification workflow](WORKFLOW.md#verification-and-handoff-checklist)
own ongoing maintenance. Historical implementation order and model recommendations
do not constrain future work.
