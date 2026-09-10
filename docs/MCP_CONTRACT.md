# Local MCP adapter contract

Contract version 1, decided 2026-09-10 for backlog 4a. This is the authoritative
implementation contract; all eight document and authoring tools are implemented.
Priority and delivery state live in [PROGRESS.md](PROGRESS.md), and increment boundaries in
[MCP_PLAN.md](MCP_PLAN.md#delivery-increments-and-session-boundaries).

## Protocol and compatibility decision

Use **MCP 2026-07-28 over stdio**, with required backward compatibility for
**2025-06-18 and 2025-11-25**, one client-launched local process per configured
workspace. This updates 4a's modern-only decision under the user's explicit
2026-09-10 requirement change. Use the official Python SDK **mcp 2.2.0**, pinned
exactly in the adapter extra. Client acceptance must record the actual protocol.

The normative [core metadata specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/index)
requires protocol version and client capabilities in each request's `params._meta`;
client identity is optional. The
[stdio binding](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
uses UTF-8 newline-delimited JSON-RPC, stdout exclusively for protocol messages,
stderr for diagnostics, and EOF for shutdown. Discovery is optional. No application
state may depend on initialization, connection identity or client metadata.

Use SDK-backed protocol selection and low-level registration to declare only
implemented capabilities. Modern requests require per-request version/capability
metadata before dispatch. Reject absent/invalid metadata before legacy
initialization with JSON-RPC invalid params. Accept a legacy `initialize` only for
2025-06-18 or 2025-11-25; the SDK owns handshake lifecycle and era-specific result
serialization. Once a connection selects an era it cannot switch eras. Reject
unsupported versions explicitly. Notifications use their normative rules.
This is an intentional compatibility surface, with the same tool arguments,
application result envelope and document service for every accepted version.
No document state depends on the negotiated transport connection.

Advertise only tools, with `listChanged=false`; no resources, prompts, roots,
sampling, elicitation, tasks, subscriptions, HTTP listener or legacy SSE endpoint.
Implement `server/discover`, `ping`, `tools/list`, `tools/call`, cancellation and
EOF through the SDK. Tool listings are deterministic, sorted by name. Modern
results use `ttlMs=0`, `cacheScope="private"` where cacheable, and
`resultType="complete"`. Legacy wire results follow their negotiated SDK schema.
Publish input/output JSON Schemas and structured results with a serialized JSON
text block, following the
[tools specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools).

| Component checked on 2026-09-10 | Evidence and decision |
| --- | --- |
| Python SDK 2.2.0, Python 3.13.9, Windows | Actual stdio call before discovery, listing and discovery passed. The [SDK v2 documentation](https://github.com/modelcontextprotocol/python-sdk) and [release metadata](https://pypi.org/project/mcp/2.2.0/) require Python >=3.10. |
| OpenCode 1.18.29 client backend | Installed binary sends `initialize` at `2025-11-25`; modern-only negative probe fails, legacy control connects and lists tools. This motivates required 2025-11-25 support; actual adapter/Desktop acceptance is separate. |
| OpenCode Desktop candidate, backend source 1.18.29 | [MCP backend](https://github.com/anomalyco/opencode/blob/v1.18.29/packages/opencode/src/mcp/index.ts) implements stdio and remote Streamable HTTP with SSE fallback; [dependency declaration](https://github.com/anomalyco/opencode/blob/v1.18.29/packages/opencode/package.json) pins TS SDK 1.29.0. Desktop compatibility is inferred from that backend, not a tested GUI version; no GUI acceptance is claimed. |

The user's preferred replacement candidate is Codex. Installed Codex 0.154.0
also sends `initialize` at `2025-06-18` over stdio, including with
`features.mcp_2026_07_28=true` and with the app-server runtime explicitly accepting
that feature's enablement. The modern server rejects it; a legacy control connects
and lists tools. Thus generic Codex MCP support and the existence of an experimental
flag do not establish modern stdio conformance. Codex 0.154.0 is now the selected
client on the 2025-06-18 compatibility path: all five document tools passed through
its CLI client, including save/reopen and both closes. No modern desktop conformance or
second-PC acceptance is claimed. See the dated
[client survey and wire evidence](REVIEW.md#mcp-foundation-and-client-compatibility---2026-09-10).
Reprobe a new Codex/OpenCode version or another named client when its
source/release claims modern per-request stdio support.
Use the SDK's existing compatibility machinery; do not fork a client or build a
separate application service for older protocols.

## Packaging and process ownership

`cambam_builder.mcp_adapter` is explicitly included in setuptools'
package list. Geometry/XML behavior stays in the framework. The optional extra is
`mcp = ["mcp==2.2.0; python_version >= '3.10'"]`; base library users retain Python
>=3.9 and their existing dependency surface. The launcher must clearly reject
Python 3.9 and missing extras before importing the SDK. Never silently run with
missing dependencies on 3.9. Revisit the pin only with protocol/adapter checks.

Declare `cambam-mcp = "cambam_builder.mcp_adapter.__main__:main"`, with equivalent
`python -m cambam_builder.mcp_adapter`. Use argparse and SDK dependencies; no new
web framework, database, plugin system or automatic updater. Planned commands:

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe '.[mcp]'
.venv/Scripts/python.exe -m cambam_builder.mcp_adapter --workspace D:/CAD/AgentWork
```

These commands launch the implemented foundation. `--workspace`
is mandatory, absolute and already exists; no implicit current-directory access.
Client configuration uses the absolute environment Python executable, `-m`, module
name, `--workspace`, absolute directory as separate command-array elements. The
client owns start/stop; closing stdin exits promptly. Interactive local launch may
use Ctrl+C. Log no file contents, CAD descriptions or credentials. Imported labels
are escaped data in results, never server instructions. No model/network calls.

## Documents, revisions and atomicity

Use explicit **in-memory document handles**. `.cb` files are durable user artifacts;
they are not a hidden state database. A handle is `<boot UUID>:<document UUID>`.
Each call carries `workspace_id`, the canonical configured root's SHA-256 digest,
and the handle where needed. Create/open accept that exact workspace ID. The
workspace ID identifies a root; it is not a
credential. No process-global active document exists.

Compute that digest over UTF-8 `os.path.normcase(str(Path(root).resolve()))` on
the server OS. Discovery metadata and startup stderr publish the value; it
must be stable across process restarts for the same canonical root on that OS.

The SDK owns the normative discovery envelope. Add exactly
`result._meta["cambam-builder/workspace"] = {schema_version: 1,
workspace_id: Workspace, boot_id: UUID}`; its value validates against
`$defs.WorkspaceBootstrap` in the schema. Discovery instructions also explain
that tool calls require this workspace ID. Startup writes one stderr line
`CAMBAM_MCP_WORKSPACE ` followed by compact JSON of that same object. The client
can use either source; direct tool calls need no prior discovery if the ID is
already known. `_meta["io.modelcontextprotocol/serverInfo"]` remains SDK-owned.
Legacy initialization also publishes the bootstrap and includes the actual
workspace ID in server instructions, so ordinary clients can discover it.

Create/open start at integer revision 0. All document mutations, save and close
require `expected_revision`. Serialize operations with a per-document lock; check
revision inside the lock. A successful geometry/MOP edit increments once, even a
zero translation. Inspect and save do not increment; close invalidates the handle.
Two edits using the same revision cannot both succeed. Inspect takes the same
lock and returns one complete snapshot. Save holds the document lock from revision
check through clone, filesystem publication and ledger completion; its response
revision is exactly the revision represented by the saved artifact. An edit racing
save waits or wins the lock first, in which case the stale save fails. Never expose
partially updated entities.

A separate registry/capacity lock reserves a live-document slot for create/open
before expensive staging. Reservations count toward 16; failed/canceled staging
releases them. Publish the handle or close it under that lock. At 15 documents,
two concurrent opens cannot both succeed. Never hold the registry lock during
file I/O or while waiting for a document lock.

Stage edits on an independent full-project clone, validate, then publish the clone
and increment together. The owning framework must provide a public clone operation
`CamBamProject.clone()` that rebinds project references; do not copy private registries in handlers or use
pickle/XML as an internal transaction representation. Saving likewise uses a clone
because the writer updates primitive output precision. Exceptions or a framework
`None` failure discard staging and leave the live revision, relationships and files
unchanged. A multi-operation transaction tool is outside the initial surface.

All state-changing calls, including create/open/save/close, require caller-generated
UUID `request_id`, distinct from the JSON-RPC ID. Keep a process-lifetime request
ledger keyed by `(workspace_id, request_id)`. Under a ledger lock reserve the key
before execution, then release the ledger lock before waiting for a document lock
or running framework/I/O work. Compare canonical validated tool name/arguments, with defaults
expanded and excluding `request_id`. Identical retries join an in-flight operation
or return its original completed result **before** stale-revision/closed-handle
checks. Different arguments with a reused key return `REQUEST_ID_CONFLICT`.
Cache terminal application failures too; after changing a failed operation use a
new key. `replayed=true` is the only response field allowed to differ on replay.

Ledger states are `reserved -> completed`; all waiters share that completion.
After reservation, cancellation before publication discards staging, releases any
capacity reservation and completes the ledger with `REQUEST_CANCELLED`. Suppress
the original canceled wire response; retrying its key retrieves that terminal
failure, and a new key is required to try the operation again. A request canceled
before reservation has no record and no effects. Shield publication and ledger
completion together from cancellation. Cancellation after publication cannot
undo a mutation or save; retain success and suppress the canceled wire response.
Retry with the same key to recover it. Stdio reconnect within the
same process does not select/reset a document. Process restart creates a new boot
UUID, invalidates every old handle and loses unsaved edits and the retry ledger.
An old-boot handle returns `DOCUMENT_EXPIRED`, not a newly opened file. There is no
cross-restart exactly-once claim for create/open; clients must start a new workflow.
For interrupted saves inspect/open the intended path and compare its byte hash;
never blindly repeat an uncertain save to a different name after restart.

Initial hard limits: 16 live documents, 10 MiB source XML or saved XML, 10,000
primitives and 1,000 MOPs per document. At 10,000 reserved/completed **regular**
ledger entries, reject new create/open/edit requests before reservation with
`LIMIT_EXCEEDED`. This admission refusal is not cached and has no effects.
Existing-key retries and new save/close requests remain allowed so users can save
and drain live documents before restart. Save/close records are additional and
never evicted; this threshold is not a hard bound on total ledger memory. Normal
terminal failures consume their reserved regular slot. Close frees document
capacity, not ledger entries. A restart clears volatile capacity. Inspection is
paginated as defined below. Limit changes require schema
and behavior tests, not new dependencies.

## Workspace, import and save policy

All file arguments are workspace-relative `/`-separated paths ending exactly in
`.cb` (case-insensitive). Reject absolute, drive-relative, UNC/device, colon/ADS,
NUL, `.`/`..`, empty components, backslashes, Windows reserved device names,
and components ending in a dot or space. Do not expand environment variables,
home markers or URLs. Existing ancestors must be ordinary directories inside the
resolved root; reject symlinks/junctions/reparse points on any descendant component
and non-regular input files. Do not create parent directories implicitly. Reject
linked input files with more than one hard link. Validate canonical containment
using path components and OS case rules, never a string prefix comparison.

The configured root is a trusted directory controlled by the local user. Concurrent
hostile filesystem mutation by another local principal is not supported. Use
no-follow/handle validation where available and recheck ancestors at I/O; do not
claim this policy is an OS sandbox against a same-user process swapping paths.
No extra roots from the client may expand access. Path failures must not disclose
outside-workspace paths or file existence.

Open reads a bounded byte snapshot, calculates SHA-256, and uses a public
`read_cambam_bytes(data: bytes, *, source_name: str = "", strict: bool = True)`
entry point added to the reader. It returns a project or raises `ValueError` with
bounded diagnostic context; existing `read_cambam_file` behavior remains compatible.
Reject DTD/entity declarations and unsafe XML before parsing; enforce this
in the reader's owning contract. Strict import must reject unsupported MOP
types instead of allowing their current warning-and-skip behavior. Reader failures
become `IMPORT_FAILED` without publishing a document. Unknown XML fields still have
no blanket preservation guarantee: return `INTERCHANGE_LIMITED` on every open/save.
Preserve supported native parameter templates through public framework behavior.
Do not offer arbitrary XML patches, raw XML results or pickle loading.

**Save always creates a new file; overwrite is unsupported in version 1.** This
protects source files whose unsupported metadata may not survive interchange.
Existing destinations, including the originally opened file, return `PATH_EXISTS`.
There is no overwrite flag. Save-as uses the same tool with a different path and
leaves the handle/source association unchanged. Return the actual absolute path,
relative path, bytes and SHA-256 only after publication succeeds.

Do not call `project.save(destination)` directly: its `os.replace` semantics can
overwrite a racing destination. Save a clone to a unique sibling temporary `.cb`,
check size/hash and flush, then publish with an atomic **no-replace** filesystem
operation (e.g. hard-link temporary to destination, then unlink the task-owned
temporary). If unavailable, fail `IO_ERROR`; no unsafe overwrite fallback. A
destination appearing between validation and publication must survive unchanged.
Cleanup failure after publication is a successful save with `CLEANUP_PENDING`, not
a retryable save failure. Crash leftovers are task-prefixed temporary files, never
user filenames; cleanup may only target verified adapter-owned leftovers. Atomic
visibility is promised; survival of sudden power loss is not.

## Tool schema conventions

The [machine-readable schema](mcp_contract_v1.schema.json) owns field structure,
required fields and bounds. Its packaged copy `mcp_adapter/contract_v1.schema.json`
must remain identical (verified by regression); it makes installed wheels
self-contained. Each tool has `<name>_input` and `<name>_output` in
`$defs`. Bundle the shared definitions into each advertised input/output schema.
The tables below map those schemas to framework behavior. Implement strict typed
models; do not coerce
strings to numbers, booleans to numbers, or ignore unknown properties. Every field
is required unless marked `?`; optional fields have the stated default. `null` is
permitted only where explicitly stated. Reject NaN/infinity and oversized messages
(1 MiB per JSON-RPC input line). All UUIDs use canonical lowercase hyphenated text.

| Type | Definition |
| --- | --- |
| `Workspace` | 64 lowercase hexadecimal characters; must match configured root digest |
| `Handle` | Two UUIDs separated by `:`; opaque to callers |
| `Revision` | Integer 0 through 2^53-1; refuse overflow |
| `Name` | String 1..128 characters, no control characters; explicit user identifier, unique across project identifiers |
| `Path` | String 1..1024 characters satisfying the workspace path policy |
| `Number` | Finite JSON number; geometry/translation magnitude <=1e9 |
| `Positive` | Finite JSON number >0 and <=1e9 |
| `Read` | `{workspace_id: Workspace, document: Handle}` |
| `Write` | `Read` plus `{expected_revision: Revision, request_id: UUID}` |
| `New` | `{workspace_id: Workspace, request_id: UUID, units: "mm" | "in"}` |

`units` is the caller's explicit interpretation of raw framework coordinates;
the framework has no unit conversion or project-unit API. Preserve this assertion
in the handle and results. Open requires it again; never infer units from numbers.
The current writer does not persist this assertion as a verified CamBam units
setting. Emit `UNITS_ASSUMED` on create/open/save, stating the assertion and that
CamBam's units must match. 4e must verify this explicitly. No conversion, stock
material inference or feed calculation is provided by the adapter. Distances and
Z depths use asserted units, feeds units/minute, spindle speed revolutions/minute.
Coordinates use the framework's XY plane; positive Z is up, depth is an absolute
Z coordinate, and translation is relative to the existing local transform.

## Initial tools and public API mapping

| Tool | Closed input record | Public framework mapping / result data |
| --- | --- | --- |
| `document_create` | `New + {name: Name}` | `CBProject(name)`; empty document. Return `DocumentSummary`. |
| `document_open` | `New + {path: Path}` | Strict `read_cambam_bytes` of a bounded snapshot; return `DocumentSummary` with source path/hash. |
| `document_inspect` | `Read + {offset?: integer >=0 =0, limit?: integer 1..100 =100, expected_revision?: Revision}` | Public `list_*`, relationship getters and world-coordinate/bounds queries; return `InspectionPage`. If supplied, revision must match. |
| `geometry_add_rectangle` | `Write + {identifier: Name, layer: Name, x: Number, y: Number, width: Positive, height: Positive, z?: Number =0}` | `add_rect(layer, corner=(x,y), width=width, height=height, identifier=identifier, elevation=z)`; absent layer created through public API. Return primitive UUID and layer name. |
| `machining_add_profile` | `Write + {identifier: Name, part: Name, targets: UUID[1..100], side: "Inside" | "Outside", target_depth: Number, depth_increment: Positive, tool_diameter: Positive, cut_feedrate: Positive, plunge_feedrate: Positive, spindle_speed: integer 1..1000000, stock_surface?: Number =0, clearance_plane: Number, enabled?: boolean =true}` | `add_part` if absent, then `add_profile_mop(part, targets=..., identifier=identifier, name=identifier, profile_side=side, ...)`; return MOP UUID, part name and resolved target UUIDs. |
| `geometry_translate` | `Write + {entity_id: UUID, dx: Number, dy: Number}` | `translate_primitive(entity_id, dx, dy, bake=False)`; return primitive UUID. Initially accept only root Rects created/opened within the declared slice. |
| `document_save` | `Write + {path: Path}` | Clone + `save` + no-replace publication above; return `SavedArtifact`. |
| `document_close` | `Write` | Drop handle after revision check; return `{closed: true}`. Unsaved edits are discarded explicitly. |

Set tool annotations `openWorldHint=false` for all tools, `readOnlyHint=true` only
for inspect, and `idempotentHint=true` for inspect and ledger-protected writes.
Mark close/geometry mutations destructive; new-file save and create/open are
nondestructive. Hints describe behavior, not authorization or protocol enforcement.

All creation identifiers must be supplied; no generated human names. Rectangle
creation is axis-aligned with no parent/groups, identity XY matrix and zero local
Z offset. New layers use the public `add_layer` defaults (green, visible, unlocked,
alpha/pen width 1). Layers and parts are addressed by unique user names because
their UUIDs are not persisted in XML. Primitive/MOP IDs are `internal_id` UUIDs;
transient XML integer IDs are never tool arguments. Native files lacking framework
identity get new UUIDs on open; independent opens need not agree in that case.

Profile targets are unique UUIDs resolving to root Rects in the same document;
reject missing, duplicate, wrong-kind or transformed-out-of-scope targets before
mutation. Require `target_depth < stock_surface` and `clearance_plane > stock_surface`.
New parts use enabled=true, zero stock dimensions, empty material, origin (0,0),
and no spindle/tool override; no fabricated MDF/stock-size defaults. This is
unspecified stock, not a zero-thickness machining recommendation. Explicit profile
parameters bypass framework inferred tool/feed/depth defaults. Pin other profile
settings to the current public defaults, except `lead_in_type="None"` for this
slice: XY, EndMill, CW, ExactStop, Conventional, roughing clearance 0, stepover 0.4,
tool number 0, collision detection true, corner overcut false, final increment 0,
DepthFirst, no tabs, empty custom header/footer. Inspect returns these values too.
Unexposed parameters remain fixed; no unrestricted `**kwargs` input.

Inspection serializes copies, never mutable entity objects. `DocumentSummary` is
`{name, units, source: null | {path, sha256}, counts: {layers, parts, primitives,
mops}}`; counts are nonnegative integers. Source is informational and never means
save-in-place. `SavedArtifact` is `{path, absolute_path, sha256, bytes}` with positive
bytes. `InspectionPage` is `{summary: DocumentSummary, offset, next_offset: null |
integer, entities: EntityRecord[]}`. Enumerate layers in project order, parts in
project order, primitives by UUID, MOPs in part/MOP order; concatenate in that order
and paginate. Later pages should provide the first page's `expected_revision`.

`EntityRecord` is a discriminated union: layers `{kind:"layer", name}`; parts
`{kind:"part", name, enabled, stock_width, stock_height, stock_thickness,
stock_material}`; primitives `{kind:"primitive", id, identifier: string|null,
type, layer, parent: UUID|null, children: UUID[], world_xyz: array|null,
bounds: [xmin,ymin,xmax,ymax]|null}`; MOPs `{kind:"mop", id, identifier: string|null,
type, part, targets: UUID[], parameters}`. For a Rect, `world_xyz` contains four
XYZ corners in framework order. Profile `parameters` is a closed record containing
the named inputs (`side` becomes `profile_side`) and fixed settings above, using
public dataclass fields. Typed details cover the exact authoring slice only. For
other imported shapes/MOPs, including Profiles with inherited or out-of-slice
settings, return type/identity/relationships but null geometry or
empty parameters and `INSPECTION_UNSUPPORTED`; no invented geometry or calculated
effective inheritance values. Full multi-shape inspection belongs to 4d.

## Results and failures

Every completed tool result's `structuredContent` conforms to this closed envelope:

```text
{schema_version: 1, ok: boolean, workspace_id: Workspace,
 document: Handle|null, revision: Revision|null, request_id: UUID|null,
 replayed: boolean, data: ToolResultData|null,
 diagnostics: [{code: string, message: string}],
 error: null|{code: ErrorCode, message: string, field: string|null,
             current_revision: Revision|null}}
```

Success has `ok=true`, `error=null`, the tool's specific result data and
`isError=false`. Failure has `ok=false`, `data=null`, `error` set and `isError=true`.
Inspect uses null request ID; create/open failures use null handle/revision.
Close success reports its final revision. Other failed document calls report a
current revision only if the handle exists in this workspace. Diagnostic messages
are bounded to 1024 characters; no traces, raw imported XML or outside paths.
No partially successful mutation results exist in version 1.

`ErrorCode` is one of `INVALID_ARGUMENT`, `WORKSPACE_MISMATCH`, `DOCUMENT_NOT_FOUND`,
`DOCUMENT_EXPIRED`, `STALE_REVISION`, `REQUEST_ID_CONFLICT`, `ENTITY_NOT_FOUND`,
`IDENTIFIER_CONFLICT`, `UNSUPPORTED_OPERATION`, `PATH_INVALID`, `PATH_EXISTS`,
`IMPORT_FAILED`, `EXPORT_FAILED`, `LIMIT_EXCEEDED`, `IO_ERROR`, `INTERNAL_ERROR`.
`REQUEST_CANCELLED` is the terminal pre-publication cancellation code.
Unknown tools and malformed protocol envelopes are JSON-RPC errors. Known-tool
argument/domain failures use the application envelope, including failures rejected
by typed validation. Every envelope uses the server's configured workspace ID,
including missing/malformed/mismatched workspace argument failures. Invalid or
missing request IDs/handles are represented as null, never echoed into typed fields.
A ledger entry is reserved only once request identity is valid.
Never translate a framework log-and-None into a successful result.

## Acceptance and handoff to implementation

4b implements protocol/state/path contracts plus create/open/inspect/save/close.
It may add the minimal public framework clone and strict bounded XML import entry
points needed for that foundation; these must have their own focused regressions.
The generic inspection inventory may defer typed geometry/MOP details to 4c.
It must not advertise the three editing tools until their handlers exist.

4b acceptance: real subprocess discovery/list/direct-call without handshake;
legacy negotiation at both declared versions, modern metadata enforcement and
cross-era rejection; strict input/output schema tests; wrong
workspace/boot/handle/revision; concurrent same-revision edits; idempotent success,
failure, in-flight retry and closed-handle replay; limit exhaustion; cancellation
before/after commit; two creates/opens at 15 documents; exhausted regular ledger
still allowing save/close; concurrent save/edit artifact revision; workspace hash
equivalence for Windows case, separator, trailing-slash and resolved aliases;
EOF/restart; traversal, ADS, device, symlink/junction and
hard-link rejection; oversized/entity-declaration XML; unsupported-MOP rejection;
failed import/export and destination-race byte preservation. Save/reopen must
retain supported primitive/MOP IDs; layer/part UUID equality is not required.

4c acceptance uses asserted mm and fixed synthetic inputs: create `slice`, add
`outline` on `Geometry` at (0,0,0), width 20, height 10; add `profile` in `Part`,
outside, depth -1, increment 0.5, cutter 3, feeds 300/100, spindle 12000,
surface 0, clearance 5. Save `A.cb`; open it as a new handle; translate outline
by (5,2); save `B.cb`. Expect B world corners (5,2,0), (25,2,0), (25,12,0),
(5,12,0), allowing cyclic order, absolute coordinate error <=1e-9. Expect unchanged
primitive/MOP UUIDs, resolved targets, names, ordering and explicit machining
values; A bytes stay unchanged. Compare independently authored direct public API
results and XML semantics, not random UUIDs or whitespace across separate runs.
Create/open revision is 0; add Rect ->1, add Profile ->2; save stays 2; reopened
handle starts 0, translate ->1, second save stays 1. Retrying translate with its
original key never moves twice. Failed mutation preserves complete inspection.

No manual CamBam check adds evidence to 4a's architecture/protocol probes. 4c prepares
synthetic artifacts; 4e owns clean second-PC installation, actual named desktop
connection and CamBam units/geometry/property/toolpath acceptance. Current production
toolpath acceptance is not extended by this contract.

4d broadens only documented mappings: Circle/Arc/Pline/Points/Text/Region adders
and their public world queries; public parenting/groups and copy/transfer APIs;
public transform/bake methods; Pocket/Engrave/Drill and public MOP target/state
operations. Each family needs schema/parity/negative tests before advertisement.
Remote hosting, arbitrary Python/private registries, pickle, generic field setters,
deletion/batch edits, raw XML editing, overwrite and machine/G-code execution remain
excluded. Reopen volatile storage only for a demonstrated unsaved-recovery need;
reopen overwrite only with expected-file-hash concurrency and fidelity acceptance;
reopen additional protocol versions only with a concrete client need and wire
evidence; supported legacy versions remain required acceptance coverage.
