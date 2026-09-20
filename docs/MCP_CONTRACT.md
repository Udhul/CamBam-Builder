# Local MCP adapter contract

Contract version 1, decided 2026-09-10 for backlog 4a. This is the authoritative
implementation contract; all thirty-seven version 1 document, planning and authoring tools
are implemented, including cross-document copy/transfer between two open
documents (batch 5).
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

The initial deployment boundary is a same-machine client launching one local stdio
server process. This is the standard MCP stdio process model, not a TCP service:
there is no host, port or independently started server to register. Streamable HTTP,
HTTPS and remote-PC access are unimplemented, non-urgent future scope. Adding them
requires a separate bind/origin/authentication, lifecycle and HTTP-era conformance
contract; never expose the write-capable adapter on a LAN by changing only a bind
address.

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
use Ctrl+C. After the first accepted request, stderr emits one content-free
`CAMBAM_MCP_PROTOCOL` JSON record with the selected protocol version and
`legacy`/`modern` mode so named-client acceptance can record the actual path. Log no file contents, CAD descriptions or credentials. Imported labels
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

Create/open/import start at integer revision 0. All document mutations, save and close
require `expected_revision`. Serialize operations with a per-document lock; check
revision inside the lock. A successful geometry/MOP edit increments once, even a
zero translation. Inspect and save do not increment; close invalidates the handle.
Two edits using the same revision cannot both succeed. Inspect and export take the
same lock and return one complete snapshot. Export is read-only and does not increment
the revision. Save holds the document lock from revision
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

All state-changing calls, including create/open/import/save/close, require caller-generated
UUID `request_id`, distinct from the JSON-RPC ID. Keep a process-lifetime request
ledger keyed by `(workspace_id, request_id)`. Under a ledger lock reserve the key
before execution, then release the ledger lock before waiting for a document lock
or running framework/I/O work. Compare canonical validated tool name/arguments, with defaults
expanded and excluding `request_id`; replace imported XML content in the retained
signature with its UTF-8 byte count and SHA-256 digest. Identical retries join an in-flight operation
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

`document_list` is the recovery/discovery surface for a reconnect, a new agent
turn without retained handles, or ambiguous session state. It returns the current
`boot_id`, absolute `workspace_path`, and every live handle with its revision and
`DocumentSummary`. Its source
hash identifies the bytes originally opened/imported; it is not a hash of later
unsaved MCP mutations. An empty list or a changed boot is normal after restart.
No tool automatically selects the newest or similarly named document.

The durable client-local file remains authoritative across human and AI work. A
client that may have allowed a manual edit must reread and hash the file before a
later AI mutation. When its hash differs from the last opened/imported or successfully
written/exported artifact, a same-host client copies the file bytes under a fresh
name in `workspace_path`, verifies the hash, and opens that relative path with
`expected_sha256`; a cross-host client imports the complete current content with the
same hash guard. Either operation creates a new revision-0 handle. Never retarget or
patch the older handle implicitly, and never fall back to a similarly named older
workspace artifact after `CONTENT_MISMATCH`. The older handle may be closed after
the new snapshot succeeds. If the hash is
unchanged and the handle remains listed under the same boot, continue with its
current revision. On `STALE_REVISION`, inspect without `expected_revision`, review
whether the proposed change still applies, then retry with the returned revision
and a new `request_id`; the failed request ID retains its terminal stale result.
Never issue multiple mutations concurrently against the same document: every success
advances its revision, so await it and pass the returned revision to the next mutation.
Read-only calls and mutations against independent documents may run concurrently.
Track the last successfully written export revision as well as its hash. If the
durable file changed and the live handle has newer MCP mutations than that common
synchronization point, the versions have diverged: preserve both, export the MCP
candidate under a different client-local name, and ask the user which changes to
keep or merge. Never overwrite either side automatically.

## Cross-document copy and transfer

Batch 5 (2026-09-11) completes 4d with two-document subtree operations. The
`relationship_copy_tree_between` and `relationship_transfer_tree_between` tools
accept exactly two **distinct live handles** in this workspace and boot:
`source_document` and `target_document`, each with its own required revision
assertion (`source_expected_revision`, `target_expected_revision`). Any open
document works as either side (created, opened or imported); there is no special
staging document and no cross-document identity relationship beyond the one
operation. Equal source and target handles return `INVALID_ARGUMENT`.

Both document locks are acquired in canonical sorted-handle order and held from
the revision checks through publication; a concurrent close cannot remove a
locked document, and concurrent edits or saves on either side wait. After
acquiring both locks, both handles' liveness and both revisions are rechecked.
Copy stages on an independent full-project clone of the target and reads the
live source, because the public copy operation never mutates the source.
Transfer stages on independent clones of both documents and runs the public
`transfer_primitive_tree` between the clones, so the source removal and target
insertion are one transactional framework operation. Publication replaces the
staged projects and increments revisions inside a cancellation shield with no
await between assignments: a two-document publication cannot be observed
partially, and process restart still leaves only durable saved files. Copy
increments only the target revision; transfer increments both by exactly one.
A revision overflow on either document fails `LIMIT_EXCEEDED` before
publication.

Per-call failures never partially apply: a missing root (`ENTITY_NOT_FOUND`),
non-primitive root (`UNSUPPORTED_OPERATION`), framework collision/topology
rejection (`INVALID_ARGUMENT` with the bounded framework message, field `root`),
staged-target limit overflow (`LIMIT_EXCEEDED`) or any per-document
handle/revision failure leave both documents, both revisions and all files
unchanged. Envelope addressing: a failing **target** document check (expired,
not found, or stale target revision) references the target handle in
`document` and reports its current revision; every other outcome references
the source handle. Success sets `revision` to the source document's revision
after the operation, and `data` always carries both handles and both resulting
revisions:

```text
data = {mapping: {UUID: UUID}, source_document: Handle, target_document: Handle,
        source_revision: Revision, target_revision: Revision}
```

Ledger rules are unchanged: these are regular edit tools with one entry per
`(workspace_id, request_id)`; an identical retry joins or replays the original
two-document result before any stale/closed check, and a different-argument
reuse returns `REQUEST_ID_CONFLICT`. `document_close` on either side still
requires its own revision and frees only that document.

Initial hard limits: 16 live documents, 10 MiB imported/opened/exported/saved XML, 10,000
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

## Workspace and document interchange policy

`document_open` and `document_save` operate on the server workspace. Their file
arguments are workspace-relative `/`-separated paths ending exactly in
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

Open and content import read a bounded byte snapshot, calculate SHA-256, and use a public
`read_cambam_bytes(data: bytes, *, source_name: str = "", strict: bool = True)`
entry point added to the reader. It returns a project or raises `ValueError` with
bounded diagnostic context; existing `read_cambam_file` behavior remains compatible.
Reject DTD/entity declarations and unsafe XML before parsing; enforce this
in the reader's owning contract. Strict import must reject unsupported MOP
types instead of allowing their current warning-and-skip behavior. Reader failures
become `IMPORT_FAILED` without publishing a document. `document_import` accepts a
complete UTF-8 XML string supplied by the client plus a client-facing leaf filename;
it never interprets that name as a server path. It uses the same strict reader,
10 MiB encoded-byte limit, entity limits, capacity reservation and atomic handle
publication as open. Unknown XML fields still have no blanket preservation guarantee:
return `INTERCHANGE_LIMITED` on every open/import/save/export. Preserve supported
native parameter templates through public framework behavior. Serialized XML is a
complete interchange artifact only; do not offer arbitrary XML patches, treat XML
as instructions, or offer pickle loading.

`document_export` clones and serializes the revision under the document lock without
publishing a user-visible server file. Return `{kind: "cambam_document", mime_type:
"application/xml", encoding: "utf-8", suggested_filename, sha256, bytes, content,
delivery: "inline_content_only", file_created: false}` plus an
`INLINE_ONLY_NO_FILE` diagnostic. These fields mean that no client or server
workspace file exists; `suggested_filename` is not a saved path.
The client should write the UTF-8 `content` unchanged to its requested local `.cb`
destination and may verify `sha256`; the server cannot write a client-local path.
Export is a read-only call without ledger retention, preventing a
process-lifetime ledger from retaining repeated 10 MiB artifacts. Inline delivery is
the version 1 compatibility baseline. MCP resources may be reconsidered in 4e only
after named clients prove that resource results remain accessible to their agents.
For compatibility with clients that attach one UUID to every tool invocation,
export, inspect, list and the depth planner accept an optional valid `request_id`,
ignore it, return `request_id:null`, and never reserve or replay a ledger entry.
Portable content exchange remains the cross-host baseline. When the client and server
share a filesystem, `document_save` may also create an exact-byte handoff artifact in
the server workspace. The client copies that returned `absolute_path` with a
deterministic binary operation to its requested local destination and verifies the
returned byte count and SHA-256. The live handle remains authoritative until that
destination changes; a later manual change follows the fresh staging/open flow above.
The server path is not itself the client-local
destination, and `document_save` never accepts or expands an arbitrary client path.
When the filesystems differ, use `document_export` and write its inline content.
Every successful `document_save` includes a `SERVER_WORKSPACE_ARTIFACT` diagnostic
that states this boundary and fallback. Its structured delivery fields are
`delivery: "server_workspace_handoff"`, `workspace_file_created: true` and
`client_file_created: false`; a client-local task is not delivered until the client
copies and verifies that artifact at its intended destination.

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

The [machine-readable schema](../cambam_builder/mcp_adapter/contract_v1.schema.json)
owns field structure, required fields and bounds. It lives in the adapter package so
installed wheels are self-contained; there is no separately maintained documentation
copy. Each tool has `<name>_input` and `<name>_output` in
`$defs`. Bundle the shared definitions into each advertised input/output schema.
The tables below map those schemas to framework behavior. Implement strict typed
models; do not coerce
strings to numbers, booleans to numbers, or ignore unknown properties. Every field
is required unless marked `?`; optional fields have the stated default. `null` is
permitted only where explicitly stated. Reject NaN/infinity and oversized messages
(32 MiB per JSON-RPC input line, allowing a 10 MiB UTF-8 XML payload plus JSON
escaping/envelope overhead). All UUIDs use canonical lowercase hyphenated text.

| Type | Definition |
| --- | --- |
| `Workspace` | 64 lowercase hexadecimal characters; must match configured root digest |
| `Handle` | Two UUIDs separated by `:`; opaque to callers |
| `Revision` | Integer 0 through 2^53-1; refuse overflow |
| `Name` | String 1..128 characters, no control characters; explicit user identifier, unique across project identifiers |
| `Path` | String 1..1024 characters satisfying the workspace path policy |
| `Filename` | Client-facing leaf filename ending `.cb`, 4..255 characters, with no path separators or control characters |
| `XmlContent` | Complete UTF-8 XML text; encoded bytes, not character count, are limited to 10 MiB |
| `TextContent` | String 1..1024 characters with at least one non-whitespace character and no forbidden control characters; its regex stays within the portable no-lookaround subset used by OpenAI-backed tool-schema compilers |
| `Number` | Finite JSON number; geometry/translation magnitude <=1e9 |
| `Positive` | Finite JSON number >0 and <=1e9 |
| `NonNegative` | Finite JSON number >=0 and <=1e9 |
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
Circle `x,y` inputs are absolute center coordinates. Pline vertices are absolute
coordinates in traversal order. A bulge on vertex i curves its directed segment to
vertex i+1 (or the first vertex for the last segment of a closed Pline): positive
bulge is a counter-clockwise sweep lying to the right of that directed chord,
negative lies to its left, and `abs(bulge)=tan(abs(sweep)/4)`. Direction mnemonics
alone are insufficient for semantic "inward"/"outward" requests because contour
winding changes which chord side contains the interior.

For computed or multi-point geometry, the caller must compare the creation result
and `document_inspect` data against requested absolute bounds, center, dimensions,
symmetry and containment before attaching MOPs or exporting. Clarified requirements
invalidate prior calculations: recompute them rather than changing only one field.
The adapter validates representability and topology where declared; it cannot infer
the intended shape from valid but incorrect coordinates.

MOPs are appended to their Part in tool-call order, and inspection/export preserve
that order. The caller must plan the complete sequence before adding operations:
normally machine enclosed/internal features and non-releasing details before a
through-cut Outside Profile that releases their containing part. Not every Outside
Profile is a cutout, so unclear stock thickness, target depth or holding intent must
be resolved with the user. Contract v1 has no MOP reorder tool; a wrong order must
currently be rebuilt. Reopen a dedicated reorder increment using the public
`assign_mop_to_part` operation if named-client acceptance still misorders operations
or demonstrates a practical need to repair an existing sequence.

## CamBam value semantics and relative coordinates

The adapter exposes user-meaningful values, while the writer retains CamBam's native
encoding. These are not interchangeable coordinate systems:

- A Part's `machining_origin` is the drawing point CamBam treats as machine XY zero
  for that Part. `stock_offset` is a Part-local displacement from that origin, not an
  absolute drawing point. The stock lower-left **top** corner in drawing coordinates is
  therefore `(machining_origin_x + stock_offset_x, machining_origin_y +
  stock_offset_y, stock_surface)`. Inspection calls this derived value
  `stock_drawing_origin`; it is not another XML field. CamBam encodes the local stock
  box as `PMin=(offset_x, offset_y, surface-thickness)` and
  `PMax=(offset_x+width, offset_y+height, surface)`. For origin `(12,-7)`, offset
  `(12,-7)` and surface `4.5`, the reported drawing-space stock corner is
  `(24,-14,4.5)`.
- Part stock overrides Machining-level stock for that Part. The framework preserves
  an imported top-level `MachiningOptions` context but does not currently author or
  resolve its inherited stock/origin settings. New MCP Parts instead carry explicit
  Part values. Do not combine top-level and Part stock numerically; they are separate
  scopes shown by CamBam.
- A MOP `stock_surface` and `target_depth` are absolute Z planes. The requested cut
  depth is `stock_surface - target_depth`; `target_depth` is not a positive distance.
  `depth_increment` is a positive maximum step between successive Z levels, and
  `clearance_plane` is another absolute Z plane that must be above `stock_surface`.
- Nesting repeats every MOP in a Part; it does not copy source geometry. Multiple
  MOPs, including MOPs in different Parts, may target the same primitive. Grid
  `spacing` is clearance between the outermost generated toolpaths, so changing an
  enabled MOP or cutter diameter can change the nested placement. Nesting does not
  enlarge or replicate a Part stock definition. Stock itself is optional. When it is
  defined, checking that the complete nested toolpath envelope fits is good practice,
  not a framework validity requirement. PointList nesting references the
  XML ID of a real Points primitive and may share that Points object across Parts.
  Each point is a translation in CamBam's nesting coordinate system applied to the
  unnested Part toolpaths, not an absolute replacement for the source geometry's
  origin. With the default drawing origin, `effective path = unnested path + point
  vector`; moving both source geometry and the point compounds both translations.
  An alternate Part/Machining origin changes the nesting coordinate system in which
  the point is expressed. New records contain only fields relevant to their method;
  imported valid Manual/PointList placement data is preserve-only at the MCP boundary.
- On open Plines, Profile `Inside` selects the left side and `Outside` the right side
  relative to stored vertex traversal. Reversing the vertices reverses the physical
  side. Closed contours retain their usual interior/exterior meaning.
- Automatic tab count starts from perimeter divided by `tab_distance`, rounded and
  clamped between minimum and maximum; distance zero selects the minimum. Contours
  whose perimeter is below `size_threshold` receive no automatic tabs. `width` is
  the thinnest retained width, so CamBam's displayed tab/toolpath gap is wider due to
  cutter-radius compensation. `height` is measured upward from target depth. Square
  and Triangle retain material; Skip rapids over the interval and is intended for
  non-contact cutting such as plasma. `UseLeadIns` affects only Square tabs and only
  when the Profile has an active `LeadInMove`; because this MCP operation pins
  `lead_in_type=None`, it requires `tab_use_leadins=false`.
- All exposed automatic-tab controls are durable: method, width, height, minimum and
  maximum count, distance, size threshold, lead-in flag (currently constrained false)
  and Square/Triangle/Skip style. CamBam changes a tab set to Manual when positions
  are moved or explicitly added/removed. Imported native Manual tabs are preserved,
  but fresh core and MCP authoring reject Manual until their native point encoding is
  modeled; incomplete placeholder XML is never emitted.
- The native enum token is `VCutter` (`ToolProfiles.VCutter`), although CamBam's UI
  displays “V-Cutter” and some prose documentation says “Vcutter”. An Engrave MOP
  still follows the selected curve or Text outlines at its ordinary depth settings;
  both `EndMill` and `VCutter` are supported tool-shape selections, and neither turns
  that path-following operation into filled-area or skeleton/width-driven V-carving.
  That remains the separately planned V-carving capability.
- `roughing_clearance` is a signed horizontal allowance exposed for Profile, Pocket,
  Engrave and SpiralMill Drill. Zero uses the normal compensated Profile/Pocket
  boundary or exact Engrave line placement; positive leaves material and negative
  overcuts. Fresh MCP CannedCycle authoring requires zero; an imported native
  CannedCycle value remains core-readable, preserved and structured-inspectable.
- For `SpiralMill_CW`/`SpiralMill_CCW`, let `H` be `hole_diameter`,
  `R` signed `roughing_clearance`, and `T` effective tool diameter. The resulting cut
  diameter is `H - 2R`; cutter fit requires `H - 2R > T`, equivalently
  `H > T + 2R`. Thus `H=6`, `T=4`, `R=-1` produces diameter `8`. Fresh core export
  enforces this when both diameters resolve explicitly. MCP Point targets
  require explicit `H`; all-Circle selections may omit it for CamBam Auto sizing,
  and the adapter validates the resolved world diameter of every selected Circle. This exact
  relationship is user-validated
  CamBam Plus 1.0 behavior supplementing the vendor's documented signed clearance
  and spiral-drill support.
- The core model can preserve and directly author a nonempty `CustomScript` Drill,
  but the MCP boundary intentionally exposes only CannedCycle and the two SpiralMill
  methods. CustomScript is literal controller/postprocessor-sensitive G-code, has no
  native execution acceptance in this repository, and is excluded from MCP authoring
  and structured MOP inspection rather than presented as a safe declarative machining
  option. Imported CustomScript records still round-trip through core XML.
- Engrave inspection retains `roughing_finishing` for framework/API parity, and MCP
  authoring pins it to `Roughing`. CamBam documents this published property as
  effective only for Lathe and 3D Profile, so neither value implies a different
  Engrave toolpath through this adapter.

## Framework/MCP MOP parity matrix

This matrix is the authoritative exposure map for the four modeled MOP families.
Field meanings and XML state/omission rules remain in the
[core MOP parameter interchange contract](structure_spec.md#mop-parameter-interchange-contract).
Here **A/R/E/P** means direct-core fresh author, read, edit and imported-template
preserve. All currently modeled fields have A/R/E/P support, subject to the documented
mode dependencies and export-time validation. MCP **inspect** below means a typed
`parameters` record, not mere identity/target visibility. MCP **mutate** is `No` for
every parameter row: contract v1 can replace explicit targets but cannot patch a MOP
parameter.

The rows account for all 18 common, 18 Profile, 12 Pocket, 3 Engrave and 9 Drill
executable field-policy slots. `enabled` is listed with the common authoring controls
but is an XML attribute rather than one of those 60 policy slots.

| Capability / modeled fields | Core A/R/E/P | MCP author | MCP inspect | MCP mutate | Classification |
| --- | --- | --- | --- | --- | --- |
| Common explicit controls: `enabled`, `target_depth`, `depth_increment`, `stock_surface`, `roughing_clearance`, `clearance_plane`, `spindle_speed`, `tool_diameter`, `plunge_feedrate`, `cut_feedrate` | Yes | Explicit for all four families; CannedCycle authoring alone requires zero roughing clearance | Every present modeled raw value is returned with native state and applicability, independently of authoring pins and targets | No | Supported; required safety/cutting inputs stay explicit for authoring |
| Common pins: `spindle_direction`, `velocity_mode`, `work_plane`, `optimisation_mode`, `tool_number`, `max_crossover_distance`, `custom_mop_header`, `custom_mop_footer` | Yes | `CW`, `ExactStop`, `XY`, `Standard`, `0`, `0.7`, empty, empty | Present alternate values and nonempty header/footer text are returned; omitted fields remain absent rather than becoming constructor defaults | No | Deliberate bounded-authoring policy, not a core limitation |
| Common `tool_profile` | Yes | Profile/Pocket `EndMill`; Engrave `EndMill` or `VCutter`; Drill `Drill`, `EndMill` or `Unspecified` with method-aware default | Any present modeled raw string is returned; this does not expand authoring support | No | Evidence-backed family authoring restriction |
| Profile inputs: `profile_side`, `corner_overcut`, `tab_method`, `tab_width`, `tab_height`, `tab_min_tabs`, `tab_max_tabs`, `tab_distance`, `tab_size_threshold`, `tab_use_leadins`, `tab_style` | Yes | Explicit; method dependencies apply and `tab_use_leadins` is currently constrained false | Present modeled values are returned with applicability; Manual point collections remain opaque | No | Supported; open-Pline side remains vertex-order-relative |
| Profile pins: `stepover`, `milling_direction`, `collision_detection`, `lead_in_type`, `lead_in_spiral_angle`, `final_depth_increment`, `cut_ordering` | Yes | `0.4`, Conventional, true, None, inactive `30`, `0`, DepthFirst | Present modeled values are returned; applicability distinguishes the active lead mode. Unsupported lead modes remain opaque | No | Spiral lead-in is core-authorable but intentionally hidden at MCP authoring pending broader lead controls |
| Pocket subtype: `stepover`, `stepover_feedrate`, `milling_direction`, `collision_detection`, `lead_in_type`, `lead_in_spiral_angle`, `final_depth_increment`, `cut_ordering`, `region_fill_style`, `finish_stepover`, `finish_stepover_at_target_depth`, `roughing_finishing` | Yes | All pinned to the canonical Spiral/InsideOutsideOffsets/Roughing slice described below | Present modeled values and supported lead dependencies are returned with state/applicability | No | Core breadth is intentionally narrowed, not unsupported |
| Engrave subtype: `roughing_finishing`, `final_depth_increment`, `cut_ordering` | Yes | Roughing, `0`, DepthFirst | Present alternate modeled values are returned | No | Compatibility metadata retained; no Engrave path effect is promised for roughing/finishing |
| Drill method fields: `drilling_method`, `peck_distance`, `retract_height`, `dwell`, `hole_diameter`, `drill_lead_out`, `spiral_flat_base`, `lead_out_length` | Yes | CannedCycle and SpiralMill CW/CCW with method-dependent inputs; Auto diameter only for all-Circle targets | Present raw discriminator/dependent values are returned; applicability identifies the active supported method | No | Supported, method-aware surface |
| Drill `custom_script` | Yes, nonempty CustomScript only | No | No typed parameters | No | Intentional exclusion: literal controller/postprocessor-sensitive G-code lacks native execution acceptance |

Imported values outside an MCP authoring pin remain independently inspectable.
`parameters` contains modeled typed values whose native nodes are present;
`parameter_metadata` reports `native_state` (`Value`, `Default`, `Unspecified`,
`Omitted`, or the `Enabled` attribute) and dependency `applicable` for each safe
modeled field. A `Default` value is cached native text, never an evaluated CAM-style
value, and an omitted native node does not expose its constructor fallback.
`unsupported_fields` names maximal opaque native paths; the page diagnostic repeats
those paths without blanking unrelated parameters. Literal CustomScript, Manual tab
point collections, unsupported lead modes and unknown extensions remain opaque.

| Target / native capability | Core | MCP author | MCP inspect | MCP mutate | Classification |
| --- | --- | --- | --- | --- | --- |
| Explicit targets | Any registered Primitive, including an empty selection; this is syntactic encoding, not per-kind machining validation | 1..100 unique IDs. Profile: Rect/Circle/open-or-closed Pline/Text/Region; Pocket: Rect/Circle/closed Pline/Text/Region; Engrave: Rect/Circle/Arc/open-or-closed Pline/Text; Drill: Points/Circle. The separate target-slice checks also apply | IDs and independently safe typed parameters remain visible for populated or empty selections | Atomic nonempty replacement with the same checks | MCP restrictions are deliberate safety policy |
| Live `target_group` source | Author/read/edit in memory and same-version pickle; XML intentionally materializes a target snapshot | No | `target_group` names the live source and `targets` shows its current resolved members; parameters remain inspectable | No | Intent is visible in memory but not durable CamBam XML |
| Group membership on an explicit target | Metadata does not change the core target or geometry | Accepted for otherwise eligible targets | Typed geometry and MOP parameters remain visible | Accepted for otherwise eligible targets | Group names are selection metadata, not a machining-coordinate relationship |
| Supported transforms, parent/child relationships or local Z | Core stores and exports the relationship/transform | Only zero-local-Z similarity targets with no parent or children | Outside that slice is identity-only/unsupported | Rejected | Keep bounded until machining-coordinate semantics are evidenced |
| Native `Default` state | Top-level state author/edit plus read/preserve; nested native state is preserved | Required MCP values are explicit; Spiral Auto diameter is the single active Default | Cached typed value and `native_state=Default` are returned without claiming an effective style value; container state governs nested leaves | No | Effective style values are unknowable without the external style library |
| `Style`, `StartPoint`, `SpindleRange`, independent lead-out, unsupported lead fields/modes, Manual tab points and unknown extensions | No semantic model; preserved inside an otherwise supported imported MOP template | No | Named in `unsupported_fields`; opaque content is not returned | No | Preserve-only; do not invent nominal parity |
| Unknown Drill methods | Preserve-only; switching is rejected | No | Common fields and the raw method discriminator remain visible; unknown dependents are named and opaque | No | Preserve-only |
| Unsupported MOP kinds such as 3D Profile, Lathe or plugins | Not modeled; permissive import skips them and strict import rejects them | No | No | No | Intentionally absent from both layers; unlike unknown fields, these are not round-trip-preserved |

No MCP authoring promise was found that the core writer cannot faithfully encode.
The checked source-level classification is guarded by
`tests/test_mcp_mop_parity.py`, which compares every modeled dataclass field with the
closed author-input and inspection schemas. Preservation-aware typed MOP inspection
and group-neutral explicit-target eligibility close the identified compositional
gaps. Parameter patching remains dependent on a
validated core patch contract and the native evidence requested by backlog 8c;
CustomScript literals and unmodeled native content remain intentionally opaque.

These rules follow the CamBam Plus 1.0 documentation for [CAM Parts](https://www.cambam.info/doc/plus/cam/CAMPart.htm),
[Machining Options](https://www.cambam.info/doc/plus/cam/MachiningOptions.htm),
[nesting](https://www.cambam.info/doc/1.0/cam/nesting.html), [holding tabs](https://www.cambam.info/doc/1.0/cam/holding-tabs.html),
and [Engrave](https://www.cambam.info/doc/plus/cam/Engrave.htm), plus the recorded
CamBam Plus 1.0 native-file checks in `docs/PROGRESS.md`.

## Tools and public API mapping

| Tool | Closed input record | Public framework mapping / result data |
| --- | --- | --- |
| `document_create` | `New + {name: Name}` | `CBProject(name)`; empty document. Return `DocumentSummary`. |
| `document_import` | `New + {source_name: Filename, content: XmlContent, expected_sha256?: Sha256}` | If supplied, require the complete UTF-8 content to match `expected_sha256` before parsing. Strict `read_cambam_bytes(content.encode("utf-8"))`; publish a new revision-0 volatile handle and return `DocumentSummary` with client-source name/hash/bytes. |
| `document_open` | `New + {path: Path, expected_sha256?: Sha256}` | Read one bounded workspace snapshot; if supplied, reject a hash mismatch before strict parsing or publication. Return a new revision-0 handle and `DocumentSummary` with source path/hash. |
| `document_inspect` | `Read + {offset?: integer >=0 =0, limit?: integer 1..100 =100, expected_revision?: Revision}` | Public `list_*`, relationship getters and world-coordinate/bounds queries; return `InspectionPage`. If supplied, revision must match. |
| `document_list` | `{workspace_id: Workspace, request_id?: UUID}` | Return the current boot ID, absolute shared `workspace_path`, and all live handles with revision and summary. The optional request ID is ignored. This discovers volatile snapshots and the same-host staging boundary; it does not inspect client-local files or detect manual changes itself. |
| `geometry_add_rectangle` | `Write + {identifier: Name, layer: Name, x: Number, y: Number, width: Positive, height: Positive, z?: Number =0}` | `add_rect(layer, corner=(x,y), width=width, height=height, identifier=identifier, elevation=z)`; absent layer created through public API. Return primitive UUID and layer name. |
| `geometry_add_circle` | `Write + {identifier: Name, layer: Name, x: Number, y: Number, diameter: Positive, z?: Number =0}` | `add_circle(layer, center=(x,y), diameter=diameter, identifier=identifier, elevation=z)`; absent layer created through public API. Return primitive UUID, layer name and typed Circle geometry so the caller can verify its absolute center and bounds immediately. |
| `geometry_add_arc` | `Write + {identifier: Name, layer: Name, x: Number, y: Number, radius: Positive, start_angle: Number, extent_angle: Number, z?: Number =0}` | `add_arc(layer, center=(x,y), radius=radius, start_angle=start_angle, extent_angle=extent_angle, identifier=identifier, elevation=z)`; degrees, CCW-positive signed sweep; return primitive UUID and layer name. |
| `geometry_add_pline` | `Write + {identifier: Name, layer: Name, points: VertexPoint[2..10000], closed?: boolean =false}` where `VertexPoint` is `{x: Number, y: Number, z?: Number =0, bulge?: Number =0}` | `add_pline(layer, points=[Vertex(...)], closed=closed, identifier=identifier)`; the bulge stored on one vertex curves the segment that starts there. Return primitive UUID, layer name and typed Pline geometry with world vertices, bulges and curved bounds. |
| `geometry_add_points` | `Write + {identifier: Name, layer: Name, points: PlainPoint[1..10000]}` where `PlainPoint` is `{x: Number, y: Number, z?: Number =0}` | `add_points(layer, points=[Vertex(...)], identifier=identifier)`; bulge input is rejected by the schema; return primitive UUID and layer name. |
| `geometry_add_text` | `Write + {identifier: Name, layer: Name, text: TextContent 1..1024 non-whitespace-only, x: Number, y: Number, height?: Positive =10, font?: FontName ="Arial", style?: FontStyle ="", line_spacing?: Positive =1, align_horizontal?: "left"\|"center"\|"right" ="center", align_vertical?: "top"\|"center"\|"bottom" ="center", z?: Number =0}` | `add_text(layer, text, position=(x,y), height, font, style, line_spacing, align_horizontal, align_vertical, identifier, elevation=z)`; return primitive UUID and layer name. The optional unused `xml_p2_*` interchange fields are not authorable inputs. |
| `geometry_add_region` | `Write + {identifier: Name, layer: Name, outer: {points: VertexPoint[2..10000]}, holes?: {points: VertexPoint[2..10000]}[0..100] =[]}` | Contours become closed `Pline` records; `add_region(layer, outer_curve=..., hole_curves=..., identifier=identifier)`; XY topology (closed, simple, nonzero area, contained disjoint holes) is validated by the framework and topology failures return `INVALID_ARGUMENT` with the bounded framework message; return primitive UUID and layer name. |
| `geometry_replace_with_region` | `Write + {identifier: Name, outer_id: UUID, hole_ids?: UUID[0..100] =[]}` | Atomically convert the world geometry of existing supported root Rect, Circle or closed-Pline contours on one layer into one Region. Validate topology on a staged clone, remove the source primitives only on success, retarget explicit Profile/Pocket selections from any source to the new Region, and reject sources used by Engrave/Drill or carrying unsupported parent/child/group relationships. Return the new UUID, layer and ordered removed UUIDs. |
| `geometry_update_region` | `Write + {entity_id: UUID, outer: ContourInput, holes?: ContourInput[0..100] =[]}` | Atomically replace one supported root Region's absolute contours after topology validation. Preserve its UUID, identifier, layer and project-owned Profile/Pocket target relationships; return the UUID, layer and typed Region geometry. |
| `machining_add_profile` | `Write + {identifier: Name, part: Name, targets: UUID[1..100], side: "Inside" | "Outside", target_depth: Number, depth_increment: Positive, tool_diameter: Positive, cut_feedrate: Positive, plunge_feedrate: Positive, spindle_speed: integer 1..1000000, stock_surface?: Number =0, roughing_clearance?: Number =0, clearance_plane: Number, enabled?: boolean =true, corner_overcut?: boolean =false, tab_method?: None\|Automatic =None, tab_width?: Positive =6, tab_height?: Positive =1.5, tab_min_tabs?: integer =3, tab_max_tabs?: integer =3, tab_distance?: NonNegative =40, tab_size_threshold?: NonNegative =4, tab_use_leadins?: false =false, tab_style?: Square\|Triangle\|Skip =Square}` | `add_part` if absent, then `add_profile_mop(...)`; targets may include Text as well as the other supported contours. Signed roughing clearance adjusts the boundary allowance. Automatic holding tabs map to CamBam's bounded `HoldingTabs` record and require minimum <= maximum. Manual tab point placement is not exposed. `tab_use_leadins` is false because the operation also pins `lead_in_type=None`. Closed boundaries return `side_semantics=ClosedBoundary`; on an open Pline Inside is left and Outside right relative to traversal, with a directional diagnostic because reversing traversal swaps the physical side. CamBam `corner_overcut` adds an extra move into inside corners that otherwise would not be cut; it deliberately overcuts stock and is useful for fitted parts such as slot joints or inlays. Return MOP UUID, part name, echoed side/semantics and resolved targets. |
| `machining_add_pocket` | `Write + {identifier: Name, part: Name, targets: UUID[1..100], target_depth, depth_increment: Positive, tool_diameter: Positive, cut_feedrate: Positive, plunge_feedrate: Positive, spindle_speed: integer 1..1000000, stock_surface?: Number =0, roughing_clearance?: Number =0, clearance_plane: Number, enabled?: boolean =true}` | `add_pocket_mop(...)` with the pocket settings pinned in the schema record (stepover 0.4, `InsideOutsideOffsets` fill, Spiral lead-in, Roughing); targets may include Text, and signed roughing clearance adjusts the retained/overcut boundary allowance. Return MOP UUID, part name and resolved targets. |
| `machining_add_engrave` | Same closed record as Pocket (no side, no pocket-specific inputs) plus `tool_profile?: EndMill\|VCutter =EndMill` | `add_engrave_mop(...)` with Engrave settings pinned in the schema record (Roughing, final increment 0, DepthFirst) and the selected native tool-profile enum. At roughing clearance zero it follows the selected line placement; a signed nonzero value offsets that path. EndMill and VCutter do not fill Text/regions or request width/depth-varying V-carving. Return MOP UUID, part name and resolved targets. |
| `machining_add_drill` | Pocket-like common record plus `drilling_method?: CannedCycle\|SpiralMill_CW\|SpiralMill_CCW =CannedCycle`, optional cutter metadata, and method fields. CannedCycle: `peck_distance?: NonNegative =0`, `retract_height?: Number =5`, `dwell?: NonNegative =0`. SpiralMill: `hole_diameter?: Positive` (Circle-only Auto when omitted), `roughing_clearance?: Number =0`, `drill_lead_out?: boolean =false`, `spiral_flat_base?: boolean =true`, `lead_out_length?: Number =0`. | `add_drill_mop(...)` with method-aware validation. Point targets require explicit SpiralMill diameter; all-Circle targets may use Auto. Explicit spiral diameter must satisfy `hole_diameter - 2*roughing_clearance > tool_diameter`. A nonzero lead-out length requires lead-out enabled; a positive centerward length is capped at effective hole radius. Selected relevant parameters serialize as `Value`; fresh SpiralMill XML omits CannedCycle-only fields and unused CustomScript rather than emitting prompt-causing Default records. Imported native XML remains preserved. Omitted tool profile resolves to `Drill` for CannedCycle and `Unspecified` for SpiralMill. Return MOP UUID, part name and resolved targets. |
| `machining_calculate_depth_increment` | `Read + {units: "mm"|"in", stock_thickness: Positive, cut_through: Positive, exactly one of pass_count: integer 1..10000 or max_depth_increment: Positive, rounding_increment?: Positive}` | Pure planning calculation with no document handle or mutation. Upward rounding defaults to 0.1 mm or 0.001 in. Return the increment, actual clamped depths, nominal overshoot, final-pass depth/stock/cut-through, stock fraction and `recommendation_met`. A valid explicit constraint that misses the one-third recommendation is returned with diagnostics rather than rejected; the supplied maximum must already reflect material/tool safety. |
| `machining_configure_part` | `Write + {part: Name, enabled?, stock_width/height/thickness?: NonNegative, stock_material?: string <=128, stock_color?: Name, stock_offset_x/y?: Number, stock_surface?: Number, machining_origin_x/y?: Number, default_tool_diameter?: NonNegative|null, default_spindle_speed?: integer|null, nest_method?: None\|Grid\|IsoGrid, nest_rows/columns?: integer, nest_spacing?: NonNegative, grid_order?: eight directional orders, grid_alternate?: boolean}` | Patch an existing Part, preserving every omitted field and valid native placement data when its nesting method is unchanged; a new minimally specified Part starts with zero/unspecified stock and empty material. Stock definition is optional. The result reports both local stock offset and derived `stock_drawing_origin_*`. Grid/IsoGrid repeats the whole Part's MOP sequence over shared source geometry but does not enlarge defined stock; multiple copies plus nonzero XY stock return the non-blocking `NESTED_STOCK_FIT_ADVISORY`. Imported Manual/PointList nesting and placement metadata is inspectable and survives unrelated patches, but authoring it is outside this tool; switching methods discards placement fields belonging to the old method. `default_spindle_speed`, when supplied, is diagnosed as session-only framework context; durable MOPs carry their own spindle speed. |
| `machining_set_mop_targets` | `Write + {mop_id: UUID, targets: UUID[1..100]}` | Public `set_mop_targets`; atomically replaces the MOP's explicit target selection after the same per-kind target rules and slice checks; return MOP UUID and the project's UUID-sorted resolved targets. |
| `document_set_layer_properties` | `Write + {layer: Name, color?: Name, alpha?: 0..1, pen_width?: NonNegative, visible?: boolean, locked?: boolean}` | Create or update layer display properties. CamBam uses pen width zero as its default one-pixel display mode. These are presentation settings and do not affect machining geometry or MOP semantics. |
| `relationship_set_parent` | `Write + {entity_id: UUID, parent_id: UUID\|null}` | Public `link_primitive_parent`; null detaches. Local transforms are kept, so the world pose follows the new frame; self links and cycles return `INVALID_ARGUMENT`, missing/non-primitive entities `ENTITY_NOT_FOUND`/`UNSUPPORTED_OPERATION`. Return the child UUID and the resulting parent UUID or null. |
| `relationship_add_to_group` | `Write + {entity_id: UUID, group: Name}` | Public `add_primitive_to_group`; return the entity UUID and its sorted group names. |
| `relationship_remove_from_group` | `Write + {entity_id: UUID, group: Name}` | Public `remove_primitive_from_group`; return the entity UUID and its sorted remaining group names. |
| `relationship_copy_tree` | `Write + {root: UUID, include_mops?: boolean =false, identifier_map?: {Name: Name} ={}, group_map?: {Name: Name} ={}}` | Public `copy_primitive_tree` into the same document with `preserve_ids=False`; copies get fresh UUIDs. Included layers, groups and (with `include_mops`) their parts/MOPs must have unmapped names remapped via the maps or the framework's collision rejection returns `INVALID_ARGUMENT`; MOP selections must lie inside the subtree. Return the source-to-copy UUID mapping (including copied layers/parts). |
| `relationship_copy_tree_between` | `Write` (document/expected_revision become `source_document`/`source_expected_revision`) plus `{target_document: Handle, target_expected_revision: Revision, root: UUID, include_mops?: boolean =false, identifier_map?: {Name: Name} ={}, group_map?: {Name: Name} ={}}` | Public `copy_primitive_tree` from the source project into a staged clone of the target project with `preserve_ids=False`; both documents must be open and distinct. Same mapping/name rules as the same-document copy. Only the target revision increments. Return `{mapping, source_document, target_document, source_revision, target_revision}`. |
| `relationship_transfer_tree_between` | Same closed record as `relationship_copy_tree_between` | Public `transfer_primitive_tree` between staged clones of both documents: the source subtree is removed and both revisions increment together under one transactional publication. Return `{mapping, source_document, target_document, source_revision, target_revision}`. |
| `geometry_translate_z` | `Write + {entity_id: UUID, dz: Number}` | Public `translate_primitive_z(..., bake=True)`: one explicit stored-geometry Z shift that keeps world matrices; return the entity UUID. |
| `geometry_rotate` | `Write + {entity_id: UUID, angle_deg: Number, cx?: Number, cy?: Number}` (cx/cy together or absent) | Public `rotate_primitive_deg`; absent center uses the framework's geometric center. The world pose stays a similarity, so typed inspection remains available; return the entity UUID. |
| `geometry_scale` | `Write + {entity_id: UUID, factor: Positive, cx?: Number, cy?: Number}` (cx/cy together or absent) | Uniform `scale_primitive(factor, factor, ...)`; non-uniform scale is unsupported because it leaves the similarity slice; return the entity UUID. |
| `geometry_mirror` | `Write + {entity_id: UUID, axis: "x"\|"y", position?: Number}` | Public `mirror_primitive_x` (across y=position) or `mirror_primitive_y` (across x=position); absent position uses the geometric center. Bulged vertices flip sign under reflection; return the entity UUID. |
| `geometry_bake` | `Write + {entity_id: UUID}` | Public `bake_geometry()` on the staged primitive: folds the world transform into stored geometry and resets the matrix to identity. Non-axis-aligned Rects become closed Plines (reported `type`); Text bakes only translation/positive uniform scale and otherwise fails `UNSUPPORTED_OPERATION`; return the entity UUID and resulting type. |
| `geometry_translate` | `Write + {entity_id: UUID, dx: Number, dy: Number}` | `translate_primitive(entity_id, dx, dy, bake=False)`; return primitive UUID. Accepts root Rect/Circle/Arc/Pline/Points/Text/Region primitives inside the similarity slice: a finite non-degenerate XY similarity world matrix (translation, rotation, uniform scale, reflection), zero local Z offset, no parent/children/groups and valid positive geometry. |
| `document_export` | `{workspace_id: Workspace, document: Handle, expected_revision: Revision, suggested_filename: Filename, request_id?: UUID}` | Serialize a clone of the exact revision; return a complete `SerializedArtifact` with `delivery=inline_content_only`, `file_created=false`, and `INLINE_ONLY_NO_FILE` without publishing any file. The optional request ID is ignored. |
| `document_save` | `Write + {path: Path}` | Clone + `save` + no-replace publication above; return `SavedArtifact`. |
| `document_close` | `Write` | Drop handle after revision check; return `{closed: true}`. Unsaved edits are discarded explicitly. |

Set tool annotations `openWorldHint=false` for all tools, `readOnlyHint=true` only
for list/inspect/export and the depth planner, and `idempotentHint=true` for
read-only calls and ledger-protected writes.
Mark close/geometry mutations destructive; new-file save and create/open are
nondestructive. Hints describe behavior, not authorization or protocol enforcement.

All creation identifiers must be supplied; no generated human names. Creation
adds root primitives with no parent/groups, identity XY matrix and zero local Z
offset; Rect is axis-aligned, Circle/Arc use the framework center/parameter
fields, Pline/Points/Region contours store the given vertices verbatim, and
Text stores the given annotation fields. New layers use the public `add_layer`
defaults (green, visible, unlocked, alpha/pen width 1). Layers
and parts are addressed by unique user names because their UUIDs are not
persisted in XML. Primitive/MOP IDs are `internal_id` UUIDs; transient XML
integer IDs are never tool arguments. Native files lacking framework identity
get new UUIDs on open; independent opens need not agree in that case.

MOP targets are unique UUIDs resolving to supported root primitives in the same
document. Harmless group membership is allowed; reject missing, duplicate, wrong-kind,
parent/child, non-similarity or nonzero-local-Z targets before mutation. Per-kind supported target sets: Pocket accepts root
Rect/Circle/closed-Pline/Text/Region shapes; Profile accepts those plus open Plines; Engrave
accepts root Rect/Circle/Arc/Pline/Text curves (Plines may be open or closed); Drill accepts root
Points/Circle primitives. Profile produces a cutter-radius-compensated contour:
Outside preserves the selected boundary as the finished exterior part edge and cuts
in surrounding stock. Inside preserves it as the finished opening edge and cuts on
the removable interior side. Pocket clears the entire bounded interior into chips;
for a through-opening that may release a slug, Profile Inside is normally preferred.
For an open Pline, Profile side is relative to stored vertex traversal rather than a
closed interior/exterior; inspection cannot turn it into an invariant physical side.
Text can therefore be pocketed or profiled with the same round-cutter reach limits as
other compound contours. Corner overcut adds an extra machining move into inside
corners that otherwise remain uncut; the resulting stock overcut is useful for fitted
parts such as slot joints or inlays. Engrave
follows the selected curve or Text outline as the tool path when roughing clearance is
zero; a nonzero signed roughing clearance offsets it. Drill operates at
point-list entries or circle centers rather than tracing circle boundaries. Select
among them by requested machining intent; never silently replace an unavailable
Profile with compensated helper geometry and Engrave. CamBam resolves the final
toolpaths from these targets; 4e owns visual acceptance. Require
`target_depth < stock_surface`
and `clearance_plane > stock_surface`. New parts use enabled=true, zero stock
dimensions, empty material, origin (0,0), and no spindle/tool override; no
fabricated MDF/stock-size defaults. This is unspecified stock, not a
zero-thickness machining recommendation. Explicit MOP parameters bypass
framework inferred tool/feed/depth defaults. The agent must not silently invent
target depth, depth increment, feeds or spindle speed. It may offer a reasoned proposal
from known stock, material and tool context, but must elicit user confirmation.
Fresh MCP-authored MOPs supply all required machining scalars explicitly, so their
common fields serialize as `Value`. Empty `custom_mop_header` and
`custom_mop_footer` values are intentionally represented by absent XML elements;
inspection accepts that canonical omission while still rejecting inherited
`Default` state for MCP-required fields. Imported nonempty/native header and footer
content remains preserved by the core template path.

For a through-cut, let `D = stock_surface - target_depth` be total requested depth,
`S` be stock thickness, `n` the pass count and `I` the depth increment. First constrain
`I` by the tool/material-safe stepdown. Then prefer `n*I > D` by a small margin and
`(n-1)*I < S`, so a rounded multiple neither lands exactly at target depth nor leaves
a final pass that cuts only below the stock. The actual final pass is
`D-(n-1)*I`; normally at least one third of it should remain in stock before crossing
the bottom. For 9 mm stock with a 0.5 mm cut-through, `D=9.5`; three nominal passes at
`I=3.2` produce actual depths 3.2, 6.4 and 9.5 mm, leaving 2.6 mm of stock plus 0.5 mm
cut-through in the final 3.1 mm pass. This is a sequencing heuristic, not permission
to exceed the safe stepdown or infer unknown stock/tool/material data. A valid value
explicitly requested by the user takes precedence: report any divergence as advisory
diagnostics, but do not reject or silently substitute it solely because it misses this
recommendation. Pin other settings to the bounded authoring surface: Profile keeps
`lead_in_type="None"` and exposes signed roughing clearance plus bounded automatic
holding-tab parameters; Pocket pins Spiral lead-in, stepover 0.4,
`InsideOutsideOffsets` fill, Roughing and finish stepover 0 while exposing signed
roughing clearance; Engrave pins Roughing, final increment 0 and DepthFirst while
allowing signed roughing clearance and EndMill or the native `VCutter` enum. Drill
supports CannedCycle plus SpiralMill CW/CCW: CannedCycle defaults to a Drill profile
with peck 0, retract 5, dwell 0 and requires zero roughing clearance for fresh MCP
authoring; SpiralMill exposes its signed clearance, explicit/Circle-Auto diameter,
lead-out and flat-base controls with a method-aware tool-profile default. All share
XY, CW, ExactStop, optimisation Standard, tool number 0, crossover 0.7 and empty
custom header/footer; family-specific unexposed parameters remain fixed. Inspect
returns the closed per-kind records described by the parity matrix. No unrestricted
`**kwargs` input.

These adapter pins use the core's declarative Profile/Pocket encoding policy.
Profile's pinned `LeadInType=None` is an explicit `Value` but emits no inactive
SpiralAngle and no invented lead-out fields. Pocket's Spiral lead-in emits only the
mode and angle. `FinalDepthIncrement=0` remains an explicit choice. Profile
`HoldingTabs` always explicitly selects None or Automatic; only Automatic carries
the bounded dependent tab values. The adapter's required `tab_use_leadins=false`
therefore cannot encode an inert true value. Manual tabs, other lead modes and
independent lead-out authoring are not part of this MCP surface.

Inspection serializes copies, never mutable entity objects. `DocumentSummary` is
`{name, units, source: null | {path, sha256} | {name, sha256, bytes}, counts: {layers, parts, primitives,
mops}}`; counts are nonnegative integers. Source is informational and never means
save-in-place. `SavedArtifact` is `{path, absolute_path, sha256, bytes,
delivery: "server_workspace_handoff", workspace_file_created: true,
client_file_created: false}` with positive bytes. `SerializedArtifact` is the complete typed inline-content record defined above;
its suggested filename does not assert file creation.
`InspectionPage` is `{summary: DocumentSummary, offset, next_offset: null |
integer, entities: EntityRecord[]}`. Enumerate layers in project order, parts in
project order, primitives by UUID, MOPs in part/MOP order; concatenate in that order
and paginate. Later pages should provide the first page's `expected_revision`.

`EntityRecord` is a discriminated union: layers `{kind:"layer", name, color, alpha,
pen_width, visible, locked}`; parts
`{kind:"part", name, enabled, stock_width, stock_height, stock_thickness,
stock_material, stock_color, stock_offset: [x,y], stock_surface,
machining_origin, default_tool_diameter,
default_spindle_speed, nest_method, nest_rows,
nest_columns, nest_spacing, grid_order, grid_alternate}`; primitives
`{kind:"primitive", id, identifier: string|null,
type, layer, parent: UUID|null, children: UUID[], groups: string[], geometry:
Geometry|null}`; MOPs `{kind:"mop", id, identifier: string|null, type, part,
targets: UUID[], parameters}`. `geometry` is a closed typed record when the
primitive is inside the inspection geometry slice (no parent or children, finite
non-degenerate similarity world matrix, zero local Z offset), otherwise `null`
with `INSPECTION_UNSUPPORTED`: Rect
`{kind:"rect", world_xyz: four XYZ corners in framework order, bounds}`; Circle
`{kind:"circle", center: XYZ, diameter, bounds}`; Arc `{kind:"arc", center,
radius, start_angle, extent_angle, bounds}` with degrees, start normalized to
`[0,360)` and a CCW-positive signed sweep; Pline `{kind:"pline", world_xyz:
stored vertex order, bulges: parallel per-vertex values (the bulge at index i
curves the segment starting at vertex i), closed, bounds}`; Points
`{kind:"points", world_xyz, bounds}`. Bounds are world `[xmin,ymin,xmax,ymax]`
projections; Arc and bulged-Pline bounds use the directed analytic sweep
extrema. World points carry each stored Z plus the total Z offset; Text
reports the anchor/height/font/style/line-spacing/alignment parameter record
plus the optional unused `p2` interchange field, and deliberately no bounds
because the framework's text extent is a font-dependent estimate, not a
computed geometry query. Region reports `outer_curve` plus
`hole_curves` contour payloads (`world_xyz`/`bulges`) and world bounds. MOP
`parameters` is a closed per-kind record for Profile/Pocket/Engrave/Drill
containing the named inputs (`side` becomes `profile_side`) and pinned settings
above, using public dataclass fields. For out-of-slice primitives, return
type/identity/relationships but null geometry and `INSPECTION_UNSUPPORTED`. MOP
parameters remain independently inspectable under the preservation-aware rules
above; no invented geometry or calculated effective inheritance values are returned.
Cross-document copy/transfer between two open documents is
specified in [its own section](#cross-document-copy-and-transfer).

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
Read-only inspect/export/list/planning calls use null request ID even when a valid
compatibility UUID was supplied; create/open failures use null handle/revision.
Close success reports its final revision. Other failed document calls report a
current revision only if the handle exists in this workspace. Diagnostic messages
are bounded to 1024 characters; no traces, raw imported XML or outside paths.
No partially successful mutation results exist in version 1.

`ErrorCode` is one of `INVALID_ARGUMENT`, `WORKSPACE_MISMATCH`, `DOCUMENT_NOT_FOUND`,
`DOCUMENT_EXPIRED`, `STALE_REVISION`, `REQUEST_ID_CONFLICT`, `ENTITY_NOT_FOUND`,
`IDENTIFIER_CONFLICT`, `UNSUPPORTED_OPERATION`, `PATH_INVALID`, `PATH_EXISTS`,
`IMPORT_FAILED`, `CONTENT_MISMATCH`, `EXPORT_FAILED`, `LIMIT_EXCEEDED`, `IO_ERROR`, `INTERNAL_ERROR`.
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
cross-era rejection; strict input/output schema tests; client-content import/edit/export
round trips (including a real stdio payload above the former 1 MiB framing limit),
exact 10 MiB acceptance and over-limit rejection; wrong
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
synthetic artifacts; 4e owns isolated clean installation, actual named local-client
connection and CamBam units/geometry/property/toolpath acceptance. Current production
toolpath acceptance is not extended by this contract.

4d broadens only documented mappings, one outcome-sized family per session with
schema/parity/negative tests before advertisement. **Batches 1-5 are
implemented:** the Circle/Arc/Pline/Points adders, the Text/Region adders
with framework-validated XY topology, typed world-geometry inspection for all
seven supported primitive kinds under the similarity slice (including
bulge-aware bounds and the Text parameter record without font-dependent
bounds), the translation/rotation/uniform-scale/mirror/Z/bake transform tools,
the Pocket/Engrave/Drill adders plus public `set_mop_targets` with per-kind
target rules and closed parameter records, the parenting/group/copy
relationship tools with fresh-identity same-document copies, and batch 5's
cross-document copy/transfer under the two-document contract in
[its section](#cross-document-copy-and-transfer), with direct-framework parity
for both operations, per-document revision semantics and failure addressing,
limit atomicity and ledger replay tests before advertisement. 4d is complete;
4e local-stdio user acceptance follows. Remote hosting, arbitrary
Python/private registries, pickle, generic field setters, deletion/batch edits,
arbitrary XML editing, overwrite and machine/G-code execution remain excluded.
Reopen volatile storage only for a demonstrated unsaved-recovery need; reopen
overwrite only with expected-file-hash concurrency and fidelity acceptance;
reopen additional protocol versions only with a concrete client need and wire
evidence; supported legacy versions remain required acceptance coverage.
