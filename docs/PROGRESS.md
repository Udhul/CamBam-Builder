# Current status

Baseline reviewed 2026-09-07: package version 0.1.0. The modern package provides
entity registries, layer/group/parent and part-MOP relationships, transforms,
XML reader/writer and optional same-code-version pickle snapshots. These are implemented capabilities,
not a guarantee of complete round-trip fidelity. MOP target selections are project-owned.

## Active work and next priority

**2026-09-20 unreleased compatibility boundary enforced.** “Fresh/imported” now
explicitly means optimal fresh CamBam XML and arbitrary supported CamBam-saved XML,
not files or Python calls from earlier unreleased framework builds. CamBam `.cb` XML
is the exchange boundary. Pickle remains only a trusted same-code-version WIP/cache
snapshot for resuming the full object graph; it has no migration promise, and a
future release-grade state format will be assessed from a robust contract rather
than inherited pickle history. Old Part/Primitive pickle default-fill shims, their
synthetic migration regression, two unused `_v2` matrix aliases and positional
`add_part` ordering compatibility were removed. The current 253-test suite passes
with the existing Windows symlink-privilege skip; no CamBam behavior changed.

**2026-09-20 backlog 8a modeled MOP field inventory complete.** Engrave and Drill
now join the common and Profile/Pocket slices with declarative fresh-export policies.
Drill emits mutually exclusive CannedCycle, SpiralMill or CustomScript method fields,
reconciles modeled fields on supported imported method switches, and preserves but
does not switch unknown native methods. Engrave omits an unset final increment and
classifies its inert RoughingFinishing compatibility property honestly. At that
slice's completion, the then-current 250-test suite passed with the existing Windows
symlink-privilege skip. The 8b framework/MCP parity matrix is now complete and
source-checked. Backlog 8c native evidence is now complete: two minimal fixture sets
establish state/omission behavior and the remaining Engrave/Drill effects. The
preservation-aware structured inspection gap in 8d is now the next MOP increment,
followed by the independent 8e group-target correction; no production G-code
acceptance is claimed.

**2026-09-19 SpiralMill Drill MCP authoring implemented and accepted in CamBam Plus 1.0.**
`machining_add_drill` now authors CannedCycle, SpiralMill CW and SpiralMill CCW.
Spiral methods expose signed roughing clearance, explicit or Circle-derived Auto
hole diameter, lead-out enable/length, flat-base behavior and cutter-profile metadata.
Point targets require explicit hole diameter; explicit and Circle-Auto combinations must satisfy
`hole_diameter - 2*roughing_clearance > tool_diameter`. Selected spiral controls are
written with native `Value` state; fresh SpiralMill XML omits CannedCycle-only
peck/retract/dwell and unused CustomScript fields. This follows native testing that
found cached Default text caused risky file-open reconciliation prompts. Nonzero
lead-out length requires lead-out enabled, and positive centerward length is capped
at effective hole radius. All 78 MCP tests and all 239 repository tests pass with the
single existing Windows symlink-privilege skip. Three strict-reopened CamBam Plus 1.0
acceptance files and exact criteria are in
[`output/spiral-drill-190926/`](../output/spiral-drill-190926/); native toolpath
geometry was accepted for all three original files, and the user confirmed the
regenerated files open without any revert/reconciliation prompt.

**2026-09-19 MCP/CamBam alignment correction accepted in CamBam Plus 1.0.** The
first CamBam review accepted zero boundaries, all Grid order
variants, shared-geometry feasibility, open-Pline side behavior and automatic tab
creation, while exposing four contract errors: stock offset was described without
its relation to Part machining origin, PointList validation contained invented XML,
`UseLeadIns=true` was inert with `LeadInMove=None`, and `Vcutter` was not CamBam's
serialized enum token. The contract now defines the derived stock lower-left top
corner as `(machining_origin + stock_offset, stock_surface)` and returns it as
`stock_drawing_origin`; the original example therefore resolves to `(24,-14,4.5)`.
New nesting XML is method-specific, valid imported PointList placement survives
unrelated Part patches, and switching nesting methods drops old method placement
fields. A regression proves MOPs in different Parts can target one primitive.
Automatic-tab descriptions now cover perimeter count/clamping, size threshold,
tool-compensated visible width, target-depth-relative height and Square/Triangle/Skip
roles; MCP authoring rejects tab lead-ins while its Profile lead-in remains None.
Engrave uses the exact `VCutter` enum and explicitly does not promise skeleton or
width/depth-varying V-carving. Parent Machining context remains import-preserved but
not authored/resolved; explicit Part stock overrides it for that Part.

The user opened all eight corrected native-review artifacts under
[`output/1909/`](../output/1909/) in CamBam Plus 1.0. The review accepted stock/origin
relative placement, zero values, Grid and PointList nesting, open-Pline side,
automatic tabs, ordinary Text/VCutter Engrave and native metadata preservation. It
also established that PointList coordinates translate the unnested Part toolpaths:
source-geometry and point translations compound rather than the point replacing the
source origin. This operational detail supplements the official documentation's
PointList-position and nesting-origin description. Part stock is optional and does
not expand with nesting; when nonzero XY stock and multiple copies are both defined,
configuration returns only a non-blocking good-practice fit advisory. All automatic tab fields remain
supported. Imported native Manual tabs are preserve-only; fresh direct-core Manual
authoring now fails explicitly rather than emitting an incomplete point collection.
Manual placement authoring is separately backlogged below. Final verification passes
all 75 MCP tests and all 234 repository tests with the single existing Windows
symlink-privilege skip; compileall, schema JSON parsing and `git diff --check` pass.

**2026-09-19 Text machining and roughing-clearance follow-up implemented.** Native
CamBam behavior confirms that Text is not Engrave-only: Pocket may clear its closed
interiors and Profile may offset its outlines, subject to the same round-cutter reach
limits as other contours. In the vendor's terms, corner overcut adds an extra move
into inside corners that otherwise remain uncut, deliberately overcutting stock for
fitted parts such as slot joints or inlays. MCP Profile and Pocket now accept root Text.
The core already modeled, serialized and read `RoughingClearance` on all four MOP
classes, but MCP authoring had pinned it to zero and inspection rejected nonzero
values. Signed roughing clearance is now authorable and inspectable for Profile,
Pocket and Engrave; zero keeps the normal boundary/line placement, positive leaves
stock and negative overcuts. This increment initially kept fresh Drill authoring at
CannedCycle/zero; the newer SpiralMill entry above supersedes that restriction. Verification
passes all 77 MCP tests and all 237 repository tests with the single existing Windows
symlink-privilege skip; compileall, strict duplicate-key/schema JSON parsing and
`git diff --check` pass.

Direct-core SpiralMill clearance now has an explicit geometry contract from the
user's CamBam Plus 1.0 observation: effective cut diameter is
`hole_diameter - 2*roughing_clearance`, and it must be greater than effective tool
diameter. Equivalently, `hole_diameter > tool_diameter + 2*roughing_clearance`.
Fresh explicit-diameter SpiralMill export enforces that constraint; a native/Auto
hole diameter remains CamBam-resolved.

**2026-09-19 consumer onboarding correction implemented; fresh named-agent rerun
pending.** The first live use of the reusable template exposed an instruction defect:
"ask one concise, grouped set" produced one overloaded prompt mixing durable project
policy with the current test document's filename and machining values. The template
now has an explicit pending/completed gate, requires one separately answerable prompt
per unresolved durable policy (across multiple dialogs when necessary), prohibits
document-instance inputs during onboarding, persists the answers, and resumes the
preserved first request only after setup. A static contract test guards those
distinctions. This supersedes the earlier bootstrap wording; runtime MCP behavior is
unchanged by this correction.

The first successful onboarding/create-file rerun then exposed two smaller template
issues. Save-As/version naming and overwrite/backup were redundantly asked as separate
questions; they are now one file-history policy with versioned Save As recommended.
Artifact validation is explicitly limited to MCP inspection/results, delivery hashes
and configured human CamBam/toolpath review—not repository tests or invented Python
test scripts. Deterministic local helper scripts are encouraged for complex geometry,
layout, nesting and constraint calculations, while `.cb` authoring remains MCP-owned.
Transcript inspection confirmed all sixteen CamBam MCP calls succeeded and the final
file was copied and hash-verified. The unrelated `pytest -q` run failed during
collection after a recursive project scan discovered a nested source checkout; it was
not `.cb` validation. The same run also invented enabled-MOP fixture values despite a
user-provided-per-task policy. Shallow discovery and explicit task-level confirmation
for any non-production fixture values now close those remaining consumer-policy gaps.
The review question is now explicitly post-save: agent completion, artifact delivery
and human acceptance are separate states. The recommended collaboration loop saves and
self-checks every revision before optional CamBam review and further corrections;
required human confirmation gates only accepted/production-ready claims, not delivery.

**2026-09-18 transcript corrections implemented; fresh named-agent rerun pending.**
The 2026-09-19 rerun verified exact same-host staging/open, but exposed three
remaining consuming-agent gaps. Four MOP failures carried valid revisions but a
manually corrupted document handle; malformed-handle errors now tell clients to copy
the opaque value exactly. `document_save` had created only server-workspace files
while the agent repeatedly reported them as saved deliverables; saved artifacts now
carry `delivery=server_workspace_handoff`, `workspace_file_created=true` and
`client_file_created=false`, with stronger completion guidance. Finally, the agent
compared no baseline before repeating a manual edit: stable-ID bounds prove all three
Part A entities moved only `(0,+50)` while Part B was unchanged, yet it invented a
90-degree rotation and alignment translation. Server guidance and the new reusable
consumer-agent template require baseline/current comparison and propagation of only
the measured common delta. Fresh named-agent validation of these refinements remains
pending.

The latest same-host rerun confirmed that the actual 18,594-byte CamBam-saved file
strictly imports with all three layers, five primitives (including the Region), two
Parts and five MOPs. The failed import received a model-reconstructed 16,179-byte
string instead, correctly returned `CONTENT_MISMATCH`, and was followed by an
unguarded open of an older 12,911-byte workspace file. That stale snapshot caused
the false conclusion that native Region/nesting content was unsupported. The
same-host ingress path now mirrors exact egress: `document_list` returns
`workspace_path`; clients binary-copy the current file under a fresh staging name;
and `document_open(expected_sha256=...)` rejects stale/wrong workspace bytes before
parsing or publishing a handle. Tool/server guidance prohibits model-mediated XML
and fallback to similarly named old artifacts. A successful refresh deliberately
returns a new revision-0 snapshot; ordinary mutations still remain sequential.
The MCP suite passes 71 tests with one Windows symlink-privilege skip and the full
suite passes 226 with the same skip; compileall and `git diff --check` pass.

The prior rerun against the combined changes exposed an additional agent-sequencing
and result-interpretation gap. OpenCode issued six same-document geometry mutations
concurrently at revision 0; one correctly committed and five correctly returned
`STALE_REVISION`, after which sequential retries all succeeded. Mutation descriptions
now share one mandatory sequential rule, and stale errors state the current revision,
new-request-ID recovery, and concurrency cause. A later name collision now identifies
the occupying entity type. `document_export` now returns machine-readable
`delivery=inline_content_only`, `file_created=false`, and `INLINE_ONLY_NO_FILE`, so
its suggested filename cannot be presented as an existing workspace file.

The first supplied OpenCode dump was produced against `main`, not this branch; its
missing schema-field extraction and provider-schema failures therefore are not
evidence against commit `e2fd2c3`. The newer dump did exercise that commit. All 36
successful MCP calls retained the four MOPs and both exports returned complete,
stable artifacts; the only failed MCP call correctly rejected malformed client-made
XML. The regression occurred because the agent manually rebuilt 13,099/15,654-byte
exports as different 6,177/4,768-byte files, ignored their different hashes, then
copied a stale file during recovery. Detailed evidence and the baseline/staged split
are recorded in [REVIEW.md](REVIEW.md#mcp-opencode-local-delivery-and-existing-region-audit--2026-09-18).

The retained staged fixes remain distinct: origin Rect import compatibility,
actionable bounded import/schema errors, and atomic conversion of existing root
contours into a Region. The adapter now advertises thirty-seven tools. The new
`geometry_update_region` changes an existing Region atomically while preserving its
UUID, layer, identifier and Profile/Pocket targets. Exact delivery no longer needs
model-mediated XML on a shared filesystem: `document_save` is an explicit atomic
handoff artifact, while portable `document_export` remains the cross-host fallback;
the destination is reread and `document_import(expected_sha256=...)` rejects a byte
mismatch before XML parsing. Region and Part descriptions now explain pocket islands
and that an empty Part has nothing to nest. The 70-test MCP suite and 225-test full
suite pass with the existing Windows symlink-privilege skip; strict-import and
compile checks plus `git diff --check` also pass. A fresh OpenCode/CamBam acceptance
run remains pending.

**4e local stdio acceptance: implementation and automated evidence complete;
named-agent and CamBam acceptance pending** (2026-09-11). The user accepted a
same-machine external client launching its own server subprocess as the initial
deployment boundary; unavailable second-PC hardware no longer blocks this increment.
The Windows wheel matrix passes for the base library on Python 3.9 and the MCP extra
on 3.10-3.13. Removing `mcp`/`mcp-types` disables the adapter with its guarded error
while the direct `CBProject` API remains usable. OpenCode 1.18.30 connected to the
real adapter from an isolated configuration without changing user configuration.
The server now emits one content-free negotiated-protocol record, covered for modern
2026-07-28 and legacy 2025-11-25 subprocesses. A maintained artifact verifier and
exact OpenCode/CamBam acceptance procedure are in the
[development runbook](DEVELOPMENT.md#4e-opencode-and-cambam-acceptance). Completion
still requires the user's OpenCode 4c workflow plus CamBam mm/geometry/Profile/toolpath
report. Streamable HTTP/remote-PC hosting is non-urgent future scope and must add
origin validation, authentication and an explicit exposure policy before any LAN bind.
The first real agentic OpenCode attempt exposed two release-blocking usability defects:
the agent treated server-workspace save paths as client-local delivery, and the adapter
artificially rejected Profile targets except Rects. The model-facing instructions now
make import/export plus client-local read/write the default, reserve open/save for
explicit server-workspace use, and explain Profile/Pocket/Engrave/Drill by machining
intent. Profile now accepts root Circle, closed-Pline and Region boundaries as well as
Rect; schema errors identify their top-level field. The 63-test MCP suite and
217-test full suite pass with the existing Windows symlink-privilege skip. Repeat
OpenCode plus CamBam acceptance remains required before 4e completes.
The next OpenCode attempt accepted the portable export/client-write workflow, closing
that usability defect. It exposed a separate geometry-planning failure: one Pline
vertex and the derived hole center were arithmetically wrong, the agent changed only
the bulge after clarification, never inspected the resulting bounds, and invented
unspecified machining depth/feed values. Circle and Pline creation now return typed
geometry immediately; server/tool guidance requires bounds/center/symmetry/containment
verification before CAM/export and prohibits invented machining parameters. Profile
and Pocket descriptions now distinguish preserved exterior edge, released opening
slug and fully cleared cavity. The transcript itself records Profile `Outside`; if a
corrected file still generates an inside path, its artifact or screenshot is needed
to investigate a distinct CamBam interpretation defect. Generic containment checks
and higher-level parametric polygon construction remain candidate future affordances,
not requirements for this bounded 4e correction.
The runtime package schema is now the single machine-readable contract owner; the
redundant `docs/` copy was removed and documentation links target the packaged asset.
Through-cut guidance now permits a contextual, user-confirmed depth-increment proposal:
respect the material/tool stepdown limit, make nominal multiples slightly exceed total
depth, keep the penultimate pass above the stock bottom, and normally keep at least one
third of the final pass engaged in stock. This avoids a separate below-stock cleanup
pass without presenting the heuristic as a universally safe cutting parameter.
The heuristic is now implemented as the read-only
`machining_calculate_depth_increment` tool, bringing the advertised total to thirty-two.

It accepts either an exact pass count or a maximum stepdown, uses unit-appropriate
upward rounding, returns the complete clamped pass sequence and final engagement, and
marks whether the plan retains the one-third final-stock fraction. A valid explicit
constraint that diverges is returned with diagnostics rather than rejected or replaced.
The tool deliberately does not infer material/tool limits; the caller supplies and
confirms that constraint.

The follow-up settings increment adds explicit MCP operations for Part stock/native
Grid nesting and layer display properties. A Part can now be configured before MOP
creation, so clients need not clone geometry to represent a 3x3 nested Part or leave
the exported stock undefined.

The latest OpenCode transcript exposed missing Part-level controls: MOP creation
defaulted to zero/undefined stock and a requested 3x3 repetition was represented as
nine geometry copies while `<Nesting>` remained `None`. The adapter now exposes
`machining_configure_part` for stock dimensions/material and native Grid/IsoGrid
settings, and inspection returns those resolved values. A separate
`document_set_layer_properties` operation exposes the existing layer presentation
API; it is not a requirement of the cited transcript. Imported native nesting is
preserved while unchanged and modeled settings become authoritative after explicit
direct-framework or MCP reconfiguration. Export and strict re-import cover every
modeled nesting field; cross-entity names return `IDENTIFIER_CONFLICT`, and the
legacy positional `add_part` ordering remains compatible. Named-client/CamBam
acceptance of generated nesting toolpaths remains pending.

The following OpenCode transcript audit found 17 MCP calls with two recoverable
input errors: a primitive identifier collided with its new layer, and a read-only
`document_export` call included the then-unsupported `request_id` field. Both retries
succeeded, the artifact was written in the client workspace, and its hash matched.
The adapter now states the identifier constraint directly, reports object-level schema
failures with the offending top-level field, and tolerates an optional valid request
UUID on every read-only call without retaining it in the retry ledger. No transport
or stdio mismatch was found; repeat the named-agent/CamBam acceptance with a fresh
client process after these descriptions are loaded.

Human/AI continuity is now an explicit contract rather than assumed conversational
state. `document_list` brought the then-advertised total to thirty-five and discovers the
current boot's live handles, revisions and original source hashes. Client-local files
remain authoritative: after a manual save, hash and import the complete current XML
as a new revision-0 handle; after restart do the same; after `STALE_REVISION`, inspect
the current snapshot and retry a still-applicable mutation with a new request ID.
Automated coverage exercises manual-change re-import, discovery, follow-up editing,
hashes, ignored read-only UUIDs and stale-revision recovery.

An OpenAI-backed client then rejected the tool listing because `TextContent` used a
positive-lookahead regex (`(?=\S)`), which its JSON-schema compiler does not support.
The packaged pattern now expresses the same non-whitespace/content rule without
lookaround, and a schema regression rejects reintroducing lookaround syntax.

Profile `CornerOvercut` is now explicitly exposed through the public
`add_profile_mop` API and the MCP `machining_add_profile` input. It defaults to
`false`; when enabled, CamBam performs its native round-tool inside-corner
overcut calculation. Pocket, Engrave and Drill have no corresponding native
CamBam field and remain unchanged.

**4d batch 5, cross-document copy/transfer: implemented and verified**
(2026-09-11). The adapter advertises thirty-one version 1 tools after adding
`relationship_copy_tree_between` and `relationship_transfer_tree_between`
under the two-document contract recorded in the [MCP contract](MCP_CONTRACT.md#cross-document-copy-and-transfer).
Both tools take two distinct live handles (any two imported, opened or created
documents), each with its own required revision. Both document locks are held
in sorted-handle order from the revision checks through publication; copy
stages on a target clone and reads the live source, transfer stages both
documents and runs the public `transfer_primitive_tree` between the clones;
publication and both revision increments happen inside a cancellation shield,
so copy increments only the target and transfer advances both together, never
partially. All per-call failures leave both documents and all files unchanged;
a failing target document check addresses the target handle with its current
revision, other outcomes address the source, and success data carries the
mapping plus both handles and revisions. Tests copy and transfer between two
populated imported documents, compare both saved sides against independently
authored public-framework copy/transfer projects by XML semantics, and cover
replay, request conflicts, stale handling, closed/expired targets, limit and
cancellation atomicity, same-revision transfers, and opposing-direction lock
ordering under a timeout. The six cross-document tests pass; the full suite
reports 207 tests (206 passed; one Windows symlink-privilege
skip). Evidence is in the
[review](REVIEW.md#mcp-cross-document-copytransfer---2026-09-11).

**4d state: complete.** All five 4d batches plus portable interchange are
implemented and verified, closing the documented adapter surface. It handed off
to the active 4e local-stdio acceptance entry above.

**4d portable client document interchange: implemented and verified**
(2026-09-11). The adapter then advertised twenty-nine version 1 tools.
`document_import` accepts complete client-supplied UTF-8 `.cb` XML and publishes
a new volatile revision-0 handle through the same strict reader, 10 MiB byte
limit, entity limits, capacity reservation and retry rules as workspace open;
its retained retry signature stores only the content hash and byte count.
`document_export` returns the exact locked revision as a typed inline artifact
with filename, media type, encoding, byte count, SHA-256 and complete content,
without publishing a server workspace file or retaining large results in the
request ledger. The stdio framing bound is 32 MiB so the unchanged 10 MiB XML
limit remains usable after JSON escaping. Automated evidence covers client
import/edit/export/reimport, hashes, revisions, malformed/declaration XML,
exact-limit acceptance, over-limit rejection, cleanup/failure behavior, and an
exact-10-MiB protocol round trip. Invalid content types and an export/edit lock
race are also covered. MCP resources remain deferred until 4e named-client tests
show they improve delivery without
making artifacts inaccessible to agents. Evidence is in the
[review](REVIEW.md#mcp-portable-document-interchange---2026-09-11).

**First MCP authoring and round-trip slice: complete** (2026-09-10; backlog
4c). The adapter then advertised all eight version 1 tools and staged rectangle,
Profile and translation edits on independent project clones before one atomic
revision publication. Inspection returns typed world Rect corners/bounds and the
closed explicit Profile parameter record, while unsupported imported detail stays
diagnostic rather than inferred. The exact create/add/save/open/translate/save
workflow preserves primitive/MOP identities, targets, order and A's bytes and
matches an independently authored public-framework reference. A reusable
demonstration writes verified A/B artifacts to a unique ignored workspace.
The five authoring tests and all 28 MCP tests pass (one Windows symlink-privilege
skip); the full suite reports 179 passes and the same skip.
Detailed parity, negative-case and verification evidence is in the
[review](REVIEW.md#mcp-first-authoring-and-round-trip-slice---2026-09-10).

**4d batch 1, curve/point geometry breadth: implemented and verified**
(2026-09-10). The adapter advertises twelve version 1 tools after adding
`geometry_add_circle`, `geometry_add_arc`, `geometry_add_pline` and
`geometry_add_points` with their closed schema records (per-vertex Z/bulge,
2..10000/1..10000 point bounds), and `geometry_translate` now accepts root
Rect/Circle/Arc/Pline/Points primitives inside the then translation-only
slice.
Inspection moved to a typed closed `geometry` payload per supported kind —
Rect corners, Circle center/diameter, Arc center/radius/degree angles, Pline
vertices with parallel per-vertex bulges and closed flag, Points vertices — all
with directed analytic world bounds; Text, Region and out-of-slice transforms
stay diagnostic (`geometry: null`, `INSPECTION_UNSUPPORTED`). Retry identity now
canonicalizes nested defaults, so omitted `z`/`bulge` and explicit zeros replay
as the same request. Parity fixtures compare adapter payloads, XML semantics
(order- and id-normalized because primitive document order is the framework's
UUID-sorted listing) and save/reopen identity against independently authored
public-framework projects; negative cases cover schema bounds, identifier and
layer-name conflicts, stale revisions, wrong-kind translate/profile targets and
atomic failure. Five new geometry suites and the full 185-test suite pass
(one Windows symlink-privilege skip). Remaining limits: Text/Region typed detail,
Pocket/Engrave/Drill, parenting/groups/copy-transfer and bake transforms are
still unsupported; Profile targets remain root Rects only. Evidence is in the
[review](REVIEW.md#mcp-geometry-breadth-batch-1---2026-09-10).
**4d batch 4, relationships and transforms: implemented and verified**
(2026-09-10). The adapter advertises twenty-seven version 1 tools after adding
`relationship_set_parent` (public `link_primitive_parent`; null detaches;
cycles/self links are `INVALID_ARGUMENT`), `relationship_add_to_group`/
`relationship_remove_from_group`, `relationship_copy_tree` (public
`copy_primitive_tree` into the same document with `preserve_ids=False`; fresh
copy identities; included layers/groups and optional MOPs require explicit
identifier/group maps or the framework's collision rejection is surfaced as
`INVALID_ARGUMENT`), and the transform tools `geometry_translate_z` (baked
stored-geometry Z shift), `geometry_rotate`, `geometry_scale` (uniform only),
`geometry_mirror` and `geometry_bake` (folds the world transform into stored
geometry; non-axis-aligned Rects become closed Plines). The inspection slice is
now similarity-based: root primitives whose world matrix is a finite
non-degenerate similarity (translation, rotation, uniform scale, reflection)
keep typed geometry and exact analytic bounds, so rotated/scaled/mirrored
results stay diagnosable; shear/non-uniform-scale, nonzero local Z and any
parent/children/groups relationship remain diagnostic. Primitive inspection
records now also carry their sorted `groups`. Parity fixtures compare adapter
and independently authored public-framework results for transforms (including
bulge sign flips under reflection and bake), parent/group/copy state, and XML
semantics; negative cases cover cycle/self links, non-primitive parents,
copy identifier/layer/group collisions, wrong-kind roots, Text bake
restrictions, cx-without-cy pairs, stale revisions, replay and concurrent
same-revision serialization. Post-review resilience checks also cover atomic
primitive/MOP limit rejection for same-document copies and diagnostic fallback
for schema-oversized imported geometry. Five relationship/transform suites pass;
the full suite reports 197 tests (196 passed; one Windows symlink-privilege skip).
Tool annotations now match the destructive/nondestructive contract. Remaining
limit: cross-document copy/transfer is deferred — it needs a two-document
staging, revision and failure-semantics contract decision before tools are
advertised. Evidence is in the
[review](REVIEW.md#mcp-relationship-and-transform-breadth-batch-4---2026-09-10).

**4d batch 3, Pocket/Engrave/Drill machining breadth: implemented and
verified** (2026-09-10). The adapter advertises eighteen version 1 tools after
adding `machining_add_pocket`, `machining_add_engrave`, `machining_add_drill`
(closed explicit parameter records with pinned public defaults; Pocket uses
Spiral lead-in/`InsideOutsideOffsets`/Roughing, Engrave pins Roughing and
DepthFirst with an EndMill, Drill pins the CannedCycle method with a `Drill`
tool profile and peck/retract/dwell inputs) and `machining_set_mop_targets`
(public `set_mop_targets`) for explicit target replacement. MOP target rules
are per kind: Profile accepts root Rect/Circle/Pline/Text/Region shapes; Pocket
accepts root Rect/Circle/closed-Pline/Text/Region shapes; Engrave accepts root
Rect/Circle/Arc/Pline/Text curves; Drill accepts root Points/Circle primitives.
Inspection returns closed per-kind parameter records with the same
explicit-Value XML-state guard as Profile, and out-of-slice/inherited MOPs stay
diagnostic. Parity fixtures compare adapter payloads, resolved targets and XML
semantics against independently authored public-framework projects (including
target replacement and reopen), and negative cases cover per-kind target
rejections, depth/clearance relations, drill scalar bounds, duplicate targets,
wrong-kind retargeting, stale revisions, replay and concurrent same-revision
serialization. Two new machining suites pass; the full suite reports 192 passes
(one Windows symlink-privilege skip). Remaining limits: parenting/groups/
copy-transfer and bake transforms are still unsupported; CamBam toolpath
semantics (including drill positions) remain 4e acceptance. Evidence is in the
[review](REVIEW.md#mcp-pocket-engrave-drill-breadth-batch-3---2026-09-10).

**4d batch 2, Text and Region authoring: implemented and verified**
(2026-09-10). The adapter advertises fourteen version 1 tools after adding
`geometry_add_text` (anchor, font, style, line spacing, alignment, elevation;
content bounded 1..1024 with newline-only control allowance) and
`geometry_add_region` (closed outer contour plus 0..100 hole contours, 2..10000
vertex records with per-vertex Z/bulge) whose XY topology failures
(self-intersection, zero-length segments, holes outside/disjoint-violating or
nested) return `INVALID_ARGUMENT` with the bounded framework message.
Inspection adds closed `text` (anchor/height/font/style/line-spacing/
alignments, optional unused `p2`; deliberately no bounds because text extents
are a font-dependent estimate) and `region` (outer/hole contour
`world_xyz`+`bulges`, directed analytic bounds) payloads, and
`geometry_translate` now also accepts root Text/Region primitives. Parity
fixtures compare adapter payloads and XML semantics against independently
authored public-framework projects (bulged-outer Region with an exact `1e-9`
`-1` sweep dip verified), save/reopen retains identities and geometry, and
negative cases cover schema bounds, alignment enums, topology violations,
identifier/layer conflicts, stale revisions and atomic failure. Five new
Text/Region suites pass; the full suite reports 190 passes (one Windows
symlink-privilege skip). Remaining limits: Pocket/Engrave/Drill,
parenting/groups/copy-transfer and bake transforms are still unsupported;
Profile targets remain root Rects only. Evidence is in the
[review](REVIEW.md#mcp-text-and-region-breadth-batch-2---2026-09-10).

**4d state after batches 1-4 plus portable interchange (historical snapshot,
2026-09-11):** complete except cross-document copy/transfer, which was then the
next increment and needed a two-document staging/revision/failure contract
decision in [MCP_CONTRACT.md](MCP_CONTRACT.md) before its tools were
advertised. That decision is now recorded and batch 5 is implemented; see the
active entries above and the
[review](REVIEW.md#mcp-cross-document-copytransfer---2026-09-11).

**Secure MCP server and document foundation: complete** (2026-09-10; backlog
4b). The optional Python >=3.10 adapter provides a strict stdio boundary, an
explicit volatile workspace/document model and the five document tools needed to
create, open, inspect, save and close `.cb` documents. It contains all workspace
paths, rejects links and unsafe XML, bounds input/state growth, keeps revisioned
idempotency ledgers and publishes new files atomically. Public project cloning and
strict byte import provide the staged-document prerequisites.

The server targets MCP 2026-07-28 while also negotiating the deployed 2025-06-18
and 2025-11-25 client versions. Codex CLI 0.154.0 advertised 2025-06-18 even with
its `mcp_2026_07_28` feature enabled, so backward protocol compatibility is now a
required adapter boundary rather than deferred work. A deterministic Codex
app-server session completed create/inspect/save/open/inspect/close/close against
the installed adapter and independently verified the saved file hash. Detailed
security, client and packaging evidence is in the
[review](REVIEW.md#mcp-foundation-and-client-compatibility---2026-09-10).
The final suite completes successfully: 174 tests pass and one Windows
symlink-privilege case skips; the equivalent junction/reparse-point rejection
is exercised and passes.
The authoring boundary subsequently advanced through the completed 4c slice.

**MCP protocol/client/adapter contract: complete** (2026-09-10; backlog 4a).
The [current contract guide](MCP_CONTRACT.md) and [twenty-nine tool schemas](../cambam_builder/mcp_adapter/contract_v1.schema.json)
settle stdio MCP 2026-07-28, optional SDK 2.2.0/Python >=3.10, volatile explicit
handles, revision/retry rules, workspace containment and new-file-only saves.
SDK stdio direct-call/discovery probes passed. OpenCode 1.18.29's installed backend
uses the 2025-11-25 handshake and failed the original modern-only probe; 4b now
supports that deployed protocol era. Desktop GUI and second-PC acceptance remain
for 4e. Scope,
schema checks, compatibility evidence and reopening criteria are persisted in the
[contract](MCP_CONTRACT.md#acceptance-and-handoff-to-implementation) and
[review](REVIEW.md#mcp-protocol-and-adapter-contract---2026-09-10).
No adapter runtime or project dependency was added in 4a; 4b subsequently added
the SDK as an optional Python-version-gated extra.

**Packaging and supported Python validation: complete** (2026-09-10; backlog
item 3). Published metadata now declares Python >=3.9 and NumPy >=1.23.5 directly
in `pyproject.toml`, the sole dependency source;
`uv sync --python 3.13` created the local `.venv`. The generated lockfile is
intentionally ignored so supported-version checks continue resolving currently
compatible NumPy releases independently. Isolated wheel and sdist builds
contain both package roots and all active modern modules, with version 0.1.0,
`Requires-Python: >=3.9`, and the bounded NumPy dependency. Clean wheel installs from
outside the repository pass imports, metadata/source-path checks, representative
construction/XML behavior and all 142 tests on Python 3.9, 3.10, 3.11, 3.12 and
3.13. The sdist independently installs and passes all 142 tests on the 3.9
minimum. Exact interpreter/NumPy versions and reproduction guidance live in the
[development runbook](DEVELOPMENT.md#packaging-and-supported-python-validation),
with [review evidence](REVIEW.md#packaging-and-supported-python-verification).
Python 3.8, CLI entry points and publishing remain unsupported/out of scope.

**Varying-Z bulged Pline and Region interchange: implemented and verified**
(2026-09-10). User-supplied CamBam-generated XML establishes that Pline and
Region contours may combine bulge with unequal endpoint Z and that Region
topology is consumed as an XY projection. The framework now retains and emits
those vertex records; Region no longer imposes contour coplanarity, while its
existing XY simplicity, containment, disjointness and curve-transform rules
remain intact. Geometry queries still report stored endpoints, and bounds remain
XY projections; no intermediate spatial-curve or MOP toolpath calculation was
added. The native-derived Region regression covers an outer curve and hole with
mixed Z/bulge through two XML cycles at `1e-9` tolerance. Four focused suites pass
49 tests and the full suite passes all 142 tests. Owners: entities, Region,
focused elevation/vertex/Region tests, specification and
[review evidence](REVIEW.md#varying-z-bulged-pline-and-region-interchange).

**Canonical Pline/Points vertex records: implemented; automated verification
complete** (2026-09-09). `Vertex(x, y, z=0, *, bulge=0)` now keeps coordinates,
elevation and segment bulge together. Pline, Points and Region-owned contours use
only `vertices`; old `vertex_z`, `relative_points` and `(x, y, bulge)` inputs are
removed. XY/XYZ tuple shorthand is unambiguous, Points rejects nonzero bulge, and
existing geometry-query outputs remain unchanged. Repository callers and tests
were migrated explicitly. Current-version pickle and repeated XML interchange
retain the new representation; old pickle migration is unsupported as authorized.
Owners: entities, project adders, reader, Region, focused tests, specification and
[review evidence](REVIEW.md#canonical-vertex-record-refactor).
Six focused vertex tests and all 141 suite tests pass. Two new framework round
trips compare Pline, Points and Region XML numerically with the accepted
pre-refactor B fixture at `1e-8`; maximum observed difference is zero, and hashes
confirm the accepted A/B/C files were unchanged. Output semantics did not change,
so the existing CamBam acceptance remains applicable and was not repeated.

**Region and all-shape Z parity: implemented; automated verification, review
and CamBam display/property acceptance complete** (2026-09-09; backlog item 2).
Pline/Points vertex elevations, Circle/Arc/Rect/Text elevation fields, independent
parent/local Z offsets and one owned-contour Region primitive now extend the
existing project/XML/persistence/transfer model. Full/global subtree bakes stage
geometry changes before publication. Unsupported spatial matrices and invalid
Region topology fail explicitly. User confirmed the unreleased API may change.

Owners: entities, Region module, project transform/API orchestration, matrix
helpers, reader, transfer helper and focused regressions.
[Implemented contract](structure_spec.md#shape-elevation-and-region-contract) and
[implementation/review evidence](REVIEW.md#region-and-shape-elevation-implementation).
All 135 tests pass, along with compile/import and independent XML inspection
checks. The final topology review found and verified repairs for arc endpoint
containment, topology-changing XML rounding and small uniform scales.
The prepared A/B generator verifies eight primitives covering seven types,
XYZ/identity preservation, two Region holes and one MOP source through repeated
round trips. The user accepted the A/B display and property checks in CamBam Plus
1.0. Native C and fresh/moved Text evidence establish that serialized Text `p2`
is an optional, currently unused interchange field and omitted `p1` means the
default origin; a C-to-D framework round trip
retains it and all other declared data. See the recorded
[CamBam validation](DEVELOPMENT.md#manual-shape-parity-acceptance). The private native
Region sample contains independently confirmed self-intersections and is rejected
by the declared strict topology contract; it remains unchanged.

**Copy/transfer utilities implemented; automated checks complete** (2026-09-08; backlog item 1b).
Contract recorded in
[copy and transfer contract](structure_spec.md#copy-and-transfer-contract).
Provides complete subtree copy/move with preserved world pose, explicit identity
remapping, collision rejection, closed MOP targets and staged registry updates.
Owners: project API/transfer helper, focused copy-transfer tests and specification.
Acceptance: atomic failure and relationship/collision regressions plus two XML
round trips checking identity, references, geometry and parameters. Sixteen focused
tests and the full 88-test suite pass; compile/import checks also pass. No blocker
is recorded and no manual CamBam validation is required for this registry/XML scope.

**Curved-geometry bounds: implemented; automated checks complete** (2026-09-08).
Arc and bulged-Pline bounds now use directed analytic sweep extrema under the
complete finite affine transform, including reflection, nonuniform scale, shear
and singular projection. Seven focused regressions cover wrap-around, full and
negative sweeps, open/closed and zero-bulge segments, tolerances, malformed
inputs and overflow-resistant large finite cases. All 72 tests pass. Bounds are
not serialized, so no manual CamBam validation adds evidence for this numerical
slice. See the [contract](structure_spec.md#curved-geometry-bounds-contract) and
[verification](REVIEW.md#curved-geometry-bounds-verification).

**MOP core-model ownership and CamBam interchange: implemented; automated checks
complete; CamBam 1.0 display/property interchange accepted** (2026-09-08).
Owners: project target registry/API, entity MOP parameter encoding, reader/writer,
focused MOP tests and updated callers. Breaking API/storage changes were authorized;
no legacy adapters or old-pickle migration were added.

Acceptance: one project-owned target selection per MOP; explicit targets and live
named groups with validated atomic mutation and deletion cleanup; native XML target
edits/order remain authoritative; supported parameter and Default/Value reconstruction;
synthetic round trips plus prepared CamBam-edited input acceptance.
Implementation and automated verification are complete. The user accepted the
prepared A/B and native-edited C/framework-roundtripped D display/property cases
on CamBam Plus 1.0. The C-to-D automated comparison preserved operation order,
targets, exact parameter/state sets and geometry; C and D loaded without error
and all stated visual/property checks matched. This establishes the supported
interchange case, not production toolpaths or complete `.cb` coverage.
[Target contract](structure_spec.md#mop-target-ownership-contract).

Rect rotation/shear baking repair: **implemented; automated checks complete; synthetic display accepted**
(2026-09-08). Owners:
`cambam_entities.py`, the explicit bake helper in `cambam_project.py`,
`tests/test_rect_bake_defect.py`, and the implemented specification.
Acceptance: preserve exact closed outlines, identities and relationships through
full/explicit/global baking and two XML round trips; retain axis-aligned Rects
and preserve descendant world geometry. Non-axis-aligned results become Plines
in place so existing Python references remain valid. No implementation blocker.
All 51 suite tests pass, including nine Rect regressions. Synthetic A/B files
were accepted by the user on 2026-09-08: all conditions met, full outline match
and identity transforms. Later clarification establishes CamBam Plus 1.0 as the
validation environment. See
[acceptance evidence](REVIEW.md#rect-baking-display-acceptance).
The following MOP group-source compatibility increment is recorded above;
core ownership/interchange is active above. General component ordering,
curved geometry and alignment remain outside this increment.
Phases 1 and 2 (project discovery and working
agreement) and [phase 3 (initial engineering review)](REVIEW.md#phase-3-completion-audit)
are complete; [workflow completion evidence](REVIEW.md#phase-1-and-2-completion-audit)
records the foundations. Product acceptance remains separate.

**Parent round-trip fidelity: implemented; automated checks complete.** Preserve parent
identity and world geometry through XML export/import. Owners: `cambam_writer.py`,
`cambam_reader.py`, new focused `tests/`, and the specification for any clarified
local/world transform contract. [Defect evidence](REVIEW.md#parent-identity-and-transform-reconstruction).

Acceptance:

- Preserve UUIDs, identifiers, layers and parent edges regardless of XML order.
- Preserve root/child/grandchild world matrices and geometry through two round
  trips, including non-identity parent transforms and child offsets.
- Document and test world-pose behavior for parentless/missing/invalid parents
  and explicit failure/fallback behavior for singular parent transforms.
- Add regression tests under `tests/` that fail before the fix and pass afterward; assert
  counts/relationships and numeric tolerances. Record the test command in the
  runbook and pass relevant syntax/import checks; inspect synthetic XML.
- Record CamBam geometry validation separately; do not claim it from local tests.

Verification: nine authored unittest tests pass on Python 3.10.9 / NumPy 1.23.5,
with syntax/import checks complete. See [repair evidence](REVIEW.md#parent-round-trip-repair-verification).
Synthetic parent A/B geometry/placement **accepted by the user on 2026-09-07**:
both files match the expected world coordinates; resetting B's matrices restores
local placement. Later clarification establishes CamBam Plus 1.0 as the validation
environment. See
[acceptance evidence](REVIEW.md#parent-ab-user-acceptance).
Packaging and broader CamBam compatibility remain unverified.

**Parent-cycle rejection: implemented; automated checks complete.**
Owner: `cambam_project.py`; focused API and synthetic XML regression tests.
Acceptance: reject self/descendant cycles before mutation; preserve both parent
indexes, memberships and transforms; valid reparent/detach remain usable; cyclic
XML produces an acyclic hierarchy with preserved world geometry.
[Verification evidence](REVIEW.md#parent-cycle-rejection-verification) records
the regression and checks. Related parent A/B display validation is accepted.
General transform refactoring and MOP migration remain outside this slice. Rollback: revert only the slice's
patch; do not rewrite existing CAD files or saved pickle state.

## Completed line: hierarchy and global transform fidelity

State: **implemented; automated checks complete; synthetic display accepted**. Owner: `transform_primitive` in `cambam_project.py`,
`tests/test_global_transforms.py`, and the implemented specification. Acceptance:
world points in the selected subtree receive `M @ world_point` exactly once in
both matrix and baked modes; ancestors/siblings/registries remain unchanged.
Matrix mode retains descendant local matrices; baked mode retains every effective
matrix. Reject invalid affine inputs and unsolvable coordinate frames before
mutation. Verify wrappers and repeated XML round trips on straight polylines.
Curved/component baking and alignment's separate implementation remain outside
this slice. Eight new regressions and all 26 suite tests pass.
[Before/reference/result files and criteria](DEVELOPMENT.md#manual-global-transform-acceptance)
were accepted by the user: A/B coordinates match, B retains matrices, and child/leaf
move (+5,-3) from Before while root stays fixed. Later clarification establishes
CamBam Plus 1.0 as the validation environment.
[Verification evidence](REVIEW.md#global-transform-ordering-verification)
records the failure and coordinate-frame policy.

Full-bake hierarchy fidelity: **implemented; automated checks complete; synthetic
A/B user acceptance complete**. The user confirmed matching final coordinates,
identity matrices in both files and the intended layer difference. See
[repair and acceptance evidence](REVIEW.md#full-bake-hierarchy-verification).

## Completed implementation: MOP round-trip identity

**Implemented; automated checks complete; synthetic CamBam acceptance complete.** Entity
encoders store UUID/identifier metadata separately from display names. The reader
restores identity before registration, retains XML operation order and targets,
allocates fresh identities for legacy/malformed Tags, and fails import on explicit
identity collisions. No MOP registry migration or live group semantic change.
Five new regressions and all 31 suite tests pass. The original reader fails the
new suite. See [evidence](REVIEW.md#mop-identity-round-trip-verification) and
[prepared A/B acceptance](DEVELOPMENT.md#manual-mop-identity-acceptance).
The stopping condition for implementation is met; complete parameter coverage,
Default/Value fidelity and production toolpaths remain unverified.

## Completed implementation: export failures and state paths

**Implemented; automated checks complete.** Owners: `cambam_writer.py`,
`CamBamProject.save_state`, `tests/test_export_failures.py`, and
`tests/test_state_persistence.py`. Encoder failures propagate; export replaces
the destination only after complete serialization and close. Bare state filenames
work, and directory errors propagate. Eight export and three persistence tests
plus all prior regressions pass (42 total). Acceptance covers existing/new
destinations, failure propagation/cleanup, XML round trips and restored pickle
project links. No blocker or manual validation remains for this scope.
See [contract](structure_spec.md#export-failure-and-state-saving-contract) and
[verification](REVIEW.md#export-failure-and-state-path-verification).

## Remaining backlog, in order

1a. **Completed 2026-09-08.** Exact Arc sweep and bulged-Pline extrema, the
    finite-affine transform/tolerance policy and focused regressions are recorded
    in the implemented specification and review.
1b. **Completed 2026-09-08:** copy/transfer utilities; scope and
    acceptance are recorded above.
2. **Completed 2026-09-09:** Core CamBam shape parity: Region shapes and Z coordinates across
   Pline, Circle, Rect, Arc, Points, Text and Region. Extend existing entity/API/XML
   patterns with backward-compatible coordinate semantics, identity preservation
   and synthetic round-trip/display checks. This is independent upstream feature
   support, after core design/correctness and before downstream integrations.
   [Scope, evidence, delivery and acceptance plan](SHAPE_PARITY_PLAN.md).
   The completed canonical vertex-record refactor subsequently replaced the
   temporary parallel Pline/Points coordinate/elevation storage without changing
   the accepted XML geometry or query contracts.
3. **Completed 2026-09-10:** packaging and supported Python 3.9-3.13 validation.
   Add CLI/publishing work only for an actual need.
4. Build and maintain a local stateless MCP adapter for AI-assisted CamBam generation
   and load/modify/save workflows on another PC. User-requested future work, split
   into fresh-session increments with independent stopping conditions:
   - **4a — completed 2026-09-10:** [protocol/client and adapter contract](MCP_CONTRACT.md),
     eight tool schemas and compatibility probes. OpenCode 1.18.29 is incompatible
     with a modern-only server; backward negotiation is implemented in 4b and
     desktop acceptance remains pending in 4e.
   - **4b — completed 2026-09-10:** secure server/document foundation, required
     public clone/strict-import contracts and backward protocol negotiation.
   - **4c — completed 2026-09-10:** one authoring/MOP/inspect/save/reload parity
     slice, including the runnable synthetic demonstration.
   - **4d — completed 2026-09-11 (batches 1-5):** expand the explicitly
      supported API and resilience coverage. Batches 1-4 delivered the seven
      primitive authoring families with typed world queries, all four basic
      MOPs with explicit target replacement, the similarity transform/bake
      tools, and the parenting/group/copy relationship tools; batch 5 added
      cross-document copy/transfer under the recorded two-document contract.
      Sol/high, with Terra/xhigh suitable for settled coverage batches.
   - **4e — automated/local connection evidence complete; user acceptance pending:**
     validate installation, named local-client interoperability and CamBam user
     acceptance. Same-machine stdio is the accepted initial boundary because
     second-PC hardware is unavailable.
   Each increment must persist its decisions, evidence and remaining limits before
   handoff; later breadth must not leak into earlier scopes. Full acceptance and
   maintenance details: [MCP delivery plan](MCP_PLAN.md#delivery-increments-and-session-boundaries).
5. **Non-urgent future MCP transport:** add authenticated Streamable HTTP only for
   a concrete remote-PC or multi-client need. Define modern/legacy HTTP behavior,
   bind/origin/authentication/TLS policy, document-handle lifecycle and cancellation
   before implementation; never expose the current write-capable service by merely
   binding to a LAN interface.
6. **Planned future feature:** rest-area calculation and rest
   machining helpers for pocket and inside/outside profile MOPs. Five outcomes:
   pure rest regions; safe expansion for a smaller endmill; pointed/flat-tip
   V-cutter preparation; a general bounded XYZ V-carving path calculator; and
   V-cutter edge tracing with corner cleanup. Region topology and shape Z fidelity
   are independently prioritized upstream dependencies owned by the
   [shape parity plan](SHAPE_PARITY_PLAN.md), not part of rest implementation.
   Calculations must run programmatically in this framework without CamBam;
   `.cb` files provide interchange, not access to a headless CAM engine.
   [Problem, reasoning, support gaps and acceptance plan](REST_MACHINING_PLAN.md).
   Promote for a concrete workflow after higher-priority work; no implementation
   or machining acceptance is claimed by this planning entry.
7. **Deferred MCP/core capability: author Manual Profile holding-tab positions.**
   Current automatic authoring covers width, height, minimum/maximum count, distance,
   size threshold, the constrained lead-in flag and Square/Triangle/Skip style.
   Imported native Manual tabs remain inspectable and round-trip preserved, while
   fresh Manual authoring is rejected. Reopen only with a CamBam Plus 1.0 A/B fixture
   saved before and after moving/adding/removing tabs, so the exact native point
   collection, coordinate frame, identity/order behavior and generated toolpaths can
   be modeled and round-trip tested without inventing vendor XML.
8. **Future MOP semantic, native-export and framework/MCP parity audit.** Perform this
   as three bounded increments after the current SpiralMill prompt-free acceptance;
   do not treat every CamBam feature found as automatically in scope for
   implementation.
   **8a common-field slice implemented and automatically verified (2026-09-19).**
   All 18 modeled fields shared by Profile, Pocket, Engrave and Drill now have a
   durable semantics inventory and one declarative fresh-export policy. Explicit
   or deliberately context-resolved values use `Value`; unset optional depth/feed
   fields and empty headers/footers are omitted; imported template preservation
   remains separate. Undocumented depth/feed fallbacks and fresh unmodeled
   SpindleRange, StartPoint and Drill RoughingFinishing records were removed.
   Focused regressions cover all four families.
   **8a Profile/Pocket subtype and nested-policy slice implemented and automatically
   verified (2026-09-20).** All modeled Profile/Pocket subtype fields now have an
   executable declarative fresh-export policy and durable semantics inventory.
   Unset final depth increment is omitted; fresh lead records contain only modeled,
   mode-applicable fields and no invented mirrored lead-out; Automatic-tab children
   are emitted only in Automatic mode. Supported imported mode switches reconcile
   modeled siblings while preserving unknown native children. Manual tab transitions
   and unmodeled lead modes remain preserve-only. At that checkpoint, remaining 8a
   work was the Engrave/Drill subtype and method inventory below.
   Verification: 13 focused policy tests, 25 adjacent MCP/round-trip tests, and all
   245 repository tests pass (one existing Windows symlink-privilege skip).
   **8a Engrave/Drill subtype and method-policy slice implemented and automatically
   verified (2026-09-20).** All three Engrave subtype fields and all nine Drill
   method/subtype fields now have executable declarative fresh-export policies and a
   durable semantics inventory. Engrave no longer writes cached Default text for an
   unset final increment; its RoughingFinishing compatibility field is explicitly
   classified as having no promised Engrave toolpath effect. Drill fresh output is
   mutually exclusive across CannedCycle, SpiralMill CW/CCW and CustomScript, with
   Auto HoleDiameter as the sole policy-selected Default state. Supported imported
   method switches remove stale modeled siblings while retaining unknown extensions;
   unknown native methods are preserve-only. Empty CustomScript authoring is rejected.
   This completes the modeled field-encoding inventory in 8a; remaining MOP audit work
   is the separately bounded 8b parity matrix and 8c native evidence acquisition.
   Verification: 18 focused policy tests, 14 adjacent MCP/core round-trip tests, and
   all 250 repository tests pass (one existing Windows symlink-privilege skip).
   **8b framework/MCP parity matrix completed and automatically checked
   (2026-09-20).** The authoritative matrix in `MCP_CONTRACT.md` classifies all 60
   common/subtype field-policy slots plus enabled state, target sources and native-only
   content across core author/read/edit/preserve and MCP author/inspect/mutate. No MCP
   authoring promise exceeds the core writer. The audit corrected stale CannedCycle-
   only and universal-zero-clearance prose, retained CustomScript/live-group/unmodeled
   native exclusions, and identified two useful bounded gaps as 8d/8e below. Five
   executable parity tests fail if a modeled dataclass field or closed MCP author/
   inspection schema or target-kind rule drifts out of classification.
   Verification: 5 focused parity tests and all 253 repository tests pass with the
   existing Windows symlink-privilege skip; compileall and `git diff --check` pass.
   - **8a — field semantics and encoding audit, one MOP family at a time.** Inventory
     every modeled common and subtype field for Profile, Pocket, Engrave and Drill,
     including nested lead moves/tabs and method-specific Drill fields. For each,
     record its physical meaning, units, coordinate/reference frame, sign convention,
     valid range, applicability, dependencies, CamBam `Default`/`Value` inheritance
     behavior, when fresh XML should omit it, and preservation behavior after import.
     Replace scattered encoder decisions with one declarative field-encoding policy
     (or equivalently centralized reusable logic) that selects exactly one fresh-export
     disposition per field and applicable MOP mode:
     `Value` for user-decided or safety/determinism-critical values that CamBam must
     not replace; `Default` only for evidenced intentional inheritance/Auto semantics
     where an element must remain present, or when faithfully preserving imported
     native state (including a user-requested edit that deliberately clears a prior
     explicit override, if native A/B evidence shows omission is not equivalent); and
     omission for irrelevant, unset or safely profile-derived fields
     when absence is CamBam's prompt-free inheritance form. Keep imported-template
     preservation distinct from fresh authoring. The policy must cover nested
     containers, method switches and dependencies, prevent cached Default text from
     causing file-open prompts, and be directly regression tested.
     Compare implementation and tests with the CamBam Plus 1.0 manual, SDK/API
     reference and source-generated native XML. Explicitly examine current fixed or
     derived choices such as feed fallback, depth increment fallback, work plane,
     optimisation/velocity modes, spindle direction/range, tool profile/number,
     max crossover distance, roughing/finishing, cut ordering, stepover/fill styles,
     lead moves, headers/footers, StartPoint and Style inheritance. Classify every
     assumption as documented, native-observed, deliberate product policy, or
     unsupported/invented; remove or narrow unsafe assumptions and add regressions.
   - **8b - framework/MCP parity and capability-gap audit (completed 2026-09-20).**
     The checked matrix maps CamBam capability to core author/read/edit/preserve and
     MCP author/inspect/mutate support. It includes target-kind rules, explicit versus
     Auto values, import-only/preserve-only fields, adapter-pinned values, inspection
     rejection paths and method-dependent parameters. The comparison found no MCP
     promise the core cannot encode; it records hidden core capabilities, valid native
     records excluded from structured inspection and CamBam capabilities intentionally
     absent from both. Only the evidenced useful gaps became scoped 8d/8e increments;
     justified exclusions remain explicit rather than forcing nominal parity.
   - **8c — native evidence acquisition for unresolved semantics (completed
     2026-09-20).** Prepare a minimal
     requested-fixture list for the user to generate in CamBam Plus 1.0. Each A/B pair
     must change one property or state only and include the source `.cb`, exact UI
     setting, expected toolpath effect and whether a configured CAM Style/tool library
     is involved. Prioritize ambiguities that can make a file unsafe or prompt on open:
     field omission versus cached Default text, Style inheritance, lead-in/out
     dependencies and coordinate frames, finishing/final-pass controls, Profile and
     Pocket offset/fill variants, Drill CannedCycle/Spiral/CustomScript variants,
     tool-profile metadata, mixed target behavior and any native collections not yet
     modeled. Framework-generated XML is not evidence for an unknown native encoding.
     **Initial state evidence accepted 2026-09-20.** Four CamBam-native Profile files
     in `output/state-fixtures-01/` establish that editing one field changes only that
     field to `Value`; present Default records carry installation-local cached text
     and can prompt on a different installation, while omission resolves silently.
     A CamBam resave restored only the removed `HoldingTabs` container, retained 20
     other omissions and removed three empty Default records, so materialization is
     field-specific rather than universal. This confirms the framework policy:
     relevant decisions use `Value`; irrelevant/unset target-resolved fields are
     omitted; imported Default is preserved; explicit inheritance and evidenced Auto
     remain available. It also confirms that a fallback Python attribute for an
     omitted imported field is not an effective style value.

     The second native set in `output/state-fixtures-02/` completes the prioritized
     gaps using one baseline and four variants. Except for document name and CamBam's
     modification counters, each variant changes only its requested property. A
     Default cached FinalDepthIncrement `0` and explicit Value `0` produce identical
     motion; LevelFirst changes the two-target depth traversal from target-complete
     ordering to a level-wise serpentine ordering; CannedCycle Default
     RetractHeight `5` emits `G81 ... R5.0` while explicit Value `1` emits
     `G81 ... R1.0`; and the Value CustomScript edit expands literally from
     `(fixture-a x=60 y=40 z=-5)` to `(fixture-b x=60 y=40 z=-5)`. The generated
     headers identify CamBam's `Default` postprocessor, and the user identified the
     CAM style as `standard-mm`; style selection is not serialized in these MOPs.
     All five `.cb` files pass strict framework import. No further manual report is
     required: prompt behavior was already established by the first set, and reopen
     prompt status is not inferable from static files. Reopen 8c only for a concrete
     unsupported native encoding, a state-policy contradiction, or a safety-relevant
     MOP ambiguity; production execution remains out of scope.
   - **8d - preservation-aware structured MOP inspection.** After 8c establishes the
     remaining native state semantics, separate read fidelity from the canonical MCP
     authoring pins. Return modeled raw values plus field-scoped native state and
     applicability for valid imported Profile/Pocket/Engrave/Drill records without
     treating cached `Default` text as an evaluated CAM-style value. Unknown fields,
     Manual tab points, unsupported lead modes and literal CustomScript remain opaque.
     Acceptance requires alternate modeled pin values, inherited common/nested state,
     supported method dependencies and explicit/empty/group target sources to retain
     all independently safe typed values and round-trip unchanged; diagnostics must
     identify only unsupported fields instead of blanking the whole parameter record.
   - **8e - decouple harmless group membership from MCP MOP target eligibility.** A
     root supported-kind primitive currently loses typed geometry and its MOP loses
     structured parameters merely because `relationship_add_to_group` adds framework
     metadata. Split target eligibility from the geometry relationship predicate so
     group membership alone neither blocks author/retarget nor erases inspection.
     Acceptance covers every evidenced target kind before/after group add/remove,
     exact save/reopen targets and unchanged parameter records. Parent/child targets,
     non-similarity transforms and nonzero local Z remain excluded until their
     machining-coordinate semantics are separately evidenced.
   **Deliverables/stopping condition:** durable semantics go to `structure_spec.md`,
   MCP exposure and intentional pins to `MCP_CONTRACT.md`, dated comparisons and
   rejected interpretations to `REVIEW.md`, and only remaining prioritized work stays
   here. Stop when every currently modeled/exported MOP field and every MCP MOP field
   is classified, all dangerous unsupported assumptions found are fixed or explicitly
   isolated, representative source-native round trips pass, and unresolved questions
   have concrete fixture requests and reopening criteria. Production G-code safety
   and exhaustive CamBam feature parity remain outside this audit unless separately
   authorized.
9. **Future feeds, speeds and engagement planning helpers** (requested 2026-09-20).
   Replace ad hoc machine-specific guesses with pure, unit-explicit calculation and
   recommendation helpers; never restore an implicit MOP export fallback. The removed
   `round(350 * abs(target_depth) + 6500)` rule is retained only as historical evidence:
   it was useful for one user's machine/material context but has no general machining
   basis. Authoritative formula references agree on the stable relationships: table
   feed is chip load per tooth times RPM times effective flute count; spindle speed is
   derived from cutting surface speed and effective cutter diameter; engagement,
   feed and depth determine material-removal rate; and specific cutting force can
   estimate power and torque. See Sandvik Coromant's
   [metric milling formulas](https://cdn.sandvik.coromant.com/files/sitecollectiondocuments/services/metal-cutting-e-learning/formulas-and-definitions/formulas-and-deinitions-for-milling-metric-enu.pdf)
   and Kennametal's
   [speed/feed formulas](https://www.kennametal.com/us/en/resources/engineering-calculators/miscellaneous/speed-and-feed.html).

   Treat recommended surface speed, chip load, axial/radial engagement and entry
   limits as sourced starting data, not universal formulas. They depend on the exact
   work material/condition, tool substrate/coating/geometry and diameter, flute count,
   operation and engagement, coolant/chip evacuation, stickout/runout, workholding,
   machine rigidity, spindle power/torque and RPM/feed limits. LMT Onsrud, for example,
   publishes separate [routing recommendations by material](https://onsrud.com/Forms/Cutting-Data-Recommendations.asp),
   while Harvey Performance warns that both excessive and insufficient chip load can
   damage cutting behavior in its
   [speeds and feeds guidance](https://www.harveyperformance.com/in-the-loupe/speeds-and-feeds-101/).
   Do not copy a vendor table into a generic material label without compatible usage
   terms, provenance, applicable tool family and version/date.

   Implement as bounded subincrements:

   - **9a — dimensional formula kernel and constraint solver.** Add deterministic
     helpers for surface-speed/RPM conversion, chip-load/feed conversion, material
     removal rate and optional cutting power/torque. Accept `mm` and `in` explicitly,
     reject dimensionally invalid or conflicting inputs, preserve exact fixed user
     values, solve only identifiable missing values, and report assumptions,
     active constraints and capped results. Model axial depth/stepdown and radial
     engagement separately. Target depth determines total travel/pass planning, not
     feed directly. Radial chip-thinning or circular-interpolation compensation must
     be separate evidenced models, not hidden multipliers.
   - **9b — recommendation profiles and extension API.** Define immutable tool,
     material and machine capability records plus a pluggable strategy interface.
     Support fixed user rates, user-supplied chip-load/surface-speed tables and custom
     pure calculation callables before shipping any curated catalog. Every suggested
     value must carry provenance, units, applicable input range and whether it is a
     manufacturer starting point, measured shop policy or user override. Plunge,
     ramp and helical-entry recommendations require an entry-capable tool and their
     own sourced rule/profile; do not infer plunge as a universal percentage of cut
     feed. Define serialization/versioning only after a persistent owner is chosen.
   - **9c — pass planning and optional MCP exposure.** Integrate the existing
     `machining_calculate_depth_increment` final-stock/cut-through planner with a
     caller-supplied safe axial-depth constraint from 9a/9b. If useful to named-client
     workflows, expose a read-only planner returning candidate RPM, cut/plunge feed,
     stepdown, stepover, chip load, MRR and power/torque diagnostics; it must not
     mutate a document or silently author a MOP. User-confirmed/fixed values win, and
     underdetermined inputs return the missing requirements rather than guessed values.

   Acceptance requires formula inverse/property tests, metric/imperial equivalence,
   partial-input and overconstraint tests, machine-limit clamping diagnostics,
   provenance round trips for user profiles, comparison against multiple official
   worked examples, and explicit unsafe/unknown-data cases. Document that results are
   starting recommendations requiring tool-manufacturer guidance, machine limits,
   workholding review and supervised test cuts; production machining safety is not
   established by the calculator. Keep this after the active 8a serialization audit:
   it is a distinct planning subsystem and depends on an explicit units/tool/material/
   machine profile contract. Reopen sooner only for a concrete workflow that supplies
   those inputs and acceptance data.

This supersedes the former five broad increments; their pending scope is retained
above. Detailed contracts remain in `docs/structure_spec.md`, and review evidence in
`docs/REVIEW.md`. No repository-linked issue tracker was found.

## Blockers and decisions

No blocker for the completed MOP ownership/interchange scope. Production
toolpaths and complete `.cb` format coverage remain outside that acceptance.
The old framework API and pickle format did not constrain the redesign.
MCP 4e still requires the named OpenCode agent workflow and CamBam acceptance. The adapter
retains 2026-07-28 as its target and also supports 2025-06-18 and 2025-11-25;
reprobe the clients during 4e and retire an older version only after deployed
clients no longer need it. No blocker prevents 4d implementation; keep broader
CAD/CAM coverage explicit and test each advertised family before discovery.

The second OpenCode run also created the final outside cutout before its enclosed
hole operation. Server instructions and all four MOP descriptions now state that MOPs
append in call order and that enclosed/internal or non-releasing detail work normally
precedes a through-cut Outside Profile that releases the containing part. Contract v1
still cannot reorder an existing sequence. Reopen a focused reorder-tool increment via
the public `assign_mop_to_part` API if another named-client run misorders operations or
needs to repair an existing file; do not expand 4e merely for hypothetical ordering.

## Completion and verification

- MOP group-source compatibility: historically characterized with six tests;
  the redesign supersedes its old runtime ownership contract.
- Working agreement: implemented; documentation checks recorded in the review.
- Parent XML slice: implemented and automated checks complete; nine regression tests.
- Parent-cycle rejection: implemented and automated checks complete.
- Broader product baseline: limited local checks only.
- User/CamBam acceptance: parent, full-bake, global-transform and Rect-bake
  synthetic display cases were accepted. MOP A/B and native C/D display/property
  interchange were accepted in CamBam Plus 1.0. Production toolpaths and broader
  `.cb` coverage were not validated.

The completed MOP implementation and automated verification include a final
suite reports 65 passed tests in
`output/mop-core-checks-ncsbkpgz/suite.log`; compileall for
`cambam_builder`, `legacy_cambam_builder`, `tests` and `demos` exits 0. The A/B
MOP generator exits 0 and verifies counts, operation targets, depths and feeds.
Lead XML inspection also confirmed rectangle coordinates, triangle vertices
and drill points. The
context regression also reproduces and guards the former lost global
`Style`/`StyleLibrary` state. See
[MOP review evidence](REVIEW.md#mop-core-ownership-and-interchange-redesign).

[Verification commands](DEVELOPMENT.md) and [work lifecycle](WORKFLOW.md)
are authoritative. MOP ownership, supported interchange, curved geometry bounds
(1a), copy/transfer (1b) and shape parity (2) are recorded above with their
verification and acceptance state.

This is a good fresh-session breakpoint: the Text/roughing-clearance follow-up is
implemented, verified and recorded. The distinct remaining 4e work is the
named OpenCode agent workflow, not a repeat of these native encoding checks.
Suggested commit: `fix: expose Text machining and roughing clearance`.
