# CamBam CAD/CAM Framework – Core Project Structure and Relationship Management Specification

Section 0 describes the implemented architecture; section 5's copy/transfer contract
is implemented in the current runtime, while the other sections describe intended
design, not a verified inventory of implemented behavior. See [current status](PROGRESS.md) for implementation gaps and
the [topic map](README.md) for documentation ownership. MOP target ownership is defined below; the development compatibility policy governs API changes.

This specification describes the core architecture for the CamBam CAD/CAM framework. In this design, all relationships between entities (primitives, layers, parts, and machine operations (MOPs)) are maintained in a central registry managed by the project object. This approach minimizes duplication of relationship data in the individual entities and provides a single source of truth for linking. It also simplifies propagation of transformations, transferring of entities between projects, and robust XML serialization.

---

## 0. Implemented architecture and change ownership

This is a local Python library with an optional stdio MCP adapter and no database
or frontend. The public library entry point is `CamBamProject`, also
exported as `CBProject`. Package declarations include the modern and legacy
packages, the detached `cam_core`, `cam_extensions` and CamBam integration subpackages; `inactive/` and demos
are outside that runtime package list.

| Owner | Implemented responsibility | Start here when changing |
| --- | --- | --- |
| `cambam_builder/native/project.py` | UUID entity registries, identifier lookup, ordered layers/parts/MOPs, relationship updates, transform orchestration and persistence | Public creation/query/mutation APIs and relationship invariants |
| `cambam_builder/native/transfer.py` | Transactional primitive-tree copy/transfer staging, identity mapping, collision validation and relationship publication | Copy/transfer semantics and atomic registry updates; inspect project wrappers and tests |
| `cambam_builder/native/core.py` | Shared vertex/bounds/numeric foundations, `CamBamEntity`, and `Primitive`, including local and parent-composed transforms | Identity, base inheritance, shared validation, bounds foundations, or primitive transform context |
| `cambam_builder/native/cad.py` | `Layer` plus ordinary Pline/Circle/Rect/Arc/Points/Text geometry and XML behavior | Ordinary CAD fields, geometry, bounds, baking, or entity XML |
| `cambam_builder/native/region.py` | Owned Region contours, planar curved topology validation and typed Region XML; depends directly on core and ordinary CAD owners | Region geometry and interchange; project and reader use this owner |
| `cambam_builder/native/cam.py` | `Part`, MOP classes, and MOP XML path/encoding policy inventories | Part stock/nesting or MOP parameters and XML policy |
| `cambam_builder/cambam_entities.py` and nine old root module paths | Compatibility/discovery imports of canonical native objects; no native implementation remains at root | Preserve public and documented direct imports; implementation modules import owners directly |
| `cambam_builder/native/transformations.py` | NumPy matrix construction, composition, decomposition and XML matrix conversion | Numerical conventions; inspect entity and project callers together |
| `cambam_builder/stock.py` | Exact conditional disk-sweep occupancy, remaining-section bounds and supplied section-motion verification | Horizontal cuts and explicit travel inside rectangular stock/target; no generated or native path integration |
| `cambam_builder/cam_core/` | Document-independent CAM planning, motion and stock analysis; `replay.py` owns shared ordered XYZ/cut-sweep values and `curved_region.py` owns bounded circular-arc access/rest approximation | Exact nominal and curved endmill stock, analytic slot and placed mixed trace; no native entity, XML, MCP or machine-output dependency |
| `cambam_builder/cam_extensions/strategy.py` | Deterministic selection among separately audited ordered routes, including partial and infeasible outcomes | Policy over evidence records; no XML or native entity dependency |
| `cambam_builder/planar.py` and `_planar_shapely.py` | Detached nominal planar values, error/provenance policy, analytic feasible centers and private optional GEOS adapter | Pure planar geometry; no document or stock/path ownership |
| `cambam_builder/machining_calculations.py` | Pure unit-explicit milling formulas, partial-input constraint solving and derived RPM/feed machine caps | Arithmetic planning kernel; composed by the separate pass planner |
| `cambam_builder/machining_recommendations.py` | Immutable tool/material/machine contexts, provenance-bearing recommendations, user diameter tables and pluggable pure strategies | Recommendation selection only; contains no curated catalog, persistence, document mutation or safety claim |
| `cambam_builder/machining_planning.py` | Pure through-cut pass balancing and composition of recommendation profiles with formula/machine diagnostics | Candidate planning only; the existing MCP depth tool delegates here, while full profile construction remains a direct-Python API |
| `cambam_builder/native/writer.py` | XML ID assignment and layer/part traversal; delegates individual encoding to entities | Output structure and reference resolution |
| `cambam_builder/native/reader.py` | XML parsing, entity reconstruction, ID mapping and deferred parent/MOP linking | Import defaults, malformed data and round-trip reconstruction |
| `cambam_builder/integrations/cambam/` | Native `.cb` input/attachment, M2 curved candidates, actual posted MOP-series normalization and bounded posted-stock comparison; depends on native model and detached CAM values | Bridge between native documents, generated motion and CamBam output; no source model ownership |
| `cambam_builder/integrations/direct_*.py` | Bounded headless V and RC01 reference-dialect writers, parsed-output audits and evidence manifests | Output adapters; consume detached plans/traces and preserve their target verifiers |
| `cambam_builder/integrations/{uccnc_m5,uccnc_reader,grbl_m5_reader,m5_portability,m5_decoded}.py` | Bounded controller fixture emission, independent complete-byte dialect decoding, common decoded-stage values and shared T1/T3 stock audit | Consume the detached M4 plan; no controller syntax in `cam_core` |
| `cambam_builder/__init__.py` | Public alias and version | Import surface and version metadata |
| `cambam_builder/mcp_adapter/` | Optional local stdio launcher, SDK protocol boundary, volatile documents, retry ledger, schema validation and workspace I/O | [MCP contract](MCP_CONTRACT.md); `server.py` owns wire behavior, `service.py` owns application state, `paths.py` owns filesystem policy |

### Package organization decision and migration plan

**User preference recorded 2026-09-23:** the package layout must make native
CamBam document capability, reusable CAM core, extended machining capability and
their adapters visibly distinct. A root file named `region.py` is the native
CamBam Region entity, not a general planar region. Historical `cambam_*`,
`cam_*` and unprefixed root filenames obscure that distinction. New features
must choose an owner from this map before adding a module; the repository root
is a compatibility/import surface, not the default home for new behavior.

| Target owner | Responsibility and examples | Dependency direction |
| --- | --- | --- |
| `native/` | CamBam project, identity/relationships, CAD shapes including `Region`, Part/MOPs, transforms and `.cb` XML reader/writer. This is native document *intent* and interchange, not stock replay. | May use shared mathematical values; never import strategies or an output adapter. |
| `cam_core/` | Document-independent geometry, tool/motion values, stock and verification predicates, numerical bounds. No `.cb`, MOP, MCP or controller knowledge. | Inward-only foundation; may use declared numerical backends. |
| `cam_extensions/` | Optional policy and generated strategies: machining recommendations/pass planning, rest machining, V-carving, combined strategies and bounded reference jobs. RC01's exact recipe is a reference job, not a generic core primitive. | Depends on `cam_core`; no native model or XML imports. |
| `integrations/cambam/` | Explicit normalization from native intent, candidate document attachment, CamBam-posted motion readers and output evidence. RC01's current adapter and reader live here. | Depends on native, core and extensions; validates changes after lowering. |
| `integrations/` controller adapters (M5 bounded fixtures) | Dialect/profile capability checks, emission, independent decoding and setup/transition effect normalization. UCCNC and Grbl v1.1 fixtures currently live directly here; a subpackage needs measured breadth. | Depends on detached values; direct output must not require native XML or a CamBam private helper. |
| `mcp_adapter/` | Optional client protocol, document sessions and workspace transport. | Calls the owners above; does not own their domain rules. |

Current root files classified by that target map:

| Current files | Target owner |
| --- | --- |
| `cambam_project.py`, `cambam_transfer.py`, `entity_core.py`, `cad_entities.py`, `region.py`, `cam_entities.py`, `cad_transformations.py`, `cambam_reader.py`, `cambam_writer.py` | Moved to `native/` on 2026-09-24; the old root paths are compatibility imports |
| `planar.py`, `_planar_shapely.py`, `stock.py`, reusable formulas in `machining_calculations.py` | `cam_core/` |
| `machining_recommendations.py`, `machining_planning.py`, future rest/V-carve strategies | `cam_extensions/` |
| `cam_core/rc01.py` | Current pure reference implementation; exact recipe later in `cam_extensions/reference_jobs/`, reusable verifier/motion values stay in `cam_core/` |
| `__init__.py`, `cambam_entities.py` | Root public/compatibility facade; implementations move behind it |

This is one distribution, not a plugin architecture. Under a package, use the
domain name (`project.py`, `region.py`, `stock.py`, `rest.py`, `vcarve.py`) rather
than repeating its package prefix. Keep `cambam_` only where a root compatibility
name or a cross-system adapter needs to identify CamBam explicitly. The pure
RC01 generator/verifier currently remains in `cam_core/rc01.py`; its shared
ordered motion/stock contract now has a second consumer in `cam_core/replay.py`,
while the RC01 process and residual oracle stay in that reference module.
Relocate the exact recipe only with a concrete consumer and complete dependency
checks. Do not create empty target
packages as placeholders.

Migration is staged around executable slices:

1. **Current RC01 bridge:** keep `cam_core.rc01` pure; put `.cb` normalization,
   candidate attachment and post comparison in `integrations/cambam/` and include
   that package in the wheel. This is implemented.
2. **Native owner consolidation:** move project/transfer, entity and Region,
   Part/MOP, transform, reader and writer implementations from the root into
   `native/` together. Preserve the public `CBProject` and documented old import
   paths while callers migrate. Check representative Region, MOP identity,
   target-reference and stock-offset XML round trips plus clean-wheel imports.
   Implemented 2026-09-24 after the RC01 output contracts were fixed. The nine
   canonical implementation modules are under `native/`; root paths forward
   imports to the same classes and functions, and runtime callers use native
   owners directly. The wheel declares `cambam_builder.native`. Existing
   same-code-version pickle snapshots use the new canonical module names;
   old-pickle migration is outside the stated snapshot contract.
3. **Detached and extended consolidation:** move root `planar.py`, `stock.py`
   and reusable machining math into `cam_core/`; place recommendation/pass
   planning and later rest/V-carve strategies in `cam_extensions/`. The shared
   replay contract has its second consumer in the cone slot; RC01's exact recipe
   and specialized process oracle still await relocation. Check dependency
   direction, focused suites, wheel contents and imports outside the source tree
   when that migration is justified by an output or package-consumer need.

Each move updates the owner table, callers, packaging, runbook and review evidence
in the same increment. Remaining root detached and policy modules are authoritative
until a separately justified migration; the target table does not imply capabilities
or imports that already exist.

### Data flow and relationship boundaries

1. Callers create and mutate entities through `CamBamProject`. UUIDs key the
   registries; the identifier registry provides human-readable lookup. The project
   maintains both directions of layer membership and parent/child links, group
   indexes, and part/MOP association and order. Update these together through the
   owning APIs rather than editing individual dictionaries in application code.
2. Primitives hold geometry and an effective matrix, plus a weak project reference
   used to compose ancestor transforms. Groups are also represented on entities;
   MOP targeting lives solely in the project `_mop_targets` registry. MOP entities
   hold machining parameters without target or part references.
3. `save()` / `export()` delegate to `save_cambam_file()` and `build_xml_tree()`.
   Entities encode geometry and metadata; primitives export world matrices. The
   reader constructs entities, then resolves references. See the review for known
   fidelity defects; serialization success alone does not establish equivalence.
4. `save_state()` / `load_state()` provide an optional trusted, same-code-version
   Python pickle snapshot and restore primitive project links after loading. This is
   not interchange, durable storage or a versioned migration contract. Trust
   requirements live in the runbook.

The reader's `PRIMITIVE_TAG_TO_CLASS` and `MOP_TAG_TO_CLASS` are the executable
supported-tag inventory, not a claim of complete CamBam coverage. Consult those
maps and corresponding entity encoders before adding a type.

### Execution boundary and detached CAM core

The current document pipeline authors native geometry/MOP instructions and exports
`.cb`; CamBam then generates toolpaths and posts G-code. Document fidelity is not
motion equivalence or evidence of removed stock. Detached planar and stock helpers
below do not change that execution boundary: supplied section motions are not an
implemented general XYZ generator, route optimizer or postprocessor.

The [execution architecture proposal](REST_MACHINING_PLAN.md#execution-architecture-refinement---2026-09-23)
defines a document-independent motion/stock core with separate strategies,
verification and CamBam/direct-G-code output adapters. Bounded M1-M4 slices
implement this separation; general controller output remains M5 work. Native-MOP authoring remains
independently useful. Execution authority, manual-edit invalidation, output
acceptance and delivery decisions belong to that active plan; no future
API or native path-equivalence claim is implied by this specification entry.
The accepted target is an embeddable capability library: consuming applications
own workflow sequencing and synchronization, while the core owns input validity,
derived-result freshness and machining evidence. Import, analysis, regeneration
and export remain separable capabilities; see the plan's
[caller-owned workflow contract](REST_MACHINING_PLAN.md#caller-owned-workflows-and-reusable-capabilities).

The detached implementation has `cambam_builder.cam_core`. Reusable
toolpath/stock values and verification belong there; optional rest machining,
V-carving, combined strategies and policy planning belong in `cam_extensions`
under the [organization plan](#package-organization-decision-and-migration-plan).
Neither owner imports `CamBamProject`, native CAD/MOP entities, XML I/O or MCP.
Native `.cb` input/output and future controller posting use explicit adapters
outside those packages; adapters normalize source data and separately validate
emitted motion. The controller-neutral ordered plan and stock verifier feed
capability-declared output adapters. An adapter owns one named dialect,
transport and setup, such as UCCNC, LinuxCNC or Grbl; the core does not assume
that every controller accepts the same G-code or even a file. UCCNC is the
user's production controller and first output fixture, not an architecture
dependency. An independent reader must decode every emitted command stream
back to ordered motion and state before stock replay. An interpreter or
controller runtime trace, when available, supplies a separate execution-level
check and cannot certify another dialect.
The core motion uses a physical tool-tip datum in the **intended CAM frame**.
Imported operations resolve that frame and path from preserved MOP semantics
and actual posts; framework-generated paths use their validated plan. A
Part/Machining stock object is optional for importing a native project; it
contributes to path values only where CamBam `Auto` properties
request it. Stock/fixture geometry can be supplied later for stock-dependent
verification without rewriting explicit MOP coordinates. A setup separately
provides an explicit CAM-to-controller work-frame map and declares the active
work and tool-length offsets. Identity mapping preserves an imported program
by default; it does not attest a machine's physical setup. An adapter applies
only declared transforms; the verifier inverts them before comparing decoded
motion with the original path. A stock object's
top elevation alone never commands a Z shift. Work-coordinate touch-off,
tool-table length compensation, probing and preset tools are distinct ways to
establish the required effective tip datum. Missing stock leaves stock checks
unassessed rather than invalidating import; unresolved `Auto` values remain
unresolved until sufficient source or actual-post evidence exists. Generated
rest paths require a caller-supplied initial-stock model and prior-motion
evidence even when the native project has no stock object.
Tool changes are plan events with old/new tool, safe pre/post tip, spindle,
effective offset, stock-lineage and completion/abort requirements. Static
verification records assumed postconditions; actual execution needs satisfied
conditions before cutting resumes. A per-transition policy
selects manual or automatic actor, in-program or split-program handoff, and
tool measurement/offset method. The controller adapter maps the policy to
supported commands, macros or sender actions and rejects unknown effects.
Different transitions in one job may use different policies. The first UCCNC
slice uses separate per-tool files with a checked manual handoff. The user's
manual `M6` form remains a supported policy once its macro/resume behavior is
declared. Automatic changer effects must be modeled and checked, with physical
changer acceptance kept separate. Same-tool CamBam MOPs may be grouped in
Parts for separate posting, but each final post and cross-file stock handoff
still needs an audit.
Existing root-level `stock`, `planar` and machining modules
remain active owners until their staged migration.

### Mediation invariants and evidence contract

**Design contract, 2026-09-26; bounded M5 portability fixtures implemented.** These rules apply
at native/core/controller boundaries. They do not widen the currently verified
fixed-axis milling domain or promise universal controller/kinematics support.

| Representation | Authority and required boundary behavior |
| --- | --- |
| Native document | Owns authored intent, identity, ordering, enabled state and parameter states. Preserve absent stock and `Auto`/`Default`/explicit values; a resolved value must not silently replace its authored state. Preserve unsupported content through existing XML fidelity mechanisms and report any fidelity limit separately from execution support. |
| Normalized input / generated plan | Owns explicit units, frames, tool geometry, resolved inputs and ordered intended motion. Record which source/setup resolved each value. Native MOP parameters alone do not reproduce CamBam's path algorithm. |
| Emitted artifact / decoded trace | Final bytes and declared external effects own posted-motion evidence. Independently decode all files, startup/end travel and transitions; replay every tool's actual decoded path. A planned path or preview cannot supply missing posted motion. |
| Runtime / physical setup | Telemetry or an attributable operator/machine assertion owns observed installation, measurement and completion. Offline verification proves behavior conditional on declared setup; it cannot manufacture an observation. |

**Frames.** Name drawing/entity, resolved program, controller work and machine
frames separately. Resolve native transforms, machining origins and nesting
where supported before applying a program-to-work map. Identity by default
means no *additional* program-to-work conversion; it never means ignoring a
native origin or applying it twice to an already posted path. Reject unsupported
mapping for the requested analysis/output while retaining the native document.
Stock placement and effective work/tool offsets participate in the same frame
composition. The bounded M5 fixtures support explicit translations, with no inferred
translation from stock metadata. General rotations/kinematics need another
named capability and fixture.

**Order and transitions.** Keep operation identity and predecessor dependencies
through file splitting, Part attachment and tool changes. T1/T3/T1 remains that
order; grouping all T1 operations together is a scheduling change requiring new
stock/access evidence. Selecting a tool, installing it and establishing its
effective offset are separate states. A pause or file boundary does not prove
any of them. A transition records required pre/post conditions, modeled travel
and effects, and the source of any observed completion. Static analysis may
continue under explicit postcondition assumptions, recorded in its result;
execution release requires those conditions to be satisfied by the caller's
operator/machine workflow. The library need not become a machine-control service.

**Results.** Return evidence per capability (document fidelity, input resolution,
motion equivalence, stock/access/residual, runtime parity), with status, scope,
reason, assumptions and input/artifact fingerprints. Distinguish `pass`, `fail`,
`not_evaluated` and `unsupported`; unresolved inputs prevent the dependent check.
Never collapse these into an unqualified `safe` or `valid` flag. Missing stock
leaves import and supported path comparison available. Unknown macro effects
block motion certification, even when the macro text can be preserved natively.

**Freshness and numbers.** Bind evidence to source, enabled operation order,
resolved setup/frame, tools, incoming stock/prior trace, profile/macro effects,
emitted bytes, verifier version and numerical policy. A dependent edit invalidates
that evidence. Conservatively invalidate when dependencies are unknown; a
cosmetic edit may retain semantic evidence only where normalization proves it.
Record linear/angular/rounding and stock-bound tolerances with units before
testing. Motion matching within tolerance does not authorize stock replay of
the planned coordinates: use decoded coordinates, including allowed rounding.
Do not derive the reader's expected values by calling the emitter, or infer
cut/access roles from comments without checked operation/segment correspondence.

Use small immutable records at these boundaries and the existing callable
strategy/adapter pattern. Persist a versioned, hash-bound evidence manifest for
reproducible audits. Introduce only fields exercised by the current slice; no
plugin registry, workflow engine or generalized native path interpreter is
required. The [M5 implementation packet](REST_MACHINING_PLAN.md#m5-implementation-packet)
owns the sequence, negative cases and stopping conditions.

The native `Part.stock_present` flag records XML presence independently of
placeholder dimensions. Imported absence survives save/reopen and relationship
copy/transfer; newly authored Parts retain their default explicit Stock.
Explicit MCP stock edits opt an imported stockless Part into Stock emission.
Unchanged `Auto`/`Default`/explicit MOP XML states remain authored states,
not resolved path values. The bounded native-series parser counts a positive-Z
cut relative to its MOP stock surface and leaves coordinates unchanged.
`NativeSeries.parsed_evidence()` reports source and posted-motion parsing
separately from stock/access/residual `not_evaluated`; parsing alone grants
no stock certificate.

`integrations.m4_curved_workflow.load_selected_plan` exposes the accepted
source-bound rounded raster plan, supplied T1 trace and initial tip to both
controller adapter. `cam_core.v_region.complete_motion` assembles safe travel
around the detached V plan. `integrations.uccnc_m5` emits separate T1/T3 files
for one declared G54, metric, absolute, exact-stop, no-length-compensation
profile; `integrations.uccnc_reader` independently decodes its strict G0/G1
subset and rejects unknown commands. The manifest pins source/prior/plan,
profile/setup, ordered tool handoff and file hashes. Its auditor compares every
decoded move to the intended path within 0.000051 mm, reconstructs T1 stock and
T3 rounded V paths from decoded coordinates, then computes section and volume
bounds on that decoded chain. The per-tool installation/touch-off conditions
remain explicit assumptions. `integrations.grbl_m5_reader` separately decodes
the bounded Grbl v1.1 whole-program subset, including ordered `M0` pauses and
per-stage `G49`/`G43.1` state. `integrations.m5_portability` verifies manual
and mixed-policy fixtures with the same `uccnc_m5.audit_decoded_pair`
stock/access/residual audit; a final safe T1 stage and synthetic changer
effect travel are checked without inferring physical completion. The manual
fixture resolves a +4.5 mm CAM surface through a declared -4.5 mm work map;
the mixed fixture uses fixed G54 and external table-derived length values.
`integrations.m5_decoded` owns only shared decoded-stage values; dialect
readers do not depend on one another. Generic native MOP generation, further
controller dialects, controller runtime parity and physical machine acceptance
are outside this bounded fixture set.

### Directional analytic stock section bounds

`cambam_builder.stock` owns a detached, backend-independent consumer:
`bound_horizontal_sweep(SectionRectangle, HorizontalSweep, frame_id=...,
section_z_mm=..., radius_min_mm=..., radius_max_mm=..., position_error_mm=...)`.
It accepts one axis-aligned exact rectangular initial stock and one explicitly
supplied horizontal center segment (including a zero-length disk placement).
Used alone, that stock is also the required removal target; the target-aware
composition below validates a distinct required target. All coordinates, radii
and errors are millimetres in one caller-named Cartesian frame at an explicit
fixed Z. No native geometry,
unit conversion, frame inference, nominal planar result or general topology is
accepted. Invalid, unsupported or boundary-violating inputs raise `ValueError`
without returning partial bounds. No new dependency or MCP/native API is added.

The physical model requires complete parameter-wise traversal: at each nominal
segment position the actual cutting disk center lies within Euclidean distance
`e`, and its radius lies in `[rmin, rmax]`, with `0 < rmin <= rmax` and `e >= 0`.
There is no additional cutting motion in this section. Initial stock and its
placement are exact; stock/setup uncertainty must not be silently inferred from
`e`. Radius variation and center error may vary along the sweep. Z uncertainty,
entry, links, holders and machine/process feasibility are outside this model.

For nominal segment `P`, the returned capsules are
`C_lo = P + disk(rmin-e)` and `C_hi = P + disk(rmax+e)`.
When `e > rmin`, `C_lo` is empty (`None`); equality retains the line/point set.
For every parameter, the triangle inequality places the smaller nominal disk
inside every allowed actual disk and every actual disk inside the larger nominal
disk. Taking unions proves `C_lo subset C_true subset C_hi` even for varying
errors/radii. The complete traversal assumption is essential to the lower bound.

The outer capsule's four exact extrema must lie in the rectangle; otherwise the
whole operation fails. Rectangle boundary contact is allowed, and the exterior
is protected. Stock bounds (also required rest when stock equals target) reverse
removal direction:
`remaining_lower = S0 \ C_hi`, `remaining_upper = S0 \ C_lo`.
These are immutable analytic set expressions, not tessellated polygons. Removal
uses closed-set membership; subtraction excludes the removed boundary. Only the
lower removal bound represents conditionally guaranteed removal. Feasible-center
area is never substituted for a supplied swept segment.

Arithmetic converts finite int/float/Fraction inputs to exact rational values
(float means its exact binary value, not an inferred decimal). Membership and
containment use squared distances and rational comparisons without tolerance,
rounding, polygon offsets or backend numeric error. Capsule area is
`2*r*segment_length + pi*r*r`; `area_interval` encloses it with rational
`333/106 < pi < 355/113`, and subtraction reverses those bounds. The lower rest
area limit is clamped to zero. These scalar intervals are deliberately coarse
rigorous enclosures, not an error-budget claim. No computed square root or pi
approximation participates in containment.

`SweepBounds` retains the frame, section, input stock/sweep, radius interval and
position error with evidence class `conditional_analytic_section_bounds`.
It is a programmatic geometric result, not an authenticated certificate, actual
machine execution evidence, or authorization for known-free travel. Callers must
use the validating function; manually constructed records establish no proof.

`compose_sweep_bounds(stock, sources, frame_id=..., section_z_mm=...,
grid_size=32)` accepts a finite iterable of validated `SweepBounds` with the same
exact stock, frame identity and Z. It reconstructs each source to reject altered
bounds, preserves the ordered source tuple (including duplicates) and retains each
pass's radius/position uncertainty. Empty input removes nothing. For every prefix,
`removal_lower = union(C_lo)` and `removal_upper = union(C_hi)`; `RemainingSection`
subtracts the opposite union. No independence of errors between passes is assumed.
Membership remains exact, including zero-area line/point guarantees and excluded
removal boundaries. Composition does not imply cutting travel between segments.

`CapsuleUnion.area_interval` partitions the exact stock rectangle into
`grid_size ** 2` equal rational cells (positive integer, default 32 per axis).
A cell contributes to the lower area if one positive-radius capsule contains all
four corners (convexity); it contributes to the upper area if its exact minimum
distance from any positive-radius center segment is at most that capsule radius.
Each cell counts at most once per bound, regardless of overlap or duplicate passes.
Zero-radius components have zero area but retain membership. Thus aggregate area
is conservatively enclosed without summing overlapping removal. These intervals
can be much wider than single-capsule analytic intervals; they are not exact union
areas or a requested accuracy guarantee. Increasing the grid costs O(n*n*passes)
exact arithmetic and constant auxiliary cell storage. Integer-multiple refinement
can only tighten intervals. For a fixed stock/grid, adding passes makes both
remaining-area endpoints nonincreasing, as well as shrinking remaining membership.
Changing grid size between prefixes does not guarantee endpoint monotonicity.
Use finer grids only when a consumer needs tighter area evidence; polygon output,
area-driven stopping tolerances and general topology remain outside this slice.

`SectionTarget(stock, outer, island)` separates initial exact rectangular stock
from an exact required removal rectangle with one strictly interior rectangular
island. `outer` must lie within stock; the island must have positive clearance
from every outer edge. Target membership includes the outer and island boundaries
but excludes the island's open interior. This permits exact wall tangency, while
any cutter occupancy in the island interior or outside the outer rectangle is
protected-material overrun. The area is `outer.area - island.area`; boundary
membership has zero effect on area but does affect pointwise rest membership.

`compose_target_rest_bounds(target, sources, frame_id=..., section_z_mm=...,
grid_size=32)` first revalidates and composes the same ordered `SweepBounds`
sources against `target.stock`. It then requires every outer capsule to fit in
`target.outer` and have squared segment-to-island distance at least its squared
radius. Equality is allowed tangency; a rational amount of penetration fails the
whole call. Since each true sweep lies inside its outer capsule, this protects
the original target exterior and island for every allowed error/radius, regardless
of stock already cleared. A later sweep can cross a rest/cleared-space interface:
that interface is not a protected design boundary.

`TargetStockBounds.stock_bounds` retains whole-stock cumulative removal and
remaining-stock bounds with source provenance. Its `rest_lower` is the original
target minus outer removal union; `rest_upper` is the original target minus inner
removal union. Both use exact analytic membership and the same closed-removal
boundary convention as `RemainingSection`. Their area intervals subtract the
union's conservative grid interval from exact target area, clamping the lower
endpoint to zero. Empty sources leave stock and target intact. Directly
constructed result records are data, not validated evidence. Stock/rest
composition alone does not infer connecting travel or entry.

`verify_section_motion(target, paths, frame_id=..., section_z_mm=...,
grid_size=32)` validates ordered, caller-supplied `SectionMotionPath` records.
Each path has one fixed radius interval and position-error envelope, one explicit
`entry_kind`, and contiguous, directed horizontal `SectionMotionSegment` cuts or
travels. Consecutive segments in a path must meet at the same exact XY point;
different paths may represent different tools or separate entries. Every cut
reuses `bound_horizontal_sweep` and requires its outer capsule inside the original
target, regardless of earlier removal. Accepted cuts are appended in traversal
order and the final `TargetStockBounds` is derived from exactly those cuts. Travel
adds no removal. The function is atomic and returns a
`conditional_analytic_section_motion` result only if all entries and segments pass.
The verifier reconstructs supplied motion records before use, rejecting altered
fields or segment kinds rather than trusting a constructed record.

An entry is the first segment's start point and its inflated disk. `cutting`
entry requires a cut first and verifies that disk inside the original target.
`cleared` entry cites a zero-based *earlier cut* index and requires the disk inside
that cut's guaranteed inner capsule. `outside_stock` entry requires the disk to
be strictly disjoint from the exact closed initial stock. Each travel segment
either cites one earlier cut via `cover_source_index`, or, with no index, is an
explicit outside-stock approach strictly disjoint from the initial stock.
Indices refer to cuts across all paths, never to travels, and cannot refer to
the current or later cut. A connector crossing several guaranteed capsules may
be split into contiguous pieces and cite one cover per piece. This conservative
single-cover rule may reject travel safely covered only by a union; it cannot
turn unverified space into clearance.

For a cited cover, travel occupancy uses radius `rmax + e`; guaranteed prior
clearance uses `rmin - e`, or no clearance when `e > rmin`. For parallel
horizontal capsules, inclusion holds exactly when both travel endpoints have
squared distance to the cited center segment at most
`(prior_guaranteed_radius - travel_occupancy_radius)^2`, with a nonnegative
radius difference. Convexity covers every intermediate center. A travel capsule
with no cover is outside stock only when its exact squared minimum distance to
the stock rectangle is **strictly greater** than its squared radius. Touching
stock is rejected. These checks use rational arithmetic and the same declared
physical uncertainty as the cuts; they do not infer additional process margins.

The certificate covers only occupancy at the stated fixed Z for complete
horizontal traversals. A `cutting` entry proves its disk is allowed at that
section, not that descent to that section is safe. A `cleared` entry proves
section clearance, not a vertical access corridor. Path changes, tool changes,
retracts, holders, stock at other heights, Z uncertainty, generated paths and
machine execution remain unverified. Callers must retain those limits when using
the result; directly constructed result records are not proof.

### RC01 generated motion and full-height replay

`cambam_builder.cam_core.rc01.generate()` emits an immutable `Program` for the accepted
synthetic [RC01 job](REST_MACHINING_PLAN.md#first-generated-acceptance-job-rc01).
The generator currently accepts only the exact nominal `Job()` value. It emits
T1 roughing at tip Z=-1,-2,-3, followed by T2 cleanup in the four prescribed
7 x 7 corner windows at those depths. Every run contains a +5 rapid position,
fed approach, cutting entry or T1-cleared T2 descent, horizontal/vertical cut,
and fed retract. Tool changes and spindle start/stop are explicit setup events;
feeds, RPM, coolant state, tool identity, frame, units and tip datum are values.
The first T1 entry is (5,5); all four T2 columns are proved from actual T1 cuts.
The T1 raster uses 0.5 mm row spacing and vertical wall traverses; T2 uses
0.25 mm row spacing. An island offset endpoint is rounded away from protected
material. This is a deterministic baseline, not an optimized path.

`verify(program, job)` checks an ordered motion stream against the original
target and each earlier stock prefix. Axis-aligned cut disk sweeps and
single-prior-cylinder access containment use rational predicates over complete
segments. Cutting length, shank and holder cylinders are checked over their
entire occupied Z intervals, including descent and retract. New axial engagement
is limited to 1 mm by requiring the whole preceding layer's disk sweep to be
covered by earlier cuts. Low rapids, unproved travel, stale inputs, unsupported
diagonal cuts, positive physical error, and changed setup/process values fail
closed. A clearance route that needs a union of several prior cylinders is
outside this bounded proof even if physically safe. This restriction does not
affect generated RC01 motion.

The verifier retains rough-only and final cut prefixes, and reports rest for
each of the three open depth slabs. Each slab's section is constant because
all accepted cuts have integer tip depths and cylindrical vertical extents.
Independent integer-nanometre Y-strip bounds enclose the union area using
directed square roots; a 0.001 mm strip yields exact rational area bounds. A
separate GEOS polygon check rejects residual more than 0.05 mm from the four
ideal outer-corner rests or the original protected boundary. Its 1024-segment
quarter-circle has radial sagitta below 0.000001 mm; GEOS floating topology is
not a formal interval-arithmetic proof. The certificate reports its area coordinate
enclosure, polygon sagitta and `location_numeric_enclosure_mm=None` separately.
The `Certificate` carries both request
and motion fingerprints, rough/final per-slab area and volume intervals, and
`partial_target_completion` because finite cutters leave sharp-corner stock.
Calling `verify(..., measure_rest=False)` yields only `motion_only` diagnostics;
it is not a residual acceptance certificate. Native/CamBam motion must be
replayed separately before any output acceptance claim.

### Bounded pointed-cone slot generation and verification

`cambam_builder.cam_core.vcarve` implements one detached 90-degree pointed-cone
reference geometry in millimetres, frame `slot-drawing`, stock top Z=0. `Slot`
defines a 12 x 4 mm rectangular opening by default. Its desired depth at a point
is the smaller of distance to its four sides and `depth_cap`; the cap is 2 mm for
the V-shaped case or 1 mm for the flat-depth case. Sections at depth `t` are
`[t,L-t] x [t,W-t]`, including a floor section at the cap. This is original
target geometry; residual is target minus actual swept cone, never a changed
target. Outside the opening is protected stock. No fixture is modeled above
stock top or below the requested target.

`PointedCone` has equal maximum cutting radius and conical length: the default
is 3 mm for each. At height `h` above the tip, its cutting radius is `h` through
that length. `generate_slot()` rejects the radius-1.5/length-1.5 tool for 2 mm
penetration. For a 2 mm cap it emits a finite centerline from `(2,2)` to `(10,2)`
at tip Z=-2. For the 1 mm cap it emits three finite passes from x=1 to 11 at
y=1, 2 and 3, all at tip Z=-1. Each pass has an explicit vertical plunge,
horizontal cut and vertical retract; consecutive passes have a rapid link at
tip Z=+1. These are ideal geometric motions without feeds, spindle events or a
controller representation.

`verify_slot` requires that complete ordered motion structure and equal-depth,
equal-span passes, then checks each against the fixed original target and tool
limit. For every stock depth `t` in `[0,d]`, the swept disk radius is `d-t`.
The four endpoint constraints
`x0>=d`, `x1<=L-d`, `y>=d`, `y<=W-d` prove that the entire horizontal sweep lies
in the inset target section. Each vertical plunge/retract sweep is contained in
its deepest endpoint disk; rapid links have their tip strictly above stock.
This gives all-height ideal conical containment without testing only sampled
planes. `SlotResult` validates its plan on creation, reports pointwise residual
membership and analytic section residual area at any supported depth. Its volume
interval uses nesting of target and cone sections; the geometric bound tightens
with `steps`, but float roundoff is not a formal directed interval bound. Results
carry a plan fingerprint and a conditional analytic evidence class. `completion`
is partial when an exposed section has positive rest; otherwise it remains
undetermined rather than inferring complete volume removal.

For the default 2 mm case, section residual area is `16-4*pi = 3.4336293856`
mm2 at the surface, `4-pi = 0.8584073464` mm2 at depth 1, and zero at the V
centerline at depth 2. Finite end corners remain. For the 1 mm cap, three passes
reduce surface residual to about 1.0319614364 mm2 and leave 20 mm2 of flat-floor
area at depth 1 because finite pointed paths have zero floor area. This is an
explicit partial result. No numerical physical uncertainty, holder, stock above
Z=0, lateral engagement, machine constraints, broader region topology or native
CamBam output is certified. Rounded/flat tips and optimizers remain separate.

### Shared RC01 and pointed-cone motion/stock replay

`cam_core.replay` accepts immutable, millimetre, fixed-axis XYZ `Trace` values
with one source fingerprint, drawing frame, starting position, ordered setup/tool
events and motions, and resolved operation/tool/target values. Cylindrical and
90-degree pointed-cone cutting profiles use a bottom/point tip datum respectively.
`replay(trace, expected_source=...)` rejects stale sources, missing or displaced
tool/spindle events, discontinuous endpoints, low rapid/approach moves, unsupported
diagonal cuts, target/depth overcuts and uncleared descent/retract. Each accepted
entry or cut appends one swept-volume value to the same stock prefix. A result
reports the ordered cut tuple, operation-end prefixes, source and motion
fingerprints, and pointwise removed/residual membership at a requested prefix.

For cylinders, continuous axis-aligned capsule containment and protected-island
separation use exact arithmetic when the input coordinates are rational. Access
uses a conservative single prior-cylinder witness. For cones, a horizontal
deepest-tip pass and its entry are bounded by the inset slot section at every
depth; a vertical retract is allowed only through its immediately preceding
same-tool cut. Other cone access, tool components, fixtures, physical uncertainty,
feeds and machine dynamics require a separate verifier and remain outside this
bounded replay. The common replay is not a general collision engine.

`rc01.replay_trace` and `vcarve.replay_trace` adapt their existing public motion
types to this representation. The RC01 verifier still enforces its setup,
spindle/feed, axial engagement, holder and rest-location obligations, then uses
the shared ordered cuts for its three-slab residual oracle. The slot verifier
retains its complete pass traversal and analytic section/volume oracle, fed by
the shared cuts. `cam_core.mixed.verify_mixed()` validates both standalone plans
and replays RC01 T1/T2 followed by a cone slot translated to x=50..62 in one
frame. The slot lies beyond RC01 stock, so their independent target oracles are
compatible; this proves mixed representation, ordering, fingerprints and stock
prefix updates, but does not certify interacting targets or emitted machine
motion. Synthetic cone setup events have no feed/RPM or physical machine claim.

### Bounded convex closed-region rest and pointed cleanup

`cam_core.replay.Target.polygon` accepts a strictly convex counterclockwise
millimetre shell in the target's XY bounds. Its ideal 90-degree V target is the
original shell eroded by Euclidean distance `z` at depth `z`, up to the target's
declared depth. Each variable-depth straight cone cut is checked against every
original polygon edge at both endpoints. Edge clearance and tip depth vary
linearly along the segment; endpoint inequalities plus convexity prove every
intermediate disk and every stock-height section remains inside the target.
The tool is a pointed cone with equal maximum radius and conical cutting length.
The polygon branch admits no holes, concavity, arcs, holders or fixtures.

`cam_core.convex_rest.generate(prior_trace, end, expected_source=...,
expected_motion=...)` consumes a complete caller-supplied ordered trace of one
same-tool cone column. It checks source and motion fingerprints and replays the
prior column before planning. The start tip depth must equal clearance to the
**original** polygon, and the requested endpoint must have rising original
clearance with depth slope below one. The cleanup enters with a shared-replay
`cleared_descent` through that exact prior column, traverses one variable-depth
line, and retracts through its new endpoint column. Combined replay has separate
prior/cleanup cut prefixes. The result exposes pointwise pure rest after prior
motion, final residual, and nominal section areas; convex clipping gives target
area, the prior disk gives pure-rest area, and the existing analytic tapered
sweep formula gives final rest. A separate midpoint row integration checks that
last formula in the [synthetic triangle case](REST_MACHINING_PLAN.md#bounded-convex-region-restv-acceptance-case-2026-09-24).
This is a detached nominal geometry result with partial target completion. It
does not turn native source geometry or a CamBam preview into executable motion;
any future output carrier needs its own posted-coordinate replay and acceptance.

#### One native triangle workflow

`integrations.cambam.native_convex_rest` strict-imports the accepted zero-Z,
unbulged counterclockwise three-vertex Region `(0,0),(12,0),(6,8)` and one
enabled, unnested Part with drawing-space stock `(-1,-1)` to `(13,9)`, top Z=0
and bottom Z=-3. The source has no MOP. Millimetres are asserted by the
separate supplied prior-motion JSON because native XML does not persist a
verified drawing unit. Other topology, geometry, stock and source MOPs fail
closed. An existing `.cb` needs an explicit `prior.json`; the synthetic example
creates that supplied-motion fixture and binds it to the exact source SHA-256.
The adapter reconstructs the five supplied ordered prior items, checks their
source binding, and passes them to `convex_rest.generate`; it does not infer
prior removal from a native machining operation.

`build_workflow` copies those source bytes to `source.cb`, writes the supplied
trace to `prior.json`, and makes separate `preview/triangle-preview.cb` and
`explicit/triangle-explicit.cb`. The preview adds one XYZ cleanup Pline and an
enabled Engrave targeting only that generated path. It is a visual inspection
candidate. The explicit file adds one Point anchor and an enabled
Drill/CustomScript carrying the prior and cleanup moves, while the MOP/post
supplies tool and spindle events. The complete expected trace includes a safe
approach from the declared `(-10,-10,+5)` tip position, feeds, retracts and
return. Both retain the original Region and Part;
neither executes the other's generated path. Strict reimport checks target
links, process fields, source geometry and stock. The source, prior, preview and
explicit bytes plus motion references are SHA-256/fingerprint pinned in
`expected-motion.json`.

`audit_post` rechecks all four artifacts and the reconstructed motion, compares
every move/event in a CamBam Default/Default mm post against the explicit
candidate, then stock-replays the parsed posted coordinates and requires the
separate `prior`/`cleanup` prefixes. The test suite exercises this gate with a
constructed NC fixture; only an actual CamBam-produced post can establish the
native emitted-motion acceptance. The retained CamBam Plus 1.0 Default post
passed this exact ten-item gate and stock replay; the user confirmed its
enter/retract/re-entry/slope/retract sequence in CAMotics. CamBam did not
display the CustomScript motion as a toolpath. The Engrave preview is not an
execution authority. Tool holder, fixture, physical error, machine setup and
controller behavior remain outside this bounded nominal claim.

### M1 polygonal endmill rest and native output candidates

`cam_core.replay.Target.region_shell/region_holes` adds one valid straight-edge
planar Region with holes and a constant XY section to the shared trace. This
branch accepts cylindrical cuts only. Each entry/cut centerline must remain in
the original Region with boundary distance at least the tool radius, and the
tip must stay within target depth and cutting length. A crossing through a
concavity or hole fails even if both endpoints are inside. Region validation,
distance and polygon topology use the optional Shapely/GEOS backend; their
floating evaluation is conditional, not a formal interval proof. The existing
convex pointed-cone target remains a separate branch.

`cam_core.polygon_rest.generate` takes a source- and motion-fingerprint-bound
complete ordered prior trace. The core accepts multiple cylindrical prior
operations on the same target; the M1 native JSON adapter supplies one T1
operation and does not yet normalize arbitrary native MOP posts. It replays
the supplied trace, enforces the supplied roughing
allowance against the **original** Region boundary, and derives pure rest from
the actual T1 cut prefix. For the letter fixture, T1 has radius 2.5 mm,
0.5 mm allowance and four levels to Z=-8. T2 has radius 1 mm and uses the
original Region eroded by radius plus 0.0005 mm to form outer/hole contours
and interior scanlines only where contour cleanup leaves real rest beyond the
0.05 mm ideal/original-boundary envelope. The letter case needs only its
outer and hole contours; the preview shows their final Z=-8 level. Every T2
descent starts at an actual full-depth
T1 cut endpoint. Its connector cuts to the T2 path inside the original Region;
links between paths retract to Z=+5. The combined trace retains all prior
prefixes then `cleanup`. Disconnected T2 center regions and paths without a direct
cleared-column connector fail closed; the planner does not silently make a new
plunge into uncleared stock. A separate 1.8 mm throat fixture rejects a
radius-1 mm low-level crossing.

Section residuals use inner/outer capsule polygons with 128 quarter-circle
segments, radius perturbation 0.000001 mm and a circumscribing outer radius.
Area intervals are integrated across constant-depth slabs to report nominal
remaining volume. For A01 the prior leaves 1097.74561–1097.84293 mm³ and
the T2 path leaves 10.50611–10.54823 mm³.
The original 1532 mm² Region, including its triangular hole, remains the
target; the prior rest boundary never becomes a wall. All four section slabs
have rough rest 137.21820–137.23037 mm² and final rest
1.31326–1.31853 mm². The separate analytic six-convex-corner finite-tool
limit is 1.190659933 mm² per slab. Inflated sweep overcut and final residual
outside the ideal-or-original-boundary 0.05 mm envelope evaluate to zero in
this fixture, conditional on GEOS topology. Completion is partial. Synthetic
tools have 10 mm cutting length; their declared shank starts 10 mm and holder
20 mm above the tip, leaving both above Z=0 at the Z=-8 floor. Physical tool
error, material forces, fixtures and controller behavior remain unassessed.

`integrations.cambam.native_polygon_rest` strictly reimports the original
zero-Z eight-edge shell, triangular hole and one Part with stock X=[-26,26],
Y=[-2,62], Z=[-8,0]. A supplied `prior.json` is bound to the exact source SHA-256
and reconstructs every ordered T1 motion; there is no inference from native
Pocket intent. Separate `preview`, `explicit` and `native` `.cb` files preserve
the source. The preview Engrave is visual only. The explicit Drill/CustomScript
contains the full T1/T2 trace and a hash-guarded item-by-item Default-post
audit; the reader now accepts a declared initial XYZ for this source. The
native candidate has T1 and T2 Pocket MOPs on the original Region, with
0.5/0 mm roughing clearance and one bounded post-control hypothesis: no
spiral/optimisation, cut-feed stepover, zero crossover and a Custom MOP Footer
that moves to setup before stopping the spindle. CamBam documents the footer
as inserted [after each MOP's blocks](https://cambam.info/doc/1.0/cam/post-processor.html).
`audit_native_post` reads its **actual** Default post independently, bounds
posted G2/G3 arcs by chord sagitta, checks motion roles and the original
protected Region, and fails closed on unbounded ramps. It accepts low vertical
rapids only with earlier cleared-column witnesses. Neither candidate's
settings certify emitted motion. The revised contour-only literal candidate
passed its actual 2,692-item Default post. The one actual
Pocket post meets bounded area/0.001 mm geometry tolerance but has 12 T2 feed
descents outside the T1-cleared stock; that native route is rejected. The user
directed scenario-based selection of native MOPs, custom Regions and framework
paths, so the audited exact literal route is selected for A01. The synthetic
T1 raster is stock evidence for this test, not a recommended roughing strategy.
CamBam does not display CustomScript motion as its generated toolpath; the
posted NC and independent replay are the execution evidence. The preview
Engrave displays final-depth T2 geometry only.

### M2 curved Region endmill rest and literal output candidate

`cam_core.curved_region.approximate` consumes directed circular-arc bulge rings
after the native `Region` owner has validated analytic topology. It subdivides
each arc at at most 0.001 mm sagitta, records the exact source arc area and the
sum of circular-segment chord area errors, and constructs nominal, inward-safe
and outward-covering GEOS Regions. It rejects nonfinite geometry,
subresolution bulges, excess subdivision and offset
topology changes. The inward-safe Region is the
`cam_core.replay.Target` for both supplied T1 cuts and generated T2 cuts;
this makes the original analytic wall and hole protected under the stated
chord bound. Outer/inner cutter-sweep polygons and outward/inward target
polygons bound remaining area at each section. Constant-depth slabs give
volume intervals. GEOS topology and offsets are conditional floating results,
not formal interval proofs. A curved throat with no cutter-center clearance
fails instead of being bridged by a low cut.

`integrations.cambam.native_curved_rest` strict-imports one zero-Z native
Region with at least one arc and one non-nested 4 mm Part. The exact `.cb` hash
binds a separately supplied ordered T1 trace; imported source geometry,
including bulges and a reflected native transform, stays authored. The bounded
fixture uses two levels, T1/T2 diameters 3/1.5 mm and 0.25 mm T1 allowance.
`polygon_rest.generate` supplies full T2 entry, cut, high link and retract
roles on the inward-safe target. Separate preview Engrave and literal
Drill/CustomScript candidates strict-reimport with the original Region and
stock intact. The preview is visual only for execution; `audit_preview`
separately compares CamBam's actual Engrave post against its source-bound
Z=-4 T2 centerlines within Default's four-decimal coordinate precision.
`audit_post` requires a matching
actual CamBam Default/Default mm program, checks every ordered move/event,
replays the emitted coordinates and recomputes curved source residual bounds.
The tracked annulus, mixed concave/hole and reflected cases passed those
separate actual CamBam posts, strict source/candidate reimport and visible
curved preview observation. CustomScript confirms literal-post transport; it
does not certify that CamBam independently generated the execution sequence.
The analytic arc area, bounded rest and replay of parsed actual coordinates
are the stock evidence, conditional on GEOS floating topology.

### M3 bounded Region V path and native candidate contract

`cam_core.v_region` defines a 2 mm capped inward V recess on the original
planar Region: its section at depth `t` is the source opening eroded by
`t * tan(included_angle/2)`. A pointed cone has `rho(h)=h*tan`; a flat tip has
`rho(h)=a+h*tan`. A spherical lowest point of radius `b` joins the cone at
`h=b*(1-sin(angle/2))`, `rho=b*cos(angle/2)` with equal radius and slope.
Maximum radius and cutting length constrain every path. This finish choice is
explicit: it does not claim to restore a square wall and flat floor left by
an endmill. The result is reported as partial when a finite stepover/tip
leaves stock. An empty center region returns an infeasible result and reason.

For a tip penetration `d` and section `t`, cutter occupancy measured from the
original boundary is `t*tan + rho(d-t) <= rho(d)` for all three supported
profiles. Each straight XYZ segment, including changing Z between vertices,
must remain inside the inward-safe source and at least `rho(max endpoint d)`
from its boundary with a clearance margin. This continuous segment test covers
the full cutting profile and rejects a chord across a hole even when its
endpoints fit. A separate process bound limits tip-depth change to 2 mm per
XY millimetre on each cut segment. Entry feeds descend at the verified first
point to at most the declared 2 mm cap; every path
retracts above stock before the next XY link. At the cap, separate contours
trace the feasible shell and holes; variable-depth scan rows add passes across
wide interiors. Disconnected feasible pieces use separate safe entries. No
path-length or retraction optimum is claimed.

The curved source uses `curved_region.approximate`'s inward-safe and outward
brackets, preserving the native analytic arc separately. `section_report`
forms inner/outer straight-segment cutter buffers and reports conditional GEOS
residual-area intervals plus an inflated protected-overcut check. For changing
Z, a segment's minimum endpoint depth supplies its inner sweep and maximum
endpoint depth supplies its outer sweep; this deliberately brackets all
intermediate cutter radii. `volume_bounds` integrates conservative slab
enclosures; its current eight-slab interval is broad and is not a physical
surface-finish guarantee. Arc sagitta is at most 0.001 mm; GEOS topology and
floating arithmetic remain conditional.

`cam_core.v_region.with_prior` replays one supplied source-bound cylindrical
trace before the V stage. Each prior cut must lie within the same capped V
target at every height: its entire line has original-boundary clearance at
least `cylinder radius + cut depth * tan(angle/2)` plus margin. A cut that
would be legal in a flat pocket but gouge the V finish therefore fails.
The prior and final section reports measure pure rest and V cleanup gain.

`integrations.cambam.native_v_region` strict-imports the accepted A01 letter
and M2 annulus/mixed native Regions and Part stock. It binds a separate
`prior.json` with source/motion fingerprints. The synthetic T1 raster is a
proof fixture, not a production roughing recommendation; an external native
source requires a matching supplied prior file. It rounds V paths to the
Default post's four-decimal grid and rechecks full occupancy. The original
Region UUID, bulges, world geometry and Part setup must survive candidate
reimport. A separate XYZ Pline/Engrave MOP is preview only; a
Drill/CustomScript MOP carries the complete T1 then T3 sequence, including
both tools' entry, cut, high link and retract roles, CW 12000 rpm, F60 entry,
F300 cut/retract and +5 mm tip clearance. The preview Engrave MOP sets
`MaxCrossoverDistance=0` to request a clearance-plane retract between
separate paths; inherited 0.7 let CamBam add shallow XY feed connectors on
six of seven actual preview posts. Source/prior/candidate SHA-256
values, target/plan/motion fingerprints and both section results are recorded
in the manifest. `audit_preview` checks all posted
Engrave XYZ centerline segments and rejects low XY rapids; it grants no
execution certificate. `audit_post` checks the complete actual Default mm
stream against the exact candidate, including both tool/spindle sequences
and feeds, then replays the parsed prior stock and rechecks V path occupancy
and residual/gain. Each audit checks its own candidate against the common
source, prior and plan; changing a preview cannot invalidate an unchanged
explicit post. Its synthetic post regression proves the reader, not
CamBam acceptance. This bounded job uses its own supplied V-safe T1 stock;
selection among the earlier M1/M2 flat-pocket routes and V strategies still
belongs to M4.

### Bounded pointed-cone CamBam output carrier

`integrations.cambam.cone_script` attaches the full-depth 12 x 4 mm slot to one
enabled Drill/CustomScript MOP in a strict-reimported `.cb`. The document has a
Rect target at X=[0,12], Y=[0,4], Part stock X=[-1,13], Y=[-1,5], Z=[-3,0],
and a setup anchor Point at (-10,-10,0). The separate resolved tool contract is
a 90-degree pointed cone with maximum radius and conical length 3 mm, tip datum,
T3, CW 12000 rpm. The MOP carries `VCutter`, diameter 6, Default mm and exact-stop
intent. Its one full-depth path is (2,2,-2) to (10,2,-2). The synthetic process
uses feed 120 from +5 to +1, plunge feed 60, cut/retract feed 300 mm/min, and
above-stock rapid positioning. These are test tokens, not cutting parameters.

The `Default` post is expected to supply the first T3 change/start and terminal
stop. Six literal script lines supply all intervening motion, including return
to the setup position at tip Z=+5. Real XML newlines separate G-code blocks.
`expected-motion.json` records the exact ordered roles, positions, feeds,
tool/spindle events, candidate/source SHA-256 values and plan/motion fingerprints.
The candidate and source hashes must still match when its post is audited.

The audit accepts only the bounded straight-line Default-post dialect, harmless
standalone Drill G98/G80 wrapper markers, one optional redundant final M5, and
the declared initial tip position. It compares every emitted move/event with
the resolved trace before replaying the **posted coordinates** through
`cam_core.replay` and the original cone slot oracle. An actual CamBam `.nc` is
required for output acceptance; a synthetic wrapper only checks the adapter and
reader. The ideal cone proof has zero physical error and no holder, fixture,
controller or material-process certification. The post header does not prove the
incoming physical machine position; the setup tip position is an assumption.

### Bounded variable-depth V groove and CamBam carrier

`cam_core.tapered_vcarve` owns one millimetre, fixed-axis 90-degree pointed-cone
case. The original finish target is the exact union of cone sections along a
straight spine from (0,2,-1) to (12,2,-2.5). Cone radius grows linearly with
tip depth and has a maximum radius/conical length of 3 mm. The generated cut is
the proper subspine (2,2,-1.25) to (10,2,-2.25); its tip Z changes during the
single cutting move. At each stock depth, the target and cut sections are unions
of disks whose radius grows linearly along X. The cut is contained in the
original target because its entire spine is a subspine with the same depth
function. Entry/retract are vertical subsets of the endpoint cones; links are
above stock. Source geometry in `.cb` is an XYZ Pline guide to this resolved
target, not a native V-carve request. Stock is X=[-2,16], Y=[-2,6], Z=[-3,0].

`cam_core.replay` accepts a sloped cone cut only for this explicit tapered-spine
target. It proves endpoint depth inequalities for the shared linear profile,
cutting-length limits and ordered access; its swept-section membership maximizes
a concave quadratic over the finite spine. The target and cut section areas have
an independent analytic reference from their two tangent sides and two endpoint
arcs. Rest is `target area - cut area`, with the cut a proven subset. At stock
depths 0/1/1.5/2/2.5 mm, nominal rest is respectively
15.091265791880026 / 7.028684027518187 / 4.21460291488107 /
1.8062583920918873 / 0 mm2. Midpoint row integration independently checks the
formula to 0.0005 mm2. Positive rest establishes partial completion. These are
ideal geometric results evaluated in ordinary floating arithmetic, without a
directed interval proof; no physical error, holder or fixture is modeled.

`integrations.cambam.variable_cone_script` builds a strict-reimported source and
`V-variable.cb` with one enabled VCutter Drill/CustomScript carrier. It lowers
the sloped core trace through the previously accepted literal-motion route,
targeting a one-point anchor; the XYZ Pline records the finish-target spine
and is not targeted by the MOP. CamBam does not generate the sloped cut from
that Pline in this route. The carrier emits the cut as literal script motion,
with T3/CW 12000 rpm, +5/+1 approach, F120/F60/F300 test feeds, setup return
and exact-stop intent. The Default/Default mm output audit pins source/candidate,
plan and motion fingerprints, compares every ordered parsed event and move,
then replays **actual posted coordinates** through the original target and
analytic rest reference. The Default post does not encode the incoming machine
position, so tip (-10,-10,+5) is an explicit setup assumption. A synthetic
Default wrapper verifies the adapter. The actual CamBam-produced Default post
passed all nine emitted-item comparisons and the original target residual
replay on 2026-09-24. This accepts only the bounded explicit carrier; a native
Pline/Engrave V-carve workflow still needs its own output evidence.

`integrations.cambam.variable_cone_engrave` is the separate visible-toolpath
probe for the same bounded plan. Its source preserves the original XYZ target
spine; a second, generated XYZ Pline contains only the calculated cut from
(2,2,-1.25) to (10,2,-2.25). One enabled VCutter Engrave targets only the
generated Pline. It sets TargetDepth=0 to avoid adding a depth offset,
OptimisationMode=None, StockSurface=0, DepthIncrement=3, exact-stop intent,
T3/CW 12000 rpm, F60 plunge and F300 cut. The source guide is never a MOP
target. Strict reimport checks both XYZ paths, their MOP relationship and all
process fields. A hash-guarded manifest retains the same nine-item explicit
motion reference and five independent rest areas. The Default-post audit
compares every parsed move/event and separately reports whether the sloped
cut appears. The cut's presence alone cannot authorize this Engrave carrier:
entry, retract, links, feed roles, setup and events must all match before
posted-coordinate stock replay. CamBam preview and actual Default post are
now recorded: the user's XZ preview shows the slope and the native post
contains one exact sloped F300 cut. Its eight-item sequence omits the F120
approach, feed retract and setup return; it adds a rapid from the cut endpoint
at negative Z. The whole-motion gate fails, so this Engrave is an inspection
carrier only. The accepted CustomScript carrier remains the execution
authority for the bounded synthetic groove.

#### Bounded native V input normalization

`cam_core.tapered_vcarve.TaperedRequest` is the detached input shared by
standalone generation and `integrations.cambam.native_variable_v`. The native
adapter strict-imports one millimetre `.cb` with an open, unbulged two-vertex
world XYZ finish spine, one Part with stock surface Z=0, and one **disabled**
Engrave targeting that spine. The native MOP records explicit T3, a VCutter
diameter equal to twice the supplemental cone radius, zero depth
offset/clearance and fixed test process fields; it is source
intent, not executable V-carve evidence. `setup.json` supplies the pointed-cone
radius/axial length, tip datum, cut interval, +1 safe plane, approach/retract
feeds and setup tip position absent from native geometry. Normalization checks
the world XYZ shape, stock placement, source selection, enabled state and every
modeled Engrave parameter as an explicit Value; inherited, missing or unsupported
changes fail closed. Supported spine/stock/cone/cut-interval edits normalize to
the same family as detached requests. Equal numeric values are canonicalized
before plan fingerprinting, so XML `2` and Python `2.0` produce identical plans.

#### Straight variable-depth V planning family

`generate(TaperedRequest(...))` accepts finite millimetre values for an
increasing-X, constant-Y target spine `(x0,x1,y,d0,d1)` with `0 < d0 < d1`
and `0 < (d1-d0)/(x1-x0) < 1`. Its target is the exact union of ideal 90-degree
cone sections centered along that spine. The pointed cone has equal positive
maximum radius and conical length, at least `d1`. The rectangular stock must
contain the whole target envelope, and its bottom must be at or below `-d1`.
The requested cut interval is a strict interior subinterval of the target
spine. Its endpoint depths are derived from the target's linear depth law;
plunge and retract occur at those endpoints and `safe_z > 0`. Other slope
directions, bulges, curved paths, asymmetric cones and intervals touching the
target ends are outside this family.

`verify` reconstructs the canonical cut and ordered motion, then uses shared
replay to check tool length, access and cone containment across every stock
height. The containment proof follows from the cut being an exact subspine of
the original cone envelope; endpoint depth inequalities bound the linear
profile throughout the cut. The analytic tangent/arc section formula supplies
target-minus-cut rest, with independent midpoint-row integration for the
original and an edited target/tool input. The original plan and motion
fingerprints remain unchanged. Edited stock bounds/bottom participate in the
plan fingerprint. These are ideal floating-point geometry checks without
physical error or interval arithmetic. `plan_native_input` exposes a planning
result from edited `.cb` source bytes and explicit setup without producing a
CamBam carrier. The preview and literal CamBam output adapters remain
restricted to the previously accepted synthetic request. The direct reference
writer also accepts a verified edited plan through strict native source and
setup; the 14 mm / 8 mm cone member has an audited output file.

For the original synthetic request, `build_native_workflow` preserves the source bytes and creates two
separate derived `.cb` files. The preview has a generated XYZ cut Pline and an
enabled Engrave targeting only that cut. The explicit file has a Point anchor
and enabled Drill/CustomScript targeting only the anchor. Each retains the
original target spine and disabled source intent and is strict-reimported to
check the source request and target links. Both derive the same plan and expected
motion reference as standalone generation. The previously posted CustomScript
case establishes the carrier mechanism; a newly built native-derived candidate
needs its own actual CamBam post before its emitted motion is accepted. The
user's CamBam Default/Default mm post of that candidate passed all nine exact
item comparisons and posted-coordinate replay on 2026-09-24. The Engrave post
remains an unaccepted execution route.

### Bounded direct variable-depth V reference output

`tapered_vcarve.output_trace` now owns the complete bounded T3/CW 12000 rpm,
F120/F60/F300 process trace used by both the CamBam CustomScript carrier and
`integrations.direct_variable_v`. Moving this process trace into the detached
core leaves the existing plan and motion fingerprints unchanged. The direct
adapter accepts the canonical standalone request or the same strict-imported
native source/setup. It emits a deterministic ASCII, absolute millimetre,
straight G0/G1 reference program with explicit G17/G21/G90/G40/G61,
T3 M6, M3, M5 and M30. The initial tip position (-10,-10,+5) remains a
declared external setup assumption; the file does not establish it physically.

For edited plans, the writer emits up to 17 fractional decimal places so the
reader reconstructs the same binary float coordinates; values needing finer
decimal representation or exceeding 1,000,000 mm are rejected. Its five
section checks use depth 0, the target's first tip depth, the two trisection
depths and the final tip depth. This preserves the accepted original five
depths and program bytes. The direct output gate reparses all nine events/moves
through the bounded
Default-dialect reader, compares exact tool, role/G0/G1, feed, coordinates,
spindle RPM and order with the core trace, then replays the **parsed output**
against the original target. Exact generated bytes and SHA-256 guarded source,
setup, output and comparison-post evidence prevent stale artifacts from
inheriting a pass. A comparison post is optional for the edited member; the
accepted native-derived CamBam post is comparable only with the original
member, where it has the same nine semantic items, two cone sweeps and five
section-rest values. Byte-for-byte G-code identity is not required. This is a
headless **reference dialect** slice,
not a selected machine controller profile; it carries no holder, fixture,
physical-position or production-process acceptance.

### Bounded direct RC01 roughing/cleanup reference output

`integrations.direct_rc01` emits the exact nominal `rc01.generate(Job())`
two-tool sequence as deterministic ASCII G0/G1 in an absolute millimetre
reference dialect. It declares G17/G21/G90/G40/G61, T1/T2 changes, CW spindle
starts at 12000 rpm, stops and M30. Every feed move carries its resolved F word;
coordinates are exact terminating decimals. The initial tip position at
(-10,-10,+5) and coolant-off state are external setup assumptions; the
reference dialect emits no coolant command. The writer accepts only the
nominal detached job; native edits and controller profiles are outside this
output route.

The direct audit requires canonical bytes and a SHA-256 guarded manifest,
parses every emitted event and motion through the bounded Default-dialect
reader, and matches tool, G0/G1 role, feed, endpoint, RPM and order exactly to
the core trace. Role and operation names are reconstructed from that exact
match. It then passes the **parsed** values to `rc01.verify` with the original
job, retaining the T1-only and T1/T2 stock prefixes and rational three-slab
rest intervals. The separate GEOS residual-location check has sub-micrometre
polygon sagitta but no formal numeric topology enclosure. This is bounded
headless output evidence, not CamBam, controller or physical acceptance.

### RC01 native input and comparison candidates

`cambam_builder.integrations.cambam.rc01_adapter` owns the bounded `.cb` adapter. `synthetic_source()`
creates one Region with a rectangular island, one Part with 50 x 40 x 10 mm stock
at drawing origin (-5,-5), and two disabled Pocket source MOPs in T1/T2 order.
Every represented path parameter is explicit in the fresh native file. The paired
`setup.json` explicitly supplies units/frame, fixture extent, tip datum, travel,
feeds, coolant, uncertainty and full cutter/shank/holder components that the
native model cannot express. `normalize()` reopens the XML through strict import,
uses project-owned targets and transformed Region coordinates, and requires the
exact accepted `Job()`. It rejects inherited or changed source fields and unknown
extra source entities. Display-name and identity changes may preserve the same
normalized job; the adapter never trusts session-held Python objects in place of
reimport. Supported cosmetic edits therefore regenerate the same fingerprint;
geometry, process and component edits return explicit unsupported diagnostics.

`build_artifacts()` clones the reimported source for A, B and C. A has three
enabled T1 Engrave candidate MOPs, B also has three T2 Engrave candidates, and C
has the same T1 candidates plus four T2 Pocket window MOPs. The Engrave MOPs
target only the generated level-cut centerlines (79 per T1 depth, 248 per T2
depth); they do **not** encode the generator's entry, link, retract or event roles.
After the first native output trial showed additive depth and path reordering,
the candidate Plines are flattened to Z=0, each level MOP has `StockSurface`
one millimetre above `TargetDepth`, and `OptimisationMode=None` asks CamBam to
retain the supplied target sequence. The repaired A post confirms the intended
depths but follows the project's UUID-sorted MOP target selection rather than
framework cut order. This is not an accepted explicit-motion carrier. It does
not encode the required feed approach, setup-position tool change or
spindle-stop event.
Those complete ordered roles, source and motion fingerprints, and independent
rough/final residual intervals live in `comparison.json`. The four C windows are
design-neutral machining boundaries. Every candidate preserves the original
Region and disabled source MOPs; the generator does not mutate the source.

`cambam_builder.integrations.cambam.rc01_post` reads a deliberately small CamBam Default-post
subset: absolute millimetre G0/G1 XYZ motion, explicit G17/G21/G90, F/S/T,
G40/G61/G64 and M3/M5/M6/M30. Unsupported commands fail closed. It checks
ordered A/B motion or C's T1 prefix against the manifest within 0.001 mm and
checks candidate `.cb` SHA-256 before a file comparison. It flags G64 blending as
unverified trajectory geometry. A Default-post Z-only startup retract before
G17/T1 is parsed with an unverified initial-machine-position warning; a tool
change while the spindle is running is also unverified, with the emitted event
position retained for comparison. A clean C prefix does not verify native T2
Pocket motion. E/N acceptance requires actual regenerated and posted motion,
including added entries/links/events, stock replay and CamBam application review.

`rc01_adapter.build_native_variant()` now prepares a separate native Pocket
probe: T1 roughing on the original Region, alone and followed by the four T2
corner-window Pockets. Source MOPs remain disabled, candidates are strict
reimports, and `comparison.json` pins their SHA-256 hashes and normalized job.
`rc01_native_post.audit_native_posts()` accepts only those unchanged candidates
and two CamBam `Default` posts with matching name headers. The native-mode reader
adds bounded XY G2/G3 with relative I/J centers to the existing absolute-mm
G0/G1 subset; the Engrave comparison remains straight-only. The audit checks
T1 move-prefix equality including arc centers, posted event positions, feeds,
travel/floor and protected XY target bounds. It derives per-slab rough/final
rest from ordered posted G1/G2/G3 cuts, including descending ramps/helices.
An inner path radius reduced by arc mismatch/flattening error and an inflated
outer radius bound tessellation error; area, location and cleanup-benefit budgets
use the original target rather than candidate window edges. Exact rational
single-prior-cut witnesses check the four required corner columns and every
native T2 vertical location against full-depth T1 straight cuts. The returned
posted trial passes those coverage and column checks but fails entry, rapid,
feed and tool-change requirements. GEOS floating topology and island tangency
are not formally enclosed; the Default post also omits its starting machine
position. This bounded audit does not establish every axial/lateral engagement
or physical-machining certificate, and Pocket settings are never treated as
removal evidence.

### RC01 literal-motion CamBam carrier

`cambam_builder.integrations.cambam.rc01_script` owns the bounded explicit
T1/T2 `Drill/CustomScript` carrier. It clones the normalized source, retains
the target Region, Part stock and two disabled source Pockets, and attaches one
enabled Drill MOP to a setup-anchor Point. Its script stores the entire
`cam_core.rc01.generate()` sequence using exact terminating decimal XYZ
coordinates and real XML text newlines. CamBam Plus 1.0 preserved literal
`|` separators on one NC line, so they are not used by this carrier.

CamBam's `Default` post supplies the initial T1 change/spindle start and the
terminal T2 spindle stop. The script supplies all intervening move roles and
T1 stop/T2 change/start. `audit_script_post()` hash-guards the strict-reimported
candidate and source, accepts only the bounded Default post with its name
header, discards standalone G98/G80 Drill wrapper markers that cause no move,
and requires every emitted move/event to match the generated sequence exactly
in decimal coordinates, feed, tool and order. Extra motion fails. The audit
reconstructs a `Program` from actual emitted endpoints and calls the
continuous RC01 `verify()` for protected stock, tool-component access, process
and rough/final rest. Its accepted synthetic setup assumes the initial tip is
at (-10,-10,+5); the Default post does not encode incoming machine position.

The user-posted newline-repaired file passed all 2,945 items, with per-slab
rough rest 7.775010615955999–7.787678472024001 mm² and final rest
0.9214411294439999–0.9263453527720001 mm². This establishes the bounded
explicit-script E output slice for the accepted synthetic RC01 job; it does
not establish XYZ/Engrave parity, native Pocket N, arbitrary controller
dialects, physical machining, or formal GEOS topology interval proof.

### Native optimiser mapping corpus boundary

`cambam_builder.integrations.cambam.optimizer_corpus` authors the bounded
CamBam Plus 1.0 atlas and interaction `.cb` pairs. Each Legacy/New pair is
cloned from one project so geometry, target IDs and MOP order match; the
document name and explicit `OptimisationMode` token are the only intended
differences. CamBam's installed enum uses `Standard` for the documented
0.9.7 Legacy mode and `Experimental` for 0.9.8 New. The candidates pin
`units="Millimeters"`, Default postprocessor, installed `Standard-mm` styles
and `Default-mm` tools, stock and modeled MOP fields. The module strictly
reimports every candidate
and records its operation/target/property snapshot and SHA-256 in a versioned
manifest. The owning inventory and acceptance limits are in the
[mapping foundation](REST_MACHINING_PLAN.md#native-cambam-optimizer-and-output-mapping-foundation).

`inspect_post()` rejects changed candidate hashes, a mismatched file title,
non-Default or non-absolute-mm posts, missing/repeated MOP comments, or a
missing `M30`. It retains native posted words and modal G0/G1/G2/G3 endpoints,
arc centers, feeds and M events with exact `.nc` hash. Canned cycles and other
unhandled codes remain explicitly unresolved. The first machine position is
unknown. A parsed post has `posted_unreviewed` status and no stock authority;
domain review and any appropriate motion/stock replay must accept each case
before its mapping may inform rest calculations. Framework-generated motion
keeps its separate fingerprint and authority.

The exact four input `.cb` files, returned `.nc` posts, input manifest and
derived `observations.json` are reusable tracked fixtures under
`tests/fixtures/optimizer_corpus/`. `observe_corpus()` recomputes the
observations from these files: candidate/post SHA-256, ordered MOP sections,
target labels, depth/feed/approach records, arc-radius checks, tool and spindle
events, unresolved cycle words and section/program motion fingerprints.
Motion fingerprints omit line numbers and timestamp headers; exact raw NC
hashes remain separate. The review owner records which observations are
accepted and their limits. A new `.cb` export with different IDs or changed
settings needs its own native post and hashes.

### Bounded native MOP-series normalization and strategy selection

`integrations.cambam.native_series.normalize_native_series` strict-reimports
the source and candidate `.cb` files, preserves original primitive identity,
analytic world geometry and Part stock, and parses a complete actual CamBam
Default post in document MOP order. The bounded subset has one non-nested
millimetre Part, unique enabled MOP names and explicit cylindrical EndMill
tools. Native XML units/post settings or an explicit matching Default-mm setup
are required. Exact source, candidate and post SHA-256 values remain separate
from the modal motion fingerprint. Every G0/G1/G2/G3 item and tool/spindle
event is retained with its MOP section; unsupported words, cycles, absent or
reordered sections, tool mismatches and unbound setup fail. A parsed post
alone has no stock authority. The retained actual M1 Pocket post normalizes as
two ordered sections but its arcs and previously observed unsafe entries do
not acquire a replay certificate from this parser.

For a source with no MOP intent, the source evidence key normalizes original
primitive UUID/world geometry, Part stock and millimetre units. Document-title
and other presentation-only edits that leave that snapshot unchanged retain
the candidate/post certificate; geometry or stock edits invalidate it. A
source containing any MOP still requires exact source bytes because its MOP
intent lacks a semantic edit classifier. Candidate and actual post bytes are
always exact-hash guarded. The bounded linear audit also requires each native
MOP to target the same original Rect or straight Region geometry and floor;
a caller-supplied larger target cannot manufacture safe stock clearance.

`NativeSeries.to_trace` lowers only bounded G0/G1 linear motion on supported
XY Pocket/Profile/Engrave stages into the shared ordered replay model. Arcs,
ramps and low XY rapids have no lowering until continuous occupancy and access
proofs exist. `native_series_audit.audit_linear_native_series` rechecks the
source/candidate/post bytes, binds one source target and supplied cutting
lengths/entry modes, replays each complete prefix and reports section residual
area intervals, integrated volume intervals and inflated protected-overcut
upper area. A failed role/access/target/overcut check creates no selectable
stage certificate. The bounded geometry is a straight-edge Region or rectangle
with one common target across stages. Curved and V profiles need their own
source-bound occupancy adapter before this audit can certify them.

`cam_extensions.strategy.select_strategy` consumes only chained `StageAudit`
evidence with current source, actual emitted-motion fingerprints, complete
motion and passed tool/entry/link/stock/target/residual/post gates. It compares
safe alternatives by feasibility first, then final upper remaining area and
volume, then a caller-declared tie order. Manual selection can retain a safe
partial route but cannot choose an unsafe one. It returns selected, partial or
infeasible diagnostics and each stage's residuals. Native, custom Region and
framework stages can be mixed when each has its own audit; this policy never
infers stock from MOP intent. The current native linear end-to-end fixture
proves the parser-to-replay-to-selector path. A composed edited curved target
with rounded-tip comparison and direct reference output remains the full M4
gate; [the milestone scorecard](REST_MACHINING_PLAN.md#bounded-epic-completion-contract-and-milestone-scorecard-2026-09-24)
owns its acceptance.

### Edited curved rounded-tip composition and offset fill

`cam_core.v_region.plan` supports `raster` and `offset` fill patterns after the
same cap-depth shell/hole contours. Offset fill erodes the feasible cutter-center
Region by successive half-step and full-step distances, traces every resulting
outer/hole ring separately, and returns to safe Z between rings. Both patterns
use the same `_depth_path`, continuous full-height occupancy `verify`, source-bound
`with_prior` replay and conservative section/volume reports. The pattern enters
the plan fingerprint; the default raster fingerprint remains compatible with
the accepted M3 evidence. Offset topology and the finite path budget fail
closed rather than introducing low links through holes.

`integrations.m4_curved_workflow` compares one reopened, MOP-free curved annulus
source with a tangent spherical/conical T3 rounded tip. It copies that source
and one supplied T1 trace into a fresh, source-hash-bound bundle; an explicitly
requested synthetic T1 trace is available solely as a proof fixture. Both fill
routes produce separate strict-reimported native XYZ Engrave previews and
T1/T3 Drill/CustomScript explicit candidates through the M3 carrier. Each also
produces a complete G21/G90/G61/G17 direct reference program. A third, T1-only
direct file gives the partial endmill alternative its own complete output. Every direct file
is parsed independently, compared item by item including tools, spindle,
feeds and coordinates, and its posted T1 prefix is replayed through the shared
stock verifier before the same V occupancy and residual bounds are checked.
The direct dialect is a reference format, not a controller profile.

The comparison has three alternatives on the **same inward V target**:
T1 endmill only, T1 plus rounded raster, and T1 plus rounded offset. The
endmill-only route is explicitly partial and independently parsed/replayed;
it is not compared with the distinct
square-wall flat-floor target of M2. Each rounded route gets an ordered
two-stage `StageAudit` only after its complete actual Default explicit post,
actual preview centerlines, and parsed direct file pass. The actual prior
prefix is checked before the V stage receives stock authority. Selection
requires section Z=-1 upper remaining area at most 2 mm², eight-slab volume
upper at most 80 mm³, and zero nominal protected overcut under the conditional
GEOS model. Manual selection cannot override a missing or failed native post
for either rounded route.
The MOP-free source's original primitive UUID/world geometry and Part stock
form a semantic edit key: a document-title change retains evidence, while
an arc, hole, stock or setup change invalidates it. Candidate and direct file
bytes remain exact-hash guarded. This is a bounded edited annulus workflow,
not generic curved native MOP replay or physical machining acceptance.

### RC01 selected posted-motion stock authority

`integrations/cambam/rc01_stock_authority.py` supplies a bounded caller-selected
stock source for the RC01 rectangular Region with an island.
`analyze_rc01_stock("native_posted", evidence_path=...)` accepts either the
recorded T1-only or paired T1/T2 native Pocket Default posts. Each evidence
record pins the comparison manifest, setup and exact `.nc` SHA-256 values; the
manifest pins the original source and selected candidate `.cb` SHA-256 values.
The source and each candidate must strict-import and normalize to the same RC01
job, and each candidate must retain its selected enabled MOPs. The reader checks
matching post titles, Default headers and selected sections before replaying
actual G0/G1/G2/G3 motion. The paired path requires an identical parsed T1
event/move prefix through the last T1 move, including spindle, tool, feed and
arc-center history. It reports T1 and combined swept-stock/rest area bounds for
Z slabs ending at -1, -2 and -3, separate line-independent motion fingerprints,
rough/final budgets, exact T1 cut witnesses for T2 vertical columns and RC01
motion-role issue counts. The rough-only format and result remain supported.
`check_native_freshness(evidence_path, result)` rejects an earlier observation
after any evidence record, source, candidate, setup, manifest or post byte
change, including either post and the combined candidate in paired mode.
Reanalysis requires explicit new pair provenance; a matching post title
alone cannot prove that an edited document was posted.

`analyze_rc01_stock("framework_generated", program=...)` instead verifies a
supplied complete core `Program` against `Job()` and returns its independent
motion fingerprint and rough/final certificate. The caller must select one
authority; missing/stale evidence never falls back to the other. The recorded
native T1 and T1/T2 rest are **bounded geometric observations**, with GEOS
floating topology limits and unknown initial machine position. The recorded
pair meets coverage and vertical-column budgets but retains 62 rough and 233
combined motion-role findings. Its `stock_dependent_use` stays blocked: native
cleanup may not treat posted T1 removal as safe executable predecessor stock.
Neither native record is a native-Pocket N or physical acceptance certificate.
The framework program's own certificate is unchanged by native output.

### Detached nominal planar core

`cambam_builder.planar` is the public, document-independent owner of immutable
`PlanarFrame`, `RegionSet`, `Polygon`, `Rectangle`, `Circle`, `ErrorBudget`,
`PlanarApproximation`, `FeasibleSet` and `PlanarResult` records. It imports no CAD
entities, development probes or backend geometry classes. `_planar_shapely.py`
privately owns lazy Shapely/GEOS admission and regularized area operations.
The `planar` optional extra selects Shapely 2.0.7 on Python 3.9 and 2.1.2 on
Python 3.10+; ordinary imports and analytic centers need no backend.

Inputs declare `mm` or `inch`, a nonempty XY frame identity, a numerical origin in
that frame and optional section Z. Coordinates normalize as `(source-origin)*scale`
into local millimetres; inverse placement is `local/scale+origin`, with exact
`127/5` inch conversion. Binary operands must match frame identity, physical origin
and section after conversion; different anchors require caller-owned explicit
remapping. No native transform, Z projection or frame alignment is inferred.

`normalize(region, budget)` admits explicitly closed line-ring polygons with
assigned holes, rigidly placed analytic rectangles and a single analytic circle.
Finite positive primitive sizes, finite XY coordinates, closure, distinct vertices,
nonzero edges/area, simple rings, strictly contained non-touching/disjoint holes,
and disjoint component interiors are required. Independent component boundary
contacts remain explicit; overlap requires `union`. No repair, snapping or area
pruning occurs. General authored arcs and multiple components containing circles
return `unsupported`: chord polygons do not establish analytic source topology.
This deliberately bounded first subset does not claim the design's full curved
`RegionSet` vocabulary. Authored inputs remain unchanged and available to callers;
analytic identity is retained in source fingerprints, never fitted from polygons.

Circle subdivision checks stable sagitta and summed area-deficit expressions
against both budgets and `max_segments`. Source spans carry component/ring/segment,
parameter interval, source label and the chord's local-mm endpoints, independent of
canonical output winding/start/order. Derived operations retain those spans as
**input lineage**, not exact output-edge attribution. Fingerprints canonicalize
polygon ring start/winding and component/hole order, use exact numeric values with
no quantization, and include frame, units, analytic meaning and requested policy.
Source labels are excluded from the geometric key and retained separately.
Provenance records input fingerprints, policy, operation parameters, adapter revision
and actual Shapely/GEOS versions. Keys express identity, not approximate equality.

`union(left, right, budget)` and `difference(left, right, budget)` consume normalized
approximations. `nominal_area_erosion(value, radius_mm, budget)` exposes only
regularized filled area. A valid backend result outside strict shell/hole topology
returns `unsupported`, including shell/hole point contact. Component/hole counts
and contact locations are recorded. Empty area never proves a complete feasible
center set is empty. Segment limits gate input/output sizes and circle refinement;
they are not a hard memory/time interrupt inside GEOS.

`feasible_centers(primitive, frame, tool_radius, tool_units, budget)` is backend-free
closed-set disk erosion of one supplied analytic rectangle or circle. Exact rational
predicates on supplied numbers precede placement and unit conversion; one-ULP near
fits never become exact fits. `FeasibleSet` separates tagged analytic areas, closed
segments and points in primitive-centered local-mm axes, plus rational `center_mm`,
rectangle rotation and the original frame. Empty has no dimensions. Rational
coordinates preserve sub-ULP distinctions; no tiny polygon substitutes for a line
or point. Reflection of these symmetric primitives is expressible by the same
center/axis placement; arbitrary affine/native transforms are not an API here.
General polygons, size uncertainty and mixed collapsed branches are unsupported.

Results distinguish `ok`, `invalid_input`, `unsupported`, `unresolved` and
`backend_failure`. Failures return no accepted value. Known error contributions
that exceed a budget, unresolved conversion loss or work limits return `unresolved`.
Coordinate ULP allowances are not numeric proofs. Input approximation, coordinate
resolution, operation approximation/numerics and output conversion remain separate;
unavailable contributions are `None`/`unknown`, never zero. All successful results
are `nominal_geometry` with `budget_certified=False`; requesting certification fails
explicitly. There is no guaranteed removal, stock/rest certificate, executable
motion, XML/MCP exposure or production machining acceptance in this API.

A minimal programmatic slice (after installing the optional extra):

```python
from cambam_builder.planar import (
    PlanarFrame, ErrorBudget, RegionSet, Rectangle,
    normalize, union, difference, nominal_area_erosion,
)
frame = PlanarFrame("mm", "drawing", (0, 0), section_z=0)
budget = ErrorBudget(boundary_mm=0.001, area_mm2=0.01)
a = normalize(RegionSet(frame, (Rectangle((5, 5), 10, 10),)), budget)
b = normalize(RegionSet(frame, (Rectangle((10, 5), 10, 10),)), budget)
assert a.status == b.status == "ok"
joined = union(a.value, b.value, budget)
assert joined.status == "ok"  # area 150 mm2
cut = difference(joined.value, b.value, budget)
assert cut.status == "ok"     # area 50 mm2
result = nominal_area_erosion(cut.value, 1, budget)
assert result.status == "ok"  # area 24 mm2, still uncertified
assert not result.budget_certified
```

### Milling formula and constraint kernel

`machining_calculations.py` is a document-independent planning kernel. Its public
atomic helpers cover surface speed/RPM, chip load/table feed, rectangular-engagement
material-removal rate, specific-force cutting power and power/RPM torque. The
`MillingConstraints` solver propagates only equations with one unknown, checks fully
specified positive values within relative tolerance `1e-9` and no absolute floor,
and returns unresolved equation requirements instead of inventing missing facts.
All inputs and immutable results are positive finite values; effective flute count is
a positive integer. Axial depth and radial engagement remain distinct, radial
engagement cannot exceed a known cutter diameter, and target depth is deliberately
absent because it controls travel/pass planning rather than feed.

The selected `units` value is exact: `mm` means diameter/chip/depth/engagement in
millimetres, surface speed in m/min, feed in mm/min, MRR in cm³/min, specific cutting
force in N/mm², power in kW and torque in N m. `in` means inches, ft/min, in/min,
in³/min, lbf/in², horsepower and lbf ft respectively. These conventions match the
published Sandvik milling equations. Conversion factors are not inferred from the
magnitudes of supplied numbers.

Every supplied value is retained exactly in `requested_values`; the solver does not
round it. `MachineLimits` accepts arbitrary optional positive minimum and maximum
RPM/feed bounds, including equal endpoints, and rejects an inverted interval. A
supplied RPM or feed is a fixed machine setting and conflicts outside the interval.
A derived RPM or feed may be raised or lowered to a bound, with the original and
applied values plus the lower/upper direction recorded in ordered
`ActiveConstraint` diagnostics; downstream achieved surface speed, chip load, MRR,
power and torque are then recalculated. The kernel assumes rectangular engagement and a caller-supplied
specific cutting force. It includes no material/tool recommendation table, target-
depth rule, chip-thinning factor, circular-interpolation factor, plunge/ramp policy,
document mutation or production-safety claim.

### Milling recommendation profiles and extension API

`machining_recommendations.py` is the separate, public recommendation-selection
layer. `ToolProfile`, `MaterialProfile` and `MachineCapabilities` are immutable and
form a unit-consistent `RecommendationContext`. Machine capabilities can declare
minimum and maximum RPM/feed bounds. An optional immutable `OperatingConstraints`
record declares narrower setup/job bounds in the context's exact `mm` or `in` unit
system. The context exposes their intersection as 9a `MachineLimits`, rejects an
empty intersection, and never lets job bounds expand machine capability. Omitted
bounds add no default or limit. The API intentionally ships no material or
tool catalog. A caller supplies `Recommendation` values through a static profile,
explicit fixed-user-value profile, diameter table, or
`CallableRecommendationStrategy`. Custom callables are contractually pure: they
receive only the immutable context and return an immutable `StrategyResult`.

Every recommendation carries a positive value, exact unit label, nonempty numeric
`ApplicableRange`, and `RecommendationProvenance` classified as a manufacturer
starting point, measured shop policy, or user override. Provenance also identifies
the source, reference, and version/date. Diameter tables are bound to an exact tool
identifier, material identifier and operation, accept only chip load or surface
speed, interpolate linearly inside their stated diameter range, and report an
unmet requirement outside it; they never extrapolate or convert units implicitly.

`recommend_milling()` validates all applicable candidates before resolving each
field, so strategy order cannot hide or manufacture a conflict. One compatible
fixed user override wins regardless of order; distinct fixed values and unresolved
distinct non-fixed values for the same field fail explicitly. Missing capabilities and
out-of-range data remain diagnostics rather than guessed values. Plunge, ramp and
helical feed suggestions each require the matching tool entry capability, their own
nonempty rule, and ordinary provenance/range metadata. The layer validates and
selects starting values only. It does not serialize profiles, mutate documents,
infer entry feeds as cut-feed percentages, apply machine power/torque derating, or
establish safe production settings. Persistence/versioning remains deferred until a
durable owner is chosen.

### Milling pass and candidate planning

`machining_planning.py` composes the preceding two layers without adding cutting
data or a document dependency. `plan_depth_passes()` is the public owner of the
existing through-cut calculation: the caller supplies either an exact pass count or
a maximum axial increment already judged safe for the exact material, tool, setup
and machine. It balances cumulative depths, preserves the requested constraint,
and reports rounding plus low-final-stock engagement as advisory diagnostics. The
existing read-only `machining_calculate_depth_increment` MCP tool delegates to this
same function and remains document-free and non-mutating.

`plan_milling()` retains the complete 9b recommendation result and provenance,
lets explicit `MillingConstraints` values override non-fixed recommendations, rejects
a fixed strategy value that conflicts with an explicit value or tool fact, and uses
the tool profile's diameter/flute count plus the context's effective RPM/feed range.
For a through-cut, recommended or fixed `axial_depth` is the safe maximum; the
balanced actual `depth_increment` is used for MRR, power and torque diagnostics, so
the two values are never conflated. Physical radial engagement is also returned as
`stepover`, with `stepover_fraction` relative to cutter diameter. Non-fixed RPM/feed
targets are adjusted to lower or upper bounds while their recommendation retains the
original target and provenance. RPM adjustment recalculates a non-fixed feed target
from chip load; fixed feed instead remains exact and produces achieved chip load.
The final coupled system and downstream MRR/power/torque are recalculated after every
adjustment. Plunge/ramp/helical values remain separately sourced and capability-gated
by 9b; each is independently adjusted to the effective feed range, never inferred
from cut feed. Fixed cut or entry feeds outside the range fail. Declared power/torque
excess remains diagnostic without an unevidenced derating rule.

Underdetermined systems return stable missing requirements. No result authors a
MOP or mutates a document. Full recommendation-profile construction is deliberately
not exposed as a closed MCP schema because profiles and provenance are caller-owned
extensibility objects without a persistence or catalog contract; the useful named-
client pass-arithmetic surface already exists. Every plan is only a starting
recommendation: exact tool-manufacturer guidance, machine limits, workholding,
rigidity, chip evacuation and supervised test cuts remain required, and production
machining safety is not established.

### Project clone and bounded XML import

`CamBamProject.clone()` returns an independent in-memory copy of the complete
project graph, preserving UUIDs, ordering, relationships and XML templates.
Primitive weak project references point to the clone. Editing or saving the clone
does not mutate the original; no pickle or XML transaction representation is used.

`cambam_reader.read_cambam_bytes(data: bytes, *, source_name="", strict=True)`
accepts a bounded UTF-8 snapshot and returns a project or raises bounded
`ValueError`. DTD/entity declarations and non-UTF-8/NUL input are rejected before
parsing. The source limit is 10 MiB; strict reconstruction limits are 10,000
primitives and 1,000 MOPs. Strict mode rejects unsupported MOP kinds rather than
silently skipping them, malformed primitive scalars, failed registration and
unresolved MOP targets. `CamBamImportLimitError`, a `ValueError` subclass, identifies
resource-limit failures. The existing `read_cambam_file` compatibility behavior
remains separate. Neither entry point guarantees preservation of arbitrary XML
metadata or computes CamBam toolpaths. Adapter opens/saves always diagnose this
interchange limit and assert units explicitly.

### Shape elevation and Region contract

Pline and Points use one canonical `Vertex` record with finite `x`, `y`, `z`
and `bulge` fields; their sole intrinsic collection is `vertices`. `Vertex(x, y)`
defaults Z and bulge to zero, `Vertex(x, y, z)` sets elevation, and curved Pline
segments use `Vertex(x, y, z, bulge=value)` (or omit Z). `bulge` is keyword-only.
Pline/Points constructors and the project `add_pline`/`add_points` methods accept
`Vertex` records plus `(x, y)` and `(x, y, z)` tuple shorthand, copying them into
records. A three-tuple always means XYZ; four-tuples and the former
`(x, y, bulge)` interpretation are unsupported. Points reject any nonzero bulge.
Edits to a stored collection use `Vertex` records so coordinate, elevation and
segment data move together during insertion or reordering.

`add_circle`, `add_arc`, `add_rect` and `add_text` accept keyword `elevation`.
Text additionally stores optional `xml_p2_position` (XY) and
`xml_p2_elevation` for serialized `p2`. A missing `p1` means the default origin.
CamBam may omit `p2` or materialize it equal to `p1` according to edit history;
imported or explicitly supplied values are retained. The
[official CamBam MText API](https://www.cambam.info/doc/api/MText.htm)
documents `P1` as the current alignment point and `P2` as currently unused,
with possible future two-point alignment or rotation behavior. The framework
therefore assigns no drawing semantics to `p2`. Text content may occur before or
after native child elements such as `mat`; the reader retains the single
non-whitespace direct text chunk and rejects ambiguous multiple chunks.

`get_absolute_coordinates()` retains the original XY query shape.
`get_absolute_coordinates_xyz()` exposes Pline `(x, y, z, bulge)` tuples,
Points/Rect XYZ triples, Circle/Arc dictionaries with an XYZ `center`, and Text
dictionaries with XYZ `position` and optional `xml_p2`. Bounds remain XY
projections, with the existing analytic bulged-Pline/Arc extrema contract.
Pline segments support mixed vertex elevations with or without bulge. Bulge
describes the segment's XY circular projection and is retained with both endpoint
elevations; the framework does not calculate intermediate spatial positions or
spatial arc length. Tilted analytic entities remain unsupported. XYZ queries
correct arc sweep and bulge orientation under reflection. Circle/Arc and bulged
Pline/Region XYZ representations require similarity transforms (rotation,
reflection and uniform nonzero scale); nonuniform scale/shear can remain in
stored XML matrices and analytic bounds, but these XYZ queries reject them
instead of describing an ellipse as a circle. The legacy XY query remains a
compatibility projection and is not a complete spatial-curve representation.

Every primitive and project adder accepts `local_z_offset`, independent of
geometry Z. World Z is geometry Z plus this offset and all parent offsets.
XY matrices remain finite affine 3x3 arrays. `get_total_transform_xyz()` exposes
the supported composed pose as a 4x4 array; it does not enable arbitrary 3D
operations. `translate_primitive_z(id, dz, bake=False)` moves a subtree once.
Its baked mode changes each member's geometry and retains local offsets.
Full `bake_primitive_transform`/`bake_all_primitives` bake local XY and Z poses;
nonrecursive full bake compensates children in both coordinates. Explicit XY
bakes and the existing XY component operations leave Z unchanged. Geometry
changes for full/global subtree baking are staged before publication. Direct
`Primitive.bake_geometry` methods bake XY only, retaining that primitive's Z
offset; use project full-bake methods for combined XY/Z hierarchy baking.
Region baking also normalizes owned contour poses into contour coordinates.
Curved geometry baking requires a similarity transform; Text baking supports
translation and positive uniform scale, rejecting glyph rotation/reflection or
shear it cannot encode intrinsically. Matrix-mode Text rotation remains supported.

CamBam matrices encode world XY plus world Z translation (field 14 in the
zero-indexed flat column-major array). The reader snapshots world offsets and
subtracts the parent's offset when reconstructing local state. The canonical
matrix decoder requires `return_z=True` to return `(xy_matrix, z_offset)` when
Z translation is present. Calling its XY-only form with nonzero Z raises.
Both matrix helper spellings reject nonfinite values, perspective, Z scaling
and any XY/Z mixing, even if tiny. Unsupported primitive/layer schemas and
invalid primitive geometry fail the whole import (`None` with logged context),
while encoder errors propagate under the export contract below.

Subtree copy/transfer carries the detached root's complete world Z offset and
retains descendant local offsets, geometry elevations and owned contours.
Same-code-version pickle snapshots retain vertex records and restore project links.
Older layouts are not migrated or default-filled; model changes may make an earlier
snapshot unloadable.

The Region creation API is `add_region(layer, outer_curve, hole_curves=(), ...)`.
Contours are closed Plines. A registered input Pline's complete world pose is
captured before detachment; standalone contours retain their own local poses.
The Region's pose subsequently acts on those owned copies. The Region is the sole
registered primitive/MOP source. Its UUID, identifier, description, groups,
parent and layer belong to the Region. Region XML uses the observed native
`entity xsi:type="Region"` spelling, an `OuterCurve` containing `pts`, and
`HoleCurves/Polyline` contours. No independent contour IDs or MOP targets are
exported. Both contour windings are accepted and preserved. Contours may contain
arbitrary finite per-vertex Z values, including bulged segments with unequal
endpoint Z; Region validation and bounds interpret topology in the XY projection.
Contours must be simple and closed; holes must be strictly inside the outer curve,
mutually disjoint and nonnested. Touching, crossing, overlapping, degenerate
and unsupported elliptical contour geometry fail validation. The outer curve
may use two complementary semicircles. Validation uses analytic line/circle
intersections and curve containment, never a tessellated Boolean engine.
Topology tolerance is `max(1e-10 * max(1, XY extent), 8 coordinate ULPs, 1e-12)`;
near-zero projected areas below `1e-10 * max(1, XY extent)^2` are rejected. These
are floating-point CAD tolerances,
not exact predicates for arbitrarily ill-conditioned inputs. Region world
matrices must remain numerically nonsingular (determinant after normalizing by
the largest linear entry must exceed machine epsilon in magnitude); curved individual contour matrices must be
similarities, while a common nonsingular Region matrix can retain affine shape.

Acceptance fixtures use XML precision 10 or 12 decimal places and absolute
comparison tolerances `1e-10` in memory / `1e-8` through XML. World error can
amplify matrix rounding; `output_decimals` is configurable, and no scale-free
accuracy guarantee is claimed. Region export revalidates rounded contour geometry
and matrices before returning XML; precision that makes a hole touch an outer
curve fails export explicitly instead of creating a file that cannot reload. Acceptance and pending CamBam evidence live in the
[runbook](DEVELOPMENT.md#manual-shape-parity-acceptance) and
[status](PROGRESS.md#active-work-and-next-priority).

### Export failure and state-saving contract

`build_xml_tree()` propagates entity encoding and MOP resolution exceptions;
it does not return a tree after catching an encoder failure. Missing ordered
layers/parts and resolved MOP targets without XML IDs also raise. Original
encoder exception types are preserved; primitive/MOP logs identify the entity.
This is not a comprehensive validator for manually corrupted private registries.
Target mutation and XML import policies are defined below.

`save()` / `export()` / `save_cambam_file()` return `None` only after a completed
XML file replaces the destination. They retain forced `.cb` extension handling,
bare filenames, parent-directory creation and optional pretty printing. Without
`ET.indent` (Python 3.8), export continues unindented; other formatting errors
propagate. XML is written to a unique temporary file in the destination directory,
closed, then published with `os.replace`. Build, write, close and replacement
failures propagate, leaving an existing destination unchanged or a new destination
absent. Temporary cleanup is attempted on failure; cleanup errors are logged
without hiding the original error. Created directories may remain.
Replacement is filesystem-dependent, does not preserve destination inode metadata
or symlink-following behavior, and is not a power-loss durability guarantee.
Export still assigns project output precision to primitive instances.

`save_state()` accepts bare filenames in the current directory and creates nested
parent directories. Directory creation and pickle-writing errors propagate;
successful calls return `None`. Pickle writing remains direct/non-atomic. Load only
trusted snapshots created by the same framework code version; no missing-field
defaults or old-layout migration are provided.

### XML MOP identity contract

All four supported MOP encoders write JSON `Tag` fields `user_id` and
`internal_id` (UUID string), independently of display `Name`. The reader restores
identity before registration and preserves XML operation order within each part.
MOPs load after other entities: complete valid metadata colliding with an existing
UUID or identifier fails import (`None` with an error), never overwrites it.
Missing, incomplete or malformed identity metadata allocates a fresh UUID with
its string as identifier. Names remain unchanged; subsequent library round trips
preserve the new identity. Unrelated custom Tag content is not retained.

Primitive XML IDs resolve to UUIDs in the linking pass. Group sources export a
snapshot and import as project-owned explicit targets under the contract below.
Parameter interchange is described separately below.
Part UUID persistence remains outside scope. CamBam loading and properties for the synthetic MOP Tag case were accepted by
the user; criteria remain in `DEVELOPMENT.md`. This does not establish
production toolpath correctness.

### Development compatibility policy

User clarifications, 2026-09-08 and 2026-09-09 (unreleased, no API consumers): the framework is in development and no important
files depend on older framework versions. Prioritize a correct, coherent core
model. Breaking Python API, internal storage and pickle changes are permitted;
do not add legacy adapters, duplicate storage or old-pickle migration solely to
preserve unused framework versions. Update repository callers, examples and tests
together when changing the model. Old pickle files are not a compatibility target.

In tests and documentation, **fresh/imported behavior** means fresh optimal CamBam
XML output versus input saved by CamBam or another XML producer. It does not mean
compatibility with XML or pickle emitted by an earlier unreleased framework build.
Regression coverage should defend the current model and the CamBam XML boundary,
not freeze accidental framework history.

The external compatibility boundary is CamBam `.cb` interchange: files produced
by the framework and files created or modified directly in CamBam. Validate
supported geometry, transforms, operation targets/order and machining parameters
against that boundary, including CamBam-authored input without framework Tags.
Framework metadata may supplement native data but must not silently override
native edits to geometry or operation targets. Missing/malformed metadata and
unsupported content need explicit handling; full format coverage is not yet
established. Local self-round trips alone do not prove CamBam interoperability.

Pickle remains only a convenient current-code WIP/cache snapshot for resuming the
object graph. It can retain framework-only in-memory intent, such as a live group
target, that native XML materializes as a target snapshot. It is unsafe for untrusted
input and is neither required by the MCP adapter nor advertised as project exchange.
Once the framework is release-ready, a stable state/exchange contract may be assessed
on its own merits; that future decision must not require preserving pre-contract
pickles or distort the current model. If durable framework state is then justified,
prefer an explicit safe versioned schema rather than extending pickle compatibility.

The validated application baseline is CamBam Plus 1.0, `CamBam.CAD`
1.0.7364.41819 and `CamBam` 1.0.7364.41821, build 2020-02-29 23:13:58. CamBam
1.0 continues to write `Version="0.9.8.0"` in `.cb` XML. Treat that value as a
legacy file-format marker, not the creating application's version. The framework
emits the same marker for CamBam 1.0 interoperability. User validation in this
project uses CamBam Plus 1.0 unless the user explicitly reports a different version.
Do not change the emitted marker to `1.0` merely to label the target application:
the field is an internal file version, and `1.0` is not the value produced by the
target application. Reconsider only if a documented schema change or a controlled
CamBam compatibility comparison shows a benefit.

### MOP target ownership contract

The project `_mop_targets` registry is the sole target relationship owner. Each
MOP UUID maps to either an immutable set of primitive UUIDs or a live group name.
MOP entities contain no `pid_source`; old API and pickle migration are unsupported.

- `add_*_mop(part, targets=[primitive, identifier, uuid], ...)` creates an explicit
  selection. Targets default to empty; list/tuple inputs are copied and deduplicated.
- `add_*_mop(part, target_group="name", ...)` selects a live named group.
  Combining a group with nonempty explicit targets raises `ValueError`.
- `set_mop_targets(mop, targets)` and `set_mop_target_group(mop, group)` replace the
  selection atomically. Invalid/missing explicit primitives or MOPs raise
  `ValueError`; bare-string targets raise `TypeError`. Empty group names are rejected.
- `get_mop_targets(mop)` returns a detached UUID-sorted list of current targets;
  `get_mop_target_group(mop)` returns the group name or `None` for explicit mode.

Live groups support repeated programmatic construction: members added later are
selected, and recreating the same group name selects its new members. Missing
nonempty group names resolve empty. Explicit selection is stable across group
changes. Primitive deletion removes explicit references; reusing its identifier
does not retarget the MOP. MOP deletion removes its selection. Group changes must
use project APIs, not direct mutation of primitive metadata. Part assignment/order
remain separate project-owned relationships; target order is not machining order.

XML `<primitive><prim>` IDs are authoritative. Export materializes current targets
without changing their in-memory selection mode; import always installs explicit
UUID targets, even if a target set matches a metadata group. Missing/unresolved
native references are warned and skipped by the reader. Framework Tags never
restore live selections or override native target edits. Operation order follows
the native part/machineops sequence. Identity metadata remains supplementary under
the identity contract above.

This preserves live-group construction while keeping the CamBam boundary
unambiguous. No group-selection metadata or compatibility facade is introduced.
The old characterization is retained as historical review evidence.

The MCP adapter applies its per-family target-kind rules independently from its
geometry-mutation eligibility. A supported zero-local-Z root primitive with no
parent or children remains an eligible explicit MOP target when it belongs to one
or more framework groups; group names are selection metadata and do not change the
primitive's coordinates. Structured inspection likewise continues to return its
typed geometry. Parent/child relationships, non-similarity transforms and nonzero
local Z remain outside the MCP target slice. Geometry mutation tools retain their
separate, narrower relationship policy and may still reject grouped primitives.

### MOP parameter interchange contract

The four supported MOP classes reconstruct their modeled common and subtype fields
from native XML. `MOP_XML_FIELD_PATHS` owns the field/path vocabulary. Imported
MOPs retain a parameter-only XML template: Name, Tag and primitive target references
are excluded and regenerated from intrinsic identity and project relationships.
Unchanged fields retain native text, attributes, Default/Value states, omitted
elements and unknown/nested parameter content (including independent lead-out
settings, tabs and Style references). Custom framework Tag content remains subject
to the identity contract above.

Native Default text is a cached value available on the Python field, **not** an
evaluated CAM-style value. The library does not load or evaluate CamBam style
libraries. Assigning a modeled field after import makes it explicit on export,
even when the assignment repeats a cached value. Assigning `None` to an imported
optional scalar restores Default. `mop.set_parameter_state("clearance_plane",
"Default")` explicitly selects inheritance; `"Value"` selects the current value.
A later field assignment supersedes that explicit state choice. The state setter
supports top-level scalar fields only; nested inheritance belongs to native
containers and is preserved on import. Editing a nested field activates its
container as Value, retaining plain scalar formatting where CamBam uses it.

CamBam Plus 1.0 source-native Profile evidence further distinguishes present
`Default` from omission. A present `state="Default"` record carries cached text;
when another installation resolves a different local/style default, CamBam asks
whether to retain or update that cached value. An omitted property resolves from the
opening installation without that reconciliation prompt. Omission is not a promise
that CamBam will materialize every resolved property on its next save: after 21
top-level/container records were removed from a native Profile, CamBam restored only
`HoldingTabs`, continued omitting the other 20, and removed three empty Default
records (`StartPoint`, header and footer). Resave normalization is therefore
field/container-specific.

For an imported omitted modeled field, the XML template's absence is authoritative.
Any Python attribute supplied by a constructor fallback is not an evaluated CamBam
style value and must not be presented as one. Untouched export preserves the absence;
an assignment or explicit state selection is a new decision. Fresh framework output
therefore uses `Value` for relevant user/product decisions, omission for irrelevant or
deliberately target-resolved fields, and `Default` only for explicit inheritance,
faithful imported preservation, or evidenced active Auto semantics such as SpiralMill
`HoleDiameter`.

Fresh common-field output is owned by the ordered
`MOP_COMMON_FIELD_POLICIES` inventory. A supplied model value, or the explicitly
resolved Part/project context described below, is written as `state="Value"`.
Unset optional fields and empty header/footer strings are omitted. In particular,
the writer no longer invents a single-pass DepthIncrement or a CutFeedrate from
TargetDepth. The scalar state setter still lets a caller deliberately emit a
present `Default` or `Value` field. Imported-template preservation remains a
separate path and never fills absent properties.

#### Common MOP field inventory and fresh-export policy

Evidence labels are **D** (CamBam manual/API), **N** (accepted native XML or
CamBam behavior recorded in this repository), and **P** (deliberate framework
product policy). CamBam's [MOP API](https://www.cambam.info/doc/api/MOPFromGeometry.htm)
establishes the common property surface; its
[CAM Styles guide](https://www.cambam.info/doc/1.0/cam/cam-style.html) establishes
that `Default` inherits through the style hierarchy, `Value` overrides it, and
cached defaults can cause a conflict alert. Numeric ranges below are domain
expectations; the permissive direct core constructors do not yet enforce all of
them, while the MCP authoring boundary enforces its narrower published ranges.

| Model field | Meaning, units/reference, range and dependencies | Fresh XML disposition | Evidence |
| --- | --- | --- | --- |
| `target_depth` | Final absolute coordinate along the work-plane normal, drawing units; normally below `stock_surface`. | `Value` when supplied; otherwise omitted for CamBam/style resolution. | D, P |
| `depth_increment` | Positive maximum depth per pass, drawing units. | `Value` when supplied; otherwise omitted. No derived single-pass value. | D, P |
| `stock_surface` | Absolute top-of-stock coordinate along the work-plane normal, drawing units. | Always `Value`; constructor default is the explicit framework choice `0`. | D, P |
| `roughing_clearance` | Signed radial/normal stock offset, drawing units; positive leaves stock, negative overcuts. | Always `Value`, including zero. | D, N, P |
| `clearance_plane` | Absolute safe rapid coordinate along the work-plane normal, drawing units; should clear stock and fixtures. | Always `Value`. | D, P |
| `spindle_direction` | `CW`, `CCW`, or `Off`; applicable when spindle output is used. | Always `Value`. | D, P |
| `spindle_speed` | Spindle revolutions/minute; positive when used. A supplied MOP value wins, then framework Part context is resolved. | Resolved value is `Value`; omitted when neither exists. | D, P |
| `velocity_mode` | Controller cornering mode (`ExactStop` or `ConstantVelocity`). | Always `Value`. | D, P |
| `work_plane` | Coordinate plane (`XY`, `XZ`, `YZ`) defining the operation axes and depth normal. | Always `Value`. | D, P |
| `optimisation_mode` | Toolpath ordering algorithm; installed CamBam Plus 1.0 enum values are `Standard` (UI Legacy 0.9.7), `Experimental` (UI New 0.9.8), and `None`. Literal `Legacy`/`New` are not XML enum values. | Always `Value`. | D, P |
| `tool_diameter` | Positive cutter diameter, drawing units. A supplied MOP value wins, then Part, then project context. | Resolved value is `Value`; omitted only if no level supplies one. | D, P |
| `tool_number` | Tool-library/controller number; zero means current/no tool change in the framework contract. | Always `Value`. | D, P |
| `tool_profile` | Cutter shape metadata such as `EndMill`, `VCutter`, or `Drill`; affects simulation and some paths. | Always `Value`. | D, N, P |
| `plunge_feedrate` | Positive depth-axis feed, drawing units/minute. | Always `Value`. | D, P |
| `cut_feedrate` | Positive cutting feed, drawing units/minute. | `Value` when supplied; otherwise omitted. No depth-based fallback. | D, P |
| `max_crossover_distance` | Maximum cutting crossover as a fraction of tool diameter, conventionally 0..1. | Always `Value`. | D, P |
| `custom_mop_header` | Literal operation-prefix G-code text; postprocessor/controller semantics apply. | Nonempty text is `Value`; empty text is omitted. | D, P |
| `custom_mop_footer` | Literal operation-suffix G-code text; postprocessor/controller semantics apply. | Nonempty text is `Value`; empty text is omitted. | D, P |

`Enabled`, `Name`, `Tag`, and primitive targets are identity/relationship records,
not stateful parameters. `Style`, `StartPoint`, and `SpindleRange` are native
common properties but are not modeled; fresh files omit them rather than inventing
values. `RoughingFinishing` is currently modeled only where a concrete MOP class
exposes it; Drill no longer writes an unmodeled fixed value. Untouched imported
templates preserve all of these fields, their attributes, cached text, and absence.
The same preservation rule covers unknown common extensions.

#### Profile/Pocket subtype and nested-field inventory

The [Profile manual](https://www.cambam.info/doc/1.0/cam/profile.html),
[holding-tab manual](https://www.cambam.info/doc/1.0/cam/holding-tabs.html), MOP API,
and accepted native checks use the same **D**, **N**, and **P** evidence labels as
the common table. `MOP_PROFILE_FIELD_POLICIES` and
`MOP_POCKET_FIELD_POLICIES` are the ordered executable fresh-export inventory.
Every applicable modeled choice is `Value`; an irrelevant dependent is omitted.
No fresh subtype field uses cached `Default` text.

| Field(s) | Meaning, units/reference, range and dependencies | Fresh XML disposition | Evidence |
| --- | --- | --- | --- |
| Profile/Pocket `stepover` | Lateral path spacing as a fraction of effective tool diameter; normally greater than 0 and at most 1. | Always `Value`. | D, P |
| Profile `profile_side` | Cutter compensation side (`Inside`/`Outside`); for open paths this is left/right relative to traversal. | Always `Value`. | D, N, P |
| `milling_direction` | Conventional or climb direction relative to tool motion and stock side. | Always `Value`. | D, P |
| `collision_detection` | Enables CamBam's adjacent-geometry collision avoidance for the operation. | Always `Value`. | D, P |
| Profile `corner_overcut` | Extends paths into internal corners to remove otherwise unreachable material; deliberately overcuts stock. | Always `Value`. | D, N, P |
| `final_depth_increment` | Optional positive final-pass depth increment in drawing units; `0` disables the special final increment. | `Value` when supplied, including zero; omitted when `None`. | D, P |
| `cut_ordering` | Orders multi-level paths (`DepthFirst` or `LevelFirst`). | Always `Value`. | D, P |
| Pocket `stepover_feedrate` | Selects the feed-rate source for lateral stepover moves; the modeled default is `Plunge Feedrate`. | Always `Value`. | D, P |
| Pocket `region_fill_style` | Pocket clearing pattern (`InsideOutsideOffsets`, `HorizontalScanline`, or `VerticalScanline`). | Always `Value`. | D, P |
| Pocket `finish_stepover` | Finishing pass distance in drawing units; zero disables a distinct finish stepover. | Always `Value`, including zero. | D, P |
| Pocket `finish_stepover_at_target_depth` | Limits the finish stepover behavior to the final depth. | Always `Value`; false is explicit even when the current finish stepover is zero. | D, P |
| Pocket `roughing_finishing` | Selects roughing, finishing, or combined path behavior. | Always `Value`. | D, P |

`LeadInMove` is a `Value` container because `None` must explicitly disable style
inheritance. Fresh authoring supports the modeled `None` and `Spiral` modes only.
`LeadInType` is always a `Value`; `SpiralAngle` is a degree-valued descent angle and
is `Value` only for Spiral. The writer no longer invents unmodeled
`TangentRadius`, `LeadInFeedrate`, or a mirrored `LeadOutMove`. Other native lead
modes and independent lead-out records are import-preserve-only until their full
parameters and coordinate semantics have native A/B evidence.

Profile `HoldingTabs` is likewise a `Value` container. `TabMethod=None` is the sole
fresh child when tabs are disabled. `Automatic` additionally writes plain child
values governed by that container: positive Width and Height in drawing units;
integer MinimumTabs/MaximumTabs with minimum no greater than maximum; nonnegative
perimeter TabDistance; nonnegative SizeThreshold below which tabs are suppressed;
UseLeadIns; and Square/Triangle/Skip TabStyle. Width is tool-compensated in CamBam's
display, Height is relative to target depth, and Skip is a non-contact/plasma mode.
`UseLeadIns=true` requires Automatic Square tabs plus an active lead-in. Manual tab
point placement and transitions to or from imported Manual records remain
preserve-only because their native point collection is not modeled.

An untouched imported lead/tab subtree retains its container/leaf states, cached
text, unknown children, and independent lead-out. Editing a modeled nested leaf
activates its container. Switching `None`/`Spiral` or `None`/`Automatic` rebuilds
the applicable modeled siblings and removes modeled dependents that became
irrelevant, while preserving unknown children. This prevents a mode switch from
combining stale cached children with a new discriminator.

#### Engrave subtype inventory

The [Engrave manual](https://www.cambam.info/doc/plus/cam/Engrave.htm) and MOP API
use the same evidence labels as the tables above. `MOP_ENGRAVE_FIELD_POLICIES` is
the executable inventory. An applicable modeled value is explicit `Value`; an unset
optional final increment is omitted rather than exported as cached `Default` text.

| Field | Meaning, units/reference, range and dependencies | Fresh XML disposition | Evidence |
| --- | --- | --- | --- |
| `roughing_finishing` | Published compatibility property with Roughing/Finishing-style values. CamBam documents it as effective only for Lathe and 3D Profile, so this framework does not promise an Engrave toolpath effect. | Always `Value`; retained for API/interchange compatibility and pinned to Roughing by MCP authoring. | D, P |
| `final_depth_increment` | Optional depth of the final machining pass in drawing units; native G-code confirms that explicit `0` disables a distinct final increment. | `Value` when supplied, including zero; omitted when `None`. | D, N, P |
| `cut_ordering` | Orders multi-level paths `DepthFirst` or `LevelFirst`; native `LevelFirst` traverses targets per level and may serpentine between targets to avoid redundant returns. | Always `Value`. | D, N, P |

Engrave follows selected geometry, including its Z movement. Tool profile and signed
roughing clearance are common fields, not a second V-carving subtype: `VCutter`
describes the cutter but does not request skeleton or width/depth-varying V-carving.
Untouched imported subtype fields retain their native states, cached text and absence.

#### Drill method and subtype inventory

The [Drill manual](https://www.cambam.info/doc/plus/cam/Drill.htm), accepted
SpiralMill native checks and framework constraints use the same evidence labels.
`MOP_DRILL_FIELD_POLICIES` is the executable method-aware inventory. Fresh output
always makes the method explicit and emits only that method's applicable fields.

| Field(s) | Meaning, units/reference, range and dependencies | Fresh XML disposition | Evidence |
| --- | --- | --- | --- |
| `drilling_method` | Selects `CannedCycle` (G81/G82/G83 through the postprocessor), clockwise or counterclockwise `SpiralMill`, or `CustomScript`. | Always `Value`. Unknown native methods are preserve-only. | D, P |
| CannedCycle `peck_distance` | Nonnegative incremental drilling depth before each retract, in drawing units; zero selects no pecking. | `Value` for CannedCycle; otherwise omitted. | D, P |
| CannedCycle `retract_height` | Work-plane-normal cycle start/return (R-plane) coordinate in drawing units; native G81 output maps it directly to `R`. It should remain below the clearance plane and clear the stock. | `Value` for CannedCycle; otherwise omitted. | D, N, P |
| CannedCycle `dwell` | Nonnegative pause at the hole bottom. Time units are controller/interpreter dependent, not fixed by this library. | `Value` for CannedCycle; otherwise omitted. | D, P |
| SpiralMill `hole_diameter` | Requested hole-boundary diameter in drawing units. Explicit `H`, signed radial roughing clearance `R`, and effective tool diameter `T` must satisfy `H - 2R > T`. Auto derives each diameter from selected Circle geometry; a Point has no derivable size. | Explicit diameter is `Value`. `None` is the evidenced active Auto case and remains present as empty `Default`. Omitted for other methods. | D, N, P |
| SpiralMill `drill_lead_out` | Enables a bottom-of-spiral radial move before retracting. | `Value` for SpiralMill; otherwise omitted. | D, N, P |
| SpiralMill `spiral_flat_base` | Adds a complete circle at the spiral base when true; false can be useful for thread milling. | `Value` for SpiralMill; otherwise omitted. | D, N, P |
| SpiralMill `lead_out_length` | Signed radial distance in drawing units when lead-out is enabled: positive moves centerward, negative outward. A nonzero value requires lead-out; positive values may not exceed the effective hole radius. CamBam documents enabled zero as moving to the center. | `Value` for SpiralMill; otherwise omitted. | D, N, P |
| CustomScript `custom_script` | Literal drilling G-code template expanded once per point using CamBam's documented `$c/$d/$f/$h/$n/$p/$q/$r/$s/$t/$x/$y/$z` macros. Documentation describes `|` as a newline marker, but CamBam Plus 1.0 preserved it literally in the RC01 post; real XML text newlines produced separate NC blocks. Native output also confirms literal text and `$x/$y/$z` expansion. Controller/postprocessor semantics apply. | Nonempty text is `Value` only for CustomScript. Fresh empty CustomScript authoring is rejected; other methods omit it. | D, N, P |

The CannedCycle-only trio, SpiralMill quartet and CustomScript text are mutually
exclusive in fresh XML. This prevents irrelevant cached defaults from triggering
CamBam file-open reconciliation; the accepted SpiralMill files specifically proved
that omission is prompt-free. On an imported switch among the four modeled methods,
the writer removes stale modeled dependents, materializes the new method's complete
modeled record and preserves unknown extension children. An untouched imported
record preserves every original field/state, including irrelevant cached values.
Unknown native methods are preserve-only and cannot be switched because their
dependent field set is not modeled.

Explicit spiral diameter `H`, signed radial roughing clearance `R` and effective
tool diameter `T` must satisfy `H - 2R > T`. Auto diameter is resolved by CamBam
from Circle targets; the MCP adapter therefore requires explicit diameter whenever
a Point target is used. A nonzero lead-out length requires DrillLeadOut, and a
positive centerward move must not exceed the effective hole radius.

Imported global MachiningOptions and unmodeled part machining settings are
retained, including Style/StyleLibrary. Part ToolDiameter retains native state/text
while its modeled value is unchanged; changing the value exports the edit.
This preserves inheritance context within the file, not the referenced external
style libraries. Stock/origin still follow the existing part model.

MCP structured inspection reads state and presence from the retained native template,
not from constructor fallbacks or the authoring pin set. Its flat `parameters` map
contains each safe modeled value whose XML node is present. Parallel
`parameter_metadata` reports the governing native state and policy applicability;
a parent container in `Default` state governs its nested leaves even when a child
still carries cached `Value` metadata. `Omitted` fields have metadata but no value.
Fresh in-memory records use their generated policy XML for the same classification.
`Enabled` is classified as an attribute rather than a stateful parameter.

Inspection is independent of MOP target eligibility: explicit populated or empty
selections and live group sources retain the same parameter detail, while
`target_group` distinguishes live intent from explicit targets. Maximal unmodeled
native subtrees and unknown parameter attributes are named in `unsupported_fields`
without exposing their contents or blanking sibling modeled values. Manual tab point
collections, unsupported lead modes and literal CustomScript remain opaque. This
inspection contract does not authorize parameter mutation or evaluate CAM styles.

This is supported-MOP interchange, not full CamBam format or toolpath coverage.
Unsupported MOP types are warned and skipped; arbitrary references inside unknown
extensions are not interpreted or remapped. Native application acceptance is
tracked separately in the [runbook](DEVELOPMENT.md#manual-mop-core-model-and-cambam-interchange-acceptance).

### XML parent identity and world-pose contract

Primitive `Tag.parent` stores the parent's internal UUID string. XML `mat` stores
the world matrix, while an in-memory `effective_transform` is local to its parent.
The reader snapshots all imported world matrices before linking. For each resolved
parent edge it solves `parent_world @ child_local = child_world`, using those
snapshots rather than partially reconstructed ancestor chains. This preserves
world pose independently of object/layer order and across repeated round trips,
subject to XML numeric precision. Primitive UUIDs, identifiers and layer membership
are retained; layer UUID persistence is not part of the XML format.

Absent, malformed or unresolved parent references leave the primitive parentless
with its imported world matrix. Rejected self-parent and cycle-closing links likewise leave
the world pose unchanged. A resolved parent with a singular world matrix makes
the entire import fail: the reader logs the child/parent UUIDs and returns `None`
through its existing error boundary. Even a compatible child world matrix cannot
uniquely recover local coordinates; no pseudoinverse or silent detachment is used.
A singular primitive without children is valid. For cyclic XML metadata, the
reader retains edges accepted in input order and rejects the edge closing a cycle;
the resulting root can depend on XML order, while world poses remain unchanged
for invertible parent matrices. Singular-parent failure still takes precedence
when the reader cannot solve the candidate local matrix.

### Parent-link mutation contract

`link_primitive_parent` returns `False` for self-parenting, a proposed descendant
parent, or an already-cyclic proposed ancestor chain. Validation walks parent
UUIDs iteratively before mutation, leaving both relationship indexes, entity
registries, memberships and local matrices unchanged on rejection. Valid
reparenting, repeated links and detachment with `None` return `True` and retain
local matrices (so reparenting can change world pose). The guard prevents new
cycles through this API; it does not repair directly mutated registries or old
pickle state.

### Global transform application

`transform_primitive(node, M, bake=False)` applies the finite affine 3x3 matrix
`M` in world coordinates to the selected subtree once. Points are column vectors:
`world = P @ E @ local`, where `P` is the parent's world matrix (identity for a
root). Matrix mode solves `P @ E_new = M @ P @ E` and changes only the selected
local effective matrix. Descendant local matrices and all geometry remain intact;
ancestors, siblings, identities and memberships are unchanged.

With `bake=True`, effective matrices remain intact. For each subtree primitive,
the API solves `W @ B = M @ W` using its original world matrix `W`, then passes
`B` to its geometry baker. This is an explicit world operation, distinct from
full baking below, which removes existing matrices. All solves finish before
geometry changes. Invalid/nonfinite/non-affine inputs or a singular required
frame return `False` without mutation. Matrix mode requires an invertible parent
frame; baked mode requires every affected world frame to be invertible. A
singular target matrix is allowed in matrix mode with an invertible parent.
No pseudoinverse or silent reparenting is used.

Tests establish ordering for straight polylines and the translate/rotate wrappers,
including rotation about a world-space center. Curved geometry bakers retain
their existing limitations; exceptions during geometry mutation can leave partial
changes, and some entity bakers log failures internally. This API does not promise
transactional geometry baking. `align_primitive` has a separate implementation
and remains unverified. `combine_transformations(A, B)` returns `A @ B`, so `B`
acts first; it does not reorder arguments into chronological application order.

### Full local-transform baking

`bake_primitive_transform(..., transform_to_bake=None)` folds the selected local
matrix into its geometry and resets that matrix to identity. Every direct child
absorbs the removed matrix by left multiplication (`old_parent_local @ child_local`)
to preserve the descendant transform chain. With `recursive=False`, child geometry
is untouched; with `recursive=True`, that process continues down the subtree.
Ancestor matrices and parent/child identities remain unchanged. No inverse is
needed, including for singular scales. Recursive failures reported by the API
propagate as `False`; this operation is not transactional.

Synthetic verification covers straight polylines under translation, rotation,
nonuniform scale, reflection and shear, including repeated baking and XML round
trips. Entity-specific curved geometry/Rect conversion, explicit-matrix baking,
component baking and global transform application are not covered by this result.

### Rect baking representation contract

`Rect.bake_geometry` transforms all four local vertices. If the closed outline
matches an axis-aligned rectangle (cyclic/reversed comparison, absolute tolerance
`1e-12`, no relative tolerance), it retains Rect corner/width/height storage.
Otherwise the same Python object becomes a closed `Pline` with four ordered
vertices and zero bulges. Existing references still designate the registered
entity; its UUID, identifier, description, precision, project link and relationship
registries remain intact. Its runtime type changes and Rect-only geometry fields
are removed; callers must use the common Primitive API or check its current type.
No registry replacement or fresh identity is involved. The dataclasses share an
ordinary Python object layout; an incompatible subclass layout raises before
geometry edits. `to_pline_representation` remains a separate XML helper, not this
identity-preserving conversion operation.

Implicit baking resets the effective matrix; explicit baking preserves it.
Project full baking retains the descendant compensation policy above, while
global baking applies the existing world-to-local conjugated matrix to each
outline. These entry points share the entity conversion policy. Pure rotation
component baking also benefits; general component ordering remains unverified.
Finite affine inputs and resulting corners are checked before entity mutation;
errors propagate rather than logging an approximation as success. Multi-entity
operations remain nontransactional. Numerical and repeated XML checks cover
rotation, shear, axis-preserving controls and hierarchy relationships. Synthetic
CamBam display acceptance is recorded separately in the runbook/review.

### Curved geometry bounds contract

`Arc.get_bounding_box()` and `Pline.get_bounding_box()` return analytic
world-space bounds for their local circular sweeps. Bounds use the complete
local-plus-ancestor affine matrix. For each world coordinate, candidate extrema
are the sweep endpoints and the in-sweep stationary angles of
`u*cos(angle) + v*sin(angle)`. This makes identity, translation, rotation,
reflection, uniform and nonuniform scale, shear, and singular affine projections
exact within floating-point arithmetic. It does not change the stored Arc or
bulge representation: a non-rigid transform may display a circular source as an
ellipse, and the bounds describe that affine image directly.

Arc sweeps retain their sign and may wrap through zero. Magnitudes at least
`360 - 1e-9` degrees are treated as full circles, including negative and
multi-turn sweeps; a zero sweep is the start point. Stationary-angle membership
uses `1e-12` radians of absolute tolerance. Pline bulge belongs to the segment
from its vertex to the next vertex. The last bulge closes to the first vertex
only when `closed=True`; an open Pline ignores it. Bulges with magnitude at most
`1e-12` are straight, and chord lengths at most `1e-12` are coincident-point
segments. Stable half-chord formulas avoid avoidable overflow for large finite
endpoints.

The total matrix must be a finite real affine 3x3 matrix with homogeneous row
exactly `[0, 0, 1]`. Complex, perspective, nonfinite or unrepresentable geometry
returns the existing invalid `BoundingBox` rather than silently using an
endpoint-only or average-radius approximation. Empty Plines remain invalid;
single-point Plines have point bounds. Bounds are calculated only at runtime and
do not alter XML serialization. Curved geometry baking and the approximate
shape returned by `get_absolute_coordinates()` remain separate limitations.

The legacy package exposes `CamBam` and aliases through its own `__init__.py`;
it is a separate implementation, not the modern reader's fallback. Its CLI file
is not a declared installed entry point. Legacy compatibility requires its own
scope and checks. File/artifact handling belongs in the [topic map](README.md).

## 1. Overview

The framework is centered on a **Project Manager** (the `CamBamProject` class) that:
- Maintains registries for each entity type (layers, parts, primitives, and MOPs).
- Uses a **relationship registry** to track all associations between primitives and their parents, as well as assignments to layers and MOPs.
- Provides robust reference resolution so that every entity may be referenced by object, UUID, or user-friendly identifier.
- Enables operations such as adding, removing, copying, and transferring entities while ensuring that all relationships are updated in a single place.
- Supports CamBam XML interchange and an optional same-code-version pickle snapshot;
  only XML is an external compatibility surface.

---

## 2. Entity Model

### 2.1 Core Entity Classes

- **CamBamEntity (Base Class):**  
  - Contains an **internal ID** (a UUID) that serves as the primary key.
  - Contains a **user identifier** (a human-friendly string) that must be unique within the project.
  - **Note:** This class does not include relationship information.

- **Layer:**  
  - **Purpose:** Acts as a container for primitives.
  - **Attributes:**  
    - Visual properties (color, alpha, pen width, etc.).
  - **Relationship:**  
    - The project registry maps layer IDs (or unique layer names) to the list of primitives that belong to that layer.

- **Part:**  
  - **Purpose:** Represents a machining part with stock definitions and default machining parameters.
  - **Attributes:**  
    - Stock dimensions, machining origin, and default parameters.
  - **Relationship:**  
    - The project registry associates each part with its corresponding machine operations (MOPs).

- **Primitive (Abstract Base Class):**  
  - **Purpose:** Represents a geometric element (e.g. line, circle, rectangle, text).
  - **Intrinsic Attributes:**  
    - Geometry and transformation state.
  - **Relationship Attributes (Decoupled):**  
    - **Layer Assignment:**  
      - The project registry is solely responsible for mapping each primitive’s UUID to its assigned layer.
    - **MOP Assignment:**  
      - The project registry records the association between primitives and MOPs.  
      - Primitives do not store any MOP assignment data.
    - **Parent/Child Linking:**  
      - The project maintains a registry mapping primitive UUIDs to their parent UUID (if any).
      - At XML‐output time, the project “decorates” each primitive’s `<Tag>` with the parent’s UUID.
  - **Classification:**  
    - Each primitive may also have a list of groups and a free-text description.
    - At file output, these values are included in a JSON object in the `<Tag>` element.

- **MOP (Machine Operation, Abstract Base Class):**  
  - **Purpose:** Represents a machining operation.
  - **Intrinsic Attributes:**  
    - Machining parameters; target selections are project-owned under section 0.
  - **Relationship:**  
    - The project registry records MOP assignments by mapping a MOP to the primitives that should be processed.
    - If a referenced MOP is not found, the project may create a disabled dummy MOP (and dummy part) to ensure consistency.

---

## 3. Central Relationship Management

### 3.1 Project-Level Registries

The `CamBamProject` object manages not only the entity registries but also dedicated registries for relationships:
- **Layer Registry:**  
  - Maps layer IDs to the primitives assigned to that layer.
  - When a primitive is added or its layer assignment is changed, the project updates this registry.
  - During XML output, the project uses this registry to place primitives in the correct `<objects>` container.
- **MOP Registry:**  
  - Owns MOP targeting relationships in one authoritative location; the redesign defines group selection semantics under the section 0 development compatibility policy.
  - The project is solely responsible for maintaining MOP associations; primitives do not store any MOP assignment data.
  - If a primitive refers to a MOP that does not exist, the project creates a dummy (disabled) MOP.
- **Parent/Child Registry:**  
  - Maintains a mapping from a primitive’s UUID to its parent’s UUID.
  - No duplication occurs inside primitives; the relationship is stored solely in the project.
  - When a primitive is created with a parent, the project records the association in this registry.
  - This registry is used for propagating transformation changes and for serializing the parent–child relationship in the XML `<Tag>` element.

### 3.2 Helper Methods for Relationship Updates

The project provides helper methods to add, remove, or update relationships:
- **Assigning a Layer:**  
  - When a primitive is created (or updated), its intended layer (passed as a unique layer name or object) is registered in the project.
  - The project updates its layer registry so that the primitive is associated with the correct layer.
- **Assigning a MOP:**  
  - When a primitive is created or updated, the MOP association is recorded solely in the project’s MOP registry.
  - The primitive does not store any MOP assignment data; instead, the project injects this information into the `<Tag>` JSON when building the XML.
- **Linking Primitives:**  
  - The project maintains a parent mapping (a dictionary mapping each primitive UUID to its parent UUID).
  - Helper methods update this registry when a primitive is linked to a parent.
  - When building the XML, the project queries this registry and injects the parent reference into the `<Tag>` JSON.

### 3.3 Transformation Propagation (Conceptual)

- **Centralized Transformation Methods:**  
  - Transformation methods (e.g., translate, rotate) are implemented at the project level.
  - When a transformation is applied to a primitive (by updating its transformation matrix or by baking the changes into its geometry), the project looks up any linked children (via the parent/child registry) and recursively applies the transformation.
- **Final State in Primitives:**  
  - The resulting transformation matrix or updated geometry is stored directly in the primitive objects.
- **Decoupled from Relationship Storage:**  
  - Since relationship data is maintained solely in the project’s registries, transformation propagation relies on these registries as the single source of truth.

---

## 4. Serialization and Reconstruction

### 4.1 XML Serialization

- **XML Generation:**  
  - The XML writer (in `cambam_writer.py`) queries the project’s registries to build the final CamBam XML file.
  - For each layer, the project uses its layer registry to create a `<layer>` element with an `<objects>` container.
  - Primitives are placed in the correct container based on the layer assignment provided by the project registry.
  - Each primitive’s `<Tag>` element is populated with a JSON object that includes:
    - `user_id`
    - `internal_id`
    - `groups`
    - `parent` (obtained from the parent/child registry)
    - `description`
- **MOP Associations:**  
  - The XML writer uses the MOP registry to generate MOP sections. Each MOP element includes a `<primitive>` element listing the XML IDs of the primitives associated with that MOP.

### 4.2 Optional WIP/cache snapshot

- The current project object, including relationship registries, may be snapshotted
  with pickle to resume trusted same-code-version work in progress.
- Primitive state hooks remove and restore only the transient weak project reference;
  they do not migrate old fields or layouts.
- CamBam XML is the supported exchange surface. A future release-grade framework
  state/exchange format requires a demonstrated need and a robust explicit contract;
  unreleased pickle history does not constrain that design.

### 4.3 Reconstruction (Importing a CamBam File)

The reconstruction process (in the XML reader module) follows a precise order:
1. **File Base:**  
   - Begin with the root of the XML file.
2. **Parts:**  
   - Parse the `<parts>` element to reconstruct all parts.
3. **MOPs:**  
   - For each part, parse the `<machineops>` element and reconstruct MOPs.
   - Extract the list of primitive XML IDs from each MOP’s `<primitive>` element.
4. **Layers:**  
   - Parse the `<layers>` element to rebuild layer objects.
   - Initially, layers are created without their contained primitives.
5. **Primitives:**  
   - Iterate over each layer’s `<objects>` container to extract primitive elements.
   - For each primitive, parse the `<Tag>` JSON to recover `user_id`, `internal_id`, `groups`, `parent`, and `description`.
   - If the `<Tag>` is missing or incomplete, default values (or new UUIDs) are assigned.
6. **Linking Registries:**  
   - Using the XML structure, the project rebuilds its relationship registries:
     - **Layer Registry:** Each primitive is associated with the layer in which it was found.
     - **Parent/Child Registry:** The parent link is re-established from the `parent` field in the `<Tag>` JSON.
     - **MOP Registry:** For each MOP, the primitive XML IDs (converted back to UUIDs using a temporary mapping) are used to re-associate primitives with the corresponding MOP.
   
This process reverses the file-writing procedure so that a CamBamProject object is fully reconstructed from the XML file, even for files not originally created by the framework (provided the expected metadata is available).

---

## 5. Transferring Entities Between Projects

### Copy and transfer contract

`source.copy_primitive_tree(root, target_project, *, preserve_ids=True,
identifier_map=None, group_map=None, include_mops=True)` copies a complete
descendant subtree. `transfer_primitive_tree` has the same arguments and removes
the originals after staging succeeds. Both return a source-to-destination UUID
dictionary for every included primitive, layer, part and MOP.

- The selected root becomes parentless with its original world matrix as its
  local matrix; descendants retain their local matrices and internal parent
  edges. Finite affine matrices are required. No geometry baking occurs.
- Assigned layers are cloned. With `include_mops=True`, every MOP whose resolved
  selection intersects the subtree is included, together with its owning part.
  A selection that also targets outside primitives rejects the entire operation.
  Empty and unrelated operations are excluded. `include_mops=False` copies only
  geometry, layers and group memberships. Missing required containers reject.
- Container properties, intrinsic geometry and MOP parameters are deep copies.
  Layer/part order and relative MOP order follow the source and append to the
  destination. Existing destination entities retain their Python identities.
- UUIDs are preserved by default; `preserve_ids=False` generates fresh UUIDs for
  **all** included entities. `identifier_map` maps source user identifiers to new
  nonempty names; `group_map` similarly renames copied groups. Unspecified names
  are preserved. Unknown mapping keys and duplicate destination names reject.
- Any destination UUID or user-identifier collision rejects, across entity
  types. Containers are always cloned, never implicitly reused or overwritten.
  This supersedes the earlier conceptual overwrite-on-UUID-collision proposal:
  UUID equality alone cannot justify replacing destination relationships.
- Group memberships are restricted to the subtree. Explicit MOP targets remap
  through the UUID dictionary; live group selections remain live under their
  mapped names in memory. Closure validation uses current group membership.
  Existing XML interchange writes resolved targets and reconstructs explicit
  snapshots, not live selectors. Destination group names must be unused, including names reserved
  by live selectors with no current members. Groups are never implicitly merged.
- Same-project copying requires fresh UUIDs and distinct entity/group names;
  same-project transfer rejects. Copy leaves the source unchanged. Transfer
  removes included primitives and MOPs, cleans source relationship indexes and
  retained selections, clears removed primitives' project links, and retains
  source layers, parts and unrelated entities.
- Validation and deep copying complete before either project's registries are
  published. Anticipated validation/cloning errors leave both projects and
  existing entity references unchanged. This is an in-memory operation, without
  thread synchronization or filesystem transaction guarantees.

The stopping condition is relationship/identity/collision coverage plus two
synthetic XML round trips preserving counts, references, world geometry and MOP
parameters. Automatic overwrite/merge, arbitrary selection closure, destination
parent placement and whole-project/settings transfer are outside this API.
Destination project machining/style context remains authoritative; copied
Default-state parameters may therefore inherit different effective values.
Primitive/MOP UUIDs survive XML round trips; existing layer/part XML encoding
reconstructs container identities, so their UUID mapping is an in-memory contract.
Rect XML conversion to Pline bakes the complete world outline with an identity
XML matrix, including ancestors. Its representation may change on interchange;
world geometry and descendant poses are preserved rather than the Rect matrix.

---

## 6. Transformation Propagation (Conceptual)

- **Centralized Transformation Handling:**  
  - Although the detailed transformation methods are not covered in this specification, the design requires that transformation operations are executed at the project level.
  - When a transformation is applied to a primitive, the project queries its parent/child registry and recursively propagates the transformation to all linked children.
- **Final State in Primitives:**  
  - The resulting transformation matrix or the updated geometry is stored directly in the primitive objects.
- **Decoupled from Relationship Storage:**  
  - Because relationship data is maintained exclusively by the project, transformation propagation relies on a single source of truth, ensuring consistency even after transferring or reloading entities.

---

## 7. Summary

This specification defines a robust and decoupled architecture for the CamBam CAD/CAM framework based on a central project object that maintains all relationship data in dedicated registries. Key points include:

- **Centralized Relationship Management:**  
  - The `CamBamProject` object holds dedicated registries for layer assignments, MOP associations, and parent/child links.
  - Primitives do not store layer or MOP assignment data internally; these associations are maintained solely by the project and injected into the XML output via the `<Tag>` element.

- **Simplified Transformation Propagation:**  
  - Transformation functions are implemented at the project level and rely on the centralized parent/child registry to recursively propagate changes.
  - The final transformation state or updated geometry resides directly in the primitive objects.

- **Robust Serialization and Reconstruction:**  
  - XML output is generated by querying the project registries, ensuring that primitives are placed in the correct layer and MOP containers and that parent links are recorded in the `<Tag>` metadata.
  - The reconstruction process follows a strict order—starting with parts, then MOPs, layers, primitives, and finally linking primitives to layers, MOPs, and parent relationships—so that a complete project can be rebuilt accurately.
  - The framework reconstructs supported relationships from XML. Trusted
    same-code-version pickle snapshots are an optional development convenience,
    not an alternative interchange contract.

- **Inter-Project Transferability:**  
  - Copy and transfer methods operate on the centralized registries, allowing entire linked trees of primitives to be transferred without duplicating relationship data.
  - UUIDs are preserved to maintain existing links; if a conflict occurs in the target project, the primitive is updated accordingly.
  - The relative structure is preserved and reflected in the XML output.

This design minimizes duplication by centralizing all relationship data within the project object’s registries, ensuring that the management, propagation, and serialization of entity relationships remain robust and maintainable.
