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
packages, the detached `cam_core` and CamBam integration subpackages; `inactive/` and demos
are outside that runtime package list.

| Owner | Implemented responsibility | Start here when changing |
| --- | --- | --- |
| `cambam_builder/cambam_project.py` | UUID entity registries, identifier lookup, ordered layers/parts/MOPs, relationship updates, transform orchestration and persistence | Public creation/query/mutation APIs and relationship invariants |
| `cambam_builder/cambam_transfer.py` | Transactional primitive-tree copy/transfer staging, identity mapping, collision validation and relationship publication | Copy/transfer semantics and atomic registry updates; inspect project wrappers and tests |
| `cambam_builder/entity_core.py` | Shared vertex/bounds/numeric foundations, `CamBamEntity`, and `Primitive`, including local and parent-composed transforms | Identity, base inheritance, shared validation, bounds foundations, or primitive transform context |
| `cambam_builder/cad_entities.py` | `Layer` plus ordinary Pline/Circle/Rect/Arc/Points/Text geometry and XML behavior | Ordinary CAD fields, geometry, bounds, baking, or entity XML |
| `cambam_builder/region.py` | Owned Region contours, planar curved topology validation and typed Region XML; depends directly on core and ordinary CAD owners | Region geometry and interchange; project and reader use this owner |
| `cambam_builder/cam_entities.py` | `Part`, MOP classes, and MOP XML path/encoding policy inventories | Part stock/nesting or MOP parameters and XML policy |
| `cambam_builder/cambam_entities.py` | Explicit compatibility/discovery facade re-exporting canonical objects from the four entity owners | Preserve public entity imports; implementation modules must import owners directly |
| `cambam_builder/cad_transformations.py` | NumPy matrix construction, composition, decomposition and XML matrix conversion | Numerical conventions; inspect entity and project callers together |
| `cambam_builder/stock.py` | Exact conditional disk-sweep occupancy, remaining-section bounds and supplied section-motion verification | Horizontal cuts and explicit travel inside rectangular stock/target; no generated or native path integration |
| `cambam_builder/cam_core/` | Document-independent CAM planning, motion and stock-analysis package; currently owns `rc01.py` | Exact nominal rectangular two-tool job; no native entity, XML, MCP or machine-output dependency |
| `cambam_builder/planar.py` and `_planar_shapely.py` | Detached nominal planar values, error/provenance policy, analytic feasible centers and private optional GEOS adapter | Pure planar geometry; no document or stock/path ownership |
| `cambam_builder/machining_calculations.py` | Pure unit-explicit milling formulas, partial-input constraint solving and derived RPM/feed machine caps | Arithmetic planning kernel; composed by the separate pass planner |
| `cambam_builder/machining_recommendations.py` | Immutable tool/material/machine contexts, provenance-bearing recommendations, user diameter tables and pluggable pure strategies | Recommendation selection only; contains no curated catalog, persistence, document mutation or safety claim |
| `cambam_builder/machining_planning.py` | Pure through-cut pass balancing and composition of recommendation profiles with formula/machine diagnostics | Candidate planning only; the existing MCP depth tool delegates here, while full profile construction remains a direct-Python API |
| `cambam_builder/cambam_writer.py` | XML ID assignment and layer/part traversal; delegates individual encoding to entities | Output structure and reference resolution |
| `cambam_builder/cambam_reader.py` | XML parsing, entity reconstruction, ID mapping and deferred parent/MOP linking | Import defaults, malformed data and round-trip reconstruction |
| `cambam_builder/integrations/cambam/` | RC01 `.cb` input/attachment and bounded posted-motion comparison; depends on native model and detached RC01 values | Bridge between native documents, generated motion and CamBam output; no source model ownership |
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
| `mcp_adapter/` | Optional client protocol, document sessions and workspace transport. | Calls the owners above; does not own their domain rules. |

Current root files classified by that target map:

| Current files | Target owner |
| --- | --- |
| `cambam_project.py`, `cambam_transfer.py`, `entity_core.py`, `cad_entities.py`, `region.py`, `cam_entities.py`, `cad_transformations.py`, `cambam_reader.py`, `cambam_writer.py` | `native/` |
| `planar.py`, `_planar_shapely.py`, `stock.py`, reusable formulas in `machining_calculations.py` | `cam_core/` |
| `machining_recommendations.py`, `machining_planning.py`, future rest/V-carve strategies | `cam_extensions/` |
| `cam_core/rc01.py` | Current pure reference implementation; exact recipe later in `cam_extensions/reference_jobs/`, reusable verifier/motion values stay in `cam_core/` |
| `__init__.py`, `cambam_entities.py` | Root public/compatibility facade; implementations move behind it |

This is one distribution, not a plugin architecture. Under a package, use the
domain name (`project.py`, `region.py`, `stock.py`, `rest.py`, `vcarve.py`) rather
than repeating its package prefix. Keep `cambam_` only where a root compatibility
name or a cross-system adapter needs to identify CamBam explicitly. The pure
RC01 generator/verifier currently remains in `cam_core/rc01.py`; extracting its
general motion/stock contracts from its exact job is required before relocating
the recipe to `cam_extensions/reference_jobs/`. Do not create empty target
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
   This is a distinct follow-up after RC01's first CamBam output trial has fixed
   the actual adapter needs; a mass rename during that trial would mix semantic
   output failures with import churn.
3. **Detached and extended consolidation:** move root `planar.py`, `stock.py`
   and reusable machining math into `cam_core/`; place recommendation/pass
   planning and later rest/V-carve strategies in `cam_extensions/`. Split RC01's
   exact recipe from generic motion/verification only when another consumer uses
   those contracts. Check dependency direction, focused suites, wheel contents
   and imports outside the source tree. This waits for a concrete second consumer
   or the E/N output findings, so the abstraction follows demonstrated reuse.

Each move updates the owner table, callers, packaging, runbook and review evidence
in the same increment. Existing root modules remain authoritative until moved;
the target table does not imply capabilities or imports that already exist.

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

### Execution boundary and future CAM core

The current document pipeline authors native geometry/MOP instructions and exports
`.cb`; CamBam then generates toolpaths and posts G-code. Document fidelity is not
motion equivalence or evidence of removed stock. Detached planar and stock helpers
below do not change that execution boundary: supplied section motions are not an
implemented general XYZ generator, route optimizer or postprocessor.

The [execution architecture proposal](REST_MACHINING_PLAN.md#execution-architecture-refinement---2026-09-23)
defines a future document-independent motion/stock core with separate strategies,
verification and CamBam/direct-G-code output adapters. Native-MOP authoring remains
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
emitted motion. Existing root-level `stock`, `planar` and machining modules
remain active owners until their staged migration.

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
