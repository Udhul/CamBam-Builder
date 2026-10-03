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
| `cambam_builder/cam_core/` | Document-independent CAM planning, motion and stock analysis; `replay.py` owns ordered XYZ/cut-sweep values, `ordered_job.py` owns caller-supplied stage/state and decoded stock auditing, `volume3d.py` owns bounded layered 3D stock, `surface3d.py` owns affine-plane and spherical-bowl ball contact/stock, `inlay.py` owns bounded circular paired-target assembly/stock, `occupancy.py` owns bounded tool-body/box clearance, and `curved_region.py` owns bounded circular-arc access/rest approximation | Exact nominal and curved endmill stock, stepped-volume, analytic sloped/bowl ball, circular inlay and holder/fixture evidence slices; no native entity, XML or MCP dependency. `inlay.audit_pair` is the documented output-orchestration exception: it calls `integrations.ordered_output.audit_files`. |
| `cambam_builder/cam_extensions/strategy.py` | Deterministic selection among separately audited ordered routes, including partial and infeasible outcomes | Policy over evidence records; no XML or native entity dependency |
| `cambam_builder/planar.py` and `_planar_shapely.py` | Detached nominal planar values, error/provenance policy, analytic feasible centers and private optional GEOS adapter | Pure planar geometry; no document or stock/path ownership |
| `cambam_builder/machining_calculations.py` | Pure unit-explicit milling formulas, partial-input constraint solving and derived RPM/feed machine caps | Arithmetic planning kernel; composed by the separate pass planner |
| `cambam_builder/machining_recommendations.py` | Immutable tool/material/machine contexts, provenance-bearing recommendations, user diameter tables and pluggable pure strategies | Recommendation selection only; contains no curated catalog, persistence, document mutation or safety claim |
| `cambam_builder/machining_planning.py` | Pure through-cut pass balancing and composition of recommendation profiles with formula/machine diagnostics | Candidate planning only; the existing MCP depth tool delegates here, while full profile construction remains a direct-Python API |
| `cambam_builder/native/writer.py` | XML ID assignment and layer/part traversal; delegates individual encoding to entities | Output structure and reference resolution |
| `cambam_builder/native/reader.py` | XML parsing, entity reconstruction, ID mapping and deferred parent/MOP linking | Import defaults, malformed data and round-trip reconstruction |
| `cambam_builder/integrations/cambam/` | Native `.cb` input/attachment, editable rest Pocket authoring/certification, M2 curved candidates, actual posted MOP-series normalization and bounded posted-stock comparison; depends on native model and detached CAM values | Bridge between native documents, generated motion and CamBam output; no source model ownership |
| `cambam_builder/integrations/direct_*.py` | Bounded headless V and RC01 reference-dialect writers, parsed-output audits and evidence manifests | Output adapters; consume detached plans/traces and preserve their target verifiers |
| `cambam_builder/integrations/{uccnc_m5,uccnc_reader,grbl_m5_reader,m5_portability,m5_decoded}.py` | Bounded controller fixture emission, independent complete-byte dialect decoding, common decoded-stage values and shared T1/T3 stock audit | Consume the detached M4 plan; no controller syntax in `cam_core` |
| `cambam_builder/integrations/{ordered_dialects,ordered_output}.py` and `integrations/cambam/native_ordered_job.py` | Reusable strict UCCNC/Grbl lowering and independent decoding, hash-bound output audit, and supported native-series-to-job mapping | Controller bytes/effects stay outside `cam_core`; native input retains its source/post freshness contract |
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

The current delivery is one distribution. Strategy/backend extension interfaces
do not require a discovery registry or additional distribution now. Under a package, use the
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

### Framework direction and extension principles

**User intent and engineering direction, 2026-09-26.** Build a reusable CAM
framework with strong, editable CamBam interchange, its own machining analysis
and path calculation, and supported combinations of native and generated work.
V-carving and combined rest machining drive the current implementation. Surface
and volume contour following, additional machining methods and versioned CamBam
behavior compatibility are intended extensions. The user's current machine and
manual workflow supply test cases, not the framework's capability ceiling.

**Expanded product direction, 2026-09-30; target contracts, not implementation.**
The user's illustrative consumers include pointed/flat/combined V carving,
varied cutter angles and depth caps, paired decorative inlays with
fit/glue/seating allowances and post-assembly
finishing, mixed endmill/V rest jobs, fully native rest MOP/primitive synthesis,
and shape-aware tool/path planning for ornaments, friezes and 3D reliefs.
The [expectation map and follow-up contracts](REST_MACHINING_PLAN.md#product-expectations-and-capability-follow-ups-2026-09-30)
own detailed product behavior and staged acceptance. PROGRESS owns priority.

**Fundamentals-first clarification, 2026-09-30.** These consumers expose domain
requirements; they do not enumerate the framework's possible workflows or define
its module boundaries. Develop reusable machining capabilities that callers can
compose for anticipated and previously unlisted jobs. Domain expertise should
inform the fundamentals and reveal missing semantics, rather than accumulate a
special core implementation for each ornament, inlay or machining recipe.

The enduring facts are design/protected geometry and allowed removal; bodies,
stock and coordinate/setup relations; cutter and non-cutting body geometry;
motion, operation effects and dependency order; geometric/contact/residual
queries with error bounds; process constraints; and provenance/validity evidence.
Strategies and scheduling policies consume those facts to propose work. Output
adapters and caller orchestration preserve their meaning at integration boundaries.
Pair fit, assembly and post-assembly finishing motivate reusable body relations,
insertion/contact and stock-transform queries. Inlay-specific allowance choices
belong in a domain consumer, not a conditional branch in generic stock replay.

An increment is accepted for the reusable capability it establishes, demonstrated
by a meaningful end-to-end consumer. Also exercise a materially different supported
geometry, tool choice or caller composition that challenges its assumptions;
varying only a filename is insufficient. Keep sample dimensions, motif topology,
tool counts and workflow order outside shared contracts. A new consumer that uses
already-supported fundamentals should compose them without changing core
verification. When it requires a genuinely missing fact or evaluator, extend
that owning contract and declare its capability/error limits. Never infer complete
domain coverage from a finite set of examples.

This direction does not select a universal representation or require a generic
workflow engine in advance. Establish narrow interfaces from domain semantics
and demonstrated reuse, while allowing future representations and strategies to
extend them without imposing today's example sequence on every caller.

The design surface/volume must have its own identity independent of cutter
profile and independent of derived MOP machining boundaries. Required removal,
allowed relief and evolving known-free stock remain distinct. Candidate tools,
paths and order are checked against that same target. Paired parts retain their
separate stock/setup identities plus an assembly relation and a specified final
finish plane. Feature-guided strategies and cost-aware bundle search consume
geometric/contact/stock capabilities; they do not redefine finish intent or
replace independent motion/coverage verification. General surface contact,
composite assembly stock, native derived-boundary binding and search are staged
foundation work. Current tool-defined V targets, circular inlays and supplied
candidate ranking retain the narrower guarantees stated below until implemented
extensions pass their own acceptance. No backend or new dependency is selected
by this product-direction refinement.

The tables above identify present owners; this section defines the target
contracts. A completed reference job does not imply these contracts are all
implemented. The existing [small shared model](REST_MACHINING_PLAN.md#small-shared-model)
and [target/stock semantics](REST_MACHINING_PLAN.md#canonical-target-construction-and-uncertainty)
remain the design basis. Progress and the next implementation slice belong in
[PROGRESS](PROGRESS.md#active-work-and-next-priority).

Keep these responsibilities distinct so that improving one method does not
require replacing the framework:

| Responsibility | Durable boundary |
| --- | --- |
| Design and operation intent | Preserve native CAD/MOP identity and raw states; normalize only supported semantics into explicit units, frames, target/protected geometry, tools and constraints. Native authoring remains useful independently of framework path support. |
| Path generation | Interchangeable strategies propose geometric tool poses and cutting passes. Raster, contour offsets, V tracing, future surface following and supplied native motion share the supported plan/evidence contracts, without requiring identical algorithms or paths. |
| Sequence, entry and links | A separate planning responsibility completes candidate passes with feasible access and dependencies. Callers may supply order or request scheduling; reordered or changed motion needs fresh stock/access evidence. |
| Stock and geometric queries | Target geometry, remaining material and fixtures have separate identities. Section, containment, cutter contact, sweep and residual queries declare representation and error limits. Strategies request capabilities, not a particular polygon or voxel storage layout. |
| Verification | Evaluate actual continuous motion and state transitions against target, stock, tool/holder and setup constraints. Accept supplied/generated motion without trusting the generator's own completion claim; retain independent analytic or alternative-reference checks. |
| Output and execution | Adapters lower semantic motion and state to CamBam carriers, controller commands or external transport. Decode and normalize the final output and any declared external effects before granting route-specific evidence. |
| Workflow | Library calls support analysis, planning, attachment and export independently. Applications and agents compose them; no mandatory file sequence, live session, GUI, MCP client or approval loop belongs in the machining core. |

**Path geometry and machine trajectory are separate.** CAM methods decide
where the tool should cut; ordering/link policies connect those cuts; feed
scheduling, acceleration, jerk, controller blending and machine kinematics
determine time-dependent execution. A future offline dynamics provider may
evaluate or propose timing under declared machine limits. It must not become a
dependency for pure geometry/rest analysis or tie every strategy to one motion
controller. Record permitted path deviation and recheck its effect on material
and clearance when smoothing or controller blending changes geometry.
[LinuxCNC's trajectory guide](https://linuxcnc.org/docs/stable/html/user/user-concepts.html)
illustrates why programmed coordinates and feed alone do not describe actual
timing or corner following; this is a design distinction, not a LinuxCNC
dependency.

**Represent capabilities without making current shortcuts universal.** The
present fixed-axis XYZ and planar/section methods are legitimate initial
capabilities. Their positive-depth convention, one-cylinder predecessor,
T1/T3 labels, safe point, file names, feeds and sample dimensions must not become
public job invariants. Explicit frames, tool geometry and ordered stage identity
belong in inputs. Future curves/poses, tools or operation effects require defined
semantics and a capable evaluator; unsupported combinations return a precise
diagnostic while native source remains usable. Unknown effects cannot be silently
dropped or treated as zero motion.

**Use more than one geometric representation when the problem requires it.**
Planar sets and depth sections remain effective for their supported domain.
Surface contact, general solids and evolving volume stock may need different
backends. A height field cannot represent every overhang or disconnected vertical
interval; a surface model alone does not represent remaining material. Choose a
backend from representative jobs, conservative error/occupancy requirements,
performance and maintenance evidence. Keep conversion error and source identity
explicit. Reuse established kernels and auditable algorithms where they fit;
own machining semantics and verification. Extensibility does not require writing
every geometry primitive here or introducing a speculative plugin registry.

**CamBam compatibility has named levels.** Preserve documents; resolve supported
operation intent; compare resulting geometry/removal and process semantics; and,
where required, reproduce a versioned native path behavior. Each has separate
tests. Equivalent final shape alone cannot certify safe entries or the same
intermediate stock. Exact native path/order replication remains a supported
development objective when a named compatibility case needs it, without becoming
a condition imposed on innovative strategies. A native motion provider is an
adapter and requires actual motion authority; it is not a core dependency.

**Promote abstractions by demonstrated reuse.** Implement the common contract
across direct inputs, native normalization and at least two distinct consumers
before declaring it stable. Keep fixture orchestration and process values in
examples/tests, and verification algorithms outside controller-specific modules.
Use small typed values and narrow callable interfaces, with declared capabilities,
versioned provenance and deterministic diagnostics. Geometric paths, stock
certificates, artifact lineage and observed execution are distinct records.
Capability limits protect honest evidence; they do not justify rejecting a new
workflow merely because a sample script did not use it.

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
implement this separation; M5 adds named controller fixtures, and the bounded
[ordered-job API](#reusable-ordered-job-output-and-verification) now composes
caller-supplied stages. Native-MOP authoring remains
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
Neither owner directly imports `CamBamProject`, native CAD/MOP entities, XML I/O
or MCP. One current layering exception is `cam_core.inlay.audit_pair`, which
imports `integrations.ordered_output.audit_files` locally to orchestrate two
program audits. Its assembly and stock calculations remain in the core; this
convenience function is workflow, not a reusable controller-independent kernel.
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

### Capability and public API boundary map

Session 1 review baseline: `d38d8acf46be68c27c2c1e109761667b5e8babfb`, with
the review repairs recorded in [REVIEW](REVIEW.md#branch-review-session-1---2026-09-27).
This map states implemented contracts, not mathematical certification. Direct
Python submodule functions below are caller entry points even when absent from
the root exports. Private helpers and exact reference recipes are distinguished
explicitly. Detailed numerical domains remain in the linked contracts below;
the initial behavior-to-test matrix and audit gaps belong in REVIEW.

| Capability / entry points | Owner and dependencies | Supported domain and exclusions | Status, evidence and extension seam |
| --- | --- | --- | --- |
| Document creation, mutation, XML: `CBProject`, `native.reader.read_cambam_bytes`, `native.writer.serialize_cambam_bytes` | `native.project/core/cad/region/cam/reader/writer/transfer/transformations`; NumPy transforms | Mutable CAD/MOP intent, typed supported XML fields, UUID relationships; no toolpath parity from XML success. Old root imports resolve to canonical objects; same-version snapshots only. | Public native API; identity and synthetic round-trip tests plus bounded CamBam observations. Add entity/field semantics here, never to an output adapter. Manual-tab parent references follow geometry ID renumbering. |
| Native intent versus actual motion: `native_variable_v.normalize_bytes`, `native_series.normalize_native_series`, `NativeSeries.to_trace` | `integrations.cambam`; native owners and replay | Explicit supported units/frames/tools, one bounded Part and supported MOP/post subset. Intent can normalize without proving removal; actual post parsing still requires replay. Unsupported effects and unresolved setup reject. | Public bounded adapters; source/candidate/post and setup freshness. No native algorithm parity beyond specifically observed cases. Extend normalization and emitted-motion evidence together. |
| Planar geometry/access: `planar.RegionSet`, `ErrorBudget`, `PlanarFrame`, `feasible_centers` | Root `planar`; private `_planar_shapely` adapter | Nominal planar regions, analytic primitives and optional pinned GEOS operations; explicit mm/inch input frames, millimetre results and uncertainty. Native curves require an adapter. | Public bounded foundation; analytic and conditional numerical evidence. Missing backend/capability is diagnostic, not clearance. Backend seam stays private. |
| Section stock: `stock.SectionTarget`, `SectionMotionPath`, `verify_section_motion`, `compose_target_rest_bounds` | Root `stock`; exact rational section geometry | Rectangular stock/target and supplied disk-sweep section motion; explicit section depth and units. Not an XYZ generator or a general solid engine. | Public analytic/conditional foundation. RC01 consumes it; future representations must preserve enclosure directions. |
| Ordered motion/stock: `replay.ToolProfile`, `Target`, `Operation`, `Trace`, `replay` | `cam_core.replay`; optional Shapely for Region targets | Fixed-axis cylinders and 90-degree pointed cones; bounded linear, XY arc and descending helix cuts. Positive target depth below Z=0; negative cutting Z. No general ramps, rising arcs or arbitrary cutter bodies. | Public detached supplied-motion API, no generator required. Target sequences are immutable tuples; constructors and replay enforce different gates. Add supported sweep models here, process rules remain separate. |
| Cutter/contact/3D evidence: `v_region.VProfile`, `surface3d.contact_tip_z/bowl_contact_tip_z/replay_stages`, `volume3d.LayeredTarget/replay_stages` | `cam_core`; analytic profiles/contact and conservative cells/layers | Pointed/flat/rounded V; ball on an affine plane or spherical bowl; bounded axis-aligned stepped targets. Explicit pitch/error bounds, no overhang/freeform claim. | Public bounded kernels, not a general 3D backend. Direct analytic references and decoded-job evidence; S2 audits numerical guarantees. |
| Non-cutting occupancy: `occupancy.ToolBody`, `OccupancySetup`, `verify` | `cam_core.occupancy` and replay segments | Fixed-axis cylindrical tool bands and axis-aligned stock/fixture boxes; continuous supported travel, separate cutter/shank/holder checks. Unsupported transition motion rejects when occupancy is requested. | Public bounded verification. Cutting-stock pass alone says nothing about holder/fixture clearance. Extend for a named unsupported fixture/job. |
| Primary V: `vcarve.generate_slot/verify_slot`, `tapered_vcarve.generate/verify`, `v_region.plan/verify/section_report/volume_bounds` | Current `cam_core` strategy modules and replay; Region planner requires Shapely | Slot and increasing-X straight variable-depth families use pointed 90-degree cones; Region planner accepts polygon/curved bounds and pointed/flat/rounded profiles, raster/offset fills. No roughing predecessor needed to plan or analyze. Finite-tool residual is partial; no fitting center yields infeasible. | Public bounded machining capabilities. Target/profile/strategy are separate values; no GUI, files or fixture IDs required. General rest smoothing and globally optimal paths are not implemented. |
| Rest analysis/generation: `convex_rest.generate`, `polygon_rest.generate`, `curved_region.approximate/generate`, `v_region.with_prior` | Current `cam_core`; replayed supplied predecessor, target and cutter | Convex/straight/curved planar domains with explicit numerical envelopes. A source/motion fingerprint is required where exposed. `with_prior` requires one complete cylindrical operation on the same target; earlier overcut is not forgiven by later removal. | Public bounded strategies and stock queries. Cleared-overlap/path fitting is distinct from changing target edges; generic conditional smoothing remains an extension. |
| Feature-aware V/rest: `planar_rest.generate`, `cutting_sweep_clear`, `Candidate.section/floor_cusp` | `cam_core.planar_rest`; fixed design and composed stock | Contact/medial guidance, conservative union-air pruning and located residual/cusp evidence; see [RP01](#feature-aware-planar-vrest-candidates-rp01). | Public bounded strategy; high links and explicit axial/setup gates. Partial coverage and retained overlap remain visible. |
| Reference jobs: `rc01.generate/verify`, `mixed.verify_mixed`, `inlay.generate/assembly/audit_pair` | Current `cam_core`; section/replay/profile values; paired audit also calls output integration | RC01 nominal rectangle/island/process recipe; mixed RC01/slot recipe; circular pointed-V receiver/plug family with independent stocks. Reference dimensions/tool recipes are not generic framework invariants. | Public reference conveniences with bounded offline evidence. Keep reusable geometry/stock separate; move orchestration when a concrete caller requires it. `audit_pair` is the explicit layering exception above. |
| Ornamental paired stock: `ornamental_inlay.Design/verify_pair/finish_envelope`, `tapered_inlay.Design/PartStock/section/verify_pair`, `composite_inlay.Assembly/Finish/generate/verify` | `cam_core` stock/assembly; complete-byte audits in `integrations.inlay_output` | [Straight-wall](#ornamental-straight-wall-paired-stock-and-assembly-in01), [profile-aware tapered](#tapered-profile-aware-paired-stock-in01) and [composite facing](#composite-stock-facing-and-final-inlay-verification-in01) with supplied independent parts and continuous rigid insertion. | Public bounded offline verification of facing plane, motif enclosures and retained nominal core/floor; declared sanding envelope remains separate. Physical fit and controller execution have separate gates. |
| Recommendation/pass policy: root `machining_calculations`, `machining_recommendations`, `machining_planning` | Formula kernel, immutable contexts, pluggable recommendation strategy | Unit-explicit inputs, feed/RPM/range constraints and through-cut pass planning. No toolpath generation, stock clearance or curated material authority. | Existing public APIs, unchanged owners followed as dependencies. A candidate recommendation is not a stock certificate. |
| Route selection: `cam_extensions.strategy.select_strategy` | Detached policy over caller-supplied `StageAudit` and residual bounds | Checks source/predecessor chain, required gates and completeness; ranks feasible then safe partial candidates by upper residual area, volume and declared tie order. Manual choice cannot select an unsafe candidate. | Public supplied-candidate ranking, not bundle search, cutting-time optimization or independent validation of caller assertions. No global optimum claim. |
| Caller-owned execution: `ordered_job.Job/Stage/Transition/AxialLimits`, `from_prior_v`, `audit` | `cam_core.ordered_job`; replay, V and bounded 3D evaluators | Caller labels/tools/order/feed/RPM; resolved mm/G54 translation. All-cylinder, standalone/cumulative all-V, bounded mixed cylinder/V, homogeneous layered/surface/inlay sequences have distinct evaluators. Mixed V requires explicit axial/entry limits and whole-tool setup. | Public bounded composition. `audit` consumes already-decoded values; `ordered_output` is the complete-byte/native-binding authority. Unsupported mixtures must not be read as stock success. |
| Editable native rest: `rest_boundaries.derive`, `native_rest.prepare/author/audit`, `RestBinding` | Detached boundary derivation plus native integration | [NR01 contract](#editable-native-rest-boundaries-and-mops-nr01); compensated planar Regions, native Pocket and original-design certification | Public bounded API; synthetic offline evidence and actual native planner acceptance are separate. |
| Native/generated composition: `native_ordered_job.from_native_series/from_native_v`, `NativeBinding.check` | Native integration to detached ordered job | Repeated tools and supported native stage order; one native cylinder plus source-bound V finish. `from_native_circle_cleanup` is specifically the diameter-24/depth-2/T1-T2 observed recipe. | Public bounded adapter plus named reference helper. Source/current-post binding is required at output; no mandatory global document session. |
| Output and evidence: `ordered_dialects.render/decode`, `ordered_output.emit/audit_files/write_bundle/audit_bundle`; `direct_variable_v`/`direct_rc01` | `integrations`; core values, strict independent decoders | UCCNC split and Grbl pause profiles, explicit offset/transition effects, 4/6 decimal coordinates. Older `direct_*` writers use a strict reference dialect, not machine profiles. File helpers have fixed artifacts; in-memory emit/decode do not require them. | Public output adapters and bounded reference conveniences. `direct_rc01` reuses the CamBam Default-post reader; it is not fully independent of that adapter package. Runtime/physical setup remain unassessed. |
| Agent protocol: document/MOP tools and schema | `mcp_adapter.service/schema/server/paths`; native project and existing planning owners | Volatile document sessions, validated requests and workspace transport. No exposure of arbitrary detached CAM/replay/controller entry points through MCP. | Public versioned tool contract, separate from direct Python API. Mirror supported native authoring changes in schema/service/tests; do not relocate machining truth into protocol handlers. |

**Common caller contract.** Native document units follow the document and may
include unresolved `Auto` properties; detached CAM uses millimetres, mm/min and
RPM with stock top Z=0 unless an explicit adapter resolves a frame. Do not pass
native target-depth words straight to a positive-depth target constructor.
Region `VPath` uses positive penetration, while its `VMotion` uses negative tip Z.
Stock bounds describe the initial material; the target describes permitted
removal/protected boundaries. Matching bounding boxes alone is not target parity.
Shared field names do not make native MOPs, recommendation `ToolProfile`, replay
`ToolProfile` and `VProfile` interchangeable. Each entry point validates its own
supported cutter model.

Native projects and MCP document registries are mutable. Detached values use
frozen dataclasses/tuples; frozen containers are not a security boundary against
deliberate `object.__setattr__` or forged evidence. `StageAudit` is a caller-supplied
assertion, not an opaque certificate. Recompute derived plans/audits after source,
tool, target, order or setup changes. Output reports are mutable dictionaries:
retain the source/bytes and re-audit, rather than editing a report into a pass.
`planar` exposes structured diagnostics; most CAM contract failures raise
`ValueError`, and file APIs can additionally raise I/O/parse errors. A `partial`
plan, `unsupported` stock evaluator, `not_evaluated` gate and failed check are
different outcomes. File emission success or an outer `ordered_output_pass`
does not turn every nested gate into a pass.

**Detached trace.** `tapered_vcarve.generate(TaperedRequest(...))` validates a
caller target/tool and returns a plan without document state or predecessor.
`trace_for` supplies detached operations/events/motions; `replay.replay` accepts
that supplied trace independently, and `TaperedResult` measures residual.
`output_trace` adds the bounded reference process/setup. `direct_variable_v.render`
lowers it; its text audit parses complete output, compares semantic moves and
replays decoded stock. `build_program`/`audit_program` add optional native input
and file hashes. The fixed reference start/feed/tool are adapter limits, not
requirements of primary V planning. A separate reusable generated route is
`v_region.plan` + supplied cylindrical `Trace` -> `ordered_job.from_prior_v` ->
`ordered_output.emit` -> independent decode -> `ordered_job.audit` -> replay and
V residual; caller tool names and translations are exercised by ordered-job tests.

**Native plus generated trace.** `normalize_native_series(source, candidate,
post, setup=...)` binds native intent and actual posted motion. `to_trace` and
`from_native_v` replay the cylinder and check the generated V target/source;
the resulting job is detached. `ordered_output.emit/write_bundle` requires
`NativeBinding.check`, decodes emitted bytes, compares each move and replays
decoded predecessor stock before measuring the decoded V finish. `audit_bundle`
rechecks current source/post/setup and complete bytes. The caller owns file
selection/order and physical tool installation; neither native MOP intent nor
the transition completion token proves actual execution. Analysis-only callers
may stop before attachment/output; supplied-motion callers may start at replay.

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
controller fixtures. `cam_core.v_region.complete_motion` assembles safe travel
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
stock/access/residual audit, now a compatibility wrapper over
`cam_core.ordered_job.audit`; a final safe T1 stage and synthetic changer
effect travel are checked without inferring physical completion. The manual
fixture resolves a +4.5 mm CAM surface through a declared -4.5 mm work map;
the mixed fixture uses fixed G54 and external table-derived length values.
`integrations.m5_decoded` owns shared decoded-stage values, still containing
G-code numbers/startup words rather than a complete core semantic state model.
The Grbl fixture checks fixed offset values and assumes each stage's registered
tip; the synthetic host effect is validated against its pinned writer model.
These limits motivate the
[ordered-job packet](REST_MACHINING_PLAN.md#next-implementation-packet-reusable-ordered-jobs-and-verification).
Dialect readers do not depend on one another. Generic native MOP generation, further
controller dialects, controller runtime parity and physical machine acceptance
are outside this bounded fixture set.

### Reusable ordered-job output and verification

**Implemented bounded contract, 2026-09-26.**
`cam_core.ordered_job` accepts immutable `Job`, `Stage`, `JobMove` and
`Transition` values. A job names its source revision, ordered operation IDs,
resolved tools, explicit program-to-G54 translation, units, stock availability,
per-stage spindle/feed and length state, and predecessor fingerprints. Stages
may end at a different safe tip; the next stage begins there. A transition
declares manual split or in-program pause, or a synthetic host effect and its
expected travel. Completion tokens are assumptions, not observed events.
Operator transitions accept no modeled travel or effect; a modeled effect must
use the synthetic-host path and supply independently checked effect bytes.
Native `NativeSeries` can be mapped through `native_ordered_job.from_native_series`
after strict source/candidate/post normalization and replay; its original
document and posted trace remain separate. Native output and re-audit require
`NativeBinding`, which rechecks those three current files; a supplied stale job
cannot certify an edited source. Direct Python V plans use
`ordered_job.from_prior_v` with either existing raster or offset strategy.
For one native cylindrical stage, `native_ordered_job.from_native_v` can attach
one generated terminal V plan to the same source and stock. It replays the
complete native post before admitting the V plan. `NativeBinding` checks the
native stage against its unchanged source/candidate/post and the V target against
the original planar source target. A fresh native post is required after a
candidate edit. This hybrid route accepts one supported planar native cylinder
before the terminal V finish. Level XY G2/G3 cuts are representable; the
single-depth native Pocket G3 fixture has a fresh accepted CamBam post and
decoded hybrid stock audit. The bounded Circle Pocket route below also admits
posted descending G2 helices and a generated cylindrical cleanup.
CamBam's vertical G0 retract is represented as `rapid_retract`: the output
retains G0, while stock replay applies the same cleared-column proof as a
feed retract. A low XY rapid still fails the native normalizer.

`ordered_dialects.render/decode` lowers and independently reads complete
UCCNC split files or one Grbl v1.1 program for supported G0/G1 and level or
descending XY G2/G3 cuts with relative I/J centers. Both readers preserve the decoded center,
direction, endpoints and feed. The auditor compares them with the native post
and replays the decoded arc rather than the planned source center.
Tool IDs, stage count, RPM, feeds, endpoints and translation are caller values.
Grbl length-state changes move displayed work Z at stationary machine position;
the adapter emits and decodes a real compensating rapid before the stage, and
the common auditor checks the effective physical tip from declared tool length
and offset. UCCNC split uses G49 and an asserted registered tip at each file
boundary. A pause or stage comment does not install a tool. Unknown commands,
missing compensation, changed offsets and unsupported boundary policies fail.
Both UCCNC writers explicitly select G94 (units per minute); strict controller
readers require their complete startup and literal LF block delimiters. The
older M5 Grbl portability fixture also carries position/offset state and decodes
its compensating rapids; its synthetic changer retains the active offset until
the following NC command applies the new tool length.

`ordered_output.emit/audit_files/write_bundle/audit_bundle` bind versioned
profile, numerical policy, job/source and prefix fingerprints, final program
hashes and any external-effect hashes. The controller-neutral auditor compares
every decoded move to intended motion within 0.000051 mm, then replays actual
decoded cylindrical cuts stage by stage, standalone or cumulative all-V plans,
or the bounded mixed cylindrical/V composition below. It reuses `replay` and `v_region` geometry
evaluators and never calls the selected generator to fill absent decoded
motion. A stockless job can retain source and motion comparison while stock
evidence is `not_evaluated`; an unsupported stock stage mix is `unsupported`.
Synthetic external travel is independently decoded and must match the declared
model and clear the supplied flat fixture-top plane. Runtime and physical setup
remain `not_evaluated`. Re-audit after a native edit requires a freshly
normalized source/job; an old bundle's fingerprint cannot certify it.

The low-level `ordered_job.audit` takes decoded values, not source or program
files. Native document fidelity and external-effect evidence therefore remain
`not_evaluated` there. Only `ordered_output` promotes those fields after a concrete
`NativeBinding` and complete effect-byte check. Native binding re-observes the
source/candidate/post, requires the native frame and initial tip, and checks the
actual native prefix. Replacing an observation's fields while retaining its
input hashes does not preserve acceptance. Native stock audits require explicit
Part stock containing the replay target at stock top Z=0; stockless parsing and
motion comparison remain available without stock authority.

Verifier `ordered-job-v6-mixed-v-stock` includes the program/work frame,
units and matching tolerance in prefix identity as well as whole-job identity.
Pre-review ordered bundles must be regenerated and audited under this verifier;
their stored reports cannot be carried forward. Prior actual CamBam observations
retain only their unchanged source/post scope. The exact-rendering direct RC01
and variable-V auditors preserve original line endings when comparing bytes;
a rehashed newline conversion cannot inherit canonical-byte acceptance.

For a cylindrical arc, `replay.arc_segments` encloses the continuous
centerline by chords with at most 0.0001 mm sagitta, plus the posted endpoint
radius mismatch and a numeric margin. Outer cutter radius uses that path error
for protected-boundary checks; inner radius uses it for removed-stock and
residual-upper checks. A descending Z helix uses linear Z interpolation along
each arc chord; section stock starts only where the tip reaches that depth.
Source arc endpoints are exact replay endpoints. Curved rapid travel, rising
arcs, straight ramps, full-circle same-XY words and unresolved arc radius fail
closed. GEOS buffer/Boolean results remain conditional numerical
evidence, not a physical controller trajectory guarantee. Linear `JobMove`
representation remains stable; verifier-version changes invalidate old job evidence.

The planar geometry domain supports fixed-axis cylindrical replay and standalone
or cumulative pointed, flat or rounded Region-V stages and bounded cylinder/V
interleavings against one fixed design;
the separate stepped-volume contract below
extends the same decoded boundary. Generic native Pocket/Profile
path reproduction, cross-stage mixed cleared-access credit, arbitrary macros,
non-flat fixtures, rotations/kinematics and controller runtime parity require
named extensions. The [ordered-job packet](REST_MACHINING_PLAN.md#next-implementation-packet-reusable-ordered-jobs-and-verification)
and [verification](REVIEW.md#reusable-ordered-jobs-and-verification---2026-09-26)
record the acceptance scope.

#### Standalone Region-V virgin-stock contract

Construct one `Stage` with `v_plan`, `source_revision=plan.fingerprint`, and
`JobMove` values from `v_region.complete_motion(plan, initial_tip)`. Construct
`Job(plan.target.source_id, (stage,), initial_tip, stock_present=True)` with the
resolved program frame and translation. Existing constructors suffice; this
route does not invent a cylindrical predecessor. `stock_present=True` declares
virgin material at program Z=0 throughout the finite Region target and its depth
cap. The target's safe/outer planar bounds, cap and frozen design angle define the measured
V volume; the complement (including holes) is protected. This is a caller stock
assertion, not observation of physical stock. `stock_present=False` returns stock
`not_evaluated`, never virgin-stock success. Every V stage's revision must match
its plan fingerprint and its target source ID must match the job source, even
when stock evidence is not evaluated. Native jobs still require `NativeBinding`.

Region-V stock supports linear G0/G1 motion only. A V `JobMove` with an arc is
refused rather than stock-replayed as its endpoint chord; supported cylindrical
arc stages retain their separate arc replay contract.
The auditor reconstructs paths from translated decoded coordinates and verifies
the resulting motion/profile before computing evidence. Entry is a feed plunge
along the same XY column as the first cut vertex into virgin material; retract
follows the last vertex's column, and every decoded rapid (including the complete
approach and final return) stays strictly above Z=0. Omitted exterior links must
retain their rapid role. The byte audit also checks V rapids above the declared
flat fixture-top plane after resolving the physical/program transform. A rounded
return on Z=0 fails even within motion-matching tolerance.
The continuous segment/boundary test protects every cutter height, the depth
cap, cutting length and maximum radius; variable-depth cuts also obey the existing
2 mm/mm slope bound. Authored plans also require finite positive safe height and
stepover and a finite margin greater than 0.00001 mm; a forged negative or NaN
margin cannot disable protected containment. It proves geometric access, not an
engagement load, chip
evacuation, holder/fixture clearance or controller execution. The optional
occupancy setup retains its separate stock/tool/body validation.

`stock_access_residual.status=pass` means decoded verification passed, while
`plan_status=partial` remains explicit. Finite spacing/tips and short flutes can
leave material; pass does not mean zero residual. `initial_stock=virgin` has
empty `cuts_by_prefix` and no `decoded_prior_fingerprint`. Initial and final
residual evidence uses `prior_section_1_mm2`, `section_1_mm2`,
`prior_volume_mm3`, `volume_mm3` and `decoded_v_plan_fingerprint`. The historic
section field names are retained; `section_depth_mm=min(1, cap_depth)` specifies
their actual depth, including shallow targets. Each section tuple is residual
lower, residual upper, conservative outer-sweep area outside the safe section.
Volume bounds use eight slabs and measure only the fixed V design. Initial
bounds contain no assumed removed material. These are conditional GEOS bounds
with chord/numeric allowances, not interval-certified Booleans; independent
analytic witnesses assess their enclosure. No fitting tool yields an
`infeasible` plan with no executable paths and cannot be emitted as a cutting job.

The one-cylinder-then-V route retains its replayed prefix, initial bounds and
decoded predecessor fingerprint. Homogeneous all-V sequences use the cumulative
contract below. Other cylinder/V interleavings use the bounded MX01 contract.
The v6
verifier invalidates all earlier ordered bundles: regenerate their reports and
program binding before reuse. Unchanged actual source/post observations retain
their original byte scope. Synthetic proof and misuse evidence live in
`tests/test_standalone_v.py` and the dated review; machine acceptance is separate.

#### Cumulative Region-V virgin-stock contract (MV01)

`cam_core.v_region.VSequence(plans)` accepts a nonempty ordered tuple of
`VPlan` values. The frozen container stores the supplied plans, with no cached
geometry or caller-supplied cleared-stock claim. Construction and section
consumption verify every plan and require identical target fingerprints: source,
safe/outer opening geometry, depth cap, design angle, frame and evaluator identity.
Freeze legacy tool-defined targets once before composing cutter alternatives.
Sequence fingerprints bind the ordered plan fingerprints; a reordered sequence
requires fresh evidence even when its final geometric union is unchanged.

The existing `section_evidence`, `section_report` and `volume_bounds` APIs
accept a sequence. Each stage contributes its own pointed, flat or rounded
profile at its actual path penetration. Known-free inner/outer geometry is the
union of those bounded sweeps, never a sum of areas or a replacement by the last
stage. Duplicate passes preserve stock; a safe additional sweep preserves stock
inclusion. `final=False` on a sequence excludes all V sweeps and measures the
initial virgin target. A prefix is represented by `VSequence(plans[:n])`.
Residual bounds always refer to the original fixed design. Whole-segment,
all-height cutting-profile protection is re-established for every plan; no
earlier cut grants permission to cross an island, wall or depth cap.

For decoded output, callers supply all-V `Job.stages` through the existing
`Stage`/`JobMove` constructors and explicit `Transition` values. The auditor
checks source revisions, source/frame/design identity, tool/feed/spindle/offset
state, ordered boundaries and complete decoded motion. It reconstructs and
verifies each stage's decoded V paths, including strictly above-stock rapid
approach, links and return, before creating cumulative stock. The byte integration
retains complete-program and declared fixture-top authority. Raster and offset
plans, repeated tools and more than two stages use the same evaluator.

For more than one V stage, `stock_access_residual` reports
`scope="decoded cumulative Region V from virgin stock"`, `initial_stock="virgin"`
and `plan_status="partial"`. It carries `design_fingerprint`,
`design_angle_degrees`, `design_frame`, `section_depth_mm`, initial
`prior_section_1_mm2`/`prior_volume_mm3`, final `section_1_mm2`/`volume_mm3`,
`decoded_sequence_fingerprint` and ordered `prefixes`. Each prefix records
`stage_id`, `prefix_fingerprint`, `decoded_v_plan_fingerprint`,
`decoded_sequence_fingerprint`, `plan_status`, `section_1_mm2` and `volume_mm3`.
The historical section keys use `min(1, cap_depth)`; section tuples contain
residual lower/upper and possible-overcut area, and volume tuples contain
residual lower/upper using eight slabs. Empty initial sweep and every decoded
prefix are separately identifiable. Single-stage and one-cylinder/V report
shapes retain their existing contracts.

Stockless jobs retain `not_evaluated`; unknown stock evaluator sequences report
`unsupported`. Cylinder/V interleavings use the bounded MX01 contract below. Finite
spacing and profile reach retain partial stock and bound uncertainty; full-target
plans may recut removed material. This verifies cumulative effect, not rest-only
optimization, removal efficiency or a completeness claim. Bounds retain the
existing conditional GEOS/chord/numeric policy, not interval-certified topology.
Slab bounds use nested design and sweep sections rather than assuming monotone
residual area. Cutting-profile protection does not establish holder clearance,
engagement limits, controller runtime or physical setup; optional occupancy and
production acceptance retain their separate gates. Synthetic witnesses live in
`tests/test_multistage_v.py`; standalone and cylinder/V regressions remain owners
of their earlier compatibility scope.

#### Mixed cylindrical/V composition and whole-tool access (MX01)

`v_region.VComposition(target, stages)` accepts one frozen `VTarget` and an
ordered nonempty tuple of verified `VPlan` values or single-operation cylinder
`replay.Trace` values. Construction and evidence consumption re-establish each
stage's source/frame and original design. Cylinder targets must equal the
original opening and full depth cap, including holes; derived center Regions
cannot replace that target. Every cylinder entry/cut must keep its full radius,
path error and `depth * design.tangent` inside the original boundary. V stages
retain DT01 full-profile protection. No stage can authorize cutting protected
walls, islands or stock below the cap.

The existing located section and residual-volume queries union all actual
cylinder and per-profile V sweeps. Prefixes use `stages[:n]`; `final=False`
measures virgin design stock. Overlap and repeated passes are unioned, never
summed. Cylinder stock and clearance are replayed from each complete stage;
a stage may use its own proved clearance but cannot infer a cavity from earlier
V or cylinder stages. [RP01](#feature-aware-planar-vrest-candidates-rp01) adds
cutting-profile union clearance for air pruning; cross-stage low descent/link
and body cavity credit still require a concrete independently verified consumer.

The ordered auditor reconstructs each stage from decoded coordinates. Beyond
the historical one-cylinder/terminal-V compatibility slice, mixed stages require
`Stage.axial_limits=AxialLimits(max_stepdown_mm, max_entry_depth_mm)` on every
stage and a matching `Job.occupancy_setup`. The axial gate supports linear
motion, vertical stock-cutting entry and exact XY retraces. Tip advance on a new
segment starts at virgin depth zero; a deeper segment can credit one earlier
identical-profile segment with the same XY endpoints (either direction).
Endpoint depths bound the full linear-depth advance. Earlier crossing-depth
passes keep separate witnesses: their deepest endpoints must not be combined
into fictitious interior clearance. Entry credits only prior cutting endpoints,
never itself, and separately obeys the absolute plunge-depth limit. Changed or
omitted earlier passes fail independently of final depth-cap protection. This
axial comparison allows 1e-7 mm numerical tolerance on decoded tip depths. The
conservative policy does not grant credit for nearby/partially overlapping paths,
different profiles, arcs, or unknown engagement/load limits.

`v_region.depth_passes(plan, max_stepdown_mm, depth_cap_mm=None)` clips verified
XY paths to shallow-to-deep axial levels, preserving their original design.
A positive optional cap at or below the target cap leaves a declared axial
allowance; a later stage may finish deeper. This is an axial scheduling helper,
not normal-to-wall allowance, rest-only generation or a process recommendation.
Callers still declare entry permission, feed/RPM and body/setup. Unknown radial
engagement, chip load and machine limits remain unassessed.

The existing continuous occupancy verifier checks every decoded entry, cut,
retract and rapid against closed fixture boxes for cutter/shank/holder bands.
Shank and holder clear the initial stock box without cavity credit; tool bands
must match each stage's cutting length and enclose its actual cutter radius.
Modeled/decoded transition travel remains unsupported with occupancy. Thus a
tip or cutting-stock success alone cannot establish whole-tool clearance.

Mixed `stock_access_residual` uses
`scope="decoded cumulative cylindrical/Region V from virgin stock"`,
`initial_stock="virgin"`, `plan_status="partial"`, the same fixed-design and
initial/final section/volume keys as MV01, and ordered `prefixes` containing
`stage_id`, `prefix_fingerprint`, `decoded_stage_fingerprint`,
`decoded_sequence_fingerprint`, `plan_status`, `section_1_mm2`, `volume_mm3`.
`axial_process_limits` reports declared limits and actual maximum advance/plunge
per stage, with engagement/load `not_evaluated`; `tool_fixture_occupancy` remains
separate. `access_policy` names independent cylinder replay and above-stock V
links; `cross_stage_cavity_credit="not_evaluated"` prevents inferred union access.
Verifier v6 invalidates stored ordered bundles, including bundles
whose NC bytes remain identical. Legacy one-cylinder/V and all-V construction
retain their previous optional-setup behavior and report shapes.

Tests in `tests/test_mixed_v_composition.py` own synthetic composition and safety
witnesses; the [runbook](DEVELOPMENT.md#mixed-cylindricalv-composition-checks)
owns executable checks. Conditional GEOS/numerical bounds, partial residual,
operator tool-installation assumptions and unevaluated controller/physical
acceptance remain explicit. This shared capability contains no motif or fixed
tool-count discriminator; different supported consumers use the same queries.

### Feature-aware planar V/rest candidates (RP01)

`cam_core.planar_rest.generate(prior, tool, ...)` consumes a verified
`v_region.VComposition` against one frozen design. Contact contours retain every
guide vertex and all components/island rings; sampled-boundary Voronoi edges
guide narrow detail. This is not an exact medial-axis or global search claim.
Continuous all-height `v_region.verify` checks each proposed segment before
pruning. The design, pure residual, known-free stock and center guides remain
separate. No design corner is filleted or island removed to accommodate a path.

Controls declare maximum stepover, floor cusp, XY sampling, margin, safe height
and finite guide/site/path budgets. Pitch is at most `2 * tool.radius(cusp)`;
this is a floor-spacing criterion, not an assumed global coverage bound.
`Candidate.floor_cusp()` checks actual composed inner removal at `cap - cusp`
against the original capped-floor outer enclosure. Empty `unproved` geometry
establishes the axial cusp bound over that floor; otherwise it locates the gap
and reports partial. Walls and uncapped narrow details require separate located
section residuals. Retained guide vertices bound XY guide deviation by
`sagitta_mm + sqrt(2) * 0.5e-7` mm; this does not bound medial-axis approximation
error or certify machined-wall finish. The deviation is relative to constructed
polygonal guides; curved-source enclosures remain owned by `VTarget`, without an
additional native-offset Hausdorff claim. No fitted arc or generic smoothing is
introduced.

`cutting_sweep_clear(prior, tool, a, b, slabs=8)` proves cutting-profile clearance
through the union of verified cylinder/V stages. Each slab compares an inflated
maximum-penetration candidate footprint at the upper plane with prior inner
removal at the lower plane. Monotone sections establish intervening heights;
failure means unproved. Exact same-profile XY retraces, including reverse
orientation and shallower endpoint depths, independently prove clearance where
a pointed floor has zero area. This query grants no rapid, body or fixture
permission. Candidate pruning omits only such proved-air segments; every retained
segment can still contain air cutting. Pure-rest clipping alone is not clearance.

The returned `Candidate` holds the verified ordinary `VPlan`, prior composition,
proposed/omitted XY cutting lengths and guide error. `section(depth)` exposes
located residuals and conditional new-removal/overlap area intervals using
candidate inner minus prior outer, candidate outer minus prior inner, and
corresponding intersections. `composition` adds retained sweeps to all prior
stages. These reports recompute their stock evidence; stored result geometry is
not an independent authority. Empty retained paths mean no admissible unproved-
clear guide was found, not a proof that the target is unreachable or complete.

Entry is target-contained vertical stock cutting; links retract above stock.
Use explicit `v_region.depth_passes`, axial/entry limits and whole-tool setup for
ordered output. Cross-stage low linking and body cavity credit remain unsupported;
this candidate does not need them. Unknown engagement/load, controller/runtime,
physical finish, optimization and broader native representability remain separate.
Tests generate a lobed/island/valley/broad-cap frieze and challenge reuse with a
different-angle cylinder/V island job through both output dialects. Measured
coverage, cost and acceptance belong to the
[RP01 evidence](REVIEW.md#rp01-feature-aware-planar-rest-candidates---2026-10-02).

### Editable native rest boundaries and MOPs (NR01)

`cam_core.rest_boundaries.derive(trace, radius_mm=..., overlap_mm=...,
margin_mm=0.01)` consumes complete replayable cylindrical predecessor motion
against one planar Region target. It distinguishes pure floor rest, feasible
smaller-tool centers and native Pocket targets. Feasible centers lie inside the
original design eroded by radius plus margin. Reachable residual components
select nearby centers; outward compensation produces windows with deliberate
overlap into prior cleared space. Overlapping windows merge before authoring.
The original outline and holes remain unchanged. These are full-depth machining
boundaries, not a promise of rest-only motion or an entry/access certificate.

`RestBoundaries` retains the target, predecessor motion fingerprint, controls,
editable polygon rings, pure-rest area interval and reachable-rest area.
Window simplification uses `margin_mm / 4`, preserves topology and is followed
by nine-decimal interchange rounding and original-design containment checks.
The margin and polygonal sweep oracle have conditional GEOS numerical/topology
limits; no exact offset, universal finish tolerance or continuous fitted-path
claim is made. Empty, invalid, untraversable or non-useful windows reject.
Supported predecessors reach the target floor with constant-depth cylinder
sweeps; helical/variable-depth predecessors and V/rounded/surface cleanup need
their separate generated-output or interoperability routes.

`integrations.cambam.native_rest.prepare` first audits the pinned native
predecessor and original source stock, then derives a `RestBinding`.
Rectangular targets normalize to polygonal shells. `author` appends editable
Regions and one native Pocket with an explicitly numbered smaller endmill,
caller-supplied depth increment, clearance, spindle and feeds. It refuses output
aliases of original/predecessor inputs and strictly reopens the saved candidate.
The final predecessor gets a setup-only native MOP footer: its already-observed
positive-Z return followed by `M5`. Default otherwise delays that return under
the next MOP's section and omits the explicit stop before M6. The footer contains
no cutting motion; every cutting path remains native planner output. Its exact
state is certified against the predecessor post. Literal cutting transport and
unproved predecessor returns reject.
Each Region's persisted description records original identity, semantic source,
predecessor evidence and derivation fingerprint. The binding recomputes geometry
from fresh predecessor evidence; provenance text alone grants no authority.

The existing target-equality gate remains the default. Its explicit
`derived_binding=RestBinding` route certifies current window rings/provenance,
Pocket selection/type/cutter/floor, original CAD and complete predecessor motion
instead. The composed post must retain every predecessor event and movement,
ignoring only line numbers. An edited boundary, tool, floor, selection,
source/predecessor or transport header invalidates that certificate. Boundary
edits require fresh derivation/certification; valid parameter edits still require
a fresh actual post. `native_rest.audit` replays all actual moves against the
original target, reports cumulative section/volume residual and protected
overcut, and enforces a caller-declared positive new floor-removal minimum.
Virgin vertical entry is modeled stock cutting. A cleared descent must be
proved from actual preceding sweeps; overlap does not assume material absent.
`NativeBinding(..., derived_binding=binding)` extends both ordered controller
consumers through the same original-design certification.

Synthetic posts establish offline authoring/verifier behavior only. Actual
CamBam Pocket algorithm acceptance requires a complete externally observed post;
Engrave previews, CustomScript and literal-motion carriers do not establish it.
Arbitrary feature-guided or variable-Z paths are not representable by this
native Pocket contract. Controller runtime, engagement/load, whole-holder/setup
occupancy and physical machining require their own declared evidence. Current
acceptance and the observed complete native posts live in
[NR01 evidence](REVIEW.md#nr01-editable-native-rest-preparation---2026-10-02).

### Fixed V design and independent cutter contract (DT01)

`cam_core.v_region.VTarget` owns caller-supplied source identity, safe/outer
opening bounds (including islands), cap, design included angle and coordinate
frame. Its evaluator is an **ideal pointed V envelope**, truncated at the cap
with a flat floor: `section(t)` erodes the opening by
`t*tan(design_angle_degrees/2)`. Design-flat and design-rounded surfaces are
future evaluators; cutter tip style never selects the design convention.
Explicit `polygon(..., design_angle_degrees=90, frame="program")` and `curved`
constructors default to the program frame. Frames label the supplied coordinates;
no target-frame transform is inferred. Job translation retains its existing
program-to-output role.

Legacy constructors omitting the angle remain supported. `target.freeze(tool)`
maps their historical tool-defined intent once; `plan` and `VPlan` construction
perform that mapping automatically. Use `first_plan.target` or the returned
frozen value for cutter alternatives, rather than planning the unresolved
opening again. A legacy omitted frame remains unbound for existing native/XY
callers; callers migrating to frame-bound evidence must set it explicitly.
Unresolved `section(t, tangent)` remains a compatibility query; resolved targets
reject a conflicting tangent. New queries use `section(t)`.

Target fingerprints bind source, both geometric bounds, cap, sagitta, design
angle, frame and evaluator version. Operation fingerprints additionally bind
cutter and motion. Tool-only changes preserve target identity but invalidate
operation evidence. `ordered-job-v6-mixed-v-stock` invalidates older bundles;
regenerate them and recompute reports. Native source/post observations retain
their original byte/geometry scope, but old target-bound manifests must be
recomputed. Job and prior-trace frames must match a bound target, even for jobs
whose stock is not evaluated.

`contact_radius(target, tool, d)` maximizes `t*q + rho(d-t)` continuously for
`0 <= t <= d`, where `q` is the **design** tangent. Linear pointed/flat profiles
need only endpoint checks. A rounded profile also checks the sphere/cone join
and spherical stationary height `h=R*(1-q/sqrt(1+q*q))`, clipped to its spherical
domain and penetration. This envelope is monotone in penetration. Planner
inversion limits each segment's depth by its whole-line boundary clearance;
verification uses maximum endpoint depth for the whole segment, conservatively
covering every intermediate XYZ position and the entry/retract columns. A
sharper cutter can leave wall stock, a broader cutter can leave deeper detail,
and a rounded cutter can conflict at an interior height. Requested cap is never
reduced by flute or profile reach. Finite paths remain `partial`; no positive
admissible path is `infeasible`. No unlike-angle finish completeness is claimed.

Required removal and allowed removal coincide for this ideal design; no extra
relief is granted. Safe/outer sections are geometric uncertainty bounds, not
different permissions. `target.volume_bounds(slabs)` measures desired removal
independently of tools. `section_evidence` returns required inner/outer geometry,
known-free sweep bounds, located residual inner/outer geometry and possible
overcut, bound to design fingerprint/depth. Only verified cutting motion or a
replayed prefix establishes known-free material. `section_report` retains the
existing residual-lower/upper/possible-overcut area tuple. GEOS/chord/numeric
enclosures remain conditional, not interval-certified Booleans.

`target.roughing_centers(depth, radius, margin_mm=...)` derives an inward-safe
cylinder-center boundary from the design section at that depth. It is a query
result, never a new finish target or pre-cleared region. The existing cylinder
prefix contract still binds its replay target to the original source opening
and checks `radius + path_error + depth*q` continuously. A full-opening floor
pocket that fits the opening but erases the V wall is rejected. The
[NR01 derived-boundary binding](#editable-native-rest-boundaries-and-mops-nr01)
supports planar cylindrical Pocket cleanup; variable-Z/native V-wall boundary
binding remains outside that scope. Cumulative all-V stock follows the
[MV01 contract](#cumulative-region-v-virgin-stock-contract-mv01). These
contracts cover the cutting profile, not holder occupancy, engagement limits,
controller runtime or physical setup, which retain their separate gates.

The reusable API is challenged by a holed polygonal ornament with broad floor
and a narrow wall band, and a curved annulus through the same evaluator, planner,
section evidence and decoded UCCNC/Grbl workflow. Orchestration and dimensions
live in `tests/test_fixed_v_design.py`, with no core motif discriminator.

### Bounded native helical Circle Pocket and generated cleanup

The accepted CamBam Plus 1.0 Default post for one diameter-24 mm Circle Pocket
contains ten Z-descending G2 entries and level G2 circles at Z=-1 and -2.
`native_series` preserves every posted endpoint, relative-I/J center, direction,
feed, Z and safe return. Core replay admits level and descending XY arcs only
when their radius closes within 0.001 mm, the cutter stays inside the protected
target and depth/cutting length resolve. The clipped variable-depth cylindrical
sweeps keep partial helix cuts out of deeper sections. Integrated volume for a
helical cylinder still rejects until a bounded depth integrator exists.

`native_series_audit.circle_target` binds the source Circle's analytic world
center/radius to a deterministic 256-vertex inscribed polygon. The polygon is a
conservative protected boundary; its radial deficit is at most
`r * (1 - cos(pi/256)) + r*1e-9`, and its area deficit is reported against the
analytic circle. This approximation does not claim exact curved topology.
`native_ordered_job.from_native_circle_cleanup` first replays the complete T1
post, then generates one T2 2 mm cutter contour from a proven full-depth T1
anchor. The T2 entry must descend through prior cleared stock, its radial link
and four G2 quadrants are replayed, and it returns to the original safe tip.
`NativeBinding` recomputes the exact T2 stage and checks unchanged source/post
bytes, Circle geometry, floor and native prefix before UCCNC output or audit.
The offline bundle decodes both stage files and checks the same source-bound
stock. Tool installation, external stage travel, controller runtime, physical
holding, helix body occupancy, arbitrary Circle sizes and general native
Pocket parity are outside this fixture.

### Bounded Manual-tab cutout with interior V operation

`integrations.cambam.tabbed_cutout` binds one accepted 60 x 30 mm straight
Pline Outside Profile, its 80 x 50 x 3 mm Part stock, 6 mm wide/1 mm high
Manual Square tabs and the complete actual CamBam Default post. The B and C
fixtures have four and
five source tab records. The native series parser checks the source/post
identity and every G0/G1/G3 motion; `cam_core.replay` checks the 3 mm
cylindrical tool's access and all cut sweeps. CamBam's tab traversal is a
vertical G0 lift to Z=-2, a 9 mm XY G1 feed across previously cleared stock,
then a G1 descent to Z=-3. The feed traversal is retained as posted motion;
no missing bottom-depth cut is inferred from the XML alone.

The generated operation is one 60-degree pointed V path from (25,25) to
(55,25) at Z=-0.5, bounded by a 30.6 x 0.6 mm straight groove
`v_region.VTarget`. `v_region.verify` checks the complete cutter envelope.
A split UCCNC T3 file
includes a high XY approach, entry, cut, retract and high return. The
independent decoder checks all five moves against the resolved job, and the
decoded path is reverified. The original T1 Default post remains the final
cutout file; this adapter records an offline stage order and operator tool
installation/positioning assumption, not an executable combined program.
The Default post does not encode its pre-start XY; this fixture declares the
initial T1/T3 program-frame tip at (0,0,+5) mm.

At Z=-0.25 and Z=-2.5, both stages' swept removal is replayed on the same
80 x 50 mm stock section in declared order. The shallow V operation removes
no stock at the deeper section. The native Profile's enlarged buffer cover
bounds possible protected-part overlap by 0.2 mm2, including numerical
enclosure at a nominally tangent boundary. Its swept removal must remain
disjoint from the interior V sweep. At Z=-2.5, source tab positions
must match exactly four or five posted 9 mm raised traversals. An 8 mm radial
cross-section through each gap remains stock, and the central part and outer
stock must lie in one connected component. The ideal 3 mm tool and 9 mm
centerline gap leave 6 mm at each bridge centerline. These are conditional
GEOS section and connectivity checks for the bounded flat fixture, not a
physical holding-strength estimate.

The versioned handoff binds source/post/program hashes, decoded native and V
motion, stage order and per-section stock prefixes. Changed bytes or a changed
order invalidate its certificate; a reverse order needs a fresh generated
file and replay. This case-specific verifier covers V before Profile, which
the generic `ordered_job` stock evaluator still reports as unsupported.
Arbitrary contours, tab shapes, fixtures, stage-transition travel and
controller or machine behavior require separate evidence. See the
[runbook](DEVELOPMENT.md#manual-tab-cutout-and-interior-v-evidence) and
[accepted result](REVIEW.md#manual-tab-cutout-and-interior-v-evidence---2026-09-27).

### Bounded layered 3D stock and waterline evidence

`cam_core.volume3d` represents rectangular initial stock from Z=0 downward,
an immutable union of disjoint rectangular removal prisms, and a declared
protected rectangle. A `VolumeOperation` names a flat cylindrical cutter,
cutting length and waterline/rest role. `ordered_job.Stage` can carry that
operation; two or more such stages share one target and pass independently
decoded UCCNC/Grbl motion to `replay_stages`. The job and prefix fingerprints
bind the target, tool and stages. No native 3D Surface MOP or continuous mesh
is inferred from these values.

The supported 3D moves are safe Z-positive rapids, vertical feed entries and
retracts, horizontal constant-Z cuts and vertical `cleared_descent`. Cutter
occupancy is a circular swept section at every affected depth. The evaluator
splits stock at each target and cut depth, unions all prior decoded cuts, and
reports conservative per-slab residual area and whole-volume intervals. A
cleared descent requires its complete outer cutter footprint to lie within
the predecessor stages' inner cleared sweep at every relevant depth. Each
outer cut sweep must stay inside the original target, protecting the thin rib.
Cut depth cannot exceed tool cutting length. This evaluator alone has no
modeled holder shape; its declared cutting length leaves the holder start
above Z=0 at every cut. The optional body/box gate below adds a bounded
clearance check to decoded ordered motion.

Shapely's 24-segment-per-quadrant buffer is an inscribed cutter bound. An
enlarged buffer by `1/cos(pi/96)` supplies the outer footprint. The
inscribed sagitta is about 0.0005354 times cutter radius; the outer radius
enlargement is about 0.0005357 times cutter radius. An independent capsule formula
checks that area lies between the two buffers. GEOS Boolean operations are
floating-point calculations, so these are conservative geometric models
subject to a 1e-7 mm² topology/area tolerance, not formal interval proofs.
`compare_representations` measures exact prism sections against 0.25/0.125 mm
XY columns whose wholly covered/intersected cells bound volume. It reports
wall time and Python peak allocations, excluding native GEOS memory. The
two-prism section backend is chosen for this bounded target because its exact
volume and low storage avoid the columns' boundary error and cost; this does
not select a general freeform surface or solid backend. See the
[runbook](DEVELOPMENT.md#layered-3d-stock-and-waterline-evidence) and
[evidence](REVIEW.md#layered-3d-stock-and-waterline-evidence---2026-09-26).

### Bounded sloped surface and ball cutter evidence

`cam_core.surface3d` owns one affine height field within rectangular Z=0 stock:
the requested floor is `Z=-(intercept+slope*x)`. Its exact target volume is
rectangle area times center depth; horizontal section area follows the plane
crossing. For a ball of radius `r`, the tip path at center X is
`Z=f(X)+r*(sqrt(1+slope²)-1)+clearance`; tangent contact occurs at
`X-r*slope/sqrt(1+slope²)`. A straight Y pass spanning the stock has an
independent swept-volume oracle when its disk fits inside X and its ball center
stays below Z=0: `Y_length*(-2*r*center_Z+pi*r²/2)`.

`ordered_job.Stage` carries `SurfaceOperation`; stages with this one target
replay independently decoded UCCNC or Grbl XYZ moves. Ball occupancy uses the
lowest spherical envelope of every straight center move, with the vertical
shank filling to the stock top. A plane-offset inequality at both endpoints
of each linear move proves no penetration of the infinite protected plane.
`cleared_descent` requires an earlier stage's same-XY vertical entry with at
least the new radius and depth; this is a bounded prior-sweep witness. Cutting
length places the holder start above Z=0, and the report gives its minimum
height. Motion outside the rectangular stock is clipped from material
accounting. The optional body/box gate below checks a supplied holder and
external rectangular fixtures on the same decoded stage motion.

Remaining target volume is enclosed by XY cells. At a cell center, virtual
ball radii `r ± half_cell_diagonal` bound occupancy throughout the cell;
affine floor depth is bounded by its X edges. A bracket error term encloses
the unexamined maximum along each XYZ segment. These bounds include contact
and boundary uncertainty and are deliberately wider than the exact-plane
target volume. Floating arithmetic and decoded-coordinate precision limit
formal guarantees. The slope case selects exact affine contact/protection
plus conservative cells for stock; it does not select a general mesh, signed
distance or curved surface backend. See the
[runbook](DEVELOPMENT.md#sloped-surface-and-ball-cutter-evidence) and
[evidence](REVIEW.md#sloped-surface-and-ball-cutter-evidence---2026-09-26).

### Bounded spherical-bowl ball finish and rest

`cam_core.surface3d.SphericalBowlTarget` adds one non-overhanging spherical-cap
recess inside rectangular Z=0 stock. The rim circle is strictly inside the
stock rectangle; the rest of the top plane is protected. For rim radius `R`,
depth `h` with `0<h<R`, and sphere radius `S=(R²+h²)/(2h)`, target depth at
radial position `q<=R` is `sqrt(S²-q²)-(S-h)` and zero outside. Exact target
volume is `pi*h*(3R²+h²)/6`; the section at depth `d<h` is the disk with
area `pi*(S²-(S-h+d)²)`. These formulas are independent references for the
conservative XY-column representation and decoded stock replay.

For a ball of radius `r<R`, the concentric-sphere contact offset gives tip
height `(S-h)-r-sqrt((S-r)²-rho²)` at center radius `rho`. Below-stock
cutting centers must satisfy `rho<=R-r`, keeping the whole ball footprint
inside the recess and leaving the flat rim protected. The admissible center
disk and offset surface are convex, so straight decoded chords between safe
endpoints remain inside that disk and above the protected surface. The
synthetic path adds 0.001 mm vertical clearance to accommodate four-decimal
output; replay permits zero geometric penetration. Entries and cutting
length use the existing ball-stage access rules; the dependent smaller ball
descends at the prior larger ball's center entry. The optional body/box gate
checks cutter, shank, holder and a declared side clamp continuously on decoded
motion.

XY cells enclose target depth using their nearest/farthest radius and bound
ball occupancy using radius erosion/expansion and the segment optimizer's
bracket. A new local gain lower bound compares final lower removal with prior
upper removal in each same cell. This can prove positive rest cutting even
when whole-volume prior/final residual intervals overlap. The selected
representation is exact spherical contact and section references plus
conservative cells for evolving stock; no freeform mesh, overhang, general
surface offset, real holder installation or controller runtime is claimed.
See the [packet 3 runbook](DEVELOPMENT.md#packet-3-spherical-bowl-ball-finish-and-rest)
and [evidence](REVIEW.md#packet-3-spherical-bowl-ball-finish-and-rest--2026-09-27).

### Ornamental straight-wall paired stock and assembly (IN01)

`cam_core.ornamental_inlay.Design` defines a tool-independent Polygon motif,
finite stock footprint and separate receiver/plug removal Regions. Receiver
depth is seating plus bottom glue gap; plug clearing depth is seating plus
surface gap, with backing retained below it. Positive signed side fit erodes
the plug laterally; negative fit expands it and requests interference within a
caller-supplied limit. Offsets use the declared GEOS polygon approximation
(128 segments/quadrant), so this is not an exact curved-surface contract.
The minimum-web gate erodes by half the requested web and requires a connected
core with unchanged hole count. It detects the representative bridge collapse;
it is not a general local-thickness or material-strength certificate.

`targets(side)` produces reusable `replay.Target` Regions for independently
supplied cylindrical plans. Plug exterior clearing and hole pockets are separate
components. Clearing is bounded by the stock footprint plus explicit edge access;
that extension permits cutter travel outside the blank, not through fixtures.
The plug machining frame is mirrored in X. Assembly maps physical XYZ by
`(x,y,-u) -> (-x+dx,y+dy,u-insertion)`, a proper rigid flip rather than a Z-only
reflection. General rotations, tilted assemblies and undercuts are unsupported.

`verify_pair` requires each complete `replay.Trace`, design/side/frame binding
and expected motion fingerprint. It replays tool dimensions, target identity,
entries, cuts and retracts; only level linear cylindrical cuts are supported.
Existing polygonal sweep bounds establish actual retained stock as finite blank
minus removed stock. No required removal or nominal target earns machined-stock
credit. Both tools may differ without changing the target design.

The verifier splits seated stock at every cutter-depth discontinuity. Top-down
cylindrical subtraction makes retained plug sections grow with machining depth,
so every earlier insertion section is contained in the seated section at the
same receiver depth. This proves continuous straight insertion, not just sampled
poses. It also checks the full requested bottom clearance and above-surface
shoulder clearance. `pass` requires zero possible collision and both clearance
checks; positive lower collision volume is `collision`; otherwise the result is
`unresolved`. Zero nominal side fit is permitted but numerical/actual residual
stock can prevent certification. Evidence includes both motion hashes and XY
registration. GEOS floating-point topology and existing sweep enclosures remain
the numerical boundary; there is no nonzero area tolerance that grants fit.

`finish_envelope` checks a declared full-plane removal range after explicit
assembly, cure and renewed-setup declarations. It reports worst missing/excess
visible plug area, section-enclosure topology, receiver floor and minimum **nominal-core** retained
thickness over all section intervals. Excess ledges have no minimum-thickness
claim. Matching enclosure topology does not prove actual material topology in
the uncertainty band. This is a sanding/facing removal model, not an executed finishing path or
proof of cure. `integrations.inlay_output.audit_pair` separately binds both
detached ordered jobs and every output-file hash, audits UCCNC/Grbl bytes and
passes their reconstructed decoded traces to the core assembly verifier.

The [verification record](REVIEW.md#in01-straight-wall-ornamental-assembly---2026-10-02)
owns the asymmetric holed bridge and independent rectangular consumer evidence.
Tapered mating is covered by the [profile-aware extension](#tapered-profile-aware-paired-stock-in01).
Executable finishing is owned by the [composite facing contract](#composite-stock-facing-and-final-inlay-verification-in01).
Native output, holder/fixture access and physical material-fit coupons remain
outside this slice.

### Tapered profile-aware paired stock (IN01)

`cam_core.tapered_inlay.Design` combines the straight-wall allowance value with
an independent design angle and explicit receiver/plug overtravel. Its receiver
opening is the motif; its retained plug tip is the motif offset inward by
`seating*tan(angle/2) + side_fit`. Signed fit is therefore referenced at the
seated tip plane. Plug removal is the exterior and each hole in the mirrored
machining frame. Each removal component is a fixed `v_region.VTarget`, shrinking
with machining depth under the existing inner/outer offset convention. The plug
grows towards its backing. Offsetting inward and outward need not recover the
original motif around corners; equal design/tool angles never establish a fit.

The minimum-web/topology check covers the smallest nominal receiver and plug
sections. Edge access must clear the finite blank boundary even at the deepest
plug section. Overtravel permits additional machining beyond the requested
minimum bottom/surface gaps; it must leave receiver thickness and plug backing.
It is particularly relevant to pointed and rounded tips whose nominal floor
contact alone cannot clear a finite floor area. It is not inferred from a tool.

`PartStock` accepts component `VComposition` values, including independent
pointed/flat/rounded plans and source-bound cylinder stages. `section` returns
the union of actual removed-stock inner/outer bounds. Missing components remain
uncut. Core consumers revalidate each composition against its original target,
frame, revision and expected stock fingerprint; changing a tool changes that
fingerprint. Shared V verification owns complete motion and whole-profile
containment. This extension supplies targets and verification, not a new planner.

`verify_pair` uses the same proper rigid flip as straight-wall assembly. All
supported profiles have radius nondecreasing with height; removal shrinks with
depth and retained plug grows towards its backing. At each seated depth slab,
the largest plug and smallest cavity enclose possible collision, and the
smallest plug and largest cavity enclose definite collision. Endpoint bounds
cover continuously varying surfaces and depth discontinuities. Every earlier
fixed-XY insertion section is contained in the seated plug section, so zero
upper collision plus both gap checks certifies continuous insertion. The
requested bottom gap uses the actual cavity at its deepest requested plane;
the surface gap uses actual plug stock at the deepest requested shoulder plane.
This is conservative: `unresolved` may need finer slabs, different paths/tools
or changed explicit allowances. Positive lower collision is `collision`; no
nonzero area tolerance grants a `pass`. Reports bind both stock fingerprints,
registration, gap results and every collision slab. GEOS and the existing
V/cylindrical sweep enclosures remain the numerical boundary.

`integrations.inlay_output.audit_tapered_pair` takes tuples of `ComponentOutput`
values for each part. Each component has a complete direct ordered job, files,
expected job fingerprint and all expected file hashes. The integration audits
UCCNC/Grbl bytes, reconstructs decoded V/cylinder sweeps and assembles those
stocks. Components have independent complete setups; no connecting low motion
or pre-cleared access is inferred. Nominal target geometry earns no removal.
The [verification evidence](REVIEW.md#in01-tapered-profile-aware-assembly---2026-10-02)
owns the representative consumers and acceptance. The following composite
contract owns final visible motif/thickness and facing; native output and
physical fit remain separate gates.

### Composite stock facing and final inlay verification (IN01)

`cam_core.composite_inlay.Assembly` accepts a straight-wall or tapered design,
both actual part stocks and their expected fingerprints, plus fixed XY
registration. Construction and finishing verification require the paired
insertion/gap audit to pass. `section(depth)` returns separate receiver and plug
retained inner/outer bounds in assembly coordinates, clipped to their finite
blanks and thicknesses. Depth is positive below the receiver surface; negative
depths include the shoulder and backing. The plug uses the existing proper
rigid flip and registration. No target is substituted for machined stock.

`Finish` binds removal depth, plane and lateral motif tolerances, minimum plug
core and receiver floor, edge access, and nonempty assembly/cure/renewed-setup
declarations. Its source and frame fingerprints include those declarations and
the complete assembly identity. They are caller process claims, not cure or
machine telemetry. The requested plane band must stay within the seated plug.
Facing motion uses assembled backing top Z=0, at
`surface_gap + backing` above the receiver top. Thus a requested receiver-plane
removal `d` requires cutting depth `surface_gap + backing + d`.

`generate` supplies level cylindrical raster passes across the bounds of both
registered blanks with explicit stepover, stepdown and positive clearance.
Every row has a vertical entry and high retract/link; cutting length must reach
the full depth and edge access must contain the cutter. The bounding rectangle
plus declared edge access permits outside-blank cutting, not fixture clearance.
Supplied complete traces may also be verified. The finite target permits only
the requested plane tolerance below the face; deeper motion rejects in replay.

`verify` replays actual motion and requires the inner removed-section enclosure
to cover both entire blank footprints at the shallowest acceptable plane.
Maximum actual cut depth bounds the deepest plane. Missing rows or incomplete
depth remain `unresolved`, with an unfaced-area bound; no area tolerance grants
coverage. An incomplete plane has no certified plane interval or final motif
area values (reported as `None`), rather than treating a predicted exposed
section as finished stock. Plug retained-section endpoints conservatively enclose all material
visible across that plane band, including continuous taper and discontinuities.
The fixed design motif is `plug_xy` for straight walls and the complement of
nominal plug-clearing sections at the finish plane for taper. The lower retained
bound must cover its inward lateral-tolerance offset; its outward offset must
cover the upper bound. Required/inner/outer sections must remain single polygons
with the intended hole count. Missing/excess areas are reported independently.
Matching enclosure topology does not prove actual topology in the uncertainty
band, and tolerance erosion must not erase or disconnect the required motif.

Thickness is certified over the **registered nominal plug tip core**: actual
tip inner stock must contain it, and monotonicity guarantees this core through
the retained depth `seating - deepest face`. Tapered edges and excess ledges have
no minimum-thickness claim. The receiver floor bound subtracts the larger of
the allowed component cut depth and actual facing depth from receiver thickness;
it is conservative even when component removal stops shallower. A `pass`
requires plane, motif and both thickness gates; numerical ambiguity remains
`unresolved`. GEOS and the existing cylinder/V enclosures remain the boundary.

`integrations.inlay_output.assemble_outputs` audits complete independent
`ComponentOutput` jobs and reconstructs decoded paired stock before assembly.
Straight parts supply one complete ordered job each; tapered parts may supply
multiple component jobs. `facing_job` adapts the trace to existing ordered output;
`audit_facing` binds the exact facing job and file hashes, then independently
reconstructs and verifies decoded motion for UCCNC or Grbl. Its ordered audit
reports a bounding-prism residual; only the separate finishing result describes
composite stock. Core-stock and decoded-component evidence remain distinguishable
by the caller's assembly construction route. Controller runtime, native CamBam
facing, holder/fixture access, adhesive mechanics and physical fit are unclaimed.
The [acceptance evidence](REVIEW.md#in01-composite-facing-and-final-verification---2026-10-03)
owns the two synthetic consumers and regression results.

### Bounded circular paired V-carve inlay

`cam_core.inlay` composes two independent one-stage ordered jobs from one
circular design contour and one pointed `v_region.VProfile`. The receiver
removes a disk; the plug removes the annulus outside a retained tapered core.
The two jobs have separate machining frames and stock replays. The assembly
frame reverses plug Z and accepts a declared XY registration offset. For
contour radius `R`, cutter half-angle tangent `t`, engagement `H`, cut depth
`D>H`, and nonnegative radial clearance `c`, the receiver wall is
`R-z*t` at receiver depth `z`. The plug wall at machining depth `u` is
`R-H*t-c+u*t`; at assembly `u=H-z`, so the seated radial gap is exactly `c`.
At insertion depth `s<=H`, the gap is `c+(H-s)*t` before registration. A
registration offset reduces minimum gap by its magnitude. The receiver floor
is `D-H` below the plug front; the retained plug backing begins behind the
seated face. The backing face contacts the receiver top at full insertion.

Both operations cut concentric full G3 circles at tip depth `D`, with vertical
entries, retracts and safe travel included in the decoded program. The
receiver's outer centerline is `R-D*t`; the plug's inner centerline is
`R+(D-H)*t-c`. A circle of center radius `r` cuts the radial interval
`[r-(D-z)*t, r+(D-z)*t]` at section depth `z`. The decoded-circle stock
evaluator unions these intervals independently for each part. A ring pitch
strictly below `2*(D-H)*t` guarantees continuous coverage throughout the
insertion envelope `0..H`; the two final wall circles establish its exact
boundaries. Section residual and protected overcut are reported at the
surface, mid-engagement, seated face, midpoint of the bottom/facing allowance
and cut depth. Residual in the final `D-H` allowance is expected; it does not
enter the assembled insertion envelope.

`integrations.ordered_output` renders complete UCCNC or Grbl programs and
decodes every move, tool, feed, spindle and arc center. `inlay.audit_pair`
replays both decoded stocks, compares their assembled wall radii and binds
the pair, source/tool geometry and output hashes. Wrong flip, excessive
registration, impossible clearance, changed tool geometry and stale program
bytes fail. The cross-section/insertion formulas are an independent assembly
oracle. The supported geometry is a circle with a parallel-plane Z flip and
one pointed cutter; process loads, plunge capability, material response,
machine setup and physical fit are unassessed. Native CamBam emission is not
part of this detached route. See the [runbook](DEVELOPMENT.md#packet-4-paired-v-carve-inlay)
and [evidence](REVIEW.md#packet-4-paired-v-carve-inlay--2026-09-27).
Decoded inlay entries and retracts must be vertical between the declared cut
depth and a positive safe height. Their role labels cannot hide diagonal stock
motion. Supplying body/fixture occupancy for inlay is explicitly unsupported;
the independent-part stock evaluator cannot silently skip that requested gate.

### Bounded tool-body and fixture occupancy

`cam_core.occupancy` checks an optional `ordered_job.Job.occupancy_setup`
after independent UCCNC/Grbl decoding and motion comparison. The setup names
the program frame, the same Z=0 initial stock box as each supported stage,
closed fixture boxes and one coaxial tool body per stage tool. Each body has
contiguous tip-relative cutter, shank and holder cylinders. The cutter cylinder
conservatively encloses the operation radius and ends at exactly its declared
cutting length; it may cut
stock, but every band must clear every fixture, while shank and holder must
clear initial stock. This conservative stock check does not credit cavities
removed by earlier stages. Setup, tool and fixture changes invalidate the job
and prefix fingerprints. Jobs without a setup have no tool-body/fixture clearance
result; verifier-version freshness still applies.

The setup stock must match each stage's bounded 3D stock, planar replay target
bounds/depth, or terminal V target bounds/cap depth. A planar cylinder's cutter
band encloses its resolved tool radius and matches its cutting length. A V band's
radius encloses the profile radius at its cutting length and ends at that same
length. Extending the exempt cutter band would hide non-cutting body from stock
checks, so inconsistent lengths reject. Native source/post
binding remains a separate required gate for a native job. The accepted linear
Profile plus generated rounded V case uses one declared program-frame stock
and side clamp across both stages; its unchanged actual Default post and two
decoded UCCNC files pass stock and body checks. The new setup changes job
fingerprints without changing the source/post or stage program bytes.

For every decoded straight stage move, the checker restricts the XY center
segment to the parameter interval where a band overlaps a box in Z. It then
compares the exact segment-to-rectangle distance with the band's radius;
touching within 1e-9 mm rejects. This covers entries, cuts, retracts and rapids
between endpoints. For supported level XY G2/G3 cuts, it uses the same bounded
arc subdivision as stock replay. The minimum chord-to-box distance minus the
arc's endpoint-radius/chord error and the band radius is a conservative
clearance lower bound for the continuous sweep. The arc model reports its
count and largest enclosure error separately; the straight-only v1 report
remains unchanged. Helices, unresolved arcs and modeled or decoded transition
travel still reject. The actual G3 Pocket plus generated V case uses a
declared raised box outside the cutting envelope; moving that box into a G3
holder sweep rejects between clear arc endpoints. The bounded evidence does
not represent tool body above the declared holder top, clamps of other shapes,
fixture uncertainty, machine kinematics or controller runtime. The
[synthetic runbook](DEVELOPMENT.md#bounded-holder-and-fixture-occupancy),
[linear source-bound runbook](DEVELOPMENT.md#source-bound-nativegenerated-occupancy-case),
[G3 source-bound runbook](DEVELOPMENT.md#source-bound-g3-nativegenerated-occupancy-case)
and [G3 review](REVIEW.md#source-bound-g3-nativegenerated-occupancy---2026-09-27)
record the fixtures and results.

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
remaining volume against the original Region. Completion remains partial;
the prior rest boundary never becomes a wall. The
[M1 actual-post record](REVIEW.md#m1-first-actual-cambam-posts-and-contour-only-revision---2026-09-24)
owns the A01 section/volume witnesses, finite-tool limit and native-route
failure. Physical tool error, material forces, fixtures and controller
behavior remain unassessed.

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

The error-margin buffers compensate their own polygonal round joins: with
64 segments per quadrant the offset magnitude is
`sagitta_mm / cos(pi/128)`. The conservative angle includes GEOS rounding of
non-quadrant fillet subdivision counts. This closes chord loss at shell and
hole corners; it does not supply a formal floating-point topology certificate.
`rest_volume` rejects changing-depth cylindrical sweeps: constant-depth midpoint
slabs cannot integrate a helix. Section queries remain available with depth
clipping; a bounded helical volume integrator is a separate extension.

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

The rounded-profile inverse also supports a cutting length below the ball/cone
join. Its spherical branch uses `r*r / (b + sqrt(b*b-r*r))` to avoid cancellation
at tiny positive clearances. A clearance below the tip radius is infeasible,
including the immediately adjacent representable value below a flat tip.

For a tip penetration `d` and section `t`, cutter occupancy measured from the
original boundary is `t*design_tan + rho(d-t)`. The analytic maximum over all
heights replaces the original same-angle bound `rho(d)`; see DT01 below.
Each straight XYZ segment, including changing Z between vertices,
must remain inside the inward-safe source and at least this maximum evaluated
at its maximum endpoint depth from the boundary with a clearance margin. This continuous segment test covers
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

The outer disk/capsule sweep radius is divided by `cos(pi/128)` at 32 segments
per quadrant; merely adding the `1e-6` mm arithmetic margin does not enclose
circle chords. Target sections use 64 segments per quadrant: the inner target
erodes by `t*tan / cos(pi/128)`, while the outer target retains nominal erosion.
The larger inner erosion compensates inscribed round joins at holes and concave
corners, including non-quadrant fillet rounding. Prior cylindrical helices use
`Sweep.section_segment(t)` before buffering, so unreached motion cannot count as
removed material. These directions propagate to the residual and volume bounds.

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

The audit accepts only the bounded straight-line Default-post dialect,
standalone Drill G98/G80 wrappers, one optional redundant final M5, and
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

### Bounded generated RC01 UCCNC output

`integrations.rc01_controller` adapts exactly `rc01.generate(rc01.Job())` to
the public `ordered_job.Job` stages: T1 roughing, then T2 cleanup. The source
job and generated motion fingerprints bind both stages, their cylindrical
operations, tool dimensions, feeds, RPM, datum and drawing frame. The T2 stage
uses a split-file operator transition at the declared setup tip
(-10,-10,+5) mm. Its completion token is an explicit offline assertion of
installed T2 and registered tip; neither the token nor the NC header observes
the controller or operator. The UCCNC profile uses G54, G49 and an external
initial-tip assumption. No fixture or tool-change travel is emitted.

The ordered UCCNC writer now accepts either its original four-decimal
coordinate policy or a six-decimal policy selected by the caller. RC01
requires six decimals: four-decimal rounding of the generated island-adjacent
endpoint X14.341687 to X14.3417 crosses protected material. Feed/RPM rendering is
unchanged. The handoff records the chosen numerical policy and hashes both
complete files; audit checks the canonical rendering and independently decodes
all commands, tool/spindle states, stage boundaries, feeds and coordinates.
After the shared decoded cylindrical stock replay, the RC01 adapter feeds the
decoded coordinates and feeds to `rc01.verify`. That second check retains
continuous-height tool component, stock-dependent access, process and
rough/final three-slab residual obligations. Roles and RC01 process events are
reconstructed only after the strict ordered comparison of the decoded stream.
The separate `rc01-evidence.json` binds source motion, ordered job, handoff
bytes and RC01's decoded certificate. Altered files, commands, source, precision
or transition token reject. The accepted result is offline controller-dialect
output for this synthetic job; runtime state, physical tools/fixtures and
machining suitability have no acceptance. See the
[runbook](DEVELOPMENT.md#packet-5-generated-rc01-uccnc-output) and
[evidence](REVIEW.md#packet-5-generated-rc01-uccnc-output--2026-09-27).

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
header, interprets standalone G98/G80 Drill wrappers without inventing motion,
and requires every emitted move/event to match the generated sequence exactly
in decimal coordinates, feed, tool and order. Extra motion fails. The audit
reconstructs a `Program` from actual emitted endpoints and calls the
continuous RC01 `verify()` for protected stock, tool-component access, process
and rough/final rest. Its accepted synthetic setup assumes the initial tip is
at (-10,-10,+5); the Default post does not encode incoming machine position.
G80 cancels modal motion; subsequent coordinates require explicit interpolation.
Wrappers after M30 reject, as do mixed motion/M blocks, tool selection without
M6, changed spindle speed without M3 and hidden control separators. Native
LF/CRLF formatting remains supported. G98 is a return-mode selection only;
no canned-cycle execution is admitted by the wrapper option.

The [literal-motion acceptance record](REVIEW.md#rc01-literal-motion-cambam-output-acceptance---2026-09-23)
owns the user-posted item comparisons and rough/final residual witnesses.
That bounded explicit-script route does not establish XYZ/Engrave parity,
native Pocket output, arbitrary controller dialects, physical machining
or formal GEOS topology interval proof.

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
two ordered sections but its previously observed unsafe T2 entries do not
acquire a replay certificate from this parser.

For a source with no MOP intent, the source evidence key normalizes original
primitive UUID/world geometry, Part stock and millimetre units. Document-title
and other presentation-only edits that leave that snapshot unchanged retain
the candidate/post certificate; geometry or stock edits invalidate it. A
source containing any MOP still requires exact source bytes because its MOP
intent lacks a semantic edit classifier. Candidate and actual post bytes are
always exact-hash guarded. Freshness re-normalizes those files and compares the
complete stored observation, so changed stage/tool/motion values cannot retain
acceptance merely by copying input hashes. The bounded planar audit also requires each native
MOP to target the same original Rect or straight Region geometry and floor;
a caller-supplied larger target cannot manufacture safe stock clearance.

`NativeSeries.to_trace` lowers bounded G0/G1 motion and level or descending XY G2/G3
cylindrical cuts on supported XY Pocket/Profile/Engrave stages into the shared
ordered replay model. Arc centers, direction and posted radius mismatch are
retained; the continuous-sweep enclosure is specified above. Straight ramps,
rising arcs and low XY rapids still have no certified lowering. The helical
Circle Pocket has its separate source-bound section audit above; the generic
planar native-series audit below remains a straight-target, level-volume gate.
`native_series_audit.audit_linear_native_series` (its historical API name)
rechecks the
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

### Strategy guarantees and composition limits

The session 3 review distinguishes generated motion, its verification and policy
over verified candidates. These are the current contracts; the broader proposal
in [the CAM design](REST_MACHINING_PLAN.md#wide-areas-rest-access-and-smoothing)
does not make missing search or smoothing capabilities available.

| Owner | Guarantee and scope | Limit relevant to reuse |
| --- | --- | --- |
| `vcarve.generate_slot`, `tapered_vcarve.generate` | Primary 90-degree pointed-cone machining without a predecessor; explicit target, finite tool and complete above-stock links, with independent residual references | Finite passes leave corner/floor residual. Capped-slot generation is specifically width 4/cap 1 mm; full-depth slots and the straight increasing-depth groove support their declared variable dimensions. Output setup constants belong to the reference adapter. |
| `convex_rest.generate` | Replays one supplied cone column and extends it along a verified rising-clearance straight line in a convex target | One prior column and fixed cone family; this is not an arbitrary Region rest planner. |
| `polygon_rest.generate`, `curved_region.generate` | Source-bound supplied cylindrical prefixes, smaller-tool original-boundary contours, interior rows and replayed cleared descent/cutting connectors; curved input uses inward-safe geometry | One connected feasible center Region, full-depth predecessor and supported tool pair required. Candidate rows/access are bounded heuristics; residual reports, not tool reachability, establish coverage. No arbitrary tabs or released-body model. |
| `v_region.plan`, `verify`, `with_prior`, `VSequence`, `VComposition`, `depth_passes` | Raster/offset planning on inward-safe geometry, full-profile containment, high links, partial residual and supplied all-V or mixed cylinder/V sweep unions; bounded retraced axial passes preserve the design | Planning is not a completeness or global path-search proof. Short flutes retain deeper residual. Mixed ordered output requires entry/pass limits and whole-tool setup; cross-stage cavity credit and air-cut minimization remain separate work. |
| `planar_rest.generate`, `cutting_sweep_clear` | Contact/medial-guided V candidates, union-proved air pruning, located new-removal/overlap/residual and capped-floor cusp checks; see [RP01](#feature-aware-planar-vrest-candidates-rp01) | Sampled guidance and finite budgets remain partial. Above-stock links, explicit depth passes and whole-tool setup are required; retained paths can contain overlap. No generic fitting, low-link or global optimization claim. |
| `ordered_job.audit` | Decoded Region-V stock supports standalone/cumulative all-V stages, one endmill/V pair and bounded cylinder/V interleavings against one fixed design; see the [ordered-job contracts](#reusable-ordered-job-output-and-verification) | Other mixed stock evaluators remain unsupported. MX01 requires explicit axial limits and whole-tool setup; partial plans retain residual bounds and infeasible empty plans cannot emit a cutting job. |
| `replay._covered` | A cleared descent/link requires an enclosing prior sweep at the queried depth | Cylinder coverage is proved against one prior sweep at a time. Union-only access and a smaller cylinder around an approximated helical chord can conservatively reject. This is not a general clearance-path finder. |
| `strategy.select_strategy` | Supplied safe routes rank by budget feasibility, final upper residual area, upper volume, then declared tie order. Manual choice preserves safe partial status; failed gates cannot win | No bundle generation, automatic tool choice, cutting/air/time/tool-change cost, engagement objective or global optimum. Caller audits must describe the same physical target and metric; records remain trusted assertions. |

Residual consistency in selection uses the minimum upper area/volume bound from
**every earlier stage**, with the existing `1e-9` comparison allowance. A wide
intermediate enclosure cannot erase earlier proof of less remaining stock.
Overlapping intervals are not proof of an increase and remain eligible. Selection
does not tighten or manufacture the supplied final report.

Region-V's finish angle is stored independently in `VTarget`; see the
[fixed design contract](#fixed-v-design-and-independent-cutter-contract-dt01).
Legacy tool-defined openings must first be frozen to one design before comparing
unlike-angle tools. Tip style, radius, flute length and cutter angle can then
vary without redefining the finish envelope; reach and coverage may be partial.

Region-V planning halves its shallow seed when necessary and inserts an interior
row for each reachable component missed by the normal raster/offset passes.
This prevents a coarse pitch from silently omitting a short component; it does
not prove complete coverage below its `1e-6` mm path threshold or across every
possible topology. Smaller-endmill interior pitch is at most 1.5 times cutter
radius (and at most the existing 1.5 mm nominal pitch), preventing the former
fixed-pitch gap for sub-millimetre-radius tools. Original boundaries, replay
checks and residual measurements remain authoritative after either change.

`mixed.verify_mixed` is an explicitly synthetic RC01/slot reuse probe with fixed
placement and names. The independent replay API and caller-authored ordered jobs
are the composition seams. Layered-volume and plane/bowl modules verify supplied
stages and stock; their test jobs do not implement a general waterline, freeform
surface or automatic ball-rest path planner. Inlay is a bounded circular paired
stock consumer with an analytic assembly model, not the definition of primary V.
For `SurfaceOperation` and `VolumeOperation`, `strategy="rest"` is descriptive.
The supplied motion's `cleared_descent` role requires predecessor evidence;
`entry` permits target-contained stock-removing descent. A label alone neither
requires a predecessor nor establishes cleared access. Positive stock reduction
in the hand-authored surface jobs is not a general finish-coverage budget.

Conditional rest smoothing, arc fitting and an overlap budget are not implemented
by the current strategy modules. Offsets and conservative buffers construct
feasible/covered sets; they do not certify a fitted path. A future smoother must
preserve the original target, distinguish pure rest from allowed cleared overlap,
and recheck coverage and full swept containment after fitting, including holes,
thin bridges and any modeled tabs. Physical surface finish, engagement and
controller acceleration remain separate from geometric residual evidence.

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
It also recomputes the observation and rejects edited residuals, role findings
or verdicts that retain the correct input hashes.
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

### Foundation assumptions and numerical guarantees

The session 2 audit distinguishes exact predicates, analytic formulas evaluated
in floating point, conditional GEOS enclosures and finite sampling. Lengths are
millimetres, areas mm² and volumes mm³ in the detached CAM modules; positive
depth is into stock from Z=0. `planar` alone admits explicit inch conversion and
rigid frames. Native normalizers obtain world XYZ and reflected bulges from
`Region.get_absolute_coordinates_xyz`; a curved non-similarity transform is
unsupported. A 2D buffer is set dilation/erosion, not an open-curve offset API or
a general 3D swept-volume implementation.

| Owner / calculation | Assumptions and guarantee | Independent evidence / boundary |
| --- | --- | --- |
| `planar.feasible_centers` | Rational rectangle/circle radius comparison preserves area, line, point and empty feasible sets; floating placement is separately qualified. | Exact fit and adjacent representable sizes, mixed units/rigid placement in `test_planar`. |
| `planar.normalize`, `_planar_shapely` overlays | Authored polygon rings and converted rings must both be valid; no silent topology repair. Output preserves representable holes/components. Nominal GEOS operations do not claim certified numerical error bounds. | Source-touch conversion regression, hole/split/contact cases and backend evaluation tools. |
| `stock.bound_horizontal_sweep`, composition and access | A horizontal capsule has area `2*r*length + pi*r²`. Radius/position uncertainty shrinks guaranteed and expands possible removal. Rational membership and cell classification compose union bounds; residual reverses removal enclosure direction. Travel uses earlier guaranteed stock only. | Capsule/lens areas, exact tangencies, overlap/order/refinement and denied uncleared connectors in `test_stock`. No arbitrary orientation or general 3D access claim. |
| `replay.Sweep`, `arc_segments` | Closed cylinder/cone membership; downward helix chords clip at each depth. Arc enclosure includes radius mismatch, twice requested sagitta and reconstruction margin. Every prior cut remains subject to its protected target. One tool/target name has one meaning per trace. | Semicircle tube area, depth-clipped membership, malformed islands and conflicting identity tests. Rising/full-circle forms remain unsupported. |
| `curved_region.approximate` | Bulge sweep `theta=4*atan(b)`; arc area adds `R²*(theta-sin(theta))/2`. Chord sagitta plus directional margin buffers bracket validated analytic rings. Offset topology changes reject. | Annulus area, reflected mixed arcs, narrow access and round-corner margin tests. Analytic source topology validation is a caller precondition. |
| `v_region.VProfile`, `section_report`, `volume_bounds` | Half-angle profiles and tangent sphere/cone join; full-line distance bounds every cutter height. Nested target/removal sections yield slab residual bounds without assuming residual itself is monotone. | Capsule section/integrated volume, holed erosion, inverse/join continuity and helix-depth regressions in `test_v_region`. Floating GEOS topology remains conditional. |
| `vcarve`, `tapered_vcarve` analytic sections | Pointed 90-degree cone; slot equal-depth/equal-span row partitions or straight increasing-X radius envelope with slope strictly between 0 and 1. | Independent row integration and support-function corpus references; other angles/orientations require another owner contract. |
| `volume3d.LayeredTarget` | Axis-aligned positive-depth rectangular prisms have disjoint interiors and do not overlap protected stock; even tiny positive overlap rejects. Column bounds classify rectangle coverage without area cutoffs. | Exact prism sums, thin features, disk/capsule sweep formula and decoded dependent stages in `test_volume3d`. Boundary contact is allowed; no overhang/mesh claim. |
| `surface3d` | Affine plane ball contact follows the surface normal. Spherical-cap volume is `pi*h*(3*a²+h²)/6`; rationalized depth/contact and cap-height sections retain shallow bowls. Cell extrema and enlarged/shrunken cutter radii bracket each column. | Signed plane contact, cap references, shallow bowl, refinement and dependent stock/rim tests. No freeform surface or formal interval arithmetic claim. |
| `inlay` | Circular pointed-V radial intervals; flip maps plug depth `u` to receiver depth `engagement-u`. Minimum assembly gap is clearance minus XY offset, plus `(engagement-insertion)*tan(alpha)`. | Separate decoded part stocks and explicit radii/areas/gaps in `test_paired_inlay`. Reported tolerances are model/output allowances, not physical fit evidence. |
| `occupancy.verify` | Axial tool-band/box overlap clips segment parameters; minimum XY segment/rectangle distance minus band radius and arc error must exceed the declared contact tolerance. | Mid-segment/arc collisions, touching axial partitions and overflow rejection in `test_occupancy`. Only declared boxes and contiguous cylindrical bands are modeled. |

The detailed review and test-adequacy dispositions are in
[session 2 evidence](REVIEW.md#branch-review-session-2---2026-09-28).
For the affine floor `z=-(a+s*x)`, ball contact tip Z is
`-depth(X)+r*(sqrt(1+s*s)-1)` and the contact X coordinate is
`X-r*s/sqrt(1+s*s)`. Their linear endpoint inequalities protect complete
straight moves. For a bowl of rim radius `a`, depth `h` and sphere radius
`S=(a*a+h*h)/(2*h)`, section area uses `pi*u*S*(2-u/S)` with `u=h-depth`;
depth/contact expressions are rationalized to avoid subtracting almost equal
radii. The rounded V tip uses the same cancellation-avoidance principle.

Surface cells use half-diagonal `delta`: cutter radii `max(0,r-delta)` and
`r+delta` at the cell center enclose removal throughout that cell. The concave
ball envelope along a segment is maximized with a shrinking ternary bracket;
the upper error allowance is `sqrt(2*r*L*w)+abs(dz)*w`, where `L` is XY length
and `w` the remaining parameter width. Residual bounds use
`max(0,target_low-cut_high)` and `max(0,target_high-cut_low)` times cell area.
Same-cell final lower minus prior upper removal proves incremental gain.
Integer-nested refinement supports tighter bounds; arbitrary unrelated pitches
need not yield monotone intervals. Runtime layered-stock overcut tolerance
`1e-7` mm² is separate from rectangle admission, which has no area cutoff.

Exact arithmetic in the bounded stock owner does not make the other owners
exact. GEOS topology, floating reconstruction and fixed process/output tolerances
must remain visible to callers; successful reference jobs do not establish
arbitrary topology, scales, freeform surfaces or machine execution.

### Detached nominal planar core

`cambam_builder.planar` is the public, document-independent owner of immutable
`PlanarFrame`, `RegionSet`, `Polygon`, `Rectangle`, `Circle`, `ErrorBudget`,
`PlanarApproximation`, `FeasibleSet` and `PlanarResult` records. It imports no CAD
entities, development probes or backend geometry classes. `_planar_shapely.py`
privately owns lazy Shapely/GEOS admission and regularized area operations.
The `planar` optional extra selects Shapely 2.1.2 on the supported Python >=3.12
interpreters; ordinary imports and analytic centers need no backend.

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
fresh child when tabs are disabled. `Automatic` and bounded `Manual` additionally write plain child
values governed by that container: positive Width and Height in drawing units;
integer MinimumTabs/MaximumTabs with minimum no greater than maximum; nonnegative
perimeter TabDistance; nonnegative SizeThreshold below which tabs are suppressed;
UseLeadIns; and Square/Triangle/Skip TabStyle. Width is tool-compensated in CamBam's
display, Height is relative to target depth, and Skip is a non-contact/plasma mode.
`UseLeadIns=true` requires Automatic Square tabs plus an active lead-in. CamBam
Plus 1.0 native B/C/D files show that Manual positions live in a sibling
`Tabs/HoldingTab` collection. Each record names the target primitive XML ID,
has a perimeter-normalized `ParametricPoint`, an outward XY normal, and
`NormalInverted=false`. In the observed B/C/D sequence, moving one tab changes
only its fraction, the added later-perimeter tab appears last, and removing it
restores the four prior records. On the
observed 60 x 30 mm counterclockwise rectangle, a fraction of `1/6` places a
tab at `(40,10)` and the left-edge fractions `0.872222222222222` and
`0.961111111111111` place tabs at `(10,33)` and `(10,17)`.

Fresh direct-core and MCP Manual authoring accepts drawing XY pairs on one
explicit root, identity-posed, flat straight closed counterclockwise Pline
with an Outside Profile, zero roughing clearance, Square or Triangle style,
and no tab lead-ins. The count must fit MinimumTabs/MaximumTabs; each point
must lie on exactly one edge far enough from its corners for the compensated
tab width, with nonoverlapping gaps. The writer orders records by perimeter
fraction, resolves the actual primitive XML ID, derives its outward normal,
and writes all Manual scalar settings alongside the sibling collection.
Recognized imported records inspect as XY positions without rewriting their
native template. Unrecognized imported collections retain their opaque fields;
known `Tabs/HoldingTab/ParentEntityID` references are remapped from the imported
native ID through primitive UUID identity to the current export ID, including
copy/transfer UUID remapping. Unknown or reassigned parent references reject
export with a replacement-points diagnostic; preservation must not silently
detach tabs or attach them to another target. Inspection likewise does not
report the old points as valid after target reassignment.
Switching away from an imported Manual method remains unsupported. The native
posts establish Square lifts and Triangle ramps for this contour. Fresh B/C
Square output now has its own accepted CamBam Default posts: every machine
command matches the corresponding native B/C post after comments are removed.
This proves native output for the observed rectangular slice. Fresh Triangle,
transformed, curved, reversed and multi-target Manual Profiles remain outside
that output-acceptance bound.

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
without exposing their contents or blanking sibling modeled values. Recognized
Manual tab point collections on the bounded straight-Pline slice inspect as XY;
unrecognized collections, unsupported lead modes and literal CustomScript remain
opaque. This
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
