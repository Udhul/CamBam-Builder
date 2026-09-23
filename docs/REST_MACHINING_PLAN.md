# Future rest machining and V-cutter paths

Status: **active design refinement, 2026-09-23; bounded geometry and supplied
section-motion foundations plus nominal RC01 generation implemented; native
output adapters pending**.
Requested 2026-09-08 and expanded 2026-09-22/23. Priority belongs only to [PROGRESS.md](PROGRESS.md#remaining-backlog-in-order).
This document owns the problem, proposed outcomes, technical reasoning and future
acceptance criteria. Only the explicitly marked bounded slices claim implementation;
none authorize machine execution.
The five original outcomes below are retained. The current design proposal and
unresolved product decisions are in [Design refinement](#design-refinement---2026-09-22)
and the latest [execution architecture refinement](#execution-architecture-refinement---2026-09-23).
The [first generated acceptance job](#first-generated-acceptance-job-rc01)
now gives user-accepted synthetic inputs and separate standalone/native acceptance gates.

## Programmatic execution requirement

The framework must calculate supported geometry and paths programmatically and
load/save CamBam files without running CamBam. No supported standalone headless
CamBam engine has been established for this workflow. CamBam does expose
in-application scripting/plugins, including a G-code generation example; that is
different from an independently callable, supported headless service. Do not design around GUI automation,
CamBam plugins/scripts, or manual toolpath extraction as execution dependencies.
Implement the needed geometry, cutter and supported MOP path algorithms in the
framework, using an appropriate programmatic geometry library where justified.
For existing native-MOP authoring, CamBam remains the toolpath generator and
postprocessor: opening the `.cb` and generating/exporting G-code is required.
For future framework-generated motion, CamBam is an optional inspection/editing
environment and one possible output route. A direct postprocessor is a separate
capability; calculating paths alone does not yet make the workflow headless to G-code.

The useful proxy here is a **file representation**: calculate XYZ tool-center
paths ourselves, save them as shapes and attach an Engrave MOP. It is not a way
to invoke CamBam's calculations. Manually exported CamBam paths may serve as
optional comparison fixtures, but normal operation and automated tests must not
require them. CamBam's scripting examples document possible file/motion behavior;
they are not a runtime interface for this framework.

## Execution architecture refinement - 2026-09-23

This section records the user's new context and the lead's recommended design.
Recommendations and unanswered choices are not recorded as user agreement.
It refines the detached-core proposal below, not a replacement implementation.

### User context and current capability

- Preserve programmatic CAD/Part/MOP authoring and `.cb` interchange. The product
  goal is native object coverage; current support is bounded, not full CamBam parity.
- Today's workflow opens that file in CamBam, generates paths with Ctrl+T and
  exports G-code with Ctrl+W (the user's workflow also regenerates paths there).
  CamBam's selected postprocessor and configuration determine the machine output;
  `.nc` is an example extension, not the interface contract.
- Users may inspect and edit layout, geometry, enabled Parts/MOPs and output grouping
  before machining. Preserve this workflow alongside future headless execution.
- Explore an independent toolpath/stock core, reusable beyond rest and V-carving,
  with eventual direct G-code output and alternative generators/optimizers.
  Exact replication of CamBam path shapes/order is not a requirement expressed
  here. Future shape-following 3D routing is an extension interest, not current scope.

The detached geometry/cutter/stock design, generated-motion evidence, native versus
explicit-path adapters and extensible strategies were already proposed. The new
emphasis is two complete output routes, a clear execution authority, edit/reimport
semantics, and an explicit delivery decision for direct postprocessing. Existing
`planar.py` provides nominal geometry; `stock.py` verifies supplied horizontal
section motions; `machining_planning.py` balances depths/process parameters. None
currently generates a general XYZ path, optimizes routes or posts G-code. Existing
MOP `optimisation_mode` is a serialized CamBam setting, not our own optimizer.

Primary sources checked 2026-09-23: CamBam's [automation documentation](https://www.cambam.info/doc/plus/Automation.htm)
documents scripting and .NET plugins, and its [MOP Automate example](https://www.cambam.info/ref/script.mop-automate)
describes opening a source file, inserting operations and producing G-code.
Its [script instructions](https://www.cambam.info/ref/script) run inside CamBam.
Therefore "CamBam has no API/automation" is too strong. These sources do not
establish a supported standalone headless engine or its deployment/licensing
contract; no such integration was tested here. Reopen an optional native bridge
only with a supported interface and concrete consumer need. It must not become
a dependency of the independent core.

### Execution authority and compatibility

The decisive invariant is that removal evidence describes the motions intended
for execution. Our generated pocket path cannot certify clearance left by a
different native pocket path, even when both use the same outline and cutter.
Ideal reachable area is an upper bound on possible removal, not guaranteed
cleared space. Conservatively assuming unknown material remains can support
analysis, but may prevent useful links or invalidate cutting-engagement limits.

| Route | Motion authority and stock evidence | Output and limit |
| --- | --- | --- |
| Native MOP workflow | CamBam generates motions; framework estimates remain estimates unless a supported trajectory source or independently proven lower removal bound exists. | Existing editable `.cb`; CamBam generates and posts. No exact native-rest guarantee from MOP parameters alone. |
| Framework motion through CamBam | Core generates roughing and dependent cleanup; stock comes from those ordered motions. | Proposed XYZ/Engrave adapter must establish actual emitted cutting, entry/link/retract and order semantics. Native regeneration must not silently substitute a different plan. |
| Framework motion through direct post | The same core plan is lowered to a declared controller dialect and independently checked. | Future headless G-code plus optional `.cb` inspection artifact; bounded dialect support, not universal postprocessor compatibility. |

CamBam [documents a GCode machining operation](https://cambamcnc.org/doc/1.0/toolpaths-and-gcode.html)
as well as Engrave. A supported NC-reference/import route may be worth comparing
with XYZ/Engrave for motion inspection; it is not currently implemented or selected.
Neither route should be called lossless before testing its complete motion
semantics. An inspection-only file must say so; a path polyline does not encode
all feed, spindle, tool-change, rapid and program-state semantics.

For native interoperability distinguish three claims: document fidelity, equivalent
machining intent/result within declared tolerances, and identical trajectories.
The first remains a library concern; the second is the useful future acceptance
goal. The third is optional compatibility research, not the default design target.

### Accepted integration requirement - 2026-09-23

The user accepts framework ownership of both roughing and cleanup for the first
reliable combined sequence, conditional on supporting both standalone operation
and native CamBam shape/MOP integration. Native integration is a required product
workflow, not merely a way to preview independent paths. The user also accepts
designing direct headless G-code output now and implementing it after the first
useful rest/V-carve workflow. The third answer establishes caller-owned workflow
orchestration: the framework supplies reusable capabilities for iterative import/edit
and embedded headless applications, without imposing a fixed use-case sequence.
These decisions do not assert that native path equivalence is already established.

Both workflows use the same detached capabilities to calculate rest, tool access, V-carve motions
and candidate combinations. CamBam geometry/MOP inputs are normalized through an
adapter; direct Python inputs need no CamBam document. Keep these result forms
independently callable and composable:

| Requested result | Core result and CamBam attachment contract |
| --- | --- |
| Rest analysis | Pure rest with source-motion evidence, original protected target, uncertainty and residuals; optionally attach closed preview shapes. |
| Native endmill cleanup | Derive tool-specific closed machining regions with permitted overlap into cleared/free space; attach a smaller-tool Pocket, or a Profile only where its finite cutting band covers the intended cleanup. Preserve original geometry/MOPs and explicitly configure the new operation. |
| Explicit endmill cleanup | Generate cutting and access motions from the same target/stock request; return motion values independently or use a supported path-output adapter. |
| V-carve cleanup | Generate variable-depth tool-center paths using the original finish target, tool profile, residual stock and allowed overlap; attach XYZ Pline shapes and correctly configured Engrave MOPs through a validated adapter. |
| Combined strategy | Compare permitted tool/operation sequences against the same target and constraints; return the selected sequence, evidence, residual and tradeoffs. Native and explicit outputs can be combined only where their stock dependencies are supported. |

Do not pass an expanded rest polygon to a general V-carver as a new finish design.
Its artificial boundary against cleared space would alter the intended wall/depth
solution. Keep pure rest, allowed machining domain, protected target and tool-center
paths separate. For endmill attachment, distinguish a cutter-center region from
the boundary consumed by a native Pocket/Profile; passing centers as boundaries
would apply tool compensation again. Overlap is tool- and access-dependent, not
a uniform arbitrary outward offset or a substitute for stepover.

Support curved output as an extension of the motion contract. Preserving endpoint
Z and bulge in XML does not prove the intervening spatial motion. First use bounded
XYZ line segments; add bulged paths only after core interpolation, native Engrave
behavior and relevant post output agree within declared deviation/sweep limits.
Do not advertise a CamBam bulge as an arbitrary spatial-curve representation.

Attach evidence to each source operation rather than to an entire document mode.
When CamBam generates a preceding native path, analysis based on our estimated
counterpart remains estimated. Native region/MOP cleanup authoring is still useful,
but cannot silently promote that estimate to guaranteed clearance for later links
or stock-dependent cuts. Unsupported mixed sequences must report the specific
missing evidence; no silent switch of execution authority is allowed.

Acceptance must eventually cover both input routes and each advertised attachment:
equivalent normalized requests produce equivalent core results; native cleanup
regions retain holes/protected walls and admit the chosen cutter; actual native
Pocket/Profile output achieves the claimed cleanup within tolerance; XYZ/Engrave
output preserves the required motion; and combinations retain ordered stock
dependencies. This is a product requirement and future verification plan, not
current native or production acceptance.

### Proposed shared core and adapters

Keep one distribution. Reusable detached calculations, stock/volume analysis,
motion values and verification belong in `cambam_builder.cam_core`; optional
rest/V-carve strategies and policy planning belong in `cam_extensions`. The
first exact reference job currently lives in `cam_core.rc01`; the
[package organization plan](structure_spec.md#package-organization-decision-and-migration-plan)
records its later split and the native/extended migration order. Root-level
stock, planar and calculation modules remain active owners until moved with
their callers. Native CamBam integration now lives in `integrations/cambam/`;
future controller adapters remain separate and depend inward on immutable core
values. No separate distribution/service or generic plugin registry is needed.
Names below describe responsibilities, not new public APIs.

| Boundary | Responsibility |
| --- | --- |
| Input normalization | Convert direct Python requests or a CamBam snapshot into resolved geometry, operation intent, tools, setup, units and constraints. Resolve inherited styles/defaults, enabled order, nesting/transforms and stock offsets; reject missing required values or unsupported semantics. |
| Target and stock | Keep desired final geometry/protected material distinct from evolving stock. Evaluate cutter occupancy/removal, residual and uncertainty against ordered motions; analysis is callable without a generator. |
| Strategies | Pocket/profile/rest/V-carve algorithms propose cutting passes using shared geometry and stock queries. A V-carve target is independent of which clearing tools precede it. Strategies do not serialize `.cb` or controller commands. |
| Motion representation | An explicit ordered plan of cutting, entry, linking, retract and setup/tool events with resolved tools, feed/spindle requirements, physical tip datum, frame/units and provenance. Begin with fixed-axis XYZ lines; add arcs with explicit plane/center/sweep/interpolation contracts when needed. |
| Scheduling and optimization | Respect stock and operation dependencies, tool/process constraints and caller-fixed order. Generate a deterministic baseline first; later improve travel, entry, segmentation or cutting strategy under explicit objectives. Controller acceleration/look-ahead remains a separate machine concern. |
| Verification | Evaluate final continuous motions, including cutting versus non-cutting tool/holder occupancy and between-height behavior. Replay stock after motion changes. Report unsupported/unassessed process or machine constraints separately from geometric success. |
| Output adapters | Map a verified plan into a bounded CamBam path representation or controller-specific program. Keep ordinary native-MOP document export independently available. Validate any changed interpolation or inserted motions after lowering. |

Toolpath semantics must not depend on CamBam entity IDs, XML, a Shapely object, or
a controller's textual G-code. Thin adapters retain mappings back to document
objects. Use direct pure Python APIs first; MCP remains a caller of the same core.
Preserve the existing SectionMotion contract as a bounded verifier, not the public
model for all future motion. Extend or adapt it only where the first XYZ consumer
proves reuse, avoiding a speculative rewrite of working geometry foundations.

For the first rest/V-carve scope, fixed-axis cutters and bounded depth sections
with interval coverage are a reasonable representation. Sampling a few Z planes
does not prove clearance between them. Introduce mesh/dexel/voxel stock only if
a concrete 3D target cannot be represented efficiently and conservatively by the
section contract. Future surface-following methods can share motions, tool values
and provenance while requiring a different target/stock evaluator.

Optimization is not a prerequisite for stock understanding. A sweep can be
evaluated for supplied or generated trajectories independently of how they were
chosen. Reordering cuts may preserve final removal yet change intermediate access
and engagement; simplification/arc fitting can change removal itself. Each proposed
optimization must retain constraints and revalidate affected stock prefixes and
output motion. Prefer repeatable useful plans before pursuing better rankings.

Direct postprocessing adds controller-specific units, coordinates, feed modes,
arc support, tool/length offsets, spindle/coolant, program start/end and file/tool
transitions. Unsupported commands/macros/cycles must fail explicitly. A CamBam
postprocessor name or custom header/footer is not sufficient to reproduce its
semantics; do not execute or silently transplant them. Begin with one declared
controller profile and parse/backplot its supported output for comparison against
the final motion plan, including rounding and any post-added moves. G-code emission
does not imply machine execution or production acceptance.

### Caller-owned workflows and reusable capabilities

**User clarification, 2026-09-23:** use-case workflows are situational and belong
to consuming applications. The framework may participate in repeated conversational
iterations, import manually edited documents, or act as a programmatic design-to-
output adapter inside another application. Neither explicit manual reimport nor
automatic synchronization is a mandatory framework workflow. The earlier binary
question is superseded by this separation of capability and orchestration.

Expose independently useful operations for document import/inspection, resolved
job construction, analysis, path generation, verification, result attachment and
export. A caller can compose them, provide supported existing motion/stock inputs,
or request only regions or diagnostics. Convenience composition may implement a
common sequence without making that sequence the only entry point. Calls must not
require a GUI, interactive approval, MCP session, network service or an implicit
global "current job". These are target API requirements, not claims about new APIs
already implemented. Existing document and MCP contracts remain their own owners.

The consuming application chooses when to ingest changed data, which work to
recompute, whether to ask a user, which strategy/tool sequence to request and where
to write outputs. The core owns geometric/motion validity: a caller's workflow
policy cannot turn stale removal evidence into valid clearance. Return structured
results distinguishing current, stale, uncertain, unsupported, invalid and partial
states where applicable, with the affected dependencies and reasons; do not resolve
missing facts by an interactive prompt or silently change strategy. Keep freshness,
evidence level and machining completeness distinct rather than one success flag.

Native and standalone paths share these functions. File watching, application
events or a future native bridge can call them to automate synchronization; they
are integration responsibilities, not new core infrastructure required now.

### Manual edits, provenance and regeneration

Treat generated plans and stock as derived artifacts tied to immutable input
snapshots. Retain fingerprints of geometry, resolved operations, tools, stock/setup,
ordered predecessors, algorithm/tolerances and output profile. Accept a fresh
document or supported in-memory request independently; when a prior snapshot is
available, identify relevant semantic changes and affected derived results. Imported
data must be usable without a prior framework session or hidden sidecar state.
Persistence/transport of provenance is a separate adapter concern.

"Understand user edits" means correctly interpreting the resulting supported
geometry, relationships, transforms, enabled/order state and effective MOP/tool
values, and diagnosing changed assumptions when comparison evidence exists. It
does not mean inferring why the user edited them. Native numeric ID renumbering or
XML formatting alone must not masquerade as a changed cut; ambiguous entity matches
must cause conservative invalidation rather than invented identity. Missing external
styles/tool settings and unsupported imported features must remain explicit gaps.
Import preservation does not imply those features are supported for planning.

Geometry/layout/tool/depth/enable/order changes invalidate affected plans and stock.
The caller chooses whether and when to regenerate, inspect or export the edited
document; verification must reject stale claims for the affected results. Preserve
authored originals separately from derived path geometry. Edited generated paths
can be supplied as explicit motion only through a supported interpretation and
renewed verification; do not reconstruct pocket or V-carve design intent from them
automatically. If derivation links are lost, treat paths as independent input with
unknown provenance rather than overwrite them as if they were still owned output.

Disabling a roughing MOP invalidates cleanup that assumes its removal. Splitting
output files is valid only with explicit stock/setup and execution dependencies;
a later file is not independently runnable merely because it contains one Part.
Changing a postprocessor can invalidate output evidence without changing the
abstract cutting plan. A complete setup translation can reuse evidence only under
a proven frame transformation; moving geometry relative to fixtures cannot.
Do not silently run both native source MOPs and their generated replacements.

The core can only assess state supplied to it; it cannot observe unsaved GUI edits.
The caller may supply updated state manually or automatically. Neither approach
changes the dependency/freshness rules, and ingestion need not regenerate paths.
Round-trip retention of provenance in native files needs its own compatibility
evidence; do not assume arbitrary metadata survives CamBam save.

Future acceptance must cover fresh edited-file import without session history,
equivalent direct/document inputs, relevant versus cosmetic changes, disabled or
reordered predecessors, ambiguous/lost identity, stale-result rejection and explicit
recomputation. Demonstrate both an iterative document consumer and a noninteractive
embedded caller using the same core. Do not require live synchronization to prove
these contracts. These checks are separate from native motion and machine acceptance.

### Delivery recommendation and open decisions

Keep rest/V-carving as the active capability driving the shared core. Do not expand
the immediate task to all native MOPs, a generic optimizer, general 3D stock, or a
CamBam postprocessor interpreter. The earlier fixed-section work supplies useful
regression evidence, but another isolated capsule refinement does not by itself
deliver the user's workflow.

Recommended progression (priority and actual next work live in PROGRESS):

1. **Definition and synthetic-case selection complete, 2026-09-23:**
   [RC01](#first-generated-acceptance-job-rc01) applies the accepted boundaries to
   one roughing-plus-cleanup consumer, with stock height, tools/reach, target,
   entry/retract, process and tolerance requirements, plus a cone feasibility guard.
   The user selected the proposed synthetic case; no physical setup is required now.
2. Deliver the first generated sequence with shared motion values, all-height
   occupancy/access, stock replay and independently checked residual/overcut.
   First prove an endmill sequence; then pointed-cone variable-depth and capped-depth
   V-carving using the same contracts. Broader topology should serve this outcome.
3. Prove the CamBam explicit-path adapter against actual emitted motions before
   relying on it for execution. Design the direct-post boundary now; the user
   accepted delivery after the first useful rest/V-carve workflow. If native output
   cannot preserve required semantics, report that blocker and propose revisiting
   the order rather than weaken the motion contract.
4. Add one direct controller post with semantic output checks, then expand strategies,
   optimization and 3D target representations only against concrete jobs and measured
   limitations. Matching native motion byte-for-byte is not a completion criterion.

Workflow decision status, asked 2026-09-23:

- **Answered:** the first reliable combined workflow may own roughing and cleanup,
  while both standalone and native shape/MOP workflows are required. See the
  [accepted integration requirement](#accepted-integration-requirement---2026-09-23).
  Native-generated predecessors still need a separate motion-evidence contract.
- **Answered:** design for headless G-code now; implement after the first useful
  rest/V-carve workflow. Keep the motion model independent of output format.
  Select the first controller dialect, tool-change/setup behavior and acceptance
  environment before that later postprocessor increment.
- **Answered by reframing:** workflows belong to callers. Supply complete import,
  interpretation, change/freshness diagnostics and explicit regeneration capabilities
  for supported inputs; applications decide whether ingestion/recomputation is manual,
  automatic or iterative. See [caller-owned workflows](#caller-owned-workflows-and-reusable-capabilities).
- **Answered, 2026-09-23:** the user selected the proposed synthetic
  [RC01](#first-generated-acceptance-job-rc01) case, including its test-only
  plunge/feed limits. Its geometry, tools, process bounds and residual/error
  thresholds are the first acceptance baseline. Physical material, workholding,
  machine limits and production parameters remain unestablished; accepting the
  test inputs does not accept generated motion, native output or production use.

This refinement closes when context, boundaries, recommendations and unanswered
choices are recorded and reviewed. Documentation-only checks suffice; no pytest,
native-file validation or machining trial is needed for this round. Implementation,
output compatibility and production acceptance remain separate future gates.

## First generated acceptance job RC01

**Defined and selected by the user 2026-09-23; accepted synthetic test inputs,
standalone generation implemented, native output acceptance pending.** The original
definition did not generate executable paths or request a CamBam/machine trial;
the standalone implementation is now described in the
[implemented contract](structure_spec.md#rc01-generated-motion-and-full-height-replay).
Framework ownership
of roughing and cleanup, both input routes, native integration and later direct
posting are already accepted in the [integration requirement](#accepted-integration-requirement---2026-09-23).

RC01 turns the existing rectangular target/island foundations into a useful
two-tool pocket sequence with descent, stock-dependent access and output checks.
It deliberately uses a rectangular opening and zero roughing allowance to retain
an independent analytic residual oracle. It replaces the earlier letter-like,
0.5 mm allowance example **as the first generated job only**; that broader case
remains in the acceptance corpus. Native shape/MOP integration is a required gate,
not satisfied by an inspection-only path drawing. Direct G-code posting follows
the first useful rest/V-carve delivery, as previously agreed.

### RC01 inputs and process bounds

All dimensions are millimetres in one right-handed drawing frame. Z is the
physical cutter-tip height, positive upward; stock top is Z=0. No nesting,
transforms, external styles, tool-library lookup or inferred defaults are needed.

| Input | Accepted synthetic value / interpretation |
| --- | --- |
| Initial stock | Box X=[-5,45], Y=[-5,35], Z=[-10,0]; 50 x 40 x 10. |
| Finish target | Remove X=[0,40], Y=[0,30], Z=[-3,0], except the protected island X=[16,24], Y=[11,19]. Vertical walls, flat floor Z=-3, no through-cut, tabs or released bodies. |
| Protected material | Entire stock outside that removal volume, including the island through the full stock height and the 7 mm floor thickness. Allow boundary contact; forbid penetration. |
| Fixtures | Synthetic support box X=[-5,45], Y=[-5,35], Z=[-15,-10]. No other obstacles in the modeled tool/holder workspace. This is an explicit test assumption, not an inferred real setup. |
| T1 rougher | Flat, center-cutting endmill, diameter 6; cutting cylinder at tip-relative heights [0,6], radius 3; non-cutting shank (6,20), radius 3; holder [20,40], radius 10. |
| T2 cleanup | Flat, center-cutting endmill, diameter 2; cutting cylinder [0,6], radius 1; non-cutting shank (6,20), radius 3; holder [20,40], radius 10. |
| Tool reach | 20 tip-to-holder, 6 cutting length, for both tools; fixed +Z tool axis. All tool components participate in collision checks. |
| Roughing | T1 at tip Z=-1,-2,-3 in that order; zero radial/floor allowance. Maximum newly engaged axial depth 1; maximum lateral step between adjacent clearing passes 2.4. Full-width initial slotting and center-cutting plunge are permitted synthetic operations. |
| Cleanup | T2 after all T1 cuts, also at Z=-1,-2,-3; maximum newly engaged axial depth 1, lateral stepover 0.8. Full-width cutting is permitted; no constant-load/adaptive claim. |
| Process tokens | Both tools: spindle clockwise 12000 rpm; cutting feed 300 mm/min, cutting plunge 60 mm/min, cleared vertical feed 120 mm/min, retract feed 300 mm/min, coolant off. These values exercise propagation and limits only; material cutting suitability is unassessed. |
| Travel/setup | Start, tool change and end at tip (-10,-10,+5). Clearance plane +5; approach plane +1. Above-stock XY positioning is rapid, vertical motions are feed moves. Allowed tip travel X=[-15,55], Y=[-15,45], Z=[-3,10]. Tool changes are explicit stopped-spindle events at the setup position. |
| Geometry/error | Main oracle uses exact synthetic stock/tool dimensions and zero physical position/radius error. Maximum geometric approximation 0.001; comparisons use a separately reported numerical enclosure at most 0.000001 mm. Neither is permission to overcut. Positive-error rejection is a separate required case below. |

The 11 mm minimum island-to-outer-wall gap admits T1 and T2; its topology is
deliberately uncomplicated. At the lowest tip height, each shank starts at Z=+3
and each holder at Z=+17. These arithmetic facts simplify this job, but the
verifier must derive clearance from supplied component dimensions rather than
assume holders always clear the stock.

### RC01 generated motion and all-height obligations

Use a deterministic baseline of fixed-axis XYZ straight segments and explicit
tool/setup events. Strategy details and exact pass coordinates belong to the
generator; no optimal route or replication of native Pocket paths is required.
Repeat the same T1 XY coverage at each depth so new axial engagement stays bounded.
Provide T1-cleared vertical access columns centered at (5,5), (35,5), (35,25)
and (5,25), from Z=0 through -3. Each must contain at least the radius-1 T2
occupancy, proved from actual T1 cuts; a feasible-center calculation is insufficient.

T1's first entry is at (5,5): position at +5, feed to +1, then use a cutting
plunge through stock top to -1. Subsequent levels may descend in the previously
cleared column and cut only the next 1 mm. T2 approaches each corner through its
proven column, feeding to the requested level before cutting remaining material.
Confine cleanup to the four corner windows specified in gate N below, retaining
the original pocket/island as the protected-target authority. Verify the full-target
residual afterward, including any T1 approximation remainder outside those windows;
do not discard it from the budget or silently repocket the entire target with T2.
Between disconnected cutting runs, retract vertically to +5 before XY travel;
this first job does not require low-level rapid links or ramps/helices. A tool
change requires retract, travel to the setup position, spindle stop, tool event
and restart before the next approach. Each output must preserve that dependency.

The shared motion contract must resolve units/frame, tip datum, component geometry,
ordered endpoints, move role (cutting entry, cut, cleared travel, retract, rapid),
feed/spindle state, operation/tool identity and input/evidence fingerprints.
Events cannot be hidden in arbitrary output strings. Each move starts at the
previous endpoint; tool changes do not imply teleportation. The verifier evaluates
the complete continuous sweep over all occupied heights, not endpoints or a few
sampled Z planes:

- Cutting occupancy intersected with initial stock may remove only the original
  target; the whole tool must avoid fixtures. Check axial/process limits
  checked against the stock prefix immediately before that move. Intended air cuts
  may overlap already cleared space; they do not redefine the finish boundary.
- Non-cutting tool components must avoid remaining stock and fixtures throughout.
  Cleared travel/rapid moves remove nothing and need guaranteed free occupancy.
  Retracts must remain in the just-cleared column or other proven free volume.
- Recompute stock from actual ordered cutting sweeps, including plunges. Retain
  rough-only and combined snapshots; a later cut cannot erase an earlier overcut
  or invalid access result. Exact prismatic intervals suffice for this job if
  continuous coverage is proven; mesh stock is not a prerequisite.
- Polygonal approximation around the island must be conservative for the entire
  cutter, including between vertices. Chords of a nominal offset arc are not
  automatically safe. Geometry approximation and physical uncertainty stay distinct.

### RC01 independent result oracle and rejection cases

The removal section has area `40*30 - 8*8 = 1136 mm^2`, volume `3408 mm^3`.
For ideal complete coverage by a radius-r cylinder, only the four outer concave
corners are inaccessible: total section rest is `(4-pi)*r^2`. The convex protected
island creates no additional ideal inaccessible material at these wide clearances.
This reference is an attainable-area bound, not evidence that generated paths
have covered it.

| Result | Analytic ideal | Required generated result |
| --- | --- | --- |
| T1 rest | 7.7256661177 mm^2 per section; 23.1769983531 mm^3 over 3 mm depth | Rest bounded between ideal and ideal +0.5 mm^2 at every Z in (-3,0), and volume between ideal and ideal +1.5 mm^3. |
| T1 then T2 rest | 0.8584073464 mm^2 per section; 2.5752220392 mm^3 | Same +0.5 mm^2 / +1.5 mm^3 upper budgets. Report partial target completion: finite endmills retain the sharp-corner material. |
| Cleanup benefit | Ideal rest reduction 6.8672587713 mm^2; 20.6017763138 mm^3 | Guaranteed reduction at least 6.3672587713 mm^2 per section and 19.1017763138 mm^3 overall. |
| Protected material / access | Zero overcut, zero fixture or non-cutting component collision, zero unproved travel | Prove containment/disjointness continuously; unresolved numerical boundary cases are indeterminate, not passes. |

Also bound residual location: outside the analytic ideal corner rest, no remaining
point may be more than 0.05 mm from either that rest or the original protected
boundary. Together with the area/volume limits this rejects missed interior strips
and excessive wall stock; section area alone cannot hide a floor error. Include
the floor boundary in occupancy/containment checks and prove coverage throughout
the depth intervals. Independent rectangle/circle references and swept-volume
checks must not simply call the generator's own feasibility result as their oracle.
Do not reuse the existing coarse section-area grid as a pass certificate if its
enclosure cannot resolve these budgets; improve only the accuracy RC01 needs.

Required negative variants of this same job:

- Remove/reorder T1, or feed T2 down an uncut corner instead of its cleared column:
  reject the claimed cleared access and identify the missing predecessor/volume.
- Insert an XY rapid at Z=-1 across the island; insert a cut below -3 or into an
  outer wall: reject continuous occupancy, even if segment endpoints look valid.
- Shorten cutting length to 2 while keeping the wider T2 shank, or lower its holder
  start to tip+2: reject any resulting stock collision; do not extrapolate the
  fixed-section certificate to certify reach.
- Declare radius uncertainty +/-0.005 and isotropic position error 0.005 on an
  exact-wall-tangent nominal plan: its nominal clearance certificate is invalid.
  Reject or return unverified until regenerated and checked with inflated occupancy
  and reduced guaranteed removal. This round does not promise positive-error
  completion within the nominal residual budgets.
- Change geometry, tool, depth or enabled/order state after generation: reject stale
  evidence. Cosmetic XML/ID changes alone must not change normalized geometry or
  removal; fresh import without prior session state must still work.

### RC01 standalone and CamBam output gates

These are distinct recorded outcomes. Passing an earlier gate does not imply later
acceptance, and caller applications can invoke each capability independently.

| Gate | Required evidence and boundary |
| --- | --- |
| S: standalone generated sequence | Implemented for the exact nominal RC01 request with deterministic motion, ordered rough/final prefixes, rational per-slab area intervals and conservative residual-location diagnostics. Automated negative cases pass. The GEOS location test has <0.000001 mm polygon sagitta but no formal floating-topology interval proof, so strict numerical certification remains conditional. See the [implementation contract](structure_spec.md#rc01-generated-motion-and-full-height-replay) and [evidence](REVIEW.md#rc01-standalone-generated-sequence---2026-09-23). |
| I: native input and document attachment | A synthetic `.cb` with outer/island Region, Part stock and explicit T1/T2 Pocket intent, plus explicitly supplied tool-component/setup values absent from native fields, normalizes to the same request/results as S. Preserve authored geometry/source MOPs and identity references; attach derived results separately. Reopen/export and import an edited file without hidden session state; supported edits recompute explicitly and invalidate old evidence. Unresolved inherited or unsupported values return diagnostics. |
| E: explicit framework motion through CamBam | Attach both generated operations using a candidate XYZ-Pline/Engrave adapter or an explicitly bounded, role-bearing CamBam carrier. In the established CamBam Plus 1.0 environment, inspect actual regenerated and posted motion: coordinates, interpolation, ordering, feeds, spindle/tool events and every inserted entry/link/retract. Reverify the actual motion against RC01, including its rough-only stock prefix. A path drawing or successful XML round trip alone fails this gate. Record the accepted carrier; one carrier's result does not imply parity of another. |
| N: native smaller-tool region/Pocket cleanup | Preserve the original target and attach T2 Pocket MOPs to four closed 7 x 7 corner windows: [0,7]x[0,7], [33,40]x[0,7], [33,40]x[23,30], [0,7]x[23,30]. These are machining boundaries, not cutter-center regions. Each contains its entire T1 corner rest plus overlap into cleared material; none touches the island. Verify actual CamBam-generated cleanup against the same target, process and residual criteria, after the E-verified T1 prefix. Do not infer native removal from nominal Pocket settings. |
| P: production / later direct posting | Separate pending gates. S is headless planning, not G-code delivery. A later controller adapter must reverify its emitted motion. Physical machining requires an identified machine/material/setup, real tools/workholding and separately agreed tolerances/process limits. No execution is requested by RC01. |

**2026-09-23 I preparation:** the [native adapter contract](structure_spec.md#rc01-native-input-and-comparison-candidates)
now constructs and strict-reimports the source `.cb`, normalizes its Region,
stock and disabled source MOPs with explicit supplemental setup, and rejects
unsupported/inherited edits. The A/B/C `.cb` files and independent standalone
comparison manifest are reproducible. This was automated input/attachment
evidence; CamBam output had not yet been checked at that point. The candidate
Engraves carry only level-cut
centerlines, so the E probe must inspect CamBam-added entries, links and events
before any output pass. The [runbook](DEVELOPMENT.md#rc01-native-input-and-abc-comparison-preparation)
owns the exact generation and comparison commands.

**2026-09-23 first output trial:** the user posted A/B/C with CamBam Plus 1.0;
the [review](REVIEW.md#rc01-first-cambam-output-trial---2026-09-23) records
floor overcut, path reordering and invalid T2 tool-change sequencing. E and N
failed. The repaired A post confirms the additive-depth fix, but still fails
ordered/role motion; B/C repaired variants have not been posted. The existing
N gate still requires a verified T1 prefix; a future
fully native Pocket roughing variant may supply one only after its actual
motion independently passes the same stock, access and residual checks.

**2026-09-23 native Pocket trial:** a full-target T1 Pocket and four T2
corner-window Pockets were posted separately as rough-only and combined
Default programs. Their T1 move prefixes match, both stop at Z=-3, posted
rough/final area and residual-location budgets pass numerically, and the
required four plus eight actual T2 vertical columns have exact T1 cut
witnesses. N nevertheless fails the accepted execution gate: low rapid moves,
ramped entry, F60 low-level XY, displaced T2 change without a spindle stop,
and unresolved protected-island tangencies remain. See the
[dated trial](REVIEW.md#rc01-native-pocket-posted-motion-trial---2026-09-23).
Coverage alone does not satisfy the full motion contract. The next output
increment should establish a carrier/post strategy that preserves the required
setup, approach, retract and tool-event roles before repeating RC01 posting;
general area precision and optimizer work remain deferred.

For N, configure tool 2/diameter 2, stock surface 0, target depth -3, depth increment
1, stepover 0.4 of diameter, roughing clearance 0, clearance plane +5 and the supplied
feeds/spindle explicitly; resolve remaining path-affecting fields during adapter
implementation. The artificial window edges lie in already cleared space; they
must never become new protected design walls. Inspect native entry/depth order:
if CamBam inserts a motion or entry that violates the job, N fails even if its final
rest looks correct. This job proves Pocket attachment; Profile substitution and
general minimal rest-boundary construction remain separate follow-ups.

No fixture combines enabled native source roughing with its generated replacement.
Preserve source MOPs disabled in execution variants and report the enabled sequence.
For E, require at most 0.001 mm continuous centerline deviation from each intended
cut, preserve cut direction and stock-dependent order, and separately check all
adapter/post-added moves. This deviation budget never overrides zero protected
overcut or access requirements. Unknown emitted commands/modal state leave the
output unverified; parsing only XYZ endpoints is insufficient.
E and N may legitimately produce different paths, but both must meet the same
removal/access requirements with evidence from the motions actually executed.
If Engrave cannot carry the required roles/events, record an adapter blocker and
revisit output representation; do not silently reinterpret every line as a cut.
CamBam-posted comparison output is evidence for E/N, not implementation of our
future standalone postprocessor.

Before requesting manual E/N acceptance, the implementation increment must create
and inspect artifacts in a unique ignored `output/rc01-<unique>/` directory:
`A-rough.cb` (verified roughing prefix), `B-explicit.cb` (roughing plus explicit
cleanup), `C-native-cleanup.cb` (same roughing plus native Pocket cleanup), and the
independent motion/residual comparison. Keep source geometry and disabled source
MOPs inspectable in each. Record the chosen CamBam postprocessor/settings and the
comparison reader's supported command subset before E/N; the controller dialect
for later direct posting is a separate choice. Provide clickable files, exact enabled-MOP/property and
motion expectations, bounded steps to regenerate/save comparison output, and a
report format with I/E/N pass/fail plus deviations. Inspect returned emitted motion
locally before recording acceptance. Manual file inspection adds native application
evidence later; it adds no evidence to this documentation-only definition today.

### Next RC01 output milestone after native Pocket trial

Treat the shared E/N role and event blocker as one integrated output increment,
not a queue of individual rapid, ramp, feed and tool-change edits. Select the
most credible bounded carrier/post route using the actual failed posts, implement
it for the complete T1 roughing plus T2 cleanup job, and inspect its emitted
program. A candidate must be able to express the required setup position,
ordered entry, approach, retract, feed, spindle stop/start and tool change
before another CamBam export is requested. Native Pocket settings alone already
passed the numerical coverage check and cannot certify those emitted roles.

**Increment acceptance:** strict-reimport any new `.cb` candidate, preserve the
original target and disabled source MOPs, inspect the full posted program and
independently replay continuous motion, protected stock, T1-to-T2 access,
process limits and per-depth rest. Pass one complete output route under its own
E or N gate; retain the other gate as an explicit overall requirement and keep
physical acceptance separate. A coverage-only pass or another Default-post
sequence with the same role/event failures does not complete this increment.

**Stop or change route:** if emitted evidence after local repairs or a documented
CamBam interface limit shows the selected carrier cannot preserve those roles,
record the exact blocker and present the direct-posting timing as a concrete
user decision. Its currently accepted delivery order remains after the first
useful rest/V-carve workflow until the user changes it. Do not create a series
of smaller RC01 follow-ups around already diagnosed symptoms. After a successful
output slice, compare the remaining E/N gate with the first variable-depth
V-carve and broader rest consumer before choosing another increment. General
topology, numeric precision, optimizer and package-layout work reopen only
when this output gate or a named consumer supplies a failing case.

**2026-09-23 route result:** one native Pocket/Default MOP repair pair was
strict-reimported with no spiral lead, no path optimisation, cut-feed stepover
and zero crossover. Those fields address three observed path-generation
symptoms, but this carrier still cannot independently encode RC01's first
entry, feed approach, feed retract and setup-position stopped tool change.
The Default post's rapid formatter covers all rapids; changing it globally
would also change permitted above-stock XY rapid travel. MOP header/footer
act at operation boundaries, not every entry and retract. The existing
posted pair proves coverage but fails these roles, and the repaired pair was
not posted because it cannot meet the candidate precondition above. At this
route-trial point, E and N remained separate failed/pending output gates.
A raw-motion carrier through
CamBam's documented NCFile operation would require producing the complete
G-code first, so it moves bounded direct posting forward. That timing would
need a user decision if no other CamBam carrier passed; the later literal-script
pass leaves direct posting after the first useful rest/V-carve workflow. The
current user workflow remains agent-prepared `.cb` files followed
by user-generated G-code in CamBam; moving direct posting forward changes that
boundary and is not inferred from this role trial. Neither choice certifies a
controller or machine setup.

**User clarification and selected follow-through, 2026-09-23:** retain the
agent-prepared `.cb` / user-posted CamBam `.nc` workflow. The agent prepared
one complete literal-motion `Drill/CustomScript` MOP in a `.cb`, using the
already verified T1/T2 framework program. This is an alternate E carrier,
not a repair of native Pocket N or proof of XYZ/Engrave parity. It can encode
the required roles within one CamBam operation; its wrapper and literal
motion were checked in the actual Default post. The prepared file,
manifest and replay command are in the
[runbook](DEVELOPMENT.md#rc01-literal-motion-cambam-carrier). The first post
preserved literal `|` separators
on one invalid NC line; the agent repaired the `.cb` to contain actual XML
newlines. The user-posted revision passed exact comparison of all 2,945
emitted items and continuous whole-height stock/access/process/residual
replay, including the T1-only prefix and T2 cleanup. This meets the
**one-complete-route output milestone** under the alternate literal-script
E carrier for the accepted synthetic RC01 request. It does not accept the
earlier XYZ/Engrave implementation, the native Pocket N route, or physical
machining. Direct standalone posting retains its later timing. The next
increment is the user-prioritized native optimizer/output mapping foundation
below; reopen N when its native motion source and role controls have enough
evidence to satisfy the independent gate.

### RC01 cone guard, acceptance scope and stopping condition

Pair the cylindrical job with an analytic pointed 90-degree included-angle cone
guard, not a claim that a cone can finish its vertical sharp corners. In the
interior of a 4 mm straight slot defining a V-shaped target, the candidate tip
depth is 2 mm and its radius at the surface is 2 mm. A supplied cone with maximum
cutting radius 3 and conical axial length 3 admits that local section; a radius-1.5
cap does not. With an intentional 1 mm target depth cap, surface radius is 1 mm:
one centerline pass cannot clear the 4 mm opening. Finite ends, whole-body sweep
and generated variable-depth/capped-depth completion belong to the next cone
slice. Original V-target geometry, residual and tool limits must remain separate.

**User acceptance, 2026-09-23:** the user answered "Use the proposed synthetic case"
when offered RC01's 40 x 30 x 3 mm pocket, 8 x 8 mm island, 6 mm rougher, 2 mm cleanup
tool and proposed test-only plunge/feed limits versus tailoring a physical setup.
Use the documented RC01 inputs and acceptance criteria as the implementation
baseline, including expected finite-tool corner residual and synthetic plunge/
full-slotting bounds. No first-job refinement is pending. This accepts the test
case, not implementation correctness, CamBam output or physical machining.
Real-machine acceptance remains separate and will require material, machine
travel/controller, actual cutter/reach/holder, fixtures, allowed entries and process
limits before numerical production acceptance can be defined.

Definition and user refinement are complete with inputs, oracles, access obligations,
limits and output gates recorded. The exact nominal T1/T2 sequence and continuous
all-height replay are implemented. The I/E/N adapter gates follow; cone path delivery
follows through the same contracts. Do not call full target removal or native output
accepted while the corresponding gate remains partial, blocked or untested.
Reopen general topology, positive-error completion, low links, optimizers or finer
area infrastructure only when an RC01 gate or a named next consumer requires it.

## Native CamBam optimizer and output mapping foundation

**Priority:** next distinct increment now that the RC01 explicit-script
CamBam output carrier has a recorded pass. The user
requested a one-time reusable evidence base covering CamBam's native shape
and machining-operation classes under both its 0.9.7 Legacy and 0.9.8 New
optimisation modes. The agent prepares complete `.cb` files; the user's only
required action is generating the corresponding native G-code in CamBam.
Keep the number of posts small by grouping independent cases, but separate
cases where one MOP or tool event could alter another's output.

First inventory all classes available in the CamBam Plus 1.0 baseline and
label each as authorable here, import/preserve-only, native-only, or unsupported.
Include Rect, Circle, Arc, open/closed Pline, Points, Text and Region, and
every discovered MOP class rather than assuming the current library's four
authorable kinds are the whole CamBam set. Use controlled geometry, stock,
tool and units so the expected path can be judged; include a small set of
interaction fixtures for islands, multi-target ordering, multiple depths,
lead/rapid/crossover moves, tool changes, arcs and styles. Pin the optimiser,
postprocessor/profile, CamBam build and every relevant MOP property in each
candidate. Preserve source `.cb` hashes and exact returned `.nc` hashes.

For each supported case, record the native target-to-path relationship, path
order/direction, depth sequence, interpolation/rounding, entry/link/retract,
feed/spindle/tool events and any post-added moves. Use deterministic parsers
and stock replay where supported; record indeterminate geometry or controller
semantics rather than estimating removal from intent. The durable outcome is
a versioned mapping and reusable fixture/reader tests in their code and
documentation owners, with a coverage table and explicit unresolved modes.
It must let later callers select either native CamBam-derived posted motion
or framework-generated motion as the stock/rest authority, with separate
fingerprints and invalidation. Neither authority is silently substituted
for the other.

**Stopping condition:** every inventoried shape/MOP class has at least one
accepted native-output observation in each applicable optimiser mode, or an
explicit unsupported/blocked result with reopening criteria; the bounded
interaction cases have reproducible input/output hashes and documented
parser limits. This is finite class and behavior coverage, not every numeric
parameter combination or a universal CamBam emulator. Expand combinations
only when a specific consumer exposes a missing dependency. The first output
slice should prove the corpus method end to end before generating the full
set of fixtures. No physical machining is part of this mapping.

## Problem and machining intent

An endmill cannot reach every point of a sharp internal corner without cutting
beyond the intended boundary. Pocketing and inside/outside profiling can leave
remaining material, or **rest areas**, because of tool size, access, allowances
and the actual passes executed. Larger tool radii and tighter concave features
generally increase the inaccessible material. Classify corners from the removal
side: an outside profile is not an instruction to pocket the entire exterior.

Rest machining uses knowledge of prior removal to target that material with a
subsequent operation, often a smaller endmill. The original MOP's corner-overcut
setting uses its existing tool; it deliberately changes the cut near corners and
can remove surrounding material to obtain clearance. It is useful for some joints
but is not equivalent to a new, smaller-tool operation that preserves the original
boundary. CamBam illustrates residual corners using its cut-width display in the
[pocket tutorial](https://www.cambam.info/doc/1.0/tutorials/pocket.html).

Do not assume corner overcut makes the rest set empty. Compute the actual sweep:
it may clear a corner while leaving other material and may create a separate
overcut set. CamBam also documents ramped corner overcuts for a V-Cutter tool
profile. Thus the gap is a general, constrained rest/V-carving planner, not an
absolute absence of native V-cutter behavior. Version-specific behavior needs a
fixture when implementing. See [CamBam corner-overcut documentation](https://www.cambam.info/doc/plus/WhatsNew.htm).

The desired workflow is: inspect original MOPs and removal intent, calculate
remaining material, preview it, choose a cleanup strategy/tool, generate bounded
geometry and a new MOP, then verify the resulting tool sweep and residual stock.
Original shapes, operations and tool settings remain independently editable.

## Material model and evidence levels

The following is a proposed engineering model, not an implemented API. Start with
planar constant-depth endmill operations, then extend to depth slices and V-cutter
volume checks. Always carry units, setup/work coordinates, stock surface and depth
convention. Use positive downward depth in planner calculations and convert to
explicit machine/drawing Z at the interchange boundary.

At a specified depth plane, define:

- `S`: stock present before the chosen source operations.
- `I`: intended removal within stock, bounded by the design and operation scope.
- `C`: modeled material removed by the ordered source operations at this plane.
- `R = I \\ C`: pure rest area.
- `O = (C intersect S) \\ I`: removal outside the intended scope, classified
  against permitted allowances/relief before calling it a defect.
- `F`: protected material, islands, tabs and other forbidden cutter occupancy.

An already empty area outside stock is free space but is not prior removed stock.
Keep that distinction explicit. For outside profiles require stock bounds and a
finite intended removal band/cut width or an explicit removal region; the
complement of a closed outline is unbounded. Distinguish deliberately retained
roughing allowance and holding tabs from unintended inaccessible stock.

Two calculation modes must not be confused:

| Mode | Input and meaning | Limit |
| --- | --- | --- |
| Reachability estimate | Intended region and cutter shape; feasible-center offsets predict geometrically reachable material | Does not prove that passes visited every feasible point |
| Toolpath-derived removal | Generated cutting trajectories, tool geometry, depths and ordered operations | Predicts the nominal sweep, not measured physical removal or tool deflection |

The normal trajectory source is the framework's own supported operation planner.
Label its results as framework-planned removal; equivalence to CamBam-generated
pocket/profile passes is a separate compatibility claim requiring evidence.
Externally supplied trajectories are optional inputs, not the primary workflow.

For an ideal flat endmill of radius `r`, a cutting center trajectory `T` sweeps
`T + disk(r)` at a plane intersected by its cutting portion. Union segment and arc
sweeps; exclude rapids, air cuts and depths that never reach that plane. This is
a Minkowski sum; polygon/disc offsetting is described in the
[CGAL geometry reference](https://doc.cgal.org/Manual/3.4/doc_html/cgal_manual/Minkowski_sum_2/Chapter_main.html).
For unrestricted ideal pocket filling, the opening `(I eroded by disk(r))`
`dilated by disk(r)` estimates reachable removal. It is an upper bound on coverage
under those assumptions, not a substitute for actual stepover, entry/access,
tabs, allowances, depth passes or overcut behavior.

Resolve effective MOP/tool values, including inherited defaults, before analysis.
Include finish passes, stock-to-leave, profile side/cut width, region fill,
stepover, entry/exit/ramp moves, tabs and enabled execution order. A string naming
a tool or the outline plus diameter alone is insufficient to reconstruct CamBam's
full toolpath. Other CAM systems likewise distinguish geometric rest estimates
from stock-aware calculations; see [Autodesk's explanation of 2D rest limitations](https://help.autodesk.com/view/fusion360/ENU/?caas=caas%2Fsfdcarticles%2Fsfdcarticles%2FEmpty-toolpath-warning-on-2D-Adaptive-operations-using-rest-machining-and-conventional-milling-in-Fusion-360-MFG.html).

## Five desired outcomes

### 1. Pure rest areas

Return closed regions with holes, disconnected components, per-depth results and
source provenance. Preserve pure `R` independently of any later access expansion.
Report area (and volume only where depth modeling supports it), boundaries,
unreachable components, intentional allowance, overcut, numerical uncertainty and
whether the result is estimated or toolpath-derived. Produce preview layers for
intended, cut, rest and excess-removal geometry; do not create a cleanup MOP merely
because an area exists.

Acceptance: analytic pockets and bounded inside/outside profile fixtures agree
with independent sweeps; include circles/arcs, sharp corners, islands, narrow
slots, multiple prior tools, partial depths and overcut enabled/disabled. An empty
rest result must follow geometry evidence, not an overcut-property shortcut.

### 2. Endmill rest regions expanded into safe cleared space

A small isolated rest polygon may be too narrow for a pocket MOP to admit the
cleanup tool. Expand its machining region into already cleared space to give the
new tool access, while retaining the original protected boundary. The expansion
depends on the planned cleanup diameter and the actual clearance at that depth.
An arbitrary outward offset of the rest polygon can gouge adjacent material.

Let `A` be the allowed cutter footprint domain: rest intended for removal plus
verified cleared/free space, excluding protected material. Feasible tool centers
are `F_r = A eroded by disk(r)`. Candidate centers that can touch the rest lie in
`Q = F_r intersect (R dilated by disk(r))`. Their possible coverage is
`R intersect (Q dilated by disk(r))`. Connected access, entry and actual generated
passes must still be checked; this set calculation alone is not a route.

An initial expanded region can be `E = R union ((R dilated by disk(e)) intersect C)`
with `e` derived from cleanup radius/diameter and required overlap. Treat this as
a candidate, not a universal radius formula: offsetting again inside a pocket
MOP can exclude necessary centers. Grow only within the permitted domain, verify
the resulting MOP sweep and stop when coverage meets the goal or no safe gain is
possible. Prefer center-path construction where region expansion becomes lossy.
Keep rest, expanded machining boundary and feasible-center geometry distinct.

Acceptance: expansion outside pure rest is confined to approved cleared/free
space; simulated cutter occupancy avoids `F`; newly removed material remains in
the permitted removal region. Compare achieved residual against the smaller-tool
reachability limit and report still-inaccessible sharp corners. Never promise
that a finite-radius endmill cuts a mathematical sharp internal corner exactly.

### 3. Rest regions prepared for V-cutters

Use the same pure rest result, then select the intended finish: surface corner
sharpening, a bounded V-shaped recess, chamfering, or volumetric cleanup down to a
specified floor. These are different target geometries. A pointed tip may reach
a sharp boundary point by rising toward it, but does not guarantee removal of an
entire vertical corner column while retaining square walls and a flat floor.
The widening cone may intersect adjacent walls; extra depth must be explicitly
allowed, bounded and reported rather than assumed harmless.

For an ideal conical tool with included angle `theta` and flat-tip radius `a`,
the radius at height `h` above its tip is `rho(h) = a + h * tan(theta/2)` within
the cutting length and maximum diameter. At a reference surface and penetration
`d`, this gives `rho(d)`. A local available radius `q >= a` suggests
`d = (q-a)/tan(theta/2)`, subject to all depth/clearance limits. A radius below
`a` is unreachable without changing the allowed geometry. This formula is a
geometric derivation for an ideal cone, not a complete path/stock algorithm.

Pointed tools often need no artificial expansion of the rest boundary to admit
the tip. Nevertheless, the rest boundary against empty space is not a protected
wall: the planner may safely place the tool over that space to improve coverage.
For flat tips, account for the finite tip footprint and the entire conical body,
using the same clearance logic at every affected height, not just one fixed XY
offset. Model tip radius/rounding, usable flute length and shank separately when
the ideal cone is insufficient. Return reachable and remaining material plus
any requested relief below the original floor.

Acceptance: pointed/flat-tip cases, acute corners, islands, different floor
levels and maximum-depth restrictions yield bounded cutter sweeps. No claimed
complete cleanup from a 2D surface silhouette alone.

### 4. General bounded V-cutter path calculator

Accept a valid closed shape or region with holes independently of any source
MOP; rest machining is one caller. Produce ordered XYZ cutting paths and explicit
link/retract requirements, with a report of achieved geometry and residual stock.
Use medial-axis/distance-field candidates or adaptive offsets, with geometric
checks of the swept tool. A straight skeleton is a candidate generator, not in
general the Euclidean clearance field of a round tool. Use additional passes for
wide, depth-capped areas; the centerline alone may not clear them.

Parameters include tool included angle, tip flat/rounding, maximum diameter and
cutting length; reference surface, start/max depth and permitted extra depth;
protected regions/stock; stepover or cusp tolerance; maximum pass depth, slope,
plunge limits, feeds and safe link height; chord/Z tolerances and minimum feature
thresholds. Report features removed by simplification rather than silently
discarding them. A general closed shape means supported, validated topology;
reject or explicitly repair self-intersections and ambiguous touching rings.

Proxy output: encode computed XYZ polylines, then attach an Engrave MOP with
explicit `TargetDepth=0` to follow the shape. CamBam's
[sloped-line example](https://www.cambam.info/ref/script.sloped-lines) also specifies
`OptimisationMode=None` to retain path ordering. Verify stock-surface conventions,
pass generation, start points, direction, linking and generated G-code in the
target version; a zero target-depth setting alone does not establish equivalence.
Keep this export/attachment helper unavailable until target-version varying-Z
toolpath motion has been verified. Lossless XYZ/bulge Pline and Region interchange
is supported, but does not establish Engrave interpolation or generated motion.
Native V-carving systems combine boundary width, tool angle, variable depth and
optional flat-depth clearing, supporting this separation of tasks; see
[Vectric's V-Carve documentation](https://docs.vectric.com/docs/V12.0/VCarveDesktop/ENU/Help/page/single-page/index.html).

Acceptance: canonical shapes and holes pass independent XYZ sweep checks; path
depth/pass limits hold between vertices as well as at them. Two XML round trips
preserve XYZ, closure, identities, source order and MOP parameters. CamBam-generated
motion must match intended cutting paths and acceptable links before machining
acceptance is claimed. Use straight XYZ segments first; the native interchange
evidence for varying-Z bulges does not yet establish their exact intermediate Z
parameterization or generated toolpath semantics.

### 5. V-cutter edge tracing and corner cleanup

From a pocket/profile boundary, generate a tool-center route offset and lowered
according to the intended contact surface. Simply placing the tool tip on the
boundary at depth can gouge it. Add local corner excursions, retracing or multiple
passes where a single edge pass does not cover the rest. Verify coverage against
the residual model after each proposed pass.

Prefer one continuous route per connected closed boundary where cutting and
linking are feasible. Minimize redundant travel and unnecessary retractions, but
allow branches, disconnected paths, ramps or retractions when continuity would
cross protected material or violate engagement/depth limits. Holes are separate
boundaries. A continuous route may revisit edges and need not be a simple loop.
Return the reason when continuity or complete coverage is impossible.

Acceptance: concave corners requiring extra passes, narrow access and islands
prove that route optimization preserves sweep limits and coverage. Compare path
length, retractions and residual material with the general calculator; optimize
only after correctness.

## Framework support and proposed helpers

Runtime architecture remains owned by [structure_spec.md](structure_spec.md).
Future modules should separate deterministic geometry, removal analysis, cutter
models and routing from project registration and XML adaptation. The project
owns identity, layers/groups, hierarchy and MOP attachment. These are proposed
responsibilities and names, not public APIs or a dependency decision:

| Helper | Responsibility |
| --- | --- |
| `resolve_operation_context` | Snapshot resolved tool/MOP parameters, ordered sources, world geometry, units, stock and setup |
| `plan_supported_mop` | Calculate cutting trajectories locally for an explicitly supported pocket/profile parameter subset; return unsupported semantics rather than silently approximating them |
| `normalize_regions` | Validate rings, holes, winding, arcs and tolerance; preserve source mapping |
| `sweep_toolpath` / `estimate_reachable_region` | Separate actual trajectory-derived cuts from ideal reachability estimates |
| `compute_rest_regions` | Boolean difference per depth, allowance/overcut classification and provenance |
| `expand_rest_region` / `feasible_tool_centers` | Tool-specific access without altering protected material |
| `tool_radius_at_height` / `check_tool_clearance` | Cutter cross-sections and full-height occupancy checks |
| `plan_vcarve` / `trace_vcut_boundary` | General fill and boundary cleanup with explicit limits and residuals |
| `verify_cleanup` | Independent swept-volume/slice checks; remaining material, overcut and constraint diagnostics |
| `attach_cleanup_operation` | Preview then explicitly register derived shapes and MOPs through project APIs |

Useful workflows include ranking candidate cleanup tools by predicted coverage
gain versus tool changes/time, combining overlapping rest components only when
safe access permits it, rest analysis over multiple prior MOPs, and cleanup
iteration until residual tolerance or a no-further-progress condition. Keep
minimum-area filters separate from machining success. Report intentional
unmachined features. A dogbone/relief alternative may be offered only with an
explicit change to the allowed design geometry.

Store provenance with results: source UUIDs and resolved group snapshot, operation
order, effective parameters, tool profile, units/depths, geometry/transforms,
algorithm/tolerance and evidence mode. Invalidate previews when any of these
change. Do not silently retarget source operations or assume live groups are
stable; coordinate with the higher-priority MOP source-compatibility work.

## Support prerequisites and implementation sequence

### Upstream dependencies

Region topology and Z-coordinate parity across shape types are owned by the
separate [shape parity plan](SHAPE_PARITY_PLAN.md); backlog 2 records their
completion. Consume those public geometry and XML contracts rather than reopening
Region storage, adders or Z migration. XYZ file fidelity is available, while
varying-Z Engrave motion still needs separate target-version acceptance.

The existing curved-bounds correctness item also supplies arc-aware bounds (or
rest analysis must use independently bounded arc normalization). Never derive
stock or clearance from vertex-only bounds of bulged contours. Rest-specific
Boolean/offset/sweep logic remains here; fixing the shared bounds contract does
not become a new rest-planner responsibility.

MOP-specific import remains partial and the writer serializes instructions rather
than generating trajectories. Audit every parameter used by the planner; local
operation/path calculation remains a rest-machining responsibility. Existing
MOP identity tests do not establish toolpath correctness.

### Proposed sequence

1. Consume the accepted Region/Z and shared geometry contracts from upstream.
   Keep XYZ/Engrave attachment gated on verified XYZ interchange and target-version
   motion. Do not broaden this workstream into shape parity implementation.
2. Prove pure rest geometry on synthetic planar pockets with known cutting
   trajectories before reproducing a full CAM engine. Reusable Boolean/offset
support must handle holes, disconnected pieces and bounded approximation.
   Evaluate geometry dependencies for robustness, license, packaging and measured
   need when this item is promoted; no dependency is selected now.
3. Add operation snapshots and local pocket/profile trajectory calculation for
   a declared subset. Begin with flat endmills, planar closed contours, explicit
   depths/allowances and one deterministic fill/profile strategy. Add stepover,
   depth passes and supported entry/link semantics with independent sweep tests.
   Implement or explicitly reject tabs, overcut and other settings before claiming
   toolpath-derived fidelity for operations using them. Compute distance fields,
   offsets and medial-axis candidates locally; no CamBam skeleton helper is an
   execution dependency.
   Ordinary MOP XML supplies instructions, not generated trajectories. If the
   original MOP will later be generated in CamBam, report whether the local model
   has verified equivalent removal for that supported subset; otherwise return
   a labeled estimate/uncertainty and do not certify unverified space as cleared.
   Where exact planned motion is required, investigate exporting the framework's
   own primary cutting paths through the verified XYZ/Engrave representation too,
   so primary and cleanup removal derive from explicit paths. This requires the
   same motion/link acceptance as the V-cutter proxy.
   Optional manually supplied reference paths or G-code may validate the planner,
   but acquisition is outside normal execution. Any future import must define
   supported modal/arc/tool/setup semantics and provenance.
4. Deliver endmill-safe expansion and verify the new generated MOP's coverage.
   Then add the constrained tool model and general XYZ V-carving calculator;
   V-specific rest preparation can use that calculator. Edge-tracing optimization
   comes after correctness, although its outcome is separately specified above.

Each future increment needs numerical fixtures and acceptance limits before
implementation. Use exact analytic cases where possible, then an independent
sweep/reference or conservatively bounded approximation; avoid validating an
offset algorithm only against itself. Compare symmetric-difference area, boundary
distance, residual volume when supported, topology and identity, not just bounds.
Choose tolerances from units, feature size and machining requirements; label
approximation and unsupported cases. Test tool-diameter/tip-radius extremes,
acute/obtuse corners, islands, touching/invalid contours, narrow channels, tabs,
allowances, overcut, multiple depths, mirrored/transformed shapes and unit changes.
For access decisions, use conservative known-clear regions: underestimating
removed stock is preferable to declaring unverified material empty. Propagate
offset/tessellation uncertainty, shrink claimed clearance or inflate obstacles
as appropriate, and report the resulting loss of reachable coverage. Keep nominal
geometry tolerance distinct from physical allowances for runout, tool wear and
setup error; these require explicit process inputs rather than invented defaults.

Initial analytic references, derived from the geometry model above:

| Synthetic case | Expected result under the stated ideal assumptions |
| --- | --- |
| 10-by-10 square pocket, full reachable-area coverage, flat endmill radius 2, no allowance/overcut | Four corner rest pieces; total area `(4-pi)*2^2 = 3.4336293856` square units |
| Same pocket after full reachable-area cleanup with radius 1 | Total remaining area `(4-pi)*1^2 = 0.8584073464`; a finite tool still leaves corners |
| 90-degree included-angle cone, available radius 2 at the reference plane | Pointed tip needs penetration 2; flat tip radius 0.5 needs penetration 1.5 to present radius 2 at that plane; neither result alone proves whole-tool clearance |

These are reachability/section references, not a prediction of arbitrary CamBam
MOP passes. The generated-path fixture must independently establish coverage.

For machine-facing acceptance, prepare synthetic A/B geometry, inspect CamBam
toolpaths/G-code and simulate the swept cutter before requesting a controlled
test cut. Include extra-depth and protected-volume checks explicitly. This backlog
documentation itself needs no manual machining validation.

## Promotion and stopping conditions

The 2026-09-22 request promotes design refinement for the concrete multi-tool
letter/region workflow. Implementation follows agreement on target semantics and
the bounded first outcome below. Region/Z parity is a completed upstream
dependency, not new rest scope. A separately reported data-loss bug can still be
prioritized independently on its own evidence.

The original proposal recorded all five outcomes. Refined design closure additionally
requires resolution of the product decisions below and numerical acceptance for
the selected first slice; those decisions are not yet accepted.
Future execution acceptance must demonstrate load, analyze, calculate and save
on a machine without CamBam installed. Manual CamBam compatibility checks remain
separate from this standalone execution requirement.
Do not implement a stock simulator, routing engine or CamBam integration during
this design task.

## Design refinement - 2026-09-22

### Intended product and accepted scope

Build a deterministic, standalone engine that combines caller-supplied tools to
remove an explicitly defined target volume while preserving protected material.
Expose region-only analysis, path-only planning, verified removal and optional
CamBam project attachment as independently useful operations. All calculations
run without CamBam installed. This is a proposed contract, not an implemented API.

User decisions accepted 2026-09-22:

1. Support the finish modes through a segmented, reusable model; include paired
   pocket/plug inlays with assembly gaps and constraints in the design scope.
2. Tools/catalogs belong to the caller or future CamBam client library. The engine
   consumes supplied tool specifications and restrictions, and can recommend
   combinations from that set. No hardcoded catalog belongs in the engine.
3. Continue this work on `feat/rest-machining-and-vcarving`; the user committed
   the first documentation round. This authorizes design work, not agent commits
   or merge. No branch delivery/readiness claim is made by this design record.
4. The engine is for general artistic and technical regions. The A is an example
   fixture, not a domain restriction; inlay is one application, not the organizing
   model for the core. Consider owning the geometry algorithms in this repository
   as well as using established low-level libraries. The user subsequently confirmed
   that proven packages are welcome: prefer established geometry primitives when
   they pass our contract, keeping machining semantics and verification owned here.

The [accepted RC01 case](#first-generated-acceptance-job-rc01) supplies the first
synthetic target, tools and residual/error criteria; production limits remain open.
The [2026-09-23 answer](#accepted-integration-requirement---2026-09-23) permits a first
framework-generated sequence while requiring native shape/MOP integration too.
No numerical production defaults or implicit
permission to change wall shape, floor depth or allowance are established here.

### Target geometry precedes tool strategy

| Finish mode | Target and meaning of completion |
| --- | --- |
| V-shaped recess | Opening plus nominal design angle/tip convention define a depth envelope; narrow corners rise to the surface. Completion means matching that envelope within stated tolerances. |
| Flat-depth V-carve | The same design envelope truncated at an explicit floor. Wide interiors can be cleared by endmills; the sloped boundary remains protected from roughing. |
| Vertical-wall pocket | Region extruded to a floor. A smaller endmill may reduce rest; a V-cutter cannot generally finish sharp vertical corners while preserving the whole wall. |
| Through-cut/profile separation | Explicit kerf/removal band, stock thickness, breakthrough and retained bodies/tabs. A separated central slug is released material, not material swept away by a cutter. |
| Edge finish/chamfer | Explicit contact surface and width/depth, not an implicit conversion of a pocket into a V-shaped recess. |

Keep the design envelope fixed when comparing tools. Do not define success as
whatever the selected tool happened to remove. An ideal unlimited cone may define
the requested V-shaped surface; the real cutter's finite diameter, rounded/flat
tip and flute length can make that surface infeasible. A depth cap is an intentional
design choice only when the caller selects a capped finish. A machine/tool limit
instead produces a limitation report and alternatives.

For the user's 5 mm endmill with 0.5 mm radial stock-to-leave, a locally straight
inside edge requires the center 3.0 mm from the design boundary; 0.5 mm remains
after the 2.5 mm radius sweep. In corners, calculate the sweep rather than assuming
a uniform strip. A Profile leaves the center material unless its passes actually
cover it. A Pocket must demonstrate interior coverage. If the final finish has
sloped V walls, derive roughing bounds at each depth from that target: a pocket
following the full top opening down to the floor may already destroy the slope.

### Mathematical contract

Use fixed vertical tool orientation (3-axis), explicit length units and positive
downward depth `d`; convert only at the adapter boundary to `Z = surface_z - d`.
Each depth slice owns desired removal `I(z)`, actual remaining stock `S_k(z)`,
protected stock/fixtures and independently established free space. With cutting
sweep `W_k`, update `S_(k+1) = S_k \\ W_k`; pure rest is `S_k intersect I`.
Report unintended removal separately and never let a subsequent operation hide it.

Represent an axisymmetric cutter by cutting radius `rho_cut(h)` and occupied
radius `rho_body(h)` at height `h` above its physical lowest point. The body includes
non-cutting neck/shank/holder where modeled. Missing body data limits collision
claims; a maximum cutting diameter is not permission to extend the cone forever.

- Ideal pointed/truncated cone: `rho_cut(h) = a + h*tan(alpha)`, where `alpha` is
  half the included angle, `a` is flat-tip radius, and `h` is within its declared
  cutting segment. The conical segment ends at `(Rmax-a)/tan(alpha)` or earlier
  at the specified cutting-length limit. Do not clamp the radius and pretend the
  resulting cylinder is cutting unless the tool specification actually says so.
- Optional spherical tip tangent to the cone: with ball radius `b`, use
  `sqrt(2*b*h-h*h)` for `0 <= h <= b*(1-sin(alpha))`; above the join, use
  `b*cos(alpha) + (h-b*(1-sin(alpha)))*tan(alpha)` up to the declared cone end.
  Validate tangency, continuity, diameter and segment domains. A generic rounded
  tool must supply its profile; a vague tip-radius value does not establish shape.
  This is a spherical lowest-point tip, not a rounded edge around a nonzero flat.
  Its cone-end height is `h_join + (Rmax-rho_join)/tan(alpha)`, not the flat-tip
  cone-end formula. Require `0 < alpha < pi/2` and a maximum radius compatible
  with the declared join; otherwise use a different explicit profile.
- At slice depth `z` and tip depth `d`, evaluate the tool section at `h=d-z`
  only where that section exists. Union the moving disks along the complete path,
  including changing radii, not just endpoint disks. Check non-cutting occupancy
  against current stock and fixtures separately from cutting removal.

For a flat endmill in allowed footprint domain `A`, valid centers are
`A eroded by disk(r)`. Restrict useful cutting candidates to centers whose sweep
intersects rest, as in outcome 2. For a V-cutter at depth `d`, intersect the
corresponding center constraints across all affected heights. Thus an empty floor
slice does not license the wider upper cone to cross a protected upper wall.

For a simple uncut planar opening, Euclidean clearance `q(x,y)` gives the candidate
cone depth `(q-a)/tan(alpha)` where `q >= a`. The medial axis supplies maximal-disc
candidates and branches; it is not the complete stock-aware solution. In a rest
operation, compute clearance against protected geometry, not every edge of the
rest polygon. A raster distance field may seed candidates, but thin-feature loss
and its error must be bounded before it can support a clearance claim.

Prefer depth-indexed planar sets with adaptive slice refinement for this fixed-axis
scope. The public stock contract must not require a particular grid resolution.
Slice checks need conservative between-slice bounds; an arbitrary stack of sampled
planes is not proof of volumetric clearance. Also bound error along XYZ segments.
For a cone, an XY clearance uncertainty `epsilon` implies depth uncertainty
`epsilon/tan(alpha)`; narrow-angle tools amplify error. Rounded tips require local
profile-aware bounds near zero radius. Separate geometric approximation, path
fitting, stock slicing and physical process allowances in reports.

### Wide areas, rest access and smoothing

The default strategy proposal is to derive feasible poses first, then cover target
rest using boundary/medial-axis candidates plus interior passes where needed.
Simply clipping the depth of a medial-axis path can leave wide side bands uncut.
Use a caller-selected flat-depth finish and clearance tools, or additional verified
passes with the V-cutter where feasible. If the requested uncapped target extends
beyond tool capability, return residual/infeasibility and suggest a larger/different
tool or an explicit target change. Never silently lower the floor or widen edges.

For a pointed cone clearing a nominal flat floor with parallel tracks at equal tip
depth, finite spacing leaves ridges: the ideal cross-section cusp is
`s/(2*tan(alpha))` for spacing `s`. Thus a pointed tip does not guarantee a perfectly
flat floor with finitely many passes. A flat-tip cutter has a finite floor footprint;
a rounded tip needs its own scallop calculation. Select passes using cusp/remaining
thickness and coverage criteria, not a fixed fraction of maximum cone diameter.

Keep original design, pure rest, allowed overlap domain, feasible centers and export
boundaries separate. Smoothing is an optional path/access operation, not permission
to fillet the desired corner. At interfaces with verified cleared space, extend
access and fit arcs only within the allowed domain. Protect holes and tabs at their
actual heights. Fit within a declared deviation and recheck the swept tool after
fitting, including added/removed coverage and topology changes. Reject a fit that
violates containment; straight segments are an acceptable result.

Minimize redundant cutting only after coverage and containment hold. Stepover
controls spacing between passes; overlap into cleared space is a separate access
and cost policy, and radial engagement depends on the current stock. Allow necessary
overlap by default in the proposal, report its amount, and honor an explicit hard
limit by reporting the coverage it prevents. Do not equate overlap with stepover.

### Detached core and native integration

Proposed boundaries (names are illustrative and not a committed public API):

| Layer | Contract |
| --- | --- |
| Geometry and cutter values | Immutable region sets, explicit frame/units, tolerance budget and piecewise tool profiles; no project IDs or XML requirement. |
| Target and stock | Fixed finish specification, stock snapshots, protected volumes, prior removal and evidence mode. |
| Analysis | Reachability, pure rest, safe access regions and infeasibility diagnostics; callable without planning paths. |
| Strategy and routing | Supported pocket/profile/V-carve strategies yield ordered motions, tool assignment, pass/entry/link roles and predicted residuals. |
| Verification | Independent or conservatively bounded sweep evaluator checks final interpolated motions and stock updates. |
| Composition | Compare supplied tools/operation sequences; preserve constraints, update stock in order and return tradeoffs and provenance. |
| CamBam adapter | Snapshot resolved world geometry/MOP values; map pure results to registered Region/Pline entities and explicitly selected MOPs. |

Use simple stable value types before inventing a plugin hierarchy. The existing
`Region` owns CAD topology/XML, not general Booleans or CAM stock simulation.
`cam_entities.py` owns MOP parameters and `cambam_project.py` registration. Compose
existing `machining_planning.py` process constraints where applicable; its endmill
diameter-based stepover/MRR model does not automatically model varying V engagement.
Keep these domain differences explicit instead of overloading its formulas.

Results must expose requested/achieved target, rest by depth, uncertainty, unreachable
features and causes, overcut, violated/missing constraints, operation order and source
fingerprints. Distinguish complete-within-tolerance, partial, infeasible, unsupported
and invalid input. An optimization timeout returns the best verified partial plan,
never a false completeness result. Ranking should first enforce hard constraints,
then residual quality, then caller-weighted time/tool changes/redundant travel.
Do not promise a globally optimal tool sequence or time without a machine model.

Keep two adapter modes explicit: native Profile/Pocket instructions delegate path
generation to CamBam; explicit path export delegates only path following. Native MOP
metadata alone cannot certify prior cleared space. Prefer framework-generated paths
for the first deterministic combined workflow. Until native equivalence is established,
native-source rest is an estimate with uncertainty, not guaranteed air for linking.
Engrave's `VCutter` enum describes the tool; it is not a V-carve calculation request.
Verify tool-tip Z, zero offsets, order/direction, depth passes, retracts and emitted
G-code before claiming the XYZ/Engrave adapter follows the computed trajectories.
Do not interpolate across disconnected paths by joining them into one polyline.

### Research basis and backend decision

Primary sources checked 2026-09-22; the architecture and formulas above are our
proposed synthesis/derivations, not claims that these products implement it identically.

- [Vectric V12 documentation](https://docs.vectric.com/docs/V12.0/VCarveDesktop/ENU/Help/page/single-page/)
  describes flat-depth carving, ordered clearance tools and subsequent tools handling
  areas earlier tools could not fit. This supports combined clearance/finish strategies.
- [Autodesk 3D Adaptive reference](https://help.autodesk.com/view/fusion360/ENU/?contextId=MFG-REF-3D-ADAPTIVE-CMD)
  distinguishes tool containment and stock sources for rest machining, and explicitly
  combines machining and geometry approximation tolerances. These motivate separate
  target, stock provenance, footprint constraints and error budgets.
- [CGAL straight-skeleton manual](https://doc.cgal.org/Manual/3.4/doc_html/cgal_manual/Straight_skeleton_2/Chapter_main.html)
  explains why straight skeletons differ from Euclidean medial axes at reflex vertices.
  Do not use skeleton event time unverified as circular cutter clearance.
- [OpenVoronoi author's medial-axis pocketing work](https://www.anderswallin.net/2012/02/medial-axis-pocketing/)
  supplies an established candidate approach worth evaluating, not proof that its
  implementation meets this project's packaging or machining requirements.
- [Clipper2 overview](https://www.angusj.com/clipper2/Docs/Overview.htm) and
  [arc tolerance](https://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/ArcTolerance.htm)
  document integer-scaled clipping and polygonal arc approximation. Evaluate coordinate
  scaling and approximation explicitly if selecting this backend.
- [Shapely manual](https://shapely.readthedocs.io/en/stable/manual.html) provides planar
  set operations and buffering. It is a candidate planar backend, not a 3D sweep engine.
- [OpenCAMLib documentation](https://opencamlib.readthedocs.io/en/latest/)
  describes drop/push cutter operations and cylindrical, ball, cone and composite
  models. It is a candidate for later surface work, not automatically a rest planner.

At this proposal stage no dependency was selected; the later
[Shapely/GEOS decision](#shapelygeos-evaluation-decision---2026-09-22) supersedes that
open choice. Backend evaluation criteria are analytic offset
accuracy, holes/near-tangencies, deterministic topology, supported Python 3.9-3.13
and Windows installation, license/distribution obligations and measured performance.
Start with planar Booleans/offsets; add a medial-axis backend only if the first
V-carve slice demonstrates a need. A package's existence does not justify a new
dependency or replacing the already accepted CAD representation.

### First useful increment and acceptance

The concrete [RC01 job](#first-generated-acceptance-job-rc01) now bounds the first
generated slice and includes native integration gates. The earlier broader example
below remains a follow-up acceptance case, not an additional first-job requirement.
After product decisions, implement one standalone end-to-end slice: explicit flat
endmill paths on a synthetic letter-like region with a hole, a 0.5 mm allowance,
stock/rest analysis, safe smaller-endmill access and generated cleanup paths with
independent verification. This establishes the same target/stock/clearance contract
that V-carving needs and directly addresses the user's artificial-rest-corner defect.
Pair it with an analytic cone feasibility fixture so the contracts do not accidentally
assume a cylindrical tool. It does not require reproducing every native MOP strategy.
Next deliver pointed-cone V-shaped and flat-depth carving with wide-area coverage;
then rounded/flat tips and combined-tool execution, followed by native adapter motion
acceptance. Optimize routing only after correctness. These are capability boundaries,
not a claim that all general engineering cases can be implemented in one increment.

Retain the earlier square-pocket references and add these executable acceptance
cases before implementation. Synthetic metric geometry may use 0.001 mm geometric
error for backend evaluation; this is a proposed fixture tolerance, not a machining
default. Area/volume and between-sample error thresholds must be derived separately.

| Case | Required evidence |
| --- | --- |
| Letter-like opening with hole, 5 mm endmill, 0.5 mm allowance | Straight-edge rest 0.5 mm; hole preserved; profile center remains unless actually swept; pocket interior coverage checked. |
| Smaller tool on resulting rest | No new wall at the rest/cleared-space interface; full target protected boundary preserved; residual compared with smaller-tool reachability. |
| Pointed 90-degree cone, 4 mm straight slot | Candidate tip penetration 2 mm; finite-length paths and end/corner behavior independently checked. |
| Same cone, 1 mm design depth cap | Effective top diameter 2 mm; additional paths/clearance required for a wider target; no centerline-only completeness claim. |
| 90-degree cone, 0.5 mm flat-tip radius, local clearance 2 mm | Candidate depth 1.5 mm; features narrower than the tip are reported unreachable. |
| Tangent spherical/conical tip | Radius and derivative continuous at analytic join; inverse profile and sweep agree with independent section values. |
| Vertical-wall sharp corner to fixed floor | V-cutter failure or residual reported; no surface-silhouette-only completeness. |
| Multiple depths, island, narrow neck, capped diameter, flute/shank | All-height containment and actual stock-aware links; partial results explain infeasibility. |
| Arc fitting and route simplification | Reverified between-point clearance, deviation and coverage; acute corners not silently deleted. |
| Units, translated/mirrored input and reordered operations | Equivalent physical result under unit/frame changes; stock evolution follows operation order. |

Broad scope explicitly excluded initially: tilted/multi-axis cutters, undercuts,
arbitrary mesh stock, automatic slug removal/workholding dynamics, inferred tool
catalogs and replication of undocumented native strategies. Report unsupported
cases rather than approximating them silently. Reopen each when a concrete job
needs it and the existing target/stock representation is shown insufficient.

Design refinement stops at a reviewed proposal and recorded open decisions. No machine
validation is needed for documentation. Before later manual acceptance, generate
and inspect A/B files and exact expected motions as required by the development
runbook. The A is suitable as one example; inlay implementation order is not a
core-design blocker. Complete the general geometry acceptance matrix and kernel
ownership evaluation before declaring the design ready for implementation.

## Shared core and paired inlay design

This section refines the preceding proposal after the user's scope decisions.
It is still design, not a declaration of implemented or production-accepted APIs.

### Small shared model

Use immutable values and pure functions; reserve extensibility for geometry
backends, target evaluation and path strategies where actual alternatives exist.
Do not build a generic workflow/plugin framework or a public class per machining
method. Proposed value groups:

| Value | Minimum responsibility |
| --- | --- |
| `RegionSet` | Multiple components, explicit outer/hole nesting, frame and units; bounded normalization from lines/arcs; stable geometry fingerprint. |
| `JobGeometry` | Required removal, permitted removal, initial stock, fixtures and setup; depth sections with conservative interval bounds. |
| `ToolSpec` | Cutting/non-cutting meridian profiles, physical tip datum, reach and capability facts; caller ID plus immutable geometry revision. |
| `PlanRequest` | Job, supplied tools, known prior motions/stock evidence, hard limits and optional optimization preferences. |
| `MotionPlan` / `AnalysisResult` | Ordered typed motion blocks and tool references, or areas alone; residual and clearance bounds, diagnostics and provenance. |

An inlay recipe produces two `JobGeometry` values and an assembly relation. It uses
the same analysis/planning functions as ordinary jobs. A future catalog adapts its
records into `ToolSpec`; neither catalog access nor CamBam entity creation occurs
inside the engine. Keep process/material recommendations separate from measured
cutter geometry. A catalog rename must not alter a cached result, while a cutter
geometry change must invalidate it even if the catalog ID is unchanged.

Keep constraints typed and scoped rather than an unrestricted options dictionary:
tool facts (profile, reach, center-cutting/ramp capability); job limits (floor,
breakthrough, allowance, fixtures, permitted relief); pass limits (axial increment,
radial engagement, ramp slope, feeds); finish/error limits (cusp, residual thickness,
boundary deviation); and preferences (tool-change cost, path length, overlap).
Distinguish unknown from zero, and hard limits from recommendations. A missing
process limit can still allow region analysis, but prevents claiming an executable
plan has passed that limit. Add new fields only with defined units and an owning
evaluator; "all relevant parameters" is not permission to accept inert options.
For a V-cutter, contact radius and engaged axial range vary along the path; a
single maximum diameter is insufficient for local cutting-speed or chip-load
claims. Keep geometric feasibility available without inventing a cutting-force
model, and report unassessed process constraints separately from geometric success.

The shared mathematical operations are section, offset/set operations, feasible
pose evaluation and motion sweep. Specific strategies propose paths using them.
Verification evaluates the actual proposed/exported motion, not the strategy's
claim of what it intended to cover. Independent analytic references and conservative
bounds are required; a second wrapper around the same offset call is not independence.

### Canonical target construction and uncertainty

Keep three different spatial permissions:

- `I`: material required to be removed for the final finish.
- `L`: material allowed to be removed, including explicitly authorized relief;
  ordinarily `I` is contained in `L`. Extra removal within `L` but outside `I` is
  reported as relief, not credited as required coverage.
- `E_k`: space known empty before motion `k`, including verified prior removal.
  Free space is not an instruction to remove material elsewhere.

Cutting occupancy may enter `L union E_k`, excluding fixtures/protected volumes.
Non-cutting occupancy must avoid remaining stock as well as fixtures. A rapid may
not cut even when its path lies in `L`. Check entry/ramp/exit motion against evolving
stock; do not use the final operation's removed volume to justify its own entry.
These are geometric conditions; engagement and machine/process limits also apply.

A useful exact design definition for an ideal pointed V recess is particularly
small. Let `Omega` be the top opening, `q(x)` its internal Euclidean distance to
all boundaries including islands, `alpha` the design half-angle and `D` an optional
design depth cap. Then, in positive downward coordinates:

`H(x) = min(D, q(x)/tan(alpha))`

`I(z) = Omega eroded by disk(z*tan(alpha)),  0 <= z <= D`

Here `H` is desired depth, not a tool-center trajectory. The second equation
follows directly from `H(x) >= z`. With no cap, use the first expression without
`min`. A vertical-wall pocket instead has `I(z)=Omega` through its floor depth.
The capped V-carve's flat floor exists only on `Omega eroded by disk(D*tan(alpha))`;
narrow features may never reach that floor. A cap does not give the whole opening
a flat bottom.
Design-flat/rounded-tip recesses need an explicit alternative target definition;
an actual flat/rounded cutter does not silently change this ideal pointed target.
Allow custom bounded section evaluators later rather than forcing every job into
one angle formula. These definitions use nominal ideal geometry; finite tools,
access and tolerances determine whether it can be manufactured.

This makes roughing simple to specify: at depth `z`, a cylindrical endmill of
radius `r` needs centers inside `I(z) eroded by disk(r)` when preserving the exact
target. For this nested ideal target that floor slice also bounds the upper
cutting cylinder. This shortcut does not apply to arbitrary non-nested stock,
holder occupancy or other target evaluators; those need all-height checks.

Prefer analytic primitives and adaptive depth slabs, not a universal voxel grid.
Initial target slices may be nested, but residual slices need not be. Preserve
per-slab bounds and mandatory events at floors, stock surfaces, tool-profile joins,
tab heights and offset topology changes. Refine other intervals until the error
budget is met or return unresolved uncertainty.

Use inner/outer removal bounds `C_lo subset C_true subset C_hi`. For exact initial
stock `S0`, remaining stock is bounded by `S0 \\ C_hi` and `S0 \\ C_lo`; intersect
both with `I` to bound rest. Only guaranteed removal can justify known-free travel.
Do not replace nominal geometry by a single rounded approximation and label it
exact. Tool/setup uncertainty is a separately declared physical envelope.

One compact sweep primitive can support many strategies: at a fixed depth plane,
a straight XY segment with a linearly varying nonnegative cutter-section radius
sweeps the convex hull of its two endpoint disks. Constant radius is the capsule
special case. Split an XYZ segment at tool section entry/exit and profile joins
before applying this result; conical segments then admit affine-radius treatment.
Rounded sections need analytic treatment or conservative adaptive radius bounds.
This avoids relying on densely sampled endpoint disks that can leave false gaps.

### Topology is part of the contract

Keep authored CAD contours intact; normalize a bounded approximation for the
backend and retain its error/source mapping. Closed polylines alone are not enough:
explicitly represent holes, separate components, boundary contacts and collapsed
features. Do not silently repair self-intersections or join nearly touching islands.
Return a proposed repair and its geometric/topological change when repair is needed.

Offsets can split, merge or eliminate components legitimately. Record these events
and their depth. A small area is not automatically insignificant: a long thin strip
can violate a wall requirement, and a tiny lost bridge can disconnect an inlay.
Use boundary deviation, remaining thickness, component connectivity and volume/area
together. Report features below representable resolution rather than dropping them.

Feasible centers can be curves or isolated points: a slot exactly one tool diameter
wide has a centerline, not a positive-area center region. A circle exactly matching
the cutter has a single feasible center. Preserve these strata or explicitly report
them unsupported; an empty polygon erosion alone must not mean "unreachable".
Practical clearance/entry constraints can still rule out an exact-fit nominal case.

Geometry fitting must preserve containment and required sharp features within its
error budget. Medial-axis branches can be sensitive to small contour perturbations;
prune only after demonstrating negligible effect on required coverage. Offset
loops need component-aware routing; merging them is not permission to cross an island.

### Inlays as paired targets

Use an assembly coordinate frame plus separate pocket and plug machining frames.
Define the retained plug solid and receiving cavity first, then derive the removal
volumes from their respective stock. The plug is a bounded exterior-clearing job,
not the unbounded complement of an outline. Preserve backing stock and required
connections until an explicitly ordered release/face-off operation.

| Parameter | Meaning in the engine |
| --- | --- |
| Seating/engagement depth | Intended insertion relative to the finished receiver surface. |
| Bottom glue gap | Axial separation at designated bottom-facing surfaces; report where narrow features have no common flat floor. |
| Surface clearance | Separation between designated opposing stock faces at the seated pose; provides room for assembly and later backing removal. |
| Side-fit allowance | Signed lateral or surface-normal allowance with its measurement convention explicit; separate from bottom glue gap. |
| Backing thickness / facing allowance | Material retained for handling and the later removal needed to expose the finished inlay. |
| Assembly transform | Physical flip/rotation, registration and insertion; derive any mirrored 2D machining projection from it. |

Nominal side contact with a bottom gap is a useful geometric baseline. A rigid-body
model can detect intended interference but cannot predict compression fit in wood,
glue flow, shrinkage or required assembly force. Such process allowances are supplied,
not invented. Side clearance may expose a visible seam and must not be conflated with
hidden bottom glue space. Negative allowance is a deliberate interference request.

For a local straight tapered wall `x(z)=b-z*tan(alpha)`, changing seating by axial
amount `delta_z` changes lateral fit magnitude by `abs(delta_z)*tan(alpha)`;
normal separation magnitude is `abs(delta_z)*sin(alpha)`. Thus seating, visible
seam and side-fit allowance are coupled. This local relation is not a universal
offset recipe at corners, rounded tips or capped floors; check assembled solids.
Do not copy another CAM product's Start Depth formulas into the engine's semantics.
A reference depth does not prove the stock above it was already removed.

Verify pocket and plug after applying their assembly transform: no unintended solid
intersection, requested gap/contact at designated surfaces, sufficient insertion,
no premature backing/shoulder contact and the expected visible outline after
face-off. Check the insertion motion as well as the final pose. Two tools with
different shapes are permitted only if their achievable surfaces satisfy these
tests; "same nominal angle" alone is insufficient. First support flips preserving
parallel depth planes; tilted assemblies/undercuts require a separate representation.

### Machining methods and extension points

| Scenario or method | Shared capability and design implication |
| --- | --- |
| Contour-parallel pocketing | Successive offsets and component routing; straightforward first strategy, but check residual strips and corner engagement. |
| Raster/zigzag clearing | Clip parallel passes to feasible centers; explicit turns/retracts, islands and boundary finish. Useful alternative for wide flat areas. |
| Medial-axis V finishing | Clearance controls depth; branch traversal and variable-Z motion; limited diameter/depth requires additional coverage. |
| Adaptive/trochoidal/spiral roughing | Reuse stock and occupancy but add engagement/curvature control; constant stepover does not imply constant tool load. Defer until measured need. |
| Rest with a smaller endmill | Reuse target and updated stock; cut only useful portions but allow access through known-clear regions. |
| Flat/rounded-tip cleanup | Same swept-profile model; different reachability and floor-cusp limits. |
| Profile separation and tabs | Finite cut band, retained/released bodies and ordering; do not treat a loose slug as known empty. |
| Chamfer after pocketing | Separate desired contact surface; interior free space helps access but upper-wall occupancy still constrains the cone. |
| Tapered or straight-wall inlays | Two targets plus fit/assembly checks; shared planners and stock verification. |
| Rest from an imported/native job | Require resolved parameters and evidence mode; estimates cannot authorize assumed-air rapids. |

The first strategy should be deterministic and easy to verify, not the most elaborate
high-speed path. Choose among supplied tool sequences with hard constraints first.
Support caller-fixed order and bounded automatic search. A greedy largest-first
plan is a useful candidate, not a universal optimum: it can add a tool change for
little gain or constrain entry. Report quality/time/tool-change tradeoffs; label
time as an estimate when acceleration, feeds or tool-change times are unknown.
Reuse stock snapshots across candidates; stop on bounded search budget, tolerance
satisfaction or no verified progress. Never prune a tool solely because it has a
larger nominal diameter: tip shape, reach and engagement capabilities also matter.

### Engineering sources and library direction

The following primary references inform this refinement; the model above is our
design synthesis, with formulas derived under the stated assumptions.

- [LaValle, configuration-space planning](https://lavalle.pl/planning/node161.html)
  gives the obstacle/shape translation framework behind cutter-center constraints.
  Machining additionally changes stock; static collision-free poses are only one layer.
- [Held and Spielberger, spiral pocket machining](https://www.cad-journal.net/files/vol_11/CAD_11%283%29_2014_346-357.pdf)
  studies Voronoi-based pocket decomposition, stepover, curvature and engagement.
  This supports keeping stock correctness independent of later route optimization.
- [Vectric V12.5 inlay documentation](https://docs.vectric.com/docs/V12.5/Aspire/ENU/Help/form/VCarve%20Inlay%20Toolpath/)
  distinguishes glue gap and surface clearance and generates paired operations.
  Its same-V-bit rule is product behavior, not our proof of arbitrary-tool compatibility.
- [Boost.Polygon Voronoi](https://www.boost.org/doc/libs/1_63_0/libs/polygon/doc/voronoi_main.htm)
  handles point/segment sites with integral inputs and nonintersection preconditions.
  It is a candidate for true segment-clearance graphs, not a full machining engine.
- [Shapely precision model](https://shapely.readthedocs.io/en/2.1.0/reference/shapely.set_precision.html)
  explicitly documents collapse/removal of narrow features under precision reduction.
  Backend validity therefore does not prove preservation of manufacturing intent.
- [Shapely 2.1 release requirements](https://shapely.readthedocs.io/en/2.1.0/release/2.x.html)
  require Python 3.10+, whereas this project supports 3.9-3.13. A compatible version
  strategy must be tested; do not silently increase the project's minimum Python.
- [CGAL Minkowski sums](https://doc.cgal.org/latest/Minkowski_sum_2/group__PkgMinkowskiSum2Ref.html)
  offers exact or guaranteed-approximation offsets and lists GPL licensing for this
  package. Distribution and binding implications must be evaluated before adoption.

No backend is preferred solely because it offers a short Python API. With the
user's clarification, evaluate established planar packages first under the ownership
criteria below; a full competing local kernel is not a prerequisite. Shapely/GEOS
and Clipper2 remain candidates, not dependencies or
selected architecture. Choose one production implementation per primitive rather
than maintaining equivalent competing kernels. Separately evaluate segment Voronoi
when required by V-finishing; avoid raster skeletonization as an unbounded
substitute. A general mesh/solid kernel is not required for the current fixed-axis scope.

### Algorithm ownership and dependency policy

The user considered an in-repository implementation and then explicitly accepted
proven packages. Prefer established low-level geometry where acceptance demonstrates
fitness; implement bounded gaps locally when justified. There is no requirement
to build a second general polygon kernel to make that decision. Own the domain
semantics and algorithms regardless of how low-level operations are provided:

| Responsibility | Proposed ownership |
| --- | --- |
| Cutter profiles, target evaluators, rest/stock semantics and error budgets | Implement and maintain here. |
| Feasible-pose constraints, tool combinations, strategies, entry/link policies | Implement and maintain here; no opaque third-party CAM engine. |
| Analytic line/arc distances, profile inversion, simple sweeps and adaptive subdivision | Prefer small in-repository implementations with analytic checks. |
| General polygon overlay/offset topology and segment Voronoi construction | Evaluate established packages first using adversarial acceptance and maintenance evidence; own bounded gaps if needed. |
| Numeric arrays and acceleration | NumPy or justified low-level support; not a substitute for robust geometric predicates. |
| CamBam registration/XML and client catalogs | Existing adapters/owners; no backend objects or catalog dependencies in the public machining model. |

Implementing general Booleans means owning intersection classification, splitting
edges, consistent vertex identity, coincident-edge handling, ring/hole reconstruction
and valid set topology. Offsetting additionally needs self-intersection resolution
and component split/merge/collapse handling. Robust orientation/incircle predicates
alone do not guarantee robust constructed intersections or assembled topology.
NumPy floating-point operations and a single global epsilon do not solve these
problems. Exact predicates also do not make irrational arc/offset coordinates exact.

For an in-repository candidate, declare the coordinate model and supported inputs
first: bounded polygonal approximation versus analytic arcs; scaled integers/exact
rationals where appropriate versus filtered floating-point predicates with exact
fallbacks. Track snapping and construction error separately. Reuse existing Region
validation only where its contract fits; its CAD topology checks do not establish
a Boolean or medial-axis kernel. Do not grow new general CAM algorithms inside the
Region XML/entity owner.

Evaluate implementation ownership by correctness evidence, auditable failure modes,
supported platforms, dependency/binding burden, license/distribution requirements,
performance on representative contours, and long-term maintenance. Count the robust
kernel and regression burden, not only wrapper lines. A local implementation removes
upstream-change risk but transfers algorithmic defects and maintenance to us; neither
local ownership nor library popularity establishes correctness.

The first decision experiment covers union/difference, disk offsets, holes,
coincident/near-tangent boundaries and narrow-feature preservation using the
[acceptance corpus](#acceptance-corpus-v1). Evaluate Shapely/GEOS first; examine
Clipper2 or a bounded local primitive if a demonstrated gap justifies it. Record
why a failure belongs to the package, our adapter, or an unsupported representation
before replacing a backend. Isolate established primitives behind a small internal function
boundary with owned value types. No backend plugin framework is needed. External
libraries may also serve as development-only comparison oracles without becoming
runtime dependencies; agreement is corroboration, not proof.

Evidence: [GEOS](https://libgeos.org/) identifies PostGIS, QGIS, GDAL and Shapely
as consumers, so it should not be classified as an unproven niche CAM library.
[Shewchuk's robust-predicate research](https://www.cs.cmu.edu/~quake/robust.html)
explains why floating-point sign errors matter and supplies adaptive-precision
orientation/incircle algorithms. These are reasons to evaluate maturity and numeric
contracts explicitly, not blanket endorsements or commitments to vendor code.

### Acceptance additions and next decision

One proposed combined-tool fixture (the user accepts the A as an example):
millimetres, stock top at Z=0, stock thickness 8; outer A-shaped contour
`(-24,0), (-8,60), (8,60), (24,0), (12,0), (6,18), (-6,18), (-12,0)`;
triangular hole `(-4,28), (4,28), (0,44)`. Use the ideal 90-degree included-angle
target capped at 3 mm depth. Supply a 5 mm flat endmill and a pointed 90-degree
V-cutter with 12 mm maximum cutting diameter and at least 6 mm conical cutting
height. Roughing retains a 0.5 mm planar allowance against each target section;
finishing removes that allowance within the final target. These are synthetic
geometry inputs, not manufacturer/process recommendations or approved feeds.
Keep the earlier square and straight-slot cases as independent analytic references;
this A fixture exercises composition, holes and varying local width.

Add analytic checks for ideal V target sections, affine-radius segment sweeps,
uncertainty ordering, exact-width slot/point-center cases, a narrow bridge near the
precision limit, and assembled tapered-wall fit. Inlay verification must test a
correct pair, wrong flip, premature bottom/backing contact, excessive visible side
gap and a narrow feature without a flat bottom. Prove geometry before optimizing.

The A cannot stand in for general-shape acceptance. Organize the fixture matrix by
geometric difficulty and machining outcome rather than application names:

| Family | What it must establish |
| --- | --- |
| Rectangles, circles, annuli, exact-width slots | Analytic coverage, offsets, isolated/curve center sets and dimensional limits. |
| Concave technical pockets, fillets, multiple islands | Arc handling, topology, disconnected rest and safe overlap. |
| Organic/artistic contours, fine strokes, acute wedges | Curvature, varying width, detail retention and bounded approximation. |
| Disconnected regions and nested ring hierarchies | Explicit fill semantics, component ownership and independent routing. |
| Near-tangencies, coincident edges, tiny bridges, large coordinate ranges | Robust predicates/constructions, explicit uncertainty and no silent topology damage. |
| Wide/depth-capped areas, multiple floors, tabs | Tool limits, full-height occupancy and stock evolution. |
| Paired mating parts, including inlays | Composition of ordinary jobs and assembly constraints; no core specialization for lettering or inlays. |

Use analytic cases, adversarial constructions and deterministic generated shapes.
Add unit/translation/rotation invariance, valid similarity scaling with scaled tool
and tolerance inputs, contour-order independence and stock-removal monotonicity.
Nonuniform scaling of a design does not scale a physical round cutter into an ellipse.
Use set identities and independent bounds, not merely pictures or agreement between
two calls to the same algorithm.

The standalone endmill-rest slice remains the internal foundation, with pointed-cone
target and sweep checks alongside it. Inlay order can be chosen later without
blocking that foundation. Design closure needs numerical tolerances and expected
results for the general-shape matrix plus the kernel ownership decision; choosing
a showcase application is not a prerequisite or a substitute.

### Acceptance corpus v1

The reusable [JSON corpus](../tests/fixtures/rest_vcarve_acceptance.json) owns exact
synthetic inputs and numerical reference values. This section owns their meaning,
acceptance rules and limits. The [reference checks](../tests/test_rest_vcarve_acceptance_fixtures.py)
validate analytic expectations and the existing CAD input representation without
installing a candidate geometry package. Passing them does **not** accept a backend,
stock engine, path planner, Engrave adapter or manufactured result.

All inputs use millimetres. `reference_numeric_tolerance=1e-9` checks scalar oracle
consistency only; it is not a machining accuracy claim. For the future backend
runner, require maximum boundary distance <= 0.001 mm, area error <= 0.01 mm2 and
volume error <= 0.01 mm3 when applicable. These are independently enforced synthetic
limits: satisfying boundary distance alone does not imply the area/volume limits.
Refine curves/integration as necessary; never loosen the expected reference to
match the candidate. Units/scaling tests must scale each limit dimensionally.

For regions with an independent reference construction, check symmetric-difference
area as well as signed area error; equal area alone does not prove equal geometry.
Check boundary distance in both directions and exact expected component/hole counts.
For line/point feasible sets, compare dimension and locations, not area. Geometric
approximation limits do not authorize cutting outside protected stock: the future
planner must bound occupancy conservatively and report unresolved uncertainty.

| Case | Required outcome and evidence class |
| --- | --- |
| G01 rectangle overlay | Union, intersection and difference have analytic areas; shared boundaries do not create spurious holes/components. |
| R01 square reachability | Four ideal inaccessible corner pieces for each radius; smaller-tool gain is the difference in ideal rest. Not generated-path coverage. |
| R02 profile versus pocket | Explicit closed square-center trajectory sweeps 236.5663706144 mm2; ideal pocket coverage is 357.5663706144 mm2. The profile retains an 11 mm square center. |
| G02 annulus inset | Outer radius shrinks and hole radius grows; one component and one hole survive. |
| V01 square V target | At depth 2, floor area is 36 mm2; volume to the cap is 130.6666666667 mm3. This is a desired volume, not proof a path cuts it. |
| V02 circular V target | Sections remain concentric circles; analytic volume and floor extent match numerical integration. |
| C01 pointed/flat cones | Depth inversion and finite cone extent use the physical tip datum; clearance below flat-tip radius is unreachable. |
| C02 spherical/conical tip | Radius and slope meet continuously; maximum diameter uses the joined-profile height, not the pointed-cone formula. |
| S01 constant-radius sweep | A 6 mm segment with radius 2 sweeps a capsule, including the space between endpoint disks. |
| S02 affine-radius sweep | The disk convex hull has area 20.2309724380 mm2; a separate support-function integral checks the reference. Endpoint union is insufficient. |
| T01 exact-fit centers | Rectangle yields a segment; circle yields one point. Zero center area is not nominal infeasibility. Entry/physical clearance remains separate. |
| T02 near contact | Overlap and shared-edge contact form one interior component; the 0.004 mm gap preserves two. Do not close it through precision reduction. |
| T03 narrow bridge | Connected input area 32.04 mm2; an inset larger than the bridge half-width leaves two components without holes. |
| A01 general composition | Valid concave outer contour and triangular hole, opening area 1532 mm2. Combined-tool path acceptance deliberately remains pending. |

S01/S02 describe continuous XY sweeps at one depth. They do not establish complete
XYZ motion: the runner must later split at profile joins and slice entry/exit and
prove between-depth bounds. R02's pocket value assumes exhaustive ideal reachable
coverage, whereas its profile value comes from the explicit supplied trajectory.
Keep these evidence classes visible in any machine-readable evaluation report.

The first backend evaluation should consume the planar cases and target sections,
report actual errors/topology by case ID, and identify lower-dimensional operations
needing local support. Do not require a polygon-only package to encode a centerline
as a positive-area polygon. Acceptance is of the package plus its bounded adapter
contract, with unsupported capabilities explicit, not of the package in isolation.

Reference validation command, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_rest_vcarve_acceptance_fixtures.py -v
```

The [adversarial supplement](REVIEW.md#adversarial-planar-acceptance-and-internal-contract---2026-09-22)
adds bounded invalid-input, arc-heavy/organic and transformation evidence without
changing corpus v1. Continuous multi-depth stock and protected-volume checks,
capped wide-area path coverage, motion ordering and entry, then application-level
paired-inlay assembly remain later gates. These fourteen reference cases and the
supplement are not a claim of general engine acceptance.

### Shapely/GEOS evaluation decision - 2026-09-22

Select Shapely/GEOS as the planar primitive candidate for the detached-core design.
The bounded corpus supports continuing with its overlay and disk-offset operations;
it does not justify a replacement polygon kernel or a second package comparison.
This is a design selection, not runtime dependency adoption or general machining
acceptance. The [evaluation evidence](REVIEW.md#shapelygeos-planar-evaluation---2026-09-22)
records measured errors, the Windows matrix and the exact-fit limitation.
The reusable [development runner](../tools/evaluate_shapely.py) stays outside the
runtime package and normal backend-independent reference tests.

The proposed internal boundary must own these requirements:

- Pass finite, valid, explicitly filled XY regions to GEOS. Preserve component/hole
  meaning in owned values; do not expose Shapely objects in the public model or
  silently repair invalid topology. Z/depth, stock evidence and cutter semantics
  remain the detached core's responsibility.
- Set curve approximation from a dimensional error budget and check both area and
  boundary error. The experiment's `quad_segs=256` is sufficient for this corpus,
  not a universal resolution. Analytic arcs require bounded conversion. Offset
  approximation does not establish conservative protected-stock containment.
- Preserve nominal feasible sets with dimensions 0, 1 and 2. Polygon erosion alone
  cannot implement exact-fit centers. Bounded analytic rectangle/circle handling
  is the first local support to prove; general collapsed sets remain unsupported
  until a declared algorithm and uncertainty policy cover them.
- Do not apply an implicit precision grid. Snapping can merge the corpus's 0.004 mm
  gap. A future grid policy must account for construction error and topology loss
  separately and reject unresolved features rather than infer infeasibility.
- Restrict the first implementation to the common tested 2.0/2.1 API. The tested
  version policy is `shapely==2.0.7; python_version < '3.10'` and
  `shapely==2.1.2; python_version >= '3.10'` on CPython Windows x64 3.9-3.13.
  Keep the existing Python minimum. Re-run this runner and platform checks before
  widening versions; 32-bit Windows, ARM64 and source builds are not verified.
  Add an optional machining dependency only when a runtime slice needs it;
  `pyproject.toml` remains the sole dependency declaration owner.

The [internal planar contract](#internal-planar-value-and-error-contract) below
closes the next design boundary. Its adversarial evidence supplements corpus v1;
continuous multi-depth stock, motion ordering and paired inlays remain later work.
Reopen backend selection if supported-input topology or the declared error budget
cannot be met, the supported wheel matrix fails, or representative work shows a
material performance bottleneck.

### Internal planar value and error contract

This is the design boundary for detached implementation. The first nominal
runtime subset is now implemented; its API and explicit exclusions live in
[the implemented specification](structure_spec.md#detached-nominal-planar-core).
Development probes demonstrate bounded cases; they must
not be imported by runtime code. Geometry/error ownership belongs to the detached
core, with one private Shapely adapter. Existing CAD `Region` validation and XML
rounding remain separate contracts: passing CAD validation is not a machining
accuracy certificate. The implementation adds an optional `planar` dependency
extra, while this design contract introduces no path planner.

**Owned inputs and values.** Use immutable tuples/scalars and explicit tagged
variants, rather than backend objects or a general options dictionary:

| Value | Required fields and meaning |
| --- | --- |
| `PlanarFrame` | Source units (`mm` or `inch`), explicit XY frame identity and origin, and section Z when applicable. Convert to local millimetres once; retain the inverse mapping and conversion provenance. Never infer a frame from coordinates. |
| `RegionSet` | Tuple of filled components, each with one closed shell and its explicitly assigned holes. Segments are lines or directed circular arcs with finite endpoints, center/radius/sweep and source references. Empty sets are explicit. Analytic circle and rectangle identity is retained when supplied; it is not inferred from an approximate polygon. |
| `PlanarApproximation` | Valid polygonal coordinates, source-segment/parameter mapping, measured or bounded error contributions, frame mapping and original analytic input fingerprint. Authored contours remain intact. |
| `FeasibleSet` | Separate area components, segment chains and isolated points, with dimension set drawn from `{0,1,2}`; the empty set has no dimensions. An analytic disk/rectangle area can remain analytic until polygon conversion is requested. Dimensions are never classified by an area threshold. |
| `ErrorBudget` | Positive finite boundary distance and area limits, optional volume limit only when an owning evaluator exists, and work limits for refinement. Numeric units are explicit. Physical clearance, runout, tool uncertainty and desired finish allowance are separate inputs. |
| `PlanarResult` | Status, operation/evidence class, optional owned value, requested budget, error ledger, topology events, diagnostics and provenance. Partial/unresolved geometry is diagnostic only and cannot be consumed as accepted stock or access. |

The native adapter must resolve effective transforms and section elevation before
detachment. A varying-Z contour is not silently projected into a planar opening;
return `unsupported` unless an explicit section/projection operation supplies that
meaning. Similarity-transformed circles/rectangles retain analytic provenance;
arbitrary transformed contours do not gain it from bounding boxes or curve fitting.
Binary operations require the same explicit XY frame and applicable section Z
after unit conversion, or an explicitly supplied mapping into a common frame.
Coincident numeric coordinates do not establish alignment. Differing sections are
not combined by a planar operation.

Source references identify component/ring/segment and parameter interval; they
do not require project IDs, layers, XML or mutable entity references. Provenance
records normalized input and policy fingerprints, operation parameters, adapter
revision and actual Shapely/GEOS versions. Fingerprints canonicalize ring start,
winding and component/hole order without quantizing coordinates; they include
frame/units, analytic segment meaning and error policy. Source mapping is kept
separately so reordered sources can share a geometric cache key without returning
stale source labels. Hash identity means identical canonical inputs/policy, not
approximate geometric equivalence or proof of accuracy.

**Admission and topology.** Validate finite numbers, positive primitive sizes,
arc consistency and explicit closure before handing coordinates to GEOS. Reject
zero-length edges, fewer than three distinct vertices in polygonal rings or
approximations, zero-area rings,
self-crossings/overlaps, outside or nested/overlapping holes and any hole touching
its shell or another hole. Ring winding and cyclic start are representation
choices; shell/hole role comes from ownership, never winding alone. Independent
components must have disjoint interiors; boundary contacts are allowed and must
remain explicit. Combining overlapping filled inputs is an explicit union
operation, not a silent normalization repair. Union may coalesce shared edges;
point contact must not be credited as a traversable positive-width passage.

Validate both the analytic source and its approximation. A valid chord polygon
does not prove that the original arcs are valid. If source topology or preservation
through approximation cannot be established within the work/error limits, return
`unresolved` (or `unsupported` for an unimplemented predicate). Do not call
`make_valid`, `buffer(0)`, apply a precision grid, remove small areas or merge close
vertices implicitly. Diagnostics identify the offending sources and reason;
repair proposals require a separate explicit operation and a topology/error diff.
An offset can legitimately split or remove an area component; record before/after
component and hole counts, while keeping feasible-set dimension claims separate.
The polygon vertex-count rule does not apply to authored analytic arc rings:
a circle or two-semicircle boundary is allowed when its analytic topology is valid.

Region operations use regularized filled-area semantics: the closure of the
interior of the set result. Thus polygon difference can retain its cut boundary,
and boundary-only intersection is an empty **area** result, not proof that the
literal intersection has no points. Exact closed-set erosion for `FeasibleSet`
is a different operation. Do not use regularized area difference to prove that
protected boundaries are untouched. Validate output representability as well as
backend validity: a valid Boolean output with a shell/hole contact, touching holes
or a pinched ring outside the strict owned representation returns `unsupported`
with the contact diagnostic. Never report it as empty or as a backend defect.
Independent components touching at points remain representable, with contact
locations recorded; first-slice output closure is therefore explicitly bounded,
not a promise to encode every GEOS polygon as an accepted `RegionSet`.

**Dimensional error ledger.** For an arc of radius `R` and per-chord angle `a`,
the chord deviation is `R*(1-cos(a/2))`, and its enclosed area deficit is
`R*R*(a-sin(a))/2`. Sum absolute segment deficits for an area budget; do not allow
shell/hole cancellation to hide error. Subdivide adaptively using both limits,
with stable small-angle evaluation and a bounded segment count. A smooth curve
needs its own derivative/interpolation bound; merely doubling sample density is
an observation, not a continuous error bound. Preserve corners and arc endpoints.

Track input approximation, operation approximation, numeric/coordinate effects
and output conversion separately. A certified operation must establish how these
contributions propagate and sum within the requested budget; an input chord bound
alone does not bound an offset near a topology event. Record observed metrics as
observations and proven bounds as bounds. In particular, the GEOS acceptance
measurements are not a universal floating-point or conservative containment proof.
An unavailable contribution is `unknown`, never zero. A result may pass nominal
synthetic gates while remaining ineligible for safety/stock certification.
The boundary metric is the continuous bidirectional maximum distance between
boundaries; the area metric is symmetric-difference area against the intended
region. Scalar area difference is an additional diagnostic, not a substitute.
Empty/nonempty mismatches and changed connectivity cannot pass by having small
area error. `nominal_geometry` may carry unknown operation/numeric bounds only
with `budget_certified=false` and the unknown terms exposed; any known limit
violation returns `unresolved`. A request requiring certified bounds cannot be
satisfied by nominal success. Even a certified geometric approximation needs a
separate directional containment proof before it is used as guaranteed removal.

Normalize near a deterministic local origin before backend operations and account
for source-coordinate quantization and reconstruction. Recentering cannot restore
detail already lost at a large world origin. Reject nonfinite conversion, overflow,
underflow that loses a feature, or numeric resolution that exhausts the budget;
otherwise retain the resolution allowance in the ledger. The coordinate probe
tests a 1e9 mm translation and rejects 1e15 mm for a 0.001 mm boundary budget;
these are evidence points, not a universal coordinate-range promise. Similarity
scale `s` scales distances by `abs(s)`, areas by `s*s`, and volumes by `abs(s)**3`.
Apply the same scaling to budgets, tools and section depths. Inch conversion uses
25.4 mm/inch. Reflections reverse arc direction and winding without changing fill.
Anisotropic transforms turn circles into ellipses: reject them for this bounded
analytic adapter rather than silently preserving circular metadata.

**Exact-fit support.** The first local analytic support is a closed disk eroded by
a disk and a filled rectangle eroded by a disk, at one depth, without holes. Use
the supplied analytic parameters in its local frame, not a GEOS erosion or fitted
polygon. Classify using authoritative parameters before rounded placement/unit
conversion; exact rational conversion (`127/5` mm/inch) preserves equalities when
combining units. For rectangle width `w`, height `h`, radius `r >= 0`, compare `w-2*r` and
`h-2*r` exactly for the supplied binary64 parameters (exact rational comparison
is sufficient). Negative extent is nominal empty; two positive extents give area;
one zero gives a closed segment; two zeros give a point. For circle radius `R`,
compare `R-r`: positive is a disk, zero is its center point, negative is empty.
Conversion to output coordinates still consumes the numerical error budget.

Never promote a nearly fitting case to exact fit using the boundary tolerance.
Uncertain dimensions/tool sizes that straddle equality give `unresolved`, with
the nominal result optionally attached for diagnosis. Exact equality is an ideal
geometric result for declared inputs, not proof of practical clearance or entry.
Rigid/reflected placement retains the analytic result and dimensions. General
collapsed feasible sets, and collapsed branches alongside positive-area erosion,
remain `unsupported` until a declared recovery algorithm is verified. A polygon
buffer alone cannot prove absence of these strata, even when it returns some area.
General polygon erosion may therefore be returned only as **nominal area erosion**,
not as a complete `FeasibleSet` or a proof of infeasibility. No tolerance-sized
polygon is substituted for a point or line.

**Result and failure semantics.** Successful results declare `nominal_geometry`
or a specifically established bounded evidence class; neither means executable
motion or machining acceptance. Empty is a successful mathematical value only
where the operation's declared semantics establish it. Other statuses are
`invalid_input` (with source diagnostics), `unsupported` (missing capability),
`unresolved` (budget, conditioning, topology ambiguity or work limit), and
`backend_failure` (caught backend exception or invalid/nonfinite result).
These statuses are distinct from a later planner's `infeasible`. Validate backend
outputs, reject unexpected dimensions, and return no accepted value on failure.
Operations are pure and atomic: errors cannot mutate inputs, a project, cached
stock or an earlier accepted result. No automatic retry with looser tolerances.

The first runtime slice should prove normalization, explicit union/difference,
nominal area erosion and the bounded analytic `FeasibleSet` end to end on detached
values. Keep conservative swept-stock and rest certification gated on their own
bounds, followed by motion/entry verification. This is more useful now than more
standalone kernel comparisons: the selected backend has evidence, while consumers
still need an owned boundary that prevents nominal geometry becoming a false
machining guarantee.

### First directional stock consumer - 2026-09-22

The [implemented analytic section contract](structure_spec.md#directional-analytic-stock-section-bounds)
now establishes the directional occupancy gate for one supplied horizontal disk
sweep in exact rectangular stock/target. The input uncertainty model, triangle-
inequality propagation, protected boundary check, reversed remaining-stock bounds
and rejection rules live there. This is conditional section evidence, not the
full rectangular feasible-center opening or an executed pocket operation.
Independent scalar area references and translated actual-sweep membership checks
are in `tests/test_stock.py`. General Boolean success remains nominal and cannot
feed this consumer. Finite supplied sweep composition now preserves exact union
membership and reversed stock subtraction, with conservative overlap-safe rational
grid area intervals and per-pass provenance. The implementation contract remains
in the specification. A separate exact rectangular target with one protected
rectangular island now gives required rest against the original target while the
same composition retains whole-stock state. Exact island/exterior containment and
independent residual membership are checked; supplied sweeps may cross the
artificial rest/cleared-space interface. The subsequent bounded section-motion
verifier checks caller-supplied horizontal cutting and explicit entry/connection
motion against original target protection and prior guaranteed free space, with
independent residual membership. It does not establish vertical access, stock at
other Z heights, generated paths or general target topology. Its contract remains
in the specification and verification evidence in the review.
Next priority stays in [PROGRESS](PROGRESS.md#next-detached-stockrest-increment).
