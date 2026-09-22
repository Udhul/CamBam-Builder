# Future rest machining and V-cutter paths

Status: **active design refinement, 2026-09-22; implementation not started**.
Requested 2026-09-08 and expanded 2026-09-22. Priority belongs only to [PROGRESS.md](PROGRESS.md#remaining-backlog-in-order).
This document owns the problem, proposed outcomes, technical reasoning and future
acceptance criteria. It does not claim implementation or authorize machine execution.
The five original outcomes below are retained. The current design proposal and
unresolved product decisions are in [Design refinement](#design-refinement---2026-09-22).

## Programmatic execution requirement

The framework must calculate geometry and paths programmatically and load/save
CamBam files without running CamBam. No headless CamBam service or callable CAM
engine is available to this workflow. Do not design around GUI automation,
CamBam plugins/scripts, or manual toolpath extraction as execution dependencies.
Implement the needed geometry, cutter and supported MOP path algorithms in the
framework, using an appropriate programmatic geometry library where justified.
CamBam is the file consumer and an optional manual validation environment.

The useful proxy here is a **file representation**: calculate XYZ tool-center
paths ourselves, save them as shapes and attach an Engrave MOP. It is not a way
to invoke CamBam's calculations. Manually exported CamBam paths may serve as
optional comparison fixtures, but normal operation and automated tests must not
require them. CamBam's scripting examples document possible file/motion behavior;
they are not a runtime interface for this framework.

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

### Intended product and decisions awaiting user input

Build a deterministic, standalone engine that combines caller-supplied tools to
remove an explicitly defined target volume while preserving protected material.
Expose region-only analysis, path-only planning, verified removal and optional
CamBam project attachment as independently useful operations. All calculations
run without CamBam installed. This is a proposed contract, not an implemented API.

Two questions have been presented to the user; their answers are pending:

1. Should the product support both decorative V-shaped carving and vertical-wall
   pocket/through-cut targets as explicit modes? Recommended: yes. A pointed tip
   alone does not make a sharp vertical corner column removable by a widening cone.
2. Should the first planner compare a supplied tool set, or also discover tools
   in a catalog? Recommended: supplied tools, composing existing caller-owned
   recommendation profiles; a catalog is a separate maintenance commitment.

After these answers, settle the first acceptance target (outline, floor/through
depth, wall shape, tools), allowed residual boundary/thickness/volume tolerances,
and whether the first workflow requires native Profile/Pocket execution or accepts
framework-generated explicit paths. No numerical production defaults or implicit
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

No dependency is selected. Compare a small backend shortlist on analytic offset
accuracy, holes/near-tangencies, deterministic topology, supported Python 3.9-3.13
and Windows installation, license/distribution obligations and measured performance.
Start with planar Booleans/offsets; add a medial-axis backend only if the first
V-carve slice demonstrates a need. A package's existence does not justify a new
dependency or replacing the already accepted CAD representation.

### First useful increment and acceptance

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

This round stops at a reviewed proposal and recorded open decisions. No machine
validation is needed for documentation. Before later manual acceptance, generate
and inspect A/B files and exact expected motions as required by the development
runbook. Continue this design conversation while product decisions are pending;
a fresh implementation session becomes appropriate after their answers and the
first fixture contract are recorded here.
