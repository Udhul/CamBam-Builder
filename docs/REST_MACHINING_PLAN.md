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

Still settle the first acceptance target (outline, floor/through
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
artificial rest/cleared-space interface. This does not add connections, entry,
generated paths or general target topology.
Next priority stays in [PROGRESS](PROGRESS.md#next-detached-stockrest-increment).
