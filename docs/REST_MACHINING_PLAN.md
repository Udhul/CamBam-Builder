# Future rest machining and V-cutter paths

Status: **backlog, low priority; optional future capability, not foundation work**.
Requested 2026-09-08. Priority belongs only to [PROGRESS.md](PROGRESS.md#remaining-backlog-in-order).
This document owns the problem, proposed outcomes, technical reasoning and future
acceptance criteria. It does not claim implementation or authorize machine execution.
The request names three tiers but enumerates five outcomes; all five are retained.

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
Keep this export/attachment helper unavailable until lossless XYZ interchange
and target-version varying-Z motion have been verified. Never pass XYZ triples
through the current XY/bulge Pline interface as a temporary implementation.
Native V-carving systems combine boundary width, tool angle, variable depth and
optional flat-depth clearing, supporting this separation of tasks; see
[Vectric's V-Carve documentation](https://docs.vectric.com/docs/V12.0/VCarveDesktop/ENU/Help/page/single-page/index.html).

Acceptance: canonical shapes and holes pass independent XYZ sweep checks; path
depth/pass limits hold between vertices as well as at them. Two XML round trips
preserve XYZ, closure, identities, source order and MOP parameters. CamBam-generated
motion must match intended cutting paths and acceptable links before machining
acceptance is claimed. Use straight XYZ segments first; varying-Z arc semantics
need their own evidence.

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

Runtime architecture remains owned by [structure_spec.md](../structure_spec.md).
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
separate [shape parity plan](SHAPE_PARITY_PLAN.md), with higher priority in the
backlog. This plan consumes its public geometry and XML contracts; it does not
implement Region storage, adders, Z migration or schema handling. Pure planar
rest prototypes can use existing supported contours, but Region inputs and XYZ
path export must wait for the applicable parity acceptance.

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

This remains below the existing correctness, compatibility, packaging and MCP
backlog items. Promote only for a concrete rest-machining workflow with approved
input semantics and a bounded first outcome. Region/Z parity has its own higher
backlog priority and plan; it is an upstream dependency, not optional rest scope. A separately
reported data-loss bug can be prioritized independently on its own evidence.

Planning is complete when all five outcomes, repository gaps, source distinctions,
support requirements and executable acceptance directions are recorded and linked.
Future execution acceptance must demonstrate load, analyze, calculate and save
on a machine without CamBam installed. Manual CamBam compatibility checks remain
separate from this standalone execution requirement.
The next implementation priority remains MOP group-source compatibility. Do not
implement a full stock simulator, routing engine or CamBam integration during this
documentation task.
