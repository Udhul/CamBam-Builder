# Region and Z-coordinate shape parity

Status: **completed; automated verification/review and synthetic CamBam
display/property acceptance complete** (2026-09-09). Core feature parity is independent of
rest-machining implementation. The implemented API and schema contract now live
in [the specification](structure_spec.md#shape-elevation-and-region-contract);
[prepared acceptance](DEVELOPMENT.md#manual-shape-parity-acceptance) owns user checks.
Priority belongs to [PROGRESS.md](PROGRESS.md#remaining-backlog-in-order): after
core stability/design and the existing geometry/relationship correctness work,
before packaging/examples, MCP integration and optional rest machining.

The later bounded vertex-record refactor supersedes this plan's historical
parallel Pline/Points storage and tuple-compatibility assumptions. `Vertex`
records and unambiguous XY/XYZ tuple shorthand are authoritative in the
implemented specification; the planning evidence below remains historical.

## Objective and scope

Add Region shapes and Z-coordinate support across every currently supported
modern shape type, plus Region. These are useful CAD/file interchange capabilities
in their own right. Rest machining depends on them, but does not own their design
or implementation. The [rest-machining plan](REST_MACHINING_PLAN.md) remains a
separate downstream consumer.

Build on the existing entity, project-adder, XML encoder/reader and relationship
patterns. This should be a bounded extension rather than a new geometry framework
or CAM engine. The main compatibility points are the existing XY/bulge tuple
contract, shape-specific elevation semantics, typed Region XML and transforms;
settle those with focused fixtures rather than assuming a mechanical extra tuple
field is sufficient. No new dependency is selected for storage/interchange work.

Coverage means Pline, Circle, Rect, Arc, Points, Text and Region. It does not
implicitly add unrelated CamBam entities such as surfaces or solids. Legacy-package
changes require their own explicit scope. Runtime computation and load/save must
work without CamBam installed; CamBam is used separately for manual compatibility
acceptance.

### Repository evidence

Local inspection of the user-provided [Region example](../output/region_example.cb)
found a `CADFile` root named `region_example` with version `0.9.8.0`, a `layer` element under
`layers`, and an object marked `xsi:type=Region`. The Region holds `OuterCurve`
with direct `pts`, and `HoleCurves` with nested `Polyline` contours, rather than the modern writer's
simple primitive-tag dialect. It has one closed outer curve and two closed holes;
the outer curve uses bulges extensively. All sampled point Z values are zero.
There are no source MOPs. This proves a relevant Region example, not varying-Z
support, a complete schema, or rest-machining behavior. The private fixture was
read locally only and remains unchanged; retain reusable synthetic equivalents
in future tests rather than making the ignored user file a suite dependency.

On 2026-09-10 the user supplied separate CamBam-generated Pline and Region XML
with bulged segments whose endpoints have unequal Z, plus native viewport and MOP
behavior: CamBam displays the sloping curves, while Pocket and Profile derive cut
depth from their machining parameters rather than contour Z. This establishes
the storage/interchange behavior; it does not establish intermediate spatial-curve
parameterization or implement MOP toolpath calculations in this framework. A
synthetic equivalent of the Region XML is retained in the automated suite.

| Runtime owner | Historical planning gap and consequence |
| --- | --- |
| `cambam_reader.PRIMITIVE_TAG_TO_CLASS` and `_reconstruct_primitive` | No Region mapping; unsupported primitive tags are skipped. Add typed-object/layer schema handling for this fixture as well as contour parsing, or explicitly scope a tested adapter. Do not silently import an empty project as success. |
| Historical Pline storage, geometry query and XML reader/writer | The third tuple element was bulge; import dropped Z and export emitted zero. The implemented follow-up uses canonical `Vertex` records while retaining query outputs. Curved bounds were also repaired separately. |
| `CamBamProject.add_pline` and other primitive adders | No Region adder. Establish first-class Region identity with owned contours and MOP targeting; derived preview Plines must not silently replace holes with filled independent pockets. |


### Native example boundary discovered during implementation

The typed-object schema is implemented. Strict topology validation rejects the
private native example because its first hole contains interior crossings
between segments 2/4 and 3/4 (zero-based). Independent 2,048-chord-per-arc checks
confirmed both crossings, separate from the Region validator. The source was
read locally, remains unchanged and is not a reusable suite dependency. This is
a geometry rejection, not a skipped entity or an empty successful import.
Authored synthetic fixtures establish the supported schema with valid topology.
Preserving self-intersecting Regions for interchange would require an explicitly
different topology contract; do not claim the private sample itself is accepted.

## Shape coverage and compatibility contract

| Shape | Required Z behavior |
| --- | --- |
| Pline | Independent per-vertex Z and bulge; retain closure, order and curved-segment data |
| Points | Independent per-point Z, including mixed elevations |
| Circle | Center elevation with diameter preserved; initially an XY-plane circle at that elevation |
| Arc | Center/plane elevation with radius, start angle and extent preserved |
| Rect | Corner/plane elevation with dimensions preserved; Rect-to-Pline baking and export preserve elevation on every resulting vertex |
| Text | Preserve the supported `p1` anchor elevation and optional serialized `p2`. Missing `p1` means the origin; CamBam documents `P2` as currently unused and may omit or materialize it according to edit history. Retain it without assigning display semantics. |
| Region | Outer boundary and owned hole contours with their Z values and bulges; preserve contour topology and establish supported planarity constraints |

At planning time Circle/Arc/Rect/Points/Text geometry stored XY, their encoders synthesized
zero Z, and corresponding reader branches discarded elevation. Pline also discarded
Z while its third tuple value already means bulge. The relevant owners are the
shape classes in `cambam_entities.py` and `_reconstruct_primitive` in
`cambam_reader.py`. Coverage must include all rows, not just the XYZ Pline needed
by engraving.

Keep existing 2D call sites valid, with omitted Z defaulting to zero. The delivered
follow-up deliberately breaks the unreleased three-tuple input: it now means XYZ,
while nonzero bulge requires an explicit named `Vertex`. Four-tuples are rejected.
The implemented specification owns the exact conversion and serialization contract.
Existing geometry-query output shapes remain unchanged alongside explicit XYZ access.

Geometry elevation and transform elevation are distinct. Current
`cad_transformations.apply_transform` uses 3x3 XY affine matrices; CamBam matrix
serialization currently hardcodes zero Z translation. Extend or adapt the
transform boundary so imported supported Z offsets, world/local hierarchy and
baking agree. Existing XY transforms should leave Z unchanged unless a Z operation
is requested. Include parent Z offsets in tests: supporting point Z while dropping
matrix Z does not meet the supported round-trip contract.

Arbitrary 3D rotation, tilted analytic entities and a general solid-modeling engine
are not implied by adding elevation. During implementation classify incoming
matrix forms and shape planarity explicitly. Preserve supported geometry and Z
translation exactly; detect unsupported mixing of Z with XY or nonplanar analytic
forms and report the limitation before silently flattening or partially importing
them. The first XYZ slice is not complete parity until this boundary is documented
and all declared supported cases pass. The later native examples establish that
Pline and Region XML may retain varying-Z bulged segments. Current geometry
consumers interpret bulge in the XY projection and do not claim an intermediate
spatial-curve parameterization.

## Region ownership and XML support

Represent a Region as one registered Primitive with an outer contour and owned
hole contours. Its UUID, identifier, description, layer/group memberships and
MOP source identity belong to the Region. Contour points/segments are geometry,
not new project primitives with independent MOP targets. Define winding,
closure, hole containment and degenerate/touching-contour behavior explicitly.
Do not turn holes into filled independent pockets during conversion or export.

Extend the existing project creation and XML dispatch patterns. Support the
user's typed-object Region fixture, including its layer/container form, and
verify the output schema with CamBam rather than inventing a `<region>` spelling.
Normalize supported XML forms into one internal representation. Distinguish an
unsupported schema from a successful empty import. Keep the example private and
unchanged; create authored synthetic outer/hole/bulge examples for regressions.

Build geometry queries and transformations on the existing Primitive contract.
Use the shared curved-bounds correctness work for arc extrema; include bulged
Region contours in its integration tests. Region support does not require the
rest planner's Boolean kernel, offsets, stock model or cutter calculations.

## Bounded delivery sequence

1. Confirm shape and matrix XML semantics with minimal synthetic elevation
   fixtures. Set the additive Z API and old-input/query compatibility contract.
   Prove Pline/Points XYZ creation, export/import and transformation end to end.
2. Extend elevation through Circle, Arc, Rect and Text using the same established
   patterns. Cover Rect representation changes and parent/world poses. Keep
   conversion, state persistence and copy/transfer behavior consistent with the
   existing APIs; do not create competing relationship registries.
3. Add the Region entity, project adder, contour representation and XML adapters.
   Reuse the Z/vertex contract and shared bounds implementation. Exercise Region
   identity as a MOP source with holes and bulges through repeated round trips.
4. Complete cross-shape regression and prepared CamBam display/interchange
   acceptance, then publish the supported API/schema boundaries in the existing
   architecture owner and record completion in PROGRESS.md. Downstream rest work
   consumes those contracts without reopening storage design.

These are related increments in one parity outcome, not a requirement for a
large architectural rewrite. Reassess only if schema evidence or compatibility
failures show the existing patterns cannot support the intended behavior.

## Acceptance and stopping condition

- All seven shape types can be constructed with their supported Z coordinates,
  queried, saved and loaded without losing elevation, identity or geometry.
- Test positive, negative, zero and mixed per-point Z where supported; nonzero
  parent/local Z offsets; XY translation/rotation; baking; and Rect conversion.
  Verify bulge and Z independently with deliberately different values.
- Region tests cover one outer curve, multiple holes, curved segments, elevated
  planar contours, topology and MOP targeting. Invalid/unsupported topology and
  matrix forms produce explicit failures instead of silently losing geometry.
- Two XML round trips retain declared geometry, UUIDs, relationships and MOP
  references. Check the actual XML Z fields, not only the reconstructed object.
  Use a declared precision and absolute tolerance (initial synthetic targets:
  `1e-10` in memory, `1e-8` XML); include extrema beyond arc endpoints in bounds.
- Existing 2D API/regression results remain valid. Pickle and available copy/transfer
  operations retain new fields and project links; define defaults for supported
  older state data rather than claiming unrestricted pickle compatibility.
- Prepare synthetic CamBam A/B files with exact XYZ expectations. Inspect display
  and coordinate properties separately from automated XML checks. The initial
  planar Region example did not establish varying-Z acceptance; the user's later
  CamBam-generated Pline and Region examples establish that interchange case.
- Construction, transforms and load/save run without CamBam installed. No CAM
  engine, toolpath generation or rest-machining algorithm is required for closure.

Stop when every shape's declared Z contract and Region interchange are implemented,
verified and the required synthetic acceptance is recorded. Do not close this
item merely because XYZ Plines work. New entity families or unrestricted 3D
modeling need their own scope. Implementation, verification and the reported
CamBam Plus 1.0 acceptance now belong to the specification, review record and
runbook. The acceptance covers display/property interchange, not production
toolpaths.
