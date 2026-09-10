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
packages; `inactive/` and demos are outside that runtime package list.

| Owner | Implemented responsibility | Start here when changing |
| --- | --- | --- |
| `cambam_builder/cambam_project.py` | UUID entity registries, identifier lookup, ordered layers/parts/MOPs, relationship updates, transform orchestration and persistence | Public creation/query/mutation APIs and relationship invariants |
| `cambam_builder/cambam_transfer.py` | Transactional primitive-tree copy/transfer staging, identity mapping, collision validation and relationship publication | Copy/transfer semantics and atomic registry updates; inspect project wrappers and tests |
| `cambam_builder/cambam_entities.py` | Entity dataclasses, primitive geometry/bounds, local effective matrices, parent-composed world transforms and entity XML encoding | Geometry or entity fields; inspect reader/writer callers for I/O changes |
| `cambam_builder/region.py` | Owned Region contours, planar curved topology validation and typed Region XML | Region geometry and interchange; project and reader use this owner |
| `cambam_builder/cad_transformations.py` | NumPy matrix construction, composition, decomposition and XML matrix conversion | Numerical conventions; inspect entity and project callers together |
| `cambam_builder/cambam_writer.py` | XML ID assignment and layer/part traversal; delegates individual encoding to entities | Output structure and reference resolution |
| `cambam_builder/cambam_reader.py` | XML parsing, entity reconstruction, ID mapping and deferred parent/MOP linking | Import defaults, malformed data and round-trip reconstruction |
| `cambam_builder/__init__.py` | Public alias and version | Import surface and version metadata |
| `cambam_builder/mcp_adapter/` | Optional local stdio launcher, SDK protocol boundary, volatile documents, retry ledger, schema validation and workspace I/O | [MCP contract](MCP_CONTRACT.md); `server.py` owns wire behavior, `service.py` owns application state, `paths.py` owns filesystem policy |
| `cambam_builder/cad_common.py` | Logging configuration with placeholder type/constant sections | Check callers before treating it as an established shared utility layer |

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
4. `save_state()` / `load_state()` use Python pickle, restoring primitive project
   links after loading. This is distinct from XML interchange and is not a promised
   version-stable migration format. Trust requirements live in the runbook.

The reader's `PRIMITIVE_TAG_TO_CLASS` and `MOP_TAG_TO_CLASS` are the executable
supported-tag inventory, not a claim of complete CamBam coverage. Consult those
maps and corresponding entity encoders before adding a type.

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
Current-version pickle round trips retain vertex records and restore project
links. Old Pline/Points pickle layouts with parallel coordinate/elevation storage
are not migrated. Text `xml_p2` defaults to absent in older state; this is not a
general versioned pickle migration.

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
successful calls return `None`. Pickle writing remains direct/non-atomic, and
`load_state()` behavior and trusted-input requirements are unchanged.

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

The external compatibility boundary is CamBam `.cb` interchange: files produced
by the framework and files created or modified directly in CamBam. Validate
supported geometry, transforms, operation targets/order and machining parameters
against that boundary, including CamBam-authored input without framework Tags.
Framework metadata may supplement native data but must not silently override
native edits to geometry or operation targets. Missing/malformed metadata and
unsupported content need explicit handling; full format coverage is not yet
established. Local self-round trips alone do not prove CamBam interoperability.

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

New MOPs retain the existing constructor/encoder authoring defaults and convenience
fallbacks, including derived depth increment/feedrate when unspecified. These are
not CAM-style evaluation. The scalar state setter also works for new MOPs and
allows explicit Default output. An imported untouched MOP never recomputes those
authoring fallbacks or fills in absent properties.

Imported global MachiningOptions and unmodeled part machining settings are
retained, including Style/StyleLibrary. Part ToolDiameter retains native state/text
while its modeled value is unchanged; changing the value exports the edit.
This preserves inheritance context within the file, not the referenced external
style libraries. Stock/origin still follow the existing part model.

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
- Supports serialization and reconstruction of a complete project via XML (which includes relationship metadata) and via pickle.

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

### 4.2 Pickle Serialization

- **Native Serialization:**  
  - The project object (including its relationship registries) is serialized using pickle.
  - Custom `__getstate__` and `__setstate__` methods remove transient or non-serializable attributes (such as weak references).

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
  - The framework supports reconstructing a project from XML (and optionally from pickle), ensuring that all relationships are correctly reassembled.

- **Inter-Project Transferability:**  
  - Copy and transfer methods operate on the centralized registries, allowing entire linked trees of primitives to be transferred without duplicating relationship data.
  - UUIDs are preserved to maintain existing links; if a conflict occurs in the target project, the primitive is updated accordingly.
  - The relative structure is preserved and reflected in the XML output.

This design minimizes duplication by centralizing all relationship data within the project object’s registries, ensuring that the management, propagation, and serialization of entity relationships remain robust and maintainable.
