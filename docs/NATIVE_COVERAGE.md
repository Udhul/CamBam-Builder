# Native CamBam coverage

This is the maintained class-to-library/API/test/evidence map for the CamBam
Plus 1.0 baseline. It owns coverage classification and the evidence needed to
extend each boundary. Detailed behavior stays in the linked specification,
executable code and tests; current priority and implementation state stay in
[PROGRESS](PROGRESS.md#active-work-and-next-priority). Native interchange does
not imply independent CAM calculation, posted-motion replay or physical machining
acceptance.

## Reading the map

- **Author/edit** means the direct library can create, import, edit and export
  the stated modeled representation. It does not imply every native property or
  every target/MOP combination is modeled.
- **Preserved-only** means an imported payload survives under its stated editing
  limits, without a general authoring or semantic interpretation API.
- **Unsupported** means no supported import/author/edit/export representation.
  There is no opaque class-level preservation fallback.
- **Software** evidence checks library behavior. **Native interchange** evidence
  observes installed native serialization/API behavior or application loading,
  saving/displaying, with the actual level stated per claim. **Native post**
  evidence observes exact CamBam-generated output. These are separate levels;
  neither software round trips nor native XML serialization prove toolpath effects.

`CamBamProject` (also `CBProject`) owns entities and relationships.
[native/project.py](../cambam_builder/native/project.py) supplies the adders,
lookups, transforms, groups, hierarchy, cloning, copying and MOP targeting.
The canonical [reader](../cambam_builder/native/reader.py) and
[writer](../cambam_builder/native/writer.py) own XML interchange. Use
`read_cambam_bytes(..., strict=True)` for fail-closed import. Unsupported geometry
fails the whole import; unsupported MOPs fail strict import, while the legacy
`read_cambam_file` and non-strict path warn and skip those MOPs. Skipping is not
preservation. [test_strict_import.py](../tests/test_strict_import.py) protects
this distinction.

## Geometry classes and drawing commands

The inventory distinguishes a drawing command from a serialized entity class.
The [native Line/Spline evidence](REVIEW.md#nc01-a-native-geometry-coverage---2026-10-26)
establishes distinct native classes and XML forms. An application's Line drawing
command need not create the native `Line` class.

| Native class / XML or command | Direct-library representation and coverage | Contract and executable evidence | Native evidence |
| --- | --- | --- | --- |
| Polyline / `pline`, open or closed | **Author/edit:** `Pline`, `add_pline`; editable `Vertex` XYZ/bulge records. | [cad.py](../cambam_builder/native/cad.py); [shape contract](structure_spec.md#shape-elevation-and-region-contract); [shape parity](../tests/test_shape_parity.py), [elevation](../tests/test_shape_elevation.py). | [Shape interchange](REVIEW.md#region-and-shape-elevation-implementation); open/closed targets in [optimizer posts](REVIEW.md#native-optimiser-corpus-posted-output---2026-09-23). |
| PolyRectangle / `rect` | **Author/edit:** `Rect`, `add_rect`; corner, dimensions and elevation. Representation can become Pline when an edit cannot remain a Rect. | [cad.py](../cambam_builder/native/cad.py); [Rect representation](structure_spec.md#rect-baking-representation-contract); [shape parity](../tests/test_shape_parity.py). | Shape interchange and optimizer posts linked above. |
| Circle / `circle` | **Author/edit:** `Circle`, `add_circle`; center, diameter and elevation. | [cad.py](../cambam_builder/native/cad.py); [shape contract](structure_spec.md#shape-elevation-and-region-contract); [elevation](../tests/test_shape_elevation.py). | Shape interchange and optimizer posts linked above. |
| Arc / `arc` | **Author/edit:** `Arc`, `add_arc`; center, radius, start/extent angles and elevation. | [cad.py](../cambam_builder/native/cad.py); [curved bounds](structure_spec.md#curved-geometry-bounds-contract); [elevation](../tests/test_shape_elevation.py). | Shape interchange and optimizer posts linked above. |
| PointList / `points` | **Author/edit:** `Points`, `add_points`; XYZ vertices without bulge; owned Part nesting reference where selected. | [cad.py](../cambam_builder/native/cad.py); [interchange contract](structure_spec.md#native-interchange-integrity-cf01); [PointList nesting](../tests/test_pointlist_nesting.py). | Shape interchange; Drill target posts linked above. CF01 nesting is software evidence in [CF01 review](REVIEW.md#cf01-native-interchange-integrity---2026-10-26). |
| MText / `text` | **Author/edit:** `Text`, `add_text`; text, alignment anchor, font/height and elevation. Optional native `p2` is retained without assigning drawing semantics. | [cad.py](../cambam_builder/native/cad.py); [shape contract](structure_spec.md#shape-elevation-and-region-contract); [Text child order and optional-p2 tests](../tests/test_shape_parity.py). | Native-saved Text and display acceptance in shape interchange; combined Text/Arc Engrave posts linked above. |
| Region / `entity xsi:type="Region"` | **Author/edit:** `Region`, `add_region`; one owned outer Pline plus hole Plines and one registered target identity. | [region.py](../cambam_builder/native/region.py); [Region contract](structure_spec.md#shape-elevation-and-region-contract); [topology, parsing and edits](../tests/test_region.py). | Shape interchange and Region/Pocket optimizer posts linked above; invalid private Region sample remains explicitly unaccepted in the shape review. |
| Line / `line` | **Author/edit:** `Line`, `add_line(layer, points)`; two or more XYZ vertices representing connected open straight segments, without bulge. | [cad.py](../cambam_builder/native/cad.py); [Line/Spline contract](structure_spec.md#native-line-and-spline-interchange-nc01-a); [focused geometry tests](../tests/test_native_spline.py), [mixed interchange workflow](../tests/test_native_geometry_interchange.py). | [NC01-A installed serializer/API evidence](REVIEW.md#nc01-a-native-geometry-coverage---2026-10-26). No native posted-output claim. |
| Spline / `spline` | **Author/edit:** `Spline`, `add_spline(layer, control_points, knots, degree=...)`; complete control-point/knot/positive-weight representation. Flags, raw tangent fields and tolerance fields are retained without semantic interpretation; no inferred closure or curve evaluation. | [cad.py](../cambam_builder/native/cad.py); [Line/Spline contract](structure_spec.md#native-line-and-spline-interchange-nc01-a); [focused geometry tests](../tests/test_native_spline.py), [mixed interchange workflow](../tests/test_native_geometry_interchange.py). | [NC01-A installed serializer/API evidence](REVIEW.md#nc01-a-native-geometry-coverage---2026-10-26). No GUI display or posted-output claim. |
| Surface / `surface` | **Author/edit:** `Surface`, `add_surface`; indexed XYZ vertices and oriented triangle faces, DisplayEdges, identity and supported placement. No implicit solid/relief interpretation. | [NR03 contract](structure_spec.md#bounded-native-relief-interchange-nr03); [mesh tests](../tests/test_native_surface.py), [owned interchange](../tests/test_native_relief_interchange.py). | [NR03 installed serializer, transform and native-post evidence](REVIEW.md#nr03-native-relief-milestone---2026-10-26). |
| ScriptEntity / historical `script` inventory label | **Unsupported.** Distinct from literal Drill CustomScript text. | No native entity implementation or acceptance test. | Reopen with a deterministic self-contained script and explicit execution boundary. |
| CADBitmap / `bitmap` | **Author/edit:** `CADBitmap`, `add_bitmap`; source reference, lower-left XYZ, drawing dimensions, display/vectorizer state and supported placement. Image capture/packaging is explicit. | [Bitmap contract](structure_spec.md#native-bitmap-artwork-registration); [entity tests](../tests/test_native_bitmap.py), [dependency tests](../tests/test_bitmap_bundle.py), [consumer](../demos/native_bitmap_workflow.py). | [Installed placement/save/reopen and image dependency evidence](REVIEW.md#constructive-artwork-bundle---2026-10-26). No GUI/vectorization or machining claim. |

The shared [shape contract](structure_spec.md#shape-elevation-and-region-contract)
owns intrinsic Z, XY affine matrices, hierarchy/world pose, supported queries and
bakes. It explicitly excludes arbitrary 3D matrices, lossy curved/Text bakes and
invalid Region topology. A retained affine matrix is not evidence that every
derived geometric query can express its result. Geometry fields are modeled;
arbitrary unknown geometry children are not promised to survive reconstruction.

For Spline, missing/empty imported weights and incomplete or fit-only records are
unsupported; fresh `weights=None` authors explicit unit weights. Controls and
stored fit points follow supported XY affine transforms and Z translation; native
tangent fields remain unchanged. Defining-geometry queries do not evaluate the
curve. Curve bounds/center, implicit tessellation, arbitrary 3D transforms and
general spline CAM calculation remain unsupported. See the
[Line/Spline contract](structure_spec.md#native-line-and-spline-interchange-nc01-a)
for validation and explicit-pivot transform behavior. Extend these limits only
with a concrete consumer and evidence for the missing representation or geometric
operation; do not approximate an imported curve during interchange.

## Machining operation classes

The implemented classes live in [native/cam.py](../cambam_builder/native/cam.py).
Their field/state inventories are authoritative in the
[MOP parameter contract](structure_spec.md#mop-parameter-interchange-contract),
including applicability, units and fresh versus imported encodings. This map
does not duplicate that field inventory.

| Native MOP class / XML | Direct-library representation and coverage | Contract and executable evidence | Native evidence / extension requirement |
| --- | --- | --- | --- |
| Profile / `profile` | **Author/edit:** `ProfileMop`, `add_profile_mop`; modeled common/Profile fields, bounded lead and tab modes. **Preserved-only:** unmodeled imported fields/modes under the rules below. | [Profile/Pocket fields](structure_spec.md#profilepocket-subtype-and-nested-field-inventory); [parameters](../tests/test_mop_parameters.py), [identity and Manual tab references](../tests/test_mop_roundtrip.py); [Triangle/Square retained-stock consumer](structure_spec.md#bounded-native-triangle-tab-consumer). | [Optimizer posts](REVIEW.md#native-optimiser-corpus-posted-output---2026-09-23); [historical Manual tab evidence](REVIEW.md#manual-profile-tab-native-fixture--2026-09-27); [fresh paired native tab posts](REVIEW.md#native-triangle-holding-tab-consumer---2026-10-26). |
| Pocket / `pocket` | **Author/edit:** `PocketMop`, `add_pocket_mop`; modeled common/Pocket fields and bounded lead modes. **Preserved-only:** imported unmodeled content. | [Profile/Pocket fields](structure_spec.md#profilepocket-subtype-and-nested-field-inventory); [parameters](../tests/test_mop_parameters.py), [native edits](../tests/test_mop_native_edits.py). | Optimizer posts above include closed Pline and Region-with-island targets. |
| Engrave / `engrave` | **Author/edit:** `EngraveMop`, `add_engrave_mop`; modeled fields, including compatibility fields whose presence does not promise a toolpath effect. **Preserved-only:** imported unmodeled content. | [Engrave fields](structure_spec.md#engrave-subtype-inventory); [parameters](../tests/test_mop_parameters.py), [native edits](../tests/test_mop_native_edits.py). | Optimizer posts above and [state/effect evidence](REVIEW.md#native-default-value-and-omission-evidence---2026-09-20). |
| Drill / `drill` | **Author/edit:** `DrillMop`, `add_drill_mop`; CannedCycle, SpiralMill_CW, SpiralMill_CCW and nonempty CustomScript. **Preserved-only:** unsupported imported method records; they cannot be switched. | [Drill methods/fields](structure_spec.md#drill-method-and-subtype-inventory); [method switches, omission, preserved unknown modes](../tests/test_mop_parameters.py). | [SpiralMill acceptance](REVIEW.md#spiralmill-drill-mcp-authoring--2026-09-19), [state/effect evidence](REVIEW.md#native-default-value-and-omission-evidence---2026-09-20), optimizer raw G98/G81 observations. No general cycle-motion replay. |
| 3D Surface / `surface3d` | **Author/edit:** `Surface3DMop`, `add_surface3d_mop`; bounded Horizontal/Vertical ball Finishing. **Preserved-only:** unedited imported unsupported methods/state. | [NR03 fields and limits](structure_spec.md#bounded-native-relief-interchange-nr03); [state/edit tests](../tests/test_native_surface_mop.py), [source/post audit](../tests/test_native_relief.py). | [NR03 native state/save/post evidence](REVIEW.md#nr03-native-relief-milestone---2026-10-26). Broader strategies/tool profiles need another native consumer and independent posted-stock evidence. |
| NCFile / `ncfile` | **Author/edit:** `NCFileMop`, `add_ncfile_mop`; targetless source reference and plain DistanceMode/ArcCenterMode fields. **Preserved-only:** untouched imported unknown modes and empty source. Explicit dependency capture/portable packaging is separate from ordinary save. | [NC02 contract](structure_spec.md#external-program-native-interchange-nc02); [entity tests](../tests/test_native_ncfile.py), [dependency tests](../tests/test_ncfile_bundle.py), [mixed consumer](../demos/native_ncfile_workflow.py). | [NC02 installed native save/inclusion observations](REVIEW.md#nc02-external-program-consumer---2026-10-26). Tested optimizer modes do not rewrite supplied text; no backplot or motion-safety claim. |
| Lathe plugin / historical `mop xsi:type="MOPLathe"` label | **Unsupported.** Installed plugin inventory does not establish library support. | No native class implementation or acceptance test. | Reopen with a native-saved minimal turned profile, axis/post configuration and bounded interpretation. |
| Bas Relief / historical `MOPBasRelief` assembly type | **Unsupported historical type**, not asserted as a current 1.0 UI class. | No native class implementation or acceptance test. | The historical inventory records that [3D Profile replaced it](https://cambam.info/doc/1.0/cam/3d.html). Reopen for a concrete legacy file or current UI entry. |

### Preserved-only fields and document context

Preservation is scoped to supported containers, not a promise of lossless whole
document round trips. Tests distinguish untouched imported templates from edits:

| Payload | Supported boundary and test owner |
| --- | --- |
| Supported MOP native template | Unknown children, parameter attributes and untouched Default/Value cached state survive. Modeled edits update their fields; modeled mode changes reconcile their applicable dependents. [test_mop_parameters.py](../tests/test_mop_parameters.py). |
| Unmodeled lead/lead-out and tab data | Untouched imported subtrees remain. Fresh lead authoring supports None/Spiral; unsupported imported lead modes cannot be switched. Manual tab authoring and reference remapping have the bounded [Profile contract](structure_spec.md#profilepocket-subtype-and-nested-field-inventory), not universal tab editing. [test_mop_roundtrip.py](../tests/test_mop_roundtrip.py). |
| Unknown Drill method | Original method and its payload survive; unsupported method changes reject. No semantic evaluation. [test_mop_parameters.py](../tests/test_mop_parameters.py). |
| MachiningOptions, Part machining/style context | Retained imported context supports file-local inheritance. Referenced external style/tool libraries are not embedded or evaluated. [test_mop_context.py](../tests/test_mop_context.py). |
| Tags | Primitive/MOP operator notes coexist with framework metadata; imported root/layer/Part Tag paths retain their bounded context. [CF01 contract](structure_spec.md#native-interchange-integrity-cf01), [test_native_tags.py](../tests/test_native_tags.py). |
| Unknown classes, arbitrary root/geometry XML and references inside opaque extensions | **Unsupported preservation guarantee.** Strict class admission and known reference remapping do not interpret arbitrary extension references. A new preservation capability needs explicit source examples and an identity/dependency contract. |

## Shared integration and evidence owners

These tests cover behavior shared by the modeled classes. Class-specific rows
identify additional tests; this list does not claim an exhaustive cross-product
of geometry, transforms and machining parameters.

| Concern | API/contract owner | Executable evidence |
| --- | --- | --- |
| Units and numeric preservation | [CF01](structure_spec.md#native-interchange-integrity-cf01) | [test_native_units.py](../tests/test_native_units.py), [test_native_interchange.py](../tests/test_native_interchange.py) |
| Identity, layer, group, parent and world pose | [Parent identity](structure_spec.md#xml-parent-identity-and-world-pose-contract), [project.py](../cambam_builder/native/project.py) | [test_parent_roundtrip.py](../tests/test_parent_roundtrip.py), [test_shape_parity.py](../tests/test_shape_parity.py) |
| Clone, detached copy/transfer and included MOP references | [Copy/transfer](structure_spec.md#copy-and-transfer-contract) | [test_copy_transfer.py](../tests/test_copy_transfer.py), [test_shape_parity.py](../tests/test_shape_parity.py) |
| MOP targets/order/identity | [Target ownership](structure_spec.md#mop-target-ownership-contract), [MOP identity](structure_spec.md#xml-mop-identity-contract) | [test_mop_group_sources.py](../tests/test_mop_group_sources.py), [test_mop_roundtrip.py](../tests/test_mop_roundtrip.py) |
| Atomic import/export failures | [Export failure](structure_spec.md#export-failure-and-state-saving-contract) | [test_strict_import.py](../tests/test_strict_import.py), [test_export_failures.py](../tests/test_export_failures.py) |
| Hash-bound native optimizer observations | [Corpus boundary](structure_spec.md#native-optimiser-mapping-corpus-boundary) | [test_optimizer_corpus.py](../tests/test_optimizer_corpus.py); [dated native observations](REVIEW.md#native-optimiser-corpus-posted-output---2026-09-23) |

The [historical corpus preparation](REST_MACHINING_PLAN.md#inventoried-baseline-and-prepared-slice-2026-09-23)
retains the original observation scope. It is not the current coverage owner.
When extending a row, add a source-backed representation, direct-library
contract and fault-sensitive regression before promoting its classification;
record any new native observation in REVIEW with its actual acceptance scope.
