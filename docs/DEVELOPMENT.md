# Development runbook

## CamBam validation baseline

User-performed CamBam checks use CamBam Plus 1.0 unless explicitly stated
otherwise: `CamBam.CAD` 1.0.7364.41819, `CamBam` 1.0.7364.41821, build
2020-02-29 23:13:58. CamBam 1.0 writes `Version="0.9.8.0"` into `.cb` files;
that legacy XML marker does not mean the file was created by CamBam 0.9.8.
Do not ask for the version again for this environment. Record a different version
only if the user explicitly reports one.

## Environment and setup

The declared and verified toolchain is Python >=3.9 and setuptools/wheel
(`pyproject.toml`). NumPy is declared directly in project metadata. `uv.lock` is
intentionally ignored: supported-version checks resolve the currently compatible
NumPy release independently on each interpreter. No `setup.py`, separate
requirements file or CI runner is declared.

Create or update the repository-managed environment from the root with:

```powershell
uv sync --python 3.13
```

This creates `.venv` and installs the project plus a currently compatible NumPy
version. The command may require dependency downloads. After setup,
explicitly select its interpreter for every command; activation is optional. If
the IDE uses another project environment, set `$ProjectPython` to that interpreter
instead. Confirm it before running checks:

```powershell
$ProjectPython = '.\.venv\Scripts\python.exe'
& $ProjectPython -c "import sys; print(sys.executable); print(sys.version)"
```

Without a project environment, the available interpreter can perform baseline
checks if dependencies are already present; record that limitation. The initial
review did this without installing or changing dependencies.

## Verification entry points

The optional MCP document foundation is described in the
[MCP contract](MCP_CONTRACT.md). See [local MCP setup](#local-mcp-setup-and-verification)
below for installation and protocol acceptance. The base library needs no MCP SDK.

Run from the repository root, using the interpreter selected above:

```powershell
& $ProjectPython -m compileall -q cambam_builder legacy_cambam_builder
& $ProjectPython -c "from cambam_builder import CBProject; p = CBProject('smoke'); assert p.project_name == 'smoke'; print('import/construct OK')"
git diff --check
```

These are syntax/import checks. Run the authored synthetic XML regressions with:

```powershell
& $ProjectPython -m unittest discover -s tests -v
```

Copy/transfer relationship checks use the focused regression command:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_copy_transfer.py -v
```

The minimal in-memory example below copies a primitive subtree, then transfers
the same source subtree to a separate destination. Both operations return a
source-to-destination UUID mapping; the copy leaves `source` available for the
subsequent transfer.

```python
from cambam_builder import CBProject

source = CBProject("source")
root = source.add_rect("Geometry", identifier="root", width=20, height=10)
source.add_circle("Geometry", (5, 5), 2, identifier="child", parent=root)

copied_target = CBProject("copied")
copy_map = source.copy_primitive_tree(root, copied_target)

transferred_target = CBProject("transferred")
transfer_map = source.transfer_primitive_tree(root, transferred_target)
assert copy_map[root.internal_id] == root.internal_id
assert transfer_map[root.internal_id] == root.internal_id
```

See the [copy and transfer contract](structure_spec.md#copy-and-transfer-contract)
for collision, relationship, world-pose and atomicity rules. Destination global
style context remains authoritative; XML round trips represent group targets
as the currently resolved primitive snapshot.

Packaging installation and supported Python versions require separate validation;
a local import does not prove them.

## Packaging and supported Python validation

Verified 2026-09-10 using `uv 0.10.2`. The minimum is Python 3.9: Python 3.8
was not available on the validation machine, and the separately shipped legacy
package evaluates PEP 585 built-in generic annotations that require Python 3.9.
The wheel metadata declares version 0.1.0, `Requires-Python: >=3.9`, and
`Requires-Dist: numpy>=1.23.5`. The wheel and sdist contain both `cambam_builder` and
`legacy_cambam_builder`; all current modern modules are present.

The wheel was installed into a separate environment per interpreter. From a
temporary working directory outside the repository, each environment imported
both package roots, verified installed version/dependency metadata and source-path
exclusion, constructed a modern and legacy project, performed a modern Rect XML
write/read cycle, and passed the full 142-test suite:

| Python | NumPy resolved by clean wheel install | Result |
| --- | --- | --- |
| 3.9.0 | 2.0.2 | pass |
| 3.10.9 | 2.2.6 | pass |
| 3.11.0 | 2.4.6 | pass |
| 3.12.10 | 2.5.3 | pass |
| 3.13.9 | 2.5.3 | pass |

The sdist was independently installed under Python 3.9 with NumPy 2.0.2 and
passed the same imports, installed-path assertion and all 142 tests. Build
artifacts inspected locally are under
`output/packaging-validation-20260910-c/`; that ignored directory is supporting
evidence, not a release location. `legacy_cambam_builder.cambam_builder_cli` is
a dormant historical module with no declared entry point and is outside the
supported import surface; no CLI or publishing behavior was added.

Reproduce the development environment with `uv sync`. Because the lockfile is
local and ignored, dependency versions may advance over time. For a fresh
artifact check, build with `uv build --out-dir <unique-output-directory>`, install
the resulting wheel or sdist into a clean `uv venv --python <version>`, copy the
tests to a directory outside the repository, and run discovery there with that
environment's interpreter. This prevents the repository root from satisfying
imports accidentally.

### Required checks by change

| Change | Minimum evidence before technical closure |
| --- | --- |
| Documentation only | Review changed local links/headings, factual claims against their owner and `git diff --check`; run example code if changed |
| Runtime defect | Assertion-based regression that fails before the fix and passes afterward, focused test command, syntax/import checks |
| Shared relationship/transform/XML contract | Relevant regression suite plus an end-to-end synthetic round trip checking counts, references, world geometry and affected parameters |
| Packaging/dependencies | Isolated install/build and import from outside the source tree; declared Python compatibility checks appropriate to the change |

The suite uses standard-library `unittest` and the runtime NumPy dependency.
Test observable contracts, including invalid inputs relevant to the fix, rather
than mirroring internals.

Rect baking has a focused regression command:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_rect_bake_defect.py -v
```

The tests assert exact outline preservation for rotated and sheared 4x2 Rects,
closed-Pline conversion, identity matrices, metadata and two XML round trips.
The repair scope and representation policy are recorded in the
[Rect investigation and repair criteria](REVIEW.md#rect-baking-loss-investigation).

Curved Arc and bulged-Pline bounds have a focused regression command:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_curved_bounds.py -v
```

The tests cover directed wrap-around and full sweeps, positive/negative and
closing bulges, exact affine extrema under rigid, reflected, nonuniform, sheared
and singular transforms, numeric tolerances, large finite values and invalid
input boundaries. Bounds are runtime calculations and are not serialized, so
these analytic checks provide the relevant acceptance evidence; no separate
CamBam display check is required for this slice.

Canonical Pline/Points vertex records have a focused regression command:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_vertex_records.py -v
```

It checks `Vertex` defaults and keyword-only bulge, XY/XYZ tuple shorthand,
four-tuple and malformed-input rejection, Points bulge rejection, collection
insertion/reordering, current pickle storage and two XML round trips. The shape
elevation and Region suites additionally cover varying-Z bulged Pline segments
and a native-derived varying-Z Region with a hole through two XML round trips.
The shape-parity fixture uses mixed-Z/bulge geometry while retaining transforms,
holes and the Region's Profile MOP reference across full project round trips. The
accepted CamBam A/B/C files must remain unchanged; compare a new
framework round trip against them only in a unique ignored output directory.

The MOP ownership and interchange slice has focused automated and manual checks:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_mop_group_sources.py -v
& $ProjectPython output\mop-core-validation-1roowlbtpza\generate.py
```

The tests cover project-owned target selections, group membership and repeated
round trips. The generator creates a synthetic metadata-free A input (no MOP
identity Tags; this is not CamBam-authored evidence), verifies A/B entity counts,
targets, order, enabled states and machining values, and prepares the native-edit
leg. After a user saves C from CamBam and reports its path, run the generator
with `--native-edited <reported C path>`. It writes D and its inspection JSON
only under `output/mop-core-validation-1roowlbtpza/`, refuses a source/output
collision, and checks C versus D semantically. Production toolpaths remain
outside the automated evidence.

Keep reusable synthetic fixtures and expected results with authored tests.
Use a unique task directory under ignored `output/` for disposable diagnostics,
generated XML and verbose logs; do not overwrite previous runs. Durable evidence
must include reproduction inputs/steps and commands in the review or tests so a
fresh checkout can reproduce it without ignored files. Local logs are supplementary.
Remove only temporary artifacts created by that task when authorized; never infer
that an ignored directory is safe to clear. Inspect each exit status separately:
PowerShell can continue after a native command fails.

`demos/test.py` is an interactive example: its main block expects an input file
that is not supplied by the repository, and logs/returns on some failures. It is
not a pass/fail check. Select a relevant demo explicitly and inspect its output.
Generated files belong under `output/`; avoid overwriting existing examples or
user artifacts. Never load an untrusted pickle: `load_state` uses pickle loading.

For XML changes, compare entity counts, IDs, relationships, geometry and machining
parameters after writing/reading synthetic fixtures. Opening the result in CamBam
and checking geometry/toolpaths requires user/domain validation before production
use. Keep private CAD inputs and generated reports local.

## Manual Rect-bake acceptance

**Accepted by the user on 2026-09-08:** all conditions met, full outline match
and identity transforms. The CamBam Plus 1.0 baseline above applies. The retained criteria
below document the accepted synthetic case; do not repeat without a relevant
change. See [acceptance evidence](REVIEW.md#rect-baking-display-acceptance).

The prepared local fixtures are [A_reference.cb](../output/rect-bake-validation-20260908-a/A_reference.cb)
and [B_baked.cb](../output/rect-bake-validation-20260908-a/B_baked.cb). The generator
at [generate.py](../output/rect-bake-validation-20260908-a/generate.py) creates A as
explicit closed Plines and creates B from two 4x2 Rects, then performs a full bake
and two XML round trips. It also includes a transformed Rect parent and a closed
child outline. Its command is:

```powershell
& $ProjectPython output\rect-bake-validation-20260908-a\generate.py
```

In CamBam, open A and B separately in the top/XY view and zoom to the whole drawing.
Both files must show three closed outlines on the two named layers. Compare each
outline by its object name; every listed vertex must match within 0.01 drawing units
(Z=0):

| Object | Expected vertices in order (X, Y) |
| --- | --- |
| rotated-rect-parent | (10, 0), (12.828427, 2.828427), (11.414214, 4.242641), (8.585786, 1.414214) |
| rotated-child | (13.535534, 3.535534), (14.242641, 4.242641), (13.535534, 4.949747), (12.828427, 4.242641) |
| sheared-rect | (20, 12), (24, 12), (26, 14), (22, 14) |

Pass requires matching outlines without an axis-aligned bounding-box expansion
and identity transforms after baking. The generator checks zero bulges, metadata
and the `rotated-child` parent link automatically. Report CamBam version, A/B
pass or fail, and any displayed outline, vertex or matrix difference.

## Manual parent-transform acceptance

The parent A/B case was accepted by the user on 2026-09-07 (CamBam version
not supplied). The one-off generator was retired from `demos/` to the local
`output/parent-validation-kjytha6k/` directory alongside the accepted files.
No regeneration command depends on an ignored script. Reusable automated parent
regressions remain under `tests/`; the original helper is also in git history.

The accepted files are `A_reference.cb` (explicit world coordinates, identity
matrices) and `B_parent_roundtrip.cb` (parent transforms, rejected cycles and two
XML round trips). The criteria below document that acceptance; do not ask the
user to repeat it without a relevant behavior change.

In CamBam:

1. Open A and B separately. Both must load without errors.
2. Use top/XY view and zoom to the whole drawing in each. Each must contain exactly
   three open, straight polylines, one per `RootLayer`, `ChildLayer`, `GrandchildLayer`.
3. Compare placement/orientation and check world endpoints below using the drawing
   coordinates or measurement tools (not untransformed local point properties).
   Pass tolerance: 0.01 drawing units; all geometry is at Z=0.

| Layer | World endpoints (X, Y) | Length |
| --- | --- | --- |
| RootLayer | (20, 10) to (20, 20) | 10 |
| ChildLayer | (20, 25) to (15, 25) | 5 |
| GrandchildLayer | (10, 25) to (10, 30) | 5 |

Pass only if A and B both match all counts/endpoints, with no additional geometry,
displacement, scaling or rotation. Report CamBam version, A/B pass or fail, and
any differing layer/observed endpoint. This accepts the synthetic display/XML
slice only; no MOPs or toolpaths are included. Parent-cycle rejection and registry
integrity are automated API checks and need no manual reproduction.

## Manual full-bake acceptance

Accepted by the user: endpoints, identity matrices and layer differences confirmed.
The retained criteria below do not require repetition for unchanged behavior.

Prepared local files: `output/full-bake-validation-n132g8ro/A_reference.cb` and
`B_baked.cb` in that same directory. The one-off `generate.py` lives alongside
them, outside the git tree; reusable synthetic fixtures/tests live in
`tests/test_full_bake.py`. A uses explicit world coordinates; B first bakes the
root and child nonrecursively, then bakes the whole hierarchy. All B matrices
must be identity. This case tests baked geometry, beyond the accepted parent
matrix display case above.

Open A and B in CamBam, use top/XY view and zoom to the whole drawing. Both must
load without errors and contain exactly three open straight polylines at Z=0.
A uses separate named layers; B uses one `Geometry` layer. Check endpoints:

| Reference name | World endpoints (X, Y) in both files |
| --- | --- |
| root | (16, 13) to (22, 22) |
| child | (16, 37) to (10, 41.5) |
| leaf | (32, 46) to (26, 59.5) |

Pass tolerance is 0.01 drawing units. Pass requires matching placement and counts
with no extra geometry, and identity Transform properties on every B polyline
(resetting to identity must not move them). Report CamBam version, pass/fail and
any differing endpoint or matrix. No toolpath validation is needed for this case.

## Manual global-transform acceptance

Accepted by the user: A/B placement and endpoints match; B retains transforms;
child and leaf move (+5,-3) from Before as specified. The CamBam Plus 1.0 baseline
above applies. These retained criteria need no repetition for unchanged behavior.

Prepared directory: `output/global-transform-validation-6_s8bvg_/`. Its one-off
`generate.py` is ignored; regression tests live in `tests/test_global_transforms.py`.

Open `Before.cb`, `A_reference.cb`, and `B_global_result.cb` in top/XY view.
Before is a transformed hierarchy. B is that hierarchy after calling
`translate_primitive("child", 5, -3)`; A independently stores expected final points
with identity matrices. B retains transforms, unlike full-bake validation.

All files must load without errors and contain three open straight polylines at
Z=0, one on each named layer. Check these world endpoints (tolerance 0.01 units):

| Layer | Before | Both A and B |
| --- | --- | --- |
| rootLayer | (16, 13) to (22, 22) | (16, 13) to (22, 22) |
| childLayer | (16, 37) to (10, 41.5) | (21, 34) to (15, 38.5) |
| leafLayer | (32, 46) to (26, 59.5) | (37, 43) to (31, 56.5) |

Pass requires a stationary root and both descendants moved exactly +5 X / -3 Y,
with no extra geometry; A and B must coincide. Inspect world coordinates rather
than B's local point properties. Report CamBam version, pass/fail and the layer
and observed coordinates for any mismatch. No toolpath checks are needed.

## Manual MOP identity acceptance

Accepted by the user on 2026-09-07: all described A/B load, operation, target and
property outcomes were confirmed. The CamBam Plus 1.0 baseline above applies. The retained
criteria need no repetition unless relevant behavior changes.
Prepared and XML-inspected files in `output/mop-validation-kpk5hxop/`:
[A reference](../output/mop-validation-kpk5hxop/A_reference.cb) omits MOP Tags;
[B result](../output/mop-validation-kpk5hxop/B_roundtrip.cb) retains them after two
library round trips. The ignored generator can be run from the repository root:

```powershell
& $ProjectPython -c "import runpy; runpy.run_path('output/mop-validation-kpk5hxop/generate.py', run_name='__main__')"
```

Open A and B separately. Pass requires no load errors and exactly two enabled
Profile operations under Part1, both named `Duplicate`, in this order:

| Operation | Target geometry | TargetDepth | CutFeedrate |
| --- | --- | --- | --- |
| First | Left square, X=0..10, Y=0..10 | -1 | 300 |
| Second | Right square, X=20..30, Y=0..10 | -2 | 600 |

Both use Outside, tool diameter 3, depth increment 0.5 and spindle speed 12000.
Check properties and highlight each operation's referenced primitive. Tolerance:
0.01 drawing units for geometry/depth/diameter; feed and spindle match exactly.
Primitive XML IDs are assigned by UUID and need not be 1 then 2.
Report CamBam version, A/B pass or fail and any differing operation/property.
No machine execution is needed; this accepts load/display/properties only.
UUID/identifier registry behavior is automated and needs no manual reproduction.

## Manual MOP core-model and CamBam interchange acceptance

The user accepted the prepared A/B CamBam display/property case on 2026-09-08
and the final native C/D case on 2026-09-08 in the CamBam 1.0 baseline above.
This accepts supported display/property interchange, not production toolpaths or
complete native interoperability. The disposable fixture
directory is [mop-core-validation-1roowlbtpza](../output/mop-core-validation-1roowlbtpza/).
Regenerate its files from the repository root with:

```powershell
& $ProjectPython output\mop-core-validation-1roowlbtpza\generate.py
```

The generator writes [A_reference.cb](../output/mop-core-validation-1roowlbtpza/A_reference.cb),
[B_roundtrip.cb](../output/mop-core-validation-1roowlbtpza/B_roundtrip.cb) after two
library XML round trips, the intermediate `B_roundtrip_1.cb`, and
[manifest.json](../output/mop-core-validation-1roowlbtpza/manifest.json). The A file
has native primitive references but no framework MOP identity Tags. It is a
synthetic metadata-free input, not CamBam-authored evidence. The XML inspection
must show one layer, one part, four
primitives and four enabled operations in this order: Profile, Pocket, Engrave,
Drill. Their targets and geometry are:

| Operation | Target | Geometry | TargetDepth / CutFeedrate / ToolDiameter |
| --- | --- | --- | --- |
| Profile | `profile-square` | square X=0..10, Y=0..10 | -1 / 300 / 3 |
| Pocket | `pocket-square` | square X=20..30, Y=0..10 | -1.5 / 350 / 3 |
| Engrave | `engrave-outline` | triangle (40,0), (50,0), (45,8) | -0.4 / 250 / 1 |
| Drill | `drill-points` | points (65,2), (72,8) | -2 / 200 / 2 |

Open A and B separately in CamBam's top/XY view, zoom to drawing extents, and
confirm the four shapes and operation properties load without errors. Highlight
each operation and confirm its primitive reference matches the table; geometry,
depth and diameter tolerance is 0.01 drawing units, while feedrates must match
exactly. `Drill/CustomScript` is the Default-state example; the listed common
machining values are Value-state examples. This checks display and interchange,
not generated toolpaths or machine output.

For the native-edit leg, open B and make the following exact edits. Change the
existing Profile operation's primitive reference to `pocket-square` using
CamBam's Primitive IDs/target editor, set its CutFeedrate to 450, and set its
ClearancePlane property to Default. Reorder the existing operations to Drill,
Engrave, Pocket, Profile. Create one new native Profile operation targeting
`profile-square` and leave it last. All five operations must be enabled. Save
the result as `C_native_edited.cb` wherever convenient and report that saved
path; the agent runs the comparison after receiving it. The new operation may
use CamBam's generated display name.

The native edit was completed in
[C_native_edited.cb](../output/mop-core-validation-1roowlbtpza/C_native_edited.cb).
The comparison command was run successfully (exit 0):

```powershell
& $ProjectPython output\mop-core-validation-1roowlbtpza\generate.py `
  --native-edited <reported-C-path>
```

It wrote [C_framework_roundtrip.cb](../output/mop-core-validation-1roowlbtpza/C_framework_roundtrip.cb)
and its inspection JSON beside D; it never writes beside or over C. The automated
comparison passed: both sides retain five enabled operations in the order Drill,
Engrave, Pocket, Profile, Profile; the fourth targets `pocket-square` with
CutFeedrate 450 and ClearancePlane Default; and the last targets `profile-square`.
Exact operation parameter/state sets and the four-shape geometry/identity-transform
comparison also passed, with no warning/error output in
`native-check.log` or `native-geometry-check.log`.

Final acceptance was reported on 2026-09-08: C and D both load without error and
all stated checks match. The displayed order, targets, CutFeedrate 450,
ClearancePlane Default and final `profile-square` target were preserved. Do not
repeat the accepted A/B or C/D checks unless relevant behavior changes.

## Troubleshooting

- Import failure: verify the selected interpreter and install declared dependencies
  through that interpreter; do not change package metadata merely to suit a shell.
- Export failure: exceptions now propagate and the destination is replaced only
  after the temporary XML file closes successfully. Inspect the exception and
  primitive/MOP error context; fix the cause before retrying. If temporary cleanup
  fails, its path is logged. Successful export is not full schema/fidelity validation.
- State saving accepts `project.pkl` in the current directory. Parent-directory
  creation failures raise; pickle writes remain non-atomic.
- Consult [review evidence](REVIEW.md) before retrying a known approach.

### Export and persistence regression checks

```powershell
& $ProjectPython -m unittest discover -s tests -p test_export_failures.py -v
& $ProjectPython -m unittest discover -s tests -p test_state_persistence.py -v
```

These synthetic tests check exception propagation, destination preservation,
temporary cleanup, successful XML content/identity/targets and pickle restoration.
No new manual CamBam acceptance is required for this filesystem/error-boundary
slice: the successful XML format is unchanged and covered by the round-trip suite.

### Shape-parity regression checks

```powershell
python -m unittest discover -s tests -p 'test_shape*.py' -v
python -m unittest discover -s tests -p test_region.py -v
python -m unittest discover -s tests -p test_z_matrices.py -v
python -m unittest discover -s tests -p test_parity_bake_failures.py -v
python -m unittest discover -s tests -v
```

The selected interpreter for this increment is the existing Python 3.10.9
installation with NumPy 1.23.5; no `.venv` or new dependency was installed.
All suites run from the repository root without CamBam. Synthetic cases check
all seven shapes, actual XML Z/matrix fields, repeated round trips, Region
geometry/identity/MOP targets, hierarchy baking, copy/transfer and pickle links.
The legacy curved-bounds corruption test now mutates an already-constructed
entity because constructors explicitly reject invalid affine matrices.

## Manual shape-parity acceptance

Status: **accepted by the user on 2026-09-09**. Use the recorded
CamBam Plus 1.0 baseline. The synthetic files cover all seven supported shape
types, an extra Points parent, independent geometry/parent/local Z, two Region
holes, a semicircular bulged edge and one Region pocket target.

Generate and automatically inspect the local A/B files from the repository root:

```powershell
python output/shape-parity-kvr0bdc2/generate.py
python output/shape-parity-kvr0bdc2/verify_acceptance_xml.py
python output/shape-parity-kvr0bdc2/verify_native_roundtrip.py
```

- [A: geometry elevations plus hierarchy matrices](../output/shape-parity-kvr0bdc2/A_matrix_elevations.cb)
- [B: baked world coordinates and identity matrices](../output/shape-parity-kvr0bdc2/B_baked_elevations.cb)
- [Exact world geometry](../output/shape-parity-kvr0bdc2/expected-world-geometry.json)

Open A and B in CamBam. Both must load without error and show matching geometry
in top and oblique views, with eight primitives and one Region containing two
empty holes. B has identity transforms, so its coordinate properties are the
world values below. In A, the geometry coordinates and matrix offsets together
produce the same world values. Compare to absolute tolerance `1e-8` drawing
units, or the finest displayed precision if the property grid rounds values.

| Identifier | Expected world geometry in B |
| --- | --- |
| parent | Point `(0,10,0)` |
| mixed-pline | `(10,20,3)`, `(15,20,1)`, `(20,25,7)`; open, zero bulges |
| mixed-points | `(10,30,0)`, `(15,30,3)`, `(20,30,5)` |
| circle | Center `(30,25,-3)`, diameter 6 |
| arc | Center `(45,25,5)`, radius 4, start 0 degrees, sweep 180 degrees; upper extent Y=29 |
| rect | Corner `(55,20,3)`, width 8, height 6; every corner Z=3 |
| text | `Z parity`, height 3; visible anchor `(30,35,2)`. Optional XML `p2="35,35,7"` is retained but is not a visible/current alignment property. |
| region | Outer vertices `(70,20,6)`, `(90,20,6)`, `(90,40,6)`, `(70,40,6)`; bulge 1 on second vertex, rightmost X=100; all other bulges zero |
| region holes | Squares X=74..78 / Y=24..28 and X=82..86 / Y=32..36; all Z=6 |

Select the `Region pocket` machining operation: it must reference only the Region, with both holes still
part of that Region. No generated toolpath or machining result is required.
The user reported every visible A/B check passed. Text displayed at anchor
`(30,35,2)` with center/center alignment, line spacing 1 and the expected font
and style. CamBam does not expose `p2` as a property. The user saved B as
[C](../output/shape-parity-kvr0bdc2/C.cb); native XML retained
`p2="35,35,7"`. A separate fresh-session
[Text fixture](../output/shape-parity-kvr0bdc2/text_test.cb) created in CamBam
contains only `p1` despite `align="bottom,left"`, proving alignment is encoded
separately and normal Text does not require `p2`. This matches the
[official CamBam MText API](https://www.cambam.info/doc/api/MText.htm), which
states that P2 is currently unused. A second native observation showed that
CamBam omits both points at the default origin, then materializes equal `p1` and
`p2` values after moving the Text. The reader treats omitted `p1` as `(0,0,0)`;
the writer suppresses that default and preserves optional `p2`. The C-to-D framework check
also retains mixed-content Text, all geometry, identities, hierarchy, Region
holes and the Region-only pocket target. This acceptance covers display and
coordinate/property interchange, not generated production toolpaths.

## Local MCP setup and verification

The optional server requires Python 3.10+ and exactly `mcp==2.2.0`. Base-library
Python 3.9 installations remain supported; launching the adapter there produces
a clear version error. Install from the repository root:

```powershell
uv sync --python 3.13 --extra mcp
.venv/Scripts/python.exe -m cambam_builder.mcp_adapter --workspace D:/CAD/AgentWork
```

The workspace must already exist and be absolute. The equivalent installed
entry point is `cambam-mcp --workspace D:/CAD/AgentWork`. The client normally starts
and stops this process. Close stdin to stop it, or use Ctrl+C for an interactive
launch. Stdout is exclusively MCP; stderr emits the content-free workspace/boot
bootstrap. Restart loses all open handles, unsaved edits and retry records.

Eight tools create, open, inspect, save and close documents; add root Rects and
explicit Profile MOPs; and translate supported root Rects. Inspection reports
typed world geometry/bounds and machining parameters for this Rect/Profile slice,
while diagnosing other entity detail as unsupported until 4d. Saves only create new
`.cb` paths under the workspace and never overwrite. File parent directories must
already exist. Units are assertions, not conversions or a verified CamBam units
setting. Read the [state and safety contract](MCP_CONTRACT.md) before client use.

Codex client configuration (replace both absolute paths):

```toml
[mcp_servers.cambam]
command = "D:/Projects/CamBam-Builder/.venv/Scripts/python.exe"
args = ["-m", "cambam_builder.mcp_adapter", "--workspace", "D:/CAD/AgentWork"]
startup_timeout_sec = 15
```

Do not enable `mcp_2026_07_28` expecting it to establish modern stdio conformance
on Codex 0.154.0. Its observed stdio connection uses 2025-06-18, supported by the
required compatibility path. Modern SDK clients can call directly at 2026-07-28
or use discovery; the same document schemas and safety policy apply. Initialization
instructions/discovery expose the actual workspace ID required in tool calls.
No persistent client configuration is changed by the repository test suite.

Focused and full checks, from the root with the extra installed:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v
.venv/Scripts/python.exe demos/mcp_authoring_slice.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_project_clone.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -p test_strict_import.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m compileall -q cambam_builder legacy_cambam_builder tests demos
git diff --check
```

Protocol checks launch real subprocesses. Adapter tests require the optional extra;
base-only environments skip them. If the managed Windows sandbox creates temporary
directories it cannot reopen, rerun these checks with the tool's approved local
filesystem access; this environment failure is not an adapter failure. Test-owned
fixtures are synthetic. The demonstration creates a unique ignored workspace under
`output/`, runs the exact 4c authoring/retry/round-trip sequence and prints verified
artifact hashes, revisions and B's world corners. Disposable client probes, build
environments and logs are under `output/mcp-foundation-20260910/`; durable evidence is in the
[review record](REVIEW.md#mcp-foundation-and-client-compatibility---2026-09-10).

Manual CamBam validation adds no evidence for 4b's transport or 4c's framework/MCP
parity. 4c prepares authored CAD artifacts; 4e retains desktop/second-PC,
CamBam units/geometry/property and toolpath acceptance.
