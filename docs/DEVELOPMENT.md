# Development runbook

## Environment and setup

The declared toolchain is Python >=3.8 and setuptools/wheel (`pyproject.toml`).
`setup.py` loads runtime dependencies from `requirements.txt`. No lockfile,
repository-managed environment, test dependency or CI runner is declared.
Use an existing project environment when available; otherwise create an isolated
environment from the repository root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The setup command may require dependency downloads. Do not recreate an existing
environment. After setup, explicitly select its interpreter for every command;
activation is optional. If the IDE uses another project environment, set
`$ProjectPython` to that interpreter instead. Confirm it before running checks:

```powershell
$ProjectPython = '.\.venv\Scripts\python.exe'
& $ProjectPython -c "import sys; print(sys.executable); print(sys.version)"
```

Without a project environment, the available interpreter can perform baseline
checks if dependencies are already present; record that limitation. The initial
review did this without installing or changing dependencies.

## Verification entry points

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

Packaging installation and supported Python versions require separate validation;
a local import does not prove them.

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
child and leaf move (+5,-3) from Before as specified. CamBam version was not
supplied. These retained criteria need no repetition for unchanged behavior.

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
property outcomes were confirmed. CamBam version was not supplied. The retained
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
