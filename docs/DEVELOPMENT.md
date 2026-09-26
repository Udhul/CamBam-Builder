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

These are syntax/import and tracked-working-diff checks. `git diff --check` does not
inspect untracked files. Before calling uncommitted work ready to commit, also inspect
the intended untracked text files; this PowerShell check covers the repository's
normal source and documentation formats:

```powershell
$UntrackedText = git ls-files --others --exclude-standard -- '*.py' '*.md' '*.toml' '*.json'
if ($UntrackedText) { Select-String -Path $UntrackedText -Pattern '[ \t]+$' }
```

Before calling a committed branch merge-ready, replace `main` if another target base
was requested and run:

```powershell
git status --short --branch
git log --oneline main..HEAD
git merge-base --is-ancestor main HEAD
git diff --check main...HEAD
git diff --stat main...HEAD
```

The worktree must be clean, the commit range must contain the intended work, and all
commands must succeed. Review the complete `main...HEAD` diff, not merely its stat.
Run required behavior checks after the last material content edit; after a later
commit, rerun these branch-level gates against the final `HEAD`.

The repository preference is to preserve visible feature-branch topology. Unless the
user explicitly requests linear history, hand off these merge commands after the
feature branch passes the gates above:

```powershell
git switch main
git merge --no-ff <feature-branch> -m "merge: <feature description>"
git status --short --branch
git log --graph --decorate --oneline -12
```

Do not substitute `--ff-only`, squash or rebase integration by default: each removes
the explicit branch-and-join merge point the project wants visible in its graph.

Run the authored synthetic XML regressions with:

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

### Native owner migration checks

The canonical CamBam document modules are in `cambam_builder.native`; the old
root module paths remain import-compatible. From the repository root, use the
declared interpreter for the focused contract checks:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_entity_module_boundaries.py -v
& $ProjectPython -m unittest discover -s tests -p test_native_owner_migration.py -v
& $ProjectPython -m unittest discover -s tests -p test_region.py -v
& $ProjectPython -m unittest discover -s tests -p test_mop_roundtrip.py -v
& $ProjectPython -m unittest discover -s tests -p test_mop_context.py -v
& $ProjectPython -m unittest discover -s tests -p test_copy_transfer.py -v
```

The migration test checks canonical/legacy import identity, two full XML cycles
with a Region target, Pocket MOP identity/reference and Part-local stock offset,
and a same-code-version pickle snapshot. Broaden to the full suite for changes to
native owners or adapters. A wheel check must build into a unique ignored
`output/` directory, install it with its declared NumPy dependency into an
isolated environment and import from outside the repository source path. Verify
both `cambam_builder.native` and old root paths resolve to the installed wheel,
not to the checkout; the dated
[review evidence](REVIEW.md#native-owner-consolidation---2026-09-24) records one
such check.

Reproduce the development environment with `uv sync`. Because the lockfile is
local and ignored, dependency versions may advance over time. For a fresh
artifact check, build with `uv build --out-dir <unique-output-directory>`, install
the resulting wheel or sdist into a clean `uv venv --python <version>`, copy the
tests to a directory outside the repository, and run discovery there with that
environment's interpreter. This prevents the repository root from satisfying
imports accidentally.

### Detached nominal planar runtime checks

Install the optional runtime backend with `uv sync --extra planar --python 3.13`
(add `--extra mcp` when also exercising MCP). The base dependency set remains
NumPy-only; `cambam_builder.planar` imports lazily and analytic feasible centers
work without Shapely. Metadata pins the evaluated Windows-compatible releases:
Shapely 2.0.7 for Python 3.9 and 2.1.2 for Python 3.10+.

```powershell
& $ProjectPython -m unittest discover -s tests -p test_planar.py -v
& $ProjectPython -m unittest discover -s tests -v
```

The focused suite verifies the detached API end to end and its explicit failure
states. Without the extra, backend tests skip explicitly while analytic/optional-
import tests still run; those skips are not backend acceptance. The runtime suite
can also run from the repository root with the previously isolated evaluation
interpreters below, without changing their installed dependencies. Nominal evidence
and API exclusions are in the [specification](structure_spec.md#detached-nominal-planar-core).
No manual CamBam validation is needed for this nonserialized geometry slice.

### Directional stock section checks

```powershell
& $ProjectPython -m unittest discover -s tests -p test_stock.py -v
```

Twenty backend-independent analytic tests exercise exact sweep/rest areas, directional
containment under radius/position uncertainty, boundary contact and exact overrun
rejection, collapsed/empty guarantees, large translations and unsupported inputs.
Composition checks add independent overlapping capsule/disk-lens and disjoint area
references, duplicate/order invariance, empty input, uncertain actual unions,
monotonic stock prefixes, grid refinement and mismatched/altered-source rejection.
The target-aware checks independently compare required rest and whole-stock state
against a rectangular target with one island and supplied large/small-tool sweeps.
They check crossing into uncleared target, protected-area preservation under
uncertainty, exact wall tangency and rational overrun rejection.
Section-motion checks add ordered cutting and explicit entry/travel verification
against prior guaranteed removal or strictly outside-stock space. The independent
cleanup reference checks pointwise residual membership; rejected cases include
uncleared connectors, an island crossing, forged future clearance, and a tiny
uncertainty overrun. This remains a fixed-Z section certificate: vertical access,
tool changes and other heights are unverified.
No CamBam manual validation adds evidence: this slice has no serialization, path
generation or execution claim. Physical uncertainty limits remain caller inputs.

### RC01 standalone generated-sequence checks

Install the already declared optional planar backend (`uv sync --extra planar
--python 3.13`), then run from the repository root:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_rc01.py -v
& $ProjectPython -m unittest discover -s tests -p test_stock.py -v
& $ProjectPython -m compileall -q cambam_builder
git diff --check
```

The RC01 suite checks deterministic ordered T1/T2 motion, exact all-height
access/component and process rejection, independent three-slab residual bounds,
and missing-bottom-layer rejection. A passing result is synthetic standalone
evidence only. GEOS powers the conservative residual-location check; its
polygon sagitta is below 0.000001 mm, while floating topology is not a formal
interval proof. Native input, explicit Engrave motion, native Pocket motion and
physical machining have separate [RC01 gates](REST_MACHINING_PLAN.md#rc01-standalone-and-cambam-output-gates).
No manual CamBam check adds evidence to this standalone implementation itself.

### Bounded pointed-cone slot checks

From the repository root, using the declared project interpreter:

```powershell
& $ProjectPython -m unittest tests.test_vcarve_slot -v
& $ProjectPython -m compileall -q cambam_builder/cam_core tests/test_vcarve_slot.py
git diff --check
```

The focused suite checks finite plunge/cut/retract and above-stock links for the
12 x 4 mm slot, exact 2 mm cone guard, full-height containment rejection,
independent row-integrated section residuals, analytic V and capped-depth
references, and volume enclosures. No CamBam file is generated. Manual CamBam
validation adds no evidence to this detached geometric slice; it remains a
conditional ideal-stock calculation, not a production toolpath or native output
acceptance.

### Shared RC01/cone motion and stock replay checks

From the repository root, using the declared project interpreter:

```powershell
& $ProjectPython -m unittest tests.test_rc01 tests.test_vcarve_slot tests.test_mixed_replay tests.test_rc01_native tests.test_rc01_stock_authority -v
& $ProjectPython -m compileall -q cambam_builder tests/test_mixed_replay.py
git diff --check
```

The mixed test checks a single ordered source/motion fingerprint and stock cut
prefix across RC01 T1, T2 and the translated cone slot. It checks independent
residual membership before and after the cone cut, and rejects stale source,
removed cone entry, low link and missing tool change. The adjacent suites
retain each original target, access and numerical residual oracle; the native
RC01 checks guard its existing adapter consumers. This is detached synthetic
evidence only. At that checkpoint, a manual CamBam check added no evidence;
the posted cone output audit is the separate next section.

### Convex closed-region rest and pointed cleanup checks

From the repository root with the declared interpreter:

```powershell
& $ProjectPython -m unittest tests.test_convex_rest tests.test_variable_vcarve tests.test_mixed_replay tests.test_vcarve_slot -v
& $ProjectPython -m compileall -q cambam_builder/cam_core tests/test_convex_rest.py
git diff --check
```

The triangle case checks exact original and pure-rest section areas at 0, 1 and
2 mm, plus a separate midpoint row integration of the final variable-radius
cone sweep within 0.002 mm². It checks ordered prior/cleanup prefixes, a
prior-cleared descent, partial completion, stale source/motion, invalid prior,
overcut, unsupported polygon/tool and low-link rejection. This is detached
synthetic geometry only. No manual CamBam validation adds evidence for this
slice; source normalization, preview and emitted execution remain distinct
future gates.

### Native triangle source, preview and post gate

The bounded native consumer is `integrations.cambam.native_convex_rest`. From
the repository root, build a fresh synthetic case with the declared interpreter:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_convex_rest output/native-convex-rest-NEW
& $ProjectPython -m unittest tests.test_native_convex_rest tests.test_convex_rest -v
```

For an existing accepted triangle, provide both its `.cb` and matching
source-SHA-bound supplied prior trace. The generated example's `prior.json`
shows the exact schema. Existing source without `--prior` is rejected:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_convex_rest output/native-convex-rest-EDITED --source path/to/source.cb --prior path/to/prior.json
```

The retained synthetic delivery is `output/native-convex-rest-20260924-04/`.
Its `source.cb`, `prior.json`, `preview/triangle-preview.cb` and
`explicit/triangle-explicit.cb` are separate files. The source SHA-256 is
`fd811374e7cfb12783d25bc90f937f14167bee524fc8b18176f50b84dd6fc79d`;
the prior JSON is `d3af3561df9b18bf91283dc0939223c1628a6a5f272d04918d4b7b7aac95c1dd`;
the preview is `7da156b420a7db4f1267f66db85dab0efc353696ee664592cdf02f40c85d0a68`;
and the explicit candidate is
`e3bf022ab0f3cabfa78906e5466d714dc4644d24d36469c4b6fddae95c841c60`.
The manifest pins these hashes, the prior and full motion fingerprints, every
expected motion item and section residuals. The prior tip enters at `(3,2,1)`,
plunges to `(3,2,-1.2)` and retracts. The generated cleanup descends through
that cleared column, cuts from `(3,2,-1.2)` to `(4,2,-2)`, then retracts.
Pure rest at depths 0/1/2 mm is
`43.476106579/21.207669627/5.333333333` mm²; final partial rest is
`35.160992224/18.089501744/5.333333333` mm² (1e-9 mm² comparison for the
manifest calculations).

CamBam validation adds two pieces of evidence. First open the **preview**
`.cb` in CamBam Plus 1.0, confirm millimetres, the three original Region
vertices, stock XY `(-1,-1)` to `(13,9)` and Z `-3` to `0`, and that the enabled Engrave
targets only the generated XYZ cleanup line. Generate its toolpath and report
whether the displayed segment visibly slopes from Z=-1.2 to Z=-2; this checks
preview usability only. Then open the **explicit** `.cb` separately with
**Default** postprocessor and **Default mm** profile, confirm only its
Drill/CustomScript is enabled, generate toolpaths and post it as
`explicit/triangle-explicit.nc`. Do not post the preview. Preserve all four
pinned files unchanged. The declared setup assumes the tip starts at
`(-10,-10,+5)` and uses test-only T3, 90-degree pointed 3 mm radius/length,
CW 12000 rpm, entry 60 and cut/retract 300 mm/min; the post does not prove
physical setup.

Audit the actual CamBam-produced NC from the root:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_convex_rest output/native-convex-rest-20260924-04/expected-motion.json output/native-convex-rest-20260924-04/explicit/triangle-explicit.nc
```

Pass requires `bounded_triangle_post_pass`, the exact candidate hash above,
all ten ordered events/moves, `prior` then `cleanup` cut prefixes, and the
stated pure/final rest values. Any missing/extra move, feed or coordinate,
changed source/prior/candidate, unsupported post word or replay failure is a
fail or unverified result. Report preview slope pass/fail, source/stock/units
pass/fail, the generated NC path and audit result. Keep the files until that
acceptance is recorded; there is no production or controller claim.

The user generated the retained `explicit/triangle-explicit.nc` in CamBam
Plus 1.0 and observed its enter/retract/re-enter/sloped-cut/retract sequence in
CAMotics. The posted file SHA-256 is
`2cfe5e2c9cda871342f7b229fd4a77658a800b7af01dc36746adedc03a3aea5a`.
The command above returned `bounded_triangle_post_pass`: all ten items matched,
the parsed post replayed `prior` then `cleanup`, and the final section rest
values match the manifest. CamBam did not show the CustomScript motion as a
toolpath; inspect the separate preview file for a visible generated path.
`preview/triangle-preview.nc` was also parsed and contains the exact F300
segment `(3,2,-1.2)` to `(4,2,-2)`. The user confirmed the preview's sloped
polyline is visible in CamBam. They confirmed the source triangle in the
millimetre drawing and the Part's offset `(-1,-1)`, size `(14,10,3)` and
stock surface Z=0, putting the global stock bottom at Z=-3. This completes
the bounded native display and posted-motion acceptance. Do not repeat the
accepted explicit post unless the candidate,
source, prior or output setup changes. Physical machining remains unverified.

### M1 polygonal Region rest and smaller-endmill output gate

Install the optional planar backend (`uv sync --extra planar`) and use the
declared project interpreter. The retained synthetic fixture and posted
CamBam candidates are under `output/m1-polygon-20260924-04/`. `source.cb`
has the eight-edge letter-like Region and triangular hole. `prior.json`
contains the complete supplied T1 motion and its exact source binding.
`expected-motion.json` pins the source, supplied motion and all candidates.
The preview and execution files are separate:

- `preview/m1-preview.cb`: two final-level Z=-8 T2 contour centerlines,
  one at the shell and one at the hole, as a visual Engrave. The full
  four-level sequence is in the explicit candidate, not this preview.
- `explicit/m1-explicit.cb`: one literal T1/T2 Drill/CustomScript execution
  carrier; its displayed Drill toolpath is not its executable motion.
- `native/m1-native.cb`: independent original-Region T1/T2 Pocket trial.

For another existing source, supply both its `.cb` and matching source-bound
`prior.json`; an absent prior fails rather than inferring Pocket removal:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_polygon_rest output/m1-polygon-NEW --source path/to/source.cb --prior path/to/prior.json
& $ProjectPython -m unittest tests.test_polygon_rest tests.test_rest_vcarve_acceptance_fixtures -v
```

The retained source SHA-256 is
`be092ec2827810ca0c47f6bc7a9a901a2d9fa2d609b096599261995bf1e5947f`;
the supplied prior is
`875becc966121baa362299afe756e12fb7159257e2f03f03110f25cd0ac3c98c`.
The revised explicit candidate is
`3691b0221882247ca3b9be577dd7f1b4d5fa5138a9523bf20b8ef445dbd44973`;
the native candidate is
`4b2140b4d317511cc798cfa6acadfeee8cbfa02748974f9506129fbc38811460`.
The latter is byte-identical to the already posted native trial; no native
repost is needed. Do not modify the pinned files. The synthetic setup assumes
millimetres, original Region area 1532 mm², Part stock X=-26..26, Y=-2..62,
Z=-8..0, initial tip (-30,-10,+5), T1 diameter 5, T2 diameter 2,
10 mm cutting length, 2 mm axial levels to Z=-8, CW 12000 rpm, feed 60
for entry and 300 mm/min for cut/retract. The T1 allowance is 0.5 mm.
These are test tokens, not real material cutting parameters.

Manual validation adds evidence beyond automated checks. In CamBam Plus 1.0,
open the **preview** and confirm the source shell/hole, stock and a visible
T2 centerline after generating toolpaths; do not post this file. Open the
**explicit** file separately, select `Default` postprocessor and `Default mm`
profile, confirm only the literal Drill MOP is enabled, generate toolpaths and
post to `explicit/m1-explicit.nc`. Return that exact NC file and report
source/stock/units and whether both preview contours are visible. The
CustomScript motion may be absent from CamBam's toolpath display; its actual
post is the execution evidence.

Audit both posts from the repository root:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_polygon_rest output/m1-polygon-20260924-04/expected-motion.json output/m1-polygon-20260924-04/explicit/m1-explicit.nc --route explicit
& $ProjectPython -m cambam_builder.integrations.cambam.native_polygon_rest output/m1-polygon-20260924-04/expected-motion.json output/m1-polygon-20260924-04/native/m1-native.nc --route native
```

The revised explicit gate returned `bounded_m1_explicit_post_pass` for actual
Default NC SHA-256
`e214571c55c67cb525a0a2d17a8b0508b9aa11da776e97a0c138b866ee172ba5`.
It verified all 2,692
ordered items, `prior` then `cleanup` stock prefixes, and the same section
rest interval at depths 1/3/5/7 mm: rough
137.218201–137.230366 mm² and final 1.313264–1.318529 mm². The
independent finite-tool lower limit is 1.190659933 mm²; the allowed final
upper limit is 1.690659933 mm². Inflated nominal protected overcut and
residual outside the ideal/original-boundary 0.05 mm envelope must be zero.
The separate native gate needs `bounded_m1_native_post_pass` with no motion
findings and the same final upper budget. Its Pocket strategy may emit
different coordinates from the explicit trace; it cannot borrow that trace's
certificate. The actual retained native post has SHA-256
`c675f00cbaf050c3b7776d8ecab45164a403419db71d473d3ee8d85b22471d05`.
Its area passes, 180 low vertical rapids have prior-cut witnesses and
boundary shortfall is at most 0.000397 mm within the declared 0.001 mm
backend tolerance, but 12 T2 feed descents have no T1-cleared column.
The native command returns `native_motion_gate_failed`. The selected M1 route
is the audited explicit post; the native Pocket route is excluded. Do not
repeat the native post or iterate minor Pocket controls. The posted explicit
NC contains the fixture's supplied T1 raster, with 208 long horizontal feed
segments, followed by the two-contour T2 cleanup. CamBam does not render the
CustomScript motion as its generated toolpath; inspect the NC in CAMotics for
the full sequence. The preview Engrave shows only the final-depth T2 contours.
For A01, integrated rough rest is 1097.74561–1097.84293 mm³ and final rest
is 10.50611–10.54823 mm³, conditional on the section geometry backend.
The synthetic T1 raster is a stock-proof fixture, not a recommended machining
strategy. The user's revised preview display and source/stock UI observations
were not separately reported; strict file reimport and actual NC acceptance
are recorded independently.
Neither gate is controller or physical machining acceptance. The Default
post does not encode the incoming machine position, and GEOS topology is not
a formal numerical interval proof.

### M2 curved Region rest and smaller-endmill output gate

Use `.venv\Scripts\python.exe` from the repository root with the optional
planar backend installed. The tracked acceptance inputs and numeric limits are
`M2_*` in [the rest corpus](../tests/fixtures/rest_vcarve_acceptance.json).
The three prepared, ignored synthetic jobs are:

| Case | Directory | Analytic opening area | Rough/final area interval at Z=-1 and -3 mm |
| --- | --- | ---: | ---: |
| Annulus | `output/m2-annulus-20260925-02/` | 241.902634242 mm² | 17.96728–18.10760 / 0.03366–0.17300 mm² |
| Mixed line/arc concave Region with circular hole | `output/m2-mixed-20260925-01/` | 631.292105800 mm² | 35.06282–35.31860 / 0.64979–0.90388 mm² |
| Translated/reflected mixed Region | `output/m2-reflected-20260925-01/` | 631.292105800 mm² | 35.06282–35.31860 / 0.64979–0.90388 mm² |

Each directory contains exact `source.cb`, `prior.json`, separate
`preview/m2-preview.cb` and `explicit/m2-explicit.cb`, and a hash-guarded
`expected-motion.json`. The source preserves native bulges; the reflected
case retains a native Region transform. The preview Engrave shows only final
depth T2 paths. The literal Drill/CustomScript carries all supplied T1 and
generated T2 motion, including entries, high links and retracts. Its displayed
Drill path does not show the CustomScript motion. T1 raster/contour motion is
a stock-proof fixture, not a production roughing recommendation. The declared
millimetre setup uses stock Z=0..-4, T1/T2 diameters 3/1.5 mm, both cutting
lengths 8 mm, CW 12000 rpm, F60 entries, F300 cuts/retracts and a +5 mm tip
clearance. These are test tokens, not material-safe feeds or controller output.
The CustomScript post checks literal-motion transport through CamBam; it does
not show that CamBam independently planned the same path. The separate Engrave
post checks CamBam's native interpretation of the generated T2 preview
centerlines. Neither preview operation supplies the T1/entry/link certificate.
The analytic arc area and independently fixed residual budgets, source-bound
stock replay, and complete actual explicit post provide that bounded result.

Rebuild a new isolated job with an exact existing `.cb` and matching supplied
motion, then run focused checks:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_curved_rest output/m2-new-UNIQUE --source path/to/source.cb --prior path/to/prior.json
& $ProjectPython -m unittest tests.test_curved_rest tests.test_polygon_rest tests.test_region -v
```

The source alone cannot assert T1 removal; a missing or stale `prior.json`
fails. Generated area and volume are conditional GEOS bounds around an
analytic circular-arc source. The 0.001 mm maximum chord sagitta and explicit
inner/outer Regions are recorded in each manifest. The curved narrow-annulus
fixture rejects a tool wider than its 0.8 mm radial throat; replay also rejects
cuts across protected holes. The independent source-area and residual budgets
are in the corpus, and a posted file must satisfy the same budget.

For each of the three jobs, open its `source.cb` and `preview/m2-preview.cb`
in CamBam Plus 1.0. Confirm millimetres, the native curved shell/hole and Part
stock, and visible generated T2 centerlines at Z=-4. For the reflected job,
confirm its curved Region appears at drawing X=24..56 and Y about -5..17.
Then open `explicit/m2-explicit.cb` separately, choose **Default** postprocessor
and **Default mm** profile, confirm only its literal Drill is enabled, generate
toolpaths and post to `explicit/m2-explicit.nc`. Inspect the complete T1/T2
motion in an NC viewer: every low cut stays in the opening and every link
retracts to +5 mm. The original source and expected files must stay unchanged.

Audit each actual file from the repository root:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_curved_rest output/m2-annulus-20260925-02/expected-motion.json output/m2-annulus-20260925-02/explicit/m2-explicit.nc
& $ProjectPython -m cambam_builder.integrations.cambam.native_curved_rest output/m2-mixed-20260925-01/expected-motion.json output/m2-mixed-20260925-01/explicit/m2-explicit.nc
& $ProjectPython -m cambam_builder.integrations.cambam.native_curved_rest output/m2-reflected-20260925-01/expected-motion.json output/m2-reflected-20260925-01/explicit/m2-explicit.nc
& $ProjectPython -m cambam_builder.integrations.cambam.native_curved_rest output/m2-annulus-20260925-02/expected-motion.json output/m2-annulus-20260925-02/preview/m2-preview.nc --preview
& $ProjectPython -m cambam_builder.integrations.cambam.native_curved_rest output/m2-mixed-20260925-01/expected-motion.json output/m2-mixed-20260925-01/preview/m2-preview.nc --preview
& $ProjectPython -m cambam_builder.integrations.cambam.native_curved_rest output/m2-reflected-20260925-01/expected-motion.json output/m2-reflected-20260925-01/preview/m2-preview.nc --preview
```

Pass requires `bounded_m2_curved_post_pass` and
`m2_native_preview_centerlines_pass` for all three unchanged candidates,
zero inflated protected overcut, matching every ordered event/move and the
manifest's stock prefixes and residual intervals. Report each NC path, audit
result and the source/stock/preview observations. A constructed NC regression
tests the parser but is not actual CamBam acceptance. Controller and physical
machining acceptance remain separate.

**2026-09-25 actual CamBam Plus 1.0 acceptance.** The user reported visible
curved preview paths in CamBam, generated all three preview and explicit NC
files, and confirmed that CustomScript motion is absent from the Drill
toolpath display. Strict native reimport checked the original Region bulges,
Part stock and candidate source identities. All three preview posts passed the
separate 4-decimal T2 centerline comparison at Z=-4; the annulus has 2 paths
with 214/502 vertices, and both mixed cases have 2 with 192/442 vertices.
All three explicit Default posts passed exact ordered event/move comparison,
source/candidate hashes, stock replay and the fixed residual/overcut budgets:

| Case | Preview post SHA-256 | Explicit post SHA-256 | Ordered items | Final area interval at Z=-1/-3 mm |
| --- | --- | --- | ---: | ---: |
| Annulus | `4f19ee35e7ca9763c8aaced736e70ac2caa0b798ab9cdb325e5f06e47ab84080` | `c7fb042a63380f2c94970aed2b645f627511e527b975273eb29e12690e4fecf5` | 2,984 | 0.033661–0.172997 mm² |
| Mixed | `0851e8774b1feede38f1e3853f09ef619e28536ecff5a71a1800049602549988` | `a8499c356958d8d849556841ca8750172e662faf30fcc4c5f4b4fb1e84cff6e2` | 2,688 | 0.649797–0.903873 mm² |
| Reflected | `718e45c8b8cee4af17f8ef79735e7a2db11b4367ab17c39793bc8b43632bb375` | `879bc3c0e2416e1e75d2d8cebc39f2e3480de406b6a21ed9f9cdc4a4243a9b4c` | 2,684 | 0.649797–0.903873 mm² |

The two mixed v1 manifests retained fingerprints incorporating an older replay
target representation. Their source hashes, every supplied T1 item, generated
T2 item, script line, analytic area and residual bound match exactly. The
legacy acceptance branch permits that fingerprint drift only when the supplied
T1 items equal the source-derived synthetic fixture; other supplied traces
still need a current exact fingerprint. A 0.1 mm preview NC coordinate tamper
returned `deviation`. No physical or controller acceptance is implied.

### M3 Region V paths and native output gate

The ignored `output/m3-v-suite-20260925-02/` holds seven source-bound jobs.
Earlier `m3-v-suite-20260925-01` files were standalone V probes and are not
the M3 acceptance candidates; use the `-02` combined jobs below.
The seven original explicit NC posts passed their complete-stream gate.
Six original preview posts exposed shallow CamBam crossover feeds, so the
corrected preview candidates are under
`output/m3-v-preview-retract-20260925-01/`. They use the same source, prior,
tool and path geometry, with Engrave `MaxCrossoverDistance=0`. Preserve the
original exports as failure evidence. All fourteen corrected actual posts
passed their separate gates; the user also exported fresh explicit posts.
Each has `source.cb`, `prior.json`, `preview/m3-preview.cb`,
`explicit/m3-explicit.cb` and `expected-motion.json`. The first six pair the
accepted A01 letter and M2 annulus with pointed, 0.25 mm flat-tip and 0.5 mm
tangent rounded-tip V tools; `mixed-rounded` adds the accepted concave line/arc
shell and circular hole. All use a 2 mm capped inward V recess, 1 mm raster
stepover, 0.01 mm center clearance margin, 2 mm tip-Z change per XY
millimetre maximum cut slope and +5 mm tip clearance. The
included angles are 90/90/60 degrees respectively. Tool maximum radius is
4 mm and cutting length 3 mm. A separate source-bound synthetic T1 trace
uses a 1 mm cylinder and cuts only the interior of the V target before T3.
T1 and T3 use CW 12000 rpm, F60 entries and F300 cuts/retracts. These are
test tokens, not material or controller settings. The
original source Region and Part stock must stay unchanged.

| Job suffix | Section Z=-1 residual interval, mm² | Generated literal moves |
| --- | ---: | ---: |
| `letter-pointed` | 3.95018–4.36033 | 2368 |
| `letter-flat` | 4.90681–6.19275 | 2281 |
| `letter-rounded` | 4.17639–4.80768 | 2316 |
| `annulus-pointed` | 1.27043–1.53412 | 453 |
| `annulus-flat` | 1.26181–1.51851 | 434 |
| `annulus-rounded` | 1.31719–1.52394 | 437 |
| `mixed-rounded` | 2.64310–4.19296 | 924 |

These are conditional GEOS bounds from the four-decimal candidate paths;
every inflated nominal protected-overcut area at Z=-1 is zero. At Z=-1 the
prior upper residual is 626.47571 mm² for letter pointed/flat, 587.12897 mm²
for letter rounded, 112.36541 mm² for annulus pointed/flat, 103.12817 mm² for
annulus rounded and 260.20373 mm² for mixed rounded. Minimum prior-to-V gains
are 500/90/200 mm² for letter/annulus/mixed. The independent
upper budgets are 5/7/5 mm² for the letter, 2 mm² for every annulus profile
and 5 mm² for mixed rounded. The annulus rounded eight-slab volume upper bound
is 80 mm³. The tracked [M3 corpus](../tests/fixtures/rest_vcarve_acceptance.json)
owns those criteria and the narrow curved flat-tip infeasibility fixture.
The [core contract](structure_spec.md#m3-bounded-region-v-path-and-native-candidate-contract)
explains the finish target and numeric limits. The supplied T1 raster is a
stock-proof fixture, not a production roughing recommendation. The result
does not claim a square-wall flat-floor finish.

Run the focused automated checks from the root:

```powershell
& $ProjectPython -m unittest tests.test_v_region tests.test_native_v_region tests.test_rest_vcarve_acceptance_fixtures -v
```

To reproduce the accepted native inspection, for each suffix above open
`output/m3-v-preview-retract-20260925-01/<suffix>/preview/m3-preview.cb`
in CamBam Plus 1.0.
The user's original seven previews already established visible source Regions,
shell/hole edge paths and variable-Z fill paths; the corrected candidates have
the same source and plan fingerprints, so that check need not be repeated.
Generate toolpaths and post with **Default** / **Default mm** to
`preview/m3-preview.nc` beside that corrected `.cb`. The corrected
`explicit/m3-explicit.cb` was also posted with Default mm. Its
displayed Drill path need not show the CustomScript cuts; the NC contains the
complete T1/T3 motion: no XY rapid at Z<=0, no cut outside the Region or below
Z=-2, and a +5 mm retract between disconnected paths. The proposed feed and
spindle values need no physical machining trial for this gate.

Audit the fourteen corrected posts from the repository root:

```powershell
$Names = 'letter-pointed','letter-flat','letter-rounded','annulus-pointed','annulus-flat','annulus-rounded','mixed-rounded'
foreach ($Name in $Names) {
  $Job = Join-Path 'output/m3-v-preview-retract-20260925-01' $Name
  & $ProjectPython -m cambam_builder.integrations.cambam.native_v_region preview "$Job/expected-motion.json" "$Job/preview/m3-preview.nc"
  & $ProjectPython -m cambam_builder.integrations.cambam.native_v_region audit "$Job/expected-motion.json" "$Job/explicit/m3-explicit.nc"
}
```

Each corrected preview returned `m3_v_preview_centerlines_match`; each
corrected explicit post returned `bounded_m3_v_post_pass`. A parser failure,
`deviation`, changed hash, missing segment or unsafe link fails that job.
Record the fourteen combined audit statuses, including any first failing
line/reason. The original seven visible preview observations remain accepted;
the corrected jobs retain their source and plan fingerprints. The synthetic
post test checks the parser only, while these actual CamBam posts establish
the bounded native gate. Controller and physical machining acceptance are
separate. Exact accepted NC hashes are in the
[dated review](REVIEW.md#m3-corrected-native-output-acceptance---2026-09-25).

To regenerate a new isolated job or audit one file:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_v_region build output/m3-new-UNIQUE --case annulus --profile rounded
& $ProjectPython -m cambam_builder.integrations.cambam.native_v_region audit output/m3-new-UNIQUE/expected-motion.json output/m3-new-UNIQUE/explicit/m3-explicit.nc
```

For an existing native source, pass both `--source path/to/source.cb` and
`--prior path/to/prior.json`; a source alone has no removal authority.

### Bounded cone CustomScript carrier and posted replay

Build one new ignored directory from the repository root:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.cone_script output/cone-script-NEW
& $ProjectPython -m unittest tests.test_cone_script tests.test_vcarve_slot tests.test_mixed_replay -v
```

The current prepared file is
`output/cone-script-20260924-01/V-cone.cb` (SHA-256
`59cd263d5dafac98a5f2dd794ea30906545118e3ebf3631d015e8fed62143150`).
`source.cb` is the synthetic Rect/Part input; `expected-motion.json` gives the
exact nine expected items, six literal NC lines, process values, hashes and
analytic residual references. Both `.cb` files strict-reimport. The ignored
`refresh_expected.py` updated only that manifest to the final complete-item
format after the candidate was generated; it did not edit the candidate.

In CamBam Plus 1.0, open `V-cone.cb`, use the **Default** postprocessor and
**Default mm** profile, generate toolpaths (Ctrl+T), and post G-code (Ctrl+W)
to `V-cone.nc` in that same directory. Do not edit either `.cb`. The expected
motion begins with T3/M6 and M3/S12000 at the declared setup tip
(-10,-10,+5), rapids to (2,2,+5), feeds to (2,2,+1), plunges to (2,2,-2),
cuts to (10,2,-2), feeds up to (10,2,+1), rapids back to setup, then stops.
The `expected-motion.json` file is the complete coordinate/feed/event reference.

Audit the **CamBam-produced** file with:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.cone_script output/cone-script-20260924-01/expected-motion.json output/cone-script-20260924-01/V-cone.nc
```

Pass requires `bounded_emitted_cone_motion_pass`, nine matched posted items,
one `slot` prefix with two ideal cut sweeps, and residual areas at depths
0/1/2 of 3.4336293856408275 / 0.8584073464102069 / 0 mm2 (floating
comparison tolerance 1e-9 mm2). Extra, missing, reordered or changed moves,
feeds, tool/spindle events, low rapid and protected-wall overcut fail. The
Default post does not encode initial machine position; physically confirm the
declared setup separately before any machine use. The result is an ideal-stock
output observation, not physical machining acceptance.

The user posted the prepared file on 2026-09-24 and confirmed CamBam displayed
the Drill toolpath after closer inspection. Its Ctrl+W Default post emitted
all six literal motion blocks. `V-cone.nc` SHA-256 is
`1c4281c930e26fade89d5fc75064a36908466f6ffa3ed96d2b5d30b390761cd4`.
The command above returned `bounded_emitted_cone_motion_pass`: nine items,
one `slot` prefix/two cone sweeps, partial target completion and the exact
0/1/2 mm residual references. No repeat CamBam export is needed for this gate.

### Bounded variable-depth V groove carrier and posted replay

From the repository root, using the declared project interpreter:

```powershell
& $ProjectPython -m unittest tests.test_variable_vcarve tests.test_variable_cone_script tests.test_cone_script tests.test_vcarve_slot tests.test_mixed_replay -v
& $ProjectPython -m cambam_builder.integrations.cambam.variable_cone_script output/variable-v-NEW
```

The prepared file is `output/variable-v-20260924-01/V-variable.cb`, SHA-256
`5f58068f39ffcaef6cf40147750a62f4e0be55d5580a42c90a01c9ae26d064b0`.
Its strict-reimported `source.cb`, SHA-256
`652a830b75ee19fb10b860cf14f796ed85ad1a0115a2d4157ce3d144262b1d2c`,
contains the exact XYZ finish-spine guide and stock. The ignored
`expected-motion.json` pins the source/candidate and all nine expected items.
The guide runs from (0,2,-1) to (12,2,-2.5); the one generated sloped cut runs
from (2,2,-1.25) to (10,2,-2.25). Both use the same 90-degree pointed-cone
depth law. The finite ends remain partial target stock.

To repeat the accepted output gate, open `V-variable.cb` in CamBam Plus 1.0,
select **Default** postprocessor and **Default mm** profile, generate toolpaths
(Ctrl+T), then post G-code (Ctrl+W) to `V-variable.nc` in the same directory.
Leave the prepared `.cb` files unchanged. The Drill MOP intentionally targets
the one-point script anchor; the sloped XYZ Pline is a finish-target guide and
does not drive a native V-carve MOP. The posted bytes are the motion authority.
The script's six
lines and expected ordered roles are in `expected-motion.json`. In particular,
the cut must feed from Z=-1.25 to Z=-2.25 while X moves from 2 to 10. The
test-only feed tokens are 120 approach, 60 plunge and 300 cut/retract mm/min.

Audit the **CamBam-produced** file with:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.variable_cone_script output/variable-v-20260924-01/expected-motion.json output/variable-v-20260924-01/V-variable.nc
```

Pass requires `bounded_emitted_variable_v_motion_pass`, all nine exact emitted
items, one `variable-v` prefix with two cone sweeps, and ideal section rest at
depths 0/1/1.5/2/2.5 mm of 15.091265791880026 / 7.028684027518187 /
4.21460291488107 / 1.8062583920918873 / 0 mm2, each within 1e-9 mm2 of
the recorded analytic calculation. Independent row integration agrees within
0.0005 mm2. Any extra, missing, reordered or changed event, coordinate, G0/G1
role or feed fails. The post does not encode incoming physical machine position;
tip (-10,-10,+5) is a declared setup assumption. No physical machining is
authorized by this ideal geometric test.

The user exported the prepared file on 2026-09-24. The actual `V-variable.nc`
SHA-256 is `3d37cd3ecd5c69efbdb3ddab381f283dda2352d470a3733e9477606da0e6c570`.
The command above returned `bounded_emitted_variable_v_motion_pass`, nine
matched items, one `variable-v` prefix/two sweeps and the exact five residual
references. No repeat post is needed for this bounded Default/Default mm gate.
The user reported the one-point Drill target; that is the declared explicit
carrier relationship, not native sloped-Pline toolpath generation.

### Bounded variable-depth XYZ Engrave preview and post probe

The one candidate is
`output/variable-v-engrave-20260924-01/V-variable-engrave.cb` (SHA-256
`0b7d2e3d42a04785b2989f4b19b0d43916ce09452f62f76d5182c3fb141a5585`).
Its `source.cb` (SHA-256
`aea8757a00e0e4fcc3aa0bd5b43005332c3c4635d29231b7f81265271c19bf4a`)
holds the original target spine. The candidate adds the generated cut as a
different XYZ Pline and enables one Engrave MOP targeting **only** that cut.
`expected-motion.json` pins the source/candidate hashes, plan/motion
fingerprints, complete nine-item reference and five rest-area references.
TargetDepth=0, OptimisationMode=None and DepthIncrement=3 are explicit. The
synthetic Default wrapper passes the whole-motion audit; a wrapper with the
correct sloped cut but missing approach/retract fails. This establishes local
adapter behavior only. The later CamBam preview and native post result are
recorded below.

In CamBam Plus 1.0, open the candidate above with **Default** postprocessor
and **Default mm** profile. Select `CANDIDATE variable-depth XYZ Engrave`,
generate toolpaths (Ctrl+T), and inspect the XZ view. Report whether the
Engrave toolpath contains exactly one sloped cutting segment from
(2,2,-1.25) to (10,2,-2.25), with no extra depth pass. Then post G-code
(Ctrl+W) to `V-variable-engrave.nc` in the same directory. Leave the `.cb`
and manifest unchanged. This is synthetic evidence gathering; do not run the
post on a machine. The original target guide is separate from the selected
cut Pline, and the Drill/CustomScript MOP is absent from this candidate.

Audit the actual CamBam-produced file from the repository root with:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.variable_cone_engrave output/variable-v-engrave-20260924-01/expected-motion.json output/variable-v-engrave-20260924-01/V-variable-engrave.nc
```

The strict acceptance status is `bounded_emitted_variable_v_engrave_pass`:
one exact sloped F300 cut, all nine ordered moves/events including F120
approach, F60 entry, F300 retract, setup return and T3/spindle events, one
`variable-v` stock prefix with two cone sweeps, and rest areas at depths
0/1/1.5/2/2.5 of 15.091265791880026 / 7.028684027518187 /
4.21460291488107 / 1.8062583920918873 / 0 mm2 within 1e-9 mm2. Any
missing/extra/reordered/changed item, low rapid, overcut or unsupported
post word fails. Record the actual NC SHA-256 and the user's preview report
before promoting Engrave to an executable carrier. If it fails, retain the
accepted script route and report the precise visual/operational split. The
Default post does not encode incoming machine position; tip
(-10,-10,+5) remains a declared setup assumption, and physical machining
is outside this probe.

The user completed the preview and post on 2026-09-24. The XZ toolpath
appeared to slope directly along the Pline with no extra pass. The actual
`V-variable-engrave.nc` SHA-256 is
`69367ec77654c9c2e005fd2db7dbc44c8c4d98bcacbc617a12b682e2cd926981`.
The audit command above returned `engrave_emitted_motion_deviation` (exit 1):
one exact sloped F300 cut and no other XY feed cuts, but eight posted items
versus nine required. The post rapids to +2.75 before plunging at F60,
rapids out of the cut to +5, and stops at (10,2,+5) instead of returning to
setup. It omits F120 approach and F300 feed retract. The
[review finding](REVIEW.md#bounded-xyz-engrave-cambam-post-finding---2026-09-24)
records the precise split. This closes the bounded probe: use this file for
visible path inspection only and the accepted `V-variable.cb` CustomScript
carrier for exact execution. No unchanged repost is needed.

### Bounded native V input normalization

The native source has the original XYZ finish spine, one Part stock and one
disabled Engrave targeting the original spine. `setup.json` explicitly supplies
the 90-degree cone dimensions and non-native setup controls. From the repository
root, create a new ignored directory with:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_variable_v output/native-variable-v-NEW
& $ProjectPython -m unittest tests.test_native_variable_v tests.test_variable_vcarve tests.test_variable_cone_script tests.test_variable_cone_engrave -v
```

To verify an edited native spine, stock or pointed tool for detached planning,
pass the source and its explicit setup without creating output candidates:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.native_variable_v --plan-only path/to/source.cb --setup path/to/setup.json
```

The planner strict-imports the native `.cb` and returns the canonical target,
cut, stock, tool, plan fingerprint and section rest. It accepts an increasing-X,
constant-Y, unbulged two-point XYZ target inside Part stock, a 90-degree cone
whose explicit setup radius/length match the source VCutter diameter, and a
strict interior cut interval. The fixed process/setup fields, disabled source
Engrave and explicit Value states remain required. Unsupported edits fail.
The result is planning evidence only; no CamBam post or physical setup is
accepted for the changed case. The detached family and its limits are in the
[specification](structure_spec.md#straight-variable-depth-v-planning-family).

The no-argument example builder still requires the original request. It keeps
the source bytes and makes separate `preview/V-variable-engrave.cb` and
`explicit/V-variable.cb`. The preview's enabled Engrave targets only the
generated sloped cut Pline; the explicit file's enabled Drill/CustomScript
targets only a Point anchor. The original finish spine and disabled source
Engrave remain in both files. Their `expected-motion.json` manifests pin the
source/candidate hashes and motion reference. A new native-derived candidate
does not inherit the acceptance of an older post merely because its planned
script is identical.

The prepared example is under `output/native-variable-v-20260924-02/`.
`source.cb` SHA-256 is
`ab97360d39640dc26da0cd65257ef2fec2d44e3afb279cc66f5a4cf700f45411`;
the preview `.cb` is
`04e1658da909ed6e1a3d93352243b2243287d12962e73b9889e6ba6bfba6f746`,
and the explicit `.cb` is
`edef7f1eb33e3dffcc5dbaf3f2cfda2c2ff65a96866fe532d1613ad3c5d6c376`.
The canonical plan fingerprint is
`4e0c0f4249c94da0f51a8fb4b38f9da718bb60fe12134fb391beb235c1fd0f46`;
the expected motion fingerprint is
`53867dca493fbc394dbaa3c49feea92f8527d1ed5146cc9dc8dd62aa2adf4244`.
Automated checks establish edited input normalization, independent section
rest and original-case file separation. Manual CamBam action adds no evidence
to the detached planning claim.

For the separate **native-derived emitted-output** gate, open
`explicit/V-variable.cb` in CamBam Plus 1.0 with **Default** postprocessor and
**Default mm** profile, generate toolpaths (Ctrl+T), then post (Ctrl+W) as
`explicit/V-variable.nc`. Leave the `.cb` and manifest unchanged. Do not use
the preview candidate for execution. Audit the actual CamBam-produced file:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.variable_cone_script output/native-variable-v-20260924-02/explicit/expected-motion.json output/native-variable-v-20260924-02/explicit/V-variable.nc
```

Pass is `bounded_emitted_variable_v_motion_pass`: nine exact ordered items,
including the F120 approach, F60 entry, sloped F300 cut from (2,2,-1.25) to
(10,2,-2.25), F300 retract and setup return; one `variable-v` prefix/two cone
sweeps; section rest at 0/1/1.5/2/2.5 mm of 15.091265791880026 /
7.028684027518187 / 4.21460291488107 / 1.8062583920918873 / 0 mm2 within
1e-9 mm2. Report the audit JSON and posted file SHA-256. Any deviation fails
this candidate's output gate. The declared initial tip position
(-10,-10,+5) and ideal-tool assumptions remain outside the post; this is no
physical machining acceptance.

The user exported this exact native-derived candidate on 2026-09-24.
`explicit/V-variable.nc` SHA-256 is
`969e0bcceb3556747aec2d2005ad7f91bd992915479933b38dd09181a25709bf`.
The audit command above returned `bounded_emitted_variable_v_motion_pass` with
all nine items, one `variable-v` prefix/two sweeps and the exact five rest
references. No repeat CamBam export is needed for this bounded gate.

### Bounded direct variable-depth V reference output

The direct writer consumes the same detached plan and verified process trace
as the accepted native-derived CamBam carrier. It emits an absolute millimetre
G0/G1 reference program without starting CamBam. To regenerate in a new ignored
directory from a native input, run:

```powershell
& $ProjectPython -m cambam_builder.integrations.direct_variable_v output/direct-variable-v-NEW --source output/native-variable-v-20260924-02/source.cb --setup output/native-variable-v-20260924-02/setup.json --compare output/native-variable-v-20260924-02/explicit/V-variable.nc
& $ProjectPython -m cambam_builder.integrations.direct_variable_v output/direct-variable-v-NEW/direct-evidence.json --compare output/native-variable-v-20260924-02/explicit/V-variable.nc
```

Omit `--source`, `--setup` and `--compare` for the equivalent standalone
request without a native comparison. The prepared program is
`output/direct-variable-v-20260924-01/direct-V-variable.nc`, SHA-256
`220faa9b9e7836ee5a80be263a6150371adee456375b314638d292ce678b86c5`.
Its `direct-evidence.json` pins source, setup, program, accepted CamBam post,
plan and motion fingerprints. The independent reader parses nine exact items;
posted-coordinate replay returns one `variable-v` prefix with two cone sweeps
and partial rest at 0/1/1.5/2/2.5 mm of 15.091265791880026 /
7.028684027518187 / 4.21460291488107 / 1.8062583920918873 / 0 mm2.
The direct and CamBam programs have different text but the same parsed event
and motion sequence. Any changed source/setup/program/comparison post,
unsupported command, altered role/feed/coordinate or stale manifest fails.
This verifies the bounded reference dialect only. It assumes initial tip
(-10,-10,+5) and does not select or certify a controller, real tool, material
or physical setup; do not treat it as a machine-ready program. Manual CamBam
validation adds no evidence to this headless output gate.

An edited member of the straight family uses an edited native source and matching
explicit setup. Build into a new empty ignored directory, then audit that exact
manifest without `--compare` unless there is a separately accepted post for the
same edited source and motion:

```powershell
& $ProjectPython -m cambam_builder.integrations.direct_variable_v output/direct-variable-v-edited-NEW --source path/to/edited-source.cb --setup path/to/edited-setup.json
& $ProjectPython -m cambam_builder.integrations.direct_variable_v output/direct-variable-v-edited-NEW/direct-evidence.json
```

The retained 14 mm spine / 8 mm cone example is under
`output/direct-variable-v-edited-20260924-01/`. Its ignored `generate.py` recreates
the edited native source and setup from the synthetic fixture, then builds and
audits `direct/direct-V-variable.nc`. The direct file SHA-256 is
`e5b33039303c53bcccf4b104e907b1bdc9835537482580cc6f427bc7f5710a69`.
The cut runs from (5,2,-1.0285714285714287) to
(15,2,-2.1714285714285717). Reparsed motion has nine exact items and two
cone sweeps. Its five target-minus-cut rest areas at depths
0/0.8/1.3333333333333335/1.8666666666666667/2.4 mm are
13.86847630534713 / 7.426634715521917 / 4.518313632820263 /
1.9880579984409152 / 0 mm2. This is partial target completion. The
comparison-post field is null because the accepted CamBam post belongs to the
original member; no CamBam export is required for this direct output gate.

### Bounded direct RC01 roughing/cleanup reference output

The direct writer consumes the exact nominal detached `rc01.generate(Job())`
trace. It writes one ASCII absolute-mm G0/G1 reference file, reparses all
2,945 emitted items and replays the parsed motion against the original stock,
island, tool components, process constraints and independent three-slab rest
bounds. From the repository root, with `$ProjectPython` set above:

```powershell
& $ProjectPython -m unittest tests.test_direct_rc01 tests.test_rc01 -v
& $ProjectPython -m cambam_builder.integrations.direct_rc01 output/rc01-direct-NEW
& $ProjectPython -m cambam_builder.integrations.direct_rc01 output/rc01-direct-NEW/direct-evidence.json
```

Use a new empty ignored directory for each generated run. The retained
`output/rc01-direct-20260924-01/direct-RC01.nc` has SHA-256
`390a6b31f961088a0224c957396a09c28b6dca4f5e2604af001b472911d5b8af`.
The manifest pins that file, nominal job and motion fingerprints. It reports
2,939 moves, partial target completion, rough rest per depth
[7.775010615955999, 7.787678472024001] mm2, and final rest per depth
[0.9214411294439999, 0.9263453527720001] mm2. The same intervals apply
to each open depth slab (-1,0), (-2,-1) and (-3,-2); rough/final volume
intervals are [23.325031847867997, 23.363035416072] and
[2.7643233883319995, 2.7790360583160005] mm3. The area coordinate
enclosure is 0.000000001 mm; the GEOS residual-location topology has no
formal numeric interval proof. A changed file, stale manifest or changed
parsed role/feed/coordinate fails the audit.

This is a strict reference dialect under the declared initial tip position
(-10,-10,+5) and initial coolant-off state. No controller or physical setup is
selected. Manual CamBam
validation adds no evidence to this headless output gate.

### RC01 native input and A/B/C comparison preparation

From the repository root, use a new unique ignored directory (the example name
must be changed for a later run):

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.rc01_adapter output/rc01-native-20260923-example
& $ProjectPython -m unittest discover -s tests -p test_rc01_native.py -v
```

The builder saves `source.cb`, `setup.json`, `A-rough.cb`, `B-explicit.cb`,
`C-native-cleanup.cb` and `comparison.json`. It strict-imports and normalizes
the source and every candidate before returning. The setup file is the explicit
non-native component/setup input, not CamBam style inheritance. Each candidate
retains the original Region, 50 x 40 x 10 mm Part stock at drawing origin
(-5,-5), and two disabled source Pocket MOPs with one target reference each.
`comparison.json` records source/candidate hashes, exact framework motion,
rough/final section and volume intervals, and pending emitted-motion status.

| File | Enabled candidate operations | Expected targets |
| --- | --- | --- |
| A | Three T1 Engrave MOPs, Z=-1,-2,-3 | 79 generated level-cut Plines per depth |
| B | A plus three T2 Engrave MOPs, Z=-1,-2,-3 | 248 T2 level-cut Plines per depth |
| C | A plus four T2 Pocket MOPs | Closed 7 x 7 mm Regions at the four specified corner windows |

The standalone A rough reference is 7.7750–7.7877 mm² per open depth slab;
the B final reference is 0.9214–0.9264 mm² per slab. Each of the three slabs is
1 mm high, so the corresponding volume intervals are three times those area
intervals. For C, the actual native cleanup must leave at most 1.358408 mm²
per slab and 4.075223 mm³ overall, with no protected overcut and at least
6.367258 mm² per-slab cleanup benefit. These are checks on posted motion, not
inferences from a displayed path or MOP property.

The candidate Engraves carry cut centerlines only. Their XML does not encode
framework approach, entry, retract, rapid or tool/spindle events; CamBam may add
or reorder those motions. This makes the A/B/C files comparison probes, not
accepted E/N output. The C Pocket settings are tool 2, diameter 2, stock
surface 0, target depth -3, increment 1, stepover 0.4, roughing clearance 0,
clearance plane +5, cut feed 300, plunge 60 and CW spindle 12000. The source
Pocket MOPs remain disabled in all three files.

The first user-posted trial is recorded in
[the review](REVIEW.md#rc01-first-cambam-output-trial---2026-09-23). Its B/C
posts remain under `output/rc01-output-20260923-130450/`; the original A post
was overwritten by a repeat export, so use its recorded first-trial hash and
findings. Those original posts cut to Z=-6 and are not accepted RC01 motion.
The repaired probes are under `output/rc01-repair-20260923-132551/`.
Their Engrave Plines lie at Z=0; each level MOP uses stock surface 0/-1/-2 and
target depth -1/-2/-3 respectively, with `OptimisationMode=None`. Preserve the
old artifacts for comparison. A repeat export is useful only for a focused
depth/order finding until an output carrier can express required approach,
retract and tool events; do not treat another A/B/C post as an E/N acceptance
request by itself. The repaired A was posted: the first plunge reaches Z=-1
and the minimum Z is -3, confirming the depth fix, but the first rapid goes to
(26.5981,9.5,+5) instead of (5,5,+5). The MOP's UUID-sorted target selection
explains that order. It also posts rapid approaches/retracts where RC01 requires
feed moves. See [the focused result](REVIEW.md#rc01-repaired-a-cambam-output-check---2026-09-23).
No additional Engrave export is requested until a carrier can encode the
required motion roles. B/C repaired variants have not been posted.

When CamBam Plus 1.0 validation is available, open each file, inspect the source
Region/Part and enabled MOPs above, regenerate toolpaths and post each separately
to `A-rough.nc`, `B-explicit.nc`, `C-native-cleanup.nc` in the same output
directory. Record the actual postprocessor and CAM style. The current reader is
scoped to CamBam's `Default` post with millimetres, absolute XY/XYZ coordinates,
G0/G1 straight moves, explicit G17/G21/G90, F/S/T, G40/G61/G64 and
M3/M5/M6/M30. It rejects arcs, cycles, cutter compensation and unknown modal
commands. It accepts the observed Z-only startup retract before G17/T1 but
flags the unencoded initial machine position. It also flags a tool change
without an explicit spindle stop. `G64` blending leaves trajectory deviation unverified even when listed
endpoints match. These are comparison-reader limits, not permissions to machine.

Run each returned post through the reader, for example:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.rc01_post output/rc01-native-20260923-example/comparison.json A output/rc01-native-20260923-example/A-rough.nc
```

Repeat with `B` and `C`. The result gives `sequence_matches`, `prefix_matches`,
`deviation` with first line/field, or `unverified` with the reason. A/B must
match every expected event and move within 0.001 mm with no extra motion before
E can proceed to independent stock replay; C checks only the T1 prefix, so N
requires a separate full T2 native-motion replay. Exact regenerated/post-added
entries, feeds, order, tool changes, shank/holder access, rest and overcut remain
the acceptance authority. Report I/E/N individually with the first deviation,
the actual postprocessor/style and the three `.nc` files. A CamBam open/display
pass alone establishes only native readability. No physical cutting is part of
this test.

### RC01 native Pocket roughing and corner-cleanup probe

Build a fresh ignored pair from the repository root:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.rc01_adapter output/rc01-native-mop-NEW --native
```

The command requires a new or empty directory. It writes `source.cb`,
`setup.json`, `N-rough.cb`, `N-native-cleanup.cb` and a SHA-256 guarded
`comparison.json`; it strict-reimports both candidates. `N-rough.cb` enables
one T1 native Pocket on the original Region. `N-native-cleanup.cb` adds four T2
native Pockets on the 7 x 7 mm corner windows in window order. Both retain the
two disabled source Pocket MOPs. The full-target T1 and window T2 operations
pin tool number/diameter, stock surface 0, target depth -3, increment 1,
stepover 0.4, clearance plane +5, zero roughing clearance, CW 12000 rpm,
plunge 60 and cut 300. These settings are intent, not removal evidence.

In CamBam Plus 1.0, open each candidate, regenerate toolpaths with Ctrl+T and
post with Ctrl+W using the **Default** postprocessor and **Default mm** profile.
Save `N-rough.nc` and `N-native-cleanup.nc` beside the candidates, without
editing the `.cb` files. Then run:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.rc01_native_post output/rc01-native-mop-NEW/comparison.json output/rc01-native-mop-NEW/N-rough.nc output/rc01-native-mop-NEW/N-native-cleanup.nc
```

The audit rejects changed candidates, a post header naming another file and
unsupported Default-post commands. Native mode accepts G0/G1 straight moves
and XY G2/G3 arcs with relative I/J centers, radius mismatch at most 0.001 mm,
and linear or helical Z. Explicit Engrave comparison remains straight-only.
The audit compares the posted T1 move prefix including arc centers, checks
motion/event and protected-target conditions, and computes radial-enclosed
rest intervals at all three slab bottoms. Arc flattening has 0.0001 mm maximum
chord sagitta; its radius mismatch and flattening error widen the bounds. It
reports rough/final area, cleanup benefit, residual outside the 0.05 mm
location envelope, exact single-prior-cut witnesses for T2 vertical access,
issue counts by kind and the first 24 findings per post. GEOS floating topology
is not formally enclosed. The Default post's startup position is absent from
G-code. The audit does not prove every stock-dependent link or axial/lateral
engagement; an issue-free result remains access-unverified and needs further
motion proof before RC01 N acceptance. Physical use has a separate gate.

The user-posted pair is under `output/rc01-native-mop-20260923-02/`.
Its candidate SHA-256 values are `e14afa4b5594814c754fe828898e0e13de5672c9d5bd55a4d470aec457f7ea6c`
and `23e44d90b3d0be46389f0a86b915dad42a578835e5ab4978f52c620c331bd2b1`.
`N-rough.nc` and `N-native-cleanup.nc` have SHA-256 values
`cde87d91d6f4c746cdabe7a80b04808d65551bc4d15db47e724550c6b0d444a2`
and `6c36c80766c84d8442cb28c6c1808da9f219dcb66d551333db951bf712109e20`.
The ignored `audit.json` contains the reproducible detailed result. The posted
T1 prefixes match; both outputs reach only Z=-3. Rough rest is
7.72558–7.72584 mm² per slab and final rest is 0.85840–0.85843 mm² per slab;
the area, benefit and location budgets pass within the stated numeric limits.
All four required RC01 corner columns and eight native T2 vertical locations
have exact full-depth T1 cut witnesses. **RC01 N still fails:** rapid approaches
and retracts below +5, ramped entries, low-level XY at F60, and a T2 change
away from setup without an explicit spindle stop violate the accepted motion
contract. Island tangencies remain numerically unresolved. See the
[trial evidence](REVIEW.md#rc01-native-pocket-posted-motion-trial---2026-09-23).

### RC01 Pocket role-carrier assessment

The local role trial can be reproduced in a new ignored directory:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.rc01_adapter output/rc01-roletrial-NEW --native --role-trial
& $ProjectPython -m unittest discover -s tests -p test_rc01_native.py -v
```

It prepares T1-only `R-rough.cb` and complete T1/T2
`R-native-cleanup.cb`, plus `source.cb`, `setup.json` and a hash manifest.
Enabled Pockets pin no spiral lead, no optimisation, cut-feed stepover and
zero crossover; the original Region and disabled source MOPs are retained.
Each file is strict-reimported before the builder returns. The prepared pair
is under `output/rc01-roletrial-20260923-1845/`; the manifest hashes are
recorded in the [review](REVIEW.md#rc01-pocketdefault-role-carrier-assessment---2026-09-23).

This is a diagnostic carrier assessment. It does not encode RC01's feed
approach/retract and exact setup/tool-change roles, so no CamBam post is
requested for it. The user performs any CamBam G-code generation from prepared
`.cb` files; return of the actual emitted `.nc` is required before a future
route can be audited. The subsequent role-bearing `.cb` carrier is documented
below; the route decision is in the
[plan](REST_MACHINING_PLAN.md#next-rc01-output-milestone-after-native-pocket-trial).

### RC01 literal-motion CamBam carrier

The agent prepares the complete T1 roughing plus T2 cleanup `.cb`:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.rc01_script output/rc01-script-NEW
& $ProjectPython -m unittest discover -s tests -p test_rc01_native.py -v
```

The current prepared candidate is
`output/rc01-script-lines-20260923-01/S-combined.cb` (SHA-256
`91cb871c22429c9a94bcea0f7cbbee0532fa462eb52e3d8558d69094fa6cf933`).
It retains the RC01 Region, Part stock and two disabled source Pockets. Its one
enabled `RC01 T1 rough plus T2 cleanup literal motion` Drill/CustomScript MOP
uses one anchor point at setup (-10,-10), tool 1, CW 12000 rpm, clearance +5
and exact-stop output. The 2,942 literal XML text lines encode both tool sections,
feeds, approaches, retracts and the T2 stop/change/restart. CamBam's wrapper
supplies the first T1 change/start and terminal stop. The ignored manifest
records the candidate hash and framework fingerprints. The source, script
and candidate strict-reimport; a synthetic wrapper replay passes, but only
the actual CamBam post establishes E output behavior.

The first posted file under `output/rc01-script-20260923-2115/` is retained
as failure evidence: CamBam preserved the previous `|` separators as literal
text on one NC line, so the reader rejected line 14. Do not repost that `.cb`.
The repaired candidate uses actual text newlines and has passed strict reimport
and the local synthetic post/audit checks. The user posted the revised `.nc`;
its 2,945 emitted items passed the exact-sequence and continuous RC01 replay.
See the [acceptance evidence](REVIEW.md#rc01-literal-motion-cambam-output-acceptance---2026-09-23).

The completed user CamBam action was to open the prepared `S-combined.cb` in
CamBam Plus 1.0 with the **Default** postprocessor and **Default mm** profile,
generate toolpaths (Ctrl+T), then produce G-code (Ctrl+W) and save
`S-combined.nc` beside the `.cb`. No repeat export is needed for this result.
The agent audited the post with:

```powershell
& $ProjectPython -m cambam_builder.integrations.cambam.rc01_script output/rc01-script-lines-20260923-01/comparison.json output/rc01-script-lines-20260923-01/S-combined.nc
```

The audit hash-guards the candidate, checks every posted event and move
against the framework program, rejects extra motion, and replays the actual
coordinates through continuous stock/access, process and three-slab residual
verification. CamBam may alter or omit literal script lines; a synthetic pass
alone cannot predict its actual emitted result. The Default post does not encode
the incoming machine position, so RC01 retains the declared test setup
(-10,-10,+5); physical acceptance is separate.

### RC01 selected native posted-stock replay

The reusable synthetic fixture under `tests/fixtures/rc01_native_stock/`
contains the original RC01 source, T1-only and T1/T2 native Pocket candidates,
both user-posted Default outputs, setup, comparison manifest and pinned rough
and paired evidence records. Run:

```powershell
& $ProjectPython -m unittest tests.test_rc01_stock_authority -v
& $ProjectPython -c "from cambam_builder.integrations.cambam.rc01_stock_authority import analyze_rc01_stock; print(analyze_rc01_stock('native_posted', evidence_path='tests/fixtures/rc01_native_stock/evidence.json')['rough_rest_by_depth'])"
& $ProjectPython -c "from cambam_builder.integrations.cambam.rc01_stock_authority import analyze_rc01_stock; r=analyze_rc01_stock('native_posted', evidence_path='tests/fixtures/rc01_native_stock/paired_evidence.json'); print(r['rough_prefix_identical'], r['coverage_budget_met'], r['motion_role_issue_counts'], r['stock_dependent_use'])"
```

Call `check_native_freshness(evidence_path, result)` before reusing a prior
native observation. `framework_generated` takes a supplied complete RC01
`Program` and verifies it separately; no selection fallback occurs. The
paired result retains both rough and final rest, exact T1 witnesses for the
actual and required T2 vertical columns, and full motion-role issue counts.
Its recorded coverage passes, but the 62 rough and 233 combined findings block
stock-dependent execution and native cleanup acceptance. Both native records
support analysis only, with no production-use claim. A new CamBam post or
edited `.cb` needs newly reviewed, explicitly pinned source/post provenance;
an unchanged repost adds no evidence.

### Native optimiser shape/MOP mapping corpus

The portable corpus lives in
`tests/fixtures/optimizer_corpus/`: four exact `.cb` inputs, four user-posted
Default `.nc` outputs, the original input `manifest.json`, and derived
`observations.json`. It no longer depends on ignored `output/`. The user
confirmed the Legacy (0.9.7) and New (0.9.8) MOP selections in CamBam Plus
1.0. The posted headers establish Default and G21/G90 for these files;
the manifest pins source XML and the expected installed system-file hashes.
The [dated review](REVIEW.md#native-optimiser-corpus-posted-output---2026-09-23)
owns the accepted observations and limits.
`manifest.json` is the unchanged generation snapshot, so its
`pending_native_post` labels are historical; the current output state is in
`observations.json` and the review.

Recheck exact source/post hashes, MOP sections, parsed modal motion, and the
checked-in observation map from the repository root:

```powershell
& $ProjectPython -m unittest tests.test_optimizer_corpus -v
$taskDir = Join-Path 'output' ('optimizer-corpus-audit-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $taskDir | Out-Null
& $ProjectPython -m cambam_builder.integrations.cambam.optimizer_corpus tests/fixtures/optimizer_corpus --observe --record "$taskDir/observations.json"
```

The test compares a freshly computed map with the tracked observation JSON;
the optional audit copy belongs only in its unique ignored task directory.
The original `output/optimizer-corpus-20260923-04/` remains local trial
history. Regenerating via `build()` makes new UUIDs and candidate hashes, so
new exports require their own posts and provenance; do not overwrite this
accepted fixture set. `G98`/`G81` CannedCycle words remain raw/unresolved in
the modal parser. Incoming machine position, controller execution and stock
removal are not certified by this mapping.

### Native MOP-series normalization and strategy selection checks

The bounded M4 building block consumes a strict native `.cb` source,
candidate and **actual** CamBam Default post. It binds the original source
primitive identity, analytic geometry and Part stock, ordered enabled MOP
sections, explicit cylindrical tool fields and declared millimetre/Default
setup when native XML omits those values. The parser retains posted arcs but
does not grant them stock authority. Linear XY motion can be lowered to
`cam_core.replay`; each prefix then receives bounded area, volume and
protected-overcut measurements for one source-bound rectangle or straight
Region. The strategy policy compares only complete, current, fully audited
stage chains and reports selected, partial or infeasible outcomes. A MOP-free
source permits document-title presentation edits when its original primitive
UUID/world geometry and Part stock are unchanged; sources with existing MOPs
still use exact-byte freshness. Candidate/post bytes always stay exact.

Run the focused gate from the repository root:

```powershell
& $ProjectPython -m unittest tests.test_native_series tests.test_native_series_audit tests.test_strategy_selection -v
```

The native-series test also normalizes the retained actual M1 two-Pocket post
under its separately recorded Default-mm setup. This proves ordered post
reading, not that the M1 native route is safe: its arcs and 12 previously
observed T2 entries still block a generic linear replay certificate. The
synthetic two-MOP linear post test proves full parser-to-stock-to-selector
behavior; it is constructed test data, not new CamBam acceptance. No new
manual CamBam validation adds evidence for the pure selector or this bounded
normalizer because the actual M1 post is already retained. Reopen the linear
certificate boundary when an actual, source-bound candidate with supported
linear motion and safe entries is available. A curved/rounded-tip combined
edited job and parsed direct reference output remain the full M4 acceptance
gate.

The 2026-09-25 packaging gate built both wheel and sdist under
`output/m4-series-20260925-03/`, installed the local wheel without dependencies
under `output/m4-wheel-smoke-20260925-02/installed`, then imported
`cam_extensions.strategy`, `integrations.cambam.native_series_audit` and
`cam_core.curved_region` after changing the interpreter's working directory
outside the checkout. All three module paths resolved inside the installed
wheel target. This checks package inclusion and import, not a clean dependency
resolution or all supported Python versions.

### M4 edited curved rounded-tip comparison and output gate

`output/m4-edited-curved-20260925-02/` is the retained validation bundle.
`create_source.py` reopens the accepted M3 annulus source, enlarges its analytic
circular hole from radius 2 to 2.1 mm while preserving the Region UUID,
and writes `edited-source.cb`. The outer radius is 9 mm; Part stock remains
X/Y `[-12,12]` mm, thickness 4 mm and top Z=0. The nested `comparison/`
contains exact copies of the edited source and an explicitly synthetic,
source-bound T1 proof trace. This T1 raster is a stock fixture, not a
recommended roughing recipe. Each `raster/` and `offset/` child has native
preview/explicit `.cb` candidates, `expected-motion.json`, and its own
parsed, stock-replayed `direct-reference.nc`. The complete T1-only
`comparison/endmill-direct.nc` is the partial alternative. The offset fill follows nested
contours around both annulus boundaries; the raster fill uses horizontal rows.
The direct output is a strict millimetre reference dialect, not controller NC.

From the repository root, run the focused implementation gate and recheck all three
retained direct files:

```powershell
& $ProjectPython -m unittest tests.test_m4_curved_workflow tests.test_v_region tests.test_native_v_region tests.test_strategy_selection -v
$Comparison = 'output/m4-edited-curved-20260925-02/comparison'
& $ProjectPython -c "from cambam_builder.integrations.m4_curved_workflow import audit_direct, audit_endmill_direct; p='output/m4-edited-curved-20260925-02/comparison/comparison.json'; print(audit_endmill_direct(p)['status']); print(audit_direct(p,'raster')['status']); print(audit_direct(p,'offset')['status'])"
```

For actual CamBam Plus 1.0 acceptance, open each of the four native files:
`comparison/raster/preview/m3-preview.cb`,
`comparison/offset/preview/m3-preview.cb`,
`comparison/raster/explicit/m3-explicit.cb`, and
`comparison/offset/explicit/m3-explicit.cb`. The two previews should visibly
show the 9/2.1 mm annulus and, respectively, horizontal rows and concentric
offset rings. The explicit Drill/CustomScript toolpath display may omit literal
cuts, as in M3; its complete NC is the execution evidence. Generate toolpaths
and export each with **Default** post / **Default mm** profile to `m3-preview.nc`
or `m3-explicit.nc` beside its `.cb`. Use separate files; do not overwrite the
M3 accepted posts. Report whether both visible previews match those patterns,
and provide the four exported NC files. Audit them with:

```powershell
& $ProjectPython -m cambam_builder.integrations.m4_curved_workflow audit `
  "$Comparison/comparison.json" `
  --source 'output/m4-edited-curved-20260925-02/edited-source.cb' `
  --raster-preview "$Comparison/raster/preview/m3-preview.nc" `
  --offset-preview "$Comparison/offset/preview/m3-preview.nc" `
  --raster-post "$Comparison/raster/explicit/m3-explicit.nc" `
  --offset-post "$Comparison/offset/explicit/m3-explicit.nc"
```

Pass requires both `m3_v_preview_centerlines_match`, both
`bounded_m3_v_post_pass`, both `bounded_m4_direct_pass`, the
`bounded_m4_endmill_direct_pass` baseline, and a `selected`
strategy. The raster/offset previews contain 25/8 paths and 307/283 cut
segments on this edited source. The direct programs contain 450/375 ordered
items. Their Z=-1 mm final residual upper bounds are 1.52633/1.53003 mm²,
with eight-slab volume upper bounds 42.676/41.509 mm³; nominal protected
overcut must remain zero. The endmill-only prior remains partial at Z=-1
with 103.32511 mm² upper residual. Any extra/missing move, shallow crossover,
changed source/candidate bytes or stock result fails. The test suite's
constructed Default posts validate the reader and selector only. The user
subsequently supplied all four actual exports and the preview observation;
their result follows. Physical/controller acceptance belongs to M5.

**2026-09-26 actual CamBam Plus 1.0 acceptance.** The user saw the source
primitives and Engrave toolpaths directly on the generated Plines in both
preview documents. All four actual Default-mm NC files pass their separate
source-bound audits. The direct T1-only, raster and offset programs still
return `bounded_m4_endmill_direct_pass` and two
`bounded_m4_direct_pass` results. Use the audit command above to reproduce:

Both rounded routes pass zero nominal protected overcut, the 2 mm² area and
80 mm³ volume upper budgets, and the same edit-aware selector. Its area-first
policy chooses `rounded_raster`; `rounded_offset` remains a fully audited
feasible alternative. The endmill-only direct route is safe but partial.
The result accepts this bounded edited annulus workflow; it does not certify
native Pocket planning, a controller dialect or physical machining. The
[dated acceptance](REVIEW.md#m4-edited-curved-actual-output-acceptance---2026-09-26)
retains the four exact NC hashes and area/volume evidence.

### M5 UCCNC output and automatic evidence

The first bounded split-file adapter is executable. From the repository root,
use a **new** ignored directory for each build; the recorded synthetic bundle
is `output/m5-uccnc-20260926-02/`:

```powershell
$ProjectPython = '.\.venv\Scripts\python.exe'
& $ProjectPython -m cambam_builder.integrations.uccnc_m5 output/m5-uccnc-NEW --m4-manifest output/m4-edited-curved-20260925-02/comparison/comparison.json
& $ProjectPython -m cambam_builder.integrations.uccnc_m5 output/m5-uccnc-NEW/handoff.json
& $ProjectPython -m unittest tests.test_stockless_native tests.test_native_series tests.test_uccnc_m5 -v
```

The build writes `T1.nc`, `T3.nc` and a versioned `handoff.json`. The audit
command rechecks source/prior/plan and setup lineage, exact file hashes, both
decoded programs and chained stock. Its first profile requires G54, G21, G90,
G17, G61, G40 and G49; each file identifies its installed tool in a comment,
sets S12000/F60/F300 in its move stream, and ends with M5/M30. Neither file
installs or measures a tool. The recorded audit has 61 T1 plus 383 T3 moves,
30 T1 cuts, Z=-1 final remaining-area upper 1.526323 mm2 and final volume
upper 42.675342 mm3. Treat the emitted parameters and files as synthetic test
data, not as a production machine setup. The manifest reports runtime and
physical setup `not_evaluated`. The independent reader rejects unknown G/M/T
words and the auditor rejects missing/reordered/stale files or handoff values.
The [dated evidence](REVIEW.md#m5-stockless-and-uccnc-split-output---2026-09-26)
records exact hashes and limits.

For the separate **native stockless acceptance**, the prepared
[source](../output/m5-stockless-acceptance-20260926-01/stockless-source.cb)
and [framework round trip](../output/m5-stockless-acceptance-20260926-01/framework-roundtrip.cb)
both have zero Part/MachiningOptions Stock nodes and Profile `StockSurface=4.5`,
`TargetDepth=2.0`, `ClearancePlane=8.0` with `Value` states; XML parsing
confirmed these exact values. The user accepted the stockless source in CamBam
Plus 1.0 and supplied its fresh
[Default post](../output/m5-stockless-acceptance-20260926-01/framework-roundtrip.nc).
The native-series reader parsed its complete bounded command set without a
datum shift: one Profile/T1 stage, 45 moves, ordered feed descent endpoints
Z4.5 through Z2.0 and a final Z8 retract. Repeat the parser audit from the
repository root with the declared interpreter:

```powershell
& $ProjectPython -c "from pathlib import Path; from cambam_builder.integrations.cambam.native_series import normalize_native_series; d=Path('output/m5-stockless-acceptance-20260926-01'); s=normalize_native_series(d/'framework-roundtrip.cb',d/'framework-roundtrip.cb',d/'framework-roundtrip.nc',initial_position=(0,0,8),setup={'units':'mm','postprocessor':'Default'}); print(s.parsed_evidence())"
```

The `(0,0,8)` mm initial tip is an **assumption**, because the post does not
encode the machine's initial position. Parsing reports stock/access/residual
`not_evaluated` without a separate trustworthy initial-stock model and replay
certificate. The post's `T1 M6` has no certified controller or physical
effect here. This native acceptance is separate from the automated UCCNC pair;
the [dated record](REVIEW.md#m5-stockless-actual-cambam-post-acceptance---2026-09-26)
owns hashes, exact values and reopening criteria.

The first slice followed the [implementation packet](REST_MACHINING_PLAN.md#m5-implementation-packet)
and [mediation invariants](structure_spec.md#mediation-invariants-and-evidence-contract).
The original stockless Part persistence defect is recorded with
synthetic files and a replay command in the
[architecture review](REVIEW.md#m5-mediation-architecture-review---2026-09-26).
The new stockless tests inspect saved XML for **absence of Stock**, in addition
to MOP values. Run `test_mop_parameters.py`, `test_mop_roundtrip.py`,
`test_native_series.py`, affected copy/clone/transfer tests and the full suite
for shared Part/persistence changes.

The user selected UCCNC for production. The [M5 evidence contract](REST_MACHINING_PLAN.md#m5-controller-coverage-and-automatic-evidence---2026-09-26)
keeps the plan and verifier controller-neutral, with one declared dialect/setup
per output adapter. The accepted source and first selected route remain under
`output/m4-edited-curved-20260925-02/comparison/`; its raster
`direct-reference.nc` has 450 ordered events/moves from a declared
(-17,-17,+5) mm initial tip. It is an oracle for comparison, not a UCCNC file.

For the first Windows-side UCCNC gate, create a unique ignored
`output/m5-uccnc-.../` bundle with **T1 and T3 files** and an ordered handoff
manifest. Start from the accepted M4 rounded-raster source/plan. In this
synthetic setup, declare G54, metric absolute XY-plane exact-stop motion, no
active tool-length compensation, one unchanged XY datum and work Z=0 at the
physical stock top for each installed tool. The selected M4 path is already
programmed with its intended surface at Z=0, so the CAM-to-work Z map is
identity; its stock model also happens to have a top at Z=0. This models the
user's manual surface-touch-off method as a **per-tool work-coordinate setup**, without
assuming that the touch-off writes a tool-length table entry. Pin the safe
initial tip `(-17,-17,+5)` mm, stock/fixture identity, tool geometry,
required T1-to-T3 order, source, prior, profile and output hashes. The test
setup is synthetic and must not be copied to a machine. Each file must set
its own modal state, run one tool's motion, stop the spindle and end without
calling `M6`. Its installed tool and registered tip datum are explicit handoff
preconditions; the file cannot verify that the operator actually touched off.
Decode both final NC files with a separate strict reader, compare all ordered
events/coordinates including setup and end roles, and replay the decoded T1
motion into T3's initial stock, then replay decoded T3 motion too. Do not reuse
the M4 audit's planned V paths to certify rounded or transformed output.
Apply the M4 stock,
access, rounded-cutter, residual and volume checks to the chained job.
Reject controller commands, transforms or handoff states whose effects are
unknown. This automated gate is the precise whole-program check; no manual
visual comparison of hundreds of moves is requested.

Before generalizing the output adapter, import a CamBam file with **no stock
object** and explicit operation `Stock Surface`/`Target Depth` values. Preserve
its XML states and compare a fresh post without inventing a Z shift; missing
stock must not make import fail. Report stock-dependent material/access results
as not evaluated until a stock
model is supplied. Check separately that a stock-object edit does not rewrite
explicit MOP Z values; if either MOP field is `Auto`, resolve its inherited or
stock-dependent value by CamBam semantics or an actual post before comparing
motion. CamBam documents this [Auto dependency](https://www.cambam.info/doc/1.0/cam/machining-options.html).

Use a distinct **explicit datum-map** fixture to test translation: declare a
resolved program surface at Z `+4.5 mm` and a controller setup that touches
the same physical surface to work Z=0. Only this declared mismatch calls for
a `-4.5 mm` post shift; decode and invert the shift to compare with the
original path. A stock object's top value alone must never trigger it. Reject
a stale per-tool datum, a path that violates a declared mapping, or a
double-applied touch-off/tool-length correction when claiming verified output.
A separate declared tool-table/preset fixture should retain a fixed work
origin and apply its controller length offset. These are setup alternatives,
not changes to the imported CAM program.

`C:\UCCNC\UCCNC.exe` is installed locally. CNCdrive documents an unlicensed
Windows demo mode, but the installed `Profiles\Macro_Default\M6.txt` is an
example automatic changer that commands `G53` movements and hardware actions.
Do not run the M4 two-tool file with that macro as a harmless test. A separate
demo load may check UCCNC compatibility only after a safe isolated profile is
prepared; a screenshot or a 25 Hz position sample is not exact path evidence.
The installed plugin sample has no confirmed export of every interpreted move.
No UCCNC motion-output file is expected from the user. Keep exact UCCNC
runtime parity `not_evaluated` unless a documented or tested per-move trace
route appears; do not make that research a prerequisite for the portable M5
emitted-program gate. The framework audits its own final NC bytes before they
reach any controller. A demo load can add a bounded compatibility observation,
but sampled screen positions cannot replace complete decoded-motion evidence.
The user's manual `M6` profile is a possible one-file implementation after
its exact pause, offset and resume behavior is pinned. A controller-specific
`M0` stop block between MOPs is another implementation; its script and
post-stop tool/offset state must be decoded and verified before use. Neither
command is the framework's tool-change API. A transition policy chooses the
operator or automatic changer, one or several programs, and the measurement
and offset method; the dialect adapter owns command syntax and configuration.
An automatic changer fixture needs declared tool, tip, offset and extra-motion
effects, with physical changer acceptance kept separate. CamBam Parts can
group MOPs by tool and post separate files, but their actual posts still need
the same ordered file and chained-stock audit. Do not silently treat a stop,
macro or file boundary as verified tool installation. The first gate should
reject requests for unverified policies and include an explicit failure test
for a missing or altered second file or mismatched handoff setup.
After the split-file gate, use the same source-bound plan for a bounded
one-file manual-stop policy, a declared automatic-changer effect fixture, and
a second named controller dialect, Grbl v1.1, selected because
its documented commands include `M0` but omit `M6`; a LinuxCNC fixture can
exercise its configured manual/automatic `M6` and separate `G43` rule. Decode
each emitted stream and its handoff events, then run the same whole-job stock
audit. Reject altered stop blocks, unmodeled changer motion and any transition
without a declared post-change tool, offset and resume state. This is the M5
proof of policy and dialect selection; no synthetic fixture certifies a
physical changer or the user's UCCNC macro.

LinuxCNC's optional command-line `rs274` interpreter can provide an additional
machine-readable check for a shared command subset if a Linux runner is later
provisioned. No Docker executable or accessible WSL distribution was found in
this Windows environment, and LinuxCNC cannot stand in for UCCNC-specific
macro or offset behavior. Actual UCCNC production limits, offsets, tools,
workholding and M6 behavior remain separate inputs from the user's machine.

### M5 Grbl portability and transition fixtures

The [recorded synthetic bundle](../output/m5-portability-20260926-01/handoff.json)
contains `manual.nc`, `mixed.nc`, `mixed-changer.json` and a pinned handoff.
Build into a **new** ignored directory, or audit the recorded one, from the
repository root with the declared interpreter:

```powershell
$ProjectPython = '.\.venv\Scripts\python.exe'
& $ProjectPython -m cambam_builder.integrations.m5_portability output/m5-portability-NEW --m4-manifest output/m4-edited-curved-20260925-02/comparison/comparison.json
& $ProjectPython -m cambam_builder.integrations.m5_portability output/m5-portability-NEW/handoff.json
& $ProjectPython -m unittest tests.test_m5_portability tests.test_uccnc_m5 -v
```

The adapter reads the same detached M4 rounded-raster plan and synthetic T1
prior as the UCCNC pair. The separate strict Grbl reader consumes every byte of
each whole program, including G21/G90/G17/G94/G61/G40/G49/G54 startup,
per-stage spindle, feed, G0/G1, G43.1 or G49 length state, M5, M0 and M30.
The [official Grbl v1.1 supported-code list](https://github.com/gnea/grbl/blob/master/README.md)
includes `M0` and dynamic `G43.1`, but no `M6`; the
[realtime command guide](https://github.com/gnea/grbl/blob/master/doc/markdown/commands.md)
documents cycle start/resume after M0. In this fixture, stage comments are
identities for the offline auditor, not controller tool-install commands.
Each M0 requires the declared external completion and safe-tip state before
resume. Unknown blocks, altered pause/offset state or undeclared motion block
the offline claim. `$32=0` spindle mode, a fixed G54 work origin and an RPM
range containing 12000 are setup assertions, not measured machine settings.

`manual.nc` tests one in-program **operator** T1-to-T3 handoff. Its synthetic
resolved CAM surface is +4.5 mm while the work surface is zero; the explicitly
declared -4.5 mm CAM-to-work Z map is inverted over all 444 decoded T1/T3
moves before the same source-bound stock audit. An unchanged or double-shifted
emitted path fails. `mixed.nc` tests manual T1-to-T3 and modeled automatic
T3-to-T1 transitions in one program, with fixed G54 and externally supplied
tool-table lengths 2/3/2 mm expressed via Grbl `G43.1`. Grbl has no native
tool-table or automatic changer claim here: the table and changer belong to
the synthetic host setup. The strictly read `mixed-changer.json` declares
three safe, spindle-off tip-travel segments, T3/T1 installation and length
registration; the last T1 stage contains two separately decoded safe rapid
moves. Its completion and return state are assumptions. This reader compares
against the same pinned effect model used by the fixture writer; it does not
independently derive arbitrary changer motion or compose machine/tool offsets.
Length values are checked against the table and each stage's registered tip is
assumed. The checker rejects
changed or missing effect bytes, even if the handoff is edited, and does not
infer installation from `M0` alone.

Both fixtures replay decoded T1 stock into decoded rounded T3 motion. Each has
61 T1 moves, 383 T3 moves, 30 T1 cuts, Z=-1 mm final remaining-area upper
1.526323 mm2 and eight-slab final volume upper 42.675342 mm3; the mixed
fixture also has two safe T1 return moves. Runtime parity and physical setup
are `not_evaluated`. The report's phrase "complete decoded Grbl program and
transitions" is conditional on these fixed transition assumptions; it is not
general machine-state simulation. No manual sign-off or new G-code is required
to close the named offline fixture gate. The
[engineering review](REVIEW.md#framework-direction-and-engineering-acceptance---2026-09-26)
records the remaining native order/edit case and reusable state-model work.
Use the user's actual controller, sender, offset and changer configuration
only after a separate production profile and machine acceptance exist.

### Isolated planar backend evaluation

The original Shapely experiment remains development-only. Its runners do not
change runtime dependencies; the later runtime slice above separately owns the
optional `planar` extra. Use [the corpus/decision owner](REST_MACHINING_PLAN.md#shapelygeos-evaluation-decision---2026-09-22)
for acceptance meaning and [the dated evidence](REVIEW.md#shapelygeos-planar-evaluation---2026-09-22)
for the tested Windows x64 versions and limits. From the repository root:

```powershell
$taskDir = Join-Path 'output' ('shapely-evaluation-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $taskDir | Out-Null
foreach ($version in @('3.9', '3.10', '3.11', '3.12', '3.13')) {
    $tag = 'py' + $version.Replace('.', '')
    $candidate = if ($version -eq '3.9') { 'shapely==2.0.7' } else { 'shapely==2.1.2' }
    uv --cache-dir "$taskDir/cache" venv --python $version "$taskDir/$tag"
    if ($LASTEXITCODE -ne 0) { throw "Environment failed: $version" }
    uv --cache-dir "$taskDir/cache" pip install --python "$taskDir/$tag/Scripts/python.exe" --only-binary :all: $candidate 'numpy>=1.23.5'
    if ($LASTEXITCODE -ne 0) { throw "Install failed: $version" }
    & "$taskDir/$tag/Scripts/python.exe" tools/evaluate_shapely.py --output "$taskDir/$tag-geometry.json"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $version" }
}
```

Expected per interpreter: 11 planar cases pass, T01 is `expected_limitation`, and
C01/C02 are `out_of_planar_scope`; `unexpected_failures` is zero. Exit zero means
the bounded experiment behaved as recorded, not that every desired operation is
supported. Reports retain individual scalar errors, topology, boundary bounds,
evidence classes and corpus/runner hashes. Keep the reports in that task directory.
The dense bidirectional distance check intentionally costs more than the geometry
operations; do not interpret total runner time as backend throughput.

For packaging/coexistence verification, also build with the declared `uv build`
command, install the resulting project wheel into each candidate environment with
`uv pip install --python <candidate-python> --no-deps <wheel>`, and run an isolated
`-I` import/construct probe checking that modern/legacy imports resolve inside
`sys.prefix`. Inspect Shapely's `WHEEL`, bundled native-library versions and license
files. This is a bounded dependency compatibility check, not the full packaging
regression matrix above. Do not add Shapely to project metadata merely to run it.

For the subsequent adversarial and internal-contract gates, reuse those isolated
interpreters and create a new task directory for reports:

```powershell
$acceptanceDir = Join-Path 'output' ('planar-adversarial-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $acceptanceDir | Out-Null
foreach ($tag in @('py39', 'py310', 'py311', 'py312', 'py313')) {
    & "$taskDir/$tag/Scripts/python.exe" tools/evaluate_planar_adversarial.py --output "$acceptanceDir/$tag.json"
    if ($LASTEXITCODE -ne 0) { throw "Adversarial acceptance failed: $tag" }
}
```

Here `$taskDir` is the environment directory created above; when reusing an
existing evaluation, set it to that directory first. Expect zero
`unexpected_failures`. The report includes all checked values/limits, actual
Python/Shapely/GEOS versions and both runner hashes. Its helpers prototype strict
polygon admission and analytic rectangle/circle center classification, not a
runtime API. The [internal contract](REST_MACHINING_PLAN.md#internal-planar-value-and-error-contract)
owns result/error semantics; the [adversarial review](REVIEW.md#adversarial-planar-acceptance-and-internal-contract---2026-09-22)
owns measured evidence and unsupported cases. Run the backend-independent corpus
reference tests too. No CamBam manual check adds evidence to this detached design
increment, and no runtime or packaging changes are implied by these commands.

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

The pure milling formula and constraint kernel has a focused regression command:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_machining_calculations.py -v
```

It checks formula inverses, official metric/imperial worked examples, unit-system
equivalence, partial and conflicting inputs, separate axial/radial engagement,
machine-cap propagation and invalid numeric boundaries. This kernel does not read or
write CamBam files, so a manual CamBam display/toolpath check adds no evidence.

Recommendation selection, operating-range composition and integrated candidate
planning have adjacent focused regressions:

```powershell
& $ProjectPython -m unittest tests.test_machining_recommendations tests.test_machining_planning tests.test_machining_calculations -v
```

These tests cover fixed-value precedence, strategy permutations, machine/job range
intersection, RPM/feed coupling, metric/imperial feed units, entry-feed bounds and
downstream achieved-load diagnostics. They are pure, nonserialized calculations;
manual CamBam validation adds no evidence.

Canonical Pline/Points vertex records have a focused regression command:

```powershell
& $ProjectPython -m unittest discover -s tests -p test_vertex_records.py -v
```

It checks `Vertex` defaults and keyword-only bulge, XY/XYZ tuple shorthand,
four-tuple and malformed-input rejection, Points bulge rejection, collection
insertion/reordering, same-code-version pickle snapshots and two XML round trips. The shape
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
user artifacts. Never load an untrusted pickle or one from another framework code
version: `load_state` is an unsafe WIP/cache resume convenience, not migration or
exchange. Use `.cb` XML for the supported external boundary.

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
and the `rotated-child` parent link automatically. The fixed CamBam baseline above
applies. Report A/B pass or fail and any displayed outline, vertex or matrix
difference.

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
displacement, scaling or rotation. The fixed CamBam baseline above applies. Report
A/B pass or fail and any differing layer/observed endpoint. This accepts the
synthetic display/XML slice only; no MOPs or toolpaths are included. Parent-cycle
rejection and registry integrity are automated API checks and need no manual
reproduction.

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
(resetting to identity must not move them). The fixed CamBam baseline above applies.
Report pass/fail and any differing endpoint or matrix. No toolpath validation is
needed for this case.

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
than B's local point properties. The fixed CamBam baseline above applies. Report
pass/fail and the layer and observed coordinates for any mismatch. No toolpath
checks are needed.

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
The fixed CamBam baseline above applies. Report A/B pass or fail and any differing
operation/property.
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
- Optional same-code-version state snapshots accept `project.pkl` in the current
  directory. Parent-directory creation failures raise; pickle writes remain
  non-atomic and no old-field/layout defaults are supplied.
- Consult [review evidence](REVIEW.md) before retrying a known approach.

### Export and persistence regression checks

```powershell
& $ProjectPython -m unittest discover -s tests -p test_export_failures.py -v
& $ProjectPython -m unittest discover -s tests -p test_state_persistence.py -v
```

These synthetic tests check exception propagation, destination preservation,
temporary cleanup, successful XML content/identity/targets and current-code pickle
project-link restoration. They do not test old snapshot compatibility.
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
bootstrap and, after the first accepted request, a `CAMBAM_MCP_PROTOCOL` record
containing the actual version and `legacy`/`modern` mode. Restart loses all open
handles, unsaved edits and retry records.

Thirty-seven tools create, open/import, list/inspect, save/export and close documents; add root
Rect, Circle, Arc, Pline, Points, Text and Region primitives; atomically replace existing
same-layer root Rect/Circle/closed-Pline contours with one Region; add explicit
Profile, Pocket, Engrave and Drill MOPs and replace MOP targets; link/group
and same-document copy primitives; copy or transfer a subtree between two
distinct open documents under per-document revisions; and translate, rotate,
uniformly scale, mirror, Z-shift or bake those supported root primitives.
Inspection reports
typed world geometry/bounds and per-kind machining parameter records for the
similarity slice, while diagnosing other entity detail as unsupported. Saves
only create new `.cb` paths under the workspace and never overwrite. File
parent directories must already exist. Units are assertions, not conversions
or a verified CamBam units setting. Read the
[state and safety contract](MCP_CONTRACT.md) before client use.

For a client-local `.cb` file on a shared filesystem, first call `document_list` to
obtain `workspace_path`. Binary-copy the file under a fresh safe `.cb` name in that
directory, verify source/copy SHA-256 equality, then call `document_open` with the
relative staged name and `expected_sha256`. Do not pass a client path to
`document_open`, reuse an older staged filename, or reconstruct XML in model text.
Across hosts, read the complete UTF-8 XML and call `document_import` with
`source_name`, `units`, `content` and (when known) `expected_sha256`. After edits,
call `document_export` with the
current revision and a suggested leaf filename, write the returned `content`
unchanged to the desired client-local path, and verify its SHA-256 when possible.
Both directions accept up to 10 MiB of XML. Inline results are intentionally the
compatibility baseline; large results may consume substantial model context.
This content flow is the portable fallback. On a shared filesystem, `document_save`
can create a non-overwriting exact-byte handoff artifact in the MCP workspace. Copy
its returned `absolute_path` to the client-local destination with a deterministic
binary copy and verify bytes and SHA-256. The live handle remains the synchronized
working copy until the durable file changes. Do not treat the server path as the
client destination or pass a client path to `document_save`.

### Human/AI follow-up and session recovery

For consuming project folders, start from the packaged
[`consumer_AGENTS.template.md`](../cambam_builder/mcp_adapter/consumer_AGENTS.template.md).
Its one-time bootstrap gathers project preferences and replaces itself with daily
instructions. Those project-context defaults are intentionally separate from MCP
protocol guarantees and this repository's development `AGENTS.md`.

The literal `FIRST_RUN_SETUP: PENDING` marker is a gate, not a heading for an agent to
silently skip. Each unresolved durable policy is a separately displayed onboarding
question; if the client limits questions per dialog, setup continues in another dialog
rather than bundling fields. The agent saves the answers into the copied `AGENTS.md`,
marks setup complete, and only then resumes the preserved first task. Current-file
names and per-job geometry, stock, material, tool, feed, speed, depth and nesting
choices belong to that task rather than onboarding.

Overwrite/backup/Save-As behavior and edit-version naming are one file-history policy,
not separate onboarding questions. The recommended option is versioned Save As unless
the consuming project establishes another convention. "Validation" in this template
means MCP result/inspection checks, delivery-hash verification and any configured human
CamBam/toolpath review. It does not expose or prescribe this repository's development
test suite. A consuming agent may write a deterministic local helper for difficult
geometry or planning calculations, but must author through MCP tools rather than use a
script to generate or patch `.cb` XML.

The review preference is a post-save handoff policy, not a completion gate. The agent
first performs available MCP checks, saves and hash-verifies a reviewable project-local
revision, then reports it ready for the configured optional or required human CamBam
review. Required human confirmation gates only acceptance/production-ready claims;
feedback begins another edit/save/review cycle. It never prevents initial delivery.

Onboarding discovery is deliberately shallow. A consuming project may contain a
nested source checkout, but that checkout's tests are not evidence about a generated
CamBam artifact. Likewise, describing a requested document as a test, demo or example
does not waive the no-invented-machining-input rule: a non-production fixture set must
be proposed, labeled and confirmed for that task before enabled MOPs are created.

Treat the client-local `.cb` file as the durable shared artifact and an MCP handle
as one server-process working snapshot. Keep the last successful file SHA-256,
handle, boot ID and successfully written export revision when the client can do so.
Before a later AI edit—most
importantly after the user has opened and saved the file manually—reread the complete
file and compare its hash with the last imported or exported-and-written hash.

- If the file changed on a shared filesystem, binary-copy it under a fresh name in
  `document_list.workspace_path`, verify the copy hash, and call `document_open`
  with that hash as `expected_sha256`. Across hosts, call `document_import` with its
  complete current UTF-8 content, leaf `source_name` and hash. Continue from the
  returned new handle at revision 0; do not apply the manual edit to an older handle
  or fall back to a similarly named workspace artifact. If the older handle contains
  MCP mutations newer than the last export actually written to the durable file,
  preserve both versions, export the MCP candidate under a different client-local
  name, and ask the user which changes to keep or merge.
- If conversational context was lost or the client reconnected, call
  `document_list`. Reuse a handle only when it is listed under the current boot and
  the durable file hash is still the expected one. A server restart normally returns
  an empty list and requires staging/opening or importing the durable file again.
- If a mutation returns `STALE_REVISION`, call `document_inspect` without
  `expected_revision`, review the current contents, and retry only if the requested
  change still applies. Use the reported revision and a new `request_id`; failed
  request IDs replay their original terminal result.
- Never launch multiple mutations concurrently against one document. Every successful
  mutation advances its revision, so await each result and pass that returned revision
  to the next mutation with a fresh request ID. Read-only calls and mutations against
  different documents may still run concurrently.
- After cross-host `document_export`, write `content` unchanged and verify `sha256`.
  For a same-host `document_save` handoff, binary-copy rather than reconstructing
  XML and verify the destination hash. Retain that hash as the synchronization point;
  if the destination later changes, stage/open it as above. A valid UUID accidentally supplied to a
  read-only list/inspect/export/planning call is ignored and does not create retry
  state.

The source hash shown by `document_list` is the originally imported/opened byte
snapshot. A revision greater than zero means MCP has mutated that in-memory document;
export it to obtain the current serialized content and hash. Because the server
cannot see a client-local path, it cannot detect a manual file save without the
client hashing it and staging/opening or importing that content.

The required identity fields for that flow are:

```text
document_list:    {workspace_id} -> workspace_path plus live snapshots
document_open:    {workspace_id, new request_id, units, fresh staged path, expected_sha256}
document_import:  {workspace_id, new request_id, units, source_name, complete content, expected_sha256} (cross-host)
document_inspect: {workspace_id, document}; add expected_revision only for a pinned page
mutation:         {workspace_id, document, current expected_revision, new request_id, ...}
document_export:  {workspace_id, document, current expected_revision, suggested_filename}
```

An export result with `delivery=inline_content_only` and `file_created=false` has not
created `suggested_filename` anywhere. Use `document_save` for a same-host workspace
artifact, or write the returned inline content and verify its SHA-256.

Use a fresh UUID for each new mutation intent. Reuse a write request UUID only to
recover the result of the exact same arguments; changed arguments always need a new
UUID. `source_name` and `suggested_filename` are leaf names, never client paths, and
`content` is the complete XML document rather than a diff or selected fragment.

Circle and Pline creation results include their typed geometry. For any geometry
calculated from a verbal dimension, bounding box, center, symmetry or containment
constraint, compare those returned coordinates/bounds and a `document_inspect`
snapshot with every requested constraint before creating MOPs. Do not silently invent
missing stock thickness, target depth, depth increment, feeds or spindle speed. A
reasoned proposal may use known stock, material and tool context, but requires user
confirmation. For a through-cut, choose a tool/material-safe increment whose nominal
multiples slightly exceed total depth, whose penultimate pass remains above the stock
bottom, and whose final pass normally spends at least one third of its depth in stock.
Use `machining_calculate_depth_increment` for that arithmetic. Supply either the exact
desired pass count or the maximum depth increment already judged safe for the material
and tool, never both. Inspect its returned pass depths and final-stock fraction, then
obtain user confirmation before passing the proposed increment to a MOP add tool.
If the user explicitly chooses a different valid value, preserve it. Treat a low final
stock fraction or relaxed rounding as a warning to explain, not an authorization to
reject or replace the user's machining decision.

Plan MOP creation order before calling the add tools because each operation appends
to its Part. Verify the inspection order places enclosed/internal and non-releasing
detail work before any through-cut Outside Profile that releases the containing part.
The v1 MCP surface cannot reorder an existing sequence; rebuild it rather than export
a known-unsafe order.

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

OpenCode local configuration (replace the Python and workspace paths) uses the same
standard stdio model. Do not start the command separately: OpenCode launches and
stops it. This server has no localhost port.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "cambam": {
      "type": "local",
      "command": [
        "D:/CAD/CamBamMcp/Scripts/python.exe",
        "-m", "cambam_builder.mcp_adapter",
        "--workspace", "D:/CAD/AgentWork"
      ],
      "enabled": true,
      "timeout": 15000
    }
  }
}
```

#### Reusable isolated OpenCode agent run on Windows

For a development or acceptance run that must not change normal OpenCode project
configuration, use a unique ignored `output/<task>-<unique>/` root with separate
client and MCP-workspace directories. Point `XDG_CONFIG_HOME`, `XDG_STATE_HOME`,
`XDG_CACHE_HOME` and `OPENCODE_CONFIG_DIR` into that root, provide only the task MCP
entry through `OPENCODE_CONFIG_CONTENT`, and disable project configuration, default
plugins and auto-update. Do not set `XDG_DATA_HOME` when the run is intentionally
reusing the user's existing OpenCode provider authentication; this avoids copying or
printing credentials. Set it to another task-owned directory only for a fully
credential-isolated probe that does not invoke a hosted model.

The model is an environment choice, not a repository default. Confirm what the
current provider workspace intentionally permits rather than assuming a `latest`
alias or a model used on another machine. `opencode models openrouter` shows the
provider catalog, not necessarily the API key's effective workspace policy. For the
2026-09-21 `local-cam` acceptance environment, the user confirmed GLM-5.3-Flash,
DeepSeek V4.1 Flash and GPT-5.6 Luna; GLM-5.3-Flash completed the final run. Other
development environments may use different providers, model identifiers or policy.

This is the reusable shape; replace every example path and select a model confirmed
for the current provider workspace:

```powershell
$taskRoot = Join-Path (Resolve-Path "output").Path "mcp-client-<unique>"
$client = Join-Path $taskRoot "client"
$serverWorkspace = Join-Path $taskRoot "server-workspace"
$python = (Resolve-Path ".venv/Scripts/python.exe").Path
New-Item -ItemType Directory -Path $client,$serverWorkspace -Force | Out-Null

$config = @{
    mcp = @{
        cambam = @{
            type = "local"
            command = @($python, "-m", "cambam_builder.mcp_adapter",
                        "--workspace", $serverWorkspace)
            enabled = $true
            timeout = 15000
        }
    }
} | ConvertTo-Json -Depth 8 -Compress

$env:XDG_CONFIG_HOME = Join-Path $taskRoot "config"
$env:XDG_STATE_HOME = Join-Path $taskRoot "state"
$env:XDG_CACHE_HOME = Join-Path $taskRoot "cache"
$env:OPENCODE_CONFIG_DIR = Join-Path $taskRoot "config"
$env:OPENCODE_CONFIG_CONTENT = $config
$env:OPENCODE_DISABLE_PROJECT_CONFIG = "true"
$env:OPENCODE_DISABLE_DEFAULT_PLUGINS = "true"
$env:OPENCODE_DISABLE_AUTOUPDATE = "true"

opencode.cmd mcp list --pure

$model = "openrouter/<provider>/<model-confirmed-for-this-workspace>"
$title = "CamBam MCP <task> <unique>"
$prompt = @(
    "First complete task paragraph."
    "Second complete task paragraph."
    "Final reporting and stopping criteria."
) -join " "
opencode.cmd run --pure --model $model --title $title --dir $client $prompt
```

Keep the prompt in one native-process argument for non-interactive Windows runs. In
the OpenCode 1.18.31 acceptance harness, passing a multiline PowerShell string caused
the stored user message to stop at the first paragraph (705 characters), even though
the terminal command appeared to contain the full prompt. This is an observed
CLI/harness boundary, not a general claim about interactive OpenCode, other shells or
future releases. Joining paragraphs with spaces, as above, preserved the complete
2,111-character prompt. Do not use `--auto` for acceptance: permission denials are
part of the evidence and broad automatic approval would weaken the test.

If behavior suggests lost instructions, inspect only the task-owned session rather
than blaming the model or server. Use `opencode session list --pure --format json` to
find the session by its unique title/directory, then
`opencode export <session-id> --pure`; check the user-message length and a few
required phrases without publishing credentials or unrelated sessions. A model
stopping after `document_list` is not valid usability evidence if the creation/task
paragraphs never reached it.

For this adapter, success still requires more than a connected status: inspect the
tool transcript, verify client/server copy hashes, run
`demos/mcp_client_acceptance_verify.py` on the delivered A/B files, and separate that
automated evidence from CamBam acceptance. Ask for the **CamBam MCP** tool count
excluding ordinary client-local tools; otherwise a model may report the combined
tool inventory. Keep private/user CAD and secrets out of hosted-model prompts unless
the task explicitly authorizes them.

### Clean wheel installation and rollback

Use a dedicated environment so client removal cannot disturb another Python
application. Copy the built wheel to the target machine, then run:

```powershell
uv venv --python 3.13 D:/CAD/CamBamMcp
uv pip install --python D:/CAD/CamBamMcp/Scripts/python.exe "D:/Install/cambam_builder-0.1.0-py3-none-any.whl[mcp]"
New-Item -ItemType Directory -Path D:/CAD/AgentWork
D:/CAD/CamBamMcp/Scripts/python.exe -c "from cambam_builder import CBProject; print(CBProject('smoke').project_name)"
```

Supported and checked on Windows are base-library Python 3.9 and MCP-enabled
Python 3.10, 3.11, 3.12 and 3.13. Python 3.9 deliberately rejects the adapter.
The MCP extra pins `mcp==2.2.0`; all other versions are unsupported until the
protocol suite is rerun.

To disable the adapter while retaining the direct library in that environment,
remove the SDK packages and remove/disable the client's `cambam` configuration:

```powershell
uv pip uninstall --python D:/CAD/CamBamMcp/Scripts/python.exe mcp mcp-types
D:/CAD/CamBamMcp/Scripts/python.exe -c "from cambam_builder import CBProject; print(CBProject('rollback').project_name)"
D:/CAD/CamBamMcp/Scripts/python.exe -m cambam_builder.mcp_adapter --workspace D:/CAD/AgentWork
```

The final command must exit with the missing `[mcp]` extra message while the direct
API command succeeds. Reinstall the wheel extra to re-enable the adapter. Removing
the entire dedicated environment is also a valid rollback after its client entry is
removed; user `.cb` files and workspace artifacts are not removed automatically.

### 4e OpenCode and CamBam acceptance

Start in a client project directory outside the MCP workspace, register the local
command above, and confirm `opencode mcp list` reports `cambam connected`. Then ask
the OpenCode agent:

```text
Use cambam MCP tools for document/CAD/CAM work and normal local file tools only for
deterministic copies and complete .cb reads. Because this client and server share a
filesystem, call document_list to get workspace_path. Use document_save with unique
server-workspace leaf names as an exact-byte handoff, copy each returned absolute_path
to the requested client-local file, and verify its SHA-256. To reload a client file,
binary-copy it under a new unique leaf in workspace_path, verify both hashes, and call
document_open on that relative leaf with expected_sha256. Do not manually reconstruct
XML, reuse an old staging name, or treat the server path as the final destination.
Report the server workspace ID and the CamBam MCP tool count, excluding ordinary
client-local file and shell tools. Create a document named slice with asserted
mm units. Add Rect outline on
layer Geometry at (0,0,0), width 20 and height 10. Add an enabled outside Profile
named profile in Part targeting outline: target depth -1, depth increment 0.5,
tool diameter 3, cut/plunge feeds 300/100, spindle 12000, stock surface 0 and
clearance plane 5. Inspect it, save it under a unique server-workspace leaf, copy the
returned absolute_path exactly to A.cb in this client project, verify its SHA-256,
stage A.cb back under a fresh workspace leaf, and open it with expected_sha256 as a
new document. Translate outline
by (5,2), inspect it, save it under another unique server-workspace leaf, copy it
exactly to B.cb here, verify its SHA-256, stage B.cb under another fresh leaf, and
open it with expected_sha256.
Close all handles. Do not retry failed
mutations with changed arguments under the same request ID. Report every resulting
revision, the final world corners, target ID, hashes, errors, and whether any MCP
argument needed manual correction.
```

The expected revisions are create 0, Rect 1, Profile/A 2, reopened 0, translated/B
1. The final corners are `(5,2,0)`, `(25,2,0)`, `(25,12,0)`, `(5,12,0)` in cyclic
order. Before opening either file in CamBam, independently validate the artifacts:

```powershell
.venv/Scripts/python.exe demos/mcp_client_acceptance_verify.py C:/path/to/client/project/A.cb C:/path/to/client/project/B.cb
```

The verifier checks strict import, hashes, identities, relationships, A/B geometry,
zero/unspecified stock and all explicit Profile values. It does not establish the
client connection or native CamBam behavior.

Also run one unprimed usability check from that client directory: ask the agent in
ordinary terms to create and save a new `.cb` there containing a non-rectangular
closed outside boundary, a circular internal opening and suitable machining
operations to cut both with a stated end-mill diameter. Do not mention import/export
or operation names. Pass requires export followed by a client-local write with no
server-workspace access request, an Inside Profile targeting the original circle, and
an Outside Profile targeting the original closed boundary. It must not add compensated
helper geometry or substitute Engrave. Open the result in CamBam, generate toolpaths,
and verify the cutter centerline is offset inward from the opening and outward from
the exterior by half the tool diameter. Report any permission prompt, wrong tool call,
argument correction, rejected target, unexpected helper geometry or wrong-side path.

For stock and repetition acceptance, explicitly request a configured Part rather
than nine copied shapes. The agent should call `machining_configure_part` with the
sheet width/height/thickness/material, then set `nest_method` to `Grid` (or
`IsoGrid`) with the requested rows, columns and spacing before adding the MOPs.
Inspection and exported XML must show the nonzero `<Stock>` bounds and native
`<Nesting>` settings. This is distinct from geometry copies: CamBam nesting repeats
the complete Part's MOP sequence at generated positions. Part stock is optional and
is not expanded for those copies. When nonzero XY stock is defined, checking in
CamBam that the nested envelope fits it is good practice; the adapter's
`NESTED_STOCK_FIT_ADVISORY` is non-blocking and absence of stock or an unchecked fit
does not make the document invalid.

For the merge-alignment additions, include a Part whose stock starts at a nonzero XY
offset and whose top surface is nonzero. After a save/reopen cycle, inspection and
CamBam must show the same PMin/PMax coordinates. Patch only one nesting field and
confirm the other modeled values and any native `BasePoint`, `PointListID` or
`GCodeOrder` child remain unchanged. Imported `Manual`/`PointList` methods must inspect
without an output-schema error; the MCP tool deliberately does not author their
placement data. For PointList, verify that each point translates the unnested Part
toolpaths rather than replacing the source geometry origin. With the default drawing
origin, moving source geometry by `(sx,sy)` and a nesting point by `(px,py)` must move
the corresponding path by their combined translation. Pen width `0`, Part tool
diameter `0`, zero/unspecified stock and all eight native grid orders must likewise
remain inspectable.

Also author one Text targeted by an Engrave MOP with `tool_profile=VCutter`, one open
Pline targeted by Profile, and one closed Outside Profile with Automatic holding tabs
(`tab_distance=0`; test Square and Triangle for retained stock, and Skip only as the
non-contact/plasma behavior). Keep `tab_use_leadins=false` because this MCP slice pins
the Profile lead-in to None. Pass requires CamBam Plus 1.0 to load the file without repair, show the V-cutter and holding-tab
properties, retain the Text/open-Pline targets, and generate the expected open offset
and tabbed closed toolpaths. Also exercise Text directly with Pocket and Profile
Inside: these clear or offset its outlines with the same round-cutter reach limits as
other shapes. Corner overcut must add the documented extra move into inside corners
that otherwise remain uncut, deliberately overcutting stock so fitted parts such as
slot joints or inlays can enter. EndMill must remain available for ordinary path-following Engrave;
neither Engrave tool profile fills Text interiors. At `roughing_clearance=0`, Engrave
follows the line placement; test one signed nonzero clearance as an offset. Profile
and Pocket use the same signed allowance convention: positive leaves stock and
negative overcuts. CannedCycle Drill remains fixed at zero. For SpiralMill CW and
CCW, test Point targets with explicit hole diameter and Circle targets with both
explicit and Auto diameter. Verify `effective diameter = hole diameter - 2 *
roughing clearance`, that it remains greater than the tool diameter, and that
HoleDiameter, DrillLeadOut, SpiralFlatBase, LeadOutLength, RoughingClearance and the
selected method are `state="Value"` when supplied. Auto HoleDiameter remains
`state="Default"`; fresh SpiralMill XML must omit
CannedCycle-only PeckDistance/RetractHeight/Dwell and unused CustomScript entirely.
Confirm CamBam opens without asking to revert those irrelevant fields. A nonzero
LeadOutLength requires DrillLeadOut true, and a positive centerward length must not
exceed the effective hole radius.
Manual tab authoring is not part of this check: imported native Manual tabs are
preserve-only and fresh direct-core/MCP authoring must reject them rather than emit an
incomplete points collection. Reverse the open Pline in a separate copy and confirm its
Inside/Outside physical side swaps; this is why the adapter reports
`VertexOrderRelative`. Do not generate production G-code. Report each property and
toolpath check separately; automated XML round trips do not replace this native CAM
acceptance.

For required domain acceptance in CamBam Plus 1.0, open A and B separately. The fixed
version baseline above applies and must not be requested again. Confirm the drawing
is using millimeters—the XML
does not persist a verified unit setting—and reject the case if CamBam interprets
the numbers in another unit. Confirm A's outline spans `(0,0)` to `(20,10)` and B's
spans `(5,2)` to `(25,12)`. CamBam displays the primitive by its native type and
primitive ID, such as `PolyRectangle (1)`; it does not natively name primitives.
Confirm `outline` is preserved in that primitive's `Tag` metadata instead. The layer
must display the native name `Geometry`, while the Part and MOP must display their
native names `Part` and `profile`. In the advanced Profile properties confirm
Outside, TargetDepth `-1`, DepthIncrement `0.5`, ToolDiameter `3`, CutFeedrate `300`, PlungeFeedrate
`100`, SpindleSpeed `12000`, StockSurface `0` and ClearancePlane `5`.

Generate the Profile toolpath (select the operation and use **Generate Toolpath**,
or use `Ctrl+T`). Pass requires an outside toolpath around the rectangle, two depth
levels at `-0.5` and `-1.0`, no error, and no unexpected geometry movement. Do not
produce machine G-code for this acceptance. Report pass/fail for units, geometry,
properties and toolpath separately, plus whether the agent workflow was usable
without manual MCP-call repair. Keep A/B until the result is recorded.

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
parity. 4c prepares authored CAD artifacts; 4e retains named local-client and
CamBam units/geometry/property and toolpath acceptance.
