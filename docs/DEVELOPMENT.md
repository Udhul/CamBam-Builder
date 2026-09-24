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
