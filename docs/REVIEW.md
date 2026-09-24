# Initial workflow and engineering review — 2026-09-07

## Bounded XYZ Engrave CamBam post finding - 2026-09-24

The user opened the prepared `V-variable-engrave.cb` in CamBam Plus 1.0,
generated toolpaths and reported that the Engrave toolpath slopes directly
along the Pline with no visible extra pass. The resulting
`output/variable-v-engrave-20260924-01/V-variable-engrave.nc` has SHA-256
`69367ec77654c9c2e005fd2db7dbc44c8c4d98bcacbc617a12b682e2cd926981`.
The candidate hash remains
`0b7d2e3d42a04785b2989f4b19b0d43916ce09452f62f76d5182c3fb141a5585`.
The header identifies `V-variable-engrave` and `Default`; `G21 G90 G61 G40`
and the modal reader establish the bounded millimetre/absolute post dialect.

The exact audit returned `engrave_emitted_motion_deviation` (exit 1): eight
parsed items versus nine required. It found **one exact sloped XY feed cut**,
`G1 F300 X10 Z-2.25`, from (2,2,-1.25) to (10,2,-2.25), and no other XY
feed cuts. This corroborates the user's visible slope and the generated
cut-path geometry. The original ideal cone target/rest calculation therefore
still describes that isolated cut, conditional on the stated tool model; the
native *whole motion* did not pass stock/process replay.

The post first rapids from assumed setup (-10,-10,+5) to (2,2,+5), then
rapids down to (2,2,+2.75). It omits the required F120 feed approach to +1
and feeds at F60 directly from +2.75 to -1.25. At the far end it rapids from
(10,2,-2.25) to +5 instead of feeding at F300 to +1. M5 occurs at
(10,2,+5); there is no rapid return to setup. The tool change and M3 S12000
match the expected initial events only under the declared incoming setup
position; the post cannot prove that physical position. The rapid retract
violates the required process role even though it follows the cut endpoint.

**Decision:** the bounded XYZ Pline/Engrave carrier passes the native
*visible sloped toolpath and isolated cutting-segment* check, but fails the
complete emitted-motion gate. It is an inspection carrier only. Keep the
previously accepted Drill/CustomScript post as execution authority for this
synthetic V groove. Do not enable both carriers for the same cut. No changed
Engrave candidate or unchanged repost is justified by this finding. Reopen
Engrave execution only with a demonstrated carrier/post control for F120
approach, feed retract and setup return, followed by a fresh whole-post audit.
Neither route has physical machining acceptance.

## Bounded XYZ Engrave candidate preparation - 2026-09-24

The explicit variable-depth post passed previously, but its Drill/CustomScript
motion is not a visible CamBam toolpath. This bounded probe uses the same
target/plan and no new geometric breadth. The source holds the original target
spine (0,2,-1) to (12,2,-2.5). `V-variable-engrave.cb` adds a separate
generated cut Pline (2,2,-1.25) to (10,2,-2.25) and one enabled VCutter
Engrave selecting only that cut. TargetDepth=0 prevents the additive-Z error
observed in RC01; OptimisationMode=None requests source order. The local
strict-reimport check confirms both paths and the MOP/process fields.

The ignored candidate/source SHA-256 values are respectively
`0b7d2e3d42a04785b2989f4b19b0d43916ce09452f62f76d5182c3fb141a5585`
and `aea8757a00e0e4fcc3aa0bd5b43005332c3c4635d29231b7f81265271c19bf4a`.
The manifest pins the same nine-item complete motion and five rest areas as
the accepted explicit route. A synthetic Default wrapper passes exact
comparison and posted-coordinate replay. A deliberately native-like wrapper
with the expected sloped cut but omitted approach and feed retract fails
whole-motion acceptance while reporting that the sloped cut is present.
This is parser/adapter evidence, not CamBam-produced output or preview.
The focused RC01/cone/variable test selection passed 17 tests; compileall,
tracked `git diff --check`, untracked source/test whitespace inspection,
strict reimport and candidate/source hash verification passed. This work is
uncommitted and ready to commit, not merge-ready.

At preparation, CamBam Plus 1.0 preview and `V-variable-engrave.nc` were pending. The
[runbook](DEVELOPMENT.md#bounded-variable-depth-xyz-engrave-preview-and-post-probe)
gives the exact steps and pass criteria. Do not infer native execution from
the visible cut alone; re-open only with actual Default post and preview
evidence. The accepted script route remains the bounded execution authority.

## Bounded variable-depth V groove posted output - 2026-09-24

The user exported `output/variable-v-20260924-01/V-variable.nc` from the
prepared candidate using CamBam's Default post. Actual post SHA-256 is
`3d37cd3ecd5c69efbdb3ddab381f283dda2352d470a3733e9477606da0e6c570`;
the candidate hash remained
`5f58068f39ffcaef6cf40147750a62f4e0be55d5580a42c90a01c9ae26d064b0`.
The post contains the literal `G1 F60 X2 Y2 Z-1.25` entry followed by
`G1 F300 X10 Y2 Z-2.25`, then retract and setup return. The exact audit returned
`bounded_emitted_variable_v_motion_pass`: all nine emitted items matched,
including tool/spindle events, G0/G1 roles, feeds and coordinates. Shared replay
found one `variable-v` prefix with two cone sweeps. The independent target
oracle returned partial completion with rest at depths 0/1/1.5/2/2.5 mm of
15.091265791880026 / 7.028684027518187 / 4.21460291488107 /
1.8062583920918873 / 0 mm2. This accepts the bounded explicit output route.

The user observed one Drill MOP targeting the one-point primitive rather than
the sloped Pline. Strict reimport confirms that relationship: the Point is the
CustomScript carrier anchor, and the XYZ Pline is a target guide. The emitted
G-code, not a native Pline-driven MOP, establishes the sloped motion. No claim
is made that CamBam displayed that slope as a toolpath, computed a native
V-carve, or verified a physical setup. The post does not encode incoming
machine position; ideal tool, holder/fixture and process limits remain as
declared in the [contract](structure_spec.md#bounded-variable-depth-v-groove-and-cambam-carrier).
The user later confirmed that CAMotics displays the script-produced sloped
line, while CamBam does not display the script's resulting toolpath. This
supports output visibility as the next concrete gap; it adds no physical
machining acceptance. The [output direction](REST_MACHINING_PLAN.md#programmatic-execution-requirement)
retains generated XYZ Pline/Engrave as the inspectable candidate, with exact
post and full-role replay still required. The prepared implementation was committed
as `ba46a7d`; this posted-output record is a separate documentation change.

## Bounded variable-depth V groove preparation - 2026-09-24

The next backlog-6 consumer uses one exact tapered finish groove: a 90-degree
pointed cone's ideal swept envelope along (0,2,-1) to (12,2,-2.5). The generated
proper subspine runs (2,2,-1.25) to (10,2,-2.25), so it changes tip Z during a
single feed cut and leaves finite-end rest. Its conical radius/length is 3 mm;
the X slope is 0.125, below the cone's unit radial slope. The core checks
all-height containment by the common target/cut linear depth law, and shared
replay records two ordered cone sweeps. A separate closed-form section-area
reference uses two tangent sides and endpoint arcs; an independent midpoint
row integral agrees within 0.0005 mm2 at depths 0/1/1.5/2/2.5. Analytic rest
there is 15.091265791880026 / 7.028684027518187 / 4.21460291488107 /
1.8062583920918873 / 0 mm2. This is partial target completion.

`output/variable-v-20260924-01/V-variable.cb` strict-reimports with one enabled
VCutter Drill/CustomScript MOP and six literal motion lines. Candidate SHA-256 is
`5f58068f39ffcaef6cf40147750a62f4e0be55d5580a42c90a01c9ae26d064b0`;
source SHA-256 is `652a830b75ee19fb10b860cf14f796ed85ad1a0115a2d4157ce3d144262b1d2c`.
The source XYZ Pline guide was reimported with exact endpoint Z=-1/-2.5.
`expected-motion.json` pins both file hashes, fingerprints, all nine ordered
items and five residual references. A synthetic Default wrapper passed the
exact reader and posted-coordinate replay, and mutations of depth, G0/G1,
RPM and approach fail. This is adapter evidence only; no CamBam-produced
`V-variable.nc` has yet been received or audited.

The shared replay change was checked by 34 RC01/cone/variable tests, all passed.
`compileall -q cambam_builder tests/test_variable_vcarve.py
tests/test_variable_cone_script.py` and tracked `git diff --check` passed;
the four untracked source/test files have no trailing whitespace and end in
newlines. The target is an ideal cone-sweep envelope represented in `.cb` by
an XYZ guide; the guide is not a native V-carve MOP. The Default post's initial
machine position is assumed, and no holder, fixture, physical tolerance or
production process is certified. The [runbook](DEVELOPMENT.md#bounded-variable-depth-v-groove-carrier-and-posted-replay)
owns the pending user post/audit. Reopen geometry breadth for a named target
that this straight, linearly tapered case cannot represent. At preparation,
work was uncommitted and ready to commit, not merge-ready; it was subsequently
committed as `ba46a7d` before the actual post arrived.

## Bounded cone carrier preparation - 2026-09-24

The prior RC01 literal Drill/CustomScript carrier had passed an actual Default
post, so the same carrier shape was used for one full-depth 12 x 4 mm pointed-cone
slot. `output/cone-script-20260924-01/V-cone.cb` strict-reimports with one
VCutter/CustomScript MOP and six real-newline motion blocks; candidate SHA-256 is
`59cd263d5dafac98a5f2dd794ea30906545118e3ebf3631d015e8fed62143150`.
The exact expected manifest includes nine setup/motion items, source and motion
fingerprints, and the original analytic residual references. It checks a tip
path from (2,2,-2) to (10,2,-2) with +5/+1 approach, feeds, setup return and
T3 spindle events. The ideal cone has 3 mm maximum radius/conical length.

The focused synthetic Default-wrapper test accepts the nine parsed items and
replays two cone sweeps through shared stock and the independent slot area
oracle. It rejects a rapid plunge, excess depth, changed spindle RPM, missing
approach, stale manifest and changed candidate. This verified the adapter's
boundary before native posting. The initial JSON manifest check exposed a
tuple/list serialization mismatch; the audit now compares canonical JSON lists.

The user posted `output/cone-script-20260924-01/V-cone.nc` in CamBam Plus 1.0
with Default/Default mm. Its SHA-256 is
`1c4281c930e26fade89d5fc75064a36908466f6ffa3ed96d2b5d30b390761cd4`.
The user initially saw no path, then confirmed CamBam displayed the Drill
toolpath on closer inspection. The Ctrl+W file has the exact six
newline-separated script blocks. Its wrapper has G21/G90/G61/G40, T3/M6,
G17, M3/S12000, G98/G80 and terminal M5/M30. The fail-closed reader matched
all nine ordered items, including G0/G1, F120/F60/F300, setup return and stop.
The parsed coordinates replay as one `slot` prefix with two cone cut sweeps.
The original target oracle reports partial target completion: surface rest
3.4336293856408275 mm2, depth-1 rest 0.8584073464102069 mm2 and depth-2
rest 0 mm2. This passes the bounded explicit cone output gate. The Drill
toolpath display is a native UI observation; the posted file establishes the
literal motion and stock replay result.

The Default post does not encode the incoming physical machine position. The
audit assumes tip (-10,-10,+5); it models an ideal pointed cone with zero
physical error and no holder, fixture, controller or material-process proof.
Physical machining and broader variable-depth V-carving remain separate. Reopen
the output finding if the postprocessor/profile, carrier or cone geometry changes.

Verification on `.venv/Scripts/python.exe`: 20 focused cone/slot/mixed/RC01
native tests pass after the manifest fix; `compileall` and import/construct
smoke pass. The actual `V-cone.nc` audit returns
`bounded_emitted_cone_motion_pass`; `git diff --check` and the untracked text
whitespace scan pass. The broad suite was not rerun because the shared replay
and native RC01 contracts were not changed in this increment. Work is
uncommitted and ready to commit, not merge-ready.

## Shared RC01/cone motion and stock replay - 2026-09-24

The bounded increment extracted `cam_core.replay`: one immutable ordered XYZ
trace and source/motion fingerprint, resolved cylinder or pointed-cone profile,
target, tool/spindle events, and cumulative cut-sweep prefixes. RC01 and the
cone slot retain their own strict process/geometry checks and independent
residual oracles, but both now feed their residual calculations from shared
ordered cuts. The synthetic mixed trace runs RC01 T1/T2 and a cone slot beside
the RC01 stock in one drawing frame. Keeping the targets disjoint prevents a
false claim that cone cuts remove material already cleared by the endmills.

The focused RC01, cone, mixed, RC01 native and RC01 stock-authority suites passed
28 tests. The mixed regression checks operation prefix order, residual membership
before/after the cone cut, source/motion identity, and fail-closed stale source,
removed cone entry, low link and missing tool event. The pre-existing RC01
and cone tests retain their all-height/target and numerical residual references.
No new CamBam post or manual application check was needed for this detached
representation. There is no claim for interacting targets, arbitrary 3D access,
cone holder/fixture clearance, output parity or physical machining.

The next gate was the explicit cone carrier with actual user-posted motion
replayed through this source contract; its result is recorded above. Reopen
the shared geometry evaluator for overlapping endmill/cone targets only when a
concrete combined target requires
cross-profile stock access or residual credit; the disjoint case does not prove
that behavior. Native Pocket N still needs a carrier capable of its missing
motion roles and fresh whole-motion evidence.

## Bounded pointed-cone slot - 2026-09-23

The [implemented contract](structure_spec.md#bounded-pointed-cone-slot-generation-and-verification)
fixes a 12 x 4 mm rectangular opening, 90-degree pointed cone and stock top Z=0.
The 2 mm candidate has one finite x=2..10, y=2 pass. A 3 mm maximum
radius/conical length admits it; a 1.5 mm radius/length rejects it. At a 1 mm
design cap, three x=1..11 passes at y=1,2,3 improve coverage over the
centerline-only case. Every path has plunge, cut and retract; above-stock rapid
links are explicit. The verifier rejects incomplete/reordered motions, target
wall overrun, insufficient tool radius and a changed access path.

Analytic section references for the full V are 3.4336293856408275 mm2 rest at
the surface, 0.8584073464102069 mm2 at depth 1 and zero at the centerline at
depth 2. The capped three-pass result has 1.0319614364481282 mm2 surface rest,
0.643805509807656 mm2 at depth 0.5 and 20 mm2 of floor rest at depth 1. Its
centerline-only surface rest would be `28-pi` mm2. Independent direct row
integration agrees with analytic section area within 0.002 mm2 at four depths
per case. At 4096 slabs, geometric volume enclosures are
2.2670001928631014..2.3116855369897467 mm3 for the full V and
5.643645141380579..5.661947885370508 mm3 for the capped result.

The all-height containment argument is algebraic: at stock depth `t`, each
deepest-path disk radius is `d-t`; endpoint-to-wall margin is at least `d`.
Vertical traversal is a subset of that deepest disk and rapid links begin above
stock. The volume enclosures use nested sets and floating area calculations;
they are geometric bounds subject to floating roundoff, not directed numeric
intervals. No machine feeds, holder, fixtures, process uncertainty, native
CamBam motion or physical cut is accepted. Reopen wider topology or tool-profile
classes only when a named consumer fixes their acceptance requirements. The
[next shared-motion increment](PROGRESS.md#active-work-and-next-priority) is
separate from this completed reference geometry.

Verification on `.venv/Scripts/python.exe` (Python 3.13): the five new tests
pass. The related cone/RC01/stock/reference suite has 44 passing tests;
`compileall` and tracked `git diff --check` pass, and the two new untracked
files pass a direct trailing-whitespace check. A broader discovery run executed
375 tests but failed to import two MCP modules because the optional `anyio`
dependency is absent (73 skips); this does not establish a passing full suite.
Manual CamBam validation adds no evidence to the detached geometry. Work remains
uncommitted and ready to commit, not merge-ready.

## Bounded section-motion verification - 2026-09-22

`verify_section_motion` accepts ordered, supplied horizontal cutting and travel
segments for the exact 30 x 20 mm stock, 26 x 16 mm target and 4 x 6 mm island
scenario. It recomputes cutting bounds before updating stock, checks every cut's
outer occupancy against the original protected target, and allows non-cutting
motion only inside one cited earlier cut's guaranteed inner capsule or strictly
outside initial stock. Cutting, prior-cleared and outside-stock entry disks are
explicit. Travel contributes no removal. Paths retain the caller's radius and
position uncertainty; a separate path represents a new section entry or tool.

The accepted two-tool case cuts `(6,5)` to `(24,5)` with radius 2 mm, travels
back to `(10,5)` inside that pass's guarantee, then enters with a radius-1 mm
tool in the cleared region and cuts to `(3,5)`. An independent quarter-millimetre
grid uses direct point-to-segment squared distances for both cutting sweeps and
compares the resulting required-rest membership at all 9,801 points. The
protected island interior at `(15,10)` remains stock but is not required rest;
`(3,5)` is removed by cleanup and `(3,8)` remains required rest. A connector to
`(3,5)` before cleanup, travel without a prior cover, a future-source citation,
and a segment through the island all fail. A separate uncertain case accepts
equal 1.75 mm guaranteed and inflated travel radii, then rejects an exact
`1e-30` mm overrun and a prior cut whose lower removal is empty. Outside-stock
travel passes with positive clearance and fails at boundary contact. Broken
segment continuity, invalid entries and an altered segment kind fail before any
result is returned.

Exactness rests on the [section-motion contract](structure_spec.md#directional-analytic-stock-section-bounds):
outer cutting containment, complete-traversal lower removal, endpoint/convexity
containment of parallel travel capsules, and strict minimum distance for an
outside-stock approach. Finite point checks corroborate residual membership;
they do not replace those analytic proofs. Requiring one cited capsule per
travel piece is intentionally conservative. Reopen union-cover or non-horizontal
motion only for a concrete consumer that cannot split or supply the needed path.

Verification on existing `.venv/Scripts/python.exe` (Python 3.14.5; this does
not renew the supported 3.9-3.13 version matrix):

- `-m unittest discover -s tests -p test_stock.py -v`: 20 passed.
- `-m unittest discover -s tests -q`: 356 tests, OK with 12 skips for optional
  capabilities.
- `-m compileall -q cambam_builder tests` and `git diff --check`: passed.

No CamBam manual validation adds evidence: this is a detached, nonserialized
section calculation. Entry disks and connectors are checked at one Z only;
descent, retracts, tool changes, holder occupancy, stock at other heights and
execution remain unverified. There is no production-machining claim. The bounded
increment is ready to commit, not merge-ready. The next input and reopening
criterion are in [PROGRESS](PROGRESS.md#next-detached-stockrest-increment).

## Holed target stock/rest bounds - 2026-09-22

`SectionTarget` and `compose_target_rest_bounds` separate original required
removal from initial stock for one exact outer rectangle and one strictly interior
rectangular island. The target includes both walls but excludes the island's open
interior. Every validated source's outer capsule must remain inside the outer
target and have exact segment-to-island distance at least its radius. Wall
tangency passes; any rational penetration fails the whole composition. The
existing ordered whole-stock bounds and uncertainty provenance remain available,
while lower/upper rest subtract outer/inner removal from the original target.
This permits a later supplied sweep to overlap cleared space and extend into
still-required material without treating their interface as a protected wall.

Independent exact point-membership checks use a 30 x 20 mm stock, a 26 x 16 mm
outer target and a 4 x 6 mm island (required area 392 mm2). A large radius-2
sweep, a smaller radius-1 sweep crossing its cleared interface, and a second
small sweep tangent to the island preserve all protected sample points on a
quarter-millimetre grid. Rest at `(3, 5)` changes from present to removed;
stock at `(15, 10)` remains while that island-interior point is never required
rest. An independent `320 - 4*pi` mm2 reference lies within the one-pass rest
area interval. A two-pass uncertain example compares allowed actual displaced
disks against both removal/rest bounds. Exact wall contacts pass and `1e-30` mm
island or outer-wall overrun is rejected. These finite samples support, but do
not replace, the exact containment and triangle-inequality argument in the
[contract](structure_spec.md#directional-analytic-stock-section-bounds).

Verification on the existing `.venv/Scripts/python.exe` (Python 3.14.5; this
does not renew the supported 3.9-3.13 version matrix):

- `-m unittest discover -s tests -p test_stock.py -v`: 15 passed.
- `-m unittest discover -s tests`: 351 tests, OK with 12 skips for optional
  capabilities.
- `-m compileall -q cambam_builder tests` and `git diff --check`: passed.

No manual CamBam acceptance adds evidence for this detached, nonserialized
section model. The result is conditional on caller-supplied radius/position
envelopes and complete segment traversal. It does not establish safe entry,
connecting motion, other Z sections or production machining. No new geometry
backend or dependency was added. This bounded increment is ready to commit;
the next priority and reopening criteria are in
[PROGRESS](PROGRESS.md#next-detached-stockrest-increment). A fresh session is
appropriate after the final gates because the remaining motion-verification
scope is distinct and its contract/limits are persisted here and in the spec.

## Directional analytic stock/rest bounds - 2026-09-22

Implemented `cambam_builder.stock` for exact rectangular stock equal to required
target and one supplied horizontal disk sweep at fixed Z. The
[contract and proof](structure_spec.md#directional-analytic-stock-section-bounds)
separate feasible centers, supplied sweep coverage and actual execution evidence.
Radius/position uncertainty gives inner/outer capsules; remaining-stock direction
reverses removal direction. Protected exterior crossing fails without a result.
General nominal polygons, curves and Boolean outputs are deliberately not admitted.

Independent acceptance: a 12 mm segment with radius 2 mm removes
`48 + 4*pi` mm2 from a 200 mm2 rectangle, leaving `152 - 4*pi` mm2.
Rational area intervals enclose both references with width below 0.00034 mm2.
The uncertainty case uses radius `[1.75, 2.25]` mm and position error 0.25 mm,
producing radii 1.5 and 2.5 mm. Independent disk/strip membership over 81x41
points, four allowed displacement vectors and three radii verifies removal and
rest inclusion. These samples supplement the parameter-wise triangle-inequality
proof; they are not its basis. Varying errors along a sweep are covered by the
proof, not claimed as independently sampled execution evidence.

Other regressions preserve exact boundary contact, reject an additional
`1e-30` mm occupancy beyond it, retain zero-radius line/point guarantees,
represent empty lower guarantees explicitly, and preserve results after a
`10**30` mm exact translation. Unsupported/nonfinite inputs fail explicitly.
Independent review found no factory containment defect; its direct-construction
finding led to validating capsule containment in `RemainingSection` as well.

Verification from the repository root with `.venv/Scripts/python.exe` (Python
3.14.5; this run does not renew the documented Python 3.9-3.13 matrix):

- `-m unittest discover -s tests -p test_stock.py -v`: 6 passed, repeated after
  the review repair.
- `-m unittest discover -s tests -v`: 342 ran, OK, 11 optional-planar-backend skips and one Windows symlink-privilege skip.
  Full run overlapped the review repair; the final focused rerun covers the
  changed stock module. Skips do not constitute optional-backend acceptance.
- `-m compileall -q cambam_builder legacy_cambam_builder`: passed.
- `CBProject('smoke')` import/construction assertion: passed.
- `git diff --check`, untracked Python whitespace inspection and changed local
  Markdown target/anchor checks: passed.

The verbose suite log is local in
`output/directional-stock-20260922-202751/tests.log`; durable evidence is above.
No manual CamBam validation adds evidence because no serialization or machining
path changed. Caller-declared physical envelopes and complete traversal remain
assumptions, not measured production acceptance. Implementation is ready to
commit, not merge-ready. This closes the single-sweep dependency; cumulative
removal with overlap-safe area bounds is the next useful outcome in
[the backlog](PROGRESS.md#next-detached-stockrest-increment). Reopen general curves,
rotated sweeps or separate stock/target topology only for a blocking consumer need.
A fresh session is appropriate: no pending manual acceptance or decisions remain.

## Detached nominal planar runtime - 2026-09-22

The first runtime increment is implemented in `cambam_builder/planar.py` and its
private `_planar_shapely.py` adapter. The user explicitly authorized implementation
after the design-only rounds. The [implemented contract](structure_spec.md#detached-nominal-planar-core)
owns supported types, frame mapping, nominal error policy and exclusions; the
[status owner](PROGRESS.md#next-detached-stockrest-increment) owns the next priority.
No development evaluation helper was imported into runtime.

The end-to-end regression normalizes two 10x10 mm overlapping squares, unions to
150 mm2, subtracts the second square to 50 mm2, then erodes by a 1 mm disk to 24 mm2.
It checks source lineage, operation fingerprints, actual backend versions, unchanged
inputs and explicit unknown bounds. Other gates cover strict holes, splits, point
contacts, source winding/start invariance, numeric-key equivalence, mm/inch and
1e9 mm origin normalization, rejection at 1e15 mm under the 0.001 mm boundary budget,
work limits, circle chord-error/area references, rounded-hole erosion and empty area
versus exact-fit feasible segments. The default area budget is 0.01 mm2.

Analytic predicates classify exact-fit rectangle lines/points and circle points,
plus both adjacent binary64 values without tolerance snapping. A 10x20 inch
rectangle with a 127 mm tool radius remains exactly a segment after unit conversion
and rotated placement. A fresh process with Shapely blocked still imports the API
and computes analytic centers; backend operations return `unsupported`.

Independent review repaired three concrete boundary defects before final checks:
integer/float and signed-zero spellings now share canonical exact numeric keys;
sequence-bearing owned results copy to immutable tuples; and distinct source
vertices that collapse during conversion return `unresolved`, not `invalid_input`.
The thin inch fixture X=[1.5000000000000002, 1.5000000000000004], Y=[0,1] permanently
covers conversion loss. Malformed normalized component types fail explicitly.
Source spans now include chord endpoints so canonical ring ordering cannot obscure
their original component/ring/segment and parameter attribution.

Verification from the repository root:

- `output/shapely-evaluation-20260922-175502/<py39|py310|py311|py312|py313>/Scripts/python.exe
  -m unittest discover -s tests -p test_planar.py -v`: **18 tests pass on each of
  Python 3.9, 3.10, 3.11, 3.12 and 3.13**. These reuse the evaluated environments,
  with Shapely 2.0.7/GEOS 3.11.4 on 3.9 and Shapely 2.1.2/GEOS 3.13.1 on 3.10-3.13.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: **336 tests run,
  324 pass, 12 skip**. This existing base environment is Python 3.14.5 and lacks
  Shapely: 11 backend tests skip explicitly, as does the existing Windows symlink
  privilege check. Backend acceptance comes from the five focused runs above.
- Python 3.13 `-m compileall -q cambam_builder legacy_cambam_builder`: pass.
- `uv build --wheel --out-dir output/nominal-planar-20260922-runtime/dist
  --cache-dir output/nominal-planar-20260922-runtime/cache`: pass after sandbox
  networking blocked the first build-dependency fetch. The approved retry succeeds.
  Wheel inspection confirms both modules, correct Python-version/extra markers,
  NumPy-only unconditional requirements and absence of output/tools/tests payload.
  Import and nominal erosion directly from the wheel pass on Python 3.13 with its
  existing backend; this is a packaged-runtime smoke, not a new clean-install matrix.
- `git diff --check` and whitespace inspection of all intended new Python files:
  pass. No staging or commit performed.

Verbose run logs and disposable build artifacts are in
`output/nominal-planar-20260922-runtime/`. The setuptools-generated root `build/`
was moved into that task directory after verifying both absolute paths; no
pre-existing/user inputs were removed. Package-build permission differences required
an approved read-only wheel check outside the sandbox.

Manual CamBam validation adds no evidence: no entity, XML, MOP, MCP or toolpath
behavior changed. Automated geometry acceptance is complete for this subset;
production machining and conservative stock/rest certification are not claimed.
General arcs/compound analytic-circle topology and general collapsed center sets
remain explicitly unsupported. Reopen those capabilities for a concrete blocking
consumer and independent topology/error references, not merely another nearby
edge case. The next useful scope establishes directional removal bounds for one
analytic endmill/rest case before any nominal result can feed guaranteed stock.


## Adversarial planar acceptance and internal contract - 2026-09-22

The bounded design increment closes adversarial planar acceptance with a concrete
[owned value/error contract](REST_MACHINING_PLAN.md#internal-planar-value-and-error-contract).
The new [development runner](../tools/evaluate_planar_adversarial.py) prototypes
admission and analytic exact-fit handling without changing runtime modules,
dependencies, corpus v1 or user CAD files. This accepts the measured nominal
planar scope, not a complete geometry adapter, stock engine or machining result.

The contract separates regularized filled-area Boolean results from complete
closed-set feasible centers; explicit frames/units and source mappings from CAD
entities; error observations from certified bounds; and invalid, unsupported,
unresolved and backend-failure outcomes. A nominal result can expose unknown error
terms only with `budget_certified=false`. General polygon erosion cannot establish
the absence of collapsed branches, even alongside nonempty area. Complete nominal
feasible-center results are initially limited to analytic rectangles and circles.
No tolerance-sized polygon stands in for a segment or point.

Independent contract review identified and resolved four implementation ambiguities:
regularized versus literal set semantics, valid backend outputs outside the strict
shell/hole representation, frame/section compatibility, and explicit error metrics
and certification status. The polygon vertex-count rule also explicitly excludes
valid authored analytic circle/two-arc rings. A square minus an interior triangle
touching its shell at a point produces a valid GEOS polygon but fails our strict
owned topology: the probe reports unsupported output topology. Shared-edge square
intersection is a raw line and an empty regularized **area**, not an empty literal set.

The runner enforces 0.001 mm boundary and 0.01 mm2 area gates independently:

| Gate | Construction and accepted evidence |
| --- | --- |
| Invalid input | NaN/infinities, XYZ tuples and coordinates at 1e15 mm rejected before GEOS; self-crossing/degenerate rings, duplicate edges, outside/touching holes and overlapping/nested/touching holes rejected. Valid strict holes and optional terminal ring closure pass. |
| Organic contour | `r(t)=20+2*sin(17*t)` has exact area `402*pi` mm2. Analytic derivative bounds control chord interpolation; the input approximation has one component and no holes. No arbitrary organic inset oracle is claimed. |
| Arc-heavy contour | Radius-20 mm shell with six radius-2 mm holes on a radius-10 mm circle, before and after 0.5 mm inset. Independent trigonometric reference has shell radius `20-d`, hole radius `2+d`, and area `pi*((20-d)^2-6*(2+d)^2)`. One component/six holes survive. |
| Invariants | Union, intersection, difference and inset of a concave shell with two holes: ring start/winding/hole order, 37-degree rotation, reflection, scale 3.7, translation `(1e9,-1e9)` mm and mm/inch conversion. Union/intersection also check operand order. Results mapped back to mm pass area, symmetric difference, bidirectional boundary and topology checks. This is metamorphic consistency, not an independent arbitrary-offset oracle. |
| Exact fit | Analytic rectangle segment/point and circle point; tool radius one ULP and 1e-6 mm below/above exact fit gives area/empty respectively. Zero tool radius retains area. Exact rational predicates preserve dimensions and endpoint/point locations under a rigid reflected placement and rational mm/inch conversion. Invalid primitive parameters fail; general collapse is explicitly unsupported. |

For the radial curve `p(t)`, `||p''|| <= R+abs(A)*(1+2*n+n*n)` gives linear
interpolation deviation `M*h*h/8`. Candidate bound is <=0.000004 mm and reference
bound <=0.000001 mm. The derived area allowance uses the boundary tube bound
`2*L*epsilon+pi*epsilon^2`, summed over separate curves; the separated circular
offset construction has a bounded chord/offset contribution <=0.000008 mm.
These analytic approximation bounds exclude floating-point computation error.
The reference area integral is independent of GEOS. Symmetric difference and
distance measurements still use GEOS and are numerical observations.

Both directed boundary checks sample at <=0.001 mm spacing and charge 0.0005 mm
for unsampled positions plus reference interpolation error. For the dense curves,
an STRtree of target segments accelerates nearest-segment distance without changing
the metric. An initial unindexed dense run exceeded three minutes and was stopped;
this is a validation-cost observation, not a backend throughput failure. The
original invariant helper retains the earlier unindexed metric. Input admission
reserves four coordinate ULPs within one tenth of the boundary budget; that is a
resolution filter, not an end-to-end error proof or a universal world-coordinate
range. Local recentering remains a runtime contract to implement.

Maximum curved-case metrics across the tested matrix (rounded):

| Case | Absolute analytic area error (mm2) | Symmetric difference (mm2) | Boundary upper estimate (mm) | Derived approximation area allowance (mm2) |
| --- | ---: | ---: | ---: | ---: |
| Organic input | 0.000053574 | 0.000148095 | 0.000504592 | 0.002035505 |
| Six-hole circular input | 0.000133999 | 0.000410703 | 0.000504999 | 0.003216993 |
| Six-hole circular inset | 0.000103184 | 0.000434676 | 0.000504999 | 0.003468320 |

The small scalar area errors alone would understate actual geometric differences;
all four gates and topology are required. The boundary upper estimate includes
the sampling remainder, but is not an interval-arithmetic certificate.
Across all invariant operations/transforms, maximum area difference was
`1.86775e-8` mm2, symmetric difference `1.63600e-7` mm2 and bidirectional boundary
estimate `0.000500065` mm, all from the inset translation check. Every expected
component/hole count was preserved on both GEOS lines.

Final matrix: Windows x64 CPython 3.9.13 with Shapely 2.0.7/GEOS 3.11.4 and
CPython 3.10.11, 3.11.9, 3.12.10 and 3.13.5 with Shapely 2.1.2/GEOS 3.13.1;
each passes all 233 checks with zero unexpected failures. These reuse the prior
isolated wheel environments; packaging was not repeated or newly claimed.
Reports and task-owned validation helpers remain under
`output/planar-adversarial-20260922-193000/`, with runner hashes checked against
the final source. The first completed attempts failed JSON serialization of a
NumPy dimension scalar; converting it to a Python integer fixed reporting, and
all final runs started after the correction. No geometry thresholds were loosened.
A disposable runner copy changed the organic reference from `402*pi` to `403*pi`:
exactly `organic.analytic_area_mm2` failed and the runner exited 1.

Verification commands are in the [runbook](DEVELOPMENT.md#isolated-planar-backend-evaluation).
The declared project interpreter passed all 13 backend-independent corpus reference
tests and `py_compile` for the new runner. All 118 local documentation file links
resolved; changed headings were reviewed, new-runner whitespace was inspected explicitly, and
`git diff --check` passed. No runtime suite rerun was needed because runtime code
and dependency declarations are unchanged. Manual CamBam validation adds no evidence
to this design-only slice and is not required.

Source Arc/Pline-bulge detachment, general curve topology normalization, full
immutable values/results, canonical fingerprints and end-to-end error propagation
remain implementation work, not capabilities of these prototypes. Conservative
containment, stock/rest certification, motion/entry, general collapsed center sets
and inlay assembly remain unaccepted. Reopen kernel selection for a demonstrated
gap under the bounded contract; do not infer a need for another package from an
explicitly deferred capability. The first detached nominal runtime slice is the
next coherent project outcome in [PROGRESS](PROGRESS.md#active-work-and-next-priority).
The design/evidence/limits are persisted, making this a good fresh-session
breakpoint. Work is uncommitted; this record is not a merge-readiness claim.

## Shapely/GEOS planar evaluation - 2026-09-22

The bounded evaluation supports selecting Shapely/GEOS for planar primitives in
the detached-core design. It does not adopt a runtime dependency or accept an
endmill-rest engine. The [decision and internal-boundary requirements](REST_MACHINING_PLAN.md#shapelygeos-evaluation-decision---2026-09-22)
own the contract; [PROGRESS](PROGRESS.md#active-work-and-next-priority) owns priority.
No changes were made to runtime modules, project dependencies or the user's CAD
files. Work began from a clean tracked worktree; Git reported an inaccessible
ignored `.pytest_cache`, which was not touched.

The reusable [runner](../tools/evaluate_shapely.py) consumes corpus v1. Candidate
buffers use 256 segments per quadrant; independent circular references use direct
trigonometric coordinates with at least 512 segments per quadrant. S02's reference
uses analytic common-tangent endpoints and exposed arcs, not a second convex-hull
call. R01 remains ideal reachability; R02 distinguishes an explicit closed Profile
trajectory from ideal Pocket coverage. V01/V02 integrate desired target sections
with Simpson's rule, exact for their analytic quadratic section-area functions.
They do not establish a motion sweep or manufactured result.

Each applicable case checks analytic area, symmetric-difference area and topology.
Both directed boundary distances are sampled with spacing <=0.001 mm; adding
0.0005 mm bounds the unsampled part by the 1-Lipschitz distance property. Reference
arc sagitta is also charged to the 0.001 mm boundary budget. This bounds continuous
polygon/reference approximation subject to floating-point distance arithmetic;
it is not interval arithmetic or a conservative stock-occupancy proof. The runner
retains the discrete Hausdorff observation separately. T03 has only input area and
inset topology references; A01 accepts input composition only. Neither is promoted
to an independently checked general offset/path case.

Case metrics below are maxima across applicable operations/sections, rounded for
display (mm2 for area, mm for boundary). A dash means no such oracle is claimed,
not zero error. Scalar limits are 0.01 mm2, 0.001 mm and 0.01 mm3 respectively.
The boundary column includes the 0.0005 mm sampling remainder, so it intentionally
overestimates error even for exactly matching rectangles.

| Case | Absolute area error | Symmetric difference | Boundary upper bound | Topology / disposition |
| --- | ---: | ---: | ---: | --- |
| G01 | 0 | 0 | 0.000500000 | one component, no holes per operation |
| R01 | 0.000078853 | 0.000059140 | 0.000511766 | four rest components at both radii |
| R02 | 0.000078853 | 0.000059140 | 0.000511766 | Profile has one hole; profile rest has two components/one hole; ideal Pocket is solid |
| G02 | 0.000372088 | 0.000428763 | 0.000528826 | one component, one hole |
| V01 | 0 | <3e-15 | 0.000500001 | one solid section; volume error 0 mm3 |
| V02 | 0.000492831 | 0.000369623 | 0.000529414 | one solid section; volume error 0.000860811 mm3 |
| S01 | 0.000078853 | 0.000059140 | 0.000511766 | solid capsule, midpoint contained |
| S02 | 0.000061144 | 0.000049851 | 0.000510538 | solid affine-radius hull, midpoint contained |
| T01 | - | - | - | nominal line/point lost; expected limitation |
| T02 | <3e-14 | 0 | 0.000500000 | overlap/contact/gap preserve 1/1/2 components |
| T03 | <8e-15 (input) | - | - | one input component becomes two after inset; no holes |
| A01 | 0 | 0 | 0.000500000 | one component and triangular hole; input-only evidence |
| C01/C02 | - | - | - | outside planar scope; reference checks remain separate |

R01's smaller-tool gain differs from its analytic reference by 0.000059140 mm2.
All volume, area and topology comparisons retain full precision in JSON. The
reference construction is independent of candidate buffering, but Shapely still
measures polygon distance and symmetric difference; this is corroborated numerical
evidence, not an independent exact-arithmetic geometry implementation.

### Windows packaging and throughput

Fresh `uv venv` environments used installed CPython interpreters and binary-only
Shapely/NumPy installs. Each imported a separately built CamBam Builder wheel with
`-I`, asserted both modern and legacy modules resolve under that environment's
`sys.prefix`, and constructed `CBProject`. No compiler or separately installed GEOS
was required. Actual wheel tags are `cp39-cp39-win_amd64` through
`cp313-cp313-win_amd64`. This is CPython Windows x64 evidence; no claim extends to
32-bit Windows, Windows ARM64, source builds or the full optional MCP dependency set.

| Python | Shapely | Bundled GEOS | NumPy | Binary install and project coexistence |
| --- | --- | --- | --- | --- |
| 3.9.13 | 2.0.7 | 3.11.4 | 2.0.2 | pass |
| 3.10.11 | 2.1.2 | 3.13.1 | 2.2.6 | pass |
| 3.11.9 | 2.1.2 | 3.13.1 | 2.4.6 | pass |
| 3.12.10 | 2.1.2 | 3.13.1 | 2.5.3 | pass |
| 3.13.5 | 2.1.2 | 3.13.1 | 2.5.3 | pass |

The split is deliberate: [Shapely's release requirements](https://shapely.readthedocs.io/en/stable/release/2.x.html#packaging)
raise the 2.1 line to Python 3.10, while the installed 2.0.7 metadata supports 3.9.
No project Python minimum change is needed. Dependency upgrades require repeating
the matrix because this policy carries two GEOS lines; it is not a maintenance
guarantee for either upstream series. The project sdist and wheel were built using
`uv --cache-dir <task>/cache build --out-dir <task>/dist`; the generated source
manifest had no `output`/cache entries despite uv's cache-location warning.

A deterministic 4,096-vertex radial contour (`r=20+2*sin(17*t)`, uniformly sampled)
was inset by 0.5 mm, expanded by 0.5 mm and differenced from its opening. Ten runs
after one warmup had medians 0.294, 0.297, 0.300, 0.294 and 0.296 seconds respectively
in matrix order; the largest individual run was 0.369 seconds. Outputs were valid
and the inset stayed covered by the opening. This checks synthetic throughput,
not organic-shape accuracy, production latency, memory scaling or a planner budget.
There is no measured need for another backend or acceleration infrastructure yet.

### Failures, representation gaps and distribution obligations

- The default geometry-method disk buffer at radius 5 mm has area error
  0.1261040761 mm2 and sagitta 0.0060227190 mm, exceeding both corpus limits.
  Explicit refinement is necessary; the experiment does not bless the defaults.
- T01 is an expected limitation, never a passed nominal-feasibility case. Exact
  rectangle erosion returns `POLYGON EMPTY` on both GEOS lines instead of the
  required segment `(2,2)-(8,2)`. For the circle, GEOS 3.11.4 returns a tiny polygon
  of area `2.9222580301e-9 mm2`; GEOS 3.13.1 returns empty. Neither is the required
  point `(0,0)`. The initial runner incorrectly expected emptiness on both versions;
  that failed observation is retained as `py39-initial-erosion-finding.json`.
  The corrected report records actual type, extent and nominal dimension mismatch.
  The package's [polygonal buffer contract](https://shapely.readthedocs.io/en/stable/reference/shapely.buffer.html)
  makes dimensional recovery an adapter responsibility, not a reason to replace
  its general polygon kernel. Treating a tiny positive area as valid access would
  be an adapter error.
- A 0.01 mm precision grid changes T02's two regions separated by 0.004 mm into
  one component on both GEOS lines. Unsnapped union preserves the gap. This is
  consistent with [documented precision collapse](https://shapely.readthedocs.io/en/stable/reference/shapely.set_precision.html);
  an implicit grid would violate this project's topology contract.
- C01/C02 cutter-profile inversion/joins are outside planar scope. Analytic arcs,
  general lower-dimensional feasible sets, full XYZ occupancy, segment medial axes,
  stock evolution, entry/link motion and inlay assembly are not supplied or accepted
  by this experiment. Shapely's [manual](https://shapely.readthedocs.io/en/stable/manual.html#geometric-objects)
  explicitly describes planar analysis; carrying Z coordinates does not add a 3D
  engine. No unsupported capability was replaced by a claimed area-only success.

Installed wheels were inspected: Shapely has BSD-3-Clause `LICENSE.txt`, bundled
GEOS has `LICENSE_GEOS` identifying LGPLv2.1, and `LICENSE_win32` covers bundled
Microsoft runtime files. Native DLL/PYD paths and SHA-256 values are retained in
platform reports. Keep copyright/license notices with redistributed binaries;
bundling GEOS also requires an appropriate LGPL source/relinking-compliance route,
and bundled Microsoft runtime terms must be preserved. See the
[Shapely license](https://github.com/shapely/shapely/blob/main/LICENSE.txt) and
[GEOS license information](https://libgeos.org/usage/faq/). This experiment uses
upstream wheels locally and does not establish a redistributed/frozen application
compliance package. The future packaging owner must handle that if binaries are
bundled; declaring a dependency does not make those binaries MIT-licensed.

Supporting artifacts and the one-off platform/throughput probe are under
`output/shapely-evaluation-20260922-175502/`; durable results remain here.
Reproduction commands live in the [runbook](DEVELOPMENT.md#isolated-planar-backend-evaluation).
All five final geometry runs exited zero with 160 checks each: 11 planar cases
pass, T01 remains `expected_limitation`, and C01/C02 remain `out_of_planar_scope`.
The table above summarizes the final matrix; all reports match the final runner
and corpus hashes. A synthetic negative control changing G01's expected union
area from 150 to 151 mm2 correctly produced one failure and exit status 1.
The declared `.venv/Scripts/python.exe -m unittest discover -s tests -p
test_rest_vcarve_acceptance_fixtures.py -v` command passed all 13 reference tests
on the existing Python 3.14.5 project environment. `py_compile` passed for the
runner; 107 local documentation file links resolved, the untracked runner passed
explicit whitespace inspection, and `git diff --check` passed. The runtime suite
was not rerun because runtime code and dependency declarations did not change.
Manual CamBam validation adds no evidence to a detached planar experiment and is
not requested. The next adversarial acceptance/value-contract increment has a
distinct scope and persisted inputs, so this is a useful fresh-session breakpoint.

## Rest and V-carving design refinement - 2026-09-22

Acceptance-corpus increment: the user explicitly accepted proven packages and
requested concrete acceptance cases. Established geometry primitives are now the
first evaluation route; an independently built general kernel is not a prerequisite.
The [corpus](../tests/fixtures/rest_vcarve_acceptance.json) supplies 14 reusable cases,
with 13 [reference checks](../tests/test_rest_vcarve_acceptance_fixtures.py) covering
analytic areas/volumes, cutter profiles, sweeps and existing Region input validity.
They pass using the declared interpreter and focused unittest discovery command
in the [acceptance owner](REST_MACHINING_PLAN.md#acceptance-corpus-v1). The affine-radius
sweep reference was checked with a separate support-function integral; capped-target
volumes use section integration. No candidate backend, stock engine or path planner
was executed, so no backend/machining acceptance follows from these results.
Runtime source and dependencies remain unchanged. Manual CamBam validation adds no
evidence for this reference-corpus increment. The first dependency evaluation now
has stable inputs and quantitative gates; later motion/stock/inlay gates remain
explicitly pending in the design owner.
Focused unittest discovery passed all 13 tests; `py_compile` passed for the new
reference checker. All 92 checked local documentation links resolved, both new
files passed whitespace inspection, and `git diff --check` passed. The broader
runtime suite was not rerun because runtime source/contracts were not changed.

Follow-up scope clarification: the A is one fixture and inlay is one application;
neither defines the core. The user requested consideration of locally owned maths
and geometry algorithms. The design now compares a bounded in-repository kernel
with established primitives, records robust-predicate/construction obligations,
and requires a general-shape acceptance matrix. GEOS adoption evidence and
Shewchuk's predicate research were checked in primary sources. No backend was
installed, benchmarked or selected in this clarification round; build-versus-adopt
remains an engineering decision requiring evidence, not a user permission blocker.
This follow-up passed 85 local documentation-link checks and `git diff --check`;
no runtime tests or manual machining validation were needed for documentation.

The user accepted segmented finish modes, caller-owned tool catalogs and future
paired inlays, and moved the work to `feat/rest-machining-and-vcarving`. This round
changed design documentation only. The [design owner](REST_MACHINING_PLAN.md#shared-core-and-paired-inlay-design)
contains the mathematical contracts, primary research, backend shortlist and
remaining first-demonstration decisions; no runtime API or dependency was added.

Fifteen scalar analytic checks passed using `.venv/Scripts/python.exe`: pointed
and flat-tip cone inversion, spherical/conical join radius and slope at three
angles, ideal V target sections, and nominal inlay gap/fit relationships. These
check design identities, not an implemented sweep/planner or manufacturing behavior.
The local [inlay cross-section](../output/rest-vcarve-design-20260922-164431/inlay-gaps.svg)
illustrates a 4 mm pocket, 3.7 mm insertion, 0.3 mm bottom glue gap and 1 mm surface
clearance with nominal side contact. SVG XML parsing and coordinate identities
were checked; no browser-render or machine acceptance is claimed. The companion
`analytic-checks.json` records the checked identities in the same ignored directory.
The proposed A contour and triangular hole also passed the existing `Region`
topology validator; analytic shoelace areas are 1596 and 64 square millimetres,
giving 1532 square millimetres net opening area. This checks the proposed input,
not the unimplemented V-target or toolpath output.

Rejected design shortcuts: define success from the actual tool's own sweep; treat
an estimated native MOP as known-clear stock; use a uniform expansion of rest as
the finished boundary; equate stepover with engagement; or implement inlays as a
second path engine. Reopen only with an explicit alternative contract and evidence
that required geometry, topology and motion constraints remain satisfied.

Manual CamBam validation adds no evidence for this documentation round. Native
varying-Z motion and manufactured inlay fit remain separate future acceptance.

A bounded independent mathematical review confirmed the ideal distance-field
target, affine-radius sweep direction, stock-bound ordering and separate inlay
assembly model. It prompted explicit spherical-tip versus rounded-flat distinction,
the spherical/conical profile's finite-height formula, and the statement that a
capped V-carve has a flat floor only where the opening is wide enough. This is
design review, not independent implementation verification. All 84 checked local
documentation links resolved (ignored output artifacts excluded); `git diff --check`
passed. No runtime regression suite was run for these documentation-only changes.

## Pre-merge review of 8a–8e and 9a–9c - 2026-09-21

Reviewed the twelve-commit sequence and aggregate `main...3cfa1df` change against
the implemented architecture, MOP preservation/MCP contracts and recorded native
acceptance. Initial working tree was clean (`git status --short --branch` also
warned that `.pytest_cache/` could not be opened). This was a focused review, not
another full verification run. The reported 293-test baseline, existing Windows
symlink skip, compileall and schema checks were not rerun.

Three P2 (medium) issues block merging the advertised 9b/9c contracts. They are
framework composition defects, not questions about native CamBam behavior. Runtime
code was left unchanged; repair must stay in recommendation selection/planning and
their focused regressions, without starting the entity-module refactor.

### R1: Fixed override selection depends on strategy order

Evidence: `machining_recommendations.py:480-489` raises immediately on two
conflicting non-fixed recommendations, before a later fixed override is considered.
With chip-load recommendations 0.03 and 0.04 mm/tooth and a fixed user override of
0.025, two of the six strategy permutations raise `RecommendationError`; four
return 0.025. This violates `structure_spec.md`'s recommendation contract that a
fixed override wins regardless of order. A valid user resolution can therefore be
rejected solely because of profile ordering.

Correction: validate and collect applicable candidates before resolving conflicts
per field. Select a compatible fixed candidate when present; reject unresolved
non-fixed conflicts and incompatible fixed candidates. Keep units, provenance,
applicability and entry-capability validation; do not bypass validation by sorting
and short-circuiting. Acceptance: all six permutations return the same fixed
recommendation/provenance; two conflicting suggestions without an override still
fail; conflicting fixed values still fail in every order. Existing tests only
permute one non-fixed strategy with one fixed strategy and miss this case.

### R2: An RPM-only cap creates a false feed conflict

Evidence: `machining_planning.py:239-250` caps a non-fixed spindle recommendation
and drops only its surface-speed conjugate. It then passes non-fixed chip load and
feed into the kernel as supplied constraints. For a two-flute tool, the consistent
targets 8000 RPM, 0.04 mm/tooth and 640 mm/min become 6000 RPM, 0.04 mm/tooth and
640 mm/min under a 6000 RPM cap and 1000 mm/min feed cap. Planning raises
`conflicting feed inputs: feed_rate=640.0, expected 480.0`. This violates the 9a/9c
cap-propagation contract in `structure_spec.md:79-84,130-139`: a valid capped
starting candidate should produce achieved downstream values and diagnostics.

Correction: preserve whether operational inputs are fixed or suggested when
composing the solver request. Apply the existing chip-load-to-feed dependency to
non-fixed feed targets after an RPM cap, rather than treating the old feed as fixed.
Do not weaken the pure kernel's checks or suppress genuinely inconsistent original
constraints. Acceptance: this example returns 6000 RPM and 480 mm/min with the
spindle cap recorded; MRR/power/torque use achieved settings. Cover RPM-only,
feed-only and simultaneous caps, and a fixed feed variant that remains unchanged
with achieved chip load recalculated. The existing direct-target test caps both RPM
and feed, which removes chip load and masks the RPM-only failure.

### R3: Planner silently discards conflicting fixed strategy values

Evidence: `machining_planning.py:235-236` skips any recommendation whose field is
already in `supplied`, including `fixed=True` recommendations. A fixed strategy
setting 5000 RPM plus `fixed_values=MillingConstraints('mm', spindle_speed=4000)`
returns 4000 RPM without a conflict or diagnostic, while retained recommendation
provenance still says fixed 5000 RPM. The specification at `structure_spec.md:130-132`
allows explicit constraints to override **non-fixed** recommendations; it does not
authorize silently resolving contradictory fixed decisions. The same skip also
affects fixed effective-flute recommendations that conflict with the tool profile.

Correction: before skipping a supplied field, compare a fixed recommendation with
the explicit/tool-profile fact and reject incompatible values with a field-specific
error. Preserve ordinary non-fixed override behavior and the intentional safe-axial
maximum versus actual-stepdown distinction. Acceptance: equal fixed values succeed;
5000-versus-4000 fixed RPM fails; conflicting fixed effective-flute recommendations
fail against the tool profile; explicit inputs still override non-fixed values;
through-cut axial maxima still produce balanced actual stepdown. Existing tests
cover fixed values above machine caps, not two conflicting fixed sources.

### Review verification and disposition

No additional merge-blocking MOP defect was established. The reviewed policy
inventories, template preservation, supported method switches, group-neutral target
eligibility and inspection/schema contracts remain cohesive with recorded native
evidence. Default cached text is not an evaluated style value, and inactive raw
fields do not establish toolpath behavior. Two inspection concerns were not promoted
to findings: effective Part defaults can differ from raw attributes when a fresh
core project is injected directly into the service, but normal MCP authoring pins
those inputs and XML import materializes their native values; known inactive fields
on an unknown Drill method remain raw values marked inapplicable, while unknown
paths remain opaque. Neither establishes an incorrect supported MCP workflow.
Native production/toolpath acceptance and external CAM-style resolution remain
outside the implemented guarantee; this review did not repeat accepted fixtures.

The three failures were reproduced through the public package API using synthetic
tool/material/provenance data. Retained helper:
`output/premerge-review-20260921-3cfa1df/reproduce.py`. Exact repeatable command from
the repository root:

```powershell
Get-Content output/premerge-review-20260921-3cfa1df/reproduce.py | .venv\Scripts\python.exe -
```

Exit 0 confirms the helper observed all three defects; it is not a passing product
regression suite. An earlier equivalent inline probe produced the same failures.
`git diff --check` passed after recording the review (only Git's LF/CRLF conversion
warnings); `git check-ignore` confirmed the synthetic helper remains ignored.
No manual CamBam validation adds evidence for these pure-library repairs. Merge
recommendation: **blocked** until R1-R3 are corrected and focused acceptance passes.
Next priority is recorded in `PROGRESS.md`; backlog 10 remains deferred. A fresh
session can implement the repairs from this record without conversational context.

## Staged R1-R3 and 9d review - 2026-09-21

Reviewed all eleven staged files; initially there were no unstaged changes. R1's
collect-before-select repair, R3's same-field comparison, immutable arbitrary
machine/job interval intersection and the original direct-RPM R2 reproduction are
covered by appropriate focused tests. Two P2 issues still block merge; no runtime
edits were made. This supersedes the completion/merge-ready claim below.

1. **Explicit fixed inputs can change after range adjustment.** At
   `machining_planning.py:321-331`, the post-solve fixed-value check visits only
   fixed strategy recommendations, not explicit `fixed_values`. With a 6 mm,
   two-flute tool, machine `max_feed_rate=300`, no strategies, and
   `fixed_values=MillingConstraints('mm', spindle_speed=5000, chip_load=0.04)`,
   the plan returns chip load 0.03 and feed 300 without rejecting the changed fixed
   chip load. An equivalent fixed strategy is checked and rejected. This violates
   the requirement to recheck fixed constraints after adjustment and gives two
   public ways of supplying a fixed decision different meanings. Check all
   explicit fixed inputs as well as fixed recommendations against the achieved
   solution, retaining the documented through-cut axial maximum exception. Reject
   infeasible fixed combinations with field-specific errors; do not weaken the
   kernel's separate requested-versus-achieved contract. Acceptance: this example
   must fail for chip_load; equivalent fixed-strategy input must behave identically;
   non-fixed chip load may adjust; fixed RPM/feed and balanced depth regressions
   must continue to pass. Include lower as well as upper feed bounds.

2. **Derived RPM adjustment still treats a suggested feed as fixed.** At
   `machining_planning.py:263-298`, stale feed removal depends on the spindle
   recommendation being explicitly present and adjusted in the planner. When RPM
   is derived inside the kernel, that path is skipped. For the same 6 mm/two-flute
   tool, non-fixed surface speed `8000*pi*6/1000` m/min, chip load 0.04 mm/tooth,
   feed 640 mm/min, and machine bounds `max_spindle_speed=6000,
   max_feed_rate=1000`, the plan returns 6000 RPM, feed 640 and chip load
   0.05333333333333334. The equivalent direct RPM recommendation returns feed 480
   and chip load 0.04. This violates the documented RPM-adjustment propagation
   contract (`structure_spec.md`, Milling pass and candidate planning). Preserve
   fixed-versus-suggested ownership across both direct and derived RPM adjustment;
   do not pass a stale non-fixed feed target as a fixed kernel setting. Acceptance:
   direct RPM and equivalent surface-speed inputs produce 6000 RPM/480 mm/min,
   matching achieved MRR/power/torque and retained provenance; explicitly fixed
   feed remains 640 with achieved chip load recalculated. Exercise minimum RPM
   adjustment and coupled feed bounds too.

Both cases were reproduced using an inline synthetic script through public package
imports with `.venv\Scripts\python.exe -`. Each recommendation used shop provenance
and `ApplicableRange('cutter_diameter', 'mm', 1, 10)`; no native/user assets were used.
The script printed the actual values listed above. Verification:
`.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_machining_*.py' -q`
ran 44 tests, all passing. `git diff --cached --check` passed. Full-suite,
compileall and native CamBam checks were not repeated; native validation adds no
evidence for these pure composition defects. Only review/status documentation was
edited, left unstaged; the user's staged implementation was preserved. A fresh
session can repair these two cases from this record; backlog 10 remains deferred.

### Resolution - 2026-09-21

Both composition defects are corrected in the planning owner. The planner now
retains the unbounded validated solution long enough to detect direct or derived
RPM range adjustment, removes a stale suggested feed before the final solve, and
lets a coupled feed bound apply to the feed recalculated from achieved RPM. Fixed
feed remains exact and recalculates achieved chip load. After solving, every
explicit `fixed_values` field and every fixed strategy field is compared with the
achieved value; only the documented balanced through-cut axial maximum differs by
design. Upper and lower feed-bound regressions cover changed fixed chip load, and
derived upper/lower RPM regressions cover feed, MRR, power, torque, provenance and
constraint ordering.

Verification from the repository-managed interpreter: all 46 focused machining
tests and all 302 repository tests pass, with the existing Windows symlink-privilege
skip in the full suite. Compileall, public import/construct smoke, `git diff --check`
and `git diff --cached --check` pass; the diff checks report only the existing
LF/CRLF conversion warnings. Native CamBam validation would add no evidence for
this pure, nonserialized composition change. The staged-review merge block is
resolved; backlog 10 remains the next distinct increment.

## R1-R3 repair and caller-defined operating ranges - 2026-09-21

The three pre-merge planning findings above are corrected, and backlog 9d is
implemented in the same pure public calculation/recommendation/planning owners.
R1 now collects and validates every applicable candidate before resolving a field:
all six permutations of two conflicting suggestions plus one fixed override select
the same fixed value and provenance, while unresolved suggestions and incompatible
fixed values still fail. R2 validates the original coupled candidate, then lets an
adjusted non-fixed RPM own achieved feed: the reproduced 8000 RPM, 0.04 mm/tooth,
640 mm/min target under a 6000 RPM-only cap returns 6000 RPM and 480 mm/min; a fixed
640 mm/min feed remains exact and recalculates achieved chip load. R3 compares fixed
strategy values with explicit inputs and tool facts before skipping them; equal
values succeed while conflicting RPM and effective-flute values fail by field.

`MachineCapabilities` now accepts optional minimum as well as maximum RPM/feed
bounds. The new immutable `OperatingConstraints` record carries optional setup/job
bounds in explicit metric or imperial units. `RecommendationContext.effective_limits()`
intersects the two records, rejects empty intervals and retains independent context
state. Omitted bounds add no limit; equal endpoints are valid. Non-fixed direct or
derived RPM, cut-feed and entry-feed targets move to an effective lower/upper bound
with ordered direction-bearing `ActiveConstraint` evidence. Fixed values outside
the range fail. The final coupled solution recalculates surface speed, chip load,
MRR, power and torque; power/torque overage remains diagnostic without derating.
No range creates a missing cutting recommendation, changes capability/provenance
requirements, or establishes that a raised minimum is safe.

Verification from the repository-managed interpreter: 46 focused 9a-9d tests pass;
all 302 repository tests pass with the existing Windows symlink-privilege skip;
compileall and the public import/construct smoke check pass. `git diff --check`
passes with only the repository's LF/CRLF conversion warnings. This pure,
nonserialized arithmetic change needs no native CamBam validation. No catalog,
persistence format, MCP profile schema or document mutation was added. The merge
block recorded above is resolved. Backlog 10 is now the next coherent increment.

## Group-neutral MCP MOP target eligibility - 2026-09-20

Backlog 8e removed group membership from two decisions where it has no coordinate
meaning: typed primitive inspection and explicit MOP target eligibility. The old
implementation routed target validation through the geometry-mutation slice, so
adding an otherwise harmless framework group both blanked typed geometry and rejected
MOP authoring/retargeting. Intrinsic finite similarity geometry is now checked once,
while inspection, geometry mutation and MOP targeting own distinct relationship
policies. Inspection and MOP targeting allow groups but still reject a parent or
child; geometry mutation retains its narrower no-group rule.

The focused regression covers Rect, Circle, Arc, open and closed Pline, Points, Text
and Region before grouping, while grouped and after group removal. It authors all four
MOP families against grouped primitives, retargets them across every supported
family/kind combination, and compares exact UUID target lists plus the complete
`parameters`, `parameter_metadata` and `unsupported_fields` records after save/reopen.
Typed geometry remains byte-for-byte equivalent at the structured-record level.
Parent and child retarget attempts remain rejected; the existing non-similarity
regression remains unchanged, and the zero-local-Z gate remains explicit in the
intrinsic geometry predicate.

Broadening the general geometry-mutation slice was rejected because that would also
change transform and Region-replacement behavior without need or acceptance evidence.
Treating groups like parent/child coordinate relationships was rejected because group
membership changes selection metadata only. Parent/child, non-similarity and nonzero
local-Z targets were not inferred safe from this result and remain outside the MCP
machining slice.

Verification: the 9-test MCP MOP suite, 5 relationship/transform tests and 11 MCP
authoring tests pass; all 86 MCP tests pass with the existing Windows
symlink-privilege skip; all 256 repository tests pass with that same skip. Compileall
for the modern package, legacy package, tests and demos and `git diff --check` pass.
Manual CamBam validation would add no evidence for this metadata-only eligibility
change because exact exported/reopened native targets and parameter records are
already compared; production toolpath safety remains out of scope.

## Preservation-aware structured MOP inspection - 2026-09-20

Backlog 8d replaced the MCP adapter's all-or-nothing MOP inspection gate. The old
gate coupled read fidelity to canonical authoring pins, required every inspected XML
path to be explicit `Value`, and also required a nonempty eligible target selection.
Consequently one alternate pin, inherited container or empty/group target erased the
whole typed parameter record even though the core safely preserved it.

The accepted contract retains flat `parameters` for direct value access and adds
field-scoped `parameter_metadata` for native state and policy applicability.
Presence/state is derived from the native template (or fresh policy XML), never the
reader's backfilled constructor baseline. A `Default` container governs nested leaves
even when a cached child says `Value`; omitted nodes have metadata but no reported
value. Parameter inspection is independent of target eligibility, and `target_group`
distinguishes a live source from explicit targets. Maximal opaque subtrees and unknown
parameter attributes are named in `unsupported_fields` and the diagnostic message,
while their content stays private and round-trip-preserved.

Rejected interpretations were: wrapping every value in a new per-field object, which
would unnecessarily break the established flat access path; continuing to return
constructor defaults for omitted imported XML, which would falsely imply style
resolution; accepting nested child `Value` beneath a `Default` container as active;
and using geometry/target eligibility as a proxy for parameter readability. Literal
CustomScript, Manual tab point collections and unsupported lead modes remain opaque
rather than acquiring nominal typed support.

Evidence: two focused regressions cover all four MOP families, alternate pins,
top-level and nested inheritance, supported/inapplicable method fields, unknown child
and attribute paths, literal CustomScript, explicit empty targets, a live group source
and unchanged opaque XML after save. The 43 adjacent MOP/MCP tests pass, and the full
255-test suite passes with one existing Windows symlink-privilege skip. No manual
CamBam validation adds evidence for this inspection-only contract; reopening is
justified by a malformed modeled value leaking a parser fallback, an unreported
opaque path, or a native state/applicability contradiction.

## Delegation routing correction - 2026-09-09

The initial revision mixed project policy, role selection, provider setup and worker
execution behavior. It also omitted the Sol role registration, duplicated stale role
descriptions, used a false-precision score and recorded availability before an
invocation passed. The corrected ownership is: `AGENTS.md` states the short project
rule, [MODEL_ROUTING.md](MODEL_ROUTING.md) is the sole role selector,
[WORKFLOW.md](WORKFLOW.md#delegation-execution) owns packet/integration mechanics, and
user-level role TOMLs govern only the selected worker's execution.

Provider evidence: a direct read-only `codex exec` using `model_provider=openrouter`
and `z-ai/glm-5.3-flash` returned the exact requested marker. A native-parent custom
GLM spawn failed with HTTP 400 because Codex 0.150.1 reapplied the ChatGPT account
transport to the OpenRouter child. This is a mixed-provider limitation, not an API-key
or model-name failure. The corrected user-level `openrouter-glm` profile selects
OpenRouter/GLM and overrides the inherited subagent model/effort defaults. A fresh
profile-backed process spawned `glm_retriever`; the consolidated-profile retest returned
`GLM_SUBAGENT_ROUTE_OK Model routing`. A separate fresh native process spawned the newly
registered `sol_scoped_worker` and returned `SOL_SUBAGENT_OK`. All tests were read-only.

Reopen routing if reviewed worker output shows regressions or coordination cost
erases the expected benefit. Reopen provider setup if a profile-backed GLM subagent
fails or a later Codex release claims cross-provider children are supported; retain
exact errors rather than inferring model quality or access.

## Scope and workflow assessment

The initial working tree was clean. This pass changed documentation and ignored
local-environment configuration only; no runtime implementation was changed.
The modern package is the review scope. Legacy and inactive code were preserved.

Useful existing practices: modular project/entity/XML/matrix boundaries, a central
relationship design, dependency declarations, examples and a progress tracker.
Main gaps were an almost empty README, no agent entry point or topic ownership,
overconfident round-trip status, broad overlapping increments, and no dedicated
assertion-based test suite or CI. The specification repeats transformation concepts
in sections 3.3 and 6 and summaries; those remain in one design owner rather than
being copied into new guides. Its MOP PID-source wording is ambiguous relative to
central registry ownership. The progress tracker is now the sole priority owner.

No separate decision/failure records or tracker links were found in the existing
tracked documentation. In-code TODOs and comments are evidence of unresolved work,
not proof that experiments were run. No historical logs or user CAD assets were
loaded. This record starts durable negative knowledge without inventing history.

Session capabilities: local PowerShell/Python, patching, repository search and
native subagents were callable. One native read-only retrieval worker reviewed
matrix code; the lead checked its actionable finding locally. External-provider
credentials/routing and price telemetry were not inspected or claimed available.
No external provider, web lookup, account connector or added service was needed.
The MIT license does not relax private-input or external-processing boundaries.

## Ranked engineering findings

Order balances impact, risk, leverage, confidence and cost; the active priority
and remaining backlog live only in [PROGRESS.md](PROGRESS.md).

| Rank | Finding and classification | Impact / confidence / approximate cost | Leverage |
| --- | --- | --- | --- |
| 1 | **Verified:** parent links are lost on XML round trip; a UUID-only repair also changes world geometry | High impact/confidence; small-to-medium coherent I/O slice | Establishes hierarchy/geometry regression foundation |
| 2 | **Verified:** two-node parent cycles are accepted | High: traversal can fail to terminate; high confidence; small validation slice | Protects every hierarchy consumer |
| 3 | **Verified:** non-baked global transform order contradicts the method contract | High impact; high confidence for root case; medium cost across nested/baked cases | Restores predictable editing operations |
| 4 | **Verified:** duplicate MOP display names lose operations during XML import | High: machining operation omitted; high confidence; small-to-medium cost | Preserves machining intent through interchange |
| 5 | **Verified:** export tree construction catches entity serialization exceptions and returns incomplete output | High: incomplete output appears successful; high confidence; medium cost for failure/atomic-write contract | Reliable success/error boundary for Python and future MCP |
| 6 | **Verified:** bare-filename state save returns without creating a file | Medium impact; high confidence; low cost | Small persistence regression and clearer I/O behavior |
| 7 | **Source-confirmed maintenance risk:** converted Rect duplicates primitive Tag construction | Low-to-medium impact; high confidence in duplication, future drift risk; low cost | One metadata owner when serialization next changes |
| 8 | **Known gaps / enhancements:** MOP registry migration, project-default import, curved bounds, transfer APIs, packaging/test coverage | Variable impact; source/TODO/spec evidence; separate bounded increments | Extend capability after fidelity and tests |

### Parent identity and transform reconstruction

`cambam_writer.build_xml_tree` obtains a `Primitive` from
`CamBamProject.get_parent_of_primitive` and passes it as `parent_uuid`.
`Primitive._add_common_xml_attributes` converts it with `str()`, producing an entity
representation instead of a UUID. `cambam_reader._reconstruct_primitive` cannot
parse that parent reference. A synthetic two-rectangle round trip kept both
primitives but lost the child's parent.

The entity writer exports `get_total_transform()`. The reader stores that world
matrix as `effective_transform` and later links parents without converting to
local coordinates. Editing only the synthetic XML parent field to the correct
UUID changed the child's world X translation from 10 to 20 after import.

**Negative knowledge:** a writer-only UUID substitution is insufficient. Reopen
that approach only alongside a tested world-to-local reconstruction rule. Likewise,
do not recursively apply a non-baked parent transform to descendants already
inheriting it; `transform_primitive` comments identify this double-application risk.

### Parent round-trip repair verification

2026-09-07: implemented the bounded parent XML repair. The writer now supplies
the parent's UUID. The reader snapshots XML world matrices and solves for each
child's local matrix before assigning it with the resolved parent link. Numeric
XML ID references accept strings and JSON integers; malformed/unresolved parents
retain the imported world pose without a parent. The current contract lives in
`docs/structure_spec.md`, section 0, “XML parent identity and world-pose contract.”

Decision: a singular resolved parent fails the whole import through the existing
logged-error/`None` boundary, including when the child world matrix is compatible.
World-only XML cannot uniquely recover the local transform. Pseudoinverse recovery
and silent detachment were rejected because they introduce arbitrary local state
or lose hierarchy. Reopen if an explicit local-matrix metadata format or product
requirement supplies a reconstruction rule. Singular roots without children load.

Authored reproduction: `python -m unittest discover -s tests -v`. The initial
eight tests ran before runtime edits. A subsequent isolated temporary copy of the
original HEAD reader/writer with the expanded nine-test suite returned exit 1:
four failures (UUID emission, world reconstruction, reversed order, singular
parent rejection) and one error (numeric parent reference remained unlinked).
No working-tree runtime files were reverted for that comparison.

After the repair, all nine tests pass on the available Python 3.10.9 / NumPy
1.23.5 interpreter without installing dependencies. Tests inspect synthetic XML
IDs, parent tags and world matrices, and assert hierarchy in both directions,
primitive identities, layer membership and world geometry through two round
trips, including reordered XML. Numeric comparisons use `rtol=0, atol=2e-8`.
Missing/malformed/self references, singular parents and singular roots are covered.
`python -m compileall -q cambam_builder legacy_cambam_builder tests`, the runbook's
import/construct smoke command, and `git diff --check` pass.

This is automated acceptance of the bounded contract, not CamBam or packaging
acceptance. CamBam placement/geometry checks remain required. General transform
order, baking/Rect conversion, cycles, MOP migration and legacy behavior were not
repaired or established by these tests.

### Parent cycles

`CamBamProject.link_primitive_parent` rejects self-links but not ancestor cycles.
Create A, create B with parent A, then link A to B: it returns `True` and both
relationships exist. `Primitive.get_total_transform` walks parents in an unguarded
`while` loop. An initial traversal probe did not finish and was interrupted;
the bounded reproduction verifies the cycle without traversing it.

### Global transform order

`CamBamProject.transform_primitive` promises a global operation but computes
`effective_transform @ matrix`. On a root rectangle, translate by (10, 0), then
rotate 90 degrees about explicit origin (0, 0): the local origin remains at
(10, 0), where a global rotation yields (0, 10). The lead reproduced this through
the public convenience methods. A nested target requires conversion through its
parent frame; simply changing multiplication order everywhere is not an accepted
fix. Baked/non-baked equivalence needs independent tests.

### Other bounded observations and uncertainties

- `build_xml_tree` catches primitive/MOP exceptions and returns the remaining tree;
  artifact completeness must be checked in addition to successful file creation.
- `read_cambam_file` leaves project defaults as a TODO. `Mop.pid_source` remains
  the implemented MOP relationship source; central migration is an enhancement.
- Pline bulge bounds, arc bounds and text bounds are marked approximate in
  `cambam_entities.py`; rotated/sheared rectangle baking warns of approximation.
- The primary matrix encoder/decoder are internally paired. Their comments and
  alternative v2 functions do not establish CamBam compatibility or prove a
  layout defect. Do not replace them based solely on row/column terminology;
  use a known CamBam fixture and geometry checks. Invalid matrix lengths also need
  a controlled parsing contract; this pass did not broaden into parser hardening.
- No exhaustive source-system support matrix, supported-Python build validation,
  legacy compatibility audit or real machining acceptance was performed.

## Recommendation from this review

The parent round-trip finding warrants the smallest coherent repair across writer
and reader. Current scope, acceptance criteria and priority are owned exclusively
by [PROGRESS.md](PROGRESS.md). This dated review owns the defect evidence and
the reasons a writer-only repair is insufficient.

## Validation performed

Environment: available Python 3.10.9, NumPy 1.23.5 on Windows. No dependencies were
installed. No project-managed environment or lockfile existed. Results below are
local checks, not claims about every declared Python version.

| Exact command | Result |
| --- | --- |
| `git status --short` (before editing) | Clean |
| `python -m compileall -q cambam_builder legacy_cambam_builder` | Exit 0 |
| `python -c "from cambam_builder import CBProject; p = CBProject('smoke'); assert p.project_name == 'smoke'; print('import/construct OK')"` | Exit 0; import/construct OK |
| `python -m output.review_baseline` | Exit 0; parent loss and accepted two-node cycle confirmed |
| `python -m output.review_parent_transform` | Exit 0; corrected synthetic UUID alone gives X=10 -> 20 |
| `python -m output.review_transform_order` | Exit 0; global-origin rotation discrepancy confirmed |
| `python -m output.review_doc_links` | Exit 0; local Markdown file targets exist |
| `git diff --check` | Exit 0; only line-ending conversion notices |

The three synthetic diagnostic scripts and documentation-link checker are retained
locally under ignored `output/`; they are not a committed regression suite and
assert current defects, not desired behavior. They use synthetic data only and
temporary XML files. The initial unbounded cycle probe was interrupted and replaced
with bounded registry assertions. Future regression tests must assert desired
behavior using the scenarios above, not preserve those diagnostic assertions.

Working-agreement implementation and local documentation verification are complete.
Runtime repairs, package installation/build, automated regression coverage and user
acceptance remain unperformed. No staging or commits were made.

Suggested commit message: `docs: establish agent workflow and evidence-based project status`

## Phase 1 and 2 completion audit

Follow-up on 2026-09-07: the working tree was clean at the start. A native read-only
documentation worker independently checked gaps; the lead validated ownership and
data flow against targeted implementation symbols. No runtime changes, private
assets, dependency installation or external-provider processing were needed.

The initial pass covered the foundations. This follow-up closes gaps in current
architecture, interpreter selection, lifecycle/evidence recording and single-owner
acceptance criteria. The user requested completion of these phases; product repair
remains a subsequent increment.

| Requirement | Authoritative result / completion evidence |
| --- | --- |
| Inspect working tree; preserve changes | Clean initial status; documentation-only final diff |
| Locate entry points, architecture, status, plans and checks | README -> topic map -> named owners; existing tracker/test/CI limitations recorded |
| Current architecture and ownership | Specification section 0, grounded in project/entity/reader/writer symbols; target design remains explicitly separate |
| Authored/generated/runtime/legacy boundaries | Topic map artifact rules and specification module ownership |
| Toolchain, commands and troubleshooting | Development runbook with explicit interpreter selection and checks by change type |
| Status, backlog, decisions and rejected approaches | One PROGRESS surface; workflow lifecycle; dated review evidence with reopening conditions |
| Competing documents | Live acceptance moved from this review into PROGRESS; module overview moved from the map to the specification |
| Capabilities, privacy, security and licensing | Session observations above; capability discovery/delegation rules; private-data boundaries, trusted-pickle rule and MIT owner |
| Compact agent contract and judgment | Root AGENTS; minor reversible choices autonomous, material missing decisions escalated |
| Progressive context and no competing wiki | Workflow search examples, evidence-triggered expansion/stopping rules and documentation maintenance ownership |
| Agile lifecycle and acceptance separation | Explicit backlog/active/blocked/completion states and reusable completion record |
| Cost-aware delegation | Capability matrix, bounded packet, total coordination cost, exclusive ownership, evidence escalation and local lead validation |
| Verification and handoff | Per-change minimum evidence and exact-results template; future test entry point required before runtime closure |
| Avoid unnecessary infrastructure | Existing documents reused; no nested agent files, new dependencies, services or indexing systems |

Implementation: complete for phases 1 and 2. Documentation validation: complete.
User/production acceptance is not required for these repository workflow documents;
no source-system behavior changed. This does not claim user approval of runtime
behavior or CamBam acceptance. No unresolved user decision blocks use of the
working agreement. External tracker existence and full product compatibility remain
explicit uncertainties rather than assumptions presented as facts.

Follow-up checks (available Python 3.10.9, without installing dependencies):

- `git diff --check`: passed.
- `python -m output.review_doc_links`: passed; checked current local file targets.
  This pre-existing local helper is supplementary, not required infrastructure.
- README Python code block executed with `python -`: passed, including an assertion
  that the returned project contains the `outline` primitive.
- Inspected the final changed-file diff, ownership links and referenced heading
  targets. No runtime file changed; no staging, commits or publication performed.

Next increment: the selected parent round-trip slice in PROGRESS. Suggested commit:
`docs: complete project overview and agent working agreement`

## Phase 3 completion audit

Completed 2026-09-07 after the working agreement was established. The original
review supplied the main correctness findings and next-slice recommendation. This
follow-up adds bounded I/O and modularity coverage, local verification of three
additional failure cases, and explicit leverage in the ranking above. A native
read-only worker supplied focused findings; the lead checked their source evidence
and reproduced the two additional I/O defects. No product fixes were made.

### Additional evidence

- **Duplicate MOP names:** `cambam_reader._reconstruct_mop` assigns `mop_name` as
  the identifier, while `_register_entity` rejects duplicate identifiers. Creating
  two profiles with identifiers `first`/`second` but display name `Same`, targeting
  the same rectangle, is accepted by the public API. Write/read of that synthetic
  project returns one MOP from an original two. Fix identity reconstruction at the
  I/O boundary; do not silently require globally unique display names in callers.
- **Bare state path:** `save_state('state.pkl')` calls `os.makedirs('')`, catches
  the error and returns before writing. The lead reproduced this in a fresh
  temporary working directory and asserted no file exists. Other write errors in
  that method raise, while `read_cambam_file` generally returns `None` on failure.
  A future adapter needs a deliberate error mapping; wrapping these return values
  without checking outcomes is insufficient.
- **Incomplete XML:** with one rectangle, patch its class's `to_xml_element` to
  raise `ValueError('synthetic serialization failure')`; `build_xml_tree` returns
  normally, with zero `./layers/layer/objects/*` nodes. This upgrades the earlier
  source-level risk to an injected-failure reproduction. It does not establish
  filesystem atomicity or exhaustively test MOP failures.
- **Metadata duplication:** `Rect.to_xml_element` replaces the Tag after conversion
  to Pline, separately from `Primitive._add_common_xml_attributes`. The duplicated
  fields and filtering/formatting differ. This is a maintenance risk, not proof
  that every converted rectangle is currently wrong. Consolidate at the owning
  serialization abstraction when a covered change requires it.

### Architecture and coverage judgment

The project/entity/matrix/reader/writer split is adequate for the next increments.
The reader/writer directly depend on private registries and entities encode XML;
these are coupling points worth testing, not sufficient grounds for a broad layer
rewrite. Bidirectional indexes and duplicate entity/group metadata need invariant
tests before expanding mutation or transfer APIs. Do not start by migrating MOP
ownership: its live-group semantics are unresolved and would broaden the repair.

Source-system coverage is bounded by the reader tag maps and corresponding entity
encoders. Unsupported tags are skipped; project defaults and approximate curved
bounds remain documented gaps. Full fidelity needs authorized CamBam fixtures and
domain validation. This initial review is complete without pretending to be an
exhaustive feature, security or machining certification.

| Original phase 3 requirement | Result |
| --- | --- |
| Review correctness, architecture, duplication, I/O, validation, tests, modularity and source behavior | Ranked findings, additional evidence and architecture judgment above; earlier transform/hierarchy probes retained |
| Rank by impact, risk, leverage, confidence and cost | Updated ranking; PROGRESS owns execution order |
| Separate verified defects, hypotheses and enhancements | Classification explicit; source compatibility and future metadata drift not claimed proven defects |
| Check decisions/failures before recommending | Reviewed existing record; UUID-only and double-transform approaches remain rejected without new evidence |
| Recommend smallest high-impact slice with acceptance, owners, tests, user validation and rollback | Parent round-trip slice in PROGRESS remains the recommendation; newly found I/O defects do not force a wider first patch |
| Avoid broad speculative refactoring | No runtime edits; retain current module boundaries and repair demonstrated contracts |

Checks performed this follow-up: two synthetic programs run with PowerShell
here-strings piped to `python -`, both exit 0. The first used
`unittest.mock.patch.object(type(rect), 'to_xml_element', side_effect=ValueError(...))`
and asserted the original count was 1 and output count 0. The second used
`tempfile.TemporaryDirectory`, `save_state('state.pkl')`, and `build_xml_tree` /
`read_cambam_file`; assertions confirmed missing state output and MOP count 2 -> 1.
Inputs and assertion scenarios are specified above; no user fixtures were loaded.
`python -m output.review_doc_links` and `git diff --check` passed after documentation
updates. No packaging, server/client interoperability or CamBam acceptance was run.

All three initial phases are complete at their requested discovery/agreement/review
scope. Runtime implementation and user/domain acceptance remain pending. The new
MCP plan records user-authorized future direction, not a reason to defer the
selected correctness fix or claim an implemented integration.

Suggested commit: `docs: backlog stateless MCP integration and complete initial review`


### Parent-cycle rejection verification

2026-09-07: `link_primitive_parent` now walks the proposed parent's ancestor
chain before removing the child's old edge. Self/descendant links and already
cyclic candidate chains return `False` without mutation. Iteration with a visited
set avoids introducing a recursion limit or hanging on malformed prior state.
The mutation API owns this invariant; no transform or MOP refactor was needed.

The new cyclic-XML regression failed before the fix in both forward/reversed
XML order: three parent edges were retained instead of two. With the guard,
import rejects the closing edge and preserves all three world matrices and
geometries across another export/import. Which edge is rejected depends on XML
order; malformed cycles have no uniquely intended root. Existing singular-parent
import failure behavior is retained. Direct dictionary mutation/old pickle repair
and general traversal hardening remain outside this API guarantee.

The run also exposed an existing UUID-order-dependent singular-parent test:
either root or child can be the first singular parent encountered. Its assertion
now checks the reported child/parent pair against either valid failing edge.

Verification: `python -m unittest discover -s tests -v`,
`python -m compileall -q cambam_builder legacy_cambam_builder`, import/construction
smoke check, and `git diff --check` pass with the available Python 3.10.9 / NumPy
1.23.5 (no project environment or dependency changes). API tests check atomic
registry and transform preservation plus valid hierarchy mutations; XML tests
inspect parent edge counts, traversal, world poses and repeated round trips.
CamBam geometry/placement acceptance and packaging compatibility remain unverified.
Reopen if another supported relationship mutation bypasses the guard, or cyclic
XML requires an order-independent rejection policy.


### Parent A/B user acceptance

2026-09-07: the user confirmed all listed coordinates in both
`A_reference.cb` and `B_parent_roundtrip.cb` from the prepared parent validation
case. B achieves placement through transform properties; resetting them to
identity restores original local placement, as expected. This accepts the
synthetic parent XML/display criteria in the development runbook. CamBam version
was not supplied; no machining, curved geometry or general baking acceptance is
inferred. The one-off generator was archived under the same ignored local output
directory and removed from the tracked demos; original source remains in git
history, with reusable regression coverage retained in `tests/`.


### Full-bake hierarchy verification

2026-09-07: `bake_primitive_transform(..., recursive=False)` removed the target
matrix without compensating children, moving their world coordinates. Two new
straight-polyline regressions failed before repair (root bake and child bake under
a transformed parent). The fix always transfers the removed local matrix to direct
children; recursion controls whether their geometry is baked. Reported recursive
failure results now propagate. This is not rollback/transaction support.

Five new tests cover both nonrecursive cases, recursive root/child baking, and
recursive singular scaling. They assert world geometry, selected identity matrices,
untouched descendant local geometry when nonrecursive, UUID/object/relationship,
layer and group preservation, idempotence, and two XML round trips. All 18 tests
pass with `python -m unittest discover -s tests -v` on Python 3.10.9 / NumPy 1.23.5.
Syntax compilation, import/construction and `git diff --check` also pass.

A one-off ignored generator produced an independent explicit-coordinate A and
fully baked B under `output/full-bake-validation-n132g8ro/`, checking the loaded
coordinates/counts. Manual criteria live in the development runbook; user
acceptance is pending. Prior parent display acceptance does not establish baked
coordinate fidelity. General transform, component and curved-shape baking remain
unverified. In particular, `transform_primitive` currently postmultiplies despite
claiming global application; that is the next bounded slice. Pline bulges are
preserved under reflections/nonuniform scales and need a separate curved-geometry
contract. Reopen full-bake hierarchy repair if descendant pose changes under the
tested straight-polyline affine conditions.


### Full-bake A/B user acceptance

The user confirmed all full-bake A/B endpoints, identity matrices and final stored
coordinates in both files, and the intentional separate-versus-shared layers.
They subsequently confirmed understanding that A was constructed directly and B
was baked before export, and authorized the next increment after committing.
This completes the synthetic full-bake display acceptance; CamBam version was not
supplied. The original verification entry's pending acceptance is superseded by
this report. Broader curved/component baking and production machining remain
unverified. Future before/after cases should include the pre-operation file when
it helps the user inspect what changed.


### Global transform ordering verification

2026-09-07: the pre-fix root probe used local endpoints (1,0)/(2,0), an existing
translation (10,0), and a requested global 90-degree rotation. It produced
(10,1)/(10,2), failing the expected (0,11)/(0,12). `transform_primitive` previously
postmultiplied the effective matrix and baked the same world matrix into every
node's local geometry. Both violate the public global-coordinate contract.

Matrix mode now solves in the parent frame; baked mode precomputes a conjugated
operation in every affected original world frame, preserving effective matrices.
Invalid affine input and singular required frames fail before mutation. Even a
compatible singular-frame operation is rejected because the local reconstruction
is not unique. No pseudoinverse or reparenting fallback is used. Geometry-baker
failures are not transactional; entity-specific approximation/error handling is
unchanged. `combine_transformations` documentation now describes its existing
rightmost-first point application; its implementation is unchanged.

`python -m unittest discover -s tests -v`: all 26 tests pass (eight new ordering,
wrapper, singular/invalid-input and repeated round-trip regressions).
`python -m compileall -q cambam_builder legacy_cambam_builder tests`, import/construct
smoke check and `git diff --check` pass on Python 3.10.9 / NumPy 1.23.5.
The tests verify affected world geometry, unchanged frames/geometry where required,
relationship integrity and XML counts/UUIDs/layers across two round trips.

Prepared ignored Before/A/B files were parsed and reloaded numerically. A has
identity matrices; B retains the root world matrix and changes both descendant
world translations by (+5,-3). The development runbook owns precise manual
criteria; user acceptance remains pending. Existing full-bake acceptance is
recorded above and need not be repeated. Curved/component baking and alignment
remain outside this increment. Reopen if supported callers relied on the former
local-coordinate ordering, or a singular-frame policy needs expansion.


Final bounded review reproduced two additional failures, now covered by tests:
a baked 1e-9 translation was discarded by Pline's approximate identity check,
and a nested-list matrix containing `10**1000` raised `OverflowError` instead of
returning `False`. Pline geometry baking now skips only exact identity; matrix
conversion catches overflow. The final `python -m unittest discover -s tests -q`
run passes all 26 tests. No curved-bulge fidelity claim follows from this change.


### Global-transform acceptance and workstream checkpoint

The user confirmed A/B endpoints and placement, transform properties in B versus
explicit coordinates in A, and the correct (+5,-3) child/leaf movement from Before.
They reported committing the implementation. This supersedes pending manual
acceptance in the earlier verification record. Later user clarification establishes
CamBam Plus 1.0 as the validation environment.

The parent/hierarchy, full-bake and global-transform straight-polyline line has
reached its scoped stopping condition. Component baking is distinct (fold one
existing component into geometry while retaining the rest), but adjacency alone
is insufficient to prioritize it. The documented duplicate-name MOP import loss
(two operations become one; `_reconstruct_mop` uses the display name as identifier)
has greater immediate impact on preserving machining intent. It is now the next
recommended outcome, without expanding into registry migration. Remaining
transform gaps stay explicit in PROGRESS with reopening criteria. A fresh session
can start from repository documentation without retaining this conversation.

## MOP identity round-trip verification

2026-09-07: fixed the I/O identity boundary in `Mop._add_common_mop_elements`
and `_reconstruct_mop`. Previously display Name became the unique identifier,
merging/rejecting duplicate names and replacing UUIDs. JSON Tag now carries
`user_id`/`internal_id`; restoration precedes registration. Reconstruction follows
geometry to detect cross-entity collisions. Native test delegation was limited to
the synthetic test module; lead reviewed and extended it and verified locally.

Complete valid colliding identity metadata fails the entire import, preserving no
partial result. Missing/incomplete/malformed metadata gets a new UUID/identifier;
this keeps ordinary metadata-free CamBam files usable without unique names.
No registry migration, group semantics change or broad parameter parser expansion.
Reopen metadata policy for an authorized real interoperability fixture showing
CamBam strips/rejects Tags or requires different identity handling.

Verification on existing Python 3.10.9 / NumPy 1.23.5 (no project venv):

- `python -m unittest discover -s tests -p test_mop_roundtrip.py -q`: 5 tests pass.
  Four MOP types, same-part/across-part duplicate names, name collisions with
  layers/parts, UUID/identifier lookups, per-part order, target UUIDs, representative
  supported explicit parameters and two round trips. Also legacy fallback and
  subsequent identity stability, malformed Tags and conflicting MOP/geometry IDs.
- Loading `git show HEAD:cambam_builder/cambam_reader.py` into an isolated Python
  module and substituting its reader in the new suite gives 7 assertion failures
  across 5 tests, zero errors. Working files were not reverted. The fixed writer
  was retained, proving the old reader loses operations even with identity Tags.
- `python -m unittest discover -s tests -q`: all 31 tests pass.
- `python -m compileall -q cambam_builder legacy_cambam_builder`, import/construct
  smoke and `git diff --check`: pass.
- Generated and inspected `output/mop-validation-kpk5hxop/` A/B XML: two Profiles
  named Duplicate, separate left/right square references, ordered depths -1/-2
  and feeds 300/600 retained after two round trips. Reference omits new MOP Tags.
  Initial generator incorrectly assumed UUID-assigned primitive IDs followed
  creation order; corrected its assertion to resolve IDs through primitive Tags.

Implementation and automated verification are complete. CamBam acceptance is
pending under [the prepared criteria](DEVELOPMENT.md#manual-mop-identity-acceptance);
no machine execution is required. Existing parameter coverage and Default/Value
limitations remain; this is not production machining acceptance. The next useful
increment is export failure handling, whose silent omissions remain a data-loss
risk. Suggested commit: `fix: preserve MOP identities across XML round trips`.

## MOP A/B user acceptance

2026-09-07: the user confirmed that all described expected outcomes were validated
and true. This accepts both prepared files loading successfully, two enabled
Duplicate Profiles under Part1 in the expected order, left/right square targets,
depths -1/-2, feeds 300/600, Outside, diameter 3, depth increment 0.5 and spindle
12000. Later user clarification establishes CamBam Plus 1.0 as the validation
environment. This supersedes the pending acceptance
in the preceding verification record; no repeated check is required without a
relevant behavior change.

Scope is the synthetic load/display/property case, not machine execution or
complete parameter compatibility. Identity/registry preservation remains covered
by automated tests. No runtime changes were made for this acceptance update;
document consistency and `git diff --check` were checked. The local A/B artifacts
are retained. Export failure behavior remains the next implementation priority;
this accepted slice is a good fresh-session breakpoint.

## Export failure and state path verification

2026-09-08: completed the export error boundary and separate bare-filename
persistence slice. `build_xml_tree` re-raises encoder/resolution errors, retaining
their original types and logged entity context. Ordered missing layers/parts and
resolved targets without XML IDs now fail instead of being skipped. XML saves
serialize into a unique sibling temporary file, close it, then use `os.replace`.
Failure cleanup preserves the original exception. Formatting failures propagate,
except unavailable Python 3.8 indentation retains the existing unindented fallback.
`save_state` creates the absolute parent directory, so bare names work, and raises
directory errors. The lasting contract and limits live in specification section 0.

Evidence:

- The initial four `test_export_failures.py` tests against the old writer produced
  14 failing subtests: swallowed entity errors, partial destination writes and
  missing replacement behavior. The fixed writer passed those tests. Expanded
  coverage checks preparation/resolution failures, missing ordered entities,
  bare filenames and unavailable indentation (eight tests total).
- Loading only HEAD's `save_state` method into the current class in a separate
  Python process produced two failures across the three state tests: no bare-name
  file and no raised directory error. No working files were reverted.
- `python -m unittest discover -s tests -v`: 42 tests passed on the available
  Python 3.10.9 / NumPy 1.23.5; no repository virtual environment was present.
  Tests assert unchanged bytes or absent destinations after failures, no leftover
  temporary files under normal cleanup, successful XML counts/UUIDs/MOP targets
  and depth, plus restored pickle primitive project links. Existing geometry and
  repeated round-trip regressions remain green.
- `python -m compileall -q cambam_builder legacy_cambam_builder`, the runbook
  import/construct smoke command and `git diff --check` passed.
- Supplementary local logs are in
  `output/export-persistence-89fec11716684f3db4282876edf95cb8/`;
  reusable failure fixtures are in the two new test modules.

Direct destination writes were rejected because late serialization/I/O failure can
truncate a previous good document. Best-effort encoder continuation was rejected
because it falsely signals successful export. No custom exception hierarchy or
dependency was needed. Atomic pickle writing, crash durability, comprehensive
corrupt-registry validation, PID filtering/group semantics and full schema
validation remain outside this bounded contract. Reopen those areas for a
reproduced failure or an explicit workflow requirement.

Implementation and automated verification are complete. No manual CamBam check
adds evidence for this filesystem/exception change; successful XML structure is
unchanged and existing synthetic product acceptance remains scoped as recorded.
This does not establish broader production machining acceptance. Overall priority
now returns to remaining transform fidelity; bound it with a reproduced case
before choosing the repair. This is a good fresh-session breakpoint: required
evidence and limits are saved, with no pending acceptance or decision.
Suggested commit: `fix: propagate export failures and support bare state filenames`.

## Rect baking loss investigation

2026-09-08. Scope: reproduce and bound the next transform defect, without changing
runtime behavior. The candidate is confirmed in `Rect.bake_geometry`: after
transforming its four corners, it saves only their minima/maxima as a new local
axis-aligned Rect. The original outline is irrecoverable after this assignment.
The current representation cannot encode a general rotated rectangle or sheared
parallelogram with an identity matrix. Bounds-only comparisons hide the loss.

The warning is insufficient: `is_rectangular_after_transform` checks
perpendicularity, not axis alignment, so pure rotation is accepted. The additional
matrix object-identity condition suppresses warnings for separate explicit bake
matrices. Project bake methods may return `True` despite the changed outline;
the writer's error propagation cannot detect successful but lossy geometry edits.

### Reproduction and verification

Reusable fixture: `tests/test_rect_bake_defect.py`, a root Rect at `(0,0)` with
width 4 and height 2. Commands run from the repository root:

- `python -m unittest tests.test_rect_bake_defect -v`: eight characterization
  tests passed. Known-defect tests assert the current wrong result explicitly;
  they are not claims of repaired fidelity.
- `python -m unittest discover -s tests -v`: all 50 tests passed after adding
  translation/scale controls and XML count/UUID/type checks. Python 3.10.9,
  NumPy 1.23.5; no repository virtual environment was present.
- `python -m compileall -q cambam_builder legacy_cambam_builder` and the runbook
  import/construct smoke command passed. `git diff --check` passed.

| Operation on the fixture | Required outline area | Actual baked area | Observed boundary |
| --- | --- | --- | --- |
| Full bake of 45-degree rotation about origin | 8 | 18 (+125%) | Success, identity matrix, no entity warning |
| Full bake of unit X shear (`x += y`) | 8 | 12 (+50%) | Success with loss warning |
| Global bake of the same shear from identity | 8 | 12 | Success without entity warning |
| Explicit-matrix 45-degree rotation bake | 8 | 18 | Success, original identity matrix retained |
| Rotation-component bake of pure 45-degree rotation | 8 | 18 | Success, identity matrix |

The shear matrix is `[[1,1,0],[0,1,0],[0,0,1]]`; the existing helper spells this
`skew_matrix(angle_y_deg=45)`. Correct shear corners are `(0,0), (4,0), (6,2),
(2,2)`; baking produces `(0,0), (6,0), (6,2), (0,2)`. At 45 degrees the rotated
outline fits in a square of side `3*sqrt(2)`, but is not that square.

Translation, diagonal scaling, reflection and 90-degree rotation preserve the
tested corner sets and areas. Two XML round trips preserve the already-damaged
rotated Rect, its UUID, count and identity matrix. Unbaked root rotation and shear
instead preserve their expected corners/area and UUID/count through two trips:
rotation stays a Rect with a matrix, shear is serialized as a closed Pline.
These root controls do not establish conversion fidelity under a parent.

Investigation and automated checks are complete; runtime repair remains backlog.
No manual acceptance is needed for this numerical reproduction. Hierarchies,
nonidentity frames in explicit/global bake mode, mixed component decomposition,
degenerate shapes and general curved geometry were not validated by this slice.
The fixture deliberately isolates representation loss from those other contracts.
This is a good fresh-session breakpoint: evidence, limits and next priority are
persisted. Suggested commit: `test: reproduce and bound Rect baking geometry loss`.

### Bounded repair recommendation

Prioritize exact Rect baking over MOP registry migration: this is reproduced
geometry corruption through existing public operations, whereas registry migration
still needs group-source semantics. Start with full local-transform baking of a
Rect and its hierarchy, using a closed straight polyline when the transformed
outline cannot be represented by an axis-aligned Rect. Preserve the outline for
axis-preserving cases without unnecessary conversion. Extend the same geometry
policy to explicit/global bake entry points; do not couple the repair to general
component decomposition, curved entities or alignment ordering.

Acceptance for that repair:

- Compare all four transformed vertices/closed edges and area, allowing cyclic
  reordering or reversed winding, at absolute tolerance `1e-10` in memory and
  `1e-8` after two XML round trips with adequate output precision. Equal bounds
  alone do not pass. Include rotation, shear and axis-preserving controls.
- Preserve UUID, identifier, description, layer/group membership, parent/child
  links and MOP target resolution through any representation change. Specify
  behavior of existing Python object references before implementing conversion;
  `to_pline_representation` currently creates a new identity and is not a safe
  registry replacement operation by itself.
- Preserve descendant world geometry for recursive and nonrecursive full baking,
  reset only the matrices required by that API, and retain effective matrices for
  explicit/global geometry baking. Test a transformed parent and unaffected sibling.
- Never report success after approximating the outline. If a supported conversion
  cannot be completed, surface failure before destructive geometry/registry edits;
  do not substitute a warning for the geometry contract.
- Replace known-loss characterization assertions with preservation regressions
  during repair. Prepare and inspect synthetic CamBam A/B files for any changed
  representation, then record user display acceptance separately.

An export-only workaround cannot recover discarded corners. A blanket rejection
of all rotation would unnecessarily reject quarter turns. General transform
refactoring is deferred: reopen component ordering, alignment, curved geometry
or degenerate/nonfinite edge cases only for a failing workflow fixture or an
explicit dependency. No manual CamBam check is needed to establish this numerical
defect; no runtime behavior or previously accepted display fixture changed.


## Rect baking repair verification

2026-09-08. The repair replaces bounding-box baking with a closed-outline
comparison and in-place conversion to a four-vertex, zero-bulge Pline when needed.
Axis-aligned results remain Rects. The owning contract is
[Rect baking representation](structure_spec.md#rect-baking-representation-contract).
The same Python object survives conversion, including UUID, project link and
metadata, so registries, MOP source resolution and external references continue
to designate it. Callers must accept the runtime type change; Rect-specific
corner/width/height attributes are removed. Creating a separate replacement
would leave external references stale and require registry coordination, while
retaining an AABB or rejecting all rotation would violate the repair criteria.

The explicit project helper now calls the entity's explicit-matrix API directly,
so validation failure does not leave a temporary effective matrix installed.
Full-bake identity detection uses exact equality to avoid dropping small but
observable transforms. Rect input/corner validation occurs before mutation and
raises on failure. This does not establish transactional subtree baking or
repair general component ordering, curved geometry or alignment.

Verification on system Python 3.10.9 / NumPy 1.23.5 (no project venv):

- `python -m unittest tests.test_rect_bake_defect -q`: nine tests pass, checking
  closed edges/area, axis-preserving controls, recursive/nonrecursive full bake
  under a transformed ancestor, explicit/global matrices and descendants,
  repeated baking, pickle and two XML round trips, identity/metadata/MOP targets,
  small shear and invalid explicit input without mutation.
- The worker substituted the original HEAD Rect method in memory: eight
  regression failures confirmed detection of the former loss (before the two
  additional small-shear/error tests were added).
- `python -m unittest discover -s tests -q`: all 51 tests pass. Expected error
  logs come from negative-path tests; no test failures.
- `python -m compileall -q cambam_builder legacy_cambam_builder`, the runbook
  import/construct smoke command and `git diff --check` pass.
- `python output/rect-bake-validation-20260908-a/generate.py`: generated and
  inspected reference/result XML, three closed four-point zero-bulge Plines,
  identity matrices, metadata/parent linkage and vertices within `1e-8` after
  two round trips. [Manual criteria and files](DEVELOPMENT.md#manual-rect-bake-acceptance)
  use display tolerance `0.01`. These ignored artifacts remain available.
- Native review found no additional runtime defect; its in-progress test shape
  finding was resolved by normalizing Pline triples to XY in the final tests.

Implementation and automated verification are complete. Manual validation adds
CamBam renderer/load evidence beyond library XML reconstruction; report version,
A/B pass/fail and any differing outline or matrix against the prepared criteria.
CamBam user display acceptance is now complete (see below); previous accepted fixtures are
unchanged and need no repetition. Reopen this scope for a failing supported
outline/relationship fixture, object-layout compatibility need, or reported
CamBam display discrepancy.

Suggested commit: `fix: preserve Rect outlines when baking rotation and shear`.
The next priority is defining MOP group-source
compatibility before registry migration, as recorded in [PROGRESS.md](PROGRESS.md). This is a
coherent accepted breakpoint; use a fresh session for that distinct design scope. Nothing was staged or committed.


## Rect baking display acceptance

2026-09-08. The user confirmed all validation conditions were met, a full match
and identity transforms for the prepared Rect A/B case. This accepts loading and
displaying the three closed outlines with the documented vertices within `0.01`
drawing units and identity matrices in A/B. Later user clarification establishes
CamBam Plus 1.0 as the validation environment.
[Files and retained criteria](DEVELOPMENT.md#manual-rect-bake-acceptance) remain
available; no regeneration or repeat acceptance is required for unchanged code.

Implementation, automated verification and synthetic display acceptance are
complete. This acceptance does not extend to production toolpaths, other curved
entities or general component ordering. No runtime code changed in this round;
`git diff --check` passed and the documentation links/status were reviewed.
No automated tests were rerun because only acceptance documentation changed.
Suggested commit: `docs: record Rect baking display acceptance`.

## Rest machining future-feature planning

2026-09-08. Recorded the user's five requested outcomes as a low-priority optional
feature in [PROGRESS.md](PROGRESS.md#remaining-backlog-in-order), with one
[plan owner](REST_MACHINING_PLAN.md) for problem, methods, dependencies and future
acceptance. Existing priorities and runtime behavior are unchanged.

Reasoning separates ideal tool reachability from trajectory-derived remaining
stock, pure rest from access expansion, and surface V-corner sharpening from
complete volumetric removal. Corner overcut is modeled from its sweep rather
than presumed to eliminate all rest. Native CamBam V-corner support and the
XYZ-polyline/Engrave proxy are attributed to official documentation in the plan.
Region and Z are feature prerequisites, not newly promoted foundation work.

The user-authorized Region example was inspected locally only and left unchanged.
Its typed XML outer/hole curves and the current reader's Region/Z gaps are
recorded in the plan. No private fixture content was sent to external providers.
The proposal includes numerical references, workflow helpers, provenance and
invalidation, five separate acceptance outcomes and promotion/stopping criteria.
Native technical review prompted explicit gates on lossless XYZ proxy export
and arc-extrema-aware contour bounds before using bulged Regions in analysis.
The user clarified that standalone programmatic calculation is essential: CamBam
cannot supply a headless engine/API to this workflow. Removed the proposed
toolpath-acquisition/scripting dependency in favor of local supported MOP and
geometry algorithms. Optional CamBam exports are validation fixtures only. The
retained XYZ/Engrave proxy is a file encoding for paths calculated by the framework.

Documentation verification: local links/anchors and analytic example values
checked; `git diff --check` passed. No runtime tests or manual machining validation
are needed for this documentation-only increment. Implementation, CamBam path
acceptance and production machining remain future work. Planning is a coherent
fresh-session breakpoint; the next task remains MOP group-source compatibility.
Suggested commit: `docs: plan optional rest machining and V-cutter workflows`.


## Shape parity separated from rest machining

The user clarified that Region and all-shape Z support are missing core CamBam
feature parity, independently useful and upstream of rest machining. Moved their
repository evidence and implementation contract to [SHAPE_PARITY_PLAN.md](SHAPE_PARITY_PLAN.md).
The [rest plan](REST_MACHINING_PLAN.md#upstream-dependencies) now consumes those
contracts instead of owning their implementation. This supersedes the earlier
planning decision that treated them as support inside the optional rest workstream.

Backlog placement is after MOP design/ownership and geometry/relationship
correctness, before packaging, MCP and rest machining. Existing patterns remain
the implementation approach; tuple/bulge compatibility, shape elevation and matrix
Z semantics require focused evidence. No runtime behavior changed. Local links,
plan ownership and `git diff --check` were checked; runtime tests and manual
validation are not needed for this documentation change. The next immediate task
remains MOP group-source compatibility; the distinct parity scope is ready for a
fresh session when its priority is reached.
Suggested commit: `docs: separate and prioritize Region and Z shape parity`.


## MOP group-source compatibility

**Policy superseded by [user clarification](#development-compatibility-priority);
observed behavior and test results below remain historical evidence.**

2026-09-08: compatibility definition and characterization before backlog item 1's
registry migration. Runtime behavior and XML schema are unchanged.

Evidence: `CamBamProject._add_mop_internal` retains string sources and resolves
list entries once to primitive UUIDs; `resolve_pid_source_to_uuids` performs live
group lookup, filters missing primitives and returns sorted unique UUIDs.
`Mop._add_common_mop_elements` writes concrete XML references and identity-only
Tags. `_reconstruct_mop` and the reader's deferred linking pass reconstruct UUID
lists, regardless of primitive group metadata.

Decision: preserve live groups in memory and snapshots after XML import. The
then-current contract constrained migration to retain source intent and the public
`pid_source` surface; this is superseded by the development compatibility priority
and [current target contract](structure_spec.md#mop-target-ownership-contract).
Freezing all sources at MOP creation would change existing live behavior. Inferring
groups from matching targets is ambiguous (multiple groups may match), and adding
live-group Tag metadata would change how reimported jobs respond to membership
edits. Neither is necessary for ownership migration. Reopen persistent live XML
sources only for an explicit workflow with a defined opt-in/schema and external
edit conflict policy; do not silently retarget existing files.

Scope excludes registry implementation, old-pickle compatibility changes,
Default/Value parameter fidelity and broader malformed metadata handling. Existing
MOP identity tests already cover malformed JSON/nonobject/invalid UUID Tags and
identity collisions; they do not establish exhaustive metadata/default coverage.

Verification (available system Python 3.10.9 / NumPy 1.23.5; no project venv):

- `python -m unittest discover -s tests -p test_mop_group_sources.py -v`: six pass.
- `python -m unittest discover -s tests -v`: all 57 pass.
- `python -m compileall -q cambam_builder legacy_cambam_builder`: pass.
- `python -c "from cambam_builder import CBProject; p = CBProject('smoke'); assert p.project_name == 'smoke'; print('import/construct OK')"`: pass.
- `git diff --check`: pass; changed documentation links/headings reviewed.

The new tests cover membership/deletion/recreation, group-versus-list selection,
normalization and caller-list isolation, empty/missing sources, source mode
switching, UUID non-rebinding, and all four MOP types through two XML round trips.
Synthetic XML assertions inspect concrete refs; reimports check target snapshots,
part order, group metadata and sample geometry before testing membership edits.
These are characterization tests against unchanged runtime, not a runtime defect
repair, so no before-fix failure is claimed. Manual validation is
not required for this slice because no emitted schema or runtime behavior changes.
Existing synthetic CamBam acceptance is not expanded to production machining.


## Development compatibility priority

2026-09-08 user clarification: no important files depend on older framework
versions; the framework has not been used in production. Correct architecture
and the core model take priority over backward compatibility with those versions.
The required compatibility boundary is CamBam `.cb` files, including framework
output and files created or modified directly in CamBam.

This supersedes the prior obligation to retain `pid_source`, its source modes
and old-pickle handling during registry migration. Breaking APIs/storage are
allowed; update repository callers/tests together and avoid legacy adapters or
duplicate ownership. The six characterization tests remain evidence of current
behavior, not a veto on redesign. Native CamBam edits must not be silently undone
by stale framework metadata. Domain semantics and external-format fidelity still
need design and verification; this clarification does not claim new format support.

Updated the specification policy, backlog and test runbook. Documentation-only
change: references reviewed and `git diff --check` passed; no runtime tests or
manual CamBam validation needed. Next: implement the MOP core-model increment
against this policy. Suggested commit: `docs: prioritize core model and CamBam interchange`.


## MOP core ownership and interchange redesign

2026-09-08 (continued after a usage-limit interruption). Target ownership moved
from MOP instances to one project registry. Explicit UUID sets support stable
selection; a separate live group-name mode supports incremental construction.
Bare-string ambiguity and silently skipped invalid API targets were removed.
Setters validate before mutation, deletion cleans references, and XML snapshots
remain authoritative on import. No old API adapter or pickle migration was added.

Rejected restoring live-group mode from framework Tags: CamBam can edit native
primitive IDs independently, so group restoration could silently undo a native
edit. The native list is now the import authority even when metadata survives.
CamBam's [Machining Basics](https://www.cambam.info/doc/plus/cam/Basics.htm)
documents target editing, operation reordering and Default inheritance.
Its [CAM Styles guide](https://www.cambam.info/doc/fr/cam/CAMStyles.htm)
describes Style/StyleLibrary resolution through the hierarchy and external style
libraries. This supports retaining native state/context, not evaluating CAM styles
inside this library.

The interrupted parameter work had parsing without state-aware encoding. A focused
native-context regression reproduced dropped global Style/StyleLibrary and
ClearancePlane values before capture/serialization was integrated. The replacement
must retain native parameter content and explicit Default/Value state while allowing
Python edits; synthetic tests do not establish native application acceptance.

Implementation and automated verification are complete for this slice. The user
accepted the prepared A/B CamBam display/property case on 2026-09-08; CamBam
version was not supplied. The final
checks were:

- `python -m unittest discover -s tests -v`: 65 tests passed; the retained log is
  `output/mop-core-checks-ncsbkpgz/suite.log`.
- `python -m compileall -q cambam_builder legacy_cambam_builder tests demos`:
  exit 0.
- `python output/mop-core-validation-1roowlbtpza/generate.py`: exit 0; counts,
  Profile/Pocket/Engrave/Drill targets, depths and feeds verified. Lead XML
  inspection confirmed rectangle coordinates, triangle vertices and drill points.
- The tightened native comparison helper passed synthetic C-to-D checks for
  operation names/order, targets and exact parameter/state sets. The clearly
  named Synthetic_C/D files remain in the checks directory; these are not
  native CamBam acceptance.
- Import/construct smoke, `git diff --check`, and local Markdown path/anchor
  validation passed. Independent read-only reviews found no material target or
  parameter issue; the lead integrated subsequent assignment/state fixes.
- `test_mop_context` reproduced the lost global `Style`/`StyleLibrary` context
  before reader capture and now passes after the fix.

The native edit is retained as
[`C_native_edited.cb`](../output/mop-core-validation-1roowlbtpza/C_native_edited.cb).
The C-to-D import/export comparison exited 0 and preserved five enabled
operations in the order Drill, Engrave, Pocket, Profile, Profile. It preserved
the operation names/order/targets and exact parameter/state sets: the fourth
Profile targets `pocket-square`, has CutFeedrate 450 and ClearancePlane Default,
and the last Profile targets `profile-square`. Independent XML comparison also
matched all four shapes' coordinates, closure flags and identity transforms; the
native check logs report no warnings or errors. See
[`C_framework_roundtrip.cb`](../output/mop-core-validation-1roowlbtpza/C_framework_roundtrip.cb)
and the adjacent inspection JSON, `native-check.log` and
`native-geometry-check.log`.

Final native acceptance was reported on 2026-09-08 using CamBam Plus 1.0:
`CamBam.CAD` 1.0.7364.41819, `CamBam` 1.0.7364.41821, build 2020-02-29
23:13:58. C and D loaded without error and every stated display/property check
matched. This closes the supported MOP interchange slice. No full `.cb` fidelity,
production toolpath or broader production acceptance claim is made. Default
fields are cached but CAM styles are not evaluated; nested state setters are
intentionally unavailable while imported nested state is preserved. Unknown MOP
types remain skipped. Existing Python 3.10.9 and NumPy 1.23.5 were used; no
virtual environment or dependency changes were made.

The user clarified that CamBam Plus 1.0 itself saves `.cb` files with XML
`Version="0.9.8.0"`. That marker is therefore a legacy file-format value and is
not evidence that CamBam 0.9.8 created the file. Use the CamBam Plus 1.0 baseline
above for the user's past and future validation reports unless they explicitly
state a different application version. Keep the framework marker at `0.9.8.0`:
using `1.0` would depart from the native producer without evidence that the field
is an application label or that the alternate value improves compatibility.
Reopen this decision for a documented format revision or controlled native test.

The next backlog work is split into curved-geometry bounds (1a) followed by
copy/transfer utilities (1b). Starting with 1a is a coherent fresh-session
breakpoint with implementation, automated evidence, and accepted A/B and native
C/D results persisted. No staging or commit was performed. Suggested commit:
`refactor: centralize MOP targets and preserve native parameters`.

## Model routing and curved-workstream split

2026-09-08: the next backlog work was split after assessing its reasoning and
coordination risk. The combined bounds plus copy/transfer item scores about 11/13
because transfer semantics are registry-coupled and its UUID/collision contract
is not yet settled. Treating it as one undivided implementation would justify
Astra. Splitting it makes curved bounds (1a) a bounded Sol task and leaves
copy/transfer (1b) as a contract-first follow-up.

Decision: start the next session with GPT-5.6 Sol at high or xhigh reasoning. Use
native Luna `project_explorer`, `focused_implementer` and `deep_reviewer` roles
for bounded retrieval, implementation/tests and review. Escalate the main thread
to GPT-6 Astra if 1b must be designed and implemented as one change or if the
UUID/collision contract remains unresolved. The reusable factors, thresholds,
role mapping and privacy constraints are in [MODEL_ROUTING.md](MODEL_ROUTING.md).

Acceptance for the documentation decision: the topic map links the routing
guide; `PROGRESS.md` owns the 1a/1b ordering and next-session handoff; and
`WORKFLOW.md` points to the reusable model-selection rules. No runtime behavior
changed, so runtime tests and CamBam validation are not required for this slice.

## Curved geometry bounds verification

2026-09-08: endpoint-only Pline bounds and full-circle Arc bounds reproduced the
documented curved-geometry gap. The bounded repair adds analytic directed-sweep
extrema for Arcs and for every active Pline bulge segment. It evaluates extrema
after the complete local-plus-ancestor affine map, so rigid transforms,
reflections, nonuniform scales, shears and singular affine projections share one
exact parametric contract. Open Plines ignore the final vertex's bulge; closed
Plines use it for the closing segment. The implemented tolerance and invalid-box
boundaries are owned by the
[specification](structure_spec.md#curved-geometry-bounds-contract).

The implementation avoids sampling and the former average-radius approximation.
It also rejects complex, perspective and nonfinite transforms consistently with
the project transform boundary. Deep review found no ordinary-range underbound
in the sweep or bulge derivation. Its findings led to strict homogeneous-row
validation, overflow-safe full-sweep norms and half-chord bulge construction,
and exact partial-affine and singular-transform assertions. A second review found
no remaining blocker; 1,000 randomly sampled transformed bulge arcs produced no
underbounds.

Verification on Python 3.10.9:

- Pre-change baseline: `python -m unittest discover -s tests -v` passed 65 tests.
- `python -m unittest discover -s tests -p test_curved_bounds.py -v` passed all
  seven focused tests.
- `python -m unittest discover -s tests -v` passed all 72 tests.
- `python -m compileall -q cambam_builder legacy_cambam_builder tests demos`, the
  import/construct smoke check and `git diff --check` passed.

No XML fields or stored geometry changed, so a CamBam display round trip would
not test this runtime calculation and is not required. Curved geometry baking,
the approximate curved shape returned by `get_absolute_coordinates()`, Region
support and copy/transfer APIs remain outside this result. Item 1b is ready as a
fresh contract-first increment; reopen 1a for a demonstrated numerical underbound
or a new non-affine geometry policy. Suggested commit:
`fix: calculate exact curved geometry bounds`.

## Copy and transfer contract decision

2026-09-08: backlog 1b began with no modern copy/transfer API. Section 5's
conceptual UUID-overwrite proposal did not define ownership of existing target
children, group selections or operation order. The project registration helper
also permits UUID overwrite while separately rejecting identifier collisions;
using it incrementally would not establish atomic transfer.

Chosen contract: complete descendant subtree, detached root preserving world
pose, cloned containers, UUID preservation or explicit full remapping, and
rejection of destination collisions. Associated MOPs must have targets wholly
inside the subtree; callers may explicitly omit MOPs. Group names cannot merge
implicitly because that would expand existing live MOP targets. Stage container
and relationship changes before publication, preserving unrelated object
identities. The [current contract](structure_spec.md#copy-and-transfer-contract)
owns API details and limitations.

Overwrite/merge and automatic expansion to unrelated MOP targets were rejected
for this increment: both need additional destination-ownership decisions and
could change unrelated geometry or machining selections. Reopen for a concrete
synchronization or assembly workflow with explicit merge acceptance criteria.
Destination global machining/style context stays authoritative; parameter
preservation does not establish identical inherited settings or toolpaths.

Lead owns the shared contract and integration; native bounded workers handle
retrieval, implementation and independent tests. External authorization and cost
telemetry were unavailable. No budget saving is asserted from measured usage.
Automated evidence is recorded below. Manual
CamBam checking adds no distinct evidence for registry cloning once numerical
and serialized relationship regressions pass. Existing display acceptance is
not extended to production machining by this work.

Integration exposed a pre-existing Rect XML defect: the conversion branch used
`to_pline_representation()`, which bakes only the local matrix into an unlinked
Pline. A transformed parent was therefore lost on export. The new two-round-trip
test first failed its matrix assertion, then its world-outline assertion;
checking geometry confirmed this was a real defect, not merely an alternate
representation. The bounded repair applies the complete world matrix to Rect
corners and emits a temporary Pline with an identity XML matrix, retaining the
existing world-baked representation. Keeping corners local with a world matrix
was also geometrically valid but broke the existing root-Rect identity-matrix
control; retaining that representation avoids an unnecessary compatibility change.
It does not change the public conversion helper or bake behavior. Required
evidence includes the complete outline, parent links and descendant world poses
through both round trips. The existing Pline XML representation needs no new
schema/display acceptance; no production machining acceptance is inferred.

### Copy and transfer verification

2026-09-08: the implemented `copy_primitive_tree` and
`transfer_primitive_tree` APIs pass the focused copy/transfer suite (16 tests)
and the full suite (88 tests) on global Python 3.10.9 with NumPy 1.23.5; no
project virtual environment is present. Coverage includes atomic collision and
late-clone failures, subtree relationships, UUID/name/group remapping, source
preservation, transfer cleanup, MOP closure, world pose, and two XML round trips
covering outlines, root/leaf poses, primitive/MOP UUIDs, parent/layer assignments,
MOP parameters and targets.
The retained logs are [focused.log](../output/copy-transfer-checks-final-7c34ab/focused.log)
and [full.log](../output/copy-transfer-checks-final-7c34ab/full.log).

The bounded `Rect.to_xml_element` repair applies the complete ancestor world
transform when emitting the world-baked Pline representation, preserving its
identity XML matrix and outline through the transfer round trips. Compileall and
the import/construct smoke check also pass; [compile.log](../output/copy-transfer-checks-final-7c34ab/compile.log)
and [import.log](../output/copy-transfer-checks-final-7c34ab/import.log) are retained.
`git diff --check` exits 0 with only line-ending normalization warnings. No manual
CamBam validation adds evidence for these registry and known XML-encoding checks;
production toolpaths remain outside scope. Packaging is unverified.

This is a good fresh-session breakpoint with implementation, evidence and limits
persisted. The next priority is Region/all-shape Z parity under
[`SHAPE_PARITY_PLAN.md`](SHAPE_PARITY_PLAN.md). Suggested commit:
`feat: add atomic primitive tree copy and transfer`.

## Region and shape-elevation implementation

2026-09-09. Backlog item 2 adds geometry elevation and Region interchange to the modern
package. The user reaffirmed that the API is unreleased and may change. The
[specification](structure_spec.md#shape-elevation-and-region-contract) owns the
implemented API, topology, matrix and query/bake boundaries; the
[runbook](DEVELOPMENT.md#manual-shape-parity-acceptance) owns prepared acceptance.

The chosen representation keeps intrinsic XY/bulge data, with separate explicit
per-vertex Z or analytic elevation and independent local Z offsets. This avoids
reinterpreting the third Pline tuple element or replacing all affine/relationship
machinery with a general 3D framework. XYZ queries are explicit; reflected arc
sweeps and bulges are corrected, while representations that cannot express an
ellipse reject a nonsimilarity query. XML can retain supported XY affine matrices.
Spatial matrices, varying-Z bulged segments and lossy analytic/Text bakes were
rejected pending native evidence instead of flattened or approximated. The later
varying-Z interchange evidence is recorded below. Existing 2D constructor/query
usage remains valid except that invalid affine constructor matrices now raise.

Region is one registered primitive with owned contours and a single MOP identity.
It supports the observed typed `entity xsi:type="Region"` schema, `OuterCurve/pts`
and `HoleCurves/Polyline`, including contour poses and elevation. A copied
registered contour captures its complete world pose before detachment. The
validator handles lines/arcs analytically, either winding, two complementary
semicircles, holes and nondegenerate planar topology. Direct shifts/bakes and
project full/global subtree bakes stage changes before publication. Review added
coverage for invalid topology, coincident arcs, near-identity bakes, reflected
queries, finite Z, contour ownership and export revalidation.

### Native Region fixture finding

Local import of the private `output/region_example.cb` exercises the expected
schema but fails strict topology validation: the first hole self-intersects
between zero-based contour segments 2/4 and 3/4. A separate local script sampled
each circular segment into 2,048 chords and independently found an interior
crossing in each pair. This diagnostic does not call the Region intersection
validator. Hash comparison before/after the check confirms no source change.
No GLM/OpenRouter processing was used; delegated edit ownership was restricted
to source code and authored synthetic tests.

The declared model rejects self-intersection rather than silently repairing the
hole, filling it or importing an empty/partial project. Therefore the native
sample itself is **not accepted**. Supported typed XML is covered with authored
valid synthetic cases. Reopen this boundary only for an explicit requirement to
retain invalid contours for interchange, or evidence that CamBam interprets the
same supported bulge data differently. Such a requirement needs a separate
invalid-topology representation, not bypassing validation in the current Region.

### Verification and acceptance state

Environment: existing Python 3.10.9 / NumPy 1.23.5; no project `.venv`, dependency
installation, legacy-package changes or CamBam runtime dependency. Checks run from
the repository root:

- `python -m unittest discover -s tests -v`: 135 tests passed after final review. The new tests cover all seven types, two XML round trips, actual
  XML geometry/matrix Z, parent/world poses, bulges, bounds, holes, identities,
  groups/MOP targeting, full/nonrecursive/Z bakes, pickle and detached copy/transfer.
- `python -m compileall -q cambam_builder legacy_cambam_builder tests demos`:
  exit 0; `CBProject('smoke')` import/construction assertion passed.
- `python output/shape-parity-kvr0bdc2/generate.py`: passed; A/B retain eight
  primitives across all seven types, XYZ, identities, two Region holes and one
  Region pocket target through repeated round trips.
- `python output/shape-parity-kvr0bdc2/verify_acceptance_xml.py`: passed; an
  independent hardcoded expected-value inspection checks every runbook coordinate,
  dimension, bulge, hole, identity matrix and exact Region MOP reference in B.
- `python output/shape-parity-kvr0bdc2/verify_native_roundtrip.py`: passed; native
  C and framework D retain all eight identities, XYZ geometry, hierarchy, two
  Region holes, the Region-only pocket target and Text content/optional `p2`.
- `git diff --check`: passed after final review.

The final independent topology review found three concrete issues, all repaired
and reverified with the original synthetic reproducers and 21 focused Region
tests. Arc ray containment now uses exact stored endpoints, preventing a wholly
outside hole at an arc endpoint height from passing validation. XML export
validates the rounded geometry, rejecting a hole gap erased by output precision;
increasing precision to preserve the gap permits the round trip. Matrix
conditioning uses normalized linear entries, accepting a well-conditioned
`1e-8` scale that yields a useful 10-by-10 contour. These checks complete the
review scope; unrestricted numerical geometry and arbitrary 3D remain excluded.

The first integration run exposed test assumptions about Rect XYZ triples versus
converted Pline XYZ/bulge tuples, and XML precision amplification in computed arc
angles. Tests now compare geometry projections across representation changes and
explicitly choose 12-decimal XML precision for repeated transformed fixtures;
they retain `1e-10` in-memory / `1e-8` XML tolerances. XML tests compare numeric
values, allowing equivalent lexical forms such as `1` and `1.0`. The acceptance
A/B fixture uses 10 decimal places and exact translation matrices.

The user accepted every specified A/B display and property check in CamBam Plus
1.0, including matching geometry, all declared coordinates, Text at its `p1`
anchor, Region holes and the Region-only MOP target. CamBam saved B as native C
while retaining the framework's optional `p2="35,35,7"`. A Text created in a
fresh CamBam session uses `align="bottom,left"` and only `p1`; alignment therefore
does not require `p2`. This agrees with the
[official CamBam MText API](https://www.cambam.info/doc/api/MText.htm), which
documents `P1` as the current alignment point and `P2` as currently unused.

Further native inspection showed that CamBam suppresses both points for Text at
the default origin and may materialize equal `p1`/`p2` after it is moved. The
reader now defaults absent `p1` to `(0,0,0)`, and the writer suppresses the same
default. Together with the earlier non-origin `p1`-only file, this establishes
that optional `p2` reflects native serialization history rather than alignment.

Native C placed Text content after its `mat` child. That exposed a reader defect
that only read leading element text; the reader now retains one non-whitespace
direct mixed-content chunk and rejects ambiguous multiple chunks. Focused native
child-order and absent-`p2` regressions pass. The C-to-D check verifies the repair
without modifying C. This acceptance establishes the declared display/property
interchange case, not generated production toolpaths. Item 2 is complete and the
next priority is packaging and supported Python-version validation. This is a
good fresh-session breakpoint because the next scope does not depend on unresolved
item-2 decisions.

Suggested commit: `feat: add Region and all-shape elevation parity`.

## Canonical vertex-record refactor

2026-09-09. The bounded follow-up replaces Pline and Points parallel coordinate,
bulge and elevation storage with one public `Vertex(x, y, z=0, *, bulge=0)`
record and a sole `vertices` collection. Region-owned Pline contours use the same
records. Constructors and project adders normalize only named records, `(x, y)`
and `(x, y, z)` tuples. Thus three-tuples now unambiguously mean XYZ, four-tuples
are rejected, nonzero bulge requires a named record, and Points rejects nonzero
bulge at API and XML boundaries. The API is unreleased and the user authorized
these breaking changes; no compatibility property or old-pickle migration was
added.

Geometry consumers, bounds, reflected/similarity bakes, intrinsic Z shifts,
Rect-to-Pline conversion, Region topology/ownership/bakes, XML read/write and
current persistence were migrated. Existing XY and XYZ geometry-query return
shapes remain unchanged. Mutable collection validation requires records after
construction, so insertion and reordering carry X/Y/Z/bulge together instead of
depending on an index-aligned elevation list. Repository callers and authored
tests preserve every former bulge explicitly with `Vertex(..., bulge=value)`.
Analytic elevation fields, Text `xml_p2`, MOP ownership and Region topology rules
are unchanged.

Verification on Python 3.10.9 / NumPy 1.23.5:

- `python -m unittest discover -s tests -p test_vertex_records.py -v`: six tests
  passed, covering constructor defaults, keyword-only bulge, tuple meanings,
  malformed/four-value rejection, Points rejection, collection edits, adders,
  current pickle and two XML round trips.
- `python -m unittest discover -s tests -q`: all 141 tests passed. Existing
  Region holes/topology, curved bounds, transforms/bakes, identities,
  relationships, MOP references, copy/transfer and persistence remain covered.
- `python -c "import runpy; runpy.run_path('output/vertex-record-parity-20260909-a/compare.py', run_name='__main__')"`:
  passed. It read the accepted pre-refactor `B_baked_elevations.cb`, wrote two
  results only under the new ignored task directory, and compared all serialized
  Pline, Points and Region point/bulge/matrix numbers at absolute tolerance
  `1e-8`. Maximum observed difference was `0.0`. SHA-256 checks before/after
  confirmed accepted A/B/C inputs were unchanged.

No CamBam acceptance was repeated because serialized geometry and query/output
semantics are unchanged. At completion, remaining limits included the then-declared
rejection of varying-Z bulged segments, general spatial matrices and invalid Region
topology, plus the explicit absence of old-pickle compatibility. The varying-Z
restriction was subsequently superseded by native evidence and the increment
below. This is a good
fresh-session breakpoint once final compile/import/diff checks pass; packaging
and supported-Python validation remain the separate next priority.

Suggested commit: `refactor: consolidate pline and points vertices`.

## Varying-Z bulged Pline and Region interchange

2026-09-10. The user supplied CamBam-generated XML demonstrating both a Pline and
a Region whose bulged segments have unequal endpoint Z values. The Region includes
mixed-Z outer and hole contours. The user additionally verified that CamBam renders
the sloping curves, while Pocket and Profile operations use their machining depth
parameters and ignore the Region contour elevations. This is direct evidence for
native storage/interchange and projected Region topology, but not for an exact
intermediate Z parameterization or this framework's future MOP calculations.

The owning Pline validation now accepts every finite Z/bulge combination. Region
validation no longer requires coplanar contours and continues to perform simplicity,
intersection, containment, nesting, degeneracy and bounds work entirely in XY.
Reader/writer, copy, Z shift, endpoint queries and similarity/reflection bakes
already operated on canonical Vertex records, so no schema or output-contract
change was needed. Non-similarity transforms of bulged geometry remain rejected
because the XY circular projection becomes elliptical. Points still reject bulge,
and tilted matrices and intermediate spatial-curve evaluation remain unsupported.

Verification on Python 3.10.9 / NumPy 1.23.5:

- `python -m unittest discover -s tests -p test_shape_elevation.py -v`: 11 passed.
- `python -m unittest discover -s tests -p test_vertex_records.py -v`: 6 passed;
  its two project XML cycles now include unequal-Z endpoints on a bulged segment.
- `python -m unittest discover -s tests -p test_region.py -v`: 22 passed; the
  native-derived mixed-Z/bulge outer and hole survive two XML cycles within the
  default ten-decimal serialization tolerance (`1e-9` asserted).
- `python -m unittest discover -s tests -p test_shape_parity.py -v`: 10 passed;
  a mixed-Z/bulge Pline and Region retain world geometry, identity and the Region's
  Profile MOP reference across two complete project XML round trips.
- `python -m unittest discover -s tests -q`: all 142 tests passed, including
  relationships, Region holes, MOP references, transforms, bounds and persistence.

No additional CamBam acceptance is required: the new supported forms are derived
from XML the user created and exercised in CamBam. Production toolpath generation
and exact interpolation between varying-Z arc endpoints remain separate work.

Suggested commit: `feat: support varying-z bulges in plines and regions`.

# Packaging and supported Python verification

Date: 2026-09-10.

The former Python >=3.8 declaration was not credible because the distributed
legacy implementation evaluates built-in generic annotations such as
`list[tuple]`; the validation host also had no Python 3.8 interpreter. The
user authorized Python 3.9 as the minimum, so published metadata now matches the
lowest interpreter actually exercised rather than claiming an untested repair.
NumPy moved from setuptools' dynamic `setup.py` adapter into standard static
project metadata, allowing pip, build frontends and `uv` to consume one contract.
The redundant `setup.py` and `requirements.txt` files were removed. The lower
bound is NumPy 1.23.5, the oldest version with recorded full-suite evidence in
this repository; compatible newer releases remain independently resolvable.
The generated `uv.lock` is intentionally local and ignored at the user's request;
the tradeoff is that future environment resolution is current rather than exactly
reproducible, which matches the requirement to exercise unpinned NumPy separately.

`uv build` produced an sdist and a `py3-none-any` wheel in the ignored
`output/packaging-validation-20260910-c/` directory. Archive inspection confirmed
both package roots, all active modern modules, the MIT license and metadata for
version 0.1.0, Python >=3.9 and NumPy. Separate clean wheel environments used
Python/NumPy 3.9.0/2.0.2, 3.10.9/2.2.6, 3.11.0/2.4.6, 3.12.10/2.5.3 and
3.13.9/2.5.3. Each asserted that the imported modern package lived outside the
repository, imported both package roots, checked installed metadata, completed a
representative construction and XML write/read, and passed all 142 tests. A
separate sdist install on Python 3.9/NumPy 2.0.2 passed the same installed-path
and import checks plus all 142 tests.

After removing the compatibility `setup.py` and duplicate `requirements.txt`, a
fresh sdist and wheel were rebuilt. Neither removed file appears in the sdist;
the wheel imports both package roots on Python 3.9-3.13, and the setup-free sdist
installs on Python 3.9 and passes all 142 tests from outside the repository.

No CLI entry point or publishing flow was added. The shipped historical CLI
module is not imported by the legacy package root and remains outside the
supported surface. Reopen Python 3.8 only if a concrete consumer requires it and
the entire distribution plus dependency resolution can be tested there; reopen
CLI/distribution work only for an actual user workflow.

# MCP protocol and adapter contract - 2026-09-10

Backlog 4a is a contract/probe outcome, not an implemented adapter. The lasting
decisions and first-slice acceptance are in [MCP_CONTRACT.md](MCP_CONTRACT.md),
with [machine-readable input/output schemas](../cambam_builder/mcp_adapter/contract_v1.schema.json).
No runtime source, base dependency, installed client configuration or launcher
was changed. Disposable scripts, environments, public-source snapshots and
synthetic artifacts are under `output/mcp-contract-haugv1gu/`.

Primary evidence checked on 2026-09-10:

- The [normative core](https://modelcontextprotocol.io/specification/2026-07-28/basic/index)
  requires per-request protocol version and client capabilities; client identity
  is optional. The [stdio binding](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
  retains local subprocess transport without an initialization requirement.
- The [SDK v2 migration guide](https://py.sdk.modelcontextprotocol.io/migration/)
  describes `MCPServer`, split `mcp_types` and modern support. Actual PyPI metadata
  selected `mcp==2.2.0`, `mcp-types==2.2.0`, Python >=3.10. An isolated environment
  used Python 3.13.9, Pydantic 2.13.5 and jsonschema 4.26.0. No SDK was installed
  into the project `.venv`.
- [OpenCode MCP source at 1.18.29](https://github.com/anomalyco/opencode/blob/v1.18.29/packages/opencode/src/mcp/index.ts)
  has stdio, Streamable HTTP and legacy SSE transports. Its
  [package declaration](https://github.com/anomalyco/opencode/blob/v1.18.29/packages/opencode/package.json)
  pins TS SDK 1.29.0. The installed backend binary reports 1.18.29.

The Python SDK subprocess successfully answered `tools/call` as the first request,
then `tools/list`, then `server/discover`, with `2026-07-28` metadata on each.
The synthetic typed `echo(value: int) -> EchoResult` returned structured
`{"value":7}` and `resultType="complete"`; closing stdin exited 0. A bare `dict`
return annotation did not produce `structuredContent`, so adapter handlers must use
explicit typed result models/output schemas. The contract requires a middleware
version gate because SDK v2 also supports legacy clients; the probe did not claim
that the SDK's default server rejects legacy initialization.

The installed OpenCode backend was run through `mcp list --pure` with separate
XDG config/data/state/cache directories, project config/default plugins/model fetch
and auto-update disabled, and only one synthetic local server. It sent:

```json
{"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{"roots":{}},"clientInfo":{"name":"opencode","version":"1.18.29"}},"jsonrpc":"2.0","id":0}
```

A mock returning method-not-found to initialization made OpenCode report `failed`.
A control mock answering initialization at `2025-11-25`, advertising tools and
returning an empty catalog made it report `connected`, followed by
`notifications/initialized` and `tools/list`. Both CLI commands exited **0**;
the wire transcript and displayed status, not exit status, establish the result.
The modern mock is a negative compatibility detector, not a complete conforming
MCP server. This proves the installed backend's handshake behavior; it does not
prove the version or behavior of an installed Desktop GUI. Modern Desktop
compatibility remains unaccepted and gates 4e. No model was invoked or user CAD
data sent to a service. Reopen with a client release that claims modern support,
then require modern wire evidence rather than mere successful connection.

Commands actually run from the repository root:

```powershell
uv venv --python .venv/Scripts/python.exe output/mcp-contract-haugv1gu/sdk-env
uv pip install --python output/mcp-contract-haugv1gu/sdk-env/Scripts/python.exe mcp==2.2.0
.venv/Scripts/python.exe output/mcp-contract-haugv1gu/client_probe.py
output/mcp-contract-haugv1gu/sdk-env/Scripts/python.exe output/mcp-contract-haugv1gu/sdk_probe.py
output/mcp-contract-haugv1gu/sdk-env/Scripts/python.exe output/mcp-contract-haugv1gu/validate_schemas.py
.venv/Scripts/python.exe output/mcp-contract-haugv1gu/framework_probe.py
git diff --check
```

The initial default-sandbox external fetch and uv cache access failed; authorized
escalated commands completed the disposable environment/probes. PowerShell's
`opencode.ps1` shim was blocked by execution policy, so the installed `.exe` was
used without changing policy. One probe print initially hit cp1252 console encoding;
UTF-8 output corrected it. These were harness/environment failures, not protocol
support. Final schema synchronization was interrupted when automatic approval
review hit the session usage limit. After the user resumed the task, the approved
verification completed; no approval rejection remains unresolved.

Reproduction in a fresh checkout does not require the ignored scripts:

1. Create a unique `output/mcp-contract-<suffix>/` environment with the two uv
   commands above, substituting that directory. Run a Python subprocess server
   using `from mcp.server import MCPServer`, a Pydantic model `EchoResult` with
   `value: int`, and an `echo` tool returning `EchoResult(value=value)`. Call
   `server.run()` for stdio. A parent process writes each of the following JSON-RPC
   requests as one line, reads and checks the matching response before continuing,
   then closes stdin and waits for exit:

   ```text
   params._meta = {"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                   "io.modelcontextprotocol/clientCapabilities":{},
                   "io.modelcontextprotocol/clientInfo":{"name":"cambam-probe","version":"0"}}
   id 1: tools/call, params.name="echo", params.arguments={"value":7}
   id 2: tools/list
   id 3: server/discover
   ```

2. For OpenCode, configure `mcp.probe` as `type="local"`, a command array for
   environment Python plus a synthetic mock script, and `timeout=3000`. Use a
   copied subprocess environment with `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
   `XDG_STATE_HOME`, `XDG_CACHE_HOME` pointing inside the task directory;
   `OPENCODE_CONFIG_DIR` points to its config directory and
   `OPENCODE_CONFIG_CONTENT` contains only that MCP configuration. Set
   `OPENCODE_DISABLE_PROJECT_CONFIG`, `OPENCODE_DISABLE_DEFAULT_PLUGINS`,
   `OPENCODE_DISABLE_MODELS_FETCH`, `OPENCODE_DISABLE_AUTOUPDATE` to `true`.
   Run the installed executable's `mcp list --pure` with that directory as cwd
   and a 40-second process timeout. The mock records each inbound JSON line;
   for any request with `id`, echo the ID and return JSON-RPC error `-32601`.
   Repeat with a control that returns initialization result
   `{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"synthetic-probe","version":"0"}}`
   and `{"tools":[]}` to `tools/list`; ignore notifications. Expected transcripts
   and interpretation are above. Do not read real client credentials/config.
3. Validate the tracked schema with `jsonschema.Draft202012Validator.check_schema`.
   To validate a tool, construct `{"$ref":"#/$defs/<tool>_input","$defs":...}`
   (or `_output`) using its definitions. Exercise every tool's example inputs,
   missing required fields, unknown properties, zero width, boolean width, string
   translation, oversized page, absolute/wrong-extension paths, duplicate targets,
   and every success/error envelope. Schema validation alone does not prove
   canonical filesystem containment or state/retry behavior.
4. Author the exact direct-framework A/B slice in the contract's acceptance section
   using `CBProject`, `add_rect`, `add_part`, `add_profile_mop`, `save`,
   `read_cambam_file`, `translate_primitive`, then `save`/reload. This independently
   confirms that 4c's target values are supported by the current framework.

Results: schema meta-validation plus **98 assertions** across eight tool inputs
and success/error outputs passed. After contract review, **7 additional assertions**
passed for the workspace bootstrap schema and terminal cancellation error, for
**105 assertions total**. The direct A/B workflow retained primitive/MOP
UUIDs, targets, six explicit machining values and the original A bytes. B corners
matched (5,2,0), (25,2,0), (25,12,0), (5,12,0), absolute tolerance `1e-9`.
This is a direct API reference probe, not MCP parity implementation.

The API review also reproduced that `copy.deepcopy(project)` leaves primitive
project links detached: a cloned child loses its translated parent's world pose.
`Primitive.__getstate__/__setstate__` deliberately remove the link; only pickle
loading currently repairs it. 4b needs public `CamBamProject.clone()` that rebinds
cloned primitives, plus independent-mutation/parent-transform/MOP-template tests.
The reader currently skips unknown MOP types; 4b needs the strict byte-snapshot
reader contract rather than duplicate XML policy inside tool handlers. These
prerequisites are intentionally small owning-framework changes, not adapter access
to private registries or a broader geometry refactor.

Independent contract review identified bootstrap/error-envelope ambiguity,
concurrent document-capacity reservation, canceled ledger entries, exhaustion
recovery and saved-artifact revision ambiguity. The final contract specifies the
discovery metadata key and stderr grammar, canonical root hash, configured workspace
ID in error envelopes, a registry reservation lock, terminal `REQUEST_CANCELLED`,
save/close access after regular ledger exhaustion, and a document lock held through
save publication and ledger completion. The schemas include the bootstrap record
and cancellation error. These were reviewed design rules at the 4a boundary;
the 4b implementation and verification are recorded in the next section. Final
local documentation link/anchor and schema-reference checks passed, as did
`git diff --check`.

No manual validation was needed for 4a. Runtime/security implementation and full
framework regression checks are now recorded under 4b; actual desktop/second-PC
use and CamBam production acceptance remain separate future evidence. Volatile handles
deliberately lose unsaved work at restart; `.cb` interchange still has limited
unknown-field preservation and no verified persisted unit setting. These limits,
new-file-only save policy and reopening criteria are in the contract.

## MCP foundation and client compatibility - 2026-09-10

The client survey distinguishes transport support, advertised protocol support,
observed wire behavior and user acceptance. The earlier modern-only 4a decision
was explicitly expanded by the user to require backward protocol compatibility.
2025-06-18 and 2025-11-25 are the initial compatibility targets because they match
the observed Codex and OpenCode clients. The modern protocol remains supported;
legacy success does not count as 2026-07-28 conformance.

| Client | Evidence | Scope of conclusion |
| --- | --- | --- |
| Codex CLI 0.154.0, Windows | `codex features list` exposes `mcp_2026_07_28`, under development and false by default. Both `codex exec` with the flag and CLI app-server with explicit runtime enablement send `initialize` at 2025-06-18. Modern-only adapter returns -32601. Legacy control accepts handshake and receives `tools/list`. | Actual negative modern-stdio test, not an inference from generic MCP documentation. The [official MCP guide](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) confirms stdio support but does not promise this modern protocol version. |
| OpenCode 1.18.29 | 4a captured 2025-11-25 initialization and successful legacy control. | Actual backend probe; Desktop GUI not tested. |
| Claude Code | [Official runtime documentation](https://code.claude.com/docs/en/mcp#mcp-client-runtimes) states v2.1.232+ has SDK v2 support, subject to provider/feature conditions; stdio modern negotiation needs `MCP_PROTOCOL_NEGOTIATION=auto`, and runtime can be selected with `MCP_SDK_GENERATION=v2`. | Documented conditional support, no local test. |
| Gemini CLI | Retrieved [core dependency declaration](https://raw.githubusercontent.com/google-gemini/gemini-cli/main/packages/core/package.json) pins `@modelcontextprotocol/sdk` 1.23.0. | Evidence suggesting a legacy client path in retrieved source; no installed-version or modern wire acceptance. |
| Cursor | [Official MCP documentation](https://cursor.com/docs/mcp) lists stdio/HTTP/SSE and features but no explicit 2026-07-28 claim. | Modern compatibility unverified; lack of a claim is not proof of incompatibility. |

This is a small targeted survey, not a statistical claim about all clients.
Compatibility friction is sufficient evidence to require legacy acceptance rather
than block delivery on modern client adoption. Recheck exact installed versions
and wire traffic when expanding the client matrix.

Disposable evidence is retained in `output/mcp-foundation-20260910/`:
`modern-wire.jsonl`/`modern-codex.stderr`, `runtime-wire.jsonl`/
`runtime-host.jsonl`, and `control-wire.jsonl`. The runtime feature-enable response
confirms the flag was accepted before the negative probe; the legacy control
removes a launch/stdio failure as the explanation. Only synthetic task data was
used; no global MCP configuration was edited. The CLI app-server probe calls the
installed client directly without an LLM request. Early harness issues (buffered
stdio read, SDK snake_case result access) were corrected before acceptance runs.

The foundation introduces the optional adapter package, independent public
project clone and strict bounded UTF-8 XML snapshot reader. Document operations
use explicit workspace/boot handles, locks, capacity reservations, revision checks
and a process-lifetime terminal-result ledger. Saving stages a clone, checks and
flushes bytes, and publishes with no-replace hard-link creation. Failed staging or
destination races preserve existing bytes; successful publication survives retry
and post-publication cancellation. Typed CAD detail and authoring tools remain
4c scope. Framework regressions are separate from protocol/client acceptance.

After adding SDK-backed legacy support, `codex exec` completed create, inspect,
save, open and reopened inspection. Its default noninteractive `never` approval
policy refused close; this was a client approval configuration issue, not a server
error. A second synthetic workflow through `codex app-server` used a process-local
`tools.document_close.approval_mode="auto"` override and completed all seven calls,
including both closes. The lead independently checked every application result,
both closed results and the saved file's SHA-256 against the returned artifact.
`compat-host-wire.jsonl` records actual 2025-06-18 negotiation;
`compat-host-host.jsonl` records client results. `compat-codex.jsonl` retains the
first agentic workflow and its approval limitation. The saved empty document is
`client-compat-host/Codex.cb`, 410 bytes, SHA-256
`36e3244907051031ec52e5acdd2d411eb9b0d5af0ace9488144b7ab02adfe688`.
This selects **Codex 0.154.0 on the legacy compatibility path** for subsequent
work. Desktop GUI, second-PC and authoring/CAM acceptance remain separate.

The final full run completed 175 tests in 15.676 seconds: 174 passed and one
symlink-creation case skipped because this Windows account lacks that privilege.
The separate junction/reparse-point rejection test passed. `compileall` and
`git diff --check` also pass. A fresh wheel and sdist in
`output/mcp-foundation-final-20260910-3/dist/` contain the packaged contract.
Clean wheel installs on Python 3.10.9 and 3.13.9 load MCP 2.2.0 and the five
document schemas from site-packages. A clean Python 3.9.0 base install imports
and clones projects without installing MCP, while `cambam-mcp` exits with the
documented Python-version message. The installed Python 3.13 executable completed
discovery/listing at 2026-07-28 and initialize/listing at both legacy versions.
Independent final review found that an initial metadata-free direct request could
otherwise make the SDK select its legacy loop implicitly. The stdio reader now
rejects that request with `INVALID_PARAMS` before era selection; only an explicit
legacy `initialize` can select legacy. The subprocess regression and installed-wheel
probe confirm rejection followed by a valid modern discovery on the same process.

## MCP first authoring and round-trip slice - 2026-09-10

Backlog 4c implements the three reserved authoring tools and advertises the complete
eight-tool version 1 catalog. Rectangle, Profile and translation mutations clone
the project under the document lock, validate all slice-specific preconditions and
publish the staged project with one revision increment. Identifier and target
resolution uses public framework getters; absent layers/parts are created through
the public adders, with the contract's zero-stock Part values supplied explicitly.
Root Rect inspection uses public world-coordinate and bounding-box queries. Profile
inspection returns the closed explicit parameter record only when type, target,
parameter values and imported inheritance state remain inside 4c; other details
stay identity/relationship-only with `INSPECTION_UNSUPPORTED`.

The focused regression in `tests/test_mcp_authoring.py` runs the exact contract
workflow and an independently authored `CBProject` reference. It checks revision
0/1/2 and reopened 0/1 transitions; primitive/MOP UUIDs, target and ordering
preservation; all exposed Profile values and fixed settings; cyclic world corners
within `1e-9`; semantic XML equality after normalizing UUIDs and equivalent numeric
spellings; translation retry; concurrent same-revision serialization; and unchanged
A bytes. Negative cases cover malformed arguments, stale revisions, missing,
duplicate, wrong-kind and transformed targets, cross-kind identifier conflicts,
out-of-slice inspection and complete inspection preservation after failed edits.
The regression exposed that schema reference expansion had discarded sibling
defaults such as rectangle `z`; expansion now merges those keywords and the tests
exercise omitted optional defaults.

Independent runtime review found two pre-closure inspection gaps: fixed public
Profile fields omitted from the returned record were not yet part of slice
classification, and native parent-container `Default` states could look explicit.
The final classifier checks every fixed public field and requires the expected XML
leaf/ancestor path to be present, never `Default`, and explicitly `Value` at some
level. Regressions mutate `optimisation_mode` and `HoldingTabs` inheritance to keep
both cases identity-only with `INSPECTION_UNSUPPORTED`. The same review found no
other high/medium issue in atomicity, cancellation, revisions, ledger behavior,
error mapping or schema paths.

Verification on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_authoring.py -v`:
  5 passed.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  27 passed and one Windows symlink-privilege test skipped; junction/reparse
  rejection passes.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 179 passed and
  the same one skip.
- `.venv/Scripts/python.exe -m compileall -q cambam_builder legacy_cambam_builder tests demos`,
  contract meta-validation/tool-catalog smoke and `git diff --check`: pass.
- `.venv/Scripts/python.exe demos/mcp_authoring_slice.py`: passed and wrote
  inspected A/B files to
  `output/mcp-authoring-demo-bba34075e1d0/`. A is 3,821 bytes with SHA-256
  `3b04f2abe882de5d7105c9729e20a3025d5f531d1893042ea3470d0c71639896`; B is
  3,826 bytes with SHA-256
  `ca08657e667b508cb6d0043245c0581dbc1771ca2250b6794f4490bf55ee78c6`.
  The script verified B corners `(5,2,0)`, `(25,2,0)`, `(25,12,0)`, `(5,12,0)`,
  exact revision milestones, retry replay, returned hashes and unchanged A bytes.

The standard managed sandbox cannot reopen Python-created OS temporary directories;
the focused and full temporary-workspace tests therefore used the approved local
filesystem path, matching the existing MCP runbook. No manual check adds evidence
to this framework/adapter parity slice. Named desktop/second-PC connection and
CamBam units, property and toolpath acceptance remain backlog 4e. Broader entity,
relationship and operation coverage remains 4d and must not be inferred from this
Rect/Profile proof.

## MCP geometry breadth batch 1 - 2026-09-10

Backlog 4d's reopening criterion asked for one concrete high-value family from
the contract's documented mappings. The curve/point geometry family
(Circle/Arc/Pline/Points adders with their public world queries) was selected:
Rect-only authoring blocked most drawing work, and later Drill/Pocket MOP
breadth needs Points/Circle targets to be expressible at all. Text, Region,
Pocket/Engrave/Drill, parenting/groups/copy-transfer and bake transforms stay
explicitly unsupported with reopening criteria in
[PROGRESS.md](PROGRESS.md#active-work-and-next-priority).

Implemented contract surface (now owned by the packaged schema
`cambam_builder/mcp_adapter/contract_v1.schema.json`; contract sections were updated
in the same increment):

- Four new tools map only to public framework adders:
  `geometry_add_circle` (center/diameter/elevation), `geometry_add_arc`
  (center/radius/CCW-positive degree sweep/elevation), `geometry_add_pline`
  (2..10000 vertex records with per-vertex Z and bulge, optional `closed`) and
  `geometry_add_points` (1..10000 plain XYZ points; bulge input is rejected by
  the schema). All create root primitives on a possibly new default layer and
  return the primitive UUID plus layer name.
- `geometry_translate` accepts root Rect/Circle/Arc/Pline/Points primitives in
  the translation-only slice (identity or pure-translation world matrix, zero
  local Z offset, no parent/children/groups, valid positive geometry) and still
  never bakes. Profile targets deliberately remain root Rects only.
- Inspection's primitive record now carries one closed typed `geometry` payload
  instead of top-level `world_xyz`/`bounds`: `rect` corners, `circle`
  center/diameter, `arc` center/radius/degree angles, `pline` vertices with a
  parallel `bulges` array plus `closed`, and `points` vertices, each with
  directed analytic world bounds. The bulge convention is documented: the bulge
  stored on vertex i curves the segment that starts at vertex i. Unsupported
  primitives (Text, Region, transformed or related shapes) keep
  `geometry: null` with `INSPECTION_UNSUPPORTED`; no geometry is invented.
- Retry identity canonicalizes nested defaults, so omitting `z`/`bulge` and
  sending explicit zeros replay as the same request instead of conflicting.
- Verification notes: the Arc world `start_angle` comes from the public
  direction-based query and may differ from the authored angle by float noise
  (30 degrees reports 29.999999999999996 before the 9-decimal XML round trip),
  and primitive document order is the framework's UUID-sorted
  `list_primitives()` order, not insertion order. Parity fixtures therefore
  compare inspection records with a tolerance-aware deep comparison and XML
  with order/id-canonicalized semantics; the saved artifacts themselves are
  bit-compared only within one run (A bytes unchanged across reopen).

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_geometry.py -v`:
  5 passed (analytic guard, round-trip parity, translate parity/replay,
  negative/conflict/atomicity, imported out-of-slice diagnostics).
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  33 tests, one Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 185 tests with
  the same one skip (184 passed).
- `.venv/Scripts/python.exe -m compileall -q cambam_builder legacy_cambam_builder tests demos`,
  `git diff --check`, and `.venv/Scripts/python.exe demos/mcp_authoring_slice.py`
  (updated for the `geometry` payload): pass.
- Direct-framework parity confirmed for all four families: adapter inspection
  payloads equal independently computed public query values (`1e-9` for the
  noise-bearing arc angle), XML semantics match after UUID/order/id
  normalization, save/reopen retains primitive identities and typed geometry,
  and failed edits preserve complete inspection.

Remaining limits: no Text/Region detail, no Pocket/Engrave/Drill, no
parenting/groups/copy-transfer, no bake transforms, Profile targets still
Rect-only, and inspection of a very large single primitive is not separately
bounded. The next batch reopens with Text/Region authoring or another
family if user needs change the priority.

## MCP Text and Region breadth batch 2 - 2026-09-10

4d batch 2 adds the annotation/composite authoring family: `geometry_add_text`
and `geometry_add_region`, with typed inspection detail for both and
`geometry_translate` extended to root Text/Region primitives. The reopening
criterion (curve/point family complete in batch 1) made Text/Region the next
mapped family; Pocket/Engrave/Drill MOP breadth remains next.

Implemented contract surface (schema copies updated and re-verified identical):

- `geometry_add_text` maps only to the public `add_text` adder: required
  identifier/layer/content/anchor, optional height (default 10), font (default
  Arial), style (default empty), line spacing (default 1) and alignment enums,
  plus elevation. Content is bounded 1..1024 characters, rejects control
  characters except newlines, and must contain at least one non-whitespace
  character because whitespace-only content does not survive the XML round
  trip. The optional unused `xml_p2_*` interchange fields are not authorable.
- `geometry_add_region` maps only to the public `add_region` adder: one closed
  outer contour plus 0..100 closed hole contours, each 2..10000 vertex records
  with per-vertex Z/bulge. The framework validates XY topology (closed, simple,
  finite nonzero area, holes contained and disjoint, no nesting) and topology
  failures are surfaced as `INVALID_ARGUMENT` with the bounded framework
  message by re-running the public `Region` validation on the rejected inputs;
  staged clones keep failure atomic.
- Inspection adds closed `text` payloads (anchor/height/font/style/line
  spacing/alignment plus the optional unused `p2`, no bounds because text
  extents are a font-dependent framework estimate) and `region` payloads
  (outer/hole contour `world_xyz`+`bulges` plus directed analytic bounds).
  Out-of-slice Text/Region (transformed or related primitives) stay
  `geometry: null` with `INSPECTION_UNSUPPORTED`.
- Verification notes: the batch-1 parity findings carry over (arc/text angle
  noise is absent here, but primitive document order remains the UUID-sorted
  listing), and the `objects`/id-canonicalized XML comparison is reused. A
  bulged outer Region (bulge 0.2 on the bottom segment, center (5,12), radius
  13) dips exactly to y=-1 and both the in-memory payload and the reloaded XML
  agree.

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_text_region.py -v`:
  5 passed (analytic guard, round-trip parity, translate parity/replay,
  negative/topology/conflict/atomicity, imported out-of-slice diagnostics).
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  38 tests, one Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 190 tests with
  the same one skip (189 passed).
- `compileall`, contract schema-copy identity, `git diff --check` and
  `demos/mcp_authoring_slice.py`: pass.

Remaining limits: no Pocket/Engrave/Drill MOPs, no parenting/groups/
copy-transfer, no bake transforms, Profile targets still Rect-only, text bounds
deliberately not reported. The next batch reopens with Pocket/Engrave/Drill
machining breadth or another family if user needs change the priority.

## MCP Pocket, Engrave, Drill breadth batch 3 - 2026-09-10

4d batch 3 adds the remaining basic machining MOPs and the public target-state
operation: `machining_add_pocket`, `machining_add_engrave`,
`machining_add_drill` and `machining_set_mop_targets`. This completes the four
basic CamBam MOP kinds behind closed, explicit parameter records.

Implemented contract surface (schema copies updated and re-verified identical):

- The three adders map only to the public `add_pocket_mop`/`add_engrave_mop`/
  `add_drill_mop` adders. Inputs are the shared closed machining record
  (targets, depth/increment, tool, feeds, spindle, stock surface, clearance,
  enabled); Pocket additionally pins stepover 0.4, `Plunge Feedrate` stepover
  feed, Conventional milling, collision detection, Spiral lead-in,
  `InsideOutsideOffsets` fill, finish stepover 0 and Roughing; Engrave pins
  Roughing, final increment 0 and DepthFirst with an EndMill; Drill pins the
  CannedCycle method, a `Drill` tool profile and exposes peck distance,
  retract height and dwell (defaults 0/5/0). All non-input settings match the
  current public framework defaults; Profile keeps its 4c record unchanged.
- `machining_set_mop_targets` maps to the public `set_mop_targets` and enforces
  the same per-kind target rules as creation before replacing the
  project-owned selection atomically on the staged clone. Targets are returned
  in the project's UUID-sorted snapshot order in results and inspection.
- Per-kind target rules are explicit: Profile root Rects; Pocket root
  Rect/Circle/closed-Pline/Region; Engrave root Rect/Circle/Arc/Pline; Drill
  root Points/Circle. CamBam resolves drill positions (point locations, circle
  centers) at toolpath time; that production semantics acceptance stays with 4e.
- Inspection generalizes the Profile parameter guard to all four kinds: a
  record is returned only when every exposed value matches the pinned fixed
  settings, every unexposed public field keeps its default, all scalar/range
  checks pass, every returned field's XML path is explicitly `Value` (never
  inherited) and every target resolves slice-valid under the kind rule.
  Imported or edited MOPs outside the slice stay diagnostic (`{}` parameters
  with `INSPECTION_UNSUPPORTED`).
- The shared prelude also centralizes depth/clearance relations, identifier
  and part-name conflicts, part auto-creation and limit checks for all four
  MOP adders; behavior for Profile is unchanged.

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_mops.py -v`:
  2 passed (pocket/engrave/drill creation + target replacement round trip with
  direct-framework XML parity; target-kind rules, negatives, replay,
  stale-revision and concurrency serialization).
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  40 tests, one Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 192 tests with
  the same one skip (191 passed).
- `compileall`, contract schema-copy identity, `git diff --check` and
  `demos/mcp_authoring_slice.py`: pass.

Remaining limits: no parenting/groups/copy-transfer tools, no bake/rotate/scale/
mirror/translate-Z tools, CamBam production toolpath semantics unverified
(4e), and MOP parameter editing beyond target replacement remains excluded.
The next batch reopens with the relationship/transform family.

## MCP relationship and transform breadth batch 4 - 2026-09-10

4d batch 4 adds the relationship and transform families and generalizes the
inspection slice from translation-only to similarity transforms, completing
the documented 4d mapping except cross-document copy/transfer.

Implemented contract surface (schema copies updated and re-verified identical):

- `relationship_set_parent` maps to the public `link_primitive_parent` (null
  detaches). Local transforms are preserved, so the world pose follows the new
  frame; self links and framework-rejected cycles return `INVALID_ARGUMENT`
  with complete inspection preservation. After linking, both endpoints leave
  the similarity slice by contract (relationships are diagnostic), which is
  asserted in inspection.
- `relationship_add_to_group`/`relationship_remove_from_group` map to the
  public group membership methods and return the sorted group names. Primitive
  inspection records now carry their sorted `groups` field.
- `relationship_copy_tree` maps to the public `copy_primitive_tree` into the
  same document with `preserve_ids=False`: copies get fresh UUIDs, and the
  framework's collision rejection (layer, primitive, group, part, MOP names)
  surfaces as `INVALID_ARGUMENT` unless the caller supplies the explicit
  identifier/group maps. `include_mops` (default false) copies MOPs whose
  selections lie inside the subtree. The result mapping includes copied
  layers/parts as well as primitives. Copied childless primitives return to
  the supported slice with typed geometry, and copied MOPs keep valid
  parameter records.
- Transform tools map to public methods only: `geometry_translate_z` uses
  `translate_primitive_z(..., bake=True)` (stored-geometry Z shift, world
  matrices retained), `geometry_rotate`/`geometry_scale`/`geometry_mirror`
  compose similarity world transforms (`bake=False`; scale is uniform-only
  because non-uniform scale leaves the slice), and `geometry_bake` calls the
  public `bake_geometry()` (non-axis-aligned Rects become closed Plines and
  the result type is returned; Text bakes only translation/positive uniform
  scale and otherwise fails `UNSUPPORTED_OPERATION`).
- The similarity slice: `_slice_supported` now accepts any root primitive
  whose world matrix is a finite non-degenerate XY similarity (translation,
  rotation, uniform scale, reflection) with zero local Z offset and no
  relationships. All public world queries and analytic bounds already support
  similarities, so rotated/scaled/mirrored primitives keep typed payloads and
  reflection flips bulge signs via the framework's orientation rule. The
  previously rotated-out-of-slice expectations in earlier suites were updated
  to non-uniform scale/shear cases, which remain diagnostic.
- Deliberately deferred: cross-document copy/transfer. The framework supports
  it (`transfer_primitive_tree` to another project), but an adapter tool
  spans two document handles, two revisions and one request ledger key, and
  cannot publish atomically across documents. A two-document staging,
  revision and failure-semantics decision must be recorded in the contract
  before such tools are advertised; the reopening criterion is in
  [PROGRESS.md](PROGRESS.md#active-work-and-next-priority).

Post-review corrections close three resilience/contract gaps before commit:
same-document copies now recheck the 10,000-primitive/1,000-MOP hard limits
before publication; typed geometry payloads are checked against their advertised
output schema so oversized imported Pline/Text detail returns `geometry: null`
with `INSPECTION_UNSUPPORTED`; and create/open/new-file save annotations are
nondestructive as required. Boundary regressions verify atomic state/revision
preservation for both copy limits, strict reopen of a 10,001-vertex Pline and a
1,025-character Text, and exact annotations for all advertised tools.

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_relationships_transforms.py -v`:
  5 passed (similarity transform parity incl. bake; parent/group/copy parity
  with cycle/self/missing rejection and copy collisions; copy-with-MOPs slice
  validity and limit atomicity; transform negatives, replay, stale and concurrency).
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  45 tests, one Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 197 tests with
  the same one skip (196 passed).
- `compileall`, contract schema-copy identity, `git diff --check` and
  `demos/mcp_authoring_slice.py`: pass.

Remaining limits: cross-document copy/transfer (deferred above), no generic
field setters or deletion tools, and CamBam production acceptance stays with
4e. Same-document copy requires explicit identifier/group maps for every
included non-unique name by framework contract.

## MCP portable document interchange - 2026-09-11

Client-local `.cb` files no longer need to be copied into the server workspace.
The adapter adds `document_import`, which accepts a complete UTF-8 XML snapshot
and publishes a revision-0 volatile handle through the existing strict reader,
and `document_export`, which serializes the locked revision to an in-memory byte
snapshot and returns a typed client artifact with filename, media type, encoding,
byte count, SHA-256 and complete XML content. Import retains only the content
digest and byte count in its process-lifetime retry signature; export is read-only
and is not retained in that ledger. Neither operation interprets a client name as
a server path, and export creates no server filesystem artifact.

The established 10 MiB XML limit remains unchanged and is enforced on encoded
UTF-8 bytes in both directions. Stdio framing increased from 1 MiB to 32 MiB only
to accommodate a valid 10 MiB payload after JSON string escaping and envelope
overhead. Inline content is the compatibility baseline because named-client
resource-result behavior is not yet accepted; 4e may add resource delivery if
it demonstrably reduces context cost without hiding the artifact from agents.
The standard tool result currently includes both structured content and the
backward-compatible serialized text block, so a large export may cross the client
boundary twice. This is a known context/network cost, not a correctness gap.

Automated coverage verifies import/edit/export/reimport geometry and identity,
source and artifact hashes, revisions, import retry/conflict behavior, malformed
and declaration-bearing XML rejection, exact 10 MiB acceptance, over-limit
rejection, invalid content types, stale export, an export/edit lock race and
bounded export failure. A real stdio subprocess moves an exact-10-MiB XML request
through the modern protocol path and recovers the exported artifact; a legacy
initialized connection also exports and reimports a document. Existing save
failure/atomic replacement regressions pass after extracting the shared in-memory
byte serializer.

Verification on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_documents.py -v`:
  18 tests, one existing Windows symlink-privilege skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_protocol.py -v`:
  9 passed, including the above-1-MiB real stdio round trip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_export_failures.py -v`:
  8 passed.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  49 tests, the same one skip.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 201 tests with
  the same one skip (200 passed).
- `compileall`, schema-copy identity (inside the document suite),
  `demos/mcp_authoring_slice.py` and `git diff --check`: pass.

No manual CamBam validation adds evidence for this transport and serialization
slice because it uses the same XML builder and strict reader already covered by
the existing interchange acceptance. Named desktop-client handling of a local
file and returned inline artifact remains 4e user acceptance. The next coherent
increment remains 4d batch 5 cross-document copy, now testable with two documents
imported from client content.

## MCP cross-document copy/transfer - 2026-09-11

4d batch 5 completes the documented 4d surface with two-document subtree
operations, advertised as `relationship_copy_tree_between` and
`relationship_transfer_tree_between` (thirty-one version 1 tools).

Recorded contract (now in the
[MCP contract](MCP_CONTRACT.md#cross-document-copy-and-transfer)): the tools
require two distinct live handles in this workspace and boot, each with its own
required revision assertion. Both document locks are acquired in canonical
sorted-handle order and held from the revision checks through publication;
concurrent edits/saves wait and concurrent closes cannot remove a locked
document. Copy stages on an independent full-project clone of the target and
reads the live source (the public copy never mutates it); transfer stages
independent clones of both documents and runs the public
`transfer_primitive_tree` between the clones, so source removal and target
insertion are one transactional framework operation. Publication replaces the
staged projects and increments revisions inside a cancellation shield with no
await between assignments: copy increments only the target revision, transfer
increments both by exactly one, and no partial two-document publication is
observable. Per-call failures (root resolution, framework collision/topology
rejection as `INVALID_ARGUMENT` with the bounded message, staged-target
limit overflow, per-document handle/revision mismatch) leave both documents,
both revisions and all files unchanged. `DomainError` gained an optional
addressed handle: a failing target document check (expired, not found, stale
target revision) references the target handle in the result envelope with its
current revision, and every other outcome references the source. Success data
always carries `{mapping, source_document, target_document, source_revision,
target_revision}` while the envelope's `document`/`revision` refer to the
source. Ledger rules are unchanged: regular edit entry per
`(workspace_id, request_id)`, replay before stale/closed checks, and
`REQUEST_ID_CONFLICT` on argument reuse.

Coverage: six tests in `tests/test_mcp_cross_document.py` — copy parity
between two imported documents (parented/grouped subtree with layer, part and
Profile MOP closure; a second childless copy returns to the typed similarity
slice; both saved sides match independently authored public-framework projects
by XML semantics); transfer into a populated imported document with both-revision
advancement, preservation of its existing content, source removal keeping the
untouched primitive/layer/part, and two-sided XML parity; failure addressing and
state preservation (same-handle
rejection, expired and closed targets addressed to the target, stale
source/target fields each reporting that document's current revision, missing
and non-primitive roots, framework collision message, failure replay with the
original key, and `REQUEST_ID_CONFLICT`); and limit/concurrency atomicity
(patched primitive and MOP limits reject without publication; two concurrent
same-revision transfers produce exactly one success and one `STALE_REVISION`
with correct final counts). Cancellation before publication preserves both live
project objects and revisions and replays `REQUEST_CANCELLED`; cancellation
inside the shielded publication completes both revisions and is recoverable as
a successful replay. Opposing A-to-B and B-to-A transfers complete under a
five-second timeout, proving both directions contend in canonical lock order.

Results on the repository Python 3.13 environment with MCP 2.2.0:

- `.venv/Scripts/python.exe -m unittest discover -s tests -p test_mcp_cross_document.py -v`:
  6 passed.
- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_mcp_*.py' -v`:
  55 tests, one Windows symlink-privilege skip (tool listing/annotation
  expectations updated to the thirty-one tools).
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: 207 tests with
  the same one skip (206 passed).
- `compileall`, contract schema-copy identity (inside the document suite),
  `demos/mcp_authoring_slice.py` and `git diff --check`: pass.

Remaining limits: no deletion or batch tools, no cross-restart recovery of
volatile state, and CamBam production acceptance stays with 4e. 4d is
complete; the next increment is 4e desktop/second-PC acceptance.

## MCP agentic OpenCode usability correction - 2026-09-11

The user's first agent-driven OpenCode run supplied production evidence that the
connection itself worked but the advertised behavior did not guide the agent reliably.
It exported a current document and then unnecessarily saved an older server-workspace
copy, tried to access that server absolute path with client filesystem tools, and later
copied the stale server copy instead of the current export. The second document's
successful export-to-client-write flow demonstrated that stdio inline transfer was
working; the failure was the model-facing contract and default workflow.

Initialization instructions and document tool descriptions now make portable content
exchange the default for client-project files: client read -> `document_import` for
existing files, and `document_export` -> client write for every new or changed result.
They prohibit `document_save` or access to its returned server `absolute_path` for
client-local delivery. Open/save remain available only for explicitly requested
server-workspace artifacts, and successful saves now repeat this boundary in a
`SERVER_WORKSPACE_ONLY` diagnostic. The README and 4e runbook use this workflow.

This 2026-09-11 policy was superseded by the 2026-09-18 audit below: same-host
clients may use a server-generated save only as an exact-byte intermediate, copy it
deterministically, verify it and re-import it. Arbitrary client paths remain forbidden.

The same session also found an artificial implementation restriction: the adapter
allowed Profile only on Rect, despite the public framework accepting primitive targets.
Profile now accepts the bounded closed-contour set Rect, Circle, closed Pline and Region;
open Pline remains rejected because the adapter's required Inside/Outside input does not
express the direction-dependent semantics of open geometry. Regression creates one
Profile over Circle/closed-Pline/Region targets, inspects it, saves and reopens it with
all target UUIDs retained, and checks open-Pline/Points rejection. This matches CamBam's
documented semantics: [Profile offsets inside or outside selected shapes](https://www.cambam.info/doc/api/MOPProfile.htm),
[Engrave follows selected geometry while Pocket clears a bounded area](https://www.cambam.info/doc/1.0/cam/basics.html),
and [Drill uses point-list entries or circle centers](https://www.cambam.info/doc/plus/cam/Drill.htm).
The four machining tool descriptions now state those intent differences and warn that
Engrave has no cutter-radius compensation. A Profile failure must not be silently
worked around by inventing compensated Engrave geometry.
This historical restriction is superseded by the 2026-09-19 alignment correction,
which exposes open-Pline Profiles with explicit vertex-order-relative semantics.

The transcript also exposed two smaller diagnostics issues. An invalid non-hex request
UUID had returned a schema error without a field; top-level JSON Schema failures now
report their input field (`request_id` in that case). A Part name collision with an
existing entity was valid behavior, and all machining descriptions now say the Part
name must be project-unique.

Verification on the repository Python 3.13 environment with MCP 2.2.0:

- Focused MOP/protocol regressions: 12 passed.
- `python -m unittest discover -s tests -p 'test_mcp_*.py' -v`: 57 tests,
  56 passed and the existing Windows symlink-privilege skip.
- `python -m unittest discover -s tests -v`: 209 tests, 208 passed with the same skip.
- `compileall` and `git diff --check`: passed.

Automated behavior is complete for this correction. It does not prove how OpenCode's
model will act from the revised descriptions or that CamBam generates the expected
toolpaths. The following external run accepted content-transfer behavior; geometry/CAM
planning and CamBam visual Profile inspection remain required 4e checks.

## MCP agent geometry feedback correction - 2026-09-11

The second external OpenCode run successfully used `document_export` followed by the
client's local write tool without attempting to access the server workspace. This
accepts the portable file-boundary behavior for OpenCode 1.18.30 in the tested session.

Its geometry was nevertheless wrong before reaching the adapter. A requested 200 by
200 bounding box with lower-left `(10,20)` has center `(110,120)`, but the agent sent
the circle center as `(210,220)`. Seven Pline vertices followed the intended translated
radius-100 construction while one bottom vertex was sent as `(110,49.2893)` instead of
`(110,20)`, breaking symmetry. After useful clarification questions, the agent changed
only the bulge signs and reused both incorrect coordinates without inspecting the
result. The service preserved the submitted values exactly; this was not a coordinate
transform or serialization defect.

The transcript also demonstrated why only describing bulge as a per-vertex value was
insufficient. Its exact convention is now advertised: the value belongs to the directed
segment from its vertex to the next, positive is a counter-clockwise sweep on the right
of that chord, negative is on the left, and magnitude is `tan(abs(sweep)/4)`. Which sign
means semantic inward depends on contour winding. Circle and Pline creation results now
echo their complete typed geometry, including absolute center/world vertices, bulges and
exact curved bounds. Initialization and tool descriptions require comparing those
results and an inspection snapshot with requested dimensions, center, symmetry and
containment before dependent geometry, MOPs or export; clarifications require a full
recalculation.

Machining guidance now describes resulting material rather than only operation names:
Profile Outside preserves an exterior part edge, Profile Inside preserves an opening
edge and may release a slug, while Pocket clears the complete bounded area into chips.
Profile results echo the accepted side as well as target IDs. The agent had also invented
10 mm depth and feed/spindle values; instructions now require missing machining values
from the user or established project data.

The agent created the final outside cutout MOP before the enclosed hole MOP. The adapter
appends MOPs to a Part in tool-call order, so the exported order was faithful but unsafe
for a through-cut that releases the stock-held part. Initialization instructions now
carry the cross-tool rule: plan the whole sequence, machine enclosed/internal and
non-releasing detail work first, and place a releasing Outside Profile last. Each MOP
description reinforces the relevant local consequence. Because depth and workholding
determine whether an Outside Profile actually releases material, the server directs the
agent to ask rather than infer when that intent is unclear.

Follow-up clarified that a useful depth-increment proposal need not be a fabricated
default. For a known stock thickness, cut-through allowance, material and tool, the
instructions now tell the agent to respect the safe stepdown limit while selecting a
pass count whose nominal increment multiples slightly exceed total depth. The
penultimate pass must remain above the stock bottom, and normally at least one third of
the final pass should remain engaged in stock before entering the cut-through allowance.
The user must confirm the proposed parameter. The contract records the 9 mm stock,
0.5 mm cut-through and 3.2 mm increment example and its actual clamped final pass.
This arithmetic is now exposed through the read-only
`machining_calculate_depth_increment` tool. Exact-pass mode reproduces 3.2, 6.4 and
9.5 mm depths for that example. Maximum-stepdown mode derives the minimum feasible
pass count. Both modes flag rounding or pass constraints that reach the stock bottom
before the final pass or leave less than one third of that pass in stock. They still
return a valid explicitly requested plan rather than overruling the user. The result
exposes every derived value so an agent can explain it before confirmation.

The underlying project API supports ordered reassignment, but contract v1 exposes no
MOP reorder tool. That repair affordance remains a bounded follow-up, to be reopened if
the next named-client run still misorders operations or needs to edit an existing file.

The transcript and both exported XML payloads actually contain `Profile Outside` on the
outer Pline, not Inside. Therefore no Profile-side persistence bug is established by
this evidence. If a geometrically corrected file still displays or generates an inside
path in CamBam, retain that artifact or a property/toolpath screenshot for a separate
reader/writer/native interpretation investigation.

Verification on Python 3.13 with MCP 2.2.0:

- Focused geometry/authoring/protocol verification: 20 passed.
- MCP suite: 58 tests, 57 passed and the existing Windows symlink-privilege skip.
- Full suite: 210 tests, 209 passed with the same skip.
- The packaged schema is now the single machine-readable contract owner; its asset test,
  `compileall` and `git diff --check` pass.

### Part stock, native nesting and layer settings follow-up — 2026-09-12

The next named-client transcript showed that the MCP surface had no operation for
configuring a Part after an MOP implicitly created it, so exported stock stayed
zero/undefined. It also had no native nesting settings: the client copied nine
geometries instead of asking CamBam to repeat one Part's complete MOP sequence. The
core `Part` model now writes and reads `NestMethod`, `Rows`, `Columns`, `Spacing`,
`GridOrder` and `GridDirectionAlternate`; imported native nesting subtrees are
preserved in ordering until explicitly reconfigured. The new
`machining_configure_part` MCP tool sets stock dimensions/material/color, machining
origin, defaults and Grid/IsoGrid settings. `document_inspect` echoes these values.
Layer display properties are similarly exposed through
`document_set_layer_properties` and inspection; this is a general capability, not a
claim that the cited request asked for white layers.

Verification at that checkpoint: focused MCP/authoring/protocol/context checks
passed; the later branch review observed 212 full-suite tests with the existing
Windows symlink-privilege skip. Named OpenCode and
CamBam acceptance must still confirm that CamBam generates the requested native
3x3 nested toolpaths and interprets the stock dimensions as intended.

Generic closed-shape containment analysis and a higher-level regular-polygon constructor
could further reduce model arithmetic, but are deferred as separate API affordances;
the current increment first makes existing tools self-describing and their actual output
immediately auditable.

### OpenCode error transcript audit — 2026-09-12

The next OpenCode 1.18.30 transcript contained 17 tool calls. Fifteen completed on
their first attempt; the two failures were bounded client-input mistakes, not a
transport or dispatch mismatch. The first `geometry_add_pline` call used the same
`identifier` and newly-created `layer` name (`Octagon`), correctly returning
`IDENTIFIER_CONFLICT`; the retry used `OctagonOutline` and succeeded. The first
`document_export` call supplied an extra `request_id`, while export is intentionally
read-only and has no request ledger; it returned `INVALID_ARGUMENT`, after which the
client omitted that field and succeeded. The export artifact was then written locally
and its SHA-256 verified.

The strict contract is retained. Geometry tool descriptions and initialization now
state that a new primitive identifier must differ from its layer name, and the export
description explicitly says to omit `request_id`. Schema validation also reports the
unexpected/missing top-level field when JSON Schema rejects an object-level property,
making future client retries actionable. This audit does not establish a protocol,
stdio framing, or server-workspace boundary defect; a fresh OpenCode run remains the
named-agent acceptance required by 4e.

### OpenAI tool-schema compatibility — 2026-09-15

An OpenAI-backed OpenCode run failed before any MCP session because the shared
`TextContent` definition used the positive-lookahead pattern `(?=\S)`. The model
provider reported that regex lookaround is unsupported while compiling the tool
schemas. The pattern was rewritten as an anchored expression with an explicit
non-whitespace character and no lookaround; it still permits ordinary whitespace
around text, preserves the existing control-character exclusion, and rejects
whitespace-only values. A maintained schema test now scans every packaged pattern
for lookaround syntax. This is a provider schema-compatibility fix, not a change to
MCP transport or tool behavior.

### Profile corner overcut exposure — 2026-09-15

The framework already modeled CamBam's Profile-only `CornerOvercut` XML field,
including import and export. The public `add_profile_mop` signature now exposes
the boolean directly, and the MCP profile tool accepts and reports it through
the typed inspection record. The default remains `false`; enabling it delegates
the actual toolpath calculation to CamBam and is documented as potentially
removing extra material beside inside corners. No synthetic equivalent was added
to Pocket, Engrave or Drill because their native XML has no such field.

### MCP human/AI continuity and imported nesting repair — 2026-09-15

Branch review found that explicitly reconfiguring an imported Part deleted its
preserved native nesting marker, after which the writer attempted to append a null
XML node. Direct framework reconfiguration instead left the preserved node in place
and silently serialized the old nesting values. The repair keeps an unchanged native
subtree, including unknown children and ordering, but emits the current modeled
nesting after a direct or MCP edit. `add_part` retains its historical positional
`target_identifier`/`place_last` order and makes the new nesting options keyword-only.
The two settings tools now report cross-entity name conflicts as
`IDENTIFIER_CONFLICT`. A supplied Part default spindle speed is diagnosed as
session-only because the supported writer has no durable CamBam Part field for it.

Human/AI file continuity now has an executable contract. The read-only
`document_list` tool exposes the current boot's live handles, revisions and original
source hashes. Manual client-local saves are detected by the client rereading and
hashing the durable file; changed content is imported into a new revision-0 handle.
Restart likewise requires re-import. A stale MCP mutation is recovered by inspecting
without a revision and retrying a still-applicable change with the returned revision
and a fresh request ID. Valid UUIDs supplied uniformly to list/inspect/export/planning
are accepted and ignored rather than causing a schema-only retry.

Verification on Python 3.13 with MCP 2.2.0:

- Focused protocol, document, authoring and native-context suites: 41 tests passed
  with one Windows symlink-privilege skip.
- MCP suite: 63 tests, 62 passed with the same skip.
- Full suite: 217 tests, 216 passed with the same skip.
- `compileall`, import/construct smoke and `git diff --check` pass.

Named OpenCode and CamBam acceptance remains required for actual agent behavior,
manual-save follow-up usability and generated native nesting toolpaths.

## MCP local stdio installation and client acceptance preparation - 2026-09-11

The user accepted same-machine stdio as the initial 4e deployment boundary because
second-PC hardware is unavailable. This follows the normative stdio process model:
the named client launches one local server subprocess. Streamable HTTP, HTTPS and
remote-PC hosting remain unimplemented, non-urgent future scope; they require an
explicit origin/authentication/exposure and lifecycle contract rather than a LAN
bind added to this server.

Clean packaging evidence on Windows used a fresh wheel built from the sdist in
`output/mcp-4e-local-20260911/dist/`. Python 3.9.0 installed the base distribution,
constructed `CBProject` and produced the required Python-version guard when the MCP
launcher was attempted. Isolated Python 3.10.9, 3.11.0, 3.12.10 and 3.13.9
environments each installed the wheel's `[mcp]` extra, resolved exactly `mcp==2.2.0`,
constructed the direct project API and loaded all thirty-one packaged tool
definitions. Removing `mcp` and `mcp-types` from the Python 3.13 environment made
the launcher return its missing-extra guard while direct `CBProject` construction
still passed. This demonstrates adapter rollback without uninstalling the library or
touching `.cb` files. An initial probe referenced a nonexistent schema constant and
was replaced by the public module's `tool_definitions()` function; that harness error
occurred after a successful install and is not package evidence.

OpenCode 1.18.30 ran with isolated config/data/state/cache directories, model fetch,
project config, default plugins and auto-update disabled. No model was invoked and no
normal user configuration was read or changed. It reported the real adapter
`connected`, first from the source environment and then from the clean Python 3.12
wheel environment prepared for manual acceptance. A task-owned transparent stdio
capture showed the real client request
`initialize` with `protocolVersion=2025-11-25`, client name `opencode` and version
`1.18.30`; the real server emitted
`CAMBAM_MCP_PROTOCOL {"protocol_version":"2025-11-25","mode":"legacy"}`.
This is actual named-client/adapter negotiation, not an inference from OpenCode
source. The capture and response records are disposable evidence under
`output/mcp-4e-local-20260911/`.

The server now emits that content-free protocol record once after its first accepted
request. Subprocess regressions cover the legacy 2025-11-25 handshake and modern
2026-07-28 first request; the pre-existing compatibility suite continues to cover
2025-06-18 acceptance and cross-era rejection. A new maintained verifier accepts
the exact external-client A/B Rect/Profile artifacts only when strict import,
identities, relationships, explicit properties and translated world geometry match
the 4c contract, and rejects an untranslated B. The development runbook owns the
OpenCode configuration/prompt, rollback commands and separated CamBam acceptance.

Verification on the repository Python 3.13 environment:

- `python -m unittest discover -s tests -p 'test_mcp_*.py' -v`: 57 tests,
  56 passed and one Windows symlink-privilege skip. The first sandboxed attempt could not resolve
  test-owned Windows temporary directories; the approved filesystem run passed.
- `python -m unittest discover -s tests -v`: 209 tests, 208 passed with the same skip.
- `compileall`, `demos/mcp_authoring_slice.py`, the new verifier against that demo's
  A/B output, and `git diff --check`: passed.

Automated installation, rollback, protocol regression and named-client connection
are complete. The agentic OpenCode 4c workflow is intentionally still pending: a
connection/listing probe is not evidence that an agent can select and sequence the
tools. CamBam Plus acceptance also remains pending for asserted millimeters, A/B
geometry, explicit Profile properties and generated outside toolpaths at depths
`-0.5` and `-1.0`. 4e and the initial adapter are not complete until the user reports
those results; preserve their A/B artifacts until the report is recorded.
# MCP CamBam-save/Region/Pocket transcript correction - 2026-09-17

The supplied OpenCode transcript distinguishes three failure classes. First, the
adapter successfully created, inspected and serialized the initial document, but
`document_export` returned content rather than creating `builder-smoke-test.cb`.
The agent first claimed the suggested filename was ready, then used a client patch
write that added a newline: the resulting 856 bytes did not match the returned
855-byte SHA-256. This was a client-delivery failure even though the MCP export call
itself succeeded. Initialization and export-tool guidance now prohibit claiming
delivery before the client write and require correcting any byte/hash mismatch.

Second, CamBam's manual save omitted `p` from the Rect at its default origin while
retaining `w`, `h`, an identity matrix and a redundant `<pts>` cache. Strict import
previously passed `None` to the coordinate parser and returned opaque
`IMPORT_FAILED`. Omitted Rect position now resolves to `(0,0,0)`, and the adapter
preserves the strict reader's already-sanitized, bounded reason. The identity matrix,
RGB layer color, modification counters, missing Tag on the new Rect, two-coordinate
project stock and active-layer metadata were not causes.

Third, the Pocket call omitted required `expected_revision`. Its Region target and
all supplied machining values were valid, and the rejected call was atomic at
revision 1. The agent incorrectly inferred that Pocket did not support Region even
though the description and implementation did. Missing required properties now
produce an actionable value-free message such as
`Missing required field: expected_revision`; MOP descriptions repeat the current
revision/fresh request-ID requirement. The transcript's `field:null` came from
`main`, because OpenCode had not been reloaded onto this branch. Commit `e2fd2c3`
already contained top-level field extraction; the staged change improves its message
but does not duplicate that fix. External acceptance must still restart/reconnect the
MCP process after code changes.

Finally, import plus `geometry_add_region` was insufficient to perform the user's
requested conversion because it could only leave the source Rects in place or rebuild
a different document. `geometry_replace_with_region` now performs that replacement
on a staged clone, accepts supported root Rect/Circle/closed-Pline contours on one
layer, validates Region topology before publication, removes sources atomically and
retargets compatible explicit Profile/Pocket selections. Engrave/Drill or unsupported
relationship cases reject without publication.

Focused evidence: the strict-import, document, MOP and Region-replacement suites ran
37 tests successfully with the existing Windows symlink-privilege skip. The exact
missing-revision regression confirms a fresh-ID retry at revision 1 creates a Pocket
targeting the Region and advances to revision 2. The complete suite ran 223 tests:
222 passed with the same single skip. A task-owned replay under
`output/transcript-regression-20260917/` imported the actual CamBam-saved file,
replaced both Rects, added the Pocket, exported 5,195 bytes and strictly re-imported
exactly one Region and one MOP; revisions were 0/1/2 and Region bounds remained
`[0,0,40,20]` with one hole. Named-client evidence remains pending, and no CamBam
toolpath claim follows from these automated document/contract checks.

## MCP OpenCode local-delivery and existing-Region audit — 2026-09-18

The second supplied session ran against exact commit `e2fd2c3`, without the staged
changes. It made 37 MCP calls: 36 succeeded and the final strict import correctly
rejected malformed, manually reconstructed XML. The adapter created six primitives,
configured stock and 3x2 IsoGrid nesting, added Drill/Pocket/Engrave/Profile MOPs,
and consistently inspected four MOPs. Its revision-11 export was 13,099 bytes with
SHA-256 `d8182f...`; the agent instead authored 6,177 different bytes with hash
`646e31...`. At revision 14, the authoritative export was 15,654 bytes with hash
`455b94...`; the agent authored 4,768 bytes with hash `f61889...`, omitting every
MOP, claimed success despite the mismatch, and later copied the stale 6,177-byte
file. This is an agent/file-transport failure, not lost MCP state or failed export.

The baseline already provided portable import/export, source hashes, document-list
recovery, optional read-only request IDs, required write revisions, Pocket targets
on Regions, and OpenAI-compatible schemas. Staged wording strengthens those existing
contracts; it must not be described as newly implementing them. Genuinely new staged
work is the omitted-origin Rect import fix, propagation of bounded reader details,
value-free actionable schema messages, and atomic contour-to-Region conversion.

One transcript request remained uncovered: “change the region shape.” The staged
conversion tool accepts Rect/Circle/closed-Pline sources, not an existing Region.
`geometry_update_region` now replaces an existing root Region's absolute contours
on a staged clone while retaining its UUID and project-owned layer/MOP relationships.
Invalid topology leaves the revision and original geometry unchanged. Descriptions
also state that Region holes are excluded islands for Pocket and that a Part without
MOPs produces no nested work.

For exact local delivery, same-host stdio clients may now call the existing atomic,
non-overwriting `document_save`, copy the generated server-workspace artifact with a
deterministic binary operation, and verify byte count/SHA-256. The current handle
remains authoritative until a manual edit changes the client file; reloading then
uses a fresh exact workspace staging copy and guarded `document_open`, as recorded
below. The server still accepts no arbitrary client path;
inline `document_export` remains the portable cross-host fallback. This consolidates
delivery around server-generated bytes rather than adding a second serializer or
trying to make a language model reproduce large XML exactly.

Verification on the repository Python 3.14 environment: the MCP suite ran 70 tests
successfully with one Windows symlink-privilege skip; the complete suite ran 225
tests successfully with the same skip. The focused strict-import suite, `compileall`
and `git diff --check` passed. Named OpenCode and CamBam acceptance remains pending.

## MCP OpenCode mutation sequencing and false-delivery audit — 2026-09-18

The next OpenCode rerun made 34 CamBam calls. It submitted the first six geometry
mutations concurrently with `expected_revision:0`. The rectangle committed revision
1; the other five correctly returned `STALE_REVISION` at revision 1 and made no
changes. OpenCode then retried them sequentially at revisions 1 through 5, and all
succeeded. This is correct optimistic-concurrency behavior, but the per-tool contract
did not make the non-parallel requirement salient enough. All same-document mutation
descriptions now receive one generated sequential-write suffix, while stale errors
state the current revision and require a still-applicable retry with a fresh request
ID. Concurrency remains valid for reads and independent documents.

Two later failures were also correct: `PartA_Pocket` collided with a Layer of that
name, and the deliberately attempted Engrave-on-Region target was unsupported. The
identifier failure now names the occupying entity type and requests a different
project-unique name; the Engrave rule was already explicit and was not broadened.

The final call was `document_export`, not `document_save`. It successfully returned
18,793 inline bytes and SHA-256 `d2aafa...`, created no file, and OpenCode nevertheless
claimed that `CamBam_Builder_Test.cb` existed in the shared workspace. Export results
now include `delivery=inline_content_only`, `file_created=false`, and an
`INLINE_ONLY_NO_FILE` diagnostic. The same-host file-producing path remains the
atomic, non-overwriting `document_save`; portable export still requires an explicit
client write and hash verification.

## MCP manual-save reload and stale-workspace audit — 2026-09-18

The latest OpenCode run created and saved the revision-10 document, reopened that
exact workspace artifact, added a Region and Pocket, then saved a 17,091-byte update
under the distinct workspace name `cambam_builder_test_region.cb` and copied it to
the client file. CamBam subsequently rewrote the client file as 18,594 bytes with
SHA-256 `e27a3983...`. OpenCode's text reconstruction delivered only 16,179 bytes to
`document_import`, which correctly rejected the mismatch. It then opened the older
workspace path `cambam_builder_test.cb` (12,911 bytes, SHA-256 `724138d4...`) without
a hash guard. Inspection of that valid but stale snapshot omitted the Region/Pocket,
leading the agent to incorrectly blame parser support and ask the user to identify
the change.

Direct strict reading of the attached CamBam-saved bytes succeeds. It reconstructs
three layers, five primitives including `Region_Test_PlateA`, two Parts, and five
MOPs; native `ModificationCount`, identity formatting, nesting items, holding-tab
containers and reordered MOP fields do not prevent import. The visible intentional
geometry change is a Region Y translation from bounds `12,8`–`40,30` to
`12,40.0624390837628`–`40,62.0624390837628`; much of the remaining XML difference
is CamBam normalization/native state. No reader relaxation is justified by this
evidence.

The owning failure was same-host ingress. `document_list` now exposes the already
trusted workspace root as `workspace_path`; guidance requires an exact binary copy
of the current client file under a fresh workspace-relative name followed by
`document_open(expected_sha256=...)`. Open checks the bounded byte snapshot before
strict parsing and returns `CONTENT_MISMATCH` without publishing a handle if an old
or wrong artifact is selected. Cross-host `document_import` remains available, but
its mismatch must not be bypassed by falling back to a similarly named workspace
file. Successful open/import intentionally produces a new revision-0 snapshot;
same-document mutation revisions remain sequential and unchanged.

Focused automated evidence covers matching and mismatching guarded opens, zero
publication on mismatch, source-hash reporting and workspace-path discovery. A fresh
OpenCode/CamBam run remains required to validate that the model follows exact staging,
compares the new snapshot with the prior live snapshot, applies the analogous change,
and saves the result back through an exact handoff.

Verification on the repository Python 3.14 environment: all 71 MCP tests pass with
the existing Windows symlink-privilege skip, and the full suite passes all 226 tests
with that same skip. `compileall` and `git diff --check` pass.

## MCP consuming-agent workflow audit — 2026-09-19

The next OpenCode run used the exact shared-filesystem staging path successfully:
after the user changed the client file, it hashed and binary-copied the file under a
fresh workspace name, opened it with `expected_sha256`, and received the correct new
revision-0 snapshot. This closes the stale-workspace regression from the prior run.

The reported `INVALID_ARGUMENT` failures were not missing revisions. Engrave/Profile
calls included `expected_revision` 8 or 9 but changed the opaque document handle from
`d6e8007c-c263-41ba-9c48-ab81f1ab641f:509263b6-829c-4e5f-9b97-1f6fc6b904df`
to the malformed `d6e8007c-c263-41ba-9c97-1f6fc6b904df`. Multiple retries repeated
the same transcription error, including one replay under the same request ID.
Schema validation correctly rejected every call before publication. The error and
Handle schema now direct clients to copy the complete opaque handle exactly from the
latest result or `document_list`.

The revision requirement remains deliberate optimistic concurrency. A mutation says
which state it was planned against; if another call advances that state, the stale
mutation is rejected rather than silently applying in a different order or overwriting
newer intent. The same run again launched three revision-0 geometry mutations in
parallel; one committed and two safely failed stale before sequential retries. Queueing
or silently rebasing these calls would hide a planning error and can corrupt dependent
edits, so the adapter retains the guard and the consumer template explicitly forbids
same-document mutation concurrency.

File delivery remained misleading. Two successful `document_save` calls created
verified workspace artifacts, but OpenCode described those internal paths as saved
files until the user explicitly requested the project directory. Saved artifacts now
state machine-readably that a workspace handoff exists and a client file does not.
The reusable project template makes durable project-local delivery part of every
create/edit task, defaults edits to the source path, and requires a pre-replacement
source-hash check plus destination-hash verification.

For the manual edit, the prior revision-10 inspection and new revision-0 inspection
shared stable primitive IDs. Their bounds were: Part A outline `[0,0,80,50]` to
`[0,50,80,100]`, circle `[15,20,25,30]` to `[15,70,25,80]`, and Region
`[45,10,72,40]` to `[45,60,72,90]`; Part B remained `[110,0,160,50]`. Thus the
demonstrated common edit was exactly `(dx=0, dy=+50)` with no rotation. The agent
instead inferred a 90-degree rotation about `(0,50)`, then translated Part B into an
unrequested alignment and saved under a new name. General server/template guidance
now requires baseline/current comparison by stable IDs, separation of CamBam
serialization noise, the smallest common demonstrated delta, and clarification when
the evidence is ambiguous.

The consuming project instructions supplied with the transcript also conflicted with
the adapter: they hard-coded a different workspace ID and prescribed portable
import/export despite a shared filesystem. The new packaged
`consumer_AGENTS.template.md` discovers runtime identity, bootstraps durable project
defaults once, then replaces its setup block with daily instructions. Project units,
paths, overwrite/Save-As preference and manufacturing sources stay local; handle,
revision, staging and delivery semantics remain adapter-owned.

An external OpenCode restart needed to load changed MCP schemas is acceptance setup,
not a reason to start a fresh development-agent conversation. The prior handoff
conflated those independent lifecycles. The repository development `AGENTS.md`
already scopes its breakpoint decision to the current work session and is unchanged.

### Consumer template first-run acceptance finding — 2026-09-19

The first test of the copied consumer template did not enter a recognizable onboarding
flow. Its instruction to ask "one concise, grouped set" was followed literally: the
agent emitted one overloaded free-text question and mixed durable project conventions
with the current document's filename, material, stock, tooling and machining values.
This was a template defect rather than an MCP call failure.

The setup section now uses a literal pending/completed gate. While pending, it requires
the agent to preserve the initiating task, inspect read-only, present each unresolved
durable policy as its own answerable prompt (continuing across dialogs if necessary),
and avoid all document-instance values. After answers, the agent must persist concrete
durable defaults and then resume the original request. A source-level contract test
guards the state markers, separate-question requirement, task/setup boundary and
resumption rule. Live client acceptance remains necessary because compliance with
repository prose cannot be guaranteed by the adapter runtime.

The next live run passed the first-run gate and completed the requested CamBam file,
but file-history questions overlapped: Save As appeared once against overwrite and
again against backup. These are one policy because an always-versioned Save As choice
is itself a preservation strategy and needs its naming convention at the same time.
The template now recommends versioned Save As and asks that combined decision once.

The final onboarding wording also allowed "validation" to be interpreted as a software
test-suite preference. Consumer validation now names only evidence available in the
workflow: MCP operation results and `document_inspect`, exact delivery hashes, and
optional human CamBam/native-toolpath review. The adapter does not expose this
repository's development suite or a CamBam toolpath engine. Separately, computational
scripts are again encouraged when they materially improve advanced geometry or layout
accuracy, but they may calculate inputs only; they must not bypass the adapter by
generating or patching `.cb` XML.

The supplied transcript confirms the mechanism. Shallow project context was enough,
but the agent recursively listed a nested CamBam-Builder checkout, offered repository
tests as its recommended artifact-validation option, and later ran `pytest -q` there.
That command failed collection with fourteen imports resolving against a different
checkout; it neither exercised nor invalidated the delivered `.cb`. In contrast, all
sixteen CamBam MCP calls succeeded, including inspection and workspace save, and the
project-local copy's SHA-256 matched the handoff.

The transcript also records the user selecting job-specific inputs as user-provided
per task. The agent nevertheless inserted an unrequested builder-fixture exception
into the completed `AGENTS.md` and created enabled MOPs with invented stock, material,
tool, spindle, feed and depth values. The consumer template now states that “test,”
“demo” or “example” is not permission to invent these values. A clearly labeled
non-production fixture proposal requires task-level confirmation and cannot silently
become durable project policy.

A follow-up found a circular implication in "review required before reporting
completion": a user cannot inspect a file that has not yet been delivered. The
consumer contract now separates agent completion, saved artifact handoff and human
acceptance. Its recommended loop is self-check, save/hash-verify, report ready for
review, then apply requested corrections. A configured human CamBam/toolpath check can
gate an accepted or production-ready label, but never gates delivery of the revision
that must be reviewed.

Verification on the repository Python 3.14 environment: all 71 MCP tests and all
226 repository tests pass with the existing single Windows symlink-privilege skip.
The wheel build succeeds and contains both the contract JSON and packaged consumer
template. Compileall and `git diff --check` pass. External OpenCode behavior remains
the required user acceptance boundary.

## MCP/CamBam API alignment correction — 2026-09-19

A final branch-to-main and CamBam documentation audit identified four merge blockers.
Valid native files could contain zero layer pen width, zero Part tool diameter,
zero/unspecified stock, Manual/PointList nesting, or any of eight grid traversal
orders that the MCP output schema rejected. Part configuration replaced omitted
fields and discarded native nesting placement metadata. Stock PMin/PMax offsets and
nonzero top surfaces were normalized away. Finally, common CamBam workflows exposed by
the core model were missing at the adapter edge: Text/V-cutter engraving, open-Pline
Profiles and Automatic holding tabs.

The comparison used CamBam's published [nesting](https://www.cambam.info/doc/1.0/cam/nesting.html),
[Profile](https://www.cambam.info/doc/1.0/cam/profile.html),
[holding-tab](https://www.cambam.info/doc/1.0/cam/holding-tabs.html) and
[text-engraving](https://www.cambam.info/doc/1.0/tutorials/text-engraving.html)
behavior as the external source of truth, then checked the repository's public API,
reader/writer and MCP schemas as one end-to-end contract.

The correction expands inspection fidelity without pretending to author unsupported
placement data. `machining_configure_part` remains limited to None/Grid/IsoGrid, but
is now a field patch and preserves imported Manual/PointList records and native-only
children. Stock offset/surface are modeled and unchanged native Stock XML is retained.
Open Profiles report `VertexOrderRelative`; Text is a valid Engrave target; VCutter
and bounded Automatic tab properties are explicit closed-schema inputs. Manual tab
points and Manual/PointList placement authoring remain excluded. A pickle migration
supplies safe defaults to pre-change Parts.

Regression coverage exercises native nesting variants and metadata, zero boundary
values, stock coordinates, minimal Part creation, omission-preserving updates,
open-Profile diagnostics, Text/V-cutter engraving, automatic tab boundary values,
invalid tab ranges, XML save/reopen and older Part state loading. The normative MCP
contract, server/tool guidance, README capability summary and 4e native acceptance
procedure now describe the same surface. Verification passed 75 MCP tests and 231
full-suite tests (one pre-existing Windows symlink privilege skip), compileall, schema
and 37-tool inventory validation, and `git diff --check`. The fresh wheel/sdist is in
`output/mcp-alignment-fixes-20260919-a/dist/`; wheel inspection found 26 entries and
confirmed the service, updated schema and consumer template with no build/output tree
leakage. CamBam Plus toolpath and property inspection remains a separate pending
acceptance; no production G-code claim is made.

### Native user-encoding correction after CamBam Plus 1.0 review — 2026-09-19

The user's eight-file CamBam Plus 1.0 review refined and partially rejected the first
alignment claim. Files 01, 02, 03, 05, 06 and 08 opened, and the observed Grid,
zero-value, open-side and tab behavior was usable. The review established that Part
stock offset is relative to Part machining origin: origin `(12,-7)` plus stock offset
`(12,-7)` places the stock lower-left top corner at drawing coordinate
`(24,-14,4.5)`. It also confirmed that Profile Inside is left and Outside right for an
open Pline relative to vertex traversal, and that a requested tab width of 5 displays
wider because CamBam compensates for cutter diameter.

Three validation assumptions were invalid. File 04 included invented vendor nodes and
an invalid `GCodeOrder`, so CamBam rejected it; the replacement uses a real Points
primitive, `NestMethod=PointList`, `PointListID`, and `GCodeOrder=Auto` only. File 06
set `UseLeadIns=true` while the Profile's lead-in type was None, making it inert. The
MCP contract now requires false until lead-in authoring exists and documents automatic
count, perimeter threshold, compensated width, target-depth-relative height and the
Square/Triangle/Skip distinction. File 07 serialized `Vcutter`, which CamBam displayed
as Unspecified. The official automation reference names the API enum
`ToolProfiles.VCutter`; MCP now accepts/emits `VCutter`, and the direct API normalizes
the common `Vcutter` and `V-Cutter` spellings. This remains ordinary path-following
Engrave, not skeleton or width/depth-varying V-carving.

Method-aware serialization now emits Grid fields only for Grid/IsoGrid, preserves an
unchanged valid imported PointList subtree across unrelated Part edits, and discards
old placement fields when switching methods. Inspection exposes a derived
`stock_drawing_origin` alongside the local encoded stock values. Regression coverage
also proves that MOPs in different Parts may reference the same primitive. All 75 MCP
tests and 233 full-suite tests pass with the existing one Windows privilege skip; all
eight corrected files in `output/1909/` strict-reopen and await the focused native
recheck described by their manifest.

### Corrected native encoding acceptance and final hardening — 2026-09-19

The user opened all eight corrected files in `output/1909/` with CamBam Plus 1.0.
Files 01–08 opened successfully. The reported observations accepted the relative
Part stock/origin placement, exact zero boundaries, shared-geometry Grid nesting,
all presented PointList variants, left-side open Profile behavior, every automatic
tab style, Text outline Engrave with `VCutter`, and native metadata preservation.
The stock check confirms the XY relationship—Part origin `(12,-7)` plus local stock
offset `(12,-7)` places its origin at `(24,-14)`—while the serialized/reopened source
retains stock surface `4.5`; no inference is made from the separately reported
rounded/displayed Z value.

The PointList observation provides behavior more precise than the vendor prose. The
official nesting documentation says Points define copy positions and identifies the
drawing or alternate machining origin as the nesting coordinate origin. The native
check further established that a point is a translation applied to the unnested
toolpath, not an absolute anchor replacing source geometry: moving source geometry by
`(3,2)` and the first point from `(12,7)` to `(15,9)` puts that source corner at
`(18,11)`. The normative contract now records this as user-validated CamBam Plus 1.0
behavior rather than attributing the full vector-addition rule to the documentation.

The final hardening closes two remaining misuse paths. Configuring more than one Grid
or IsoGrid copy with nonzero XY stock returns a non-blocking stock-fit advisory,
because Part nesting repeats toolpaths but does not enlarge the optional Part stock
definition. It is good-practice guidance, not a validity requirement. Fresh direct-core
`ProfileMop(tab_method="Manual")` export now raises before serialization: the previous
writer could emit scalar Manual settings without the required native position
collection. Imported native Manual MOP XML still follows the unchanged preserve path,
and the MCP schema continues to exclude Manual authoring. Exact Manual point
authoring is deferred pending a native CamBam A/B fixture; all documented automatic
tab parameters remain supported. `EndMill` and `VCutter` remain valid Engrave tool
profiles, and neither changes its path-following operation into area fill or V-carve
topology.

Verification after hardening: all 75 MCP tests pass with the existing Windows
symlink-privilege skip; all 234 repository tests pass with the same single skip.
Compileall for `cambam_builder`, `legacy_cambam_builder`, `tests` and `demos`, schema
JSON parsing and `git diff --check` all exit successfully. The expected negative-path
writer log confirms that unsupported fresh Manual tab authoring fails before XML is
returned. No repeat native CamBam check is required because these final changes do
not alter any accepted automatic-tab or nesting XML syntax.

### Text machining and roughing-clearance contract correction — 2026-09-19

The follow-up identified two adapter restrictions not present in CamBam Plus 1.0.
Text is a valid direct source for Profile and Pocket as well as Engrave: Pocket clears
closed letter interiors and Profile offsets letter outlines. These operations retain
the ordinary limitations of a round cutter at narrow strokes and corners. Profile
corner overcut wording now follows the vendor definition: it adds an extra machining
move into inside corners that otherwise remain uncut, deliberately overcutting stock
for fitted parts such as slot joints or inlays.

`RoughingClearance` was already a common core MOP field with reader, writer, native
state and inspection mappings for Profile, Pocket, Engrave and Drill. The MCP input
and closed inspection records nevertheless fixed it to zero. Profile, Pocket and
Engrave now accept any finite signed value. CamBam documents positive values as stock
left for finishing and negative values as overcut; Engrave at zero follows its source
line placement and a nonzero value offsets that path. Imported Drill values remain
inspectable, but fresh MCP Drill authoring stays at zero because this adapter exposes
only CannedCycle and CamBam's release documentation specifically associates drilling
roughing-clearance support with spiral drilling. This limitation was subsequently
removed by the SpiralMill authoring increment below.

The user's spiral-drill observation supplies the exact relationship absent from the
vendor property table. For explicit hole diameter `H`, signed radial clearance `R`
and effective tool diameter `T`, the cut diameter is `H - 2R` and must satisfy
`H - 2R > T` (equivalently `H > T + 2R`). With `H=6`, `T=4`, `R=-1`, the result is an
8 mm hole. Fresh direct-core SpiralMill CW/CCW serialization now enforces this when
both diameters are resolved. Imported native templates and Auto hole diameters remain
preserved/deferred to CamBam rather than being rejected from incomplete information.

Regression coverage authors Profile, Pocket and Engrave directly against one Text
primitive with positive and negative clearances, verifies the exact serialized values,
strict-reopens the file and compares all three parameter records. A separate native
core fixture proves that an imported CannedCycle Drill with nonzero clearance remains
inspectable. Stock diagnostics are absent for unspecified zero stock and advisory for
defined XY stock with multiple copies. All 77 MCP tests and all 237 repository tests
pass with the existing single Windows symlink-privilege skip; compileall, strict
duplicate-key/schema JSON parsing and `git diff --check` pass.

### SpiralMill Drill MCP authoring — 2026-09-19

The user supplied a CamBam Plus 1.0 native SpiralMill CW record and clarified native
state semantics. MCP Drill now supports both `SpiralMill_CW` and `SpiralMill_CCW` in
addition to `CannedCycle`. Spiral authoring covers explicit/Auto hole diameter,
signed roughing clearance, lead-out enable/length, flat-base behavior and native tool
profile metadata. Auto diameter is bounded to all-Circle target selections because a
Point carries no diameter; Point targets require an explicit value.

Fresh selected values serialize with `state="Value"`. Initial acceptance files kept
the CannedCycle-only `PeckDistance`, `RetractHeight` and `Dwell` records as Default,
but CamBam prompted to revert the cached RetractHeight 5 to the user's configured
3 mm default. Empty Default CustomScript caused the same class of prompt. Fresh
SpiralMill XML now omits all four irrelevant fields; untouched imported native XML
continues to round-trip exactly. Auto `HoleDiameter` still uses `Default` because it
is semantically active Auto sizing. Explicit diameters and the known world diameter
of each Auto Circle enforce the user-validated relationship `H - 2R > T`. The user
accepted the three original files' method, diameter/clearance and generated toolpath
behavior; file 02 also confirmed that a
lead-out length has no effect while DrillLeadOut is false. The contract now rejects
that inert combination and caps positive centerward length at effective hole radius.
After regeneration, the user reopened all three files and confirmed CamBam no longer
asked to revert or reconcile the omitted fields. This accepts the fresh SpiralMill
field-presence/state correction in CamBam Plus 1.0.

### Common MOP field semantics and encoding audit — 2026-09-19

The first backlog 8a family slice inventoried all 18 common fields modeled by
`Mop` and compared them with CamBam's MOP API, CAM Styles documentation, existing
native round-trip fixtures, and the accepted SpiralMill cache-conflict evidence.
Fresh-export decisions now live in one ordered `MOP_COMMON_FIELD_POLICIES` table.
Explicit and framework-context-resolved values serialize as `Value`; unset optional
depth/feed fields and empty custom header/footer fields are omitted. Imported
templates remain semantics-preserving for state, cached text, omission, and unknown
fields, with deliberate state edits handled by the existing setter.

Rejected interpretations: `None` does not authorize the framework to invent a
single-pass DepthIncrement or the undocumented `350 * abs(TargetDepth) + 6500`
CutFeedrate; those fallbacks were removed. An unmodeled field is not evidence of
user intent, so fresh `SpindleRange=0`, empty `StartPoint Default`, and Drill's
fixed `RoughingFinishing=Roughing` were also removed. This does not claim that
omission and `Default` are universally equivalent: a caller may deliberately
request `Default`, and imported native state remains authoritative.

Focused regression coverage applies the common policy to fresh Profile, Pocket,
Engrave, and Drill records, checks Part/project resolution, exact `Value` states,
omissions, and absence of unmodeled fixed records, and proves the old depth/feed
fallbacks are gone. Existing tests continue to cover imported `Default` cache
preservation, missing optional elements, deliberate state edits, nested containers,
and method-dependent Drill omission. Subtype fields, lead containers, tabs, and
method switches remain the next 8a slices; no new native CamBam acceptance is
claimed for this common-field-only increment.

Verification on 2026-09-19: the focused common-policy suite passed 9 tests; the
MCP authoring, MOP breadth, and native-edit suites passed 20 tests after their
inspection/fixture contracts were aligned with canonical empty-field omission.
The complete repository suite passed 241 tests with the one existing Windows
symlink-privilege skip. `compileall` for the modern package, legacy package, and
tests and `git diff --check` both exited successfully.

### Profile/Pocket subtype and nested encoding audit — 2026-09-20

The second backlog 8a slice compared the Profile/Pocket model with CamBam's
published Profile, holding-tab, MOP and CAM-style semantics, the accepted native
tab/open-Profile observations, and the repository's source-generated XML. The
previous imperative encoder had three unsupported interpretations: it mirrored the
modeled lead-in into an independent unmodeled LeadOutMove, invented TangentRadius
and LeadInFeedrate values, and serialized an unset final increment as cached
`Default` text `0.0`. It also left stale or incomplete modeled children when an
imported lead/tab discriminator changed.

Fresh subtype decisions now live in ordered `MOP_PROFILE_FIELD_POLICIES` and
`MOP_POCKET_FIELD_POLICIES` tables. Applicable user/product choices are explicit
`Value`; `FinalDepthIncrement=None` and mode-inapplicable nested children are
omitted. Explicit None lead/tab discriminators remain present to prevent CAM styles
from re-enabling them. Fresh leads are limited to the fully modeled None and Spiral
modes, and fresh output no longer creates a lead-out. Automatic tabs materialize
their complete modeled record; Manual points and unsupported lead modes remain
import-preserve-only. On supported imported mode switches, modeled siblings are
reconciled without deleting unknown native children or independent lead-out data.

Rejected alternatives: omission of the whole disabled lead/tab container would
cede the disabling decision to a CAM Style; retaining irrelevant Spiral/tab children
would preserve stale cached data in fresh files; and silently accepting Tangent or
Manual transitions would claim parameters/collections the model does not own.
Focused policy coverage passed 13 tests. The MCP MOP/authoring and core round-trip/
native-edit suites passed 25 tests. The full repository suite passed 245 tests with
the one existing Windows symlink-privilege skip; compileall and `git diff --check`
also passed. No new CamBam toolpath or production G-code acceptance is claimed.

### Engrave/Drill subtype and method encoding audit â€” 2026-09-20

The final modeled-field slice of backlog 8a compared Engrave and Drill against the
official operation manuals and MOP API, the accepted SpiralMill source-native/open
behavior, and existing round-trip evidence. Three gaps remained. Engrave still wrote
an unset FinalDepthIncrement as cached `Default` text `0.0`; Drill method fields were
split across imperative branches instead of an auditable inventory; and changing an
imported Drill method retained stale fields belonging to the old method. A nonempty
CustomScript could also leak into another fresh method, while a fresh empty
CustomScript operation silently encoded an incomplete request.

`MOP_ENGRAVE_FIELD_POLICIES` now owns all three Engrave subtype fields and
`MOP_DRILL_FIELD_POLICIES` owns the method plus its eight dependents. Applicable
user/product choices are `Value`; Engrave FinalDepthIncrement `None` is omitted;
SpiralMill Auto HoleDiameter is the sole policy-selected empty `Default`. CannedCycle,
SpiralMill and CustomScript dependents are mutually exclusive. A supported imported
method switch rebuilds only modeled dependents and preserves unknown extensions.
Unknown native methods remain round-trip-preserved but cannot be switched, and fresh
CustomScript requires nonempty text. Existing SpiralMill diameter and lead-out safety
checks now also run when relevant fields on an imported operation are edited.

The official Engrave manual says RoughingFinishing is effective only for Lathe and
3D Profile. Removing the public field would break compatibility without improving
interchange, so it remains explicit framework/MCP metadata but has no promised
Engrave path effect. The official Drill manual establishes controller-dependent dwell
units, the CannedCycle R-plane role, Auto Circle diameter, signed lead-out direction,
flat-base circle and CustomScript macros. Direct-core CustomScript support is retained,
but MCP authoring and structured inspection intentionally exclude this literal,
controller-sensitive G-code surface pending a separately justified capability.

No new vendor behavior is inferred from framework XML. Backlog 8c now requests three
source-native A/B checks: Engrave final increment/cut ordering, CannedCycle retract
state and behavior, and a nonempty CustomScript text edit. Reopen this encoding only
if those files conflict with the inventory or cause a reconciliation prompt.

Verification on 2026-09-20: the focused policy suite passed 18 tests; the adjacent
MCP MOP, core MOP round-trip and native-edit suites passed 14 tests; and all 250
repository tests passed with the existing Windows symlink-privilege skip. Compileall
and `git diff --check` also passed. No new CamBam toolpath or production G-code
acceptance is claimed.

### Unreleased compatibility and WIP-state boundary correction â€” 2026-09-20

The user clarified that regression coverage for “fresh/imported behavior” must not
freeze files or APIs produced by this unreleased framework. The supported exchange
boundary is CamBam `.cb` XML: fresh output should use the best current encoding, and
the reader should accept and preserve supported content from CamBam-saved files.
Imported-template and method-switch tests exercise that external XML boundary, not
framework-version compatibility.

The repository already stated this policy, but one later increment had contradicted
it by adding missing-field defaults for older Part pickles and a synthetic migration
test. Primitive pickle hooks also supplied defaults for four older geometry fields.
Those shims and the migration test are removed. Two unused `_v2` matrix aliases and
a regression preserving positional `add_part` ordering were likewise removed because
their only purpose was unreleased Python API history; `target_identifier` and
`place_last` are now keyword-only.

Pickle itself remains useful for one narrower purpose: a trusted same-code-version
WIP/cache snapshot can resume framework-only object relationships, including live
group targeting that CamBam XML deliberately materializes as a target snapshot. It
is unsafe for untrusted input, non-atomic, not used by MCP, and carries no migration
or exchange promise. A future release-grade state/exchange format may be considered
after a robust contract exists, without preserving pre-contract pickle layouts. Real
external compatibility requirements remain separate: CamBam's native file marker
and evidence-backed MCP protocol versions continue to serve actual applications.

Four focused state tests, three matrix tests, five MOP-context tests and the wider 34
persistence/geometry tests pass. The complete current suite passes 248 tests with the
existing Windows symlink-privilege skip. No manual CamBam validation adds evidence
because XML encoding and reader behavior did not change.

## Framework/MCP MOP parity audit - 2026-09-20

Backlog 8b compared the executable core field-policy inventories, all four MOP
dataclasses and reader/template preservation path with the closed MCP author-input and
inspection schemas, service mappings, target rules and negative-path regressions. The
resulting authoritative exposure matrix is in `MCP_CONTRACT.md`; an executable test
now requires every modeled dataclass field to have exactly one MCP authoring
disposition (input, pin or exclusion) and one inspection disposition (typed or
hidden), and checks those classifications against the schemas.

No MCP authoring promise exceeded the core writer. The bounded add tools intentionally
pin many direct-core choices, and CustomScript remains excluded because it is literal
controller/postprocessor-sensitive G-code without native execution acceptance. Live
target groups remain core-session intent that CamBam XML snapshots to explicit IDs;
Manual tab points, unsupported lead modes, unknown Drill methods and other unmodeled
native fields remain preserve-only. Unsupported MOP kinds are different: the core does
not round-trip them, and strict import rejects them. Arbitrary core target-kind linking
is only syntactic encoding, so it was not promoted as evidence that CamBam can machine
every combination.

The audit found two contract wording errors rather than writer defects. A later MCP
paragraph still described Drill as CannedCycle-only and roughing clearance as
universally pinned to zero. It now describes the implemented CannedCycle/SpiralMill
split and limits the zero-clearance rule to fresh MCP CannedCycle authoring; imported
native CannedCycle clearance remains core-readable, preserved and inspectable.

The largest useful gap is read-only: one alternate but modeled pin, inherited required
state, live/empty target source or out-of-slice target currently blanks the entire
structured `parameters` record. Backlog 8d therefore separates preservation-aware
inspection from the canonical authoring slice after 8c supplies the remaining native
state evidence. A second compositional defect is independent of native machining:
adding harmless framework group metadata to an explicit target makes both typed
geometry and MOP parameters disappear because target eligibility reuses the geometry
relationship predicate. Backlog 8e isolates that correction while retaining the
parent/child, transform and local-Z exclusions.

Generic parameter mutation was not promoted. The direct core currently relies on raw
attribute assignment with important validation deferred to export, so a safe MCP
patch tool first needs a validated atomic core patch contract and the 8c evidence for
ambiguous final-pass, retract and CustomScript states. MOP reorder also retains its
existing named-client reopening condition rather than being duplicated here. Nominal
parity for live groups, inherited effective values, CustomScript or unknown native
content was rejected because it would imply durability, style evaluation or execution
safety the project does not provide.

Verification on 2026-09-20: the five focused parity tests pass; the complete suite
passes all 253 tests with the existing single Windows symlink-privilege skip. All 83
MCP tests pass with that same skip. Compileall for the
modern package, legacy package, tests and demos and `git diff --check` both exit
successfully. This classification-only increment changes no CamBam XML or toolpath
behavior, so manual CamBam validation would add no evidence.

## Native Default, Value and omission evidence - 2026-09-20

The user supplied four CamBam-native Profile files under
`output/state-fixtures-01/`. `no-mop-values-set.cb` is a fresh operation whose
stateful properties are present as Default with the local CamBam profile's cached
text. `one-mop-value-set.cb` changes only TargetDepth to `Value=-10`; all other MOP
states remain Default. The user confirmed the cross-installation behavior: a present
Default whose cached text differs from the opening installation's resolved default
causes CamBam to ask whether it should retain or update the file value, whereas an
omitted property is populated from the opening installation without that prompt.

`missing-entries.cb` manually removes 21 Profile top-level/container records from the
second file. It opens successfully. CamBam's native resave adds only the complete
Default `HoldingTabs` container, continues omitting the other 20 removed records, and
removes the still-present empty Default `StartPoint`, `CustomMOPHeader` and
`CustomMOPFooter`. The MOP child counts are 36, 15 and 13 for the complete, edited and
resaved records respectively (including primitive and Name). Thus omission is a
prompt-free resolution mechanism but resave materialization is property-specific;
“CamBam writes all missing defaults on save” is rejected.

All four files pass the framework's strict byte reader. The two omitted variants also
show why raw model fields cannot stand in for effective style values: absent
DepthIncrement is represented as `None`, while other absent fields expose constructor
fallbacks such as Profile side Inside, clearance 15 and ExactStop despite those values
not occurring in the source XML. The template preserves the absence correctly, but
8d must expose absence/state separately before returning such values as structured
inspection.

Fixture hashes (SHA-256): `no-mop-values-set.cb`
`67402ef03f58fca7fe29b6d2cabdeee13b5e78bc4c2daa9f2dafec28ea47513f`;
`one-mop-value-set.cb`
`624804b66106b4b97d3a8290e3baa31afce9c35a701f89cd6f8fa9474aca5ddd`;
`missing-entries.cb`
`2801fb865c02116aecd3c62d6cd9b8197916b44e82f5a24e9014e8a2d2e97389`;
`missing-entries-resaved.cb`
`847ac67e3e981ccb0d3e7cf482bbba94e7c7a46ce3e8b3b67a8a24dba3ae62e3`.

This evidence removes the need for separate files devoted only to proving the generic
Default/Value transition. The remaining operation-specific questions were resolved by
the second set below.

### Engrave and Drill state/effect fixtures

The user supplied one CamBam-native baseline and four variants under
`output/state-fixtures-02/`, together with CamBam-generated `.nc` for every file.
CamBam Plus 1.0 is the fixed project validation baseline and need not be requested
again. The user reported CAM style `standard-mm`; no style record appears in the MOP
XML. The generated G-code headers say `Post processor: Default`, which is sufficient
context for interpreting this evidence—no explicit postprocessor selection was made.

Static XML comparison shows that `01` differs in MOP semantics only by
`FinalDepthIncrement state="Default">0` becoming `state="Value">0`; its G-code body
after the changing file/date header is byte-for-byte identical to the baseline. Both
cut target `(0,0)` through depths `-0.5,-1,-1.5,-2`, then target `(70,50)` through the
same depths. Thus explicit zero disables a distinct final-depth pass just as the
baseline's cached zero did, while making that choice installation-independent.

`02` changes only Engrave CutOrdering from Default cached `DepthFirst` to Value
`LevelFirst`, apart from the document name and CamBam-maintained ModificationCount
values. Its plunge sequence is `(A,-0.5), (B,-0.5), (B,-1), (A,-1), (A,-1.5),
(B,-1.5), (B,-2), (A,-2)`: levels are traversed across both targets in a serpentine
order that avoids an unnecessary return between adjacent levels. This contrasts with
the baseline's four depths on A followed by four depths on B.

`03` changes only Canned_State_Test RetractHeight from Default cached `5` to Value
`1`, apart from name/counters. The generated cycle line changes from
`G81 X60.0 Y40.0 Z-2.0 R5.0 F300.0` to
`G81 X60.0 Y40.0 Z-2.0 R1.0 F300.0`, directly confirming that RetractHeight is the
work-plane R coordinate. `04` changes only the Value CustomScript text from fixture-a
to fixture-b, apart from name/counters; its emitted line changes correspondingly from
`(fixture-a x=60 y=40 z=-5)` to `(fixture-b x=60 y=40 z=-5)`, confirming literal text
preservation and `$x/$y/$z` expansion for the selected point.

All five `.cb` files pass `read_cambam_bytes(..., strict=True)`. Reopen prompt status
cannot be recovered from static XML/G-code and is not claimed. It does not block this
increment: the first fixture set already established prompt semantics, while every
second-set changed decision is a native explicit Value. No further user observation
or fixture is required for 8c.

Second-set fixture SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `00-baseline.cb` | `e1feb5869f9658f7ef4ebc56d91aca09399087a684690601eca219d19a7db699` |
| `00-baseline.nc` | `018b175933bbfb841a4d3670de8a205d5afaa1bdb56121ce7267673400cb8f10` |
| `01-engrave-final-zero.cb` | `c270f177e0e3504d32d35a1a6cfbeb6bbd12a633a69db2fd8f34a509a37774ff` |
| `01-engrave-final-zero.nc` | `0ea487c5607cf0eb23cce9add1a871fb3f640ada507d5b03a141ae68c8550b12` |
| `02-engrave-level-first.cb` | `5dfb2f3df3cb42a44f8bd55051466c4925b1145a66ccce2587c0c00fc4ec8345` |
| `02-engrave-level-first.nc` | `08bdfea225986cc926ce72a99a5d0bdd82251017af066c63f42f9ba43438f0b5` |
| `03-canned-retract-one.cb` | `60c2f9912f08894386fcf8e259a24713ef6fb03c3c80aa4ab56fd4b9b7fae1d7` |
| `03-canned-retract-one.nc` | `25c22cba1844dbb7a86077862869468b5a75ada0d24f57c6c4b10b53e31546b2` |
| `04-custom-script-edit.cb` | `1bc3cc7900df2f2ed8f0abd16981c2227811d3753117c4cd745cd659041511b2` |
| `04-custom-script-edit.nc` | `8f8f8193d528b4cfc9d532a119ce456b48cc31fc151d14f66bc9d73a0a676db5` |

## Dimensional milling formula kernel - 2026-09-21

Backlog 9a is implemented as a pure library module, separate from CamBam documents,
MOP defaults and the MCP adapter. The formula conventions follow Sandvik Coromant's
metric and inch milling references: table feed is chip load times RPM times effective
cutting edges; cutting speed uses effective cutter diameter; rectangular MRR uses
axial depth, radial engagement and table feed; specific cutting force estimates net
cutting power; and power plus RPM estimates cutter torque. The tests independently
compare Kennametal's published thread-milling examples: 150 m/min with a 20 mm cutter
rounds to 2387 RPM and 0.1 mm/tooth at one effective edge gives 238.7 mm/min; 500
ft/min with a 0.79 inch cutter rounds to 2418 RPM and 0.004 in/tooth gives about 9.67
in/min. Exact calculations use `math.pi`, rather than the example's display rounding.

One generic “drawing units” formula was rejected because it would hide the factor
between mm and metres or inches and feet. The result therefore names every unit,
including the source conventions of cm³/min plus kW/N m for metric and in³/min plus
hp/lbf ft for imperial. A single depth field was also rejected: axial depth and radial
engagement have different physical roles, while target depth belongs to later pass
planning. Chip-thinning and circular-interpolation multipliers are absent rather than
being implicit assumptions.

The solver retains caller inputs exactly in an immutable requested-value record and
separately returns achieved values. It rejects inconsistent fully specified equations,
nonfinite/nonpositive values, nonintegral effective flute counts and radial engagement
beyond a known cutter diameter. Underdetermined groups return stable missing-input
requirements. RPM and feed caps affect only derived machine settings, record both the
uncapped and applied value, and propagate through achieved chip load, MRR, power and
torque. A supplied RPM/feed that exceeds its cap is an explicit conflict; silently
overwriting a user-fixed machine setting was rejected.

This establishes arithmetic consistency, not a safe cutting recommendation. The
specific cutting force and every operating target remain caller-supplied. Tool-maker
guidance, material/tool provenance, machine power/torque behavior, rigidity,
workholding, chip evacuation and supervised test cuts remain required. Recommendation
profiles belong to 9b; pass planning and optional read-only MCP exposure belong to 9c.

Verification: all 19 focused formula/solver tests and all 275 repository tests pass
with the existing Windows symlink-privilege skip. The tests cover every algebraic
inverse in both unit systems, metric/imperial physical equivalence, near-zero relative
conflicts, overflow/underflow boundaries, exact fixed-value retention and cap
propagation. Compileall for the modern package, legacy package, tests and demos and
`git diff --check` both exit successfully. The code has no XML, document or toolpath
behavior, so manual CamBam validation would not add evidence.

## Milling recommendation profiles and extension API - 2026-09-21

Backlog 9b is implemented as a pure public library layer over the 9a arithmetic
kernel. The selected boundary uses immutable tool, material and machine capability
records to identify one recommendation context, but keeps source cutting data in
caller-owned strategies. No generic material label or bundled vendor table was added:
compatibility depends on the exact tool family, material condition, operation and
diameter, and this repository has no licensed, versioned catalog owner.

Every emitted recommendation retains its exact units, a numeric applicable input
range, and provenance type, source, reference and version/date. User-supplied chip-
load and surface-speed tables bind to explicit tool/material/operation identifiers,
linearly interpolate only within their cutter-diameter range and report missing data
rather than extrapolating. Static profiles and pure callable adapters cover shop rules
and custom formulas. Strategy order does not silently settle disagreement: a fixed
user override wins in either order, identical values coalesce, and differing
non-fixed or fixed values fail as conflicts.

Entry feeds deliberately did not gain a universal cut-feed multiplier. Plunge, ramp
and helical suggestions require a matching capability on the tool plus their own
rule, provenance and range. The extension boundary can validate a callable's result
type and metadata, while purity itself remains a caller obligation; the immutable
context/result contract gives the callable no document handle to mutate. Machine
power and torque are recorded capabilities but are not silently converted into a
derating model. Profile serialization was also rejected for this increment because
no persistent owner or migration policy exists.

The focused tests cover immutability and unit consistency, provenance/range
requirements, table matching/interpolation/out-of-range behavior, fixed precedence,
conflicts, custom-callable validation, entry gating, solver composition, invalid
profiles and unsupported units. This pure nonserialized layer has no CamBam XML or
toolpath effect, so manual CamBam validation adds no evidence. Pass planning and any
read-only MCP presentation remain separate 9c work.

Verification: all 11 focused recommendation tests, all 30 combined 9a/9b tests and
all 286 repository tests pass with the existing Windows symlink-privilege skip.
Compileall covers the modern package, legacy package, tests and demos, and
`git diff --check` passes.

## Milling pass planning and bounded MCP exposure - 2026-09-21

Backlog 9c adds a pure public planning layer rather than coupling recommendation
profiles to CamBam documents. `plan_depth_passes()` now owns the previously
service-local through-cut arithmetic. Its exact-pass and safe-maximum modes preserve
the existing upward-rounding, final-stock engagement and advisory behavior. The
existing `machining_calculate_depth_increment` MCP tool delegates to that owner and
retains its schema and wire output, so the useful named-client workflow remains
read-only and document-free.

`plan_milling()` composes immutable 9b strategy output with the 9a constraint solver.
Explicit caller values supersede related non-fixed surface-speed/chip-load targets;
fixed settings above machine caps still fail rather than being silently changed.
Derived RPM/feed caps propagate to achieved surface speed, chip load, MRR, power and
torque. For through-cuts, sourced `axial_depth` remains visible as the safe maximum,
while balanced `depth_increment` is the actual axial engagement used in load
diagnostics. Radial engagement is returned both as a physical stepover and cutter-
diameter fraction. Declared power/torque excess produces diagnostics instead of an
invented derating model, and absent safe axial data returns a missing requirement.

A new closed MCP schema for arbitrary profiles was rejected: the strategies are a
direct-Python extension boundary with caller-owned provenance and no persistence or
catalog contract, so serializing them now would create a competing weaker profile
model. No planner accepts a document or authors a MOP. The returned safety notice
requires exact tool-maker guidance, machine-limit and workholding review, rigidity/
chip-evacuation checks and supervised test cuts; production safety is not claimed.

Verification: seven focused 9c tests cover immutability, existing pass arithmetic,
metric/imperial equivalence, invalid and unknown inputs, provenance retention,
fixed-value precedence, RPM/feed caps, power/torque diagnostics and actual-stepdown
load propagation. The 46 focused 9a/9b/9c/MCP-MOP tests and all 293 repository tests
pass with the existing Windows symlink-privilege skip. Compileall, strict schema JSON
parsing and `git diff --check` pass. No CamBam XML or toolpath behavior changed, so
manual native validation would add no evidence.

## Entity module boundary assessment - 2026-09-21

The requested post-9c maintainability review found a real mixed-owner problem but not
a reason to fold Region into the existing entity file. `cambam_entities.py` is 2,488
lines: shared numeric/vertex/bounds helpers occupy roughly lines 44-257; the common
entity and Layer/Part/Primitive bases lines 262-604; six ordinary concrete primitives
lines 609-1740; and MOP policy/model code lines 1745-2488. `region.py` is 838 lines,
with topology, intersections and contour handling occupying roughly lines 35-580,
the `Region(Primitive)` implementation lines 584-763 and typed XML parsing through
line 838. Region therefore has a coherent specialist reason to be separate even
though it belongs to the same inheritance tree.

The preferred design is a stable `cambam_entities.py` re-export facade over three
coarse implementation owners: `entity_core.py` for shared values and the
`CamBamEntity`/`Primitive` bases, `cad_entities.py` for Layer and ordinary primitives,
and `cam_entities.py` for Part and MOP models/policies. `region.py` remains a fourth,
specialized CAD owner and imports downward from core/CAD rather than through the
facade. This keeps the base inheritance chain together, gives callers one discovery
surface, preserves canonical class identity and yields a simple dependency direction:
transformations to core to CAD to Region, with CAM depending separately on core.

Three alternatives were compared. A strict shared/CAD/CAM split that physically
merges Region ranks second: its taxonomy is simple, but the CAD file would immediately
approach 2,000 lines and mix routine shape behavior with a large topology engine and
Region parser. Per-shape modules were rejected as needless navigation and import
surface. Separating data models, geometry algorithms and XML codecs ranks last for
now because each entity's behavior would span several files and Region parsing would
create callback or circular-dependency pressure. The specialist Region module plus a
facade provides the logical grouping without the physical merge.

Fourteen test modules, one demo and the project/reader/writer/transfer/MCP runtime
currently import `cambam_entities`; the project, reader, MCP service and two tests also
import Region directly. Region currently depends on `BoundingBox`, `Pline`,
`Primitive`, `Vertex` and the bulge tolerance from `cambam_entities`, with no reverse
import. A facade must therefore be introduced only after Region imports the extracted
implementation modules; importing Region back from the current monolith first would
create a cycle. Import-order and class-identity regressions are required before moves.

There is limited real DRY cleanup. Both entity files duplicate the same finite-float
validation and should converge on one core implementation. Their affine and similarity
helpers only look duplicated: Region requires a nonsingular transform and uses
topology-specific tolerances, while ordinary primitive bounds accept a broader affine
contract. Those semantics need characterization before any consolidation.
`cad_common.py` has no callers, types, constants or tests. Its sole executable behavior
is `logging.basicConfig`, which would mutate application-wide logging if imported.
It should be deleted in the refactor rather than reused as an unowned utility bucket.
No runtime file is changed by this assessment; priority remains 9b, then 9c, then the
new module-boundary backlog item.

### Entity module boundary implementation - 2026-09-21

Implemented the assessed coarse split without changing entity behavior. The former
2,488-line `cambam_entities.py` owner is now a 69-line explicit facade. Canonical
definitions reside in `entity_core.py` (480 lines), `cad_entities.py` (1,184 lines),
`region.py` (839 lines), and `cam_entities.py` (850 lines). Project, reader, writer,
transfer, package and MCP service imports now name those owners directly. Region
imports `Primitive`/shared values from core and `Pline` from CAD; no implementation
module imports the facade. `cad_common.py` and its import-time `logging.basicConfig`
side effect were removed. The similar Region affine/similarity validators were left
separate because their topology-specific nonsingularity and tolerance rules differ.

`tests/test_entity_module_boundaries.py` starts clean subprocesses with five varied
module orders, checks facade/owner identity for every entity family, and asserts that
Region directly inherits the canonical `Primitive`. Focused boundary, Region, MOP,
copy/transfer and same-code-version pickle checks passed. The full suite passed 305
tests with one expected Windows symlink-privilege skip in 44.716 seconds;
`compileall` passed. Existing XML round-trip, transform, reader/writer map, clone,
copy/transfer and state-snapshot regressions ran within that suite. No serialized
contract or native CamBam behavior changed, so manual native validation would add no
evidence.

### Merge-readiness verification correction - 2026-09-21

The first post-commit merge review found 60 trailing-whitespace errors in the exact
`main...HEAD` diff: 57 in the newly extracted `cad_entities.py` and three in
`entity_core.py`. Earlier `git diff --check` evidence had inspected only tracked
working-tree changes while those new modules were still untracked, so it could not
see their whitespace. Behavior remained sound: the committed tree compiled, passed
the import/construct smoke, and passed all 305 tests with the expected Windows
symlink-privilege skip.

The whitespace was removed without semantic edits. `AGENTS.md`, `WORKFLOW.md`, and
the development runbook now distinguish ready-to-commit from merge-ready, require
explicit inspection of intended untracked text, and require clean committed-branch
checks against the named base before any merge-ready declaration. Reopen this process
issue if another readiness claim omits final-tree evidence or if the documented Git
commands miss another concrete change state.

Post-correction verification: `git diff --check main` passed for the combined
committed-plus-working-tree result; the Python cleanup has an empty diff from `HEAD`
when end-of-line whitespace is ignored. Compileall, import/construct smoke and all
three entity-boundary tests pass. The immediately preceding committed-tree run passed
all 305 tests with the expected single skip; the correction changed no Python tokens.

## MCP 4e acceptance completion - 2026-09-21

OpenCode 1.18.31 was run from the isolated ignored directory
`output/mcp-4e-acceptance-20260921/`, with project configuration, default plugins and
auto-update disabled. Its local MCP definition launched the repository environment
against a separate task-owned server workspace. `opencode mcp list --pure` reported
`cambam connected`, proving the current named client can launch and negotiate with the
real adapter without changing normal OpenCode configuration.

The user clarified that the intentionally enabled OpenRouter model set is
GLM-5.3-Flash, DeepSeek V4.1 Flash and GPT-5.6 Luna; `glm-latest`, Sol and Luna Pro
are intentionally outside that workspace policy. Initial runs appeared to show all
three enabled models stopping or losing the task after `document_list`. Exporting
only those task-owned sessions found that each stored user message was exactly 705
characters and contained only the file-delivery paragraph: the PowerShell/native-CLI
harness had split the multiline argument, so the creation, geometry, Profile and
completion instructions never reached any model. Those runs are invalid acceptance
evidence and do not establish model or adapter failures.

The corrected harness joined the complete prompt into one native-process argument
and reran GLM-5.3-Flash in a fresh client/server workspace. Without a follow-up prompt
or manual MCP repair, it called `document_list`, created revision 0, added the Rect at
revision 1, added and inspected the Outside Profile at revision 2, saved and
hash-copied A, staged/hash-guarded/reopened A at revision 0, translated and inspected
B at revision 1, saved and hash-copied B, staged/hash-guarded/reopened B at revision
0, then closed all three handles. There were no failed calls, retries, permission
denials or argument corrections. It reported the expected final corners and stable
target UUID. Its one non-operational report error was calling 42 the CamBam MCP tool
count; deterministic `tool_definitions()` returns 37, so it apparently counted local
client tools too. The runbook now asks explicitly for the CamBam count excluding
ordinary client-local tools.

Exporting the successful task-owned session independently confirmed one 2,111-character
user message containing the create, Profile-value and complete-before-reply clauses.

The GLM-created client artifacts are
`output/mcp-4e-acceptance-20260921/client-glm53-complete/A.cb` (3,131 bytes,
SHA-256 `673cd5446ea3938d2288ef4e577b15884f6605e8b23439eff33b4dd0df5c11a2`)
and `B.cb` (3,136 bytes,
SHA-256 `40a1f6237cf7ca1a4ed1cabec551d7d21f1a263049c9472a556ccb6980794eca`).
The maintained acceptance verifier independently passed strict import, identities,
targets, explicit Profile values and exact A/B world geometry. This completes the
named-agent gate.

The reusable development choice is now owned by the OpenCode setup section of the
development runbook: task-owned config/state/cache and client/server directories,
existing provider authentication reused without copying it, a model selected from
the current workspace policy rather than a universal alias, a single native-process
prompt argument on the observed Windows/OpenCode 1.18.31 path, no broad `--auto`, and
task-scoped transcript export when prompt delivery is in doubt. These are bounded
operational safeguards, not requirements for other shells, clients, providers or
future OpenCode versions; retain them only while they prevent a reproduced failure.

Manual native acceptance was prepared independently so it does not depend on that
agent failure. `demos/mcp_authoring_slice.py` generated fresh ignored fixtures under
`output/mcp-authoring-demo-1015e836bb20/`. The maintained verifier accepted strict
import, stable identities and targets, every explicit Profile value, A corners
`(0,0,0)` through `(20,10,0)`, and B corners `(5,2,0)` through `(25,12,0)`.
SHA-256 is `36ba98ba60a2fac96d27c65195b775015f311c86960a61d67e18c6be170f8eb3`
for A and `f6457b0a8216f1f79db453f2a7fd83a7691efe9aba2d0f0d4afed5e797eaf79f`
for B. All 86 focused MCP tests passed with the existing single Windows
symlink-privilege skip.

The user accepted both files in CamBam Plus 1.0 on 2026-09-21. Interpreted
millimetres, A/B display geometry, the explicit Profile properties and outside
two-level toolpaths at `-0.5` and `-1.0` all passed; no G-code was generated. They
also clarified the native display contract: CamBam does not support user names for
primitives and displays a native type plus primitive ID such as
`PolyRectangle (1)`. The framework's stable `outline` identifier is present in the
primitive `Tag` metadata as designed. Layers, Parts and MOPs do support native names.
The 4e runbook now states that distinction instead of requiring a nonexistent native
primitive name. Together with the independently verified named-agent run above, this
completes same-machine local-stdio 4e acceptance. It does not establish remote
transport or broader production machining acceptance.


## Cumulative directional stock/rest bounds - 2026-09-22

`compose_sweep_bounds` extends the detached analytic consumer to finite supplied
horizontal passes in one exact rectangular stock/target. Each pass is revalidated;
stock/frame/Z mismatches and altered certificates fail before returning a result.
Ordered sources retain individual radius and position uncertainty, including
repeated passes. Exact union membership and opposite-direction stock subtraction
preserve the existing conditional evidence level without inventing connecting paths.

Aggregate areas use rational stock-grid cell classification, not sums of capsule
areas. Convex-corner coverage proves lower cells; exact minimum segment-to-cell
distance admits upper cells. Each cell is counted once, giving overlap-safe bounds
and monotonic endpoints for stock prefixes on the same grid. The tradeoff is coarse
area enclosures and O(grid_size squared times pass count) arithmetic. Reopen area
precision/performance only for a concrete consumer tolerance or blocking workload;
this increment claims neither exact union areas nor production machining acceptance.

Verification on the existing `.venv\Scripts\python.exe` (Python 3.14.5):

- `-m unittest discover -s tests -p test_stock.py -v`: 11 tests passed.
- `-m unittest discover -s tests -v`: 347 tests, OK with 12 skips for optional
  capabilities (including absent planar backend and Windows symlink privilege).
  This is not new optional-backend acceptance or a supported-version matrix run.
- Runtime import succeeded; `git diff --check` passed.

Independent scalar references cover collinear capsule overlap, disjoint disks,
identical passes and an overlapping disk lens. Regression checks also cover
prefix monotonicity, order/duplicate invariance, empty composition, nested grid
refinement, per-pass uncertain actual unions, zero-dimensional guarantees,
protected-boundary contact and invalid/altered source rejection. The first focused
run exposed test methods accidentally moved into the new class; restoring their
original class fixed the test-layout error. The first broad run reported OK but
PowerShell stderr redirection returned status 1; explicit subprocess log/exit
capture was used to resolve the wrapper ambiguity. Verbose output remains under
`output/stock-composition-20260922-a/`, not a second status document.

No manual CamBam acceptance adds evidence for these detached, nonserialized set
expressions. Physical uncertainty and complete supplied-pass coverage remain caller
assumptions. Implementation is uncommitted; target islands and generated cleanup
remain outside this completed scope. Next priority and reopening criteria are in
[PROGRESS](PROGRESS.md#next-detached-stockrest-increment).

## RC01 standalone generated sequence - 2026-09-23

The accepted nominal [RC01 request](REST_MACHINING_PLAN.md#rc01-inputs-and-process-bounds)
now generates ordered T1 roughing and T2 corner cleanup in
`cambam_builder.cam_core.rc01`.
The verifier checks every move's continuous planar sweep
and occupied Z interval against original protected geometry, prior guaranteed
removal, tool components and process bounds. T2's four access columns have actual
T1 predecessors. Rough and final stock prefixes are retained separately.

The independent rational 0.001 mm Y-strip oracle reports, at each of the three
depth slabs, T1 rest [7.775010615955999, 7.787678472024001] mm² and final rest
[0.9214411294439999, 0.9263453527720001] mm². Their volume bounds are
[23.325031847867997, 23.363035416072] and
[2.7643233883319995, 2.7790360583160005] mm³. These are above the ideal
finite-tool references 7.7256661177 and 0.8584073464 mm² and meet the accepted
+0.5 mm² / +1.5 mm³ budgets. Result status is partial target completion.
The area interval's arithmetic uses integer nanometres and directed integer
square roots, independently of the path generator and GEOS. The separate
0.05 mm residual-location check uses GEOS polygons at <0.000001 mm circle
sagitta; GEOS floating topology is not a formal numerical interval proof.
This is the remaining strict-S evidence limit, to revisit if native comparison
or a consumer requires a formal location certificate.

Verification with `.venv\Scripts\python.exe` (Python 3.13.5, declared Shapely
2.1.2 planar extra): `-m unittest discover -s tests -p test_rc01.py -v`
passed 6 tests; `-m unittest discover -s tests -p test_stock.py -v` passed 20;
`-m compileall -q cambam_builder` and `git diff --check` passed. Negative
variants reject a missing lower-layer strip despite complete upper layers,
missing rough predecessor, stale or uncertain inputs, continuous island
crossing, a low rapid, a shortened cutter or lowered holder and a floor overrun.
The new source/test files were inspected separately because `git diff --check`
does not include untracked files. No native `.cb` attachment, posted-motion
comparison or physical validation was performed. The next [I/E/N gates](REST_MACHINING_PLAN.md#rc01-standalone-and-cambam-output-gates)
need A/B/C files and actual CamBam motion evidence before native acceptance.

The 2026-09-23 modular boundary follow-up moved this standalone implementation
to `cambam_builder.cam_core.rc01`. The former root-level `rc01.py` was removed;
there was no released API or identified caller requiring a compatibility path.
Its direct dependency remains the detached `stock` owner;
the implementation has no direct native entity, XML or MCP import. The explicit
setuptools package list now includes `cambam_builder.cam_core`. A wheel and sdist
built with `uv build --out-dir output/cam-core-direct-mudxdu8y/` include
`cam_core/rc01.py` and exclude the root `rc01.py`. The revised wheel was
installed in an isolated Python 3.13 environment under
`output/cam-core-boundary-mudww54n/`; a `python -I` probe from outside the
source root located `cam_core.rc01` in that environment's `site-packages`,
found no root RC01 module and generated 2945 motion items. Optional Shapely
was absent from the isolated
environment, showing generation does not eagerly require the residual backend.
This is package-boundary evidence, not a new CamBam or production acceptance.

## RC01 native input and comparison preparation - 2026-09-23

The user's organization correction is recorded in the
[package organization plan](structure_spec.md#package-organization-decision-and-migration-plan):
native CamBam document entities, detached reusable CAM core, optional extended
strategies and system adapters need visible owners. In this increment the new
RC01 `.cb` bridge and post reader were moved from the package root to
`cambam_builder/integrations/cambam/`; `cam_core.rc01` remains pure. The root
`region.py` is explicitly identified as a native CamBam shape implementation
and has a staged move to `native/` after the first RC01 output trial. Existing
root ownership remains authoritative until then; no broad import rewrite was
mixed into the native-output experiment.

The adapter created `source.cb`, a separate explicit `setup.json`, A/B/C
candidate `.cb` files and `comparison.json` under the ignored
`output/rc01-native-20260923-1029/` directory. It strict-reimported the source
and each candidate, normalized to the accepted `Job()` fingerprint
`1d7c3e93e8b45cd8df9ff31a12ff120e413043156bbc42fe9494caa19295b2c6`,
then regenerated the 2945-item program with motion fingerprint
`39fde4a4c04295d50c7e53875478445cd5881eb9784f0c00d8ef676c4d89ffbd`.
A/B/C have 5/8/9 total MOPs respectively; the first two source Pocket MOPs
stay disabled, with preserved identity and target references. Enabled counts
are 3/6/7. T1 has 79 candidate level-cut paths per depth; B adds 248 T2
paths per depth; C adds four closed native Pocket windows. The source and all
three candidate SHA-256 values were independently checked against the manifest.
The independent rough and final area references are the three-slab S intervals
above. No CamBam-generated toolpath or posted `.nc` has been supplied or
accepted. Candidate Engraves omit explicit entry/link/retract/setup roles;
actual posted motion must resolve that representation before E/N passes.

Checks with the declared Python interpreter: `test_rc01_native.py` 3 tests,
`test_rc01.py` 6, `test_mop_roundtrip.py` 5 and `test_region.py` 22 passed;
`compileall` and `git diff --check` passed. The focused native tests cover
XML reopen, cosmetic identity edits, changed/inherited fields, unsupported
geometry/tool changes, preserved source identities/targets in A/B/C,
candidate hash invalidation, and posted-reader fail-closed behavior. `uv build`
produced a wheel and sdist with both integration modules and no root RC01
adapter files. A `python -S` import from outside the source root located both
modules inside the built wheel and generated the expected motion fingerprint;
the current environment's NumPy was supplied on the search path. This is a
wheel-content/import check, not a separate clean dependency install.

The next high-value increment is the CamBam Plus 1.0 output trial for these
three artifacts, with returned posted motion and first deviations recorded
under the [E/N gates](REST_MACHINING_PLAN.md#rc01-standalone-and-cambam-output-gates).
Then move the existing native owners as one reviewed package-layout slice;
the current trial should first identify the adapter contracts worth preserving.

## RC01 first CamBam output trial - 2026-09-23

The user opened the prepared A/B/C candidates in CamBam Plus 1.0 and exported
`A-rough.nc`, `B-explicit.nc`, and `C-native-cleanup.nc` under the ignored
`output/rc01-output-20260923-130450/` directory. The user reported the
`Default` postprocessor and Default mm profile; the G-code headers independently
name `Default`. The posts at the time of the first trial had SHA-256 values
below. The original A file was later overwritten by a repeat export in that
directory; its first-trial hash and inspected findings remain recorded here.

| Post | SHA-256 |
| --- | --- |
| `A-rough.nc` | `2cf5b21cb853a87351fb4c5a3b94fc306254c0bba73590b01fc38fe0b5eeadf3` |
| `B-explicit.nc` | `d4804ad441e7a4087ece8d6ca3a25981cbc2656570908e4da2aa8ca0ac061099` |
| `C-native-cleanup.nc` | `3ace6c9373f31e070d153792adc43acc69b807203de54b069fe94a2da0fd9f15` |

The original comparison reader stopped at each post's startup `G0 Z5.0` before
`G17` and T1. The Default post uses `G21 G90 G61 G40`, then that Z-only retract,
then `T1 M6`, `G17` and `M3 S12000`; requiring G17/T1 for the startup retract
was a reader defect. The reader now parses it but reports that the initial
machine position is not encoded. It retains emitted tool-event positions and
flags tool changes without an explicit spindle stop.

With the reader corrected, A's first ordered deviation is line 12: its rapid
ends at `(27,18.5,5)` instead of the framework's `(5,5,5)`. B and C similarly
start T1 at `(3,24,5)` and `(26.958,10.5,5)` on line 13. The first A plunge
at line 14 reaches Z=-2 although its MOP is named Z=-1. All three posts reach
Z=-6, violating the Z=-3 target floor and tool travel bound. This comes from
putting the same negative level in the Pline vertex and Engrave target depth;
CamBam adds them. The Z=-2 and Z=-3 MOPs also generate multiple incremental
levels from stock surface 0, adding redundant and deeper cuts. No residual
coverage result can rescue this protected-floor overcut.

B changes to T2 at `(5,5,5)` on line 987; C changes at `(3,8.5,5)` on line 996.
Both are away from the declared `(-10,-10,5)` setup position, and neither post
contains an `M5` before `T2 M6`. C's four native Pocket operations do produce
motion, but their cleanup is downstream of the invalid T1 prefix. There is no
E or N acceptance and no physical machining claim. The user raised concern
that these A/B/C probes should not be treated as the final rest strategy.

The first adapter repair puts candidate Plines at Z=0, uses stock surfaces
0/-1/-2 for the -1/-2/-3 single-depth MOPs, and sets CamBam's documented
`OptimisationMode=None` to request generated target order. The reader handles
the observed Default-post preamble and records unsafe event-state warnings.
Fresh candidate files and a hash-guarded manifest were generated and strict-
reimported under `output/rc01-repair-20260923-132551/`; their motion fingerprint
remains `39fde4a4c04295d50c7e53875478445cd5881eb9784f0c00d8ef676c4d89ffbd`.
These files have **not** been posted through CamBam, so corrected depth and
order are not native-verified. The Engrave representation still cannot encode
the required feed approach, retract, setup-position tool change and explicit
spindle stop. Reopen explicit-motion delivery only with a carrier that can
express and preserve those roles. For a first useful `.cb` workflow, prioritize
native MOP roughing and corner cleanup with independent replay of the actual
posted motion; the current C cleanup cannot be certified from its MOP settings.

Verification with `.venv/Scripts/python.exe`: `test_rc01_native.py` 3 tests,
`test_rc01.py` 6 tests and `test_mop_parameters.py` 18 tests passed;
`compileall -q cambam_builder` and `git diff --check` passed. The original
source/candidate hashes still match their comparison manifest after the user
exported G-code. The corrected A/B/C candidate hashes match their new manifest;
the files were strict-reimported during generation. Automated checks verify
the adapter's changed XML fields and reader behavior, not CamBam's motion for
the corrected candidates. The focused native result follows; full E/N requires
a suitable carrier and actual-motion replay.

## RC01 repaired-A CamBam output check - 2026-09-23

After commit `6dd3417`, the user posted the repaired
`output/rc01-repair-20260923-132551/A-rough.cb` through CamBam Plus 1.0 to
`A-rough.nc` in that directory. Its SHA-256 is
`3f94e4f9ed700f529e01cb43414c4273dc37d02bfdc356185baa83d8d455b24e`.
The candidate `.cb` still matches the comparison manifest hash. A first repeat
export landed over the original A post in
`output/rc01-output-20260923-130450/`; it had the old Z=-6 body and
new timestamp, so it was not evidence for the repaired `.cb`. The second,
correct export is the repaired-A post above. Preserve the user-provided files.

The repaired post's first T1 plunge ends at Z=-1 (line 14); feed endpoints are
limited to -1/-2/-3, and its minimum Z is -3. Thus the additive-depth and
single-level MOP fix passes this narrow native check. The Default post reader
decodes 1185 items and warns that the startup machine position is not encoded.
The exact comparison still fails at item 2, line 12: the first XY rapid ends
at `(26.5981,9.5,5)` instead of `(5,5,5)`. The actual first target equals the
first UUID-sorted target selected by the MOP after strict `.cb` import.
The first four posted run starts match the first four selected target starts:
`(26.5981,9.5)`, `(3,3)`, `(3,20.5)`, `(5,5)`.
`CamBamProject` owns explicit target selections as unordered sets and
`get_mop_targets()` sorts UUIDs; the writer exports that order. For the first
four runs, CamBam's `OptimisationMode=None` follows this exported sequence,
not the framework motion order. Changing that shared selection contract solely
for this probe would still leave the carrier unable to encode movement roles.

The first approach is `G0 Z1` from clearance +5, whereas RC01 requests a feed
approach to +1. After the first cut, `G0 Z5` is a rapid retract where RC01
requests a feed retract; the post has 236 rapid upward moves from negative Z.
The final spindle stop is at `(37,15,5)`, not the declared setup/end point.
These are independent E failures even if target order were restored. B/C were
not reposted; N remains unverified. The first practical next increment is a
native-MOP roughing and corner-cleanup variant with independent replay of its
actual output. Reopen exact explicit-motion delivery with a carrier that can
preserve ordered targets and the required move/event roles.

Post-commit gates against `main`: clean branch worktree before this evidence
edit, commit range `main..HEAD` nonempty, `main` ancestor of HEAD, and
`git diff --check main...HEAD` passed. Focused post-commit checks passed:
`test_rc01_native.py` 3 tests, `test_rc01.py` 6 tests and
`compileall -q cambam_builder`. The branch-level diff has 25 changed files;
no merge-ready conclusion is made from this check because the complete
`main...HEAD` content has not been reviewed for integration and this evidence
edit is uncommitted.

## RC01 native Pocket variant preparation - 2026-09-23

The next output probe uses a native T1 Pocket on the original RC01 Region rather
than an Engrave rendering of framework T1 centerlines. The rough-only candidate
enables that one Pocket; the combined candidate appends four T2 native Pockets
on the prescribed 7 x 7 mm corner windows. Both preserve the original Region,
Part stock and disabled source T1/T2 intent. The builder strict-reimported each
candidate and confirmed the source normalizes to the accepted `Job()`.

The prepared, ignored files are under `output/rc01-native-mop-20260923-02/`.
`N-rough.cb` has SHA-256
`e14afa4b5594814c754fe828898e0e13de5672c9d5bd55a4d470aec457f7ea6c`;
`N-native-cleanup.cb` has
`23e44d90b3d0be46389f0a86b915dad42a578835e5ab4978f52c620c331bd2b1`.
The `comparison.json` manifest guards both. No `.nc` exists for this pair yet,
so these hashes are input evidence only. The user-facing CamBam procedure is
in the [runbook](DEVELOPMENT.md#rc01-native-pocket-roughing-and-corner-cleanup-probe).

`rc01_native_post` parses the existing bounded Default-post dialect, compares
the T1 move prefix between both posts, and measures rough/final area and
residual location at all three depth-slab bottoms from posted G1 straight
segments. It checks basic event, rapid, feed, travel, floor and protected XY
bounds, while keeping GEOS topology and unencoded startup machine position as
explicit limits. It is not an independent toolpath generator.

A stress replay of the **older invalid** C post decoded 1402 items. It exposed
the already recorded missing T1/T2 spindle stop, displaced tool change and
rapid below +5. Its posted T1 cuts give polygon-radius rough-rest intervals
`[7.7255777, 7.7258435] mm²` per slab, but this is no acceptance: that post
reaches Z=-6. The lower bound below the ideal `7.7256661 mm²` is consistent
with a separately flagged roughly 0.00004 mm inward island rounding at the
first contour (line 15). This confirms that a plausible area result cannot
override protected overcut or motion failures. A focused synthetic regression
also rejects a posted pair with too little coverage and a rapid below clearance.
The independent posted-rest oracle accepts the known generated RC01 coverage
within both rough and final budgets. Final focused verification used
`.venv/Scripts/python.exe`: `test_rc01_native.py` 5 tests passed,
`compileall -q cambam_builder` passed, and `git diff --check` plus the untracked
text trailing-whitespace scan found no errors. These tests do not substitute
for posting the new candidates.

**Reopen/next evidence:** obtain `N-rough.nc` and `N-native-cleanup.nc` from
CamBam Plus 1.0 using the stated Default post/profile, retain candidate hashes,
run the local audit, then inspect its actual cut order, T1 access to T2 entries,
tool events, floor and residuals. If Default-post rounding again penetrates a
protected boundary or its entries/retracts violate RC01, record N as failed;
changing acceptance bounds or assuming MOP settings certify removal is not a
valid repair. The first native result decides whether to adjust a native
parameter/post precision, provide an explicit setup witness, or move to a
different output carrier. E remains a separate explicit-motion blocker.

## RC01 native Pocket posted-motion trial - 2026-09-23

The user saved both new posts from CamBam. Their headers name the expected
candidate and `Default` postprocessor; `G21 G90 G61` confirms millimetres,
absolute coordinates and exact-stop mode in the emitted files. The profile name
is not encoded in G-code. Candidate SHA-256 values still match `comparison.json`.

| File | SHA-256 |
| --- | --- |
| `N-rough.nc` | `cde87d91d6f4c746cdabe7a80b04808d65551bc4d15db47e724550c6b0d444a2` |
| `N-native-cleanup.nc` | `6c36c80766c84d8442cb28c6c1808da9f219dcb66d551333db951bf712109e20` |

The previous reader rejected native `G3`/I/J at rough line 17. The native-only
mode now decodes bounded XY G2/G3 relative-center arcs, including helical Z,
while the explicit Engrave comparison still rejects them. It checks arc radius
consistency within 0.001 mm and brackets the flattened path using 0.0001 mm
chord sagitta plus emitted radius mismatch. The audit decoded 168 rough items
and 590 combined items, including 45 arcs in each. The T1 posted move prefixes
match, including arc centers; both minimum tip heights are exactly -3 mm.

| Posted result, each of the three open depth slabs | Area interval (mm²) | RC01 coverage budget |
| --- | ---: | --- |
| T1 rough rest | 7.7255777–7.7258435 | Upper <= 8.2256661: passes |
| T1 then T2 rest | 0.8583975–0.8584271 | Upper <= 1.3584073: passes |
| Guaranteed cleanup benefit | >= 6.8671506 | >= 6.3672588: passes |

The corresponding rough and final volume intervals are 23.1767330–23.1775306
and 2.5751926–2.5752812 mm³. The polygonal residual outside the allowed ideal
corner/boundary 0.05 mm envelope is 0 at all three slabs. These are bounded
coverage results from posted G1/G3 cuts, not a claim that the sequence is safe.
The lower area interval can lie slightly below the analytic ideal because the
outer path/radius enclosure deliberately overestimates possible removal; it
does not prove protected overcut. GEOS floating topology still lacks a formal
interval bound.

The four required T2 columns at (5,5), (35,5), (35,25), (5,25) and all eight
actual native T2 vertical locations each have an exact single prior T1 straight
cut witness at tip Z=-3. Rational point-to-segment distance proves the radius-1
T2 disk fits in a radius-3 T1 swept cylinder through the full pocket depth.
The ignored `output/rc01-native-mop-20260923-02/audit.json` retains each witness
line and the complete per-slab comparison. Small positive GEOS differences at
two boundary-tangent columns are not used as a clearance pass; the exact
witnesses are the authority for this bounded column result.

**N output fails the accepted motion contract despite coverage.** Rough T1
starts at (7.8,21.5598), not the required (5,5). The first approach is `G0 Z1`
from +5 (rough line 13; combined line 14), and later rapid descents reach -1;
the audit found 12 low rapids in rough-only and 36 in combined output. The
native Pocket inserts 9 T1 and 117 combined ramped XY/Z moves whose axial
engagement is not proved by the bounded RC01 process contract. Twelve rough
and 48 combined low-level XY moves use F60, which is not an accepted RC01
cut/cleared-travel role. At combined line 180, `T2 M6` occurs at
(31.1081,27,5), away from (-10,-10,5), without `M5`; the next `M3` starts
while the reader still has the spindle running. Final spindle stops also occur
away from setup. The Default post does not encode its initial machine position.
Twenty-seven T1 island tangencies remain numerically unresolved, so zero
protected overcut cannot be certified from these rounded arcs/lines.

The user's file submission establishes that CamBam produced both Default posts,
and the bounded rest/column checks pass. It does not accept the emitted RC01
motion or physical machining. This trial closes the native Pocket coverage
probe with N failed; E remains separately blocked by its Engrave carrier.
**Reopen** N only with an output representation or post strategy that preserves
RC01's setup, approach, retract, feed and tool-event roles, followed by fresh
posted-motion replay. An unchanged Pocket/Default repost would repeat the
observed failures. Exact island tangency and any changed arc/reader dialect
need a fresh protected-stock proof before acceptance.

Final focused checks after the arc/access audit change: `.venv/Scripts/python.exe
-m unittest discover -s tests -p test_rc01_native.py -v` passed 7 tests;
`compileall -q cambam_builder`, working `git diff --check`, and committed
`git diff --check main...HEAD` passed. `main` is an ancestor of the current
branch. The returned posts were replayed into the ignored `audit.json` after
the final audit-code edit. This review change is uncommitted; the branch is not
merge-ready and no physical acceptance was requested.

## RC01 Pocket/Default role-carrier assessment - 2026-09-23

Selected route: one complete native T1 Pocket plus four native T2 window
Pockets using CamBam's Default post. The user performs CamBam posting from
prepared `.cb` files. The preceding actual posted pair proved T1/T2 coverage
but failed entry, low rapid, ramp, feed, setup and tool events together.
This assessment tested the remaining local MOP controls as one repair, without
repeating the unchanged export.

`build_native_variant(..., role_trial=True)` produced a rough-only and a
complete T1/T2 candidate in `output/rc01-roletrial-20260923-1845/`.
It changed only enabled candidate Pockets: `LeadInType=None`,
`OptimisationMode=None`, `StepoverFeedrate=Cut Feedrate` and
`MaxCrossoverDistance=0`. The original Region, Part stock and two disabled
source MOPs remain; each candidate strict-reimports with the same RC01 job.
The SHA-256 values are `6fe3782fceb731bede232fcb4a1b0a9868cb13fdcf6faa0e3a65f3cf78f49304`
for `R-rough.cb` and `a062175ac5df11d1825589c144a9298dee9714146597fa04d5e692b3d796fe26`
for `R-native-cleanup.cb`. The manifest records hashes and exact enabled order.

[CamBam's lead-move documentation](https://cambamcnc.org/doc/1.0/cam/lead-moves.html)
says `None` changes a ramp to direct plunge. Its
[postprocessor documentation](https://cambamcnc.org/doc/1.0/cam/post-processor.html)
defines general rapid/feed formatting, MOP boundary header/footer scripts,
and an internally generated block stream. These controls do not expose a
separate move role for feed approach from +5 to +1, feed retract to +5, or the
exact first (5,5) cutting entry. Reformatting every rapid as a feed would also
change the required above-stock XY rapid role. Header/footer scripts cannot
replace every internally generated entry/link/retract. The Default post's
earlier T2 change at (31.1081,27,5) without `M5` also needs a complete
setup-position event sequence, not a feed or lead setting. A custom post-build
motion rewrite or a G-code/NCFile carrier would own new motion semantics and
must be independently replayed; it is not established by this repair.

**Result:** the selected Pocket/Default carrier fails the RC01 role-capability
precondition despite its already proven bounded rest coverage. The repaired
`.cb` files were not posted, so no claim is made about their actual path order,
rest or removal. Requesting their post would add no proof that this carrier
can express the missing roles. E remains failed on Engrave order/role evidence;
N remains failed on the posted Pocket pair. No route passed full emitted motion,
protected-stock, access and process verification, and no physical acceptance
was requested. Reopen with an explicitly role-bearing carrier, not another
Pocket/Default settings trial. The subsequent user clarification and
role-bearing CamBam carrier are recorded below; direct standalone posting
retains its previous timing under the
[RC01 milestone](REST_MACHINING_PLAN.md#next-rc01-output-milestone-after-native-pocket-trial).

Checks: `.venv/Scripts/python.exe -m unittest discover -s tests -p
test_rc01_native.py -v` passed 8 tests, including strict reimport and
preserved-source checks for this full T1/T2 pair. These are artifact/input
checks, not posted-motion or CamBam application acceptance. Replaying the
previous actual posts with the current audit still returns `fails_RC01`,
identical T1 prefixes, both rest budgets met, 62 rough and 233 combined
findings, 9/117 ramped XY moves and a T2 change at (31.1081,27,+5).
The repeat result is retained locally as `prior-post-audit.json` beside the
role-trial artifacts; it is evidence about the previous posts only.

## RC01 literal-motion CamBam carrier preparation - 2026-09-23

The user clarified the handoff: the agent creates the complete `.cb`; the user
generates native G-code in CamBam and returns the `.nc` only when needed.
No standalone direct-post timing change was requested. The selected next
carrier is one enabled Drill/CustomScript MOP with a single setup anchor Point
at (-10,-10). Its literal text contains the complete generated T1/T2 sequence
from `cam_core.rc01`, including all feed/rapid move roles and the internal
T1 stop, T2 change and restart. The MOP/Default post is expected to wrap it
with the initial T1 change/start and terminal stop; actual behavior is pending
the user's native post. This is an alternate explicit E carrier. It is not
native Pocket N, and it does not establish XYZ/Engrave parity.

`output/rc01-script-20260923-2115/S-combined.cb` is 64,034 bytes, SHA-256
`14b53e40c771ad69b9412729d3110185c3f32ad83c2da10f094ffdcda8c6a8a3`.
It preserves the original Region, Part and two disabled source Pockets and
strict-reimports with the same RC01 job. The carrier has 2,942 literal lines
representing 2,945 framework items once wrapper events are included. Its
manifest records job and motion fingerprints and pending native output.
The local audit accepts only the bounded Default-post dialect and standalone
G98/G80 Drill wrapper markers. It compares every emitted move/event to the
planned role, rejects extras, and replays actual decimal coordinates through
the continuous RC01 verifier. The exact-centre decimal writer rejects
nonterminating coordinates rather than rounding a protected boundary.

A synthetic wrapper with all 2,945 items passed T1/final stock, access,
process and rest verification and reported finite-tool partial completion.
Changing the first feed approach to a rapid was rejected as a sequence
deviation. This test proves the builder/audit boundary, not CamBam behavior:
CamBam may modify, omit or insert script motion. The only requested user
artifact is a Default/Default mm `S-combined.nc` posted from the prepared
`.cb`; the [runbook](DEVELOPMENT.md#rc01-literal-motion-cambam-carrier) gives
the exact steps. After receipt, inspect the native post and record E, N and
physical acceptance separately. Existing GEOS residual topology and the
declared initial setup position remain limits.

## RC01 first literal-motion CamBam post and newline repair - 2026-09-23

The user posted `output/rc01-script-20260923-2115/S-combined.nc` from the
prepared `.cb`. Candidate SHA-256 still matches the manifest:
`14b53e40c771ad69b9412729d3110185c3f32ad83c2da10f094ffdcda8c6a8a3`.
The returned post SHA-256 is
`1dcd09c96989dbfbfa80b070b1fe52ed8e9a2e2d062ecc96fe5ca798f6d6a240`.
The header names `S-combined` and `Default`, starts in `G21 G90 G61 G40`,
posts `T1 M6`, `M3 S12000`, moves to the setup anchor and emits `G98`.
Then all 2,942 literal motion lines appear on **one physical NC line** with
2,941 literal `|` characters (line 14, 55,731 characters). The whole post
has only 19 physical lines. The fail-closed reader rejects line 14 as
unsupported command text before any stock replay. This NC file does not
establish executable T1/T2 motion, protected-stock clearance or rest coverage.
No machine use is accepted.

As a diagnostic only, the agent copied the returned `.nc` to
`S-combined-split-diagnostic.nc` in the same ignored directory and replaced
the 2,941 literal separators with physical newlines. This altered file is
**not** CamBam output. Its CamBam wrapper and all 2,945 logical items then
match the framework program and pass continuous stock/access/process replay;
rough rest is 7.7750106–7.7876785 mm² and final rest is
0.9214411–0.9263454 mm² in each slab, with expected finite-tool partial
completion. This isolates newline encoding as the observed blocker and makes
another native post of the repaired `.cb` a focused, useful check. The
diagnostic result cannot substitute for a valid CamBam `.nc`.

The documented `|` newline shorthand was not expanded by this CamBam Plus
1.0 CustomScript output; the actual post controls this finding. The builder
now encodes real XML text newlines, and the synthetic audit test consumes
the reimported script as-is rather than converting pipe separators for it.
The repaired, strict-reimported candidate is
`output/rc01-script-lines-20260923-01/S-combined.cb`, SHA-256
`91cb871c22429c9a94bcea0f7cbbee0532fa462eb52e3d8558d69094fa6cf933`.
It retains the same RC01 job and motion fingerprints and 2,942 logical
script lines. No G-code source file was generated by the agent. The user
needs to post only this revised `.cb`; the first `.nc` remains failure
evidence. The actual revised output must pass the full emitted-motion audit
before E can close. N and physical acceptance remain separate.

Focused `test_rc01_native.py` passed 9 tests after the repair. The real-post
failure is a regression case for literal newline preservation. At that point,
no repaired CamBam post or rest replay existed yet.

## RC01 literal-motion CamBam output acceptance - 2026-09-23

The user generated `output/rc01-script-lines-20260923-01/S-combined.nc` in
CamBam from the prepared `.cb`. The candidate SHA-256 is
`91cb871c22429c9a94bcea0f7cbbee0532fa462eb52e3d8558d69094fa6cf933`;
the actual post SHA-256 is
`1e972c5834198c9bbfe11c272ee6db7716e2acb28578d7e57257b7f3e5bf502d`.
This is the revised, newline-repaired candidate, not the failed pipe-separated
post above. The post has 2,960 physical lines, zero literal `|` separators,
and two each of `T[12] M6`, `M3 S12000` and `M5`. Its Default wrapper starts
in `G21 G90 G61 G40`, changes to T1, starts the spindle and reaches the
(-10,-10) setup anchor before the script. It ends with `G80`, clearance
`G0 Z5`, spindle stop and `M30`.

The bounded audit command was:

```powershell
.venv\Scripts\python.exe -m cambam_builder.integrations.cambam.rc01_script output/rc01-script-lines-20260923-01/comparison.json output/rc01-script-lines-20260923-01/S-combined.nc
```

It returned `status=bounded_emitted_motion_pass`, with all 2,945 ordered
program items matched in tool, feed, motion mode and decimal endpoint.
The audit reconstructs the program from the **actual posted motion** and
replays the T1-only and complete T1/T2 sequences through continuous RC01
verification. Protected stock, tool-component and stock-dependent access,
setup/tool/feed/process limits, and required T1-to-T2 residual columns pass.
Each of the three depth slabs has rough rest
`[7.775010615955999, 7.787678472024001]` mm² and final rest
`[0.9214411294439999, 0.9263453527720001]` mm². The certificate is
`partial_target_completion`, the expected finite-tool outcome. The exact
audit result is retained locally as `audit.json` beside the post.

This accepts **one bounded explicit-script E output route** for the synthetic
RC01 T1 roughing plus T2 cleanup sequence. The original XYZ/Engrave E
carrier has no parity acceptance; the native Pocket N posts still fail their
motion-role requirements despite passing rest coverage. The Default post
does not encode the incoming machine position, so the replay assumes the tip
begins at (-10,-10,+5). Residual-location GEOS topology has no formal
interval proof. No physical machining or controller acceptance is claimed.
No further user validation is needed for this posted-motion finding; another
CamBam export adds no new evidence unless the carrier or settings change.

## Native optimiser corpus preparation - 2026-09-23

The installed CamBam Plus 1.0 `CamBam.CAD.dll` enum reports
`OptimisationModes.Standard=0`, `Experimental=1`, `None=-1`; the
[CamBam 1.0 Profile guide](https://www.cambam.info/doc/1.0/cam/profile.html)
labels the first two Legacy (0.9.7) and New (0.9.8). The installed XML roots
and `lathe-test.cb` sample support the
[inventory](REST_MACHINING_PLAN.md#inventoried-baseline-and-prepared-slice-2026-09-23).
The system files used for this baseline have SHA-256:

| Installed profile file | SHA-256 |
| --- | --- |
| `post/Default.cbpp` | `adc3034633f0f6ad6ddc4c9eabc63844cb59173be8ee780ac47e198a7145c263` |
| `styles/Standard-mm.xml` | `d3887ee96277a1a8e8d452b223f2d02d34b341a9de7b7e3b3e7d7080b224e4c4` |
| `tools/Default-mm.xml` | `afe4b2c7abcfe17a44a9fd4752a1d40847ce0e399fde9b105bd8da66160d08db` |

The accepted-for-posting candidate set is
`output/optimizer-corpus-20260923-04/`. Its `manifest.json` guards these
strict-reimported `.cb` hashes:

| Candidate | SHA-256 | Gate |
| --- | --- | --- |
| `atlas-legacy.cb` | `972348f5122fc933368f9410d597b4596b59101bcc874d4d637b34c48de9acc3` | First post |
| `atlas-new.cb` | `ba7457271fbbbed186e5c322cd94df351940c8520fef99d18e1308ca18f5c910` | First post |
| `links-legacy.cb` | `8faf51989dc149cd8a70de419c7a6297e67553a910774430763afb4c9fc797de` | Held interaction |
| `links-new.cb` | `957fa0ad7eff19a9ed57487763bac285180cb8168d8d6529c84f88e7c534bd02` | Held interaction |

Each pair is cloned before changing the mode; an XML regression compared
them after removing only document name and mode text. Every candidate pins
millimeter units, Default post, installed style/tool library names and
explicit modeled MOP fields. The interaction pair selects the installed
`cutout` CAM style on its final Profile while pinning its modeled fields.
Strict import retains all nine MOPs across the two families and their
targets/properties per mode. The corpus parser read the previously returned
native RC01 `N-rough.nc` as 166 G0/G1/G2/G3 motion words, four M events and
zero unsupported words. That demonstrates intake of one real Default post;
it is **not** an atlas observation or proof of Legacy/New mapping.

`tests.test_optimizer_corpus` passed three tests for mode-pair identity,
strict snapshot/hash intake, modal arcs and unresolved cycles. The broad
base-environment `unittest discover -s tests -v` ran 365 tests and failed
only two module imports (`test_mcp_cross_document`, `test_mcp_mop_parity`):
the optional MCP dependency `anyio` is absent from this base environment.
That broad run is not claimed as passed; no MCP code changed. Package
`compileall`, import/construct smoke and `git diff --check` passed;
untracked text was checked separately for trailing whitespace. The first
manual gate is the two atlas CamBam posts and UI mode/settings confirmation
in the [runbook](DEVELOPMENT.md#native-optimiser-shapemop-mapping-corpus).
Until their exact `.nc` hashes, path/event sections and limitations are
reviewed, all mapping cases remain pending and no native stock/rest
authority is available. Native-only class reopening criteria remain in the
inventory; the interaction posts wait for the atlas method verdict.

## Native optimiser corpus posted output - 2026-09-23

The user exported all four prepared `.cb` files in CamBam Plus 1.0 and
confirmed the displayed Legacy (0.9.7) and New (0.9.8) MOP settings in the
corresponding documents. That report confirms the mode labels only; no
physical cut or stock result was reported. Exact copies of the four candidate
`.cb` files, four returned Default `.nc` files and the original
`manifest.json` are now tracked under `tests/fixtures/optimizer_corpus/`.
The derived `observations.json` in the same directory contains source/post
hashes, ordered section/path summaries, events, unresolved words and separate
program/section motion fingerprints. Its source manifest SHA-256 is
`2608d86faa4aa11cd199559707e751a8ca64cfe0322cb6e1a9880efe938fc400`.
The manifest's `pending_native_post` fields are preserved as input-time
provenance; the generated observation status remains `unreviewed` by design,
and this section owns the later bounded semantic acceptance.
The three installed Default post/style/tool file hashes still match the
preparation record above. Each post has its matching candidate title,
`Post processor: Default`, `G21 G90 G61 G40`, all expected MOP comments and
`M30`; each `.cb` retains its preparation hash.

| Native program | Exact `.nc` SHA-256 | Parsed G0/G1/G2/G3 words / M events / unresolved words |
| --- | --- | --- |
| `atlas-legacy.nc` | `f8ded8ff8e3934205a78aa1866aec72fe7e802e748b8d61dd0e43d98388e2470` | 362 / 6 / 3 |
| `atlas-new.nc` | `ff886629f486058118901aa42f65e2dfb5ce5a8d4f57ebc6f5f72dcfdeecc568` | 385 / 6 / 3 |
| `links-legacy.nc` | `08a20cbc0d91c885b8a86c796827d946175c91bf9138b1af957dc90738b86a8d` | 93 / 8 / 0 |
| `links-new.nc` | `5e959f11685fba2a35d528d657fdfae01edfa6a2a60bb3efbf5d2ff484768622` | 93 / 8 / 0 |

The following are **accepted native posted-output observations** for these
exact source/profile hashes. Each MOP comment names a spatially distinct
synthetic target; the retained NC records every direction, decimal endpoint,
arc center, entry, link and retract. Counts below are posted motion words,
not independent toolpath counts or stock certificates.

| MOP section and source target(s) | Legacy vs New observed mapping |
| --- | --- |
| `ATLAS_PROFILE_RECT_CIRCLE`: rect, circle | 33 vs 33 identical motion words; first +5 XY approaches `(5,4)`, then `(25.1381,8.8331)`; cutting endpoints reach Z=-1 and -2. |
| `ATLAS_PROFILE_OPEN_PLINE`: open-pline | 12 vs 12 identical; first approach `(48.6585,4.2474)`; Z=-1 and -2. |
| `ATLAS_POCKET_CLOSED_PLINE`: closed-pline | 70 vs 70 identical; first approach `(76,10)`; Z=-1 and -2. |
| `ATLAS_POCKET_REGION_ISLAND`: region-island | 200 vs 222, different motion. Legacy first approach `(157.6,9.904)` and has 52 G3/zero G2 words; New starts `(157.6,9.6)` and has 34 G3/20 G2 words, with more rapid links. Both reach Z=-1 and -2. This is a bounded mode effect inferred from paired inputs that differ only in document name and optimiser token. |
| `ATLAS_ENGRAVE_ARC_TEXT`: text, arc | 43 vs 44, different text-stroke order/entry. Legacy first approaches `(111.9707,4.3516)`; New first approaches `(110.3184,2)` and inserts a separate text-stroke rapid. Both later approach the Arc at `(136.8869,6.4685)` and reach Z=-1 and -2. |
| `ATLAS_DRILL_POINTS`: points | Both posts emit `G98` then `G81 X94 Y8 Z-2 R2 F60` and a second `G81 X102 Z-2`, with `G80` cancellation. The reader records these exact words but does not expand controller cycle motion; the decoded three words in this section are rapids only. |
| `LINKS_MULTI_TARGET_DEPTH_LEAD`: link-circle-3, -1, -2 in XML target order | Both modes emit the same 71 words, including G3 helical lead moves and Z=-1/-2/-3. Posted +5 XY approaches are right-to-left: circle 3 `(66.2639,7.3887)`, circle 2 `(42.2639,7.3887)`, circle 1 `(19.2639,7.3887)`. This differs from XML target order. |
| `LINKS_ARC_SECOND_TOOL`: link-arc; `LINKS_RETURN_FIRST_TOOL`: finish-rect with `cutout` style | Both modes emit identical 7-word Arc and 14-word return Profile sections, with Z=-1/-2. The Arc section uses G2/G3 and T3; the final Profile uses T1. All modeled fields on the styled MOP are explicit, so this pair does not isolate a style-inheritance effect. |

All decoded arcs have endpoint-radius mismatch below 0.000087 mm at posted
decimal precision. Program fingerprints exclude timestamp headers and line
numbers: the atlas pair differs in only the Region Pocket and Text/Arc
Engrave sections; the complete links pair has equal normalized motion.
All four posts use F60 and F240 for the modeled entry/cut feeds. Each initial
`G0 Z5` leaves the incoming machine position unknown. The atlas changes T1
to T2, and links changes T1 to T3 to T1, without an explicit `M5` before
those `M6` commands; each program has a final `M5`. These are recorded
output events, not physical/controller acceptance.

`G98`/`G81` remain explicit unsupported words for stock replay; post-added
versus CamBam-generated path moves are not separable from the `.nc` alone.
The native-only class results in the inventory retain their reopening
criteria. No one of these observations certifies region-island clearance,
tool/holder access, entire stock removal, rest area, controller execution or
production machining. A later caller may select native posted motion only
after a separate verified stock authority contract and source/post freshness
check; framework motion keeps its own fingerprint. The bounded corpus
stopping condition is satisfied by these accepted observations plus the
explicit blocked/historical inventory results. No additional user validation
is needed for this output-mapping finding.

The final focused command,
`.venv\Scripts\python.exe -m unittest tests.test_optimizer_corpus -v`,
passed four tests against the tracked input/output fixtures. The test
recomputes `observations.json`, checks all source/post hashes and locks the
mode comparison, posted target order, raw Drill cycle and tool-event
sequences. Package `compileall`, corpus import and `git diff --check` passed;
untracked fixture text was checked separately for trailing whitespace. The
earlier broad-suite limitation from optional missing `anyio` is unchanged;
no MCP or shared XML behavior was edited in this increment. No physical
CamBam revalidation is requested after the supplied posts.

## RC01 selected native posted-stock authority - 2026-09-23

The first caller-selectable stock/rest slice uses the previously user-posted
RC01 T1 Region/Pocket pair, now retained with its source, setup, manifest and
versioned evidence under `tests/fixtures/rc01_native_stock/`. The T1 candidate
SHA-256 is `e14afa4b5594814c754fe828898e0e13de5672c9d5bd55a4d470aec457f7ea6c`;
the actual `.nc` SHA-256 is
`cde87d91d6f4c746cdabe7a80b04808d65551bc4d15db47e724550c6b0d444a2`.
The original source is pinned by the unchanged comparison manifest. These
are copies of the accepted posted trial bytes, not a new CamBam export.

`analyze_rc01_stock("native_posted", ...)` strict-reimports both source and
candidate, checks the selected MOP and all pair hashes, then computes rest
from posted T1 G1/G2/G3 motion. All three slabs ending at Z=-1,-2,-3 return
7.72557766758–7.72584353670 mm². The parsed-motion fingerprint is
`a0fdecec1c961b146179371b20b0055b771a566a0412a97853a94aec9e5f5177`.
The same entry point with explicit `framework_generated` selection verifies
the supplied core program independently: its motion fingerprint is
`39fde4a4c04295d50c7e53875478445cd5881eb9784f0c00d8ef676c4d89ffbd`
and rough rest is 7.77501061596–7.78767847202 mm² per slab. The differing
numbers and fingerprints demonstrate why native intent cannot stand in for
either motion source.

The native observation retains 62 RC01 motion-role findings, including low
rapids, so stock-dependent execution remains blocked. The initial machine
position is absent from Default output; the geometric area uses Shapely/GEOS
polygon enclosures and has no formal topology interval proof. No native T2
cleanup, complete safe trajectory or physical machining acceptance follows.
`check_native_freshness()` rejects prior results after evidence record, source,
candidate, setup, manifest or post byte changes. A repinned edited post with
the old reviewed motion fingerprint is also rejected. Pairing a new source
with a new post requires explicit provenance and review; the filename/header
is insufficient evidence of that pairing.

The combined focused command,
`.venv\Scripts\python.exe -m unittest tests.test_rc01_stock_authority tests.test_rc01_native tests.test_optimizer_corpus -v`,
passed all 15 tests. After the final diagnostic-message edit, the two new
authority tests passed again. Package `compileall`, import/construct
smoke and `git diff --check` passed; the latter covers tracked files only, so
untracked text was also scanned for trailing whitespace. Manual CamBam
validation adds no evidence for replaying these already posted exact bytes.
Reopen the native Pocket/Default execution claim only if a carrier can encode
the missing roles and its actual post passes renewed whole-motion checks.

## RC01 paired native posted-stock assessment - 2026-09-23

The previously user-posted `N-native-cleanup.cb` and `N-native-cleanup.nc` were
copied byte-for-byte beside the accepted T1 fixture under
`tests/fixtures/rc01_native_stock/`. Their SHA-256 values are respectively
`23e44d90b3d0be46389f0a86b915dad42a578835e5ab4978f52c620c331bd2b1`
and `6c36c80766c84d8442cb28c6c1808da9f219dcb66d551333db951bf712109e20`.
The original comparison manifest still pins both candidates to the same source.
`paired_evidence.json` pins that manifest, the setup, both exact posts and their
parsed-motion fingerprints. This reuses the posted trial; it is no new CamBam
export or claim about the unposted Pocket role-trial candidates.

`analyze_rc01_stock("native_posted", evidence_path=...paired_evidence.json)`
strict-reimports the source and both candidates, checks their selected MOPs,
headers, sections, source/candidate/post hashes and identical parsed T1
event/move prefix through the last T1 move. The T1-only and combined parsed
fingerprints are `a0fdecec1c961b146179371b20b0055b771a566a0412a97853a94aec9e5f5177`
and `c3ba1b5c07e09f5d6a2777dcf7da65b0718365153f5539a71c0a97da4c45fd35`.
`check_native_freshness()` invalidates a prior result after changes to either
post, either candidate, source, setup, manifest or evidence record. Re-pinning
a changed post alone cannot inherit the accepted motion interpretation.

The paired replay confirms rough rest `7.72557766758–7.72584353670 mm²` and
final rest `0.85839751862–0.85842705963 mm²` in each of three depth slabs.
Both area/location budgets pass, all four required and eight actual T2 vertical
columns have exact prior full-depth T1 cut witnesses, and the parsed T1 prefix
matches. This supports **bounded geometric T2 coverage conditional on the
recorded T1 cut footprint**. It does not establish safe T1 execution or safe T2
entry, linking and engagement. The replay retains 62 rough and 233 combined
motion-role findings: low rapids, ramped engagement, unsupported F60 low-level
XY moves, a displaced T2 change without spindle stop, unresolved island
tangencies and the missing initial machine position. The returned
`stock_dependent_use` is `blocked_by_motion_or_rest`. Native Pocket N, physical
machining and production stock authority remain unaccepted. GEOS floating
topology has no formal interval proof; the arc/radial polygon enclosure does
not resolve that limit.

Focused `.venv\Scripts\python.exe -m unittest tests.test_rc01_stock_authority
tests.test_rc01_native -v` passed 13 tests, including real-pair replay,
freshness and a re-pinned T1-prefix mutation. No manual CamBam validation adds
evidence to this replay of exact accepted bytes. Reopen native cleanup only
after a changed carrier/post can encode the required motion roles and a newly
posted pair passes full trajectory and protected-stock checks. Continuing to
expand the same Pocket/Default parameter trial has no demonstrated path to
those roles.

## Bounded native V input normalization - 2026-09-24

The prepared `output/native-variable-v-20260924-02/source.cb` contains the
original XYZ spine (0,2,-1) to (12,2,-2.5), an 18 x 8 x 3 mm Part with
drawing origin (-2,-2,0), and one disabled VCutter Engrave targeting that spine.
Its explicit `setup.json` supplies the 3 mm radius/length pointed cone and
non-native setup controls. The source SHA-256 is
`ab97360d39640dc26da0cd65257ef2fec2d44e3afb279cc66f5a4cf700f45411`.
Strict import and normalization give the same `TaperedRequest` and plan
fingerprint `4e0c0f4249c94da0f51a8fb4b38f9da718bb60fe12134fb391beb235c1fd0f46`
as the standalone generator. The integration test compares the complete
nine-item expected motion and literal script with the standalone carrier,
then passes the whole-motion audit on a synthetic Default wrapper. The
synthetic wrapper checks the adapter, not a CamBam-produced post.

The preview candidate SHA-256 is
`04e1658da909ed6e1a3d93352243b2243287d12962e73b9889e6ba6bfba6f746`;
its sole enabled Engrave selects only the generated cut Pline. The explicit
candidate SHA-256 is
`edef7f1eb33e3dffcc5dbaf3f2cfda2c2ff65a96866fe532d1613ad3c5d6c376`;
its sole enabled Drill/CustomScript selects only the Point anchor. Both retain
the original spine UUID and disabled source Engrave relationship after strict
reimport. Negative checks reject changed world spine, stock width, VCutter
diameter, inherited diameter, altered cone setup and an unmodeled Engrave XML
field. A cosmetic source MOP display-name edit preserves the normalized
request. The XML integer/float fingerprint mismatch found on first integration
run was corrected by canonicalizing the bounded request before generation.

This closes automated native-input normalization and document separation for
one bounded case. The accepted older CustomScript post belongs to a different
`.cb`; this new native-derived candidate awaits its own actual CamBam post and
exact whole-motion audit. No manual CamBam check adds evidence to the input
normalization claim. Reopen the bounded input contract for changed target/tool
values only with a new core request and independent residual/output evidence;
reopen the explicit output claim when the prepared candidate is actually posted.

Verification after the final XML-field guard: `.venv/Scripts/python.exe -m
unittest tests.test_native_variable_v tests.test_variable_vcarve
tests.test_variable_cone_script tests.test_variable_cone_engrave
tests.test_cone_script tests.test_vcarve_slot tests.test_mixed_replay -q`
passed 19 tests. Package `compileall`, import/construct smoke, `git diff
--check`, and the untracked Python/Markdown/JSON trailing-whitespace scan
passed. The broader `unittest discover -s tests -q` run reached 389 tests
with no assertion failures but ended with two import errors because this base
environment lacks optional `anyio` for MCP-only test modules; 73 tests were
skipped. The requested work does not change or exercise the MCP adapter, so
the framework-focused gate above is the relevant final check. No MCP dependency
was installed. Reopen the broader MCP gate only when that adapter is in scope
and its declared optional environment is prepared.
