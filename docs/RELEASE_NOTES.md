# Consumer checkpoints and compatibility notes

These notes define versioned consumer scope. Package metadata comes from
`pyproject.toml` and `cambam_builder.__version__`; current delivery state lives in
[PROGRESS](PROGRESS.md). A version in source or a locally built archive does not
mean a tag, registry release or public distribution has been delivered.

## 0.3.0 native engine delivery

Checkpoint date: 2026-10-26. This additive checkpoint connects finite planar
tool/path selection to source-bound editable native files and a separate audit
of their complete CamBam-generated posts. It retains the 0.2.0 base-library
surface and dependency policy. Registry delivery is not implied.

- `cam_extensions.bundle_search` and `inlay_search` add the bounded BO01 planar,
  mixed-tool and paired-inlay search APIs. Finite budgets, manual choices,
  unresolved alternatives and estimated costs remain explicit; see the
  [search contracts](structure_spec.md#finite-planar-toolpath-bundle-search-bo01-a).
- With the `planar` extra, `integrations.cambam.bundle_document` exposes
  `bind_design`, `author`, `NativeDesign` and `NativeDocument`.
  `integrations.cambam.bundle_post.audit_post` checks actual Default-post bytes.
  The [native output contract](structure_spec.md#source-bound-native-bundle-documents-and-posted-motion)
  owns supported source geometry, motion, stock and freshness boundaries.
- `Bundle.design_fingerprint` identifies the independently declared fixed design.
  Existing positional construction remains possible, but an empty or mismatched
  fingerprint cannot enter native binding. Regenerate old/unbound search snapshots;
  changing that string alone does not establish geometric safety.
- Default-post parsing now tracks active G64/G61 mode. G64 in a header followed by
  G61 before motion is accepted; actual blended motion remains unverified. Earlier
  blended motion cannot be cleared by a later G61.
- Planar ordered jobs accept an occupancy stock box enclosing the removal target;
  native bundle binding requires that box to match the complete selected Part
  stock. Removal bounds stay target-specific and tool-body checks use full stock.
  Surface/volume stock matching is unchanged; the volume occupancy branch now
  reads the layered target's actual depth field instead of raising an attribute error.
- The short closed-offset sampling repair from BO01 preserves nonzero traversal
  and rounded-coordinate uniqueness; existing strict geometry gates remain.

The [native search example](../demos/native_search_workflow.py) combines these
APIs with CF02 setup composition. Geometry and MOPs stay editable, but edits or
placement changes invalidate the corresponding source/plan/post evidence. Native
post acceptance does not transfer selected search timing/optimality and does not
claim controller execution or physical cutting. Full 3D surfaces, wider native
classes, packing and unrestricted posts remain outside this checkpoint.

Build/install as in the [0.2.0 local artifact instructions](#installation-and-environment-compatibility),
using artifact version `0.3.0`; use `[planar]` for these engines. Python support and
the base/optional split are unchanged. The [compatibility policy](structure_spec.md#development-compatibility-policy)
applies within each documented minor version.

## 0.2.0 consumer checkpoint

Checkpoint date: 2026-10-26. This is the first explicitly scoped consumer baseline
after the unversioned development work carried under 0.1.0. It packages the accepted
native foundation and manufacturing composition without changing their runtime
behavior. See the [PK01 acceptance record](REVIEW.md#pk01-consumer-checkpoint---2026-10-26)
for fresh package checks and the exact reuse of CF02 evidence.

### Accepted consumer scope

An application can install the base library, author CAD geometry and native
Parts/MOPs, save editable `.cb` files, and reopen supported content for further
edits. Repeated components can be composed into caller-selected stock and separate
front/back outputs. The [README example](../README.md#start-with-the-library),
[native board](../demos/native_board.py), [cabinet](../demos/manufacturing_cabinet.py)
and [sensor fixture](../demos/manufacturing_fixture.py) are executable consumers.

| Public surface | Contract and boundary |
| --- | --- |
| `from cambam_builder import CBProject, CamBamProject, Vertex` | Both project names are the same canonical class. Documented project methods own geometry, hierarchy, groups, Part/MOP authoring, copying and export. |
| `cambam_builder.native.{project,core,cad,region,cam,reader,writer,transformations,transfer}` | Canonical native owners. Supported classes, methods and variants follow the [native coverage map](NATIVE_COVERAGE.md) and [capability map](structure_spec.md#capability-and-public-api-boundary-map), not every importable implementation symbol. |
| `from cambam_builder.native.manufacturing import SetupStock, SetupFrame, ComponentPlacement, compose_setup` | [Composition contract](structure_spec.md#reusable-manufacturing-composition-cf02): explicit stock, planar placement, face selection and operation order. Caller code owns design recipes and layout. |
| Root compatibility modules such as `cambam_builder.cambam_project`, `cambam_reader`, `cambam_writer`, and `cambam_entities` | Existing documented imports remain available and resolve to canonical native objects; see the complete [owner/migration map](structure_spec.md#package-organization-decision-and-migration-plan). Prefer native owners for new direct imports. |
| Top-level milling calculation, recommendation and planning exports; documented `planar`, `cam_core`, `cam_extensions` and `integrations` APIs | Retain their existing bounded [capability contracts](structure_spec.md#capability-and-public-api-boundary-map). Installation/import checks do not expand their geometry, stock or controller coverage. Backend-dependent engines require the planar extra. |

CF01 preserves units, owned PointList references and native operator Tags. NC01-A
adds editable Line/Spline interchange within the coverage map. CF02 preserves
native bindings and ordering through repeated and reflected/rotated placements.
Their detailed invariants and accepted limits stay in the linked contracts.

### Installation and environment compatibility

Python >=3.12 is required. The verified matrix is 3.12 and 3.13 on Windows; metadata
eligibility for newer Python versions is not a verification claim. Other operating
systems have not been certified by this checkpoint. NumPy is the base dependency;
neither Shapely, MCP nor an installed CamBam application is needed for base document
authoring. `planar` and `mcp` are separate opt-in extras; exact dependency constraints
remain in [pyproject.toml](../pyproject.toml).

Build from this version's checkout, then install a generated artifact into a
separate application environment. Replace the directory placeholder with a unique
task directory:

```powershell
uv build --out-dir output/<task>-<unique>/dist
python -m pip install ./output/<task>-<unique>/dist/cambam_builder-0.2.0-py3-none-any.whl
# Independent source install, or optional planar engines:
python -m pip install ./output/<task>-<unique>/dist/cambam_builder-0.2.0.tar.gz
python -m pip install "./output/<task>-<unique>/dist/cambam_builder-0.2.0-py3-none-any.whl[planar]"
```

Here `python` is the consuming application's interpreter. These are local artifact
paths, not registry installation instructions. For reproducible package checks use
the [packaging runbook](DEVELOPMENT.md#packaging-and-supported-python-validation).

### Upgrade and interchange notes

- There are no Python API removals or runtime changes between the accepted CF02
  tree and 0.2.0. Moving from an arbitrary earlier 0.1.0 checkout requires reviewing
  the native coverage and composition contracts: that development version covered
  multiple evolving trees and is not an identifiable compatibility baseline.
- The [compatibility policy](structure_spec.md#development-compatibility-policy)
  defines patch/minor upgrade expectations. Undocumented internals and old pickle
  snapshots have no cross-version compatibility guarantee. Use supported native
  `.cb` interchange for editable project exchange.
- Native compatibility is bounded by the recorded CamBam Plus 1.0
  [validation baseline](DEVELOPMENT.md#cambam-validation-baseline). Its XML
  `Version="0.9.8.0"` marker is preserved; it is not this package's version.
  No universal CamBam version or arbitrary unknown-XML preservation is claimed.
- Drawing units default to millimetres for new projects. Imported explicit or
  unspecified units retain their meaning without numeric conversion. Recompute
  affected derived paths and stock evidence after operator edits.
- Composition supports planar XY placement with unchanged Z within its contract,
  not arbitrary 3D workpiece flips. Manual-tab remapping, mixed document contexts,
  source machining origins/nesting and packing/workholding retain the explicit
  CF02 limits. Broader native shapes, general independent G-code, actual toolpaths,
  controller execution and physical machining need their own acceptance.
- MCP remains an optional bounded interface with its separate
  [protocol/coverage contract](MCP_CONTRACT.md). Shipping its launcher and resources
  does not imply parity with the direct library. The separately packaged
  `legacy_cambam_builder` is retained; new consumers should use `cambam_builder`.
