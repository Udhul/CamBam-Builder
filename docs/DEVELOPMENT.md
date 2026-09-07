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

## Troubleshooting

- Import failure: verify the selected interpreter and install declared dependencies
  through that interpreter; do not change package metadata merely to suit a shell.
- Successful export with missing content: inspect logged errors and XML structure;
  writer exception handling can continue after individual serialization failures.
- Consult [review evidence](REVIEW.md) before retrying a known approach.
