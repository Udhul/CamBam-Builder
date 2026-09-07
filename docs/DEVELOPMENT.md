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

The setup command may require dependency downloads. The initial review used the
available Python interpreter without installing or changing dependencies.

## Verification entry points

Run from the repository root, using the selected environment's Python:

```powershell
python -m compileall -q cambam_builder legacy_cambam_builder
python -c "from cambam_builder import CBProject; p = CBProject('smoke'); assert p.project_name == 'smoke'; print('import/construct OK')"
git diff --check
```

These are syntax/import checks, not a regression suite. There is currently no
canonical test-suite command. Add assertion-based regression tests with the next
runtime fix and document their command here. Packaging installation and supported
Python versions require separate validation; a local import does not prove them.

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
