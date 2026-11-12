# Stage 0 scenario baselines

These files pin the legacy scenario builder before the intervention migration. They use
stored model versions, not simulated or re-estimated fixtures:

| Model | Stored version |
| --- | --- |
| `bvar.sv` | `51ead9204774f7ef0b4657c4e1a3997c77c584a902ebb6d306c68439977658f8` |
| `msvar.markov` | `ecaa07d09611885aa4f98de116ea4b870fc31caa34788dafea3998e2e9ed57ab` |

Both versions were inspected through MacroStress `model(list)` and `model(describe)`.
They were estimated as of 2026-10-29; their panel ends in September 2026 and contains
23 macro variables. The paths include September 2026 (`t=0`) through September 2027.
Every case uses 256 draws, seed 20261112, horizon 12 and `inner=1`.

## Inputs and outputs

- [empty.json](empty.json) has no conditions or impacts.
- [shifts-only.json](shifts-only.json) has varying additive impacts on the policy rate
  and Brent, including an internal zero and months outside the impact range.
- [constraints-no-causes.json](constraints-no-causes.json) has a point rate condition
  and an interval on a weighted cross-month oil change, with no named causes.

The numbers are regression inputs, not an economic event or a corpus-derived scenario.
They are marked `assumption`; `approved_by` remains null. The tests call the production
`build` function directly, not the user-facing approval/calculation service.

Each `<version-prefix>-<case>.npz` stores all `macro` paths in reported own units and all
`shadow` paths in model scale. Each matching JSON stores ordered variable ids, calendar
months, draw count, seed and the entire price object, including its one-sided price.
Empty and shifts-only scenarios retain the existing absent-price representation (`null`),
not an invented zero. Array dtype, shape and bytes must match; price JSON uses round-trip
float serialization, not rounded numbers or tolerances.

An actual empty `SatelliteBundle` isolates the macro behavior. The all-zero satellite hash
in the old Spec inputs is a sentinel for that empty bundle, not a stored satellite version.
There are **no driver or position-price checks** in these baselines. The two macro versions
are loaded with production `mst.macro.version.load`; production code is not patched.

## Capture and verification

From the repository root, using the project's Python environment:

```powershell
.\.venv\Scripts\python.exe -m tests.scenario.golden.capture
.\.venv\Scripts\python.exe -m pytest tests/scenario/test_golden.py -q
.\.venv\Scripts\python.exe -m pytest tests/scenario/test_golden.py -q
.\.venv\Scripts\ruff.exe check tests/scenario/test_golden.py tests/scenario/golden/capture.py
```

[capture.py](capture.py) refuses to overwrite any baseline. Do not regenerate them to make
a migration pass; an unexplained mismatch blocks the next stage. The test never captures
or changes expectations.

By default, tests load `<repository>/workspace/models`. If either stored version is absent,
these external-data tests skip with the exact missing hashes; no synthetic model replaces
it. Set `MACROSTRESS_WORKSPACE` to another workspace root to require its versions: missing
versions in an explicitly selected workspace fail. Six passing tests (no skips) are needed
for local stage acceptance.

Capture environment: Windows 11 (10.0.26200), Python 3.12.14, NumPy 2.4.4, SciPy 1.17.1.
Bitwise expectations are tied to that environment and its numerical libraries. A platform
difference is a mismatch to investigate, not a reason to loosen the comparison.

The cross-month logarithmic difference is valid in the old contract. It cannot be translated
as an ordinary linear sum of own-unit prices. It is retained to expose that migration limit,
not replaced with an easier case after capture.

## Check evidence

- Before capture, the first BVAR empty test failed with `FileNotFoundError` for its missing
  NPZ, after building finite `(256, 13, 23)` paths from the real version.
- Capture produced all six finite `(256, 13, 23)` macro/shadow pairs. The capture was
  interrupted after five cases while the logarithmic-difference migration limit was
  discussed; those outputs and inputs were retained unchanged. The remaining MS-VAR
  constraints case was captured separately with exclusive-create writes.
- First verification: `6 passed in 75.91s`; second consecutive verification:
  `6 passed in 74.97s`. Neither run skipped a case.
- Constraints-only prices: BVAR `7.872804342723472`; MS-VAR `6.7025265785330745`.
  Empty and shifts-only prices are `null` for both models.
- Ruff passed. Additional checks confirmed capture refuses existing outputs; missing
  default versions skip whereas missing explicitly selected versions fail; shifted and
  constrained paths differ from the empty paths; months outside the shifts are unchanged
  bitwise; both support conditions hold on every saved constrained path.
- No production edits or commits were made. Coverage/CRAP, scored duplication and mutation
  tools (`coverage`, `radon`, `jscpd`, `mutmut`) were unavailable in the project environment;
  no metric or mutation score is claimed.
