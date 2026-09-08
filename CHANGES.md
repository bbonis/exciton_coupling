# Changes in this package

Original uploads were retained separately. This package uses canonical filenames and the latest uploaded one-electron and AO-transformation modules.

## Fixed

- Created `final_codes/` with `__init__.py` to satisfy existing package imports.
- Removed unused `from typeguard import value` from `calc_BAAA.py`.
- Changed both integral processors to respect an existing `TWOELINT_AUTOTUNE` value.
- Removed the first of two duplicate shared-vector allocations in each processor, keeping the BLAS-limited preparation.
- Replaced uninitialized-local overlap fallbacks in `twoelint_calc.py` with explicit required-key validation. Valid, fully supplied vector calculations retain their expressions. The alternative processor's argument-based fallback behavior was retained.
- Disabled forced MO-converter debug execution, added an argument-count check, and moved executable work into a guarded `main()`; conversion formulas and occupation selection were retained.
- Added input-generator argument-count validation and stripped trailing whitespace from the basis name.
- Used raw regex strings for the two AO-transform whitespace separators, preserving the regex meaning.
- Removed machine-specific launcher paths. Added project-relative helper/module resolution and optional `CFOUR_BIN`/`CT_DRIVER` settings.
- Set launcher default monomers to `forma`/`forma`, matching the supplied formaldehyde input bundle. Exposed `MONOMER_A`, `MONOMER_B`, and `DISTANCE_NUMERATORS` overrides.
- Added required-file/program/module checks and command-failure handling before later copying or cleanup.
- Removed invalid `$!`/`wait` usage after foreground calculations.
- Made directory creation repeat-safe and AO basis-size extraction validate its result.
- Restricted cleanup to known calculation directories and file/symlink entries; directory matches such as `twoel_results/` are skipped. Original cleanup filename patterns remain.
- Made `CT_LOG_LEVEL` configurable, defaulting to `INFO`.
- Copied distance-specific TURBOMOLE output into `results/`.
- Added direct dependency list, generated-file ignore rules, and updated README.

## Not changed or reconstructed

No Coulomb/exchange permutations, left/right scientific conventions, Fortran contractions, result schemas, occupation heuristics, or supplied numerical input data were changed. The missing driver and `electrostatic.py` were not invented. External electronic-structure exports and numerical correctness still need validation with those sources and real reference calculations.

## Latest supplied driver and launcher

- Included `CT_coupling_mixed_method_new_deriv_4C_final.py` and `electrostatic.py` under canonical names.
- Corrected the driver's AO-transform import to the supplied package module and replaced two `delim_whitespace=True` calls with the equivalent `sep=r"\s+"`. Added a five-argument CLI check.
- Preserved the corrected launcher's formaldehyde selection, all twelve transition-density copies, `True` driver argument, and 21-point distance grid. Retained prior setup checks, logging, configurable paths, and guarded cleanup; no invalid background-process waits were reintroduced.
- Added SciPy and Matplotlib to the direct requirements and launcher checks.
- Kept the driver's `twoelint_calc_fortran` import unchanged: this exact backend is still missing, and the supplied alternatives have not been substituted.
- Verified real input generation in six launcher modes, alongside syntax and simulated-launcher checks. No real electronic-structure or complete coupling calculation was run.

The earlier statement about the missing driver/electrostatic module describes the previous package revision; both are now included. `twoelint_calc_fortran.py` is the remaining identified source dependency.

## Final backend supplied

- Included `final_codes/twoelint_calc_fortran.py` exactly as uploaded.
- Static inspection found all its project imports and Fortran module sources in the package; no further missing project source dependency was identified.
- Confirmed the active driver call uses parameters accepted by `process_file`.
- Updated the README to distinguish the active Fortran-wrapper processor from the two alternatives, including its retained autotune/shared-memory behavior.
- The preceding missing-backend notes describe the earlier revision and are now resolved. A compiled extension and real calculation remain unvalidated.

## Example output PDFs

Added the supplied 10 × 10 formaldehyde dimer overlap and coupling heatmap PDFs under `examples/results/`, with descriptive filenames and README links. PDF bytes were preserved and checked against the uploads. No calculation code or numerical results were modified.

## Repeatable project checks

Added `scripts/check_project.py` and `tests/test_launcher_mock.py` for syntax checks, real dependency/module imports, and four isolated launcher simulations. Added run instructions and an explanation of `psutil` to the README. Syntax and all four mock tests passed; import checks explicitly report the review environment's missing `psutil`. Calculation code and all supplied PDF examples remain unchanged.
