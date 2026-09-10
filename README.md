# Fragment excited-state coupling calculations

Python and Fortran routines for calculating one-electron, two-electron, and overlap contributions between local excited states and charge-transfer states on molecular fragments A and B. A Bash launcher coordinates CFOUR calculations, TURBOMOLE Fock generation, and coupling post-processing over a distance scan.

**All project source dependencies identified by static inspection are included.** The package also contains example heatmap PDFs and repeatable syntax, import, and simulated-launcher checks.

## Example results

The following supplied PDFs illustrate the 10 × 10 overlap and coupling matrices for the formaldehyde dimer (DZ, full overlap treatment, as identified by the original filenames):

| Example | PDF |
| --- | --- |
| Overlap matrix heatmaps | [formaldehyde_dimer_DZ_10x10_overlap.pdf](examples/results/formaldehyde_dimer_DZ_10x10_overlap.pdf) |
| Coupling matrix heatmaps | [formaldehyde_dimer_DZ_10x10_coupling.pdf](examples/results/formaldehyde_dimer_DZ_10x10_coupling.pdf) |

These PDFs are included unchanged as example outputs. They were not regenerated or numerically validated against this packaged source version. Their inclusion does not establish that the current launcher directly produces these 10 × 10 plots.

The examples are stored in `examples/results/`, which is tracked by Git; the existing `/results/` ignore rule applies only to the top-level calculation-output directory.

## Source completeness

The final driver, `electrostatic.py`, and `final_codes/twoelint_calc_fortran.py` are now included. Static inspection found no further missing project modules. The active driver's `process_file(...)` call matches the supplied backend's parameter names.

The backend selects Fortran wrappers, imports the compiled extension `twoelint_f90`, and contains Python fallback paths when the extension is unavailable. All six Fortran source files are included; a compatible compiled extension must still be built for Fortran execution. 

CFOUR (`xcfour`) and the expected TURBOMOLE executable (`dscf_smp_fock`) must also be available on the target machine, including the export behavior that produces the coefficient, AO transformation, integral, and Fock files used below.

## Included files

| Path | Purpose |
| --- | --- |
| `run_coupling_calculations.sh` | Runs monomer and dimer calculations, collects transition densities, generates Fock data, invokes the final driver, and handles logs and cleanup. |
| `CT_coupling_mixed_method_new_deriv_4C_final.py` | Main driver: reads electronic-structure exports, prepares coefficients/overlaps, calls the one- and two-electron processors, and writes coupling results. |
| `electrostatic.py` | Geometry-block parsing, rotation, unit conversion, and additional electrostatic utilities. |
| `coupling_input_gen.py` | Generates CFOUR `ZMAT`/`ZMAT.IP`/`ZMAT.EA` and TURBOMOLE `coord` inputs. Uses the included `electrostatic.py`. |
| `new_mos_c4_to_tm.py` | Converts monomer orbitals into the `new_mos` file used by TURBOMOLE. |
| `cis_vector_fixer.py` | Corrects the coefficient-file header count in place. |
| `final_codes/onel_calc.py` | One-electron couplings, overlap, AO-to-MO transformation, effective Fock, and disconnected terms. |
| `final_codes/sao_cao_transform.py` | CFOUR/TURBOMOLE spherical/Cartesian AO transformations, ordering, and normalization. |
| `final_codes/calc_BAAA.py`, `calc_BBAA.py`, `calc_BBBA.py` | Python integral contractions for the three mixed-fragment categories. |
| `final_codes/twoelint_calc_fortran.py` | Active driver backend: 198-field schema, Fortran wrappers, shared-memory binary processing, and Python fallback paths. Included exactly as last uploaded. |
| `final_codes/twoelint_calc.py` | Binary streaming, shared AO vectors, classification, Python kernel dispatch, and reduction. |
| `twoelint_core_mod.f90` | Shared integral permutation/indexing utilities. |
| `twoelint_{aaaa,bbbb,baaa,bbaa,bbba}_zero_copy.f90` | Five Fortran contraction modules; braces here abbreviate five actual filenames. |
| `input_files/` | Supplied geometry, coupling settings, TURBOMOLE basis/control, and monomer AO transformations. |
| `requirements.txt` | Direct third-party Python imports; versions are not pinned or fully validated. |
| `scripts/check_project.py` | Repeatable syntax/import checks and entry point for the simulated launcher tests. |
| `tests/test_launcher_mock.py` | Four isolated launcher tests using temporary dummy programs and inputs. |
| `CHANGES.md` | Exact scope of the setup fixes and remaining limitations. |

The package removes attachment suffixes such as `(1)` and `(2)`. The launcher adds both the project root and `final_codes/` to `PYTHONPATH` to support package-qualified and legacy direct imports. The launcher makes `final_codes/twoelint_calc_fortran.py` importable as `twoelint_calc_fortran`.

## Input bundle and system consistency

| Input | Status and interpretation |
| --- | --- |
| `input_files/coupling_input` | Included; references the coefficient, orbital, overlap, and integral filenames generated by the launcher. Selects right coefficient index 1 and left index 2. These are read as coefficient selections; their correspondence to the exported states needs validation with real outputs. |
| `input_files/geom_input` | Included; header identifies `forma-forma`, `PVDZ`, distance unit `Angstrohm`, and monomer method `CCSD`. Contains geometry and excitation blocks plus reference tables. |
| `input_files/control` | Included; specifies cc-pVDZ, 8 atoms, 16 doubly occupied orbitals, and one SCF iteration. |
| `input_files/basis` | Included; contains cc-pVDZ definitions for H, C, and O. No nitrogen basis is included. |
| `input_files/saocao_mon_A.dat`, `saocao_mon_B.dat` | Included; index extents are 38 spherical and 40 Cartesian AOs for each monomer. |
| `input_files/coupling_input_CCSD`, `geom_input_DZ_pp_CCSD` | Alternative supplied inputs. The launcher uses the canonical `coupling_input` and `geom_input` names, not these alternatives automatically. |

The input bundle is consistent in these basic counts with a formaldehyde dimer, and the launcher defaults to `forma`/`forma`. Orbital/excitation ordering and AO transformations still need validation against actual program outputs.

For another molecular system, change both monomer names **and** the matching geometry, excitation specifications, basis, control, and transformation files. Do not reuse the fixed control file or AO transformations solely because a new geometry block is present.

The generator reads the first ten geometry-input lines positionally. In particular, line 6 controls distance units, lines 7–9 control rotation/shift settings, and line 10 supplies the CFOUR basis name. Keep this structure. The literal unit spelling `Angstrohm` triggers the generator's Angstrom-to-bohr conversion. Geometry coordinates are subsequently written with `UNIT=BOHR`.

## Setup

Use Linux/Bash, Python 3.10 or newer as a baseline, and the external calculation tools expected by the launcher. Record the versions actually validated on your machine.

Install the direct Python dependencies with the Python interpreter used for calculations:

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` includes all six direct Python dependencies: `numpy`, `pandas`, `psutil`, `threadpoolctl`, `scipy`, and `matplotlib`. The command above installs them together; a separate `psutil` installation command is unnecessary. Versions remain unpinned until the versions from a working calculation environment are recorded.

Make `xcfour`, `dscf_smp_fock`, and `bc` available on `PATH`. Optionally set `CFOUR_BIN` to the directory containing `xcfour`. The launcher uses absolute paths to its Python helpers and invokes them with `python3`; executable permission on those `.py` files is not required.

The launcher checks required inputs, executables, driver existence, and direct module availability before creating calculation directories. These checks do not prove compiled-extension compatibility or that the electronic-structure exports are configured correctly.

### Record the installed versions

On the calculation computer, activate the environment normally used for the scripts and run:

```bash
python3 - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError

print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print()
for package in ("numpy", "pandas", "psutil", "threadpoolctl", "scipy", "matplotlib"):
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
PY
```

After confirming that this environment runs the calculations successfully, use the six `package==version` lines to pin `requirements.txt`. Keep the Python version and executable information in your environment notes, outside the requirements file.

## Running

From the unpacked project directory containing `input_files/`:

```bash
bash run_coupling_calculations.sh
```

Alternatively, run the launcher by absolute path from a separate run directory containing its own `input_files/`.

The corrected default scan uses 2.0–5.0 in 0.2 increments, followed by 6.0, 7.0, 8.0, 9.0, and 10.0 (21 distances). Numerators are divided by 10. With the supplied input this gives distances in Angstrom. To select just 3.0 and 4.0:

```bash
DISTANCE_NUMERATORS="30 40" bash run_coupling_calculations.sh
```

| Setting | Meaning |
| --- | --- |
| `MONOMER_A`, `MONOMER_B` | Geometry-block names, each defaulting to `forma`. |
| `DISTANCE_NUMERATORS` | Space-separated nonnegative integers to be divided by 10. |
| `CFOUR_BIN` | Optional CFOUR executable directory prepended to `PATH`. |
| `CT_DRIVER` | Driver path; defaults to `CT_coupling_mixed_method_new_deriv_4C_final.py` in the project root. |
| `CT_EXTRA_PYTHONPATH` | Extra colon-separated module directories inserted by the logging wrapper. |
| `CT_LOG_LEVEL` | `INFO` by default; use `DEBUG` for more verbose logging. |
| `TWOELINT_WORKERS` | Explicit integral worker-process count. |
| `TWOELINT_OMP_THREADS` | OpenMP thread setting used by the processor. |
| `TWOELINT_BLAS_THREADS` | BLAS thread limit during parent AO-vector preparation. |
| `TWOELINT_AUTOTUNE` | The latest active `twoelint_calc_fortran.py` forces `1` at import. The two alternative processors were previously changed to respect an existing environment value. |

The workflow runs monomer A and B local excitation, IP, and EA jobs, collects coefficients and orbitals, then loops over dimer distances. Each distance runs CFOUR, generates TURBOMOLE orbitals and Fock data, and invokes the coupling driver from `results/` with:

```text
coupling_input DIST NBAS_A NBAS_B True
```

The final positional argument enables transition-density processing, as in the corrected launcher. Each local, IP, and EA monomer calculation must export `LTRANDENS_MO` and `RTRANDENS_MO`. The launcher copies them to the twelve files `LTRANDENS_A/B`, `RTRANDENS_A/B`, `LTRANDENS_IP_A/B`, `RTRANDENS_IP_A/B`, `LTRANDENS_EA_A/B`, and `RTRANDENS_EA_A/B` under `results/`. These are generated calculation outputs, not missing repository inputs.

Run in a dedicated calculation directory. The script still cleans calculation intermediates after successful steps using the original filename patterns, restricted to files in the five known working directories. It skips directories, so `twoel_results/` is preserved. It stops on command failure before subsequent copies/cleanup; earlier successfully completed steps may already have cleaned their intermediates. This is not a resume/restart implementation.

## Outputs

| Location | Contents |
| --- | --- |
| `mon_1/`, `mon_2/`, `dimer/`, `tm_fock/` | External-program working directories. |
| `results/CCSD_matrix_A.csv`, `CCSD_matrix_B.csv` | Local excitation coefficient files. |
| `results/CCSD_vector_IP_A.csv`, `CCSD_vector_IP_B.csv`, `CCSD_vector_EA_A.csv`, `CCSD_vector_EA_B.csv` | EA/IP coefficient files. |
| `results/NEWMOS_A`, `NEWMOS_B` | Monomer orbitals. |
| `results/onel_molecu_A.csv`, `onel_molecu_B.csv` | Monomer one-electron exports. |
| `results/out.mon_*`, `out.dimer_<dist>`, `out.tm_dim_<dist>` | Electronic-structure logs. The TURBOMOLE log is now also copied into results. |
| `results/logs/ct_<dist>.out`, `ct_<dist>.log` | Driver stdout and stderr/logging. |
| `results/onel_results/<term>.txt`, `twoel_results/<term>.txt` | Per-term contribution tables, written for each returned term, such as `31.txt`. |
| `results/results_S.txt` | State-overlap table, using the supplied `results_file_name = results` prefix. |
| `results/determ_MO.txt` | Determinant/normalization diagnostics. |
| `results/final_couplings_*.txt` | Coupling variants written by the driver's final assembly, including full and two-electron-only results. |
| `results/eigenvector_<dist>.txt` | Distance-specific eigenvector output in the driver's final analysis. |

Temporary dimer data include `twoel.bin` (copied from CFOUR `IIII`), `overlap.csv`, `onel_molecu.csv`, `fock.sao`, `saocao_dim.dat`, `AO2SO_dim.txt`, and `AO2SOINV_dim.txt`. These are overwritten per distance and removed after successful post-processing.

## Python and Fortran backends

| Processor | Active dispatch | Extension name | Declared result fields |
| --- | --- | --- | --- |
| `twoelint_calc_fortran.py` | Fortran wrappers with Python fallback paths; selected by the final driver | `twoelint_f90` | 198 |
| `twoelint_calc.py` | Python kernels | Lazily loads `twoelint_f90`, but active chunk processing calls Python | 198 |

All three processors cover `12`, `13`, `14`, `21`, `23`, `24`, `31`, `32`, `34`, `41`, `42`, and `43`. Their schemas and wrapper interfaces differ. Compiling the Fortran files does not change `twoelint_calc.py` to Fortran dispatch. The final driver selects `twoelint_calc_fortran.py`. Its 198-field schema matches the field count of the Python processor; equal field counts alone do not prove numerical agreement.

A GNU Fortran/F2PY build outline, to adapt and validate against the selected wrapper, is:

```bash
python3 -m numpy.f2py -c -m twoelint_f90 \
  twoelint_core_mod.f90 \
  twoelint_aaaa_zero_copy.f90 \
  twoelint_bbbb_zero_copy.f90 \
  twoelint_baaa_zero_copy.f90 \
  twoelint_bbba_zero_copy.f90 \
  twoelint_bbaa_zero_copy.f90 \
  --f90flags="-O3 -ffp-contract=off -fPIC -ffree-line-length-none"
```

This build was not executed. The alternative processor expects `-m twoelint_f90_2`, but changing the name alone does not prove source/wrapper compatibility. The supplied kernels contain no OpenMP parallel directives, so increasing OpenMP thread counts does not parallelize their loops. Multiprocessing and shared-memory capacity remain relevant.

The binary reader expects Fortran sequential-unformatted WRSEQ records: float64 values, packed uint64 indices, and a trailing int32/int64 `NUT`, with detected endianness and 4/8-byte record markers. The `threshold` argument is the fragment-A basis-index boundary, normally `NBAS_A`, not a magnitude cutoff. `NBAS` is used as the total basis-index bound.

`process_file()` requires prepared coefficient/orbital dictionaries and overlap blocks. It returns nested dictionaries `{term: {contribution_name: value}}`; `C` and `X` identify Coulomb and exchange contributions. The included driver performs final assembly and export; numerical conventions still need validation with reference calculations. Direct multiprocessing callers need an `if __name__ == "__main__":` guard.

## Utilities

```bash
python3 cis_vector_fixer.py CIS_vector.csv
python3 new_mos_c4_to_tm.py NBAS_A NBAS_B out.mon_A_local_exc out.mon_B_local_exc
```

The coefficient fixer changes only the second header field to `(N - 1) / 2` for `N` lines, requiring an integer result and a header with at least three comma-separated fields. It rewrites the file in place. The launcher skips it when the `mon calc method` block is exactly `CIS`.

The MO converter now uses its command-line arguments and local working files, and importing it no longer starts a calculation. Its existing orbital-occupation heuristic infers the HOMO boundary from the first positive orbital energy. That heuristic remains unchanged and needs review for negative virtual energies. `sao_cao_transform.py` remains principally an importable module; its standalone block is not a complete conversion CLI.

## Running the checks yourself

Run these commands from the project root with your calculation Python environment active. Install the direct dependencies for the real import checks:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/check_project.py
```

The command runs all three groups and returns exit status 0 when they pass, or 1 if any group fails. It prints the Python interpreter path so you can confirm which environment was checked. You can run each group separately:

| Check | Command | What it establishes |
| --- | --- | --- |
| Syntax | `python3 scripts/check_project.py syntax` | Python sources compile in memory and Bash accepts the launcher with `bash -n`. No calculation is executed. |
| Real imports | `python3 scripts/check_project.py imports` | Dependencies and importable project modules load in fresh Python processes; versions and source paths are printed when available. |
| Simulated launcher | `python3 scripts/check_project.py mock` | The real launcher passes arguments and handles output files/errors as expected when its calculation programs are replaced with temporary stand-ins. |

The syntax and simulated-launcher checks need Python and Bash, without CFOUR, TURBOMOLE, or the scientific Python dependencies. The import group uses the actual installed dependencies and never supplies mock replacements. `coupling_input_gen.py` runs its CLI work during import, so this check does not import it; it is included in syntax checking instead. The Fortran extension is loaded lazily by the backend and is not validated by ordinary module imports.

The four launcher tests check a successful one-distance run including twelve transition-density copies, failure of an external calculation, failure of the coupling driver with integral/Fock inputs retained, and rejection of a missing driver before work directories are created. The tests run in disposable temporary folders. Their stand-ins are created there at test time; your real calculation programs, inputs, and output directories are not used. Existing calculation-related environment variables are cleared for these simulated runs.

Failures report the relevant module or test. For example, `ModuleNotFoundError: No module named 'psutil'` means that the interpreter shown by the check cannot find this dependency. In that same environment, run `python3 -m pip install -r requirements.txt`, then repeat `python3 scripts/check_project.py imports`. Existing syntax warnings can be printed even when the sources parse successfully. A passing result does not establish correct molecular energies, coupling formulas, or Python/Fortran numerical agreement.


## Validation

The following results describe checks performed while preparing this package. Run the included checks on your calculation computer to verify its environment.

| Check | Observed result | Scope |
| --- | --- | --- |
| Python and Bash syntax | Passed | Sources have valid language structure; calculation logic is not assessed. |
| Real module imports | Passed | The three mixed Python kernels, one-electron/AO-transform modules, MO converter, and other checked modules loaded. |
| Simulated launcher | All four tests passed | Checked argument passing, all twelve transition-density copies, successful cleanup, early missing-driver rejection, and stopping on external-program or driver failure with available files retained. |
| Actual input generation | Passed for six launcher modes | Both monomers in local/CT modes and dimer CFOUR/TURBOMOLE modes generated inputs; the dimer contained eight atoms. No electronic-structure program was run. |
| Driver/backend interface | Static check passed | The active driver call uses parameters accepted by `process_file`. |
| Fortran compilation and numerical calculations | Not performed | No compiled-extension validation, CFOUR/TURBOMOLE calculation, or numerical Python/Fortran comparison was carried out. |

