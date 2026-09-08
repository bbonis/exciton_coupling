#!/usr/bin/env python3
"""Run syntax, import, or simulated-launcher checks without a chemistry job."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tokenize

ROOT = Path(__file__).resolve().parents[1]


def check_syntax() -> bool:
    """Compile sources in memory and ask Bash to parse shell files."""
    paths = list(ROOT.glob('*.py'))
    shells = list(ROOT.glob('*.sh'))
    for folder in ('final_codes', 'scripts', 'tests'):
        paths.extend((ROOT / folder).rglob('*.py'))
        shells.extend((ROOT / folder).rglob('*.sh'))
    passed = True
    for path in sorted(set(paths)):
        try:
            with tokenize.open(path) as source:
                compile(source.read(), str(path), 'exec', dont_inherit=True)
        except (SyntaxError, UnicodeError, OSError) as error:
            print(f'FAIL syntax: {path.relative_to(ROOT)}: {error}', flush=True)
            passed = False
    bash = shutil.which('bash')
    if not bash:
        print('FAIL syntax: bash was not found on PATH.', flush=True)
        return False
    for path in sorted(set(shells)):
        result = subprocess.run([bash, '-n', str(path)], capture_output=True, text=True)
        if result.returncode:
            print(f'FAIL syntax: {path.relative_to(ROOT)}\n{result.stderr}', flush=True)
            passed = False
    if passed:
        print(f'PASS syntax: {len(set(paths))} Python files and {len(set(shells))} Bash files.', flush=True)
    return passed


IMPORT_MODULES = (
    'numpy', 'pandas', 'psutil', 'threadpoolctl', 'scipy', 'matplotlib',
    'final_codes.calc_BAAA', 'final_codes.calc_BBAA', 'final_codes.calc_BBBA',
    'final_codes.onel_calc', 'final_codes.sao_cao_transform',
    'final_codes.twoelint_calc',
    'final_codes.twoelint_calc_ov_binary_2_fortran_2_og',
    'twoelint_calc_fortran', 'electrostatic', 'cis_vector_fixer',
    'new_mos_c4_to_tm', 'CT_coupling_mixed_method_new_deriv_4C_final',
)


def check_imports() -> bool:
    """Import real modules in individual processes. Never insert mocks here."""
    passed = True
    with tempfile.TemporaryDirectory(prefix='coupling-import-check-') as td:
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(ROOT / 'final_codes')))
        if os.environ.get('PYTHONPATH'):
            env['PYTHONPATH'] += os.pathsep + os.environ['PYTHONPATH']
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['MPLBACKEND'] = 'Agg'
        env['MPLCONFIGDIR'] = td
        code = (
            'import importlib, sys; '
            'm = importlib.import_module(sys.argv[1]); '
            'print("PASS import:", sys.argv[1], '
            'getattr(m, "__version__", ""), "from", getattr(m, "__file__", "built-in"))'
        )
        for module in IMPORT_MODULES:
            try:
                result = subprocess.run(
                    [sys.executable, '-B', '-c', code, module], cwd=td, env=env,
                    capture_output=True, text=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                print(f'FAIL import: {module}: timed out after 30 seconds.', flush=True)
                passed = False
                continue
            if result.returncode:
                print(f'FAIL import: {module}\n{result.stdout}{result.stderr}', flush=True)
                passed = False
            else:
                print(result.stdout.strip(), flush=True)
        print(
            'NOTE: coupling_input_gen.py runs code at import time; it is covered by '
            'syntax checks, not imported here. Lazy loading means these import checks '
            'do not test the compiled twoelint_f90 extension.', flush=True,
        )
    return passed


def check_mock() -> bool:
    """Run the isolated unittest suite with stand-ins for external programs."""
    return subprocess.run(
        [sys.executable, '-B', str(ROOT / 'tests' / 'test_launcher_mock.py'), '-v'],
        cwd=ROOT,
    ).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', nargs='?', choices=('all', 'syntax', 'imports', 'mock'), default='all')
    args = parser.parse_args()
    print(f'Python: {sys.executable}\nProject: {ROOT}', flush=True)
    checks = {'syntax': check_syntax, 'imports': check_imports, 'mock': check_mock}
    selected = checks if args.check == 'all' else {args.check: checks[args.check]}
    passed = True
    for name, function in selected.items():
        print(f'\nRunning {name} checks...', flush=True)
        try:
            ok = function()
        except Exception as error:
            print(f'FAIL {name}: {error}', flush=True)
            ok = False
        passed = ok and passed
    print('\nChecks passed.' if passed else '\nSome checks failed; see the output above.', flush=True)
    print('These checks do not validate scientific results or Python/Fortran numerical agreement.', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
