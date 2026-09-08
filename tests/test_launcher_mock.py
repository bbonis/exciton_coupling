#!/usr/bin/env python3
"""Test launcher control flow with temporary stand-ins, never chemistry jobs.

Only the launcher and coefficient-header fixer are the real project scripts.
The input generator, MO converter, coupling driver, external executables, and
preflight module names are replaced inside a disposable temporary directory.
Actual dependency/module imports are tested separately by check_project.py.
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = 'CT_coupling_mixed_method_new_deriv_4C_final.py'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='coupling-launcher-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.pkg = self.base / 'project'
        self.pkg.mkdir()
        for name in ('run_coupling_calculations.sh', 'cis_vector_fixer.py'):
            shutil.copyfile(ROOT / name, self.pkg / name)
        for module in (
            'numpy', 'pandas', 'psutil', 'threadpoolctl', 'scipy', 'matplotlib',
            'electrostatic', 'twoelint_calc_fortran',
        ):
            (self.pkg / f'{module}.py').write_text('# Preflight stand-in used only in this temporary test.\n')
        self.write_script(self.pkg / 'coupling_input_gen.py', '''
            from pathlib import Path
            import sys
            if len(sys.argv) != 6 or sys.argv[2:4] != ['forma', 'forma']:
                raise SystemExit('Unexpected input-generator arguments')
            if sys.argv[4] not in ('mon_A', 'mon_B', 'dimer'):
                raise SystemExit('Unexpected fragment argument')
            if sys.argv[5] not in ('Frenkel', 'CT', 'TM'):
                raise SystemExit('Unexpected calculation mode')
            for name in ('ZMAT', 'ZMAT.IP', 'ZMAT.EA', 'coord'):
                Path(name).write_text('dummy input\\n')
        ''')
        self.write_script(self.pkg / 'new_mos_c4_to_tm.py', '''
            from pathlib import Path
            import sys
            expected = ['38', '38', 'out.mon_A_local_exc', 'out.mon_B_local_exc']
            if sys.argv[1:] != expected:
                raise SystemExit('Unexpected MO converter arguments')
            Path('new_mos').write_text('dummy orbitals\\n')
        ''')
        self.write_script(self.pkg / DRIVER, '''
            from pathlib import Path
            import os, sys
            expected = ['coupling_input', '3.00', '38', '38', 'True']
            if sys.argv[1:] != expected:
                raise SystemExit('Unexpected coupling-driver arguments')
            if not Path('twoel.bin').is_file() or not Path('fock.sao').is_file():
                raise SystemExit('Missing copied integral/Fock inputs')
            if os.getenv('COUPLING_TEST_FAILURE') == 'driver':
                raise SystemExit(9)
            Path('onel_results/mock.txt').write_text('completed')
        ''')
        self.bin = self.base / 'bin'
        self.bin.mkdir()
        # Always use the same interpreter as this test, regardless of PATH/venvs.
        python_wrapper = self.bin / 'python3'
        python_wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        python_wrapper.chmod(0o755)
        self.write_script(self.bin / 'xcfour', '''
            from pathlib import Path
            import os
            if os.getenv('COUPLING_TEST_FAILURE') == 'xcfour':
                Path('failure_marker').write_text('retained')
                raise SystemExit(7)
            print('There are 38 functions in the AO basis')
            for name in ('LTRANDENS_MO', 'RTRANDENS_MO', 'NEWMOS', 'AO2SO.txt',
                         'AO2SOINV.txt', 'onel_molecu.csv', 'overlap.csv', 'IIII'):
                Path(name).write_text('dummy export\\n')
            Path('CIS_vector.csv').write_text('header,0,third\\n1,2,3\\n1,2,3\\n')
        ''')
        self.write_script(self.bin / 'dscf_smp_fock', '''
            from pathlib import Path
            Path('fock.sao').write_text('dummy fock\\n')
            Path('saocao.dat').write_text('dummy transform\\n')
        ''')
        self.write_script(self.bin / 'bc', '''
            from decimal import Decimal
            import re, sys
            expression = sys.stdin.read().strip()
            match = re.fullmatch(r'scale=2;\\s*(\\d+)\\s*/\\s*(\\d+)', expression)
            if match is None:
                raise SystemExit('Unexpected distance expression')
            print(f'{Decimal(match[1]) / Decimal(match[2]):.2f}')
        ''')
        self.env = dict(os.environ)
        # User calculation settings must never redirect a test to real programs.
        for name in ('CFOUR_BIN', 'CT_DRIVER', 'CT_EXTRA_PYTHONPATH', 'PYTHONPATH',
                     'MONOMER_A', 'MONOMER_B', 'CT_LOG_LEVEL', 'COUPLING_TEST_FAILURE'):
            self.env.pop(name, None)
        self.env.update({
            'PATH': str(self.bin) + os.pathsep + os.defpath,
            'DISTANCE_NUMERATORS': '30',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONOPTIMIZE': '0',
        })
        self.run_dir = self.base / 'run'
        self.run_dir.mkdir()
        inputs = self.run_dir / 'input_files'
        inputs.mkdir()
        for name in ('coupling_input', 'saocao_mon_A.dat', 'saocao_mon_B.dat', 'control', 'basis'):
            (inputs / name).write_text('dummy test input\\n')
        (inputs / 'geom_input').write_text('mon calc method\nCCSD\n')

    @staticmethod
    def write_script(path, source):
        path.write_text('#!/usr/bin/env python3\n' + textwrap.dedent(source).lstrip())
        path.chmod(0o755)

    def launch(self, failure=''):
        env = dict(self.env)
        if failure:
            env['COUPLING_TEST_FAILURE'] = failure
        return subprocess.run(
            ['bash', str(self.pkg / 'run_coupling_calculations.sh')],
            cwd=self.run_dir, env=env, capture_output=True, text=True, timeout=30,
        )

    def test_success_copies_densities_passes_arguments_and_preserves_result_directory(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        outputs = self.run_dir / 'results'
        self.assertEqual((outputs / 'onel_results/mock.txt').read_text(), 'completed')
        self.assertTrue((outputs / 'twoel_results').is_dir())
        self.assertTrue((outputs / 'out.tm_dim_3.00').is_file())
        self.assertTrue((outputs / 'logs/ct_3.00.log').is_file())
        for frag in ('A', 'B'):
            for state in ('', 'IP_', 'EA_'):
                for side in ('L', 'R'):
                    self.assertTrue((outputs / f'{side}TRANDENS_{state}{frag}').is_file())
        self.assertEqual((outputs / 'CCSD_matrix_A.csv').read_text().splitlines()[0], 'header,1,third')
        self.assertFalse((outputs / 'twoel.bin').exists())

    def test_external_program_failure_stops_before_copying_or_cleanup(self):
        result = self.launch('xcfour')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.run_dir / 'mon_1/failure_marker').is_file())
        self.assertFalse((self.run_dir / 'results/CCSD_matrix_A.csv').exists())
        self.assertIn('Calculation failed', result.stderr)

    def test_driver_failure_retains_integral_and_fock_inputs(self):
        result = self.launch('driver')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.run_dir / 'results/twoel.bin').is_file())
        self.assertTrue((self.run_dir / 'results/fock.sao').is_file())
        self.assertFalse((self.run_dir / 'results/onel_results/mock.txt').exists())

    def test_missing_driver_is_rejected_before_creating_work_directories(self):
        (self.pkg / DRIVER).unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing driver/modules', result.stderr)
        self.assertFalse((self.run_dir / 'results').exists())
        self.assertFalse((self.run_dir / 'mon_1').exists())


if __name__ == '__main__':
    unittest.main()
