#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from numbers import Real
from typing import Any, Dict, List, Tuple, Iterator, Optional, Sequence, Union
from collections.abc import Mapping

import math
import sys
import numpy as np
import pandas as pd
from final_codes import sao_cao_transform as soa_cao_trasnform
import os.path

import multiprocessing as mp
import time
import psutil
import logging
import scipy.linalg

import twoelint_calc_fortran as tc
#import twoelint_calc_ov_binary_2_fortran_2 as tc_ov
#import twoelint_calc_dbg_version as tc_dbg
import matplotlib.pylab as plt
#import twoel_transform as tt
import re

# Add the directory containing the desired onel.py to the front of sys.path
#desired_onel_path = "/home/bonis/python_scripts/executables/onel_calc.py"
#if desired_onel_path not in sys.path:
#    sys.path.insert(0, desired_onel_path)
#
# Now, import onel.py
import onel_calc as oc
from typing import Dict, Union

# Drop-in replacements for your two functions + small helpers.
# Keep this in the same module where you call them.



@contextmanager
def _file_lock(fp):
    """
    Best-effort POSIX advisory lock; no-op on platforms without fcntl.
    Prevents concurrent writers from corrupting header/rows.
    """
    try:
        import fcntl  # type: ignore
    except Exception:
        yield
        return

    fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def _read_header_columns(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    if not first:
        return []
    parts = first.split()
    if not parts:
        return []
    # Your format: "Distance   <k1>     <k2> ..."
    if parts[0].lower() == "distance":
        return parts[1:]
    return parts


def _format_value(v: Any, precision: int, missing_value: str) -> str:
    if v is None:
        return missing_value
    # bool is int subclass; keep as int-ish
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, Real) and not isinstance(v, int):
        return f"{float(v):.{precision}f}"
    return str(v)


def _rewrite_with_extended_header(
    filepath: str,
    old_cols: List[str],
    new_cols: List[str],
    sep: str,
) -> None:
    """
    Rewrite existing file to include new header columns, padding old rows.
    Preserves old numeric formatting by copying token strings verbatim.
    """
    dirpath = os.path.dirname(filepath) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=dirpath, text=True)
    os.close(fd)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as src, open(
            tmp_path, "w", encoding="utf-8"
        ) as dst:
            # Consume old header
            _ = src.readline()

            # Write new header
            dst.write("Distance" + sep + sep.join(new_cols) + "\n")

            # Rewrite rows
            for line in src:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if not tokens:
                    continue
                dist = tokens[0]
                vals = tokens[1:]
                mapping = {k: (vals[i] if i < len(vals) else "") for i, k in enumerate(old_cols)}
                row_vals = [mapping.get(k, "") for k in new_cols]
                dst.write(dist + sep + sep.join(row_vals) + "\n")

        os.replace(tmp_path, filepath)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _coerce_twoel_results(
    results: Any,
    *,
    result_fields: Optional[Sequence[Tuple[int, str]]] = None,
) -> Dict[int, Dict[str, Union[float, int]]]:
    """
    Normalize results into the nested dict format:
      {term: {key: value}}

    Accepts:
      - already nested dict: Dict[int, Dict[str, ...]]
      - flat dict keyed by (term, key)
      - flat vector/tuple/np.ndarray aligned with RESULT_FIELDS
      - (vec, used_mask) where used_mask is boolean aligned with RESULT_FIELDS
    """
    if isinstance(results, Mapping):
        # Already nested?
        if all(isinstance(k, int) for k in results.keys()):
            return dict(results)  # type: ignore

        # Dict[(term,key)] -> nested
        if all(isinstance(k, tuple) and len(k) == 2 for k in results.keys()):
            out: Dict[int, Dict[str, Union[float, int]]] = {}
            for (term, key), val in results.items():  # type: ignore
                out.setdefault(int(term), {})[str(key)] = float(val)  # type: ignore
            return out

        raise TypeError("Unsupported mapping shape for results")

    fields = result_fields
    if fields is None:
        fields = globals().get("RESULT_FIELDS", None)
    if fields is None:
        raise ValueError("RESULT_FIELDS not available; pass result_fields=...")

    # (vec, used_mask)
    if isinstance(results, tuple) and len(results) == 2:
        vec, used = results
        vec = np.asarray(vec, dtype=np.float64)
        used = np.asarray(used)
        if used.dtype != bool:
            used = used.astype(bool, copy=False)

        out: Dict[int, Dict[str, Union[float, int]]] = {}
        n = min(len(vec), len(fields), len(used))
        for i in range(n):
            if not bool(used[i]):
                continue
            term, key = fields[i]
            out.setdefault(int(term), {})[str(key)] = float(vec[i])
        return out

    # flat vector / tuple
    vec = np.asarray(results, dtype=np.float64)
    out: Dict[int, Dict[str, Union[float, int]]] = {}
    n = min(len(vec), len(fields))
    for i in range(n):
        term, key = fields[i]
        out.setdefault(int(term), {})[str(key)] = float(vec[i])
    return out


def write_results_to_file(
    results: Dict[str, Union[float, int]],
    filepath: str,
    distance: int = 10000000,
    precision: int = 24,
    *,
    sep: str = "     ",
    missing_value: str = "",
    on_new_keys: str = "extend",  # "extend" | "error" | "ignore"
) -> None:
    """
    Append a row of results to a whitespace-tabulated file.
    Guarantees column alignment by using the file header column order.

    Behavior:
      - First write creates header with sorted keys.
      - Later writes follow header order (fill missing keys with missing_value).
      - If new keys appear:
          - extend (default): rewrite file with new header + padded old rows
          - error: raise
          - ignore: drop extra keys
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    file_exists = os.path.isfile(filepath)
    file_empty = (not file_exists) or os.path.getsize(filepath) == 0

    if file_empty:
        cols = sorted(results.keys())
        with open(filepath, "a", encoding="utf-8") as f, _file_lock(f):
            if os.path.getsize(filepath) == 0:
                f.write("Distance" + sep + sep.join(cols) + "\n")
            row = [ _format_value(results.get(k), precision, missing_value) for k in cols ]
            f.write(str(distance) + sep + sep.join(row) + "\n")
        return

    # Non-empty file: respect existing header
    with open(filepath, "r+", encoding="utf-8", errors="ignore") as f, _file_lock(f):
        cols = _read_header_columns(filepath)

        extra = [k for k in results.keys() if k not in cols]
        if extra:
            if on_new_keys == "error":
                raise KeyError(f"New keys not in header: {extra}")
            if on_new_keys == "extend":
                new_cols = cols + sorted(extra)
                _rewrite_with_extended_header(filepath, cols, new_cols, sep)
                cols = new_cols
            # on_new_keys == "ignore": keep existing cols and drop extras

    # Append row in header order
    with open(filepath, "a", encoding="utf-8") as f, _file_lock(f):
        row = [_format_value(results.get(k), precision, missing_value) for k in cols]
        f.write(str(distance) + sep + sep.join(row) + "\n")


def write_twoel_results(
    nested_results: Any,
    base_filepath: str,
    distance: int = 10000000,
    precision: int = 24,
    *,
    result_fields: Optional[Sequence[Tuple[int, str]]] = None,
) -> None:
    """
    Accepts nested dict results OR vector/tuple results aligned with RESULT_FIELDS.
    Writes each term to: f"{base_filepath}{term}.txt"
    """
    #normalized = _coerce_twoel_results(nested_results, result_fields=result_fields)

    for term, sub_results in nested_results.items():
        distance_filepath = f"{base_filepath}{term}.txt"
        write_results_to_file(
            sub_results,
            distance_filepath,
            distance=distance,
            precision=precision,
        )


def read_file_to_matrix(file_name: str, num_of_indices: int):
    df = pd.read_csv(file_name, header=None).to_numpy(dtype=np.float64)
    if num_of_indices == 2:
        df_type = np.array([int(0), int(0), np.float64(
            0.0)])  # https://stackoverflow.com/questions/27098529/numpy-float64-vs-python-float ez alapján jó a sima float?
    elif num_of_indices == 4:
        df_type = np.zeros((1, 5))
    else:
        print(f'Wrong number of indices')
    for i in range(len(df)):
        if num_of_indices == 2:
            df_type_row = np.array([int(df[i, 0]), int(df[i, 1]), np.float64(df[i, 2])])
            df_type = np.row_stack((df_type, df_type_row))
        elif num_of_indices == 4:
            if df[i, :].all() == (0 or 0.):
                break
            df_type_row = np.array([int(df[i, 0]), int(df[i, 1]), int(df[i, 2]), int(df[i, 3]), np.float64(df[i, 4])])
            df_type = np.row_stack((df_type, df_type_row))
    df_type = np.delete(df_type, 0, 0)
    return df_type



def create_matrix(raw_matrix, num_of_indices: int, N_SAO: int):
    matrix_size = int(np.max(raw_matrix))
    prefactor = np.float64(1.)
    if num_of_indices == 2:
        result_matrix = np.zeros((matrix_size, matrix_size))
        matrix_2 = np.zeros((matrix_size, matrix_size))
        matrix_C = np.zeros((matrix_size, matrix_size))
        matrix_X = np.zeros((matrix_size, matrix_size))
    elif num_of_indices == 4:
        sys.exit('this part is commented out')
        result_matrix = np.zeros((matrix_size, matrix_size, matrix_size, matrix_size))
        matrix_2 = np.zeros((matrix_size, matrix_size, matrix_size, matrix_size))
        matrix_C = np.zeros((matrix_size, matrix_size, matrix_size, matrix_size))
        matrix_X = np.zeros((matrix_size, matrix_size, matrix_size, matrix_size))
    for i in range(len(raw_matrix)):
        if num_of_indices == 2:
            j = int(raw_matrix[i, 0] - 1)
            k = int(raw_matrix[i, 1] - 1)
            value = np.float64(raw_matrix[i, 2])
            result_matrix[j, k] = value
        elif num_of_indices == 4:
            sys.exit('this part is commented out')
            j = int(raw_matrix[i, 0] - 1)
            k = int(raw_matrix[i, 1] - 1)
            l = int(raw_matrix[i, 2] - 1)
            m = int(raw_matrix[i, 3] - 1)
            value = np.float64(raw_matrix[i, 4])
 #           if result_matrix[j, k, l, m] == 0. and result_matrix[k, j, l, m] == 0. and result_matrix[j, k, m, l] == 0. and result_matrix[j, k, m, l] == 0.:
  #              result_matrix[j, k, l, m] = value
            # for the in-core part, to debug the two electron terms
   #             result_matrix[k, j, l, m] = value
    #            result_matrix[j, k, m, l] = value
     #           result_matrix[k, j, m, l] = value
            #else:
             #   print('Something is wrong!!! This place has already been filled up')
              #  break
            if (j > k and l > m and j > m and l > k and j >= N_SAO > k and l >= N_SAO > m):
                if j == l:
                    if k == m:
                        prefactor = np.float64(1.)
                    else:
                        prefactor = np.float64(1.)  
                matrix_2[j, k, l, m] = -prefactor * value
                matrix_2[k, j, l, m] = -prefactor * value
                matrix_2[j, k, m, l] = -prefactor * value
                matrix_2[k, j, m, l] = -prefactor * value
                matrix_2[l, m, j, k] = -prefactor * value
                matrix_2[m, l, j, k] = -prefactor * value
                matrix_2[l, m, k, j] = -prefactor * value
                matrix_2[m, l, k, j] = -prefactor * value
                matrix_X[j, k, l, m] = -prefactor * value
                matrix_X[k, j, l, m] = -prefactor * value
                matrix_X[j, k, m, l] = -prefactor * value
                matrix_X[k, j, m, l] = -prefactor * value
                matrix_X[l, m, j, k] = -prefactor * value
                matrix_X[m, l, j, k] = -prefactor * value
                matrix_X[l, m, k, j] = -prefactor * value
                matrix_X[m, l, k, j] = -prefactor * value
                prefactor = np.float64(1.)
                prefactor = np.float64(1.)
                result_matrix[j, k, l, m] = -value
            # for the in-core part, to debug the two electron terms
                result_matrix[k, j, l, m] = -value
                
                result_matrix[j, k, m, l] = -value
                
                result_matrix[k, j, m, l] = -value
                result_matrix[l, k, j, m] = -value
                result_matrix[k, l, j, m] = -value
                result_matrix[l, k, m, j] = -value
                result_matrix[k, l, m, j] = -value
                if j == l:
                    result_matrix[j, k, j, m] = -value
                    result_matrix[k, j, j, m] = -value
                    result_matrix[j, k, m, j] = -value
                    result_matrix[k, j, m, j] = -value
                    result_matrix[j, m, j, k] = -value
                    result_matrix[m, j, j, k] = -value
                    result_matrix[j, m, k, j] = -value
                    result_matrix[m, j, k, j] = -value
            elif (j > l and k > m and j > m and k > l and j >= N_SAO > l and k >= N_SAO > m):
                if j == k:
                    if l == m:
                        prefactor = np.float64(1.)
                    else:
                        prefactor = np.float64(1.)                    
                matrix_2[j, k, l, m] = prefactor * value
                matrix_2[k, j, l, m] = prefactor * value
                matrix_2[j, k, m, l] = prefactor * value
                matrix_2[k, j, m, l] = prefactor * value
                matrix_2[l, m, j, k] = prefactor * value
                matrix_2[m, l, j, k] = prefactor * value
                matrix_2[l, m, k, j] = prefactor * value
                matrix_2[m, l, k, j] = prefactor * value
                matrix_C[j, k, l, m] = prefactor * value
                matrix_C[k, j, l, m] = prefactor * value
                matrix_C[j, k, m, l] = prefactor * value
                matrix_C[k, j, m, l] = prefactor * value
                matrix_C[l, m, j, k] = prefactor * value
                matrix_C[m, l, j, k] = prefactor * value
                matrix_C[l, m, k, j] = prefactor * value
                matrix_C[m, l, k, j] = prefactor * value
                prefactor = np.float64(1.)
                result_matrix[j, k, l, m] = value
            # for the in-core part, to debug the two electron terms
                result_matrix[k, j, l, m] = value
                result_matrix[j, k, m, l] = value
                result_matrix[k, j, m, l] = value
                result_matrix[l, m, j, k] = value
                result_matrix[m, l, j, k] = value
                result_matrix[l, m, k, j] = value
                result_matrix[m, l, k, j] = value
                if j == k:
                    result_matrix[j, j, l, m] = value
                    result_matrix[j, j, m, l] = value
                    result_matrix[l, m, j, j] = value
                    result_matrix[m, l, j, j] = value
                elif m == l:
                    result_matrix[m, m, k, j] = value
                    result_matrix[m, m, j, k] = value
                    result_matrix[l, m, j, j] = value
                    result_matrix[m, l, j, j] = value
    if num_of_indices == 4:
        np.allclose(result_matrix, matrix_2)
    return result_matrix, matrix_2, matrix_X, matrix_C


def CIS_VECTOR_INTERPRETER(file_name: str, num_of_exc: int, output_file: 'str', CIS_type: 'str'):
    df = pd.read_csv(file_name).to_numpy(dtype=np.float64)
    header_row = pd.read_csv(file_name, index_col=0, nrows=0).columns.tolist()
    CIS_v_length = int(header_row[0])
    CIS_matrix = []
    CIS_vector = []
    print('!!!!!!!! HARDCODED for frozen core!!!!!!!!!!!!!')
    print('!!!!!!!!!!! HARDCODED FOR SYMMERY=OFF!!!!!!!!!!!')
    if CIS_type == 'CT':
        IP = 'IP'
        EA = 'EA'
        CIS_vector = np.array([0.])
        if IP in file_name:
            row_num = int((num_of_exc - 1) * CIS_v_length)
            tmp_CIS = df[row_num:(row_num + CIS_v_length), :]
            # now get the LUMO which is the cont. orb.
            # '!!!!!Az jobb deszkriptor, aminek a pályaenergiája pontosan 0!!!!!!!!!!!!4')
            IP_orb = int(get_cont_orb(output_file))
            CIS_vector_IP = CIS_vector
            for i in range(len(tmp_CIS)):  # ha occupied, akkor beleteszi
                if tmp_CIS[i, 1] == IP_orb:
                    CIS_vector_IP = np.vstack((CIS_vector_IP, tmp_CIS[i, 2]))
            for j in range(0, int(min(tmp_CIS[:, 0]) - 1)):
                CIS_vector_IP = np.vstack((np.array(0), CIS_vector_IP))
            for k in range(IP_orb, int(max(tmp_CIS[:, 1]))):
                CIS_vector_IP = np.vstack((CIS_vector_IP, np.array(0.)))
            CIS_vector_2 = CIS_vector_IP
            print(
                '!!!!this is really bad, but I filled the exc. orbitals with zeros to match the overlap dimensions!!!!!!!')
            print('!!!ha van befagyasztott pálya két index között egy befagyasztott, az nem müködik!!!!!')
            CIS_vector_2 = np.delete(CIS_vector_2, 0, axis=0)
            CIS_vector = np.sign(CIS_vector_2[np.abs(CIS_vector_2).argmax()])*CIS_vector_2
            
            return CIS_vector
        elif EA in file_name:
            row_num = int((num_of_exc - 1) * CIS_v_length)
            tmp_CIS = df[row_num:(row_num + CIS_v_length), :]
            # now get the LUMO which is the cont. orb.
            EA_orb = get_cont_orb(output_file)
            CIS_vector_EA = np.array(0.)
            for i in range(len(tmp_CIS)):
                if tmp_CIS[i, 0] == EA_orb:  # ha virtual, akkor beleteszi
                    CIS_vector_EA = np.vstack((CIS_vector_EA, tmp_CIS[i, 2]))
                else:
                    continue
            # ez csinálja a felkiáltó jeleket
            for j in range(0, EA_orb - 1):
                CIS_vector_EA = np.vstack((np.array(0.), CIS_vector_EA))
            CIS_vector_2 = CIS_vector_EA
            CIS_vector_2 = np.delete(CIS_vector_2, 0, axis=0)
            print(
                '!!!!this is really bad, but I filled the occ. orbitals with zeros to match the overlap dimensions!!!!!!!')
            CIS_vector = np.sign(CIS_vector_2[np.abs(CIS_vector_2).argmax()])*CIS_vector_2
            return CIS_vector
        else:
            print(f'!!!!!!File name must contain {IP} or {EA} !!!!!!!!!!')
            sys.exit()
    elif CIS_type == 'exc':
        #ROW: from (occ) COLUMN: to orbitals (virt)
        row_num = int((num_of_exc - 1) * CIS_v_length)
        tmp_CIS = df[row_num:(row_num + CIS_v_length), :]
        N_occ = int(max(tmp_CIS[:, 0]))
        N_virt = int(max(tmp_CIS[:, 1]))
        CIS_matrix_2 = np.zeros((N_virt, N_virt), dtype=np.float64)
        for row_1 in range(len(tmp_CIS)):
            value = tmp_CIS[row_1, 2]
            N_from_orb = int(tmp_CIS[row_1, 0]) - 1
            N_to_orb = int(tmp_CIS[row_1, 1]) - 1
            CIS_matrix_2[N_from_orb, N_to_orb] = value
        CIS_matrix = np.sign(CIS_matrix_2.flat[np.abs(CIS_matrix_2).argmax()])*CIS_matrix_2
        return CIS_matrix

# CIS vector, sor vektorként!!! NEWMOS ban a egy oszlop jelent egy MO-t


def process_NEWMOS(file_name: str, NBAS: int, modes: str):  # NBAS means number of basis functions
    function_modes = ['IP', 'EA', 'None']
    if modes not in function_modes:
        raise ValueError("Invalid mode type. Expected one of: %s" % function_modes)
    with open(file_name) as file:
        LCAO_raw = pd.read_csv(file, delimiter='   ', header=None, engine='python').to_numpy(dtype=np.float64)
    # a file elrendezés, ha jól értem: 1 MO --> 12 Ao egymás alá, 2. MO  (2 oszlop) --> 12 Ao ...
    # az első  alatt kezdődik az 5.
    if modes == 'IP' or modes == 'EA':
        NBAS = NBAS + 1
    else:
        NBAS = NBAS
    Nstacked = int(np.float64(len(LCAO_raw)) / np.float64(NBAS))
    NBAS = int(NBAS)
    proc_LCAO = LCAO_raw[0:NBAS, :]
    for i in range(1, Nstacked):
        tmp = LCAO_raw[(i * NBAS):((i + 1) * NBAS), :]
        if i == Nstacked - 1:  # LCAO_LCAO_T if the last block is full, and correct the cours accordingly
            for j in range(0, 4):
                if math.isnan(tmp[0, j]):
                    tmp = tmp.copy()[:, 0:j]
                    break
        proc_LCAO = np.hstack((proc_LCAO, tmp))
    if modes == 'IP':
        proc_LCAO = np.delete(proc_LCAO, obj=0, axis=1)
        proc_LCAO = np.delete(proc_LCAO, obj=-1, axis=0)
    elif modes == 'EA':
        proc_LCAO = np.delete(proc_LCAO, obj=-1, axis=1)
        proc_LCAO = np.delete(proc_LCAO, obj=-1, axis=0)
    else:
        proc_LCAO = proc_LCAO
    return proc_LCAO


def get_MO_fock(output_file: str, NBAS: int):
    found_it = False
    MO_fock  = np.zeros((NBAS, NBAS), dtype=np.float64)
    with open(output_file) as f:
        for lines in f:
            if lines.find('ORBITAL EIGENVALUES') != -1:
                found_it = True
            elif found_it and lines.find('VSCF finished.') != -1:
                break
            elif len(lines.split()) == 0:
                continue
            elif found_it and lines.split()[0].isdigit():
                MO_fock[int(lines.split()[0]) - 1, int(lines.split()[0]) - 1] = lines.split()[2]
            else:
                continue
    return MO_fock


def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append((line_number))
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results


def get_cont_orb(file: str):# returns eiganvalue ordering!
    is_it_close = False
    split_row = False
    orb_num = 0
    table_row = ['NOPE']
    with open(file, 'r') as f:
        for line in f:
            # CAREFUL at this point I decided to look at the symmetries, which always will be 'XXXX' or 'empty'. 
            # HAD TO BE CHANGED to 0.00000000 since the XXXX do not appear in some cases
            # if the keyword 'CONTINUUM=...' is used
            if 'ORBITAL EIGENVALUE' in line:
                is_it_close = True
                split_row = True
            elif split_row:
                table_row = line.split()
                #if len(table_row) < 6:
                 #   continue
                if (table_row
                    and table_row[0].isdigit()
                    and is_it_close
                    and table_row[2] in {'0.0000000000', '-0.0000000000'}
                    and table_row[3] in {'0.0000000000', '-0.0000000000'}):
                    orb_num = int(table_row[0])
                    break
                else:
                    continue
            else:
                continue
    return orb_num


def input_reader(file_name):
    df = pd.read_csv(file_name, sep=r"\s+").to_numpy(dtype=str)
    header_row = pd.read_csv(file_name, nrows=1, sep=r"\s+").columns.tolist()
    input_values = [str(0)] * 33
    output_counter = 0.
    cis_coeff_counter = 0.
    exc_num_counter = 0.
    calc_type = 'asd'
    for i in range(len(df)):
        if df[i, 0] == 'calc_type':
            input_values[0] = str(df[i, 2])
            calc_type = str(df[i, 2])
        elif df[i, 0] == 'overlap_file':
            input_values[3] = str(df[i, 2])
        elif df[i, 0] == 'cross_onel_file':
            input_values[26] = str(df[i, 2])
        elif df[i, 0] == 'twoelint_file':
            input_values[4] = str(df[i, 2])
        elif df[i, 0] == 'output_file_A':
            output_counter = output_counter + 1
            input_values[5] = str(df[i, 2])
        elif df[i, 0] == 'output_file_EA_A':
            output_counter = output_counter + 1
            input_values[6] = str(df[i, 2])
        elif df[i, 0] == 'output_file_B':
            output_counter = output_counter + 1
            input_values[7] = str(df[i, 2])
        elif df[i, 0] == 'output_file_EA_B':
            output_counter = output_counter + 1
            input_values[8] = str(df[i, 2])
        elif df[i, 0] == 'CIS_matrix_A':
            cis_coeff_counter = cis_coeff_counter + 1
            input_values[9] = str(df[i, 2])  
        elif df[i, 0] == 'CIS_vector_EA_A':
            cis_coeff_counter = cis_coeff_counter + 1
            input_values[10] = str(df[i, 2])    
        elif df[i, 0] == 'CIS_matrix_B':
            cis_coeff_counter = cis_coeff_counter + 1
            input_values[11] = str(df[i, 2])
        elif df[i, 0] == 'CIS_vector_EA_B':
            cis_coeff_counter = cis_coeff_counter + 1
            input_values[12] = str(df[i, 2])
        elif df[i, 0] == 'LCAO_A':
            input_values[13] = str(df[i, 2])
        elif df[i, 0] == 'LCAO_B':
            input_values[14] = str(df[i, 2])
        elif df[i, 0] == 'number_of_exc_A_right':
            exc_num_counter += 1
            input_values[15] = str(df[i, 2])  
        elif df[i, 0] == 'number_of_EA_A_right':
            exc_num_counter += 1
            input_values[16] = str(df[i, 2])
            print(f'!!!! WARNING !!!! \n EVERY excited state should be included in the numbering of IP/EA')
        elif df[i, 0] == 'number_of_exc_B_right':
            exc_num_counter += 1
            input_values[17] = str(df[i, 2])
        elif df[i, 0] == 'number_of_EA_B_right':
            exc_num_counter += 1
            input_values[18] = str(df[i, 2])
            print(f'!!!! WARNING !!!! \n EVERY excited state should be included in the numbering of IP/EA')
        elif df[i, 0] == 'results_file_name':
            input_values[19] = str(f'{df[i, 2]}')
        elif df[i, 0] == 'CIS_vector_IP_A':
            cis_coeff_counter = cis_coeff_counter + 1
            input_values[20] = str(df[i, 2])
        elif df[i, 0] == 'number_of_IP_A_right':
            exc_num_counter += 1
            input_values[21] = str(df[i, 2])
            print(f'!!!! WARNING !!!! \n EVERY excited state should be included in the numbering of IP/EA')
        elif df[i, 0] == 'CIS_vector_IP_B':
            cis_coeff_counter = cis_coeff_counter + 1
            input_values[22] = str(df[i, 2])
        elif df[i, 0] == 'number_of_IP_B_right':
            exc_num_counter += 1
            input_values[23] = str(df[i, 2])
            print(f'!!!! WARNING !!!! \n EVERY excited state should be included in the numbering of IP/EA')
        elif df[i, 0] == 'output_file_IP_B':
            output_counter = output_counter + 1
            input_values[24] = str(df[i, 2])
        elif df[i, 0] == 'output_file_IP_A':
            output_counter = output_counter + 1
            input_values[25] = str(df[i, 2])
        elif df[i, 0] == 'number_of_exc_A_left':
            exc_num_counter += 1
            input_values[27] = str(df[i, 2])  
        elif df[i, 0] == 'number_of_exc_B_left':
            exc_num_counter += 1
            input_values[28] = str(df[i, 2])    
        elif df[i, 0] == 'number_of_EA_A_left':
            exc_num_counter += 1
            input_values[29] = str(df[i, 2])
            print(f'!!!! WARNING !!!! \n EVERY excited state should be included in the numbering of IP/EA')
        elif df[i, 0] == 'number_of_EA_B_left':
            exc_num_counter += 1
            input_values[30] = str(df[i, 2])
            print(f'!!!! WARNING !!!! \n EVERY excited state should be included in the numbering of IP/EA')
        elif df[i, 0] == 'number_of_IP_A_left':
            exc_num_counter += 1
            input_values[31] = str(df[i, 2])
            print(f'!!!! WARNING !!!! \n EVERY excited state should be included in the numbering of IP/EA')
        elif df[i, 0] == 'number_of_IP_B_left':
            exc_num_counter += 1
            input_values[32] = str(df[i, 2])
            print(f'!!!! WARNING !!!! \n EVERY excited state should be included in the numbering of IP/EA')
        else:
            print(f'Unknown keyword {df[i, 0]} in the input file!')
            sys.exit('Unknown keyword')
    if calc_type == 'local' or calc_type == 'Frenkel':
        if output_counter != 2 or cis_coeff_counter != 2 or exc_num_counter !=2:
            sys.exit('The number of output files and/or CIS coeffs is not correct for Frenkel/Local coupling!')
        else:
            pass
    if calc_type == 'CT' or calc_type == 'Charge_Transfer':
        if output_counter != 4 or cis_coeff_counter != 4 or exc_num_counter != 4:
            sys.exit('The number of output files and/or CIS coeffs is not correct for CT coupling!')
        else:
            pass
    if calc_type == 'mix' or calc_type == 'Frenkel-CT':
        if output_counter != 6 or cis_coeff_counter != 6 or exc_num_counter != 6:
            sys.exit('The number of output files and/or CIS coeffs is not correct for mix coupling!')
        else:
            pass    
    return input_values, header_row
        
        
def get_gr_energies(output_file):
    with open(output_file) as f:
        for lines in f:
            if lines.find('E(SCF)') != -1:
                found_it = lines.split()
                break
    return np.float64(found_it[1])


def get_exc_energy(output_file: str, num_of_exc: int):
    found_it = []
    with open(output_file) as f:
        for lines in f:
            if lines.find('Total TDA electronic energy:') != -1:
                number = lines.split()[-2]
                found_it.append(np.float64(number))
    return np.float64(found_it[int(num_of_exc) - 1])


def get_total_exc_state_num(input_file_name: str):
    with open(input_file_name) as f:
        for lines in f:
            if lines.find('ESTATE_SYM') != -1:
                number = lines.split('=')[-1]
                break
    return int(number)


def process_onel_int_from_molecu(file_name: str, NBAS_A, NBAS_B):
    # first column: indecies, 2nd column: kinetic cont, 3rd column: onel Ham, 
    # Nucl. pot: onelH - kinetic
    # only the top triangle matrix is saved, so we must do some shifting
    # !!!! MUST BE TESTED WITH THE OVERLAP! similarly from molecu
    NBAS_tot = NBAS_A + NBAS_B
    onelh = np.zeros((NBAS_tot, NBAS_tot))
    onelT = np.zeros((NBAS_tot, NBAS_tot))
    data = np.genfromtxt(file_name, delimiter=',', dtype=np.float64, encoding=None)

    # Sort the array by the first column
    df_sorted = data[np.argsort(data[:, 0])]
    i_ind = 0
    j_ind = 0
    j_off = 0
    end_ind = NBAS_tot
    j_ind_start = 0
    # this hurt my brain. But what this does is as we hit the end of a row
    # we decreas the line length, set the i index, and off_set j. At the and, we must 
    # subtract from it the previous length
    line_length = 0.
    for line in df_sorted:
        # we must leave the j between 0 and 75
        onelT[i_ind, j_ind] = np.float64(line[1])
        onelh[i_ind, j_ind] = np.float64(line[2])
        if i_ind == j_ind:
            i_ind += 1
            j_ind = 0
        else:# i_ind != j_ind:
            onelT[j_ind, i_ind] = np.float64(line[1])
            onelh[j_ind, i_ind] = np.float64(line[2])
            j_ind += 1
    return onelT, onelh


def onel_cross_terms_Frenkel(onel_matfrix_AB, CIS_matrix_A, CIS_matrix_B, LCAO_A, S_AB,
                            LCAO_B, NBAS_A: int, NBAS_B: int):
    # Initializ some stuff
    # Instead of cutting the LCAOs I filled the other stuff up.
    AB_term = np.float64(0.)
    BA_term = np.float64(0.)
    # Get the final matrices for first term
    AS_1 = LCAO_A.T @ S_AB
    ASA_1 = AS_1 @ LCAO_B
    ASAC_1 = ASA_1 @ CIS_matrix_B
    ASACA_1 = ASAC_1 @ LCAO_B.T
    ASACAS_1 = ASACA_1 @ S_AB.T
    ASACASA_1 = ASACAS_1 @ LCAO_A
    ASACASAA_1 = ASACASA_1 @ LCAO_A.T
    ASACASAAh_1 = ASACASAA_1 @ onel_matfrix_AB
    ASACASAAhA_1 = ASACASAAh_1 @ LCAO_B
    ASACASAAhAA_1 = ASACASAAhA_1 @ LCAO_B.T
    ASACASAAhAAS_1 = ASACASAAhAA_1 @ S_AB.T
    ASACASAAhAASA_1 = ASACASAAhAAS_1 @ LCAO_A
    term_1 = CIS_matrix_A @ ASACASAAhAASA_1.T
    # Get the final matrices for second term
    CA_2 = CIS_matrix_A @ LCAO_A.T
    CAS_2 = CA_2 @ S_AB
    CASA_2 = CAS_2 @ LCAO_B
    Ah_3 = LCAO_A.T @ onel_matfrix_AB
    AhA_3 = Ah_3 @ LCAO_B
    AhAC_3 = AhA_3 @ CIS_matrix_B
    term_2 = CASA_2 @ AhAC_3.T
    for i_A in range(NBAS_A):
        AB_term += (term_1[i_A, i_A] - term_2[i_A, i_A])
    # BA term 1
    B_CA_1 = CIS_matrix_B @ LCAO_B.T
    B_CAF_1 = B_CA_1 @ onel_matfrix_AB.T
    B_CAFA_1 = B_CAF_1 @ LCAO_A
    B_AS_2 = LCAO_B.T @ S_AB.T
    B_ASA_2 = B_AS_2 @ LCAO_A
    B_ASAC_2 = B_ASA_2 @ CIS_matrix_A
    B_term_1 = B_ASAC_2.T @ B_CAFA_1  
    # BA term 2
    B_AS_3 = LCAO_A.T @ S_AB
    B_ASA_3 = B_AS_3 @ LCAO_B
    B_ASAA_3 = B_ASA_3 @ LCAO_B.T
    B_ASAAF_3 = B_ASAA_3 @ onel_matfrix_AB.T
    B_ASAAFA_3 = B_ASAAF_3 @ LCAO_A
    B_ASAAFAA_3 = B_ASAAFA_3 @ LCAO_A.T
    B_ASAAFAAS_3 = B_ASAAFAA_3 @ S_AB
    B_ASAAFAASA_3 = B_ASAAFAAS_3 @ LCAO_B
    B_ASAAFAASAC_3 = B_ASAAFAASA_3 @ CIS_matrix_B
    B_ASAAFAASACA_3 = B_ASAAFAASAC_3 @ LCAO_B.T
    B_ASAAFAASACAS_3 = B_ASAAFAASACA_3 @ S_AB.T
    B_ASAAFAASACASA_3 = B_ASAAFAASACAS_3 @ LCAO_A
    B_term_2 = CIS_matrix_A.T @ B_ASAAFAASACASA_3
    for a_A in range(NBAS_A):
        BA_term += (B_term_1[a_A, a_A] - B_term_2[a_A, a_A])
    return AB_term, BA_term


def plot_M(M, thrd=1e-10, max_val=None, data=[]):
    """
    Plot a matrix with values below the threshold in white and the rest with a continuous colormap.
    Allows for setting a maximum value to make smaller elements more visible.
   
    Parameters:
    M (ndarray): The input matrix to plot.
    thrd (float): Threshold below which values are masked (plotted in white).
    max_val (float, optional): Maximum value for colormap normalization. If None, defaults to the maximum of M.
    data (list, optional): Contains x_lines and y_lines for grid-like visualization.
    """
    # Create a new figure for each matrix
    plt.figure()
   
    # Mask the matrix values below the threshold
    masked_matrix = np.ma.masked_where(np.abs(M) < thrd, M)
   
    # Determine the normalization limits
    vmin = np.min(masked_matrix)
    vmax = max_val if max_val is not None else np.max(masked_matrix)
   
    # Handle edge cases for uniform matrices
    if vmin == vmax:
        vmax = vmin + 1e-10
   
    # Set up the colormap
    cmap = plt.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)
   
    # Add white color for values below the threshold
    cmap_with_white = cmap(np.arange(cmap.N))
    cmap_with_white = np.vstack((np.array([1, 1, 1, 1]), cmap_with_white))  # Add white to the colormap
    custom_cmap = ListedColormap(cmap_with_white)
   
    # Plot the matrix
    plt.imshow(np.zeros_like(M), cmap=ListedColormap(['white']))  # White background
    plt.imshow(masked_matrix, cmap=custom_cmap, norm=norm)  # Plot with the custom colormap
   
    # Plot optional grid lines
    if data:
        x_lines, y_lines = data
        for x in x_lines:
            plt.axvline(x=x - 0.5, color='gray', linestyle='-')  # Vertical lines
        for y in y_lines:
            plt.axhline(y=y - 0.5, color='gray', linestyle='-')  # Horizontal lines
   
    # Add a colorbar
    plt.colorbar()
    plt.show()


def process_NEWFOCK(file_name: str, NBAS: int, modes: str):  # NBAS means number of basis functions
    function_modes = ['IP', 'EA', 'None']
    if modes not in function_modes:
        raise ValueError("Invalid mode type. Expected one of: %s" % function_modes)
    with open(file_name) as file:
        LCAO_raw = pd.read_csv(file, delimiter='   ', header=None, skiprows=3, engine='python').to_numpy(dtype=np.float64)
    # a file elrendezés, ha jól értem: 1 MO --> 12 Ao egymás alá, 2. MO  (2 oszlop) --> 12 Ao ...
    # az első  alatt kezdődik az 5.
    if modes == 'IP' or modes == 'EA':
        NBAS = NBAS + 1
    else:
        NBAS = NBAS
    Nstacked = int(np.float64(len(LCAO_raw)) / np.float64(NBAS))
    NBAS = int(NBAS)
    proc_LCAO = LCAO_raw[0:NBAS, :]
    for i in range(1, Nstacked):
        tmp = LCAO_raw[(i * NBAS):((i + 1) * NBAS), :]
        if i == Nstacked - 1:  # LCAO_LCAO_T if the last block is full, and correct the cours accordingly
            for j in range(0, 4):
                if math.isnan(tmp[0, j]):
                    tmp = tmp.copy()[:, 0:j]
                    break
        proc_LCAO = np.hstack((proc_LCAO, tmp))
    if modes == 'IP':
        proc_LCAO = np.delete(proc_LCAO, obj=0, axis=1)
        proc_LCAO = np.delete(proc_LCAO, obj=-1, axis=0)
    elif modes == 'EA':
        proc_LCAO = np.delete(proc_LCAO, obj=-1, axis=1)
        proc_LCAO = np.delete(proc_LCAO, obj=-1, axis=0)
    else:
        proc_LCAO = proc_LCAO
    # The MOS are the columns, while the SAOs are the rows
    return proc_LCAO

def sum_contributions(
    contributions: dict,
    exclude: tuple[str, ...] = ("AAAA", "BBBB"),
):
    return sum(
        value
        for name, value in contributions.items()
        if not any(pattern in name for pattern in exclude)
    )


def get_calc_type(input_values):
    is_Frenkel = False
    is_CT = False
    is_mix = False
    if input_values == 'all':
        is_Frenkel = True
        is_CT = True
        is_mix = True
        return is_Frenkel, is_CT, is_mix
    elif input_values == 'Frenkel,mix' or input_values == 'mix,Frenkel':
        is_Frenkel = True
        is_CT = False
        is_mix = True
        return is_Frenkel, is_CT, is_mix
    elif input_values == 'Frenkel,CT' or input_values == 'CT,Frenkel':
        is_Frenkel = True
        is_CT = True
        is_mix = False
        return is_Frenkel, is_CT, is_mix
    elif input_values == 'CT,mix' or input_values == 'mix,CT':
        is_Frenkel = False
        is_CT = True
        is_mix = True
        return is_Frenkel, is_CT, is_mix
    elif input_values == 'local' or input_values == 'Frenkel':
        is_Frenkel = True
        is_CT = False
        is_mix = False
        return is_Frenkel, is_CT, is_mix
    elif input_values == 'CT' or input_values == 'Charge Transfer':
        is_Frenkel = False
        is_CT = True
        is_mix = False
        return is_Frenkel, is_CT, is_mix
    elif input_values == 'mix':
        is_Frenkel = False
        is_CT = False
        is_mix = True
        return is_Frenkel, is_CT, is_mix
    else:
        sys.exit('Unkown calc type')


    # Function to calculate the signed amplitude

def signed_amplitude(z):
    real_part = -1 * z.real
    imaginary_part = -1 * z.imag
    
    if z.imag > 0:
        return ((real_part + imaginary_part))  # Positive real part (1st and 4th Quadrant)
    elif z.imag < 0:
        return ((real_part + imaginary_part))  # Negative real part (2nd and 3rd Quadrant)
    else:
        return (real_part)


def diagonalize_final_H(H_21, H_31, H_32, H_41, H_42, H_43, H_12, H_13, H_23, H_14, H_24, H_34,  H_01, H_02, H_03, H_04, ov_matrix, ov_matrix_Gs,
                        SCF_E_gs_A, SCF_E_gs_B, CIS_E_exc_A,
                        CIS_E_exc_B, CIS_E_IP_A, CIS_E_IP_B, CIS_E_EA_A, CIS_E_EA_B, dist, vdW, elstat_exc, elstat_gr, norm, exch, CT_elstat, ref_cont):
    #supersystem_H = [[0, H_21, H_31, H_41], [H_21, 0, H_32, H_42], [H_31, H_32, 0, H_43], [H_41, H_42, H_43, 0]]
    vdW_cont = np.float64(0.)
    elstat_cont = np.float64(0.)
    for i in range(len(vdW)):
        if round(float(vdW[i][0]), 1) == float(dist):
            vdW_cont = (np.float64(vdW[i][1]))*norm
            exch_cont = (np.float64(exch[i][1]))*norm
            elstat_excited = (np.float64(elstat_exc[i][1]))*norm
            elstat_ground = (np.float64(elstat_gr[i][1]))*norm
            CT_elstat_cont = (np.float64(CT_elstat[i][1]))*norm
            ref_cont = (np.float64(ref_cont[i][1]))*norm
            break
        else:
            pass
    print(f'dist {dist}')
    #supersystem_H =    [[SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont, H_21, H_31 + H_41 ], 
    #                    [H_21, SCF_E_gs_B + CIS_E_exc_A + elstat_cont + vdW_cont, H_32 + H_42 ], 
    #                    [H_31 + H_41, H_32 + H_42, CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont]] 
    ##print(f'{supersystem_H}')
    #eig_val, eig_vec = scipy.linalg.eig(supersystem_H)
    #eig_val.sort()
    #supersystem_H_with_ene_no_int = [[SCF_E_gs_A + CIS_E_exc_B + elstat_excited + vdW_cont + H_21, 0., 1/2. * (H_31 + H_32 + H_41 + H_42), 1/2. * (H_31 + H_32 - H_41 - H_42)], 
    #                                [0., SCF_E_gs_B + CIS_E_exc_A + elstat_excited + vdW_cont - H_21, 1/2. * (H_31 - H_32 + H_41 - H_42), 1/2. * (H_31 - H_32 - H_41 + H_42)], 
    #                                [1/2. * (H_31 + H_32 + H_41 + H_42), 1/2. * (H_31 - H_32 + H_41 - H_42), CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont + H_43, 0.], 
    #                                [1/2. * (H_31 + H_32 - H_41 - H_42), 1/2. * (H_31 - H_32 - H_41 + H_42), 0., CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont - H_43]]
    #
    #eig_val_with_E_no_int, eig_vec = scipy.linalg.eigh(supersystem_H_with_ene_no_int)#, ov_matrix_Gs)
    #eig_val_with_E_no_int.sort()
    #
    #supersystem_H_mix_only = [[SCF_E_gs_A + CIS_E_exc_B + elstat_excited + vdW_cont, 0., H_31, H_41], 
    #                                [0., SCF_E_gs_B + CIS_E_exc_A + elstat_excited + vdW_cont, 0., 0.], 
    #                                [H_31, 0., CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont, 0.], 
    #                                [H_41, 0., 0., CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont]]
    #eig_val_with_E_mix_only, eig_vec = scipy.linalg.eigh(supersystem_H_mix_only)#, ov_matrix_Gs)
    #eig_val_with_E_mix_only.sort()
    #
    #
    #supersystem_H_Frenkel_only = [[SCF_E_gs_A + CIS_E_exc_B + elstat_excited + vdW_cont, H_21], 
    #                                [H_21, SCF_E_gs_B + CIS_E_exc_A + elstat_excited + vdW_cont]]
    #eig_val_with_E_Frenkel_only, eig_vec = scipy.linalg.eigh(supersystem_H_Frenkel_only)#, ov_matrix_Gs)
    #eig_val_with_E_Frenkel_only.sort()
    #supersystem_H_CT_only = [[CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont, H_43], 
    #                                [H_43, CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont]]
    #eig_val_with_E_CT_only, eig_vec = scipy.linalg.eigh(supersystem_H_CT_only)#, ov_matrix_Gs)
    #eig_val_with_E_CT_only.sort()
    #supersystem_H_31 = [[SCF_E_gs_A + CIS_E_exc_B + elstat_excited + vdW_cont, H_31], 
    #                                [H_31, CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont]]
    #eig_val_with_E_31, eig_vec = scipy.linalg.eigh(supersystem_H_31)#, ov_matrix_Gs)
    #eig_val_with_E_31.sort()
    #
    #supersystem_H_41 = [[SCF_E_gs_A + CIS_E_exc_B + elstat_excited + vdW_cont, H_41], 
    #                                [H_41, CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont]]
    #eig_val_with_E_41, eig_vec = scipy.linalg.eigh(supersystem_H_41)#, ov_matrix_Gs)
    #eig_val_with_E_41.sort()
#
    #eig_val_with_E_non_HERM = 0

    supersystem_H_with_ene =  np.array([[(SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont) - ref_cont, H_21, H_31, H_41], 
                                [H_12, (SCF_E_gs_B + CIS_E_exc_A + elstat_cont + vdW_cont) - ref_cont, H_32, H_42], 
                                [H_13, H_23, (CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont) - ref_cont, H_43], 
                                [H_14, H_24, H_34, (CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont) - ref_cont]] )
    #supersystem_H_with_ene =  np.array([[SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont - ref_cont, SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont - ref_cont + H_21, 0.5 * ( SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont + CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - 2*ref_cont) + H_31, 0.5 * ( SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont + CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - 2*ref_cont) + H_41], 
    #                            [ SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont - ref_cont + H_21, SCF_E_gs_B + CIS_E_exc_A + elstat_cont + vdW_cont - ref_cont, 0.5 * ( SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont + CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - 2*ref_cont) + H_32, 0.5 * ( SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont + CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - 2*ref_cont) + H_42], 
    #                            [0.5 * ( SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont + CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - 2*ref_cont) + H_31, 0.5 * ( SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont + CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - 2*ref_cont) + H_32, CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - ref_cont, CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont - ref_cont + H_43], 
    #                            [0.5 * ( SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont + CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - 2*ref_cont) + H_41, 0.5 * ( SCF_E_gs_A + CIS_E_exc_B + elstat_cont + vdW_cont + CIS_E_IP_B + CIS_E_EA_A + CT_elstat_cont + exch_cont - 2*ref_cont) + H_42, CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont - ref_cont + H_43, CIS_E_IP_A + CIS_E_EA_B + CT_elstat_cont + exch_cont - ref_cont]] )
    
    #supersystem_H_with_ene = np.zeroes(np.shape(supersystem_H_with_ene_1), dtype=np.float64)
    #
    #for i in range(4):
    #    for j in range(4):
    #        supersystem_H_with_ene[i, j] = supersystem_H_with_ene_1[i, j] * ov_matrix[i, j]
    print(f'{supersystem_H_with_ene}')
    eig_val_with_E_non_HERM_1, eig_vec_4by4 = scipy.linalg.eig(supersystem_H_with_ene, ov_matrix)
    #eig_val_with_E_non_HERM, eig_vec_4by4 = diagonalize_and_sort(supersystem_H_with_ene)

    eig_val_with_E = eig_val_with_E_non_HERM_1.copy()



    #test_diagonalization(supersystem_H_with_ene, 1e-10)
    eig_val_with_E_non_HERM_1.sort()
    eig_val_with_E_non_HERM = np.real(eig_val_with_E_non_HERM_1) + ref_cont
    if np.any(np.imag(eig_val_with_E_non_HERM_1) != 0):
        with open('non_hermitian_warning.txt', 'a+') as f:
            f.write(f'{dist} \t {np.imag(eig_val_with_E_non_HERM_1)} \n') 
        print("Warning: The eigenvalues are not purely real, indicating a non-Hermitian matrix.")
    
    #transform_matrix = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0, 0],
    #                    [1/np.sqrt(2), -1/np.sqrt(2), 0, 0],
    #                    [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
    #                    [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    #
    #transformed = supersystem_H_with_ene @ transform_matrix
    
    #np.savetxt(f'transforming_matrix_{dist}.txt', transform_matrix, fmt='%.6f')
    #np.savetxt(f'transformed_matrix_{dist}.txt', supersystem_H_with_ene_no_int, fmt='%.6f')
    
    return eig_val_with_E, eig_val_with_E_non_HERM, eig_vec_4by4


def diagonalize_and_sort(matrix: np.ndarray, ov_matrix: np.ndarray):
    """
    Diagonalizes a symmetric/Hermitian matrix and returns sorted eigenvalues and eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix, ov_matrix)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]

def test_diagonalization(matrix: np.ndarray, tol: float = 1e-10):
    """
    Tests whether the diagonalization holds: A = V D V.T
    """
    eigenvalues, eigenvectors = diagonalize_and_sort(matrix)
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    assert np.allclose(reconstructed, matrix, atol=tol), "Diagonalization failed!"

def readinput(input):
    input_matrix = ["0"]
    for row in open(input, "r"):
        new_line = row.rstrip('\n')
        input_matrix.append(str(new_line))
    input_matrix.pop(0)
    return input_matrix


def get_vdW(input_file: str, basis: str, dimer: str):
    found_it = False
    final_matrix = []
    basis = basis.strip()
    dimer = dimer.strip()
    with open(input_file, 'r+') as f:
        for lines in f:
            if lines.find(f'vdW {dimer} {basis}') != -1:
                found_it = True
                continue
            else:
                pass
            if found_it and lines.strip():
                final_matrix.append(lines.split())
            elif found_it and not lines.strip():
                break
            else:
                pass
        return final_matrix


def get_exch(input_file: str, basis: str, dimer: str):
    found_it = False
    final_matrix = []
    basis = basis.strip()
    dimer = dimer.strip()
    with open(input_file, 'r+') as f:
        for lines in f:
            if lines.find(f'exch {dimer} {basis}') != -1:
                found_it = True
                continue
            else:
                pass
            if found_it and lines.strip():
                final_matrix.append(lines.split())
            elif found_it and not lines.strip():
                break
            else:
                pass
        return final_matrix

    
def get_elstat(input_file: str, basis: str, dimer: str, exc_state: str):
    found_it = False
    final_matrix = []
    basis = basis.strip()
    dimer = dimer.strip()
    exc_state = exc_state.strip()
    with open(input_file, 'r+') as f:
        for lines in f:
            if lines.find(f'elstat {dimer} {basis} {exc_state}') != -1:
                found_it = True
                continue
            else:
                pass
            if found_it and lines.strip():
                final_matrix.append(lines.split())
            elif found_it and not lines.strip():
                break
            else:
                pass
        return final_matrix


def get_gr_energies_CC(output_file):
    with open(output_file) as f:
        for lines in f:
            if lines.find('The total energy is') != -1:
                found_it = lines.split()
                break
    return np.float64(found_it[-2])


def get_exc_energy_CC(output_file: str, num_of_exc: int):
    found_it = []
    with open(output_file) as f:
        for lines in f:
            if lines.find('Total EOMEE-CCSD electronic energy') != -1:
                number = lines.split()[-2]
                if number not in found_it:
                    found_it.append(np.float64(number))
                else:
                    pass
    return np.float64(found_it[(int(num_of_exc) - 1)*3])


def GS_coupling(onel_matfrix_AB, CIS_matrix_A, CIS_matrix_B, CIS_vector_EA_A, CIS_vector_EA_B, CIS_vector_IP_A, CIS_vector_IP_B,
                        LCAO_A, S_AB, LCAO_B, NBAS_A, NBAS_B):
    onel_matfrix_BA = onel_matfrix_AB.T
    S_BA = S_AB.T
    V_01_AB_term = np.float64(0.)
    V_01_BA_term = np.float64(0.)
    V_02_AB_term = np.float64(0.)
    V_02_BA_term = np.float64(0.)
    
    CA_01_AB = CIS_matrix_A.T @ LCAO_A.T
    CAf_01_AB = CA_01_AB @ onel_matfrix_AB
    CAfA_01_AB = CAf_01_AB @ LCAO_B
    CAfAA_01_AB = CAfA_01_AB @ LCAO_B.T
    CAfAAS_01_AB = CAfAA_01_AB @ S_BA
    CAfAASA_01_AB = CAfAAS_01_AB @ LCAO_A 
    
    CA_01_BA = CIS_matrix_A.T @ LCAO_A.T
    CAS_01_BA = CA_01_BA @ S_AB
    CASA_01_BA = CAS_01_BA @ LCAO_B
    CASAA_01_BA = CASA_01_BA @ LCAO_B.T
    CASAAf_01_BA = CASAA_01_BA @ onel_matfrix_BA
    CASAAfA_01_BA = CASAAf_01_BA @ LCAO_A 
    
    for kdx in range(NBAS_A):
        V_01_AB_term += CAfAASA_01_AB[kdx, kdx]
        V_01_BA_term += CASAAfA_01_BA[kdx, kdx]
        
    CA_02_AB = CIS_matrix_B.T @ LCAO_B.T
    CAS_02_AB = CA_02_AB @ S_BA
    CASA_02_AB = CAS_02_AB @ LCAO_A
    CASAA_02_AB = CASA_02_AB @ LCAO_A.T
    CASAAf_02_AB = CASAA_02_AB @ onel_matfrix_AB
    CASAAfA_02_AB = CASAAf_02_AB @ LCAO_B 
    
    CA_02_BA = CIS_matrix_B.T @ LCAO_B.T
    CAf_02_BA = CA_02_BA @ onel_matfrix_BA
    CAfA_02_BA = CAf_02_BA @ LCAO_A
    CAfAA_02_BA = CAfA_02_BA @ LCAO_A.T
    CAfAAS_02_BA = CAfAA_02_BA @ S_AB
    CAfAASA_02_BA = CAfAAS_02_BA @ LCAO_B 
    
    for ldx in range(NBAS_B):
        V_02_BA_term += CAfAASA_02_BA[ldx, ldx]
        V_02_AB_term += CASAAfA_02_AB[ldx, ldx]
    
    CA_03_AB = CIS_vector_IP_B @ LCAO_B.T
    CAS_03_AB = CA_03_AB @ S_BA
    CASA_03_AB = CAS_03_AB @ LCAO_A
    CASAA_03_AB = CASA_03_AB @ LCAO_A.T
    CASAAf_03_AB = CASAA_03_AB @ onel_matfrix_AB
    CASAAfA_03_AB = CASAAf_03_AB @ LCAO_B
    CASAAfAA_03_AB = CASAAfA_03_AB @ LCAO_B.T
    CASAAfAAS_03_AB = CASAAfAA_03_AB @ S_BA
    CASAAfAASA_03_AB = CASAAfAAS_03_AB @ LCAO_A
    CASAAfAASAC_03_AB = CASAAfAASA_03_AB @ CIS_vector_EA_A.T
    V_03_AB_term = CASAAfAASAC_03_AB[0, 0]
    
    CA_03_BA = CIS_vector_IP_B @ LCAO_B.T
    CAf_03_BA = CA_03_BA @ onel_matfrix_BA
    CAfA_03_BA = CAf_03_BA @ LCAO_A
    CAfAC_03_BA = CAfA_03_BA @ CIS_vector_EA_A.T
    V_03_BA_term = CAfAC_03_BA[0, 0]
    
    CA_04_AB = CIS_vector_IP_A @ LCAO_A.T
    CAf_04_AB = CA_04_AB @ onel_matfrix_AB
    CAfA_04_AB = CAf_04_AB @ LCAO_B
    CAfAC_04_AB = CAfA_04_AB @ CIS_vector_EA_B.T
    V_04_AB_term = CAfAC_04_AB[0, 0]
    
    CA_04_BA = CIS_vector_IP_A @ LCAO_A.T
    CAS_04_BA = CA_04_BA @ S_AB
    CASA_04_BA = CAS_04_BA @ LCAO_B
    CASAA_04_BA = CASA_04_BA @ LCAO_B.T
    CASAAf_04_BA = CASAA_04_BA @ onel_matfrix_BA
    CASAAfA_04_BA = CASAAf_04_BA @ LCAO_A
    CASAAfAA_04_BA = CASAAfA_04_BA @ LCAO_A.T
    CASAAfAAS_04_BA = CASAAfAA_04_BA @ S_AB
    CASAAfAASA_04_BA = CASAAfAAS_04_BA @ LCAO_B
    CASAAfAASAC_04_BA = CASAAfAASA_04_BA @ CIS_vector_EA_B.T
    V_04_BA_term = CASAAfAASAC_04_BA[0, 0]
    
    return V_01_AB_term, V_01_BA_term, V_02_AB_term, V_02_BA_term, V_03_AB_term, V_03_BA_term, V_04_AB_term, V_04_BA_term


def state_overlap(CIS_matrix_A, CIS_matrix_B, CIS_vector_EA_A, CIS_vector_EA_B, CIS_vector_IP_A, CIS_vector_IP_B,
                        LCAO_A, S_AB, LCAO_B, NBAS_A):

    S_BA = S_AB.T
    S_21_term = np.float64(0.)
    
    AS_AB = LCAO_A.T @ S_AB
    ASA_AB = AS_AB @ LCAO_B
    
    AS_BA = LCAO_B.T @ S_BA
    ASA_BA = AS_BA @ LCAO_A

    SC_21 = ASA_AB @ CIS_matrix_B.T
    SCS_21 = SC_21 @ ASA_BA
    SCSC_21 = SCS_21 @ CIS_matrix_A
    
    for kdx in range(NBAS_A):
        S_21_term += SCSC_21[kdx, kdx]
        
    CS_31 = CIS_vector_IP_B @ ASA_BA
    CSC_31 = CS_31 @ CIS_matrix_A
    S_31_AB_term = CSC_31 @ CIS_vector_EA_A.T
    
    CC_32 = CIS_vector_IP_B @ CIS_matrix_B
    CCS_32 = CC_32 @ ASA_BA
    S_32_AB_term = CCS_32 @ CIS_vector_EA_A.T
    
    CC_41 = CIS_vector_IP_A @ CIS_matrix_A
    CCS_41 = CC_41 @ ASA_AB
    S_41_AB_term = CCS_41 @ CIS_vector_EA_B.T
    
    CS_31 = CIS_vector_IP_A @ ASA_AB
    CSC_31 = CS_31 @ CIS_matrix_B
    S_42_AB_term = CSC_31 @ CIS_vector_EA_B.T
    
    CS_34_1 = CIS_vector_IP_B @ ASA_BA 
    CSC_34_1 = CS_34_1 @ CIS_vector_IP_A.T
    CS_34_2 = CIS_vector_EA_A @ ASA_AB
    CSC_34_2 = CS_34_2 @ CIS_vector_EA_B.T
    S_43_term = CSC_34_1[0, 0] * CSC_34_2[0, 0]
    
    return S_21_term, S_31_AB_term[0, 0], S_32_AB_term[0, 0], S_41_AB_term[0, 0], S_42_AB_term[0, 0], S_43_term

#'fock.sao', 76, 80, 'AO2SO_dim.txt',  'AO2SOINV_dim.txt', 'out.dimer', 'saocao_dim.dat'
def transform_fock(fock_file, nsaos, AOSO_file, AOSOINV_file, dimer_out, sao_caos_dim):
    with open(fock_file) as f:
        H_k= []
        for line in f:
            H_k.append(float(line.replace('\n','')))


    fock_normal_sao_1 = np.zeros((nsaos,nsaos))  
    k = 0
    for i in range(nsaos):
        for j in range(i+1):
            if i == j:
                fock_normal_sao_1[i,j] = H_k[k]
            else:

                fock_normal_sao_1[i,j] = H_k[k]
                fock_normal_sao_1[j,i] = H_k[k]
            k += 1
    fock_sao_c4 = soa_cao_trasnform.TM_sao_to_cfour_sao(fock_normal_sao_1, nsaos, AOSO_file, AOSOINV_file, dimer_out,sao_caos_dim )
    return fock_sao_c4


#!WARNING only works for RHF
def get_occ_orb_num(file: str):# returns eiganvalue ordering!
    is_it_close = False
    split_row = False
    occ_orb = 0
    table_row = ['NOPE']
    
    with open(file, 'r') as f:
        for line in f:
            # CAREFUL at this point I decided to look at the symmetries, which always will be 'XXXX' or 'empty'. 
            # HAD TO BE CHANGED to 0.00000000 since the XXXX do not appear in some cases
            # if the keyword 'CONTINUUM=...' is used
            if 'ORBITAL EIGENVALUE' in line:
                is_it_close = True
                split_row = True
            elif split_row:
                table_row = line.split()
                #if len(table_row) < 6:
                 #   continue
                if table_row and table_row[0].isdigit() and is_it_close:
                    occ_orb += 1
                elif table_row and '+' in table_row[0] and is_it_close:
                    break
                else:
                    continue
            else:
                continue
    return occ_orb

                                                                     
def write_term_results(results_file, V_21, V_31, V_41, V_32, V_42, V_43, V_34, V_13, V_14, V_23, V_24, V_12, dist):
    if os.path.isfile(results_file) == False:
        with open(results_file, 'a+') as final_table:
            final_table.write(f'COUPLING  \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t V_21 \t V_31 \t V_41 \t V_32 \t V_42 \t V_43 \t V_34 \t V_13 \t V_14 \t V_23 \t V_24 \t V_12 \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{dist} \t {V_21} \t  {V_31} \t {V_41} \t  {V_32} \t {V_42} \t {V_43} \t {V_34} \t {V_13} \t {V_14} \t  {V_23} \t {V_24} \t  {V_12} \n')
    else:
        with open(results_file, 'a+') as final_table:
            final_table.write(f'{dist} \t {V_21} \t  {V_31} \t {V_41} \t  {V_32} \t {V_42} \t {V_43} \t {V_34} \t {V_13} \t {V_14} \t  {V_23} \t {V_24} \t  {V_12} \n')
    return print('terms written')


def write_term_results_det(results_file, element_number, twoel_exch, f_no_Ov, f_AB, f_BA, final_element_no_Ov, final_element_full, final_element_full_with_dc, dist):
    if os.path.isfile(results_file) == False:
        with open(results_file, 'a+') as final_table:
            final_table.write(f'COUPLING {element_number} \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t Mo_det \t Mo_occ_only_square \t Total \t Norm_onel \t Norm_twoel \t swapped_norm_max \t corr_S_element \t occ_ind \t virt_ind \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{dist}  \t  {twoel_exch} \t {twoel_exch} \t  {f_no_Ov} \t {f_AB} \t {f_BA} \t {final_element_no_Ov} \t {final_element_full} \t  {final_element_full_with_dc} \n')
    else:
        with open(results_file, 'a+') as final_table:
            final_table.write(f'{dist}  \t  {twoel_exch} \t {twoel_exch} \t  {f_no_Ov} \t {f_AB} \t {f_BA} \t {final_element_no_Ov} \t {final_element_full} \t  {final_element_full_with_dc} \n')
    return print('terms written')

def write_energy_results(E_res_file, energies, dist):
    if os.path.isfile(E_res_file) == False:            
        with open(E_res_file, 'a+') as final_table:
            final_table.write(f'Final states {dimer} {state} {basis} \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t State 1 (Frenkel) \t State 2 (Frenkel) \t State 3 (CT) \t State 4 (CT) \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{dist} \t {energies[0].real} \t { energies[1].real} \t {energies[2].real} \t {energies[3].real} \n')
    else:
        with open(E_res_file, 'a+') as final_table:
            final_table.write(f'{dist} \t {energies[0].real} \t { energies[1].real} \t {energies[2].real} \t {energies[3].real} \n')
    return print('energies written')


def write_energy_results_Frenkel(E_res_file, energies, dist):
    if os.path.isfile(E_res_file) == False:            
        with open(E_res_file, 'a+') as final_table:
            final_table.write(f'Final states {dimer} {state} {basis} \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t State 1 (Frenkel) \t State 2 (Frenkel) \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{dist} \t {energies[0].real} \t { energies[1].real} \n')
    else:
        with open(E_res_file, 'a+') as final_table:
            final_table.write(f'{dist} \t {energies[0].real} \t { energies[1].real} \n')
    return print('energies written')


def write_energy_results_3by3(E_res_file, energies, dist):
    if os.path.isfile(E_res_file) == False:            
        with open(E_res_file, 'a+') as final_table:
            final_table.write(f'Final states {dimer} {state} {basis} \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t State 1 (Frenkel) \t State 2 (Frenkel) \t State 3 (CT)  \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{dist} \t {energies[0].real} \t { energies[1].real} \t {energies[2].real}  \n')
    else:
        with open(E_res_file, 'a+') as final_table:
            final_table.write(f'{dist} \t {energies[0].real} \t { energies[1].real} \t {energies[2].real}  \n')
    return print('energies written')

# Function to handle file paths
def add_path(file_name, dbg, debug_library_path):
    """Prepend the debug library path to the file name if dbg=True."""
    if dbg:
        return os.path.join(debug_library_path, file_name)
    return file_name


def get_swapped_norms():
    det_results = []
    
    # switch an orbital
    for i in range(N_occ_A+N_occ_B):
        res_row = []
        for a in range(N_occ_A, NBAS):
            if Nbas_A <= a < (Nbas_A+N_occ_B):
                continue
            interm_S = S_occ_only.copy()
            interm_S[i, :N_occ_A] = S_MO_trans[a, :N_occ_A]
            interm_S[i, N_occ_A:N_occ_A+N_occ_B] = S_MO_trans[a, Nbas_A:Nbas_A+N_occ_B]
            res_row.append(np.linalg.det(interm_S))
        det_results.append(res_row)
    return det_results

def find_all_element_indexes_numpy(matrix, target):
    arr = np.array(matrix)
    result = np.where(arr == target)
    return list(np.column_stack(result).flatten())


def get_matrix_blocks(matrix, name: str):
    
    res_dict = {
        f"{name}_AA_ii": matrix[0:N_occ_A, 0:N_occ_A],
        f"{name}_AA_ia": matrix[0:N_occ_A, N_occ_A:Nbas_A],
        f"{name}_AA_ai": matrix[N_occ_A:Nbas_A, 0:N_occ_A],
        f"{name}_AA_aa": matrix[N_occ_A:Nbas_A, N_occ_A:Nbas_A],
        f"{name}_BB_ii": matrix[Nbas_A:(Nbas_A+N_occ_B), Nbas_A:(Nbas_A+N_occ_B)],
        f"{name}_BB_ia": matrix[Nbas_A:(Nbas_A+N_occ_B), (Nbas_A+N_occ_B)::],
        f"{name}_BB_ai": matrix[(Nbas_A+N_occ_B)::, Nbas_A:(Nbas_A+N_occ_B)],
        f"{name}_BB_aa": matrix[(Nbas_A+N_occ_B)::, (Nbas_A+N_occ_B)::],        
        f"{name}_AB_ii": matrix[0:N_occ_A, Nbas_A:(Nbas_A+N_occ_B)],
        f"{name}_AB_ia": matrix[0:N_occ_A, (Nbas_A+N_occ_B)::],
        f"{name}_AB_ai": matrix[N_occ_A:Nbas_A, Nbas_A:(Nbas_A+N_occ_B)],
        f"{name}_AB_aa": matrix[N_occ_A:Nbas_A, (Nbas_A+N_occ_B)::],
        f"{name}_BA_ii": matrix[Nbas_A:(Nbas_A+N_occ_B), 0:N_occ_A],
        f"{name}_BA_ia": matrix[Nbas_A:(Nbas_A+N_occ_B), N_occ_A:Nbas_A],
        f"{name}_BA_ai": matrix[(Nbas_A+N_occ_B)::, 0:N_occ_A],
        f"{name}_BA_aa": matrix[(Nbas_A+N_occ_B)::, N_occ_A:Nbas_A],
        f"{name}_AA_pq": matrix[0:Nbas_A, 0:Nbas_A],
        f"{name}_BB_pq": matrix[Nbas_A:(Nbas_A+Nbas_B), Nbas_A:(Nbas_A+Nbas_B)],
        f"{name}_AB_pq": matrix[0:Nbas_A, Nbas_A:(Nbas_A+Nbas_B)],
        f"{name}_BA_pq": matrix[Nbas_A:(Nbas_A+Nbas_B), 0:Nbas_A],
        f"{name}_AA_iq": matrix[0:N_occ_A, 0:Nbas_A],
        f"{name}_BB_iq": matrix[Nbas_A:(Nbas_A+N_occ_B), Nbas_A:(Nbas_A+Nbas_B)],
        f"{name}_AB_iq": matrix[0:N_occ_A, Nbas_A:(Nbas_A+Nbas_B)],
        f"{name}_BA_iq": matrix[Nbas_A:(Nbas_A+N_occ_B), 0:Nbas_A],
        f"{name}_AA_pi": matrix[0:Nbas_A, 0:N_occ_A],
        f"{name}_BB_pi": matrix[Nbas_A:(Nbas_A+Nbas_B), Nbas_A:(Nbas_A+N_occ_B)],
        f"{name}_AB_pi": matrix[0:Nbas_A, Nbas_A:(Nbas_A+N_occ_B)],
        f"{name}_BA_pi": matrix[Nbas_A:(Nbas_A+Nbas_B), 0:N_occ_A],
        f"{name}_AA_aq": matrix[N_occ_A:Nbas_A, 0:Nbas_A],
        f"{name}_BB_aq": matrix[(Nbas_A+N_occ_B)::, Nbas_A:(Nbas_A+Nbas_B)],
        f"{name}_AB_aq": matrix[N_occ_A:Nbas_A, Nbas_A:(Nbas_A+Nbas_B)],
        f"{name}_BA_aq": matrix[(Nbas_A+N_occ_B)::, 0:Nbas_A],
        f"{name}_AA_pa": matrix[0:Nbas_A, N_occ_A:Nbas_A],
        f"{name}_BB_pa": matrix[Nbas_A:(Nbas_A+Nbas_B),(Nbas_A+N_occ_B)::],
        f"{name}_AB_pa": matrix[0:Nbas_A, (Nbas_A+N_occ_B)::],
        f"{name}_BA_pa": matrix[Nbas_A:(Nbas_A+Nbas_B), N_occ_A:Nbas_A]
    }
    
    return res_dict


def read_dens(fname: str, NBAS: int, N_frz: int):
    df = pd.read_csv(fname, sep=r"\s+", header=None).to_numpy(dtype=np.float64) #, ,skiprows=3
    edens = np.zeros((NBAS, NBAS))
    kdx = 0
    ldx = 0
    for idx in range(len(df)):
        for jdx in range(len(df[idx, :])):
            if kdx+N_frz == NBAS:
                ldx += 1
                kdx = 0
            if np.isnan(df[idx, jdx]):
                break
            edens[kdx+N_frz,ldx+N_frz] = df[idx, jdx]
            kdx += 1 
    return edens


def create_dimer_NEWMOS(LCAO_A, LCAO_B, NBAS_A,  NBAS_B):
    dimer_MOS_full = np.zeros((NBAS_A + NBAS_B, NBAS_A + NBAS_B))
    
    N_alpha_A = int(NBAS_A / 2.)
    N_alpha_B = int(NBAS_B / 2.)
    N_beta_A = int(NBAS_A / 2.)
    N_beta_B = int(NBAS_B / 2.)
    
    LCAO_A_alpha = LCAO_A[:, 0:N_alpha_A]
    LCAO_A_beta = LCAO_A[:, N_alpha_A::]
    LCAO_B_alpha = LCAO_B[:, 0:N_alpha_B]
    LCAO_B_beta = LCAO_B[:, N_alpha_B::]
    
    #fill the alpha blocks
    for jdx in range(N_alpha_A + N_alpha_B):
        mon_idx = math.floor(jdx / 2)
        if (jdx % 4) == 0:
            dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_alpha[:, mon_idx]
        if (jdx % 4) == 1:
            dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_alpha[:, mon_idx]
        if (jdx % 4) == 2:
            dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_alpha[:, mon_idx]#dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_alpha[:, mon_idx]
        if (jdx % 4) == 3:
            dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_alpha[:, mon_idx]#dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_alpha[:, mon_idx]
    
    #fill the beta blocks
    for jdx in range(N_alpha_A + N_alpha_B, len(dimer_MOS_full)):
        mon_idx = math.floor((jdx - (N_alpha_A + N_alpha_B))/2)
        if (jdx % 4) == 0:
            dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_beta[:, mon_idx]
        if (jdx % 4) == 1:
            dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_beta[:, mon_idx]
        if (jdx % 4) == 2:
            dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_beta[:, mon_idx]#dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_beta[:, mon_idx]
        if (jdx % 4) == 3:
            dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_beta[:, mon_idx]#dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_beta[:, mon_idx]
    return dimer_MOS_full

def neumann_inverse(S, order=1):
    I = np.eye(S.shape[0])
    result = np.zeros_like(S)
    term = I.copy()
    for n in range(order + 1):
        result += term
        term = term @ (I - S)
    return result


def test_fock(fock_file, nsaos):#, AOSO_file, AOSOINV_file, dimer_out, sao_caos_dim):
    with open(fock_file) as f:
        H_k= []
        for line in f:
            H_k.append(float(line.replace('\n','')))


    fock_normal_sao_1 = np.zeros((nsaos,nsaos))  
    k = 0
    for i in range(nsaos):
        for j in range(i+1):
            if i == j:
                fock_normal_sao_1[i,j] = H_k[k]
            else:

                fock_normal_sao_1[i,j] = H_k[k]
                fock_normal_sao_1[j,i] = H_k[k]
            k += 1
    return fock_normal_sao_1


def matrices_are_equal(A, B, tol=1e-7):
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape")

    diff_matrix = np.abs(A) - np.abs(B)
    abs_diff = np.abs(diff_matrix)

    # Mask: differences smaller than tolerance
    small_diff_mask = abs_diff <= tol

    # Set tiny diffs to exact zero
    diff_matrix[small_diff_mask] = 0.0

    equal = np.all(small_diff_mask)

    return equal, diff_matrix

def parse_bool_arg(value: str) -> bool:
    """
    Parse a boolean CLI argument safely.

    Accepted (case-insensitive):
      True:  1, true, t, yes, y, on
      False: 0, false, f, no, n, off
    """
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r}. Use True/False (or 1/0, yes/no).")

def get_frozen_core(file: str):# returns eiganvalue ordering!
    N_frz = 0
    with open(file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if 'frozen-core' in line.lower():          # e.g. "There are    2 frozen-core orbitals."
                N_frz = int(re.search(r'\b(\d+)\s+frozen-core\b', line, re.I).group(1))
                break
    return N_frz


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise SystemExit("Usage: CT_coupling_mixed_method_new_deriv_4C_final.py INPUT DIST NBAS_A NBAS_B IS_TRANDENS")
    dbg = False
    CCSD = False
    
    if dbg:
        distance = 2.0
        is_trandens = True
        debug_library_path = "/home/bonis/python_scripts/2A_input_files_H_6-311G"
        if CCSD:
            input_file_name =  add_path('coupling_input_CCSD', dbg, debug_library_path)
        else:
            input_file_name =  add_path('coupling_input', dbg, debug_library_path)
    else:
        debug_library_path = 'asd'
        distance = sys.argv[2]
        input_file_name = sys.argv[1]
    input_values, header_row = input_reader(file_name=input_file_name)
    is_parallel = True
    only_onel = False
    is_CC = True
    new_version = True
    is_aug = False
    is_Ov = True
    logging.basicConfig(level=logging.DEBUG, filename='script.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    is_Frenkel, is_CT, is_mix = get_calc_type(input_values[0])
    

    if dbg:
        Nbas_A = int(6)#6  # 29 10
        Nbas_B = int(6)#6 
        overlap_file = add_path(input_values[3], dbg, debug_library_path)
        twoelint_file = add_path(input_values[4], dbg, debug_library_path)
        onel_vmol_file = add_path(input_values[26], dbg, debug_library_path)
        output_file_A = add_path(input_values[5], dbg, debug_library_path)
        output_file_B = add_path(input_values[7], dbg, debug_library_path)
        output_file_IP_A = add_path(input_values[25], dbg, debug_library_path)
        output_file_EA_A = add_path(input_values[6], dbg, debug_library_path)
        output_file_IP_B = add_path(input_values[24], dbg, debug_library_path)
        output_file_EA_B = add_path(input_values[8], dbg, debug_library_path)
        CIS_matrix_file_A = add_path(input_values[9], dbg, debug_library_path)
        CIS_matrix_file_B = add_path(input_values[11], dbg, debug_library_path)
        EA_vector_file_A = add_path(input_values[10], dbg, debug_library_path)
        EA_vector_file_B = add_path(input_values[12], dbg, debug_library_path)
        IP_vector_file_A = add_path(input_values[20], dbg, debug_library_path)
        IP_vector_file_B = add_path(input_values[22], dbg, debug_library_path)
        num_of_exc_A_right   = int(input_values[15]) # 1 is substracted in the scriőt for python
        num_of_exc_B_right   = int(input_values[17])
        number_of_EA_A_right = int(input_values[16]) # 1 is subtracted, plus every exc state shoul be included!!
        number_of_EA_B_right = int(input_values[18]) # 1 is subtracted, plus every exc state shoul be included!!
        number_of_IP_A_right = int(input_values[21]) # 1 is subtracted, plus every exc state shoul be included!!
        number_of_IP_B_right = int(input_values[23])# 1 is subtracted, plus every exc state shoul be included!!
        
        num_of_exc_A_left   = int(input_values[27])
        num_of_exc_B_left   = int(input_values[28])
        number_of_EA_A_left = int(input_values[29])
        number_of_EA_B_left = int(input_values[30])
        number_of_IP_A_left = int(input_values[31])
        number_of_IP_B_left = int(input_values[32])# 1 is subtracted, plus every exc state shoul be included!!
        results_file_Frenkel = f'{input_values[19]}_Frenkel.txt'
        results_file_31 = f'{input_values[19]}_31.txt'
        results_file_32 = f'{input_values[19]}_32.txt'
        results_file_41 = f'{input_values[19]}_41.txt'
        results_file_42 = f'{input_values[19]}_42.txt'
        results_file_CT = f'{input_values[19]}_CT.txt'
        results_file_final = f'{input_values[19]}_diagonalized.txt'
        results_file_gs_couple = f'{input_values[19]}_Gs.txt'
        results_file_S = f'{input_values[19]}_S.txt'
        MO_file_A = add_path(input_values[13], dbg, debug_library_path)
        MO_file_B = add_path(input_values[14], dbg, debug_library_path)
        TM_aosao_file = add_path('saocao_dim.dat', dbg, debug_library_path)
        CF_aosao_file = add_path('AO2SO_dim.txt', dbg, debug_library_path)
        CF_aosaoinv_file = add_path('AO2SOINV_dim.txt', dbg, debug_library_path)
        out_dim = add_path(f'out.dimer_{float(distance):.2f}', dbg, debug_library_path)
        fock_TM = add_path('fock.sao', dbg, debug_library_path)
        onel_vmol_file = add_path('onel_molecu.csv', dbg, debug_library_path)
        onel_vmol_file_A = add_path('onel_molecu_A.csv', dbg, debug_library_path)
        onel_vmol_file_B = add_path('onel_molecu_B.csv', dbg, debug_library_path)
    else:
        Nbas_A = int(sys.argv[3])  # 29 10
        Nbas_B = int(sys.argv[4]) 
        is_trandens = parse_bool_arg(sys.argv[5])
    # Files with paths added dynamicall
    
        overlap_file = input_values[3]
        twoelint_file = input_values[4]
        onel_vmol_file = input_values[26]
        output_file_A = input_values[5]
        output_file_B = input_values[7]
        output_file_IP_A = input_values[25]
        output_file_EA_A = input_values[6]
        output_file_IP_B = input_values[24]
        output_file_EA_B = input_values[8]
        CIS_matrix_file_A = input_values[9]
        CIS_matrix_file_B = input_values[11]
        EA_vector_file_A = input_values[10]
        EA_vector_file_B = input_values[12]
        IP_vector_file_A = input_values[20]
        IP_vector_file_B = input_values[22]
        MO_file_A = input_values[13]
        MO_file_B = input_values[14]
        num_of_exc_A_right   = int(input_values[15]) # 1 is substracted in the scriőt for python
        num_of_exc_B_right   = int(input_values[17])
        number_of_EA_A_right = int(input_values[16]) # 1 is subtracted, plus every exc state shoul be included!!
        number_of_EA_B_right = int(input_values[18]) # 1 is subtracted, plus every exc state shoul be included!!
        number_of_IP_A_right = int(input_values[21]) # 1 is subtracted, plus every exc state shoul be included!!
        number_of_IP_B_right = int(input_values[23])# 1 is subtracted, plus every exc state shoul be included!!
        
        num_of_exc_A_left   = int(input_values[27])
        num_of_exc_B_left   = int(input_values[28])
        number_of_EA_A_left = int(input_values[29])
        number_of_EA_B_left = int(input_values[30])
        number_of_IP_A_left = int(input_values[31])
        number_of_IP_B_left = int(input_values[32])
        
        results_file_Frenkel = f'{input_values[19]}_Frenkel.txt'
        results_file_31 = f'{input_values[19]}_31.txt'
        results_file_32 = f'{input_values[19]}_32.txt'
        results_file_41 = f'{input_values[19]}_41.txt'
        results_file_42 = f'{input_values[19]}_42.txt'
        results_file_CT = f'{input_values[19]}_CT.txt'
        results_file_final = f'{input_values[19]}_diagonalized.txt'
        results_file_gs_couple = f'{input_values[19]}_Gs.txt'
        results_file_S = f'{input_values[19]}_S.txt'
        MO_file_A = input_values[13]
        MO_file_B = input_values[14]
        TM_aosao_file = 'saocao_dim.dat'
        CF_aosao_file = 'AO2SO_dim.txt'
        CF_aosaoinv_file = 'AO2SOINV_dim.txt'
        out_dim = f'out.dimer_{float(distance):.2f}'
        fock_TM = 'fock.sao'
        onel_vmol_file = 'onel_molecu.csv'
        onel_vmol_file_A = 'onel_molecu_A.csv'
        onel_vmol_file_B = 'onel_molecu_B.csv'
    #SCF_E_gs_A = get_gr_energies(output_file=output_file_A)
    #SCF_E_gs_B = get_gr_energies(output_file=output_file_B)
    #CIS_E_exc_A = get_exc_energy(output_file=output_file_A, num_of_exc=num_of_exc_A)
    #CIS_E_exc_B = get_exc_energy(output_file=output_file_B, num_of_exc=num_of_exc_B)
    
    #######################################################################################
    ####################################################################################
    # !HARDCODED file names #
    #####################################################################################
    
    NBAS = int(Nbas_B + Nbas_A)
    #LCAONO_A_raw = read_file_to_matrix(MO_file_A, 2)
    #LCAO_A_exp, valami, wefr, ewrf = create_matrix(LCAONO_A_raw, 2, N_SAO=Nbas_A)    
    #LCAONO_B_raw = read_file_to_matrix(MO_file_B, 2)
    #LCAO_B_exp, valami, wefr, ewrf = create_matrix(LCAONO_B_raw, 2, N_SAO=Nbas_A)
    LCAO_A_exp = process_NEWMOS(file_name=MO_file_A, NBAS=Nbas_A, modes='None')
    LCAO_B_exp = process_NEWMOS(file_name=MO_file_B, NBAS=Nbas_B, modes='None')
    #LCAO_B_exp = np.float64(-1) * LCAO_B_exp_2
    print('LCAO matrices are processed!')
    N_occ_A = get_occ_orb_num(output_file_A)
    N_occ_B = get_occ_orb_num(output_file_B)
    N_frz_A = get_frozen_core(output_file_A)
    N_frz_B = get_frozen_core(output_file_B)
    AO_overlap_raw = read_file_to_matrix(overlap_file, 2)
    AO_overlap_final, valami, wefr, ewrf = create_matrix(AO_overlap_raw, 2, N_SAO=Nbas_A)
    onel_t, onel_h = process_onel_int_from_molecu(file_name=onel_vmol_file, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
    if debug_library_path == "/home/bonis/python_scripts/2A_input_files_H_6-311G":
        onel_t_A, onel_h_A = np.zeros((Nbas_A,Nbas_A)), np.zeros((Nbas_A,Nbas_A))
        onel_t_B, onel_h_B = np.zeros((Nbas_B,Nbas_B)), np.zeros((Nbas_B,Nbas_B))
    else:        
        onel_t_A, onel_h_A = process_onel_int_from_molecu(file_name=onel_vmol_file_A, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
        onel_t_B, onel_h_B = process_onel_int_from_molecu(file_name=onel_vmol_file_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)

    
    h_AB = onel_h[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    h_AA = onel_h[0:Nbas_A, 0:Nbas_A]
    t_AB = onel_t[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    t_AA = onel_t[0:Nbas_A, 0:Nbas_A]
        
    #'fock.sao', 76, 80, 'AO2SO_dim.txt',  'AO2SOINV_dim.txt', 'out.dimer', 'saocao_dim.dat'
    #!
    if dbg:
        Ao_fock = np.zeros((12,12))
    else:
        Ao_fock = transform_fock(fock_file=fock_TM, nsaos=NBAS, AOSO_file=CF_aosao_file, AOSOINV_file=CF_aosaoinv_file, dimer_out=out_dim, sao_caos_dim=TM_aosao_file)
    Ao_fock = transform_fock(fock_file=fock_TM, nsaos=NBAS, AOSO_file=CF_aosao_file, AOSOINV_file=CF_aosaoinv_file, dimer_out=out_dim, sao_caos_dim=TM_aosao_file)

    #!!!!! A teljes overlapet kell meginvertálni!! 
    LCAO_dim_trans = np.zeros((NBAS, NBAS), dtype=np.float64)
    LCAO_dim_trans[0:Nbas_A, 0:Nbas_A] = LCAO_A_exp
    LCAO_dim_trans[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)] = LCAO_B_exp
    
    #LCAO_dim_trans = create_dimer_NEWMOS(LCAO_A=LCAO_A_exp, LCAO_B=LCAO_B_exp, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
    
    AS_MO = LCAO_dim_trans.T @ AO_overlap_final 
    S_MO_trans = AS_MO @ LCAO_dim_trans
    S_MO_inv = scipy.linalg.inv(S_MO_trans)
    
    AAABBBBBBB = S_MO_inv @ S_MO_trans
    
    SA_dim_back = AO_overlap_final @ LCAO_dim_trans
    SAS_inv_dim_back = SA_dim_back @ S_MO_trans
    SAS_invA_dim_back = SAS_inv_dim_back @ LCAO_dim_trans.T
    S_inv_AO_full_traf = SAS_invA_dim_back 
    #S_inv_AO_full_traf =  LCAO_dim_trans @ S_MO_inv @ LCAO_dim_trans.T
    
    CS_valami_valami = AO_overlap_final @ LCAO_dim_trans
    
    SA_dim_back_2 = LCAO_dim_trans.T @ scipy.linalg.inv(AO_overlap_final) @  LCAO_dim_trans # * (math.factorial(8) * math.factorial(8) / math.factorial(16))
    
    S_inv_AO_full_traf = S_inv_AO_full_traf * (np.linalg.det(S_inv_AO_full_traf))
    
    AS_MO_2 = scipy.linalg.inv(LCAO_dim_trans.T @ AO_overlap_final  @ LCAO_dim_trans) @ LCAO_dim_trans.T @ AO_overlap_final  @ LCAO_dim_trans
    
    #S_inv_AO_full_traf[Nbas_A:(Nbas_A + Nbas_B), 0:Nbas_A] = S_inv_AO_full_traf[Nbas_A:(Nbas_A + Nbas_B), 0:Nbas_A] * (math.factorial(8) * math.factorial(8) / math.factorial(16))
    #S_inv_AO_full_traf[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)] = S_inv_AO_full_traf[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)] * (math.factorial(8) * math.factorial(8) / math.factorial(16)) 
    
    #S_MO_inv_2 = neumann_inverse(S_MO_trans, order=10)
    
    S_inv_AO_trans_AB = S_inv_AO_full_traf[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    S_inv_AO_trans_BA = S_inv_AO_full_traf[Nbas_A:(Nbas_A + Nbas_B), 0:Nbas_A]
    
    #if np.allclose(S_inv_AO_trans_AB, S_inv_AO_trans_BA.T):
    #    print('AO overlap is symmetric')
    #else:
    #    print('WARNING: AO overlap is not symmetric')
    
    S_AA = AO_overlap_final[0:Nbas_A, 0:Nbas_A]
    S_BB = AO_overlap_final[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)]
    S_AB = AO_overlap_final[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    S_BA = AO_overlap_final[Nbas_A:(Nbas_A + Nbas_B), 0:Nbas_A]
    
    S_AA_mo = S_MO_trans[0:Nbas_A, 0:Nbas_A]
    S_BB_mo = S_MO_trans[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)]
    S_AB_mo = S_MO_trans[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    S_BA_mo = S_MO_trans[Nbas_A:(Nbas_A + Nbas_B), 0:Nbas_A]
    
    S_occ_only = np.zeros((N_occ_A+N_occ_B, N_occ_A+N_occ_B), dtype=np.float64)
    
    S_occ_only[:N_occ_A, :N_occ_A] = S_MO_trans[:N_occ_A, :N_occ_A]
    S_occ_only[:N_occ_A, N_occ_A:(N_occ_A+N_occ_B)] = S_MO_trans[:N_occ_A,  Nbas_A:(Nbas_A+N_occ_B)]
    S_occ_only[N_occ_A:(N_occ_A+N_occ_B), :N_occ_A] = S_MO_trans[Nbas_A:(Nbas_A+N_occ_B), :N_occ_A]
    S_occ_only[N_occ_A:(N_occ_A+N_occ_B), N_occ_A:(N_occ_A+N_occ_B)] = S_MO_trans[Nbas_A:(Nbas_A+N_occ_B), Nbas_A:(Nbas_A+N_occ_B)]
    
    N_virt_A = Nbas_A - N_occ_A
    N_virt_B = Nbas_B - N_occ_B
    
    S_vv = np.zeros((N_virt_A+N_virt_B, N_virt_A+N_virt_B), dtype=np.float64)
    
    S_vv[:N_virt_A, :N_virt_A] = S_MO_trans[N_occ_A:Nbas_A, N_occ_A:Nbas_A]
    S_vv[:N_virt_A, N_virt_A:(N_virt_A+N_virt_B)] = S_MO_trans[N_occ_A:Nbas_A,  Nbas_A+N_occ_B::]
    S_vv[N_virt_A:(N_virt_A+N_virt_B), :N_virt_A] = S_MO_trans[Nbas_A+N_occ_B::, N_occ_A:Nbas_A]
    S_vv[N_virt_A:(N_virt_A+N_virt_B), N_virt_A:(N_virt_A+N_virt_B)] = S_MO_trans[Nbas_A+N_occ_B::, Nbas_A+N_occ_B::]
            
    #S_ov = np.zeros((N_occ_A+N_occ_B, N_virt_A+N_virt_B), dtype=np.float64)
    #
    #S_ov[:N_occ_A, :N_virt_A] = S_MO_trans[:N_occ_A, N_occ_A:Nbas_A]
    #S_ov[:N_occ_A, N_virt_A:(N_virt_A+N_virt_B)] = S_MO_trans[:N_occ_A,  Nbas_A+N_occ_B::]
    #S_ov[N_occ_A:(N_occ_A+N_occ_B), :N_virt_A] = S_MO_trans[Nbas_A:(Nbas_A+N_occ_B), N_occ_A:Nbas_A]
    #S_ov[N_occ_A:(N_occ_A+N_occ_B), N_virt_A:(N_virt_A+N_virt_B)] = S_MO_trans[Nbas_A:(Nbas_A+N_occ_B), Nbas_A+N_occ_B::]
    #
    #S_vo = np.zeros((N_virt_A+N_virt_B, N_occ_A+N_occ_B), dtype=np.float64)
    #
    #S_vo[:N_virt_A, :N_occ_A] = S_MO_trans[N_occ_A:Nbas_A, :N_occ_A]
    #S_vo[:N_virt_A, N_occ_A:(N_occ_A+N_occ_B)] = S_MO_trans[N_occ_A:Nbas_A,  Nbas_A:(Nbas_A+N_occ_B)]
    #S_vo[N_virt_A:(N_virt_A+N_virt_B), :N_occ_A] = S_MO_trans[Nbas_A+N_occ_B::, :N_occ_A]
    #S_vo[N_virt_A:(N_virt_A+N_virt_B), N_occ_A:(N_occ_A+N_occ_B)] = S_MO_trans[Nbas_A+N_occ_B::, Nbas_A:(Nbas_A+N_occ_B)]

    swap_det_norms = get_swapped_norms()
    swap_det_norms_np = np.array(swap_det_norms)
    swapped_norm_mat = np.zeros(np.shape(S_MO_trans), dtype=np.float64)
    
    swapped_norm_mat[:N_occ_A, N_occ_A:Nbas_A] = swap_det_norms_np[:N_occ_A, :N_virt_A]
    swapped_norm_mat[:N_occ_A,  Nbas_A+N_occ_B::] = swap_det_norms_np[:N_occ_A, N_virt_A:(N_virt_A+N_virt_B)]
    swapped_norm_mat[Nbas_A:(Nbas_A+N_occ_B), N_occ_A:Nbas_A] = swap_det_norms_np[N_occ_A:(N_occ_A+N_occ_B), :N_virt_A]
    swapped_norm_mat[Nbas_A:(Nbas_A+N_occ_B), Nbas_A+N_occ_B::] = swap_det_norms_np[N_occ_A:(N_occ_A+N_occ_B), N_virt_A:(N_virt_A+N_virt_B)]

    norm_mat = swapped_norm_mat.copy()
    for i in range(len(swapped_norm_mat)):
        for j in range(len(swapped_norm_mat)):
            norm_mat[j, i] =  norm_mat[i, j]
    

    norm_ia = swapped_norm_mat
    norm_ai = swapped_norm_mat.T

    S_determ_approx = np.zeros(np.shape(S_MO_trans), dtype=np.float64)
    S_determ_approx[:N_occ_A, N_occ_A:Nbas_A] = S_MO_trans[:N_occ_A, N_occ_A:Nbas_A]
    S_determ_approx[:N_occ_A,  Nbas_A+N_occ_B::] = S_MO_trans[:N_occ_A,  Nbas_A+N_occ_B::]
    S_determ_approx[Nbas_A:(Nbas_A+N_occ_B), N_occ_A:Nbas_A] = S_MO_trans[Nbas_A:(Nbas_A+N_occ_B), N_occ_A:Nbas_A]
    S_determ_approx[Nbas_A:(Nbas_A+N_occ_B), Nbas_A+N_occ_B::] = S_MO_trans[Nbas_A:(Nbas_A+N_occ_B), Nbas_A+N_occ_B::]
    S_determ_approx[N_occ_A:Nbas_A, :N_occ_A] = S_MO_trans[N_occ_A:Nbas_A, :N_occ_A]
    S_determ_approx[N_occ_A:Nbas_A,  Nbas_A:(Nbas_A+N_occ_B)] = S_MO_trans[N_occ_A:Nbas_A,  Nbas_A:(Nbas_A+N_occ_B)]
    S_determ_approx[Nbas_A+N_occ_B::, :N_occ_A] = S_MO_trans[Nbas_A+N_occ_B::, :N_occ_A]
    S_determ_approx[Nbas_A+N_occ_B::, Nbas_A:(Nbas_A+N_occ_B)] = S_MO_trans[Nbas_A+N_occ_B::, Nbas_A:(Nbas_A+N_occ_B)]

    S_determ_approx = S_determ_approx @ S_determ_approx

    S_determ_approx_AA = S_determ_approx[0:Nbas_A, 0:Nbas_A]
    S_determ_approx_BB = S_determ_approx[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)]    
    
    S_AA_mo = S_MO_trans[0:Nbas_A, 0:Nbas_A]
    S_BB_mo = S_MO_trans[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)]
    S_AB_mo = S_MO_trans[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    S_BA_mo = S_MO_trans[Nbas_A:(Nbas_A + Nbas_B), 0:Nbas_A]
    
    norms = {
        "norm_occ_only": np.linalg.det(S_occ_only ),
        "norm_AA_ia": norm_ia[0:Nbas_A, 0:Nbas_A],
        "norm_BB_ia": norm_ia[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)],
        "norm_AB_ia": norm_ia[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)],
        "norm_BA_ia": norm_ia[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)],
        "norm_AA_ai": norm_ai[0:Nbas_A, 0:Nbas_A],
        "norm_BB_ai": norm_ai[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)],
        "norm_AB_ai": norm_ai[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)],
        "norm_BA_ai": norm_ai[Nbas_A:(Nbas_A + Nbas_B), 0:Nbas_A]
    }
    
    norms = get_matrix_blocks(norm_ia, 'norm')
    
    norms.update({
        "norm_occ_only": np.linalg.det(S_occ_only )})
    
    S_blocks = get_matrix_blocks(S_MO_trans, 'S')


    S_inv_blocks = get_matrix_blocks(S_MO_inv, 'S-1')


    onel_h_mo =  LCAO_dim_trans.T @ onel_h @ LCAO_dim_trans  
    
    h_AB = onel_h[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    h_AA = onel_h[0:Nbas_A, 0:Nbas_A]
    h_BB = onel_h[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)]
    #t_AB = onel_t[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    t_AA = onel_t[0:Nbas_A, 0:Nbas_A]

    h_ai = np.zeros(np.shape(S_MO_trans), dtype=np.float64)

    h_ai[N_occ_A:Nbas_A, :N_occ_A] = onel_h_mo[N_occ_A:Nbas_A, :N_occ_A]
    h_ai[N_occ_A:Nbas_A,  Nbas_A:(Nbas_A+N_occ_B)] = onel_h_mo[N_occ_A:Nbas_A,  Nbas_A:(Nbas_A+N_occ_B)]
    h_ai[Nbas_A+N_occ_B::, :N_occ_A] = onel_h_mo[Nbas_A+N_occ_B::, :N_occ_A]
    h_ai[Nbas_A+N_occ_B::, Nbas_A:(Nbas_A+N_occ_B)] = onel_h_mo[Nbas_A+N_occ_B::, Nbas_A:(Nbas_A+N_occ_B)]
    
    #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA = h_ai @ swapped_norm_mat
    
    ##!!!!!!! these should be checked, with the 10 A (or inf ones!!!!!)
    #AAAAAAAAAAAoiuhiuhiu, gdf = scipy.linalg.eigh(f_AA, S_AA)
    #AAAAAAAAAAAoiuhiuhiu.sort()
    f_BB = Ao_fock[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)]
    f_AB = Ao_fock[0:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    f_BA = Ao_fock[Nbas_A:(Nbas_A + Nbas_B), 0:Nbas_A]
    f_AA = Ao_fock[0:Nbas_A, 0:Nbas_A]
    #BBBBBBBBBBBBBoiuhiuhiu, gdf = scipy.linalg.eigh(f_BB, S_BB)
    #BBBBBBBBBBBBBoiuhiuhiu.sort()
    #!!! these are taken from cfour monomer calc SO far the previous ones where used, now im not sure which one to choose, so on 1/21/2025 i decided to stick with the cfour one since i need the orbital energiesk
    FOCK_A_MO = get_MO_fock(output_file_A, NBAS=Nbas_A)
    #sum_fock_mo = np.trace(FOCK_A_MO)
    SA_fock_A = S_AA @ LCAO_A_exp
    SAf_fock_A = SA_fock_A @ FOCK_A_MO
    SAfA_fock_A = SAf_fock_A @ LCAO_A_exp.T
    FOCK_A_AO = SAfA_fock_A @ S_AA
        
    FOCK_B_MO = get_MO_fock(output_file_B, NBAS=Nbas_B)
    
    SA_fock_B = S_BB @ LCAO_B_exp
    SAf_fock_B = SA_fock_B @ FOCK_B_MO
    SAfA_fock_B = SAf_fock_B @ LCAO_B_exp.T
    FOCK_B_AO = SAfA_fock_B @ S_BB
    

    LCAO_A = LCAO_A_exp #@ S_AA
    LCAO_B = LCAO_B_exp #@ S_BB
    det_d = np.linalg.det(S_AA) 
    det_det_det = scipy.linalg.inv(S_AA)
    #S_square = S_BB - S_AB @ det_det_det @ S_BA
    #determ_rest = np.linalg.det(S_square )
    #determ = np.square(determ_rest)
    determ_MO = np.linalg.det(S_MO_trans )
    determ_MO_occ_only = np.square(np.linalg.det(S_occ_only ))
    determ_MO_mult = np.linalg.det(S_occ_only )
    determ_AO = np.linalg.det(AO_overlap_final )

    #LCAO_dim_trans_asd = LCAO_dim_trans @ S_MO_inv
    
    dim_fock = np.zeros((NBAS, NBAS), dtype=np.float64)
    
    dim_fock[0:Nbas_A, 0:Nbas_A] = FOCK_A_MO
    dim_fock[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)] = FOCK_B_MO

    
    dim_fock_AO_2 = AO_overlap_final @ LCAO_dim_trans @ dim_fock @ LCAO_dim_trans.T @ AO_overlap_final
    
    
    diff_fock_MO_2 =  S_MO_inv @ (LCAO_dim_trans.T @ (Ao_fock - dim_fock_AO_2) @ LCAO_dim_trans)
    
    
    fock_blocks = get_matrix_blocks(diff_fock_MO_2, 'f')
    
    f_ai = np.zeros(np.shape(S_MO_trans), dtype=np.float64)
    
    if is_CC:
        
        num_of_EOMEE_A_right = num_of_exc_A_right   #* 4 + total_EE_A - 3
        num_of_EOMEE_B_right = num_of_exc_B_right   #* 4 + total_EE_B - 3
        num_of_EOMIP_A_right = number_of_IP_A_right #* 4 + total_IP_A - 3
        num_of_EOMEA_A_right = number_of_EA_A_right #* 4 + total_EA_A - 3
        num_of_EOMIP_B_right = number_of_IP_B_right #* 4 + total_IP_B - 3
        num_of_EOMEA_B_right = number_of_EA_B_right #* 4 + total_EA_B - 3
        
        num_of_EOMEE_A_left = num_of_exc_A_left   #* 4 + total_EE_A - 1
        num_of_EOMEE_B_left = num_of_exc_B_left   #* 4 + total_EE_B - 1
        num_of_EOMIP_A_left = number_of_IP_A_left #* 4 + total_IP_A - 1
        num_of_EOMEA_A_left = number_of_EA_A_left #* 4 + total_EA_A - 1
        num_of_EOMIP_B_left = number_of_IP_B_left #* 4 + total_IP_B - 1
        num_of_EOMEA_B_left = number_of_EA_B_left #* 4 + total_EA_B - 1
        
        CIS_matrix_A_right = CIS_VECTOR_INTERPRETER(file_name=CIS_matrix_file_A, num_of_exc=num_of_EOMEE_A_right, output_file=output_file_A, CIS_type='exc')#PVDZ H2_N2: 2, 1, 8, 1
        CIS_matrix_B_right = CIS_VECTOR_INTERPRETER(file_name=CIS_matrix_file_B, num_of_exc=num_of_EOMEE_B_right, output_file=output_file_B, CIS_type='exc')
        #tran_dens_A_right    = read_dens(f'RTRANDENS_1_{num_of_exc_A}_A', Nbas_A)
        #tran_dens_B_right    = read_dens(f'RTRANDENS_1_{num_of_exc_B}_B', Nbas_B)
        
        CIS_vector_EA_A_right = CIS_VECTOR_INTERPRETER(file_name=EA_vector_file_A, num_of_exc=num_of_EOMEA_A_right,
                                                        output_file=output_file_EA_A,
                                                        CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_EA_B_right = CIS_VECTOR_INTERPRETER(file_name=EA_vector_file_B, num_of_exc=num_of_EOMEA_B_right,
                                                 output_file=output_file_EA_B,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_IP_A_right = CIS_VECTOR_INTERPRETER(file_name=IP_vector_file_A, num_of_exc=num_of_EOMIP_A_right,
                                                 output_file=output_file_IP_A,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_IP_B_right = CIS_VECTOR_INTERPRETER(file_name=IP_vector_file_B, num_of_exc=num_of_EOMIP_B_right,
                                                 output_file=output_file_IP_B,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        
        CIS_matrix_A_left = CIS_VECTOR_INTERPRETER(file_name=CIS_matrix_file_A, num_of_exc=num_of_EOMEE_A_left, output_file=output_file_A, CIS_type='exc')#PVDZ H2_N2: 2, 1, 8, 1
        CIS_matrix_B_left = CIS_VECTOR_INTERPRETER(file_name=CIS_matrix_file_B, num_of_exc=num_of_EOMEE_B_left, output_file=output_file_B, CIS_type='exc')
        if is_trandens:
            tran_dens_A_left    = read_dens(add_path('LTRANDENS_A', dbg, debug_library_path), Nbas_A, N_frz_A)
            tran_dens_B_left    = read_dens(add_path('LTRANDENS_B', dbg, debug_library_path), Nbas_B, N_frz_B)

            tran_dens_A_right    = read_dens(add_path('RTRANDENS_A', dbg, debug_library_path), Nbas_A, N_frz_A)
            tran_dens_B_right    = read_dens(add_path('RTRANDENS_B', dbg, debug_library_path), Nbas_B, N_frz_B)

            tran_dens_IP_A_left    = read_dens(add_path('LTRANDENS_IP_A', dbg, debug_library_path), Nbas_A + 1, N_frz_A)
            tran_dens_IP_B_left    = read_dens(add_path('LTRANDENS_IP_B', dbg, debug_library_path), Nbas_B + 1, N_frz_B)

            tran_dens_EA_A_left    = read_dens(add_path('LTRANDENS_EA_A', dbg, debug_library_path), Nbas_A + 1, N_frz_A)
            tran_dens_EA_B_left    = read_dens(add_path('LTRANDENS_EA_B', dbg, debug_library_path), Nbas_B + 1, N_frz_B)

            tran_dens_IP_A_right    = read_dens(add_path('RTRANDENS_IP_A', dbg, debug_library_path), Nbas_A + 1, N_frz_A)
            tran_dens_IP_B_right    = read_dens(add_path('RTRANDENS_IP_B', dbg, debug_library_path), Nbas_B + 1, N_frz_B)

            tran_dens_EA_A_right    = read_dens(add_path('RTRANDENS_EA_A', dbg, debug_library_path), Nbas_A + 1, N_frz_A)
            tran_dens_EA_B_right    = read_dens(add_path('RTRANDENS_EA_B', dbg, debug_library_path), Nbas_B + 1, N_frz_B)


        CIS_vector_EA_A_left = CIS_VECTOR_INTERPRETER(file_name=EA_vector_file_A, num_of_exc=num_of_EOMEA_A_left,
                                                       output_file=output_file_EA_A,
                                                       CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_EA_B_left = CIS_VECTOR_INTERPRETER(file_name=EA_vector_file_B, num_of_exc=num_of_EOMEA_B_left,
                                                 output_file=output_file_EA_B,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_IP_A_left = CIS_VECTOR_INTERPRETER(file_name=IP_vector_file_A, num_of_exc=num_of_EOMIP_A_left,
                                                 output_file=output_file_IP_A,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_IP_B_left = CIS_VECTOR_INTERPRETER(file_name=IP_vector_file_B, num_of_exc=num_of_EOMIP_B_left,
                                                 output_file=output_file_IP_B,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        print('CIS coefficients ar read!')     
        
    else:    
        CIS_matrix_A = CIS_VECTOR_INTERPRETER(file_name=CIS_matrix_file_A, num_of_exc=num_of_exc_A, output_file=output_file_A, CIS_type='exc')#PVDZ H2_N2: 2, 1, 8, 1
        CIS_matrix_B = CIS_VECTOR_INTERPRETER(file_name=CIS_matrix_file_B, num_of_exc=num_of_exc_B, output_file=output_file_B, CIS_type='exc')
        CIS_vector_EA_A = CIS_VECTOR_INTERPRETER(file_name=EA_vector_file_A, num_of_exc=number_of_EA_A,
                                                       output_file=output_file_EA_A,
                                                       CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_EA_B = CIS_VECTOR_INTERPRETER(file_name=EA_vector_file_B, num_of_exc=number_of_EA_B,
                                                 output_file=output_file_EA_B,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_IP_A = CIS_VECTOR_INTERPRETER(file_name=IP_vector_file_A, num_of_exc=number_of_IP_A,
                                                 output_file=output_file_IP_A,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        CIS_vector_IP_B = CIS_VECTOR_INTERPRETER(file_name=IP_vector_file_B, num_of_exc=number_of_IP_B,
                                                 output_file=output_file_IP_B,
                                                 CIS_type='CT')  # PVDZ H2_N2: 2, 1, 8, 1
        print('CIS coefficients ar read!')
    #twoel_matrix =  tc.create_matrix_AAAA(twoelint_file, 4, Nbas_A, h_AA, LCAO_A_exp, S_AA, t_AA, CIS_matrix_A_left)
    
    fock_dim_trans = np.zeros((NBAS, NBAS), dtype=np.float64)
    fock_dim_trans[0:Nbas_A, 0:Nbas_A] = FOCK_A_AO
    fock_dim_trans[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)] = FOCK_B_AO
    
    #S_full = np.zeros((NBAS*4, NBAS*4))
    #dim_CIS =  np.zeros((NBAS, NBAS))
    #dim_CIS[:Nbas_A, :Nbas_A] = CIS_matrix_A_left
    #dim_CIS[Nbas_A:(Nbas_A+Nbas_B), Nbas_A:(Nbas_A+Nbas_B)] = CIS_matrix_B_left
    #for i in range(4):
    #    S_full[i*NBAS:(i+1)*NBAS, i*NBAS:(i+1)*NBAS] = S_MO_trans
    #
    #S_full[:NBAS, NBAS:(2*NBAS)] = dim_CIS.T @  S_MO_trans @ dim_CIS
    #S_full[:NBAS, NBAS:(2*NBAS)] = dim_CIS.T @  S_MO_trans @ dim_CIS
    #
    
    
    flattened = [item for row in swap_det_norms for item in row]
    norm_to_write = max(flattened, key=abs)
    # = max(swap_det_norms, key=abs)
    S_ind_from_norm = find_all_element_indexes_numpy(np.array(swap_det_norms), norm_to_write)
    
    S_ind_fin = S_ind_from_norm.copy()
    if S_ind_from_norm[0] >= N_occ_A:
        S_ind_fin[0] = S_ind_from_norm[0] + N_virt_A
    if S_ind_from_norm[1] >= N_virt_A:
        S_ind_fin[1] = S_ind_from_norm[1] + N_occ_A + N_occ_B
    else:
        S_ind_fin[1] = S_ind_from_norm[1] + N_occ_A
        
    if is_trandens:
        red_C_s = {
            "CIS_matrix_A_red_right": tran_dens_A_right.T[:N_occ_A, N_occ_A:Nbas_A],
            "CIS_matrix_B_red_right": tran_dens_B_right.T[:N_occ_B, N_occ_B:Nbas_B],
            "CIS_matrix_A_red_left":  tran_dens_A_left[:N_occ_A, N_occ_A:Nbas_A],
            "CIS_matrix_B_red_left":  tran_dens_B_left[:N_occ_B, N_occ_B:Nbas_B],

            # EA: keep the "frozen index" axis by using [idx] (or idx:idx+1)
            "CIS_vector_EA_A_red_right": tran_dens_EA_A_right[(N_occ_A+1):(Nbas_A+1), [N_frz_A]].T,
            "CIS_vector_EA_B_red_right": tran_dens_EA_B_right[(N_occ_B+1):(Nbas_B+1), [N_frz_B]].T,

            # IP: keep the last-row axis by using [-1:] (or [[-1]])
            "CIS_vector_IP_A_red_right": tran_dens_IP_A_right[-1:, 0:N_occ_A],
            "CIS_vector_IP_B_red_right": tran_dens_IP_B_right[-1:, 0:N_occ_B],

            # EA left: keep row axis by slicing N_frz:N_frz+1 (or [[N_frz]])
            "CIS_vector_EA_A_red_left": tran_dens_EA_A_left[N_frz_A:N_frz_A + 1, (N_occ_A+1):(Nbas_A+1)],
            "CIS_vector_EA_B_red_left": tran_dens_EA_B_left[N_frz_B:N_frz_B + 1, (N_occ_B+1):(Nbas_B+1)],

            # IP left: keep last-column axis by using -1: (or [:, [-1]])
            "CIS_vector_IP_A_red_left": tran_dens_IP_A_left[0:N_occ_A, -1:].T,
            "CIS_vector_IP_B_red_left": tran_dens_IP_B_left[0:N_occ_B, -1:].T,

            "CIS_vector_EA_B_right_red": tran_dens_EA_B_right[N_occ_B:, :].T,
            "CIS_vector_IP_A_right_red": tran_dens_IP_A_right[0:N_occ_A, :].T,

            "CIS_vector_EA_A": tran_dens_EA_A_left.T,
            "CIS_vector_EA_B": tran_dens_EA_B_left.T,
            "CIS_vector_IP_A": tran_dens_IP_A_left.T,
            "CIS_vector_IP_B": tran_dens_IP_B_left.T,
        }
    else:    
        red_C_s = {
            "CIS_matrix_A_red_right":  CIS_matrix_A_right[:N_occ_A, N_occ_A:Nbas_A],
            "CIS_matrix_B_red_right":   CIS_matrix_B_right[:N_occ_B, N_occ_B:Nbas_B],
            "CIS_matrix_A_red_left":  CIS_matrix_A_left[:N_occ_A, N_occ_A:Nbas_A],
            "CIS_matrix_B_red_left":   CIS_matrix_B_left[:N_occ_B, N_occ_B:Nbas_B],        
            "CIS_vector_EA_A_red_right": CIS_vector_EA_A_right[N_occ_A::].T, 
            "CIS_vector_EA_B_red_right": CIS_vector_EA_B_right[N_occ_B::].T, 
            "CIS_vector_IP_A_red_right": CIS_vector_IP_A_right[0:N_occ_A].T,
            "CIS_vector_IP_B_red_right": CIS_vector_IP_B_right[0:N_occ_B].T, 
            "CIS_vector_EA_A_red_left": CIS_vector_EA_A_left[N_occ_A::].T, 
            "CIS_vector_EA_B_red_left": CIS_vector_EA_B_left[N_occ_B::].T, 
            "CIS_vector_IP_A_red_left": CIS_vector_IP_A_left[0:N_occ_A].T,
            "CIS_vector_IP_B_red_left": CIS_vector_IP_B_left[0:N_occ_B].T, 
            "CIS_vector_EA_B_right_red": CIS_vector_EA_B_right[N_occ_B::].T,
            "CIS_vector_IP_A_right_red": CIS_vector_IP_A_right[0:N_occ_A].T,
            "CIS_vector_EA_A": CIS_vector_EA_A_left.T, 
            "CIS_vector_EA_B": CIS_vector_EA_B_left.T, 
            "CIS_vector_IP_A": CIS_vector_IP_A_left.T,
            "CIS_vector_IP_B": CIS_vector_IP_B_left.T         
            #"tran_dens_A_right": (LCAO_A_exp.T @ AO_overlap_final[:Nbas_A, :Nbas_A] @ tran_dens_A_right @ AO_overlap_final[:Nbas_A, :Nbas_A] @ LCAO_A_exp) / np.float64(2.),
            #"tran_dens_B_right": (LCAO_B_exp.T @ AO_overlap_final[Nbas_B:, Nbas_B:] @ tran_dens_B_right @ AO_overlap_final[Nbas_B:, Nbas_B:] @ LCAO_B_exp) / np.float64(2.),
            #"tran_dens_A_left" : (LCAO_A_exp.T @ AO_overlap_final[:Nbas_A, :Nbas_A] @ tran_dens_A_left  @ AO_overlap_final[:Nbas_A, :Nbas_A] @ LCAO_A_exp) / np.float64(2.),
            #"tran_dens_B_left" : (LCAO_B_exp.T @ AO_overlap_final[Nbas_B:, Nbas_B:] @ tran_dens_B_left  @ AO_overlap_final[Nbas_B:, Nbas_B:] @ LCAO_B_exp) / np.float64(2.)
        }
    
    red_LCAO_s = {
        "LCAO_A_red_occ":  LCAO_A_exp[:, :N_occ_A],
        "LCAO_B_red_occ":  LCAO_B_exp[:, :N_occ_B],
        "LCAO_A_red_virt": LCAO_A_exp[:, N_occ_A:Nbas_A],
        "LCAO_B_red_virt": LCAO_B_exp[:, N_occ_B:Nbas_B],
        "LCAO_A": LCAO_A_exp,
        "LCAO_B": LCAO_B_exp        
    }
    
    #if dbg:
    #    AO_results, MO_results = tc_dbg.twoel_in_core(twoelint_file, 4, Nbas_A, Nbas_B, N_occ_A, N_occ_B, fock_dim_trans, LCAO_A_exp, LCAO_B_exp, S_AB, t_AA, CIS_matrix_A_left, CIS_matrix_B_left,
    #                                CIS_vector_EA_A_left.T, CIS_vector_IP_B_left.T, CIS_vector_EA_B_left.T, CIS_vector_IP_A_right.T, S_MO_inv, AO_overlap_final, red_C_s, red_LCAO_s, S_blocks)
    
    start_time_onel = time.time()
    #Overlap_cont = Frenkel_coupling_overlap_alter(MO_fock_A=FOCK_A, MO_fock_B=FOCK_B, CIS_matrix_A=CIS_matrix_A, CIS_matrix_B=CIS_matrix_B, LCAO_A=LCAO_A,
    #                                           S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
    if is_CC:
        print(f'!!!!WARNING!!!!! Gs is not yet optimized for CC')
        V_01_AB_term, V_01_BA_term, V_02_AB_term, V_02_BA_term, V_03_AB_term, V_03_BA_term, V_04_AB_term, V_04_BA_term \
            = GS_coupling(onel_matfrix_AB=f_AB, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_right,
                                                                CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_left.T,
                                                                CIS_vector_IP_A=CIS_vector_IP_A_left.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T,
                                                                LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
        
        cross_onel_f_AB, cross_onel_f_BA = onel_cross_terms_Frenkel(onel_matfrix_AB=f_AB, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_left,
                                                                LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
        
        
        #V_21_term_1, V_21_term_2, V_31_term_1, V_31_term_2, V_32_term_1, V_32_term_2, V_41_term_1, V_41_term_2,  V_42_term_1, V_42_term_2, V_43_term_1, V_43_term_2 \
        #= oc.onel_cross_terms_alter(f_AB, CIS_matrix_A_right, CIS_matrix_B_left, 
        #            CIS_matrix_B_right, CIS_vector_EA_A_left, CIS_vector_EA_B_left, CIS_vector_IP_A_left,
        #            CIS_vector_IP_B_left, CIS_vector_EA_B_right, CIS_vector_IP_A_right, 
        #            f_AA, f_BB, LCAO_A, S_AB, LCAO_B, Nbas_A, Nbas_B, distance)
        
        #!WARNING OC is in MO version now!!
        overlaps, onel_couplings \
        = oc.onel_cross_terms_eff(f_AB, f_BA, CIS_matrix_A_right, CIS_matrix_B_left, 
                    CIS_matrix_B_right, CIS_vector_EA_A_left, CIS_vector_EA_B_left, CIS_vector_IP_A_left,
                    CIS_vector_IP_B_left, CIS_vector_EA_B_right, CIS_vector_IP_A_right, 
                    f_AA, f_BB, LCAO_A, S_AB, LCAO_B, Nbas_A, Nbas_B, FOCK_A_AO, FOCK_B_AO, S_inv_AO_trans_AB, S_MO_inv=S_MO_inv, N_occ_A=N_occ_A, N_occ_B=N_occ_B, normalization=norms,
                    fock_blocks=fock_blocks, S_blocks=S_blocks, S_inv_blocks=S_inv_blocks, red_C_s=red_C_s) #!h_AA changed LCAO-s as well
        
                
    else:
        cross_onel_f_AB, cross_onel_f_BA = onel_cross_terms_Frenkel(onel_matfrix_AB=f_AB, CIS_matrix_A=CIS_matrix_A, CIS_matrix_B=CIS_matrix_B,
                                                                LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
        V_31_AB_term, V_31_BA_term, V_32_AB_term, V_32_BA_term, V_41_AB_term, V_41_BA_term, V_42_AB_term, V_42_BA_term \
            = onel_cross_terms_mix(onel_matfrix_AB=f_AB, CIS_matrix_A=CIS_matrix_A, CIS_matrix_B=CIS_matrix_B,
                                                                CIS_vector_EA_A=CIS_vector_EA_A.T, CIS_vector_EA_B=CIS_vector_EA_B.T,
                                                                CIS_vector_IP_A=CIS_vector_IP_A.T, CIS_vector_IP_B=CIS_vector_IP_B.T,
                                                                LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B, onel_matrix_AA=f_AA, onel_matrix_BB=f_BB)
        V_43_AB_term, V_43_BA_term = onel_cross_terms_CT(onel_matfrix_AB=f_AB, CIS_vector_EA_A=CIS_vector_EA_A.T, CIS_vector_EA_B=CIS_vector_EA_B.T,
                            CIS_vector_IP_A=CIS_vector_IP_A.T, CIS_vector_IP_B=CIS_vector_IP_B.T, LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B)
    end_time_onel = time.time()
    print(f'time spent in the onel calculation {end_time_onel - start_time_onel}')
    
    if only_onel:
        write_twoel_results(onel_couplings, base_filepath='onel_results/', distance=distance)
        sys.exit('This is a onel couplings only script, it finishes after calculating fock matrices')

    start_time_twoel = time.time()
    if not only_onel:
        if is_CC:
            if new_version:
                #res = tc.presort_and_classify_csv(file_path=twoelint_file, output_path_prefix='asd', threshold=Nbas_A)
                if is_Ov:
                    #new_results = tc_ov.run_fortran_then_dbg_compare_with_buckets(
                    #    twoelint_file_path=twoelint_file,
                    #    threshold=Nbas_A,
                    #    NBAS=NBAS,
                    #    red_C_s=red_C_s,
                    #    red_LCAO_s=red_LCAO_s,
                    #    S_blocks=S_blocks,
                    #    S_AB=S_AB,
                    #    report_path="twoel_compare_report.txt",
                    #    enable_fortran_trace_bbba=True,
                    #    fortran_bbba_trace_path="twoel_f90_bbba_trace.txt",
                    #)
                    new_results = tc.process_file(twoelint_file_path=twoelint_file, threshold=Nbas_A, NBAS=NBAS, 
                        CIS_coeffs_matrix_A=CIS_matrix_A_right, CIS_coeffs_matrix_B=CIS_matrix_B_left, LCAO_coeffs_A=LCAO_A, LCAO_coeffs_B=LCAO_B,
                        CIS_coeffs_EOMEA_A=CIS_vector_EA_A_left.T, CIS_coeffs_EOMIP_B=CIS_vector_IP_B_left.T, CIS_coeffs_EOMEA_B=CIS_vector_EA_B_left.T, 
                        CIS_coeffs_EOMIP_A=CIS_vector_IP_A_right.T, red_C_s=red_C_s, red_LCAO_s=red_LCAO_s, S_blocks=S_blocks, S_AB=S_AB)
                else:
                    new_results = tc.process_file(twoelint_file_path=twoelint_file, threshold=Nbas_A, 
                        CIS_coeffs_matrix_A=CIS_matrix_A_right, CIS_coeffs_matrix_B=CIS_matrix_B_left, LCAO_coeffs_A=LCAO_A, LCAO_coeffs_B=LCAO_B,
                        CIS_coeffs_EOMEA_A=CIS_vector_EA_A_left.T, CIS_coeffs_EOMIP_B=CIS_vector_IP_B_left.T, CIS_coeffs_EOMEA_B=CIS_vector_EA_B_left.T, 
                        CIS_coeffs_EOMIP_A=CIS_vector_IP_A_right.T)
            else:
                V_Coulomb, V_Dexter, V_Exchange, V_Frenkel, V_Coulomb_31, V_Exchange_31, V_Coulomb_32,  V_Exchange_32, \
                    V_Coulomb_41, V_Exchange_41, V_Coulomb_42, V_Exchange_42, V_Coulomb_43, V_Exchange_43 \
                    = twoel_calculator_mixed_parallel_CC(file_name=twoelint_file, CIS_coeffs_matrix_A=CIS_matrix_A_right, CIS_coeffs_matrix_B_left=CIS_matrix_B_left, 
                                 CIS_coeffs_matrix_B_right=CIS_matrix_B_right,S_AB=S_AB, LCAO_coeffs_A=LCAO_A, LCAO_coeffs_B=LCAO_B, N_SAO_A=Nbas_A, CIS_coeffs_EOMEA_A_left=CIS_vector_EA_A_left.T, 
                                CIS_coeffs_EOMIP_B_left=CIS_vector_IP_B_left.T, CIS_coeffs_EOMEA_B_left=CIS_vector_EA_B_left.T, CIS_coeffs_EOMIP_A_left=CIS_vector_IP_A_left.T,
                                CIS_coeffs_EOMIP_A_right=CIS_vector_IP_A_right.T, CIS_coeffs_EOMEA_B_right=CIS_vector_EA_B_right.T)            
        else:   
            V_Coulomb, V_Dexter, V_Exchange, V_Frenkel, V_Coulomb_31, V_Exchange_31, V_Coulomb_32,  V_Exchange_32, \
            V_Coulomb_41, V_Exchange_41, V_Coulomb_42, V_Exchange_42, V_Coulomb_43, V_Exchange_43 \
                    = twoel_calculator_mixed_parallel(file_name=twoelint_file, CIS_coeffs_matrix_A=CIS_matrix_A, CIS_coeffs_matrix_B=CIS_matrix_B, 
                                S_AB=S_AB, LCAO_coeffs_A=LCAO_A, LCAO_coeffs_B=LCAO_B, N_SAO_A=Nbas_A, CIS_coeffs_EOMEA_A=CIS_vector_EA_A.T, 
                                CIS_coeffs_EOMIP_B=CIS_vector_IP_B.T, CIS_coeffs_EOMEA_B=CIS_vector_EA_B.T, CIS_coeffs_EOMIP_A=CIS_vector_IP_A.T)
            print(f'Full Coulomb:{V_Coulomb}, Full exchange: {V_Exchange}, Frenkel: {V_Frenkel}, Dexter: {V_Dexter}')
        print(f'Saving to output file...')
    else:
        supsys_Overlap_matrix =    [[np.float64(1.), overlaps.get(12), overlaps.get(13), overlaps.get(14)], 
                                [overlaps.get(21), np.float64(1.), overlaps.get(23), overlaps.get(24)], 
                                [overlaps.get(31), overlaps.get(32), np.float64(1.), overlaps.get(34)], 
                                [overlaps.get(41), overlaps.get(42), overlaps.get(43), np.float64(1.)]] 
    
        if os.path.isfile(results_file_S) == False:
            with open(results_file_S, 'a+') as final_table:
                final_table.write(f'FRENKEL COUPLING {header_row[0]} - {header_row[1]} {header_row[2]} \n ')
                final_table.write(f'-' * 60 + '\n')
                final_table.write(f'Distance \t S_21_term \t S_31_AB_term \t S_32_AB_term \t S_41_AB_term \t S_42_AB_term \t S_43_term \t S_12_term \t S_13_AB_term \t S_23_AB_term \t S_14_AB_term \t S_24_AB_term \t S_34_term\n')
                final_table.write(f'-' * 60 + '\n')
                final_table.write(f"{distance} \t {overlaps.get(21)} \t {overlaps.get(31)} \t {overlaps.get(32)} \t {overlaps.get(41)} \t {overlaps.get(42)} \t {overlaps.get(43)} \t {overlaps.get(12)} \t {overlaps.get(13)} \t {overlaps.get(23)} \t {overlaps.get(14)} \t {overlaps.get(24)} \t {overlaps.get(34)}\n")
        else:
            with open(results_file_S, 'a+') as final_table:
                final_table.write(f"{distance} \t {overlaps.get(21)} \t {overlaps.get(31)} \t {overlaps.get(32)} \t {overlaps.get(41)} \t {overlaps.get(42)} \t {overlaps.get(43)} \t {overlaps.get(12)} \t {overlaps.get(13)} \t {overlaps.get(23)} \t {overlaps.get(14)} \t {overlaps.get(24)} \t {overlaps.get(34)}\n")
        
        sys.exit('onel_done')
    end_time_twoel = time.time()
    print(f'time spent in the twoel calculation {end_time_twoel - start_time_twoel}')
    
    if new_version:
        #RESULT_FIELDS = tc_ov.RESULT_FIELDS  # expose RESULT_FIELDS in this module namespace
        if is_Ov:
            write_term_results_det('determ_MO.txt', 21, (norms.get('norm_occ_only')**2), determ_MO_mult*norm_to_write, norm_to_write, np.square(norm_to_write), S_MO_trans[S_ind_fin[0], S_ind_fin[1]], S_ind_fin[0], S_ind_fin[1], distance)
            write_twoel_results(new_results, base_filepath='twoel_results/', distance=distance)
            write_twoel_results(onel_couplings, base_filepath='onel_results/', distance=distance)
            #new_results = _coerce_twoel_results(new_results_2, result_fields=RESULT_FIELDS)
            V_21_total_twoel = new_results[21].get("C") + new_results[21].get("X")
            V_12_total_twoel = V_21_total_twoel
            V_21_Ov_total_twoel = sum_contributions(new_results[21]) #new_results[21].get("C") + new_results[21].get("X") + new_results[21].get("AAAB_C_CS_virt") + new_results[21].get("AAAB_X_CS_virt") + new_results[21].get("BBBA_C_SC_occ") + new_results[21].get("BBBA_X_SC_occ") +  new_results[21].get('AABB_C_2S') + new_results[21].get('AABB_X_2S')#!due to homodimers, otherwise one would BBBA NOOOO its good without it
            V_31_Ov_total_twoel = sum_contributions(new_results[31]) #new_results[31].get("C") + new_results[31].get("X") + new_results[31].get('AABB_EA_C') + new_results[31].get('AABB_EA_X') + new_results[31].get('AABB_local_occ_C') + new_results[31].get('AABB_local_occ_X') + new_results[31].get("BBBA_C_CS_EA_SC") + new_results[31].get("BBBA_X_CS_EA_SC")
            V_41_Ov_total_twoel = sum_contributions(new_results[41]) #new_results[41].get("C") + new_results[41].get("X") + new_results[41].get('AABB_local_occ_C') + new_results[41].get('AABB_local_occ_X') + new_results[41].get("AAAB_C_SC_CS_EA_2S") + new_results[41].get("AAAB_X_SC_CS_EA_2S") - new_results[41].get('AAAA_EA_C') - new_results[41].get('AAAA_EA_X') 
            V_32_Ov_total_twoel = sum_contributions(new_results[32]) #new_results[32].get('C') + new_results[32].get('X') + new_results[32].get('AABB_local_occ_C') + new_results[32].get('AABB_local_occ_X') + new_results[32].get("BBBA_32_C_SC_EA_CS") + new_results[32].get("BBBA_32_X_SC_EA_CS") - new_results[32].get('BBBB_EA_C') - new_results[32].get('BBBB_EA_X') 
            V_42_Ov_total_twoel = sum_contributions(new_results[42]) #new_results[42].get('C') + new_results[42].get('X') + new_results[42].get('AABB_local_occ_C') + new_results[42].get('AABB_local_occ_X') + new_results[42].get('AABB_EA_C') + new_results[42].get('AABB_EA_X') + new_results[42].get('AAAB_C_SC_CS_EA') + new_results[42].get('AAAB_X_SC_CS_EA')
            V_43_Ov_total_twoel = sum_contributions(new_results[43]) #new_results[43].get('C') + new_results[43].get('X') + new_results[43].get("AAAB_C_CS_EA") + new_results[43].get("AAAB_X_CS_EA") + new_results[43].get("AAAB_C_SC_IP") + new_results[43].get("AAAB_X_SC_IP") - new_results[43]['AAAA_C'] - new_results[43]['AAAA_X']
            V_12_Ov_total_twoel = sum_contributions(new_results[12]) #new_results[12].get("C") + new_results[12].get("X") + new_results[12].get("BBBA_C_CS_virt") + new_results[12].get("BBBA_X_CS_virt") + new_results[12].get("AAAB_C_SC_occ") + new_results[12].get("AAAB_X_SC_occ") +  new_results[12].get('AABB_C_2S') + new_results[12].get('AABB_X_2S')
            V_13_Ov_total_twoel = sum_contributions(new_results[13]) #new_results[13].get("C") + new_results[13].get("X") + new_results[13].get('AABB_local_virt_C') + new_results[13].get('AABB_local_virt_X') + new_results[13].get('AAAB_C_2S_IP') + new_results[13].get('AAAB_X_2S_IP') - new_results[13].get('AAAA_IP_C') - new_results[13].get('AAAA_IP_X') 
            V_14_Ov_total_twoel = sum_contributions(new_results[14]) #new_results[14].get("C") + new_results[14].get("X") + new_results[14].get('AABB_IP_C') + new_results[14].get('AABB_IP_X') + new_results[14].get('AABB_local_virt_C') + new_results[14].get('AABB_local_virt_X') +  new_results[14].get('BBBA_14_C_SC_IP_CS') + new_results[14].get('BBBA_14_X_SC_IP_CS')
            V_23_Ov_total_twoel = sum_contributions(new_results[23]) #new_results[23].get('C') + new_results[23].get('X') + new_results[23].get('AABB_local_virt_C') + new_results[23].get('AABB_local_virt_X') + new_results[23].get('AABB_IP_C')    + new_results[23].get('AABB_IP_X') + new_results[23].get('AAAB_C_2S_IP') + new_results[23].get('AAAB_X_2S_IP')
            V_24_Ov_total_twoel = sum_contributions(new_results[24]) #new_results[24].get('C') + new_results[24].get('X') + new_results[24].get('AABB_local_virt_C') + new_results[24].get('AABB_local_virt_X') + new_results[24].get("BBBA_C_2S_IP") + new_results[24].get("BBBA_X_2S_IP") - new_results[24].get('BBBB_IP_C') - new_results[24].get('BBBB_IP_X') 
            V_34_Ov_total_twoel = sum_contributions(new_results[34]) #new_results[34].get('C') + new_results[34].get('X') + new_results[34].get("BBBA_C_CS_EA") + new_results[34].get("BBBA_X_CS_EA") + new_results[34].get("BBBA_C_SC_IP") + new_results[34].get("BBBA_X_SC_IP") - new_results[43]['AAAA_C'] - new_results[43]['AAAA_X']
            
            V_21_Ov_total_twoel_no_same = sum_contributions(new_results[21], exclude=("AAAA", "BBBB")) #new_results[21].get("C") + new_results[21].get("X") + new_results[21].get("AAAB_C_CS_virt") + new_results[21].get("AAAB_X_CS_virt") + new_results[21].get("BBBA_C_SC_occ") + new_results[21].get("BBBA_X_SC_occ") +  new_results[21].get('AABB_C_2S') + new_results[21].get('AABB_X_2S') #!due to homodimers, otherwise one would BBBA NOOOO its good without it
            V_31_Ov_total_twoel_no_same = sum_contributions(new_results[31], exclude=("AAAA", "BBBB")) #new_results[31].get("C") + new_results[31].get("X") + new_results[31].get('AABB_EA_C') + new_results[31].get('AABB_EA_X') + new_results[31].get('AABB_local_occ_C') + new_results[31].get('AABB_local_occ_X') + new_results[31].get("BBBA_C_CS_EA_SC") + new_results[31].get("BBBA_X_CS_EA_SC")
            V_41_Ov_total_twoel_no_same = sum_contributions(new_results[41], exclude=("AAAA", "BBBB")) #new_results[41].get("C") + new_results[41].get("X") + new_results[41].get('AABB_local_occ_C') + new_results[41].get('AABB_local_occ_X') + new_results[41].get("AAAB_C_SC_CS_EA_2S") + new_results[41].get("AAAB_X_SC_CS_EA_2S")
            V_32_Ov_total_twoel_no_same = sum_contributions(new_results[32], exclude=("AAAA", "BBBB")) #new_results[32].get('C') + new_results[32].get('X') + new_results[32].get('AABB_local_occ_C') + new_results[32].get('AABB_local_occ_X') + new_results[32].get("BBBA_32_C_SC_EA_CS") + new_results[32].get("BBBA_32_X_SC_EA_CS")
            V_42_Ov_total_twoel_no_same = sum_contributions(new_results[42], exclude=("AAAA", "BBBB")) #new_results[42].get('C') + new_results[42].get('X') + new_results[42].get('AABB_local_occ_C') + new_results[42].get('AABB_local_occ_X') + new_results[42].get('AABB_EA_C') + new_results[42].get('AABB_EA_X') + new_results[42].get('AAAB_C_SC_CS_EA') + new_results[42].get('AAAB_X_SC_CS_EA')
            V_43_Ov_total_twoel_no_same = sum_contributions(new_results[43], exclude=("AAAA", "BBBB")) #new_results[43].get('C') + new_results[43].get('X') + new_results[43].get("AAAB_C_CS_EA") + new_results[43].get("AAAB_X_CS_EA") + new_results[43].get("AAAB_C_SC_IP") + new_results[43].get("AAAB_X_SC_IP")
            V_12_Ov_total_twoel_no_same = sum_contributions(new_results[12], exclude=("AAAA", "BBBB")) #new_results[12].get("C") + new_results[12].get("X") + new_results[12].get("BBBA_C_CS_virt") + new_results[12].get("BBBA_X_CS_virt") + new_results[12].get("AAAB_C_SC_occ") + new_results[12].get("AAAB_X_SC_occ") +  new_results[12].get('AABB_C_2S') + new_results[12].get('AABB_X_2S')
            V_13_Ov_total_twoel_no_same = sum_contributions(new_results[13], exclude=("AAAA", "BBBB")) #new_results[13].get("C") + new_results[13].get("X") + new_results[13].get('AABB_local_virt_C') + new_results[13].get('AABB_local_virt_X') +  new_results[13].get('AAAB_C_2S_IP') + new_results[13].get('AAAB_X_2S_IP')
            V_14_Ov_total_twoel_no_same = sum_contributions(new_results[14], exclude=("AAAA", "BBBB")) #new_results[14].get("C") + new_results[14].get("X") + new_results[14].get('AABB_IP_C') + new_results[14].get('AABB_IP_X') + new_results[14].get('AABB_local_virt_C') + new_results[14].get('AABB_local_virt_X') + new_results[14].get("BBBA_14_C_SC_IP_CS") + new_results[14].get("BBBA_14_X_SC_IP_CS")
            V_23_Ov_total_twoel_no_same = sum_contributions(new_results[23], exclude=("AAAA", "BBBB")) #new_results[23].get('C') + new_results[23].get('X') + new_results[23].get('AABB_local_virt_C') + new_results[23].get('AABB_local_virt_X') + new_results[23].get('AABB_IP_C')    + new_results[23].get('AABB_IP_X') + new_results[23].get('AAAB_C_2S_IP') + new_results[23].get('AAAB_X_2S_IP')
            V_24_Ov_total_twoel_no_same = sum_contributions(new_results[24], exclude=("AAAA", "BBBB")) #new_results[24].get('C') + new_results[24].get('X') + new_results[24].get('AABB_local_virt_C') + new_results[24].get('AABB_local_virt_X') + new_results[24].get("BBBA_C_2S_IP") + new_results[24].get("BBBA_X_2S_IP")
            V_34_Ov_total_twoel_no_same = sum_contributions(new_results[34], exclude=("AAAA", "BBBB")) #new_results[34].get('C') + new_results[34].get('X') + new_results[34].get("BBBA_C_CS_EA") + new_results[34].get("BBBA_X_CS_EA") + new_results[34].get("BBBA_C_SC_IP") + new_results[34].get("BBBA_X_SC_IP")
            
            V_21_no_S = new_results[21].get("C") + new_results[21].get("X") #!due to homodimers, otherwise one would BBBA NOOOO its good without it
            V_31_no_S = new_results[31].get("C") + new_results[31].get("X")
            V_41_no_S = new_results[41].get("C") + new_results[41].get("X")
            V_32_no_S = new_results[32].get('C') + new_results[32].get('X')
            V_42_no_S = new_results[42].get('C') + new_results[42].get('X')
            V_43_no_S = new_results[43].get('C') + new_results[43].get('X')
            V_12_no_S = new_results[12].get("C") + new_results[12].get("X")
            V_13_no_S = new_results[13].get("C") + new_results[13].get("X")
            V_14_no_S = new_results[14].get("C") + new_results[14].get("X")
            V_23_no_S = new_results[23].get('C') + new_results[23].get('X')
            V_24_no_S = new_results[24].get('C') + new_results[24].get('X')
            V_34_no_S = new_results[34].get('C') + new_results[34].get('X')                                    
            #V_21_Ov_twoel_my_approx = new_results[21].get("C") + new_results[21].get("X") + (new_results[21].get("AAAB_C") + new_results[21].get("AAAB_X") ) #!due to homodimers, otherwise one would BBBA NOOOO its good without it
            #V_31_Ov_twoel_my_approx = new_results[31].get('AABB_EA_C')[0] + new_results[31].get('AABB_EA_X')[0] + new_results[31]["BBBA_C_CS_EA_SC"][0] + new_results[31]["BBBA_X_CS_EA_SC"][0]
            #V_41_Ov_twoel_my_approx = new_results[41].get('AAAA_EA_C')[0] + new_results[41].get('AAAA_EA_X')[0] + new_results[31]["BBBA_C_CS_EA_SC"][0] + new_results[31]["BBBA_X_CS_EA_SC"][0]
            #V_32_Ov_twoel_my_approx = V_41_Ov_twoel_my_approx.copy()#new_results.get("32_coul") + new_results.get("32_exch") #+ new_results.get("31_coul_2") + new_results.get("31_exch_2")
            #V_42_Ov_twoel_my_approx = V_31_Ov_twoel_my_approx.copy()#new_results.get("42_coul") + new_results.get("42_exch") #+ new_results.get("31_coul_2") + new_results.get("31_exch_2")
            #V_43_Ov_twoel_my_approx = new_results[43]['AAAA_C'][0] + new_results[43]['AAAA_X'][0]
            #V_12_Ov_twoel_my_approx = V_21_Ov_twoel_my_approx.copy() 
            #V_13_Ov_twoel_my_approx = new_results[31].get('AAAA_IP_C')[0] + new_results[31].get('AAAA_IP_X')[0] + new_results[41]["BBBA_14_C_SC_IP_CS"][0] + new_results[41]["BBBA_14_X_SC_IP_CS"][0]
            #V_14_Ov_twoel_my_approx = new_results[41].get('AABB_IP_C')[0] + new_results[41].get('AABB_IP_X')[0] + new_results[41]["BBBA_14_C_SC_IP_CS"][0] + new_results[41]["BBBA_14_X_SC_IP_CS"][0]
            #V_23_Ov_twoel_my_approx = V_14_Ov_twoel_my_approx.copy()
            #V_24_Ov_twoel_my_approx = V_13_Ov_twoel_my_approx.copy()
            #V_34_Ov_twoel_my_approx = V_43_Ov_twoel_my_approx.copy()
            #
            #V_21_Ov_twoel_my_approx_avg = (V_21_Ov_twoel_my_approx + V_12_Ov_twoel_my_approx) / 2. 
            #V_31_Ov_twoel_my_approx_avg = (V_31_Ov_twoel_my_approx + V_13_Ov_twoel_my_approx) / 2. 
            #V_41_Ov_twoel_my_approx_avg = (V_41_Ov_twoel_my_approx + V_14_Ov_twoel_my_approx) / 2. 
            #V_32_Ov_twoel_my_approx_avg = (V_32_Ov_twoel_my_approx + V_23_Ov_twoel_my_approx) / 2. 
            #V_42_Ov_twoel_my_approx_avg = (V_42_Ov_twoel_my_approx + V_24_Ov_twoel_my_approx) / 2. 
            #V_43_Ov_twoel_my_approx_avg = (V_43_Ov_twoel_my_approx + V_34_Ov_twoel_my_approx) / 2. 
            #V_12_Ov_twoel_my_approx_avg = (V_21_Ov_twoel_my_approx + V_12_Ov_twoel_my_approx) / 2. 
            #V_13_Ov_twoel_my_approx_avg = (V_31_Ov_twoel_my_approx + V_13_Ov_twoel_my_approx) / 2. 
            #V_14_Ov_twoel_my_approx_avg = (V_41_Ov_twoel_my_approx + V_14_Ov_twoel_my_approx) / 2. 
            #V_23_Ov_twoel_my_approx_avg = (V_32_Ov_twoel_my_approx + V_23_Ov_twoel_my_approx) / 2. 
            #V_24_Ov_twoel_my_approx_avg = (V_42_Ov_twoel_my_approx + V_24_Ov_twoel_my_approx) / 2. 
            #V_34_Ov_twoel_my_approx_avg = (V_43_Ov_twoel_my_approx + V_34_Ov_twoel_my_approx) / 2. 
        else:
            V_21_total_twoel = new_results.get("21_coul") + new_results.get("21_exch")
            V_21_Ov_total_twoel = new_results.get("21_coul") + new_results.get("21_exch")
            V_31_Ov_total_twoel = new_results.get("31_coul") + new_results.get("31_exch")
            V_32_Ov_total_twoel = new_results.get("32_coul") + new_results.get("32_exch")
            V_41_Ov_total_twoel = new_results.get("41_coul") + new_results.get("41_exch")
            V_42_Ov_total_twoel = new_results.get("42_coul") + new_results.get("42_exch")
            V_43_Ov_total_twoel = new_results.get("43_coul") + new_results.get("43_exch")
        
#        V_21_coul = new_results.get("21_coul") + new_results.get("S1_21_coul")
#        V_31_coul = new_results.get("31_coul") + new_results.get("S1_31_coul")
#        V_32_coul = new_results.get("32_coul")                                
#        V_41_coul = new_results.get("41_coul") + new_results.get("S1_41_coul")
#        V_42_coul = new_results.get("42_coul")                                
#        V_43_coul = new_results.get("43_coul") + new_results.get("S1_43_coul")
#        
#        V_21_exch = new_results.get("21_exch") + new_results.get("S1_21_exch")
#        V_31_exch = new_results.get("31_exch") + new_results.get("S1_31_exch")
#        V_32_exch = new_results.get("32_exch")                            
#        V_41_exch = new_results.get("41_exch") + new_results.get("S1_41_exch")
#        V_42_exch = new_results.get("42_exch")                            
#        V_43_exch = new_results.get("43_exch") + new_results.get("S1_43_exch")
#
#        
#        f_cross_21 = results_21.get('cross')
#        f_cross_31 = results_31.get('AB') + results_31.get('BA')
#        f_cross_32 = results_32.get('AB')
#        f_cross_41 = results_41.get('BA')
#        f_cross_42 = results_42.get('AB') + results_42.get('BA')
#        f_cross_43 = results_34.get('BA')
#        
#        f_same_21 = results_21.get('inner')
#        f_same_31 = results_31.get('AA') + results_31.get('BB')
#        f_same_32 = results_32.get('BB')
#        f_same_41 = results_41.get('AA')
#        f_same_42 = results_42.get('AA') + results_42.get('BB')
#        f_same_43 = results_34.get('AA')
#        
#        f_sum_21 = results_21.get('inner_2')
#        f_sum_31 = results_42.get('BB') + results_42.get('AB')# f_cross_31 + f_same_31
#        f_sum_32 = results_41.get('BA_1') + results_41.get('AA')# f_cross_32 + f_same_32
#        f_sum_41 = results_41.get('BA_1') + results_41.get('AA') #f_cross_41 + f_same_41
#        f_sum_42 = results_42.get('BB') + results_42.get('AB') #f_cross_42 + f_same_42
#        f_sum_43 = f_same_43
#        
#        #f_AB_31_corr_1 = results_31.get('corr_1')
#        #f_AB_31_corr_2 = results_31.get('corr_2')
#        #f_AB_31_corr_3 = results_31.get('corr_3')
#        
#        
#        #f_dc_both_21 = results_21['dc'].get('both')
#        f_dc_both_31 = results_31['dc'].get('both')
#        #f_dc_both_32 = results_32['dc'].get('both')
#        f_dc_both_41 = results_41['dc'].get('both')
#        #f_dc_both_42 = results_42['dc'].get('both')
#        f_dc_both_32 = f_dc_both_41
#        f_dc_both_42 = f_dc_both_31
#        f_dc_both_43 = results_43_dc.get('both')        
#        
#        #f_AB_21 = results_21.get('AB')
#        f_dc_31 = results_31.get('dc_term')
#        f_dc_32 = results_32.get('dc_term')
#        f_dc_41 = results_41.get('dc_term')
#        f_dc_42 = results_42.get('dc_term')
#        f_dc_43 = results_34.get('dc_AB') + results_34.get('dc_BB')
        
        #V_21_onel_my_approx_avg = ((onel_couplings[21].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[21].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_31_onel_my_approx_avg = ((onel_couplings[31].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[31].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_41_onel_my_approx_avg = ((onel_couplings[32].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[32].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_32_onel_my_approx_avg = ((onel_couplings[41].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[41].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_42_onel_my_approx_avg = ((onel_couplings[42].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[42].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_43_onel_my_approx_avg = ((onel_couplings[34].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[34].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_12_onel_my_approx_avg = ((onel_couplings[21].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[21].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_13_onel_my_approx_avg = ((onel_couplings[13].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[13].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_14_onel_my_approx_avg = ((onel_couplings[23].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[23].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_23_onel_my_approx_avg = ((onel_couplings[14].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[14].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_24_onel_my_approx_avg = ((onel_couplings[24].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[24].get('my_approx') / norms.get('norm_occ_only'))) / 2.
        #V_34_onel_my_approx_avg = ((onel_couplings[34].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[34].get('my_approx') / norms.get('norm_occ_only'))) / 2.        
        #
        #V_21_onel_my_approx_avg_dc = (((onel_couplings[21].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[21].get('my_approx') / norms.get('norm_occ_only'))) / 2.)
        #V_31_onel_my_approx_avg_dc = (((onel_couplings[31].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[31].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[31].get('dc_term') / norms.get('norm_occ_only'))
        #V_41_onel_my_approx_avg_dc = (((onel_couplings[32].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[32].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[32].get('dc_term') / norms.get('norm_occ_only'))
        #V_32_onel_my_approx_avg_dc = (((onel_couplings[41].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[41].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[41].get('dc_term') / norms.get('norm_occ_only'))
        #V_42_onel_my_approx_avg_dc = (((onel_couplings[42].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[42].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[42].get('dc_term') / norms.get('norm_occ_only'))
        #V_43_onel_my_approx_avg_dc = (((onel_couplings[34].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[34].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[34].get('dc_BB') / norms.get('norm_occ_only'))
        #V_12_onel_my_approx_avg_dc = (((onel_couplings[21].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[21].get('my_approx') / norms.get('norm_occ_only'))) / 2.)
        #V_13_onel_my_approx_avg_dc = (((onel_couplings[13].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[13].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[13].get('dc_term') / norms.get('norm_occ_only'))
        #V_14_onel_my_approx_avg_dc = (((onel_couplings[23].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[23].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[23].get('dc_term') / norms.get('norm_occ_only'))
        #V_23_onel_my_approx_avg_dc = (((onel_couplings[14].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[14].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[14].get('dc_term') / norms.get('norm_occ_only'))
        #V_24_onel_my_approx_avg_dc = (((onel_couplings[24].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[24].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[24].get('dc_term') / norms.get('norm_occ_only'))
        #V_34_onel_my_approx_avg_dc = (((onel_couplings[34].get('my_approx') / norms.get('norm_occ_only'))  + (onel_couplings[34].get('my_approx') / norms.get('norm_occ_only'))) / 2.) + (onel_couplings[34].get('dc_BB') / norms.get('norm_occ_only'))
        
        Norm_twoel = norms.get('norm_occ_only')**2
        Norm_onel = determ_MO_occ_only
        
        V_21_final_twoel_only = V_21_Ov_total_twoel * Norm_twoel 
        V_31_final_twoel_only = V_31_Ov_total_twoel * Norm_twoel 
        V_32_final_twoel_only = V_32_Ov_total_twoel * Norm_twoel 
        V_41_final_twoel_only = V_41_Ov_total_twoel * Norm_twoel 
        V_42_final_twoel_only = V_42_Ov_total_twoel * Norm_twoel 
        V_43_final_twoel_only = V_43_Ov_total_twoel * Norm_twoel  
        V_12_final_twoel_only = V_12_Ov_total_twoel * Norm_twoel 
        V_13_final_twoel_only = V_13_Ov_total_twoel * Norm_twoel 
        V_23_final_twoel_only = V_23_Ov_total_twoel * Norm_twoel 
        V_14_final_twoel_only = V_14_Ov_total_twoel * Norm_twoel 
        V_24_final_twoel_only = V_24_Ov_total_twoel * Norm_twoel 
        V_34_final_twoel_only = V_34_Ov_total_twoel * Norm_twoel         
        
        V_21_final_no_same = V_21_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[21].get('full') * 2 #* Norm_onel      
        V_31_final_no_same = V_31_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[31].get('full') * 2 #* Norm_onel 
        V_32_final_no_same = V_32_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[32].get('full') * 2 #* Norm_onel 
        V_41_final_no_same = V_41_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[41].get('full') * 2 #* Norm_onel 
        V_42_final_no_same = V_42_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[42].get('full') * 2 #* Norm_onel 
        V_43_final_no_same = V_43_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[43].get('full') * 2 #* Norm_onel 
        V_12_final_no_same = V_12_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[12].get('full') * 2 #* Norm_onel      
        V_13_final_no_same = V_13_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[13].get('full') * 2 #* Norm_onel 
        V_23_final_no_same = V_23_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[23].get('full') * 2 #* Norm_onel 
        V_14_final_no_same = V_14_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[14].get('full') * 2 #* Norm_onel 
        V_24_final_no_same = V_24_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[24].get('full') * 2 #* Norm_onel 
        V_34_final_no_same = V_34_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[34].get('full') * 2 #* Norm_onel 

        V_21_final_no_same_dc = V_21_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[21].get('full') * 2  #* Norm_onel      
        V_31_final_no_same_dc = V_31_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[31].get('full') * 2 + onel_couplings[31].get('dc_term') * 2  #* Norm_onel 
        V_32_final_no_same_dc = V_32_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[32].get('full') * 2 + onel_couplings[32].get('dc_term') * 2  #* Norm_onel 
        V_41_final_no_same_dc = V_41_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[41].get('full') * 2 + onel_couplings[41].get('dc_term') * 2  #* Norm_onel 
        V_42_final_no_same_dc = V_42_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[42].get('full') * 2 + onel_couplings[42].get('dc_term') * 2  #* Norm_onel 
        V_43_final_no_same_dc = V_43_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[43].get('full') * 2 + ( onel_couplings[43].get('dc_AB') + onel_couplings[43].get('dc_BB')) * 2  #* Norm_onel 
        V_12_final_no_same_dc = V_12_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[12].get('full') * 2  #* Norm_onel      
        V_13_final_no_same_dc = V_13_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[13].get('full') * 2 + onel_couplings[13].get('dc_term') * 2 #* Norm_onel 
        V_23_final_no_same_dc = V_23_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[23].get('full') * 2 + onel_couplings[23].get('dc_term') * 2 #* Norm_onel 
        V_14_final_no_same_dc = V_14_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[14].get('full') * 2 + onel_couplings[14].get('dc_term') * 2 #* Norm_onel 
        V_24_final_no_same_dc = V_24_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[24].get('full') * 2 + onel_couplings[24].get('dc_term') * 2 #* Norm_onel 
        V_34_final_no_same_dc = V_34_Ov_total_twoel_no_same * Norm_twoel + onel_couplings[34].get('full') * 2 + ( onel_couplings[34].get('dc_AB') + onel_couplings[34].get('dc_BB')) * 2  #* Norm_onel 
        
        
        V_21_final = V_21_Ov_total_twoel * Norm_twoel + onel_couplings[21].get('full') * 2 #* Norm_onel      
        V_31_final = V_31_Ov_total_twoel * Norm_twoel + onel_couplings[31].get('full') * 2 #* Norm_onel 
        V_32_final = V_32_Ov_total_twoel * Norm_twoel + onel_couplings[32].get('full') * 2 #* Norm_onel 
        V_41_final = V_41_Ov_total_twoel * Norm_twoel + onel_couplings[41].get('full') * 2 #* Norm_onel 
        V_42_final = V_42_Ov_total_twoel * Norm_twoel + onel_couplings[42].get('full') * 2 #* Norm_onel 
        V_43_final = V_43_Ov_total_twoel * Norm_twoel + onel_couplings[43].get('full') * 2 #* Norm_onel 
        V_12_final = V_12_Ov_total_twoel * Norm_twoel + onel_couplings[12].get('full') * 2 #* Norm_onel      
        V_13_final = V_13_Ov_total_twoel * Norm_twoel + onel_couplings[13].get('full') * 2 #* Norm_onel 
        V_23_final = V_23_Ov_total_twoel * Norm_twoel + onel_couplings[23].get('full') * 2 #* Norm_onel 
        V_14_final = V_14_Ov_total_twoel * Norm_twoel + onel_couplings[14].get('full') * 2 #* Norm_onel 
        V_24_final = V_24_Ov_total_twoel * Norm_twoel + onel_couplings[24].get('full') * 2 #* Norm_onel 
        V_34_final = V_34_Ov_total_twoel * Norm_twoel + onel_couplings[34].get('full') * 2 #* Norm_onel 
        
        V_21_final_no_S = V_21_no_S * Norm_twoel #+ onel_couplings[21].get('full') * 2 #* Norm_onel      
        V_31_final_no_S = V_31_no_S * Norm_twoel + onel_couplings[31].get('AB') * 2 #* Norm_onel 
        V_32_final_no_S = V_32_no_S * Norm_twoel + onel_couplings[32].get('AB_1') * 2 #* Norm_onel 
        V_41_final_no_S = V_41_no_S * Norm_twoel + onel_couplings[41].get('BA_1') * 2 #* Norm_onel 
        V_42_final_no_S = V_42_no_S * Norm_twoel + onel_couplings[42].get('BA') * 2 #* Norm_onel 
        V_43_final_no_S = V_43_no_S * Norm_twoel #+ onel_couplings[34].get('full') * 2 #* Norm_onel 
        V_12_final_no_S = V_12_no_S * Norm_twoel #+ onel_couplings[21].get('full') * 2 #* Norm_onel      
        V_13_final_no_S = V_13_no_S * Norm_twoel + onel_couplings[13].get('AB') * 2 #* Norm_onel 
        V_23_final_no_S = V_23_no_S * Norm_twoel + onel_couplings[23].get('BA') * 2 #* Norm_onel 
        V_14_final_no_S = V_14_no_S * Norm_twoel + onel_couplings[14].get('AB') * 2 #* Norm_onel 
        V_24_final_no_S = V_24_no_S * Norm_twoel + onel_couplings[24].get('AB') * 2 #* Norm_onel 
        V_34_final_no_S = V_34_no_S * Norm_twoel#+ onel_couplings[34].get('full') * 2 #* Norm_onel  
        #
        #V_21_final_my_approx = V_21_Ov_twoel_my_approx * Norm_twoel + onel_couplings[21].get('my_approx') * 2 
        #V_31_final_my_approx = V_31_Ov_twoel_my_approx * Norm_twoel + onel_couplings[31].get('my_approx') * 2 
        #V_32_final_my_approx = V_41_Ov_twoel_my_approx * Norm_twoel + onel_couplings[32].get('my_approx') * 2 
        #V_41_final_my_approx = V_32_Ov_twoel_my_approx * Norm_twoel + onel_couplings[41].get('my_approx') * 2 
        #V_42_final_my_approx = V_42_Ov_twoel_my_approx * Norm_twoel + onel_couplings[42].get('my_approx') * 2 
        #V_43_final_my_approx = V_43_Ov_twoel_my_approx * Norm_twoel + onel_couplings[34].get('my_approx') * 2 
        #V_12_final_my_approx = V_12_Ov_twoel_my_approx * Norm_twoel + onel_couplings[21].get('my_approx') * 2 
        #V_13_final_my_approx = V_13_Ov_twoel_my_approx * Norm_twoel + onel_couplings[13].get('my_approx') * 2 
        #V_23_final_my_approx = V_14_Ov_twoel_my_approx * Norm_twoel + onel_couplings[23].get('my_approx') * 2 
        #V_14_final_my_approx = V_23_Ov_twoel_my_approx * Norm_twoel + onel_couplings[14].get('my_approx') * 2 
        #V_24_final_my_approx = V_24_Ov_twoel_my_approx * Norm_twoel + onel_couplings[24].get('my_approx') * 2 
        #V_34_final_my_approx = V_34_Ov_twoel_my_approx * Norm_twoel + onel_couplings[34].get('my_approx') * 2 
        #
        #V_21_final_my_approx_avg = V_21_Ov_twoel_my_approx_avg + V_21_onel_my_approx_avg * 2
        #V_31_final_my_approx_avg = V_31_Ov_twoel_my_approx_avg + V_31_onel_my_approx_avg * 2
        #V_32_final_my_approx_avg = V_41_Ov_twoel_my_approx_avg + V_41_onel_my_approx_avg * 2
        #V_41_final_my_approx_avg = V_32_Ov_twoel_my_approx_avg + V_32_onel_my_approx_avg * 2
        #V_42_final_my_approx_avg = V_42_Ov_twoel_my_approx_avg + V_42_onel_my_approx_avg * 2
        #V_43_final_my_approx_avg = V_43_Ov_twoel_my_approx_avg + V_43_onel_my_approx_avg * 2
        #V_12_final_my_approx_avg = V_12_Ov_twoel_my_approx_avg + V_12_onel_my_approx_avg * 2
        #V_13_final_my_approx_avg = V_13_Ov_twoel_my_approx_avg + V_13_onel_my_approx_avg * 2
        #V_23_final_my_approx_avg = V_14_Ov_twoel_my_approx_avg + V_14_onel_my_approx_avg * 2
        #V_14_final_my_approx_avg = V_23_Ov_twoel_my_approx_avg + V_23_onel_my_approx_avg * 2
        #V_24_final_my_approx_avg = V_24_Ov_twoel_my_approx_avg + V_24_onel_my_approx_avg * 2
        #V_34_final_my_approx_avg = V_34_Ov_twoel_my_approx_avg + V_34_onel_my_approx_avg * 2
        #
        #V_21_final_my_approx_avg_dc = V_21_Ov_twoel_my_approx_avg + V_21_onel_my_approx_avg_dc * 2
        #V_31_final_my_approx_avg_dc = V_31_Ov_twoel_my_approx_avg + V_31_onel_my_approx_avg_dc * 2
        #V_32_final_my_approx_avg_dc = V_41_Ov_twoel_my_approx_avg + V_41_onel_my_approx_avg_dc * 2
        #V_41_final_my_approx_avg_dc = V_32_Ov_twoel_my_approx_avg + V_32_onel_my_approx_avg_dc * 2
        #V_42_final_my_approx_avg_dc = V_42_Ov_twoel_my_approx_avg + V_42_onel_my_approx_avg_dc * 2
        #V_43_final_my_approx_avg_dc = V_43_Ov_twoel_my_approx_avg + V_43_onel_my_approx_avg_dc * 2
        #V_12_final_my_approx_avg_dc = V_12_Ov_twoel_my_approx_avg + V_12_onel_my_approx_avg_dc * 2
        #V_13_final_my_approx_avg_dc = V_13_Ov_twoel_my_approx_avg + V_13_onel_my_approx_avg_dc * 2
        #V_23_final_my_approx_avg_dc = V_14_Ov_twoel_my_approx_avg + V_14_onel_my_approx_avg_dc * 2
        #V_14_final_my_approx_avg_dc = V_23_Ov_twoel_my_approx_avg + V_23_onel_my_approx_avg_dc * 2
        #V_24_final_my_approx_avg_dc = V_24_Ov_twoel_my_approx_avg + V_24_onel_my_approx_avg_dc * 2
        #V_34_final_my_approx_avg_dc = V_34_Ov_twoel_my_approx_avg + V_34_onel_my_approx_avg_dc * 2
        
        #V_21_final = (V_21_Ov_total_twoel   ) * Norm_twoel + (f_sum_21) * 2 #* Norm_onel 
        #V_31_final = (V_31_Ov_total_twoel[0]) * Norm_twoel + (f_sum_31) * 2 #* Norm_onel + f_AB_31_corr_1 + f_AB_31_corr_2 + f_AB_31_corr_3
        #V_32_final = (V_32_Ov_total_twoel[0]) * Norm_twoel + (f_sum_32) * 2 #* Norm_onel 
        #V_41_final = (V_41_Ov_total_twoel[0]) * Norm_twoel + (f_sum_41) * 2 #* Norm_onel 
        #V_42_final = (V_42_Ov_total_twoel[0]) * Norm_twoel + (f_sum_42) * 2 #* Norm_onel 
        #V_43_final = (V_43_Ov_total_twoel[0]) * Norm_twoel + (f_sum_43) * 2 #* Norm_onel 
        
        
        V_21_final_with_dc = V_21_Ov_total_twoel * Norm_twoel + onel_couplings[21].get('full') * 2       
        V_31_final_with_dc = V_31_Ov_total_twoel * Norm_twoel + onel_couplings[31].get('full') * 2 + onel_couplings[31].get('dc_term') * 2 #* Norm_onel 
        V_32_final_with_dc = V_32_Ov_total_twoel * Norm_twoel + onel_couplings[32].get('full') * 2 + onel_couplings[32].get('dc_term') * 2 #* Norm_onel 
        V_41_final_with_dc = V_41_Ov_total_twoel * Norm_twoel + onel_couplings[41].get('full') * 2 + onel_couplings[41].get('dc_term') * 2 #* Norm_onel 
        V_42_final_with_dc = V_42_Ov_total_twoel * Norm_twoel + onel_couplings[42].get('full') * 2 + onel_couplings[42].get('dc_term') * 2 #* Norm_onel 
        V_43_final_with_dc = V_43_Ov_total_twoel * Norm_twoel + onel_couplings[34].get('full') * 2 + (onel_couplings[34].get('dc_AB') + onel_couplings[34].get('dc_BB')) * 2 #* Norm_onel 
        V_12_final_with_dc = V_12_Ov_total_twoel * Norm_twoel + onel_couplings[12].get('full') * 2     
        V_13_final_with_dc = V_13_Ov_total_twoel * Norm_twoel + onel_couplings[13].get('full') * 2 + onel_couplings[13].get('dc_term') * 2 #* Norm_onel 
        V_23_final_with_dc = V_23_Ov_total_twoel * Norm_twoel + onel_couplings[23].get('full') * 2 + onel_couplings[23].get('dc_term') * 2 #* Norm_onel 
        V_14_final_with_dc = V_14_Ov_total_twoel * Norm_twoel + onel_couplings[14].get('full') * 2 + onel_couplings[14].get('dc_term') * 2 #* Norm_onel 
        V_24_final_with_dc = V_24_Ov_total_twoel * Norm_twoel + onel_couplings[24].get('full') * 2 + onel_couplings[24].get('dc_term') * 2 #* Norm_onel 
        V_34_final_with_dc = V_34_Ov_total_twoel * Norm_twoel + onel_couplings[34].get('full') * 2 + (onel_couplings[34].get('dc_AB') + onel_couplings[34].get('dc_BB')) * 2 #* Norm_onel 
    else:
        V_21_total_twoel = V_Frenkel + V_Dexter
        V_21_Ov_total_twoel = V_Coulomb + V_Exchange
        V_31_Ov_total_twoel = V_Coulomb_31 + V_Exchange_31 
        V_32_Ov_total_twoel = V_Coulomb_32 + V_Exchange_32 
        V_41_Ov_total_twoel = V_Coulomb_41 + V_Exchange_41 
        V_42_Ov_total_twoel = V_Coulomb_42 + V_Exchange_42 
        V_43_Ov_total_twoel = V_Coulomb_43 + V_Exchange_43 
        V_21_final = (V_21_Ov_total_twoel / 4) + cross_onel_f_AB_alter - cross_onel_f_BA_alter
        V_31_final = (V_31_Ov_total_twoel / 4) + V_31_AB_term - V_31_BA_term
        V_32_final = (V_32_Ov_total_twoel / 4) + V_32_AB_term - V_32_BA_term
        V_41_final = (V_41_Ov_total_twoel / 4) + V_41_AB_term - V_41_BA_term
        V_42_final = (V_42_Ov_total_twoel / 4) + V_42_AB_term - V_42_BA_term
        V_43_final = (V_43_Ov_total_twoel / 4) + V_43_AB_term
    
    V_01_final = V_01_AB_term + V_01_BA_term
    V_02_final = V_02_AB_term + V_02_BA_term
    V_03_final = V_03_AB_term + V_03_BA_term
    V_04_final = V_04_AB_term + V_04_BA_term
    
    supsys_Overlap_matrix =    [[np.float64(1.), overlaps.get(12), overlaps.get(13), overlaps.get(14)], 
                                [overlaps.get(21), np.float64(1.), overlaps.get(23), overlaps.get(24)], 
                                [overlaps.get(31), overlaps.get(32), np.float64(1.), overlaps.get(34)], 
                                [overlaps.get(41), overlaps.get(42), overlaps.get(43), np.float64(1.)]] 
    
    supsys_Overlap_matrix_no_S =    [[np.float64(1.), 0., 0., 0.], 
                                [0., np.float64(1.), 0., 0.], 
                                [0., 0., np.float64(1.), 0.], 
                                [0., 0., 0., np.float64(1.)]] 
    
    supsys_Overlap_matrix_GS =     [[np.float64(1.), 0., 0., 0., 0. ],
                                    [0.,np.float64(1.), overlaps.get(12), overlaps.get(13), overlaps.get(14)], 
                                    [0.,overlaps.get(21), np.float64(1.), overlaps.get(23), overlaps.get(24)], 
                                    [0.,overlaps.get(31), overlaps.get(32), np.float64(1.), overlaps.get(34)], 
                                    [0.,overlaps.get(41), overlaps.get(42), overlaps.get(43), np.float64(1.)]] 
    
    if os.path.isfile(results_file_S) == False:
        with open(results_file_S, 'a+') as final_table:
            final_table.write(f'FRENKEL COUPLING {header_row[0]} - {header_row[1]} {header_row[2]} \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t S_21_term \t S_31_AB_term \t S_32_AB_term \t S_41_AB_term \t S_42_AB_term \t S_43_term \t S_12_term \t S_13_AB_term \t S_23_AB_term \t S_14_AB_term \t S_24_AB_term \t S_34_term\n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f"{distance} \t {overlaps.get(21)} \t {overlaps.get(31)} \t {overlaps.get(32)} \t {overlaps.get(41)} \t {overlaps.get(42)} \t {overlaps.get(43)} \t {overlaps.get(12)} \t {overlaps.get(13)} \t {overlaps.get(23)} \t {overlaps.get(14)} \t {overlaps.get(24)} \t {overlaps.get(34)}\n")
    else:
        with open(results_file_S, 'a+') as final_table:
            final_table.write(f"{distance} \t {overlaps.get(21)} \t {overlaps.get(31)} \t {overlaps.get(32)} \t {overlaps.get(41)} \t {overlaps.get(42)} \t {overlaps.get(43)} \t {overlaps.get(12)} \t {overlaps.get(13)} \t {overlaps.get(23)} \t {overlaps.get(14)} \t {overlaps.get(24)} \t {overlaps.get(34)}\n")
        


    print(f'Saving to output file...')
    

    write_term_results('final_couplings_twoel_only.txt', V_21_final_twoel_only, V_31_final_twoel_only, V_41_final_twoel_only, V_32_final_twoel_only, V_42_final_twoel_only, V_43_final_twoel_only, V_34_final_twoel_only, V_13_final_twoel_only, V_14_final_twoel_only, V_23_final_twoel_only, V_24_final_twoel_only, V_12_final_twoel_only, distance)
    write_term_results('final_couplings_full.txt', V_21_final, V_31_final, V_41_final, V_32_final, V_42_final, V_43_final, V_34_final, V_13_final, V_14_final, V_23_final, V_24_final, V_12_final, distance)
    write_term_results('final_couplings_full_no_same.txt', V_21_final_no_same, V_31_final_no_same, V_41_final_no_same, V_32_final_no_same, V_42_final_no_same, V_43_final_no_same, V_34_final_no_same, V_13_final_no_same, V_14_final_no_same, V_23_final_no_same, V_24_final_no_same, V_12_final_no_same, distance)
    write_term_results('final_couplings_full_no_same_dc.txt', V_21_final_no_same_dc, V_31_final_no_same_dc, V_41_final_no_same_dc, V_32_final_no_same_dc, V_42_final_no_same_dc, V_43_final_no_same_dc, V_34_final_no_same_dc, V_13_final_no_same_dc, V_14_final_no_same_dc, V_23_final_no_same_dc, V_24_final_no_same_dc, V_12_final_no_same_dc, distance)
    write_term_results('final_couplings_full_with_dc.txt', V_21_final_with_dc, V_31_final_with_dc, V_41_final_with_dc, V_32_final_with_dc, V_42_final_with_dc, V_43_final_with_dc, V_34_final_with_dc, V_13_final_with_dc, V_14_final_with_dc, V_23_final_with_dc, V_24_final_with_dc, V_12_final_with_dc, distance)
    write_term_results('final_couplings_no_S.txt', V_21_final_no_S, V_31_final_no_S, V_41_final_no_S, V_32_final_no_S, V_42_final_no_S, V_43_final_no_S, V_34_final_no_S, V_13_final_no_S, V_14_final_no_S, V_23_final_no_S, V_24_final_no_S, V_12_final_no_S, distance)
    #write_term_results('final_couplings_my_approx_avg.txt', V_21_final_my_approx_avg, V_31_final_my_approx_avg, V_41_final_my_approx_avg, V_32_final_my_approx_avg, V_42_final_my_approx_avg, V_43_final_my_approx_avg, V_13_final_my_approx_avg, V_14_final_my_approx_avg, V_23_final_my_approx_avg, V_24_final_my_approx_avg, distance)
    #write_term_results('final_couplings_my_approx_avg_dc.txt', V_21_final_my_approx_avg_dc, V_31_final_my_approx_avg_dc, V_41_final_my_approx_avg_dc, V_32_final_my_approx_avg_dc, V_42_final_my_approx_avg_dc, V_43_final_my_approx_avg_dc, V_13_final_my_approx_avg_dc, V_14_final_my_approx_avg_dc, V_23_final_my_approx_avg_dc, V_24_final_my_approx_avg_dc, distance)
    #write_term_results(results_file_Frenkel, 21, V_21_coul, V_21_exch, f_cross_21, f_same_21, f_sum_21, V_21_final_no_OV, V_21_final, V_21_final_with_dc, distance)
    #write_term_results(results_file_31     , 31, V_31_coul, V_31_exch, f_cross_31, f_same_31, f_sum_31, V_31_final_no_OV, V_31_final, V_31_final_with_dc, distance)
    #write_term_results(results_file_32     , 32, V_32_coul, V_32_exch, f_cross_32, f_same_32, f_sum_32, V_32_final_no_OV, V_32_final, V_32_final_with_dc, distance)
    #write_term_results(results_file_41     , 41, V_41_coul, V_41_exch, f_cross_41, f_same_41, f_sum_41, V_41_final_no_OV, V_41_final, V_41_final_with_dc, distance)
    #write_term_results(results_file_42     , 42, V_42_coul, V_42_exch, f_cross_42, f_same_42, f_sum_42, V_42_final_no_OV, V_42_final, V_42_final_with_dc, distance)
    #write_term_results(results_file_CT     , 43, V_43_coul, V_43_exch, f_cross_43, f_same_43, f_sum_43, V_43_final_no_OV, V_43_final, V_43_final_with_dc, distance)
    #write_term_results('disconnected_31.txt', 31, 2 * f_dc_both_31, f_AB_31_dc + f_BA_31_dc, f_no_Ov_31, f_AB_31, f_BA_31, 2*f_dc_both_31 + f_AB_31_dc + f_BA_31_dc, f_no_Ov_31, 2*f_dc_both_31 + f_AB_31_dc + f_BA_31_dc + f_AB_31 + f_BA_31, distance)
    #write_term_results('disconnected_41.txt', 41, 2 * f_dc_both_41, f_AB_41_dc + f_BA_41_dc, f_no_Ov_41, f_AB_41, f_BA_41, 2*f_dc_both_41 + f_AB_41_dc + f_BA_41_dc, f_no_Ov_41, 2*f_dc_both_41 + f_AB_41_dc + f_BA_41_dc + f_AB_41 + f_BA_41, distance)
    #write_term_results('disconnected_43.txt', 43, 2 * f_dc_both_43, f_AB_43_dc + f_BA_43_dc, f_no_Ov_43, f_AB_43, f_BA_43, 2*f_dc_both_43 + f_AB_43_dc + f_BA_43_dc, f_no_Ov_43, 2*f_dc_both_43 + f_AB_43_dc + f_BA_43_dc + f_AB_43 + f_BA_43, distance)
    #write_term_results_det('determ_MO.txt', 21, (norms.get('norm_occ_only')**2), determ_MO_mult*norm_to_write, norm_to_write, np.square(norm_to_write), S_MO_trans[S_ind_fin[0], S_ind_fin[1]], S_ind_fin[0], S_ind_fin[1], distance)
    #write_term_results('dc_21.txt', 21, V_21_coul, V_21_exch, V_21_AB_discon, f_AB_21, f_BA_21, V_21_final_no_OV, V_21_final, V_21_final_with_dc, distance)
    if is_aug:
        SCF_E_gs_A = -114.23274414838842 #get_gr_energies_CC(output_file=output_file_A)
        SCF_E_gs_B = -114.23274414838842 #get_gr_energies_CC(output_file=output_file_B)
        CIS_E_exc_A = -113.880293853270189 #get_exc_energy_CC(output_file=output_file_A, num_of_exc=num_of_exc_A)
        CIS_E_exc_B = -113.880293853270189 #get_exc_energy_CC(output_file=output_file_B, num_of_exc=num_of_exc_B)
        CIS_E_IP_A = -113.704707739775031 #!ez eddig rossz volt-113.647540888123743#get_exc_energy_CC(output_file=output_file_IP_A, num_of_exc=number_of_IP_A)
        CIS_E_IP_B = -113.704707739775031 #!ez eddig rossz volt-113.647540888123743#get_exc_energy_CC(output_file=output_file_IP_B, num_of_exc=number_of_IP_B)
        CIS_E_EA_A = -114.189455844357767#get_exc_energy_CC(output_file=output_file_EA_A, num_of_exc=number_of_EA_A)
        CIS_E_EA_B = -114.189455844357767#get_exc_energy_CC(output_file=output_file_EA_B, num_of_exc=number_of_EA_B)
    else:
        if not is_CC:
            SCF_E_gs_A = get_gr_energies(output_file=output_file_A)
            SCF_E_gs_B = get_gr_energies(output_file=output_file_B)
            CIS_E_exc_A = get_exc_energy(output_file=output_file_A, num_of_exc=num_of_exc_A)
            CIS_E_exc_B = get_exc_energy(output_file=output_file_B, num_of_exc=num_of_exc_B)
            CIS_E_IP_A = get_exc_energy(output_file=output_file_IP_A, num_of_exc=number_of_IP_A + 1)
            print(f'CIS_E_IP_A {CIS_E_IP_A}')
            CIS_E_IP_B = get_exc_energy(output_file=output_file_IP_B, num_of_exc=number_of_IP_B + 1)
            CIS_E_EA_A = get_exc_energy(output_file=output_file_EA_A, num_of_exc=number_of_EA_A)
            CIS_E_EA_B = get_exc_energy(output_file=output_file_EA_B, num_of_exc=number_of_EA_B)  
        else:
            #SCF_E_gs_A = get_gr_energies_CC(output_file=output_file_A)
            #SCF_E_gs_B = get_gr_energies_CC(output_file=output_file_B)
            #CIS_E_exc_A = get_exc_energy_CC(output_file=output_file_A, num_of_exc=num_of_exc_A)
            #CIS_E_exc_B = get_exc_energy_CC(output_file=output_file_B, num_of_exc=num_of_exc_B)
            #CIS_E_IP_A = get_exc_energy_CC(output_file=output_file_IP_A, num_of_exc=number_of_IP_A)
            #print(f'CIS_E_IP_A {CIS_E_IP_A}')
            #CIS_E_IP_B = get_exc_energy_CC(output_file=output_file_IP_B, num_of_exc=number_of_IP_B)
            #CIS_E_EA_A = get_exc_energy_CC(output_file=output_file_EA_A, num_of_exc=number_of_EA_A)
            #print(f'CIS_E_EA_A {CIS_E_EA_A}')
            #CIS_E_EA_B = get_exc_energy_CC(output_file=output_file_EA_B, num_of_exc=number_of_EA_B)  
            
            SCF_E_gs_A = -114.20855658180459
            SCF_E_gs_B = -114.20855658180459
            CIS_E_exc_A = -113.843007364566532 #n-pi*:-114.061044110256447
            CIS_E_exc_B = -113.843007364566532 #n-pi*:-114.061044110256447
            CIS_E_IP_A = -113.688984414929237 #n-pi*:-113.828050134980984 
            CIS_E_IP_B = -113.688984414929237 #n-pi*:-113.828050134980984 
            CIS_E_EA_A = -114.115873701891203        
            CIS_E_EA_B = -114.115873701891203        

    input_mat = readinput('geom_input')
    dimer = input_mat[1]
    state = input_mat[3]
    basis = input_mat[9]
    vdW_cont = get_vdW(input_file='geom_input', basis=basis, dimer=dimer)
    exch_cont = get_exch(input_file='geom_input', basis=basis, dimer=dimer)
    elstat_cont_exc = get_elstat(input_file='geom_input', basis=basis, dimer=dimer, exc_state=state)
    elstat_cont_CT = get_elstat(input_file='geom_input', basis=basis, dimer=dimer, exc_state='CT-CT')
    elstat_cont_gr = get_elstat(input_file='geom_input', basis=basis, dimer=dimer, exc_state='gr state')
    reference_cont = get_elstat(input_file='geom_input', basis=basis, dimer=dimer, exc_state='reference')
    eig_val_with_E, final_results, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final, H_31=V_31_final, H_32=V_32_final, H_41=V_41_final, H_42=V_42_final, H_43=V_43_final,
                                        H_12=V_12_final, H_13=V_13_final, H_23=V_23_final, H_14=V_14_final, H_24=V_24_final, H_34=V_34_final,
                                        H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
                                        SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
                                        CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) # here the distance is in angstrohm!!!
    
    with open(f'eigenvector_{distance}.txt', 'w+') as f:
        for jdx in range(len(eig_vec_4by4)):
            f.write(f'{eig_vec_4by4[:, jdx].T}')
    
    final_exc_state_split_file = 'results_final_states.txt'
    final_exc_state_file = 'E_incl_final_states.txt'
    final_exc_state_file_no_int = 'E_incl_final_states_with_GS.txt'
    E_full_file = 'E_full.txt'
    
    eig_val_with_E, final_results_no_same, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_no_same, H_31=V_31_final_no_same, H_32=V_32_final_no_same, H_41=V_41_final_no_same, H_42=V_42_final_no_same, H_43=V_43_final_no_same,
                                        H_12=V_12_final_no_same, H_13=V_13_final_no_same, H_23=V_23_final_no_same, H_14=V_14_final_no_same, H_24=V_24_final_no_same, H_34=V_34_final_no_same,
                                        H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
                                        SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
                                        CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) # here the distance is in angstrohm!!!
    E_full_file_no_same = 'E_no_same_full.txt'
    
    #E_full_file_no_mix = 'E_no_mix_full.txt'
    #E_full_file_mix_only = 'E_full_mix_only.txt'
    #E_full_file_Frenkel_only = 'E_full_Frenkel_only.txt'
    #E_full_file_CT_only = 'E_full_CT_only.txt'
    #E_full_file_31 = 'E_full_31.txt'
    #E_full_file_41 = 'E_full_41.txt'
    #E_full_file_3by3 = 'E_full_3by3.txt'
    eig_val_with_E, final_results_no_same_dc, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_no_same_dc, H_31=V_31_final_no_same_dc, H_32=V_32_final_no_same_dc, H_41=V_41_final_no_same_dc, H_42=V_42_final_no_same_dc, H_43=V_43_final_no_same_dc,
                                        H_12=V_12_final_no_same_dc, H_13=V_13_final_no_same_dc, H_23=V_23_final_no_same_dc, H_14=V_14_final_no_same_dc, H_24=V_24_final_no_same_dc, H_34=V_34_final_no_same_dc,
                                        H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
                                        SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
                                        CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) # here the distance is in angstrohm!!!
    E_full_file_no_same_dc = 'E_no_same_dc_full.txt'
    
    #E_full_file_no_mix = 'E_no_mix_full.txt'
    #E_full_file_mix_only = 'E_full_mix_only.txt'
    #E_full_file_Frenkel_only = 'E_full_Frenkel_only.txt'
    #E_full_file_CT_only = 'E_full_CT_only.txt'
    #E_full_file_31 = 'E_full_31.txt'
    #E_full_file_41 = 'E_full_41.txt'
    #E_full_file_3by3 = 'E_full_3by3.txt'
    eig_val_with_E, final_results_with_dc, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_with_dc, H_31=V_31_final_with_dc, H_32=V_32_final_with_dc, H_41=V_41_final_with_dc, H_42=V_42_final_with_dc, H_43=V_43_final_with_dc,
                                        H_12=V_12_final_with_dc, H_13=V_13_final_with_dc, H_23=V_23_final_with_dc, H_14=V_14_final_with_dc, H_24=V_24_final_with_dc, H_34=V_34_final_with_dc,
                                        H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
                                        SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
                                        CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) 
    E_full_with_dc_file = 'E_full_with_dc.txt'
    #E_strict_no_Ov_file_mix_only     = 'E_strict_no_Ov_mix_only.txt'
    #E_strict_no_Ov_file_Frenkel_only = 'E_strict_no_Ov_Frenkel_only.txt'
    #E_strict_no_Ov_file_CT_only      = 'E_strict_no_Ov_CT_only.txt'
    #E_strict_no_Ov_file_31           = 'E_strict_no_Ov_31.txt'
    #E_strict_no_Ov_file_41           = 'E_strict_no_Ov_41.txt'
    #E_strict_no_Ov_file_no_mix = 'E_no_mix_strict_no_Ov.txt'
    #E_strict_no_Ov_file_3by3 = 'E_strict_no_Ov_3by3.txt'
    eig_val_with_E, final_results_twoel_only, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_twoel_only, H_31=V_31_final_twoel_only, H_32=V_32_final_twoel_only, H_41=V_41_final_twoel_only, H_42=V_42_final_twoel_only, H_43=V_43_final_twoel_only,
                                        H_12=V_12_final_twoel_only, H_13=V_13_final_twoel_only, H_23=V_23_final_twoel_only, H_14=V_14_final_twoel_only, H_24=V_24_final_twoel_only, H_34=V_34_final_twoel_only,
                                        H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
                                        SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
                                        CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) 
    E_twoel_only_file = 'E_twoel_only.txt'
    
    eig_val_with_E, final_results_no_S, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_no_S, H_31=V_31_final_no_S, H_32=V_32_final_no_S, H_41=V_41_final_no_S, H_42=V_42_final_no_S, H_43=V_43_final_no_S,
                                        H_12=V_12_final_no_S, H_13=V_13_final_no_S, H_23=V_23_final_no_S, H_14=V_14_final_no_S, H_24=V_24_final_no_S, H_34=V_34_no_S,
                                        H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix_no_S, ov_matrix_Gs=supsys_Overlap_matrix_GS,
                                        SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
                                        CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) # here the distance is in angstrohm!!!
    E_no_S_file = 'E_no_S'
    
    #eig_val_with_E, final_results_my_approx_avg, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_my_approx_avg, H_31=V_31_final_my_approx_avg, H_32=V_32_final_my_approx_avg, H_41=V_41_final_my_approx_avg, H_42=V_42_final_my_approx_avg, H_43=V_43_final_my_approx_avg,
    #                                    H_12=V_12_final_my_approx_avg, H_13=V_13_final_my_approx_avg, H_23=V_23_final_my_approx_avg, H_14=V_14_final_my_approx_avg, H_24=V_24_final_my_approx_avg, H_34=V_34_final_my_approx_avg,
    #                                    H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
    #                                    SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
    #                                    CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) # here the distance is in angstrohm!!!
    #E_my_approx_avg_file = 'E_my_approx_avg.txt'
    #
    #eig_val_with_E, final_results_my_approx_avg_dc, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_my_approx_avg_dc, H_31=V_31_final_my_approx_avg_dc, H_32=V_32_final_my_approx_avg_dc, H_41=V_41_final_my_approx_avg_dc, H_42=V_42_final_my_approx_avg_dc, H_43=V_43_final_my_approx_avg_dc,
    #                                    H_12=V_12_final_my_approx_avg_dc, H_13=V_13_final_my_approx_avg_dc, H_23=V_23_final_my_approx_avg_dc, H_14=V_14_final_my_approx_avg_dc, H_24=V_24_final_my_approx_avg_dc, H_34=V_34_final_my_approx_avg_dc,
    #                                    H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
    #                                    SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
    #                                    CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) # here the distance is in angstrohm!!!
    #E_my_approx_avg_dc_file = 'E_my_approx_avg_dc.txt'
    
    #E_no_Ov_file_mix_only     = 'E_no_Ov_mix_only.txt'
    #E_no_Ov_file_Frenkel_only = 'E_no_Ov_Frenkel_only.txt'
    #E_no_Ov_file_CT_only      = 'E_no_Ov_CT_only.txt'
    #E_no_Ov_file_31           = 'E_no_Ov_31.txt'
    #E_no_Ov_file_41           = 'E_no_Ov_41.txt'
    #E_no_Ov_file_no_mix = 'E_no_mix_cross_only.txt'
    #E_no_Ov_file_3by3 = 'E_no_Ov_3by3.txt'
    #final_results_twoel_only, final_results_with_E, E_no_mix_twoel_only, E_twoel_only, E_twoel_mix_only, E_twoel_Frenkel_only, E_twoel_CT_only, E_twoel_31, E_twoel_41, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_twoel_only, H_31=V_31_final_twoel_only, H_32=V_32_final_twoel_only, H_41=V_41_final_twoel_only, H_42=V_42_final_twoel_only, H_43=V_43_final_twoel_only,
    #                                    H_12=V_12_final, H_13=V_13_final, H_23=V_23_final, H_14=V_14_final, H_24=V_24_final, H_34=V_34_final,
    #                                    H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
    #                                    SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
    #                                    CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) 
    #E_twoel_only_file = 'E_twoel_only.txt'
    #E_twoel_only_file_no_mix = 'E_no_mix_twoel_only.txt'
    #E_twoel_only_file_mix_only     = 'E_twoel_only_mix_only.txt'
    #E_twoel_only_file_Frenkel_only = 'E_twoel_only_Frenkel_only.txt'
    #E_twoel_only_file_CT_only      = 'E_twoel_only_CT_only.txt'
    #E_twoel_only_file_31           = 'E_twoel_only_31.txt'
    #E_twoel_only_file_41           = 'E_twoel_only_41.txt'
    #E_twoel_only_file_3by3 = 'E_twoel_only_3by3.txt'
    #final_results_full_with_dc, final_results_with_E, final_results_with_E_no_int, E_full_with_dc, E_full_with_dc_mix_only, E_full_with_dc_Frenkel_only, E_full_with_dc_CT_only, E_full_with_dc_31, E_full_with_dc_41, eig_vec_4by4 = diagonalize_final_H(H_21=V_21_final_with_dc, H_31=V_31_final_with_dc, H_32=V_32_final_with_dc, H_41=V_41_final_with_dc, H_42=V_42_final_with_dc, H_43=V_43_final_with_dc,
    #                                    H_12=V_12_final, H_13=V_13_final, H_23=V_23_final, H_14=V_14_final, H_24=V_24_final, H_34=V_34_final,
    #                                    H_01=V_01_final, H_02=V_02_final, H_03=V_03_final, H_04=V_04_final, ov_matrix=supsys_Overlap_matrix, ov_matrix_Gs=supsys_Overlap_matrix_GS,
    #                                    SCF_E_gs_A=SCF_E_gs_A, SCF_E_gs_B=SCF_E_gs_B, CIS_E_exc_A=CIS_E_exc_A, CIS_E_exc_B=CIS_E_exc_B, 
    #                                    CIS_E_IP_A=CIS_E_IP_A, CIS_E_IP_B=CIS_E_IP_B, CIS_E_EA_A=CIS_E_EA_A, CIS_E_EA_B=CIS_E_EA_B, dist=distance, vdW=vdW_cont, elstat_exc=elstat_cont_exc, elstat_gr=elstat_cont_gr, norm=np.float64(1.), exch=exch_cont, CT_elstat=elstat_cont_CT, ref_cont=reference_cont) 
    #E_full_with_dc_file_symm = 'E_full_with_dc_symm.txt'
    #E_full_with_dc_file = 'E_full_with_dc.txt'
    #E_full_with_dc_file_mix_only     = 'E_full_with_dc_mix_only.txt'
    #E_full_with_dc_file_Frenkel_only = 'E_full_with_dc_Frenkel_only.txt'
    #E_full_with_dc_file_CT_only      = 'E_full_with_dc_CT_only.txt'
    #E_full_with_dc_file_31           = 'E_full_with_dc_31.txt'
    #E_full_with_dc_file_41           = 'E_full_with_dc_41.txt'
    #E_full_with_dc_file_3by3 = 'E_full_with_dc_3by3.txt'
    write_energy_results(E_full_file, final_results, distance)
    write_energy_results(E_full_file_no_same, final_results_no_same, distance)
    write_energy_results(E_full_file_no_same_dc, final_results_no_same_dc, distance)
    write_energy_results(E_full_with_dc_file, final_results_with_dc, distance)
    write_energy_results(E_twoel_only_file, final_results_twoel_only, distance)
    write_energy_results(E_no_S_file, final_results_no_S, distance)
    #write_energy_results(E_my_approx_avg_file, final_results_my_approx_avg, distance)
    #write_energy_results(E_my_approx_avg_dc_file, final_results_my_approx_avg_dc, distance)


# %%
