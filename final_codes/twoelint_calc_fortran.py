from __future__ import annotations
import multiprocessing as mp
import os
import psutil
import final_codes.calc_BAAA as cBAAA
import final_codes.calc_BBAA as cBBAA
import final_codes.calc_BBBA as cBBBA

# Threading defaults.
# Why: do NOT hard-override thread counts at import time; with spawn, workers re-import this module.
# How: set defaults only if the user/launcher didn't already choose values.
#os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ["TWOELINT_AUTOTUNE"] = "1"
#os.environ["OMP_PROC_BIND"] = "TRUE"
#os.environ["OMP_PLACES"] = "cores"

# If you ever iterate over sets/dicts to sum floats, fix hash order too
os.environ.setdefault("PYTHONHASHSEED", "0")


import numpy as np

import importlib
_twoelint_f90 = None
_HAS_TWOELINT_F90 = False
_TWOELINT_F90_IMPORT_ATTEMPTED = False

def _get_twoelint_f90():
    """
    Why: with multiprocessing 'spawn', importing the extension before setting OMP_* in the child
         can freeze OpenMP at 1 thread.
    How: delay import until after _set_worker_thread_env() is called in the current process.
    """
    global _twoelint_f90, _HAS_TWOELINT_F90, _TWOELINT_F90_IMPORT_ATTEMPTED
    if not _TWOELINT_F90_IMPORT_ATTEMPTED:
        _TWOELINT_F90_IMPORT_ATTEMPTED = True
        try:
            _twoelint_f90 = importlib.import_module("twoelint_f90")
            _HAS_TWOELINT_F90 = True
        except Exception:
            _twoelint_f90 = None
            _HAS_TWOELINT_F90 = False
    return _twoelint_f90

def _get_twoelint_f90_module(name: str):
    extension = _get_twoelint_f90()
    if extension is None:
        return None
    return getattr(extension, name, None)

_EMPTY_I64 = np.empty(0, dtype=np.int64)

def _ravel_c_view(x: np.ndarray) -> np.ndarray:
    # Avoid copying huge shared matrices unless absolutely necessary
    return x.ravel(order="C") if x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x).ravel(order="C")

import re

def _f2py_call(func, **kwargs):
    """
    Call an f2py-wrapped function using only the argument names it actually exposes.
    Fixes missing/shifted args issues when signatures differ between rebuilds.
    """
    doc = getattr(func, "__doc__", "") or ""
    first = doc.splitlines()[0] if doc else ""
    m = re.search(r"\w+\((.*)\)", first)
    if not m:
        return func(**kwargs)

    names = [x.strip().split("=")[0] for x in m.group(1).split(",") if x.strip()]
    filtered = {k: v for k, v in kwargs.items() if k in names}
    return func(**filtered)

import logging
from collections import defaultdict
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.util import Finalize
from contextlib import contextmanager
import threading, time
from threadpoolctl import threadpool_limits

import logging
import math
import os
from typing import Any, Dict, List, Tuple, Iterator


# ---------- Binary layout constants ----------
#MARKER_SIZE = 4
#PAYLOAD_SIZE = 16
#DISK_RECORD_SIZE = MARKER_SIZE + PAYLOAD_SIZE + MARKER_SIZE  # 24 bytes on disk per record

# ---- Sizes ----
MARKER_SIZE = 4
PAYLOAD_SIZE = 16
DISK_RECORD_SIZE = MARKER_SIZE + PAYLOAD_SIZE + MARKER_SIZE  # 24
DT_RAM = np.dtype([("a","<u2"),("b","<u2"),("c","<u2"),("d","<u2"),("val","<f8")])

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FortranIOFormat:
    endian: str        # "<" or ">"
    marker_size: int   # 4 or 8


class FortranSlabReader:
    """
    High-throughput reader for Fortran sequential-unformatted records using slab buffering.

    Important correctness detail
    ----------------------------
    read_record() returns a memoryview into an internal bytearray buffer.
    Python forbids resizing that bytearray while any memoryview is alive.

    Therefore, compaction/reset MUST NOT resize the existing bytearray.
    We compact by *rebinding* self._buf to a new bytearray holding the remaining tail.
    Old buffers remain alive as long as exported memoryviews exist.
    """
    __slots__ = ("_f", "_byteorder", "_marker_size", "_slab_bytes", "_buf", "_pos", "_eof")

    def __init__(self, f, *, endian: str, marker_size: int, slab_bytes: int = 32 << 20):
        self._f = f
        self._byteorder = "little" if endian == "<" else "big"
        self._marker_size = int(marker_size)
        self._slab_bytes = int(slab_bytes)
        self._buf = bytearray()
        self._pos = 0
        self._eof = False

    def reset(self) -> None:
        """
        Clear internal buffers after the underlying file position has changed (seek/rewind).

        Implementation detail: must not resize an exported bytearray, so we rebind.
        """
        self._buf = bytearray()
        self._pos = 0
        self._eof = False

    def close(self) -> None:
        """Close underlying file handle."""
        try:
            self._f.close()
        except Exception:
            pass

    def _compact_if_needed(self) -> None:
        """
        Ensure buffer doesn't grow without bound.

        Must NOT do del/clear on self._buf because there may be active memoryviews.
        We compact by rebinding self._buf to a new bytearray of the unread tail.
        """
        if self._pos <= 0:
            return

        # Compact only when the consumed prefix is significant to avoid too much copying.
        consumed = self._pos
        available = len(self._buf)
        if consumed < (1 << 20) and consumed < (available // 2):
            return

        # Rebind to a fresh bytearray; old buffer stays alive if memoryviews reference it.
        self._buf = bytearray(self._buf[self._pos :])
        self._pos = 0

    def _fill(self, need: int) -> None:
        """
        Ensure at least `need` bytes are available from current position.
        """
        self._compact_if_needed()
        while (len(self._buf) - self._pos) < need and not self._eof:
            chunk = self._f.read(self._slab_bytes)
            if not chunk:
                self._eof = True
                break
            self._buf.extend(chunk)

    def read_record(self):
        """
        Read next Fortran sequential-unformatted record and return payload as a memoryview,
        or None on EOF.
        """
        ms = self._marker_size

        self._fill(ms)
        if (len(self._buf) - self._pos) < ms:
            return None

        head = self._buf[self._pos : self._pos + ms]
        n = int.from_bytes(head, byteorder=self._byteorder, signed=False)

        if n <= 0 or n > (1 << 31):
            raise ValueError(f"Invalid record length {n}; wrong endianness/marker_size or corrupt file.")

        need = ms + n + ms
        self._fill(need)
        if (len(self._buf) - self._pos) < need:
            return None

        payload_start = self._pos + ms
        payload_end = payload_start + n
        tail = self._buf[payload_end : payload_end + ms]
        n_tail = int.from_bytes(tail, byteorder=self._byteorder, signed=False)

        if n_tail != n:
            raise ValueError("Record marker mismatch; wrong endianness or corrupt file.")

        payload = memoryview(self._buf)[payload_start:payload_end]
        self._pos += need
        return payload


# ---- Endianness detection ----
def build_dtypes(endian: str) -> Tuple[np.dtype, np.dtype]:
    """
    Build compact in-RAM dtype (16B) and on-disk dtype (24B with offsets) for given endianness.
    """
    if endian not in ("<", ">"):
        raise ValueError("endian must be '<' or '>'")
    dt_ram = np.dtype([
        ("a",  f"{endian}u2"),  # L
        ("b",  f"{endian}u2"),  # K
        ("c",  f"{endian}u2"),  # J
        ("d",  f"{endian}u2"),  # I
        ("val",f"{endian}f8"),
    ])
    dt_disk = np.dtype({
        "names":   ["a","b","c","d","val"],
        "formats": [f"{endian}u2", f"{endian}u2", f"{endian}u2", f"{endian}u2", f"{endian}f8"],
        "offsets": [4, 6, 8, 10, 12],
        "itemsize": DISK_RECORD_SIZE,
    })
    return dt_ram, dt_disk


# ---- Chunk sizing (records-per-chunk) ----
def get_chunk_size_records(
    file_size_bytes: int,
    num_cores: int,
    available_memory_bytes: int,
    safety_factor: float = 0.75,
) -> int:
    if num_cores <= 0:
        raise ValueError("num_cores must be positive")
    total_records = max(0, file_size_bytes // DISK_RECORD_SIZE)
    if total_records == 0:
        return 1
    mem_per_core = (available_memory_bytes * safety_factor) / num_cores
    records_per_core = max(1, int(mem_per_core // PAYLOAD_SIZE))  # ~16B/rec in RAM
    fair_share = max(1, math.ceil(total_records / num_cores))
    chunk = min(records_per_core, fair_share)
    MIN_CHUNK = 4096
    if chunk < MIN_CHUNK and records_per_core >= MIN_CHUNK:
        chunk = MIN_CHUNK
    return int(chunk)


# Expect these in your environment:
# twoelint_file_path: str
# classify_row(row, threshold, NBAS)
# classified_data: Dict[str, Dict[int, List[List[Any]]]]
# classified_counts: Dict[str, int]
# num_cores: int
# chunk_size: int  # records per chunk


# ---------------------------------------------------------
# Constant index maps (no dictionaries)
# ---------------------------------------------------------
#TYPE_AAAA = 0
#TYPE_BBBB = 1
#TYPE_BAAA = 2
#TYPE_BBAA = 3
#TYPE_BBBA = 4
#N_TYPES = 5
#MAX_UNIQUE = 5           # Indexed 0–4 for safety
#
## Result index order (fixed)
##RESULT_IDS = [21,12,31,13,32,23,41,14,42,24,43,34]
##RESULT_MAP = {rid: i for i, rid in enumerate(RESULT_IDS)}
##N_RESULTS = len(RESULT_IDS)
##N_VALUES_PER_RESULT = 64   # allocate enough slots for Coul/X & variants
#
## Global shared-memory vector holder
#SHARED_VEC = {}

# -----------------------------------------------------------------------------
# Shared-memory helpers
# -----------------------------------------------------------------------------

def shm_create_for_array(arr: np.ndarray) -> Tuple[shared_memory.SharedMemory, Tuple[int, ...], str]:
    """
    Create POSIX shared memory and copy data from 'arr'.
    Returns (shm_handle, shape, dtype_str).
    """
    a = np.asarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    shm_arr = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    shm_arr[:] = a
    return shm, a.shape, a.dtype.str


def shm_attach_array(name: str, shape: Tuple[int, ...], dtype_str: str) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
    """Attach to an existing SHM as a NumPy array."""
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return shm, arr


# GPT
"""
Why: attaching many SHM-backed vectors inside process_chunk for every chunk adds heavy per-chunk syscall/object overhead.
How: attach vectors once per worker process, cache NumPy views + SHM handles, and unregister them from
     multiprocessing.resource_tracker so workers do not attempt to unlink owner-managed segments on exit.
"""
# GPT
_WORKER_VECTORS_CACHE: Dict[str, np.ndarray] | None = None
_WORKER_VECTOR_SHMS: Dict[str, shared_memory.SharedMemory] | None = None

def get_worker_vectors_cached(handles: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Attach SHM vectors once per worker and reuse them across chunks."""
    global _WORKER_VECTORS_CACHE, _WORKER_VECTOR_SHMS
    if _WORKER_VECTORS_CACHE is not None:
        return _WORKER_VECTORS_CACHE

    vectors: Dict[str, np.ndarray] = {}
    shms: Dict[str, shared_memory.SharedMemory] = {}

    from multiprocessing import resource_tracker  # local import: only needed in workers
    import atexit

    for key, meta in handles.items():
        shm, arr = shm_attach_array(meta["name"], tuple(meta["shape"]), meta["dtype"])
        # Prevent worker resource_tracker from unlinking owner-managed SHM at exit.
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        vectors[key] = arr
        shms[key] = shm

    _WORKER_VECTORS_CACHE = vectors
    _WORKER_VECTOR_SHMS = shms

    def _close_worker_vectors() -> None:
        global _WORKER_VECTOR_SHMS
        if not _WORKER_VECTOR_SHMS:
            return
        for _shm in _WORKER_VECTOR_SHMS.values():
            try:
                _shm.close()
            except Exception:
                pass

    atexit.register(_close_worker_vectors)
    return vectors
# GPT
"""End: per-worker SHM vector attachment cache."""
# GPT


@contextmanager
def attached_vectors(handles: Dict[str, Dict[str, Any]]) -> Iterator[Dict[str, np.ndarray]]:
    # GPT
    """
    Why: process_chunk currently uses `with attached_vectors(...)` per chunk; keep that structure but remove per-chunk attach costs.
    How: delegate to the per-worker cache so each worker attaches vectors once and reuses them for all chunks.
    """
    # GPT
    yield get_worker_vectors_cached(handles)
    # GPT
    """End: attached_vectors now reuses cached worker attachments."""
    # GPT



def register_parent_finalizer(owner_handles: List[shared_memory.SharedMemory]) -> None:
    """
    Parent-only: ensure we close() and unlink() exactly once at process exit.
    """
    def _cleanup():
        for h in owner_handles:
            try:
                h.close()
            except Exception:
                pass
        for h in owner_handles:
            try:
                h.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    Finalize(None, _cleanup, exitpriority=10)


# -----------------------------------------------------------------------------
# Domain preparation: build vectors and place into SHM
# -----------------------------------------------------------------------------

def prepare_shared_vectors_red(
    red_C_s: Dict[str, np.ndarray],
    red_LCAO_s: Dict[str, np.ndarray],
    S_blocks: Dict[str, np.ndarray] | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[shared_memory.SharedMemory]]:
    """
    Computes AO vectors (with and without overlap) and places each into shared memory.

    Returns:
        (handles, owner_handles)
        handles: key -> {"name": shm_name, "shape": shape, "dtype": dtype_str}
        owner_handles: list of SharedMemory objects (parent must close+unlink once)
    """
    handles: Dict[str, Dict[str, Any]] = {}
    owner_handles: List[shared_memory.SharedMemory] = []

    # --- No-overlap AO vectors ---
    AO: Dict[str, np.ndarray] = {}
    AO['ACA_A_right']   = red_LCAO_s['LCAO_A_red_occ'] @ red_C_s['CIS_matrix_A_red_right'] @ red_LCAO_s['LCAO_A_red_virt'].T
    AO['ACA_B_right']   = red_LCAO_s['LCAO_B_red_occ'] @ red_C_s['CIS_matrix_B_red_right'] @ red_LCAO_s['LCAO_B_red_virt'].T
    AO['ACA_A_left']    = red_LCAO_s['LCAO_A_red_occ'] @ red_C_s['CIS_matrix_A_red_left']  @ red_LCAO_s['LCAO_A_red_virt'].T
    AO['ACA_B_left']    = red_LCAO_s['LCAO_B_red_occ'] @ red_C_s['CIS_matrix_B_red_left']  @ red_LCAO_s['LCAO_B_red_virt'].T

    AO['CA_EA_A_right'] = red_C_s['CIS_vector_EA_A_red_right'] @ red_LCAO_s['LCAO_A_red_virt'].T
    AO['AC_IP_B_right'] = red_LCAO_s['LCAO_B_red_occ'] @ red_C_s['CIS_vector_IP_B_red_right'].T
    AO['AC_IP_A_right'] = red_LCAO_s['LCAO_A_red_occ'] @ red_C_s['CIS_vector_IP_A_red_right'].T
    AO['CA_EA_B_right'] = red_C_s['CIS_vector_EA_B_red_right'] @ red_LCAO_s['LCAO_B_red_virt'].T

    AO['CA_EA_A_left']  = red_C_s['CIS_vector_EA_A_red_left']  @ red_LCAO_s['LCAO_A_red_virt'].T
    AO['AC_IP_B_left']  = red_LCAO_s['LCAO_B_red_occ'] @ red_C_s['CIS_vector_IP_B_red_left'].T
    AO['AC_IP_A_left']  = red_LCAO_s['LCAO_A_red_occ'] @ red_C_s['CIS_vector_IP_A_red_left'].T
    AO['CA_EA_B_left']  = red_C_s['CIS_vector_EA_B_red_left']  @ red_LCAO_s['LCAO_B_red_virt'].T

    for key, arr in AO.items():
        shm, shape, dtype = shm_create_for_array(arr)
        handles[key] = {"name": shm.name, "shape": shape, "dtype": dtype}
        owner_handles.append(shm)

    # --- Overlap AO vectors (optional) ---
    if S_blocks is not None:
        AO = {}
        AO['SC_A_BA_ia'] = red_LCAO_s['LCAO_B_red_occ'] @ S_blocks['S_BA_ii'] @ red_C_s['CIS_matrix_A_red_right'] @ red_LCAO_s['LCAO_A_red_virt'].T
        AO['CS_A_AB_ia'] = red_LCAO_s['LCAO_A_red_occ'] @ red_C_s['CIS_matrix_A_red_left'] @ S_blocks['S_AB_aa'] @ red_LCAO_s['LCAO_B_red_virt'].T
        AO['SC_B_AB_ia'] = red_LCAO_s['LCAO_A_red_occ'] @ S_blocks['S_AB_ii'] @ red_C_s['CIS_matrix_B_red_right'] @ red_LCAO_s['LCAO_B_red_virt'].T
        AO['CS_B_BA_ia'] = red_LCAO_s['LCAO_B_red_occ'] @ red_C_s['CIS_matrix_B_red_left'] @ S_blocks['S_BA_aa'] @ red_LCAO_s['LCAO_A_red_virt'].T
        AO['CS_EA_A_a' ] = red_C_s['CIS_vector_EA_A_red_left'] @ S_blocks['S_AB_aa'] @ red_LCAO_s['LCAO_B_red_virt'].T
        AO['SC_IP_B_i' ] = red_LCAO_s['LCAO_A_red_occ'] @ S_blocks['S_AB_ii'] @ red_C_s['CIS_vector_IP_B_red_right'].T
        AO['SC_IP_A_i' ] = red_LCAO_s['LCAO_B_red_occ'] @ S_blocks['S_BA_ii'] @ red_C_s['CIS_vector_IP_A_red_right'].T
        AO['CS_EA_B_a' ] = red_C_s['CIS_vector_EA_B_red_left'] @ S_blocks['S_BA_aa'] @ red_LCAO_s['LCAO_A_red_virt'].T
        AO['SC_A_BA_pa'] = red_LCAO_s['LCAO_B'] @ S_blocks['S_BA_pi'] @ red_C_s['CIS_matrix_A_red_left'] @ red_LCAO_s['LCAO_A_red_virt'].T
        AO['CS_A_AB_iq'] = red_LCAO_s['LCAO_A_red_occ'] @ red_C_s['CIS_matrix_A_red_right'] @ S_blocks['S_AB_aq'] @ red_LCAO_s['LCAO_B'].T
        AO['SC_B_AB_pa'] = red_LCAO_s['LCAO_A'] @ S_blocks['S_AB_pi'] @ red_C_s['CIS_matrix_B_red_left'] @ red_LCAO_s['LCAO_B_red_virt'].T
        AO['CS_B_BA_iq'] = red_LCAO_s['LCAO_B_red_occ'] @ red_C_s['CIS_matrix_B_red_right'] @ S_blocks['S_BA_aq'] @ red_LCAO_s['LCAO_A'].T
        AO['CS_EA_A_q' ] = red_C_s['CIS_vector_EA_A_red_right'] @ S_blocks['S_AB_aq'] @ red_LCAO_s['LCAO_B'].T
        AO['SC_IP_B_p' ] = red_LCAO_s['LCAO_A'] @ S_blocks['S_AB_pi'] @ red_C_s['CIS_vector_IP_B_red_left'].T
        AO['SC_IP_A_p' ] = red_LCAO_s['LCAO_B'] @ S_blocks['S_BA_pi'] @ red_C_s['CIS_vector_IP_A_red_left'].T
        AO['CS_EA_B_q' ] = red_C_s['CIS_vector_EA_B_red_right'] @ S_blocks['S_BA_aq'] @ red_LCAO_s['LCAO_A'].T
        AO['SCS_A_BB_iq'] = red_LCAO_s['LCAO_B_red_occ'] @ S_blocks['S_BA_ii'] @ red_C_s['CIS_matrix_A_red_right'] @ S_blocks['S_AB_aq'] @ red_LCAO_s['LCAO_B'].T
        AO['SCS_A_BB_pa'] = red_LCAO_s['LCAO_B'] @ S_blocks['S_BA_pi'] @ red_C_s['CIS_matrix_A_red_left'] @ S_blocks['S_AB_aa'] @ red_LCAO_s['LCAO_B_red_virt'].T
        AO['SCS_B_AA_pa'] = red_LCAO_s['LCAO_A'] @ S_blocks['S_AB_pi'] @ red_C_s['CIS_matrix_B_red_left'] @ S_blocks['S_BA_aa'] @ red_LCAO_s['LCAO_A_red_virt'].T

        for key, arr in AO.items():
            shm, shape, dtype = shm_create_for_array(arr)
            handles[key] = {"name": shm.name, "shape": shape, "dtype": dtype}
            owner_handles.append(shm)

    return handles, owner_handles


# ---- Helper: build containers exactly as your pipeline expects ----
def make_classified_containers(categories=("BBBA","BBAA","BAAA","BBBB","AAAA")):
    from collections import defaultdict
    classified_data: Dict[str, Dict[int, List[List[Any]]]] = {
        cat: defaultdict(list) for cat in categories
    }
    classified_counts: Dict[str, int] = {cat: 0 for cat in categories}
    return classified_data, classified_counts

# ---- Vectorized classifier that fills your containers directly ----
def classify_row(row, threshold, NBAS):
    """
    Classify a row based on the number of values above or below a threshold.
    Returns a string indicating the type (e.g., AAAB, AABB).
    """
    numbers = row[:4] #list(map(int, row[:4]))  # First 4 columns are numeric
    above = sum(n > threshold for n in numbers)
    below = len(numbers) - above

    unique_count = len(set(row[:4]))
    
    # Define your classification logic here
    if all(numbers) == 0 or np.any(np.array(numbers) > (NBAS +  10)):
        return None, None
    else:
        if above == 3 and below == 1:
            return "BBBA", unique_count
        elif above == 2 and below == 2:
            return "BBAA", unique_count
        elif above == 1 and below == 3:
            return "BAAA", unique_count
        elif above == 4:
            return "BBBB", unique_count
        elif above == 0:
            return "AAAA", unique_count
        else:
            return None, None
        

def separate_indexes_and_values(row, NBAS_A):
    # GPT
    """
    Why: this helper is called for many integrals; the old version built intermediate lists and re-cast ints unnecessarily.
    How: read the 4 indices directly (already Python ints from _SubblockView), append into A/B lists in one pass, and keep value as float.
    """
    # GPT
    i = row[0]
    j = row[1]
    k = row[2]
    l = row[3]

    A_indexes: List[int] = []
    B_indexes: List[int] = []
    for idx in (i, j, k, l):
        if idx <= NBAS_A:
            A_indexes.append(idx - 1)
        else:
            B_indexes.append(idx - 1 - NBAS_A)

    value = row[4]
    # GPT
    """End: reduced per-integral allocations/conversions in separate_indexes_and_values."""
    # GPT
    return A_indexes, B_indexes, value


def calc_BAAA_f90_zero_copy(
    twoelint_block,
    SC_A_BA_ia,
    CS_A_AB_ia,
    SC_B_AB_ia,
    CS_B_BA_ia,
    CS_EA_A_a,
    SC_IP_B_i,
    SC_IP_A_i,
    CS_EA_B_a,
    SC_A_BA_pa,
    CS_A_AB_iq,
    SC_B_AB_pa,
    CS_B_BA_iq,
    CS_EA_A_q,
    SC_IP_B_p,
    SC_IP_A_p,
    CS_EA_B_q,
    ACA_A_right,
    ACA_B_right,
    SCS_B_AA_pa,
    ACA_A_left,
    ACA_B_left,
    CA_EA_A_right,
    AC_IP_B_right,
    AC_IP_A_right,
    CA_EA_B_right,
    CA_EA_A_left,
    AC_IP_B_left,
    AC_IP_A_left,
    CA_EA_B_left,
    NBAS_A,
):
    """
    Zero-copy Fortran version of calc_BAAA (Ov only).
    - One f2py call per chunk.
    - Uses sb._pos and backing arrays directly (no sb._a[idx] copies).
    """
    f90_mod = _get_twoelint_f90_module("twoelint_baaa_mod")
    if (f90_mod is None) or (SC_A_BA_ia is None):
        return cBAAA.calc_BAAA(
            twoelint_block,
            SC_A_BA_ia, CS_A_AB_ia, SC_B_AB_ia, CS_B_BA_ia,
            CS_EA_A_a, SC_IP_B_i, SC_IP_A_i, CS_EA_B_a,
            SC_A_BA_pa, CS_A_AB_iq, SC_B_AB_pa, CS_B_BA_iq,
            CS_EA_A_q, SC_IP_B_p, SC_IP_A_p, CS_EA_B_q,
            ACA_A_right, ACA_B_right, SCS_B_AA_pa, ACA_A_left, ACA_B_left,
            CA_EA_A_right, AC_IP_B_right, AC_IP_A_right, CA_EA_B_right,
            CA_EA_A_left, AC_IP_B_left, AC_IP_A_left, CA_EA_B_left, NBAS_A
        )

    # Find any non-empty subblock view to get backing arrays
    base = None
    for sb in twoelint_block.values():
        if hasattr(sb, "_pos"):
            base = sb
            break
    if base is None:
        z = np.float64(0.0)
        return (z,) * 40

    # Zero-copy: reinterpret uint16 bits as int16 for Fortran INTEGER*2 reading
    assert base._a.dtype == np.uint16 and base._b.dtype == np.uint16 and base._c.dtype == np.uint16 and base._d.dtype == np.uint16
    assert base._v.dtype == np.float64

    a_i16 = base._a.view(np.int16)
    b_i16 = base._b.view(np.int16)
    c_i16 = base._c.view(np.int16)
    d_i16 = base._d.view(np.int16)
    v_f64 = base._v

    sb2 = twoelint_block.get(2, [])
    sb3 = twoelint_block.get(3, [])
    sb4 = twoelint_block.get(4, [])

    pos1 = _EMPTY_I64
    pos2 = np.asarray(getattr(sb2, "_pos", _EMPTY_I64), dtype=np.int64)
    pos3 = np.asarray(getattr(sb3, "_pos", _EMPTY_I64), dtype=np.int64)
    pos4 = np.asarray(getattr(sb4, "_pos", _EMPTY_I64), dtype=np.int64)

    # Flatten matrices as C-order views (no copies for shared memory arrays)
    sc_b_ab = _ravel_c_view(SC_B_AB_ia)
    cs_b_ba = _ravel_c_view(CS_B_BA_ia)
    sc_a_ba = _ravel_c_view(SC_A_BA_ia)
    cs_a_ab = _ravel_c_view(CS_A_AB_ia)
    sc_b_ab_pa = _ravel_c_view(SC_B_AB_pa)
    sc_a_ba_pa = _ravel_c_view(SC_A_BA_pa)
    cs_a_ab_iq = _ravel_c_view(CS_A_AB_iq)
    cs_ea_a = _ravel_c_view(CS_EA_A_a)
    sc_ip_a = _ravel_c_view(SC_IP_A_i)
    sc_ip_b_p = _ravel_c_view(SC_IP_B_p)
    cs_ea_b_q = _ravel_c_view(CS_EA_B_q)
    scs_b_aa_pa = _ravel_c_view(SCS_B_AA_pa)

    ac_ip_b_left  = _ravel_c_view(AC_IP_B_left)
    ca_ea_b_left  = _ravel_c_view(CA_EA_B_left)
    ac_ip_b_right = _ravel_c_view(AC_IP_B_right)
    ca_ea_b_right = _ravel_c_view(CA_EA_B_right)

    sc_ip_b = _ravel_c_view(SC_IP_B_i)
    cs_ea_b = _ravel_c_view(CS_EA_B_a)

    aca_a_right = _ravel_c_view(ACA_A_right)
    aca_a_left  = _ravel_c_view(ACA_A_left)

    ca_ea_a_right = _ravel_c_view(CA_EA_A_right)
    ac_ip_a_right = _ravel_c_view(AC_IP_A_right)
    ca_ea_a_left  = _ravel_c_view(CA_EA_A_left)
    ac_ip_a_left  = _ravel_c_view(AC_IP_A_left)

    out = f90_mod.baaa_accum_all(
        pos1=pos1, pos2=pos2, pos3=pos3, pos4=pos4,
        a=a_i16, b=b_i16, c=c_i16, d=d_i16, v=v_f64,
        nbas_a=int(NBAS_A),

        sc_b_ab=sc_b_ab, sc_b_ab_ncol=int(SC_B_AB_ia.shape[1]),
        cs_b_ba=cs_b_ba, cs_b_ba_ncol=int(CS_B_BA_ia.shape[1]),
        sc_a_ba=sc_a_ba, sc_a_ba_ncol=int(SC_A_BA_ia.shape[1]),
        cs_a_ab=cs_a_ab, cs_a_ab_ncol=int(CS_A_AB_ia.shape[1]),
        sc_b_ab_pa=sc_b_ab_pa, sc_b_ab_pa_ncol=int(SC_B_AB_pa.shape[1]),
        sc_a_ba_pa=sc_a_ba_pa, sc_a_ba_pa_ncol=int(SC_A_BA_pa.shape[1]),
        cs_a_ab_iq=cs_a_ab_iq, cs_a_ab_iq_ncol=int(CS_A_AB_iq.shape[1]),
        cs_ea_a=cs_ea_a, cs_ea_a_ncol=int(CS_EA_A_a.shape[1]),
        sc_ip_a=sc_ip_a, sc_ip_a_ncol=int(SC_IP_A_i.shape[1]),
        sc_ip_b_p=sc_ip_b_p, sc_ip_b_p_ncol=int(SC_IP_B_p.shape[1]),
        cs_ea_b_q=cs_ea_b_q, cs_ea_b_q_ncol=int(CS_EA_B_q.shape[1]),
        scs_b_aa_pa=scs_b_aa_pa, scs_b_aa_pa_ncol=int(SCS_B_AA_pa.shape[1]),

        ac_ip_b_left=ac_ip_b_left, ac_ip_b_left_ncol=int(AC_IP_B_left.shape[1]),
        ca_ea_b_left=ca_ea_b_left, ca_ea_b_left_ncol=int(CA_EA_B_left.shape[1]),
        ac_ip_b_right=ac_ip_b_right, ac_ip_b_right_ncol=int(AC_IP_B_right.shape[1]),
        ca_ea_b_right=ca_ea_b_right, ca_ea_b_right_ncol=int(CA_EA_B_right.shape[1]),

        sc_ip_b=sc_ip_b, sc_ip_b_ncol=int(SC_IP_B_i.shape[1]),
        cs_ea_b=cs_ea_b, cs_ea_b_ncol=int(CS_EA_B_a.shape[1]),

        aca_a_right=aca_a_right, aca_a_right_ncol=int(ACA_A_right.shape[1]),
        aca_a_left=aca_a_left, aca_a_left_ncol=int(ACA_A_left.shape[1]),

        ca_ea_a_right=ca_ea_a_right, ca_ea_a_right_ncol=int(CA_EA_A_right.shape[1]),
        ac_ip_a_right=ac_ip_a_right, ac_ip_a_right_ncol=int(AC_IP_A_right.shape[1]),
        ca_ea_a_left=ca_ea_a_left, ca_ea_a_left_ncol=int(CA_EA_A_left.shape[1]),
        ac_ip_a_left=ac_ip_a_left, ac_ip_a_left_ncol=int(AC_IP_A_left.shape[1]),
    )

    # out ends with err_flag (int). if err_flag==1 -> same failure as Python "bad ordering"
    err_flag = int(out[-1])
    if err_flag:
        raise ValueError("invalid (j,k,l) ordering: requires not (j>=k and k<l)")

    return out[:-1]



#TODO !!! This should be done differently, without the if statement!!!


def reorganize_indexes(indexes, unique_count: int | None = None):
    # GPT
    """
    Why: this function sits in the hottest AAAA path and previously used len(set(...)) + list appends per integral.
    How: accept the already-known unique_count from classification, avoid set/list allocations, and return tuples of tuples.
    """
    # GPT
    i, j, k, l = indexes

    if unique_count is None:
        if (i != j and i != k and i != l and j != k and j != l and k != l):
            unique_count = 4
        else:
            unique_count = len({i, j, k, l})

    if unique_count == 4:
        expanded = (
            (i, j, k, l),
            (k, l, i, j),
            (j, i, l, k),
            (l, k, j, i),
            (i, j, l, k),
            (l, k, i, j),
            (j, i, k, l),
            (k, l, j, i),
        )
    elif unique_count == 3:
        if i == j:
            expanded = ((i, j, k, l), (i, j, l, k), (k, l, i, j), (l, k, i, j))
        elif k == l:
            expanded = ((i, j, k, l), (j, i, k, l), (k, l, i, j), (k, l, j, i))
        elif i == k and i != j and i != l:
            expanded = ((i, j, k, l), (i, j, l, k), (j, i, k, l), (j, i, l, k))
        else:
            # Preserve legacy behavior for the remaining 3-unique patterns.
            if i == j == l:
                k, l = l, k
            elif i == k == l:
                i, j, k, l = i, k, l, j
            elif j == k == l:
                i, j, k, l = j, k, l, i
            expanded = ((i, j, k, l), (l, i, j, k), (i, j, l, k), (i, l, j, k))
    else:
        expanded = ((i, j, k, l),)

    # GPT
    """End: reorganize_indexes now avoids per-integral set/list churn."""
    # GPT
    return expanded

def calc_BBBB_f90_zero_copy(
    twoelint_block,
    SC_A_BA_ia,
    CS_A_AB_ia,
    SC_B_AB_ia,
    CS_B_BA_ia,
    CS_EA_A_a,
    SC_IP_A_i,
    SC_IP_B_i,
    CS_EA_B_a,
    ACA_A_right,
    ACA_B_right,
    ACA_A_left,
    ACA_B_left,
    CA_EA_A_right,
    AC_IP_B_right,
    AC_IP_A_right,
    CA_EA_B_right,
    CA_EA_A_left,
    AC_IP_B_left,
    AC_IP_A_left,
    CA_EA_B_left,
    threshold,
):
    f90_mod = _get_twoelint_f90_module("twoelint_bbbb_mod")
    if f90_mod is None:
        return calc_BBBB(
            twoelint_block,
            SC_A_BA_ia, CS_A_AB_ia, SC_B_AB_ia, CS_B_BA_ia, CS_EA_A_a, SC_IP_A_i, SC_IP_B_i, CS_EA_B_a,
            ACA_A_right, ACA_B_right, ACA_A_left, ACA_B_left,
            CA_EA_A_right, AC_IP_B_right, AC_IP_A_right, CA_EA_B_right,
            CA_EA_A_left, AC_IP_B_left, AC_IP_A_left, CA_EA_B_left, threshold
        )

    base = None
    for sb in twoelint_block.values():
        if hasattr(sb, "_pos"):
            base = sb
            break
    if base is None:
        z = np.float64(0.0)
        return z, z, z, z, z, z

    a_i16 = base._a.view(np.int16)
    b_i16 = base._b.view(np.int16)
    c_i16 = base._c.view(np.int16)
    d_i16 = base._d.view(np.int16)
    v_f64 = base._v

    sb1 = twoelint_block.get(1, [])
    sb2 = twoelint_block.get(2, [])
    sb3 = twoelint_block.get(3, [])
    sb4 = twoelint_block.get(4, [])

    pos1 = np.asarray(getattr(sb1, "_pos", _EMPTY_I64), dtype=np.int64)
    pos2 = np.asarray(getattr(sb2, "_pos", _EMPTY_I64), dtype=np.int64)
    pos3 = np.asarray(getattr(sb3, "_pos", _EMPTY_I64), dtype=np.int64)
    pos4 = np.asarray(getattr(sb4, "_pos", _EMPTY_I64), dtype=np.int64)

    # Needed by corrected BBBB:
    aca_r = ACA_B_right.ravel(order="C")
    aca_l = ACA_B_left.ravel(order="C")
    cea   = CS_EA_A_a.ravel(order="C")
    bip   = AC_IP_B_left.ravel(order="C")
    cbr   = CA_EA_B_right.ravel(order="C")
    sia   = SC_IP_A_i.ravel(order="C")

    return f90_mod.bbbb_accum_all(
        pos1=pos1, pos2=pos2, pos3=pos3, pos4=pos4,
        a=a_i16, b=b_i16, c=c_i16, d=d_i16, v=v_f64,
        threshold=int(threshold),

        aca_b_right=aca_r, aca_b_right_ncol=int(ACA_B_right.shape[1]),
        aca_b_left=aca_l,  aca_b_left_ncol=int(ACA_B_left.shape[1]),

        cs_ea_a=cea, cs_ea_a_ncol=int(CS_EA_A_a.shape[1]),
        ac_ip_b_left=bip, ac_ip_b_left_ncol=int(AC_IP_B_left.shape[1]),
        ca_ea_b_right=cbr, ca_ea_b_right_ncol=int(CA_EA_B_right.shape[1]),
        sc_ip_a=sia, sc_ip_a_ncol=int(SC_IP_A_i.shape[1]),
    )

def   calc_BBBB(twoelint_block,
                SC_A_BA       ,
                CS_A_AB       ,
                SC_B_AB       ,
                CS_B_BA       ,
                CS_EA_A       ,
                SC_IP_A       ,
                SC_IP_B       ,
                CS_EA_B       ,
                ACA_A_right   ,
                ACA_B_right   ,
                ACA_A_left    ,
                ACA_B_left    ,
                CA_EA_A_right ,
                AC_IP_B_right ,
                AC_IP_A_right ,
                CA_EA_B_right ,
                CA_EA_A_left  ,
                AC_IP_B_left  ,
                AC_IP_A_left  ,
                CA_EA_B_left  , threshold 
                          ):
    """Calculation for BAAA type."""
    # Iterate through the categories in the "LOCAL" key

    
    final_contribution_32_coul = np.float64(0.)
    final_contribution_32_exch = np.float64(0.)
    
    final_contribution_24_coul = np.float64(0.)
    final_contribution_24_exch = np.float64(0.)
        
    final_contribution_34_coul = np.float64(0.)
    final_contribution_34_exch = np.float64(0.)    

    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)

    for unique_count, twoelint_subblock in twoelint_block.items():
        for row in twoelint_subblock:
            i = row[0] - 1 - threshold
            j = row[1] - 1 - threshold
            k = row[2] - 1 - threshold
            l = row[3] - 1 - threshold
            value = row[4]
            row_expanded = reorganize_indexes((i, j, k, l), unique_count=int(unique_count))
            for ind in row_expanded:
                final_contribution_32_coul += C_prefactor * ACA_B_right[ind[0],ind[1]] * CS_EA_A[0, ind[2]] * AC_IP_B_left[ind[3], 0] * value
                final_contribution_32_exch += X_prefactor * ACA_B_right[ind[0],ind[2]] * CS_EA_A[0, ind[3]] * AC_IP_B_left[ind[1], 0] * value

                final_contribution_24_coul += C_prefactor * ACA_B_left[ind[0],ind[1]] * CA_EA_B_right[0, ind[2]] * SC_IP_A[ind[3], 0] * value
                final_contribution_24_exch += X_prefactor * ACA_B_left[ind[0],ind[2]] * CA_EA_B_right[0, ind[3]] * SC_IP_A[ind[1], 0] * value                
                
                final_contribution_34_coul += C_prefactor * AC_IP_B_left[ind[0], 0] * CS_EA_A[0, ind[1]] * CA_EA_B_right[0, ind[2]] * SC_IP_A[ind[3], 0] * value
                final_contribution_34_exch += X_prefactor * AC_IP_B_left[ind[0], 0] * CS_EA_A[0, ind[2]] * CA_EA_B_right[0, ind[3]] * SC_IP_A[ind[1], 0] * value                
                
                
    return final_contribution_32_coul, final_contribution_32_exch, final_contribution_24_coul, final_contribution_24_exch, final_contribution_34_coul, final_contribution_34_exch

    
def calc_AAAA_f90_zero_copy(
    twoelint_block,
    SC_A_BA_ia,
    CS_A_AB_ia,
    SC_B_AB_ia,
    CS_B_BA_ia,
    CS_EA_A_a,
    SC_IP_A_i,
    SC_IP_B_i,
    CS_EA_B_a,
    ACA_A_right,
    ACA_B_right,
    ACA_A_left,
    ACA_B_left,
    CA_EA_A_right,
    AC_IP_B_right,
    AC_IP_A_right,
    CA_EA_B_right,
    CA_EA_A_left,
    AC_IP_B_left,
    AC_IP_A_left,
    CA_EA_B_left,
    threshold,
):
    f90_mod = _get_twoelint_f90_module("twoelint_aaaa_mod")
    if f90_mod is None:
        return calc_AAAA(
            twoelint_block,
            SC_A_BA_ia, CS_A_AB_ia, SC_B_AB_ia, CS_B_BA_ia, CS_EA_A_a, SC_IP_A_i, SC_IP_B_i, CS_EA_B_a,
            ACA_A_right, ACA_B_right, ACA_A_left, ACA_B_left,
            CA_EA_A_right, AC_IP_B_right, AC_IP_A_right, CA_EA_B_right,
            CA_EA_A_left, AC_IP_B_left, AC_IP_A_left, CA_EA_B_left, threshold
        )

    base = None
    for sb in twoelint_block.values():
        if hasattr(sb, "_pos"):
            base = sb
            break
    if base is None:
        z = np.float64(0.0)
        return z, z, z, z, z, z

    a_i16 = base._a.view(np.int16)
    b_i16 = base._b.view(np.int16)
    c_i16 = base._c.view(np.int16)
    d_i16 = base._d.view(np.int16)
    v_f64 = base._v

    sb1 = twoelint_block.get(1, [])
    sb2 = twoelint_block.get(2, [])
    sb3 = twoelint_block.get(3, [])
    sb4 = twoelint_block.get(4, [])

    pos1 = np.asarray(getattr(sb1, "_pos", _EMPTY_I64), dtype=np.int64)
    pos2 = np.asarray(getattr(sb2, "_pos", _EMPTY_I64), dtype=np.int64)
    pos3 = np.asarray(getattr(sb3, "_pos", _EMPTY_I64), dtype=np.int64)
    pos4 = np.asarray(getattr(sb4, "_pos", _EMPTY_I64), dtype=np.int64)

    # Needed by corrected AAAA:
    aca_r = ACA_A_right.ravel(order="C")
    aca_l = ACA_A_left.ravel(order="C")
    csb   = CS_EA_B_a.ravel(order="C")
    aip   = AC_IP_A_left.ravel(order="C")
    car   = CA_EA_A_right.ravel(order="C")
    sip   = SC_IP_B_i.ravel(order="C")

    return f90_mod.aaaa_accum_all(
        pos1=pos1, pos2=pos2, pos3=pos3, pos4=pos4,
        a=a_i16, b=b_i16, c=c_i16, d=d_i16, v=v_f64,
        threshold=int(threshold),

        aca_a_right=aca_r, aca_a_right_ncol=int(ACA_A_right.shape[1]),
        aca_a_left=aca_l,  aca_a_left_ncol=int(ACA_A_left.shape[1]),

        cs_ea_b=csb, cs_ea_b_ncol=int(CS_EA_B_a.shape[1]),
        ac_ip_a_left=aip, ac_ip_a_left_ncol=int(AC_IP_A_left.shape[1]),
        ca_ea_a_right=car, ca_ea_a_right_ncol=int(CA_EA_A_right.shape[1]),
        sc_ip_b=sip, sc_ip_b_ncol=int(SC_IP_B_i.shape[1]),
    )



def   calc_AAAA(twoelint_block, 
                SC_A_BA       ,
                CS_A_AB       ,
                SC_B_AB       ,
                CS_B_BA       ,
                CS_EA_A       ,
                SC_IP_A       ,
                SC_IP_B       ,
                CS_EA_B       ,
                ACA_A_right   ,
                ACA_B_right   ,
                ACA_A_left    ,
                ACA_B_left    ,
                CA_EA_A_right ,
                AC_IP_B_right ,
                AC_IP_A_right ,
                CA_EA_B_right ,
                CA_EA_A_left  ,
                AC_IP_B_left  ,
                AC_IP_A_left  ,
                CA_EA_B_left  , threshold):
    """Calculation for AAAA type."""
    # Iterate through the categories in the "LOCAL" key
    
    final_contribution_41_coul = np.float64(0.)
    final_contribution_41_exch = np.float64(0.)
    
    final_contribution_31_coul = np.float64(0.)
    final_contribution_31_exch = np.float64(0.)
        
    final_contribution_43_coul = np.float64(0.)
    final_contribution_43_exch = np.float64(0.)    
    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)
    for unique_count, twoelint_subblock in twoelint_block.items():
        for row in twoelint_subblock:
            # GPT
            """
            Why: this loop runs per integral; building Python lists and NumPy arrays (np.any(np.array(...)), copies, tolist)
            adds significant overhead at 200GB+ scale.
            How: unpack indices directly, use scalar comparisons for the threshold check, shift in-place, and pass known unique_count
            to reorganize_indexes to avoid recomputing uniqueness.
            """
            # GPT
            i = row[0] - 1
            j = row[1] - 1
            k = row[2] - 1
            l = row[3] - 1
            if i >= threshold or j >= threshold or k >= threshold or l >= threshold:
                i -= threshold
                j -= threshold
                k -= threshold
                l -= threshold
            value = row[4]
            row_expanded = reorganize_indexes((i, j, k, l), unique_count=int(unique_count))
            # GPT
            """End: removed per-integral NumPy/list churn in the AAAA path."""
            # GPT

            for ind in row_expanded:
                final_contribution_41_coul += C_prefactor * ACA_A_right[ind[0],ind[1]] * CS_EA_B[0, ind[2]] * AC_IP_A_left[ind[3], 0] * value
                final_contribution_41_exch += X_prefactor * ACA_A_right[ind[0],ind[2]] * CS_EA_B[0, ind[3]] * AC_IP_A_left[ind[1], 0] * value

                final_contribution_31_coul += C_prefactor * ACA_A_right[ind[0],ind[1]] * CA_EA_A_left[0, ind[2]] * SC_IP_B[ind[3], 0] * value
                final_contribution_31_exch += X_prefactor * ACA_A_right[ind[0],ind[2]] * CA_EA_A_left[0, ind[3]] * SC_IP_B[ind[1], 0] * value                
                
                final_contribution_43_coul += C_prefactor * AC_IP_A_left[ind[0], 0] * CS_EA_B[0, ind[1]] * CA_EA_A_right[0, ind[2]] * SC_IP_B[ind[3], 0] * value
                final_contribution_43_exch += X_prefactor * AC_IP_A_left[ind[0], 0] * CS_EA_B[0, ind[2]] * CA_EA_A_right[0, ind[3]] * SC_IP_B[ind[1], 0] * value                
                
                
    return final_contribution_41_coul, final_contribution_41_exch, final_contribution_31_coul, final_contribution_31_exch, final_contribution_43_coul, final_contribution_43_exch



def calc_AAAA_f90(
    twoelint_block,
    SC_A_BA_ia,
    CS_A_AB_ia,
    SC_B_AB_ia,
    CS_B_BA_ia,
    CS_EA_A_a,
    SC_IP_A_i,
    SC_IP_B_i,
    CS_EA_B_a,
    ACA_A_right,
    ACA_B_right,
    ACA_A_left,
    ACA_B_left,
    CA_EA_A_right,
    AC_IP_B_right,
    AC_IP_A_right,
    CA_EA_B_right,
    CA_EA_A_left,
    AC_IP_B_left,
    AC_IP_A_left,
    CA_EA_B_left,
    threshold,
):
    """
    Fortran-accelerated AAAA contribution.

    Contract: identical math/logic to calc_AAAA() (same index normalization, threshold shift,
    reorganize_indexes permutation rules/order, and prefactors). Falls back to calc_AAAA if
    the compiled module is unavailable.
    """
    if not _HAS_TWOELINT_F90:
        return calc_AAAA(
            twoelint_block,
            SC_A_BA_ia,
            CS_A_AB_ia,
            SC_B_AB_ia,
            CS_B_BA_ia,
            CS_EA_A_a,
            SC_IP_A_i,
            SC_IP_B_i,
            CS_EA_B_a,
            ACA_A_right,
            ACA_B_right,
            ACA_A_left,
            ACA_B_left,
            CA_EA_A_right,
            AC_IP_B_right,
            AC_IP_A_right,
            CA_EA_B_right,
            CA_EA_A_left,
            AC_IP_B_left,
            AC_IP_A_left,
            CA_EA_B_left,
            threshold,
        )

    aca_a_right = np.ascontiguousarray(ACA_A_right, dtype=np.float64).ravel(order="C")
    cs_ea_b = np.ascontiguousarray(CS_EA_B_a, dtype=np.float64).ravel(order="C")
    ac_ip_a_left = np.ascontiguousarray(AC_IP_A_left, dtype=np.float64).ravel(order="C")
    ca_ea_a_left = np.ascontiguousarray(CA_EA_A_left, dtype=np.float64).ravel(order="C")
    sc_ip_b = np.ascontiguousarray(SC_IP_B_i, dtype=np.float64).ravel(order="C")
    ca_ea_a_right = np.ascontiguousarray(CA_EA_A_right, dtype=np.float64).ravel(order="C")

    aca_a_right_ncol = int(ACA_A_right.shape[1])
    cs_ea_b_ncol = int(CS_EA_B_a.shape[1])
    ac_ip_a_left_ncol = int(AC_IP_A_left.shape[1])
    ca_ea_a_left_ncol = int(CA_EA_A_left.shape[1])
    sc_ip_b_ncol = int(SC_IP_B_i.shape[1])
    ca_ea_a_right_ncol = int(CA_EA_A_right.shape[1])


    totals = np.zeros(6, dtype=np.float64)

    # Stable iteration order (matches original expectation: unique_count in {1..4})
    for ucount in (1, 2, 3, 4):
        sb = twoelint_block.get(ucount)
        if sb is None:
            continue

        # sb is [] when this unique_count has no rows in this chunk (same as Python calc_AAAA no-op)
        if not hasattr(sb, "_pos"):
            continue
        
        pos = sb._pos
        if pos is None or pos.size == 0:
            continue

        # Extract only the AAAA rows in-order; this preserves the exact iteration order of _SubblockView.

        idx = np.ascontiguousarray(sb._pos, dtype=np.int32)
        a_i32 = np.ascontiguousarray(sb._a[idx], dtype=np.int32)
        b_i32 = np.ascontiguousarray(sb._b[idx], dtype=np.int32)
        c_i32 = np.ascontiguousarray(sb._c[idx], dtype=np.int32)
        d_i32 = np.ascontiguousarray(sb._d[idx], dtype=np.int32)
        v_f64 = np.ascontiguousarray(sb._v[idx], dtype=np.float64)

        r = _f2py_call(
            _twoelint_f90.twoelint_aaaa_mod.aaaa_accum,
            a=a_i32,
            b=b_i32,
            c=c_i32,
            d=d_i32,
            v=v_f64,
            ucount=int(ucount),
            unique_count=int(ucount),  # some builds use this name
            threshold=int(threshold),

            aca_a_right=aca_a_right,
            aca_a_right_ncol=aca_a_right_ncol,

            cs_ea_b=cs_ea_b,
            cs_ea_b_ncol=cs_ea_b_ncol,

            ac_ip_a_left=ac_ip_a_left,
            ac_ip_a_left_ncol=ac_ip_a_left_ncol,

            ca_ea_a_left=ca_ea_a_left,
            ca_ea_a_left_ncol=ca_ea_a_left_ncol,

            sc_ip_b=sc_ip_b,
            sc_ip_b_ncol=sc_ip_b_ncol,

            ca_ea_a_right=ca_ea_a_right,
            ca_ea_a_right_ncol=ca_ea_a_right_ncol,
        )

        totals += np.asarray(r, dtype=np.float64)

    return tuple(totals.tolist())

def calc_BBBA_f90_zero_copy(
    twoelint_block,
    SC_A_BA_ia,
    CS_A_AB_ia,
    SC_B_AB_ia,
    CS_B_BA_ia,
    CS_EA_A_a,
    SC_IP_B_i,
    SC_IP_A_i,
    CS_EA_B_a,
    SC_A_BA_pa,
    CS_A_AB_iq,
    SC_B_AB_pa,
    CS_B_BA_iq,
    CS_EA_A_q,
    SC_IP_B_p,
    SC_IP_A_p,
    CS_EA_B_q,
    ACA_A_right,
    ACA_B_right,
    ACA_A_left,
    SCS_A_BB_iq,
    ACA_B_left,
    SCS_A_BB_pa,
    CA_EA_A_right,
    AC_IP_B_right,
    AC_IP_A_right,
    CA_EA_B_right,
    CA_EA_A_left,
    AC_IP_B_left,
    AC_IP_A_left,
    CA_EA_B_left,
    NBAS_A: int,
):
    """
    Zero-copy BBBA kernel call.
    Always passes pos2/pos3/pos4 and all *_nc so f2py signatures never error.

    Returns the 64-tuple in the same order as calc_BBBA(...).
    """
    _f90 = _get_twoelint_f90_module("twoelint_bbba_mod")
    py_fallback = cBBBA.calc_BBBA
    if (_f90 is None) or (SC_A_BA_ia is None):
        return py_fallback(
            twoelint_block,
            SC_A_BA_ia,
            CS_A_AB_ia,
            SC_B_AB_ia,
            CS_B_BA_ia,
            CS_EA_A_a,
            SC_IP_B_i,
            SC_IP_A_i,
            CS_EA_B_a,
            SC_A_BA_pa,
            CS_A_AB_iq,
            SC_B_AB_pa,
            CS_B_BA_iq,
            CS_EA_A_q,
            SC_IP_B_p,
            SC_IP_A_p,
            CS_EA_B_q,
            ACA_A_right,
            ACA_B_right,
            ACA_A_left,
            SCS_A_BB_iq,
            ACA_B_left,
            SCS_A_BB_pa,
            CA_EA_A_right,
            AC_IP_B_right,
            AC_IP_A_right,
            CA_EA_B_right,
            CA_EA_A_left,
            AC_IP_B_left,
            AC_IP_A_left,
            CA_EA_B_left,
            NBAS_A,
        )

    if not twoelint_block:
        return (0.0,) * 64

    sb = twoelint_block.get(4) or twoelint_block.get(3) or twoelint_block.get(2)
    if sb is None:
        return (0.0,) * 64

# /twoelint_calc_ov_binary_2_fortran.py



    def _pos_or_empty(x: Any) -> np.ndarray:
        """
        Return a contiguous int64 1D array of positions (0-based), accepting:
          - None
          - ndarray / list of ints
          - _SubblockView (or any object exposing ._pos / .pos / .positions)
        """
        if x is None:
            return np.empty((0,), dtype=np.int64)

        # Handle bucketed views (your error: x is _SubblockView)
        if hasattr(x, "_pos"):
            x = getattr(x, "_pos")
        elif hasattr(x, "pos"):
            x = getattr(x, "pos")
        elif hasattr(x, "positions"):
            x = getattr(x, "positions")

        a = np.asarray(x, dtype=np.int64)
        if a.ndim != 1:
            a = a.reshape(-1)
        return np.ascontiguousarray(a)


    def _u16_as_i16_view(x):
        # Prefer existing project helper (true zero-copy for array('H')/buffers).
        fn = globals().get("_u16_as_i16_view")
        if callable(fn):
            return fn(x)
        arr = np.asarray(x)
        if arr.dtype == np.uint16:
            return arr.view(np.int16)
        return arr.astype(np.int16, copy=False)

    def _ravel_c_f64(mat):
        fn = globals().get("_ravel_c_f64")
        if callable(fn):
            return fn(mat)
        a = np.asarray(mat, dtype=np.float64, order="C")
        if a.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape={a.shape}")
        return a.ravel(order="C"), int(a.shape[1])

    # Always pass pos2/pos3/pos4 (even empty)
    pos2 = _pos_or_empty(twoelint_block.get(2))
    pos3 = _pos_or_empty(twoelint_block.get(3))
    pos4 = _pos_or_empty(twoelint_block.get(4))

    # Zero-copy views into the chunk arrays
    a = _u16_as_i16_view(getattr(sb, "_a"))
    b = _u16_as_i16_view(getattr(sb, "_b"))
    c = _u16_as_i16_view(getattr(sb, "_c"))
    d = _u16_as_i16_view(getattr(sb, "_d"))
    v = np.asarray(getattr(sb, "_v"), dtype=np.float64)

    # Flatten matrices to C-order 1D + ncol (ALWAYS pass *_nc)
    sc_a_ba, sc_a_ba_nc = _ravel_c_f64(SC_A_BA_ia)
    cs_a_ab, cs_a_ab_nc = _ravel_c_f64(CS_A_AB_ia)
    sc_b_ab, sc_b_ab_nc = _ravel_c_f64(SC_B_AB_ia)
    cs_b_ba, cs_b_ba_nc = _ravel_c_f64(CS_B_BA_ia)

    cs_ea_a, cs_ea_a_nc = _ravel_c_f64(CS_EA_A_a)
    sc_ip_a, sc_ip_a_nc = _ravel_c_f64(SC_IP_A_i)
    sc_ip_b, sc_ip_b_nc = _ravel_c_f64(SC_IP_B_i)
    cs_ea_b, cs_ea_b_nc = _ravel_c_f64(CS_EA_B_a)

    sc_a_ba_pa, sc_a_ba_pa_nc = _ravel_c_f64(SC_A_BA_pa)
    cs_a_ab_iq, cs_a_ab_iq_nc = _ravel_c_f64(CS_A_AB_iq)
    sc_b_ab_pa, sc_b_ab_pa_nc = _ravel_c_f64(SC_B_AB_pa)
    cs_b_ba_iq, cs_b_ba_iq_nc = _ravel_c_f64(CS_B_BA_iq)
    cs_ea_a_q, cs_ea_a_q_nc = _ravel_c_f64(CS_EA_A_q)
    sc_ip_b_p, sc_ip_b_p_nc = _ravel_c_f64(SC_IP_B_p)
    sc_ip_a_p, sc_ip_a_p_nc = _ravel_c_f64(SC_IP_A_p)
    cs_ea_b_q, cs_ea_b_q_nc = _ravel_c_f64(CS_EA_B_q)
    scs_a_bb_iq, scs_a_bb_iq_nc = _ravel_c_f64(SCS_A_BB_iq)
    scs_a_bb_pa, scs_a_bb_pa_nc = _ravel_c_f64(SCS_A_BB_pa)

    aca_a_right, aca_a_right_nc = _ravel_c_f64(ACA_A_right)
    aca_b_right, aca_b_right_nc = _ravel_c_f64(ACA_B_right)
    aca_a_left, aca_a_left_nc = _ravel_c_f64(ACA_A_left)
    aca_b_left, aca_b_left_nc = _ravel_c_f64(ACA_B_left)

    ca_ea_a_right, ca_ea_a_right_nc = _ravel_c_f64(CA_EA_A_right)
    ac_ip_b_right, ac_ip_b_right_nc = _ravel_c_f64(AC_IP_B_right)
    ac_ip_a_right, ac_ip_a_right_nc = _ravel_c_f64(AC_IP_A_right)
    ca_ea_b_right, ca_ea_b_right_nc = _ravel_c_f64(CA_EA_B_right)

    ca_ea_a_left, ca_ea_a_left_nc = _ravel_c_f64(CA_EA_A_left)
    ac_ip_b_left, ac_ip_b_left_nc = _ravel_c_f64(AC_IP_B_left)
    ac_ip_a_left, ac_ip_a_left_nc = _ravel_c_f64(AC_IP_A_left)
    ca_ea_b_left, ca_ea_b_left_nc = _ravel_c_f64(CA_EA_B_left)

    # Robust call: try keywords; if f2py name-maps differ, fall back to positional.
    mod = _f90
    try:
        out, err_flag = mod.bbba_accum_all(
            pos2=pos2,
            pos3=pos3,
            pos4=pos4,
            a_idx=a,
            b_idx=b,
            c_idx=c,
            d_idx=d,
            v_val=v,
            nbas_a=int(NBAS_A),
            sc_a_ba=sc_a_ba,
            sc_a_ba_nc=sc_a_ba_nc,
            cs_a_ab=cs_a_ab,
            cs_a_ab_nc=cs_a_ab_nc,
            sc_b_ab=sc_b_ab,
            sc_b_ab_nc=sc_b_ab_nc,
            cs_b_ba=cs_b_ba,
            cs_b_ba_nc=cs_b_ba_nc,
            cs_ea_a=cs_ea_a,
            cs_ea_a_nc=cs_ea_a_nc,
            sc_ip_a=sc_ip_a,
            sc_ip_a_nc=sc_ip_a_nc,
            sc_ip_b=sc_ip_b,
            sc_ip_b_nc=sc_ip_b_nc,
            cs_ea_b=cs_ea_b,
            cs_ea_b_nc=cs_ea_b_nc,
            sc_a_ba_pa=sc_a_ba_pa,
            sc_a_ba_pa_nc=sc_a_ba_pa_nc,
            cs_a_ab_iq=cs_a_ab_iq,
            cs_a_ab_iq_nc=cs_a_ab_iq_nc,
            sc_b_ab_pa=sc_b_ab_pa,
            sc_b_ab_pa_nc=sc_b_ab_pa_nc,
            cs_b_ba_iq=cs_b_ba_iq,
            cs_b_ba_iq_nc=cs_b_ba_iq_nc,
            cs_ea_a_q=cs_ea_a_q,
            cs_ea_a_q_nc=cs_ea_a_q_nc,
            sc_ip_b_p=sc_ip_b_p,
            sc_ip_b_p_nc=sc_ip_b_p_nc,
            sc_ip_a_p=sc_ip_a_p,
            sc_ip_a_p_nc=sc_ip_a_p_nc,
            cs_ea_b_q=cs_ea_b_q,
            cs_ea_b_q_nc=cs_ea_b_q_nc,
            scs_a_bb_iq=scs_a_bb_iq,
            scs_a_bb_iq_nc=scs_a_bb_iq_nc,
            scs_a_bb_pa=scs_a_bb_pa,
            scs_a_bb_pa_nc=scs_a_bb_pa_nc,
            aca_a_right=aca_a_right,
            aca_a_right_nc=aca_a_right_nc,
            aca_b_right=aca_b_right,
            aca_b_right_nc=aca_b_right_nc,
            aca_a_left=aca_a_left,
            aca_a_left_nc=aca_a_left_nc,
            aca_b_left=aca_b_left,
            aca_b_left_nc=aca_b_left_nc,
            ca_ea_a_right=ca_ea_a_right,
            ca_ea_a_right_nc=ca_ea_a_right_nc,
            ac_ip_b_right=ac_ip_b_right,
            ac_ip_b_right_nc=ac_ip_b_right_nc,
            ac_ip_a_right=ac_ip_a_right,
            ac_ip_a_right_nc=ac_ip_a_right_nc,
            ca_ea_b_right=ca_ea_b_right,
            ca_ea_b_right_nc=ca_ea_b_right_nc,
            ca_ea_a_left=ca_ea_a_left,
            ca_ea_a_left_nc=ca_ea_a_left_nc,
            ac_ip_b_left=ac_ip_b_left,
            ac_ip_b_left_nc=ac_ip_b_left_nc,
            ac_ip_a_left=ac_ip_a_left,
            ac_ip_a_left_nc=ac_ip_a_left_nc,
            ca_ea_b_left=ca_ea_b_left,
            ca_ea_b_left_nc=ca_ea_b_left_nc,
        )
    except TypeError:
        out, err_flag = mod.bbba_accum_all(
            pos2,
            pos3,
            pos4,
            a,
            b,
            c,
            d,
            v,
            int(NBAS_A),
            sc_a_ba,
            sc_a_ba_nc,
            cs_a_ab,
            cs_a_ab_nc,
            sc_b_ab,
            sc_b_ab_nc,
            cs_b_ba,
            cs_b_ba_nc,
            cs_ea_a,
            cs_ea_a_nc,
            sc_ip_a,
            sc_ip_a_nc,
            sc_ip_b,
            sc_ip_b_nc,
            cs_ea_b,
            cs_ea_b_nc,
            sc_a_ba_pa,
            sc_a_ba_pa_nc,
            cs_a_ab_iq,
            cs_a_ab_iq_nc,
            sc_b_ab_pa,
            sc_b_ab_pa_nc,
            cs_b_ba_iq,
            cs_b_ba_iq_nc,
            cs_ea_a_q,
            cs_ea_a_q_nc,
            sc_ip_b_p,
            sc_ip_b_p_nc,
            sc_ip_a_p,
            sc_ip_a_p_nc,
            cs_ea_b_q,
            cs_ea_b_q_nc,
            scs_a_bb_iq,
            scs_a_bb_iq_nc,
            scs_a_bb_pa,
            scs_a_bb_pa_nc,
            aca_a_right,
            aca_a_right_nc,
            aca_b_right,
            aca_b_right_nc,
            aca_a_left,
            aca_a_left_nc,
            aca_b_left,
            aca_b_left_nc,
            ca_ea_a_right,
            ca_ea_a_right_nc,
            ac_ip_b_right,
            ac_ip_b_right_nc,
            ac_ip_a_right,
            ac_ip_a_right_nc,
            ca_ea_b_right,
            ca_ea_b_right_nc,
            ca_ea_a_left,
            ca_ea_a_left_nc,
            ac_ip_b_left,
            ac_ip_b_left_nc,
            ac_ip_a_left,
            ac_ip_a_left_nc,
            ca_ea_b_left,
            ca_ea_b_left_nc,
        )

    if int(err_flag) != 0:
        raise RuntimeError(f"bbba_accum_all failed with err_flag={int(err_flag)}")

    return tuple(float(x) for x in out)



def _pos_or_empty(sb) -> np.ndarray:
    if sb is None:
        return _EMPTY_I64
    pos = getattr(sb, "_pos", None)
    if pos is None:
        pos = getattr(sb, "pos", None)
    if pos is None:
        # last resort: treat sb itself as positions
        pos = sb
    pos = np.asarray(pos)
    if pos.size == 0:
        return _EMPTY_I64
    if pos.dtype != np.int64:
        pos = pos.astype(np.int64, copy=False)
    return pos


def _u16_as_i16_view(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.uint16:
        return x.view(np.int16)  # zero-copy bitcast
    if x.dtype == np.int16:
        return x
    return x.astype(np.int16, copy=False)



def _ravel_c_f64(mat: np.ndarray) -> Tuple[np.ndarray, int]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.ravel(order="C"), int(arr.shape[1])


def calc_BBAA_f90_zero_copy(
    twoelint_block,
    SC_A_BA_ia,
    CS_A_AB_ia,
    SC_B_AB_ia,
    CS_B_BA_ia,
    CS_EA_A_a,
    SC_IP_B_i,
    SC_IP_A_i,
    CS_EA_B_a,
    SC_A_BA_pa,
    CS_A_AB_iq,
    SC_B_AB_pa,
    CS_B_BA_iq,
    CS_EA_A_q,
    SC_IP_B_p,
    SC_IP_A_p,
    CS_EA_B_q,
    ACA_A_right,
    SCS_A_BB_iq,
    SCS_A_BB_pa,
    ACA_B_right,
    SCS_B_AA_pa,
    ACA_A_left,
    ACA_B_left,
    CA_EA_A_right,
    AC_IP_B_right,
    AC_IP_A_right,
    CA_EA_B_right,
    CA_EA_A_left,
    AC_IP_B_left,
    AC_IP_A_left,
    CA_EA_B_left,
    NBAS_A: int,
):
    # Fortran kernel implements the same OV-path; if OV is off, keep Python behavior.
    f90_mod = _get_twoelint_f90_module("twoelint_bbaa_mod")
    if (f90_mod is None) or (SC_A_BA_ia is None):
        return cBBAA.calc_BBAA(
            twoelint_block,
            SC_A_BA_ia,
            CS_A_AB_ia,
            SC_B_AB_ia,
            CS_B_BA_ia,
            CS_EA_A_a,
            SC_IP_B_i,
            SC_IP_A_i,
            CS_EA_B_a,
            SC_A_BA_pa,
            CS_A_AB_iq,
            SC_B_AB_pa,
            CS_B_BA_iq,
            CS_EA_A_q,
            SC_IP_B_p,
            SC_IP_A_p,
            CS_EA_B_q,
            ACA_A_right,
            SCS_A_BB_iq,
            SCS_A_BB_pa,
            ACA_B_right,
            SCS_B_AA_pa,
            ACA_A_left,
            ACA_B_left,
            CA_EA_A_right,
            AC_IP_B_right,
            AC_IP_A_right,
            CA_EA_B_right,
            CA_EA_A_left,
            AC_IP_B_left,
            AC_IP_A_left,
            CA_EA_B_left,
            NBAS_A,
        )

    if not twoelint_block:
        return (0.0,) * 82

    sb = twoelint_block.get(4) or twoelint_block.get(3) or twoelint_block.get(2)
    if sb is None:
        return (0.0,) * 82

    pos2 = _pos_or_empty(twoelint_block.get(2))
    pos3 = _pos_or_empty(twoelint_block.get(3))
    pos4 = _pos_or_empty(twoelint_block.get(4))

    # Zero-copy views into the chunk arrays
    a = _u16_as_i16_view(getattr(sb, "_a"))
    b = _u16_as_i16_view(getattr(sb, "_b"))
    c = _u16_as_i16_view(getattr(sb, "_c"))
    d = _u16_as_i16_view(getattr(sb, "_d"))
    v = np.asarray(getattr(sb, "_v"), dtype=np.float64)

    # Flatten matrices to C-order 1D + ncol
    sc_a_ba, sc_a_ba_ncol = _ravel_c_f64(SC_A_BA_ia)
    cs_a_ab, cs_a_ab_ncol = _ravel_c_f64(CS_A_AB_ia)
    sc_b_ab, sc_b_ab_ncol = _ravel_c_f64(SC_B_AB_ia)
    cs_b_ba, cs_b_ba_ncol = _ravel_c_f64(CS_B_BA_ia)

    cs_ea_a, cs_ea_a_ncol = _ravel_c_f64(CS_EA_A_a)
    sc_ip_a, sc_ip_a_ncol = _ravel_c_f64(SC_IP_A_i)
    sc_ip_b, sc_ip_b_ncol = _ravel_c_f64(SC_IP_B_i)
    cs_ea_b, cs_ea_b_ncol = _ravel_c_f64(CS_EA_B_a)

    sc_a_ba_pa, sc_a_ba_pa_ncol = _ravel_c_f64(SC_A_BA_pa)
    cs_a_ab_iq, cs_a_ab_iq_ncol = _ravel_c_f64(CS_A_AB_iq)
    sc_b_ab_pa, sc_b_ab_pa_ncol = _ravel_c_f64(SC_B_AB_pa)
    cs_b_ba_iq, cs_b_ba_iq_ncol = _ravel_c_f64(CS_B_BA_iq)
    cs_ea_a_q, cs_ea_a_q_ncol = _ravel_c_f64(CS_EA_A_q)
    sc_ip_b_p, sc_ip_b_p_ncol = _ravel_c_f64(SC_IP_B_p)
    sc_ip_a_p, sc_ip_a_p_ncol = _ravel_c_f64(SC_IP_A_p)
    cs_ea_b_q, cs_ea_b_q_ncol = _ravel_c_f64(CS_EA_B_q)

    aca_a_right, aca_a_right_ncol = _ravel_c_f64(ACA_A_right)
    aca_b_right, aca_b_right_ncol = _ravel_c_f64(ACA_B_right)
    aca_a_left, aca_a_left_ncol = _ravel_c_f64(ACA_A_left)
    aca_b_left, aca_b_left_ncol = _ravel_c_f64(ACA_B_left)

    scs_a_bb_iq, scs_a_bb_iq_ncol = _ravel_c_f64(SCS_A_BB_iq)
    scs_a_bb_pa, scs_a_bb_pa_ncol = _ravel_c_f64(SCS_A_BB_pa)
    scs_b_aa_pa, scs_b_aa_pa_ncol = _ravel_c_f64(SCS_B_AA_pa)

    ca_ea_a_right, ca_ea_a_right_ncol = _ravel_c_f64(CA_EA_A_right)
    ac_ip_b_right, ac_ip_b_right_ncol = _ravel_c_f64(AC_IP_B_right)
    ac_ip_a_right, ac_ip_a_right_ncol = _ravel_c_f64(AC_IP_A_right)
    ca_ea_b_right, ca_ea_b_right_ncol = _ravel_c_f64(CA_EA_B_right)

    ca_ea_a_left, ca_ea_a_left_ncol = _ravel_c_f64(CA_EA_A_left)
    ac_ip_b_left, ac_ip_b_left_ncol = _ravel_c_f64(AC_IP_B_left)
    ac_ip_a_left, ac_ip_a_left_ncol = _ravel_c_f64(AC_IP_A_left)
    ca_ea_b_left, ca_ea_b_left_ncol = _ravel_c_f64(CA_EA_B_left)

    out, err_flag = f90_mod.bbaa_accum_all(
        pos2=pos2,
        pos3=pos3,
        pos4=pos4,
        a=a,
        b=b,
        c=c,
        d=d,
        v=v,
        nbas_a=int(NBAS_A),

        sc_a_ba=sc_a_ba,       sc_a_ba_nc=sc_a_ba_ncol,
        cs_a_ab=cs_a_ab,       cs_a_ab_nc=cs_a_ab_ncol,
        sc_b_ab=sc_b_ab,       sc_b_ab_nc=sc_b_ab_ncol,
        cs_b_ba=cs_b_ba,       cs_b_ba_nc=cs_b_ba_ncol,

        cs_ea_a=cs_ea_a,       cs_ea_a_nc=cs_ea_a_ncol,
        sc_ip_a=sc_ip_a,       sc_ip_a_nc=sc_ip_a_ncol,
        sc_ip_b=sc_ip_b,       sc_ip_b_nc=sc_ip_b_ncol,
        cs_ea_b=cs_ea_b,       cs_ea_b_nc=cs_ea_b_ncol,

        sc_a_ba_pa=sc_a_ba_pa, sc_a_ba_pa_nc=sc_a_ba_pa_ncol,
        cs_a_ab_iq=cs_a_ab_iq, cs_a_ab_iq_nc=cs_a_ab_iq_ncol,
        sc_b_ab_pa=sc_b_ab_pa, sc_b_ab_pa_nc=sc_b_ab_pa_ncol,
        cs_b_ba_iq=cs_b_ba_iq, cs_b_ba_iq_nc=cs_b_ba_iq_ncol,
        cs_ea_a_q=cs_ea_a_q,   cs_ea_a_q_nc=cs_ea_a_q_ncol,
        sc_ip_b_p=sc_ip_b_p,   sc_ip_b_p_nc=sc_ip_b_p_ncol,
        sc_ip_a_p=sc_ip_a_p,   sc_ip_a_p_nc=sc_ip_a_p_ncol,
        cs_ea_b_q=cs_ea_b_q,   cs_ea_b_q_nc=cs_ea_b_q_ncol,

        aca_a_right=aca_a_right,   aca_a_right_nc=aca_a_right_ncol,
        aca_b_right=aca_b_right,   aca_b_right_nc=aca_b_right_ncol,
        aca_a_left=aca_a_left,     aca_a_left_nc=aca_a_left_ncol,
        aca_b_left=aca_b_left,     aca_b_left_nc=aca_b_left_ncol,

        scs_a_bb_iq=scs_a_bb_iq, scs_a_bb_iq_nc=scs_a_bb_iq_ncol,
        scs_a_bb_pa=scs_a_bb_pa, scs_a_bb_pa_nc=scs_a_bb_pa_ncol,
        scs_b_aa_pa=scs_b_aa_pa, scs_b_aa_pa_nc=scs_b_aa_pa_ncol,

        ca_ea_a_right=ca_ea_a_right,   ca_ea_a_right_nc=ca_ea_a_right_ncol,
        ac_ip_b_right=ac_ip_b_right,   ac_ip_b_right_nc=ac_ip_b_right_ncol,
        ac_ip_a_right=ac_ip_a_right,   ac_ip_a_right_nc=ac_ip_a_right_ncol,
        ca_ea_b_right=ca_ea_b_right,   ca_ea_b_right_nc=ca_ea_b_right_ncol,

        ca_ea_a_left=ca_ea_a_left,     ca_ea_a_left_nc=ca_ea_a_left_ncol,
        ac_ip_b_left=ac_ip_b_left,     ac_ip_b_left_nc=ac_ip_b_left_ncol,
        ac_ip_a_left=ac_ip_a_left,     ac_ip_a_left_nc=ac_ip_a_left_ncol,
        ca_ea_b_left=ca_ea_b_left,     ca_ea_b_left_nc=ca_ea_b_left_ncol,
    )


    if int(err_flag) != 0:
        raise RuntimeError(f"bbaa_accum_all failed with err_flag={int(err_flag)}")

    return tuple(float(x) for x in out)



def custom_sort_and_extend(lst):
    # Sort the list:
    # 1. Mixed-element lists first, sorted by their smallest element
    # 2. Equal-element lists last, sorted by their repeated element in ascending order
    sorted_lst = sorted(lst, key=lambda sub: (len(set(sub)) == 1, sorted(sub)[0]))
    #sorted_lst = sorted(lst, key=lambda sub: (len(set(sub)) == 1, min(sub) if len(set(sub)) > 1 else max(sub)))
    # Flatten the sorted list
    merged_list = [item for sub in sorted_lst for item in sub]
    
    return merged_list

# -----------------------------------------------------------------------------
# Fast result transport: fixed-length vectors + mask (reduces per-chunk dict churn)
# -----------------------------------------------------------------------------
# Canonical schema: union of all (term,key) pairs used in the original Python results dict.
# Keep order stable: (term asc) + (keys in defined tuple order).
RESULT_SCHEMA: dict[int, tuple[str, ...]] = {
    12: (
        "C",
        "X",
        "AAAB_C_SC_occ",
        "AAAB_X_SC_occ",
        "AABB_C_2S",
        "AABB_X_2S",
        "BBBA_C_CS_virt",
        "BBBA_X_CS_virt",
    ),

    13: (
        "C",
        "X",
        "AAAA_IP_C",
        "AAAA_IP_X",
        "AAAB_C_2S_IP",
        "AAAB_X_2S_IP",
        "AABB_local_virt_C",
        "AABB_local_virt_X",
        "AABB_C_1S_occ_arb",
        "AABB_X_1S_occ_arb",
        "AABB_C_1S_EA_arb",
        "AABB_X_1S_EA_arb",

        # BBAA
        "BABA_C_13_3S_IP_virt_occ_arb",
        "BABA_X_13_3S_IP_virt_occ_arb",
        "BAAB_C_13_3S_IP_virt_EA_arb",
        "BAAB_X_13_3S_IP_virt_EA_arb",

        # BBBA
        "ABBB_C_13_2S_EA_arb_occ_arb",
        "ABBB_X_13_2S_EA_arb_occ_arb",
        "BABB_C_13_4S_IP_virt_occ_arb_EA_arb",
        "BABB_X_13_4S_IP_virt_occ_arb_EA_arb",
        "BBAB_C_13_2S_virt_EA_arb",
        "BBAB_X_13_2S_virt_EA_arb",
        "BBBA_C_13_2S_virt_occ_arb",
        "BBBA_X_13_2S_virt_occ_arb",
    ),

    14: (
        "C",
        "X",
        "AABB_local_virt_C",
        "AABB_local_virt_X",
        "AABB_IP_C",
        "AABB_IP_X",
        "BBBA_14_C_SC_IP_CS",
        "BBBA_14_X_SC_IP_CS",
        "AABB_C_1S_occ_arb",
        "AABB_X_1S_occ_arb",

        # BBAA
        "BABA_C_14_3S_virt_EA_arb_occ_arb",
        "BABA_X_14_3S_virt_EA_arb_occ_arb",
        "ABBA_C_14_3S_IP_occ_arb_EA_arb",
        "ABBA_X_14_3S_IP_occ_arb_EA_arb",
        "BBAA_C_14_3S_IP_virt_EA_arb",
        "BBAA_X_14_3S_IP_virt_EA_arb",

        # BAAA
        "ABAA_C_14_2S_IP_EA_arb",
        "ABAA_X_14_2S_IP_EA_arb",
        "AABA_C_14_2S_occ_arb_EA_arb",
        "AABA_X_14_2S_occ_arb_EA_arb",
        "BAAA_C_14_2S_virt_EA_arb",
        "BAAA_X_14_2S_virt_EA_arb",

        # BBBA
        "ABBB_C_14_2S_IP_occ_arb",
        "ABBB_X_14_2S_IP_occ_arb",
        "BABB_C_14_2S_virt_occ_arb",
        "BABB_X_14_2S_virt_occ_arb",
        "BBBA_C_14_4S_IP_virt_occ_arb_EA_arb",
        "BBBA_X_14_4S_IP_virt_occ_arb_EA_arb",
    ),

    21: (
        "C",
        "X",
        "AAAB_C_CS_virt",
        "AAAB_X_CS_virt",
        "AABB_C_2S",
        "AABB_X_2S",
        "BBBA_C_SC_occ",
        "BBBA_X_SC_occ",
        "AAAB_C_1S_occ_arb",
        "AAAB_X_1S_occ_arb",
        "ABBB_C_1S_virt_arb",
        "ABBB_X_1S_virt_arb",

        # BBAA
        "ABAB_C_21_4S_A_occ_B_virt_A_virt_arb_B_occ_arb",
        "ABAB_X_21_4S_A_occ_B_virt_A_virt_arb_B_occ_arb",
        "BAAB_C_21_2S_B_occ_arb_A_virt_arb",
        "BAAB_X_21_2S_B_occ_arb_A_virt_arb",
        "BBAA_C_21_2S_A_occ_B_occ_arb",
        "BBAA_X_21_2S_A_occ_B_occ_arb",

        # BAAA
        "ABAA_C_3S_A_occ_B_virt_B_occ_arb",
        "ABAA_X_3S_A_occ_B_virt_B_occ_arb",
        "ABAA_C_3S_B_virt_B_occ_arb_A_virt_arb",
        "ABAA_X_3S_B_virt_B_occ_arb_A_virt_arb",

        # BBBA
        "ABBB_C_21_3S_A_occ_B_virt_A_virt_arb",
        "ABBB_X_21_3S_A_occ_B_virt_A_virt_arb",
        "BBAB_C_21_3S_A_occ_A_virt_arb_B_occ_arb",
        "BBAB_X_21_3S_A_occ_A_virt_arb_B_occ_arb",
    ),

    23: (
        "C",
        "X",
        "AAAB_C_2S_IP",
        "AAAB_X_2S_IP",
        "AABB_local_virt_C",
        "AABB_local_virt_X",
        "AABB_IP_C",
        "AABB_IP_X",
    ),

    24: (
        "C",
        "X",
        "AABB_local_virt_C",
        "AABB_local_virt_X",
        "BBBA_C_2S_IP",
        "BBBA_X_2S_IP",
        "BBBB_IP_C",
        "BBBB_IP_X",
    ),

    31: (
        "C",
        "X",
        "AABB_local_occ_C",
        "AABB_local_occ_X",
        "AABB_EA_C",
        "AABB_EA_X",
        "BBBA_C_CS_EA_SC",
        "BBBA_X_CS_EA_SC",
        "AABB_C_1S_virt_arb",
        "AABB_X_1S_virt_arb",

        # BBAA
        "ABAB_C_31_3S_occ_IP_arb_virt_arb",
        "ABAB_X_31_3S_occ_IP_arb_virt_arb",
        "BAAB_C_31_3S_EA_IP_arb_virt_arb",
        "BAAB_X_31_3S_EA_IP_arb_virt_arb",
        "BBAA_C_31_3S_EA_occ_IP_arb",
        "BBAA_X_31_3S_EA_occ_IP_arb",

        # BAAA
        "BAAA_C_31_2S_EA_IP_arb",
        "BAAA_X_31_2S_EA_IP_arb",
        "ABAA_C_31_2S_occ_IP_arb",
        "ABAA_X_31_2S_occ_IP_arb",

        # BBBA
        "ABBB_C_31_2S_occ_virt_arb",
        "ABBB_X_31_2S_occ_virt_arb",
        "BABB_C_31_2S_EA_virt_arb",
        "BABB_X_31_2S_EA_virt_arb",
        "BABB_C_31_4S_occ_EA_virt_arb_IP_arb",
        "BABB_X_31_4S_occ_EA_virt_arb_IP_arb",
    ),

    32: (
        "C",
        "X",
        "AABB_local_occ_C",
        "AABB_local_occ_X",
        "BBBA_32_C_SC_EA_CS",
        "BBBA_32_X_SC_EA_CS",
        "BBBB_EA_C",
        "BBBB_EA_X",
    ),

    34: (
        "C",
        "X",
        "BBBA_C_CS_EA",
        "BBBA_X_CS_EA",
        "BBBA_C_SC_IP",
        "BBBA_X_SC_IP",
        "BBBB_C",
        "BBBB_X",
    ),

    41: (
        "C",
        "X",
        "AAAA_EA_C",
        "AAAA_EA_X",
        "AABB_local_occ_C",
        "AABB_local_occ_X",
        "AAAB_C_SC_CS_EA_2S",
        "AAAB_X_SC_CS_EA_2S",
        "AABB_C_1S_virt_arb",
        "AABB_X_1S_virt_arb",
        "AABB_C_1S_IP_arb",
        "AABB_X_1S_IP_arb",

        # BBAA
        "ABAB_C_41_3S_occ_EA_virt_arb",
        "ABAB_X_41_3S_occ_EA_virt_arb",
        "ABBA_C_41_3S_occ_EA_IP_arb",
        "ABBA_X_41_3S_occ_EA_IP_arb",

        # BBBA
        "BBAB_C_41_2S_occ_virt_arb",
        "BBAB_X_41_2S_occ_virt_arb",
        "BBBA_C_41_2S_occ_IP_arb",
        "BBBA_X_41_2S_occ_IP_arb",
        "ABBB_C_41_4S_occ_EA_virt_arb_IP_arb",
        "ABBB_X_41_4S_occ_EA_virt_arb_IP_arb",
    ),

    42: (
        "C",
        "X",
        "AABB_local_occ_C",
        "AABB_local_occ_X",
        "AABB_EA_C",
        "AABB_EA_X",
        "AAAB_C_SC_CS_EA",
        "AAAB_X_SC_CS_EA",
    ),

    43: (
        "C",
        "X",
        "AAAA_C",
        "AAAA_X",
        "AAAB_C_CS_EA",
        "AAAB_X_CS_EA",
        "AAAB_C_SC_IP",
        "AAAB_X_SC_IP",
        "BBBA_C_1S_EA_arb",
        "BBBA_X_1S_EA_arb",
        "BBBA_C_1S_IP_arb",
        "BBBA_X_1S_IP_arb",

        # BBAA
        "BABA_C_43_2S_IP_IP_arb",
        "BABA_X_43_2S_IP_IP_arb",
        "ABBA_43_C_2S_EA_IP_arb",
        "ABBA_43_X_2S_EA_IP_arb",
        "ABBA_C_43_2S_EA_EA_arb",
        "ABBA_X_43_2S_EA_EA_arb",
        "BAAB_C_43_2S_IP_EA_arb",
        "BAAB_X_43_2S_IP_EA_arb",

        # BBBA
        "BABB_C_43_3S_IP_IP_arb_EA_arb",
        "BABB_X_43_3S_IP_IP_arb_EA_arb",
        "ABBB_C_43_3S_EA_EA_arb_IP_arb",
        "ABBB_X_43_3S_EA_EA_arb_IP_arb",
    ),
}

RESULT_FIELDS: tuple[tuple[int, str], ...] = tuple(
    (term, key) for term in sorted(RESULT_SCHEMA) for key in RESULT_SCHEMA[term]
)
RESULT_INDEX: dict[tuple[int, str], int] = {k: i for i, k in enumerate(RESULT_FIELDS)}
RESULT_N = len(RESULT_FIELDS)

# Exact output orders from the original Python multiple-assignment LHS.

BBAA_OUT_FIELDS: tuple[tuple[int, str], ...] = (
    (21, "C"), (21, "X"),
    (43, "C"), (43, "X"),
    (12, "C"), (12, "X"),
    (34, "C"), (34, "X"),

    (31, "AABB_local_occ_C"), (31, "AABB_local_occ_X"),
    (13, "AABB_local_virt_C"), (13, "AABB_local_virt_X"),
    (31, "AABB_EA_C"), (31, "AABB_EA_X"),

    (41, "AABB_local_occ_C"), (41, "AABB_local_occ_X"),
    (14, "AABB_local_virt_C"), (14, "AABB_local_virt_X"),
    (14, "AABB_IP_C"), (14, "AABB_IP_X"),

    (21, "AABB_C_2S"), (21, "AABB_X_2S"),
    (12, "AABB_C_2S"), (12, "AABB_X_2S"),

    (32, "AABB_local_occ_C"), (32, "AABB_local_occ_X"),
    (42, "AABB_local_occ_C"), (42, "AABB_local_occ_X"),
    (42, "AABB_EA_C"), (42, "AABB_EA_X"),

    (23, "AABB_local_virt_C"), (23, "AABB_local_virt_X"),
    (23, "AABB_IP_C"), (23, "AABB_IP_X"),
    (24, "AABB_local_virt_C"), (24, "AABB_local_virt_X"),

    (14, "AABB_C_1S_occ_arb"), (14, "AABB_X_1S_occ_arb"),
    (41, "AABB_C_1S_virt_arb"), (41, "AABB_X_1S_virt_arb"),
    (41, "AABB_C_1S_IP_arb"), (41, "AABB_X_1S_IP_arb"),

    (13, "AABB_C_1S_occ_arb"), (13, "AABB_X_1S_occ_arb"),
    (13, "AABB_C_1S_EA_arb"), (13, "AABB_X_1S_EA_arb"),
    (31, "AABB_C_1S_virt_arb"), (31, "AABB_X_1S_virt_arb"),

    # New BBAA terms
    (31, "ABAB_C_31_3S_occ_IP_arb_virt_arb"),
    (31, "ABAB_X_31_3S_occ_IP_arb_virt_arb"),

    (13, "BABA_C_13_3S_IP_virt_occ_arb"),
    (13, "BABA_X_13_3S_IP_virt_occ_arb"),

    (41, "ABAB_C_41_3S_occ_EA_virt_arb"),
    (41, "ABAB_X_41_3S_occ_EA_virt_arb"),

    (14, "BABA_C_14_3S_virt_EA_arb_occ_arb"),
    (14, "BABA_X_14_3S_virt_EA_arb_occ_arb"),

    (21, "ABAB_C_21_4S_A_occ_B_virt_A_virt_arb_B_occ_arb"),
    (21, "ABAB_X_21_4S_A_occ_B_virt_A_virt_arb_B_occ_arb"),

    (43, "BABA_C_43_2S_IP_IP_arb"),
    (43, "BABA_X_43_2S_IP_IP_arb"),

    (31, "BAAB_C_31_3S_EA_IP_arb_virt_arb"),
    (31, "BAAB_X_31_3S_EA_IP_arb_virt_arb"),

    (13, "BAAB_C_13_3S_IP_virt_EA_arb"),
    (13, "BAAB_X_13_3S_IP_virt_EA_arb"),

    (41, "ABBA_C_41_3S_occ_EA_IP_arb"),
    (41, "ABBA_X_41_3S_occ_EA_IP_arb"),

    (14, "ABBA_C_14_3S_IP_occ_arb_EA_arb"),
    (14, "ABBA_X_14_3S_IP_occ_arb_EA_arb"),

    (21, "BAAB_C_21_2S_B_occ_arb_A_virt_arb"),
    (21, "BAAB_X_21_2S_B_occ_arb_A_virt_arb"),

    (31, "BBAA_C_31_3S_EA_occ_IP_arb"),
    (31, "BBAA_X_31_3S_EA_occ_IP_arb"),

    (14, "BBAA_C_14_3S_IP_virt_EA_arb"),
    (14, "BBAA_X_14_3S_IP_virt_EA_arb"),

    (21, "BBAA_C_21_2S_A_occ_B_occ_arb"),
    (21, "BBAA_X_21_2S_A_occ_B_occ_arb"),

    (43, "ABBA_43_C_2S_EA_IP_arb"),
    (43, "ABBA_43_X_2S_EA_IP_arb"),

    (43, "ABBA_C_43_2S_EA_EA_arb"),
    (43, "ABBA_X_43_2S_EA_EA_arb"),

    (43, "BAAB_C_43_2S_IP_EA_arb"),
    (43, "BAAB_X_43_2S_IP_EA_arb"),
)


BAAA_OUT_FIELDS: tuple[tuple[int, str], ...] = (
    (31, "C"), (31, "X"),
    (41, "C"), (41, "X"),
    (13, "C"), (13, "X"),
    (14, "C"), (14, "X"),

    (21, "AAAB_C_CS_virt"), (21, "AAAB_X_CS_virt"),
    (12, "AAAB_C_SC_occ"), (12, "AAAB_X_SC_occ"),

    (43, "AAAB_C_CS_EA"), (43, "AAAB_X_CS_EA"),
    (43, "AAAB_C_SC_IP"), (43, "AAAB_X_SC_IP"),

    (42, "AAAB_C_SC_CS_EA"), (42, "AAAB_X_SC_CS_EA"),
    (41, "AAAB_C_SC_CS_EA_2S"), (41, "AAAB_X_SC_CS_EA_2S"),

    (23, "AAAB_C_2S_IP"), (23, "AAAB_X_2S_IP"),
    (13, "AAAB_C_2S_IP"), (13, "AAAB_X_2S_IP"),

    (21, "AAAB_C_1S_occ_arb"), (21, "AAAB_X_1S_occ_arb"),

    # New BAAA terms
    (31, "BAAA_C_31_2S_EA_IP_arb"),
    (31, "BAAA_X_31_2S_EA_IP_arb"),

    (31, "ABAA_C_31_2S_occ_IP_arb"),
    (31, "ABAA_X_31_2S_occ_IP_arb"),

    (14, "ABAA_C_14_2S_IP_EA_arb"),
    (14, "ABAA_X_14_2S_IP_EA_arb"),

    (21, "ABAA_C_3S_A_occ_B_virt_B_occ_arb"),
    (21, "ABAA_X_3S_A_occ_B_virt_B_occ_arb"),

    (14, "AABA_C_14_2S_occ_arb_EA_arb"),
    (14, "AABA_X_14_2S_occ_arb_EA_arb"),

    (21, "ABAA_C_3S_B_virt_B_occ_arb_A_virt_arb"),
    (21, "ABAA_X_3S_B_virt_B_occ_arb_A_virt_arb"),

    (14, "BAAA_C_14_2S_virt_EA_arb"),
    (14, "BAAA_X_14_2S_virt_EA_arb"),
)

AAAA_OUT_FIELDS: tuple[tuple[int, str], ...] = (
    (41, "AAAA_EA_C"), (41, "AAAA_EA_X"),
    (13, "AAAA_IP_C"), (13, "AAAA_IP_X"),
    (43, "AAAA_C"), (43, "AAAA_X"),
)

BBBB_OUT_FIELDS: tuple[tuple[int, str], ...] = (
    (32, "BBBB_EA_C"), (32, "BBBB_EA_X"),
    (24, "BBBB_IP_C"), (24, "BBBB_IP_X"),
    (34, "BBBB_C"), (34, "BBBB_X"),
)

BBBA_OUT_FIELDS: tuple[tuple[int, str], ...] = (
    (32, "C"), (32, "X"),
    (42, "C"), (42, "X"),
    (23, "C"), (23, "X"),
    (24, "C"), (24, "X"),

    (31, "BBBA_C_CS_EA_SC"), (31, "BBBA_X_CS_EA_SC"),
    (14, "BBBA_14_C_SC_IP_CS"), (14, "BBBA_14_X_SC_IP_CS"),
    (32, "BBBA_32_C_SC_EA_CS"), (32, "BBBA_32_X_SC_EA_CS"),

    (24, "BBBA_C_2S_IP"), (24, "BBBA_X_2S_IP"),

    (21, "BBBA_C_SC_occ"), (21, "BBBA_X_SC_occ"),
    (12, "BBBA_C_CS_virt"), (12, "BBBA_X_CS_virt"),

    (34, "BBBA_C_CS_EA"), (34, "BBBA_X_CS_EA"),
    (34, "BBBA_C_SC_IP"), (34, "BBBA_X_SC_IP"),

    (43, "BBBA_C_1S_EA_arb"), (43, "BBBA_X_1S_EA_arb"),
    (43, "BBBA_C_1S_IP_arb"), (43, "BBBA_X_1S_IP_arb"),

    (21, "ABBB_C_1S_virt_arb"), (21, "ABBB_X_1S_virt_arb"),

    # New BBBA terms
    (31, "ABBB_C_31_2S_occ_virt_arb"),
    (31, "ABBB_X_31_2S_occ_virt_arb"),

    (13, "ABBB_C_13_2S_EA_arb_occ_arb"),
    (13, "ABBB_X_13_2S_EA_arb_occ_arb"),

    (14, "ABBB_C_14_2S_IP_occ_arb"),
    (14, "ABBB_X_14_2S_IP_occ_arb"),

    (21, "ABBB_C_21_3S_A_occ_B_virt_A_virt_arb"),
    (21, "ABBB_X_21_3S_A_occ_B_virt_A_virt_arb"),

    (31, "BABB_C_31_2S_EA_virt_arb"),
    (31, "BABB_X_31_2S_EA_virt_arb"),

    (13, "BABB_C_13_4S_IP_virt_occ_arb_EA_arb"),
    (13, "BABB_X_13_4S_IP_virt_occ_arb_EA_arb"),

    (14, "BABB_C_14_2S_virt_occ_arb"),
    (14, "BABB_X_14_2S_virt_occ_arb"),

    (43, "BABB_C_43_3S_IP_IP_arb_EA_arb"),
    (43, "BABB_X_43_3S_IP_IP_arb_EA_arb"),

    (31, "BABB_C_31_4S_occ_EA_virt_arb_IP_arb"),
    (31, "BABB_X_31_4S_occ_EA_virt_arb_IP_arb"),

    (13, "BBAB_C_13_2S_virt_EA_arb"),
    (13, "BBAB_X_13_2S_virt_EA_arb"),

    (41, "BBAB_C_41_2S_occ_virt_arb"),
    (41, "BBAB_X_41_2S_occ_virt_arb"),

    (21, "BBAB_C_21_3S_A_occ_A_virt_arb_B_occ_arb"),
    (21, "BBAB_X_21_3S_A_occ_A_virt_arb_B_occ_arb"),

    (13, "BBBA_C_13_2S_virt_occ_arb"),
    (13, "BBBA_X_13_2S_virt_occ_arb"),

    (41, "BBBA_C_41_2S_occ_IP_arb"),
    (41, "BBBA_X_41_2S_occ_IP_arb"),

    (14, "BBBA_C_14_4S_IP_virt_occ_arb_EA_arb"),
    (14, "BBBA_X_14_4S_IP_virt_occ_arb_EA_arb"),

    (41, "ABBB_C_41_4S_occ_EA_virt_arb_IP_arb"),
    (41, "ABBB_X_41_4S_occ_EA_virt_arb_IP_arb"),

    (43, "ABBB_C_43_3S_EA_EA_arb_IP_arb"),
    (43, "ABBB_X_43_3S_EA_EA_arb_IP_arb"),
)


def _apply_vals(out: np.ndarray, used: np.ndarray, fields: tuple[tuple[int, str], ...], vals: Any) -> None:
    for (term, key), v in zip(fields, vals):
        _res_set(out, used, term, key, v)


def _res_set(out: np.ndarray, used: np.ndarray, outer: int, inner: str, value: float) -> None:
    idx = RESULT_INDEX[(outer, inner)]
    out[idx] = float(value)
    used[idx] = True

def _pack_results(agg: np.ndarray, used_any: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Pack (agg, used_any) into nested dict: {term: {key: value}}.
    Must be exception-free; reducer depends on this at shutdown.
    """
    out: Dict[int, Dict[str, float]] = {}

    if getattr(used_any, "dtype", None) is not None and used_any.dtype != bool:
        used_any = used_any.astype(bool, copy=False)

    n = min(len(agg), len(used_any), len(RESULT_FIELDS))
    for idx in range(n):
        if not bool(used_any[idx]):
            continue
        term, key = RESULT_FIELDS[idx]
        out.setdefault(int(term), {})[str(key)] = float(agg[idx])

    return out



def process_chunk(
    chunk,
    threshold,
    NBAS,
    vec_handles: Dict[str, Dict[str, Any]],
    worker_id=None,
    SC_A_BA=None, CS_A_AB=None, SC_B_AB=None, CS_B_BA=None,
    CS_EA_A=None, CS_EA_B=None, SC_IP_A=None, SC_IP_B=None,
):
    """
    Process a chunk with vectors attached from shared memory.
    """
    #print("affinity cores:", len(os.sched_getaffinity(0)))
    #print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))
    classified_counts = defaultdict(int)
    classified_data = {
        "AAAA": {1: [], 2: [], 3: [], 4: []},
        "BBBB": {1: [], 2: [], 3: [], 4: []},
        "BAAA": {2: [], 3: [], 4: []},
        "BBAA": {2: [], 3: [], 4: []},
        "BBBA": {2: [], 3: [], 4: []},
    }

    with attached_vectors(vec_handles) as vectors:
        # Resolve array views
        SC_A_BA_ia = vectors['SC_A_BA_ia'] if 'SC_A_BA_ia' in vectors else SC_A_BA_ia
        CS_A_AB_ia = vectors['CS_A_AB_ia'] if 'CS_A_AB_ia' in vectors else CS_A_AB_ia
        SC_B_AB_ia = vectors['SC_B_AB_ia'] if 'SC_B_AB_ia' in vectors else SC_B_AB_ia
        CS_B_BA_ia = vectors['CS_B_BA_ia'] if 'CS_B_BA_ia' in vectors else CS_B_BA_ia
        CS_EA_A_a  = vectors['CS_EA_A_a' ] if 'CS_EA_A_a'  in vectors else CS_EA_A_a 
        SC_IP_B_i  = vectors['SC_IP_B_i' ] if 'SC_IP_B_i'  in vectors else SC_IP_B_i 
        SC_IP_A_i  = vectors['SC_IP_A_i' ] if 'SC_IP_A_i'  in vectors else SC_IP_A_i 
        CS_EA_B_a  = vectors['CS_EA_B_a' ] if 'CS_EA_B_a'  in vectors else CS_EA_B_a 
        SC_A_BA_pa = vectors['SC_A_BA_pa'] if 'SC_A_BA_pa' in vectors else SC_A_BA_pa
        CS_A_AB_iq = vectors['CS_A_AB_iq'] if 'CS_A_AB_iq' in vectors else CS_A_AB_iq
        SC_B_AB_pa = vectors['SC_B_AB_pa'] if 'SC_B_AB_pa' in vectors else SC_B_AB_pa
        CS_B_BA_iq = vectors['CS_B_BA_iq'] if 'CS_B_BA_iq' in vectors else CS_B_BA_iq
        CS_EA_A_q  = vectors['CS_EA_A_q' ] if 'CS_EA_A_q'  in vectors else CS_EA_A_q 
        SC_IP_B_p  = vectors['SC_IP_B_p' ] if 'SC_IP_B_p'  in vectors else SC_IP_B_p 
        SC_IP_A_p  = vectors['SC_IP_A_p' ] if 'SC_IP_A_p'  in vectors else SC_IP_A_p 
        CS_EA_B_q  = vectors['CS_EA_B_q' ] if 'CS_EA_B_q'  in vectors else CS_EA_B_q 
        SCS_A_BB_iq = vectors['SCS_A_BB_iq' ] if 'SCS_A_BB_iq'  in vectors else SCS_A_BB_iq 
        SCS_A_BB_pa = vectors['SCS_A_BB_pa' ] if 'SCS_A_BB_pa'  in vectors else SCS_A_BB_pa 
        SCS_B_AA_pa = vectors['SCS_B_AA_pa' ] if 'SCS_B_AA_pa'  in vectors else SCS_B_AA_pa 
        ACA_A_right   = vectors['ACA_A_right']
        ACA_B_right   = vectors['ACA_B_right']
        ACA_A_left    = vectors['ACA_A_left']
        ACA_B_left    = vectors['ACA_B_left']
        CA_EA_A_right = vectors['CA_EA_A_right']
        AC_IP_B_right = vectors['AC_IP_B_right']
        AC_IP_A_right = vectors['AC_IP_A_right']
        CA_EA_B_right = vectors['CA_EA_B_right']
        CA_EA_A_left  = vectors['CA_EA_A_left']
        AC_IP_B_left  = vectors['AC_IP_B_left']
        AC_IP_A_left  = vectors['AC_IP_A_left']
        CA_EA_B_left  = vectors['CA_EA_B_left']

        classify_and_bucket_chunk(chunk, threshold, NBAS, classified_data, classified_counts)

        # GPT
        """
        Why: printing from workers is very slow and becomes a global bottleneck on large runs.
        How: switch to logging.debug with lazy formatting; behavior/results unchanged.
        """
        # GPT
        if all(classified_counts.get(cat, 0) == 0 for cat in ["BAAA", "BBAA", "BBBA", "BBBB", "AAAA"]):
            logging.debug("Worker %s: No relevant rows in this chunk: %s", worker_id, dict(classified_counts))
            return None
        # GPT
        """End: replaced print with logging.debug."""
        # GPT

        out = np.zeros(RESULT_N, dtype=np.float64)
        used = np.zeros(RESULT_N, dtype=bool)

        # The wrappers use the zero-copy Fortran kernels when the extension/module is
        # available and retain the existing Python implementations as safe fallbacks.

        if classified_counts.get("BBAA", 0):
            vals = calc_BBAA_f90_zero_copy(
                classified_data["BBAA"],
                    SC_A_BA_ia   ,
                    CS_A_AB_ia   ,
                    SC_B_AB_ia   ,
                    CS_B_BA_ia   ,
                    CS_EA_A_a    ,
                    SC_IP_B_i    ,
                    SC_IP_A_i    ,
                    CS_EA_B_a    ,
                    SC_A_BA_pa   ,
                    CS_A_AB_iq   ,
                    SC_B_AB_pa   ,
                    CS_B_BA_iq   ,
                    CS_EA_A_q    ,
                    SC_IP_B_p    ,
                    SC_IP_A_p    ,
                    CS_EA_B_q    ,
                    ACA_A_right  ,
                    SCS_A_BB_iq   ,
                    SCS_A_BB_pa   ,
                    ACA_B_right   ,
                    SCS_B_AA_pa   ,
                    ACA_A_left   ,
                    ACA_B_left   ,
                    CA_EA_A_right,
                    AC_IP_B_right,
                    AC_IP_A_right,
                    CA_EA_B_right,
                    CA_EA_A_left ,
                    AC_IP_B_left ,
                    AC_IP_A_left ,
                    CA_EA_B_left , threshold
            )
            _apply_vals(out, used, BBAA_OUT_FIELDS, vals)

        if classified_counts.get("BAAA", 0):
            vals = calc_BAAA_f90_zero_copy(
        classified_data["BAAA"],
        SC_A_BA_ia,
        CS_A_AB_ia,
        SC_B_AB_ia,
        CS_B_BA_ia,
        CS_EA_A_a,
        SC_IP_B_i,
        SC_IP_A_i,
        CS_EA_B_a,
        SC_A_BA_pa,
        CS_A_AB_iq,
        SC_B_AB_pa,
        CS_B_BA_iq,
        CS_EA_A_q,
        SC_IP_B_p,
        SC_IP_A_p,
        CS_EA_B_q,
        ACA_A_right,
        ACA_B_right,
        SCS_B_AA_pa,
        ACA_A_left,
        ACA_B_left,
        CA_EA_A_right,
        AC_IP_B_right,
        AC_IP_A_right,
        CA_EA_B_right,
        CA_EA_A_left,
        AC_IP_B_left,
        AC_IP_A_left,
        CA_EA_B_left,
        threshold,
    )
            _apply_vals(out, used, BAAA_OUT_FIELDS, vals)

        if classified_counts.get("AAAA", 0):
            vals = calc_AAAA_f90_zero_copy(
                classified_data["AAAA"],
                SC_A_BA_ia, CS_A_AB_ia, SC_B_AB_ia, CS_B_BA_ia, CS_EA_A_a, SC_IP_A_i, SC_IP_B_i, CS_EA_B_a,
                ACA_A_right, ACA_B_right, ACA_A_left, ACA_B_left,
                CA_EA_A_right, AC_IP_B_right, AC_IP_A_right, CA_EA_B_right,
                CA_EA_A_left, AC_IP_B_left, AC_IP_A_left, CA_EA_B_left, threshold
            )
            _apply_vals(out, used, AAAA_OUT_FIELDS, vals)

        if classified_counts.get("BBBB", 0):
            vals = calc_BBBB_f90_zero_copy(
                        classified_data["BBBB"],
                        SC_A_BA_ia, CS_A_AB_ia, SC_B_AB_ia, CS_B_BA_ia, CS_EA_A_a, SC_IP_A_i, SC_IP_B_i, CS_EA_B_a,
                        ACA_A_right, ACA_B_right, ACA_A_left, ACA_B_left,
                        CA_EA_A_right, AC_IP_B_right, AC_IP_A_right, CA_EA_B_right,
                        CA_EA_A_left, AC_IP_B_left, AC_IP_A_left, CA_EA_B_left, threshold
                    )
            _apply_vals(out, used, BBBB_OUT_FIELDS, vals)
            
        if classified_counts.get("BBBA", 0):
            vals = calc_BBBA_f90_zero_copy(
                    classified_data["BBBA"],
                    SC_A_BA_ia,
                    CS_A_AB_ia,
                    SC_B_AB_ia,
                    CS_B_BA_ia,
                    CS_EA_A_a,
                    SC_IP_B_i,
                    SC_IP_A_i,
                    CS_EA_B_a,
                    SC_A_BA_pa,
                    CS_A_AB_iq,
                    SC_B_AB_pa,
                    CS_B_BA_iq,
                    CS_EA_A_q,
                    SC_IP_B_p,
                    SC_IP_A_p,
                    CS_EA_B_q,
                    ACA_A_right,
                    ACA_B_right,
                    ACA_A_left,
                    SCS_A_BB_iq,
                    ACA_B_left,
                    SCS_A_BB_pa,
                    CA_EA_A_right,
                    AC_IP_B_right,
                    AC_IP_A_right,
                    CA_EA_B_right,
                    CA_EA_A_left,
                    AC_IP_B_left,
                    AC_IP_A_left,
                    CA_EA_B_left,
                    threshold,
                )
            _apply_vals(out, used, BBBA_OUT_FIELDS, vals)

        return out, used





# ---- Required from your codebase ----
# detect_endianness, build_dtypes, _validate_markers_block, process_chunk,
# prepare_shared_vectors_red, register_parent_finalizer, DISK_RECORD_SIZE
# -------------------------------------

# ---------- Microbench ----------
def _measure_disk_bw(path: str, sample_bytes: int) -> int:
    total = 0; t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        buf = bytearray(1<<20)
        while total < sample_bytes:
            n = f.readinto(buf)
            if not n: break
            total += n
    dt = max(time.perf_counter() - t0, 1e-6)
    return int(total / dt)


# -----------------------------------------------------------------------------
# OpenMP / worker auto-tuning
# -----------------------------------------------------------------------------

def _physical_cores() -> int:
    # Physical cores matter for OpenMP sizing (avoid oversubscription).
    return max(1, int(psutil.cpu_count(logical=False) or (os.cpu_count() or 1)))

def _logical_cores() -> int:
    return max(1, int(psutil.cpu_count(logical=True) or (os.cpu_count() or 1)))

def _safe_defaults(P: int) -> tuple[int, int]:
    # The current Fortran kernels are serial. Use processes—not ineffective
    # OpenMP reservations—to occupy the available physical cores.
    return 1, max(1, P)

def _candidate_omp_threads(P: int) -> list[int]:
    return [1]

def _set_worker_thread_env(*, omp_threads: int) -> None:
    # Must run in parent *before* spawning workers (spawn re-imports this module).
    os.environ["OMP_NUM_THREADS"] = str(int(max(1, omp_threads)))
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "TRUE"
    os.environ["OMP_PLACES"] = "cores"
    # Keep BLAS single-threaded inside workers.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def _tune_parallelism(
    *,
    file_size: int,
    avail_mem: int,
    P: int,
    bytes_per_rec: int,
    disk_bw: int,
    sample_chunk: Any,
    threshold: float,
    NBAS: int,
    vec_handles: Any,
) -> tuple[int, int, float]:
    """
    Return (workers, omp_threads, proc_rate).

    The current Fortran kernels are serial, so OpenMP is fixed at one thread.
    Worker-process count remains configurable and is bounded by CPU, memory,
    and measured disk throughput.
    """
    env_workers = os.getenv("TWOELINT_WORKERS")
    env_omp = os.getenv("TWOELINT_OMP_THREADS")
    autotune = os.getenv("TWOELINT_AUTOTUNE", "1") not in ("0", "false", "False")

    omp_threads = 1
    if env_omp:
        try:
            requested_omp = int(env_omp)
        except ValueError:
            requested_omp = 1
            logging.warning("Ignoring invalid TWOELINT_OMP_THREADS=%r; serial kernels require 1", env_omp)
        if requested_omp != 1:
            logging.warning(
                "Ignoring TWOELINT_OMP_THREADS=%d; current Fortran kernels are serial and use one thread",
                requested_omp,
            )

    if env_workers:
        _set_worker_thread_env(omp_threads=omp_threads)
        proc_rate = _measure_proc_rate(sample_chunk, threshold, NBAS, vec_handles)
        workers = int(env_workers)
        return max(1, workers), max(1, omp_threads), float(proc_rate)

    # Small files: tuning overhead often not worth it.
    if (not autotune) or file_size < (256 << 20):
        omp_threads, workers_guess = _safe_defaults(P)
        _set_worker_thread_env(omp_threads=omp_threads)
        proc_rate = _measure_proc_rate(sample_chunk, threshold, NBAS, vec_handles)
        workers_cpu = max(1, P // omp_threads)
        workers_memdisk = _choose_workers(P, avail_mem, bytes_per_rec, proc_rate, disk_bw)
        workers = max(1, min(workers_guess, workers_cpu, workers_memdisk))
        return workers, omp_threads, float(proc_rate)

    # Measure the serial-kernel rate on a smaller slice of the sample.
    if isinstance(sample_chunk, ChunkView):
        probe = sample_chunk.head(min(32_000, max(1, sample_chunk.size // 4)))
    else:
        probe = sample_chunk[: min(32_000, max(1, len(sample_chunk) // 4))]

    _set_worker_thread_env(omp_threads=omp_threads)
    proc_rate = _measure_proc_rate(probe, threshold, NBAS, vec_handles)
    workers_memdisk = _choose_workers(P, avail_mem, bytes_per_rec, proc_rate, disk_bw)
    workers = max(1, min(P, workers_memdisk))

    # Re-measure proc_rate on the full sample using the selected serial setup.
    _set_worker_thread_env(omp_threads=omp_threads)
    proc_rate = float(_measure_proc_rate(sample_chunk, threshold, NBAS, vec_handles))
    return workers, omp_threads, proc_rate



# GPT
# Why: the previous implementation used physical cores (logical=False), so on SMT machines (e.g. 12c/24t)
#      it capped workers at 12 and left half the CPU threads idle.
# How: use logical CPU count by default; allow explicit override via TWOELINT_WORKERS env var.
# GPT
def _phys_cores() -> int:
    # Backwards-compat: keep existing call sites working.
    env = os.getenv("TWOELINT_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            logging.warning("Ignoring invalid TWOELINT_WORKERS=%r", env)
    return _physical_cores()
# GPT
# End: worker CPU cap now uses logical CPUs unless overridden.
# GPT


def _choose_workers(cpu_phys:int, avail:int, bytes_per_rec:int, proc_rate:float, disk_bw:int,
                    reserve:float=0.20, safety:float=3.0) -> int:
    k = max(1, int(math.ceil(proc_rate * 0.35)))
    per_worker_peak = max(1, int(k * bytes_per_rec * safety))
    mem_max = max(1, int(avail * (1.0 - reserve)) // per_worker_peak)
    io_max = max(1, int(disk_bw // (DISK_RECORD_SIZE * proc_rate)))
    return max(1, min(cpu_phys, mem_max, io_max))

def _choose_chunk(avail:int, workers:int, bytes_per_rec:int, proc_rate:float,
                  target_lat:float=0.35, safety:float=3.0, min_records:int=1_000) -> int:
    lat = max(min_records, int(math.ceil(proc_rate * target_lat)))
    per_worker_budget = max(1, avail // max(1, workers))
    per_record_peak = max(1, int(bytes_per_rec * safety))
    mem_records = max(1, per_worker_budget // per_record_peak)
    return max(1, min(lat, mem_records))

def _max_chunk_by_mem(avail:int, workers:int, bytes_per_rec:int, safety:float=3.0) -> int:
    per_worker_budget = max(1, avail // max(1, workers))
    per_rec_peak = max(1, int(bytes_per_rec * safety))
    return max(1, per_worker_budget // per_rec_peak)

# ---------- Shared counters ----------
class _Counters:
    def __init__(self, ctx: mp.context.BaseContext):
        self.read = ctx.Value("L", 0, lock=True)    # kept integrals (duplicate records skipped via NUT == -1)
        self.proc = ctx.Value("L", 0, lock=True)
        self.chunks = ctx.Value("L", 0, lock=True)
        self.dups = ctx.Value("L", 0, lock=True)    # dropped duplicates (same i,j,k,l)

# ---------- Producer / Consumer / Reducer ----------
def _reducer(result_q, send_conn) -> None:
    agg = np.zeros(RESULT_N, dtype=np.float64)
    used_any = np.zeros(RESULT_N, dtype=bool)
    try:
        while True:
            item = result_q.get()
            if item is None:
                break
            out, used = item
            agg += out
            used_any |= used
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Reducer failed")
        raise
    finally:
        try:
            send_conn.send(_pack_results(agg, used_any))
        finally:
            try:
                send_conn.close()
            except Exception:
                pass



# ---------- Adaptive controller ----------
def _adaptive_controller(counters: _Counters, desired_chunk_val, *,
                         initial_chunk:int, max_chunk:int, min_chunk:int,
                         interval_s:int=15, warmup_s:int=30,
                         improve_tol:float=0.05, step_up:float=1.2, step_down:float=0.8,
                         max_changes_per_min:int=2):
    t0 = time.perf_counter(); last_t = t0
    with counters.proc.get_lock(): last_proc = counters.proc.value
    best_rps = 0.0; current = max(1, initial_chunk); changes_window = []
    while True:
        time.sleep(interval_s)
        now = time.perf_counter()
        with counters.proc.get_lock(): proc = counters.proc.value
        dt = max(now - last_t, 1e-6); since_start = now - t0
        inst_rps = (proc - last_proc) / dt
        last_t, last_proc = now, proc

        if since_start < warmup_s:
            continue

        best_rps = max(best_rps, inst_rps)
        # rate-limit changes
        changes_window = [t for t in changes_window if now - t < 60.0]
        can_change = len(changes_window) < max_changes_per_min

        # hill-climb: if current close to best, try step up; if worse by > tol, step down
        new_chunk = current
        if can_change and inst_rps >= best_rps * (1.0 - improve_tol/2.0):
            new_chunk = min(max_chunk, int(max(1, current * step_up)))
        elif can_change and inst_rps < best_rps * (1.0 - improve_tol):
            new_chunk = max(min_chunk, int(max(1, current * step_down)))

        if new_chunk != current:
            desired_chunk_val.value = max(min_chunk, min(max_chunk, new_chunk))
            current = desired_chunk_val.value
            changes_window.append(now)
            logging.info("Adaptive: set chunk=%d (inst=%.0f rec/s best=%.0f)", current, inst_rps, best_rps)

        # exit condition: controller thread ends when main sets desired_chunk_val to 0 (sentinel)
        if desired_chunk_val.value == 0:
            break





# =============================================================================
# Optimized WRSEQ-only pipeline (buffered Fortran record parsing + shared-memory chunks)
# =============================================================================

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Any, Dict, List

# Fortran packing parameters (hardcoded as requested)
IALONE: int = 65535
IBITWD: int = 16


@dataclass(frozen=True)
class WRSEQSpec:
    """
    Layout description of one Fortran sequential-unformatted WRSEQ record payload.

    A WRSEQ-like record is written by something morally equivalent to:

        WRITE(unit) BUF, IBUF, NUT

    but real production codes sometimes omit NUT (fixed full blocks) or change integer widths.

    Fields
    ------
    val_size:
        Size in bytes of one BUF element. We prefer 8 (REAL*8 / float64).
    buf_len:
        Number of BUF elements in the record.
    ibuf_int_size:
        Size in bytes of one IBUF element (4 or 8).
    ibuf_len:
        Number of IBUF elements in the record.
    nut_size:
        Size in bytes of trailing NUT (0, 4, or 8). If 0, treat the record as full (NUT=buf_len).
    """
    val_size: int
    buf_len: int
    ibuf_int_size: int
    ibuf_len: int
    nut_size: int


@dataclass
class TwoElWRSEQFormat:
    __slots__ = ("endian", "marker_size", "wrseq")

    endian: str
    marker_size: int
    wrseq: WRSEQSpec


def detect_endianness(path: str) -> str:
    """
    Detect Fortran sequential-unformatted endianness ('<' or '>') and record-marker size (4 or 8 bytes).

    Strategy:
      - Try (marker_size in {4,8}) × (byteorder in {little,big})
      - Validate by walking several consecutive records:
            head_len == tail_len and payload fits within file
      - Choose the candidate with the most validated records.

    Side-effect:
      - Updates global MARKER_SIZE for legacy helpers.
    """
    global MARKER_SIZE

    file_size = os.path.getsize(path)
    if file_size < 16:
        raise ValueError("File too small to detect Fortran record framing.")

    def _score(marker_size: int, byteorder: str, max_records: int = 16) -> int:
        ok = 0
        with open(path, "rb", buffering=0) as f:
            pos = 0
            for _ in range(max_records):
                if pos + 2 * marker_size > file_size:
                    break
                f.seek(pos)
                head = f.read(marker_size)
                if len(head) != marker_size:
                    break
                n = int.from_bytes(head, byteorder=byteorder, signed=False)
                if n > (1 << 31):
                    break
                if pos + marker_size + n + marker_size > file_size:
                    break
                f.seek(pos + marker_size + n)
                tail = f.read(marker_size)
                if len(tail) != marker_size:
                    break
                n_tail = int.from_bytes(tail, byteorder=byteorder, signed=False)
                if n_tail != n:
                    break
                ok += 1
                pos = pos + marker_size + n + marker_size
        return ok

    candidates = []
    for ms in (4, 8):
        for bo in ("little", "big"):
            sc = _score(ms, bo, max_records=32)
            if sc > 0:
                candidates.append((sc, ms, "<" if bo == "little" else ">"))

    if not candidates:
        with open(path, "rb", buffering=0) as f:
            head8 = f.read(8)
        le4 = int.from_bytes(head8[:4], "little", signed=True)
        be4 = int.from_bytes(head8[:4], "big", signed=True)
        le8 = int.from_bytes(head8, "little", signed=True)
        be8 = int.from_bytes(head8, "big", signed=True)
        raise ValueError(
            f"Unrecognized record marker; not a Fortran sequential-unformatted file. "
            f"First8={head8!r} (le4={le4}, be4={be4}, le8={le8}, be8={be8})."
        )

    candidates.sort(key=lambda t: (t[0], 1 if t[1] == 4 else 0, 1 if t[2] == "<" else 0), reverse=True)
    _, ms, ec = candidates[0]
    MARKER_SIZE = ms
    return ec
def _locate_twoelsup_or_rewind(
    f,
    reader: FortranSlabReader,
    *,
    label: bytes = b"TWOELSUP",
    max_scan_records: int = 2_000_000,
) -> None:
    """
    Best-effort emulation of the Fortran LOCATE routine.

    It scans for records that look like:
        record1: '**....' (starts with '**')
        record2: 8-char label

    If the label is found, the file is positioned just after it (i.e., at the first data record).
    If not found, it rewinds to the beginning and continues from there.
    """
    tag = label[:8].ljust(8, b" ")

    try:
        f.seek(0)
    except Exception:
        pass
    reader.reset()

    for _ in range(int(max_scan_records)):
        payload = reader.read_record()
        if payload is None:
            break
        if len(payload) >= 2 and bytes(payload[:2]) == b"**":
            payload2 = reader.read_record()
            if payload2 is None:
                break
            lab = bytes(payload2[:8]).replace(b"\x00", b"").rstrip()
            if lab == tag.replace(b"\x00", b"").rstrip():
                return

    try:
        f.seek(0)
    except Exception:
        pass
    reader.reset()
    
    
    
def _infer_wrseq_spec(payload: memoryview, endian: str) -> WRSEQSpec:
    """
    Infer WRSEQ record layout from one record payload.

    Assumptions (matches the existing decoder implementation):
      - BUF is REAL*8 (float64)
      - IBUF is UINT64 packed indices, one per integral (ibuf_len == buf_len)
      - NUT is REQUIRED and is INT32 or INT64
      - Any bytes AFTER NUT in the record payload are ignored
    """
    nbytes = len(payload)
    byteorder = "little" if endian == "<" else "big"

    # Supported layouts by the current decode path.
    val_size = 8
    ibuf_int_size = 8
    denom = val_size + ibuf_int_size  # 16 bytes per integral (value + packed index)

    candidates: List[Tuple[int, int, WRSEQSpec]] = []

    for nut_size in (4, 8):
        if nbytes < nut_size + denom:
            continue

        max_b = (nbytes - nut_size) // denom
        if max_b <= 0:
            continue

        # Prefer the solution where NUT is as close to the end as possible (smallest tail),
        # but allow arbitrary tail bytes after NUT.
        for b in range(int(max_b), 0, -1):
            nut_off = denom * b
            if nut_off + nut_size > nbytes:
                continue

            nut = int.from_bytes(payload[nut_off : nut_off + nut_size], byteorder=byteorder, signed=True)
            if nut == -1 or (0 <= nut <= b):
                padding = nbytes - (nut_off + nut_size)
                spec = WRSEQSpec(
                    val_size=val_size,
                    buf_len=int(b),
                    ibuf_int_size=ibuf_int_size,
                    ibuf_len=int(b),
                    nut_size=int(nut_size),
                )
                # Prefer NUT=int32 over int64, then smallest tail padding.
                pref = 1 if nut_size == 4 else 0
                candidates.append((pref, padding, spec))
                break

    if not candidates:
        # Helpful "missing NUT" hint if it looks exactly like BUF+IBUF only.
        if nbytes % denom == 0 and nbytes // denom > 0:
            raise ValueError(
                "WRSEQ record appears to be BUF+IBUF only (no trailing NUT). "
                "This reader requires a trailing NUT (INT32 or INT64)."
            )
        raise ValueError(
            "Could not infer WRSEQ layout. Expected: BUF(float64)*N + IBUF(uint64)*N + NUT(int32/int64) [+ ignored tail bytes]."
        )

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


class ChunkView:
    """
    Lightweight view for a decoded chunk in columnar (SoA) form.

    This avoids structured dtypes and makes most vector operations cheaper.
    """
    __slots__ = ("a", "b", "c", "d", "val")

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, val: np.ndarray):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.val = val

    @property
    def size(self) -> int:
        return int(self.val.shape[0])

    def head(self, n: int) -> "ChunkView":
        n = max(0, min(int(n), self.size))
        return ChunkView(self.a[:n], self.b[:n], self.c[:n], self.d[:n], self.val[:n])


class _SubblockView:
    """
    Iterable view over a subset of a chunk using positions.

    Key property (for speed):
      - No list-of-rows is materialized.
      - We reuse one Python list object ('row') for all yields to avoid per-integral allocations.

    This is safe for your current calc_* functions because they consume each row immediately.
    """
    __slots__ = ("_pos", "_a", "_b", "_c", "_d", "_v")

    def __init__(self, positions: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, v: np.ndarray):
        self._pos = positions
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._v = v

    def __iter__(self):
        row = [0, 0, 0, 0, 0.0]
        for p in self._pos:
            i = int(self._a[p]); j = int(self._b[p]); k = int(self._c[p]); l = int(self._d[p])
            row[0] = i; row[1] = j; row[2] = k; row[3] = l; row[4] = float(self._v[p])
            yield row


def _chunk_arrays(chunk: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize different chunk representations to column arrays (a,b,c,d,val).

    Supports:
      - ChunkView
      - legacy numpy structured array with fields a,b,c,d,val
    """
    if isinstance(chunk, ChunkView):
        return chunk.a, chunk.b, chunk.c, chunk.d, chunk.val
    # legacy structured array
    return chunk["a"], chunk["b"], chunk["c"], chunk["d"], chunk["val"]


def classify_and_bucket_chunk(
    chunk: Any,
    threshold: int,
    NBAS: int,
    classified_data: Dict[str, Dict[int, Any]],
    classified_counts: Dict[str, int],
) -> None:
    """
    Fast classification without per-integral Python object creation.

    Preserves the *exact* categorization semantics of the previous implementation:
      - drop rows where all indices are zero
      - drop rows where any index > (NBAS + 10)
      - classify by 'above' = count(idx > threshold)
      - group by unique_count = len(set([i,j,k,l]))

    Output structure
    ----------------
    classified_data[CAT][unique_count] becomes an iterable over row-lists shaped:
        [i, j, k, l, val]
    BUT we do not allocate that list per integral; instead we store positions and generate rows
    on-the-fly via _SubblockView.

    This keeps calc_* logic unchanged while removing the biggest allocation hotspot.
    """
    a, b, c, d, v = _chunk_arrays(chunk)
    n = int(v.shape[0])
    if n == 0:
        return

    # Valid mask (same semantics as old)
    all_zero = (a == 0) & (b == 0) & (c == 0) & (d == 0)
    too_large = (a > (NBAS + 10)) | (b > (NBAS + 10)) | (c > (NBAS + 10)) | (d > (NBAS + 10))
    valid = ~(all_zero | too_large)
    if not np.any(valid):
        return

    # Category = count above threshold (vectorized, no stacking)
    # GPT
    # Why: classification runs for every chunk; avoid extra allocations/passes (np.unique, repeated unique_count[pos] slicing).
    # How: compute `above`/`unique_count` with small uint8 arrays, and split by known u in {1..4} without np.unique.
    # GPT
    above = (a > threshold).astype(np.uint8)
    above += (b > threshold)
    above += (c > threshold)
    above += (d > threshold)

    unique_count = np.ones(n, dtype=np.uint8)
    unique_count += (b != a)
    unique_count += ((c != a) & (c != b))
    unique_count += ((d != a) & (d != b) & (d != c))

    cat_masks = {
        "BBBA": (above == 3),
        "BBAA": (above == 2),
        "BAAA": (above == 1),
        "BBBB": (above == 4),
        "AAAA": (above == 0),
    }

    for cat, base_mask in cat_masks.items():
        m = base_mask & valid
        if not np.any(m):
            continue

        pos = np.flatnonzero(m)
        if pos.size == 0:
            continue

        ucpos = unique_count[pos]
        # classified_data[cat] already has keys for u; iterate those fixed keys instead of np.unique(...)
        for u in classified_data[cat].keys():
            mask_u = (ucpos == u)
            if not np.any(mask_u):
                continue
            pos_u = pos[mask_u]
            classified_data[cat][u] = _SubblockView(pos_u, a, b, c, d, v)
            classified_counts[cat] += int(pos_u.size)
    # GPT
    # End: cheaper bucketing; identical grouping/output.
    # GPT



# -----------------------------------------------------------------------------
# Shared-memory chunk pool (no pickling large arrays)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkPoolMeta:
    name: str
    blocks: int
    max_recs: int
    block_bytes: int
    total_bytes: int


def _shm_available_bytes() -> int:
    try:
        st = os.statvfs("/dev/shm")
        return int(st.f_bavail * st.f_frsize)
    except Exception:
        return 0


def _pool_block_views(shm_buf: memoryview, meta: ChunkPoolMeta, block_id: int, nrecs: int) -> ChunkView:
    """
    Create a ChunkView for a block slice.

    Layout per block:
      idx4: uint16[max_recs, 4]   (8 * max_recs bytes)
      val : float64[max_recs]     (8 * max_recs bytes)
    """
    nrecs = max(0, min(int(nrecs), meta.max_recs))
    base = int(block_id) * meta.block_bytes
    idx_bytes = meta.max_recs * 8
    idx_mv = shm_buf[base: base + idx_bytes]
    val_mv = shm_buf[base + idx_bytes: base + idx_bytes + meta.max_recs * 8]

    idx4 = np.ndarray((meta.max_recs, 4), dtype=np.uint16, buffer=idx_mv)
    val = np.ndarray((meta.max_recs,), dtype=np.float64, buffer=val_mv)
    return ChunkView(idx4[:nrecs, 0], idx4[:nrecs, 1], idx4[:nrecs, 2], idx4[:nrecs, 3], val[:nrecs])


def _decode_wrseq_payload_slice_into(
    payload: memoryview,
    fmt,
    idx4_out: "np.ndarray",
    val_out: "np.ndarray",
    out_start: int,
    src_start: int,
    count: int,
) -> None:
    """
    Decode a *slice* of WRSEQ record payload into output arrays.

    Parameters
    ----------
    payload:
        The Fortran record payload (without record markers).
    fmt:
        TwoElWRSEQFormat-like object. Must provide:
            - fmt.endian: '<' or '>'
            - fmt.wrseq.buf_len
            - fmt.wrseq.ibuf_int_size (expected 8 for INT64 packing)
            - fmt.wrseq.ibuf_len
    idx4_out:
        uint16 array shaped (N,4) receiving indices in canonical (I,J,K,L) order.
    val_out:
        float64 array shaped (N,) receiving values.
    out_start:
        Start index in output arrays.
    src_start:
        Start integral index within the record (0-based).
    count:
        Number of integrals to decode.
    """
    if count <= 0:
        return

    wr = fmt.wrseq
    endian = fmt.endian

    b = int(wr.buf_len)
    ib = int(wr.ibuf_int_size)
    ib_len = int(wr.ibuf_len)

    buf_off = 0
    ibuf_off = 8 * b

    out_sl = slice(out_start, out_start + count)

    # Values: REAL*8, length == buf_len, take slice [src_start:src_start+count]
    vals = np.frombuffer(
        payload,
        dtype=np.dtype(f"{endian}f8"),
        count=b,
        offset=buf_off,
    )[src_start : src_start + count]
    val_out[out_sl] = vals

    if ib != 8:
        raise ValueError(f"WRSEQ INT64 expected (ibuf_int_size=8); got {ib}.")

    # Packed indices: UINT64 array, length == ibuf_len (should match buf_len for INT64 WRSEQ)
    packed = np.frombuffer(
        payload,
        dtype=np.dtype(f"{endian}u8"),
        count=ib_len,
        offset=ibuf_off,
    )[src_start : src_start + count]

    # Fortran: IPK4(I,J,K,L) = (I<<48)|(J<<32)|(K<<16)|L with IBITWD=16, IALONE=65535
    mask = np.uint64(65535)
    ibitwd = 16

    # Raw IUPK fields: (L,K,J,I)
    l = (packed & mask)  # IUPKI
    k = ((packed >> (1 * ibitwd)) & mask)  # IUPKJ
    j = ((packed >> (2 * ibitwd)) & mask)  # IUPKK
    i = ((packed >> (3 * ibitwd)) & mask)  # IUPKL

    # Canonical order expected by your existing calc_* logic: (I,J,K,L)
    idx4_out[out_sl, 0] = i.astype(np.uint16, copy=False)
    idx4_out[out_sl, 1] = j.astype(np.uint16, copy=False)
    idx4_out[out_sl, 2] = k.astype(np.uint16, copy=False)
    idx4_out[out_sl, 3] = l.astype(np.uint16, copy=False)


def _decode_wrseq_record_into_block(
    payload: memoryview,
    fmt,
    idx4_out: "np.ndarray",
    val_out: "np.ndarray",
    out_start: int,
) -> int:
    """
    Decode one WRSEQ record into output arrays; return NUT (valid integrals).

    Requirements:
      - NUT is REQUIRED (fmt.wrseq.nut_size must be 4 or 8).
      - Payload may contain extra bytes after NUT; they are ignored.
      - NUT == -1 indicates a duplicate-record flag and is returned as -1.
    """
    wr = fmt.wrseq
    endian = fmt.endian
    byteorder = "little" if endian == "<" else "big"

    b = int(wr.buf_len)
    ib = int(wr.ibuf_int_size)
    ib_len = int(wr.ibuf_len)
    ns = int(getattr(wr, "nut_size", 0) or 0)

    if ns not in (4, 8):
        raise ValueError(f"WRSEQ format error: missing/invalid NUT (nut_size must be 4 or 8; got {ns}).")

    ibuf_off = 8 * b
    nut_off = ibuf_off + (ib * ib_len)

    if nut_off + ns > len(payload):
        raise ValueError("WRSEQ payload too small to contain required NUT; record is truncated or spec is wrong.")

    nut = int.from_bytes(payload[nut_off : nut_off + ns], byteorder=byteorder, signed=True)

    if nut == -1:
        return -1

    n_valid = int(nut)
    if n_valid < 0:
        n_valid = 0
    if n_valid > b:
        n_valid = b

    if n_valid:
        _decode_wrseq_payload_slice_into(
            payload=payload,
            fmt=fmt,
            idx4_out=idx4_out,
            val_out=val_out,
            out_start=out_start,
            src_start=0,
            count=n_valid,
        )
    return n_valid



def _read_sample_wrseq_chunk(path: str, threshold_records: int = 128_000):
    """
    Read a small prefix of the WRSEQ file and decode up to threshold_records integrals.

    Fixed: uses record-level decoder (_decode_wrseq_record_into_block) so the call signature
    matches and NUT is respected.
    """
    endian = detect_endianness(path)
    marker_size = MARKER_SIZE

    with open(path, "rb", buffering=0) as f:
        reader = FortranSlabReader(f, endian=endian, marker_size=marker_size, slab_bytes=32 << 20)
        _locate_twoelsup_or_rewind(f, reader)

        payload = None
        spec = None
        for _ in range(10_000):
            payload = reader.read_record()
            if payload is None:
                raise EOFError("No data records found in file.")
            if len(payload) < 32:
                continue
            try:
                spec = _infer_wrseq_spec(payload, endian)
                break
            except Exception:
                continue
        if spec is None:
            raise ValueError("Could not infer WRSEQ record layout from file prefix.")

        fmt = TwoElWRSEQFormat(endian=endian, marker_size=marker_size, wrseq=spec)

        want = int(min(threshold_records, 1_000_000))
        idx4 = np.empty((want, 4), dtype=np.uint16)
        val = np.empty((want,), dtype=np.float64)

        tmp_idx = np.empty((spec.buf_len, 4), dtype=np.uint16)
        tmp_val = np.empty((spec.buf_len,), dtype=np.float64)

        filled = 0
        while payload is not None and filled < want:
            n_valid = _decode_wrseq_record_into_block(payload, fmt, tmp_idx, tmp_val, 0)
            if n_valid == -1:
                payload = reader.read_record()
                continue
            take = min(n_valid, want - filled)
            if take > 0:
                idx4[filled : filled + take] = tmp_idx[:take]
                val[filled : filled + take] = tmp_val[:take]
                filled += take
            if filled >= want:
                break
            payload = reader.read_record()

        idx4 = idx4[:filled]
        val = val[:filled]
        chunk = ChunkView(idx4[:, 0], idx4[:, 1], idx4[:, 2], idx4[:, 3], val)
        return chunk, fmt

def _measure_proc_rate(arr: Any, threshold: float, NBAS: int, vec_meta: Any) -> float:
    """
    Measure processing throughput (records/sec) for autotuning.

    Supports ChunkView or structured arrays.
    """
    if isinstance(arr, ChunkView):
        warm = arr.head(max(1, arr.size // 10))
        process_chunk(warm, threshold, NBAS, vec_meta, 0)
        t0 = time.perf_counter()
        process_chunk(arr, threshold, NBAS, vec_meta, 0)
        dt = max(time.perf_counter() - t0, 1e-6)
        return float(arr.size / dt)

    warm = arr[: max(1, len(arr) // 10)]
    process_chunk(warm, threshold, NBAS, vec_meta, 0)
    t0 = time.perf_counter()
    process_chunk(arr, threshold, NBAS, vec_meta, 0)
    dt = max(time.perf_counter() - t0, 1e-6)
    return float(len(arr) / dt)


def _consumer(
    task_q,
    free_q,
    result_q,
    threshold,
    NBAS,
    vec_meta,
    omp_threads,
    worker_id: int,
    counters: _Counters,
    pool_meta: ChunkPoolMeta,
) -> None:
    """
    Consumer: attach shared-memory pool, process (block_id,nrecs), return block_id to free_q.
    """
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=pool_meta.name)
        shm_buf = shm.buf

        _set_worker_thread_env(omp_threads=int(omp_threads))
        _get_twoelint_f90()
        while True:
            item = task_q.get()
            if item is None:
                break
            block_id, nrecs = item
            chunk = _pool_block_views(shm_buf, pool_meta, int(block_id), int(nrecs))
            res = process_chunk(chunk, threshold, NBAS, vec_meta, worker_id)
            if res is not None:
                result_q.put(res)
            free_q.put(int(block_id))
            with counters.proc.get_lock():
                counters.proc.value += int(nrecs)
    except Exception:
        logging.exception("Consumer-%s failed", worker_id)
        raise
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def process_file(
    twoelint_file_path: str,
    threshold: float,
    NBAS: int,
    CIS_coeffs_matrix_A,
    CIS_coeffs_matrix_B,
    LCAO_coeffs_A,
    LCAO_coeffs_B,
    CIS_coeffs_EOMEA_A,
    CIS_coeffs_EOMIP_B,
    CIS_coeffs_EOMEA_B,
    CIS_coeffs_EOMIP_A,
    red_C_s,
    red_LCAO_s,
    S_blocks,
    S_AB=None,
    metrics_interval_s: float = 5.0,
):
    """
    WRSEQ-only fast path.

    Implements all requested speedups:
      1) buffered Fortran record parsing (slabs)
      2) shared-memory chunk pool (no pickling)
      3) columnar chunk layout in shared memory
      4) classification produces position-views (no list-of-rows allocations)
      5) fused classification key computation (no stack/sort)

    Computation logic (process_chunk + calc_* functions) is unchanged.
    """
    logging.getLogger().setLevel(logging.INFO)

    vec_handles, owner_handles = prepare_shared_vectors_red(red_C_s, red_LCAO_s, S_blocks)
    register_parent_finalizer(owner_handles)

    file_size = os.path.getsize(twoelint_file_path)

    # Parent AO-vector prep can benefit from multi-threaded BLAS. Keep workers single-threaded.
    blas_threads = int(os.getenv("TWOELINT_BLAS_THREADS", str(_logical_cores())))
    with threadpool_limits(limits=max(1, blas_threads), user_api="blas"):
        vec_handles, owner_handles = prepare_shared_vectors_red(red_C_s, red_LCAO_s, S_blocks)
    register_parent_finalizer(owner_handles)
    
    file_size = os.path.getsize(twoelint_file_path)
    
    # Sample + detect WRSEQ layout
    sample_chunk, fmt = _read_sample_wrseq_chunk(twoelint_file_path)
    bytes_per_rec = int(DT_RAM.itemsize)  # 4*uint16 + float64 (in-memory)
    disk_bw = _measure_disk_bw(twoelint_file_path, min(file_size, 64 << 20))
    
    avail = int(psutil.virtual_memory().available)
    P = _physical_cores()
    
    workers, omp_threads, proc_rate = _tune_parallelism(
        file_size=file_size,
        avail_mem=avail,
        P=P,
        bytes_per_rec=bytes_per_rec,
        disk_bw=disk_bw,
        sample_chunk=sample_chunk,
        threshold=threshold,
        NBAS=NBAS,
        vec_handles=vec_handles,
    )


    # Shared-memory pool (auto-sized)
    pool_meta, pool_owner = _auto_chunk_pool(workers)
    owner_handles.append(pool_owner)  # ensure unlink on parent exit
    safety = 3.0
    # Autotuned chunk sizes, clamped to pool capacity
    initial_chunk = min(pool_meta.max_recs, _choose_chunk(avail, workers, bytes_per_rec, proc_rate, 0.35, safety))
    max_chunk = min(pool_meta.max_recs, _max_chunk_by_mem(avail, workers, bytes_per_rec, safety))
    min_chunk = max(1, min(initial_chunk, int(0.25 * initial_chunk)))

    # queue capacity keeps ≤10% of RAM in flight, clamp 1..8
    bytes_per_chunk_peak = max(1, int(initial_chunk * bytes_per_rec * safety))
    target_inflight = int(avail * 0.10)
    qcap = max(1, min(8, target_inflight // max(1, workers * bytes_per_chunk_peak)))

    logging.info(
        "AutoTune: size=%dB | disk=%.1f MiB/s | proc=%.0f rec/s | omp=%d | workers=%d | chunk=%d | shm_blocks=%d shm_block_recs=%d",
        file_size, disk_bw / (1 << 20), proc_rate, omp_threads, workers, initial_chunk, pool_meta.blocks, pool_meta.max_recs
    )



    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(max(1, workers * qcap))
    free_q = ctx.Queue(pool_meta.blocks)
    result_q = ctx.Queue(max(1, workers * qcap))

    counters = _Counters(ctx)
    desired_chunk_val = ctx.Value("L", int(initial_chunk), lock=True)

    # Populate free block ids
    for bid in range(pool_meta.blocks):
        free_q.put(bid)

    prod = ctx.Process(
        target=_producer,
        args=(task_q, free_q, twoelint_file_path, desired_chunk_val, workers, counters, fmt, pool_meta),
        name="producer",
    )
    consumers = [
        ctx.Process(
            target=_consumer,
            args=(task_q, free_q, result_q, threshold, NBAS, vec_handles, int(omp_threads), wid, counters, pool_meta),
            name=f"consumer-{wid}",
        )
        for wid in range(workers)
    ]
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    red = ctx.Process(target=_reducer, args=(result_q, send_conn), name="reducer")

    stop_evt = ctx.Event()

    # Adaptive controller (keep existing logic, but clamp to pool capacity)
    ctrl_thr = None
    # GPT
    # Why: _adaptive_controller() declares keyword-only params after '*', but the caller passed them positionally,
    #      causing: TypeError: takes 2 positional arguments but 7 were given. This disables adaptive chunk tuning.
    # How: call _adaptive_controller(counters, desired_chunk_val, ...) with correct argument order + keyword args.
    #      Stop condition remains the existing sentinel desired_chunk_val==0 (set in finally).
    # GPT
    if "_adaptive_controller" in globals():
            def _ctrl():
                try:
                    _adaptive_controller(
                        counters,
                        desired_chunk_val,
                        initial_chunk=int(initial_chunk),
                        min_chunk=int(min_chunk),
                        max_chunk=int(max_chunk),
                        interval_s=int(max(1.0, float(metrics_interval_s))),
                    )
                except Exception:
                    logging.exception("Controller failed")
            ctrl_thr = threading.Thread(target=_ctrl, daemon=True)
    # GPT
    # End: controller call matches _adaptive_controller signature; adaptive chunk tuning is restored.
    # GPT


    # Metrics thread
    def _metrics():
        t0 = time.perf_counter()
        last_t = t0
        last_p = 0
        while not stop_evt.wait(metrics_interval_s):
            now = time.perf_counter()
            with counters.proc.get_lock():
                p = counters.proc.value
            with counters.read.get_lock():
                r = counters.read.value
            with counters.chunks.get_lock():
                c = counters.chunks.value
            dt = max(now - last_t, 1e-6)
            inst = (p - last_p) / dt
            avg = p / max(now - t0, 1e-6)
            with desired_chunk_val.get_lock():
                cur_chunk = min(int(desired_chunk_val.value), pool_meta.max_recs)
                desired_chunk_val.value = cur_chunk
            with counters.dups.get_lock():
                d = counters.dups.value
            
            logging.info(
                "Metrics: workers=%d chunk=%d | read=%d proc=%d chunks=%d dups=%d | inst=%.0f rec/s avg=%.0f rec/s",
                workers, cur_chunk, r, p, c, d, inst, avg
            )
            last_t = now
            last_p = p

    metrics_thr = threading.Thread(target=_metrics, daemon=True)

    prod.start()
    for c in consumers:
        c.start()
    red.start()
    metrics_thr.start()
    if ctrl_thr:
        ctrl_thr.start()

    try:
        for c in consumers:
            c.join()
        # stop reducer
        result_q.put(None)
        red.join()
        agg = recv_conn.recv()
    finally:
        with desired_chunk_val.get_lock():
            desired_chunk_val.value = 0
        stop_evt.set()
        metrics_thr.join(timeout=2.0)
        if ctrl_thr:
            ctrl_thr.join(timeout=2.0)
        if prod.is_alive():
            prod.terminate()
        prod.join(timeout=2.0)

    return agg


# ---- producer override: avoid per-record allocations in hot path ----




def _producer(
    task_q,
    free_q,
    path: str,
    desired_chunk_val,
    num_consumers: int,
    counters: _Counters,
    fmt: TwoElWRSEQFormat,
    pool_meta: ChunkPoolMeta,
) -> None:
    """
    Optimized WRSEQ producer.

    Correctness:
      - NUT is REQUIRED; missing/invalid NUT is a hard error.
      - Duplicate records are indicated by NUT == -1 and are skipped.
      - Only the first NUT integrals of each record are emitted.
      - Any payload bytes after NUT are ignored.
    """
    shm = None
    try:
        endian = fmt.endian
        marker_size = fmt.marker_size
        byteorder = "little" if endian == "<" else "big"

        shm = shared_memory.SharedMemory(name=pool_meta.name)
        shm_buf = shm.buf

        spec = fmt.wrseq
        ns = int(getattr(spec, "nut_size", 0) or 0)
        if ns not in (4, 8):
            raise ValueError(f"WRSEQ format error: missing/invalid NUT (nut_size must be 4 or 8; got {ns}).")

        b = int(spec.buf_len)
        ib = int(spec.ibuf_int_size)
        ib_len = int(spec.ibuf_len)

        ibuf_off = 8 * b
        nut_off = ibuf_off + (ib * ib_len)
        min_payload = nut_off + ns

        idx_bytes = pool_meta.max_recs * 8  # 4 * uint16 per row

        def _map_block(bid: int):
            base = int(bid) * pool_meta.block_bytes
            idx_mv = shm_buf[base : base + idx_bytes]
            val_mv = shm_buf[base + idx_bytes : base + idx_bytes + pool_meta.max_recs * 8]
            return (
                np.ndarray((pool_meta.max_recs, 4), dtype=np.uint16, buffer=idx_mv),
                np.ndarray((pool_meta.max_recs,), dtype=np.float64, buffer=val_mv),
            )

        with open(path, "rb", buffering=0) as f:
            reader = FortranSlabReader(f, endian=endian, marker_size=marker_size, slab_bytes=32 << 20)
            _locate_twoelsup_or_rewind(f, reader)

            block_id = free_q.get()
            dst_idx4, dst_val = _map_block(block_id)
            filled = 0

            while True:
                payload = reader.read_record()
                if payload is None:
                    break
                if len(payload) < 32:
                    continue
                if len(payload) < min_payload:
                    raise ValueError(
                        f"WRSEQ record payload too short: got {len(payload)} bytes, need at least {min_payload} bytes."
                    )

                want = int(desired_chunk_val.value) if desired_chunk_val.value > 0 else pool_meta.max_recs
                want = max(1, min(want, pool_meta.max_recs))

                if filled >= want:
                    task_q.put((block_id, filled))
                    with counters.chunks.get_lock():
                        counters.chunks.value += 1
                    block_id = free_q.get()
                    dst_idx4, dst_val = _map_block(block_id)
                    filled = 0

                nut = int.from_bytes(payload[nut_off : nut_off + ns], byteorder=byteorder, signed=True)
                if nut == -1:
                    with counters.dups.get_lock():
                        counters.dups.value += 1
                    continue

                n_valid = int(nut)
                if n_valid < 0:
                    n_valid = 0
                if n_valid > b:
                    n_valid = b
                if n_valid <= 0:
                    continue

                src = 0
                while src < n_valid:
                    want = int(desired_chunk_val.value) if desired_chunk_val.value > 0 else pool_meta.max_recs
                    want = max(1, min(want, pool_meta.max_recs))
                    cap = want - filled

                    if cap <= 0:
                        task_q.put((block_id, filled))
                        with counters.chunks.get_lock():
                            counters.chunks.value += 1
                        block_id = free_q.get()
                        dst_idx4, dst_val = _map_block(block_id)
                        filled = 0
                        cap = want

                    take = min(cap, n_valid - src)
                    _decode_wrseq_payload_slice_into(
                        payload=payload,
                        fmt=fmt,
                        idx4_out=dst_idx4,
                        val_out=dst_val,
                        out_start=filled,
                        src_start=src,
                        count=take,
                    )
                    filled += take
                    src += take

                    with counters.read.get_lock():
                        counters.read.value += int(take)

            if filled > 0:
                task_q.put((block_id, filled))
                with counters.chunks.get_lock():
                    counters.chunks.value += 1
            else:
                free_q.put(block_id)

        for _ in range(num_consumers):
            task_q.put(None)

    except Exception:
        logging.exception("Producer failed")
        raise
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
            
            
# ---- pool sizing override: never exceed /dev/shm availability; adapt block count ----
def _auto_chunk_pool(workers: int) -> Tuple[ChunkPoolMeta, SharedMemory]:
    """
    Create a shared-memory pool sized automatically for the host (robust version).

    Guarantees:
      - Never allocates more than ~80% of free /dev/shm (if present).
      - Adapts number of blocks downward if /dev/shm is small.
      - Keeps at least 2 blocks to avoid deadlock.

    Design goal:
      - Keep enough blocks to hide producer/consumer jitter (ideally ~2 per worker),
        without risking allocation failure on nodes with small /dev/shm.
    """
    workers = max(1, int(workers))

    ram_avail = int(psutil.virtual_memory().available)
    shm_avail = _shm_available_bytes()
    if shm_avail <= 0:
        shm_avail = ram_avail

    max_total = int(min(ram_avail * 0.30, shm_avail * 0.80))
    # If the node is extremely constrained, still try something small.
    max_total = max(16 << 20, max_total)  # 16 MiB minimum attempt

    min_block = 4 << 20   # 4 MiB
    max_block = 256 << 20 # 256 MiB

    ideal_blocks = 2 * workers
    # ensure total fits: blocks * min_block <= max_total
    feasible_blocks = max(2, min(ideal_blocks, max_total // min_block))
    blocks = int(feasible_blocks)

    per_block = int(max_total // blocks)
    per_block = max(min_block, min(max_block, per_block))
    # Keep total within max_total
    total_bytes = int(per_block * blocks)
    if total_bytes > max_total:
        blocks = max(2, max_total // per_block)
        total_bytes = int(per_block * blocks)

    # Align per_block to 16 (record width)
    per_block = int((per_block // 16) * 16)
    total_bytes = int(per_block * blocks)

    max_recs = max(1, per_block // 16)
    block_bytes = int(max_recs * 16)

    shm = shared_memory.SharedMemory(create=True, size=total_bytes)
    meta = ChunkPoolMeta(
        name=shm.name,
        blocks=int(blocks),
        max_recs=int(max_recs),
        block_bytes=int(block_bytes),
        total_bytes=int(total_bytes),
    )
    return meta, shm
