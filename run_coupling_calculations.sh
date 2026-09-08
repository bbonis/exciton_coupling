#!/bin/bash
set -Eeuo pipefail
trap 'echo "Calculation failed at line $LINENO in $PWD; available intermediates retained." >&2' ERR
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(pwd)"
export PATH="${SCRIPT_DIR}:${PATH}"
if [[ -n "${CFOUR_BIN:-}" ]]; then
    export PATH="${CFOUR_BIN}:${PATH}"
fi
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/final_codes${PYTHONPATH:+:${PYTHONPATH}}"
export CT_DRIVER="${CT_DRIVER:-${SCRIPT_DIR}/CT_coupling_mixed_method_new_deriv_4C_final.py}"
CT_DRIVER="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CT_DRIVER")"
export CT_DRIVER

# Defaults match the supplied formaldehyde/cc-pVDZ input bundle.
monomer_1="${MONOMER_A:-forma}"
monomer_2="${MONOMER_B:-forma}"
for required in coupling_input geom_input saocao_mon_A.dat saocao_mon_B.dat control basis; do
    [[ -s "input_files/$required" ]] || { echo "Missing input_files/$required" >&2; exit 1; }
done
for executable in python3 xcfour dscf_smp_fock bc; do
    command -v "$executable" >/dev/null || { echo "Missing executable: $executable" >&2; exit 1; }
done
for helper in coupling_input_gen.py new_mos_c4_to_tm.py cis_vector_fixer.py; do
    [[ -f "$SCRIPT_DIR/$helper" ]] || { echo "Missing helper: $helper" >&2; exit 1; }
done
python3 - "$CT_DRIVER" <<'CHECK'
import importlib.util
import pathlib
import sys
missing = []
if not pathlib.Path(sys.argv[1]).is_file():
    missing.append(sys.argv[1])
for module in ("numpy", "pandas", "psutil", "threadpoolctl", "scipy", "matplotlib", "electrostatic", "twoelint_calc_fortran"):
    if importlib.util.find_spec(module) is None:
        missing.append(module)
if missing:
    raise SystemExit("Missing driver/modules: " + ", ".join(missing))
CHECK

# Fail before deleting anything outside the designated working directories.
cleanup() {
    case "$PWD" in
        "$ROOT_DIR/mon_1"|"$ROOT_DIR/mon_2"|"$ROOT_DIR/dimer"|"$ROOT_DIR/tm_fock"|"$ROOT_DIR/results")
            local candidate
            for candidate in "$@"; do
                if [[ -f "$candidate" || -L "$candidate" ]]; then
                    rm -f -- "$candidate"
                fi
            done
            return 0 ;;
        *) echo "Unexpected cleanup directory: $PWD" >&2; return 1 ;;
    esac
}
read_nbas() {
    local count
    count=$(awk '/functions in the AO basis/ {print $3; exit}' "$1")
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Cannot read AO basis size from $1" >&2; return 1; }
    printf '%s\n' "$count"
}

# ===================== LOGGING SETUP (CT) =====================
ROOT_DIR="$(pwd)"
LOGDIR="${ROOT_DIR}/results/logs"
mkdir -p "$LOGDIR"

# Unbuffered stdout/stderr so logs appear immediately (tail -f works)
export PYTHONUNBUFFERED=1

# DEBUG => hotloop_debug + benchmark
# INFO  => benchmark only
export CT_LOG_LEVEL="${CT_LOG_LEVEL:-INFO}"

# Wrapper that forces logging config (works with multiprocessing spawn)
# Wrapper that forces logging config (works with multiprocessing spawn)
WRAPPER="${ROOT_DIR}/run_ct_with_logging.py"
cat > "$WRAPPER" <<'PY'
import logging
import os
import runpy
import shutil
import sys
from pathlib import Path

_level_name = os.getenv("CT_LOG_LEVEL", "INFO").upper()
_level = logging.DEBUG if _level_name == "DEBUG" else logging.INFO

logging.basicConfig(
    level=_level,
    format="%(asctime)s %(processName)s %(levelname)s: %(message)s",
)

# Prevent your code from downgrading DEBUG -> INFO later.
_root = logging.getLogger()
_orig_set_level = _root.setLevel

def _locked_set_level(level: int) -> None:
    if _level == logging.DEBUG and level > logging.DEBUG:
        return
    _orig_set_level(level)

_root.setLevel = _locked_set_level  # type: ignore
_root.setLevel(_level)

def main() -> None:
    script = os.environ["CT_DRIVER"]
    script_path = Path(script).resolve()

    # CRITICAL: ensure sibling modules next to the CT script are importable
    sys.path.insert(0, str(script_path.parent))

    # Optional: allow extra PYTHONPATH injection from env (colon-separated)
    extra = os.getenv("CT_EXTRA_PYTHONPATH", "")
    for p in [x for x in extra.split(":") if x]:
        sys.path.insert(0, p)

    runpy.run_path(str(script_path), run_name="__main__")

if __name__ == "__main__":
    main()
PY
# =============================================================

#set calculation parameters
#TODO itt majd ezt nem kÃ¼lÃ¶n kÃ©ne beÃ­rni, hanem szedje ki az inputbÃ³l, csak ahhoz majd a geom inputot is Ã¡t kell struktÃºrÃ¡lni

#set up directories
mkdir -p mon_1
mkdir -p mon_2
mkdir -p dimer
mkdir -p results
mkdir -p results/onel_results
mkdir -p results/twoel_results
mkdir -p tm_fock
cp input_files/coupling_input results
cp input_files/geom_input results

#check for input files
if [ -f "${PWD}/input_files/coupling_input" ]; then
    echo "File coupling_input exists"
    cp input_files/coupling_input dimer
else
    echo "No coupling input!"
    exit
fi

if [ -f "${PWD}/input_files/geom_input" ]; then
    echo "File geom_input exists"
    cp input_files/geom_input mon_1
    cp input_files/geom_input mon_2
    cp input_files/geom_input dimer
    cp input_files/geom_input tm_fock
else
    echo "No geometry input!"
    exit
fi

# === DROP-IN CHANGE 1: replace your trim() with this (fixes CRLF) ===
trim() {
  local s="$1"
  s="${s%$'\r'}"                             # <-- IMPORTANT: strip CR, not 'r'
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

# === DROP-IN CHANGE 2: paste this helper right after trim() ===
should_skip_cis_vector_fixer() {
  local geom_file="${1:-geom_input}"
  local line nextline

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$(trim "$line")" == "mon calc method" ]]; then
      IFS= read -r nextline || nextline=""   # immediate next line (no skipping)
      [[ "$(trim "$nextline")" == "CIS" ]]   # EXACT match only
      return
    fi
  done < "$geom_file"

  return 1
}

GEOM_INPUT_FILE="geom_input"
#create input and do the calculation for monomer 1
# pretty hardcoded for names, should be change #TODO
cd mon_1 || { echo "Error no mon_1 directory"; exit 1; }
[[ -f "$GEOM_INPUT_FILE" ]] || { echo "MISSING $PWD/$GEOM_INPUT_FILE"; exit 1; }
python3 "$SCRIPT_DIR/coupling_input_gen.py" 0 "$monomer_1" "$monomer_2" mon_A Frenkel
xcfour > out.mon_A_local_exc 2>&1

NBAS_A=$(read_nbas out.mon_A_local_exc)
echo "${NBAS_A}"
if should_skip_cis_vector_fixer "$GEOM_INPUT_FILE"; then
  echo "skipping cis_vector_fixer.py"
else
  python3 "$SCRIPT_DIR/cis_vector_fixer.py" CIS_vector.csv
fi
cp CIS_vector.csv ../results/CCSD_matrix_A.csv
cp LTRANDENS_MO ../results/LTRANDENS_A
cp RTRANDENS_MO ../results/RTRANDENS_A
cp out.mon_A_local_exc ../results
cp NEWMOS ../results/NEWMOS_A
cp out.mon_A_local_exc ../tm_fock
cp NEWMOS ../tm_fock/NEWMOS_A
cp AO2SO.txt ../tm_fock/AO2SO_A.txt
cp AO2SOINV.txt ../tm_fock/AO2SOINV_A.txt
cp onel_molecu.csv ../results/onel_molecu_A.csv
cp ../input_files/saocao_mon_A.dat ../tm_fock/saocao_A.dat
cleanup B* ./*.csv d* G* i* j* M* s* D* F* f* I* J* N* O* V* CC* LTRANDENS* RTRANDENS* A* twoel*
python3 "$SCRIPT_DIR/coupling_input_gen.py" 0 "$monomer_1" "$monomer_2" mon_A CT
cp ZMAT.IP ZMAT
xcfour > out.mon_A_IP_exc 2>&1

if should_skip_cis_vector_fixer "$GEOM_INPUT_FILE"; then
  echo "skipping cis_vector_fixer.py"
else
  python3 "$SCRIPT_DIR/cis_vector_fixer.py" CIS_vector.csv
fi
cp CIS_vector.csv ../results/CCSD_vector_IP_A.csv
cp LTRANDENS_MO ../results/LTRANDENS_IP_A
cp RTRANDENS_MO ../results/RTRANDENS_IP_A
cp out.mon_A_IP_exc ../results
cleanup B* ./*.csv d* G* i* j* M* s* D* F* f* I* J* N* O* V* CC* LTRANDENS* RTRANDENS* A* twoel*
cp ZMAT.EA ZMAT
xcfour > out.mon_A_EA_exc 2>&1

if should_skip_cis_vector_fixer "$GEOM_INPUT_FILE"; then
  echo "skipping cis_vector_fixer.py"
else
  python3 "$SCRIPT_DIR/cis_vector_fixer.py" CIS_vector.csv
fi
cp CIS_vector.csv ../results/CCSD_vector_EA_A.csv
cp LTRANDENS_MO ../results/LTRANDENS_EA_A
cp RTRANDENS_MO ../results/RTRANDENS_EA_A
cp out.mon_A_EA_exc ../results
cleanup B* ./*.csv d* G* i* j* M* s* D* F* f* I* J* N* O* V* CC* LTRANDENS* RTRANDENS* A* twoel*
cd ..

#create input and do the calculation for monomer 2
cd mon_2 || { echo "Error no mon_2 directory"; exit 1; }
python3 "$SCRIPT_DIR/coupling_input_gen.py" 0 "$monomer_1" "$monomer_2" mon_B Frenkel
xcfour > out.mon_B_local_exc 2>&1

NBAS_B=$(read_nbas out.mon_B_local_exc)
echo "${NBAS_B}"
if should_skip_cis_vector_fixer "$GEOM_INPUT_FILE"; then
  echo "skipping cis_vector_fixer.py"
else
  python3 "$SCRIPT_DIR/cis_vector_fixer.py" CIS_vector.csv
fi
cp CIS_vector.csv ../results/CCSD_matrix_B.csv
cp LTRANDENS_MO ../results/LTRANDENS_B
cp RTRANDENS_MO ../results/RTRANDENS_B
cp out.mon_B_local_exc ../results
cp NEWMOS ../results/NEWMOS_B
cp out.mon_B_local_exc ../tm_fock
cp NEWMOS ../tm_fock/NEWMOS_B
cp AO2SO.txt ../tm_fock/AO2SO_B.txt
cp AO2SOINV.txt ../tm_fock/AO2SOINV_B.txt
cp onel_molecu.csv ../results/onel_molecu_B.csv
cp ../input_files/saocao_mon_B.dat ../tm_fock/saocao_B.dat
cleanup B* ./*.csv d* G* i* j* M* s* D* F* f* I* J* N* O* V* CC* LTRANDENS* RTRANDENS* A* twoel*
python3 "$SCRIPT_DIR/coupling_input_gen.py" 0 "$monomer_1" "$monomer_2" mon_B CT
cp ZMAT.IP ZMAT
xcfour > out.mon_B_IP_exc 2>&1

if should_skip_cis_vector_fixer "$GEOM_INPUT_FILE"; then
  echo "skipping cis_vector_fixer.py"
else
  python3 "$SCRIPT_DIR/cis_vector_fixer.py" CIS_vector.csv
fi
cp CIS_vector.csv ../results/CCSD_vector_IP_B.csv
cp LTRANDENS_MO ../results/LTRANDENS_IP_B
cp RTRANDENS_MO ../results/RTRANDENS_IP_B
cp out.mon_B_IP_exc ../results
cleanup B* ./*.csv d* G* i* j* M* s* D* F* f* I* J* N* O* V* CC* LTRANDENS* RTRANDENS* A* twoel*
cp ZMAT.EA ZMAT
xcfour > out.mon_B_EA_exc 2>&1

if should_skip_cis_vector_fixer "$GEOM_INPUT_FILE"; then
  echo "skipping cis_vector_fixer.py"
else
  python3 "$SCRIPT_DIR/cis_vector_fixer.py" CIS_vector.csv
fi
cp CIS_vector.csv ../results/CCSD_vector_EA_B.csv
cp LTRANDENS_MO ../results/LTRANDENS_EA_B
cp RTRANDENS_MO ../results/RTRANDENS_EA_B
cp out.mon_B_EA_exc ../results
cleanup B* ./*.csv d* G* i* j* M* s* D* F* f* I* J* N* O* V* CC* LTRANDENS* RTRANDENS* A* twoel*
cd ..

#start leading loop: dimer calculations
divider=10
read -r -a distance_numerators <<< "${DISTANCE_NUMERATORS:-20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 60 70 80 90 100}"
for k in "${distance_numerators[@]}"; do
    [[ "$k" =~ ^[0-9]+$ ]] || { echo "Invalid distance numerator: $k" >&2; exit 1; }
    echo "${k}"
    dist=$(echo "scale=2; $k / $divider" | bc)
    cd dimer || { echo "Error no dimer directory"; exit 1; }
    python3 "$SCRIPT_DIR/coupling_input_gen.py" "$dist" "$monomer_1" "$monomer_2" dimer Frenkel
    xcfour > out.dimer_"$dist" 2>&1

    cp overlap.csv ../results/overlap.csv
    cp IIII ../results/twoel.bin
    cp onel_molecu.csv ../results/onel_molecu.csv
    cp out.dimer_"$dist" ../results
    cp AO2SO.txt ../results/AO2SO_dim.txt
    cp AO2SOINV.txt ../results/AO2SOINV_dim.txt
    cp ../input_files/control ../tm_fock
    cp ../input_files/basis ../tm_fock
    cleanup B* ./*.csv d* G* i* j* M* s* F* f* I* J* N* O* Z* A*
    cd ../tm_fock || { echo "Error no tm_fock directory"; exit 1; }
    python3 "$SCRIPT_DIR/coupling_input_gen.py" "$dist" "$monomer_1" "$monomer_2" dimer TM
    python3 "$SCRIPT_DIR/new_mos_c4_to_tm.py" "$NBAS_A" "$NBAS_B" out.mon_A_local_exc out.mon_B_local_exc
    cp new_mos mos
    dscf_smp_fock > out.tm_dim_"$dist" 2>&1
    cp out.tm_dim_"$dist" ../results/
    cp fock.sao ../results/
    cp saocao.dat ../results/saocao_dim.dat
    cleanup c* b* f* h* m* ov* p* s* d* e* n* old*
    cp ../input_files/saocao_mon_A.dat saocao_A.dat
    cp ../input_files/saocao_mon_B.dat saocao_B.dat
    cd ../results || { echo "Error no results directory"; exit 1; }

    # Run CT with forced logging; logs go to:
    #   ${LOGDIR}/ct_${dist}.log  (stderr: benchmark + hotloop_debug if CT_LOG_LEVEL=DEBUG)
    #   ${LOGDIR}/ct_${dist}.out  (stdout)
    python3 "$WRAPPER" coupling_input "$dist" "$NBAS_A" "$NBAS_B" True \
      > "${LOGDIR}/ct_${dist}.out" \
      2> "${LOGDIR}/ct_${dist}.log"

    cleanup overlap.csv twoel.bin onel_molecu.csv NEWFOCK fock.sao saocao_dim.dat AO2SO_dim.txt AO2SOINV_dim.txt twoel*
    cd ..
done
