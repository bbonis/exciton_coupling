#!/usr/bin/env python3
import sys
import numpy as np
import re
import math
import pandas as pd
from string import ascii_uppercase
from typing import Any, Mapping

# Build custom order: '1'..'9', then '0', then 'A'..'Z'
_ORDER = [str(d) for d in range(1, 10)] + ['0'] + list(ascii_uppercase)
_RANK = {tok: i + 1 for i, tok in enumerate(_ORDER)}

def parse_hash_id(s: str) -> int:
    """
    Convert labels like '#1', '#9', '#0', '#A', '#B' into sortable integers.
    Order is: 1..9 < 0 < A < B < C < ...
    Returns a stable integer rank for sorting.
    """
    if not isinstance(s, str):
        raise TypeError("hash id must be a string like '#A' or '#3'")
    tok = s.strip().lstrip('#').upper()
    if tok in _RANK:              # fast path for 1..9, 0, A..Z
        return _RANK[tok]
    # Fallback: allow multi-char numeric like '#12' if ever present.
    try:
        return int(tok)
    except ValueError as exc:
        # Unknown token -> push to the end deterministically.
        # Adjust to your needs (raise instead if you prefer strictness).
        return 10_000_000

def order_dict_keys(d: Mapping[Any, Any]) -> dict[Any, Any]:
    """
    Return a new dict where keys are ordered by the same custom order as parse_hash_id():
    1..9 < 0 < A..Z, with deterministic ordering for unknown / non-string keys.
    """
    if not hasattr(d, "items"):
        raise TypeError("d must be a mapping (dict-like)")

    def sort_key(k: Any) -> tuple[int, str]:
        if isinstance(k, str):
            return (parse_hash_id(k), k.strip().lstrip("#").upper())
        return (10_000_001, str(k))

    return dict(sorted(d.items(), key=lambda kv: sort_key(kv[0])))

# Desired reordering logic based on atom and orbital (2PX, 2PY, 2PZ)
def reorder_p_functions(data):
    reordered = []

    # Group data by atom label
    atom_groups = {}
    for entry in data:
        index, row = entry
        atom_label = row[3]  # Column with atom label (e.g., '#1', '#2', etc.)

        if atom_label not in atom_groups:
            atom_groups[atom_label] = []
        atom_groups[atom_label].append((index, row))

    # Now reorder within each group
    for atom_label in order_dict_keys(atom_groups).keys():
        group = atom_groups[atom_label]
        
        unique_elements = list(set(line[1][4] for line in group))
        
        unique_elements.sort()
        
        n_groups = int(len(group) / len(unique_elements))
        
        goal_list = unique_elements * n_groups
        
        i = 0
        kdx = 0
        ndx = 1
        
        group_sorted = []
        
        # worked for DZ need to check the rest #TODO
        while i < len(group):
            if group[i][1][4] == goal_list[kdx]:
                group_sorted.append(group[i])
                del group[i]
                kdx += 1
                ndx = 1
            else:
                group[i], group[i + ndx] = group[i + ndx], group[i]
                ndx += 1
        
        reordered.extend(group_sorted)
                           
    
    return reordered

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

# Reordering D functions to: 3D0, 3D1+, 3D1-, 3D2-, 3D2+
def reorder_d_functions(block):
    order_d = ['3DXX', '3DYY', '3DZZ', '3DXY', '3DXZ', '3DYZ']
    reordered = []

    # Group data by atom label
    atom_groups = {}
    for entry in block:
        index, row = entry
        atom_label = row[3]  # Column with atom label (e.g., '#1', '#2', etc.)

        if atom_label not in atom_groups:
            atom_groups[atom_label] = []
        atom_groups[atom_label].append((index, row))

    # Now reorder within each group
    for atom_label in order_dict_keys(atom_groups).keys():
        group = atom_groups[atom_label]
        
        unique_elements = list(set(line[1][4] for line in group))
        
        unique_elements_order = sorted(unique_elements, key=lambda x: order_d.index(x))
        
        del(unique_elements)
        
        n_groups = int(len(group) / len(unique_elements_order))
        
        goal_list = unique_elements_order * n_groups
        
        i = 0
        kdx = 0
        ndx = 1
        
        group_sorted = []
        
        total_elements =  len(group)
        # worked for DZ need to check the rest #TODO
        while i < len(group):
            if group[i][1][4] == goal_list[kdx]:
                group_sorted.append(group[i])
                del group[i]
                kdx += 1
                ndx = 1
            else:
                group[i], group[i + ndx] = group[i + ndx], group[i]
                ndx += 1
        
        reordered.extend(group_sorted)
                           
    
    return reordered



def reorder_caos(filename, matrix, NBAS):
# CFOUR order: atoms then angular quantumnumbers
# Turbomole order: angular quantum number then atoms
# D order cfour: XX, XY, XZ, YY, YZ, ZZ
#         Turbomole: XX, YY, ZZ, XY, XZ, YZ
    
    print('WARNING: does not work for more than 1 set on each atom for the orbitals')
    
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip lines before the start marker and stop when end marker is reached
    start_index = -1
    end_index = -1
    for i, line in enumerate(lines):
        if 'IRREDUCIBLE REPRESENTATION NUMBER' in line:
            start_index = i + 2  # Start processing after the marker
        if 'NUCLEAR' in line:
            end_index = i  # Stop processing at the marker

    # Now parse the lines only within the relevant table range
    table_lines = lines[start_index:end_index]
    
    # Parsing the table into a list of lists (with original indexes)
    table = []
    for idx, line in enumerate(table_lines):
        split_line = line.split()
        table.append((start_index + idx, split_line))  # S

    # Lists to store relevant sections
    #table = [re.split(r'\s+', lin.strip()) for lin in table_1]
    
    # Function to reorder the table
    # Separate S, P, and D orbitals based on the 4th column (orbital type)
    print('WARNING: not implemented for f and higher orbitals')
    s_orbitals = [row for row in table if re.match(r'1S', row[1][4])]
    p_orbitals = [row for row in table if re.match(r'2P[XYZ]', row[1][4])]
    d_orbitals = [row for row in table if re.match(r'3D[0-9+-]+', row[1][4])]

    # Sort each group by the 5th column

    s_orbitals.sort(key=lambda x: parse_hash_id(x[1][3]))
    p_orbitals.sort(key=lambda x: parse_hash_id(x[1][3]))
    d_orbitals.sort(key=lambda x: parse_hash_id(x[1][3]))
    
# D order cfour: XX, XY, XZ, YY, YZ, ZZ
    
    for k in range(len(d_orbitals)):
        if 'D2-' in d_orbitals[k][1][4]:
            d_orbitals[k][1][4] = '3DXX'
        elif 'D1-' in d_orbitals[k][1][4]:
            d_orbitals[k][1][4] = '3DXY'
        elif 'D0' in d_orbitals[k][1][4]:
            d_orbitals[k][1][4] = '3DXZ'        
        elif 'D1+' in d_orbitals[k][1][4]:
            d_orbitals[k][1][4] = '3DYY'               
        elif 'D2+' in d_orbitals[k][1][4]:
            d_orbitals[k][1][4] = '3DYZ'               
        
    # Combine all sorted blocks
    reordered_table =  table#s_orbitals + p_orbitals + d_orbitals
    
    # Get the atom type (3rd column) from the D orbital
    inserted_lines = []  # Store inserted lines for index adjustment
    num_of_d = int(len(d_orbitals) / 5.)
    index_shift = 0
    for i in range(num_of_d):
        idx = int(5 * (i+1)) - 1
        atom_name = d_orbitals[idx][1][2]
        atom_type = d_orbitals[idx][1][3]
        
        # Prepare the extra line based on the atom type and orbital block
        # Calculate current index in the reordered table considering previous insertions
        current_index = reordered_table.index(d_orbitals[idx]) #+ index_shift 
        if 'YZ' in d_orbitals[idx][1][4]:
            num_ins_line = int((idx + 1) / 5.) - int(len(inserted_lines) )  #5=number of d saos
            prev_index = d_orbitals[idx][0]
            # Get the atom type (3rd column) from the D orbital
            atom_type = d_orbitals[idx][1][3]
            
            iter_back = num_ins_line 
            # Insert the extra line after the current '3D2+' block
            for ndx in range(num_ins_line):
                # Prepare the extra line with a new index (increment by 1 for the new line)
                iter_back -= 1
                extra_line = [str(int(d_orbitals[idx][1][0])+ ndx + 1), 'Extra', atom_name, atom_type, '3DZZ', '1', '1.0']  # Modify format as needed
                reordered_table.insert(current_index + ndx + 1, (prev_index + num_ins_line - iter_back, extra_line))
                inserted_lines.append(current_index + ndx + 1)  # Keep track of where the new lines were inserted
                # Increment the index shift because we've inserted a new line
            index_shift += num_ins_line
            for kdx in range(len(reordered_table)):
                if reordered_table[kdx][0] >= (prev_index + 1) and reordered_table[kdx][1][1] != 'Extra':
                    reordered_table[kdx] = (reordered_table[kdx][0] + num_ins_line, reordered_table[kdx][1])
                else:
                    continue
                
            # Needed for the current index variable
            for ldx in range(len(d_orbitals)):
                d_orbitals[ldx] = (d_orbitals[ldx][0] + num_ins_line, d_orbitals[ldx][1])    
        else:
            continue
    
    s_orbitals_cao = [row for row in reordered_table if re.match(r'1S', row[1][4])]
    p_orbitals_cao = [row for row in reordered_table if re.match(r'2P[XYZ]', row[1][4])]
    d_orbitals_cao = [row for row in reordered_table if re.match(r'3D[XYZ][XYZ]', row[1][4])]
    
    print('!!!!!ONly works for two sets of pt (DZ)')
    
    reordered_p_cao = reorder_p_functions(p_orbitals_cao)
    
    reordered_d_cao = reorder_d_functions(d_orbitals_cao)
    
    tm_ordered_table =  s_orbitals_cao + reordered_p_cao + reordered_d_cao        
    
    tm_ordered_table_for_rows = s_orbitals_cao + p_orbitals_cao + d_orbitals_cao
            
    reordered_indexes = [(row[0] - start_index) for row in tm_ordered_table]
    
    reordered_indexes_for_rows = [(row[0] - start_index) for row in tm_ordered_table_for_rows]
    print('WARNING: hardcoded for homodimer')
    
    # yes, the columns have to be reordered as well
    mos_tmp = np.zeros(shape=np.shape(matrix), dtype=np.float64)
    mos = np.zeros(shape=np.shape(matrix), dtype=np.float64)
    numpy_matrix = np.array(matrix)
    for i in range(len(reordered_indexes)):
        mos[:, i] = numpy_matrix[:, reordered_indexes[i] ]
    #mos.tolist()
    #for i in range(len(table)-index_shift):
    #    mos[i, :] = mos_tmp[reordered_indexes_for_rows[i], :]
    
    mos.tolist()
    #with open('list.txt', 'w') as f:
    #    for item in dimer_mos.T:
    #        f.write("%s\n" % item)
    
    return  mos


# Define the custom sort key
def custom_sort_key(x):
    # Extract the target element from column 3 (index 3)
    element = x[1][3]
    # Strip special characters
    stripped = ''.join(char for char in element if char.isalnum())
    
    if stripped.isdigit():  # If it's a digit
        if stripped == "0":  # Special case: prioritize '0' after '1-9'
            return (0, 10)  # Assign '0' a high value within the digit group
        return (0, int(stripped))  # Other digits are sorted numerically
    else:  # If it's a letter
        return (1, stripped)  # Group letters next, then sort alphabetically


def reorder_caos_fock(filename):
# CFOUR order: atoms then angular quantumnumbers
# Turbomole order: angular quantum number then atoms
# D order cfour: XX, XY, XZ, YY, YZ, ZZ
#         Turbomole: XX, YY, ZZ, XY, XZ, YZ
    
    print('WARNING: does not work for more than 1 set on each atom for the orbitals')
    
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip lines before the start marker and stop when end marker is reached
    start_index = -1
    end_index = -1
    for i, line in enumerate(lines):
        if 'IRREDUCIBLE REPRESENTATION NUMBER' in line:
            start_index = i + 2  # Start processing after the marker
        if 'NUCLEAR' in line:
            end_index = i  # Stop processing at the marker

    # Now parse the lines only within the relevant table range
    table_lines = lines[start_index:end_index]
    
    # Parsing the table into a list of lists (with original indexes)
    table = []
    for idx, line in enumerate(table_lines):
        split_line = line.split()
        table.append((start_index + idx, split_line))  # S

    # Lists to store relevant sections
    #table = [re.split(r'\s+', lin.strip()) for lin in table_1]
    
    # Function to reorder the table
    # Separate S, P, and D orbitals based on the 4th column (orbital type)
    print('WARNING: not implemented for f and higher orbitals')
    s_orbitals = [row for row in table if re.match(r'1S', row[1][4])]
    p_orbitals = [row for row in table if re.match(r'2P[XYZ]', row[1][4])]
    d_orbitals = [row for row in table if re.match(r'3D[0-9+-]+', row[1][4])]

    # Sort each group by the 5th column
    s_orbitals.sort(key=lambda x: parse_hash_id(x[1][3]))
    p_orbitals.sort(key=lambda x: parse_hash_id(x[1][3]))
    d_orbitals.sort(key=lambda x: parse_hash_id(x[1][3]))
    
    
# D order cfour: XX, XY, XZ, YY, YZ, ZZ
    
    tbrenorm_ind = {
        'd2m_ind':[],
        'd1p_ind':[]
    }
    
    for k in range(len(d_orbitals)):
        if 'D2-' in d_orbitals[k][1][4]:
            tbrenorm_ind['d2m_ind'].append(int(d_orbitals[k][1][1])-1)
            d_orbitals[k][1][4] = '3DXX'
        elif 'D1-' in d_orbitals[k][1][4]:
            d_orbitals[k][1][4] = '3DXY'
        elif 'D0' in d_orbitals[k][1][4]:
            d_orbitals[k][1][4] = '3DXZ'        
        elif 'D1+' in d_orbitals[k][1][4]:
            tbrenorm_ind['d1p_ind'].append(int(d_orbitals[k][1][1])-1)
            d_orbitals[k][1][4] = '3DYY'               
        elif 'D2+' in d_orbitals[k][1][4]:
            d_orbitals[k][1][4] = '3DYZ'               
        
    # Combine all sorted blocks
    reordered_table =  s_orbitals + p_orbitals + d_orbitals
    
    # Get the atom type (3rd column) from the D orbital
    inserted_lines = []  # Store inserted lines for index adjustment
    num_of_d = int(len(d_orbitals) / 5.)
    index_shift = 0
    for i in range(num_of_d):
        idx = int(5 * (i+1)) - 1
        atom_name = d_orbitals[idx][1][2]
        atom_type = d_orbitals[idx][1][3]
        
        # Prepare the extra line based on the atom type and orbital block
        # Calculate current index in the reordered table considering previous insertions
        current_index = reordered_table.index(d_orbitals[idx]) #+ index_shift 
        if 'YZ' in d_orbitals[idx][1][4]:
            num_ins_line = int((idx + 1) / 5.) - int(len(inserted_lines) )  #5=number of d saos
            prev_index = d_orbitals[idx][0]
            # Get the atom type (3rd column) from the D orbital
            atom_type = d_orbitals[idx][1][3]
            
            iter_back = num_ins_line 
            # Insert the extra line after the current '3D2+' block
            for ndx in range(num_ins_line):
                # Prepare the extra line with a new index (increment by 1 for the new line)
                iter_back -= 1
                extra_line = [str(int(d_orbitals[idx][1][0])+ ndx + 1), 'Extra', atom_name, atom_type, '3DZZ', '1', '1.0']  # Modify format as needed
                reordered_table.insert(current_index + ndx + 1, (prev_index + num_ins_line - iter_back, extra_line))
                inserted_lines.append(current_index + ndx + 1)  # Keep track of where the new lines were inserted
                # Increment the index shift because we've inserted a new line
            index_shift += num_ins_line
            for kdx in range(len(reordered_table)):
                if reordered_table[kdx][0] >= (prev_index + 1) and reordered_table[kdx][1][1] != 'Extra':
                    reordered_table[kdx] = (reordered_table[kdx][0] + num_ins_line, reordered_table[kdx][1])
                else:
                    continue
                
            # Needed for the current index variable
            for ldx in range(len(d_orbitals)):
                d_orbitals[ldx] = (d_orbitals[ldx][0] + num_ins_line, d_orbitals[ldx][1])    
        else:
            continue
    
    s_orbitals_cao = [row for row in reordered_table if re.match(r'1S', row[1][4])]
    p_orbitals_cao = [row for row in reordered_table if re.match(r'2P[XYZ]', row[1][4])]
    d_orbitals_cao = [row for row in reordered_table if re.match(r'3D[XYZ][XYZ]', row[1][4])]
    
    print('!!!!!ONly works for two sets of pt (DZ)')
    
    reordered_p_cao = reorder_p_functions(p_orbitals_cao)
    
    reordered_d_cao = reorder_d_functions(d_orbitals_cao)
    
    tm_ordered_table =  s_orbitals_cao + reordered_p_cao + reordered_d_cao        
    
            

    reordered_indexes = [(row[0] - start_index) for row in tm_ordered_table]
    
    return reordered_indexes, tbrenorm_ind

def process_AOSAO(file_name, SAO, CAO):
    with open(file_name) as f:
        raw_caosao = f.readlines()
    
    cao_matrix = np.zeros((CAO, SAO))# not sure, if this is correct at this point
    for kdx in range(len(raw_caosao)):
        idx = math.floor(kdx / CAO)
        if idx > 0:
            jdx = kdx - (idx * CAO)
        else:
            jdx = kdx
        cao_matrix[jdx, idx] = np.float64(raw_caosao[kdx])
        
    # Constants (replace with actual values)

    NBAS = SAO
    NAOTOT = CAO

    # Total number of elements
    N =  NBAS * NAOTOT

    # Read space-padded floats from file
    with open(file_name, 'r') as f:
        data = f.read().split()

    # Convert to NumPy array
    flat_array = np.array([float(x) for x in data], dtype=np.float64)

    # Check dimensions
    assert flat_array.size == N, f"Expected {N} values but got {flat_array.size}"

    # Reshape if desired (e.g., 3D or 2D matrix)
    matrix = flat_array.reshape(( NBAS, NAOTOT))
    
    return cao_matrix


def process_AOSAOINV(file_name, SAO, CAO):
    with open(file_name) as f:
        raw_caosao = f.readlines()
    
    cao_matrix = np.zeros((CAO, SAO))# not sure, if this is correct at this point
    for kdx in range(len(raw_caosao)):
        idx = math.floor(kdx / SAO)
        if idx > 0:
            jdx = kdx - (idx * SAO)
        else:
            jdx = kdx
        cao_matrix[idx, jdx] = np.float64(raw_caosao[kdx])
    return cao_matrix


def read_saocao(fname, nsao, ncao):
    """
    Read the SAO->CAO transformation matrix made by modified dscf or tm2molden.
    This matrix also convert orbital order to atomic order. 
    
    To transform from CAO to SAO one must use the inverse of this matrix!
    """
        
    saocao = np.zeros((nsao, ncao))

    with open(fname,"r") as f:
        for line in f:
            split = line.split()
            i, j, item = int(split[0])-1, int(split[1])-1, float(split[2])
            saocao[i,j] = item
    
    saocao_inv = np.linalg.pinv(saocao).T
    return saocao, saocao_inv


def renorm_read_saocao(fname, nsao, ncao):
    """
    Read the SAO->CAO transformation matrix made by modified dscf or tm2molden.
    This matrix also convert orbital order to atomic order. 
    
    To transform from CAO to SAO one must use the inverse of this matrix!
    """
        
    saocao = np.zeros((nsao, ncao))

    with open(fname,"r") as f:
        for line in f:
            split = line.split()
            i, j, item = int(split[0])-1, int(split[1])-1, float(split[2])
            if item == 0.0 or item == 1.0:
                saocao[i,j] = item
            else:
                if math.isclose(abs(item), 0.5):
                    saocao[i,j] = item * 2.0
                else:
                    saocao[i,j] = item * np.sqrt(12.0)
    
    saocao_inv = np.linalg.pinv(saocao).T
    return saocao, saocao_inv

def cfour_sao_to_TM_sao(MO_file_A, nsaos, aosao_file, aosaoinv_file, output_file, TM_aosao):
    LCAO = process_NEWMOS(file_name=MO_file_A, NBAS=nsaos, modes='None')
    df = pd.read_csv(TM_aosao, sep=r'\s+', header=None, dtype=str).to_numpy(dtype=np.float64)
    ncaos = int(max(df[:, 1]))
    del(df)
    aosao = process_AOSAO(file_name=aosao_file, SAO=nsaos, CAO=ncaos)
    aosaoinv = process_AOSAOINV(file_name=aosaoinv_file, SAO=nsaos, CAO=ncaos)

    #it worked like this, but needs to be tested!
    #aosao_LCAO = aosao @ LCAO
    cao = LCAO.T @ aosao.T #aosao_LCAO
    #aosao_LCAO = aosao.T @ LCAO
    #cao_2 = aosao_LCAO @ aosaoinv.T
    # We have cfour CAOs
    reordered_CAOs = reorder_caos(output_file, cao, ncaos)
    # We have TM ordered caos

    TM_sao_cao, TM_sao_cao_inv = read_saocao(TM_aosao, nsaos, ncaos)

    # This is the appropriate way to this in trubomole
    #TM_interm = TM_sao_cao_inv @ reordered_CAOs
#
    TM_saos = reordered_CAOs @ TM_sao_cao_inv.T
    
    print(f'CFOR SAO to TM Sao transformation is done')
    return TM_saos


def TM_sao_to_cfour_sao(fock_TM, nsaos, aosao_file, aosaoinv_file, output_file, TM_aosao):
    df = pd.read_csv(TM_aosao, sep=r'\s+', header=None, dtype=str).to_numpy(dtype=np.float64)
    ncaos = int(max(df[:, 1]))
    del(df)
    
    TM_sao_cao, TM_sao_cao_inv = read_saocao(TM_aosao, nsaos, ncaos)

    
    
    # This is the appropriate way to this in trubomole
    TM_interm = TM_sao_cao.T @ fock_TM

    fock_caos = TM_interm @ TM_sao_cao
    
    reord_ind, renorm = reorder_caos_fock(output_file)
    
    aosao = process_AOSAO(file_name=aosao_file, SAO=nsaos, CAO=ncaos)
    aosaoinv = process_AOSAOINV(file_name=aosaoinv_file, SAO=nsaos, CAO=ncaos)
    
    # yes, the columns have to be reordered as well
    fock_tmp = np.zeros(shape=(ncaos, ncaos), dtype=np.float64)
    fock_c4_cao = np.zeros(shape=(ncaos, ncaos), dtype=np.float64)
    fock_c4_cao_2 = np.zeros(shape=(ncaos, ncaos), dtype=np.float64)
    for i in range(len(fock_caos)):
        fock_tmp[reord_ind[i], : ] = fock_caos[i, :]
    for i in range(len(fock_caos)):
        fock_c4_cao[:, reord_ind[i]] = fock_tmp[:, i]
        

    for idx in range(len(fock_caos)):
        for jdx in range(len(fock_caos)):
            fock_c4_cao_2[reord_ind[idx], reord_ind[jdx]] = fock_caos[idx, jdx]
    aosaoinv_fin = np.linalg.pinv(aosao)
    aosao_fin = np.linalg.pinv(aosaoinv)
    #it worked like this, but needs to be tested!
    aosao_LCAO =  aosaoinv.T @ fock_c4_cao
    fock_sao = aosao_LCAO @ aosaoinv

    aosao_LCAO_2 = aosao.T @ fock_c4_cao_2
    fock_sao_2 = aosao_LCAO_2 @ aosaoinv
    # We have cfour CAOs
    # We have TM ordered caos
    for element in renorm['d2m_ind']:
        fock_sao[element, :] = fock_sao[element, :] * 12.0
        fock_sao[:, element] = fock_sao[:, element] * 12.0
    for element in renorm['d1p_ind']:
        fock_sao[element, :] = fock_sao[element, :] * 4.0
        fock_sao[:, element] = fock_sao[:, element] * 4.0
    
    np.savetxt('fock_sao_cfour_2', fock_sao_2, fmt='%.10f')
    print(f'CFOR SAO to TM Sao transformation is done')
    return fock_sao



if __name__ == '__main__':
    debug = True
    if debug:
        nsaos = 38
        ncaos = 40
        output_file = '/home/bonis/python_scripts/4A_input_files/out.dimer'
    else:
        nsaos = int(sys.argv[1])
        ncaos = int(sys.argv[2])
        output_file = str(sys.argv[3])
    
    MO_file_A = 'NEWMOS_A'
    MO_file_B = 'NEWMOS_B'
    aosao_file = 'AO2SO.txt'
    aosaoinv_file = 'AO2SOINV.txt'
    TM_aosao = 'saocao.dat'
    
    
    MO_file_A = 'NEWMOS_A'
    MO_file_B = 'NEWMOS_B'

    print('done')
