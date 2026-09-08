#!/usr/bin/env python3
import sys
import numpy as np
import final_codes.sao_cao_transform as sao_cao_transform
import pandas as pd
import math
import os
#from davids_turbomole import TM


def read_order_from_input(filename, matrix, which_fragment, NBAS_A, NBAS_B):
    # cfour order: for 2nd set of P: PX, PX, PY,PY, PZ, PZ
    #              for d: D-2, D-1, D0, D+1, D+2
    # turbomole order: for 2nd set of P: PX, PY, PZ, PX, PY, PZ
    #                  for d: D0, D+1, D-1, D-2, D+2 
    # Reading the input file and storing lines
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
    reordered_table = []
    two_p_blocks = []
    d_functions = []
    others = []

    # Function to reorder a block of 6 2P functions (if applicable)
    def reorder_two_p_block(block):
        if len(block) == 6:
            # Reordering 2P functions: 2PX, 2PY, 2PZ, 2PX, 2PY, 2PZ
            reordered = []
            count = len(block) // 3  # There should be 2 sets (6/3=2)
            for i in range(count):
                reordered.extend([block[i], block[i + count], block[i + 2 * count]])
            return reordered
        return block  # If not 6 rows, return the block as-is

    # Reordering D functions to: 3D0, 3D1+, 3D1-, 3D2-, 3D2+
    def reorder_d_functions(block):
        order_d = ['3D0', '3D1+', '3D1-', '3D2-', '3D2+']
        reordered = []
        for order in order_d:
            for row in block:
                if row[1][4] == order:
                    reordered.append(row)
        return reordered

    # Parsing through the rows and sorting into categories
    i = 0
    while i < len(table):
        index, data = table[i]
        if data[4] in ['2PX', '2PY', '2PZ']:
            # Find all consecutive 2P rows
            block = []
            while i < len(table) and table[i][1][4] in ['2PX', '2PY', '2PZ']:
                block.append(table[i])
                i += 1
            # Reorder block only if it contains 6 rows (two sets)
            if len(block) == 6:
                block = reorder_two_p_block(block)
            reordered_table.extend(block)
        elif data[4].startswith('3D'):
            # Collect all consecutive 3D rows for reordering
            block = []
            while i < len(table) and table[i][1][4].startswith('3D'):
                block.append(table[i])
                i += 1
            # Reorder D function block
            block = reorder_d_functions(block)
            reordered_table.extend(block)
        else:
            # All other rows remain unchanged
            reordered_table.append(table[i])
            i += 1

    # Extract the original indexes of the reordered rows
    reordered_indexes = [(row[0] - start_index)  for row in reordered_table]
    
    print('WARNING: hardcoded for homodimer')
    
    dimer_mos = np.zeros(shape=(NBAS_A, NBAS_A + NBAS_B), dtype=np.float64)
    dimer_mos.tolist()
    if which_fragment == 'A':
        for i in range(len(matrix)):
            dimer_mos[i][:NBAS_A] = matrix[reordered_indexes[i], :]
    
    if which_fragment == 'B':
        for i in range(len(matrix)):
            dimer_mos[i][NBAS_A:] = matrix[reordered_indexes[i], :]

    #with open('list.txt', 'w') as f:
    #    for item in dimer_mos.T:
    #        f.write("%s\n" % item)
    
    return  dimer_mos



def read_eigs_from_output(file_name):
    eig_vals = []
    is_lumo = True
    homo = []
    split_row = False
    with open(file_name, 'r') as f:
        for line in f:
            # CAREFUL at this point I decided to look at the symmetries, which always will be 'XXXX' or 'empty'. 
            # HAD TO BE CHANGED to 0.00000000 since the XXXX do not appear in some cases
            # if the keyword 'CONTINUUM=...' is used
            if 'ORBITAL EIGENVALUE' in line:
                split_row = True
            elif split_row:
                table_row = line.split()
                #if len(table_row) < 6:
                #   continue
                if table_row and table_row[0].isdigit():
                    if is_lumo and np.float64(table_row[2]) > 0:
                        homo = int(table_row[0]) - 1
                        is_lumo = False
                    else:
                        pass
                    eig_vals.append(np.float64(table_row[2]))
                elif table_row and table_row[0].strip('\n') == 'VSCF':
                    break
                else:
                    continue
            else:
                continue
    return int(homo), eig_vals


# From David Jelenfi
def TMformat(n):
    a = '{:.13E}'.format(float(abs(n)))
    e = a.find('E')
    if n < 0:
        return '-.{}{}D{}{:02d}'.format(a[0],a[2:e],a[e+1:e+2],abs(int(a[e+1:])*1+1))
    else:
        return '0.{}{}D{}{:02d}'.format(a[0],a[2:e],a[e+1:e+2],abs(int(a[e+1:])*1+1))

# From David Jelenfi
def write_turbomole_mo_file(nsao, supsys_eigs, supsys_mos):
    with open("new_mos","w") as f:
        f.write("$scfmo    scfconv=7   format(4d20.14)\n")
        for i in range(nsao):
            title  = "     "
            title += str(i+1)
            title += " a      eigenvalue="
            title += str(TMformat(supsys_eigs[i]))
            title += "   nsaos="
            title += str(nsao)
            title += "\n"
            f.write(title)

            coeff = ""
            for j in range(nsao):
                coeff += str(TMformat(supsys_mos[i,j]))
                if (j+1) % 4 == 0:
                    coeff += "\n"
            if coeff[-1] != "\n":
                coeff += "\n"

            f.write(coeff)

        f.write("$end\n")
    return print('Writing new_mos for turbomole is done!')


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


def add_path(file_name, dbg, debug_library_path):
    """Prepend the debug library path to the file name if dbg=True."""
    if dbg:
        return os.path.join(debug_library_path, file_name)
    return file_name


def format_number(val):
    # Handle sign of the number
    sign = "-" if val < 0 else " "
    
    # Get absolute value of val and convert it to scientific notation
    coeff, exponent = f"{abs(val):.20e}".split('e')
    
    # Convert coefficient to float and adjust exponent for normalization
    coeff_float = float(coeff)
    
    # Normalize the coefficient so that it starts with '0.'
    # This means shifting the decimal point so the first non-zero digit is after '0.'
    if coeff_float != 0:  # Avoid processing zero values
        while coeff_float >= 1:  # Shift coefficient down if necessary
            coeff_float /= 10
            exponent = int(exponent) + 1  # Adjust exponent accordingly
        
        while coeff_float < 0.1:  # Shift coefficient up if necessary
            coeff_float *= 10
            exponent = int(exponent) - 1  # Adjust exponent accordingly

    # Format the number with the adjusted coefficient and exponent
    formatted_number = f"{sign}{coeff_float:.20f}E{int(exponent):+03d}"
    
    return formatted_number



def create_dimer_NEWMOS(LCAO_A, LCAO_B, NBAS_A,  NBAS_B, file_path):
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
            dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_alpha[:, mon_idx]
        if (jdx % 4) == 3:
            dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_alpha[:, mon_idx]
    
    #fill the beta blocks
    for jdx in range(N_alpha_A + N_alpha_B, len(dimer_MOS_full)):
        mon_idx = math.floor((jdx - (N_alpha_A + N_alpha_B))/2)
        if (jdx % 4) == 0:
            dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_beta[:, mon_idx]
        if (jdx % 4) == 1:
            dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_beta[:, mon_idx]
        if (jdx % 4) == 2:
            dimer_MOS_full[0:NBAS_A, jdx] = LCAO_A_beta[:, mon_idx]
        if (jdx % 4) == 3:
            dimer_MOS_full[NBAS_A::, jdx] = LCAO_B_beta[:, mon_idx]
    
    dimer_MOs_formatted = dimer_MOS_full[0:len(dimer_MOS_full), 0:4]
    idx = 4
    while idx <= ( NBAS_A + NBAS_B):
        if idx + 4 > (NBAS_A + NBAS_B) and idx != (NBAS_A + NBAS_B):
             MOs_tmp = dimer_MOS_full[:,idx::]
             dimer_MOs_formatted = np.vstack((dimer_MOs_formatted, MOs_tmp))
             break
        elif idx == (NBAS_A + NBAS_B):
            break
        MOs_tmp = dimer_MOS_full[:, idx:(idx + 4)]
        dimer_MOs_formatted = np.vstack((dimer_MOs_formatted, MOs_tmp))
        idx += 4
    with open(file_path, 'w+') as f:
        for row in dimer_MOs_formatted:
            formatted_row = [format_number(x) for x in row]
            f.write("   " + "   ".join(formatted_row) + "\n")
    return print('NEWMOS file written')

def transform_fock(fock_file, nsaos):#, AOSO_file, AOSOINV_file, dimer_out, sao_caos_dim):
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


def main():
    if len(sys.argv) != 5:
        raise SystemExit("Usage: new_mos_c4_to_tm.py NBAS_A NBAS_B OUTPUT_A OUTPUT_B")
    dbg = False
    if dbg:
        distance = 2.00
        debug_library_path = "/home/bonis/python_scripts/10A_input_files"
        debug_library_path = "/home/bonis/python_scripts/3A_pp_tm_fock"#2A_input_files_corr_tm_fock""#3_21G_tm_fock"
    
        output_file_A = add_path('out.mon_A_local_exc', dbg, debug_library_path)
        output_file_B = add_path('out.mon_B_local_exc', dbg, debug_library_path)
        #output_file_dim = add_path(f'out.dimer_{float(distance):.2f}', dbg, debug_library_path)
        nsaos_A = 95 #38
        nsaos_B = 95 #40
    else:
        nsaos_A = int(sys.argv[1])
        nsaos_B = int(sys.argv[2])
        output_file_A = sys.argv[3]
        output_file_B = sys.argv[4]
    #    output_file_dim = sys.argv[5]


    if dbg:
    
        MO_file_A = add_path('NEWMOS_A', dbg, debug_library_path)
        MO_file_B = add_path('NEWMOS_B', dbg, debug_library_path)
        TM_aosao_file_A = add_path('saocao_A.dat', dbg, debug_library_path)
        TM_aosao_file_B = add_path('saocao_B.dat', dbg, debug_library_path)
        CF_aosao_file_A = add_path('AO2SO_A.txt', dbg, debug_library_path)
        CF_aosaoinv_file_A = add_path('AO2SOINV_A.txt', dbg, debug_library_path)
        CF_aosao_file_B = add_path('AO2SO_B.txt', dbg, debug_library_path)
        CF_aosaoinv_file_B = add_path('AO2SOINV_B.txt', dbg, debug_library_path)
        #TM_aosao_file_dim = add_path('saocao_dim.dat', dbg, debug_library_path)
        #CF_aosao_file_dim = add_path('AO2SO_dim.txt', dbg, debug_library_path)
        #CF_aosaoinv_file_dim = add_path('AO2SOINV_dim.txt', dbg, debug_library_path)    
        #dim_MOS_file = add_path('NEWMOS_dim', dbg, debug_library_path) 
    else:
        MO_file_A = 'NEWMOS_A'
        MO_file_B = 'NEWMOS_B'
        TM_aosao_file_A = 'saocao_A.dat'
        TM_aosao_file_B = 'saocao_B.dat'
        CF_aosao_file_A = 'AO2SO_A.txt'
        CF_aosaoinv_file_A = 'AO2SOINV_A.txt'
        CF_aosao_file_B = 'AO2SO_B.txt'
        CF_aosaoinv_file_B = 'AO2SOINV_B.txt'
        TM_aosao_file_dim = 'saocao_dim.dat'
        CF_aosao_file_dim =    'AO2SO_dim.txt'   
        CF_aosaoinv_file_dim = 'AO2SOINV_dim.txt'    
        dim_MOS_file =         'NEWMOS_dim'       
    
    #mine = transform_fock('../tm_fock_test_10A/tm_fock/fock.mo', nsaos_A + nsaos_B)    
    #david = transform_fock('fock.mo_davids', nsaos_A + nsaos_B)  
    #fmat =  transform_fock('../tm_fock_test_10A/tm_fock/fmat1.sao', nsaos_A + nsaos_B)
    #sfmat =  transform_fock('../tm_fock_test_10A/tm_fock/sfmat1.sao', nsaos_A + nsaos_B)
    #overlap =  transform_fock('../tm_fock_test_10A/tm_fock/overlap.sao', nsaos_A + nsaos_B)
    #
    #sfmat_2 = overlap @ fmat
    #
    #
    homo_A, eigvals_A = read_eigs_from_output(output_file_A)
    homo_B, eigvals_B = read_eigs_from_output(output_file_B)
    #supsys_eigs = eigvals_A + eigvals_B
    # For homodimer i need to sort this according to the energy. 
    print('!!!!!Warning, this could lead to problems, if we have negative virtuals!!!!!!!!!')
    # Get the sorted list
    # Get the original indices along with values
    # Manual selection sort to sort list1 and get reordering indices

    # According to David, the ordering is the same, as in cfour for the SAO-s, ie first atoms then angular quantum number
    # except for the d orbitals!
    LCAO_A = process_NEWMOS(file_name=MO_file_A, NBAS=nsaos_A, modes='None')
    LCAO_B = process_NEWMOS(file_name=MO_file_B, NBAS=nsaos_B, modes='None')

    reordered_MOS_A = sao_cao_transform.cfour_sao_to_TM_sao(MO_file_A, nsaos_A, CF_aosao_file_A, CF_aosaoinv_file_A, output_file_A, TM_aosao_file_A)
    reordered_MOS_B = sao_cao_transform.cfour_sao_to_TM_sao(MO_file_B, nsaos_B, CF_aosao_file_B, CF_aosaoinv_file_B, output_file_B, TM_aosao_file_B)
    zero_matrix = np.zeros((nsaos_B, nsaos_A))

    #l = ["../mon_A"]#,"../mon_B"]
    #M = len(l)
    #nsaos_A = 38
    #nsaos_B = 38
    #
    ##root = "../EMBEDDED_SUBSYSTEMS/SUBSYSTEM_AA"
    #mon_mos = []
    #mon_eigs = []
    #nsaos = []
    #homos = []
    #
    #
    #for system in l:
    #    path = system
    #    mon = TM(path)
    #    mon.read_mos()
    #    mon_mos.append(mon.mos)
    #    
    #TM_sao_cao, TM_sao_cao_inv = soa_cao_trasnform.read_saocao(TM_aosao_file_A, 38, 40)
    #
    #    
    #    # This is the appropriate way to this in trubomole
    #TM_interm = TM_sao_cao.T @ np.array(mon.mos[0])
    #
    #fock_caos = np.array(mon.mos) @ TM_sao_cao    #TM_sao_cao.T @ 
    #    
    #    
    #MOOOOOOOOOOOOO = reordered_MOS_A.T
    #LLLLLLCAO_A = LCAO_A.T
    #diff_mat, equality = matrices_are_equal(reordered_MOS_A,  mon.mos)
    #diff_mat_2, equality_2 = matrices_are_equal(LCAO_A.T, mon.mos)

    unordered_supsys_mos = np.hstack((np.vstack((reordered_MOS_A.T, zero_matrix)), np.vstack((zero_matrix.T, reordered_MOS_B.T)))).tolist()
    #unordered_supsys_mos = np.hstack(( np.vstack((zero_matrix, reordered_MOS_B.T)), np.vstack((reordered_MOS_A.T, zero_matrix)))).tolist()
    #unordered_supsys_mos = reordered_MOS_A.T.tolist()

    #LCAO_dim = create_dimer_NEWMOS(LCAO_A, LCAO_B, nsaos_A,  nsaos_B, dim_MOS_file)
    #unordered_supsys_mos = soa_cao_trasnform.cfour_sao_to_TM_sao(dim_MOS_file, nsaos_A+nsaos_B, CF_aosao_file_dim, CF_aosaoinv_file_dim, output_file_dim, TM_aosao_file_dim)


    l = ["A",'B'] #
    M = len(l)
    nsaos = [nsaos_A, nsaos_B]
    homos = [homo_A, homo_B] 
    nsao = int(nsaos_A) + int(nsaos_B)#np.sum(nsaos)
    homo = int(homo_A) + int(homo_B)#np.sum(homos)
    supsys_mos = np.zeros((nsao,nsao))
    supsys_eigs = np.zeros((nsao))
    n = len(supsys_eigs)
    reordering_indices = list(range(n))  # Initialize reordering indices [0, 1, 2, ...]

    mon_mos = [reordered_MOS_A, reordered_MOS_B]
    mon_eigs = [eigvals_A, eigvals_B]

    i,j = 0, homos[0]
    ni,nj = 0, nsaos[0]
    k,l = homo,homo+(nsaos[0] - homos[0])
    for system in range(M):
        supsys_mos[i:j,ni:nj] = mon_mos[system][:homos[system]]
        supsys_mos[k:l,ni:nj] = mon_mos[system][homos[system]:]
        supsys_eigs[i:j] = mon_eigs[system][:homos[system]]
        supsys_eigs[k:l] = mon_eigs[system][homos[system]:]
        if system < M-1:
            i +=  homos[system]
            j +=  homos[system+1]
            ni += nsaos[system]
            nj += nsaos[system+1]
            k +=  nsaos[system] - homos[system]
            l +=  nsaos[system+1] - homos[system+1]


    #for i in range(n):
    #    # Find the index of the smallest element in list1[i:]
    #    min_index = i
    #    for j in range(i + 1, n):
    #        if supsys_eigs[j] < supsys_eigs[min_index]:
    #            min_index = j
    #    
    #    # Swap the values in list1
    #    supsys_eigs[i], supsys_eigs[min_index] = supsys_eigs[min_index], supsys_eigs[i]
    #    
    #    # Swap the indices in reordering_indices to track the sorting process
    #    reordering_indices[i], reordering_indices[min_index] = reordering_indices[min_index], reordering_indices[i]

    # Now reorder the matrix according to reordering_indices
    #supsys_mos = [unordered_supsys_mos[i] for i in reordering_indices]

    write_turbomole_mo_file(nsao=nsao, supsys_eigs=supsys_eigs, supsys_mos=supsys_mos)



if __name__ == "__main__":
    main()
