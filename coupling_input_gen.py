#!/usr/bin/env python3
from __future__ import annotations
import electrostatic as e
import numpy as np
import sys
from numpy.linalg import inv

# file: geom_parser.py


def extract_block_from_header_until_blank(
    header_marker: str,
    file_path: str = "geom_input",
    *,
    include_blank_line: bool = False,
    encoding: str = "utf-8",
) -> str:
    """
    Finds `header_marker` (e.g. "local exc") and returns everything AFTER it until the next blank line.

    - The header line itself is NOT included.
    - Preserves original newlines everywhere EXCEPT the last returned line: its trailing line break is removed.
    - Raises ValueError if the header marker is not found.
    """
    target = header_marker.strip()
    found_header = False
    collecting = False
    out: list[str] = []

    with open(file_path, "r", encoding=encoding, newline="") as f:
        for line in f:
            if not collecting:
                if line.strip() == target:
                    found_header = True
                    collecting = True
                continue

            if line.strip() == "":
                if include_blank_line:
                    out.append(line)
                break

            out.append(line)

    if not found_header:
        raise ValueError(f"Header marker {header_marker!r} not found in {file_path!r}")

    if out:
        last = out[-1]
        if last.endswith("\r\n"):
            out[-1] = last[:-2]
        elif last.endswith("\n") or last.endswith("\r"):
            out[-1] = last[:-1]

    return "".join(out)

def add_atoms(matrix, mol_name, mode):
    molecule = e.parameterfrominput(input_matrix, "geom %s" % mol_name, "string") #pyrrole
    mol_help = np.array(molecule)  # innentől
    help_matrix = np.array(mol_help[:, 0])
    out_geom = np.zeros((len(molecule), 4), dtype=object)
    if mode == 'normal':
        for j in range(0, len(molecule)):
            out_geom[j, :] = [str(help_matrix.copy()[j]), float(matrix.copy()[j, 0]), float(matrix.copy()[j, 1]),
                                float(matrix.copy()[j, 2])]
    if mode == 'CP':
        for j in range(0, len(molecule)):
            out_geom[j, :] = [str('GH'), float(matrix.copy()[j, 0]), float(matrix.copy()[j, 1]),
                                float(matrix.copy()[j, 2])]
    return out_geom


def rot_around_axis(molecule2, axis, rot):
    rot_mol2 = np.zeros((len(molecule2), 3), float)
    for k in range(0, len(molecule2)):
        print(type(k))
        print(molecule2.copy()[k, :])
        if axis == "z":
            rot_mol2[k, :] = e.rotation(molecule2.copy()[k, :], rot, 0., 0., "Dipole")
        elif axis == "y":
            rot_mol2[k, :] = e.rotation(molecule2.copy()[k, :], 0., rot, 0., "Dipole")
        elif axis == "x":
            rot_mol2[k, :] = e.rotation(molecule2.copy()[k, :], 0., 0., rot, "Dipole")
        else:
            print("Wrong axis")
    return rot_mol2


def shift_axis(molecule2, rot_mol2, distance_inp, axis, unit):
    mon_1 = np.zeros((len(molecule2), 3), float)
    mon_2 = np.zeros((len(rot_mol2), 3), float)
    distance = float(distance_inp)
    if unit == "Angstrohm":
        new_distance = e.angstromtobohr(distance)
    else:
        new_distance = distance
    print(new_distance)
    for i in range(0, len(molecule2)):
        if axis == "z":
            mon_1[i, :] = [molecule2.copy()[i, 0], molecule2.copy()[i, 1], molecule2.copy()[i, 2] - new_distance / 2.]
        elif axis == "y":
            mon_1[i, :] = [molecule2.copy()[i, 0], molecule2.copy()[i, 1] - new_distance / 2., molecule2.copy()[i, 2]]
        elif axis == "x":
            mon_1[i, :] = [molecule2.copy()[i, 0] - new_distance / 2., molecule2.copy()[i, 1], molecule2.copy()[i, 2]]
        else:
            print("Wrong axis")
    for j in range(0, len(rot_mol2)):
        if axis == "z":
            mon_2[j, :] = [rot_mol2.copy()[j, 0], rot_mol2.copy()[j, 1], rot_mol2.copy()[j, 2] + new_distance / 2.]
        elif axis == "y":
            mon_2[j, :] = [rot_mol2.copy()[j, 0], rot_mol2.copy()[j, 1] + new_distance / 2., rot_mol2.copy()[j, 2]]
        elif axis == "x":
            mon_2[j, :] = [rot_mol2.copy()[j, 0] + new_distance / 2., rot_mol2.copy()[j, 1], rot_mol2.copy()[j, 2]]
        else:
            print("Wrong axis")
    return mon_1, mon_2


def special_basis(mol_name1, mol_name2, set):
    molecule1 = e.parameterfrominput(input_matrix, "geom %s" % mol_name1, "string")  # pyrrole
    molecule2 = e.parameterfrominput(input_matrix, "geom %s" % mol_name2, "string")
    mol_help1 = np.array(molecule1)  # innentől
    mol_help2 = np.array(molecule2)
    help_matrix1 = np.array(mol_help1[:, 0])
    help_matrix2 = np.array(mol_help2[:, 0])
    full_help = np.concatenate((help_matrix1, help_matrix2))
    basis = np.zeros(( len(full_help), 1), dtype=object)
    for o in range(0, len(basis)):
        basis[o] = '%s' % full_help[o] + ':' + set
    return basis

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
    
def get_elstat(input_file: str, basis: str, dimer: str, exc_state: str):
    found_it = False
    final_matrix = []
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


def excite_interpreter(input_file: str, exc_state_type: str, og_num_state):
    num_of_exc = 0
    num_of_lines = 0
    line_counter = 0
    exc_counter = 0
    final_line_to_write = ''
    found_it = False
    is_line_written = False
    with open(input_file, 'r+') as f:
        for lines in f:
            if lines.find(f'%excite* {exc_state_type}') != -1:
                found_it = True
                final_line_to_write += f'%excite*\n'
                is_line_written = True
                continue
            else:
                pass
            if found_it:
                num_of_exc = int(lines.split()[0])
                final_line_to_write += f'{num_of_exc}\n'
                found_it = False
                continue
            else:   
                pass
            if num_of_exc > 0 and num_of_lines == 0:
                exc_counter += 1
                num_of_lines = int(lines.split()[0])
                final_line_to_write += f'{num_of_lines}\n'
            elif num_of_exc > 0 and num_of_lines >= 0:
                line_counter += 1
                if line_counter <  num_of_lines:
                    final_line_to_write += f'{lines}'
                elif line_counter ==  num_of_lines:
                    final_line_to_write += f'{lines}'
                    line_counter = 0
                    num_of_lines = 0
                    if num_of_exc == exc_counter and num_of_exc != 0:
                        break
                    else:
                        continue
                else:
                    pass
            elif found_it and not lines.strip():
                break
            else:
                pass
    if is_line_written:
        return final_line_to_write, num_of_exc
    else:
        print(f'There was no excitation specified')
        return final_line_to_write, og_num_state


if len(sys.argv) != 6:
    raise SystemExit("Usage: coupling_input_gen.py DIST MONOMER_A MONOMER_B mon_A|mon_B|dimer Frenkel|CT|TM")

debug = False
row_list = ['0']
input_matrix = readinput("geom_input")
calc_method: str = input_matrix[0]
#calc_type = input_matrix[4]
num_of_atoms: int = input_matrix[2]
print(type(num_of_atoms))
unit: str = input_matrix[5]
rot_axis: str = input_matrix[6]
shift_a: str = input_matrix[7]
rot_quantity = float(input_matrix[8])
basis = input_matrix[9].strip()
counter = 0
if debug:
    v_str = '0'
    mon_A_name = 'forma'
    mon_B_name = 'forma'
    which_input = 'dimer'
    calc_type = 'TM'
else:
    v_str = sys.argv[1]
    mon_A_name = sys.argv[2]
    mon_B_name = sys.argv[3]
    which_input = sys.argv[4] 
    calc_type = sys.argv[5]
excite_method = ''
exc_state_properties = ''
if 'CC' in calc_method:
    excite_method = 'EOMEE'
    exc_state_properties = ',ESTATE_PROP=1,EOMFOLLOW=OVERLAP'
    state_num = '5'
elif 'HF' in calc_method or 'CIS' in calc_method:
    excite_method = 'CIS'
    state_num = '50'
else:
    print('wrong input')
if mon_A_name == mon_B_name:
    print('HOMODIMER!')
    molecule = np.array(e.parameterfrominput(input_matrix, f'geom {mon_A_name}', "float"))
    #rot_mol_2 = rot_around_axis(molecule, 'y', 76)
    #rot_mol_3 = rot_around_axis(rot_mol_2, 'x', 143)
    
    rot_mol = rot_around_axis(molecule, '%s' % rot_axis, rot_quantity)
    mon_A_matrix, mon_B_matrix = shift_axis(molecule, rot_mol, '%s' % v_str, '%s' % shift_a, '%s' % unit)
else:
    molecule1 = np.array(e.parameterfrominput(input_matrix, f'geom {mon_A_name}', "float"))#ezt kell átírni a cyt uranal
    molecule2 = np.array(e.parameterfrominput(input_matrix, f'geom {mon_B_name}', "float"))#meg ezt
    mon_A_matrix, mon_B_matrix = shift_axis(molecule1, molecule2, '%s' % v_str, '%s' % shift_a, '%s' % unit)
if calc_type == 'TM' and which_input == 'dimer':
    with open('coord', 'w+') as f:
        mon_A_write = add_atoms(mon_A_matrix, mon_A_name, 'normal')#át
        mon_B_write = add_atoms(mon_B_matrix, mon_B_name, 'normal')# pyrrolehoz:
        n_atoms_A = len(mon_A_write)
        n_atoms_B = len(mon_B_write)
        n_atom = int(n_atoms_A + n_atoms_B)
        f.write(f'$coord  natoms=     {n_atom}\n')
        for row in mon_A_write:
            f.write(f"{float(row[1]):>16.12f} {float(row[2]):>16.12f} {float(row[3]):>16.12f}    {row[0].lower()}\n")
        for row in mon_B_write:
            f.write(f"{float(row[1]):>16.12f} {float(row[2]):>16.12f} {float(row[3]):>16.12f}    {row[0].lower()}\n")
        f.write(f'$user-defined bonds\n')
        f.write(f'$end')
elif calc_type == 'Frenkel' or calc_type == 'local':
    if which_input == 'mon_A':
        with open('ZMAT', 'w+') as f:
        #f.write('%s-' % mol + '%s ' % mol + '%s A ' % v_str + '%s' % calc_type + '\n')
            f.write(f'{mon_A_name} monomer calc, used for coupling \n')
            mon_A_write = add_atoms(mon_A_matrix, mon_A_name, 'normal')#át
            for u in range(0, len(mon_A_write)):
                f.write('%s ' % mon_A_write[u, 0] + '%f\t' % mon_A_write[u, 1] +
                    '%f\t' % mon_A_write[u, 2] + '%f\t' % mon_A_write[u, 3] + '\n')
            excite = extract_block_from_header_until_blank("local exc mon_A", "geom_input")
            mon_method = extract_block_from_header_until_blank("mon calc method", "geom_input") 
            excite_method='EOMEE'
            excitation_spec = ',ESTATE_PROP=1,EOMFOLLOW=OVERLAP\nEOM_NSTATE=MULTIROOT'   
            state_num = 50
            if 'CIS' in mon_method:
                excite_method=mon_method
                excitation_spec = ''
                state_num = 50
            f.write('\n')
            f.write(f'*CFOUR(CALC={mon_method},BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
                    f'\nCOORDINATES=CARTESIAN,EXCITE={excite_method}{excitation_spec}\nESTATE_SYM={state_num}\nFROZEN_CORE=ON'
                    f'\nMEMORY=50,MEM_UNIT=GB,PRINT=20)')
            f.write(f'\n\n{excite}\n \n \n \n')
            f.close()
    elif which_input == 'mon_B':
        with open('ZMAT', 'w+') as f:
        #f.write('%s-' % mol + '%s ' % mol + '%s A ' % v_str + '%s' % calc_type + '\n')
            f.write(f'{mon_B_name} monomer calc, used for coupling \n')
            mon_B_write = add_atoms(mon_B_matrix, mon_B_name, 'normal')# pyrrolehoz:
            for w in range(0, len(mon_B_write)):
                f.write('%s ' % mon_B_write[w, 0] + '%f\t' % mon_B_write[w, 1] +
                    '%f\t' % mon_B_write[w, 2] + '%f\t' % mon_B_write[w, 3] + '\n')
            #final_line_EE, state_num = excite_interpreter("geom_input", 'EE', state_num)
            excite = extract_block_from_header_until_blank("local exc mon_B", "geom_input")
            mon_method = extract_block_from_header_until_blank("mon calc method", "geom_input")    
            excite_method='EOMEE'
            excitation_spec = ',ESTATE_PROP=1,EOMFOLLOW=OVERLAP\nEOM_NSTATE=MULTIROOT'
            state_num = 50
            if 'CIS' in mon_method:
                excite_method=mon_method
                excitation_spec = ''
                state_num = 50
            f.write('\n')
            f.write(f'*CFOUR(CALC={mon_method},BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
                    f'\nCOORDINATES=CARTESIAN,EXCITE={excite_method}{excitation_spec}\nESTATE_SYM={state_num}\nFROZEN_CORE=ON'
                    f'\nMEMORY=50,MEM_UNIT=GB,PRINT=20)')
            f.write(f'\n\n{excite}\n \n \n \n')
            f.close()
    elif which_input == 'dimer':
        with open('ZMAT', 'w+') as f:
            #f.write('%s-' % mol + '%s ' % mol + '%s A ' % v_str + '%s' % calc_type + '\n')
            f.write('cyt-cyt dist  rot %s \n' % v_str)
            mon_A_write = add_atoms(mon_A_matrix, mon_A_name, 'normal')#át
            #mon2_write = add_atoms(mon2, 'pyr', 'normal')#át ha másik dimerkutfuvztt akarsz
            mon_B_write = add_atoms(mon_B_matrix, mon_B_name, 'normal')# pyrrolehoz:
            for u in range(0, len(mon_A_write)):
                f.write('%s ' % mon_A_write[u, 0] + '%f\t' % mon_A_write[u, 1] +
                    '%f\t' % mon_A_write[u, 2] + '%f\t' % mon_A_write[u, 3] + '\n')
            for w in range(0, len(mon_B_write)):
                f.write('%s ' % mon_B_write[w, 0] + '%f\t' % mon_B_write[w, 1] +
                    '%f\t' % mon_B_write[w, 2] + '%f\t' % mon_B_write[w, 3] + '\n')
            f.write('\n')
            f.write(f'*CFOUR(CALC=HF,BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
                        '\nCOORDINATES=CARTESIAN\nFROZEN_CORE=ON,SCF_MAXC=0'
                        '\nMEMORY=50,MEM_UNIT=GB,PRINT=20)')
            f.write('\n \n \n \n \n')
            f.close()
    else:
        sys.exit('wrong monomer')
elif calc_type == 'CT' or calc_type == 'Charge Transfer':#TODO CT and mix
    if which_input == 'mon_A':
        mon_A_write = add_atoms(mon_A_matrix, mon_A_name, 'normal')
#        with open('ZMAT.vee', 'w+') as f:
#            f.write(f'{mon_A_name} monomer calc, used for coupling LCAO\n')
#            #át
#            for u in range(0, len(mon_A_write)):
#                f.write('%s ' % mon_A_write[u, 0] + '%f\t' % mon_A_write[u, 1] +
#                    '%f\t' % mon_A_write[u, 2] + '%f\t' % mon_A_write[u, 3] + '\n')
#            f.write('\n')
#            f.write(f'*CFOUR(CALC={calc_method},BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
#                    '\nCOORDINATES=CARTESIAN,EXCITE=CIS\nESTATE_SYM=5\nFROZEN_CORE=ON'
#                    '\nMEMORY=50,MEM_UNIT=GB)')
#            f.write('\n \n \n \n \n')
#            f.close()
        with open('ZMAT.IP', 'w+') as f:
            f.write(f'{mon_A_name} monomer calc IP, used for coupling \n')
            for u in range(0, len(mon_A_write)):
                f.write('%s ' % mon_A_write[u, 0] + '%f\t' % mon_A_write[u, 1] +
                    '%f\t' % mon_A_write[u, 2] + '%f\t' % mon_A_write[u, 3] + '\n')
            excite = extract_block_from_header_until_blank("IP exc mon_A", "geom_input")
            mon_method = extract_block_from_header_until_blank("mon calc method", "geom_input")   
            excite_method='EOMEE'
            excitation_spec = ',ESTATE_PROP=1,EOMFOLLOW=OVERLAP\nEOM_NSTATE=MULTIROOT'
            state_num = 50
            if 'CIS' in mon_method:
                excite_method=mon_method
                excitation_spec = ''
                state_num = 50
            f.write('\n')
            f.write(f'*CFOUR(CALC={mon_method},BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
                    f'\nCOORDINATES=CARTESIAN,EXCITE={excite_method}{excitation_spec}\nESTATE_SYM={state_num}\nFROZEN_CORE=ON,CONTINUUM=VIRTUAL'
                    f'\nMEMORY=50,MEM_UNIT=GB,PRINT=20)')
            f.write(f'\n\n{excite}\n \n \n \n')
            f.close()
        with open('ZMAT.EA', 'w+') as f:
            f.write(f'{mon_A_name} monomer calc EA, used for coupling \n')
            for u in range(0, len(mon_A_write)):
                f.write('%s ' % mon_A_write[u, 0] + '%f\t' % mon_A_write[u, 1] +
                    '%f\t' % mon_A_write[u, 2] + '%f\t' % mon_A_write[u, 3] + '\n')
            excite = extract_block_from_header_until_blank("EA exc mon_A", "geom_input")
            mon_method = extract_block_from_header_until_blank("mon calc method", "geom_input") 
            excitation_spec = ',ESTATE_PROP=1,EOMFOLLOW=OVERLAP\nEOM_NSTATE=MULTIROOT'
            excite_method='EOMEE'
            state_num = 50
            if 'CIS' in mon_method:
                excite_method=mon_method
                excitation_spec = ''
                state_num = 50
            f.write('\n')
            f.write(f'*CFOUR(CALC={mon_method},BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
                    f'\nCOORDINATES=CARTESIAN,EXCITE={excite_method}{excitation_spec}\nESTATE_SYM={state_num}\nFROZEN_CORE=ON,CONTINUUM=OCCUPIED,CHARGE=-2'
                    f'\nMEMORY=50,MEM_UNIT=GB,PRINT=20)')
            f.write(f'\n\n{excite}\n \n \n \n')
            f.close()
    elif which_input == 'mon_B':
        mon_B_write = add_atoms(mon_B_matrix, mon_B_name, 'normal')
#        with open('ZMAT', 'w+') as f:
#            f.write(f'{mon_B_name} monomer calc, used for coupling \n')
#            mon_B_write = add_atoms(mon_B_matrix, mon_B_name, 'normal')# pyrrolehoz:
#            for w in range(0, len(mon_B_write)):
#                f.write('%s ' % mon_B_write[w, 0] + '%f\t' % mon_B_write[w, 1] +
#                    '%f\t' % mon_B_write[w, 2] + '%f\t' % mon_B_write[w, 3] + '\n')
#            f.write('\n')
#            f.write(f'*CFOUR(CALC={calc_method},BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
#                    '\nCOORDINATES=CARTESIAN,EXCITE=CIS\nESTATE_SYM=10\nFROZEN_CORE=ON'
#                    '\nMEMORY=50,MEM_UNIT=GB)')
#            f.write('\n \n \n \n \n')
#            f.close()
        with open('ZMAT.IP', 'w+') as f:
            f.write(f'{mon_B_name} monomer calc, used for coupling \n')
            for w in range(0, len(mon_B_write)):
                f.write('%s ' % mon_B_write[w, 0] + '%f\t' % mon_B_write[w, 1] +
                    '%f\t' % mon_B_write[w, 2] + '%f\t' % mon_B_write[w, 3] + '\n')
            excite = extract_block_from_header_until_blank("IP exc mon_B", "geom_input")
            mon_method = extract_block_from_header_until_blank("mon calc method", "geom_input")   
            excitation_spec = ',ESTATE_PROP=1,EOMFOLLOW=OVERLAP\nEOM_NSTATE=MULTIROOT' 
            excite_method='EOMEE'
            state_num = 50
            if 'CIS' in mon_method:
                excite_method=mon_method
                excitation_spec = ''
                state_num = 50
            f.write('\n')
            f.write(f'*CFOUR(CALC={mon_method},BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
                    f'\nCOORDINATES=CARTESIAN,EXCITE={excite_method}{excitation_spec}\nESTATE_SYM={state_num}\nFROZEN_CORE=ON,CONTINUUM=VIRTUAL'
                    f'\nMEMORY=50,MEM_UNIT=GB,PRINT=20)')
            f.write(f'\n\n{excite}\n \n \n \n')
            f.close()
        with open('ZMAT.EA', 'w+') as f:
            f.write(f'{mon_B_name} monomer calc, used for coupling \n')
            for w in range(0, len(mon_B_write)):
                f.write('%s ' % mon_B_write[w, 0] + '%f\t' % mon_B_write[w, 1] +
                    '%f\t' % mon_B_write[w, 2] + '%f\t' % mon_B_write[w, 3] + '\n')
            excite = extract_block_from_header_until_blank("EA exc mon_B", "geom_input")
            mon_method = extract_block_from_header_until_blank("mon calc method", "geom_input") 
            excitation_spec = ',ESTATE_PROP=1,EOMFOLLOW=OVERLAP\nEOM_NSTATE=MULTIROOT'
            excite_method='EOMEE'
            state_num = 50
            if 'CIS' in mon_method:
                excite_method=mon_method
                excitation_spec = ''
                state_num = 50
            f.write('\n')
            f.write(f'*CFOUR(CALC={mon_method},BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
                    f'\nCOORDINATES=CARTESIAN,EXCITE={excite_method}{excitation_spec}\nESTATE_SYM={state_num}\nFROZEN_CORE=ON,CONTINUUM=OCCUPIED,CHARGE=-2'
                    f'\nMEMORY=50,MEM_UNIT=GB,PRINT=20)')
            f.write(f'\n\n{excite}\n \n \n \n')
            f.close()
    elif which_input == 'dimer':
        with open('ZMAT', 'w+') as f:
            #f.write('%s-' % mol + '%s ' % mol + '%s A ' % v_str + '%s' % calc_type + '\n')
            f.write('cyt-cyt dist  rot %s \n' % v_str)
            mon_A_write = add_atoms(mon_A_matrix, mon_A_name, 'normal')#át
            #mon2_write = add_atoms(mon2, 'pyr', 'normal')#át ha másik dimerkutfuvztt akarsz
            mon_B_write = add_atoms(mon_B_matrix, mon_B_name, 'normal')# pyrrolehoz:
            for u in range(0, len(mon_A_write)):
                f.write('%s ' % mon_A_write[u, 0] + '%f\t' % mon_A_write[u, 1] +
                    '%f\t' % mon_A_write[u, 2] + '%f\t' % mon_A_write[u, 3] + '\n')
            for w in range(0, len(mon_B_write)):
                f.write('%s ' % mon_B_write[w, 0] + '%f\t' % mon_B_write[w, 1] +
                    '%f\t' % mon_B_write[w, 2] + '%f\t' % mon_B_write[w, 3] + '\n')
            f.write('\n')
            f.write(f'*CFOUR(CALC=HF,BASIS={basis}\nUNIT=BOHR,SYMMETRY=OFF,FIXGEOM=ON'
                        '\nCOORDINATES=CARTESIAN\nFROZEN_CORE=ON'
                        '\nMEMORY=50,MEM_UNIT=GB,PRINT=20)')
            f.write('\n \n \n \n \n')
            f.close()
    else:
        sys.exit('wrong monomer')
else:
    sys.exit('Not ready')
    mon1_write = add_atoms(mon1, "forma", 'normal')#át
    mon2_write = add_atoms(mon2, "forma", 'normal')#át
     #ezt majd űgy akarom megoldani, hogy sorrenben legyenek a töltések és simán hozzáírja
    f = open('ZMAT.%i.qmmm' % counter, 'w+')
    f.write('forma-forma dist %s' % v_str + "A aTZ charges aTZ\n")
    #f.write('%s-' % mol + '%s b as charges' % mol + '%s A ' % v_str + '%s' % calc_type + '\n')
    #f.write('#cyt with ura as charges %s A ' % v_str + '%s' % calc_type + 'ChelpG\n')
    point_charge = e.parameterfrominput(input_matrix, "point charges forma aTZ CHELPG", "float")#ezt át kell írni
    mon2_write_del = np.delete(mon2_write, 0, 1)
    mon2_charged = LJ_matrix = [0., 0.,  0., 0.]
    for z in range(0, len(point_charge)):
        seged = np.append(mon2_write_del[z, :], point_charge[z])
        mon2_charged = np.vstack([mon2_charged, seged.copy()])
    mon2_charged = np.delete(mon2_charged, 0, 0)
    for u in range(0, len(mon1_write)):
        f.write('%s ' % mon1_write[u, 0] + '%f\t' % mon1_write[u, 1] +
            '%f\t' % mon1_write[u, 2] + '%f\t' % mon1_write[u, 3] + '\n')
    f.write('\n')
    num_of_atoms = 4  # pyrrolenál ez nem kell
#inntöl            f.write('basis=aug-cc-pVDZ\ncalc=CCSD\nmem=100GB\nqmmm=AMBER\ndfbasis_scf=none\ndfbasis_cor=none'
 #           '\nunit=bohr\n\ngeom=xyz\n%s\n' % num_of_atoms)
  #  for u in range(0, len(mon1_write)):
   #     f.write('%s ' % mon1_write[u, 0] + '%f\t' % mon1_write[u, 1] +
    #        '%f\t' % mon1_write[u, 2] + '%f\t' % mon1_write[u, 3] + '\n')
#edg MRCC            f.write('\npointcharges\n10\n\n')
    f.write('*CFOUR(CALC=CCSD,BASIS=AUG-PVTZ\nUNIT=BOHR,EXCITE=EOMEE,EOM_NSTATE=MULTIROOT'
            '\nCOORDINATES=CARTESIAN,ESTATE_SYM=10 \n'
            'FROZEN_CORE=ON,ESTATE_PROP=1\nEXTERN_POT=ON\n'
            'MEM=50,MEM_UNIT=GB)\n\n%extern_pot*\n' + '%s' % num_of_atoms + '\n')
  #        f.write('*CFOUR(CALC=CC2,BASIS=PVDZ\nUNIT=BOHR,EXCITE=EOMEE\nEOMFOLLOW=OVERLAP,EOM_NSTATE=MULTIROOT'
#                   '\nCOORDINATES=CARTESIAN,ESTATE_SYM=6\n'
#               'FROZEN_CORE=ON,ESTATE_PROP=1\nEXTERN_POT=ON\n'
 #              'MEM=80,MEM_UNIT=GB)\n\n%extern_pot*\n' + '%s' % num_of_atoms + '\n')
    for w in range(0, len(mon2_charged)):
        f.write('%f   ' % mon2_charged[w, 0] + '%f   ' % mon2_charged[w, 1] +
            '%f    ' % mon2_charged[w, 2] + '%f' % mon2_charged[w, 3] + '\n')
    #f.write('\n%excite*\n5\n1 \n1 18 0 19 0 1.0\n1 \n1 18 0 20 0 1.0 \n1\n1 18 0 21 0 1.0\n1\n'
     #       '1 17 0 19 0 1.0\n1\n1 18 0 22 0 1.0    ')
    f.write('\n \n \n \n \n')
    f.close()
    f = open('ZMAT.%i.b' % counter, 'w+')
    f.write('%s-' % mol + '%s a as charges' % mol + '%s A ' % v_str + '%s' % calc_type + '\n')
    #f.write('#ura with cyt as charges %s A ' % v_str + '%s' % calc_type + '\n')
    point_charge = e.parameterfrominput(input_matrix, "point charges forma aDZ CHELPG", "float")#át
    mon1_write_del = np.delete(mon1_write, 0, 1)
    mon1_charged = LJ_matrix = [0., 0.,  0., 0.]
    for z in range(0, len(point_charge)):
        seged = np.append(mon1_write_del[z, :], point_charge[z])
        mon1_charged = np.vstack([mon1_charged, seged.copy()])
#inntöl            f.write('basis=aug-cc-pVDZ\ncalc=CCSD\nmem=100GB\nqmmm=AMBER\ndfbasis_scf=none\ndfbasis_cor=none'
 #           '\nunit=bohr\n\ngeom=xyz\n%s\n' % num_of_atoms)
    mon1_charged = np.delete(mon1_charged, 0, 0)
    for u in range(0, len(mon2_write)):
        f.write('%s ' % mon2_write[u, 0] + '%f\t' % mon2_write[u, 1] +
            '%f\t' % mon2_write[u, 2] + '%f\t' % mon2_write[u, 3] + '\n')
    f.write('\n')
    #num_of_atoms = 13 #pyrrolenál ez nem kell
    f.write('*CFOUR(CALC=CCSD,BASIS=PVDZ\nUNIT=BOHR,EXCITE=EOMEE\nEOM_NSTATE=MULTIROOT'
            '\nCOORDINATES=CARTESIAN,ESTATE_SYM=6\n'
            'FROZEN_CORE=ON,ESTATE_PROP=1\nEXTERN_POT=ON\n'
            'MEM=100,MEM_UNIT=GB)\n\n%extern_pot*\n' + '%s' % num_of_atoms + '\n')
  #  f.write('*CFOUR(CALC=CCSD,BASIS=AUG-PVDZ\nUNIT=BOHR\nABCDTYPE=AOBASIS\nCOORDINATES=CARTESIAN\nFROZEN_CORE=ON\nEXTERN_POT=ON\n'
    #        'MEM=100,MEM_UNIT=GB\nCC_PROG=NCC)\n\n%extern_pot*\n' + '%s' % num_of_atoms + '\n')
#eys mrcc            f.write('\npointcharges\n10\n')
    for w in range(0, len(mon1_charged)):
        f.write('%f   ' % mon1_charged[w, 0] + '%f   ' % mon1_charged[w, 1] +
            '%f    ' % mon1_charged[w, 2] + '%f' % mon1_charged[w, 3] + '\n')
    f.write('\n \n \n \n \n')
    f.close()
