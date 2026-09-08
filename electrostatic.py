import numpy as np
import re
from numpy.linalg import inv
import pandas as pd
import os.path
from scipy.spatial.transform import Rotation as R


def bohrtosangstrom(bohr):
   angstrom = float(bohr) * float(0.529177249)
   return angstrom


def angstromtobohr(angstrom):
    bohr = angstrom*(1. / 0.529177249)
    return bohr


def convert(string):
    li = list(string.split("   "))
    return li


def convert2(string):
    li = list(string.split("  "))
    return li



def readinput(input):
    input_matrix = ["0"]
    for row in open(input, "r"):
        new_line = row.rstrip('\n')
        input_matrix.append(str(new_line))
    input_matrix.pop(0)
    return input_matrix


def gridfrominput(matrix, parameter, type1):
    out = []
    counter = -1
    for row in matrix:
        counter += 1
        if row == parameter:
            is_geom = True
        else:
            is_geom = False
        if is_geom:
            atom_number = int(matrix[counter + 1])
            start = int(counter + 2)
            for i in range(start, start + atom_number):
                opk = matrix[i]
                opk = re.sub(r"\s+", "", opk, flags=re.UNICODE)
#                betettem a file ba egy számot, ennyi atom lesz, ezt bele kell írni!!!
                row_list = list((opk.split(',')))
                converted_list = []
                if type1 == "float":
#                    row_list.pop(0)
                    for element in row_list:
                        converted_row = float(element)
                        converted_list.append(converted_row)
                elif type1 == "string":
                    converted_list = row_list
                out.append(converted_list)
                if opk == " ":
                    break
    out = np.array(out)
    return out



def parameterfrominput(matrix, parameter, type1):
    out = []
    counter = -1
    for row in matrix:
        counter += 1
        if row == parameter:
            is_geom = True
        else:
            is_geom = False
        if is_geom:
            atom_number = int(matrix[counter + 1])
            start = int(counter + 2)
            for i in range(start, start + atom_number):
                opk = matrix[i]
                opk = re.sub(r"\s+", "", opk, flags=re.UNICODE)
#                betettem a file ba egy számot, ennyi atom lesz, ezt bele kell írni!!!
                row_list = list((opk.split(',')))
                converted_list = []
                if type1 == "float":
                    row_list.pop(0)
                    for element in row_list:
                        converted_row = float(element)
                        converted_list.append(converted_row)
                elif type1 == "string":
                    converted_list = row_list
                out.append(converted_list)
                if opk == " ":
                    break
    return out


def dipolefrominput(matrix, moment_name):
    counter = -1
    for row in matrix:
        counter += 1
        if row == moment_name:
            is_geom = True
        else:
            is_geom = False
        if is_geom:
            atom_number = int(matrix[counter + 1])
            start = int(counter + 2)
            for i in range(start, start + atom_number):
                opk = matrix[i]
                opk = re.sub(r"\s+", "", opk, flags=re.UNICODE)
#                betettem a file ba egy számot, ennyi atom lesz, ezt bele kell írni!!!
                row_list = list((opk.split(',')))
                converted_list = []
                for element in row_list:
                    converted_row = float(element)
                    converted_list.append(converted_row)
                dipole = converted_list
                if opk == " ":
                    break
    return dipole


def quadrupolefrominput(matrix, moment_name):
    quadrupole = []
    counter = -1
    for row in matrix:
        counter += 1
        if row == moment_name:
            is_geom = True
        else:
            is_geom = False
        if is_geom:
            atom_number = int(matrix[counter + 1])
            start = int(counter + 2)
            for i in range(start, start + atom_number):
                opk = matrix[i]
                opk = re.sub(r"\s+", "", opk, flags=re.UNICODE)
#                betettem a file ba egy számot, ennyi atom lesz, ezt bele kell írni!!!
                row_list = list((opk.split(',')))
                converted_list = []
                for element in row_list:
                    converted_row = float(element)
                    converted_list.append(converted_row)
                quadrupole.append(converted_list)
                if opk == " ":
                    break
    quadrupole = np.array(quadrupole)
    return quadrupole


def converttofloat(matrix):
    float_matrix = matrix.astype(np.float)
    return float_matrix


def converttostring(matrix):
    string_matrix = matrix.astype(np.string)
    return string_matrix

#csekkold le a SciPy rotationt, mielőtt bármi mást csinálsz!!!
#alpha Z, beta az Y, gamma az X tengely körül forgat
def rotation(matrix, alpha, beta, gamma, moment_type):
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    rotationmatrix = [
            np.cos(alpha) * np.cos(beta), (np.cos(alpha) * np.sin(beta) * np.sin(gamma)) - (np.sin(alpha) * np.cos(gamma)), (np.cos(alpha) * np.sin(beta) * np.cos(gamma)) + (np.sin(alpha) * np.sin(gamma))
        ], [
            np.sin(alpha) * np.cos(beta), (np.sin(alpha) * np.sin(beta) * np.sin(gamma)) + (np.cos(alpha) * np.cos(gamma)), (np.sin(alpha) * np.sin(beta) * np.cos(gamma)) - (np.cos(alpha) * np.sin(gamma))
        ], [
            -np.sin(beta), np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma)
        ]
    if str(moment_type) == "Dipole":
        matrix = np.array(matrix)
        matrix2 = matrix.transpose()
        rotationmatrix = np.array(rotationmatrix)
        rotated_moment = np.matmul(rotationmatrix, matrix2)
        rotated_moment = rotated_moment.transpose()
   #     print(rotated_moment)
    if str(moment_type) == "Quadrupole":
        inv_rot = inv(rotationmatrix)
   #     print(matrix)
        matrix2 = matrix.transpose() #no need for this, cuz its symmetric
        afk = np.matmul(rotationmatrix, matrix2)
        #afk2 = afk.transpose()
        rotated_moment = np.matmul(afk, inv_rot)
    return rotated_moment

#külön kell a kettőt megadni, mert a távolságok picit különböznek
def chargeinteraction(charges1, charges2, num_of_atoms):#megváltoztatva, a határok
    U_charge = 0.
    U_2 = 0.
 #   print(charges1)
#    print(charges2)
    num_of_atoms = int(num_of_atoms)
  #  print(len(charges1))
    for i in range(0, len(charges1)):
        szam = 0.
  #      print(szam)
        for j in range(0, len(charges2)):
            dist = np.sqrt(np.square(charges1[i, 0] - charges2[j, 0]) + np.square(charges1[i, 1] - charges2[j, 1]) +
                          np.square(charges1[i, 2] - charges2[j, 2]))
 #           if j < i:
            U_charge = U_charge + ((charges1[i, 3] * charges2[j, 3]) / dist)
            #print(U_charge)
     #       if i != j:
     #       szam = szam + (charges2[j, 3] / dist)
 #               print(szam)
     #   U_2 = U_2 + charges1[i, 3] * szam
#        print(U_2)
    #        #U_2 = U_2
  #  print(U_2)
  #  print(U_charge)
    return U_charge


def dipoleinteraction(dipole1, dipole2, distance, measure, axis):#nem teljesen ok, írd át, a tolások nem jók
    if axis == "z":
        shift_vector1 = np.array([0., 0., 1.])
    elif axis == "y":
        shift_vector1 = np.array([0., 1., 0.])
    elif axis == "x":
        shift_vector1 = np.array([1., 0., 0.])
#    if orientation == "opposite":#elforgatva
#        new_dipole2 = np.array([-dipole2[0], dipole2[1], dipole2[2]])
#    else:#legyen konzisztens a kvadrupóllal
#        new_dipole2 = np.array([dipole2[0], dipole2[1], dipole2[2]])
    if measure == "Angstrom":
        new_distance = distance
    else:
        new_distance = angstromtobohr(distance)
    dipole1 = np.array(dipole1[0])
    dipole2 = np.array(dipole2[0])
 #   print(dipole1)
 #   print(dipole2)
    v1 = np.dot(dipole1, dipole2)
    v2 = 3 * np.dot(shift_vector1, dipole2) * np.dot(shift_vector1, dipole1)
#    print(v1)
#    print(v2)
    V = (np.dot(dipole1, dipole2) - 3 * np.dot(shift_vector1, dipole2) *
         np.dot(shift_vector1, dipole1))/(new_distance ** 3)
    return V


def quadrupoleinteraction(quadrupole1, quadrupole2, distance, measure, axis):
    if measure == "Angstrom":
        new_distance = distance
    else:
        new_distance = angstromtobohr(distance)
  #  print(new_distance)
    if axis == "z":
        shift_vector = np.array([0., 0., 1.])
    elif axis == "y":
        shift_vector = np.array([0., 1., 0.])
    elif axis == "x":
        shift_vector = np.array([1., 0., 0.])
    V_belso = 0.
    V_kozepso = 0.
    V_kulso = 0.
    for i in range(0, len(quadrupole1)):
        for j in range(0, len(quadrupole1)):
            for k in range(0, len(quadrupole2)):
                for l in range(0, len(quadrupole2)):
                    V_belso = V_belso + (shift_vector[i] * quadrupole1[i, j] * shift_vector[j]) * (
                                shift_vector[k] * quadrupole2[k, l] * shift_vector[l])
    for i in range(0, len(quadrupole1)):
        for j in range(0, len(quadrupole1)):
            for k in range(0, len(quadrupole2)):
                V_kozepso = V_kozepso + (shift_vector[i] * quadrupole1[i, j] * quadrupole2[j, k] * shift_vector[k])
    for i in range(0, len(quadrupole1)):
        for j in range(0, len(quadrupole1)):
            V_kulso = V_kulso + (quadrupole1[i, j] * quadrupole2[i, j])
    V_kulso = ((2. / 3.)) * V_kulso - ((20. / 3.) * V_kozepso) + ((35. / 3.) * V_belso)
    V_final = V_kulso / (new_distance ** 5)
    return V_final


def dip_article(dipole1, dipole2, distance, measure, axis):
    shift_vector = np.array([0., 0., 0.])
    if measure == "Angstrom":
        new_distance = distance
    else:
        new_distance = angstromtobohr(distance)
    #   print(new_distance)
    if axis == "z":
        shift_vector = np.array([0., 0., 1.])
    elif axis == "y":
        shift_vector = np.array([0., 1., 0.])
    elif axis == "x":
        shift_vector = np.array([1., 0., 0.])
  #  dipole2 = np.array([dipole2])
 #   print(dipole1)
  #  print(dipole2)
    n_ab = shift_vector
    n_ba = -n_ab
    first_element = np.dot(dipole1, dipole2)
    second_element = 3 * np.dot(n_ab, dipole1) * np.dot(dipole2, n_ba)
    V_dip = (first_element + second_element) / (new_distance ** 3)
    return V_dip


def quad_from_article(quadrupole1, quadrupole2, distance, measure, axis):
    shift_vector = np.array([0., 0., 0.])
    if measure == "Angstrom":
        new_distance = distance
    else:
        new_distance = angstromtobohr(distance)
 #   print(new_distance)
    if axis == "z":
        shift_vector = np.array([0., 0., 1.])
    elif axis == "y":
        shift_vector = np.array([0., 1., 0.])
    elif axis == "x":
        shift_vector = np.array([1., 0., 0.])
    n_ab = shift_vector
    n_ba = -n_ab
    n2_ab = np.outer(n_ab.transpose(), n_ab)
    n2_ba = np.outer(n_ba.transpose(), n_ba)
    first_element = 2 * two_dot_product(quadrupole1, quadrupole2)
    second_element = 20 * np.dot(np.matmul(n_ab, quadrupole1), np.matmul(n_ba, quadrupole2))
    third_element = 35 * two_dot_product(n2_ab, quadrupole1) * two_dot_product(n2_ba, quadrupole2)
    V_quad_quad = (first_element + second_element + third_element) / (3 * (new_distance ** 5))
    return V_quad_quad


#át lehet írni az egészet dottal

def rank_three_product(vector, rank_3_tensor):
    k = np.matmul(vector, np.matmul(vector, np.matmul(vector, rank_3_tensor)))
    return k

def two_dot_product_v_o(vector, rank_3_tensor):
    p = np.matmul(vector, np.matmul(vector, rank_3_tensor))
    return p

def three_dot_product(rank_3_tensor1, rank_3_tensor2):
    valami = 0.
    for i in range(len(rank_3_tensor2)):
        for j in range(len(rank_3_tensor2)):
            for k in range(len(rank_3_tensor2)):
                valami = valami + rank_3_tensor1[i, j, k] * rank_3_tensor2[i, j, k]
    return valami

def two_dot_product(matrix1, matrix2):
    ered = 0.
    for i in range(len(matrix1)):
        for j in range(len(matrix2)):
            ered = ered + matrix1[i, j] * matrix2[i, j]
    return ered

def dip_quad_int(dipole_a, dipole_b, quadrupole_a, quadrupole_b, distance, measure, axis):
    shift_vector = np.zeros((1, 3))
    #itt jegyezd meg, hogy egyébként nem kell ugyanannak lenni a két kvadrupólnak, így írd , ugyanez a másikra, plusz
    # egyébként nem is egyezik meg mert ellentétes, szal így folytasd
    if measure == "Angstrom":
        new_distance = distance
    else:
        new_distance = angstromtobohr(distance)
    if axis == "z":
        shift_vector = np.array([0., 0., 1.])
    elif axis == "y":
        shift_vector = np.array([0., 1., 0.])
    elif axis == "x":
        shift_vector = np.array([1., 0., 0.])
    n_ab = shift_vector
    n_ba = -n_ab
    #print(n_ba)
    n2_ab = np.outer(n_ab.copy(), n_ab.copy())
    n2_ba = np.outer(n_ba.copy(), n_ba.copy())
  #  print(n2_ba)
    first_element = 2 * np.dot(dipole_a, quadrupole_b.dot(n_ba))
  #  print(quadrupole_b.dot(n_ba))
  #  print(dipole_a)
   # print(np.matmul(quadrupole_b, n_ba))
  #  print(first_element)
    second_element = 2 * np.dot(quadrupole_a.dot(n_ab), dipole_b)
  #  print(quadrupole_a.dot(n_ab))
  #  print(dipole_b)
  #  print(second_element)
    third_element = 5 * np.dot(n_ab, dipole_a) * two_dot_product(quadrupole_b, n2_ba)
  #  print(third_element)
    fourth_element = 5 * two_dot_product(quadrupole_a, n2_ab) * np.dot(dipole_b, n_ba)
  #  print(fourth_element)
    V_dip_quad = (first_element + second_element + third_element + fourth_element) / (new_distance ** 4)
    return V_dip_quad

def r5_nem_quad_quad(dipole_a, dipole_b, octupole_a, octupole_b, distance, measure, axis):
    shift_vector = np.zeros((1, 3))
    # itt jegyezd meg, hogy egyébként nem kell ugyanannak lenni a két kvadrupólnak, így írd , ugyanez a másikra, plusz
    # egyébként nem is egyezik meg mert ellentétes, szal így folytasd
    if measure == "Angstrom":
        new_distance = distance
    else:
        new_distance = angstromtobohr(distance)
    if axis == "z":
        shift_vector = np.array([0., 0., 1.])
    elif axis == "y":
        shift_vector = np.array([0., 1., 0.])
    elif axis == "x":
        shift_vector = np.array([1., 0., 0.])
    n_ab = shift_vector
    n_ba = -n_ab
    n2_ab = np.outer(n_ab, n_ab)
    n2_ba = np.outer(n_ba, n_ba)
 #   print(n2_ba)
    n3_ab = np.multiply.outer(n_ab, n2_ab)
    n3_ba = np.multiply.outer(n_ba, n2_ba)
    fourth_element = 9 * np.dot(two_dot_product(n2_ab, octupole_a), dipole_b)
 #   print(fourth_element)
    fifth_element = 9 * np.dot(dipole_a, two_dot_product(n2_ba, octupole_b))
  #  print(fifth_element)
    sixth_element = 21 * rank_three_product(n_ab, octupole_a) * np.dot(dipole_b, n_ba)
   # print(sixth_element)
    seventh_element = 21 * rank_three_product(n_ba, octupole_b) * np.dot(dipole_a, n_ab)
    #print(seventh_element)
    V_r5 = (fourth_element + fifth_element + sixth_element + seventh_element) / (3 * (new_distance ** 5))
    return V_r5

def oct_quad(quadrupole_a, quadrupole_b, octupole_a, octupole_b, distance, measure, axis):
    shift_vector = np.zeros((1, 3))
    # itt jegyezd meg, hogy egyébként nem kell ugyanannak lenni a két kvadrupólnak, így írd , ugyanez a másikra, plusz
    # egyébként nem is egyezik meg mert ellentétes, szal így folytasd
    if measure == "Angstrom":
        new_distance = distance
    else:
        new_distance = angstromtobohr(distance)
    if axis == "z":
        shift_vector = np.array([0., 0., 1.])
    elif axis == "y":
        shift_vector = np.array([0., 1., 0.])
    elif axis == "x":
        shift_vector = np.array([1., 0., 0.])
    n_ab = shift_vector
    n_ba = -n_ab
    n2_ab = np.outer(n_ab, n_ab)
    n2_ba = np.outer(n_ba, n_ba)
    first_element = 2 * two_dot_product(np.matmul(n_ab, octupole_a), quadrupole_b)
   # print(first_element)
    second_element = 2 * two_dot_product(quadrupole_a, np.matmul(octupole_b, n_ba))
   # print(second_element)
    third_element = 14 * np.dot(np.matmul(n_ab, quadrupole_a), two_dot_product_v_o(n_ba, octupole_b))
   # print(third_element)
    fourth_element = 14 * np.dot(two_dot_product_v_o(n_ab, octupole_a), np.matmul(quadrupole_b, n_ba))
   # print(fourth_element)
    fifth_element = 21 * rank_three_product(n_ab, octupole_a) * two_dot_product(quadrupole_b, n2_ba)
   # print(fifth_element)
    sixth_element = 21 * two_dot_product(quadrupole_a, n2_ab) * rank_three_product(n_ba, octupole_b)
   # print(sixth_element)
    V_quad_oct = (first_element + second_element + third_element + fourth_element + fifth_element
                  + sixth_element) / (new_distance ** 6)
    return V_quad_oct

def oct_oct_interaction(octupole_a, octupole_b, distance, measure, axis):
    shift_vector = np.zeros((1, 3))
    # itt jegyezd meg, hogy egyébként nem kell ugyanannak lenni a két kvadrupólnak, így írd , ugyanez a másikra, plusz
    # egyébként nem is egyezik meg mert ellentétes, szal így folytasd
    if measure == "Angstrom":
        new_distance = distance
    else:
        new_distance = angstromtobohr(distance)
    if axis == "z":
        shift_vector = np.array([0., 0., 1.])
    elif axis == "y":
        shift_vector = np.array([0., 1., 0.])
    elif axis == "x":
        shift_vector = np.array([1., 0., 0.])
    n_ab = shift_vector
    n_ba = -n_ab
    n2_ab = np.outer(n_ab, n_ab)
    n2_ba = np.outer(n_ba, n_ba)
    first_element = 2 * three_dot_product(octupole_a, octupole_b)
    second_element = 42 * two_dot_product(np.matmul(n_ab, octupole_a), np.matmul(octupole_b, n_ba))
    third_element = 126 * np.dot(two_dot_product_v_o(n_ab, octupole_a), two_dot_product_v_o(n_ba, octupole_b))
    fourth_element = 231 * rank_three_product(n_ab, octupole_a) * rank_three_product(n_ba, octupole_b)
    V_oct_oct = (first_element + second_element + third_element + fourth_element) / (5 * (new_distance ** 7))
    return V_oct_oct


def addcharge(molecule, charges):
    LJ_matrix = [0, 0,  0, 0, 0]
    MOLE = np.array(molecule)
    CHA = np.array(charges)
    for row in MOLE:
        xd = row
        print(xd)
        for line in CHA:
            kek = line
            if xd[0] == kek[0]:
                kek = np.delete(kek, 0, 0)
                top = np.append(xd, kek, 0)
                LJ_matrix = np.vstack([LJ_matrix, top])
                print(LJ_matrix)
    print(LJ_matrix)
    LJ_matrix = np.delete(LJ_matrix, 0, 0)
    print(LJ_matrix)
    LJ_matrix = np.delete(LJ_matrix, 0, 1)
    return LJ_matrix

#az átváltásokkal majd még szórakozhats, ez itt egyenlőre bohrban lesz
def setdistance(molecule1, distance, axis):
    new_distance = angstromtobohr(distance)
 # print(new_distance)
    if axis == "x":
        molecule1[:, 0] = molecule1[:, 0] + new_distance
    elif axis == "y":
        molecule1[:, 1] = molecule1[:, 1] + new_distance
    elif axis == "z":
        molecule1[:, 2] = molecule1[:, 2] + new_distance
    else:
        print("Wrong axis")
    return molecule1


def quadrupolemoment(charges, num_atoms):
    Q = np.zeros((3, 3), float)
    kroenecker_delta = np.identity(3)
    for i in range(0, 3):
        for j in range(0, 3):
            for l in range(0, int(num_atoms)):
                r_l = np.sqrt(np.square(charges[l, 0]) + np.square(charges[l, 1]) +
                    np.square(charges[l, 2]))
                Q[i, j] = Q[i, j] + (charges[l, 3] * ((3 * charges[l, i] * charges[l, j]) - (np.square(r_l) * kroenecker_delta[i, j])))
    Q = Q/2
    return Q


def dipmom(charges, num_atoms):
    d = np.zeros((1, 3), float)
#    print(d)
    for i in range(0, num_atoms):
        d = d + (charges[i, 3] * np.array([charges[i, 0], charges[i, 1], charges[i, 2]]))
    return d


def quadmom2(charges, num_atoms):
    Q = np.zeros((3, 3), float)
    for i in range(0, num_atoms):
        Q[0, 0] = Q[0, 0] + (charges[i, 3] * np.square(charges[i, 0]))
    for i in range(0, num_atoms):
        Q[1, 1] = Q[1, 1] + (charges[i, 3] * np.square(charges[i, 1]))
    for i in range(0, num_atoms):
        Q[2, 2] = Q[2, 2] + (charges[i, 3] * np.square(charges[i, 2]))
    for i in range(0, num_atoms):
        Q[0, 1] = Q[0, 1] + (charges[i, 3] * charges[i, 0] * charges[i, 1])
    for i in range(0, num_atoms):
        Q[0, 2] = Q[0, 2] + (charges[i, 3] * charges[i, 0] * charges[i, 2])
    for i in range(0, num_atoms):
        Q[1, 2] = Q[1, 2] + (charges[i, 3] * charges[i, 1] * charges[i, 2])
    Q[1, 0] = Q[0, 1]
    Q[2, 0] = Q[0, 2]
    Q[2, 1] = Q[1, 2]
    return Q

def oct_mom(charges, num_atoms):
    octupole = np.zeros((3, 3, 3), dtype=np.float64)
 #   print(octupole)
    kroenecker_delta = np.identity(3)
    print(charges)
  #  identity = np.array([[[1., 0., 0.],
   #   [0., 1., 0],
  #   [0., 0., 1.]],
  #  [[1., 0., 0.],
  #   [0., 1., 0.],
   # [0., 0., 1.]],
   # [[1., 0., 0],
   #  [0., 1., 0.],
   # [0., 0., 1.]]])
    for l in range(0, int(num_atoms)):
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    r_l = np.sqrt(np.square(charges[l, 0]) + np.square(charges[l, 1]) + np.square(charges[l, 2]))
                    octupole[i, j, k] = octupole[i, j, k] + (charges[l, 3] * ((15 * charges[l, i] * charges[l, j] * charges[l, k]) -
                                                                              3 * (charges[l, i] * kroenecker_delta[j, k] +
                                                                     charges[l, j] * kroenecker_delta[i, k] +
                                                                     charges[l, k] * kroenecker_delta[i, j]) *
                                                                np.square(r_l)))
                    #print(octupole)
    return octupole

#ctupole[i, j, k] = octupole[i, j, k] + (charges[l, 3] * ((15 * charges[l, i] * charges[l, j] * charges[l, k]) -
                             #                                   3 * charges[l, i] * kroenecker_delta[j, k] *
                              #                                  np.square(r_l)))

##########################################################################
#                                                                        #
# EGYELŐRE HARDCODED AZ ÁTMENETEK KIVÁLASZTÁSA, ÁLTALÁNOSÍTHATÓ          #
#                                                                        #
##########################################################################

# does the full and Rydberg decoupled analysis from the OmFrag.txt
def omfrag_analysis(input_file_name): #a fragmense számára hardcoded
    df = pd.read_fwf(input_file_name)
    df.drop('4', inplace=True, axis=1)
    symmetries = np.vstack(df["Unnamed: 1"])
    df.drop('Unnamed: 1', inplace=True, axis=1)
    Om_matrix = np.zeros([20, 1], dtype=float)
    for i in range(2, 19):
        seged = np.vstack(df["Unnamed: %i" % i])
        Om_matrix = np.concatenate((Om_matrix.copy(), seged), axis=1)
    Om_matrix = np.delete(Om_matrix, 0, axis=1)
    CT_character = float(0)
    decoupled_CT_char = float(0)
    CT_char_table = np.array([1., 1.], dtype=float)
    for i in range(0, np.shape(Om_matrix)[0]):
        norm = Om_matrix[i, 0]
        decoupled_norm = Om_matrix[i, 0]
        # theta = float(0)
        for j in range(0, np.shape(Om_matrix)[1]):
            if j == 2 or j == 5:
                decoupled_CT_char = decoupled_CT_char + Om_matrix[i, j]
                CT_character = CT_character + Om_matrix[i, j]
            elif j == 4 or j == 7 or j >= 9:
                CT_character = CT_character + Om_matrix[i, j]
                decoupled_norm = decoupled_norm - Om_matrix[i, j]
                # print(f"3. if {CT_character}")
                # theta = theta + Om_matrix[i, j]
                # print(CT_character)
                # print(decoupled_norm)
            else:
                continue  # ez lehet nem jó
        #print(CT_character)
        normalized_CT_char = CT_character / norm #CT_character/Om_matrix[i, 0]
        #print(f"normalized_CT{normalized_CT_char}")
        norm_decoupled_CT_char = decoupled_CT_char / decoupled_norm
        #print(f"normalized_CT{norm_decoupled_CT_char}")
        row_CT_char_table = np.hstack([normalized_CT_char, norm_decoupled_CT_char])
        #print(normalized_CT_char)
        CT_char_table = np.vstack([CT_char_table, row_CT_char_table])
        CT_character = float(0)
        decoupled_CT_char = float(0)
    CT_char_table = np.delete(CT_char_table, 0, axis=0)
    return symmetries, CT_char_table


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

##########################################################################
#                                                                        #
# CT_file név hardcoded                                                  #
#                                                                        #
##########################################################################
def exc_state_energy_extract(file: str, comment_line: str, num_of_symmetry: int
                             , num_of_exc_state: int, output_file: str, **kwargs: str):
    if os.path.isfile(file):
        f = open(file, 'r')
        seged = f.readlines()
        dist_row = [x for x in seged if re.search(comment_line, x)][0]
        dist = re.findall(r"[-+]?\d*\.\d+|\d+", dist_row)[0]
        f.close()
        CT_type = "None"
        CT_file = "None"
        if "CT_type" in kwargs:
            CT_type = kwargs['CT_type']
        if "CT_file" in kwargs:
            CT_file = kwargs["CT_file"]
        if CT_type == 'normal':
            CT_table_row_needed_2 = \
                [x for x in seged if re.search(f"{num_of_exc_state + 1}\(1\)Sy{num_of_symmetry}", x)][0].split()
        buta = search_string_in_file(file, 'Total EOMEE-CCSD electronic energy') #   Transition energy
        out = open(output_file, "a+")
        if not buta:
            out.write(f"MISSING Point")
            return print(f"problem in {file}")
        elif len(buta) <= num_of_exc_state:
            out.write(f"MISSING Point")
            return print(f"Less exc state in {file}")            
        num = seged[buta[num_of_exc_state]-1].strip(
            'Total EOMEE-CCSD electronic energy ' + 'a.u.\n')  #' ez leszedi a mínusz jelet is!
        #out = open(output_file, "a+")
        if CT_type == 'normal':
            out.write("distance \t Energy \t state\tdE(eV)\tf\tOm\tPOS\tPOSi\tPOSf\tPR\tDEL\tCT\tCTnt\n")
            out.write('%s ' % dist + '\t %s \t' % num)
            CT_table_row_needed_2 = \
                [x for x in seged if re.search(f"{num_of_exc_state + 1}\(1\)Sy{num_of_symmetry}", x)][0].split()
            for word in CT_table_row_needed_2:
                out.write(f"{word} \t")
        if CT_type == 'Rydberg_decoupled':
            out.write(f"{dist} \t -{num} \t")
            sym, CT_table = omfrag_analysis(input_file_name=CT_file)
            #print(CT_table)
            pos_of_excitation = np.where(sym == f"{num_of_exc_state + 1}(1)Sy{num_of_symmetry}")[0]
            #print(pos_of_excitation)
            current_state = sym[pos_of_excitation]
            #print(current_state)
            CT_characters = CT_table[pos_of_excitation]
            print(CT_characters)
            out.write(f"{current_state[0, 0]} \t {CT_characters[0, 0]: .4f} \t {CT_characters[0, 1]: .4f} \n")
        else:
            out.write(f"{dist} \t -{num} \n")
        out.close()
    return "Done"


##########################################################################
#                                                                        #
# NCC programhoz, CCSDT-re!!!!                                           #
#                                                                        #
##########################################################################
def exc_state_from_NCC(file: str, comment_line: str, num_of_exc_state: int, output_file: str):
    if os.path.isfile(file):
        with open(file, 'r') as f:
            seged = f.readlines()
        dist_row = [x for x in seged if re.search(comment_line, x)][0]
        dist = re.findall(r"[-+]?\d*\.\d+|\d+", dist_row)[0]
        buta = search_string_in_file(file, 'Total EOMEE-CCSDT energy')
        print(buta)
        print(len(seged))
        num = seged[buta[num_of_exc_state] - 1].strip(
            'Total EOMEE-CCSDT energy: ')  # ez leszedi a mínusz jelet is!
        with open(output_file, "a+") as out:
            out.write(f"{dist} \t -{num}")
    return "Done"


#get guess vector from multiroot, #hardcoded for three lines
def get_guess_vector(filename: str, line_to_look_for: str, num_of_exc: int):
    vector = np.array([3], dtype=str)
    with open(filename) as file:
        seged = file.readlines()
        print(line_to_look_for)
        num_of_rows = search_string_in_file(filename, line_to_look_for)
        print(num_of_rows)
    for i in range(5, 8):
        element = int(num_of_rows[0]) + i
        vector_from_file = seged[element].strip('AA  \n')
        vector_to_write = f"1 {vector_from_file}"
        vector = np.vstack([vector, vector_to_write])
    return vector


##########################################################################
#                                                                        #
# Hardcoded for two fragments from Turbomole CC2 and 1 = right, 2 = left TRDM #
#                                                                        #
##########################################################################
def frenkel_coupling_components(filename: str, interaction_needed: str):
    with open(filename) as file:
        seged = file.readlines()
        num_of_rows = search_string_in_file(filename, "R   Pair   V12(Foerster)       "
                                                                                   "V12(Full)       V12(Coul)        "
                                                                                   "V12(Exc)         S12")
    dist_row = [x for x in seged if re.search('r12', x)][0]
    dist = re.findall(r"[-+]?\d*\.\d+|\d+", dist_row)[1]
    table = np.array([1.], dtype=str)
    for i in range(num_of_rows[0], num_of_rows[0] + 8):
        table = np.hstack([table, seged[i]])
    table = np.delete(table, 0, axis=0)
    result: str
    splitted_result = "None"
    if interaction_needed == "RR":
        result = table[1].strip('R    1  1 ')
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        splitted_result = [float(x) for x in re.findall(match_number, result)]
    if interaction_needed == "LL":
        result = table[2].strip('R    2  2 ')
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        splitted_result = [float(x) for x in re.findall(match_number, result)]
    if interaction_needed == "avg same":
        result = table[3].strip('R  -avrg- ')
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        splitted_result = [float(x) for x in re.findall(match_number, result)]
    if interaction_needed == "RL":
        result = table[4].strip('R    1  2 ')
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        splitted_result = [float(x) for x in re.findall(match_number, result)]
    if interaction_needed == "LR":
        result = table[5].strip('R    2  1 ')
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        splitted_result = [float(x) for x in re.findall(match_number, result)]
    if interaction_needed == "avg opposite":
        result = table[6].strip('R  -avrg- ')
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        splitted_result = [float(x) for x in re.findall(match_number, result)]
    if interaction_needed == "sum":
        result = table[7].strip('R  -sum-  ')
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        splitted_result = [float(x) for x in re.findall(match_number, result)]
    return dist, splitted_result

##########################################################################
#                                                                        #
# cfour az outputhoz SCF, correlation energy                             #
#                                                                        #
##########################################################################
def gr_state_from_cfour(file: str, comment_line: str, which_energy: str, output_file: str):
    if os.path.isfile(file):
        with open(file, 'r') as f:
            seged = f.readlines()
        dist_row = [x for x in seged if re.search(comment_line, x)][0]
        dist = re.findall(r"[-+]?\d*\.\d+|\d+", dist_row)[0]
        buta = search_string_in_file(file, 'A miracle has come to pass. The CC iterations have converged.')
        print(buta)
        num = seged[buta[0]]#.strip('Total EOMEE-CCSDT energy: ')  # ez leszedi a mínusz jelet is!A miracle has come to pass. The CC iterations have converged.
        if which_energy == 'SCF energy':
            num = seged[buta[0]].strip(f'The reference energy is      ' + 'a.u. \n')  # ez leszedi a mínusz jelet is!'The reference energy is ' + 'a.u. \n'
            print(num)
        if which_energy == 'correlation energy':
            num = seged[buta[0] + 1].strip('The correlation energy is' + f'a.u.\n')  #
        if which_energy == 'CC energy':
            num = seged[buta[0] + 2].strip('The total energy is' + f'a.u.\n')  # ez leszedi a mínusz jelet is!
        else:
            "Wrong statement at which energy"
        with open(output_file, "a+") as out:
            out.write(f"{dist} \t {num} \n")
    return "Done"

##########################################################################
#                                                                        #
# cfour ncc outputhoz SCF, correlation energy                            #
#                                                                        #
##########################################################################
def gr_state_from_ncc(file: str, comment_line: str, output_file: str):
    if os.path.isfile(file):
        with open(file, 'r') as f:
            seged = f.readlines()
        dist_row = [x for x in seged if re.search(comment_line, x)][0]
        dist = re.findall(r"[-+]?\d*\.\d+|\d+", dist_row)[0]
        SCF_E_place = search_string_in_file(file, 'Second-order MP correlation energies:')
        CCSD_E_line = search_string_in_file(file, f'Total CCSD energy:')
        print(CCSD_E_line)
        CCSDappT_line = search_string_in_file(file, f'Total CCSD(T) energy:')
        SCF_energy = seged[SCF_E_place[0] + 1].strip(f'E(SCF)                  =' + 'a.u. \n')
        MP2_energy = seged[SCF_E_place[0] + 5].strip(f'Total MP2 energy        =' + 'a.u. \n')
        CCSD_energy = seged[CCSD_E_line[0] - 1].strip(f'Total CCSD energy:' + '\n')
        CCSDappT_energy = seged[CCSDappT_line[0] - 1].strip(f'Total CCSD(T) energy:' + '\n')
        with open(output_file, "a+") as out:
            out.write(f"{dist} \t {SCF_energy} \t {MP2_energy} \t {CCSD_energy} \t {CCSDappT_energy} \n")
    return "Done"

##########################################################################
#                                                                        #
# enrgies from dalton snoop CC gr state calc.                            #
#                                                                        #
##########################################################################
def gr_state_snoop_dalton(file: str, comment_line: str, output_file: str):
    if os.path.isfile(file):
        with open(file, 'r') as f:
            seged = f.readlines()
        dist_row = [x for x in seged if re.search(comment_line, x)][0]
        dist = re.findall(r"[-+]?\d*\.\d+|\d+", dist_row)[0]
        SCF_ref_E_place = search_string_in_file(file, 'Full system     --- HF   energy :')
        SCF_ref_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[SCF_ref_E_place[0] - 1])[0]
        CCSD_ref_corr_E_place = search_string_in_file(file, f'CCSD correlation energy :')
        CCSD_ref_corr_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[CCSD_ref_corr_E_place[0] - 1])[0]
        CCSDappT_ref_line = search_string_in_file(file, f' Full system     --- tot  energy :')
        CCSDappT_ref_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[CCSDappT_ref_line[0] - 1])[0]
        SCF_Sys1_E_place = search_string_in_file(file, 'Subsystem:     1 --- HF   energy:')
        SCF_Sys1_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[SCF_Sys1_E_place[0] - 1])[1]
        CCSD_Sys1_corr_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[CCSD_ref_corr_E_place[1] - 1])[0]
        CCSDappT_Sys1_line = search_string_in_file(file, f' Subsystem:     1 --- tot  energy:')
        CCSDappT_Sys1_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[CCSDappT_Sys1_line[0] - 1])[1]
        SCF_Sys2_E_place = search_string_in_file(file, 'Subsystem:     2 --- HF   energy:')
        SCF_Sys2_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[SCF_Sys2_E_place[0] - 1])[1]
        CCSD_Sys2_corr_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[CCSD_ref_corr_E_place[2] - 1])[0]
        CCSDappT_Sys2_line = search_string_in_file(file, f' Subsystem:     2 --- tot  energy:')
        CCSDappT_Sys2_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[CCSDappT_Sys2_line[0] - 1])[1]
        SCF_INT_E_place = search_string_in_file(file, 'HF Interaction energy      =')
        SCF_INT_E_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[SCF_INT_E_place[0] - 1])[0]
        CCSDappT_INT_E_place = search_string_in_file(file, 'Total Interaction energy   =')
        CCSDappT_INT_E_energy = re.findall(r"[-+]?\d*\.\d+|\d+", seged[CCSDappT_INT_E_place[0] - 1])[0]
        print(CCSD_ref_corr_energy)
        print(CCSD_Sys1_corr_energy)
        print(CCSD_Sys2_corr_energy)
        with open(output_file, "a+") as out:
            out.write(f"{dist} \t {SCF_ref_energy} \t {float(SCF_ref_energy) + float(CCSD_ref_corr_energy):.10f} \t {CCSDappT_ref_energy} \t")
            out.write(f"{SCF_Sys1_energy} \t {float(SCF_Sys1_energy) + float(CCSD_Sys1_corr_energy):.10f} \t {CCSDappT_Sys1_energy} \t")
            out.write(f"{SCF_Sys2_energy} \t {float(SCF_Sys2_energy) + float(CCSD_Sys2_corr_energy):.10f} \t {CCSDappT_Sys2_energy} \t")
            out.write(f"{SCF_INT_E_energy} \t {CCSDappT_INT_E_energy} \n")
    return "Done"


def rot_around_axis(molecule2, axis, rot):
    rot_mol2 = np.zeros((len(molecule2), 3), float)
    for k in range(0, len(molecule2)):
        print(type(k))
        print(molecule2.copy()[k, :])
        if axis == "z":
            rot_mol2[k, :] = rotation(molecule2.copy()[k, :], rot, 0., 0., "Dipole")
        elif axis == "y":
            rot_mol2[k, :] = rotation(molecule2.copy()[k, :], 0., rot, 0., "Dipole")
        elif axis == "x":
            rot_mol2[k, :] = rotation(molecule2.copy()[k, :], 0., 0., rot, "Dipole")
        else:
            print("Wrong axis")
    return rot_mol2


def get_trans_moms(file: str, comment_line: str, which_moment: str, num_of_exc_state: int, rot_axis: str,
                    rot_degree: float, output_file: str):
    if which_moment == 'left':
        trans_mom = 'Left Transition Moment'
    elif which_moment == 'right':
        trans_mom = 'Right Transition Moment'
    if os.path.isfile(file):
        with open(file, 'r') as trans_mom_f:
            seged = trans_mom_f.readlines()
        dist_row = [x for x in seged if re.search(comment_line, x)][0]
        dist = re.findall(r"[-+]?\d*\.\d+|\d+", dist_row)[0]
        left = [x for x in seged if re.search(trans_mom, x)]
        for k in range(0, len(left)):
            if k == num_of_exc_state:
                trans_mom_write = left[k].strip(trans_mom + '\n')
                trans_mom_write = trans_mom_write.split('   ')
                trans_mom_write = list(filter(None, trans_mom_write))
    print(trans_mom_write)
    trans_mom_rot = np.array([trans_mom_write])
    trans_mom_rot = converttofloat(trans_mom_rot)
    rot_trans_mom = rot_around_axis(trans_mom_rot, axis=rot_axis, rot=rot_degree)
    with open(output_file, "a+") as out:
        out.write(f"{dist} \t {trans_mom_write[0]} \t {trans_mom_write[1]} \t {trans_mom_write[2]} \t rotated: \t "
                  f"{rot_trans_mom[0, 0]:10f} \t {rot_trans_mom[0, 1]:10f} \t {rot_trans_mom[0, 2]:10f}\n")
    return print("Done")
