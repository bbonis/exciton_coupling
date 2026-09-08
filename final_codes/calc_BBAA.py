import numpy as np


def get_vectors_with_overlap_red_2(red_C_s, red_LCAO_s, S_blocks, S_inv_blocks, use_S_inv: bool = False):
    #S_AB_mo = oc.Ao_to_MO_trafo(LCAO_coeffs_A, S_AB, LCAO_coeffs_B)
    #S_BA_mo = oc.Ao_to_MO_trafo(LCAO_coeffs_B, S_AB.T, LCAO_coeffs_A)
    # -------------------------------
    # Standard AO vectors (no S)
    # -------------------------------
    AO = {}

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


    # -------------------------------
    # Optional: Overlap AO vectors (AO_C_S)
    # -------------------------------
    if S_blocks is not None:


        AO['SC_A_BA_ia'] = red_LCAO_s['LCAO_B_red_occ'] @ S_blocks['S_BA_ii'] @ red_C_s['CIS_matrix_A_red_left'] @ red_LCAO_s['LCAO_A_red_virt'].T
        AO['CS_A_AB_ia'] = red_LCAO_s['LCAO_A_red_occ'] @ red_C_s['CIS_matrix_A_red_right'] @ S_blocks['S_AB_aa'] @ red_LCAO_s['LCAO_B_red_virt'].T
        AO['SC_B_AB_ia'] = red_LCAO_s['LCAO_A_red_occ'] @ S_blocks['S_AB_ii'] @ red_C_s['CIS_matrix_B_red_left'] @ red_LCAO_s['LCAO_B_red_virt'].T
        AO['CS_B_BA_ia'] = red_LCAO_s['LCAO_B_red_occ'] @ red_C_s['CIS_matrix_B_red_right'] @ S_blocks['S_BA_aa'] @ red_LCAO_s['LCAO_A_red_virt'].T
        AO['CS_EA_A_a' ] = red_C_s['CIS_vector_EA_A_red_right'] @ S_blocks['S_AB_aa'] @ red_LCAO_s['LCAO_B_red_virt'].T
        AO['SC_IP_B_i' ] = red_LCAO_s['LCAO_A_red_occ'] @ S_blocks['S_AB_ii'] @ red_C_s['CIS_vector_IP_B_red_left'].T
        AO['SC_IP_A_i' ] = red_LCAO_s['LCAO_B_red_occ'] @ S_blocks['S_BA_ii'] @ red_C_s['CIS_vector_IP_A_red_left'].T
        AO['CS_EA_B_a' ] = red_C_s['CIS_vector_EA_B_red_right'] @ S_blocks['S_BA_aa'] @ red_LCAO_s['LCAO_A_red_virt'].T
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

        return     AO



def separate_indexes_and_values(row, NBAS_A):
    indexes = list(map(int, row[:4]))
    
    A_indexes = [(index - 1) for index in indexes if index <= NBAS_A]
    #A_indexes.sort(reverse=True)
    B_indexes = [(index - 1 - NBAS_A) for index in indexes if index > NBAS_A]
    #B_indexes.sort(reverse=True)
    value = np.float64(row[4])
    return  A_indexes, B_indexes, value


def calc_term_BBAA_2(ind_A, ind_B, value, is_coul, SC_A_BA_ia      ,
CS_A_AB_ia       ,
SC_B_AB_ia       ,
CS_B_BA_ia        ,
CS_EA_A_a       ,
SC_IP_A_i       ,
SC_IP_B_i        ,
CS_EA_B_a       ,
SC_A_BA_pa    ,
CS_A_AB_iq    ,
SC_B_AB_pa    ,
CS_B_BA_iq    ,
CS_EA_A_q     ,
SC_IP_B_p     ,
SC_IP_A_p     ,
CS_EA_B_q     ,
ACA_A_right   ,
SCS_A_BB_iq   ,
SCS_A_BB_pa   ,
ACA_B_right   ,
SCS_B_AA_pa   ,
ACA_A_left    ,
ACA_B_left    ,
CA_EA_A_right ,
AC_IP_B_right ,
AC_IP_A_right ,
CA_EA_B_right ,
CA_EA_A_left  ,
AC_IP_B_left  ,
AC_IP_A_left  ,
CA_EA_B_left  ,):
    term_21_X = np.float64(0.)
    term_21_C = np.float64(0.)
    term_21_X_2S = np.float64(0.)
    term_21_C_2S = np.float64(0.)
    term_12_X_2S = np.float64(0.)
    term_12_C_2S = np.float64(0.)
    
    term_43_X = np.float64(0.)
    term_43_C = np.float64(0.)
    
    term_12_X = np.float64(0.)
    term_12_C = np.float64(0.)    
    term_34_X = np.float64(0.)
    term_34_C = np.float64(0.)    
    
    term_13_C_local_virt = np.float64(0.)
    term_13_X_local_virt = np.float64(0.)
    term_31_C_local_occ = np.float64(0.)
    term_31_X_local_occ = np.float64(0.)
    term_31_C_EA = np.float64(0.)
    term_31_X_EA = np.float64(0.)    
    
    term_14_C_local_virt = np.float64(0.)
    term_14_X_local_virt = np.float64(0.)
    term_41_C_local_occ = np.float64(0.)
    term_41_X_local_occ = np.float64(0.)
    term_14_C_IP = np.float64(0.)
    term_14_X_IP = np.float64(0.)   

    term_32_C_local_occ = np.float64(0.)
    term_32_X_local_occ = np.float64(0.)
    term_23_C_local_virt = np.float64(0.)
    term_23_X_local_virt = np.float64(0.)
    term_23_C_IP = np.float64(0.)
    term_23_X_IP = np.float64(0.) 

    term_24_C_local_virt = np.float64(0.)
    term_24_X_local_virt = np.float64(0.)
    term_42_C_local_occ = np.float64(0.)
    term_42_X_local_occ = np.float64(0.)
    term_42_C_EA = np.float64(0.)
    term_42_X_EA = np.float64(0.)        

    term_41_C_1S_IP_arb   = np.float64(0.)
    term_41_X_1S_virt_arb = np.float64(0.)
    term_13_C_1S_EA_arb   = np.float64(0.)
    term_13_X_1S_occ_arb  = np.float64(0.)
    
    term_41_C_1S_virt_arb = np.float64(0.)
    term_41_X_1S_IP_arb   = np.float64(0.)
    term_13_C_1S_occ_arb  = np.float64(0.)
    term_13_X_1S_EA_arb   = np.float64(0.)
    term_31_C_1S_virt_arb = np.float64(0.)
    term_31_X_1S_virt_arb = np.float64(0.) 
    term_14_C_1S_occ_arb  = np.float64(0.)
    term_14_X_1S_occ_arb  = np.float64(0.)

    term_31_C_3S_occ_IP_arb_virt_arb = np.float64(0.)
    term_13_C_3S_IP_virt_occ_arb = np.float64(0.)
    term_41_C_3S_occ_EA_virt_arb = np.float64(0.)
    term_14_C_3S_virt_EA_arb_occ_arb = np.float64(0.)
    term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb = np.float64(0.)
    term_43_C_2S_IP_IP_arb = np.float64(0.)
    term_31_X_3S_EA_IP_arb_virt_arb = np.float64(0.)
    term_13_X_3S_IP_virt_EA_arb = np.float64(0.)
    term_41_X_3S_occ_EA_IP_arb = np.float64(0.)
    term_14_X_3S_IP_occ_arb_EA_arb = np.float64(0.)
    term_21_X_2S_B_occ_arb_A_virt_arb = np.float64(0.)
    term_31_C_3S_EA_occ_IP_arb = np.float64(0.)
    term_31_X_3S_EA_occ_IP_arb = np.float64(0.)
    term_14_C_3S_IP_virt_EA_arb = np.float64(0.)
    term_14_X_3S_IP_virt_EA_arb = np.float64(0.)
    term_21_C_2S_A_occ_B_occ_arb = np.float64(0.)
    term_21_X_2S_A_occ_B_occ_arb = np.float64(0.)
    term_43_C_2S_EA_IP_arb = np.float64(0.)
    term_43_X_2S_EA_IP_arb = np.float64(0.)
    term_43_C_2S_EA_EA_arb = np.float64(0.)
    term_43_X_2S_EA_EA_arb = np.float64(0.)
    term_31_X_3S_occ_IP_arb_virt_arb = np.float64(0.)
    term_13_X_3S_IP_virt_occ_arb = np.float64(0.)
    term_41_X_3S_occ_EA_virt_arb = np.float64(0.)
    term_41_X_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_14_X_3S_virt_EA_arb_occ_arb = np.float64(0.)
    term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb = np.float64(0.)
    term_43_X_2S_IP_IP_arb = np.float64(0.)
    term_31_C_3S_EA_IP_arb_virt_arb = np.float64(0.)
    term_13_C_3S_IP_virt_EA_arb = np.float64(0.)
    term_41_C_3S_occ_EA_IP_arb = np.float64(0.)
    term_14_C_3S_IP_occ_arb_EA_arb = np.float64(0.)
    term_21_C_2S_B_occ_arb_A_virt_arb = np.float64(0.)
    term_43_C_2S_IP_EA_arb = np.float64(0.)
    term_43_X_2S_IP_EA_arb = np.float64(0.)


    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)

    if is_coul: # itt megfordul az integrálok típusa!!!
        term_43_X_2S_IP_EA_arb += X_prefactor * AC_IP_A_left[ind_A[0], :] * CA_EA_B_left[:, ind_B[0]] * SC_IP_B_i[ind_A[1], :] * CS_EA_A_q[:, ind_B[1]] * value

        term_31_C_3S_occ_IP_arb_virt_arb += C_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * CA_EA_A_left[:, ind_A[0]] * SC_IP_B_p[ind_A[1], :] * value
        term_13_C_3S_IP_virt_occ_arb += C_prefactor * SCS_A_BB_pa[ind_B[0], ind_B[1]] * CA_EA_A_right[:, ind_A[0]] * SC_IP_B_i[ind_A[1], :] * value
        term_41_C_3S_occ_EA_virt_arb += C_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * CS_EA_B_a[:, ind_A[0]] * AC_IP_A_left[ind_A[1], :] * value
        term_14_C_3S_virt_EA_arb_occ_arb += C_prefactor * SCS_A_BB_pa[ind_B[0], ind_B[1]] * CS_EA_B_q[:, ind_A[0]] * AC_IP_A_right[ind_A[1], :] * value
        term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * SCS_B_AA_pa[ind_A[0], ind_A[1]] * value
        term_43_C_2S_IP_IP_arb += C_prefactor * CA_EA_B_left[:, ind_B[0]] * SC_IP_B_i[ind_A[0], :] * CA_EA_A_right[:, ind_A[1]] * SC_IP_A_p[ind_B[1], :] * value
        term_31_X_3S_EA_IP_arb_virt_arb += X_prefactor * CS_A_AB_iq[ind_A[0], ind_B[0]] * CS_EA_A_a[:, ind_B[1]] * SC_IP_B_p[ind_A[1], :] * value
        term_13_X_3S_IP_virt_EA_arb += X_prefactor * CS_A_AB_ia[ind_A[0], ind_B[0]] *  CS_EA_A_q[:, ind_B[1]] * SC_IP_B_i[ind_A[1], :] * value
        term_41_X_3S_occ_EA_IP_arb += X_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * CS_EA_B_a[:, ind_A[1]] * SC_IP_A_p[ind_B[1], :] * value
        term_14_X_3S_IP_occ_arb_EA_arb += X_prefactor * SC_A_BA_pa[ind_B[0], ind_A[0]] * CS_EA_B_q[:, ind_A[1]] * SC_IP_A_i[ind_B[1], :] * value
        term_21_X_2S_B_occ_arb_A_virt_arb += X_prefactor * CS_A_AB_iq[ind_A[0], ind_B[0]] * SC_B_AB_pa[ind_A[1], ind_B[1]] * value
        
        term_31_X_local_occ += X_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * CA_EA_A_left[0, ind_A[1]] *  AC_IP_B_left[ind_B[1], 0 ] * value
        term_31_C_EA += C_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * CS_EA_A_a[0, ind_B[0]] *  AC_IP_B_left[ind_B[1], 0] * value
        
        term_21_X_2S += X_prefactor * SC_A_BA_ia[ind_B[0], ind_A[1]] * CS_B_BA_ia[ind_B[1], ind_A[0]] * value
    
        term_12_X_2S += X_prefactor * SC_B_AB_ia[ind_A[0], ind_B[1]] * CS_A_AB_ia[ind_A[1], ind_B[0]] * value
        
        term_42_X_local_occ += X_prefactor * SC_B_AB_ia[ind_A[0], ind_B[0]] * CA_EA_B_left[0, ind_B[1]] *  AC_IP_A_left[ind_A[1], 0 ] * value
        term_42_C_EA += C_prefactor * ACA_B_right[ind_B[0], ind_B[1]] * CS_EA_B_a[0, ind_A[0]] *  AC_IP_A_left[ind_A[1], 0] * value
        
        term_14_X_local_virt += X_prefactor * CS_A_AB_ia[ind_A[0], ind_B[0]] * AC_IP_A_right[ind_A[1], 0] *  CA_EA_B_right[0, ind_B[1]] * value

        term_14_C_IP += C_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * SC_IP_A_i[ind_B[0], 0] *  CA_EA_B_right[0, ind_B[1]] * value
        term_23_X_local_virt += X_prefactor * CS_B_BA_ia[ind_B[0], ind_A[0]] * AC_IP_B_right[ind_B[1], 0] *  CA_EA_A_right[0, ind_A[1]] * value
        term_23_C_IP += C_prefactor * ACA_B_left[ind_B[0], ind_B[1]] * SC_IP_B_i[ind_A[0], 0] *  CA_EA_A_right[0, ind_A[1]] * value
        
        term_21_C += C_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * ACA_B_left[ind_B[0], ind_B[1]] * value
        term_12_C += C_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * ACA_B_right[ind_B[0], ind_B[1]] * value
        term_43_X_2S_EA_IP_arb += X_prefactor * CS_EA_B_a[:, ind_A[0]] * AC_IP_B_right[ind_B[0], :] * CA_EA_A_right[:, ind_A[1]] * SC_IP_A_p[ind_B[1], :] * value
        term_43_C_2S_EA_EA_arb += C_prefactor * CS_EA_B_a[:, ind_A[0]] * AC_IP_B_right[ind_B[0], :] * CS_EA_A_q[:, ind_B[1]] * AC_IP_A_left[ind_A[1], :] * value

        term_41_C_1S_IP_arb   += C_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * SC_IP_A_p[ind_B[0], 0] * CA_EA_B_left[0, ind_B[1]] * value
        term_41_X_1S_virt_arb += X_prefactor * CS_A_AB_iq[ind_A[0], ind_B[0]] * AC_IP_A_left[ind_A[1], 0] * CA_EA_B_left[0, ind_B[1]] * value
        term_13_C_1S_EA_arb   += C_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * AC_IP_B_right[ind_B[0], 0] * CS_EA_A_q[0, ind_B[1]] * value
        term_13_X_1S_occ_arb  += X_prefactor * SC_A_BA_pa[ind_B[0], ind_A[1]] * AC_IP_B_right[ind_B[1], 0] * CA_EA_A_right[0, ind_A[0]] * value
        
    else:
        term_43_C_2S_IP_EA_arb += C_prefactor * AC_IP_A_left[ind_A[0], :] * CA_EA_B_left[:, ind_B[0]] * SC_IP_B_i[ind_A[1], :] * CS_EA_A_q[:, ind_B[1]] * value

        term_31_X_3S_occ_IP_arb_virt_arb += X_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * CA_EA_A_left[:, ind_A[0]] * SC_IP_B_p[ind_A[1], :] * value
        term_13_X_3S_IP_virt_occ_arb += X_prefactor * SCS_A_BB_pa[ind_B[0], ind_B[1]] * CA_EA_A_right[:, ind_A[0]] * SC_IP_B_i[ind_A[1], :] * value
        term_41_X_3S_occ_EA_virt_arb += X_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * CS_EA_B_a[:, ind_A[0]] * AC_IP_A_left[ind_A[1], :] * value
        term_14_X_3S_virt_EA_arb_occ_arb += X_prefactor * SCS_A_BB_pa[ind_B[0], ind_B[1]] * CS_EA_B_q[:, ind_A[0]] * AC_IP_A_right[ind_A[1], :] * value
        term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * SCS_B_AA_pa[ind_A[0], ind_A[1]] * value
        term_43_X_2S_IP_IP_arb += X_prefactor * CA_EA_B_left[:, ind_B[0]] * SC_IP_B_i[ind_A[0], :] * CA_EA_A_right[:, ind_A[1]] * SC_IP_A_p[ind_B[1], :]  * value
        term_31_C_3S_EA_IP_arb_virt_arb += C_prefactor * CS_A_AB_iq[ind_A[0], ind_B[0]] * CS_EA_A_a[:, ind_B[1]] * SC_IP_B_p[ind_A[1], :] * value
        term_13_C_3S_IP_virt_EA_arb += C_prefactor * CS_A_AB_ia[ind_A[0], ind_B[0]] *  CS_EA_A_q[:, ind_B[1]] * SC_IP_B_i[ind_A[1], :] * value
        term_41_C_3S_occ_EA_IP_arb += C_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * CS_EA_B_a[:, ind_A[1]] * SC_IP_A_p[ind_B[1], :] * value
        term_14_C_3S_IP_occ_arb_EA_arb += C_prefactor * SC_A_BA_pa[ind_B[0], ind_A[0]] * CS_EA_B_q[:, ind_A[1]] * SC_IP_A_i[ind_B[1], :] * value
        term_21_C_2S_B_occ_arb_A_virt_arb += C_prefactor * CS_A_AB_iq[ind_A[0], ind_B[0]] * SC_B_AB_pa[ind_A[1], ind_B[1]] * value
        term_31_C_3S_EA_occ_IP_arb += C_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * CS_EA_A_a[:, ind_B[1]] * SC_IP_B_p[ind_A[1], :] * value
        term_31_X_3S_EA_occ_IP_arb += X_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * CS_EA_A_a[:, ind_B[1]] * SC_IP_B_p[ind_A[1], :] * value
        term_14_C_3S_IP_virt_EA_arb += C_prefactor * CS_A_AB_ia[ind_A[0], ind_B[0]] * CS_EA_B_q[:, ind_A[1]] * SC_IP_A_i[ind_B[1], :] * value
        term_14_X_3S_IP_virt_EA_arb += X_prefactor * CS_A_AB_ia[ind_A[0], ind_B[0]] * CS_EA_B_q[:, ind_A[1]] * SC_IP_A_i[ind_B[1], :] * value
        term_21_C_2S_A_occ_B_occ_arb += C_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * SC_B_AB_pa[ind_A[1], ind_B[1]] * value
        term_21_X_2S_A_occ_B_occ_arb += X_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * SC_B_AB_pa[ind_A[1], ind_B[1]] * value
        term_43_C_2S_EA_IP_arb += C_prefactor * CS_EA_B_a[:, ind_A[0]] * AC_IP_B_right[ind_B[0], :] * CA_EA_A_right[:, ind_A[1]] * SC_IP_A_p[ind_B[1], :] * value
        term_43_X_2S_EA_EA_arb += X_prefactor * CS_EA_B_a[:, ind_A[0]] * AC_IP_B_right[ind_B[0], :] * CS_EA_A_q[:, ind_B[1]] * AC_IP_A_left[ind_A[1], :] * value

        term_41_C_1S_virt_arb += C_prefactor * CS_A_AB_iq[ind_A[0], ind_B[0]] * AC_IP_A_left[ind_A[1], 0] * CA_EA_B_left[0, ind_B[1]] * value
        term_41_X_1S_IP_arb   += X_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * SC_IP_A_p[ind_B[0], 0] * CA_EA_B_left[0, ind_B[1]] * value
        term_13_C_1S_occ_arb  += C_prefactor * SC_A_BA_pa[ind_B[0], ind_A[1]] * AC_IP_B_right[ind_B[1], 0] * CA_EA_A_right[0, ind_A[0]] * value
        term_13_X_1S_EA_arb   += X_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * AC_IP_B_right[ind_B[0], 0] * CS_EA_A_q[0, ind_B[1]] * value
        term_31_C_1S_virt_arb += C_prefactor * CS_A_AB_iq[ind_A[0], ind_B[0]] * AC_IP_B_right[ind_B[1], 0] * CA_EA_A_right[0, ind_A[1]] * value
        term_31_X_1S_virt_arb += X_prefactor * CS_A_AB_iq[ind_A[0], ind_B[0]] * AC_IP_B_right[ind_B[1], 0] * CA_EA_A_right[0, ind_A[1]] * value
        term_14_C_1S_occ_arb  += C_prefactor * SC_A_BA_pa[ind_B[0], ind_A[0]] * AC_IP_A_left[ind_A[1], 0] * CA_EA_B_left[0, ind_B[1]] * value
        term_14_X_1S_occ_arb  += X_prefactor * SC_A_BA_pa[ind_B[0], ind_A[0]] * AC_IP_A_left[ind_A[1], 0] * CA_EA_B_left[0, ind_B[1]] * value

        term_21_C_2S += C_prefactor * SC_A_BA_ia[ind_B[0], ind_A[1]] * CS_B_BA_ia[ind_B[1], ind_A[0]] * value
        
        term_12_C_2S += C_prefactor * SC_B_AB_ia[ind_A[0], ind_B[1]] * CS_A_AB_ia[ind_A[1], ind_B[0]] * value
        
        term_31_C_local_occ += C_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * CA_EA_A_left[0, ind_A[1]] *   AC_IP_B_left[ind_B[1], 0] * value
        term_31_X_EA += X_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * CS_EA_A_a[0, ind_B[0]] *   AC_IP_B_left[ind_B[1], 0] * value
        term_13_C_local_virt += C_prefactor * CS_A_AB_ia[ind_A[0], ind_B[0]] * CA_EA_A_right[0, ind_A[1]] *   AC_IP_B_right[ind_B[1], 0] * value
        term_13_X_local_virt += X_prefactor * CS_A_AB_ia[ind_A[0], ind_B[0]] * CA_EA_A_right[0, ind_A[1]] *   AC_IP_B_right[ind_B[1], 0] * value
        
        term_24_C_local_virt += C_prefactor * CS_B_BA_ia[ind_B[0], ind_A[0]] * CA_EA_B_right[0, ind_B[1]] *  AC_IP_A_right[ind_A[1], 0] * value
        term_24_X_local_virt += X_prefactor * CS_B_BA_ia[ind_B[0], ind_A[0]] * CA_EA_B_right[0, ind_B[1]] *  AC_IP_A_right[ind_A[1], 0] * value            
        
        term_42_C_local_occ += C_prefactor * SC_B_AB_ia[ind_A[0], ind_B[0]] * CA_EA_B_left[0, ind_B[1]] *  AC_IP_A_left[ind_A[1], 0] * value
        term_42_X_EA += X_prefactor * ACA_B_right[ind_B[0], ind_B[1]] * CS_EA_B_a[0, ind_A[0]] *  AC_IP_A_left[ind_A[1], 0] * value
        
        term_14_C_local_virt += C_prefactor * CS_A_AB_ia[ind_A[0], ind_B[0]] * AC_IP_A_right[ind_A[1], 0] *  CA_EA_B_right[0, ind_B[1]] * value

        term_14_X_IP += X_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * SC_IP_A_i[ind_B[0], 0] *  CA_EA_B_right[0, ind_B[1]] * value
        
        term_41_C_local_occ += C_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * AC_IP_A_left[ind_A[1], 0] *  CA_EA_B_left[0, ind_B[1]] * value
        term_41_X_local_occ += X_prefactor * SC_A_BA_ia[ind_B[0], ind_A[0]] * AC_IP_A_left[ind_A[1], 0] *  CA_EA_B_left[0, ind_B[1]] * value
        
        term_23_C_local_virt += C_prefactor * CS_B_BA_ia[ind_B[0], ind_A[0]] * AC_IP_B_right[ind_B[1], 0] *  CA_EA_A_right[0, ind_A[1]] * value
        term_23_X_IP += X_prefactor * ACA_B_left[ind_B[0], ind_B[1]] * SC_IP_B_i[ind_A[0], 0] *  CA_EA_A_right[0, ind_A[1]] * value
        
        term_32_C_local_occ += C_prefactor * SC_B_AB_ia[ind_A[0], ind_B[0]] *  AC_IP_B_left[ind_B[1], 0] *  CA_EA_A_left[0, ind_A[1]] * value
        term_32_X_local_occ += X_prefactor * SC_B_AB_ia[ind_A[0], ind_B[0]] *  AC_IP_B_left[ind_B[1], 0] *  CA_EA_A_left[0, ind_A[1]] * value
        
        term_21_X += X_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * ACA_B_left[ind_B[0], ind_B[1]] * value
        term_12_X += X_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * ACA_B_right[ind_B[0], ind_B[1]] * value
        
        term_43_X += X_prefactor * AC_IP_A_left[ind_A[0], 0] * CA_EA_A_right[0, ind_A[1]] * AC_IP_B_right[ind_B[0], 0] * CA_EA_B_left[0, ind_B[1]] * value    
        term_43_C += C_prefactor * AC_IP_A_left[ind_A[0], 0] * CA_EA_A_right[0, ind_A[1]] * AC_IP_B_right[ind_B[0], 0] * CA_EA_B_left[0, ind_B[1]] * value   
        term_34_X += X_prefactor * AC_IP_A_right[ind_A[0], 0] * CA_EA_A_left[0, ind_A[1]] * AC_IP_B_left[ind_B[0], 0] * CA_EA_B_right[0, ind_B[1]] * value    
        term_34_C += C_prefactor * AC_IP_A_right[ind_A[0], 0] * CA_EA_A_left[0, ind_A[1]] * AC_IP_B_left[ind_B[0], 0] * CA_EA_B_right[0, ind_B[1]] * value   
    return          term_21_X, term_21_C, term_43_X, term_43_C, term_12_X, term_12_C, term_34_X, term_34_C, term_13_C_local_virt, term_13_X_local_virt, term_31_C_local_occ, term_31_X_local_occ, term_31_C_EA, term_31_X_EA, term_14_C_local_virt, term_14_X_local_virt,\
                    term_41_C_local_occ, term_41_X_local_occ, term_14_C_IP, term_14_X_IP, term_21_X_2S, term_21_C_2S, term_12_X_2S, term_12_C_2S, term_32_C_local_occ, term_32_X_local_occ, term_42_C_local_occ, term_42_X_local_occ, term_42_C_EA, term_42_X_EA, \
                    term_23_C_local_virt, term_23_X_local_virt, term_23_C_IP, term_23_X_IP, term_24_C_local_virt, term_24_X_local_virt, term_41_C_1S_IP_arb, term_41_X_1S_virt_arb, term_13_C_1S_EA_arb, term_13_X_1S_occ_arb,  \
                    term_41_C_1S_virt_arb, term_41_X_1S_IP_arb, term_13_C_1S_occ_arb, term_13_X_1S_EA_arb, term_31_C_1S_virt_arb, term_31_X_1S_virt_arb, term_14_C_1S_occ_arb, term_14_X_1S_occ_arb, \
                    term_31_C_3S_occ_IP_arb_virt_arb, term_13_C_3S_IP_virt_occ_arb, term_41_C_3S_occ_EA_virt_arb, term_14_C_3S_virt_EA_arb_occ_arb, term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, \
                    term_43_C_2S_IP_IP_arb, term_31_X_3S_EA_IP_arb_virt_arb, term_13_X_3S_IP_virt_EA_arb, term_41_X_3S_occ_EA_IP_arb , term_14_X_3S_IP_occ_arb_EA_arb , term_21_X_2S_B_occ_arb_A_virt_arb, \
                    term_31_C_3S_EA_occ_IP_arb, term_31_X_3S_EA_occ_IP_arb, term_14_C_3S_IP_virt_EA_arb, term_14_X_3S_IP_virt_EA_arb, term_21_C_2S_A_occ_B_occ_arb, term_21_X_2S_A_occ_B_occ_arb, \
                    term_43_C_2S_EA_IP_arb, term_43_X_2S_EA_IP_arb, term_43_C_2S_EA_EA_arb, term_43_X_2S_EA_EA_arb, term_31_X_3S_occ_IP_arb_virt_arb, term_13_X_3S_IP_virt_occ_arb, term_41_X_3S_occ_EA_virt_arb, \
                    term_14_X_3S_virt_EA_arb_occ_arb, term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, term_43_X_2S_IP_IP_arb, term_31_C_3S_EA_IP_arb_virt_arb, \
                    term_13_C_3S_IP_virt_EA_arb, term_41_C_3S_occ_EA_IP_arb, term_14_C_3S_IP_occ_arb_EA_arb, term_21_C_2S_B_occ_arb_A_virt_arb, term_43_C_2S_IP_EA_arb, term_43_X_2S_IP_EA_arb 


def calc_term_BBAA_3_coul(indexes, term_21, term_12, ACA_right   , ACA_left    , which_frag):
    term_21_C = np.float64(0.)
    term_12_C = np.float64(0.)
    
    #perms_2 = list(distinct_permutations(indexes))

    a, b = indexes
    p0 = (a, b)
    p1 = (b, a)
    #if a == c:
    #    b, a, c = indexes
    #    p0 = (a, b, c)
    #    p2 = (b, c, a)
    #    p1 = (b, a, c)

    perms = [p1, p0]
    
    #if perms != perms_2:
    #    print('problem in calc_term_BBAA_3_coul perms')
        
    
    C_prefactor = np.float64(4.)
    
    if which_frag == 'A':
        for element in perms:
            term_21_C += C_prefactor * ACA_right[element[0], element[1]] * term_21
            term_12_C += C_prefactor * ACA_left[element[0], element[1]] * term_12
    
        return term_21_C, term_12_C
    elif which_frag == 'B':
        for element in perms:
            term_21_C += C_prefactor * ACA_left[element[0], element[1]] * term_21
            term_12_C += C_prefactor * ACA_right[element[0], element[1]] * term_12
    
        return term_21_C, term_12_C
    else:
        print('problem in calc_term_BBAA_3_coul')
        return None


def calc_term_BBAA_3_coul_with_Ov(ind_A, ind_B, value, perm: str, SC_A_BA_ia      ,
                                                                    CS_A_AB_ia       ,
                                                                    SC_B_AB_ia       ,
                                                                    CS_B_BA_ia        ,
                                                                    CS_EA_A_a       ,
                                                                    SC_IP_A_i       ,
                                                                    SC_IP_B_i        ,
                                                                    CS_EA_B_a       ,
                                                                    SC_A_BA_pa    ,
                                                                    CS_A_AB_iq    ,
                                                                    SC_B_AB_pa    ,
                                                                    CS_B_BA_iq    ,
                                                                    CS_EA_A_q     ,
                                                                    SC_IP_B_p     ,
                                                                    SC_IP_A_p     ,
                                                                    CS_EA_B_q     ,
                                                                    ACA_A_right   ,
                                                                    SCS_A_BB_iq   ,
                                                                    SCS_A_BB_pa   ,
                                                                    ACA_B_right   ,
                                                                    SCS_B_AA_pa   ,
                                                                    ACA_A_left    ,
                                                                    ACA_B_left    ,
                                                                    CA_EA_A_right ,
                                                                    AC_IP_B_right ,
                                                                    AC_IP_A_right ,
                                                                    CA_EA_B_right ,
                                                                    CA_EA_A_left  ,
                                                                    AC_IP_B_left  ,
                                                                    AC_IP_A_left  ,
                                                                    CA_EA_B_left  , ):
    
    #term_31_C_local_virt = np.float64(0.)
    #term_31_X_local_virt = np.float64(0.)
    #term_31_C_local_occ = np.float64(0.)
    term_31_X_local_occ = np.float64(0.)
    term_31_C_EA = np.float64(0.)
    term_21_X_2S = np.float64(0.)
    term_12_X_2S = np.float64(0.)
    #term_31_X_EA = np.float64(0.)    
    
    term_21_C = np.float64(0.)
    term_12_C = np.float64(0.)

    #term_41_C_local_virt = np.float64(0.)
    term_14_X_local_virt = np.float64(0.)
    #term_41_C_local_occ = np.float64(0.)
    #term_41_X_local_occ = np.float64(0.)
    term_14_C_IP = np.float64(0.)
    #term_41_X_IP = np.float64(0.)   
    term_23_X_local_virt = np.float64(0.)
    #term_41_C_local_occ = np.float64(0.)
    #term_41_X_local_occ = np.float64(0.)
    term_23_C_IP = np.float64(0.)
    
    term_42_X_local_occ = np.float64(0.)
    term_42_C_EA = np.float64(0.)    
    
    term_41_C_1S_IP_arb   = np.float64(0.)
    term_41_X_1S_virt_arb = np.float64(0.)
    term_13_C_1S_EA_arb   = np.float64(0.)
    term_13_X_1S_occ_arb  = np.float64(0.)

    term_31_C_3S_occ_IP_arb_virt_arb = np.float64(0.)
    term_13_C_3S_IP_virt_occ_arb = np.float64(0.)
    term_41_C_3S_occ_EA_virt_arb = np.float64(0.)
    term_14_C_3S_virt_EA_arb_occ_arb = np.float64(0.)
    term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb = np.float64(0.)
    term_43_C_2S_IP_IP_arb = np.float64(0.)
    term_31_X_3S_EA_IP_arb_virt_arb = np.float64(0.)
    term_13_X_3S_IP_virt_EA_arb = np.float64(0.)
    term_41_X_3S_occ_EA_IP_arb = np.float64(0.)
    term_14_X_3S_IP_occ_arb_EA_arb = np.float64(0.)
    term_21_X_2S_B_occ_arb_A_virt_arb = np.float64(0.)
    term_43_X_2S_EA_IP_arb = np.float64(0.)
    term_43_C_2S_EA_EA_arb = np.float64(0.)
    term_43_X_2S_IP_EA_arb = np.float64(0.)

    C_prefactor = np.float64(4.)  
    X_prefactor = np.float64(-2.)
    

    if perm == 'A':
        #perms_2 = list(distinct_permutations(ind_A))
        a, b = ind_A
        p0 = (a, b)
        p1 = (b, a)
        #if a == c:
        #    b, a, c = indexes
        #    p0 = (a, b, c)
        #    p2 = (b, c, a)
        #    p1 = (b, a, c)
        perms = [p1, p0]
        #if perms != perms_2:
        #    print('problem in calc_term_BBAA_3_coul perms')
        for element in perms:
            term_43_X_2S_IP_EA_arb += X_prefactor * AC_IP_A_left[element[0], :] * CA_EA_B_left[:, ind_B[0]] * SC_IP_B_i[element[1], :] * CS_EA_A_q[:, ind_B[1]] * value

            term_31_C_3S_occ_IP_arb_virt_arb += C_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * CA_EA_A_left[:, element[0]] * SC_IP_B_p[element[1], :] * value
            term_13_C_3S_IP_virt_occ_arb += C_prefactor * SCS_A_BB_pa[ind_B[0], ind_B[1]] * CA_EA_A_right[:, element[0]] * SC_IP_B_i[element[1], :] * value
            term_41_C_3S_occ_EA_virt_arb += C_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * CS_EA_B_a[:, element[0]] * AC_IP_A_left[element[1], :] * value
            term_14_C_3S_virt_EA_arb_occ_arb += C_prefactor * SCS_A_BB_pa[ind_B[0], ind_B[1]] * CS_EA_B_q[:, element[0]] * AC_IP_A_right[element[1], :] * value
            term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * SCS_B_AA_pa[element[0], element[1]] * value
            term_43_C_2S_IP_IP_arb += C_prefactor * CA_EA_B_left[:, ind_B[0]] * SC_IP_B_i[element[0], :] * CA_EA_A_right[:, element[1]] * SC_IP_A_p[ind_B[1], :] * value
            term_31_X_3S_EA_IP_arb_virt_arb += X_prefactor * CS_A_AB_iq[element[0], ind_B[0]] * CS_EA_A_a[:, ind_B[1]] * SC_IP_B_p[element[1], :] * value
            term_13_X_3S_IP_virt_EA_arb += X_prefactor * CS_A_AB_ia[element[0], ind_B[0]] *  CS_EA_A_q[:, ind_B[1]] * SC_IP_B_i[element[1], :] * value
            term_41_X_3S_occ_EA_IP_arb += X_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * CS_EA_B_a[:, element[1]] * SC_IP_A_p[ind_B[1], :] * value
            term_14_X_3S_IP_occ_arb_EA_arb += X_prefactor * SC_A_BA_pa[ind_B[0], element[0]] * CS_EA_B_q[:, element[1]] * SC_IP_A_i[ind_B[1], :] * value
            term_21_X_2S_B_occ_arb_A_virt_arb += X_prefactor * CS_A_AB_iq[element[0], ind_B[0]] * SC_B_AB_pa[element[1], ind_B[1]] * value

            term_43_C_2S_EA_EA_arb += C_prefactor * AC_IP_A_left[element[0], :] * CS_EA_B_a[:, element[1]] * AC_IP_B_right[ind_B[0], :] * CS_EA_A_q[:, ind_B[1]]       * value
            term_43_X_2S_EA_IP_arb += X_prefactor * SC_IP_A_p[ind_B[0], :]      * CS_EA_B_a[:, element[0]] * AC_IP_B_right[ind_B[1], :] * CA_EA_A_right[:, element[1]] * value
            term_21_C += C_prefactor * ACA_A_right[element[0], element[1]] * ACA_B_left[ind_B[0], ind_B[1]] * value
            term_12_C += C_prefactor * ACA_A_left[element[0], element[1]] * ACA_B_right[ind_B[0], ind_B[1]] * value
            term_31_C_EA += C_prefactor * ACA_A_right[element[0], element[1]] * CS_EA_A_a[0, ind_B[0]] *  AC_IP_B_left[ind_B[1], 0] * value
            term_31_X_local_occ += X_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * CA_EA_A_left[0, element[1]] *  AC_IP_B_left[ind_B[1], 0 ] * value
            term_12_X_2S += X_prefactor * SC_B_AB_ia[element[0], ind_B[0]] * CS_A_AB_ia[element[1], ind_B[1]] * value
            term_42_C_EA += C_prefactor * ACA_B_right[ind_B[1], ind_B[0]] * CS_EA_B_a[0, element[1]] *  AC_IP_A_left[element[0], 0] * value
            term_42_X_local_occ += X_prefactor * SC_B_AB_ia[element[0], ind_B[0]] * CA_EA_B_left[0, ind_B[1]] *  AC_IP_A_left[element[1], 0 ] * value
            term_14_C_IP += C_prefactor * ACA_A_left[element[0], element[1]] * SC_IP_A_i[ind_B[0], 0] *  CA_EA_B_right[0, ind_B[1]] * value
            term_14_X_local_virt += X_prefactor * CS_A_AB_ia[element[0], ind_B[0]] * AC_IP_A_right[element[1], 0] *  CA_EA_B_right[0, ind_B[1]] * value     
            term_23_C_IP += C_prefactor * ACA_B_left[ind_B[0], ind_B[1]] * SC_IP_B_i[element[0], 0] *  CA_EA_A_right[0, element[1]] * value
            term_23_X_local_virt += X_prefactor * CS_B_BA_ia[ind_B[1], element[1]] * AC_IP_B_right[ind_B[0], 0] *  CA_EA_A_right[0, element[0]] * value    
            term_41_C_1S_IP_arb   += C_prefactor * ACA_A_right[element[0], element[1]] * SC_IP_A_p[ind_B[0], 0] * CA_EA_B_left[0, ind_B[1]] * value
            term_41_X_1S_virt_arb += X_prefactor * CS_A_AB_iq[element[0], ind_B[0]] * AC_IP_A_left[element[1], 0] * CA_EA_B_left[0, ind_B[1]] * value
            term_13_C_1S_EA_arb   += C_prefactor * ACA_A_left[element[0], element[1]] * AC_IP_B_right[ind_B[0], 0] * CS_EA_A_q[0, ind_B[1]] * value
            term_13_X_1S_occ_arb  += X_prefactor * SC_A_BA_pa[ind_B[0], element[1]] * AC_IP_B_right[ind_B[1], 0] * CA_EA_A_right[0, element[0]] * value
            term_21_X_2S += (
            X_prefactor
            * SC_A_BA_ia[ind_B[0], element[0]]
            * CS_B_BA_ia[ind_B[1], element[1]]
            * value
        )
    else:
        #perms = list(distinct_permutations(ind_B))
        #perms_2 = list(distinct_permutations(ind_B))
        a, b = ind_B
        p0 = (a, b)
        p1 = (b, a)
        #if a == c:
        #    b, a, c = indexes
        #    p0 = (a, b, c)
        #    p2 = (b, c, a)
        #    p1 = (b, a, c)
        perms = [p1, p0]
        #if perms != perms_2:
        #    print('problem in calc_term_BBAA_3_coul perms')
        for element in perms:
            term_43_X_2S_IP_EA_arb += X_prefactor * AC_IP_A_left[ind_A[0], :] * CA_EA_B_left[:, element[0]] * SC_IP_B_i[ind_A[1], :] * CS_EA_A_q[:, element[1]] * value

            term_31_C_3S_occ_IP_arb_virt_arb += C_prefactor * SCS_A_BB_iq[element[0], element[1]] * CA_EA_A_left[:, ind_A[0]] * SC_IP_B_p[ind_A[1], :] * value
            term_13_C_3S_IP_virt_occ_arb += C_prefactor * SCS_A_BB_pa[element[0], element[1]] * CA_EA_A_right[:, ind_A[0]] * SC_IP_B_i[ind_A[1], :] * value
            term_41_C_3S_occ_EA_virt_arb += C_prefactor * SCS_A_BB_iq[element[0], element[1]] * CS_EA_B_a[:, ind_A[0]] * AC_IP_A_left[ind_A[1], :] * value
            term_14_C_3S_virt_EA_arb_occ_arb += C_prefactor * SCS_A_BB_pa[element[0], element[1]] * CS_EA_B_q[:, ind_A[0]] * AC_IP_A_right[ind_A[1], :] * value
            term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[element[0], element[1]] * SCS_B_AA_pa[ind_A[0], ind_A[1]] * value
            term_43_C_2S_IP_IP_arb += C_prefactor * CA_EA_B_left[:, element[0]] * SC_IP_B_i[ind_A[0], :] * CA_EA_A_right[:, ind_A[1]] * SC_IP_A_p[element[1], :] * value
            term_31_X_3S_EA_IP_arb_virt_arb += X_prefactor * CS_A_AB_iq[ind_A[0], element[0]] * CS_EA_A_a[:, element[1]] * SC_IP_B_p[ind_A[1], :] * value
            term_13_X_3S_IP_virt_EA_arb += X_prefactor * CS_A_AB_ia[ind_A[0], element[0]] *  CS_EA_A_q[:, element[1]] * SC_IP_B_i[ind_A[1], :] * value
            term_41_X_3S_occ_EA_IP_arb += X_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * CS_EA_B_a[:, ind_A[1]] * SC_IP_A_p[element[1], :] * value
            term_14_X_3S_IP_occ_arb_EA_arb += X_prefactor * SC_A_BA_pa[element[0], ind_A[0]] * CS_EA_B_q[:, ind_A[1]] * SC_IP_A_i[element[1], :] * value
            term_21_X_2S_B_occ_arb_A_virt_arb += X_prefactor * CS_A_AB_iq[ind_A[0], element[0]] * SC_B_AB_pa[ind_A[1], element[1]] * value
            term_43_X_2S_EA_IP_arb += X_prefactor * SC_IP_A_p[element[0], :]    * CS_EA_B_a[:, ind_A[0]] * AC_IP_B_right[element[1], :] * CA_EA_A_right[:, ind_A[1]]   * value
            term_43_C_2S_EA_EA_arb += C_prefactor * AC_IP_A_left[ind_A[1], :]   * CS_EA_B_a[:, ind_A[0]] * AC_IP_B_right[element[0], :] * CS_EA_A_q[:, element[1]]     * value


            term_21_C += C_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * ACA_B_left[element[0], element[1]] * value
            term_12_C += C_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * ACA_B_right[element[0], element[1]] * value
            term_31_C_EA += C_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * CS_EA_A_a[0, element[0]] *  AC_IP_B_left[element[1], 0] * value
            term_31_X_local_occ += X_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * CA_EA_A_left[0, ind_A[1]] *  AC_IP_B_left[element[1], 0 ] * value
            
            term_12_X_2S += X_prefactor * SC_B_AB_ia[ind_A[0], element[0]] * CS_A_AB_ia[ind_A[1], element[1]] * value
            term_42_C_EA += C_prefactor * ACA_B_right[element[1], element[0]] * CS_EA_B_a[0, ind_A[1]] *  AC_IP_A_left[ind_A[0], 0] * value
            term_42_X_local_occ += X_prefactor * SC_B_AB_ia[ind_A[0], element[0]] * CA_EA_B_left[0, element[1]] *  AC_IP_A_left[ind_A[1], 0 ] * value
            
            term_14_C_IP += C_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * SC_IP_A_i[element[0], 0] *  CA_EA_B_right[0, element[1]] * value
            term_14_X_local_virt += X_prefactor * CS_A_AB_ia[ind_A[0], element[0]] * AC_IP_A_right[ind_A[1], 0] *  CA_EA_B_right[0, element[1]] * value
            term_23_C_IP += C_prefactor * ACA_B_left[element[0], element[1]] * SC_IP_B_i[ind_A[0], 0] *  CA_EA_A_right[0, ind_A[1]] * value
            term_23_X_local_virt += X_prefactor * CS_B_BA_ia[element[0], ind_A[0]] * AC_IP_B_right[element[1], 0] *  CA_EA_A_right[0, ind_A[1]] * value
            term_21_X_2S += (
                X_prefactor
                * SC_A_BA_ia[element[0], ind_A[0]]
                * CS_B_BA_ia[element[1], ind_A[1]]
                * value
            )
            term_41_C_1S_IP_arb   += C_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * SC_IP_A_p[element[0], 0] * CA_EA_B_left[0, element[1]] * value
            term_41_X_1S_virt_arb += X_prefactor * CS_A_AB_iq[ind_A[0], element[0]] * AC_IP_A_left[ind_A[1], 0] * CA_EA_B_left[0, element[1]] * value
            term_13_C_1S_EA_arb   += C_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * AC_IP_B_right[element[0], 0] * CS_EA_A_q[0, element[1]] * value
            term_13_X_1S_occ_arb  += X_prefactor * SC_A_BA_pa[element[0], ind_A[1]] * AC_IP_B_right[element[1], 0] * CA_EA_A_right[0, ind_A[0]] * value
    return term_21_C, term_12_C, term_31_X_local_occ, term_31_C_EA, term_14_X_local_virt, term_14_C_IP, term_21_X_2S, term_12_X_2S, term_42_X_local_occ, term_42_C_EA, term_23_X_local_virt, term_23_C_IP, \
            term_41_C_1S_IP_arb, term_41_X_1S_virt_arb, term_13_C_1S_EA_arb, term_13_X_1S_occ_arb, \
            term_31_C_3S_occ_IP_arb_virt_arb, term_13_C_3S_IP_virt_occ_arb, term_41_C_3S_occ_EA_virt_arb, term_14_C_3S_virt_EA_arb_occ_arb, term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, \
            term_43_C_2S_IP_IP_arb, term_31_X_3S_EA_IP_arb_virt_arb, term_13_X_3S_IP_virt_EA_arb, term_41_X_3S_occ_EA_IP_arb , term_14_X_3S_IP_occ_arb_EA_arb , term_21_X_2S_B_occ_arb_A_virt_arb, \
            term_43_C_2S_EA_EA_arb, term_43_X_2S_EA_IP_arb, term_43_X_2S_IP_EA_arb


def calc_term_BBAA_3_exch(indexes, term_21, term_43, term_12, term_34, ACA_right   , ACA_left    , CA_EA_right , AC_IP_right , CA_EA_left  , AC_IP_left  ,  which_frag):
    term_21_X = np.float64(0.)
    term_43_X = np.float64(0.)
    term_43_C = np.float64(0.)
    
    term_12_X = np.float64(0.)
    term_34_X = np.float64(0.)
    term_34_C = np.float64(0.)    

    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)
    #perms = list(distinct_permutations(indexes))
    #perms_2 = list(distinct_permutations(indexes))
    a, b = indexes
    p0 = (a, b)
    p1 = (b, a)
    #if a == c:
    #    b, a, c = indexes
    #    p0 = (a, b, c)
    #    p2 = (b, c, a)
    #    p1 = (b, a, c)
    perms = [p1, p0]
    #if perms != perms_2:
    #    print('problem in calc_term_BBAA_3_coul perms')
    if which_frag == 'A':
        for element in perms:
            term_21_X += X_prefactor * ACA_right[element[0], element[1]]  * term_21
            term_43_X += X_prefactor * AC_IP_left[element[0], 0] * CA_EA_right[0, element[1]] * term_43
            term_43_C += C_prefactor * AC_IP_left[element[0], 0] * CA_EA_right[0, element[1]] * term_43
            
            term_12_X += X_prefactor * ACA_left[element[0], element[1]]  * term_12
            term_34_X += X_prefactor * AC_IP_right[element[0], 0] * CA_EA_left[0, element[1]] * term_34
            term_34_C += C_prefactor * AC_IP_right[element[0], 0] * CA_EA_left[0, element[1]] * term_34  
            
        return term_21_X, term_43_X, term_43_C, term_12_X, term_34_X, term_34_C    
    elif which_frag == 'B':
        for element in perms:
            term_21_X += X_prefactor * ACA_left[element[0], element[1]]  * term_21
            term_43_X += X_prefactor * AC_IP_right[element[0], 0] * CA_EA_left[0, element[1]] * term_43
            term_43_C += C_prefactor * AC_IP_right[element[0], 0] * CA_EA_left[0, element[1]] * term_43  
            
            term_12_X += X_prefactor * ACA_right[element[0], element[1]]  * term_12
            term_34_X += X_prefactor * AC_IP_left[element[0], 0] * CA_EA_right[0, element[1]] * term_34
            term_34_C += C_prefactor * AC_IP_left[element[0], 0] * CA_EA_right[0, element[1]] * term_34              
        return term_21_X, term_43_X, term_43_C, term_12_X, term_34_X, term_34_C
    else:
        raise ValueError("which_frag must be 'A' or 'B'")    
    


def calc_term_BBAA_3_exch_with_Ov(ind_A, ind_B, value, perm: str, SC_A_BA_ia      ,
                                                CS_A_AB_ia       ,
                                                SC_B_AB_ia       ,
                                                CS_B_BA_ia        ,
                                                CS_EA_A_a       ,
                                                SC_IP_A_i       ,
                                                SC_IP_B_i        ,
                                                CS_EA_B_a       ,
                                                SC_A_BA_pa    ,
                                                CS_A_AB_iq    ,
                                                SC_B_AB_pa    ,
                                                CS_B_BA_iq    ,
                                                CS_EA_A_q     ,
                                                SC_IP_B_p     ,
                                                SC_IP_A_p     ,
                                                CS_EA_B_q     ,
                                                ACA_A_right   ,
                                                SCS_A_BB_iq   ,
                                                SCS_A_BB_pa   ,
                                                ACA_B_right   ,
                                                SCS_B_AA_pa   ,
                                                ACA_A_left    ,
                                                ACA_B_left    ,
                                                CA_EA_A_right ,
                                                AC_IP_B_right ,
                                                AC_IP_A_right ,
                                                CA_EA_B_right ,
                                                CA_EA_A_left  ,
                                                AC_IP_B_left  ,
                                                AC_IP_A_left  ,
                                                CA_EA_B_left  , ):
    
    term_13_C_local_virt = np.float64(0.)
    term_13_X_local_virt = np.float64(0.)
    term_31_C_local_occ = np.float64(0.)
    #term_31_X_local_occ = np.float64(0.)
    #term_31_C_EA = np.float64(0.)
    term_31_X_EA = np.float64(0.)    
    
    term_21_C_2S = np.float64(0.)
    term_12_C_2S = np.float64(0.)
    term_43_X = np.float64(0.)
    term_43_C = np.float64(0.)
    term_34_X = np.float64(0.)
    term_34_C = np.float64(0.)
    term_21_X = np.float64(0.)
    term_12_X = np.float64(0.)
    term_14_C_local_virt = np.float64(0.)
    term_14_X_IP = np.float64(0.)      
    #term_41_X_local_virt = np.float64(0.)
    term_41_C_local_occ = np.float64(0.)
    term_41_X_local_occ = np.float64(0.)
    #term_41_C_IP = np.float64(0.)

    term_24_C_local_virt = np.float64(0.)
    term_24_X_local_virt = np.float64(0.)
    term_42_C_local_occ = np.float64(0.)
    term_42_X_EA = np.float64(0.)    
    
    term_32_C_local_occ = np.float64(0.)
    term_32_X_local_occ = np.float64(0.)    
    term_23_C_local_virt = np.float64(0.)
    term_23_X_IP = np.float64(0.)     
    
    term_41_C_1S_virt_arb = np.float64(0.)
    term_41_X_1S_IP_arb   = np.float64(0.)
    term_13_C_1S_occ_arb  = np.float64(0.)
    term_13_X_1S_EA_arb   = np.float64(0.)
    term_31_C_1S_virt_arb = np.float64(0.)
    term_31_X_1S_virt_arb = np.float64(0.) 
    term_14_C_1S_occ_arb  = np.float64(0.)
    term_14_X_1S_occ_arb  = np.float64(0.)
    
    term_31_C_3S_EA_occ_IP_arb = np.float64(0.)
    term_31_X_3S_EA_occ_IP_arb = np.float64(0.)
    term_14_C_3S_IP_virt_EA_arb = np.float64(0.)
    term_14_X_3S_IP_virt_EA_arb = np.float64(0.)
    term_21_C_2S_A_occ_B_occ_arb = np.float64(0.)
    term_21_X_2S_A_occ_B_occ_arb = np.float64(0.)
    term_43_C_2S_EA_IP_arb = np.float64(0.)
    term_43_X_2S_EA_EA_arb = np.float64(0.)
    term_31_X_3S_occ_IP_arb_virt_arb = np.float64(0.)
    term_13_X_3S_IP_virt_occ_arb = np.float64(0.)
    term_41_X_3S_occ_EA_virt_arb = np.float64(0.)
    term_14_X_3S_virt_EA_arb_occ_arb = np.float64(0.)
    term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb = np.float64(0.)
    term_43_X_2S_IP_IP_arb = np.float64(0.)
    term_31_C_3S_EA_IP_arb_virt_arb = np.float64(0.)
    term_13_C_3S_IP_virt_EA_arb = np.float64(0.)
    term_41_C_3S_occ_EA_IP_arb = np.float64(0.)
    term_14_C_3S_IP_occ_arb_EA_arb = np.float64(0.)
    term_21_C_2S_B_occ_arb_A_virt_arb = np.float64(0.)
    term_43_C_2S_IP_EA_arb = np.float64(0.)
    C_prefactor = np.float64(4.)  
    X_prefactor = np.float64(-2.)


    if perm == 'A':
        #perms = list(distinct_permutations(ind_A))
        #perms_2 = list(distinct_permutations(ind_A))
        a, b = ind_A
        p0 = (a, b)
        p1 = (b, a)
        #if a == c:
        #    b, a, c = indexes
        #    p0 = (a, b, c)
        #    p2 = (b, c, a)
        #    p1 = (b, a, c)
        perms = [p1, p0]
        #if perms != perms_2:
        #    print('problem in calc_term_BBAA_3_coul perms')
        for element in perms:
            term_43_C_2S_IP_EA_arb += C_prefactor * AC_IP_A_left[element[0], :] * CA_EA_B_left[:, ind_B[0]] * SC_IP_B_i[element[1], :] * CS_EA_A_q[:, ind_B[1]] * value

            term_31_X_3S_occ_IP_arb_virt_arb += X_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * SC_IP_B_p[element[0], :] * CA_EA_A_left[:, element[1]] * value
            term_13_X_3S_IP_virt_occ_arb += X_prefactor * SCS_A_BB_pa[ind_B[0], ind_B[1]] * SC_IP_B_i[element[0], :] * CA_EA_A_right[:, element[1]] * value
            term_41_X_3S_occ_EA_virt_arb += X_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * AC_IP_A_left[element[0], :] * CS_EA_B_a[:, element[1]] * value
            term_14_X_3S_virt_EA_arb_occ_arb += X_prefactor * SCS_A_BB_pa[ind_B[0], ind_B[1]] * AC_IP_A_right[element[0], :] * CS_EA_B_q[:, element[1]] * value
            term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[ind_B[0], ind_B[1]] * SCS_B_AA_pa[element[0], element[1]] * value
            term_43_X_2S_IP_IP_arb += X_prefactor * CA_EA_B_left[:, ind_B[0]] * SC_IP_B_i[element[0], :] * CA_EA_A_right[:, element[1]] * SC_IP_A_p[ind_B[1], :]  * value
            term_31_C_3S_EA_IP_arb_virt_arb += C_prefactor * CS_A_AB_iq[element[0], ind_B[0]] * CS_EA_A_a[:, ind_B[1]] * SC_IP_B_p[element[1], :] * value
            term_13_C_3S_IP_virt_EA_arb += C_prefactor * CS_A_AB_ia[element[0], ind_B[0]] *  CS_EA_A_q[:, ind_B[1]] * SC_IP_B_i[element[1], :] * value
            term_41_C_3S_occ_EA_IP_arb += C_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * CS_EA_B_a[:, element[1]] * SC_IP_A_p[ind_B[1], :] * value
            term_14_C_3S_IP_occ_arb_EA_arb += C_prefactor * SC_A_BA_pa[ind_B[0], element[0]] * CS_EA_B_q[:, element[1]] * SC_IP_A_i[ind_B[1], :] * value
            term_21_C_2S_B_occ_arb_A_virt_arb += C_prefactor * CS_A_AB_iq[element[0], ind_B[0]] * SC_B_AB_pa[element[1], ind_B[1]] * value
            term_31_C_3S_EA_occ_IP_arb += C_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * CS_EA_A_a[:, ind_B[1]] * SC_IP_B_p[element[1], :] * value
            term_31_X_3S_EA_occ_IP_arb += X_prefactor * SC_A_BA_ia[ind_B[0], element[1]] * CS_EA_A_a[:, ind_B[1]] * SC_IP_B_p[element[0], :] * value
            term_14_C_3S_IP_virt_EA_arb += C_prefactor * CS_A_AB_ia[element[0], ind_B[0]] * CS_EA_B_q[:, element[1]] * SC_IP_A_i[ind_B[1], :] * value
            term_14_X_3S_IP_virt_EA_arb += X_prefactor * CS_A_AB_ia[element[1], ind_B[0]] * CS_EA_B_q[:, element[0]] * SC_IP_A_i[ind_B[1], :] * value
            term_21_C_2S_A_occ_B_occ_arb += C_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * SC_B_AB_pa[element[1], ind_B[1]] * value
            term_21_X_2S_A_occ_B_occ_arb += X_prefactor * SC_A_BA_ia[ind_B[0], element[1]] * SC_B_AB_pa[element[0], ind_B[1]] * value
            term_43_C_2S_EA_IP_arb += C_prefactor * CS_EA_B_a[:, element[0]] * AC_IP_B_right[ind_B[0], :] * CA_EA_A_right[:, element[1]] * SC_IP_A_p[ind_B[1], :] * value
            term_43_X_2S_EA_EA_arb += X_prefactor * CS_EA_B_a[:, element[0]] * AC_IP_B_right[ind_B[0], :] * CS_EA_A_q[:, ind_B[1]] * AC_IP_A_left[element[1], :] * value

            term_41_C_1S_virt_arb += C_prefactor * CS_A_AB_iq[element[0], ind_B[0]] * AC_IP_A_left[element[1], 0] * CA_EA_B_left[0, ind_B[1]] * value
            term_41_X_1S_IP_arb   += X_prefactor * ACA_A_right[element[0], element[1]] * SC_IP_A_p[ind_B[0], 0] * CA_EA_B_left[0, ind_B[1]] * value
            term_13_C_1S_occ_arb  += C_prefactor * SC_A_BA_pa[ind_B[0], element[1]] * AC_IP_B_right[ind_B[1], 0] * CA_EA_A_right[0, element[0]] * value
            term_13_X_1S_EA_arb   += X_prefactor * ACA_A_left[element[0], element[1]] * AC_IP_B_right[ind_B[0], 0] * CS_EA_A_q[0, ind_B[1]] * value
            term_31_C_1S_virt_arb += C_prefactor * CS_A_AB_iq[element[0], ind_B[0]] * AC_IP_B_right[ind_B[1], 0] * CA_EA_A_right[0, element[1]] * value
            term_31_X_1S_virt_arb += X_prefactor * CS_A_AB_iq[element[1], ind_B[0]] * AC_IP_B_right[ind_B[1], 0] * CA_EA_A_right[0, element[0]] * value
            term_14_C_1S_occ_arb  += C_prefactor * SC_A_BA_pa[ind_B[0], element[0]] * AC_IP_A_left[element[1], 0] * CA_EA_B_left[0, ind_B[1]] * value
            term_14_X_1S_occ_arb  += X_prefactor * SC_A_BA_pa[ind_B[0], element[1]] * AC_IP_A_left[element[0], 0] * CA_EA_B_left[0, ind_B[1]] * value
            term_43_X += X_prefactor * AC_IP_A_left[element[0], 0] * CA_EA_A_right[0, element[1]] * AC_IP_B_right[ind_B[0], 0] * CA_EA_B_left[0, ind_B[1]] * value
            term_43_C += C_prefactor * AC_IP_A_left[element[1], 0] * CA_EA_A_right[0, element[0]] * AC_IP_B_right[ind_B[0], 0] * CA_EA_B_left[0, ind_B[1]] * value
            term_34_X += X_prefactor * AC_IP_A_right[element[0], 0] * CA_EA_A_left[0, element[1]] * AC_IP_B_left[ind_B[0], 0] * CA_EA_B_right[0, ind_B[1]] * value
            term_34_C += C_prefactor * AC_IP_A_right[element[1], 0] * CA_EA_A_left[0, element[0]] * AC_IP_B_left[ind_B[0], 0] * CA_EA_B_right[0, ind_B[1]] * value
            term_31_C_local_occ += C_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * CA_EA_A_left[0, element[1]] *  AC_IP_B_left[ind_B[1], 0] * value
            term_31_X_EA += X_prefactor * ACA_A_right[element[0], element[1]] * CS_EA_A_a[0, ind_B[0]] *  AC_IP_B_left[ind_B[1], 0] * value
            term_13_C_local_virt += C_prefactor * CS_A_AB_ia[element[0], ind_B[0]] * CA_EA_A_right[0, element[1]] *  AC_IP_B_right[ind_B[1], 0] * value
            term_13_X_local_virt += X_prefactor * CS_A_AB_ia[element[0], ind_B[0]] * CA_EA_A_right[0, element[1]] *  AC_IP_B_right[ind_B[1], 0] * value   
            term_42_C_local_occ += C_prefactor * SC_B_AB_ia[element[0], ind_B[0]] * CA_EA_B_left[0, ind_B[1]] *  AC_IP_A_left[element[1], 0] * value
            term_42_X_EA += X_prefactor * ACA_B_right[ind_B[1], ind_B[0]] * CS_EA_B_a[0, element[1]] *  AC_IP_A_left[element[0], 0] * value
            term_24_C_local_virt += C_prefactor * CS_B_BA_ia[ind_B[1], element[1]] * CA_EA_B_right[0, ind_B[0]] *  AC_IP_A_right[element[0], 0] * value
            term_24_X_local_virt += X_prefactor * CS_B_BA_ia[ind_B[1], element[1]] * CA_EA_B_right[0, ind_B[0]] *  AC_IP_A_right[element[0], 0] * value                  
            term_21_X += X_prefactor * ACA_A_right[element[0], element[1]] * ACA_B_left[ind_B[0], ind_B[1]] * value
            term_12_X += X_prefactor * ACA_A_left[element[0], element[1]] * ACA_B_right[ind_B[0], ind_B[1]] * value
            term_21_C_2S += C_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * CS_B_BA_ia[ind_B[1], element[1]] * value
            term_12_C_2S += C_prefactor * SC_B_AB_ia[element[0], ind_B[0]] * CS_A_AB_ia[element[1], ind_B[1]] * value
            term_14_C_local_virt += C_prefactor * CS_A_AB_ia[element[0], ind_B[0]] * AC_IP_A_right[element[1], 0] *  CA_EA_B_right[0, ind_B[1]] * value
            
            term_14_X_IP += X_prefactor * ACA_A_left[element[0], element[1]] * SC_IP_A_i[ind_B[0], 0] *  CA_EA_B_right[0, ind_B[1]] * value
            term_41_C_local_occ += C_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * AC_IP_A_left[element[1], 0] *  CA_EA_B_left[0, ind_B[1]] * value
            term_41_X_local_occ += X_prefactor * SC_A_BA_ia[ind_B[0], element[0]] * AC_IP_A_left[element[1], 0] *  CA_EA_B_left[0, ind_B[1]] * value                 
            term_32_C_local_occ += C_prefactor * SC_B_AB_ia[element[0], ind_B[0]] * AC_IP_B_left[ind_B[1], 0] *  CA_EA_A_left[0, element[1]] * value
            term_32_X_local_occ += X_prefactor * SC_B_AB_ia[element[0], ind_B[0]] * AC_IP_B_left[ind_B[1], 0] *  CA_EA_A_left[0, element[1]] * value      
            term_23_C_local_virt += C_prefactor * CS_B_BA_ia[ind_B[1], element[1]] * AC_IP_B_right[ind_B[0], 0] *  CA_EA_A_right[0, element[0]] * value
            term_23_X_IP += X_prefactor * ACA_B_left[ind_B[0], ind_B[1]] * SC_IP_B_i[element[0], 0] *  CA_EA_A_right[0, element[1]] * value

    else:
        #perms = list(distinct_permutations(ind_B))
        #perms_2 = list(distinct_permutations(ind_B))
        a, b = ind_B
        p0 = (a, b)
        p1 = (b, a)
        #if a == c:
        #    b, a, c = indexes
        #    p0 = (a, b, c)
        #    p2 = (b, c, a)
        #    p1 = (b, a, c)
        perms = [p1, p0]
        #if perms != perms_2:
        #    print('problem in calc_term_BBAA_3_coul perms')
        for element in perms:
            term_43_C_2S_IP_EA_arb += C_prefactor * AC_IP_A_left[ind_A[0], :] * CA_EA_B_left[:, element[0]] * SC_IP_B_i[ind_A[1], :] * CS_EA_A_q[:, element[1]] * value

            term_31_X_3S_occ_IP_arb_virt_arb += X_prefactor * SCS_A_BB_iq[element[0], element[1]] * SC_IP_B_p[ind_A[0], :] * CA_EA_A_left[:, ind_A[1]] * value
            term_13_X_3S_IP_virt_occ_arb += X_prefactor * SCS_A_BB_pa[element[0], element[1]] * SC_IP_B_i[ind_A[0], :] * CA_EA_A_right[:, ind_A[1]] * value
            term_41_X_3S_occ_EA_virt_arb += X_prefactor * SCS_A_BB_iq[element[0], element[1]] * AC_IP_A_left[ind_A[0], :] * CS_EA_B_a[:, ind_A[1]] * value
            term_14_X_3S_virt_EA_arb_occ_arb += X_prefactor * SCS_A_BB_pa[element[0], element[1]] * AC_IP_A_right[ind_A[0], :] * CS_EA_B_q[:, ind_A[1]] * value
            term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[element[0], element[1]] * SCS_B_AA_pa[ind_A[0], ind_A[1]] * value
            term_43_X_2S_IP_IP_arb += X_prefactor * CA_EA_B_left[:, element[0]] * SC_IP_B_i[ind_A[0], :] * CA_EA_A_right[:, ind_A[1]] * SC_IP_A_p[element[1], :]  * value
            term_31_C_3S_EA_IP_arb_virt_arb += C_prefactor * CS_A_AB_iq[ind_A[0], element[0]] * CS_EA_A_a[:, element[1]] * SC_IP_B_p[ind_A[1], :] * value
            term_13_C_3S_IP_virt_EA_arb += C_prefactor * CS_A_AB_ia[ind_A[0], element[0]] *  CS_EA_A_q[:, element[1]] * SC_IP_B_i[ind_A[1], :] * value
            term_41_C_3S_occ_EA_IP_arb += C_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * CS_EA_B_a[:, ind_A[1]] * SC_IP_A_p[element[1], :] * value
            term_14_C_3S_IP_occ_arb_EA_arb += C_prefactor * SC_A_BA_pa[element[0], ind_A[0]] * CS_EA_B_q[:, ind_A[1]] * SC_IP_A_i[element[1], :] * value
            term_21_C_2S_B_occ_arb_A_virt_arb += C_prefactor * CS_A_AB_iq[ind_A[0], element[0]] * SC_B_AB_pa[ind_A[1], element[1]] * value
            term_31_C_3S_EA_occ_IP_arb += C_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * CS_EA_A_a[:, element[1]] * SC_IP_B_p[ind_A[1], :] * value
            term_31_X_3S_EA_occ_IP_arb += X_prefactor * SC_A_BA_ia[element[1], ind_A[0]] * CS_EA_A_a[:, element[0]] * SC_IP_B_p[ind_A[1], :] * value
            term_14_C_3S_IP_virt_EA_arb += C_prefactor * CS_A_AB_ia[ind_A[0], element[0]] * CS_EA_B_q[:, ind_A[1]] * SC_IP_A_i[element[1], :] * value
            term_14_X_3S_IP_virt_EA_arb += X_prefactor * CS_A_AB_ia[ind_A[0], element[1]] * CS_EA_B_q[:, ind_A[1]] * SC_IP_A_i[element[0], :] * value
            term_21_C_2S_A_occ_B_occ_arb += C_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * SC_B_AB_pa[ind_A[1], element[1]] * value
            term_21_X_2S_A_occ_B_occ_arb += X_prefactor * SC_A_BA_ia[element[1], ind_A[0]] * SC_B_AB_pa[ind_A[1], element[0]] * value
            term_43_C_2S_EA_IP_arb += C_prefactor * CS_EA_B_a[:, ind_A[0]] * AC_IP_B_right[element[0], :] * CA_EA_A_right[:, ind_A[1]] * SC_IP_A_p[element[1], :] * value
            term_43_X_2S_EA_EA_arb += X_prefactor * CS_EA_B_a[:, ind_A[0]] * AC_IP_B_right[element[0], :] * CS_EA_A_q[:, element[1]] * AC_IP_A_left[ind_A[1], :] * value

            term_41_C_1S_virt_arb += C_prefactor * CS_A_AB_iq[ind_A[0], element[0]] * AC_IP_A_left[ind_A[1], 0] * CA_EA_B_left[0, element[1]] * value
            term_41_X_1S_IP_arb   += X_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * SC_IP_A_p[element[0], 0] * CA_EA_B_left[0, element[1]] * value
            term_13_C_1S_occ_arb  += C_prefactor * SC_A_BA_pa[element[0], ind_A[1]] * AC_IP_B_right[element[1], 0] * CA_EA_A_right[0, ind_A[0]] * value
            term_13_X_1S_EA_arb   += X_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * AC_IP_B_right[element[0], 0] * CS_EA_A_q[0, element[1]] * value
            term_31_C_1S_virt_arb += C_prefactor * CS_A_AB_iq[ind_A[0], element[0]] * AC_IP_B_right[element[1], 0] * CA_EA_A_right[0, ind_A[1]] * value
            term_31_X_1S_virt_arb += X_prefactor * CS_A_AB_iq[ind_A[0], element[1]] * AC_IP_B_right[element[0], 0] * CA_EA_A_right[0, ind_A[1]] * value
            term_14_C_1S_occ_arb  += C_prefactor * SC_A_BA_pa[element[0], ind_A[0]] * AC_IP_A_left[ind_A[1], 0] * CA_EA_B_left[0, element[1]] * value
            term_14_X_1S_occ_arb  += X_prefactor * SC_A_BA_pa[element[1], ind_A[0]] * AC_IP_A_left[ind_A[1], 0] * CA_EA_B_left[0, element[0]] * value
            term_13_C_local_virt += C_prefactor * CS_A_AB_ia[ind_A[0], element[0]] * CA_EA_A_right[0, ind_A[1]] *  AC_IP_B_right[element[1], 0] * value
            term_13_X_local_virt += X_prefactor * CS_A_AB_ia[ind_A[0], element[0]] * CA_EA_A_right[0, ind_A[1]] *  AC_IP_B_right[element[1], 0] * value  
            term_31_C_local_occ += C_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * CA_EA_A_left[0, ind_A[1]] *  AC_IP_B_left[element[1], 0] * value
            term_31_X_EA += X_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * CS_EA_A_a[0, element[0]] *  AC_IP_B_left[element[1], 0] * value
            term_42_C_local_occ += C_prefactor * SC_B_AB_ia[ind_A[0], element[0]] * CA_EA_B_left[0, element[1]] *  AC_IP_A_left[ind_A[1], 0] * value
            term_42_X_EA += X_prefactor * ACA_B_right[element[0], element[1]] * CS_EA_B_a[0, ind_A[0]] *  AC_IP_A_left[ind_A[1], 0] * value
            term_24_C_local_virt += C_prefactor * CS_B_BA_ia[element[1], ind_A[1]] * CA_EA_B_right[0, element[0]] *  AC_IP_A_right[ind_A[0], 0] * value
            term_24_X_local_virt += X_prefactor * CS_B_BA_ia[element[1], ind_A[1]] * CA_EA_B_right[0, element[0]] *  AC_IP_A_right[ind_A[0], 0] * value  
            term_21_X += X_prefactor * ACA_A_right[ind_A[0], ind_A[1]] * ACA_B_left[element[0], element[1]] * value
            term_12_X += X_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * ACA_B_right[element[0], element[1]] * value
            term_14_C_local_virt += C_prefactor * CS_A_AB_ia[ind_A[0], element[0]] * AC_IP_A_right[ind_A[1], 0] *  CA_EA_B_right[0, element[1]] * value
            term_14_X_IP += X_prefactor * ACA_A_left[ind_A[0], ind_A[1]] * SC_IP_A_i[element[0], 0] *  CA_EA_B_right[0, element[1]] * value
            term_41_C_local_occ += C_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * AC_IP_A_left[ind_A[1], 0] *  CA_EA_B_left[0, element[1]] * value
            term_41_X_local_occ += X_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * AC_IP_A_left[ind_A[1], 0] *  CA_EA_B_left[0, element[1]] * value 
            term_43_X += X_prefactor * AC_IP_A_left[ind_A[0], 0] * CA_EA_A_right[0, ind_A[1]] * AC_IP_B_right[element[0], 0] * CA_EA_B_left[0, element[1]] * value
            term_43_C += C_prefactor * AC_IP_A_left[ind_A[0], 0] * CA_EA_A_right[0, ind_A[1]] * AC_IP_B_right[element[1], 0] * CA_EA_B_left[0, element[0]] * value
            term_34_X += X_prefactor * AC_IP_A_right[ind_A[0], 0] * CA_EA_A_left[0, ind_A[1]] * AC_IP_B_left[element[0], 0] * CA_EA_B_right[0, element[1]] * value
            term_34_C += C_prefactor * AC_IP_A_right[ind_A[0], 0] * CA_EA_A_left[0, ind_A[1]] * AC_IP_B_left[element[1], 0] * CA_EA_B_right[0, element[0]] * value
            term_21_C_2S += C_prefactor * SC_A_BA_ia[element[0], ind_A[0]] * CS_B_BA_ia[element[1], ind_A[1]] * value
            term_12_C_2S += C_prefactor * SC_B_AB_ia[ind_A[0], element[0]] * CS_A_AB_ia[ind_A[1], element[1]] * value
            term_32_C_local_occ += C_prefactor * SC_B_AB_ia[ind_A[0], element[0]] * AC_IP_B_left[element[1], 0] *  CA_EA_A_left[0, ind_A[1]] * value
            term_32_X_local_occ += X_prefactor * SC_B_AB_ia[ind_A[0], element[0]] * AC_IP_B_left[element[1], 0] *  CA_EA_A_left[0, ind_A[1]] * value 
            term_23_C_local_virt += C_prefactor * CS_B_BA_ia[element[0], ind_A[0]] * AC_IP_B_right[element[1], 0] *  CA_EA_A_right[0, ind_A[1]] * value
            term_23_X_IP += X_prefactor * ACA_B_left[element[0], element[1]] * SC_IP_B_i[ind_A[0], 0] *  CA_EA_A_right[0, ind_A[1]] * value

    return term_43_X, term_43_C, term_34_X, term_34_C, term_21_X, term_12_X, \
                term_13_C_local_virt, term_13_X_local_virt, term_31_C_local_occ, term_31_X_EA, term_14_C_local_virt, term_41_C_local_occ, term_41_X_local_occ, term_14_X_IP, term_21_C_2S, term_12_C_2S,\
                term_32_C_local_occ, term_32_X_local_occ, term_42_C_local_occ, term_42_X_EA, term_23_C_local_virt, term_23_X_IP, term_24_C_local_virt, term_24_X_local_virt, \
                term_41_C_1S_virt_arb, term_41_X_1S_IP_arb, term_13_C_1S_occ_arb, term_13_X_1S_EA_arb, term_31_C_1S_virt_arb, term_31_X_1S_virt_arb, term_14_C_1S_occ_arb, term_14_X_1S_occ_arb, \
                term_31_C_3S_EA_occ_IP_arb, term_31_X_3S_EA_occ_IP_arb, term_14_C_3S_IP_virt_EA_arb, term_14_X_3S_IP_virt_EA_arb, term_21_C_2S_A_occ_B_occ_arb, term_21_X_2S_A_occ_B_occ_arb, \
                term_43_C_2S_EA_IP_arb, term_43_X_2S_EA_EA_arb, term_31_X_3S_occ_IP_arb_virt_arb, term_13_X_3S_IP_virt_occ_arb, term_41_X_3S_occ_EA_virt_arb, \
                term_14_X_3S_virt_EA_arb_occ_arb, term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, term_43_X_2S_IP_IP_arb, term_31_C_3S_EA_IP_arb_virt_arb, \
                term_13_C_3S_IP_virt_EA_arb, term_41_C_3S_occ_EA_IP_arb, term_14_C_3S_IP_occ_arb_EA_arb, term_21_C_2S_B_occ_arb_A_virt_arb, term_43_C_2S_IP_EA_arb\


def calc_term_BBAA_4_coul(indexes_A, indexes_B, value,SC_A_BA_ia      ,
                                                                    CS_A_AB_ia       ,
                                                                    SC_B_AB_ia       ,
                                                                    CS_B_BA_ia        ,
                                                                    CS_EA_A_a       ,
                                                                    SC_IP_A_i       ,
                                                                    SC_IP_B_i        ,
                                                                    CS_EA_B_a       ,
                                                                    SC_A_BA_pa    ,
                                                                    CS_A_AB_iq    ,
                                                                    SC_B_AB_pa    ,
                                                                    CS_B_BA_iq    ,
                                                                    CS_EA_A_q     ,
                                                                    SC_IP_B_p     ,
                                                                    SC_IP_A_p     ,
                                                                    CS_EA_B_q     ,
                                                                    ACA_A_right   ,
                                                                    SCS_A_BB_iq   ,
                                                                    SCS_A_BB_pa   ,
                                                                    ACA_B_right   ,
                                                                    SCS_B_AA_pa   ,
                                                                    ACA_A_left    ,
                                                                    ACA_B_left    ,
                                                                    CA_EA_A_right ,
                                                                    AC_IP_B_right ,
                                                                    AC_IP_A_right ,
                                                                    CA_EA_B_right ,
                                                                    CA_EA_A_left  ,
                                                                    AC_IP_B_left  ,
                                                                    AC_IP_A_left  ,
                                                                    CA_EA_B_left  , ):
    term_21_C = np.float64(0.)
    term_12_C = np.float64(0.)
    term_21_X_2S = np.float64(0.)
    term_12_X_2S = np.float64(0.)

    #term_31_C_local_virt = np.float64(0.)
    #term_31_X_local_virt = np.float64(0.)
    #term_31_C_local_occ = np.float64(0.)
    term_31_X_local_occ = np.float64(0.)
    term_31_C_EA = np.float64(0.)
    #term_31_X_EA = np.float64(0.)    

    #term_14_C_local_virt = np.float64(0.)
    term_14_X_local_virt = np.float64(0.)
    #term_41_C_local_occ = np.float64(0.)
    #term_41_X_local_occ = np.float64(0.)
    term_14_C_IP = np.float64(0.)
    #term_41_X_IP = np.float64(0.)        

    #term_32_C_local_virt = np.float64(0.)
    term_23_X_local_virt = np.float64(0.)
    term_23_C_IP = np.float64(0.)
    #term_32_C_local_occ = np.float64(0.)
    #term_32_X_local_occ = np.float64(0.)
    #term_32_X_IP = np.float64(0.)           

    term_41_C_1S_IP_arb   = np.float64(0.)
    term_41_X_1S_virt_arb = np.float64(0.)
    term_13_C_1S_EA_arb   = np.float64(0.)
    term_13_X_1S_occ_arb  = np.float64(0.)

    term_42_X_local_occ = np.float64(0.)
    term_42_C_EA = np.float64(0.)

    term_31_C_3S_occ_IP_arb_virt_arb = np.float64(0.)
    term_13_C_3S_IP_virt_occ_arb = np.float64(0.)
    term_41_C_3S_occ_EA_virt_arb = np.float64(0.)
    term_14_C_3S_virt_EA_arb_occ_arb = np.float64(0.)
    term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb = np.float64(0.)
    term_43_C_2S_IP_IP_arb = np.float64(0.)
    term_31_X_3S_EA_IP_arb_virt_arb = np.float64(0.)
    term_13_X_3S_IP_virt_EA_arb = np.float64(0.)
    term_41_X_3S_occ_EA_IP_arb = np.float64(0.)
    term_14_X_3S_IP_occ_arb_EA_arb = np.float64(0.)
    term_21_X_2S_B_occ_arb_A_virt_arb = np.float64(0.)
    term_43_X_2S_EA_IP_arb = np.float64(0.)
    term_43_C_2S_EA_EA_arb = np.float64(0.)
    term_43_X_2S_IP_EA_arb = np.float64(0.)
    #perms_A = list(distinct_permutations(indexes_A))
    #perms_B = list(distinct_permutations(indexes_B))

    #perms_A_2 = list(distinct_permutations(indexes_A))
    a, b = indexes_A
    p0 = (a, b)
    p1 = (b, a)
    perms_A = [p1, p0]

    c, d = indexes_B
    p2 = (c, d)
    p3 = (d, c)
    perms_B = [p3, p2]

    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)


    for element_A in perms_A:
        for element_B in perms_B:
            term_43_X_2S_IP_EA_arb += X_prefactor * AC_IP_A_left[element_A[0], :] * CA_EA_B_left[:, element_B[0]] * SC_IP_B_i[element_A[1], :] * CS_EA_A_q[:, element_B[1]] * value

            term_43_X_2S_EA_IP_arb += X_prefactor * SC_IP_A_p[element_B[0], :]    * CS_EA_B_a[:, element_A[1]] * AC_IP_B_right[element_B[1], :] * CA_EA_A_right[:, element_A[0]] * value
            term_43_C_2S_EA_EA_arb += C_prefactor * AC_IP_A_left[element_A[1], :] * CS_EA_B_a[:, element_A[0]] * AC_IP_B_right[element_B[0], :] * CS_EA_A_q[:, element_B[1]]     * value

            term_31_C_3S_occ_IP_arb_virt_arb += C_prefactor * SCS_A_BB_iq[element_B[0], element_B[1]] * CA_EA_A_left[:, element_A[0]] * SC_IP_B_p[element_A[1], :] * value
            term_13_C_3S_IP_virt_occ_arb += C_prefactor * SCS_A_BB_pa[element_B[0], element_B[1]] * CA_EA_A_right[:, element_A[0]] * SC_IP_B_i[element_A[1], :] * value
            term_41_C_3S_occ_EA_virt_arb += C_prefactor * SCS_A_BB_iq[element_B[0], element_B[1]] * CS_EA_B_a[:, element_A[0]] * AC_IP_A_left[element_A[1], :] * value
            term_14_C_3S_virt_EA_arb_occ_arb += C_prefactor * SCS_A_BB_pa[element_B[0], element_B[1]] * CS_EA_B_q[:, element_A[0]] * AC_IP_A_right[element_A[1], :] * value
            term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[element_B[0], element_B[1]] * SCS_B_AA_pa[element_A[0], element_A[1]] * value
            term_43_C_2S_IP_IP_arb += C_prefactor * CA_EA_B_left[:, element_B[0]] * SC_IP_B_i[element_A[0], :] * CA_EA_A_right[:, element_A[1]] * SC_IP_A_p[element_B[1], :] * value
            term_31_X_3S_EA_IP_arb_virt_arb += X_prefactor * CS_A_AB_iq[element_A[0], element_B[0]] * CS_EA_A_a[:, element_B[1]] * SC_IP_B_p[element_A[1], :] * value
            term_13_X_3S_IP_virt_EA_arb += X_prefactor * CS_A_AB_ia[element_A[0], element_B[0]] *  CS_EA_A_q[:, element_B[1]] * SC_IP_B_i[element_A[1], :] * value
            term_41_X_3S_occ_EA_IP_arb += X_prefactor * SC_A_BA_ia[element_B[0], element_A[0]] * CS_EA_B_a[:, element_A[1]] * SC_IP_A_p[element_B[1], :] * value
            term_14_X_3S_IP_occ_arb_EA_arb += X_prefactor * SC_A_BA_pa[element_B[0], element_A[0]] * CS_EA_B_q[:, element_A[1]] * SC_IP_A_i[element_B[1], :] * value
            term_21_X_2S_B_occ_arb_A_virt_arb += X_prefactor * CS_A_AB_iq[element_A[0], element_B[0]] * SC_B_AB_pa[element_A[1], element_B[1]] * value

            term_21_C += C_prefactor * ACA_A_right[element_A[0], element_A[1]] * ACA_B_left[element_B[0], element_B[1]] * value
            term_12_C += C_prefactor * ACA_A_left[element_A[0], element_A[1]] * ACA_B_right[element_B[0], element_B[1]] * value
            term_31_C_EA += C_prefactor * ACA_A_right[element_A[0], element_A[1]] * CS_EA_A_a[0, element_B[1]] *  AC_IP_B_left[element_B[0], 0] * value
            term_31_X_local_occ += X_prefactor * SC_A_BA_ia[element_B[0], element_A[0]] * CA_EA_A_left[0, element_A[1]] *  AC_IP_B_left[element_B[1], 0 ] * value
            term_42_C_EA += C_prefactor * ACA_B_right[element_B[1], element_B[0]] * CS_EA_B_a[0, element_A[1]] *  AC_IP_A_left[element_A[0], 0] * value
            term_42_X_local_occ += X_prefactor * SC_B_AB_ia[element_A[1], element_B[1]] * CA_EA_B_left[0, element_B[0]] *  AC_IP_A_left[element_A[0], 0 ] * value
            term_21_X_2S += X_prefactor * SC_A_BA_ia[element_B[0], element_A[0]] * CS_B_BA_ia[element_B[1], element_A[1]] * value
            term_12_X_2S += X_prefactor * SC_B_AB_ia[element_A[0], element_B[0]] * CS_A_AB_ia[element_A[1], element_B[1]] * value
            term_14_X_local_virt += X_prefactor * CS_A_AB_ia[element_A[0], element_B[0]] * AC_IP_A_right[element_A[1], 0] *  CA_EA_B_right[0, element_B[1]] * value
            term_14_C_IP += C_prefactor * ACA_A_left[element_A[0], element_A[1]] * SC_IP_A_i[element_B[0], 0] *  CA_EA_B_right[0, element_B[1]] * value
            term_23_X_local_virt += X_prefactor * CS_B_BA_ia[element_B[0], element_A[0]] * AC_IP_B_right[element_B[1], 0] *  CA_EA_A_right[0, element_A[1]] * value
            term_23_C_IP += C_prefactor * ACA_B_left[element_B[0], element_B[1]] * SC_IP_B_i[element_A[0], 0] *  CA_EA_A_right[0, element_A[1]] * value
            term_41_C_1S_IP_arb   += C_prefactor * ACA_A_right[element_A[0], element_A[1]] * SC_IP_A_p[element_B[0], 0] * CA_EA_B_left[0, element_B[1]] * value
            term_41_X_1S_virt_arb += X_prefactor * CS_A_AB_iq[element_A[0], element_B[0]] * AC_IP_A_left[element_A[1], 0] * CA_EA_B_left[0, element_B[1]] * value
            term_13_C_1S_EA_arb   += C_prefactor * ACA_A_left[element_A[0], element_A[1]] * AC_IP_B_right[element_B[0], 0] * CS_EA_A_q[0, element_B[1]] * value
            term_13_X_1S_occ_arb  += X_prefactor * SC_A_BA_pa[element_B[0], element_A[0]] * AC_IP_B_right[element_B[1], 0] * CA_EA_A_right[0, element_A[1]] * value
    return  term_21_C, term_12_C, term_31_X_local_occ, term_31_C_EA, term_14_X_local_virt, term_14_C_IP, term_21_X_2S, term_12_X_2S, term_42_X_local_occ, term_42_C_EA, term_23_X_local_virt, term_23_C_IP, \
                        term_41_C_1S_IP_arb, term_41_X_1S_virt_arb, term_13_C_1S_EA_arb, term_13_X_1S_occ_arb, \
                        term_31_C_3S_occ_IP_arb_virt_arb, term_13_C_3S_IP_virt_occ_arb, term_41_C_3S_occ_EA_virt_arb, term_14_C_3S_virt_EA_arb_occ_arb, term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, \
                        term_43_C_2S_IP_IP_arb, term_31_X_3S_EA_IP_arb_virt_arb, term_13_X_3S_IP_virt_EA_arb, term_41_X_3S_occ_EA_IP_arb , term_14_X_3S_IP_occ_arb_EA_arb , term_21_X_2S_B_occ_arb_A_virt_arb, \
                        term_43_X_2S_EA_IP_arb, term_43_C_2S_EA_EA_arb, term_43_X_2S_IP_EA_arb



def calc_term_BBAA_4_exch(indexes_A, indexes_B, value,  SC_A_BA_ia      ,
                                                                    CS_A_AB_ia       ,
                                                                    SC_B_AB_ia       ,
                                                                    CS_B_BA_ia        ,
                                                                    CS_EA_A_a       ,
                                                                    SC_IP_A_i       ,
                                                                    SC_IP_B_i        ,
                                                                    CS_EA_B_a       ,
                                                                    SC_A_BA_pa    ,
                                                                    CS_A_AB_iq    ,
                                                                    SC_B_AB_pa    ,
                                                                    CS_B_BA_iq    ,
                                                                    CS_EA_A_q     ,
                                                                    SC_IP_B_p     ,
                                                                    SC_IP_A_p     ,
                                                                    CS_EA_B_q     ,
                                                                    ACA_A_right   ,
                                                                    SCS_A_BB_iq   ,
                                                                    SCS_A_BB_pa   ,
                                                                    ACA_B_right   ,
                                                                    SCS_B_AA_pa   ,
                                                                    ACA_A_left    ,
                                                                    ACA_B_left    ,
                                                                    CA_EA_A_right ,
                                                                    AC_IP_B_right ,
                                                                    AC_IP_A_right ,
                                                                    CA_EA_B_right ,
                                                                    CA_EA_A_left  ,
                                                                    AC_IP_B_left  ,
                                                                    AC_IP_A_left  ,
                                                                    CA_EA_B_left  , ):
    term_21_X = np.float64(0.)
    term_43_X = np.float64(0.)
    term_43_C = np.float64(0.)

    term_12_X = np.float64(0.)
    term_34_X = np.float64(0.)
    term_34_C = np.float64(0.)

    term_13_C_local_virt = np.float64(0.)
    term_13_X_local_virt = np.float64(0.)
    term_31_C_local_occ = np.float64(0.)
    term_31_X_EA = np.float64(0.)  

    term_21_C_2S = np.float64(0.)
    term_12_C_2S = np.float64(0.)

    term_14_C_local_virt = np.float64(0.)
    term_14_X_IP = np.float64(0.)        
    term_41_C_local_occ = np.float64(0.)
    term_41_X_local_occ = np.float64(0.)

    term_32_C_local_occ = np.float64(0.)
    term_32_X_local_occ = np.float64(0.)
    term_23_C_local_virt = np.float64(0.)
    term_23_X_IP = np.float64(0.)     

    term_24_C_local_virt = np.float64(0.)
    term_24_X_local_virt = np.float64(0.)    
    term_42_C_local_occ = np.float64(0.)
    term_42_X_EA = np.float64(0.)  

    term_41_C_1S_virt_arb = np.float64(0.)
    term_41_X_1S_IP_arb   = np.float64(0.)
    term_13_C_1S_occ_arb  = np.float64(0.)
    term_13_X_1S_EA_arb   = np.float64(0.)
    term_31_C_1S_virt_arb = np.float64(0.)
    term_31_X_1S_virt_arb = np.float64(0.) 
    term_14_C_1S_occ_arb  = np.float64(0.)
    term_14_X_1S_occ_arb  = np.float64(0.)

    term_31_C_3S_EA_occ_IP_arb = np.float64(0.)
    term_31_X_3S_EA_occ_IP_arb = np.float64(0.)
    term_14_C_3S_IP_virt_EA_arb = np.float64(0.)
    term_14_X_3S_IP_virt_EA_arb = np.float64(0.)
    term_21_C_2S_A_occ_B_occ_arb = np.float64(0.)
    term_21_X_2S_A_occ_B_occ_arb = np.float64(0.)
    term_43_C_2S_EA_IP_arb = np.float64(0.)
    term_43_X_2S_EA_EA_arb = np.float64(0.)
    term_31_X_3S_occ_IP_arb_virt_arb = np.float64(0.)
    term_13_X_3S_IP_virt_occ_arb = np.float64(0.)
    term_41_X_3S_occ_EA_virt_arb = np.float64(0.)
    term_14_X_3S_virt_EA_arb_occ_arb = np.float64(0.)
    term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb = np.float64(0.)
    term_43_X_2S_IP_IP_arb = np.float64(0.)
    term_31_C_3S_EA_IP_arb_virt_arb = np.float64(0.)
    term_13_C_3S_IP_virt_EA_arb = np.float64(0.)
    term_41_C_3S_occ_EA_IP_arb = np.float64(0.)
    term_14_C_3S_IP_occ_arb_EA_arb = np.float64(0.)
    term_21_C_2S_B_occ_arb_A_virt_arb = np.float64(0.)
    term_43_C_2S_IP_EA_arb = np.float64(0.)
    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)

    a, b = indexes_A
    p0 = (a, b)
    p1 = (b, a)
    perms_A = [p1, p0]

    c, d = indexes_B
    p2 = (c, d)
    p3 = (d, c)
    perms_B = [p3, p2]

    for idx in range(len(perms_B)):
        term_43_C_2S_IP_EA_arb += C_prefactor * AC_IP_A_left[perms_A[idx][0], :] * CA_EA_B_left[:, perms_B[idx][0]] * SC_IP_B_i[perms_A[idx][1], :] * CS_EA_A_q[:, perms_B[idx][1]] * value

        term_31_X_3S_occ_IP_arb_virt_arb += X_prefactor * SCS_A_BB_iq[perms_B[idx][0], perms_B[idx][1]] * SC_IP_B_p[perms_A[idx][0], :] * CA_EA_A_left[:, perms_A[idx][1]] * value
        term_13_X_3S_IP_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms_B[idx][0], perms_B[idx][1]] * SC_IP_B_i[perms_A[idx][0], :] * CA_EA_A_right[:, perms_A[idx][1]] * value
        term_41_X_3S_occ_EA_virt_arb += X_prefactor * SCS_A_BB_iq[perms_B[idx][0], perms_B[idx][1]] * AC_IP_A_left[perms_A[idx][0], :] * CS_EA_B_a[:, perms_A[idx][1]] * value
        term_14_X_3S_virt_EA_arb_occ_arb += X_prefactor * SCS_A_BB_pa[perms_B[idx][0], perms_B[idx][1]] * AC_IP_A_right[perms_A[idx][0], :] * CS_EA_B_q[:, perms_A[idx][1]] * value
        term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[perms_B[idx][0], perms_B[idx][1]] * SCS_B_AA_pa[perms_A[idx][0], perms_A[idx][1]] * value
        term_43_X_2S_IP_IP_arb += X_prefactor * SC_IP_A_p[perms_B[idx][0], :]  * CA_EA_B_left[:, perms_B[idx][1]] * SC_IP_B_i[perms_A[idx][0], :] * CA_EA_A_right[:, perms_A[idx][1]] * value
        term_31_C_3S_EA_IP_arb_virt_arb += C_prefactor * CS_A_AB_iq[perms_A[idx][0], perms_B[idx][0]] * CS_EA_A_a[:, perms_B[idx][1]] * SC_IP_B_p[perms_A[idx][1], :] * value
        term_13_C_3S_IP_virt_EA_arb += C_prefactor * CS_A_AB_ia[perms_A[idx][0], perms_B[idx][0]] *  CS_EA_A_q[:, perms_B[idx][1]] * SC_IP_B_i[perms_A[idx][1], :] * value
        term_41_C_3S_occ_EA_IP_arb += C_prefactor * SC_A_BA_ia[perms_B[idx][0], perms_A[idx][0]] * CS_EA_B_a[:, perms_A[idx][1]] * SC_IP_A_p[perms_B[idx][1], :] * value
        term_14_C_3S_IP_occ_arb_EA_arb += C_prefactor * SC_A_BA_pa[perms_B[idx][0], perms_A[idx][0]] * CS_EA_B_q[:, perms_A[idx][1]] * SC_IP_A_i[perms_B[idx][1], :] * value
        term_21_C_2S_B_occ_arb_A_virt_arb += C_prefactor * CS_A_AB_iq[perms_A[idx][0], perms_B[idx][0]] * SC_B_AB_pa[perms_A[idx][1], perms_B[idx][1]] * value
        term_31_C_3S_EA_occ_IP_arb += C_prefactor * SC_A_BA_ia[perms_B[idx][0], perms_A[idx][0]] * CS_EA_A_a[:, perms_B[idx][1]] * SC_IP_B_p[perms_A[idx][1], :] * value
        term_31_X_3S_EA_occ_IP_arb += X_prefactor * SC_A_BA_ia[perms_B[idx][0], perms_A[idx][1]] * CS_EA_A_a[:, perms_B[idx][1]] * SC_IP_B_p[perms_A[idx][0], :] * value
        term_14_C_3S_IP_virt_EA_arb += C_prefactor * CS_A_AB_ia[perms_A[idx][0], perms_B[idx][0]] * CS_EA_B_q[:, perms_A[idx][1]] * SC_IP_A_i[perms_B[idx][1], :] * value
        term_14_X_3S_IP_virt_EA_arb += X_prefactor * CS_A_AB_ia[perms_A[idx][0], perms_B[idx][1]] * CS_EA_B_q[:, perms_A[idx][1]] * SC_IP_A_i[perms_B[idx][0], :] * value
        term_21_C_2S_A_occ_B_occ_arb += C_prefactor * SC_A_BA_ia[perms_B[idx][0], perms_A[idx][0]] * SC_B_AB_pa[perms_A[idx][1], perms_B[idx][1]] * value
        term_21_X_2S_A_occ_B_occ_arb += X_prefactor * SC_A_BA_ia[perms_B[idx][1], perms_A[idx][0]] * SC_B_AB_pa[perms_A[idx][1], perms_B[idx][0]] * value
        term_43_C_2S_EA_IP_arb += C_prefactor * SC_IP_A_p[perms_B[idx][0], :]    * CS_EA_B_a[:, perms_A[idx][0]] * AC_IP_B_right[perms_B[idx][1], :] * CA_EA_A_right[:, perms_A[idx][1]] * value
        term_43_X_2S_EA_EA_arb += X_prefactor * AC_IP_A_left[perms_A[idx][0], :] * CS_EA_B_a[:, perms_A[idx][1]] * AC_IP_B_right[perms_B[idx][0], :] * CS_EA_A_q[:, perms_B[idx][1]]     * value


        term_13_C_local_virt += C_prefactor * CS_A_AB_ia[perms_A[idx][0], perms_B[idx][0]] * CA_EA_A_right[0, perms_A[idx][1]] *  AC_IP_B_right[perms_B[idx][1], 0] * value
        term_13_X_local_virt += X_prefactor * CS_A_AB_ia[perms_A[idx][0], perms_B[idx][1]] * CA_EA_A_right[0, perms_A[idx][1]] *  AC_IP_B_right[perms_B[idx][0], 0] * value
        term_31_C_local_occ += C_prefactor * SC_A_BA_ia[perms_B[idx][0], perms_A[idx][0]] * CA_EA_A_left[0, perms_A[idx][1]] *  AC_IP_B_left[perms_B[idx][1], 0] * value
        term_31_X_EA += X_prefactor * ACA_A_right[perms_A[idx][0], perms_A[idx][1]] * CS_EA_A_a[0, perms_B[idx][1]] *  AC_IP_B_left[perms_B[idx][0], 0] * value
        term_21_C_2S += C_prefactor * SC_A_BA_ia[perms_B[idx][0], perms_A[idx][0]] * CS_B_BA_ia[perms_B[idx][1], perms_A[idx][1]] * value
        term_12_C_2S += C_prefactor * SC_B_AB_ia[perms_A[idx][0], perms_B[idx][0]] * CS_A_AB_ia[perms_A[idx][1], perms_B[idx][1]] * value
        term_42_C_local_occ += C_prefactor * SC_B_AB_ia[perms_A[idx][1], perms_B[idx][1]] * CA_EA_B_left[0, perms_B[idx][0]] *  AC_IP_A_left[perms_A[idx][0], 0] * value
        term_42_X_EA += X_prefactor * ACA_B_right[perms_B[idx][0], perms_B[idx][1]] * CS_EA_B_a[0, perms_A[idx][1]] *  AC_IP_A_left[perms_A[idx][0], 0] * value            
        term_24_C_local_virt += C_prefactor * CS_B_BA_ia[perms_B[idx][1], perms_A[idx][1]] * CA_EA_B_right[0, perms_B[idx][0]] *  AC_IP_A_right[perms_A[idx][0], 0] * value
        term_24_X_local_virt += X_prefactor * CS_B_BA_ia[perms_B[idx][1], perms_A[idx][0]] * CA_EA_B_right[0, perms_B[idx][0]] *  AC_IP_A_right[perms_A[idx][1], 0] * value
        term_14_C_local_virt += C_prefactor * CS_A_AB_ia[perms_A[idx][0], perms_B[idx][0]] * AC_IP_A_right[perms_A[idx][1], 0] *  CA_EA_B_right[0, perms_B[idx][1]] * value
        term_14_X_IP += X_prefactor * ACA_A_left[perms_A[idx][0], perms_A[idx][1]] * SC_IP_A_i[perms_B[idx][0], 0] *  CA_EA_B_right[0, perms_B[idx][1]] * value
        term_41_C_local_occ += C_prefactor * SC_A_BA_ia[perms_B[idx][0], perms_A[idx][0]] * AC_IP_A_left[perms_A[idx][1], 0] *  CA_EA_B_left[0, perms_B[idx][1]] * value
        term_41_X_local_occ += X_prefactor * SC_A_BA_ia[perms_B[idx][0], perms_A[idx][1]] * AC_IP_A_left[perms_A[idx][0], 0] *  CA_EA_B_left[0, perms_B[idx][1]] * value
        term_32_C_local_occ += C_prefactor * SC_B_AB_ia[perms_A[idx][0], perms_B[idx][0]] * AC_IP_B_left[perms_B[idx][1], 0] *  CA_EA_A_left[0, perms_A[idx][1]] * value
        term_32_X_local_occ += X_prefactor * SC_B_AB_ia[perms_A[idx][0], perms_B[idx][1]] * AC_IP_B_left[perms_B[idx][0], 0] *  CA_EA_A_left[0, perms_A[idx][1]] * value
        term_23_C_local_virt += C_prefactor * CS_B_BA_ia[perms_B[idx][1], perms_A[idx][1]] * AC_IP_B_right[perms_B[idx][0], 0] *  CA_EA_A_right[0, perms_A[idx][0]] * value
        term_23_X_IP += X_prefactor * ACA_B_left[perms_B[idx][0], perms_B[idx][1]] * SC_IP_B_i[perms_A[idx][0], 0] *  CA_EA_A_right[0, perms_A[idx][1]] * value
        term_43_X += X_prefactor * AC_IP_A_left[perms_A[idx][0], 0] * CA_EA_A_right[0, perms_A[idx][1]] * AC_IP_B_right[perms_B[idx][0], 0] * CA_EA_B_left[0, perms_B[idx][1]] * value
        term_43_C += C_prefactor * AC_IP_A_left[perms_A[idx][1], 0] * CA_EA_A_right[0, perms_A[idx][0]] * AC_IP_B_right[perms_B[idx][0], 0] * CA_EA_B_left[0, perms_B[idx][1]] * value
        term_34_X += X_prefactor * AC_IP_A_right[perms_A[idx][0], 0] * CA_EA_A_left[0, perms_A[idx][1]] * AC_IP_B_left[perms_B[idx][0], 0] * CA_EA_B_right[0, perms_B[idx][1]] * value
        term_34_C += C_prefactor * AC_IP_A_right[perms_A[idx][1], 0] * CA_EA_A_left[0, perms_A[idx][0]] * AC_IP_B_left[perms_B[idx][0], 0] * CA_EA_B_right[0, perms_B[idx][1]] * value
        term_21_X += X_prefactor * ACA_A_right[perms_A[idx][0], perms_A[idx][1]] * ACA_B_left[perms_B[idx][0], perms_B[idx][1]] * value
        term_12_X += X_prefactor * ACA_A_left[perms_A[idx][0], perms_A[idx][1]] * ACA_B_right[perms_B[idx][0], perms_B[idx][1]] * value
        term_41_C_1S_virt_arb += C_prefactor * CS_A_AB_iq[perms_A[idx][0], perms_B[idx][0]] * AC_IP_A_left[perms_A[idx][1], 0] * CA_EA_B_left[0, perms_B[idx][1]] * value
        term_41_X_1S_IP_arb   += X_prefactor * ACA_A_right[perms_A[idx][0], perms_A[idx][1]] * SC_IP_A_p[perms_B[idx][0], 0] * CA_EA_B_left[0, perms_B[idx][1]] * value
        term_13_C_1S_occ_arb  += C_prefactor * SC_A_BA_pa[perms_B[idx][0], perms_A[idx][0]] * AC_IP_B_right[perms_B[idx][1], 0] * CA_EA_A_right[0, perms_A[idx][1]] * value
        term_13_X_1S_EA_arb   += X_prefactor * ACA_A_left[perms_A[idx][0], perms_A[idx][1]] * AC_IP_B_right[perms_B[idx][0], 0] * CS_EA_A_q[0, perms_B[idx][1]] * value
        term_31_C_1S_virt_arb += C_prefactor * CS_A_AB_iq[perms_A[idx][0], perms_B[idx][0]] * AC_IP_B_left[perms_B[idx][1], 0] * CA_EA_A_left[0, perms_A[idx][1]] * value
        term_31_X_1S_virt_arb += X_prefactor * CS_A_AB_iq[perms_A[idx][0], perms_B[idx][1]] * AC_IP_B_left[perms_B[idx][0], 0] * CA_EA_A_left[0, perms_A[idx][1]] * value
        term_14_C_1S_occ_arb  += C_prefactor * SC_A_BA_pa[perms_B[idx][0], perms_A[idx][0]] * AC_IP_A_right[perms_A[idx][1], 0] * CA_EA_B_right[0, perms_B[idx][1]] * value
        term_14_X_1S_occ_arb  += X_prefactor * SC_A_BA_pa[perms_B[idx][0], perms_A[idx][1]] * AC_IP_A_right[perms_A[idx][0], 0] * CA_EA_B_right[0, perms_B[idx][1]] * value

    return term_21_X, term_43_X, term_43_C, term_12_X, term_34_X, term_34_C, term_13_C_local_virt, term_13_X_local_virt, term_31_C_local_occ, term_31_X_EA, term_14_C_local_virt, term_14_X_IP, term_41_C_local_occ, term_41_X_local_occ, term_21_C_2S, term_12_C_2S,\
                term_32_C_local_occ, term_32_X_local_occ, term_42_C_local_occ, term_42_X_EA, term_23_C_local_virt, term_23_X_IP, term_24_C_local_virt, term_24_X_local_virt,    \
                term_41_C_1S_virt_arb, term_41_X_1S_IP_arb, term_13_C_1S_occ_arb, term_13_X_1S_EA_arb, term_31_C_1S_virt_arb, term_31_X_1S_virt_arb, term_14_C_1S_occ_arb, term_14_X_1S_occ_arb, \
                term_31_C_3S_EA_occ_IP_arb, term_31_X_3S_EA_occ_IP_arb, term_14_C_3S_IP_virt_EA_arb, term_14_X_3S_IP_virt_EA_arb, term_21_C_2S_A_occ_B_occ_arb, term_21_X_2S_A_occ_B_occ_arb, \
                term_43_C_2S_EA_IP_arb, term_43_X_2S_EA_EA_arb, term_31_X_3S_occ_IP_arb_virt_arb, term_13_X_3S_IP_virt_occ_arb, term_41_X_3S_occ_EA_virt_arb, \
                term_14_X_3S_virt_EA_arb_occ_arb, term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, term_43_X_2S_IP_IP_arb, term_31_C_3S_EA_IP_arb_virt_arb, \
                term_13_C_3S_IP_virt_EA_arb, term_41_C_3S_occ_EA_IP_arb, term_14_C_3S_IP_occ_arb_EA_arb, term_21_C_2S_B_occ_arb_A_virt_arb, term_43_C_2S_IP_EA_arb


def   calc_BBAA(twoelint_block, 
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
CA_EA_B_left , NBAS_A 
                          ):
    """Calculation for BAAA type."""
    # Iterate through the categories in the "LOCAL" key

    fc_21_coul = np.float64(0.)
    fc_21_exch = np.float64(0.)
    
    fc_12_coul = np.float64(0.)
    fc_12_exch = np.float64(0.)
    fc_34_coul = np.float64(0.)
    fc_34_exch = np.float64(0.)
    
    fc_21_coul_2S = np.float64(0.)
    fc_21_exch_2S = np.float64(0.)    
    fc_12_coul_2S = np.float64(0.)
    fc_12_exch_2S = np.float64(0.)      
    fc_43_coul = np.float64(0.)
    fc_43_exch = np.float64(0.)
    
    fc_31_coul_local_occ = np.float64(0.)
    fc_31_exch_local_occ = np.float64(0.)
    fc_13_coul_local_virt = np.float64(0.)
    fc_13_exch_local_virt = np.float64(0.)
    fc_31_coul_EA = np.float64(0.)
    fc_31_exch_EA = np.float64(0.)    
    fc_13_coul_1S_occ_arb = np.float64(0.) #! 21 X
    fc_13_exch_1S_occ_arb = np.float64(0.) #! 21 C
    fc_13_coul_1S_EA_arb = np.float64(0.) #! 21 coul
    fc_13_exch_1S_EA_arb = np.float64(0.) #! 21 X
    fc_31_coul_1S_virt_arb = np.float64(0.) #! AABB
    fc_31_exch_1S_virt_arb = np.float64(0.) #! AABB
    
    fc_41_coul_local_occ = np.float64(0.)
    fc_41_exch_local_occ = np.float64(0.)
    fc_14_coul_local_virt = np.float64(0.)
    fc_14_exch_local_virt = np.float64(0.)
    fc_14_coul_IP = np.float64(0.)
    fc_14_exch_IP = np.float64(0.)
    fc_14_coul_1S_occ_arb  = np.float64(0.) #! AABB
    fc_14_exch_1S_occ_arb  = np.float64(0.) #! AABB
    fc_41_coul_1S_virt_arb = np.float64(0.) #! 21 X
    fc_41_exch_1S_virt_arb = np.float64(0.) #! 21 coul 
    fc_41_coul_1S_IP_arb   = np.float64(0.) #! 21 coul
    fc_41_exch_1S_IP_arb   = np.float64(0.) #! 21 X  

    fc_32_coul_local_occ = np.float64(0.)
    fc_32_exch_local_occ = np.float64(0.)
    fc_23_coul_local_virt = np.float64(0.)
    fc_23_exch_local_virt = np.float64(0.)   
    fc_23_coul_IP = np.float64(0.)
    fc_23_exch_IP = np.float64(0.)    
    
    fc_24_coul_local_virt = np.float64(0.)
    fc_24_exch_local_virt = np.float64(0.)    
    fc_42_coul_local_occ = np.float64(0.)
    fc_42_exch_local_occ = np.float64(0.)
    fc_42_coul_EA = np.float64(0.)
    fc_42_exch_EA = np.float64(0.)   
    
    fc_31_coul_3S_occ_IP_arb_virt_arb = np.float64(0.)
    fc_31_exch_3S_occ_IP_arb_virt_arb = np.float64(0.)
    fc_13_coul_3S_IP_virt_occ_arb = np.float64(0.)
    fc_13_exch_3S_IP_virt_occ_arb = np.float64(0.)
    fc_41_coul_3S_occ_EA_virt_arb = np.float64(0.)
    fc_41_exch_3S_occ_EA_virt_arb = np.float64(0.)
    fc_14_coul_3S_virt_EA_arb_occ_arb = np.float64(0.)
    fc_14_exch_3S_virt_EA_arb_occ_arb = np.float64(0.)
    fc_21_coul_4S_A_occ_B_virt_A_virt_arb_B_occ_arb = np.float64(0.)
    fc_21_exch_4S_A_occ_B_virt_A_virt_arb_B_occ_arb = np.float64(0.)
    fc_43_coul_2S_IP_IP_arb = np.float64(0.)
    fc_43_exch_2S_IP_IP_arb = np.float64(0.)
    fc_31_coul_3S_EA_IP_arb_virt_arb = np.float64(0.)
    fc_31_exch_3S_EA_IP_arb_virt_arb = np.float64(0.)
    fc_13_coul_3S_IP_virt_EA_arb = np.float64(0.)
    fc_13_exch_3S_IP_virt_EA_arb = np.float64(0.)
    fc_41_coul_3S_occ_EA_IP_arb = np.float64(0.)
    fc_41_exch_3S_occ_EA_IP_arb = np.float64(0.)
    fc_14_coul_3S_IP_occ_arb_EA_arb = np.float64(0.)
    fc_14_exch_3S_IP_occ_arb_EA_arb = np.float64(0.)
    fc_21_coul_2S_B_occ_arb_A_virt_arb = np.float64(0.)
    fc_21_exch_2S_B_occ_arb_A_virt_arb = np.float64(0.)
    fc_31_coul_3S_EA_occ_IP_arb = np.float64(0.)
    fc_31_exch_3S_EA_occ_IP_arb = np.float64(0.)
    fc_14_coul_3S_IP_virt_EA_arb = np.float64(0.)
    fc_14_exch_3S_IP_virt_EA_arb = np.float64(0.)
    fc_21_coul_2S_A_occ_B_occ_arb = np.float64(0.)
    fc_21_exch_2S_A_occ_B_occ_arb = np.float64(0.)
    fc_43_coul_2S_EA_IP_arb = np.float64(0.)
    fc_43_exch_2S_EA_IP_arb = np.float64(0.)
    fc_43_coul_2S_EA_EA_arb = np.float64(0.)
    fc_43_exch_2S_EA_EA_arb = np.float64(0.)
    fc_43_coul_2S_IP_EA_arb = np.float64(0.)
    fc_43_exch_2S_IP_EA_arb = np.float64(0.)


    is_Ov = True


    for unique_count, twoelint_subblock in twoelint_block.items():
        if unique_count == 4:
            for row in twoelint_subblock:
                i = int(row[0])
                j = int(row[1])
                k = int(row[2])
                l = int(row[3])
                if j > NBAS_A:
                    if is_Ov:
                        ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)

                        term_21_C, term_12_C, term_31_X_local_occ, term_31_C_EA, term_14_X_local_virt, term_14_C_IP, term_21_X_2S, term_12_X_2S, term_42_X_local_occ, term_42_C_EA, term_23_X_local_virt, term_23_C_IP, \
                        term_41_C_1S_IP_arb, term_41_X_1S_virt_arb, term_13_C_1S_EA_arb, term_13_X_1S_occ_arb, \
                        term_31_C_3S_occ_IP_arb_virt_arb, term_13_C_3S_IP_virt_occ_arb, term_41_C_3S_occ_EA_virt_arb, term_14_C_3S_virt_EA_arb_occ_arb, term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, \
                        term_43_C_2S_IP_IP_arb, term_31_X_3S_EA_IP_arb_virt_arb, term_13_X_3S_IP_virt_EA_arb, term_41_X_3S_occ_EA_IP_arb , term_14_X_3S_IP_occ_arb_EA_arb , term_21_X_2S_B_occ_arb_A_virt_arb, \
                        term_43_X_2S_EA_IP_arb, term_43_C_2S_EA_EA_arb, term_43_X_2S_IP_EA_arb \
                        = calc_term_BBAA_4_coul(ind_A, ind_B, value,SC_A_BA_ia      ,
                                                                    CS_A_AB_ia       ,
                                                                    SC_B_AB_ia       ,
                                                                    CS_B_BA_ia        ,
                                                                    CS_EA_A_a       ,
                                                                    SC_IP_A_i       ,
                                                                    SC_IP_B_i        ,
                                                                    CS_EA_B_a       ,
                                                                    SC_A_BA_pa    ,
                                                                    CS_A_AB_iq    ,
                                                                    SC_B_AB_pa    ,
                                                                    CS_B_BA_iq    ,
                                                                    CS_EA_A_q     ,
                                                                    SC_IP_B_p     ,
                                                                    SC_IP_A_p     ,
                                                                    CS_EA_B_q     ,
                                                                    ACA_A_right   ,
                                                                    SCS_A_BB_iq   ,
                                                                    SCS_A_BB_pa   ,
                                                                    ACA_B_right   ,
                                                                    SCS_B_AA_pa   ,
                                                                    ACA_A_left    ,
                                                                    ACA_B_left    ,
                                                                    CA_EA_A_right ,
                                                                    AC_IP_B_right ,
                                                                    AC_IP_A_right ,
                                                                    CA_EA_B_right ,
                                                                    CA_EA_A_left  ,
                                                                    AC_IP_B_left  ,
                                                                    AC_IP_A_left  ,
                                                                    CA_EA_B_left  , )#, twoel_AO_AAAB_canonical, ind_B[0], value)
                        fc_43_exch_2S_EA_IP_arb += term_43_X_2S_EA_IP_arb
                        fc_43_coul_2S_EA_EA_arb += term_43_C_2S_EA_EA_arb
                        fc_31_coul_3S_occ_IP_arb_virt_arb += term_31_C_3S_occ_IP_arb_virt_arb
                        fc_13_coul_3S_IP_virt_occ_arb += term_13_C_3S_IP_virt_occ_arb
                        fc_41_coul_3S_occ_EA_virt_arb += term_41_C_3S_occ_EA_virt_arb
                        fc_14_coul_3S_virt_EA_arb_occ_arb += term_14_C_3S_virt_EA_arb_occ_arb
                        fc_21_coul_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb
                        fc_43_coul_2S_IP_IP_arb += term_43_C_2S_IP_IP_arb 
                        fc_31_exch_3S_EA_IP_arb_virt_arb += term_31_X_3S_EA_IP_arb_virt_arb 
                        fc_13_exch_3S_IP_virt_EA_arb += term_13_X_3S_IP_virt_EA_arb 
                        fc_41_exch_3S_occ_EA_IP_arb += term_41_X_3S_occ_EA_IP_arb 
                        fc_14_exch_3S_IP_occ_arb_EA_arb += term_14_X_3S_IP_occ_arb_EA_arb 
                        fc_21_exch_2S_B_occ_arb_A_virt_arb += term_21_X_2S_B_occ_arb_A_virt_arb
                        fc_43_exch_2S_IP_EA_arb += term_43_X_2S_IP_EA_arb

                        fc_12_coul += term_12_C
                        
                        fc_41_coul_1S_IP_arb   += term_41_C_1S_IP_arb  
                        fc_41_exch_1S_virt_arb += term_41_X_1S_virt_arb
                        fc_13_coul_1S_EA_arb   += term_13_C_1S_EA_arb  
                        fc_13_exch_1S_occ_arb  += term_13_X_1S_occ_arb 


                        fc_42_coul_EA += term_42_C_EA
                        fc_42_exch_local_occ += term_42_X_local_occ
                        
                        fc_21_exch_2S += term_21_X_2S
                        fc_12_exch_2S += term_12_X_2S
                        
                        fc_31_exch_local_occ  += term_31_X_local_occ
                        fc_31_coul_EA += term_31_C_EA
                        fc_14_exch_local_virt += term_14_X_local_virt
                        fc_14_coul_IP += term_14_C_IP                      
                        
                        fc_23_exch_local_virt += term_23_X_local_virt
                        fc_23_coul_IP += term_23_C_IP                             
                        
                        fc_21_coul += term_21_C
                    else:
                        print('No ov is not implemented for BBAA_4')
                            
                        #ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                        #term_21_C = calc_term_BBAA_4_coul(ind_A, ind_B, value, AC_no_S=AC_no_S)#, twoel_AO_AAAB_canonical, ind_B[0], value)
                        #fc_21_coul += term_21_C
                
                else:
                    
                    if i < k:
                        row[0], row[1], row[2], row[3] = row[2], row[3], row[0], row[1]
                    #if row[1] < row[3]:
                    #    #! careful, this operation is somewhat illegal (it woulb be a different integral) 
                    #    #However, this makes the coding easier
                    #    is_type_1 = True
                    #    #row[1], row[3] = row[3], row[1]
                    ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                    if is_Ov:
                        term_21_X, term_43_X, term_43_C, term_12_X, term_34_X, term_34_C, term_13_C_local_virt, term_13_X_local_virt, term_31_C_local_occ, term_31_X_EA, term_14_C_local_virt, term_14_X_IP, term_41_C_local_occ, term_41_X_local_occ, term_21_C_2S, term_12_C_2S,\
                            term_32_C_local_occ, term_32_X_local_occ, term_42_C_local_occ, term_42_X_EA, term_23_C_local_virt, term_23_X_IP, term_24_C_local_virt, term_24_X_local_virt,    \
                            term_41_C_1S_virt_arb, term_41_X_1S_IP_arb, term_13_C_1S_occ_arb, term_13_X_1S_EA_arb, term_31_C_1S_virt_arb, term_31_X_1S_virt_arb, term_14_C_1S_occ_arb, term_14_X_1S_occ_arb, \
                            term_31_C_3S_EA_occ_IP_arb, term_31_X_3S_EA_occ_IP_arb, term_14_C_3S_IP_virt_EA_arb, term_14_X_3S_IP_virt_EA_arb, term_21_C_2S_A_occ_B_occ_arb, term_21_X_2S_A_occ_B_occ_arb, \
                            term_43_C_2S_EA_IP_arb, term_43_X_2S_EA_EA_arb, term_31_X_3S_occ_IP_arb_virt_arb, term_13_X_3S_IP_virt_occ_arb, term_41_X_3S_occ_EA_virt_arb, \
                            term_14_X_3S_virt_EA_arb_occ_arb, term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, term_43_X_2S_IP_IP_arb, term_31_C_3S_EA_IP_arb_virt_arb, \
                            term_13_C_3S_IP_virt_EA_arb, term_41_C_3S_occ_EA_IP_arb, term_14_C_3S_IP_occ_arb_EA_arb, term_21_C_2S_B_occ_arb_A_virt_arb, term_43_C_2S_IP_EA_arb\
                            = calc_term_BBAA_4_exch(ind_A, ind_B, value, SC_A_BA_ia      ,
                                                                        CS_A_AB_ia       ,
                                                                        SC_B_AB_ia       ,
                                                                        CS_B_BA_ia        ,
                                                                        CS_EA_A_a       ,
                                                                        SC_IP_A_i       ,
                                                                        SC_IP_B_i        ,
                                                                        CS_EA_B_a       ,
                                                                        SC_A_BA_pa    ,
                                                                        CS_A_AB_iq    ,
                                                                        SC_B_AB_pa    ,
                                                                        CS_B_BA_iq    ,
                                                                        CS_EA_A_q     ,
                                                                        SC_IP_B_p     ,
                                                                        SC_IP_A_p     ,
                                                                        CS_EA_B_q     ,
                                                                        ACA_A_right   ,
                                                                        SCS_A_BB_iq   ,
                                                                        SCS_A_BB_pa   ,
                                                                        ACA_B_right   ,
                                                                        SCS_B_AA_pa   ,
                                                                        ACA_A_left    ,
                                                                        ACA_B_left    ,
                                                                        CA_EA_A_right ,
                                                                        AC_IP_B_right ,
                                                                        AC_IP_A_right ,
                                                                        CA_EA_B_right ,
                                                                        CA_EA_A_left  ,
                                                                        AC_IP_B_left  ,
                                                                        AC_IP_A_left  ,
                                                                        CA_EA_B_left  , )#, twoel_AO_AAAB_canonical, ind_B[0], value)
                        

                        fc_31_coul_3S_EA_occ_IP_arb += term_31_C_3S_EA_occ_IP_arb
                        fc_31_exch_3S_EA_occ_IP_arb += term_31_X_3S_EA_occ_IP_arb
                        fc_14_coul_3S_IP_virt_EA_arb += term_14_C_3S_IP_virt_EA_arb
                        fc_14_exch_3S_IP_virt_EA_arb += term_14_X_3S_IP_virt_EA_arb
                        fc_21_coul_2S_A_occ_B_occ_arb += term_21_C_2S_A_occ_B_occ_arb
                        fc_21_exch_2S_A_occ_B_occ_arb += term_21_X_2S_A_occ_B_occ_arb
                        fc_43_coul_2S_EA_IP_arb += term_43_C_2S_EA_IP_arb
                        fc_43_exch_2S_EA_EA_arb += term_43_X_2S_EA_EA_arb
                        fc_43_coul_2S_IP_EA_arb += term_43_C_2S_IP_EA_arb

                        fc_31_exch_3S_occ_IP_arb_virt_arb += term_31_X_3S_occ_IP_arb_virt_arb
                        fc_13_exch_3S_IP_virt_occ_arb += term_13_X_3S_IP_virt_occ_arb
                        fc_41_exch_3S_occ_EA_virt_arb += term_41_X_3S_occ_EA_virt_arb
                        fc_14_exch_3S_virt_EA_arb_occ_arb += term_14_X_3S_virt_EA_arb_occ_arb
                        fc_21_exch_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb
                        fc_43_exch_2S_IP_IP_arb += term_43_X_2S_IP_IP_arb
                        fc_31_coul_3S_EA_IP_arb_virt_arb += term_31_C_3S_EA_IP_arb_virt_arb
                        fc_13_coul_3S_IP_virt_EA_arb += term_13_C_3S_IP_virt_EA_arb
                        fc_41_coul_3S_occ_EA_IP_arb += term_41_C_3S_occ_EA_IP_arb
                        fc_14_coul_3S_IP_occ_arb_EA_arb += term_14_C_3S_IP_occ_arb_EA_arb
                        fc_21_coul_2S_B_occ_arb_A_virt_arb += term_21_C_2S_B_occ_arb_A_virt_arb

                        fc_31_coul_1S_virt_arb += term_31_C_1S_virt_arb
                        fc_31_exch_1S_virt_arb += term_31_X_1S_virt_arb
                        fc_14_coul_1S_occ_arb  += term_14_C_1S_occ_arb 
                        fc_14_exch_1S_occ_arb  += term_14_X_1S_occ_arb 

                        fc_41_exch_1S_IP_arb   += term_41_X_1S_IP_arb  
                        fc_41_coul_1S_virt_arb += term_41_C_1S_virt_arb
                        fc_13_exch_1S_EA_arb   += term_13_X_1S_EA_arb  
                        fc_13_coul_1S_occ_arb  += term_13_C_1S_occ_arb 

                        fc_12_exch += term_12_X
                        fc_34_coul += term_34_C
                        fc_34_exch += term_34_X
                                                
                        fc_42_exch_EA += term_42_X_EA   
                        fc_42_coul_local_occ += term_42_C_local_occ
                        fc_24_coul_local_virt += term_24_C_local_virt
                        fc_24_exch_local_virt += term_24_X_local_virt
                        
                        fc_32_coul_local_occ += term_32_C_local_occ
                        fc_32_exch_local_occ += term_32_X_local_occ
                        fc_23_coul_local_virt += term_23_C_local_virt
                        fc_23_exch_IP += term_23_X_IP
                        fc_14_coul_local_virt += term_14_C_local_virt                

                        fc_21_coul_2S += term_21_C_2S   
                        fc_12_coul_2S += term_12_C_2S   
                        
                        fc_31_coul_local_occ  += term_31_C_local_occ
                        fc_13_coul_local_virt += term_13_C_local_virt
                        fc_13_exch_local_virt += term_13_X_local_virt
                        fc_31_exch_EA += term_31_X_EA   
                        fc_41_coul_local_occ  += term_41_C_local_occ
                        fc_41_exch_local_occ  += term_41_X_local_occ
                        
                        fc_14_exch_IP += term_14_X_IP    
                                                
                        fc_43_coul += term_43_C
                        fc_21_exch += term_21_X
                        fc_43_exch += term_43_X                           
                    else:
                        print('No ov is not implemented for BBAA_4')
                        
                        #term_21_X, term_43_X, term_43_C= calc_term_BBAA_4_exch(ind_A, ind_B, value, AC_no_S=AC_no_S)#, twoel_AO_AAAB_canonical, ind_B[0], value)
                        #
                        #fc_43_coul += term_43_C
                        #fc_21_exch += term_21_X
                        #fc_43_exch += term_43_X   

        elif unique_count == 3: 
            #twoel_AO_AAAB_canonical = defaultdict(lambda: {
            #            'value': {}, 'ind': None, 'type': [], 'final': []
            #        })
            for row in twoelint_subblock:
                i = int(row[0])
                j = int(row[1])
                k = int(row[2])
                l = int(row[3])
                
                if j > NBAS_A:
                    ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                    if is_Ov:

                        if i == j:
                            term_21_C, term_12_C, term_31_X_local_occ, term_31_C_EA, term_14_X_local_virt, term_14_C_IP, term_21_X_2S, term_12_X_2S, term_42_X_local_occ, term_42_C_EA, term_23_X_local_virt, term_23_C_IP, \
                        term_41_C_1S_IP_arb, term_41_X_1S_virt_arb, term_13_C_1S_EA_arb, term_13_X_1S_occ_arb, \
                        term_31_C_3S_occ_IP_arb_virt_arb, term_13_C_3S_IP_virt_occ_arb, term_41_C_3S_occ_EA_virt_arb, term_14_C_3S_virt_EA_arb_occ_arb, term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, \
                        term_43_C_2S_IP_IP_arb, term_31_X_3S_EA_IP_arb_virt_arb, term_13_X_3S_IP_virt_EA_arb, term_41_X_3S_occ_EA_IP_arb , term_14_X_3S_IP_occ_arb_EA_arb , term_21_X_2S_B_occ_arb_A_virt_arb, \
                        term_43_C_2S_EA_EA_arb, term_43_X_2S_EA_IP_arb, term_43_X_2S_IP_EA_arb \
                                = calc_term_BBAA_3_coul_with_Ov(ind_A, ind_B, value, 'A', SC_A_BA_ia      ,
                                                                                        CS_A_AB_ia       ,
                                                                                        SC_B_AB_ia       ,
                                                                                        CS_B_BA_ia        ,
                                                                                        CS_EA_A_a       ,
                                                                                        SC_IP_A_i       ,
                                                                                        SC_IP_B_i        ,
                                                                                        CS_EA_B_a       ,
                                                                                        SC_A_BA_pa    ,
                                                                                        CS_A_AB_iq    ,
                                                                                        SC_B_AB_pa    ,
                                                                                        CS_B_BA_iq    ,
                                                                                        CS_EA_A_q     ,
                                                                                        SC_IP_B_p     ,
                                                                                        SC_IP_A_p     ,
                                                                                        CS_EA_B_q     ,
                                                                                        ACA_A_right   ,
                                                                                        SCS_A_BB_iq   ,
                                                                                        SCS_A_BB_pa   ,
                                                                                        ACA_B_right   ,
                                                                                        SCS_B_AA_pa   ,
                                                                                        ACA_A_left    ,
                                                                                        ACA_B_left    ,
                                                                                        CA_EA_A_right ,
                                                                                        AC_IP_B_right ,
                                                                                        AC_IP_A_right ,
                                                                                        CA_EA_B_right ,
                                                                                        CA_EA_A_left  ,
                                                                                        AC_IP_B_left  ,
                                                                                        AC_IP_A_left  ,
                                                                                        CA_EA_B_left  ,)#, twoel_AO_AAAB_canonical, ind_B[0], value)
                            fc_43_exch_2S_EA_IP_arb += term_43_X_2S_EA_IP_arb
                            fc_43_coul_2S_EA_EA_arb += term_43_C_2S_EA_EA_arb
                            fc_31_coul_3S_occ_IP_arb_virt_arb += term_31_C_3S_occ_IP_arb_virt_arb
                            fc_13_coul_3S_IP_virt_occ_arb += term_13_C_3S_IP_virt_occ_arb
                            fc_41_coul_3S_occ_EA_virt_arb += term_41_C_3S_occ_EA_virt_arb
                            fc_14_coul_3S_virt_EA_arb_occ_arb += term_14_C_3S_virt_EA_arb_occ_arb
                            fc_21_coul_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb
                            fc_43_coul_2S_IP_IP_arb += term_43_C_2S_IP_IP_arb 
                            fc_31_exch_3S_EA_IP_arb_virt_arb += term_31_X_3S_EA_IP_arb_virt_arb 
                            fc_13_exch_3S_IP_virt_EA_arb += term_13_X_3S_IP_virt_EA_arb 
                            fc_41_exch_3S_occ_EA_IP_arb += term_41_X_3S_occ_EA_IP_arb 
                            fc_14_exch_3S_IP_occ_arb_EA_arb += term_14_X_3S_IP_occ_arb_EA_arb 
                            fc_21_exch_2S_B_occ_arb_A_virt_arb += term_21_X_2S_B_occ_arb_A_virt_arb

                            fc_43_exch_2S_IP_EA_arb += term_43_X_2S_IP_EA_arb
                            fc_42_coul_EA += term_42_C_EA
                            fc_42_exch_local_occ += term_42_X_local_occ
                            fc_41_coul_1S_IP_arb   += term_41_C_1S_IP_arb  
                            fc_41_exch_1S_virt_arb += term_41_X_1S_virt_arb
                            fc_13_coul_1S_EA_arb   += term_13_C_1S_EA_arb  
                            fc_13_exch_1S_occ_arb  += term_13_X_1S_occ_arb 
                            fc_21_exch_2S += term_21_X_2S
                            fc_12_exch_2S += term_12_X_2S
                            fc_31_exch_local_occ  += term_31_X_local_occ
                            fc_31_coul_EA += term_31_C_EA
                            fc_14_exch_local_virt += term_14_X_local_virt
                            fc_14_coul_IP += term_14_C_IP

                            fc_21_coul += term_21_C
                            fc_12_coul += term_12_C

                            fc_23_coul_IP += term_23_C_IP
                            fc_23_exch_local_virt += term_23_X_local_virt
                        else:
                            term_21_C, term_12_C, term_31_X_local_occ, term_31_C_EA, term_14_X_local_virt, term_14_C_IP, term_21_X_2S, term_12_X_2S, term_42_X_local_occ, term_42_C_EA, term_23_X_local_virt, term_23_C_IP, \
                            term_41_C_1S_IP_arb, term_41_X_1S_virt_arb, term_13_C_1S_EA_arb, term_13_X_1S_occ_arb, \
                        term_31_C_3S_occ_IP_arb_virt_arb, term_13_C_3S_IP_virt_occ_arb, term_41_C_3S_occ_EA_virt_arb, term_14_C_3S_virt_EA_arb_occ_arb, term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, \
                        term_43_C_2S_IP_IP_arb, term_31_X_3S_EA_IP_arb_virt_arb, term_13_X_3S_IP_virt_EA_arb, term_41_X_3S_occ_EA_IP_arb , term_14_X_3S_IP_occ_arb_EA_arb , term_21_X_2S_B_occ_arb_A_virt_arb, \
                        term_43_C_2S_EA_EA_arb, term_43_X_2S_EA_IP_arb, term_43_X_2S_IP_EA_arb \
                                = calc_term_BBAA_3_coul_with_Ov(ind_A, ind_B, value, 'B', 
                                                                SC_A_BA_ia      ,
                                                                CS_A_AB_ia       ,
                                                                SC_B_AB_ia       ,
                                                                CS_B_BA_ia        ,
                                                                CS_EA_A_a       ,
                                                                SC_IP_A_i       ,
                                                                SC_IP_B_i        ,
                                                                CS_EA_B_a       ,
                                                                SC_A_BA_pa    ,
                                                                CS_A_AB_iq    ,
                                                                SC_B_AB_pa    ,
                                                                CS_B_BA_iq    ,
                                                                CS_EA_A_q     ,
                                                                SC_IP_B_p     ,
                                                                SC_IP_A_p     ,
                                                                CS_EA_B_q     ,
                                                                ACA_A_right   ,
                                                                SCS_A_BB_iq   ,
                                                                SCS_A_BB_pa   ,
                                                                ACA_B_right   ,
                                                                SCS_B_AA_pa   ,
                                                                ACA_A_left    ,
                                                                ACA_B_left    ,
                                                                CA_EA_A_right ,
                                                                AC_IP_B_right ,
                                                                AC_IP_A_right ,
                                                                CA_EA_B_right ,
                                                                CA_EA_A_left  ,
                                                                AC_IP_B_left  ,
                                                                AC_IP_A_left  ,
                                                                CA_EA_B_left  , )#, twoel_AO_AAAB_canonical, ind_B[0], value)
                            fc_43_exch_2S_EA_IP_arb += term_43_X_2S_EA_IP_arb
                            fc_43_coul_2S_EA_EA_arb += term_43_C_2S_EA_EA_arb
                            fc_31_coul_3S_occ_IP_arb_virt_arb += term_31_C_3S_occ_IP_arb_virt_arb
                            fc_13_coul_3S_IP_virt_occ_arb += term_13_C_3S_IP_virt_occ_arb
                            fc_41_coul_3S_occ_EA_virt_arb += term_41_C_3S_occ_EA_virt_arb
                            fc_14_coul_3S_virt_EA_arb_occ_arb += term_14_C_3S_virt_EA_arb_occ_arb
                            fc_21_coul_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb
                            fc_43_coul_2S_IP_IP_arb += term_43_C_2S_IP_IP_arb 
                            fc_31_exch_3S_EA_IP_arb_virt_arb += term_31_X_3S_EA_IP_arb_virt_arb 
                            fc_13_exch_3S_IP_virt_EA_arb += term_13_X_3S_IP_virt_EA_arb 
                            fc_41_exch_3S_occ_EA_IP_arb += term_41_X_3S_occ_EA_IP_arb 
                            fc_14_exch_3S_IP_occ_arb_EA_arb += term_14_X_3S_IP_occ_arb_EA_arb 
                            fc_21_exch_2S_B_occ_arb_A_virt_arb += term_21_X_2S_B_occ_arb_A_virt_arb
                            fc_43_exch_2S_IP_EA_arb += term_43_X_2S_IP_EA_arb

                            fc_41_coul_1S_IP_arb   += term_41_C_1S_IP_arb  
                            fc_41_exch_1S_virt_arb += term_41_X_1S_virt_arb
                            fc_13_coul_1S_EA_arb   += term_13_C_1S_EA_arb  
                            fc_13_exch_1S_occ_arb  += term_13_X_1S_occ_arb 
                            fc_42_coul_EA += term_42_C_EA
                            fc_42_exch_local_occ += term_42_X_local_occ
                            fc_21_exch_2S += term_21_X_2S
                            fc_12_exch_2S += term_12_X_2S
                            
                            fc_21_coul += term_21_C
                            fc_12_coul += term_12_C

                            fc_31_exch_local_occ  += term_31_X_local_occ
                            fc_31_coul_EA += term_31_C_EA
                            fc_14_exch_local_virt += term_14_X_local_virt
                            fc_14_coul_IP += term_14_C_IP
                            
                            fc_23_coul_IP += term_23_C_IP
                            fc_23_exch_local_virt += term_23_X_local_virt                  
                
                else:
                    if is_Ov:

                        if i == k:
                            ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                            term_43_X, term_43_C, term_34_X, term_34_C, term_21_X, term_12_X, \
                            term_13_C_local_virt, term_13_X_local_virt, term_31_C_local_occ, term_31_X_EA, term_14_C_local_virt, term_41_C_local_occ, term_41_X_local_occ, term_14_X_IP, term_21_C_2S, term_12_C_2S,\
                                term_32_C_local_occ, term_32_X_local_occ, term_42_C_local_occ, term_42_X_EA, term_23_C_local_virt, term_23_X_IP, term_24_C_local_virt, term_24_X_local_virt, \
                                term_41_C_1S_virt_arb, term_41_X_1S_IP_arb, term_13_C_1S_occ_arb, term_13_X_1S_EA_arb, term_31_C_1S_virt_arb, term_31_X_1S_virt_arb, term_14_C_1S_occ_arb, term_14_X_1S_occ_arb, \
                            term_31_C_3S_EA_occ_IP_arb, term_31_X_3S_EA_occ_IP_arb, term_14_C_3S_IP_virt_EA_arb, term_14_X_3S_IP_virt_EA_arb, term_21_C_2S_A_occ_B_occ_arb, term_21_X_2S_A_occ_B_occ_arb, \
                            term_43_C_2S_EA_IP_arb, term_43_X_2S_EA_EA_arb, term_31_X_3S_occ_IP_arb_virt_arb, term_13_X_3S_IP_virt_occ_arb, term_41_X_3S_occ_EA_virt_arb, \
                            term_14_X_3S_virt_EA_arb_occ_arb, term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, term_43_X_2S_IP_IP_arb, term_31_C_3S_EA_IP_arb_virt_arb, \
                            term_13_C_3S_IP_virt_EA_arb, term_41_C_3S_occ_EA_IP_arb, term_14_C_3S_IP_occ_arb_EA_arb, term_21_C_2S_B_occ_arb_A_virt_arb, term_43_C_2S_IP_EA_arb\
                                    = calc_term_BBAA_3_exch_with_Ov(ind_A, ind_B, value, 'A', SC_A_BA_ia      ,
                                                                                            CS_A_AB_ia       ,
                                                                                            SC_B_AB_ia       ,
                                                                                            CS_B_BA_ia        ,
                                                                                            CS_EA_A_a       ,
                                                                                            SC_IP_A_i       ,
                                                                                            SC_IP_B_i        ,
                                                                                            CS_EA_B_a       ,
                                                                                            SC_A_BA_pa    ,
                                                                                            CS_A_AB_iq    ,
                                                                                            SC_B_AB_pa    ,
                                                                                            CS_B_BA_iq    ,
                                                                                            CS_EA_A_q     ,
                                                                                            SC_IP_B_p     ,
                                                                                            SC_IP_A_p     ,
                                                                                            CS_EA_B_q     ,
                                                                                            ACA_A_right   ,
                                                                                            SCS_A_BB_iq   ,
                                                                                            SCS_A_BB_pa   ,
                                                                                            ACA_B_right   ,
                                                                                            SCS_B_AA_pa   ,
                                                                                            ACA_A_left    ,
                                                                                            ACA_B_left    ,
                                                                                            CA_EA_A_right ,
                                                                                            AC_IP_B_right ,
                                                                                            AC_IP_A_right ,
                                                                                            CA_EA_B_right ,
                                                                                            CA_EA_A_left  ,
                                                                                            AC_IP_B_left  ,
                                                                                            AC_IP_A_left  ,
                                                                                            CA_EA_B_left  , )#, twoel_AO_AAAB_canonical, ind_B[0], value)
                            fc_43_coul_2S_IP_EA_arb += term_43_C_2S_IP_EA_arb
                            
                            fc_31_coul_3S_EA_occ_IP_arb += term_31_C_3S_EA_occ_IP_arb
                            fc_31_exch_3S_EA_occ_IP_arb += term_31_X_3S_EA_occ_IP_arb
                            fc_14_coul_3S_IP_virt_EA_arb += term_14_C_3S_IP_virt_EA_arb
                            fc_14_exch_3S_IP_virt_EA_arb += term_14_X_3S_IP_virt_EA_arb
                            fc_21_coul_2S_A_occ_B_occ_arb += term_21_C_2S_A_occ_B_occ_arb
                            fc_21_exch_2S_A_occ_B_occ_arb += term_21_X_2S_A_occ_B_occ_arb
                            fc_43_coul_2S_EA_IP_arb += term_43_C_2S_EA_IP_arb
                            fc_43_exch_2S_EA_EA_arb += term_43_X_2S_EA_EA_arb
                            fc_31_exch_3S_occ_IP_arb_virt_arb += term_31_X_3S_occ_IP_arb_virt_arb
                            fc_13_exch_3S_IP_virt_occ_arb += term_13_X_3S_IP_virt_occ_arb
                            fc_41_exch_3S_occ_EA_virt_arb += term_41_X_3S_occ_EA_virt_arb
                            fc_14_exch_3S_virt_EA_arb_occ_arb += term_14_X_3S_virt_EA_arb_occ_arb
                            fc_21_exch_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb
                            fc_43_exch_2S_IP_IP_arb += term_43_X_2S_IP_IP_arb
                            fc_31_coul_3S_EA_IP_arb_virt_arb += term_31_C_3S_EA_IP_arb_virt_arb
                            fc_13_coul_3S_IP_virt_EA_arb += term_13_C_3S_IP_virt_EA_arb
                            fc_41_coul_3S_occ_EA_IP_arb += term_41_C_3S_occ_EA_IP_arb
                            fc_14_coul_3S_IP_occ_arb_EA_arb += term_14_C_3S_IP_occ_arb_EA_arb
                            fc_21_coul_2S_B_occ_arb_A_virt_arb += term_21_C_2S_B_occ_arb_A_virt_arb

                            fc_31_coul_1S_virt_arb += term_31_C_1S_virt_arb
                            fc_31_exch_1S_virt_arb += term_31_X_1S_virt_arb
                            fc_14_coul_1S_occ_arb  += term_14_C_1S_occ_arb 
                            fc_14_exch_1S_occ_arb  += term_14_X_1S_occ_arb 
                            fc_42_exch_EA += term_42_X_EA
                            fc_42_coul_local_occ += term_42_C_local_occ
                            fc_24_coul_local_virt += term_24_C_local_virt
                            fc_24_exch_local_virt += term_24_X_local_virt
                            fc_32_coul_local_occ += term_32_C_local_occ
                            fc_32_exch_local_occ += term_32_X_local_occ
                            fc_23_coul_local_virt += term_23_C_local_virt
                            fc_23_exch_IP += term_23_X_IP    
                            fc_41_exch_1S_IP_arb   += term_41_X_1S_IP_arb  
                            fc_41_coul_1S_virt_arb += term_41_C_1S_virt_arb
                            fc_13_exch_1S_EA_arb   += term_13_X_1S_EA_arb  
                            fc_13_coul_1S_occ_arb  += term_13_C_1S_occ_arb 
                            fc_21_coul_2S += term_21_C_2S                            
                            fc_12_coul_2S += term_12_C_2S                            
                            
                            fc_31_coul_local_occ  += term_31_C_local_occ
                            fc_13_coul_local_virt += term_13_C_local_virt
                            fc_13_exch_local_virt += term_13_X_local_virt
                            fc_31_exch_EA += term_31_X_EA   
                            fc_41_coul_local_occ  += term_41_C_local_occ
                            fc_41_exch_local_occ  += term_41_X_local_occ
                            fc_14_coul_local_virt += term_14_C_local_virt
                            fc_14_exch_IP += term_14_X_IP   
                            fc_43_coul += term_43_C
                            fc_21_exch += term_21_X
                            fc_43_exch += term_43_X  
                            fc_12_exch += term_12_X
                            fc_34_exch += term_34_X
                            fc_34_coul += term_34_C 
                        else:
                            ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)

                            term_43_X, term_43_C, term_34_X, term_34_C, term_21_X, term_12_X, \
                            term_13_C_local_virt, term_13_X_local_virt, term_31_C_local_occ, term_31_X_EA, term_14_C_local_virt, term_41_C_local_occ, term_41_X_local_occ, term_14_X_IP, term_21_C_2S, term_12_C_2S,\
                            term_32_C_local_occ, term_32_X_local_occ, term_42_C_local_occ, term_42_X_EA, term_23_C_local_virt, term_23_X_IP, term_24_C_local_virt, term_24_X_local_virt, \
                            term_41_C_1S_virt_arb, term_41_X_1S_IP_arb, term_13_C_1S_occ_arb, term_13_X_1S_EA_arb, term_31_C_1S_virt_arb, term_31_X_1S_virt_arb, term_14_C_1S_occ_arb, term_14_X_1S_occ_arb, \
                            term_31_C_3S_EA_occ_IP_arb, term_31_X_3S_EA_occ_IP_arb, term_14_C_3S_IP_virt_EA_arb, term_14_X_3S_IP_virt_EA_arb, term_21_C_2S_A_occ_B_occ_arb, term_21_X_2S_A_occ_B_occ_arb, \
                            term_43_C_2S_EA_IP_arb, term_43_X_2S_EA_EA_arb, term_31_X_3S_occ_IP_arb_virt_arb, term_13_X_3S_IP_virt_occ_arb, term_41_X_3S_occ_EA_virt_arb, \
                            term_14_X_3S_virt_EA_arb_occ_arb, term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, term_43_X_2S_IP_IP_arb, term_31_C_3S_EA_IP_arb_virt_arb, \
                            term_13_C_3S_IP_virt_EA_arb, term_41_C_3S_occ_EA_IP_arb, term_14_C_3S_IP_occ_arb_EA_arb, term_21_C_2S_B_occ_arb_A_virt_arb, term_43_C_2S_IP_EA_arb\
                                    = calc_term_BBAA_3_exch_with_Ov(ind_A, ind_B, value, 'B', SC_A_BA_ia      ,
                                                                                                CS_A_AB_ia    ,
                                                                                                SC_B_AB_ia    ,
                                                                                                CS_B_BA_ia    ,
                                                                                                CS_EA_A_a     ,
                                                                                                SC_IP_A_i     ,
                                                                                                SC_IP_B_i     ,
                                                                                                CS_EA_B_a     ,
                                                                                                SC_A_BA_pa    ,
                                                                                                CS_A_AB_iq    ,
                                                                                                SC_B_AB_pa    ,
                                                                                                CS_B_BA_iq    ,
                                                                                                CS_EA_A_q     ,
                                                                                                SC_IP_B_p     ,
                                                                                                SC_IP_A_p     ,
                                                                                                CS_EA_B_q     ,
                                                                                                ACA_A_right   ,
                                                                                                SCS_A_BB_iq   ,
                                                                                                SCS_A_BB_pa   ,
                                                                                                ACA_B_right   ,
                                                                                                SCS_B_AA_pa   ,
                                                                                                ACA_A_left    ,
                                                                                                ACA_B_left    ,
                                                                                                CA_EA_A_right ,
                                                                                                AC_IP_B_right ,
                                                                                                AC_IP_A_right ,
                                                                                                CA_EA_B_right ,
                                                                                                CA_EA_A_left  ,
                                                                                                AC_IP_B_left  ,
                                                                                                AC_IP_A_left  ,
                                                                                                CA_EA_B_left  , )#, twoel_AO_AAAB_canonical, ind_B[0], value)
                            fc_43_coul_2S_IP_EA_arb += term_43_C_2S_IP_EA_arb
                            fc_31_coul_3S_EA_occ_IP_arb += term_31_C_3S_EA_occ_IP_arb
                            fc_31_exch_3S_EA_occ_IP_arb += term_31_X_3S_EA_occ_IP_arb
                            fc_14_coul_3S_IP_virt_EA_arb += term_14_C_3S_IP_virt_EA_arb
                            fc_14_exch_3S_IP_virt_EA_arb += term_14_X_3S_IP_virt_EA_arb
                            fc_21_coul_2S_A_occ_B_occ_arb += term_21_C_2S_A_occ_B_occ_arb
                            fc_21_exch_2S_A_occ_B_occ_arb += term_21_X_2S_A_occ_B_occ_arb
                            fc_43_coul_2S_EA_IP_arb += term_43_C_2S_EA_IP_arb
                            fc_43_exch_2S_EA_EA_arb += term_43_X_2S_EA_EA_arb
                            fc_31_exch_3S_occ_IP_arb_virt_arb += term_31_X_3S_occ_IP_arb_virt_arb
                            fc_13_exch_3S_IP_virt_occ_arb += term_13_X_3S_IP_virt_occ_arb
                            fc_41_exch_3S_occ_EA_virt_arb += term_41_X_3S_occ_EA_virt_arb
                            fc_14_exch_3S_virt_EA_arb_occ_arb += term_14_X_3S_virt_EA_arb_occ_arb
                            fc_21_exch_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb
                            fc_43_exch_2S_IP_IP_arb += term_43_X_2S_IP_IP_arb
                            fc_31_coul_3S_EA_IP_arb_virt_arb += term_31_C_3S_EA_IP_arb_virt_arb
                            fc_13_coul_3S_IP_virt_EA_arb += term_13_C_3S_IP_virt_EA_arb
                            fc_41_coul_3S_occ_EA_IP_arb += term_41_C_3S_occ_EA_IP_arb
                            fc_14_coul_3S_IP_occ_arb_EA_arb += term_14_C_3S_IP_occ_arb_EA_arb
                            fc_21_coul_2S_B_occ_arb_A_virt_arb += term_21_C_2S_B_occ_arb_A_virt_arb
                            
                            fc_31_coul_1S_virt_arb += term_31_C_1S_virt_arb
                            fc_31_exch_1S_virt_arb += term_31_X_1S_virt_arb
                            fc_14_coul_1S_occ_arb  += term_14_C_1S_occ_arb 
                            fc_14_exch_1S_occ_arb  += term_14_X_1S_occ_arb 
                            fc_42_exch_EA += term_42_X_EA
                            fc_42_coul_local_occ += term_42_C_local_occ
                            fc_24_coul_local_virt += term_24_C_local_virt
                            fc_24_exch_local_virt += term_24_X_local_virt
                            fc_32_coul_local_occ += term_32_C_local_occ
                            fc_32_exch_local_occ += term_32_X_local_occ
                            fc_23_coul_local_virt += term_23_C_local_virt
                            fc_23_exch_IP += term_23_X_IP    
                            fc_41_exch_1S_IP_arb   += term_41_X_1S_IP_arb  
                            fc_41_coul_1S_virt_arb += term_41_C_1S_virt_arb
                            fc_13_exch_1S_EA_arb   += term_13_X_1S_EA_arb  
                            fc_13_coul_1S_occ_arb  += term_13_C_1S_occ_arb 
                            fc_21_coul_2S += term_21_C_2S                            
                            fc_12_coul_2S += term_12_C_2S                            
                            
                            fc_31_coul_local_occ  += term_31_C_local_occ
                            fc_13_coul_local_virt += term_13_C_local_virt
                            fc_13_exch_local_virt += term_13_X_local_virt
                            fc_31_exch_EA += term_31_X_EA   
                            fc_41_coul_local_occ  += term_41_C_local_occ
                            fc_41_exch_local_occ  += term_41_X_local_occ
                            fc_14_coul_local_virt += term_14_C_local_virt
                            fc_14_exch_IP += term_14_X_IP  
                            fc_43_coul += term_43_C
                            fc_21_exch += term_21_X
                            fc_43_exch += term_43_X  
                            fc_12_exch += term_12_X
                            fc_34_exch += term_34_X
                            fc_34_coul += term_34_C                 

        else:
            for row in twoelint_subblock:
                i = int(row[0])
                j = int(row[1])
                k = int(row[2])
                l = int(row[3])
                is_coul = False
                if i == j:
                    is_coul = True

                #term_21_X, term_21_C, term_43_X, term_43_C = calc_term_BBAA_2(ind_A, ind_B, value, is_coul, AC_no_S=AC_no_S)
                ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                if is_Ov:

                    term_21_X, term_21_C, term_43_X, term_43_C, term_12_X, term_12_C, term_34_X, term_34_C, term_13_C_local_virt, term_13_X_local_virt, term_31_C_local_occ, term_31_X_local_occ, term_31_C_EA, term_31_X_EA, term_14_C_local_virt, term_14_X_local_virt,\
                    term_41_C_local_occ, term_41_X_local_occ, term_14_C_IP, term_14_X_IP, term_21_X_2S, term_21_C_2S, term_12_X_2S, term_12_C_2S, term_32_C_local_occ, term_32_X_local_occ, term_42_C_local_occ, term_42_X_local_occ, term_42_C_EA, term_42_X_EA, \
                        term_23_C_local_virt, term_23_X_local_virt, term_23_C_IP, term_23_X_IP, term_24_C_local_virt, term_24_X_local_virt, term_41_C_1S_IP_arb, term_41_X_1S_virt_arb, term_13_C_1S_EA_arb, term_13_X_1S_occ_arb,  \
                        term_41_C_1S_virt_arb, term_41_X_1S_IP_arb, term_13_C_1S_occ_arb, term_13_X_1S_EA_arb, term_31_C_1S_virt_arb, term_31_X_1S_virt_arb, term_14_C_1S_occ_arb, term_14_X_1S_occ_arb, \
                        term_31_C_3S_occ_IP_arb_virt_arb, term_13_C_3S_IP_virt_occ_arb, term_41_C_3S_occ_EA_virt_arb, term_14_C_3S_virt_EA_arb_occ_arb, term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, \
                        term_43_C_2S_IP_IP_arb, term_31_X_3S_EA_IP_arb_virt_arb, term_13_X_3S_IP_virt_EA_arb, term_41_X_3S_occ_EA_IP_arb , term_14_X_3S_IP_occ_arb_EA_arb , term_21_X_2S_B_occ_arb_A_virt_arb, \
                            term_31_C_3S_EA_occ_IP_arb, term_31_X_3S_EA_occ_IP_arb, term_14_C_3S_IP_virt_EA_arb, term_14_X_3S_IP_virt_EA_arb, term_21_C_2S_A_occ_B_occ_arb, term_21_X_2S_A_occ_B_occ_arb, \
                            term_43_C_2S_EA_IP_arb, term_43_X_2S_EA_IP_arb, term_43_C_2S_EA_EA_arb, term_43_X_2S_EA_EA_arb, term_31_X_3S_occ_IP_arb_virt_arb, term_13_X_3S_IP_virt_occ_arb, term_41_X_3S_occ_EA_virt_arb, \
                            term_14_X_3S_virt_EA_arb_occ_arb, term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, term_43_X_2S_IP_IP_arb, term_31_C_3S_EA_IP_arb_virt_arb, \
                            term_13_C_3S_IP_virt_EA_arb, term_41_C_3S_occ_EA_IP_arb, term_14_C_3S_IP_occ_arb_EA_arb, term_21_C_2S_B_occ_arb_A_virt_arb, term_43_C_2S_IP_EA_arb, term_43_X_2S_IP_EA_arb \
                            = calc_term_BBAA_2(ind_A, ind_B, value, is_coul, SC_A_BA_ia      ,
                                                            CS_A_AB_ia    ,
                                                            SC_B_AB_ia    ,
                                                            CS_B_BA_ia    ,
                                                            CS_EA_A_a     ,
                                                            SC_IP_A_i     ,
                                                            SC_IP_B_i     ,
                                                            CS_EA_B_a     ,
                                                            SC_A_BA_pa    ,
                                                            CS_A_AB_iq    ,
                                                            SC_B_AB_pa    ,
                                                            CS_B_BA_iq    ,
                                                            CS_EA_A_q     ,
                                                            SC_IP_B_p     ,
                                                            SC_IP_A_p     ,
                                                            CS_EA_B_q     ,
                                                            ACA_A_right   ,
                                                            SCS_A_BB_iq   ,
                                                            SCS_A_BB_pa   ,
                                                            ACA_B_right   ,
                                                            SCS_B_AA_pa   ,
                                                            ACA_A_left    ,
                                                            ACA_B_left    ,
                                                            CA_EA_A_right ,
                                                            AC_IP_B_right ,
                                                            AC_IP_A_right ,
                                                            CA_EA_B_right ,
                                                            CA_EA_A_left  ,
                                                            AC_IP_B_left  ,
                                                            AC_IP_A_left  ,
                                                            CA_EA_B_left  , )
                    fc_43_coul_2S_IP_EA_arb += term_43_C_2S_IP_EA_arb
                    fc_43_exch_2S_IP_EA_arb += term_43_X_2S_IP_EA_arb
                    fc_31_coul_3S_EA_occ_IP_arb += term_31_C_3S_EA_occ_IP_arb
                    fc_31_exch_3S_EA_occ_IP_arb += term_31_X_3S_EA_occ_IP_arb
                    fc_14_coul_3S_IP_virt_EA_arb += term_14_C_3S_IP_virt_EA_arb
                    fc_14_exch_3S_IP_virt_EA_arb += term_14_X_3S_IP_virt_EA_arb
                    fc_21_coul_2S_A_occ_B_occ_arb += term_21_C_2S_A_occ_B_occ_arb
                    fc_21_exch_2S_A_occ_B_occ_arb += term_21_X_2S_A_occ_B_occ_arb
                    fc_43_coul_2S_EA_IP_arb += term_43_C_2S_EA_IP_arb
                    fc_43_exch_2S_EA_IP_arb += term_43_X_2S_EA_IP_arb
                    fc_43_coul_2S_EA_EA_arb += term_43_C_2S_EA_EA_arb
                    fc_43_exch_2S_EA_EA_arb += term_43_X_2S_EA_EA_arb
                    fc_31_exch_3S_occ_IP_arb_virt_arb += term_31_X_3S_occ_IP_arb_virt_arb
                    fc_13_exch_3S_IP_virt_occ_arb += term_13_X_3S_IP_virt_occ_arb
                    fc_41_exch_3S_occ_EA_virt_arb += term_41_X_3S_occ_EA_virt_arb
                    fc_14_exch_3S_virt_EA_arb_occ_arb += term_14_X_3S_virt_EA_arb_occ_arb
                    fc_21_exch_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += term_21_X_4S_A_occ_B_virt_A_virt_arb_B_occ_arb
                    fc_43_exch_2S_IP_IP_arb += term_43_X_2S_IP_IP_arb
                    fc_31_coul_3S_EA_IP_arb_virt_arb += term_31_C_3S_EA_IP_arb_virt_arb
                    fc_13_coul_3S_IP_virt_EA_arb += term_13_C_3S_IP_virt_EA_arb
                    fc_41_coul_3S_occ_EA_IP_arb += term_41_C_3S_occ_EA_IP_arb
                    fc_14_coul_3S_IP_occ_arb_EA_arb += term_14_C_3S_IP_occ_arb_EA_arb
                    fc_21_coul_2S_B_occ_arb_A_virt_arb += term_21_C_2S_B_occ_arb_A_virt_arb

                    fc_31_coul_3S_occ_IP_arb_virt_arb += term_31_C_3S_occ_IP_arb_virt_arb
                    fc_13_coul_3S_IP_virt_occ_arb += term_13_C_3S_IP_virt_occ_arb
                    fc_41_coul_3S_occ_EA_virt_arb += term_41_C_3S_occ_EA_virt_arb
                    fc_14_coul_3S_virt_EA_arb_occ_arb += term_14_C_3S_virt_EA_arb_occ_arb
                    fc_21_coul_4S_A_occ_B_virt_A_virt_arb_B_occ_arb += term_21_C_4S_A_occ_B_virt_A_virt_arb_B_occ_arb
                    fc_43_coul_2S_IP_IP_arb += term_43_C_2S_IP_IP_arb 
                    fc_31_exch_3S_EA_IP_arb_virt_arb += term_31_X_3S_EA_IP_arb_virt_arb 
                    fc_13_exch_3S_IP_virt_EA_arb += term_13_X_3S_IP_virt_EA_arb 
                    fc_41_exch_3S_occ_EA_IP_arb += term_41_X_3S_occ_EA_IP_arb 
                    fc_14_exch_3S_IP_occ_arb_EA_arb += term_14_X_3S_IP_occ_arb_EA_arb 
                    fc_21_exch_2S_B_occ_arb_A_virt_arb += term_21_X_2S_B_occ_arb_A_virt_arb

                    fc_31_coul_1S_virt_arb += term_31_C_1S_virt_arb
                    fc_31_exch_1S_virt_arb += term_31_X_1S_virt_arb
                    fc_14_coul_1S_occ_arb  += term_14_C_1S_occ_arb 
                    fc_14_exch_1S_occ_arb  += term_14_X_1S_occ_arb 
                    fc_41_coul_1S_IP_arb   += term_41_C_1S_IP_arb  
                    fc_41_exch_1S_virt_arb += term_41_X_1S_virt_arb
                    fc_13_coul_1S_EA_arb   += term_13_C_1S_EA_arb  
                    fc_13_exch_1S_occ_arb  += term_13_X_1S_occ_arb 
                    fc_21_coul_2S += term_21_C_2S  
                    fc_21_exch_2S += term_21_X_2S                          
                    fc_41_coul_1S_virt_arb += term_41_C_1S_virt_arb
                    fc_41_exch_1S_IP_arb   += term_41_X_1S_IP_arb  
                    fc_13_coul_1S_occ_arb  += term_13_C_1S_occ_arb 
                    fc_13_exch_1S_EA_arb   += term_13_X_1S_EA_arb  

                    fc_12_coul_2S += term_12_C_2S  
                    fc_12_exch_2S += term_12_X_2S 
                    
                    fc_12_coul += term_12_C
                    fc_12_exch += term_12_X
                    fc_34_coul += term_34_C
                    fc_34_exch += term_34_X
                    
                    fc_31_coul_local_occ  += term_31_C_local_occ
                    fc_31_exch_local_occ  += term_31_X_local_occ
                    fc_13_coul_local_virt += term_13_C_local_virt
                    fc_13_exch_local_virt += term_13_X_local_virt
                    fc_31_coul_EA += term_31_C_EA
                    fc_31_exch_EA += term_31_X_EA   
                    
                    fc_41_coul_local_occ  += term_41_C_local_occ
                    fc_41_exch_local_occ  += term_41_X_local_occ
                    fc_14_coul_local_virt += term_14_C_local_virt
                    fc_14_exch_local_virt += term_14_X_local_virt
                    fc_14_coul_IP += term_14_C_IP
                    fc_14_exch_IP += term_14_X_IP
                    
                    fc_32_coul_local_occ += term_32_C_local_occ
                    fc_32_exch_local_occ += term_32_X_local_occ
                    fc_23_coul_local_virt += term_23_C_local_virt
                    fc_23_exch_local_virt += term_23_X_local_virt
                    fc_23_coul_IP += term_23_C_IP
                    fc_23_exch_IP += term_23_X_IP
                    
                    fc_42_coul_local_occ += term_42_C_local_occ
                    fc_42_exch_local_occ += term_42_X_local_occ
                    fc_42_coul_EA += term_42_C_EA
                    fc_42_exch_EA += term_42_X_EA
                    fc_24_coul_local_virt += term_24_C_local_virt
                    fc_24_exch_local_virt += term_24_X_local_virt
                    fc_21_coul += term_21_C
                    fc_43_coul += term_43_C
                    fc_21_exch += term_21_X
                    fc_43_exch += term_43_X        


    return fc_21_coul, fc_21_exch, fc_43_coul, fc_43_exch, fc_12_coul, fc_12_exch, fc_34_coul, fc_34_exch,\
            fc_31_coul_local_occ, fc_31_exch_local_occ, fc_13_coul_local_virt, fc_13_exch_local_virt, fc_31_coul_EA, fc_31_exch_EA, \
            fc_41_coul_local_occ, fc_41_exch_local_occ, fc_14_coul_local_virt, fc_14_exch_local_virt, fc_14_coul_IP, fc_14_exch_IP, \
            fc_21_coul_2S, fc_21_exch_2S, fc_12_coul_2S, fc_12_exch_2S, fc_32_coul_local_occ, fc_32_exch_local_occ, fc_42_coul_local_occ, fc_42_exch_local_occ, fc_42_coul_EA, fc_42_exch_EA,\
            fc_23_coul_local_virt, fc_23_exch_local_virt, fc_23_coul_IP, fc_23_exch_IP, fc_24_coul_local_virt, fc_24_exch_local_virt, \
            fc_14_coul_1S_occ_arb, fc_14_exch_1S_occ_arb, fc_41_coul_1S_virt_arb, fc_41_exch_1S_virt_arb, fc_41_coul_1S_IP_arb, fc_41_exch_1S_IP_arb, \
            fc_13_coul_1S_occ_arb, fc_13_exch_1S_occ_arb, fc_13_coul_1S_EA_arb, fc_13_exch_1S_EA_arb, fc_31_coul_1S_virt_arb, fc_31_exch_1S_virt_arb, \
            fc_31_coul_3S_occ_IP_arb_virt_arb, fc_31_exch_3S_occ_IP_arb_virt_arb, fc_13_coul_3S_IP_virt_occ_arb, fc_13_exch_3S_IP_virt_occ_arb, \
            fc_41_coul_3S_occ_EA_virt_arb, fc_41_exch_3S_occ_EA_virt_arb, fc_14_coul_3S_virt_EA_arb_occ_arb, fc_14_exch_3S_virt_EA_arb_occ_arb, \
            fc_21_coul_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, fc_21_exch_4S_A_occ_B_virt_A_virt_arb_B_occ_arb, fc_43_coul_2S_IP_IP_arb, fc_43_exch_2S_IP_IP_arb, \
            fc_31_coul_3S_EA_IP_arb_virt_arb, fc_31_exch_3S_EA_IP_arb_virt_arb, fc_13_coul_3S_IP_virt_EA_arb, fc_13_exch_3S_IP_virt_EA_arb, fc_41_coul_3S_occ_EA_IP_arb, fc_41_exch_3S_occ_EA_IP_arb, \
            fc_14_coul_3S_IP_occ_arb_EA_arb, fc_14_exch_3S_IP_occ_arb_EA_arb, fc_21_coul_2S_B_occ_arb_A_virt_arb, fc_21_exch_2S_B_occ_arb_A_virt_arb, fc_31_coul_3S_EA_occ_IP_arb, fc_31_exch_3S_EA_occ_IP_arb, \
            fc_14_coul_3S_IP_virt_EA_arb, fc_14_exch_3S_IP_virt_EA_arb, fc_21_coul_2S_A_occ_B_occ_arb, fc_21_exch_2S_A_occ_B_occ_arb, fc_43_coul_2S_EA_IP_arb, fc_43_exch_2S_EA_IP_arb, \
            fc_43_coul_2S_EA_EA_arb, fc_43_exch_2S_EA_EA_arb, fc_43_coul_2S_IP_EA_arb, fc_43_exch_2S_IP_EA_arb
