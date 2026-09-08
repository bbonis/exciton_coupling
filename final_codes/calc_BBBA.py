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



def calc_term_BBBA_4(indexes, 
SC_A_BA_traf   ,
SC_A_BA_traf_pa,
CS_A_AB_traf   ,
CS_A_AB_traf_iq,
SC_B_AB_traf   ,
SC_B_AB_traf_pa,
CS_B_BA_traf   ,
term_32        ,
term_42        ,
term_23        ,
term_24        ,
term_CS_EA_B_a ,
term_CS_EA_B_q ,
term_SC_IP_B_i ,
term_SC_IP_B_p ,
CS_EA_A_a      ,
SC_IP_A_i      ,
CS_EA_A_q      ,
SC_IP_A_p      ,
ACA_B_right    ,
SCS_A_BB_iq    ,
ACA_B_left     ,
SCS_A_BB_pa    ,
AC_IP_B_right  ,
CA_EA_B_right  ,
AC_IP_B_left   ,
CA_EA_B_left   , value):
    term_32_X = np.float64(0.)
    term_32_C = np.float64(0.)
    term_42_X = np.float64(0.)
    term_42_C = np.float64(0.)
    
    term_23_X = np.float64(0.)
    term_23_C = np.float64(0.)
    term_24_X = np.float64(0.)
    term_24_C = np.float64(0.)
    
    term_32_C_CS_EA_SC = np.float64(0.)
    term_32_X_CS_EA_SC = np.float64(0.)    
            
    term_31_C_CS_EA_SC = np.float64(0.)
    term_31_X_CS_EA_SC = np.float64(0.)
    
    term_14_C_SC_IP_CS = np.float64(0.)
    term_14_X_SC_IP_CS = np.float64(0.)
    
    term_24_C_2S_IP = np.float64(0.)
    term_24_X_2S_IP = np.float64(0.)
    
    term_21_C_SC_occ = np.float64(0.)
    term_21_X_SC_occ = np.float64(0.)
    
    term_12_C_CS_virt = np.float64(0.)
    term_12_X_CS_virt = np.float64(0.)    
    
    term_21_C_CS_virt_arb = np.float64(0.)
    term_21_X_CS_virt_arb = np.float64(0.)

    term_34_C_CS_EA = np.float64(0.)
    term_34_X_CS_EA = np.float64(0.)    

    term_34_C_SC_IP = np.float64(0.)
    term_34_X_SC_IP = np.float64(0.)      
        
    term_43_C_1S_EA_arb = np.float64(0.)
    term_43_X_1S_EA_arb = np.float64(0.)    
    term_43_C_1S_IP_arb = np.float64(0.)
    term_43_X_1S_IP_arb = np.float64(0.)  

    term_31_C_2S_occ_virt_arb = np.float64(0.)
    term_31_X_2S_occ_virt_arb = np.float64(0.)
    term_13_C_2S_EA_arb_occ_arb = np.float64(0.)
    term_13_X_2S_EA_arb_occ_arb = np.float64(0.)
    term_14_C_2S_IP_occ_arb = np.float64(0.)
    term_14_X_2S_IP_occ_arb = np.float64(0.)
    term_21_C_3S_A_occ_B_virt_A_virt_arb = np.float64(0.)
    term_21_X_3S_A_occ_B_virt_A_virt_arb = np.float64(0.)
    term_31_C_2S_EA_virt_arb = np.float64(0.)
    term_31_X_2S_EA_virt_arb = np.float64(0.)
    term_13_C_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_13_X_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_14_C_2S_virt_occ_arb = np.float64(0.)
    term_14_X_2S_virt_occ_arb = np.float64(0.)
    term_43_C_3S_IP_IP_arb_EA_arb = np.float64(0.)
    term_43_X_3S_IP_IP_arb_EA_arb = np.float64(0.)
    term_31_C_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_31_X_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_13_C_2S_virt_EA_arb = np.float64(0.)
    term_13_X_2S_virt_EA_arb = np.float64(0.)
    term_41_C_2S_occ_virt_arb = np.float64(0.)
    term_41_X_2S_occ_virt_arb = np.float64(0.)
    term_21_C_3S_A_occ_A_virt_arb_B_occ_arb = np.float64(0.)
    term_21_X_3S_A_occ_A_virt_arb_B_occ_arb = np.float64(0.)
    term_13_C_2S_virt_occ_arb = np.float64(0.)
    term_13_X_2S_virt_occ_arb = np.float64(0.)
    term_41_C_2S_occ_IP_arb = np.float64(0.)
    term_41_X_2S_occ_IP_arb = np.float64(0.)
    term_14_C_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_14_X_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)

    term_41_C_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_41_X_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_43_C_3S_EA_EA_arb_IP_arb = np.float64(0.)
    term_43_X_3S_EA_EA_arb_IP_arb = np.float64(0.)

    #perms_2 = list(distinct_permutations(indexes))

    i0, i1, i2 = indexes

    # explicit perms
    p5 = (i0, i1, i2)
    p3 = (i0, i2, i1)
    p4 = (i1, i0, i2)
    p1 = (i1, i2, i0)
    p2 = (i2, i0, i1)
    p0 = (i2, i1, i0)

    perms = [p0, p1, p2, p3, p4, p5]
    
    #if perms != perms_2:
    #    print('permutations do not match!!')
        

    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)

    if SC_A_BA_traf is not None and CS_A_AB_traf is not None and SC_IP_A_i is not None and CS_EA_A_a is not None:
        for idx in [0,1]:
            term_32_X_CS_EA_SC += X_prefactor * SC_B_AB_traf[perms[idx][0]] * CS_EA_A_a[0, perms[idx][1]] * AC_IP_B_left[perms[idx][2], 0] * value
            term_24_C_2S_IP += C_prefactor * CS_B_BA_traf[perms[idx][2]] * SC_IP_A_i[perms[idx][1], 0] * CA_EA_B_right[0, perms[idx][0]] * value              
            term_14_X_SC_IP_CS += X_prefactor * CS_A_AB_traf[perms[idx][0]] * SC_IP_A_i[perms[idx][2], 0] * CA_EA_B_right[0, perms[idx][1]] * value
            term_32_C += C_prefactor * ACA_B_right[perms[idx][0], perms[idx][1]] * AC_IP_B_left[perms[idx][2], 0] * term_32
            term_13_C_2S_virt_occ_arb += C_prefactor * SCS_A_BB_pa[perms[idx][0], perms[idx][1]] * AC_IP_B_right[perms[idx][2], :] * term_23
            term_21_C_3S_A_occ_B_virt_A_virt_arb += C_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * CS_B_BA_traf[perms[idx][2]] * value
            term_14_C_2S_IP_occ_arb += C_prefactor * SC_IP_A_i[perms[idx][0], :] * CA_EA_B_right[:, perms[idx][1]] * SC_A_BA_traf_pa[perms[idx][2]] * value
            term_31_C_2S_occ_virt_arb += C_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * AC_IP_B_left[perms[idx][2], :] * term_32
            term_43_C_1S_IP_arb += C_prefactor * SC_IP_A_p[perms[idx][0], 0] * CA_EA_B_left[0, perms[idx][1]] * AC_IP_B_right[perms[idx][2], 0] * term_23            
            term_41_C_2S_occ_IP_arb += C_prefactor * SC_IP_A_p[perms[idx][0], :] * CA_EA_B_left[:, perms[idx][1]] * SC_A_BA_traf[perms[idx][2]] * value
            term_21_X_3S_A_occ_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * SC_B_AB_traf_pa[perms[idx][2]] * value
            
            term_13_C_2S_EA_arb_occ_arb += C_prefactor * AC_IP_B_right[perms[idx][0], :] * CS_EA_A_q[:, perms[idx][1]] * SC_A_BA_traf_pa[perms[idx][2]] * value
            term_14_C_4S_IP_virt_occ_arb_EA_arb += C_prefactor * SCS_A_BB_pa[perms[idx][0], perms[idx][1]] * SC_IP_A_i[perms[idx][2], :] * term_CS_EA_B_q
            term_43_C_3S_IP_IP_arb_EA_arb += C_prefactor * CA_EA_B_left[:, perms[idx][0]] * CS_EA_A_q[:, perms[idx][1]] * SC_IP_A_p[perms[idx][2], :] * term_SC_IP_B_i

            term_41_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * SC_IP_A_p[perms[idx][2], :] * term_CS_EA_B_a
            term_43_C_3S_EA_EA_arb_IP_arb += C_prefactor * AC_IP_B_right[perms[idx][0], :] * CS_EA_A_q[:, perms[idx][1]] * SC_IP_A_p[perms[idx][2], :] * term_CS_EA_B_a
            term_31_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * CS_EA_A_a[:, perms[idx][2]] * term_SC_IP_B_p

            term_23_C += C_prefactor * ACA_B_left[perms[idx][0], perms[idx][1]] * AC_IP_B_right[perms[idx][2], 0] * term_23
            term_21_X_SC_occ += X_prefactor * SC_A_BA_traf[perms[idx][0]] * ACA_B_left[perms[idx][1], perms[idx][2]] * value
            term_34_X_SC_IP += X_prefactor * AC_IP_B_left[perms[idx][0], 0] * SC_IP_A_i[perms[idx][1], 0] * CA_EA_B_right[0, perms[idx][2]] * term_32            
            term_34_X_CS_EA += X_prefactor * CA_EA_B_right[0, perms[idx][0]] * CS_EA_A_a[0, perms[idx][1]] * AC_IP_B_left[perms[idx][2], 0] * term_24
            term_41_X_2S_occ_virt_arb += X_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * CA_EA_B_left[:, perms[idx][2]] * term_42
            term_41_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SC_IP_A_p[perms[idx][0], :] * SCS_A_BB_iq[perms[idx][1], perms[idx][2]] * term_CS_EA_B_a
            
            term_43_X_1S_EA_arb += X_prefactor * CS_EA_A_q[0, perms[idx][0]] * AC_IP_B_right[perms[idx][1], 0] * CA_EA_B_left[0, perms[idx][2]] * term_42
            term_42_X += X_prefactor * ACA_B_right[perms[idx][0], perms[idx][1]] * CA_EA_B_left[0, perms[idx][2]] * term_42
            term_24_X += X_prefactor * ACA_B_left[perms[idx][0], perms[idx][1]] * CA_EA_B_right[0, perms[idx][2]] * term_24
            term_14_X_2S_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms[idx][0], perms[idx][1]] * CA_EA_B_right[:, perms[idx][2]] * term_24
            term_31_X_2S_EA_virt_arb += X_prefactor * AC_IP_B_left[perms[idx][0], :] * CS_EA_A_a[:, perms[idx][1]] * CS_A_AB_traf_iq[perms[idx][2]] * value
            term_21_X_CS_virt_arb += X_prefactor * ACA_B_left[perms[idx][0], perms[idx][1]] * CS_A_AB_traf_iq[perms[idx][2]] * value
            term_13_X_2S_virt_EA_arb += X_prefactor * AC_IP_B_right[perms[idx][0], :] * CS_EA_A_q[:, perms[idx][1]] * CS_A_AB_traf[perms[idx][2]] * value

        for idx in [3,5]:
            term_13_C_2S_virt_EA_arb += C_prefactor * AC_IP_B_right[perms[idx][0], :] * CS_EA_A_q[:, perms[idx][1]] * CS_A_AB_traf[perms[idx][2]] * value
            term_21_C_3S_A_occ_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * SC_B_AB_traf_pa[perms[idx][2]] * value
            term_41_C_2S_occ_virt_arb += C_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * CA_EA_B_left[:, perms[idx][2]] * term_42

            term_24_X_2S_IP += X_prefactor * CS_B_BA_traf[perms[idx][2]] * SC_IP_A_i[perms[idx][1], 0] * CA_EA_B_right[0, perms[idx][0]] * value

            term_32_C_CS_EA_SC += C_prefactor * SC_B_AB_traf[perms[idx][0]] * CS_EA_A_a[0, perms[idx][1]] * AC_IP_B_left[perms[idx][2], 0] * value

            
            term_21_C_SC_occ += C_prefactor * SC_A_BA_traf[perms[idx][0]] * ACA_B_left[perms[idx][1], perms[idx][2]] * value
            term_12_X_CS_virt += X_prefactor * CS_A_AB_traf[perms[idx][1]] * ACA_B_right[perms[idx][0], perms[idx][2]] * value
            
            term_34_C_SC_IP += C_prefactor * AC_IP_B_left[perms[idx][0], 0] * SC_IP_A_i[perms[idx][1], 0] * CA_EA_B_right[0, perms[idx][2]] * term_32
            term_31_C_CS_EA_SC += C_prefactor * SC_A_BA_traf[perms[idx][0]] * CS_EA_A_a[0, perms[idx][1]] * AC_IP_B_left[perms[idx][2], 0] * value
            term_13_X_4S_IP_virt_occ_arb_EA_arb += X_prefactor * CS_EA_A_q[:, perms[idx][0]] * SCS_A_BB_pa[perms[idx][1], perms[idx][2]] * term_SC_IP_B_i

            term_34_C_CS_EA += C_prefactor * CA_EA_B_right[0, perms[idx][0]] * CS_EA_A_a[0, perms[idx][1]] *  AC_IP_B_left[perms[idx][2], 0] * term_24
            term_42_C += C_prefactor * ACA_B_right[perms[idx][0], perms[idx][1]] * CA_EA_B_left[0, perms[idx][2]] * term_42
            term_24_C += C_prefactor * ACA_B_left[perms[idx][0], perms[idx][1]] * CA_EA_B_right[0, perms[idx][2]] * term_24
            term_14_C_2S_virt_occ_arb += C_prefactor * SCS_A_BB_pa[perms[idx][0], perms[idx][1]] * CA_EA_B_right[:, perms[idx][2]] * term_24
            term_31_C_2S_EA_virt_arb += C_prefactor * AC_IP_B_left[perms[idx][0], :] * CS_EA_A_a[:, perms[idx][1]] * CS_A_AB_traf_iq[perms[idx][2]] * value

            term_21_C_CS_virt_arb += C_prefactor * ACA_B_left[perms[idx][0], perms[idx][1]] * CS_A_AB_traf_iq[perms[idx][2]] * value
            term_31_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * CS_EA_A_a[:, perms[idx][2]] * term_SC_IP_B_p

        for idx in [2,4]:
            term_13_C_4S_IP_virt_occ_arb_EA_arb += C_prefactor * CS_EA_A_q[:, perms[idx][0]] * SCS_A_BB_pa[perms[idx][1], perms[idx][2]] * term_SC_IP_B_i 

            term_31_X_CS_EA_SC += X_prefactor * SC_A_BA_traf[perms[idx][0]] * CS_EA_A_a[0, perms[idx][1]] * AC_IP_B_left[perms[idx][2], 0] * value
            
            term_14_C_SC_IP_CS += C_prefactor * CS_A_AB_traf[perms[idx][1]] * SC_IP_A_i[perms[idx][0], 0] * CA_EA_B_right[0, perms[idx][2]] * value   
            term_43_C_1S_EA_arb += C_prefactor * CS_EA_A_q[0, perms[idx][0]] * AC_IP_B_right[perms[idx][1], 0] * CA_EA_B_left[0, perms[idx][2]] * term_42

            term_12_C_CS_virt += C_prefactor * CS_A_AB_traf[perms[idx][1]] * ACA_B_right[perms[idx][0], perms[idx][2]] * value
            term_32_X += X_prefactor * ACA_B_right[perms[idx][0], perms[idx][1]] * AC_IP_B_left[perms[idx][2], 0] * term_32
            term_43_X_3S_IP_IP_arb_EA_arb += X_prefactor * CA_EA_B_left[:, perms[idx][0]] * CS_EA_A_q[:, perms[idx][1]] * SC_IP_A_p[perms[idx][2], :] * term_SC_IP_B_i
            term_21_X_3S_A_occ_B_virt_A_virt_arb += X_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * CS_B_BA_traf[perms[idx][2]] * value
            term_14_X_2S_IP_occ_arb += X_prefactor * SC_IP_A_i[perms[idx][0], :] * CA_EA_B_right[:, perms[idx][1]] * SC_A_BA_traf_pa[perms[idx][2]] * value
            term_13_X_2S_EA_arb_occ_arb += X_prefactor * AC_IP_B_right[perms[idx][0], :] * CS_EA_A_q[:, perms[idx][1]] * SC_A_BA_traf_pa[perms[idx][2]] * value
            term_31_X_2S_occ_virt_arb += X_prefactor * SCS_A_BB_iq[perms[idx][0], perms[idx][1]] * AC_IP_B_left[perms[idx][2], :] * term_32
            term_43_X_1S_IP_arb += X_prefactor * SC_IP_A_p[perms[idx][0], 0] * CA_EA_B_left[0, perms[idx][1]] * AC_IP_B_right[perms[idx][2], 0] * term_23            
            term_23_X += X_prefactor * ACA_B_left[perms[idx][0], perms[idx][1]] * AC_IP_B_right[perms[idx][2], 0] * term_23
            term_43_X_3S_EA_EA_arb_IP_arb += X_prefactor * AC_IP_B_right[perms[idx][0], :] * CS_EA_A_q[:, perms[idx][1]] * SC_IP_A_p[perms[idx][2], :] * term_CS_EA_B_a
            term_14_X_4S_IP_virt_occ_arb_EA_arb += X_prefactor * SCS_A_BB_pa[perms[idx][0], perms[idx][1]] * SC_IP_A_i[perms[idx][2], :] * term_CS_EA_B_q
            term_41_X_2S_occ_IP_arb += X_prefactor * SC_IP_A_p[perms[idx][0], :] * CA_EA_B_left[:, perms[idx][1]] * SC_A_BA_traf[perms[idx][2]] * value
            term_13_X_2S_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms[idx][0], perms[idx][1]] * AC_IP_B_right[perms[idx][2], :] * term_23
        return term_32_X, term_32_C, term_42_X, term_42_C, term_23_X, term_23_C, term_24_X, term_24_C, term_31_X_CS_EA_SC, term_31_C_CS_EA_SC, term_14_X_SC_IP_CS, term_14_C_SC_IP_CS, term_32_X_CS_EA_SC, term_32_C_CS_EA_SC, \
                term_24_C_2S_IP, term_24_X_2S_IP, term_21_C_SC_occ, term_21_X_SC_occ, term_12_C_CS_virt, term_12_X_CS_virt, term_34_C_CS_EA, term_34_X_CS_EA, term_34_C_SC_IP, term_34_X_SC_IP, \
                term_43_C_1S_EA_arb, term_43_X_1S_EA_arb, term_43_C_1S_IP_arb, term_43_X_1S_IP_arb, term_21_C_CS_virt_arb, term_21_X_CS_virt_arb, term_31_C_2S_occ_virt_arb, term_31_X_2S_occ_virt_arb, term_13_C_2S_EA_arb_occ_arb, term_13_X_2S_EA_arb_occ_arb, term_14_C_2S_IP_occ_arb, term_14_X_2S_IP_occ_arb,\
                    term_21_C_3S_A_occ_B_virt_A_virt_arb, term_21_X_3S_A_occ_B_virt_A_virt_arb, term_31_C_2S_EA_virt_arb, term_31_X_2S_EA_virt_arb, term_13_C_4S_IP_virt_occ_arb_EA_arb, term_13_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_14_C_2S_virt_occ_arb, term_14_X_2S_virt_occ_arb, term_43_C_3S_IP_IP_arb_EA_arb, term_43_X_3S_IP_IP_arb_EA_arb, term_31_C_4S_occ_EA_virt_arb_IP_arb, term_31_X_4S_occ_EA_virt_arb_IP_arb, \
                    term_13_C_2S_virt_EA_arb, term_13_X_2S_virt_EA_arb, term_41_C_2S_occ_virt_arb, term_41_X_2S_occ_virt_arb, term_21_C_3S_A_occ_A_virt_arb_B_occ_arb, term_21_X_3S_A_occ_A_virt_arb_B_occ_arb, \
                    term_13_C_2S_virt_occ_arb, term_13_X_2S_virt_occ_arb, term_41_C_2S_occ_IP_arb, term_41_X_2S_occ_IP_arb, term_14_C_4S_IP_virt_occ_arb_EA_arb, term_14_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_41_C_4S_occ_EA_virt_arb_IP_arb, term_41_X_4S_occ_EA_virt_arb_IP_arb, term_43_C_3S_EA_EA_arb_IP_arb, term_43_X_3S_EA_EA_arb_IP_arb

    else:
        return print('No ov not implemented yet')
        #for idx in [0,1]:
        #    term_32_C += C_prefactor * ACA_B[perms[idx][0], perms[idx][1]] * AC_IP_B[perms[idx][2], 0] * term_32
        #    term_42_C += C_prefactor * ACA_B[perms[idx][0], perms[idx][1]] * CA_EA_B[0, perms[idx][2]] * term_42


def calc_term_BBBA_3(indexes, is_type_1     ,
SC_A_BA_traf   ,
SC_A_BA_traf_pa,
CS_A_AB_traf   ,
CS_A_AB_traf_iq,
SC_B_AB_traf   ,
SC_B_AB_traf_pa,
CS_B_BA_traf   ,
term_32        ,
term_42        ,
term_23        ,
term_24        ,
term_CS_EA_B_a ,
term_CS_EA_B_q ,
term_SC_IP_B_i ,
term_SC_IP_B_p ,
CS_EA_A_a      ,
SC_IP_A_i      ,
CS_EA_A_q      ,
SC_IP_A_p      ,
ACA_B_right    ,
SCS_A_BB_iq    ,
ACA_B_left     ,
SCS_A_BB_pa    ,
AC_IP_B_right  ,
CA_EA_B_right  ,
AC_IP_B_left   ,
CA_EA_B_left   , value,
term32_bucket_ctx=None):
    term_32_X = np.float64(0.)
    term_32_C = np.float64(0.)
    term_42_X = np.float64(0.)
    term_42_C = np.float64(0.)
    
    term_23_X = np.float64(0.)
    term_23_C = np.float64(0.)
    term_24_X = np.float64(0.)
    term_24_C = np.float64(0.)
    
    term_32_C_CS_EA_SC = np.float64(0.)
    term_32_X_CS_EA_SC = np.float64(0.)    
    
    term_31_C_CS_EA_SC = np.float64(0.)
    term_31_X_CS_EA_SC = np.float64(0.)
    term_21_C_CS_virt_arb = np.float64(0.)
    term_21_X_CS_virt_arb = np.float64(0.)
    term_14_C_SC_IP_CS = np.float64(0.)
    term_14_X_SC_IP_CS = np.float64(0.)
    
    term_24_C_2S_IP = np.float64(0.)
    term_24_X_2S_IP = np.float64(0.)
    
    term_21_C_SC_occ = np.float64(0.)
    term_21_X_SC_occ = np.float64(0.)
    
    term_12_C_CS_virt = np.float64(0.)
    term_12_X_CS_virt = np.float64(0.)    
    
    term_34_C_CS_EA = np.float64(0.)
    term_34_X_CS_EA = np.float64(0.)    

    term_34_C_SC_IP = np.float64(0.)
    term_34_X_SC_IP = np.float64(0.)      
    
    term_43_C_1S_EA_arb = np.float64(0.)
    term_43_X_1S_EA_arb = np.float64(0.)
    term_43_C_1S_IP_arb = np.float64(0.)
    term_43_X_1S_IP_arb = np.float64(0.)  

    term_31_C_2S_occ_virt_arb = np.float64(0.)
    term_31_X_2S_occ_virt_arb = np.float64(0.)
    term_13_C_2S_EA_arb_occ_arb = np.float64(0.)
    term_13_X_2S_EA_arb_occ_arb = np.float64(0.)
    term_14_C_2S_IP_occ_arb = np.float64(0.)
    term_14_X_2S_IP_occ_arb = np.float64(0.)
    term_21_C_3S_A_occ_B_virt_A_virt_arb = np.float64(0.)
    term_21_X_3S_A_occ_B_virt_A_virt_arb = np.float64(0.)
    term_31_C_2S_EA_virt_arb = np.float64(0.)
    term_31_X_2S_EA_virt_arb = np.float64(0.)
    term_13_C_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_13_X_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_14_C_2S_virt_occ_arb = np.float64(0.)
    term_14_X_2S_virt_occ_arb = np.float64(0.)
    term_43_C_3S_IP_IP_arb_EA_arb = np.float64(0.)
    term_43_X_3S_IP_IP_arb_EA_arb = np.float64(0.)
    term_31_C_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_31_X_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_13_C_2S_virt_EA_arb = np.float64(0.)
    term_13_X_2S_virt_EA_arb = np.float64(0.)
    term_41_C_2S_occ_virt_arb = np.float64(0.)
    term_41_X_2S_occ_virt_arb = np.float64(0.)
    term_21_C_3S_A_occ_A_virt_arb_B_occ_arb = np.float64(0.)
    term_21_X_3S_A_occ_A_virt_arb_B_occ_arb = np.float64(0.)
    term_13_C_2S_virt_occ_arb = np.float64(0.)
    term_13_X_2S_virt_occ_arb = np.float64(0.)
    term_41_C_2S_occ_IP_arb = np.float64(0.)
    term_41_X_2S_occ_IP_arb = np.float64(0.)
    term_14_C_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_14_X_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)

    term_41_C_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_41_X_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_43_C_3S_EA_EA_arb_IP_arb = np.float64(0.)
    term_43_X_3S_EA_EA_arb_IP_arb = np.float64(0.)

    #perms_2 = list(distinct_permutations(indexes))

    a, b, c = indexes
    p2 = (a, b, c)
    p0 = (b, c, a)
    p1 = (b, a, c)
    if a == c:
        b, a, c = indexes
        p0 = (a, b, c)
        p2 = (b, c, a)
        p1 = (b, a, c)

    perms = [p0, p1, p2]

    #if perms != perms_2:
    #    print('permutations do not match!!')

    C_prefactor = np.float64(4.)
    X_prefactor = np.float64( -2.)

    if SC_A_BA_traf is not None and CS_A_AB_traf is not None and SC_IP_A_i is not None and CS_EA_A_a is not None:
        if is_type_1:#type_1 = 6010

            term_31_C_2S_occ_virt_arb += C_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * AC_IP_B_left[perms[1][2], :] * term_32
            term_31_X_2S_occ_virt_arb += X_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * AC_IP_B_left[perms[2][2], :] * term_32
            term_31_C_2S_occ_virt_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], :] * term_32
            term_31_X_2S_occ_virt_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], :] * term_32

            term_13_C_2S_EA_arb_occ_arb += C_prefactor * AC_IP_B_right[perms[1][0], :] * CS_EA_A_q[:, perms[1][1]] * SC_A_BA_traf_pa[perms[1][2]] * value
            term_13_X_2S_EA_arb_occ_arb += X_prefactor * AC_IP_B_right[perms[2][0], :] * CS_EA_A_q[:, perms[2][1]] * SC_A_BA_traf_pa[perms[2][2]] * value
            term_13_C_2S_EA_arb_occ_arb += C_prefactor * AC_IP_B_right[perms[0][0], :] * CS_EA_A_q[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value
            term_13_X_2S_EA_arb_occ_arb += X_prefactor * AC_IP_B_right[perms[0][0], :] * CS_EA_A_q[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value

            term_14_C_2S_IP_occ_arb += C_prefactor * SC_IP_A_i[perms[1][0], :] * CA_EA_B_right[:, perms[1][1]] * SC_A_BA_traf_pa[perms[1][2]] * value
            term_14_X_2S_IP_occ_arb += X_prefactor * SC_IP_A_i[perms[2][0], :] * CA_EA_B_right[:, perms[2][1]] * SC_A_BA_traf_pa[perms[2][2]] * value
            term_14_C_2S_IP_occ_arb += C_prefactor * SC_IP_A_i[perms[0][0], :] * CA_EA_B_right[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value
            term_14_X_2S_IP_occ_arb += X_prefactor * SC_IP_A_i[perms[0][0], :] * CA_EA_B_right[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value

            term_21_C_3S_A_occ_B_virt_A_virt_arb += C_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * CS_B_BA_traf[perms[1][2]] * value
            term_21_X_3S_A_occ_B_virt_A_virt_arb += X_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * CS_B_BA_traf[perms[2][2]] * value
            term_21_C_3S_A_occ_B_virt_A_virt_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value
            term_21_X_3S_A_occ_B_virt_A_virt_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value

            term_31_C_2S_EA_virt_arb += C_prefactor * AC_IP_B_left[perms[2][0], :] * CS_EA_A_a[:, perms[2][1]] * CS_A_AB_traf_iq[perms[2][2]] * value
            term_31_X_2S_EA_virt_arb += X_prefactor * AC_IP_B_left[perms[1][0], :] * CS_EA_A_a[:, perms[1][1]] * CS_A_AB_traf_iq[perms[1][2]] * value
            term_31_C_2S_EA_virt_arb += C_prefactor * AC_IP_B_left[perms[1][0], :] * CS_EA_A_a[:, perms[1][1]] * CS_A_AB_traf_iq[perms[1][2]] * value
            term_31_X_2S_EA_virt_arb += X_prefactor * AC_IP_B_left[perms[0][0], :] * CS_EA_A_a[:, perms[0][1]] * CS_A_AB_traf_iq[perms[0][2]] * value

            term_14_C_2S_virt_occ_arb += C_prefactor * SCS_A_BB_pa[perms[2][0], perms[2][1]] * CA_EA_B_right[:, perms[2][2]] * term_24
            term_14_X_2S_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms[1][0], perms[1][1]] * CA_EA_B_right[:, perms[1][2]] * term_24
            term_14_C_2S_virt_occ_arb += C_prefactor * SCS_A_BB_pa[perms[1][0], perms[1][1]] * CA_EA_B_right[:, perms[1][2]] * term_24
            term_14_X_2S_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * CA_EA_B_right[:, perms[0][2]] * term_24

            term_43_C_3S_IP_IP_arb_EA_arb += C_prefactor * CA_EA_B_left[:, perms[1][0]] * CS_EA_A_q[:, perms[1][1]] * SC_IP_A_p[perms[1][2], :] * term_SC_IP_B_i
            term_43_X_3S_IP_IP_arb_EA_arb += X_prefactor * CA_EA_B_left[:, perms[2][0]] * CS_EA_A_q[:, perms[2][1]] * SC_IP_A_p[perms[2][2], :] * term_SC_IP_B_i
            term_43_C_3S_IP_IP_arb_EA_arb += C_prefactor * CA_EA_B_left[:, perms[0][0]] * CS_EA_A_q[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_SC_IP_B_i
            term_43_X_3S_IP_IP_arb_EA_arb += X_prefactor * CA_EA_B_left[:, perms[0][0]] * CS_EA_A_q[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_SC_IP_B_i

            term_31_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * CS_EA_A_a[:, perms[1][2]] * term_SC_IP_B_p
            term_31_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * CS_EA_A_a[:, perms[2][2]] * term_SC_IP_B_p
            term_31_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_EA_A_a[:, perms[0][2]] * term_SC_IP_B_p
            term_31_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * CS_EA_A_a[:, perms[1][2]] * term_SC_IP_B_p

            term_13_C_2S_virt_EA_arb += C_prefactor * AC_IP_B_right[perms[2][0], :] * CS_EA_A_q[:, perms[2][1]] * CS_A_AB_traf[perms[2][2]] * value
            term_13_X_2S_virt_EA_arb += X_prefactor * AC_IP_B_right[perms[1][0], :] * CS_EA_A_q[:, perms[1][1]] * CS_A_AB_traf[perms[1][2]] * value
            term_13_C_2S_virt_EA_arb += C_prefactor * AC_IP_B_right[perms[1][0], :] * CS_EA_A_q[:, perms[1][1]] * CS_A_AB_traf[perms[1][2]] * value
            term_13_X_2S_virt_EA_arb += X_prefactor * AC_IP_B_right[perms[0][0], :] * CS_EA_A_q[:, perms[0][1]] * CS_A_AB_traf[perms[0][2]] * value

            term_41_C_2S_occ_virt_arb += C_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * CA_EA_B_left[:, perms[2][2]] * term_42
            term_41_X_2S_occ_virt_arb += X_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * CA_EA_B_left[:, perms[1][2]] * term_42
            term_41_C_2S_occ_virt_arb += C_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * CA_EA_B_left[:, perms[1][2]] * term_42
            term_41_X_2S_occ_virt_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CA_EA_B_left[:, perms[0][2]] * term_42

            term_21_C_3S_A_occ_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * SC_B_AB_traf_pa[perms[2][2]] * value
            term_21_X_3S_A_occ_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * SC_B_AB_traf_pa[perms[1][2]] * value
            term_21_C_3S_A_occ_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * SC_B_AB_traf_pa[perms[1][2]] * value
            term_21_X_3S_A_occ_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_B_AB_traf_pa[perms[0][2]] * value

            term_13_C_2S_virt_occ_arb += C_prefactor * SCS_A_BB_pa[perms[1][0], perms[1][1]] * AC_IP_B_right[perms[1][2], :] * term_23
            term_13_X_2S_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms[2][0], perms[2][1]] * AC_IP_B_right[perms[2][2], :] * term_23
            term_13_C_2S_virt_occ_arb += C_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], :] * term_23
            term_13_X_2S_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], :] * term_23

            term_41_C_2S_occ_IP_arb += C_prefactor * SC_IP_A_p[perms[1][0], :] * CA_EA_B_left[:, perms[1][1]] * SC_A_BA_traf[perms[1][2]] * value
            term_41_X_2S_occ_IP_arb += X_prefactor * SC_IP_A_p[perms[2][0], :] * CA_EA_B_left[:, perms[2][1]] * SC_A_BA_traf[perms[2][2]] * value
            term_41_C_2S_occ_IP_arb += C_prefactor * SC_IP_A_p[perms[0][0], :] * CA_EA_B_left[:, perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value
            term_41_X_2S_occ_IP_arb += X_prefactor * SC_IP_A_p[perms[0][0], :] * CA_EA_B_left[:, perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value

            term_14_C_4S_IP_virt_occ_arb_EA_arb += C_prefactor * SCS_A_BB_pa[perms[1][0], perms[1][1]] * SC_IP_A_i[perms[1][2], :] * term_CS_EA_B_q
            term_14_X_4S_IP_virt_occ_arb_EA_arb += X_prefactor * SCS_A_BB_pa[perms[2][0], perms[2][1]] * SC_IP_A_i[perms[2][2], :] * term_CS_EA_B_q
            term_14_C_4S_IP_virt_occ_arb_EA_arb += C_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * SC_IP_A_i[perms[0][2], :] * term_CS_EA_B_q
            term_14_X_4S_IP_virt_occ_arb_EA_arb += X_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * SC_IP_A_i[perms[0][2], :] * term_CS_EA_B_q

            term_21_C_SC_occ += C_prefactor * SC_A_BA_traf[perms[1][0]] * ACA_B_left[perms[1][1], perms[1][2]] * value
            term_21_X_SC_occ += X_prefactor * SC_A_BA_traf[perms[1][0]] * ACA_B_left[perms[1][1], perms[1][2]] * value
            term_21_C_SC_occ += C_prefactor * SC_A_BA_traf[perms[2][0]] * ACA_B_left[perms[2][1], perms[2][2]] * value
            term_21_X_SC_occ += X_prefactor * SC_A_BA_traf[perms[0][0]] * ACA_B_left[perms[0][1], perms[0][2]] * value        

            term_12_C_CS_virt += C_prefactor * CS_A_AB_traf[perms[2][1]] * ACA_B_right[perms[2][0], perms[2][2]] * value        
            term_12_X_CS_virt += X_prefactor * CS_A_AB_traf[perms[1][1]] * ACA_B_right[perms[1][0], perms[1][2]] * value
            term_12_C_CS_virt += C_prefactor * CS_A_AB_traf[perms[0][1]] * ACA_B_right[perms[0][0], perms[0][2]] * value
            term_12_X_CS_virt += X_prefactor * CS_A_AB_traf[perms[2][1]] * ACA_B_right[perms[2][0], perms[2][2]] * value

            term_13_C_4S_IP_virt_occ_arb_EA_arb += C_prefactor * CS_EA_A_q[:, perms[2][0]] * SCS_A_BB_pa[perms[2][1], perms[2][2]] * term_SC_IP_B_i 
            term_13_X_4S_IP_virt_occ_arb_EA_arb += X_prefactor * CS_EA_A_q[:, perms[1][0]] * SCS_A_BB_pa[perms[1][1], perms[1][2]] * term_SC_IP_B_i 
            term_13_C_4S_IP_virt_occ_arb_EA_arb += C_prefactor * CS_EA_A_q[:, perms[0][0]] * SCS_A_BB_pa[perms[0][1], perms[0][2]] * term_SC_IP_B_i 
            term_13_X_4S_IP_virt_occ_arb_EA_arb += X_prefactor * CS_EA_A_q[:, perms[2][0]] * SCS_A_BB_pa[perms[2][1], perms[2][2]] * term_SC_IP_B_i 

            term_14_C_SC_IP_CS += C_prefactor * CS_A_AB_traf[perms[1][0]] * SC_IP_A_i[perms[1][1], 0] * CA_EA_B_right[0, perms[1][2]] * value 
            term_14_X_SC_IP_CS += X_prefactor * CS_A_AB_traf[perms[2][0]] * SC_IP_A_i[perms[2][1], 0] * CA_EA_B_right[0, perms[2][2]] * value
            term_14_C_SC_IP_CS += C_prefactor * CS_A_AB_traf[perms[2][0]] * SC_IP_A_i[perms[2][1], 0] * CA_EA_B_right[0, perms[2][2]] * value    
            term_14_X_SC_IP_CS += X_prefactor * CS_A_AB_traf[perms[0][0]] * SC_IP_A_i[perms[0][2], 0] * CA_EA_B_right[0, perms[0][1]] * value

            term_32_X_CS_EA_SC += X_prefactor * SC_B_AB_traf[perms[1][0]] * CS_EA_A_a[0, perms[1][1]] * AC_IP_B_left[perms[1][2], 0] * value        
            term_32_C_CS_EA_SC += C_prefactor * SC_B_AB_traf[perms[1][0]] * CS_EA_A_a[0, perms[1][1]] * AC_IP_B_left[perms[1][2], 0] * value
            term_32_X_CS_EA_SC += X_prefactor * SC_B_AB_traf[perms[0][0]] * CS_EA_A_a[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * value            
            term_32_C_CS_EA_SC += C_prefactor * SC_B_AB_traf[perms[2][0]] * CS_EA_A_a[0, perms[2][1]] * AC_IP_B_left[perms[2][2], 0] * value

            term_42_C += C_prefactor * ACA_B_right[perms[2][0], perms[2][1]] * CA_EA_B_left[0, perms[2][2]] * term_42
            term_42_X += X_prefactor * ACA_B_right[perms[1][0], perms[1][1]] * CA_EA_B_left[0, perms[1][2]] * term_42
            term_42_C += C_prefactor * ACA_B_right[perms[1][0], perms[1][1]] * CA_EA_B_left[0, perms[1][2]] * term_42
            term_42_X += X_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * CA_EA_B_left[0, perms[0][2]] * term_42            

            term_43_C_1S_EA_arb += C_prefactor * CS_EA_A_q[0, perms[2][0]] * AC_IP_B_right[perms[2][1], 0] * CA_EA_B_left[0, perms[2][2]] * term_42
            term_43_X_1S_EA_arb += X_prefactor * CS_EA_A_q[0, perms[1][0]] * AC_IP_B_right[perms[1][1], 0] * CA_EA_B_left[0, perms[1][2]] * term_42
            term_43_C_1S_EA_arb += C_prefactor * CS_EA_A_q[0, perms[0][0]] * AC_IP_B_right[perms[0][1], 0] * CA_EA_B_left[0, perms[0][2]] * term_42
            term_43_X_1S_EA_arb += X_prefactor * CS_EA_A_q[0, perms[0][0]] * AC_IP_B_right[perms[0][1], 0] * CA_EA_B_left[0, perms[0][2]] * term_42

            term_21_X_CS_virt_arb += X_prefactor * ACA_B_left[perms[1][0], perms[1][1]] * CS_A_AB_traf_iq[perms[1][2]] * value
            term_21_C_CS_virt_arb += C_prefactor * ACA_B_left[perms[1][0], perms[1][1]] * CS_A_AB_traf_iq[perms[1][2]] * value
            term_21_X_CS_virt_arb += X_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * CS_A_AB_traf_iq[perms[0][2]] * value
            term_21_C_CS_virt_arb += C_prefactor * ACA_B_left[perms[2][0], perms[2][1]] * CS_A_AB_traf_iq[perms[2][2]] * value        

            term_24_C += C_prefactor * ACA_B_left[perms[2][0], perms[2][1]] * CA_EA_B_right[0, perms[2][2]] * term_24
            term_24_X += X_prefactor * ACA_B_left[perms[1][0], perms[1][1]] * CA_EA_B_right[0, perms[1][2]] * term_24
            term_24_C += C_prefactor * ACA_B_left[perms[1][0], perms[1][1]] * CA_EA_B_right[0, perms[1][2]] * term_24
            term_24_X += X_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * CA_EA_B_right[0, perms[0][2]] * term_24            

            term_34_C_CS_EA += C_prefactor * CA_EA_B_right[0, perms[2][0]] * CS_EA_A_a[0, perms[2][1]] * AC_IP_B_left[perms[2][2], 0] * term_24
            term_34_X_CS_EA += X_prefactor * CA_EA_B_right[0, perms[1][0]] * CS_EA_A_a[0, perms[1][1]] * AC_IP_B_left[perms[1][2], 0] * term_24
            term_34_C_CS_EA += C_prefactor * CA_EA_B_right[0, perms[1][0]] * CS_EA_A_a[0, perms[1][1]] * AC_IP_B_left[perms[1][2], 0] * term_24
            term_34_X_CS_EA += X_prefactor * CA_EA_B_right[0, perms[0][0]] * CS_EA_A_a[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * term_24

            term_34_X_SC_IP += X_prefactor * AC_IP_B_left[perms[1][0], 0] * SC_IP_A_i[perms[1][1], 0] * CA_EA_B_right[0, perms[1][2]] * term_32
            term_34_C_SC_IP += C_prefactor * AC_IP_B_left[perms[1][0], 0] * SC_IP_A_i[perms[1][1], 0] * CA_EA_B_right[0, perms[1][2]] * term_32
            term_34_X_SC_IP += X_prefactor * AC_IP_B_left[perms[0][0], 0] * SC_IP_A_i[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * term_32
            term_34_C_SC_IP += C_prefactor * AC_IP_B_left[perms[2][0], 0] * SC_IP_A_i[perms[2][1], 0] * CA_EA_B_right[0, perms[2][2]] * term_32            
            
            term_31_X_CS_EA_SC += X_prefactor * SC_A_BA_traf[perms[1][0]] * AC_IP_B_left[perms[1][1], 0] * CS_EA_A_a[0, perms[1][2]] * value
            term_31_C_CS_EA_SC += C_prefactor * SC_A_BA_traf[perms[1][0]] * AC_IP_B_left[perms[1][1], 0] * CS_EA_A_a[0, perms[1][2]] * value
            term_31_C_CS_EA_SC += C_prefactor * SC_A_BA_traf[perms[2][0]] * AC_IP_B_left[perms[2][1], 0] * CS_EA_A_a[0, perms[2][2]] * value            
            term_31_X_CS_EA_SC += X_prefactor * SC_A_BA_traf[perms[0][0]] * AC_IP_B_left[perms[0][1], 0] * CS_EA_A_a[0, perms[0][2]] * value        
            
            term_24_C_2S_IP += C_prefactor * CS_B_BA_traf[perms[1][0]] * SC_IP_A_i[perms[1][1], 0] * CA_EA_B_right[0, perms[1][2]] * value    
            term_24_X_2S_IP += X_prefactor * CS_B_BA_traf[perms[0][0]] * SC_IP_A_i[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * value
            term_24_C_2S_IP += C_prefactor * CS_B_BA_traf[perms[0][2]] * SC_IP_A_i[perms[0][1], 0] * CA_EA_B_right[0, perms[0][0]] * value 
            term_24_X_2S_IP += X_prefactor * CS_B_BA_traf[perms[2][0]] * SC_IP_A_i[perms[2][2], 0] * CA_EA_B_right[0, perms[2][1]] * value            
            
            term_32_C += C_prefactor * ACA_B_right[perms[1][0], perms[1][1]] * AC_IP_B_left[perms[1][2], 0] * term_32
            term_32_X += X_prefactor * ACA_B_right[perms[2][0], perms[2][1]] * AC_IP_B_left[perms[2][2], 0] * term_32
            term_32_C += C_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * term_32
            term_32_X += X_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * term_32

            term_43_C_1S_IP_arb += C_prefactor * SC_IP_A_p[perms[1][0], 0] * CA_EA_B_left[0, perms[1][1]] * AC_IP_B_right[perms[1][2], 0] * term_23            
            term_43_X_1S_IP_arb += X_prefactor * SC_IP_A_p[perms[2][0], 0] * CA_EA_B_left[0, perms[2][1]] * AC_IP_B_right[perms[2][2], 0] * term_23            
            term_43_C_1S_IP_arb += C_prefactor * SC_IP_A_p[perms[0][0], 0] * CA_EA_B_left[0, perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23            
            term_43_X_1S_IP_arb += X_prefactor * SC_IP_A_p[perms[0][0], 0] * CA_EA_B_left[0, perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23            
            
            term_23_C += C_prefactor * ACA_B_left[perms[1][0], perms[1][1]] * AC_IP_B_right[perms[1][2], 0] * term_23            
            term_23_X += X_prefactor * ACA_B_left[perms[2][0], perms[2][1]] * AC_IP_B_right[perms[2][2], 0] * term_23    
            term_23_C += C_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23
            term_23_X += X_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23 
    
            term_41_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * SC_IP_A_p[perms[1][2], :] * term_CS_EA_B_a
            term_41_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * SC_IP_A_p[perms[2][2], :] * term_CS_EA_B_a
            term_41_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_CS_EA_B_a
            term_41_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_CS_EA_B_a
            
            term_43_C_3S_EA_EA_arb_IP_arb += C_prefactor * AC_IP_B_right[perms[1][0], :] * CS_EA_A_q[:, perms[1][1]] * SC_IP_A_p[perms[1][2], :] * term_CS_EA_B_a
            term_43_X_3S_EA_EA_arb_IP_arb += X_prefactor * AC_IP_B_right[perms[2][0], :] * CS_EA_A_q[:, perms[2][1]] * SC_IP_A_p[perms[2][2], :] * term_CS_EA_B_a
            term_43_C_3S_EA_EA_arb_IP_arb += C_prefactor * AC_IP_B_right[perms[0][0], :] * CS_EA_A_q[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_CS_EA_B_a
            term_43_X_3S_EA_EA_arb_IP_arb += X_prefactor * AC_IP_B_right[perms[0][0], :] * CS_EA_A_q[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_CS_EA_B_a

        else:
            term_31_X_CS_EA_SC += X_prefactor * SC_A_BA_traf[perms[1][0]] * CS_EA_A_a[0, perms[1][1]] * AC_IP_B_left[perms[1][2], 0] * value  
            term_31_C_CS_EA_SC += C_prefactor * SC_A_BA_traf[perms[2][0]] * CS_EA_A_a[0, perms[2][1]] * AC_IP_B_left[perms[2][2], 0] * value
            
            term_24_X_2S_IP += X_prefactor * CS_B_BA_traf[perms[1][2]] * SC_IP_A_i[perms[1][0], 0] * CA_EA_B_right[0, perms[1][1]] * value
            term_24_C_2S_IP += C_prefactor * CS_B_BA_traf[perms[2][0]] * SC_IP_A_i[perms[2][1], 0] * CA_EA_B_right[0, perms[2][2]] * value
                        
            term_34_X_SC_IP += X_prefactor * AC_IP_B_left[perms[0][0], 0] * SC_IP_A_i[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * term_32 
            term_34_C_SC_IP += C_prefactor * AC_IP_B_left[perms[2][0], 0] * SC_IP_A_i[perms[2][1], 0] * CA_EA_B_right[0, perms[2][2]] * term_32
            
            term_14_X_SC_IP_CS += X_prefactor * CS_A_AB_traf[perms[1][0]] * SC_IP_A_i[perms[1][1], 0] * CA_EA_B_right[0, perms[1][2]] * value
            term_14_C_SC_IP_CS += C_prefactor * CS_A_AB_traf[perms[2][0]] * SC_IP_A_i[perms[2][1], 0] * CA_EA_B_right[0, perms[2][2]] * value

            term_32_C_CS_EA_SC += C_prefactor * SC_B_AB_traf[perms[2][0]] * CS_EA_A_a[0, perms[2][1]] * AC_IP_B_left[perms[2][2], 0] * value  
            term_32_X_CS_EA_SC += X_prefactor * SC_B_AB_traf[perms[0][0]] * CS_EA_A_a[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * value            
            
            term_34_X_CS_EA += X_prefactor * CA_EA_B_right[0, perms[0][0]] * CS_EA_A_a[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] *  term_24
            term_34_C_CS_EA += C_prefactor * CA_EA_B_right[0, perms[2][0]] * CS_EA_A_a[0, perms[2][1]] * AC_IP_B_left[perms[2][2], 0] *  term_24
            
            term_43_X_1S_EA_arb += X_prefactor * CS_EA_A_q[0, perms[0][0]] * AC_IP_B_right[perms[0][1], 0] * CA_EA_B_left[0, perms[0][2]] * term_42
            term_43_C_1S_EA_arb += C_prefactor * CS_EA_A_q[0, perms[1][0]] * AC_IP_B_right[perms[1][1], 0] * CA_EA_B_left[0, perms[1][2]] * term_42


            term_42_C += C_prefactor * ACA_B_right[perms[2][0], perms[2][1]] * CA_EA_B_left[0, perms[2][2]] * term_42
            term_42_X += X_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * CA_EA_B_left[0, perms[0][2]] * term_42  

            term_21_X_CS_virt_arb += X_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * CS_A_AB_traf_iq[perms[0][2]] * value
            term_21_C_CS_virt_arb += C_prefactor * ACA_B_left[perms[2][0], perms[2][1]] * CS_A_AB_traf_iq[perms[2][2]] * value

            term_24_C += C_prefactor * ACA_B_left[perms[2][0], perms[2][1]] * CA_EA_B_right[0, perms[2][2]] * term_24
            term_24_X += X_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * CA_EA_B_right[0, perms[0][2]] * term_24  

            term_31_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_EA_A_a[:, perms[0][2]] * term_SC_IP_B_p
            term_31_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * CS_EA_A_a[:, perms[2][2]] * term_SC_IP_B_p

            term_43_C_3S_IP_IP_arb_EA_arb += C_prefactor * CA_EA_B_left[:, perms[0][0]] * CS_EA_A_q[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_SC_IP_B_i
            term_43_X_3S_IP_IP_arb_EA_arb += X_prefactor * CA_EA_B_left[:, perms[1][0]] * CS_EA_A_q[:, perms[1][1]] * SC_IP_A_p[perms[1][2], :] * term_SC_IP_B_i

            term_14_C_2S_virt_occ_arb += C_prefactor * SCS_A_BB_pa[perms[2][0], perms[2][1]] * CA_EA_B_right[:, perms[2][2]] * term_24
            term_14_X_2S_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * CA_EA_B_right[:, perms[0][2]] * term_24

            term_31_C_2S_EA_virt_arb += C_prefactor * AC_IP_B_left[perms[2][0], :] * CS_EA_A_a[:, perms[2][1]] * CS_A_AB_traf_iq[perms[2][2]] * value
            term_31_X_2S_EA_virt_arb += X_prefactor * AC_IP_B_left[perms[0][0], :] * CS_EA_A_a[:, perms[0][1]] * CS_A_AB_traf_iq[perms[0][2]] * value

            term_21_X_SC_occ += X_prefactor * SC_A_BA_traf[perms[0][0]] * ACA_B_left[perms[0][1], perms[0][2]] * value
            term_21_C_SC_occ += C_prefactor * SC_A_BA_traf[perms[2][0]] * ACA_B_left[perms[2][1], perms[2][2]] * value

            term_13_C_4S_IP_virt_occ_arb_EA_arb += C_prefactor * CS_EA_A_q[:, perms[0][0]] * SCS_A_BB_pa[perms[0][2], perms[0][1]] * term_SC_IP_B_i 
            term_13_X_4S_IP_virt_occ_arb_EA_arb += X_prefactor * CS_EA_A_q[:, perms[2][0]] * SCS_A_BB_pa[perms[2][1], perms[2][2]] * term_SC_IP_B_i 

            term_12_X_CS_virt += X_prefactor * CS_A_AB_traf[perms[2][1]] * ACA_B_right[perms[2][0], perms[2][2]] * value
            term_12_C_CS_virt += C_prefactor * CS_A_AB_traf[perms[1][1]] * ACA_B_right[perms[1][0], perms[1][2]] * value

            term_32_C += C_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * term_32
            term_32_X += X_prefactor * ACA_B_right[perms[1][0], perms[1][1]] * AC_IP_B_left[perms[1][2], 0] * term_32

            term_41_C_2S_occ_IP_arb += C_prefactor * SC_IP_A_p[perms[0][0], :] * CA_EA_B_left[:, perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value
            term_41_X_2S_occ_IP_arb += X_prefactor * SC_IP_A_p[perms[1][0], :] * CA_EA_B_left[:, perms[1][1]] * SC_A_BA_traf[perms[1][2]] * value

            term_21_C_3S_A_occ_B_virt_A_virt_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value
            term_21_X_3S_A_occ_B_virt_A_virt_arb += X_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * CS_B_BA_traf[perms[1][2]] * value

            term_14_C_2S_IP_occ_arb += C_prefactor * SC_IP_A_i[perms[0][0], :] * CA_EA_B_right[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value
            term_14_X_2S_IP_occ_arb += X_prefactor * SC_IP_A_i[perms[1][0], :] * CA_EA_B_right[:, perms[1][1]] * SC_A_BA_traf_pa[perms[1][2]] * value

            term_13_C_2S_EA_arb_occ_arb += C_prefactor * AC_IP_B_right[perms[0][0], :] * CS_EA_A_q[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value
            term_13_X_2S_EA_arb_occ_arb += X_prefactor * AC_IP_B_right[perms[1][0], :] * CS_EA_A_q[:, perms[1][1]] * SC_A_BA_traf_pa[perms[1][2]] * value

            term_31_C_2S_occ_virt_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], :] * term_32
            term_31_X_2S_occ_virt_arb += X_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * AC_IP_B_left[perms[1][2], :] * term_32

            term_43_C_1S_IP_arb += C_prefactor * SC_IP_A_p[perms[0][0], 0] * CA_EA_B_left[0, perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23            
            term_43_X_1S_IP_arb += X_prefactor * SC_IP_A_p[perms[1][0], 0] * CA_EA_B_left[0, perms[1][1]] * AC_IP_B_right[perms[1][2], 0] * term_23   

            term_23_C += C_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23
            term_23_X += X_prefactor * ACA_B_left[perms[1][0], perms[1][1]] * AC_IP_B_right[perms[1][2], 0] * term_23
            
            term_41_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_CS_EA_B_a
            term_41_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SCS_A_BB_iq[perms[1][0], perms[1][1]] * SC_IP_A_p[perms[1][2], :] * term_CS_EA_B_a

            term_43_C_3S_EA_EA_arb_IP_arb += C_prefactor * AC_IP_B_right[perms[0][0], :] * CS_EA_A_q[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_CS_EA_B_a
            term_43_X_3S_EA_EA_arb_IP_arb += X_prefactor * AC_IP_B_right[perms[1][0], :] * CS_EA_A_q[:, perms[1][1]] * SC_IP_A_p[perms[1][2], :] * term_CS_EA_B_a

            term_14_C_4S_IP_virt_occ_arb_EA_arb += C_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * SC_IP_A_i[perms[0][2], :] * term_CS_EA_B_q
            term_14_X_4S_IP_virt_occ_arb_EA_arb += X_prefactor * SCS_A_BB_pa[perms[1][0], perms[1][1]] * SC_IP_A_i[perms[1][2], :] * term_CS_EA_B_q

            term_13_C_2S_virt_occ_arb += C_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], :] * term_23
            term_13_X_2S_virt_occ_arb += X_prefactor * SCS_A_BB_pa[perms[1][0], perms[1][1]] * AC_IP_B_right[perms[1][2], :] * term_23

            term_21_C_3S_A_occ_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * SC_B_AB_traf_pa[perms[2][2]] * value
            term_21_X_3S_A_occ_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_B_AB_traf_pa[perms[0][2]] * value

            term_41_C_2S_occ_virt_arb += C_prefactor * SCS_A_BB_iq[perms[2][0], perms[2][1]] * CA_EA_B_left[:, perms[2][2]] * term_42
            term_41_X_2S_occ_virt_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CA_EA_B_left[:, perms[0][2]] * term_42

            term_13_C_2S_virt_EA_arb += C_prefactor * AC_IP_B_right[perms[2][0], :] * CS_EA_A_q[:, perms[2][1]] * CS_A_AB_traf[perms[2][2]] * value
            term_13_X_2S_virt_EA_arb += X_prefactor * AC_IP_B_right[perms[0][0], :] * CS_EA_A_q[:, perms[0][1]] * CS_A_AB_traf[perms[0][2]] * value

        return term_32_X, term_32_C, term_42_X, term_42_C, term_23_X, term_23_C, term_24_X, term_24_C, term_31_X_CS_EA_SC, term_31_C_CS_EA_SC, term_14_X_SC_IP_CS, term_14_C_SC_IP_CS,\
                term_32_C_CS_EA_SC, term_32_X_CS_EA_SC, term_24_C_2S_IP, term_24_X_2S_IP, term_21_C_SC_occ, term_21_X_SC_occ, \
                term_12_C_CS_virt, term_12_X_CS_virt, term_34_C_CS_EA, term_34_X_CS_EA, term_34_C_SC_IP, term_34_X_SC_IP, \
                term_43_C_1S_EA_arb, term_43_X_1S_EA_arb, term_43_C_1S_IP_arb, term_43_X_1S_IP_arb, term_21_C_CS_virt_arb, term_21_X_CS_virt_arb, term_31_C_2S_occ_virt_arb, term_31_X_2S_occ_virt_arb, term_13_C_2S_EA_arb_occ_arb, term_13_X_2S_EA_arb_occ_arb, term_14_C_2S_IP_occ_arb, term_14_X_2S_IP_occ_arb,\
                    term_21_C_3S_A_occ_B_virt_A_virt_arb, term_21_X_3S_A_occ_B_virt_A_virt_arb, term_31_C_2S_EA_virt_arb, term_31_X_2S_EA_virt_arb, term_13_C_4S_IP_virt_occ_arb_EA_arb, term_13_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_14_C_2S_virt_occ_arb, term_14_X_2S_virt_occ_arb, term_43_C_3S_IP_IP_arb_EA_arb, term_43_X_3S_IP_IP_arb_EA_arb, term_31_C_4S_occ_EA_virt_arb_IP_arb, term_31_X_4S_occ_EA_virt_arb_IP_arb, \
                    term_13_C_2S_virt_EA_arb, term_13_X_2S_virt_EA_arb, term_41_C_2S_occ_virt_arb, term_41_X_2S_occ_virt_arb, term_21_C_3S_A_occ_A_virt_arb_B_occ_arb, term_21_X_3S_A_occ_A_virt_arb_B_occ_arb, \
                    term_13_C_2S_virt_occ_arb, term_13_X_2S_virt_occ_arb, term_41_C_2S_occ_IP_arb, term_41_X_2S_occ_IP_arb, term_14_C_4S_IP_virt_occ_arb_EA_arb, term_14_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_41_C_4S_occ_EA_virt_arb_IP_arb, term_41_X_4S_occ_EA_virt_arb_IP_arb, term_43_C_3S_EA_EA_arb_IP_arb, term_43_X_3S_EA_EA_arb_IP_arb 
    else:
        return print("No oV is not implemented for BBBA_3")
        #if is_type_1:#type_1 = 6010
        #    term_32_C += C_prefactor * ACA_B[perms[1][0], perms[1][1]] * AC_IP_B[perms[1][2], 0] * term_32
        #    term_42_X += X_prefactor * ACA_B[perms[1][0], perms[1][1]] * CA_EA_B[0, perms[1][2]] * term_42
        #    term_42_C += C_prefactor * ACA_B[perms[1][0], perms[1][1]] * CA_EA_B[0, perms[1][2]] * term_42
        #    
        #    term_42_X += X_prefactor * ACA_B[perms[2][0], perms[2][1]] * CA_EA_B[:,perms[2][2]] * term_42
        #    term_32_X += X_prefactor * ACA_B[perms[2][0], perms[2][1]] * AC_IP_B[perms[2][2], 0] * term_32    
        #            
        #    term_42_C += C_prefactor * ACA_B[perms[0][0], perms[0][1]] * CA_EA_B[0, perms[0][2]] * term_42
        #    term_32_C += C_prefactor * ACA_B[perms[0][0], perms[0][1]] * AC_IP_B[perms[0][2],:] * term_32
        #    term_32_X += X_prefactor * ACA_B[perms[0][0], perms[0][1]] * AC_IP_B[perms[0][2],:] * term_32 
    #
        #else:
        #    
        #    term_42_X += X_prefactor * ACA_B[perms[2][0], perms[2][1]] * CA_EA_B[0, perms[2][2]] * term_42
        #    term_32_C += C_prefactor * ACA_B[perms[0][0], perms[0][1]] * AC_IP_B[perms[0][2], 0] * term_32
        #    
        #    
        #    term_32_X += X_prefactor * ACA_B[perms[1][0], perms[1][1]] * AC_IP_B[perms[1][2], 0] * term_32
        #    term_42_C += C_prefactor * ACA_B[perms[0][0], perms[0][1]] * CA_EA_B[0, perms[0][2]] * term_42  
        #    
        #return term_32_X, term_32_C, term_42_X, term_42_C


def calc_term_BBBA_2(indexes,
SC_A_BA_traf   ,
SC_A_BA_traf_pa,
CS_A_AB_traf   ,
CS_A_AB_traf_iq,
SC_B_AB_traf   ,
SC_B_AB_traf_pa,
CS_B_BA_traf   ,
term_32        ,
term_42        ,
term_23        ,
term_24        ,
term_CS_EA_B_a ,
term_CS_EA_B_q ,
term_SC_IP_B_i ,
term_SC_IP_B_p ,
CS_EA_A_a      ,
SC_IP_A_i      ,
CS_EA_A_q      ,
SC_IP_A_p      ,
ACA_B_right    ,
SCS_A_BB_iq    ,
ACA_B_left     ,
SCS_A_BB_pa    ,
AC_IP_B_right  ,
CA_EA_B_right  ,
AC_IP_B_left   ,
CA_EA_B_left   , value):
    term_32_X = np.float64(0.)
    term_32_C = np.float64(0.)
    term_42_X = np.float64(0.)
    term_42_C = np.float64(0.)

    term_23_X = np.float64(0.)
    term_23_C = np.float64(0.)
    term_24_X = np.float64(0.)
    term_24_C = np.float64(0.)

    term_32_C_CS_EA_SC = np.float64(0.)
    term_32_X_CS_EA_SC = np.float64(0.)
        
    term_31_C_CS_EA_SC = np.float64(0.)
    term_31_X_CS_EA_SC = np.float64(0.)
    
    term_14_C_SC_IP_CS = np.float64(0.)
    term_14_X_SC_IP_CS = np.float64(0.)
    term_21_C_CS_virt_arb = np.float64(0.)
    term_21_X_CS_virt_arb = np.float64(0.)
    term_24_C_2S_IP = np.float64(0.)
    term_24_X_2S_IP = np.float64(0.)
    
    term_21_C_SC_occ = np.float64(0.)
    term_21_X_SC_occ = np.float64(0.)
    
    term_12_C_CS_virt = np.float64(0.)
    term_12_X_CS_virt = np.float64(0.)    
    
    term_34_C_CS_EA = np.float64(0.)
    term_34_X_CS_EA = np.float64(0.)    

    term_34_C_SC_IP = np.float64(0.)
    term_34_X_SC_IP = np.float64(0.)  

    term_43_C_1S_EA_arb = np.float64(0.)
    term_43_X_1S_EA_arb = np.float64(0.)
    term_43_C_1S_IP_arb = np.float64(0.)
    term_43_X_1S_IP_arb = np.float64(0.)  

    term_31_C_2S_occ_virt_arb = np.float64(0.)
    term_31_X_2S_occ_virt_arb = np.float64(0.)
    term_13_C_2S_EA_arb_occ_arb = np.float64(0.)
    term_13_X_2S_EA_arb_occ_arb = np.float64(0.)
    term_14_C_2S_IP_occ_arb = np.float64(0.)
    term_14_X_2S_IP_occ_arb = np.float64(0.)
    term_21_C_3S_A_occ_B_virt_A_virt_arb = np.float64(0.)
    term_21_X_3S_A_occ_B_virt_A_virt_arb = np.float64(0.)
    term_31_C_2S_EA_virt_arb = np.float64(0.)
    term_31_X_2S_EA_virt_arb = np.float64(0.)
    term_13_C_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_13_X_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_14_C_2S_virt_occ_arb = np.float64(0.)
    term_14_X_2S_virt_occ_arb = np.float64(0.)
    term_43_C_3S_IP_IP_arb_EA_arb = np.float64(0.)
    term_43_X_3S_IP_IP_arb_EA_arb = np.float64(0.)
    term_31_C_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_31_X_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_13_C_2S_virt_EA_arb = np.float64(0.)
    term_13_X_2S_virt_EA_arb = np.float64(0.)
    term_41_C_2S_occ_virt_arb = np.float64(0.)
    term_41_X_2S_occ_virt_arb = np.float64(0.)
    term_21_C_3S_A_occ_A_virt_arb_B_occ_arb = np.float64(0.)
    term_21_X_3S_A_occ_A_virt_arb_B_occ_arb = np.float64(0.)
    term_13_C_2S_virt_occ_arb = np.float64(0.)
    term_13_X_2S_virt_occ_arb = np.float64(0.)
    term_41_C_2S_occ_IP_arb = np.float64(0.)
    term_41_X_2S_occ_IP_arb = np.float64(0.)
    term_14_C_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    term_14_X_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    
    term_41_C_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_41_X_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    term_43_C_3S_EA_EA_arb_IP_arb = np.float64(0.)
    term_43_X_3S_EA_EA_arb_IP_arb = np.float64(0.)

    #perms = list(distinct_permutations(indexes))

    a, b, c = indexes

    p0 = (a, b, c)
    perms = [p0]

    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)

    term_41_C_4S_occ_EA_virt_arb_IP_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_CS_EA_B_a
    term_41_X_4S_occ_EA_virt_arb_IP_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_CS_EA_B_a
    term_43_C_3S_EA_EA_arb_IP_arb += C_prefactor * SC_IP_A_p[perms[0][0], :] * AC_IP_B_right[perms[0][1], :] * CS_EA_A_q[:, perms[0][2]] * term_CS_EA_B_a
    term_43_X_3S_EA_EA_arb_IP_arb += X_prefactor * SC_IP_A_p[perms[0][0], :] * AC_IP_B_right[perms[0][1], :] * CS_EA_A_q[:, perms[0][2]] * term_CS_EA_B_a
    term_31_C_2S_occ_virt_arb               += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], :] * term_32
    term_31_X_2S_occ_virt_arb               += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], :] * term_32
    term_13_C_2S_EA_arb_occ_arb             += C_prefactor * SC_A_BA_traf_pa[perms[0][0]] *  CS_EA_A_q[:, perms[0][1]] * AC_IP_B_right[perms[0][2], :] * value
    term_13_X_2S_EA_arb_occ_arb             += X_prefactor * SC_A_BA_traf_pa[perms[0][0]] *  CS_EA_A_q[:, perms[0][1]] * AC_IP_B_right[perms[0][2], :] * value
    term_14_C_2S_IP_occ_arb                 += C_prefactor * SC_A_BA_traf_pa[perms[0][0]] * CA_EA_B_right[:, perms[0][1]] * SC_IP_A_i[perms[0][2], :] * value
    term_14_X_2S_IP_occ_arb                 += X_prefactor * SC_A_BA_traf_pa[perms[0][0]] * CA_EA_B_right[:, perms[0][1]] * SC_IP_A_i[perms[0][2], :] * value
    term_21_C_3S_A_occ_B_virt_A_virt_arb    += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value
    term_21_X_3S_A_occ_B_virt_A_virt_arb    += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value
    term_31_C_2S_EA_virt_arb                += C_prefactor * CS_A_AB_traf_iq[perms[0][0]] * CS_EA_A_a[:, perms[0][1]] * AC_IP_B_left[perms[0][2], :] * value
    term_31_X_2S_EA_virt_arb                += X_prefactor * CS_A_AB_traf_iq[perms[0][0]] * CS_EA_A_a[:, perms[0][1]] * AC_IP_B_left[perms[0][2], :] * value
    term_13_C_4S_IP_virt_occ_arb_EA_arb     += C_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] *  CS_EA_A_q[:, perms[0][2]] * term_SC_IP_B_i 
    term_13_X_4S_IP_virt_occ_arb_EA_arb     += X_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] *  CS_EA_A_q[:, perms[0][2]] * term_SC_IP_B_i 
    term_14_C_2S_virt_occ_arb               += C_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * CA_EA_B_right[:, perms[0][2]] * term_24
    term_14_X_2S_virt_occ_arb               += X_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * CA_EA_B_right[:, perms[0][2]] * term_24
    term_43_C_3S_IP_IP_arb_EA_arb           += C_prefactor * CA_EA_B_left[:, perms[0][0]] * CS_EA_A_q[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_SC_IP_B_i
    term_43_X_3S_IP_IP_arb_EA_arb           += X_prefactor * CA_EA_B_left[:, perms[0][0]] * CS_EA_A_q[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * term_SC_IP_B_i
    term_31_C_4S_occ_EA_virt_arb_IP_arb     += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_EA_A_a[:, perms[0][2]] * term_SC_IP_B_p
    term_31_X_4S_occ_EA_virt_arb_IP_arb     += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CS_EA_A_a[:, perms[0][2]] * term_SC_IP_B_p
    term_13_C_2S_virt_EA_arb                += C_prefactor * CS_A_AB_traf[perms[0][0]] * CS_EA_A_q[:, perms[0][1]] * AC_IP_B_right[perms[0][2], :] * value
    term_13_X_2S_virt_EA_arb                += X_prefactor * CS_A_AB_traf[perms[0][0]] * CS_EA_A_q[:, perms[0][1]] * AC_IP_B_right[perms[0][2], :] * value
    term_41_C_2S_occ_virt_arb               += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CA_EA_B_left[:, perms[0][2]] * term_42
    term_41_X_2S_occ_virt_arb               += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * CA_EA_B_left[:, perms[0][2]] * term_42
    term_21_C_3S_A_occ_A_virt_arb_B_occ_arb += C_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_B_AB_traf_pa[perms[0][2]] * value
    term_21_X_3S_A_occ_A_virt_arb_B_occ_arb += X_prefactor * SCS_A_BB_iq[perms[0][0], perms[0][1]] * SC_B_AB_traf_pa[perms[0][2]] * value
    term_13_C_2S_virt_occ_arb               += C_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], :] * term_23
    term_13_X_2S_virt_occ_arb               += X_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], :] * term_23
    term_41_C_2S_occ_IP_arb                 += C_prefactor * SC_A_BA_traf[perms[0][0]] * CA_EA_B_left[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * value
    term_41_X_2S_occ_IP_arb                 += X_prefactor * SC_A_BA_traf[perms[0][0]] * CA_EA_B_left[:, perms[0][1]] * SC_IP_A_p[perms[0][2], :] * value
    term_14_C_4S_IP_virt_occ_arb_EA_arb     += C_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * SC_IP_A_i[perms[0][2], :] * term_CS_EA_B_q
    term_14_X_4S_IP_virt_occ_arb_EA_arb     += X_prefactor * SCS_A_BB_pa[perms[0][0], perms[0][1]] * SC_IP_A_i[perms[0][2], :] * term_CS_EA_B_q
    term_31_C_CS_EA_SC += C_prefactor * SC_A_BA_traf[perms[0][0]] * CS_EA_A_a[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * value       
    term_31_X_CS_EA_SC += X_prefactor * SC_A_BA_traf[perms[0][0]] * CS_EA_A_a[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * value 
    term_21_C_SC_occ  += C_prefactor * SC_A_BA_traf[perms[0][0]] * ACA_B_left[perms[0][1], perms[0][2]] * value       
    term_21_X_SC_occ  += X_prefactor * SC_A_BA_traf[perms[0][0]] * ACA_B_left[perms[0][1], perms[0][2]] * value 
    term_12_C_CS_virt += C_prefactor * CS_A_AB_traf[perms[0][0]] * ACA_B_right[perms[0][1], perms[0][2]] * value       
    term_12_X_CS_virt += X_prefactor * CS_A_AB_traf[perms[0][0]] * ACA_B_right[perms[0][1], perms[0][2]] * value 
    term_21_C_CS_virt_arb += C_prefactor * CS_A_AB_traf_iq[perms[0][0]] * ACA_B_left[perms[0][1], perms[0][2]] * value       
    term_21_X_CS_virt_arb += X_prefactor * CS_A_AB_traf_iq[perms[0][0]] * ACA_B_left[perms[0][1], perms[0][2]] * value 
    term_32_C_CS_EA_SC += C_prefactor * SC_B_AB_traf[perms[0][0]] * CS_EA_A_a[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * value       
    term_32_X_CS_EA_SC += X_prefactor * SC_B_AB_traf[perms[0][0]] * CS_EA_A_a[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * value 
    
    term_14_C_SC_IP_CS += C_prefactor * CS_A_AB_traf[perms[0][0]] * SC_IP_A_i[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * value
    term_14_X_SC_IP_CS += X_prefactor * CS_A_AB_traf[perms[0][0]] * SC_IP_A_i[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * value
    
    term_24_C_2S_IP += C_prefactor * CS_B_BA_traf[perms[0][0]] * SC_IP_A_i[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * value
    term_24_X_2S_IP += X_prefactor * CS_B_BA_traf[perms[0][0]] * SC_IP_A_i[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * value        
    term_34_C_SC_IP += C_prefactor * SC_IP_A_i[perms[0][0], 0] * AC_IP_B_left[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * term_32
    term_34_X_SC_IP += X_prefactor * SC_IP_A_i[perms[0][0], 0] * AC_IP_B_left[perms[0][1], 0] * CA_EA_B_right[0, perms[0][2]] * term_32
    term_34_C_CS_EA += C_prefactor * CS_EA_A_a[0, perms[0][0]] * CA_EA_B_right[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * term_24       
    term_34_X_CS_EA += X_prefactor * CS_EA_A_a[0, perms[0][0]] * CA_EA_B_right[0, perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * term_24 
    
    term_42_C += C_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * CA_EA_B_left[0, perms[0][2]] * term_42
    term_42_X += X_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * CA_EA_B_left[0, perms[0][2]] * term_42
    
    term_32_C += C_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * term_32
    term_32_X += X_prefactor * ACA_B_right[perms[0][0], perms[0][1]] * AC_IP_B_left[perms[0][2], 0] * term_32
    term_24_C += C_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * CA_EA_B_right[0, perms[0][2]] * term_24
    term_24_X += X_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * CA_EA_B_right[0, perms[0][2]] * term_24
    term_43_C_1S_IP_arb += C_prefactor * SC_IP_A_p[perms[0][0], 0] * CA_EA_B_left[0, perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23            
    term_43_X_1S_IP_arb += X_prefactor * SC_IP_A_p[perms[0][0], 0] * CA_EA_B_left[0, perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23   
    term_43_X_1S_EA_arb += X_prefactor * CS_EA_A_q[0, perms[0][0]] * CA_EA_B_left[0, perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_42
    term_43_C_1S_EA_arb += C_prefactor * CS_EA_A_q[0, perms[0][0]] * CA_EA_B_left[0, perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_42
    term_23_C += C_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23       
    term_23_X += X_prefactor * ACA_B_left[perms[0][0], perms[0][1]] * AC_IP_B_right[perms[0][2], 0] * term_23         
    return term_32_C_CS_EA_SC, term_32_X_CS_EA_SC, term_31_X_CS_EA_SC, term_31_C_CS_EA_SC, term_14_X_SC_IP_CS, term_14_C_SC_IP_CS,\
        term_32_X, term_32_C, term_42_X, term_42_C, term_23_X, term_23_C, term_24_X, term_24_C, term_24_C_2S_IP, term_24_X_2S_IP, term_21_C_SC_occ, term_21_X_SC_occ, \
        term_12_C_CS_virt, term_12_X_CS_virt, term_34_C_CS_EA, term_34_X_CS_EA, term_34_C_SC_IP, term_34_X_SC_IP, term_43_C_1S_EA_arb, term_43_X_1S_EA_arb, term_43_C_1S_IP_arb, term_43_X_1S_IP_arb, \
        term_21_C_CS_virt_arb, term_21_X_CS_virt_arb, term_31_C_2S_occ_virt_arb, term_31_X_2S_occ_virt_arb, term_13_C_2S_EA_arb_occ_arb, term_13_X_2S_EA_arb_occ_arb, term_14_C_2S_IP_occ_arb, term_14_X_2S_IP_occ_arb,\
        term_21_C_3S_A_occ_B_virt_A_virt_arb, term_21_X_3S_A_occ_B_virt_A_virt_arb, term_31_C_2S_EA_virt_arb, term_31_X_2S_EA_virt_arb, term_13_C_4S_IP_virt_occ_arb_EA_arb, term_13_X_4S_IP_virt_occ_arb_EA_arb, \
        term_14_C_2S_virt_occ_arb, term_14_X_2S_virt_occ_arb, term_43_C_3S_IP_IP_arb_EA_arb, term_43_X_3S_IP_IP_arb_EA_arb, term_31_C_4S_occ_EA_virt_arb_IP_arb, term_31_X_4S_occ_EA_virt_arb_IP_arb, \
        term_13_C_2S_virt_EA_arb, term_13_X_2S_virt_EA_arb, term_41_C_2S_occ_virt_arb, term_41_X_2S_occ_virt_arb, term_21_C_3S_A_occ_A_virt_arb_B_occ_arb, term_21_X_3S_A_occ_A_virt_arb_B_occ_arb, \
        term_13_C_2S_virt_occ_arb, term_13_X_2S_virt_occ_arb, term_41_C_2S_occ_IP_arb, term_41_X_2S_occ_IP_arb, term_14_C_4S_IP_virt_occ_arb_EA_arb, term_14_X_4S_IP_virt_occ_arb_EA_arb, \
        term_41_C_4S_occ_EA_virt_arb_IP_arb, term_41_X_4S_occ_EA_virt_arb_IP_arb, term_43_C_3S_EA_EA_arb_IP_arb, term_43_X_3S_EA_EA_arb_IP_arb


def _freeze_term32_bucket(bucket: dict) -> dict:
    out: dict[int, dict] = {}
    for uc, d in bucket.items():
        uc_i = int(uc)
        out[uc_i] = {
            "C": {tuple(k): float(v) for k, v in (d.get("C", {}) or {}).items()},
            "X": {tuple(k): float(v) for k, v in (d.get("X", {}) or {}).items()},
            "C_examples": {tuple(k): list(v) for k, v in (d.get("C_examples", {}) or {}).items()},
            "X_examples": {tuple(k): list(v) for k, v in (d.get("X_examples", {}) or {}).items()},
        }
    return out


def   calc_BBBA(twoelint_block, SC_A_BA_ia   ,
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
ACA_B_right  ,
ACA_A_left   ,
SCS_A_BB_iq    ,
ACA_B_left     ,
SCS_A_BB_pa    ,
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
    
    fc_32_coul = np.float64(0.)
    fc_32_exch = np.float64(0.)
    
    fc_32_C_SC_EA_CS = np.float64(0.)
    fc_32_X_SC_EA_CS = np.float64(0.)    
    
    fc_42_coul = np.float64(0.)
    fc_42_exch = np.float64(0.)
    
    fc_23_coul = np.float64(0.)
    fc_23_exch = np.float64(0.)
    fc_24_coul = np.float64(0.)
    fc_24_exch = np.float64(0.)

    fc_31_C_CS_EA_SC = np.float64(0.)
    fc_31_X_CS_EA_SC = np.float64(0.)
    
    fc_14_C_SC_IP_CS = np.float64(0.)
    fc_14_X_SC_IP_CS = np.float64(0.)
    
    fc_24_C_2S_IP = np.float64(0.)
    fc_24_X_2S_IP = np.float64(0.)
    
    fc_21_C_SC_occ = np.float64(0.)
    fc_21_X_SC_occ = np.float64(0.)  
    
    fc_12_C_CS_virt = np.float64(0.)
    fc_12_X_CS_virt = np.float64(0.) 
    
    fc_21_C_CS_virt_arb = np.float64(0.)
    fc_21_X_CS_virt_arb = np.float64(0.) 

    fc_34_C_CS_EA = np.float64(0.)
    fc_34_X_CS_EA = np.float64(0.)  
    
    fc_34_C_SC_IP = np.float64(0.)
    fc_34_X_SC_IP = np.float64(0.)  
    
    fc_43_C_1S_EA_arb = np.float64(0.)
    fc_43_X_1S_EA_arb = np.float64(0.)
    fc_43_C_1S_IP_arb = np.float64(0.)
    fc_43_X_1S_IP_arb = np.float64(0.)
    
    fc_coul_31_2S_occ_virt_arb = np.float64(0.)           
    fc_exch_31_2S_occ_virt_arb = np.float64(0.)            
    fc_coul_13_2S_EA_arb_occ_arb = np.float64(0.)
    fc_exch_13_2S_EA_arb_occ_arb = np.float64(0.)
    fc_coul_14_2S_IP_occ_arb = np.float64(0.)
    fc_exch_14_2S_IP_occ_arb = np.float64(0.)
    fc_coul_21_3S_A_occ_B_virt_A_virt_arb = np.float64(0.)
    fc_exch_21_3S_A_occ_B_virt_A_virt_arb = np.float64(0.)

    fc_coul_31_2S_EA_virt_arb = np.float64(0.)
    fc_exch_31_2S_EA_virt_arb = np.float64(0.)
    fc_coul_13_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    fc_exch_13_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    fc_coul_14_2S_virt_occ_arb = np.float64(0.)
    fc_exch_14_2S_virt_occ_arb = np.float64(0.)
    fc_coul_43_3S_IP_IP_arb_EA_arb = np.float64(0.)
    fc_exch_43_3S_IP_IP_arb_EA_arb = np.float64(0.)

    fc_coul_31_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    fc_exch_31_4S_occ_EA_virt_arb_IP_arb = np.float64(0.)
    fc_coul_13_2S_virt_EA_arb = np.float64(0.)
    fc_exch_13_2S_virt_EA_arb = np.float64(0.)
    fc_coul_41_2S_occ_virt_arb = np.float64(0.)
    fc_exch_41_2S_occ_virt_arb = np.float64(0.)
    fc_coul_21_3S_A_occ_A_virt_arb_B_occ_arb = np.float64(0.)
    fc_exch_21_3S_A_occ_A_virt_arb_B_occ_arb = np.float64(0.)
    fc_coul_13_2S_virt_occ_arb = np.float64(0.)
    fc_exch_13_2S_virt_occ_arb = np.float64(0.)
    fc_coul_41_2S_occ_IP_arb = np.float64(0.)
    fc_exch_41_2S_occ_IP_arb = np.float64(0.)
    fc_coul_14_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)
    fc_exch_14_4S_IP_virt_occ_arb_EA_arb = np.float64(0.)

    fc_coul_41_4S_occ_EA_virt_arb_IP_arb = np.float64(0.0)
    fc_exch_41_4S_occ_EA_virt_arb_IP_arb = np.float64(0.0)
    fc_coul_43_3S_EA_EA_arb_IP_arb = np.float64(0.0)
    fc_exch_43_3S_EA_EA_arb_IP_arb = np.float64(0.0)

#! BAD fc_coul_13_4S_IP_virt_occ_arb_EA_arb 
#! fc_coul_13_2S_virt_EA_arb
#! fc_coul_21_3S_A_occ_A_virt_arb_B_occ_arb

#C_AO_ABBB_31_2S_virt_occ_arb               SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * CA_EA_A_left[:,c] * AC_IP_B_left[k-N_SAO_A, :] 
#X_AO_ABBB_31_2S_virt_occ_arb               SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * CA_EA_A_left[:,c] * AC_IP_B_left[k-N_SAO_A, :] 
#C_AO_ABBB_13_2S_EA_arb_occ_arb             SC_A_BA_pa[p-N_SAO_A,a] *  CS_EA_A_q[:,q-N_SAO_A] * AC_IP_B_right[k-N_SAO_A, :]
#X_AO_ABBB_13_2S_EA_arb_occ_arb             SC_A_BA_pa[p-N_SAO_A,a] *  CS_EA_A_q[:,q-N_SAO_A] * AC_IP_B_right[k-N_SAO_A, :]
#C_AO_ABBB_14_2S_IP_occ_arb                 SC_A_BA_pa[p-N_SAO_A,a] * CA_EA_B_right[:,d-N_SAO_A] * SC_IP_A_i[n-N_SAO_A, :]
#X_AO_ABBB_14_2S_IP_occ_arb                 SC_A_BA_pa[p-N_SAO_A,a] * CA_EA_B_right[:,d-N_SAO_A] * SC_IP_A_i[n-N_SAO_A, :]
#C_AO_ABBB_21_3S_A_occ_B_virt_A_virt_arb    SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * CS_B_BA_ia[j-N_SAO_A, e]
#X_AO_ABBB_21_3S_A_occ_B_virt_A_virt_arb    SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * CS_B_BA_ia[j-N_SAO_A, e]

#C_AO_BABB_31_2S_EA_virt_arb                CS_A_AB_iq[i,q-N_SAO_A] * CS_EA_A_a[:,e-N_SAO_A] * AC_IP_B_left[k-N_SAO_A, :] 
#X_AO_BABB_31_2S_EA_virt_arb                CS_A_AB_iq[i,q-N_SAO_A] * CS_EA_A_a[:,e-N_SAO_A] * AC_IP_B_left[k-N_SAO_A, :] 
#C_AO_BABB_13_4S_IP_virt_occ_arb_EA_arb     SCS_A_BB_pa[p-N_SAO_A,f-N_SAO_A] *  CS_EA_A_q[:,q-N_SAO_A] * SC_IP_B_i[n, :]
#X_AO_BABB_13_4S_IP_virt_occ_arb_EA_arb     SCS_A_BB_pa[p-N_SAO_A,f-N_SAO_A] *  CS_EA_A_q[:,q-N_SAO_A] * SC_IP_B_i[n, :]
#C_AO_BABB_14_2S_virt_occ_arb               SCS_A_BB_pa[p-N_SAO_A,f-N_SAO_A] * CA_EA_B_right[:,d-N_SAO_A] * AC_IP_A_right[l, :]
#X_AO_BABB_14_2S_virt_occ_arb               SCS_A_BB_pa[p-N_SAO_A,f-N_SAO_A] * CA_EA_B_right[:,d-N_SAO_A] * AC_IP_A_right[l, :]
#C_AO_BABB_43_3S_IP_IP_arb_EA_arb           CA_EA_B_left[:,d-N_SAO_A] * SC_IP_B_i[n, :] * CS_EA_A_q[:,q-N_SAO_A] * SC_IP_A_p[p-N_SAO_A, :]
#X_AO_BABB_43_3S_IP_IP_arb_EA_arb           CA_EA_B_left[:,d-N_SAO_A] * SC_IP_B_i[n, :] * CS_EA_A_q[:,q-N_SAO_A] * SC_IP_A_p[p-N_SAO_A, :]

#C_AO_BBAB_13_2S_virt_EA_arb                CS_A_AB_ia[i,f-N_SAO_A] *  CS_EA_A_q[:,q-N_SAO_A] * AC_IP_B_right[k-N_SAO_A, :]
#X_AO_BBAB_13_2S_virt_EA_arb                CS_A_AB_ia[i,f-N_SAO_A] *  CS_EA_A_q[:,q-N_SAO_A] * AC_IP_B_right[k-N_SAO_A, :]
#C_AO_BBAB_41_2S_occ_virt_arb               SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * CA_EA_B_left[:,d-N_SAO_A] * AC_IP_A_left[l, :]
#X_AO_BBAB_41_2S_occ_virt_arb               SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * CA_EA_B_left[:,d-N_SAO_A] * AC_IP_A_left[l, :]
#C_AO_BBAB_21_3S_A_occ_A_virt_arb_B_occ_arb SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * SC_B_AB_pa[p, b-N_SAO_A]
#X_AO_BBAB_21_3S_A_occ_A_virt_arb_B_occ_arb SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * SC_B_AB_pa[p, b-N_SAO_A]
#C_AO_BBAB_31_4S_occ_EA_virt_arb_IP_arb     SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * CS_EA_A_a[:,e-N_SAO_A] * SC_IP_B_p[p, :]
#X_AO_BBAB_31_4S_occ_EA_virt_arb_IP_arb     SCS_A_BB_iq[n-N_SAO_A,q-N_SAO_A] * CS_EA_A_a[:,e-N_SAO_A] * SC_IP_B_p[p, :]

#C_AO_BBBA_13_2S_virt_occ_arb               SCS_A_BB_pa[p-N_SAO_A,f-N_SAO_A] * CA_EA_A_right[:,c] * AC_IP_B_right[k-N_SAO_A, :]
#X_AO_BBBA_13_2S_virt_occ_arb               SCS_A_BB_pa[p-N_SAO_A,f-N_SAO_A] * CA_EA_A_right[:,c] * AC_IP_B_right[k-N_SAO_A, :]
#C_AO_BBBA_41_2S_occ_IP_arb                 SC_A_BA_ia[n-N_SAO_A,a] * CA_EA_B_left[:,d-N_SAO_A] * SC_IP_A_p[p-N_SAO_A, :]
#X_AO_BBBA_41_2S_occ_IP_arb                 SC_A_BA_ia[n-N_SAO_A,a] * CA_EA_B_left[:,d-N_SAO_A] * SC_IP_A_p[p-N_SAO_A, :]
#C_AO_BBBA_14_4S_IP_virt_occ_arb_EA_arb     SCS_A_BB_pa[p-N_SAO_A,f-N_SAO_A] * CS_EA_B_q[:,q] * SC_IP_A_i[n-N_SAO_A, :]
#X_AO_BBBA_14_4S_IP_virt_occ_arb_EA_arb     SCS_A_BB_pa[p-N_SAO_A,f-N_SAO_A] * CS_EA_B_q[:,q] * SC_IP_A_i[n-N_SAO_A, :]




    is_Ov = True


    for unique_count, twoelint_subblock in twoelint_block.items():
        if unique_count == 4:
            for row in twoelint_subblock:
                # This is needed, because how cfour writes the integrals
                i = int(row[0]); j = int(row[1]); k = int(row[2]); l = int(row[3])
                if tuple(sorted((i, j, k, l))) == (1, 72, 73, 74):
                    print('asd')
                # --- normalize by NBAS_A membership ---
                l_in_A = (l <= NBAS_A)
                j_in_A = (j <= NBAS_A)

                if l_in_A:
                    # row[0],row[1],row[2],row[3] = row[3], row[2], row[0], row[1]
                    i, j, k, l = k,l,i,j#l, k, i, j
                elif j_in_A:
                    # row[0], row[1] = row[1], row[0]
                    i, j = j, i
                else:
                    raise ValueError("problem")  # neither l nor j in A-block

                # --- compact j,k,l rule ---
                swap = (j < k) and (j <= l)
                bad  = (j >= k) and (k < l)
                if bad:
                    raise ValueError("problem 716")
                if swap:
                    k, l = l, k  # swap j<=l path

                # --- write back ---
                row[0], row[1], row[2], row[3] = i, j, k, l
                ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                
                if is_Ov:
                    SC_A_BA_traf = SC_A_BA_ia[:, ind_A[0]]
                    SC_A_BA_traf_pa = SC_A_BA_pa[:, ind_A[0]]
                    CS_A_AB_traf = CS_A_AB_ia[ind_A[0], :]
                    CS_A_AB_traf_iq = CS_A_AB_iq[ind_A[0], :]
                    SC_B_AB_traf = SC_B_AB_ia[ind_A[0], :]
                    SC_B_AB_traf_pa = SC_B_AB_pa[ind_A[0], :]
                    CS_B_BA_traf = CS_B_BA_ia[:, ind_A[0]]                    
                    term_32 = value * CA_EA_A_left[0, ind_A[0]]
                    term_42 = value * AC_IP_A_left[ind_A[0], 0]          
                    term_23 = value * CA_EA_A_right[0, ind_A[0]]
                    term_24 = value * AC_IP_A_right[ind_A[0], 0]      
                    term_CS_EA_B_a = value * CS_EA_B_a[0, ind_A[0]]
                    term_CS_EA_B_q = value * CS_EA_B_q[0, ind_A[0]]
                    term_SC_IP_B_i = value * SC_IP_B_i[ind_A[0], 0]
                    term_SC_IP_B_p = value * SC_IP_B_p[ind_A[0], 0]

                    term_32_X, term_32_C, term_42_X, term_42_C, term_23_X, term_23_C, term_24_X, term_24_C, term_31_X_CS_EA_SC, term_31_C_CS_EA_SC, term_14_X_SC_IP_CS, term_14_C_SC_IP_CS, term_32_X_CS_EA_SC, term_32_C_CS_EA_SC, \
                    term_24_C_2S_IP, term_24_X_2S_IP, term_21_C_SC_occ, term_21_X_SC_occ, term_12_C_CS_virt, term_12_X_CS_virt, term_34_C_CS_EA, term_34_X_CS_EA, term_34_C_SC_IP, term_34_X_SC_IP, \
                    term_43_C_1S_EA_arb, term_43_X_1S_EA_arb, term_43_C_1S_IP_arb, term_43_X_1S_IP_arb,  \
                    term_21_C_CS_virt_arb, term_21_X_CS_virt_arb, term_31_C_2S_occ_virt_arb, term_31_X_2S_occ_virt_arb, term_13_C_2S_EA_arb_occ_arb, term_13_X_2S_EA_arb_occ_arb, term_14_C_2S_IP_occ_arb, term_14_X_2S_IP_occ_arb,\
                    term_21_C_3S_A_occ_B_virt_A_virt_arb, term_21_X_3S_A_occ_B_virt_A_virt_arb, term_31_C_2S_EA_virt_arb, term_31_X_2S_EA_virt_arb, term_13_C_4S_IP_virt_occ_arb_EA_arb, term_13_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_14_C_2S_virt_occ_arb, term_14_X_2S_virt_occ_arb, term_43_C_3S_IP_IP_arb_EA_arb, term_43_X_3S_IP_IP_arb_EA_arb, term_31_C_4S_occ_EA_virt_arb_IP_arb, term_31_X_4S_occ_EA_virt_arb_IP_arb, \
                    term_13_C_2S_virt_EA_arb, term_13_X_2S_virt_EA_arb, term_41_C_2S_occ_virt_arb, term_41_X_2S_occ_virt_arb, term_21_C_3S_A_occ_A_virt_arb_B_occ_arb, term_21_X_3S_A_occ_A_virt_arb_B_occ_arb, \
                    term_13_C_2S_virt_occ_arb, term_13_X_2S_virt_occ_arb, term_41_C_2S_occ_IP_arb, term_41_X_2S_occ_IP_arb, term_14_C_4S_IP_virt_occ_arb_EA_arb, term_14_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_41_C_4S_occ_EA_virt_arb_IP_arb, term_41_X_4S_occ_EA_virt_arb_IP_arb, term_43_C_3S_EA_EA_arb_IP_arb, term_43_X_3S_EA_EA_arb_IP_arb \
                            = calc_term_BBBA_4(ind_B, 
                                                SC_A_BA_traf   ,
                                                SC_A_BA_traf_pa,
                                                CS_A_AB_traf   ,
                                                CS_A_AB_traf_iq,
                                                SC_B_AB_traf   ,
                                                SC_B_AB_traf_pa,
                                                CS_B_BA_traf   ,
                                                term_32        ,
                                                term_42        ,
                                                term_23        ,
                                                term_24        ,
                                                term_CS_EA_B_a ,
term_CS_EA_B_q ,
                                                term_SC_IP_B_i ,
                                                term_SC_IP_B_p ,
                                                CS_EA_A_a      ,
                                                SC_IP_A_i      ,
                                                CS_EA_A_q      ,
                                                SC_IP_A_p      ,
                                                ACA_B_right    ,
                                                SCS_A_BB_iq    ,
                                                ACA_B_left     ,
                                                SCS_A_BB_pa    ,
                                                AC_IP_B_right  ,
                                                CA_EA_B_right  ,
                                                AC_IP_B_left   ,
                                                CA_EA_B_left   , value=value)#,, dbg_detail=dbg_detail_32, dbg_meta=dbg_meta  twoel_AO_AAAB_canonical_2, ind_A[0], value)                    
                    
                    fc_coul_41_4S_occ_EA_virt_arb_IP_arb += term_41_C_4S_occ_EA_virt_arb_IP_arb
                    fc_exch_41_4S_occ_EA_virt_arb_IP_arb += term_41_X_4S_occ_EA_virt_arb_IP_arb
                    fc_coul_43_3S_EA_EA_arb_IP_arb += term_43_C_3S_EA_EA_arb_IP_arb
                    fc_exch_43_3S_EA_EA_arb_IP_arb += term_43_X_3S_EA_EA_arb_IP_arb
                    fc_coul_31_2S_occ_virt_arb += term_31_C_2S_occ_virt_arb  
                    fc_exch_31_2S_occ_virt_arb += term_31_X_2S_occ_virt_arb          
                    fc_coul_13_2S_EA_arb_occ_arb += term_13_C_2S_EA_arb_occ_arb            
                    fc_exch_13_2S_EA_arb_occ_arb += term_13_X_2S_EA_arb_occ_arb            
                    fc_coul_14_2S_IP_occ_arb += term_14_C_2S_IP_occ_arb                
                    fc_exch_14_2S_IP_occ_arb += term_14_X_2S_IP_occ_arb            
                    fc_coul_21_3S_A_occ_B_virt_A_virt_arb += term_21_C_3S_A_occ_B_virt_A_virt_arb   
                    fc_exch_21_3S_A_occ_B_virt_A_virt_arb += term_21_X_3S_A_occ_B_virt_A_virt_arb   
                    fc_coul_31_2S_EA_virt_arb += term_31_C_2S_EA_virt_arb               
                    fc_exch_31_2S_EA_virt_arb += term_31_X_2S_EA_virt_arb    
                    fc_coul_13_4S_IP_virt_occ_arb_EA_arb += term_13_C_4S_IP_virt_occ_arb_EA_arb    
                    fc_exch_13_4S_IP_virt_occ_arb_EA_arb += term_13_X_4S_IP_virt_occ_arb_EA_arb   
                    fc_coul_14_2S_virt_occ_arb += term_14_C_2S_virt_occ_arb              
                    fc_exch_14_2S_virt_occ_arb += term_14_X_2S_virt_occ_arb          
                    fc_coul_43_3S_IP_IP_arb_EA_arb += term_43_C_3S_IP_IP_arb_EA_arb          
                    fc_exch_43_3S_IP_IP_arb_EA_arb += term_43_X_3S_IP_IP_arb_EA_arb    
                    fc_coul_31_4S_occ_EA_virt_arb_IP_arb += term_31_C_4S_occ_EA_virt_arb_IP_arb    
                    fc_exch_31_4S_occ_EA_virt_arb_IP_arb += term_31_X_4S_occ_EA_virt_arb_IP_arb
                    fc_coul_13_2S_virt_EA_arb += term_13_C_2S_virt_EA_arb              
                    fc_exch_13_2S_virt_EA_arb += term_13_X_2S_virt_EA_arb              
                    fc_coul_41_2S_occ_virt_arb += term_41_C_2S_occ_virt_arb              
                    fc_exch_41_2S_occ_virt_arb += term_41_X_2S_occ_virt_arb
                    fc_coul_21_3S_A_occ_A_virt_arb_B_occ_arb += term_21_C_3S_A_occ_A_virt_arb_B_occ_arb
                    fc_exch_21_3S_A_occ_A_virt_arb_B_occ_arb += term_21_X_3S_A_occ_A_virt_arb_B_occ_arb
                    fc_coul_13_2S_virt_occ_arb += term_13_C_2S_virt_occ_arb              
                    fc_exch_13_2S_virt_occ_arb += term_13_X_2S_virt_occ_arb              
                    fc_coul_41_2S_occ_IP_arb += term_41_C_2S_occ_IP_arb                
                    fc_exch_41_2S_occ_IP_arb += term_41_X_2S_occ_IP_arb
                    fc_coul_14_4S_IP_virt_occ_arb_EA_arb += term_14_C_4S_IP_virt_occ_arb_EA_arb    
                    fc_exch_14_4S_IP_virt_occ_arb_EA_arb += term_14_X_4S_IP_virt_occ_arb_EA_arb
                    fc_43_C_1S_EA_arb += term_43_C_1S_EA_arb
                    fc_43_X_1S_EA_arb += term_43_X_1S_EA_arb
                    fc_43_C_1S_IP_arb += term_43_C_1S_IP_arb
                    fc_43_X_1S_IP_arb += term_43_X_1S_IP_arb

                    fc_21_C_CS_virt_arb += term_21_C_CS_virt_arb
                    fc_21_X_CS_virt_arb += term_21_X_CS_virt_arb

                    fc_23_coul += term_23_C
                    fc_23_exch += term_23_X
                    fc_24_coul += term_24_C
                    fc_24_exch += term_24_X
                    
                    fc_21_C_SC_occ  += term_21_C_SC_occ
                    fc_21_X_SC_occ  += term_21_X_SC_occ   
                    fc_12_C_CS_virt += term_12_C_CS_virt
                    fc_12_X_CS_virt += term_12_X_CS_virt     
                    fc_34_C_CS_EA   += term_34_C_CS_EA
                    fc_34_X_CS_EA   += term_34_X_CS_EA
                    fc_34_C_SC_IP   += term_34_C_SC_IP
                    fc_34_X_SC_IP   += term_34_X_SC_IP
                    
                    fc_24_C_2S_IP += term_24_C_2S_IP
                    fc_24_X_2S_IP += term_24_X_2S_IP
                    fc_32_C_SC_EA_CS += term_32_C_CS_EA_SC
                    fc_32_X_SC_EA_CS += term_32_X_CS_EA_SC
                    
                    fc_31_C_CS_EA_SC += term_31_C_CS_EA_SC 
                    fc_31_X_CS_EA_SC += term_31_X_CS_EA_SC
                    fc_14_C_SC_IP_CS += term_14_C_SC_IP_CS
                    fc_14_X_SC_IP_CS += term_14_X_SC_IP_CS
                    fc_32_coul += term_32_C
                    fc_42_coul += term_42_C
                    fc_32_exch += term_32_X
                    fc_42_exch += term_42_X

                else:
                    print('No ov is not implemented for BBBA_4')
                    
                    #term_32 = value * CA_EA_A[0, ind_A[0]]  
                    #term_42 = value * AC_IP_A[ind_A[0], 0]
                    #term_32_X, term_32_C, term_42_X, term_42_C = calc_term_BBBA_4(ind_B, term_32, term_42, ACA_B, CA_EA_B, AC_IP_B)#,  twoel_AO_AAAB_canonical_2, ind_A[0], value)
#
                    #fc_32_coul += term_32_C
                    #fc_42_coul += term_42_C
                    #fc_32_exch += term_32_X
                    #fc_42_exch += term_42_X
        elif unique_count == 3: 
            for row in twoelint_subblock:
                i = int(row[0]); j = int(row[1]); k = int(row[2]); l = int(row[3])

                # Normalize by block membership (only two legal cases)
                if l <= NBAS_A:
                    i, j, k, l = l,k,j,i#l, k, i, j        # rotate
                elif j <= NBAS_A:
                    i, j = i, j                    # swap i,j
                else:
                    raise ValueError("problem")    # unexpected layout

                # Type flag
                is_type_1 = (k != l)

                # Final swap rule
                if j == k:
                    k, l = l, k


                # Persist back
                row[0], row[1], row[2], row[3] = i, j, k, l
                ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                
                if is_Ov:
                    SC_A_BA_traf = SC_A_BA_ia[:, ind_A[0]]
                    SC_A_BA_traf_pa = SC_A_BA_pa[:, ind_A[0]]
                    CS_A_AB_traf = CS_A_AB_ia[ind_A[0], :]
                    CS_A_AB_traf_iq = CS_A_AB_iq[ind_A[0], :]
                    SC_B_AB_traf = SC_B_AB_ia[ind_A[0], :]
                    SC_B_AB_traf_pa = SC_B_AB_pa[ind_A[0], :]
                    CS_B_BA_traf = CS_B_BA_ia[:, ind_A[0]]                    
                    term_32 = value * CA_EA_A_left[0, ind_A[0]]
                    term_42 = value * AC_IP_A_left[ind_A[0], 0]          
                    term_23 = value * CA_EA_A_right[0, ind_A[0]]
                    term_24 = value * AC_IP_A_right[ind_A[0], 0]      
                    term_CS_EA_B_a = value * CS_EA_B_a[0, ind_A[0]]
                    term_CS_EA_B_q = value * CS_EA_B_q[0, ind_A[0]]
                    term_SC_IP_B_p = value * SC_IP_B_p[ind_A[0], 0] 
                    term_SC_IP_B_i = value * SC_IP_B_i[ind_A[0], 0] 

                    term_32_X, term_32_C, term_42_X, term_42_C, term_23_X, term_23_C, term_24_X, term_24_C, term_31_X_CS_EA_SC, term_31_C_CS_EA_SC, term_14_X_SC_IP_CS, term_14_C_SC_IP_CS,\
                    term_32_C_CS_EA_SC, term_32_X_CS_EA_SC, term_24_C_2S_IP, term_24_X_2S_IP, term_21_C_SC_occ, term_21_X_SC_occ, \
                    term_12_C_CS_virt, term_12_X_CS_virt, term_34_C_CS_EA, term_34_X_CS_EA, term_34_C_SC_IP, term_34_X_SC_IP,   \
                    term_43_C_1S_EA_arb, term_43_X_1S_EA_arb, term_43_C_1S_IP_arb, term_43_X_1S_IP_arb,  \
                    term_21_C_CS_virt_arb, term_21_X_CS_virt_arb, term_31_C_2S_occ_virt_arb, term_31_X_2S_occ_virt_arb, term_13_C_2S_EA_arb_occ_arb, term_13_X_2S_EA_arb_occ_arb, term_14_C_2S_IP_occ_arb, term_14_X_2S_IP_occ_arb,\
                    term_21_C_3S_A_occ_B_virt_A_virt_arb, term_21_X_3S_A_occ_B_virt_A_virt_arb, term_31_C_2S_EA_virt_arb, term_31_X_2S_EA_virt_arb, term_13_C_4S_IP_virt_occ_arb_EA_arb, term_13_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_14_C_2S_virt_occ_arb, term_14_X_2S_virt_occ_arb, term_43_C_3S_IP_IP_arb_EA_arb, term_43_X_3S_IP_IP_arb_EA_arb, term_31_C_4S_occ_EA_virt_arb_IP_arb, term_31_X_4S_occ_EA_virt_arb_IP_arb, \
                    term_13_C_2S_virt_EA_arb, term_13_X_2S_virt_EA_arb, term_41_C_2S_occ_virt_arb, term_41_X_2S_occ_virt_arb, term_21_C_3S_A_occ_A_virt_arb_B_occ_arb, term_21_X_3S_A_occ_A_virt_arb_B_occ_arb, \
                    term_13_C_2S_virt_occ_arb, term_13_X_2S_virt_occ_arb, term_41_C_2S_occ_IP_arb, term_41_X_2S_occ_IP_arb, term_14_C_4S_IP_virt_occ_arb_EA_arb, term_14_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_41_C_4S_occ_EA_virt_arb_IP_arb, term_41_X_4S_occ_EA_virt_arb_IP_arb, term_43_C_3S_EA_EA_arb_IP_arb, term_43_X_3S_EA_EA_arb_IP_arb \
                        = calc_term_BBBA_3(ind_B,   is_type_1     ,
                                                SC_A_BA_traf   ,
                                                SC_A_BA_traf_pa,
                                                CS_A_AB_traf   ,
                                                CS_A_AB_traf_iq,
                                                SC_B_AB_traf   ,
                                                SC_B_AB_traf_pa,
                                                CS_B_BA_traf   ,
                                                term_32        ,
                                                term_42        ,
                                                term_23        ,
                                                term_24        ,
                                                term_CS_EA_B_a ,
                                                term_CS_EA_B_q ,
                                                term_SC_IP_B_i ,
                                                term_SC_IP_B_p ,
                                                CS_EA_A_a      ,
                                                SC_IP_A_i      ,
                                                CS_EA_A_q      ,
                                                SC_IP_A_p      ,
                                                ACA_B_right    ,
                                                SCS_A_BB_iq    ,
                                                ACA_B_left     ,
                                                SCS_A_BB_pa    ,
                                                AC_IP_B_right  ,
                                                CA_EA_B_right  ,
                                                AC_IP_B_left   ,
                                                CA_EA_B_left   , 
                                                    value=value,)#, twoel_AO_AAAB_canonical, ind_A[0], value)
                    fc_coul_41_4S_occ_EA_virt_arb_IP_arb += term_41_C_4S_occ_EA_virt_arb_IP_arb
                    fc_exch_41_4S_occ_EA_virt_arb_IP_arb += term_41_X_4S_occ_EA_virt_arb_IP_arb
                    fc_coul_43_3S_EA_EA_arb_IP_arb += term_43_C_3S_EA_EA_arb_IP_arb
                    fc_exch_43_3S_EA_EA_arb_IP_arb += term_43_X_3S_EA_EA_arb_IP_arb
                    
                    fc_coul_31_2S_occ_virt_arb += term_31_C_2S_occ_virt_arb  
                    fc_exch_31_2S_occ_virt_arb += term_31_X_2S_occ_virt_arb          
                    fc_coul_13_2S_EA_arb_occ_arb += term_13_C_2S_EA_arb_occ_arb            
                    fc_exch_13_2S_EA_arb_occ_arb += term_13_X_2S_EA_arb_occ_arb            
                    fc_coul_14_2S_IP_occ_arb += term_14_C_2S_IP_occ_arb                
                    fc_exch_14_2S_IP_occ_arb += term_14_X_2S_IP_occ_arb            
                    fc_coul_21_3S_A_occ_B_virt_A_virt_arb += term_21_C_3S_A_occ_B_virt_A_virt_arb   
                    fc_exch_21_3S_A_occ_B_virt_A_virt_arb += term_21_X_3S_A_occ_B_virt_A_virt_arb   
                    fc_coul_31_2S_EA_virt_arb += term_31_C_2S_EA_virt_arb               
                    fc_exch_31_2S_EA_virt_arb += term_31_X_2S_EA_virt_arb    
                    fc_coul_13_4S_IP_virt_occ_arb_EA_arb += term_13_C_4S_IP_virt_occ_arb_EA_arb    
                    fc_exch_13_4S_IP_virt_occ_arb_EA_arb += term_13_X_4S_IP_virt_occ_arb_EA_arb   
                    fc_coul_14_2S_virt_occ_arb += term_14_C_2S_virt_occ_arb              
                    fc_exch_14_2S_virt_occ_arb += term_14_X_2S_virt_occ_arb          
                    fc_coul_43_3S_IP_IP_arb_EA_arb += term_43_C_3S_IP_IP_arb_EA_arb          
                    fc_exch_43_3S_IP_IP_arb_EA_arb += term_43_X_3S_IP_IP_arb_EA_arb    
                    fc_coul_31_4S_occ_EA_virt_arb_IP_arb += term_31_C_4S_occ_EA_virt_arb_IP_arb    
                    fc_exch_31_4S_occ_EA_virt_arb_IP_arb += term_31_X_4S_occ_EA_virt_arb_IP_arb
                    fc_coul_13_2S_virt_EA_arb += term_13_C_2S_virt_EA_arb              
                    fc_exch_13_2S_virt_EA_arb += term_13_X_2S_virt_EA_arb              
                    fc_coul_41_2S_occ_virt_arb += term_41_C_2S_occ_virt_arb              
                    fc_exch_41_2S_occ_virt_arb += term_41_X_2S_occ_virt_arb
                    fc_coul_21_3S_A_occ_A_virt_arb_B_occ_arb += term_21_C_3S_A_occ_A_virt_arb_B_occ_arb
                    fc_exch_21_3S_A_occ_A_virt_arb_B_occ_arb += term_21_X_3S_A_occ_A_virt_arb_B_occ_arb
                    fc_coul_13_2S_virt_occ_arb += term_13_C_2S_virt_occ_arb              
                    fc_exch_13_2S_virt_occ_arb += term_13_X_2S_virt_occ_arb              
                    fc_coul_41_2S_occ_IP_arb += term_41_C_2S_occ_IP_arb                
                    fc_exch_41_2S_occ_IP_arb += term_41_X_2S_occ_IP_arb
                    fc_coul_14_4S_IP_virt_occ_arb_EA_arb += term_14_C_4S_IP_virt_occ_arb_EA_arb    
                    fc_exch_14_4S_IP_virt_occ_arb_EA_arb += term_14_X_4S_IP_virt_occ_arb_EA_arb

                    fc_21_C_CS_virt_arb += term_21_C_CS_virt_arb
                    fc_21_X_CS_virt_arb += term_21_X_CS_virt_arb
                    
                    fc_43_C_1S_EA_arb += term_43_C_1S_EA_arb
                    fc_43_X_1S_EA_arb += term_43_X_1S_EA_arb
                    fc_43_C_1S_IP_arb += term_43_C_1S_IP_arb
                    fc_43_X_1S_IP_arb += term_43_X_1S_IP_arb
                    
                    fc_23_coul += term_23_C
                    fc_24_coul += term_24_C
                    fc_23_exch += term_23_X
                    fc_24_exch += term_24_X  
                    
                    fc_21_C_SC_occ  += term_21_C_SC_occ
                    fc_21_X_SC_occ  += term_21_X_SC_occ   
                    fc_12_C_CS_virt += term_12_C_CS_virt
                    fc_12_X_CS_virt += term_12_X_CS_virt  
                    fc_34_C_CS_EA   += term_34_C_CS_EA
                    fc_34_X_CS_EA   += term_34_X_CS_EA
                    fc_34_C_SC_IP   += term_34_C_SC_IP
                    fc_34_X_SC_IP   += term_34_X_SC_IP
                    
                    fc_24_C_2S_IP += term_24_C_2S_IP
                    fc_24_X_2S_IP += term_24_X_2S_IP
                    
                    fc_32_C_SC_EA_CS += term_32_C_CS_EA_SC
                    fc_32_X_SC_EA_CS += term_32_X_CS_EA_SC   
                    fc_31_C_CS_EA_SC += term_31_C_CS_EA_SC 
                    fc_31_X_CS_EA_SC += term_31_X_CS_EA_SC
                    fc_14_C_SC_IP_CS += term_14_C_SC_IP_CS
                    fc_14_X_SC_IP_CS += term_14_X_SC_IP_CS
                    fc_32_coul += term_32_C
                    fc_42_coul += term_42_C
                    fc_32_exch += term_32_X
                    fc_42_exch += term_42_X  
                else:
                    print('No ov is not implemented for BBBA_3')
                    #    
                    #term_32 = value * CA_EA_A[0, ind_A[0]]  
                    #term_42 = value * AC_IP_A[ind_A[0], 0]
                    #term_32_X, term_32_C, term_42_X, term_42_C = calc_term_BBBA_3(ind_B, term_32, term_42, ACA_B, CA_EA_B, AC_IP_B, is_type_1)#, twoel_AO_AAAB_canonical, ind_A[0], value)
#
                    #fc_32_coul += term_32_C
                    #fc_42_coul += term_42_C
                    #fc_32_exch += term_32_X
                    #fc_42_exch += term_42_X
        else:
            for row in twoelint_subblock:
                i = int(row[0]); j = int(row[1]); k = int(row[2]); l = int(row[3])
                ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                if is_Ov:
                    SC_A_BA_traf = SC_A_BA_ia[:, ind_A[0]]
                    SC_A_BA_traf_pa = SC_A_BA_pa[:, ind_A[0]]
                    CS_A_AB_traf = CS_A_AB_ia[ind_A[0], :]
                    CS_A_AB_traf_iq = CS_A_AB_iq[ind_A[0], :]
                    SC_B_AB_traf = SC_B_AB_ia[ind_A[0], :]
                    SC_B_AB_traf_pa = SC_B_AB_pa[ind_A[0], :]
                    CS_B_BA_traf = CS_B_BA_ia[:, ind_A[0]]                    
                    term_32 = value * CA_EA_A_left[0, ind_A[0]]
                    term_42 = value * AC_IP_A_left[ind_A[0], 0]          
                    term_23 = value * CA_EA_A_right[0, ind_A[0]]
                    term_24 = value * AC_IP_A_right[ind_A[0], 0]      
                    term_CS_EA_B_q = value * CS_EA_B_q[0, ind_A[0]]
                    term_CS_EA_B_a = value * CS_EA_B_a[0, ind_A[0]]
                    term_SC_IP_B_i = value * SC_IP_B_i[ind_A[0], 0]
                    term_SC_IP_B_p = value * SC_IP_B_p[ind_A[0], 0]

                    term_32_C_CS_EA_SC, term_32_X_CS_EA_SC, term_31_X_CS_EA_SC, term_31_C_CS_EA_SC, term_14_X_SC_IP_CS, term_14_C_SC_IP_CS,\
                    term_32_X, term_32_C, term_42_X, term_42_C, term_23_X, term_23_C, term_24_X, term_24_C, term_24_C_2S_IP, term_24_X_2S_IP, term_21_C_SC_occ, term_21_X_SC_occ, \
                    term_12_C_CS_virt, term_12_X_CS_virt, term_34_C_CS_EA, term_34_X_CS_EA, term_34_C_SC_IP, term_34_X_SC_IP, term_43_C_1S_EA_arb, term_43_X_1S_EA_arb, term_43_C_1S_IP_arb, term_43_X_1S_IP_arb,  \
                    term_21_C_CS_virt_arb, term_21_X_CS_virt_arb, term_31_C_2S_occ_virt_arb, term_31_X_2S_occ_virt_arb, term_13_C_2S_EA_arb_occ_arb, term_13_X_2S_EA_arb_occ_arb, term_14_C_2S_IP_occ_arb, term_14_X_2S_IP_occ_arb,\
                    term_21_C_3S_A_occ_B_virt_A_virt_arb, term_21_X_3S_A_occ_B_virt_A_virt_arb, term_31_C_2S_EA_virt_arb, term_31_X_2S_EA_virt_arb, term_13_C_4S_IP_virt_occ_arb_EA_arb, term_13_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_14_C_2S_virt_occ_arb, term_14_X_2S_virt_occ_arb, term_43_C_3S_IP_IP_arb_EA_arb, term_43_X_3S_IP_IP_arb_EA_arb, term_31_C_4S_occ_EA_virt_arb_IP_arb, term_31_X_4S_occ_EA_virt_arb_IP_arb, \
                    term_13_C_2S_virt_EA_arb, term_13_X_2S_virt_EA_arb, term_41_C_2S_occ_virt_arb, term_41_X_2S_occ_virt_arb, term_21_C_3S_A_occ_A_virt_arb_B_occ_arb, term_21_X_3S_A_occ_A_virt_arb_B_occ_arb, \
                    term_13_C_2S_virt_occ_arb, term_13_X_2S_virt_occ_arb, term_41_C_2S_occ_IP_arb, term_41_X_2S_occ_IP_arb, term_14_C_4S_IP_virt_occ_arb_EA_arb, term_14_X_4S_IP_virt_occ_arb_EA_arb, \
                    term_41_C_4S_occ_EA_virt_arb_IP_arb, term_41_X_4S_occ_EA_virt_arb_IP_arb, term_43_C_3S_EA_EA_arb_IP_arb, term_43_X_3S_EA_EA_arb_IP_arb \
                        = calc_term_BBBA_2(ind_B,
                                                SC_A_BA_traf   ,
                                                SC_A_BA_traf_pa,
                                                CS_A_AB_traf   ,
                                                CS_A_AB_traf_iq,
                                                SC_B_AB_traf   ,
                                                SC_B_AB_traf_pa,
                                                CS_B_BA_traf   ,
                                                term_32        ,
                                                term_42        ,
                                                term_23        ,
                                                term_24        ,
                                                term_CS_EA_B_a ,
                                                term_CS_EA_B_q ,
                                                term_SC_IP_B_i ,
                                                term_SC_IP_B_p ,
                                                CS_EA_A_a      ,
                                                SC_IP_A_i      ,
                                                CS_EA_A_q      ,
                                                SC_IP_A_p      ,
                                                ACA_B_right    ,
                                                SCS_A_BB_iq    ,
                                                ACA_B_left     ,
                                                SCS_A_BB_pa    ,
                                                AC_IP_B_right  ,
                                                CA_EA_B_right  ,
                                                AC_IP_B_left   ,
                                                CA_EA_B_left   , value=value) #dbg_detail=dbg_detail_32, dbg_meta=dbg_meta
                    
                    fc_coul_41_4S_occ_EA_virt_arb_IP_arb += term_41_C_4S_occ_EA_virt_arb_IP_arb
                    fc_exch_41_4S_occ_EA_virt_arb_IP_arb += term_41_X_4S_occ_EA_virt_arb_IP_arb
                    fc_coul_43_3S_EA_EA_arb_IP_arb += term_43_C_3S_EA_EA_arb_IP_arb
                    fc_exch_43_3S_EA_EA_arb_IP_arb += term_43_X_3S_EA_EA_arb_IP_arb

                    fc_coul_31_2S_occ_virt_arb += term_31_C_2S_occ_virt_arb  
                    fc_exch_31_2S_occ_virt_arb += term_31_X_2S_occ_virt_arb          
                    fc_coul_13_2S_EA_arb_occ_arb += term_13_C_2S_EA_arb_occ_arb            
                    fc_exch_13_2S_EA_arb_occ_arb += term_13_X_2S_EA_arb_occ_arb            
                    fc_coul_14_2S_IP_occ_arb += term_14_C_2S_IP_occ_arb                
                    fc_exch_14_2S_IP_occ_arb += term_14_X_2S_IP_occ_arb            
                    fc_coul_21_3S_A_occ_B_virt_A_virt_arb += term_21_C_3S_A_occ_B_virt_A_virt_arb   
                    fc_exch_21_3S_A_occ_B_virt_A_virt_arb += term_21_X_3S_A_occ_B_virt_A_virt_arb   
                    fc_coul_31_2S_EA_virt_arb += term_31_C_2S_EA_virt_arb               
                    fc_exch_31_2S_EA_virt_arb += term_31_X_2S_EA_virt_arb    
                    fc_coul_13_4S_IP_virt_occ_arb_EA_arb += term_13_C_4S_IP_virt_occ_arb_EA_arb    
                    fc_exch_13_4S_IP_virt_occ_arb_EA_arb += term_13_X_4S_IP_virt_occ_arb_EA_arb   
                    fc_coul_14_2S_virt_occ_arb += term_14_C_2S_virt_occ_arb              
                    fc_exch_14_2S_virt_occ_arb += term_14_X_2S_virt_occ_arb          
                    fc_coul_43_3S_IP_IP_arb_EA_arb += term_43_C_3S_IP_IP_arb_EA_arb          
                    fc_exch_43_3S_IP_IP_arb_EA_arb += term_43_X_3S_IP_IP_arb_EA_arb    
                    fc_coul_31_4S_occ_EA_virt_arb_IP_arb += term_31_C_4S_occ_EA_virt_arb_IP_arb    
                    fc_exch_31_4S_occ_EA_virt_arb_IP_arb += term_31_X_4S_occ_EA_virt_arb_IP_arb
                    fc_coul_13_2S_virt_EA_arb += term_13_C_2S_virt_EA_arb              
                    fc_exch_13_2S_virt_EA_arb += term_13_X_2S_virt_EA_arb              
                    fc_coul_41_2S_occ_virt_arb += term_41_C_2S_occ_virt_arb              
                    fc_exch_41_2S_occ_virt_arb += term_41_X_2S_occ_virt_arb
                    fc_coul_21_3S_A_occ_A_virt_arb_B_occ_arb += term_21_C_3S_A_occ_A_virt_arb_B_occ_arb
                    fc_exch_21_3S_A_occ_A_virt_arb_B_occ_arb += term_21_X_3S_A_occ_A_virt_arb_B_occ_arb
                    fc_coul_13_2S_virt_occ_arb += term_13_C_2S_virt_occ_arb              
                    fc_exch_13_2S_virt_occ_arb += term_13_X_2S_virt_occ_arb              
                    fc_coul_41_2S_occ_IP_arb += term_41_C_2S_occ_IP_arb                
                    fc_exch_41_2S_occ_IP_arb += term_41_X_2S_occ_IP_arb
                    fc_coul_14_4S_IP_virt_occ_arb_EA_arb += term_14_C_4S_IP_virt_occ_arb_EA_arb    
                    fc_exch_14_4S_IP_virt_occ_arb_EA_arb += term_14_X_4S_IP_virt_occ_arb_EA_arb

                    fc_21_C_CS_virt_arb += term_21_C_CS_virt_arb
                    fc_21_X_CS_virt_arb += term_21_X_CS_virt_arb
                    fc_43_C_1S_EA_arb += term_43_C_1S_EA_arb
                    fc_43_X_1S_EA_arb += term_43_X_1S_EA_arb
                    fc_43_C_1S_IP_arb += term_43_C_1S_IP_arb
                    fc_43_X_1S_IP_arb += term_43_X_1S_IP_arb

                    fc_21_C_SC_occ  += term_21_C_SC_occ
                    fc_21_X_SC_occ  += term_21_X_SC_occ   
                    fc_12_C_CS_virt += term_12_C_CS_virt
                    fc_12_X_CS_virt += term_12_X_CS_virt       
                    fc_34_C_CS_EA   += term_34_C_CS_EA
                    fc_34_X_CS_EA   += term_34_X_CS_EA
                    fc_34_C_SC_IP   += term_34_C_SC_IP
                    fc_34_X_SC_IP   += term_34_X_SC_IP
                    
                    fc_24_C_2S_IP += term_24_C_2S_IP
                    fc_24_X_2S_IP += term_24_X_2S_IP
                    
                    fc_32_C_SC_EA_CS += term_32_C_CS_EA_SC
                    fc_32_X_SC_EA_CS += term_32_X_CS_EA_SC
                    fc_31_C_CS_EA_SC += term_31_C_CS_EA_SC 
                    fc_31_X_CS_EA_SC += term_31_X_CS_EA_SC
                    fc_14_C_SC_IP_CS += term_14_C_SC_IP_CS
                    fc_14_X_SC_IP_CS += term_14_X_SC_IP_CS
                    fc_32_coul += term_32_C
                    fc_42_coul += term_42_C
                    fc_32_exch += term_32_X
                    fc_42_exch += term_42_X
                    
                    fc_23_coul += term_23_C
                    fc_24_coul += term_24_C
                    fc_23_exch += term_23_X
                    fc_24_exch += term_24_X                     
                else:
                    print('No ov is not implemented for BBBA_2')
                    #
                    #term_32 = value * CA_EA_A[0, ind_A[0]]
                    #term_42 = value * AC_IP_A[ind_A[0], 0] 
                    #term_32_X, term_32_C, term_42_X, term_42_C = calc_term_BBBA_2(ind_B, term_32, term_42, ACA_B, CA_EA_B, AC_IP_B)
#
                    #fc_32_coul += term_32_C
                    #fc_42_coul += term_42_C
                    #fc_32_exch += term_32_X
    return fc_32_coul, fc_32_exch, fc_42_coul, fc_42_exch, fc_23_coul, fc_23_exch, fc_24_coul, fc_24_exch, fc_31_C_CS_EA_SC, fc_31_X_CS_EA_SC,\
        fc_14_C_SC_IP_CS, fc_14_X_SC_IP_CS, fc_32_C_SC_EA_CS, fc_32_X_SC_EA_CS, \
            fc_24_C_2S_IP, fc_24_X_2S_IP, fc_21_C_SC_occ, fc_21_X_SC_occ, fc_12_C_CS_virt, fc_12_X_CS_virt, \
            fc_34_C_CS_EA, fc_34_X_CS_EA, fc_34_C_SC_IP, fc_34_X_SC_IP, fc_43_C_1S_EA_arb, fc_43_X_1S_EA_arb, fc_43_C_1S_IP_arb, fc_43_X_1S_IP_arb, fc_21_C_CS_virt_arb, fc_21_X_CS_virt_arb, \
            fc_coul_31_2S_occ_virt_arb, fc_exch_31_2S_occ_virt_arb, fc_coul_13_2S_EA_arb_occ_arb, fc_exch_13_2S_EA_arb_occ_arb, fc_coul_14_2S_IP_occ_arb, fc_exch_14_2S_IP_occ_arb, fc_coul_21_3S_A_occ_B_virt_A_virt_arb, fc_exch_21_3S_A_occ_B_virt_A_virt_arb, \
            fc_coul_31_2S_EA_virt_arb, fc_exch_31_2S_EA_virt_arb, fc_coul_13_4S_IP_virt_occ_arb_EA_arb, fc_exch_13_4S_IP_virt_occ_arb_EA_arb, fc_coul_14_2S_virt_occ_arb, fc_exch_14_2S_virt_occ_arb, \
            fc_coul_43_3S_IP_IP_arb_EA_arb, fc_exch_43_3S_IP_IP_arb_EA_arb, fc_coul_31_4S_occ_EA_virt_arb_IP_arb, fc_exch_31_4S_occ_EA_virt_arb_IP_arb, \
            fc_coul_13_2S_virt_EA_arb, fc_exch_13_2S_virt_EA_arb, \
            fc_coul_41_2S_occ_virt_arb, fc_exch_41_2S_occ_virt_arb, fc_coul_21_3S_A_occ_A_virt_arb_B_occ_arb, fc_exch_21_3S_A_occ_A_virt_arb_B_occ_arb, \
            fc_coul_13_2S_virt_occ_arb, fc_exch_13_2S_virt_occ_arb, fc_coul_41_2S_occ_IP_arb, fc_exch_41_2S_occ_IP_arb, fc_coul_14_4S_IP_virt_occ_arb_EA_arb, fc_exch_14_4S_IP_virt_occ_arb_EA_arb, \
            fc_coul_41_4S_occ_EA_virt_arb_IP_arb, fc_exch_41_4S_occ_EA_virt_arb_IP_arb, fc_coul_43_3S_EA_EA_arb_IP_arb, fc_exch_43_3S_EA_EA_arb_IP_arb


