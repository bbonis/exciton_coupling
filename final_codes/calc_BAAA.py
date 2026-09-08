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



def calc_term_BAAA_4(
    indexes,
    value,
    SC_B_AB_traf        ,
    SC_B_AB_traf_pa     ,
    CS_B_BA_traf        ,
    SC_A_BA_traf        ,
    SC_A_BA_traf_pa     ,
    CS_A_AB_traf        ,
    CS_A_AB_traf_iq        ,
    term_31             ,
    term_41             ,
    term_13             ,
    term_14             ,
    term_14_2S_IP_EA_arb,
    term_31_2S_EA_IP_arb,
    term_43_EA          ,
    term_43_IP          ,
    SC_IP_B_i           ,
    SC_IP_B_p           ,
    CS_EA_B_a           ,
    CS_EA_B_q           ,
    ACA_A_right         ,
    SCS_B_AA_pa         ,
    ACA_A_left          ,
    CA_EA_A_right       ,
    AC_IP_A_right       ,
    CA_EA_A_left        ,
    AC_IP_A_left        ,
):
    term_31_X = np.float64(0.0)
    term_31_C = np.float64(0.0)
    term_41_X = np.float64(0.0)
    term_41_C = np.float64(0.0)

    term_13_X = np.float64(0.0)
    term_13_C = np.float64(0.0)
    term_14_X = np.float64(0.0)
    term_14_C = np.float64(0.0)

    term_41_X_2S = np.float64(0.0)
    term_41_C_2S = np.float64(0.0)
    term_23_X_2S = np.float64(0.0)
    term_23_C_2S = np.float64(0.0)
    term_13_X_2S = np.float64(0.0)
    term_13_C_2S = np.float64(0.0)
    term_42_X = np.float64(0.0)
    term_42_C = np.float64(0.0)
    term_21_X_CS_virt = np.float64(0.0)
    term_21_C_CS_virt = np.float64(0.0)
    term_12_X_SC_occ = np.float64(0.0)
    term_12_C_SC_occ = np.float64(0.0)
    term_21_X_SC_occ_arb = np.float64(0.0)
    term_21_C_SC_occ_arb = np.float64(0.0)
    term_43_X_CS_EA = np.float64(0.0)
    term_43_C_CS_EA = np.float64(0.0)
    term_43_X_SC_IP = np.float64(0.0)
    term_43_C_SC_IP = np.float64(0.0)

    term_31_C_2S_EA_IP_arb = np.float64(0.0)
    term_31_X_2S_EA_IP_arb = np.float64(0.0)
    term_31_C_2S_occ_IP_arb = np.float64(0.0)
    term_31_X_2S_occ_IP_arb = np.float64(0.0)
    term_14_C_2S_IP_EA_arb = np.float64(0.0) 
    term_14_X_2S_IP_EA_arb = np.float64(0.0)
    term_21_C_3S_A_occ_B_virt_B_occ_arb = np.float64(0.0)
    term_21_X_3S_A_occ_B_virt_B_occ_arb = np.float64(0.0)
    term_21_C_3S_B_virt_B_occ_arb_A_virt_arb = np.float64(0.0)
    term_21_X_3S_B_virt_B_occ_arb_A_virt_arb = np.float64(0.0)
    term_14_C_2S_occ_arb_EA_arb = np.float64(0.0)
    term_14_X_2S_occ_arb_EA_arb = np.float64(0.0)

    term_14_C_2S_virt_EA_arb = np.float64(0.0)
    term_14_X_2S_virt_EA_arb = np.float64(0.0)

    i0, i1, i2 = indexes

    p5 = (i0, i1, i2)
    p3 = (i0, i2, i1)
    p4 = (i1, i0, i2)
    p1 = (i1, i2, i0)
    p2 = (i2, i0, i1)
    p0 = (i2, i1, i0)

    perms = [p0, p1, p2, p3, p4, p5]

    C_prefactor = np.float64(4.0)
    X_prefactor = np.float64(-2.0)


    for idx in [2, 4]:
        term_14_X_2S_virt_EA_arb += X_prefactor * AC_IP_A_right[perms[idx][0], :] * CS_EA_B_q[:, perms[idx][1]] * CS_A_AB_traf[perms[idx][2]] * value
        term_41_X += X_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * AC_IP_A_left[perms[idx][2], 0] * term_41
        term_31_X_2S_EA_IP_arb += X_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * SC_IP_B_p[perms[idx][2], :] * term_31_2S_EA_IP_arb
        term_14_X += X_prefactor * ACA_A_left[perms[idx][0], perms[idx][1]] * AC_IP_A_right[perms[idx][2], 0] * term_14
        term_43_X_SC_IP += X_prefactor * SC_IP_B_i[perms[idx][0], 0] * AC_IP_A_left[perms[idx][2], 0] * CA_EA_A_right[0, perms[idx][1]] * term_43_EA
        term_21_X_3S_B_virt_B_occ_arb_A_virt_arb += X_prefactor * SCS_B_AA_pa[perms[idx][0], perms[idx][1]] * CS_A_AB_traf_iq[perms[idx][2]] * value
        term_13_X_2S += X_prefactor * CS_A_AB_traf[perms[idx][0]] * CA_EA_A_right[0, perms[idx][1]] * SC_IP_B_i[perms[idx][2], 0] * value
        
        term_42_X += X_prefactor * SC_B_AB_traf[perms[idx][0]] * CS_EA_B_a[0, perms[idx][1]] * AC_IP_A_left[perms[idx][2], 0] * value
        term_12_X_SC_occ += X_prefactor * ACA_A_left[perms[idx][0], perms[idx][1]] * SC_B_AB_traf[perms[idx][2]] * value
        term_21_X_SC_occ_arb += X_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * SC_B_AB_traf_pa[perms[idx][2]] * value
        
   
    for idx in [0, 1]:
        term_14_C_2S_virt_EA_arb += C_prefactor * AC_IP_A_right[perms[idx][0], :] * CS_EA_B_q[:, perms[idx][1]] * CS_A_AB_traf[perms[idx][2]] * value
        term_41_C += C_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * AC_IP_A_left[perms[idx][2], 0] * term_41
        term_31_C_2S_EA_IP_arb += C_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * SC_IP_B_p[perms[idx][2], :] * term_31_2S_EA_IP_arb
        term_14_C += C_prefactor * ACA_A_left[perms[idx][0], perms[idx][1]] * AC_IP_A_right[perms[idx][2], 0] * term_14
        term_12_C_SC_occ += C_prefactor * ACA_A_left[perms[idx][0], perms[idx][1]] * SC_B_AB_traf[perms[idx][2]] * value
        term_21_C_SC_occ_arb += C_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * SC_B_AB_traf_pa[perms[idx][2]] * value
        term_31_X += X_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * CA_EA_A_left[0, perms[idx][2]] * term_31
        term_13_X += X_prefactor * ACA_A_left[perms[idx][0], perms[idx][1]] * CA_EA_A_right[0, perms[idx][2]] * term_13
        term_21_C_3S_B_virt_B_occ_arb_A_virt_arb += C_prefactor * SCS_B_AA_pa[perms[idx][0], perms[idx][1]] * CS_A_AB_traf_iq[perms[idx][2]] * value
        term_41_X_2S += X_prefactor * AC_IP_A_left[perms[idx][2], 0] * SC_A_BA_traf[perms[idx][0]] * CS_EA_B_a[0, perms[idx][1]] * value
        term_23_X_2S += X_prefactor * SC_IP_B_i[perms[idx][2], 0] * CS_B_BA_traf[perms[idx][1]] * CA_EA_A_right[0, perms[idx][0]] * value
        term_43_C_SC_IP += C_prefactor * SC_IP_B_i[perms[idx][0], 0] * AC_IP_A_left[perms[idx][2], 0] * CA_EA_A_right[0, perms[idx][1]] * term_43_EA
        term_43_C_CS_EA += C_prefactor * CA_EA_A_right[0, perms[idx][0]] * CS_EA_B_a[0, perms[idx][1]] * AC_IP_A_left[perms[idx][2], 0] * term_13
        term_21_C_CS_virt += C_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * CS_B_BA_traf[perms[idx][2]] * value
        term_14_X_2S_occ_arb_EA_arb += X_prefactor * AC_IP_A_right[perms[idx][0], :] * CS_EA_B_q[:, perms[idx][1]] * SC_A_BA_traf_pa[perms[idx][2]] * value
        term_21_C_3S_A_occ_B_virt_B_occ_arb += C_prefactor * SC_A_BA_traf[perms[idx][0]] * SCS_B_AA_pa[perms[idx][2], perms[idx][1]] * value
        term_14_C_2S_IP_EA_arb += C_prefactor * CS_EA_B_q[:, perms[idx][0]] * ACA_A_left[perms[idx][2], perms[idx][1]] * term_14_2S_IP_EA_arb
        term_31_C_2S_occ_IP_arb += C_prefactor * SC_A_BA_traf[perms[idx][0]] * CA_EA_A_left[:, perms[idx][1]] * SC_IP_B_p[perms[idx][2], :] * value
    for idx in [3, 5]:
        term_31_C += C_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * CA_EA_A_left[0, perms[idx][2]] * term_31
        term_13_C += C_prefactor * ACA_A_left[perms[idx][0], perms[idx][1]] * CA_EA_A_right[0, perms[idx][2]] * term_13
        term_41_C_2S += C_prefactor * SC_A_BA_traf[perms[idx][0]] * CS_EA_B_a[0, perms[idx][1]] * AC_IP_A_left[perms[idx][2], 0] * value
        
        term_14_C_2S_occ_arb_EA_arb += C_prefactor * AC_IP_A_right[perms[idx][0], :] * CS_EA_B_q[:, perms[idx][1]] * SC_A_BA_traf_pa[perms[idx][2]] * value
        term_13_C_2S += C_prefactor * CS_A_AB_traf[perms[idx][0]] * CA_EA_A_right[0, perms[idx][1]] * SC_IP_B_i[perms[idx][2], 0] * value
        
        term_42_C += C_prefactor * SC_B_AB_traf[perms[idx][0]] * CS_EA_B_a[0, perms[idx][1]] * AC_IP_A_left[perms[idx][2], 0] * value
        term_43_X_CS_EA += X_prefactor * CA_EA_A_right[0, perms[idx][0]] * CS_EA_B_a[0, perms[idx][1]] * AC_IP_A_left[perms[idx][2], 0] * term_13
        term_23_C_2S += C_prefactor * CS_B_BA_traf[perms[idx][0]] * SC_IP_B_i[perms[idx][1], 0] * CA_EA_A_right[0, perms[idx][2]] * value
        term_21_X_3S_A_occ_B_virt_B_occ_arb += X_prefactor * SC_A_BA_traf[perms[idx][0]] * SCS_B_AA_pa[perms[idx][2], perms[idx][1]] * value
        term_14_X_2S_IP_EA_arb += X_prefactor * CS_EA_B_q[:, perms[idx][0]] * ACA_A_left[perms[idx][2], perms[idx][1]] * term_14_2S_IP_EA_arb
        term_31_X_2S_occ_IP_arb += X_prefactor * SC_A_BA_traf[perms[idx][0]] * CA_EA_A_left[:, perms[idx][1]] * SC_IP_B_p[perms[idx][2], :] * value
        term_21_X_CS_virt += X_prefactor * ACA_A_right[perms[idx][0], perms[idx][1]] * CS_B_BA_traf[perms[idx][2]] * value
    return (
        term_31_X,
        term_31_C,
        term_41_X,
        term_41_C,
        term_13_X,
        term_13_C,
        term_14_X,
        term_14_C,
        term_21_X_CS_virt,
        term_21_C_CS_virt,
        term_12_X_SC_occ,
        term_12_C_SC_occ,
        term_43_X_CS_EA,
        term_43_C_CS_EA,
        term_43_X_SC_IP,
        term_43_C_SC_IP,
        term_42_X,
        term_42_C,
        term_41_X_2S,
        term_41_C_2S,
        term_23_X_2S,
        term_23_C_2S,
        term_13_X_2S,
        term_13_C_2S,
        term_21_X_SC_occ_arb,
        term_21_C_SC_occ_arb,
        term_31_C_2S_EA_IP_arb, 
        term_31_X_2S_EA_IP_arb, 
        term_31_C_2S_occ_IP_arb, 
        term_31_X_2S_occ_IP_arb, 
        term_14_C_2S_IP_EA_arb, 
        term_14_X_2S_IP_EA_arb, 
        term_21_C_3S_A_occ_B_virt_B_occ_arb,
        term_21_X_3S_A_occ_B_virt_B_occ_arb,
        term_14_C_2S_occ_arb_EA_arb,
        term_14_X_2S_occ_arb_EA_arb,
        term_21_C_3S_B_virt_B_occ_arb_A_virt_arb, 
        term_21_X_3S_B_virt_B_occ_arb_A_virt_arb,
        term_14_C_2S_virt_EA_arb,
        term_14_X_2S_virt_EA_arb
    )



def calc_term_BAAA_3(indexes, is_type_1, value         ,
                            SC_B_AB_traf        ,
                            SC_B_AB_traf_pa     ,
                            CS_B_BA_traf        ,
                            SC_A_BA_traf        ,
                            SC_A_BA_traf_pa     ,
                            CS_A_AB_traf        ,
                            CS_A_AB_traf_iq     ,
                            term_31             ,
                            term_41             ,
                            term_13             ,
                            term_14             ,
                            term_14_2S_IP_EA_arb,
                            term_31_2S_EA_IP_arb,
                            term_43_EA          ,
                            term_43_IP          ,
                            SC_IP_B_i           ,
                            SC_IP_B_p           ,
                            CS_EA_B_a           ,
                            CS_EA_B_q           ,
                            ACA_A_right         ,
                            SCS_B_AA_pa         ,
                            ACA_A_left          ,
                            CA_EA_A_right       ,
                            AC_IP_A_right       ,
                            CA_EA_A_left        ,
                            AC_IP_A_left        ,):# twoel_AO_AAAB_canonical, BBBBB,
                            
    term_31_X = np.float64(0.)
    term_31_C = np.float64(0.)
    term_41_X = np.float64(0.)
    term_41_C = np.float64(0.)
    
    term_13_X = np.float64(0.)
    term_13_C = np.float64(0.)
    term_14_X = np.float64(0.)
    term_14_C = np.float64(0.)    
    
    term_41_X_2S = np.float64(0.)
    term_41_C_2S = np.float64(0.)
    term_23_X_2S = np.float64(0.)
    term_23_C_2S = np.float64(0.)
    term_13_X_2S = np.float64(0.)
    term_13_C_2S = np.float64(0.)
    term_42_X = np.float64(0.)
    term_42_C = np.float64(0.)   
    term_21_X_CS_virt = np.float64(0.)
    term_21_C_CS_virt = np.float64(0.)
    term_12_X_SC_occ  = np.float64(0.)
    term_12_C_SC_occ  = np.float64(0.)      
    term_43_X_CS_EA   = np.float64(0.)
    term_43_C_CS_EA   = np.float64(0.)    
    term_43_X_SC_IP   = np.float64(0.)
    term_43_C_SC_IP   = np.float64(0.)   
    #perms = list(distinct_permutations(indexes))
    term_21_X_SC_occ_arb = np.float64(0.0)
    term_21_C_SC_occ_arb = np.float64(0.0)

    term_31_C_2S_EA_IP_arb = np.float64(0.0)
    term_31_X_2S_EA_IP_arb = np.float64(0.0)
    term_31_C_2S_occ_IP_arb = np.float64(0.0)
    term_31_X_2S_occ_IP_arb = np.float64(0.0)
    term_14_C_2S_IP_EA_arb = np.float64(0.0) 
    term_14_X_2S_IP_EA_arb = np.float64(0.0)
    term_21_C_3S_A_occ_B_virt_B_occ_arb = np.float64(0.0)
    term_21_X_3S_A_occ_B_virt_B_occ_arb = np.float64(0.0)
    term_14_C_2S_occ_arb_EA_arb = np.float64(0.0)
    term_14_X_2S_occ_arb_EA_arb = np.float64(0.0)
    term_21_C_3S_B_virt_B_occ_arb_A_virt_arb = np.float64(0.0)
    term_21_X_3S_B_virt_B_occ_arb_A_virt_arb = np.float64(0.0)

    term_14_C_2S_virt_EA_arb = np.float64(0.0)
    term_14_X_2S_virt_EA_arb = np.float64(0.0)

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

    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)

    if is_type_1:#type_1 = 6010
        
        term_14_C_2S_virt_EA_arb += C_prefactor * AC_IP_A_right[perms[1][0], :] * CS_EA_B_q[:, perms[1][1]] * CS_A_AB_traf[perms[1][2]] * value
        term_14_X_2S_virt_EA_arb += X_prefactor * AC_IP_A_right[perms[0][0], :] * CS_EA_B_q[:, perms[0][1]] * CS_A_AB_traf[perms[0][2]] * value
        term_14_C_2S_virt_EA_arb += C_prefactor * AC_IP_A_right[perms[0][0], :] * CS_EA_B_q[:, perms[0][1]] * CS_A_AB_traf[perms[0][2]] * value
        term_14_X_2S_virt_EA_arb += X_prefactor * AC_IP_A_right[perms[2][0], :] * CS_EA_B_q[:, perms[2][1]] * CS_A_AB_traf[perms[2][2]] * value

        term_12_X_SC_occ += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * SC_B_AB_traf[perms[0][2]] * value
        term_12_C_SC_occ += C_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * SC_B_AB_traf[perms[0][2]] * value
        term_12_X_SC_occ += X_prefactor * ACA_A_left[perms[2][0], perms[2][1]] * SC_B_AB_traf[perms[2][2]] * value                                
        term_12_C_SC_occ += C_prefactor * ACA_A_left[perms[1][0], perms[1][1]] * SC_B_AB_traf[perms[1][2]] * value
        
        term_21_X_SC_occ_arb += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * SC_B_AB_traf_pa[perms[0][2]] * value
        term_21_C_SC_occ_arb += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * SC_B_AB_traf_pa[perms[0][2]] * value
        term_21_X_SC_occ_arb += X_prefactor * ACA_A_right[perms[2][0], perms[2][1]] * SC_B_AB_traf_pa[perms[2][2]] * value                                
        term_21_C_SC_occ_arb += C_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * SC_B_AB_traf_pa[perms[1][2]] * value
        term_31_X_2S_EA_IP_arb += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * SC_IP_B_p[perms[0][2], :] * term_31_2S_EA_IP_arb
        term_31_C_2S_EA_IP_arb += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * SC_IP_B_p[perms[0][2], :] * term_31_2S_EA_IP_arb
        term_31_X_2S_EA_IP_arb += X_prefactor * ACA_A_right[perms[2][0], perms[2][1]] * SC_IP_B_p[perms[2][2], :] * term_31_2S_EA_IP_arb
        term_31_C_2S_EA_IP_arb += C_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * SC_IP_B_p[perms[1][2], :] * term_31_2S_EA_IP_arb
        term_14_X += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * AC_IP_A_right[perms[0][2], 0] * term_14
        term_14_C += C_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * AC_IP_A_right[perms[0][2], 0] * term_14
        term_14_X += X_prefactor * ACA_A_left[perms[2][0], perms[2][1]] * AC_IP_A_right[perms[2][2], 0] * term_14                                
        term_14_C += C_prefactor * ACA_A_left[perms[1][0], perms[1][1]] * AC_IP_A_right[perms[1][2], 0] * term_14
        
        term_21_X_3S_B_virt_B_occ_arb_A_virt_arb += X_prefactor * SCS_B_AA_pa[perms[0][0], perms[0][1]] * CS_A_AB_traf_iq[perms[0][2]] * value
        term_21_C_3S_B_virt_B_occ_arb_A_virt_arb += C_prefactor * SCS_B_AA_pa[perms[0][0], perms[0][1]] * CS_A_AB_traf_iq[perms[0][2]] * value
        term_21_X_3S_B_virt_B_occ_arb_A_virt_arb += X_prefactor * SCS_B_AA_pa[perms[2][0], perms[2][1]] * CS_A_AB_traf_iq[perms[2][2]] * value
        term_21_C_3S_B_virt_B_occ_arb_A_virt_arb += C_prefactor * SCS_B_AA_pa[perms[1][0], perms[1][1]] * CS_A_AB_traf_iq[perms[1][2]] * value
        
        term_13_X_2S += X_prefactor * SC_IP_B_i[perms[0][2], 0] * CS_A_AB_traf[perms[0][0]] * CA_EA_A_right[0, perms[0][1]] * value            
        term_13_C_2S += C_prefactor * SC_IP_B_i[perms[2][2], 0] * CS_A_AB_traf[perms[2][0]] * CA_EA_A_right[0, perms[2][1]] * value              
        term_13_X_2S += X_prefactor * SC_IP_B_i[perms[2][2], 0] * CS_A_AB_traf[perms[2][0]] * CA_EA_A_right[0, perms[2][1]] * value
        term_13_C_2S += C_prefactor * SC_IP_B_i[perms[1][2], 0] * CS_A_AB_traf[perms[1][0]] * CA_EA_A_right[0, perms[1][1]] * value             
        
        term_23_X_2S += X_prefactor * SC_IP_B_i[perms[2][1], 0] * CS_B_BA_traf[perms[2][0]] * CA_EA_A_right[0, perms[2][2]] * value
        term_23_C_2S += C_prefactor * SC_IP_B_i[perms[2][1], 0] * CS_B_BA_traf[perms[2][0]] * CA_EA_A_right[0, perms[2][2]] * value
        term_23_X_2S += X_prefactor * SC_IP_B_i[perms[0][1], 0] * CS_B_BA_traf[perms[0][0]] * CA_EA_A_right[0, perms[0][2]] * value
        term_23_C_2S += C_prefactor * SC_IP_B_i[perms[1][1], 0] * CS_B_BA_traf[perms[1][0]] * CA_EA_A_right[0, perms[1][2]] * value
        term_42_X += X_prefactor * SC_B_AB_traf[perms[0][0]] * CS_EA_B_a[0, perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * value
        term_42_C += C_prefactor * SC_B_AB_traf[perms[2][0]] * CS_EA_B_a[0, perms[2][1]] * AC_IP_A_left[perms[2][2], 0] * value
        term_42_X += X_prefactor * SC_B_AB_traf[perms[2][0]] * CS_EA_B_a[0, perms[2][1]] * AC_IP_A_left[perms[2][2], 0] * value                                
        term_42_C += C_prefactor * SC_B_AB_traf[perms[1][0]] * CS_EA_B_a[0, perms[1][1]] * AC_IP_A_left[perms[1][2], 0] * value
        
        
        term_21_X_CS_virt += X_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * CS_B_BA_traf[perms[1][2]] * value             
        term_21_C_CS_virt += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value            
        term_21_X_CS_virt += X_prefactor * ACA_A_right[perms[2][0], perms[2][1]] * CS_B_BA_traf[perms[2][2]] * value              
        term_21_C_CS_virt += C_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * CS_B_BA_traf[perms[1][2]] * value
        
        term_31_X += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * CA_EA_A_left[0, perms[0][2]] * term_31            
        term_31_C += C_prefactor * ACA_A_right[perms[2][0], perms[2][1]] * CA_EA_A_left[0, perms[2][2]] * term_31              
        term_31_X += X_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * CA_EA_A_left[0, perms[1][2]] * term_31
        term_31_C += C_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * CA_EA_A_left[0, perms[1][2]] * term_31 
        term_14_X_2S_occ_arb_EA_arb += X_prefactor * AC_IP_A_right[perms[0][0], :] * CS_EA_B_q[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value
        term_14_C_2S_occ_arb_EA_arb += C_prefactor * AC_IP_A_right[perms[2][0], :] * CS_EA_B_q[:, perms[2][1]] * SC_A_BA_traf_pa[perms[2][2]] * value
        term_14_X_2S_occ_arb_EA_arb += X_prefactor * AC_IP_A_right[perms[1][0], :] * CS_EA_B_q[:, perms[1][1]] * SC_A_BA_traf_pa[perms[1][2]] * value
        term_14_C_2S_occ_arb_EA_arb += C_prefactor * AC_IP_A_right[perms[1][0], :] * CS_EA_B_q[:, perms[1][1]] * SC_A_BA_traf_pa[perms[1][2]] * value
        term_13_X += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * CA_EA_A_right[0, perms[0][2]] * term_13            
        term_13_C += C_prefactor * ACA_A_left[perms[2][0], perms[2][1]] * CA_EA_A_right[0, perms[2][2]] * term_13              
        term_13_X += X_prefactor * ACA_A_left[perms[1][0], perms[1][1]] * CA_EA_A_right[0, perms[1][2]] * term_13
        term_13_C += C_prefactor * ACA_A_left[perms[1][0], perms[1][1]] * CA_EA_A_right[0, perms[1][2]] * term_13 
        
        term_41_X_2S += X_prefactor * AC_IP_A_left[perms[0][1], 0] * SC_A_BA_traf[perms[0][2]] * CS_EA_B_a[0, perms[0][0]] * value
        term_41_C_2S += C_prefactor * AC_IP_A_left[perms[2][1], 0] * SC_A_BA_traf[perms[2][0]] * CS_EA_B_a[0, perms[2][2]] * value
        term_41_X_2S += X_prefactor * AC_IP_A_left[perms[1][0], 0] * SC_A_BA_traf[perms[1][1]] * CS_EA_B_a[0, perms[1][2]] * value
        term_41_C_2S += C_prefactor * AC_IP_A_left[perms[1][1], 0] * SC_A_BA_traf[perms[1][0]] * CS_EA_B_a[0, perms[1][2]] * value
        term_41_C += C_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * AC_IP_A_left[perms[1][2], 0] * term_41
        term_41_X += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * term_41
        term_41_C += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * term_41
        term_41_X += X_prefactor * ACA_A_right[perms[2][0], perms[2][1]] * AC_IP_A_left[perms[2][2], 0] * term_41                                
        
        term_21_X_3S_A_occ_B_virt_B_occ_arb += X_prefactor * SCS_B_AA_pa[perms[0][0], perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value
        term_21_X_3S_A_occ_B_virt_B_occ_arb += X_prefactor * SCS_B_AA_pa[perms[1][0], perms[1][1]] * SC_A_BA_traf[perms[1][2]] * value
        term_21_C_3S_A_occ_B_virt_B_occ_arb += C_prefactor * SCS_B_AA_pa[perms[2][0], perms[2][1]] * SC_A_BA_traf[perms[2][2]] * value
        term_21_C_3S_A_occ_B_virt_B_occ_arb += C_prefactor * SCS_B_AA_pa[perms[1][0], perms[1][1]] * SC_A_BA_traf[perms[1][2]] * value
        term_14_X_2S_IP_EA_arb += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * CS_EA_B_q[:, perms[0][2]] * term_14_2S_IP_EA_arb
        term_14_C_2S_IP_EA_arb += C_prefactor * ACA_A_left[perms[2][0], perms[2][1]] * CS_EA_B_q[:, perms[2][2]] * term_14_2S_IP_EA_arb
        term_14_X_2S_IP_EA_arb += X_prefactor * ACA_A_left[perms[1][0], perms[1][1]] * CS_EA_B_q[:, perms[1][2]] * term_14_2S_IP_EA_arb
        term_14_C_2S_IP_EA_arb += C_prefactor * ACA_A_left[perms[1][0], perms[1][1]] * CS_EA_B_q[:, perms[1][2]] * term_14_2S_IP_EA_arb
        term_31_X_2S_occ_IP_arb += X_prefactor * SC_IP_B_p[perms[0][0], :] * CA_EA_A_left[:, perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value
        term_31_C_2S_occ_IP_arb += C_prefactor * SC_IP_B_p[perms[2][0], :] * CA_EA_A_left[:, perms[2][1]] * SC_A_BA_traf[perms[2][2]] * value
        term_31_X_2S_occ_IP_arb += X_prefactor * SC_IP_B_p[perms[1][0], :] * CA_EA_A_left[:, perms[1][1]] * SC_A_BA_traf[perms[1][2]] * value
        term_31_C_2S_occ_IP_arb += C_prefactor * SC_IP_B_p[perms[1][0], :] * CA_EA_A_left[:, perms[1][1]] * SC_A_BA_traf[perms[1][2]] * value
        term_43_X_SC_IP += X_prefactor * CA_EA_A_right[0, perms[0][1]] * SC_IP_B_i[perms[0][0], 0] * AC_IP_A_left[perms[0][2], 0] * term_43_EA
        term_43_C_SC_IP += C_prefactor * CA_EA_A_right[0, perms[0][1]] * SC_IP_B_i[perms[0][0], 0] * AC_IP_A_left[perms[0][2], 0] * term_43_EA
        term_43_X_SC_IP += X_prefactor * CA_EA_A_right[0, perms[2][1]] * SC_IP_B_i[perms[2][0], 0] * AC_IP_A_left[perms[2][2], 0] * term_43_EA                                
        term_43_C_SC_IP += C_prefactor * CA_EA_A_right[0, perms[1][1]] * SC_IP_B_i[perms[1][0], 0] * AC_IP_A_left[perms[1][2], 0] * term_43_EA
        
        # idx = 1, C
        term_43_X_CS_EA += X_prefactor * CA_EA_A_right[0, perms[1][0]] * CS_EA_B_a[0, perms[1][1]] * AC_IP_A_left[perms[1][2], 0] * term_13
        term_43_C_CS_EA += C_prefactor * CA_EA_A_right[0, perms[1][0]] * CS_EA_B_a[0, perms[1][1]] * AC_IP_A_left[perms[1][2], 0] * term_13
        term_43_X_CS_EA += X_prefactor * CA_EA_A_right[0, perms[2][0]] * CS_EA_B_a[0, perms[2][1]] * AC_IP_A_left[perms[2][2], 0] * term_13
        term_43_C_CS_EA += C_prefactor * CA_EA_A_right[0, perms[0][0]] * CS_EA_B_a[0, perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * term_13
    else:
        
        term_14_C_2S_virt_EA_arb += C_prefactor * AC_IP_A_right[perms[2][1], :] * CS_EA_B_q[:, perms[2][2]] * CS_A_AB_traf[perms[2][0]] * value
        term_14_X_2S_virt_EA_arb += X_prefactor * AC_IP_A_right[perms[1][0], :] * CS_EA_B_q[:, perms[1][1]] * CS_A_AB_traf[perms[1][2]] * value

        term_12_X_SC_occ += X_prefactor * ACA_A_left[perms[1][0], perms[1][1]] * SC_B_AB_traf[perms[1][2]] * value
        term_12_C_SC_occ += C_prefactor * ACA_A_left[perms[2][1], perms[2][2]] * SC_B_AB_traf[perms[2][0]] * value   
        term_21_X_SC_occ_arb += X_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * SC_B_AB_traf_pa[perms[1][2]] * value
        term_21_C_SC_occ_arb += C_prefactor * ACA_A_right[perms[2][1], perms[2][2]] * SC_B_AB_traf_pa[perms[2][0]] * value   
        term_31_C += C_prefactor * ACA_A_right[perms[2][0], perms[2][1]] * CA_EA_A_left[0, perms[2][2]] * term_31
        term_31_X += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * CA_EA_A_left[0, perms[0][2]] * term_31
        term_14_C_2S_occ_arb_EA_arb += C_prefactor * AC_IP_A_right[perms[2][0], :] * CS_EA_B_q[:, perms[2][1]] * SC_A_BA_traf_pa[perms[2][2]] * value
        term_14_X_2S_occ_arb_EA_arb += X_prefactor * AC_IP_A_right[perms[0][0], :] * CS_EA_B_q[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value
        term_41_C_2S += C_prefactor * AC_IP_A_left[perms[2][2], 0] * SC_A_BA_traf[perms[2][0]] * CS_EA_B_a[0, perms[2][1]] * value
        term_41_X_2S += X_prefactor * AC_IP_A_left[perms[0][2], 0] * SC_A_BA_traf[perms[0][0]] * CS_EA_B_a[0, perms[0][1]] * value
        term_41_C += C_prefactor * ACA_A_right[perms[2][1], perms[2][2]] * AC_IP_A_left[perms[2][0], 0] * term_41    
        term_41_X += X_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * AC_IP_A_left[perms[1][2], 0] * term_41
        term_21_C_CS_virt += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value
        term_21_X_CS_virt += X_prefactor * ACA_A_right[perms[2][0], perms[2][1]] * CS_B_BA_traf[perms[2][2]] * value
        term_13_C += C_prefactor * ACA_A_left[perms[2][0], perms[2][1]] * CA_EA_A_right[0, perms[2][2]] * term_13
        term_13_X += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * CA_EA_A_right[0, perms[0][2]] * term_13
        
        term_21_C_3S_A_occ_B_virt_B_occ_arb += C_prefactor * SCS_B_AA_pa[perms[2][0], perms[2][1]] * SC_A_BA_traf[perms[2][2]] * value
        term_21_X_3S_A_occ_B_virt_B_occ_arb += X_prefactor * SCS_B_AA_pa[perms[0][0], perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value
        term_14_C_2S_IP_EA_arb += C_prefactor * ACA_A_left[perms[2][0], perms[2][1]] * CS_EA_B_q[:, perms[2][2]] * term_14_2S_IP_EA_arb
        term_14_X_2S_IP_EA_arb += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * CS_EA_B_q[:, perms[0][2]] * term_14_2S_IP_EA_arb
        term_31_C_2S_occ_IP_arb += C_prefactor * SC_IP_B_p[perms[2][0], :] * CA_EA_A_left[:, perms[2][1]] * SC_A_BA_traf[perms[2][2]] * value
        term_31_X_2S_occ_IP_arb += X_prefactor * SC_IP_B_p[perms[0][0], :] * CA_EA_A_left[:, perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value
        term_23_C_2S += C_prefactor * SC_IP_B_i[perms[2][1], 0] * CS_B_BA_traf[perms[2][0]] * CA_EA_A_right[0, perms[2][2]] * value
        term_23_X_2S += X_prefactor * SC_IP_B_i[perms[1][1], 0] * CS_B_BA_traf[perms[1][0]] * CA_EA_A_right[0, perms[1][2]] * value
        term_31_X_2S_EA_IP_arb += X_prefactor * ACA_A_right[perms[1][0], perms[1][1]] * SC_IP_B_p[perms[1][2], :] * term_31_2S_EA_IP_arb  
        term_31_C_2S_EA_IP_arb += C_prefactor * ACA_A_right[perms[2][1], perms[2][2]] * SC_IP_B_p[perms[2][0], :] * term_31_2S_EA_IP_arb  
        term_14_C += C_prefactor * ACA_A_left[perms[2][1], perms[2][2]] * AC_IP_A_right[perms[2][0], 0] * term_14  
        term_14_X += X_prefactor * ACA_A_left[perms[1][0], perms[1][1]] * AC_IP_A_right[perms[1][2], 0] * term_14
        
        term_21_C_3S_B_virt_B_occ_arb_A_virt_arb += C_prefactor * SCS_B_AA_pa[perms[2][1], perms[2][2]] * CS_A_AB_traf_iq[perms[2][0]] * value
        term_21_X_3S_B_virt_B_occ_arb_A_virt_arb += X_prefactor * SCS_B_AA_pa[perms[1][0], perms[1][1]] * CS_A_AB_traf_iq[perms[1][2]] * value
        term_13_X_2S += X_prefactor * SC_IP_B_i[perms[1][2], 0] * CS_A_AB_traf[perms[1][0]] * CA_EA_A_right[0, perms[1][1]] * value
        term_13_C_2S += C_prefactor * SC_IP_B_i[perms[2][2], 0] * CS_A_AB_traf[perms[2][0]] * CA_EA_A_right[0, perms[2][1]] * value             
        
        term_42_X += X_prefactor * SC_B_AB_traf[perms[1][0]] * CS_EA_B_a[0, perms[1][1]] * AC_IP_A_left[perms[1][2], 0] * value
        term_42_C += C_prefactor * SC_B_AB_traf[perms[2][0]] * CS_EA_B_a[0, perms[2][1]] * AC_IP_A_left[perms[2][2], 0] * value  
        term_43_X_SC_IP += X_prefactor * CA_EA_A_right[0, perms[1][1]] * SC_IP_B_i[perms[1][0], 0] * AC_IP_A_left[perms[1][2], 0] * term_43_EA
        term_43_C_SC_IP += C_prefactor * CA_EA_A_right[0, perms[2][2]] * SC_IP_B_i[perms[2][1], 0] * AC_IP_A_left[perms[2][0], 0] * term_43_EA    
        # idx = 2, X
        term_43_C_CS_EA += C_prefactor * CA_EA_A_right[0, perms[0][0]] * CS_EA_B_a[0, perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * term_13
        term_43_X_CS_EA += X_prefactor * CA_EA_A_right[0, perms[2][0]] * CS_EA_B_a[0, perms[2][1]] * AC_IP_A_left[perms[2][2], 0] * term_13
    return term_31_X, term_31_C, term_41_X, term_41_C, term_13_X, term_13_C, term_14_X, term_14_C, term_21_X_CS_virt, term_21_C_CS_virt,\
        term_12_X_SC_occ, term_12_C_SC_occ, term_43_X_CS_EA, term_43_C_CS_EA, term_43_X_SC_IP, term_43_C_SC_IP,\
        term_42_X, term_42_C, term_41_X_2S, term_41_C_2S, term_23_X_2S, term_23_C_2S, term_13_X_2S, term_13_C_2S, term_21_X_SC_occ_arb, term_21_C_SC_occ_arb, \
        term_31_C_2S_EA_IP_arb, term_31_X_2S_EA_IP_arb, term_31_C_2S_occ_IP_arb, term_31_X_2S_occ_IP_arb, term_14_C_2S_IP_EA_arb, term_14_X_2S_IP_EA_arb, term_21_C_3S_A_occ_B_virt_B_occ_arb, \
        term_21_X_3S_A_occ_B_virt_B_occ_arb, term_14_C_2S_occ_arb_EA_arb, term_14_X_2S_occ_arb_EA_arb, term_21_C_3S_B_virt_B_occ_arb_A_virt_arb, term_21_X_3S_B_virt_B_occ_arb_A_virt_arb, \
        term_14_C_2S_virt_EA_arb, term_14_X_2S_virt_EA_arb



def calc_term_BAAA_2(indexes, value         ,
                            SC_B_AB_traf        ,
                            SC_B_AB_traf_pa     ,
                            CS_B_BA_traf        ,
                            SC_A_BA_traf        ,
                            SC_A_BA_traf_pa     ,
                            CS_A_AB_traf        ,
                            CS_A_AB_traf_iq     ,
                            term_31             ,
                            term_41             ,
                            term_13             ,
                            term_14             ,
                            term_14_2S_IP_EA_arb,
                            term_31_2S_EA_IP_arb,
                            term_43_EA          ,
                            term_43_IP          ,
                            SC_IP_B_i           ,
                            SC_IP_B_p           ,
                            CS_EA_B_a           ,
                            CS_EA_B_q           ,
                            ACA_A_right         ,
                            SCS_B_AA_pa         ,
                            ACA_A_left          ,
                            CA_EA_A_right       ,
                            AC_IP_A_right       ,
                            CA_EA_A_left        ,
                            AC_IP_A_left        ,):
    
    term_31_X = np.float64(0.)
    term_31_C = np.float64(0.)
    term_41_X = np.float64(0.)
    term_41_C = np.float64(0.)
    
    term_13_X = np.float64(0.)
    term_13_C = np.float64(0.)
    term_14_X = np.float64(0.)
    term_14_C = np.float64(0.)    
    
    term_41_X_2S = np.float64(0.)
    term_41_C_2S = np.float64(0.)
    term_42_X = np.float64(0.)
    term_42_C = np.float64(0.)    
    term_21_X_CS_virt = np.float64(0.)
    term_21_C_CS_virt = np.float64(0.)
    term_12_X_SC_occ  = np.float64(0.)
    term_12_C_SC_occ  = np.float64(0.)      
    term_43_X_CS_EA   = np.float64(0.)
    term_43_C_CS_EA   = np.float64(0.)    
    term_43_X_SC_IP   = np.float64(0.)
    term_43_C_SC_IP   = np.float64(0.) 
    term_23_C_2S_IP = np.float64(0.)
    term_23_X_2S_IP = np.float64(0.)
    term_13_C_2S_IP = np.float64(0.)
    term_13_X_2S_IP = np.float64(0.)
    #perms = list(distinct_permutations(indexes))
    term_21_X_SC_occ_arb = np.float64(0.0)
    term_21_C_SC_occ_arb = np.float64(0.0)

    term_31_C_2S_EA_IP_arb = np.float64(0.0)
    term_31_X_2S_EA_IP_arb = np.float64(0.0)
    term_31_C_2S_occ_IP_arb = np.float64(0.0)
    term_31_X_2S_occ_IP_arb = np.float64(0.0)
    term_14_C_2S_IP_EA_arb = np.float64(0.0) 
    term_14_X_2S_IP_EA_arb = np.float64(0.0)
    term_21_C_3S_A_occ_B_virt_B_occ_arb = np.float64(0.0)
    term_21_X_3S_A_occ_B_virt_B_occ_arb = np.float64(0.0)
    term_14_C_2S_occ_arb_EA_arb = np.float64(0.0)
    term_14_X_2S_occ_arb_EA_arb = np.float64(0.0)
    term_21_C_3S_B_virt_B_occ_arb_A_virt_arb = np.float64(0.0)
    term_21_X_3S_B_virt_B_occ_arb_A_virt_arb = np.float64(0.0)

    term_14_C_2S_virt_EA_arb = np.float64(0.0)
    term_14_X_2S_virt_EA_arb = np.float64(0.0)

    a, b, c = indexes

    p0 = (a, b, c)
    perms = [p0]

    C_prefactor = np.float64(4.)
    X_prefactor = np.float64(-2.)
    

#

#term_21_C_3S_A_occ_B_virt_B_occ_arb += C_prefactor * SC_A_BA_traf[a] * SCS_B_AA_pa[p, e] * value
#X_AO_ABAA_21_3S_A_occ_B_virt_B_occ_arb
#C_AO_AAAB_43_2S_IP_EA_arb CA_EA_B_left[:,d-N_SAO_A] * SC_IP_B_i[n, :] * [:,q-N_SAO_A] #!term_41 !!!!

    term_14_C_2S_virt_EA_arb += C_prefactor * AC_IP_A_right[perms[0][0], :] * CS_EA_B_q[:, perms[0][1]] * CS_A_AB_traf[perms[0][2]] * value
    term_14_X_2S_virt_EA_arb += X_prefactor * AC_IP_A_right[perms[0][0], :] * CS_EA_B_q[:, perms[0][1]] * CS_A_AB_traf[perms[0][2]] * value

    term_21_C_3S_B_virt_B_occ_arb_A_virt_arb += C_prefactor * SCS_B_AA_pa[perms[0][0], perms[0][1]] * CS_A_AB_traf_iq[perms[0][2]] * value
    term_21_X_3S_B_virt_B_occ_arb_A_virt_arb += X_prefactor * SCS_B_AA_pa[perms[0][0], perms[0][1]] * CS_A_AB_traf_iq[perms[0][2]] * value
    term_21_C_3S_A_occ_B_virt_B_occ_arb += C_prefactor * SCS_B_AA_pa[perms[0][0], perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value
    term_21_X_3S_A_occ_B_virt_B_occ_arb += X_prefactor * SCS_B_AA_pa[perms[0][0], perms[0][1]] * SC_A_BA_traf[perms[0][2]] * value
    term_14_C_2S_IP_EA_arb += C_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * CS_EA_B_q[:, perms[0][2]] * term_14_2S_IP_EA_arb
    term_14_X_2S_IP_EA_arb += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * CS_EA_B_q[:, perms[0][2]] * term_14_2S_IP_EA_arb
    term_14_C_2S_occ_arb_EA_arb += C_prefactor * AC_IP_A_right[perms[0][0], :] * CS_EA_B_q[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value
    term_14_X_2S_occ_arb_EA_arb += X_prefactor * AC_IP_A_right[perms[0][0], :] * CS_EA_B_q[:, perms[0][1]] * SC_A_BA_traf_pa[perms[0][2]] * value
    term_31_C_2S_occ_IP_arb += C_prefactor * SC_A_BA_traf[perms[0][0]] * CA_EA_A_left[:, perms[0][1]] * SC_IP_B_p[perms[0][2], :] * value
    term_31_X_2S_occ_IP_arb += X_prefactor * SC_A_BA_traf[perms[0][0]] * CA_EA_A_left[:, perms[0][1]] * SC_IP_B_p[perms[0][2], :] * value
    term_31_C_2S_EA_IP_arb += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * SC_IP_B_p[perms[0][2], :] * term_31_2S_EA_IP_arb
    term_31_X_2S_EA_IP_arb += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * SC_IP_B_p[perms[0][2], :] * term_31_2S_EA_IP_arb
    term_12_X_SC_occ += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * SC_B_AB_traf[perms[0][2]] * value
    term_12_C_SC_occ += C_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * SC_B_AB_traf[perms[0][2]] * value        
    term_21_X_SC_occ_arb += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * SC_B_AB_traf_pa[perms[0][2]] * value
    term_21_C_SC_occ_arb += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * SC_B_AB_traf_pa[perms[0][2]] * value        
    term_21_X_CS_virt += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value    
    term_21_C_CS_virt += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * CS_B_BA_traf[perms[0][2]] * value   
    
    term_42_X += X_prefactor * SC_B_AB_traf[perms[0][0]] * CS_EA_B_a[0, perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * value
    term_42_C += C_prefactor * SC_B_AB_traf[perms[0][0]] * CS_EA_B_a[0, perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * value
    
    term_41_X += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * term_41
    term_41_C += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * term_41
    term_14_X += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * AC_IP_A_right[perms[0][2], 0] * term_14
    term_14_C += C_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * AC_IP_A_right[perms[0][2], 0] * term_14
    
    term_41_X_2S += X_prefactor * AC_IP_A_left[perms[0][0], 0] * SC_A_BA_traf[perms[0][1]] * CS_EA_B_a[0, perms[0][2]] * value   
    term_41_C_2S += C_prefactor * AC_IP_A_left[perms[0][0], 0] * SC_A_BA_traf[perms[0][1]] * CS_EA_B_a[0, perms[0][2]] * value  
    term_23_X_2S_IP += X_prefactor * SC_IP_B_i[perms[0][0], 0] * CS_B_BA_traf[perms[0][1]] * CA_EA_A_right[0, perms[0][2]] * value   
    term_23_C_2S_IP += C_prefactor * SC_IP_B_i[perms[0][0], 0] * CS_B_BA_traf[perms[0][1]] * CA_EA_A_right[0, perms[0][2]] * value  
    term_13_X_2S_IP += X_prefactor * SC_IP_B_i[perms[0][0], 0] * CS_A_AB_traf[perms[0][1]] * CA_EA_A_right[0, perms[0][2]] * value   
    term_13_C_2S_IP += C_prefactor * SC_IP_B_i[perms[0][0], 0] * CS_A_AB_traf[perms[0][1]] * CA_EA_A_right[0, perms[0][2]] * value  

    term_31_X += X_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * CA_EA_A_left[0, perms[0][2]] * term_31    
    term_31_C += C_prefactor * ACA_A_right[perms[0][0], perms[0][1]] * CA_EA_A_left[0, perms[0][2]] * term_31  
    term_13_X += X_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * CA_EA_A_right[0, perms[0][2]] * term_13    
    term_13_C += C_prefactor * ACA_A_left[perms[0][0], perms[0][1]] * CA_EA_A_right[0, perms[0][2]] * term_13 
    
    term_43_X_SC_IP += X_prefactor * CA_EA_A_right[0, perms[0][0]] * SC_IP_B_i[perms[0][1], 0] * AC_IP_A_left[perms[0][2], 0] * term_43_EA
    term_43_C_SC_IP += C_prefactor * CA_EA_A_right[0, perms[0][0]] * SC_IP_B_i[perms[0][1], 0] * AC_IP_A_left[perms[0][2], 0] * term_43_EA
    
    term_43_C_CS_EA += C_prefactor * CA_EA_A_right[0, perms[0][0]] * CS_EA_B_a[0, perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * term_43_IP
    term_43_X_CS_EA += X_prefactor * CA_EA_A_right[0, perms[0][0]] * CS_EA_B_a[0, perms[0][1]] * AC_IP_A_left[perms[0][2], 0] * term_43_IP
    
    return term_31_X, term_31_C, term_41_X, term_41_C, term_13_X, term_13_C, term_14_X, term_14_C,term_21_X_CS_virt, term_21_C_CS_virt,\
        term_12_X_SC_occ, term_12_C_SC_occ, term_43_X_CS_EA, term_43_C_CS_EA, term_43_X_SC_IP, term_43_C_SC_IP, term_42_X, term_42_C, term_41_X_2S, term_41_C_2S,\
        term_23_X_2S_IP, term_23_C_2S_IP, term_13_C_2S_IP, term_13_X_2S_IP, term_21_X_SC_occ_arb, term_21_C_SC_occ_arb, \
        term_31_C_2S_EA_IP_arb, term_31_X_2S_EA_IP_arb, term_31_C_2S_occ_IP_arb, term_31_X_2S_occ_IP_arb, term_14_C_2S_IP_EA_arb, term_14_X_2S_IP_EA_arb, term_21_C_3S_A_occ_B_virt_B_occ_arb, \
        term_21_X_3S_A_occ_B_virt_B_occ_arb, term_14_C_2S_occ_arb_EA_arb, term_14_X_2S_occ_arb_EA_arb, term_21_C_3S_B_virt_B_occ_arb_A_virt_arb, term_21_X_3S_B_virt_B_occ_arb_A_virt_arb, \
        term_14_C_2S_virt_EA_arb, term_14_X_2S_virt_EA_arb


def   calc_BAAA(twoelint_block,
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
ACA_B_right  ,
SCS_B_AA_pa  ,
ACA_A_left   ,
ACA_B_left   ,
CA_EA_A_right,
AC_IP_B_right,
AC_IP_A_right,
CA_EA_B_right,
CA_EA_A_left ,
AC_IP_B_left ,
AC_IP_A_left ,
CA_EA_B_left ,
NBAS_A ):
    """Calculation for BAAA type."""
    # Iterate through the categories in the "LOCAL" key

    fc_21_coul_CS_virt = np.float64(0.)
    fc_21_exch_CS_virt = np.float64(0.)
    fc_12_coul_SC_occ  = np.float64(0.)
    fc_12_exch_SC_occ  = np.float64(0.)

    fc_21_coul_SC_occ_arb  = np.float64(0.)
    fc_21_exch_SC_occ_arb  = np.float64(0.)

    fc_43_coul_CS_EA   = np.float64(0.)
    fc_43_exch_CS_EA   = np.float64(0.)
    fc_43_coul_SC_IP   = np.float64(0.)
    fc_43_exch_SC_IP   = np.float64(0.)

    fc_31_coul = np.float64(0.)
    fc_31_exch = np.float64(0.)
    
    fc_13_coul_2S_IP = np.float64(0.)
    fc_13_exch_2S_IP = np.float64(0.)
    
    fc_41_coul = np.float64(0.)
    fc_41_exch = np.float64(0.)
    
    fc_14_coul = np.float64(0.)
    fc_14_exch = np.float64(0.)    
    fc_13_coul = np.float64(0.)
    fc_13_exch = np.float64(0.)
    
    fc_42_coul = np.float64(0.)
    fc_42_exch = np.float64(0.)
    
    fc_41_coul_2S = np.float64(0.)
    fc_41_exch_2S = np.float64(0.)
    
    fc_23_coul_2S_IP = np.float64(0.)
    fc_23_exch_2S_IP = np.float64(0.)    
    
    fc_coul_31_2S_EA_IP_arb = np.float64(0.)
    fc_exch_31_2S_EA_IP_arb = np.float64(0.)
    fc_coul_31_2S_occ_IP_arb = np.float64(0.)
    fc_exch_31_2S_occ_IP_arb = np.float64(0.)
    fc_coul_14_2S_IP_EA_arb = np.float64(0.)
    fc_exch_14_2S_IP_EA_arb = np.float64(0.)
    fc_coul_21_3S_A_occ_B_virt_B_occ_arb = np.float64(0.)
    fc_exch_21_3S_A_occ_B_virt_B_occ_arb = np.float64(0.)
    fc_coul_14_2S_occ_arb_EA_arb = np.float64(0.)
    fc_exch_14_2S_occ_arb_EA_arb = np.float64(0.)
    fc_coul_21_3S_B_virt_B_occ_arb_A_virt_arb = np.float64(0.)
    fc_exch_21_3S_B_virt_B_occ_arb_A_virt_arb = np.float64(0.)
    fc_coul_14_2S_virt_EA_arb = np.float64(0.)
    fc_exch_14_2S_virt_EA_arb = np.float64(0.)



    is_Ov = True
    for unique_count, twoelint_subblock in twoelint_block.items():
        if unique_count == 4:

            for row in twoelint_subblock:
                i = int(row[0])
                j = int(row[1])
                k = int(row[2])
                l = int(row[3])
                # Branch-minimized logic
                row_input_1based = tuple(int(x) for x in row[:4])
                swap = (j < k) and (j <= l)
                bad  = (j >= k) and (k < l)
            
                if bad:
                    raise ValueError("invalid (j,k,l) ordering: requires not (j>=k and k<l)")
            
                if swap:
                    row[2], row[3] = row[3], row[2]  # swap j,l           
                ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                if is_Ov:
                    
                    SC_B_AB_traf  = SC_B_AB_ia[:, ind_B[0]]
                    SC_B_AB_traf_pa  = SC_B_AB_pa[:, ind_B[0]]
                    CS_B_BA_traf  = CS_B_BA_ia[ind_B[0], :]
                    SC_A_BA_traf  = SC_A_BA_ia[ind_B[0], :]
                    SC_A_BA_traf_pa  = SC_A_BA_pa[ind_B[0], :]
                    CS_A_AB_traf  = CS_A_AB_ia[:, ind_B[0]]
                    CS_A_AB_traf_iq  = CS_A_AB_iq[:, ind_B[0]]
                    term_31       = value * AC_IP_B_left[ind_B[0], 0] 
                    term_41       = value * CA_EA_B_left[0, ind_B[0]]      
                    term_13       = value * AC_IP_B_right[ind_B[0], 0] 
                    term_14       = value * CA_EA_B_right[0, ind_B[0]]     
                    term_43_EA    = value * CA_EA_B_left[0, ind_B[0]]
                    term_43_IP    = value * AC_IP_B_right[ind_B[0], 0] 
                    term_14_2S_IP_EA_arb = value * SC_IP_A_i[ind_B[0], :]
                    term_31_2S_EA_IP_arb = value * CS_EA_A_a[:,ind_B[0]] 

                    term_31_X, term_31_C, term_41_X, term_41_C, term_13_X, term_13_C, term_14_X, term_14_C, term_21_X_CS_virt, term_21_C_CS_virt, term_12_X_SC_occ, term_12_C_SC_occ, term_43_X_CS_EA, term_43_C_CS_EA,\
                        term_43_X_SC_IP, term_43_C_SC_IP, term_42_X, term_42_C, term_41_X_2S, term_41_C_2S, term_23_X_2S, term_23_C_2S, term_13_X_2S, term_13_C_2S, term_21_X_SC_occ_arb, term_21_C_SC_occ_arb, \
                        term_31_C_2S_EA_IP_arb, term_31_X_2S_EA_IP_arb, term_31_C_2S_occ_IP_arb, term_31_X_2S_occ_IP_arb, term_14_C_2S_IP_EA_arb, term_14_X_2S_IP_EA_arb, term_21_C_3S_A_occ_B_virt_B_occ_arb, \
                            term_21_X_3S_A_occ_B_virt_B_occ_arb, term_14_C_2S_occ_arb_EA_arb, term_14_X_2S_occ_arb_EA_arb, term_21_C_3S_B_virt_B_occ_arb_A_virt_arb, term_21_X_3S_B_virt_B_occ_arb_A_virt_arb, \
                            term_14_C_2S_virt_EA_arb, term_14_X_2S_virt_EA_arb \
                            = calc_term_BAAA_4(ind_A,   value         ,
                            SC_B_AB_traf        ,
                            SC_B_AB_traf_pa     ,
                            CS_B_BA_traf        ,
                            SC_A_BA_traf        ,
                            SC_A_BA_traf_pa     ,
                            CS_A_AB_traf        ,
                            CS_A_AB_traf_iq     ,
                            term_31             ,
                            term_41             ,
                            term_13             ,
                            term_14             ,
                            term_14_2S_IP_EA_arb,
                            term_31_2S_EA_IP_arb,
                            term_43_EA          ,
                            term_43_IP          ,
                            SC_IP_B_i           ,
                            SC_IP_B_p           ,
                            CS_EA_B_a           ,
                            CS_EA_B_q           ,
                            ACA_A_right         ,
                            SCS_B_AA_pa         ,
                            ACA_A_left          ,
                            CA_EA_A_right       ,
                            AC_IP_A_right       ,
                            CA_EA_A_left        ,
                            AC_IP_A_left        ,)
                    
                    fc_coul_14_2S_virt_EA_arb += term_14_C_2S_virt_EA_arb
                    fc_exch_14_2S_virt_EA_arb += term_14_X_2S_virt_EA_arb

                    fc_coul_21_3S_B_virt_B_occ_arb_A_virt_arb += term_21_C_3S_B_virt_B_occ_arb_A_virt_arb
                    fc_exch_21_3S_B_virt_B_occ_arb_A_virt_arb += term_21_X_3S_B_virt_B_occ_arb_A_virt_arb

                    fc_21_coul_SC_occ_arb += term_21_C_SC_occ_arb
                    fc_21_exch_SC_occ_arb += term_21_X_SC_occ_arb

                    fc_coul_31_2S_EA_IP_arb += term_31_C_2S_EA_IP_arb
                    fc_exch_31_2S_EA_IP_arb += term_31_X_2S_EA_IP_arb
                    fc_coul_31_2S_occ_IP_arb += term_31_C_2S_occ_IP_arb
                    fc_exch_31_2S_occ_IP_arb += term_31_X_2S_occ_IP_arb
                    fc_coul_14_2S_IP_EA_arb += term_14_C_2S_IP_EA_arb
                    fc_exch_14_2S_IP_EA_arb += term_14_X_2S_IP_EA_arb
                    fc_coul_21_3S_A_occ_B_virt_B_occ_arb += term_21_C_3S_A_occ_B_virt_B_occ_arb
                    fc_exch_21_3S_A_occ_B_virt_B_occ_arb += term_21_X_3S_A_occ_B_virt_B_occ_arb
                    fc_coul_14_2S_occ_arb_EA_arb += term_14_C_2S_occ_arb_EA_arb
                    fc_exch_14_2S_occ_arb_EA_arb += term_14_X_2S_occ_arb_EA_arb
                    fc_14_coul += term_14_C
                    fc_14_exch += term_14_X                    
                    fc_13_coul += term_13_C
                    fc_13_exch += term_13_X

                    fc_13_coul_2S_IP += term_13_C_2S
                    fc_13_exch_2S_IP += term_13_X_2S    
                    fc_31_coul += term_31_C
                    fc_31_exch += term_31_X
                    fc_21_coul_CS_virt += term_21_C_CS_virt 
                    fc_21_exch_CS_virt += term_21_X_CS_virt 
                    fc_12_coul_SC_occ  += term_12_C_SC_occ  
                    fc_12_exch_SC_occ  += term_12_X_SC_occ  
                    fc_43_coul_CS_EA   += term_43_C_CS_EA   
                    fc_43_exch_CS_EA   += term_43_X_CS_EA   
                    fc_43_coul_SC_IP   += term_43_C_SC_IP   
                    fc_43_exch_SC_IP   += term_43_X_SC_IP   
                    
                    fc_41_coul += term_41_C
                    fc_41_exch += term_41_X   
                    
                    fc_42_coul += term_42_C
                    fc_42_exch += term_42_X
                    
                    fc_41_coul_2S += term_41_C_2S
                    fc_41_exch_2S += term_41_X_2S 
                    
                    fc_23_coul_2S_IP += term_23_C_2S
                    fc_23_exch_2S_IP += term_23_X_2S               
                else:
                    print('No ov is not implemented for BAAA_4')
                    
                    #term_31 = value * AC_IP_B[ind_B[0], 0] 
                    #term_41 = value * CA_EA_B[0, ind_B[0]] 
                    #term_31_X, term_31_C, term_41_X, term_41_C = calc_term_BAAA_4(ind_A, term_31, term_41, ACA_A, CA_EA_A, AC_IP_A)
#
                    #fc_31_coul += term_31_C
                    #fc_41_coul += term_41_C
                    #fc_31_exch += term_31_X
                    #fc_41_exch += term_41_X
        elif unique_count == 3: 
            for row in twoelint_subblock:
                i = int(row[0])
                j = int(row[1])
                k = int(row[2])
                l = int(row[3])
                row_input_1based = tuple(int(x) for x in row[:4])
                row_norm_1based = row_input_1based
                is_type_1 = (k != l)
                if j == k:
                    row[2], row[3] = row[3], row[2]
                ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                if is_Ov:
                    #!itt valami nem jó, de csak az S
                    SC_B_AB_traf  = SC_B_AB_ia[:, ind_B[0]]
                    SC_B_AB_traf_pa  = SC_B_AB_pa[:, ind_B[0]]
                    CS_B_BA_traf  = CS_B_BA_ia[ind_B[0], :]
                    SC_A_BA_traf  = SC_A_BA_ia[ind_B[0], :]
                    SC_A_BA_traf_pa  = SC_A_BA_pa[ind_B[0], :]
                    CS_A_AB_traf  = CS_A_AB_ia[:, ind_B[0]]
                    CS_A_AB_traf_iq  = CS_A_AB_iq[:, ind_B[0]]
                    term_31       = value * AC_IP_B_left[ind_B[0], 0] 
                    term_41       = value * CA_EA_B_left[0, ind_B[0]]      
                    term_13       = value * AC_IP_B_right[ind_B[0], 0] 
                    term_14       = value * CA_EA_B_right[0, ind_B[0]]     
                    term_43_EA    = value * CA_EA_B_left[0, ind_B[0]]
                    term_43_IP    = value * AC_IP_B_right[ind_B[0], 0]                 
                    term_14_2S_IP_EA_arb = value * SC_IP_A_i[ind_B[0], :]
                    term_31_2S_EA_IP_arb = value * CS_EA_A_a[:,ind_B[0]] 

                    term_31_X, term_31_C, term_41_X, term_41_C, term_13_X, term_13_C, term_14_X, term_14_C, term_21_X_CS_virt, term_21_C_CS_virt,\
                        term_12_X_SC_occ, term_12_C_SC_occ, term_43_X_CS_EA, term_43_C_CS_EA, term_43_X_SC_IP, term_43_C_SC_IP,\
                        term_42_X, term_42_C, term_41_X_2S, term_41_C_2S, term_23_X_2S, term_23_C_2S, term_13_X_2S, term_13_C_2S, term_21_X_SC_occ_arb, term_21_C_SC_occ_arb, \
                        term_31_C_2S_EA_IP_arb, term_31_X_2S_EA_IP_arb, term_31_C_2S_occ_IP_arb, term_31_X_2S_occ_IP_arb, term_14_C_2S_IP_EA_arb, term_14_X_2S_IP_EA_arb, term_21_C_3S_A_occ_B_virt_B_occ_arb, \
                            term_21_X_3S_A_occ_B_virt_B_occ_arb, term_14_C_2S_occ_arb_EA_arb, term_14_X_2S_occ_arb_EA_arb, term_21_C_3S_B_virt_B_occ_arb_A_virt_arb, term_21_X_3S_B_virt_B_occ_arb_A_virt_arb, \
                            term_14_C_2S_virt_EA_arb, term_14_X_2S_virt_EA_arb \
                            = calc_term_BAAA_3(ind_A, is_type_1, value         ,
                            SC_B_AB_traf        ,
                            SC_B_AB_traf_pa     ,
                            CS_B_BA_traf        ,
                            SC_A_BA_traf        ,
                            SC_A_BA_traf_pa     ,
                            CS_A_AB_traf        ,
                            CS_A_AB_traf_iq     ,
                            term_31             ,
                            term_41             ,
                            term_13             ,
                            term_14             ,
                            term_14_2S_IP_EA_arb,
                            term_31_2S_EA_IP_arb,
                            term_43_EA          ,
                            term_43_IP          ,
                            SC_IP_B_i           ,
                            SC_IP_B_p           ,
                            CS_EA_B_a           ,
                            CS_EA_B_q           ,
                            ACA_A_right         ,
                            SCS_B_AA_pa         ,
                            ACA_A_left          ,
                            CA_EA_A_right       ,
                            AC_IP_A_right       ,
                            CA_EA_A_left        ,
                            AC_IP_A_left        ,)

                    fc_coul_14_2S_virt_EA_arb += term_14_C_2S_virt_EA_arb
                    fc_exch_14_2S_virt_EA_arb += term_14_X_2S_virt_EA_arb

                    fc_coul_21_3S_B_virt_B_occ_arb_A_virt_arb += term_21_C_3S_B_virt_B_occ_arb_A_virt_arb
                    fc_exch_21_3S_B_virt_B_occ_arb_A_virt_arb += term_21_X_3S_B_virt_B_occ_arb_A_virt_arb

                    fc_coul_31_2S_EA_IP_arb += term_31_C_2S_EA_IP_arb
                    fc_exch_31_2S_EA_IP_arb += term_31_X_2S_EA_IP_arb
                    fc_coul_31_2S_occ_IP_arb += term_31_C_2S_occ_IP_arb
                    fc_exch_31_2S_occ_IP_arb += term_31_X_2S_occ_IP_arb
                    fc_coul_14_2S_IP_EA_arb += term_14_C_2S_IP_EA_arb
                    fc_exch_14_2S_IP_EA_arb += term_14_X_2S_IP_EA_arb
                    fc_coul_21_3S_A_occ_B_virt_B_occ_arb += term_21_C_3S_A_occ_B_virt_B_occ_arb
                    fc_exch_21_3S_A_occ_B_virt_B_occ_arb += term_21_X_3S_A_occ_B_virt_B_occ_arb
                    fc_coul_14_2S_occ_arb_EA_arb += term_14_C_2S_occ_arb_EA_arb
                    fc_exch_14_2S_occ_arb_EA_arb += term_14_X_2S_occ_arb_EA_arb
                    fc_21_coul_SC_occ_arb += term_21_C_SC_occ_arb
                    fc_21_exch_SC_occ_arb += term_21_X_SC_occ_arb

                    fc_13_coul += term_13_C
                    fc_13_exch += term_13_X
                    fc_14_coul += term_14_C
                    fc_14_exch += term_14_X

                    fc_13_coul_2S_IP += term_13_C_2S
                    fc_13_exch_2S_IP += term_13_X_2S                    
                    fc_23_coul_2S_IP += term_23_C_2S
                    fc_23_exch_2S_IP += term_23_X_2S
                    
                    fc_31_coul += term_31_C
                    fc_31_exch += term_31_X
                    fc_21_coul_CS_virt += term_21_C_CS_virt 
                    fc_21_exch_CS_virt += term_21_X_CS_virt 
                    fc_12_coul_SC_occ  += term_12_C_SC_occ  
                    fc_12_exch_SC_occ  += term_12_X_SC_occ  
                    fc_43_coul_CS_EA   += term_43_C_CS_EA   
                    fc_43_exch_CS_EA   += term_43_X_CS_EA   
                    fc_43_coul_SC_IP   += term_43_C_SC_IP   
                    fc_43_exch_SC_IP   += term_43_X_SC_IP   
                    
                    fc_41_coul += term_41_C
                    fc_41_exch += term_41_X  
                    
                    fc_42_coul += term_42_C
                    fc_42_exch += term_42_X
                    fc_41_coul_2S += term_41_C_2S
                    fc_41_exch_2S += term_41_X_2S 
                else:
                    print('No ov is not implemented for BAAA_3')
                    #
                    #term_31 = value * AC_IP_B[ind_B[0], 0] 
                    #term_41 = value * CA_EA_B[0, ind_B[0]] 
                    #term_31_X, term_31_C, term_41_X, term_41_C= calc_term_BAAA_3(ind_A, term_31, term_41, ACA_A, CA_EA_A, AC_IP_A, is_type_1)#, twoel_AO_AAAB_canonical, ind_B[0], value)
#
                    #fc_31_coul += term_31_C
                    #fc_41_coul += term_41_C
                    #fc_31_exch += term_31_X
                    #fc_41_exch += term_41_X
        else:
            for row in twoelint_subblock:
                ind_A, ind_B, value = separate_indexes_and_values(row, NBAS_A)
                if is_Ov:
                    SC_B_AB_traf  = SC_B_AB_ia[:, ind_B[0]]
                    SC_B_AB_traf_pa  = SC_B_AB_pa[:, ind_B[0]]
                    CS_B_BA_traf  = CS_B_BA_ia[ind_B[0], :]
                    SC_A_BA_traf_pa  = SC_A_BA_pa[ind_B[0], :]
                    SC_A_BA_traf  = SC_A_BA_ia[ind_B[0], :]
                    CS_A_AB_traf  = CS_A_AB_ia[:, ind_B[0]]
                    CS_A_AB_traf_iq  = CS_A_AB_iq[:, ind_B[0]]
                    term_31       = value * AC_IP_B_left[ind_B[0], 0] 
                    term_41       = value * CA_EA_B_left[0, ind_B[0]]      
                    term_13       = value * AC_IP_B_right[ind_B[0], 0] 
                    term_14       = value * CA_EA_B_right[0, ind_B[0]]     
                    term_43_EA    = value * CA_EA_B_left[0, ind_B[0]]
                    term_43_IP    = value * AC_IP_B_right[ind_B[0], 0]    
                    term_14_2S_IP_EA_arb = value * SC_IP_A_i[ind_B[0], :]
                    term_31_2S_EA_IP_arb = value * CS_EA_A_a[:,ind_B[0]] 

                    term_31_X, term_31_C, term_41_X, term_41_C, term_13_X, term_13_C, term_14_X, term_14_C,term_21_X_CS_virt, term_21_C_CS_virt,\
                    term_12_X_SC_occ, term_12_C_SC_occ, term_43_X_CS_EA, term_43_C_CS_EA, term_43_X_SC_IP, term_43_C_SC_IP, term_42_X, term_42_C, term_41_X_2S, term_41_C_2S,\
                    term_23_X_2S_IP, term_23_C_2S_IP, term_13_C_2S_IP, term_13_X_2S_IP, term_21_X_SC_occ_arb, term_21_C_SC_occ_arb, \
                    term_31_C_2S_EA_IP_arb, term_31_X_2S_EA_IP_arb, term_31_C_2S_occ_IP_arb, term_31_X_2S_occ_IP_arb, term_14_C_2S_IP_EA_arb, term_14_X_2S_IP_EA_arb, term_21_C_3S_A_occ_B_virt_B_occ_arb, \
                    term_21_X_3S_A_occ_B_virt_B_occ_arb, term_14_C_2S_occ_arb_EA_arb, term_14_X_2S_occ_arb_EA_arb, term_21_C_3S_B_virt_B_occ_arb_A_virt_arb, term_21_X_3S_B_virt_B_occ_arb_A_virt_arb, \
                    term_14_C_2S_virt_EA_arb, term_14_X_2S_virt_EA_arb \
                        = calc_term_BAAA_2(ind_A, value         ,
                            SC_B_AB_traf        ,
                            SC_B_AB_traf_pa     ,
                            CS_B_BA_traf        ,
                            SC_A_BA_traf        ,
                            SC_A_BA_traf_pa     ,
                            CS_A_AB_traf        ,
                            CS_A_AB_traf_iq     ,
                            term_31             ,
                            term_41             ,
                            term_13             ,
                            term_14             ,
                            term_14_2S_IP_EA_arb,
                            term_31_2S_EA_IP_arb,
                            term_43_EA          ,
                            term_43_IP          ,
                            SC_IP_B_i           ,
                            SC_IP_B_p           ,
                            CS_EA_B_a           ,
                            CS_EA_B_q           ,
                            ACA_A_right         ,
                            SCS_B_AA_pa         ,
                            ACA_A_left          ,
                            CA_EA_A_right       ,
                            AC_IP_A_right       ,
                            CA_EA_A_left        ,
                            AC_IP_A_left        ,)
                    
                    fc_coul_14_2S_virt_EA_arb += term_14_C_2S_virt_EA_arb
                    fc_exch_14_2S_virt_EA_arb += term_14_X_2S_virt_EA_arb

                    fc_coul_21_3S_B_virt_B_occ_arb_A_virt_arb += term_21_C_3S_B_virt_B_occ_arb_A_virt_arb
                    fc_exch_21_3S_B_virt_B_occ_arb_A_virt_arb += term_21_X_3S_B_virt_B_occ_arb_A_virt_arb

                    fc_coul_31_2S_EA_IP_arb +=  term_31_C_2S_EA_IP_arb
                    fc_exch_31_2S_EA_IP_arb +=  term_31_X_2S_EA_IP_arb
                    fc_coul_31_2S_occ_IP_arb += term_31_C_2S_occ_IP_arb
                    fc_exch_31_2S_occ_IP_arb += term_31_X_2S_occ_IP_arb
                    fc_coul_14_2S_IP_EA_arb += term_14_C_2S_IP_EA_arb
                    fc_exch_14_2S_IP_EA_arb += term_14_X_2S_IP_EA_arb
                    fc_coul_21_3S_A_occ_B_virt_B_occ_arb += term_21_C_3S_A_occ_B_virt_B_occ_arb
                    fc_exch_21_3S_A_occ_B_virt_B_occ_arb += term_21_X_3S_A_occ_B_virt_B_occ_arb
                    fc_coul_14_2S_occ_arb_EA_arb += term_14_C_2S_occ_arb_EA_arb
                    fc_exch_14_2S_occ_arb_EA_arb += term_14_X_2S_occ_arb_EA_arb

                    fc_21_coul_SC_occ_arb += term_21_C_SC_occ_arb
                    fc_21_exch_SC_occ_arb += term_21_X_SC_occ_arb

                    fc_13_coul += term_13_C
                    fc_13_exch += term_13_X
                    fc_14_coul += term_14_C
                    fc_14_exch += term_14_X
                    
                    fc_13_coul_2S_IP += term_13_C_2S_IP
                    fc_13_exch_2S_IP += term_13_X_2S_IP
                    fc_23_coul_2S_IP += term_23_C_2S_IP
                    fc_23_exch_2S_IP += term_23_X_2S_IP
                    
                    fc_31_coul += term_31_C
                    fc_31_exch += term_31_X
                    
                    fc_21_coul_CS_virt += term_21_C_CS_virt 
                    fc_21_exch_CS_virt += term_21_X_CS_virt 
                    fc_12_coul_SC_occ  += term_12_C_SC_occ  
                    fc_12_exch_SC_occ  += term_12_X_SC_occ  
                    fc_43_coul_CS_EA   += term_43_C_CS_EA   
                    fc_43_exch_CS_EA   += term_43_X_CS_EA   
                    fc_43_coul_SC_IP   += term_43_C_SC_IP   
                    fc_43_exch_SC_IP   += term_43_X_SC_IP   
                    
                    fc_41_coul += term_41_C
                    fc_41_exch += term_41_X  
                    
                    fc_42_coul += term_42_C
                    fc_42_exch += term_42_X
                    
                    fc_41_coul_2S += term_41_C_2S
                    fc_41_exch_2S += term_41_X_2S 
                else:
                    print('No ov is not implemented for BAAA_2')
                    #
                    #term_31 = value * AC_IP_B[ind_B[0], 0] 
                    #term_41 = value * CA_EA_B[0, ind_B[0]] 
                    #term_31_X, term_31_C, term_41_X, term_41_C = calc_term_BAAA_2(ind_A, term_31, term_41, ACA_A, CA_EA_A, AC_IP_A)
                    #fc_31_coul += term_31_C
                    #fc_31_exch += term_31_X
#
                    #fc_41_coul += term_41_C
                    #fc_41_exch += term_41_X        

    return fc_31_coul, fc_31_exch, fc_41_coul, fc_41_exch, fc_13_coul, fc_13_exch, fc_14_coul, fc_14_exch, fc_21_coul_CS_virt, fc_21_exch_CS_virt, \
        fc_12_coul_SC_occ, fc_12_exch_SC_occ, fc_43_coul_CS_EA, fc_43_exch_CS_EA, fc_43_coul_SC_IP, fc_43_exch_SC_IP, \
        fc_42_coul, fc_42_exch, fc_41_coul_2S, fc_41_exch_2S, fc_23_coul_2S_IP, fc_23_exch_2S_IP, fc_13_coul_2S_IP, fc_13_exch_2S_IP, \
        fc_21_coul_SC_occ_arb, fc_21_exch_SC_occ_arb, fc_coul_31_2S_EA_IP_arb, fc_exch_31_2S_EA_IP_arb, fc_coul_31_2S_occ_IP_arb, fc_exch_31_2S_occ_IP_arb, \
        fc_coul_14_2S_IP_EA_arb, fc_exch_14_2S_IP_EA_arb, fc_coul_21_3S_A_occ_B_virt_B_occ_arb, fc_exch_21_3S_A_occ_B_virt_B_occ_arb, fc_coul_14_2S_occ_arb_EA_arb, fc_exch_14_2S_occ_arb_EA_arb, \
        fc_coul_21_3S_B_virt_B_occ_arb_A_virt_arb, fc_exch_21_3S_B_virt_B_occ_arb_A_virt_arb, fc_coul_14_2S_virt_EA_arb, fc_exch_14_2S_virt_EA_arb