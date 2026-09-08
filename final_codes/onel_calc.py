import numpy as np
import os
import sys

def onel_cross_terms_Frenkel_alter(onel_matfrix_AB, CIS_matrix_A, CIS_matrix_B, LCAO_A, S_AB,
                            LCAO_B, NBAS_A: int, NBAS_B: int, dist):
    # Initializ some stuff
    # Instead of cutting the LCAOs I filled the other stuff up.
    V_21_term_1 = np.float64(0.)
    V_21_term_2 = np.float64(0.)
    # Get the final matrices for first term
    CA_1 = CIS_matrix_B @ LCAO_B.T
    CAf_1 = CA_1 @ onel_matfrix_AB.T
    CAfA_1 = CAf_1 @ LCAO_A 
    CAfAC_1 = CAfA_1 @ CIS_matrix_A.T
    CAfACA_1 = CAfAC_1 @ LCAO_A.T
    CAfACAS_1 = CAfACA_1 @ S_AB
    CAfACASA_1 = CAfACAS_1 @ LCAO_B
    for kdx in range(NBAS_B):
        V_21_term_1 += CAfACASA_1[kdx, kdx]
        
    # term 2
    CA_2 = CIS_matrix_A.T @ LCAO_A.T
    CAf_2 = CA_2 @ onel_matfrix_AB
    CAfA_2 = CAf_2 @ LCAO_B
    CAfAC_2 = CAfA_2 @ CIS_matrix_B
    CAfACA_2 = CAfAC_2 @ LCAO_B.T
    CAfACAS_2 = CAfACA_2 @ S_AB.T
    CAfACASA_2 = CAfACAS_2 @ LCAO_A
    for jdx in range(NBAS_A):
        V_21_term_2 += CAfACASA_2[jdx, jdx]
    
    # sum
    V_21 = V_21_term_1 - V_21_term_2
    if os.path.isfile('f_21_TM_contributions_alter.txt') == False:            
        with open('f_21_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'Mix V_42 COUPLING \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t F_AB term 1 \t F_AB term 2 \t  F_AB term \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{dist} \t {V_21_term_1} \t {V_21_term_2} \t {V_21}\n')
    else:
        with open('f_21_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'{dist} \t {V_21_term_1} \t {V_21_term_2} \t {V_21}\n')
    return V_21_term_1, V_21_term_2


def V_31_onel_alter(onel_matfrix_AB, onel_matrix_AA, CIS_matrix_A, CIS_vector_EA_A,
                        CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, distance):

# AB term 1
    S_BA = S_AB.T
    CA_EA_A = CIS_vector_EA_A @ LCAO_A.T
    CAf_EA_A = CA_EA_A @ onel_matrix_AA
    CAfA_EA_A = CAf_EA_A @ LCAO_A
    CAfAC_EA_A = CAfA_EA_A @ CIS_matrix_A.T
    CAfACA_EA_A = CAfAC_EA_A @ LCAO_A.T
    CAfACAS_EA_A = CAfACA_EA_A @ S_AB
    CAfACASA_EA_A = CAfACAS_EA_A @ LCAO_B
    V_31_term_1 = CAfACASA_EA_A @ CIS_vector_IP_B.T

# AB term 2
    CC_EA_A = CIS_vector_EA_A @ CIS_matrix_A.T
    CCA_EA_A = CC_EA_A @ LCAO_A.T
    CCAf_EA_A = CCA_EA_A @ onel_matfrix_AB
    CCAfA_EA_A = CCAf_EA_A @ LCAO_B
    V_31_term_2 = CCAfA_EA_A @ CIS_vector_IP_B.T
    
    V_31 = V_31_term_1 - V_31_term_2
    
    if os.path.isfile('f_31_TM_contributions_alter.txt') == False:            
        with open('f_31_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'Mix V_41 COUPLING  \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t F_AB term \t F_BA term \t sum \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{distance} \t  {V_31_term_1} \t {V_31_term_2} \t {V_31} \n')
    else:
        with open('f_31_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'{distance} \t  {V_31_term_1} \t {V_31_term_2} \t {V_31} \n')
    

    return V_31_term_1, V_31_term_2


def V_32_onel_alter(onel_matfrix_AB, onel_matrix_BB, CIS_matrix_B, CIS_vector_EA_A,
            CIS_vector_IP_B, LCAO_A, S_AB,LCAO_B, distance):

    # AB term 1
    CA_EA_A = CIS_vector_EA_A @ LCAO_A.T
    CAf_EA_A = CA_EA_A @ onel_matfrix_AB
    CAfA_EA_A = CAf_EA_A @ LCAO_B
    CAfAC_EA_A = CAfA_EA_A @ CIS_matrix_B.T
    V_32_term_1 = CAfAC_EA_A @ CIS_vector_IP_B.T
    
    # AB term 2
    CA_IP_B = CIS_vector_EA_A @ LCAO_A.T
    CAS_IP_B = CA_IP_B @ S_AB
    CASA_IP_B = CAS_IP_B @ LCAO_B
    CASAC_IP_B = CASA_IP_B @ CIS_matrix_B.T
    CASACA_IP_B = CASAC_IP_B @ LCAO_B.T
    CASACAf_IP_B = CASACA_IP_B @ onel_matrix_BB
    CASACAfA_IP_B = CASACAf_IP_B @ LCAO_B
    V_32_term_2 = CASACAfA_IP_B @ CIS_vector_IP_B.T
    
    V_32 = V_32_term_1 - V_32_term_2
    
    if os.path.isfile('f_32_TM_contributions_alter.txt') == False:            
        with open('f_32_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'Mix V_41 COUPLING  \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t F_AB term \t F_BA term \t sum \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{distance} \t  {V_32_term_1} \t {V_32_term_2} \t {V_32} \n')
    else:
        with open('f_32_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'{distance} \t  {V_32_term_1} \t {V_32_term_2} \t {V_32} \n')
    
    return V_32_term_1, V_32_term_2


def V_41_onel_alter(onel_matfrix_AB, onel_matrix_AA, CIS_matrix_A, CIS_vector_IP_A,
            CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, distance):
    V_41_AB_term = 0
    S_BA = S_AB.T
    onel_BA = onel_matfrix_AB.T
    
    # AB term 1
    CA_1 = CIS_vector_EA_B @ LCAO_B.T
    CAf_1 = CA_1 @ onel_BA 
    CAfA_1 = CAf_1 @ LCAO_A
    CAfAC_1 = CAfA_1 @ CIS_matrix_A.T
    V_41_AB_term = CAfAC_1 @ CIS_vector_IP_A.T
    
    # BA term 1
    CAS_2 = CA_1 @ S_BA
    CASA_2 = CAS_2 @ LCAO_A
    CASAC_2 = CASA_2 @ CIS_matrix_A.T
    CASACA_2 = CASAC_2 @ LCAO_A.T
    CASACAf_2 = CASACA_2 @ onel_matrix_AA
    CASACAfA_2 = CASACAf_2 @ LCAO_A
    V_41_BA_term = CASACAfA_2 @ CIS_vector_IP_A.T
    # sum
    V_41 = V_41_AB_term - V_41_BA_term
    
    if os.path.isfile('f_41_TM_contributions_alter.txt') == False:            
        with open('f_41_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'Mix V_41 COUPLING  \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t F_AB term \t F_BA term \t sum \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{distance} \t  {V_41_AB_term} \t {V_41_BA_term} \t {V_41} \n')
    else:
        with open('f_41_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'{distance} \t  {V_41_AB_term} \t {V_41_BA_term} \t {V_41} \n')
    return V_41_AB_term, V_41_BA_term


def V_42_onel_alter(onel_matfrix_AB, onel_BB, CIS_matrix_B, CIS_vector_IP_A,
                                           CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, distance):
    # AB term 1
    S_BA = S_AB.T
    onel_BA = onel_matfrix_AB.T

    CA_1 = CIS_vector_EA_B @ LCAO_B.T
    CAf_1 = CA_1 @ onel_BB
    CAfA_1 = CAf_1 @ LCAO_B
    CAfAC_1 = CAfA_1 @ CIS_matrix_B.T
    CAfACA_1 = CAfAC_1 @ LCAO_B.T
    CAfACAS_1 = CAfACA_1 @ S_BA
    CAfACASA_1 = CAfACAS_1 @ LCAO_A
    V_42_term_1 = CAfACASA_1 @ CIS_vector_IP_A.T
    

    # BA term 1
    CC_2 = CIS_vector_EA_B @ CIS_matrix_B.T
    CCA_2 = CC_2 @ LCAO_B.T
    CCAf_2 = CCA_2 @ onel_BA
    CCAfA_2 = CCAf_2 @ LCAO_A
    V_42_term_2 = CCAfA_2 @ CIS_vector_IP_A.T

    # sum
    V_42 = V_42_term_1 - V_42_term_2

    if os.path.isfile('f_42_TM_contributions_alter.txt') == False:            
        with open('f_42_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'Mix V_42 COUPLING \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t F_AB term 1 \t F_AB term 2 \t  F_AB term \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{distance} \t {V_42_term_1} \t {V_42_term_2} \t {V_42}\n')
    else:
        with open('f_42_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'{distance} \t {V_42_term_1} \t {V_42_term_2} \t {V_42}\n')

    return V_42_term_1, V_42_term_2


#def onel_cross_terms_mix(onel_matfrix_AB, CIS_matrix_A, CIS_matrix_B, CIS_vector_EA_A, CIS_vector_EA_B,
#                         CIS_vector_IP_A, CIS_vector_IP_B, LCAO_A, S_AB,
#                            LCAO_B, NBAS_A: int, NBAS_B: int, onel_matrix_AA, onel_matrix_BB, dist):
#    # Initializ some stuff
#    # Instead of cutting the LCAOs I filled the other stuff up.
#    V_31_term_1, V_31_term_2 = V_31_onel_alter(onel_matfrix_AB=onel_matfrix_AB, onel_matrix_AA=onel_matrix_AA,
#                                               CIS_matrix_A=CIS_matrix_A, CIS_vector_EA_A=CIS_vector_EA_A,
#                                           CIS_vector_IP_B=CIS_vector_IP_B, LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, distance=dist)
#    V_32_term_1, V_32_term_2 = V_32_onel_alter(onel_matfrix_AB, onel_matrix_BB, CIS_matrix_B, CIS_vector_EA_A,
#                                           CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, dist)
#    V_41_term_1, V_41_term_2 = V_41_onel_alter(onel_matfrix_AB, onel_matrix_AA,  CIS_matrix_A, CIS_vector_IP_A,
#                                           CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, dist)
#    V_42_term_1, V_42_term_2 = V_42_onel_alter(onel_matfrix_AB, onel_matrix_BB,  CIS_matrix_A, CIS_vector_IP_A,
#                                           CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, dist)
#    return V_31_term_1, V_31_term_2, V_32_term_1, V_32_term_2, V_41_term_1, V_41_term_2,  V_42_term_1, V_42_term_2


def onel_cross_terms_CT_alter(onel_matfrix_AB, CIS_vector_EA_A, CIS_vector_EA_B,
                         CIS_vector_IP_A, CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, dist):
    # Initializ some stuff
    # Instead of cutting the LCAOs I filled the CIS coeffs with zeroes.
    CA_IP_1 = CIS_vector_IP_B @ LCAO_B.T
    CAS_IP_1 = CA_IP_1 @ S_AB.T
    CASA_IP_1 = CAS_IP_1 @ LCAO_A
    CASAC_IP_1 = CASA_IP_1 @ CIS_vector_IP_A.T
    
    CA_EA_1 = CIS_vector_EA_A @ LCAO_A.T
    CAf_EA_1 = CA_EA_1 @ onel_matfrix_AB
    CAfA_EA_1 = CAf_EA_1 @ LCAO_B
    CAfAC_EA_1 = CAfA_EA_1 @ CIS_vector_EA_B.T
    
    CA_IP_2 = CIS_vector_IP_A @ LCAO_A.T
    CAf_IP_2 = CA_IP_2 @ onel_matfrix_AB
    CAfA_IP_2 = CAf_IP_2 @ LCAO_B
    CAfAC_IP_2 = CAfA_IP_2 @ CIS_vector_IP_B.T
    CAS_EA_2 = CA_EA_1 @ S_AB
    CASA_EA_2 = CAS_EA_2 @ LCAO_B
    CASAC_EA_2 = CASA_EA_2 @ CIS_vector_EA_B.T
    
    V_43_term_1 = (CASAC_IP_1 * CAfAC_EA_1)
    V_43_term_2 = (CASAC_EA_2 * CAfAC_IP_2)
    V_43 = V_43_term_1 - V_43_term_2
    if os.path.isfile('f_43_TM_contributions_alter.txt') == False:            
        with open('f_43_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'Mix V_42 COUPLING \n ')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'Distance \t F_AB term 1 \t F_AB term 2 \t  F_AB term \n')
            final_table.write(f'-' * 60 + '\n')
            final_table.write(f'{dist} \t {V_43_term_1} \t {V_43_term_2} \t {V_43}\n')
    else:
        with open('f_43_TM_contributions_alter.txt', 'a+') as final_table:
            final_table.write(f'{dist} \t {V_43_term_1} \t {V_43_term_2} \t {V_43}\n')
    
    return V_43_term_1, V_43_term_2


def onel_cross_terms_alter(f_AB, CIS_matrix_A_right, CIS_matrix_B_left, 
                        CIS_matrix_B_right, CIS_vector_EA_A_left, CIS_vector_EA_B_left, CIS_vector_IP_A_left,
                        CIS_vector_IP_B_left, CIS_vector_EA_B_right, CIS_vector_IP_A_right, 
                        f_AA, f_BB, LCAO_A, S_AB, LCAO_B, Nbas_A, Nbas_B, distance):
    
    V_21_term_1, V_21_term_2 = onel_cross_terms_Frenkel_alter(onel_matfrix_AB=f_AB, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_left,
                                                LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B, dist=distance)
    
    V_31_term_1, V_31_term_2, V_32_term_1, V_32_term_2, V_41_term_1, V_41_term_2,  V_42_term_1, V_42_term_2 \
        = onel_cross_terms_mix(onel_matfrix_AB=f_AB, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_right,
                                                            CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_left.T,
                                                            CIS_vector_IP_A=CIS_vector_IP_A_left.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T,
                                                            LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B, onel_matrix_AA=f_AA, onel_matrix_BB=f_BB, dist=distance)
        
    V_43_term_1, V_43_term_2 = onel_cross_terms_CT(onel_matfrix_AB=f_AB, CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_right.T,
                    CIS_vector_IP_A=CIS_vector_IP_A_right.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T, LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, dist=distance)
    
    return V_21_term_1, V_21_term_2, V_31_term_1, V_31_term_2, V_32_term_1, V_32_term_2, V_41_term_1, V_41_term_2,  V_42_term_1, V_42_term_2, V_43_term_1, V_43_term_2


def V_31_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_EA_A,
                        CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s, S_inv_blocks):

# AB term 1
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    #BBBBBBBBBBBBB = red_C_s.get('CIS_vector_EA_A_red') @ red_C_s.get('CIS_matrix_A_red').T @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red').T
    #BBBBBBBBBBBBB_2 = red_C_s.get('CIS_vector_EA_B_red') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red').T @ red_C_s.get('CIS_vector_IP_A_red').T
    
    #final_norm = np.float64(0.)
    #for c in range(len(red_C_s.get('CIS_vector_EA_A_red')[0, :])):
    #    for k in range(len(red_C_s.get('CIS_vector_IP_B_red')[0,:])):
    #        if c == 0 and k == 4:
    #            print('stop')
    #        final_norm += (red_C_s.get('CIS_vector_EA_A_red')[:, c]**2) * (red_C_s.get('CIS_vector_IP_B_red')[:, k]**2)
    #NNNNNorm = final_norm * (norms.get('norm_occ_only')**2)
    #NNNNNorm_31_ext_4 = red_C_s.get('CIS_vector_IP_B_red') @ S_blocks.get('S_AB_ia') @ red_C_s.get('CIS_vector_EA_A_red').T
    #NNNNNorm_31_ext_3 = red_C_s.get('CIS_vector_IP_B_red') @ S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ia') @ red_C_s.get('CIS_vector_EA_A_red').T
    #NNNNNorm_2 = (NNNNNorm_31_ext_4**2) * (norms.get('norm_occ_only')**2)
    #NNNNNorm_3 = NNNNNorm_31_ext_4 * NNNNNorm_31_ext_3 * (norms.get('norm_occ_only'))
    #
    #
    #int_CS_AB_1 = CIS_vector_IP_B @ S_BA_mo
    #int_CSC_AB_1 = int_CS_AB_1 @ CIS_matrix_A
    #int_CSCS_AB_1 = int_CSC_AB_1 @ S_AB_mo
    #int_CSCSf_AB_1 = int_CSCS_AB_1 @ f_BA_mo
    #term_AB_1 = int_CSCSf_AB_1 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
#
    #NNNNNorm_31 = red_C_s.get('CIS_vector_EA_A_red') @ red_C_s.get('CIS_matrix_A_red').T @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red').T 
    #NNNNNorm_31_ext_1 = red_C_s.get('CIS_vector_EA_A_red') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red').T 
    #NNNNNorm_31_ext_2 =  np.trace(S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ia') @ red_C_s.get('CIS_matrix_A_red').T)
    #NNNNNORM_ext = NNNNNorm_31_ext_1 * NNNNNorm_31_ext_2 * norms.get('norm_occ_only')
    
    #! előjelek jelenleg Péter verziójában vannak
    #cont_AB_1 = []
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_AB_ii')]:#, S_AB_mo @ norms.get('norm_BB_ai')]:
    #    for i, mat_2 in enumerate([S_blocks.get('S_BA_ia'), -S_blocks.get('S_BA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_BA_aa') * norms.get('norm_occ_only'), -S_blocks.get('S_BA_ai') @ norms.get('norm_AA_ia')]):
    #        f_AB_mo = F_blocks.get('f_AB_ai')
    #        if i > 1:
    #            f_AB_mo = F_blocks.get('f_AB_aa')
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_aa'))), -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ CIS_matrix_A @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ f_AB_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                #cont = CIS_vector_IP_B @ mat_1 @ CIS_matrix_A @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_EA_A.T * norms.get('norm_occ_only')
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ f_AB_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * norms.get('norm_occ_only')
    #            cont_AB_1.append(cont)

    cont_AA = red_C_s.get('CIS_vector_EA_A_red_left')  @ F_blocks.get('f_AA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    
    cont_AA_2 = red_C_s.get('CIS_vector_EA_A_red_left')  @ F_blocks.get('f_AA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    cont_AA_3 = red_C_s.get('CIS_vector_EA_A_red_left') @ red_C_s.get('CIS_matrix_A_red_right').T  @ F_blocks.get('f_AA_ii') @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    #cont_AA = red_C_s.get('CIS_vector_IP_B_red')  @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_matrix_A_red') @ F_blocks.get('f_AA_aa') @ red_C_s.get('CIS_vector_EA_A_red').T * (norms.get('norm_occ_only')**2)
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_AB_ii')]:#, S_AB_mo @ norms.get('norm_BB_ai')]:S_blocks.get
    #    for i, mat_2 in enumerate([np.identity(len(S_blocks.get('S_AA_aa'))), -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_aa'))), -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ CIS_matrix_A @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ F_blocks.get('f_AA_aa') @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                #cont = CIS_vector_IP_B @ mat_1 @ CIS_matrix_A @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_EA_A.T * norms.get('norm_occ_only')
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ F_blocks.get('f_AA_aa') @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * norms.get('norm_occ_only')
    #            cont_AA.append(cont)

    #AAAAAAAAAAAAAAa = red_C_s.get('CIS_vector_EA_A_red') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    

    cont_AB = red_C_s.get('CIS_vector_EA_A_red_left') @ red_C_s.get('CIS_matrix_A_red_right').T @ F_blocks.get('f_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    #first_pass = True
    #for mat_1 in [np.identity(max(np.shape(red_C_s.get('CIS_vector_IP_B_red'))))]:#, -S_blocks.get('S_BB_ai') * norms.get('norm_occ_only')]:
    #    f_AB_mo = F_blocks.get('f_AB_ii').copy()
    #    for i, mat_2 in enumerate([np.identity(len(S_blocks.get('S_AA_ii'))), S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T , S_blocks.get('S_AB_ii') @ norms.get('norm_BB_ai').T]):
    #        if i > 0:
    #            f_AB_mo = F_blocks.get('f_AB_ai').copy()
    #        elif i > 1:
    #            f_AB_mo = F_blocks.get('f_BB_ai').copy()
    #            #norms.get('norm_AB_ia') @ S_blocks.get('S_BA_ai', S_AB_mo @ norms.get('norm_BA_ai')]:
    #        for j, mat_3 in enumerate([np.identity(len(S_blocks.get('S_AA_aa'))), -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AB_ai') @ S_blocks.get('S_BA_ia')]):#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BB_ai').T ]):
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ CIS_matrix_A @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_AB_mo @ mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                #cont = CIS_vector_IP_B @ mat_1 @ CIS_matrix_A @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_EA_A.T * norms.get('norm_occ_only')
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_AB_mo @ mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * norms.get('norm_occ_only')
    #            cont_AB_2.append(cont)
    #            
    #f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    #f_BA_mo = onel_matfrix_BA
    #int_Cf_AB_2 = CIS_vector_IP_B @ f_BA_mo
    #int_CfC_AB_2 = int_Cf_AB_2 @ CIS_matrix_A
    #term_AB_2 = int_CfC_AB_2 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2) 
    
    #term_AB_corr_1 = -int_CfC_AB_2 @ S_AB_mo @ norms.get('norm_BA_ia') @ CIS_vector_EA_A.T #+
    #term_AB_corr_2 = CIS_vector_IP_B @ S_BA_mo @ norms.get('norm_AB_ai') @ f_BA_mo @ CIS_matrix_A @ CIS_vector_EA_A.T #-
    #term_AB_corr_3 = -CIS_vector_IP_B @ f_BA_mo @ S_AB_mo @ norms.get('norm_BA_ai') @ CIS_matrix_A @ CIS_vector_EA_A.T #+
    #
    #term_AB_corr_4 = CIS_vector_IP_B @ S_BA_mo @ norms.get('norm_AB_ai') @ f_BA_mo @ CIS_matrix_A @ S_AB_mo @ norms.get('norm_BA_ia') @ CIS_vector_EA_A.T #-
    #term_AB_corr_5 = CIS_vector_IP_B @ S_BA_mo @ norms.get('norm_AB_ai') @ f_BA_mo @ S_AB_mo @ norms.get('norm_BA_ai') @ CIS_matrix_A @ CIS_vector_EA_A.T #-
    #term_AB_corr_6 = -CIS_vector_IP_B @ f_BA_mo @ S_AB_mo @ norms.get('norm_BA_ai') @ CIS_matrix_A @ S_AB_mo @ norms.get('norm_BA_ia') @ CIS_vector_EA_A.T #+
    #
    #term_AB_corr_7 = CIS_vector_IP_B @ S_BA_mo @ norms.get('norm_AB_ai') @ f_BA_mo @ S_AB_mo @ norms.get('norm_BA_ai') @ CIS_matrix_A @ S_AB_mo @ norms.get('norm_BA_ia') @ CIS_vector_EA_A.T #-
    #
    #term_AB_2_1 = term_AB_2 * (norms.get('norm_occ_only')**2) + (term_AB_corr_1 + term_AB_corr_2 + term_AB_corr_3 + term_AB_corr_4 + term_AB_corr_5 + term_AB_corr_6 + term_AB_corr_7 )* norms.get('norm_occ_only')
    
    #term_31_AB = sum(cont_AB_1) - sum(cont_AB_2)#term_AB_2
    #term_31_AB_2 = cont_AB_1_2[0] - cont_AB_2[0]#term_AB_2
    
# BA term

    cont_BA = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_aa') @ F_blocks.get('f_BA_aa')  @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    
    cont_AB_2 = red_C_s.get('CIS_vector_EA_A_red_left') @ F_blocks.get('f_AB_aa') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_AB_ii')]:
    #    f_BA_mo = F_blocks.get('f_BA_aa').copy()#, S_AB_mo @ norms.get('norm_BB_ai')]:
    #    for i, mat_2 in enumerate([S_blocks.get('S_AB_aa'), S_blocks.get('S_AB_ai')]):#* norms.get('norm_occ_only'), -S_blocks.get('S_AB_aa') @ S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia'), -S_blocks.get('S_AB_ai') @ S_blocks.get('S_BA_ii') @ norms.get('norm_AB_ia')]):
    #        if i == 1:
    #            f_BA_mo = F_blocks.get('f_BA_ia').copy()
    #        else:
    #            f_BA_mo = F_blocks.get('f_BA_aa').copy()#, S_AB_mo @ norms.get('norm_BB_ai')]:
    #            #, S_AB_mo @ norms.get('norm_BB_ai')]:
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_aa')))]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ CIS_matrix_A @ mat_3 @ f_AB_mo @ mat_2 @ CIS_vector_EA_A.T* (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_matrix_A_red').T @ mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_matrix_A_red').T @ mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #            cont_BA_1.append(cont)


    #f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    #f_BA_mo = onel_matfrix_BA
    #int_CSCf_BA_1 = int_CSC_AB_1 @ f_AB_mo
    #int_CSCfS_BA_1 = int_CSCf_BA_1 @ S_BA_mo
    #term_BA_1 = int_CSCfS_BA_1 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)

    #cont_BA_2 = []
    #first_pass = True
    #for j, mat_1 in enumerate([S_blocks.get('S_AB_ii'), -S_blocks.get('S_AB_ai') * norms.get('norm_occ_only')]):
    #    f_BA_mo = F_blocks.get('f_BA_ii').copy()
    #    if j > 0:
    #        f_BA_mo = F_blocks.get('f_BA_ia')#, S_AB_mo @ norms.get('norm_BB_ai')]:
    #    for k, mat_2 in enumerate([S_blocks.get('S_AB_ii'), S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        #, S_AB_mo @ norms.get('norm_BB_ai')]:
    #        if k > 0 and j == 0:
    #            f_BA_mo = f_BA_mo = F_blocks.get('f_BA_ai').copy()
    #        elif k > 0 and j > 0:
    #            f_BA_mo = f_BA_mo = F_blocks.get('f_BA_aa').copy()
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_aa'))),  -norms.get('norm_AB_ia').T @ S_blocks.get('S_BA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_BA_mo @  mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_BA_mo @  mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * norms.get('norm_occ_only')
    #            cont_BA_2.append(cont)

    cont_BB = red_C_s.get('CIS_vector_EA_A_red_left') @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ F_blocks.get('f_BB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    #first_pass = True
    #for j, mat_1 in enumerate([np.identity(len(S_blocks.get('S_BB_ii')))]):
    #    f_BA_mo = F_blocks.get('f_BB_ii').copy()#, S_AB_mo @ norms.get('norm_BB_ai')]:
    #    if j > 1:
    #        f_BA_mo = F_blocks.get('f_BB_ii').copy()
    #    for k, mat_2 in enumerate([S_blocks.get('S_AB_ii')]):#, S_blocks.get('S_AB_ia') * norms.get('norm_occ_only'), S_blocks.get('S_AB_ii')  @ norms.get('norm_BA_ai').T,  -S_blocks.get('S_AB_ia') @ S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]):
    #        #, S_AB_mo @ norms.get('norm_BB_ai')]:
    #        if k > 0:
    #            f_BA_mo = F_blocks.get('f_BB_ai').copy()
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_aa'))),  - S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_BA_mo @  mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_3 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_BA_mo @  mat_1 @ red_C_s.get('CIS_vector_IP_B_red').T * norms.get('norm_occ_only')
    #            cont_BB.append(cont)
#
    #f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    #f_BA_mo = onel_matfrix_BA

    #int_CSf_BA_2 = int_CS_AB_1 @ f_AB_mo
    #int_CSfS_BA_2 = int_CSf_BA_2 @ S_BA_mo
    #int_CSfSC_BA_2 = int_CSfS_BA_2 @ CIS_matrix_A
    #term_BA_2 = int_CSfSC_BA_2 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    # sum
    #term_31_BA = sum(cont_BA_1) - sum(cont_BA_2) #term_BA_1 - term_BA_2
    #term_31_BA_1 = cont_BA_1[0] + cont_BA_1[2] - cont_BA_2_2[0] - cont_BA_2_2[2] #term_BA_1 - term_BA_2
    #results = {
    #    'AB':    term_31_AB_2[0,0],
    #    'BA':    term_31_BA_1[0,0],
    #    'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
    #    
    #}
    
    #CSC_MO_IP_B_EA_A = red_C_s.get('CIS_vector_EA_A_red') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red').T
    #CF_AA_ai = np.trace(red_C_s.get('CIS_matrix_A_red') @ F_blocks.get('f_AA_ai') ) 
    #dc_term = CSC_MO_IP_B_EA_A * CF_AA_ai * (norms.get('norm_occ_only')**2)
    
    dc_prefac = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red_left').T 
    
    cont_dc = []
    for mat in [F_blocks.get('f_AA_ia'), S_blocks.get('S_AB_ii') @ F_blocks.get('f_BA_ia')]:#, S_blocks.get('S_AB_ia') @ F_blocks.get('f_BA_aa')]:
        cont = np.trace(mat @ red_C_s.get('CIS_matrix_A_red_right').T)
        cont_dc.append(dc_prefac * cont * (norms.get('norm_occ_only')**2))
    
    results = {
        'AB':    -cont_AB[0,0],
        'BA':    cont_BA[0,0],# + cont_BA_1[1][0,0],
        'AA':    cont_AA[0,0],
        'BB':    -cont_BB[0,0],# - cont_BB[2][0,0],
        'full':   cont_AA[0,0] - cont_BB[0,0] - cont_AB[0,0] + cont_BA[0,0], # - cont_BB[2][0,0],
        'AA_2':   cont_AA_2[0,0],
        'AA_3':   -cont_AA_3[0,0],
        'AB_2':   cont_AB_2[0,0],
        'full_2':  cont_AA_2[0,0] + cont_AB_2[0,0] - cont_AB[0,0] - cont_AA_3[0,0], # - cont_BB[2][0,0],
        'my_approx': cont_BA[0,0],
        'dc_term':    np.sum(cont_dc)
        #'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
        
    }
    
    return results

def V_13_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_EA_A,
                        CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s, S_inv_blocks):

# AB term 1
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)

    cont_AA_1 = red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_matrix_A_red_left') @ F_blocks.get('f_AA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)

    cont_AB = red_C_s.get('CIS_vector_IP_B_red_right') @ F_blocks.get('f_BA_ii') @ red_C_s.get('CIS_matrix_A_red_left') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)

    cont_BA = red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_matrix_A_red_left') @ S_blocks.get('S_AB_aa') @ F_blocks.get('f_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)

    cont_AA_2 = red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ F_blocks.get('f_AA_ii') @ red_C_s.get('CIS_matrix_A_red_left') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)
    
    dc_prefac = red_C_s.get('CIS_vector_EA_A_red_right') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red_right').T 
    
    cont_dc = []
    for mat in [F_blocks.get('f_AA_ia'), S_blocks.get('S_AB_ii') @ F_blocks.get('f_BA_ia')]:#, S_blocks.get('S_AB_ia') @ F_blocks.get('f_BA_aa')]:
        cont = np.trace(mat @ red_C_s.get('CIS_matrix_A_red_left').T)
        cont_dc.append(dc_prefac * cont * (norms.get('norm_occ_only')**2))
    
    results = {
        'AB':    -cont_AB[0,0],
        'BA':    cont_BA[0,0],
        'AA_1':    cont_AA_1[0,0],
        'AA_2':  - cont_AA_2[0,0], # - cont_BB[2][0,0],
        'full':  cont_AA_1[0,0] - cont_AA_2[0,0] - cont_AB[0,0] + cont_BA[0,0], # - cont_BB[2][0,0],
        'my_approx':  cont_AA_1[0,0] - cont_AA_2[0,0] + cont_BA[0,0], # - cont_BB[2][0,0],
        'dc_term':    np.sum(cont_dc)
        #'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
        
    }
    
    return results

def V_32_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_B, CIS_vector_EA_A,
            CIS_vector_IP_B, LCAO_A, S_AB,LCAO_B, norms, S_blocks, F_blocks, red_C_s):

    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    cont_AB_1 = red_C_s.get('CIS_vector_EA_A_red_left') @ F_blocks.get('f_AB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
#
    #for mat_1 in [np.identity(len(S_blocks.get('S_AA_aa')))]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [np.identity(len(S_blocks.get('S_BB_ii')))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_1 @ F_blocks.get('f_AB_aa') @ mat_2 @ red_C_s.get('CIS_matrix_B_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #            #    first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_AB_1.append(cont)
    
    
    cont_BB_1 = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_aa') @ F_blocks.get('f_BB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
#
    #for mat_1 in [S_blocks.get('S_AB_aa') @ F_blocks.get('f_BB_aa')]:#, S_blocks.get('S_AB_ai') @ F_blocks.get('f_BB_ia')]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [np.identity(len(S_blocks.get('S_BB_ii')))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_1 @ mat_2 @ red_C_s.get('CIS_matrix_B_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #            #    first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_BB_1.append(cont)
    
    cont_BB_2 = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ F_blocks.get('f_BB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    
    #!!negative!!
    #first_pass = True
#
    #for mat_1 in [S_blocks.get('S_AB_aa')]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_BB_ii')))]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [np.identity(len(S_blocks.get('S_BB_ii')))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_1 @ red_C_s.get('CIS_matrix_B_red').T @ mat_2 @ F_blocks.get('f_BB_ii') @ mat_3 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #            #    first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_BB_2.append(cont)
                
    cont_AB_2 = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') @ F_blocks.get('f_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    
    #!!negative!!
    #first_pass = True
#
    #for mat_1 in [S_blocks.get('S_AB_aa')]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [S_blocks.get('S_BA_ii') @ F_blocks.get('f_AB_ii')]:#, S_blocks.get('S_BA_ia') @ F_blocks.get('f_AB_ai')]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [np.identity(len(S_blocks.get('S_BB_ii')))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_A_red') @ mat_1 @ red_C_s.get('CIS_matrix_B_red').T @ mat_2 @ mat_3 @ red_C_s.get('CIS_vector_IP_B_red').T * (norms.get('norm_occ_only')**2)
    #            #    first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_AB_2.append(cont)
    
    dc_prefac = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red_left').T 
    
    cont_dc = []
    for mat in [F_blocks.get('f_BB_ia'), S_blocks.get('S_BA_ii') @ F_blocks.get('f_AB_ia')]:#, S_blocks.get('S_BA_ia') @ F_blocks.get('f_AB_aa')]:
        cont = np.trace(mat @ red_C_s.get('CIS_matrix_B_red_right').T)
        cont_dc.append(dc_prefac * cont * (norms.get('norm_occ_only')**2))
        
    results = {
        'AB_1':    cont_AB_1[0,0],
        'AB_2':  - np.sum(cont_AB_2),
        'BB_1':    np.sum(cont_BB_1),
        'BB_2':  - cont_BB_2[0,0],
        'full': cont_AB_1[0,0] - np.sum(cont_AB_2) + np.sum(cont_BB_1) - cont_BB_2[0,0],
        'my_approx': - np.sum(cont_AB_2) + np.sum(cont_BB_1) - cont_BB_2[0,0],
        'dc_term':    np.sum(cont_dc)
        #'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
        
    }
    return results

def V_23_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_EA_A,
                        CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s, S_inv_blocks):

# AB term 1
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)

    cont_AA = red_C_s.get('CIS_vector_IP_B_red_right') @ red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @ F_blocks.get('f_AA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)
#
    cont_AB = red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ F_blocks.get('f_AB_ii') @ red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)
#
    cont_BA = red_C_s.get('CIS_vector_IP_B_red_right') @ red_C_s.get('CIS_matrix_B_red_left') @ F_blocks.get('f_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)
#
    cont_BB = red_C_s.get('CIS_vector_IP_B_red_right') @ F_blocks.get('f_BB_ii') @ red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)
    
    dc_prefac = red_C_s.get('CIS_vector_EA_A_red_right') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red_right').T 
    
    cont_dc = []
    for mat in [F_blocks.get('f_BB_ia'), S_blocks.get('S_BA_ii') @ F_blocks.get('f_AB_ia')]:#, S_blocks.get('S_AB_ia') @ F_blocks.get('f_BA_aa')]:
        cont = np.trace(mat @ red_C_s.get('CIS_matrix_B_red_left').T)
        cont_dc.append(dc_prefac * cont * (norms.get('norm_occ_only')**2))
    
    results = {
        'AB':    -cont_AB[0,0],
        'BA':    cont_BA[0,0],
        'AA':    cont_AA[0,0],
        'BB':  - cont_BB[0,0],
        'full': cont_AA[0,0] - cont_BB[0,0] - cont_AB[0,0] + cont_BA[0,0],
        'my_approx': -cont_AB[0,0], 
        'dc_term':    np.sum(cont_dc)
        #'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
        
    }
    
    return results

def V_41_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_IP_A,
            CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s):
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    #f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    #f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)    
    # BA term 1
    #int_CC_BA_1 = CIS_vector_IP_A @ CIS_matrix_A
    #int_CCf_BA_1 = int_CC_BA_1 @ f_AB_mo
    #term_1_BA = int_CCf_BA_1 @ CIS_vector_EA_B.T
    
    cont_BA_1 = red_C_s.get('CIS_vector_EA_B_red_left') @ F_blocks.get('f_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    #first_pass = True
#
    #for mat_1 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_AA_aa')))]:# enumerate([np.identity(len(S_blocks.get('S_AA_aa'))), -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        f_BA_mo = F_blocks.get('f_BA_aa').copy()
    #        #if i > 1:
    #        #    f_BA_mo = F_blocks.get('f_BA_ai').copy()
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_BA_1.append(cont)
    
    #BA term 2
    
    #int_Cf_BA_2 = CIS_vector_IP_A @ f_AB_mo
    #int_CfS_BA_2 = int_Cf_BA_2 @ S_BA_mo
    #int_CfSC_BA_2 = int_CfS_BA_2 @ CIS_matrix_A
    #int_CfSCS_BA_2 = int_CfSC_BA_2 @ S_AB_mo

    
    #cont_BA_2 = []
    #first_pass = True
    #for mat_1 in [S_BA_mo]:#, -S_BA_mo @ norms.get('norm_AA_ia')]:
    #    for mat_2 in [S_AB_mo]:#, -S_AB_mo @ norms.get('norm_BB_ai')]:
    #        for mat_3 in [np.identity(len(CIS_matrix_A))]:#, -S_AB_mo @ norms.get('norm_BA_ai')]:#! ??? is it norm_BA_ai
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = CIS_vector_EA_B @ mat_1 @ CIS_matrix_A.T @ mat_2 @ onel_matfrix_BA @ mat_3 @ CIS_vector_IP_A.T   * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = CIS_vector_EA_B @ mat_1 @ CIS_matrix_A.T @ mat_2 @ onel_matfrix_BA @ mat_3 @ CIS_vector_IP_A.T   * (norms.get('norm_occ_only'))
    #            cont_BA_2.append(cont)    
                
    cont_AA_1 = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ F_blocks.get('f_AA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)

    cont_AA_3 = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ F_blocks.get('f_AA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_aa')]:#, -S_BA_mo @ norms.get('norm_AA_ia')]:
    #    f_BA_mo = F_blocks.get('f_AA_ii').copy()
    #    for mat_2 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, -S_AB_mo @ norms.get('norm_BB_ai')]:
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, -S_AB_mo @ norms.get('norm_BA_ai')]:#! ??? is it norm_BA_ai
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T   * (norms.get('norm_occ_only')**2)
    #            #    first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T   * (norms.get('norm_occ_only')**2)
    #            cont_AA.append(cont)   
    
    #V_41_BA_term = sum(cont_BA_1) - sum(cont_BA_2)
    #V_41_BA_term_1 = cont_BA_1[0] - cont_BA_2_2[0]
    
    # AB term 1
    
    #int_CCS_AB_1 = int_CC_BA_1 @ S_AB_mo
    #int_CCSf_AB_1 = int_CCS_AB_1 @ f_BA_mo
    #int_CCSfS_AB_1 = int_CCSf_AB_1 @ S_AB_mo
    #term_1_AB = int_CCSfS_AB_1 @ CIS_vector_EA_B.T
    
    #cont_AB_1 = []
    #first_pass = True
    #for mat_1 in [S_BA_mo]:#, -S_BA_mo @ norms.get('norm_AA_ia')]:
    #    for mat_2 in [S_BA_mo]:#, -S_BA_mo @ norms.get('norm_AA_ia')]:
    #        for mat_3 in [np.identity(len(CIS_matrix_A))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = CIS_vector_EA_B @ mat_1 @ onel_matfrix_AB @ mat_2 @ CIS_matrix_A.T @ mat_3 @ CIS_vector_IP_A.T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = CIS_vector_EA_B @ mat_1 @ onel_matfrix_AB @ mat_2 @ CIS_matrix_A.T @ mat_3 @ CIS_vector_IP_A.T * (norms.get('norm_occ_only'))
    #            cont_AB_1.append(cont)
    
    
    cont_AA_2 = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ F_blocks.get('f_AA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    cont_BB_1 = red_C_s.get('CIS_vector_EA_B_red_left') @ F_blocks.get('f_BB_aa') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    cont_AB_1 = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ F_blocks.get('f_AB_ii') @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_aa') @ F_blocks.get('f_AA_aa').copy()]:#, S_blocks.get('S_BA_ai') @ F_blocks.get('f_AA_ia').copy()]:# enumerate([S_blocks.get('S_BA_ai'), S_blocks.get('S_BA_aa')]):
    #    f_AB_mo = F_blocks.get('f_AA_aa').copy()
    #    #if k == 1:
    #    #    f_AB_mo = F_blocks.get('f_AA_aa').copy()
    #    for mat_2 in [np.identity(len(S_blocks.get('S_AA_aa')))]:
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #                #first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_AB_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #            cont_AA_2.append(cont)
    # BA term 2
    #int_CS_AB_2 = CIS_vector_IP_A @ S_AB_mo
    #int_CSf_AB_2 = int_CS_AB_2 @ f_BA_mo
    #int_CSfC_AB_2 = int_CSf_AB_2 @ CIS_matrix_A
    #int_CSfCS_AB_2 = int_CSfC_AB_2 @ S_AB_mo
    #term_2_AB = int_CSfCS_AB_2 @ CIS_vector_EA_B.T
    
    #cont_AB_2 = []
    #first_pass = True
    #for mat_1 in [S_BA_mo]:#, -S_BA_mo @ norms.get('norm_AA_ia')]:
    #    for mat_2 in [S_BA_mo]:#, -S_BA_mo @ norms.get('norm_AA_ai')]:
    #        for mat_3 in [np.identity(len(CIS_matrix_A))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = CIS_vector_EA_B @ mat_1 @ CIS_matrix_A.T @ mat_3 @ onel_matfrix_AB @ mat_2 @ CIS_vector_IP_A.T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = CIS_vector_EA_B @ mat_1 @ CIS_matrix_A.T @ mat_3 @ onel_matfrix_AB @ mat_2 @ CIS_vector_IP_A.T * (norms.get('norm_occ_only'))
    #            cont_AB_2.append(cont)
    
    cont_BA_2 = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ F_blocks.get('f_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T   * (norms.get('norm_occ_only')**2)
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_aa')]:
    #    for mat_2 in [S_blocks.get('S_AB_ii')]:# enumerate([S_blocks.get('S_AB_ii'), S_blocks.get('S_AB_ia')]):
    #        #, -S_AB_mo @ norms.get('norm_BB_ai')]:
    #        f_BA_mo = F_blocks.get('f_BA_ii').copy()
    #        #if l == 1:
    #        #    f_BA_mo = F_blocks.get('f_BA_ai').copy()
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, -S_AB_mo @ norms.get('norm_BA_ai')]:#! ??? is it norm_BA_ai
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T   * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T   * (norms.get('norm_occ_only')**2)
    #            cont_BA.append(cont)    
    
    # sum
    #V_41_AB_term = sum(cont_AB_1) - sum(cont_AB_2)
    #V_41_AB_term_1 = sum(cont_AB_1_1) - sum(cont_AB_2_2)
    
    dc_prefac = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_ai') @ red_C_s.get('CIS_vector_IP_A_red_left').T
    
    cont_dc = []
    for mat in [F_blocks.get('f_AA_ia'), S_blocks.get('S_AB_ii') @ F_blocks.get('f_BA_ia')]: #, S_blocks.get('S_AB_ia') @ F_blocks.get('f_BA_aa')
        cont = np.trace(mat @ red_C_s.get('CIS_matrix_A_red_right').T)
        cont_dc.append(dc_prefac * cont * (norms.get('norm_occ_only')**2))
    
    
    results = {
        'BA_1':    cont_BA_1[0,0],
        'BA_2': - np.sum(cont_BA_2), #
        'AA_1':   - cont_AA_1[0,0],
        'AA_2':  np.sum(cont_AA_2),
        'full': cont_BA_1[0,0] - np.sum(cont_BA_2) - cont_AA_1[0,0] + np.sum(cont_AA_2),
        'AB_1':    -cont_AB_1[0,0],
        'AA_3':    -cont_AA_3[0,0],
        'BB_1':    cont_BB_1[0,0],
        'full_2': cont_BA_1[0,0] - cont_AB_1[0,0] - cont_AA_3[0,0] + cont_BB_1[0,0],
        'my_approx': - np.sum(cont_BA_2) + np.sum(cont_AA_2) - cont_AA_1[0,0], # - cont_BA_1[0,0],
        'dc_term':    np.sum(cont_dc)
        #'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
        
    }
    #
    #results = {
    #    'AB': V_41_AB_term_1[0,0],
    #    'BA': V_41_BA_term_1[0,0],
    #    'dc_term': dc_term[0,0],
    #    'cross': cont_BA_1[0][0,0] - sum(cont_AB_2_2)[0,0]
    #}
    return results

def V_14_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_IP_A,
            CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s):
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
                    
    cont_AB = red_C_s.get('CIS_vector_IP_A_red_right') @ red_C_s.get('CIS_matrix_A_red_left') @ F_blocks.get('f_AB_aa').copy() @ red_C_s.get('CIS_vector_EA_B_red_right').T * (norms.get('norm_occ_only')**2)

    cont_AA = red_C_s.get('CIS_vector_IP_A_red_right') @ F_blocks.get('f_AA_ii') @ red_C_s.get('CIS_matrix_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T * (norms.get('norm_occ_only')**2)

    cont_BB = red_C_s.get('CIS_vector_IP_A_red_right') @ red_C_s.get('CIS_matrix_A_red_left') @ S_blocks.get('S_AB_aa') @ F_blocks.get('f_BB_aa').copy() @ red_C_s.get('CIS_vector_EA_B_red_right').T * (norms.get('norm_occ_only')**2)
    
    cont_BA = red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @  F_blocks.get('f_BA_ii').copy() @ red_C_s.get('CIS_matrix_A_red_left')  @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T   * (norms.get('norm_occ_only')**2)

    # sum
    #V_41_AB_term = sum(cont_AB_1) - sum(cont_AB_2)
    #V_41_AB_term_1 = sum(cont_AB_1_1) - sum(cont_AB_2_2)
    
    dc_prefac = red_C_s.get('CIS_vector_EA_B_red_right') @ S_blocks.get('S_BA_ai') @ red_C_s.get('CIS_vector_IP_A_red_right').T
    
    cont_dc = []
    for mat in [F_blocks.get('f_AA_ia'), S_blocks.get('S_AB_ii') @ F_blocks.get('f_BA_ia')]: #, S_blocks.get('S_AB_ia') @ F_blocks.get('f_BA_aa')
        cont = np.trace(mat @ red_C_s.get('CIS_matrix_A_red_left').T)
        cont_dc.append(dc_prefac * cont * (norms.get('norm_occ_only')**2))
    
    
    results = {
        'AB':    cont_AB[0,0],
        'BA':    -cont_BA[0,0], #cont_BA_1[0][0,0] 
        'AA':    -cont_AA[0,0],
        'BB':    cont_BB[0,0],
        'full': cont_AB[0,0] - cont_BA[0,0] - cont_AA[0,0] + cont_BB[0,0],
        'my_approx': - cont_BA[0,0], # - cont_AB[0,0],
        'dc_term':    np.sum(cont_dc)
        #'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
        
    }

    return results

def V_42_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_B, CIS_vector_IP_A,
                                           CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s):
    # AB term 1
# AB term 1
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    cont_BA = red_C_s.get('CIS_vector_EA_B_red_left') @ red_C_s.get('CIS_matrix_B_red_right').T @ F_blocks.get('f_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)

    #for mat_1 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_BB_ii')))]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_BA.append(cont)
    
    cont_AB = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ F_blocks.get('f_AB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
#
    #for mat_1 in [S_blocks.get('S_BA_aa') @ F_blocks.get('f_AB_aa')]:#, S_blocks.get('S_BA_ai') @ F_blocks.get('f_AB_ia') ]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [S_blocks.get('S_BA_ii')]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ mat_2 @ red_C_s.get('CIS_matrix_B_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #            #    first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_AB.append(cont)
    #
    cont_BB = red_C_s.get('CIS_vector_EA_B_red_left') @ F_blocks.get('f_BB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
#
    #for mat_1 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [S_blocks.get('S_BA_ii')]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ F_blocks.get('f_BB_aa') @ mat_2 @ red_C_s.get('CIS_matrix_B_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #            #    first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_BB.append(cont)
    
    cont_AA = red_C_s.get('CIS_vector_EA_B_red_left') @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') @ F_blocks.get('f_AA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
#
    #for mat_1 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, S_blocks.get('S_BA_ai') @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [S_blocks.get('S_BA_ii') @ F_blocks.get('f_AA_ii')]:#, S_blocks.get('S_BA_ia') @ F_blocks.get('f_AA_ai')]:#, -S_blocks.get('S_AB_ai') @ norms.get('norm_BA_ai').T, -S_blocks.get('S_AA_ii') @ norms.get('norm_AA_ia'), -S_blocks.get('S_AB_ii') @ norms.get('norm_BA_ai').T]):
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, S_AB_mo @ norms.get('norm_BA_ai')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo  @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_matrix_B_red').T @ mat_2 @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #            #    first_pass = False
    #            #else:
    #            #    cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ f_BA_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T  * (norms.get('norm_occ_only'))
    #            cont_AA.append(cont)
    
    dc_prefac = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_ai') @ red_C_s.get('CIS_vector_IP_A_red_left').T 
    
    cont_dc = []
    for mat in [F_blocks.get('f_BB_ia'), S_blocks.get('S_BA_ii') @ F_blocks.get('f_AB_ia')]:#, S_blocks.get('S_BA_ia') @ F_blocks.get('f_AB_aa')
        cont = np.trace(mat @ red_C_s.get('CIS_matrix_B_red_right').T)
        cont_dc.append(dc_prefac * cont * (norms.get('norm_occ_only')**2))
    
    results = {
        'AB':    np.sum(cont_AB),
        'BA':    -cont_BA[0,0],
        'BB':    np.sum(cont_BB),
        'AA':    -np.sum(cont_AA),
        'full':   -cont_BA[0,0] + np.sum(cont_AB) + np.sum(cont_BB) - np.sum(cont_AA),
        'my_approx':   np.sum(cont_AB),
        'dc_term':    np.sum(cont_dc),
        #'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
        
    }
    
    
    return results

def V_24_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_IP_A,
            CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s):
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
                    
    cont_AB = red_C_s.get('CIS_vector_IP_A_red_right') @ F_blocks.get('f_AB_ii').copy() @ red_C_s.get('CIS_matrix_B_red_left') @ red_C_s.get('CIS_vector_EA_B_red_right').T * (norms.get('norm_occ_only')**2)
#
    cont_BB_1 = red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_matrix_B_red_left') @ F_blocks.get('f_BB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T * (norms.get('norm_occ_only')**2)
#
    cont_BB_2 = red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ F_blocks.get('f_BB_ii').copy() @ red_C_s.get('CIS_matrix_B_red_left') @ red_C_s.get('CIS_vector_EA_B_red_right').T * (norms.get('norm_occ_only')**2)
    #
    cont_BA = red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @  F_blocks.get('f_AB_aa').copy() @ red_C_s.get('CIS_vector_EA_B_red_right').T   * (norms.get('norm_occ_only')**2)

    # sum
    #V_41_AB_term = sum(cont_AB_1) - sum(cont_AB_2)
    #V_41_AB_term_1 = sum(cont_AB_1_1) - sum(cont_AB_2_2)
    
    dc_prefac = red_C_s.get('CIS_vector_EA_B_red_right') @ S_blocks.get('S_BA_ai') @ red_C_s.get('CIS_vector_IP_A_red_right').T
    
    cont_dc = []
    for mat in [F_blocks.get('f_AA_ia'), S_blocks.get('S_AB_ii') @ F_blocks.get('f_BA_ia')]: #, S_blocks.get('S_AB_ia') @ F_blocks.get('f_BA_aa')
        cont = np.trace(mat @ red_C_s.get('CIS_matrix_A_red_left').T)
        cont_dc.append(dc_prefac * cont * (norms.get('norm_occ_only')**2))
    
    
    results = {
        'AB':    -cont_AB[0,0],
        'BA':    cont_BA[0,0], #cont_BA_1[0][0,0] 
        'BB_1':    cont_BB_1[0,0],
        'BB_2':    -cont_BB_2[0,0],
        'full': cont_BA[0,0] + cont_BB_1[0,0] - cont_BB_2[0,0] - cont_AB[0,0],
        'my_approx': cont_BA[0,0] - cont_BB_2[0,0] + cont_BB_1[0,0],
        'dc_term':    np.sum(cont_dc)
        #'cross': cont_BA_1[0][0,0] + cont_BA_1[2][0,0] - cont_AB_2[0][0,0]
        
    }

    return results

def onel_cross_terms_mix(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_matrix_B, CIS_vector_EA_A, CIS_vector_EA_B,
                         CIS_vector_IP_A, CIS_vector_IP_B, LCAO_A, S_AB,
                            LCAO_B, NBAS_A: int, NBAS_B: int, onel_matrix_AA, onel_matrix_BB, normalization, S_blocks, F_blocks, red_C_s, S_inv_blocks):
    # Initializ some stuff
    # Instead of cutting the LCAOs I filled the other stuff up.
    results_31 = V_31_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_EA_A,
                                           CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, normalization, S_blocks, F_blocks, red_C_s, S_inv_blocks)
    results_13 = V_13_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_EA_A,
                                           CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, normalization, S_blocks, F_blocks, red_C_s, S_inv_blocks)
    results_32 = V_32_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_B, CIS_vector_EA_A,
                                           CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, normalization, S_blocks, F_blocks, red_C_s)
    results_23 = V_23_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_B, CIS_vector_EA_A,
                                           CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, normalization, S_blocks, F_blocks, red_C_s, S_inv_blocks)
    results_41 = V_41_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_IP_A,
                                           CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, normalization, S_blocks, F_blocks, red_C_s)
    results_14 = V_14_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_vector_IP_A,
                                           CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, normalization, S_blocks, F_blocks, red_C_s)
    results_42 = V_42_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_B, CIS_vector_IP_A,
                                           CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, normalization, S_blocks, F_blocks, red_C_s)
    results_24 = V_24_onel(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_B, CIS_vector_IP_A,
                                           CIS_vector_EA_B, LCAO_A, S_AB, LCAO_B, normalization, S_blocks, F_blocks, red_C_s)    
    return results_31, results_32, results_41, results_42, results_13, results_23, results_14, results_24


def onel_cross_terms_43(onel_matfrix_AB, onel_matfrix_BA, CIS_vector_EA_A, CIS_vector_EA_B,
                         CIS_vector_IP_A, CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s):
    # Initializ some stuff
    # Instead of cutting the LCAOs I filled the CIS coeffs with zeroes.
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    #NNNorm = red_C_s.get('CIS_vector_IP_B_red') @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red').T * red_C_s.get('CIS_vector_EA_B_red')  @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red').T * (norms.get('norm_occ_only')**2)
    
    #AB 
    #CS_AB_1_1 = CIS_vector_IP_B @ S_BA_mo
    #CSC_AB_1_1 = CS_AB_1_1 @ CIS_vector_IP_A.T
    #
    #Cf_AB_1_2 = CIS_vector_EA_A @ f_AB_mo
    #CfC_AB_1_2 = Cf_AB_1_2 @ CIS_vector_EA_B.T
    #
    #term_AB_1 = CSC_AB_1_1 * CfC_AB_1_2
    
        
    #cont_BA_1 = []
    #first_pass = True
    #for mat_1 in [S_BA_mo]:#np.identity(max(np.shape(CIS_vector_EA_B))), -S_BA_mo @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(max(np.shape(CIS_vector_EA_B))), -S_BA_mo @ norms.get('norm_AB_ia')]:
    #        for mat_3 in [np.identity(max(np.shape(CIS_vector_EA_A))), -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = CIS_vector_IP_B @ mat_1 @ CIS_vector_IP_A.T * CIS_vector_EA_B @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = CIS_vector_IP_B @ mat_1 @ CIS_vector_IP_A.T * CIS_vector_EA_B @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only'))
    #            cont_BA_1.append(cont)
    
    cont_BA_1 = ((red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T) * (red_C_s.get('CIS_vector_EA_B_red_left') @ F_blocks.get('f_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T)) * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_ii')]:
    #    f_BA_mo = F_blocks.get('f_BA_aa')#np.identity(max(np.shape(CIS_vector_EA_B))), -S_BA_mo @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_BB_aa')))]:
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_aa')))]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_IP_B_red') @ mat_1 @ red_C_s.get('CIS_vector_IP_A_red').T * red_C_s.get('CIS_vector_EA_B_red') @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_vector_EA_A_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_IP_B_red') @ mat_1 @ red_C_s.get('CIS_vector_IP_A_red').T * red_C_s.get('CIS_vector_EA_B_red') @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_vector_EA_A_red').T * (norms.get('norm_occ_only'))
    #            cont_BA_1_1.append(cont)
    
    #CS_AB_2_1 = CIS_vector_EA_A @ S_AB_mo
    #CSC_AB_2_1 = CS_AB_2_1 @ CIS_vector_EA_B.T
    #
    #Cf_AB_2_2 = CIS_vector_IP_B @ f_BA_mo
    #CfC_AB_2_2 = Cf_AB_2_2 @ CIS_vector_IP_A.T
    #
    #term_AB_2 = CSC_AB_2_1 * CfC_AB_2_2
    
    #cont_BA_2 = []
    #first_pass = True
    #for mat_1 in [S_BA_mo, -S_BA_mo @ norms.get('norm_AA_ia')]:#np.identity(max(np.shape(CIS_vector_EA_B))), ]:
    #    for mat_2 in [np.identity(max(np.shape(CIS_vector_EA_B)))]:#, -S_BA_mo @ norms.get('norm_AB_ia')]:
    #        for mat_3 in [np.identity(max(np.shape(CIS_vector_EA_A)))]:#, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = CIS_vector_EA_B @ mat_1 @ CIS_vector_EA_A.T  * CIS_vector_IP_B @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_IP_A.T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = CIS_vector_EA_B @ mat_1 @ CIS_vector_EA_A.T  * CIS_vector_IP_B @ mat_2 @ f_BA_mo @ mat_3 @ CIS_vector_IP_A.T * (norms.get('norm_occ_only'))
    #            cont_BA_2.append(cont)
    
    cont_BA_2 = ((red_C_s.get('CIS_vector_EA_B_red_left')  @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T) * (red_C_s.get('CIS_vector_IP_B_red_right') @ F_blocks.get('f_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T)) * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_aa')]:
    #    f_BA_mo = F_blocks.get('f_BA_ii')#np.identity(max(np.shape(CIS_vector_EA_B))), ]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_BB_ii')))]:#, -S_BA_mo @ norms.get('norm_AB_ia')]:
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red')  @ mat_1 @ red_C_s.get('CIS_vector_EA_A_red').T  * red_C_s.get('CIS_vector_IP_B_red') @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_B_red')  @ mat_1 @ red_C_s.get('CIS_vector_EA_A_red').T  * red_C_s.get('CIS_vector_IP_B_red') @ mat_2 @ f_BA_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only'))
    #            cont_BA_2_2.append(cont)
    
    #term_BA_43 = sum(cont_BA_1) - sum(cont_BA_2)
    #term_BA_43_1 = sum(cont_BA_1_1) - sum(cont_BA_2_2)
    
    # BA
    #CSC_BA_1_1 = CSC_AB_1_1
    #
    #CS_BA_1_2 = CIS_vector_EA_A @ S_AB_mo
    #CSf_BA_1_2 = CS_BA_1_2 @ f_BA_mo
    #CSfS_BA_1_2 = CSf_BA_1_2 @ S_AB_mo
    #CSfSC_BA_1_2 = CSfS_BA_1_2 @ CIS_vector_EA_B.T
    #
    #term_BA_1 = CSC_BA_1_1 * CSfSC_BA_1_2
    #
    #CSC_BA_2_1 = CSC_AB_2_1
    #
    #CS_BA_2_2 = CIS_vector_IP_A @ S_AB_mo
    #CSf_BA_2_2 = CS_BA_2_2 @ f_BA_mo
    #CSfS_BA_2_2 = CSf_BA_2_2 @ S_AB_mo
    #CSfSC_BA_2_2 = CSfS_BA_2_2 @ CIS_vector_IP_B.T
    #
    #term_BA_2 = CSC_BA_2_1 * CSfSC_BA_2_2
    
    #cont_AB_1 = []
    #first_pass = True
    #for mat_1 in [S_BA_mo]:#np.identity(max(np.shape(CIS_vector_EA_B))), -S_BA_mo @ norms.get('norm_AB_ia')]:
    #    for mat_2 in [S_BA_mo, -S_BA_mo @ norms.get('norm_AA_ia')]:
    #        for mat_3 in [S_BA_mo, -S_BA_mo @ norms.get('norm_AA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = CIS_vector_IP_B @ mat_1 @ CIS_vector_IP_A.T * CIS_vector_EA_B @ mat_2 @ f_AB_mo @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = CIS_vector_IP_B @ mat_1 @ CIS_vector_IP_A.T * CIS_vector_EA_B @ mat_2 @ f_AB_mo @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only'))
    #            cont_AB_1.append(cont)
    
    cont_AA_1 = ((red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T) * (red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ F_blocks.get('f_AA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T)) * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_ii')]:#np.identity(max(np.shape(CIS_vector_EA_B))), -S_BA_mo @ norms.get('norm_AB_ia')]:
    #    for i, mat_2 in enumerate([S_blocks.get('S_BA_ai'), S_blocks.get('S_BA_aa')]):
    #        f_AB_mo = F_blocks.get('f_AA_ia')
    #        if i > 0:
    #            f_AB_mo = F_blocks.get('f_AA_aa')
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_aa')))]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_IP_B_red') @ mat_1 @ red_C_s.get('CIS_vector_IP_A_red').T * red_C_s.get('CIS_vector_EA_B_red') @ mat_2 @ f_AB_mo @ mat_3 @ red_C_s.get('CIS_vector_EA_A_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_IP_B_red') @ mat_1 @ red_C_s.get('CIS_vector_IP_A_red').T * red_C_s.get('CIS_vector_EA_B_red') @ mat_2 @ f_AB_mo @ mat_3 @ red_C_s.get('CIS_vector_EA_A_red').T * (norms.get('norm_occ_only')**2)
    #            cont_AA_1_1.append(cont)
    
    #CS_AB_2_1 = CIS_vector_EA_A @ S_AB_mo
    #CSC_AB_2_1 = CS_AB_2_1 @ CIS_vector_EA_B.T
    #
    #Cf_AB_2_2 = CIS_vector_IP_B @ f_BA_mo
    #CfC_AB_2_2 = Cf_AB_2_2 @ CIS_vector_IP_A.T
    #
    #term_AB_2 = CSC_AB_2_1 * CfC_AB_2_2
    
    cont_AA_2 = ((red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T)  * (red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ F_blocks.get('f_AA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T)) * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_aa')]:#np.identity(max(np.shape(CIS_vector_EA_B))), ]:
    #    for k, mat_2 in enumerate([S_blocks.get('S_BA_ii'), S_blocks.get('S_BA_ia')]):
    #        f_AB_mo = F_blocks.get('f_AA_ii')
    #        if k > 0:
    #            f_AB_mo = F_blocks.get('f_AA_ai')
    #            #, -S_BA_mo @ norms.get('norm_AB_ia')]:
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:#, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_vector_EA_A_red').T  * red_C_s.get('CIS_vector_IP_B_red') @ mat_2 @ f_AB_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_vector_EA_A_red').T  * red_C_s.get('CIS_vector_IP_B_red') @ mat_2 @ f_AB_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #            cont_AA_2_2.append(cont)
    
    #cont_AB_2 = []
    #first_pass = True
    #for mat_1 in [S_BA_mo, -S_BA_mo @ norms.get('norm_AA_ia')]:#np.identity(max(np.shape(CIS_vector_EA_B))), ]:
    #    for mat_2 in [S_BA_mo]:#, -S_BA_mo @ norms.get('norm_AB_ia')]:
    #        for mat_3 in [S_BA_mo]:#, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont = CIS_vector_EA_B @ mat_1 @ CIS_vector_EA_A.T  * CIS_vector_IP_B @ mat_2 @ f_AB_mo @ mat_3 @ CIS_vector_IP_A.T * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = CIS_vector_EA_B @ mat_1 @ CIS_vector_EA_A.T  * CIS_vector_IP_B @ mat_2 @ f_AB_mo @ mat_3 @ CIS_vector_IP_A.T * (norms.get('norm_occ_only'))
    #            cont_AB_2.append(cont)    
    
    #term_AB_43 = sum(cont_AB_1) - sum(cont_AB_2)
    #term_AA = sum(cont_AA_1_1) - sum(cont_AA_2_2)
    #! have to be rethought for heterodimers
    
    S_BA_MO = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    S_AB_MO = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    f_AB_eff_MO = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_eff_MO = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    # for both
    
    CSC_MO_EA_B_IP_A = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_ai') @ red_C_s.get('CIS_vector_IP_A_red_left').T
    CSC_MO_IP_B_EA_A = red_C_s.get('CIS_vector_EA_A_red_right') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red_right').T
    
    CfC_MO_IP_A_EA_B = red_C_s.get('CIS_vector_EA_B_red_left') @ F_blocks.get('f_BA_ai') @ red_C_s.get('CIS_vector_IP_A_red_left').T
    CfC_MO_EA_A_IP_B = red_C_s.get('CIS_vector_EA_A_red_right') @ F_blocks.get('f_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red_right').T
    
    term_AB = (CSC_MO_EA_B_IP_A * CfC_MO_EA_A_IP_B + CSC_MO_IP_B_EA_A * CfC_MO_IP_A_EA_B) * (norms.get('norm_occ_only')**2)
    
    #CSfC_MO_IP_A_EA_B_aa = red_C_s.get('CIS_vector_IP_A_red') @ S_blocks.get('S_AB_ia') @ F_blocks.get('f_BB_aa') @ red_C_s.get('CIS_vector_EA_B_red').T
    #CSfC_MO_IP_A_EA_B_ia = red_C_s.get('CIS_vector_IP_A_red') @ S_blocks.get('S_AB_ii') @ F_blocks.get('f_BB_ia') @ red_C_s.get('CIS_vector_EA_B_red').T
    CSfC_AB_ai = red_C_s.get('CIS_vector_EA_A_red_right') @ F_blocks.get('f_AA_ai') @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_right').T
    CSfC_BA_ai = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ F_blocks.get('f_AA_ai') @ red_C_s.get('CIS_vector_IP_A_red_left').T
    
    term_BB = (CSC_MO_EA_B_IP_A * CSfC_AB_ai + CSC_MO_IP_B_EA_A * CSfC_BA_ai) * (norms.get('norm_occ_only')**2)
    
    results = {
        'AA_1':     cont_AA_1[0,0],
        'AA_2':    -cont_AA_2[0,0],
        'BA_1':     cont_BA_1[0,0],
        'BA_2':    -cont_BA_2[0,0],
        'full':    cont_BA_1[0,0] - cont_BA_2[0,0] - cont_AA_2[0,0] + cont_AA_1[0,0],
        'my_approx':  cont_AA_1[0,0] - cont_AA_2[0,0],
        'dc_AB':    term_AB[0,0],
        'dc_BB':    term_BB[0,0],
        #'cross': term_BA_43_1[0,0]+term_AB[0,0]
    }    
    
    return results


def onel_cross_terms_34(onel_matfrix_AB, onel_matfrix_BA, CIS_vector_EA_A, CIS_vector_EA_B,
                         CIS_vector_IP_A, CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, norms, S_blocks, F_blocks, red_C_s):
    # Initializ some stuff
    # Instead of cutting the LCAOs I filled the CIS coeffs with zeroes.
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    cont_AB_1 = ((red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T) * (red_C_s.get('CIS_vector_EA_A_red_left') @ F_blocks.get('f_AB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T)) * (norms.get('norm_occ_only')**2)
    
    cont_AB_2 = ((red_C_s.get('CIS_vector_EA_A_red_left')  @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T)  * (red_C_s.get('CIS_vector_IP_A_red_right') @ F_blocks.get('f_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T)) * (norms.get('norm_occ_only')**2)
    
    cont_BB_1 = ((red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T) *  (red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_aa') @ F_blocks.get('f_BB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T)) * (norms.get('norm_occ_only')**2)
    
    cont_BB_2 = ((red_C_s.get('CIS_vector_EA_A_red_left')  @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T) * (red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ F_blocks.get('f_BB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T)) * (norms.get('norm_occ_only')**2)
    
    #! have to be rethought for heterodimers
    
    S_BA_MO = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    S_AB_MO = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    f_AB_eff_MO = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_eff_MO = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    # for both
    
    CSC_1 = red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ia') @ red_C_s.get('CIS_vector_EA_B_red_right').T
    CSC_2 = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red_left').T
    
    CfC_1 = red_C_s.get('CIS_vector_EA_A_red_left') @ F_blocks.get('f_AB_ai') @ red_C_s.get('CIS_vector_IP_B_red_left').T
    CfC_2 = red_C_s.get('CIS_vector_IP_A_red_right') @ F_blocks.get('f_AB_ia') @ red_C_s.get('CIS_vector_EA_B_red_right').T
    
    term_AB = (CSC_1 * CfC_1 + CSC_2 * CfC_2) * (norms.get('norm_occ_only')**2)
    
    #CSfC_MO_IP_A_EA_B_aa = red_C_s.get('CIS_vector_IP_A_red') @ S_blocks.get('S_AB_ia') @ F_blocks.get('f_BB_aa') @ red_C_s.get('CIS_vector_EA_B_red').T
    #CSfC_MO_IP_A_EA_B_ia = red_C_s.get('CIS_vector_IP_A_red') @ S_blocks.get('S_AB_ii') @ F_blocks.get('f_BB_ia') @ red_C_s.get('CIS_vector_EA_B_red').T
    CSfC_1 = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_aa') @ F_blocks.get('f_BB_ai') @ red_C_s.get('CIS_vector_IP_B_red_left').T
    CSfC_2 = red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ F_blocks.get('f_BB_ia') @ red_C_s.get('CIS_vector_EA_B_red_right').T
    
    term_BB = (CSC_1 * CSfC_1 + CSC_2 * CSfC_2) * (norms.get('norm_occ_only')**2)
    
    results = {
        'AA_1':     cont_BB_1[0,0],
        'AA_2':    -cont_BB_2[0,0],
        'BA_1':     cont_AB_1[0,0],
        'BA_2':    -cont_AB_2[0,0],
        'full':    cont_AB_1[0,0] - cont_AB_2[0,0] - cont_BB_2[0,0] + cont_BB_1[0,0],
        'my_approx':  cont_BB_1[0,0] - cont_BB_2[0,0],
        'dc_AB':    term_AB[0,0],
        'dc_BB':    term_BB[0,0],
        #'cross': term_BA_43_1[0,0]+term_AB[0,0]
    }    
    
    return results


def onel_cross_terms_21(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_matrix_B, LCAO_A, S_AB,
                            LCAO_B, NBAS_A: int, NBAS_B: int, norms, S_blocks, F_blocks, red_C_s):
    # Initializ some stuff
    # Instead of cutting the LCAOs I filled the other stuff up.
    
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    AB_term = np.float64(0.)
    term_1_S = np.float64(0.)
    BA_term = np.float64(0.)
    # Get the final matrices for first term
    # BA
    
    cont_AB = red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ F_blocks.get('f_AB_ii').copy() * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_aa')]:#np.identity(max(np.shape(CIS_vector_EA_B))), ]:
    #    f_AB_mo = F_blocks.get('f_AB_ii').copy()
    #    for mat_2 in [np.identity(len(S_blocks.get('S_AA_ii')))]:
    #        for mat_3 in [np.identity(len(S_blocks.get('S_BB_ii')))]:#, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont_AB = red_C_s.get('CIS_matrix_B_red') @ mat_1 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_AB_mo @ mat_3 * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_vector_EA_A_red').T  * red_C_s.get('CIS_vector_IP_B_red') @ mat_2 @ f_AB_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #            cont_AB_1.append(cont_AB)
        
    cont_BA = red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_matrix_B_red_left') @ F_blocks.get('f_BA_aa').copy() * (norms.get('norm_occ_only')**2)
    #cont_AB_2 = red_C_s.get('CIS_matrix_B_red').T @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_matrix_A_red') @ F_blocks.get('f_AB_aa').copy() * (norms.get('norm_occ_only')**2)
    #
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_ii')]:#np.identity(max(np.shape(CIS_vector_EA_B))), ]:
    #    f_BA_mo = F_blocks.get('f_AB_aa').copy()
    #    for mat_2 in [np.identity(len(S_blocks.get('S_AA_aa')))]:
    #        for mat_3 in [np.identity(len(S_blocks.get('S_BB_aa')))]:#, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            if first_pass:
    #                #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #                cont_BA = red_C_s.get('CIS_matrix_A_red').T @ mat_1 @ red_C_s.get('CIS_matrix_B_red') @ mat_2 @ f_BA_mo @ mat_3 * (norms.get('norm_occ_only')**2)
    #                first_pass = False
    #            else:
    #                cont = red_C_s.get('CIS_vector_EA_B_red') @ mat_1 @ red_C_s.get('CIS_vector_EA_A_red').T  * red_C_s.get('CIS_vector_IP_B_red') @ mat_2 @ f_AB_mo @ mat_3 @ red_C_s.get('CIS_vector_IP_A_red').T * (norms.get('norm_occ_only')**2)
    #            cont_BA_1.append(cont_BA)
    
    term_AB = np.float64(0.)    #!min!!!
    for i_A in range(len(S_blocks.get('S_BB_ii'))):
        term_AB += cont_AB[i_A, i_A]
    
    term_BA = np.float64(0.)    
    for a_A in range(len(S_blocks.get('S_AA_aa'))):
        term_BA += cont_BA[a_A, a_A]
    
    #term_BA_2 = np.float64(0.)    
    #for a_A in range(len(S_blocks.get('S_BB_aa'))):
    #    term_BA_2 += cont_AB_2[a_A, a_A]
    
    cont_AA_aa =  S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @ F_blocks.get('f_AA_aa').copy() @ red_C_s.get('CIS_matrix_A_red_right').T * (norms.get('norm_occ_only')**2)
    #i_A i_A
    
    #! +
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_ii')]:#np.identity(max(np.shape(CIS_vector_EA_B))), ]:
    #    for mat_2 in [np.identity(len(S_blocks.get('S_AA_aa')))]:
    #        for j, mat_3 in enumerate([S_blocks.get('S_BA_ai'), S_blocks.get('S_BA_aa')]):
    #            f_AB_mo = F_blocks.get('f_AA_ia').copy()#, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            if j > 0:
    #                f_AB_mo = F_blocks.get('f_AA_aa').copy() #, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #            cont =  mat_3 @ f_AB_mo @ mat_2 @ red_C_s.get('CIS_matrix_A_red').T @ mat_1 @ red_C_s.get('CIS_matrix_B_red') * (norms.get('norm_occ_only')**2)
    #            cont_AA_aa.append(cont)
    
    #!-
    cont_BB_ii = red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ F_blocks.get('f_BB_ii').copy() * (norms.get('norm_occ_only')**2)
    
    #first_pass = True
    #for mat_1 in [S_blocks.get('S_BA_aa')]:#np.identity(max(np.shape(CIS_vector_EA_B))), ]:
    #    for k, mat_2 in enumerate([S_blocks.get('S_AB_ia'), S_blocks.get('S_AB_ii')]):
    #        f_AB_mo = F_blocks.get('f_BB_ai').copy()#, -S_AB_mo @ norms.get('norm_BA_ia')]:
    #        if k > 0:
    #            f_AB_mo = F_blocks.get('f_BB_ii').copy() #
    #        for mat_3 in [np.identity(len(S_blocks.get('S_AA_ii')))]:# -S_AB_mo @ norms.get('norm_BA_ia')]:
    #            #cont = CIS_vector_IP_B @ mat_1 @ f_AB_mo @ mat_2 @ CIS_matrix_A @ mat_3 @ CIS_vector_EA_A.T * (norms.get('norm_occ_only')**2)
    #            cont = red_C_s.get('CIS_matrix_B_red') @ mat_1 @ red_C_s.get('CIS_matrix_A_red').T @ mat_2 @ f_AB_mo @ mat_3 * (norms.get('norm_occ_only')**2)
    #            cont_BB_ii.append(cont)
    
    term_AA_ia = np.float64(0.)    #!+++
    term_AA_aa = np.float64(0.)    #!+++
    for b_B in range(len(S_blocks.get('S_AA_ii'))):
        #term_AA_ia += cont_AA_aa[0][b_B, b_B]
        term_AA_aa += cont_AA_aa[b_B, b_B]
    
    term_BB_ai = np.float64(0.)    
    term_BB_ii = np.float64(0.)    
    for j_B in range(len(S_blocks.get('S_BB_ii'))):
        #term_BB_ai += cont_BB_ii[0][j_B, j_B]
        term_BB_ii += cont_BB_ii[j_B, j_B]
    
    #SC_BA_1 = S_AB_mo @ CIS_matrix_B
    #SCf_BA_1 = SC_BA_1 @ onel_matfrix_BA
    #SCfC_BA_1 = SCf_BA_1 @ CIS_matrix_A.T
    #
    #CS_BA_2 = CIS_matrix_A @ S_AB_mo
    #CSC_BA_2 = CS_BA_2 @ CIS_matrix_B.T
    #CSCS_BA_2 = CSC_BA_2 @ S_BA_mo
    #CSCSf_BA_2 = CSCS_BA_2 @ onel_matfrix_AB
    #CSCSfS_BA_2 = CSCSf_BA_2 @ S_BA_mo
#
    ## AB term 
    #CS_AB_1 = CIS_matrix_A @ S_AB_mo
    #CSC_AB_1 = CS_AB_1 @ CIS_matrix_B.T
    #CSCf_AB_1 = CSC_AB_1 @ onel_matfrix_BA
    #
    #CS_AB_2 = CIS_matrix_A @ S_AB_mo
    #CSf_AB_2 = CS_AB_2 @ onel_matfrix_BA
    #CSfS_AB_2 = CSf_AB_2 @ S_AB_mo
    #CSfSC_AB_2 = CSfS_AB_2 @ CIS_matrix_B.T
    #CSfSCS_AB_2 = CSfSC_AB_2 @ S_BA_mo
    #
    #AAAAAAAAAAAA  = 0.
    #for i_A in range(NBAS_A):
    #    AAAAAAAAAAAA += CSCf_AB_1[i_A, i_A] 
    #    term_1_S += SCfC_BA_1[i_A, i_A] - CSCf_AB_1[i_A, i_A] 
    #    BA_term += (SCfC_BA_1[i_A, i_A]- CSCSfS_BA_2[i_A, i_A] )
    #    AB_term += (CSfSCS_AB_2[i_A, i_A] - CSCf_AB_1[i_A, i_A])
        
    results = {
        'BA': term_BA,
        'AB': -term_AB,
        'AA': term_AA_aa,
        'BB': -term_BB_ii,
        'full': term_BA - term_AB + term_AA_aa - term_BB_ii,
        'my_approx': term_AA_aa- term_BB_ii,
        'no_Ov': term_1_S
    }
    
    return results

def onel_cross_terms_12(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_matrix_B, LCAO_A, S_AB,
                            LCAO_B, NBAS_A: int, NBAS_B: int, norms, S_blocks, F_blocks, red_C_s):
    # Initializ some stuff
    # Instead of cutting the LCAOs I filled the other stuff up.
    
    S_AB_mo = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    S_BA_mo = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    f_AB_mo = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_mo = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    AB_term = np.float64(0.)
    term_1_S = np.float64(0.)
    BA_term = np.float64(0.)
    # Get the final matrices for first term
    # BA
    
    cont_BA = red_C_s.get('CIS_matrix_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ F_blocks.get('f_BA_ii').copy() * (norms.get('norm_occ_only')**2)
    
    cont_AB = red_C_s.get('CIS_matrix_A_red_left') @ F_blocks.get('f_AB_aa').copy() @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') * (norms.get('norm_occ_only')**2)
    
    term_BA = np.float64(0.)    #!min!!!
    term_AB = np.float64(0.)    
    for i_A in range(len(S_blocks.get('S_AA_ii'))):
        term_BA += cont_BA[i_A, i_A]
        term_AB += cont_AB[i_A, i_A]
    
    #term_AB = np.float64(0.)    
    #for a_A in range(len(S_blocks.get('S_AA_aa'))):
        
    
    
    cont_BB_aa = red_C_s.get('CIS_matrix_A_red_left') @ S_blocks.get('S_AB_aa') @ F_blocks.get('f_BB_aa').copy() @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') * (norms.get('norm_occ_only')**2)
    
    #!-
    cont_AA_ii = red_C_s.get('CIS_matrix_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') @ F_blocks.get('f_AA_ii').copy() * (norms.get('norm_occ_only')**2)
    
    #term_AA_ia = np.float64(0.)    #!+++
    term_BB_aa = np.float64(0.)    
    term_AA_ii = np.float64(0.)    #!+++
    for i_A in range(len(S_blocks.get('S_AA_ii'))):
        term_AA_ii += cont_AA_ii[i_A, i_A]
        term_BB_aa += cont_BB_aa[i_A, i_A]

        
    results = {
        'BA': -term_BA,
        'AB': term_AB,
        'AA': -term_AA_ii,
        'BB': term_BB_aa,
        'full': - term_BA + term_AB + term_BB_aa - term_AA_ii,
        'my_approx': term_BB_aa- term_AA_ii,
        'no_Ov': term_1_S
    }
    
    return results


#f_AB: AO, S_INV_AO, ene_X: AO
# RESULT: AO
def get_eff_fock(f_AB, f_BA, ene_A, ene_B, S_inv_AO_AB, S_inv_AO_BA, LCAO_A, LCAO_B):
    
    # f_AB_effective in AO
    
    term_AB_2 = ene_A @ S_inv_AO_AB
    
    term_AB_3 = S_inv_AO_AB @ ene_B
    
    Sf_term_AB_4 = S_inv_AO_AB @ f_BA 
    term_AB_4 = Sf_term_AB_4 @ S_inv_AO_AB
    
    f_AB_AO_eff = f_AB #+ term_AB_4 + term_AB_2 + term_AB_3
    
    # f_BA_effective in AO
    
    term_BA_2 = S_inv_AO_BA @ ene_A 
    
    term_BA_3 = ene_B @ S_inv_AO_BA
    
    Sf_term_BA_4 = S_inv_AO_BA @ f_AB
    term_BA_4 = Sf_term_BA_4 @ S_inv_AO_BA
    
    f_BA_AO_eff = f_BA #+ term_BA_2 + term_BA_3 + term_BA_4
    
    return f_AB_AO_eff, f_BA_AO_eff


def V_21_discon(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_matrix_B,
                    LCAO_A, S_AB, LCAO_B, NBAS_A, NBAS_B):
    
    term_1 = np.float64(0.)
    term_2 = np.float64(0.)
    
    # term_1
    AS_1 = LCAO_A.T @ S_AB
    ASA_1 = AS_1 @ LCAO_B
    ASAC_1 = ASA_1 @ CIS_matrix_B.T
    ASACA_1 = ASAC_1 @ LCAO_B.T
    ASACAS_1 = ASACA_1 @ S_AB.T
    ASACASA_1 = ASACAS_1 @ LCAO_A
    ASACASAC_1 = ASACASA_1 @ CIS_matrix_A
    
    # term_2
    
    Af_1 = LCAO_A.T @ onel_matfrix_AB
    AfA_1 = Af_1 @ LCAO_B
    AfAA_1 = AfA_1 @ LCAO_B.T
    AfAAS_1 = AfAA_1 @ S_AB.T
    AfAASA_1 = AfAAS_1 @ LCAO_A
        
    
    
    for a_A in range(NBAS_A):
        term_1 += (ASACASAC_1[a_A, a_A] )
        term_2 += (AfAASA_1[a_A, a_A] )
    
    V_21_AB_discon = term_1 * term_2
    
    return V_21_AB_discon


# coewff dimensions: AO (row) MO  (column)
def Ao_to_MO_trafo(left_coeff, AO_matrix, right_coeff):
    intermediate = left_coeff.T  @ AO_matrix
    MO_matrix = intermediate @ right_coeff
    return MO_matrix
    

def state_overlap(red_C_s, S_blocks, norms):
    S_21_term = np.float64(0.)
    S_12_term = np.float64(0.)
    
    SCSC_21 = red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') 
    SCSC_12 = red_C_s.get('CIS_matrix_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') 
    
    
    for kdx in range(SCSC_21.shape[0]):
        S_21_term += SCSC_21[kdx, kdx]
    for kdx in range(SCSC_12.shape[0]):
        S_12_term += SCSC_12[kdx, kdx]    
    S_21_final_term = S_21_term * (norms.get('norm_occ_only')**2)
    S_12_final_term = S_12_term * (norms.get('norm_occ_only')**2)
    
    S_31_AB_term = red_C_s.get('CIS_vector_EA_A_red_left') @ red_C_s.get('CIS_matrix_A_red_right').T @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)  
    S_13_AB_term = red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_matrix_A_red_left') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)  
    
    
    S_41_AB_term = red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_matrix_A_red_right').T @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)
    S_14_AB_term = red_C_s.get('CIS_vector_IP_A_red_right') @ red_C_s.get('CIS_matrix_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T * (norms.get('norm_occ_only')**2)
    
    #asd = 0.
    #non_zero_terms_1 = []
    #for d in range(np.max(np.shape(red_C_s.get('CIS_vector_EA_B_red')))):
    #    for l in range(np.max(np.shape(red_C_s.get('CIS_vector_IP_A_red')))):
    #        for a in range(np.max(np.shape(red_C_s.get('CIS_vector_EA_A_red')))):
    #            term = red_C_s.get('CIS_vector_EA_B_red')[:, d] * S_blocks.get('S_full')[d+38+8, a+8] * red_C_s.get('CIS_matrix_A_red')[l,a] * red_C_s.get('CIS_vector_IP_A_red')[:, l] * (norms.get('norm_occ_only')**2)
    #            asd += term
    #            if np.abs(term) > 1e-10:
    #                non_zero_terms_1.append([term, d+38+8, a+8, l, S_blocks.get('S_full')[d+38+8, a+8], red_C_s.get('CIS_matrix_A_red')[l,a]]) 
                    
    
    #S_32_AB_term_inter = red_C_s.get('CIS_vector_EA_A_red') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_matrix_B_red').T
    S_32_AB_term = red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_matrix_B_red_right').T @ red_C_s.get('CIS_vector_IP_B_red_left').T * (norms.get('norm_occ_only')**2)
    #S_32_AB_term_3 = red_C_s.get('CIS_vector_IP_B_red') @ red_C_s.get('CIS_matrix_B_red') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red') * (norms.get('norm_occ_only')**2)
    #S_32_AB_term_2 = red_C_s.get('CIS_vector_IP_B_red') @ red_C_s.get('CIS_matrix_B_red') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red').T * (norms.get('norm_occ_only')**2)
    S_23_AB_term = red_C_s.get('CIS_vector_IP_B_red_right') @ red_C_s.get('CIS_matrix_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T * (norms.get('norm_occ_only')**2)
    
    #asd_2 = 0.
    #non_zero_terms_2 = []
    #for c in range(np.max(np.shape(red_C_s.get('CIS_vector_EA_A_red')))):
    #    for k in range(np.max(np.shape(red_C_s.get('CIS_vector_IP_B_red')))):
    #        for b in range(np.max(np.shape(red_C_s.get('CIS_vector_EA_B_red')))):
    #            term = red_C_s.get('CIS_vector_EA_A_red')[:, c] * S_blocks.get('S_full')[c+8, b+38+8] * red_C_s.get('CIS_matrix_B_red')[k,b] * red_C_s.get('CIS_vector_IP_B_red')[:, k] * (norms.get('norm_occ_only')**2)
    #            asd_2 += term
    #            if np.abs(term) > 1e-10:
    #                non_zero_terms_2.append([term, c+8, b+38+8, k+38, S_blocks.get('S_full')[c+8, b+38+8], red_C_s.get('CIS_matrix_B_red')[k,b]])
    
    S_42_AB_term = red_C_s.get('CIS_vector_EA_B_red_left') @ red_C_s.get('CIS_matrix_B_red_right').T @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T * (norms.get('norm_occ_only')**2)  
    S_24_AB_term = red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_matrix_B_red_left') @ red_C_s.get('CIS_vector_EA_B_red_right').T * (norms.get('norm_occ_only')**2)  
            
    S_43_term = (red_C_s.get('CIS_vector_EA_B_red_left') @ S_blocks.get('S_BA_aa') @ red_C_s.get('CIS_vector_EA_A_red_right').T) * ( red_C_s.get('CIS_vector_IP_B_red_right') @ S_blocks.get('S_BA_ii') @ red_C_s.get('CIS_vector_IP_A_red_left').T) * (norms.get('norm_occ_only')**2)
    S_34_term = (red_C_s.get('CIS_vector_EA_A_red_left') @ S_blocks.get('S_AB_aa') @ red_C_s.get('CIS_vector_EA_B_red_right').T) * ( red_C_s.get('CIS_vector_IP_A_red_right') @ S_blocks.get('S_AB_ii') @ red_C_s.get('CIS_vector_IP_B_red_left').T) * (norms.get('norm_occ_only')**2)

    overlaps = {
        12: S_12_final_term,
        21: S_21_final_term,
        13: S_13_AB_term[0, 0],
        31: S_31_AB_term[0, 0],
        14: S_14_AB_term[0, 0],
        41: S_41_AB_term[0, 0],
        23: S_23_AB_term[0, 0],
        32: S_32_AB_term[0, 0],
        24: S_24_AB_term[0, 0],
        42: S_42_AB_term[0, 0],
        34: S_34_term[0, 0],
        43: S_43_term[0, 0]
    }
    
    return overlaps


def diag_results_check(f_AA, CIS_matrix_A, LCAO_A, f_AA_cfour, N_occ_A):
    
    f_AA_MO_TM = Ao_to_MO_trafo(LCAO_A, f_AA, LCAO_A)
    
    fC_AO_1 = CIS_matrix_A @ f_AA_MO_TM
    term_1 = fC_AO_1 @ CIS_matrix_A.T
    
    res_1 = np.trace(term_1)
    
    fC_AO_2 = CIS_matrix_A.T @ f_AA_MO_TM
    term_2 = fC_AO_2 @ CIS_matrix_A
    
    res_2 = np.trace(term_2)
    
    fin_res = res_1 - res_2
    
    CC_1 = CIS_matrix_A @ CIS_matrix_A.T
    
    dc_1 = np.trace(f_AA_MO_TM)
    
    dc_term_Ene = np.float64(0.)
    
    for i in range(N_occ_A):
        dc_term_Ene += np.float64(2) * f_AA_MO_TM[i,i] 
    
    f_AA_MO_C4 = Ao_to_MO_trafo(LCAO_A, f_AA_cfour, LCAO_A)
    
    fC_AO_11 = CIS_matrix_A @ f_AA_MO_C4
    term_11 = fC_AO_11 @ CIS_matrix_A.T
    
    res_11 = np.trace(term_11)
    
    fC_AO_22 = CIS_matrix_A.T @ f_AA_MO_C4
    term_22 = fC_AO_22 @ CIS_matrix_A
    
    res_22 = np.trace(term_22)
    
    fin_res_2 = res_11 - res_22
    
    dc_2 = np.trace(f_AA_MO_C4)
    
    dc_term_2 = np.trace(CC_1) * dc_2
    
    fin_res_with_dc_2 = fin_res_2 + dc_term_2
    
    return fin_res, dc_term_Ene, fin_res_2, fin_res_with_dc_2


def V_31_discon(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_matrix_B, CIS_vector_EA_A, CIS_vector_EA_B,
                        CIS_vector_IP_A, CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, NBAS_A, NBAS_B):
    term_trace_1 = np.float64(0.)
    term_trace_2 = np.float64(0.)
    term_trace_3 = np.float64(0.)
    
    KKKKKKK = CIS_vector_EA_A @ CIS_vector_EA_A.T
    LLLLLLL = CIS_vector_IP_A @ CIS_vector_IP_A.T
    
    S_BA_MO = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    S_AB_MO = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    f_AB_eff_MO = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_eff_MO = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    CS_MO_both_term_1 = CIS_vector_IP_B @ S_BA_MO
    CSC_MO_both_term_1 = CS_MO_both_term_1 @ CIS_matrix_A
    CSCC_MO_both_term_1 = CSC_MO_both_term_1 @ CIS_vector_EA_A.T
    
    fS_both = f_AB_eff_MO @ S_BA_MO
    
    for a_A in range(NBAS_A):
        term_trace_1 += fS_both[a_A, a_A]
        
    dc_both_term_1 = term_trace_1 * CSCC_MO_both_term_1
    
    CS_both_term_2 = CIS_vector_IP_B @ S_BA_MO
    CSC_both_term_2 = CS_both_term_2 @ CIS_vector_EA_A.T
    
    fS_AB_term_2 = f_AB_eff_MO @ S_BA_MO
    fSC_AB_term_2 = fS_AB_term_2 @ CIS_matrix_A.T
    
    for a_A in range(NBAS_A):
        term_trace_2 += fSC_AB_term_2[a_A, a_A]
    
    SF_BA_term_2 = S_AB_MO @ f_BA_eff_MO
    SFC_BA_term_2 = SF_BA_term_2 @ CIS_matrix_A.T
    
    for a_A in range(NBAS_A):
        term_trace_3 += SFC_BA_term_2[a_A, a_A]
        
    dc_AB_term_2 = CSC_both_term_2 * term_trace_2
    dc_BA_term_2 = CSC_both_term_2 * term_trace_3
    
    results = {
        'both': dc_both_term_1[0,0],
        'AB'  : dc_AB_term_2[0,0],
        'BA'  : dc_BA_term_2[0,0]
    }
    
    return results


def V_41_discon(onel_matfrix_AB, onel_matfrix_BA, CIS_matrix_A, CIS_matrix_B, CIS_vector_EA_A, CIS_vector_EA_B,
                        CIS_vector_IP_A, CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, NBAS_A, NBAS_B):
    term_trace_1 = np.float64(0.)
    term_trace_2 = np.float64(0.)
    term_trace_3 = np.float64(0.)
    
    if np.shape(CIS_matrix_A) != np.shape(CIS_matrix_B):
        sys.exit('not implemented for heterodimers')
    #! have to be rethought for heterodimers
    
    S_BA_MO = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    S_AB_MO = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    f_AB_eff_MO = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_eff_MO = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    CS_MO_both_term_1 = CIS_vector_EA_B @ S_BA_MO
    CSC_MO_both_term_1 = CS_MO_both_term_1 @ CIS_matrix_A.T
    CSCC_MO_both_term_1 = CSC_MO_both_term_1 @ CIS_vector_IP_A.T
    
    fS_both = f_AB_eff_MO @ S_BA_MO
    
    for a_A in range(NBAS_A):
        term_trace_1 += fS_both[a_A, a_A]
        
    dc_both_term_1 = term_trace_1 * CSCC_MO_both_term_1
    
    CS_both_term_2 = CIS_vector_IP_A @ S_AB_MO
    CSC_both_term_2 = CS_both_term_2 @ CIS_vector_EA_B.T
    
    fS_AB_term_2 = f_AB_eff_MO @ S_BA_MO
    fSC_AB_term_2 = fS_AB_term_2 @ CIS_matrix_A.T
    
    for i_A in range(NBAS_A):
        term_trace_2 += fSC_AB_term_2[i_A, i_A]
    
    SF_BA_term_2 = S_AB_MO @ f_BA_eff_MO
    SFC_BA_term_2 = SF_BA_term_2 @ CIS_matrix_A.T
    
    for i_A in range(NBAS_A):
        term_trace_3 += SFC_BA_term_2[i_A, i_A]
        
    dc_AB_term_2 = CSC_both_term_2 * term_trace_2
    dc_BA_term_2 = CSC_both_term_2 * term_trace_3
    
    results = {
        'both': dc_both_term_1[0,0],
        'AB'  : dc_AB_term_2[0,0],
        'BA'  : dc_BA_term_2[0,0]
    }
    
    return results


def V_43_discon(onel_matfrix_AB, onel_matfrix_BA, CIS_vector_EA_A, CIS_vector_EA_B,
                        CIS_vector_IP_A, CIS_vector_IP_B, LCAO_A, S_AB, LCAO_B, NBAS_A, NBAS_B):
    term_trace_1 = np.float64(0.)
    #! have to be rethought for heterodimers
    
    S_BA_MO = Ao_to_MO_trafo(LCAO_B, S_AB.T, LCAO_A)
    S_AB_MO = Ao_to_MO_trafo(LCAO_A, S_AB, LCAO_B)
    f_AB_eff_MO = onel_matfrix_AB #Ao_to_MO_trafo(LCAO_A, onel_matfrix_AB, LCAO_B)
    f_BA_eff_MO = onel_matfrix_BA #Ao_to_MO_trafo(LCAO_B, onel_matfrix_BA, LCAO_A)
    
    # for both
    
    CS_MO_EA = CIS_vector_EA_B @ S_BA_MO
    CSC_MO_EA_EA = CS_MO_EA @ CIS_vector_EA_A.T
    
    CSC_MO_EA_B_IP_A =  CS_MO_EA @ CIS_vector_IP_A.T
    
    CS_MO_IP = CIS_vector_IP_B @ S_BA_MO
    CS_MO_IP_IP = CS_MO_IP @ CIS_vector_IP_A.T
    
    CSC_MO_EA_A_IP_B = CS_MO_IP @ CIS_vector_EA_A.T
    
    fS_both = f_AB_eff_MO @ S_BA_MO
    SFS_BA = S_BA_MO @ fS_both
    
    for a_A in range(NBAS_A):
        term_trace_1 += fS_both[a_A, a_A]
    
    #AB terms
    term_both_1 = CSC_MO_EA_A_IP_B[0,0] * CSC_MO_EA_B_IP_A[0,0] * term_trace_1

    CSFS_AB_1 = CIS_vector_IP_B @ SFS_BA
    CSFSC_AB_1 = CSFS_AB_1 @ CIS_vector_EA_A.T
    term_AB_2 = CSC_MO_EA_B_IP_A[0,0] * CSFSC_AB_1[0,0]

    CSFS_AB_2 = CIS_vector_EA_B @ SFS_BA
    CSFSC_AB_2 = CSFS_AB_2 @ CIS_vector_IP_A.T
    term_AB_3 = CSC_MO_EA_A_IP_B[0,0] * CSFSC_AB_2[0,0]

    term_both_2 = CSC_MO_EA_EA[0,0] * CS_MO_IP_IP[0,0] * term_trace_1
    #! ITT tartok!!!!!!
    
    # BA terms
    
    CF_BA_1 = CIS_vector_IP_B @ f_BA_eff_MO
    CFC_BA_1 = CF_BA_1 @ CIS_vector_EA_A.T
    term_BA_2 = CSC_MO_EA_B_IP_A[0,0] * CFC_BA_1[0,0]

    CF_BA_2 = CIS_vector_EA_B @ f_BA_eff_MO
    CFC_BA_2 = CF_BA_2 @ CIS_vector_IP_A.T
    term_BA_3 = CSC_MO_EA_A_IP_B[0,0] * CFC_BA_2[0,0]
    
    results = {
        'both': term_both_1 + term_both_2,
        'AB'  : term_AB_2 + term_AB_3,
        'BA'  : term_BA_2 + term_BA_3
    }
    
    return results


def onel_cross_terms_eff(f_AB, f_BA, CIS_matrix_A_right, CIS_matrix_B_left, 
                        CIS_matrix_B_right, CIS_vector_EA_A_left, CIS_vector_EA_B_left, CIS_vector_IP_A_left,
                        CIS_vector_IP_B_left, CIS_vector_EA_B_right, CIS_vector_IP_A_right, 
                        f_AA, f_BB, LCAO_A, S_AB, LCAO_B, Nbas_A, Nbas_B, ene_A, ene_B, S_AB_inv, S_MO_inv, N_occ_A, N_occ_B, normalization, fock_blocks, S_blocks, S_inv_blocks, red_C_s):
    
    
    #CIS_matrix_A_right_2 = CIS_matrix_A_right @ S_MO_inv[:Nbas_A, :Nbas_A]
    #CIS_matrix_B_left_2 = CIS_matrix_B_left @ S_MO_inv[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)]
    ##CIS_vector_EA_A_left_2[N_occ_A::].T =  
    ##CIS_vector_EA_B_left_2[N_occ_B::].T =  
    #CIS_vector_IP_A_left_2 = S_MO_inv[:Nbas_A, :Nbas_A] @ CIS_vector_IP_A_left 
    #CIS_vector_IP_B_left_2 = S_MO_inv[Nbas_A:(Nbas_A + Nbas_B), Nbas_A:(Nbas_A + Nbas_B)] @ CIS_vector_IP_B_left 
    #
    #CIS_matrix_A_right_3 = CIS_matrix_B_left @ S_MO_inv[Nbas_A:(Nbas_A + Nbas_B), :Nbas_A]
    #CIS_matrix_B_left_3 = CIS_matrix_A_right @ S_MO_inv[:Nbas_A, Nbas_A:(Nbas_A + Nbas_B)]
    ##CIS_vector_EA_A_left_2[N_occ_A::].T =  
    ##CIS_vector_EA_B_left_2[N_occ_B::].T =  
    #CIS_vector_IP_A_left_3 = S_MO_inv[:Nbas_A , Nbas_A:(Nbas_A + Nbas_B)] @ CIS_vector_IP_B_left 
    #CIS_vector_IP_B_left_3 = S_MO_inv[Nbas_A:(Nbas_A + Nbas_B), :Nbas_A,] @  CIS_vector_IP_A_left 
    
    onel_couplings = {
    }
    
    #f_AB_eff, f_BA_eff = get_eff_fock(f_AB, f_BA, ene_A, ene_B, S_AB_inv, S_BA_inv, LCAO_A, LCAO_B)
    #red_C_s_2 = {
    #    "CIS_matrix_A_red":  CIS_matrix_A_right_2[:N_occ_A, N_occ_A:Nbas_A],
    #    "CIS_matrix_B_red":   CIS_matrix_B_left_2[:N_occ_B, N_occ_B:Nbas_B],
    #    "CIS_matrix_B_right_red": CIS_matrix_B_right[:N_occ_B, N_occ_B:Nbas_B], 
    #    "CIS_vector_EA_A_red": CIS_vector_EA_A_left[N_occ_A::].T, 
    #    "CIS_vector_EA_B_red": CIS_vector_EA_B_left[N_occ_B::].T, 
    #    "CIS_vector_IP_A_red": CIS_vector_IP_A_left_2[0:N_occ_A].T,
    #    "CIS_vector_IP_B_red": CIS_vector_IP_B_left_2[0:N_occ_B].T, 
    #    "CIS_vector_EA_B_right_red": CIS_vector_EA_B_right[N_occ_B::].T,
    #    "CIS_vector_IP_A_right_red": CIS_vector_IP_A_right[0:N_occ_A].T
    #}
    #
    #red_C_s_3 = {
    #    "CIS_matrix_A_red":  CIS_matrix_A_right_3[:N_occ_A, N_occ_A:Nbas_A],
    #    "CIS_matrix_B_red":   CIS_matrix_B_left_3[:N_occ_B, N_occ_B:Nbas_B],
    #    "CIS_matrix_B_right_red": CIS_matrix_B_right[:N_occ_B, N_occ_B:Nbas_B], 
    #    "CIS_vector_EA_A_red": CIS_vector_EA_A_left[N_occ_A::].T, 
    #    "CIS_vector_EA_B_red": CIS_vector_EA_B_left[N_occ_B::].T, 
    #    "CIS_vector_IP_A_red": CIS_vector_IP_A_left_3[0:N_occ_A].T,
    #    "CIS_vector_IP_B_red": CIS_vector_IP_B_left_3[0:N_occ_B].T, 
    #    "CIS_vector_EA_B_right_red": CIS_vector_EA_B_right[N_occ_B::].T,
    #    "CIS_vector_IP_A_right_red": CIS_vector_IP_A_right[0:N_occ_A].T
    #}
    
    overlaps = state_overlap(red_C_s=red_C_s, S_blocks=S_blocks, norms=normalization)

    
    #mon_A_TM, mon_A_TM_dc, mon_A_C4, mon_A_C4_dc = diag_results_check(f_AA=f_AA, CIS_matrix_A=CIS_matrix_A_right, LCAO_A=LCAO_A, f_AA_cfour=ene_A, N_occ_A=N_occ_A)
    #
    #V_21_AB_discon = V_21_discon(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_left,
    #                                            LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
    
    onel_couplings[21] = onel_cross_terms_21(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_left,
                                                LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B,  norms=normalization, S_blocks=S_blocks, F_blocks=fock_blocks, red_C_s=red_C_s)    

    onel_couplings[12] = onel_cross_terms_12(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_left,
                                                LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B,  norms=normalization, S_blocks=S_blocks, F_blocks=fock_blocks, red_C_s=red_C_s)  
    
    onel_couplings[31], onel_couplings[32], onel_couplings[41], onel_couplings[42], onel_couplings[13], onel_couplings[23], onel_couplings[14], onel_couplings[24] \
        = onel_cross_terms_mix(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_right,
                                                            CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_left.T,
                                                            CIS_vector_IP_A=CIS_vector_IP_A_left.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T,
                                                            LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B, onel_matrix_AA=f_AA, onel_matrix_BB=f_BB, normalization=normalization, F_blocks=fock_blocks, S_blocks=S_blocks, red_C_s=red_C_s, S_inv_blocks=S_inv_blocks)
        
    onel_couplings[34] = onel_cross_terms_34(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_right.T,
                    CIS_vector_IP_A=CIS_vector_IP_A_right.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T, LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, norms=normalization, S_blocks=S_blocks, F_blocks=fock_blocks, red_C_s=red_C_s)
        
    onel_couplings[43] = onel_cross_terms_43(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_right.T,
                    CIS_vector_IP_A=CIS_vector_IP_A_right.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T, LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, norms=normalization, S_blocks=S_blocks, F_blocks=fock_blocks, red_C_s=red_C_s)
    
    #results_21_2 = onel_cross_terms_Frenkel(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_left,
    #                                            LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B,  norms=normalization, S_blocks=S_blocks, F_blocks=fock_blocks, red_C_s=red_C_s_2)    
    #
    #results_31_2, results_32_2, results_41_2, results_42_2, results_13_2, results_23_2, results_14_2, results_24_2 \
    #    = onel_cross_terms_mix(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_right,
    #                                                        CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_left.T,
    #                                                        CIS_vector_IP_A=CIS_vector_IP_A_left.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T,
    #                                                        LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B, onel_matrix_AA=f_AA, onel_matrix_BB=f_BB, normalization=normalization, F_blocks=fock_blocks, S_blocks=S_blocks, red_C_s=red_C_s_2, S_inv_blocks=S_inv_blocks)
    #    
    #results_34_2 = onel_cross_terms_CT(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_right.T,
    #                CIS_vector_IP_A=CIS_vector_IP_A_right.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T, LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, norms=normalization, S_blocks=S_blocks, F_blocks=fock_blocks, red_C_s=red_C_s_2)    
    #
    #results_21_3 = onel_cross_terms_Frenkel(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_left,
    #                                            LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B,  norms=normalization, S_blocks=S_blocks, F_blocks=fock_blocks, red_C_s=red_C_s_3)    
    #
    #results_31_3, results_32_3, results_41_3, results_42_3, results_13_3, results_23_3, results_14_3, results_24_3 \
    #    = onel_cross_terms_mix(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_right,
    #                                                        CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_left.T,
    #                                                        CIS_vector_IP_A=CIS_vector_IP_A_left.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T,
    #                                                        LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B, onel_matrix_AA=f_AA, onel_matrix_BB=f_BB, normalization=normalization, F_blocks=fock_blocks, S_blocks=S_blocks, red_C_s=red_C_s_3, S_inv_blocks=S_inv_blocks)
    #    
    #results_34_3 = onel_cross_terms_CT(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_right.T,
    #                CIS_vector_IP_A=CIS_vector_IP_A_right.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T, LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, norms=normalization, S_blocks=S_blocks, F_blocks=fock_blocks, red_C_s=red_C_s_3)    
        
    
    #results_31['dc'] =  V_31_discon(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA, CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_right,
    #                                                        CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_left.T,
    #                                                        CIS_vector_IP_A=CIS_vector_IP_A_left.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T,
    #                                                        LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
    #
    #results_41['dc'] =  V_41_discon(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA,  CIS_matrix_A=CIS_matrix_A_right, CIS_matrix_B=CIS_matrix_B_right,
    #                                                        CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_left.T,
    #                                                        CIS_vector_IP_A=CIS_vector_IP_A_left.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T,
    #                                                        LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
    #
    #results_43_dc =  V_43_discon(onel_matfrix_AB=f_AB, onel_matfrix_BA=f_BA,
    #                                                        CIS_vector_EA_A=CIS_vector_EA_A_left.T, CIS_vector_EA_B=CIS_vector_EA_B_left.T,
    #                                                        CIS_vector_IP_A=CIS_vector_IP_A_left.T, CIS_vector_IP_B=CIS_vector_IP_B_left.T,
    #                                                        LCAO_A=LCAO_A, S_AB=S_AB, LCAO_B=LCAO_B, NBAS_A=Nbas_A, NBAS_B=Nbas_B)
    
    return overlaps, onel_couplings