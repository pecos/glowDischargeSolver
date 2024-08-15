# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:19:00 2023

@author: Malamas Tsagkaridis
"""

import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt 
import pandas as pd

from Constants import *
import ModelParameters as parameters

#----------------------------------------------------------------------------------

sigma_factor_AtomExc = 4*np.pi*a0_H**2*Eion_H**2 * M_Ar/M_H * xi_Ar**2
mass_factor = 2*spc.m_e/(M_Ar + spc.m_e)

mu_ei = (M_Ar - spc.m_e)*spc.m_e/M_Ar # Reduced mass
# lambda_factor = spc.h**2/(2.0*np.pi*spc.m_e*spc.k)*K_eV
lambda_factor = spc.h**2/(2.0*np.pi*mu_ei*spc.k)*K_eV


# Atom Impact Excitation
fij = [6.78e-2, 2.56e-1, 1.0e-3, 3.4e-2, 9.51e-2, 1.8e-2, 7.94e-2 ] # absorption oscillator strength of allowed transitions
atomImpactExcitationTransitionsFromGroundSate = ["Ar(4s[3/2]1)", "Ar(4s'[1/2]1)", "Ar(3d[1/2]1)", "Ar(5s[3/2]1)", "Ar(3d[3/2]1)", "Ar(5s'[1/2]1)", "Ar(3d'[3/2]1)"]

atomImpactExcitationTransitions_i = ["Ar(4s[3/2]2)","Ar(4s[3/2]2)","Ar(4s[3/2]2)","Ar(4s[3/2]1)","Ar(4s[3/2]1)","Ar(4s'[1/2]0)"]
atomImpactExcitationTransitions_j = ["Ar(4s[3/2]1)","Ar(4s'[1/2]0)","Ar(4s'[1/2]1)","Ar(4s'[1/2]0)","Ar(4s'[1/2]1)","Ar(4s'[1/2]1)"]
beta_ij = [1.79e-24, 4.8e-26, 4.8e-26, 4.8e-26, 4.8e-26, 1.79e-24]


# Should I use the Bohr radius of Ar or H?
sigma_factor_VanRegemorter = 8*np.pi/np.sqrt(3)*np.pi*a0_H**2*Eion_H**2 
gauntFactor_BoundBound = 1.0  # How can I calculate this one?
OscillatorStrength_VR = 0.001 # How can I calculate this one?

sigma_factor_Drawin = 4*np.pi*a0_H**2

#----------------------------------------------------------------------------------

def CharacteriseTransitions(*p):

    #  Selection rules
    # 1) J must change by +1,-1 or 0, however, the transition J = 0 to J = 0 is not allowed.
    # 2) l must change by +1 or -1.
    # 3) The parity must change.
    # 4) The multiplicity S must remain unchanged. 
    
    
    # unpack parameters
    (J_lvl, E_lvl, Racah_lvl,  DictRacah_lvl, 
    SubShell_lvl, isPrimed_lvl, AngMomQuantNum_lvl, Parity_lvl, Spin_lvl,
    f_ji, Racah_i, Racah_j,EmissionTransitions,
    CollTransition_Dict,CollTransition_ij,CollTransitionsList,
    GlobalToNistIndex,NistToGlobalIndex) = p  


    q = len(E_lvl)
    CollTransition_Status = {}

    i = 0
    for j in range(i+1,q):        
        CollTransition_Status[i,j] = 'n/a'
        
        Delta_J = abs(J_lvl[i]-J_lvl[j])
        Delta_l = abs(AngMomQuantNum_lvl[i] - AngMomQuantNum_lvl[j])
        Delta_Parity = abs(Parity_lvl[i] - Parity_lvl[j])
        ParityForbidden = (Delta_Parity == 0)
        Delta_Spin = Spin_lvl[i] - Spin_lvl[j]
                
        JFromZeroToZero = (J_lvl[i] == 0 and J_lvl[j] == 0)
        if (Delta_J == 0 or Delta_J == 1) and not JFromZeroToZero: 
            if Delta_l == 1: 
                if ParityForbidden:
                    CollTransition_Status[i,j] = 'AP' # Optically Allowed
                else:
                    CollTransition_Status[i,j] = 'A' # Optically Allowed
                    
            else:
                if ParityForbidden:
                    CollTransition_Status[i,j] = 'LPS' # Parity Forbidden
                else: 
                    CollTransition_Status[i,j] = 'LS' # Forbidden   

        elif JFromZeroToZero: 
                if Delta_l == 1:
                    if ParityForbidden:
                        CollTransition_Status[i,j] = 'J0PS' # Parity
                    else: 
                        CollTransition_Status[i,j] = 'J0S' # Parity
                else:
                    if ParityForbidden:
                        CollTransition_Status[i,j] = 'J0LPS' # Parity
                    else: 
                        CollTransition_Status[i,j] = 'J0LS' # Parity                            
        else:
                if Delta_l == 1:
                    if ParityForbidden:
                        CollTransition_Status[i,j] = 'JPS' # Parity
                    else: 
                        CollTransition_Status[i,j] = 'JS' # Parity
                else:
                    if ParityForbidden:
                        CollTransition_Status[i,j] = 'JLPS' # Parity
                    else: 
                        CollTransition_Status[i,j] = 'JLS' # Parity 
            

    for i in range(1,q):
        for j in range(i+1,q):        
            # The multiplicity is always unchanged.
            CollTransition_Status[i,j] = 'n/a'
        
            Delta_J = abs(J_lvl[i]-J_lvl[j])
            Delta_l = abs(AngMomQuantNum_lvl[i] - AngMomQuantNum_lvl[j])
            Delta_Parity = abs(Parity_lvl[i] - Parity_lvl[j])
            ParityForbidden = (Delta_Parity == 0)
            Delta_Spin = Spin_lvl[i] - Spin_lvl[j]
        
            JFromZeroToZero = J_lvl[i] == 0 and J_lvl[j] == 0
            if (Delta_J == 0 or Delta_J == 1) and not JFromZeroToZero: 
                if Delta_l == 1: 
                    if ParityForbidden:                    
                        CollTransition_Status[i,j] = 'AP' # Optically Allowed
                    else:     
                        CollTransition_Status[i,j] = 'A' # Parity Forbidden
                else:
                    if ParityForbidden:
                        CollTransition_Status[i,j] = 'LP' # Parity Forbidden                  
                    else: 
                        CollTransition_Status[i,j] = 'L' # Forbidden                                        
            elif JFromZeroToZero:
                if Delta_l == 1:
                    if ParityForbidden:                    
                        CollTransition_Status[i,j] = 'J0P' # Parity
                    else: 
                        CollTransition_Status[i,j] = 'J0' # Parity                                        
                else:
                    if ParityForbidden:                    
                        CollTransition_Status[i,j] = 'J0LP' # Parity
                    else:                                                 
                        CollTransition_Status[i,j] = 'J0L' # Parity                        
            else:
                if Delta_l == 1:
                    if ParityForbidden:                    
                        CollTransition_Status[i,j] = 'JP' # Parity
                    else: 
                        CollTransition_Status[i,j] = 'J' # Parity                                        
                else:
                    if ParityForbidden:                    
                        CollTransition_Status[i,j] = 'JLP' # Parity
                    else:                                                 
                        CollTransition_Status[i,j] = 'JL' # Parity   
                
                
 
        
    NTransAllowed = 0
    NTransParityForbidden = 0
    NTransSpinForbidden = 0
    NTransForbiddenRest = 0
    NTransCase = 0

    # for iCollTrans in CollTransitionsList: 
        # i = CollTransition_ij[iCollTrans,0] # Lower lever
        # j = CollTransition_ij[iCollTrans,1] # Upper level

    for i in range(q):
        for j in range(i+1,q): 
            Delta_J = abs(J_lvl[i]-J_lvl[j])
            Delta_l = abs(AngMomQuantNum_lvl[i] - AngMomQuantNum_lvl[j])
            Delta_Parity = abs(Parity_lvl[i] - Parity_lvl[j])
            ParityForbidden = (Delta_Parity == 0)

            if CollTransition_Status[i,j] == 'A':
                NTransAllowed = NTransAllowed +1

            if CollTransition_Status[i,j] == 'P':
                NTransParityForbidden = NTransParityForbidden +1        

            if CollTransition_Status[i,j] == 'S':
                NTransSpinForbidden = NTransSpinForbidden +1   

            if CollTransition_Status[i,j] == 'F':
                NTransForbiddenRest = NTransForbiddenRest +1  

            # # Perform some checks
            # if not ParityForbidden and (Delta_l != 1):  # This is not satisfied if I include many transitions
            #     print('Problem with ParityForbidden and Delta_l!') 
            #     raise SystemExit(0)

            # Perform some checks
            if Delta_l == 1: # if Delta_l == 1 then the parity changes! Therefore it can not be ParityForbidden.
                if ParityForbidden:
                    print('Problem with ParityForbidden and Delta_l!') 
                    raise SystemExit(0)
        
            if CollTransition_Status[i,j] == 'A': 
                if ParityForbidden: 
                    print('This transition is parity forbidden but was assigned as "Allowed"')
                    print(i,j,Racah_lvl[i],'->',Racah_lvl[j], CollTransition_Status[i,j])
                    raise SystemExit(0)

            if CollTransition_Status[i,j] == 'A': 
                iCollTrans = CollTransition_Dict[i,j]
                if iCollTrans in GlobalToNistIndex.keys():
                    itrans = GlobalToNistIndex[iCollTrans]
                    OscillatorStrength = f_ji[itrans]
                else:
                    OscillatorStrength = 0                      
                # print(iCollTrans,i,j,Racah_lvl[i],'->',Racah_lvl[j], CollTransition_Status[i,j],OscillatorStrength)
                
            if CollTransition_Status[i,j] == 'JP':
                NTransCase = NTransCase +1  


    for itrans in EmissionTransitions: 
        MyString = Racah_i[itrans] + "->" + Racah_j[itrans] + " "

        i = DictRacah_lvl[Racah_i[itrans]]
        j = DictRacah_lvl[Racah_j[itrans]]

        iCollTrans = NistToGlobalIndex[itrans]

        i = CollTransition_ij[iCollTrans,0]
        j = CollTransition_ij[iCollTrans,1]         
        
        Delta_l = abs(AngMomQuantNum_lvl[i] - AngMomQuantNum_lvl[j])

        # print(MyString,CollTransition_Status[i,j],Delta_l,f_ji[itrans])
        # if CollTransition_Status[i,j] != 'A':
            # print(MyString,CollTransition_Status[i,j],' ',f_ji[itrans],Source_ji[itrans],)


# Ar(4f[9/2]4) -> Ar(4d'[3/2]1)  J   0.00088 MCDHF
# Ar(3d[3/2]2) -> Ar(5f[7/2]4)  J   2.82e-06 MCDHF
# Ar(4d[3/2]2) -> Ar(5f[7/2]4)  J   1.37e-05 MCDHF


    print("Number of measured transitions = ", len(EmissionTransitions))
    print("Number of Allowed transitions = ", NTransAllowed)
    print("Number of Parity Forbidden transitions = ", NTransParityForbidden)
    print("Number of Spin Forbidden transitions = ", NTransSpinForbidden)
    print("Number of Forbidden transitions = ", NTransForbiddenRest)
    print("Number of transitions = ", NTransCase)

    # raise SystemExit(0)


    """
    Read Kimura et al 1985 datat
    """
    os.chdir(homeDir)
    c1,c2,unprimed,primed = LoadKimura1985Data_K('./CRModel/Data/','Kimura1985.csv')

    KimuraFactor_K = {} 
    for iCollTrans in CollTransitionsList: 
        i = CollTransition_ij[iCollTrans,0] # Lower lever
        j = CollTransition_ij[iCollTrans,1] # Upper level
        
        for element in range(len(c1)):
            if c1[element] == SubShell_lvl[i] and c2[element] == SubShell_lvl[j]:
                if (not isPrimed_lvl[i]) and (not isPrimed_lvl[j]):
                    KimuraFactor_K[iCollTrans] = unprimed[element]  
                elif (isPrimed_lvl[i]) and (isPrimed_lvl[j]):         
                    KimuraFactor_K[iCollTrans] = primed[element] 
                else:
                     KimuraFactor_K[iCollTrans] = 1.0 #min(unprimed[element],primed[element])
                break

                
        if element == len(c1) -1:  
            KimuraFactor_K[iCollTrans] = 1.0

        # if CollTransition_Status[i,j] == 'P':
        #     if (not isPrimed_lvl[i]) and (not isPrimed_lvl[j]):
        #         if SubShell_lvl[i] == '4s' and SubShell_lvl[i] == '4s': 
        #             KimuraFactor_K[iCollTrans] = 10.0 

        #         if SubShell_lvl[i] == '3d' and SubShell_lvl[i] == '3d': 
        #             KimuraFactor_K[iCollTrans] = 100.0 

        #         if SubShell_lvl[i] == '4p' and SubShell_lvl[i] == '4p': 
        #             KimuraFactor_K[iCollTrans] = 100.0 
        
        #     elif (isPrimed_lvl[i]) and (isPrimed_lvl[j]):         
        #         if SubShell_lvl[i] == '4s' and SubShell_lvl[i] == '4s': 
        #             KimuraFactor_K[iCollTrans] = 10.0 
                    
        #         if SubShell_lvl[i] == '3d' and SubShell_lvl[i] == '3d': 
        #             KimuraFactor_K[iCollTrans] = 100.0 

        #         if SubShell_lvl[i] == '4p' and SubShell_lvl[i] == '4p': 
        #             KimuraFactor_K[iCollTrans] = 100.0                     
                    
                                    
        # print(SubShell_lvl[i],SubShell_lvl[j],KimuraFactor_K[iCollTrans])




    return CollTransition_Status,KimuraFactor_K




def ExcitationCrossSections():
    """    
    Arguments:
    """  
    
    # unpack parameters
    p = parameters.modelParameters(1000)

    E_lvl = p.E_lvl
    Racah_lvl = p.Racah_lvl
    DictRacah_lvl = p.DictRacah_lvl 
    Spin_lvl = p.Spin_lvl
    f_ji = p.f_ji
    collDict = p.collDict
    CollTransition = p.CollTransition
    CollTransition_ij = p.CollTransition_ij
    CollTransitions_Rest = p.CollTransitions_Rest
    GlobalToNistIndex = p.GlobalToNistIndex
    CollTransition_Status = p.CollTransition_Status

    # For Electron Energy Distribution Function (EEDF) :
    eRange = np.logspace(np.log10(1e-2),np.log10(300),2000)  # [eV]
    
        
    ################## Elecrton impact de/excitation ##################
    sigma_BSR = {}
    sigma_Drawin= {}
    sigma_Drawin_A= {}
    sigma_Drawin_P= {}
    sigma_Drawin_S= {}
    sigma_VanRegemorter= {}

    # From LXCat BSR data
    icount = 0
    for key1 in collDict:
        for key2 in collDict[key1]:
            icount += 1
            # print(icount,key1,key2,key2.split(" "))
        
            if (key2.split(" ")[2] != 'Ar(Rydberg)'):
                i = DictRacah_lvl[key2.split(" ")[0]]
                j = DictRacah_lvl[key2.split(" ")[2]]
                eij = (E_lvl[j] - E_lvl[i])*cm_eV
            
                # if (i > j):
                #     print(i,Racah_lvl[i],E_lvl[i]*cm_eV)
                #     print(j,Racah_lvl[j],E_lvl[j]*cm_eV)
                #     raise SystemExit(0)
                if (i<j):                    
                    sigma_ij_list = collDict[key1][key2]
                    sigma_ij = np.interp(eRange,sigma_ij_list[:,0],sigma_ij_list[:,1])
                    sigma_ij[np.where(eRange < eij)] = 0
                    
                    sigma_BSR[i,j] = sigma_ij
                    
                    
                    iCollTrans = CollTransition[i,j]
                    
                
                    # Drawin's Formula (unit parameters have been assumed)
                    CollStatus = CollTransition_Status[i,j]
                    sigma_ij_pre = sigma_factor_Drawin*(eRange/eij-1)/(eRange/eij)**2
                    if CollStatus == 'A' or CollStatus == 'AP': 
                        if iCollTrans in GlobalToNistIndex.keys():
                            itrans = GlobalToNistIndex[iCollTrans]
                            OscillatorStrength = f_ji[itrans]
                        else:
                            OscillatorStrength = OscillatorStrength_VR 
                        sigma_ij = sigma_ij_pre * OscillatorStrength * (Eion_H/eij)**2 * np.log(1.25*eRange/eij)
                    elif CollStatus == 'P' or 'LP' or 'L' or 'JP' or 'J' or 'JLP' or 'JL' or 'JOP' or 'JOLP' or 'JOL' or 'LPS' or 'J0PS' or 'J0LPS' or 'J0LS' or 'JLPS' or 'JLS':
                        sigma_ij = sigma_ij_pre 
                    elif CollStatus == 'J0' or 'LS' or  'J0S' or 'JPS' or 'JS':
                        sigma_ij = sigma_ij_pre * (eRange/eij+1)/(eRange/eij)**3  
                                                                                
                    sigma_ij[np.where(eRange < eij)] = 0

                    sigma_Drawin[i,j] = sigma_ij 


                    # if CollStatus == 'A': 
                    sigma_ij = sigma_ij_pre * (Eion_H/eij)**2 * np.log(1.25*eRange/eij)
                    sigma_ij[np.where(eRange < eij)] = 0
                    sigma_Drawin_A[i,j] = sigma_ij
                    # elif CollStatus == 'P' or CollStatus == 'F':
                    sigma_Drawin_P[i,j] = sigma_ij_pre
                    sigma_ij[np.where(eRange < eij)] = 0
                    sigma_ij = sigma_ij
                    # elif CollStatus == 'S':
                    sigma_ij = sigma_ij_pre * (eRange/eij+1)/(eRange/eij)**3                                                          
                    sigma_ij[np.where(eRange < eij)] = 0
                    sigma_Drawin_S[i,j] = sigma_ij

                if (i<j):                    
                    # Van Regemorter 1962 
                    if CollStatus == 'A':
                         sigma_ij = sigma_factor_VanRegemorter*gauntFactor_BoundBound*OscillatorStrength/eRange/eij
                    else:
                         sigma_ij = sigma_factor_VanRegemorter*gauntFactor_BoundBound*OscillatorStrength_VR/eRange/eij                         
                    sigma_ij[np.where(eRange < eij)] = 0
                    sigma_VanRegemorter[i,j] = sigma_ij

                        
                    
                    
                                     
    for iCollTrans in CollTransitions_Rest: 
        i = CollTransition_ij[iCollTrans,0] # Lower lever
        j = CollTransition_ij[iCollTrans,1] # Upper level
        # CollTransition_Dict[i,j] should be equal to iCollTrans
        # if (i > j): # Perform a test
            # raise SystemExit(0)
                        
        eij = (E_lvl[j] - E_lvl[i])*cm_eV
             
        if (eij >0.0):                  
                
            # Drawin's Formula (unit parameters have been assumed)
            CollStatus = CollTransition_Status[i,j]
            sigma_ij_pre = sigma_factor_Drawin*(eRange/eij-1)/(eRange/eij)**2
            if CollStatus == 'A' or CollStatus == 'AP': 
                if iCollTrans in GlobalToNistIndex.keys():
                    itrans = GlobalToNistIndex[iCollTrans]
                    OscillatorStrength = f_ji[itrans]
                else:
                    OscillatorStrength = OscillatorStrength_VR 
                sigma_ij = sigma_ij_pre * OscillatorStrength * (Eion_H/eij)**2 * np.log(1.25*eRange/eij)
            elif CollStatus == 'P' or 'LP' or 'L' or 'JP' or 'J' or 'JLP' or 'JL' or 'JOP' or 'JOLP' or 'JOL' or 'LPS' or 'J0PS' or 'J0LPS' or 'J0LS' or 'JLPS' or 'JLS':
                sigma_ij = sigma_ij_pre 
            elif CollStatus == 'J0' or 'LS' or  'J0S' or 'JPS' or 'JS':
                sigma_ij = sigma_ij_pre * (eRange/eij+1)/(eRange/eij)**3  
                                                                                
            sigma_ij[np.where(eRange < eij)] = 0

            sigma_Drawin[i,j] = sigma_ij 


            # if CollStatus == 'A': 
            sigma_ij = sigma_ij_pre * (Eion_H/eij)**2 * np.log(1.25*eRange/eij)
            sigma_ij[np.where(eRange < eij)] = 0
            sigma_Drawin_A[i,j] = sigma_ij
            # elif CollStatus == 'P' or CollStatus == 'F':
            sigma_Drawin_P[i,j] = sigma_ij_pre
            sigma_ij[np.where(eRange < eij)] = 0
            sigma_ij = sigma_ij
                # elif CollStatus == 'S':
            sigma_ij = sigma_ij_pre * (eRange/eij+1)/(eRange/eij)**3                                                          
            sigma_ij[np.where(eRange < eij)] = 0
            sigma_Drawin_S[i,j] = sigma_ij

            if (i<j):                    
                # Van Regemorter 1962 
                if CollStatus == 'A':
                    sigma_ij = sigma_factor_VanRegemorter*gauntFactor_BoundBound*OscillatorStrength/eRange/eij
                else:
                    sigma_ij = sigma_factor_VanRegemorter*gauntFactor_BoundBound*OscillatorStrength_VR/eRange/eij                         
                sigma_ij[np.where(eRange < eij)] = 0
                sigma_VanRegemorter[i,j] = sigma_ij







    # fig,ax = plt.subplots(dpi=140)
    # i = 0; j = 1
    # Title = Racah_lvl[i] + '->' + Racah_lvl[j] + "   Status = " + CollTransition_Status[i,j]
    # ax.plot(eRange,sigma_BSR[i,j],c='b',label="BSR")
    # ax.plot(eRange,sigma_Drawin_A[i,j],'r.',label="Drawin - A")
    # ax.plot(eRange,sigma_Drawin_P[i,j],'m.',label="Drawin - P")
    # ax.plot(eRange,sigma_Drawin_S[i,j],'g.',label="Drawin - S")
    # ax.plot(eRange,sigma_VanRegemorter[i,j],'c.',label="Van Regemorter")
    # ax.loglog()
    # # plt.ylim([1e2,1e26])
    # plt.xlabel('E [eV]')
    # plt.ylabel('$\sigma_{ex}$ [m$^2]$')
    # plt.title(Title)
    # plt.grid(True)
    # plt.legend()

    # fig,ax = plt.subplots(dpi=140)
    # sigma = collDict['Ar']['Ar -> Ar(4s[3/2]2)']
    # ax.plot(sigma[:,0],sigma[:,1],c='b',label="Ar -> Ar(4s[3/2]2)")
    # ax.loglog()
    # # plt.ylim([1e2,1e26])
    # plt.xlabel('E [eV]')
    # plt.ylabel('$\sigma_{ex}$ [m$^2]$')
    # plt.title('Excitation cross-sections')
    # plt.grid(True)
    # plt.legend()


    for key1 in collDict:
        for key2 in collDict[key1]:
            icount += 1
            # print(icount,key1,key2,key2.split(" "))
        
            if (key2.split(" ")[2] != 'Ar(Rydberg)'):
                i = DictRacah_lvl[key2.split(" ")[0]]
                j = DictRacah_lvl[key2.split(" ")[2]]
                eij = (E_lvl[j] - E_lvl[i])*cm_eV

                iCollTrans = CollTransition[i,j]

        
                if (i<j):
                    # if CollTransition_Status[i,j] == 'P' and SubShell_lvl[i] == SubShell_lvl[j] and SubShell_lvl[i] == '5s':   
                    if CollTransition_Status[i,j] == 'A':   
                        Delta_Spin = Spin_lvl[i] - Spin_lvl[j]

                        if iCollTrans in GlobalToNistIndex.keys():
                            itrans = GlobalToNistIndex[iCollTrans]
                            OscillatorStrength = f_ji[itrans]
                        else:
                            OscillatorStrength = OscillatorStrength_VR 
                            

                        fig,ax = plt.subplots(dpi=140)
                        # Title = str(i) + ' ' + str(j) + ' '  + Racah_lvl[i] + '->' + Racah_lvl[j] + "   Status = " + CollTransition_Status[i,j] + " K[i,j] = " + str(KimuraFactor_K[iCollTrans]) 
                        Title = str(i) + ' ' + str(j) + ' '  + Racah_lvl[i] + '->' + Racah_lvl[j] + "   Status = " + CollTransition_Status[i,j] + " fji = " + str(OscillatorStrength) 
                        ax.plot(eRange,sigma_BSR[i,j],c='b',label="BSR")
                        ax.plot(eRange,sigma_Drawin[i,j],'k.',label="Drawin - 1")

                        ax.plot(eRange,sigma_Drawin_A[i,j],'r.',label="Drawin - A")
                        ax.plot(eRange,sigma_Drawin_P[i,j],'m.',label="Drawin - P")
                        ax.plot(eRange,sigma_Drawin_S[i,j],'g.',label="Drawin - S")
                        # ax.plot(eRange,sigma_VanRegemorter[i,j],'c.',label="Van Regemorter")
                        ax.loglog()
                        # plt.ylim([1e2,1e26])
                        plt.xlabel('E [eV]')
                        plt.ylabel('$\sigma_{ex}$ [m$^2]$')
                        plt.title(Title)
                        plt.grid(True)
                        plt.legend()
                        plt.show()



    for iCollTrans in CollTransitions_Rest: 
        i = CollTransition_ij[iCollTrans,0]
        j = CollTransition_ij[iCollTrans,1]           
        Delta_Spin = Spin_lvl[i] - Spin_lvl[j]

        if CollTransition_Status[i,j] == 'A':   
  
            if iCollTrans in GlobalToNistIndex.keys():
                itrans = GlobalToNistIndex[iCollTrans]
                OscillatorStrength = f_ji[itrans]
            else:
                OscillatorStrength = OscillatorStrength_VR 

            fig,ax = plt.subplots(dpi=140)
            # Title = str(i) + ' ' + str(j) + ' '  + Racah_lvl[i] + '->' + Racah_lvl[j] + "   Status = " + CollTransition_Status[i,j] + " K[i,j] = " + str(KimuraFactor_K[iCollTrans]) 
            Title = str(i) + ' ' + str(j) + ' '  + Racah_lvl[i] + '->' + Racah_lvl[j] + "   Status = " + CollTransition_Status[i,j] + " f_ji = " + str(OscillatorStrength) 
            # ax.plot(eRange,sigma_BSR[i,j],c='b',label="BSR")
            ax.plot(eRange,sigma_Drawin[i,j],'k.',label="Drawin - 1")

            ax.plot(eRange,sigma_Drawin_A[i,j],'r.',label="Drawin - A")
            ax.plot(eRange,sigma_Drawin_P[i,j],'m.',label="Drawin - P")
            ax.plot(eRange,sigma_Drawin_S[i,j],'g.',label="Drawin - S")
            # ax.plot(eRange,sigma_VanRegemorter[i,j],'c.',label="Van Regemorter")
            ax.loglog()
            # plt.ylim([1e2,1e26])
            plt.xlabel('E [eV]')
            plt.ylabel('$\sigma_{ex}$ [m$^2]$')
            plt.title(Title)
            plt.grid(True)
            plt.legend()
            plt.show()


                                  
    rflag = True
    return rflag



def LoadKimura1985Data_K(fileDir,fileName): 
    os.chdir(fileDir)
    
    # I need to modify that in order to get data for P = 1 atm!!!!
    Data = []
    Data = pd.read_csv(fileName)
    c1 = Data['c1'].to_list()  
    c2 = Data['c2'].to_list()  
    unprimed = Data['unprimed'].to_numpy(dtype='float64') 
    primed = Data['primed'].to_numpy(dtype='float64') 

    # c1 = Data.iloc[1:,1].to_numpy('float64')
    # c2 = Data.iloc[1:1000,2].to_numpy('float64')
    # unprimed = Data.iloc[1:1000,3].to_numpy('float64')
    # primed = Data.iloc[1:,4].to_numpy('float64')
    
    # i=11
    # print(c1[i],c2[i],unprimed[i],primed[i])


    return c1,c2,unprimed,primed
    

def LoadKimura1985Data_A(fileDir,fileName): 
    os.chdir(fileDir)
    
    # I need to modify that in order to get data for P = 1 atm!!!!
    Data = []
    Data = pd.read_csv(fileName)
    c1 = Data['c1'].to_list()  
    c2 = Data['c2'].to_list()  
    unprimed = Data['unprimed'].to_numpy(dtype='float64') 
    primed = Data['primed'].to_numpy(dtype='float64') 

    # c1 = Data.iloc[1:,1].to_numpy('float64')
    # c2 = Data.iloc[1:1000,2].to_numpy('float64')
    # unprimed = Data.iloc[1:1000,3].to_numpy('float64')
    # primed = Data.iloc[1:,4].to_numpy('float64')
    
    # i=11
    # print(c1[i],c2[i],unprimed[i],primed[i])


    return c1,c2,unprimed,primed
    
