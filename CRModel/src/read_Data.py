# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:19:00 2023

@author: Malamas Tsagkaridis
"""

import os
import numpy as np
import pandas as pd
from Constants import *

from ConvertToRacahNotation import ConvertToRacah,FindSubShell
from GeneralFunctions import CalcEinsteinCoef

def NIST_read_ArI(fileDir,fileName, Nlvl): 
    os.chdir(fileDir)

    """
    Read in considered states of Ar I and their properties
    """
    print(fileDir,fileName)
    States = []
    States = pd.read_csv(fileName)
    g = States['g'].to_numpy(dtype='float64')
    Nlvl_f =len(g) 
    Nlvl = min(Nlvl_f,Nlvl)
    g= g[0:Nlvl] 
    E = States['Level (cm-1)'].to_numpy(dtype='float64'); E = E[0:Nlvl]

    Configuration = States['Configuration'].to_list();  Configuration = Configuration[0:Nlvl] 
    Term = States['Term'].to_list(); Term = Term[0:Nlvl]   
    J = States['J'].to_numpy(); J = J[0:Nlvl]   
    

    for ilvl in range(len(Configuration)): 
        Configuration[ilvl] = np.char.strip(Configuration[ilvl])
        Term[ilvl] = np.char.strip(Term[ilvl])


    # Convert to Racah notation 
    DictRacah = {}
    Racah,  DictRacah = ConvertToRacah(Configuration,Term,J)
    SubShell, isLevelPrimed, PrinQuantNum, AngMomQuantNum, Parity, Spin = FindSubShell(Configuration,Term,J,Racah)

    # for i in range(len(Configuration)): 
        # print(Racah[i],DictRacah[Racah[i]],E[i])
        # print(Racah[i],' ',E[i])
    # Example of how to access data having Racah notation 
    # Racah_level = 'Ar(4p[3/2]1)'
    # print(DictRacah[Racah_level], DictRacah.index(Racah_level))
    # Racah_level = 'Ar(3d[3/2]1)'
    # print(DictRacah[Racah_level])
    
    for i in range(1,len(E)):
        if (E[i] - E[i-1] < 0.0): 
            print("Energy levels are not ordered. Program will stop")
            raise SystemExit(0)
    
    p = (Configuration, Term, J, E, g, Racah,  DictRacah,  \
        SubShell, isLevelPrimed, PrinQuantNum, AngMomQuantNum, Parity, Spin)
    return p 
    


def NIST_read_ArII(fileDir,fileName): 
    os.chdir(fileDir)

    """
    Read in considered states of Ar II and their properties
    """
    States = []
    States = pd.read_csv(fileName)
    g = States['g'].to_numpy(dtype='float64') 
    E = States['Level (cm-1)'].to_numpy(dtype='float64')

    Configuration = States['Configuration'].to_list()  
    Term = States['Term'].to_list()  
    J = States['J'].to_numpy()  

    for ilvl in range(len(Configuration)): 
        Configuration[ilvl] = np.char.strip(Configuration[ilvl])
        Term[ilvl] = np.char.strip(Term[ilvl])

    for i in range(1,len(E)):
        if (E[i] - E[i-1] < 0.0): 
            print("Energy levels for ArII are not ordered. Program will stop")
            raise SystemExit(0)


    p = (Configuration, Term, J, E, g)
    return p
    


def read_ArI_transitions(fileDir,fileName, *levels): 
    os.chdir(fileDir)
    
    """
    Read in transition data for Ar I
    i -> lower level
    j -> upper level
    """

    (Configuration_lvl, Term_lvl, J_lvl, E_lvl, g_lvl, Racah_lvl,  DictRacah_lvl) = levels

    # ArI_Data = pd.read_csv(fileName,skiprows=0)
    ArI_Data = pd.read_csv(fileName)
    A_ji = ArI_Data.iloc[:,3].to_numpy('float64')
    f_ji = ArI_Data.iloc[:,4].to_numpy('float64')

    E_i = ArI_Data.iloc[:,6].to_numpy('float64')
    E_j = ArI_Data.iloc[:,7].to_numpy('float64')
    g_i = ArI_Data.iloc[:,14].to_numpy('float64')
    g_j = ArI_Data.iloc[:,15].to_numpy('float64')

    Configuration_i = ArI_Data.iloc[:,8].to_list()
    Term_i = ArI_Data.iloc[:,9].to_list()
    J_i = ArI_Data.iloc[:,10].to_numpy()


    Configuration_j = ArI_Data.iloc[:,11].to_list()
    Term_j = ArI_Data.iloc[:,12].to_list()
    J_j = ArI_Data.iloc[:,13].to_numpy()

    Source_ji = ArI_Data.iloc[:,19].to_list()

    for itrans in range(len(Configuration_i)): 

        Configuration_i[itrans] = np.char.strip(Configuration_i[itrans])
        Term_i[itrans] = np.char.strip(Term_i[itrans])
        Configuration_j[itrans] = np.char.strip(Configuration_j[itrans])
        Term_j[itrans] = np.char.strip(Term_j[itrans])



    # Convert to Racah notation 
    DictRacah_i = {}
    Racah_i,  DictRacah_i = ConvertToRacah(Configuration_i,Term_i,J_i)

    DictRacah_j = {}
    Racah_j,  DictRacah_j = ConvertToRacah(Configuration_j,Term_j,J_j)

        # DictRacah_i, DictRacah_j give the wrong index because the keys in these 
        # dictionaries are multiply difined as we loop over all radiative transitions



    for itrans in range(len(Configuration_i)):
        # i = DictRacah_lvl[Racah_i[itrans]]
        # j = DictRacah_lvl[Racah_j[itrans]]   
        # print(itrans,i,j,print(Racah_lvl[i],Racah_lvl[j]))

        if  A_ji[itrans] == -1:

            i = DictRacah_lvl[Racah_i[itrans]]
            j = DictRacah_lvl[Racah_j[itrans]]            
            
            E_i[itrans] = E_lvl[i]
            E_j[itrans] = E_lvl[j]
            g_i[itrans] = g_lvl[i]
            g_j[itrans] = g_lvl[j]
             
            A_ji[itrans] = CalcEinsteinCoef(E_i[itrans],E_j[itrans],g_i[itrans],g_j[itrans],f_ji[itrans])
            
            lambda_0 = spc.h*spc.c/((E_j[itrans]-E_i[itrans])*cm_eV*spc.e) # wavelength of transition

            # print(itrans,A_ji[itrans],f_ji[itrans],E_i[itrans],E_j[itrans],Racah_i[itrans]," -> ",Racah_j[itrans],g_i[itrans],g_j[itrans]   )
            print(lambda_0*1e9)
        
 
    p = (A_ji, f_ji, E_i, E_j, g_i, g_j, Configuration_i, Term_i, J_i,  
        Configuration_j, Term_j, J_j, 
        Racah_i, DictRacah_i, Racah_j, DictRacah_j,Source_ji)
    
    return p



#----------------------------------------------------------------------------------


def LXCat_read(fileDir,fileName): 
    os.chdir(fileDir)

    print('Reading the LXCat excitation collision cross-section data.')    
    
    with open(fileName) as f:
        lines = f.readlines()
    
    del lines[0:71]
    
    count = 0
    inCount = 0
    countTrans = 0
    collect = False
    transDict = {}
    stateTransDict = {}
    upState = []
    for i in lines:
        
        if '*' in i:
            if count == 0:        
                count += 1
            else:
                stateTransDict[upState] = transDict
                count += 1
                  
            upState = i.translate({ord(c): None for c in '*'})
            upState = upState.translate({ord(c): None for c in ' '})
            upState = upState.strip()
            transDict = {}
            inCount = 1     
            continue           
                    
        if inCount == 3:
            transition = i.strip()
            inCount += 1
            countTrans += 1
            continue
        
        if '---' in i and inCount > 3 and collect == False:
            data = []
            collect = True
            continue
              
        if '---' in i and collect == True:
            collect = False
            data = np.vstack(data)
            transDict[transition] = data
            data = []
            inCount = 1
            continue

        if collect == True:
            tempData = i.strip()
            tempData = tempData.split('\t')
            tempData = np.array(tempData,dtype='float64')
            data.append(tempData) 
            
        if 'xxxx' in i:
            stateTransDict[upState] = transDict
            break

        # if icount == 370:
        #     break

        inCount += 1
                

    print('Number of LXCat levels considered: ', count)    
    print('Number of LXCat collisional transitions considered: ', countTrans)    
              

    return stateTransDict,count,countTrans
    

