# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:19:00 2023

@author: Malamas Tsagkaridis
"""

import os
import numpy as np



def ConvertToRacah(Configuration,Term,J): 
    
    Racah_list = []
    # Convert to Racah notation 
    Racah_Dict = {}
    Racah = ''
    temp = ''
    for i in range(len(Configuration)):
        
        if Configuration[i] == "3s2.3p6":
            Racah = "Ar"               
        else:
            temp = str(Configuration[i])
            temp = temp.split('.')    
            Racah = "Ar(" + str(temp[-1])
            if "<1/2>" in str(Configuration[i]):
                Racah += "'"

            start = '['
            end = ']'
            temp = str(Term[i])
            temp = temp.split(start)[1].split(end)[0]
            Racah = Racah + "[" + temp + "]"
            Racah = Racah + str(J[i]) + ")"
    
        Racah_Dict[Racah] = i
        Racah_list.append(Racah)
        # print(i,Configuration[i],Term[i],J[i],Racah)

    ##### Example on how to access data having Racah notation 
    # Racah_level = 'Ar(4p[3/2]1)'
    # print(DictRacah_lvl[Racah_level], Racah_lvl.index(Racah_level))

    return Racah_list, Racah_Dict


def FindSubShell(Configuration,Term,J,Racah): 
    SubShell = ''
    SubShell_list = []
    
    isLevelPrimed_list = []

    PrinQuantNum_list = []
    AngMomQuantNum_list = []  # s -> 0, p -> 1, d -> 2, f -> 3, g -> 4, h -> 5, 
    Parity_list = []
    Spin_list = []

    temp = ''
    
    for i in range(len(Configuration)):
        isLevelPrimed = False
        temp = str(Configuration[i])
        temp = temp.split('.')    

        
        if Configuration[i] == "3s2.3p6":
            SubShell = " "  
            PrinQuantNum = 3      
            AngMomQuantNum = 1
            Spin = 0       
        else:
            SubShell = str(temp[-1])
            PrinQuantNum = int(temp[-1][0:-1])

            temp_2 = str(Term[i])
            temp_2 = temp_2.split('[')[1].split(']')[0]
            temp_2 = temp_2.split('/')

            Spin = float(J[i]) - float(temp_2[0])/float(temp_2[1]) 
            
            l = temp[-1][-1]
            if l == 's':
               AngMomQuantNum = 0
            elif l == 'p': 
               AngMomQuantNum = 1
            elif l == 'd': 
               AngMomQuantNum = 2            
            elif l == 'f': 
               AngMomQuantNum = 3            
            elif l == 'g': 
               AngMomQuantNum = 4
            elif l == 'h': 
               AngMomQuantNum = 5               
               
               
               
        SubShell_list.append(SubShell)
        
        if "'" in str(Racah[i]):
            isLevelPrimed = True

        isLevelPrimed_list.append(isLevelPrimed)


        if "*" in str(Term[i]):
            Parity = 1 # Odd parity
        else:
            Parity = 0 # Even parity
           
            
        PrinQuantNum_list.append(PrinQuantNum)
        AngMomQuantNum_list.append(AngMomQuantNum)
        Parity_list.append(Parity)
        Spin_list.append(Spin)

        # print(i,Configuration[i],Term[i],J[i],Racah[i],SubShell_list[i],isLevelPrimed_list[i])

    return SubShell_list,isLevelPrimed_list,PrinQuantNum_list, AngMomQuantNum_list, Parity_list, Spin_list


