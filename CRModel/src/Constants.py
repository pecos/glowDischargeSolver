# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:04:49 2023

@author: Malamas Tsagkaridis
"""
#----------------------------------------------------------------------------------

import numpy as np
import scipy.constants as spc
import os

#----------------------------------------------------------------------------------

# raise SystemExit(0) 
# print(type(K_eV))    
# print(np.size(npop))

# Figure Format
#----------------------------------------------------------------------------------
tfs = 14
xfs = 18
yfs = 18
lfs = 14

# Define constants
#----------------------------------------------------------------------------------
homeDir = os.getcwd()

cm_eV = spc.h*spc.c/spc.e*100  # Convert energy units: from cm^-1 to eV
K_eV = spc.k/spc.e             # Convert energy units: from K to eV

# pi = np.pi

# Hydrogen
Eion_H = np.array([13.598434599702]) # ionization energy of H atom in [eV]
Mr_H = 1.00784/1000.0       # [kg/mol]
M_H = Mr_H/spc.N_A        # [kg] mass of argon atom (6.63352088e-26 kg)
a0_H = 52.9e-12   # [m] Bohr radius of Hydrogen

# Argon
xi_Ar = 6                   # [#] number of optical electrons of argon
Mr_Ar = 39.948/1000.0       # [kg/mol]
R = spc.R/Mr_Ar             # R = 208.13 [J/kg/K]  Argon
M_Ar = Mr_Ar/spc.N_A        # [kg] mass of argon atom (6.63352088e-26 kg)
RydEn = spc.Rydberg*spc.c*spc.h/spc.e # [eV]
a0 = 71e-12 # [m] Bohr radius of argon
g_ion = 4.0 # There should be two ionizasion levels. What do we do then??
g_ion_1 = 4.0 # There are two ionizasion levels.
g_ion_2 = 2.0 

Eion = 15.7596119 # ionization energies of Ar in [eV]
# Eion = np.array([15.7596119]) # ionization energies of Ar in [eV]
Eion_Ar_1 = np.array([15.7596119]) # ionization energy of Ar in [eV]
Eion_Ar_2 = np.array([15.7596119+0.17749368]) # ionization energy of Ar  in [eV]

#----------------------------------------------------------------------------------

VacPermittivity = 8.8541878128e-12

#----------------------------------------------------------------------------------

# Indices
iNg = 0
iNe = -3 
iNion = -2 
iEe = -1
# iEh = -1

#----------------------------------------------------------------------------------

"""
No. of ionization levels considered
"""
z = 1
