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
xfs = 16
yfs = 16
lfs = 14

# Define constants
#----------------------------------------------------------------------------------
homeDir = os.getcwd()

cm_eV = spc.h*spc.c/spc.e*100  # Convert energy units: from cm^-1 to eV
K_eV = spc.k/spc.e             # Convert energy units: from K to eV

# pi = np.pi

# Hydrogen
Mr_H = 1.00784/1000.0       # [kg/mol]
M_H = Mr_H/spc.N_A          # [kg] mass of hydrogen atom 
R_H = spc.R/Mr_H            # [J/kg/K] gas constant of hydrogen 
a0_H = 52.9e-12   # [m] Bohr radius of Hydrogen
Eion_H = np.array([13.598434599702]) # ionization energy of H atom in [eV]

# Argon
xi = 6                      # [#] number of optical electrons of argon
Mr_Ar = 39.948/1000.0       # [kg/mol]
M_Ar = Mr_Ar/spc.N_A        # [kg] mass of argon atom (6.63352088e-26 kg)
R_Ar = spc.R/Mr_Ar          # [J/kg/K] gas constant of argon R = 208.13 [J/kg/K]

RydEn = spc.Rydberg*spc.c*spc.h/spc.e # [eV]
a0 = 71e-12 # [m] Bohr radius of argon

# Electrons
Mr_e = 5.48579908782496e-7  # [kg/mol]
M_e = Mr_e/spc.N_A          # [kg] mass of electron
R_e = spc.R/Mr_e            # [J/kg/K] gas constant of electrons

# Argon Ions
Mr_ArIon = Mr_Ar - Mr_e     # [kg/mol]
M_ArIon = Mr_ArIon/spc.N_A  # [kg] mass of argon ion 
R_ArIon = spc.R/Mr_ArIon    # [J/kg/K] gas constant of argon ions

g_ion = 4.0 # There should be two ionizasion levels. What do we do then??

Eion = np.array([15.7596119]) # ionization energies of Ar in [eV]
Eion_Ar_1 = np.array([15.7596119]) # ionization energy of Ar in [eV]
Eion_Ar_2 = np.array([15.7596119+0.17749368]) # ionization energy of Ar  in [eV]



UNIVERSALGASCONSTANT = spc.R #8.314462618 # J mol^-1 K^-1

# MAXSPECIES = 200;
# UNIVERSALGASCONSTANT = 8.3144598;  // J * mol^(-1) * K^(-1)
# AVOGADRONUMBER = 6.0221409e+23;    // mol^(-1)
# BOLTZMANNCONSTANT = UNIVERSALGASCONSTANT / AVOGADRONUMBER;
# PLANCKCONSTANT = 6.62607015e-34; // m^2 kg / s
# VACUUMPERMITTIVITY = 8.8541878128e-12;
# ELECTRONCHARGE = 1.60218e-19;
# MOLARELECTRONCHARGE = ELECTRONCHARGE * AVOGADRONUMBER;
# ELECTRONMASS = 9.1093837015e-31; // kg
# qeOverkB = ELECTRONCHARGE / BOLTZMANNCONSTANT;

# IonizationEnergy_Argon = 13.598434599702; // eV


# gas_constant = 208.1 # [J/kg/K]   // For Argon
# specific_heat_ratio = 1.666666666; # For Monoatomic Gas
# gas_constant = 287.058; # [J/kg/K]   // For Air
# specific_heat_ratio = 1.4; # For Air

# gas_constant = 214.40995734; # ??? from the table. just a test.
# specific_heat_ratio = 1.4833993105634689; 

#----------------------------------------------------------------------------------
# Global constants
VacPermittivity = 8.8541878128e-12 # [F/m]

#----------------------------------------------------------------------------------


# Indices
iNe = -3 
iEh = -2
iEe = -1

#----------------------------------------------------------------------------------

"""
No. of ionization levels considered
"""
z = 1
