# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:04:49 2023

@author: Malamas Tsagkaridis
"""

import numpy as np
import matplotlib.pyplot as plt
import os


from Constants import *
from CRModel import CollisionalRadiativeModel

from CalcIonizationCrossSections import IonizationCrossSections
from ExcitationCrossSections import CharacteriseTransitions, ExcitationCrossSections
from MCDHFData import Read_MCDHF_Data

#----------------------------------------------------------------------------------
# raise SystemExit(0) 
#----------------------------------------------------------------------------------


"""
Initial conditions for bulk gas and electron gas
"""
T_e0 = 1.0              # [eV]   #T_g0*K_eV #  Electron temperature [eV]
# T_e0 = 1.57              # [eV]   #T_g0*K_eV #  Electron temperature [eV]


# # Glow discharge
# T_g0 = 300 #4500      # [K]          # Gas temperature
# p_0  = 0.4*spc.torr   # 101325 [Pa]  # Pressure

# ICP torch
T_g0 = 1.0*T_e0/K_eV #9000.0         # [K]  # Gas temperature
# T_g0 = 300.0        # [K]  # Gas temperature

p_0  = 101325.0       # [Pa] # Pressure
# p_0  = 133.322       # [Pa] # Pressure


#----------------------------------------------------------------------------------

# iflag = ExcitationCrossSections()
# IonizationCrossSections()

# os.chdir(homeDir)
# iflag = Read_MCDHF_Data('./Data/MCDHF',*p)
# os.chdir(homeDir)
    
#----------------------------------------------------------------------------------


# Instantiate solver class
os.chdir(homeDir)
# Ns = 43 +1 
# Ns = 30 + 1 +1 # ground state + excited states + ground ion state.  The pressure is assumed to be constant. Ideal gas law is included in the system of equations.
Ns = 14 + 1 +1 # ground state + excited states + ground ion state.  The pressure is assumed to be constant. Ideal gas law is included in the system of equations.
NT = 2 # add two equations for energies Eh, Ee
backgroundSpecieActivationFactor = 0
TimeIntegrationMethod = 'BE' # RK23, LSODA, BDF, Radau, BE

cr = CollisionalRadiativeModel(Ns, NT, T_e0, p_0, T_g0, 
                               backgroundSpecieActivationFactor, 
                               scenario=0, scheme=TimeIntegrationMethod)


#----------------------------------------------------------------------------------
############## Solver ##############
#----------------------------------------------------------------------------------

# TotalTime = 10.0
# TotalTime = 1e-3
TotalTime = 1e-4

MaxTimeStep = 1e-6

RelativeTolerance = 1e-6
AbsoluteTolerance = 1e-12

# SteadyState.terminal = True

time0 = 0.0
Nstep = 100
dt = (TotalTime-time0)/Nstep


# Default IC (overwritten below if we are restarting)
q, iniGuess = cr.initialConditions()
cr.U1 = iniGuess
## Initialize rest of state
cr.U0 = np.copy(cr.U1)
cr.U2 = np.copy(cr.U1)


cr.solve(time0, dt, TotalTime, Nstep, dt_max = MaxTimeStep,
         rtol = RelativeTolerance, atol = AbsoluteTolerance , 
         savedata=True, verbose=False)


plt.show()
raise SystemExit(0)
