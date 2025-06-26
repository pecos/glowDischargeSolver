import numpy as np
from scipy.interpolate import CubicSpline
import scipy.constants as spc

import csv
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
import h5py as h5
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

import logging

def is_sorted(arr):
    return np.all(np.diff(arr) >= 0)


# Define a nonlinear model function
def EMobModel(EN, a, b):
    return b * EN**(-a)

# Define a nonlinear model function
def ArIonModel(EN, a, b, c):
    # a = 0.007
    # b = 1.5
    # c = 0.33
    # Nmui_v_EN = 4 * 1e21 / (1 + (0.007 * EN  )**1.5)**0.33  
    Nmui_v_EN = 4.11 * 1e21 / (1 + (a * EN  )**b)**c  
    return Nmui_v_EN

# Define a nonlinear model function
def ArIon2Model(EN, a, b, c):
    Nmui_v_EN = 6.4859 * 1e21 / (1 + (a * EN  )**b)**c  
    return Nmui_v_EN

# Define a nonlinear model function
def EDiffModel(EN, a, b):
    # return b * EN**a
    return np.exp(np.log(b) + a**2 * np.log(EN)**2)

#--------------------------------------------------------------------------------------

class Reaction(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

class Diffusivity(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

class Mobility(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

#--------------------------------------------------------------------------------------

def setPsaapProperties_6Species(gam, inputV0, inputVDC, params, Ns, Nr, iSample):
    """Sets non-dimensional properties corresponding to Liu 2014 paper.

    Inputs:
      gam : Secondary electron emission coefficient
      params : chebSolver.modelParams class

    Outputs: None
      params data is overwritten using values from Liu 2014.
    """
    ###################################################################
    # User specified parameters (you may change these if you wish to
    # run a different scenario from Liu 2014)
    ###################################################################

    Pressure  = 1.0*spc.torr  # [Pa] 
    GasTemperature = 293.15 # [K]
    nAr = Pressure/GasTemperature/spc.k    # [#/m^3] Number density based on bulk temperature (not necessarily true density in two-temperature gas)

    # densities
    # nAr = 3.22e22     # background number density of Ar [1/m^3] (corresponds to p = 1 Torr)
    np0 = 8e16        # "nominal" electron density [1/m^3]

    # masses
    # me = 9.10938356e-31        # mass of an electron [kg]
    # me = 5.489e-4              # mass of an electron [u]
    me = 0.511e6                 # mass of an electron [eV/c2]
    # mAr = 39.948               # mass of an argon atom [u]
    # mAr = 39.948 * 1.66054e-27 # mass of an argon atom [kg]
    mAr = 37.2158e9              # mass of an argon atom [eV/c2]
    # u = 931.4941e6             # eV/c2
    c = 299792458                # speed of light [m/s]
    se = 40                      # momentum cross section [A^2]

    Mr_Ar = 39.948/1000.0       # [kg/mol]
    M_Ar = Mr_Ar/spc.N_A        # [kg] mass of argon atom (6.63352088e-26 kg)
    M_ArIon = M_Ar - spc.m_e    # [kg] mass of argon ion 
    

    # nominal electron energy
    e0 = 1.0  # [eV]

    # pressure
    p  = Pressure*1.5      # [J/m^3] *1.5 to convert it to energy (1 Torr)

    # gas energy at the wall
    # Tg0 = 0.038778    # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot
    Tg0 = 3/2*GasTemperature*spc.k/spc.e # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot

    # characteristics of driving voltage
    V0  = inputV0                 # amplitude of driving voltage [V]
    verticalShift = inputVDC      # DC voltage (vertical shift in driving voltage)
    tau = (1./13.56e6)             # period of driving voltage [s]
    L   = 2.00*0.005              # half-gap-width [m] (gap width is 2 cm)
    electrodeArea = np.pi*0.05**2 # electrode area [m^2] (electrode diameter = 0.1 m)

    # Set density and temperature floors (minimum values). Values in the simulations that 
    # are lower that the below are clipped.
    density_floor = 1e6/np0
    temperature_floor = 0.02526171245797859 # [eV] = 293.15 [K]
    energy_floor = 3.0/2.0 * temperature_floor * density_floor 

    # Chemistry parameters (charge number, Cv, Cp)
    Z = np.zeros(Ns+1); Cv = np.zeros(Ns+1);    Cp = np.zeros(Ns+1)
    Z[0]    = -1;       Cv[0]    = 3.0/2.0;     Cp[0]    = 5.0/2.0;     # E
    Z[1]    =  1;       Cv[1]    = 3.0/2.0;     Cp[1]    = 5.0/2.0;     # Ar+
    Z[2]    =  1;       Cv[2]    = 5.0/2.0;     Cp[2]    = 7.0/2.0;     # Ar2+
    Z[3]    =  0;       Cv[3]    = 5.0/2.0;     Cp[3]    = 7.0/2.0;     # Ar2m
    Z[4:Ns] =  0;       Cv[4:Ns] = 3.0/2.0;     Cp[4:Ns] = 5.0/2.0;     # Ar(m), Ar(r), Ar(4p), Ar
    Z[-1]   = -1;       Cv[-1]   = 3.0/2.0;     Cp[-1]   = 5.0/2.0;     # Ee electron energy 

    # transport parameters
    nmue = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
    nmum = 0.0
    nmur = 0.0
    nmu4p = 0.0
    #nmui = 8.0e19
    nmui = 4.65e19   # Transport coefficients from Lymberopoulos & Economou, 1993
    nDe  = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi  = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]
    nDm  = 2.42e18   # argon number density times AR(m) diffusivity [1/(cm*s)]
    nDr  = 2.42e18
    nD4p = 2.42e18


    # reaction parameters (NB: k_i = Ck*Ee^B*exp(-A/Ee))
    #                          Ee = 3/2*Te (Te in eV)
    #                          -> k_i = [Ck*(2/3)^B] * Ee^B * exp[-(3/2)*A/Ee]
    # nominal

    # pre-exponential factors [m^3/s]
    Ck = np.array([0.0,2.1e-15,5.0e-16,6.4e-16,2.1e-21,1.85e3,1.46e7,6.80e6,
                   0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                   5.0e-18,4.0e-19,0.0,0.0,0.0,0.0,2.5e-17,2.5e-17,1.0e-15,1.0e-15,
                   0.0,0.0,
                   2.5e-43, 1.36e-12, 7.34158788397042e-14, 9.78729818477016e-14, 9.78729818477016e-14, 6.3e-16, 
                   1.0e-44, 1.0e-14, 7.0e-16, 7.0e-16, 7.0e-16,1e6]) 


    # Ck = np.array([2.0e-13,2.1e-15,5.0e-16,6.4e-16,2.1e-21,1.85e3,1.46e7,6.80e6,
    #                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    #                5.0e-18,4.0e-19,0.0,0.0,0.0,0.0,2.5e-17,2.5e-17,1.0e-15,1.0e-15,0.0,0.0]) # pre-exponential factors [m^3/s]
    B  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0,
                   0, 0, -0.67, -0.61, -0.61, 0, 0, 0, 0, 0, 0, 0]) # Temperature Power
    A  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
                   0, 2.095, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]) # activation temperature [eV]
    dH = np.array([0.0,-7.541,-10.577,-7.393,0.0,0.0,0.0,0.0,11.577,11.725,13.168,15.76,-11.577,
                   4.183,0.148,1.592,2.592,-1.444,-1.592,-11.725,-0.148,1.444,0.0,0.0,-4.183,-4.035,
                   -2.592,-15.76,0.0,0.0,-8.985,-9.133,4.035,-13.168, 
                   0, 1.259, 1.259, 1.259, 1.259, -8.653, 0, 0.012237, -8.640763, -10.231763, -8.628526, 0]) # energy lost per electron due to ionization rxn [eV]
    dEps = np.array([0.0, 15.76, 14.501, 11.564763, 11.577, 11.725, 13.168, 0.0]) # E, AR+, AR(m), AR(r), AR(4p), AR # E, Ar+, Ar+2, Ar2, Ar(m), Ar(r), Ar(4p), Ar 

    # Dissociation energy of Ar2* in ground state is  0.012237 eV
    # Dissociation energy of Ar2+ is 1.3144 eV

    # BC parameters
    # ks = 1.19e7  # electron recombination rate [cm/s]
    ks = 1.366109824889323e7 # electron recombination rate [cm/s/eV]
    # ks = 1/4 * np.sqrt(2/3*spc.e * 8/np.pi/spc.m_e) * 100.0 # electron recombination rate [cm/s/eV] 

    ksion = 0.0#1/4 * np.sqrt(8*spc.k*GasTemperature/np.pi/M_ArIon) * 100.0 # ion rate [cm/s] 

    ksa = 1/4 * np.sqrt(8*spc.k*GasTemperature/np.pi/M_Ar) * 100.0 # atom rate [cm/s] 

    ###################################################################
    # Constants of nature (probably shouldn't change unless you have
    # root privileges on universe)
    ###################################################################
    qe   = spc.e #1.60217663e-19    # unit charge [C]
    eps0 = spc.epsilon_0 #8.86e-12  # permittivity of free space [F/m]
    kB   = spc.k #1.380649e-23      # Boltzmann constant [J/K]
    # kB   = 8.62e-5 # Boltzmann constant [eV/K]
    eV = qe/kB

    ###################################################################
    # Calculate non-dimensional parameters
    ###################################################################

    # 1) Convert input units to base SI (except eV)
    nDe  *= 100. # 1/(m*s)
    nDi  *= 100. # 1/(m*s)
    nDm  *= 100.
    nDr  *= 100.
    nD4p *= 100.
    nmue *= 100. # 1/(V*m*s)
    nmui *= 100. # 1/(V*m*s)
    ks   *= 0.01 # m/s
    ksion *= 0.01 # m/s
    ksa   *= 0.01 # m/s
    se   *= 1.0e-20  # m^2

    # 2) Compute "raw" transport parameters
    De  = nDe/nAr
    Di  = nDi/nAr
    Dm  = nDm/nAr
    Dr  = nDr/nAr
    D4p = nD4p/nAr

    mue = nmue/nAr
    mui = nmui/nAr
    mum = nmum/nAr
    mur = nmur/nAr
    mu4p = nmu4p/nAr

    # 3) Compute non-dimensional properties required by solver
    De    = De*tau/(L*L)
    Di    = Di*tau/(L*L)
    Dm    = Dm*tau/(L*L)
    Dr    = Dr*tau/(L*L)
    D4p   = D4p*tau/(L*L)

    mue   = mue*V0*tau/(L*L)
    mui   = mui*V0*tau/(L*L)
    mum   = mum*V0*tau/(L*L)
    mur   = mur*V0*tau/(L*L)
    mu4p  = mu4p*V0*tau/(L*L)

    Ck[0:4]   *= tau*np0
    Ck[4]     *= tau*nAr
    Ck[5:8]   *= tau
    Ck[8:12]  *= tau*nAr
    Ck[12:24] *= tau*np0
    Ck[24:28] *= tau*np0*np0
    Ck[28:30] *= tau*nAr
    Ck[30:34] *= tau*np0
    Ck[34]    *= tau*nAr*nAr
    Ck[35:40] *= tau*np0
    Ck[40]    *= tau*nAr*nAr
    Ck[41:45] *= tau*np0
    Ck[45]    *= tau
    
    A        = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH       = dH/e0
    qStar    = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha    = qe*np0*L*L/(V0*eps0)
    ks       = ks*tau/L
    ksion    = ksion*tau/L
    ksa      = ksa*tau/L
    p0       = p/qe/np0

    # non-dimensional parameter for the effective electric field for ions
    vmStar = V0 * tau**2 / L**2 * qe / M_ArIon

    ThermalConductivity = 17.7e-3 # [W/m/K] at 300K
    
    kappaB   = (2/3)*tau/L**2*ThermalConductivity/np0/kB
        
    # kappaB   = 4.878171165833662*1.6129 # non-dimensional thermal conductivity of background specie
    #                            # (2/3)*tau/L**2*Kb/np0/kB,
    #                            # where Kb is the thermal conductivity of background specie

    params.beta = np.array([[0,1,1,1,0,0,0,0,1,1,1,2,1,2,1,1,2,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,2,1, 0,1,0,0,0,1,0,1,1,1,1,0],                     # E
                            [0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0, 0,1,0,0,0,0,0,0,0,0,0,0],                     # AR+
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,0,0,1,0,0,1,1,1,0],                     # AR2+
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,0,0,0,0,0],                     # AR2*                            
                            [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0, 0,0,0,1,0,0,0,1,0,0,0,0],                     # AR(m)
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0,0,0,0],                     # AR(r)
                            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0],                     # AR(4p)
                            [2,1,1,1,2,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1, 1,1,2,1,1,0,1,1,1,1,2,2]], dtype=np.int64)    # AR

    params.alfa = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,0,0,0,1,1, 0,1,1,1,1,0,0,1,0,0,0,0],                     # E
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0,0,0,0],                     # AR+
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,1,1,1,1,0,0,0,0,0,0,0],                     # AR2+
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,1,1,2,1],                     # AR2*                            
                            [2,1,0,2,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0, 0,0,0,0,0,2,1,0,1,0,0,0],                     # AR(m)
                            [0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0, 0,0,0,0,0,0,0,0,0,0,0,0],                     # AR(r)
                            [0,0,2,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1, 0,0,0,0,0,0,0,0,0,1,0,0],                     # AR(4p)
                            [0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 2,0,0,0,0,0,2,0,0,0,0,0]], dtype=np.int64)    # AR
    
    # Rxn1:  2AR(m)         ->   2AR
    # Rxn2:  AR(m) + AR(r)  ->   E + AR+ + AR
    # Rxn3:  2AR(4p)        ->   E + AR+ + AR
    # Rxn4:  2AR(m)         ->   E + AR+ + AR
    # Rxn5:  AR(m) + AR     ->   2AR
    # Rxn6:  AR(r)          ->   AR + hv
    # Rxn7:  AR(4p)         ->   AR(m) + hv
    # Rxn8:  AR(4p)         ->   AR(r) + hv
    # Rxn9:  E + AR         ->   E + AR(m)
    # Rxn10: E + AR         ->   E + AR(r)
    # Rxn11: E + AR         ->   E + AR(4p)
    # Rxn12: E + AR         ->   2E + AR+
    # Rxn13: E + AR(m)      ->   E + AR
    # Rxn14: E + AR(m)      ->   2E + AR+
    # Rxn15: E + AR(m)      ->   E + AR(r)
    # Rxn16: E + AR(m)      ->   E + AR(4p)
    # Rxn17: E + AR(4p)     ->   2E + AR+
    # Rxn18: E + AR(4p)     ->   E + AR(r)
    # Rxn19: E + AR(4p)     ->   E + AR(m)
    # Rxn20: E + AR(r)      ->   E + AR
    # Rxn21: E + AR(r)      ->   E + AR(m)
    # Rxn22: E + AR(r)      ->   E + AR(4p)
    # Rxn23: E + AR+        ->   AR(m) + hv
    # Rxn24: E + AR+        ->   AR(4p) + hv
    # Rxn25: 2E + AR+       ->   E + AR(m)
    # Rxn26: 2E + AR+       ->   E + AR(r)
    # Rxn27: 2E + AR+       ->   E + AR(4p)
    # Rxn28: 2E + AR+       ->   E + AR
    # Rxn29: AR + AR(4p)    ->   AR + AR(m)
    # Rxn30: AR + AR(4p)    ->   AR + AR(r)
    # Rxn31: AR(m) + AR(4p) ->   E + AR+ + AR
    # Rxn32: AR(r) + AR(4p) ->   E + AR+ + AR
    # Rxn33: E + AR(r)      ->   2E + AR+
    # Rxn34: E + AR(4p)     ->   E + AR

    rxnNameDict = { 0: "2Ar(m) => 2Ar",
                    1: "Ar(m) + Ar(r) => E + Ar+ + Ar",
                    2: "2Ar(4p) => E + Ar+ + Ar",
                    3: "2Ar(m) => E + Ar+ + Ar",
                    4: "Ar(m) + Ar => 2Ar",
                    5: "Ar(r) => Ar",
                    6: "Ar(4p) => Ar(m)",
                    7: "Ar(4p) => Ar(r)",
                    8: "Excitation_Metastable",
                    9: "Excitation_Resonant",
                   10: "Excitation_4p",
                   11: "Ionization",
                   12: "DeExcitation_Metastable",
                   13: "StepIonization_Metastable",
                   14: "E + Ar(m) => E + Ar(r)",
                   15: "E + Ar(m) => E + Ar(4p)",
                   16: "StepIonization_4p",
                   17: "E + Ar(4p) => E + Ar(r)",
                   18: "E + Ar(4p) => E + Ar(m)",
                   19: "DeExcitation_Resonant",
                   20: "E + Ar(r) => E + Ar(m)",
                   21: "E + Ar(r) => E + Ar(4p)",
                   22: "E + Ar+ => Ar(m)",
                   23: "E + Ar+ => Ar(4p)",
                   24: "3BdyRecomb_Metastable",
                   25: "3BdyRecomb_Resonant",
                   26: "3BdyRecomb_4p",
                   27: "3BdyRecomb_Ground",
                   28: "Ar + Ar(4p) => Ar + Ar(m)",
                   29: "Ar + Ar(4p) => Ar + Ar(r)",
                   30: "Ar(m) + Ar(4p) => E + Ar+ + Ar",
                   31: "Ar(r) + Ar(4p) => E + Ar+ + Ar",
                   32: "StepIonization_Resonant",
                   33: "DeExcitation_4p"}

    # 4) Set values in params class
    params.Z[:]     = Z[:]
    params.Cv[:]    = Cv[:]
    params.Cp[:]    = Cp[:]

    params.D[0]    = De
    params.D[1]    = Di
    params.D[2]    = Di
    params.D[3]    = Dm
    params.D[4:]   = Dm
    params.D[-1]   = 5.0/3.0*De # Electron Energy

    params.mu[0]   = mue
    params.mu[1]   = mui
    params.mu[2]   = mui
    params.mu[3]   = mum
    params.mu[4:]  = mum
    params.mu[-1]  = 5.0/3.0*mue # Electron Energy

    params.A[:]    = Ck[:]
    params.B[:]    = B[:]
    params.C[:]    = A[:]

    # Account for the 2/3 term to convert from electron temperature to electron energy
    for i in range(len(params.A)):
        params.A[i] *= (2/3)**(params.B[i])

    # Non-dimensionalization parameters
    params.np0         = np0   # "nominal" electron density [1/m^3]
    params.nAr         = nAr
    params.nAronp0     = nAr / np0
    params.tau         = tau
    params.tauOvernp0  = tau/np0
    params.tauOvernAr  = tau/nAr

    params.dH[:]   = dH[:]
    params.dEps[:] = dEps[:]
    params.qStar   = qStar
    params.alpha   = alpha
    params.ks      = ks
    params.ksion   = ksion
    params.ksa     = ksa
    params.gam     = gam
    params.kappaB  = kappaB
    params.p0      = p0
    params.Tg0     = Tg0
    params.EC      = 2.0 * me / mAr \
        * np.sqrt(16.0 * (me + mAr) * e0 * c**2
                  / (3.0 * np.pi * me * mAr)) * se * nAr * tau
    # params.EC = 2.0 * me / mAr * 3.8e9 * tau

    params.verticalShift = verticalShift / V0
    params.vmStar  = vmStar

    # Parameters needed to compute the current with dimensions
    params.V0Ltau  = V0 / (L * tau)
    params.V0L     = V0 / L
    params.LLV0tau = (L*L) / (V0*tau)
    params.tauL    = L / tau
    # params.np0     = np0           # "nominal" electron density [1/m^3]
    params.qe      = qe            # unit charge [C]
    params.eps0    = eps0          # unit charge [C]
    params.eArea   = electrodeArea # electrode area [m^2]

    # Floor
    params.density_floor = density_floor
    params.temperature_floor = temperature_floor
    params.energy_floor = energy_floor


    reactionExpressionslist = [f"{params.A[i]} * energy**{params.B[i]} * np.exp(-{params.C[i]} / energy)" for i in range(Nr) ]

    reactionTExpressionslist = [f"{params.A[i]} * (energy**({params.B[i]}-1)) * np.exp(-{params.C[i]}/energy) * ({params.B[i]} + {params.C[i]}/energy)" for i in range(Nr) ]


    reactionExpressionTypelist =  np.array([False,False,False,False,False,False,False,False, # Rxns 1-8
                                           True,True,True,True,True,True,True,True,True,True,True,True,True,True, # Rxns 9-22
                                           False,False,True,True,True,True,False,False,False,False,True,True, # Rxns 23-34
                                           False,False,False,False,False,False,False,False,False,False,False,False]) # Rxns 35-46
    
    thresholded_rxn = np.array([False, False, False, False, False, False, False, False,
                                True, True, True, True, False, True, True, True, True, False, False, False, False,
                                True, False, False, False, False, False, False, False, False, False, False, True,
                                False,
                                False,False,False,False,False,False,False,False,False,False,False,False])
    reactionsList = []
    LOGFilename = 'interpolationSample.log'
    f = open(LOGFilename, 'w')
    

    for i in range(Nr):
        if reactionExpressionTypelist[i]:
            if i < 14 or i == 16 or i == 19 or i > 23:
                f = h5.File("./BOLSIGChemistry_NominalRates/{0:s}.h5".format(rxnNameDict[i]), 'r')
                dataset = f["table"]
            else:
                f = h5.File("./BOLSIGChemistry_NominalRates/StepExcitation.h5", 'r')
                dataset = f[rxnNameDict[i]]

            Te = dataset[:,0]
            Te /= 11604
            rateCoeff = dataset[:,1]
            if i > 23 and i < 28:
                rateCoeff /= 6.022e23**2
            else:
                rateCoeff /= 6.022e23

            ## Removing BOLSIG failures
            fail_inds = []
            for j in range(len(rateCoeff)):
                if rateCoeff[j] == 0.0 and j > np.nonzero(rateCoeff)[0][0]:
                    fail_inds.append(j)


 
            #Te = np.delete(Te, fail_inds)
            #rateCoeff = np.delete(rateCoeff, fail_inds)
            if len(fail_inds) != 0:
                rateCoeff[0:fail_inds[-1]] = 0.0


            # Sorting mean energy array and rate coefficient array based on
            # the mean energy array.
            Teinds = Te.argsort()
            rateCoeff = rateCoeff[Teinds]
            Te = Te[Teinds]
            

            # Find duplicates
            TeDuplicateinds = np.where(np.abs(np.diff(Te, axis=0)) > 0.0)
            TeDuplicateindsForLog = np.where(np.abs(np.diff(Te, axis=0)) == 0.0)
            rateCoeff = rateCoeff[TeDuplicateinds]
            Te = Te[TeDuplicateinds]


            # Nondimensionalization of mean energy.
            Te *= 1.5

            if thresholded_rxn[i] == True:
                # Find first non-zero value of the coefficient rate.
                I = np.nonzero(rateCoeff)

                diffRateCoeff = [j-i for i, j in zip(rateCoeff[:-1], rateCoeff[1:])]
                diffTe = [j-i for i, j in zip(Te[:-1], Te[1:])]

                Monotonicity = np.asarray([j/i for i, j in zip(diffTe, diffRateCoeff)])
                Monotonicity = np.insert(Monotonicity, 0, 0.0, axis=0)

                Nan = np.isnan(Monotonicity)
                Inf = np.isinf(Monotonicity)
                indexPositive = np.where(Monotonicity>0.0)
                #if thresholded_rxn[i] == True:
                #    indexPositive = np.where(Monotonicity>0.0)
                #else:
                #    indexPositive = np.where(Monotonicity<0.0)
                Positive = np.full(Monotonicity.shape, False, dtype=bool)
                Positive[indexPositive] = True
            
                indices = Nan + Inf + Positive
                
                # lastFalse = np.nonzero(rateCoeff)[0][0]
                # for k in range(len(Te)):
                #   if Te[k] < 6.0 and indices[k] == False:
                #      lastFalse = k + 2
                lastFalse = np.nonzero(rateCoeff)[0][0]
                # Transformation to log scale.
                TeLog = np.log(Te)

                # Compute the slope of the rate coefficient between its first two non-zero values.
                # Finite differences are used.
                dydx = (rateCoeff[lastFalse + 1] - rateCoeff[lastFalse]) \
                 / (Te[lastFalse + 1] - Te[lastFalse])

                # Arrhenius form: kf = A * exp(-C / Te)
                # C = Te[lastFalse]**2.0*dydx / rateCoeff[lastFalse]
                C = Te[lastFalse+1]*Te[lastFalse]*np.log(rateCoeff[lastFalse+1]/rateCoeff[lastFalse])**1.5/(Te[lastFalse+1]-Te[lastFalse])

                # Compute pre-exponential coefficient, A, in log scale.
                ALog = np.log(rateCoeff[lastFalse]) + C / Te[lastFalse]

                # Transform rate coefficient in log scale.
                rateCoeffLog = np.zeros(rateCoeff.shape)
                rateCoeffLog[lastFalse:] = np.log(rateCoeff[lastFalse:])
                # For the troublesome values, we use the Arrhenius form.
                rateCoeffLog[0:lastFalse] = ALog - C / Te[0:lastFalse]
            
                Te_add = np.linspace(1e-4, Te[0]*0.99, 100)
                TeLog = np.concatenate((np.log(Te_add), TeLog))
                rateCoeffLog_add = np.zeros(100)
                for m in range(len(rateCoeffLog_add)):
                    fac = 0.999**(100-m)
                    rateCoeffLog_add[m] = rateCoeffLog[0]/fac
                rateCoeffLog = np.concatenate((rateCoeffLog_add, rateCoeffLog))
            else:
                TeLog = np.log(Te)
                rateCoeffLog = np.log(rateCoeff)

            # Nondimensionalization in log scale.
            if (i < 12):
                rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
            elif i > 23 and i < 28:
                rateCoeffLog += - np.log(1.0/tau) + 2*np.log(np0)
            else:
                rateCoeffLog += - np.log(1.0/tau) + np.log(np0)

            # Interpolation in log scale.
            reactionExpressionsLog = CubicSpline(TeLog, rateCoeffLog)

            #fig,ax = plt.subplots()
            #ax.plot(np.exp(TeLog), np.exp(rateCoeffLog), marker = 'o', markerfacecolor = 'None', label = 'Rate Coefficient - Corrected')
            #ax.plot(Te, rateCoeff, label = 'Rate Coefficient - Original')
            #ax.plot(TeLog, rateCoeffLog)
            #ax.legend()
            #plt.show()

            # Gradient in log scale
            reactionTExpressionsLog = CubicSpline.derivative(reactionExpressionsLog)

            reaction = Reaction(rxnAlfa = params.alfa[:,[i]], rxnBeta = params.beta[:,[i]],
                                rxnBolsig = reactionExpressionTypelist[i],
                                kf_log = reactionExpressionsLog,
                                kf_T_log = reactionTExpressionsLog)
            reactionsList.append(reaction)

            logging.basicConfig(filename=LOGFilename)
            logging.warning('Interpolation info (Reaction %s):', i + 1)
            logging.warning('First non-zero entry in the rate coefficient: %s', I[0][0])
            logging.warning('Monotonicity of the rate coefficient start from entry: %s', lastFalse)
            logging.warning('Position of possible duplicates in mean energy array: %s',
                          TeDuplicateindsForLog[0])

        else:
            rxn   = eval("lambda energy :" + reactionExpressionslist[i])
            rxn_T = eval("lambda energy :" + reactionTExpressionslist[i])

            reaction = Reaction(rxnAlfa = params.alfa[:,[i]], rxnBeta = params.beta[:,[i]],
                                rxnBolsig = reactionExpressionTypelist[i],
                                kf = rxn, kf_T = rxn_T)
            reactionsList.append(reaction)

    params.reactionsList = reactionsList






    #-------------------------------------------------------------------------------
    #########   Electron and Ion Transport Data  #########  
    diffList = []
    muList = []

    EN_Td = np.logspace(np.log10(1e-2),np.log10(10000),2000,dtype=np.float64) #  Electric field / N [Td]         
    Te_eV = np.linspace(0.1,20,3000,dtype=np.float64)
    threshold_Te0 = 1.5    
    indices_Te_eV0 = np.searchsorted(Te_eV, threshold_Te0)


    #-------------------------------------------------------------------------------
    ###### Bolsig+
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_3.h5", 'r')
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_Biagi_BSR_GlowDischarge.h5", 'r')
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_2.5Torr.h5", 'r')
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_5Torr.h5", 'r')
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_1Torr_2.h5", 'r')
    transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_1Torr_3.h5", 'r')


    ElectricFieldData = transport["reduced_electric_field"] 
    EN = ElectricFieldData[:,1]
    Te_trans = ElectricFieldData[:,0]
    Te_trans /= eV 
    indices_Te0 = np.searchsorted(Te_trans, threshold_Te0)

    mobilityData = transport["mobility"] 
    Nmue_v_Te = mobilityData[:,1]

    diffusivityData = transport["diffusivity"] 
    NDe_v_Te = diffusivityData[:,1]


    fig, ax = plt.subplots()
    ax.set_title('Electric field / N (Td)')
    ax.set_ylabel(r"$ E / N_{Ar} \, $ [$ \, Td$]")
    ax.set_xlabel('Te [eV]')
    ax.plot(Te_trans,EN, marker = '.', label = 'bolsig+')
    # energy_fit = 2.5 * np.sqrt(EN) * (1+(EN/5)**2)**(-0.2) * (1+(EN/1000)**2)**0.7
    # ax.plot(EN,energy_fit*2.0/3.0, marker = '.', label = 'Empirical fit')
    ax.legend()
    



    ###### LXCat

    fileName = "./Data/ElectronEnergy/LXCat/Energy.csv"
    Data = pd.read_csv(fileName)
    EN_LXCat = Data.iloc[:,0].to_numpy('float64')  # [Td]
    Ee_v_EN = Data.iloc[:,1].to_numpy('float64') # [eV]

    sorted_indices = np.argsort(EN_LXCat)
    EN_LXCat = EN_LXCat[sorted_indices]
    Ee_v_EN = Ee_v_EN[sorted_indices]

    threshold_EN = 0.01
    indices_EN = np.searchsorted(EN_LXCat, threshold_EN)

    EN_LXCat = EN_LXCat[indices_EN:]
    Ee_v_EN = Ee_v_EN[indices_EN:]

    # Perform nonlinear least squares fit
    # popt, pcov = curve_fit(EDiffModel, EN_LXCat[-15:], Ee_v_EN[-15:], p0=(0.14629452,3.08078198))
    popt = [0.14629452,3.08078198]
    # popt, pcov = curve_fit(EDiffModel, EN_LXCat[-15:], Ee_v_EN[-15:], p0=(0.219128,1.77595))
    # popt = [0.21540825, 1.83724349]
    # Evaluate the fitted model
    EN_Td_tmp = np.logspace(np.log10(EN_LXCat[-1]+0.1),np.log10(10000),10,dtype=np.float64)#  Electric field / N [Td]         
    Ee_v_EN_tmp = EDiffModel(EN_Td_tmp, *popt)

    EN_Td_data = np.concatenate((EN_LXCat, EN_Td_tmp)) # Combine datasets
    Ee_v_EN_data = np.concatenate((Ee_v_EN, Ee_v_EN_tmp)) # Combine datasets
    Ee_v_EN_interp = np.interp(EN_Td,EN_Td_data,Ee_v_EN_data)
    Ee_v_EN_interp_smoothed = uniform_filter1d(Ee_v_EN_interp, size=40)

    # fig,ax = plt.subplots()
    # ax.set_title('Energy.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$E_e \, $ [$ \, eV$]")
    # # ax.scatter(EN_LXCat, Ee_v_EN, marker = '.', label = 'raw')
    # # ax.scatter(EN_LXCat, Ee_v_EN, marker = '.', label = '3/2 raw')
    # ax.scatter(EN_Td, Ee_v_EN_interp, marker = '.', label = 'Fit')
    # # ax.scatter(EN_Td, Ee_v_EN_interp_smoothed, marker = '.', label = 'Fit - smoothed')
    # ax.scatter(EN, Te_trans, marker = '.', label = 'Bolsig+')
    # ax.legend()
    # ax.loglog()

    # fig,ax = plt.subplots()
    # ax.set_title('Energy.')
    # ax.set_ylabel('E/nAr [Td]')
    # ax.set_xlabel(r"$E_e \, $ [$ \, eV$]")
    # ax.scatter(Ee_v_EN,EN_LXCat, marker = '.', label = 'raw')
    # # ax.scatter(Ee_v_EN_interp, EN_Td, marker = '.', label = 'Fit')
    # ax.scatter(Ee_v_EN_interp_smoothed, EN_Td, marker = '.', label = 'Fit - smoothed')
    # ax.scatter(Te_trans, EN, marker = '.', label = 'Bolsig+')
    # ax.legend()
    # ax.loglog()
















    #-------------------------------------------------------------------------------
    ######### Electron Mobility  #########

    # From Bolsig+
    Nmue_v_Te_interp = np.interp(Te_eV,Te_trans,Nmue_v_Te)
    Nmue_v_Te_interp[0:indices_Te_eV0] = Nmue_v_Te_interp[indices_Te_eV0]
    Nmue_v_Te_interp = uniform_filter1d(Nmue_v_Te_interp, size=30)

    # mue_interp = (Nmue_v_Te_interp[:]/nAr)*V0*tau/(L*L)
    # mue_spline = CubicSpline(Te_eV, mue_interp)
    # mue_Te_spline = CubicSpline.derivative(mue_spline)
    # mobility = Mobility(interpolate = True, mu_expression = mue_spline, mu_T_expression = mue_Te_spline)
    # muList.append(mobility)
  
    # fig, ax = plt.subplots()
    # ax.set_title('Mobility Coef.')
    # ax.set_xlabel('Te [eV]')
    # ax.set_ylabel(r"$\mu_e \, $ [$ \, m^{2}/V/s$]")
    # # ax.loglog(Te_trans, Nmue_v_Te[:]/nAr, marker = 'o', label = 'Nominal')
    # ax.plot(Te_eV, mue_interp, marker = '.', label = 'raw')
    # # ax.plot(Te_trans, mue_Te_spline(Te_trans), marker = '*', label = 'raw - grad')
    # plt.axhline(y=params.mu[0], color='k', linestyle='--')
    # ax.legend()



    # From LXCat
    fileName = "./Data/ElectronMobility/LXCat/ElectronMobility.csv"
    Electron_Data = pd.read_csv(fileName)
    EN_LXCat_1 = Electron_Data.iloc[:,0].to_numpy('float64')  # [Td]
    Nmue_v_EN_1 = Electron_Data.iloc[:,1].to_numpy('float64') # [1/(m V s)]
    EN_LXCat_1 = EN_LXCat_1[0:20]; Nmue_v_EN_1 = Nmue_v_EN_1[0:20]
    EN_LXCat_2 = Electron_Data.iloc[:,2].to_numpy('float64')  # [Td]
    Nmue_v_EN_2 = Electron_Data.iloc[:,3].to_numpy('float64') # [1/(m V s)]
    EN_LXCat = np.concatenate((EN_LXCat_1, EN_LXCat_2)) # Combine datasets
    Nmue_v_EN = np.concatenate((Nmue_v_EN_1, Nmue_v_EN_2)) # Combine datasets
    sorted_indices = np.argsort(EN_LXCat)
    EN_LXCat = EN_LXCat[sorted_indices]
    Nmue_v_EN = Nmue_v_EN[sorted_indices]
    Nmue_v_EN_interp = np.interp(EN_Td,EN_LXCat,Nmue_v_EN)

    # # Perform nonlinear least squares fit
    popt, pcov = curve_fit(EMobModel, EN_LXCat_2[:33], Nmue_v_EN_2[:33], p0=(0.8246723166929772,3.098899585859967e+24))
    # print(popt)
    # # popt = [8.42974577e-01, 2.95712109e+24]
    # Evaluate the fitted model
    EN_Td_tmp = np.logspace(np.log10(EN_LXCat_1[9]),np.log10(EN_LXCat_2[33]),40,dtype=np.float64)#  Electric field / N [Td]         
    Nmue_v_EN_tmp = EMobModel(EN_Td_tmp, *popt)

    EN_LXCat_2_tmp = EN_LXCat_2[34:]
    Nmue_v_EN_2_tmp = uniform_filter1d(Nmue_v_EN_2[34:], size=1)
    
    EN_Td_data = np.concatenate((EN_Td_tmp, EN_LXCat_2_tmp)) # Combine datasets
    Nmue_v_EN_data = np.concatenate((Nmue_v_EN_tmp, Nmue_v_EN_2_tmp)) # Combine datasets    
    
    Nmue_v_EN_interp = np.interp(EN_Td,EN_Td_data,Nmue_v_EN_data)
    Nmue_v_EN_interp_smoothed = uniform_filter1d(Nmue_v_EN_interp, size=40)

    # threshold_EN0 = 0.5    
    # indices_EN0 = np.searchsorted(EN_Td, threshold_EN0)
    # Nmue_v_EN_interp_cliped = np.copy(Nmue_v_EN_interp)
    # Nmue_v_EN_interp_cliped[0:indices_EN0] = Nmue_v_EN_interp_cliped[indices_EN0]
    # Nmue_v_EN_interp_cliped = uniform_filter1d(Nmue_v_EN_interp_cliped, size=50)
    # Nmue_v_EN_interp_cliped = uniform_filter1d(Nmue_v_EN_interp_cliped, size=50)

    # mue_interp = (Nmue_v_EN_interp_cliped[:]/nAr)*V0*tau/(L*L)
    # EN_interp = EN_Td / (1e21 * params.V0L/params.nAr) * 2/3 # We multiply with 2/3 here to make it compatible with  mobility_U function.
    # mue_spline = CubicSpline(EN_interp, mue_interp)
    # mue_EN_spline = CubicSpline.derivative(mue_spline)
    # mobility = Mobility(interpolate = True, mu_expression = mue_spline, mu_T_expression = mue_EN_spline)
    # muList.append(mobility)


    # fig, ax = plt.subplots()
    # ax.set_title('Electron Mobility Coef.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$\mu_e \cdot N \, $ [$ \, \frac{1}{m V s}$]")
    # ax.loglog(EN, Nmue_v_Te, marker = '.', label = 'Bolsig+')
    # ax.loglog(EN_LXCat_1, Nmue_v_EN_1, marker = '.', label = 'Pack and Phelps 1961')
    # ax.loglog(EN_LXCat_2, Nmue_v_EN_2, marker = '.', label = ' J L Hernandez-Avila et al 2004')
    # # ax.loglog(EN_LXCat_unique, Nmue_v_EN_unique, marker = '.', label = ' LXCat')
    # ax.loglog(EN_Td, Nmue_v_EN_interp, marker = '.', label = ' interp')
    # # ax.loglog(EN_LXCat_unique, Nmue_v_EN_spline(EN_LXCat_unique), marker = '.', label = ' LXCat')
    # # ax.loglog(EN_Td, Nmue_v_EN_spline(EN_Td), marker = '.', label = ' LXCat')
    # # Nmue_v_EN_spline_dev = CubicSpline.derivative(Nmue_v_EN_spline)
    # # ax.plot(EN_LXCat_unique[8:], Nmue_v_EN_spline_dev(EN_LXCat_unique[8:]), marker = '.', label = ' LXCat')
    # ax.loglog(EN_Td, Nmue_v_EN_interp, marker = '.', label = ' fit')
    # # ax.loglog(EN_Td, Nmue_v_EN_interp_cliped, marker = '.', label = 'fit - clipped and smoothed ')
    # plt.axhline(y=params.mu[0]*nAr/(V0*tau/(L*L)), color='k', linestyle='--')
    # ax.legend()






    # Nmue_v_Te_interp_2 = np.interp(Te_eV,Ee_v_EN_interp,Nmue_v_EN_interp)
    # Nmue_v_EN_interp_2_smoothed = uniform_filter1d(Nmue_v_Te_interp_2, size=10)

    Nmue_v_Te_interp_smoothed_2 = np.interp(Te_eV,Ee_v_EN_interp_smoothed,Nmue_v_EN_interp_smoothed)
    Nmue_v_Te_interp_smoothed_2_smoothed = uniform_filter1d(Nmue_v_Te_interp_smoothed_2, size=10)

    mue_interp = (Nmue_v_Te_interp_smoothed_2_smoothed[:]/nAr)*V0*tau/(L*L)
    mue_spline = CubicSpline(Te_eV, mue_interp)
    mue_Te_spline = CubicSpline.derivative(mue_spline)
    mobility = Mobility(interpolate = True, mu_expression = mue_spline, mu_T_expression = mue_Te_spline)
    muList.append(mobility)
  

    # fig, ax = plt.subplots()
    # ax.set_title('Mobility Coef.')
    # ax.set_xlabel('Te [eV]')
    # ax.set_ylabel(r"$\mu_e \, $ [$ \, m^{2}/V/s$]")
    # # ax.loglog(Te_trans, Nmue_v_Te[:]/nAr, marker = 'o', label = 'Nominal')
    # ax.plot(Te_eV, Nmue_v_Te_interp, marker = '.', label = 'Bolsig+')
    # ax.loglog(Ee_v_EN_interp, Nmue_v_EN_interp, marker = '.', label = ' LXCat')
    # # ax.loglog(Ee_v_EN_interp_smoothed, Nmue_v_EN_interp_smoothed, marker = '.', label = ' LXCat smoothed')
    # # ax.loglog(Te_eV, Nmue_v_Te_interp_2, marker = '.', label = ' LXCat 2')
    # # ax.loglog(Te_eV, Nmue_v_EN_interp_2_smoothed, marker = '.', label = ' LXCat smoothed 2')
    # ax.loglog(Te_eV, Nmue_v_Te_interp_smoothed_2_smoothed, marker = '.', label = ' LXCat smoothed 2 smoothed')
    # # ax.plot(Te_trans, mue_Te_spline(Te_trans), marker = '*', label = 'raw - grad')
    # plt.axhline(y=params.mu[0]*nAr/(V0*tau/(L*L)), color='k', linestyle='--')
    # ax.legend()



    #-------------------------------------------------------------------------------
    #########  Ion Mobility. #########
    P0 = 101325; T0 = 273.15
    N0 = P0/T0/spc.k

    # Cold-cathode discharges and breakdown in argon: surface and gas phase production of 
    # secondary electrons (A V Phelps and Z Lj Petrovic ́ 1999)
    # Nmui_v_Te = 4 * 1e21 / (1 + (22.1 * 1e29 * EN_Td *1e-21))**0.33    # [1/(V*m*s)]
    Efield = EN_Td *1e-21 * nAr
    DriftVelocity_Ion = 4 * EN_Td / (1 + (0.007 * EN_Td  )**1.5)**0.33 # [m/s]
    Nmui_v_EN = DriftVelocity_Ion * nAr / Efield    
    K0 = Nmui_v_EN/nAr  # [m^2/(V*s)]

    # Determination of mobility and diffusion coefficients of Ar+ and Ar2+ ions in argon gas (Jasmiyanaa et al  2020)
    EN_Td_exp = np.array([34.49, 39.93, 45.81, 56.44, 60.55, 61.84, 67.85, 73.52, 79.25, 85.74, 92.25, 
                          94.95, 103.73, 108.32, 116.58, 117.59, 133.51, 138.43, 152.3, 172.97, 195.45,
                          1.00E+03, 2.00E+03, 3.00E+03, 5.00E+03, 1.00E+04, 2.00E+04, 3.00E+04, 5.00E+04, 1.00E+05]) # [Td]
    K02 = np.array([1.494, 1.486, 1.445, 1.432, 1.423, 1.411, 1.392, 1.376, 1.365, 1.341, 
                   1.322, 1.337, 1.291, 1.274, 1.246, 1.259, 1.218, 1.196, 1.178, 1.141, 1.097,
                   6.10E-01, 4.42E-01, 3.69E-01, 2.94E-01, 2.15E-01, 1.59E-01, 1.33E-01, 1.07E-01, 8.03E-02]) # [cm^2/V/s] at STD
    Nmui_v_EN_2 = K02 * 1e-4 * N0 # [1/( m V s)] 
    # K02 *= 101325/Pressure * GasTemperature/273.15 * 1e-4 # [m^2/V/s] 
    K02 = Nmui_v_EN_2/nAr # [m^2/V/s] 


    # From LXCat
    fileName = "./Data/IonMobility/LXCat/IonMobility.csv"
    Data = pd.read_csv(fileName)
    EN_LXCat = Data.iloc[:,0].to_numpy('float64')  # [Td]
    K03 = Data.iloc[:,1].to_numpy('float64') # [cm2/V s]  at STD?
    Nmui_v_EN_3 = K03 * 1e-4 * N0 # [1/( m V s)] 
    K03 *= 101325/Pressure * 1e-4 # [m^2/V/s] 



    EN_Td_data = np.concatenate((EN_Td_exp, EN_LXCat)) # Combine datasets
    Nmui_v_EN_data = np.concatenate((Nmui_v_EN_2, Nmui_v_EN_3)) # Combine datasets
    sorted_indices = np.argsort(EN_Td_data)
    EN_Td_data = EN_Td_data[sorted_indices]
    Nmui_v_EN_data = Nmui_v_EN_data[sorted_indices]
    # Perform nonlinear least squares fit
    # popt, pcov = curve_fit(ArIonModel, EN_Td_data, Nmui_v_EN_data, p0=(0.007,1.5,0.33))
    # print(popt)
    popt = [0.00891562, 1.59508181, 0.27209231]
    # Evaluate the fitted model
    Nmui_v_EN_fit = ArIonModel(EN_Td, *popt)



    mui_interp = (Nmui_v_EN_fit[:]/nAr)*V0*tau/(L*L)  # + params.mu[1]*1e-2   
    EN_interp = EN_Td / (1e21 * params.V0L/params.nAr) * 2/3 # We multiply with 2/3 here to make it compatible with  mobility_U function.
    mui_spline = CubicSpline(EN_interp, mui_interp)
    mui_EN_spline = CubicSpline.derivative(mui_spline)
    mobility = Mobility(interpolate = True, mu_expression = mui_spline, mu_T_expression = mui_EN_spline)
    muList.append(mobility)

    # fig, ax = plt.subplots()
    # ax.set_title('Ion Mobility Coef.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$\mu_i \, $ [$ \, cm^{2}/V/s$]")
    # ax.semilogx(EN_Td, mui_interp/(V0*tau/(L*L))*1e4, marker = 'o', label = 'raw')
    # # ax.semilogx(EN_Td, mui_EN_spline(EN_interp), marker = '*', label = 'Grad')
    # plt.axhline(y=params.mu[1]/(V0*tau/(L*L))*1e4, color='k', linestyle='--')
    # ax.legend()
    # # ax.semilogy()
    # # ax.loglog()

    # fig, ax = plt.subplots()
    # ax.set_title('Ion Mobility Coef.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$\mu_i \, $ [$ \, 1/m/V/s$]")
    # ax.semilogx(EN_Td, Nmui_v_EN, marker = 'o', label = 'raw')
    # # ax.scatter(EN_Td_exp, Nmui_v_EN_2, marker = 'o', label = 'Exp')
    # # ax.scatter(EN_LXCat, Nmui_v_EN_3, marker = 'd', label = 'LXCat')
    # ax.scatter(EN_Td_data, Nmui_v_EN_data, marker = '*', label = 'LXCat')
    # ax.plot(EN_Td, Nmui_v_EN_fit, marker = '*', label = 'Fit')
    # # plt.axhline(y=params.mu[1]*nAr/(V0*tau/(L*L)), color='k', linestyle='--')
    # ax.legend()
    # ax.semilogx()
    # # ax.loglog()


    #-------------------------------------------------------------------------------
    #########  Argon Molecular Ion Mobility. #########

    # From LXCat
    fileName = "./Data/Ar2IonMobility/LXCat/Ar2IonMobility.csv"
    Data = pd.read_csv(fileName)
    EN_LXCat = Data.iloc[:,0].to_numpy('float64')  # [Td]
    mui2_v_EN = Data.iloc[:,1].to_numpy('float64') # [cm2/V s] 
    Nmui2_v_EN = mui2_v_EN * 1e-4 * N0 # [1/( m V s)] 
    mui2_v_EN *= 101325/Pressure * GasTemperature/273.15 * 1e-4 # [m^2/V/s] 


    # Perform nonlinear least squares fit
    # popt, pcov = curve_fit(ArIon2Model, EN_LXCat[-4:], Nmui2_v_EN[-4:], p0=(0.3,1.5,0.33))
    # print(popt)
    popt = [0.00891562, 1.59508181, 0.27209231]
    # Evaluate the fitted model
    EN_Td_tmp = 170 + np.logspace(np.log10(1e-3),np.log10(10000),100,dtype=np.float64)#  Electric field / N [Td]         
    Nmui2_v_EN_tmp = ArIon2Model(EN_Td_tmp, *popt)

    EN_Td_data = np.concatenate((EN_LXCat, EN_Td_tmp)) # Combine datasets
    Nmui2_v_EN_data = np.concatenate((Nmui2_v_EN, Nmui2_v_EN_tmp)) # Combine datasets
    Nmue2_v_EN_interp = np.interp(EN_Td,EN_Td_data,Nmui2_v_EN_data)

    Nmue2_v_EN_interp_smoothed = uniform_filter1d(Nmue2_v_EN_interp, size=30)

    mui2_interp = (Nmue2_v_EN_interp_smoothed[:]/nAr)*V0*tau/(L*L)  # + params.mu[1]*1e-2   
    EN_interp = EN_Td / (1e21 * params.V0L/params.nAr) * 2/3 # We multiply with 2/3 here to make it compatible with  mobility_U function.
    mui2_spline = CubicSpline(EN_interp, mui2_interp)
    mui2_EN_spline = CubicSpline.derivative(mui2_spline)
    mobility = Mobility(interpolate = True, mu_expression = mui2_spline, mu_T_expression = mui2_EN_spline)
    muList.append(mobility)


    # fig, ax = plt.subplots()
    # ax.set_title('Ion Mobility Coef.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$\mu_i Ar2^{+} \, $ [$ \, 1/m/V/s$]")
    # ax.plot(EN_LXCat, Nmui2_v_EN, marker = 'o', label = 'LXCat')
    # ax.plot(EN_Td_tmp, Nmui2_v_EN_tmp, marker = '+', label = 'Extrapolation')
    # ax.plot(EN_Td, Nmue2_v_EN_interp, marker = '.', label = 'fit')
    # ax.plot(EN_Td, Nmue2_v_EN_interp_smoothed, marker = '.', label = 'fit - smoothed')
    # ax.legend()
    # # ax.semilogy()
    # # ax.loglog()

    # plt.show()
    # exit(0)

    #-------------------------------------------------------------------------------
    #########  Species Mobility #########
    for i in range(3, Ns):
        muList.append(Mobility(interpolate = False))

    #-------------------------------------------------------------------------------
    #########   Electron Energy Mobility #########
    energymobilityData = transport["energy_mobility"] 
    Nmuee_v_Te = energymobilityData[:,1]
    Nmuee_v_Te[0:indices_Te0] = Nmuee_v_Te[indices_Te0]
    muee_interp = (Nmuee_v_Te[:]/nAr)*V0*tau/(L*L)
    muee_interp = uniform_filter1d(muee_interp, size=40)
    muee_spline = CubicSpline(Te_trans, muee_interp)
    muee_Te_spline = CubicSpline.derivative(muee_spline)
    energymobility = Mobility(interpolate = False, mu_expression = muee_spline, mu_T_expression = muee_Te_spline)
    muList.append(energymobility)

    # fig, ax = plt.subplots()
    # ax.set_title('Energy Mobility Coef.')
    # ax.set_xlabel('Te [eV]')
    # ax.set_ylabel(r"$\mu_\epsilon \, $ [$ \, m^{2}/V/s$]")
    # ax.plot(Te_trans, muee_interp, marker = 'o', label = 'raw')
    # ax.plot(Te_trans, muee_Te_spline(Te_trans), marker = '*', label = 'Grad')
    # plt.axhline(y=params.mu[-1], color='k', linestyle='--')
    # ax.legend()




    #-------------------------------------------------------------------------------
    #########   Electron Diffusion Coef.  #########
    # From LXCat
    diffusivityData = transport["diffusivity"] 
    NDe_v_Te = diffusivityData[:,1]
    # NDe_v_Te[0:indices_Te0] = NDe_v_Te[indices_Te0]
        
    # De_interp = (NDe_v_Te[:]/nAr)*tau/(L*L)
    # De_interp = uniform_filter1d(De_interp, size=20)
    # De_spline = CubicSpline(Te_trans, De_interp)
    # De_Te_spline = CubicSpline.derivative(De_spline)
    # diffusivity = Diffusivity(interpolate = False, D_expression = De_spline, D_T_expression = De_Te_spline)
    # diffList.append(diffusivity)

    # # fig,ax = plt.subplots()
    # # ax.set_title('Diffusion Coef.')
    # # ax.set_xlabel('Te [eV]')
    # # ax.set_ylabel(r"$D_e \, $ [$ \, m^{2}/s$]")
    # # ax.plot(Te_trans, De_interp, marker = '.', label = 'raw')
    # # # ax.plot(Te_trans, De_Te_spline(Te_trans), marker = '*', label = 'raw - grad')
    # # plt.axhline(y=params.D[0], color='k', linestyle='--')
    # # ax.legend()


    # From LXCat
    NDe_v_EN_interp =  Ee_v_EN_interp * Nmue_v_EN_interp
    NDe_v_EN_interp_smoothed = Ee_v_EN_interp_smoothed * Nmue_v_EN_interp_smoothed

    # NDe_v_EN_interp[0:indices_EN0] = NDe_v_EN_interp[indices_EN0]

    threshold_EN0 = 0.05    
    indices_EN0 = np.searchsorted(EN_Td, threshold_EN0)
    NDe_v_EN_interp_cliped = np.copy(NDe_v_EN_interp_smoothed)
    NDe_v_EN_interp_cliped[0:indices_EN0] = NDe_v_EN_interp_cliped[indices_EN0]

    NDe_v_EN_interp_cliped = uniform_filter1d(NDe_v_EN_interp_cliped, size=50)

    De_interp = (NDe_v_EN_interp_cliped[:]/nAr)*tau/(L*L)
    EN_interp = EN_Td / (1e21 * params.V0L/params.nAr) * 2/3 # We multiply with 2/3 here to make it compatible with  mobility_U function.
    De_spline = CubicSpline(EN_interp, De_interp)
    De_EN_spline = CubicSpline.derivative(De_spline)
    diffusivity = Diffusivity(interpolate = False, D_expression = De_spline, D_T_expression = De_EN_spline)
    diffList.append(diffusivity)

    # fig,ax = plt.subplots()
    # ax.set_title('Diffusion Coef.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$D_e \cdot N \, $ [$ \, \frac{1}{m s}$]")
    # ax.plot(EN, NDe_v_Te, marker = '.', label = 'Bolsig+') 
    # ax.plot(EN_Td, NDe_v_EN_interp, marker = '.', label = 'fit')
    # ax.plot(EN_Td, NDe_v_EN_interp_cliped, marker = '.', label = 'fit - smoothed and clipped')
    # plt.axhline(y=params.D[0]*nAr/(tau/(L*L)), color='k', linestyle='--')
    # ax.legend()
    # ax.loglog()




    #-------------------------------------------------------------------------------
    #########  Ion Diffusion Coef. #########
    
    # Determination of mobility and diffusion coefficients of Ar+ and Ar2+ ions in argon gas (Jasmiyanaa et al  2020)
    # EN_Td_exp_2 = np.array([34.5, 39.9, 45.8, 49.2, 56.4, 60.6, 61.8, 67.8, 73.5, 79.2, 85.7, 92.3, 
    #                         95 ,103.7, 108.3, 116.6, 117.6, 133.5, 138.4, 152.3, 173, 195.4, 208.7]) # [Td]

    # DLOverK0 = np.array([25.77, 31.9, 31.16, 27.78, 30.42, 29.54, 36.18, 35.34, 34.9, 34.83, 36.27, 
    #                38.75, 39.63, 41.23, 40.1, 48.23, 41.67, 47.29, 54.35, 50.37, 62.7, 62.13, 56.42]) # [mV]
    # DLOverK0 = DLOverK0*1e-3 # [V]
    # DLOverK0 = np.interp(EN_Td,EN_Td_exp_2,DLOverK0)  
    # DL = DLOverK0 * K0

    # Determination of mobility and diffusion coefficients of Ar+ and Ar2+ ions in argon gas (Jasmiyanaa et al  2020)
    fileName = "./BOLSIGChemistry_Transport/ArIonDiffMobilityRatio_inmVvsTd.csv"
    ArIon_Data = pd.read_csv(fileName)
    EN_Td_exp_Diff_fit = ArIon_Data.iloc[:,0].to_numpy('float64')  # [Td]
    DLOverK0_fit = ArIon_Data.iloc[:,1].to_numpy('float64') # [mV]
    DLOverK0_fit = DLOverK0_fit*1e-3 # [V]
    sorted_indices = np.argsort(EN_Td_exp_Diff_fit)
    EN_Td_exp_Diff_fit = EN_Td_exp_Diff_fit[sorted_indices]
    DLOverK0_fit = DLOverK0_fit[sorted_indices]

    DLOverK0_fit_interp = np.interp(EN_Td,EN_Td_exp_Diff_fit,DLOverK0_fit)  
    DL = DLOverK0_fit_interp * K0
    # DL_Ein = K0 * spc.k * GasTemperature / spc.e  # Einstein Relation
    threshold_EN = 188    
    indices_EN_max = np.searchsorted(EN_Td, threshold_EN)
    DL[indices_EN_max:] = DL[indices_EN_max]

    EN_interp = EN_Td / (1e21 * params.V0L/params.nAr) * 2/3 # We multiply with 2/3 here to make it compatible with  mobility_U function.
    Di_interp = DL[:]*tau/(L*L)
    # Di_interp = uniform_filter1d(Di_interp, size=1)
    Di_spline = CubicSpline(EN_interp, Di_interp)
    Di_EN_spline = CubicSpline.derivative(Di_spline)
    diffusivity = Diffusivity(interpolate = False, D_expression = Di_spline, D_T_expression = Di_EN_spline)
    diffList.append(diffusivity)

    # fig,ax = plt.subplots()
    # ax.set_title('Diffusion Coef.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$D_i \, $ [$ \, m^{2}/s$]")
    # ax.plot(EN_Td, Di_interp, marker = '.', label = 'raw')
    # ax.plot(EN_Td, Di_EN_spline(Di_interp), marker = '*', label = 'raw - grad')
    # plt.axhline(y=params.D[1], color='k', linestyle='--')
    # ax.legend()


    #-------------------------------------------------------------------------------
    #########   Species Diffusion Coef.  #########
    for i in range(2, Ns):
        diffList.append(Diffusivity(interpolate = False))


    #-------------------------------------------------------------------------------
    #########   Electron Energy Diffusion Coef.  #########
    energydiffusivityData = transport["energy_diffusivity"] 
    NDee_v_Te = energydiffusivityData[:,1]
    NDee_v_Te[0:indices_Te0] = NDee_v_Te[indices_Te0]
    Dee_interp = (NDee_v_Te[:]/nAr)*tau/(L*L)
    Dee_interp = uniform_filter1d(Dee_interp, size=20)
    Dee_spline = CubicSpline(Te_trans, Dee_interp)
    Dee_Te_spline = CubicSpline.derivative(Dee_spline)
    energydiffusivity = Diffusivity(interpolate = False, D_expression = Dee_spline, D_T_expression = Dee_Te_spline)
    diffList.append(energydiffusivity)

    # fig,ax = plt.subplots()
    # ax.set_title('Energy Diffusion Coef.')
    # ax.set_xlabel('Te [eV]')
    # # ax.set_ylabel('D [m2/s]')
    # ax.set_ylabel(r"$D_{\epsilon} \, $ [$ \, m^{2}/s$]")
    # ax.plot(Te_trans, Dee_interp, marker = '.', label = 'raw')
    # ax.plot(Te_trans, Dee_Te_spline(Te_trans), marker = '*', label = 'raw - grad')
    # plt.axhline(y=params.D[-1], color='k', linestyle='--')
    # ax.legend()



    #-------------------------------------------------------------------------------
    #########   ---------------END----------------  #########

    
    params.diffusivityList = diffList
    params.mobilityList = muList
      
    # 5) Dump to screen
    params.print()


    ### Indexing
    # GlowDischarge Indexing 
    # i = 0       -> electrons 
    # i = 1       -> ions 
    # i = 2       -> argon molecular ions 
    # i = 3       -> argon excimer molecules 
    # i = 4:Ns-1  -> excited levels
    # i = Ns - 1  -> ground state
    # i = Ns      -> electron energy
    # i = Nv - 1  -> ion Effective Electric field

    # CR Indexing 
    # i = 0       -> ground state
    # i = 1:Ns-4  -> excited levels
    # i = Ns - 4  -> argon excimer molecules 
    # i = Ns - 3  -> argon molecular ions 
    # i = Ns - 2  -> electrons
    # i = Ns - 1  -> ions
    # i = Ns      -> electron energy

