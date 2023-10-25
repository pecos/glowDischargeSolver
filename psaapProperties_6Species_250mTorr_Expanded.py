import numpy as np
from scipy.interpolate import CubicSpline
import csv
import h5py as h5

import logging

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


def setPsaapProperties_6Species_250mTorr_Expanded(gam, inputV0, inputVDC, params, Nr, iSample):
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

    # densities
    nAr = 8.05e21     # background number density of Ar [1/m^3] (corresponds to p = 250 mTorr)
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

    # nominal electron energy
    e0 = 1.0  # [eV]

    # pressure
    p  = 33.3*1.5      # [J/m^3] *1.5 to convert it to energy (250 mTorr)

    # gas energy at the wall
    Tg0 = 0.038778    # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot

    # characteristics of driving voltage
    V0  = inputV0                 # amplitude of driving voltage [V]
    verticalShift = inputVDC      # DC voltage (vertical shift in driving voltage)
    tau = (1./13.56e6)             # period of driving voltage [s]
    L   = 2.00*0.005              # half-gap-width [m] (gap width is 2 cm)
    electrodeArea = np.pi*0.05**2 # electrode area [m^2] (electrode diameter = 0.1 m)

    # transport parameters
    nmue = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
    nmum = 0.0
    nmur = 0.0
    nmu4p = 0.0
    nmui = 8.0e19
    nDe  = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi  = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]
    nDm  = 2.42e18   # argon number density times AR(m) diffusivity [1/(cm*s)]
    nDr  = 2.42e18
    nD4p = 2.42e18

    # reaction parameters (NB: k_i = Ck*Ee^B*exp(-A/Ee))
    #                          Ee = 3/2*Te (Te in eV)
    #                          -> k_i = [Ck*(2/3)^B] * Ee^B * exp[-(3/2)*A/Ee]
    # nominal
    Ck = np.array([2.0e-13,2.1e-15,5.0e-16,6.4e-16,2.1e-21,1.32e8,1.72e7,1.50e7,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0e-18,4.0e-19,0.0,0.0,0.0,0.0,2.5e-17,2.5e-17,1.0e-15,1.0e-15,0.0,0.0]) # pre-exponential factors [m^3/s]
    B  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0]) # Temperature Power
    A  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # activation temperature [eV]
    dH = np.array([0.0,-7.541,-10.577,-7.393,0.0,0.0,0.0,0.0,11.577,11.725,13.168,15.76,-11.577,4.183,0.148,1.592,2.592,-1.444,-1.592,-11.725,-0.148,1.444,0.0,0.0,-4.183,-4.035,-2.592,-15.76,0.0,0.0,-8.985,-9.133,4.035,-13.168]) # energy lost per electron due to ionization rxn [eV]
    dEps = np.array([0.0,15.76,11.577,11.725,13.168,0.0]) # E, AR+, AR(m), AR(r), AR(4p), AR

    # BC parameters
    # ks = 1.19e7  # electron recombination rate [cm/s]
    ks = 1.366109824889323e7 # electron recombination rate [cm/s/eV]

    ###################################################################
    # Constants of nature (probably shouldn't change unless you have
    # root privileges on universe)
    ###################################################################
    qe   = 1.6e-19   # unit charge [C]
    eps0 = 8.86e-12  # permittivity of free space [F/m]
    kB   = 1.38e-23  # Boltzmann constant [J/K]
    # kB   = 8.62e-5 # Boltzmann constant [eV/K]


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

    Ck[0:4]  *= tau*np0
    Ck[4]    *= tau*nAr
    Ck[5:8]  *= tau
    Ck[8:12] *= tau*nAr
    Ck[12:24]  *= tau*np0
    Ck[24:28] *= tau*np0*np0
    Ck[28:30] *= tau*nAr
    Ck[30:] *= tau*np0
    A        = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH       = dH/e0
    qStar    = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha    = qe*np0*L*L/(V0*eps0)
    ks       = ks*tau/L
    p0       = p/qe/np0
    kappaB   = 4.878171165833662*1.6129 # non-dimensional thermal conductivity of background specie
                                # (2/3)*tau/L**2*Kb/np0/kB,
                                # where Kb is the thermal conductivity of background specie

    params.beta = np.array([[0,1,1,1,0,0,0,0,1,1,1,2,1,2,1,1,2,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,2,1],                     # E
                            [0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],                     # AR+
                            [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0],                     # AR(m)
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],                     # AR(r)
                            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0],                     # AR(4p)
                            [2,1,1,1,2,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1]], dtype=np.int64)    # AR

    params.alfa = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,0,0,0,1,1],                     # E
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0],                     # AR+
                            [2,1,0,2,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],                     # AR(m)
                            [0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0],                     # AR(r)
                            [0,0,2,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1],                     # AR(4p)
                            [0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0]], dtype=np.int64)    # AR
    
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
    params.D[0]    = De
    params.D[1]    = Di
    params.D[2]    = Dm
    params.D[3]    = Dr
    params.D[4]    = D4p

    params.mu[0]   = mue
    params.mu[1]   = mui
    params.mu[2]   = mum
    params.mu[3]   = mur
    params.mu[4]   = mu4p

    params.A[:]    = Ck[:]
    params.B[:]    = B[:]
    params.C[:]    = A[:]

    # Account for the 2/3 term to convert from electron temperature to electron energy
    for i in range(len(params.A)):
        params.A[i] *= (2/3)**(params.B[i])

    params.dH[:]   = dH[:]
    params.dEps[:] = dEps[:]
    params.qStar   = qStar
    params.alpha   = alpha
    params.ks      = ks
    params.gam     = gam
    params.kappaB  = kappaB
    params.nAronp0 = nAr / np0
    params.p0      = p0
    params.Tg0     = Tg0
    params.EC      = 2.0 * me / mAr \
        * np.sqrt(16.0 * (me + mAr) * e0 * c**2
                  / (3.0 * np.pi * me * mAr)) * se * nAr * tau
    # params.EC = 2.0 * me / mAr * 3.8e9 * tau

    params.verticalShift = verticalShift / V0

    # Parameters needed to compute the current with dimensions
    params.V0Ltau  = V0 / (L * tau)
    params.V0L     = V0 / L
    params.LLV0tau = (L*L) / (V0*tau)
    params.tauL    = L / tau
    params.np0     = np0           # "nominal" electron density [1/m^3]
    params.qe      = qe            # unit charge [C]
    params.eps0    = eps0          # unit charge [C]
    params.eArea   = electrodeArea # electrode area [m^2]


    reactionExpressionslist = [f"{params.A[0]} * energy**{params.B[0]} * np.exp(-{params.C[0]} / energy)",
                               f"{params.A[1]} * energy**{params.B[1]} * np.exp(-{params.C[1]} / energy)",
                               f"{params.A[2]} * energy**{params.B[2]} * np.exp(-{params.C[2]} / energy)",
                               f"{params.A[3]} * energy**{params.B[3]} * np.exp(-{params.C[3]} / energy)",
                               f"{params.A[4]} * energy**{params.B[4]} * np.exp(-{params.C[4]} / energy)",
                               f"{params.A[5]} * energy**{params.B[5]} * np.exp(-{params.C[5]} / energy)",
                               f"{params.A[6]} * energy**{params.B[6]} * np.exp(-{params.C[6]} / energy)",
                               f"{params.A[7]} * energy**{params.B[7]} * np.exp(-{params.C[7]} / energy)",
                               f"{params.A[8]} * energy**{params.B[8]} * np.exp(-{params.C[8]} / energy)",
                               f"{params.A[9]} * energy**{params.B[9]} * np.exp(-{params.C[9]} / energy)",
                               f"{params.A[10]} * energy**{params.B[10]} * np.exp(-{params.C[10]} / energy)",
                               f"{params.A[11]} * energy**{params.B[11]} * np.exp(-{params.C[11]} / energy)",
                               f"{params.A[12]} * energy**{params.B[12]} * np.exp(-{params.C[12]} / energy)",
                               f"{params.A[13]} * energy**{params.B[13]} * np.exp(-{params.C[13]} / energy)",
                               f"{params.A[14]} * energy**{params.B[14]} * np.exp(-{params.C[14]} / energy)",
                               f"{params.A[15]} * energy**{params.B[15]} * np.exp(-{params.C[15]} / energy)",
                               f"{params.A[16]} * energy**{params.B[16]} * np.exp(-{params.C[16]} / energy)",
                               f"{params.A[17]} * energy**{params.B[17]} * np.exp(-{params.C[17]} / energy)",
                               f"{params.A[18]} * energy**{params.B[18]} * np.exp(-{params.C[18]} / energy)",
                               f"{params.A[19]} * energy**{params.B[19]} * np.exp(-{params.C[19]} / energy)",
                               f"{params.A[20]} * energy**{params.B[20]} * np.exp(-{params.C[20]} / energy)",
                               f"{params.A[21]} * energy**{params.B[21]} * np.exp(-{params.C[21]} / energy)",
                               f"{params.A[22]} * energy**{params.B[22]} * np.exp(-{params.C[22]} / energy)",
                               f"{params.A[23]} * energy**{params.B[23]} * np.exp(-{params.C[23]} / energy)",
                               f"{params.A[24]} * energy**{params.B[24]} * np.exp(-{params.C[24]} / energy)",
                               f"{params.A[25]} * energy**{params.B[25]} * np.exp(-{params.C[25]} / energy)",
                               f"{params.A[26]} * energy**{params.B[26]} * np.exp(-{params.C[26]} / energy)",
                               f"{params.A[27]} * energy**{params.B[27]} * np.exp(-{params.C[27]} / energy)",
                               f"{params.A[28]} * energy**{params.B[28]} * np.exp(-{params.C[28]} / energy)",
                               f"{params.A[29]} * energy**{params.B[29]} * np.exp(-{params.C[29]} / energy)",
                               f"{params.A[30]} * energy**{params.B[30]} * np.exp(-{params.C[30]} / energy)",
                               f"{params.A[31]} * energy**{params.B[31]} * np.exp(-{params.C[31]} / energy)",
                               f"{params.A[32]} * energy**{params.B[32]} * np.exp(-{params.C[32]} / energy)",
                               f"{params.A[33]} * energy**{params.B[33]} * np.exp(-{params.C[33]} / energy)"]


    reactionTExpressionslist = [f"{params.A[0]} * (energy**({params.B[0]}-1)) * np.exp(-{params.C[0]}/energy) * ({params.B[0]} + {params.C[0]}/energy)",
                                f"{params.A[1]} * (energy**({params.B[1]}-1)) * np.exp(-{params.C[1]}/energy) * ({params.B[1]} + {params.C[1]}/energy)",
                                f"{params.A[2]} * (energy**({params.B[2]}-1)) * np.exp(-{params.C[2]}/energy) * ({params.B[2]} + {params.C[2]}/energy)",
                                f"{params.A[3]} * (energy**({params.B[3]}-1)) * np.exp(-{params.C[3]}/energy) * ({params.B[3]} + {params.C[3]}/energy)",
                                f"{params.A[4]} * (energy**({params.B[4]}-1)) * np.exp(-{params.C[4]}/energy) * ({params.B[4]} + {params.C[4]}/energy)",
                                f"{params.A[5]} * (energy**({params.B[5]}-1)) * np.exp(-{params.C[5]}/energy) * ({params.B[5]} + {params.C[5]}/energy)",
                                f"{params.A[6]} * (energy**({params.B[6]}-1)) * np.exp(-{params.C[6]}/energy) * ({params.B[6]} + {params.C[6]}/energy)",
				f"{params.A[7]} * (energy**({params.B[7]}-1)) * np.exp(-{params.C[7]}/energy) * ({params.B[7]} + {params.C[7]}/energy)",
		                f"{params.A[8]} * (energy**({params.B[8]}-1)) * np.exp(-{params.C[8]}/energy) * ({params.B[8]} + {params.C[8]}/energy)",
                                f"{params.A[9]} * (energy**({params.B[9]}-1)) * np.exp(-{params.C[9]}/energy) * ({params.B[9]} + {params.C[9]}/energy)",
                                f"{params.A[10]} * (energy**({params.B[10]}-1)) * np.exp(-{params.C[10]}/energy) * ({params.B[10]} + {params.C[10]}/energy)",
                                f"{params.A[11]} * (energy**({params.B[11]}-1)) * np.exp(-{params.C[11]}/energy) * ({params.B[11]} + {params.C[11]}/energy)",
                                f"{params.A[12]} * (energy**({params.B[12]}-1)) * np.exp(-{params.C[12]}/energy) * ({params.B[12]} + {params.C[12]}/energy)",
                                f"{params.A[13]} * (energy**({params.B[13]}-1)) * np.exp(-{params.C[13]}/energy) * ({params.B[13]} + {params.C[13]}/energy)",
                                f"{params.A[14]} * (energy**({params.B[14]}-1)) * np.exp(-{params.C[14]}/energy) * ({params.B[14]} + {params.C[14]}/energy)",
                                f"{params.A[15]} * (energy**({params.B[15]}-1)) * np.exp(-{params.C[15]}/energy) * ({params.B[15]} + {params.C[15]}/energy)",
                                f"{params.A[16]} * (energy**({params.B[16]}-1)) * np.exp(-{params.C[16]}/energy) * ({params.B[16]} + {params.C[16]}/energy)",
                                f"{params.A[17]} * (energy**({params.B[17]}-1)) * np.exp(-{params.C[17]}/energy) * ({params.B[17]} + {params.C[17]}/energy)",
                                f"{params.A[18]} * (energy**({params.B[18]}-1)) * np.exp(-{params.C[18]}/energy) * ({params.B[18]} + {params.C[18]}/energy)",
                                f"{params.A[19]} * (energy**({params.B[19]}-1)) * np.exp(-{params.C[19]}/energy) * ({params.B[19]} + {params.C[19]}/energy)",
                                f"{params.A[20]} * (energy**({params.B[20]}-1)) * np.exp(-{params.C[20]}/energy) * ({params.B[20]} + {params.C[20]}/energy)",
                                f"{params.A[21]} * (energy**({params.B[21]}-1)) * np.exp(-{params.C[21]}/energy) * ({params.B[21]} + {params.C[21]}/energy)",
                                f"{params.A[22]} * (energy**({params.B[22]}-1)) * np.exp(-{params.C[22]}/energy) * ({params.B[22]} + {params.C[22]}/energy)",
                                f"{params.A[23]} * (energy**({params.B[23]}-1)) * np.exp(-{params.C[23]}/energy) * ({params.B[23]} + {params.C[23]}/energy)",
                                f"{params.A[24]} * (energy**({params.B[24]}-1)) * np.exp(-{params.C[24]}/energy) * ({params.B[24]} + {params.C[24]}/energy)",
                                f"{params.A[25]} * (energy**({params.B[25]}-1)) * np.exp(-{params.C[25]}/energy) * ({params.B[25]} + {params.C[25]}/energy)",
                                f"{params.A[26]} * (energy**({params.B[26]}-1)) * np.exp(-{params.C[26]}/energy) * ({params.B[26]} + {params.C[26]}/energy)",
                                f"{params.A[27]} * (energy**({params.B[27]}-1)) * np.exp(-{params.C[27]}/energy) * ({params.B[27]} + {params.C[27]}/energy)",
                                f"{params.A[28]} * (energy**({params.B[28]}-1)) * np.exp(-{params.C[28]}/energy) * ({params.B[28]} + {params.C[28]}/energy)",
                                f"{params.A[29]} * (energy**({params.B[29]}-1)) * np.exp(-{params.C[29]}/energy) * ({params.B[29]} + {params.C[29]}/energy)",
                                f"{params.A[30]} * (energy**({params.B[30]}-1)) * np.exp(-{params.C[30]}/energy) * ({params.B[30]} + {params.C[30]}/energy)",
                                f"{params.A[31]} * (energy**({params.B[31]}-1)) * np.exp(-{params.C[31]}/energy) * ({params.B[31]} + {params.C[31]}/energy)",
                                f"{params.A[32]} * (energy**({params.B[32]}-1)) * np.exp(-{params.C[32]}/energy) * ({params.B[32]} + {params.C[32]}/energy)",
                                f"{params.A[33]} * (energy**({params.B[33]}-1)) * np.exp(-{params.C[33]}/energy) * ({params.B[33]} + {params.C[33]}/energy)"]
           

    reactionExpressionTypelist =  np.array([False,False,False,False,False,False,False,False, # Rxns 1-8
                                           True,True,True,True,True,True,True,True,True,True,True,True,True,True, # Rxns 9-21
                                           False,False,True,True,True,True,False,False,False,False,True,True]) # Rxns 22-34

    thresholded_rxn = np.array([False, False, False, False, False, False, False, False,
                                True, True, True, True, False, True, True, True, True, False, False, False, False,
                                True, False, False, False, False, False, False, False, False, False, False, True,
                                False])
    reactionsList = []
    LOGFilename = 'interpolationSample.log'
    f = open(LOGFilename, 'w')

    for i in range(Nr):
        if reactionExpressionTypelist[i]:
            if i < 14 or i == 16 or i == 19 or i > 23:
                f = h5.File("../../../BOLSIGChemistry_NominalRates/{0:s}.h5".format(rxnNameDict[i]), 'r')
                dataset = f["table"]
            else:
                f = h5.File("../../../BOLSIGChemistry_NominalRates/StepExcitation.h5", 'r')
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

            Te = np.delete(Te, fail_inds)
            rateCoeff = np.delete(rateCoeff, fail_inds)

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

            # Find first non-zero value of the coefficient rate.
            I = np.nonzero(rateCoeff)

            diffRateCoeff = [j-i for i, j in zip(rateCoeff[:-1], rateCoeff[1:])]
            diffTe = [j-i for i, j in zip(Te[:-1], Te[1:])]

            Monotonicity = np.asarray([j/i for i, j in zip(diffTe, diffRateCoeff)])
            Monotonicity = np.insert(Monotonicity, 0, 0.0, axis=0)

            Nan = np.isnan(Monotonicity)
            Inf = np.isinf(Monotonicity)
            if thresholded_rxn[i] == True:
                indexPositive = np.where(Monotonicity>0.0)
            else:
                indexPositive = np.where(Monotonicity<0.0)
            Positive = np.full(Monotonicity.shape, False, dtype=bool)
            Positive[indexPositive] = True
            
            indices = Nan + Inf + Positive


            #lastFalse = np.where(indices==False)[-1][-1] + 2
            for k in range(len(Te)):
               if Te[k] < 4.5 and indices[k] == False:
                  lastFalse = k + 2

            # Transformation to log scale.
            TeLog = np.log(Te)

            # Compute the slope of the rate coefficient between its first two non-zero values.
            # Finite differences are used.
            dydx = (rateCoeff[lastFalse + 1] - rateCoeff[lastFalse]) \
                 / (Te[lastFalse + 1] - Te[lastFalse])

            # Arrhenius form: kf = A * exp(-C / Te)
            C = Te[lastFalse]**2.0*dydx / rateCoeff[lastFalse]

            # Compute pre-exponential coefficient, A, in log scale.
            ALog = np.log(rateCoeff[lastFalse]) + C / Te[lastFalse]

            # Transform rate coefficient in log scale.
            rateCoeffLog = np.zeros(rateCoeff.shape)
            rateCoeffLog[lastFalse:] = np.log(rateCoeff[lastFalse:])
            # For the troublesome values, we use the Arrhenius form.
            rateCoeffLog[0:lastFalse] = ALog - C / Te[0:lastFalse]
            # Nondimensionalization in log scale.
            if (i < 12):
                rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
            elif i > 23 and i < 28:
                rateCoeffLog += - np.log(1.0/tau) + 2*np.log(np0)
            else:
                rateCoeffLog += - np.log(1.0/tau) + np.log(np0)

            # Interpolation in log scale.
            reactionExpressionsLog = CubicSpline(TeLog, rateCoeffLog)
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

    ## Electron Transport Data
    diffList = []
    transport = h5.File("../../../BOLSIGChemistry_NominalRates/nominal_transport.h5", 'r')
    NDe_v_Te = transport["diffusivity"]
    Te_trans = NDe_v_Te[:,0]
    Te_trans /= 11604
    De_interp = (NDe_v_Te[:,1]/nAr)*tau/(L*L)
    De_spline = CubicSpline(Te_trans, De_interp)
    De_Te_spline = CubicSpline.derivative(De_spline)
    diffusivity = Diffusivity(interpolate = True, D_expression = De_spline, D_T_expression = De_Te_spline)
    diffList.append(diffusivity)

    Ns = 6
    for i in range(1, Ns):
        diffList.append(Diffusivity(interpolate = False))

    params.diffusivityList = diffList

    muList = []
    Nmue_v_Te = transport["mobility"]
    mue_interp = (Nmue_v_Te[:,1]/nAr)*V0*tau/(L*L)
    mue_spline = CubicSpline(Te_trans, mue_interp)
    mue_Te_spline = CubicSpline.derivative(mue_spline)
    mobility = Mobility(interpolate = True, mu_expression = mue_spline, mu_T_expression = mue_Te_spline)
    muList.append(mobility)

    Ns = 6
    for i in range(1, Ns):
        muList.append(Mobility(interpolate = False))

    params.mobilityList = muList
    
    # 5) Dump to screen
    params.print()
