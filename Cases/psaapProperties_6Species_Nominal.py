import numpy as np
from scipy.interpolate import CubicSpline
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py as h5

import logging

class Reaction(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

def setPsaapProperties_6Species_Nominal(gam, inputV0, inputVDC, params, Nr, iSample):
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
    nAr = 3.22e22     # background number density of Ar [1/m^3] (corresponds to p=100 mTorr)
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
    p  = 133.3224*1.5      # [J/m^3] *1.5 to convert it to energy (1 Torr)

    # gas energy at the wall
    Tg0 = 0.038778    # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot

    # characteristics of driving voltage
    V0  = inputV0                 # amplitude of driving voltage [V]
    verticalShift = inputVDC      # DC voltage (vertical shift in driving voltage)
    tau = (1./13.56e6)             # period of driving voltage [s]
    L   = 2.00*0.005              # half-gap-width [m] (gap width is 2 cm)
    electrodeArea = np.pi*0.05**2 # electrode area [m^2] (electrode diameter = 0.1 m)

    # Chemistry parameters (charge number, Cv, Cp)
    Ns = 6
    Z = np.zeros(Ns+1); Cv = np.zeros(Ns+1);    Cp = np.zeros(Ns+1)
    Z[0]    = -1;       Cv[0]    = 3.0/2.0;     Cp[0]    = 5.0/2.0;     # E
    Z[1]    =  1;       Cv[1]    = 3.0/2.0;     Cp[1]    = 5.0/2.0;     # Ar+
    Z[2:Ns] =  0;       Cv[2:Ns] = 3.0/2.0;     Cp[2:Ns] = 5.0/2.0;     # Ar2+
    Z[-1]   = -1;       Cv[-1]   = 3.0/2.0;     Cp[-1]   = 5.0/2.0;     # Ee electron energy 

    # transport parameters
    nmue = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
    #nmui = 4.65e19   # argon number density times ion mobility [1/(V*cm*s)]
    #nmum = 1 / (np.sqrt(16.0 * (mAr + mAr) * 300 * 8.62e-5 * c**2
    #                    / (3.0 * np.pi * mAr * mAr)) * se * mAr * 1.6e-19 / c**2)
    nmum = 0.0
    nmur = 0.0
    nmu4p = 0.0
    nmui = 8.0e19
    #nmum = 9.35e19
    nDe  = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi  = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]
    nDm  = 2.42e18   # argon number density times AR(m) diffusivity [1/(cm*s)]
    nDr  = 2.42e18
    nD4p = 2.42e18
    #nDm  = 3.914e20

    # reaction parameters (NB: k_i = Ck*Ee^B*exp(-A/Ee))
    #                          Ee = 3/2*Te (Te in eV)
    #                          -> k_i = [Ck*(2/3)^B] * Ee^B * exp[-(3/2)*A/Ee]
    # nominal
    Ck = np.array([2.0e-7,2.1e-9,5.0e-10,6.4e-10,2.1e-15,1.0e5,3.2e7,3.0e7,3.0e7,0.0,0.0,0.0,0.0,4.3e-10,0.0,3.7e-8,8.9e-7,1.8e-7,3.0e-7,3.0e-7,4.3e-10,9.1e-7,8.9e-7]) # pre-exponential factors [cm^3/s]
    B  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0.74,0,0,0.51,0.61,0.51,0.51,0.74,0,0.51])
    A  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.59,2.61,0,0,0,0,1.59]) # activation temperature [eV]
    dH = np.array([0.0,-7.412,-10.054,-7.336,0.0,0.0,0.0,0.0,0.0,11.548,11.624,12.907,15.76,-11.548,4.212,0.076,1.359,2.853,-1.283,-1.359,-11.624,-0.076,0.983]) # energy lost per electron due to ionization rxn [eV]
    dEps = np.array([0.0,15.76,11.548,11.624,12.907,0.0]) # E, AR+, AR(m), AR(r), AR(4p), AR

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
    Ck[0:5] *= 1e-6 # m^3/s
    Ck[5:9] *= 1 # 1/s
    Ck[9:22] *= 1e-6 # m^3/s
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
    Ck[5:9]  *= tau
    Ck[9:13] *= tau*nAr
    Ck[13:]  *= tau*np0
    A        = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH       = dH/e0
    qStar    = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha    = qe*np0*L*L/(V0*eps0)
    ks       = ks*tau/L
    p0       = p/qe/np0
    kappaB   = 4.878171165833662*1.6129 # non-dimensional thermal conductivity of background specie
                                # (2/3)*tau/L**2*Kb/np0/kB,
                                # where Kb is the thermal conductivity of background specie

    params.beta = np.array([[0,1,1,1,0,0,0,0,0,1,1,1,2,1,2,1,1,2,1,1,1,1,1],                     # E
                            [0,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0],                     # AR+
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0],                     # AR(m)
                            [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0],                     # AR(r)
                            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1],                     # AR(4p)
                            [2,1,1,1,2,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0]], dtype=np.int64)    # AR

    params.alfa = np.array([[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],                     # E
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],                     # AR+
                            [2,1,0,2,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0],                     # AR(m)
                            [0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],                     # AR(r)
                            [0,0,2,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0],                     # AR(4p)
                            [0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)    # AR
	# Rxn1:  2AR(m)         ->   2AR
	# Rxn2:  AR(m) + AR(r)  ->   E + AR+ + AR
	# Rxn3:  2AR(4p)        ->   E + AR+ + AR
	# Rxn4:  2AR(m)         ->   E + AR+ + AR
	# Rxn5:  AR(m) + AR     ->   2AR
	# Rxn6:  AR(r)          ->   AR
	# Rxn7:  AR(4p)         ->   AR
	# Rxn8:  AR(4p)         ->   AR(m)
	# Rxn9:  AR(4p)         ->   AR(r)
    # Rxn10: E + AR         ->   E + AR(m)
    # Rxn11: E + AR         ->   E + AR(r)
    # Rxn12: E + AR         ->   E + AR(4p)
    # Rxn13: E + AR         ->   2E + AR+
    # Rxn14: E + AR(m)      ->   E + AR
    # Rxn15: E + AR(m)      ->   2E + AR+
    # Rxn16: E + AR(m)      ->   E + AR(r)
    # Rxn17: E + AR(m)      ->   E + AR(4p)
    # Rxn18: E + AR(4p)     ->   2E + AR+
    # Rxn19: E + AR(4p)     ->   E + AR(r)
    # Rxn20: E + AR(4p)     ->   E + AR(m)
    # Rxn21: E + AR(r)      ->   E + AR
    # Rxn22: E + AR(r)      ->   E + AR(m)
    # Rxn23: E + AR(r)      ->   E + AR(4p)

    rxnNameDict = { 0: "2Ar(m) => 2Ar",
                    1: "Ar(m) + Ar(r) => E + Ar+ + Ar",
                    2: "2Ar(4p) => E + Ar+ + Ar",
                    3: "2Ar(m) => E + Ar+ + Ar",
                    4: "Ar(m) + Ar => 2Ar",
                    5: "Ar(r) => Ar",
                    6: "Ar(4p) => Ar",
                    7: "Ar(4p) => Ar(m)",
                    8: "Ar(4p) => Ar(r)",
                    9: "lumped.metastable",
                   10: "lumped.resonance",
                   11: "lumped.2p",
                   12: "ionization",
                   13: "E + Ar(m) => E + Ar",
                   14: "step_ionization",
                   15: "E + Ar(m) => E + Ar(r)",
                   16: "E + Ar(m) => E + Ar(4p)",
                   17: "E + Ar(4p) => 2E + Ar+",
                   18: "E + Ar(4p) => E + Ar(r)",
                   19: "E + Ar(4p) => E + Ar(m)",
                   20: "E + Ar(r) => E + Ar",
                   21: "E + Ar(r) => E + Ar(m)",
                   22: "E + Ar(r) => E + Ar(4p)"}

    # 4) Set values in params class
    params.Z[:]     = Z[:]
    params.Cv[:]    = Cv[:]
    params.Cp[:]    = Cp[:]

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

    params.EeBC = 0.75

    params.clip_state = False


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
                               f"{params.A[22]} * energy**{params.B[22]} * np.exp(-{params.C[22]} / energy)"]

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
                                f"{params.A[22]} * (energy**({params.B[22]}-1)) * np.exp(-{params.C[22]}/energy) * ({params.B[22]} + {params.C[22]}/energy)"]
           

    reactionExpressionTypelist =  np.array([False,False,False,False,False,False,False,False,False,
                                           True,True,True,True,False,True,True,True,False,True,True,False,True,True])

    reactionsList = []
    LOGFilename = 'interpolationSample%s.log'%str(iSample)
    f = open(LOGFilename, 'w')
	# import h5py as h5
	# rxnName = ["Ionization", ...] <- dictionary containing reaction name root string
	# for r in range(Nr):
	#	if reactionExpressionTypeList[r]:
	#		fileName = "{0:s}.{1:08d}.h5".format(rxnName[r], iSample)
	#		f = h5.File(filename, "r")
	#		D = f["table"]

    for i in range(Nr):
        if reactionExpressionTypelist[i]:
            #Nsample = 1
            #N300 = 200

            #root_dir = ".."
            #rate_file = open("{0:s}/BOLSIGChemistry_6SpeciesRates/reaction300K_{1:d}.txt".format(root_dir, i), 'r')
            #temp_file = open("{0:s}/BOLSIGChemistry_6SpeciesRates/reaction300K_Te.txt".format(root_dir), 'r')
            
            #rateCoeff = np.fromfile(rate_file)
            #rateCoeff = np.genfromtxt(rate_file)
            #rate_file.close()
            #rateCoeff = np.reshape(rateCoeff,[Nsample, N300]).T[:,iSample]

            #Te = np.fromfile(temp_file)
            #Te = np.genfromtxt(temp_file)
            #temp_file.close()
            #Te = np.reshape(Te,[Nsample, N300]).T[:,iSample]

            if (i < 15):
                f = h5.File("../BOLSIGChemistry_6SpeciesRates/{0:s}.h5".format(rxnNameDict[i]), 'r')
                dataset = f["table"]
            else:
                f = h5.File("../BOLSIGChemistry_6SpeciesRates/StepwiseExcitations.nominal.h5", 'r')
                dataset = f[rxnNameDict[i]]

            Te = dataset[:,0]
            Te /= 11604
            rateCoeff = dataset[:,1]
            rateCoeff /= 6.022e23

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
            indexPositive = np.where(Monotonicity>0.0)
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
            if (i < 13):
                rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
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

    # 5) Dump to screen
    params.print()
