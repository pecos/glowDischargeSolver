import numpy as np
from scipy.interpolate import CubicSpline
import csv
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
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


def setPsaapProperties_5Species_5Torr_Nominal(gam, inputV0, inputVDC, params, Nr, iSample):
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
    nAr = 1.62e23     # background number density of Ar [1/m^3] (corresponds to p=100 mTorr)
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
    p  = 666.6*1.5      # [J/m^3] *1.5 to convert it to energy (1 Torr)

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
    Ck = np.array([0.0,0.0,0.0,0.0,5.6e-14,5.0e-18,1.25e-19,0.0,0.0,0.0,2.29e8,4.47e6,0.0,0.0,0.0,0.0,2.0e-13,6.4e-16,1.56e-16,1.76e-15,7.78e-18,2.3e-21]) # pre-exponential factors [m^3/s]
    B  = np.array([0,0,0,0,0.61,0,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    A  = np.array([0,0,0,0,2.61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # activation temperature [eV]
    dH = np.array([11.548,12.023,15.76,4.212,3.737,0,0,-15.76,-4.212,-3.737,0,0,0.475,-0.475,-11.548,-12.023,0,-7.336,-8.286,-7.811,0,0]) # energy lost per electron due to ionization rxn [eV]
    dEps = np.array([0.0,15.76,11.548,12.023,0.0]) # E, AR+, AR(m), AR(r), AR

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

    Ck[0:3]  *= tau*nAr
    Ck[3:7]  *= tau*np0
    Ck[7:10] *= tau*np0*np0
    Ck[10:12]*= tau
    Ck[12:20]*= tau*np0
    Ck[20:]  *= tau*nAr
    A        = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH       = dH/e0
    qStar    = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha    = qe*np0*L*L/(V0*eps0)
    ks       = ks*tau/L
    p0       = p/qe/np0
    kappaB   = 4.878171165833662*1.6129 # non-dimensional thermal conductivity of background specie
                                # (2/3)*tau/L**2*Kb/np0/kB,
                                # where Kb is the thermal conductivity of background specie

    params.beta = np.array([[1,1,2,2,2,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0],                     # E
                            [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0],                     # AR+
                            [1,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0],                     # AR(m)
                            [0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],                     # AR(r)
                            [0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,2,1,1,1,1,2]], dtype=np.int64)    # AR

    params.alfa = np.array([[1,1,1,1,1,1,1,2,2,2,0,0,1,1,1,1,0,0,0,0,0,0],                     # E
                            [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],                     # AR+
                            [0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,2,2,0,1,0,1],                     # AR(m)
                            [0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,0,0,2,1,1,0],                     # AR(r)
                            [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]], dtype=np.int64)    # AR
	# Rxn1:  E + Ar         ->   E + Ar(m)
	# Rxn2:  E + Ar         ->   E + Ar(r)
	# Rxn3:  E + Ar         ->   2E + Ar+
	# Rxn4:  E + Ar(m)      ->   2E + Ar+
	# Rxn5:  E + Ar(r)      ->   2E + Ar+
	# Rxn6:  E + Ar+        ->   Ar(m)
	# Rxn7:  E + Ar+        ->   Ar(r)
	# Rxn8:  2E + Ar+       ->   E + Ar
	# Rxn9:  2E + Ar+       ->   E + AR(m)
    # Rxn10: 2E + Ar+       ->   E + Ar(r)
    # Rxn11: Ar(r)          ->   Ar
    # Rxn12: Ar(r)          ->   Ar(m)
    # Rxn13: E + Ar(m)      ->   E + Ar(r)
    # Rxn14: E + AR(r)      ->   E + Ar(m)
    # Rxn15: E + AR(m)      ->   E + Ar
    # Rxn16: E + AR(r)      ->   E + Ar
    # Rxn17: 2Ar(m)         ->   2Ar
    # Rxn18: 2Ar(m)         ->   E + Ar+ + Ar
    # Rxn19: 2Ar(r)         ->   E + Ar+ + Ar
    # Rxn20: Ar(m) + Ar(r)  ->   E + Ar+ + Ar
    # Rxn21: Ar(r) + Ar     ->   Ar(m) + Ar
    # Rxn22: Ar(m) + Ar     ->   2Ar

    rxnNameDict = { 0: "1s-metastable",
                    1: "1s-resonance",
                    2: "Ionization",
                    3: "StepIonization",
                    4: "StepIonization",
                    5: "E + Ar+ => Ar(m)",
                    6: "E + Ar+ => Ar(r)",
                    7: "3BdyRecomb-ground",
                    8: "3BdyRecomb-metastable",
                    9: "3BdyRecomb-metastable",
                   10: "Ar(r) => Ar",
                   11: "Ar(r) => Ar(m)",
                   12: "E + Ar(m) => E + Ar(r)",
                   13: "E + Ar(r) => E + Ar(m)",
                   14: "Deexci-metastable",
                   15: "Deexci-resonance",
                   16: "2Ar(m) => 2Ar",
                   17: "2Ar(m) => E + Ar+ + Ar",
                   18: "2Ar(r) => E + Ar+ + Ar",
                   19: "Ar(m) + Ar(r) => E + Ar+ + Ar",
                   20: "Ar(r) + Ar => Ar(m) + Ar",
                   21: "Ar(m) + Ar => 2Ar",}

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
                               f"{params.A[21]} * energy**{params.B[21]} * np.exp(-{params.C[21]} / energy)"]

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
                                f"{params.A[21]} * (energy**({params.B[21]}-1)) * np.exp(-{params.C[21]}/energy) * ({params.B[21]} + {params.C[21]}/energy)"]
           

    reactionExpressionTypelist = np.array([True,True,True,True,False,False,False,True,True,True,False,False,True,True,True,True,False,False,False,False,False,False])

    reactionsList = []
    LOGFilename = 'interpolationSample.log'
    f = open(LOGFilename, 'w')

    for i in range(Nr):
        if reactionExpressionTypelist[i]:
            if i < 12 or i > 13:
                f = h5.File("../../../BOLSIGChemistry_5SpeciesRates/{0:s}.h5".format(rxnNameDict[i]), 'r')
                dataset = f["table"]
            else:
                f = h5.File("../../../BOLSIGChemistry_5SpeciesRates/StepwiseExcitations.nominal.h5", 'r')
                dataset = f[rxnNameDict[i]]

            Te = dataset[:,0]
            Te /= 11604
            rateCoeff = dataset[:,1]
            if i > 6 and i < 10:
                rateCoeff /= 6.022e23**2
            else:
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
            if (i < 3):
                rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
            elif i > 6 and i < 10:
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
    transport = h5.File("../../../BOLSIGChemistry_Transport/nominal_transport.h5", 'r')
    NDe_v_Te = transport["diffusivity"]
    Te_trans = NDe_v_Te[:,0]
    Te_trans /= 11604
    print("Te_min = {0:.6e}".format(NDe_v_Te[0,0]))
    print("Te_max = {0:.6e}".format(NDe_v_Te[-1,0]))
    De_interp = (NDe_v_Te[:,1]/nAr)*tau/(L*L)
    De_spline = CubicSpline(Te_trans, De_interp)
    De_Te_spline = CubicSpline.derivative(De_spline)
    diffusivity = Diffusivity(interpolate = True, D_expression = De_spline, D_T_expression = De_Te_spline)
    diffList.append(diffusivity)

    Ns = 5
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

    Ns = 5
    for i in range(1, Ns):
        muList.append(Mobility(interpolate = False))

    params.mobilityList = muList
    
    # 5) Dump to screen
    params.print()
