import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
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


def setPsaapProperties_6Species_Sampling_1Torr_Sandia(gam, inputV0, inputVDC, params, Nr, iSample):
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
    nAr = 3.22e22     # background number density of Ar [1/m^3] (corresponds to p = 1 Torr)
    np0 = 8e16        # "nominal" electron density [1/m^3]

    # masses
    # me = 9.10938356e-31        # mass of an electron [kg]
    # me = 5.489e-4              # mass of an electron [u]
    me = 0.511e6                 # mass of an electron [eV/c2]
    # mAr = 39.948               # mass of an argon atom [u]
    # mAr = 39.948 * 1.66054e-27 # mass of an argon atom [kg]
    mAr = 37.2158e9              # mass of an electron [eV/c2]
    # u = 931.4941e6             # eV/c2
    c = 299792458                # speed of light [m/s]
    se = 40                      # momentum cross section [A^2]

    # nominal electron energy
    e0 = 1.0  # [eV]

    # pressure
    p  = 133.3*1.5      # [J/m^3] *1.5 to convert it to energy (1 Torr)

    # gas energy at the wall
    Tg0 = 0.038778    # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot

    # characteristics of driving voltage
    V0  = inputV0                 # amplitude of driving voltage [V]
    verticalShift = inputVDC      # DC voltage (vertical shift in driving voltage)
    tau = (1./13.56e6)             # period of driving voltage [s]
    L   = 4.00*0.005              # half-gap-width [m] (gap width is 2 cm)
    electrodeArea = np.pi*0.07**2 # electrode area [m^2] (electrode diameter = 0.1 m)

    # Add voltage uncertainty
    V0 += h5.File('../../../BOLSIGChemistry_Voltage/Voltage.%08d.h5' % (iSample), 'r')["V_Err"][0]

    # transport parameters
    nmue = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
    nmum = 0.0
    nmui = 8.0e19
    nDe  = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi  = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]
    nDm  = 2.42e18   # argon number density times metastable diffusivity [1/(cm*s)]

    # reaction parameters (NB: k_i = Ck*Ee^B*exp(-A/Ee))
    #                          Ee = 3/2*Te (Te in eV)
    #                          -> k_i = [Ck*(2/3)^B] * Ee^B * exp[-(3/2)*A/Ee]
    # nominal
    Ck = np.array([2.0e-13,2.1e-15,5.0e-16,6.4e-16,2.1e-21,1.32e8,1.72e7,1.50e7,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0e-18,4.0e-19,0.0,0.0,0.0,0.0,2.5e-17,2.5e-17,1.0e-15,1.0e-15,0.0,0.0]) # pre-exponential factors [m^3/s]
    B  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,0,0,0,0,0,0,0,0,0,0])
    A  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # activation temperature [eV]
    dH = np.array([0.0,-7.541,-10.577,-7.393,0.0,0.0,0.0,0.0,11.577,11.725,13.168,15.76,-11.577,4.183,0.148,1.592,2.592,-1.444,-1.592,-11.725,-0.148,1.444,0.0,0.0,-4.183,-4.035,-2.592,-15.76,0.0,0.0,-8.985,-9.133,4.035,-13.168]) # energy lost per electron due to ionization rxn [eV]
    dEps = np.array([0.0,15.76,11.577,11.725,13.168,0.0])

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
    # kB   = 8.62e−5 # Boltzmann constant [eV/K]


    ###################################################################
    # Calculate non-dimensional parameters
    ###################################################################

    # 1) Convert input units to base SI (except eV)
    nDe  *= 100. # 1/(m*s)
    nDi  *= 100. # 1/(m*s)
    nDm  *= 100.
    nmue *= 100. # 1/(V*m*s)
    nmui *= 100. # 1/(V*m*s)
    ks   *= 0.01 # m/s
    se   *= 1.0e-20  # m^2

    # 2) Compute "raw" transport parameters
    De  = nDe/nAr
    Di  = nDi/nAr
    Dm  = nDm/nAr

    mue = nmue/nAr
    mui = nmui/nAr
    mum = nmum/nAr

    # 3) Compute non-dimensional properties required by solver
    De    = De*tau/(L*L)
    Di    = Di*tau/(L*L)
    Dm    = Dm*tau/(L*L)

    mue   = mue*V0*tau/(L*L)
    mui   = mui*V0*tau/(L*L)
    mum   = mum*V0*tau/(L*L)

    Ck[0:4]  *= tau*np0
    Ck[4]    *= tau*nAr
    Ck[5:8]  *= tau
    Ck[8:12] *= tau*nAr
    Ck[12:24]  *= tau*np0
    Ck[24:28] *= tau*np0*np0
    Ck[28:30] *= tau*nAr
    Ck[30:] *= tau*np0
    A       = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH      = dH/e0
    qStar   = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha   = qe*np0*L*L/(V0*eps0)
    ks      = ks*tau/L
    p0      = p/qe/np0
    kappaB  = 4.878171165833662*1.6129 # non-dimensional thermal conductivity of background specie
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

    rxnNameDict = {0: "Ar(m) + Ar(m) => Ar + Ar",
                   1: "Ar(m) + Ar(r) => E + Ar + Ar+",
                   2: "Ar(4p) + Ar(4p) => E + Ar + Ar+",
                   3: "Ar(m) + Ar(m) => E + Ar + Ar+",
                   4: "Ar + Ar(m) => Ar + Ar",
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

    params.mu[0]   = mue
    params.mu[1]   = mui
    params.mu[2]   = mum

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


    reactionExpressionTypelist =  np.array([False,False,False,False,False,False,False,False,
                                           True,True,True,True,True,True,True,True,True,True,True,True,True,True,
                                           False,False,True,True,True,True,False,False,False,False,True,True])

    thresholded_rxn = np.array([False, False, False, False, False, False, False, False,
                                True, True, True, True, False, True, True, True, True, False, False, False, False,
                                True, False, False, False, False, False, False, False, False, False, False, True,
                                False])

    reactionsList = []
    LOGFilename = 'interpolationSample%s.log'%str(iSample)
    f = open(LOGFilename, 'w')

    for i in range(Nr):
        sample_root_dir = "../../../BOLSIGChemistry_6SpeciesRates"
        if reactionExpressionTypelist[i]:
            if i < 14 or i == 16 or i == 19 or i > 23:
                fileString = sample_root_dir + "/" + rxnNameDict[i]
                fileName = "%s.%08d.h5" % (fileString, iSample)
                f = h5.File(fileName, 'r')
                dataset = f["table"]
            else:
                fileString = sample_root_dir + "/" + "StepExcitation"
                fileName = "%s.%08d.h5" % (fileString, iSample)
                f = h5.File(fileName, 'r')
                dataset = f[rxnNameDict[i]]
            
            rateCoeff = dataset[:,1]
            if i > 23 and i < 28:
                rateCoeff /= 6.022e23**2
            else:
                rateCoeff /= 6.022e23
            Te = dataset[:,0]
            Te /= 11604.

            ## Removing BOLSIG failures
            fail_inds = []
            for j in range(len(rateCoeff)):
                if rateCoeff[j] == 0.0 and j > np.nonzero(rateCoeff)[0][0]:
                    fail_inds.append(j)
            if len(fail_inds) != 0:
                rateCoeff[0:fail_inds[-1]] = 0.0

            #Te = np.delete(Te, fail_inds)
            #rateCoeff = np.delete(rateCoeff, fail_inds)

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
                
                lastFalse = np.nonzero(rateCoeff)[0][0]
                #lastFalse = np.where(indices==False)[-1][-1] + 2
                #for k in range(len(Te)):
                #    if (Te[k] < 6.0 and indices[k] == False):
                #        lastFalse = k+2
    
                # Transformation to log scale.
                TeLog = np.log(Te)
    
                # Compute the slope of the rate coefficient between its first two non-zero values.
                # Finite differences are used.
                dydx = (rateCoeff[lastFalse + 1] - rateCoeff[lastFalse]) \
                     / (Te[lastFalse + 1] - Te[lastFalse])
    
                # Arrhenius form: kf = A * exp(-C / Te)
                #if i == 11:
                C = Te[lastFalse+1]*Te[lastFalse]*np.log(rateCoeff[lastFalse+1]/rateCoeff[lastFalse])**1.5/(Te[lastFalse+1]-Te[lastFalse])
                #else:
                #    C = Te[lastFalse]**2.0*dydx / rateCoeff[lastFalse]

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
                
            #print("RXN {}:".format(i+1))
            #print(rateCoeffLog)
            # Nondimensionalization in log scale.
            if i < 12:
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
            logging.warning('Interpolation info (Reaction %s):', i)
            logging.warning('First non-zero entry in the rate coefficient: %s', I[0][0])
            logging.warning('Monotonicity of the rate coefficient start from entry: %s', lastFalse)
            logging.warning('Position of possible duplicates in mean energy array: %s',
                          TeDuplicateindsForLog[0])

        else:
            fileName = "%s/Arrhenius.%08d.h5" % (sample_root_dir, iSample)
            f = h5.File(fileName, 'r')
            arrh_Coeffs = f[rxnNameDict[i]][...]
            A, B, C = arrh_Coeffs

            # Non-dimensionalize the coefficients
            if i < 4:
                A /= 6.022e23
                A *= tau*np0
            elif i == 4:
                A /= 6.022e23
                A *= tau*nAr
            elif i > 4 and i < 8:
                A *= tau
            elif i > 27 and i < 30:
                A /= 6.022e23
                A *= tau*nAr
            else:
                A /= 6.022e23
                A *= tau*np0

            A *= ((2./3.)*11604)**B 
            C *= 1.5/(e0*11604)

            rxn = eval("lambda energy :" + f"{A} * energy**{B} * np.exp(-{C} / energy)")
            rxn_T = eval("lambda energy :" + f"{A} * energy**({B}-1) * np.exp(-{C} / energy) * ({B} + {C} / energy)")

            reaction = Reaction(rxnAlfa = params.alfa[:,[i]], rxnBeta = params.beta[:,[i]],
                                rxnBolsig = reactionExpressionTypelist[i],
                                kf = rxn, kf_T = rxn_T)
            reactionsList.append(reaction)

    params.reactionsList = reactionsList

    diffList = []
    # Data from BOLSIG
    # Te in [eV]
    # De * N in [1/(m*s)]
    transport = h5.File("../../../BOLSIGChemistry_6SpeciesRates/Transport.%08d.h5" % (iSample), 'r')
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
    # Data from BOLSIG
    # Te in [eV]
    # Mue * N in [1/(V*m*s)]
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
