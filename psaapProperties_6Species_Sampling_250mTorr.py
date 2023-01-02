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


def setPsaapProperties_6Species_Sampling_250mTorr(gam, inputV0, inputVDC, params, Nr, iSample):
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
    nAr = 8.05e21     # background number density of Ar [1/m^3] (corresponds to p=100 mTorr)
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
    p  = 33.33*1.5      # [J/m^3] *1.5 to convert it to energy (1 Torr)

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
    nmui = 8.0e19
    nDe  = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi  = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]
    nDm  = 2.42e18   # argon number density times metastable diffusivity [1/(cm*s)]

    # reaction parameters (NB: k_i = Ck*Ee^B*exp(-A/Ee))
    #                          Ee = 3/2*Te (Te in eV)
    #                          -> k_i = [Ck*(2/3)^B] * Ee^B * exp[-(3/2)*A/Ee]
    # nominal
    Ck = np.array([2.0e-7,2.1e-9,5.0e-10,6.4e-10,2.1e-15,1.32e8,0.0,3.0e7,3.0e7,0.0,0.0,0.0,0.0,4.3e-10,0.0,3.7e-8,8.9e-7,1.8e-7,3.0e-7,3.0e-7,4.3e-10,9.1e-7,8.9e-7]) # pre-exponential factors [cm^3/s]
    B  = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.74,0.0,0.0,0.51,0.61,0.51,0.51,0.74,0.0,0.51])
    A  = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.59,2.61,0.0,0.0,0.0,0.0,1.59]) # activation temperature [eV]
    dH = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,11.548,11.624,12.907,15.76,-11.548,4.212,0.076,1.359,2.853,-1.283,-1.359,-11.624,-0.076,0.983]) # energy lost per electron due to ionization rxn [eV]
    dEps = np.array([0.0,15.76,11.548,11.624,12.907,0.0])

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
    Ck[0:5]  *= 1e-6 # m^3/s
    Ck[5:9]  *= 1 # 1/s
    Ck[9:]   *= 1e-6 # m^6/s
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
    Ck[5:9]  *= tau
    Ck[9:13] *= tau*nAr
    Ck[13:]  *= tau*np0
    A       = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH      = dH/e0
    qStar   = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha   = qe*np0*L*L/(V0*eps0)
    ks      = ks*tau/L
    p0      = p/qe/np0
    kappaB  = 4.878171165833662*1.6129 # non-dimensional thermal conductivity of background specie
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

    rxnNameDict = {0: "Ar(m) + Ar(m) => Ar + Ar",
                   1: "Ar(m) + Ar(r) => E + Ar + Ar+",
                   2: "Ar(4p) + Ar(4p) => E + Ar + Ar+",
                   3: "Ar(m) + Ar(m) => E + Ar + Ar+",
                   4: "Ar + Ar(m) => Ar + Ar",
                   5: "Ar(r) => Ar",
                   6: "Ar(4p) => Ar",
                   7: "Ar(4p) => Ar(m)",
                   8: "Ar(4p) => Ar(r)",
                   9: "1s-metastable",
                   10: "1s-resonance",
                   11: "2p-lumped",
                   12: "Ionization",
                   13: "E + Ar(m) => E + Ar",
                   14: "StepIonization",
                   15: "E + Ar(m) => E + Ar(r)",
                   16: "E + Ar(m) => E + Ar(4p)",
                   17: "E + Ar(4p) => E + E + Ar+",
                   18: "E + Ar(4p) => E + Ar(r)",
                   19: "E + Ar(4p) => E + Ar(m)",
                   20: "E + Ar(r) => E + Ar",
                   21: "E + Ar(r) => E + Ar(m)",
                   22: "E + Ar(r) => E + Ar(4p)"}

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


    reactionExpressionTypelist =  np.array([False,False,False,False,False,False,False,False,False,
                                           True,True,True,True,False,True,True,True,False,True,True,False,True,True])

    reactionsList = []
    LOGFilename = 'interpolationSample%s.log'%str(iSample)
    f = open(LOGFilename, 'w')

    for i in range(Nr):
        sample_root_dir = "../../BOLSIGChemistry_6SpeciesRates"
        if reactionExpressionTypelist[i]:
            if (i < 15):
                fileString = sample_root_dir + "/" + rxnNameDict[i]
                fileName = "%s.%08d.h5" % (fileString, iSample)
                f = h5.File(fileName, 'r')
                dataset = f["table"]
            else:
                fileString = sample_root_dir + "/" + "StepwiseExcitations"
                fileName = "%s.%08d.h5" % (fileString, iSample)
                f = h5.File(fileName, 'r')
                dataset = f[rxnNameDict[i]]
            
            rateCoeff = dataset[:,1]
            rateCoeff /= 6.022e23
            Te = dataset[:,0]
            Te /= 11604.

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
                if (Te[k] < 4.5 and indices[k] == False):
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
            if i == 9:
                rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
            elif i == 10:
                rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
            elif i == 11:
                rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
            elif i == 12:
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
            if (i < 4):
                A /= 6.022e23
                A *= tau*np0
            elif (i == 4):
                A /= 6.022e23
                A *= tau*nAr
            elif (i > 4 and i < 9):
                A *= tau
            #elif (i > 8 and i < 13):
            #    A /= 6.022e23
            #    A *= tau*nAr
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
    #NDe_v_Te = np.array([ [-0.05, 1.8e25],         [0.0, 1.8e25],           [0.05, 1.8e25],          [0.1, 1.8e25],
    #                      [0.2103718, 1.8e25],     [0.2244455, 1.8e25],     [0.2390528, 1.8e25],     [0.2544605, 1.8e25],
    #                      [0.2710021, 1.8e25],     [0.2886776, 1.8e25],     [0.3078205, 1.8e25],     [0.3286309, 1.8e25],
    #                      [0.3513089, 1.8e25],     [0.3760546, 1.8e25],     [0.4032682, 1.8e25],     [0.4331498, 1.8e25],
    #                      [0.4659662, 1.8e25],     [0.5021843, 1.8e25],     [0.5424044, 1.8e25],     [0.5868266, 1.8e25],
    #                      [0.6359845, 1.8e25],     [0.690345, 1.8e25],      [0.749041, 1.8e25],      [0.813073, 1.8e25],
    #                      [1.217942, 1.46E+25],    [1.319326, 1.39E+25],    [1.430048, 1.32E+25],    [1.550108, 1.26E+25],
    #                      [1.681507, 1.20E+25],    [1.823578, 1.15E+25],    [1.977655, 1.09E+25],    [2.143071, 1.05E+25],
    #                      [2.319826, 9.98E+24],    [2.508587, 9.54E+24],    [2.709354, 9.13E+24],    [2.916124, 8.75E+24],
    #                      [3.112889, 8.45E+24],    [3.272969, 8.24E+24],    [3.383691, 8.14E+24],    [3.456394, 8.09E+24],
    #                      [3.505085, 8.07E+24],    [3.544438, 8.05E+24],    [3.579789, 8.03E+24],    [3.616474, 8.01E+24],
    #                      [3.656494, 7.97E+24],    [3.701183, 7.93E+24],    [3.751875, 7.87E+24],    [3.807903, 7.81E+24],
    #                      [3.871268, 7.75E+24],    [3.941303, 7.68E+24],    [4.018675, 7.61E+24],    [4.102717, 7.53E+24],
    #                      [4.193429, 7.5E+24],     [4.290811, 7.5E+24],     [4.39553, 7.5E+24],      [4.507586, 7.5E+24],
    #                      [5., 7.5E+24],           [6., 7.5E+24],           [7., 7.5E+24],           [8., 7.5E+24],
    #                      [9., 7.5E+24],           [10., 7.5E+24],          [11., 7.5E+24],          [12., 7.5E+24]])

    transport = h5.File("../../BOLSIGChemistry_6SpeciesRates/Transport.%08d.h5" % (iSample), 'r')
    #Te = NDe_v_Te[:,0]
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

    Ns = 6
    for i in range(1, Ns):
        diffList.append(Diffusivity(interpolate = False))

    params.diffusivityList = diffList

    muList = []
    # Data from BOLSIG
    # Te in [eV]
    # Mue * N in [1/(V*m*s)]
    #Nmue_v_Te = np.array([[0.02846756, 1.289e+26], [0.02975487, 1.33e+26],   [0.03173586, 1.383e+26],  [0.03477738,1.448e+26],
    #                      [0.03937968, 1.523e+26], [0.04614973, 1.604e+26],  [0.05561446, 1.679e+26],  [0.0681674,1.735e+26],
    #                      [0.0835084, 1.749e+26],  [0.1008504, 1.718e+26],   [0.1187927, 1.639e+26],   [0.1363348,1.522e+26],
    #                      [0.1528764, 1.386e+26],  [0.1682841, 1.245e+26],   [0.1826913, 1.108e+26],   [0.1965649,9.809e+25],
    #                      [0.2103718, 8.662e+25],  [0.2244455, 7.641e+25],   [0.2390528, 6.74e+25],    [0.2544605,5.947e+25],
    #                      [0.2710021, 5.249e+25],  [0.2886776, 4.636e+25],   [0.3078205, 4.097e+25],   [0.3286309,3.623e+25],
    #                      [0.3513089, 3.206e+25],  [0.3760546, 2.839e+25],   [0.4032682, 2.517e+25],   [0.4331498,2.232e+25],
    #                      [0.4659662, 1.981e+25],  [0.5021843, 1.76e+25],    [0.5424044, 1.565e+25],   [0.5868266,1.392e+25],
    #                      [0.6359845, 1.239e+25],  [0.690345, 1.103e+25],    [0.749041,  9.8e+24],     [0.813073,8.699e+24],
    #                      [0.882441, 7.711e+24],   [0.957145, 6.828e+24],    [1.037185,  6.04e+24],    [1.123895,5.344e+24],
    #                      [1.217942, 4.729e+24],   [1.319326, 4.186e+24],    [1.430048,  3.707e+24],   [1.550108,3.283e+24],
    #                      [1.681507, 2.907e+24],   [1.823578, 2.574e+24],    [1.977655,  2.278e+24],   [2.143071,2.015e+24],
    #                      [2.319826, 1.779e+24],   [2.508587, 1.569e+24],    [2.709354,  1.384e+24],   [2.916124,1.228e+24],
    #                      [3.112889, 1.115e+24],   [3.272969, 1.055e+24],    [3.383691,  1.038e+24],   [3.456394,1.044e+24],
    #                      [3.505085, 1.054e+24],   [3.544438, 1.062e+24],    [3.579789,  1.064e+24],   [3.616474, 1.059e+24],
    #                      [3.656494, 1.048e+24],   [3.701183, 1.033e+24],    [3.751875,  1.014e+24],   [3.807903, 9.928e+23],
    #                      [3.871268, 9.697e+23],   [3.941303,  9.459e+23],   [4.018675, 9.22e+23],     [4.102717, 8.991e+23],
    #                      [4.193429, 8.773e+23],   [4.290811,  8.57e+23],    [4.39553, 8.383e+23],     [4.507586, 8.21e+23],
    #                      [4.627646, 8.051e+23],   [4.757711,  7.901e+23],   [4.899115, 7.757e+23],    [5.054526, 7.617e+23],
    #                      [5.227946, 7.479e+23],   [5.423377,  7.339e+23],   [5.646822, 7.198e+23],    [5.905618, 7.056e+23],
    #                      [6.20977, 6.91e+23],     [6.571284,  6.757e+23],   [7.01017, 6.607e+23],     [7.54377, 6.458e+23],
    #                      [8.19743, 6.303e+23],    [9.01117,   6.155e+23],   [10.02501, 6e+23],        [11.30565, 5.843e+23],
    #                      [12.89978, 5.677e+23],   [14.91412,  5.51e+23],    [17.41537, 5.326e+23],    [20.57028, 5.149e+23],
    #                      [24.51225, 4.968e+23],   [29.45472,  4.784e+23],   [35.6845, 4.602e+23],     [43.58845, 4.421e+23],
    #                      [53.86692, 4.269e+23],   [67.367,    4.134e+23],   [85.4427, 4.026e+23],     [100.0, 4.026e+23]])

    #Te = Nmue_v_Te[:,0]
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
