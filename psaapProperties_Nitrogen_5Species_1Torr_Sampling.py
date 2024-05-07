import numpy as np
from scipy.interpolate import CubicSpline
import csv
import matplotlib.pyplot as plt
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


def setPsaapProperties_Nitrogen_5Species_1Torr_Sampling(gam, inputV0, inputVDC, params, Nr, iSample):
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
    nN2 = 3.22e22     # background number density of N2  [1/m^3] (corresponds to p = 1 Torr)
    np0 = 8e16        # "nominal" electron density [1/m^3]

    # masses
    # me = 9.10938356e-31        # mass of an electron [kg]
    # me = 5.489e-4              # mass of an electron [u]
    me = 0.511e6                 # mass of an electron [eV/c2]
    # mAr = 39.948               # mass of an argon atom [u]
    # mAr = 39.948 * 1.66054e-27 # mass of an argon atom [kg]
    mN2 = 26.0943e9              # mass of an N2 atom [eV/c2]
    # u = 931.4941e6             # eV/c2
    c = 299792458                # speed of light [m/s]
    se = 10.3                      # momentum cross section [A^2]

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
    L   = 2.00*0.005              # half-gap-width [m] (gap width is 2 cm)
    electrodeArea = np.pi*0.05**2 # electrode area [m^2] (electrode diameter = 0.1 m)
    # Add voltage uncertainty
    V0 += h5.File('../../../BOLSIGChemistry_Voltage/Voltage.%08d.h5' % (iSample), 'r')["V_Err"][0]

    # transport parameters
    nmue = 9.66e21   # N2 number density times electron mobility [1/(V*cm*s)]
    nmuN2i = 4.62e19 # Reduced mobility from Moseley et al (1969)
    nmuNi = 8.09e19  # Reduced mobility from Moseley et al (1969)
    nmuN = 0.0
    #nmuN2 = 0.0
    nDe  = 3.86e22   # N2 number density times electron diffusivity [1/(cm*s)]
    nDN2i= 1.194e18   # N2 number density times ion diffusivity [1/(cm*s)]
    nDNi = 2.1e18   # N2 number density times AR(m) diffusivity [1/(cm*s)]
    nDN  = 2.1e18   # Diff. coefficients from Moseley et al (1968)
    #nDN2 = 1.194e18

    # reaction parameters (NB: k_i = Ck*Ee^B*exp(-A/Ee))
    #                          Ee = 3/2*Te (Te in eV)
    #                          -> k_i = [Ck*(2/3)^B] * Ee^B * exp[-(3/2)*A/Ee]
    # nominal
    Ck = np.array([0.0,0.0,0.0,0.0,0.0,4.0e-19,7.2e-45,7.2e-45,7.2e-45,7.2e-45,7.2e-45,0.0,3.12e-35,3.12e-35,3.12e-35,3.12e-35,5.0e-39,1.0e-41,1.0e-41,1.0e-41,1.0e-41,1.0e-41,1.0e-18,1.66e-12,3.12e-35,3.12e-35,3.12e-35,3.12e-35,5.0e-39]) # pre-exponential factors [m^n/s]
    B  = np.array([0,0,0,0,0,-0.5,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-4.5,0,0,0,0,0,0,-0.7,-1.5,-1.5,-1.5,-1.5,-4.5]) # Temperature Power
    A  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # activation temperature [eV]
    dH = np.array([1.5,6.5,15.6,2.39,14.54,0.0,0.0,0.0,0.0,0.0,0.0,9.75,0.0,0.0,0.0,0.0,-14.54,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-15.6]) # energy lost per electron due to collisions [eV]
    dEps = np.array([0.0,15.6,14.54,0.0,0.0]) # E, N2+, N+, N, N2
    params.Z = np.array([-1,1,1,0,0]) # Charge Number array (must be in the same order as species are listed!!)
    params.posIonIdx = np.where(params.Z == 1)[0]
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
    nDN2i*= 100. # 1/(m*s)
    nDNi *= 100.
    nDN  *= 100.
    #nDN2 *= 100.
    nmue *= 100. # 1/(V*m*s)
    nmuN2i *= 100. # 1/(V*m*s)
    nmuNi *= 100.
    ks   *= 0.01 # m/s
    se   *= 1.0e-20  # m^2

    # 2) Compute "raw" transport parameters
    De  = nDe/nN2
    DN2i= nDN2i/nN2
    DNi = nDNi/nN2
    DN  = nDN/nN2
    #DN2  = nDN2/nN2

    mue = nmue/nN2
    muN2i = nmuN2i/nN2
    muNi = nmuNi/nN2
    muN = nmuN/nN2
    #muN2 = nmuN2/nN2

    # 3) Compute non-dimensional properties required by solver
    De    = De*tau/(L*L)
    DN2i    = DN2i*tau/(L*L)
    DNi    = DNi*tau/(L*L)
    DN    = DN*tau/(L*L)
    #DN2   = DN2*tau/(L*L)

    mue   = mue*V0*tau/(L*L)
    muN2i   = muN2i*V0*tau/(L*L)
    muNi   = muNi*V0*tau/(L*L)
    muN   = muN*V0*tau/(L*L)
    #muN2  = muN2*V0*tau/(L*L)

    Ck[0:3]   *= tau*nN2
    Ck[3:6]   *= tau*np0
    Ck[6:10]  *= tau*np0*np0
    Ck[10]    *= tau*np0*nN2
    Ck[11]    *= tau*nN2
    Ck[12:15] *= tau*np0*np0
    Ck[15]    *= tau*np0*nN2
    Ck[16:21] *= tau*np0*np0
    Ck[21]    *= tau*np0*nN2
    Ck[22:24] *= tau*np0
    Ck[24:27] *= tau*np0*np0
    Ck[27]    *= tau*np0*nN2
    Ck[28]    *= tau*np0*np0

    A        = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH       = dH/e0
    qStar    = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha    = qe*np0*L*L/(V0*eps0)
    ks       = ks*tau/L
    p0       = p/qe/np0
    kappaB   = (2./3.)*tau/(L*L*np0*kB)*0.026
    #kappaB   = 4.878171165833662*1.6129 # non-dimensional thermal conductivity of background specie
                                # (2/3)*tau/L**2*Kb/np0/kB,
                                # where Kb is the thermal conductivity of background specie

    params.beta = np.array([[1,1,2,1,2,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1],                     # E
                            [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,2,1,1,1,0,0,1,0,0,0,0],                     # N2+
                            [0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0],                     # N+
                            [0,0,0,1,0,1,0,0,0,1,0,2,1,1,2,1,1,0,0,0,1,0,0,2,0,0,1,0,0],                     # N
                            [1,1,0,0,0,0,1,1,1,1,2,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,1,2,1]], dtype=np.int64)    # N2

    params.alfa = np.array([[1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,2,1,0,0,0,0,0,1,1,1,1,1,2],                     # E
                            [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,2,1,1,1,1],                     # N2+
                            [0,0,0,0,0,1,0,0,1,0,0,0,1,2,1,1,1,1,1,2,1,1,0,0,0,1,0,0,0],                     # N+
                            [0,0,0,1,1,0,2,2,2,3,2,0,0,0,1,0,0,1,1,1,2,1,1,0,0,0,1,0,0],                     # N
                            [1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0]], dtype=np.int64)    # N2
    
    # Rxn1:  E + N2         ->   E + N2 [Vibrational]
    # Rxn2:  E + N2         ->   E + N2 [Electronic]
    # Rxn3:  E + N2         ->   2E + N2+ 
    # Rxn4:  E + N          ->   E + N [Excited]
    # Rxn5:  E + N          ->   2E + N+
    # Rxn6:  E + N+         ->   N
    # Rxn7:  2N + E         ->   N2 + E
    # Rxn8:  2N + N2+       ->   N2 + N2+
    # Rxn9:  2N + N+        ->   N2 + N+
    # Rxn10: 2N + N         ->   N2 + N
    # Rxn11: 2N + N2        ->   2N2
    # Rxn12:  E + N2        ->   E + 2N
    # Rxn13:  E + N2+ + N+  ->   N + N2+
    # Rxn14:  E + N+ + N+   ->   N + N+
    # Rxn15:  E + N + N+    ->   2N
    # Rxn16:  E + N2 + N+   ->   N + N2
    # Rxn17: 2E + N+        ->   E + N
    # Rxn18: N+ + N + E     ->   N2+ + E
    # Rxn19: N+ + N + N2+   ->   2N2+
    # Rxn20: N+ + N + N+    ->   N2+ + N+
    # Rxn21: N+ + N + N     ->   N2+ + N
    # Rxn22: N+ + N + N2    ->   N2+ + N2
    # Rxn23: N2+ + N        ->   N+ + N2
    # Rxn24: E + N2+        ->   2N
    # Rxn25: E + N2+ + N2+  ->   N2 + N2+
    # Rxn26: E + N2+ + N+   ->   N2 + N+
    # Rxn27: E + N2+ + N    ->   N2 + N
    # Rxn28: E + N2+ + N2   ->   2N2
    # Rxn29: 2E + N2+       ->   E + N2

    rxnNameDict = { 0: "Excitation_Vibrational",
                    1: "Excitation_Electronic",
                    2: "Ionization_N2",
                    3: "Excitation_N",
                    4: "Ionization_N",
                    5: "E + N+ => N",
                    6: "2N + E => N2 + E",
                    7: "2N + N2+ => N2 + N2+",
                    8: "2N + N+ => N2 + N+",
                    9: "2N + N => N2 + N",
                    10: "2N + N2 => N2 + N2",
                    11: "Dissociation",
                    12: "E + N2+ + N+ => N + N2+",
                    13: "E + N+ + N+ => N + N+",
                    14: "E + N + N+ => N + N",
                    15: "E + N2 + N+ => N + N2",
                    16: "2E + N+ => E + N",
                    17: "N+ + N + E => N2+ + E",
                    18: "N+ + N + N2+ => N2+ + N2+",
                    19: "N+ + N + N+ => N2+ + N+",
                    20: "N+ + N + N => N2+ + N",
                    21: "N+ + N + N2 => N2+ + N2",
                    22: "N2+ + N => N+ + N2",
                    23: "E + N2+ => 2N",
                    24: "E + N2+ + N2+ => N2 + N2+",
                    25: "E + N2+ + N+ => N2 + N+",
                    26: "E + N2+ + N => N2 + N",
                    27: "E + N2+ + N2 => N2 + N2",
                    28: "2E + N2+ => E + N2"}

    # 4) Set values in params class
    params.D[0]    = De
    params.D[1]    = DN2i
    params.D[2]    = DNi
    params.D[3]    = DN
    #params.D[4]    = DN2

    params.mu[0]   = mue
    params.mu[1]   = muN2i
    params.mu[2]   = muNi
    params.mu[3]   = muN
    #params.mu[4]   = muN2

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
    params.nAronp0 = nN2 / np0
    params.p0      = p0
    params.Tg0     = Tg0
    params.EC      = 2.0 * me / mN2 \
        * np.sqrt(16.0 * (me + mN2) * e0 * c**2
                  / (3.0 * np.pi * me * mN2)) * se * nN2 * tau
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


    reactionExpressionTypelist =  np.array([True,True,True,True,True,False,False,False,False,False,False,True, # Rxns 1-12
                                           False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]) # Rxns 13-29
    
    thresholded_rxn = np.array([True,True,True,True,True,False,False,False,False,False,False,True,
                                False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])
    reactionsList = []
    LOGFilename = 'interpolationSample%s.log' %str(iSample)
    f = open(LOGFilename, 'w')

    for i in range(Nr):
        sample_root_dir = '../../../BOLSIGChemistry_N2_5SpeciesRates'
        if reactionExpressionTypelist[i]:
            filestring = sample_root_dir + '/' + rxnNameDict[i]
            fileName = '%s.%08d.h5'%(filestring, iSample)
            f = h5.File(fileName, 'r')
            dataset = f["table"]

            Te = dataset[:,0]
            Te /= 11604
            rateCoeff = dataset[:,1]
            rateCoeff /= 6.022e23

            ## Removing BOLSIG failures
            fail_inds = []
            for j in range(len(rateCoeff)):
                if rateCoeff[j] == 0.0 and j > np.nonzero(rateCoeff)[0][0]:
                    fail_inds.append(j)

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
                Positive = np.full(Monotonicity.shape, False, dtype=bool)
                Positive[indexPositive] = True
            
                indices = Nan + Inf + Positive
                
                lastFalse = np.nonzero(rateCoeff)[0][0]
                
                # Transformation to log scale.
                TeLog = np.log(Te)

                # Compute the slope of the rate coefficient between its first two non-zero values.
                # Finite differences are used.
                dydx = (rateCoeff[lastFalse + 1] - rateCoeff[lastFalse]) / (Te[lastFalse + 1] - Te[lastFalse])

                # Arrhenius form: kf = A * exp(-C / Te)
                #C = Te[lastFalse]**2.0*dydx / rateCoeff[lastFalse]
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
            if (i < 3) or i == 11:
                rateCoeffLog += - np.log(1.0/tau) + np.log(nN2)
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
            fileName = '%s/Arrhenius.%08d.h5'%(sample_root_dir, iSample)
            f = h5.File(fileName, 'r')
            arrh_Coeffs = f[rxnNameDict[i]][...]
            A, B, C = arrh_Coeffs
            if i > 5 and i < 10:
                A /= 6.022e23**2
                A *= tau*np0*np0
            elif i > 11 and i < 15:
                A /= 6.022e23**2
                A *= tau*np0*np0
            elif i > 15 and i < 21:
                A /= 6.022e23**2
                A *= tau*np0*np0
            elif i > 23 and i < 27 or i == 28:
                A /= 6.022e23**2
                A *= tau*np0*np0
            elif i == 10 or i == 15 or i == 21 or i == 27:
                A /= 6.022e23**2
                A *= tau*np0*nN2
            else:
                A /= 6.022e23
                A *= tau*np0

            A *= ((2./3.)*11604)**B
            C *= 1.5/(e0*11604)

            rxn   = eval("lambda energy :" + f"{A} * energy**{B} * np.exp(-{C} / energy)")
            rxn_T = eval("lambda energy :" + f"{A} * energy**({B}-1) * np.exp(-{C} / energy) * ({B} + {C} / energy)")

            reaction = Reaction(rxnAlfa = params.alfa[:,[i]], rxnBeta = params.beta[:,[i]],
                                rxnBolsig = reactionExpressionTypelist[i],
                                kf = rxn, kf_T = rxn_T)
            reactionsList.append(reaction)

    params.reactionsList = reactionsList

    ## Electron Transport Data
    diffList = []
    transport = h5.File("../../../BOLSIGChemistry_N2_5SpeciesRates/Transport.%08d.h5" % (iSample), 'r')
    NDe_v_Te = transport["diffusivity"]
    Te_trans = NDe_v_Te[:,0]
    Te_trans /= 11604
    idx = np.where(Te_trans <= 1.0)[0][-1]
    De_interp = (NDe_v_Te[:,1]/nN2)*tau/(L*L)
    De_interp[0:idx] = De_interp[idx]
    De_spline = CubicSpline(Te_trans, De_interp)
    De_Te_spline = CubicSpline.derivative(De_spline)
    diffusivity = Diffusivity(interpolate = False, D_expression = De_spline, D_T_expression = De_Te_spline)
    diffList.append(diffusivity)

    Ns = 5
    for i in range(1, Ns):
        diffList.append(Diffusivity(interpolate = False))

    params.diffusivityList = diffList

    muList = []
    Nmue_v_Te = transport["mobility"]
    mue_interp = (Nmue_v_Te[:,1]/nN2)*V0*tau/(L*L)
    mue_interp[0:idx] = mue_interp[idx]
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
