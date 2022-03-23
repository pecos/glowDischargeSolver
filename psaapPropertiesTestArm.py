import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

def setPsaapPropertiesTestArm(gam, inputV0, inputVDC, params, Nr):
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
    mAr = 37.2158e9              # mass of an electron [eV/c2]
    # u = 931.4941e6             # eV/c2
    c = 299792458                # speed of light [m/s]
    se = 40                      # momentum cross section [m^2]

    # nominal electron energy
    e0 = 1.0  # [eV]

    # pressure
    p  = 133.3224*1.5      # [J/m^3] *1.5 to convert it to energy (1 Torr)

    # gas energy at the wall
    Tg0 = 0.038778    # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot

    # characteristics of driving voltage
    V0  = inputV0                 # amplitude of driving voltage [V]
    verticalShift = inputVDC      # DC voltage (vertical shift in driving voltage)
    tau = (1./13.6e6)             # period of driving voltage [s]
    L   = 2.54*0.005              # half-gap-width [m] (gap width is 2.54cm)
    electrodeArea = np.pi*0.05**2 # electrode area [m^2] (electrode diameter = 0.1 m)

    # transport parameters
    nmue = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
    nmui = 4.65e19   # argon number density times ion mobility [1/(V*cm*s)]
    nmum = 1 / (np.sqrt(16.0 * (mAr + mAr) * 300 * 8.62e-5 * c**2
                        / (3.0 * np.pi * mAr * mAr)) * se * mAr * 1.6e-19 / c**2)
    nDe  = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi  = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]
    nDm  = 2.42e18   # argon number density times metastable diffusivity [1/(cm*s)]

    # reaction parameters (NB: k_i = Ck*exp(-A/Te))
    #Ck = np.array([1.235e-7,3.712e-8,2.05e-7,1.818e-9,2e-7]) # pre-exponential factors [cm^3/s]
    #A  = np.array([18.687,15.06,4.95,2.14,0.0])      # activation temperature [eV]
    #dH = np.array([15.7,11.56,4.14,-11.56,0.0])        # energy lost per electron due to ionization rxn [eV]

    #Ck = np.array([1.235e-7,3.712e-8,2.05e-7,1.818e-9]) # pre-exponential factors [cm^3/s]
    #A  = np.array([18.687,15.06,4.95,2.14])      # activation temperature [eV]
    #dH = np.array([15.7,11.56,4.14,-11.56])        # energy lost per electron due to ionization rxn [eV]

    #Ck = np.array([1.235e-7,0.0,0.0,0.0,0.0]) # pre-exponential factors [cm^3/s]
    #A  = np.array([18.687,15.06,4.95,2.14,0.0])      # activation temperature [eV]
    #dH = np.array([15.7,0.0,0.0,0.0,0.0])        # energy lost per electron due to ionization rxn [eV]

    # nominal
    # Ck = np.array([1.235e-7,3.712e-8,2.05e-7,1.818e-9,2e-7,6.2e-10,3.0e-15,1.1e-31]) # pre-exponential factors [cm^3/s]
    Ck = np.array([1.235e-7,3.712e-8,2.05e-7,4.0e-13,2e-7,5.0e-10,2.5e-15]) # pre-exponential factors [cm^3/s]

    # slow excitation rate
    #Ck = np.array([1.235e-7,0.5*3.712e-8,2.05e-7,1.818e-9,2e-7,6.2e-10,3.0e-15,1.1e-31]) # pre-exponential factors [cm^3/s]

    ## fast excitation rate
    #Ck = np.array([1.235e-7,2.0*3.712e-8,2.05e-7,1.818e-9,2e-7,6.2e-10,3.0e-15,1.1e-31]) # pre-exponential factors [cm^3/s]

    # A  = np.array([18.687,15.06,4.95,2.14,0.0,0.0,0.0,0.0]) # activation temperature [eV]
    # dH = np.array([15.7,11.56,4.14,-11.56,0.0,0.0,0.0,0.0]) # energy lost per electron due to ionization rxn [eV]
    # dEps = np.array([0.0,15.7,11.56,0.0])

    A  = np.array([18.687,15.06,4.95,0.0,0.0,0.0,0.0]) # activation temperature [eV]
    dH = np.array([15.76,11.56,4.2,0.0,-11.56,-7.56,-11.56]) # energy lost per electron due to ionization rxn [eV]
    dEps = np.array([0.0,15.76,11.56,0.0])

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
    Ck   *= 1e-6 # m^3/s
    # Ck[7] *= 1e-6 # Ck[7] is now in m^6/s
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

    Ck[0:2] = Ck[0:2]*tau*nAr
    #Ck[2:5] = Ck[2:5]*tau*np0
    Ck[2:6] = Ck[2:6]*tau*np0
    Ck[6]  *= tau*nAr
    # Ck[7]  *= tau*nAr*nAr
    A       = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH      = dH/e0
    qStar   = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha   = qe*np0*L*L/(V0*eps0)
    ks      = ks*tau/L
    p0      = p/qe/np0
    kappaB  = 4.878171165833662 # non-dimensional thermal conductivity of background specie
                                # (2/3)*tau/L**2*Kb/np0/kB,
                                # where Kb is the thermal conductivity of background specie

    #params.beta = np.array([[2,2,2,1,1],[1,0,1,0,0],[0,1,0,0,0],[0,0,0,1,1]])
    #params.alfa = np.array([[1,1,1,1,1],[0,0,0,0,0],[0,0,1,1,1],[1,1,0,0,0]])
    #params.beta = np.array([[2,1,2,1],[1,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.int)
    #params.alfa = np.array([[1,1,1,1],[0,0,0,0],[0,0,1,1],[1,1,0,0]], dtype=np.int)
    # params.beta = np.array([[2,1,2,1,1,1,0,0],[1,0,1,0,0,1,0,0],[0,1,0,0,0,0,0,0],[0,0,0,1,0,1,2,1]], dtype=np.int64)
    # params.alfa = np.array([[1,1,1,1,1,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,1,1,2,1,1],[1,1,0,0,0,0,1,2]], dtype=np.int64)
    params.beta = np.array([[2,1,2,0,1,1,0],[1,0,1,0,0,1,0],[0,1,0,1,0,0,0],[0,0,0,0,1,1,2]], dtype=np.int64)
    params.alfa = np.array([[1,1,1,1,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,1,2,1],[1,1,0,0,0,0,1]], dtype=np.int64)

    # 4) Set values in params class
    params.D[0]    = De
    params.D[1]    = Di
    params.D[2]    = Dm

    params.mu[0]   = mue
    params.mu[1]   = mui
    params.mu[2]   = mum

    params.A[:]    = Ck[:]
    params.B[:]    = 0.0
    params.C[:]    = A[:]

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

    # reactionExpressionslist = [f"{params.A[0]} * energy**{params.B[0]} * np.exp(-{params.C[0]} / energy)",
    #                            f"{params.A[1]} * energy**{params.B[1]} * np.exp(-{params.C[1]} / energy)",
    #                            f"{params.A[2]} * energy**{params.B[2]} * np.exp(-{params.C[2]} / energy)",
    #                            f"{params.A[3]} * energy**{params.B[3]} * np.exp(-{params.C[3]} / energy)",
    #                            f"{params.A[4]} * energy**{params.B[4]} * np.exp(-{params.C[4]} / energy)",
    #                            f"{params.A[5]} * energy**{params.B[5]} * np.exp(-{params.C[5]} / energy)",
    #                            f"{params.A[6]} * energy**{params.B[6]} * np.exp(-{params.C[6]} / energy)",
    #                            f"{params.A[7]} * energy**{params.B[7]} * np.exp(-{params.C[7]} / energy)"]

    # reactionTExpressionslist = [f"{params.A[0]} * (energy**({params.B[0]}-1)) * np.exp(-{params.C[0]}/energy) * ({params.B[0]} + {params.C[0]}/energy)",
    #                             f"{params.A[1]} * (energy**({params.B[1]}-1)) * np.exp(-{params.C[1]}/energy) * ({params.B[1]} + {params.C[1]}/energy)",
    #                             f"{params.A[2]} * (energy**({params.B[2]}-1)) * np.exp(-{params.C[2]}/energy) * ({params.B[2]} + {params.C[2]}/energy)",
    #                             f"{params.A[3]} * (energy**({params.B[3]}-1)) * np.exp(-{params.C[3]}/energy) * ({params.B[3]} + {params.C[3]}/energy)",
    #                             f"{params.A[4]} * (energy**({params.B[4]}-1)) * np.exp(-{params.C[4]}/energy) * ({params.B[4]} + {params.C[4]}/energy)",
    #                             f"{params.A[5]} * (energy**({params.B[5]}-1)) * np.exp(-{params.C[5]}/energy) * ({params.B[5]} + {params.C[5]}/energy)",
    #                             f"{params.A[6]} * (energy**({params.B[6]}-1)) * np.exp(-{params.C[6]}/energy) * ({params.B[6]} + {params.C[6]}/energy)",
    #                             f"{params.A[7]} * (energy**({params.B[7]}-1)) * np.exp(-{params.C[7]}/energy) * ({params.B[7]} + {params.C[7]}/energy)"]

    reactionExpressionslist = [f"{params.A[0]} * energy**{params.B[0]} * np.exp(-{params.C[0]} / energy)",
                                f"{params.A[1]} * energy**{params.B[1]} * np.exp(-{params.C[1]} / energy)",
                                f"{params.A[2]} * energy**{params.B[2]} * np.exp(-{params.C[2]} / energy)",
                                f"{params.A[3]} * energy**({params.B[3]}-0.5)",
                                f"{params.A[4]} * energy**{params.B[4]} * np.exp(-{params.C[4]} / energy)",
                                f"{params.A[5]} * energy**{params.B[5]} * np.exp(-{params.C[5]} / energy)",
                                f"{params.A[6]} * energy**{params.B[6]} * np.exp(-{params.C[6]} / energy)"]

    reactionTExpressionslist = [f"{params.A[0]} * (energy**({params.B[0]}-1)) * np.exp(-{params.C[0]}/energy) * ({params.B[0]} + {params.C[0]}/energy)",
                                f"{params.A[1]} * (energy**({params.B[1]}-1)) * np.exp(-{params.C[1]}/energy) * ({params.B[1]} + {params.C[1]}/energy)",
                                f"{params.A[2]} * (energy**({params.B[2]}-1)) * np.exp(-{params.C[2]}/energy) * ({params.B[2]} + {params.C[2]}/energy)",
                                f"{params.A[3]} * (energy**({params.B[3]}-0.5-1)) * np.exp(-{params.C[3]}/energy) * ({params.B[3]}-0.5 + {params.C[3]}/energy)",
                                f"{params.A[4]} * (energy**({params.B[4]}-1)) * np.exp(-{params.C[4]}/energy) * ({params.B[4]} + {params.C[4]}/energy)",
                                f"{params.A[5]} * (energy**({params.B[5]}-1)) * np.exp(-{params.C[5]}/energy) * ({params.B[5]} + {params.C[5]}/energy)",
                                f"{params.A[6]} * (energy**({params.B[6]}-1)) * np.exp(-{params.C[6]}/energy) * ({params.B[6]} + {params.C[6]}/energy)"]

    reactionExpressionTypelist =  np.array([True,True,True,False,False,False,False])

    reactionsList = []
    for i in range(Nr):
        if reactionExpressionTypelist[i]:
            # RateCoeff = np.load('./RateCoeff/RateCoeffLiu'+'.npy').T
            rateCoeffTest = np.genfromtxt("./RateCoeff/RateCoeffTable.csv", dtype=str,
                                          encoding=None, delimiter=",").astype(np.float64)

            Nsample = 72
            N300 = 200

            rateCoeff = np.fromfile('./BOLSIGChemistry/reaction300K_%s.dat' %str(i))
            rateCoeff = np.reshape(rateCoeff,[Nsample, N300]).T[:,0]

            Te = np.fromfile('./BOLSIGChemistry/reaction300K.Te.dat')
            Te = np.reshape(Te,[Nsample, N300]).T[:,0]

            # Te[:,0] *= 1.5
            # TeLog = np.log(Te[:, 0])

            # nonZeroIndex = np.nonzero(rateCoeff[:, 0])
            # rateCoeffYLog = np.zeros(rateCoeff.shape[0])
            # rateCoeffYLog[nonZeroIndex[0][0]:] = np.log(rateCoeff[nonZeroIndex[0][0]:, 0])
            # rateCoeffYLog[0:nonZeroIndex[0][0]] = rateCoeffYLog[nonZeroIndex[0][0]]
            # rateCoeffYLog += - np.log(1.0/tau) + np.log(nAr)

            # filtering = np.where(rateCoeffYLog<rateCoeffYLog[nonZeroIndex[0][0]+1])
            # index = filtering[-1][-1]
            # rateCoeffYLog[0:index] = rateCoeffYLog[index]

            # reactionExpressionsLog = CubicSpline(TeLog[:], rateCoeffYLog[:])

            # reactionTExpressionsLog = CubicSpline.derivative(reactionExpressionsLog)
            # reactionTArrayLog = reactionTExpressionsLog(TeLog[:])
            # reactionTArrayLog[0:index] = -1000 # reactionTArrayLog[index]

            # reactionTExpressionsLogFiltered = CubicSpline(TeLog[:],
            #                                               reactionTArrayLog)
            
            # Nondimensionalization of mean energy and transformation to log scale.
            # Te *= 1.5
            TeLog = np.log(Te)

            # Find first non-zero value of the coefficient rate.
            I = np.nonzero(rateCoeff)

            # Compute the slope of the rate coefficient between its first two non-zero values.
            # Finite differences are used.
            dydx = (rateCoeff[I[0][0] + 1] - rateCoeff[I[0][0]]) \
                 / (Te[I[0][0] + 1] - Te[I[0][0]])

            # Arrhenius form: kf = A * exp(-C / Te)
            # C = (dkf/dTe) / kf * Te**2.0
            # A = kf / exp(-C / Te)
            C = Te[I[0][0]]**2.0*dydx / rateCoeff[I[0][0]]
            # A = rateCoeff[I[0][0]] / np.exp(-C/Te[I[0][0]])

            # Compute pre-exponential coefficient, A, in log scale.
            ALog = np.log(rateCoeff[I[0][0]]) + C / Te[I[0][0]]

            # Transform rate coefficient in log scale.
            rateCoeffLog = np.zeros(rateCoeff.shape)
            rateCoeffLog[I[0][0]:] = np.log(rateCoeff[I[0][0]:])
            # For the troublesome values, we use the Arrhenius form.
            rateCoeffLog[0:I[0][0]] = ALog - C / Te[0:I[0][0]]
            # Nondimensionalization in log scale.
            if i < 2:
                rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
            else:
                rateCoeffLog += - np.log(1.0/tau) + np.log(np0)
            # Nondimensionalization of the original rate, used for the plot and comparison.
            rateCoeff *= tau * nAr

            # Interpolation in log scale.
            reactionExpressionsLog = CubicSpline(TeLog, rateCoeffLog)
            # Gradient in log scale
            reactionTExpressionsLog = CubicSpline.derivative(reactionExpressionsLog)

            reaction = Reaction(rxnAlfa = params.alfa[:,[i]], rxnBeta = params.beta[:,[i]],
                                rxnBolsig = reactionExpressionTypelist[i],
                                kf_log = reactionExpressionsLog,
                                kf_T_log = reactionTExpressionsLog)
            reactionsList.append(reaction)

            # rxn   = eval("lambda energy :" + reactionExpressionslist[i])
            # rxn_T = eval("lambda energy :" + reactionTExpressionslist[i])
            # # setting the axes at the centre
            # fig ,ax = plt.subplots(figsize=(9, 6))
            # ax.spines["top"].set_visible(True)
            # ax.spines["right"].set_visible(True)
            # ax.set_yscale('log')
            # ax.set_xscale('log')

            # # plot the function
            # # plt.plot(rateCoeffXFiner, np.exp(reactionExpressions_cubicSplineDerivative_log(rateCoeffXFiner)),
            # #  		 color='salmon', linestyle='--', label='interBolsig')
            # # plt.plot(rateCoeffXFine, np.exp(reactionExpressions_cubicSpline_log(rateCoeffXFine)),
            # #  		 color='lightgreen', linestyle='--', label='interBolsig')
            # # plt.plot(Te[:,0], reactionTExpressionsLogFiltered(TeLog[:]) * np.exp(reactionExpressionsLog(TeLog[:])) / Te[:,0],
            # #  		 color='blue', linestyle='-', label='interBolsig')
            # plt.plot(Te, np.exp(reactionExpressionsLog(TeLog[:])),
            #  		 color='green', linestyle='-', label='Bolsig')
            # plt.plot(Te, rxn(Te),
            #  		 color='salmon', linestyle='--', label='Liu')
            # # plt.plot(Te[:,0], rxn_T(Te[:,0]),
            # #  		 color='red', linestyle='--', label='interBolsig')
            # plt.xlim((0.05,100))
            # plt.ylim((1e-50,300))
            # plt.legend()
            # plt.savefig("./Rates_%s.pdf" %str(i), dpi=300)
            # # plt.xlim((-0.0001,0.0255))
            # plt.show()
        else:
            rxn   = eval("lambda energy :" + reactionExpressionslist[i])
            rxn_T = eval("lambda energy :" + reactionTExpressionslist[i])

            reaction = Reaction(rxnAlfa = params.alfa[:,[i]], rxnBeta = params.beta[:,[i]],
                                rxnBolsig = reactionExpressionTypelist[i],
                                kf = rxn, kf_T = rxn_T)
            reactionsList.append(reaction)

    params.reactionsList = reactionsList
    #params.Nr = 1

    diffList = []
    Te = np.linspace(0, 1000, 10)
    De_interp = params.D[0]*np.ones(10)
    De_spline = CubicSpline(Te, De_interp)
    De_Te_spline = CubicSpline.derivative(De_spline)
    diffusivity = Diffusivity(interpolate = False, D_expression = De_spline, D_T_expression = De_Te_spline)
    diffList.append(diffusivity)

    Ns = 4
    for i in range(1, Ns):
        diffList.append(Diffusivity(interpolate = False))

    params.diffusivityList = diffList

    muList = []
    Te = np.linspace(0, 1000, 10)
    mue_interp = params.mu[0]*np.ones(10)
    mue_spline = CubicSpline(Te, mue_interp)
    mue_Te_spline = CubicSpline.derivative(mue_spline)
    mobility = Mobility(interpolate = False, mu_expression = mue_spline, mu_T_expression = mue_Te_spline)
    muList.append(mobility)

    Ns = 4
    for i in range(1, Ns):
        muList.append(Mobility(interpolate = False))

    params.mobilityList = muList

    # 5) Dump to screen
    params.print()
