import numpy as np
from scipy.interpolate import CubicSpline

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

def setPsaapPropertiesTestArmInterpTrans(gam, inputV0, inputVDC, params, Nr):
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
    Ck = np.array([1.235e-7,3.712e-8,2.05e-7,1.818e-9,2e-7,6.2e-10,3.0e-15,1.1e-31]) # pre-exponential factors [cm^3/s]

    # slow excitation rate
    #Ck = np.array([1.235e-7,0.5*3.712e-8,2.05e-7,1.818e-9,2e-7,6.2e-10,3.0e-15,1.1e-31]) # pre-exponential factors [cm^3/s]

    ## fast excitation rate
    #Ck = np.array([1.235e-7,2.0*3.712e-8,2.05e-7,1.818e-9,2e-7,6.2e-10,3.0e-15,1.1e-31]) # pre-exponential factors [cm^3/s]

    A  = np.array([18.687,15.06,4.95,2.14,0.0,0.0,0.0,0.0]) # activation temperature [eV]
    dH = np.array([15.7,11.56,4.14,-11.56,0.0,0.0,0.0,0.0]) # energy lost per electron due to ionization rxn [eV]
    dEps = np.array([0.0,15.7,11.56,0.0])

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
    Ck[7] *= 1e-6 # Ck[7] is now in m^6/s
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
    Ck[7]  *= tau*nAr*nAr
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
    params.beta = np.array([[2,1,2,1,1,1,0,0],[1,0,1,0,0,1,0,0],[0,1,0,0,0,0,0,0],[0,0,0,1,0,1,2,1]], dtype=np.int64)
    params.alfa = np.array([[1,1,1,1,1,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,1,1,2,1,1],[1,1,0,0,0,0,1,2]], dtype=np.int64)

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

    reactionExpressionslist = [f"{params.A[0]} * energy**{params.B[0]} * np.exp(-{params.C[0]} / energy)",
                               f"{params.A[1]} * energy**{params.B[1]} * np.exp(-{params.C[1]} / energy)",
                               f"{params.A[2]} * energy**{params.B[2]} * np.exp(-{params.C[2]} / energy)",
                               f"{params.A[3]} * energy**{params.B[3]} * np.exp(-{params.C[3]} / energy)",
                               f"{params.A[4]} * energy**{params.B[4]} * np.exp(-{params.C[4]} / energy)",
                               f"{params.A[5]} * energy**{params.B[5]} * np.exp(-{params.C[5]} / energy)",
                               f"{params.A[6]} * energy**{params.B[6]} * np.exp(-{params.C[6]} / energy)",
                               f"{params.A[7]} * energy**{params.B[7]} * np.exp(-{params.C[7]} / energy)"]

    reactionTExpressionslist = [f"{params.A[0]} * (energy**({params.B[0]}-1)) * np.exp(-{params.C[0]}/energy) * ({params.B[0]} + {params.C[0]}/energy)",
                                f"{params.A[1]} * (energy**({params.B[1]}-1)) * np.exp(-{params.C[1]}/energy) * ({params.B[1]} + {params.C[1]}/energy)",
                                f"{params.A[2]} * (energy**({params.B[2]}-1)) * np.exp(-{params.C[2]}/energy) * ({params.B[2]} + {params.C[2]}/energy)",
                                f"{params.A[3]} * (energy**({params.B[3]}-1)) * np.exp(-{params.C[3]}/energy) * ({params.B[3]} + {params.C[3]}/energy)",
                                f"{params.A[4]} * (energy**({params.B[4]}-1)) * np.exp(-{params.C[4]}/energy) * ({params.B[4]} + {params.C[4]}/energy)",
                                f"{params.A[5]} * (energy**({params.B[5]}-1)) * np.exp(-{params.C[5]}/energy) * ({params.B[5]} + {params.C[5]}/energy)",
                                f"{params.A[6]} * (energy**({params.B[6]}-1)) * np.exp(-{params.C[6]}/energy) * ({params.B[6]} + {params.C[6]}/energy)",
                                f"{params.A[7]} * (energy**({params.B[7]}-1)) * np.exp(-{params.C[7]}/energy) * ({params.B[7]} + {params.C[7]}/energy)"]

    reactionsList = []
    for i in range(Nr):
        rxn   = eval("lambda energy :" + reactionExpressionslist[i])
        rxn_T = eval("lambda energy :" + reactionTExpressionslist[i])

        reaction = Reaction(rxnAlfa = params.alfa[:,[i]], rxnBeta = params.beta[:,[i]],
                            kf = rxn, kf_T = rxn_T)
        reactionsList.append(reaction)

    params.reactionsList = reactionsList
    #params.Nr = 1

    diffList = []
    # data from Bolsig
    # Te in eV
    # De*N in 1/(m*s)
    # first row is made up to handle when temperature goes out of bounds
#    NDe_v_Te = np.array([[-1, 4.44E+24],  [-0.5, 4.44E+24],  [-0.25, 4.44E+24],  [0., 4.44E+24],
#                          [0.02846756, 4.44E+24],  [0.02975487, 5.09E+24],  [0.03173586, 6.08E+24],  [0.03477738, 7.57E+24],
#                          [0.03937968, 9.74E+24],  [0.04614973, 1.28E+25],  [0.05561446, 1.68E+25],  [0.0681674, 2.18E+25],
#                          [0.0835084, 2.74E+25],   [0.1008504, 3.30E+25],   [0.1187927, 3.80E+25],   [0.1363348, 4.18E+25],
    # NDe_v_Te = np.array([ [-0.05, 5E+25],   [0.0, 5E+25],   [0.05, 5E+25],   [0.1, 5E+25],
    #                       [0.2103718, 4.57E+25],   [0.2244455, 4.47E+25],   [0.2390528, 4.34E+25],   [0.2544605, 4.20E+25],
    #                       [0.2710021, 4.04E+25],   [0.2886776, 3.88E+25],   [0.3078205, 3.72E+25],   [0.3286309, 3.56E+25],
    #                       [0.3513089, 3.39E+25],   [0.3760546, 3.23E+25],   [0.4032682, 3.07E+25],   [0.4331498, 2.92E+25],
    #                       [0.4659662, 2.77E+25],   [0.5021843, 2.62E+25],   [0.5424044, 2.48E+25],   [0.5868266, 2.35E+25],
    #                       [0.6359845, 2.22E+25],   [0.690345, 2.10E+25],    [0.749041, 1.99E+25],    [0.813073, 1.89E+25],
    #                       [0.882441, 1.79E+25],    [0.957145, 1.70E+25],    [1.037185, 1.61E+25],    [1.123895, 1.53E+25],
    NDe_v_Te = np.array([ [-0.05, 1.8e25],       [0.0, 1.8e25],         [0.05, 1.8e25],   [0.1, 1.8e25],
                          [0.2103718, 1.8e25],   [0.2244455, 1.8e25],   [0.2390528, 1.8e25],   [0.2544605, 1.8e25],
                          [0.2710021, 1.8e25],   [0.2886776, 1.8e25],   [0.3078205, 1.8e25],   [0.3286309, 1.8e25],
                          [0.3513089, 1.8e25],   [0.3760546, 1.8e25],   [0.4032682, 1.8e25],   [0.4331498, 1.8e25],
                          [0.4659662, 1.8e25],   [0.5021843, 1.8e25],   [0.5424044, 1.8e25],   [0.5868266, 1.8e25],
                          [0.6359845, 1.8e25],   [0.690345, 1.8e25],    [0.749041, 1.8e25],    [0.813073, 1.8e25],
                          [1.217942, 1.46E+25],    [1.319326, 1.39E+25],    [1.430048, 1.32E+25],    [1.550108, 1.26E+25],
                          [1.681507, 1.20E+25],    [1.823578, 1.15E+25],    [1.977655, 1.09E+25],    [2.143071, 1.05E+25],
                          [2.319826, 9.98E+24],    [2.508587, 9.54E+24],    [2.709354, 9.13E+24],    [2.916124, 8.75E+24],
                          [3.112889, 8.45E+24],    [3.272969, 8.24E+24],    [3.383691, 8.14E+24],    [3.456394, 8.09E+24],
                          [3.505085, 8.07E+24],    [3.544438, 8.05E+24],    [3.579789, 8.03E+24],    [3.616474, 8.01E+24],
                          [3.656494, 7.97E+24],    [3.701183, 7.93E+24],    [3.751875, 7.87E+24],    [3.807903, 7.81E+24],
                          [3.871268, 7.75E+24],    [3.941303, 7.68E+24],    [4.018675, 7.61E+24],    [4.102717, 7.53E+24],
                          [4.193429, 7.5E+24],    [4.290811, 7.5E+24],    [4.39553, 7.5E+24],     [4.507586, 7.5E+24],
                          [5., 7.5E+24],    [6., 7.5E+24],    [7., 7.5E+24],     [8., 7.5E+24],
                          [9., 7.5E+24],    [10., 7.5E+24],    [11., 7.5E+24],     [12., 7.5E+24]])

                          # [4.193429, 7.45E+24],    [4.290811, 7.37E+24],    [4.39553, 7.28E+24],     [4.507586, 7.19E+24],
                          # [4.627646, 7.09E+24],    [4.757711, 6.99E+24],    [4.899115, 6.88E+24],    [5.054526, 6.77E+24],
                          # [5.227946, 6.67E+24],    [5.423377, 6.58E+24],    [5.646822, 6.49E+24],    [5.905618, 6.41E+24],
                          # [6.20977, 6.36E+24],     [6.571284, 6.33E+24],    [7.01017, 6.33E+24],     [7.54377, 6.38E+24],
                          # [8.19743, 6.48E+24],     [9.01117, 6.67E+24],     [10.02501, 6.96E+24],    [11.30565, 7.37E+24],
                          # [12.89978, 7.94E+24],    [14.91412, 8.68E+24],    [17.41537, 9.63E+24],    [20.57028, 1.08E+25],
                          # [24.51225, 1.23E+25],    [29.45472, 1.41E+25],    [35.6845, 1.64E+25],     [43.58845, 1.93E+25],
                          # [53.86692, 2.30E+25],    [67.367, 2.80E+25],      [85.4427, 3.48E+25],     [100.0, 3.48E+25]])


    Te = NDe_v_Te[:,0]
    print("Te_min = {0:.6e}".format(NDe_v_Te[0,0]))
    print("Te_max = {0:.6e}".format(NDe_v_Te[-1,0]))
    De_interp = (NDe_v_Te[:,1]/nAr)*tau/(L*L)
    #De_interp = 0.5 * (NDe_v_Te[:,1]/nAr)*tau/(L*L)

    #Te = np.linspace(0, 1000, 10)
    #De_interp = params.D[0]*np.ones(10)
    #De_interp = (1.61e25/nAr)*tau/(L*L)*np.ones(10)

    De_spline = CubicSpline(Te, De_interp)
    De_Te_spline = CubicSpline.derivative(De_spline)
    diffusivity = Diffusivity(interpolate = True, D_expression = De_spline, D_T_expression = De_Te_spline)
    #diffusivity = Diffusivity(interpolate = False, D_expression = De_spline, D_T_expression = De_Te_spline)
    diffList.append(diffusivity)

    Tplt = np.linspace(-0.5, 5, 1025)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(Tplt, De_spline(Tplt), 'b-')
    plt.plot(Tplt, De * np.ones(Tplt.shape), 'k--')
    plt.plot(Te, De_interp, 'rx')
    #plt.plot(Tplt, De_Te_spline(Tplt), 'g--')
    plt.xlim(-1, 5)
    plt.show()

    Ns = 4
    for i in range(1, Ns):
        diffList.append(Diffusivity(interpolate = False))

    params.diffusivityList = diffList

    muList = []

    Nmue_v_Te = np.array([[0.02846756, 1.289e+26], [0.02975487, 1.33e+26],  [0.03173586, 1.383e+26], [0.03477738,1.448e+26],
                          [0.03937968, 1.523e+26], [0.04614973, 1.604e+26], [0.05561446, 1.679e+26],  [0.0681674,1.735e+26],
                          [0.0835084, 1.749e+26],  [0.1008504, 1.718e+26],  [0.1187927, 1.639e+26],  [0.1363348,1.522e+26],
                          [0.1528764, 1.386e+26],  [0.1682841, 1.245e+26],  [0.1826913, 1.108e+26],  [0.1965649,9.809e+25],
                          [0.2103718, 8.662e+25],  [0.2244455, 7.641e+25],  [0.2390528, 6.74e+25],   [0.2544605,5.947e+25],
                          [0.2710021, 5.249e+25],  [0.2886776, 4.636e+25],  [0.3078205, 4.097e+25],  [0.3286309,3.623e+25],
                          [0.3513089, 3.206e+25],  [0.3760546, 2.839e+25],  [0.4032682, 2.517e+25],  [0.4331498,2.232e+25],
                          [0.4659662, 1.981e+25],  [0.5021843, 1.76e+25],   [0.5424044, 1.565e+25],  [0.5868266,1.392e+25],
                          [0.6359845, 1.239e+25],  [0.690345, 1.103e+25],  [0.749041,  9.8e+24],    [0.813073,8.699e+24],
                          [0.882441, 7.711e+24],   [0.957145, 6.828e+24],  [1.037185,  6.04e+24],   [1.123895,5.344e+24],
                          [1.217942, 4.729e+24],   [1.319326, 4.186e+24],  [1.430048,  3.707e+24],   [1.550108,3.283e+24],
                          [1.681507, 2.907e+24],   [1.823578, 2.574e+24],  [1.977655,  2.278e+24],   [2.143071,2.015e+24],
                          [2.319826, 1.779e+24],   [2.508587, 1.569e+24],  [2.709354,  1.384e+24],   [2.916124,1.228e+24],
                          [3.112889, 1.115e+24],   [3.272969, 1.055e+24],   [3.383691,  1.038e+24],  [3.456394,1.044e+24],
                          [3.505085, 1.054e+24],   [3.544438, 1.062e+24],   [3.579789,  1.064e+24], [3.616474, 1.059e+24],
                          [3.656494, 1.048e+24],   [3.701183, 1.033e+24],   [3.751875,  1.014e+24], [3.807903, 9.928e+23],
                          [3.871268, 9.697e+23],   [3.941303,  9.459e+23],   [4.018675, 9.22e+23],  [4.102717, 8.991e+23],
                          [4.193429, 8.773e+23],   [4.290811,  8.57e+23],    [4.39553, 8.383e+23], [4.507586, 8.21e+23],
                          [4.627646, 8.051e+23],   [4.757711,  7.901e+23],   [4.899115, 7.757e+23], [5.054526, 7.617e+23],
                          [5.227946, 7.479e+23],   [5.423377,  7.339e+23],   [5.646822, 7.198e+23], [5.905618, 7.056e+23],
                          [6.20977, 6.91e+23],     [6.571284,  6.757e+23],   [7.01017, 6.607e+23], [7.54377, 6.458e+23],
                          [8.19743, 6.303e+23],    [9.01117,   6.155e+23],   [10.02501, 6e+23],     [11.30565, 5.843e+23],
                          [12.89978, 5.677e+23],   [14.91412,  5.51e+23],    [17.41537, 5.326e+23], [20.57028, 5.149e+23],
                          [24.51225, 4.968e+23],   [29.45472,  4.784e+23],   [35.6845, 4.602e+23], [43.58845, 4.421e+23],
                          [53.86692, 4.269e+23],   [67.367,    4.134e+23],   [85.4427, 4.026e+23], [100.0, 4.026e+23]])


    Te = Nmue_v_Te[:,0]
    mue_interp = (Nmue_v_Te[:,1]/nAr)*V0*tau/(L*L)

    #Te = np.linspace(0, 10, 10)
    #mue_interp = params.mu[0]*np.ones(10)
    #mue_interp = params.mu[0]*np.append(np.linspace(2, 1, 5), np.ones(5))

    #mue_interp = (9.22e23/nAr)*V0*tau/(L*L) * np.ones(Te.shape)
    #mue_interp = (4.0e23/nAr)*V0*tau/(L*L) * np.ones(10)
    #mue_interp = (6.04e24/nAr)*V0*tau/(L*L) * np.ones(Te.shape)

    mue_spline = CubicSpline(Te, mue_interp)
    mue_Te_spline = CubicSpline.derivative(mue_spline)
    mobility = Mobility(interpolate = True, mu_expression = mue_spline, mu_T_expression = mue_Te_spline)
    #mobility = Mobility(interpolate = False, mu_expression = mue_spline, mu_T_expression = mue_Te_spline)
    muList.append(mobility)

    Tplt = np.linspace(0.5, 5, 129)

    #plt.figure()
    #plt.plot(Tplt, mue_spline(Tplt), 'b-')
    #plt.plot(Tplt, mue * np.ones(Tplt.shape), 'k--')
    #plt.show()


    Ns = 4
    for i in range(1, Ns):
        muList.append(Mobility(interpolate = False))

    params.mobilityList = muList

    # 5) Dump to screen
    params.print()
