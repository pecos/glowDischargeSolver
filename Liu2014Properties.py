import numpy as np


class Reaction(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

def setLiu2014Properties(gam, inputV0, inputVDC, params, Nr):
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
    nAr = 3.22e22     # background number density of Ar [1/m^3] (corresponds to p=1Torr)
    np0 = 8e16        # "nominal" electron density [1/m^3]

    # masses
    # me = 9.10938356e-31  # mass of an electron [kg]
    # me = 5.489e-4        # mass of an electron [u]
    me = 0.511e6           # mass of an electron [eV/c2]
    # mAr = 39.948         # mass of an argon atom [u]
    # mAr = 39.948 * 1.66054e-27 # mass of an argon atom [kg]
    mAr = 37.2158e9        # mass of an electron [eV/c2]
    # u = 931.4941e6       # eV/c2
    c = 299792458          # speed of light [m^2/s]
    se = 40                # momentum cross section [(ångström) Å^2 = 1.0e-20 m^2]

    # nominal electron energy
    e0 = 1.0  # [eV]

    # pressure
    p = 133.3224 * 1.5  # [J/m^3] *1.5 to convert it to energy

    # gas energy at the wall
    Tg0 = 0.038778    # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot

    # characteristics of driving voltage
    V0            = inputV0       # amplitude of driving voltage [V]
    verticalShift = inputVDC      # DC voltage (vertical shift in driving voltage)
    tau           = (1./13.6e6)   # period of driving voltage [s]
    L             = 2.54*0.005    # half-gap-width [m] (gap width is 1in)
    electrodeArea = np.pi*0.05**2 # electrode area [m^2] (electrode diameter = 0.1 m)

    # transport parameters
    nmue   = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
    nmui   = 4.65e19   # argon number density times ion mobility [1/(V*cm*s)]
    nDe    = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi    = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]

    # reaction parameters (NB: k_i = Ck*exp(-A/Te))
    Ck = 1.235e-7    # ionization rate pre-exponential factor [cm^3/s]
    A  = 18.687      # activation temperature [eV]
    dH = 15.7        # energy lost per electron due to ionization rxn [eV]

    # BC parameters
    # ks = 1.19e7            # electron recombination rate [cm/s]
    ks = 1.366109824889323e7 # electron recombination rate [cm/s/eV]

    ###################################################################
    # Constants of nature (probably shouldn't change unless you have
    # root privileges on universe)
    ###################################################################
    qe   = 1.6e-19   # unit charge [C]
    eps0 = 8.86e-12  # permittivity of free space [F/m]
    kB   = 1.38e-23  # Boltzmann constant [J/K]
    # kB = 8.62e−5 # Boltzmann constant [eV/K]


    ###################################################################
    # Calculate non-dimensional parameters
    ###################################################################

    # 1) Convert input units to base SI (except eV)
    nDe  *= 100. # 1/(m*s)
    nDi  *= 100. # 1/(m*s)
    nmue *= 100. # 1/(V*m*s)
    nmui *= 100. # 1/(V*m*s)
    Ck   *= 1e-6 # m^3/s
    ks   *= 0.01 # m/s
    se   *= 1.0e-20  # m^2

    # 2) Compute "raw" transport parameters
    De  = nDe/nAr
    Di  = nDi/nAr
    mue = nmue/nAr
    mui = nmui/nAr

    # 3) Compute non-dimensional properties required by solver
    De     = De*tau/(L*L)
    Di     = Di*tau/(L*L)
    mue    = mue*V0*tau/(L*L)
    mui    = mui*V0*tau/(L*L)
    Ck     = Ck*tau*nAr
    A      = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH     = dH/e0
    dEps   = np.array([0.0,15.7,0.0])
    qStar  = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha  = qe*np0*L*L/(V0*eps0)
    ks     = ks*tau/L
    p0     = p/qe/np0
    kappaB = 4.878171165833662 #4.42 # non-dimensional thermal conductivity of background specie
                                     # (2/3)*tau/L**2*Kb/np0/kB,
                                     # where Kb is the thermal conductivity of background specie

    params.beta = np.array([[2],[1],[0]], dtype=np.int64)
    params.alfa = np.array([[1],[0],[1]], dtype=np.int64)

    # 4) Set values in params class
    params.D[0]    = De
    params.D[1]    = Di
    params.mu[0]   = mue
    params.mu[1]   = mui
    params.A[0]    = Ck
    params.B[0]    = 0.0
    params.C[0]    = A
    params.dH[0]   = dH
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

    reactionExpressionslist = [f"{params.A[0]} * energy**{params.B[0]} * np.exp(-{params.C[0]} / energy)"]

    reactionTExpressionslist = [f"{params.A[0]} * (energy**({params.B[0]}-1)) * np.exp(-{params.C[0]}/energy) * ({params.B[0]} + {params.C[0]}/energy)"]

    reactionsList = []
    for i in range(Nr):
        rxn   = eval("lambda energy :" + reactionExpressionslist[i])
        rxn_T = eval("lambda energy :" + reactionTExpressionslist[i])

        reaction = Reaction(rxnAlfa = params.alfa, rxnBeta = params.beta,
                            kf = rxn, kf_T = rxn_T)
        reactionsList.append(reaction)

    params.reactionsList = reactionsList

    # 5) Dump to screen
    params.print()
