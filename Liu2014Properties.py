def setLiu2014Properties(gam, params):
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
    nAr = 3.22e22     # background number density of Ar [1/m^3]
    np0 = 8e16        # "nominal" electron density [1/m^3]

    # nominal electron energy
    e0 = 1.0  # [eV]

    # pressure
    p  = 133.3224*1.5 # [J/m^3] *1.5 to convert it to energy

    # characteristics of driving voltage
    V0  = 100.0       # amplitude of driving voltage [V]
    tau = (1./13.6e6) # period of driving voltage [s]
    L   = 2.54*0.005  # half-gap-width [m] (gap width is 1in)

    # transport parameters
    nmue   = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
    nmui   = 4.65e19   # argon number density times ion mobility [1/(V*cm*s)]
    nDe    = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi    = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]
    kappaB = 4.42      # thermal conductivity of background specie
                       # !!!Don't understand this value.

    # reaction parameters (NB: k_i = Ck*exp(-A/Te))
    Ck = 1.235e-7    # ionization rate pre-exponential factor [cm^3/s]
    A  = 18.687      # activation temperature [eV]
    dH = 15.7        # energy lost per electron due to ionization rxn [eV]

    # BC parameters
    ks = 1.19e7  # electron recombination rate [cm/s]

    ###################################################################
    # Constants of nature (probably shouldn't change unless you have
    # root privileges on universe)
    ###################################################################
    qe   = 1.6e-19   # unit charge [C]
    eps0 = 8.86e-12  # permittivity of free space [F/m]
    kB   = 1.38e-23  # Boltzmann constant [J/K]


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
    qStar  = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha  = qe*np0*L*L/(V0*eps0)
    ks     = ks*tau/L
    p0     = p/qe/np0

    # 4) Set values in params class
    params.D[0]   = De
    params.D[1]   = Di
    params.mu[0]  = mue
    params.mu[1]  = mui
    params.A[0]   = Ck
    params.B[0]   = 0.0
    params.C[0]   = A
    params.dH[0]  = dH
    params.qStar  = qStar
    params.alpha  = alpha
    params.ks     = ks
    params.gam    = gam
    params.kappaB = kappaB
    params.nAronp0 = nAr / np0
    params.p0      = p0

    # 5) Dump to screen
    params.print()
