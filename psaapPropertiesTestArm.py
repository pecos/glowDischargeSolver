import numpy as np

def setPsaapPropertiesTestArm(gam, params):
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

    # nominal electron energy
    e0 = 1.0  # [eV]

    # characteristics of driving voltage
    V0  = 100.0 #1000.0      # amplitude of driving voltage [V]
    tau = (1./13.56e6) # period of driving voltage [s]
    L   = 2.54*0.005 #1*0.005  # half-gap-width [m] (gap width is 1cm)

    # transport parameters
    nmue = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
    nmui = 4.65e19   # argon number density times ion mobility [1/(V*cm*s)]
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
    nDm  *= 100.
    nmue *= 100. # 1/(V*m*s)
    nmui *= 100. # 1/(V*m*s)
    Ck   *= 1e-6 # m^3/s
    Ck[7] *= 1e-6 # Ck[7] is now in m^6/s
    ks   *= 0.01 # m/s

    # 2) Compute "raw" transport parameters
    De  = nDe/nAr
    Di  = nDi/nAr
    Dm  = nDm/nAr

    mue = nmue/nAr
    mui = nmui/nAr

    # 3) Compute non-dimensional properties required by solver
    De    = De*tau/(L*L)
    Di    = Di*tau/(L*L)
    Dm    = Dm*tau/(L*L)

    mue   = mue*V0*tau/(L*L)
    mui   = mui*V0*tau/(L*L)

    Ck[0:2]    = Ck[0:2]*tau*nAr
    #Ck[2:5]    = Ck[2:5]*tau*np0
    Ck[2:6]    = Ck[2:6]*tau*np0
    Ck[6] *= tau*nAr
    Ck[7] *= tau*nAr*nAr
    A     = A*1.5/e0  # 1.5 to convert from temperature to energy
    dH    = dH/e0
    qStar = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha = qe*np0*L*L/(V0*eps0)
    ks    = ks*tau/L

    #params.beta = np.array([[2,2,2,1,1],[1,0,1,0,0],[0,1,0,0,0],[0,0,0,1,1]])
    #params.alfa = np.array([[1,1,1,1,1],[0,0,0,0,0],[0,0,1,1,1],[1,1,0,0,0]])
    #params.beta = np.array([[2,1,2,1],[1,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.int)
    #params.alfa = np.array([[1,1,1,1],[0,0,0,0],[0,0,1,1],[1,1,0,0]], dtype=np.int)
    params.beta = np.array([[2,1,2,1,1,1,0,0],[1,0,1,0,0,1,0,0],[0,1,0,0,0,0,0,0],[0,0,0,1,0,1,2,1]], dtype=np.int)
    params.alfa = np.array([[1,1,1,1,1,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,1,1,2,1,1],[1,1,0,0,0,0,1,2]], dtype=np.int)

    # 4) Set values in params class
    params.D[0]  = De
    params.D[1]  = Di
    params.D[2]  = Dm

    params.mu[0] = mue
    params.mu[1] = mui
    params.mu[2] = 0

    params.A[:]  = Ck[:]
    params.B[:]  = 0.0
    params.C[:]  = A[:]

    params.dH[:] = dH[:]
    params.dEps[:] = dEps[:]
    params.qStar = qStar
    params.alpha = alpha
    params.ks    = ks
    params.gam   = gam

    #params.Nr = 1

    # 5) Dump to screen
    params.print()
