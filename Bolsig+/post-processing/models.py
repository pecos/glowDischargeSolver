import numpy as np

# Rydberg energy
R0 = 13.605693122994  # eV
# Bohr radius
a0 = 5.29177e-11     # m
# electron charge
qe = 1.60217662e-19  # C
# Boltzmann constant
kB = 1.380649e-23    # m2 kg s-2 K-1
# Avogadro number
NA = 6.0221408e23    # mol-1
# electron mass
me = 9.10938356e-31  # kg
# speed of light
c0 = 2.99792458e8    # m/s
# mc2 in eV
mc2 = me * c0 * c0 / qe
# hbar = h / 2pi
hbar = 6.62607004e-34 / 2.0 / np.pi   # m2 kg / s
# static polarizability
alpha = 11.08     # a0^3
# E (eV) from k^2 (a0^(-2))
Efromk2 = hbar * hbar / 2.0 / me / qe / a0 / a0

# Excitation energy levels in eV, from NIST
E_ext = [11.54835442, 11.62359272, 11.72316039, 11.82807116]
E_ext_2p = [1.29100E+01, 1.30800E+01, 1.31000E+01, 1.31500E+01, 1.31700E+01,
            1.32700E+01, 1.32800E+01, 1.33000E+01, 1.33300E+01, 1.34800E+01]
# Ionization energy levels in eV, from NIST
E_ion = [15.7596119, 27.62967]

# Meta-stable ionization energy
E_step_ion = [E_ion[0] - E_ext[0]]

# All take theta as model parameters, E as electron energy in eV.
def Excite_metastable(n,theta,E):
    # n: excited energy level, starting from 1,2,...
    b, gamma = theta
    return 4. * np.pi * a0 * a0 * R0 / E * b * ( 1.0 - E_ext[n-1]/E ) * ( (2. * E)**(-gamma) )

def Excite_resonance_modified(n,theta,E):
    # n: excited energy level, starting from 1,2,...
    if (len(theta)==3):
        F0, beta, C1 = theta
    elif (len(theta)==2):
        F0, beta = theta
        C1 = 1.0
    elif (len(theta)==1):
        F0 = theta[0]
        beta, C1 = 1.0, 1.0

#     print('params: ',F0, beta, alpha0, C1)
    # from eV to (m/s)^2
    v2 = qe * E * 2.0 / me
    beta2 = v2 / c0 / c0
    # relativistic factor
    rel_factor = np.log(1.0 / (1.0 - beta2) * E / E_ext[n-1] * C1 ) - beta2
    emp_factor = np.exp( - beta / (1.0 + E/E_ext[n-1]) )

    return 4. * np.pi * a0 * a0 * R0 * R0 / E * F0 / E_ext[n-1] * rel_factor * emp_factor

def Excite_resonance(n,theta,E):
    # n: excited energy level, starting from 1,2,...
    if (len(theta)==4):
        F0, beta, alpha0, C1 = theta
    elif (len(theta)==3):
        F0, beta, alpha0 = theta
        C1 = 1.0
    elif (len(theta)==2):
        F0, beta = theta
        C1, alpha0 = 1.0, 1.0
    elif (len(theta)==1):
        F0 = theta[0]
        beta, alpha0, C1 = 1.0, 1.0, 1.0

#     print('params: ',F0, beta, alpha0, C1)
    # from eV to (m/s)^2
    v2 = qe * E * 2.0 / me
    beta2 = v2 / c0 / c0
    # relativistic factor
    rel_factor = np.log(1.0 / (1.0 - beta2) * E / E_ext[n-1] * C1 ) - beta2
#     rel_factor = np.log(beta2 / (1.0 - beta2) * mc2 / 2.0 / E_ext[n-1] ) - beta2
#     largeE_factor = 0.5 * (1.0 + C1) + 0.5 * (C1 - 1.0) * np.tanh( beta*(E - alpha0 - E_ext[n-1]) )
    return 4. * np.pi * a0 * a0 * R0 * R0 / E * F0 / E_ext[n-1] * rel_factor * ( 1.0 - (E_ext[n-1]/E)**alpha0 ) ** beta
#     return 4. * np.pi * a0 * a0 * R0 * R0 / E * F0 / E_ext[n-1] * rel_factor * largeE_factor

def total_Ion_BED(theta, E, deltaE=E_ion[0]):
    a, b, c = theta
    t = E / deltaE
    return 4. * np.pi * a0 * a0 / t * ( a * np.log(t) + b * (1. - 1. / t) + c * np.log(t) / (t + 1.) )

def elastic_MERT(theta,E,N=10):
    NE = len(E)
    k = np.sqrt( E / Efromk2 ) # wavenumber in a0^(-1)
    crs = np.zeros((NE,))

    A, D, F, E1 = theta
    eta0 = - A * ( 1. + 4. / 3. * alpha * k * k * np.log(k) ) - np.pi / 3. * alpha * k + D * k**2 + F * k**3
    eta0 = np.arctan(eta0 * k)

    eta1 = np.pi / 15. * alpha * k * ( 1. - np.sqrt(E/E1) )
    eta1 = np.arctan(eta1 * k)

    crs += np.sin(eta0 - eta1)**2

    for L in range(1,N):
        eta0 = np.copy(eta1)
        L1 = L+1
        eta1 = np.pi * alpha * k / (2.*L1 + 3.) / (2.*L1 + 1.) / (2.*L1 - 1.)
        eta1 = np.arctan(eta1 * k)

        crs += (L + 1.) * np.sin(eta0 - eta1)**2

    return crs * 4. * np.pi / k / k * a0 * a0

def elastic_modified_MERT(theta,E,N=10):
    NE = len(E)
    k = np.sqrt( E / Efromk2 ) # wavenumber in a0^(-1)
    crs = np.zeros((NE,))

    A, D, F, E1, E2, E3 = theta
    eta0 = - A * ( 1. + 4. / 3. * alpha * k * k * np.log(k) ) - np.pi / 3. * alpha * k + D * k**2 + F * k**3
    eta0 = np.arctan(eta0 * k)

    # one more parameter added on eta1.
    eta1 = np.pi / 15. * alpha * k * ( 1. - np.sqrt(E)/E1 + E/E2 )
    eta1 = np.arctan(eta1 * k)

    crs += np.sin(eta0 - eta1)**2

    for L in range(1,N):
        eta0 = np.copy(eta1)
        L1 = L+1
        eta1 = np.pi * alpha * k / (2.*L1 + 3.) / (2.*L1 + 1.) / (2.*L1 - 1.)
        if (L==1):
            eta1 *= (1.0 + E/E3)
        eta1 = np.arctan(eta1 * k)

        crs += (L + 1.) * np.sin(eta0 - eta1)**2

    return crs * 4. * np.pi / k / k * a0 * a0

def elastic_shifted_MERT(theta,E_input,N=10):
    NE = len(E_input)
    E = np.copy(E_input)

    A, D, F, E1, t1, t2, t3 = theta

    E *= 0.5*(1.+t1) - 0.5 * (1. - t1) * np.tanh( (E - t2) / t3 )

    k = np.sqrt( E / Efromk2 ) # wavenumber in a0^(-1)
    crs = np.zeros((NE,))

#     A, D, F, E1 = theta
    eta0 = - A * ( 1. + 4. / 3. * alpha * k * k * np.log(k) ) - np.pi / 3. * alpha * k + D * k**2 + F * k**3
    eta0 = np.arctan(eta0 * k)

    eta1 = np.pi / 15. * alpha * k * ( 1. - np.sqrt(E/E1) )
    eta1 = np.arctan(eta1 * k)

    crs += np.sin(eta0 - eta1)**2

    for L in range(1,N):
        eta0 = np.copy(eta1)
        L1 = L+1
        eta1 = np.pi * alpha * k / (2.*L1 + 3.) / (2.*L1 + 1.) / (2.*L1 - 1.)
        eta1 = np.arctan(eta1 * k)

        crs += (L + 1.) * np.sin(eta0 - eta1)**2

    return crs * 4. * np.pi / k / k * a0 * a0
