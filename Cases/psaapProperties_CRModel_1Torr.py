import numpy as np
from scipy.interpolate import CubicSpline
import scipy.constants as spc

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


# class EnergyDiffusivity(object):
#     def __init__(self, *initial_data, **kwargs):
#         for dictionary in initial_data:
#             for key in dictionary:
#                 setattr(self, key, dictionary[key])
#         for key in kwargs:
#             setattr(self, key, kwargs[key])

# class EnergyMobility(object):
#     def __init__(self, *initial_data, **kwargs):
#         for dictionary in initial_data:
#             for key in dictionary:
#                 setattr(self, key, dictionary[key])
#         for key in kwargs:
#             setattr(self, key, kwargs[key])


def setPsaapProperties_CRModel_1Torr(gam, inputV0, inputVDC, params, Ns):
    """Sets non-dimensional properties corresponding to a Collisional Radiative 
       model for Argon.

    Inputs:
      gam : Secondary electron emission coefficient
      params : chebSolver.modelParams class

    Outputs: None
      params data is overwritten using values from Liu 2014.
    """
    ###################################################################
    # User specified parameters (you may change these if you wish to
    # run a different scenario)
    ###################################################################

    Pressure  = 1.0*spc.torr  # [Pa] 
    GasTemperature = 293.15 # [K]
    nAr = Pressure/GasTemperature/spc.k    # [#/m^3] Number density based on bulk temperature (not necessarily true density in two-temperature gas)

    # densities
    # nAr = 3.22e22     # background number density of Ar [1/m^3] (corresponds to p = 1 Torr)
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
    p  = Pressure*1.5      # [J/m^3] *1.5 to convert it to energy (1 Torr)

    # gas energy at the wall
    # Tg0 = 0.038778    # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot
    Tg0 = 3/2*GasTemperature*spc.k/spc.e # 3/2*300K*kB ~ (p0 - nT[:,0])/ntot

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
    # Ck = np.array([2.0e-13,0.0]) # pre-exponential factors [m^3/s]
    # B  = np.array([0,0]) # Temperature Power
    # A  = np.array([0,0]) # activation temperature [eV]
    # dH = np.array([0.0,-7.541]) # energy lost per electron due to ionization rxn [eV]
    # dEps = np.array([0.0,15.76,11.577,11.725,13.168,0.0]) # E, AR+, AR(m), AR(r), AR(4p), AR



    # BC parameters
    # ks = 1.19e7  # electron recombination rate [cm/s]
    ks = 1.366109824889323e7 # electron recombination rate [cm/s/eV]

    ###################################################################
    # Constants of nature (probably shouldn't change unless you have
    # root privileges on universe)
    ###################################################################
    qe   = spc.e #1.60217663e-19    # unit charge [C]
    eps0 = spc.epsilon_0 #8.86e-12  # permittivity of free space [F/m]
    kB   = spc.k #1.380649e-23      # Boltzmann constant [J/K]
    # kB   = 8.62e-5 # Boltzmann constant [eV/K]
    eV = qe/kB



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

    qStar    = V0/e0 # qe*V0/e0, since e0 in eV, need qe*V0 in eV, which is just V0 in V
    alpha    = qe*np0*L*L/(V0*eps0)
    ks       = ks*tau/L
    p0       = p/qe/np0

    ThermalConductivity = 17.7e-3 # [W/m/K] at 300K
    
    kappaB   = (2/3)*tau/L**2*ThermalConductivity/np0/kB
    
    # kappaB   = 4.878171165833662*1.6129 # non-dimensional thermal conductivity of background specie
    #                             # (2/3)*tau/L**2*Kb/np0/kB,
    #                             # where Kb is the thermal conductivity of background specie


    # 4) Set values in params class
    params.D[0]    = De
    params.D[1]    = Di
    params.D[2:]   = Dm
    params.D[-1]   = 5.0/3.0*De # Electron Energy
    
    params.mu[0]   = mue
    params.mu[1]   = mui
    params.mu[2:]  = mum
    params.mu[-1]  = 5.0/3.0*mue # Electron Energy


    # Non-dimensionalization parameters
    params.np0     = np0           # "nominal" electron density [1/m^3]
    params.nAr     = nAr
    params.nAronp0 = nAr / np0
    params.tau     = tau

    params.tauOvernp0 = tau/np0
    params.tauOvernAr = tau/nAr

    
    # params.dH[:]   = dH[:]
    # params.dEps[:] = dEps[:]
    params.qStar   = qStar
    params.alpha   = alpha
    params.ks      = ks
    params.gam     = gam
    params.kappaB  = kappaB
    params.p0      = p0
    params.Tg0     = Tg0
    params.EC      = 2.0 * me / mAr \
        * np.sqrt(16.0 * (me + mAr) * e0 * c**2
                  / (3.0 * np.pi * me * mAr)) * se * nAr * tau
    # params.EC = 2.0 * me / mAr * 3.8e9 * tau
    
    params.verticalShift = verticalShift / V0


    # What about params.ksion?
    # params.ksion = 1/4 * np.sqrt(8*kB*Tg0/pi/mi) # I need to nondimensionalize this


    
    # Parameters needed for the CR model
    params.Pressure = Pressure
    params.GasTemperature = GasTemperature

    # Parameters needed to compute the current with dimensions
    params.V0Ltau  = V0 / (L * tau)
    params.V0L     = V0 / L
    params.LLV0tau = (L*L) / (V0*tau)
    params.tauL    = L / tau
    # params.np0     = np0           # "nominal" electron density [1/m^3]
    params.qe      = qe            # unit charge [C]
    params.eps0    = eps0          # unit charge [C]
    params.eArea   = electrodeArea # electrode area [m^2]


    # Electron Transport Data 
    diffList = []
    muList = []

    transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_2.h5", 'r')

    # Electron Diffusion Coef.  
    diffusivityData = transport["diffusivity"] 
    NDe_v_Te = diffusivityData[:,1]
    Te_trans = diffusivityData[:,0]
    Te_trans /= eV
    threshold_Te0 = 2.0    
    indices_Te0 = np.searchsorted(Te_trans, threshold_Te0)
    NDe_v_Te[0:indices_Te0] = NDe_v_Te[indices_Te0]
    De_interp = (NDe_v_Te[:]/nAr)*tau/(L*L)
    De_spline = CubicSpline(Te_trans, De_interp)
    De_Te_spline = CubicSpline.derivative(De_spline)
    diffusivity = Diffusivity(interpolate = False, D_expression = De_spline, D_T_expression = De_Te_spline)
    diffList.append(diffusivity)

    # fig,ax = plt.subplots()
    # ax.set_title('Diffusion Coef.')
    # ax.set_xlabel('Te [eV]')
    # # ax.set_ylabel('D [m2/s]')
    # ax.set_ylabel(r"$D_e \, $ [$ \, m^{2}/s$]")
    # # ax.loglog(Te_trans, NDe_v_Te[:]/nAr, marker = 'o', label = 'Nominal')
    # ax.plot(Te_trans, De_interp, marker = 'o', label = 'Nominal')
    # DEf = diffList[-1].D_T_expression(Te_trans)
    # ax.plot(Te_trans, DEf, marker = '*', label = 'spline')
    # # plt.axhline(y=params.D[0], color='k', linestyle='--')
    # ax.legend()
    # # plt.savefig('EinsteinRelation.png')

    for i in range(1, Ns):
        diffList.append(Diffusivity(interpolate = False))

    # Electron Energy Diffusion Coef.  
    energydiffusivityData = transport["energy_diffusivity"] 
    NDee_v_Te = energydiffusivityData[:,1]
    NDee_v_Te[0:indices_Te0] = NDee_v_Te[indices_Te0]
    Dee_interp = (NDee_v_Te[:]/nAr)*tau/(L*L)
    Dee_spline = CubicSpline(Te_trans, Dee_interp)
    Dee_Te_spline = CubicSpline.derivative(Dee_spline)
    energydiffusivity = Diffusivity(interpolate = False, D_expression = Dee_spline, D_T_expression = Dee_Te_spline)
    diffList.append(energydiffusivity)

    # fig,ax = plt.subplots()
    # ax.set_title('Energy Diffusion Coef.')
    # ax.set_xlabel('Te [eV]')
    # # ax.set_ylabel('D [m2/s]')
    # ax.set_ylabel(r"$D_{ee} \, $ [$ \, m^{2}/s$]")
    # # ax.loglog(Te_trans, NDe_v_Te[:]/nAr, marker = 'o', label = 'Nominal')
    # ax.plot(Te_trans, Dee_interp, marker = 'o', label = 'Nominal')
    # DEEf = diffList[-1].D_T_expression(Te_trans)
    # ax.plot(Te_trans, DEEf, marker = '*', label = 'spline')
    # # plt.axhline(y=params.D[0], color='k', linestyle='--')
    # ax.legend()



    #  Electron Mobility 
    muList = []
    mobilityData = transport["mobility"] 
    Nmue_v_Te = mobilityData[:,1]
    # Nmue_v_Te[indices_Te0] = Nmue_v_Te[indices_Te0[-1]+1]
    Nmue_v_Te[0:indices_Te0] = Nmue_v_Te[indices_Te0]
    mue_interp = (Nmue_v_Te[:]/nAr)*V0*tau/(L*L)
    mue_spline = CubicSpline(Te_trans, mue_interp)
    mue_Te_spline = CubicSpline.derivative(mue_spline)
    mobility = Mobility(interpolate = True, mu_expression = mue_spline, mu_T_expression = mue_Te_spline)
    muList.append(mobility)

    # fig, ax = plt.subplots()
    # ax.set_title('Mobility Coef.')
    # ax.set_xlabel('Te [eV]')
    # ax.set_ylabel(r"$\mu_e \, $ [$ \, m^{2}/V/s$]")
    # # ax.loglog(Te_trans, Nmue_v_Te[:]/nAr, marker = 'o', label = 'Nominal')
    # ax.plot(Te_trans, mue_interp, marker = 'o', label = 'Nominal')
    # mu_e = muList[-1].mu_T_expression(Te_trans) 
    # ax.plot(Te_trans, mu_e, marker = '*', label = 'Nominal')
    # # plt.axhline(y=params.mu[0], color='k', linestyle='--')
    # ax.legend()


    for i in range(1, Ns):
        muList.append(Mobility(interpolate = False))

    #  Electron Energy Mobility 
    energymobilityData = transport["energy_mobility"] 
    Nmuee_v_Te = energymobilityData[:,1]
    Nmuee_v_Te[0:indices_Te0] = Nmuee_v_Te[indices_Te0]
    muee_interp = (Nmuee_v_Te[:]/nAr)*V0*tau/(L*L)
    muee_spline = CubicSpline(Te_trans, muee_interp)
    muee_Te_spline = CubicSpline.derivative(muee_spline)
    energymobility = Mobility(interpolate = False, mu_expression = muee_spline, mu_T_expression = muee_Te_spline)
    muList.append(energymobility)

    # fig, ax = plt.subplots()
    # ax.set_title('Energy Mobility Coef.')
    # ax.set_xlabel('Te [eV]')
    # ax.set_ylabel(r"$\mu_e \, $ [$ \, m^{2}/V/s$]")
    # # ax.loglog(Te_trans, Nmue_v_Te[:]/nAr, marker = 'o', label = 'Nominal')
    # ax.plot(Te_trans, muee_interp, marker = 'o', label = 'Nominal')
    # mu_ee = muList[-1].mu_T_expression(Te_trans) 
    # ax.plot(Te_trans, mu_ee, marker = '*', label = 'Nominal')
    # # plt.axhline(y=params.mu[0], color='k', linestyle='--')
    # ax.legend()





    params.diffusivityList = diffList
    params.mobilityList = muList
      
    # 5) Dump to screen
    params.print()


    ### Indexing
    # GlowDischarge Indexing 
    # i = 0       -> electrons 
    # i = 1       -> ions        
    # i = 2:Ns-1  -> excited levels
    # i = Ns - 1  -> ground state
    # i = Ns      -> electron energy

    # CR Indexing 
    # i = 0       -> ground state
    # i = 1:Ns-2  -> excited levels
    # i = Ns - 2  -> electrons
    # i = Ns - 1  -> ions
    # i = Ns      -> electron energy


