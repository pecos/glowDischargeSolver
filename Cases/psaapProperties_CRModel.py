import numpy as np
from scipy.interpolate import CubicSpline
import scipy.constants as spc

import csv
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
import h5py as h5
from scipy.ndimage import uniform_filter1d

import logging

def smooth_clip(Te, threshold=1.5, width=0.2):
    """
    Smoothly clip the function values for Te < threshold.
    
    :param Te: Electron temperature
    :param threshold: The temperature below which the clipping occurs
    :param width: The width of the transition region
    :return: Clipping factor
    """
    
    return 1 / (1 + np.exp(-(Te - threshold) / width))


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


def setPsaapProperties_CRModel(gam, inputV0, inputVDC, params, Ns):
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

    Mr_Ar = 39.948/1000.0       # [kg/mol]
    M_Ar = Mr_Ar/spc.N_A        # [kg] mass of argon atom (6.63352088e-26 kg)
    M_ArIon = M_Ar - spc.m_e    # [kg] mass of argon ion 

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
    # nmui = 4.65e19   # Transport coefficients from Lymberopoulos & Economou, 1993
    nDe  = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
    nDi  =  nmui * spc.k * GasTemperature / spc.e   # 2.07e18 # argon number density times ion diffusivity [1/(cm*s)]
    nDm  = 2.42e18 #2.42e18 #4.3763e18   # argon number density times AR(m) diffusivity [1/(cm*s)]
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
    # ks = 1/4 * np.sqrt(2/3*spc.e * 8/np.pi/spc.m_e) * 100.0 # electron recombination rate [cm/s/eV] 

    ksion = 1/4 * np.sqrt(8*spc.k*GasTemperature/np.pi/M_ArIon) * 100.0 # ion rate [cm/s] 

    ksa = 1/4 * np.sqrt(8*spc.k*GasTemperature/np.pi/M_Ar) * 100.0 # atom rate [cm/s] 


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
    nDe   *= 100. # 1/(m*s)
    nDi   *= 100. # 1/(m*s)
    nDm   *= 100.
    nDr   *= 100.
    nD4p  *= 100.
    nmue  *= 100. # 1/(V*m*s)
    nmui  *= 100. # 1/(V*m*s)
    ks    *= 0.01 # m/s
    ksion *= 0.01 # m/s
    ksa   *= 0.01 # m/s
    se    *= 1.0e-20  # m^2

    # 2) Compute "raw" transport parameters
    De  = nDe/nAr
    Di  = nDi/nAr
    Dm  = nDm/nAr
    Dr  = nDr/nAr
    D4p = nD4p/nAr





    # m1 = Mr_Ar*1000
    # m2 = Mr_Ar*1000
    # mkl = 1/m1 + 1/m2
    
    # Sigma_klvi_g = 16.1
    # Sigma_klvi_m = 183.2
    

    # Dkl = 1e-4 * 1e-3 * GasTemperature**1.75 / (Pressure/101325.0) * np.sqrt(mkl) / (Sigma_klvi_g**(1/3) + Sigma_klvi_m**(1/3))**2 

    # D0 = 0.156
    # n0 = 1.92
    # P0 = 760 * spc.torr
    # Dargon = 1e-4*D0 * (P0 / Pressure) / (300/273.15)**n0 

    # # R. Byron Bird, Warren E. Stewart, Edwin N. Lightfoot - Transport Phenomena.-Wiley (2001)
    # sigma = 3.432; epsilonOverKappa = 122.4 # Lennard-Jones parameters
    # Tstar = GasTemperature / epsilonOverKappa
    # Omega_d = 1.06036 / Tstar**0.15610 + 0.19300 / np.exp(0.47635 * Tstar) + 1.03587 / np.exp(1.52996 * Tstar) + 1.76474 / np.exp(3.89411 * Tstar)  
    # Dab = 1e-4 * 0.0018583 * GasTemperature**1.5 / (Pressure/101325.0) * np.sqrt(mkl) / sigma**2 / Omega_d

    # # print(Dkl, Dm, Dargon, Dab)
    # print(Dkl*nAr/100, Dm*nAr/100, Dargon*nAr/100, Dab*nAr/100)

    # exit(-1)



    mue = nmue/nAr
    mui = nmui/nAr
    mum = nmum/nAr
    mur = nmur/nAr
    mu4p = nmu4p/nAr




    # nu_e =  qe / spc.m_e / mue
    # nu_i =  qe / M_ArIon / (mui*0.5)
    # print(nu_e/(1/tau), nu_i/(1/tau))
    # exit(-1)



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
    ksion    = ksion*tau/L
    ksa      = ksa*tau/L
    p0       = p/qe/np0

    # non-dimensional parameter for the effective electric field for ions
    vmStar = V0 * tau**2 / L**2 * qe / M_ArIon


    ThermalConductivity = 17.7e-3 # [W/m/K] at 300K at atmospheric pressure. 
                                  # Thermal conductivity of monatomic gases is idependent of pressure. 

    # # R. Byron Bird, Warren E. Stewart, Edwin N. Lightfoot - Transport Phenomena.-Wiley (2001)
    # sigma = 3.432; epsilonOverKappa = 122.4 # Lennard-Jones parameters for Ar
    # Tstar = GasTemperature / epsilonOverKappa
    # Omega_k = 1.16145 / Tstar**0.14874 + 0.52487 / np.exp(0.77320 * Tstar) + 2.16178 / np.exp(2.43787 * Tstar)  
    # ThermalConductivity_2 = 1.9891e-4 * np.sqrt(GasTemperature / (Mr_Ar*1000)) / sigma**2 / Omega_k  # [cal/cm/K/s]
    # ThermalConductivity_2 *= 4.184 * 1e2 # [W/m/K]
    
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
    params.np0         = np0   # "nominal" electron density [1/m^3]
    params.nAr         = nAr
    params.nAronp0     = nAr / np0
    params.tau         = tau
    params.tauOvernp0  = tau/np0
    params.tauOvernAr  = tau/nAr

    
    # params.dH[:]   = dH[:]
    # params.dEps[:] = dEps[:]
    params.qStar   = qStar
    params.alpha   = alpha
    params.ks      = ks
    params.ksion   = ksion
    params.ksa     = ksa
    params.gam     = gam
    params.kappaB  = kappaB
    params.p0      = p0
    params.Tg0     = Tg0
    params.EC      = 2.0 * me / mAr \
        * np.sqrt(16.0 * (me + mAr) * e0 * c**2
                  / (3.0 * np.pi * me * mAr)) * se * nAr * tau
    # params.EC = 2.0 * me / mAr * 3.8e9 * tau
    
    params.verticalShift = verticalShift / V0
    params.vmStar  = vmStar
    
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
    params.eps0    = eps0          # permittivity of free space
    params.eArea   = electrodeArea # electrode area [m^2]





    #-------------------------------------------------------------------------------
    #########   Electron and Ion Transport Data  #########
    diffList = []
    muList = []

    # transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_3.h5", 'r')
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_Biagi_BSR_GlowDischarge.h5", 'r')
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_2.5Torr.h5", 'r')
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_5Torr.h5", 'r')
    # transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_1Torr_2.h5", 'r')
    transport = h5.File("./BOLSIGChemistry_Transport/transport_BSR_1Torr_3.h5", 'r')


    ElectricFieldData = transport["reduced_electric_field"] 
    EN = ElectricFieldData[:,1]
    Te_trans = ElectricFieldData[:,0]
    Te_trans /= eV
    threshold_Te0 = 1.5    
    indices_Te0 = np.searchsorted(Te_trans, threshold_Te0)

    # threshold_Te0_2 = 2.5    
    # indices_Te0_2 = np.searchsorted(Te_trans, threshold_Te0_2)
    # clip_factor = smooth_clip(Te_trans, threshold_Te0)

    # fig, ax = plt.subplots()
    # ax.set_title('Electric field / N (Td)')
    # ax.set_ylabel(r"$ E / N_{Ar} \, $ [$ \, Td$]")
    # ax.set_xlabel('Te [eV]')
    # ax.plot(Te_trans,EN, marker = '.', label = 'bolsig+')
    # # energy_fit = 2.5 * np.sqrt(EN) * (1+(EN/5)**2)**(-0.2) * (1+(EN/1000)**2)**0.7
    # # ax.plot(EN,energy_fit*2.0/3.0, marker = '.', label = 'Empirical fit')
    # ax.legend()
    



    #-------------------------------------------------------------------------------
    ######### Electron Mobility  #########
    mobilityData = transport["mobility"] 
    Nmue_v_Te = mobilityData[:,1]
    Nmue_v_Te[0:indices_Te0] = Nmue_v_Te[indices_Te0]
    mue_interp = (Nmue_v_Te[:]/nAr)*V0*tau/(L*L)
    mue_interp = uniform_filter1d(mue_interp, size=20)
    # mue_interp_2 = (Nmue_v_Te[:]/nAr)*V0*tau/(L*L)*clip_factor + (1 - clip_factor) * params.mu[0]
    mue_spline = CubicSpline(Te_trans, mue_interp)
    mue_Te_spline = CubicSpline.derivative(mue_spline)
    # mue_spline_2 = CubicSpline(Te_trans, mue_interp_2)
    # mue_Te_spline_2 = CubicSpline.derivative(mue_spline_2)
    mobility = Mobility(interpolate = True, mu_expression = mue_spline, mu_T_expression = mue_Te_spline)
    muList.append(mobility)

    # fig, ax = plt.subplots()
    # ax.set_title('Mobility Coef.')
    # ax.set_xlabel('Te [eV]')
    # ax.set_ylabel(r"$\mu_e \, $ [$ \, m^{2}/V/s$]")
    # # ax.loglog(Te_trans, Nmue_v_Te[:]/nAr, marker = 'o', label = 'Nominal')
    # ax.plot(Te_trans, mue_interp, marker = '.', label = 'raw')
    # # ax.plot(Te_trans, mue_interp_2, marker = '.', label = 'smoothed')
    # ax.plot(Te_trans, mue_Te_spline(Te_trans), marker = '*', label = 'raw - grad')
    # # ax.plot(Te_trans, mue_Te_spline_2(Te_trans), marker = '*', label = 'smoothed - derivative')
    # plt.axhline(y=params.mu[0], color='k', linestyle='--')
    # ax.legend()



    #-------------------------------------------------------------------------------
    #########  Ion Mobility. #########
    EN_Td = np.logspace(np.log10(1e-2),np.log10(5000),3000,dtype=np.float64) #  Electric field / N [Td]         

    # Nmui_v_Te = 4 * 1e21 / (1 + (22.1 * 1e29 * EN_Td *1e-21))**0.33    # [1/(V*m*s)]

    Efield = EN_Td *1e-21 * nAr
    DriftVelocity_Ion = 4 * EN_Td / (1 + (0.007 * EN_Td  )**1.5)**0.33 # [m/s]
    Nmui_v_Te = DriftVelocity_Ion * nAr / Efield    
    K0 = Nmui_v_Te/nAr  # [m^2/(V*s)]

    # Determination of mobility and diffusion coefficients of Ar+ and Ar2+ ions in argon gas (Jasmiyanaa et al  2020)
    # EN_Td_exp = np.array([34.49, 39.93, 45.81, 56.44, 60.55, 61.84, 67.85, 73.52, 79.25, 85.74, 92.25, 
    #                       94.95, 103.73, 108.32, 116.58, 117.59, 133.51, 138.43, 152.3, 172.97, 195.45,
    #                       1.00E+03, 2.00E+03, 3.00E+03, 5.00E+03, 1.00E+04, 2.00E+04, 3.00E+04, 5.00E+04, 1.00E+05]) # [Td]
    # K0 = np.array([1.494, 1.486, 1.445, 1.432, 1.423, 1.411, 1.392, 1.376, 1.365, 1.341, 
    #                1.322, 1.337, 1.291, 1.274, 1.246, 1.259, 1.218, 1.196, 1.178, 1.141, 1.097,
    #                6.10E-01, 4.42E-01, 3.69E-01, 2.94E-01, 2.15E-01, 1.59E-01, 1.33E-01, 1.07E-01, 8.03E-02]) # [cm^2/V/s] at STD
    # K0 = K0 * 101325/Pressure * GasTemperature/273.15 * 1e-4 # [m^2/V/s] 
    

    mui_interp = (Nmui_v_Te[:]/nAr)*V0*tau/(L*L)  # + params.mu[1]*1e-2   
    EN_interp = EN_Td / (1e21 * params.V0L/params.nAr) * 2/3 # We multiply with 2/3 here to make it compatible with  mobility_U function.
    mui_spline = CubicSpline(EN_interp, mui_interp)
    mui_EN_spline = CubicSpline.derivative(mui_spline)
    mobility = Mobility(interpolate = True, mu_expression = mui_spline, mu_T_expression = mui_EN_spline)
    muList.append(mobility)

    # fig, ax = plt.subplots()
    # ax.set_title('Ion Mobility Coef.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$\mu_i \, $ [$ \, m^{2}/V/s$]")
    # ax.plot(EN_Td, mui_interp/params.mu[1], marker = 'o', label = 'raw')
    # # ax.plot(EN_Td, mui_EN_spline(EN_interp), marker = '*', label = 'Grad')
    # # plt.axhline(y=params.mu[1], color='k', linestyle='--')
    # ax.legend()
    # # ax.semilogy()
    # # ax.loglog()

    #-------------------------------------------------------------------------------
    #########  Species Mobility #########
    for i in range(2, Ns):
        muList.append(Mobility(interpolate = False))

    #-------------------------------------------------------------------------------
    #########   Electron Energy Mobility #########
    energymobilityData = transport["energy_mobility"] 
    Nmuee_v_Te = energymobilityData[:,1]
    Nmuee_v_Te[0:indices_Te0] = Nmuee_v_Te[indices_Te0]
    muee_interp = (Nmuee_v_Te[:]/nAr)*V0*tau/(L*L)
    muee_interp = uniform_filter1d(muee_interp, size=20)
    muee_spline = CubicSpline(Te_trans, muee_interp)
    muee_Te_spline = CubicSpline.derivative(muee_spline)
    energymobility = Mobility(interpolate = False, mu_expression = muee_spline, mu_T_expression = muee_Te_spline)
    muList.append(energymobility)

    # fig, ax = plt.subplots()
    # ax.set_title('Energy Mobility Coef.')
    # ax.set_xlabel('Te [eV]')
    # ax.set_ylabel(r"$\mu_\epsilon \, $ [$ \, m^{2}/V/s$]")
    # ax.plot(Te_trans, muee_interp, marker = 'o', label = 'raw')
    # ax.plot(Te_trans, muee_Te_spline(Te_trans), marker = '*', label = 'Grad')
    # plt.axhline(y=params.mu[-1], color='k', linestyle='--')
    # ax.legend()




    #-------------------------------------------------------------------------------
    #########   Electron Diffusion Coef.  #########
    diffusivityData = transport["diffusivity"] 
    NDe_v_Te = diffusivityData[:,1]
    NDe_v_Te[0:indices_Te0] = NDe_v_Te[indices_Te0]
    De_interp = (NDe_v_Te[:]/nAr)*tau/(L*L)
    De_interp = uniform_filter1d(De_interp, size=20)
    De_spline = CubicSpline(Te_trans, De_interp)
    De_Te_spline = CubicSpline.derivative(De_spline)
    diffusivity = Diffusivity(interpolate = False, D_expression = De_spline, D_T_expression = De_Te_spline)
    diffList.append(diffusivity)

    # fig,ax = plt.subplots()
    # ax.set_title('Diffusion Coef.')
    # ax.set_xlabel('Te [eV]')
    # ax.set_ylabel(r"$D_e \, $ [$ \, m^{2}/s$]")
    # ax.plot(Te_trans, De_interp, marker = '.', label = 'raw')
    # # ax.plot(Te_trans, De_Te_spline(Te_trans), marker = '*', label = 'raw - grad')
    # plt.axhline(y=params.D[0], color='k', linestyle='--')
    # ax.legend()



    #-------------------------------------------------------------------------------
    #########  Ion Diffusion Coef. #########
    
    # Determination of mobility and diffusion coefficients of Ar+ and Ar2+ ions in argon gas (Jasmiyanaa et al  2020)
    # EN_Td_exp_2 = np.array([34.5, 39.9, 45.8, 49.2, 56.4, 60.6, 61.8, 67.8, 73.5, 79.2, 85.7, 92.3, 
    #                         95 ,103.7, 108.3, 116.6, 117.6, 133.5, 138.4, 152.3, 173, 195.4, 208.7]) # [Td]

    # DLOverK0 = np.array([25.77, 31.9, 31.16, 27.78, 30.42, 29.54, 36.18, 35.34, 34.9, 34.83, 36.27, 
    #                38.75, 39.63, 41.23, 40.1, 48.23, 41.67, 47.29, 54.35, 50.37, 62.7, 62.13, 56.42]) # [mV]
    # DLOverK0 = DLOverK0*1e-3 # [V]
    # DLOverK0 = np.interp(EN_Td,EN_Td_exp_2,DLOverK0)  
    # DL = DLOverK0 * K0

    # Determination of mobility and diffusion coefficients of Ar+ and Ar2+ ions in argon gas (Jasmiyanaa et al  2020)
    fileName = "./BOLSIGChemistry_Transport/ArIonDiffMobilityRatio_inmVvsTd.csv"
    ArIon_Data = pd.read_csv(fileName)
    EN_Td_exp_Diff_fit = ArIon_Data.iloc[:,0].to_numpy('float64')  # [Td]
    DLOverK0_fit = ArIon_Data.iloc[:,1].to_numpy('float64') # [mV]
    DLOverK0_fit = DLOverK0_fit*1e-3 # [V]
    sorted_indices = np.argsort(EN_Td_exp_Diff_fit)
    EN_Td_exp_Diff_fit = EN_Td_exp_Diff_fit[sorted_indices]
    DLOverK0_fit = DLOverK0_fit[sorted_indices]

    DLOverK0_fit_interp = np.interp(EN_Td,EN_Td_exp_Diff_fit,DLOverK0_fit)  
    DL = DLOverK0_fit_interp * K0
    # DL_Ein = K0 * spc.k * GasTemperature / spc.e  # Einstein Relation
    threshold_EN = 188    
    indices_EN_max = np.searchsorted(EN_Td, threshold_EN)
    DL[indices_EN_max:] = DL[indices_EN_max]

    EN_interp = EN_Td / (1e21 * params.V0L/params.nAr) * 2/3 # We multiply with 2/3 here to make it compatible with  mobility_U function.
    Di_interp = DL[:]*tau/(L*L)
    # Di_interp = uniform_filter1d(Di_interp, size=1)
    Di_spline = CubicSpline(EN_interp, Di_interp)
    Di_EN_spline = CubicSpline.derivative(Di_spline)
    diffusivity = Diffusivity(interpolate = False, D_expression = Di_spline, D_T_expression = Di_EN_spline)
    diffList.append(diffusivity)

    # fig,ax = plt.subplots()
    # ax.set_title('Diffusion Coef.')
    # ax.set_xlabel('E/nAr [Td]')
    # ax.set_ylabel(r"$D_i \, $ [$ \, m^{2}/s$]")
    # ax.plot(EN_Td, Di_interp, marker = '.', label = 'raw')
    # ax.plot(EN_Td, Di_EN_spline(Di_interp), marker = '*', label = 'raw - grad')
    # plt.axhline(y=params.D[1], color='k', linestyle='--')
    # ax.legend()


    #-------------------------------------------------------------------------------
    #########   Species Diffusion Coef.  #########
    for i in range(2, Ns):
        diffList.append(Diffusivity(interpolate = False))


    #-------------------------------------------------------------------------------
    #########   Electron Energy Diffusion Coef.  #########
    energydiffusivityData = transport["energy_diffusivity"] 
    NDee_v_Te = energydiffusivityData[:,1]
    NDee_v_Te[0:indices_Te0] = NDee_v_Te[indices_Te0]
    Dee_interp = (NDee_v_Te[:]/nAr)*tau/(L*L)
    Dee_interp = uniform_filter1d(Dee_interp, size=20)
    Dee_spline = CubicSpline(Te_trans, Dee_interp)
    Dee_Te_spline = CubicSpline.derivative(Dee_spline)
    energydiffusivity = Diffusivity(interpolate = False, D_expression = Dee_spline, D_T_expression = Dee_Te_spline)
    diffList.append(energydiffusivity)

    # fig,ax = plt.subplots()
    # ax.set_title('Energy Diffusion Coef.')
    # ax.set_xlabel('Te [eV]')
    # # ax.set_ylabel('D [m2/s]')
    # ax.set_ylabel(r"$D_{\epsilon} \, $ [$ \, m^{2}/s$]")
    # ax.plot(Te_trans, Dee_interp, marker = '.', label = 'raw')
    # ax.plot(Te_trans, Dee_Te_spline(Te_trans), marker = '*', label = 'raw - grad')
    # plt.axhline(y=params.D[-1], color='k', linestyle='--')
    # ax.legend()



    #-------------------------------------------------------------------------------
    #########   ---------------END----------------  #########
    # plt.show()
    # exit(-1)
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
    # i = Nv - 1  -> ion Effective Electric field


    # CR Indexing 
    # i = 0       -> ground state
    # i = 1:Ns-2  -> excited levels
    # i = Ns - 2  -> electrons
    # i = Ns - 1  -> ions
    # i = Ns      -> electron energy


