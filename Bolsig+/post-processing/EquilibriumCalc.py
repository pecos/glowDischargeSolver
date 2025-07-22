# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:04:49 2023

@author: Malamas Tsagkaridis
"""

import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt

from Constants import *

lambda_factor = spc.h**2/(2.0*np.pi*spc.m_e*spc.k)*K_eV

#----------------------------------------------------------------------------------

def PartitionFunctions(T_e,E_lvl,g_lvl,E_lvl_ArII,g_lvl_ArII,iflag):
    
    # Calculate partition functions from energy levels
    Q_n_calc = (np.sum(g_lvl*np.exp(-E_lvl*cm_eV/T_e)))             # Electronic partition function of neutral Argon
    Q_i_calc = (np.sum(g_lvl_ArII*np.exp(-E_lvl_ArII*cm_eV/T_e)))   # Electronic partition function of ion Argon
    # Q_i_calc = g_ion  # Electronic partition function of ion Argon

    # Calculate partition functions from analytical expressions
    Q_n = (1+12*np.exp(-11.6/T_e))                                  # Electronic partition function of neutral Argon
    Q_i = (4+2*np.exp(-0.178/T_e)+2*np.exp(-13.5/T_e))             # Electronic partition function of ion Argon

    if iflag == 0: # from energy levels
        PartitionFun_n = Q_n_calc
        PartitionFun_i = Q_i_calc
    elif iflag ==1: # from analytical expressions
        PartitionFun_n = Q_n
        PartitionFun_i = Q_i        

    # print("Electronic partition function of neutral Argon = ", PartitionFun_n) 
    # print("Electronic partition function of ion Argon = ", PartitionFun_i) 

    return PartitionFun_n,PartitionFun_i

#----------------------------------------------------------------------------------


def SahaRelation_V(p_0,T_e,T_g,Q_n,Q_i):

    # We assume constant mass here. n_0 = n_neutral + n_e 
    n_0 = p_0/T_g/spc.k    # [1/m^3] Number density based on bulk temperature (not necessarily true density in two-temperature gas)

    """
    Construct initial condition of system of equations based on LTE assumption
    Saha relation
    """
    # Ionized ground states population assuimng two-temperature plasma
    nion_LTE = np.zeros(z)
    n_e_LTE = 0
    # solve for ionized state, considering only one ionized state
    for i in range(z):
        lambda_e = spc.h/(np.sqrt(2*np.pi*spc.m_e*spc.k*T_e/K_eV))
        Qrat = Q_i/Q_n # # ratio of partition functions 
        tempSaha = Qrat*2*(lambda_e)**(-3)*np.exp(-Eion[i]/T_e)  # Saha equation for singly ionized system
        nion_LTE[i] = -tempSaha/2 + np.sqrt(tempSaha**2/4 + n_0*tempSaha)
        n_e_LTE = n_e_LTE + nion_LTE[i]*(i+1)
           
    n_neutral = n_0 - n_e_LTE

    # print("Print some initial conditions:")
    # print("Initial number density [#/m^3]      = ", n_0)
    # print("Number density of atoms [#/m^3]     = ", n_neutral)
    # print("Number density of electrons [#/m^3] = ", n_e_LTE)
    # print("Number density of ions [#/m^3]      = ", nion_LTE[0])
    # print("Degree of ionization [%]            = ", n_e_LTE/(n_neutral+nion_LTE[0])*100.0) # How do we define the degree of ionization? 


    return n_0,n_neutral,n_e_LTE,nion_LTE

def SahaRelation_P(p_0,T_e,T_g,Q_n,Q_i):

    # We assume constant pressure here. p_0 = p_n + p_i + p_e 
    n_0 = p_0/T_g/spc.k    # [1/m^3] Number density based on bulk temperature (not necessarily true density in two-temperature gas)

    """
    Construct initial condition of system of equations based on LTE assumption
    Saha relation
    """
    # Ionized ground states population assuimng two-temperature plasma
    nion_LTE = np.zeros(z)
    n_e_LTE = 0
    # solve for ionized state, considering only one ionized state
    tempRatio = (T_e/K_eV/T_g + 1)
    for i in range(z):
        lambda_e = spc.h/(np.sqrt(2*np.pi*spc.m_e*spc.k*T_e/K_eV))
        Qrat = Q_i/Q_n # # ratio of partition functions 
        tempSaha = Qrat*2*(lambda_e)**(-3)*np.exp(-Eion[i]/T_e)  # Saha equation for singly ionized system
        nion_LTE[i] = -tempRatio*tempSaha/2 + np.sqrt(tempRatio**2*tempSaha**2/4 + n_0*tempSaha)
        n_e_LTE = n_e_LTE + nion_LTE[i]*(i+1)
    
    n_neutral = n_0 - tempRatio*n_e_LTE

    

    return n_0,n_neutral,n_e_LTE,nion_LTE

#----------------------------------------------------------------------------------


def SahaDistribution(n_e_LTE,T_e,E_lvl,g_lvl):
    DeltaE = 0.0 # Ionization potential lowering
    q = len(E_lvl)
    # Excited level populations
    npop_Saha = np.zeros(q)
    for i in range(q):
        npop_Saha[i] = n_e_LTE**2 * g_lvl[i]/2.0/g_ion * (lambda_factor/T_e)**(1.5) * np.exp((Eion-E_lvl[i]*cm_eV - DeltaE)/T_e)

    return npop_Saha

#----------------------------------------------------------------------------------

def BoltzmannDistribution(n_tot,T_e,Qtot,E_lvl,g_lvl):

    q = len(E_lvl)
    # Excited level populations
    npop_LTE = np.zeros(q)
    for i in range(q):
        if i == 0:
            npop_LTE[i] = g_lvl[i]/Qtot*n_tot
            continue    
        npop_LTE[i] = g_lvl[i]*np.exp(-E_lvl[i]*cm_eV/T_e)/Qtot*n_tot
    
    # fig,ax = plt.subplots(dpi=140)
    # ax.scatter(E_lvl*cm_eV,npop_LTE/g_lvl,c='b',label='LTE Boltzmann')
    # ax.semilogy()
    # # plt.ylim([1e2,1e26])
    # plt.xlabel('E [ev]')
    # plt.ylabel('n [m$^{-3}]$')
    # plt.title('State populations')
    # plt.grid(True)
    # plt.legend()
       
    return npop_LTE
              
#----------------------------------------------------------------------------------


def MaxwellianDistribution(eRange,T,title):
    # T -> [eV]
    # eRange -> [eV]
    """
    Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
    """

    # compute EEDF
    EDF = 2*np.sqrt(eRange/np.pi)*(T)**(-1.5)*np.exp(-eRange/(T))

    # fig,ax = plt.subplots(dpi=140)
    # ax.scatter(eRange,EDF,c='r',label='EDF')
    # # plt.ylim([1e2,1e26])
    # plt.legend()
    # ax.semilogx()
    # plt.xlabel('E [ev]')
    # plt.ylabel('f(E)')
    # plt.title(title)
    # plt.grid(True)

     
    return EDF

#----------------------------------------------------------------------------------
def PlanckFunction_lambda(lambda_P,T):
    # T - > [K]
    # lambda_P -> [m]
    # B_lambda -> [W/m^3/sr]
    B_lambda = 2.0*spc.h*spc.c**2/lambda_P**5/(np.exp(spc.h*spc.c/(lambda_P*spc.k*T))-1)  

    return B_lambda

def PlanckFunction_freq(freq,T):
    # T - > [K]
    # freq -> [m]
    # B_freq -> [W/m^3/sr]
    B_freq = 2.0*spc.h*freq**3/spc.c**2/(np.exp(spc.h*freq/(spc.k*T))-1)  

    return B_freq

def PlanckFunctionPartialDerivative_T(lambda_P,T):
    # T - > [K]
    # lambda_P -> [m]
    # dB_lambda_dT  -> [W/m^3/sr/K]
    a = 2.0*spc.h*spc.c**2/lambda_P**5
    b = spc.h*spc.c/(lambda_P*spc.k)
    dB_lambda_dT = a*b*np.exp(b/T)/T**2/(np.exp(b/T) -1)**2 

    return dB_lambda_dT
