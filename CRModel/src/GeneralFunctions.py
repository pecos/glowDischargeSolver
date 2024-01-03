# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:04:49 2023

@author: Malamas Tsagkaridis
"""

import numpy as np
import scipy.constants as spc

from Constants import *



def CalcOscillationStrength(E_i,E_j,g_i,g_j,A_ji):
    
    lambda_0 = spc.h*spc.c/((E_j-E_i)*cm_eV*spc.e) # wavelength of transition

    # OscillationStrength = lambda_0**2 / 8.0 / np.pi**2 * A_ji[itrans] / spc.e**2 * spc.m_e * spc.c * g_j[itrans]/g_i[itrans] # Hanson's
    # OscillationStrength = lambda_0**2 / 8.0 / np.pi * A_ji[itrans] * g_j[itrans]/g_i[itrans] * ( 0.0265e-4) / VacPermittivity

    OscillationStrength = lambda_0**2 / 8.0 / np.pi * A_ji * g_j/g_i * ( 4 * spc.m_e * spc.c * VacPermittivity/ spc.e**2) # I need to check this one! It gives the same oscillation strengths with NIST but the relationship is different than Hansons.

    return OscillationStrength

def CalcEinsteinCoef(E_i,E_j,g_i,g_j,f_ji):
    
    lambda_0 = spc.h*spc.c/((E_j-E_i)*cm_eV*spc.e) # wavelength of transition


    A_ji = 1.0/(lambda_0**2 / 8.0 / np.pi * g_j/g_i * ( 4 * spc.m_e * spc.c * VacPermittivity/ spc.e**2)) * f_ji

    return A_ji



def escapeFactCalc(n_i,E_j,E_i,g_j,g_i,A_ji,Mspecies,T_g,R,L):
    # Calculations for escape factor
    lambda_0 = spc.h*spc.c/((E_j-E_i)*cm_eV*spc.e) # wavelength of transition
    k0 = lambda_0**3*n_i*g_j*A_ji*Mspecies**0.5/(8*np.pi*g_i*(2*spc.k*np.pi*T_g)**0.5) # absorption coefficient at line center, for Doppler absorption
    q0 = R
    Lq = L/(2*q0)              
    # eta = (2 - np.exp(-1e-3*k0*R))/(1 + k0*R) # Mewe (1967)
    # eta = 1.
    if k0*(L/2) > 1 and k0*q0 > 1: # compute escape factor      
        # eta = 1.6/(k0*R*(np.pi*np.log(k0*R))**0.5) # Iordanova/Holstein
        eta = (2/(np.sqrt(np.pi*np.log(k0*L/2))*k0*L)/(2*Lq**2 + 2)
                + 1/(np.sqrt(np.pi*np.log(k0*q0))*k0*2*q0)*
                (Lq/(Lq**2 + 1) + np.arctan(Lq))) # Chai & Kwon Doppler lineshape
        # eta = (1/(np.sqrt(np.pi*k0*L/2))*(2/3 - 2*Lq**1.5/(3*(Lq**2 + 1)**0.75))
        #         + 1/(2*np.sqrt(np.pi*k0*q0))*
        #         4*Lq*np.sqrt(Lq)/(3*(Lq**2 + 1)**0.75)) # Golubovskii et al. Lorentz lineshape                       
        # eta = 0.0
    else:
        eta = 1.0    
    
    eta = min(eta,1.0)    
    return eta


def escapeFactCalc_vec(n_i,E_j,E_i,g_j,g_i,A_ji,Mspecies,T_g,R,L):
    # Calculations for escape factor

    lambda_0 = spc.h*spc.c/((E_j-E_i)*cm_eV*spc.e) # wavelength of transition
    k0 = lambda_0**3*n_i*g_j*A_ji*Mspecies**0.5/(8*np.pi*g_i*(2*spc.k*np.pi*T_g)**0.5) # absorption coefficient at line center, for Doppler absorption

    q0 = R
    Lq = L/(2*q0)      
    
    # Compute escape factor using vectorized conditions    
    eta = np.where(((k0 * (L / 2) > 1.0) & (k0 * q0 > 1.0)),
                   (2 / (np.sqrt(np.pi * np.log(k0 * L / 2)) * k0 * L) / (2 * Lq**2 + 2)
                    + 1 / (np.sqrt(np.pi * np.log(k0 * q0)) * k0 * 2 * q0) *
                    (Lq / (Lq**2 + 1) + np.arctan(Lq))),
                   1.0)
    
    eta = np.where(eta > 1.0,1.0, eta)   
    return eta


def CalcLineRadiationLosses(npop,n_e,*p):
    """
    Calculates the absorption coefficients for each transition
    
    Arguments:
        npop :  vector of the state variables:
                  npop = [n_1,n_2,....,n_i]
        n_e :  number density of electrons          
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """  
    
    # unpack parameters    
    (T_g,R,L,
    E_lvl, DictRacah_lvl,
    A_ji, E_i, E_j, g_i, g_j,Racah_i,  Racah_j, EmissionTransitions,
    ) = p 


    EmittedRadiation = 0.0
    ################# Radiation processes ##################
    # nTrans = len(A_ji)
    # for itrans in range(nTrans): 
    for itrans in EmissionTransitions: 
        """
        Transition data for Ar I
        i -> lower level
        j -> upper level
        """
        i = DictRacah_lvl[Racah_i[itrans]]
        j = DictRacah_lvl[Racah_j[itrans]]

        eij = (E_lvl[j] - E_lvl[i])*cm_eV
         
        # Calculations for escape factor
        eta = escapeFactCalc(npop[i],E_j[itrans],E_i[itrans],g_j[itrans],g_i[itrans],A_ji[itrans],M_Ar,T_g,R,L)  
        
        # print(itrans,i,j,eta)

        EmittedRadiation = EmittedRadiation + npop[j]*A_ji[itrans]* eij*eta 


    return EmittedRadiation*spc.e*1e-3 # kW / m^3

  
