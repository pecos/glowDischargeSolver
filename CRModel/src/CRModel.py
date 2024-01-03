# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:04:49 2023

@author: Malamas Tsagkaridis
"""

import numpy as np
# import matplotlib.pyplot as plt
# import os
import scipy.constants as spc
# import csv
# import scipy.io as sio
# import pandas as pd

# from scipy.optimize import fsolve,least_squares,root
# from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
from scipy.optimize._numdiff import approx_derivative,_eps_for_method



from dataclasses import dataclass

from Constants import *

from EquilibriumCalc import *
import ModelParameters as parameters
from GeneralFunctions import escapeFactCalc,escapeFactCalc_vec,CalcLineRadiationLosses

#----------------------------------------------------------------------------------
# raise SystemExit(0) 
#----------------------------------------------------------------------------------


class CollisionalRadiativeModel:
    """
    A 0-D collisional-radiative (CR) model for argon plasma for conditions 
    relevant to atmospheric ICP.

    The plasma is assumed to be homogeneous and neutral. It assumes a Maxwellian 
    electron energy distribution function (EEDF) for now. It includes 
    higher-lying excited states (up to 8s level), which are considered to be 
    radiatively important for the torch.

    Input parameters: Pressure, electron temperature, heavy-particle temperature.

    Atomic processes considered: 
    1) Electron impact excitation / deexcitation 
    2) Electron impact ionization / three-body recombination 
    3) Radiation trapping via escape factors 
    4) Radiative recombination 
    5) Atom-atom collision excitation / deexcitation 
    5) Atom-atom collision ionization / three-body recombination

    Attributes:
        Ns   -- Number of species
        NT   -- Number of temperatures
        Nv   -- Number of state variables (Ns+NT)
        Ndof -- Total number of degrees of freedom

        U0, U1, U2 -- State vectors necessary for BDF2 time step
        params -- modelClosures class (provides model parameters)
    """



    #----------------------------------------------------------------------------------
    ############## Constructor ##############
    #----------------------------------------------------------------------------------

    def __init__(self, Ns, NT, Pressure, GasTemperature, backgroundSpecieActivationFactor = 0):
        """Initializes storage and operators required for solve."""

        # Input parameters
        self.p_0 = Pressure; self.T_g0 = GasTemperature
        
        # Load/read parameters
        self.p = parameters.modelParameters(Ns)
        self.N_lvl = self.p.N_lvl

        # Set number of state variables 
        if (self.p.N_lvl < Ns -2):
            self.Ns = self.p.N_lvl +1 +1  # Number of species (ground state + excited levels + electrons + ground ion state)
        else:
            self.Ns = Ns    # Number of species (ground state + excited levels + ground ion state)
                        
        self.backgroundSpecieActivationFactor = backgroundSpecieActivationFactor
        if (not backgroundSpecieActivationFactor):
            self.Ns = self.Ns -1
        
        self.NT = NT    # Number of temperatures
        self.Nv = self.Ns+NT # Total number of 'state' variables

        self.Ndof = self.Nv # total number of dofs

        # State vector (3 vectors for BDF2)
        self.U2 = np.zeros((self.Ndof,1))
        self.U1 = np.zeros((self.Ndof,1))
        self.U0 = np.zeros((self.Ndof,1))

        # Jacobian storage
        self.jac  = np.zeros((self.Ndof, self.Ndof))

        # Length Scales 
        # # Glow discharge
        # self.R = 0.0129 # discharge radius
        # self.L = 0.0129 # discharge length
        # ICP torch
        # dischR = 0.015 #  radius of the nozle
        scalingFactor = 1.0
        self.R = 0.028/scalingFactor #  radius of the torch
        self.L = self.R*2 # length 


        # if(scenario==1):
        #     setPsaapProperties_6Species_100mTorr_Expanded(gam, V0, VDC, self.params, Nr, iSample)
        # elif(scenario==2):
        #     setPsaapProperties_6Species_Sampling_100mTorr_Expanded(gam, V0, VDC, self.params, Nr, iSample)
        
        self.icall = 0


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

        self.FromCRToGlowDischargeIndexing  = [self.Ns-2, self.Ns-1] + list(range(1,self.Ns-2)) + [0, self.Ns]
        self.FromGlowDischargeToCRIndexing  = [self.Ns-1] + list(range(2,self.Ns-1)) + [0, 1, self.Ns]

        #----------------------------------------------------------------------------------
        """
        For Electron Energy Distribution Function (EEDF) :
        """
        self.eRange = np.logspace(np.log10(1e-3),np.log10(300),1000)  # [eV]
        self.eRange = self.eRange[:, np.newaxis] # I have added np.newaxis so that it works in both vectorized 
                                                 # and non-vectorised versions of the jaccobian calculation. 
                                                 # However, when I remove it, the non-vectrorized version runs faster. 
        self.eVel = np.sqrt(2*self.eRange*spc.e/spc.m_e) # Electron velocity v = sqrt(2 E / m_e)            
        self.aVel = np.sqrt(2*self.eRange*spc.e/M_Ar) # Atom velocity v = sqrt(2 E / m)

        self.p.EvaluateCrossSections(self.eRange)

        # Get the nodes (x) and weights (w) for Gauss-Laguerre quadrature using numpy
        n = 175
        self.xi, self.wi = np.polynomial.laguerre.laggauss(n)                   


        #----------------------------------------------------------------------------------

        # charge number
        self.Z = np.zeros(Ns)
        self.Z[0] = -1 # electrons are always -1
        self.Z[1] =  1 # ions are always 1
        self.Z[2] =  0 # background specie should be 0

    def charge(self,i):
        return self.Z[i]


    def electronImpactIonRateIntegrand(self,Te): 
        # Te -> eV
        # eVel * EEDF * exp(-xi)       
        return np.sqrt(8.0*spc.e/spc.m_e/np.pi/Te) * self.xi

    def atomImpactIonRateIntegrand(self,Tg): 
        # Tg -> eV
        # eVel * EEDF * exp(-xi)       
        return np.sqrt(8.0*spc.e/M_Ar/np.pi/Tg) * self.xi




    def MaxwellianDistribution_vec(self, eRange,T):
        # T -> [eV]
        # eRange -> [eV]
        """
        Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
        """

        # compute EEDF
        EDF = 2 * np.sqrt(eRange/ np.pi) * (T[:] ** (-1.5)) * np.exp(-eRange / T[:])


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
    ############## Rates / Right-hand side ##############
    #----------------------------------------------------------------------------------
  

    def rxnSourceTermJac(self, Uin):
        """Evaluates the Jacobian for backward Euler time marching.
        """

        xk = np.asarray(Uin, float)
        f0 = self.rxnSourceTerm(xk)
        return approx_derivative(self.rxnSourceTerm, xk, method='2-point', abs_step=1e-8,f0=f0)

        # eps=1.0e-12
        # omega_U = approx_fprime(Uin, self.rxnSourceTerm, epsilon=1e-8)
        # return omega_U


    def rxnSourceTermJac_2(self, Uin):
        """Evaluates the Jacobian for backward Euler time marching.
        """

        epsilon=1e-8

        # user specifies an absolute step
        x0 = Uin
        method='2-point'
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = epsilon

        # cannot have a zero step. This might happen if x0 is very large
        # or small. In which case fall back to relative step.
        
        dx = ((x0 + h) - x0)
        h = np.where(dx == 0,
                     _eps_for_method(x0.dtype, x0.dtype, method) *
                     sign_x0 * np.maximum(1.0, np.abs(x0)),
                     h)

        omega_U = np.zeros((self.Ns+1,self.Ns+1),dtype=np.float64)

        omega = self.rxnSourceTerm(Uin)
        Uin_perturbed = Uin.copy()
       
        for i in range(0,self.Ns+1):
            # Perturb the input at index i
            Uin_perturbed[i] += h[i]
            omega_perturbed = self.rxnSourceTerm(Uin_perturbed)
            Uin_perturbed[i] = Uin[i]

            # Compute the partial derivative with respect to the i-th input using finite differences
            # omega_U[:, i] = (omega_perturbed - omega) / epsilon
            omega_U[:, i] = (omega_perturbed - omega) / h[i]
           

        return omega_U


    def rxnSourceTermJac_2_vec(self, Uin):
        """Evaluates the Jacobian for backward Euler time marching.
        """


        epsilon=1e-8

        # user specifies an absolute step
        x0 = Uin
        method='2-point'
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = epsilon

        # cannot have a zero step. This might happen if x0 is very large
        # or small. In which case fall back to relative step.
        dx = ((x0 + h) - x0)
        h = np.where(dx == 0,
                     _eps_for_method(x0.dtype, x0.dtype, method) *
                     sign_x0 * np.maximum(1.0, np.abs(x0)),
                     h)
    


        omega_U = np.zeros((self.Ns+1,self.Ns+1,Uin.shape[0]),dtype=np.float64)

        omega = self.rxnSourceTerm_vec(Uin)
        Uin_perturbed = Uin.copy()
       
       
        for i in range(0,self.Ns+1):
            # Perturb the input at index i
            Uin_perturbed[:,i] += h[:,i]
            omega_perturbed = self.rxnSourceTerm_vec(Uin_perturbed)
            Uin_perturbed[:,i] = Uin[:,i]

            # Compute the partial derivative with respect to the i-th input using finite differences
            h_i = h[:,i]
            omega_U[:, i, :] = np.transpose((omega_perturbed[:,:] - omega[:,:]) / h_i[:, np.newaxis])
            
        return omega,omega_U

    #----------------------------------------------------------------------------------
              
    def rxnSourceTerm(self,y):

        """
        Defines the non-linear rate equations for an Argon I collisional-radiative
        model to compute population of levels.
    
        Arguments:
            npop :  vector of the state variables:
                    npop = [n_0,n_1,....,n_i,n(z=1)]
                    n_g -> atom ground state
                    n_0, ..., n_i -> atom excited states
                    n(z=1) -> ion ground state
                  
            T_e :  electron temperature in [eV] 
        """  
        
        # Indexing 
        # i = 0       -> ground state
        # i = 1:Ns-2  -> excited levels
        # i = Ns - 2  -> electrons
        # i = Ns - 1  -> ions
        # i = Ns      -> electron energy

        # T_g -> [K]  
        # T_e -> [eV] 

        # Clip negative values
        y[np.where(y <= 0.0)] = 0.0

        n_g = y[iNg]    # [#/m^3]
        ne = y[iNe]     # [#/m^3]
        nion = y[iNion] # [#/m^3]
        T_e = y[iEe]    # [eV]
        # Ee = y[iEe] # [eV/m^3] 
        # Eh = y[iEh] # [eV/m^3] 

        p_0 = self.p_0

        # allocate arrays
        npop = np.zeros(self.Ns-2) # ground state + excited levels
        dydt = np.zeros(self.Ns+1) # ground state + excited levels + electrons + ions + Ee #+ Eh
        
        npop[0] = n_g
        npop[1:] = y[1:self.Ns-2]

        # Temperature of heavy species (from ideal gas law)
        #  Ideal gas law: p_0 = p_n + p_i + p_e
        T_g = (p_0/spc.k - ne * T_e/K_eV) / (np.sum(npop) + nion)   # [K]
        
 
        """
        Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
        """
        EEDF= MaxwellianDistribution(self.eRange,T_e,"Electrons")
        AEDF= MaxwellianDistribution(self.eRange,T_g*K_eV,"Atoms")
        EEDFnorm = np.trapz(EEDF,self.eRange,axis=0)
        AEDFnorm = np.trapz(AEDF,self.eRange,axis=0)
        
        # keylist = self.p.collDict.keys()
        
        ################## Elecrton impact Ionization ##################
        # M.T. !< Ionization of Ni due to electrom-atom collisions
        Qi_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)

        i = iNg # from ground state (BSR Data from LXCat)
        deltaIon = Eion - self.p.E_lvl[i]*cm_eV 
                                    
        Si = np.trapz(self.p.sigma_ionBSR*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm   

        
        Qi = self.p.g_lvl[i]*Qi_factor*np.exp(deltaIon/T_e)*Si # I need to check again this one.
            
        Rsi = n_g * ne * Si     # Electron impact ionization
        Rqi = nion * ne * ne * Qi 
        
        dydt[i] = dydt[i] - Rsi + Rqi 
        dydt[iNe] = dydt[iNe] + Rsi - Rqi # rate of change of ion number density
        dydt[iEe] = dydt[iEe] + deltaIon * (Rqi - Rsi) # rate of change of eletron energy

    
        for i in range(1,self.Nv-3):               
                
            # isPrimed_lvl    
            deltaIon = Eion - self.p.E_lvl[i]*cm_eV # Which Eion should I use? There are two!                    
            
            Si = np.trapz(self.p.sigma_ij_Ion[i]*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm
                        
            # sigma_ion = np.interp(self.xi*T_e, self.eRange, self.p.sigma_ij_Ion[i])
            # Si = np.sum(self.wi * sigma_ion *self.electronImpactIonRateIntegrand(T_e) * T_e)
                                     
            Qi = self.p.g_lvl[i]*Qi_factor*np.exp(deltaIon/T_e)*Si # I need to check again this one.
       
            Rsi = npop[i]*ne*Si # Electron impact ionization
            Rqi = nion*ne*ne*Qi 
            
            dydt[i] = dydt[i] - Rsi + Rqi 
            dydt[iNe] = dydt[iNe] + Rsi - Rqi # rate of change of ion number density
            dydt[iEe] = dydt[iEe] + deltaIon * (Rqi - Rsi) # rate of change of eletron energy


        ################## Elecrton impact de/excitation ##################
        # From LXCat BSR data
        for iCollTrans in self.p.CollTransitions_LXCat_BSR: 
            i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.p.CollTransition_ij[iCollTrans,1] # Upper level                    
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
                                  
            Cij = np.trapz(self.p.sigma_ij_Exc_BSR[iCollTrans]*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm                        

            # sigma_ij = np.interp(self.xi*T_e ,self.p.collDict_list[iCollTrans][:,0],self.p.collDict_list[iCollTrans][:,1])
            # sigma_ij[np.where(self.xi*T_e < eij)] = 0
            # Cij = np.sum(self.wi * sigma_ij *self.electronImpactIonRateIntegrand(T_e) * T_e)            
            # print("diff = " , abs(Cij - Cij_2)/Cij*100)

            Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*np.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
            Rcij = Cij*npop[i]*ne
            Rfji = Fji*npop[j]*ne
            dydt[i] = dydt[i] - Rcij + Rfji 
            dydt[j] = dydt[j] + Rcij - Rfji 
            dydt[iEe] = dydt[iEe] + eij * (Rfji - Rcij) # rate of change of eletron energy
        

        for iCollTrans in self.p.CollTransitions_Rest: 
            i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.p.CollTransition_ij[iCollTrans,1] # Upper level                            
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
                          
            Cij = np.trapz(self.p.sigma_ij_Exc[iCollTrans]*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm

            # sigma_ij = np.interp(self.xi*T_e,self.eRange,self.p.sigma_ij_Exc[iCollTrans])
            # Cij_2 = np.sum(self.wi * sigma_ij *self.electronImpactIonRateIntegrand(T_e) * T_e)            
            # print("diff = " , abs(Cij - Cij_2)/Cij*100)

            Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*np.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
            Rcij = Cij*npop[i]*ne
            Rfji = Fji*npop[j]*ne
            dydt[i] = dydt[i] - Rcij + Rfji 
            dydt[j] = dydt[j] + Rcij - Rfji 
            dydt[iEe] = dydt[iEe] + eij * (Rfji - Rcij) # rate of change of eletron energy


        ################# Radiation processes ##################
        for itrans in self.p.EmissionTransitions: 
            """
            Transition data for Ar I
            i -> lower level
            j -> upper level
            """     
        
            i = self.p.index_i_lvl[itrans]
            j = self.p.index_j_lvl[itrans]

            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
            
            # Calculations for escape factor
            eta = escapeFactCalc(npop[i],self.p.E_j[itrans],self.p.E_i[itrans],self.p.g_j[itrans],self.p.g_i[itrans],\
                                 self.p.A_ji[itrans],M_Ar,T_g,self.R,self.L) 
             
            Rspem =  npop[j]*self.p.A_ji[itrans]*eta        
            dydt[i] = dydt[i] + Rspem # radiative transitions into lower state
            dydt[j] = dydt[j] - Rspem # radiative transitions out of higher state
            # dydt[iEh] = dydt[iEh] - eij * Rspem  # Do I need to include that???
        
        
        
        ################## Atom impact Ionization ##################
        Wm_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)
        
        deltaIon = Eion - self.p.E_lvl[0]*cm_eV
        
        Vm = np.trapz(self.p.sigma_1a_ion*self.aVel*AEDF,self.eRange,axis=0)/AEDFnorm        
        Wm = self.p.g_lvl[0]*Wm_factor*np.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent
    
        Rvm = n_g * n_g * Vm
        Rwm = n_g * ne * nion * Wm  
        dydt[0] = dydt[0] - Rvm + Rwm  # remove particle in particular state due to ionization
        dydt[iNe] = dydt[iNe] + Rvm - Rwm # change number of ions 
        # dydt[iEh] = dydt[iEh] + deltaIon * (Rwm - Rvm) # rate of change of eletron energy

        for i in range(1,self.Nv-3):
            
            # Ionization due to atom impact from any level
            deltaIon = Eion - self.p.E_lvl[i]*cm_eV
                                 
            Vm = np.trapz(self.p.sigma_ia_ion[i]*self.aVel*AEDF,self.eRange,axis=0)/AEDFnorm
                        
            # sigma_ia_ion = np.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ia_ion[i])
            # Vm = np.sum(self.wi * sigma_ia_ion * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)
                        
            Wm = self.p.g_lvl[i]*Wm_factor*np.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent

            Rvm = npop[i] * n_g * Vm
            Rwm = n_g * ne * nion * Wm 
                            
            dydt[i] = dydt[i] - Rvm + Rwm # remove particle in particular state due to ionization
            dydt[iNe] = dydt[iNe] + Rvm - Rwm # change number of ions 
            # dydt[iEh] = dydt[iEh] + deltaIon * (Rwm - Rvm) # rate of change of eletron energy
        
            

        ################## Atom impact de/excitation ##################

        i = self.p.i_Atom_ExcFromGround 
        for itrans in self.p.itrans_Atom_ExcFromGround:
            j = self.p.j_Atom_ExcFromGround[itrans]
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

            Kij = np.trapz(self.p.sigma_ij_Atom_ExcFromGround[itrans]*self.aVel*AEDF,self.eRange,axis=0)/AEDFnorm
            # sigma_ij_a = np.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ij_Atom_ExcFromGround[itrans])
            # Kij_2 = np.sum(self.wi * sigma_ij_a * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)  
            # print("diff = ", abs(Kij_2-Kij)/Kij*100)           

            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*np.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            Rkij = npop[i] * n_g * Kij
            Rlji = npop[j] * n_g * Lji  
            dydt[i] = dydt[i] - Rkij + Rlji 
            dydt[j] = dydt[j] + Rkij - Rlji
            # dydt[iEh] = dydt[iEh] + eij * (Rlji - Rkij)


        for itrans in self.p.itrans_Atom_Exc:
            i = self.p.i_Atom_Exc[itrans]
            j = self.p.j_Atom_Exc[itrans] 

            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
            
            Kij = np.trapz(self.p.sigma_ij_Atom_Exc[itrans]*self.aVel*AEDF,self.eRange,axis=0)/AEDFnorm
  
            # sigma_ij_a = np.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ij_Atom_Exc[itrans])           
            # Kij = np.sum(self.wi * sigma_ij_a * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)            
            
            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*np.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            Rkij = npop[i] * n_g * Kij
            Rlji = npop[j] * n_g * Lji  
            dydt[i] = dydt[i] - Rkij + Rlji 
            dydt[j] = dydt[j] + Rkij - Rlji
            # dydt[iEh] = dydt[iEh] + eij * (Rlji - Rkij)                



        ################# Photorecombination/photoionization ##################
        nTrans = 5
        for i in range(0,nTrans): # We include also the photoionization from ground state which has a different cross section

            Ri = np.trapz(self.p.sigma_c_ion[i]*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm
            Ri_prime = np.trapz(self.p.sigma_c_ion[i]*self.eVel*self.eRange*EEDF,self.eRange,axis=0)/EEDFnorm

            # sigma_c_ion = np.interp(self.xi*T_e, self.eRange, self.p.sigma_c_ion[i])
            # Ri = np.sum(self.wi * sigma_c_ion *self.electronImpactIonRateIntegrand(T_e) * T_e)
            # Ri_prime = np.sum(self.wi * sigma_c_ion * self.xi*T_e * self.electronImpactIonRateIntegrand(T_e) * T_e)

            Rri = ne * nion * Ri # What is the reverse process here?
            Rri_prime = ne * nion * Ri_prime
            
            dydt[i] = dydt[i] + Rri 
            dydt[iNe] = dydt[iNe] - Rri 
            dydt[iEe] = dydt[iEe] - Rri_prime # rate of change of eletron energy


        # Bremsstrahlung emission
        # Rbremsstrahlung = 1.42e-40 * parameters.Zeff**2 * np.sqrt(T_e/K_eV) * ne * ne /spc.e  # [eV/m^3/s]
        
        # Energy transfer between electrons and heavy particles 

        # # sigma_el_e1 = np.interp(self.eRange,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # # sigma_el_e1[np.where(sigma_el_e1 < 0)] = 0
        # ken = np.trapz(self.p.sigma_el_e1*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm

        # sigma_el_e1 = np.interp(self.xi*T_e,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # sigma_el_e1[np.where(sigma_el_e1 < 0)] = 0
        # ken = np.sum(self.wi * sigma_el_e1 *self.electronImpactIonRateIntegrand(T_e) * T_e)

        
        # MeanThermalVelocity_e = np.sqrt(8.0*spc.k*T_e/K_eV/np.pi/spc.m_e)
        # Lambda_ei = 1.24e7 * np.sqrt((T_e/K_eV)**3/ne) # Check the units !!!?????
        # MeanSigma_ei = 5.85e-10 * np.log(Lambda_ei)/(T_e/K_eV)**2 # Check the units !!!?????
        # kei = MeanThermalVelocity_e * MeanSigma_ei
            
        # Rtransfer_n = 3.0 * spc.m_e * ne * n_g * (T_g*K_eV - T_e) * ken / M_Ar
        # Rtransfer_i = 3.0 * spc.m_e * ne * ne * (T_g*K_eV - T_e) * kei / (M_Ar - spc.m_e)
        
        # dydt[iEe] = dydt[iEe] #- Rbremsstrahlung + Rtransfer_n + Rtransfer_i
        # dydt[iEh] = dydt[iEh] - Rtransfer_n - Rtransfer_i


        dydt[iNion] = dydt[iNe]  # These rates are always the same! 
                                 # I need to think how we can exploit this to make the computation faster. 
                                 # Especialy for the calculation of the jacobian



        return dydt

    #----------------------------------------------------------------------------------

              
    def rxnSourceTerm_vec(self,y):

        """
        Defines the non-linear rate equations for an Argon I collisional-radiative
        model to compute population of levels.
    
        Arguments:
            npop :  vector of the state variables:
                    npop = [n_0,n_1,....,n_i,n(z=1)]
                    n_g -> atom ground state
                    n_0, ..., n_i -> atom excited states
                    n(z=1) -> ion ground state
                  
            T_e :  electron temperature in [eV] 
        """  
        
        # Indexing 
        # i = 0       -> ground state
        # i = 1:Ns-2  -> excited levels
        # i = Ns - 2  -> electrons
        # i = Ns - 1  -> ions
        # i = Ns      -> electron energy

        # T_g -> [K]  
        # T_e -> [eV] 

        # Clip negative values
        
        # y[:,np.where(y <= 0.0)] = 0.0
        y[y <= 0.0] = 0.0

        n_g = y[:,iNg]    # [#/m^3]
        ne = y[:,iNe]     # [#/m^3]
        nion = y[:,iNion] # [#/m^3]
        T_e = y[:,iEe]    # [eV]
        # Ee = y[:,iEe] # [eV/m^3] 
        # Eh = y[:,iEh] # [eV/m^3] 

        p_0 = self.p_0

        # allocate arrays
        npop = np.zeros((y.shape[0],self.Ns-2)) # ground state + excited levels
        dydt = np.zeros((y.shape[0],self.Ns+1)) # ground state + excited levels + electrons + ions + Ee #+ Eh
        
        npop[:,0] = n_g
        npop[:,1:] = y[:,1:self.Ns-2]

        # Temperature of heavy species (from ideal gas law)
        #  Ideal gas law: p_0 = p_n + p_i + p_e
        T_g = (p_0/spc.k - ne * T_e/K_eV) / (np.sum(npop, axis=1) + nion)   # [K]
 
        """
        Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
        """

        EEDF= self.MaxwellianDistribution_vec(self.eRange,T_e)
        AEDF= self.MaxwellianDistribution_vec(self.eRange,T_g*K_eV)
        EEDFnorm = np.trapz(EEDF,self.eRange, axis=0 )
        AEDFnorm = np.trapz(AEDF,self.eRange, axis=0 )
        
            
        # keylist = self.p.collDict.keys()
        
        ################## Elecrton impact Ionization ##################
        # M.T. !< Ionization of Ni due to electrom-atom collisions
        Qi_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)

        i = iNg # from ground state (BSR Data from LXCat)
        deltaIon = Eion - self.p.E_lvl[i]*cm_eV 
                                    
        Si = np.trapz(self.p.sigma_ionBSR*self.eVel*EEDF,self.eRange, axis=0 )/EEDFnorm        
        Qi = self.p.g_lvl[i]*Qi_factor*np.exp(deltaIon/T_e)*Si # I need to check again this one.


        Rsi = n_g * ne * Si     # Electron impact ionization
        Rqi = nion * ne * ne * Qi 
        
        dydt[:,i] = dydt[:,i] - Rsi + Rqi 
        dydt[:,iNe] = dydt[:,iNe] + Rsi - Rqi # rate of change of ion number density
        dydt[:,iEe] = dydt[:,iEe] + deltaIon * (Rqi - Rsi) # rate of change of eletron energy

    
        for i in range(1,self.Nv-3):               
                
            # isPrimed_lvl    
            deltaIon = Eion - self.p.E_lvl[i]*cm_eV # Which Eion should I use? There are two!                    
            
            Si = np.trapz(self.p.sigma_ij_Ion[i]*self.eVel*EEDF,self.eRange, axis=0 )/EEDFnorm
                        
            # sigma_ion = np.interp(self.xi*T_e, self.eRange, self.p.sigma_ij_Ion[i])
            # Si = np.sum(self.wi * sigma_ion *self.electronImpactIonRateIntegrand(T_e) * T_e)
                                     
            Qi = self.p.g_lvl[i]*Qi_factor*np.exp(deltaIon/T_e)*Si # I need to check again this one.
       
            Rsi = npop[:,i]*ne*Si # Electron impact ionization
            Rqi = nion*ne*ne*Qi 
            
            dydt[:,i] = dydt[:,i] - Rsi + Rqi 
            dydt[:,iNe] = dydt[:,iNe] + Rsi - Rqi # rate of change of ion number density
            dydt[:,iEe] = dydt[:,iEe] + deltaIon * (Rqi - Rsi) # rate of change of eletron energy


        ################## Elecrton impact de/excitation ##################
        # From LXCat BSR data
        for iCollTrans in self.p.CollTransitions_LXCat_BSR: 
            i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.p.CollTransition_ij[iCollTrans,1] # Upper level                    
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
                                  
            Cij = np.trapz(self.p.sigma_ij_Exc_BSR[iCollTrans]*self.eVel*EEDF,self.eRange, axis=0 )/EEDFnorm                        

            # sigma_ij = np.interp(self.xi*T_e ,self.p.collDict_list[iCollTrans][:,0],self.p.collDict_list[iCollTrans][:,1])
            # sigma_ij[np.where(self.xi*T_e < eij)] = 0
            # Cij = np.sum(self.wi * sigma_ij *self.electronImpactIonRateIntegrand(T_e) * T_e)            
            # print("diff = " , abs(Cij - Cij_2)/Cij*100)

            Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*np.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
            Rcij = Cij*npop[:,i]*ne
            Rfji = Fji*npop[:,j]*ne
            dydt[:,i] = dydt[:,i] - Rcij + Rfji 
            dydt[:,j] = dydt[:,j] + Rcij - Rfji 
            dydt[:,iEe] = dydt[:,iEe] + eij * (Rfji - Rcij) # rate of change of eletron energy
        

        for iCollTrans in self.p.CollTransitions_Rest: 
            i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.p.CollTransition_ij[iCollTrans,1] # Upper level                            
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
                          
            Cij = np.trapz(self.p.sigma_ij_Exc[iCollTrans]*self.eVel*EEDF,self.eRange, axis=0 )/EEDFnorm

            # sigma_ij = np.interp(self.xi*T_e,self.eRange,self.p.sigma_ij_Exc[iCollTrans])
            # Cij_2 = np.sum(self.wi * sigma_ij *self.electronImpactIonRateIntegrand(T_e) * T_e)            
            # print("diff = " , abs(Cij - Cij_2)/Cij*100)

            Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*np.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
            Rcij = Cij*npop[:,i]*ne
            Rfji = Fji*npop[:,j]*ne
            dydt[:,i] = dydt[:,i] - Rcij + Rfji 
            dydt[:,j] = dydt[:,j] + Rcij - Rfji 
            dydt[:,iEe] = dydt[:,iEe] + eij * (Rfji - Rcij) # rate of change of eletron energy


        ################# Radiation processes ##################
        for itrans in self.p.EmissionTransitions: 
            """
            Transition data for Ar I
            i -> lower level
            j -> upper level
            """     
        
            i = self.p.index_i_lvl[itrans]
            j = self.p.index_j_lvl[itrans]

            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

            
            # Calculations for escape factor
            eta = escapeFactCalc_vec(npop[:,i],self.p.E_j[itrans],self.p.E_i[itrans],self.p.g_j[itrans],self.p.g_i[itrans],\
                                 self.p.A_ji[itrans],M_Ar,T_g,self.R,self.L) 

            
            Rspem =  npop[:,j]*self.p.A_ji[itrans]*eta        
            dydt[:,i] = dydt[:,i] + Rspem # radiative transitions into lower state
            dydt[:,j] = dydt[:,j] - Rspem # radiative transitions out of higher state
            # dydt[:,iEh] = dydt[:,iEh] - eij * Rspem  # Do I need to include that???

                
        ################## Atom impact Ionization ##################
        Wm_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)
        
        deltaIon = Eion - self.p.E_lvl[0]*cm_eV
        
        Vm = np.trapz(self.p.sigma_1a_ion*self.aVel*AEDF,self.eRange, axis=0 )/AEDFnorm        
        Wm = self.p.g_lvl[0]*Wm_factor*np.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent
    
        Rvm = n_g * n_g * Vm
        Rwm = n_g * ne * nion * Wm  
        dydt[:,0] = dydt[:,0] - Rvm + Rwm  # remove particle in particular state due to ionization
        dydt[:,iNe] = dydt[:,iNe] + Rvm - Rwm # change number of ions 
        # dydt[:,iEh] = dydt[:,iEh] + deltaIon * (Rwm - Rvm) # rate of change of eletron energy

        for i in range(1,self.Nv-3):
            
            # Ionization due to atom impact from any level
            deltaIon = Eion - self.p.E_lvl[i]*cm_eV
                                 
            Vm = np.trapz(self.p.sigma_ia_ion[i]*self.aVel*AEDF,self.eRange, axis=0 )/AEDFnorm
                        
            # sigma_ia_ion = np.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ia_ion[i])
            # Vm = np.sum(self.wi * sigma_ia_ion * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)
                        
            Wm = self.p.g_lvl[i]*Wm_factor*np.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent

            Rvm = npop[:,i] * n_g * Vm
            Rwm = n_g * ne * nion * Wm 
                            
            dydt[:,i] = dydt[:,i] - Rvm + Rwm # remove particle in particular state due to ionization
            dydt[:,iNe] = dydt[:,iNe] + Rvm - Rwm # change number of ions 
            # dydt[:,iEh] = dydt[:,iEh] + deltaIon * (Rwm - Rvm) # rate of change of eletron energy
        
            

        ################## Atom impact de/excitation ##################

        i = self.p.i_Atom_ExcFromGround 
        for itrans in self.p.itrans_Atom_ExcFromGround:
            j = self.p.j_Atom_ExcFromGround[itrans]
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

            Kij = np.trapz(self.p.sigma_ij_Atom_ExcFromGround[itrans]*self.aVel*AEDF,self.eRange, axis=0 )/AEDFnorm
            # sigma_ij_a = np.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ij_Atom_ExcFromGround[itrans])
            # Kij_2 = np.sum(self.wi * sigma_ij_a * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)  
            # print("diff = ", abs(Kij_2-Kij)/Kij*100)                    

            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*np.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            Rkij = npop[:,i] * n_g * Kij
            Rlji = npop[:,j] * n_g * Lji  
            dydt[:,i] = dydt[:,i] - Rkij + Rlji 
            dydt[:,j] = dydt[:,j] + Rkij - Rlji
            # dydt[iEh] = dydt[iEh] + eij * (Rlji - Rkij)


        for itrans in self.p.itrans_Atom_Exc:
            i = self.p.i_Atom_Exc[itrans]
            j = self.p.j_Atom_Exc[itrans] 

            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
            
            Kij = np.trapz(self.p.sigma_ij_Atom_Exc[itrans]*self.aVel*AEDF,self.eRange, axis=0 )/AEDFnorm
  
            # sigma_ij_a = np.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ij_Atom_Exc[itrans])           
            # Kij = np.sum(self.wi * sigma_ij_a * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)            
            
            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*np.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            Rkij = npop[:,i] * n_g * Kij
            Rlji = npop[:,j] * n_g * Lji  
            dydt[:,i] = dydt[:,i] - Rkij + Rlji 
            dydt[:,j] = dydt[:,j] + Rkij - Rlji
            # dydt[:,iEh] = dydt[:,iEh] + eij * (Rlji - Rkij)                



        ################# Photorecombination/photoionization ##################
        nTrans = 5
        for i in range(0,nTrans): # We include also the photoionization from ground state which has a different cross section

            Ri = np.trapz(self.p.sigma_c_ion[i]*self.eVel*EEDF,self.eRange, axis=0 )/EEDFnorm
            Ri_prime = np.trapz(self.p.sigma_c_ion[i]*self.eVel*self.eRange*EEDF,self.eRange, axis=0 )/EEDFnorm

            # sigma_c_ion = np.interp(self.xi*T_e, self.eRange, self.p.sigma_c_ion[i])
            # Ri = np.sum(self.wi * sigma_c_ion *self.electronImpactIonRateIntegrand(T_e) * T_e)
            # Ri_prime = np.sum(self.wi * sigma_c_ion * self.xi*T_e * self.electronImpactIonRateIntegrand(T_e) * T_e)

            Rri = ne * nion * Ri # What is the reverse process here?
            Rri_prime = ne * nion * Ri_prime
            
            dydt[:,i] = dydt[:,i] + Rri 
            dydt[:,iNe] = dydt[:,iNe] - Rri 
            dydt[:,iEe] = dydt[:,iEe] - Rri_prime # rate of change of eletron energy


        # Bremsstrahlung emission
        # Rbremsstrahlung = 1.42e-40 * parameters.Zeff**2 * np.sqrt(T_e/K_eV) * ne * ne /spc.e  # [eV/m^3/s]
        
        # Energy transfer between electrons and heavy particles 

        # # sigma_el_e1 = np.interp(self.eRange,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # # sigma_el_e1[np.where(sigma_el_e1 < 0)] = 0
        # ken = np.trapz(self.p.sigma_el_e1*self.eVel*EEDF,self.eRange, axis=0 )/EEDFnorm

        # sigma_el_e1 = np.interp(self.xi*T_e,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # sigma_el_e1[np.where(sigma_el_e1 < 0)] = 0
        # ken = np.sum(self.wi * sigma_el_e1 *self.electronImpactIonRateIntegrand(T_e) * T_e)

        
        # MeanThermalVelocity_e = np.sqrt(8.0*spc.k*T_e/K_eV/np.pi/spc.m_e)
        # Lambda_ei = 1.24e7 * np.sqrt((T_e/K_eV)**3/ne) # Check the units !!!?????
        # MeanSigma_ei = 5.85e-10 * np.log(Lambda_ei)/(T_e/K_eV)**2 # Check the units !!!?????
        # kei = MeanThermalVelocity_e * MeanSigma_ei
            
        # Rtransfer_n = 3.0 * spc.m_e * ne * n_g * (T_g*K_eV - T_e) * ken / M_Ar
        # Rtransfer_i = 3.0 * spc.m_e * ne * ne * (T_g*K_eV - T_e) * kei / (M_Ar - spc.m_e)
        
        # dydt[:,iEe] = dydt[:,iEe] #- Rbremsstrahlung + Rtransfer_n + Rtransfer_i
        # dydt[:,iEh] = dydt[:,iEh] - Rtransfer_n - Rtransfer_i


        dydt[:,iNion] = dydt[:,iNe] # These rates are always the same! 
                                    # I need to think how we can exploit this to make the computation faster. 
                                    # Especialy for the calculation of the jacobian



        return dydt

    #----------------------------------------------------------------------------------





