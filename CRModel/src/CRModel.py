# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:04:49 2023

@author: Malamas Tsagkaridis
"""
import sys
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
# from scipy.optimize._numdiff import _eps_for_method,approx_derivative

# import cupyx.scipy.sparse.linalg

import time as cpu_time

try:
  import cupy as cp
  #CUDA_NUM_DEVICES=cp.cuda.runtime.getDeviceCount()
except ImportError:
  print("Please install CuPy for GPU use")
  #sys.exit(0)
except:
  print("CUDA not configured properly !!!")
  sys.exit(0)



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

    def __init__(self, args, Ns, NT, Pressure, GasTemperature, backgroundSpecieActivationFactor = 0):
        """Initializes storage and operators required for solve."""

        self.args       = args
        self.xp_module  = np
        
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
        # if (not backgroundSpecieActivationFactor):
        #     self.Ns = self.Ns -1
        
        self.NT = NT    # Number of temperatures
        self.Nv = self.Ns+NT # Total number of 'state' variables

        self.Ndof = self.Nv # total number of dofs

        # State vector (3 vectors for BDF2)
        # self.U2 = np.zeros((self.Ndof,1))
        # self.U1 = np.zeros((self.Ndof,1))
        # self.U0 = np.zeros((self.Ndof,1))

        # Jacobian storage
        # self.jac  = np.zeros((self.Ndof, self.Ndof))

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
        self.eRange_diff = np.diff(self.eRange, axis=0) # Needed to optemize trapz intregration


        self.eVel = np.sqrt(2*self.eRange*spc.e/spc.m_e) # Electron velocity v = sqrt(2 E / m_e)            
        self.aVel = np.sqrt(2*self.eRange*spc.e/M_Ar) # Atom velocity v = sqrt(2 E / m)

        self.p.EvaluateCrossSections(self.eRange)
        self.p.ConvertCrossSectionsToNumPy()

        # Get the nodes (x) and weights (w) for Gauss-Laguerre quadrature using numpy
        n = 175
        self.xi, self.wi = np.polynomial.laguerre.laggauss(n)                   


        #----------------------------------------------------------------------------------

        # charge number
        self.Z = np.zeros(Ns)
        self.Z[0] = -1 # electrons are always -1
        self.Z[1] =  1 # ions are always 1
        self.Z[2] =  0 # background specie should be 0

        #----------------------------------------------------------------------------------

        self.npop = np.zeros((150,self.Ns-2)) # ground state + excited levels
        self.dydt = np.zeros((150,self.Ns+1)) # ground state + excited levels + electrons + ions + Ee #+ Eh
     
 



    def charge(self,i):
        return self.Z[i]


    def electronImpactIonRateIntegrand(self,Te): 
        xp = self.xp_module
        # Te -> eV
        # eVel * EEDF * exp(-xi)       
        return xp.sqrt(8.0*spc.e/spc.m_e/xp.pi/Te) * self.xi

    def atomImpactIonRateIntegrand(self,Tg): 
        xp = self.xp_module
        # Tg -> eV
        # eVel * EEDF * exp(-xi)       
        return xp.sqrt(8.0*spc.e/M_Ar/xp.pi/Tg) * self.xi


    def MaxwellianDistribution_vec(self, eRange,T):
        xp = self.xp_module

        # T -> [eV]
        # eRange -> [eV]
        """
        Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
        """

        # compute EEDF
        EDF = 2 * xp.sqrt(eRange/ xp.pi) * (T[:] ** (-1.5)) * xp.exp(-eRange / T[:])
        
        return EDF




    def copy_operators_Host2Device(self, dev_id):
      
      if self.args.use_gpu==0:
        return
      
      with cp.cuda.Device(dev_id):     
        
        # self.U2             = cp.asarray(self.U2)
        # self.U1             = cp.asarray(self.U1)
        # self.U0             = cp.asarray(self.U0)

        self.eRange         = cp.asarray(self.eRange)
        self.eRange_diff    = cp.asarray(self.eRange_diff)

        self.eVel           = cp.asarray(self.eVel)
        self.aVel           = cp.asarray(self.aVel)

        # self.jac            = cp.asarray(self.jac)
        self.xi             = cp.asarray(self.xi)
        self.wi             = cp.asarray(self.wi)


        # for key in self.p.sigma_ij_Exc_BSR: 
            # self.p.sigma_ij_Exc_BSR[key] = cp.asarray(self.p.sigma_ij_Exc_BSR[key]) 
        self.p.sigma_ij_Exc_BSR = cp.asarray(self.p.sigma_ij_Exc_BSR) 

        for key in self.p.sigma_ij_Exc: 
            self.p.sigma_ij_Exc[key] = cp.asarray(self.p.sigma_ij_Exc[key])

        self.p.sigma_ionBSR  = cp.asarray(self.p.sigma_ionBSR)

        # for key in self.p.sigma_ij_Ion:
            # self.p.sigma_ij_Ion[key] = cp.asarray(self.p.sigma_ij_Ion[key])  
        self.p.sigma_ij_Ion = cp.asarray(self.p.sigma_ij_Ion)  

        self.p.sigma_1a_ion = cp.asarray(self.p.sigma_1a_ion)

        # for key in self.p.sigma_ia_ion:
            # self.p.sigma_ia_ion[key] = cp.asarray(self.p.sigma_ia_ion[key])
        self.p.sigma_ia_ion = cp.asarray(self.p.sigma_ia_ion)

        for key in self.p.sigma_ij_Atom_ExcFromGround:
            self.p.sigma_ij_Atom_ExcFromGround[key]  = cp.asarray(self.p.sigma_ij_Atom_ExcFromGround[key] )

        for key in self.p.sigma_ij_Atom_Exc:
            self.p.sigma_ij_Atom_Exc[key] = cp.asarray(self.p.sigma_ij_Atom_Exc[key] )

        # for key in self.p.sigma_c_ion:
            # self.p.sigma_c_ion[key] = cp.asarray(self.p.sigma_c_ion[key])
        self.p.sigma_c_ion = cp.asarray(self.p.sigma_c_ion)


        self.p.sigma_el_e1 = cp.asarray(self.p.sigma_el_e1)
        self.p.eRange_elastic_e1 = cp.asarray(self.p.eRange_elastic_e1)
        self.p.sigma_elastic_e1 = cp.asarray(self.p.sigma_elastic_e1)

        
        # self.op_rate = [cp.asarray(self.op_rate[i]) for i in range(len(self.op_rate))]

        
      return
    



    def copy_operators_Device2Host(self, dev_id):
      
      if self.args.use_gpu==0:
        return

    #   with cp.cuda.Device(dev_id):     
        
    #     self.U2             = cp.asnumpy(self.U2)
    #     self.U1             = cp.asnumpy(self.U1)
    #     self.U0             = cp.asnumpy(self.U0)

      return



    #----------------------------------------------------------------------------------
    ############## Rates / Right-hand side ##############
    #----------------------------------------------------------------------------------
  

    def rxnSourceTermJac(self, Uin):
        """Evaluates the Jacobian for backward Euler time marching.
        """

        # xk = np.asarray(Uin, float)
        # f0 = self.rxnSourceTerm(xk)
        # return approx_derivative(self.rxnSourceTerm, xk, method='2-point', abs_step=1e-8,f0=f0)

        # eps=1.0e-12
        return approx_fprime(Uin, self.rxnSourceTerm, epsilon=1e-8)


    def rxnSourceTermJac_2(self, Uin):
        """Evaluates the Jacobian for backward Euler time marching.
        """
        xp = self.xp_module

        epsilon=xp.sqrt(xp.finfo(float).eps)

        # user specifies an absolute step
        x0 = Uin
        method='2-point'
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = epsilon

        # cannot have a zero step. This might happen if x0 is very large
        # or small. In which case fall back to relative step.
                
        dx = ((x0 + h) - x0)
        h = xp.where(dx == 0, epsilon * sign_x0 * xp.maximum(1.0, xp.abs(x0)), h)

        omega_U = xp.zeros((self.Ns+1,self.Ns+1),dtype=xp.float64)

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
        xp = self.xp_module

        epsilon=xp.sqrt(xp.finfo(float).eps)

        # user specifies an absolute step
        x0 = Uin
        method='2-point'
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = epsilon

        # cannot have a zero step. This might happen if x0 is very large
        # or small. In which case fall back to relative step.
        dx = ((x0 + h) - x0)
        h = xp.where(dx == 0, epsilon * sign_x0 * xp.maximum(1.0, xp.abs(x0)), h)
    


        omega_U = xp.zeros((self.Ns+1,self.Ns+1,Uin.shape[0]),dtype=xp.float64)

        omega = self.rxnSourceTerm_vec(Uin)
        Uin_perturbed = Uin.copy()
       
       
        for i in range(0,self.Ns+1):
            # Perturb the input at index i
            Uin_perturbed[:,i] += h[:,i]
            omega_perturbed = self.rxnSourceTerm_vec(Uin_perturbed)
            Uin_perturbed[:,i] = Uin[:,i]

            # Compute the partial derivative with respect to the i-th input using finite differences
            h_i = h[:,i]
            omega_U[:, i, :] = xp.transpose((omega_perturbed[:,:] - omega[:,:]) / h_i[:, xp.newaxis])
            
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
        xp = self.xp_module
        
        # Indexing 
        # i = 0       -> ground state
        # i = 1:Ns-2  -> excited levels
        # i = Ns - 2  -> electrons
        # i = Ns - 1  -> ions
        # i = Ns      -> electron energy

        # T_g -> [K]  
        # T_e -> [eV] 

        # Clip negative values
        y[xp.where(y <= 0.0)] = 0.0

        n_g = y[iNg]    # [#/m^3]
        ne = y[iNe]     # [#/m^3]
        nion = y[iNion] # [#/m^3]
        T_e = y[iEe]    # [eV]
        # Ee = y[iEe] # [eV/m^3] 
        # Eh = y[iEh] # [eV/m^3] 

        p_0 = self.p_0

        # allocate arrays
        npop = xp.zeros(self.Ns-2) # ground state + excited levels
        dydt = xp.zeros(self.Ns+1) # ground state + excited levels + electrons + ions + Ee #+ Eh
        
        npop[0] = n_g
        npop[1:] = y[1:self.Ns-2]

        # Temperature of heavy species (from ideal gas law)
        #  Ideal gas law: p_0 = p_n + p_i + p_e
        T_g = (p_0/spc.k - ne * T_e/K_eV) / (xp.sum(npop) + nion)   # [K]
        
 
        """
        Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
        """
        EEDF= MaxwellianDistribution(self.eRange,T_e,"Electrons")
        AEDF= MaxwellianDistribution(self.eRange,T_g*K_eV,"Atoms")
        EEDFnorm = self.trapz(EEDF,self.eRange,axis=0)
        AEDFnorm = self.trapz(AEDF,self.eRange,axis=0)
        # EEDF /= EEDFnorm
        # AEDF /= AEDFnorm
        
        # keylist = self.p.collDict.keys()
        
        ################## Elecrton impact Ionization ##################
        # M.T. !< Ionization of Ni due to electrom-atom collisions
        Qi_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)

        i = iNg # from ground state (BSR Data from LXCat)
        deltaIon = Eion - self.p.E_lvl[i]*cm_eV 
                                    
        Si = self.trapz(self.p.sigma_ionBSR*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm   

        
        Qi = self.p.g_lvl[i]*Qi_factor*xp.exp(deltaIon/T_e)*Si # I need to check again this one.
            
        Rsi = n_g * ne * Si     # Electron impact ionization
        Rqi = nion * ne * ne * Qi 
        
        dydt[i] = dydt[i] - Rsi + Rqi 
        dydt[iNe] = dydt[iNe] + Rsi - Rqi # rate of change of ion number density
        dydt[iEe] = dydt[iEe] + deltaIon * (Rqi - Rsi) # rate of change of eletron energy

    
        for i in range(1,self.Nv-3):               
                
            # isPrimed_lvl    
            deltaIon = Eion - self.p.E_lvl[i]*cm_eV # Which Eion should I use? There are two!                    
            
            Si = self.trapz(self.p.sigma_ij_Ion[i]*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm
                        
            # sigma_ion = xp.interp(self.xi*T_e, self.eRange, self.p.sigma_ij_Ion[i])
            # Si = xp.sum(self.wi * sigma_ion *self.electronImpactIonRateIntegrand(T_e) * T_e)
                                     
            Qi = self.p.g_lvl[i]*Qi_factor*xp.exp(deltaIon/T_e)*Si # I need to check again this one.
       
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
                                  
            Cij = self.trapz(self.p.sigma_ij_Exc_BSR[iCollTrans]*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm                        

            # sigma_ij = xp.interp(self.xi*T_e ,self.p.collDict_list[iCollTrans][:,0],self.p.collDict_list[iCollTrans][:,1])
            # sigma_ij[xp.where(self.xi*T_e < eij)] = 0
            # Cij = xp.sum(self.wi * sigma_ij *self.electronImpactIonRateIntegrand(T_e) * T_e)            
            # print("diff = " , abs(Cij - Cij_2)/Cij*100)

            Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
            Rcij = Cij*npop[i]*ne
            Rfji = Fji*npop[j]*ne
            dydt[i] = dydt[i] - Rcij + Rfji 
            dydt[j] = dydt[j] + Rcij - Rfji 
            dydt[iEe] = dydt[iEe] + eij * (Rfji - Rcij) # rate of change of eletron energy
        

        for iCollTrans in self.p.CollTransitions_Rest: 
            i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.p.CollTransition_ij[iCollTrans,1] # Upper level                            
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
                          
            Cij = self.trapz(self.p.sigma_ij_Exc[iCollTrans]*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm

            # sigma_ij = xp.interp(self.xi*T_e,self.eRange,self.p.sigma_ij_Exc[iCollTrans])
            # Cij_2 = xp.sum(self.wi * sigma_ij *self.electronImpactIonRateIntegrand(T_e) * T_e)            
            # print("diff = " , abs(Cij - Cij_2)/Cij*100)

            Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
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
        
        Vm = self.trapz(self.p.sigma_1a_ion*self.aVel*AEDF,self.eRange,axis=0)/AEDFnorm        
        Wm = self.p.g_lvl[0]*Wm_factor*xp.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent
    
        Rvm = n_g * n_g * Vm
        Rwm = n_g * ne * nion * Wm  
        dydt[0] = dydt[0] - Rvm + Rwm  # remove particle in particular state due to ionization
        dydt[iNe] = dydt[iNe] + Rvm - Rwm # change number of ions 
        # dydt[iEh] = dydt[iEh] + deltaIon * (Rwm - Rvm) # rate of change of eletron energy

        for i in range(1,self.Nv-3):
            
            # Ionization due to atom impact from any level
            deltaIon = Eion - self.p.E_lvl[i]*cm_eV
                                 
            Vm = self.trapz(self.p.sigma_ia_ion[i]*self.aVel*AEDF,self.eRange,axis=0)/AEDFnorm
                        
            # sigma_ia_ion = xp.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ia_ion[i])
            # Vm = xp.sum(self.wi * sigma_ia_ion * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)
                        
            Wm = self.p.g_lvl[i]*Wm_factor*xp.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent

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

            Kij = self.trapz(self.p.sigma_ij_Atom_ExcFromGround[itrans]*self.aVel*AEDF,self.eRange,axis=0)/AEDFnorm
            # sigma_ij_a = xp.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ij_Atom_ExcFromGround[itrans])
            # Kij_2 = xp.sum(self.wi * sigma_ij_a * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)  
            # print("diff = ", abs(Kij_2-Kij)/Kij*100)           

            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            Rkij = npop[i] * n_g * Kij
            Rlji = npop[j] * n_g * Lji  
            dydt[i] = dydt[i] - Rkij + Rlji 
            dydt[j] = dydt[j] + Rkij - Rlji
            # dydt[iEh] = dydt[iEh] + eij * (Rlji - Rkij)


        for itrans in self.p.itrans_Atom_Exc:
            i = self.p.i_Atom_Exc[itrans]
            j = self.p.j_Atom_Exc[itrans] 

            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
            
            Kij = self.trapz(self.p.sigma_ij_Atom_Exc[itrans]*self.aVel*AEDF,self.eRange,axis=0)/AEDFnorm
  
            # sigma_ij_a = xp.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ij_Atom_Exc[itrans])           
            # Kij = xp.sum(self.wi * sigma_ij_a * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)            
            
            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            Rkij = npop[i] * n_g * Kij
            Rlji = npop[j] * n_g * Lji  
            dydt[i] = dydt[i] - Rkij + Rlji 
            dydt[j] = dydt[j] + Rkij - Rlji
            # dydt[iEh] = dydt[iEh] + eij * (Rlji - Rkij)                



        ################# Photorecombination/photoionization ##################
        nTrans = 5
        for i in range(0,nTrans): # We include also the photoionization from ground state which has a different cross section

            Ri = self.trapz(self.p.sigma_c_ion[i]*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm
            Ri_prime = self.trapz(self.p.sigma_c_ion[i]*self.eVel*self.eRange*EEDF,self.eRange,axis=0)/EEDFnorm

            # sigma_c_ion = xp.interp(self.xi*T_e, self.eRange, self.p.sigma_c_ion[i])
            # Ri = xp.sum(self.wi * sigma_c_ion *self.electronImpactIonRateIntegrand(T_e) * T_e)
            # Ri_prime = xp.sum(self.wi * sigma_c_ion * self.xi*T_e * self.electronImpactIonRateIntegrand(T_e) * T_e)

            Rri = ne * nion * Ri # What is the reverse process here?
            Rri_prime = ne * nion * Ri_prime
            
            dydt[i] = dydt[i] + Rri 
            dydt[iNe] = dydt[iNe] - Rri 
            dydt[iEe] = dydt[iEe] - Rri_prime # rate of change of eletron energy


        # Bremsstrahlung emission
        # Rbremsstrahlung = 1.42e-40 * parameters.Zeff**2 * xp.sqrt(T_e/K_eV) * ne * ne /spc.e  # [eV/m^3/s]
        
        # Energy transfer between electrons and heavy particles 

        # # sigma_el_e1 = xp.interp(self.eRange,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # # sigma_el_e1[xp.where(sigma_el_e1 < 0)] = 0
        # ken = self.trapz(self.p.sigma_el_e1*self.eVel*EEDF,self.eRange,axis=0)/EEDFnorm

        # sigma_el_e1 = xp.interp(self.xi*T_e,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # sigma_el_e1[xp.where(sigma_el_e1 < 0)] = 0
        # ken = xp.sum(self.wi * sigma_el_e1 *self.electronImpactIonRateIntegrand(T_e) * T_e)

        
        # MeanThermalVelocity_e = xp.sqrt(8.0*spc.k*T_e/K_eV/xp.pi/spc.m_e)
        # Lambda_ei = 1.24e7 * xp.sqrt((T_e/K_eV)**3/ne) # Check the units !!!?????
        # MeanSigma_ei = 5.85e-10 * xp.log(Lambda_ei)/(T_e/K_eV)**2 # Check the units !!!?????
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


        xp = self.xp_module


        # if xp == cp:
        #   cp.cuda.runtime.deviceSynchronize()

    
        # Indexing 
        # i = 0       -> ground state
        # i = 1:Ns-2  -> excited levels
        # i = Ns - 2  -> electrons
        # i = Ns - 1  -> ions
        # i = Ns      -> electron energy

        # T_g -> [K]  
        # T_e -> [eV] 

        # Clip negative values
        
        # y[:,xp.where(y <= 0.0)] = 0.0
        y[y <= 0.0] = 0.0

        n_g = y[:,iNg]    # [#/m^3]
        ne = y[:,iNe]     # [#/m^3]
        nion = y[:,iNion] # [#/m^3]
        T_e = y[:,iEe]    # [eV]
        # Ee = y[:,iEe] # [eV/m^3] 
        # Eh = y[:,iEh] # [eV/m^3] 

        p_0 = self.p_0

        # allocate arrays
        npop = xp.zeros((y.shape[0],self.Ns-2)) # ground state + excited levels
        dydt = xp.zeros((y.shape[0],self.Ns+1)) # ground state + excited levels + electrons + ions + Ee #+ Eh

        # npop = self.npop
        # dydt = self.dydt
        # dydt[:] = 0.0
        
        # dEhdt = xp.zeros((y.shape[0]))
        
        npop[:,0] = n_g
        npop[:,1:] = y[:,1:self.Ns-2]

        # Temperature of heavy species (from ideal gas law)
        #  Ideal gas law: p_0 = p_n + p_i + p_e
        T_g = (p_0/spc.k - ne * T_e/K_eV) / (xp.sum(npop, axis=1) + nion)   # [K]
 
        # T_g[T_g < 290.0] = 290.0 # eeeeeeeeee???????
        

        """
        Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
        """
        EEDF= self.MaxwellianDistribution_vec(self.eRange,T_e)
        AEDF= self.MaxwellianDistribution_vec(self.eRange,T_g*K_eV)
        EEDFnorm = self.trapz(EEDF,self.eRange, axis=0 )
        AEDFnorm = self.trapz(AEDF,self.eRange, axis=0 )
        # EEDF /= EEDFnorm
        # AEDF /= AEDFnorm        

        eVelTimesEEDF =  self.eVel*EEDF/EEDFnorm
        aVelTimesAEDF =  self.aVel*AEDF/AEDFnorm


        # tic = cpu_time.time()
        # for i in range(100):
        #     a = xp.trapz(AEDF,self.eRange, axis=0 )
        # print(f"CPU Time / timestep is {cpu_time.time() - tic} seconds.")

        # tic = cpu_time.time()
        # for i in range(100):
        #     b = self.trapz(AEDF,self.eRange, axis=0 )
        # print(f"CPU Time / timestep is {cpu_time.time() - tic} seconds.")
        
        # # print(np.shape(a),np.shape(b))
        # # print(np.array_equal(a,b) )    
        # print(np.allclose(a, b, atol=1e-100))

        # exit(-1) 

                
        # keylist = self.p.collDict.keys()
        
        ################## Elecrton impact Ionization ##################
        # M.T. !< Ionization of Ni due to electrom-atom collisions
        Qi_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)

        i = iNg # from ground state (BSR Data from LXCat)
        deltaIon = Eion - self.p.E_lvl[i]*cm_eV 
                                    
        Si = self.trapz(self.p.sigma_ionBSR*eVelTimesEEDF,self.eRange, axis=0 )        
        Qi = self.p.g_lvl[i]*Qi_factor*xp.exp(deltaIon/T_e)*Si # I need to check again this one.



        Rsi = n_g * ne * Si     # Electron impact ionization
        Rqi = nion * ne * ne * Qi 
        
        dydt[:,i] = dydt[:,i] - Rsi + Rqi 
        dydt[:,iNe] = dydt[:,iNe] + Rsi - Rqi # rate of change of ion number density
        dydt[:,iEe] = dydt[:,iEe] + deltaIon * (Rqi - Rsi) # rate of change of eletron energy

        
        for i in range(1,self.Nv-3):                           
            # isPrimed_lvl    
            deltaIon = Eion - self.p.E_lvl[i]*cm_eV # Which Eion should I use? There are two!                    
            
            Si = self.trapz(self.p.sigma_ij_Ion[i-1]*eVelTimesEEDF,self.eRange, axis=0 )
                        
            # sigma_ion = xp.interp(self.xi*T_e, self.eRange, self.p.sigma_ij_Ion[i])
            # Si = xp.sum(self.wi * sigma_ion *self.electronImpactIonRateIntegrand(T_e) * T_e)
                                     
            Qi = self.p.g_lvl[i]*Qi_factor*xp.exp(deltaIon/T_e)*Si # I need to check again this one.
       
            Rsi = npop[:,i]*ne*Si # Electron impact ionization
            Rqi = nion*ne*ne*Qi 
            
            dydt[:,i] = dydt[:,i] - Rsi + Rqi 
            dydt[:,iNe] = dydt[:,iNe] + Rsi - Rqi # rate of change of ion number density
            dydt[:,iEe] = dydt[:,iEe] + deltaIon * (Rqi - Rsi) # rate of change of eletron energy



        ################## Elecrton impact de/excitation ##################
        # From LXCat BSR data
        # for iCollTrans in self.p.CollTransitions_LXCat_BSR:
        for iter in range(self.p.NCollTrans_LXCat_BSR):
            i = self.p.CollTransition_ij_LXCat_BSR[iter] # Lower lever 
            j = self.p.CollTransition_ij_LXCat_BSR[iter + self.p.NCollTrans_LXCat_BSR] # Upper level

            # i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            # j = self.p.CollTransition_ij[iCollTrans,1] # Upper level     
            
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

            Cij = self.trapz(self.p.sigma_ij_Exc_BSR[iter]*eVelTimesEEDF,self.eRange, axis=0 )                        
            # Cij = self.trapz(self.p.sigma_ij_Exc_BSR[iCollTrans]*eVelTimesEEDF,self.eRange, axis=0 )                        

            # sigma_ij = xp.interp(self.xi*T_e ,self.p.collDict_list[iCollTrans][:,0],self.p.collDict_list[iCollTrans][:,1])
            # sigma_ij[xp.where(self.xi*T_e < eij)] = 0
            # Cij = xp.sum(self.wi * sigma_ij *self.electronImpactIonRateIntegrand(T_e) * T_e)            
            # print("diff = " , abs(Cij - Cij_2)/Cij*100)

            Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
            Rcij = Cij*npop[:,i]*ne
            Rfji = Fji*npop[:,j]*ne
            dydt[:,i] = dydt[:,i] - Rcij + Rfji 
            dydt[:,j] = dydt[:,j] + Rcij - Rfji 
            dydt[:,iEe] = dydt[:,iEe] + eij * (Rfji - Rcij) # rate of change of eletron energy
        
        
        for iCollTrans in self.p.CollTransitions_Rest: 
            i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.p.CollTransition_ij[iCollTrans,1] # Upper level                            
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
                          
            Cij = self.trapz(self.p.sigma_ij_Exc[iCollTrans]*eVelTimesEEDF,self.eRange, axis=0 )


            # sigma_ij = xp.interp(self.xi*T_e,self.eRange,self.p.sigma_ij_Exc[iCollTrans])
            # Cij_2 = xp.sum(self.wi * sigma_ij *self.electronImpactIonRateIntegrand(T_e) * T_e)            
            # print("diff = " , abs(Cij - Cij_2)/Cij*100)

            Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
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
            if (i == 0):  # For now, we only calculate the escape factors for the reasonance lines. 
                eta = escapeFactCalc_vec(npop[:,i],self.p.E_j[itrans],self.p.E_i[itrans],self.p.g_j[itrans],self.p.g_i[itrans],\
                                 self.p.A_ji[itrans],M_Ar,T_g,self.R,self.L) 
            else:
                eta=1.0


            
            Rspem =  npop[:,j]*self.p.A_ji[itrans]*eta        
            dydt[:,i] = dydt[:,i] + Rspem # radiative transitions into lower state
            dydt[:,j] = dydt[:,j] - Rspem # radiative transitions out of higher state
            # dydt[:,iEh] = dydt[:,iEh] - eij * Rspem  # Do I need to include that???
            # dEhdt = dEhdt - eij * Rspem


                
        ################## Atom impact Ionization ##################
        Wm_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)
        
        deltaIon = Eion - self.p.E_lvl[0]*cm_eV
        
        Vm = self.trapz(self.p.sigma_1a_ion*aVelTimesAEDF,self.eRange, axis=0 )        
        Wm = self.p.g_lvl[0]*Wm_factor*xp.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent

    
        Rvm = n_g * n_g * Vm
        Rwm = n_g * ne * nion * Wm  
        dydt[:,0] = dydt[:,0] - Rvm + Rwm  # remove particle in particular state due to ionization
        dydt[:,iNe] = dydt[:,iNe] + Rvm - Rwm # change number of ions 
        # dydt[:,iEh] = dydt[:,iEh] + deltaIon * (Rwm - Rvm) # rate of change of eletron energy
        # dEhdt = dEhdt + deltaIon * (Rwm - Rvm)


        for i in range(1,self.Nv-3):
            
            # Ionization due to atom impact from any level
            deltaIon = Eion - self.p.E_lvl[i]*cm_eV
                                 
            Vm = self.trapz(self.p.sigma_ia_ion[i-1]*aVelTimesAEDF,self.eRange, axis=0 )                        
            # sigma_ia_ion = xp.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ia_ion[i])
            # Vm = xp.sum(self.wi * sigma_ia_ion * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)
                        
            Wm = self.p.g_lvl[i]*Wm_factor*xp.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent

            Rvm = npop[:,i] * n_g * Vm
            Rwm = n_g * ne * nion * Wm 
                            
            dydt[:,i] = dydt[:,i] - Rvm + Rwm # remove particle in particular state due to ionization
            dydt[:,iNe] = dydt[:,iNe] + Rvm - Rwm # change number of ions 
            # dydt[:,iEh] = dydt[:,iEh] + deltaIon * (Rwm - Rvm) # rate of change of eletron energy
            # dEhdt = dEhdt + deltaIon * (Rwm - Rvm)
        

            
        ################## Atom impact de/excitation ##################


        i = self.p.i_Atom_ExcFromGround 
        for itrans in self.p.itrans_Atom_ExcFromGround:
            
            j = self.p.j_Atom_ExcFromGround[itrans]
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV


            Kij = self.trapz(self.p.sigma_ij_Atom_ExcFromGround[itrans]*aVelTimesAEDF,self.eRange, axis=0 )
            # sigma_ij_a = xp.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ij_Atom_ExcFromGround[itrans])
            # Kij_2 = xp.sum(self.wi * sigma_ij_a * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)  
            # print("diff = ", abs(Kij_2-Kij)/Kij*100)                    

            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            Rkij = npop[:,i] * n_g * Kij
            Rlji = npop[:,j] * n_g * Lji  
            dydt[:,i] = dydt[:,i] - Rkij + Rlji 
            dydt[:,j] = dydt[:,j] + Rkij - Rlji
            # dydt[iEh] = dydt[iEh] + eij * (Rlji - Rkij)
            # dEhdt = dEhdt + eij * (Rlji - Rkij)
            



        for itrans in self.p.itrans_Atom_Exc:
            i = self.p.i_Atom_Exc[itrans]
            j = self.p.j_Atom_Exc[itrans] 
            
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
            
            Kij = self.trapz(self.p.sigma_ij_Atom_Exc[itrans]*aVelTimesAEDF,self.eRange, axis=0 )
  
            # sigma_ij_a = xp.interp(self.xi*T_g*K_eV, self.eRange, self.p.sigma_ij_Atom_Exc[itrans])           
            # Kij = xp.sum(self.wi * sigma_ij_a * self.atomImpactIonRateIntegrand(T_g*K_eV) * T_g*K_eV)            
            
            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            Rkij = npop[:,i] * n_g * Kij
            Rlji = npop[:,j] * n_g * Lji  
            dydt[:,i] = dydt[:,i] - Rkij + Rlji 
            dydt[:,j] = dydt[:,j] + Rkij - Rlji
            # dydt[:,iEh] = dydt[:,iEh] + eij * (Rlji - Rkij)                
            # dEhdt = dEhdt + eij * (Rlji - Rkij)
            


        ################# Photorecombination/photoionization ##################
        nTrans = 5
        for i in range(0,nTrans): # We include also the photoionization from ground state which has a different cross section

            Ri = self.trapz(self.p.sigma_c_ion[i]*eVelTimesEEDF,self.eRange, axis=0 )
            Ri_prime = self.trapz(self.p.sigma_c_ion[i]*self.eRange*eVelTimesEEDF,self.eRange, axis=0 )

            # sigma_c_ion = xp.interp(self.xi*T_e, self.eRange, self.p.sigma_c_ion[i])
            # Ri = xp.sum(self.wi * sigma_c_ion *self.electronImpactIonRateIntegrand(T_e) * T_e)
            # Ri_prime = xp.sum(self.wi * sigma_c_ion * self.xi*T_e * self.electronImpactIonRateIntegrand(T_e) * T_e)

            Rri = ne * nion * Ri # What is the reverse process here?
            Rri_prime = ne * nion * Ri_prime
            
            dydt[:,i] = dydt[:,i] + Rri 
            dydt[:,iNe] = dydt[:,iNe] - Rri 
            dydt[:,iEe] = dydt[:,iEe] - Rri_prime # rate of change of eletron energy


        # Bremsstrahlung emission
        # Rbremsstrahlung = 1.42e-40 * parameters.Zeff**2 * xp.sqrt(T_e/K_eV) * ne * ne /spc.e  # [eV/m^3/s]
        
        # Energy transfer between electrons and heavy particles 

        # sigma_el_e1 = xp.interp(self.eRange,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # # sigma_el_e1[xp.where(sigma_el_e1 < 0)] = 0
        ken = self.trapz(self.p.sigma_el_e1*eVelTimesEEDF,self.eRange, axis=0 )
            
        # sigma_el_e1 = xp.interp(self.xi*T_e,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # sigma_el_e1[xp.where(sigma_el_e1 < 0)] = 0
        # ken = xp.sum(self.wi * sigma_el_e1 *self.electronImpactIonRateIntegrand(T_e) * T_e)

        
        # MeanThermalVelocity_e = xp.sqrt(8.0*spc.k*T_e/K_eV/xp.pi/spc.m_e)
        # Lambda_ei = 1.24e7 * xp.sqrt((T_e/K_eV)**3/ne) # Check the units !!!?????
        # MeanSigma_ei = 5.85e-10 * xp.log(Lambda_ei)/(T_e/K_eV)**2 # Check the units !!!?????
        # kei = MeanThermalVelocity_e * MeanSigma_ei
            
        # Rtransfer_n = 3.0 * spc.m_e * ne * n_g * (T_g*K_eV - T_e) * ken / M_Ar
        # Rtransfer_i = 3.0 * spc.m_e * ne * ne * (T_g*K_eV - T_e) * kei / (M_Ar - spc.m_e)
        
        # dydt[:,iEe] = dydt[:,iEe] #- Rbremsstrahlung + Rtransfer_n + Rtransfer_i
        # dydt[:,iEh] = dydt[:,iEh] - Rtransfer_n - Rtransfer_i
        
        # dEhdt = dEhdt - Rtransfer_n - Rtransfer_i


        dydt[:,iNion] = dydt[:,iNe] # These rates are always the same! 
                                    # I need to think how we can exploit this to make the computation faster. 
                                    # Especialy for the calculation of the jacobian



        return dydt

    #----------------------------------------------------------------------------------


    def trapz(self, y, x=None, dx=1.0, axis=-1):

        xp = self.xp_module

        # y = xp.asanyarray(y)        
        d = self.eRange_diff
        nd = y.ndim
        slice1 = [slice(None)]*nd
        slice2 = [slice(None)]*nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)

 
        #xp.linalg.multi_dot #NOTE(malamast): Check if it speeds up with multi_dot??        
        ret = xp.transpose(xp.dot(d.T, (y[tuple(slice1)] + y[tuple(slice2)])) )/ 2.0
        return ret.reshape(-1)

        # ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
        # return ret


