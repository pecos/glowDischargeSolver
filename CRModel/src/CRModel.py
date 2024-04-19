# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:04:49 2023

@author: Malamas Tsagkaridis
"""
import sys
import numpy as np
# import matplotlib.pyplot as plt
# import os
# import csv
# import scipy.io as sio
# import pandas as pd
import h5py as h5

import scipy.constants as spc
# from scipy.optimize import fsolve,least_squares,root
# from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
# from scipy.optimize._numdiff import _eps_for_method,approx_derivative
from scipy.interpolate import RegularGridInterpolator

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
# from GeneralFunctions import escapeFactCalc,escapeFactCalc_vec,CalcLineRadiationLosses

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

    def __init__(self, args, Ns, NT, Np, Pressure, GasTemperature, backgroundSpecieActivationFactor = 0):
        """Initializes storage and operators required for solve."""

        self.args       = args
        self.xp_module  = np
        
        # Input parameters
        self.p_0 = Pressure; self.T_g0 = GasTemperature
        self.Np = Np
        
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
        # self.R = 0.0129
        self.R = 0.05 # discharge radius (electrode diameter = 0.1 m)
        self.L = 0.02 # discharge length (gap width is 2 cm)
        # ICP torch
        # dischR = 0.015 #  radius of the nozle
        # scalingFactor = 1.0
        # self.R = 0.028/scalingFactor #  radius of the torch
        # self.L = self.R*2 # length 


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
        self.NeRange = 1000
        self.eRange = np.logspace(np.log10(1e-3),np.log10(300),self.NeRange,dtype=np.float64)  # [eV]
        self.eRange = self.eRange[:, np.newaxis] # I have added np.newaxis so that it works in both vectorized 
                                                 # and non-vectorised versions of the jaccobian calculation. 
                                                 # However, when I remove it, the non-vectrorized version runs faster. 
        self.eRange_diff = np.diff(self.eRange, axis=0) # Needed to optemize trapz intregration


        # self.ones_eRange = np.ones_like(self.eRange)
        self.ones_eRange = np.ones((self.NeRange),dtype=np.float64)

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

        self.npop = np.zeros((self.Np,self.Ns-2), dtype=np.float64) # ground state + excited levels
        self.dydt_saved = np.zeros((self.Np,self.Ns+1), dtype=np.float64) # ground state + excited levels + electrons + ions + Ee #+ Eh

        self.ElecrtonImpactIonizationRate = np.zeros((self.N_lvl,self.Np,2), dtype=np.float64)       
        self.ElecrtonImpactExcitationRate = np.zeros((self.p.NCollTrans,self.Np,2), dtype=np.float64)

        self.ElecrtonImpactExcitationRate_LXCat = np.zeros((self.p.NCollTrans_LXCat_BSR,self.Np,2), dtype=np.float64)

        # self.Aeff_ji = np.zeros((self.p.NRadTrans,self.Np), dtype=np.float64)

        self.AtomImpactIonizationRate = np.zeros((self.N_lvl,self.Np,2), dtype=np.float64) 

        self.AtomImpactExcitationGrRate = np.zeros((len(self.p.itrans_Atom_ExcFromGround),self.Np,2), dtype=np.float64)
        self.AtomImpactExcitationRate = np.zeros((len(self.p.itrans_Atom_Exc),self.Np,2), dtype=np.float64)
      
        nTrans = 5
        self.PhotorecombinationRate = np.zeros((nTrans,self.Np,2), dtype=np.float64) 

        self.RbremsstrahlungFactor = np.zeros((self.Np), dtype=np.float64) 
        self.Rtransfer_n_Factor = np.zeros((self.Np), dtype=np.float64) 
        self.Rtransfer_i_Factor = np.zeros((self.Np), dtype=np.float64) 

        #----------------------------------------------------------------------------------


        filename = "./CRModel/Data/EEDF/EEDF_BSR_2.h5"
        self.Te_eedf, self.EEDF_list, self.EEDFinterpolator = self.readEEDFFile(filename)




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



    def escapeFactCalc_vec(self,n_i,E_j,E_i,g_j,g_i,A_ji,Mspecies,T_g,R,L):
        # Calculations for escape factor
        xp = self.xp_module

        lambda_0 = spc.h*spc.c/((E_j-E_i)*cm_eV*spc.e) # wavelength of transition
        k0 = lambda_0**3*n_i*g_j*A_ji*Mspecies**0.5/(8*xp.pi*g_i*(2*spc.k*xp.pi*T_g)**0.5) # absorption coefficient at line center, for Doppler absorption

        q0 = R
        Lq = L/(2*q0)      
        
        # Compute escape factor using vectorized conditions    
        eta = xp.where(((k0 * (L / 2) > 1.0) & (k0 * q0 > 1.0)),
                    (2.0 / (xp.sqrt(xp.pi * xp.log(k0 * L / 2.0)) * k0 * L) / (2.0 * Lq**2 + 2.0)
                        + 1 / (xp.sqrt(xp.pi * xp.log(k0 * q0)) * k0 * 2.0 * q0) *
                        (Lq / (Lq**2 + 1.0) + xp.arctan(Lq))),
                    1.0)
        
        eta = xp.where(eta > 1.0,1.0, eta)   
        return eta




    def copy_operators_Host2Device(self, dev_id):
      
      if self.args.use_gpu==0:
        return
      
      with cp.cuda.Device(dev_id):     
        
        # self.U2             = cp.asarray(self.U2)
        # self.U1             = cp.asarray(self.U1)
        # self.U0             = cp.asarray(self.U0)

        self.eRange         = cp.asarray(self.eRange)
        self.eRange_diff    = cp.asarray(self.eRange_diff)
        self.ones_eRange    = cp.asarray(self.ones_eRange)


        self.eVel           = cp.asarray(self.eVel)
        self.aVel           = cp.asarray(self.aVel)

        # self.jac            = cp.asarray(self.jac)
        self.xi             = cp.asarray(self.xi)
        self.wi             = cp.asarray(self.wi)


        self.p.sigma_ij_Exc = cp.asarray(self.p.sigma_ij_Exc) 

        # for key in self.p.sigma_ij_Exc_BSR: 
            # self.p.sigma_ij_Exc_BSR[key] = cp.asarray(self.p.sigma_ij_Exc_BSR[key]) 
        self.p.sigma_ij_Exc_BSR = cp.asarray(self.p.sigma_ij_Exc_BSR) 

        for key in self.p.sigma_ij_Exc_Rest: 
            self.p.sigma_ij_Exc_Rest[key] = cp.asarray(self.p.sigma_ij_Exc_Rest[key])

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

        self.p.deltaIon = cp.asarray(self.p.deltaIon) 

        self.dydt_saved = cp.asarray(self.dydt_saved) 
        self.npop       = cp.asarray(self.npop) 

        self.ElecrtonImpactIonizationRate       = cp.asarray(self.ElecrtonImpactIonizationRate) 
        self.ElecrtonImpactExcitationRate       = cp.asarray(self.ElecrtonImpactExcitationRate) 

        self.ElecrtonImpactExcitationRate_LXCat = cp.asarray(self.ElecrtonImpactExcitationRate_LXCat) 
        # self.Aeff_ji                            = cp.asarray(self.Aeff_ji) 
        self.AtomImpactIonizationRate           = cp.asarray(self.AtomImpactIonizationRate) 
        self.AtomImpactExcitationGrRate         = cp.asarray(self.AtomImpactExcitationGrRate) 
        self.AtomImpactExcitationRate           = cp.asarray(self.AtomImpactExcitationRate) 

        self.PhotorecombinationRate             = cp.asarray(self.PhotorecombinationRate) 

        self.RbremsstrahlungFactor              = cp.asarray(self.RbremsstrahlungFactor) 
        self.Rtransfer_n_Factor                 = cp.asarray(self.Rtransfer_n_Factor) 
        self.Rtransfer_i_Factor                 = cp.asarray(self.Rtransfer_i_Factor) 

        
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
  



    def rxnSourceTermJac_vec(self, Uin):
        """
        Evaluates the Jacobian for backward Euler time marching.
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
    
        omega_U = xp.zeros((self.Ns+1,self.Ns+1,self.Np),dtype=xp.float64) # We have already calculated that. 

        self.rxnSourceTerm_UpdateTemperatureDependentPart_vec(Uin) # We have already called that. 
        omega = self.rxnSourceTerm_vec(Uin)
        # omega = self.dydt_saved
        Uin_perturbed = Uin.copy()
       
        # for i in range(0,self.Ns+1):
        for i in range(0,self.Ns):
            # Perturb the input at index i
            Uin_perturbed[:,i] += h[:,i]
            # self.rxnSourceTerm_UpdateTemperatureDependentPart_vec(Uin_perturbed)
            omega_perturbed = self.rxnSourceTerm_vec(Uin_perturbed)
            Uin_perturbed[:,i] = Uin[:,i]

            # Compute the partial derivative with respect to the i-th input using finite differences
            h_i = h[:,i]
            omega_U[:, i, :] = xp.transpose((omega_perturbed[:,:] - omega[:,:]) / h_i[:, xp.newaxis])
 
        # Treat perturbation of Temperature. Update the temperature dependent rates.
        i = self.Ns
        # Perturb the input at index i
        Uin_perturbed[:,i] += h[:,i]
        self.rxnSourceTerm_UpdateTemperatureDependentPart_vec(Uin_perturbed)
        omega_perturbed = self.rxnSourceTerm_vec(Uin_perturbed)
        Uin_perturbed[:,i] = Uin[:,i]

        # Compute the partial derivative with respect to the i-th input using finite differences
        h_i = h[:,i]
        omega_U[:, i, :] = xp.transpose((omega_perturbed[:,:] - omega[:,:]) / h_i[:, xp.newaxis])
  
            
        return omega,omega_U



    #----------------------------------------------------------------------------------

    def rxnSourceTerm_Update_vec(self,y):

        self.dydt_saved = self.rxnSourceTerm_vec(y)

        return

    #----------------------------------------------------------------------------------

              
    def rxnSourceTerm_UpdateTemperatureDependentPart_vec(self,y):

        """
        Computes the temperature dependent part of the rates.
    
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
        y[y < 0.0] = 0.0

        n_g = y[:,iNg]    # [#/m^3]
        ne = y[:,iNe]     # [#/m^3]
        nion = y[:,iNion] # [#/m^3]
        T_e = y[:,iEe]    # [eV]
        # Ee = y[:,iEe] # [eV/m^3] 
        # Eh = y[:,iEh] # [eV/m^3] 
        
        npop = y[:,:self.Ns-2]  # ground state + excited levels  
        npop[:,0] = n_g

        p_0 = self.p_0

        # allocate arrays        
        # dEhdt = xp.zeros((self.Np))


        # Temperature of heavy species (from ideal gas law)
        #  Ideal gas law: p_0 = p_n + p_i + p_e
        T_g = (p_0/spc.k - ne * T_e/K_eV) / (xp.sum(npop, axis=1) + nion)   # [K]
 
        # T_g[T_g < 290.0] = 290.0 # eeeeeeeeee???????
        # T_e = np.where(T_e < T_g*K_eV,T_g*K_eV, T_e)       

        """
        Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
        """  

        # EEDF= self.MaxwellianDistribution_vec(self.eRange,T_e) # np.shape(EEDF) -> (1000, 150)
        # EEDFnorm = self.trapz(EEDF, axis=0 )
        # eVelTimesEEDF =  self.eVel*EEDF/EEDFnorm 

        # Te_index = np.searchsorted(self.Te_eedf, T_e)
        # Te_index[T_e > self.Te_eedf[-1]] = len(self.Te_eedf)-1
        # EEDF = np.transpose(self.EEDF_list[Te_index])     
        # eVelTimesEEDF =  self.eVel*EEDF # np.shape(eVelTimesEEDF) -> (1000, 150)
        

        EEDF = np.zeros((self.NeRange,self.Np), dtype=np.float64)        
        for ip in range(self.Np): 
            points = np.column_stack((self.ones_eRange * T_e[ip], self.eRange[:,0]))
            EEDF[:,ip] = self.EEDFinterpolator(points)

        EEDF[EEDF<0.0] = 0.0
        eVelTimesEEDF =  self.eVel*EEDF # np.shape(eVelTimesEEDF) -> (1000, 150)


        
        AEDF= self.MaxwellianDistribution_vec(self.eRange,T_g*K_eV)
        AEDFnorm = self.trapz(AEDF, axis=0 )
        aVelTimesAEDF =  self.aVel*AEDF/AEDFnorm

          
        # i_mid = 6         
        # Te_index_tmp = np.searchsorted(self.Te_eedf, T_e[i_mid])
        # Te_index_tmp = min(len(self.Te_eedf)-1,Te_index_tmp)
        # EEDF_tmp = self.EEDF_list[Te_index_tmp]      

        # Te_tmp = T_e[i_mid]
        # EEDF_Maxwellian = 2*np.sqrt(self.eRange[:,0]/np.pi)*(Te_tmp)**(-1.5)*np.exp(-self.eRange[:,0]/(Te_tmp))

        # fig, ax = plt.subplots()
        # ax.set_title('EEDF ')
        # ax.set_xlabel('Electron Energy [eV]')
        # ax.set_ylabel('f($\epsilon$)  [eV^-1]')
        # ax.plot(self.eRange,EEDF_tmp,"*",label='Bolsig+')
        # # ax.plot(self.eRange[:,0], EEDF_bolsig[:,i_mid],"^",label='Bolsig+ 2')
        # ax.plot(self.eRange[:,0], EEDF_Maxwellian,"*",label='Maxwellian ')
        # ax.plot(self.eRange[:,0], EEDF[:,i_mid],"^",label='EEDF')
        # ax.semilogx()
        # ax.legend()
        # plt.show()
     
        # exit(-1)
     
              
        ################## Elecrton impact Ionization ##################
        # M.T. !< Ionization of Ni due to electrom-atom collisions
        Qi_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)
        for i in range(self.N_lvl):                           
            deltaIon = self.p.deltaIon[i]
            
            Si = self.trapz(self.p.sigma_ij_Ion[i]*eVelTimesEEDF, axis=0 )                                                       
            Qi = self.p.g_lvl[i]*Qi_factor*xp.exp(deltaIon/T_e)*Si # I need to check again this one.            

            self.ElecrtonImpactIonizationRate[i,:,0] = Si
            self.ElecrtonImpactIonizationRate[i,:,1] = Qi
            

        ################## Elecrton impact de/excitation ##################
        for iCollTrans in range(self.p.NCollTrans): 

            i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.p.CollTransition_ij[iCollTrans,1] # Upper level     
            
            eij = self.p.eij_CollTrans[iCollTrans]

            Cij = self.trapz(self.p.sigma_ij_Exc[iCollTrans]*eVelTimesEEDF, axis=0 )                        

            # Fji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/T_e)*Cij # superelastic collision by principle of detailed balance
            Fji = self.trapz(self.p.sigma_ij_deExc[iCollTrans]*eVelTimesEEDF, axis=0)                        

            self.ElecrtonImpactExcitationRate[iCollTrans,:,0] = Cij
            self.ElecrtonImpactExcitationRate[iCollTrans,:,1] = Fji


        # ################# Radiation processes ##################
        # for itrans in self.p.EmissionTransitions: 
        #     """
        #     Transition data for Ar I
        #     i -> lower level
        #     j -> upper level
        #     """     
        
        #     i = self.p.index_i_lvl[itrans]
        #     j = self.p.index_j_lvl[itrans]

        #     # eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

        #     # Calculations for escape factor
        #     if (i == 0):  # For now, we only calculate the escape factors for the reasonance lines. 
        #         eta = self.escapeFactCalc_vec(npop[:,i],self.p.E_j[itrans],self.p.E_i[itrans],self.p.g_j[itrans],self.p.g_i[itrans],\
        #                          self.p.A_ji[itrans],M_Ar,T_g,self.R,self.L) 
        #     else:
        #         eta=1.0

        #     self.Aeff_ji[itrans,:] = self.p.A_ji[itrans]*eta 

                
        ################## Atom impact Ionization ##################
        Wm_factor = 1.0/2.0/g_ion*(parameters.lambda_factor/T_e)**(1.5)
        for i in range(self.N_lvl):
            
            # Ionization due to atom impact from any level
            deltaIon = self.p.deltaIon[i]
                                 
            Vm = self.trapz(self.p.sigma_ia_ion[i]*aVelTimesAEDF, axis=0 )   
            Wm = self.p.g_lvl[i]*Wm_factor*xp.exp(deltaIon/(T_g*K_eV))*Vm # Check that I use T_g in the exponent
                
            self.AtomImpactIonizationRate[i,:,0] = Vm
            self.AtomImpactIonizationRate[i,:,1] = Wm
        

            
        ################## Atom impact de/excitation ##################

        i = self.p.i_Atom_ExcFromGround         
        for itrans in self.p.itrans_Atom_ExcFromGround:
            j = self.p.j_Atom_ExcFromGround[itrans]

            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

            Kij = self.trapz(self.p.sigma_ij_Atom_ExcFromGround[itrans]*aVelTimesAEDF, axis=0 )            
            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            self.AtomImpactExcitationGrRate[itrans,:,0] = Kij
            self.AtomImpactExcitationGrRate[itrans,:,1] = Lji

            
        for itrans in self.p.itrans_Atom_Exc:
            i = self.p.i_Atom_Exc[itrans]
            j = self.p.j_Atom_Exc[itrans] 
            
            eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV
            
            Kij = self.trapz(self.p.sigma_ij_Atom_Exc[itrans]*aVelTimesAEDF, axis=0 )             
            Lji = self.p.g_lvl[i]/self.p.g_lvl[j]*xp.exp(eij/(T_g*K_eV))*Kij # Check that I use T_g in the exponent

            self.AtomImpactExcitationRate[itrans,:,0] = Kij
            self.AtomImpactExcitationRate[itrans,:,1] = Lji
            
 

        ################# Photorecombination/photoionization ##################
        nTrans = 5
        for i in range(0,nTrans): # We include also the photoionization from ground state which has a different cross section

            Ri = self.trapz(self.p.sigma_c_ion[i]*eVelTimesEEDF, axis=0 )
            Ri_prime = self.trapz(self.p.sigma_c_ion[i]*self.eRange*eVelTimesEEDF, axis=0 )

            self.PhotorecombinationRate[i,:,0] = Ri
            self.PhotorecombinationRate[i,:,1] = Ri_prime


        # # ################## Bremsstrahlung emission ##################
        # self.RbremsstrahlungFactor = 1.42e-40 * parameters.Zeff**2 * xp.sqrt(T_e/K_eV) /spc.e  # [eV/m^3/s * m^3 * m^3]
        
        # # ################## Energy transfer between electrons and heavy particles ################## 
        # ken = self.trapz(self.p.sigma_el_e1*eVelTimesEEDF, axis=0 )
            
        # # sigma_el_e1 = xp.interp(self.xi*T_e,parameters.eRange_elastic_e1,parameters.sigma_elastic_e1)*1e-20
        # # sigma_el_e1[xp.where(sigma_el_e1 < 0)] = 0
        # # ken = xp.sum(self.wi * sigma_el_e1 *self.electronImpactIonRateIntegrand(T_e) * T_e)

        # MeanThermalVelocity_e = xp.sqrt(8.0*spc.k*T_e/K_eV/xp.pi/spc.m_e)
        # Lambda_ei = 1.24e7 * xp.sqrt((T_e/K_eV)**3/ne) # Check the units !!!?????
        # MeanSigma_ei = 5.85e-10 * xp.log(Lambda_ei)/(T_e/K_eV)**2 # Check the units !!!?????
        # kei = MeanThermalVelocity_e * MeanSigma_ei
        
        # self.Rtransfer_n_Factor = 3.0 * spc.m_e / M_Ar * ken
        # self.Rtransfer_i_Factor = 3.0 * spc.m_e / (M_Ar - spc.m_e) * kei    


        return

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
        y[y < 0.0] = 0.0

        n_g = y[:,iNg]    # [#/m^3]
        ne = y[:,iNe]     # [#/m^3]
        nion = y[:,iNion] # [#/m^3]
        T_e = y[:,iEe]    # [eV]
        # Ee = y[:,iEe] # [eV/m^3] 
        # Eh = y[:,iEh] # [eV/m^3] 

        npop = y[:,:self.Ns-2]  # ground state + excited levels
        npop[:,0] = n_g        

        p_0 = self.p_0

        # allocate arrays
        dydt = xp.zeros((self.Np,self.Ns+1)) # ground state + excited levels + electrons + ions + Ee #+ Eh                        
        # dEhdt = xp.zeros((self.Np))


        # Temperature of heavy species (from ideal gas law)
        #  Ideal gas law: p_0 = p_n + p_i + p_e
        T_g = (p_0/spc.k - ne * T_e/K_eV) / (xp.sum(npop, axis=1) + nion)   # [K]
 
        # T_g[T_g < 290.0] = 290.0 # eeeeeeeeee???????
        

        """
        Compute Electron Energy Distribution Function (EEDF) based on a Maxwellian distribution:
        """
        # EEDF= self.MaxwellianDistribution_vec(self.eRange,T_e)
        # AEDF= self.MaxwellianDistribution_vec(self.eRange,T_g*K_eV)
        # EEDFnorm = self.trapz(EEDF,self.eRange, axis=0 )
        # AEDFnorm = self.trapz(AEDF,self.eRange, axis=0 )
        # # EEDF /= EEDFnorm
        # # AEDF /= AEDFnorm        

        # eVelTimesEEDF =  self.eVel*EEDF/EEDFnorm
        # aVelTimesAEDF =  self.aVel*AEDF/AEDFnorm


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
        for i in range(self.N_lvl):                           
            deltaIon = self.p.deltaIon[i]        
                        
            Si = self.ElecrtonImpactIonizationRate[i,:,0]
            Qi = self.ElecrtonImpactIonizationRate[i,:,1]

            Rsi = npop[:,i]*ne*Si # Electron impact ionization
            Rqi = nion*ne*ne*Qi 
            
            dydt[:,i] += - Rsi + Rqi 
            dydt[:,iNe] += + Rsi - Rqi # rate of change of ion number density
            dydt[:,iEe] += deltaIon * (Rqi - Rsi) # rate of change of eletron energy


        ################## Elecrton impact de/excitation ##################

        for iCollTrans in range(self.p.NCollTrans):
               
            i = self.p.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.p.CollTransition_ij[iCollTrans,1] # Upper level   
            eij = self.p.eij_CollTrans[iCollTrans]

            Cij = self.ElecrtonImpactExcitationRate[iCollTrans,:,0] 
            Fji = self.ElecrtonImpactExcitationRate[iCollTrans,:,1] 
            
            Rcij = Cij*npop[:,i]*ne
            Rfji = Fji*npop[:,j]*ne
            dydt[:,i] += - Rcij + Rfji 
            dydt[:,j] += + Rcij - Rfji 
            dydt[:,iEe] +=  eij * (Rfji - Rcij) # rate of change of eletron energy
        

 
        ################# Radiation processes ##################
        for itrans in self.p.EmissionTransitions: 
            """
            Transition data for Ar I
            i -> lower level
            j -> upper level
            """     
        
            i = self.p.index_i_lvl[itrans]
            j = self.p.index_j_lvl[itrans]

            # eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

            
            # Calculations for escape factor
            if (i == 0):  # For now, we only calculate the escape factors for the reasonance lines. 
                eta = self.escapeFactCalc_vec(npop[:,i],self.p.E_j[itrans],self.p.E_i[itrans],\
                                                self.p.g_j[itrans],self.p.g_i[itrans],\
                                            self.p.A_ji[itrans],M_Ar,T_g,self.R,self.L) 
            else:
                eta=1.0
                
            
            Rspem =  npop[:,j]*self.p.A_ji[itrans]*eta        
            dydt[:,i] += + Rspem # radiative transitions into lower state
            dydt[:,j] += - Rspem # radiative transitions out of higher state
            # dydt[:,iEh] += - eij * Rspem  # Do I need to include that???
            # dEhdt += - eij * Rspem

    
        ################## Atom impact Ionization ##################

        for i in range(self.N_lvl):
            
            # Ionization due to atom impact from any level
            # deltaIon = self.p.deltaIon[i]
                                 
            Vm = self.AtomImpactIonizationRate[i,:,0]
            Wm = self.AtomImpactIonizationRate[i,:,1]

            Rvm = npop[:,i] * n_g * Vm
            Rwm = n_g * ne * nion * Wm 
                            
            dydt[:,i] +=  - Rvm + Rwm # remove particle in particular state due to ionization
            dydt[:,iNe] += + Rvm - Rwm # change number of ions 
            # dydt[:,iEh] += + deltaIon * (Rwm - Rvm) # rate of change of eletron energy
            # dEhdt += + deltaIon * (Rwm - Rvm)
        

            
        ################## Atom impact de/excitation ##################
        i = self.p.i_Atom_ExcFromGround 
        for itrans in self.p.itrans_Atom_ExcFromGround:
            
            j = self.p.j_Atom_ExcFromGround[itrans]
            # eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

            Kij = self.AtomImpactExcitationGrRate[itrans,:,0] 
            Lji = self.AtomImpactExcitationGrRate[itrans,:,1] 


            Rkij = npop[:,i] * n_g * Kij
            Rlji = npop[:,j] * n_g * Lji  
            dydt[:,i] += - Rkij + Rlji 
            dydt[:,j] += + Rkij - Rlji
            # dydt[iEh] += eij * (Rlji - Rkij)
            # dEhdt += eij * (Rlji - Rkij)
            



        for itrans in self.p.itrans_Atom_Exc:
            i = self.p.i_Atom_Exc[itrans]
            j = self.p.j_Atom_Exc[itrans] 
            
            # eij = (self.p.E_lvl[j] - self.p.E_lvl[i])*cm_eV

            Kij = self.AtomImpactExcitationRate[itrans,:,0] = Kij
            Lji = self.AtomImpactExcitationRate[itrans,:,1] = Lji
            
            Rkij = npop[:,i] * n_g * Kij
            Rlji = npop[:,j] * n_g * Lji  
            dydt[:,i] +=  - Rkij + Rlji 
            dydt[:,j] +=  + Rkij - Rlji
            # dydt[:,iEh] += eij * (Rlji - Rkij)                
            # dEhdt += eij * (Rlji - Rkij)
            


        ################# Photorecombination/photoionization ##################
        nTrans = 5
        for i in range(0,nTrans): # We include also the photoionization from ground state which has a different cross section

            Ri = self.PhotorecombinationRate[i,:,0] 
            Ri_prime = self.PhotorecombinationRate[i,:,1]

            Rri = ne * nion * Ri # What is the reverse process here?
            Rri_prime = ne * nion * Ri_prime
            
            dydt[:,i] +=  Rri 
            dydt[:,iNe] += - Rri 
            dydt[:,iEe] += - Rri_prime # rate of change of eletron energy



        # ################## Bremsstrahlung emission ##################
        # dydt[:,iEe] -= self.RbremsstrahlungFactor * ne * ne  # [eV/m^3/s]
        # # Rbremsstrahlung = 1.42e-40 * parameters.Zeff**2 * xp.sqrt(T_e/K_eV) * ne * ne /spc.e  # [eV/m^3/s]
        # # dydt[:,iEe] += - Rbremsstrahlung

        
        # ################## Energy transfer between electrons and heavy particles ################## 
        # Rtransfer_n = self.Rtransfer_n_Factor * ne * n_g * (T_g*K_eV - T_e)
        # Rtransfer_i = self.Rtransfer_i_Factor * ne * ne * (T_g*K_eV - T_e)
        
        # dydt[:,iEe] += + Rtransfer_n + Rtransfer_i
        # # dydt[:,iEh] += - Rtransfer_n - Rtransfer_i
        
        # # dEhdt += - Rtransfer_n - Rtransfer_i


        dydt[:,iNion] = dydt[:,iNe] # These rates are always the same! 
                                    # I need to think how we can exploit this to make the computation faster. 
                                    # Especialy for the calculation of the jacobian


        return dydt


    #----------------------------------------------------------------------------------

    def trapz(self, y, x=None, axis=-1):

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


    #----------------------------------------------------------------------------------

            
    def readEEDFFile(self,filename):
            
            

        EEDF_Bolsig = []
        df = h5.File(filename, 'r')
        Te_eedf = []; 
        Te_eedf_2 = df["temperature"][:]*K_eV; 
        

        print("Reading EEDF ...")
        for i in range(len(df)-1):
            key = f'EEDF_{i}'
            dset = df[key]

            data = dset[()]  # This retrieves all the data from the dataset

            if (not np.array_equal(dset[:,0], data[:,0]) or not np.array_equal(dset[:,1], data[:,1])):
                print("Problem with converting dataset to data.")
                exit(-1)


            EEDF_Bolsig.append(data)

            # name0 = dset.attrs['name0'] 
            # name1 = dset.attrs['name1'] 
            # unit0 = dset.attrs['unit0']
            # unit1 = dset.attrs['unit1']           
            index_eedf = dset.attrs['index'] 
            
            if index_eedf != i: 
                print("Something is wrong with the reading of the EEDFs.")
                exit(-1)
            
            Te_tmp = dset.attrs['Electron Temperature']            
            Te_eedf.append(Te_tmp*K_eV)
            
            # self.eRange = np.logspace(np.log10(1e-3),np.log10(300),1000)  # [eV]

            # eRange = EEDF_Bolsig[index_eedf][:,0]
            # print("Reading EEDF ",index_eedf, "Te = ",round(Te_tmp*K_eV,2), "e energy range = " , np.min(eRange), np.max(eRange))

            # print(key, i)


        Te_eedf = np.asarray(Te_eedf)
        


        if len(EEDF_Bolsig) != len(df)-1:
            print("Number of EEDFs is not correct.")
            exit(-1)
        else:
            print("Number of EEDFs read is = ", len(EEDF_Bolsig))


        # ic = 1
        # Te_index = np.argmin(np.abs(Te[ic] - 5.0))
        # Te_index2 = np.argmin(np.abs(Te2[ic] - 5.0))
        # if Te_index != Te_index2: 
        #     print("Problem with reading of EEDFs.")
        #     exit(-1)

        # Te_index = np.argmin(np.abs(Te*K_eV - 5.0))
        # print('energy = ', nominalOutput.outputs[3].data[Te_index,1])

        # Te_index2 = np.searchsorted(Te, 5.0/K_eV)
        # print(Te_index, Te_index2)

        # eRange = EEDF_list[ic][Te_index][:,0] 
        # EEDF = EEDF_list[ic][Te_index][:,1]*np.sqrt(eRange)
        # EEDF_Maxwellian = 2*np.sqrt(eRange/np.pi)*(Te[ic][Te_index])**(-1.5)*np.exp(-eRange/(Te[ic][Te_index]))


        EEDF_list = np.zeros((len(EEDF_Bolsig), len(self.eRange)), dtype=np.float64) 



        for Te_index in range(len(EEDF_Bolsig)):

            eRange_Bolsig = EEDF_Bolsig[Te_index][:,0] 
            EEDF_tmp = EEDF_Bolsig[Te_index][:,1]*np.sqrt(eRange_Bolsig)


            EEDF_list[Te_index,:] = np.interp(self.eRange[:,0],eRange_Bolsig,EEDF_tmp, left=EEDF_tmp[0], right=0.0)  

            EEDFnorm = np.trapz(EEDF_list[Te_index,:],self.eRange[:,0])
            

            EEDF_list[Te_index,:] /= EEDFnorm
            

            EEDF_Maxwellian = 2*np.sqrt(eRange_Bolsig/np.pi)*(Te_eedf[Te_index])**(-1.5)*np.exp(-eRange_Bolsig/(Te_eedf[Te_index]))
            # EEDF_Maxwellian_2= self.MaxwellianDistribution_vec(self.eRange, Te_eedf[Te_index])



        EEDFinterpolator = RegularGridInterpolator((Te_eedf, self.eRange[:,0]), EEDF_list, bounds_error=False, fill_value=None)

        # T_e = 4.0 
        # points = np.column_stack((np.ones_like(self.eRange) * T_e, self.eRange))
        # interpolated_EEDF = EEDFinterpolator(points)
        
    
        # Te_index = np.searchsorted(Te_eedf, T_e)
        # EEDF = np.transpose(EEDF_list[Te_index])    
        # EEDF_Maxwellian = 2*np.sqrt(self.eRange/np.pi)*(T_e)**(-1.5)*np.exp(-self.eRange/T_e)
        
        # fig, ax = plt.subplots()
        # ax.set_title('EEDF at Te = ' + str(round(Te_eedf[Te_index],2)))
        # ax.set_xlabel('Electron Energy [eV]')
        # # ax.set_ylabel('f($\epsilon$)  [eV^-3/2]')
        # ax.set_ylabel('f($\epsilon$)  [eV^-1]')

        # ax.plot(self.eRange,EEDF,".", label='Bolsig+ ')
        # ax.plot(self.eRange, EEDF_Maxwellian,".",label='Maxwellian 3')
        # ax.plot(self.eRange,interpolated_EEDF,"r.", label='interpolated ')
                
        # # ax.semilogy()
        # ax.legend()
        # plt.show()         





        return Te_eedf, EEDF_list, EEDFinterpolator
            
                
        

