
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:19:00 2023

@author: Malamas Tsagkaridis
"""
from dataclasses import dataclass
import matplotlib.pyplot as plt


from Constants import *
from read_Data import NIST_read_ArI, NIST_read_ArII, read_ArI_transitions,LXCat_read
from crossSections import multipleCrossSections, typeDictS2I, typeDictI2S
from ExcitationCrossSections import CharacteriseTransitions

#----------------------------------------------------------------------------------

sigma_factor_AtomExc = 4*np.pi*a0_H**2*Eion_H**2 * M_Ar/M_H
mass_factor = 2*spc.m_e/(M_Ar + spc.m_e)

mu_ei = (M_Ar - spc.m_e)*spc.m_e/M_Ar # Reduced mass
lambda_factor = spc.h**2/(2.0*np.pi*spc.m_e*spc.k)*K_eV
# lambda_factor = spc.h**2/(2.0*np.pi*mu_ei*spc.k)*K_eV


# Atom Impact Excitation
fij = [6.78e-2, 2.56e-1, 1.0e-3, 3.4e-2, 9.51e-2, 1.8e-2, 7.94e-2 ] # absorption oscillator strength of allowed transitions
atomImpactExcitationTransitionsFromGroundSate = ["Ar(4s[3/2]1)", "Ar(4s'[1/2]1)", "Ar(3d[1/2]1)", "Ar(5s[3/2]1)", "Ar(3d[3/2]1)", "Ar(5s'[1/2]1)", "Ar(3d'[3/2]1)"]

atomImpactExcitationTransitions_i = ["Ar(4s[3/2]2)","Ar(4s[3/2]2)","Ar(4s[3/2]2)","Ar(4s[3/2]1)","Ar(4s[3/2]1)","Ar(4s'[1/2]0)"]
atomImpactExcitationTransitions_j = ["Ar(4s[3/2]1)","Ar(4s'[1/2]0)","Ar(4s'[1/2]1)","Ar(4s'[1/2]0)","Ar(4s'[1/2]1)","Ar(4s'[1/2]1)"]
beta_ij = [1.79e-24, 4.8e-26, 4.8e-26, 4.8e-26, 4.8e-26, 1.79e-24]

# photoionization
gamma_i = [0.0, 0.0763, 0.0458, 0.0305, 0.0915]
photoionization_factor = 1.0/(2.0*spc.m_e*spc.c**2)

Zeff = np.sqrt(1.67) # effective charge
gauntFactor_FreeFree = 1.0 

# electron neutral elastic cross sections
sigma_elastic_e1 =  \
    [4.0231, 2.3425, 1.5247, 1.0261, 0.6998, 0.3291, 0.1599, 0.1198, 0.0983, \
    0.0917, 0.0969, 0.1316, 0.1573, 0.1724, 0.3396, 0.5349, 0.7381, 0.9318, \
    1.1139, 1.2835, 1.4280, 2.1438, 2.8318, 4.4046, 6.3502, 8.6317, 14.6559, 17.8325]
eRange_elastic_e1 = \
    [0.01, 0.03, 0.05, 0.07, 0.09, 0.13, 0.17, 0.19, 0.21, 0.23, \
    0.25, 0.29,  0.31, 0.32, 0.41, 0.51, 0.61, 0.71, 0.81, \
    0.91, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00] 

# Should I use the Bohr radius of Ar or H?
sigma_factor_VanRegemorter = 8*np.pi/np.sqrt(3)*np.pi*a0_H**2*Eion_H**2 
gauntFactor_BoundBound = 1.0  # How can I calculate this one?
OscillatorStrength_VR = 0.001 # How can I calculate this one?

sigma_factor_Drawin = 4*np.pi*a0_H**2

# Cross sections for electron impact ionization from the ground state (BSR data 2017)
# SPECIES: e / Ar
# PROCESS: E + Ar -> E + E + Ar+, Ionization
# PARAM.:  E = 15.76 eV
IonizationBSR = np.zeros((25,2))

IonizationBSR[:,0] =  \
    [1.576E+01, 1.80E+01, 1.90E+01, 2.00E+01, 2.10E+01, 2.20E+01, 2.30E+01, 2.40E+01, \
     2.50E+01, 3.00E+01, 4.00E+01, 5.00E+01, 6.00E+01, 7.00E+01, 8.00E+01, 9.00E+01, \
     1.00E+02, 1.25E+02, 1.50E+02, 1.75E+02, 2.00E+02, 2.25E+02, 2.50E+02, 2.75E+02, 3.00E+02]

IonizationBSR[:,1] =  \
    [0.0000E+00, 1.2772E-21, 2.0747E-21, 3.2199E-21, 4.4087E-21, 5.4052E-21, 6.6720E-21, 7.6630E-21, \
     8.7000E-21, 1.2251E-20, 1.7256E-20, 1.9517E-20, 2.1179E-20, 2.2269E-20, 2.2834E-20, 2.3777E-20, \
     2.4576E-20, 2.4366E-20, 2.3142E-20, 2.1902E-20, 2.0735E-20, 1.9672E-20, 1.8712E-20, 1.7691E-20, 1.7249E-20]

# Electron impact ionization (H. Deutsch et al, 2003)

@dataclass
class AngularMomentumParam:
    a: float; b: float; c: float; d: float

s = AngularMomentumParam(1.06,0.23,1.0,1.1)
p = AngularMomentumParam(2.0,1.0,1.0,1.0)
d = AngularMomentumParam(3.0/2.0,3.0,2.0/3.0,1.0)
f = AngularMomentumParam(3.0/2.0,1.0,2.0/3.0,1.0)
g = AngularMomentumParam(3.0/2.0,1.0,2.0/3.0,1.0) # There were no data for g so I used f's
h = AngularMomentumParam(3.0/2.0,1.0,2.0/3.0,1.0) # There were no data for h so I used f's

ParamAngMom = {}
ParamAngMom['s'] = s ; ParamAngMom['p'] = p ; ParamAngMom['d'] = d ; ParamAngMom['f'] = f ; ParamAngMom['g'] = g ; ParamAngMom['h'] = h
 
 
Radius_nl = {}
Radius_nl['4s'] =  2.49E-10; Radius_nl['5s'] =  6.35E-10; Radius_nl['6s'] =  1.26E-9; Radius_nl['7s'] =  2.08E-9; 
Radius_nl['4p'] =  3.40E-10; Radius_nl['5p'] =  8.15E-10; Radius_nl['6p'] =  1.52E-9; Radius_nl['7p'] =  2.35E-9; 
Radius_nl['3d'] =  4.36E-10; 
Radius_nl['4d'] =  1.05E-9;  Radius_nl['5d'] =  1.84E-9;  Radius_nl['6d'] =  2.84E-9; Radius_nl['7d'] =  3.89E-9; 
# No data were found for the cases below.
Radius_nl['8s'] = 4E-9; Radius_nl['9s'] = 4E-9; Radius_nl['10s'] = 4E-9; 
Radius_nl['8p'] = 4E-9; Radius_nl['9p'] = 4E-9; 
Radius_nl['8d'] = 4E-9; 
Radius_nl['4f'] = 4E-9; Radius_nl['5f'] = 4E-9; Radius_nl['6f'] = 4E-9; Radius_nl['7f'] = 4E-9; 
Radius_nl['5g'] = 4E-9; Radius_nl['6g'] = 4E-9; Radius_nl['7g'] = 4E-9; 
Radius_nl['7h'] = 4E-9; 

# H. Deutsch et al./International Journal of Mass Spectrometry 197 (2000) 37–69
WeightingFactor_g_nl = {} 
WeightingFactor_g_nl['1s'] = 38.20; WeightingFactor_g_nl['2s'] = 12.0; WeightingFactor_g_nl['3s'] = 9.80; 
WeightingFactor_g_nl['4s'] = 7.40;  WeightingFactor_g_nl['5s'] = 6.35; WeightingFactor_g_nl['6s'] = 5.40

WeightingFactor_g_nl['2p'] = 32.50; WeightingFactor_g_nl['3p'] = 31.50; WeightingFactor_g_nl['4p'] = 31.00; 
WeightingFactor_g_nl['5p'] = 30.50; WeightingFactor_g_nl['6p'] = 30.00; 

WeightingFactor_g_nl['3d'] = 13.60; WeightingFactor_g_nl['4d'] = 11.20; 
WeightingFactor_g_nl['5d'] = 8.85; WeightingFactor_g_nl['6d'] = 6.50; 

WeightingFactor_g_nl['4f'] = 20.00; WeightingFactor_g_nl['5f'] = 1.00; 

# No data were found for the cases below.
WeightingFactor_g_nl['7s'] = 4.00;  WeightingFactor_g_nl['8s'] = 3.00; WeightingFactor_g_nl['9s'] = 1.00; WeightingFactor_g_nl['10s'] = 1.00; 
WeightingFactor_g_nl['7p'] = 29.00; WeightingFactor_g_nl['8p'] = 20.00;  WeightingFactor_g_nl['9p'] = 10.00;   
WeightingFactor_g_nl['7d'] = 4.00; WeightingFactor_g_nl['8d'] = 2.50; 

WeightingFactor_g_nl['6f'] = 1.00; WeightingFactor_g_nl['7f'] = 1.00;
WeightingFactor_g_nl['5g'] = 1.00; WeightingFactor_g_nl['6g'] = 1.00; WeightingFactor_g_nl['7g'] = 1.00; 
WeightingFactor_g_nl['7h'] = 1.00; 

xi_nl = 1.0 # Number of electrons in the sub-shell

#----------------------------------------------------------------------------------



class modelParameters:
    """Class providing model parameters."""

    def __init__(self, Ns):
        """Set model parameter values.  
        """
        # self.Ns = Ns # number of species

        homeDir = os.getcwd()
        
        #----------------------------------------------------------------------------------
        """
        Read in NIST data for states of Ar I  
        """
        
        p = NIST_read_ArI('./CRModel/Data','ArI-States.csv', Ns-4)
        os.chdir(homeDir)

        (self.Configuration_lvl, self.Term_lvl, self.J_lvl, 
         self.E_lvl, self.g_lvl, 
         self.Racah_lvl, self.DictRacah_lvl, 
         self.SubShell_lvl, 
         self.isPrimed_lvl, 
         self.PrinQuantNum_lvl, 
         self.AngMomQuantNum_lvl, 
         self.Parity_lvl, 
         self.Spin_lvl
        ) = p

        self.N_lvl = len(self.E_lvl)
        print("No. of excited levels of A I considered (including ground state) =",self.N_lvl)



        for i in range(self.N_lvl):    
            deltaIon = Eion - self.E_lvl[i]*cm_eV 
            if deltaIon < 0.0:
                print(" Some levels have larger energy than the Ionization Energie. \
                      Fix electron-impact ionization process to proceed.")
                raise SystemExit(0) 

        self.deltaIon = np.zeros(self.N_lvl, dtype=np.float64)
        for i in range(self.N_lvl):    
            if (not self.isPrimed_lvl[i]): # There are two ion ground levels
                Eion_Ar = Eion_Ar_1
            else:
                Eion_Ar = Eion_Ar_2
            deltaIon = Eion_Ar - self.E_lvl[i]*cm_eV
            self.deltaIon[i] = deltaIon 


        self.E_lvl_ArIon    = 15.7596119
        self.E_lvl_Ar2Ion   = 14.501
        self.E_lvl_Ar2m     = 11.564763 # 11.577-0.012237
        self.Ar2Ion_DissEn  = 1.2586119
        self.Ar2m_DissEn    = 0.012237

        # Dissociation energy of Ar2* in ground state is  0.012237 eV
        # Dissociation energy of Ar2+ is 1.3144 eV

            

    
        #----------------------------------------------------------------------------------
        """
        Read in considered states of Ar II and their properties
        """

        p = NIST_read_ArII('./CRModel/Data','ArII-States.csv')
        os.chdir(homeDir)
        (self.Configuration_lvl_ArII, self.Term_lvl_ArII, self.J_lvl_ArII, 
         self.E_lvl_ArII, self.g_lvl_ArII
        ) = p


        self.N_lvl_ArII = len(self.E_lvl_ArII)
        print("No. of excited levels of A II considered (including ground state) =",self.N_lvl_ArII)

        #----------------------------------------------------------------------------------


        """
        Read in transition data for Ar I
        i -> lower level
        j -> upper level
        """

        levels = (self.Configuration_lvl, self.Term_lvl, self.J_lvl, 
                  self.E_lvl, self.g_lvl, self.Racah_lvl,  self.DictRacah_lvl) 

        p = read_ArI_transitions('./CRModel/Data','ArI-Transitions.csv',*levels)
        os.chdir(homeDir)


        (self.A_ji, self.f_ji, self.E_i, self.E_j, self.g_i, self.g_j, 
        self.Configuration_i, self.Term_i, self.J_i,  
        self.Configuration_j, self.Term_j, self.J_j, 
        self.Racah_i, DictRacah_i, self.Racah_j, DictRacah_j, self.Source_ji) = p


        # DictRacah_i, DictRacah_j give the wrong index because the keys in these 
        # dictionaries are multiply difined as we loop over all radiative transitions

        # Find for which transitions we include the associated levels 
        NRadTrans_tot = len(self.Configuration_i)

        self.EmissionTransitions = []
        icount = 0
        for itrans in range(NRadTrans_tot): 
            if (self.Racah_i[itrans] in self.DictRacah_lvl.keys()) and (self.Racah_j[itrans] in self.DictRacah_lvl.keys()):
                # if (Racah_i[itrans] == 'Ar(3d[3/2]1)' or Racah_j[itrans] == 'Ar(3d[3/2]1)' ):
                icount = icount +1
                # print(itrans,icount-1,'skata')
                self.EmissionTransitions.append(itrans)
                # print(itrans,Racah_i[itrans],Racah_j[itrans])    


        self.NRadTrans = len(self.EmissionTransitions)

        # Discard extra values from radiative transitions arrays
        EmissionTransitions_np = np.array(self.EmissionTransitions)
        
        self.A_ji = self.A_ji[EmissionTransitions_np]       
        self.f_ji = self.f_ji[EmissionTransitions_np] 
        
        self.E_i = self.E_i[EmissionTransitions_np] 
        self.E_j = self.E_j[EmissionTransitions_np] 
        self.g_i = self.g_i[EmissionTransitions_np] 
        self.g_j = self.g_j[EmissionTransitions_np] 

        self.Configuration_i = [self.Configuration_i[itrans] for itrans in EmissionTransitions_np]
        self.Term_i = [self.Term_i[itrans] for itrans in EmissionTransitions_np]
        self.J_i = self.J_i[EmissionTransitions_np] 

        self.Configuration_j = [self.Configuration_j[itrans] for itrans in EmissionTransitions_np]
        self.Term_j = [self.Term_j[itrans] for itrans in EmissionTransitions_np]
        self.J_j = self.J_j[EmissionTransitions_np] 

        self.Racah_i = [self.Racah_i[itrans] for itrans in EmissionTransitions_np]
        self.Racah_j = [self.Racah_j[itrans] for itrans in EmissionTransitions_np]

        self.Source_ji = [self.Source_ji[itrans] for itrans in EmissionTransitions_np]

        # Correct the EmissionTransitions list
        self.EmissionTransitions = range(self.NRadTrans)
        
        # Find indices 
        # self.index_i_lvl = {} 
        # self.index_j_lvl = {}
        self.index_i_lvl = np.zeros(self.NRadTrans,dtype=np.int32)
        self.index_j_lvl = np.zeros(self.NRadTrans,dtype=np.int32)        
        
        for itrans in self.EmissionTransitions: 
            i_lvl = -1
            index_temp = np.where(self.Term_lvl == self.Term_i[itrans])
            for i in index_temp[0]:
                if  (self.Configuration_lvl[i] == self.Configuration_i[itrans] and self.J_lvl[i] == self.J_i[itrans]):
                    i_lvl = i
                    # print(itrans,i_lvl,Configuration_lvl[i],Term_lvl[i],J_lvl[i] )           
            if i_lvl < 0:        
                print(itrans,"Couldn't find the index of the corresponding level while looping through the transitions.")
                raise SystemExit(0)    
            self.index_i_lvl[itrans] = i_lvl


            j_lvl = -1
            index_temp = np.where(self.Term_lvl == self.Term_j[itrans]) 
            for j in index_temp[0]:
                if  (self.Configuration_lvl[j] == self.Configuration_j[itrans] and self.J_lvl[j] == self.J_j[itrans]):
                    j_lvl = j
                    # print(itrans,i_lvl,Configuration_lvl[i],Term_lvl[i],J_lvl[i] )     
                  
            if j_lvl < 0:   
                ii = self.DictRacah_lvl[self.Racah_i[itrans]]
                jj = self.DictRacah_lvl[self.Racah_j[itrans]]        
     
                print(itrans,self.Configuration_j[itrans],self.Term_j[itrans],self.J_j[itrans],self.Source_ji[itrans],self.E_j[itrans],self.Racah_j[itrans])   
                print(jj,self.Configuration_lvl[jj],self.Term_lvl[jj],self.J_lvl[jj],self.E_lvl[jj],self.Racah_lvl[jj])   

                print(itrans,"Couldn't find the index of the corresponding level j while looping through the transitions.")
                raise SystemExit(0)  

            self.index_j_lvl[itrans] = j_lvl  


    
        for itrans in self.EmissionTransitions:
            """
            Transition data for Ar I
            i -> lower level
            j -> upper level
            """
            i = self.DictRacah_lvl[self.Racah_i[itrans]]
            j = self.DictRacah_lvl[self.Racah_j[itrans]]
        
            ii = self.index_i_lvl[itrans]
            jj = self.index_j_lvl[itrans]
    
            if i != ii:
                print(itrans,i,ii,"Problem with mapping transition levels to the corresponding indexing of the levels. i")
                raise SystemExit(0)     

            if j != jj:
                print(itrans,j,jj,"Problem with mapping transition levels to the corresponding indexing of the levels. j")
                raise SystemExit(0)     



        for it_1 in self.EmissionTransitions: 
            for it_2 in self.EmissionTransitions:
              if it_1 != it_2:
                if self.Racah_i[it_1] ==  self.Racah_i[it_2] and self.Racah_j[it_1] ==  self.Racah_j[it_2]:
                    print(it_1,it_2,self.Source_ji[it_1],self.Source_ji[it_1],self.Racah_i[it_1],self.Racah_j[it_1].
                          self.f_ji[it_1],self.f_ji[it_2],self.E_j[it_1]-self.E_i[it_1],self.E_j[it_2]-self.E_i[it_2],
                          self.E_i[it_1],self.E_j[it_1],self.g_i[it_1],self.g_i[it_2])
                    print('Some radiative transitions exist multiple times. Program will stop.')
                    raise SystemExit(0)
            
                   

        imax = 0; jmax = 0
        self.TransitionsToGroundState = []
        for itrans in self.EmissionTransitions:
            i = self.index_i_lvl[itrans]
            j = self.index_j_lvl[itrans]

            imax = max(i,imax)
            jmax = max(j,jmax)
            if (i > j): # Transition from a 
                print(i,self.Racah_lvl[i],self.Configuration_lvl[i], self.Term_lvl[i], self.J_lvl[i],self.E_lvl[i]*cm_eV)
                print(j,self.Racah_lvl[j],self.Configuration_lvl[j], self.Term_lvl[j], self.J_lvl[j],self.E_lvl[j]*cm_eV)
                raise SystemExit(0)
    
            if i ==0: 
                self.TransitionsToGroundState.append(itrans)    



        #----------------------------------------------------------------------------------


        # Calculate number of possible collisional transitions 
        NCollTrans = 0
        for i in range(self.N_lvl):
            for j in range(i+1,self.N_lvl):
                NCollTrans = NCollTrans + 1

        print("Number of all possible collisional transitions =", NCollTrans)
        self.NCollTrans = NCollTrans

        # Construct indices
        self.CollTransitionsList = range(NCollTrans)

        self.CollTransition = np.zeros([self.N_lvl,self.N_lvl], dtype=int)
        self.CollTransition_Dict = {}
        self.CollTransition_ij = np.zeros([NCollTrans,2], dtype=int)

        self.GlobalToNistIndex = {}
        self.NistToGlobalIndex = {}

        iCollTrans = 0
        for i in range(self.N_lvl):
            for j in range(i+1,self.N_lvl):
                # print(i,j,NCollTrans)
                self.CollTransition[i,j] = iCollTrans
                self.CollTransition_Dict[i,j] = iCollTrans
                self.CollTransition_ij[iCollTrans,0] = i
                self.CollTransition_ij[iCollTrans,1] = j        

                for itrans in self.EmissionTransitions:
                    ii = self.index_i_lvl[itrans]
                    jj = self.index_j_lvl[itrans]
                
                    if i == ii and j == jj: 
                        # if ii == 0 and jj == 83: 
                        #     print(i,j,itrans,iCollTrans,Racah_i[itrans],Racah_j[itrans])
                        #     raise SystemExit(0)
                        self.GlobalToNistIndex[iCollTrans] = itrans
                        self.NistToGlobalIndex[itrans] = iCollTrans
                        break

                iCollTrans = iCollTrans + 1
        

        self.eij_CollTrans = np.zeros(self.NCollTrans, dtype=np.float64)

        for iCollTrans in range(self.NCollTrans):  
            i = self.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.CollTransition_ij[iCollTrans,1] # Upper level                                
            eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV            
            self.eij_CollTrans[iCollTrans] = eij
    

        # iCollTrans = 300
        # i = self.CollTransition_ij[iCollTrans,0] # Lower lever
        # j = self.CollTransition_ij[iCollTrans,1] # Upper level
        # print(i,j,self.CollTransition_Dict[i,j])


        #----------------------------------------------------------------------------------
        """
        Read in LXCat data for excitation collision cross-sections
        """

        #----------------------------------------------------------------------------------

        # self.collDict, self.Nlvl_InExcDat, self.NTrans_InExcDat = \
            # LXCat_read('./CRModel/Data/LXCat-Data/Excitation/','Cross section.txt')
        # os.chdir(homeDir)
 
   
        # # fig,ax = plt.subplots(dpi=140)
        # # sigma = self.collDict['Ar']['Ar -> Ar(4s[3/2]2)']
        # # ax.plot(sigma[:,0],sigma[:,1],c='b',label="Ar -> Ar(4s[3/2]2)")
        # # ax.loglog()
        # #  # plt.ylim([1e2,1e26])
        # # plt.xlabel('E [eV]')
        # # plt.ylabel('$\sigma_{ex}$ [m$^2]$')
        # # plt.title('Excitation cross-sections')
        # # plt.grid(True)
        # # plt.legend()

        # # print(self.collDict.keys())
        # # print(self.collDict['Ar'].keys())
        # # print(self.collDict['Ar']['Ar -> Ar(4s[3/2]2)'][0,0])


        # self.collDict_list_2 = {}
        # self.CollTransitions_LXCat_2 = []
        # icount = 0
        # for key1 in self.collDict:
        #     for key2 in self.collDict[key1]:
        #         if (key2.split(" ")[2] != 'Ar(Rydberg)'):
        #             if (key2.split(" ")[0] in self.DictRacah_lvl.keys()) and  \
        #                 (key2.split(" ")[2] in self.DictRacah_lvl.keys()):
        #                 icount += 1
        #                 i = self.DictRacah_lvl[key2.split(" ")[0]]
        #                 j = self.DictRacah_lvl[key2.split(" ")[2]]
        #                 # if (i > j):
        #                 #     print(i,self.Racah_lvl[i],self.p.E_lvl[i]*cm_eV)
        #                 #     print(j,self.Racah_lvl[j],self.p.E_lvl[j]*cm_eV)
        #                 #     raise SystemExit(0)
        #                 if (i < j):
        #                     iCollTrans = self.CollTransition_Dict[i,j]
        #                     self.CollTransitions_LXCat_2.append(iCollTrans)
        #                     self.collDict_list_2[iCollTrans] = self.collDict[key1][key2]


        #----------------------------------------------------------------------------------


        # crsFileName = './CRModel/Data/LXCat-Data/Excitation/Cross section.txt'
        # crsFileName = './CRModel/Data/LXCat-Data/BSR/Excitation/Download/Cross section.txt'
        crsFileName = './CRModel/Data/LXCat-Data/Case1_BSR/Cross section.txt'
        # crsFileName = './CRModel/Data/LXCat-Data/Case2_Biagi+BSR/Cross section.txt'
        # crsFileName = './CRModel/Data/LXCat-Data/Case3_NGFSRDW/Cross section.txt'
        # crsFileName = './CRModel/Data/LXCat-Data/Case4_IST+BSR/Cross section.txt'

        CrossSections = multipleCrossSections(crsFileName)
        # self.collDict, self.Nlvl_InExcDat, self.NTrans_InExcDat 

        self.collDict_list = {}
        self.CollTransitions_LXCat = []
        icount = 0
        icoll = 0
        for c in CrossSections.crs:
            if typeDictI2S[c.colType] == "EXCITATION":

                lower_lvl = c.colName.split(" ")[0]
                upper_lvl = c.colName.split(" ")[2]
                # if (upper_lvl != 'Ar(Rydberg)' and  upper_lvl != 'Ar(HIGH)'):
                

                # icoll += 1              
                # print(icoll, " ",lower_lvl, " -> ", upper_lvl) 
                  
                if (lower_lvl in self.DictRacah_lvl.keys()) and  \
                    (upper_lvl in self.DictRacah_lvl.keys()):
                 
                    # icoll += 1              
                    # print(icoll, " ",lower_lvl, " -> ", upper_lvl) 

                    icount += 1
                    i = self.DictRacah_lvl[lower_lvl]
                    j = self.DictRacah_lvl[upper_lvl]
                    # if (i > j):
                    #     print(i,self.Racah_lvl[i],self.p.E_lvl[i]*cm_eV)
                    #     print(j,self.Racah_lvl[j],self.p.E_lvl[j]*cm_eV)
                    #     raise SystemExit(0)
                    if (i < j):
                        iCollTrans = self.CollTransition_Dict[i,j]
                        self.CollTransitions_LXCat.append(iCollTrans)
                        self.collDict_list[iCollTrans] = c.data

                    # if i==0:
                    #     print(i,j,max(self.collDict_list[iCollTrans][:,1]))

            if typeDictI2S[c.colType] == "IONIZATION": # NOTE(malamast): LXCat data includes cs only for ground state.
                self.IonizationGround = c.data

        #----------------------------------------------------------------------------------

        self.CollTransitions_Rest = []
        for element in self.CollTransitionsList:
            if element not in self.CollTransitions_LXCat:
                self.CollTransitions_Rest.append(element)


        self.CollTransitions_LXCat = np.array(self.CollTransitions_LXCat)
        self.NCollTrans_LXCat = len(self.CollTransitions_LXCat)

        self.eij_LXCat = np.zeros([self.NCollTrans_LXCat])

        self.CollTransition_ij_LXCat = np.zeros([2*self.NCollTrans_LXCat],dtype=np.int32)
        for iter in range(self.NCollTrans_LXCat):
            iCollTrans = self.CollTransitions_LXCat[iter]
            i = self.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.CollTransition_ij[iCollTrans,1] # Upper level    
            self.CollTransition_ij_LXCat[iter] = i 
            self.CollTransition_ij_LXCat[iter + self.NCollTrans_LXCat] = j 

            eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV  
            self.eij_LXCat[iter] = eij




        # # LXCat data - Map
        # sigma_ij = np.zeros([self.N_lvl,self.N_lvl])
        # for iCollTrans in self.CollTransitions_LXCat:
            
        #     i = self.CollTransition_ij[iCollTrans,0] # Lower lever
        #     j = self.CollTransition_ij[iCollTrans,1] # Upper level                               
        #     eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV 

        #     sigma_ij[i,j] = max(self.collDict_list[iCollTrans][:,1])
            
                        
        # sigma_ij[np.where(sigma_ij == 0.0)] = 'nan'
        # from matplotlib.colors import LogNorm

        # # creating a plot
        # pixel_plot = plt.figure(dpi=140)
        # # plotting a plot
        # # pixel_plot.add_axes()
        # # customizing plot
        # plt.title("Max $\sigma$ for all possible transitions")
        # # pixel_plot = plt.imshow(sigma_ij, cmap='jet', interpolation='nearest', origin='lower')
        # pixel_plot = plt.imshow(sigma_ij, cmap='jet',origin='lower', norm=LogNorm())
        # plt.xlabel('upper level index')
        # plt.ylabel('lower level index')    
        # plt.colorbar(pixel_plot)
        # # ## save a plot
        # # ## plt.savefig('pixel_plot.png')

        # # fig,ax = plt.subplots(dpi=140)
        # # ax.plot(range(self.N_lvl),self.E_lvl*cm_eV,'.')
        # # # ax.plot(E_lvl*cm_eV,'.')
        # # # ax.semilogy()
        # # # plt.ylim([1e2,1e26])
        # # plt.ylabel('E [ev]')
        # # plt.xlabel('level index')
        # # plt.title('Energy levels')
        # # plt.grid(True)
        # # # plt.legend()
            

        #----------------------------------------------------------------------------------

        p = (self.J_lvl, self.E_lvl, self.Racah_lvl,  self.DictRacah_lvl, 
            self.SubShell_lvl, self.isPrimed_lvl, self.AngMomQuantNum_lvl, self.Parity_lvl, self.Spin_lvl,
            self.f_ji, self.Racah_i, self.Racah_j, self.EmissionTransitions, self.CollTransition_Dict,
            self.CollTransition_ij,self.CollTransitionsList,
            self.GlobalToNistIndex,self.NistToGlobalIndex)


        self.CollTransition_Status,self.KimuraFactor_K = CharacteriseTransitions(*p)
        os.chdir(homeDir)

        # self.makeSets()


    def makeSets(self):      
        self.EmissionTransitions = set(self.EmissionTransitions)
        # self.CollTransitions_LXCat = set(self.CollTransitions_LXCat)
        # self.CollTransitions_Rest = set(self.CollTransitions_Rest)


    def EvaluateCrossSections(self, eRange):

        # Excitation
        # self.sigma_ij_Exc_2 = np.zeros([self.NCollTrans,len(eRange)])
        
        
        # Excitation LXCat
        self.sigma_ij_Exc_LXCat = {}
        for iCollTrans in self.CollTransitions_LXCat: 
            i = self.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.CollTransition_ij[iCollTrans,1] # Upper level
            eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV

            self.sigma_ij_Exc_LXCat[iCollTrans] = np.interp(eRange,self.collDict_list[iCollTrans][:,0],self.collDict_list[iCollTrans][:,1])
            self.sigma_ij_Exc_LXCat[iCollTrans][np.where(eRange < eij)] = 0

            # self.sigma_ij_Exc_2[iCollTrans] = self.sigma_ij_Exc_LXCat[iCollTrans]


        # Excitation Rest
        self.sigma_ij_Exc_Rest = {}
        for iCollTrans in self.CollTransitions_Rest: 
            i = self.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.CollTransition_ij[iCollTrans,1] # Upper level
            # CollTransition_Dict[i,j] should be equal to iCollTrans
            # if (i > j): # Perform a test
                # raise SystemExit(0)
                            
            eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV
                
            if (eij >0.0):                  
                    
                # Drawin's Formula (unit parameters have been assumed)
                CollStatus = self.CollTransition_Status[i,j]
                sigma_ij_pre = sigma_factor_Drawin*(eRange/eij-1)/(eRange/eij)**2
                if CollStatus == 'A' or CollStatus == 'AP': 
                    if iCollTrans in self.GlobalToNistIndex.keys():
                        itrans = self.GlobalToNistIndex[iCollTrans]
                        OscillatorStrength = self.f_ji[itrans]
                    else:
                        OscillatorStrength = OscillatorStrength_VR 
                    sigma_ij = sigma_ij_pre * OscillatorStrength * (Eion_H/eij)**2 * np.log(1.25*eRange/eij)
                elif CollStatus == 'J0' or 'LS' or  'J0S' or 'JPS' or 'JS':
                    sigma_ij = sigma_ij_pre * (eRange/eij+1)/(eRange/eij)**3  
                else: 
                # elif CollStatus == 'P' or 'LP' or 'L' or 'JP' or 'J' or 'JLP' or 'JL' or 'JOP' or 'JOLP' or 'JOL' or 'LPS' or 'J0PS' or 'J0LPS' or 'J0LS' or 'JLPS' or 'JLS':
                    sigma_ij = sigma_ij_pre 
                                                                                    
                # if (i<j):                    
                #     # Van Regemorter 1962 
                #     if (CollStatus == 'A'): 
                #         if (iCollTrans in self.p.GlobalToNistIndex.keys()):
                #             itrans = self.p.GlobalToNistIndex[iCollTrans]
                #             OscillatorStrength = self.p.f_ji[itrans]
                #         else: 
                #             OscillatorStrength = OscillatorStrength_VR                         
                #     else:
                #         OscillatorStrength = OscillatorStrength_VR 

                #     sigma_ij = sigma_factor_VanRegemorter*gauntFactor_BoundBound*OscillatorStrength/self.p.eRange/eij
                #     sigma_ij[np.where(self.p.eRange < eij)] = 0

                sigma_ij[np.where(eRange < eij)] = 0
                
                self.sigma_ij_Exc_Rest[iCollTrans] = sigma_ij
                # self.sigma_ij_Exc_2[iCollTrans] = self.sigma_ij_Exc_Rest[iCollTrans]

            else:
                print("Some collisional transitions are reversed; deexcitation? Check if level energies are ordered.")
                exit(-1)


        # Electron Impact Excitation (Packed)
        self.sigma_ij_Exc = []

        for iCollTrans in range(self.NCollTrans):
            if iCollTrans in self.CollTransitions_LXCat:
                self.sigma_ij_Exc.append(self.sigma_ij_Exc_LXCat[iCollTrans])
            elif iCollTrans in self.CollTransitions_Rest:
                self.sigma_ij_Exc.append(self.sigma_ij_Exc_Rest[iCollTrans])
            else:
                print("No excitation cross section was defined for collisional transition: ", iCollTrans)
                exit(-1)
                
        if len(self.sigma_ij_Exc) != self.NCollTrans:
            print(len(self.sigma_ij_Exc), " transitions were defined, while the number of collisional transition is: ", self.NCollTrans)
            exit(-1)
                



        # De-excitation (reverse) procesess  # by principle of detailed balance
        self.sigma_ij_deExc = []
        for iCollTrans in range(self.NCollTrans): 

            i = self.CollTransition_ij[iCollTrans,0] # Lower lever
            j = self.CollTransition_ij[iCollTrans,1] # Upper level                               
            eij = self.eij_CollTrans[iCollTrans]
            

            sigma_dexc = self.g_lvl[i]/self.g_lvl[j]*(eRange[:,0] + eij)/eRange[:,0]*\
                np.interp(eRange[:,0],eRange[:,0]-eij,self.sigma_ij_Exc[iCollTrans][:,0] )  
            self.sigma_ij_deExc.append(sigma_dexc[:, np.newaxis])

   
        # Electron Impact Ionization
        self.sigma_ionLXCat = np.interp(eRange,self.IonizationGround[:,0],self.IonizationGround[:,1])  
        # deltaIon = Eion - self.E_lvl[0]*cm_eV 
        deltaIon = self.deltaIon[0]

        self.sigma_ionLXCat[np.where(eRange < deltaIon)] = 0

        # sigma_ion = self.sigma_ionLXCat     
        # sigma_ion_2 = 4*np.pi*a0**2*RydEn**2/(eRange + 3.25*deltaIon)*(5/(3*deltaIon) - 1/eRange - 2*deltaIon/(3*eRange**2))   # (Vriens and Smeets, 1980) Which Borh radius do I need here?         
        # # sigma_ion[np.where(eRange < deltaIon)] = 0

        # fig,ax = plt.subplots(dpi=160)
        # ax.plot(eRange,sigma_ion,'b*')
        # ax.plot(parameters.IonizationBSR[:,0],parameters.IonizationBSR[:,1],'r*')
        # ax.plot(eRange,sigma_ion_2,'k')
        # plt.legend()
        # ax.semilogy()
        # plt.grid(True)
        # plt.show()

        self.sigma_ij_Ion = {}
        self.sigma_ij_Ion[0] = self.sigma_ionLXCat
        for i in range(1,self.N_lvl):              
            SubShell = str(self.SubShell_lvl[i])
            # print(SubShell)
            char = SubShell[-1] 
                
            # isPrimed_lvl    
            # deltaIon = Eion - self.E_lvl[i]*cm_eV   
            deltaIon = self.deltaIon[i]   
                                    
            # (Vriens and Smeets, 1980) Which Borh radius do I need here?
            # sigma_ion = 4*np.pi*a0_H**2*RydEn**2/(eRange + 3.25*deltaIon)*(5/(3*deltaIon) - 1/eRange - 2*deltaIon/(3*eRange**2))     

            # (H. Deutsch et al, 2003) 
            u = eRange/deltaIon
            u[np.where(eRange < deltaIon)] = 1.0
            f = ParamAngMom[char].d/u * ((u-1)/(u+1))**ParamAngMom[char].a * \
                (ParamAngMom[char].b + ParamAngMom[char].c*(1.0-0.5/u)*np.log(2.7 + np.sqrt(u-1.0)))
            sigma_ion = (WeightingFactor_g_nl[SubShell]/deltaIon) * np.pi  * \
                Radius_nl[SubShell]**2 * xi_nl * f
                                        
            sigma_ion[np.where(eRange < deltaIon)] = 0
            self.sigma_ij_Ion[i] = sigma_ion


        # # Absolute partial and total cross sections for electron-impact ionization of argon from threshold to 1000 eV (H. C. Straub et al., 1995)
        # e_exp = np.array([17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 140, 160, 180, 200, 225, 
        #                   250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
        # sigma_Ion_exp = np.array([0.017, 0.46, 1.24, 1.84, 2.26, 2.55, 2.66, 2.7, 2.69, 2.67, 2.67, 2.67, 2.66, 2.69, 2.7, 2.69, 2.67, 2.64, 2.61, 2.55, 2.45, 2.35,
        #                           2.27, 2.18, 2.1, 1.99, 1.87, 1.79, 1.63, 1.51, 1.39, 1.31, 1.23, 1.16, 1.09, 1.03, 0.976, 0.932, 0.901, 0.865, 0.824, 0.795]) *1e-20

        # self.sigma_ij_Ion2 = {}
        # for i in range(0,self.N_lvl):              
        #     SubShell = str(self.SubShell_lvl[i])
        #     # print(SubShell)
        #     char = SubShell[-1] 
                
        #     # isPrimed_lvl    
        #     # deltaIon = Eion - self.E_lvl[i]*cm_eV   
        #     deltaIon = self.deltaIon[i]   
                                    
        #     # (Vriens and Smeets, 1980) Which Borh radius do I need here?
        #     sigma_ion = 4*np.pi*a0_H**2*RydEn**2/(eRange + 3.25*deltaIon)*(5/(3*deltaIon) - 1/eRange - 2*deltaIon/(3*eRange**2))     

        #     sigma_ion[np.where(eRange < deltaIon)] = 0
        #     self.sigma_ij_Ion2[i] = sigma_ion

        # fig,ax = plt.subplots(dpi=160)
        # ax.plot(eRange,self.sigma_ij_Ion[0],c='blue',label="Ar")
        # ax.plot(eRange,self.sigma_ij_Ion[1],c='green',label="s5")
        # ax.plot(eRange,self.sigma_ij_Ion[2],c='red',label="s4")
        # ax.plot(eRange,self.sigma_ij_Ion[3],c='cyan',label="s3")
        # ax.plot(eRange,self.sigma_ij_Ion[4],c='magenta',label="s2")
        # ax.plot(eRange,self.sigma_ij_Ion[5],c='yellow',label="p10")
        # ax.plot(eRange,self.sigma_ij_Ion[6],c='black',label="p9")
        # ax.plot(eRange,self.sigma_ij_Ion[7],c='orange',label="p8")
        # ax.plot(eRange,self.sigma_ij_Ion[8],c='purple',label="p7")

        # # ax.plot(eRange,self.sigma_ij_Ion2[0],c='blue',ls='--')
        # # ax.plot(eRange,self.sigma_ij_Ion2[1],c='green',ls='--')
        # # ax.plot(eRange,self.sigma_ij_Ion2[2],c='red',ls='--')
        # # ax.plot(eRange,self.sigma_ij_Ion2[3],c='cyan',ls='--')
        # # ax.plot(eRange,self.sigma_ij_Ion2[4],c='magenta',ls='--')
        # # ax.plot(eRange,self.sigma_ij_Ion2[5],c='yellow',ls='--')
        # # ax.plot(eRange,self.sigma_ij_Ion2[6],c='black',ls='--')
        # # ax.plot(eRange,self.sigma_ij_Ion2[7],c='orange',ls='--')
        # # ax.plot(eRange,self.sigma_ij_Ion2[8],c='purple',ls='--')

        # ax.plot(e_exp,sigma_Ion_exp,c='olive',label="Exp")
        
        # plt.legend()
        # ax.semilogy()
        # plt.grid(True)
        # plt.show()  
        # exit(-1)      
        


        # 3-body recombination (reverse of electron-impact ionization) procesess  # by principle of detailed balance
        self.sigma_ij_Recomb = {}
        for i in range(0,self.N_lvl):              

            # deltaIon = Eion - self.E_lvl[i]*cm_eV                   
            deltaIon = self.deltaIon[i]   

            if i == 0:
                g_plus = g_ion_1 + g_ion_2               
            elif (not self.isPrimed_lvl[i]):
                g_plus = g_ion_1
            else:
                g_plus = g_ion_2
                
            sigma_Recomb = self.g_lvl[i] / g_plus * (eRange[:,0] + deltaIon) / eRange[:,0] * \
                np.interp(eRange[:,0],eRange[:,0]-deltaIon,self.sigma_ij_Ion[i][:,0] )  
                                   
            self.sigma_ij_Recomb[i] =  sigma_Recomb[:, np.newaxis]   



        # Atom impact Ionization 
        # deltaIon = Eion - self.E_lvl[0]*cm_eV
        deltaIon = self.deltaIon[0]   
        eRange_temp = eRange - Eion
        eRange_temp[np.where(eRange_temp < 0.0)] = 0.0
        
        self.sigma_1a_ion = 1.8e-25*(eRange_temp)**1.3
        self.sigma_1a_ion[np.where(eRange < deltaIon)] = 0.0

        self.sigma_ia_ion = {}
        self.sigma_ia_ion[0] = self.sigma_1a_ion
        for i in range(1,self.N_lvl):
            
            # deltaIon = Eion - self.E_lvl[i]*cm_eV
            deltaIon = self.deltaIon[i]   
            
            if i==0: 
                xi = xi_Ar
            else:
                xi = 1
                     
            # self.sigma_ia_ion = 4*np.pi*a0_H**2*(Eion_H/deltaIon)**2 * M_Ar/M_H * xi**2 * 2*spc.m_e/(M_Ar + spc.m_e) * (eRange/deltaIon - 1) \
            # / (1.0 + 2*spc.m_e/(M_Ar + spc.m_e) * (eRange/deltaIon - 1) )**2
            sigma_ia_ion = sigma_factor_AtomExc * xi**2 / deltaIon**2 * mass_factor  * \
                (eRange/deltaIon - 1) / (1.0 + mass_factor * (eRange/deltaIon - 1) )**2  
            sigma_ia_ion[np.where(eRange < deltaIon)] = 0.0
            self.sigma_ia_ion[i] = sigma_ia_ion
              


        # 3-body recombination (reverse of electron-impact ionization) procesess  # by principle of detailed balance
        self.sigma_ia_Recomb = {}
        for i in range(0,self.N_lvl):              

            # deltaIon = Eion - self.E_lvl[i]*cm_eV                   
            deltaIon = self.deltaIon[i]   

            if i == 0:
                g_plus = g_ion_1 + g_ion_2               
            elif (not self.isPrimed_lvl[i]):
                g_plus = g_ion_1
            else:
                g_plus = g_ion_2

            sigma_Recomb = self.g_lvl[i] / g_plus * (eRange[:,0] + deltaIon) / eRange[:,0] * \
                np.interp(eRange[:,0],eRange[:,0]-deltaIon,self.sigma_ia_ion[i][:,0] )  
                
            self.sigma_ia_Recomb[i] =  sigma_Recomb[:, np.newaxis]   





        # Atom impact de/excitation from ground
        self.sigma_ij_Atom_ExcFromGround = {}
        self.j_Atom_ExcFromGround = {}
        self.i_Atom_ExcFromGround = self.DictRacah_lvl["Ar"] 
        self.itrans_Atom_ExcFromGround = []
        
        i = self.i_Atom_ExcFromGround 
        nTrans = len(atomImpactExcitationTransitionsFromGroundSate)
        for itrans in range(nTrans):
            if  (atomImpactExcitationTransitionsFromGroundSate[itrans] in self.DictRacah_lvl.keys()):

                j = self.DictRacah_lvl[atomImpactExcitationTransitionsFromGroundSate[itrans]]             
                eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV
                    
                sigma_ij_a = fij[itrans] * sigma_factor_AtomExc * xi_Ar**2 / eij**2 * mass_factor  * \
                        (eRange/eij - 1) / (1.0 + mass_factor * (eRange/eij - 1) )**2
                sigma_ij_a[np.where(eRange < eij)] = 0.0

                self.sigma_ij_Atom_ExcFromGround[itrans] = sigma_ij_a
                self.j_Atom_ExcFromGround[itrans] = j 
                self.itrans_Atom_ExcFromGround.append(itrans)
                

        # Make a test to see of itrans_Atom_ExcFromGround is ordered.
        if not all(self.itrans_Atom_ExcFromGround[i] == i for i in range(len(self.itrans_Atom_ExcFromGround))):
            print("self.itrans_Atom_ExcFromGround is not ordered. Program will stop.")
            exit(-1)
                        
        
        # Atom impact de/excitation rest
        self.sigma_ij_Atom_Exc = {}
        self.j_Atom_Exc= {}
        self.i_Atom_Exc= {}
        self.itrans_Atom_Exc = []

        nTrans = len(atomImpactExcitationTransitions_j)
        for itrans in range(nTrans):
            if  (atomImpactExcitationTransitions_i[itrans] in self.DictRacah_lvl.keys() and \
                atomImpactExcitationTransitions_j[itrans] in self.DictRacah_lvl.keys()):

                i = self.DictRacah_lvl[atomImpactExcitationTransitions_i[itrans]]         
                j = self.DictRacah_lvl[atomImpactExcitationTransitions_j[itrans]] 

                eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV

                sigma_ij_a = beta_ij[itrans] * (eRange - eij) / eij**2.26 
                sigma_ij_a[np.where(eRange < eij)] = 0.0

                self.sigma_ij_Atom_Exc[itrans] = sigma_ij_a
                self.i_Atom_Exc[itrans] = i 
                self.j_Atom_Exc[itrans] = j 
                self.itrans_Atom_Exc.append(itrans)
                
          

        # Make a test to see of itrans_Atom_ExcFromGround is ordered.
        if not all(self.itrans_Atom_Exc[i] == i for i in range(len(self.itrans_Atom_Exc))):
            print("self.itrans_Atom_Exc ins not ordered. Program will stop.")
            exit(-1)





        # Atom impact de/excitation rest
        self.sigma_ij_Atom_Exc_2 = {}
        self.i_Atom_Exc_2= {}
        self.j_Atom_Exc_2= {}
        self.itrans_Atom_Exc_2 = []

        self.itrans_Atom_Exc_2 = self.EmissionTransitions

        for itrans in self.itrans_Atom_Exc_2: 

            i = self.index_i_lvl[itrans]
            j = self.index_j_lvl[itrans]            
            
            self.i_Atom_Exc_2[itrans] = i 
            self.j_Atom_Exc_2[itrans] = j 
            
            eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV

            if i==0: 
                xi = xi_Ar
            else:
                xi = 1
                    
            sigma_ij_a = self.f_ji[itrans] * sigma_factor_AtomExc * xi**2 / eij**2 * mass_factor  * \
                        (eRange/eij - 1) / ( 1.0 + mass_factor * (eRange/eij - 1) )**2

            sigma_ij_a[np.where(eRange < eij)] = 0.0

            self.sigma_ij_Atom_Exc_2[itrans] = sigma_ij_a
            

        for itrans in self.itrans_Atom_Exc_2:  
            i = self.i_Atom_Exc_2[itrans]  
            j = self.j_Atom_Exc_2[itrans]
            nTrans = len(atomImpactExcitationTransitions_j)
            for itrans_a in range(nTrans):
                if self.Racah_lvl[j] == atomImpactExcitationTransitions_i[itrans_a] and \
                   self.Racah_lvl[j] == atomImpactExcitationTransitions_j[itrans_a]:
                    print("Multiple atom-atom transitions have been added. Program will stop. ")
                    print(i,j)
                    exit(-1)


        nTrans = len(atomImpactExcitationTransitions_j)
        self.itrans_Atom_Exc_2 = range(len(self.EmissionTransitions)+nTrans)
        for itrans in range(nTrans):
            if  (atomImpactExcitationTransitions_i[itrans] in self.DictRacah_lvl.keys() and \
                atomImpactExcitationTransitions_j[itrans] in self.DictRacah_lvl.keys()):

                i = self.DictRacah_lvl[atomImpactExcitationTransitions_i[itrans]]         
                j = self.DictRacah_lvl[atomImpactExcitationTransitions_j[itrans]] 

                eij = (self.E_lvl[j] - self.E_lvl[i])*cm_eV

                sigma_ij_a = beta_ij[itrans] * (eRange - eij) / eij**2.26 
                sigma_ij_a[np.where(eRange < eij)] = 0.0

                itrans_a = len(self.EmissionTransitions) + itrans

                self.i_Atom_Exc_2[itrans_a] = i 
                self.j_Atom_Exc_2[itrans_a] = j 
                self.sigma_ij_Atom_Exc_2[itrans_a] = sigma_ij_a





        # Photorecombination/photoionization 
        self.sigma_c_ion = {}        
        
        i = 0 # from ground state
        # deltaIon = Eion - self.E_lvl[i]*cm_eV
        deltaIon = self.deltaIon[i]   

        g_plus = g_ion_1 + g_ion_2  

        sigma_c_ion = 2.8e-20 * (Eion_H / (eRange + deltaIon) )**3
        sigma_c_ion[np.where(eRange <= 2*Eion_H - deltaIon)] = 3.5e-21                                                
        sigma_c_ion = sigma_c_ion * self.g_lvl[i] / g_plus * (eRange + deltaIon)**2 \
            / eRange * photoionization_factor *spc.e # Mind the units of the energy
        # sigma_c_ion[np.where(self.p.eRange < deltaIon)] = 0.0 # Mind that the limits of the integral have changed

        self.sigma_c_ion[0] =  sigma_c_ion
      

        nTrans = 5
        for i in range(1,nTrans):
            # deltaIon = Eion - self.E_lvl[i]*cm_eV 
            deltaIon = self.deltaIon[i]   
            
            if (not self.isPrimed_lvl[i]):
                g_plus = g_ion_1
            else:
                g_plus = g_ion_2

            sigma_c_ion = 7.91 * (deltaIon/Eion_H)**2.5 * (Eion_H / (eRange + deltaIon))**3
            sigma_c_ion[np.where(eRange <= 0.59*Eion_H - deltaIon)] = 2.0                    
            sigma_c_ion = sigma_c_ion * 1e-22 * gamma_i[i] * self.g_lvl[i] / g_plus * \
                photoionization_factor * (eRange + deltaIon)**2 /eRange *spc.e # Mind the units of the energy
            
            # sigma_c_ion[np.where(eRange < deltaIon)] = 0.0 # Mind that the limits of the integral have changed
            self.sigma_c_ion[i] =  sigma_c_ion

        # Elastic Collisions
        self.sigma_el_e1 = np.interp(eRange,eRange_elastic_e1,sigma_elastic_e1)*1e-20
        self.sigma_el_e1[np.where(self.sigma_el_e1 < 0)] = 0.0  

        
        self.eRange_elastic_e1 = eRange_elastic_e1
        self.sigma_elastic_e1 = sigma_elastic_e1                       



    def ConvertCrossSectionsToNumPy(self):

        self.sigma_ij_Exc = np.array(self.sigma_ij_Exc)                
        self.sigma_ij_deExc = np.array(self.sigma_ij_deExc)  
        # self.sigma_ij_Exc_comb = np.stack((self.sigma_ij_Exc, self.sigma_ij_deExc), axis=2)

        self.sigma_ij_Exc_LXCat = np.array(list(self.sigma_ij_Exc_LXCat.values()))        
        self.sigma_ij_Ion = np.array(list(self.sigma_ij_Ion.values()))
        self.sigma_ij_Recomb = np.array(list(self.sigma_ij_Recomb.values()))

        
        self.sigma_ia_ion = np.array(list(self.sigma_ia_ion.values()))
        self.sigma_ia_Recomb = np.array(list(self.sigma_ia_Recomb.values()))

        self.sigma_c_ion = np.array(list(self.sigma_c_ion.values()))
        
        self.sigma_ij_Atom_ExcFromGround = np.array(list(self.sigma_ij_Atom_ExcFromGround.values()))
        self.j_Atom_ExcFromGround = np.array(list(self.j_Atom_ExcFromGround.values()))

        self.sigma_ij_Atom_Exc = np.array(list(self.sigma_ij_Atom_Exc.values()))
        self.j_Atom_Exc = np.array(list(self.j_Atom_Exc.values()))
        self.i_Atom_Exc = np.array(list(self.i_Atom_Exc.values()))
        
        
        self.sigma_ij_Atom_Exc_2 = np.array(list(self.sigma_ij_Atom_Exc_2.values()))
        self.j_Atom_Exc_2 = np.array(list(self.j_Atom_Exc_2.values()))
        self.i_Atom_Exc_2 = np.array(list(self.i_Atom_Exc_2.values()))


    
        
