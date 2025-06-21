import os
import sys

solver_dir = os.path.dirname(os.path.abspath(__file__))
case_dir = os.path.join(solver_dir, 'Cases')
crmodel_dir = os.path.join(solver_dir, 'CRModel', 'src')

print(solver_dir)
print(case_dir)
print(crmodel_dir)

sys.path.append(case_dir)
sys.path.append(crmodel_dir)

#sys.path.append('./Cases/')  # Add the path to the folder containing my_module.py
#sys.path.append('./CRModel/src/')  # Add the path to the folder containing my_module.py


from os import environ
N_THREADS = '8'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

import matplotlib.pyplot as plt

import numpy as np
import numpy.polynomial.chebyshev as cheb
# from scipy.sparse.linalg import cg
# from scipy.linalg import solve
# from scipy.sparse.linalg import gmres
from scipy.linalg import lu_factor, lu_solve   # CPU

import time as cpu_time
import cProfile


CUDA_NUM_DEVICES      = 0
try:
  import cupy as cp
  #CUDA_NUM_DEVICES=cp.cuda.runtime.getDeviceCount()
  # from cupyx.scipy.linalg import lu_factor, lu_solve
except ImportError:
  print("Please install CuPy for GPU use")
  #sys.exit(0)
except:
  print("CUDA not configured properly !!!")
  sys.exit(0)


from Liu2014Properties import setLiu2014Properties
from psaapPropertiesTestArm import setPsaapPropertiesTestArm
from psaapPropertiesTestArmInterpTrans import setPsaapPropertiesTestArmInterpTrans
from psaapProperties_6Species_Nominal import setPsaapProperties_6Species_Nominal

from psaapProperties_6Species import setPsaapProperties_6Species

from psaapProperties_CRModel import setPsaapProperties_CRModel
from CRModel import CollisionalRadiativeModel

import Constants as CRconst


class modelClosures:
    """Class providing model parameters."""

    def __init__(self, Ns, Nr):
        """Set model parameter values.  These values are non-dimensionalized
        using the following quantities:

        L: the half-gap-width of the device
        \tau: the period of the driving RF voltage
        V_0: the amplitude of the driving RF voltage
        n_b: the "background" number density
        n_0: the "nominal" electron number density
        e_0: the "nominal" electron energy

        The non-dimensional parameters provided by the class are

        mue: Electron mobility
            mu_e^{\ast} = \frac{mu_e V_0 \tau}{L^2}

        mui: Ion mobility (non-dimensionalized same as mue)

        De: Electron diffusivity
            D_e^{\ast} = \frac{D_e \tau}{L^2}

        Di: Ion diffusivity (non-dimensionalized same as Di)

        Ck: Ionization rate pre-exponential factor
            C_k^{\ast} = C_k \tau n_b

        A: Ionization rate "activation energy"
            A^{\ast} = A / e_0

        dH: Energy lost per electron in ionization reaction
            dH^{\ast} = dH / e_0

        qStar: Multiplier in front of Joule heating term (denoted
            qStar since it takes place of unit charge in dimensional
            equations)

            qStar = \frac{q_e V_0}{e_0} (where q_e is unit charge)

        alpha: Multiplier in Poisson eqn (Gauss' law)
            \alpha = \frac{q_e n_p L^2}{V_0 \epsilon_0}
            (where \epsilon_0 is permittivity of free space)
        """
        self.xp_module = np
        
        self.Ns = Ns # number of species
        self.Nr = Nr # number of reactions

        # charge number
        self.Z = np.zeros(Ns+1)
        self.Z[0] = -1 # electrons are always -1
        self.Z[1] =  1 # ions are always 1

        # specific heat at constant volume and pressure
        self.Cv = np.zeros(Ns+1)
        self.Cp = np.zeros(Ns+1)


        # mobility
        self.mu = np.zeros(Ns+1) # NOTE(malamast): We now include transport coefficients for the electron energy equation

        # diffusivity
        self.D = np.zeros(Ns+1)

        # reaction rate data

        # Modified Arrhenius rxn rate coefficients for now
        # kf(T) = A*(T**B)*exp(-C/T)
        self.A = np.zeros(Nr) #self.Ck = 272.0
        self.B = np.zeros(Nr)
        self.C = np.zeros(Nr) #18.687*(3./2.);

        # energy gain/loss in electrons
        self.dH = np.zeros(Nr) 
        self.dEps = np.zeros(Ns)

        # stoichiometric coefficients (Ns+1 b/c we store coefficient
        # for the background gas... it is only used for
        # non-dimensionalization purposes)
        self.beta = np.zeros((Ns,Nr),dtype=np.int64) # products
        self.alfa = np.zeros((Ns,Nr),dtype=np.int64) # reactants

        # this represents a single rxn: Ar+e -> Ar+ + e + e
        #self.beta[0,0] = 2
        #self.beta[1,0] = 1
        #self.beta[2,0] = 0

        #self.alfa[0,0] = 1
        #self.alfa[1,0] = 0
        #self.alfa[2,0] = 1

        # other non-dimensional parameters
        self.qStar = 100.0
        self.alpha = 2.33e3

        # boundary condition parameters
        self.gam   = 0.01
        self.ks    = 6.89e-1
        self.ksion = 0.0

        # background specie density
        self.kappaB  = 4.42
        self.p0      = 133.3224*1.5/1.6e-19/8e16
        self.nAronp0 = 3.22e22 / 8e16
        self.Tg0     = 0.038778

        # Non-dimensionalization parameters
        self.nAr     = 3.22e22          # "nominal" ground number density [1/m^3]
        self.np0     = 8e16             # "nominal" electron density [1/m^3]
        self.tau     = 1./13.56e6
        self.tauOvernp0 = self.tau/self.np0
        self.tauOvernAr = self.tau/self.nAr

        self.TwoOverThree = 2.0/3.0     
        self.ThreeOverTwo = 3.0/2.0     

        # coefficient for the elastic collision term
        self.EC = 2.0 * 0.511e6 / 37.2158e9 * 3.8e9 * (1./13.6e6)
        

        # DC voltage (vertical shift in the driving voltage)
        self.verticalShift = 0.0

        # electron energy Dirichlet BC
        self.EeBC = 0.75
        # self.EeBC = 1.5
        self.electron_energy_dirichlet = True

        # Parameters needed to compute the current with dimensions
        self.V0Ltau  = 100 / (2.54 * 0.005 * (1./13.6e6))
        self.V0L     = 100 / (2.54 * 0.005)
        self.LLV0tau = (2.54 * 0.005)**2 / (100 * (1./13.6e6))
        self.tauL    = 2.54 * 0.005 / (1./13.6e6)
        # self.np0     = 8e16             # "nominal" electron density [1/m^3]
        self.qe      = 1.6e-19          # unit charge [C]
        self.eps0    = 8.86e-12         # permittivity of free space [F/m]
        self.eArea   = np.pi * 0.05**2  # electrode area [m^2]

        # non-dimensional parameter for the effective electric field for ions
        self.vmStar = 100 * (1./13.6e6)**2 / (2.54 * 0.005)**2 * 1.6e-19 / 6.63352088e-26
        # mass of argon atom 6.63352088e-26 [kg]


        # Parameters needed for the Collisional-Radiative model
        self.Pressure = 0.0
        self.GasTemperature = 0.0


        self.reactionsList =[]
        self.diffusivityList =[]
        self.mobilityList =[]

      

    def copy_operators_H2D(self, dev_id):
      
      if self.args.use_gpu==0:
        return
      
      with cp.cuda.Device(dev_id):

        self.Z      = cp.asarray(self.Z)
        self.mu     = cp.asarray(self.mu)
        self.D      = cp.asarray(self.D)
        self.A      = cp.asarray(self.A)
        self.B      = cp.asarray(self.B)
        self.C      = cp.asarray(self.C)

        self.Cv      = cp.asarray(self.Cv)
        self.Cp      = cp.asarray(self.Cp)

        # self.ksion = cp.asarray(self.ksion)

      return


    def charge(self,i):
        return self.Z[i]


    def mobility(self, i, energy, nb):
        xp  = self.xp_module

        mu = xp.zeros((nb.shape[0],1),dtype=xp.float64)

        if (len(self.mobilityList)>i and self.mobilityList[i].interpolate):
            indFix = (energy[:,i]<=0.0)
            energy[indFix,i] = 0.0

            indFix = (energy[:,i]>10.0)
            energy[indFix,i] = 10.0

            mu[:,0] = self.mobilityList[i].mu_expression((2./3)*energy[:,i]) / nb
        else:
            mu[:,0] = self.mu[i] / nb

        return mu[:,0]

    def mobility_U(self, i, j, energy, energy_U, mu, nb):
        xp  = self.xp_module

        mu_U = xp.zeros((nb.shape[0],nb.shape[0]),dtype=xp.float64)

        if (len(self.mobilityList) > i and self.mobilityList[i].interpolate):
            indFixL = (energy[:,i]<=0.0)
            energy[indFixL,i] = 0.0
            energy_U[i,j,indFixL,:] = 0.0

            indFixH = (energy[:,i]>10.0)
            energy[indFixH,i] = 10.0
            energy_U[i,j,indFixH,:] = 0.0

            mu_ee = (2./3)*self.mobilityList[i].mu_T_expression((2./3)*energy[:,i]) / nb

            # mu_U_tmp = mu_ee * xp.diag(energy_U[i,j,:,:])
            # mu_U[:,:] = xp.diag(mu_U_tmp)
            mu_U[:,:] = xp.diag(mu_ee) @ energy_U[i,j,:,:]

        if (j == self.Ns - 1):
            mu_U -= xp.diag(mu[:,i] / nb)

        return mu_U


    def diffusivity(self, i, energy, mu, nb, Te, Tg, EinsteinForm, EinsteinFormIon = False):
        xp  = self.xp_module

        DEf = xp.zeros((energy.shape[0],1),dtype=xp.float64)

        if (len(self.diffusivityList) > i and self.diffusivityList[i].interpolate):
            indFix = (energy[:,0]<=0.0)
            energy[indFix,0] = 0.0

            indFix = (energy[:,0]>10.0)
            energy[indFix,0] = 10.0

            DEf[:,0] = self.diffusivityList[i].D_expression((2./3)*energy[:,i]) / nb

        elif EinsteinForm and i == 0:
            V0 =  self.qStar * 1.0 # V0 = qStar * 1eV
            DEf[:,0] = 2.0 / 3.0 * xp.multiply(Te, mu[:,i]) / V0

        elif EinsteinFormIon and self.Z[i] != 0 and i != 0:
            V0 =  self.qStar * 1.0 # V0 = qStar * 1eV
            DEf[:,0] = 2.0 / 3.0 * xp.multiply(Tg, mu[:,i]) / V0

        else:
            DEf[:,0] = self.D[i] / nb

        return DEf[:,0]

    def diffusivity_U(self, i, j, energy, energy_U, mu, D, nb, Te, Te_U, Tg, Tg_U, EinsteinForm, EinsteinFormIon = False):
        xp  = self.xp_module

        D_U = xp.zeros((energy_U.shape[2], energy_U.shape[2]),dtype=xp.float64)


        if (len(self.diffusivityList) > i and self.diffusivityList[i].interpolate):
            indFixL = (energy[:,i]<=0.0)
            energy[indFixL,i] = 0.0
            energy_U[i,j,indFixL,:] = 0.0

            indFixH = (energy[:,i]>10.0)
            energy[indFixH,i] = 10.0
            energy_U[i,j,indFixH,:] = 0.0

            D_ee = (2./3)*self.diffusivityList[i].D_T_expression((2./3)*energy[:,i]) / nb
            # D_U_tmp = D_ee * xp.diag(energy_U[i,j,:,:])
            # D_U[:,:] = xp.diag(D_U_tmp)
            D_U[:,:] = xp.diag(D_ee) @ energy_U[i,j,:,:]


        elif EinsteinForm and i == 0:
            Imat = xp.identity(Te_U.shape[0])
            V0 =  self.qStar * 1.0 # V0 = qStar * 1eV

            D_U[:,:] = 2.0 / 3.0 * np.diag(mu[:,i]) @ xp.multiply(Imat,Te_U[:,j]) / V0

            if (len(self.mobilityList) > i and self.mobilityList[i].interpolate):
                mu_ee = (2./3)*self.mobilityList[i].mu_T_expression((2./3)*energy[:,i]) / nb
                D_U[:,:] += xp.diag(mu_ee * 2.0 / 3.0 * Te / V0) @ energy_U[i,j,:,:]

        elif EinsteinFormIon and self.Z[i] != 0 and i != 0:
            Imat = xp.identity(Te_U.shape[0])
            V0 =  self.qStar * 1.0 # V0 = qStar * 1eV
            D_U[:,:] = 2.0 / 3.0 * np.diag(mu[:,i]) @ xp.multiply(Imat,Tg_U[:,j]) / V0

            if (len(self.mobilityList) > i and self.mobilityList[i].interpolate):
                mu_ee = (2./3)*self.mobilityList[i].mu_T_expression((2./3)*energy[:,i]) / nb
                D_U[:,:] += xp.diag(mu_ee * 2.0 / 3.0 * Tg / V0) @ energy_U[i,j,:,:]

        if (j == self.Ns - 1):
            D_U[:,:] -= xp.diag(D[:,i] / nb)


        return D_U



    def rxnSourceTerm(self, energy, density):
        xp  = self.xp_module

        G = self.progressRate(energy,density)

        omega = xp.zeros((energy.shape[0], self.Ns+1),dtype=xp.float64)
        for i in range(0,self.Ns):
            for j in range(0,self.Nr):
                omega[:,i] += (self.reactionsList[j].rxnBeta[i,0] - self.reactionsList[j].rxnAlfa[i,0])*G[:,j]

        for j in range(0,self.Nr):
            omega[:,self.Ns] -= self.dH[j]*G[:,j]

        return omega

    def rxnSourceTermJac(self, energy, density):
        xp  = self.xp_module

        G_U = self.progressRateJac(energy,density) # For each reaction, we have the derivatives wrt species at all locations

        omega_U = xp.zeros((self.Ns+1,self.Ns+1,energy.shape[0]),dtype=xp.float64)
        for i in range(0,self.Ns):
            for j in range(0,self.Nr):
                omega_U[i,:,:] += (self.reactionsList[j].rxnBeta[i,0] - self.reactionsList[j].rxnAlfa[i,0])*G_U[j,:,:]

        for j in range(0,self.Nr):
            omega_U[self.Ns,:,:] -= self.dH[j]*G_U[j,:,:]

        return omega_U

    def progressRate(self, energy, density):
        xp  = self.xp_module

        G = xp.zeros((energy.shape[0],self.Nr))
        for i in range(0,self.Nr):
            kf = self.rxnRateCoefficient(energy, i)
            G[:,i] = kf[:,0]
            for j in range(0,self.Ns):
                if (self.reactionsList[i].rxnAlfa[j,0]>0):
                    G[:,i] *= density[:,j]**self.reactionsList[i].rxnAlfa[j,0]

        return G

    def progressRateJac(self, energy, density):
        xp  = self.xp_module

        G = self.progressRate(energy,density)
        G_U = xp.zeros((self.Nr,self.Ns+1, energy.shape[0]))
        for i in range(0,self.Nr):
            kf   = self.rxnRateCoefficient(energy, i)
            kf_T = self.rxnRateCoefficientJac(energy,i)

            #G[:,i] *= density[:,j]**self.reactionsList[i].rxnAlfa[j,0]
            G_U[i,self.Ns,:] = kf_T[:,0]

            for k in range(0,self.Ns):
                #G_U[i,k,:] = self.reactionsList[i].rxnAlfa[k,0]*G[:,i]/density[:,k]
                if (self.reactionsList[i].rxnAlfa[k,0]==0):
                    G_U[i,k,:] = 0
                else:
                    G_U[i,k,:] = kf[:,0]
                    for j in range(0,self.Ns):
                        if (j==k):
                            G_U[i,k,:] *= self.reactionsList[i].rxnAlfa[k,0]*density[:,k]**(self.reactionsList[i].rxnAlfa[j,0]-1)
                        else:
                            G_U[i,k,:] *= density[:,j]**self.reactionsList[i].rxnAlfa[j,0]

                G_U[i,self.Ns,:] *= density[:,k]**self.reactionsList[i].rxnAlfa[k,0]

            #G_U[i,self.Ns,:] = kf_T[:,0]*G[:,i]/kf[:,0]

        return G_U

    def rxnRateCoefficient(self, energy, i):
        """Returns ionization reaction rate constant"""
        xp  = self.xp_module

        indFix = (energy[:,0]<=0.0)
        energy[indFix,0] = 1.0
        if self.reactionsList[i].rxnBolsig:
            kf = xp.exp(self.reactionsList[i].kf_log(xp.log(energy)))
        else:
            kf = self.reactionsList[i].kf(energy)
        kf[indFix,0] = 0

        return kf #a * (energy**b) * xp.exp(-Ea/energy)

    def rxnRateCoefficientJac(self, energy, i):
        """Returns derivative of ionization reaction rate constant wrt
        energy
        """
        xp  = self.xp_module

        indFix = (energy[:,0]<=0.0)
        energy[indFix,0] = 1.0
        if self.reactionsList[i].rxnBolsig:
            kf_T = self.reactionsList[i].kf_T_log(xp.log(energy)) \
                 * xp.exp(self.reactionsList[i].kf_log(xp.log(energy))) / energy
        else:
            kf_T = self.reactionsList[i].kf_T(energy)
        kf_T[indFix,0] = 0

        return kf_T #a * (energy**(b-1)) * xp.exp(-Ea/energy) * (b + Ea/energy)


    def print(self):
        """Print parameters to the screen"""
        print("# The non-dimensional transport and chemstry properties are")
        print("#   De    = {0:.6e}".format(self.D[0]))
        print("#   Di    = {0:.6e}".format(self.D[1]))
        print("#   mue   = {0:.6e}".format(self.mu[0]))
        print("#   mui   = {0:.6e}".format(self.mu[1]))
        print("#   A  = {}".format(self.A))
        print("#   B  = {}".format(self.B))
        print("#   C  = {}".format(self.C))
        print("#   dH = {}".format(self.dH))
        print("# alfa = {}".format(self.alfa))
        print("# beta = {}".format(self.beta))
        print("#   qStar = {0:.6e}".format(self.qStar))
        print("#   alpha = {0:.6e}".format(self.alpha))
        print("#   ks    = {0:.6e}".format(self.ks))
        print("#   gam   = {0:.6e}".format(self.gam))



class timeDomainCollocationSolver:
    """Provides ability to solve a drift-diffusion approximation of an RF
    glow discharge device using a Chebyshev-collocation approach in
    space coupled with a backward-difference-formula (BDF1 or BDF2) in
    time.

    Attributes:
        Ns   -- Number of species
        NT   -- Number of temperatures
        Nv   -- Number of state variables (Ns+NT)
        Np   -- Number of Chebyshev DOFs per state variable
        deg  -- Degree of Chebyshev polynomials (Np-1)
        Ndof -- Total number of degrees of freedom

        U0, U1, U2 -- State vectors necessary for BDF2 time step
        phi        -- Electric potential

        params -- modelClosures class (provides model parameters)

        xp -- Gauss-Chebyshev-Lobatto points corresponding to
              Chebyshev polynomials of degree deg.  The state vectors
              U0,U1,U2 hold the value of the state at these points

        xc -- Collocation points (allowed to different from xp for now)
    """

    def __init__(self, args, Ns, NT, Np, elasticCollisionActivationFactor,
                 backgroundSpecieActivationFactor, EinsteinForm, IonEffEField,
                 gam=0.01, V0 = 100.0, VDC = 0.0,
                 scenario=0, scheme="BE", iSample = 0):
        """Initializes storage and operaters required for solve."""

        self.args      = args
        self.xp_module = np

        # parameters of the time marching scheme
        self.temporal_scheme = scheme  
        
        if not (self.temporal_scheme in ["BE", "CN", "LCN"]):
            print("ERROR: Unrecognized temporal scheme.")
            print("Please use 'BE' (backward Euler), 'CN' (Crank-Nicolson), or 'LCN' (linearized Crank-Nicolson).")
            exit(-1)
        self.solveCRModel = False
        
        
        # Indexing 
        # i = 0       -> electrons 
        # i = 1       -> ions 
        # i = 2       -> argon molecular ions 
        # i = 3       -> argon excimer molecules 
        # i = 4:Ns-1  -> excited levels
        # i = Ns - 1  -> ground state
        # i = Ns      -> electron energy
        # i = Nv - 1  -> ion Effective Electric field

        self.Ns = Ns        # Number of species
        self.NT = NT        # Number of temperatures
        self.Nv = Ns+NT     # Total number of 'state' variables
        if IonEffEField:
            self.Nv = self.Nv + 1 # One more state variable for the effective electric field for Ions

        self.Nion = 1 # TODO(trevilo): don't hardcode this!

        self.deg = Np-1 # degree of Chebyshev polys we use
        self.Np = Np # Number of points used to define state in space
        self.Nc = Np-2 # number of collocation pts (Np-2 b/c BCs)

        self.Ndof = self.Nv*self.Np # total number of dofs

        # state (3 vectors for BDF2)
        self.U2 = np.zeros((self.Ndof,1),dtype=np.float64)
        self.U1 = np.zeros((self.Ndof,1),dtype=np.float64)
        self.U0 = np.zeros((self.Ndof,1),dtype=np.float64)

        # electric potential (not part of state b/c we solve for it
        # given state)
        self.phi = np.zeros((self.Np,1))

        # closures
        if(scenario==0):
            Nr = 1
        elif(scenario==1):
            Nr = 1
        elif(scenario==2):
            Nr = 8
        elif(scenario==3):
            Nr = 8
        elif(scenario==4):
            Nr = 9
        elif(scenario==5):
            Nr = 7
        elif(scenario==6):
            Nr = 23
        elif(scenario==8):
            Nr = 23
        elif(scenario==9):
            Nr = 23
        elif(scenario==10):
            Nr = 34
        elif(scenario==12):
            Nr = 23
        elif(scenario==13):
            Nr = 23
        elif(scenario==14):
            Nr = 23
        elif(scenario==21):
            Nr = 8
        elif(scenario==7):
            # Nr = 34
            Nr = 46
            self.Nion = 2
        elif(scenario==15 or scenario==16):
            Nr = 0 # It is evaluated within the model based on the number of species you include.            
            self.solveCRModel = True
            self.Nion = 2
            elasticCollisionActivationFactor = 0.0 # Elastic collision term is being handled within the CR model.
            print("#   Elastic collision term is being handled within the CR model. ")
            # if self.temporal_scheme != "BE" or self.temporal_scheme != "CN":
            #     print("ERROR: Curently we suppport only the 'BE' and 'CN' temporal schemes for the CR model.")
            #     exit(-1)
        else:
            print("ERROR: scenario = {} not understood.".format(scenario))
            exit(-1)

        self.elasticCollisionActivationFactor = elasticCollisionActivationFactor
        self.backgroundSpecieActivationFactor = backgroundSpecieActivationFactor
        self.EinsteinForm = EinsteinForm
        self.EinsteinFormIon = args.EinsteinFormIon
        self.IonEffEField = IonEffEField

        self.params = modelClosures(self.Ns, Nr)

        if(scenario==0):
            setLiu2014Properties(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==1):
            setPsaapProperties(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==2):
            setPsaapPropertiesTestArm(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==3):
            setPsaapPropertiesCurrentTestCase(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==4):
            setPsaapProperties_4Species_Nominal(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==5):
            setPsaapPropertiesWithSampling(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==6):
            setPsaapProperties_6Species_Sampling(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==8):
            setPsaapProperties_6Species_Sampling_500mTorr(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==9):
            setPsaapProperties_6Species(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==10):
            setPsaapProperties_6Species_100mTorr_Expanded(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==12):
            setPsaapProperties_6Species_Nominal(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==13):
            setPsaapProperties_6Species_500mTorr(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==14):
            setPsaapProperties_6Species_1Torr_Expanded(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==21):
            setPsaapPropertiesTestArmInterpTrans(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==7):
             setPsaapProperties_6Species(gam, V0, VDC, self.params, Ns, Nr, iSample)
        elif(scenario==15 or scenario==16):
            setPsaapProperties_CRModel(gam, V0, VDC, self.params, Ns)
            self.params.electron_energy_dirichlet = False
            self.cr = CollisionalRadiativeModel(self.args, Ns, NT, self.Np, \
                self.params.Pressure, self.params.GasTemperature, backgroundSpecieActivationFactor)
            self.params.dEps[0] = 0.0; self.params.dEps[1] = 15.7596119; self.params.dEps[self.Ns-1] = 0.0
            self.params.dEps[2] = 14.501; self.params.dEps[3] = 11.564763 # I need to re-evaluate the last one!
            self.params.dEps[4:self.Ns-1] = self.cr.p.E_lvl[1:self.Ns-4]*CRconst.cm_eV;

        # Points used to define state and collocation
        # (Gauss-Lobatto-Chebyshev points)
        self.xp = -np.cos(np.pi*np.linspace(0,self.deg,self.Np)/self.deg)

        # Jacobian storage
        self.jac  = np.zeros((self.Ndof, self.Ndof))
        self.jac0 = np.zeros((self.Ndof, self.Ndof))

        self.A1 = np.zeros((self.Ndof, self.Ndof))
        self.A0 = np.identity(self.Ndof)
        self.rhsSens = np.zeros((self.Ndof, self.Ndof))

        # Operators
        ident = np.identity(self.Np)

        # V0p: Coefficients to values at xp
        self.V0p = cheb.chebvander(self.xp, self.deg)

        # V0pinv: xp values to coefficients
        self.V0pinv = np.linalg.solve(self.V0p, ident)

        # V1p: coefficients to derivatives at xp
        self.V1p = np.zeros((self.Np,self.Np))
        for i in range(0,self.Np):
            self.V1p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=1))

        # Dp: values at xp to derivatives at xp
        self.Dp = self.V1p @ self.V0pinv

        # V2p: coefficients to 2nd derivatives at xp
        self.V2p = np.zeros((self.Np,self.Np))
        for i in range(0,self.Np):
            self.V2p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=2))

        # Lp: values at xp to 2nd derivatives at xp
        self.Lp = self.V2p @ self.V0pinv

        # LpD: values at xp to 2nd derivatives at xc, with identity
        # for top and bottom row (for Dirichlet BCs)
        self.LpD            = np.identity(self.Np)
        self.LpD[1:-1,:]    = self.Lp[1:-1,:]
        self.LpD_inv        = np.linalg.solve(self.LpD, np.eye(self.Np)) 


        # solve poisson equation for phi_ne
        ident0 = np.identity(self.Np)
        ident0[0,0] = ident0[-1,-1] = 0.0
        self.phi_n  = np.dot(self.LpD_inv, -self.params.alpha*ident0)       
        # self.phi_ni = np.dot(self.LpD_inv, -self.params.alpha*ident0)
        # self.phi_ne = -self.phi_ni

        # self.phi_x_ne = self.Dp @ self.phi_ne
        # self.phi_x_ni = self.Dp @ self.phi_ni
        self.phi_x_n = self.Dp @ self.phi_n



        self.I_Np =  np.identity(self.Np)
        self.I_Ndof = np.identity(self.Ndof)

        self.ones_Np =  np.ones(self.Np)

        self.totalCurrent    = np.zeros((2,1),dtype=np.float64)
        self.electronCurrent = np.zeros((2,1),dtype=np.float64)
        self.ionCurrent      = np.zeros((2,1),dtype=np.float64)
        

        self.electricField = np.zeros((self.Np,1),dtype=np.float64)
        self.electricPotential = np.zeros((self.Np,1),dtype=np.float64)
        if IonEffEField:
            self.effElectricField = np.zeros((self.Np,1),dtype=np.float64)
        
        
        self.ntot_U = np.zeros((self.Np, self.Nv)) 
        # all but background
        for i in range(1, self.Ns-1):
            self.ntot_U[:,i] += self.ones_Np
        # background contribution
        self.ntot_U[:,self.Ns-1] += self.params.nAronp0*self.ones_Np
    
        self.TwoOverThree = 2.0/3.0     
        self.ThreeOverTwo = 3.0/2.0  

        self.dt_adaptive = 0.0


        
    

    def copy_operators_H2D(self, dev_id):
      
      if self.args.use_gpu==0:
        return
      
      with cp.cuda.Device(dev_id):


        self.U2                 = cp.asarray(self.U2)
        self.U1                 = cp.asarray(self.U1)
        self.U0                 = cp.asarray(self.U0)

        self.totalCurrent       = cp.asarray(self.totalCurrent)
        self.ionCurrent         = cp.asarray(self.ionCurrent)
        self.electronCurrent    = cp.asarray(self.electronCurrent)

        self.electricField      = cp.asarray(self.electricField)
        self.electricPotential  = cp.asarray(self.electricPotential)
        self.effElectricField   = cp.asarray(self.effElectricField)
        
        
        self.Dp                 = cp.asarray(self.Dp)
        self.LpD                = cp.asarray(self.LpD)
        self.LpD_inv            = cp.asarray(self.LpD_inv)

        # self.phi_x_ne           = cp.asarray(self.phi_x_ne)
        # self.phi_x_ni           = cp.asarray(self.phi_x_ni)
        self.phi_x_n            = cp.asarray(self.phi_x_n)

        self.I_Np               = cp.asarray(self.I_Np)
        self.I_Ndof             = cp.asarray(self.I_Ndof)

        self.ones_Np             = cp.asarray(self.ones_Np)
        self.ntot_U             = cp.asarray(self.ntot_U)


      return
    
    def copy_operators_D2H(self, dev_id):
      
      if self.args.use_gpu==0:
        return
      
      with cp.cuda.Device(dev_id):

        self.U2             = cp.asnumpy(self.U2)
        self.U1             = cp.asnumpy(self.U1)
        self.U0             = cp.asnumpy(self.U0)   
        
        self.Dp             = cp.asnumpy(self.Dp)

        
      return  


    def filter(self):
        """Filter state by zeroing out the last Chebyshev coefficient.
        This feature is experimental.
        """
        ind = int(self.Np-1)

        U0 = self.V0pinv @ self.U2[0:self.Np]

        U0[ind:] = 0.0
        self.U2[0:self.Np] = self.V0p @ U0

        U0 = self.V0pinv @ self.U2[self.Np:2*self.Np]
        U0[ind:] = 0.0
        self.U2[self.Np:2*self.Np] = self.V0p @ U0

        U0 = self.V0pinv @ self.U2[2*self.Np:]
        U0[ind:] = 0.0
        self.U2[2*self.Np:] = self.V0p @ U0

    def solve_poisson(self, dens,time):
        xp = self.xp_module

        """Solve Gauss' law for the electric potential.

        Inputs:
          ne   : Values of electron density at xp
          ni   : Values of ion density at xp
          time : Current time

        Outputs: None (sets self.phi to computed potential)
        """

        r = xp.zeros((self.Np, 1),dtype=xp.float64)

        for i in range(self.Ns):
            if self.params.charge(i) != 0:
                r += self.params.charge(i) * dens[:,[i]] 
        
        r *= -self.params.alpha  
        
        # r = -self.params.alpha*(ni-ne)
        r[0] = 0.0
        r[-1] = xp.sin(2*xp.pi*time) + self.params.verticalShift
        # self.phi = xp.linalg.solve(self.LpD, r)
        self.phi = xp.dot(self.LpD_inv, r)


    def spatial_residual(self, Uin, time, dt, weak_bc=False):
        """Evaluates the residual.

        Inputs:
          Uin  : Current state
          time : Current time
          dt   : Time step

        Outputs:
          returns residual vector

        Notes:
          This function currently assumes that Ns=2 and NT=1
        """
        xp = self.xp_module

        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        iele  = [0]             # Electrons
        iion  = list(range(1, 1 + self.Nion)) # Ions Ar+, Ar2+
        inb   = [self.Ns-1]     # Background species
        iee   = [self.Ns]       # Electron Energy
        # iEeff = [self.Nv -1]    # Ion Effective Electric field

     
        
        # pull off state for convenience
        dens = xp.zeros((self.Np, self.Ns),dtype=xp.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = xp.zeros((self.Np, 1),dtype=xp.float64)
        nT = Uin[self.Ns*self.Np:(self.Ns+1)*self.Np] # assumes just 1 temperature!
        Te = nT/dens[:,iele]
  
        if self.IonEffEField: 
            Eeff = Uin[(self.Nv-1)*self.Np:(self.Nv)*self.Np] # Effective electric field for argon ions


        ntot = xp.zeros((self.Np, 1),dtype=xp.float64)
        # add all heavies but background
        for i in range(1, self.Ns-1):
            ntot[:,0] += dens[:,i]

        # add background contribution (accounting for non-dim difference)
        ntot[:,0] += self.params.nAronp0 * dens[:,inb[0]]

        # Temperature (from ideal gas law)
        Tg = xp.zeros((self.Np, 1),dtype=xp.float64)
        Tg = (self.params.p0 - nT)/ntot

        # NOTE(malamast): I clip the electron temperature when a low value occurs.
        Te = np.where(Te < Tg,Tg, Te) 

          
        # solve poisson equation for phi
        # now have self.phi
        self.solve_poisson(dens,time)

        # Form fluxes at grid points
        dens_x = self.Dp @ dens
        nT_x   = self.Dp @ nT
        phi_x  = self.Dp @ self.phi

        # Reduced Electric field  E / N 
        EN =  xp.abs(- phi_x)  / dens[:,inb] 
        # EN_Td = 1e21 * xp.abs(- phi_x) * self.params.V0L  / (self.params.nAr * dens[:,inb]) #  Electric field / N [Td]   
        if self.IonEffEField:
            ENion =  xp.abs(Eeff)  / dens[:,inb] 
        else:
            ENion =  EN


        # Temperature of species
        energy = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64) #NOTE(malamast): We now have self.Ns+1 instead of Ns

        # Initialize all energies to heavy species...
        for i in range(0,self.Ns):
            energy[:,i] = Tg[:,0] # For species

        # and overwrite for some
        energy[:,iele[0]] = Te[:,0] # Electrons

        for i in range(0, self.Nion):
          energy[:,iion[i]] = ENion[:,0] # For ions

        energy[:,iee[0]] = Te[:,0] # Electron energy

        # Transport Properties for species
        mu              = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64)
        diffusivity     = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64)
        
        for i in range(0,self.Ns+1):
            mu[:,i]  = self.params.mobility(i, energy, dens[:,inb[0]])
            diffusivity[:,i] = self.params.diffusivity(i, energy, mu, dens[:,inb[0]],
                                                       Te[:,0], Tg[:,0], self.EinsteinForm, self.EinsteinFormIon)

        if self.EinsteinForm:
            mu[:,iee[0]] = (5./3.) * mu[:,iele[0]] 
            diffusivity[:,iee[0]] = (5./3.) * diffusivity[:,iele[0]] 
            

        # Form species fluxes
        fspec = xp.zeros((self.Np, self.Ns),dtype=xp.float64)
        for i in range(0,self.Ns):
            if self.params.charge(i) != 0.0:
                if self.IonEffEField and self.params.charge(i) > 0:
                    fspec[:,i] += self.params.charge(i)*xp.multiply(mu[:,i], dens[:,i])*(Eeff[:,0])
                else: 
                    fspec[:,i] += self.params.charge(i)*xp.multiply(mu[:,i], dens[:,i])*(-phi_x[:,0])
            fspec[:,i] -= xp.multiply(diffusivity[:,i],dens_x[:,i])  

        # Form electron energy flux                        
        # (See G J M Hagelaar and L C Pitchford 2005 Plasma Sources Sci. Technol. 14 722)
        fT = xp.zeros((self.Np, 1),dtype=xp.float64)
        #NOTE(malamas): Do I need to add the dDe/dx part here? The Bolsig paper reports De is within the derivative.
        fT[:,0] = -mu[:,iee[0]]*nT[:,0]*(-phi_x[:,0]) -  xp.multiply(diffusivity[:,iee[0]], nT_x[:,0])

        if (weak_bc):
        # overwrite endpoints in fi (weakly impose BC)
        # in 1D, the unit normal vector is nx = −1 on the ’left’ and nx = 1 on the ’right’ boundary

            if self.IonEffEField:
                for i in range(0, self.Nion):
                    fspec[ 0,iion[i]] = -self.params.ksion * dens[ 0,iion[i]] + mu[ 0,iion[i]] * dens[ 0,iion[i]] * ( max(Eeff[ 0,0]*(-1),0.0) ) * (-1)
                    fspec[-1,iion[i]] =  self.params.ksion * dens[-1,iion[i]] + mu[-1,iion[i]] * dens[-1,iion[i]] * ( max(Eeff[-1,0],0.0) )
            else:
                for i in range(0, self.Nion):
                    fspec[ 0,iion[i]] = -self.params.ksion * dens[ 0,iion[i]] + mu[ 0,iion[i]] * dens[ 0,iion[i]] * ( max(-phi_x[ 0,0]*(-1),0.0) ) * (-1)
                    fspec[-1,iion[i]] =  self.params.ksion * dens[-1,iion[i]] + mu[-1,iion[i]] * dens[-1,iion[i]] * ( max(-phi_x[-1,0],0.0) )

        # Strong BC equations
        rstrg = xp.zeros(8)

        fionL = 0.0
        fionR = 0.0
        for i in range(0, self.Nion):
            fionL += fspec[ 0, iion[i]]
            fionR += fspec[-1, iion[i]]

        rstrg[0] = fspec[ 0,iele[0]] - (- self.params.ks * dens[ 0,iele[0]] * Te[ 0,0]**0.5 - self.params.gam * fionL )
        rstrg[1] = fspec[-1,iele[0]] - (+ self.params.ks * dens[-1,iele[0]] * Te[-1,0]**0.5 - self.params.gam * fionR )

        if self.IonEffEField:
            for i in range(0, self.Nion):
                rstrg[2 + 2*i] = fspec[ 0,iion[i]] - (- self.params.ksion*dens[ 0,iion[i]] + mu[ 0,iion[i]] * dens[ 0,iion[i]] * max(Eeff[ 0,0]*(-1),0.0) * (-1) )
                rstrg[3 + 2*i] = fspec[-1,iion[i]] - (+ self.params.ksion*dens[-1,iion[i]] + mu[-1,iion[i]] * dens[-1,iion[i]] * max(Eeff[-1,0],0.0) )
        else:
            for i in range(0, self.Nion):
                rstrg[2 + 2*i] = fspec[ 0,iion[i]] - (- self.params.ksion*dens[ 0,iion[i]] + mu[ 0,iion[i]] * dens[ 0,iion[i]] * max(-phi_x[ 0,0]*(-1),0.0) * (-1) )
                rstrg[3 + 2*i] = fspec[-1,iion[i]] - (+ self.params.ksion*dens[-1,iion[i]] + mu[-1,iion[i]] * dens[-1,iion[i]] * max(-phi_x[-1,0],0.0) )


        rstrg[6] = fT[ 0,0] - (- (5./3.)*self.params.ks * Te[ 0,0]**0.5 * nT[ 0,0] - self.params.gam * fionL * self.params.EeBC )
        rstrg[7] = fT[-1,0] - (+ (5./3.)*self.params.ks * Te[-1,0]**0.5 * nT[-1,0] - self.params.gam * fionR * self.params.EeBC )

        # form derivatives of fluxes at collocation points
        fspec_x = self.Dp @ fspec

        # form derivative of electron energy flux at collocation points
        fT_x = self.Dp @ fT

        # Radiation heating (used in the estimation of the S field)
        Qrad = xp.zeros((self.Np,1),dtype=xp.float64)

        # Form reaction source terms at collocation points
        if (self.solveCRModel):
                        
            vars_CR = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64)                
            # ground state. background species is first in the CR model arrangement            
            vars_CR[:,0:self.Ns] = dens[:, self.cr.FromGlowDischargeToCRIndexing[0:-1]]*self.params.np0
            vars_CR[:,0] *= self.params.nAronp0
            vars_CR[:,self.Ns] = Te[:,0] * self.params.TwoOverThree  # electron temperature [eV]

            # omega_CR = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64)                          
            self.cr.rxnSourceTerm_UpdateTemperatureDependentPart_vec(vars_CR)
            omega_CR, Qrad_CR = self.cr.rxnSourceTerm_vec(vars_CR)
            # self.cr.rxnSourceTerm_Update_vec(vars_CR)
            # omega_CR = self.cr.dydt_saved

            omega = xp.zeros_like(omega_CR)
            omega = omega_CR[:, self.cr.FromCRToGlowDischargeIndexing] * self.params.tauOvernp0
            omega [:,self.Ns-1] /=  self.params.nAronp0  
            Qrad[:,0] = Qrad_CR * self.params.tauOvernp0   
            
        else:
            omega = self.params.rxnSourceTerm(Te, dens)
        
        # Joule Heating Term    
        # Mind that the joule heating term involves fspec[:,iele] and not fT[:,0] 
        # (See G J M Hagelaar and L C Pitchford 2005 Plasma Sources Sci. Technol. 14 722)
        SJ = -self.params.qStar*fspec[:,iele]*(-phi_x) 

        # Elastic collision term at collocation points
        SEC  = -self.params.EC * (nT - xp.multiply(dens[:, iele], Tg)) 
        SEC *= self.elasticCollisionActivationFactor
        
        

        
        # evaluate S---the source term required in the background
        # species evolution to ensure constant pressure
        fa = xp.copy(fT) # Here we include the contribution from electrons
        
        for i in range(1,self.Ns-1): 
            CpOverCv = self.params.Cp[i]/self.params.Cv[i]            
            if self.params.charge(i) != 0:                
                if self.IonEffEField and self.params.charge(i) > 0:
                    fa[:,0] += CpOverCv * (self.params.charge(i) * xp.multiply(mu[:,i], 
                                xp.multiply(dens[:,i], self.params.Cv[i] * self.TwoOverThree*Tg[:,0] )  * (Eeff[:,0])))
                else:
                    fa[:,0] += CpOverCv * (self.params.charge(i) * xp.multiply(mu[:,i],
                                xp.multiply(dens[:,i],self.params.Cv[i] * self.TwoOverThree*Tg[:,0]) * (-phi_x[:,0])))
            fa[:,0] -= CpOverCv * (xp.multiply(diffusivity[:,i], 
                        (self.Dp @ xp.multiply(dens[:,i],self.params.Cv[i] * self.TwoOverThree*Tg[:,0])))) 
            
        # background thermal conductivity contribution
        fa[:,0] += - self.params.kappaB * (self.Dp @ ( self.params.Cv[inb[0]] * self.TwoOverThree*Tg[:,0])) 

        fa_x = self.Dp @ fa

        sOmEp = xp.zeros((self.Np,1),dtype=xp.float64)
        for i in range(0, self.Ns-1):
            sOmEp[:,0] += omega[:,i]*self.params.dEps[i] 


        joule = xp.zeros((self.Np,1),dtype=xp.float64)
        for i in range(0, self.Ns-1): 
            if self.params.charge(i) != 0:
                if self.IonEffEField and self.params.charge(i) > 0:
                    joule[:,0] += self.params.qStar*self.params.charge(i)*fspec[:,i]*(Eeff[:,0])        
                else:
                    joule[:,0] += self.params.qStar*self.params.charge(i)*fspec[:,i]*(-phi_x[:,0])        

        S = xp.zeros((self.Np,1),dtype=xp.float64)
        # NOTE(malamast): The species fluxes -fspec_x * dEps cancel out with some terms in -dq/dx
        S[:,0] = (sOmEp[:,0] + fa_x[:,0] - joule[:,0] -Qrad[:,0]) / \
                    (self.params.Cv[inb[0]] * self.TwoOverThree*Tg[:,0])/self.params.nAronp0



        # form full residual
        res = xp.zeros((self.Nv*self.Np,1))

        # spatial part
        # standard species
        for i in range(0,self.Ns-1):
            res[i*self.Np:(i+1)*self.Np,0] = dt*(fspec_x[:,i] - omega[:,i])

        # background species (fixed at IC for now)
        res[(self.Ns-1)*self.Np:self.Ns*self.Np] = -dt*S
        res[(self.Ns-1)*self.Np:self.Ns*self.Np,0] *= self.backgroundSpecieActivationFactor

        #  Electron Energy
        res[self.Ns*self.Np:(self.Ns+1)*self.Np]        = dt*(fT_x - omega[:,iee] - SJ  - SEC)


        #  Effective electric field for ions
        if self.IonEffEField:
            res[(self.Nv-1)*self.Np:self.Nv*self.Np] = -dt * self.params.vmStar * (-phi_x - Eeff) / mu[:,[iion[0]]]
        
        

        ############################################################
        # Computation of total, displacement, and particle current #
        ############################################################
        self.electricField[:]     = - phi_x * self.params.V0L
        self.electricPotential[:] = self.phi * self.params.qStar
        if self.IonEffEField:
            self.effElectricField[:]     = Eeff * self.params.V0L

        
        E_currentTimeStep = - phi_x
 
        # pull off state for convenience
        dens_previousTimeStep = xp.ndarray((self.Np, self.Ns),dtype=xp.float64)
        for i in range(0,self.Ns):
            dens_previousTimeStep[:,i] = self.U1[i*self.Np:(i+1)*self.Np,0]

        # solve poisson equation for phi
        # now have self.phi
        self.solve_poisson(dens_previousTimeStep,time)

        # form fluxes at grid points
        E_previousTimeStep  = -self.Dp @ self.phi

        displacementCurrent = self.params.eps0 \
            * (E_currentTimeStep - E_previousTimeStep) / dt * self.params.V0Ltau

        particleCurrent = xp.zeros((2,self.Ns),dtype=xp.float64)
        particleCurrent[0,iion[0]]  = mu[0,1] * self.params.LLV0tau \
            * dens[ 0,iion[0]] * self.params.np0 * (-phi_x[0,0]) * self.params.V0L \
                * self.params.qe * self.params.charge(1)
        particleCurrent[-1,iion[0]] = mu[-1,1] * self.params.LLV0tau \
            * dens[-1,iion[0]] * self.params.np0 * (-phi_x[-1,0]) * self.params.V0L \
                * self.params.qe * self.params.charge(1)

        particleCurrent[0,iele[0]]  = (- self.params.ks * self.params.tauL \
            *  dens[ 0,iele[0]] * self.params.np0 * Te[0,0]**0.5 * self.params.qe \
            - self.params.gam * particleCurrent[0,iion[0]]) * self.params.charge(0) #NOTE(malamast): Why do we multiply with the charge here?
        particleCurrent[-1,iele[0]] = (+ self.params.ks * self.params.tauL \
            * dens[-1,iele[0]] * self.params.np0 * Te[-1,0]**0.5 * self.params.qe \
            - self.params.gam * particleCurrent[-1,iion[0]]) * self.params.charge(0) #NOTE(malamast): Why do we multiply with the charge here?

        self.totalCurrent[ 0,0] = (displacementCurrent[ 0,0] \
            + particleCurrent[ 0,iion[0]] + particleCurrent[ 0,iele[0]]) * self.params.eArea
        self.totalCurrent[-1,0] = (displacementCurrent[-1,0] \
            + particleCurrent[-1,iion[0]] + particleCurrent[-1,iele[0]]) * self.params.eArea
            
        self.ionCurrent[:]      = particleCurrent[:,[iion[0]]] * self.params.eArea
        self.electronCurrent[:] = particleCurrent[:,iele] * self.params.eArea


        return res, rstrg

    def residual(self, Uin, time, dt, weak_bc=False):
        """Evaluates the residual.

        Inputs:
          Uin    : Current state
          time   : Current time
          dt     : Time step
          weak_bc: Weak electron flux BC flag (boolean)
          scheme : Temporal scheme indicator (string)

        Outputs:
          returns residual vector

        Notes:
          This function currently assumes that Ns=2 and NT=1
        """
        if (self.temporal_scheme=="BE"):
            res = self.residualBE(Uin, time, dt, weak_bc)
        elif (self.temporal_scheme=="CN"):
            res = self.residualCN(Uin, time, dt, weak_bc)
        else:
            print("Time marching scheme not recognized")
            exit(-1)

        return res

    def residualBE(self, Uin, time, dt, weak_bc=False):
        """Evaluates the residual for backward Euler.  See
        timeDomainCollocationSolver.residua() for additional documentation.
        """
        xp = self.xp_module

        res, rstrg = self.spatial_residual(Uin, time, dt, weak_bc)

        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        iele = [0]
        iion  = list(range(1, 1 + self.Nion)) # Ions Ar+, Ar2+

        # pull off state for convenience
        dens = xp.ndarray((self.Np, self.Ns),dtype=xp.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = Uin[self.Ns*self.Np:(self.Ns+1)*self.Np] # assumes just 1 temperature!
        # Te = nT/dens[:,iele]
        

        # time derivative part (backward Euler)
        res += Uin - self.U1

        # boundary conditions (strongly enforced)
        # electron flux
        res[0]           = rstrg[0]
        res[self.Np-1]   = rstrg[1]

        if (not weak_bc):
            # ion flux
            for i in range(0, self.Nion):
              res[iion[i] * self.Np]     = rstrg[2 + 2*i]
              res[(iion[i]+1)*self.Np-1] = rstrg[3 + 2*i]

        # electron temperature
        res[self.Ns*self.Np  ]     = rstrg[6]
        res[(self.Ns+1)*self.Np-1] = rstrg[7]

        for i in range(self.Nion+1,self.Ns-1):
            res[i*self.Np  ] = dens[ 0,i] - 0.0
            res[(i+1)*self.Np-1] = dens[-1,i] - 0.0

        # if solving for background density, enforce Dirichlet condition on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            ntot = xp.zeros(self.Np)

            # add all heavies but background
            for i in range(1, self.Ns-1):
                ntot += dens[:,i]

            # add background contribution (accounting for non-dim difference)
            #ntot += self.params.nAronp0 * dens[:,self.Ns-1]

            res[(self.Ns-1)*self.Np] = dens[  0,self.Ns-1] \
                - ((self.params.p0 - nT[  0]) / self.params.Tg0 - ntot[ 0]) / self.params.nAronp0
            res[ self.Ns*self.Np-1 ] = dens[ -1,self.Ns-1] \
                - ((self.params.p0 - nT[ -1]) / self.params.Tg0 - ntot[-1]) / self.params.nAronp0

        # electron temperature
        if (self.params.electron_energy_dirichlet):
            res[self.Ns*self.Np  ] = (nT[ 0] - self.params.EeBC * dens[0,iele])
            res[(self.Ns+1)*self.Np-1] = (nT[-1] - self.params.EeBC * dens[-1,iele])

        # No BC for Eeff (the effective electric field for ions)

        return res

    def residualCN(self, Uin, time, dt, weak_bc=False):
        """Evaluates the residual for Crank-Nicolson.  See
        timeDomainCollocationSolver.residua() for additional documentation.
        """
        xp = self.xp_module

        res0, rstrgold = self.spatial_residual(self.U1, time-dt, dt, weak_bc)
        res1, rstrg    = self.spatial_residual(    Uin, time   , dt, weak_bc)
        res = 0.5*(res0+res1)

        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        iele = [0]
        iion  = list(range(1, 1 + self.Nion)) # Ions Ar+, Ar2+

        # pull off state for convenience
        dens = xp.ndarray((self.Np, self.Ns),dtype=xp.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = Uin[self.Ns*self.Np:(self.Ns+1)*self.Np] # assumes just 1 temperature!
        # Te = nT/dens[:,iele]

        # time derivative part (backward Euler)
        res += Uin - self.U1

        # boundary conditions (strongly enforced)
        # electron flux
        res[0]           = rstrg[0]
        res[self.Np-1]   = rstrg[1]

        if (not weak_bc):
            # ion flux
            for i in range(0, self.Nion):
              res[iion[i] * self.Np]     = rstrg[2 + 2*i]
              res[(iion[i]+1)*self.Np-1] = rstrg[3 + 2*i]

        # electron temperature
        res[self.Ns*self.Np  ]     = rstrg[6]
        res[(self.Ns+1)*self.Np-1] = rstrg[7]

        for i in range(self.Nion+1,self.Ns-1):
            res[i*self.Np  ] = dens[ 0,i] - 0.0
            res[(i+1)*self.Np-1] = dens[-1,i] - 0.0

        # if solving for background density, enforce Dirichlet condition on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            ntot = xp.zeros(self.Np)

            # add all heavies but background
            for i in range(1, self.Ns-1):
                ntot += dens[:,i]

            # add background contribution (accounting for non-dim difference)
            #ntot += self.params.nAronp0 * dens[:,self.Ns-1]

            res[(self.Ns-1)*self.Np] = dens[  0,self.Ns-1] \
                - ((self.params.p0 - nT[  0]) / self.params.Tg0 - ntot[ 0]) / self.params.nAronp0
            res[ self.Ns*self.Np-1 ] = dens[ -1,self.Ns-1] \
                - ((self.params.p0 - nT[ -1]) / self.params.Tg0 - ntot[-1]) / self.params.nAronp0

        # electron temperature
        if (self.params.electron_energy_dirichlet):
            res[self.Ns*self.Np  ] = (nT[ 0] - self.params.EeBC * dens[0,iele])
            res[(self.Ns+1)*self.Np-1] = (nT[-1] - self.params.EeBC * dens[-1,iele])

        # No BC for Eeff (the effective electric field for ions)

        return res

    def residualLCN(self, Uin, time, dt, weak_bc=False):
        """Evaluates the residual for linearized Crank-Nicolson.  See
        timeDomainCollocationSolver.residua() for additional documentation.
        """
        xp = self.xp_module

        res0, rstrgold = self.spatial_residual(self.U1, time-dt, dt, weak_bc)
        res1, rstrg    = self.spatial_residual(    Uin, time   , dt, weak_bc)
        res = 0.5*(res0+res1)

        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        iele = [0]
        iion = [1]

        # pull off state for convenience
        dens = xp.ndarray((self.Np, self.Ns),dtype=xp.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = Uin[self.Ns*self.Np:(self.Ns+1)*self.Np] # assumes just 1 temperature!
        Te = nT/dens[:,iele]

        # time derivative part
        # no contribution from time derivative in LCN, so do nothing here

        # boundary conditions (strongly enforced)

        # electron flux
        if (not weak_bc):
            print("Error: Only weak electron flux BCs supported for linearized CN.")
            exit(-1)

        # NB: Setting residuals to zero here enforces that the correct
        # change in the associated variables is zero.  However, this
        # *only* works if the IC satisfies the BC.  Any errors
        # introduced by the IC will never be eliminated.
        #if (self.Ns>2):
        #    res[2*self.Np  ] = 0.0 #dens[ 0,2] - 0.0
        #    res[3*self.Np-1] = 0.0 #dens[-1,2] - 0.0

        if (self.Ns > 2):
            for i in range(2, self.Ns-1):
                res[i*self.Np      ] = 0.0
                res[(i+1)*self.Np-1] = 0.0

        #if (self.Ns == 6):
            #res[2*self.Np  ] = 0.0
            #res[3*self.Np-1] = 0.0
        #    res[3*self.Np  ] = 0.0
        #    res[4*self.Np-1] = 0.0
        #    res[4*self.Np  ] = 0.0
        #    res[5*self.Np-1] = 0.0

        # electron temperature
        res[self.Ns*self.Np  ] = 0.0 #(nT[ 0] - 0.75*dens[0,iele])
        res[(self.Ns+1)*self.Np-1] = 0.0 #(nT[-1] - 0.75*dens[-1,iele])

        return res


    def spatial_jacobian(self, Uin, time, dt, weak_bc=False, solve_poisson=False):
        """Evaluates the residual.

        Inputs:
          Uin  : Current state
          time : Current time
          dt   : Time step

        Outputs: None (sets self.jac)

        Notes:
          This function currently assumes that Ns=2 and NT=1
        """
        xp = self.xp_module

        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        iele  = [0]             # Electrons
        iion  = list(range(1, 1 + self.Nion)) # Ions Ar+, Ar2+
        iAr2m = [3]             # Molecular Argon Ar2m
        inb   = [self.Ns-1]     # Background species
        iee   = [self.Ns]       # Electron Energy
        # iEeff = [self.Nv -1]    # Ion Effective Electric field

                
        Imat    = self.I_Np 

        # pull off state for convenience
        dens = xp.zeros((self.Np, self.Ns),dtype=xp.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = Uin[self.Ns*self.Np:(self.Ns+1)*self.Np] # assumes just 1 temperature!
        Te = nT/dens[:,iele]


        if self.IonEffEField:
            Eeff = Uin[(self.Nv-1)*self.Np:self.Nv*self.Np] # Effective electric field for argon ions


        ntot = xp.zeros((self.Np,1),dtype=xp.float64)
        # ntot_U = xp.zeros((self.Np, self.Nv))
        
        # all but background
        for i in range(1, self.Ns-1):
            ntot[:,0] += dens[:,i]
            # ntot_U[:,i] += xp.ones(self.Np)

        # background contribution
        ntot[:,0] += self.params.nAronp0 * dens[:,inb[0]]
        # ntot_U[:,self.Ns-1] += self.params.nAronp0*xp.ones(self.Np)
        ntot_U = self.ntot_U

        # Temperature (from ideal gas law)
        Tg = xp.zeros((self.Np, 1),dtype=xp.float64)
        Tg = (self.params.p0 - nT)/ntot

        Tg_U = xp.zeros((self.Np, self.Nv),dtype=xp.float64)
        for i in range(0, self.Nv):
            Tg_U[:,i] = -(Tg[:,0]/ntot[:,0])*ntot_U[:,i]

        Tg_U[:,self.Ns] += -self.ones_Np/ntot[:,0]

        # NOTE(malamast): I clip the electron temperature when a low value occurs. 
        Te = xp.where(Te < Tg,Tg, Te) 

        Te_U = xp.zeros((self.Np, self.Nv),dtype=xp.float64)
        Te_U[:,0] = -Te[:,0]/dens[:,iele[0]]
        Te_U[:,iee[0]] = 1./dens[:,iele[0]]

        Te_ne = -xp.multiply(Te/dens[:,iele],Imat)
        Te_nT = xp.multiply(Imat,1./dens[:,iele])          

        #print("Mean gas temperature = {0:.6e}".format((2./3)*xp.mean(Tg)*11604.))

        # force solving poisson equation again
        if (solve_poisson):
            self.solve_poisson(dens,time)


        # form flux Jacobians
        dens_x = self.Dp @ dens
        nT_x   = self.Dp @ nT
        phi_x  = self.Dp @ self.phi

        # phi_x_ne = self.phi_x_ne
        # phi_x_ni = self.phi_x_ni
        phi_x_n = self.phi_x_n


        # Reduced Electric field  E / N 
        EN =  xp.abs(- phi_x)  / dens[:,inb] 
        # EN_Td = 1e21 * xp.abs(- phi_x) * self.params.V0L  / (self.params.nAr * dens[:,inb]) #  Electric field / N [Td]   
        # Esign = xp.ones_like(phi_x)
        Esign = -phi_x/dens[:,inb]/EN # NOTE(malamast): This is needed to get the derivatives of EN based on those of -phe_x
        if self.IonEffEField:
            ENion =  xp.abs(Eeff)  / dens[:,inb] 
            Eeffsign = Eeff/dens[:,inb]/ENion 
        else:
            ENion =  EN
            Eeffsign = Esign



        # NOTE(malamast): We now include transport coefficients for the electron energy equation
        energy = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64)
        mu     = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64)
        diffusivity = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64)

        # Initialize all energies to heavy species...
        for i in range(0,self.Ns):
            energy[:,i] = Tg[:,0] # For species

        # and overwrite for some
        energy[:,iele[0]] = Te[:,0] # Electrons

        for i in range(0, self.Nion):
          energy[:,iion[i]] = ENion[:,0] # For ions

        energy[:,iee[0]] = Te[:,0] # Electron energy

        for i in range(0,self.Ns+1):
            mu[:,i]  = self.params.mobility(i, energy, dens[:,inb[0]])
            diffusivity[:,i] = self.params.diffusivity(i, energy, mu, dens[:,inb[0]],
                                                       Te[:,0], Tg[:,0], self.EinsteinForm, self.EinsteinFormIon)

        if self.EinsteinForm:
            mu[:,iee[0]] = (5./3.) * mu[:,iele[0]] 
            diffusivity[:,iee[0]] = (5./3.) * diffusivity[:,iele[0]] 


        energy_U = xp.zeros((self.Ns+1, self.Nv, self.Np, self.Np),dtype=xp.float64)


        energy_U = xp.zeros((self.Ns+1, self.Nv, self.Np, self.Np),dtype=xp.float64)
        # Use this if me is a function of Te
        energy_U[0,0,:,:] = Te_ne; energy_U[0,self.Ns,:,:] = Te_nT 
        energy_U[iee[0],0,:,:] = Te_ne; energy_U[iee[0],self.Ns,:,:] = Te_nT
        # # Use this if me is a function of EN
        # for j in range(0,self.Ns): 
        #     if self.params.charge(j) != 0:
        #         energy_U[iele[0],j,:,:]         = xp.diag(1/dens[:,inb[0]]) @ (Esign*(self.params.charge(j)*(-phi_x_n)))
        #         energy_U[iee[0],j,:,:]          = xp.diag(1/dens[:,inb[0]]) @ (Esign*(self.params.charge(j)*(-phi_x_n)))
        # energy_U[iele[0],self.Ns-1,:,:] = -xp.multiply(EN/dens[:,inb],Imat) 
        # energy_U[iee[0],self.Ns-1,:,:]  = -xp.multiply(EN/dens[:,inb],Imat) 

        if not self.IonEffEField:            
            for j in range(0,self.Ns): 
                if self.params.charge(j) != 0:
                    for i in range(0, self.Nion):
                        energy_U[iion[i],j,:,:]  = xp.diag(1/dens[:,inb[0]]) @ (Esign*(self.params.charge(j)*(-phi_x_n)))
        else:
            for i in range(0, self.Nion):
                energy_U[iion[i],self.Nv-1,:,:] = xp.diag(Eeffsign[:,0]/dens[:,inb[0]])

        for i in range(0, self.Nion):
          energy_U[iion[i],self.Ns-1,:,:] = -xp.multiply(ENion/dens[:,inb],Imat)


        for i in range(self.Nion+1,self.Ns): 
            for j in range(1,self.Nv):
                energy_U[i,j,:,:] = xp.multiply(Imat,Tg_U[:,j]) # NOTE(malamast): This assumes that the diffusion coefficients of species 
                                                                # is a function of Tg. It is not used for constant D's


        diffusivity_U = xp.zeros((self.Ns+1, self.Nv, self.Np, self.Np),dtype=xp.float64)
        mu_U = xp.zeros((self.Ns+1, self.Nv, self.Np, self.Np),dtype=xp.float64)
        for i in range(0,self.Ns+1):
            for j in range(0,self.Nv):
                diffusivity_U[i,j,:,:] = self.params.diffusivity_U(i, j, energy, energy_U,
                                                                   mu, diffusivity, dens[:,self.Ns-1],
                                                                   Te[:,0],Te_U,Tg[:,0],Tg_U,self.EinsteinForm, self.EinsteinFormIon)
                mu_U[i,j,:,:] = self.params.mobility_U(i, j, energy, energy_U, mu, dens[:,self.Ns-1])


        if self.EinsteinForm:
            for j in range(0,self.Nv):
                mu_U[iee[0],j,:,:] = (5./3.) * mu_U[iele[0],j,:,:]
                diffusivity_U[iee[0],j,:,:] = (5./3.) * diffusivity_U[iele[0],j,:,:]
            

        
        # must have these for joule heating terms
        fspec = xp.zeros((self.Np, self.Ns),dtype=xp.float64)
        for i in range(0,self.Ns):
            if self.params.charge(i) != 0:
                if self.IonEffEField and self.params.charge(i) > 0:
                    fspec[:,i] += self.params.charge(i) * mu[:,i] * dens[:,i] * (Eeff[:,0])
                else:
                    fspec[:,i] += self.params.charge(i) * mu[:,i] * dens[:,i] * (-phi_x[:,0])
            fspec[:,i] -= xp.multiply(diffusivity[:,i], dens_x[:,i]) 
      

        # must have electron flux for use in Jacobian of Joule heating
        fe = xp.zeros((self.Np, 1),dtype=xp.float64) 
        fe[:,0] = fspec[:,0]


        if (weak_bc):
        # overwrite endpoints in fi (weakly impose BC)
        # in 1D, the unit normal vector is nx = −1 on the ’left’ and nx = 1 on the ’right’ boundary

            if self.IonEffEField:
                for i in range(0, self.Nion):
                    fspec[ 0,iion[i]] = -self.params.ksion * dens[ 0,iion[i]] + mu[ 0,iion[i]] * dens[ 0,iion[i]] * ( max(Eeff[ 0,0]*(-1),0.0) ) * (-1)
                    fspec[-1,iion[i]] =  self.params.ksion * dens[-1,iion[i]] + mu[-1,iion[i]] * dens[-1,iion[i]] * ( max(Eeff[-1,0],0.0) )
            else:
                for i in range(0, self.Nion):
                    fspec[ 0,iion[i]] = -self.params.ksion * dens[ 0,iion[i]] + mu[ 0,iion[i]] * dens[ 0,iion[i]] * ( max(-phi_x[ 0,0]*(-1),0.0) ) * (-1)
                    fspec[-1,iion[i]] =  self.params.ksion * dens[-1,iion[i]] + mu[-1,iion[i]] * dens[-1,iion[i]] * ( max(-phi_x[-1,0],0.0) )


        # species equations
        fspec_U = xp.zeros((self.Ns, self.Nv,self.Np, self.Np),dtype=xp.float64)
        for i in range(0,self.Ns-1): #  We do not include the background species here.
            fspec_U[i,i,:,:] -= xp.multiply(diffusivity[:,[i]], self.Dp) 
            if self.params.charge(i) != 0.0:
                if self.IonEffEField and self.params.charge(i) > 0:
                    fspec_U[i,i,:,:] += self.params.charge(i) \
                                    * xp.multiply(mu[:,[i]], xp.multiply(Imat,Eeff))
                    fspec_U[i,self.Nv-1,:,:] += self.params.charge(i) \
                                * xp.multiply(mu[:,[i]], xp.multiply(Imat,dens[:,[i]]))                                    
                else:  
                    fspec_U[i,i,:,:] += self.params.charge(i) \
                                    * xp.multiply(mu[:,[i]], xp.multiply(Imat,-phi_x))
                    for j in range(0,self.Ns): 
                        if self.params.charge(j) != 0:
                            fspec_U[i,j,:,:] += self.params.charge(i) \
                                        * xp.multiply(mu[:,[i]],xp.multiply(dens[:,[i]],(self.params.charge(j)*(-phi_x_n))))

                
        for i in range(0,self.Ns-1):
            for j in range(0,self.Nv):
                if self.params.charge(i) != 0.0:
                    if self.IonEffEField and self.params.charge(i) > 0:
                        fspec_U[i,j,:,:] += self.params.charge(i) * xp.multiply(mu_U[i,j,:,:], xp.multiply(dens[:,[i]],Eeff))
                    else:
                        fspec_U[i,j,:,:] += self.params.charge(i) * xp.multiply(mu_U[i,j,:,:], xp.multiply(dens[:,[i]],-phi_x))
                fspec_U[i,j,:,:] -= xp.multiply(diffusivity_U[i,j,:,:], dens_x[:,[i]])
        

        # energy equations
        fT = xp.zeros((self.Np, 1),dtype=xp.float64)
        fT[:,0] = -xp.multiply(mu[:,iee[0]],nT[:,0]) * (-phi_x[:,0]) -  xp.multiply(diffusivity[:,iee[0]], nT_x[:,0])
      
        fT_U = xp.zeros((self.Nv,self.Np, self.Np),dtype=xp.float64)
        for j in range(0,self.Ns): 
            if self.params.charge(j) != 0:
                fT_U[j,:,:]  = -mu[:,iee] * xp.multiply(nT,self.params.charge(j)*(-phi_x_n)) #NOTE(malamast): Here we need to have iee instead of iee[0] to make it consistent. Why? Need to discuss with Todd.

        for j in range(0, self.Nv):
            fT_U[j,:,:] -=  xp.multiply(mu_U[iee[0],j,:,:], xp.multiply(nT,-phi_x))
            fT_U[j,:,:] -=  xp.multiply(diffusivity_U[iee[0], j, :, :], nT_x[:,0])

        fT_U[self.Ns,:,:] +=  -xp.multiply(mu[:,iee],xp.multiply(Imat,-phi_x)) \
                              -xp.multiply(diffusivity[:,iee], self.Dp)


        if (weak_bc):
        # overwrite endpoints in fi (weakly impose BC)
        # in 1D, the unit normal vector is nx = −1 on the ’left’ and nx = 1 on the ’right’ boundary

            for i in range(0, self.Nion):
                for j in range(0,self.Nv):
                    fspec_U[iion[i],j,0,:] = 0
                    fspec_U[iion[i],j,-1,:] = 0

                if self.IonEffEField:
                    fspec_U[iion[i],self.Nv-1,0,0] = mu[0,iion[i]] * dens[0,iion[i]]
                    for j in range(0,self.Nv):
                        fspec_U[iion[i],j,0,:] += mu_U[iion[i],j,0,:] * dens[0,iion[i]] * (max(Eeff[0,0]*(-1),0.0)*(-1))
                    fspec_U[iion[i],iion[i],0,0] += mu[0,iion[i]] * (max(Eeff[0,0]*(-1),0.0)*(-1))
                else:
                    for j in range(0,self.Ns):
                        if self.params.charge(j) != 0:
                            fspec_U[iion[i],j,0,:] = mu[0,iion[i]] * dens[0,iion[i]] * self.params.charge(j)*(-phi_x_n[ 0,:])
                    for j in range(0,self.Nv):
                        fspec_U[iion[i],j,0,:] += mu_U[iion[i],j,0,:] * dens[0,iion[i]] * (max(-phi_x[0,0]*(-1),0.0)*(-1))
                    fspec_U[iion[i],iion[i],0,0] += mu[0,iion[i]] * (max(-phi_x[0,0]*(-1),0.0)*(-1))
                fspec_U[iion[i],iion[i],0,0] += -self.params.ksion

                if self.IonEffEField:
                    fspec_U[iion[i],self.Nv-1,-1,-1] = mu[-1,iion[i]] * dens[-1,iion[i]]
                    for j in range(0,self.Nv):
                        fspec_U[iion[i],j,-1,:] += mu_U[iion[i],j,-1,:] * dens[-1,iion[i]] * (max(Eeff[-1,0],0.0))
                    fspec_U[iion[i],iion[i],-1,-1] += mu[-1,iion[i]] * (max(Eeff[-1,0],0.0))
                else:
                    for j in range(0,self.Ns):
                        if self.params.charge(j) != 0:
                            fspec_U[iion[i],j,-1,:] = mu[-1,iion[i]] * dens[-1,iion[i]] * self.params.charge(j)*(-phi_x_n[-1,:])
                    for j in range(0,self.Nv):
                        fspec_U[iion[i],j,-1,:] += mu_U[iion[i],j,-1,:] * dens[-1,iion[i]] * (max(-phi_x[-1,0],0.0))
                    fspec_U[iion[i],iion[i],-1,-1] += mu[-1,iion[i]] * (max(-phi_x[-1,0],0.0))
                fspec_U[iion[i],iion[i],-1,-1] += self.params.ksion


        rstrg_U = xp.zeros((8,self.Nv*self.Np))

        # electron flux BC
        for j in range(0,self.Nv):
            rstrg_U[0,j*self.Np:(j+1)*self.Np] = fspec_U[iele[0],j,0,:]
            for i in range(0, self.Nion):
                rstrg_U[0,j*self.Np:(j+1)*self.Np] += - (- self.params.gam * fspec_U[iion[i],j,0,:] )
        rstrg_U[0,0] += self.params.ks * (Te[0,0]**0.5 + 0.5 * Te[0,0]**(-0.5) * Te_ne[0,0]* dens[0,iele[0]])
        rstrg_U[0,self.Ns*self.Np] += self.params.ks * (0.5 * Te[0,0]**(-0.5) * Te_nT[0,0] * dens[0,iele[0]])

        for j in range(0,self.Nv):
            rstrg_U[1,j*self.Np:(j+1)*self.Np] = fspec_U[iele[0],j,-1,:]
            for i in range(0, self.Nion):
                rstrg_U[1,j*self.Np:(j+1)*self.Np] += - (- self.params.gam * fspec_U[iion[i],j,-1,:])
        rstrg_U[1,self.Np-1] -= self.params.ks * (Te[-1,0]**0.5 + 0.5 * Te[-1,0]**(-0.5) * Te_ne[-1,-1]* dens[-1,iele[0]])
        rstrg_U[1,(self.Ns+1)*self.Np-1] -= self.params.ks * (0.5 * Te[-1,0]**(-0.5) * Te_nT[-1,-1] * dens[-1,iele[0]])


        # ion flux BC
        for i in range(0, self.Nion):
            if self.IonEffEField:
                for j in range(0,self.Nv):
                    rstrg_U[2 + 2*i,j*self.Np:(j+1)*self.Np] = fspec_U[iion[i],j,0,:] - mu_U[iion[i],j,0,:] * dens[0,iion[i]] * (max(Eeff[0,0]*(-1),0.0) * (-1))
                rstrg_U[2+2*i,iion[i]*self.Np] -= ( -self.params.ksion + mu[0,iion[i]] * (max(Eeff[0,0]*(-1),0.0) * (-1)) )
                if Eeff[0,0] < 0.0:
                    rstrg_U[2+2*i,(self.Nv-1)*self.Np] -= ( mu[0,iion[i]] * dens[0,iion[i]] ) # NOTE(malamast): I need to reconsider this one!
            else:
                for j in range(0,self.Nv):
                    rstrg_U[2+2*i,j*self.Np:(j+1)*self.Np] = fspec_U[iion[i],j,0,:] - mu_U[iion[i],j,0,:] * dens[0,iion[i]] * (max(-phi_x[0,0]*(-1),0.0) * (-1))
                if -phi_x[0,0] < 0.0:
                    for j in range(0,self.Ns):
                        if self.params.charge(j) != 0:
                            rstrg_U[2+2*i,j*self.Np:(j+1)*self.Np] -= (mu[0,iion[i]] * dens[0,iion[i]] * self.params.charge(j)*(-phi_x_n[ 0,:]))
                rstrg_U[2,iion[i]*self.Np] -= ( -self.params.ksion + mu[0,iion[i]] * (max(-phi_x[0,0]*(-1),0.0) * (-1)) )


            if self.IonEffEField:
                for j in range(0,self.Nv):
                    rstrg_U[3+2*i,j*self.Np:(j+1)*self.Np] = fspec_U[iion[i],j,-1,:] - mu_U[iion[i],j,-1,:] * dens[-1,iion[i]] * (max(Eeff[-1,0],0.0))
                rstrg_U[3+2*i,(iion[i]+1)*self.Np-1] -=  ( self.params.ksion + mu[-1,iion[i]] * (max(Eeff[-1,0],0.0)) )
                if Eeff[-1,0] > 0.0:
                    rstrg_U[3+2*i,self.Nv*self.Np-1] -=  ( mu[-1,iion[i]] * dens[-1,iion[i]])
            else:
                for j in range(0,self.Nv):
                    rstrg_U[3+2*i,j*self.Np:(j+1)*self.Np] = fspec_U[iion[i],j,-1,:] - mu_U[iion[i],j,-1,:] * dens[-1,iion[i]] * (max(-phi_x[-1,0],0.0))
                if -phi_x[-1,0] > 0.0:
                    for j in range(0,self.Ns):
                        if self.params.charge(j) != 0:
                            rstrg_U[3+2*i,j*self.Np:(j+1)*self.Np] -= mu[-1,iion[i]] * dens[-1,iion[i]] * self.params.charge(j)*(-phi_x_n[-1,:])
                rstrg_U[3+2*i,(iion[i]+1)*self.Np-1] -=  ( self.params.ksion + mu[-1,iion[i]] * (max(-phi_x[-1,0],0.0)) )

        # electron energy BC
        for j in range(0,self.Nv):
            rstrg_U[6,j*self.Np:(j+1)*self.Np] = fT_U[j,0,:]
            for i in range(0, self.Nion):
                rstrg_U[6,j*self.Np:(j+1)*self.Np] += - ( - self.params.gam * (fspec_U[iion[i],j,0,:]) * self.params.EeBC )
        rstrg_U[6,0] += (5./3.)*self.params.ks * nT[0,0] * (0.5 * Te[0,0]**(-0.5) * Te_ne[0,0])
        rstrg_U[6,self.Ns*self.Np] += (5./3.)*self.params.ks * (Te[0,0]**(0.5) + nT[0,0] * 0.5 * Te[0,0]**(-0.5) * Te_nT[0,0] )

        for j in range(0,self.Nv):
            rstrg_U[7,j*self.Np:(j+1)*self.Np] = fT_U[j,-1,:]
            for i in range(0, self.Nion):
                rstrg_U[7,j*self.Np:(j+1)*self.Np] += - ( - self.params.gam * (fspec_U[iion[i],j,-1,:]) * self.params.EeBC )
        rstrg_U[7,self.Np-1] -= (5./3.)*self.params.ks * nT[-1,0] * (0.5 * Te[-1,0]**(-0.5) * Te_ne[-1,-1])
        rstrg_U[7,(self.Ns+1)*self.Np-1] -= (5./3.)*self.params.ks * (Te[-1,0]**(0.5) + nT[-1,0] * 0.5 * Te[-1,0]**(-0.5) * Te_nT[-1,-1])

        # form Jacobians of derivatives of fluxes at collocation points
        fspec_x_U = xp.zeros((self.Ns, self.Nv, self.Np, self.Np),dtype=xp.float64)

        for i in range(0,self.Ns):
            for j in range(0,self.Nv):
                fspec_x_U[i,j,:,:] = self.Dp @ fspec_U[i,j,:,:]


        fT_x_U = xp.zeros((self.Nv, self.Np, self.Np),dtype=xp.float64)
        for j in range(0,self.Nv):
            fT_x_U[j, :,:] = self.Dp @ fT_U[j,:,:]

        # Radiation heating (used in the estimation of the S field)
        Qrad = xp.zeros((self.Np,1),dtype=xp.float64)

        # form source terms at collocation points
        # omega_V returns derivatives of chemical src terms wrt ne, ni, ..., Te
        if (self.solveCRModel):
 
            vars_CR = xp.zeros((self.Np, self.Ns+1),dtype=xp.float64)                                        
            vars_CR[:,0:self.Ns] = dens[:, self.cr.FromGlowDischargeToCRIndexing[0:-1]]*self.params.np0
            vars_CR[:,0] *= self.params.nAronp0
            vars_CR[:,self.Ns] = Te[:,0] * self.params.TwoOverThree   
                        

            # omega_CR = xp.ndarray((self.Np, self.Ns+1),dtype=xp.float64)              
            # omega_V_CR = xp.zeros((self.Ns+1,self.Ns+1,self.Np),dtype=xp.float64)

            omega_CR, omega_V_CR, Qrad_CR, Qrad_V_CR = self.cr.rxnSourceTermJac_vec(vars_CR)

            omega = xp.zeros_like(omega_CR)
            omega[:,:] = omega_CR[:, self.cr.FromCRToGlowDischargeIndexing] * self.params.tauOvernp0
            omega[:,self.Ns-1] /=  self.params.nAronp0   

            Qrad[:,0] = Qrad_CR * self.params.tauOvernp0                        

            omega_V = xp.zeros_like(omega_V_CR)
            omega_V[:,:,:] = omega_V_CR[self.cr.FromCRToGlowDischargeIndexing,:,:][:,self.cr.FromCRToGlowDischargeIndexing,:]  
            omega_V[:,0:self.Ns - 1,:] *= self.params.np0 # derivatives wrt ni
            omega_V[:,self.Ns - 1,:] *= self.params.nAr # derivatives wrt ground state
            omega_V[:,self.Ns,:] *=  self.params.TwoOverThree  # derivatives wrt temperature

            omega_V *= self.params.tauOvernp0 # nondimensionalize rates for ni
            omega_V[self.Ns - 1,:,:] /=  self.params.nAronp0  # correction for ground state                                     


            Qrad_V = xp.zeros_like(Qrad_V_CR)
            Qrad_V[:,:] = Qrad_V_CR[self.cr.FromCRToGlowDischargeIndexing,:]

            Qrad_V *= self.params.tauOvernp0 # nondimensionalize of rates 
            Qrad_V[0:self.Ns - 1,:] *= self.params.np0 # derivatives wrt ni
            Qrad_V[self.Ns - 1,:] *= self.params.nAr # derivatives wrt ground state
            Qrad_V[self.Ns,:] *=  self.params.TwoOverThree  # derivatives wrt temperature


        else:
            omega = self.params.rxnSourceTerm(Te, dens)
            omega_V = self.params.rxnSourceTermJac(Te, dens)
            Qrad_V = xp.zeros((self.Ns+1,self.Np),dtype=xp.float64) 
            
        # chain rule to get derivatives wrt ne, ni, ..., nT 
        omega_U = xp.zeros_like(omega_V)
        for i in range(0,self.Ns+1):
            omega_U[i,0,:] = omega_V[i,0,:] + omega_V[i,self.Ns,:]*xp.diag(Te_ne)
            omega_U[i,self.Ns,:] = omega_V[i,self.Ns,:]*xp.diag(Te_nT)
        omega_U[:,1:self.Ns,:] = omega_V[:,1:self.Ns,:]

        if (self.solveCRModel):
          Qrad_U = xp.zeros_like(Qrad_V) 
          Qrad_U[0,:] = Qrad_V[0,:] + Qrad_V[self.Ns,:]*xp.diag(Te_ne)
          Qrad_U[self.Ns,:] = Qrad_V[self.Ns,:]*xp.diag(Te_nT)
          Qrad_U[1:self.Ns,:] = Qrad_V[1:self.Ns,:]
        else:
          Qrad_U = xp.zeros((self.Nv, self.Np), dtype=xp.float64)



        # joule heating
        SJ_U = xp.zeros((self.Nv, self.Np, self.Np), dtype=xp.float64)
        for j in range(0, self.Nv):
            SJ_U[j,:,:] = -self.params.qStar * xp.multiply(fspec_U[iele[0],j,:,:],-phi_x)

        for j in range(0, self.Ns):
            if self.params.charge(j) != 0:
                SJ_U[j,:,:] -= self.params.qStar *  xp.multiply(fe , self.params.charge(j)*(-phi_x_n))           


        # elastic collisions
        SEC_U = xp.zeros((self.Nv, self.Np, self.Np), dtype=xp.float64)
        for j in range(0, self.Nv):
            SEC_U[j, :, :] = self.params.EC * dens[:, iele] \
                           * xp.multiply(Imat, xp.diag(Tg_U[:, j]))
        SEC_U[self.Ns, :, :] -= self.params.EC * Imat
        SEC_U[      0, :, :] += self.params.EC * xp.multiply(Imat, Tg)
        SEC_U *= self.elasticCollisionActivationFactor



        # evaluate S---the source term required in the background
        # species evolution to ensure constant pressure
        # fa = xp.zeros((self.Np,1),dtype=xp.float64)
        # fa_U = xp.zeros((self.Ns+1,self.Np, self.Np),dtype=xp.float64)
        fa = xp.copy(fT)
        fa_U = xp.copy(fT_U)

        naTg = xp.zeros((self.Np,1),dtype=xp.float64)        
        for i in range(1,self.Ns-1):
            naTg[:,0] = dens[:,i]*Tg[:,0]*self.TwoOverThree * self.params.Cv[i]
            CpOverCv = self.params.Cp[i]/self.params.Cv[i]

            fa[:,0] -= CpOverCv * xp.multiply(diffusivity[:,i], (self.Dp @ naTg[:,0] ))
            if self.params.charge(i) != 0:  
                if self.IonEffEField and self.params.charge(i)>0:
                    fa[:,0] += CpOverCv * self.params.charge(i) * \
                                xp.multiply(mu[:,i],xp.multiply(naTg[:,0],Eeff[:,0])) 

                    fa_U[self.Nv-1,:,:] += CpOverCv * (self.params.charge(i)*xp.multiply(mu[:,[i]], xp.multiply(Imat,naTg)))
                    fa_U[i,:,:] += CpOverCv * self.params.charge(i)*xp.multiply(mu[:,[i]], 
                                            xp.multiply(xp.diag(self.params.Cv[i] * self.TwoOverThree*Tg[:,0]),Eeff))
                else:                              
                    fa[:,0] += CpOverCv * self.params.charge(i) * \
                                xp.multiply(mu[:,i],xp.multiply(naTg[:,0],(-phi_x[:,0]))) 

                    for j in range(0, self.Ns):
                        if self.params.charge(j) != 0: 
                            fa_U[j,:,:] += CpOverCv *(self.params.charge(i)*xp.multiply(mu[:,[i]], xp.multiply(naTg,self.params.charge(j)*(-phi_x_n))))
                
                    fa_U[i,:,:] += CpOverCv * self.params.charge(i)*xp.multiply(mu[:,[i]], 
                                            xp.multiply(xp.diag(self.params.Cv[i] * self.TwoOverThree*Tg[:,0]),-phi_x))
                
            fa_U[i,:,:] -= CpOverCv * xp.multiply(diffusivity[:,[i]], self.Dp @ xp.diag(self.params.Cv[i] * self.TwoOverThree*Tg[:,0]))
                                    
            for j in range(0, self.Nv):
                fa_U[j,:,:] -= CpOverCv * xp.multiply(diffusivity[:,[i]], (self.Dp @ xp.multiply(dens[:,[i]],
                                                xp.diag(self.params.Cv[i] * self.TwoOverThree*Tg_U[:,j]))))
                if self.params.charge(i) != 0.0: 
                    if self.IonEffEField and self.params.charge(i)>0:
                        fa_U[j,:,:] += CpOverCv * self.params.charge(i) * xp.multiply(mu[:,[i]], xp.multiply(dens[:,i]*(Eeff),
                                                                                xp.diag(self.params.Cv[i] * self.TwoOverThree*Tg_U[:,j])))  
                        fa_U[j,:,:] += CpOverCv * self.params.charge(i) * xp.multiply(mu_U[i,j,:,:],xp.multiply(naTg[:,0],(Eeff[:,0])))
                    else:
                        fa_U[j,:,:] += CpOverCv * self.params.charge(i) * xp.multiply(mu[:,[i]], xp.multiply(dens[:,i]*(-phi_x),
                                                                                xp.diag(self.params.Cv[i] * self.TwoOverThree*Tg_U[:,j]))) 
                        fa_U[j,:,:] += CpOverCv * self.params.charge(i) * xp.multiply(mu_U[i,j,:,:],xp.multiply(naTg[:,0],(-phi_x[:,0])))
                    
                fa_U[j,:,:] -= CpOverCv * xp.multiply(self.Dp @ naTg, diffusivity_U[i,j,:,:])

        # background thermal conductivity contribution
        fa[:,0] += - self.params.kappaB * (self.Dp @ (self.params.Cv[inb[0]] * self.TwoOverThree*Tg[:,0])) 
        for j in range(0,self.Nv):
            fa_U[j,:,:] += - self.params.kappaB * self.Dp @ xp.diag(self.params.Cv[inb[0]] * self.TwoOverThree*Tg_U[:,j])

        fa_x = self.Dp @ fa

        fa_x_U = xp.zeros((self.Nv,self.Np, self.Np),dtype=xp.float64)
        for j in range(0,self.Nv):
            fa_x_U[j,:,:] = self.Dp @ fa_U[j,:,:]



        sOmEp = xp.zeros((self.Np,1),dtype=xp.float64)
        sOmEp_U = xp.zeros((self.Nv, self.Np, self.Np), dtype=xp.float64) 
        for i in range(0, self.Ns-1):
            sOmEp[:,0] += omega[:,i]*self.params.dEps[i]
            for j in range(0,self.Ns+1):
                sOmEp_U[j,:,:] += xp.diag(omega_U[i,j,:]*self.params.dEps[i])


        joule = xp.zeros((self.Np,1),dtype=xp.float64)
        joule_U = xp.zeros((self.Nv, self.Np, self.Np), dtype=xp.float64)
        for i in range(0, self.Ns-1):
            if self.params.charge(i) != 0.0:
                if self.IonEffEField and self.params.charge(i)>0:                       
                    joule[:,0] += self.params.qStar*self.params.charge(i)*xp.multiply(fspec[:,i],(Eeff[:,0]))
                    for j in range(0,self.Nv):
                        joule_U[j,:,:] += self.params.qStar*self.params.charge(i)*xp.multiply(fspec_U[i,j,:,:],Eeff)
                    joule_U[self.Nv-1,:,:] += self.params.qStar*self.params.charge(i)*xp.multiply(fspec[:,[i]],Imat)                    
                else:
                    joule[:,0] += self.params.qStar*self.params.charge(i)*xp.multiply(fspec[:,i],(-phi_x[:,0]))
                    for j in range(0,self.Nv):
                        joule_U[j,:,:] += self.params.qStar*self.params.charge(i)*xp.multiply(fspec_U[i,j,:,:],(-phi_x))
                    for j in range(0,self.Ns):
                        if self.params.charge(j) != 0:
                            joule_U[j,:,:] += self.params.qStar*self.params.charge(i)*xp.multiply(fspec[:,[i]],self.params.charge(j)*(-phi_x_n))        


        Qrad_UI = xp.zeros((self.Nv, self.Np, self.Np), dtype=xp.float64)
        for j in range(0,self.Ns+1):
            Qrad_UI[j,:,:] = xp.diag(Qrad_U[j,:]) 
        
        S  = (sOmEp + fa_x - joule -Qrad) / (self.params.Cv[inb[0]] * self.TwoOverThree*Tg) / self.params.nAronp0
        S *= self.backgroundSpecieActivationFactor

        S_U = xp.zeros((self.Nv, self.Np, self.Np), dtype=xp.float64)
        S_U = (sOmEp_U + fa_x_U - joule_U -Qrad_UI)/(self.params.Cv[inb[0]] * self.TwoOverThree*Tg) / self.params.nAronp0
        S_U *= self.backgroundSpecieActivationFactor
        for j in range(0,self.Nv):
            S_U[j,:,:] += xp.multiply(xp.diag( -(S/Tg)*Tg_U[:,j] ),Imat) #NOTE(malamast): But the numerator is also a function of Tg

        # form the full jacobian
        self.jac = xp.zeros((self.Ndof,self.Ndof))

        # spatial part

        # fluxes: involve spatial derivatives, leading to dense matrices

        # 'standard' continuity eqns
        for i in range(0,self.Ns-1):
            for j in range(0,self.Nv):
                self.jac[i*self.Np:(i+1)*self.Np,j*self.Np:(j+1)*self.Np] = dt*(fspec_x_U[i,j,:,:])

        # electron energy eqn
        for j in range(0,self.Ns+1):
            self.jac[self.Ns*self.Np:(self.Ns+1)*self.Np,j*self.Np:(j+1)*self.Np] = dt*(fT_x_U[j,:,:])


        # chemistry: spatially local, coupling across species and energy
        # use xp.einsum to extract diagonal of each Jacobian block for updating
        # NB: This affects the background eqns (erroneously) but it is overwritten later
        for i in range(0,self.Ns+1):
            for j in range(0,self.Ns+1):
                jac_diag  = xp.einsum('ii->i', self.jac[i*self.Np:(i+1)*self.Np,j*self.Np:(j+1)*self.Np])
                jac_diag -= dt*omega_U[i,j,:]


        # Joule heating (electron energy eqn)
        for j in range(0,self.Nv):
            self.jac[self.Ns*self.Np:(self.Ns+1)*self.Np,j*self.Np:(j+1)*self.Np] -= dt*(SJ_U[j, :, :] + SEC_U[j, :, :]) 
        
        
        # overwrite the background (wrt all variables)
        for j in range(0,self.Nv):
            self.jac[(self.Ns-1)*self.Np:self.Ns*self.Np,j*self.Np:(j+1)*self.Np] = -dt*(S_U[j,:,:])


        #  effective electric field for ions
        if self.IonEffEField:
            for j in range(0,self.Ns):
                if self.params.charge(j) != 0:
                    self.jac[(self.Ns+1)*self.Np:(self.Ns+2)*self.Np,j*self.Np:(j+1)*self.Np] -= dt * self.params.vmStar / mu[:,[iion[0]]] * self.params.charge(j)*(-phi_x_n)
            
            self.jac[(self.Ns+1)*self.Np:(self.Ns+2)*self.Np,(self.Ns+1)*self.Np:(self.Ns+2)*self.Np] -= dt * self.params.vmStar * \
                                                                                                         xp.multiply(1.0/mu[:,[iion[0]]] , (-Imat))
            for j in range(0,self.Nv):
                self.jac[(self.Ns+1)*self.Np:(self.Ns+2)*self.Np,j*self.Np:(j+1)*self.Np] -= dt * self.params.vmStar * \
                                                        (-1.0 / mu[:,[iion[0]]]**2) * xp.multiply(mu_U[iion[0],j,:,:],(-phi_x - Eeff))      


        return rstrg_U


    def jacobian(self, Uin, time, dt, weak_bc=False, solve_poisson=False):
        """Evaluates the Jacobian.

        Inputs:
          Uin    : Current state vector
          time   : Current time (double)
          dt     : Time step (double)
          weak_bc: Weak electron flux BC flag (boolean)

        Outputs: None (sets self.jac)
        """
        if (self.temporal_scheme=="BE"):
            self.jacobianBE(Uin, time, dt, weak_bc, solve_poisson)
            # self.jacobianFD(Uin, time, dt)
            
        elif (self.temporal_scheme=="CN"):
            self.jacobianCN(Uin, time, dt, weak_bc, solve_poisson)
        else:
            print("Time marching scheme not recognized")
            exit(-1)


    def jacobianBE(self, Uin, time, dt, weak_bc=False, solve_poisson=False):
        """Evaluates the Jacobian for backward Euler time marching.
        See timeDomainCollocationSolver.jacobian() for further documentaion.
        """
        xp = self.xp_module

        iion  = list(range(1, 1 + self.Nion)) # Ions Ar+, Ar2+

        # Jacobian of spatial contribution to residual
        rstrg_U = self.spatial_jacobian(Uin, time, dt, weak_bc, solve_poisson)


        # Jacobian of unsteady contribution to residual
        self.jac += self.I_Ndof

        # boundary condition modifications (for strongly enforced BCs)
        # electron flux
        self.jac[0          ,:] = rstrg_U[0,:]
        self.jac[self.Np-1  ,:] = rstrg_U[1,:]

        if (not weak_bc):
            # ion flux
            for i in range(0, self.Nion):
                self.jac[iion[i] * self.Np    ,:] = rstrg_U[2 + 2*i,:]
                self.jac[(iion[i] + 1)*self.Np-1,:] = rstrg_U[3 + 2*i,:]

        # electron energy flux
        self.jac[self.Ns*self.Np,:] = rstrg_U[6,:] 
        self.jac[(self.Ns+1)*self.Np-1,:] = rstrg_U[7,:] 

        for i in range(self.Nion+1,self.Ns-1):
            self.jac[i*self.Np,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac[i*self.Np,i*self.Np] = 1.0

            self.jac[(i+1)*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac[(i+1)*self.Np-1,(i+1)*self.Np-1] = 1.0

        # Dirichlet on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            self.jac[(self.Ns-1)*self.Np,:] = xp.zeros((1,self.Nv*self.Np))

            for i in range(1,self.Ns-1):
                self.jac[(self.Ns-1)*self.Np,i*self.Np] = 1.0 / self.params.nAronp0

            self.jac[(self.Ns-1)*self.Np,(self.Ns-1)*self.Np] = 1.0
            self.jac[(self.Ns-1)*self.Np,self.Ns*self.Np] = 1.0 / self.params.Tg0 / self.params.nAronp0

            self.jac[self.Ns*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))

            for i in range(1,self.Ns-1):
                self.jac[self.Ns*self.Np-1,(i+1)*self.Np-1] = 1.0 / self.params.nAronp0

            self.jac[self.Ns*self.Np-1,self.Ns*self.Np-1] = 1.0
            self.jac[self.Ns*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0 / self.params.Tg0 / self.params.nAronp0


        # Dirichlet condition on electron energy
        if (self.params.electron_energy_dirichlet):
            self.jac[self.Ns*self.Np,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac[self.Ns*self.Np,self.Ns*self.Np] = 1.0
            self.jac[self.Ns*self.Np,0] = -self.params.EeBC

            self.jac[(self.Ns+1)*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac[(self.Ns+1)*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0
            self.jac[(self.Ns+1)*self.Np-1,self.Np-1] = -self.params.EeBC

        # No BC for Eeff (the effective electric field for ions)


    def jacobianCN(self, Uin, time, dt, weak_bc=False, solve_poisson=False):
        """Evaluates the Jacobian for Crank-Nicolson time marching.
        See timeDomainCollocationSolver.jacobian() for further documentaion.
        """
        xp = self.xp_module

        iion  = list(range(1, 1 + self.Nion)) # Ions Ar+, Ar2+

        # Jacobian of spatial contribution to residual
        rstrg_U = self.spatial_jacobian(Uin, time, dt, weak_bc, solve_poisson)
        self.jac *= 0.5

        # Jacobian of unsteady contribution to residual
        self.jac += self.I_Ndof

        # boundary condition modifications (for strongly enforced BCs)
        self.jac[0          ,:] = rstrg_U[0,:]
        self.jac[self.Np-1  ,:] = rstrg_U[1,:]

        if (not weak_bc):
            # ion flux
            for i in range(0, self.Nion):
                self.jac[iion[i] * self.Np    ,:] = rstrg_U[2 + 2*i,:]
                self.jac[(iion[i] + 1)*self.Np-1,:] = rstrg_U[3 + 2*i,:]

        # electron energy flux
        self.jac[self.Ns*self.Np,:] = rstrg_U[6,:] 
        self.jac[(self.Ns+1)*self.Np-1,:] = rstrg_U[7,:] 

        for i in range(self.Nion+1,self.Ns-1):
            self.jac[i*self.Np,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac[i*self.Np,i*self.Np] = 1.0

            self.jac[(i+1)*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac[(i+1)*self.Np-1,(i+1)*self.Np-1] = 1.0

        # Dirichlet on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            self.jac[(self.Ns-1)*self.Np,:] = xp.zeros((1,self.Nv*self.Np))

            for i in range(1,self.Ns-1):
                self.jac[(self.Ns-1)*self.Np,i*self.Np] = 1.0 / self.params.nAronp0

            self.jac[(self.Ns-1)*self.Np,(self.Ns-1)*self.Np] = 1.0
            self.jac[(self.Ns-1)*self.Np,self.Ns*self.Np] = 1.0 / self.params.Tg0 / self.params.nAronp0

            self.jac[self.Ns*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))

            for i in range(1,self.Ns-1):
                self.jac[self.Ns*self.Np-1,(i+1)*self.Np-1] = 1.0 / self.params.nAronp0

            self.jac[self.Ns*self.Np-1,self.Ns*self.Np-1] = 1.0
            self.jac[self.Ns*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0 / self.params.Tg0 / self.params.nAronp0

        # Dirichlet condition on electron energy
        if (self.params.electron_energy_dirichlet):
            self.jac[self.Ns*self.Np,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac[self.Ns*self.Np,self.Ns*self.Np] = 1.0
            self.jac[self.Ns*self.Np,0] = -self.params.EeBC

            self.jac[(self.Ns+1)*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac[(self.Ns+1)*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0
            self.jac[(self.Ns+1)*self.Np-1,self.Np-1] = -self.params.EeBC

        # No BC for Eeff (the effective electric field for ions)



    def jacobianLCN(self, Uin, time, dt, weak_bc=False):
        """Evaluates the Jacobian for Crank-Nicolson time marching.
        See timeDomainCollocationSolver.jacobian() for further documentaion.
        """
        xp = self.xp_module

        # Jacobian of spatial contribution to residual
        self.spatial_jacobian(Uin, time, dt, weak_bc)
        self.jac *= 0.5

        # Jacobian of unsteady contribution to residual
        self.jac += self.I_Ndof

        # boundary condition modifications (for strongly enforced BCs)
        if (not weak_bc):
            print("Error: Only weak electron flux BCs supported for linearized CN.")
            exit(-1)

        #if (self.Ns>2):
        #    self.jac[2*self.Np,:] = xp.zeros((1,self.Nv*self.Np))
        #    self.jac[2*self.Np,2*self.Np] = 1.0

        #    self.jac[3*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
        #    self.jac[3*self.Np-1,3*self.Np-1] = 1.0

        if (self.Ns > 2):
            for i in range(2,self.Ns-1):
                self.jac[i*self.Np,:] = xp.zeros((1,self.Nv*self.Np))
                self.jac[i*self.Np,i*self.Np] = 1.0
                self.jac[(i+1)*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
                self.jac[(i+1)*self.Np-1,(i+1)*self.Np-1] = 1.0


        self.jac[self.Ns*self.Np,:] = xp.zeros((1,self.Nv*self.Np))
        self.jac[self.Ns*self.Np,self.Ns*self.Np] = 1.0
        self.jac[self.Ns*self.Np,0] = -self.params.EeBC

        self.jac[(self.Ns+1)*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
        self.jac[(self.Ns+1)*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0
        self.jac[(self.Ns+1)*self.Np-1,self.Np-1] = -self.params.EeBC


    def jacobian0(self, time, dt, weak_bc=False):
        """Evaluate the Jacobian of the residual with respect to the state at
        the previous time step

        Inputs:
          dt     : Time step (double)
          weak_bc: Weak electron flux BC flag (boolean)

        Outputs: None (sets self.jac0)
        """
        xp = self.xp_module

        if (self.temporal_scheme=="BE"):
            self.jac0 = -self.I_Ndof

        elif (self.temporal_scheme=="CN"):
            self.spatial_jacobian(self.U1, time-dt, dt, weak_bc, solve_poisson=True)
            self.jac *= 0.5

            self.jac0 = xp.copy(self.jac)

            self.jac0 -= self.I_Ndof
        else:
            print("Time marching scheme not recognized")
            exit(-1)

        # boundary condition modifications (for strongly enforced BCs)
        # NB: For BCs that are strongly enforced, corresponding
        # residual has no dependence on previous state
        self.jac0[0          ,:] = xp.zeros((1,self.Nv*self.Np))
        self.jac0[self.Np-1  ,:] = xp.zeros((1,self.Nv*self.Np))

        if (not weak_bc):
            self.jac0[self.Np    ,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac0[2*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac0[2*self.Np  ,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac0[3*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
            
        for i in range(self.Nion+1,self.Ns-1):
            self.jac0[i*self.Np      ,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac0[(i+1)*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))


        # Dirichlet on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            self.jac0[(self.Ns-1)*self.Np,:] = xp.zeros((1,self.Nv*self.Np))
            self.jac0[self.Ns*self.Np-1  ,:] = xp.zeros((1,self.Nv*self.Np))

        # Dirichlet on electron temperature
        self.jac0[self.Ns*self.Np      ,:] = xp.zeros((1,self.Nv*self.Np))
        self.jac0[(self.Ns+1)*self.Np-1,:] = xp.zeros((1,self.Nv*self.Np))
        
        # No BC for Eeff (the effective electric field for ions)


    def jacobianFD(self, Uin, time, dt):
        """Evaluates the Jacobian at Uin, but using a finite difference
        approximation.  Useful for testing, but very slow.

        Inputs:
          Uin  : Current state
          time : Current time
          dt   : Time step

        Outputs: None (sets self.jac)
        """
        xp = self.xp_module

        # save residual at Uin
        r0 = self.residual(Uin, time, dt)

        # perturb each component of Uin to form finite differenc approx
        for k in range(0,Uin.shape[0]):
            dU = xp.sqrt(xp.finfo(xp.float64).eps)*xp.absolute(Uin[k])
            Up = xp.copy(Uin)

            if (xp.absolute(dU) < xp.finfo(xp.float64).eps):
                dU = xp.finfo(xp.float64).eps

            Up[k] += dU

            rp = self.residual(Up, time, dt)
            self.jac[:,k] = (rp[:,0] - r0[:,0])/dU


    def step(self, time, dt, iter_max=20,
             rtol=1e-6, atol=1e-12, verbose=True, weak_bc=False, jac_frequency=1):
        """Take a single time step.

        Inputs
          time       : Current time
          dt         : Time step
          iter_max   : Maximum number of iters in nonlinear solve
          rtol       : Relative tolerance for nonlinear solve
          atol       : Absolute tolerance for nonlinear solve
          verbose    : If true, print nonlinear solve info

        Outputs: None (self.U2 is set to solution for this time step)
        """
        xp = self.xp_module

        r = self.residual(self.U2, time, dt, weak_bc)
        self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True) #NOTE(malamast): Comment out if you want to use xp.linalg.solve in the loop
        # jac_inv  = xp.linalg.inv(self.jac)       
        lu, piv = lu_factor(self.jac)            # one O(N^3) factorisation

        normr = normr0 = xp.linalg.norm(r)

        # if xp == cp:
        #   cp.cuda.runtime.deviceSynchronize()

        count = 0
        converged = ((normr/normr0 < rtol) or (normr < atol))

        if (verbose):
            print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                count, normr, normr/normr0))

        while( not converged and (count < iter_max) ):

            # self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)
            if count > 0 and count % jac_frequency == 0:
                self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)
                # jac_inv  = xp.linalg.inv(self.jac)   
                lu, piv = lu_factor(self.jac)            # one O(N^3) factorisation
            
            # dU = xp.dot(jac_inv, -r)
            dU = lu_solve((lu, piv), -r)

            U2_new = self.U2 + dU

            r = self.residual(U2_new, time, dt, weak_bc)
            normr = xp.linalg.norm(r)

            if not np.isfinite(normr):
                self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)
                # jac_inv  = xp.linalg.inv(self.jac) 
                # dU = xp.dot(jac_inv, -r)
                lu, piv = lu_factor(self.jac)            # one O(N^3) factorisation
                dU = lu_solve((lu, piv), -r)

                U2_new = self.U2 + dU
                r = self.residual(U2_new, time, dt, weak_bc)
                normr = xp.linalg.norm(r)

            self.U2[:] = U2_new


            count += 1
            if (verbose):
                print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                    count, normr, normr/normr0))

            converged = ((normr/normr0 < rtol) or (normr < atol))
            if not np.isfinite(normr) or (not (normr < 0.9 * normr0)): 
                converged = False
                break
                         
        return converged, count     # count is the Newton iteration count



        
    def step_adaptive(self, time, dt, verbose=True, rtol=1e-8, 
                      weak_bc=False, computeSensitivity=False,              
                      dt_init=None, dt_min=1e-5, dt_max=0.0625,
                      iter_target=6, iter_max=14, safety=0.3):

        """
        Integrates from time to time + dt using variable sub-steps.
        """
        xp = self.xp_module
        if dt_init is None:
            dt_init = float(dt / 128)        # safe heuristic


        if self.dt_adaptive > 0.0:
            dt_sub = self.dt_adaptive
        else:
            dt_sub  = min(dt_init, dt_max)

        dt_sub = max(min(dt_sub, dt_max), dt_min)



        t_local = 0.0                    # time elapsed *inside* this outer step

        while t_local < dt - 1e-15:
            if dt_sub > dt - t_local:
                self.dt_adaptive = dt_sub
                dt_sub = dt - t_local   # final sliver closes the gap


            # prepare for next step
            self.U0 = xp.copy(self.U1)
            self.U1 = xp.copy(self.U2)

            # save state in case we must reject
            U0_save, U1_save, U2_save = self.U0.copy(), self.U1.copy(), self.U2.copy()

            if computeSensitivity:
                A0_save, A1_save = self.A0.copy(), self.A1.copy()

            # try the sub-step
            converged, newt_iters = self.step(time + t_local + dt_sub, dt_sub, iter_max=iter_max,
                                    rtol=1e-8, atol=1e-12, verbose=True, weak_bc=False)

            if converged: # accept

                # propagate sensitivity for this accepted sub-step
                if computeSensitivity:
                    self.A0 = xp.copy(self.A1)
                    self.stepSensitivity(time + t_local + dt_sub, dt_sub,
                                        verbose=False, weak_bc=weak_bc)


                t_local += dt_sub
                if verbose:
                    print(f" 1/dt = {int(1/dt_sub):2d},  iters = {newt_iters:2d},  time = {time+t_local:.2e}")

                # adapt dt_sub for the *next* trial
                grow   = 1 + safety * max(0, (iter_target - newt_iters)/iter_target)
                shrink = 1 / (1 + safety * max(0, (newt_iters - iter_target)/iter_target))

                if newt_iters <= iter_target: 
                    grow   = 1 + safety * max(0, (iter_target - newt_iters)/iter_target)
                    dt_sub *= grow
                else:
                    shrink = 1 / (1 + safety * max(0, (newt_iters - iter_target)/iter_target))
                    dt_sub *= shrink

                dt_sub = max(min(dt_sub, dt_max), dt_min)

            else: # reject, roll back
                self.U0, self.U1, self.U2 = U0_save, U1_save, U2_save

                if computeSensitivity:
                    self.A0, self.A1 = A0_save, A1_save

                dt_sub *= 0.5
                if verbose:
                    print(f" Step failed — reducing dt to {dt_sub:.2e}")
                if dt_sub < dt_min:
                    raise RuntimeError("step_adaptive: dt dropped below dt_min")



    def step_fixed_dt(self, time, dt, iter_max=20,
                      rtol=1e-6, atol=1e-12, verbose=True, weak_bc=False, jac_frequency=1):
        """Take a single time step.

        Inputs
          time       : Current time
          dt         : Time step
          iter_max   : Maximum number of iters in nonlinear solve
          rtol       : Relative tolerance for nonlinear solve
          atol       : Absolute tolerance for nonlinear solve
          verbose    : If true, print nonlinear solve info

        Outputs: None (self.U2 is set to solution for this time step)
        """
        xp = self.xp_module

        r = self.residual(self.U2, time, dt, weak_bc)
        self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True) #NOTE(malamast): Comment out if you want to use xp.linalg.solve in the loop
        # jac_inv  = xp.linalg.inv(self.jac)       
        lu, piv = lu_factor(self.jac)            # one O(N^3) factorisation

        normr = normr0 = xp.linalg.norm(r)

        # if xp == cp:
        #   cp.cuda.runtime.deviceSynchronize()

        count = 0
        converged = ((normr/normr0 < rtol) or (normr < atol))

        if (verbose):
            print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                count, normr, normr/normr0))
        while( not converged and (count < iter_max) ):

            # self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)
            if count > 0 and count % jac_frequency == 0:
                self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)
                # jac_inv  = xp.linalg.inv(self.jac)   
                lu, piv = lu_factor(self.jac)            # one O(N^3) factorisation
            
            # dU = xp.dot(jac_inv, -r)
            dU = lu_solve((lu, piv), -r)

            U2_new = self.U2 + dU

            r = self.residual(U2_new, time, dt, weak_bc)
            normr = xp.linalg.norm(r)


            if not np.isfinite(normr):
                self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)
                # jac_inv  = xp.linalg.inv(self.jac) 
                # dU = xp.dot(jac_inv, -r)
                lu, piv = lu_factor(self.jac)            # one O(N^3) factorisation
                dU = lu_solve((lu, piv), -r)

                U2_new = self.U2 + dU
                r = self.residual(U2_new, time, dt, weak_bc)
                normr = xp.linalg.norm(r)

            self.U2[:] = U2_new

            count += 1
            if (verbose):
                print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                    count, normr, normr/normr0))

            converged = ((normr/normr0 < rtol) or (normr < atol))
            if not np.isfinite(normr): 
                break


        if (not converged):
            # if non-convergence encountered, save state and die
            print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                count, normr, normr/normr0))
            xp.save("nonconverged_U2.npy", self.U2)
            xp.save("nonconverged_U1.npy", self.U1)
            xp.save("nonconverged_U0.npy", self.U0)
            
            # EN_Td = 1e21 * xp.abs(- phi_x) * self.params.V0L  / (self.params.nAr * dens[:,inb]) #  Electric field / N [Td]   
            # pull off state for convenience
            dens = np.zeros((self.Np, self.Ns),dtype=np.float64)
            for i in range(0,self.Ns):
                dens[:,i] = self.U0[i*self.Np:(i+1)*self.Np,0]                
            self.solve_poisson(dens,time)
            phi_x  = self.Dp @ self.phi
            EN_Td = 1e21 * (- phi_x) * self.params.V0L  / (self.params.nAr * dens[:,self.Ns-1])
            xp.save("nonconverged_U0_Efield.npy", EN_Td)
            
            print("Step did not converge")
            exit(-1)            
                
        return converged, count     # count is the Newton iteration count


    def step_old(self, time, dt, iter_max=20,
                 rtol=1e-6, atol=1e-12, verbose=True, weak_bc=False, freeze_jacobian=False):
        """Take a single time step.

        Inputs
          time       : Current time
          dt         : Time step
          iter_max   : Maximum number of iters in nonlinear solve
          rtol       : Relative tolerance for nonlinear solve
          atol       : Absolute tolerance for nonlinear solve
          verbose    : If true, print nonlinear solve info

        Outputs: None (self.U2 is set to solution for this time step)
        """
        xp = self.xp_module

        r = self.residual(self.U2, time, dt, weak_bc)

        if freeze_jacobian:
            self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)
            jac_inv  = xp.linalg.inv(self.jac)

        normr = normr0 = xp.linalg.norm(r)

        # if xp == cp:
        #   cp.cuda.runtime.deviceSynchronize()

        count = 0
        converged = ((normr/normr0 < rtol) or (normr < atol))

        if (verbose):
            print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                count, normr, normr/normr0))

        while( not converged and (count < iter_max) ):
            if (not freeze_jacobian):
                self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)

            try:
                if freeze_jacobian:
                    dU = xp.dot(jac_inv, -r)
                else:
                    dU = xp.linalg.solve(self.jac, -r)

                self.U2 += dU

                # zero the last mode
                #self.filter()

            except:
                # if exception encountered, save state and die
                xp.save("residual.npy", r)
                xp.save("jacobian.npy", self.jac)
                xp.save("exception_U2.npy", self.U2)
                xp.save("exception_U1.npy", self.U1)
                xp.save("exception_U0.npy", self.U0)
                print("Solve failed!", flush=True)
                exit(-1)


            # self.U2[self.U2<0.0] = 0.0 # NOTE(malamast): This causes the periodic solver to fail. 
            #                              # Some small negative values can occur close to the boundaries 
            #                              # where the number densities are zero.

            r = self.residual(self.U2, time, dt, weak_bc)

            normr = xp.linalg.norm(r)
            count += 1
            if (verbose):
                print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                    count, normr, normr/normr0))

            converged = ((normr/normr0 < rtol) or (normr < atol))
            
                

        if (not converged):
            # if non-convergence encountered, save state and die
            print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                count, normr, normr/normr0))
            xp.save("nonconverged_U2.npy", self.U2)
            xp.save("nonconverged_U1.npy", self.U1)
            xp.save("nonconverged_U0.npy", self.U0)
            
            # EN_Td = 1e21 * xp.abs(- phi_x) * self.params.V0L  / (self.params.nAr * dens[:,inb]) #  Electric field / N [Td]   
            # pull off state for convenience
            dens = np.zeros((self.Np, self.Ns),dtype=np.float64)
            for i in range(0,self.Ns):
                dens[:,i] = self.U0[i*self.Np:(i+1)*self.Np,0]                
            self.solve_poisson(dens,time)
            phi_x  = self.Dp @ self.phi
            EN_Td = 1e21 * (- phi_x) * self.params.V0L  / (self.params.nAr * dens[:,self.Ns-1])
            xp.save("nonconverged_U0_Efield.npy", EN_Td)
            
            print("Step did not converge")
            exit(-1)

            

    def stepLCN(self, time, dt, verbose=False, weak_bc=False):
        """Take a single time step.

        Inputs
          time       : Current time
          dt         : Time step
          verbose    : If true, print nonlinear solve info

        Outputs: None (self.U2 is set to solution for this time step)
        """
        xp = self.xp_module

        r = self.residualLCN(self.U1, time, dt, weak_bc)
        self.jacobianLCN(self.U1, time, dt, weak_bc)

        dU = xp.linalg.solve(self.jac, -r)
        self.U2 += dU

    def stepSensitivity(self, time, dt, verbose=False, weak_bc=False):
        """Advance the sensitivity matrix

        Inputs
          time       : Current time
          dt         : Time step
          verbose    : If true, print nonlinear solve info

        Outputs: None (self.A1 is set to sensitivity at the end of the time step)
        """
        xp = self.xp_module
        

        # do this first b/c it may modify self.jac!
        # TODO: should probably change this design...
        #       storing a single jacobian as a member makes things confusing
        self.jacobian0(time, dt, weak_bc)

        # evaluate the required Jacobians
        self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True)
        #self.jacobianFD(self.U2, time, dt)


        # for the RHS
        self.rhsSens = -(self.jac0 @ self.A0)

        # solve the sensitivity update system
        self.A1 = xp.linalg.solve(self.jac, self.rhsSens)

        if (verbose):
            print("# Advancing sensitivity system.")



    def solve_one_period(self, time0, dt, Nstep, savedata=None, verbose=True,
                         rtol=1e-6, weak_bc=False, jac_frequency=1):
        
        xp = self.xp_module
        # xp = np

        Usave=xp.ndarray((Nstep+1,self.U2.shape[0]),dtype=xp.float64)
        TotalCurrentSave=xp.ndarray((Nstep+1,self.totalCurrent.shape[0]),dtype=xp.float64)
        IonCurrentSave=xp.ndarray((Nstep+1,self.ionCurrent.shape[0]),dtype=xp.float64)
        ElectronCurrentSave=xp.ndarray((Nstep+1,self.electronCurrent.shape[0]),dtype=xp.float64)
        electricFieldSave=xp.ndarray((Nstep+1,self.electricField.shape[0]),dtype=xp.float64)
        electricPotentialSave=xp.ndarray((Nstep+1,self.electricPotential.shape[0]),dtype=xp.float64)
        if IonEffEField:
            effElectricFieldSave=xp.ndarray((Nstep+1,self.effElectricField.shape[0]),dtype=xp.float64)

        Usave[0,:] = self.U2[:,0]
        TotalCurrentSave[0,:] = self.totalCurrent[:,0]
        IonCurrentSave[0,:] = self.ionCurrent[:,0]
        ElectronCurrentSave[0,:] = self.electronCurrent[:,0]
        electricFieldSave[0,:] = self.electricField[:,0]
        electricPotentialSave[0,:] = self.electricPotential[:,0]
        if IonEffEField:
            effElectricFieldSave[0,:] = self.effElectricField[:,0]


        # assume initial condition has been set in U1!
        time = time0+dt
        self.step(time, dt, verbose=verbose, rtol=rtol, weak_bc=weak_bc, jac_frequency=jac_frequency)
        print("{0:d} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e}  {7:.6e}".format(
            0, time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max(),
            self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
            self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()))


        Usave[1,:] = self.U2[:,0]
        TotalCurrentSave[1,:] = self.totalCurrent[:,0]
        IonCurrentSave[1,:] = self.ionCurrent[:,0]
        ElectronCurrentSave[1,:] = self.electronCurrent[:,0]
        electricFieldSave[1,:] = self.electricField[:,0]
        electricPotentialSave[1,:] = self.electricPotential[:,0]
        if IonEffEField:
            effElectricFieldSave[1,:] = self.effElectricField[:,0]

        for istep in range(1, Nstep):
            
            # prepare for next step
            self.U0 = xp.copy(self.U1)
            self.U1 = xp.copy(self.U2)
            time += dt

            # advance
            self.step(time, dt, verbose=verbose, rtol=rtol, weak_bc=weak_bc)
            #self.filter()
            print("{0:d} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e} {7:.6e}".format(
                istep, time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()), flush=True)

            Usave[istep+1,:] = self.U2[:,0]
            TotalCurrentSave[istep+1,:] = self.totalCurrent[:,0]
            IonCurrentSave[istep+1,:] = self.ionCurrent[:,0]
            ElectronCurrentSave[istep+1,:] = self.electronCurrent[:,0]
            electricFieldSave[istep+1,:] = self.electricField[:,0]
            electricPotentialSave[istep+1,:] = self.electricPotential[:,0]
            if IonEffEField:
                effElectricFieldSave[istep+1,:] = self.effElectricField[:,0]

        
        xp.save(savedata,Usave)
        xp.save("TotalCurrent_" + savedata, TotalCurrentSave)
        xp.save("IonCurrent_" + savedata, IonCurrentSave)
        xp.save("ElectronCurrent_" + savedata, ElectronCurrentSave)
        xp.save("ElectricField_" + savedata, electricFieldSave)
        xp.save("ElectricPotential_" + savedata, electricPotentialSave)
        if IonEffEField:
            xp.save("EffElectricField_" + savedata, effElectricFieldSave)




    def solve_adaptive(self, time0, dt, Nstep, savedata=None, verbose=False,
              rtol=1e-6, computeSensitivity=False, weak_bc=False):


        if self.args.use_gpu==1:
            self.copy_operators_H2D(self.args.gpu_device_id)
            self.xp_module = cp
            self.params.xp_module = cp
            if (self.solveCRModel):
                self.cr.xp_module = cp
                self.cr.copy_operators_Host2Device(self.args.gpu_device_id)
        else:
            self.xp_module = np
        
        xp = self.xp_module
        # xp = np

        assert dt == 1.0 , "dt must be exactly 1.0"

        # Restart parameters
        save_every_cycles = 20          # <-- change to whatever you like
        steps_per_cycle = int(round(1.0 / dt))      # number of fixed-dt steps per RF cycle
        cycle_idx       = 0                         # which RF cycle we are in


        # Adaptive solver parameters
        dt_init = None 
        dt_min = 1e-5 
        dt_max = float(1/8) 
        iter_target = 6           # desired Newton iterations
        iter_max = 14             # Maximum number of nonlinear iteration before it reduces the timestep
        safety = 0.5              # How fast the timestep grows (default was 0.9)

        dt_init = None
        if dt_init is None:
            dt_init = float(dt / 128)  # safe heuristic
        if dt_max is None:
            dt_max = dt


        print("#")
        print("# {0:8s} {1:10s} {2:12s} {3:12s} {4:12s} {5:12s} {6:12s} {7:12s}".format(
            "Iter", "Time", "min ne", "max ne", "min Te", "max Te", "min nb", "max nb"))
        print("{0:d} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e} {7:.6e}".format(
            -1, time0, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max(),
            self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
            self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()))

        # assume initial condition has been set in U1!
        time = time0

        for istep in range(0, Nstep): # RF steps
            # start_time = cpu_time.time()
            
            # prepare for next step
            self.U0 = xp.copy(self.U1)
            self.U1 = xp.copy(self.U2)

            # advance
            # self.step(time, dt, verbose=verbose, rtol=rtol, weak_bc=weak_bc)
            self.step_adaptive(time, dt, verbose=verbose, rtol=rtol, weak_bc=weak_bc, computeSensitivity=computeSensitivity,              
                               dt_init=dt_init, dt_min=dt_min, dt_max=dt_max, 
                               iter_target=iter_target, iter_max=iter_max, safety=safety)

            time += dt # dt = 1 ->  a RF period

            #self.filter()
            print("{0:d} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e} {7:.6e}".format(
                istep, time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()), flush=True)


            # ---------------------------------------------------------------
            # SAVE only when an RF period is complete
            # ---------------------------------------------------------------
            if (istep + 1) % steps_per_cycle == 0:        # +1 because istep starts at 0
                cycle_idx += 1

                # Update restart file
                np.save('restart.npy', self.U2)

                if cycle_idx % save_every_cycles == 0:
                    np.save(f"restart_cycle_{int(time0)+cycle_idx:04d}.npy", self.U2)


            # Update restart file
            np.save('restart.npy', self.U2)


            # print(f"CPU Time / timestep is {cpu_time.time() - start_time} seconds.")
        



    def solve(self, time0, dt, Nstep, savedata=None, verbose=False,
              rtol=1e-6, computeSensitivity=False, weak_bc=False, jac_frequency=1):


        if self.args.use_gpu==1:
            self.copy_operators_H2D(self.args.gpu_device_id)
            self.xp_module = cp
            self.params.xp_module = cp
            if (self.solveCRModel):
                self.cr.xp_module = cp
                self.cr.copy_operators_Host2Device(self.args.gpu_device_id)
        else:
            self.xp_module = np
        
        xp = self.xp_module
        # xp = np

        # Restart parameters
        save_every_cycles = 20          # <-- change to whatever you like
        steps_per_cycle = int(round(1.0 / dt))      # number of fixed-dt steps per RF cycle
        cycle_idx       = 0                         # which RF cycle we are in

        print("#")
        print("# {0:8s} {1:10s} {2:12s} {3:12s} {4:12s} {5:12s} {6:12s} {7:12s}".format(
            "Iter", "Time", "min ne", "max ne", "min Te", "max Te", "min nb", "max nb"))
        print("{0:d} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e} {7:.6e}".format(
            -1, time0, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max(),
            self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
            self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()))

        # assume initial condition has been set in U1!
        time = time0+dt
        self.step_fixed_dt(time, dt, verbose=verbose, rtol=rtol, weak_bc=weak_bc, jac_frequency=jac_frequency)
        print("{0:d} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e}  {7:.6e}".format(
            0, time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max(),
            self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
            self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()))

        if(computeSensitivity):
            self.stepSensitivity(time, dt, verbose=verbose, weak_bc=weak_bc)


        for istep in range(1, Nstep):
            # start_time = cpu_time.time()
            
            # prepare for next step
            self.U0 = xp.copy(self.U1)
            self.U1 = xp.copy(self.U2)
            time += dt

            if (computeSensitivity):
                self.A0 = xp.copy(self.A1)

            # advance
            self.step(time, dt, verbose=verbose, rtol=rtol, weak_bc=weak_bc)
            #self.filter()
            print("{0:d} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e} {7:.6e}".format(
                istep, time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()), flush=True)
                            
            if(computeSensitivity):
                self.stepSensitivity(time, dt, verbose=verbose, weak_bc=weak_bc)


            # ---------------------------------------------------------------
            # SAVE only when an RF period is complete
            # ---------------------------------------------------------------
            if (istep + 1) % steps_per_cycle == 0:        # +1 because istep starts at 0
                cycle_idx += 1

                # Update restart file
                np.save('restart.npy', self.U2)

                if cycle_idx % save_every_cycles == 0:
                    np.save(f"restart_cycle_{cycle_idx:04d}.npy", self.U2)


            # print(f"CPU Time / timestep is {cpu_time.time() - start_time} seconds.")
        


    def solveLCN(self, time0, dt, Nstep, savedata=None, verbose=False,
                 computeSensitivity=False, weak_bc=False):

        if self.args.use_gpu==1:
            self.copy_operators_H2D(self.args.gpu_device_id)
            #NOTE(malamast): add here the H2D function for the CR model
            self.xp_module = cp
        else:
            self.xp_module = np
        
        # xp = self.xp_module
        xp = np

        if(savedata!=None):
            Usave=xp.ndarray((Nstep+1,self.U2.shape[0]),dtype=xp.float64)
            TotalCurrentSave=xp.ndarray((Nstep+1,self.totalCurrent.shape[0]),dtype=xp.float64)
            IonCurrentSave=xp.ndarray((Nstep+1,self.ionCurrent.shape[0]),dtype=xp.float64)
            ElectronCurrentSave=xp.ndarray((Nstep+1,self.electronCurrent.shape[0]),dtype=xp.float64)
            Usave[0,:] = self.U2[:,0]
            TotalCurrentSave[0,:] = self.totalCurrent[:,0]
            IonCurrentSave[0,:] = self.ionCurrent[:,0]
            ElectronCurrentSave[0,:] = self.electronCurrent[:,0]

        print("#")
        print("# {0:10s} {1:12s} {2:12s} {3:12s} {4:12s}".format(
            "Time", "min ne", "max ne", "min Te", "max Te"))
        print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
            time0, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max()))

        # assume initial condition has been set in U1!
        time = time0+dt
        self.stepLCN(time, dt, verbose=verbose, weak_bc=weak_bc)
        print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
            time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max()))

        #if(computeSensitivity):
        #    self.stepSensitivity(time, dt, verbose=verbose, weak_bc=weak_bc)


        if(savedata!=None):
            Usave[1,:] = self.U2[:,0]
            TotalCurrentSave[1,:] = self.totalCurrent[:,0]
            IonCurrentSave[1,:] = self.ionCurrent[:,0]
            ElectronCurrentSave[1,:] = self.electronCurrent[:,0]

        for istep in range(1, Nstep):
            # prepare for next step
            self.U0 = xp.copy(self.U1)
            self.U1 = xp.copy(self.U2)
            time += dt

            #if (computeSensitivity):
            #    self.A0 = xp.copy(self.A1)

            # advance
            self.stepLCN(time, dt, verbose=verbose, weak_bc=weak_bc)
            print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
                time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].min(), self.U2[self.Ns*self.Np:(self.Ns+1)*self.Np].max()), flush=True)

            if(savedata!=None):
                Usave[istep+1,:] = self.U2[:,0]
                TotalCurrentSave[istep+1,:] = self.totalCurrent[:,0]
                IonCurrentSave[istep+1,:] = self.ionCurrent[:,0]
                ElectronCurrentSave[istep+1,:] = self.electronCurrent[:,0]

            #if(computeSensitivity):
            #    self.stepSensitivity(time, dt, verbose=verbose, weak_bc=weak_bc)

        if(savedata!=None):
            xp.save(savedata,Usave)
            xp.save("TotalCurrent_" + savedata, TotalCurrentSave)
            xp.save("IonCurrent_" + savedata, IonCurrentSave)
            xp.save("ElectronCurrent_" + savedata, ElectronCurrentSave)


    def plot(self, col, create=True):
        xplot, w = cheb.chebgauss(2*self.Np)

        fig = plt.figure(num=1,figsize=(16,27))
        if (create):
            ax = []
            for i in range(0,self.Ns):
                ax.append(plt.subplot(self.Ns+1,1,i+1,label='n_{0:d}'.format(i)))

            ax.append(plt.subplot(self.Ns+1,1,self.Ns+1,label='Te',sharex=ax[0]))
        else:
            ax = fig.get_axes()

        ax[0].plot(xplot, cheb.chebval(xplot, (self.V0pinv @ self.U2[0:self.Np])[:,0]), col, lw=3)
        ax[0].grid(True)
        plt.setp(ax[0].get_xticklabels(),visible=False)
        plt.setp(ax[0].get_yticklabels(),fontsize=14)
        ax[0].set_ylabel(r'$n_e$',fontsize=16)

        for i in range(1,self.Ns):
            ax[i].plot(xplot, cheb.chebval(xplot, (self.V0pinv @ self.U2[i*self.Np:(i+1)*self.Np])[:,0]), col, lw=3)
            ax[i].grid(True)
            plt.setp(ax[i].get_xticklabels(),visible=False)
            plt.setp(ax[i].get_yticklabels(),fontsize=14)
            ax[i].set_ylabel(r'$n_{0:d}$'.format(i),fontsize=16)

        ax[self.Ns].plot(xplot, cheb.chebval(xplot, (self.V0pinv @ self.U2[self.Ns*self.Np:])[:,0]), col, lw=3)
        ax[self.Ns].grid(True)
        plt.setp(ax[self.Ns].get_xticklabels(),fontsize=14)
        plt.setp(ax[self.Ns].get_yticklabels(),fontsize=14)
        ax[self.Ns].set_ylabel(r'$T_e$',fontsize=16)
        ax[self.Ns].set_xlabel(r'$x$',fontsize=16)



if __name__ == "__main__":
    desc  = "# \n"
    desc += "# chebSolver: A program for simulating glow discharge devices\n"
    desc += "#             using a 1-D, time-domain, drift-diffusion model\n"
    desc += "#             discretized with a Chebyshev-collocation in    \n"
    desc += "#             space and fully coupled implicit time marching.\n"
    desc += "#"
    print(desc)

    # Define and parse command line arguments
    import argparse
    usage = "python3 ./chebSolver"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('--Np', metavar='Np', default=100,
                        type=int, help='Number of Chebyshev points')
    parser.add_argument('--Nt', metavar='Nt', default=16,
                        type=int, help='Number of time steps')
    parser.add_argument('--dt', metavar='dt', default=0.0625,
                        type=float, help='Size of time step')
    parser.add_argument('--t0', metavar='t0', default=0.0,
                        type=float, help='Initial time')
    parser.add_argument('--scenario', metavar='scenario', default=0,
                        type=int, help='Scenario index')
    parser.add_argument('--rtol',metavar='rtol', default=1e-6,
                        type=float, help="Relative tolerance for non-linear solve")
    parser.add_argument('--restart', metavar='rst.npy', default=None,
                        help='Restart file (*.npy format, must have same Np)')
    parser.add_argument('--outfile', metavar='out.npy', default='result.npy',
                        help='Filename to save restart file')
    parser.add_argument('--savedata', metavar='save.npy',default=None,
                        help='Filename to save every time step')
    parser.add_argument('--tscheme', metavar='time_disc',default="BE",
                        help='Temporal scheme indicator [BE, CN, or LCN]')
    parser.add_argument('--verbose',default=False,
                        action='store_true', help='Be extra chatty')
    parser.add_argument('--weakbc',default=False,
                        action='store_true', help='Enforce ion flux BC weakly')
    parser.add_argument('--plot', default=False,
                        action='store_true', help="Plot the final state for inspection.")
    parser.add_argument('--V0', metavar='V0', default=100.0,
                        type=float, help='Voltage amplitude')
    parser.add_argument('--VDC', metavar='VDC', default=0.0,
                        type=float, help='Vertical shift of voltage sinusoidal')
    parser.add_argument('--elasticCollisionActivation', default=False,
                         action='store_true', help="Activate the elastic collision term.")
    parser.add_argument('--backgroundSpecieActivation', default=False,
                        action='store_true', help="Activate the background specie density equation.")
    parser.add_argument('--EinsteinForm', default=False,
                        action='store_true', help="Activate Einstein's form for diffusion coefficient for electrons.")
    parser.add_argument('--EinsteinFormIon', default=False,
                        action='store_true', help="Activate Einstein's form for diffusion coefficient for ions.")
    parser.add_argument('--IonEffEField', default=False,
                        action='store_true', help="Activate effective electric field for ions.")
    parser.add_argument('--iSample', metavar='iSample', default=0,
                        type=int, help='Sample index, if BOLSIG chemistry is used.')
    parser.add_argument('--gam', metavar='gam', default=0.01, type=float, help='Secondary Electron Emission Coefficient')
    parser.add_argument("-use_gpu", "--use_gpu", help="use GPUs", type=int, default=0)
    parser.add_argument("-gpu_device_id", "--gpu_device_id", help="GPU device id to use", type=int, default=0)
    parser.add_argument('--jacfreq', metavar='J', default=1, type=int,
                        help='Evaluate Jacobian every J Newton iterations during time step')

    parser.add_argument('--adaptive', default=False,
                        action='store_true', help="Use adaptive time-steping for time integration.")

    args = parser.parse_args()

    # Dump inputs to the screen for posterity
    print("# Input parameters:")

    print("#   Temporal scheme (tscheme)       = {0:s}".format(args.tscheme))
    print("#   Number of Chebyshev points (Np) = {0:d}".format(args.Np))
    print("#   Number of time steps (Nt)       = {0:d}".format(args.Nt))
    print("#   Size of time step (dt)          = {0:.6e}".format(args.dt))
    print("#   Initial time (t0)               = {0:.6e}".format(args.t0))
    print("#   Relative tolerance (rtol)       = {0:.6e}".format(args.rtol))

    if(args.weakbc):
        print("#")
        print("#   Imposing ion flux BC weakly.")

    if(args.restart!=None):
        print("#")
        print("#   Restarting from {0:s}".format(args.restart))
    else:
        print("#")
        print("#   No restart file provided.")
        print("#   Using uniform IC with ne = ni = 1e-4, Te = 0.5.")

    print("#   Save file time step to {0:s}".format(args.outfile))

    Ns = 3
    if(args.scenario==0):
        print("#   Running scenario = 0 (3 species, 1 rxn, Liu 2014)")
        Ns = 3
    elif(args.scenario==1):
        print("#   Running scenario = 1 (3 species, 1 rxn, PSAAP config)")
        Ns = 3
    elif(args.scenario==2):
        print("#   Running scenario = 2 (4 species, 8 rxn, Liu 2017)")
        Ns = 4
    elif(args.scenario==3):
        print("#   Running scenario = 3 (4 species, 8 rxn, Liu 2017)")
        Ns = 4
    elif(args.scenario==4):
        print('#   Running scenario = 4 (4 species, 9 rxn, 1Torr, Nominal)')
        Ns = 4
    elif(args.scenario==5):
        print('#   Running scenario = 5 (4 species, 9 rxn, 1Torr, Sampling)')
        Ns = 4
    elif(args.scenario==6):
        print('#   Running scenario = 6 (6 species, 23 rxn, 1Torr, 100V, Sampling)')
        Ns = 6
    elif(args.scenario==8):
        print('#   Running scenario = 8 (6 species, 23 rxn, 500mTorr, 100V, Sampling)')
        Ns = 6
    elif(args.scenario==9):
        print("#   Running scenario = 9 (6 species, 23 rxn)")
        Ns = 6
    elif(args.scenario==10):
        print('#   Running scenario = 10 (6species, 34 rxn, 100mTorr, Nominal)')
        Ns = 6
    elif(args.scenario==12):
        print('#   Running scenario = 12 (6 species, 23 rxn, 1Torr, 100V, Nominal)')
        Ns = 6
    elif(args.scenario==13):
        print('#   Running scenario = 13 (6 species, 23 rxn, 500mTorr, 100V, Nominal)')
        Ns = 6
    elif(args.scenario==14):
        print('#   Running scenario = 14 (6 species, 34 rxn, 1Torr, Nominal)')
        Ns = 6
    elif(args.scenario==21):
        print("#   Running scenario = 21 (4 species, 8 rxn, Liu 2017, interpolated transport)")
        Ns = 4
    elif(args.scenario==7):
        print('#   Running scenario = 7 (6 species, 34 rxn, Nominal)')
        Ns = 6 + 2   # E, Ar+, Ar+2, Ar2, Ar(m), Ar(r), Ar(4p), Ar(g) 
    elif(args.scenario==15):
        print('#   Running CR model = 15 (17 species, Nominal)')
        Ns = 1+14+2+1+1 # Ar(g), Ar(i), Ar+2, Ar2, E, Ar+
    elif(args.scenario==16):
        print('#   Running CR model = 16 (33 species, Nominal)')
        Ns = 1+30+2+1+1 # Ar(g), Ar(i), Ar+2, Ar2, E, Ar+       
    else:
        print("ERROR: Scenario = {0:d} not recognized.  Exiting.".format(args.scenario))
        exit(-1)

    elasticCollisionActivationFactor = 1.0
    if(args.elasticCollisionActivation==True):
         print("#   The elastic collision term is included.")
         elasticCollisionActivationFactor = 1.0
    else:
         print("#   The elastic collision term is not included.")
         elasticCollisionActivationFactor = 0.0

    backgroundSpecieActivationFactor = 1.0
    if(args.backgroundSpecieActivation==True):
        print("#   The background specie density is not fixed.")
        backgroundSpecieActivationFactor = 1.0
    else:
        print("#   The background specie density is fixed.")
        backgroundSpecieActivationFactor = 0.0

    EinsteinForm = True
    if(args.EinsteinForm==True):
        print("#   The Einstein's form for diffusion coefficient is used for electrons.")
        EinsteinForm = True
    else:
        print("#   The Einstein's form for diffusion coefficient is not used for electrons.")
        EinsteinForm = False

    if(args.EinsteinFormIon==True):
        print("#   The Einstein's form for diffusion coefficient is used for ions.")
    else:
        print("#   The Einstein's form for diffusion coefficient is not used for ions.")

    IonEffEField = False
    if(args.IonEffEField==True):
        print("#   An effective electric field is used for ions.")
        IonEffEField = True

    if(args.savedata!=None):
        print("#")
        print("#   Saving every time step to {0:s}".format(args.savedata))
    else:
        print("#")
        print("#   Not saving every time step (use --savedata for this option).")

    print("#")

    # Instantiate solver class
    tds = timeDomainCollocationSolver(args, Ns, 1, args.Np, elasticCollisionActivationFactor,
                                      backgroundSpecieActivationFactor, EinsteinForm, IonEffEField,
                                      gam=args.gam, V0 = args.V0, VDC = args.VDC,
                                      scenario=args.scenario, scheme=args.tscheme,
                                      iSample = args.iSample)

    # Default IC (overwritten below if we are restarting)
    initialCondition = np.zeros((tds.Np, 1),dtype=np.float64)
    initialCondition[:,0] = 1e-4 - (1e-4 - 1e-6) * (tds.xp)**2  
    for i in range(tds.Ns-1):
        tds.U1[i*tds.Np:(i+1)*tds.Np] = initialCondition
    # tds.U1[0:(tds.Ns-1)*tds.Np] = 1.0e-4           # 'usual' species
    tds.U1[(tds.Ns-1)*tds.Np:tds.Ns*tds.Np] = 1.0  # background specie
    tds.U1[tds.Ns*tds.Np:(tds.Ns+1)*tds.Np] = tds.params.EeBC*tds.U1[0:tds.Np] # electron energy
    if IonEffEField:
        # pull off state for convenience
        dens = np.zeros((tds.Np, tds.Ns),dtype=np.float64)
        for i in range(0,tds.Ns):
            dens[:,i] = tds.U1[i*tds.Np:(i+1)*tds.Np,0]                
        tds.solve_poisson(dens,args.t0+args.dt)
        tds.U1[(tds.Ns+1)*tds.Np:(tds.Ns+2)*tds.Np] = tds.phi # effective electric field for ions


    # If restart file provided, read it.
    # NOTE: currently we do a lazy restart in that only the final
    # state is saved, so we have to restart with a backward Euler step.
    if (args.restart!=None):
        tds.U1 = np.load(args.restart)

    # Initialize rest of state
    tds.U0 = np.copy(tds.U1)
    tds.U2 = np.copy(tds.U1)
    

    if args.use_gpu==1:
        gpu_device = cp.cuda.Device(args.gpu_device_id)
        gpu_device.use()

    # profile = cProfile.Profile()
    # profile.enable()
    tic = cpu_time.time()

    # Run for desired number of time steps
    if (args.tscheme=="LCN"):
        print("# ***** WARNING: Linearized Crank-Nicolson time marching is   *****")
        print("# *****          an experimental feature that may not work    *****")
        print("# *****          and is not fully supported.  Please beware.  *****")

        tds.solveLCN(args.t0, args.dt, args.Nt,
                     args.savedata, args.verbose, weak_bc=args.weakbc)
    else:
        if(args.savedata!=None):
            tds.solve_one_period(args.t0, args.dt, args.Nt,
                                 args.savedata, args.verbose, args.rtol, weak_bc=args.weakbc,
                                 jac_frequency=args.jacfreq)

        elif (args.adaptive):
            tds.solve_adaptive(args.t0, args.dt, args.Nt,
                               args.savedata, args.verbose, args.rtol, weak_bc=args.weakbc,
                               jac_frequency=args.jacfreq)
        else:
            tds.solve(args.t0, args.dt, args.Nt,
                      args.savedata, args.verbose, args.rtol, weak_bc=args.weakbc,
                      jac_frequency=args.jacfreq)

    # profile.disable()
    # profile.print_stats(sort='tottime')
    # profile.print_stats(sort='cumulative')
    # profile.print_stats(sort='line')
    # profile.print_stats(sort='nfl')

    toc = cpu_time.time()
    print(f"Total CPU Time = {toc -tic} seconds.")
    print(f"Mean CPU Time / timestep is {(toc -tic)/args.Nt} seconds.")


    # Save the result    
    np.save(args.outfile, tds.U2)
    np.save("Current_" + args.outfile, tds.totalCurrent)

    if(args.plot):
        tds.plot('b-')
        plt.show()
        
    print("Finished successfully.")    
