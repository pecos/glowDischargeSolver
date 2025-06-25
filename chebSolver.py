import numpy as np
import numpy.polynomial.chebyshev as cheb
import time
from scipy.interpolate import interp1d

from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

from Liu2014Properties import setLiu2014Properties
from psaapPropertiesTestArm import setPsaapPropertiesTestArm
from psaapPropertiesTestArmInterpTrans import setPsaapPropertiesTestArmInterpTrans
from psaapProperties_6Species_Nominal import setPsaapProperties_6Species_Nominal

from psaapProperties_8Species_1Torr_EC import setPsaapProperties_8Species_1Torr_EC
from psaapProperties_8Species_Sampling_1Torr_EC import setPsaapProperties_8Species_Sampling_1Torr_EC
from psaapProperties_8Species_250mTorr_EC import setPsaapProperties_8Species_250mTorr_EC
from psaapProperties_8Species_Sampling_250mTorr_EC import setPsaapProperties_8Species_Sampling_250mTorr_EC
from psaapProperties_8Species_500mTorr_EC import setPsaapProperties_8Species_500mTorr_EC
from psaapProperties_8Species_Sampling_500mTorr_EC import setPsaapProperties_8Species_Sampling_500mTorr_EC
from psaapProperties_8Species_5Torr_EC import setPsaapProperties_8Species_5Torr_EC
from psaapProperties_8Species_Sampling_5Torr_EC import setPsaapProperties_8Species_Sampling_5Torr_EC

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
        self.Ns = Ns # number of species
        self.Nr = Nr # number of reactions

        # charge number
        self.Z = np.zeros(Ns)
        self.Z[0] = -1 # electrons are always -1
        self.Z[1] =  1 # ions are always 1
        self.Z[2] =  0 # background specie should be 0
        
        # Ion Species Indices
        self.posIonIdx = np.where(self.Z == 1)[0]
        self.iele = [0]

        # mobility
        self.mu = np.zeros(Ns)

        # diffusivity
        self.D = np.zeros(Ns)

        # EC momentum transfer frequency
        self.nu = np.zeros(1)

        # reaction rate data

        # Modified Arrhenius rxn rate coefficients for now
        # kf(T) = A*(T**B)*exp(-C/T)
        self.A = np.zeros(Nr) #self.Ck = 272.0
        self.B = np.zeros(Nr)
        self.C = np.zeros(Nr) #18.687*(3./2.);

        # energy gain/loss in electrons
        self.dH = np.zeros(Nr) #15.7
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

        # coefficient for the elastic collision term
        self.EC = 2.0 * 0.511e6 / 37.2158e9 * 3.8e9 * (1./13.6e6)

        # DC voltage (vertical shift in the driving voltage)
        self.verticalShift = 0.0

        # electron energy Dirichlet BC
        #self.EeBC = 0.5
        self.EeBC = 0.75

        # Parameters needed to compute the current with dimensions
        self.V0Ltau  = 100 / (2.54 * 0.005 * (1./13.6e6))
        self.V0L     = 100 / (2.54 * 0.005)
        self.LLV0tau = (2.54 * 0.005)**2 / (100 * (1./13.6e6))
        self.tauL    = 2.54 * 0.005 / (1./13.6e6)
        self.np0     = 8e16             # "nominal" electron density [1/m^3]
        self.qe      = 1.6e-19          # unit charge [C]
        self.eps0    = 8.86e-12         # unit charge [C]
        self.eArea   = np.pi * 0.05**2  # electrode area [m^2]

        ## Added parameters for radial diffusion
        self.R       = 0.05 # electrode radius [m]
        self.L       = 0.02 # electrode gap [m]
        self.R_loss  = 0.05  # radial diffusion loss characteristic length in the range (0, R]

        self.reactionsList =[]
        self.diffusivityList =[]
        self.mobilityList =[]

    def charge(self,i):
        return self.Z[i]

    ## Add densities for all positively charged species
    def totPosDens(self, dens):
        posDens = np.zeros((dens.shape[0],1),dtype=np.float64)
        for ionIdx in self.posIonIdx:
            posDens[:,] += dens[:,[ionIdx]]
        return posDens

    def mobility(self, i, energy, nb):
        mu = np.zeros((nb.shape[0],1),dtype=np.float64)

        if (len(self.mobilityList)>i and self.mobilityList[i].interpolate):
            indFix = (energy[:,0]<=0.0)
            energy[indFix,0] = 0.0

            indFix = (energy[:,0]>10.0)
            energy[indFix,0] = 10.0

            mu = self.mobilityList[i].mu_expression((2./3)*energy[:,[i]]) / nb
        else:
            mu[:,0] = self.mu[i] / nb

        return mu[:,0]

    def mobility_U(self, i, j, energy, energy_U, mu, nb):
        mu_U = np.zeros((nb.shape[0],nb.shape[0]),dtype=np.float64)

        if (len(self.mobilityList) > i and self.mobilityList[i].interpolate):
            indFixL = (energy[:,i]<=0.0)
            energy[indFixL,i] = 0.0
            energy_U[i,j,indFixL,:] = 0.0

            indFixH = (energy[:,i]>10.0)
            energy[indFixH,i] = 10.0
            energy_U[i,j,indFixH,:] = 0.0

            mu_ee = (2./3)*self.mobilityList[i].mu_T_expression((2./3)*energy[:,i]) / nb
            mu_U_tmp = mu_ee * np.diag(energy_U[i,j,:,:])
            mu_U[:,:] = np.diag(mu_U_tmp)

        if (j == self.Ns - 1):
            mu_U -= np.diag(mu[:,i] / nb)

        return mu_U

    def diffusivity(self, i, energy, mu, nb, EinsteinForm):
        DEf = np.zeros((energy.shape[0],1),dtype=np.float64)

        if (len(self.diffusivityList) > i and self.diffusivityList[i].interpolate):
            indFix = (energy[:,0]<=0.0)
            energy[indFix,0] = 0.0

            indFix = (energy[:,0]>10.0)
            energy[indFix,0] = 10.0

            DEf[:,0] = self.diffusivityList[i].D_expression((2./3)*energy[:,i]) / nb

        elif EinsteinForm and self.Z[i] == -1:
            V0 =  self.qStar * 1.0 # V0 = qStar * 1eV
            DEf = 2.0 / 3.0 * np.multiply(energy[:,[i]], mu[:,[i]]) / V0

        else:
            DEf[:,0] = self.D[i] / nb

        return DEf[:,0]

    def diffusivity_U(self, i, j, energy, energy_U, mu, D, nb, EinsteinForm):
        D_U = np.zeros((energy_U.shape[2], energy_U.shape[2]),dtype=np.float64)

        if (len(self.diffusivityList) > i and self.diffusivityList[i].interpolate):
            indFixL = (energy[:,i]<=0.0)
            energy[indFixL,i] = 0.0
            energy_U[i,j,indFixL,:] = 0.0

            indFixH = (energy[:,i]>10.0)
            energy[indFixH,i] = 10.0
            energy_U[i,j,indFixH,:] = 0.0

            D_ee = (2./3)*self.diffusivityList[i].D_T_expression((2./3)*energy[:,i]) / nb
            D_U_tmp = D_ee * np.diag(energy_U[i,j,:,:])
            D_U[:,:] = np.diag(D_U_tmp)

        elif EinsteinForm and self.Z[i] == -1:
            V0 =  self.qStar * 1.0 # V0 = qStar * 1eV
            D_U = 2.0 / 3.0 * np.multiply(mu[:,[i]], energy_U[i,j,:,:]) / V0
            
            if (len(self.mobilityList) > i and self.mobilityList[i].interpolate):
                mu_ee = (2./3)*self.mobilityList[i].mu_T_expression((2./3)*energy[:,i]) / nb
                mu_U_tmp = mu_ee * np.diag(energy_U[i,j,:,:]) * 2.0/3.0 * energy[:,i] / V0
                D_U[:,:] += np.diag(mu_U_tmp)


        if (j == self.Ns - 1):
            D_U[:,:] -= np.diag(D[:,i] / nb)

        return D_U


    def momFrequency(self, energy, nb):
        nu = np.zeros((nb.shape[0],1),dtype=np.float64)

        if (self.EC.interpolate):
            nu = self.EC.nu_expression((2./3)*energy[:,[0]]) * nb

        return nu[:,0]

    def momFrequency_U(self, j, energy, energy_U, nu, nb):
        nu_U = np.zeros((nb.shape[0],nb.shape[0]),dtype=np.float64)

        if (self.EC.interpolate):
            nu_ee = (2./3)*self.EC.nu_T_expression((2./3)*energy[:,0]) * nb
            nu_U_tmp = nu_ee * np.diag(energy_U[0,j,:,:])
            nu_U[:,:] = np.diag(nu_U_tmp)

        if (j == self.Ns - 1):
            nu_U += np.diag(nu[:,0])

        return nu_U

    def rxnSourceTerm(self, energy, density):
        G = self.progressRate(energy,density)

        omega = np.zeros((energy.shape[0], self.Ns+1),dtype=np.float64)
        for i in range(0,self.Ns):
            for j in range(0,self.Nr):
                omega[:,i] += (self.reactionsList[j].rxnBeta[i,0] - self.reactionsList[j].rxnAlfa[i,0])*G[:,j]

        for j in range(0,self.Nr):
            omega[:,self.Ns] -= self.dH[j]*G[:,j]

        return omega

    def rxnSourceTermJac(self, energy, density):
        G_U = self.progressRateJac(energy,density)

        omega_U = np.zeros((self.Ns+1,self.Ns+1,energy.shape[0]),dtype=np.float64)
        for i in range(0,self.Ns):
            for j in range(0,self.Nr):
                omega_U[i,:,:] += (self.reactionsList[j].rxnBeta[i,0] - self.reactionsList[j].rxnAlfa[i,0])*G_U[j,:,:]

        for j in range(0,self.Nr):
            omega_U[self.Ns,:,:] -= self.dH[j]*G_U[j,:,:]

        return omega_U

    def progressRate(self, energy, density):
        G = np.zeros((energy.shape[0],self.Nr))
        for i in range(0,self.Nr):
            kf = self.rxnRateCoefficient(energy, i)
            G[:,i] = kf[:,0]
            for j in range(0,self.Ns):
                #print('Species #: {}'.formatg(j+1))
                if (self.reactionsList[i].rxnAlfa[j,0]>0):
                    #print('Density:')
                    #print(density[:,j])
                    #print('')
                    #print('Stoic. Coeffs:')
                    #print(self.reactionsList[i].rxnAlfa[j,0])
                    #print('')
                    G[:,i] *= density[:,j]**self.reactionsList[i].rxnAlfa[j,0]
        #for i in range(len(G)):
        #    print('{:2E}'.format(G[i,0]))
        return G

    def progressRateJac(self, energy, density):
        G = self.progressRate(energy,density)
        G_U = np.zeros((self.Nr,self.Ns+1, energy.shape[0]))
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

        indFix = (energy[:,0]<=0.0)
        energy[indFix,0] = 1.0
        if self.reactionsList[i].rxnBolsig:
            kf = np.exp(self.reactionsList[i].kf_log(np.log(energy)))
        else:
            kf = self.reactionsList[i].kf(energy)
        kf[indFix,0] = 0

        #if self.reactionsList[i].rxnBolsig and np.any(np.isinf(kf)):
            #print("RXN #{}".format(i+1))
            #print("ENERGY VALUES:")
            #print(energy)
            #print("")
            #print("k_f:")
            #print(kf)

        return kf #a * (energy**b) * np.exp(-Ea/energy)

    def rxnRateCoefficientJac(self, energy, i):
        """Returns derivative of ionization reaction rate constant wrt
        energy
        """

        indFix = (energy[:,0]<=0.0)
        energy[indFix,0] = 1.0
        if self.reactionsList[i].rxnBolsig:
            kf_T = self.reactionsList[i].kf_T_log(np.log(energy)) \
                 * np.exp(self.reactionsList[i].kf_log(np.log(energy))) / energy
        else:
            kf_T = self.reactionsList[i].kf_T(energy)
        kf_T[indFix,0] = 0

        return kf_T #a * (energy**(b-1)) * np.exp(-Ea/energy) * (b + Ea/energy)


    def radialDiffSourceTerm(self, i, density, D):
        s_dot = np.zeros((density.shape[0], 1), dtype = np.float64)

        s_dot[:,0] = (-2.0 / (self.R * self.R_loss)) * np.multiply(D[:,i], density[:,i])

        return s_dot[:,0]

    def radialDiffSourceTermJac(self, i, j, density, D, D_U):
        s_dot_U = np.zeros((density.shape[0], density.shape[0]), dtype = np.float64)

        s_dot_U_tmp = (-2.0 / (self.R * self.R_loss)) * np.multiply(density[:,i], np.diag(D_U[i,j,:,:]))
        if i == j:
            s_dot_U_tmp += (-2.0 / (self.R * self.R_loss)) * D[:,i]

        s_dot_U[:,:] = np.diag(s_dot_U_tmp)

        return s_dot_U


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
        print("#   ksion = {0:.6e}".format(self.ksion))
        print("#   Te BC = {0:.6e}".format(self.EeBC*(2./3.)))
        print("#   gam   = {0:.6e}".format(self.gam))
        print('#   Z     = ', self.Z)
        print('#   Lloss = ', self.R_loss)
        if len(self.posIonIdx) > 1:
            print('#   MULTIPLE Ion Species Detected! \n')
        else:
            print('#   SINGLE Ion Species: ', self.posIonIdx, '\n')



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

    def __init__(self, Ns, NT, Np, elasticCollisionActivationFactor,
                 backgroundSpecieActivationFactor, EinsteinForm,
                 radialDiffusionActivationFactor,
                 gam=0.01, V0 = 100.0, VDC = 0.0,
                 scenario=0, scheme="BE", iSample = 0):
        """Initializes storage and operaters required for solve."""

        # parameters of the time marching scheme
        self.temporal_scheme = scheme
        if not (self.temporal_scheme in ["BE", "CN", "LCN"]):
            print("ERROR: Unrecognized temporal scheme.")
            print("Please use 'BE' (backward Euler), 'CN' (Crank-Nicolson), or 'LCN' (linearized Crank-Nicolson).")
            exit(-1)

        self.Ns = Ns    # Number of species
        self.NT = NT    # Number of temperatures
        self.Nv = Ns+NT # Total number of 'state' variables

        self.deg = Np-1 # degree of Chebyshev polys we use
        self.Np = Np # Number of points used to define state in space
        self.Nc = Np-2 # number of collocation pts (Np-2 b/c BCs)

        self.Ndof = self.Nv*self.Np # total number of dofs

        # state (3 vectors for BDF2)
        self.U2 = np.zeros((self.Ndof,1))
        self.U1 = np.zeros((self.Ndof,1))
        self.U0 = np.zeros((self.Ndof,1))
        

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
        elif(scenario==7):
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
        elif(scenario==16):
            Nr = 23
        elif(scenario==21):
            Nr = 8
        else:
            print("ERROR: scenario = {} not understood.".format(scenario))
            exit(-1)

        self.elasticCollisionActivationFactor = elasticCollisionActivationFactor
        self.backgroundSpecieActivationFactor = backgroundSpecieActivationFactor
        self.EinsteinForm = EinsteinForm
        self.radialDiffusionActivationFactor = radialDiffusionActivationFactor

        self.params = modelClosures(self.Ns, Nr)

        if(scenario==0):
            setLiu2014Properties(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==1):
            setPsaapProperties_6Species_100mTorr_Expanded(gam, V0, VDC, self.params, Nr, iSample)
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
        elif(scenario==7):
            setPsaapProperties_6Species_Sampling_250mTorr(gam, V0, VDC, self.params, Nr, iSample)
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
        elif(scenario==15):
            setPsaapProperties_6Species_Sampling_1Torr_Expanded(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==16):
            setPsaapProperties_6Species_5Torr(gam, V0, VDC, self.params, Nr, iSample)
        elif(scenario==21):
            setPsaapPropertiesTestArmInterpTrans(gam, V0, VDC, self.params, Nr, iSample)

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
        self.LpD = np.identity(self.Np)
        self.LpD[1:-1,:] = self.Lp[1:-1,:]
        self.LpD_inv = np.linalg.solve(self.LpD, np.eye(self.Np))

        # solve poisson equation for phi_ne
        ident0 = np.identity(self.Np)
        ident0[0,0] = ident0[-1,-1] = 0.0
        self.phi_ni = np.dot(self.LpD_inv, -self.params.alpha*ident0)
        self.phi_ne = -self.phi_ni

        self.phi_x_ne = self.Dp @ self.phi_ne
        self.phi_x_ni = self.Dp @ self.phi_ni

        self.I_Np =  np.identity(self.Np)
        self.I_Ndof = np.identity(self.Ndof)

        self.ones_Np =  np.ones(self.Np)

        self.ntot_U = np.zeros((self.Np, self.Nv))
        # all but background
        for i in range(1, self.Ns-1):
            self.ntot_U[:,i] += self.ones_Np
        # background contribution
        self.ntot_U[:,self.Ns-1] += self.params.nAronp0*self.ones_Np


        self.totalCurrent    = np.zeros((2,1),dtype=np.float64)
        self.electronCurrent = np.zeros((2,1),dtype=np.float64)
        self.ionCurrent      = np.zeros((2,len(self.params.posIonIdx)),dtype=np.float64)
        

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

    def solve_poisson(self, dens, time):
        """Solve Gauss' law for the electric potential.

        Inputs:
          ne   : Values of electron density at xp
          ni   : Values of ion density at xp
          time : Current time

        Outputs: None (sets self.phi to computed potential)
        """
        #r = -self.params.alpha*(ni-ne)
        posDens = self.params.totPosDens(dens)
        r = -self.params.alpha*(posDens[:,] - dens[:,self.params.iele])
        r[0] = 0.0
        r[-1] = np.sin(2*np.pi*time) + self.params.verticalShift
        #self.phi = np.linalg.solve(self.LpD, r)
        self.phi = np.dot(self.LpD_inv, r)
    
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
        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        #iele = [0]
        #iion = [1]
        # pull off state for convenience
        dens = np.ndarray((self.Np, self.Ns),dtype=np.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]
        
        # Floor densities (added 03/13)
        #densFloor = np.full(dens.shape, (1.0e8/self.params.np0), dtype=np.float64)
        #dens = np.where(dens < densFloor, densFloor, dens)

        nT = np.zeros((self.Np, 1),dtype=np.float64)
        nT = Uin[self.Ns*self.Np:] # assumes just 1 temperature!
        Te = nT/dens[:,self.params.iele]

        ntot = np.zeros((self.Np, 1),dtype=np.float64)

        # add all heavies but background
        for i in range(1, self.Ns-1):
            ntot[:,0] += dens[:,i]

        # add background contribution (accounting for non-dim difference)
        ntot[:,0] += self.params.nAronp0 * dens[:,self.Ns-1]

        # Temperature (from ideal gas law)
        Tg = np.zeros((self.Np, 1),dtype=np.float64)
        Tg = (self.params.p0 - nT)/ntot

        # Floor Te (added 03/13)
        Te = np.where(Te < Tg, Tg, Te)

        # solve poisson equation for phi
        # now have self.phi
        self.solve_poisson(dens,time)

        # form fluxes at grid points
        dens_x = self.Dp @ dens
        nT_x   = self.Dp @ nT
        phi_x  = self.Dp @ self.phi

        energy = np.zeros((self.Np, self.Ns),dtype=np.float64)
        mu     = np.zeros((self.Np, self.Ns),dtype=np.float64)
        diffusivity = np.zeros((self.Np, self.Ns),dtype=np.float64)
        nu = np.zeros((self.Np, 1), dtype = np.float64)
        energy[:,0] = Te[:,0]
        for i in range(1,self.Ns):
            energy[:,i] = Tg[:,0]

        for i in range(0,self.Ns):
            mu[:,i]  = self.params.mobility(i, energy, dens[:,self.Ns-1])
            diffusivity[:,i] = self.params.diffusivity(i, energy, mu,
                                                       dens[:,self.Ns-1],
                                                       self.EinsteinForm)
        nu[:,0] = self.params.momFrequency(energy, dens[:,self.Ns-1])
        ## Ambipolar diffusion coefficient for electrons:
        #diffusivity[:,0] = (diffusivity[:,0] + np.multiply(np.divide(diffusivity[:,1], mu[:,1]), mu[:,0])) / (1 + (Te[:,0] / Tg[:,0]))

        fspec = np.ndarray((self.Np, self.Ns),dtype=np.float64)
        for i in range(0,self.Ns):
            fspec[:,i] = (   self.params.charge(i)*np.multiply(mu[:,i], dens[:,i])*(-phi_x[:,0])
                           - np.multiply(diffusivity[:,i],dens_x[:,i]) )
        
        fT = np.zeros((self.Np, 1),dtype=np.float64)
        fT[:,0] = (5./3.)*(-mu[:,0]*nT[:,0]*(-phi_x[:,0]) -  np.multiply(diffusivity[:,0], nT_x[:,0])) # <- Need to change for energy transport

        # overwrite endpoints in fi (weakly impose BC)
        #fspec[ 0,1] = -self.params.ksion*dens[ 0,[1]] + mu[0,1]*dens[ 0,[1]]*(-phi_x[ 0])
        #fspec[-1,1] =  self.params.ksion*dens[-1,[1]] + mu[-1,1]*dens[-1,[1]]*(-phi_x[-1])
        # BCs for second ion flux (hard-coded for now)
        #if self.Ns == 5:
        #    fspec[ 0,2] = -self.params.ksion*dens[ 0,[2]] + mu[ 0,2]*dens[ 0,[2]]*(-phi_x[ 0])
        #    fspec[-1,2] =  self.params.ksion*dens[-1,[2]] + mu[-1,2]*dens[-1,[2]]*(-phi_x[-1])
        for ionIdx in self.params.posIonIdx:
            fspec[ 0,ionIdx] = -self.params.ksion*dens[ 0,[ionIdx]] + mu[ 0,ionIdx]*dens[ 0,[ionIdx]]*(-phi_x[ 0])
            fspec[-1,ionIdx] =  self.params.ksion*dens[-1,[ionIdx]] + mu[-1,ionIdx]*dens[-1,[ionIdx]]*(-phi_x[-1])
        
        # overwrite endpoints in fe (weakly impose BC)
        rstrg = np.zeros(2)
        if (weak_bc):
            #fspec[ 0,0] = (- self.params.ks*dens[ 0,iele] * Te[0,0]**0.5
            #               - self.params.gam*fspec[ 0,iion])
            #fspec[-1,0] = (+ self.params.ks*dens[-1,iele] * Te[-1,0]**0.5
            #               - self.params.gam*fspec[-1,iion])
            fspec[ 0,0] = -self.params.ks*dens[ 0,self.params.iele] * Te[ 0,0]**0.5
            fspec[-1,0] = +self.params.ks*dens[-1,self.params.iele] * Te[-1,0]**0.5

            #fspec[ 0,0] += -self.params.gam*fspec[ 0,[1]]
            #fspec[-1,0] +=  self.params.gam*fspec[-1,[1]]
            # Contribution from second ion (hard-coded for now)
            #if self.Ns == 5:
            #    fspec[ 0,0] += -self.params.gam*fspec[ 0,[2]]
            #    fspec[-1,0] +=  self.params.gam*fspec[-1,[2]]
            for ionIdx in self.params.posIonIdx:
                fspec[ 0,0] += -self.params.gam*fspec[ 0,[ionIdx]]
                fspec[-1,0] +=  self.params.gam*fspec[-1,[ionIdx]]

        else:
            #rstrg[0] = fspec[ 0,iele] - (- self.params.ks*dens[ 0,iele] * Te[0,0]**0.5
            #                             - self.params.gam*fspec[ 0,iion])
            #rstrg[1] = fspec[-1,iele] - (+ self.params.ks*dens[-1,iele] * Te[-1,0]**0.5
            #                             - self.params.gam*fspec[-1,iion])
            rstrg[0] = fspec[ 0,self.params.iele] - (- self.params.ks*dens[ 0,self.params.iele] * Te[ 0,0]**0.5)
            rstrg[1] = fspec[-1,self.params.iele] - (+ self.params.ks*dens[-1,self.params.iele] * Te[-1,0]**0.5)

            #rstrg[0] -= -self.params.gam*fspec[ 0,[1]]
            #rstrg[1] -= -self.params.gam*fspec[-1,[1]]
            # Contribution from second ion (hard-coded for now)
            #if self.Ns == 5:
            #    rstrg[0] -= -self.params.gam*fspec[ 0,[2]]
            #    rstrg[1] -= -self.params.gam*fspec[-1,[2]]
            for ionIdx in self.params.posIonIdx:
                rstrg[0] -= (- self.params.gam*fspec[ 0,[ionIdx]])
                rstrg[1] -= (- self.params.gam*fspec[-1,[ionIdx]])

        # form derivatives of fluxes at collocation points
        fspec_x = self.Dp @ fspec
        fT_x = self.Dp @ fT

        # form source terms at collocation points
        omega = self.params.rxnSourceTerm(Te, dens)
        SJ = -self.params.qStar*fspec[:,self.params.iele]*(-phi_x)

        # elastic collision term at collocation points
        SEC  = nu[:,] * (nT - np.multiply(dens[0, self.params.iele], Tg))
        #SEC = -self.params.EC * (nT - np.multiply(dens[0, self.params.iele], Tg)) * np.sqrt(Te)
        SEC *= self.elasticCollisionActivationFactor

        ## Radial difussion source term (all heavy species)
        s_dot_RD = np.zeros((self.Np, self.Ns), dtype = np.float64)
        for i in range(1, self.Ns-1):
            s_dot_RD[:,i] = self.params.radialDiffSourceTerm(i, dens, diffusivity)

        # Electron radial fluxes should be equal to the sum of positive ion fluxes
        for ionIdx in self.params.posIonIdx:
            s_dot_RD[:,0] += s_dot_RD[:,ionIdx]
        
        s_dot_RD *= self.radialDiffusionActivationFactor

        ## Radial Diffusion Electron Energy source term
        SERD = np.multiply(s_dot_RD[:,self.params.iele], energy[:,self.params.iele])
        
        # evaluate S---the source term required in the background
        # specie evolution to ensure constant pressure
        fa = np.copy(fT)
        for i in range(1,self.Ns-1):
            fa[:,0] += (5./3.)*(self.params.charge(i) * np.multiply(mu[:,i],
                                np.multiply(dens[:,i],Tg[:,0]) * (-phi_x[:,0]))
                                - np.multiply(diffusivity[:,i], (self.Dp @ np.multiply(dens[:,i],Tg[:,0]))))

        # background thermal conductivity contribution
        fa[:,0] += - self.params.kappaB * (self.Dp @ Tg[:,0])

        fa_x = self.Dp @ fa

        sOmEp = np.zeros((self.Np,1),dtype=np.float64)
        for i in range(0, self.Ns-1):
            sOmEp[:,0] += omega[:,i]*self.params.dEps[i]

        joule = np.zeros((self.Np,1),dtype=np.float64)
        for i in range(0, self.Ns-1):
            joule[:,0] += self.params.qStar*self.params.charge(i)*fspec[:,i]*(-phi_x[:,0])

        S = np.zeros((self.Np,1),dtype=np.float64)
        S_RD = np.zeros((self.Np,1), dtype=np.float64)
        S[:,0] = (sOmEp[:,0] + fa_x[:,0] - joule[:,0])/Tg[:,0]/self.params.nAronp0

        ## All heavy species that diffuse to side-wall are quenched and converted to Ar
        for i in range(1, self.Ns-1):
            S_RD[:,0] -= s_dot_RD[:,i] / Tg[:,0] / self.params.nAronp0

        # form full residual
        res = np.zeros((self.Nv*self.Np,1))

        # spatial part
        # standard species
        for i in range(0,self.Ns-1):
            res[i*self.Np:(i+1)*self.Np,0] = dt*(fspec_x[:,i] - omega[:,i] - s_dot_RD[:,i])
        
        # background specie (fixed at IC for now)
        res[(self.Ns-1)*self.Np:self.Ns*self.Np] = -dt*(S + S_RD)
        res[(self.Ns-1)*self.Np:self.Ns*self.Np,0] *= self.backgroundSpecieActivationFactor

        # energy
        res[self.Ns*self.Np:]        = dt*(fT_x - omega[:,[self.Ns]] - SJ  - SEC - SERD)

        ############################################################
        # Computation of total, displacement, and particle current #
        ############################################################
        E_currentTimeStep = - phi_x

        # pull off state for convenience
        dens_previousTimeStep = np.ndarray((self.Np, self.Ns),dtype=np.float64)
        for i in range(0,self.Ns):
            dens_previousTimeStep[:,i] = self.U1[i*self.Np:(i+1)*self.Np,0]

        # solve poisson equation for phi
        # now have self.phi
        self.solve_poisson(dens_previousTimeStep,time)

        # form fluxes at grid points
        E_previousTimeStep  = -self.Dp @ self.phi

        displacementCurrent = self.params.eps0 \
            * (E_currentTimeStep - E_previousTimeStep) / dt * self.params.V0Ltau

        particleCurrent = np.zeros((2,self.Ns),dtype=np.float64)
        particleCurrent[ 0,self.params.iele] = (-self.params.ks * self.params.tauL \
                * dens[ 0,self.params.iele] * self.params.np0 * Te[ 0,0]**0.5 * self.params.qe) * self.params.charge(self.params.iele)
        particleCurrent[-1,self.params.iele] = (+self.params.ks * self.params.tauL \
                * dens[-1,self.params.iele] * self.params.np0 * Te[-1,0]**0.5 * self.params.qe) * self.params.charge(self.params.iele)

        for ionIdx in self.params.posIonIdx:
            particleCurrent[ 0,[ionIdx]] = mu[ 0,ionIdx] * self.params.LLV0tau \
                    * dens[ 0,[ionIdx]] * self.params.np0 * (-phi_x[ 0]) * self.params.V0L \
                    * self.params.qe * self.params.charge(ionIdx)
            particleCurrent[-1,[ionIdx]] = mu[-1,ionIdx] * self.params.LLV0tau \
                    * dens[-1,[ionIdx]] * self.params.np0 * (-phi_x[-1]) * self.params.V0L \
                    * self.params.qe * self.params.charge(ionIdx)

            particleCurrent[ 0,self.params.iele] += -self.params.gam * particleCurrent[ 0,[ionIdx]] * self.params.charge(self.params.iele)
            particleCurrent[-1,self.params.iele] += -self.params.gam * particleCurrent[-1,[ionIdx]] * self.params.charge(self.params.iele)

        self.totalCurrent[ 0,0] = displacementCurrent[ 0] + particleCurrent[ 0,self.params.iele]
        self.totalCurrent[-1,0] = displacementCurrent[-1] + particleCurrent[-1,self.params.iele]
        for ionIdx in self.params.posIonIdx:
            self.totalCurrent[ 0,0] += particleCurrent[ 0,[ionIdx]]
            self.totalCurrent[-1,0] += particleCurrent[-1,[ionIdx]]
            
            self.ionCurrent[:,[ionIdx-1]] = particleCurrent[:,[ionIdx]]
        #particleCurrent[0,iion]  = mu[0,1] * self.params.LLV0tau \
        #    * dens[ 0,iion] * self.params.np0 * (-phi_x[ 0]) * self.params.V0L \
        #        * self.params.qe * self.params.charge(1)
        #particleCurrent[-1,iion] = mu[-1,1] * self.params.LLV0tau \
        #    * dens[-1,iion] * self.params.np0 * (-phi_x[-1]) * self.params.V0L \
        #        * self.params.qe * self.params.charge(1)
            
        #particleCurrent[0,iele]  = (- self.params.ks * self.params.tauL \
        #    *  dens[ 0,iele] * self.params.np0 * Te[0,0]**0.5 * self.params.qe \
        #    - self.params.gam * particleCurrent[0,iion]) * self.params.charge(0)
        #particleCurrent[-1,iele] = (+ self.params.ks * self.params.tauL \
        #    * dens[-1,iele] * self.params.np0 * Te[-1,0]**0.5 * self.params.qe \
        #    - self.params.gam * particleCurrent[-1,iion]) * self.params.charge(0)
        #self.totalCurrent[ 0,0] = displacementCurrent[ 0] \
        #    + particleCurrent[ 0,iion] + particleCurrent[ 0,iele]
        #self.totalCurrent[-1,0] = displacementCurrent[-1] \
        #    + particleCurrent[-1,iion] + particleCurrent[-1,iele]
        #self.ionCurrent[:]      = particleCurrent[:,iion]
        self.electronCurrent[:] = particleCurrent[:,self.params.iele]
        
        #print('Total Current: ', self.totalCurrent, '\n')
        #for ionIdx in self.params.posIonIdx:
            #print('Ion Current: ', self.ionCurrent[:,ionIdx-1], '\n')
        #print('Electron Current: ', self.electronCurrent, '\n')
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
        res, rstrg = self.spatial_residual(Uin, time, dt, weak_bc)

        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        #iele = [0]
        #iion = [1]

        # pull off state for convenience
        dens = np.ndarray((self.Np, self.Ns),dtype=np.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = Uin[self.Ns*self.Np:] # assumes just 1 temperature!
        Te = nT/dens[:,self.params.iele]

        # time derivative part (backward Euler)
        res += Uin - self.U1

        # boundary conditions (strongly enforced)

        # electron flux
        if (not weak_bc):
            res[0]           = rstrg[0] #fspec[ 0,iele]  - (-self.params.ks*dens[ 0,iele] - self.params.gam*fspec[ 0,iion])
            res[self.Np-1]   = rstrg[1] #fspec[-1,iele]  - ( self.params.ks*dens[-1,iele] - self.params.gam*fspec[-1,iion])

        for i in range(len(self.params.posIonIdx)+1,self.Ns-1):
            res[i*self.Np  ] = dens[ 0,i] - 0.0
            res[(i+1)*self.Np-1] = dens[-1,i] - 0.0

        # if solving for background density, enforce Dirichlet condition on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            ntot = np.zeros(self.Np)

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
        res[self.Ns*self.Np  ] = (nT[ 0] - self.params.EeBC * dens[0,self.params.iele])
        res[(self.Ns+1)*self.Np-1] = (nT[-1] - self.params.EeBC * dens[-1,self.params.iele])

        #print(res[0], res[149], '\n')
        return res

    def residualCN(self, Uin, time, dt, weak_bc=False):
        """Evaluates the residual for Crank-Nicolson.  See
        timeDomainCollocationSolver.residua() for additional documentation.
        """
        res0, rstrgold = self.spatial_residual(self.U1, time-dt, dt, weak_bc)
        res1, rstrg    = self.spatial_residual(    Uin, time   , dt, weak_bc)
        res = 0.5*(res0+res1)

        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        #iele = [0]
        #iion = [1]

        # pull off state for convenience
        dens = np.ndarray((self.Np, self.Ns),dtype=np.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = Uin[self.Ns*self.Np:] # assumes just 1 temperature!
        Te = nT/dens[:,self.params.iele]

        # time derivative part (backward Euler)
        res += Uin - self.U1

        # boundary conditions (strongly enforced)

        # electron flux
        if (not weak_bc):
            res[0]           = rstrg[0] #fspec[ 0,iele]  - (-self.params.ks*dens[ 0,iele] - self.params.gam*fspec[ 0,iion])
            res[self.Np-1]   = rstrg[1] #fspec[-1,iele]  - ( self.params.ks*dens[-1,iele] - self.params.gam*fspec[-1,iion])

        for i in range(len(self.params.posIonIdx)+1,self.Ns-1):
            res[i*self.Np  ] = dens[ 0,i] - 0.0
            res[(i+1)*self.Np-1] = dens[-1,i] - 0.0

        # if solving for background density, enforce Dirichlet condition on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            ntot = np.zeros(self.Np)

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
        res[self.Ns*self.Np  ] = (nT[ 0] - self.params.EeBC*dens[0,self.params.iele])
        res[(self.Ns+1)*self.Np-1] = (nT[-1] - self.params.EeBC*dens[-1,self.params.iele])

        return res

    def residualLCN(self, Uin, time, dt, weak_bc=False):
        """Evaluates the residual for linearized Crank-Nicolson.  See
        timeDomainCollocationSolver.residua() for additional documentation.
        """
        res0, rstrgold = self.spatial_residual(self.U1, time-dt, dt, weak_bc)
        res1, rstrg    = self.spatial_residual(    Uin, time   , dt, weak_bc)
        res = 0.5*(res0+res1)

        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        #iele = [0]
        #iion = [1]

        # pull off state for convenience
        dens = np.ndarray((self.Np, self.Ns),dtype=np.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = Uin[self.Ns*self.Np:] # assumes just 1 temperature!
        Te = nT/dens[:,self.params.iele]

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
            for i in range(len(self.params.posIonIdx)+1, self.Ns-1):
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
        # indices of electrons/ions (in list s.t. dens[:,iele].shape = (Np,1)
        #iele = [0]
        #iion = [1]

        # pull off state for convenience
        dens = np.ndarray((self.Np, self.Ns),dtype=np.float64)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]
        
        # Floor Densities (added on 03/12/24)
        #print(1.0e9/self.params.np0)
        #densFloor = np.full(dens.shape, (1.0e8/self.params.np0), dtype=np.float64)
        #dens = np.where(dens < densFloor, densFloor, dens)

        nT = Uin[self.Ns*self.Np:] # assumes just 1 temperature!
        Te = nT/dens[:,self.params.iele]

        Te_ne = -np.multiply(Te/dens[:,self.params.iele],np.identity(self.Np))
        Te_nT = np.multiply(np.identity(self.Np),1./dens[:,self.params.iele])

        ntot = np.zeros((self.Np,1),dtype=np.float64)
        ntot_U = np.zeros((self.Np, self.Nv))

        # all but background
        for i in range(1, self.Ns-1):
            ntot[:,0] += dens[:,i]
            ntot_U[:,i] += np.ones(self.Np)

        # background contribution
        ntot[:,0] += self.params.nAronp0 * dens[:,self.Ns-1]
        ntot_U[:,self.Ns-1] += self.params.nAronp0*np.ones(self.Np)

        # Temperature (from ideal gas law)
        Tg = np.zeros((self.Np, 1),dtype=np.float64)
        Tg = (self.params.p0 - nT)/ntot
        
        # Floor Te (added on 03/12/24)
        Te = np.where(Te < Tg, Tg, Te) 

        Tg_U = np.zeros((self.Np, self.Nv))
        for i in range(0, self.Nv):
            Tg_U[:,i] = -(Tg[:,0]/ntot[:,0])*ntot_U[:,i]

        Tg_U[:,-1] += -np.ones(self.Np)/ntot[:,0]

        #print("Mean gas temperature = {0:.6e}".format((2./3)*np.mean(Tg)*11604.))

        # force solving poisson equation again
        if (solve_poisson):
            self.solve_poisson(dens ,time)

        energy = np.zeros((self.Np, self.Ns),dtype=np.float64)
        mu     = np.zeros((self.Np, self.Ns),dtype=np.float64)
        nu     = np.zeros((self.Np, 1), dtype=np.float64)
        diffusivity = np.zeros((self.Np, self.Ns),dtype=np.float64)
        energy[:,0] = Te[:,0]
        for i in range(1,self.Ns):
            energy[:,i] = Tg[:,0]

        for i in range(0,self.Ns):
            mu[:,i]  = self.params.mobility(i, energy, dens[:,self.Ns-1])
            diffusivity[:,i] = self.params.diffusivity(i, energy, mu,
                                                       dens[:,self.Ns-1],
                                                       self.EinsteinForm)
        nu[:,0] = self.params.momFrequency(energy, dens[:,self.Ns-1])

        energy_U = np.zeros((self.Ns, self.Nv, self.Np, self.Np),dtype=np.float64)
        energy_U[0,0,:,:] = Te_ne
        energy_U[0,self.Ns,:,:] = Te_nT
        for i in range(1,self.Ns):
            for j in range(1,self.Nv):
                energy_U[i,j,:,:] = np.multiply(np.identity(self.Np),Tg_U[:,j])

        diffusivity_U = np.zeros((self.Ns, self.Nv, self.Np, self.Np),dtype=np.float64)
        mu_U = np.zeros((self.Ns, self.Nv, self.Np, self.Np),dtype=np.float64)
        nu_U = np.zeros((self.Nv, self.Np, self.Np), dtype = np.float64)
        for i in range(0,self.Ns):
            for j in range(0,self.Nv):
                diffusivity_U[i,j,:,:] = self.params.diffusivity_U(i, j,
                                                                   energy, energy_U,
                                                                   mu, diffusivity, dens[:,self.Ns-1],
                                                                   self.EinsteinForm)
                mu_U[i,j,:,:] = self.params.mobility_U(i, j, energy, energy_U, mu, dens[:,self.Ns-1])

        for j in range(0, self.Nv):
            nu_U[j,:,:] = self.params.momFrequency_U(j, energy, energy_U, nu, dens[:,self.Ns-1])
        ## Ambipolar diffusion coefficient for electrons:
        #diffusivity[:,0] = (diffusivity[:,0] + np.multiply(np.divide(diffusivity[:,1], mu[:,1]), mu[:,0])) / (1 + (Te[:,0] / Tg[:,0]))
        #for i in range(0, self.Nv):
        #    diffusivity_U[0,i,:,:] = np.multiply((1 + np.divide(Te[:,0],Tg[:,0])), (diffusivity_U[0,i,:,:] +
        #        np.multiply(np.divide(diffusivity[:,1], mu[:,1]), mu_U[0,i,:,:]) + np.multiply(mu[:,0],
        #            np.divide((np.multiply(mu[:,1], diffusivity_U[1,i,:,:]) - np.multiply(diffusivity[:,1],
        #                mu_U[1,i,:,:])), np.multiply(mu[:,1], mu[:,1]))))) \
        #            - np.multiply((diffusivity[:,0] + np.multiply(np.divide(diffusivity[:,1], mu[:,1]), mu[:,0])), np.divide((np.multiply(Tg[:,0],
        #                energy_U[0,i,:,:]) - np.multiply(Te[:,0], energy_U[1,i,:,:])), np.multiply(Tg[:,0], Tg[:,0])))

        # solve poisson equation for phi_ne
        ident0 = np.identity(self.Np)
        ident0[0,0] = ident0[-1,-1] = 0.0
        phi_ni = np.linalg.solve(self.LpD, -self.params.alpha*ident0)
        phi_ne = -phi_ni
        
        # form flux Jacobians
        dens_x = self.Dp @ dens
        nT_x   = self.Dp @ nT
        phi_x  = self.Dp @ self.phi

        phi_x_ne = self.Dp @ phi_ne
        phi_x_ni = self.Dp @ phi_ni

        # must have electron flux for use in Jacobian of Joule heating
        fe = np.zeros((self.Np, 1),dtype=np.float64)
        fe[:,0] = -np.multiply(mu[:,0], dens[:,0]) * (-phi_x[:,0]) - np.multiply(diffusivity[:,0], dens_x[:,0])
        fe = fe.reshape((self.Np,1))

        # must have these for joule heating erms
        fspec = np.ndarray((self.Np, self.Ns),dtype=np.float64)
        for i in range(0,self.Ns):
            fspec[:,i] = ( self.params.charge(i) * mu[:,i] * dens[:,i] * (-phi_x[:,0])
                           - np.multiply(diffusivity[:,i], dens_x[:,i]) )


        fT = (5./3.)*(-np.multiply(mu[:,0], nT[:,0]) * (-phi_x[:,0]) - np.multiply(diffusivity[:,0], nT_x[:,0]))
        fT = fT.reshape((self.Np,1))

        # overwrite endpoints in fi (weakly impose BC)
        for ionIdx in self.params.posIonIdx:
            fspec[ 0,ionIdx] = -self.params.ksion * dens[ 0,[ionIdx]] + mu[ 0,ionIdx] * dens[ 0,[ionIdx]] * (-phi_x[ 0])
            fspec[-1,ionIdx] =  self.params.ksion * dens[-1,[ionIdx]] + mu[-1,ionIdx] * dens[-1,[ionIdx]] * (-phi_x[-1])
        
        #fspec[ 0,1] = -self.params.ksion * dens[ 0,[1]] \
        #            + mu[0,1] * dens[ 0,[1]] * (-phi_x[ 0])
        #fspec[-1,1] = +self.params.ksion * dens[-1,[1]] \
        #            + mu[-1,1] * dens[-1,[1]] * (-phi_x[-1])
        # Second ion flux BCs (hard-coded for now)
        #if self.Ns == 5:
        #    fspec[ 0,2] = -self.params.ksion * dens[ 0,[2]] \
        #                + mu[0,2] * dens[ 0,[2]] * (-phi_x[ 0])
        #    fspec[-1,2] = +self.params.ksion * dens[-1,[2]] \
        #                + mu[-1,2] * dens[-1,[2]] * (-phi_x[-1])
        
        # species equations
        fspec_U = np.zeros((self.Ns, self.Ns+1,self.Np, self.Np),dtype=np.float64)
        for i in range(0,self.Ns-1):
            fspec_U[i,i,:,:] = (  self.params.charge(i)
                                * np.multiply(mu[:,[i]], np.multiply(np.identity(self.Np),-phi_x))
                                - np.multiply(diffusivity[:,[i]], self.Dp) )

            fspec_U[i,0,:,:] += self.params.charge(i) \
                * np.multiply(mu[:,[i]],np.multiply(dens[:,[i]],-phi_x_ne))
            for ionIdx in self.params.posIonIdx:
                fspec_U[i,ionIdx,:,:] += self.params.charge(i) \
                        * np.multiply(mu[:,[i]], np.multiply(dens[:,[i]],-phi_x_ni))
            #fspec_U[i,1,:,:] += self.params.charge(i) \
            #    * np.multiply(mu[:,[i]], np.multiply(dens[:,[i]],-phi_x_ni))
            # Add contribution of second ion (hard-coded for now)
            #if self.Ns == 5:
            #    fspec_U[i,2,:,:] += self.params.charge(i) \
            #        * np.multiply(mu[:,[i]], np.multiply(dens[:,[i]],-phi_x_ni))

        for i in range(0,self.Ns-1):
            for j in range(0,self.Nv):
                fspec_U[i,j,:,:] += self.params.charge(i) * np.multiply(mu_U[i,j,:,:], np.multiply(dens[:,[i]],-phi_x))
                fspec_U[i,j,:,:] -= np.multiply(diffusivity_U[i,j,:,:], dens_x[:,[i]])
        
        # energy equations
        #fT = (5./3.)*(-np.multiply(mu[:,0], nT[:,0]) * (-phi_x[:,0]) - np.multiply(diffusivity[:,0], nT_x[:,0]))

        fT_U = np.zeros((self.Ns+1,self.Np, self.Np),dtype=np.float64)
        fT_U[0,:,:] = (5./3.)*(-mu[:,self.params.iele]*np.multiply(nT,-phi_x_ne))
        for ionIdx in self.params.posIonIdx:
            fT_U[ionIdx,:,:] = (5./3.)*(-mu[:,self.params.iele]*np.multiply(nT,-phi_x_ni))
        #fT_U[1,:,:] = (5./3.)*(-mu[:,self.params.iele]*np.multiply(nT,-phi_x_ni))
        # Contribution from second ion (hard-coded for now)
        #if self.Ns == 5:
        #    fT_U[2,:,:] = (5./3.)*(-mu[:,self.params.iele]*np.multiply(nT,-phi_x_ni))
        
        for j in range(0, self.Nv):
            fT_U[j,:,:] += (5./3.) * np.multiply(-mu_U[0,j,:,:], np.multiply(nT,-phi_x))
            fT_U[j,:,:] -= (5./3.) * np.multiply(diffusivity_U[0, j, :, :], nT_x[:,0])


        fT_U[self.Ns,:,:] += (5./3.)*( -np.multiply(mu[:,self.params.iele],np.multiply(np.identity(self.Np),-phi_x))
                                       -np.multiply(diffusivity[:,self.params.iele], self.Dp))

        # overwrite endpoints in fi (weakly impose BC)
        # Assuming singly ionized species only -> phi_x_ni is the same for all positive ions!
        
        #for i in range(0,self.Nv):
        #    fspec_U[1,i,0,:] = 0
        #    fspec_U[1,i,-1,:] = 0
        #    if self.Ns == 5:
        #        fspec_U[2,i,0,:] = 0
        #        fspec_U[2,i,-1,:] = 0

        #fspec_U[1,0,0,:] = mu[0,1] * dens[0,1] * (-phi_x_ne[ 0,:])
        #fspec_U[1,1,0,:] = mu[0,1] * dens[0,1] * (-phi_x_ni[ 0,:])
        #if self.Ns == 5:
        #    fspec_U[1,2,0,:] = mu[0,1] * dens[0,1] * (-phi_x_ni[ 0,:])

        #    fspec_U[2,0,0,:] = mu[0,2] * dens[0,2] * (-phi_x_ne[ 0,:])
        #    fspec_U[2,1,0,:] = mu[0,2] * dens[0,2] * (-phi_x_ni[ 0,:])
        #    fspec_U[2,2,0,:] = mu[0,2] * dens[0,2] * (-phi_x_ni[ 0,:])

        #for i in range(0, self.Nv):
        #    fspec_U[1,i,0,:] += mu_U[1,i,0,:] * dens[0,1] * (-phi_x[0,0])
        #    if self.Ns == 5:
        #        fspec_U[2,i,0,:] += mu_U[2,i,0,:] * dens[0,2] * (-phi_x[0,0])

        #fspec_U[1,1,0,0] += -self.params.ksion + mu[0,1] * (-phi_x[ 0])
        #if self.Ns == 5:
        #    fspec_U[2,2,0,0] += -self.params.ksion + mu[0,2] * (-phi_x[ 0])

        #fspec_U[1,0,-1,:] = mu[-1,1] * dens[-1,1] * (-phi_x_ne[-1,:])
        #fspec_U[1,1,-1,:] = mu[-1,1] * dens[-1,1] * (-phi_x_ni[-1,:])
        #if self.Ns == 5:
        #    fspec_U[1,2,-1,:] = mu[-1,1] * dens[-1,1] * (-phi_x_ni[-1,:])

        #    fspec_U[2,0,-1,:] = mu[-1,2] * dens[-1,2] * (-phi_x_ne[-1,:])
        #    fspec_U[2,1,-1,:] = mu[-1,2] * dens[-1,2] * (-phi_x_ni[-1,:])
        #    fspec_U[2,2,-1,:] = mu[-1,2] * dens[-1,2] * (-phi_x_ni[-1,:])

        #for i in range(0, self.Nv):
        #    fspec_U[1,i,-1,:] += mu_U[1,i,-1,:] * dens[-1,1] * (-phi_x[-1,0])
        #    if self.Ns == 5:
        #        fspec_U[2,i,-1,:] += mu_U[2,i,-1,:] * dens[-1,2] * (-phi_x[-1,0])

        #fspec_U[1,1,-1,-1] += self.params.ksion + mu[-1,1] * (-phi_x[-1])
        #if self.Ns == 5:
        #    fspec_U[2,2,-1,-1] += self.params.ksion + mu[-1,2] * (-phi_x[-1])
        for ionIdx in self.params.posIonIdx:
            for i in range(0,self.Nv):
                fspec_U[ionIdx,i,0,:] = 0.0
                fspec_U[ionIdx,i,-1,:] = 0.0

            fspec_U[ionIdx,0,0,:] = mu[0,ionIdx] * dens[0,ionIdx] * (-phi_x_ne[ 0,:])
            for i in range(len(self.params.posIonIdx)):
                fspec_U[ionIdx, self.params.posIonIdx[i],0,:] = mu[0,ionIdx]*dens[0,ionIdx]*(-phi_x_ni[ 0,:])    
            #fspec_U[ionIdx,ionIdx,0,:] = mu[0,ionIdx] * dens[0,ionIdx] * (-phi_x_ni[ 0,:])

            for i in range(0,self.Nv):
                fspec_U[ionIdx,i,0,:] += mu_U[ionIdx,i,0,:] * dens[0,ionIdx] * (-phi_x[ 0,0])
            
            fspec_U[ionIdx,ionIdx,0,0] += -self.params.ksion + mu[0,ionIdx] * (-phi_x[ 0])

            fspec_U[ionIdx,0,-1,:] = mu[-1,ionIdx] * dens[-1,ionIdx] * (-phi_x_ne[-1,:])

            for i in range(len(self.params.posIonIdx)):
                fspec_U[ionIdx,self.params.posIonIdx[i],-1,:] = mu[-1,ionIdx]*dens[-1,ionIdx]*(-phi_x_ni[-1,:])
            #fspec_U[ionIdx,ionIdx,-1,:] = mu[-1,ionIdx] * dens[-1,ionIdx] * (-phi_x_ni[-1,:])


            for i in range(0,self.Nv):
                fspec_U[ionIdx,i,-1,:] += mu_U[ionIdx,i,-1,:] * dens[-1,ionIdx] * (-phi_x[-1,0])

            fspec_U[ionIdx,ionIdx,-1,-1] += self.params.ksion + mu[-1,ionIdx] * (-phi_x[-1])

        ## BCs for electron flux:
        rstrg_U = np.zeros((2,self.Nv*self.Np))
        if (weak_bc):
            fspec_U_tmp = np.zeros((1,1,1,fspec_U.shape[-1]))
            fspec_U_tmp2 = np.zeros((1,1,1,fspec_U.shape[-1]))
            fspec_U_tmp3 = np.zeros((1,1,1,fspec_U.shape[-1]))
            fspec_U_tmp4 = np.zeros((1,1,1,fspec_U.shape[-1]))
            for ionIdx in self.params.posIonIdx:
                #fspec_U[0,0,0,:] = (- self.params.gam*fspec_U[ 1,0,0,:])
                fspec_U_tmp += (-self.params.gam*fspec_U[ionIdx,0,0,:])
                fspec_U[0,ionIdx,0,:] = (- self.params.gam*fspec_U[ ionIdx,ionIdx,0,:])
                fspec_U_tmp2 += (-self.params.gam*fspec_U[ionIdx,self.Ns-1,0,:])
                #fspec_U[0,self.Ns-1,0,:] = (- self.params.gam*fspec_U[ 1,self.Ns-1,0,:])
                fspec_U[0,0,0,0] -= self.params.ks \
                    * (Te[0,0]**0.5 + 0.5 * Te[0,0]**(-0.5) * Te_ne[0,0]* dens[0,0])
                fspec_U[0,self.Ns,0,0] -= self.params.ks \
                    * (0.5 * Te[0,0]**(-0.5) * Te_nT[0,0] * dens[0,0])

                #fspec_U[0,0,-1,:] = (- self.params.gam*fspec_U[1,0,-1,:])
                fspec_U_tmp3 += (-self.params.gam*fspec_U[ionIdx,0,-1,:]) 
                fspec_U[0,ionIdx,-1,:] = (- self.params.gam*fspec_U[ionIdx,ionIdx,-1,:])
                #fspec_U[0,self.Ns,-1,:] = (- self.params.gam*fspec_U[ 1,self.Ns,-1,:])
                fspec_U_tmp4 += (-self.params.gam*fspec_U[ionIdx,self.Ns,-1,:])
                fspec_U[0,0,-1,-1] += self.params.ks \
                    * (Te[-1,0]**0.5 + 0.5 * Te[-1,0]**(-0.5) * Te_ne[-1,-1]* dens[-1,0])
                fspec_U[0,self.Ns,-1,-1] += self.params.ks \
                    * (0.5 * Te[-1,0]**(-0.5) * Te_nT[-1,-1] * dens[-1,0])

            fspec_U[0,0,0,:] = fspec_U_tmp
            fspec_U[0,self.Ns-1,0,:] = fspec_U_tmp2
            fspec_U[0,0,-1,:] = fspec_U_tmp3
            fspec_U[0,self.Ns,-1,:] = fspec_U_tmp4
        else:
            #rstrg_U[0,0:self.Np] = fspec_U[0,0,0,:] - (-self.params.gam*fspec_U[1,0,0,:])
            #rstrg_U[0,self.Np:2*self.Np] = fspec_U[0,1,0,:] - (-self.params.gam*fspec_U[1,1,0,:])
            #rstrg_U[0,(self.Ns-1)*self.Np:self.Ns*self.Np] = fspec_U[0,self.Ns-1,0,:] - (-self.params.gam*fspec_U[1,self.Ns-1,0,:])
            #rstrg_U[0,self.Ns*self.Np:] = fspec_U[0,self.Ns,0,:] - (-self.params.gam*fspec_U[1,self.Ns,0,:])
            #if self.Ns == 5:
            #    rstrg_U[0,0:self.Np] -= (-self.params.gam*fspec_U[2,0,0,:])
            #    rstrg_U[0,self.Np:2*self.Np] -= (-self.params.gam*fspec_U[2,1,0,:])
            #    rstrg_U[0,2*self.Np:3*self.Np] = fspec_U[0,2,0,:] - (-self.params.gam*fspec_U[1,2,0,:] - self.params.gam*fspec_U[2,2,0,:])
            #    rstrg_U[0,(self.Ns-1)*self.Np:self.Ns*self.Np] -= (-self.params.gam*fspec_U[2,self.Ns-1,0,:])
            #    rstrg_U[0,self.Ns*self.Np:] -= (-self.params.gam*fspec_U[2,self.Ns,0,:])

            #rstrg_U[1,0:self.Np] = fspec_U[0,0,-1,:] - (-self.params.gam*fspec_U[1,0,-1,:])
            #rstrg_U[1,self.Np:2*self.Np] = fspec_U[0,1,-1,:] - (-self.params.gam*fspec_U[1,1,-1,:])
            #rstrg_U[1,(self.Ns-1)*self.Np:self.Ns*self.Np] = fspec_U[0,self.Ns-1,-1,:] - (-self.params.gam*fspec_U[1,self.Ns-1,-1,:])
            #rstrg_U[1,self.Ns*self.Np:] = fspec_U[0,self.Ns,-1,:] - (-self.params.gam*fspec_U[1,self.Ns,-1,:])
            #if self.Ns == 5:
            #    rstrg_U[1,0:self.Np] -= (-self.params.gam*fspec_U[2,0,-1,:])
            #    rstrg_U[1,self.Np:2*self.Np] -= (-self.params.gam*fspec_U[2,1,-1,:])
            #    rstrg_U[1,2*self.Np:3*self.Np] = fspec_U[0,2,-1,:] - (-self.params.gam*fspec_U[1,2,-1,:] - self.params.gam*fspec_U[2,2,-1,:])
            #    rstrg_U[1,(self.Ns-1)*self.Np:self.Ns*self.Np] -= (-self.params.gam*fspec_U[2,self.Ns-1,-1,:])
            #    rstrg_U[1,self.Ns*self.Np:] -= (-self.params.gam*fspec_U[2,self.Ns,-1,:])
            rstrg_U[0,0:self.Np] = fspec_U[0,0,0,:]
            rstrg_U[1,0:self.Np] = fspec_U[0,0,-1,:]
            rstrg_U[0,(self.Ns-1)*self.Np:self.Ns*self.Np] = fspec_U[0,self.Ns-1,0,:]
            rstrg_U[1,(self.Ns-1)*self.Np:self.Ns*self.Np] = fspec_U[0,self.Ns-1,-1,:]
            rstrg_U[0,self.Ns*self.Np:] = fspec_U[0,self.Ns,0,:]
            rstrg_U[1,self.Ns*self.Np:] = fspec_U[0,self.Ns,-1,:]
            for ionIdx in self.params.posIonIdx:
                rstrg_U[0,0:self.Np] -= (-self.params.gam*fspec_U[ionIdx,0,0,:])
                rstrg_U[0,ionIdx*self.Np:(ionIdx+1)*self.Np] = fspec_U[0,ionIdx,0,:]
                #rstrg_U[0,ionIdx*self.Np:(ionIdx+1)*self.Np] = fspec_U[0,ionIdx,0,:] - (-self.params.gam*fspec_U[ionIdx,ionIdx,0,:])
                for i in range(len(self.params.posIonIdx)):
                    rstrg_U[0,ionIdx*self.Np:(ionIdx+1)*self.Np] -= (-self.params.gam*fspec_U[self.params.posIonIdx[i],ionIdx,0,:])
                rstrg_U[0,(self.Ns-1)*self.Np:self.Ns*self.Np] -= (-self.params.gam*fspec_U[ionIdx,self.Ns-1,0,:])
                rstrg_U[0,self.Ns*self.Np:] -= (-self.params.gam*fspec_U[ionIdx,self.Ns,0,:])

                rstrg_U[1,0:self.Np] -= (-self.params.gam*fspec_U[ionIdx,0,-1,:])
                rstrg_U[1,ionIdx*self.Np:(ionIdx+1)*self.Np] = fspec_U[0,ionIdx,-1,:]
                #rstrg_U[1,ionIdx*self.Np:(ionIdx+1)*self.Np] = fspec_U[0,ionIdx,-1,:] - (-self.params.gam*fspec_U[ionIdx,ionIdx,-1,:])
                for i in range(len(self.params.posIonIdx)):
                    rstrg_U[1,ionIdx*self.Np:(ionIdx+1)*self.Np] -= (-self.params.gam*fspec_U[self.params.posIonIdx[i],ionIdx,-1,:])
                rstrg_U[1,(self.Ns-1)*self.Np:self.Ns*self.Np] -= (-self.params.gam*fspec_U[ionIdx,self.Ns-1,-1,:])
                rstrg_U[1,self.Ns*self.Np:] -= (-self.params.gam*fspec_U[ionIdx,self.Ns,-1,:])

            #rstrg_U[0,0:self.Np] = fspec_U[0,0,0,:] - (- self.params.gam*fspec_U[ 1,0,0,:])
            #rstrg_U[0,self.Np:2*self.Np] = fspec_U[0,1,0,:] - (- self.params.gam*fspec_U[ 1,1,0,:])
            #rstrg_U[0,(self.Ns-1)*self.Np:self.Ns*self.Np] = fspec_U[0,self.Ns-1,0,:] - (- self.params.gam*fspec_U[ 1,self.Ns-1,0,:])
            #rstrg_U[0,self.Ns*self.Np:] = fspec_U[0,self.Ns,0,:] - (- self.params.gam*fspec_U[ 1,self.Ns,0,:])

            rstrg_U[0,0] += self.params.ks \
                * (Te[0,0]**0.5 + 0.5 * Te[0,0]**(-0.5) * Te_ne[0,0]* dens[0,0])
            rstrg_U[0,self.Ns*self.Np] += self.params.ks \
                * (0.5 * Te[0,0]**(-0.5) * Te_nT[0,0] * dens[0,0])
    
            #rstrg_U[1,0:self.Np] = fspec_U[0,0,-1,:] - (- self.params.gam*fspec_U[1,0,-1,:])
            #rstrg_U[1,self.Np:2*self.Np] = fspec_U[0,1,-1,:] - (- self.params.gam*fspec_U[1,1,-1,:])
            #rstrg_U[1,(self.Ns-1)*self.Np:self.Ns*self.Np] = fspec_U[0,self.Ns-1,-1,:] - (- self.params.gam*fspec_U[ 1,self.Ns-1,-1,:])
            #rstrg_U[1,self.Ns*self.Np:] = fspec_U[0,self.Ns,-1,:] - (- self.params.gam*fspec_U[ 1,self.Ns,-1,:])
            
            rstrg_U[1,self.Np-1] -= self.params.ks \
                * (Te[-1,0]**0.5 + 0.5 * Te[-1,0]**(-0.5) * Te_ne[-1,-1]* dens[-1,0])
            rstrg_U[1,self.Nv*self.Np-1] -= self.params.ks \
                * (0.5 * Te[-1,0]**(-0.5) * Te_nT[-1,-1] * dens[-1,0])
        
        
        # form Jacobians of derivatives of fluxes at collocation points
        fspec_x_U = np.ndarray((self.Ns, self.Ns+1, self.Np, self.Np),dtype=np.float64)

        for i in range(0,self.Ns):
            for j in range(0,self.Ns+1):
                fspec_x_U[i,j,:,:] = self.Dp @ fspec_U[i,j,:,:]

        fT_x_U = np.ndarray((self.Ns+1, self.Np, self.Np),dtype=np.float64)
        for j in range(0,self.Ns+1):
            fT_x_U[j, :,:] = self.Dp @ fT_U[j,:,:]

        # form source terms at collocation points
        # omega_V returns derivatives of chemical src terms wrt ne, ni, ..., Te
        omega = self.params.rxnSourceTerm(Te, dens)
        omega_V = self.params.rxnSourceTermJac(Te, dens)

        # chain rule to get derivatives wrt ne, ni, ..., nT
        omega_U = np.ndarray(np.shape(omega_V))
        for i in range(0,self.Ns+1):
            omega_U[i,0,:] = omega_V[i,0,:] + omega_V[i,self.Ns,:]*np.diag(Te_ne)
            omega_U[i,self.Ns,:] = omega_V[i,self.Ns,:]*np.diag(Te_nT)

        omega_U[:,1:self.Ns,:] = omega_V[:,1:self.Ns,:]

        # joule heating
        SJ_ne = -self.params.qStar*( np.multiply(fspec_U[0,0,:,:],-phi_x) + np.multiply(fe,-phi_x_ne))
        SJ_ni = np.zeros((SJ_ne.shape[0],SJ_ne.shape[1],len(self.params.posIonIdx)))
        for ionIdx in self.params.posIonIdx:
            SJ_ni[:,:,ionIdx-1] += -self.params.qStar*(np.multiply(fspec_U[0,ionIdx,:,:],-phi_x) + np.multiply(fe,-phi_x_ni))
        #SJ_ni = -self.params.qStar*(np.multiply(fspec_U[0,1,:,:],-phi_x) + np.multiply(fe,-phi_x_ni))
        #if self.Ns == 5:
        #    SJ_ni2 = -self.params.qStar*(np.multiply(fspec_U[0,2,:,:],-phi_x) + np.multiply(fe,-phi_x_ni))
        
        SJ_nb = -self.params.qStar * np.multiply(fspec_U[0,self.Ns-1,:,:],-phi_x)
        SJ_nT = -self.params.qStar * np.multiply(fspec_U[0,self.Ns,:,:],-phi_x)

        # elastic collisions
        SEC_U = np.zeros((self.Ns + 1, self.Np, self.Np), dtype=np.float64)
        for j in range(0, self.Nv):
            SEC_U[j, :, :] = -nu_U[j,:,:] * (nT - np.multiply(dens[:,self.params.iele], Tg)) \
                             -nu[:,] * dens[:, self.params.iele] * energy_U[0,j,:,:] \
                             -nu[:,] * dens[:, self.params.iele] * np.multiply(np.identity(self.Np), np.diag(Tg_U[:,j]))

        #SEC_U[self.Ns+1, :, :] -= 
        SEC_U[        0, :, :] -= np.multiply(np.identity(self.Np), (nu[:,] * energy[:,0])) + np.multiply(np.identity(self.Np), (nu[:,0] * Tg[:,0]))
        SEC_U *= self.elasticCollisionActivationFactor

        ## Radial Diffusion Source Term
        s_dot_RD = np.zeros((self.Np, self.Ns), dtype = np.float64)
        s_dot_RD_U = np.zeros((self.Ns, self.Ns+1, self.Np, self.Np), dtype = np.float64)

        #s_dot_RD[:,:] = self.params.radialDiffSourceTerm(self.xp, dens, diffusivity)
        for i in range(1, self.Ns-1):
            s_dot_RD[:,i] = self.params.radialDiffSourceTerm(i, dens, diffusivity)

        for ionIdx in self.params.posIonIdx:
            s_dot_RD[:,0] += s_dot_RD[:,ionIdx]

        s_dot_RD *= self.radialDiffusionActivationFactor

        #s_dot_RD_U = self.params.radialDiffSourceTermJac(dens, diffusivity, diffusivity_U)
        for i in range(1, self.Ns-1):
            for j in range(0, self.Nv):
                s_dot_RD_U[i,j,:,:] = self.params.radialDiffSourceTermJac(i, j, dens, diffusivity, diffusivity_U)

        for ionIdx in self.params.posIonIdx:
            s_dot_RD_U[0,:,:,:] += s_dot_RD_U[ionIdx,:,:,:]

        s_dot_RD_U *= self.radialDiffusionActivationFactor
        
        # evaluate S---the source term required in the background
        # specie evolution to ensure constant pressure
        fa = np.zeros((self.Np,1),dtype=np.float64)
        fa_U = np.zeros((self.Ns+1,self.Np, self.Np),dtype=np.float64)
        fa = np.copy(fT)
        fa_U = np.copy(fT_U)

        naTg = np.zeros((self.Np,1),dtype=np.float64)
        for i in range(1,self.Ns-1):
            naTg[:,0] = dens[:,i]*Tg[:,0]
            fa[:,0] += (5./3.)*(self.params.charge(i)*np.multiply(mu[:,i],np.multiply(naTg[:,0],(-phi_x[:,0]))) -
                           np.multiply(diffusivity[:,i], (self.Dp @ naTg[:,0] )))

            fa_U[0,:,:] += (5./3.)*(self.params.charge(i)*np.multiply(mu[:,[i]], np.multiply(naTg,-phi_x_ne)))
            #fa_U[1,:,:] += (5./3.)*(self.params.charge(i)*np.multiply(mu[:,[i]], np.multiply(naTg,-phi_x_ni)))
            #if self.Ns == 5:
            #    fa_U[2,:,:] += (5./3.)*(self.params.charge(i)*np.multiply(mu[:,[i]], np.multiply(naTg,-phi_x_ni)))
            for ionIdx in self.params.posIonIdx:
                fa_U[ionIdx,:,:] += (5./3.)*(self.params.charge(i)*np.multiply(mu[:,[i]], np.multiply(naTg,-phi_x_ni)))
            fa_U[i,:,:] += (5./3.)*(self.params.charge(i)*np.multiply(mu[:,[i]], np.multiply(np.diag(Tg[:,0]),-phi_x))
                                    -np.multiply(diffusivity[:,[i]], self.Dp @ np.diag(Tg[:,0])))
            for j in range(0, self.Nv):
                fa_U[j,:,:] += (5./3.)*(self.params.charge(i)
                                        *np.multiply(mu[:,[i]], np.multiply(dens[:,i]*(-phi_x),np.diag(Tg_U[:,j]))) -
                                        np.multiply(diffusivity[:,[i]], (self.Dp @ np.multiply(dens[:,[i]],np.diag(Tg_U[:,j])))))
                fa_U[j,:,:] += (5./3.)*self.params.charge(i)*np.multiply(mu_U[i,j,:,:],np.multiply(naTg[:,0],(-phi_x[:,0])))
                fa_U[j,:,:] -= (5./3.) * np.multiply(self.Dp @ naTg, diffusivity_U[i,j,:,:])
        
        # background thermal conductivity contribution
        fa[:,0] += - self.params.kappaB * (self.Dp @ Tg[:,0])
        for j in range(0,self.Nv):
            fa_U[j,:,:] += - self.params.kappaB * self.Dp @ np.diag(Tg_U[:,j])

        fa_x = self.Dp @ fa

        fa_x_U = np.zeros((self.Ns+1,self.Np, self.Np),dtype=np.float64)
        for j in range(0,self.Nv):
            fa_x_U[j,:,:] = self.Dp @ fa_U[j,:,:]

        sOmEp = np.zeros((self.Np,1),dtype=np.float64)
        sOmEp_U = np.zeros((self.Nv, self.Np, self.Np), dtype=np.float64)
        for i in range(0, self.Ns-1):
            sOmEp[:,0] += omega[:,i]*self.params.dEps[i]
            for j in range(0,self.Nv):
                sOmEp_U[j,:,:] += np.diag(omega_U[i,j,:]*self.params.dEps[i])

        joule = np.zeros((self.Np,1),dtype=np.float64)
        joule_U = np.zeros((self.Nv, self.Np, self.Np), dtype=np.float64)
        for i in range(0, self.Ns-1):
            joule[:,0] += self.params.qStar*self.params.charge(i)*np.multiply(fspec[:,i],(-phi_x[:,0]))
            for j in range(0,self.Nv):
                joule_U[j,:,:] += self.params.qStar*self.params.charge(i)*np.multiply(fspec_U[i,j,:,:],(-phi_x))

            joule_U[0,:,:] += self.params.qStar*self.params.charge(i)*np.multiply(fspec[:,[i]],(-phi_x_ne))
            #joule_U[1,:,:] += self.params.qStar*self.params.charge(i)*np.multiply(fspec[:,[i]],(-phi_x_ni))
            #if self.Ns == 5:
            #    joule_U[2,:,:] += self.params.qStar*self.params.charge(i)*np.multiply(fspec[:,[i]],(-phi_x_ni))
            for ionIdx in self.params.posIonIdx:
                joule_U[ionIdx,:,:] += self.params.qStar*self.params.charge(i)*np.multiply(fspec[:,[i]],(-phi_x_ni))
        
        S  = (sOmEp + fa_x - joule)/Tg/self.params.nAronp0
        S *= self.backgroundSpecieActivationFactor

        S_RD = np.zeros((self.Np,1), dtype=np.float64)
        S_RD_U = np.zeros((self.Nv, self.Np, self.Np), dtype=np.float64)
        SERD_U = np.zeros((self.Nv, self.Np, self.Np), dtype=np.float64)
        for i in range(1, self.Ns-1):
            S_RD[:,0] -= s_dot_RD[:,i] / self.params.nAronp0
        S_RD *= self.radialDiffusionActivationFactor
        for j in range(0, self.Nv):
            for i in range(1, self.Ns-1):
                S_RD_U[j,:,:] -= s_dot_RD_U[i,j,:,:] / self.params.nAronp0
        #S_RD_U *= self.radialDiffusionActivationFactor

        for j in range(0, self.Nv):
            SERD_U[j,:,:] = np.multiply(np.diag(energy[:,0]), s_dot_RD_U[0,j,:,:]) + np.multiply(np.diag(s_dot_RD[:,0]),
                    energy_U[0,j,:,:])

        S_U = np.zeros((self.Nv, self.Np, self.Np), dtype=np.float64)
        S_U = (sOmEp_U + fa_x_U - joule_U)/Tg/self.params.nAronp0
        S_U *= self.backgroundSpecieActivationFactor
        for j in range(0,self.Nv):
            S_U[j,:,:] += np.multiply(np.diag( -(S/Tg)*Tg_U[:,j] ),np.identity(self.Np))
            #S_RD_U[j, :, :] += np.multiply(np.diag( -(S_RD/Tg)*Tg_U[:,j] ), np.identity(self.Np))

        # form the full jacobian
        self.jac = np.zeros((self.Ndof,self.Ndof))

        # spatial part

        # fluxes: involve spatial derivatives, leading to dense matrices

        # 'standard' continuity eqns
        for i in range(0,self.Ns-1):
            for j in range(0,self.Nv):
                self.jac[i*self.Np:(i+1)*self.Np,j*self.Np:(j+1)*self.Np] = dt*(fspec_x_U[i,j,:,:])
                self.jac[i*self.Np:(i+1)*self.Np,j*self.Np:(j+1)*self.Np] -= dt*(s_dot_RD_U[i,j,:,:])

        # electron energy eqn
        for j in range(0,self.Ns+1):
            self.jac[self.Ns*self.Np:(self.Ns+1)*self.Np,j*self.Np:(j+1)*self.Np] = dt*(fT_x_U[j,:,:])

        # chemistry: spatially local, coupling across species and energy
        # use np.einsum to extract diagonal of each Jacobian block for updating
        # NB: This affects the background eqns (erroneously) but it is overwritten later
        for i in range(0,self.Ns+1):
            for j in range(0,self.Ns+1):
                jac_diag  = np.einsum('ii->i', self.jac[i*self.Np:(i+1)*self.Np,j*self.Np:(j+1)*self.Np])
                jac_diag -= dt*omega_U[i,j,:]

        # Joule heating (electron energy eqn)
        self.jac[self.Ns*self.Np:,0:self.Np]         -= dt*(SJ_ne - SEC_U[0, :, :])
        #self.jac[self.Ns*self.Np:,self.Np:2*self.Np] -= dt*(SJ_ni + SEC_U[1, :, :])
        #if self.Ns == 5:
        #    self.jac[self.Ns*self.Np:,2*self.Np:3*self.Np] -= dt*(SJ_ni2 + SEC_U[2, :, :])
        for ionIdx in self.params.posIonIdx:
            self.jac[self.Ns*self.Np:,ionIdx*self.Np:(ionIdx+1)*self.Np] -= dt*(SJ_ni[:,:,ionIdx-1] - SEC_U[ionIdx, :, :])
        self.jac[self.Ns*self.Np:,(self.Ns-1)*self.Np:self.Ns*self.Np] -= dt*SJ_nb
        self.jac[self.Ns*self.Np:,self.Ns*self.Np:] -= dt*(SJ_nT - SEC_U[self.Ns, :, :])

        for j in range(0, self.Nv):
            self.jac[self.Ns*self.Np:,j*self.Np:(j+1)*self.Np] -= dt*SERD_U[j,:,:]

        # overwrite the background (wrt all variables)
        for j in range(0,self.Nv):
            self.jac[(self.Ns-1)*self.Np:self.Ns*self.Np,j*self.Np:(j+1)*self.Np] = -dt*(S_U[j,:,:] + S_RD_U[j,:,:])

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
        elif (self.temporal_scheme=="CN"):
            self.jacobianCN(Uin, time, dt, weak_bc, solve_poisson)
        else:
            print("Time marching scheme not recognized")
            exit(-1)
    
    def jacobianBE(self, Uin, time, dt, weak_bc=False, solve_poisson=False):
        """Evaluates the Jacobian for backward Euler time marching.
        See timeDomainCollocationSolver.jacobian() for further documentaion.
        """
        # Jacobian of spatial contribution to residual
        rstrg_U = self.spatial_jacobian(Uin, time, dt, weak_bc, solve_poisson)

        # Jacobian of unsteady contribution to residual
        #self.jac += np.identity(self.Ndof)
        self.jac += self.I_Ndof

        # boundary condition modifications (for strongly enforced BCs)
        if (not weak_bc):
            self.jac[0,:] = rstrg_U[0,:]
            self.jac[self.Np-1,:] = rstrg_U[1,:]

        for i in range(len(self.params.posIonIdx)+1,self.Ns-1):
            self.jac[i*self.Np,:] = np.zeros((1,self.Nv*self.Np))
            self.jac[i*self.Np,i*self.Np] = 1.0

            self.jac[(i+1)*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))
            self.jac[(i+1)*self.Np-1,(i+1)*self.Np-1] = 1.0

        # Dirichlet on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            self.jac[(self.Ns-1)*self.Np,:] = np.zeros((1,self.Nv*self.Np))

            for i in range(1,self.Ns-1):
                self.jac[(self.Ns-1)*self.Np,i*self.Np] = 1.0 / self.params.nAronp0

            self.jac[(self.Ns-1)*self.Np,(self.Ns-1)*self.Np] = 1.0
            self.jac[(self.Ns-1)*self.Np,self.Ns*self.Np] = 1.0 / self.params.Tg0 / self.params.nAronp0

            self.jac[self.Ns*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))

            for i in range(1,self.Ns-1):
                self.jac[self.Ns*self.Np-1,(i+1)*self.Np-1] = 1.0 / self.params.nAronp0

            self.jac[self.Ns*self.Np-1,self.Ns*self.Np-1] = 1.0
            self.jac[self.Ns*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0 / self.params.Tg0 / self.params.nAronp0


        # Dirichlet condition on electron energy
        self.jac[self.Ns*self.Np,:] = np.zeros((1,self.Nv*self.Np))
        self.jac[self.Ns*self.Np,self.Ns*self.Np] = 1.0
        self.jac[self.Ns*self.Np,0] = -self.params.EeBC

        self.jac[(self.Ns+1)*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))
        self.jac[(self.Ns+1)*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0
        self.jac[(self.Ns+1)*self.Np-1,self.Np-1] = -self.params.EeBC


    def jacobianCN(self, Uin, time, dt, weak_bc=False, solve_poisson=False):
        """Evaluates the Jacobian for Crank-Nicolson time marching.
        See timeDomainCollocationSolver.jacobian() for further documentaion.
        """
        # Jacobian of spatial contribution to residual
        rstrg_U = self.spatial_jacobian(Uin, time, dt, weak_bc, solve_poisson)
        self.jac *= 0.5

        # Jacobian of unsteady contribution to residual
        #self.jac += np.identity(self.Ndof)
        self.jac += self.I_Ndof

        # boundary condition modifications (for strongly enforced BCs)
        if (not weak_bc):
            self.jac[0,:] = rstrg_U[0,:]
            self.jac[self.Np-1,:] = rstrg_U[1,:]

        for i in range(len(self.params.posIonIdx)+1,self.Ns-1):
            self.jac[i*self.Np,:] = np.zeros((1,self.Nv*self.Np))
            self.jac[i*self.Np,i*self.Np] = 1.0

            self.jac[(i+1)*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))
            self.jac[(i+1)*self.Np-1,(i+1)*self.Np-1] = 1.0

        # Dirichlet on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            self.jac[(self.Ns-1)*self.Np,:] = np.zeros((1,self.Nv*self.Np))

            for i in range(1,self.Ns-1):
                self.jac[(self.Ns-1)*self.Np,i*self.Np] = 1.0 / self.params.nAronp0

            self.jac[(self.Ns-1)*self.Np,(self.Ns-1)*self.Np] = 1.0
            self.jac[(self.Ns-1)*self.Np,self.Ns*self.Np] = 1.0 / self.params.Tg0 / self.params.nAronp0

            self.jac[self.Ns*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))

            for i in range(1,self.Ns-1):
                self.jac[self.Ns*self.Np-1,(i+1)*self.Np-1] = 1.0 / self.params.nAronp0

            self.jac[self.Ns*self.Np-1,self.Ns*self.Np-1] = 1.0
            self.jac[self.Ns*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0 / self.params.Tg0 / self.params.nAronp0

        self.jac[self.Ns*self.Np,:] = np.zeros((1,self.Nv*self.Np))
        self.jac[self.Ns*self.Np,self.Ns*self.Np] = 1.0
        self.jac[self.Ns*self.Np,0] = -self.params.EeBC

        self.jac[(self.Ns+1)*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))
        self.jac[(self.Ns+1)*self.Np-1,(self.Ns+1)*self.Np-1] = 1.0
        self.jac[(self.Ns+1)*self.Np-1,self.Np-1] = -self.params.EeBC

    def jacobianLCN(self, Uin, time, dt, weak_bc=False):
        """Evaluates the Jacobian for Crank-Nicolson time marching.
        See timeDomainCollocationSolver.jacobian() for further documentaion.
        """
        # Jacobian of spatial contribution to residual
        self.spatial_jacobian(Uin, time, dt, weak_bc)
        self.jac *= 0.5

        # Jacobian of unsteady contribution to residual
        #self.jac += np.identity(self.Ndof)
        self.jac += self.I_Ndof

        # boundary condition modifications (for strongly enforced BCs)
        if (not weak_bc):
            print("Error: Only weak electron flux BCs supported for linearized CN.")
            exit(-1)

        #if (self.Ns>2):
        #    self.jac[2*self.Np,:] = np.zeros((1,self.Nv*self.Np))
        #    self.jac[2*self.Np,2*self.Np] = 1.0

        #    self.jac[3*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))
        #    self.jac[3*self.Np-1,3*self.Np-1] = 1.0

        if (self.Ns > 2):
            for i in range(len(self.params.posIonIdx)+1,self.Ns-1):
                self.jac[i*self.Np,:] = np.zeros((1,self.Nv*self.Np))
                self.jac[i*self.Np,i*self.Np] = 1.0
                self.jac[(i+1)*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))
                self.jac[(i+1)*self.Np-1,(i+1)*self.Np-1] = 1.0


        self.jac[self.Ns*self.Np,:] = np.zeros((1,self.Nv*self.Np))
        self.jac[self.Ns*self.Np,self.Ns*self.Np] = 1.0
        self.jac[self.Ns*self.Np,0] = -self.params.EeBC

        self.jac[(self.Ns+1)*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))
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

        if (self.temporal_scheme=="BE"):
            #self.jac0 = -np.identity(self.Ndof)
            self.jac0 = -self.I_Ndof

        elif (self.temporal_scheme=="CN"):
            self.spatial_jacobian(self.U1, time-dt, dt, weak_bc, solve_poisson=True)
            self.jac *= 0.5

            self.jac0 = np.copy(self.jac)

            #self.jac0 -= np.identity(self.Ndof)
            self.jac0 -= self.I_Ndof
        else:
            print("Time marching scheme not recognized")
            exit(-1)

        # boundary condition modifications (for strongly enforced BCs)
        # NB: For BCs that are strongly enforced, corresponding
        # residual has no dependence on previous state
        if (not weak_bc):
            self.jac0[0        ,:] = np.zeros((1,self.Nv*self.Np))
            self.jac0[self.Np-1,:] = np.zeros((1,self.Nv*self.Np))

        for i in range(len(self.params.posIonIdx)+1,self.Ns-1):
            self.jac0[i*self.Np,:] = np.zeros((1,self.Nv*self.Np))
            self.jac0[(i+1)*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))


        # Dirichlet on heavy species temperature
        if (self.backgroundSpecieActivationFactor > 0):
            self.jac0[(self.Ns-1)*self.Np,:] = np.zeros((1,self.Nv*self.Np))
            self.jac0[self.Ns*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))

        # Dirichlet on electron temperature
        self.jac0[self.Ns*self.Np  ,:] = np.zeros((1,self.Nv*self.Np))
        self.jac0[(self.Ns+1)*self.Np-1,:] = np.zeros((1,self.Nv*self.Np))


    def jacobianFD(self, Uin, time, dt):
        """Evaluates the Jacobian at Uin, but using a finite difference
        approximation.  Useful for testing, but very slow.

        Inputs:
          Uin  : Current state
          time : Current time
          dt   : Time step

        Outputs: None (sets self.jac)
        """
        # save residual at Uin
        r0 = self.residual(Uin, time, dt)

        # perturb each component of Uin to form finite differenc approx
        for k in range(0,Uin.shape[0]):
            dU = np.sqrt(np.finfo(np.float64).eps)*np.absolute(Uin[k])
            Up = np.copy(Uin)

            if (np.absolute(dU) < np.finfo(np.float64).eps):
                dU = np.finfo(np.float64).eps

            Up[k] += dU

            rp = self.residual(Up, time, dt)
            self.jac[:,k] = (rp[:,0] - r0[:,0])/dU


    def step(self, time, dt, iter_max=100,
             rtol=1e-6, atol=1e-12, verbose=False, weak_bc=False):
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
        r = self.residual(self.U2, time, dt, weak_bc)
        #self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True) # <- Added
        #jac_inv = np.linalg.inv(self.jac) # <- Added

        normr = normr0 = np.linalg.norm(r)
        count = 0
        converged = ((normr/normr0 < rtol) or (normr < atol))
        if (verbose):
            print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                count, normr, normr/normr0))
        while( not converged and (count < iter_max) ):
            #self.jacobianFD(self.U2, time, dt)
            #if count % 2 == 0:
            #    self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True) # <- Commented out
            #    jac_inv = np.linalg.inv(self.jac)
            self.jacobian(self.U2, time, dt, weak_bc, solve_poisson=True) # <- Commented out
            try:
                dU = np.linalg.solve(self.jac, -r) # <- Commented out
                #dU = np.dot(jac_inv, -r) # <- Added
                self.U2 += dU
                
                # zero the last mode
                #self.filter()

            except:
                # if exception encountered, save state and die
                np.save("residual.npy", r)
                np.save("jacobian.npy", self.jac)
                np.save("exception_U2.npy", self.U2)
                np.save("exception_U1.npy", self.U1)
                np.save("exception_U0.npy", self.U0)
                print("Solve failed!", flush=True)
                exit(-1)
            r = self.residual(self.U2, time, dt, weak_bc)
            normr = np.linalg.norm(r)
            count += 1
            if (verbose):
                print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                    count, normr, normr/normr0))
            converged = ((normr/normr0 < rtol) or (normr < atol))
        if (not converged):
            # if non-convergence encountered, save state and die
            print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                count, normr, normr/normr0))
            np.save("nonconverged_U2.npy", self.U2)
            np.save("nonconverged_U1.npy", self.U1)
            np.save("nonconverged_U0.npy", self.U0)
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
        r = self.residualLCN(self.U1, time, dt, weak_bc)
        self.jacobianLCN(self.U1, time, dt, weak_bc)

        dU = np.linalg.solve(self.jac, -r)
        self.U2 += dU

    def stepSensitivity(self, time, dt, verbose=False, weak_bc=False):
        """Advance the sensitivity matrix

        Inputs
          time       : Current time
          dt         : Time step
          verbose    : If true, print nonlinear solve info

        Outputs: None (self.A1 is set to sensitivity at the end of the time step)
        """
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
        self.A1 = np.linalg.solve(self.jac, self.rhsSens)

        if (verbose):
            print("# Advancing sensitivity system.")


    def solve(self, time0, dt, Nstep, savedata=None, verbose=False,
              rtol=1e-6, computeSensitivity=False, weak_bc=False):

        if(savedata!=None):
            Usave=np.ndarray((Nstep+1,self.U2.shape[0]),dtype=np.float64)
            TotalCurrentSave=np.ndarray((Nstep+1,self.totalCurrent.shape[0]),dtype=np.float64)
            IonCurrentSave=np.ndarray((Nstep+1,self.ionCurrent.shape[0],self.ionCurrent.shape[1]),dtype=np.float64)
            ElectronCurrentSave=np.ndarray((Nstep+1,self.electronCurrent.shape[0]),dtype=np.float64)
            Usave[0,:] = self.U2[:,0]
            TotalCurrentSave[0,:] = self.totalCurrent[:,0]
            for ionIdx in self.params.posIonIdx:
                IonCurrentSave[0,:,ionIdx-1] = self.ionCurrent[:,ionIdx-1]
            ElectronCurrentSave[0,:] = self.electronCurrent[:,0]

        if len(self.params.posIonIdx) > 1:
            print('#')
            print('# {0:10s} {1:12s} {2:12s} {3:12s} {4:12s} {5:12s} {6:12s} {7:12s} {8:12s} {9:12s} {10:12s}'.format(
                "Time", "min ne", "max ne", "min ni", "max ni", "min ni2", "max ni2", "min Te", "max Te", "min nb", "max nb"))
            print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e} {7:.6e} {8:.6e} {9:.6e} {10:.6e}".format(
                time0, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Np:2*self.Np].min(), self.U2[self.Np:2*self.Np].max(),
                self.U2[2*self.Np:3*self.Np].min(), self.U2[2*self.Np:3*self.Np].max(),
                self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()), flush=True)
        else:
            print("#")
            print("# {0:10s} {1:12s} {2:12s} {3:12s} {4:12s} {5:12s} {6:12s}".format(
                "Time", "min ne", "max ne", "min Te", "max Te", "min nb", "max nb"))
            print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e}".format(
                time0, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()))

        # assume initial condition has been set in U1!
        time = time0+dt
        self.step(time, dt, verbose=verbose, rtol=rtol, weak_bc=weak_bc)
        if len(self.params.posIonIdx) > 1:
            print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e} {7:.6e} {8:.6e} {9:.6e} {10:.6e}".format(
                time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Np:2*self.Np].min(), self.U2[self.Np:2*self.Np].max(),
                self.U2[2*self.Np:3*self.Np].min(), self.U2[2*self.Np:3*self.Np].max(),
                self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()), flush=True)
        else:
            print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e}".format(
                time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()))

        if(computeSensitivity):
            self.stepSensitivity(time, dt, verbose=verbose, weak_bc=weak_bc)


        if(savedata!=None):
            Usave[1,:] = self.U2[:,0]
            TotalCurrentSave[1,:] = self.totalCurrent[:,0]
            for ionIdx in self.params.posIonIdx:
                IonCurrentSave[1,:,ionIdx-1] = self.ionCurrent[:,ionIdx-1]
            ElectronCurrentSave[1,:] = self.electronCurrent[:,0]

        for istep in range(1, Nstep):
            # prepare for next step
            self.U0 = np.copy(self.U1)
            self.U1 = np.copy(self.U2)
            time += dt

            if (computeSensitivity):
                self.A0 = np.copy(self.A1)

            # advance
            self.step(time, dt, verbose=verbose, rtol=rtol, weak_bc=weak_bc)
            #self.filter()
            if len(self.params.posIonIdx) > 1:
                print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e} {7:.6e} {8:.6e} {9:.6e} {10:.6e}".format(
                time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Np:2*self.Np].min(), self.U2[self.Np:2*self.Np].max(),
                self.U2[2*self.Np:3*self.Np].min(), self.U2[2*self.Np:3*self.Np].max(),
                self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()), flush=True)
            else:
                print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e} {5:.6e} {6:.6e}".format(
                    time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                    self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max(),
                    self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].min(),
                    self.U2[(self.Ns-1)*self.Np:self.Ns*self.Np].max()), flush=True)


            if(savedata!=None):
                Usave[istep+1,:] = self.U2[:,0]
                TotalCurrentSave[istep+1,:] = self.totalCurrent[:,0]
                for ionIdx in self.params.posIonIdx:
                    IonCurrentSave[istep+1,:,ionIdx-1] = self.ionCurrent[:,ionIdx-1]
                ElectronCurrentSave[istep+1,:] = self.electronCurrent[:,0]
               # np.save('stepSave.npy', Usave)

            if(computeSensitivity):
                self.stepSensitivity(time, dt, verbose=verbose, weak_bc=weak_bc)

        if(savedata!=None):
            np.save(savedata,Usave)
            np.save("TotalCurrent_" + savedata, TotalCurrentSave)
            np.save("IonCurrent_" + savedata, IonCurrentSave)
            np.save("ElectronCurrent_" + savedata, ElectronCurrentSave)


    def solveLCN(self, time0, dt, Nstep, savedata=None, verbose=False,
                 computeSensitivity=False, weak_bc=False):

        if(savedata!=None):
            Usave=np.ndarray((Nstep+1,self.U2.shape[0]),dtype=np.float64)
            TotalCurrentSave=np.ndarray((Nstep+1,self.totalCurrent.shape[0]),dtype=np.float64)
            IonCurrentSave=np.ndarray((Nstep+1,self.ionCurrent.shape[0]),dtype=np.float64)
            ElectronCurrentSave=np.ndarray((Nstep+1,self.electronCurrent.shape[0]),dtype=np.float64)
            Usave[0,:] = self.U2[:,0]
            TotalCurrentSave[0,:] = self.totalCurrent[:,0]
            IonCurrentSave[0,:] = self.ionCurrent[:,0]
            ElectronCurrentSave[0,:] = self.electronCurrent[:,0]

        print("#")
        print("# {0:10s} {1:12s} {2:12s} {3:12s} {4:12s}".format(
            "Time", "min ne", "max ne", "min Te", "max Te"))
        print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
            time0, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max()))

        # assume initial condition has been set in U1!
        time = time0+dt
        self.stepLCN(time, dt, verbose=verbose, weak_bc=weak_bc)
        print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
            time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max()))

        #if(computeSensitivity):
        #    self.stepSensitivity(time, dt, verbose=verbose, weak_bc=weak_bc)


        if(savedata!=None):
            Usave[1,:] = self.U2[:,0]
            TotalCurrentSave[1,:] = self.totalCurrent[:,0]
            IonCurrentSave[1,:] = self.ionCurrent[:,0]
            ElectronCurrentSave[1,:] = self.electronCurrent[:,0]

        for istep in range(1, Nstep):
            # prepare for next step
            self.U0 = np.copy(self.U1)
            self.U1 = np.copy(self.U2)
            time += dt

            #if (computeSensitivity):
            #    self.A0 = np.copy(self.A1)

            # advance
            self.stepLCN(time, dt, verbose=verbose, weak_bc=weak_bc)
            print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
                time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[self.Ns*self.Np:].min(), self.U2[self.Ns*self.Np:].max()), flush=True)

            if(savedata!=None):
                Usave[istep+1,:] = self.U2[:,0]
                TotalCurrentSave[istep+1,:] = self.totalCurrent[:,0]
                IonCurrentSave[istep+1,:] = self.ionCurrent[:,0]
                ElectronCurrentSave[istep+1,:] = self.electronCurrent[:,0]

            #if(computeSensitivity):
            #    self.stepSensitivity(time, dt, verbose=verbose, weak_bc=weak_bc)

        if(savedata!=None):
            np.save(savedata,Usave)
            np.save("TotalCurrent_" + savedata, TotalCurrentSave)
            np.save("IonCurrent_" + savedata, IonCurrentSave)
            np.save("ElectronCurrent_" + savedata, ElectronCurrentSave)


    def plot(self, col, create=True):
        import matplotlib.pyplot as plt
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
                        action='store_true', help='Enforce electron flux BC weakly')
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
                        action='store_true', help="Activate Einstein's form for diffusion coefficient.")
    parser.add_argument('--radialDiffusion', default = False, 
                        action='store_true', help="Activate the radial diffusion loss source term")
    parser.add_argument('--iSample', metavar='iSample', default=0,
                        type=int, help='Sample index, if BOLSIG chemistry is used.')
    parser.add_argument('--gam', metavar='gam', default=0.01, type=float, help='Secondary Electron Emission Coefficient')
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
        print("#   Imposing electron flux BC weakly.")
        print("# ***** WARNING: This is an experimental feature that may not work *****")
        print("# *****          and is not fully supported.  Please beware.       *****")

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
    elif(args.scenario==7):
        print('#   Running scenario = 7 (6 species, 23 rxn, 250mTorr, 100V, Sampling)')
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
        print("#   The Einstein's form for diffusion coefficient is used.")
        EinsteinForm = True
    else:
        print("#   The Einstein's form for diffusion coefficient is not used.")
        EinsteinForm = False

    radialDiffusionActivationFactor = 1.0
    if(args.radialDiffusion==True):
        print('#   Radial Diffusion Losses activated.')
        radialDiffusionActivationFactor = 1.0
    else:
        radialDiffusionActivationFactor = 0.0

    if(args.savedata!=None):
        print("#")
        print("#   Saving every time step to {0:s}".format(args.savedata))
    else:
        print("#")
        print("#   Not saving every time step (use --savedata for this option).")

    print("#")

    # Instantiate solver class
    tds = timeDomainCollocationSolver(Ns, 1, args.Np, elasticCollisionActivationFactor,
                                      backgroundSpecieActivationFactor, EinsteinForm,
                                      radialDiffusionActivationFactor,
                                      gam=args.gam, V0 = args.V0, VDC = args.VDC,
                                      scenario=args.scenario, scheme=args.tscheme,
                                      iSample = args.iSample)

    # Default IC (overwritten below if we are restarting)
    #tds.U1[0:tds.Ns*tds.Np] = 1e-4
    ne_0 = 1.0e-3                               # Set intial density here for easier initialization
    tds.U1[0:(tds.Ns-1)*tds.Np] = ne_0           # Electron inital density
    for ionIdx in tds.params.posIonIdx:          # Ion species initial density (the sum of all ion species should equal electrons)
        tds.U1[ionIdx*tds.Np:(ionIdx+1)*tds.Np] = ne_0/len(tds.params.posIonIdx)

    tds.U1[(tds.Ns-1)*tds.Np:tds.Ns*tds.Np] = 1.0  # background specie
    tds.U1[tds.Ns*tds.Np:] = tds.params.EeBC*tds.U1[0:tds.Np] # electron energy

    # If restart file provided, read it.
    # NOTE: currently we do a lazy restart in that only the final
    # state is saved, so we have to restart with a backward Euler step.
    if (args.restart!=None):
        tds.U1 = np.load(args.restart)

    # Initialize rest of state
    tds.U0 = np.copy(tds.U1)
    tds.U2 = np.copy(tds.U1)

    # Run for desired number of time steps
    if (args.tscheme=="LCN"):
        print("# ***** WARNING: Linearized Crank-Nicolson time marching is   *****")
        print("# *****          an experimental feature that may not work    *****")
        print("# *****          and is not fully supported.  Please beware.  *****")

        tds.solveLCN(args.t0, args.dt, args.Nt,
                     args.savedata, args.verbose, weak_bc=args.weakbc)
    else:
        tds.solve(args.t0, args.dt, args.Nt,
                  args.savedata, args.verbose, args.rtol, weak_bc=args.weakbc)

    # Save the result
    np.save(args.outfile, tds.U2)
    np.save("Current_" + args.outfile, tds.totalCurrent)

    if(args.plot):
        tds.plot('b-')
        plt.show()
