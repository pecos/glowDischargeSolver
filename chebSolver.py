import numpy as np
import numpy.polynomial.chebyshev as cheb
import matplotlib.pyplot as plt

from Liu2014Properties import setLiu2014Properties
from psaapProperties import setPsaapProperties

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
        self.Z[0] = -1 # electrons are always 0
        self.Z[1] =  1 # ions are always 1

        # mobility
        self.mu = np.zeros(Ns)

        # diffusivity
        self.D = np.zeros(Ns)

        # reaction rate data

        # Modified Arrhenius rxn rate coefficients for now
        # kf(T) = A*(T**B)*exp(-C/T)
        self.A = np.zeros(Nr) #self.Ck = 272.0
        self.B = np.zeros(Nr)
        self.C = np.zeros(Nr) #18.687*(3./2.);

        # energy gain/loss in electrons
        self.dH = np.zeros(Nr) #15.7

        # stoichiometric coefficients (Ns+1 b/c we store coefficient
        # for the background gas... it is only used for
        # non-dimensionalization purposes)
        self.beta = np.zeros((Ns+1,Nr),dtype=np.int) # products
        self.alfa = np.zeros((Ns+1,Nr),dtype=np.int) # reactants

        # this represents a single rxn: Ar+e -> Ar+ + e + e
        self.beta[0,0] = 2
        self.beta[1,0] = 1
        self.beta[2,0] = 0

        self.alfa[0,0] = 1
        self.alfa[1,0] = 0
        self.alfa[2,0] = 1


        # other non-dimensional parameters
        self.qStar = 100.0
        self.alpha = 2.33e3

        # boundary condition parameters
        self.gam = 0.01
        self.ks = 6.89e-1
        self.ksion = 0.0

    def charge(self,i):
        return self.Z[i]

    def mobility(self,i):
        return self.mu[i]

    def diffusivity(self,i):
        return self.D[i]

    def rxnSourceTerm(self, energy, density):
        G = self.progressRate(energy,density)

        omega = np.zeros((energy.shape[0], self.Ns+1),dtype=np.float)
        for i in range(0,self.Ns):
            for j in range(0,self.Nr):
                omega[:,i] += (self.beta[i,j] - self.alfa[i,j])*G[:,j]

        for j in range(0,self.Nr):
            omega[:,self.Ns] += self.dH[j]*G[:,j]

        return omega

    def progressRate(self, energy, density):
        G = np.zeros((energy.shape[0],self.Nr))
        for i in range(0,self.Nr):
            kf = self.rxnRateCoefficient(energy, i)
            G[:,i] = kf[:,0]
            for j in range(0,self.Ns):
                G[:,i] *= density[:,j]**self.alfa[j,i]

        return G

    def rxnRateCoefficient(self, energy, i):
        """Returns ionization reaction rate constant"""
        a  = self.A[i]
        b  = self.B[i]
        Ea = self.C[i]
        return a * (energy**b) * np.exp(-Ea/energy)

    def rxnRateCoefficientJac(self, energy, i):
        """Returns derivative of ionization reaction rate constant wrt
        energy
        """
        a  = self.A[i]
        b  = self.B[i]
        Ea = self.C[i]
        return a * (energy**(b-1)) * np.exp(-Ea/energy) * (b + Ea/energy)


    def print(self):
        """Print parameters to the screen"""
        print("# The non-dimensional transport and chemstry properties are")
        print("#   De    = {0:.6e}".format(self.D[0]))
        print("#   Di    = {0:.6e}".format(self.D[1]))
        print("#   mue   = {0:.6e}".format(self.mu[0]))
        print("#   mui   = {0:.6e}".format(self.mu[1]))
        print("#   A[0]  = {0:.6e}".format(self.A[0]))
        print("#   B[0]  = {0:.6e}".format(self.B[0]))
        print("#   C[0]  = {0:.6e}".format(self.C[0]))
        print("#   dH[0] = {0:.6e}".format(self.dH[0]))
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

    def __init__(self, Ns, NT, Np, gam=0.01):
        """Initializes storage and operaters required for solve."""

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
        self.params = modelClosures(self.Ns, 1)
        #setLiu2014Properties(gam, self.params)
        setPsaapProperties(gam, self.params)

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


    def solve_poisson(self, ne,ni,time):
        """Solve Gauss' law for the electric potential.

        Inputs:
          ne   : Values of electron density at xp
          ni   : Values of ion density at xp
          time : Current time

        Outputs: None (sets self.phi to computed potential)
        """
        r = -self.params.alpha*(ni-ne)
        r[0] = 0.0
        r[-1] = np.sin(2*np.pi*time)
        self.phi = np.linalg.solve(self.LpD, r)

    def residual(self, Uin, time, dt, first_step=False):
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
        iele = [0]
        iion = [1]

        # pull off state for convenience
        dens = np.ndarray((self.Np, self.Ns),dtype=np.float)
        for i in range(0,self.Ns):
            dens[:,i] = Uin[i*self.Np:(i+1)*self.Np,0]

        nT = Uin[self.Ns*self.Np:] # assumes just 1 temperature!
        Te = nT/dens[:,iele]

        # solve poisson equation for phi
        # now have self.phi
        self.solve_poisson(dens[:,iele],dens[:,iion],time)

        # form fluxes at grid points
        dens_x = self.Dp @ dens
        nT_x   = self.Dp @ nT
        phi_x  = self.Dp @ self.phi

        fspec = np.ndarray((self.Np, self.Ns),dtype=np.float)
        for i in range(0,self.Ns):
            fspec[:,i] = (   self.params.charge(i)*self.params.mobility(i)*dens[:,i]*(-phi_x[:,0])
                           - self.params.diffusivity(i)*dens_x[:,i] )

        fT = (5./3.)*(-self.params.mobility(0)*nT*(-phi_x) - self.params.diffusivity(0)*nT_x)

        # overwrite endpoints in fi (weakly impose BC)
        fspec[ 0,1] = -self.params.ksion*dens[ 0,iion] + self.params.mobility(1)*dens[ 0,iion]*(-phi_x[ 0])
        fspec[-1,1] =  self.params.ksion*dens[-1,iion] + self.params.mobility(1)*dens[-1,iion]*(-phi_x[-1])

        # form derivatives of fluxes at collocation points
        fspec_x = self.Dp @ fspec
        fT_x = self.Dp @ fT

        # form source terms at collocation points
        omega = self.params.rxnSourceTerm(Te, dens)
        SJ = -self.params.qStar*fspec[:,iele]*(-phi_x)

        # form full residual
        res = np.zeros((3*self.Np,1))

        # spatial part
        for i in range(0,self.Ns):
            res[i*self.Np:(i+1)*self.Np,0] = dt*(fspec_x[:,i] - omega[:,i])

        res[self.Ns*self.Np:]        = dt*(fT_x + omega[:,[self.Ns]] - SJ)


        # time derivative part (backward Euler)
        res += Uin - self.U1

        # boundary conditions (strongly enforced)

        # electron flux
        res[0]           = fspec[ 0,iele]  - (-self.params.ks*dens[ 0,iele] - self.params.gam*fspec[ 0,iion])
        res[self.Np-1]   = fspec[-1,iele]  - ( self.params.ks*dens[-1,iele] - self.params.gam*fspec[-1,iion])

        # electron temperature
        res[self.Ns*self.Np  ] = (nT[ 0] - 0.75*dens[0,iele])
        res[(self.Ns+1)*self.Np-1] = (nT[-1] - 0.75*dens[-1,iele])

        return res

    def jacobian(self, Uin, time, dt, first_step=False):
        """Evaluates the residual.

        Inputs:
          Uin  : Current state
          time : Current time
          dt   : Time step

        Outputs: None (sets self.jac)

        Notes:
          This function currently assumes that Ns=2 and NT=1
        """
        # pull off state
        ne = Uin[0:self.Np]
        ni = Uin[self.Np:2*self.Np]
        nT = Uin[2*self.Np:]

        Te = nT/ne
        Te_ne = -np.multiply(Te/ne,np.identity(self.Np))
        Te_nT = np.multiply(np.identity(self.Np),1./ne)

        # solve poisson equation for phi
        # now have self.phi
        ident0 = np.identity(self.Np)
        ident0[0,0] = ident0[-1,-1] = 0.0
        phi_ni = np.linalg.solve(self.LpD, -self.params.alpha*ident0)
        phi_ne = -phi_ni

        # form fluxes at grid points
        ne_x  = self.Dp @ ne
        ni_x  = self.Dp @ ni
        nT_x  = self.Dp @ nT
        phi_x = self.Dp @ self.phi
        phi_x_ne = self.Dp @ phi_ne
        phi_x_ni = self.Dp @ phi_ni

        fe = -self.params.mobility(0)*ne*(-phi_x) - self.params.diffusivity(0)*ne_x
        fe_ne = ( -self.params.mobility(0)*(np.multiply(np.identity(self.Np),-phi_x) + np.multiply(ne,-phi_x_ne))
                  -self.params.diffusivity(0)*self.Dp )
        fe_ni =   -self.params.mobility(0)*np.multiply(ne,-phi_x_ni)

        fi_ne = self.params.mobility(1)*np.multiply(ni,-phi_x_ne)
        fi_ni = ( +self.params.mobility(1)*(np.multiply(ni,-phi_x_ni) + np.multiply(np.identity(self.Np), -phi_x))
                  -self.params.diffusivity(1)*self.Dp )

        fT_ne = (5./3.)*(-self.params.mobility(0)*np.multiply(nT,(-phi_x_ne)))
        fT_ni = (5./3.)*(-self.params.mobility(0)*np.multiply(nT,(-phi_x_ni)))
        fT_Te = (5./3.)*(-self.params.mobility(0)*np.multiply(np.identity(self.Np),(-phi_x))
                         -self.params.diffusivity(0)*self.Dp)


        fi_ne[0,:] = self.params.mobility(1)*ni[ 0]*(-phi_x_ne[ 0,:])
        fi_ni[0,:] = self.params.mobility(1)*ni[ 0]*(-phi_x_ni[ 0,:])
        fi_ni[0,0] += -self.params.ksion + self.params.mobility(1)*(-phi_x[ 0])

        fi_ne[-1,:] = self.params.mobility(1)*ni[-1]*(-phi_x_ne[-1,:])
        fi_ni[-1,:] = self.params.mobility(1)*ni[-1]*(-phi_x_ni[-1,:])
        fi_ni[-1,-1] += self.params.ksion + self.params.mobility(1)*(-phi_x[-1])

        # form derivatives of fluxes at collocation points
        fe_x_ne = self.Dp @ fe_ne
        fe_x_ni = self.Dp @ fe_ni

        fi_x_ne = self.Dp @ fi_ne
        fi_x_ni = self.Dp @ fi_ni

        fT_x_ne = self.Dp @ fT_ne
        fT_x_ni = self.Dp @ fT_ni
        fT_x_Te = self.Dp @ fT_Te

        # form source terms at collocation points
        ki = self.params.rxnRateCoefficient(Te,0)
        ki_ne = np.diag(self.params.rxnRateCoefficientJac(Te,0)[:,0]) @ Te_ne
        ki_nT = np.diag(self.params.rxnRateCoefficientJac(Te,0)[:,0]) @ Te_nT
        ome_ne = np.multiply(ki_ne, ne) + np.multiply(ki, np.identity(self.Np))
        ome_Te = np.multiply(ki_nT, ne)

        omE_ne = -self.params.dH[0]*ome_ne
        omE_Te = -self.params.dH[0]*ome_Te

        SJ_ne = -self.params.qStar*( np.multiply(fe_ne,-phi_x) + np.multiply(fe,-phi_x_ne))
        SJ_ni = -self.params.qStar*( np.multiply(fe_ni,-phi_x) + np.multiply(fe,-phi_x_ni))

        # spatial part of residual
        self.jac[0:self.Np,0:self.Np]         = dt*(fe_x_ne - ome_ne)
        self.jac[0:self.Np,self.Np:2*self.Np] = dt*(fe_x_ni         )
        self.jac[0:self.Np,2*self.Np:]        = dt*(        - ome_Te)

        self.jac[self.Np:2*self.Np,0:self.Np]         = dt*(fi_x_ne - ome_ne)
        self.jac[self.Np:2*self.Np,self.Np:2*self.Np] = dt*(fi_x_ni         )
        self.jac[self.Np:2*self.Np,2*self.Np:]        = dt*(        - ome_Te)

        self.jac[2*self.Np:,0:self.Np]         = dt*(fT_x_ne - omE_ne - SJ_ne)
        self.jac[2*self.Np:,self.Np:2*self.Np] = dt*(fT_x_ni          - SJ_ni)
        self.jac[2*self.Np:,2*self.Np:]        = dt*(fT_x_Te - omE_Te        )


        # time derivative part of residual
        if (not first_step): # BDF2
            print("Shouldn't be here!")
            exit(-1)
        else: # BDF1 = backward Euler
            self.jac[0:self.Np,0:self.Np] += np.identity(self.Np)
            self.jac[self.Np:2*self.Np,self.Np:2*self.Np] += np.identity(self.Np)
            #self.jac[2*self.Np:,0:self.Np ] += np.multiply(np.identity(self.Np),Te)
            self.jac[2*self.Np:,2*self.Np:] += np.identity(self.Np) #np.multiply(ne,np.identity(self.Np))

        # boundary conditions (strongly enforced)
        #res[0]           = fe[ 0]  - (-self.params.ks*ne[ 0] - self.params.gam*fi[ 0])
        self.jac[0,:] = np.zeros((1,3*self.Np))
        self.jac[0,0:self.Np] = fe_ne[0,:] - (- self.params.gam*fi_ne[ 0,:])
        self.jac[0,self.Np:2*self.Np] = fe_ni[0,:] - (- self.params.gam*fi_ni[ 0,:])
        self.jac[0,0] += self.params.ks

        #res[self.Np-1]   = fe[-1]  - ( self.params.ks*ne[-1] - self.params.gam*fi[-1])
        self.jac[self.Np-1,:] = np.zeros((1,3*self.Np))
        self.jac[self.Np-1,0:self.Np] = fe_ne[-1,:] - (- self.params.gam*fi_ne[-1,:])
        self.jac[self.Np-1,self.Np:2*self.Np] = fe_ni[-1,:] - (- self.params.gam*fi_ni[-1,:])
        self.jac[self.Np-1,self.Np-1] -= self.params.ks

        ## # boundary conditions (strongly enforced)
        ##res[0]           = fe[ 0]  - (-self.params.ks*ne[ 0] - self.params.gam*fi[ 0])
        #self.jac[0,:] = np.zeros((1,3*self.Np))
        #self.jac[0,0] = 1.0

        # #res[self.Np-1]   = fe[-1]  - ( self.params.ks*ne[-1] - self.params.gam*fi[-1])
        # self.jac[self.Np-1,:] = np.zeros((1,3*self.Np))
        # self.jac[self.Np-1,self.Np-1] = 1.0

        #res[2*self.Np  ] = (Te[ 0] - 0.75)
        self.jac[2*self.Np,:] = np.zeros((1,3*self.Np))
        self.jac[2*self.Np,2*self.Np] = 1.0
        self.jac[2*self.Np,0] = -0.75

        #res[3*self.Np-1] = (Te[-1] - 0.75)
        self.jac[3*self.Np-1,:] = np.zeros((1,3*self.Np))
        self.jac[3*self.Np-1,3*self.Np-1] = 1.0
        self.jac[3*self.Np-1,self.Np-1] = -0.75

        # #res[2*self.Np  ] = fT[ 0] - ((5./3.)*fe[0]*Te[ 0])
        # self.jac[2*self.Np,:] = np.zeros((1,3*self.Np))
        # self.jac[2*self.Np,0:self.Np] = fT_ne[0,:] - ((5./3.)*fe_ne[0,:]*Te[0])
        # self.jac[2*self.Np,self.Np:2*self.Np] = fT_ni[0,:] - ((5./3.)*fe_ni[0,:]*Te[0])
        # self.jac[2*self.Np,2*self.Np:] = fT_Te[0,:]
        # self.jac[2*self.Np,2*self.Np] += - (5./3.)*fe[0]

        # #res[3*self.Np-1] = fT[-1] - ((5./3.)*fe[-1]*Te[-1])
        # self.jac[3*self.Np-1,:] = np.zeros((1,3*self.Np))
        # self.jac[3*self.Np-1,0:self.Np] = fT_ne[-1,:] - ((5./3.)*fe_ne[-1,:]*Te[-1])
        # self.jac[3*self.Np-1,self.Np:2*self.Np] = fT_ni[-1,:] - ((5./3.)*fe_ni[-1,:]*Te[-1])
        # self.jac[3*self.Np-1,2*self.Np:] = fT_Te[-1,:]
        # self.jac[3*self.Np-1,3*self.Np-1] += - (5./3.)*fe[-1]

    def jacobian0(self, dt, first_step=False):
        """Evaluate the Jacobian of the residual with respect to the state at
        the previous time step

        Inputs:
          dt   : Time step

        Outputs: None (sets self.jac0)

        Notes:
          This function currently assumes that Ns=2 and NT=1

        """

        # form state at previous step at collocation points
        ne1 = self.U1[0:self.Np]
        ni1 = self.U1[self.Np:2*self.Np]
        nT1 = self.U1[2*self.Np:]


        if (not first_step): # BDF2
            print("Shouldn't be here!")
            exit(-1)
        else: # BDF1 = backward Euler
            self.jac0[0:self.Np,0:self.Np]                     = -np.identity(self.Np)
            self.jac0[self.Np:2*self.Np,self.Np:2*self.Np]     = -np.identity(self.Np)
            #self.jac0[2*self.Np:3*self.Np,0:self.Np]           = -np.multiply(np.identity(self.Np),Te1)
            self.jac0[2*self.Np:3*self.Np,2*self.Np:3*self.Np] = -np.identity(self.Np) #-np.multiply(ne1,np.identity(self.Np))

        # for boundary conditions that are strongly enforced,
        # corresponding residual has no dependence on previous state
        self.jac0[0        ,:] = np.zeros((1,3*self.Np))
        self.jac0[self.Np-1,:] = np.zeros((1,3*self.Np))

        self.jac0[2*self.Np  ,:] = np.zeros((1,3*self.Np))
        self.jac0[3*self.Np-1,:] = np.zeros((1,3*self.Np))


    def jacobianFD(self, Uin, time, dt, first_step=False):
        """Evaluates the Jacobian at Uin, but using a finite difference
        approximation.  Useful for testing, but very slow.

        Inputs:
          Uin  : Current state
          time : Current time
          dt   : Time step

        Outputs: None (sets self.jac)
        """
        # save residual at Uin
        r0 = self.residual(Uin, time, dt, first_step)

        # perturb each component of Uin to form finite differenc approx
        for k in range(0,Uin.shape[0]):
            dU = np.sqrt(np.finfo(np.float).eps)*np.absolute(Uin[k])
            Up = np.copy(Uin)

            if (np.absolute(dU) < np.finfo(np.float).eps):
                dU = np.finfo(np.float).eps

            Up[k] += dU

            rp = self.residual(Up, time, dt, first_step)
            self.jac[:,k] = (rp[:,0] - r0[:,0])/dU


    def step(self, time, dt, iter_max=10, rtol=1e-6, atol=1e-12,
             first_step=False, verbose=False):
        """Take a single time step.

        Inputs
          time       : Current time
          dt         : Time step
          iter_max   : Maximum number of iters in nonlinear solve
          rtol       : Relative tolerance for nonlinear solve
          atol       : Absolute tolerance for nonlinear solve
          first_step : If true, use backward Euler
          verbose    : If true, print nonlinear solve info

        Outputs: None (self.U2 is set to solution for this time step)
        """
        r = self.residual(self.U2, time, dt, first_step)

        normr = normr0 = np.linalg.norm(r)
        count = 0
        converged = ((normr/normr0 < rtol) or (normr < atol))
        if (verbose):
            print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(
                count, normr, normr/normr0))
        while( not converged and (count < iter_max) ):
            #self.jacobianFD(self.U2, time, dt, first_step)
            self.jacobian(self.U2, time, dt, first_step)

            try:
                dU = np.linalg.solve(self.jac, -r)
                self.U2 += dU

                # zero the last mode
                #self.filter()

            except:
                # if exception encountered, save state and die
                np.save("exception_U2.npy", self.U2)
                np.save("exception_U1.npy", self.U1)
                np.save("exception_U0.npy", self.U0)
                print("Solve failed!", flush=True)
                exit(-1)

            r = self.residual(self.U2, time, dt, first_step)
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

    def stepSensitivity(self, time, dt, first_step=False, verbose=False):
        """Advance the sensitivity matrix

        Inputs
          time       : Current time
          dt         : Time step
          first_step : If true, use backward Euler
          verbose    : If true, print nonlinear solve info

        Outputs: None (self.A1 is set to sensitivity at the end of the time step)
        """
        # evaluate the required Jacobians
        self.jacobian(self.U2, time, dt, first_step)
        self.jacobian0(dt, first_step)

        self.rhsSens = -(self.jac0 @ self.A0)

        # solve the sensitivity update system
        self.A1 = np.linalg.solve(self.jac, self.rhsSens)

        if (verbose):
            print("# Advancing sensitivity system.")


    def solve(self, time0, dt, Nstep, savedata=None, verbose=False, rtol=1e-6, computeSensitivity=False):

        if(savedata!=None):
            Usave=np.ndarray((Nstep+1,self.U2.shape[0]),dtype=np.float)
            Usave[0,:] = self.U2[:,0]

        print("#")
        print("# {0:10s} {1:12s} {2:12s} {3:12s} {4:12s}".format(
            "Time", "min ne", "max ne", "min Te", "max Te"))
        print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
            time0, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[2*self.Np:].min(), self.U2[2*self.Np:].max()))

        # assume initial condition has been set in U1!
        time = time0+dt
        self.step(time, dt, first_step=True, verbose=verbose, rtol=rtol)
        print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
            time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
            self.U2[2*self.Np:].min(), self.U2[2*self.Np:].max()))

        if(computeSensitivity):
            self.stepSensitivity(time, dt, first_step=True, verbose=verbose)


        if(savedata!=None):
            Usave[1,:] = self.U2[:,0]


        for istep in range(1, Nstep):
            # prepare for next step
            self.U0 = np.copy(self.U1)
            self.U1 = np.copy(self.U2)
            time += dt

            if (computeSensitivity):
                self.A0 = np.copy(self.A1)

            # advance
            self.step(time, dt, first_step=True, verbose=verbose, rtol=rtol)
            #self.filter()
            print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
                time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[2*self.Np:].min(), self.U2[2*self.Np:].max()), flush=True)

            if(savedata!=None):
                Usave[istep+1,:] = self.U2[:,0]

            if(computeSensitivity):
                self.stepSensitivity(time, dt, first_step=True, verbose=verbose)

        if(savedata!=None):
            np.save(savedata,Usave)


    def plot(self, col, create=True):
        import matplotlib.pyplot as plt
        xplot, w = cheb.chebgauss(2*self.Np)

        fig = plt.figure(num=1,figsize=(16,27))
        if (create):
            ax = []
            ax.append(plt.subplot(3,1,1,label='ne'))
            ax.append(plt.subplot(3,1,2,label='ne',sharex=ax[0]))
            ax.append(plt.subplot(3,1,3,label='Te',sharex=ax[0]))
        else:
            ax = fig.get_axes()

        ax[0].plot(xplot, cheb.chebval(xplot, (self.V0pinv @ self.U2[0:self.Np])[:,0]), col, lw=3)
        ax[0].grid(True)
        plt.setp(ax[0].get_xticklabels(),visible=False)
        plt.setp(ax[0].get_yticklabels(),fontsize=14)
        ax[0].set_ylabel(r'$n_e$',fontsize=16)


        ax[1].plot(xplot, cheb.chebval(xplot, (self.V0pinv @ self.U2[self.Np:2*self.Np])[:,0]), col, lw=3)
        ax[1].grid(True)
        plt.setp(ax[1].get_xticklabels(),visible=False)
        plt.setp(ax[1].get_yticklabels(),fontsize=14)
        ax[1].set_ylabel(r'$n_i$',fontsize=16)

        ax[2].plot(xplot, cheb.chebval(xplot, (self.V0pinv @ self.U2[2*self.Np:])[:,0]), col, lw=3)
        ax[2].grid(True)
        plt.setp(ax[2].get_xticklabels(),fontsize=14)
        plt.setp(ax[2].get_yticklabels(),fontsize=14)
        ax[2].set_ylabel(r'$T_e$',fontsize=16)
        ax[2].set_xlabel(r'$x$',fontsize=16)



if __name__ == "__main__":
    desc  = "# \n"
    desc += "# chebSolver: A program for simulating glow discharge devices\n"
    desc += "#             using a 1-D, time-domain, drift-diffusion model\n"
    desc += "#             discretized with a Chebyshev-collocation/BDF   \n"
    desc += "#             approach.                                      \n"
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
    parser.add_argument('--rtol',metavar='rtol', default=1e-6,
                        type=float, help="Relative tolerance for non-linear solve")
    parser.add_argument('--restart', metavar='rst.npy', default=None,
                        help='Restart file (*.npy format, must have same Np)')
    parser.add_argument('--outfile', metavar='out.npy', default='result.npy',
                        help='Filename to save restart file')
    parser.add_argument('--savedata', metavar='save.npy',default=None,
                        help='Filename to save every time step')
    parser.add_argument('--verbose',default=False,
                        action='store_true', help='Be extra chatty')
    parser.add_argument('--plot', default=False,
                        action='store_true', help="Plot the final state for inspection.")
    args = parser.parse_args()

    # Dump inputs to the screen for posterity
    print("# Input parameters:")

    print("#   Number of Chebyshev points (Np) = {0:d}".format(args.Np))
    print("#   Number of time steps (Nt)       = {0:d}".format(args.Nt))
    print("#   Size of time step (dt)          = {0:.6e}".format(args.dt))
    print("#   Initial time (t0)               = {0:.6e}".format(args.t0))
    print("#   Relative tolerance (rtol)       = {0:.6e}".format(args.rtol))

    if(args.restart!=None):
        print("#")
        print("#   Restarting from {0:s}".format(args.restart))
    else:
        print("#")
        print("#   No restart file provided.")
        print("#   Using uniform IC with ne = ni = 1e-4, Te = 0.5.")

    print("#   Save file time step to {0:s}".format(args.outfile))

    if(args.savedata!=None):
        print("#")
        print("#   Saving every time step to {0:s}".format(args.savedata))
    else:
        print("#")
        print("#   Not saving every time step (use --savedata for this option).")

    print("#")

    # Instantiate solver class
    tds = timeDomainCollocationSolver(2,1,args.Np)

    # Default IC (may be overwritten below if we are restarting)
    tds.U1[0:tds.Np] = 1e-4
    tds.U1[tds.Np:2*tds.Np] = 1e-4
    tds.U1[2*tds.Np:] = 0.75*tds.U1[0:tds.Np]

    # If restart file provided, read it.
    # NOTE: currently we do a lazy restart in that only the final
    # state is saved, so we have to restart with a backward Euler step.
    if (args.restart!=None):
        tds.U1 = np.load(args.restart)

    # Initialize rest of state
    tds.U0 = np.copy(tds.U1)
    tds.U2 = np.copy(tds.U1)

    # Run for desired number of time steps
    tds.solve(args.t0, args.dt, args.Nt,
              args.savedata, args.verbose, args.rtol)

    # Save the result
    np.save(args.outfile, tds.U2)

    if(args.plot):
        tds.plot('b-')
        plt.show()
