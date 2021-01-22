import numpy as np
import numpy.polynomial.chebyshev as cheb
import matplotlib.pyplot as plt

from Liu2014Properties import setLiu2014Properties

class modelClosures:
    """Class providing model parameters."""
    
    def __init__(self):
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
        self.mue = 1.47
        self.mui = 7.07e-3

        self.De = 5.86e-2 
        self.Di = 3.15e-6 

        self.Ck = 272.0
        self.A = 18.687*(3./2.);
        self.dH = 15.7

        self.qStar = 100.0

        self.alpha = 2.33e3

        self.gam = 0.01 
        self.ks = 6.89e-1 
        self.ksion = 0.0
        
    def eleMobility(self):
        """Returns electron mobility"""
        return self.mue

    def ionMobility(self):
        """Returns ion mobility"""
        return self.mui

    def eleDiffusivity(self):
        """Returns electron diffusivity"""
        return self.De

    def ionDiffusivity(self):
        """Returns electron diffusivity"""
        return self.Di

    def rxnRateCoefficient(self, energy):
        """Returns ionization reaction rate constant"""
        return self.Ck*np.exp(-self.A/energy)

    def rxnRateCoefficientJac(self, energy):
        """Returns derivative of ionization reaction rate constant wrt
        energy
        """
        return self.Ck*np.exp(-self.A/energy)*(self.A/(energy*energy))

    def print(self):
        """Print parameters to the screen"""
        print("# The non-dimensional transport and chemstry properties are")
        print("#   De    = {0:.6e}".format(self.De))
        print("#   Di    = {0:.6e}".format(self.Di))
        print("#   mue   = {0:.6e}".format(self.mue))
        print("#   mui   = {0:.6e}".format(self.mui))
        print("#   Ck    = {0:.6e}".format(self.Ck))
        print("#   A     = {0:.6e}".format(self.A))
        print("#   dH    = {0:.6e}".format(self.dH))
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
        self.params = modelClosures()
        setLiu2014Properties(gam, self.params)

        # Points used to define state (Gauss-Lobatto-Chebyshev points)
        self.xp = -np.cos(np.pi*np.linspace(0,self.deg,self.Np)/self.deg)

        # Points used for collocation (Gauss-Chebyshev)
        #self.xc = -np.cos(np.pi*(np.linspace(1,self.Nc,self.Nc)-0.5)/self.Nc)
        #self.xc = np.zeros(self.xp.shape)
        #self.xc[0] = -1.0
        #self.xc[1:-1] = -np.cos(np.pi*(np.linspace(1,self.Nc,self.Nc)-0.5)/self.Nc)
        #self.xc[-1] = 1.0
        self.xc = self.xp

        # Jacobian storage
        self.jac = np.zeros((self.Ndof, self.Ndof))
        
        # Operators
        ident = np.identity(self.Np)

        # V0p: Coefficients to values at xp
        self.V0p = cheb.chebvander(self.xp, self.deg)

        # V0pinv: xp values to coefficients
        self.V0pinv = np.linalg.solve(self.V0p, ident)

        # V0c: Coefficients to values at xc
        self.V0c = cheb.chebvander(self.xc, self.deg)

        # Mc: xp values to xc values
        self.Mc = self.V0c @ self.V0pinv

        # Mc0: xp values to xc, with 0 on 'top' and 'bottom'
        self.Mc0 = np.zeros((self.Np, self.Np))
        self.Mc0[1:-1,:] = self.Mc[1:-1,:]

        # V1p: coefficients to derivatives at xp
        self.V1p = np.zeros((self.Np,self.Np))
        for i in range(0,self.Np): 
            self.V1p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=1))

        # Dp: values at xp to derivatives at xp
        self.Dp = self.V1p @ self.V0pinv

        # V1c: coefficients to derivatives at xc
        #self.V1c = np.zeros((self.Nc,self.Np))
        self.V1c = np.zeros((self.Np,self.Np))
        for i in range(0,self.Np): 
            self.V1c[:,i] = cheb.chebval(self.xc, cheb.chebder(ident[i,:], m=1))

        # Dc: values at xp to derivatives at xc
        self.Dc = self.V1c @ self.V0pinv

        # V2c: coefficients to 2nd derivatives at xc
        #self.V2c = np.zeros((self.Nc,self.Np))
        self.V2c = np.zeros((self.Np,self.Np))
        for i in range(0,self.Np): 
            self.V2c[:,i] = cheb.chebval(self.xc, cheb.chebder(ident[i,:], m=2))

        # Lc: values at xp to 2nd derivatives at xc
        self.Lc = self.V2c @ self.V0pinv

        # LcD: values at xp to 2nd derivatives at xc, with identity
        # for top and bottom row (for Dirichlet BCs)
        self.LcD = np.identity(self.Np)
        self.LcD[1:-1,:] = self.Lc[1:-1,:]
        

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
        r = -self.params.alpha* (self.Mc0 @ (ni-ne))
        r[-1] = np.sin(2*np.pi*time)
        self.phi = np.linalg.solve(self.LcD, r)
        
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
        # pull off state for convenience
        nep = Uin[0:self.Np]
        nip = Uin[self.Np:2*self.Np]
        Tep = Uin[2*self.Np:]

        # solve poisson equation for phi
        # now have self.phi
        self.solve_poisson(nep,nip,time)

        # form state at collocation points
        ne = self.Mc @ nep
        ni = self.Mc @ nip
        Te = self.Mc @ Tep

        ne1 = self.Mc @ self.U1[0:self.Np]
        ni1 = self.Mc @ self.U1[self.Np:2*self.Np]
        Te1 = self.Mc @ self.U1[2*self.Np:]

        ne0 = self.Mc @ self.U0[0:self.Np]
        ni0 = self.Mc @ self.U0[self.Np:2*self.Np]
        Te0 = self.Mc @ self.U0[2*self.Np:]


        # form fluxes at grid points
        ne_x  = self.Dp @ nep
        ni_x  = self.Dp @ nip
        Te_x  = self.Dp @ Tep
        phi_x = self.Dp @ self.phi

        #fe = -self.params.eleMobility()*nep*(-phi_x) - self.params.eleDiffusivity()*ne_x
        fe = -self.params.eleMobility()*nep*(-phi_x)
        #fi =  self.params.ionMobility()*nip*(-phi_x) - self.params.ionDiffusivity()*ni_x
        fi =  self.params.ionMobility()*nip*(-phi_x)
        fT = (5./3.)*(-self.params.eleMobility()*nep*Tep*(-phi_x) - self.params.eleDiffusivity()*(ne_x*Tep + nep*Te_x))
        #fT = (5./3.)*(-self.params.eleMobility()*nep*Tep*(-phi_x) - self.params.eleDiffusivity()*(nep*Te_x))
        #fT = (5./3.)*(-self.params.eleMobility()*nep*Tep*(-phi_x) - self.params.eleDiffusivity()*(self.Dp @ (nep*Tep) ))

        # overwrite endpoints in fi (weakly impose BC)
        fi[ 0] = -self.params.ksion*nip[ 0] + self.params.ionMobility()*nip[ 0]*(-phi_x[ 0])
        fi[-1] =  self.params.ksion*nip[-1] + self.params.ionMobility()*nip[-1]*(-phi_x[-1])

        #fe[ 0] = (-self.params.ks*nep[ 0] - self.params.gam*fi[ 0])
        #fe[-1] = ( self.params.ks*nep[-1] - self.params.gam*fi[-1])


        # form derivatives of fluxes at collocation points
        fe_x = self.Dc @ fe - self.params.eleDiffusivity()*self.Lc @ nep
        fi_x = self.Dc @ fi - self.params.ionDiffusivity()*self.Lc @ nip
        fT_x = self.Dc @ fT

        # prep for BCs and src terms
        fe = -self.params.eleMobility()*nep*(-phi_x) - self.params.eleDiffusivity()*ne_x
        #fe[0] = -self.params.eleMobility()*nep[0]*(-phi_x[0]) - self.params.eleDiffusivity()*ne_x[0]
        #fe[-1] = -self.params.eleMobility()*nep[-1]*(-phi_x[-1]) - self.params.eleDiffusivity()*ne_x[-1]

        # form source terms at collocation points
        
        ki = self.params.rxnRateCoefficient(Te)
        ome = ki*ne
        omi = ome

        omE = -self.params.dH*ome #- 0.0313*ne*Te
        SJ = -self.params.qStar*(self.Mc @ fe)*(-self.Mc @ phi_x)
        #SJ = -self.params.qStar*(self.Mc @ (fe*(-phi_x)))
        #SJ = -self.params.qStar*(self.Mc @ (-self.params.eleMobility()*nep*(-phi_x)*(-phi_x)))

        res = np.zeros((3*self.Np,1))

        # spatial part of residual
        res[0:self.Np]         = dt*(fe_x - ome)
        res[self.Np:2*self.Np] = dt*(fi_x - ome)
        res[2*self.Np:]        = dt*(fT_x - omE - SJ)
        

        # time derivative part of residual
        if (not first_step): # BDF2
            res[0:self.Np]           += 1.5*ne - 2.0*ne1 + 0.5*ne0
            res[self.Np:2*self.Np]   += 1.5*ni - 2.0*ni1 + 0.5*ni0
            res[2*self.Np:3*self.Np] += 1.5*ne*Te - 2.0*ne1*Te1 + 0.5*ne0*Te0
        else: # BDF1 = backward Euler
            res[0:self.Np]           += ne    - ne1
            res[self.Np:2*self.Np]   += ni    - ni1
            res[2*self.Np:3*self.Np] += ne*Te - ne1*Te1

        # boundary conditions (strongly enforced)
        res[0]           = fe[ 0]  - (-self.params.ks*nep[ 0] - self.params.gam*fi[ 0])
        res[self.Np-1]   = fe[-1]  - ( self.params.ks*nep[-1] - self.params.gam*fi[-1])
        #res[0]           = nep[0]
        #res[self.Np-1]   = nep[-1]

        # Dirichlet on temperature
        res[2*self.Np  ] = (Tep[ 0] - 0.75)
        res[3*self.Np-1] = (Tep[-1] - 0.75)
        #res[2*self.Np  ] = fT[ 0] - ((5./3.)*fe[0]*Tep[ 0])
        #res[3*self.Np-1] = fT[-1] - ((5./3.)*fe[-1]*Tep[-1])

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
        nep = Uin[0:self.Np]
        nip = Uin[self.Np:2*self.Np]
        Tep = Uin[2*self.Np:]

        # solve poisson equation for phi
        # now have self.phi
        phi_ni = np.linalg.solve(self.LcD, -self.params.alpha*self.Mc0)
        phi_ne = -phi_ni

        # form state at collocation points
        ne = self.Mc @ nep
        ni = self.Mc @ nip
        Te = self.Mc @ Tep

        # form fluxes at grid points
        ne_x  = self.Dp @ nep
        ni_x  = self.Dp @ nip
        Te_x  = self.Dp @ Tep
        phi_x = self.Dp @ self.phi
        phi_x_nep = self.Dp @ phi_ne
        phi_x_nip = self.Dp @ phi_ni

        fe = -self.params.eleMobility()*nep*(-phi_x) - self.params.eleDiffusivity()*ne_x
        fe_nep = ( -self.params.eleMobility()*np.multiply(np.identity(self.Np),-phi_x)
                   -self.params.eleMobility()*np.multiply(nep,-phi_x_nep) )
        #-self.params.eleDiffusivity()*self.Dp )
        fe_nip =   -self.params.eleMobility()*np.multiply(nep,-phi_x_nip)
        
        fi_nep = self.params.ionMobility()*np.multiply(nip,-phi_x_nep)
        fi_nip = (  self.params.ionMobility()*np.multiply(nip,-phi_x_nip)
                  + self.params.ionMobility()*np.multiply(np.identity(self.Np), -phi_x))
        #- self.params.ionDiffusivity()*self.Dp )
        
        fT_nep = (5./3.)*(-self.params.eleMobility()*np.multiply(np.identity(self.Np),Tep*(-phi_x))
                          -self.params.eleMobility()*np.multiply(nep*Tep,(-phi_x_nep))
                          - self.params.eleDiffusivity()*(np.multiply(self.Dp,Tep) + np.multiply(np.identity(self.Np),Te_x)))
        fT_nip = (5./3.)*(-self.params.eleMobility()*np.multiply(nep*Tep,(-phi_x_nip)))
        fT_Tep = (5./3.)*(-self.params.eleMobility()*np.multiply(nep*(-phi_x),np.identity(self.Np))
                          - self.params.eleDiffusivity()*(np.multiply(ne_x,np.identity(self.Np)) + np.multiply(nep,self.Dp)))
        
        # overwrite endpoints in fi (weakly impose BC)
        fi_nep[0,:] = self.params.ionMobility()*nip[ 0]*(-phi_x_nep[ 0,:])
        fi_nip[0,:] = self.params.ionMobility()*nip[ 0]*(-phi_x_nip[ 0,:])
        fi_nip[0,0] += -self.params.ksion + self.params.ionMobility()*(-phi_x[ 0])
        
        fi_nep[-1,:] = self.params.ionMobility()*nip[-1]*(-phi_x_nep[-1,:])
        fi_nip[-1,:] = self.params.ionMobility()*nip[-1]*(-phi_x_nip[-1,:])
        fi_nip[-1,-1] += self.params.ksion + self.params.ionMobility()*(-phi_x[-1])

        # #fe[ 0] = (-self.params.ks*nep[ 0] - self.params.gam*fi[ 0])
        # fe_nep[ 0,:] = - self.params.gam*fi_nep[ 0,:]
        # fe_nep[ 0,0] -= self.params.ks
        # fe_nip[ 0,:] = - self.params.gam*fi_nip[ 0,:]

        # #fe[-1] = ( self.params.ks*nep[-1] - self.params.gam*fi[-1])
        # fe_nep[-1,:] =  - self.params.gam*fi_nep[-1,:]
        # fe_nep[-1,-1] += self.params.ks
        # fe_nip[-1,:] =  - self.params.gam*fi_nip[-1,:]


        # form derivatives of fluxes at collocation points
        fe_x_nep = self.Dc @ fe_nep
        fe_x_nep -=  self.params.eleDiffusivity()*self.Lc
        
        fe_x_nip = self.Dc @ fe_nip

        fe_nep = ( -self.params.eleMobility()*np.multiply(np.identity(self.Np),-phi_x)
                   -self.params.eleMobility()*np.multiply(nep,-phi_x_nep) 
                   -self.params.eleDiffusivity()*self.Dp )
        fe_nip =   -self.params.eleMobility()*np.multiply(nep,-phi_x_nip)

        
        fi_x_nep = self.Dc @ fi_nep
        fi_x_nip = self.Dc @ fi_nip
        fi_x_nip -=  self.params.ionDiffusivity()*self.Lc



        fT_x_nep = self.Dc @ fT_nep
        fT_x_nip = self.Dc @ fT_nip
        fT_x_Tep = self.Dc @ fT_Tep

        # form source terms at collocation points
        
        ki = self.params.rxnRateCoefficient(Te)
        ki_Tep = np.multiply(self.params.rxnRateCoefficientJac(Te), self.Mc)
        ome_nep = np.multiply(ki, self.Mc)
        ome_Tep = np.multiply(ki_Tep, ne)
        
        #omE = -self.params.dH*ome - 0.0313*ne*Te
        omE_nep = -self.params.dH*ome_nep# - 0.0313*np.multiply(self.Mc, Te)
        omE_Tep = -self.params.dH*ome_Tep# - 0.0313*np.multiply(ne, self.Mc)

        
        #SJ = -self.params.qStar*(self.Mc @ fe)*(-self.Mc @ phi_x)
        SJ_nep = ( -self.params.qStar*np.multiply((self.Mc @ fe_nep),(-self.Mc @ phi_x))
                   -self.params.qStar*np.multiply((self.Mc @ fe    ),(-self.Mc @ phi_x_nep)))
        SJ_nip = ( -self.params.qStar*np.multiply((self.Mc @ fe_nip),(-self.Mc @ phi_x))
                   -self.params.qStar*np.multiply((self.Mc @ fe    ),(-self.Mc @ phi_x_nip)))
                   
        # spatial part of residual
        self.jac[0:self.Np,0:self.Np]         = dt*(fe_x_nep - ome_nep)
        self.jac[0:self.Np,self.Np:2*self.Np] = dt*(fe_x_nip          )
        self.jac[0:self.Np,2*self.Np:]        = dt*(         - ome_Tep)
        
        self.jac[self.Np:2*self.Np,0:self.Np]         = dt*(fi_x_nep - ome_nep)
        self.jac[self.Np:2*self.Np,self.Np:2*self.Np] = dt*(fi_x_nip          )
        self.jac[self.Np:2*self.Np,2*self.Np:]        = dt*(         - ome_Tep)
        
        self.jac[2*self.Np:,0:self.Np]         = dt*(fT_x_nep - omE_nep - SJ_nep)
        self.jac[2*self.Np:,self.Np:2*self.Np] = dt*(fT_x_nip           - SJ_nip)
        self.jac[2*self.Np:,2*self.Np:]        = dt*(fT_x_Tep - omE_Tep         )
        

        # time derivative part of residual
        if (not first_step): # BDF2
            print("Shouldn't be here!")
            exit(-1)
        else: # BDF1 = backward Euler
            self.jac[0:self.Np,0:self.Np] += self.Mc
            self.jac[self.Np:2*self.Np,self.Np:2*self.Np] += self.Mc #np.identity(self.Np)
            self.jac[2*self.Np:,0:self.Np ] += np.multiply(self.Mc,Te)
            self.jac[2*self.Np:,2*self.Np:] += np.multiply(ne,self.Mc)
            
        # boundary conditions (strongly enforced)
        #res[0]           = fe[ 0]  - (-self.params.ks*nep[ 0] - self.params.gam*fi[ 0])
        self.jac[0,:] = np.zeros((1,3*self.Np))
        self.jac[0,0:self.Np] = fe_nep[0,:] - (- self.params.gam*fi_nep[ 0,:])
        self.jac[0,self.Np:2*self.Np] = fe_nip[0,:] - (- self.params.gam*fi_nip[ 0,:])
        self.jac[0,0] += self.params.ks
                                 
        #res[self.Np-1]   = fe[-1]  - ( self.params.ks*nep[-1] - self.params.gam*fi[-1])
        self.jac[self.Np-1,:] = np.zeros((1,3*self.Np))
        self.jac[self.Np-1,0:self.Np] = fe_nep[-1,:] - (- self.params.gam*fi_nep[-1,:])
        self.jac[self.Np-1,self.Np:2*self.Np] = fe_nip[-1,:] - (- self.params.gam*fi_nip[-1,:])
        self.jac[self.Np-1,self.Np-1] -= self.params.ks

        ## # boundary conditions (strongly enforced)
        ##res[0]           = fe[ 0]  - (-self.params.ks*nep[ 0] - self.params.gam*fi[ 0])
        #self.jac[0,:] = np.zeros((1,3*self.Np))
        #self.jac[0,0] = 1.0
                                 
        # #res[self.Np-1]   = fe[-1]  - ( self.params.ks*nep[-1] - self.params.gam*fi[-1])
        # self.jac[self.Np-1,:] = np.zeros((1,3*self.Np))
        # self.jac[self.Np-1,self.Np-1] = 1.0

        #res[2*self.Np  ] = (Tep[ 0] - 0.75)
        self.jac[2*self.Np,:] = np.zeros((1,3*self.Np))
        self.jac[2*self.Np,2*self.Np] = 1.0
        
        #res[3*self.Np-1] = (Tep[-1] - 0.75)
        self.jac[3*self.Np-1,:] = np.zeros((1,3*self.Np))
        self.jac[3*self.Np-1,3*self.Np-1] = 1.0

        # #res[2*self.Np  ] = fT[ 0] - ((5./3.)*fe[0]*Tep[ 0])
        # self.jac[2*self.Np,:] = np.zeros((1,3*self.Np))
        # self.jac[2*self.Np,0:self.Np] = fT_nep[0,:] - ((5./3.)*fe_nep[0,:]*Tep[0])
        # self.jac[2*self.Np,self.Np:2*self.Np] = fT_nip[0,:] - ((5./3.)*fe_nip[0,:]*Tep[0])
        # self.jac[2*self.Np,2*self.Np:] = fT_Tep[0,:]
        # self.jac[2*self.Np,2*self.Np] += - (5./3.)*fe[0]
                
        # #res[3*self.Np-1] = fT[-1] - ((5./3.)*fe[-1]*Tep[-1])
        # self.jac[3*self.Np-1,:] = np.zeros((1,3*self.Np))
        # self.jac[3*self.Np-1,0:self.Np] = fT_nep[-1,:] - ((5./3.)*fe_nep[-1,:]*Tep[-1])
        # self.jac[3*self.Np-1,self.Np:2*self.Np] = fT_nip[-1,:] - ((5./3.)*fe_nip[-1,:]*Tep[-1])
        # self.jac[3*self.Np-1,2*self.Np:] = fT_Tep[-1,:]
        # self.jac[3*self.Np-1,3*self.Np-1] += - (5./3.)*fe[-1]
     
        
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


    def solve(self, time0, dt, Nstep, savedata=None, verbose=False, rtol=1e-6):

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

        if(savedata!=None):
            Usave[1,:] = self.U2[:,0]

        
        for istep in range(1, Nstep):
            # prepare for next step
            self.U0 = np.copy(self.U1)
            self.U1 = np.copy(self.U2)
            time += dt

            # advance
            self.step(time, dt, first_step=True, verbose=verbose, rtol=rtol)
            #self.filter()
            print("{0:.6e} {1:.6e} {2:.6e} {3:.6e} {4:.6e}".format(
                time, self.U2[0:self.Np].min(), self.U2[0:self.Np].max(),
                self.U2[2*self.Np:].min(), self.U2[2*self.Np:].max()), flush=True)

            if(savedata!=None):
                Usave[istep+1,:] = self.U2[:,0]

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
    import argparse
    desc  = "# \n"
    desc += "# chebSolver: A program for simulating glow discharge devices\n"
    desc += "#             using a 1-D, time-domain, drift-diffusion model\n"
    desc += "#             discretized with a Chebyshev-collocation/BDF   \n"
    desc += "#             approach.                                      \n"
    desc += "#"
    print(desc)

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

    
    
    print("# Input parameters:")
    
    print("#   Number of Chebyshev points (Np) = {0:d}".format(args.Np))
    print("#   Number of time steps (Nt)       = {0:d}".format(args.Nt))
    print("#   Size of time step (dt)          = {0:.6e}".format(args.dt))
    print("#   Initial time (t0)               = {0:.6e}".format(args.t0))

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

    # instantiate solver class
    tds = timeDomainCollocationSolver(2,1,args.Np)

    # Default IC (may be overwritten below if we are restarting)
    tds.U1[0:tds.Np] = 1e-4
    tds.U1[tds.Np:2*tds.Np] = 1e-4
    tds.U1[2*tds.Np:] = 0.75

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
