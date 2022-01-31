import numpy as np
import matplotlib.pyplot as plt
import chebSolver as cs

class timePeriodicSolver:

    def __init__(self, Ns, NT, Np, elasticCollisionActivationFactor,
                 backgroundSpecieActivationFactor,
                 gam, V0, VDC, restart=None, scenario=0, scheme='BE'):
        self.tds = cs.timeDomainCollocationSolver(Ns, NT, Np,
                                                  elasticCollisionActivationFactor,
                                                  backgroundSpecieActivationFactor,
                                                  gam, V0,
                                                  VDC, scenario, scheme)
        self.res = np.zeros((self.tds.Ndof,1))
        self.jac = np.zeros((self.tds.Ndof,self.tds.Ndof))

        if (restart!=None):
            self.tds.U1 = np.load(restart)

        else:
            # Default initial guess.  This should be overwritten
            # by reading restart if you want this to work.
            self.tds.U1[0:self.tds.Np] = 1e-4
            self.tds.U1[self.tds.Np:2*self.tds.Np] = 1e-4
            self.tds.U1[2*self.tds.Np:] = 0.75

        self.tds.U2 = np.copy(self.tds.U1)



    def periodicityResidual(self, Uic, Nt):
        '''
        Compute the "periodicity residual"---i.e., the difference between
        Uic and the solution 1 period later.

        Inputs:
        Uic : array specifying initial condition for time domain solver
        tds : chebSolver.timeDomainCollocationSolver class
        Nt  : Number of time steps (for single period)

        Returns:
        None.  Residual is computed
        '''
        # reset ICs for time domain solve
        self.tds.U1 = np.copy(Uic)
        self.tds.U2 = np.copy(Uic)
        self.tds.A0 = np.copy(np.identity(self.tds.Ndof))
        self.tds.A1 = np.copy(np.identity(self.tds.Ndof))

        # Run from IC for 1 period
        self.tds.solve(0.0, 1.0/Nt, Nt,
                       savedata=None, verbose=False, rtol=1e-8,
                       computeSensitivity=True, weak_bc=False)

        # Compute difference between final state and Uic
        self.res = self.tds.U2 - Uic

        # Compute the Jacobian
        A = np.identity(self.tds.Ndof)
        A[(self.tds.Ns-1)*self.tds.Np+1:self.tds.Ns*self.tds.Np-1,
          (self.tds.Ns-1)*self.tds.Np+1:self.tds.Ns*self.tds.Np-1] -= np.identity(self.tds.Np-2) \
            * (1.0 - self.tds.backgroundSpecieActivationFactor)
        self.jac = self.tds.A1 - A

        # return norm of residual
        return np.linalg.norm(self.res)

    def solveNewtonStep(self, Uic, Nt):
        # solve for newton update
        Uic += np.linalg.solve(self.jac, -self.res)


if __name__ == "__main__":
    desc  = "# \n"
    desc += "# timePeriodicSolver: A program for simulating glow discharge\n"
    desc += "#     devices using a 1-D, time-domain, drift-diffusion model\n"
    desc += "#     discretized with a Chebyshev-collocation/BDF approach  \n"
    desc += "#     with enforced time periodicity.                        \n"
    desc += "#"
    print(desc)

    # Define and parse command line arguments
    import argparse
    usage = "python3 ./timeDomainSolver"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('--Np', metavar='Np', default=100,
                        type=int, help='Number of Chebyshev points')
    parser.add_argument('--Nt', metavar='Nt', default=16,
                        type=int, help='Number of time steps (per period)')
    parser.add_argument('--Nn', metavar='Nn', default=20,
                        type=int, help='Maximum number of Newton iterations')
    parser.add_argument('--gam', metavar='gam', default=0.01,
                        type=float, help='Secondary electron emission parameter')
    parser.add_argument('--rtol',metavar='rtol', default=1e-6,
                        type=float, help="Relative tolerance for non-linear solve")
    parser.add_argument('--atol',metavar='atol', default=1e-14,
                        type=float, help="Absolute tolerance for non-linear solve")
    parser.add_argument('--scenario', metavar='scenario', default=0,
                        type=int, help='Scenario index')
    parser.add_argument('--restart', metavar='rst.npy', default=None,
                        help='Restart file (*.npy format, must have same Np)')
    parser.add_argument('--tscheme', metavar='time_disc',default="BE",
                        help='Temporal scheme indicator [BE or CN]')
    parser.add_argument('--outfile', metavar='out.npy', default='result.npy',
                        help='Filename to save restart file')
    parser.add_argument('--verbose',default=False,
                        action='store_true', help='Be extra chatty')
    parser.add_argument('--plot', default=False,
                        action='store_true', help="Plot the final state for inspection.")
    parser.add_argument('--V0', metavar='V0', default=100,
                        type=float, help='Voltage amplitude')
    parser.add_argument('--VDC', metavar='VDC', default=0.0,
                        type=float, help='Vertical shift of voltage sinusoidal')
    parser.add_argument('--elasticCollisionActivation', default=False,
                         action='store_true', help="Activate the elastic collision term.")
    parser.add_argument('--backgroundSpecieActivation', default=False,
                        action='store_true', help="Activate the background specie density equation.")
    args = parser.parse_args()

    # Dump inputs to the screen for posterity
    print("# Input parameters:")

    print("#   Temporal scheme (tscheme)       = {0:s}".format(args.tscheme))
    print("#   Number of Chebyshev points (Np) = {0:d}".format(args.Np))
    print("#   Number of time steps (Nt)       = {0:d}".format(args.Nt))
    print("#   Maximum Newton iterationss (Nn) = {0:d}".format(args.Nn))
    print("#   Secondary electron param (gam)  = {0:.6e}".format(args.gam))
    print("#   Relative tolerance (rtol)       = {0:.6e}".format(args.rtol))
    print("#   Absolute tolerance (atol)       = {0:.6e}".format(args.atol))

    if(args.restart!=None):
        print("#")
        print("#   Restarting from {0:s}".format(args.restart))
    else:
        print("#")
        print("#   No restart file provided.")
        print("#   Using uniform IC with ne = ni = 1e-4, Te = 0.5.")

    print("#   Save final state to {0:s}".format(args.outfile))

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
        print("#   Running scenario = 4 (4 species, 8 rxn, Liu 2017)")
        Ns = 4
    else:
        print("ERROR: Scenario not recognized.  Use --scenario i with i=0, 1, 2, or 3.  Exiting.")
        exit(-1)

    print("#")

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

    tps = timePeriodicSolver(Ns, 1, args.Np, elasticCollisionActivationFactor,
                             backgroundSpecieActivationFactor,
                             args.gam, args.V0, args.VDC,
                             restart=args.restart, scenario=args.scenario,
                             scheme=args.tscheme)


    # Get the IC, for use in computing the residual below
    Uic = np.copy(tps.tds.U1)

    # evaluate the residual
    rnorm0 = tps.periodicityResidual(Uic, args.Nt)
    rnorm = rnorm0
    resPrint="Newton step {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}"
    print(resPrint.format(0,rnorm,rnorm/rnorm0))

    # Newton iterations
    niter = 0
    while ( (rnorm/rnorm0 > args.rtol) and (rnorm > args.atol) and (niter<args.Nn) ):
        tps.solveNewtonStep(Uic, args.Nt)

        if (args.plot):
            tps.tds.U2 = np.copy(Uic)
            tps.tds.plot('r-',create=True)
            plt.show()

        rnorm = tps.periodicityResidual(Uic, args.Nt)
        niter += 1
        print(resPrint.format(niter,rnorm,rnorm/rnorm0))

    if (args.plot):
        tps.tds.U2 = np.copy(Uic)
        tps.tds.plot('r-',create=True)
        plt.show()

    # save final state
    np.save(args.outfile, Uic)

