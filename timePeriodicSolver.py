import numpy as np
import matplotlib.pyplot as plt
import chebSolver as cs
import time as cpu_time

class timePeriodicSolver:

    def __init__(self, args, Ns, NT, Np, elasticCollisionActivationFactor,
                 backgroundSpecieActivationFactor, EinsteinForm, IonEffEField,
                 gam, V0, VDC, restart=None, scenario=0, scheme='BE',
                 alpha0 = 1.0, increaseFac = 1.0, iSample = 0):
        self.tds = cs.timeDomainCollocationSolver(args, Ns, NT, Np,
                                                  elasticCollisionActivationFactor,
                                                  backgroundSpecieActivationFactor,
                                                  EinsteinForm,IonEffEField,
                                                  gam, V0,
                                                  VDC, scenario, scheme, iSample)

        self.args      = args
        self.xp_module = np

        self.res = np.zeros((self.tds.Ndof,1))
        self.jac = np.zeros((self.tds.Ndof,self.tds.Ndof))

        self.I_Np =  np.identity(self.tds.Np)
        self.I_Ndof = np.identity(self.tds.Ndof)

        if (restart!=None):
            self.tds.U1 = np.load(restart)

        else:
            # Default initial guess.  This should be overwritten
            # by reading restart if you want this to work.
            self.tds.U1[0:(self.tds.Ns-1)*self.tds.Np] = 1.0e-4             # 'usual' species
            self.tds.U1[(self.tds.Ns-1)*self.tds.Np:self.tds.Ns*self.tds.Np] = 1.0  # background specie
            self.tds.U1[self.tds.Ns*self.tds.Np:(self.tds.Ns+1)*self.tds.Np] = self.tds.params.EeBC*self.tds.U1[0:self.tds.Np] # electron energy
            if IonEffEField:
                # pull off state for convenience
                dens = np.zeros((self.tds.Np, self.tds.Ns),dtype=np.float64)
                for i in range(0,self.tds.Ns):
                    dens[:,i] = self.tds.U1[i*self.tds.Np:(i+1)*self.tds.Np,0]                
                self.tds.solve_poisson(dens,1.0/args.Nt)
                self.tds.U1[(self.tds.Nv-1)*self.tds.Np:self.tds.Nv*self.tds.Np] = self.tds.phi # effective electric field for ions


        self.tds.U2 = np.copy(self.tds.U1)

        self.alpha = alpha0
        self.increaseFac = increaseFac


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
        
        I_Ndof = self.I_Ndof

        # reset ICs for time domain solve
        self.tds.U1 = np.copy(Uic)
        self.tds.U2 = np.copy(Uic)
        self.tds.A0 = np.copy(I_Ndof)
        self.tds.A1 = np.copy(I_Ndof)

        # Run from IC for 1 period
        self.tds.solve(0.0, 1.0/Nt, Nt,
                       savedata=None, verbose=True, rtol=1e-8,
                       computeSensitivity=True, weak_bc=self.args.weakbc,
                       jac_frequency=self.args.jacfreq)

        # Compute difference between final state and Uic
        self.res = self.tds.U2 - Uic

        # Compute the Jacobian
        

        # A = I_Ndof
        A = np.copy(I_Ndof)
        # A = np.identity(self.tds.Ndof)

        self.jac = self.tds.A1 - A
 

        # if we aren't solving for the background specie density, need
        # to modify Jacobian to avoid having Np rows of 0 for the to
        # the background specie equations, which obviously leads to a
        # singular matrix.  This problem occurs b/c the time periodic
        # condition is satisfied for any constant, and this fix simply
        # enforces that the background specie doesn't change.
        if (self.tds.backgroundSpecieActivationFactor == 0):
            self.jac[(self.tds.Ns-1)*self.tds.Np:self.tds.Ns*self.tds.Np,
                     (self.tds.Ns-1)*self.tds.Np:self.tds.Ns*self.tds.Np] = self.I_Np 

        # return norm of residual
        return np.linalg.norm(self.res)

    def solveNewtonStep(self, Uic, Nt):
        # solve for newton update
        #Uic += np.linalg.solve(self.jac, -self.res)
        print("Solving sensitivity system...")
        Uic += self.alpha * np.linalg.solve(self.jac, -self.res)
        print("Finished.")

        if (self.alpha < 1):
            self.alpha *= self.increaseFac
            self.alpha = min(self.alpha, 1)


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
    parser.add_argument('--atol',metavar='atol', default=1e-12,
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
    parser.add_argument('--weakbc',default=False,
                        action='store_true', help='Enforce ion flux BC weakly')
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
    parser.add_argument('--EinsteinForm', default=False,
                        action='store_true', help="Activate Einstein's form for diffusion coefficient for electrons.")
    parser.add_argument('--EinsteinFormIon', default=False,
                        action='store_true', help="Activate Einstein's form for diffusion coefficient for ions.")
    parser.add_argument('--IonEffEField', default=False,
                        action='store_true', help="Activate effective electric field for ions.")
    parser.add_argument('--alpha0', metavar='alpha0', default=1.0,
                        type=float, help='Newton step under-relaxation factor')
    parser.add_argument('--increaseFac', metavar='increaseFac', default=1.0,
                        type=float, help='Increase alpha by this factor each step')
    parser.add_argument('--iSample', metavar='iSample', default=0,
                        type=int, help='Sample index, if BOLSIG chemistry is used.')
    parser.add_argument("-use_gpu", "--use_gpu", help="use GPUs", type=int, default=0)
    parser.add_argument("-gpu_device_id", "--gpu_device_id", help="GPU device id to use", type=int, default=0)
    parser.add_argument('--jacfreq', metavar='J', default=1, type=int,
                        help='Evaluate Jacobian every J Newton iterations during time step')


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
        print('#   Running scenario = 4 (4 species, 9 rxn, 1Torr, Nominal)')
        Ns = 4
    elif(args.scenario==5):
        print('# Running scenario = 5 (4 species, 9 rxn, 1Torr, Sampling)')
        Ns = 4
    elif(args.scenario==6):
        print('#   Running scenario = 6 (6 species, 23 rxn, 1Torr, 100V, Sampling)')
        Ns = 6
    elif(args.scenario==8):
        print("#   Running scenario = 9 (6 species, 23 rxn, 500mTorr, 100V, Sampling)")
        Ns = 6
    elif(args.scenario==9):
        print("#   Running scenario = 9 (6 species, 23 rxn)")
        Ns = 6
    elif(args.scenario==10):
        print("#   Running scenario = 10 (6 species, 23 rxn, 100mTorr, Nominal)")
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
    elif(args.scenario==7):
        print('#   Running scenario = 7 (6 species, 34 rxn, 1Torr, Nominal)')
        Ns = 6
    elif(args.scenario==8):
        print('#   Running scenario = 8 (6 species, 34 rxn, 1Torr, Sampling)')
        Ns = 6
    elif(args.scenario==9):
        print('#   Running scenario = 9 (6 species, 34 rxn, 2Torr, Nominal)')
        Ns = 6
    elif(args.scenario==10):
        print('#   Running scenario = 10 (6 species, 34 rxn, 2Torr, Sampling)')
        Ns = 6
    elif(args.scenario==11):
        print('#   Running scenario = 11 (6 species, 34 rxn, 5Torr, Nominal)')
        Ns = 6
    elif(args.scenario==12):
        print('#   Running scenario = 12 (6 species, 34 rxn, 5Torr, Sampling)')
        Ns = 6
    elif(args.scenario==13):
        print('#   Running scenario = 13 (6 species, 34 rxn, 10Torr, Nominal)')
        Ns = 6
    elif(args.scenario==14):
        print('#   Running scenario = 14 (6 species, 34 rxn, 10Torr, Sampling)')
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


    tps = timePeriodicSolver(args, Ns, 1, args.Np, elasticCollisionActivationFactor,
                             backgroundSpecieActivationFactor,
                             EinsteinForm, IonEffEField,
                             args.gam, args.V0, args.VDC,
                             restart=args.restart, scenario=args.scenario,
                             scheme=args.tscheme,
                             alpha0 = args.alpha0, increaseFac = args.increaseFac,
                             iSample = args.iSample)


    # Get the IC, for use in computing the residual below
    Uic = np.copy(tps.tds.U1)

    # evaluate the residual
    rnorm0 = tps.periodicityResidual(Uic, args.Nt)
    rnorm = rnorm0
    resPrint="Newton step {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}"
    print(resPrint.format(0,rnorm,rnorm/rnorm0))

    tic_0 = cpu_time.time()


    # Newton iterations
    niter = 0
    while ( (rnorm/rnorm0 > args.rtol) and (rnorm > args.atol) and (niter<args.Nn) ):
        tic = cpu_time.time()
        
        tps.solveNewtonStep(Uic, args.Nt)

        if (args.plot):
            tps.tds.U2 = np.copy(Uic)
            tps.tds.plot('r-',create=True)
            plt.show()

        print("Calling periodicityResidual...")
        rnorm = tps.periodicityResidual(Uic, args.Nt)
        print("Done...")
        niter += 1
        print(resPrint.format(niter,rnorm,rnorm/rnorm0))
        print(f"CPU Time per Newton iter is {cpu_time.time() - tic} seconds.")

        

    if (args.plot):
        tps.tds.U2 = np.copy(Uic)
        tps.tds.plot('r-',create=True)
        plt.show()

    # save final state
    np.save(args.outfile, Uic)

    toc = cpu_time.time()
    print(f"Total CPU Time = {toc -tic_0} seconds.")

    print("Periodic solver finished successfully.")    

