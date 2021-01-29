import numpy as np
import chebSolver as cs



# Instantiate time domain solver
tds = cs.timeDomainCollocationSolver(2,1,100)

# Default initial guess.  This should be overwritten
# by reading restart if you want this to work.
tds.U1[0:tds.Np] = 1e-4
tds.U1[tds.Np:2*tds.Np] = 1e-4
tds.U1[2*tds.Np:] = 0.75

# Load restart (hardcoded for now)
tds.U1 = np.load("restart_Np150_T200.npy")
#tds.U1 = np.load("newton_restart.npy")

# Initialize rest of state
tds.U0 = np.copy(tds.U1)
tds.U2 = np.copy(tds.U1)

# Save the IC, for use in computing the residual below
Uic = np.copy(tds.U1)

# Set parameters of time stepper
Nt = 64
dt = 1./Nt
t0 = 0.0

rtol = 1e-6
rnorm = 1.0
niter=0

# Newton iterations
while ( (rnorm > rtol) and (niter<30) ):

    # reset ICs for time domain solve
    tds.U1 = np.copy(Uic)
    tds.A0 = np.copy(np.identity(tds.A0.shape[0]))
    
    # Run from IC for 1 period
    tds.solve(t0, dt, Nt, savedata=None, verbose=False, rtol=1e-6, computeSensitivity=True)

    # Compute difference between final state and Uic
    res = tds.U2 - Uic
    rnorm = np.linalg.norm(res)
    print("Newton residual: ||res|| = {0:.6e}".format(rnorm))

    # Compute the Jacobian
    jac = tds.A1 - np.identity(tds.A1.shape[0])

    # Newton update
    dU = np.linalg.solve(jac, -res)

    Uic += dU

    niter += 1

np.save('newton_restart.npy', Uic)
