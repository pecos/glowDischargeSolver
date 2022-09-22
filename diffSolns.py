import numpy as np

def checkSize(rs, ts, Np, Nv):
    Dref = np.load(rs)
    Dnew = np.load(ts)

    if Dref.shape[0] != Dnew.shape[0]:
        return False
    if Dref.shape[0] != Np * Nv:
        return False

    return True

def evaluateDifference(rs, ts, Np, Nv):
    Dref = np.load(rs)
    Dnew = np.load(ts)

    Uref = Dref.reshape((Np, Nv), order="F")
    Unew = Dnew.reshape((Np, Nv), order="F")

    dnorm = np.zeros(Nv)
    rnorm = np.zeros(Nv)

    for i in range(Nv):
        delta = Unew[:,i] - Uref[:,i]
        dnorm[i] = np.linalg.norm(delta)
        rnorm[i] = np.linalg.norm(Uref[:,i])

    return dnorm, rnorm


if __name__ == "__main__":
    desc  = "# \n"
    desc += "# diffSolns: A program to compute differences between solutions\n"
    desc += "#            produced by chebSolver.py or timePeriodicSolver.py\n"
    desc += "#            for regression testing purposes.                  \n"
    desc += "#"
    print(desc)

    # Define and parse command line arguments
    import argparse
    usage = "python3 ./diffSolns.py"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('--Np', metavar='Np', default=100,
                        type=int, help='Number of Chebyshev points')
    parser.add_argument('--Nv', metavar='Nv', default=4,
                        type=int, help='Number of state variables')
    parser.add_argument('--reference', metavar='reference', default='ref_soln.npy',
                        help='Reference solution file (*.npy format)')
    parser.add_argument('--solution', metavar='solution', default='new_soln.npy',
                        help='Solution to compare to reference (*.npy format)')
    parser.add_argument('--rtol', metavar='rtol', default=1e-8,
                        type=float, help='Relative tolerance on difference')
    args = parser.parse_args()


    rel_norm_tol = args.rtol

    reference_soln = args.reference
    test_soln = args.solution
    Np = args.Np
    Nv = args.Nv

    if not checkSize(reference_soln, test_soln, Np, Nv):
        print("Size error.  Cannot compare files.")
        print("Check that reference and solution files are same size")
        print("and that they are consistent with Np and Nv")
        exit(1)

    difference_norm, reference_norm = evaluateDifference(reference_soln, test_soln, Np, Nv)
    print("Summary of differences:")
    for i in range(Nv):
        print("  State {0:d}: {1:.6e}".format(i, difference_norm[i]))

    test_fail = False
    for i in range(Nv):
        if difference_norm[i] / reference_norm[i] > rel_norm_tol:
            print("Component {0:d} failed relative l2 norm check", i)
            test_fail = True

    if not test_fail:
        print("All differences with tolerances.  Test passed")
    else:
        print("*** Differences outside tolerances deteced.  Test FAILED. ***");
        exit(1)

    exit(0)

