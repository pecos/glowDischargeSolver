## Overview

`glowDischargeSolver` is a collection of python functions supporting
the approximation of plasma systems in "glow discharge" regime.  The
model takes the form of a fluid model that uses the drift-diffusion
approximation so that the state variables consist of a set of species
number densities and an electron temperature.  For more details on the
physical model, see the LaTeX documentation in the `doc` directory.

These equations are discretized in space using Chebyshev collation and
in time using fully implicit schemes, either backward Euler or
Crank-Nicolson.  Further, to support the time-periodic case,
time-periodic-shooting is implemented to accelerate converged to the
time-periodic solution.
