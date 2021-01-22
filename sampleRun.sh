#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py"
Np=150
Nt=6400
dt=0.015625

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt"
screenOut="run.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
$baseCmd --t0   0.0 --outfile restart_Np150_T100.npy  > run.out || error_exit "First run failed"
$baseCmd --t0 100.0 --restart restart_Np150_T100.npy \
	            --outfile restart_Np150_T200.npy >> run.out || error_exit "Second run failed"
