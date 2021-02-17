#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py"
#Np=150
#Nt=6400
#dt=0.015625
Np=250
Nt=2560
dt=0.00390625

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt"
screenOut="run.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
$baseCmd --t0   0.0 --outfile restart_psaap_Np250_T10.npy  > run.out || error_exit "First run failed"
$baseCmd --t0  10.0 --restart restart_psaap_Np250_T10.npy \
                    --outfile restart_psaap_Np250_T20.npy >> run.out || error_exit "Second run failed"
$baseCmd --t0  20.0 --restart restart_psaap_Np250_T20.npy \
                    --outfile restart_psaap_Np250_T30.npy >> run.out || error_exit "Third run failed"
$baseCmd --t0  30.0 --restart restart_psaap_Np250_T30.npy \
                    --outfile restart_psaap_Np250_T40.npy >> run.out || error_exit "Fourth run failed"
$baseCmd --t0  40.0 --restart restart_psaap_Np250_T40.npy \
                    --outfile restart_psaap_Np250_T50.npy >> run.out || error_exit "Fifth run failed"
