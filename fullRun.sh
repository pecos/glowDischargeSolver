#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py"
Np=150
Nt=32000
dt=0.015625


baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt"
screenOut="fullRun.out"
rm -f $screenOut

# Run Chebyshev time domain solver out to 4000 periods
$baseCmd --t0    0.0 --outfile restart_Np150_T0500.npy  > $screenOut || error_exit "First run failed"
$baseCmd --t0  500.0 --restart restart_Np150_T0500.npy \
	             --outfile restart_Np150_T1000.npy >> $screenOut || error_exit "Second run failed"
$baseCmd --t0 1000.0 --restart restart_Np150_T1000.npy \
	             --outfile restart_Np150_T1500.npy >> $screenOut || error_exit "Third run failed"
$baseCmd --t0 1500.0 --restart restart_Np150_T1500.npy \
	             --outfile restart_Np150_T2000.npy >> $screenOut || error_exit "Fourth run failed"
$baseCmd --t0 2000.0 --restart restart_Np150_T2000.npy \
	             --outfile restart_Np150_T2500.npy >> $screenOut || error_exit "Fifth run failed"
$baseCmd --t0 2500.0 --restart restart_Np150_T2500.npy \
	             --outfile restart_Np150_T3000.npy >> $screenOut || error_exit "Sixth run failed"
$baseCmd --t0 3000.0 --restart restart_Np150_T3000.npy \
	             --outfile restart_Np150_T3500.npy >> $screenOut || error_exit "Seventh run failed"
$baseCmd --t0 3500.0 --restart restart_Np150_T3500.npy \
	             --outfile restart_Np150_T4000.npy >> $screenOut || error_exit "Eighth run failed"
