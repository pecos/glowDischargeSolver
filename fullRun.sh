#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py"
#Np=150
#Nt=32000
#dt=0.015625

# Liu 3 species case
Np=150
Nt=64000
dt=0.0078125
baseFile='restart_3spec_Np150_'

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt"
screenOut="fullRun.out"
rm -f $screenOut

# Run Chebyshev time domain solver out to 4000 periods
$baseCmd --t0    0.0 --outfile "${baseFile}T0500.npy" > $screenOut || error_exit "First run failed"
$baseCmd --t0  500.0 --restart "${baseFile}T0500.npy" \
	             --outfile "${baseFile}T1000.npy" >> $screenOut || error_exit "Second run failed"
$baseCmd --t0 1000.0 --restart "${baseFile}T1000.npy" \
	             --outfile "${baseFile}T1500.npy" >> $screenOut || error_exit "Third run failed"
$baseCmd --t0 1500.0 --restart "${baseFile}T1500.npy" \
	             --outfile "${baseFile}T2000.npy" >> $screenOut || error_exit "Fourth run failed"
$baseCmd --t0 2000.0 --restart "${baseFile}T2000.npy" \
	             --outfile "${baseFile}T2500.npy" >> $screenOut || error_exit "Fifth run failed"
$baseCmd --t0 2500.0 --restart "${baseFile}T2500.npy" \
	             --outfile "${baseFile}T3000.npy" >> $screenOut || error_exit "Sixth run failed"
$baseCmd --t0 3000.0 --restart "${baseFile}T3000.npy" \
	             --outfile "${baseFile}T3500.npy" >> $screenOut || error_exit "Seventh run failed"
$baseCmd --t0 3500.0 --restart "${baseFile}T3500.npy" \
	             --outfile "${baseFile}T4000.npy" >> $screenOut || error_exit "Eighth run failed"
