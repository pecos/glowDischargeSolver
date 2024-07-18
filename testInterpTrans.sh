#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py"
NEWTEXE="python3 ./timePeriodicSolver.py"

# Liu 4 species case
Np=150
Nt=128
Nt1=128
dt=0.0078125
scenario=21
baseFile="restart_4spec_CN_Np${Np}_"
newtFile="newton_4spec_CN_Np${Np}.npy"
saveFile="newton_4spec_CN_Np${Np}_fullsoln.npy"


baseCmd="time $EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --verbose"
newtCmd="time $NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --alpha0 0.1 --increaseFac 2."


diffCmd="python3 ./diffSolns.py --reference reference_solns/scenario21_Np150_solution.npy --Np $Np --Nv 5"

screenOut="run21.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver

$newtCmd --weakbc --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart reference_solns/scenario21_Np150_restart.npy \
         --outfile $newtFile >> $screenOut 2>&1 || error_exit "Shooting failed"

$diffCmd --solution $newtFile >> $screenOut || error_exit "Solution differs from reference"
