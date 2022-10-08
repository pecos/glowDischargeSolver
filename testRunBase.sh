#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py"
NEWTEXE="python3 ./timePeriodicSolver.py"

# Liu 4 species case
Np=100
Nt=6400
Nt1=128
dt=0.0078125
scenario=0
baseFile="restart_3spec_Np${Np}_"
newtFile="newton_3spec_Np${Np}.npy"
saveFile="newton_3spec_Np${Np}_fullsoln.npy"

newtCmd="time $NEWTEXE --Np $Np --Nt $Nt1 --Nn 10 --scenario $scenario"
diffCmd="python3 ./diffSolns.py --reference reference_solns/scenario0_Np100_solution.npy --Np $Np --Nv 4"

screenOut="runBase.out"
rm -f $screenOut

$newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart reference_solns/scenario0_Np100_restart.npy \
         --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

$diffCmd --solution $newtFile >> $screenOut || error_exit "Solution differs from reference"
