#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py"
NEWTEXE="python3 ./timePeriodicSolver.py"


# 6 species + 34 rxn - Nominal Rates case
Np=150
Nt=250
# dt=0.03125
dt=0.015625

Nt1=128
dt1=0.0078125
scenario=15
baseFile="restart.npy"
newtFile="restart_shooting.npy"
saveFile="fullsoln.npy"


baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --elasticCollisionActivation --backgroundSpecieActivation"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme BE --alpha0 0.1 --increaseFac 1.5 --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt1 --scenario $scenario --tscheme BE --elasticCollisionActivation --backgroundSpecieActivation"


screenOut="run.txt"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run time marching case..."
$baseCmd --V0 100 --VDC 0.0 --t0 25.5 --outfile  $baseFile --restart "restart_crashed.npy" --verbose #|| error_exit "First run failed"

# echo "Run time domain shooting..."
# $newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart $baseFile \
#                                 --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

# echo "Saving one period..."
# $saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut

