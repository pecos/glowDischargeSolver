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
Nt=6400
Nt1=128
dt=0.0078125
scenario=2
baseFile="restart_4spec_CN_Np${Np}_"
newtFile="newton_4spec_CN_Np${Np}.npy"
saveFile="newton_4spec_CN_Np${Np}_fullsoln.npy"

baseCmd="time $EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario"
newtCmd="time $NEWTEXE --Np $Np --Nt $Nt1 --Nn 10 --scenario $scenario --tscheme CN --backgroundSpecieActivation"
saveCmd="time $EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario --tscheme CN"

diffCmd="python3 ./diffSolns.py --reference reference_solns/scenario2_back_Np150_solution.npy --Np $Np --Nv 5"

screenOut="runBackground.out"
rm -f $screenOut

#echo "Run time domain shooting... Save to ${newtFile}"
$newtCmd --weakbc --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart reference_solns/scenario2_Np150_solution.npy \
         --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

$diffCmd --solution $newtFile >> $screenOut || error_exit "Solution differs from reference"

