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
newtCmd="time $NEWTEXE --Np $Np --Nt $Nt1 --Nn 10 --scenario $scenario --tscheme CN"
saveCmd="time $EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario --tscheme CN"

screenOut="runCN.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run 0 to 50... Save to ${baseFile}T050.npy"
$baseCmd --V0 100 --VDC 0.0 --t0   0.0 --outfile "${baseFile}T050.npy"  > $screenOut || error_exit "First run failed"

echo "Run 100 to 200... Save to ${baseFile}T100.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 100.0 --restart "${baseFile}T050.npy"  \
         --outfile "${baseFile}T100.npy" >> $screenOut || error_exit "Second run failed"

echo "Run time domain shooting... Save to ${newtFile}"
$newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T100.npy" \
         --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

echo "Saving one period to ${saveFile}"
$saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut
