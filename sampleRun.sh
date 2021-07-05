#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py"
NEWTEXE="python3 ./timePeriodicSolver.py"

# psaap 2 species case
#Np=250
#Nt=2560
#dt=0.00390625
#baseFile='restart_psaap_Np250_'

# Liu 3 species case
#Np=150
#Nt=6400
#Nt1=128
#dt=0.0078125
#scenario=0
#baseFile="restart_3spec_Np${Np}_"
#newtFile="newton_3spec_Np${Np}.npy"
#saveFile="newton_3spec_Np${Np}_fullsoln.npy"

# Liu 3 species case
Np=150
Nt=6400
Nt1=128
dt=0.0078125
scenario=2
baseFile="restart_4spec_Np${Np}_"
newtFile="newton_4spec_Np${Np}.npy"
saveFile="newton_4spec_Np${Np}_fullsoln.npy"

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 10 --scenario $scenario"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario"
screenOut="run.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run 0 to 50...${baseFile}T50.npy"
$baseCmd --t0    0.0 --outfile "${baseFile}T50.npy"  > run.out || error_exit "First run failed"
echo "Run 50 to 100...${baseFile}T100.npy"
$baseCmd --t0   50.0 --restart "${baseFile}T50.npy"  \
                     --outfile "${baseFile}T100.npy" >> run.out || error_exit "Second run failed"

echo "Run time domain shooting...${newtFile}"
$newtCmd --gam 0.01 --rtol 1e-8 --restart "${baseFile}T100.npy" \
                                --outfile $newtFile >> run.out || error_exit "Shooting failed"

echo "Saving one period...${saveFile}"
$saveCmd --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> run.out
