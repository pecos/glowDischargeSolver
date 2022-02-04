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
#Np=300
#Nt=25600
#Nt1=128
#dt=0.0078125
#scenario=0
#baseFile="restart_3spec_CN_Np${Np}_"
#newtFile="newton_3spec_CN_Np${Np}.npy"
#saveFile="newton_3spec_CN_Np${Np}_fullsoln.npy"

# Liu 4 species case
Np=300
Nt=25600
Nt1=128
dt=0.0078125
scenario=2
baseFile="restart_4spec_CN_Np${Np}_"
newtFile="newton_4spec_CN_Np${Np}.npy"
saveFile="newton_4spec_CN_Np${Np}_fullsoln.npy"

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 10 --scenario $scenario --tscheme CN"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario --tscheme CN"

# Activation of background specie and elastic collisions.
#baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --elasticCollisionActivation --backgroundSpecieActivation"
#newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 10 --scenario $scenario --tscheme CN --elasticCollisionActivation --backgroundSpecieActivation"
#saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario --tscheme CN --elasticCollisionActivation --backgroundSpecieActivation"

screenOut="runCN.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run 0 to 200...${baseFile}T200.npy"
$baseCmd --V0 100 --VDC 0.0 --t0   0.0 --outfile "${baseFile}T200.npy"  > $screenOut || error_exit "First run failed"
echo "Run 200 to 400...${baseFile}T400.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 200.0 --restart "${baseFile}T200.npy"  \
                --outfile "${baseFile}T400.npy" >> $screenOut || error_exit "Second run failed"
echo "Run 400 to 600...${baseFile}T600.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 400.0 --restart "${baseFile}T400.npy"  \
                --outfile "${baseFile}T600.npy" >> $screenOut || error_exit "Second run failed"

echo "Run time domain shooting...${newtFile}"
$newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T600.npy" \
                                --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

echo "Saving one period...${saveFile}"
$saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut
