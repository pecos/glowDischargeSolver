#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ../chebSolver.py"
NEWTEXE="python3 ../timePeriodicSolver.py"

# 4 species + 9 rxn - Nominal Rates case
Np=150
Nt=25600
Nt1=128
dt=0.0078125
scenario=5
baseFile="restart_4spec_CN_Np${Np}_"
newtFile="newton_4spec_CN_Np${Np}.npy"
saveFile="newton_4spec_CN_Np${Np}_fullsoln.npy"

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --elasticCollisionActivation --backgroundSpecieActivation"
#newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 10 --scenario $scenario --tscheme CN"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme CN --alpha0 0.1 --increaseFac 1.5 --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario --tscheme CN --elasticCollisionActivation --backgroundSpecieActivation"
screenOut="runCN.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run 0 to 200...${baseFile}T200.npy"
$baseCmd --V0 100 --VDC 0.0 --t0   0.0 --outfile "${baseFile}T200.npy"  > $screenOut || error_exit "First run failed"
#echo "Run 200 to 400...${baseFile}T400.npy"
#$baseCmd --V0 100 --VDC 0.0 --t0 200.0 --restart "${baseFile}T200.npy"  \
#                --outfile "${baseFile}T400.npy" >> $screenOut || error_exit "Second run failed"
#echo "Run 400 to 600...${baseFile}T600.npy"
#$baseCmd --V0 100 --VDC 0.0 --t0 400.0 --restart "${baseFile}T400.npy"  \
#                --outfile "${baseFile}T600.npy" >> $screenOut || error_exit "Second run failed"
#echo "Run 600 to 800...${baseFile}T800.npy"
#$baseCmd --V0 100 --VDC 0.0 --t0 600.0 --restart "${baseFile}T600.npy"  \
#                --outfile "${baseFile}T800.npy" >> $screenOut || error_exit "Second run failed"

echo "Run time domain shooting...${newtFile}"
$newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T200.npy" \
                                --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

echo "Saving one period...${saveFile}"
$saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut
