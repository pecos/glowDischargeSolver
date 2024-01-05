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
Nt=12800
dt=0.015625

Nt1=128
dt1=0.0078125

scenario=15
# crashedFile="restart_CR_Np150_crashed.npy"
baseFile="restart_CR_Np${Np}_"
# newtFile="temp_CR_Np${Np}.npy"
newtFile="${baseFile}T600.npy"
saveFile="CR_Np${Np}_fullsoln.npy"


baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --elasticCollisionActivation --backgroundSpecieActivation"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme BE --alpha0 0.1 --increaseFac 1.5 --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt1 --scenario $scenario --tscheme BE --elasticCollisionActivation --backgroundSpecieActivation"


screenOut="run.txt"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run 0 to 200...${baseFile}T200.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 0.0 \
               --outfile "${baseFile}T200.npy" > $screenOut || error_exit "First run failed"
echo "Run 200 to 400...${baseFile}T400.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 200.0 --restart "${baseFile}T200.npy"  \
               --outfile "${baseFile}T400.npy" >> $screenOut || error_exit "Second run failed"
echo "Run 400 to 600...${baseFile}T600.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 400.0 --restart "${baseFile}T400.npy"  \
               --outfile "${baseFile}T600.npy" >> $screenOut || error_exit "Second run failed"
#echo "Run 600 to 800...${baseFile}T800.npy"
#$baseCmd --V0 100 --VDC 0.0 --t0 600.0 --restart "${baseFile}T600.npy"  \
#                --outfile "${baseFile}T800.npy" >> $screenOut || error_exit "Second run failed"
#echo "Run 800 to 1000...${baseFile}T1000.npy"
#$baseCmd --V0 100 --VDC 0.0 --t0 800.0 --restart "${baseFile}T800.npy" \
#                --outfile "${baseFile}T1000.npy" >> $screenOut || error_exit "Fifth run failed"

# echo "Run time domain shooting...${newtFile}"
# $newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T200.npy" \
#                                 --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

echo "Saving one period...${saveFile}"
$saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut


