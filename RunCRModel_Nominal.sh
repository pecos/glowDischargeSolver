#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

# EXE="python3 ./chebSolver.py"
# NEWTEXE="python3 ./timePeriodicSolver.py"
EXE="python3 ./chebSolver.py --use_gpu 0 --gpu_device_id 0"
NEWTEXE="python3 ./timePeriodicSolver.py --use_gpu 0 --gpu_device_id 0"


# 6 species + 34 rxn - Nominal Rates case
Np=150
Nt=32000
dt=0.0078125

Nt1=128
dt1=0.0078125

scenario=15
# crashedFile="restart_CR_Np150_crashed.npy"
baseFile="restart_CR_Np${Np}_"
# newtFile="temp_CR_Np${Np}.npy"
newtFile="${baseFile}T2000.npy"
saveFile="CR_Np${Np}_fullsoln.npy"


baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --elasticCollisionActivation --backgroundSpecieActivation"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme BE --alpha0 0.1 --increaseFac 1.5 --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt1 --scenario $scenario --tscheme BE --elasticCollisionActivation --backgroundSpecieActivation"


screenOut="run.txt"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run 0 to 250...${baseFile}T250.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 0.0 \
               --outfile "${baseFile}T250.npy" > $screenOut || error_exit "First run failed"
echo "Run 250 to 500...${baseFile}T500.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 250.0 --restart "${baseFile}T250.npy"  \
               --outfile "${baseFile}T500.npy" >> $screenOut || error_exit "Second run failed"
echo "Run 500 to 750...${baseFile}T750.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 500.0 --restart "${baseFile}T500.npy"  \
               --outfile "${baseFile}T750.npy" >> $screenOut || error_exit "Third run failed"
echo "Run 750 to 1000...${baseFile}T1000.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 750.0 --restart "${baseFile}T750.npy"  \
               --outfile "${baseFile}T1000.npy" >> $screenOut || error_exit "Fourt run failed"
echo "Run 1000 to 1250...${baseFile}T1250.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 1000.0 --restart "${baseFile}T1000.npy" \
               --outfile "${baseFile}T1250.npy" >> $screenOut || error_exit "Fifth run failed"
echo "Run 1250 to 1500...${baseFile}T1500.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 1250.0 --restart "${baseFile}T1250.npy" \
               --outfile "${baseFile}T1500.npy" >> $screenOut || error_exit "Sixth run failed"
echo "Run 1500 to 1750...${baseFile}T1750.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 1500.0 --restart "${baseFile}T1500.npy" \
               --outfile "${baseFile}T1750.npy" >> $screenOut || error_exit "Seventh run failed"
echo "Run 1750 to 2000...${baseFile}T2000.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 1750.0 --restart "${baseFile}T1750.npy" \
               --outfile "${baseFile}T2000.npy" >> $screenOut || error_exit "Seventh run failed"

# echo "Run time domain shooting...${newtFile}"
# $newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T200.npy" \
#                                 --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

echo "Saving one period...${saveFile}"
$saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut


