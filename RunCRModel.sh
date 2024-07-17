#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py --use_gpu 0 --gpu_device_id 0"
NEWTEXE="python3 ./timePeriodicSolver.py --use_gpu 0 --gpu_device_id 0"

# Solver Parameters
Np=150
Nt=16000
# Nt=10
# dt=0.0078125
dt=0.00390625
# dt=0.001953125

# Nt1=128
# dt1=0.0078125
Nt1=256
dt1=0.00390625

# Model Selection
scenario=15

baseFile="restart_CR_BE_Np${Np}_"
newtFile="newton_CR_CN_Np${Np}.npy"
saveFile="newton_CR_CN_Np${Np}_fullsoln.npy"

FLAGS="--V0 150 --VDC 0.0 --gam 0.1 --scenario $scenario --EinsteinForm --backgroundSpecieActivation"
# --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation

baseCmd="$EXE $FLAGS --Np $Np --Nt $Nt --dt $dt --tscheme BE"
newtCmd="$NEWTEXE $FLAGS --Np $Np --Nt $Nt1 --Nn 20 --tscheme CN --alpha0 0.05 --increaseFac 1.4"
saveCmd="$EXE $FLAGS --Np $Np --Nt $Nt1 --dt $dt1 --tscheme CN"


screenOut="run_CR.out"
rm -f $screenOut


# echo "Run 1 period ..."
# $baseCmd --t0 0.0 --verbose --restart "newton_CR_CN_Np150.npy" --outfile "discard.npy"
$baseCmd --t0 0.0 --verbose --outfile "discard.npy"

# 5.800781e+00
# 5.804688e+00
# 5.808594e+00

# echo "Run 0 to 125...${baseFile}T125.npy"
# $baseCmd --t0 5.804688e+00 --restart "nonconverged_U0_Ion.npy" --outfile "${baseFile}T125.npy" > $screenOut || error_exit "First run failed"

# echo "Run time domain shooting...${newtFile}"
# $newtCmd --rtol 1e-8 --restart "${baseFile}T125.npy" \
#                                 --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

# echo "Saving one period...${saveFile}"
# $saveCmd --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut
