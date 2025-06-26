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
Nt=512
# Nt=10
dt=0.0078125
# dt=0.00390625
# dt=0.001953125

Nt1=128
dt1=0.0078125
# Nt1=256
# dt1=0.00390625
# Nt1=512
# dt1=0.001953125

Nt2=256
dt2=0.00390625

# # For adaptive
Nt=400
dt=1.0
Nt1=1
dt1=1.0


# Model Selection
scenario=7
# scenario=15

baseFile="restart_CR_BE_Np${Np}_"
newtFile="newton_CR_CN_Np${Np}.npy"
saveFile="newton_CR_CN_Np${Np}_fullsoln.npy"

# baseFile="restart_6spec_CN_Np${Np}_"
# newtFile="newton_6spec_CN_Np${Np}.npy"
# saveFile="newton_6spec_CN_Np${Np}_fullsoln.npy"

# baseFile="restart_8spec_CN_Np${Np}_"
# newtFile="newton_8spec_CN_Np${Np}.npy"
# saveFile="newton_8spec_CN_Np${Np}_fullsoln.npy"

FLAGS="--V0 75 --VDC 0.0 --gam 0.1 --scenario $scenario --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation --IonEffEField"
# --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation --IonEffEField --lineSearch

baseCmd="$EXE $FLAGS --Np $Np --Nt $Nt --dt $dt --tscheme BE"
newtCmd="$NEWTEXE $FLAGS --Np $Np --Nt $Nt1 --Nn 20 --tscheme CN --alpha0 0.1 --increaseFac 1.5 --adaptive"
saveCmd="$EXE $FLAGS --Np $Np --Nt $Nt2 --dt $dt2 --tscheme CN"


screenOut="runPrint.out"
rm -f $screenOut


# echo "Run 1 period ..."
# $baseCmd --t0 0.0 --restart "${baseFile}T125.npy" --outfile "discard.npy"
# $baseCmd --t0 0.0 --restart "discard.npy"  --verbose --outfile "discard.npy"  --adaptive
# $baseCmd --rtol 1e-6 --t0 200.0 --restart "restart_cycle_0200.npy" --outfile "discard_adaptive.npy" --verbose --adaptive
# $baseCmd --t0 0.0 --restart "newton_CR_CN_Np150.npy"  --verbose --outfile "discard.npy" 


# echo "Run 0 to 125...${baseFile}T125.npy"
# $baseCmd --t0 0.0 --outfile "${baseFile}T125.npy" > $screenOut || error_exit "First run failed"

# echo "Run time domain shooting...${newtFile}"
# $newtCmd --rtol 1e-6 --restart "${baseFile}T125.npy" \
#                                 --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

$newtCmd --rtol 1e-6 --restart "restart.npy" \
                                --outfile $newtFile || error_exit "Shooting failed"


# echo "Saving one period...${saveFile}"
# $saveCmd --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut
# $saveCmd --rtol 1e-8 --savedata $saveFile --verbose --outfile discard.npy >> $screenOut