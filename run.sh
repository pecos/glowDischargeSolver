#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 ./chebSolver.py --use_gpu 0 --gpu_device_id 0"
NEWTEXE="python3 ./timePeriodicSolver.py --use_gpu 0 --gpu_device_id 0"


# 6 species + 34 rxn - Nominal Rates case
Np=150
# Nt=32000
Nt=128

dt=0.0078125
# dt=0.00390625
# dt=0.001953125
# dt=0.0009765625

Nt1=128
dt1=0.0078125
scenario=15
baseFile="restart.npy"
newtFile="restart_shooting.npy"
saveFile="fullsoln.npy"

# baseFile="restart_CR_BE_Np${Np}_"
# newtFile="newton_CR_BE_Np${Np}.npy"
# saveFile="newton_CR_BE_Np${Np}_fullsoln.npy"
# --EinsteinForm
baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --tscheme BE --elasticCollisionActivation --backgroundSpecieActivation"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme CN --alpha0 0.05 --increaseFac 1.2 --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt1 --scenario $scenario --tscheme CN --elasticCollisionActivation --backgroundSpecieActivation"


screenOut="run.txt"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run time marching case..."
$baseCmd --V0 75 --VDC 0.0 --t0 0.0 --outfile  $baseFile --verbose #|| error_exit "First run failed"

echo "Run time domain shooting..."
$newtCmd --V0 75 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart $baseFile \
                                --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

# echo "Saving one period..."
# $saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $baseFile --savedata $saveFile --outfile discard.npy
# $saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut


# Run an example case of the Chebyshev time domain solver
# echo "Run time marching case..."
# $baseCmd --V0 75 --VDC 0.0 --t0 14.0 --restart "nonconverged_U0.npy" \
              #  --outfile  $baseFile --verbose #|| error_exit "First run failed"


# echo "Saving one period...${saveFile}"
# $saveCmd --V0 100 --VDC 0.0 --t0 500.0 --rtol 1e-8 --restart "restart_CR_Np150_T500.npy" --savedata $saveFile --outfile discard_1.npy

# Run an example case of the Chebyshev time domain solver
# echo "Run time marching case..."
# $baseCmd --V0 100 --VDC 0.0 --t0 500.0 --restart "restart_CR_Np150_T500.npy" --outfile discard_1.npy # --verbose #|| error_exit "First run failed"
