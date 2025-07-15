#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

EXE="python3 -u ./chebSolver.py --use_gpu 0 --gpu_device_id 0"

# Solver Parameters
Np=150

Nt=256
dt=0.00390625

# # For adaptive
#Nt=250
#dt=1.0

# Model Selection
# scenario=7
scenario=15


baseFile="restart_CR_BE_Np${Np}_"

FLAGS="--V0 75 --VDC 0.0 --gam 0.1 --scenario $scenario --EinsteinForm --EinsteinFormIon --elasticCollisionActivation --backgroundSpecieActivation --IonEffEField"

#baseCmd="$EXE $FLAGS --Np $Np --Nt $Nt --dt $dt --tscheme BE --verbose --adaptive"
baseCmd="$EXE $FLAGS --Np $Np --Nt $Nt --dt $dt --tscheme BE --jacfreq 4 --verbose"

screenOut="runPrint.out"
rm -f $screenOut


echo "Run 0 to 1...${baseFile}T1.npy"
$baseCmd --t0 0.0 --outfile "${baseFile}T1.npy" > $screenOut || error_exit "First run failed"


diffCmd="python3 ./diffSolns.py --reference reference_solns/crmodel_Np150_1period_solution.npy --Np $Np --Nv 21"

$diffCmd --solution "${baseFile}T1.npy" >> $screenOut || error_exit "Solution differs from reference"
