#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}

tmp_dir=$(mktemp -d -t tmp-test-XXXXXXXXXX --tmpdir=.)
echo "Running 6 species test in $tmp_dir"
cd $tmp_dir
EXE="python3 ../chebSolver.py"
NEWTEXE="python3 ../timePeriodicSolver.py"

# 6 species + 23 rxn - Nominal Rates case
Np=250
Nt=25600
Nt1=128
dt=0.0078125
scenario=12
baseFile="restart_6spec_CN_Np${Np}_"
newtFile="newton_6spec_CN_Np${Np}.npy"
saveFile="newton_6spec_CN_Np${Np}_fullsoln.npy"

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme CN --alpha0 0.25 --increaseFac 2.0 --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario --tscheme CN --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"

diffCmd="python3 ../diffSolns.py --reference ../reference_solns/Nominal_Np250_solution.npy --Np $Np --Nv 7"


screenOut="run6species.out"
rm -f $screenOut

$newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "../reference_solns/Nominal_Np250_restart.npy" \
                                --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

$diffCmd --solution $newtFile >> $screenOut || error_exit "Solution differs from reference"

# if test passed, delete tmp dir
cd ..
rm -rf $tmp_dir
