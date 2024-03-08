#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}
module load python/3.10.8
EXE="python3 ../chebSolver.py"
NEWTEXE="python3 ../timePeriodicSolver.py"

# 6 species + 34 rxn - Nominal Rates case
Np=150
Nt=51200
Nt1=128
dt=0.0078125
scenario=7
baseFile="restart_6spec_CN_Np${Np}_"
newtFile="newton_6spec_CN_Np${Np}.npy"
saveFile="newton_6spec_CN_Np${Np}_fullsoln.npy"

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --elasticCollisionActivation --backgroundSpecieActivation"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme CN --alpha0 0.1 --increaseFac 1.5 --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario --tscheme CN --elasticCollisionActivation --backgroundSpecieActivation"
screenOut="runCN.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
echo "Run 0 to 400...${baseFile}T400.npy"
$baseCmd --V0 100 --VDC 0.0 --t0   0.0 --outfile "${baseFile}T400.npy"  > $screenOut || error_exit "1st run failed"


echo "Run 400 to 800...${baseFile}T800.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 400.0 --restart "${baseFile}T400.npy" --outfile "${baseFile}T800.npy" >> $screenOut || error_exit "2nd run failed"
echo "Run 800 to 1200...${baseFile}T1200.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 800.0 --restart "${baseFile}T800.npy" --outfile "${baseFile}T1200.npy" >> $screenOut || error_exit "3rd run failed"
echo "Run 1200 to 1600...${baseFile}T1600.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 1200.0 --restart "${baseFile}T1200.npy" --outfile "${baseFile}T1600.npy" >> $screenOut || error_exit "4th run failed"
echo "Run 1600 to 2000...${baseFile}T2000.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 1600.0 --restart "${baseFile}T1600.npy" --outfile "${baseFile}T2000.npy" >> $screenOut || error_exit "6th run failed"
echo "Run 2000 to 2400...${baseFile}T2400.npy"
$baseCmd --V0 100 --VDC 0.0 --t0 2000.0 --restart "${baseFile}T2000.npy" --outfile "${baseFile}T2400.npy" >> $screenOut || error_exit "7th run failed"




echo "Saving one period...fullsoln_T400"
$saveCmd --V0 100 --VDC 0.0 --t0 400.0 --rtol 1e-8 --restart "${baseFile}T400.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T400.npy" --outfile discard.npy >> $screenOut

echo "Saving one period...fullsoln_T800.npy"
$saveCmd --V0 100 --VDC 0.0 --t0 800.0 --rtol 1e-8 --restart "${baseFile}T800.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T800.npy" --outfile discard.npy >> $screenOut

echo "Saving one period...fullsoln_T1200.npy"
$saveCmd --V0 100 --VDC 0.0 --t0 1200.0 --rtol 1e-8 --restart "${baseFile}T1200.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T1200.npy" --outfile discard.npy >> $screenOut

echo "Saving one period...fullsoln_T1600.npy"
$saveCmd --V0 100 --VDC 0.0 --t0 1600.0 --rtol 1e-8 --restart "${baseFile}T1600.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T1600.npy" --outfile discard.npy >> $screenOut

echo "Saving one period...fullsoln_T2000.npy"
$saveCmd --V0 100 --VDC 0.0 --t0 2000.0 --rtol 1e-8 --restart "${baseFile}T2000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T2000.npy" --outfile discard.npy >> $screenOut

echo "Saving one period...fullsoln_T2400.npy"
$saveCmd --V0 100 --VDC 0.0 --t0 2400.0 --rtol 1e-8 --restart "${baseFile}T2400.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T2400.npy" --outfile discard.npy >> $screenOut





# echo "Run time domain shooting...${newtFile}"
# $newtCmd --V0 100 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T400.npy" \
#                                 --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

# echo "Saving one period...${saveFile}"
# $saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut




