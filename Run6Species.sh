#!/bin/bash

error_exit()
{
  echo "$1" 1>&2
  exit 1
}
# module load python/3.10.8
# EXE="python3 ../chebSolver.py"
# NEWTEXE="python3 ../timePeriodicSolver.py"

EXE="python3 ./chebSolver.py --use_gpu 0 --gpu_device_id 0"
NEWTEXE="python3 ./timePeriodicSolver.py --use_gpu 0 --gpu_device_id 0"

# 6 species + 34 rxn - Nominal Rates case
Np=150
# That is for 400 cycles
# Nt=51200
# Nt=128
# That is for 1000 cycles
# Nt=128000
# That is for 2000 cycles
Nt=256000

Nt1=128
dt=0.0078125
scenario=7
baseFile="restart_6spec_CN_Np${Np}_"
newtFile="newton_6spec_CN_Np${Np}.npy"
saveFile="newton_6spec_CN_Np${Np}_fullsoln.npy"

# --EinsteinForm

baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme BE --alpha0 0.1 --increaseFac 1.5 --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt --scenario $scenario --tscheme BE --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"
screenOut="run_6sp.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
# echo "Run 1200 to 1600...${baseFile}T1600.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 1200.0 --restart "${baseFile}T1200.npy" --outfile "discard.npy" >> $screenOut || error_exit "4th run failed"


# echo "Run 0 to 400...${baseFile}T400.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0   0.0 --outfile "${baseFile}T400.npy"  > $screenOut || error_exit "1st run failed"
# $baseCmd --V0 75 --VDC 0.0 --t0   0.0 --restart "${baseFile}T2000.npy" --outfile "${baseFile}T400.npy"  > $screenOut || error_exit "1st run failed"

# echo "Run 400 to 800...${baseFile}T800.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 400.0 --restart "${baseFile}T400.npy" --outfile "${baseFile}T800.npy" >> $screenOut || error_exit "2nd run failed"
# echo "Run 800 to 1200...${baseFile}T1200.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 800.0 --restart "${baseFile}T800.npy" --outfile "${baseFile}T1200.npy" >> $screenOut || error_exit "3rd run failed"
# echo "Run 1200 to 1600...${baseFile}T1600.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 1200.0 --restart "${baseFile}T1200.npy" --outfile "${baseFile}T1600.npy" >> $screenOut || error_exit "4th run failed"
# echo "Run 1600 to 2000...${baseFile}T2000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 1600.0 --restart "${baseFile}T1600.npy" --outfile "${baseFile}T2000.npy" >> $screenOut || error_exit "5th run failed"
# echo "Run 2000 to 2400...${baseFile}T2400.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 2000.0 --restart "${baseFile}T2000.npy" --outfile "${baseFile}T2400.npy" >> $screenOut || error_exit "6th run failed"
# echo "Run 2400 to 2800...${baseFile}T2800.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 2400.0 --restart "${baseFile}T2400.npy" --outfile "${baseFile}T2800.npy" >> $screenOut || error_exit "7th run failed"
# echo "Run 2800 to 3200...${baseFile}T3200.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 2800.0 --restart "${baseFile}T2800.npy" --outfile "${baseFile}T3200.npy" >> $screenOut || error_exit "8th run failed"
# echo "Run 3200 to 3600...${baseFile}T3600.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 3200.0 --restart "${baseFile}T3200.npy" --outfile "${baseFile}T3600.npy" >> $screenOut || error_exit "9th run failed"
# echo "Run 3600 to 4000...${baseFile}T4000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 3600.0 --restart "${baseFile}T3600.npy" --outfile "${baseFile}T4000.npy" >> $screenOut || error_exit "10th run failed"

# echo "Run 4000 to 5000...${baseFile}T5000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 4000.0 --restart "${baseFile}T4000.npy" --outfile "${baseFile}T5000.npy" >> $screenOut || error_exit "1st run failed"
# echo "Run 5000 to 6000...${baseFile}T6000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 5000.0 --restart "${baseFile}T5000.npy" --outfile "${baseFile}T6000.npy" >> $screenOut || error_exit "2nd run failed"
# echo "Run 6000 to 7000...${baseFile}T7000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 6000.0 --restart "${baseFile}T6000.npy" --outfile "${baseFile}T7000.npy" >> $screenOut || error_exit "3rd run failed"
# echo "Run 7000 to 8000...${baseFile}T8000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 7000.0 --restart "${baseFile}T7000.npy" --outfile "${baseFile}T8000.npy" >> $screenOut || error_exit "4th run failed"
# echo "Run 8000 to 9000...${baseFile}T9000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 8000.0 --restart "${baseFile}T8000.npy" --outfile "${baseFile}T9000.npy" >> $screenOut || error_exit "5th run failed"
# echo "Run 9000 to 10000...${baseFile}T10000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 9000.0 --restart "${baseFile}T9000.npy" --outfile "${baseFile}T10000.npy" >> $screenOut || error_exit "6th run failed"

# echo "Run 10000 to 11000...${baseFile}T11000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 10000.0 --restart "${baseFile}T10000.npy" --outfile "${baseFile}T11000.npy" >> $screenOut || error_exit "1th run failed"
# echo "Run 11000 to 12000...${baseFile}T12000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 11000.0 --restart "${baseFile}T11000.npy" --outfile "${baseFile}T12000.npy" >> $screenOut || error_exit "2th run failed"
# echo "Run 12000 to 13000...${baseFile}T13000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 12000.0 --restart "${baseFile}T12000.npy" --outfile "${baseFile}T13000.npy" >> $screenOut || error_exit "3th run failed"
# echo "Run 13000 to 14000...${baseFile}T14000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 13000.0 --restart "${baseFile}T13000.npy" --outfile "${baseFile}T14000.npy" >> $screenOut || error_exit "4th run failed"
# echo "Run 14000 to 15000...${baseFile}T15000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 14000.0 --restart "${baseFile}T14000.npy" --outfile "${baseFile}T15000.npy" >> $screenOut || error_exit "5th run failed"
# echo "Run 15000 to 16000...${baseFile}T16000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 15000.0 --restart "${baseFile}T15000.npy" --outfile "${baseFile}T16000.npy" >> $screenOut || error_exit "6th run failed"


# echo "Run 16000 to 18000...${baseFile}T18000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 16000.0 --restart "${baseFile}T16000.npy" --outfile "${baseFile}T18000.npy" >> $screenOut || error_exit "1th run failed"
# echo "Run 18000 to 20000...${baseFile}T20000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 18000.0 --restart "${baseFile}T18000.npy" --outfile "${baseFile}T20000.npy" >> $screenOut || error_exit "2th run failed"
# echo "Run 20000 to 22000...${baseFile}T22000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 20000.0 --restart "${baseFile}T20000.npy" --outfile "${baseFile}T22000.npy" >> $screenOut || error_exit "3th run failed"
# echo "Run 22000 to 24000...${baseFile}T24000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 22000.0 --restart "${baseFile}T22000.npy" --outfile "${baseFile}T24000.npy" >> $screenOut || error_exit "4th run failed"
# echo "Run 24000 to 26000...${baseFile}T26000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 24000.0 --restart "${baseFile}T24000.npy" --outfile "${baseFile}T26000.npy" >> $screenOut || error_exit "5th run failed"
# echo "Run 26000 to 28000...${baseFile}T28000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 26000.0 --restart "${baseFile}T28000.npy" --outfile "${baseFile}T28000.npy" >> $screenOut || error_exit "6th run failed"
# echo "Run 28000 to 30000...${baseFile}T30000.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 28000.0 --restart "${baseFile}T30000.npy" --outfile "${baseFile}T30000.npy" >> $screenOut || error_exit "7th run failed"






# echo "Saving one period...fullsoln_T400"
# $saveCmd --V0 75 --VDC 0.0 --t0 400.0 --rtol 1e-8 --restart "${baseFile}T400.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T400.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T800.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 800.0 --rtol 1e-8 --restart "${baseFile}T800.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T800.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T1200.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 1200.0 --rtol 1e-8 --restart "${baseFile}T1200.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T1200.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T1600.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 1600.0 --rtol 1e-8 --restart "${baseFile}T1600.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T1600.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T2000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 2000.0 --rtol 1e-8 --restart "${baseFile}T2000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T2000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T2400.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 2400.0 --rtol 1e-8 --restart "${baseFile}T2400.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T2400.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T2800.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 2800.0 --rtol 1e-8 --restart "${baseFile}T2800.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T2800.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T3200.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 3200.0 --rtol 1e-8 --restart "${baseFile}T3200.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T3200.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T3600.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 3600.0 --rtol 1e-8 --restart "${baseFile}T3600.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T3600.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T4000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 4000.0 --rtol 1e-8 --restart "${baseFile}T4000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T4000.npy" --outfile discard.npy >> $screenOut


# echo "Saving one period...fullsoln_T5000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 5000.0 --rtol 1e-8 --restart "${baseFile}T5000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T5000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T6000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 6000.0 --rtol 1e-8 --restart "${baseFile}T6000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T6000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T7000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 7000.0 --rtol 1e-8 --restart "${baseFile}T7000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T7000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T8000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 8000.0 --rtol 1e-8 --restart "${baseFile}T8000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T8000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T9000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 9000.0 --rtol 1e-8 --restart "${baseFile}T9000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T9000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T10000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 10000.0 --rtol 1e-8 --restart "${baseFile}T10000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T10000.npy" --outfile discard.npy >> $screenOut

# echo "Saving one period...fullsoln_T11000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 11000.0 --rtol 1e-8 --restart "${baseFile}T11000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T11000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T12000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 12000.0 --rtol 1e-8 --restart "${baseFile}T12000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T12000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T13000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 13000.0 --rtol 1e-8 --restart "${baseFile}T13000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T13000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T14000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 14000.0 --rtol 1e-8 --restart "${baseFile}T14000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T14000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T15000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 15000.0 --rtol 1e-8 --restart "${baseFile}T15000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T15000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T16000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 16000.0 --rtol 1e-8 --restart "${baseFile}T16000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T16000.npy" --outfile discard.npy >> $screenOut

# echo "Saving one period...fullsoln_T18000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 18000.0 --rtol 1e-8 --restart "${baseFile}T18000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T18000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T20000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 20000.0 --rtol 1e-8 --restart "${baseFile}T20000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T20000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T22000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 22000.0 --rtol 1e-8 --restart "${baseFile}T22000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T22000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T24000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 24000.0 --rtol 1e-8 --restart "${baseFile}T24000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T24000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T26000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 26000.0 --rtol 1e-8 --restart "${baseFile}T26000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T26000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T28000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 28000.0 --rtol 1e-8 --restart "${baseFile}T28000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T28000.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...fullsoln_T30000.npy"
# $saveCmd --V0 75 --VDC 0.0 --t0 30000.0 --rtol 1e-8 --restart "${baseFile}T30000.npy" --savedata "newton_6spec_CN_Np${Np}_fullsoln_T30000.npy" --outfile discard.npy >> $screenOut




# echo "Run time domain shooting...${newtFile}"
# $newtCmd --V0 75 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T10000.npy" \
#                                 --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

echo "Saving one period...${saveFile}"
$saveCmd --V0 75 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut



echo "Bash script has finished!"