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
# dt=0.0078125
dt=0.00390625

# Nt1=128
# dt1=0.0078125
Nt1=256
dt1=0.00390625

scenario=15
# crashedFile="restart_CR_Np150_crashed.npy"
baseFile="restart_CR_BE_Np${Np}_"
newtFile="newton_CR_CN_Np${Np}.npy"
saveFile="newton_CR_CN_Np${Np}_fullsoln.npy"



baseCmd="$EXE --Np $Np --Nt $Nt --dt $dt --scenario $scenario --tscheme BE --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"
newtCmd="$NEWTEXE --Np $Np --Nt $Nt1 --Nn 20 --scenario $scenario --tscheme CN --alpha0 0.05 --increaseFac 1.5 --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"
saveCmd="$EXE --Np $Np --Nt $Nt1 --dt $dt1 --scenario $scenario --tscheme CN --EinsteinForm --elasticCollisionActivation --backgroundSpecieActivation"


screenOut="run_CR.out"
rm -f $screenOut

# Run an example case of the Chebyshev time domain solver
# echo "Run 0 to 250...${baseFile}T250.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 0.0 \
#                --outfile "${baseFile}T250.npy" > $screenOut || error_exit "First run failed"
# echo "Run 250 to 500...${baseFile}T500.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 250.0 --restart "${baseFile}T250.npy"  \
#                --outfile "${baseFile}T500.npy" >> $screenOut || error_exit "Second run failed"
# echo "Run 500 to 750...${baseFile}T750.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 500.0 --restart "${baseFile}T500.npy"  \
#                --outfile "${baseFile}T750.npy" >> $screenOut || error_exit "Third run failed"
# echo "Run 750 to 1000...${baseFile}T1000.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 750.0 --restart "${baseFile}T750.npy"  \
#                --outfile "${baseFile}T1000.npy" >> $screenOut || error_exit "Fourt run failed"
# echo "Run 1000 to 1250...${baseFile}T1250.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 1000.0 --restart "${baseFile}T1000.npy" \
#                --outfile "${baseFile}T1250.npy" >> $screenOut || error_exit "Fifth run failed"
# echo "Run 1250 to 1500...${baseFile}T1500.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 1250.0 --restart "${baseFile}T1250.npy" \
#                --outfile "${baseFile}T1500.npy" >> $screenOut || error_exit "Sixth run failed"
# echo "Run 1500 to 1750...${baseFile}T1750.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 1500.0 --restart "${baseFile}T1500.npy" \
#                --outfile "${baseFile}T1750.npy" >> $screenOut || error_exit "Seventh run failed"
# echo "Run 1750 to 2000...${baseFile}T2000.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 1750.0 --restart "${baseFile}T1750.npy" \
#                --outfile "${baseFile}T2000.npy" >> $screenOut || error_exit "Seventh run failed"
# echo "Run 2000 to 2250...${baseFile}T2250.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 2000.0 --restart "${baseFile}T2000.npy" \
#                --outfile "${baseFile}T2250.npy" >> $screenOut || error_exit "8th run failed"
# echo "Run 2250 to 2500...${baseFile}T2500.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 2250.0 --restart "${baseFile}T2250.npy" \
#                --outfile "${baseFile}T2500.npy" >> $screenOut || error_exit "9th run failed"
# echo "Run 2500 to 2750...${baseFile}T2750.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 2500.0 --restart "${baseFile}T2500.npy" \
#                --outfile "${baseFile}T2750.npy" >> $screenOut || error_exit "10th run failed"




# echo "Saving one period...${saveFile}"
# $saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart "${baseFile}T1750.npy" --savedata "CR_Np${Np}_fullsoln_T1750.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...${saveFile}"
# $saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart "${baseFile}T2250.npy" --savedata "CR_Np${Np}_fullsoln_T2250.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...${saveFile}"
# $saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart "${baseFile}T2500.npy" --savedata "CR_Np${Np}_fullsoln_T2500.npy" --outfile discard.npy >> $screenOut
# echo "Saving one period...${saveFile}"
# $saveCmd --V0 100 --VDC 0.0 --rtol 1e-8 --restart "${baseFile}T2750.npy" --savedata "CR_Np${Np}_fullsoln_T2750.npy" --outfile discard.npy >> $screenOut



# echo "Run 0 to 250...${baseFile}T250.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 0.0 \
              #  --outfile "${baseFile}T250.npy" > $screenOut || error_exit "First run failed"


# echo "Run 2500 to 2750...${baseFile}T2750.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 2500.0 --restart "${baseFile}T2500.npy" \
#                --outfile "${baseFile}T2750.npy" >> $screenOut || error_exit "10th run failed"


# # Run an example case of the Chebyshev time domain solver
# echo "Run 0 to 250...${baseFile}T250.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 0.0 \
#                --outfile "${baseFile}T250.npy" > $screenOut || error_exit "First run failed"


# echo "Run 0 to 1T...restart.npy"
# $baseCmd --V0 100 --VDC 0.0 --t0 0.0 --verbose \
#                --outfile "restart.npy" > $screenOut || error_exit "First run failed"





# echo "Run 0 to 250...${baseFile}T250.npy"
# $baseCmd --V0 75 --VDC 0.0 --t0 0.0 \
#                --outfile "${baseFile}T250.npy" > $screenOut || error_exit "First run failed"

# # echo "Run 250 to 500...${baseFile}T500.npy"
# # $baseCmd --V0 75 --VDC 0.0 --t0 250.0 --restart "${baseFile}T250.npy"  \
# #                --outfile "${baseFile}T500.npy" >> $screenOut || error_exit "Second run failed"

# echo "Run time domain shooting...${newtFile}"
# $newtCmd --V0 75 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T250.npy" \
#                                 --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

# echo "Saving one period...${saveFile}"
# $saveCmd --V0 75 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut
# # $saveCmd --V0 75 --VDC 0.0 --rtol 1e-8 --restart "${baseFile}T500.npy" --savedata $saveFile --outfile discard.npy >> $screenOut






echo "Run 0 to 125...${baseFile}T125.npy"
$baseCmd --V0 75 --VDC 0.0 --t0 0.0 \
               --outfile "${baseFile}T125.npy" > $screenOut || error_exit "First run failed"

echo "Run time domain shooting...${newtFile}"
$newtCmd --V0 75 --VDC 0.0 --gam 0.01 --rtol 1e-8 --restart "${baseFile}T125.npy" \
                                --outfile $newtFile >> $screenOut || error_exit "Shooting failed"

echo "Saving one period...${saveFile}"
$saveCmd --V0 75 --VDC 0.0 --rtol 1e-8 --restart $newtFile --savedata $saveFile --outfile discard.npy >> $screenOut
