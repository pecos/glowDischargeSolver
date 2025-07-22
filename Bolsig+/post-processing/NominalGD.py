import os
import shutil
import numpy as np
import h5py as h5
import subprocess
import pandas as pd
import itertools
from dataclasses import dataclass
from SahaSolver import *
from matplotlib import pyplot as plt
from swarmParameters import bolsigOutput,bolsigEEDFOutput
from input_writer import writeInputFile
# from input_writer import glowDischargeConfigs
from crossSections import multipleCrossSections
from matplotlib.ticker import ScalarFormatter
import scipy.constants as spc

marker = itertools.cycle(('+', 's', 'o', '^', 'v', '*', 'd', 'h'))
colors = ['blue', 'yellow', 'green', 'violet', 'black', 'orange', 'turquoise']


# crsFileName = 'glow-discharge/nominal-crs/BSR.1s.step-excite.txt'
# crsFileName = 'glow-discharge/nominal-crs/LXCat-June2013.txt'
# crsFileName = 'glow-discharge/nominal-crs/crs.nominal.Biagi.txt'
# crsFileName = 'glow-discharge/nominal-crs/fullCrs.txt'
# stepCrsFileName = 'glow-discharge/nominal-crs/BSR_step.txt'
#stepioniCrsFileName = 'glow-discharge/nominal-crs/stepIonization.txt'

# crsFileName = 'glow-discharge/nominal-crs/fullCrs_2.txt'
# crsFileName = 'glow-discharge/nominal-crs/testCrs.txt'
crsFileName = 'glow-discharge/nominal-crs/BSR_CRModel.txt'
# crsFileName = 'glow-discharge/nominal-crs/BSR_CRModel_2.txt'

nominal_crs = multipleCrossSections(crsFileName)
#step_nom_crs = multipleCrossSections(stepCrsFileName)



## Constants
qe = 1.60217663e-19 # C
kB = 1.380649e-23   # J/K
m_e = 9.1e-31       # kg
h = 6.62e-34        # J-s
E_lvl_i = 15.76     # eV
NA = 6.022e23       # #/mol
eV = qe/kB

cm_eV = spc.h*spc.c/spc.e*100  # Convert energy units: from cm^-1 to eV
K_eV = spc.k/spc.e             # Convert energy units: from K to eV


# Excited States Properties
E_lvl_4s = [11.548, 11.624, 11.723, 11.828]
E_lvl_4p = [12.907, 13.075, 13.095, 13.153, 13.171, 13.273, 13.283, 13.302, 13.328, 13.479]
g_4s = [5, 3, 1, 3]
g_4p = [3, 7, 5, 3, 5, 1, 3, 5, 3, 1]

stepioni_g = g_4s + g_4p
stepioni_E_lvl = E_lvl_4s + E_lvl_4p


## Non-Equilibrium Forward Rate Coefficients
## Bolsig conditions for glow discharge
speciesList = 'Ar Ar5 Ar4 Ar3 Ar2 Ar(2p10) Ar(2p9) Ar(2p8) Ar(2p7) Ar(2p6) Ar(2p5) Ar(2p4) Ar(2p3) Ar(2p2) Ar(2p1)'


speciesList_Paschen = 'Ar Ar(1s5) Ar(1s4) Ar(1s3) Ar(1s2) Ar(2p10) Ar(2p9) Ar(2p8) Ar(2p7) Ar(2p6) Ar(2p5) Ar(2p4) Ar(2p3) Ar(2p2) Ar(2p1)' 
speciesList_Racah = "Ar Ar(4s[3/2]2) Ar(4s[3/2]1) Ar(4s'[1/2]0) Ar(4s'[1/2]1) Ar(4p[1/2]1) Ar(4p[5/2]3) Ar(4p[5/2]2) Ar(4p[3/2]1) Ar(4p[3/2]2) Ar(4p[1/2]0) Ar(4p'[3/2]1) Ar(4p'[3/2]2) Ar(4p'[1/2]1) Ar(4p'[1/2]0)"



speciesList = speciesList_Paschen

E_lvl_ev = np.array([ 0.0, 11.54835442, 11.62359272, 11.72316039, 11.82807116, 12.9070153, 13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,  13.28263902, 13.30222747, 13.32785705, 13.47988682])
E_lvl = np.array([0, 93143.76, 93750.5978, 94553.6652, 95399.8276, 104102.099, 105462.7596, 105617.27, 106087.2598, 106237.5518, 107054.272, 107131.7086, 107289.7001, 107496.4166, 108722.6194])
g_lvl = np.array([1, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1])

Te_exp = {}; ne_exp = {}; ni_exp = {}; gi_exp = {}; Ei_exp = {}
ExpCase = '1Torr-150V' # 150V is the tip-to-tip Voltage. In our case V0 would be Vmax = 75 V.
ne_exp[ExpCase] = 2.2e15 # [#/m^3]
ni_exp[ExpCase] = np.array([0.0, 9.59E+15, 3.27E+15, 3.16E+14, 9.58E+14, 2.63E+12, 1.40E+11,  7.72E+11, 
                            5.25E+11, 8.53E+11, 3.47E+11, 3.43E+11, 4.43E+11, 4.64E+11, 6.43E+11])
gi_exp[ExpCase] = np.array([1, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1])   
Ei_exp[ExpCase] =  np.array([ 0.0, 11.54835442, 11.62359272, 11.72316039, 11.82807116, 12.9070153, 13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,  13.28263902, 13.30222747, 13.32785705, 13.47988682])


Torr = 133.3    # Pa
p0 = 1.0 * Torr
T0 = 300          # K 293.15
Te0 = 44000.      # K
qe = 1.60217663e-19 # C
kB = 1.380649e-23 # m2 kg s-2 K-1
# ne0 = 5.0e14      # m-3 3.0e14
ne0 = ne_exp[ExpCase]
#nex0 = 7.0e16     # m-3
# nex0 = np.array([4e16, 2.0e13, 3e16, 1.0e13, 2.0e12, 2.0e12, 2.0e12, 1.5e12, 1.5e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]) # 1Torr - Sim
nex0 = ni_exp[ExpCase][1:] 
nTotal0 = ne0 + (p0 - ne0 * kB * Te0) / kB / T0
ionDeg = ne0 / nTotal0 # approximated assuming small ionization degree
Xex0 = nex0 / nTotal0

print('ne0 = ',ne0, ' [#/m^3]')
print('ionDeg = ',ionDeg, '[-]')



# CONDITIONS
# 0.        / Electric field / N (Td)
# 0.        / Angular field frequency / N (m3/s)
# 0.        / Cosine of E-B field angle
# 300.      / Gas temperature (K)
# 300.      / Excitation temperature (K)
# 0.        / Transition energy (eV)
# 1.0E-10   / Ionization degree 1.0e-8
# 5e14      / Plasma density (1/m3) / Default value = 1e18 / 5e14 / 5e15
# 1.        / Ion charge parameter
# 1.        / Ion/neutral mass ratio
# 1         / e-e momentum effects: 0=No; 1=Yes*
# 1         / Energy sharing: 1=Equal*; 2=One takes all
# 2         / Growth: 1=Temporal*; 2=Spatial; 3=Not included; 4=Grad-n expansion
# 0.0       / Maxwellian mean energy (eV) 
# 200       / # of grid points
# 0         / Manual grid: 0=No; 1=Linear; 2=Parabolic 
# 200.      / Manual maximum energy (eV)
# 1e-10     / Precision
# 1e-5      / Convergence 1e-4
# 10000     / Maximum # of iterations 1000, 10000
# 9.99998E-01 1.24270E-06 6.21350E-10 9.32026E-07 3.10675E-10 6.21350E-11 6.21350E-11 6.21350E-11 4.66013E-11 4.66013E-11 3.10675E-11 3.10675E-11 3.10675E-11 3.10675E-11 3.10675E-11       / Gas composition fractions       
# 1         / Normalize composition to unity: 0=No; 1=Yes

# RUNSERIES
# 2          / Variable: 1=E/N; 2=Mean energy; 3=Maxwellian energy 
# 0.411  6.562  / Min Max 0.04  100.
# 200        / Number  200 
# 3          / Type: 1=Linear; 2=Quadratic; 3=Exponential 3

# SAVERESULTS
# results-test_3.dat        / File 
# 3        / Format: 1=Run by run; 2=Combined; 3=E/N; 4=Energy; 5=SIGLO; 6=PLASIMO
# 1        / Conditions: 0=No; 1=Yes
# 1        / Transport coefficients: 0=No; 1=Yes
# 0        / Rate coefficients: 0=No; 1=Yes
# 0        / Reverse rate coefficients: 0=No; 1=Yes
# 0        / Energy loss coefficients: 0=No; 1=Yes
# 1        / Distribution function: 0=No; 1=Yes 
# 1        / Skip failed runs: 0=No; 1=Yes
# 1        / Include cross sections: 0=No; 1=Yes # This is only at the newest version


# reaction300K = {'READCOLLISIONS': ['"test-crs.txt"', speciesList, 1],
#                 'CONDITIONS': [0., 0., 0., 300., 300., 0., 1.0e-4, 8.0E16, 1., 1., 1, 1, 2, 0., 200, 0, 200., 1.0e-10, 1.0e-5, 10000, '0.99999 1e-5', 1],
#                 'RUNSERIES': [2, 0.75, 30.0, 400, 3],
#                 'SAVERESULTS': ['"reaction_rate.300K.dat"', 1, 1, 1, 0, 0, 0, 1, 1]
#                }


reaction300K = {'READCOLLISIONS': ['"test-crs.txt"', speciesList, 1],
                'CONDITIONS': [0., 0., 0., 300., 300., 0., 1.0e-4, 8.0E16, 1., 1., 1, 1, 2, 0., 200, 0, 200., 1.0e-10, 1.0e-5, 10000, '0.99999 1e-5', 1],
                'RUNSERIES': [2, 3.0, 15.0, 100, 3],
                'SAVERESULTS': ['"reaction_rate.300K.dat"', 1, 1, 1, 0, 0, 0, 1, 1]
               }


bolsigCondition =  [0., 0., 0., T0, T0,  \
                    0., ionDeg, ne0, 1., \
                    1., 1, 1, 2, 0., 400,\
                    0, 300., 1.0e-10,    \
                    1.0e-5, 10000, '%.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E'\
                    %(1.0 - sum(Xex0), Xex0[0], Xex0[1], Xex0[2], Xex0[3], Xex0[4], Xex0[5], Xex0[6], Xex0[7], \
                    Xex0[8], Xex0[9], Xex0[10], Xex0[11], Xex0[12], Xex0[13]), 1]


config = reaction300K
config['CONDITIONS'] = bolsigCondition


# BOLSIG input/output file directories
bolsigDir = '/Users/malamas/Documents/BOLSIG/bolsigplus032016-mac/bolsigminus'
inputFile = 'glow-discharge/CR/1Torr_100V/Transport/input.300K.bolsig.dat'
outputFile = 'glow-discharge/CR/1Torr_100V/Transport/output.300K.bolsig.txt'

transportFile = 'glow-discharge/CR/1Torr_100V/Transport/transport_test_3.h5'
eedfFile = 'glow-discharge/CR/1Torr_100V/EEDF/EEDF_test_3.h5'
# transportFile = 'glow-discharge/CR/1Torr_100V/Transport/transport_BSR_2.h5'
# eedfFile = 'glow-discharge/CR/1Torr_100V/EEDF/EEDF_BSR_2.h5'
# transportFile = 'glow-discharge/CR/1Torr_100V/Transport/transport_BSR_2.h5'
# eedfFile = 'glow-discharge/CR/1Torr_100V/EEDF/EEDF_BSR_2.h5'



if not os.path.exists(outputFile):
    writeInputFile(inputFile, config, crsFileName, outputFile, noscreen = False)
    # command = "./bolsigminus-linux %s" %(inputFile)
    command = bolsigDir + " %s" %(inputFile)
    subprocess.check_call(command, shell = True)


if reaction300K['SAVERESULTS'][1] == 3: 
    nominalOutput = bolsigOutput(outputFile)
elif reaction300K['SAVERESULTS'][1] == 1:
    nominalOutput = bolsigEEDFOutput(outputFile)
       
        
Te = nominalOutput.outputs[3].data[:,1]/1.5*eV

mobility = nominalOutput.outputs[4].data[:,1]
diffusivity = nominalOutput.outputs[5].data[:,1]

energy_mobility = nominalOutput.outputs[6].data[:,1]
energy_diffusivity = nominalOutput.outputs[7].data[:,1]



print('Reading BOLSIG data is done!')


# fig, ax = plt.subplots()
# ax.set_xlabel('$T_e$ [K]')
# ax.set_ylabel('Rate Coefficient [$m^6/s$]')
# ax.set_ylim(1e-43, 4e-39)
# ax.set_xlim(2e2, 1e6)
# secax = ax.secondary_xaxis('top', functions=(KtoeV, eVtoK))
# secax.set_xlabel('$T_e$ [eV]')
# ax.set_title('3-Body Recombination Rate Coefficients')
# plt.loglog(Te, recomb_max[0], linewidth = 2.0, label = '2E + Ar(+) -> E + Ar')
# for i in range(0, len(recomb_step_max)):
#     plt.loglog(Te, recomb_step_max[i], marker = next(marker), markersize = 5.0, markerfacecolor = 'None', linestyle = 'None', label = '{}'.format(recombRxnName[i]))
# plt.legend(loc = 'center right', fontsize = 7)
# plt.tight_layout()
# ax.tick_params(left=True, right=True)
# ax.minorticks_on()
# fig.canvas.draw()
# secax.xaxis.set_major_formatter(formatter)
# plt.savefig('glow-discharge/nominal-rxn/6species/FullModel_5Torr/3BdyRecomb_Excited.png')




with h5.File(transportFile,'w') as f:
    f.attrs['comment'] = 'Transport coefficients from nominal cross sections, evaluated by BOLSIG.'
    data = np.array([Te, mobility]).T
    dset = f.create_dataset('mobility', data=data)
    dset.attrs['name0'] = 'Electron Temperature'
    dset.attrs['name1'] = 'Mobility * N'
    dset.attrs['unit0'] = 'K'
    dset.attrs['unit1'] = '1/m/V/s'

    data = np.array([Te, diffusivity]).T
    dset = f.create_dataset('diffusivity', data=data)
    dset.attrs['name0'] = 'Electron Temperature'
    dset.attrs['name1'] = 'Diffusivity * N'
    dset.attrs['unit0'] = 'K'
    dset.attrs['unit1'] = '1/m/s'

    data = np.array([Te, energy_mobility]).T
    dset = f.create_dataset('energy_mobility', data=data)
    dset.attrs['name0'] = 'Electron Temperature'
    dset.attrs['name1'] = 'Energy Mobility * N'
    dset.attrs['unit0'] = 'K'
    dset.attrs['unit1'] = '1/m/V/s'

    data = np.array([Te, energy_diffusivity]).T
    dset = f.create_dataset('energy_diffusivity', data=data)
    dset.attrs['name0'] = 'Electron Temperature'
    dset.attrs['name1'] = 'Energy Diffusivity * N'
    dset.attrs['unit0'] = 'K'
    dset.attrs['unit1'] = '1/m/s'

import time as cpu_time

if reaction300K['SAVERESULTS'][1] == 1:


    dataType = nominalOutput.typeDictS2I['EEDF (eV-3/2)']
    EEDF_list = nominalOutput.outputs[dataType].data    
     

    # Te_index = np.argmin(np.abs(Te*K_eV - 5.0))
    # print('energy = ', nominalOutput.outputs[3].data[Te_index,1])

    # Te_index2 = np.searchsorted(Te, 5.0/K_eV)
    # print(Te_index, Te_index2)



    # fig, ax = plt.subplots()
    # ax.set_ylabel('Te [eV]')
    # ax.set_xlabel('element index')
    # ax.plot(Te*K_eV)
    # ax.legend()
    
    # eRange = EEDF_list[Te_index][:,0] 
    # EEDF = EEDF_list[Te_index][:,1]
    # EEDF_Maxwellian = 2*np.sqrt(eRange/np.pi)*(Te[Te_index]*K_eV)**(-1.5)*np.exp(-eRange/(Te[Te_index]*K_eV))

    # fig, ax = plt.subplots()
    # ax.set_title('EEDF')
    # ax.set_xlabel('Electron Energy [eV]')
    # ax.set_ylabel('f($\epsilon$)  [eV^-1]')
    # ax.plot(eRange,EEDF*np.sqrt(eRange), label='Bolsig+')
    # ax.plot(eRange, EEDF_Maxwellian,label='Maxwellian')
    # ax.semilogx()
    # ax.legend()
    
    # fig, ax = plt.subplots()
    # ax.set_title('EEDF')
    # ax.set_xlabel('Electron Energy [eV]')
    # ax.set_ylabel('f($\epsilon$)  [eV^-3/2]')
    # ax.plot(eRange,EEDF, label='Bolsig+')
    # ax.plot(eRange, EEDF_Maxwellian/np.sqrt(eRange),label='Maxwellian')
    # ax.semilogy()
    # ax.legend()
    # plt.show()
            
    # print(type(EEDF_list))
    # print(len(EEDF_list))
    # print(type(EEDF_list[1]))
    # print(EEDF_list[1].shape)


    with h5.File(eedfFile,'w') as f:

        f.attrs['comment'] = 'EEDFs from nominal cross sections, evaluated by BOLSIG.'
        data = np.asanyarray(Te)
        dset = f.create_dataset('temperature', data=data)
        dset.attrs['name0'] = 'Electron Temperature'
        dset.attrs['unit0'] = 'K'

        # Save each array in the list as a separate dataset
        for i, arr in enumerate(EEDF_list):
            dset = f.create_dataset(f'EEDF_{i}', data=arr)
            # print(i, f'EEDF_{i}')
            dset.attrs['name0'] = 'Electron Energy'
            dset.attrs['name1'] = 'EEDF'
            dset.attrs['unit0'] = 'eV'
            dset.attrs['unit1'] = 'eV-3/2'            
            dset.attrs['index'] = i
            dset.attrs['Electron Temperature'] = Te[i]


