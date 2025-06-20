import sys
import numpy as np
import scipy.constants as spc
from scipy.ndimage import uniform_filter1d
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import seaborn as sns



# Function to convert centimeters to inches
def cm_to_inch(cm):
    return cm / 2.54

#----------------------------------------------------------------------------------

def PartitionFunctionsAnalytical(T_e):
    
    # Calculate partition functions from analytical expressions
    Q_n = (1+12*np.exp(-11.6/T_e))                                  # Electronic partition function of neutral Argon
    Q_i = (4+2*np.exp(-0.178/T_e)+2*np.exp(-13.5/T_e))             # Electronic partition function of ion Argon

    
    return Q_n,Q_i

def BoltzmannDistribution(n_tot,T_e,Qtot,E_lvl,g_lvl):

    q = len(E_lvl)
    # Excited level populations
    npop_LTE = np.zeros(q)
    for i in range(q):
        if i == 0:
            npop_LTE[i] = g_lvl[i]/Qtot*n_tot
            continue    
        npop_LTE[i] = g_lvl[i]*np.exp(-E_lvl[i]/T_e)/Qtot*n_tot    
       
    return npop_LTE
              
def CalcBoltzmannDistribution(ic0, Te0 , model, npop,dEps, g, i_mid):


   if model[ic0] == "CR":
      Ns = 17 # electrons + ions + 4 4s levels + 10 4p levels + background state
   elif model[ic0] == "CR2":
      Ns = 33 # electrons + ions + 4 4s levels + 10 4p levels + background state

   Ntot = 0; 
   for isp in range(Ns-2):
      # print(np.mean(npop[ic0][i_mid,isp,:],axis=0))
      Ntot = Ntot + np.mean(npop[ic0][i_mid,isp,:],axis=0)
      
   # print("Te0 = ",Te0)
   Q_n,Q_i = PartitionFunctionsAnalytical(Te0)
   npop_LTE = BoltzmannDistribution(Ntot,Te0,Q_n,dEps[ic0][0:-2],g[ic0][0:-2])



   return npop_LTE

#----------------------------------------------------------------------------------
                     
def GetOpCondName(Torr, Voltage, decimal_point=3):
      operatingConditionName = str(round(Torr,decimal_point)) + "Torr-" + str(round(Voltage)) + "V"
      return operatingConditionName

def ReadLumpedAr4pData():

   # Read experimental data.
   Ar_Exp_gi = np.array([6, 6, 36])   
   Ar_Exp_Ei =  np.array([11.577, 11.725, 13.168])

   pathToData = "../ExperimentalData/Ar4p_data/"
   fileName = pathToData + "Ar(4p) Paper Data.xlsx"
   Exp_Data_m = pd.read_excel(fileName, header=None, skiprows=3)
 
 
   num_columns = Exp_Data_m.shape[1]
   num_rows = Exp_Data_m.shape[0]


   Ar_Exp = {}
   Ar_Exp["Ei"] = Ar_Exp_Ei; Ar_Exp["gi"] = Ar_Exp_gi

   for ExpID in range(num_rows):   

      OpCond = GetOpCondName(Exp_Data_m.iloc[ExpID,2], Exp_Data_m.iloc[ExpID,1], decimal_point=1)

      Ar_Exp_ni_m = Exp_Data_m.iloc[ExpID,13]
      Ar_Exp_ni_95 = Exp_Data_m.iloc[ExpID,15]
      Ar_Exp_ni_5 = Exp_Data_m.iloc[ExpID,14]

      Ar_Exp_ni = np.array([Ar_Exp_ni_m, Ar_Exp_ni_95, Ar_Exp_ni_5]) 
      Ar_Exp[OpCond] = Ar_Exp_ni

   return Ar_Exp

def ReadOESAr4pData():


   # Read experimental data.
   Ar_Exp_gi = np.array([3, 7, 5, 3, 5, 1, 3, 5, 3, 1])   
   Ar_Exp_Ei =  np.array([12.9070153, 13.07571571, 13.09487256, 13.15314387, 13.1717777,  
                          13.2730381,  13.28263902, 13.30222747, 13.32785705, 13.47988682])

   pathToData = "../ExperimentalData/Ar4p_data/"
   fileName = pathToData + "populationBayesianResult_Median.csv"
   Exp_Data_m = pd.read_csv(fileName, header=None)
 
   fileName = pathToData + "populationBayesianResult_95percentile.csv"
   Exp_Data_95 = pd.read_csv(fileName, header=None) 

   fileName = pathToData + "populationBayesianResult_5percentile.csv"
   Exp_Data_5 = pd.read_csv(fileName, header=None)
   
   num_columns = Exp_Data_m.shape[1]
   num_rows = Exp_Data_m.shape[0]

   Ar_Exp = {}
   Ar_Exp["Ei"] = Ar_Exp_Ei; Ar_Exp["gi"] = Ar_Exp_gi

   for ExpID in range(num_columns):   

      OpCond = GetOpCondName(Exp_Data_m.iloc[1,ExpID], Exp_Data_m.iloc[0,ExpID])
      # print(OpCond)

      Ar4p_Exp = Exp_Data_m.iloc[2:12,ExpID].to_numpy('float64'); Ar4p_Exp = Ar4p_Exp[::-1]
      Ar_Exp_ni_m = Ar4p_Exp

      Ar4p_Exp = Exp_Data_95.iloc[2:12,ExpID].to_numpy('float64'); Ar4p_Exp = Ar4p_Exp[::-1]
      Ar_Exp_ni_95 = Ar4p_Exp

      Ar4p_Exp = Exp_Data_5.iloc[2:12,ExpID].to_numpy('float64'); Ar4p_Exp = Ar4p_Exp[::-1]
      Ar_Exp_ni_5 = Ar4p_Exp 

      Ar_Exp_ni = np.empty((len(Ar_Exp_ni_m),3)) 
      Ar_Exp_ni[:,0] = Ar_Exp_ni_m
      Ar_Exp_ni[:,1] = Ar_Exp_ni_95
      Ar_Exp_ni[:,2] = Ar_Exp_ni_5

      Ar_Exp[OpCond] = Ar_Exp_ni



   # fig,ax = plt.subplots(dpi=160)
   # for ic in Ar_Exp:
   #    if ic != "Ei" and ic != "gi":
   #       # if ic == "0.5Torr-100V" or ic == "1.0Torr-100V" or ic == "5.0Torr-100V" :
   #       if ic == "0.1Torr-150V" or ic == "0.5Torr-150V" or ic == "1.0Torr-150V" or ic == "5.0Torr-150V" or ic == "10.0Torr-150V" :
   #       # if ic == "0.1Torr-300V" or ic == "0.5Torr-300V" or ic == "1.0Torr-300V" or ic == "5.0Torr-300V" or ic == "10.0Torr-300V":
   #       # if ic == "0.1Torr-500V" or ic == "0.5Torr-500V" :
   #       # if ic == "0.1Torr-750V" or ic == "0.5Torr-750V" :
   #       # if ic == "0.1Torr-150V" or ic == "0.1Torr-300V" or ic == "0.1Torr-500V" or ic == "0.1Torr-750V" or ic == "0.1Torr-1000V" :
   #       # if ic == "0.5Torr-100V" or ic == "0.5Torr-150V" or ic == "0.5Torr-300V" or ic == "0.5Torr-500V" or ic == "0.5Torr-750V":
   #       # if ic == "1.0Torr-100V" or ic == "1.0Torr-150V" or ic == "1.0Torr-300V" or ic == "1.0Torr-400V" :
   #       # if ic == "5.0Torr-100V" or ic == "5.0Torr-150V" or ic == "5.0Torr-300V" :
   #       # if ic == "10.0Torr-150V" or ic == "10.0Torr-300V" :
   #       # if ic == "5.0Torr-300V" or ic == "1.0Torr-300V" :
   #          Ar_Exp_ni = Ar_Exp[ic]         
   #          ls = ':'
   #          uplims =  Ar_Exp_ni[:,1]/Ar_Exp_gi
   #          lolims =  Ar_Exp_ni[:,2]/Ar_Exp_gi
   #          # plt.errorbar(Ar_Exp_Ei, Ar_Exp_ni[:,0]/Ar_Exp_gi, 
   #          #             yerr=(lolims, uplims), marker='+' ,linestyle=ls, lw=1.5, label=ic)
   #          plt.errorbar(Ar_Exp_Ei, Ar_Exp_ni[:,0]/Ar_Exp_gi, 
   #                       marker='*', linestyle=ls, lw=1.5, label=ic)
   # ax.legend(fontsize=12,loc=2)
   # # ax.loglog()
   # ax.semilogy()
   # ax.set_xlim((Ar_Exp_Ei[0]-0.1, Ar_Exp_Ei[-1]+0.1))
   # ax.set_xlabel(r"$E$ [eV]", fontsize=16)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # # ax.set_ylim((0,5.5))
   # ax.set_ylabel(r"$n_i / g_i$ [m$^{-3}$]", fontsize=16)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.show()
   # exit(-1)


   return Ar_Exp

def ReadLangmuirData():


   pathToData = "../ExperimentalData/Langmuir/"
   fileName = pathToData + "Annual Review Data - Langmuir.xlsx"
   Exp_Data_m = pd.read_excel(fileName, header=None, skiprows=5)

   num_columns = Exp_Data_m.shape[1]
   num_rows = Exp_Data_m.shape[0]

   Langmuir_Exp = {}

   for ExpID in range(num_rows):   

      OpCond = GetOpCondName(Exp_Data_m.iloc[ExpID,0], Exp_Data_m.iloc[ExpID,4])
      # print(OpCond)

      Ne_Exp = Exp_Data_m.iloc[ExpID,13]
      Te_Exp = Exp_Data_m.iloc[ExpID,15]

      Langmuir_Exp[OpCond] = np.array([Ne_Exp, Te_Exp])
      

   return Langmuir_Exp

def ReadLASAr4sData():

   Ar_Exp_gi = np.array([5, 3, 1, 3])   
   Ar_Exp_Ei =  np.array([11.54835442, 11.62359272, 11.72316039, 11.82807116])

   pathToData = "../ExperimentalData/"
   fileName = pathToData + "1Torr-150V.xlsx"
   Exp_Data_m = pd.read_excel(fileName, header=None, skiprows=4)
   # num_columns = Exp_Data_m.shape[1]
   # num_rows = Exp_Data_m.shape[0]
   # OpCond = GetOpCondName(Exp_Data_m.iloc[ExpID,0], Exp_Data_m.iloc[ExpID,4])
   OpCond = "1.0Torr-150V" # Do not change!

   Ar4s_Exp = Exp_Data_m.iloc[0:4,2].to_numpy('float64') ; Ar4s_Exp = Ar4s_Exp[::-1]

   LES_Exp = {}
   LES_Exp[OpCond] = Ar4s_Exp
   LES_Exp["Ei"] = Ar_Exp_Ei; LES_Exp["gi"] = Ar_Exp_gi   

   return LES_Exp

#----------------------------------------------------------------------------------

# sns.set_palette("colorblind")  # Use Seaborn's colorblind-friendly palette

# sys.path.insert(0, '../')
# Constants
K_eV = spc.k/spc.e             # Convert energy units: from K to eV
Eion = 15.7596119 # ionization energies of Ar in [eV]

# Flags
isPlotLines = False
isPlotMeans = True


## Species energies and degeneracies
# E, AR+, AR(m), AR(r), AR(4p), AR
dEps_6sp = np.array([0.0,15.76,11.577,11.725,13.168,0.0]) 
g_6sp = np.array([1, 4, 6, 6, 36, 1])

# E, Ar+, Ar+2, Ar2, Ar(m), Ar(r), Ar(4p), Ar     
dEps_8sp = np.array([0.0, 15.76, 14.501, 11.564763, 11.577, 11.725, 13.168, 0.0]) # I need to check these values again
g_8sp = np.array([1, 4, 1, 1, 6, 6, 36, 1]) # I need to check these values again


# electrons + ions + 4 4s levels + 10 4p levels + background state
dEps_CR = np.array([ 0.0,         15.7596119,  11.54835442, 11.62359272, 11.72316039, 11.82807116,
                     12.9070153,  13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,
                     13.28263902, 13.30222747, 13.32785705, 13.47988682,  0.0]) 
g_CR = np.array([1, 4, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1, 1])


dEps_CR3 = np.array([ 0.0,         15.7596119, 14.501, 11.564763,  11.54835442, 11.62359272, 11.72316039, 11.82807116,
                     12.9070153,  13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,
                     13.28263902, 13.30222747, 13.32785705, 13.47988682,  0.0]) 
g_CR3 = np.array([1, 4, 1, 1, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1, 1])



dEps_CR2 = np.array([ 0.0, 15.7596119, 11.54835442, 11.62359272, 11.72316039, 11.82807116, 12.9070153,
                     13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,  13.28263902,
                     13.30222747, 13.32785705, 13.47988682, 13.84503846, 13.86366857, 13.90345461,
                     13.97923734, 14.01273812, 14.06302723, 14.06829767, 14.0899685,  14.09905592,
                     14.15251505, 14.2136715,  14.23402264, 14.23610607, 14.24102775, 14.25508557,
                     14.30366841,  0.0])

g_CR2 = np.array([1, 4, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1, 
                 1, 3, 5, 9, 7, 5, 5, 3, 7, 3, 5, 5, 7, 1, 3, 3, 1])



dEps_CR4 = np.array([ 0.0, 15.7596119, 14.501, 11.564763, 11.54835442, 11.62359272, 11.72316039, 11.82807116, 12.9070153,
                     13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,  13.28263902,
                     13.30222747, 13.32785705, 13.47988682, 13.84503846, 13.86366857, 13.90345461,
                     13.97923734, 14.01273812, 14.06302723, 14.06829767, 14.0899685,  14.09905592,
                     14.15251505, 14.2136715,  14.23402264, 14.23610607, 14.24102775, 14.25508557,
                     14.30366841,  0.0])

g_CR4 = np.array([1, 4, 1, 1, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1, 
                 1, 3, 5, 9, 7, 5, 5, 3, 7, 3, 5, 5, 7, 1, 3, 3, 1])



#----------------------------------------------------------------------------------
# Read experimental data.
Ar_OES_Exp = ReadOESAr4pData()
Ar_Langmuir_Exp = ReadLangmuirData()
Ar_LAS_Exp = ReadLASAr4sData()
Ar_Lumped4p_Exp  = ReadLumpedAr4pData()

#Skata1
# ExpCase = "0.5Torr-150V"
# ExpCase = "0.5Torr-300V"
ExpCase = "1.0Torr-150V"
# ExpCase = "1.0Torr-300V"
# ExpCase = "5.0Torr-150V"
# ExpCase = "5.0Torr-300V"

Ar_OES_Exp_Ei = Ar_OES_Exp["Ei"]; Ar_OES_Exp_gi = Ar_OES_Exp["gi"];
if ExpCase in Ar_OES_Exp.keys():
   Ar_OES_Exp_ni = Ar_OES_Exp[ExpCase]
else:
   Ar_OES_Exp_ni = np.full((np.shape(Ar_OES_Exp_Ei)[0],3), np.nan)


if ExpCase in Ar_Langmuir_Exp.keys():
   Ar_Exp_Ne = Ar_Langmuir_Exp[ExpCase][0] 
   Ar_Exp_Te = Ar_Langmuir_Exp[ExpCase][1]
else:
   Ar_Exp_Ne = np.nan; Ar_Exp_Te = np.nan


Ar_LAS_Exp_Ei = Ar_LAS_Exp["Ei"]; Ar_LAS_Exp_gi = Ar_LAS_Exp["gi"];
if ExpCase in Ar_LAS_Exp.keys():
   Ar_LAS_Exp_ni = Ar_LAS_Exp[ExpCase]
else:
   Ar_LAS_Exp_ni = np.full(np.shape(Ar_LAS_Exp_Ei)[0], np.nan)


Ar_Lumped4p_Exp_Ei = Ar_Lumped4p_Exp["Ei"]; Ar_Lumped4p_Exp_gi = Ar_Lumped4p_Exp["gi"];
if ExpCase in Ar_Lumped4p_Exp.keys():
   Ar_Lumped4p_Exp_ni = Ar_Lumped4p_Exp[ExpCase]
else:
   Ar_Lumped4p_Exp_ni = np.full(3, np.nan)




# Te_exp = {}; ne_exp = {}; ni_exp = {}; gi_exp = {}; Ei_exp = {}
# # AR(m), AR(r), AR(4p)
# ni_lumped_exp = {}; Eps_lumped_exp = np.array([11.577,11.725,13.168]); gi_lumped_exp = np.array([6, 6, 36])

# ExpCase = '1Torr-150V' # 150V is the tip-to-tip Voltage. In our case V0 would be Vmax = 75 V.
# Te_exp[ExpCase] = 7.01 # [eV]
# ne_exp[ExpCase] = 2.2e15 # [#/m^3]

# ni_exp[ExpCase] = np.array([0.0, 9.59E+15, 3.27E+15, 3.16E+14, 9.58E+14, 2.63E+12, 1.40E+11,  7.72E+11, 
#                             5.25E+11, 8.53E+11, 3.47E+11, 3.43E+11, 4.43E+11, 4.64E+11, 6.43E+11])
# gi_exp[ExpCase] = np.array([1, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1])   
# Ei_exp[ExpCase] =  np.array([ 0.0, 11.54835442, 11.62359272, 11.72316039, 11.82807116, 12.9070153, 13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,  13.28263902, 13.30222747, 13.32785705, 13.47988682])
# ni_lumped_exp[ExpCase] = [ ni_exp[ExpCase][1]+ni_exp[ExpCase][3], ni_exp[ExpCase][2]+ni_exp[ExpCase][4], np.sum(ni_exp[ExpCase][5:]) ]

# ni_exp[ExpCase][5:] = np.nan


# ExpCase = '1Torr-200V' # 150V is the tip-to-tip Voltage. 
# Te_exp[ExpCase] = np.nan # [eV]
# ne_exp[ExpCase] = np.nan # [#/m^3]
# ni_exp[ExpCase] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 3.07E+12, 1.12E+12, 1.3E+12, 1.42E+12, 1.33E+12,1.86E+12, 7.62E+11, 7.85E+11, 7.95E+11, 4.35E+12])
# gi_exp[ExpCase] = gi_exp['1Torr-150V']; Ei_exp[ExpCase] =  Ei_exp['1Torr-150V']
# ni_lumped_exp[ExpCase] = [ ni_exp[ExpCase][1]+ni_exp[ExpCase][3], ni_exp[ExpCase][2]+ni_exp[ExpCase][4], np.sum(ni_exp[ExpCase][5:]) ]



# Cases
case = {}; file = {}; clr = {}; label = {}; model = {}


# ic = 11; c = True; f = '../Results/CR/1Torr75V/new/1Torr_75V_Np150_BolsigEEDF_Einstein_Qrad_Tg_1TeBC_Biagi/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Biagi old"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 1; c = True; f = '../Results/CR/1Torr75V/new/1Torr_75V_Np150_BolsigEEDF_ConstDiff_Qrad_Tg_1TeBC_Biagi_gam01/newton_CR_BE_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ns = 17"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 2; c = True; f = '../Results/CR/1Torr75V/new/1Torr_75V_Np150_BolsigEEDF_ConstDiff_Qrad_Tg_1TeBC_Biagi_33_gam01/newton_CR_BE_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ns = 33"; m = "CR2"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 3; c = True; f = '../Results/6spec/1torr_75V_Np150/periodic/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - old"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 



# ic = 3; c = True; f = '../Results/CR/1Torr75V/Ns17_ConstDiff_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "const De"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 4; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 5; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_Biagi/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Biagi"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 6; c = False; f = '../Results/CR/1Torr150V/Ns17_Ein_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "1Torr-150V"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 7; c = True; f = '../Results/CR/100mTorr150V/Ns17_Ein_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "100mTorr-150V"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 8; c = True; f = '../Results/CR/500mTorr150V/Ns17_Ein_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "500mTorr-150V"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 8; c = True; f = '../Results/CR/100mTorr150V/Ns17_MaxEEDF_ConstDiff_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "MaxEEDF - ConstDiff"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 9; c = True; f = '../Results/CR/5Torr75V/Ns17_MaxEEDF_ConstDiff_BSR/newton_CR_BE_Np150_fullsoln.npy'; cl = 'c-'; lb = "5Torr-150V"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 10; c = True; f = '../Results/CR/1Torr75V/Ns33_Ein_Biagi/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Biagi - Ns33"; m = "CR2"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 11; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_LessReactions/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "BSR - no aa react"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 12; c = True; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR/newton_6sp_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 13; c = True; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR_Np300/newton_6sp_CN_Np300_fullsoln.npy'; cl = 'c-'; lb = "6sp - Np = 300"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 14; c = True; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR_LessReactions/newton_6sp_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - less reac."; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 15; c = True; f = '../Results/CR/5Torr75V/Ns17_MaxEEDF_ConstDiff_BSR_LessReactions/newton_CR_BE_Np150_fullsoln.npy'; cl = 'c-'; lb = "5Torr-150V - no aa react."; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 16; c = True; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR_NoAAReactions/newton_6sp_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - no aa reac."; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 17; c = True; f = '../Results/CR/5Torr150V/Ns17_MaxEEDF_ConstDiff_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "5Torr-150V - Max"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 18; c = True; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR_No1stReaction/newton_6sp_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - no 1st reac."; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 19; c = True; f = '../Results/CR/5Torr75V/Ns33_MaxEEDF_ConstDiff_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "5Torr-150V"; m = "CR2"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 20; c = True; f = '../Results/CR/2.5Torr75V/Ns17_MaxEEDF_ConstDiff_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "2.5Torr-75V"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 21; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_AAReactions/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "with aa react"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 22; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_WallLosses/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Wall Losses"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 



# ic = 23; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_MaxEEDF_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR - MaxEEDF"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 24; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_2/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 25; c = True; f = '../Results/CR/100mTorr150V/Ns17_MaxEEDF_ConstDiff_BSR_gam0.3/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "MaxEEDF - ConstDiff - gam0.3"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 26; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_3/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR - 3"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 27; c = True; f = '../Results/CR/5Torr150V/Ns33_MaxEEDF_ConstDiff_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "5Torr-150V - Ns33"; m = "CR2"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 28; c = True; f = '../Results/CR/100mTorr150V/Ns17_Ein_BSR_2/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "100mTorr-150V - 2"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 29; c = True; f = '../Results/CR/500mTorr150V/Ns33_Ein_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "500mTorr-150V - Ns=33"; m = "CR2"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 30; c = True; f = '../Results/CR/5Torr150V/Ns17_Ein_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "5Torr-150V"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 31; c = True; f = '../Results/CR/2.5Torr150V/Ns17_MaxEEDF_Ein_BSR/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "2.5Torr-150V - MaxEEDF"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 32; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_33/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR - 3"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 33; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_44/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR - 4"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 34; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_Final/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "$nD_m = 2.42 \cdot 10^{18}$"; m = "CR"
ic = 34; c = False; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_Final/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Const $\mu_i$"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 35; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_Ion/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "$\mu_i = 8e18$"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 36; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_IST/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR - IST"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 37; c = True; f = '../Results/CR/1Torr75V/Ns17_ConstDiff_BSR_2/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Const - Diff"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 38; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "$nD_m = 4.38 \cdot 10^{18}$"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 39; c = False; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_VarIon/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Var $\mu_i$"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 40; c = False; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_VarIon_NAtoms/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "NAtoms"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 41; c = True; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_VarIon_Ionization/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ionization"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 



# ic = 42; c = True; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR_VarIon_StrongBC_gamma0.1/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ee Flux BC"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 43; c = True; f = '../Results/6spec/1Torr150V/6sp_Ein_BSR_VarIon_StrongBC/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "150V"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 43; c = True; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR_VarIon_StrongBC_gamma0.15/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Strong BC"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 44; c = False; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR_VarIon_OriginalBC_gamma0.15/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Original BC"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 45; c = False; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_Final_2/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 46; c = True; f = '../Results/6spec/1Torr75V/6sp_Ein_BSR_constNg/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Const Ng"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


ic = 47; c = False; f = '../Results/CR/1Torr75V/Ns17_Ein_BSR_DetailedBalance/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ns = 17"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 48; c = False; f = '../Results/CR/1Torr75V/Ns33_Ein_BSR_VarIon/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ns = 33"; m = "CR2"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


ic = 51; c = False; f = '../Results/CR/1Torr75V/Final/Ns17/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ns = 17"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 54; c = False; f = '../Results/CR/1Torr75V/Final/Ns17_NoAAExcRxn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "no aa exc rxn"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 52; c = True; f = '../Results/CR/1Torr75V/Final/Ns17_NoAARxn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "no aa rxn"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 53; c = True; f = '../Results/CR/1Torr75V/Final/Ns17_NoAAIonRxn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "no aa ion rxn"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 55; c = True; f = '../Results/CR/1Torr75V/Final/Ns17_NoAARadRecombRxn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "no rad recombrxn"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


ic = 55; c = False; f = '../Results/CR/2.5Torr75V/Final/Ns17/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "2.5Torr-75V"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 



ic = 56; c = False; f = '../Results/CR/1Torr150V/Ns17/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "1Torr-150V"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


ic = 57; c = False; f = '../Results/CR/500mTorr75V/Ns17/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "0.5Torr-75V"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 58; c = False; f = '../Results/CR/500mTorr150V/Ns17/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "0.5Torr-150V"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 59; c = False; f = '../Results/CR/2.5Torr150V/Ns17/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "2.5Torr-150V"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 60; c = False; f = '../Results/CR/5Torr75V/Final/Ns17_MaxEEDF/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "5Torr-75V"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 




ic = 61; c = False; f = '../Results/CR/1Torr75V/Final/6spec/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6spec - no elastic"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 




ic = 62; c = False; f = '../Results/CR/1Torr75V/Final/Ns17_Ion/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ion 2"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 63; c = False; f = '../Results/CR/1Torr75V/Final/Ns33/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Ns = 33"; m = "CR2"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


ic = 64; c = False; f = '../Results/CR/1Torr75V/Final/6spec_Eeff/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - Eef - 1ev"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 65; c = False; f = '../Results/CR/1Torr75V/Final/6spec_Eeff/Teb05eV/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - Eef - 0.5ev"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 66; c = True; f = '../newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp 1Torr-62V"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 67; c = False; f = '../Results/CR/1Torr75V/Final/6spec_DCbias10/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "DCbias = 5% Vpp"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 69; c = False; f = '../Results/CR/1Torr75V/Final/6spec_t1/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - 0.5ev (new)"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 70; c = False; f = '../Results/CR/1Torr75V/Final/6spec_t2/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - Eef - 0.5ev (new)"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 71; c = False; f = '../Results/CR/1Torr75V/Final/6spec_t1_1ev/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - 1eV (new)"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 72; c = False; f = '../Results/CR/1Torr75V/Final/6spec_StBCIon/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - Eef - 0.5ev - StBCIon"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


ic = 73; c = False; f = '../Results/CR/1Torr75V/Final/6spec_Const_mue/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "6sp - Eef - 0.5ev - ct_mue"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 74; c = False; f = '../Results/CR/1Torr75V/Final/8spec_Const_mue/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "8sp - Eef - 0.5ev - ct_mue"; m = "8sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 75; c = True; f = '../Results/CR/1Torr75V/Final/8spec_StEeBC/newton_8spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "8sp - Eef - StEeBC"; m = "8sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 76; c = True; f = '../Results/CR/1Torr75V/Final/8spec/newton_8spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "8sp - Eef - 0.5ev - StBCIon"; m = "8sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 77; c = False; f = '../Results/CR/1Torr75V/Final/Ns19_StBC_2/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR - Eef - StBC"; m = "CR3"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 78; c = False; f = '../Results/CR/1Torr75V/Final/8spec_StEeBC_2/newton_8spec_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "8sp - Eef - StEeBC - 2"; m = "8sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 79; c = True; f = '../Results/CR/1Torr75V/Final/8spec_StEeBC_3/newton_8spec_CN_Np150_fullsoln.npy'; cl = 'c'; lb = "8sp - Eef - StEeBC - 3"; m = "8sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 80; c = True; f = '../Results/CR/1Torr75V/Final/Ns19_StBC_3/BE/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "Ns = 14"; m = "CR3"
c#ase[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 80; c = False; f = '../Results/CR/1Torr75V/Final/Ns19_StBC_3/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "CR"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 81; c = False; f = '../Results/CR/1Torr75V/Final/Ns19_StBC_4/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Sim - 4"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 82; c = False; f = '../Results/CR/1Torr75V/Final/Ns19_StBC_5/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "Sim - 5"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 81; c = False; f = '../Results/CR/1Torr75V/Final/Ns19_StBC_Rad/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "No radiation trapping"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 82; c = False; f = '../Results/CR/1Torr75V/Final/Ns19_StBC_MaxEEDF/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "Maxwellian"; m = "CR3"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 83; c = False; f = '../Results/CR/1Torr75V/Final/Ns35_StBC/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "Ns = 30"; m = "CR4"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 84; c = True; f = '../Results/CR/1Torr75V/Final/8spec_StBC_NewRates/newton_8spec_CN_Np150_fullsoln.npy'; cl = 'g'; lb = "8sp - Eef - StBC - New Rates"; m = "8sp"
# # ic = 84; c = False; f = '../Results/CR/1Torr75V/Final/8spec_StBC_NewRates/newton_8spec_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "6-species "; m = "8sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# # ic = 85; c = True; f = '../Results/CR/2.5Torr75V/Final/Ns19/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "2.5 Torr - 75 V"; m = "CR3"
# # case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 




## ic = 86; c = True; f = '../Results/CR/L/1Torr75V/Ns19_StBC/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "Bolsig+ old"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
ic = 86; c = True; f = '../Results/CR/L/new/1Torr75V/Ns19_StBC/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "Bolsig+"; m = "CR3"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

## ic = 87; c = True; f = '../Results/CR/L/1Torr75V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'g'; lb = "Druyvesteyn old"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m
# ic = 87; c = True; f = '../Results/CR/L/new/1Torr75V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'g'; lb = "Druyvesteyn"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m


## ic = 88; c = True; f = '../Results/CR/L/1Torr75V/Ns19_StBC_Max/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "Maxwellian old"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 88; c = True; f = '../Results/CR/L/new/1Torr75V/Ns19_StBC_Max/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "Maxwellian"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


## ic = 89; c = True; f = '../Results/CR/L/1Torr75V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "Sim old"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m
# ic = 89; c = True; f = '../Results/CR/L/new/1Torr75V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "Sim"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m

# # ic = 89; c = True; f = '../Results/CR/L/2.5Torr75V/Ns19_StBC/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "2.5 Torr - 75 V"; m = "CR3"
# # case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m # Brakes quasi-neutrality

## ic = 90; c = False; f = '../Results/CR/L/500mTorr75V/Ns19_StBC/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "0.5 Torr - 75 V"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 90; c = False; f = '../Results/CR/L/new/500mTorr75V/Ns19_StBC/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "0.5 Torr - 75 V"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

## ic = 91; c = True; f = '../Results/CR/L/500mTorr75V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "Sim"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 91; c = True; f = '../Results/CR/L/new/500mTorr75V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "Sim"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

## ic = 92; c = True; f = '../Results/CR/L/500mTorr150V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "0.5 Torr - 150 V - Dn"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 92; c = True; f = '../Results/CR/L/new/500mTorr150V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "0.5 Torr - 150 V - Dn"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# # ic = 92; c = True; f = '../Results/CR/L/5Torr75V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "5 Torr - 75 V - Dn"; m = "CR3"
# # case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m  # Brakes quasi-neutrality

## ic = 93; c = False; f = '../Results/CR/L/1Torr150V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "1 Torr - 150 V - Dn"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 93; c = False; f = '../Results/CR/L/new/1Torr150V/Ns19_StBC_Dn/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "1 Torr - 150 V - Dn"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 94; c = True; f = '../Results/CR/L/2.5Torr75V/Ns19_StBC_Max/newton_CR_CN_Np150_fullsoln.npy'; cl = 'b'; lb = "2.5 Torr - 75 V - Max"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 95; c = True; f = '../Results/CR/L/2.5Torr75V/Ns19_StBC_Max_Np300/newton_CR_CN_Np300_fullsoln.npy'; cl = 'b'; lb = "Np = 300"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 96; c = True; f = '../Results/CR/L/2.5Torr75V/Ns19_StBC_Max/newton_CR_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "Np = 150"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

## ic = 97; c = True; f = '../Results/CR/L/5Torr75V/Ns19_StBC_Max_Np300/newton_CR_CN_Np300_fullsoln.npy'; cl = 'b'; lb = "5 Torr - 75 V - Max"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m  
# ic = 97; c = True; f = '../Results/CR/L/new/5Torr75V/Ns19_StBC_Max_Np300/newton_CR_CN_Np300_fullsoln.npy'; cl = 'b'; lb = "5 Torr - 75 V - Max"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m  

## ic = 98; c = True; f = '../Results/CR/L/5Torr150V/Ns19_StBC_Max_Np300/newton_CR_CN_Np300_fullsoln.npy'; cl = 'b'; lb = "5 Torr - 150 V - Max"; m = "CR3"
## case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m  
# ic = 98; c = True; f = '../Results/CR/L/new/5Torr150V/Ns19_StBC_Max_Np300/newton_CR_CN_Np300_fullsoln.npy'; cl = 'b'; lb = "5 Torr - 150 V - Max"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m  


# ic = 100; c = True; f = '../Results/CR/L/new/1Torr75V/Ns19_StBC_test/newton_CR_CN_Np150_fullsoln.npy'; cl = 'g'; lb = "Bolsig+ (new test)"; m = "CR3"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 101; c = True; f = '../Results/CR/L/new/1Torr75V/Ns19_StBC_ecp/newton_CR_CN_Np150_fullsoln.npy'; cl = 'g'; lb = "esp"; m = "CR3"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


#Skata2

# these values are required to "redimensionalize" the results
# they must be consistent with the scenario input file
Pressure  = 2.5*spc.torr               # [Pa] 
GasTemperature = 293.15                 # [K]
nAr = Pressure/GasTemperature/spc.k    # [#/m^3] Number density based on bulk temperature (not necessarily true density in two-temperature gas)

ne0 = 8e16           # "nominal" electron density [1/m^3]
L   = 2.00*0.005     # half-gap-width [m] (gap width is 2 cm)
tau = (1./13.56e6)   # period of driving voltage [s]
# V0  = 100.0          # amplitude of driving voltage [V]
V0  = 75.0          # amplitude of driving voltage [V]
p_0 = Pressure


# Number of Chebyshev modes
Np=150
# Np=300

# T = number of time steps
# T=128
T=256
# T=512

# indices to plot (0, 0.25, 0.5, 0.75)*period
t0 = 0
t1 = 32
t2 = 64
t3 = 96

# time
tp = np.linspace(0,1,T+1)
tr = tp*tau

# spatial grid
xp = -np.cos(np.pi*np.linspace(0,Np-1,Np)/(Np-1))
xr = (xp+1)*L*100 # [cm]

abs_diff = np.abs(xr - 1.0) # Calculate absolute differences between each value and the midpoint
i_mid = np.argmin(abs_diff) # Find the index of the minimum absolute difference



ne = {}; ni = {}; nb = {}; nee = {}; Te = {}; Tg = {}
npop = {}; nm = {}; nr = {}; n4p = {}; nAr2i = {}; nAr2m = {};  
n1s5 = {}; 
dEps = {}; g = {}
FromGlowDischargeToCRIndexing = {}
FromCRToGlowDischargeIndexing = {}
xr = {}; i_mid = {}
Eeff = {}

TotalCurrent = {}; IonCurrent = {}; ElectronCurrent = {}
ElectricField = {}; ElectricPotential = {}; EffElectricField = {}

isReportingCurrents = True

# if case1:
for ic in case: 
   if case[ic]: 
      print("Loading case ",ic, " from file :", file[ic])

      if ic == 95 or ic == 97 or ic == 98:
         Np = 300
      else:
         Np = 150

      # spatial grid
      xp = -np.cos(np.pi*np.linspace(0,Np-1,Np)/(Np-1))
      xr[ic] = (xp+1)*L*100 # [cm]

      abs_diff = np.abs(xr[ic] - 1.0) # Calculate absolute differences between each value and the midpoint
      i_mid[ic] = np.argmin(abs_diff) # Find the index of the minimum absolute difference


      # if ic == 13:
      #    Np=300
         
      #    # spatial grid
      #    xp = -np.cos(np.pi*np.linspace(0,Np-1,Np)/(Np-1))
      #    xr[ic] = (xp+1)*L*100 # [cm]         


                  
      # load solution file
      D = np.load(file[ic].format(Np))
      D = np.transpose(D)


      if (isReportingCurrents):
         fname = ''
         for s in range(len(file[ic].split('/'))-1):
            fname = fname + file[ic].split('/')[s] + '/'

         TotalCurrent[ic] = np.load(fname + "TotalCurrent_" + file[ic].split('/')[-1])
         IonCurrent[ic] = np.load(fname + "IonCurrent_" + file[ic].split('/')[-1])
         ElectronCurrent[ic] = np.load(fname + "ElectronCurrent_" + file[ic].split('/')[-1])

         ElectricField[ic] = np.load(fname + "ElectricField_" + file[ic].split('/')[-1])
         ElectricPotential[ic] = np.load(fname + "ElectricPotential_" + file[ic].split('/')[-1])
         if ic == 64 or ic == 65 or ic >= 70:
            EffElectricField[ic] = np.load(fname + "EffElectricField_" + file[ic].split('/')[-1])


      


      # Ns is the number of scpecies
      if model[ic] == "CR":
         Ns = 17 # electrons + ions + 4 4s levels + 10 4p levels + background state
      elif model[ic] == "CR2":
         Ns = 33 # electrons + ions + 4 4s levels + 10 4p levels + background state
      elif model[ic] == "6sp":
         Ns = 6 #  electrons + ions + nm + nr + n4p + nb 
      elif model[ic] == "8sp":
         Ns = 8 #  electrons + ions + molecular ions + molecular argon + nm + nr + n4p + nb 
      elif model[ic] == "CR3":
         Ns = 19 # electrons + ions + molecular ions + molecular argon + 4 4s levels + 10 4p levels + background state
      elif model[ic] == "CR4":
         Ns = 35 # electrons + ions + molecular ions + molecular argon + 4 4s levels + 10 4p levels + 3d + 5s + background state

   
   

         
      # pull solution out of D
      '''
         ### Indexing
         # GlowDischarge Indexing 
         # i = 0       -> electrons 
         # i = 1       -> ions        
         # i = 2:Ns-1  -> excited levels
         # i = Ns - 1  -> ground state
         # i = Ns      -> electron energy

         # CR Indexing 
         # i = 0       -> ground state
         # i = 1:Ns-2  -> excited levels
         # i = Ns - 2  -> electrons
         # i = Ns - 1  -> ions
         # i = Ns      -> electron energy

      '''


      if model[ic] == "8sp" or model[ic] == "CR3" or model[ic] == "CR4":
         FromCRToGlowDischargeIndexing[ic]  = [Ns-2, Ns-1, Ns-3, Ns-4] + list(range(1,Ns-4)) + [0] # We have excluded electron energy
         # FromGlowDischargeToCRIndexing[ic]  = [Ns-1] + list(range(4,Ns-1)) + [3, 2, 0, 1] # We have excluded electron energy
         FromGlowDischargeToCRIndexing[ic]  = [Ns-1] + list(range(4,Ns-1)) + [0, 1] # We have excluded electron energy
      else:
         FromCRToGlowDischargeIndexing[ic]  = [Ns-2, Ns-1] + list(range(1,Ns-2)) + [0] # We have excluded electron energy
         FromGlowDischargeToCRIndexing[ic]  = [Ns-1] + list(range(2,Ns-1)) + [0, 1] # We have excluded electron energy
         

      if ic == 64 or ic == 65 or ic == 70 or ic > 71:
         D_reshaped = np.reshape(D,(Np, Ns+1+1, np.shape(D)[1]),'F')         
      else:   
         D_reshaped = np.reshape(D,(Np, Ns+1, np.shape(D)[1]),'F')

      # ne  = D[0:1*Np,:]          # electron density
      # ni  = D[1*Np:2*Np,:]       # ion density
      # nm  = D[2*Np:3*Np,:]       # AR(m) density
      # nr  = D[3*Np:4*Np,:]       # AR(r) density
      # n4p = D[4*Np:5*Np,:]       # AR(4p) density
      # nb  = D[(Ns-1)*Np:Ns*Np,:] # "background" (argon neutral) density
      # nee = D[Ns*Np:,:]          # electron energy (ne * ee)
      # flag = np.array_equal(ni, D_reshaped[:,1,:])
      # print(flag)

      ne[ic]  = ne0 * D_reshaped[:,0,:]               # electron density
      ni[ic]  = ne0 * D_reshaped[:,1,:]               # ion density
      nb[ic]  = nAr * D_reshaped[:,Ns - 1,:]          # "background" (argon neutral) density
      nee[ic] = (2./3.) * ne0 * D_reshaped[:,Ns,:]    # electron energy (ne * ee)
      if ic == 64 or ic == 65 or ic > 71:
         Eeff[ic] = V0 / L * D_reshaped[:,Ns+1,:]        # effective electric field for ions 

      npop[ic] = np.ndarray((Np, Ns-2, np.shape(D)[1]),dtype=np.float64)
      npop[ic][:,0,:] = nb[ic]
      npop[ic][:,1:,:] = ne0 * D_reshaped[:,2:Ns-1,:]


      if model[ic] == "CR" or model[ic] == "CR2":
         n1s5[ic] = ne0 * D_reshaped[:,2,:]
         nm[ic] = ne0 * (D_reshaped[:,2,:] + D_reshaped[:,4,:]) 
         nr[ic] = ne0 * (D_reshaped[:,3,:] + D_reshaped[:,5,:]) 
         n4p[ic] = np.zeros_like(nr[ic])
         for i in range(6,15+1):
            n4p[ic] += ne0 * D_reshaped[:,i,:] 
      elif model[ic] == "6sp":
         n1s5[ic] = ne0 * D_reshaped[:,2,:] # Not exactly correct. The reduced order model compines the together the two 4s metastable states.
         nm[ic]  = ne0 * D_reshaped[:,2,:]
         nr[ic]  = ne0 * D_reshaped[:,3,:] 
         n4p[ic] = ne0 * D_reshaped[:,4,:] 
      elif model[ic] == "8sp":
         nAr2i[ic] = ne0 * D_reshaped[:,2,:]
         nAr2m[ic]  = ne0 * D_reshaped[:,3,:]          
         nm[ic]    = ne0 * D_reshaped[:,4,:]
         nr[ic]    = ne0 * D_reshaped[:,5,:] 
         n4p[ic]   = ne0 * D_reshaped[:,6,:] 
         n1s5[ic] = ne0 * D_reshaped[:,4,:] # Not exactly correct. The reduced order model compines the together the two 4s metastable states.


         npop[ic] = np.ndarray((Np, Ns-4, np.shape(D)[1]),dtype=np.float64)
         npop[ic][:,0,:] = nb[ic]
         npop[ic][:,1:,:] = ne0 * D_reshaped[:,4:Ns-1,:]

         # npop[ic][:,1:-2,:] = ne0 * D_reshaped[:,4:Ns-1,:]
         # npop[ic][:,-2,:] = ne0 * D_reshaped[:,2,:]
         # npop[ic][:,-1,:] = ne0 * D_reshaped[:,3,:]

      elif model[ic] == "CR3" or model[ic] == "CR4":
         nAr2i[ic] = ne0 * D_reshaped[:,2,:]
         nAr2m[ic]  = ne0 * D_reshaped[:,3,:]  
         nm[ic] = ne0 * (D_reshaped[:,4,:] + D_reshaped[:,6,:]) 
         nr[ic] = ne0 * (D_reshaped[:,5,:] + D_reshaped[:,7,:]) 
         n1s5[ic] = ne0 * D_reshaped[:,4,:]
         n4p[ic] = np.zeros_like(nr[ic])
         for i in range(8,17+1):
            n4p[ic] += ne0 * D_reshaped[:,i,:] 

         npop[ic] = np.ndarray((Np, Ns-4, np.shape(D)[1]),dtype=np.float64)
         npop[ic][:,0,:] = nb[ic]
         npop[ic][:,1:,:] = ne0 * D_reshaped[:,4:Ns-1,:]

         # npop[ic][:,1:-2,:] = ne0 * D_reshaped[:,4:Ns-1,:]
         # npop[ic][:,-2,:] = ne0 * D_reshaped[:,2,:]
         # npop[ic][:,-1,:] = ne0 * D_reshaped[:,3,:]


      # electron temp
      Te[ic] = nee[ic] / ne[ic]  

      Tg[ic] = (p_0/spc.k - ne[ic] * Te[ic]/K_eV) / (np.sum(npop[ic], axis=1) + ni[ic])   # [K]

      nee[ic] *= spc.e

      if model[ic] == "CR":
         dEps[ic] = dEps_CR[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_CR[FromGlowDischargeToCRIndexing[ic]]

      elif model[ic] == "CR2":
         dEps[ic] = dEps_CR2[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_CR2[FromGlowDischargeToCRIndexing[ic]]
         
      elif model[ic] == "6sp":
         dEps[ic] = dEps_6sp[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_6sp[FromGlowDischargeToCRIndexing[ic]]

      elif model[ic] == "8sp":
         dEps[ic] = dEps_8sp[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_8sp[FromGlowDischargeToCRIndexing[ic]]

      elif model[ic] == "CR3":
         dEps[ic] = dEps_CR3[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_CR3[FromGlowDischargeToCRIndexing[ic]]

      elif model[ic] == "CR4":
         dEps[ic] = dEps_CR4[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_CR4[FromGlowDischargeToCRIndexing[ic]]

      print("ne: {:2E}".format(np.mean(ne[ic], axis=1)[74]))
      print("n4p: {:2E}".format(np.mean(n4p[ic], axis=1)[74]))

      del D, D_reshaped 



# make some plots

'''
# ne
fig,ax = plt.subplots(dpi=160)
h = ax.contourf(xr, tr, np.transpose(ne), levels=np.linspace(0,ne0,129))
cbar = plt.colorbar(h, ticks=np.linspace(0,ne0,9))
cbar.ax.set_ylabel(r"$n_e$ [m$^{-3}$]", labelpad=6, fontsize=18)
plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
ax.contour(xr, tr, np.transpose(ne), levels=np.linspace(0,ne0,129))
ax.set_xlabel(r"$x$ [cm]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel(r"$t$ [s]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('./png/ne_6spec_contour.png')

# Te
fig,ax = plt.subplots(dpi=160)
h = ax.contourf(xr, tr, np.transpose(Te), levels=np.linspace(0,16,129))
cbar = plt.colorbar(h, ticks=np.linspace(0,16,5))
cbar.ax.set_ylabel(r"$T_e$ [eV]", labelpad=6, fontsize=18)
plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
ax.contour(xr, tr, np.transpose(Te), levels=np.linspace(0,16,129))
ax.set_xlabel(r"$x$ [cm]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel(r"$t$ [s]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('Te_6spec_contour.png')
'''

if (isReportingCurrents):
   print("Reporting Currents...")

   for ic in case: 
      if case[ic]: 
         print("Case number: ", ic)
         
         print('TotalCurrent = ', np.mean(TotalCurrent[ic][1:,:],axis=0))
         print('IonCurrent = ', np.mean(IonCurrent[ic][1:,:],axis=0))
         print('ElectronCurrent = ', np.mean(ElectronCurrent[ic][1:,:],axis=0))


   # # TotalCurrent
   # fig,ax = plt.subplots(dpi=160)
   # plt.title("TotalCurrent")
   # for ic in case: 
   #    if case[ic]: 
   #       ax.plot(tr[1:], TotalCurrent[ic][1:,0], lw=2, label=label[ic]+" (x=0)")
   #       ax.plot(tr[1:], TotalCurrent[ic][1:,1], lw=2, label=label[ic]+" (x=L)")
   # ax.set_xlim((tr[0], tr[-1]))
   # ax.legend(fontsize=12)
   # ax.set_xlabel(r"$t$ [s]", fontsize=16)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$I$ [A]", fontsize=16)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/TotalCurrent_electrodes.png')


   # # IonCurrent
   # fig,ax = plt.subplots(dpi=160)
   # plt.title("IonCurrent")
   # for ic in case: 
   #    if case[ic]: 
   #       ax.plot(tr[1:], IonCurrent[ic][1:,0], lw=2, label=label[ic]+" (x=0)")
   #       ax.plot(tr[1:], IonCurrent[ic][1:,1], lw=2, label=label[ic]+" (x=L)")
   # ax.set_xlim((tr[0], tr[-1]))
   # ax.legend(fontsize=12)
   # ax.set_xlabel(r"$t$ [s]", fontsize=16)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$I$ [A]", fontsize=16)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/IonCurrent_electrodes.png')

   # # ElectronCurrent
   # fig,ax = plt.subplots(dpi=160)
   # plt.title("ElectronCurrent")
   # for ic in case: 
   #    if case[ic]: 
   #       ax.plot(tr[1:], ElectronCurrent[ic][1:,0], lw=2, label=label[ic]+" (x=0)")
   #       ax.plot(tr[1:], ElectronCurrent[ic][1:,1], lw=2, label=label[ic]+" (x=L)")
   # ax.set_xlim((tr[0], tr[-1]))
   # ax.legend(fontsize=12)
   # ax.set_xlabel(r"$t$ [s]", fontsize=16)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$I$ [A]", fontsize=16)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/ElectronCurrent_electrodes.png')


   # Electron number density
   fig,ax = plt.subplots(dpi=160)
   plt.title("ne(t)")
   for ic in case: 
      if case[ic]:               
         ax.plot(tr[:], ne[ic][0,:], lw=2, label=label[ic]+" (x=0)")
         ax.plot(tr[:], ne[ic][-1,:], lw=2, label=label[ic]+" (x=L)")
   ax.set_xlim((tr[0], tr[-1]))
   ax.legend(fontsize=12)
   ax.set_xlabel(r"$t$ [s]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e}$ [m$^{-3}$]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/ne_electrodes.png')




   # # Electric Field - Electrodes
   # fig,ax = plt.subplots(dpi=160)
   # plt.title("E(t)")
   # for ic in case: 
   #    if case[ic]:               
   #       ax.plot(tr[1:], ElectricField[ic][1:,0], lw=2, label=label[ic]+" (x=0)")
   #       ax.plot(tr[1:], ElectricField[ic][1:,-1], lw=2, label=label[ic]+" (x=L)")
   # ax.set_xlim((tr[0], tr[-1]))
   # ax.legend(fontsize=12)
   # ax.set_xlabel(r"$t$ [s]", fontsize=16)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$E$ [V/m]", fontsize=16)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/EField_electrodes.png')


   # # Effective Electric Field - Electrodes
   # fig,ax = plt.subplots(dpi=160)
   # plt.title("Eeff(t)")
   # for ic in case: 
   #    if case[ic]:               
   #       ax.plot(tr[1:], EffElectricField[ic][1:,0], lw=2, label=label[ic]+" (x=0)")
   #       ax.plot(tr[1:], EffElectricField[ic][1:,-1], lw=2, label=label[ic]+" (x=L)")
   # ax.set_xlim((tr[0], tr[-1]))
   # ax.legend(fontsize=12)
   # ax.set_xlabel(r"$t$ [s]", fontsize=16)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$E^{ef}$ [V/m]", fontsize=16)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/EffEField_electrodes.png')

   # # Electric Potential - Electrodes
   # fig,ax = plt.subplots(dpi=160)
   # # plt.title("ne(t)")
   # for ic in case: 
   #    if case[ic]:               
   #       ax.plot(tr[1:], ElectricPotential[ic][1:,0], lw=2, label=label[ic]+" (x=0)")
   #       ax.plot(tr[1:], ElectricPotential[ic][1:,-1], lw=2, label=label[ic]+" (x=L)")
   # ax.set_xlim((tr[0], tr[-1]))
   # ax.legend(fontsize=12)
   # ax.set_xlabel(r"$t$ [s]", fontsize=16)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$\phi$ [V]", fontsize=16)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/EPotential_electrodes.png')


   # # Electric Power - Electrodes
   # fig,ax = plt.subplots(dpi=160)
   # # plt.title("ne(t)")
   # for ic in case: 
   #    if case[ic]:               
   #       ax.plot(tr[1:], TotalCurrent[ic][1:,0]*ElectricPotential[ic][1:,0], lw=2, label=label[ic]+" (x=0)")
   #       ax.plot(tr[1:], TotalCurrent[ic][1:,1]*ElectricPotential[ic][1:,-1], lw=2, label=label[ic]+" (x=L)")
   # ax.set_xlim((tr[0], tr[-1]))
   # ax.legend(fontsize=12)
   # ax.set_xlabel(r"$t$ [s]", fontsize=16)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$P$ [W]", fontsize=16)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/EPower_electrodes.png')



   # Mean Electric Field
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(ElectricField[ic],axis=0),lw=2, label=label[ic])
         ax.plot(xr[ic], np.mean(EffElectricField[ic],axis=0),lw=2, label=label[ic]+" - Eff")
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12,loc=2)
   ax.set_xlabel(r"$x$ [cm]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylim((0,8.0))
   ax.set_ylabel(r"$E$ [V/m]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/EField_mean.png')

   # Mean Electric Potential
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(ElectricPotential[ic],axis=0),lw=2, label=label[ic])
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12,loc=2)
   ax.set_xlabel(r"$x$ [cm]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylim((0,8.0))
   ax.set_ylabel(r"$\phi$ [V]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/EPotential_mean.png')




   # Mean Electric Field and Potential
   # Set the desired width in cm
   width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio
   fig,ax1 = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   plt.text(-0.25, 1.10, '(a)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')   
   for ic in case: 
      if case[ic]: 
         ax1.plot(xr[ic], np.mean(ElectricField[ic],axis=0), c='b', lw=1.5, label=r"$E$")
         ax1.set_xlim((xr[ic][0], xr[ic][-1]))

   ax1.set_xlabel(r"$x$ [cm]", fontsize=11)
   plt.setp(ax1.get_xticklabels(), fontsize=9)
   ax1.set_ylabel(r"$E$ [V/m]", fontsize=11)
   plt.setp(ax1.get_yticklabels(), fontsize=9)
   ax1.tick_params(axis='y', labelcolor='b')
   plt.grid(True)
   ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
   ax1.yaxis.get_major_formatter().set_scientific(True)
   ax1.yaxis.get_major_formatter().set_powerlimits((-3, 4))  # Control when scientific notation is used

   ax2 = ax1.twinx()  
   for ic in case: 
      if case[ic]: 
         ax2.plot(xr[ic], np.mean(ElectricPotential[ic],axis=0), c='r', ls='--', lw=1.5, label=r"$\phi$ ")
   ax2.set_ylabel(r"$\phi$ [V]", fontsize=11)
   plt.setp(ax2.get_yticklabels(), fontsize=9)
   ax2.tick_params(axis='y', labelcolor='r')
   # fig.legend(loc='lower center', bbox_to_anchor=(0.5,0.1), fontsize=8)
   fig.legend(loc='center', bbox_to_anchor=(0.53,0.33) ,fontsize=8)
   plt.grid(True)
   plt.tight_layout()
   plt.savefig('./png/EField_mean_a.png', dpi=300, bbox_inches='tight')




   # Mean Electric Field and Potential
   # Set the desired width in cm
   width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio
   fig,ax1 = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   plt.text(-0.25, 1.10, '(b)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')   
   for ic in case: 
      if case[ic]: 
         if ic == 96:
            ax1.plot(xr[ic], np.mean(ElectricPotential[ic],axis=0), c=clr[ic],ls='--', lw=1.5, label=label[ic])
         else:
            ax1.plot(xr[ic], np.mean(ElectricPotential[ic],axis=0), c=clr[ic], lw=1.5, label=label[ic])
         ax1.set_xlim((xr[ic][0], xr[ic][-1]))
   ax1.set_xlabel(r"$x$ [cm]", fontsize=11)
   plt.setp(ax1.get_xticklabels(), fontsize=9)
   ax1.set_ylabel(r"$\phi$ [V]", fontsize=11)
   plt.setp(ax1.get_yticklabels(), fontsize=9)
   # ax1.tick_params(axis='y', labelcolor='b')
   plt.grid(True)
   fig.legend(loc='center', bbox_to_anchor=(0.56,0.33) ,fontsize=8)
   plt.tight_layout()
   plt.savefig('./png/Potential_mean_MaxEEDF.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Potential_mean_Grid.png', dpi=300, bbox_inches='tight')




if (isPlotLines):
   ic = 2
   print("Plotting lines...")
   # ne
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, ne[ic][:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, ne[ic][:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, ne[ic][:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, ne[ic][:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.plot(xr, ni[ic][:,t0], 'b--', lw=2)
   ax.plot(xr, ni[ic][:,t1], 'r--', lw=2)
   ax.plot(xr, ni[ic][:,t2], 'g--', lw=2)
   ax.plot(xr, ni[ic][:,t3], 'c--', lw=2)
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e,i}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/ne_line.png')

   # nm
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, nm[ic][:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, nm[ic][:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, nm[ic][:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, nm[ic][:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(m)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nm_line.png')

   # nr
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, nr[ic][:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, nr[ic][:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, nr[ic][:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, nr[ic][:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(r)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nr_line.png')


   # n4p
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, n4p[ic][:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, n4p[ic][:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, n4p[ic][:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, n4p[ic][:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(4p)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/n4p_line.png')




   # nb
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, nb[ic][:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, nb[ic][:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, nb[ic][:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, nb[ic][:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nb_line.png')

   # Te
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, Te[ic][:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, Te[ic][:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, Te[ic][:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, Te[ic][:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylim((0,10))
   ax.set_ylabel(r"$T_e$ [eV]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Te_line.png')


if (isPlotMeans):
   print("Plotting means...")
   
   # # # ic1 = 34; ic2 = 39 
   # ic1 = 80; ic2 = 84 
   # print("Realative differences [%]:")
   # x1 = (np.mean(ne[ic1],axis=1))[i_mid[ic1]]
   # x2 = (np.mean(ne[ic2],axis=1))[i_mid[ic2]]
   # print("ne = ",np.abs(x1-x2)/x1*100) 
   # print("ne (ratio) = ", max(x1,x2) / min(x1,x2) ) 

   
   # x1 = (np.mean(nm[ic1],axis=1))[i_mid[ic1]]
   # x2 = (np.mean(nm[ic2],axis=1))[i_mid[ic2]]
   # print("nm = ",np.abs(x1-x2)/x1*100) 
   
   # x1 = (np.mean(nr[ic1],axis=1))[i_mid[ic1]]
   # x2 = (np.mean(nr[ic2],axis=1))[i_mid[ic2]]
   # print("nr = ",np.abs(x1-x2)/x1*100) 
   
   # x1 = (np.mean(n4p[ic1],axis=1))[i_mid[ic1]]
   # x2 = (np.mean(n4p[ic2],axis=1))[i_mid[ic2]]
   # print("n4p = ",np.abs(x1-x2)/x1*100) 

   # x1 = (np.mean(nb[ic1],axis=1))[i_mid[ic1]]
   # x2 = (np.mean(nb[ic2],axis=1))[i_mid[ic2]]
   # print("nb = ",np.abs(x1-x2)/x1*100) 
   
   # x1 = (np.mean(Tg[ic1],axis=1))[i_mid[ic1]]
   # x2 = (np.mean(Tg[ic2],axis=1))[i_mid[ic2]]
   # print("Tg = ",np.abs(x1-x2)/x1*100) 
   
   # x1 = (np.mean(Te[ic1],axis=1))[i_mid[ic1]]
   # x2 = (np.mean(Te[ic2],axis=1))[i_mid[ic2]]
   # print("Te = ",np.abs(x1-x2)/x1*100) 
   
   # exit(-1)

   # nee
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(nee[ic],axis=1),lw=2, label=label[ic])
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12,loc=2)
   ax.set_xlabel(r"$x$ [cm]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylim((0,8.0))
   ax.set_ylabel(r"$E_e$ [J]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Ee_mean.png')


  
   # ne
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.semilogy(xr[ic], np.mean(ne[ic],axis=1), lw=2, label=label[ic])
         # ax.semilogy(xr[ic], np.mean(ne[ic],axis=1), clr[ic], lw=2, label=label[ic])
         ax.semilogy(xr[ic], np.mean(ni[ic],axis=1), '--', lw=2)
         ax.semilogy(xr[ic], np.mean(nAr2i[ic],axis=1), '-.', lw=2)

         ax.set_xlim((xr[ic][0], xr[ic][-1]))

         ax.plot(xr[ic][i_mid[ic]], Ar_Exp_Ne,'k*', lw=1, label="Exp (Langmuir)")
   ax.legend(fontsize=12)
   ax.set_xlabel(r"$x$ [cm]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e}$ [m$^{-3}$]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/ne_mean.png')




   # ne
   # Set the desired width in cm
   width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio
   fig,ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   plt.text(-0.25, 1.10, '(b)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')   
   for ic in case: 
      if case[ic]: 
         ax.semilogy(xr[ic], np.mean(ne[ic],axis=1), c='b', ls='-' , lw=1.5, label=r"$e^{-}$")
         ax.semilogy(xr[ic], np.mean(ni[ic],axis=1), c='r', ls='--', lw=1.5, label=r"$Ar^{+}$")
         ax.semilogy(xr[ic], np.mean(nAr2i[ic],axis=1), c='g', ls='-.', lw=1.5, label=r"$Ar2^{+}$")

         ax.set_xlim((xr[ic][0], xr[ic][-1]))

   ax.legend(fontsize=8)
   ax.set_xlabel(r"$x$ [cm]", fontsize=11)
   plt.setp(ax.get_xticklabels(), fontsize=9)
   ax.set_ylabel(r"[#/m$^{3}$]", fontsize=11)
   plt.setp(ax.get_yticklabels(), fontsize=9)
   plt.tight_layout()
   plt.grid(True)
   plt.savefig('./png/ne_mean_b.png', dpi=300, bbox_inches='tight')

   # VacPermittivity = 8.8541878128e-12
   # lambda_D = np.sqrt(VacPermittivity*spc.k*Te[ic]/K_eV/(ne[ic])/spc.e**2)
   # print(np.mean(lambda_D,axis=1)[75])
   # exit(-1)

   # #skata
   # for ic in case: 
   #    if case[ic]: 
   #       print(ic, i_mid[ic] )
   #       number = np.mean(ne[ic],axis=1)[i_mid[ic]]
   #       print('n_e = ', f"{number:.2e}")

   #       number = np.mean(nAr2i[ic],axis=1)[i_mid[ic]]/np.mean(ni[ic],axis=1)[i_mid[ic]]*100
   #       print('nAr2i / n_i = ', f"{number:.2e}")

   #       number = np.mean(npop[ic][i_mid[ic],1,:],axis=0)
   #       print('n_s5 = ', f"{number:.2e}")

   #       number =  np.mean(nAr2m[ic],axis=1)[i_mid[ic]]
   #       print('nAr2m = ', f"{number:.2e}")


   # exit(0)



   # ne
   # Set the desired width in cm
   width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio
   fig,ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   plt.text(-0.25, 1.10, '(a)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')   
   for ic in case: 
      if case[ic]: 
         if ic == 96:
            ax.semilogy(xr[ic], np.mean(ne[ic],axis=1), c=clr[ic], ls='--' , lw=1.5, label=r"$e^{-}$")
         else:
            ax.semilogy(xr[ic], np.mean(ne[ic],axis=1), c=clr[ic], ls='-' , lw=1.5, label=r"$e^{-}$")
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   # ax.legend(fontsize=8)
   ax.set_xlabel(r"$x$ [cm]", fontsize=11)
   plt.setp(ax.get_xticklabels(), fontsize=9)
   ax.set_ylabel(r"$n_e$ [#/m$^{3}$]", fontsize=11)
   plt.setp(ax.get_yticklabels(), fontsize=9)
   plt.tight_layout()
   plt.grid(True)
   plt.savefig('./png/ne_mean_Grid.png', dpi=300, bbox_inches='tight')




   # n1s5
   # width_cm = 13  # Width in cm
   width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio
   fig,ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   # plt.text(-0.25, 1.10, '(a)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')   
   # plt.title("(150 Pa, 150 V)",fontsize=12)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(n1s5[ic],axis=1), lw=2, color=clr[ic] , label="Current (Sim)")
         # ax.semilogy(xr[ic], np.mean(n1s5[ic],axis=1), lw=2, color=clr[ic] , label="Current (Sim)")

         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=8)
   ax.set_xlabel(r"$x$ [cm]", fontsize=11)
   plt.setp(ax.get_xticklabels(), fontsize=9)
   ax.set_ylabel(r"$n_{AR(1s5)}$ [m$^{-3}$]", fontsize=11)
   plt.setp(ax.get_yticklabels(), fontsize=9)
   plt.tight_layout()
   plt.grid(True)
   plt.savefig('./png/n1s5_mean.png', dpi=300, bbox_inches='tight')


   plt.show()
   exit(0)


   # nm
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(nm[ic],axis=1), lw=2, label=label[ic])
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12)
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(m)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nm_mean.png')

   # nr
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(nr[ic],axis=1), lw=2, label=label[ic])
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12)
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(r)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nr_mean.png')

   # n4p
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(n4p[ic],axis=1), lw=2, label=label[ic])         
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   # ls = ''; uplims =  [Ar_Lumped4p_Exp_ni[1]]; lolims =  [Ar_Lumped4p_Exp_ni[2]]
   # plt.errorbar(xr[ic][i_mid[ic]], Ar_Lumped4p_Exp_ni[0], yerr=(lolims, uplims), 
   #              c='k', marker='s' ,linestyle=ls, lw=1.1, label="Exp (lumped)")
   ax.legend(fontsize=12)
   # ax.semilogy()
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(4p)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/n4p_mean.png')


   # nAr2i
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         # ax.semilogy(xr[ic], np.mean(nAr2i[ic],axis=1), lw=2, label=label[ic])         
         ax.semilogy(xr[ic], np.mean(nAr2i[ic],axis=1)/np.mean(ni[ic],axis=1)*100, lw=2, label=label[ic])         
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12)
   # ax.semilogy()
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{{AR_2}^{+}}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nAr2i_mean.png')


   # nAr2m
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(nAr2m[ic],axis=1), lw=2, label=label[ic])         
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12)
   # ax.semilogy()
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{{Ar_2}^{*}}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nAr2m_mean.png')


   # nb
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(nb[ic],axis=1), lw=2, label=label[ic])
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12)
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nb_mean.png')

   # Ionization Degree
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.semilogy(xr[ic], np.mean(ne[ic],axis=1)/np.mean(nb[ic],axis=1)*100, lw=2, label=label[ic])   
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12)
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r" Ion. Degree [$\%$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/IonDegree_mean.png')



   # Te
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(Te[ic],axis=1),lw=2, label=label[ic])
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
         ax.plot(xr[ic][i_mid[ic]], Ar_Exp_Te,'k*', lw=1, label="Exp (Langmuir)")
   ax.legend(fontsize=12,loc=2)
   ax.set_xlabel(r"$x$ [cm]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylim((0,8.0))
   ax.set_ylabel(r"$T_e$ [eV]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   #
   #ax2 = ax.twinx()
   #ax2.plot(xr[ic], V0*np.mean(phi,axis=1), 'r--', lw=2, label=r"$\phi$")
   #ax2.legend(fontsize=12,loc=1)
   ##ax2.set_ylim((0,70))
   #ax2.set_ylabel(r"$\phi$ [V]", fontsize=18)
   #plt.setp(ax2.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Te_mean.png')


   # Te
   # Set the desired width in cm
   width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio
   fig,ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   plt.text(-0.15, 1.10, '(c)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')
   for ic in case: 
      if case[ic]: 
         # if ic == 82:
            # ax.plot(xr[ic], uniform_filter1d(np.mean(Te[ic],axis=1), size=3),lw=1.5, c=clr[ic])
         if ic == 96:
            ax.plot(xr[ic], np.mean(Te[ic],axis=1),lw=1.5,ls='--', c=clr[ic])
         else:
            ax.plot(xr[ic], np.mean(Te[ic],axis=1),lw=1.5, c=clr[ic])
         ax.set_xlim((xr[ic][0]-0.05, xr[ic][-1]+0.05))
   # ax.legend(fontsize=8,loc=2)
   ax.set_xlabel(r"$x$ [cm]", fontsize=11)
   plt.setp(ax.get_xticklabels(), fontsize=9)
   ax.set_ylim((0,6.0))
   ax.set_ylabel(r"$T_e$ [eV]", fontsize=11)
   plt.setp(ax.get_yticklabels(), fontsize=9)
   plt.tight_layout()
   plt.grid(True)
   # plt.savefig('./png/Te_mean_c.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Te_mean_MaxEEDF.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Te_mean_EEDF.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Te_mean_Grid.png', dpi=300, bbox_inches='tight')





   # Tg
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr[ic], np.mean(Tg[ic],axis=1), lw=2, label=label[ic])
         ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.legend(fontsize=12,loc=2)
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$T_g$ [K]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Tg_mean.png')




   # # Just a estimation of the distribution
   # X_sp_i = np.array([9.99998E-01, 1.24270E-06, 6.21350E-10, 9.32026E-07, 3.10675E-10,
   #           6.21350E-11, 6.21350E-11, 6.21350E-11, 4.66013E-11, 4.66013E-11,
   #           3.10675E-11, 3.10675E-11, 3.10675E-11, 3.10675E-11, 3.10675E-11])

   # N_sp_i = X_sp_i *  Ntot 


   # Calculate Boltzmann Distribution
   # ic0 = 1; Te0 = np.mean(Te[ic0],axis=1)[i_mid[ic]]
   # npop_LTE = CalcBoltzmannDistribution(ic0, Te0 , model, npop, dEps, g, i_mid[ic])


   # ic0 = 2; Te0_2 = 0.6
   # npop_LTE_2 = CalcBoltzmannDistribution(ic0, Te0_2 , model, npop, dEps, g, i_mid[ic])


   # Distribution of population
   fig,ax = plt.subplots(dpi=160)
   plt.title(ExpCase)
   for ic in case: 
      if case[ic]: 
         ax.scatter(dEps[ic][0:-2], np.mean(npop[ic][i_mid[ic],:,:],axis=1)/g[ic][0:-2], marker='.', lw=1.5, label=label[ic])
         ax.plot(Eion, np.mean(ne[ic][i_mid[ic],:],axis=0),marker='*', lw=1.5)
         
   ax.scatter(Ar_LAS_Exp_Ei, Ar_LAS_Exp_ni/Ar_LAS_Exp_gi, c='k', marker='x', lw=1.5, label="Exp (LAS)")        
   ax.plot(Eion, Ar_Exp_Ne,'k*', lw=1, label="Exp (Langmuir)")
   ls = ''; uplims =  Ar_OES_Exp_ni[:,1]/Ar_OES_Exp_gi; lolims =  Ar_OES_Exp_ni[:,2]/Ar_OES_Exp_gi
   plt.errorbar(Ar_OES_Exp_Ei, Ar_OES_Exp_ni[:,0]/Ar_OES_Exp_gi, 
               yerr=(lolims, uplims), c='k', marker='+' ,linestyle=ls, lw=1.5, label="Exp (OES)")


   # label_tmp = "Exp (lumped) - " + ExpCase
   # ax.scatter(Eps_lumped_exp, ni_lumped_exp[ExpCase]/gi_lumped_exp, c='k', marker='.', lw=1.5, label=label_tmp)        
   # ax.scatter(dEps[ic0][0:-2], npop_LTE/g[ic0][0:-2],lw=0.8, label="Boltzmann at  Te = " + str(round(Te0,2)))   
   # ax.scatter(dEps[ic0][0:-2], npop_LTE_2/g[ic0][0:-2],marker='.',lw=1.0, label="Boltzmann at  Te = " + str(round(Te0_2,2)))   
        
   ax.legend(fontsize=12,loc=2)
   # ax.loglog()
   ax.semilogy()
   # ax.set_xlim((xr[ic][0], xr[ic][-1]))
   ax.set_xlabel(r"$E$ [eV]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylim((0,5.5))
   ax.set_ylabel(r"$n_i / g_i$ [m$^{-3}$]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Distribution_mean.png')




   # Distribution of population
   # Set the desired width in cm
   width_cm = 13  # Width in cm
   # width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   # aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio

   fig,ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   # plt.title("(1 Torr, 150 V)")
   plt.text(-0.25, 1.10, '(a)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')
   for ic in case: 
      if case[ic]: 
         ax.scatter(dEps[ic][0:-2], np.mean(npop[ic][i_mid[ic],:,:],axis=1)/g[ic][0:-2], marker='.',c=clr[ic], lw=1.5, label=label[ic])
         ax.plot(Eion, np.mean(ne[ic][i_mid[ic],:],axis=0),marker='*',c=clr[ic], lw=1.5)    
   # ax.scatter(Ar_LAS_Exp_Ei, Ar_LAS_Exp_ni/Ar_LAS_Exp_gi, c='k', marker='x', lw=1.5, label="Exp (LAS)")        
   # ax.plot(Eion, Ar_Exp_Ne,'k*', lw=1, label="Exp (Langmuir)")
   # ls = ''; uplims =  Ar_OES_Exp_ni[:,1]/Ar_OES_Exp_gi; lolims =  Ar_OES_Exp_ni[:,2]/Ar_OES_Exp_gi
   # plt.errorbar(Ar_OES_Exp_Ei, Ar_OES_Exp_ni[:,0]/Ar_OES_Exp_gi, 
   #             yerr=(lolims, uplims), c='k', marker='+' ,linestyle=ls, lw=1.0, label="Exp (OES)")
   # ax.legend(fontsize=7,loc=4)
   ax.legend(fontsize=11,loc=9)
   # ax.loglog()
   ax.semilogy()
   ax.set_xlim((dEps[ic][1]-0.2, Eion+0.2))
   ax.set_ylim((7e11, 1e17))
   ax.set_xlabel(r"$E_i$ [eV]", fontsize=12)
   plt.setp(ax.get_xticklabels(), fontsize=10)
   ax.set_ylabel(r"$n_i / g_i$ [#/m$^3$]", fontsize=12)
   plt.setp(ax.get_yticklabels(), fontsize=10)
   plt.tight_layout()
   plt.grid(True)
   # plt.savefig('./png/Distribution_a.png', dpi=300, bbox_inches='tight')
   plt.savefig('./png/Distribution_EEDF.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Distribution_ExcitedStates.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Distribution_Comparison_CR.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Distribution_OP_a.png', dpi=300, bbox_inches='tight')




   # Distribution of population
   # Set the desired width in cm
   # width_cm = 13  # Width in cm
   width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio

   fig,ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   # plt.title("(1 Torr, 150 V)")
   # plt.text(-0.25, 1.10, '(a)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')
   for ic in case: 
      if case[ic]: 
         ax.scatter(dEps[ic][0:-2], np.mean(npop[ic][i_mid[ic],:,:],axis=1)/g[ic][0:-2], marker='.',c=clr[ic], lw=1.0, label=label[ic])
         ax.plot(Eion, np.mean(ne[ic][i_mid[ic],:],axis=0),marker='*',c=clr[ic], lw=1.0)    
   # ax.scatter(Ar_LAS_Exp_Ei, Ar_LAS_Exp_ni/Ar_LAS_Exp_gi, c='k', marker='x', lw=1.5, label="Exp (LAS)")        
   # ax.plot(Eion, Ar_Exp_Ne,'k*', lw=1, label="Exp (Langmuir)")
   # ls = ''; uplims =  Ar_OES_Exp_ni[:,1]/Ar_OES_Exp_gi; lolims =  Ar_OES_Exp_ni[:,2]/Ar_OES_Exp_gi
   # plt.errorbar(Ar_OES_Exp_Ei, Ar_OES_Exp_ni[:,0]/Ar_OES_Exp_gi, 
   #             yerr=(lolims, uplims), c='k', marker='+' ,linestyle=ls, lw=1.0, label="Exp (OES)")
   # ax.legend(fontsize=7,loc=4)
   ax.legend(fontsize=11,loc=9)
   # ax.loglog()
   ax.semilogy()
   ax.set_xlim((dEps[ic][1]-0.2, Eion+0.2))
   ax.set_ylim((1e11, 1e17))
   ax.set_xlabel(r"$E_i$ [eV]", fontsize=11)
   plt.setp(ax.get_xticklabels(), fontsize=9)
   ax.set_ylabel(r"$n_i / g_i$ [#/m$^3$]", fontsize=11)
   plt.setp(ax.get_yticklabels(), fontsize=9)
   plt.tight_layout()
   plt.grid(True)
   # plt.savefig('./png/Distribution_a.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Distribution_EEDF.png', dpi=300, bbox_inches='tight')
   plt.savefig('./png/Distribution_ExcitedStates.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Distribution_Comparison_CR.png', dpi=300, bbox_inches='tight')
   # plt.savefig('./png/Distribution_OP_a.png', dpi=300, bbox_inches='tight')








   fig,ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   plt.text(-0.25, 1.10, '(b)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')
   for ic in case: 
      if case[ic]: 
         ax.scatter(dEps[ic][5:-2], np.mean(npop[ic][i_mid[ic],5:,:],axis=1)/g[ic][5:-2], marker='.',c='b', lw=1.0, label=label[ic])

   ls = ''; uplims =  Ar_OES_Exp_ni[:,1]/Ar_OES_Exp_gi; lolims =  Ar_OES_Exp_ni[:,2]/Ar_OES_Exp_gi
   plt.errorbar(Ar_OES_Exp_Ei, Ar_OES_Exp_ni[:,0]/Ar_OES_Exp_gi, 
               yerr=(lolims, uplims), c='k', marker='+' ,linestyle=ls, lw=1.0, label="Exp (OES)")
   ax.semilogy()
   ax.set_xlim((dEps[ic][5]-0.1, dEps[ic][14]+0.1))
   ax.set_ylim((1e10, 2e13))
   ax.set_xlabel(r"$E_i$ [eV]", fontsize=11)
   plt.setp(ax.get_xticklabels(), fontsize=9)
   ax.set_ylabel(r"$n_i / g_i$ [#/m$^3$]", fontsize=11)
   plt.setp(ax.get_yticklabels(), fontsize=9)
   plt.tight_layout()
   plt.grid(True)
   # circle_center_relative = (0.5, 0.5)  # Center of the circle
   # circle_radius_relative = 0.1  # Radius of the circle
   # circle = patches.Circle(circle_center_relative, circle_radius_relative, transform=ax.transAxes, color='red', fill=False, linestyle='--', linewidth=0.8)
   # ax.add_patch(circle)
   ellipse_center = (0.64, 0.475)  # Center of the ellipse
   ellipse_width = 0.175  # Width of the ellipse (horizontal diameter)
   ellipse_height = 0.78  # Height of the ellipse (vertical diameter)
   ellipse = patches.Ellipse(ellipse_center, ellipse_width, ellipse_height, transform=ax.transAxes, color='red', fill=False, linestyle='--', linewidth=1)
   ax.add_patch(ellipse)
   plt.savefig('./png/Distribution_b.png', dpi=300, bbox_inches='tight')
   plt.savefig('./png/Distribution_OP_a.png', dpi=300, bbox_inches='tight')




   # Set the desired width in cm
   # width_cm = 13  # Width in cm
   width_cm = 9  # Width in cm
   aspect_ratio = 0.85  # Example aspect ratio (height/width)
   height_cm = width_cm * aspect_ratio

   #Skata3
   ### Parameters you want to describe
   # params_text = (
   #    "Druyvesteyn \n"
   #    r"$\gamma = 0.12$"
   # )

   fig,ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)))
   plt.title("(0.5 Torr, 150 V)",fontsize=12)
   plt.text(-0.25, 1.10, '(a)', transform=plt.gca().transAxes, fontsize=11, fontweight='normal', va='top', ha='left')
   # ax.text(0.6, 0.89, params_text,
   #      transform=ax.transAxes,          # transform coordinates to Axes coordinates
   #      fontsize=10,
   #      verticalalignment='center',
      #   bbox=dict(boxstyle="round", facecolor="white", alpha=0.4))
   for ic in case: 
      if case[ic]: 
         ax.scatter(dEps[ic][5:-2], np.mean(npop[ic][i_mid[ic],5:,:],axis=1)/g[ic][5:-2], marker='.',c='b', lw=1.0, label=label[ic])

   ls = ''; uplims =  Ar_OES_Exp_ni[:,1]/Ar_OES_Exp_gi; lolims =  Ar_OES_Exp_ni[:,2]/Ar_OES_Exp_gi
   plt.errorbar(Ar_OES_Exp_Ei, Ar_OES_Exp_ni[:,0]/Ar_OES_Exp_gi, 
               yerr=(lolims, uplims), c='k', marker='+' ,linestyle=ls, lw=1.0, label="Exp (OES)")
   ax.semilogy()
   ax.set_xlim((dEps[ic][5]-0.1, dEps[ic][14]+0.1))
   ax.set_ylim((1e10, 1e15))
   ax.set_xlabel(r"$E_i$ [eV]", fontsize=11)
   plt.setp(ax.get_xticklabels(), fontsize=9)
   ax.set_ylabel(r"$n_i / g_i$ [#/m$^3$]", fontsize=11)
   plt.setp(ax.get_yticklabels(), fontsize=9)
   plt.tight_layout()
   plt.grid(True)
   # ax.legend(fontsize=10,loc=1)
   # ellipse_center = (0.64, 0.475)  # Center of the ellipse
   # ellipse_width = 0.175  # Width of the ellipse (horizontal diameter)
   # ellipse_height = 0.78  # Height of the ellipse (vertical diameter)
   # ellipse = patches.Ellipse(ellipse_center, ellipse_width, ellipse_height, transform=ax.transAxes, color='red', fill=False, linestyle='--', linewidth=1)
   # ax.add_patch(ellipse)
   plt.savefig('./png/Distribution_0.5T150V.png', dpi=300, bbox_inches='tight')










   # Distribution of lumped states
   dEps_6sp_CRIndexing = np.array([0.0,11.577,11.725,13.168,0.0, 15.76]) 
   g_6sp_CRIndexing = np.array([1, 6, 6, 36, 1, 4])
   fig,ax = plt.subplots(dpi=160)
   plt.title(ExpCase)
   for ic in case: 
      if case[ic]:
         ni_lumped = np.array([np.mean(nb[ic][i_mid[ic],:],axis=0), np.mean(nm[ic][i_mid[ic],:],axis=0), 
                               np.mean(nr[ic][i_mid[ic],:],axis=0), np.mean(n4p[ic][i_mid[ic],:],axis=0)]) 
         ax.scatter(dEps_6sp_CRIndexing[0:-2], ni_lumped/g_6sp_CRIndexing[0:-2], marker='.', lw=1.5, label=label[ic])
         ax.plot(Eion, np.mean(ne[ic][i_mid[ic],:],axis=0),marker='*', lw=1.5)
   ax.plot(Eion, Ar_Exp_Ne,'k*', lw=1, label="Exp (Langmuir)")
   Ar_LAS_Exp_nm = Ar_LAS_Exp_ni[0] + Ar_LAS_Exp_ni[2]
   Ar_LAS_Exp_nr = Ar_LAS_Exp_ni[1] + Ar_LAS_Exp_ni[3]
   ax.scatter(dEps_6sp_CRIndexing[1:3], np.array([Ar_LAS_Exp_nm, Ar_LAS_Exp_nr])/g_6sp_CRIndexing[1:3], c='k', marker='x', lw=1.5, label="Exp (LAS)")        
   ls = ''; uplims =  [Ar_Lumped4p_Exp_ni[1]/Ar_Lumped4p_Exp_gi[2]]; lolims =  [Ar_Lumped4p_Exp_ni[2]/Ar_Lumped4p_Exp_gi[2]]
   plt.errorbar(Ar_Lumped4p_Exp_Ei[2], Ar_Lumped4p_Exp_ni[0]/Ar_Lumped4p_Exp_gi[2], 
               yerr=(lolims, uplims), c='k', marker='+' ,linestyle=ls, lw=1.0, label="Exp (OES)")
   ax.legend(fontsize=12,loc=2)
   ax.semilogy()
   ax.set_xlabel(r"$E$ [eV]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_i / g_i$ [m$^{-3}$]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Distribution_lumped_mean.png')


   # # ni
   # fig,ax = plt.subplots(dpi=160)
   # for ic in case: 
   #    if case[ic]: 
   #       ax.plot(xr[ic], np.mean(npop[ic][:,1,:],axis=1), lw=2, label=r'$1s5$')
   #       ax.plot(xr[ic], np.mean(npop[ic][:,2,:],axis=1), lw=2, label=r'$1s4$')
   #       ax.plot(xr[ic], np.mean(npop[ic][:,3,:],axis=1), lw=2, label=r'$1s3$')
   #       ax.plot(xr[ic], np.mean(npop[ic][:,4,:],axis=1), lw=2, label=r'$1s2$')
   #       ax.set_xlim((xr[ic][0], xr[ic][-1]))
   # ax.legend(fontsize=12)
   # ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$n_i$ [m$^{-3}$]", fontsize=18)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/n4s_mean.png')



plt.show()


