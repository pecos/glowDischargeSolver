import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as spc




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
              

#----------------------------------------------------------------------------------


# sys.path.insert(0, '../')
# Constants
K_eV = spc.k/spc.e             # Convert energy units: from K to eV

# Flags
isPlotLines = False
isPlotMeans = True


## Species energies and degeneracies
# E, AR+, AR(m), AR(r), AR(4p), AR
dEps_6sp = np.array([0.0,15.76,11.577,11.725,13.168,0.0]) 
g_6sp = np.array([1, 4, 6, 6, 36, 1])

# electrons + ions + 4 4s levels + 10 4p levels + background state
dEps_CR = np.array([ 0.0,         15.7596119,  11.54835442, 11.62359272, 11.72316039, 11.82807116,
                     12.9070153,  13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,
                     13.28263902, 13.30222747, 13.32785705, 13.47988682,  0.0]) 
g_CR = np.array([1, 4, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1, 1])




Te_exp = {}; ne_exp = {}; ni_exp = {}; gi_exp = {}; Ei_exp = {}
# AR(m), AR(r), AR(4p)
ni_lumped_exp = {}; Eps_lumped_exp = np.array([11.577,11.725,13.168]); gi_lumped_exp = np.array([6, 6, 36])

ExpCase = '1Torr-150V' # 150V is the tip-to-tip Voltage. In our case V0 would be Vmax = 75 V.
Te_exp[ExpCase] = 7.01 # [eV]
ne_exp[ExpCase] = 2.2e15 # [#/m^3]

ni_exp[ExpCase] = np.array([0.0, 9.59E+15, 3.27E+15, 3.16E+14, 9.58E+14, 2.63E+12, 1.40E+11,  7.72E+11, 
                            5.25E+11, 8.53E+11, 3.47E+11, 3.43E+11, 4.43E+11, 4.64E+11, 6.43E+11])
gi_exp[ExpCase] = np.array([1, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1])   
Ei_exp[ExpCase] =  np.array([ 0.0, 11.54835442, 11.62359272, 11.72316039, 11.82807116, 12.9070153, 13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,  13.28263902, 13.30222747, 13.32785705, 13.47988682])
ni_lumped_exp[ExpCase] = [ ni_exp[ExpCase][1]+ni_exp[ExpCase][3], ni_exp[ExpCase][2]+ni_exp[ExpCase][4], np.sum(ni_exp[ExpCase][5:]) ]

ExpCase = '1Torr-200V' # 150V is the tip-to-tip Voltage. 
Te_exp[ExpCase] = np.nan # [eV]
ne_exp[ExpCase] = np.nan # [#/m^3]
ni_exp[ExpCase] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 3.07E+12, 1.12E+12, 1.3E+12, 1.42E+12, 1.33E+12,1.86E+12, 7.62E+11, 7.85E+11, 7.95E+11, 4.35E+12])
gi_exp[ExpCase] = gi_exp['1Torr-150V']; Ei_exp[ExpCase] =  Ei_exp['1Torr-150V']
ni_lumped_exp[ExpCase] = [ ni_exp[ExpCase][1]+ni_exp[ExpCase][3], ni_exp[ExpCase][2]+ni_exp[ExpCase][4], np.sum(ni_exp[ExpCase][5:]) ]



# Cases
case = {}; file = {}; clr = {}; label = {}; model = {}


ic = 1; c = True; f = '../Results/6spec/1torr_75V_Np150/periodic/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'b-'; lb = "6sp"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 2; c = True; f = '../Results/6spec/1torr_75V_Np150/periodic/newton_6spec_BE_Np150_fullsoln.npy'; cl = 'b-'; lb = "6sp"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 3; c = True; f = '../Results/CR/1Torr_75V_Np150_MaxEEDF_ConstDiff/periodic/newton_CR_BE_Np150_fullsoln.npy'; cl = 'r-'; lb = "CR - MaxEEDF - Const De"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 4; c = True; f = '../Results/CR/1Torr_75V_Np150_BolsigEEDF_ConstDiff/newton_CR_CN_Np150_fullsoln.npy'; cl = 'm-'; lb = "CR - Bolsig+ - Const De"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 5; c = True; f = '../Results/CR/1Torr_75V_Np150_BolsigEEDF_ConstDiff_Qrad/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR"; m = "CR"
# ic = 5; c = True; f = '../Results/CR/1Torr_75V_Np150_BolsigEEDF_ConstDiff_Qrad/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR - Bolsig+  - Const De - Qrad"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 6; c = True; f = '../Results/CR/1Torr_75V_Np150_BolsigEEDF/newton_CR_CN_Np150_fullsoln.npy'; cl = 'c-'; lb = "CR"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 1; c = True; f = '../fullsoln.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# ic = 1; c = True; f = '../Results/CR/CR_Np150_fullsoln.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# ic = 1; c = True; f = '../Results/test/1/fullsoln.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# ic = 1; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T2000.npy'; cl = 'b'; lb = "CR"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 2; c = False; f = '../Results/6spec/nominalCase_V100_P1torr_Np150/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'g'; lb = "6sp"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 3; c = True; f = '../Results/6spec/nominalCase_V100_P1torr_Np150_TimeMarching/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'm'; lb = "6sp - Time Marching"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 2; c = False; f = '../Results/6spec/nominalCase_V100_P1torr_Np150_constDiff/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "6sp - constDiff"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 5; c = False; f = '../Results/6spec/nominalCase_V100_P1torr_Np150_dEps/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'k'; lb = "6sp - dEps"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 



# ic = 2; c = True; f = '../Results/6spec/1torr_100V_Np150_constDiff/fullsoln/newton_6spec_CN_Np150_fullsoln_T2400.npy'; cl = 'b-'; lb = "6sp"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 2; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T4000.npy'; cl = 'b-'; lb = "6sp"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 1; c = True; f = '../Results/CR/1Torr_100V_Np150_MaxEEDF_ConstDiff/fullsoln/CR_Np150_fullsoln_T1250.npy'; cl = 'm-'; lb = "T1250"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 2; c = True; f = '../Results/CR/1Torr_100V_Np150_MaxEEDF_ConstDiff/fullsoln/CR_Np150_fullsoln_T1500.npy'; cl = 'g-'; lb = "T1500"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 3; c = True; f = '../Results/CR/1Torr_100V_Np150_MaxEEDF_ConstDiff/fullsoln/CR_Np150_fullsoln_T1750.npy'; cl = 'r-'; lb = "T1750"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 4; c = True; f = '../Results/CR/1Torr_100V_Np150_MaxEEDF_ConstDiff/fullsoln/CR_Np150_fullsoln_T2000.npy'; cl = 'b-'; lb = "T2000"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 5; c = True; f = '../Results/CR/1Torr_100V_Np150_MaxEEDF_ConstDiff/fullsoln/CR_Np150_fullsoln_T2250.npy'; cl = 'm-'; lb = "T2250"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 6; c = True; f = '../Results/CR/1Torr_100V_Np150_MaxEEDF_ConstDiff/fullsoln/CR_Np150_fullsoln_T2500.npy'; cl = 'm-'; lb = "T2500"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 7; c = True; f = '../Results/CR/1Torr_100V_Np150_MaxEEDF_ConstDiff/fullsoln/CR_Np150_fullsoln_T2750.npy'; cl = 'r'; lb = "T2750"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 1; c = True; f = '../Results/6spec/1torr_100V_Np150_constDiff/fullsoln/newton_6spec_CN_Np150_fullsoln_T400.npy'; cl = 'b-'; lb = "T400"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 2; c = True; f = '../Results/6spec/1torr_100V_Np150_constDiff/fullsoln/newton_6spec_CN_Np150_fullsoln_T800.npy'; cl = 'b-'; lb = "T800"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 3; c = True; f = '../Results/6spec/1torr_100V_Np150_constDiff/fullsoln/newton_6spec_CN_Np150_fullsoln_T1200.npy'; cl = 'b-'; lb = "T1200"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 4; c = True; f = '../Results/6spec/1torr_100V_Np150_constDiff/fullsoln/newton_6spec_CN_Np150_fullsoln_T1600.npy'; cl = 'b-'; lb = "T1600"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 5; c = True; f = '../Results/6spec/1torr_100V_Np150_constDiff/fullsoln/newton_6spec_CN_Np150_fullsoln_T2000.npy'; cl = 'b-'; lb = "T2000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 6; c = True; f = '../Results/6spec/1torr_100V_Np150_constDiff/fullsoln/newton_6spec_CN_Np150_fullsoln_T2400.npy'; cl = 'b-'; lb = "T2400"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 



# ic = 2; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T400.npy'; cl = 'b-'; lb = "T400"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 3; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T800.npy'; cl = 'b-'; lb = "T800"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 4; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T1200.npy'; cl = 'b-'; lb = "T1200"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 5; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T1600.npy'; cl = 'b-'; lb = "T1600"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 6; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T2000.npy'; cl = 'b-'; lb = "T2000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 7; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T2400.npy'; cl = 'b-'; lb = "T2400"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 8; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T2800.npy'; cl = 'b-'; lb = "T2800"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 9; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T3200.npy'; cl = 'b-'; lb = "T3200"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 10; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T3600.npy'; cl = 'b-'; lb = "T3600"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 11; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T4000.npy'; cl = 'b-'; lb = "T4000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 12; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T5000.npy'; cl = 'b-'; lb = "T5000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 13; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T6000.npy'; cl = 'b-'; lb = "T6000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 14; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T7000.npy'; cl = 'b-'; lb = "T7000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 15; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T8000.npy'; cl = 'b-'; lb = "T8000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 16; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T9000.npy'; cl = 'b-'; lb = "T9000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 17; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T10000.npy'; cl = 'b-'; lb = "T10000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 18; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T11000.npy'; cl = 'b-'; lb = "T11000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 19; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T12000.npy'; cl = 'b-'; lb = "T12000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 20; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T13000.npy'; cl = 'b-'; lb = "T13000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 21; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T14000.npy'; cl = 'b-'; lb = "T14000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 22; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T15000.npy'; cl = 'b-'; lb = "T15000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 23; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T16000.npy'; cl = 'b-'; lb = "T16000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 24; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T18000.npy'; cl = 'b-'; lb = "T18000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 25; c = True; f = '../Results/6spec/1torr_75V_Np150/fullsoln/newton_6spec_CN_Np150_fullsoln_T20000.npy'; cl = 'b_'; lb = "T20000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 






# these values are required to "redimensionalize" the results
# they must be consistent with the scenario input file
Pressure  = 1.0*spc.torr               # [Pa] 
GasTemperature = 300.0                 # [K]
nAr = Pressure/GasTemperature/spc.k    # [#/m^3] Number density based on bulk temperature (not necessarily true density in two-temperature gas)

ne0 = 8e16           # "nominal" electron density [1/m^3]
L   = 2.00*0.005     # half-gap-width [m] (gap width is 2 cm)
tau = (1./13.56e6)   # period of driving voltage [s]
# V0  = 100.0          # amplitude of driving voltage [V]
V0  = 75.0          # amplitude of driving voltage [V]
p_0 = Pressure


# Number of Chebyshev modes
Np=150

# T = number of time steps
T=128

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



ne = {}; ni = {}; nb = {}; nee = {}; npop = {}; Tg = {}
nm = {}; nr = {}; n4p = {}; Te = {}; dEps = {}; g = {}
FromGlowDischargeToCRIndexing = {}
FromCRToGlowDischargeIndexing = {}

# if case1:
for ic in case: 
   if case[ic]: 
      print("Loading case ",ic, " from file :", file[ic])
         
      # load solution file
      D = np.load(file[ic].format(Np))
      D = np.transpose(D)


      # Ns is the number of scpecies
      if model[ic] == "CR":
         Ns = 17 # electrons + ions + 4 4s levels + 10 4p levels + background state
      elif model[ic] == "6sp":
         Ns = 6 #  electrons + ions + nm + nr + n4p + nb 
         
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

      FromCRToGlowDischargeIndexing[ic]  = [Ns-2, Ns-1] + list(range(1,Ns-2)) + [0] # We have excluded electron energy
      FromGlowDischargeToCRIndexing[ic]  = [Ns-1] + list(range(2,Ns-1)) + [0, 1] # We have excluded electron energy

      
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

      ne[ic]  = ne0 * D_reshaped[:,0,:]          # electron density
      ni[ic]  = ne0 * D_reshaped[:,1,:]       # ion density
      nb[ic]  = nAr * D_reshaped[:,Ns - 1,:] # "background" (argon neutral) density
      nee[ic] = (2./3.) * ne0 * D_reshaped[:,Ns,:]          # electron energy (ne * ee)

      npop[ic] = np.ndarray((Np, Ns-2, np.shape(D)[1]),dtype=np.float64)
      npop[ic][:,0,:] = nb[ic]
      npop[ic][:,1:,:] = ne0 * D_reshaped[:,2:Ns-1,:]

      if model[ic] == "CR":
         nm[ic] = ne0 * (D_reshaped[:,2,:] + D_reshaped[:,4,:]) 
         nr[ic] = ne0 * (D_reshaped[:,3,:] + D_reshaped[:,5,:]) 
         n4p[ic] = np.zeros_like(nr[ic])
         for i in range(6,15+1):
            print(i)
            n4p[ic] += ne0 * D_reshaped[:,i,:] 
      elif model[ic] == "6sp":
         nm[ic]  = ne0 * D_reshaped[:,2,:]
         nr[ic]  = ne0 * D_reshaped[:,3,:] 
         n4p[ic] = ne0 * D_reshaped[:,4,:] 

      # electron temp
      Te[ic] = nee[ic] / ne[ic]  

      Tg[ic] = (p_0/spc.k - ne[ic] * Te[ic]/K_eV) / (np.sum(npop[ic], axis=1) + ni[ic])   # [K]


      if model[ic] == "CR":
         dEps[ic] = dEps_CR[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_CR[FromGlowDischargeToCRIndexing[ic]]
         
      elif model[ic] == "6sp":
         dEps[ic] = dEps_6sp[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_6sp[FromGlowDischargeToCRIndexing[ic]]

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

   # Te/phi
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
   #ax2 = ax.twinx()
   #ax2.plot(xr, V0*phi[:,t0], 'b--', lw=2)
   #ax2.plot(xr, V0*phi[:,t1], 'r--', lw=2)
   #ax2.plot(xr, V0*phi[:,t2], 'g--', lw=2)
   #ax2.plot(xr, V0*phi[:,t3], 'c--', lw=2)
   #ax2.set_ylim((-120,120))
   #ax2.set_ylabel(r"$\phi$ [V]", labelpad=-5, fontsize=18)
   #plt.setp(ax2.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Te_line.png')


if (isPlotMeans):
   print("Plotting means...")
   
   # ne
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.semilogy(xr, np.mean(ne[ic],axis=1), lw=2, label=label[ic])
         # ax.semilogy(xr, np.mean(ne[ic],axis=1), clr[ic], lw=2, label=label[ic])
         # ax.semilogy(xr, np.mean(ni[ic],axis=1), clr[ic]+'-', lw=2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e}$ [m$^{-3}$]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/ne_mean.png')



   # nm
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, np.mean(nm[ic],axis=1), lw=2, label=label[ic])
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(m)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nm_mean.png')

   # nr
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, np.mean(nr[ic],axis=1), lw=2, label=label[ic])
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(r)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nr_mean.png')

   # n4p
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, np.mean(n4p[ic],axis=1), lw=2, label=label[ic])         
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(4p)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/n4p_mean.png')

   # nb
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, np.mean(nb[ic],axis=1), lw=2, label=label[ic])
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nb_mean.png')

   # Ionization Degree
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.semilogy(xr, np.mean(ne[ic],axis=1)/np.mean(nb[ic],axis=1)*100, lw=2, label=label[ic])   
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r" Ion. Degree [$\%$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/IonDegree_mean.png')



   # Te/phi
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, np.mean(Te[ic],axis=1),lw=2, label=label[ic])
   ax.legend(fontsize=12,loc=2)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylim((0,5.5))
   ax.set_ylabel(r"$T_e$ [eV]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   #
   #ax2 = ax.twinx()
   #ax2.plot(xr, V0*np.mean(phi,axis=1), 'r--', lw=2, label=r"$\phi$")
   #ax2.legend(fontsize=12,loc=1)
   ##ax2.set_ylim((0,70))
   #ax2.set_ylabel(r"$\phi$ [V]", fontsize=18)
   #plt.setp(ax2.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Te_mean.png')


   # Tg
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, np.mean(Tg[ic],axis=1), lw=2, label=label[ic])
   ax.legend(fontsize=12,loc=2)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$T_g$ [K]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Tg_mean.png')




   # X_sp_i = np.array([9.99998E-01, 1.24270E-06, 6.21350E-10, 9.32026E-07, 3.10675E-10,
   #           6.21350E-11, 6.21350E-11, 6.21350E-11, 4.66013E-11, 4.66013E-11,
   #           3.10675E-11, 3.10675E-11, 3.10675E-11, 3.10675E-11, 3.10675E-11])

   # # Calculate Boltzmann Distribution
   # ic0 = 1; Ntot = 0
   # for isp in range(17-2):
   #    print(np.mean(npop[ic0][i_mid,isp,:],axis=0))
   #    Ntot = Ntot + np.mean(npop[ic0][i_mid,isp,:],axis=0)
      
   # Te0 = np.mean(Te[ic0],axis=1)[i_mid]
   # # print("Te0 = ",Te0)
   # Q_n,Q_i = PartitionFunctionsAnalytical(Te0)
   # npop_LTE = BoltzmannDistribution(Ntot,Te0,Q_n,dEps[ic0][0:-2],g[ic0][0:-2])

   # # Just a estimation of the distribution
   # X_sp_i = np.array([9.99998E-01, 1.24270E-06, 6.21350E-10, 9.32026E-07, 3.10675E-10,
   #           6.21350E-11, 6.21350E-11, 6.21350E-11, 4.66013E-11, 4.66013E-11,
   #           3.10675E-11, 3.10675E-11, 3.10675E-11, 3.10675E-11, 3.10675E-11])

   # N_sp_i = X_sp_i *  Ntot 


   # npop
   ExpCase = '1Torr-150V'
   # ExpCase = '1Torr-200V'
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.scatter(dEps[ic][0:-2], np.mean(npop[ic][i_mid,:,:],axis=1)/g[ic][0:-2], marker='.', lw=1.5, label=label[ic]+ " - " + ExpCase)
         ax.plot(dEps[ic][-1], np.mean(ne[ic][i_mid,:],axis=0),marker='*', lw=1.5)
   label_tmp = "Exp" " - " + ExpCase
   ax.scatter(Ei_exp[ExpCase], ni_exp[ExpCase]/gi_exp[ExpCase], c='k', marker='x', lw=1.5, label=label_tmp)        
   ax.plot(dEps[ic][-1], ne_exp[ExpCase],'k*', lw=1)
   # label_tmp = "Exp (lumped) - " + ExpCase
   # ax.scatter(Eps_lumped_exp, ni_lumped_exp[ExpCase]/gi_lumped_exp, c='k', marker='.', lw=1.5, label=label_tmp)        
   # ax.scatter(dEps[ic0][0:-2], npop_LTE/g[ic0][0:-2], c='r', label="Boltzmann")        
   ax.legend(fontsize=12,loc=2)
   ax.loglog()
   # ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$E$ [eV]", fontsize=16)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylim((0,5.5))
   ax.set_ylabel(r"$n_i / g_i$ [m$^{-3}$]", fontsize=16)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Distribution_mean.png')


plt.show()
