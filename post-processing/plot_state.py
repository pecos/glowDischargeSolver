import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as spc


# sys.path.insert(0, '../')
# Constants
K_eV = spc.k/spc.e             # Convert energy units: from K to eV



# Flags
isPlot = True

# Cases
case = {}; file = {}; clr = {}; label = {}; model = {}

# ic = 3; c = True; f = '../Results/6spec/nominalCase_V100_P1torr_Np150/newton_6spec_CN_Np150.npy'; cl = 'g'; lb = "6spec"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 1; c = True; f = '../restart.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# ic = 1; c = True; f = '../discard.npy'; cl = 'b-'; lb = "CR"; m = "CR"; m = "CR"
# ic = 1; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/restart_CR_Np150_T1750.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 2; c = False; f = '../Results/6spec/restart.npy'; cl = 'r-'; lb = "6spec"; m = "6sp"
# ic = 2; c = True; f = '../Results/CR/1Torr_100V_Np150_MaxEEDF_ConstDiff/restart_CR_Np150_T2000.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


ic = 1; c = True; f = '../nonconverged_U0.npy'; cl = 'b-'; lb = "U0"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 2; c = True; f = '../nonconverged_U1.npy'; cl = 'g-'; lb = "U1"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 3; c = True; f = '../nonconverged_U2.npy'; cl = 'k-'; lb = "U2"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 4; c = True; f = '../restart.npy'; cl = 'm-'; lb = "restart"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 4; c = True; f = '../restart_CR_BE_Np150_T125.npy'; cl = 'm-'; lb = "T125"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 4; c = True; f = '../discard.npy'; cl = 'm-'; lb = "discard"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 4; c = True; f = '../restart_CR_BE_Np150_T125.npy'; cl = 'm-'; lb = "restart"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 



# ic = 1; c = True; f = '../exception_U0.npy'; cl = 'b-'; lb = "U0"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 2; c = True; f = '../exception_U1.npy'; cl = 'g-'; lb = "U1"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 3; c = True; f = '../exception_U2.npy'; cl = 'k-'; lb = "U2"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 




# ic = 4; c = True; f = '../restart_CR_BE_Np150_T125.npy'; cl = 'r-'; lb = "Max"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 4; c = True; f = '../nonconverged_U0_t014.npy'; cl = 'r-'; lb = "U0 maxwell"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 6; c = True; f = '../restart_Maxwell_constDiff.npy'; cl = 'k-'; lb = "T1 Maxwell const De"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 7; c = True; f = '../restart_Maxwell.npy'; cl = 'r-'; lb = "T1 Maxwell"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 8; c = True; f = '../restart.npy'; cl = 'b-'; lb = "restart"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 4; c = True; f = '../Results/6spec/1torr_100V_Np150_constDiff/restart_6spec_CN_Np150_T2400.npy'; cl = 'r-'; lb = "6sp"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 5; c = True; f = '../restart_6spec_CN_Np150_T400.npy'; cl = 'g-'; lb = "400"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 6; c = True; f = '../Results/6spec/1torr_75V_Np150/restart_6spec_CN_Np150_T4000.npy'; cl = 'b-'; lb = "4000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 7; c = True; f = '../Results/6spec/1torr_75V_Np150/restart_6spec_CN_Np150_T3200.npy'; cl = 'r-'; lb = "3600"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 


# ic = 8; c = True; f = '../Results/6spec/1torr_75V_Np150/restart_6spec_CN_Np150_T14000.npy'; cl = 'b-'; lb = "14000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 9; c = True; f = '../Results/6spec/1torr_75V_Np150/restart_6spec_CN_Np150_T16000.npy'; cl = 'r-'; lb = "16000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 10; c = True; f = '../Results/6spec/1torr_75V_Np150/restart_6spec_CN_Np150_T18000.npy'; cl = 'g-'; lb = "18000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 11; c = True; f = '../Results/6spec/1torr_75V_Np150/restart_6spec_CN_Np150_T20000.npy'; cl = 'k-'; lb = "20000"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 




# these values are required to "redimensionalize" the results
# they must be consistent with the scenario input file
Pressure  = 1.0*spc.torr               # [Pa] 
GasTemperature = 293.15                 # [K]
nAr = Pressure/GasTemperature/spc.k    # [#/m^3] Number density based on bulk temperature (not necessarily true density in two-temperature gas)

ne0 = 8e16           # "nominal" electron density [1/m^3]
L   = 2.00*0.005     # half-gap-width [m] (gap width is 2 cm)
tau = (1./13.56e6)   # period of driving voltage [s]
V0  = 100.0          # amplitude of driving voltage [V]
p_0 = Pressure

# Number of Chebyshev modes
Np=150


# spatial grid
xp = -np.cos(np.pi*np.linspace(0,Np-1,Np)/(Np-1))
xr = (xp+1)*L*100 # [cm]

abs_diff = np.abs(xr - 1.0) # Calculate absolute differences between each value and the midpoint
i_mid = np.argmin(abs_diff) # Find the index of the minimum absolute difference


# Species energies and degeneracies
## E, AR+, AR(m), AR(r), AR(4p), AR
dEps_6sp = np.array([0.0,15.76,11.577,11.725,13.168,0.0]) 
g_6sp = np.array([1, 4, 6, 6, 36, 1])

## E, AR+, 4 4s levels, 10 4p levels + AR
dEps_CR = np.array([ 0.0,         15.7596119,  11.54835442, 11.62359272, 11.72316039, 11.82807116,
                     12.9070153,  13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,
                     13.28263902, 13.30222747, 13.32785705, 13.47988682,  0.0]) 
g_CR = np.array([1, 4, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1, 1])



dEps_CR = np.array([ 0.0, 15.7596119, 11.54835442, 11.62359272, 11.72316039, 11.82807116, 12.9070153,
                     13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,  13.28263902,
                     13.30222747, 13.32785705, 13.47988682, 13.84503846, 13.86366857, 13.90345461,
                     13.97923734, 14.01273812, 14.06302723, 14.06829767, 14.0899685,  14.09905592,
                     14.15251505, 14.2136715,  14.23402264, 14.23610607, 14.24102775, 14.25508557,
                     14.30366841,  0.0])

g_CR = np.array([1, 4, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1, 
                 1, 3, 5, 9, 7, 5, 5, 3, 7, 3, 5, 5, 7, 1, 3, 3, 1])


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
         Ns = 1+30+1+1

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

      

      D_reshaped = np.reshape(D,(Np, Ns+1),'F')

      ne[ic]  = ne0 * D_reshaped[:,0]              # electron density
      ni[ic]  = ne0 * D_reshaped[:,1]              # ion density
      nb[ic]  = nAr * D_reshaped[:,Ns - 1]         # "background" (argon neutral) density
      nee[ic] = (2./3.) * ne0 * D_reshaped[:,Ns]   # electron energy (ne * ee)

      npop[ic] = np.ndarray((Np, Ns-2),dtype=np.float64)
      npop[ic][:,0] = nb[ic]
      npop[ic][:,1:] = ne0 * D_reshaped[:,2:Ns-1]

      if model[ic] == "CR":
         nm[ic] = ne0 * (D_reshaped[:,2] + D_reshaped[:,4]) 
         nr[ic] = ne0 * (D_reshaped[:,3] + D_reshaped[:,5]) 
         n4p[ic] = np.zeros_like(nr[ic])
         for i in range(6,15+1):
            n4p[ic] += ne0 * D_reshaped[:,i] 

      elif model[ic] == "6sp":
         nm[ic]  = ne0 * D_reshaped[:,2]
         nr[ic]  = ne0 * D_reshaped[:,3] 
         n4p[ic] = ne0 * D_reshaped[:,4] 


      # electron temp
      Te[ic] = nee[ic] / ne[ic]  

      Tg[ic] = (p_0/spc.k - ne[ic] * Te[ic]/K_eV) / (np.sum(npop[ic], axis=1) + ni[ic])   # [K]

      if model[ic] == "CR":
         dEps[ic] = dEps_CR[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_CR[FromGlowDischargeToCRIndexing[ic]]
         
      elif model[ic] == "6sp":
         dEps[ic] = dEps_6sp[FromGlowDischargeToCRIndexing[ic]]
         g[ic] = g_6sp[FromGlowDischargeToCRIndexing[ic]]

      del D, D_reshaped 






# make some plots


if (isPlot):
   print("Plotting means...")
   
   # ne
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.semilogy(xr, ne[ic], clr[ic], lw=2, label=label[ic])
         ax.semilogy(xr, ni[ic], clr[ic]+'-', lw=2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e,i}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/ne_mean.png')

   # nm
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, nm[ic], clr[ic], lw=2, label=label[ic])
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
         ax.plot(xr, nr[ic], clr[ic], lw=2, label=label[ic])
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
         ax.plot(xr, n4p[ic], clr[ic], lw=2, label=label[ic])
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
         ax.plot(xr, nb[ic], clr[ic], lw=2, label=label[ic])
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nb_mean.png')



   # # n4p
   # for isp in range(Ns-2): 

   #    fig,ax = plt.subplots(dpi=160)
   #    for ic in case: 
   #       if case[ic] and model[ic] == "CR":
   #          ax.plot(xr, npop[ic][:,isp], clr[ic], lw=2, label=label[ic])
   #    ax.legend(fontsize=12)
   #    ax.set_xlim((xr[0], xr[-1]))
   #    ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   #    plt.setp(ax.get_xticklabels(), fontsize=12)
   #    ax.set_ylabel(r"$n_i}$ [m$^{-3}$] isp = " + str(isp), fontsize=18)
   #    plt.setp(ax.get_yticklabels(), fontsize=12)
   #    plt.savefig('./png/n4p_mean.png')






   # Te/phi
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, Te[ic], clr[ic], lw=2, label=label[ic])
   ax.legend(fontsize=12,loc=2)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylim((0,5.5))
   ax.set_ylabel(r"$T_e$ [eV]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
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
         ax.plot(xr, Tg[ic], clr[ic], lw=2, label=label[ic])
   # ax.plot(xr, Te/K_eV, clr[ic]+"-", lw=2, label=r"T_e")
   ax.legend(fontsize=12,loc=2)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$T_g$ [K]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   #ax2 = ax.twinx()
   #ax2.plot(xr, V0*np.mean(phi,axis=1), 'r--', lw=2, label=r"$\phi$")
   #ax2.legend(fontsize=12,loc=1)
   ##ax2.set_ylim((0,70))
   #ax2.set_ylabel(r"$\phi$ [V]", fontsize=18)
   #plt.setp(ax2.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Tg_mean.png')


   # # distribution
   # fig,ax = plt.subplots(dpi=160)
   # # ax.semilogy(npop[ic][75,:], clr1, lw=2, label=label[ic])
   # ax.plot(xr, nb, clr1, lw=2, label=label[ic])
   # ax.legend(fontsize=12)
   # # ax.set_xlim((xr[0], xr[-1]))
   # ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   # plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylabel(r"$n_{AR}$ [m$^{-3}$]", fontsize=18)
   # plt.setp(ax.get_yticklabels(), fontsize=12)
   # plt.savefig('./png/nb_mean.png')



plt.show()
