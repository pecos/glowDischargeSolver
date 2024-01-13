import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as spc


# sys.path.insert(0, '../')
# Constants
K_eV = spc.k/spc.e             # Convert energy units: from K to eV


# Flags
isPlot = True
# case1 = True; file1 = '../Results/6spec/nominalCase_V100_P1torr_Np150/newton_6spec_CN_Np150.npy'; clr1 = 'b-'; label1 = "6spec"
# case1 = True; file1 = '../restart.npy'; clr1 = 'b-'; label1 = "CR"
case1 = True; file1 = '../Results/CR/restart_CR_Np150_T345.npy'; clr1 = 'b-'; label1 = "CR"


# case2 = False; file2 = '../restart_crashed.npy'; clr2 = 'r-'; label2 = "CR 2"
# case2 = False; file2 = '../Results/6spec/restart.npy'; clr2 = 'r-'; label2 = "6spec"
# case2 = True; file2 = '../Results/CR/restart_CR_Np150_T115.npy'; clr2 = 'r-'; label2 = "CR"
case2 = True; file2 = '../Results/CR/restart_CR_Np150_T230.npy'; clr2 = 'r-'; label2 = "CR"


# these values are required to "redimensionalize" the results
# they must be consistent with the scenario input file
Pressure  = 1.0*spc.torr               # [Pa] 
GasTemperature = 300.0                 # [K]
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

# if case1:
# load solution file
D = np.load(file1.format(Np))
D = np.transpose(D)


# Ns is the number of scpecies
Ns = 17 # background state + 4 4s levels + 10 4p levels + electrons + ions 

# pull solution out of D
'''
   ### Indexing
   # GlowDischarge Indexing 
   # i = 0       -> electrons 
   # i = 1       -> ions        
   # i = 2:Ns-1  -> excited levels
   # i = Ns - 1  -> ground state
   # i = Ns      -> electron energy
'''
D_reshaped = np.reshape(D,(Np, Ns+1),'F')

ne  = ne0 * D_reshaped[:,0]          # electron density
ni  = ne0 * D_reshaped[:,1]       # ion density
nb  = nAr * D_reshaped[:,Ns - 1] # "background" (argon neutral) density
nee = (2./3.) * ne0 * D_reshaped[:,Ns]          # electron energy (ne * ee)

npop = np.ndarray((Np, Ns-2),dtype=np.float64)
npop[:,0] = nb
npop[:,1:] = ne0 * D_reshaped[:,2:Ns-1]

nm = ne0 * (D_reshaped[:,2] + D_reshaped[:,4]) 
nr = ne0 * (D_reshaped[:,3] + D_reshaped[:,5]) 
n4p = np.zeros_like(nr)
for i in range(6,15+1):
   print(i)
   n4p += ne0 * D_reshaped[:,i] 

# nm  = ne0 * D_reshaped[:,2]
# nr  = ne0 * D_reshaped[:,3] 
# n4p = ne0 * D_reshaped[:,4] 


# electron temp
Te = nee / ne  

Tg = (p_0/spc.k - ne * Te/K_eV) / (np.sum(npop, axis=1) + ni)   # [K]


del D, D_reshaped 








if case2:
   # load solution file
   D = np.load(file2.format(Np))
   D = np.transpose(D)

   # Ns is the number of scpecies
   Ns = 17
   # Ns = 6 #  electrons + ions + nm + nr + n4p + nb 

   # pull solution out of D
   D_reshaped = np.reshape(D,(Np, Ns+1),'F')

   ne_2  = ne0 * D_reshaped[:,0]          # electron density
   ni_2  = ne0 * D_reshaped[:,1]       # ion density
   nb_2  = nAr * D_reshaped[:,Ns - 1] # "background" (argon neutral) density
   nee_2 = (2./3.) * ne0 * D_reshaped[:,Ns]          # electron energy (ne * ee)

   npop_2 = np.ndarray((Np, Ns-2),dtype=np.float64)
   npop_2[:,0] = nb_2
   npop_2[:,1:] = ne0 * D_reshaped[:,2:Ns-1]

   nm_2 = ne0 * (D_reshaped[:,2] + D_reshaped[:,4]) 
   nr_2 = ne0 * (D_reshaped[:,3] + D_reshaped[:,5]) 
   n4p_2 = np.zeros_like(nr)
   for i in range(6,15+1):
      print(i)
      n4p_2 += ne0 * D_reshaped[:,i] 

   # nm_2  = ne0 * D_reshaped[:,2]
   # nr_2  = ne0 * D_reshaped[:,3] 
   # n4p_2 = ne0 * D_reshaped[:,4] 


   # electron temp
   Te_2 = nee_2 / ne_2  

   Tg_2 = (p_0/spc.k - ne_2 * Te_2/K_eV) / (np.sum(npop_2, axis=1) + ni_2)   # [K]


   del D, D_reshaped 





# make some plots


if (isPlot):
   print("Plotting means...")
   
   # ne
   fig,ax = plt.subplots(dpi=160)
   ax.semilogy(xr, ne, clr1, lw=2, label=label1)
   ax.semilogy(xr, ni, clr1+'-', lw=2)
   if case2:
      ax.semilogy(xr, ne_2, clr2, lw=2, label=label2)
      ax.semilogy(xr, ni_2, clr2+'-', lw=2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e,i}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/ne_mean.png')

   # nm
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, nm, clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, nm_2, clr2, lw=2, label=label2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(m)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nm_mean.png')

   # nr
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, nr, clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, nr_2, clr2, lw=2, label=label2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(r)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nr_mean.png')

   # n4p
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, n4p, clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, n4p_2, clr2, lw=2, label=label2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(4p)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/n4p_mean.png')

   # nb
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, nb, clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, nb_2, clr2, lw=2, label=label2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nb_mean.png')


   # Te/phi
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, Te, clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, Te_2, clr2, lw=2, label=label2)
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
   ax.plot(xr, Tg, clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, Tg_2, clr2, lw=2, label=label2)
   # ax.plot(xr, Te/K_eV, clr1+"-", lw=2, label=r"T_e")
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
   plt.savefig('./png/Te_mean.png')


plt.show()
