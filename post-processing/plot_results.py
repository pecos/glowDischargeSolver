import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as spc


# sys.path.insert(0, '../')
# Constants
K_eV = spc.k/spc.e             # Convert energy units: from K to eV

# Flags
isPlotLines = False
isPlotMeans = True

case = {}; file = {}; clr = {}; label = {}; model = {}



# ic = 1; c = True; f = '../fullsoln.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# ic = 1; c = True; f = '../Results/CR/CR_Np150_fullsoln.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# ic = 1; c = True; f = '../Results/test/1/fullsoln.npy'; cl = 'b-'; lb = "CR"; m = "CR"
# ic = 1; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T2000.npy'; cl = 'b'; lb = "CR"; m = "CR"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

# ic = 2; c = False; f = '../Results/6spec/nominalCase_V100_P1torr_Np150/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'g'; lb = "6sp"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 3; c = True; f = '../Results/6spec/nominalCase_V100_P1torr_Np150_TimeMarching/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'm'; lb = "6sp - Time Marching"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
ic = 2; c = False; f = '../Results/6spec/nominalCase_V100_P1torr_Np150_constDiff/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'r'; lb = "6sp - constDiff"; m = "6sp"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 
# ic = 5; c = False; f = '../Results/6spec/nominalCase_V100_P1torr_Np150_dEps/newton_6spec_CN_Np150_fullsoln.npy'; cl = 'k'; lb = "6sp - dEps"; m = "6sp"
# case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 



ic = 3; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T2000.npy'; cl = 'b-'; lb = "T2000"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 4; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T1750.npy'; cl = 'r-'; lb = "T1750"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 5; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T1500.npy'; cl = 'g-'; lb = "T1500"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 6; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T1250.npy'; cl = 'm-'; lb = "T1250"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 7; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T2250.npy'; cl = 'm-'; lb = "T2250"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 8; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T2500.npy'; cl = 'm-'; lb = "T2500"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 

ic = 9; c = True; f = '../Results/CR/1Torr_100V/Maxwellian/fullsoln/CR_Np150_fullsoln_T2750.npy'; cl = 'm-'; lb = "T2750"; m = "CR"
case[ic] = c; file[ic] = f; clr[ic] = cl; label[ic] = lb; model[ic] = m 




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


## Species energies and degeneracies
# E, AR+, AR(m), AR(r), AR(4p), AR
dEps_6sp = np.array([0.0,15.76,11.577,11.725,13.168,0.0]) 
g_6sp = np.array([1, 4, 6, 6, 36, 1])

# electrons + ions + 4 4s levels + 10 4p levels + background state
dEps_CR = np.array([ 0.0,         15.7596119,  11.54835442, 11.62359272, 11.72316039, 11.82807116,
                     12.9070153,  13.07571571, 13.09487256, 13.15314387, 13.1717777,  13.2730381,
                     13.28263902, 13.30222747, 13.32785705, 13.47988682,  0.0]) 
g_CR = np.array([1, 4, 5, 3, 1, 3, 3, 7, 5, 3, 5, 1, 3, 5, 3, 1, 1])



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
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e,i}$ [m$^{-3}$]", fontsize=18)
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


   # Te/phi
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.plot(xr, np.mean(Te[ic],axis=1), lw=2, label=label[ic])
   ax.legend(fontsize=12,loc=2)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylim((0,5.5))
   ax.set_ylabel(r"$T_e$ [eV]", fontsize=18)
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

   # npop
   fig,ax = plt.subplots(dpi=160)
   for ic in case: 
      if case[ic]: 
         ax.scatter(dEps[ic][0:-2], np.mean(npop[ic][i_mid,:,:],axis=1)/g[ic][0:-2], lw=2, label=label[ic])
         ax.plot(dEps[ic][-1], np.mean(ne[ic][i_mid,:],axis=0),'*', lw=2)
         print("Ionization degree = ", round(np.mean(ne[ic][i_mid,:],axis=0)/np.mean(npop[ic][i_mid,0,:],axis=0)*100,7), " %")      
   ax.legend(fontsize=12,loc=2)
   ax.semilogy()
   # ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$E$ [eV]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   # ax.set_ylim((0,5.5))
   ax.set_ylabel(r"$n / g$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/Distribution_mean.png')


# ax.scatter(E_lvl*cm_eV,npop_LTE/g_lvl,c='b',label='LTE Boltzmann Ar(I)')
# ax.plot(Eion,n_e_LTE,'r*',label='Ne')

# plt.xlabel('E [ev]')
# plt.ylabel('n [m$^{-3}]$')



plt.show()
