import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.constants as spc


# sys.path.insert(0, '../')

# Flags
isPlotLines = False
isPlotMeans = True

case1 = True; file1 = '../fullsoln.npy'; clr1 = 'b-'; label1 = "CR"

# case1 = True; file1 = '../Results/CR/fullsoln.npy'; clr1 = 'b-'; label1 = "CR"
# case2 = True; file2 = '../Results/6spec/fullsoln.npy'; clr2 = 'r-'; label2 = "6sp"
case2 = False; file2 = '../Results/fullsoln.npy'; clr2 = 'r-'; label2 = "CR_2"

# these values are required to "redimensionalize" the results
# they must be consistent with the scenario input file
Pressure  = 1.0*spc.torr               # [Pa] 
GasTemperature = 300.0                 # [K]
nAr = Pressure/GasTemperature/spc.k    # [#/m^3] Number density based on bulk temperature (not necessarily true density in two-temperature gas)

ne0 = 8e16           # "nominal" electron density [1/m^3]
L   = 2.00*0.005     # half-gap-width [m] (gap width is 2 cm)
tau = (1./13.56e6)   # period of driving voltage [s]
V0  = 100.0          # amplitude of driving voltage [V]


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

ne  = D_reshaped[:,0,:]          # electron density
ni  = D_reshaped[:,1,:]       # ion density
nb  = D_reshaped[:,Ns - 1,:] # "background" (argon neutral) density
nee = D_reshaped[:,Ns,:]          # electron energy (ne * ee)

npop = np.ndarray((Np, Ns-2, np.shape(D)[1]),dtype=np.float64)
npop[:,0,:] = nb
npop[:,1:,:] = D_reshaped[:,2:Ns-1,:]

nm = D_reshaped[:,2,:] + D_reshaped[:,4,:] 
nr = D_reshaped[:,3,:] + D_reshaped[:,5,:] 
n4p = np.zeros_like(nr)
for i in range(6,15+1):
   print(i)
   n4p += D_reshaped[:,i,:] 

# electron temp
Te = (2./3.) * nee / ne  

print("ne: {:2E}".format(ne0*np.mean(ne, axis=1)[74]))
print("n4p: {:2E}".format(ne0*np.mean(n4p, axis=1)[74]))

del D, D_reshaped 


if case2: 
   # load solution file
   D2 = np.load(file2.format(Np))
   D2 = np.transpose(D2)

   # Ns is the number of scpecies
   Ns = 17 #  electrons + ions + nm + nr + n4p + nb 

   # pull solution out of D
   D_reshaped = np.reshape(D2,(Np, Ns+1, np.shape(D2)[1]),'F')

   ne_2  = D_reshaped[:,0,:]       # electron density
   ni_2  = D_reshaped[:,1,:]       # ion density
   nb_2  = D_reshaped[:,Ns - 1,:]  # "background" (argon neutral) density
   nee_2 = D_reshaped[:,Ns,:]      # electron energy (ne * ee)

   npop_2 = np.ndarray((Np, Ns-2, np.shape(D2)[1]),dtype=np.float64)
   npop_2[:,0,:] = nb_2
   npop_2[:,1:,:] = D_reshaped[:,2:Ns-1,:]

   # nm_2 = D_reshaped[:,2,:]
   # nr_2 = D_reshaped[:,3,:] 
   # n4p_2 = D_reshaped[:,4,:] 

   nm_2 = D_reshaped[:,2,:] + D_reshaped[:,4,:] 
   nr_2 = D_reshaped[:,3,:] + D_reshaped[:,5,:] 
   n4p_2 = np.zeros_like(nr_2)
   for i in range(6,15+1):
      print(i)
      n4p_2 += D_reshaped[:,i,:] 
   
   # electron temp
   Te_2 = (2./3.) * nee_2 / ne_2  

   print("ne_2: {:2E}".format(ne0*np.mean(ne_2, axis=1)[74]))
   print("n4p_2: {:2E}".format(ne0*np.mean(n4p_2, axis=1)[74]))

   del D2, D_reshaped 
   


# make some plots

'''
# ne
fig,ax = plt.subplots(dpi=160)
h = ax.contourf(xr, tr, np.transpose(ne)*ne0, levels=np.linspace(0,ne0,129))
cbar = plt.colorbar(h, ticks=np.linspace(0,ne0,9))
cbar.ax.set_ylabel(r"$n_e$ [m$^{-3}$]", labelpad=6, fontsize=18)
plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
ax.contour(xr, tr, np.transpose(ne)*ne0, levels=np.linspace(0,ne0,129))
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

   print("Plotting lines...")
   # ne
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, ne0*ne[:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, ne0*ne[:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, ne0*ne[:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, ne0*ne[:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.plot(xr, ne0*ni[:,t0], 'b--', lw=2)
   ax.plot(xr, ne0*ni[:,t1], 'r--', lw=2)
   ax.plot(xr, ne0*ni[:,t2], 'g--', lw=2)
   ax.plot(xr, ne0*ni[:,t3], 'c--', lw=2)
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e,i} $   [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/ne_line.png')

   # nm
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, ne0*nm[:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, ne0*nm[:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, ne0*nm[:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, ne0*nm[:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_m$  [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nm_line.png')

   # nb
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, nAr*nb[:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, nAr*nb[:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, nAr*nb[:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, nAr*nb[:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
   ax.legend()
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_B$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nb_line.png')

   # Te/phi
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, Te[:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
   ax.plot(xr, Te[:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
   ax.plot(xr, Te[:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
   ax.plot(xr, Te[:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
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
   ax.semilogy(xr, ne0*np.mean(ne,axis=1), clr1, lw=2, label=label1)
   ax.semilogy(xr, ne0*np.mean(ni,axis=1), clr1+'-', lw=2)
   if case2:
      ax.semilogy(xr, ne0*np.mean(ne_2,axis=1), clr2, lw=2, label=label2)
      ax.semilogy(xr, ne0*np.mean(ni_2,axis=1), clr2+'-', lw=2)   
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{e,i}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/ne_mean.png')

   # nm
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, ne0*np.mean(nm,axis=1), clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, ne0*np.mean(nm_2,axis=1), clr2, lw=2, label=label2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(m)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nm_mean.png')

   # nr
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, ne0*np.mean(nr,axis=1), clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, ne0*np.mean(nr_2,axis=1), clr2, lw=2, label=label2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(r)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nr_mean.png')

   # n4p
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, ne0*np.mean(n4p,axis=1), clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, ne0*np.mean(n4p_2,axis=1), clr2, lw=2, label=label2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR(4p)}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/n4p_mean.png')

   # nb
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, nAr*np.mean(nb,axis=1), clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, nAr*np.mean(nb_2,axis=1), clr2, lw=2, label=label2)
   ax.legend(fontsize=12)
   ax.set_xlim((xr[0], xr[-1]))
   ax.set_xlabel(r"$x$ [cm]", fontsize=18)
   plt.setp(ax.get_xticklabels(), fontsize=12)
   ax.set_ylabel(r"$n_{AR}$ [m$^{-3}$]", fontsize=18)
   plt.setp(ax.get_yticklabels(), fontsize=12)
   plt.savefig('./png/nb_mean.png')


   # Te/phi
   fig,ax = plt.subplots(dpi=160)
   ax.plot(xr, np.mean(Te,axis=1), clr1, lw=2, label=label1)
   if case2:
      ax.plot(xr, np.mean(Te_2,axis=1), clr2, lw=2, label=label2)
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



plt.show()
