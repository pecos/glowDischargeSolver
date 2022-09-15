import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
#import chebSolver as cs
import numpy.polynomial.chebyshev as cheb
import matplotlib
import matplotlib.font_manager
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

sys.path.insert(0, '../')
import chebSolver as cs

# these values are required to "redimensionalize" the results
# they must be consistent with the scenario input file
ne0 = 8e16 # "nominal" electron density [1/m^3]
L   = 2.00*0.005  # half-gap-width [m] (gap width is 1in)
tau = (1./13.56e6) # period of driving voltage [s]
V0 = 100.0

# Number of Chebyshev modes
Np=150

# load solution file
D = np.load('newton_4spec_CN_Np150_fullsoln.npy')
D = np.transpose(D)

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
xr = (xp+1)*L

# pull solution out of D
ne = D[0:Np,:]        # electron density
ni = D[Np:2*Np,:]     # ion density
nm = D[2*Np:3*Np,:]   # metastable density
nb = D[3*Np:4*Np,:]   # "background" (argon neutral) density
nee = D[4*Np:,:]      # electron energy (ne * ee)


# (3/2)*electron temp
Te = D[4*Np:,:]/ne


# evaluate electric potential (this requires a poisson solve, which is
# why the timeDomainCollocationSolver is loaded)
#phi = np.zeros((Np,T+1))
#tds = cs.timeDomainCollocationSolver(4,1,Np,scenario=6)
#print("Solving poisson...")
#for i in range(0,T+1):
#    tds.solve_poisson(ne[:,i],ni[:,i],tp[i])
#    phi[:,i] = tds.phi[:]

# make some plots

# ne
fig,ax = plt.subplots()
h = ax.contourf(xr, tr, np.transpose(ne)*ne0, levels=np.linspace(0,ne0,129))
cbar = plt.colorbar(h, ticks=np.linspace(0,ne0,9))
cbar.ax.set_ylabel(r"$n_e$ [m$^{-3}$]", labelpad=6, fontsize=18)
plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
ax.contour(xr, tr, np.transpose(ne)*ne0, levels=np.linspace(0,ne0,129))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel(r"$t$ [s]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('ne_4spec_contour.pdf')

# ni
fig,ax = plt.subplots()
h = ax.contourf(xr, tr, np.transpose(ni)*ne0, levels=np.linspace(0,ne0,129))
cbar = plt.colorbar(h, ticks=np.linspace(0,ne0,9))
cbar.ax.set_ylabel(r"$n_i$ [m$^{-3}$]", labelpad=6, fontsize=18)
plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
ax.contour(xr, tr, np.transpose(ni)*ne0, levels=np.linspace(0,ne0,129))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel(r"$t$ [s]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('ni_4spec_contour.pdf')

# nm
fig,ax = plt.subplots()
h = ax.contourf(xr, tr, np.transpose(nm)*ne0, levels=np.linspace(0,5*ne0,129))
cbar = plt.colorbar(h, ticks=np.linspace(0,ne0,9))
cbar.ax.set_ylabel(r"$n^{\ast}$ [m$^{-3}$]", labelpad=6, fontsize=18)
plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
ax.contour(xr, tr, np.transpose(ni)*ne0, levels=np.linspace(0,ne0,129))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel(r"$t$ [s]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('nm_4spec_contour.pdf')

# Te
fig,ax = plt.subplots()
h = ax.contourf(xr, tr, np.transpose(Te)*(2./3.), levels=np.linspace(0,16,129))
cbar = plt.colorbar(h, ticks=np.linspace(0,16,5))
cbar.ax.set_ylabel(r"$T_e$ [eV]", labelpad=6, fontsize=18)
plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
ax.contour(xr, tr, np.transpose(Te)*(2./3.), levels=np.linspace(0,16,129))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel(r"$t$ [s]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('Te_4spec_contour.pdf')

# nb
fig,ax = plt.subplots()
h = ax.contourf(xr, tr, np.transpose(nb)*ne0, levels=np.linspace(0,5*ne0,129))
cbar = plt.colorbar(h, ticks=np.linspace(0,ne0,9))
cbar.ax.set_ylabel(r"$n_b$ [m$^{-3}$]", labelpad=6,fontsize=18)
plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
ax.contour(xr, tr, np.transpose(nb)*ne0, levels=np.linspace(0,ne0,129))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel(r"$t$ [s]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('nb_4spec_contour.pdf')

# phi
#fig,ax = plt.subplots()
#h = ax.contourf(xr, tr, np.transpose(phi)*V0, levels=np.linspace(-120,120,129))
#cbar = plt.colorbar(h, ticks=np.linspace(-120,120,13))
#cbar.ax.set_ylabel(r"$\phi$ [V]", labelpad=6, fontsize=18)
#plt.setp(cbar.ax.get_yticklabels(), rotation='horizontal', fontsize=12)
#ax.contour(xr, tr, np.transpose(phi)*V0, levels=np.linspace(-120,120,129))
#ax.set_xlabel(r"$x$ [m]", fontsize=18)
#plt.setp(ax.get_xticklabels(), fontsize=12)
#ax.set_ylabel(r"$t$ [s]", fontsize=18)
#plt.setp(ax.get_yticklabels(), fontsize=12)
#plt.savefig('phi_4spec_contour.pdf')

print("Plotting lines...")

# ne
fig,ax = plt.subplots()
ax.plot(xr, ne0*ne[:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
ax.plot(xr, ne0*ne[:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
ax.plot(xr, ne0*ne[:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
ax.plot(xr, ne0*ne[:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
ax.plot(xr, ne0*ni[:,t0], 'b--', lw=2)
ax.plot(xr, ne0*ni[:,t1], 'r--', lw=2)
ax.plot(xr, ne0*ni[:,t2], 'g--', lw=2)
ax.plot(xr, ne0*ni[:,t3], 'c--', lw=2)
ax.plot(xr, ne0*nm[:,t0], 'b:', lw=2)
ax.plot(xr, ne0*nm[:,t1], 'r:', lw=2)
ax.plot(xr, ne0*nm[:,t2], 'g:', lw=2)
ax.plot(xr, ne0*nm[:,t3], 'c:', lw=2)
ax.legend()
ax.set_xlim((0,0.0254))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylabel(r"Density [m$^{-3}$]", fontsize=18)
#ax.set_ylim((0,8e16))
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('density_4spec_line.pdf')

# Te/phi
fig,ax = plt.subplots()
ax.plot(xr, (2./3.)*Te[:,t0], 'b-', lw=2, label=r"t={0:.2f}T".format(tp[t0]))
ax.plot(xr, (2./3.)*Te[:,t1], 'r-', lw=2, label=r"t={0:.2f}T".format(tp[t1]))
ax.plot(xr, (2./3.)*Te[:,t2], 'g-', lw=2, label=r"t={0:.2f}T".format(tp[t2]))
ax.plot(xr, (2./3.)*Te[:,t3], 'c-', lw=2, label=r"t={0:.2f}T".format(tp[t3]))
ax.set_xlim((0,0.0254))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
ax.set_ylim((0,30))
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
plt.savefig('Te_4spec_line.pdf')

print("Plotting means...")

# ne
fig,ax = plt.subplots()
ax.semilogy(xr, ne0*np.mean(ne,axis=1), 'b-', lw=2, label=r"$n_e$")
ax.semilogy(xr, ne0*np.mean(ni,axis=1), 'r--', lw=2, label=r"$n_i$")
#ax.plot(xr, ne0*np.mean(nm,axis=1), 'g:', lw=2, label=r"$n^{\ast}$")
ax.legend(fontsize=12)
#ax.set_xlim((0,0.0254))
ax.set_xlim((0,0.02))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
#ax.set_ylim((0,8e16))
ax.set_ylabel(r"Density [m$^{-3}$]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('density_4spec_mean.pdf')

# nm
fig,ax = plt.subplots()
ax.plot(xr, ne0*np.mean(nm,axis=1), 'g-', lw=2, label=r"$n^{\ast}$")
ax.legend(fontsize=12)
ax.set_xlim((0,0.0254))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
#ax.set_ylim((0,4*8e16))
ax.set_ylabel(r"Density [m$^{-3}$]", fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('metastable_4spec_mean.pdf')

# Te/phi
fig,ax = plt.subplots()
ax.plot(xr, (2./3.)*np.mean(Te,axis=1), 'b-', lw=2, label=r"$T_e$")
ax.legend(fontsize=12,loc=2)
ax.set_xlim((0,0.0254))
ax.set_xlabel(r"$x$ [m]", fontsize=18)
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
plt.savefig('Te_4spec_mean.pdf')

# Write CSV File
header = ['x', 'ne', 'ni', 'nm', 'nb', 'Te']
data = []
row = []
for i in range(len(xr)):
   x_temp = xr[i]
   ne_temp = ne0*np.mean(ne,axis=1)[i]
   ni_temp = ne0*np.mean(ni,axis=1)[i]
   nm_temp = ne0*np.mean(nm,axis=1)[i]
   nb_temp = ne0*np.mean(nb,axis=1)[i]
   Te_temp = (2./3.)*11604.*np.mean(Te,axis=1)[i]
   row.append(x_temp)
   row.append(ne_temp)
   row.append(ni_temp)
   row.append(nm_temp)
   row.append(nb_temp)
   row.append(Te_temp)
   data.append(row)
   row = []

f = open('GD_mean_results.csv', 'w')
writer = csv.writer(f)
writer.writerow(header)
writer.writerows(data)
f.close()
