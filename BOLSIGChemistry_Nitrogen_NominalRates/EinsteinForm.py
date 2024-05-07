import numpy as np
from matplotlib import pyplot as plt
import h5py as h5

kB = 1.38e-23
q = 1.602e-19

L = 2*0.005
tau = (1./13.56e6)
V0 = 100
nAr = 3.22e22

## Electron Transport Data
N2_transport = h5.File("./nominal_transport.h5", 'r')
N2_NDe_v_Te = N2_transport["diffusivity"]
N2_Te_trans = N2_NDe_v_Te[:,0]
N2_Te_trans /= 11604
N2_De_interp = (N2_NDe_v_Te[:,1]/3.22e22)
#De_spline = CubicSpline(Te_trans, De_interp)
#De_Te_spline = CubicSpline.derivative(De_spline)
N2_Nmue_v_Te = N2_transport["mobility"]
N2_mue_interp = (N2_Nmue_v_Te[:,1]/3.22e22)
#mue_spline = CubicSpline(Te_trans, mue_interp)
#mue_Te_spline = CubicSpline.derivative(mue_spline)
N2_De_Eins = np.multiply(N2_mue_interp,N2_Te_trans)

## Ionization Rate Coefficient
f = h5.File('./Ionization_N2.h5', 'r')
N2_dataset = f["table"]
N2_Te = N2_dataset[:,0]
N2_Te /= 11604
N2_rateCoeff = N2_dataset[:,1]
N2_rateCoeff /= 6.022e23

## Removing BOLSIG failures
N2_fail_inds = []
for j in range(len(N2_rateCoeff)):
    if N2_rateCoeff[j] == 0.0 and j > np.nonzero(N2_rateCoeff)[0][0]:
        N2_fail_inds.append(j)

if len(N2_fail_inds) != 0:
    N2_rateCoeff[0:N2_fail_inds[-1]] = 0.0

Ar_transport = h5.File("../BOLSIGChemistry_NominalRates/nominal_transport.h5", 'r')
Ar_NDe_v_Te = Ar_transport["diffusivity"]
Ar_Te_trans = Ar_NDe_v_Te[:,0]
Ar_Te_trans /= 11604
Ar_De_interp = (Ar_NDe_v_Te[:,1]/3.22e22)
Ar_Nmue_v_Te = Ar_transport["mobility"]
Ar_mue_interp = (Ar_Nmue_v_Te[:,1]/3.22e22)
Ar_De_Eins = np.multiply(Ar_mue_interp, Ar_Te_trans)
## Ionization Rate Coefficient
g = h5.File('../BOLSIGChemistry_NominalRates/Ionization.h5', 'r')
Ar_dataset = g["table"]
Ar_Te = Ar_dataset[:,0]
Ar_Te /= 11604
Ar_rateCoeff = Ar_dataset[:,1]
Ar_rateCoeff /= 6.022e23

## Removing BOLSIG failures
Ar_fail_inds = []
for j in range(len(Ar_rateCoeff)):
    if Ar_rateCoeff[j] == 0.0 and j > np.nonzero(Ar_rateCoeff)[0][0]:
        Ar_fail_inds.append(j)

if len(Ar_fail_inds) != 0:
    Ar_rateCoeff[0:Ar_fail_inds[-1]] = 0.0


#diffInterp = np.zeros(10000)
#mobInterp = np.zeros(10000)
#TePlot = np.linspace(Te_trans[0], Te_trans[-1], diffInterp.shape[0])
#for i in range(len(diffInterp)):
#    diffInterp[i] = De_spline(TePlot[i])
#    mobInterp[i] = mue_spline(TePlot[i])

fig,ax = plt.subplots()
ax.set_title('Diffusion Coefficient (Raw)')
ax.set_xlabel('Te [eV]')
ax.set_ylabel('D [m2/s]')
ax.plot(Ar_Te_trans, Ar_De_interp, label = 'Ar')
ax.plot(N2_Te_trans, N2_De_interp, label = 'N2')
ax.set_xlim(0, 10)
ax.legend()
plt.savefig('Diffusivity_Raw.png')

fig,ax = plt.subplots()
ax.set_title('Mobility Coefficient (Raw)')
ax.set_xlabel('Te [eV]')
ax.set_ylabel('mu [m2/V-s]')
ax.plot(Ar_Te_trans, Ar_mue_interp, label = 'Ar')
ax.plot(N2_Te_trans, N2_mue_interp, label = 'N2')
ax.set_xlim(0, 10)
ax.legend()
plt.savefig('Mobility_Raw.png')

fig,ax = plt.subplots()
ax.set_title('Ionization Rate Coefficient (Raw)')
ax.set_xlabel('Te [eV]')
ax.set_ylabel('k [m3/s]')
ax.plot(Ar_Te, Ar_rateCoeff, label = 'Ar')
ax.plot(N2_Te, N2_rateCoeff, label = 'N2')
ax.legend()
plt.savefig('IonizationRate_Raw.png')

fig,ax = plt.subplots()
ax.set_title('Diffusion Coefficient (Einstein)')
ax.set_xlabel('Te [eV]')
ax.set_ylabel('D [m2/s]')
ax.plot(Ar_Te_trans, Ar_De_Eins, label = 'Ar')
ax.plot(N2_Te_trans, N2_De_Eins, label = 'N2')
ax.set_xlim(0, 10)
ax.set_ylim(0, 1000)
ax.legend()
plt.savefig('Diffusivity_Einstein.png')

N2_mu_star = N2_mue_interp*tau*V0/(L*L)
Ar_mu_star = Ar_mue_interp*tau*V0/(L*L)

N2_DE_star = N2_De_Eins*tau/(L*L)
Ar_DE_star = Ar_De_Eins*tau/(L*L)

N2_k_star = N2_rateCoeff*nAr*tau
Ar_k_star = Ar_rateCoeff*nAr*tau

fig,ax = plt.subplots()
ax.set_title('Non-Dimensional Factors (N2)')
ax.set_xlabel('Te [eV]')
ax.plot(N2_Te_trans, N2_DE_star, label = 'D*')
ax.plot(N2_Te_trans, N2_mu_star, label = 'Mu*')
ax.plot(N2_Te, N2_k_star, label = 'k*')
ax.set_xlim(0, 20)
ax.legend()
plt.savefig('Params_N2.png')


fig,ax = plt.subplots()
ax.set_title('Non-Dimensional Factors (Ar)')
ax.set_xlabel('Te [eV]')
ax.plot(Ar_Te_trans, Ar_DE_star, label = 'D*')
ax.plot(Ar_Te_trans, Ar_mu_star, label = 'Mu*')
ax.plot(Ar_Te, Ar_k_star, label = 'k*')
ax.set_xlim(0, 20)
ax.legend()
plt.savefig('Params_Ar.png')


fig,ax = plt.subplots()
ax.set_title('Non-Dimensional Factors')
ax.set_xlabel('Te [eV]')
ax.plot(Ar_Te_trans, Ar_DE_star, marker = 'o', markerfacecolor = 'None', label = 'Ar - D*')
ax.plot(Ar_Te_trans, Ar_mu_star, marker = 's', markerfacecolor = 'None', label = 'Ar - Mu*')
ax.plot(Ar_Te, Ar_k_star, label = 'Ar - k*')
ax.plot(N2_Te_trans, N2_DE_star, marker = 'o', markerfacecolor = 'None', label = 'N2 - D*')
ax.plot(N2_Te_trans, N2_mu_star, marker = 's', markerfacecolor = 'None', label = 'N2 - Mu*')
ax.plot(N2_Te, N2_k_star, label = 'N2 - k*')
ax.set_xlim(0, 20)
ax.legend()
plt.savefig('Params_Comparison.png')

fig,ax = plt.subplots()
ax.set_title('Non-Dimensional Factors')
ax.set_xlabel('Te [eV]')
ax.plot(Ar_Te_trans, Ar_DE_star, marker = 'o', markerfacecolor = 'None', label = 'Ar - D*')
ax.plot(Ar_Te_trans, Ar_mu_star, marker = 's', markerfacecolor = 'None', label = 'Ar - Mu*')
ax.plot(Ar_Te, Ar_k_star, label = 'Ar - k*')
ax.plot(N2_Te_trans, N2_DE_star, marker = 'o', markerfacecolor = 'None', label = 'N2 - D*')
ax.plot(N2_Te_trans, N2_mu_star, marker = 's', markerfacecolor = 'None', label = 'N2 - Mu*')
ax.plot(N2_Te, N2_k_star, label = 'N2 - k*')
ax.set_xlim(0, 5)
ax.legend()
plt.savefig('Params_ComparisonZoom.png')

fig,ax = plt.subplots()
ax.set_title('Non-Dimensional Factors')
ax.set_xlabel('Te [eV]')
ax.plot(Ar_Te_trans, Ar_DE_star, marker = 'o', markerfacecolor = 'None', label = 'Ar - D*')
ax.plot(Ar_Te_trans, Ar_mu_star, marker = 's', markerfacecolor = 'None', label = 'Ar - Mu*')
ax.plot(Ar_Te, Ar_k_star, label = 'Ar - k*')
ax.plot(N2_Te_trans, N2_DE_star, marker = 'o', markerfacecolor = 'None', label = 'N2 - D*')
ax.plot(N2_Te_trans, N2_mu_star, marker = 's', markerfacecolor = 'None', label = 'N2 - Mu*')
ax.plot(N2_Te, N2_k_star, label = 'N2 - k*')
ax.set_yscale('log')
ax.set_ylim(1e-2, 1e3)
ax.set_xlim(0, 5)
ax.legend()
plt.savefig('Params_ComparisonZoom2.png')


#fig, ax = plt.subplots()
#ax.set_title('Mobility Coefficient')
#ax.set_xlabel('Te [eV]')
#ax.plot(Te_trans, mue_interp, marker = 'o', label = 'Raw Data')
#ax.plot(TePlot, mobInterp, label = 'Interpolant')
#ax.legend()
#plt.show()

