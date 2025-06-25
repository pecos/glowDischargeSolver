import sys
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline



## Electron Transport Data
transport = h5.File("./nominal_transport.h5", 'r')
NDe_v_Te = transport["diffusivity"]
Te_trans = NDe_v_Te[:,0]
Te_trans /= 11604
De_interp = (NDe_v_Te[:,1]/3.22e22)

idx = np.where(Te_trans <= 1.0)[0][-1]

#De_interp[0:idx] = De_interp[idx]
De_spline = CubicSpline(Te_trans, De_interp)
De_Te_spline = CubicSpline.derivative(De_spline)

Nmue_v_Te = transport["mobility"]
mue_interp = (Nmue_v_Te[:,1]/3.22e22)
#mue_interp[0:idx] = mue_interp[idx]
mue_spline = CubicSpline(Te_trans, mue_interp)
mue_Te_spline = CubicSpline.derivative(mue_spline)

## Energy Transport Data
transport = h5.File("./energy_transport.h5", 'r')
NDE_v_Te = transport["diffusivity"]
Te_Etrans = NDe_v_Te[:,0]
Te_Etrans /= 11604
DE_interp = (NDE_v_Te[:,1]/3.22e22)

idx_E = np.where(Te_Etrans <= 1.0)[0][-1]

#DE_interp[0:idx_E] = DE_interp[idx_E]
DE_spline = CubicSpline(Te_trans, DE_interp)
DE_Te_spline = CubicSpline.derivative(DE_spline)

NmuE_v_Te = transport["mobility"]
muE_interp = (NmuE_v_Te[:,1]/3.22e22)
#muE_interp[0:idx_E] = muE_interp[idx_E]
muE_spline = CubicSpline(Te_Etrans, muE_interp)
muE_Te_spline = CubicSpline.derivative(muE_spline)


diffInterp = np.zeros(10000)
mobInterp = np.zeros(10000)
E_diffInterp = np.zeros(10000)
E_mobInterp = np.zeros(10000)
TePlot = np.linspace(Te_trans[0], Te_trans[-1], diffInterp.shape[0])
for i in range(len(diffInterp)):
    diffInterp[i] = De_spline(TePlot[i])
    mobInterp[i] = mue_spline(TePlot[i])
    E_diffInterp[i] = DE_spline(TePlot[i])
    E_mobInterp[i] = muE_spline(TePlot[i])

fig,ax = plt.subplots()
ax.set_title('Energy Diffusion Coefficient')
ax.set_xlabel('Te [eV]')
ax.plot(TePlot, (5./3.)*diffInterp, color = 'darkviolet', label = '(5/3)*D')
ax.plot(TePlot, (1.2)*diffInterp, color = 'goldenrod', label = '1.2*D')
ax.plot(TePlot, (1.1)*diffInterp, color = 'blue', label = '1.1*D')
ax.plot(TePlot, E_diffInterp, color = 'red', label = 'BOLSIG Data')
ax.plot(TePlot, diffInterp, color = 'green', label = 'Spec. Diffusion')
ax.legend()
ax.set_xlim(0,10)
plt.savefig('E_Diff.png')

fig, ax = plt.subplots()
ax.set_title('Energy Mobility Coefficient')
ax.set_xlabel('Te [eV]')
ax.plot(TePlot, (5./3.)*mobInterp, label = '(5/3)*Mu')
ax.plot(TePlot, E_mobInterp, label = 'BOLSIG Data')
ax.plot(TePlot, mobInterp, label = 'Spec. Mobility')
ax.legend()
ax.set_xlim(0,2)
plt.savefig('E_Mob.png')

fig,ax = plt.subplots()
ax.set_title('Transport Coeff. Ratios')
ax.set_xlabel('Te [eV]')
ax.plot(TePlot, mobInterp/diffInterp, label = 'Eq. Relations')
ax.plot(TePlot, E_mobInterp/E_diffInterp, label = 'BOLSIG Data')
ax.legend()
ax.set_ylabel('$mu$/D')
ax.set_xlim(0,5)
plt.savefig('Trans_Coeff_Ratios.png')
