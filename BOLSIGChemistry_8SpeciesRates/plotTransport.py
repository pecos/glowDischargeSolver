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

De_interp[0:idx] = De_interp[idx]
De_spline = CubicSpline(Te_trans, De_interp)
De_Te_spline = CubicSpline.derivative(De_spline)

Nmue_v_Te = transport["mobility"]
mue_interp = (Nmue_v_Te[:,1]/3.22e22)
mue_interp[0:idx] = mue_interp[idx]
mue_spline = CubicSpline(Te_trans, mue_interp)
mue_Te_spline = CubicSpline.derivative(mue_spline)

diffInterp = np.zeros(10000)
mobInterp = np.zeros(10000)
TePlot = np.linspace(Te_trans[0], Te_trans[-1], diffInterp.shape[0])
for i in range(len(diffInterp)):
    diffInterp[i] = De_spline(TePlot[i])
    mobInterp[i] = mue_spline(TePlot[i])

fig,ax = plt.subplots()
ax.set_title('Diffusion Coefficient')
ax.set_xlabel('Te [eV]')
ax.plot(Te_trans, De_interp, marker = 'o', label = 'Raw Data')
ax.plot(TePlot, diffInterp, label = 'Interpolant')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title('Mobility Coefficient')
ax.set_xlabel('Te [eV]')
ax.plot(Te_trans, mue_interp, marker = 'o', label = 'Raw Data')
ax.plot(TePlot, mobInterp, label = 'Interpolant')
ax.legend()
plt.show()
