import numpy as np
from matplotlib import pyplot as plt
import h5py as h5

kB = 1.38e-23
q = 1.602e-19

## Electron Transport Data
transport = h5.File("./nominal_transport.h5", 'r')
NDe_v_Te = transport["diffusivity"]
Te_trans = NDe_v_Te[:,0]
Te_trans /= 11604
De_interp = (NDe_v_Te[:,1]/3.22e22)
#De_spline = CubicSpline(Te_trans, De_interp)
#De_Te_spline = CubicSpline.derivative(De_spline)

Nmue_v_Te = transport["mobility"]
mue_interp = (Nmue_v_Te[:,1]/3.22e22)
#mue_spline = CubicSpline(Te_trans, mue_interp)
#mue_Te_spline = CubicSpline.derivative(mue_spline)

De_Eins = np.multiply(mue_interp,Te_trans)


## Electron Transport Data 2
# transport_2 = h5.File("../BOLSIGChemistry_Transport/nominal_transport_2.h5", 'r')
# NDe_v_Te_2 = transport["diffusivity"]
# Te_trans_2 = NDe_v_Te_2[:,0]
# Te_trans_2 /= 11604
# De_interp_2 = (NDe_v_Te_2[:,1]/3.22e22)
# Nmue_v_Te_2 = transport["mobility"]
# mue_interp_2 = (Nmue_v_Te_2[:,1]/3.22e22)
# De_Eins_2 = np.multiply(mue_interp_2,Te_trans_2)


#diffInterp = np.zeros(10000)
#mobInterp = np.zeros(10000)
#TePlot = np.linspace(Te_trans[0], Te_trans[-1], diffInterp.shape[0])
#for i in range(len(diffInterp)):
#    diffInterp[i] = De_spline(TePlot[i])
#    mobInterp[i] = mue_spline(TePlot[i])

fig,ax = plt.subplots()
ax.set_title('Diffusion Coefficient')
ax.set_xlabel('Te [eV]')
ax.set_ylabel('D [m2/s]')
ax.plot(Te_trans, De_interp, marker = 'o', label = 'Bolsig Data')
ax.plot(Te_trans, De_Eins, label = 'Einstein Relation')
# ax.plot(Te_trans_2, De_interp_2, marker = '.', label = 'Bolsig Data')
# ax.plot(Te_trans_2, De_Eins_2, label = 'Einstein Relation')
ax.legend()
# plt.savefig('EinsteinRelation.png')

# fig, ax = plt.subplots()
# ax.set_title('Mobility Coefficient')
# ax.set_xlabel('Te [eV]')
# ax.plot(Te_trans, mue_interp, marker = 'o', label = 'Raw Data')
# # ax.plot(Te_trans_2, mue_interp_2, marker = '.', label = 'Raw Data')
# # ax.plot(TePlot, mobInterp, label = 'Interpolant')
# ax.legend()


plt.show()

