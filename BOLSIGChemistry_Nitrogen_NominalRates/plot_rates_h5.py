import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors
import h5py as h5

f_Ar = h5.File("../BOLSIGChemistry_NominalRates/Ionization.h5", 'r')
f_N2 = h5.File('./Ionization_N2.h5', 'r')
data_Ar = f_Ar["table"]
data_N2 = f_N2["table"]

Te_Ar = data_Ar[:,0]
Te_N2 = data_N2[:,0]
Te_Ar /= 11604
Te_N2 /= 11604
rateCoeff_Ar = data_Ar[:,1]
rateCoeff_N2 = data_N2[:,1]
rateCoeff_Ar /= 6.022e23
rateCoeff_N2 /= 6.022e23


trans_Ar = h5.File('../BOLSIGChemistry_NominalRates/nominal_transport.h5', 'r')
trans_N2 = h5.File('./nominal_transport.h5', 'r')
Nmu_Ar = trans_Ar["mobility"]
Nmu_N2 = trans_N2["mobility"]
Te_trans_Ar = Nmu_Ar[:,0]
Te_trans_N2 = Nmu_N2[:,0]
Te_trans_Ar /= 11604
Te_trans_N2 /= 11605
mobility_Ar = Nmu_Ar[:,1]
mobility_N2 = Nmu_N2[:,1]
ND_Ar = trans_Ar["diffusivity"]
ND_N2 = trans_N2["diffusivity"]
diff_Ar = ND_Ar[:,1]
diff_N2 = ND_N2[:,1]

fig,ax = plt.subplots()
ax.set_title('Ionization Rate Comparison')
ax.set_xlabel('Te [eV]')
ax.set_ylabel('Rate Coefficient [m3/s]')
ax.semilogy(Te_Ar, rateCoeff_Ar, label = 'E + Ar -> 2E + Ar+')
ax.semilogy(Te_N2, rateCoeff_N2, label = 'E + N2 -> 2E + N2+')
ax.legend()
ax.set_ylim(1e-20, 1e-13)
plt.savefig('Ionization_Comparison.png')

fig,ax = plt.subplots()
ax.set_title('Mobility Comparison')
ax.set_xlabel('Te [eV]')
ax.set_ylabel('Mobility*N [1/V-m-s]')
ax.plot(Te_trans_Ar, mobility_Ar, label = 'Ar')
ax.plot(Te_trans_N2, mobility_N2, label = 'N2')
ax.set_yscale('log')
ax.legend()
ax.set_xlim(0, 10)
plt.savefig('Mobility_Comparison.png')

fig,ax = plt.subplots()
ax.set_title('Diffusion Coeff. Comparison')
ax.set_xlabel('Te [eV]')
ax.set_ylabel('Diffusion Coeff.*N [1/m-s]')
ax.plot(Te_trans_Ar, diff_Ar, label = 'Ar')
ax.plot(Te_trans_N2, diff_N2, label = 'N2')
ax.legend()
plt.savefig('Diffusion_Comparison.png')

