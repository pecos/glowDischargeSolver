import numpy as np
import h5py as h5
from matplotlib import pyplot as plt

E_lvl_m = 11.548
E_lvl_r = 11.624
E_lvl_4p = 12.907
E_lvl_i = 15.76
g_m = 5.0
g_r = 3.0
g_4p = 3.0
g_i = 4.0
n_e = 3.7e15

rates14 = []
rates21 = []
rates26 = []
rates29 = []
rates35 = []
data10 = h5.File('./1s-metastable.h5', 'r')["table"]
data11 = h5.File('./1s-resonance.h5', 'r')["table"]
data12 = h5.File('./2p-lumped.h5', 'r')["table"]
data13 = h5.File('./Ionization.h5', 'r')["table"]
data15 = h5.File('./StepIonization.h5', 'r')["table"]

Te = data10[:,0]
rates10 = data10[:,1]
rates11 = data11[:,1]
rates12 = data12[:,1]
rates13 = data13[:,1]
rates15 = data15[:,1]

for j in range(len(Te)):
    rates14.append(rates10[j] * (1 / g_m) * np.exp(E_lvl_m / (Te[j] / 11604)))
    rates21.append(rates11[j] * (1 / g_r) * np.exp(E_lvl_r / (Te[j] / 11604)))
    rates26.append(rates15[j] * (g_m / g_i) * np.exp((E_lvl_i - E_lvl_m) / (Te[j] / 11604)) / n_e * 6.022e23)
    rates29.append(rates13[j] * (1 / g_i) * np.exp(E_lvl_i / (Te[j] / 11604)) / n_e * 6.022e23)
    rates35.append(rates12[j] * (1 / g_4p) * np.exp(E_lvl_4p / (Te[j] / 11604)))


data14 = np.empty(shape = (len(Te), 2))
data21 = np.empty(shape = (len(Te), 2))
data26 = np.empty(shape = (len(Te), 2))
data29 = np.empty(shape = (len(Te), 2))
data35 = np.empty(shape = (len(Te), 2))

data14[:,0] = Te
data14[:,1] = rates14
data21[:,0] = Te
data21[:,1] = rates21
data26[:,0] = Te
data26[:,1] = rates26
data29[:,0] = Te
data29[:,1] = rates29
data35[:,0] = Te
data35[:,1] = rates35

with h5.File('./Deexci-metastable.h5', 'w') as f:
    dataset = f.create_dataset("table", data = data14)

with h5.File('./Deexci-resonance.h5', 'w') as g:
    dataset = g.create_dataset("table", data = data21)

with h5.File('./3BdyRecomb-ground.h5', 'w') as h:
    dataset = h.create_dataset("table", data = data29)

with h5.File('./3BdyRecomb-metastable.h5', 'w') as x:
    dataset = x.create_dataset("table", data = data26)

with h5.File('./Deexci-2p.h5', 'w') as y:
    dataset = y.create_dataset("table", data = data35)

#rxn10 = h5.File("./BOLSIGChemistry_6SpeciesRates/lumped.metastable.h5", 'r')
#rxn11 = h5.File("./BOLSIGChemistry_6SpeciesRates/lumped.resonance.h5", 'r')

#data10 = rxn10["table"]
#data11 = rxn11["table"]

#Te = data10[:,0]
#Te /= 11604
#rates10 = data10[:,1]
#rates11 = data11[:,1]
#rates10 /= 6.022e23
#rates11 /= 6.022e23

#rates14 = []
#rates21 = []

#for i in range(len(rates10)):
#    rates14.append(rates10[i] * (1 / g_m) * np.exp(E_lvl_m / Te[i]))
#    rates21.append(rates11[i] * (1 / g_r) * np.exp(E_lvl_r / Te[i]))


#arrh_14 = []
#arrh_21 = []

#for j in range(len(Te)):
#    arrh_14.append(4.3e-16 * Te[j]**0.74)
#    arrh_21.append(4.3e-16 * Te[j]**0.74)


#fig,ax = plt.subplots()
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.plot(Te, rates14, color = 'darkblue', label = 'CRSC - 14')
#ax.plot(Te, arrh_14, color = 'cyan', label = 'Arrhenius - 14')
#ax.plot(Te, rates21, color = 'darkred', label = 'CRSC - 21')
#ax.plot(Te, arrh_21, color = 'orange', label = 'Arrhenius - 21')
#ax.legend()
#ax.set_xlabel('Te [eV]')
#ax.set_ylabel('Rate [# / m3-s]')
#plt.savefig('RatesComparison.png')


#for i in range(len(rates14)):
#    rates14[i] *= 6.022e23
#    rates21[i] *= 6.022e23
#    Te[i] *= 11604

#data14 = np.empty(shape = (len(Te), 2))
#data21 = np.empty(shape = (len(Te), 2))

#data14[:,0] = Te
#data21[:,0] = Te
#data14[:,1] = rates14
#data21[:,1] = rates21

#with h5.File('./BOLSIGChemistry_6SpeciesRates/deexci.metastable.h5', 'w') as f:
#    dataset = f.create_dataset("table", data = data14)

#with h5.File('./BOLSIGChemistry_6SpeciesRates/deexci.resonance.h5', 'w') as g:
#    dataset = g.create_dataset("table", data = data21)



