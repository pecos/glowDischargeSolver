import numpy as np
from matplotlib import pyplot as plt
import h5py as h5

import pandas as pd
import scipy.constants as spc

# from EquilibriumCalc import PartitionFunctions, BoltzmannDistribution

## Fontsize
tfs = 16
afs = 14


## Constants
qe = 1.60217663e-19 # C
kB = 1.380649e-23   # J/K
m_e = 9.1e-31       # kg
h = 6.62e-34        # J-s
E_lvl_i = 15.76     # eV
NA = 6.022e23       # #/mol
eV = qe/kB

cm_eV = spc.h*spc.c/spc.e*100  # Convert energy units: from cm^-1 to eV
K_eV = spc.k/spc.e             # Convert energy units: from K to eV


## Non-dimentionilization 
nAr = 3.22e22     # background number density of Ar [1/m^3] (corresponds to p = 1 Torr)
np0 = 8e16        # "nominal" electron density [1/m^3]


# Transport Parameters
nmue = 9.66e21   # argon number density times electron mobility [1/(V*cm*s)]
#nmui = 8.0e19
nmui = 4.65e19   # Transport coefficients from Lymberopoulos & Economou, 1993
nDe  = 3.86e22   # argon number density times electron diffusivity [1/(cm*s)]
nDi  = 2.07e18   # argon number density times ion diffusivity [1/(cm*s)]
nDm  = 2.42e18   # argon number density times AR(m) diffusivity [1/(cm*s)]

# 1) Convert input units to base SI (except eV)
nmue *= 100. # 1/(V*m*s)
nmui *= 100. # 1/(V*m*s)
nDe  *= 100. # 1/(m*s)
nDi  *= 100. # 1/(m*s)
nDm  *= 100.

# 2) Compute "raw" transport parameters
mue_ct = nmue/nAr
mui_ct = nmui/nAr
De_ct  = nDe/nAr
Di_ct  = nDi/nAr
Dm_ct = nDm/nAr





## Path to files
file_path = 'glow-discharge/CR/1Torr_100V/Transport/'
file_path_EEDF = 'glow-discharge/CR/1Torr_100V/EEDF/'

## Electron Transport Data
df = h5.File(file_path + "./nominal_transport.h5", 'r')
Te_trans = df["diffusivity"][:,0] / eV
De_interp = df["diffusivity"][:,1] / nAr
mue_interp = df["mobility"][:,1] / nAr



## Electron Transport Data
energy = {}; Td = {}; mu = {}; De = {}; mu_e = {}; De_e = {}
label = {}; clr = {}


df = h5.File(file_path + "./transport_test.h5", 'r')
ic = 1; energy[ic] = df["diffusivity"][:,0]/eV; 
mu[ic] = df["mobility"][:,1]; De[ic] = df["diffusivity"][:,1]; 
mu_e[ic] = df["energy_mobility"][:,1]; De_e[ic] = df["energy_diffusivity"][:,1]
clr[ic] = 'b'; label[ic] = 'test'

df = h5.File(file_path + "./transport_BSR.h5", 'r')
ic = 2; energy[ic] = df["diffusivity"][:,0]/eV; 
mu[ic] = df["mobility"][:,1]; De[ic] = df["diffusivity"][:,1]; 
mu_e[ic] = df["energy_mobility"][:,1]; De_e[ic] = df["energy_diffusivity"][:,1]
clr[ic] = 'b'; label[ic] = 'BSR'

df = h5.File(file_path + "./transport_BSR_2.h5", 'r')
ic = 3; energy[ic] = df["diffusivity"][:,0]/eV; 
mu[ic] = df["mobility"][:,1]; De[ic] = df["diffusivity"][:,1]; 
mu_e[ic] = df["energy_mobility"][:,1]; De_e[ic] = df["energy_diffusivity"][:,1]
clr[ic] = 'b'; label[ic] = 'BSR 2'


# df = h5.File(file_path + "./transport_BSR_Kevin.h5", 'r')
# ic = 3; energy[ic] = df["diffusivity"][:,0]/eV; 
# mu[ic] = df["mobility"][:,1]; De[ic] = df["diffusivity"][:,1]; 
# mu_e[ic] = df["energy_mobility"][:,1]; De_e[ic] = df["energy_diffusivity"][:,1]
# clr[ic] = 'b'; label[ic] = 'BSR (Kevin)'

# df = h5.File(file_path + "./transport_LXCat-June2013.h5", 'r')
# ic = 3; energy[ic] = df["diffusivity"][:,0]/eV; 
# mu[ic] = df["mobility"][:,1]; De[ic] = df["diffusivity"][:,1]; 
# mu_e[ic] = df["energy_mobility"][:,1]; De_e[ic] = df["energy_diffusivity"][:,1]
# clr[ic] = 'b'; label[ic] = 'LXCat-June2013'


# df = h5.File(file_path + "./transport_test_2.h5", 'r')
# ic = 4; energy[ic] = df["diffusivity"][:,0]/eV; 
# mu[ic] = df["mobility"][:,1]; De[ic] = df["diffusivity"][:,1]; 
# mu_e[ic] = df["energy_mobility"][:,1]; De_e[ic] = df["energy_diffusivity"][:,1]
# clr[ic] = 'b'; label[ic] = 'test 2'

# df = h5.File(file_path + "./transport_BSR_2.h5", 'r')
# ic = 5; energy[ic] = df["diffusivity"][:,0]/eV; 
# mu[ic] = df["mobility"][:,1]; De[ic] = df["diffusivity"][:,1]; 
# mu_e[ic] = df["energy_mobility"][:,1]; De_e[ic] = df["energy_diffusivity"][:,1]
# clr[ic] = 'b'; label[ic] = 'BSR 2'


# df = h5.File(file_path + "./transport_test_3.h5", 'r')
# ic = 5; energy[ic] = df["diffusivity"][:,0]/eV; 
# mu[ic] = df["mobility"][:,1]; De[ic] = df["diffusivity"][:,1]; 
# mu_e[ic] = df["energy_mobility"][:,1]; De_e[ic] = df["energy_diffusivity"][:,1]
# clr[ic] = 'b'; label[ic] = 'test 3'

# # Read the specified sheet into a pandas DataFrame
# df = pd.read_excel(file_path+'transport_data.xlsx', sheet_name='nominal')
# ic = 3; energy[ic] = df['Energy (eV)']/1.5; Td[ic] = df['Electric field / N (Td)']; 
# mu[ic] = df['Mobility *N (1/m/V/s)']; De[ic] = df['Diffusion coefficient *N (1/m/s)']; 
# mu_e[ic] = df['Energy mobility *N (1/m/V/s)']; De_e[ic] = df['Energy diffusion coef. D*N (1/m/s)']
# clr[ic] = 'b'; label[ic] = 'nominal'

# df = pd.read_excel(file_path+'transport_data.xlsx', sheet_name='Sheet4')
# ic = 4; energy[ic] = df['Energy (eV)']/1.5; Td[ic] = df['Electric field / N (Td)']; 
# mu[ic] = df['Mobility *N (1/m/V/s)']; De[ic] = df['Diffusion coefficient *N (1/m/s)']; 
# mu_e[ic] = df['Energy mobility *N (1/m/V/s)']; De_e[ic] = df['Energy diffusion coef. D*N (1/m/s)']
# clr[ic] = 'r'; label[ic] = "high"

# df = pd.read_excel(file_path+'transport_data.xlsx', sheet_name='Sheet5')
# ic = 5; energy[ic] = df['Energy (eV)']/1.5; Td[ic] = df['Electric field / N (Td)']; 
# mu[ic] = df['Mobility *N (1/m/V/s)']; De[ic] = df['Diffusion coefficient *N (1/m/s)']; 
# mu_e[ic] = df['Energy mobility *N (1/m/V/s)']; De_e[ic] = df['Energy diffusion coef. D*N (1/m/s)']
# clr[ic] = 'r'; label[ic] = "low"





fig,ax = plt.subplots()
ax.set_title('Diffusion Coef.')
ax.set_xlabel('Te [eV]')
ax.set_ylabel(r"$D_e \, $ [$ \, m^{2}/s$]", fontsize=afs)
ax.plot(Te_trans, De_interp, marker = 'o', label = 'Nominal')

for ic in energy: 
    ax.plot(energy[ic], De[ic]/nAr, marker = 'o', label=label[ic])
    # De_Eins = np.multiply(mu[ic]/nAr,energy[ic])
    # ax.plot(energy[ic], De_Eins, label =label[ic]+' Ein. Relation')

plt.axhline(y=De_ct, color='k', linestyle='--')

# De_Eins = np.multiply(mue_interp,Te_trans)
# ax.plot(Te_trans, De_Eins, label = 'Einstein Relation')
ax.legend()
# plt.savefig('EinsteinRelation.png')


fig, ax = plt.subplots()
ax.set_title('Mobility Coef.')
ax.set_xlabel('Te [eV]')
ax.set_ylabel(r"$\mu_e \, $ [$ \, m^{2}/V/s$]", fontsize=afs)
ax.loglog(Te_trans, mue_interp, marker = 'o', label = 'Nominal')
for ic in energy: 
    ax.loglog(energy[ic], mu[ic]/nAr, marker = 'o', label=label[ic])
plt.axhline(y=mue_ct, color='k', linestyle='--')
ax.legend()


fig, ax = plt.subplots()
ax.set_title('Energy Mobility Coef.')
ax.set_xlabel('Te [eV]')
for ic in energy: 
    ax.plot(energy[ic], mu_e[ic]/nAr, marker = 'o', label=label[ic])
ax.legend()

fig, ax = plt.subplots()
ax.set_title('Energy Diffusion Coef.')
ax.set_xlabel('Te [eV]')
for ic in energy: 
    ax.plot(energy[ic], De_e[ic]/nAr, marker = 'o', label=label[ic])
ax.legend()



EEDF_list = {}; Te = {}; Te2 = {}; df_dict = {}

df = h5.File(file_path_EEDF + "./EEDF_test_2.h5", 'r')
ic = 1; df_dict[ic] = df; Te[ic] = []; Te2[ic] = df["temperature"][:]/eV; EEDF_list[ic] = []; clr[ic] = 'b'; label[ic] = 'test'
        

df = h5.File(file_path_EEDF + "./EEDF_BSR.h5", 'r')
ic = 2;  df_dict[ic] = df; Te[ic] = []; Te2[ic] = df["temperature"][:]/eV; EEDF_list[ic] = []; clr[ic] = 'r'; label[ic] = 'BSR'

df = h5.File(file_path_EEDF + "./EEDF_BSR_2.h5", 'r')
ic = 3;  df_dict[ic] = df; Te[ic] = []; Te2[ic] = df["temperature"][:]/eV; EEDF_list[ic] = []; clr[ic] = 'r'; label[ic] = 'BSR 2'
            


for ic in EEDF_list: 
    print("Reading case: ", ic)
    df = df_dict[ic]
    for i in range(len(df)-1):
        key = f'EEDF_{i}'
        dset = df[key]

        data = dset[()]  # This retrieves all the data from the dataset
        if (not np.array_equal(dset[:,0], data[:,0]) or not np.array_equal(dset[:,1], data[:,1])):
            print("Problem with converting dataset to data.")
            exit(-1)
        EEDF_list[ic].append(data)

        # name0 = dset.attrs['name0'] 
        # name1 = dset.attrs['name1'] 
        # unit0 = dset.attrs['unit0']
        # unit1 = dset.attrs['unit1']           
        index_eedf = dset.attrs['index'] 
        Te_eedf = dset.attrs['Electron Temperature'] 
        
        Te[ic].append(Te_eedf/eV)
        
        if ic ==1:
            # self.eRange = np.logspace(np.log10(1e-3),np.log10(300),1000)  # [eV]

            eRange = EEDF_list[ic][index_eedf][:,0]
            print(index_eedf, round(Te_eedf/eV), np.min(eRange), np.max(eRange))

        # print(key, i)


    Te[ic] = np.asarray(Te[ic])


# ic = 1
# Te_index = np.argmin(np.abs(Te[ic] - 5.0))
# Te_index2 = np.argmin(np.abs(Te2[ic] - 5.0))
# if Te_index != Te_index2: 
#     print("Problem with reading of EEDFs.")
#     exit(-1)


# eRange = EEDF_list[ic][Te_index][:,0] 
# EEDF = EEDF_list[ic][Te_index][:,1]*np.sqrt(eRange)
# EEDF_Maxwellian = 2*np.sqrt(eRange/np.pi)*(Te[ic][Te_index])**(-1.5)*np.exp(-eRange/(Te[ic][Te_index]))

Targeted_Temperature = 4.0
fig, ax = plt.subplots()
ax.set_title('EEDF')
ax.set_xlabel('Electron Energy [eV]')
ax.set_ylabel('f($\epsilon$)  [eV^-3/2]')
for ic in EEDF_list: 
    Te_index = np.argmin(np.abs(Te[ic] - Targeted_Temperature))
    eRange = EEDF_list[ic][Te_index][:,0] 
    EEDF = EEDF_list[ic][Te_index][:,1]*np.sqrt(eRange)
    ax.plot(eRange,EEDF/np.sqrt(eRange), label='Bolsig+ ' + label[ic])

EEDF_Maxwellian = 2*np.sqrt(eRange/np.pi)*(Te[ic][Te_index])**(-1.5)*np.exp(-eRange/(Te[ic][Te_index]))
ax.plot(eRange, EEDF_Maxwellian/np.sqrt(eRange),label='Maxwellian ')
ax.semilogy()
ax.legend()
plt.show()

# fig, ax = plt.subplots()
# ax.set_title('EEDF')
# ax.set_xlabel('Electron Energy [eV]')
# ax.set_ylabel('f($\epsilon$)  [eV^-1]')
# ax.plot(eRange,EEDF, label='Bolsig+')
# ax.plot(eRange, EEDF_Maxwellian,label='Maxwellian')
# ax.semilogx()
# ax.legend()







plt.show()


