import numpy as np
import h5py as h5
import pandas as pd

rxnNames = ['Excitation_Vibrational', 'Excitation_Electronic', 'Excitation_N', 'Ionization_N2', 'Ionization_N', 'Dissociation']

transportNames = ['E_Mobility', 'E_Diffusion']

for i in range(len(rxnNames)):
    data = pd.read_csv('{}.txt'.format(rxnNames[i]), delimiter='\t', header=None)
    table = np.zeros((len(data[1]),2))
    table[:,0] = data[1].to_numpy()
    table[:,1] = data[3].to_numpy()
    table[:,0] *= 11604
    table[:,1] *= 6.022e23
    with h5.File('{}.h5'.format(rxnNames[i]), 'w') as f:
        dset = f.create_dataset("table", table.shape, data=table)
        dset.attrs['name0'] = 'Electron Temperature'
        dset.attrs['unit0'] = 'K'
        dset.attrs['name1'] = 'Rate Coefficient'
        dset.attrs['unit1'] = 'm3/mol-s'

    print('Rxn: {} done!'.format(rxnNames[i]))

mobility = pd.read_csv('E_Mobility.txt', delimiter = '\t', header=None)
diffusion = pd.read_csv('E_Diffusion.txt', delimiter = '\t', header=None)

mobility = mobility[0].str.split(',', expand=True)
mobility[0] = mobility[0].str.split('(', expand=True)[1]
mobility[1] = mobility[1].str.split(')', expand=True)[0]
diffusion = diffusion[0].str.split(',', expand=True)
diffusion[0] = diffusion[0].str.split('(', expand=True)[1]
diffusion[1] = diffusion[1].str.split(')', expand=True)[0]

table_mob = np.zeros((len(mobility[0]),2))
table_dif = np.zeros((len(diffusion[0]),2))

table_mob[:,0] = mobility[0].to_numpy()
table_mob[:,1] = mobility[1].to_numpy()
table_dif[:,0] = diffusion[0].to_numpy()
table_dif[:,1] = diffusion[1].to_numpy()

table_mob[:,0] *= 11604
table_dif[:,0] *= 11604

with h5.File('nominal_transport.h5', 'w') as g:
    dset = g.create_dataset("mobility", table_mob.shape, data=table_mob)
    dset.attrs['name0'] = 'Electron Temperature'
    dset.attrs['unit0'] = 'K'
    dset.attrs['name1'] = 'Mobility * N'
    dset.attrs['unit1'] = '1/V-m-s'

    dset = g.create_dataset("diffusivity", table_dif.shape, data=table_dif)
    dset.attrs['name0'] = 'Electron Temperature'
    dset.attrs['unit0'] = 'K'
    dset.attrs['name1'] = 'Diffusion Coefficient * N'
    dset.attrs['unit1'] = '1/m-s'

