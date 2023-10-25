import numpy as np
import h5py as h5
import argparse

voltage_uncert = 1./3. #Volts, given as standard deviation, assuming a normal distribution
#parser = argparse.ArgumentParser(description="")
#parser.add_argument(

for i in range(7200):
    factor = np.random.normal(0, voltage_uncert)
    with h5.File('./BOLSIGChemistry_Voltage/Voltage.%08d.h5' % (i), 'w') as f:
        f.create_dataset('V_Err', data = [factor])

    if (i % 100) == 0:
        print('{}-th sample generated'.format(i))

print('Done!')
