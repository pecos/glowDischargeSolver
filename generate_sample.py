import numpy as np
import h5py
import yaml
import argparse
import re

# reaction units based on the number of reactants.
unitDict = {1: '1/s', 2: 'm3/mol/s', 3: 'm6/mol2/s'}
parser = argparse.ArgumentParser(description = "", formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('input_file', metavar = 'string', type = str, help = 'filename for an input file.\n')


def getNumberOfReactants(equation):
    reactants, direction, products = re.split("( => | <=> )", equation)
    reactants = re.split(" \+ ", reactants)
    return len(reactants)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.input_file) as c:
        dict_ = yaml.safe_load(c)

    # parse arrhenius-type reactions from the input file.
    rxnList = dict_['reactions']
    for rxn in rxnList:
        if (rxn['rate_type'] == 'arrhenius'):
            print("Equation: %s" % rxn['equation'])
            print("Nominal coefficients: %s" % rxn['arrhenius']['coefficients'])
            print("%d-body reaction" % getNumberOfReactants(rxn['equation']))
    
    # create the directory for samples.
    import os
    outputDir = dict_['directory']
    if (outputDir != '.'):
        os.makedirs(outputDir, exist_ok = True)

    prefix = dict_['prefix']      
    error = 0.01* dict_['relative_error']
    nSample = dict_['number_of_samples']
    for k in range(nSample):                                                                                                            
        filename = '%s/%s.%08d.h5' % (outputDir, prefix, k) 
        with h5py.File(filename, 'w') as f:
            for rxn in rxnList:
                if (rxn['rate_type'] =='arrhenius'):
                    rateUnit = unitDict[getNumberOfReactants(rxn['equation'])] 
                    # Take the nominal value from the input file
                    coeffs0 = np.array(rxn['arrhenius']['coefficients'], dtype = np.double)
                    # Add a relative error
                    coeffs0[0] *= (1.0 + error) ** np.random.normal()
                    # Create a dataset for the reaction
                    dset = f.create_dataset(rxn['equation'], (3,), data = coeffs0)
                    dset.attrs['rate_unit'] = rateUnit
                    dset.attrs['temperature_unit'] = 'K'
        if (k % 10 == 0):
            print("%d-th sample generated." % k)

