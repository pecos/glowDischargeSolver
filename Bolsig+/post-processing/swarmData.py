import numpy as np

MaxPoints = 1000
kB, Td = 1.38064852e-23, 1.0e21

# datasets
swarmDatasets = ["AlAminLucas1987","MilloyCrompton1977","NakamuraKurachi1988",
                 "KucukarpaciLucas1981","PackPhelps1961","Robertson1977",
                 "RobertsonRee1972","WarrenParker1962","TownsendBailey1922",
                 "GoldenFisher1961","Kruithof1940","Specht1980","Tachibana1986"]

def readNumber(str):
    try:
        x = np.double(str)
    except ValueError:
        try:
            x = np.double(str[:-4] + 'E' + str[-4:])
        except ValueError:
            raise ValueError('Cannot read the number "%s".' % str)
    return x

# class singleVar:
#     name = ''
#     unit = ''
#     rms, max = 0., 0.
#
#     def __init__(self,name,unit,rms,max):
#         self.name = name
#         self.unit = unit
#         self.rms = rms          # not percent
#         self.max = max          # not percent
#         return

class singleData:
    variables = {}
    Nvar = 0
    parameters = {}

    def __init__(self):
        self.parameters = {}
        self.variables = {}

    # def printToScreen(self):
    #     print("{0:s}\t{1:s}".format(self.inputName,self.outputName))
    #     for k in range(self.data.shape[0]):
    #         print("{0:.6e}\t{1:.6e}".format(self.data[k,0], self.data[k,1]))
    #     print('\n')

class swarmData:
    datasets = []
    Ndatasets = 0
    variables = {}

    def __init__(self, filename, convert=True):
        """Initialize by reading BOLSIG output file."""
        self.ref = ''
        self.datasets = []
        self.Ndatasets = 0
        self.variables = {}
        self.parseData(filename)
        if (convert): self.ConvertData()

    def parseData(self, filename):
        """Read swarm data files."""

        with open(filename,'r') as fp:

            Ndata = 0

            line = fp.readline()
            while (line != ''):
                temp = line.strip()
                if ( temp == 'Reference' ):
                    self.ref = fp.readline().strip()
                elif ( temp == 'Variables (Var,Unit,Rms,Max)' ):
                    tmp = fp.readline().strip()
                    if ( tmp[0] == '*' ):
                        tmp = fp.readline().strip()
                        while ( tmp[0] != '*' ):
                            item = tmp.split()
                            self.variables.update({item[0]: item[1:]})
                            tmp = fp.readline().strip()
                elif ( temp == 'Data' ):
                    self.datasets.append(singleData())

                    tmp = fp.readline().strip()
                    if ( tmp[0] == '#' ):
                        # read parameter values. Not implemented at this point.
                        tmp = fp.readline().strip()
                        while ( tmp[0] != '#' ):
                            left, right = tmp.split()
                            self.datasets[Ndata].parameters[left] = right
                            tmp = fp.readline().strip()

                    variables = fp.readline().strip().split()
                    Nvar = len(variables)
                    tempData = np.zeros([MaxPoints,Nvar])
                    pt = 0
                    # self.datasets[Ndata].variables = fp.readline().strip().split()
                    # self.datasets[Ndata].Nvar = len(self.datasets[Ndata].variables)
                    tmp = fp.readline().strip()
                    if ( tmp[0] == '=' ):
                        tmp = fp.readline().strip()
                        while (tmp[0].isdigit() and not (tmp=='')):
                            d = tmp.split()
                            tempData[pt,:] = np.array([readNumber(d[k]) for k in range(Nvar)])
                            pt += 1
                            tmp = fp.readline().strip()
                    self.datasets[Ndata].Nvar = Nvar
                    for k in range(Nvar):
                        self.datasets[Ndata].variables.update({variables[k]: tempData[:pt,k]})
                        # d = tmp.split()
                        # self.datasets[Ndata].data = [[readNumber(d[k]) for k in range(self.datasets[Ndata].Nvar)]]
                        # tmp = fp.readline().strip()
                        # while (tmp[0].isdigit() and not (tmp=='')):
                        #     d = tmp.split()
                        #     self.datasets[Ndata].data = np.append(self.datasets[Ndata].data, [[readNumber(d[k]) for k in range(self.datasets[Ndata].Nvar)]], axis=0)
                        #     tmp = fp.readline().strip()

                    Ndata += 1

                line = fp.readline()

            self.Ndatasets = Ndata
            # for c in self.outputs:
            #     self.outputs[c].printToScreen()
        return

    def ConvertData(self):

        for dataset in self.datasets:
            for var in list(dataset.variables):
                if ( (var[-4:] == '-rms') or (var[-4:] == '-max') ): continue

                if ( (var == 'E/N') and (not (self.variables[var][0] == 'Td')) ):
                    if (self.variables[var][0] == 'Vcm2'):
                        dataset.variables[var] *= 1.0e-4 * Td
                elif ( var == 'E/p' ):
                    if (self.variables[var][0] == 'V/cm/mmHg'):
                        if 'T' in dataset.parameters:
                            temperature = readNumber(dataset.parameters['T'])
                        else:
                            temperature = 300.0
                        dataset.variables[var] *= 100. / 133.322 * kB * temperature * Td
                        dataset.variables['E/N'] = dataset.variables.pop(var)
                elif ( var == 'E/p300' ):
                    if (self.variables[var][0] == 'V/cm/mmHg'):
                        temperature = 300.0
                        dataset.variables[var] *= 100. / 133.322 * kB * temperature * Td
                        # dataset.variables[var] *= 3.22e16 * 1e6 / 133.322 / Td
                        dataset.variables['E/N'] = dataset.variables.pop(var)
                elif ( var == 'W' ):
                    if (self.variables[var][0] == 'cm/s'):
                        dataset.variables[var] *= 1.0e-2

                    if (self.variables[var][1] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 * readNumber(self.variables[var][1][:-1])
                        dataset.variables.update({var+'-rms': error})
                    elif (self.variables[var][2] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 / 3.0 * readNumber(self.variables[var][2][:-1])
                        dataset.variables.update({var+'-rms': error})
                    else:
                        if (var+'-rms') in dataset.variables:
                            if ( self.variables[var+'-rms'][0] == '%' ):
                                dataset.variables[var+'-rms'] *= 1e-2 * dataset.variables[var]
                        elif (var+'-max') in dataset.variables:
                            if ( self.variables[var+'-max'][0] == '%' ):
                                dataset.variables[var+'-max'] *= 1e-2 / 3.0 * dataset.variables[var]
                                dataset.variables[var+'-rms'] = dataset.variables.pop(var+'-max')
                elif ( var == 'DLN' ):
                    if (self.variables[var][0] == '1/cm/s'):
                        dataset.variables[var] *= 1.0e2

                    if (self.variables[var][1] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 * readNumber(self.variables[var][1][:-1])
                        dataset.variables.update({var+'-rms': error})
                    elif (self.variables[var][2] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 / 3.0 * readNumber(self.variables[var][2][:-1])
                        dataset.variables.update({var+'-rms': error})
                    else:
                        if (var+'-rms') in dataset.variables:
                            if ( self.variables[var+'-rms'][0] == '%' ):
                                dataset.variables[var+'-rms'] *= 1e-2 * dataset.variables[var]
                        elif (var+'-max') in dataset.variables:
                            if ( self.variables[var+'-max'][0] == '%' ):
                                dataset.variables[var+'-max'] *= 1e-2 / 3.0 * dataset.variables[var]
                                dataset.variables[var+'-rms'] = dataset.variables.pop(var+'-max')
                elif ( var == 'DL/mu' ):
                    if (self.variables[var][1] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 * readNumber(self.variables[var][1][:-1])
                        dataset.variables.update({var+'-rms': error})
                    elif (self.variables[var][2] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 / 3.0 * readNumber(self.variables[var][2][:-1])
                        dataset.variables.update({var+'-rms': error})
                    else:
                        if (var+'-rms') in dataset.variables:
                            if ( self.variables[var+'-rms'][0] == '%' ):
                                dataset.variables[var+'-rms'] *= 1e-2 * dataset.variables[var]
                        elif (var+'-max') in dataset.variables:
                            if ( self.variables[var+'-max'][0] == '%' ):
                                dataset.variables[var+'-max'] *= 1e-2 / 3.0 * dataset.variables[var]
                                dataset.variables[var+'-rms'] = dataset.variables.pop(var+'-max')
                elif ( var == 'DT/mu' ):
                    # if (self.variables[var][0] == '1/cm/s'):
                    #     dataset.variables[var] *= 1.0e2

                    if (self.variables[var][1] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 * readNumber(self.variables[var][1][:-1])
                        dataset.variables.update({var+'-rms': error})
                    elif (self.variables[var][2] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 / 3.0 * readNumber(self.variables[var][2][:-1])
                        dataset.variables.update({var+'-rms': error})
                    else:
                        if (var+'-rms') in dataset.variables:
                            if ( self.variables[var+'-rms'][0] == '%' ):
                                dataset.variables[var+'-rms'] *= 1e-2 * dataset.variables[var]
                        elif (var+'-max') in dataset.variables:
                            if ( self.variables[var+'-max'][0] == '%' ):
                                dataset.variables[var+'-max'] *= 1e-2 / 3.0 * dataset.variables[var]
                                dataset.variables[var+'-rms'] = dataset.variables.pop(var+'-max')
                elif ( (var == 'a/N') or (var == 'a1/N') or (var == 'a2/N') or (var == 'a3/N') or (var == 'a4/N') ):
                    if (self.variables[var][0] == 'cm2'):
                        dataset.variables[var] *= 1.0e-4

                    if (self.variables[var][1] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 * readNumber(self.variables[var][1][:-1])
                        dataset.variables.update({var+'-rms': error})
                    elif (self.variables[var][2] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 / 3.0 * readNumber(self.variables[var][2][:-1])
                        dataset.variables.update({var+'-rms': error})
                    else:
                        if (var+'-rms') in dataset.variables:
                            if ( self.variables[var+'-rms'][0] == '%' ):
                                dataset.variables[var+'-rms'] *= 1e-2 * dataset.variables[var]
                        elif (var+'-max') in dataset.variables:
                            if ( self.variables[var+'-max'][0] == '%' ):
                                dataset.variables[var+'-max'] *= 1e-2 / 3.0 * dataset.variables[var]
                                dataset.variables[var+'-rms'] = dataset.variables.pop(var+'-max')
                elif ( var == 'a/E' ):
                    if (self.variables[var][1] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 * readNumber(self.variables[var][1][:-1])
                        dataset.variables.update({var+'-rms': error})
                    elif (self.variables[var][2] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 / 3.0 * readNumber(self.variables[var][2][:-1])
                        dataset.variables.update({var+'-rms': error})
                    else:
                        if (var+'-rms') in dataset.variables:
                            if ( self.variables[var+'-rms'][0] == '%' ):
                                dataset.variables[var+'-rms'] *= 1e-2 * dataset.variables[var]
                        elif (var+'-max') in dataset.variables:
                            if ( self.variables[var+'-max'][0] == '%' ):
                                dataset.variables[var+'-max'] *= 1e-2 / 3.0 * dataset.variables[var]
                                dataset.variables[var+'-rms'] = dataset.variables.pop(var+'-max')
                elif ( var == 'a/p' ):
                    if (self.variables[var][0] == '1/cm/mmHg'):
                        temperature = 300.0
                        dataset.variables[var] *= 100. / 133.322 * kB * temperature

                    if (self.variables[var][1] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 * readNumber(self.variables[var][1][:-1])
                        dataset.variables.update({var+'-rms': error})
                    elif (self.variables[var][2] != 'n/a'):
                        error = dataset.variables[var] * 1e-2 / 3.0 * readNumber(self.variables[var][2][:-1])
                        dataset.variables.update({var+'-rms': error})
                    else:
                        if (var+'-rms') in dataset.variables:
                            if ( self.variables[var+'-rms'][0] == '%' ):
                                dataset.variables[var+'-rms'] *= 1e-2 * dataset.variables[var]
                        elif (var+'-max') in dataset.variables:
                            if ( self.variables[var+'-max'][0] == '%' ):
                                dataset.variables[var+'-max'] *= 1e-2 / 3.0 * dataset.variables[var]
                                dataset.variables[var+'-rms'] = dataset.variables.pop(var+'-max')

                    dataset.variables['a/N'] = dataset.variables.pop(var)
                    dataset.variables['a/N-rms'] = dataset.variables.pop(var+'-rms')

        return

# if __name__ == '__main__':
