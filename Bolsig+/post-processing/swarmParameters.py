import numpy as np

MaxPoints = 2000

typeDictS2I = {"Electric field / N (Td)":               0,
               "Grid type":                             1,
               "Maximum energy":                        2,
               "Mean energy (eV)":                      3,
               "Mobility *N (1/m/V/s)":                 4,
               "Diffusion coefficient *N (1/m/s)":      5,
               "Energy mobility *N (1/m/V/s)":          6,
               "Energy diffusion coef. D*N (1/m/s)":    7,
               "Total collision freq. /N (m3/s)":       8,
               "Momentum frequency /N (m3/s)":          9,
               "Total ionization freq. /N (m3/s)":      10,
               "Townsend ioniz. coef. alpha/N (m2)":    11,
               "Power /N (eV m3/s)":                    12,
               "Elastic power loss /N (eV m3/s)":       13,
               "Inelastic power loss /N (eV m3/s)":     14,
               "Growth power /N (eV m3/s)":             15,
               "# of iterations":                       16,
               "# of grid trials":                      17,
               "Longitud. diffusion coef. *N (1/m/s)":  18,
               "Bulk mobility *N (1/m/V/s)":            19,
               "Bulk T diffusion coef. *N (1/m/s)":     20,
               "Bulk L diffusion coef. *N (1/m/s)":     21 }
typeDictI2S = {}
for key, value in typeDictS2I.items():
    typeDictI2S.update({value: key})
# typeDictI2S = {0: "Electric field / N (Td)",
#                1: "Grid type",
#                2: "Maximum energy",
#                3: "Mean energy (eV)",
#                4: "Mobility *N (1/m/V/s)",
#                5: "Diffusion coefficient *N (1/m/s)",
#                6: "Energy mobility *N (1/m/V/s)",
#                7: "Energy diffusion coef. D*N (1/m/s)",
#                8: "Total collision freq. /N (m3/s)",
#                8: "Momentum frequency /N (m3/s)",
#                10: "Total ionization freq. /N (m3/s)",
#                11: "Townsend ioniz. coef. alpha/N (m2)",
#                12: "Power /N (eV m3/s)",
#                13: "Elastic power loss /N (eV m3/s)",
#                14: "Inelastic power loss /N (eV m3/s)",
#                15: "Growth power /N (eV m3/s)",
#                16: "# of iterations",
#                17: "# of grid trials",
#                18: "Longitud. diffusion coef. *N (1/m/s)",
#                19: "Bulk mobility *N (1/m/V/s)",
#                20: "Bulk T diffusion coef. *N (1/m/s)",
#                21: "Bulk L diffusion coef. *N (1/m/s)" }

def readNumber(str):
    try:
        x = np.double(str)
    except ValueError:
        try:
            x = np.double(str[:-4] + 'E' + str[-4:])
        except ValueError:
            raise ValueError('Cannot read the number "%s".' % str)
    return x

class singleOutput:
    data = []
    inputName = ''
    outputName = ''
    species = ''
    collisionType = ''
    deltaE = 0.

    def __init__(self):
        self.data = []
        self.inputName = ''
        self.outputName = ''
        self.species = ''
        self.collisionType = ''
        self.deltaE = 0.

    def printToScreen(self):
        print("{0:s}\t{1:s}\t{2:.6e}".format(self.species,self.collisionType,self.deltaE))
        print("{0:s}\t{1:s}".format(self.inputName,self.outputName))
        for k in range(self.data.shape[0]):
            print("{0:.6e}\t{1:.6e}".format(self.data[k,0], self.data[k,1]))
        print('\n')

class bolsigOutput:
    options = {}
    outputs = {}
    filename = ""

    def __init__(self, filename, verbose=False):
        """Initialize by reading BOLSIG output file."""
        self.options = {}
        self.outputs = {}
        self.typeDictS2I = typeDictS2I
        self.typeDictI2S = typeDictI2S
        self.filename = filename
        self.parseOutputFile(filename, verbose)

    def parseOutputFile(self, filename, verbose=False):
        """Read output file of BOLSIG."""

        with open(filename,'r') as fp:

            line = fp.readline().strip()
            # Skip the BOLSIG version and collision input data.
            while (not (line=='') and line[0] != ' '):
                if (verbose): print(line)
                line = fp.readline().strip()

            # Read until we find condition table.
            while (line == ''):
                line = fp.readline().strip()


            # Read conditions and store them in dictionaries. Omit this functionality for now.
            while (not (line=='') and line[0] != ' '):
                if (verbose): print(line)
                line = fp.readline().strip()

            # Read tables one by one.
            nOutputs = 0
            line = fp.readline()
            while (line != ''):

                if (line.strip() != ''):
                
                    NtableLabel = len(line.strip().split('\t'))
                    Nformat = len(line.strip().split('   '))
                    # print(line.strip())
                    # print(NtableLabel,Nformat)
                    
                    
                    if ((NtableLabel==1) and (Nformat==1)):
                        inputSpecies = line.strip()
                        inputCollision = ''
                        inputE = 0.
                        line = fp.readline()
                    if ((NtableLabel==1) and (Nformat>1)):
                        # If table is not split by tab, it specified rate coefficients.
                        formatString = line.strip().split('   ')
                        if (Nformat==3):
                            if len(formatString[0].strip().split('  ')) == 2:
                                tmpDataTypeString = formatString[0].strip().split('  ')[0]
                                inputSpecies = formatString[0].strip().split('  ')[1]
                                inputCollision = formatString[1].strip()
                                inputE = readNumber(formatString[-1][:-3])
                            else:
                                tmpDataTypeString = formatString[0].strip()
                                inputSpecies = formatString[1].strip()
                                inputCollision = formatString[2].strip()
                                inputE = 0.
                        elif (Nformat==4): # last argument is simply 'eV'.
                            tmpDataTypeString = formatString[0].strip()
                            inputSpecies = formatString[1].strip()
                            inputCollision = formatString[2].strip()
                            inputE = readNumber(formatString[-1][:-3])
                        # Read table. Split by tab.
                        line = fp.readline().strip()
                        left, right = line.split('\t')
                    else:
                        # Read table. Split by tab.
                        left, right = line.split('\t')
                        tmpDataTypeString = right.strip()

                    if tmpDataTypeString not in self.typeDictS2I:
                        nDict = len(self.typeDictS2I)
                        self.typeDictS2I[tmpDataTypeString] = nDict
                        self.typeDictI2S[nDict] = tmpDataTypeString
                    dataType = self.typeDictS2I[tmpDataTypeString]

                    #if dataType in self.outputs:
                    #    print("Warning: output '%s' already exists and will be overwritten." % typeDictI2S[dataType])
                    self.outputs.update({dataType: singleOutput()})
                    self.outputs[dataType].inputName = left.strip()
                    self.outputs[dataType].outputName = right.strip()

                    
                    if (NtableLabel==1):
                        self.outputs[dataType].species = inputSpecies
                        self.outputs[dataType].collisionType = inputCollision
                        self.outputs[dataType].deltaE = inputE

                    tempData = np.zeros([MaxPoints,2])
                    pt = 0
                    # Once we find numbers, read until we don't find a number
                    tmp = fp.readline().strip()
                    while (not (tmp=='') and tmp[0].isdigit()):
                        d = tmp.split()
                        tempData[pt,:] = [readNumber(d[0]), readNumber(d[1])]
                        pt += 1
                        tmp = fp.readline().strip()
                    self.outputs[dataType].data = tempData[:pt,:]
                    self.outputs[dataType].printToScreen
                    nOutputs += 1


                line = fp.readline()


            if (verbose):
                for c in self.outputs:
                    self.outputs[c].printToScreen()
        return



class bolsigEEDFOutput:
    options = {}
    outputs = {}
    filename = ""

    def __init__(self, filename, verbose=False):
        """Initialize by reading BOLSIG output file."""
        self.options = {}
        self.outputs = {}
        self.typeDictS2I = typeDictS2I
        self.typeDictI2S = typeDictI2S
        self.filename = filename
        self.parseEEDFOutputFile(filename, verbose)

    def parseEEDFOutputFile(self, filename, verbose=False):
        with open(filename, 'r') as fp:
            line = fp.readline().strip()
            
            while(not line == '' and line[0] != 'R'):
                if(verbose):
                    print(line)
                line = fp.readline().strip()
                
            ## Skip lines all the way to the first dashed line
            line = fp.readline().strip()
            line = fp.readline().strip()
            line = fp.readline().strip()
            line = fp.readline().strip()
            
            ## Read input conditions (not stored anywhere currently)
            nOutputs = 0

            while(not line == ''): 
                line = fp.readline().strip()


                while(not line == '' and line[0] != '-'):
                    if(verbose):
                        print(line)
                    if line.strip().split('    ')[0].strip() == 'Electric field / N (Td)':
                        inputTd = readNumber(line.strip().split('    ')[-1].strip())

                        tmpDataTypeString = 'Electric field / N (Td)'
                        if tmpDataTypeString not in self.typeDictS2I:
                            nDict = len(self.typeDictS2I)
                            self.typeDictS2I[tmpDataTypeString] = nDict
                            self.typeDictI2S[nDict] = tmpDataTypeString  
                                                  
                        dataType = self.typeDictS2I[tmpDataTypeString]                       
                        if dataType not in self.outputs:
                            self.outputs.update({dataType: singleOutput()})
                            self.outputs[dataType].inputName = 'E/N (Td)'
                            self.outputs[dataType].outputName = tmpDataTypeString.strip()
                        self.outputs[dataType].data.append([inputTd, inputTd])  

                    line = fp.readline().strip()
                #print('Input Conditions Parsed!' + '\n')


                ## Read property outputs
                line = fp.readline().strip()
                while(not line == '' and line[0] != '-'):
                    if(verbose):
                        print(line)

                    formatString = []
                    tempString = line.strip().split('  ')
                    for elem in tempString:
                        if elem != '  ' or elem != ' ' or elem != '':
                            formatString.append(elem.strip())
                    tmpDataTypeString = formatString[0]

                    if tmpDataTypeString not in self.typeDictS2I:
                        nDict = len(self.typeDictS2I)
                        self.typeDictS2I[tmpDataTypeString] = nDict
                        self.typeDictI2S[nDict] = tmpDataTypeString
                    
                    
                    dataType = self.typeDictS2I[tmpDataTypeString]
                    if dataType not in self.outputs:
                        self.outputs.update({dataType: singleOutput()})
                        self.outputs[dataType].inputName = 'E/N (Td)'
                        self.outputs[dataType].outputName = tmpDataTypeString.strip()
                    
                    # tempData = np.zeros([MaxPoints,2])
                    # pt = 0
                    # tempData[pt,:] = [inputTd, readNumber(formatString[-1])]
                                    
                    # pt += 1
                    # self.outputs[dataType].data = tempData[:pt,:]
                    self.outputs[dataType].data.append([inputTd, readNumber(formatString[-1])])  

                    self.outputs[dataType].printToScreen

                    line = fp.readline().strip()


                #print('Output Properties Parsed!' + '\n')

            
                ## Read EEDF
                line = fp.readline().strip()

                if 'EEDF (eV-3/2)' not in self.typeDictS2I:
                    nDict = len(self.typeDictS2I)
                    self.typeDictS2I['EEDF (eV-3/2)'] = nDict
                    self.typeDictI2S[nDict] = 'EEDF (eV-3/2)'

                dataType = self.typeDictS2I['EEDF (eV-3/2)']
                if dataType not in self.outputs:
                    self.outputs.update({dataType: singleOutput()})
                    self.outputs[dataType].inputName = 'Energy (eV)'
                    self.outputs[dataType].outputName = 'EEDF (eV-3/2)'

                if 'Anisotropy' not in self.typeDictS2I:
                    nDict = len(self.typeDictS2I)
                    print(nDict, " " ,line)
                    self.typeDictS2I['Anisotropy'] = nDict
                    self.typeDictI2S[nDict] = 'Anisotropy'

                dataType2 = self.typeDictS2I['Anisotropy']

                if dataType2 not in self.outputs:
                    self.outputs.update({dataType2: singleOutput()})
                    self.outputs[dataType2].inputName = 'Energy (eV)'
                    self.outputs[dataType2].outputName = 'Anisotropy'


                tempData = np.zeros([MaxPoints,2])
                tempData2 = np.zeros([MaxPoints,2])
                pt = 0
                tmp = fp.readline().strip()
                while (not (tmp=='') and tmp[0].strip().isdigit()):
                    d = tmp.strip().split('\t')
                    tempData[pt,:] = [readNumber(d[0].strip()), readNumber(d[1].strip())]
                    tempData2[pt,:] = [readNumber(d[0].strip()), readNumber(d[2].strip())]
                    pt += 1
                    tmp = fp.readline().strip()

                self.outputs[dataType].data.append(tempData[:pt,:])
                self.outputs[dataType2].data.append(tempData2[:pt,:])

                    

                self.outputs[dataType].printToScreen
                self.outputs[dataType2].printToScreen

                line = fp.readline()
                line = fp.readline()
                line = fp.readline()


                nOutputs += 1
                
                # print(line[0:2])
                # print('skata')
                # exit(-1)
                

            for c in self.outputs:
                print(c, self.outputs[c].outputName, "  " ,len(self.outputs[c].data))

                if self.outputs[c].outputName != "EEDF (eV-3/2)" and self.outputs[c].outputName != "Anisotropy": 
                    self.outputs[c].data = np.array(self.outputs[c].data)
            
                
                
#            print(self.outputs[dataType].data[:,1]) 
            if(verbose):
                for c in self.outputs:
                    self.outputs[c].printToScreen()
        return




# class bolsigEEDFOutput:
#     options = {}
#     outputs = {}
#     filename = ""

#     def __init__(self, filename, verbose=False):
#         """Initialize by reading BOLSIG output file."""
#         self.options = {}
#         self.outputs = {}
#         self.typeDictS2I = typeDictS2I
#         self.typeDictI2S = typeDictI2S
#         self.filename = filename
#         self.parseEEDFOutputFile(filename, verbose)

#     def parseEEDFOutputFile(self, filename, verbose=False):
#         with open(filename, 'r') as fp:
#             line = fp.readline().strip()
            
#             while(not line == '' and line[0] != 'R'):
#                 if(verbose):
#                     print(line)
#                 line = fp.readline().strip()
                
            
#             ## Skip lines all the way to the first dashed line
#             line = fp.readline().strip()
#             line = fp.readline().strip()
#             line = fp.readline().strip()
#             line = fp.readline().strip()
            
#             ## Read input conditions (not stored anywhere currently)
#             line = fp.readline().strip()
#             while(not line == '' and line[0] != '-'):
#                 if(verbose):
#                     print(line)
#                 if line.strip().split('    ')[0].strip() == 'Electric field / N (Td)':
#                     inputTd = readNumber(line.strip().split('    ')[-1].strip())

#                 line = fp.readline().strip()
#             #print('Input Conditions Parsed!' + '\n')
            
#             ## Read property outputs
#             nOutputs = 0
#             line = fp.readline().strip()
#             while(not line == '' and line[0] != '-'):
#                 if(verbose):
#                     print(line)

#                 formatString = []
#                 tempString = line.strip().split('  ')
#                 for elem in tempString:
#                     if elem != '  ' or elem != ' ' or elem != '':
#                         formatString.append(elem.strip())
#                 tmpDataTypeString = formatString[0]

#                 if tmpDataTypeString not in self.typeDictS2I:
#                     nDict = len(self.typeDictS2I)
#                     self.typeDictS2I[tmpDataTypeString] = nDict
#                     self.typeDictI2S[nDict] = tmpDataTypeString
                
#                 dataType = self.typeDictS2I[tmpDataTypeString]
#                 self.outputs.update({dataType: singleOutput()})
#                 self.outputs[dataType].inputName = 'E/N (Td)'
#                 self.outputs[dataType].outputName = tmpDataTypeString.strip()
                
#                 tempData = np.zeros([MaxPoints,2])
#                 pt = 0
#                 tempData[pt,:] = [inputTd, readNumber(formatString[-1])]
#                 pt += 1
#                 self.outputs[dataType].data = tempData[:pt,:]
#                 self.outputs[dataType].printToScreen
#                 nOutputs += 1

#                 line = fp.readline().strip()

#             #print('Output Properties Parsed!' + '\n')

#             ## Read Rate Coefficients
#             line = fp.readline().strip()
#             line = fp.readline().strip()

#             while(not line == '' and line[0] != '-'):
#                 if(verbose):
#                     print(line)
#                 Nformat = len(line.strip().split('   ')) 
#                 formatString = line.strip().split('   ')
                
#                 if Nformat==10:
#                     ## Reading Elastic collision rate coefficient
#                     if formatString[2].strip() == 'Elastic':
#                         tmpDataTypeString = formatString[0].strip()
#                         inputSpecies = formatString[1].strip()
#                         inputCollision = formatString[2].strip()
#                         inputE = 0.0
#                 ## Read rate coefficients from C1 to C99
#                 elif Nformat == 5 or Nformat == 6:
#                     tmpDataTypeString = formatString[0].strip()
#                     inputSpecies = formatString[1].strip()
#                     inputCollision = formatString[2].strip()
#                     inputE = readNumber(formatString[3].strip().split(' ')[0])
#                 ## Read rate coefficients from C100
#                 elif Nformat == 4:
#                     tmpDataTypeString = formatString[0].strip().split('  ')[0]
#                     inputSpecies = formatString[0].strip().split('  ')[1]
#                     inputCollision = formatString[1].strip()
#                     inputE = readNumber(formatString[2].strip().split(' ')[0])
                
#                 if tmpDataTypeString not in self.typeDictS2I:
#                     nDict = len(self.typeDictS2I)
#                     self.typeDictS2I[tmpDataTypeString] = nDict
#                     self.typeDictI2S[nDict] = tmpDataTypeString

#                 dataType = self.typeDictS2I[tmpDataTypeString]
#                 self.outputs.update({dataType: singleOutput()})
#                 self.outputs[dataType].inputName = 'E/N (Td)'
#                 self.outputs[dataType].outputName = tmpDataTypeString.strip()
#                 self.outputs[dataType].species = inputSpecies
#                 self.outputs[dataType].collisionType = inputCollision
#                 self.outputs[dataType].deltaE = inputE
                
#                 tempData = np.zeros([MaxPoints,2])
#                 pt = 0
#                 tempData[pt,:] = [inputTd, readNumber(formatString[-1].strip())]
#                 pt += 1
#                 self.outputs[dataType].data = tempData[:pt,:]
#                 self.outputs[dataType].printToScreen
#                 nOutputs += 1

#                 line = fp.readline().strip()

#             #print('Rate Coefficients Parsed!')
           
#             ## Read EEDF
#             line = fp.readline().strip()
#             if 'EEDF (eV-3/2)' not in self.typeDictS2I:
#                 nDict = len(self.typeDictS2I)
#                 self.typeDictS2I['EEDF (eV-3/2)'] = nDict
#                 self.typeDictI2S[nDict] = 'EEDF (eV-3/2)'

#             dataType = self.typeDictS2I['EEDF (eV-3/2)']

#             self.outputs.update({dataType: singleOutput()})
#             self.outputs[dataType].inputName = 'Energy (eV)'
#             self.outputs[dataType].outputName = 'EEDF (eV-3/2)'

#             if 'Anistropy' not in self.typeDictS2I:
#                 nDict = len(self.typeDictS2I)
#                 self.typeDictS2I['Anisotropy'] = nDict
#                 self.typeDictI2S[nDict] = 'Anisotropy'

#             dataType2 = self.typeDictS2I['Anisotropy']

#             self.outputs.update({dataType2: singleOutput()})
#             self.outputs[dataType2].inputName = 'Energy (eV)'
#             self.outputs[dataType2].outputName = 'Anisotropy'

#             tempData = np.zeros([MaxPoints,2])
#             tempData2 = np.zeros([MaxPoints,2])
#             pt = 0

#             while(not line == ''): 
#                 #if 'EEDF (eV-3/2)' not in self.typeDictS2I:
#                 #    nDict = len(self.typeDictS2I)
#                 #    self.typeDictS2I['EEDF (eV-3/2)'] = nDict
#                 #    self.typeDictI2S[nDict] = 'EEDF (eV-3/2)'

#                 #dataType = self.typeDictS2I['EEDF (eV-3/2)']

#                 #self.outputs.update({dataType: singleOutput()})
#                 #self.outputs[dataType].inputName = 'Energy (eV)'
#                 #self.outputs[dataType].outputName = 'EEDF (eV-3/2)'

#                 #if 'Anistropy' not in self.typeDictS2I:
#                 #    nDict = len(self.typeDictS2I)
#                 #    self.typeDictS2I['Anisotropy'] = nDict
#                 #    self.typeDictI2S[nDict] = 'Anisotropy'

#                 #dataType2 = self.typeDictS2I['Anisotropy']

#                 #self.outputs.update({dataType2: singleOutput()})
#                 #self.outputs[dataType2].inputName = 'Energy (eV)'
#                 #self.outputs[dataType2].outputName = 'Anisotropy'

#                 #tempData = np.zeros([MaxPoints,2])
#                 #tempData2 = np.zeros([MaxPoints,2])
#                 #pt = 0
#                 # Once we find numbers, read until we don't find a number
#                 tmp = fp.readline().strip()
#                 while (not (tmp=='') and tmp[0].strip().isdigit()):
#                     d = tmp.strip().split('\t')
#                     tempData[pt,:] = [readNumber(d[0].strip()), readNumber(d[1].strip())]
#                     tempData2[pt,:] = [readNumber(d[0].strip()), readNumber(d[2].strip())]
#                     pt += 1
#                     tmp = fp.readline().strip()
#                     self.outputs[dataType].data = tempData[:pt,:]
#                     self.outputs[dataType2].data = tempData2[:pt,:]

#                 self.outputs[dataType].printToScreen
#                 self.outputs[dataType2].printToScreen
#                 nOutputs += 1

#                 line = fp.readline()
            

# #            print(self.outputs[dataType].data[:,1]) 
#             if(verbose):
#                 for c in self.outputs:
#                     self.outputs[c].printToScreen()
#         return



if __name__ == '__main__':
    dataset = 'Phelps'
    tmp = bolsigOutput('output/datasets/%s-transport.dat' % dataset)
    # np.savetxt('%s.muN.raw.txt' % dataset,tmp.outputs[4].data, fmt='%1.15e')
    # np.savetxt('%s.DN.raw.txt' % dataset,tmp.outputs[5].data, fmt='%1.15e')
    # np.savetxt('%s.DLN.raw.txt' % dataset,tmp.outputs[18].data, fmt='%1.15e')
