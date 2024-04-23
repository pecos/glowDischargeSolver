import numpy as np
import matplotlib.pyplot as plt

# Map rxn type names to integer
typeDictS2I = {"ELASTIC":    0,
               "EFFECTIVE":  1,
               "EXCITATION": 2,
               "IONIZATION": 3,
               "ATTACHMENT": 4 }

# Map rxn type integers to names
typeDictI2S = {0: "ELASTIC",
               1: "EFFECTIVE",
               2: "EXCITATION",
               3: "IONIZATION",
               4: "ATTACHMENT" }

# Class that holds data for a single cross section
class singleCrossSection:
    """A class representing a single cross section

    Attributes
    ----------
    colType : int
        Collision type (0=ELASTIC, 1=EFFECTIVE, 2=EXCITATION,
                        3=IONIZATION, 4=ATTACHMENT)
    colName : string
        Collision "name"; 2nd line of LXCat entry
    data : (n,2) numpy array
        Collision cross section data; 1st column is energy in eV, 2nd
        column is area in m^2
    deltaE : float
        For excitation or ionization, the electron energy loss in eV.
        For elastic and effective, the ratio of the electron mass to
        the target particle mass.  For attachment, not used (set to
        nan).

    Methods
    -------
    printToScreen():
        Prints collision information to the screen.
    plot():
        Plots collision cross section as fcn of energy.
    """
    colType = 0
    colName = ''
    data = np.zeros((1,2))
    deltaE = np.nan
    info = ''

    def printToScreen(self):
        """Prints collision information to the screen."""
        print("Cross Section Data:")
        print("  Type = {0:s}".format(typeDictI2S[self.colType]))
        print("  Name = {0:s}".format(self.colName))
        print("  DeltaE = {0:.6e}".format(self.deltaE))
        print("  Data = {0:.6e} {1:.6e}".format(self.data[0,0], self.data[0,1]))
        for i in range(1,self.data.shape[0]):
            print("         {0:.6e} {1:.6e}".format(self.data[i,0], self.data[i,1]))

    def plot(self):
        """Plots collision cross section as a fcn of energy."""
        plt.plot(self.data[:,0], self.data[:,1], 'b-x', label=self.colName)
        plt.xlabel('Energy [eV]')
        plt.ylabel('Cross Section [m2]')
        plt.title(self.colName)
        plt.show()


class multipleCrossSections:
    """A class holding multiple collision cross sections.

    Attributes
    ----------
    crs : List of singleCrossSection objects

    Methods
    -------
    parseLXCatFile(filename):
        Reads the collisions contained in filename, which is assumed
        to be in LXCat format.
    """

    crs = []

    def __init__(self, filename):
        """Initialize cross sections by reading LXCat file."""
        self.crs = []
        self.parseLXCatFile(filename)

    def parseLXCatFile(self, filename, printOut=False):
        """Read LXCat file.

        File format is assumed to obey the LXCat conventions, as
        documented below (which is an abridged version of "CROSS
        SECTION DATA FORMAT" description in LXCat file header):

        In downloaded files, each collision process is defined by a
        block consisting of

        1st line
        Keyword in capitals indicating the type of the
        collision. Possible collision types are elastic, effective,
        excitation, ionization, or attachment (capital letters
        required, key words are case sensitive).

        2nd line
        Name of the target particle species. This name is a character
        string, freely chosen by the user, e.g. "Ar".

        3rd line
        For elastic and effective collisions, the ratio of the
        electron mass to the target particle mass. For excitation or
        ionization collisions, the electron energy loss (nominally the
        threshold energy) in eV. For attachment, the 3rd line is
        missing.

        from 4th line (optionally)
        User comments and reference information, maximum 100
        lines. The only constraint on format is that these comment
        lines must not start with a number.

        Finally Table of the cross section as a function of
        energy. The table starts and ends by a line of dashes "------"
        (at least 5), and has otherwise two numbers per line: the
        energy in eV and the cross section in m2.
        """

        ncrs = 0

        with open(filename,'r') as fp:

            line = fp.readline()
            while (line != ''):
                if (line.strip() == "ELASTIC"    or line.strip() == "EFFECTIVE"  or
                    line.strip() == "EXCITATION" or line.strip() == "IONIZATION" or
                    line.strip() == "ATTACHMENT"                                    ):

                    self.crs.append(singleCrossSection())
                    self.crs[ncrs].colType = typeDictS2I[line.strip()]
                    self.crs[ncrs].colName = fp.readline().strip()

                    if (typeDictI2S[self.crs[ncrs].colType] != 'ATTACHMENT'):
                        self.crs[ncrs].deltaE  = np.double(fp.readline().strip().split()[0]) #NOTE(malamast): It was np.float and I changed it to np.double


                    # Read until we find a number
                    tmp = fp.readline().strip()
                    while (not tmp[0].isdigit() and not (tmp=='')):
                        if (not tmp[0]=='-'):
                            self.crs[ncrs].info += tmp + '\n'
                        tmp = fp.readline().strip()

                    # Once we find numbers, read until we don't find a number
                    d = tmp.split()
                    self.crs[ncrs].data[0,0] = np.double(d[0]) #NOTE(malamast): It was np.float and I changed it to np.double
                    self.crs[ncrs].data[0,1] = np.double(d[1]) #NOTE(malamast): It was np.float and I changed it to np.double
                    tmp = fp.readline().strip()
                    while (tmp[0].isdigit() and not (tmp=='')):
                        d = tmp.split()
                        self.crs[ncrs].data = \
                            np.append(self.crs[ncrs].data,
                                      [[np.double(d[0]), np.double(d[1])]], axis=0) #NOTE(malamast): It was np.float and I changed it to np.double
                        tmp = fp.readline().strip()

                    if printOut:
                        print("Found {0:s} collison: {1:s}".format(typeDictI2S[self.crs[ncrs].colType],
                                                                   self.crs[ncrs].colName))
                    ncrs +=1


                line = fp.readline()

            if printOut:
                print("\nDone reading file.  Found the following cross sections:")

        if printOut:
            for c in self.crs:
                c.printToScreen()

        #for c in self.crs:
        #    c.plot()

    def writeLXCatFile(self,filename):

        with open(filename,'w') as fp:
            for c in self.crs:
                fp.write(typeDictI2S[c.colType]+'\n')
                fp.write(c.colName+'\n')
                if (typeDictI2S[c.colType] != 'ATTACHMENT'):
                    fp.write("%.6E\n" % c.deltaE)
                fp.write(c.info)
                fp.write('-' * 30 + '\n')
                for k in range(np.size(c.data,0)):
                    fp.write("%.6E\t%.6E\n" % (c.data[k,0], c.data[k,1]))
                fp.write('-' * 30 + '\n\n')


#################################################################
# If standalone invocation, run a test case
#################################################################
if __name__ == '__main__':
    import sys

    #k=1
    #print("  Dataset = {0:s}".format(datasets[k]))
    #tmp = multipleCrossSections("./%s.txt" % datasets[k])

    dataset = "Biagi"
    filename = "./crs/%s.txt" % dataset
    print("="*50)
    print("  Dataset = {0:s}".format(dataset))
    tmp = multipleCrossSections(filename)
    print("  Number of cross sections = ", len(tmp.crs))
    print("="*50)

    output = './crs/testCRS.txt'
    tmp.writeLXCatFile(output)
    print("Done! Results were writen at: ", output)
    
