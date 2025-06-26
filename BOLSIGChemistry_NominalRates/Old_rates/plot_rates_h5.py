import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors
import h5py as h5

Nr = 34

reactionExpressionTypelist =  np.array([False,False,False,False,False,False,False,False, # Rxns 1-8
                                        True,True,True,True,True,True,True,True,True,True,True,True,True,True, # Rxns 9-22
                                        False,False,True,True,True,True,False,False,False,False,True,True]) # Rxns 23-34
                                        
rxnNameDict = { 0: "2Ar(m) => 2Ar",
                1: "Ar(m) + Ar(r) => E + Ar+ + Ar",
                2: "2Ar(4p) => E + Ar+ + Ar",
                3: "2Ar(m) => E + Ar+ + Ar",
                4: "Ar(m) + Ar => 2Ar",
                5: "Ar(r) => Ar",
                6: "Ar(4p) => Ar(m)",
                7: "Ar(4p) => Ar(r)",
                8: "Excitation_Metastable",
                9: "Excitation_Resonant",
               10: "Excitation_4p",
               11: "Ionization",
               12: "DeExcitation_Metastable",
               13: "StepIonization_Metastable",
               14: "E + Ar(m) => E + Ar(r)",
               15: "E + Ar(m) => E + Ar(4p)",
               16: "StepIonization_4p",
               17: "E + Ar(4p) => E + Ar(r)",
               18: "E + Ar(4p) => E + Ar(m)",
               19: "DeExcitation_Resonant",
               20: "E + Ar(r) => E + Ar(m)",
               21: "E + Ar(r) => E + Ar(4p)",
               22: "E + Ar+ => Ar(m)",
               23: "E + Ar+ => Ar(4p)",
               24: "3BdyRecomb_Metastable",
               25: "3BdyRecomb_Resonant",
               26: "3BdyRecomb_4p",
               27: "3BdyRecomb_Ground",
               28: "Ar + Ar(4p) => Ar + Ar(m)",
               29: "Ar + Ar(4p) => Ar + Ar(r)",
               30: "Ar(m) + Ar(4p) => E + Ar+ + Ar",
               31: "Ar(r) + Ar(4p) => E + Ar+ + Ar",
               32: "StepIonization_Resonant",
               33: "DeExcitation_4p"}

for i in range(Nr):
        if reactionExpressionTypelist[i]:
            if i < 14 or i == 16 or i == 19 or i > 23:
                f = h5.File("{0:s}.h5".format(rxnNameDict[i]), 'r')
                data = f["table"]
            else:
                f = h5.File("StepExcitation.h5", 'r')
                data = f[rxnNameDict[i]]

            Te = data[:,0]
            Te /= 11604
            rateCoeff = data[:,1]
            if i > 23 and i < 28:
                rateCoeff /= 6.022e23**2
            else:
                rateCoeff /= 6.022e23

            row_temp = []
            data = []
            header = ["", "Te(eV)", "", "kf(m3/s)", ""]
            for j in range(len(Te)):
                row_temp.append('(')
                row_temp.append(Te[j])
                row_temp.append(',')
                row_temp.append(rateCoeff[j])
                row_temp.append(')')
                data.append(row_temp)
                row_temp = []

            g = open("./CSVs/RateProfile_{}.csv".format(i + 1), 'w')
            writer = csv.writer(g)
            writer.writerow(header)
            writer.writerows(data)
            g.close()
        else:
            continue

row_temp = []
data = []

h = h5.File('nominal_transport.h5', 'r')
Nmu = h["mobility"]
Te_trans = Nmu[:,0]
Te_trans /= 11604
mobility = Nmu[:,1]

header = ["", "Te(eV)", "", "Nmu(VMS)", ""]
for k in range(len(Te_trans)):
    row_temp.append('(')
    row_temp.append(Te_trans[k])
    row_temp.append(',')
    row_temp.append(mobility[k])
    row_temp.append(')')
    data.append(row_temp)
    row_temp = []

z = open("./CSVs/Transport.csv", 'w')
writer = csv.writer(z)
writer.writerow(header)
writer.writerows(data)
z.close()

