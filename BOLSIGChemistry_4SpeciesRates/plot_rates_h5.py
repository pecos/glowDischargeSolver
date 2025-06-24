import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors
import h5py as h5

Nr = 3
reactionExpressionTypelist =  np.array([True, True, True])
rxnNameDict = {0: "ionization",
               1: "lumped_1s.excite",
               2: "step_ionization"}

for i in range(Nr):
        if reactionExpressionTypelist[i]:
            f = h5.File("{0:s}.h5".format(rxnNameDict[i]), 'r')
            data = f["table"]

            Te = data[:,0]
            rateCoeff = data[:,1]
            rateCoeff /= 6.022e23

            row_temp = []
            data = []
            header = ["Te(K)", "kf(m3/s)"]
            for j in range(len(Te)):
                row_temp.append(Te[j])
                row_temp.append(rateCoeff[j])
                data.append(row_temp)
                row_temp = []

            g = open("./RateProfile_{0:s}.csv".format(rxnNameDict[i]), 'w')
            writer = csv.writer(g)
            writer.writerow(header)
            writer.writerows(data)
            g.close()

