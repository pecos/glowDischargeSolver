import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors
import h5py as h5

tau = (1./13.56e6)
nAr = 3.22e22
np0 = 8e16
Nr = 23
iSample = 0

reactionExpressionTypelist =  np.array([False, False, False, False, False, False, False, False, False,
                                        True, True, True, True, False, True, True, True, False, True, True, False, True,
                                        True])

rxnNameDict = {0: "2AR(m) => 2AR",
               1: "AR(m) + AR(r) => E + AR+ + AR",
               2: "2AR(4p) => E + AR+ + AR",
               3: "2AR(m) => E + AR+ + AR",
               4: "AR(m) + AR => 2AR",
               5: "AR(r) => AR",
               6: "AR(4p) => AR",
               7: "AR(4p) => AR(m)",
               8: "AR(4p) => AR(r)",
               9: "lumped.metastable",
               10: "lumped.resonance",
               11: "lumped.2p",
               12: "ionization",
               13: "E + AR(m) => E + AR",
               14: "step_ionization",
               15: "E + Ar(m) => E + Ar(r)",
               16: "E + Ar(m) => E + Ar(4p)",
               17: "E + Ar(4p) => 2E + Ar+",
               18: "E + Ar(4p) => E + Ar(r)",
               19: "E + Ar(4p) => E + Ar(m)",
               20: "E + Ar(r) => E + Ar",
               21: "E + Ar(r) => E + Ar(m)",
               22: "E + Ar(r) => E + Ar(4p)"}

reactionsList = []
	# import h5py as h5
	# rxnName = ["Ionization", ...] <- dictionary containing reaction name root string
	# for r in range(Nr):
	#	if reactionExpressionTypeList[r]:
	#		fileName = "{0:s}.{1:08d}.h5".format(rxnName[r], iSample)
	#		f = h5.File(filename, "r")
	#		D = f["table"]

for i in range(Nr):
        if reactionExpressionTypelist[i]:
            Nsample = 1
            N300 = 200

            root_dir = "./"
            if (i < 15):
                f = h5.File("{0:s}.h5".format(rxnNameDict[i]), 'r')
                data = f["table"]
            else:
                f = h5.File("StepwiseExcitations.nominal.h5", 'r')
                data = f[rxnNameDict[i]]

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


            #rateCoeff = np.reshape(rateCoeff,[Nsample, N300]).T[:,iSample]
            #rate_file.close()

            #Te = data[:,0]
            
            Te /= 11604
            #Te = np.reshape(Te,[Nsample, N300]).T[:,iSample]
            #print(Te)
            #temp_file.close()

            # Sorting mean energy array and rate coefficient array based on
            # the mean energy array.
            Teinds = Te.argsort()
            rateCoeff = rateCoeff[Teinds]
            Te = Te[Teinds]

            # Find duplicates
            TeDuplicateinds = np.where(np.abs(np.diff(Te, axis=0)) > 0.0)
            TeDuplicateindsForLog = np.where(np.abs(np.diff(Te, axis=0)) == 0.0)
            rateCoeff = rateCoeff[TeDuplicateinds]
            Te = Te[TeDuplicateinds]

            # Nondimensionalization of mean energy.
            Te *= 1.5

            # Find first non-zero value of the coefficient rate.
            I = np.nonzero(rateCoeff)

            diffRateCoeff = [j-i for i, j in zip(rateCoeff[:-1], rateCoeff[1:])]
            diffTe = [j-i for i, j in zip(Te[:-1], Te[1:])]

            Monotonicity = np.asarray([j/i for i, j in zip(diffTe, diffRateCoeff)])
            Monotonicity = np.insert(Monotonicity, 0, 0.0, axis=0)

            Nan = np.isnan(Monotonicity)
            Inf = np.isinf(Monotonicity)
            indexPositive = np.where(Monotonicity>0.0)
            Positive = np.full(Monotonicity.shape, False, dtype=bool)
            Positive[indexPositive] = True

            indices = Nan + Inf + Positive

            #lastFalse = np.where(indices==False)[-1][-1] + 2
            for k in range(len(Te)):
                if (Te[k] < 4.5 and indices[k] == False):
                    lastFalse = k + 2

            # Transformation to log scale.
            TeLog = np.log(Te)

            # Compute the slope of the rate coefficient between its first two non-zero values.
            # Finite differences are used.
            dydx = (rateCoeff[lastFalse + 1] - rateCoeff[lastFalse]) \
                 / (Te[lastFalse + 1] - Te[lastFalse])

            # Arrhenius form: kf = A * exp(-C / Te)
            C = Te[lastFalse]**2.0*dydx / rateCoeff[lastFalse]

            # Compute pre-exponential coefficient, A, in log scale.
            ALog = np.log(rateCoeff[lastFalse]) + C / Te[lastFalse]

            # Transform rate coefficient in log scale.
            rateCoeffLog = np.zeros(rateCoeff.shape)
            rateCoeffLog[lastFalse:] = np.log(rateCoeff[lastFalse:])
            # For the troublesome values, we use the Arrhenius form.
            rateCoeffLog[0:lastFalse] = ALog - C / Te[0:lastFalse]
            # Nondimensionalization in log scale.
            #if i < 2:
            #    rateCoeffLog += - np.log(1.0/tau) + np.log(nAr)
            #else:
            #    rateCoeffLog += - np.log(1.0/tau) + np.log(np0)

            # Interpolation in log scale.
            reactionExpressionsLog = CubicSpline(TeLog, rateCoeffLog)
            # Gradient in log scale
            reactionTExpressionsLog = CubicSpline.derivative(reactionExpressionsLog)

            #reaction = Reaction(rxnAlfa = params.alfa[:,[i]], rxnBeta = params.beta[:,[i]],
            #                    rxnBolsig = reactionExpressionTypelist[i],
            #                    kf_log = reactionExpressionsLog,
            #                    kf_T_log = reactionTExpressionsLog)
            #reactionsList.append(reaction)


            # rxn   = eval("lambda energy :" + reactionExpressionslist[i])
            # rxn_T = eval("lambda energy :" + reactionTExpressionslist[i])
            # setting the axes at the centre
            #fig ,ax = plt.subplots(figsize=(9, 6))
            #ax.spines["top"].set_visible(True)
            #ax.spines["right"].set_visible(True)
            #ax.set_yscale('log')
            #ax.set_xscale('log')

            # plot the function
            #plt.plot(rateCoeffXFiner, np.exp(reactionExpressions_cubicSplineDerivative_log(rateCoeffXFiner)),
            #   color='salmon', linestyle='--', label='interBolsig')
            #plt.plot(rateCoeffXFine, np.exp(reactionExpressions_cubicSpline_log(rateCoeffXFine)),
            #   color='lightgreen', linestyle='--', label='interBolsig')
            #plt.plot(Te[:,0], reactionTExpressionsLogFiltered(TeLog[:]) * np.exp(reactionExpressionsLog(TeLog[:])) / Te[:,0],
            #   color='blue', linestyle='-', label='interBolsig')
            #plt.plot(Te, np.exp(reactionExpressionsLog(TeLog[:])),
            #   color='green', linestyle='-', label='Bolsig')
            #plt.plot(Te, rxn(Te),
            #   color='salmon', linestyle='--', label='Liu')
            #plt.plot(Te[:,0], rxn_T(Te[:,0]),
            #   color='red', linestyle='--', label='interBolsig')
            #plt.xlim((0.05,100))
            #plt.ylim((1e-50,300))
            #plt.legend()
            #plt.savefig("./PlotRates_%s.pdf" %str(i), dpi=300)
            #plt.xlim((-0.0001,0.0255))
            #plt.show()

            # Export rates to CSV Files
            #header = ['Te', 'kf']
            #row_temp = []
            #data = []
            #for j in range(len(Te)):
            #    row_temp.append(Te[j])
            #    row_temp.append(np.exp(reactionExpressionsLog(TeLog[j])))
            #    data.append(row_temp)
            #    row_temp = []
            
            #f = open("./PlotRates_Rxn%s.csv" %str(i), 'w')
            #writer = csv.writer(f)
            #writer.writerow(header)
            #writer.writerows(data)
           # f.close()
