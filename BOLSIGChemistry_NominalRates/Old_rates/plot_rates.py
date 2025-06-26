import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors

tau = (1./13.56e6)
nAr = 3.22e22
np0 = 8e16
Nr = 9
iSample = 0

reactionExpressionTypelist =  np.array([True,True,True,False,False,False,False,False,False])

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
            rate_file = open("{0:s}reaction300K_{1:d}.txt".format(root_dir, i), 'r', encoding='utf-8-sig')
            temp_file = open('reaction300K_Te.txt', 'r', encoding='utf-8-sig')

            rateCoeff = np.genfromtxt(rate_file, dtype='float')
            rateCoeff = np.reshape(rateCoeff,[Nsample, N300]).T[:,iSample]
            rate_file.close()

            Te = np.genfromtxt(temp_file, dtype='float')
            print(Te)
            Te = np.reshape(Te,[Nsample, N300]).T[:,iSample]
            print(Te)
            temp_file.close()

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
            # Te *= 1.5

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

            lastFalse = np.where(indices==False)[-1][-1] + 2

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
            fig ,ax = plt.subplots(figsize=(9, 6))
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.set_yscale('log')
            ax.set_xscale('log')

            # plot the function
            #plt.plot(rateCoeffXFiner, np.exp(reactionExpressions_cubicSplineDerivative_log(rateCoeffXFiner)),
            #   color='salmon', linestyle='--', label='interBolsig')
            #plt.plot(rateCoeffXFine, np.exp(reactionExpressions_cubicSpline_log(rateCoeffXFine)),
            #   color='lightgreen', linestyle='--', label='interBolsig')
            #plt.plot(Te[:,0], reactionTExpressionsLogFiltered(TeLog[:]) * np.exp(reactionExpressionsLog(TeLog[:])) / Te[:,0],
            #   color='blue', linestyle='-', label='interBolsig')
            plt.plot(Te, np.exp(reactionExpressionsLog(TeLog[:])),
               color='green', linestyle='-', label='Bolsig')
            #plt.plot(Te, rxn(Te),
            #   color='salmon', linestyle='--', label='Liu')
            #plt.plot(Te[:,0], rxn_T(Te[:,0]),
            #   color='red', linestyle='--', label='interBolsig')
            plt.xlim((0.05,100))
            plt.ylim((1e-50,300))
            plt.legend()
            plt.savefig("./PlotRates_%s.pdf" %str(i), dpi=300)
            #plt.xlim((-0.0001,0.0255))
            plt.show()

            # Export rates to CSV Files
            header = ['Te', 'kf']
            row_temp = []
            data = []
            for j in range(len(Te)):
                row_temp.append(Te[j])
                row_temp.append(np.exp(reactionExpressionsLog(TeLog[j])))
                data.append(row_temp)
                row_temp = []
            
            f = open("./PlotRates_Rxn%s.csv" %str(i), 'w')
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
            f.close()
