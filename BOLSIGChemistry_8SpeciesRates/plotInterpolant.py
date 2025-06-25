import sys
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

Nr = 34

rxnNameDict = {     0: "2Ar(m) => 2Ar",
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

reactionExpressionTypelist =  np.array([False,False,False,False,False,False,False,False, # Rxns 1-8
                                        True,True,True,True,True,True,True,True,True,True,True,True,True,True, # Rxns 9-22      
                                        False,False,True,True,True,True,False,False,False,False,True,True]) # Rxns 23-34

thresholded_rxn = np.array([False, False, False, False, False, False, False, False,
                            True, True, True, True, False, True, True, True, True, False, False, False, False,
                            True, False, False, False, False, False, False, False, False, False, False, True,
                            False])

for i in range(Nr):
    if reactionExpressionTypelist[i]:
        if i < 14 or i == 16 or i == 19 or i > 23:
            f = h5.File("./{0:s}.h5".format(rxnNameDict[i]), 'r')
            dataset = f["table"]
        else:
            f = h5.File("./StepExcitation.h5", 'r')
            dataset = f[rxnNameDict[i]]

        Te = dataset[:,0]
        Te /= 11604
        rateCoeff = dataset[:,1]
        if i > 23 and i < 28:
            rateCoeff /= 6.022e23**2
        else:
            rateCoeff /= 6.022e23

        fail_inds = []
        for j in range(len(rateCoeff)):
            if rateCoeff[j] == 0.0 and j > np.nonzero(rateCoeff)[0][0]:
                fail_inds.append(j)

        if len(fail_inds) != 0:
            rateCoeff[0:fail_inds[-1]] = 0.0

        Te *= 1.5
    
        if thresholded_rxn[i] == True:
            I = np.nonzero(rateCoeff)
            diffrateCoeff = [j-i for i,j in zip(rateCoeff[:-1], rateCoeff[1:])]
            diffTe = [j-i for i,j in zip(Te[:-1], Te[1:])]

            Monotonicity = np.asarray([j/i for i,j in zip(diffTe, diffrateCoeff)])
            Monotonicity = np.insert(Monotonicity, 0, 0.0, axis = 0)

            Nan = np.isnan(Monotonicity)
            Inf = np.isinf(Monotonicity)
            indexPositive = np.where(Monotonicity > 0.0)

            Positive = np.full(Monotonicity.shape, False, dtype = bool)
            Positive[indexPositive] = True

            indices = Nan + Inf + Positive
            lastFalse = np.nonzero(rateCoeff)[0][0]
            TeLog = np.log(Te)


            dydx = (rateCoeff[lastFalse+1] - rateCoeff[lastFalse])/(Te[lastFalse+1] - Te[lastFalse])
            #C = Te[lastFalse]**2.0*dydx / rateCoeff[lastFalse]
            C = Te[lastFalse+1]*Te[lastFalse]*np.log(rateCoeff[lastFalse+1]/rateCoeff[lastFalse])**1.5/(Te[lastFalse+1]-Te[lastFalse])
            ALog = np.log(rateCoeff[lastFalse]) + C/Te[lastFalse]
            #ALog2 = np.log(rateCoeff[lastFalse]) + C2/Te[lastFalse]

            rateCoeffLog = np.zeros(rateCoeff.shape)
            rateCoeffLog[lastFalse:] = np.log(rateCoeff[lastFalse:])
            rateCoeffLog[0:lastFalse] = ALog - C/Te[0:lastFalse]
            #rateCoeffLog2 = np.zeros(rateCoeff.shape)
            #rateCoeffLog2[lastFalse:] = np.log(rateCoeff[lastFalse:])
            #rateCoeffLog2[0:lastFalse] = ALog2 - C2/Te[0:lastFalse]

            Te_add = np.linspace(1e-4, Te[0]*0.99, 100)
            TeLog = np.concatenate((np.log(Te_add), TeLog))
            rateCoeffLog_add = np.zeros(100)
            #rateCoeffLog_add2 = np.zeros(100)
            for k in range(len(rateCoeffLog_add)):
                fac = 0.999**(100-k)
                rateCoeffLog_add[k] = rateCoeffLog[0]/fac
                #rateCoeffLog_add2[i] = rateCoeffLog2[0]/fac
            rateCoeffLog = np.concatenate((rateCoeffLog_add, rateCoeffLog))

        else:
            TeLog = np.log(Te)
            rateCoeffLog = np.log(rateCoeff)

        rateExprLog = CubicSpline(TeLog, rateCoeffLog)
        #rateExprLog2 = CubicSpline(TeLog, rateCoeffLog2)
        rateInterp = np.zeros(TeLog.shape[0])
        #rateInterp2 = np.zeros(10000)
        Energy = np.linspace(np.exp(TeLog[0]), Te[-1], rateInterp.shape[0])
        for m in range(len(Energy)):
            rateInterp[m] = rateExprLog(np.log(Energy[m]))
            #rateInterp2[m] = rateExprLog2(np.log(Energy[m]))

        fig, ax = plt.subplots()
        ax.set_title('{}'.format(rxnNameDict[i]))
        ax.set_ylabel('log(Rate Coefficient [m3/s])')
        ax.set_xlabel('Energy [eV]')
        ax.plot(Energy, rateInterp, label = 'Interpolant')
        #ax.plot(Energy, rateInterp2, label = 'Fixed Interpolant')
        ax.plot(np.exp(TeLog), rateCoeffLog, marker = 's', markerfacecolor = 'None', markersize = 4,  label = 'Hack')
        #ax.plot(Te, rateCoeffLog2, marker = 'o', label = 'Fixed Hack')
        ax.plot(Te[lastFalse:], (np.log(rateCoeff[lastFalse:])), marker = 'o', label = 'Raw Data')
        #ax.plot(Te, np.log(rateCoeff), marker = 'o', label = 'Raw Data')
        #ax.set_xlim(4.5, 6.0)
        ax.set_ylim(-50, 1)
        #ax.set_yscale('log')
        #ax.semilogy(Te, rateCoeff/13.56e6*3.22e22, label = 'Original')
        #ax.semilogy(Te, rateInterp, label = 'Interpolated')
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()
        ax.set_title('Interpolant vs Raw Data')
        ax.set_xlabel('Energy [eV]')
        ax.plot(np.exp(TeLog), (rateInterp/rateCoeffLog))
        plt.show()

    else:
        continue

## Electron Transport Data
transport = h5.File("./nominal_transport.h5", 'r')
NDe_v_Te = transport["diffusivity"]
Te_trans = NDe_v_Te[:,0]
Te_trans /= 11604
De_interp = (NDe_v_Te[:,1]/3.22e22)
De_spline = CubicSpline(Te_trans, De_interp)
De_Te_spline = CubicSpline.derivative(De_spline)

Nmue_v_Te = transport["mobility"]
mue_interp = (Nmue_v_Te[:,1]/3.22e22)
mue_spline = CubicSpline(Te_trans, mue_interp)
mue_Te_spline = CubicSpline.derivative(mue_spline)

diffInterp = np.zeros(10000)
mobInterp = np.zeros(10000)
TePlot = np.linspace(Te_trans[0], Te_trans[-1], diffInterp.shape[0])
for i in range(len(diffInterp)):
    diffInterp[i] = De_spline(TePlot[i])
    mobInterp[i] = mue_spline(TePlot[i])

fig,ax = plt.subplots()
ax.set_title('Diffusion Coefficient')
ax.set_xlabel('Te [eV]')
ax.plot(Te_trans, De_interp, marker = 'o', label = 'Raw Data')
ax.plot(TePlot, diffInterp, label = 'Interpolant')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title('Mobility Coefficient')
ax.set_xlabel('Te [eV]')
ax.plot(Te_trans, mue_interp, marker = 'o', label = 'Raw Data')
ax.plot(TePlot, mobInterp, label = 'Interpolant')
ax.legend()
plt.show()
