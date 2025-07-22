import numpy as np
import pandas as pd
from sympy import *

def neutralPartitionFunction(T):
    kB = 8.6173332e-5   # eV/k
    Q_n = 0.0

    data = pd.read_csv('./ElectronicData/ArI.txt', delimiter = '\t')
    for i in range(len(data)):
        Q_n += data["g"][i]*np.exp(-data["Level (eV)"][i]/(kB*T))
    
    return Q_n

def ionPartitionFunction(T):
    kB = 8.6173332e-5   # eV/k
    Q_i = 0.0

    data = pd.read_csv('./ElectronicData/ArII.txt', delimiter = '\t')
    for i in range(len(data)):
        Q_i += data["g"][i]*np.exp(-data["Level (eV)"][i]/(kB*T))


    return Q_i

def solveSaha(p, T):
    a = symbols('a', positive = True)

    m_e = 9.1e-31       # kg
    h = 6.62e-34        # J-s
    kB = 1.380649e-23   # J/K
    E_i = 15.76         # eV - Ionization Energy

    Q_i = ionPartitionFunction(T)
    Q_n = neutralPartitionFunction(T)
    
    ionDeg = solve((a**2/(1-a**2)) - ((2.0/p)*(2*np.pi*m_e/(h**2))**1.5*(kB*T)**2.5*Q_i/Q_n*np.exp(-E_i*11604/T)))
    
    return ionDeg[0]

def BoltzmannDist(p, T):
    sum = 0.0
    kB = 8.6173332e-5   # eV/K
    kB2 = 1.380649e-23  # J/K
    nAr_ex = [[], []]
    nAr = p/kB2/T
    ratio_p = []

    data = pd.read_csv('./ElectronicData/ArI.txt', delimiter = '\t')
    data.drop(data[data["Level (eV)"] >= 13.5].index, inplace = True)

    for i in range(1, len(data)):
        if data["Level (eV)"][i] > 12.0:
            nAr_ex[1].append(data["g"][i]*np.exp(-data["Level (eV)"][i]/(kB*T)))
            sum += data["g"][i]*np.exp(-data["Level (eV)"][i]/(kB*T))
        else:
            nAr_ex[0].append(data["g"][i]*np.exp(-data["Level (eV)"][i]/(kB*T)))
    
    ratio_m = [nAr_ex[0][0]/(nAr_ex[0][0]+nAr_ex[0][2]), nAr_ex[0][2]/(nAr_ex[0][0]+nAr_ex[0][2])]
    ratio_r = [nAr_ex[0][1]/(nAr_ex[0][1]+nAr_ex[0][3]), nAr_ex[0][3]/(nAr_ex[0][1]+nAr_ex[0][3])]
    for j in range(len(nAr_ex[1])):
        ratio_p.append(nAr_ex[1][j]/sum)

    return [nAr_ex, ratio_m, ratio_r, ratio_p]

def BoltzmannDens(p, T):
    kB = 8.6173332e-5   # eV/K
    kB2 = 1.380649e-23  # J/K
    nex0 = 0.0
    nAr = p/kB2/T

    data = pd.read_csv('./ElectronicData/ArI.txt', delimiter = '\t')
    data.drop(data[data["Level (eV)"] >= 13.5].index, inplace = True)

    for i in range(1, len(data)):
        nex0 += data["g"][i]*np.exp(-data["Level (eV)"][i]/(kB*T))
        #nex0.append(data["g"][i]*np.exp(-data["Level (eV)"][i]/(kB*T)))
    return nex0*nAr

def BoltzmannFrac(p, T):
    kB = 8.6173332e-5   # eV/K
    kB2 = 1.380649e-23  # J/K
    nex0 = []
    nAr = p/kB2/T

    data = pd.read_csv('./ElectronicData/ArI.txt', delimiter = '\t')
    data.drop(data[data["Level (eV)"] >= 13.5].index, inplace = True)

    for i in range(1, len(data)):
        nex0.append(data["g"][i]*np.exp(-data["Level (eV)"][i]/(kB*T)))
    return nex0

