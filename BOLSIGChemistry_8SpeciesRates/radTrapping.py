import numpy as np
import scipy.constants as spc

def escapeFactCalc(n_i,E_j,E_i,g_j,g_i,A_ji,Mspecies,T_g,R,L):
    cm_eV = spc.h*spc.c/spc.e*100
    # Calculations for escape factor
    lambda_0 = spc.h*spc.c/((E_j-E_i)*cm_eV*spc.e) # wavelength of transition
    k0 = lambda_0**3*n_i*g_j*A_ji*Mspecies**0.5/(8*np.pi*g_i*(2*spc.k*np.pi*T_g)**0.5) # absorption coefficient at line center, for Doppler absorption
    q0 = R
    Lq = L/(2*q0)              
    # eta = (2 - np.exp(-1e-3*k0*R))/(1 + k0*R) # Mewe (1967)
    # eta = 1.
    if k0*(L/2) > 1 and k0*q0 > 1: # compute escape factor      
        # eta = 1.6/(k0*R*(np.pi*np.log(k0*R))**0.5) # Iordanova/Holstein
        eta = (2/(np.sqrt(np.pi*np.log(k0*L/2))*k0*L)/(2*Lq**2 + 2) + 1/(np.sqrt(np.pi*np.log(k0*q0))*k0*2*q0)* (Lq/(Lq**2 + 1) + np.arctan(Lq))) # Chai & Kwon Doppler lineshape
        # eta = (1/(np.sqrt(np.pi*k0*L/2))*(2/3 - 2*Lq**1.5/(3*(Lq**2 + 1)**0.75))
        #         + 1/(2*np.sqrt(np.pi*k0*q0))*
        #         4*Lq*np.sqrt(Lq)/(3*(Lq**2 + 1)**0.75)) # Golubovskii et al. Lorentz lineshape                       
        # eta = 0.0
    else:
        eta = 1.0    
    
    eta = min(eta,1.0)    
    return eta

Tg = 300.0 #K
p0 = 666.6 #Pa
nAr = p0/Tg/spc.k
M_Ar = 39.948/1000.0/spc.N_A

# 1Torr - 200V
n_tot_m = 2.32e17
n_tot_r = 1.14e17
# 5 Torr - 200V
n_tot_m = 1.07e16
n_tot_r = 8.89e15

#n_m = [5.124e16, 9.76e15]
#n_r = [9.71e16, 9.18e16]
#n_m = [9.16e15, 1.74e15]
#n_r = [5.08e15, 4.80e15]
E_m = [93143.76, 94553.6652]
E_r = [93750.5978, 95399.8276]
E_p = [104102.0990, 105462.7596, 105617.2700, 106087.2598, 106237.5518, 107054.2720, 107131.7086, 107289.7001, 107496.4166, 108722.6194]
g_m = [5,1]
g_r = [3,3]
g_p = [3,7,5,3,5,1,3,5,3,1]
A_rg = [1.32e8, 5.32e8]
relpop_m = [0.84, 0.16]
relpop_r = [0.513, 0.487]
relpop_p = [8.9e-2, 1.99e-1, 1.41e-1, 8.37e-2, 1.39e-1, 2.71e-2, 8.1e-2, 1.34e-1, 8.01e-2, 2.57e-2]
n_m = [n_tot_m*relpop_m[0], n_tot_m*relpop_m[1]]
n_r = [n_tot_r*relpop_r[0], n_tot_r*relpop_r[1]]

A_px = [[1.89e7, 3.3e7, 9.3e6, 5.2e6, 2.45e7, 0.0, 6.3e5, 3.8e6, 6.4e6, 0.0],
        [5.4e6, 0.0, 2.15e7, 2.5e7, 4.9e6, 4.0e7, 2.2e4, 8.5e6, 1.83e6, 2.36e5],
        [9.8e5, 0.0, 0.0, 2.43e6, 0.0, 0.0, 1.86e7, 0.0, 1.17e7, 0.0], 
        [1.9e5, 0.0, 1.47e6, 1.06e6, 5.0e6, 0.0, 1.39e7, 2.23e7, 1.53e7, 4.5e7]]
#A_1s4 = 1.32e8 #1/s
#A_1s2 = 5.32e8 #1/s

#eta_1s4 = escapeFactCalc(nAr, E_1s4, 0.0, 3, 1, A_1s4, M_Ar, Tg, 0.05, 0.05)
#eta_1s2 = escapeFactCalc(nAr, E_1s2, 0.0, 3, 1, A_1s2, M_Ar, Tg, 0.05, 0.05)
eta_rg_low = []
eta_rg_high = []
eta_px_low = [[], [], [], []]
eta_px_high = [[], [], [], []]
eff_A_elem = [[],[]]
eff_A_pm = [0.0, 0.0]
eff_A_pr = [0.0, 0.0]
## Transition from Ar(r) to Ar:
for i in range(len(A_rg)):
    eta_rg_low.append(escapeFactCalc(nAr, E_r[i], 0.0, g_r[i], 1, A_rg[i], M_Ar, Tg, 0.01, 0.01))
    eta_rg_high.append(escapeFactCalc(nAr, E_r[i], 0.0, g_r[i] ,1, A_rg[i], M_Ar, Tg, 0.05, 0.05))

eff_A_rg = [relpop_r[0]*A_rg[0]*eta_rg_high[0] + relpop_r[1]*A_rg[1]*eta_rg_high[1], relpop_r[0]*A_rg[0]*eta_rg_low[0] +
        relpop_r[1]*A_rg[1]*eta_rg_low[1]]

print(eta_rg_low)

## Transitions from Ar(p) to Ar(4s) states:
for i in range(len(A_px)):
    eff_A_temp = [[], []]
    for j in range(len(E_p)):
        if i == 0:
            Ei = E_m[0]
            gi = g_m[0]
            ni = n_m[0]
        elif i == 1:
            Ei = E_r[0]
            gi = g_r[0]
            ni = n_r[0]
        elif i == 2:
            Ei = E_m[1]
            gi = g_m[1]
            ni = n_m[1]
        else:
            Ei = E_r[1]
            gi = g_r[1]
            ni = n_r[1]

        eta_px_low[i].append(escapeFactCalc(ni, E_p[j], Ei, g_p[j], gi, A_px[i][j], M_Ar, Tg, 0.01, 0.01))
        eta_px_high[i].append(escapeFactCalc(ni, E_p[j], Ei, g_p[j], gi, A_px[i][j], M_Ar, Tg, 0.05, 0.05))

        eff_A_temp[0].append(relpop_p[j]*A_px[i][j]*eta_px_high[i][j])
        eff_A_temp[1].append(relpop_p[j]*A_px[i][j]*eta_px_low[i][j])

    if i == 0 or i == 2:
        eff_A_pm[0] += sum(eff_A_temp[0])
        eff_A_pm[1] += sum(eff_A_temp[1])
    else:
        eff_A_pr[0] += sum(eff_A_temp[0])
        eff_A_pr[1] += sum(eff_A_temp[1])


#print(eta_rg)
#print(eta_rg_plus)
print(eta_px_low)
#print(eta_1s4, eta_1s2)
print("Ar(r) -> Ar:")
print('\t'.join(['{:.2e}'.format(x) for x in eff_A_rg]))
print("{:.2e}".format(np.mean(eff_A_rg)))
print("")
print("Ar(4p) -> Ar(m):")
print('\t'.join(['{:.2e}'.format(x) for x in eff_A_pm]))
print("{:.2e}".format(np.mean(eff_A_pm)))
print("")
print("Ar(4p) -> Ar(r):")
print('\t'.join(['{:.2e}'.format(x) for x in eff_A_pr]))
print("{:.2e}".format(np.mean(eff_A_pr)))

