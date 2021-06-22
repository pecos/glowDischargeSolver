#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:42:18 2021

@author: Laxminarayan L. Raja
@details: Copied from GlowSim (Ar_plasma_highPr_gas.phs)


Overview
========  
This mechanism has been developed for pressure of ~10's - 100's Torr.
It was originally used for moderately high pressure discharge simulations
such as microdischarges


Changelog
=========
2019-08-21 (Doug Breden) Added references and expanded header.
Added referenecs to comment block of each reaction pathway.
Species: E, AR, AR+, AR2+, ARm, AR2m.  14 reaction.
2020-03-31 (KS) Change Ar2+ transport.


References
==========
The full mechanism is found in (units cm3-s^-1)
T Deconinck and LL Raja, "Modeling of Mode Transition Behaviour in Argon
Microhollow Cathode Discharges", Plasma Processes and Polymers, Vol. 6, No. 5, Apr 2009, pp. 335-346

Arrhenius reaction pathways from Deconinck and Raja 2009 are from Table 2 in the paper
B Lay, RS Moss, S Rauf, MJ Kushner, Plasma Sources Sci. Technol. 12, 8 (2003)


Species specification
=====================

E  AR+  ARm  AR
(AR2+  AR2m):  Not used for now


Reactions specification
=======================

G1:   AR    +   E      ->     ARm   +   E                   (    REQUIRED)
G2:   AR    +   E      ->     AR+   +  2E                   (    REQUIRED)
G3:   ARm   +   E      ->     AR+   +  2E                   (    REQUIRED)
G4:   ARm   +   E      ->     AR    +   E                   (    REQUIRED)
G5:  2ARm              ->     E     +   AR  +  AR+          (NOT REQUIRED)
G6:   AR2m  +   E      ->     AR2+  +  2E                   (NOT REQUIRED)
G7:   E     +   AR2m   ->    2AR    +   E                   (NOT REQUIRED)
G8:   ARm   +  2AR     ->     AR2m  +   AR                  (NOT REQUIRED)
G9:   AR+   +  2AR     ->     AR2+  +   AR                  (NOT REQUIRED)
G10:  AR2m             ->    2AR                            (NOT REQUIRED)
G11: 2AR2m             ->     AR2+  +  2AR  +  E            (NOT REQUIRED)
G12:  AR+   +   E      ->     ARm                           (    REQUIRED)
G13:  AR+   +  2E      ->     ARm   +   E                   (    REQUIRED)
G14:  AR2+  +   E      ->     ARm   +   AR                  (NOT REQUIRED)
"""
import numpy as np


class ChemistryArgonPlasma():
    def __init__(self, energy):
        """
        This class will allocate, initialize and construct the global stiffness
        matrix and load vector. Finally, the equation system will be solved.
        """
        self.energy = np.copy(energy)
        self.indFix = np.copy((self.energy[:,0]<=0.0))
        self.energy[self.indFix,0] = np.copy(1.0)

        self.kf   = []
        self.kf_T = []

    def rxnRateCoefficient(self, a, b, Ea):
        """Returns ionization reaction rate constant"""
        kf = a * (self.energy**b) * np.exp(-Ea/self.energy)
        kf[self.indFix,0] = 0
        return kf #a * (energy**b) * np.exp(-Ea/energy)

    def rxnRateCoefficientJac(self, a, b, Ea):
        """Returns derivative of ionization reaction rate constant wrt
        energy
        """
        kf_T = a * (self.energy**(b-1)) * np.exp(-Ea/self.energy) * (b + Ea/self.energy)
        kf_T[self.indFix,0] = 0
        return kf_T #a * (energy**(b-1)) * np.exp(-Ea/energy) * (b + Ea/energy)

    def rxnRateCoefficientG1(self):
        a         = 1.0e-8
        b         = 0.1
        Ea        = 1.3856e5
        self.kf   = self.rxnRateCoefficient(a, b, Ea)
        self.kf_T = self.rxnRateCoefficientJac(a, b, Ea)

    def rxnRateCoefficientG2(self):
        a         = 292.4044117647058 #2.8e-11
        b         = 0.0 #6.2e-01
        Ea        = 18.687*1.5 #1.8712e5
        kf        = self.rxnRateCoefficient(a, b, Ea)
        kf_T      = self.rxnRateCoefficientJac(a, b, Ea)
        return kf, kf_T

    def rxnRateCoefficientG3(self):
        a         = 0.8e-7
        b         = 5.0e-2
        Ea        = 6.0524e4
        self.kf   = self.rxnRateCoefficient(a, b, Ea)
        self.kf_T = self.rxnRateCoefficientJac(a, b, Ea)

    def rxnRateCoefficientG4(self):
        a         = 2.0e-7
        b         = 0.0
        Ea        = 0.0
        self.kf   = self.rxnRateCoefficient(a, b, Ea)
        self.kf_T = self.rxnRateCoefficientJac(a, b, Ea)

    def rxnRateCoefficientG12(self):
        a         = 43.0e-12
        b         = -5.0e-01
        Ea        =  0.0
        self.kf   = self.rxnRateCoefficient(a, b, Ea)
        self.kf_T = self.rxnRateCoefficientJac(a, b, Ea)

    def rxnRateCoefficientG13(self):
        a         =  9.75e-9
        b         = -4.5
        Ea        =  0.0
        self.kf   = self.rxnRateCoefficient(a, b, Ea)
        self.kf_T = self.rxnRateCoefficientJac(a, b, Ea)

# *RXN 
#     AR  +  E     ->      ARm  +  E
#     *PARAM  1.0e-8 0.1 1.3856e5
#     *TDEP E 
#     *EXCI E   11.56
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS	
# 	*COMMENT  
#     *ENDCOMMENT
# *ENDRXN

# *RXN
# 	AR  +  E -> AR+  +  2E
#     *PARAM	2.8e-11 6.2e-01 1.8712e5
# 	*RTEXPRF (TEMP>42450.0)*EXP((-2.15*E(30)/(TEMP^6))+(1.34*E(26)/(TEMP^5))-(3.23*E(21)/(TEMP^4))+(3.66*E(16)/(TEMP^3))-(1.97*E(11)/(TEMP^2))+(1.67*E(5)/TEMP)-29.8)	
# 	*TDEP E	
#     *EXCI E   15.8	
# 	*RTACTVF EXPR	
#     *RXNUNITS MOLECULES-M KELVINS	
# 	*COMMENT  
#         Generated using BOLSIG+ and then fit to a polynomial function of temperature
#     *ENDCOMMENT
# *ENDRXN

# *RXN
# 	ARm  +  E     ->      AR+  +  2E
#     *PARAM	0.8e-7 5.0e-2 6.0524e4	
#     *RTEXPRF (TEMP>30900.0)*EXP((-5.5*E(28)/(TEMP^6))+(5.08*E(24)/(TEMP^5))-(1.85*E(20)/(TEMP^4))+(3.38*E(15)/(TEMP^3))-(3.34*E(10)/(TEMP^2))+(1.39*E(5)/TEMP)-29.7)
# 	*TDEP E	
#     *EXCI E   4.43	
# 	*RTACTVF EXPR	
#     *RXNUNITS MOLECULES-M KELVINS	
# 	*COMMENT  
#         Generated using BOLSIG+ and then fit to a polynomial function of temperature
#     *ENDCOMMENT
# *ENDRXN


# *RXN
# 	ARm  +  E     ->      AR  +  E
#     *PARAM	2.0e-7 0.0 0.0	
#     *RTEXPRF (TEMP>17900.0)*EXP(-(4.82*E(4)/TEMP)-32.46)
# 	*TDEP E  
#     *EXCI E -11.5	
# 	*RTACTVF EXPR	
#     *RXNUNITS MOLECULES-M KELVINS
# 	*COMMENT  
#         Generated using BOLSIG+ and then fit to a polynomial function of temperature
#     *ENDCOMMENT
# *ENDRXN

# *RXN
#     AR+  +  E     ->      ARm
#     *PARAM  43.0e-12 -5.0e-01 0.0
#     *TDEP E 
#     *EXCI AR   -4.3
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT  
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 6
#     *ENDCOMMENT
# *ENDRXN

# *RXN
#     AR+  +  2E     ->      ARm + E
#     *PARAM  9.75e-9 -4.5  0.0
#     *TDEP E 
#     *EXCI E   -4.3
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT  
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 7
#     *ENDCOMMENT
# *ENDRXN






# *RXN
# 	2ARm     ->      E  +  AR  +  AR+	
#     *PARAM	5.0e-10 0.0 0.0	
# 	*TDEP ARm   
#     *EXCI AR -7.2	
# 	*RTACTVF ARRH	
#     *RXNUNITS MOLECULES-CM KELVINS	
# 	*COMMENT  
#         Generated using BOLSIG+ and then fit to a polynomial function of temperature
#     *ENDCOMMENT
# *ENDRXN


# *RXN
#     AR2m  +  E     ->      AR2+  +  2E         
#     *PARAM  1.29e-10 7.0e-01 0.42456e5
#     *TDEP E 
#     *EXCI E   3.66
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT  
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 8
#     *ENDCOMMENT
# *ENDRXN

# *RXN
#     E  +  AR2m     ->      2AR + E           
#     *PARAM  1.0e-7 0.0 0.0
#     *TDEP AR  
#     *EXCI E -10.9 
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT  
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 9
#     *ENDCOMMENT
# *ENDRXN

# *RXN
#     ARm  +  2AR     ->      AR2m + AR       
#     *PARAM  1.14e-32 0.0 0.0
#     *TDEP AR  
#     *EXCI AR -0.6
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT  
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 21
#     *ENDCOMMENT
# *ENDRXN

# *RXN
#     AR+  +  2AR     ->      AR2+ + AR       
#     *PARAM  2.5e-31 0.0 0.0
#     *TDEP AR 
#     *EXCI AR -1.3
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT 
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 22
#     *ENDCOMMENT
# *ENDRXN

# *RXN
#     AR2m     ->      2AR                    
#     *PARAM  6.0e7 0.0 0.0
#     *TDEP AR  
#     *EXCI AR -10.9
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT 
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 24
#     *ENDCOMMENT
# *ENDRXN

# *RXN
#     2AR2m     ->      AR2+ + 2AR + E         
#     *PARAM  5.0e-10 0.0 0.0
#     *TDEP AR 
#     *EXCI AR -7.3
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT  
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 23
#     *ENDCOMMENT
# *ENDRXN

# *RXN
#     AR2+  +  E     ->      ARm + AR         
#     *PARAM  25.9e-6 -0.66  0.0
#     *TDEP E 
#     *EXCI AR   -3.0
#     *RTACTVF ARRH   
#     *RXNUNITS MOLECULES-CM KELVINS
#     *COMMENT  
#         From Lay, Moss, Rauf, Kushner, PSST 12  (2003) 8-21, Table 2, reaction 10
#     *ENDCOMMENT
# *ENDRXN