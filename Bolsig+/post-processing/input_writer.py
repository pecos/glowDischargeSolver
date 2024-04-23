import numpy as np
from SahaSolver import *
from swarmData import swarmDatasets

order = ['READCOLLISIONS', 'CONDITIONS', 'RUN', 'RUNSERIES', 'SAVERESULTS']

AlAminLucas1987 = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
                   'CONDITIONS': ['VAR', 0., 0., 300., 300., 0., 0., 1.0E18, 1., 1., 1, 1, 1, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
                   'RUN': [56.5, 84.8, 113, 141, 198, 254, 424, 565, 678, 848, 1130, 1413, 1695, 1978, 2260, 2825, 3390, 4238, 5650],
                   'SAVERESULTS': ['"output-AlAminLucas1987.dat"', 3, 1, 1, 0, 0, 0, 0, 0]
                   }
MilloyCrompton1977 = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
                      'CONDITIONS': ['VAR', 0., 0., 294., 294., 0., 0., 1.0E18, 1., 1., 1, 1, 1, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
                      'RUN': [0.0010, 0.0012, 0.0014, 0.0017, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050, 0.0060, 0.0080, 0.010, 0.012, 0.014, 0.017, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080, 0.100],
                      'SAVERESULTS': ['"output-MilloyCrompton1977.dat"', 3, 1, 1, 0, 0, 0, 0, 0]
                      }
NakamuraKurachi1988 = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
                       'CONDITIONS': ['VAR', 0., 0., 300., 300., 0., 0., 1.0E18, 1., 1., 1, 1, 4, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
                       'RUN': [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
                       'SAVERESULTS': ['"output-NakamuraKurachi1988.dat"', 3, 1, 1, 0, 0, 0, 0, 0]
                       }

transport300K = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
                 'CONDITIONS': [10., 0., 0., 300., 300., 0., 0., 1.0E18, 1., 1., 1, 1, 4, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
                 'RUNSERIES': [1, 1.0e-4, 2000., 110, 3],
                 'SAVERESULTS': ['"transport.300K.dat"', 3, 1, 1, 0, 0, 0, 0, 0]
                 }

transport77K = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
                 'CONDITIONS': [10., 0., 0., 77., 77., 0., 0., 1.0E18, 1., 1., 1, 1, 4, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
                 'RUNSERIES': [1, 1.0e-4, 1.0e-2, 30, 3],
                 'SAVERESULTS': ['"transport.300K.dat"', 3, 1, 1, 0, 0, 0, 0, 0]
                 }

transport90K = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
                 'CONDITIONS': [10., 0., 0., 90., 90., 0., 0., 1.0E18, 1., 1., 1, 1, 4, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
                 'RUNSERIES': [1, 1.0e-4, 1.0e-1, 40, 3],
                 'SAVERESULTS': ['"transport.300K.dat"', 3, 1, 1, 0, 0, 0, 0, 0]
                 }

rate300K = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
             'CONDITIONS': [10., 0., 0., 300., 300., 0., 0., 1.0E18, 1., 1., 1, 1, 1, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
             'RUNSERIES': [1, 1.0e0, 5.0e3, 50, 3],
             'SAVERESULTS': ['"transport.300K.dat"', 3, 1, 1, 1, 0, 0, 0, 0]
             }
rate273K = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
             'CONDITIONS': [10., 0., 0., 273.15, 273.15, 0., 0., 1.0E18, 1., 1., 1, 1, 1, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
             'RUNSERIES': [1, 1.0e0, 5.0e3, 50, 3],
             'SAVERESULTS': ['"transport.300K.dat"', 3, 1, 1, 1, 0, 0, 0, 0]
             }
rate273K = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar', 1],
             'CONDITIONS': [10., 0., 0., 273.15, 273.15, 0., 0., 1.0E18, 1., 1., 1, 1, 4, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, 1.0, 1],
             'RUNSERIES': [1, 1.0e0, 5.0e3, 50, 3],
             'SAVERESULTS': ['"transport.300K.dat"', 3, 1, 1, 1, 0, 0, 0, 0]
             }

reaction300K = {'READCOLLISIONS': ['"test-crs.txt"','Ar Ar5 Ar4 Ar3 Ar2 Ar(2p10) Ar(2p9) Ar(2p8) Ar(2p7) Ar(2p6) Ar(2p5) Ar(2p4) Ar(2p3) Ar(2p2) Ar(2p1)', 1],
             'CONDITIONS': [0., 0., 0., 300., 300., 0., 2.5e-6, 8.0E16, 1., 1., 1, 1, 2, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 10000, '0.99999 1e-5', 1],
#             'CONDITIONS': [0., 0., 0., 300., 300., 0., 0.0, 8.0E16, 1., 1., 1, 1, 2, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, '0.99999 1e-5', 1],
             'RUNSERIES': [2, 2.0e-1, 50., 200, 3],
             'SAVERESULTS': ['"reaction_rate.300K.dat"', 3, 1, 1, 1, 0, 0, 0, 1]
             }

reverse300K = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar Ar*', 1],
             'CONDITIONS': [0., 0., 0., 300., 300., 0., 2.5e-6, 8.0E16, 1., 1., 1, 1, 2, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 10000, '0.99999 1e-5', 1],
#             'CONDITIONS': [0., 0., 0., 300., 300., 0., 0.0, 8.0E16, 1., 1., 1, 1, 2, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 1000, '0.99999 1e-5', 1],
             'RUNSERIES': [2, 2.0e-1, 50., 200, 3],
             'SAVERESULTS': ['"reverse_rate.300K.dat"', 3, 1, 1, 0, 1, 0, 0, 1]
             }

torchRxn = {'READCOLLISIONS': ['"test-crs.txt"', 'Ar Ar*', 1],
            'CONDITIONS': [0., 0., 0., 3000., 3000., 0., 1.0e-4, 8.0E16, 1., 1., 1, 1, 2, 0., 200, 0, 200., 1.0e-10, 1.0e-4, 10000, '0.99999 1e-5', 1],
            'RUNSERIES': [2, 4.0e-1, 50., 200, 3],
            'SAVERESULTS': ['"reaction_rate.300K.dat"', 3, 1, 1, 1, 0, 0, 0, 1]
            }

expConfigs = {'AlAminLucas1987': AlAminLucas1987,
              'MilloyCrompton1977': MilloyCrompton1977,
              'NakamuraKurachi1988': NakamuraKurachi1988}

lxcatConfigs = {'transport300K': transport300K,
                'transport77K': transport77K,
                'transport90K': transport90K,
                'rate300K': rate300K,
                'rate273K': rate273K}

def glowDischargeConfigCondition():
    Torr = 133.3  # Pa
    p0 = 1.0 * Torr
    T0 = 300          # K
    Te0 = 45000.      # K
    qe = 1.60217663e-19 # C
    kB = 1.380649e-23 # m2 kg s-2 K-1
    #ne0 = 2.0e15      # m-3
    #nex0 = 7.0e16     # m-3
    #nex0 = np.array([4e16, 2.0e13, 3e16, 1.0e13, 2.0e12, 2.0e12, 2.0e12, 1.5e12, 1.5e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12])
    ne0 = 5.0e14
    #nex0 = np.array([2.5e16, 4e12, 1.0e16, 3.0e12, 1.7e12, 9.42e11, 1.19e12, 1.33e12, 8.96e11, 1.98e12, 7.90e11, 1.63e12, 2.91e12, 3.38e12])
    nex0 = np.array([2.5e16, 4e12, 1.0e16, 3.0e12, 2.0e12, 2.0e12, 1.0e12, 7.5e11, 7.5e11, 5.0e11, 5.0e11, 5.0e11, 1.0e11, 1.0e11])
    nTotal0 = ne0 + (p0 - ne0 * kB * Te0) / kB / T0
    ionDeg = ne0 / nTotal0 # approximated assuming small ionization degree
    Xex0 = nex0 / nTotal0

    bolsigCondition = [0., 0., 0., T0, T0,   \
                       0., ionDeg, ne0, 1.,   \
                       1., 1, 1, 2,              \
                       0., 200, 0, 200.,         \
                       1.0e-10,  1.0e-4, 10000, \
                       '%.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E %.5E'\
                       %(1.0 - sum(Xex0), Xex0[0], Xex0[1], Xex0[2], Xex0[3], Xex0[4], Xex0[5], Xex0[6], Xex0[7], \
                       Xex0[8], Xex0[9], Xex0[10], Xex0[11], Xex0[12], Xex0[13]), 1]

    return bolsigCondition

def torchConfigCondition():
    p0 = 101325         # Pa - Atmospheric Pressure
    T0 = 3000           # K
    Te0 = T0            # K
    qe = 1.60217663e-19 # C
    kB = 8.6173332e-5   # eV/K
    nTotal0 = p0/kB/T0
    ionDeg = solveSaha(p0, T0)
    ne0 = ionDeg*nTotal0
    nex0 = BoltzmannDens(p0, T0) 
    #nTotal0 = ne0 + (p0 - ne0 * kB * Te0) / kB / T0
    #ionDeg = ne0 / nTotal0
    Xex0 = nex0 / nTotal0

    bolsigCondition = [0., 0., 0., T0, T0,  \
                       0., ionDeg, ne0, 1., \
                       1., 1, 1, 2, 0., 200,\
                       0, 200., 1.0e-10,    \
                       1.0e-4, 10000, '%.5E %.5E' % (1.0 - Xex0, Xex0), 1]

    return bolsigCondition

#glowDischargeConfigs = {'reaction300K': reaction300K,
#                        'reverse300K': reverse300K}
glowDischargeConfigs = {'reaction300K': reaction300K}
glowDischargeConfigs['reaction300K']['CONDITIONS'] = glowDischargeConfigCondition()
# glowDischargeConfigs['reverse300K']['CONDITIONS'] = glowDischargeConfigCondition()

torchConfigs = {'reaction': torchRxn}
# torchConfigs['reaction']['CONDITIONS'] = torchConfigCondition() #NOTE(malamast): I commented this out

def writeInputFile(inputFilename, expConfig, crsFile=None, outputFile=None, noscreen = True):
    content = ''
    if (not noscreen): content += '/'
    content += 'NOSCREEN\n\n'

    for key in order:
        if key not in expConfig: continue

        if ( (key == 'READCOLLISIONS') and (crsFile is not None) ):
            if type(crsFile) == list:
                for i in range(len(crsFile)-1):
                    crsFileElem = '"' + crsFile[i] + '"'
                    expConfig[key][0] = crsFileElem
                    content += key + '\n'
                    for val in expConfig[key]:
                        content += str(val) + '\n'
                    content += '\n'
                crsFileElem = '"' + crsFile[-1] + '"'
                expConfig[key][0] = crsFileElem
            else:
                crsFile = '"' + crsFile + '"'
                expConfig[key][0] = crsFile
        elif ( (key == 'SAVERESULTS') and (outputFile is not None) ):
            outputFile = '"' + outputFile + '"'
            expConfig[key][0] = outputFile

        content += key + '\n'
        if (key == 'RUNSERIES'):
            temp = expConfig[key]
            content += str(temp[0]) + '\n'
            content += '%f  %f\n' % (temp[1], temp[2])
            content += str(temp[3]) + '\n'
            content += str(temp[4]) + '\n'
        else:
            for val in expConfig[key]:
                content += str(val) + '\n'
        content += '\n'

    fID = open(inputFilename,'w')
    fID.write(content)
    fID.close()

    return
