import numpy as np
import os
from matplotlib import pyplot as plt

#    mass is in solar units.
#    tphysf is the maximum evolution time in Myr.
#    tb is the orbital period in days.
#    kstar is the stellar type: 0 or 1 on the ZAMS - unless in evolved state. 
#    z is metallicity in the range 0.0001 -> 0.03 where 0.02 is Population I.
#    eccentricity can be anywhere in the range 0.0 -> 1.0.
#
#    neta is the Reimers mass-loss coefficent (neta*4x10^-13: 0.5 normally). 
#    bwind is the binary enhanced mass loss parameter (inactive for single).
#    hewind is a helium star mass loss factor (1.0 normally).
#    alpha1 is the common-envelope efficiency parameter (1.0).  
#    lambda is the binding energy factor for common envelope evolution (0.5).
#
#    ceflag > 0 activates spin-energy correction in common-envelope (0). #defunct#
#    ceflag = 3 activates de Kool common-envelope model (0). 
#    tflag > 0 activates tidal circularisation (1).
#    ifflag > 0 uses WD IFMR of HPE, 1995, MNRAS, 272, 800 (0). 
#    wdflag > 0 uses modified-Mestel cooling for WDs (0). 
#    bhflag > 0 allows velocity kick at BH formation (0). 
#    nsflag > 0 takes NS/BH mass from Belczynski et al. 2002, ApJ, 572, 407 (1). 
#    mxns is the maximum NS mass (1.8, nsflag=0; 3.0, nsflag=1). 
#    idum is the random number seed used by the kick routine. 
#
#
#             0 - deeply or fully convective low mass MS star
#             1 - Main Sequence star
#             2 - Hertzsprung Gap
#             3 - First Giant Branch
#             4 - Core Helium Burning
#             5 - First Asymptotic Giant Branch
#             6 - Second Asymptotic Giant Branch
#             7 - Main Sequence Naked Helium star
#             8 - Hertzsprung Gap Naked Helium star
#             9 - Giant Branch Naked Helium star
#            10 - Helium White Dwarf
#            11 - Carbon/Oxygen White Dwarf
#            12 - Oxygen/Neon White Dwarf
#            13 - Neutron Star
#            14 - Black Hole
#            15 - Massless Supernova

def bse(mass1, mass2, age, P, metallicity, ecc, display='read'):
    cmd = './bse'
    arg0 = '"'+str(mass1)+' '+str(mass2)+' '+str(age)+' '+str(P)+\
    ' 1 1 '+str(metallicity)+' '+str((ecc*(P>12)))+\
    ' 0.5 0.0 1.0 3.0 0.5\
     0 1 0 1 0 1 3.0 29769\
     0.05 0.01 0.02\
     190.0 0.125 1.0 1.5 0.001 10.0 -1.0\
     1 1 1 1 1 1 1 1"'
    if display == 'readlines':
        return os.popen('echo '+arg0+' | '+cmd,'r').readlines()
    else:
        return os.popen('echo '+arg0+' | '+cmd,'r').read()

    
#-------------------------------------------------------------------------------------------------------------------------------

from multiprocessing import Pool

agents = 100
chunksize = 1



def parallel(j):
    n = 100
    np.random.seed(j+np.random.randint(1000))
    x = np.random.rand(n)
    mass1 = ( (0.8**(-1.3)-8**(-1.3))*x + 8**(-1.3) )**(-1/1.3)
    m_ratio = np.random.rand(n)
    mass2 = mass1 * m_ratio
    age = np.random.rand(n) * 12e3
    P = 10**(np.random.normal(5.03,2.28,n))
    metallicity = 0.020*10**(-(age/1e3)**2/50+0.15) + 0.0001#[0.001 for i in range(n)]
    ecc = np.random.rand(n)
    
    so = [None for i in range(n)]
    for i in range(n):
        so[i] = bse(mass1[i], mass2[i], age[i], P[i], metallicity[i], ecc[i], 'readlines')
    
    age_cool = np.zeros(n)
    mass_WD = np.zeros(n)
    for i in range(n):
        if (so[i][-3][30:32]=='11' or so[i][-3][30:32]=='12') and so[i][-3][33:35]=='15':     
            age_cool[i] = (float(so[i][-2][0:12]) - float(so[i][-3][0:12])) /1000
            mass_WD[i] = float(so[i][-3][14:21])
    return [mass1, mass2, age, P, ecc, age_cool, mass_WD]
        

with Pool(processes=agents) as pool:
    result = pool.map(parallel, np.arange(agents), chunksize)

np.save('/datascope/menard/group/scheng/Gaia/WD_bse_binaries.npy',np.array([{'result':result\
                                                                   }]) )
print('all finished')

