import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, votable
from astropy.table import Table, vstack, hstack
import os, sys
from astropy.io import fits
from scipy.interpolate import interp1d, interp2d, CloughTocher2DInterpolator, griddata, LinearNDInterpolator
from scipy.signal import fftconvolve
import WD_models
import importlib
import emcee
import os
from multiprocessing import Pool

agents = 100
chunksize = 1


ndim, nwalkers = 4, 1000
ivar = 1. / np.random.rand(ndim)
p0 = [np.random.rand(ndim)+1 for i in range(nwalkers)]

pec_type = 'DA_thick'
model = 'CO'
WD_model = eval('WD_models.'+spec_type+'_'+model)


# Define Functions
culmulative_star_forming_rate = interp1d((0,11),(11,0),\
                                      fill_value = 1e-13, bounds_error=False) # age [Gyr]
    
vertical_scale = interp1d((0, 1, 5, 7, 11),(60,120,270,350,500),\
                          fill_value = 1e13, bounds_error=False) # age [Gyr]

velocity_scatter = interp1d((0, 1, 6, 7, 11),(6,9,17,18,25),\
                          fill_value = 1e13, bounds_error=False) 

def velocity_density(v, age):    
    return v**2 / velocity_scatter(age)**3 * np.exp(-v**2/2/velocity_scatter(age)**2)
    
def initial_mass_function(mass):
    return mass**(-2.3)

def IFMR(mass_WD):
    return (mass_WD-0.5)/0.75*7+1

dM0_dmass = interp1d(np.arange(0.50,1.35,0.01),IFMR(np.arange(0.51,1.36,0.01))-IFMR(np.arange(0.49,1.34,0.01)),\
                    fill_value = 1e-13, bounds_error=False)

def cooling_density(mass, age):
    density = WD_model['grid_density_func'](\
                      WD_model['grid_bprp_func'](mass, np.log10(age-(IFMR(mass))**(-3.5)*10 )+9), \
                      WD_model['grid_G_func'](mass, np.log10(age-(IFMR(mass))**(-3.5)*10 )+9))
    if ~np.isnan(density):
        return density
    else:
        return 1e-13

def mass_density(mass, age):
    return 1

def density_func(mass, age, v, z):
    if (age>0) * (age<11) * (v>0) * (v<300) * (mass>0.4) * (mass<1.3):
        return culmulative_star_forming_rate(age) *\
            1/vertical_scale(age) * \
            velocity_density(v, age) *\
            initial_mass_function(IFMR(mass)) * dM0_dmass(mass) *\
            cooling_density(mass, age) *\
            mass_density(mass, age)
    else:
        return 1e-13

velocity_scatter = interp1d((0, 1, 6, 7, 11),(6,9,17,18,25),\
                          fill_value = 1e13, bounds_error='extrapolate') 


def lnprob(x,ivar):
    return np.log(density_func(x[0],x[1],x[2],x[3]))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[ivar])

def parallel(i):
    a = sampler.run_mcmc(p0, 1000)
    mass = np.concatenate((mass,a[0][:,0]))
    age =  np.concatenate((age, a[0][:,1]))
    v =  np.concatenate((v, a[0][:,2]))
    bp_rp =  np.concatenate((bp_rp, WD_model['grid_bprp_func'](mass, np.log10(age-(IFMR(mass))**(-3.5)*10 )+9) ))
    G =  np.concatenate((G, WD_model['grid_G_func'](a[0][:,0], np.log10(a[0][:,1]-(IFMR(a[0][:,0]))**(-3.5)*10 )+9) ))
    return [a, mass, age, v, bp_rp, G]



with Pool(processes=agents) as pool:
    result = pool.map(parallel, np.arange(100), chunksize)

np.save('/home/Gaia/WD_HR_synthetic_results.npy',np.array([{'results':result}]))
print('all finished')

#np.save('/home/scheng/forest_ring/catalog_22000.npy',np.array([catalog]))