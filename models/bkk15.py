"""
    attempt to replicate Breyesse, Kovetz, and Kamionkowski (2015)
    we think this combines Pullen et al 2013's HM-SFR relation with SFR/FIR-LCO relations of the sort that we use in li15.py
    for simplicity's sake, we will use the end result relation (2.6)
"""

import numpy as np
import scipy.integrate as spint
import scipy.interpolate as interp
from astropy.cosmology import WMAP9
import astropy.units

def line_luminosity(halos, line_freq, A=2e-6, b=1., min_mass=1e9, cosmo=WMAP9, zgrid_npts=42):
    """
    Parameters
    halos : HaloList object
        A HaloList object containing all halos and their properties
    line_freq : 
        rest-frame line frequency [GHz]

    Returns
    lco : float array
        halo CO luminosities [Lsun]
    """
    hm          = halos.m
    hz          = halos.zcos
    
    # lazy fduty calculation
    Omega_m,Omega_k,Omega_Lambda = cosmo.Om0,cosmo.Ok0,cosmo.Ode0
    Omega_rad = 1-Omega_m-Omega_k-Omega_Lambda
    lookback_integrand = lambda z:(Omega_rad*(1+z)**4+Omega_m*(1+z)**3+Omega_k*(1+z)**2+Omega_Lambda)**(-0.5)/(1+z)
    age_func = np.vectorize(lambda z:spint.quad(lookback_integrand,z,np.inf)[0])
    zgrid = np.linspace(min(hz),max(hz),zgrid_npts);
    age_func_interp = interp.InterpolatedUnivariateSpline(zgrid,age_func(zgrid))
    h_age = cosmo._hubble_time.to(astropy.units.Gyr).value*age_func_interp(hz)
    fduty       = 0.1/h_age

    lco = np.where(
            hm >= min_mass, 
            A*fduty*hm**b,
            0. ) # Set all halos below minimum halo mass to have 0 luminosity
    return lco
