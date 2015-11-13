"""
    attempt to replicate Breyesse, Kovetz, and Kamionkowski (2015)
    we think this combines Pullen et al 2013's HM-SFR relation with SFR/FIR-LCO relations of the sort that we use in li15.py
    for simplicity's sake, we will use the end result relation (2.6)
"""

import os
import numpy as np
import scipy.interpolate

def line_luminosity(halos, line_freq, A=2e-6, b=1., min_mass=1e9):
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

    lco = np.where(
            hm >= min_mass, 
            A*hm**b,
            0. ) # Set all halos below minimum halo mass to have 0 luminosity

    return lco
    # there's also an f_duty technically, but I think that gets absorbed into A even here
