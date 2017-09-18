"""
Calculate and plot eq 26 from Clark et al 2008, which explains how the murk aerosol profile is adjusted
based on the vis DA comparison at the surface (S curve with 1.0 at surface and reducing with height)

Currently assumes a constant potential temperature profile in accordance with a dry boundary layer
Hence graph is ONLY applicable within the BL.

Created by Elliott Thurs 14th Sept 2017
"""

import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
from scipy.stats import spearmanr

from copy import deepcopy

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

savedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/figures/'

# constants
p0 = 100000.0 # Pa
g = 9.81
R = 287.0
T0 = 288.15 # 15 degC
cp = 1000 # ish for dry air
theta0 = T0 # theta0 = T0 as p0/p = 1 # assumed a constant potential temperature profile
gamma = -9.6 # dry lapse rate as I deal with clear air

# variables
p = np.arange(p0, 85000, -100)
p_hPa = p/100.0
T = theta0 / ((p0/p)**(R/cp))



# murk adjustment function (eq 26 from Clark et al 2008)
f = np.exp(-(18.0*np.log(p/p0))**2)

# same murk eq but separated out to make sure it was calculated correctly.
f1 = np.log(p/p0)
f2 = np.power((18.0*f1),2)
f3 = np.exp(-f2)

# hydrostatic eq integrated to calc z using p, p0 and T
x1 = (R / g) * T
x2 = np.log(p0/p) # opposite to eq 26 from Clark et al 2008
z = x1 * x2

# plotting

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.plot(f, p_hPa)
ax.set_ylim([p_hPa[0], p_hPa[-1]])
plt.axvline(0.0, alpha=0.1, color='black', linestyle='--')
plt.axvline(1.0, alpha=0.1, color='black', linestyle='--')
ax.set_ylabel('pressure [hPa]')
ax.set_xlabel('f')
plt.savefig(savedir + 'p_murk_profile_adj_eq26_clarkEtAl2008.png')

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.plot(f, z)
plt.axvline(0.0, alpha=0.1, color='black', linestyle='--')
plt.axvline(1.0, alpha=0.1, color='black', linestyle='--')
ax.set_ylabel('height [m]')
ax.set_ylim([z[0], z[-1]])
ax.set_xlabel('f')
plt.savefig(savedir + 'z_murk_profile_adj_eq26_clarkEtAl2008.png')