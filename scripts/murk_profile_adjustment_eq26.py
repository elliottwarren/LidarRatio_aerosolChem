"""
Calculate and plot the

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

fig, ax = plt.subplots(1, 1, figsize=(4,4))
plt.plot(f, p)
ax.set_ylim([p[-1], p[0]])