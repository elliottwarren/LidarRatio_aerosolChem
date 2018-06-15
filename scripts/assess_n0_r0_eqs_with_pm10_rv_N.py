"""
Assess the early equations in the aerFO that estiamte N and r from N0 and r0 respectively. Need to understand the
overestimation in /beta_m by the aerFO...
"""



# read in pm10
# read in rv and N
# calculate aerFO rv and N from the equations (dashed lines)
# plot all together


import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pickle
from scipy.stats import pearsonr

import numpy as np
import datetime as dt
import pandas

from copy import deepcopy
import colorsys

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon