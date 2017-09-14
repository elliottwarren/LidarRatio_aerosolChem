"""
Plot MURK aerosol for each site, integrated in the vertical

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

