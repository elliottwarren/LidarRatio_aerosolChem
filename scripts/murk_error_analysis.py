"""
Error analysis to find out how the errors in MURK (that could come from something like the use of the Best scheme...)
impact on estimations of the extinction coefficient.

Created by Elliott Warren 31/08/18
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import datetime as dt
from dateutil import tz
from scipy.stats import spearmanr
from scipy.stats import pearsonr

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

