"""
Read in and plot PM10/PM2.5 for an LAQN site in London

Created by Elliott Fri 22/09/2017
"""

import numpy as np
import ellUtils as eu


# -------------
# Setup
# -------------

# directories
maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
datadir = maindir + 'data/LAQN/'
savedir = maindir + 'figures/extras/PMratio/'

# -------------
# Read
# -------------

# get filename
fname = 'PM10_PM2p5_MR_15min_20161005-20170917.csv'
filepath = datadir + fname

# -------------
# Plot
# -------------