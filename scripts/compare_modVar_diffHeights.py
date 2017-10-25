"""
Compare model outputs at different heights

Created by Elliott Wed 25 Oct 2016
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import datetime as dt
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from copy import deepcopy
import colorsys

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

def main():

    # ==============================================================================
    # Setup
    # ==============================================================================

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/'
    datadir = maindir + 'data/'
    savedir = maindir + 'figures/' + model_type + '/clearSkyPeriod/'

    # data
    ceilDatadir = datadir + 'L1/'
    modDatadir = datadir + model_type + '/'
    rhDatadir = datadir + 'L1/'
    aerDatadir = datadir + 'LAQN/'

    # statistics to run
    pm10_stats = True
    rh_stats = False

    site = 'MR'
    ceil_id = 'CL31-C'
    # ceil = ceil_id + '_BSC_' + site
    ceil = ceil_id + '_' + site

    # day to compare
    daystrList = ['20160504']

    days_iterate = eu.dateList_to_datetime(daystrList)

    # variable to compare (whatever the mod_site_extract_calc function has named them)
    variable = 'N'

    # define statistics dictionary
    statistics = {}
    sampleSize = 0 # add to this

    # store RH(z = z_i) for each
    var_MR = []
    var_KSS45W = []

    time_MR = []
    time_KSS45W = []









































    return


if __name__ == '__main__':
    main()