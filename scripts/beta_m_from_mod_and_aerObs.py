"""
Run the aerFO with UKV data and aerosol observations. Create the control and alternative aerFO runs.
Save the output into a np file and do some plotting afterwards.

Created by Elliott 09 May 2018
"""

import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from copy import deepcopy
import colorsys

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'
    savedir = maindir + 'figures/number_concentration/'

    # data
    ceilMetaDatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'
    ceilDatadir = datadir + 'L1/'
    modDatadir = datadir + model_type + '/'
    rhDatadir = datadir + 'L1/'
    aerDatadir = datadir + 'LAQN/'

    # statistics to run
    pm10_stats = True
    rh_stats = False

    # # instruments and other settings
    #site_rh = FOcon.site_rh
    #site_rh = {'WXT_KSSW': 50.3}
    #rh_instrument = site_rh.keys()[0]

    #site = 'NK'
    #ceil_id = 'CL31-D'
    #ceil = ceil_id + '_' + site

    # instruments and other settings
    #site_rh = FOcon.site_rh
    #site_rh = {'Davis_IMU': 72.8}
    #rh_instrument = site_rh.keys()[0]

    # # pm10
    site = 'NK'
    ceil_id = 'CL31-D'
    # ceil = ceil_id + '_BSC_' + site
    ceil = ceil_id + '_' + site

    # rh
    #site = 'KSS45W'
    #ceil_id = 'CL31-A'
    ## ceil = ceil_id + '_BSC_' + site
    #ceil = ceil_id + '_' + site

    site_bsc = {ceil: FOcon.site_bsc[ceil]}
    # site_bsc = {ceil: FOcon.site_bsc[ceil], 'CL31-E_BSC_NK': 27.0 - 23.2}

    if pm10_stats == True:
        site_aer = {'PM10_'+site: FOcon.site_aer['PM10_'+site]}

    site_bsc_colours = FOcon.site_bsc_colours

    # day list
    # clear sky days (5 Feb 2015 - 31 Dec 2016)
    # daystrList = ['20150414', '20150415', '20150421', '20150611', '20160504']

    # all clear sky days for paper 2 (including about 5 from paper 1 - still need to grab the extra 7)
    daystrList = ['20160504', '20160505', '20160506', '20160823', '20160911','20161125','20161129','20161130','20161204',
         '20170120','20170122','20170325','20170408','20170526','20170828','20161102','20161205','20161227','20161229',
         '20170105','20170117','20170118','20170119','20170121','20170330','20170429','20170522','20170524','20170601',
         '20170614','20170615','20170619','20170620','20170626','20170713','20170717','20170813','20170827','20170902']

    days_iterate = eu.dateList_to_datetime(daystrList)


    # ceilometer gate number to use for backscatter comparison
    # 1 - noisy
    # 2 - more stable
    # see Kotthaus et al (2016) for more.
    ceil_gate_num = 2

    # where to pull rh height out.
    mod_rh_height_idx = 1
