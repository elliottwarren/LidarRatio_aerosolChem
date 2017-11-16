"""
Compare modelled attenuated backscatter that does/doesn't use observed total number concentration from London

Created by Elliott Thurs 26 Oct 2017
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter

import numpy as np
import datetime as dt
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from copy import deepcopy
import colorsys

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon


def read_headers(filepath):
    """
    A very specific function for the N(D) data from the TSI APS at NK during ClearfLo
    The headers are a bit awkward so, this function finds and pulls them based on the first
    header being 'Date' and no other lines beginning with it.

    :param Nfilepath:
    :return: headers
    """

    f = open(filepath, 'r')
    line = f.readline()
    while line.startswith('Time 5.0125') != True:
        line = f.readline()

    f.close()

    # remove '\n' off the end of [line]
    line = line[:-2]

    split = np.array(line.split(' '))  # split headers up and turn into numpy array
    # remove empty headers and '\n'
    idx = np.logical_or(split == '', split == '\n')  # bad headers
    headers = split[~idx]  # remove bad headers

    return headers

def create_stats_entry(site_id, statistics={}):

    """
    Define or expand the almighty statistics array

    :param site_bsc:
    :param mbe_limit_max:
    :param mbe_limit_step:
    :return: statistics (dict)

    statistics[site]['r'] = [...]
    statistics[site]['MBE'] = {'0-500': ..., '500-1000': ...}
    statistics[site]['time'] = [...]
    """

    # Structure of statistics:
    # statistics[site]['r'] = [...]
    # statistics[site]['MBE'] = {'0-500': ..., '500-1000': ...}
    # statistics[site]['time'] = [...]

    if site_id not in statistics:

        # define site based lists to store the correlation results in
        statistics[site_id] = {'r': [], 'p': [],
                               'diff': [],
                               'aer_diff': [],
                               'aer_mod': [],
                               'aer_obs': [],
                               'rh_diff': [],
                               'rh_mod': [],
                               'rh_obs': [],
                               'back_diff_log': [],
                               'back_diff_norm': [],
                               'ratio': [],
                               'back_obs': [],
                               'back_mod': [],
                               'RMSE': [],
                               'MBE': [],
                               'hr': [],
                               'datetime':[]}

    return statistics

def dateList_to_datetime(dayList):

    """ Convert list of string dates into datetimes """

    datetimeDays = []

    for d in dayList:

        datetimeDays += [dt.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]))]

    return datetimeDays

def get_nearest_ceil_mod_height_idx(mod_height, obs_height, ceil_gate_num):

    """
    Returns the ceilometer height index for the ceilometer gate number, and the idx for
    the nearest model height.

    :param mod_height:
    :param obs_height:
    :param ceil_gate_num:
    :return:
    """

    # idx and height for ceil
    ceil_gate_idx = ceil_gate_num - 1

    ceil_gate_height = obs_height[ceil_gate_idx]

    # find nearest height and idx for mod
    a = eu.nearest(mod_height, ceil_gate_height)

    mod_pair_height = a[0]
    mod_height_idx = a[1]
    height_diff = a[2]


    return ceil_gate_idx, mod_height_idx



def plot_multiple_back_point_diffs(day, aerFO_version, savedir, site_id,
                                   statistics_modN_modr, statistics_modN_pm10r, statistics_obsN_pm10r,
                                   statistics_obsN_modr):

    # Line plots of each
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.axhline(linestyle='--', color='blue', alpha=0.3)  # 0 line
    plt.plot_date(statistics_modN_modr[site_id]['datetime'], statistics_modN_modr[site_id]['back_diff_norm'],
                  label='modN modr', fmt='-')
    plt.plot_date(statistics_modN_pm10r[site_id]['datetime'], statistics_modN_pm10r[site_id]['back_diff_norm'],
                  label='modN pm10r', fmt='-')
    plt.plot_date(statistics_obsN_pm10r[site_id]['datetime'], statistics_obsN_pm10r[site_id]['back_diff_norm'],
                  label='obsN pm10r', fmt='-')
    plt.plot_date(statistics_obsN_modr[site_id]['datetime'], statistics_obsN_modr[site_id]['back_diff_norm'],
                  label='obsN modr', fmt='-')

    # prettify
    plt.xlabel('Time [HH]')
    plt.xlim([np.min(statistics_modN_modr[site_id]['datetime']), np.max(statistics_modN_modr[site_id]['datetime'])])
    plt.ylabel(r'$\beta \/\/Difference \/[m^{-1} sr^{-1}]$')
    ax.set_yticklabels(["{:.1e}".format(t) for t in ax.get_yticks()])  # standard form
    plt.legend()
    plt.suptitle(day.strftime('%Y-%m-%d') + '; modN_0 = 10668 cm-3; ' + aerFO_version)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Is it the Best or MORUSES urban scheme?
    if day < dt.datetime(2016, 3, 15):
        eu.add_at(ax, r'$Best$', loc=2)
    else:
        eu.add_at(ax, r'$MORUSES$', loc=2)

    plt.tight_layout(pad=2)
    plt.savefig(savedir +'betadiff_' +day.strftime('%Y%m%d') + '_modN_obsN_pm10r_combo.png')

    plt.close('all')

    return

def plot_multiple_back_point_ratio(day, aerFO_version, savedir, site_id,
                                   statistics_modN_modr, statistics_modN_pm10r, statistics_obsN_pm10r,
                                   statistics_obsN_modr):

    # Line plots of each
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.axhline(y=1.0, linestyle='--', color='blue', alpha=0.3)  # 0 line
    plt.plot_date(statistics_modN_modr[site_id]['datetime'], statistics_modN_modr[site_id]['ratio'],
                  label='modN modr', fmt='-')
    plt.plot_date(statistics_modN_pm10r[site_id]['datetime'], statistics_modN_pm10r[site_id]['ratio'],
                  label='modN pm10r', fmt='-')
    plt.plot_date(statistics_obsN_pm10r[site_id]['datetime'], statistics_obsN_pm10r[site_id]['ratio'],
                  label='obsN pm10r', fmt='-')
    plt.plot_date(statistics_obsN_modr[site_id]['datetime'], statistics_obsN_modr[site_id]['ratio'],
                  label='obsN modr', fmt='-')

    # prettify
    plt.xlabel('Time [HH]')
    plt.xlim([np.min(statistics_modN_modr[site_id]['datetime']), np.max(statistics_modN_modr[site_id]['datetime'])])
    plt.ylabel(r'$\beta \/\/Ratio \/[\beta_{m} / \beta_{o}]$')
    # ax.set_yticklabels(["{:.1e}".format(t) for t in ax.get_yticks()])  # standard form
    plt.legend()
    plt.suptitle(day.strftime('%Y-%m-%d') + '; modN_0 = 10668 cm-3; ' + aerFO_version)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Is it the Best or MORUSES urban scheme?
    if day < dt.datetime(2016, 3, 15):
        eu.add_at(ax, r'$Best$', loc=2)
    else:
        eu.add_at(ax, r'$MORUSES$', loc=2)

    plt.tight_layout(pad=2)
    plt.savefig(savedir +'betaratio_' +day.strftime('%Y%m%d') + '_modN_obsN_pm10r_combo.png')

    plt.close('all')

    return


def plot_Ns(day, aerFO_version, savedir, site_id,
            mod_data_modN_modr, mod_data_modN_pm10r, mod_data_obsN_pm10r,
            mod_data_obsN_modr):

    # Line plots of each N
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.plot_date(mod_data_modN_modr[site_id]['time'], mod_data_modN_modr[site_id]['N'][:, 1],
                  label='modN', fmt='-')
    plt.plot_date(mod_data_obsN_modr[site_id]['time'], mod_data_obsN_modr[site_id]['N'][:, 1],
                  label='obsN', fmt='-')

    # prettify
    plt.xlabel('Time [HH]')
    plt.xlim([np.min(mod_data_modN_modr[site_id]['time']), np.max(mod_data_modN_modr[site_id]['time'])])
    plt.ylabel(r'$Total number concentration [m-3]$')
    ax.set_yticklabels(["{:.1e}".format(t) for t in ax.get_yticks()])  # standard form
    plt.legend()
    plt.suptitle(day.strftime('%Y-%m-%d') + '; modN_0 = 10668 cm-3; ' + aerFO_version)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Is it the Best or MORUSES urban scheme?
    if day < dt.datetime(2016, 3, 15):
        eu.add_at(ax, r'$Best$', loc=2)
    else:
        eu.add_at(ax, r'$MORUSES$', loc=2)

    plt.tight_layout(pad=2)
    plt.savefig(savedir + 'N_' + day.strftime('%Y%m%d') + '_modN_obsN_pm10r_combo.png')

    plt.close('all')

    return

def plot_r_mds(day, aerFO_version, savedir, site_id,
            mod_data_modN_modr, mod_data_modN_pm10r, mod_data_obsN_pm10r,
            mod_data_obsN_modr):

    # Line plots of each N
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.plot_date(mod_data_modN_modr[site_id]['time'], mod_data_modN_modr[site_id]['r_md'][:, 1] ,
                  label='modr', fmt='-')
    plt.plot_date(mod_data_modN_pm10r[site_id]['time'], mod_data_modN_pm10r[site_id]['r_md'][:, 1],
                  label='pm10r', fmt='-')

    # prettify
    plt.xlabel('Time [HH]')
    plt.xlim([np.min(mod_data_modN_modr[site_id]['time']), np.max(mod_data_modN_modr[site_id]['time'])])
    plt.ylabel(r'$radius [m]$')
    ax.set_yticklabels(["{:.1e}".format(t) for t in ax.get_yticks()])  # standard form
    plt.legend()
    plt.suptitle(day.strftime('%Y-%m-%d') + '; modN_0 = 10668 cm-3; ' + aerFO_version)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Is it the Best or MORUSES urban scheme?
    if day < dt.datetime(2016, 3, 15):
        eu.add_at(ax, r'$Best$', loc=2)
    else:
        eu.add_at(ax, r'$MORUSES$', loc=2)

    plt.tight_layout(pad=2)
    plt.savefig(savedir + 'r_md_' + day.strftime('%Y%m%d') + '_modN_obsN_pm10r_combo.png')

    plt.close('all')

    return

def main():

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
    daystrList = ['20120202']

    days_iterate = dateList_to_datetime(daystrList)


    # ceilometer gate number to use for backscatter comparison
    # 1 - noisy
    # 2 - more stable
    # see Kotthaus et al (2016) for more.
    ceil_gate_num = 2

    # where to pull rh height out.
    mod_rh_height_idx = 1


    # ==============================================================================
    # Read data
    # ==============================================================================

    # Read Ceilometer metadata

    # ceilometer list to use
    ceilsitefile = 'CeilsCSVfull.csv'
    ceil_metadata = FO.read_ceil_metadata(ceilMetaDatadir, ceilsitefile)

    ceil_data_i = {site: ceil_metadata[site]}

    for day in days_iterate:

        # define statistics dictionary
        statistics_modN_modr = {}
        statistics_modN_pm10r = {}
        statistics_obsN_pm10r = {}
        statistics_obsN_modr = {}
        sampleSize = 0  # add to this

        print 'day = ' + day.strftime('%Y-%m-%d')


        # Read UKV forecast and automatically run the FO

        # extract MURK aerosol and calculate RH for each of the sites in the ceil metadata
        # reads all london model data, extracts site data, stores in single dictionary

        # testing with Chilbolton
        mod_data_obsN_modr = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=0.2,
                        allvars=True, obsN_site='Chilbolton', obsr_site='Chilbolton')

        # 1
        mod_data_modN_modr = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=0.2,
                        allvars=True)

        #print 'modN_pm10r started...'
        # 2
        mod_data_modN_pm10r = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=0.2,
                         allvars=True, pm10r_site='NK')
        #
        # #print 'obsN_pm10r started...'
        # #
        mod_data_obsN_pm10r = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=0.2,
                         allvars=True, obsN_site='NK', pm10r_site='NK')
        #
        # #
        mod_data_obsN_modr = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=0.2,
                         allvars=True, obsN_site='NK')


        # Read ceilometer backscatter

        # will only read in data if the site is there!
        # ToDo Remove the time sampling part and put it into its own function further down.
        bsc_obs = FO.read_ceil_obs(day, site_bsc, ceilDatadir, mod_data_modN_modr, calib=True)

        #
        # if pm10_stats == True:
        #     # read in PM10 data and extract data for the current day
        #     pm10 = FO.read_pm10_obs(site_aer, aerDatadir, mod_data)
        #
        # # read in RH data
        # if rh_stats == True:
        #     rh_obs = FO.read_all_rh_obs(day, site_rh, rhDatadir, mod_data)

        # ==============================================================================
        # Process
        # ==============================================================================

        if ceil in bsc_obs:

            bsc_site_obs = bsc_obs[ceil]

            # short site id that matches the model id
            site_id = site.split('_')[-1]
            print '     Processing for site: ' + site_id


            ceil_height_idx, mod_height_idx =\
                 get_nearest_ceil_mod_height_idx(mod_data_modN_modr[site_id]['level_height'], bsc_site_obs['height'], ceil_gate_num)

            # create entry in the dictionary if one does not exist
            statistics_modN_modr = create_stats_entry(site_id, statistics_modN_modr)
            statistics_modN_pm10r = create_stats_entry(site_id, statistics_modN_pm10r)
            statistics_obsN_pm10r = create_stats_entry(site_id, statistics_obsN_pm10r)
            statistics_obsN_modr = create_stats_entry(site_id, statistics_obsN_modr)

            # for the different combos of mod_data
            for statistics, mod_data_set in zip((statistics_modN_modr, statistics_modN_pm10r, statistics_obsN_pm10r, statistics_obsN_modr),
                                                (mod_data_modN_modr,   mod_data_modN_pm10r,   mod_data_obsN_pm10r, mod_data_obsN_modr)):

                # for each hour possible in the day
                for t in np.arange(0, 24):

                    hr = str(t)

                    # add the hour to statstics dict
                    statistics[site_id]['hr'] += [t]
                    statistics[site_id]['datetime'] += [mod_data_set[site]['time'][t]]

                    # get murk_i and rh_i regardless if pm10_stats and rh_stats is True or not
                    murk_i = mod_data_set[site]['aerosol_concentration_dry_air'][t, 0]  # 0th height = 5 m
                    rh_mod_i = mod_data_set[site]['RH'][t, mod_rh_height_idx] * 100.0  # convert from [fraction] to [%]

                    obs_back_i = bsc_site_obs['backscatter'][t, ceil_height_idx]
                    mod_back_set_i = mod_data_set[site]['backscatter'][t, mod_height_idx]

                    statistics[site_id]['back_obs'] += [obs_back_i]
                    statistics[site_id]['back_mod'] += [mod_back_set_i]

                    statistics[site_id]['aer_mod'] += [murk_i]
                    statistics[site_id]['rh_mod'] += [rh_mod_i]

                    # all extra stats slots
                    statistics[site_id]['back_diff_log'] += [np.log10(mod_back_set_i) - np.log10(obs_back_i)]
                    statistics[site_id]['back_diff_norm'] += [mod_back_set_i - obs_back_i]
                    statistics[site_id]['ratio'] += [mod_back_set_i / obs_back_i]


            # Plot once statistics are done...

            # Line plots of each
            plot_multiple_back_point_diffs(day, FOcon.aerFO_version, savedir, site_id,
                statistics_modN_modr, statistics_modN_pm10r, statistics_obsN_pm10r, statistics_obsN_modr)

            plot_multiple_back_point_ratio(day, FOcon.aerFO_version, savedir, site_id,
                statistics_modN_modr, statistics_modN_pm10r, statistics_obsN_pm10r, statistics_obsN_modr)

            # line plot radius [m]
            plot_r_mds(day, FOcon.aerFO_version, savedir, site_id,
                       mod_data_modN_modr, mod_data_modN_pm10r, mod_data_obsN_pm10r,
                       mod_data_obsN_modr)

            # line plot NumConc [m-3]
            plot_Ns(day, FOcon.aerFO_version, savedir, site_id,
                       mod_data_modN_modr, mod_data_modN_pm10r, mod_data_obsN_pm10r,
                       mod_data_obsN_modr)

    return

if __name__ == '__main__':
    main()























print 'END PROGRAM'
