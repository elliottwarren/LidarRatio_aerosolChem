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

import numpy as np
import datetime as dt
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from copy import deepcopy
import colorsys

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

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

def plot_back_point_diff(stats_site, savedir, model_type, ceil_gate_num, ceil, sampleSize, corr, var_type, c_type='hr', extra=''):

    """
    Plot the rh or aer difference vs backscatter diff
    :return:
    """

    # variable plotting against backscatter point diff
    if var_type == 'aerosol':
        var_diff = stats_site['aer_diff']
    elif var_type == 'RH':
        var_diff = stats_site['rh_diff']

    # backscatter point difference
    back_point_diff = stats_site['back_diff_norm']
    # back_point_diff = stats_site['back_diff_log']

    fig = plt.figure(figsize=(6, 3.5))
    ax = plt.subplot2grid((1, 1), (0, 0))

    # variable specific labels and names
    if var_type == 'RH':
        xlab = r'$Difference \/\mathrm{(RH_{ukv} - RH_{obs})}$'

    elif var_type == 'aerosol':
        xlab = r'$Difference \/\mathrm{(m_{MURK} - PM_{10})}$'

    # define the colormap
    cmap, norm = discrete_colour_map(40, 100, 13)

    # plot data
    scat = plt.scatter(var_diff, back_point_diff, c=stats_site[c_type], s=6, vmin=40.0, vmax=100.0, cmap=cmap, norm=norm)

    # add 0 lines
    ax.axhline(linestyle='--', color='grey', alpha=0.5)
    ax.axvline(linestyle='--', color='grey', alpha=0.5)

    # ax.set_ylim([-5e-06, 5e-06])

    # add colourbar on the side
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scat, cax=cax, norm=norm)
    cbar.set_label(r'$RH\/[\%]$', labelpad=-38, y=1.075, rotation=0)


    ax.set_xlabel(xlab)
    ax.set_ylim([-1e-5, 1e-5])
    # ax.set_ylabel(r'$Difference \/\mathrm{(log_{10}(\beta_m) - log_{10}(\beta_o))}$')
    ax.set_ylabel(r'$Difference \/\mathrm{(\beta_m - \beta_o)}$')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

    # Fake a ScalarMappable so I can display a colormap
    # cmap, norm = mcolors.from_levels_and_colors(range(24 + 1), rgb)
    # sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # fig.colorbar(sm)
    # plt.colorbar()

    fig.suptitle(ceil + '; n = ' + str(sampleSize) + '; r = ' + '{:1.2f}'.format(corr['r']) +
                 '; p = ' + '%1.2f' % corr['p'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(savedir + 'point_diff/' +
                model_type + '_' + var_type + '_diff_' + ceil + '_clearDays_gate' + str(ceil_gate_num) + '_c' + c_type +
                '_' + extra + '.png')  # filename

    plt.close(fig)

    return

def plot_back_point_diff_6hr(var_diff, back_point_diff, savedir, model_type, ceil_gate_num, ceil, sampleSize, corr, var_type):

    """
    Plot the rh or aer difference vs backscatter diff
    :return:
    """

    rgb = colour_range(24)

    fig = plt.figure(figsize=(6, 3.5))
    ax = plt.subplot2grid((1, 1), (0, 0))

    # variable specific labels and names
    if var_type == 'RH':
        xlab = r'$Difference \/\mathrm{(RH_{ukv} - RH_{obs})}$'

    elif var_type == 'aerosol':
        xlab = r'$Difference \/\mathrm{(m_{MURK} - PM_{10})}$'

    for t in [0,6,12,18]:

        t_range = np.arange(t, t+6)

        for t_i in t_range:

            hr = str(t_i)

            # hr_colour = rgb[t]

            plt.scatter(var_diff[hr], back_point_diff[hr], s=4)

        ax.set_xlabel(xlab)
        ax.set_ylabel(r'$Difference \/\mathrm{(log_{10}(\beta_m) - log_{10}(\beta_o))}$')

        # # Fake a ScalarMappable so I can display a colormap
        # cmap, norm = mcolors.from_levels_and_colors(range(6 + 1), rgb)
        # sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # fig.colorbar(sm)

        fig.suptitle(ceil + '; n = ' + str(sampleSize) + '; r = ' + '{:1.2f}'.format(corr['r']) +
                     '; p = ' + '%1.2f' % corr['p'])
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(savedir + 'point_diff/' +
                    model_type + '_' + var_type + '_diff_' + ceil + '_clearDays_gate' + str(ceil_gate_num) + 't'+str(t)+'.png')  # filename

    return fig

def colour_range(num_colours=24.0):

    """Makes a simple range of colours"""

    for i in range(num_colours):

        rgb = [colorsys.hsv_to_rgb(i / (num_colours*3.0), 1.0, 1.0) for i in range(num_colours)]

        # rgb = colorsys.hsv_to_rgb(i / 72.0, 1.0, 1.0)
        # print(i, [round(255 * x) for x in rgb])

        return rgb

def discrete_colour_map(lower_bound, upper_bound, spacing):

    """Create a discrete colour map"""

    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    # cmaplist[0] = (.5, .5, .5, 1.0)
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(lower_bound, upper_bound, spacing)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm

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
    daystrList = ['20160504']

    days_iterate = dateList_to_datetime(daystrList)


    # ceilometer gate number to use for backscatter comparison
    # 1 - noisy
    # 2 - more stable
    # see Kotthaus et al (2016) for more.
    ceil_gate_num = 2

    # where to pull rh height out.
    mod_rh_height_idx = 1

    # define statistics dictionary
    statistics_modN = {}
    statistics_obsN = {}
    sampleSize = 0 # add to this


    # ==============================================================================
    # Read data
    # ==============================================================================

    # Read Ceilometer metadata

    # ceilometer list to use
    ceilsitefile = 'CeilsCSVfull.csv'
    ceil_metadata = FO.read_ceil_metadata(ceilMetaDatadir, ceilsitefile)

    ceil_data_i = {site: ceil_metadata[site]}

    for day in days_iterate:

        print 'day = ' + day.strftime('%Y-%m-%d')

        # Read UKV forecast and automatically run the FO

        # extract MURK aerosol and calculate RH for each of the sites in the ceil metadata
        # reads all london model data, extracts site data, stores in single dictionary
        # 1
        mod_data_modN_modr = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=0.2,
                        allvars=True)
        # 2
        mod_data_modN_pm10r = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=0.2,
                        allvars=True, obsr_site='NK')

        #
        mod_data_obsN_pm10r = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 910, version=0.2,
                        allvars=True, obsN_site='NK', obsr_site='NK')

        # Read ceilometer backscatter

        # will only read in data is the site is there!
        # ToDo Remove the time sampling part and put it into its own function further down.
        bsc_obs = FO.read_ceil_obs(day, site_bsc, ceilDatadir, mod_data_modN, calib=True)

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
            statistics_modN_modr = create_stats_entry(site_id, statistics_modN)
            statistics_modN_pm10r = create_stats_entry(site_id, statistics_modN)
            statistics_obsN_pm10r = create_stats_entry(site_id, statistics_obsN)

            # for the different combos of mod_data
            for statistics, mod_data_set in zip((statistics_modN_modr, statistics_modN_pm10r, statistics_obsN_pm10r),
                                    (mod_data_modN_modr,   mod_data_modN_pm10r,   mod_data_obsN_pm10r)):



                # for each hour possible in the day
                for t in np.arange(0, 24):

                    hr = str(t)

                    # add the hour to statstics dict
                    statistics[site_id]['hr'] += [t]
                    statistics[site_id]['datetime'] += [mod_data_set[site]['time'][t]]

                    statistics_obsN[site_id]['hr'] += [t]
                    statistics_obsN[site_id]['datetime'] += [mod_data_set[site]['time'][t]]

                    # extract out all unique pairs below the upper height limit
                    # these are time and height matched now
                    #obs_x = bsc_site_obs['backscatter'][t, obs_hc_unique_pairs]
                    #mod_y = mod_data[site_id]['backscatter'][t, mod_hc_unique_pairs]

                    # # extract pairs of values used in statistics
                    # if pm10_stats == True:
                    #     pm10_i = pm10['PM10_'+site]['pm_10'][t]
                    #     # murk_i = mod_data[site]['aerosol_for_visibility'][t, 0] # 0th height = 5 m
                    #
                    # if rh_stats == True:
                    #     rh_obs_i = rh_obs[rh_instrument]['RH'][t]

                    # get murk_i and rh_i regardless if pm10_stats and rh_stats is True or not
                    murk_i = mod_data_modN[site]['aerosol_concentration_dry_air'][t, 0]  # 0th height = 5 m
                    rh_mod_i = mod_data_modN[site]['RH'][t, mod_rh_height_idx] * 100.0  # convert from [fraction] to [%]

                    obs_back_i = bsc_site_obs['backscatter'][t, ceil_height_idx]
                    mod_back_modN_i = mod_data_modN[site]['backscatter'][t, mod_height_idx]
                    mod_back_obsN_i = mod_data_obsN[site]['backscatter'][t, mod_height_idx]

                    statistics_modN[site_id]['back_obs'] += [obs_back_i]
                    statistics_modN[site_id]['back_mod'] += [mod_back_modN_i]
                    statistics_obsN[site_id]['back_obs'] += [obs_back_i]
                    statistics_obsN[site_id]['back_mod'] += [mod_back_obsN_i]

                    statistics_modN[site_id]['aer_mod'] += [murk_i]
                    statistics_modN[site_id]['rh_mod'] += [rh_mod_i]
                    statistics_obsN[site_id]['aer_mod'] += [murk_i]
                    statistics_obsN[site_id]['rh_mod'] += [rh_mod_i]

                    # STATISTICS
                    # ---------------

                    # # length of aer_diff[hr] and ['back_point_diff'] hour should and MUST be the same length
                    # # such that their idx positions line up
                    # if pm10_stats == True:
                    #     statistics[site_id]['aer_diff'] += [murk_i - pm10_i]
                    #     statistics[site_id]['aer_obs'] += [pm10_i]
                    #
                    #     # if the difference pairs do not posses an NaN (and will therefore be plotted), add 1 to sample size
                    #     if ~np.isnan(murk_i - pm10_i) & ~np.isnan(np.log10(mod_back_i) - np.log10(obs_back_i)):
                    #         sampleSize += 1
                    #
                    # if rh_stats == True:
                    #     statistics[site_id]['rh_diff'] += [rh_mod_i - rh_obs_i]
                    #     statistics[site_id]['rh_obs'] += [rh_obs_i]
                    #
                    #     # if the difference pairs do not posses an NaN (and will therefore be plotted), add 1 to sample size
                    #     if ~np.isnan(rh_mod_i - rh_obs_i) & ~np.isnan(np.log10(mod_back_i) - np.log10(obs_back_i)):
                    #         sampleSize += 1


                    # all extra stats slots
                    statistics_modN[site_id]['back_diff_log'] += [np.log10(mod_back_modN_i) - np.log10(obs_back_i)]
                    statistics_modN[site_id]['back_diff_norm'] += [mod_back_modN_i - obs_back_i]

                    statistics_obsN[site_id]['back_diff_log'] += [np.log10(mod_back_obsN_i) - np.log10(obs_back_i)]
                    statistics_obsN[site_id]['back_diff_norm'] += [mod_back_obsN_i - obs_back_i]



    # # quick remove RH > 80
    # val = 80.0
    # idx_lt = np.where(np.array(statistics_modN[site_id]['rh_mod']) <= val)
    #
    # for key in statistics[site_id].iterkeys():
    #     if len(statistics[site_id][key]) != 0:
    #         for i in idx_lt[0]:
    #             statistics[site_id][key][i] = np.nan
    # sampleSize -= len(idx_lt[0])


    # # do correlation
    # if rh_stats == True:
    #     corr = {}
    #     corr['r'], corr['p'] = spearmanr(statistics[site_id]['rh_diff'], statistics[site_id]['back_diff_norm'], nan_policy='omit')
    #     a1 = np.array(statistics[site_id]['aer_mod']) # extract the rh difference dataset from statistics
    #
    # if pm10_stats == True:
    #     corr = {}
    #     corr['r'], corr['p'] = spearmanr(statistics[site_id]['aer_diff'], statistics[site_id]['back_diff_norm'], nan_policy='omit')
    #     a1 = np.array(statistics[site_id]['aer_diff']) # extract the aerosol difference dataset from statistics
    # plot!


    # b1 = np.array(statistics[site_id]['back_diff_norm'])
    #
    # a1_idx = np.where(np.isnan(a1))
    # b1_idx = np.where(np.isnan(b1))
    #
    # a1[b1_idx] = np.nan
    # b1[a1_idx] = np.nan
    #
    # an_idx = ~np.isnan(a1)
    #
    # pearsonr(a1[an_idx],b1[an_idx])

    # # pass in statistics with site id!
    # if pm10_stats == True:
    #     plot_back_point_diff(statistics[site_id],
    #                                savedir, model_type, ceil_gate_num, ceil, sampleSize, corr, var_type='aerosol',
    #                                c_type='rh_mod', extra = '_aodChange')
    #
    # if rh_stats == True:
    #     plot_back_point_diff(statistics[site_id],
    #                                savedir, model_type, ceil_gate_num, ceil, sampleSize, corr, var_type='RH',
    #                                c_type='rh_mod')

    plt.subplots(1,1,figsize=(6,4))
    plt.plot_date(statistics_modN[site_id]['datetime'], statistics_modN[site_id]['back_diff_norm'], label='modN', fmt='--')
    plt.plot_date(statistics_obsN[site_id]['datetime'], statistics_obsN[site_id]['back_diff_norm'], label='obsN', fmt='--')
    plt.xlabel('time')
    plt.ylabel('beta')
    plt.legend()
    plt.suptitle('modN_0 = 8000 cm-3')
    plt.tight_layout(pad=2)
    plt.savefig(savedir + 'modN_and_obsN_8000.png')

    plt.close('all')

    return

if __name__ == '__main__':
    main()























print 'END PROGRAM'
