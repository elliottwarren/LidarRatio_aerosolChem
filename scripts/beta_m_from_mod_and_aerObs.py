"""
Run the aerFO with UKV data and aerosol observations. Create the control and alternative aerFO runs.
Save the output into a np file and do some plotting afterwards.

Created by Elliott 09 May 2018
"""

import sys
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/scripts')
import pm_RH_vs_mod_obs_backscatter as mvo

import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
from dateutil import tz
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from copy import deepcopy
import colorsys

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

def create_statistics():

    """
    Create the statistics dicionary
    :return:
    """

    # define site based lists to store the correlation results in
    statistics = {'r': [], 'p': [],
                       'diff': [],
                       'aer_diff': [],
                       'aer_diff_normalsed_aer_obs': [],
                       'aer_mod': [],
                       'aer_obs': [],
                       'rh_diff': [],
                       'rh_mod': [],
                       'rh_obs': [],
                       'back_diff_log': [],
                       'back_diff_norm': [],
                       'back_diff_norm_normalised_back_obs': [],
                       'abs_back_diff_log': [],
                       'abs_back_diff_norm': [],
                       'back_obs': [],
                       'back_mod': [],
                       'RMSE': [],
                       'MBE': [],
                       'hr': [],
                       'datetime': []}


    return statistics

def read_pm10_obs(mod_time, matchModSample=True):

    pm10_obs = {}
    from_zone = tz.gettz('GMT')
    to_zone = tz.gettz('UTC')

    dir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/DEFRA/'
    aer_fname = dir + 'PM10_Hr_NK_DEFRA_AURN_01012016-30052018.csv'

    raw_aer = np.genfromtxt(aer_fname, delimiter=',', skip_header=5, dtype="|S20")

    # sort out times as they are in two columns
    rawtime = [i[0] + ' ' + i[1].replace('24:00:00', '00:00:00') for i in raw_aer]
    time_endHr = np.array([dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') for i in rawtime])
    # convert from GMT to UTC and remove the timezone afterwards
    time_endHr = np.array([i.replace(tzinfo=from_zone) for i in time_endHr]) # label time as 'GMT'
    pro_time = np.array([i.astimezone(to_zone) for i in time_endHr]) # find time as 'UTC'
    pro_time = np.array([i.replace(tzinfo=None) for i in pro_time]) # remove 'UTC' timezone identifier

    # extract obs and time together as a dictionary entry for the site.
    data_obs = {'pm_10': np.array([np.nan if i == 'No data' else i for i in raw_aer[:, 2]], dtype=float),
                'time': pro_time}

    # match the time sample resolution of mod_data
    if matchModSample == True:

        # find nearest time in rh time
        # pull out ALL the nearest time idxs and differences
        t_idx = np.array([eu.nearest(data_obs['time'], t)[1] for t in mod_time])
        t_diff = np.array([eu.nearest(data_obs['time'], t)[2] for t in mod_time])

        # extract ALL nearest hours data, regardless of time difference
        pm10_obs['pm_10'] = data_obs['pm_10'][t_idx]
        pm10_obs['time'] = [data_obs['time'][i] for i in t_idx]

        # overwrite t_idx locations where t_diff is too high with nans
        # only keep t_idx values where the difference is below 5 minutes
        bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])
        pm10_obs['pm_10'][bad] = np.nan

    else:
        pm10_obs = data_obs

    return pm10_obs

def build_statistics(statistics, mod_data, bsc_obs, pm10_obs, mod_height_idx, ceil_height_idx):
    """
    build up the evaluation statistics for this day
    :param statistics:
    :param mod_data:
    :param mod_height_idx:
    :param ceil_height_idx:
    :return:
    """

    # build statistics
    statistics['hr'] += [i.hour for i in mod_data['time']]


    obs_back_i = bsc_obs['backscatter'][:, ceil_height_idx]
    mod_back_i = mod_data['backscatter'][:, mod_height_idx]

    murk_i = mod_data['aerosol_concentration_dry_air'][:, 0]  # 0th height = 5 m [micro g m-3]
    pm10_i = pm10_obs['pm_10']
    rh_mod = mod_data['RH'][:, mod_height_idx] * 100.0  # convert from [fraction] to [%]

    statistics['back_obs'] += list(obs_back_i)
    statistics['back_mod'] += list(mod_back_i)

    # all extra stats slots
    statistics['back_diff_log'] += list(np.log10(mod_back_i) - np.log10(obs_back_i))
    statistics['back_diff_norm'] += list(mod_back_i - obs_back_i)
    statistics['abs_back_diff_norm'] += list(np.abs(mod_back_i - obs_back_i))
    statistics['back_diff_norm_normalised_back_obs'] += list((mod_back_i - obs_back_i) / obs_back_i)
    statistics['abs_back_diff_log'] += list(np.abs(np.log10(mod_back_i) - np.log10(obs_back_i)))

    statistics['aer_obs'] += list(pm10_i)
    statistics['aer_mod'] += list(murk_i)
    statistics['rh_mod'] += list(rh_mod)

    statistics['aer_diff'] += list(murk_i - pm10_i)
    statistics['aer_diff_normalsed_aer_obs'] += list((murk_i - pm10_i) / pm10_i)

    return statistics

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # which modelled data to read in
    model_type = 'UKV'
    res = FOcon.model_resolution[model_type]

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/'
    savedir = maindir + 'figures/model_eval/scatter/'
    npysavedir = datadir + 'npy/model_eval/'

    # data
    ceilMetaDatadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'
    ceilDatadir = datadir + 'L1/'
    modDatadir = datadir + model_type + '/'
    aerDatadir = datadir + 'LAQN/'

    # # pm10
    site = 'NK'
    ceil_id = 'CL31-D'
    ceil = ceil_id + '_' + site
    site_bsc = {ceil: FOcon.site_bsc[ceil]}

    # wavelength (int) [nm]
    ceil_lambda_nm = 905

    site_bsc_colours = FOcon.site_bsc_colours

    # day list
    # clear sky days (5 Feb 2015 - 31 Dec 2016)
    # daystrList = ['20150414', '20150415', '20150421', '20150611', '20160504']

    # NK CTRL - all clear sky days for paper 2
    # CL31-D days '20160504', '20160505', '20160506'
    #daystrList = ['20160823','20160911','20161102','20161125','20161129','20161130','20161204','20161205','20161227','20161229',
    #    '20170105','20170117','20170118','20170119','20170120','20170121','20170122','20170325','20170330','20170408','20170429',
    #    '20170522','20170524','20170526','20170601','20170614','20170615','20170619','20170620','20170626','20170713','20170717',
    #    '20170813','20170827','20170828','20170902']

    # NK N and r accum (no murk aerosol required) (should be 2014 - 2015 inclusive,
    #   currently missing 2014 due to its-tier2 permission being bad)
    # daystrList = ['20150611', '20150521', '20150520', '20150515', '20150405'] # taken from paper 1 data dir
    daystrList = ['20150307','20150310','20150414','20150415','20150420','20150421','20150604','20150611','20150629',
                  '20150710','20150802','20151025']

    # daystrList = ['20140415']

    days_iterate = eu.dateList_to_datetime(daystrList)

    # ceilometer gate number to use for backscatter comparison
    # 1 - noisy
    # 2 - more stable
    # see Kotthaus et al (2016) for more.
    ceil_gate_num = 2

    # create statistics dictionary, ready to be filled
    statistics = create_statistics()

    # type of model evaluation and save string
    eval_type = 'obs_N_obs_r_accum'
    # eval_type = 'CTRL'
    aer_run = 'run2'
    savestr = site + '_' + eval_type + '_' +aer_run + '.npy'

    # argumets for aerFO based on run type
    if aer_run == 'run1': # CTRL
        kwargs = {}
    elif aer_run == 'run2': # N and r (accum range)
        kwargs = {'obs_N': True, 'obs_r': True}

    # savename for plotted files
    savename = 'delta_beta_delta_m_'+eval_type

    # keep some extra vars to look at?
    S_out =[]
    RH_out=[]
    f_RH_out=[]

    # version number
    version=1.1

    # ==============================================================================
    # Read data
    # ==============================================================================

    # Read Ceilometer metadata

    # ceilometer list to use
    ceilsitefile = 'CeilsCSVfull.csv'
    ceil_metadata = FO.read_ceil_metadata(ceilMetaDatadir, ceilsitefile)

    ceil_data_i = {site: ceil_metadata[site]}

    for day in days_iterate:

        # a dict just for today's stats
        day_statistics = create_statistics()

        print 'day = ' + day.strftime('%Y-%m-%d')

        # Run one aerFO type against BSC
        # Create stats then save them somewhere
        # plot all the aerFO vs BSC run results once they have all be processed and saved

        # control run - needs:
        # new f(RH)
        # new Q_ext,dry # boost the res at lower values
        # new S
        # Control run - use observed RH instead of modelled
        # mod_data = FO.mod_site_extract_calc(day, ceil_data_i, modDatadir, model_type, res, 905, version=version,
        #                 allvars=True, fullForecast=False, obs_RH=True, **kwargs)

        # run 2 - obs N and r (accum) (and obs RH so its a fair comparison with the control run)
        mod_data = {'CL31-D_NK':FO.forward_operator_from_obs(day, ceil_lambda_nm, version=1.1,
                        allvars=True, fullForecast=False, obs_N=True, obs_r=True, obs_RH=True)}

        # Read in BSC data
        bsc_obs = FO.read_ceil_obs(day, site_bsc, ceilDatadir, mod_data, calib=True, version=version)

        # if obs data for this day exists...
        if ceil not in bsc_obs:
            print 'ceil is missing!'
        else:

            mod_data = mod_data[site]
            bsc_obs = bsc_obs[ceil]

            # read in pm10 obs
            pm10_obs = read_pm10_obs(mod_data['time'], matchModSample=True)

            S_out += list(mod_data['S'])
            RH_out = list(mod_data['RH'])
            f_RH_out = list(mod_data['f_RH'])

            # get the ceilometer and model height index for the define ceilometer range gate
            ceil_height_idx, mod_height_idx =\
                 mvo.get_nearest_ceil_mod_height_idx(mod_data['level_height'], bsc_obs['height'], ceil_gate_num)

            # calculate statistics and add them to the collection
            statistics = build_statistics(statistics, mod_data, bsc_obs, pm10_obs, mod_height_idx, ceil_height_idx)
            day_statistics = build_statistics(day_statistics, mod_data, bsc_obs, pm10_obs, mod_height_idx, ceil_height_idx)

        # # plot today's stats
        # mvo.plot_back_point_diff(day_statistics,
        #                          savedir, model_type, ceil_gate_num, ceil,
        #                          savename=savename+'_' + day.strftime('%Y-%m-%d')+'.png')

    # ---------------------------

    # save the statistics
    save_dict = {'statistics': statistics, 'cases': days_iterate, 'site': site, 'ceil': ceil,
                 'ceil_lambda_nm': ceil_lambda_nm}
    np.save(npysavedir + savestr, save_dict)
    print 'data saved!: ' + npysavedir + savestr

    mvo.plot_back_point_diff(statistics,
                         savedir, model_type, ceil_gate_num, ceil, extra='_norm_back_run2')


    # scatter plot beta_o vs pm10; beta_m vs murk
    low_aer_bool = np.array(statistics['aer_mod']) <= 100
    vlow_aer_bool = np.array(statistics['aer_mod']) <= 50

    fig = plt.figure()
    ax = plt.gca()

    plt.scatter(statistics['aer_obs'], statistics['back_obs'], s=3, color='blue')
    plt.scatter(statistics['aer_mod'], statistics['back_mod'], s=3, color='red')
    #plt.scatter(np.array(statistics['aer_mod'])[low_aer_bool], np.array(statistics['back_mod'])[low_aer_bool],
    #            s=3, color='green', label='murk<=100')

    x = np.array(statistics['aer_obs'])
    y = np.array(statistics['back_obs'])
    idx = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[idx], y[idx], 1)
    r, p = pearsonr(x[idx], y[idx])
    print 'obs'
    print r
    print p
    print '\n'
    x_lim = ax.get_xlim()
    ax.plot(np.array(x_lim), m * np.array(x_lim) + b, ls='-', color='blue', label='obs')

    x = np.array(statistics['aer_mod'])
    y = np.array(statistics['back_mod'])
    idx = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[idx], y[idx], 1)
    r, p = pearsonr(x[idx], y[idx])
    print 'all mod'
    print r
    print p
    print '\n'
    x_lim = ax.get_xlim()
    ax.plot(np.array(x_lim), m * np.array(x_lim) + b, ls='-', color='red', label='all murk')

    x = np.array(statistics['aer_mod'])[low_aer_bool]
    y = np.array(statistics['back_mod'])[low_aer_bool]
    idx = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[idx], y[idx], 1)
    r, p = pearsonr(x[idx], y[idx])
    print 'mod <= 100'
    print r
    print p
    print '\n'
    x_lim = ax.get_xlim()
    ax.plot(np.array(x_lim), m * np.array(x_lim) + b, ls='-', color='green', label='murk<=100')

    x = np.array(statistics['aer_mod'])[vlow_aer_bool]
    y = np.array(statistics['back_mod'])[vlow_aer_bool]
    idx = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[idx], y[idx], 1)
    r, p = pearsonr(x[idx], y[idx])
    print 'mod <= 50'
    print r
    print p
    print '\n'
    x_lim = ax.get_xlim()
    ax.plot(np.array(x_lim), m * np.array(x_lim) + b, ls='-', color='purple', label='murk<=50')

    plt.axis('tight')
    plt.ylim([1.0e-8, 5e-5])
    plt.ylabel('beta [sr-1 m-1]')
    plt.xlabel('aerosol concentration [microgram m-3]')
    plt.legend()
    plt.tight_layout()














    print 'END PROGRAM'