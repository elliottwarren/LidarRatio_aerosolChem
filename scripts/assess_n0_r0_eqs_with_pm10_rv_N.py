"""
Assess the early equations in the aerFO that estiamte N and r from N0 and r0 respectively. Need to understand the
overestimation in /beta_m by the aerFO...
"""

# read in pm10
# read in rv and N
# calculate aerFO rv and N from the equations (dashed lines)
# plot all together

import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
from dateutil import tz

import ellUtils as eu
from forward_operator import FOconstants as FOcon

def read_pm10_obs(aer_fname):

    from_zone = tz.gettz('GMT')
    to_zone = tz.gettz('UTC')

    raw_aer = np.genfromtxt(aer_fname, delimiter=',', skip_header=5, dtype="|S20")

    # sort out times as they are in two columns
    rawtime = [i[0] + ' ' + i[1].replace('24:00:00', '00:00:00') for i in raw_aer]
    time_endHr = np.array([dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') for i in rawtime])
    # convert from GMT to UTC and remove the timezone afterwards
    time_endHr = np.array([i.replace(tzinfo=from_zone) for i in time_endHr]) # label time as 'GMT'
    pro_time = np.array([i.astimezone(to_zone) for i in time_endHr]) # find time as 'UTC'
    pro_time = np.array([i.replace(tzinfo=None) for i in pro_time]) # remove 'UTC' timezone identifier

    # extract obs and time together as a dictionary entry for the site.
    pm10_obs = {'pm_10': np.array([np.nan if i == 'No data' else i for i in raw_aer[:, 2]], dtype=float),
                'time': pro_time}

    return pm10_obs

def read_wxt_obs(years, time):

    """
    Read in RH observations from KSSW, time match them to the model data, and extend them in height to match the
    dimensions of model RH
    :param years:
    :param time:
    :return wxt_obs:
    """

    met_vars = ['RH', 'Tair', 'press']
    vars = met_vars + ['time']
    filepath = ['C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/L1/' + \
               'Davis_BGH_' + str(i) + '_15min.nc' for i in years]
    wxt_obs_raw = eu.netCDF_read(filepath, vars=vars)


    # set up array to be filled
    wxt_obs = {}
    for met_var in met_vars:
        wxt_obs[met_var] = np.empty(len(time))
        wxt_obs[met_var][:] = np.nan
    wxt_obs['time'] = time

    # find data region and create an average if appropriate
    print_step = range(1000,20000, 1000)
    for t, time_t in enumerate(time):

        if t in print_step:
            print 't ='+str(t)

        # time t-1 (start of original time period, as all data is relevent for time ENDING at time_t)
        tm1 = t-1
        time_tm1 = time_t - dt.timedelta(minutes=60)

        # # start of time period
        # idx_extent = 8000
        # s_idx = int(eu.binary_search(wxt_obs_raw['time'], time_tm1, lo=max(0, tm1 - idx_extent),
        #                           hi=min(tm1 + idx_extent, len(wxt_obs_raw['time']))))
        # # end of time period
        # e_idx = int(eu.binary_search(wxt_obs_raw['time'], time_t, lo=max(0, t - idx_extent),
        #                           hi=min(t + idx_extent, len(wxt_obs_raw['time']))))

        s_idx = int(eu.binary_search(wxt_obs_raw['time'], time_tm1))
        # end of time period
        e_idx = int(eu.binary_search(wxt_obs_raw['time'], time_t))

        # if the time_range time and data['time'] found in this iteration are within an acceptable range (15 mins)
        tm1_diff = time_tm1 - wxt_obs_raw['time'][s_idx]
        t_diff = time_t - wxt_obs_raw['time'][e_idx]


        # _, s_idx, tm1_diff = eu.nearest(wxt_obs_raw['time'], time_tm1)
        # _, e_idx, t_diff = eu.nearest(wxt_obs_raw['time'], time_t)


        if (tm1_diff.total_seconds() <= 15 * 60) & (t_diff.total_seconds() <= 15 * 60):
            for met_var in met_vars:
                wxt_obs[met_var][t] = np.nanmean(wxt_obs_raw[met_var][s_idx:e_idx+1])


    # create RH_frac using RH data
    wxt_obs['RH_frac'] = wxt_obs['RH'] / 100.0

    # calculate extra variables
    e_s_hpa = 6.112 * (np.exp((17.67 * wxt_obs['Tair']) / (wxt_obs['Tair'] + 243.5)))  # [hPa] # sat. v. pressure
    e_s = e_s_hpa * 100.0  # [Pa] # sat. v. pressure
    wxt_obs['e'] = wxt_obs['RH_frac'] * e_s  # [Pa] # v. pressure
    wxt_obs['r_v'] = wxt_obs['e'] / (1.61 * ((wxt_obs['press']*100.0) - wxt_obs['e'])) # water_vapour mixing ratio [kg kg-1]
    wxt_obs['q'] =  wxt_obs['e'] / ((1.61 * ((wxt_obs['press']*100.0) - wxt_obs['e'])) + wxt_obs['e']) # specific humidity [kg kg-1]
    wxt_obs['Tv'] = (1 + (0.61 * wxt_obs['q'])) * (wxt_obs['Tair'] + 273.15) # virtual temp [K]
    wxt_obs['air_density'] = (wxt_obs['press']*100.0) / (286.9 * wxt_obs['Tv'])# [kg m-3]

    return wxt_obs

if __name__ == '__main__':

    # -----------------------------
    # Setup
    # -----------------------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'
    pm10dir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/DEFRA/'
    savedir = maindir + 'figures/diags/'

    # data
    npydir = maindir + 'data/npy/number_distribution/'

    # year of data
    # for NK - only got 2014 and 2015 with APS data
    years = ['2014', '2015']
    year = years[0]
    # years = [str(i) for i in range(2014, 2017)]


    # -----------------------
    # Read
    # -----------------------

    # read in pm10 data
    aer_fname = pm10dir + 'PM10_Hr_NK_DEFRA_AURN_01012014-31122015.csv'
    pm10_obs_raw = read_pm10_obs(aer_fname)

    # read in N and r
    y_data = []
    for year in years:
        N_r_fname = npydir+'NK_APS_SMPS_Ntot_Dv_fine_acc_coarse_'+year+'_80-700nm.npy'
        y_data += [(np.load(N_r_fname).flat[0])]
    # merge N and r yearly data together
    N_r_obs_raw = {}
    for key, item in y_data[0].iteritems():
        N_r_obs_raw[key] = np.append(y_data[0][key], y_data[1][key])

    # time match pm10, N_r_obs and wxt data
    N_r_obs, pm10_obs = eu.time_match_datasets(N_r_obs_raw, pm10_obs_raw)

    # read in wxt obs and match it to the already time matched N and pm10 data
    wxt_obs = read_wxt_obs(years, pm10_obs['time'])

    # convert pm10 obs from [microgram m-3] to [microgram kg-1_air]
    pm10_obs['pm10_mass_mixing_ratio'] = pm10_obs['pm_10'] / wxt_obs['air_density']

    # create aerFO equation curves to overplot onto the scatter plots
    q_aer_ug_kg = np.arange(0, 90)
    q_aer_kg_kg = q_aer_ug_kg * 1e-9
    m0 = FOcon.m0_aer_urban
    p = FOcon.p_aer
    r0 = FOcon.r0_urban

    N_aer = FOcon.N0_aer_urban * np.power((q_aer_kg_kg / m0), 1.0 - (3.0 * p))
    r_md = r0 * np.power((q_aer_kg_kg / m0), p)

    plt.figure()
    plt.scatter(pm10_obs['pm10_mass_mixing_ratio'], N_r_obs['Dv_accum']/2 ,s=2)
    plt.plot(q_aer_ug_kg, r_md * 1e9, '--', color='grey', alpha=0.7)
    plt.xlabel('pm10 [microgram kg-1]')
    plt.ylabel('volume radius [nm]')
    plt.savefig(savedir + 'r_accum_vs_pm10_NK_2014-2015.png')

    plt.figure()
    plt.scatter(pm10_obs['pm10_mass_mixing_ratio'], N_r_obs['Ntot_accum'] *1e6 / 2, s=2)
    plt.plot(q_aer_ug_kg, N_aer, '--', color='grey', alpha=0.7)
    plt.xlabel('pm10 [microgram kg-1]')
    plt.ylabel('Ntot (accum) [cm-3]')
    plt.savefig(savedir + 'N_accum_vs_pm10_NK_2014-2015.png')
