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
from scipy.stats import pearsonr

import numpy as np
import datetime as dt
import pandas
from dateutil import tz


import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon

def read_pm10_obs(aer_fname):

    pm10_obs = {}
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
    years = ['2014']
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
    for year in ['2014', '2015']:
        N_r_fname = npydir+'accum_Ntot_Dv_NK_APS_SMPS_'+year+'.npy'
        y_data += [(np.load(N_r_fname).flat[0])]
    # merge N and r yearly data together
    N_r_obs_raw = {}
    for key, item in y_data[0].iteritems():
        N_r_obs_raw[key] = np.append(y_data[0][key], y_data[1][key])

    # time match pm10 and N_r_obs
    N_r_obs, pm10_obs = eu.time_match_datasets(N_r_obs_raw, pm10_obs_raw)

    # create aerFO equation curves to overplot onto the scatter plots
    q_aer_ug_kg = np.arange(0, 90)
    q_aer_kg_kg = q_aer_ug_kg * 1e-9
    m0 = FOcon.m0_aer_urban
    p = FOcon.p_aer
    r0 = FOcon.r0_urban

    N_aer = FOcon.N0_aer_urban * np.power((q_aer_kg_kg / m0), 1.0 - (3.0 * p))
    r_md = r0 * np.power((q_aer_kg_kg / m0), p)

    plt.figure()
    plt.scatter(pm10_obs['pm_10'], N_r_obs['Dv_accum']/2 ,s=2)
    plt.plot(pm10_obs['pm_10'], r_md * 1e6, '--', colour='grey', alpha=0.7)
    plt.xlabel('pm10 [microgram m-3]')
    plt.ylabel('volume radius [nm]')
    plt.savefig(savedir + 'r_accum_vs_pm10_NK_2014-2015.png')

    plt.figure()
    plt.scatter(pm10_obs['pm_10'], N_r_obs['Ntot_accum'] / 2, s=2)
    plt.plot(pm10_obs['pm_10'], N_aer *1e6, '--', colour='grey', alpha=0.7)
    plt.xlabel('pm10 [microgram m-3]')
    plt.ylabel('Ntot (accum) [cm-3]')
    plt.savefig(savedir + 'N_accum_vs_pm10_NK_2014-2015.png')
