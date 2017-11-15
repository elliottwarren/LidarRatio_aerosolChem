"""
Read in observed N(D) data and calculate N and r from them.

Created by Elliott Mon 13 Nov 2017

eqns found in: Seinfeld and Pandis (2007) Atmospheric chemistry and physics
    from air pollution to climate change

D = mid of bin diameter
dD = bin width (bin_max - bin_min) # not logged
dlogD = log(bin_max - bin_min) # bin width in logD terms
dNdD = Number concentration in bin, in normal diameter space # an array equal to the number of bins
dNdlogD = same as above but in logD space. Remember to use the 2.303 conversion! (eqn 8.18)
dVdD = volume distribution by bin # not logged
dVdlogD = volume distribution in logD space, again can use the 2.303 to convert (eqn 8.20)
Dv = volume mean diameter
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


def read_headers(filepath, startswith, delimiter):
    """
    A very specific function for the N(D) data from the TSI APS at NK during ClearfLo
    The headers are a bit awkward so, this function finds and pulls them based on the first
    header being 'Date' and no other lines beginning with it.

    :param Nfilepath:
    :return: headers [list]

    Importantly headers is read in as a list, therefore the order of the headers is preserved
    """

    f = open(filepath, 'r')
    line = f.readline()
    while line.startswith(startswith) != True:
        line = f.readline()

    f.close()

    # remove '\n' off the end of [line]
    line = line[:-2]

    split = np.array(line.split(delimiter))  # split headers up and turn into numpy array
    # remove empty headers and '\n'
    idx = np.logical_or(split == '', split == '\n')  # bad headers
    headers = split[~idx]  # remove bad headers

    return headers

def read_dmps(filepath):

    """
    Read in the smaller radii dataset from ClearfLo (5 - 500 nm)
    :param filepath:
    :return: dNdlogD
    """

    dNdlogD_raw = np.genfromtxt(filepath, delimiter=' ', skip_header=74, dtype=float, names=True)
    # N_raw = np.genfromtxt(filepath, delimiter=',', skip_header=75, dtype=float, names=True)

    # work around with the headers
    # VERY specific use with the TSI APS from NK during clearfLo
    # header line starts with [startswith]
    startswith = 'Time 5.0125'
    delimiter = ' '
    headers = read_headers(filepath, startswith, delimiter)
    header_bins = np.array(headers[2:])

    # extract time
    # extract data as 2D numpy array
    # aps_time = np.array([np.nan if i[0] == 999999 else i[0] for i in dNdlogD_raw], dtype=float)

    # turn Nrawdata into a dictionary
    dNdlogD = {}
    for i, head in enumerate(headers):
        dNdlogD[head] = np.array([np.nan if j[i] == 999999 else j[i] for j in dNdlogD_raw], dtype=float)

    # turn dates into datetimes and store in N_data['time']
    doy = np.array([int(i) for i in dNdlogD['Time']])
    doyFrac = np.array([i - int(i) for i in dNdlogD['Time']])
    base = dt.datetime(2012, 01, 01)
    dNdlogD['time'] = np.array(
        [base - dt.timedelta(days=1) + dt.timedelta(days=doy_i) + (dt.timedelta(days=1 * doyFrac_i))
         for doy_i, doyFrac_i in zip(doy, doyFrac)])
    dNdlogD['rawtime'] = dNdlogD['Time']  # original raw time
    del dNdlogD['Time']

    return dNdlogD, header_bins

def calc_bin_parameters_dmps(dNdlogD):

    """
    Calculate bin parameters for the dmps data (5 - 500 nm range)
    :return: D, dD, logD, dlogD (bin mid, bin width, log bin mid, log bin width)
    """

    # estimate bin windths as only the centre diameters have been given
    # keep D_keys as a list of bins later on, to calculate n_v(log D)
    D_keys = dNdlogD.keys()
    del D_keys[D_keys.index('time')]  # remove time so we just get bin widths
    del D_keys[D_keys.index('rawtime')]  # remove time so we just get bin widths
    D_mid = deepcopy(D_keys)
    D_mid = np.array(D_mid, dtype=float)
    D = np.sort(D_mid)

    half_widths = (D[1:] - D[:-1]) / 2.0
    widths_max = D[:-1] + half_widths
    # use last half width to define an upper bound for last diameter channel (last D value)
    widths_max = np.append(widths_max, half_widths[-1] + D[-1])
    # use first half width to define an upper bound for first diameter channel (first D value)
    widths_min = np.append(D[0] - half_widths[0], widths_max[:-1])
    # final widths used (\Delta D)
    dD = widths_max - widths_min

    # calculate logD and dlogD
    logD = np.log10(D)
    dlogD = np.log10(widths_max) - np.log10(widths_min)

    return D, dD, logD, dlogD

def read_aps(filepath_aps):

    """
    Read in the apd data from clearflo. Very similar structure to read_dmps()
    :param filepath_aps:
    :return:
    """

    N_raw = np.genfromtxt(filepath_aps, delimiter='\t', skip_header=84, dtype=float, names=True)
    # N_raw = np.genfromtxt(filepath, delimiter=',', skip_header=75, dtype=float, names=True)

    # work around with the headers
    # VERY specific use with the TSI APS from NK during clearfLo
    # header line starts with [startswith]
    startswith = 'Date	Total'
    delimiter = '\t'
    headers = read_headers(filepath_aps, startswith, delimiter) # all headers
    header_bins = np.array(headers[2:])# just the bins

    # turn raw into a dictionary
    N = {}
    for i, head in enumerate(headers):
        N[head] = np.array([np.nan if j[i] == 9999 else j[i] for j in N_raw], dtype=float)

    # turn dates into datetimes and store in N_data['time']
    doy = np.array([int(i) for i in N['Date']])
    doyFrac = np.array([i - int(i) for i in N['Date']])
    base = dt.datetime(2012, 01, 01)
    N['time'] = np.array(
        [base - dt.timedelta(days=1) + dt.timedelta(days=doy_i) + (dt.timedelta(days=1 * doyFrac_i))
         for doy_i, doyFrac_i in zip(doy, doyFrac)])
    N['rawtime'] = N['Date']  # original raw time
    del N['Date']

    return N, header_bins

def calc_bin_parameters_aps(N):

    """
    Calculate bin parameters for the aps data
    :param N:
    :return:
    """

    D_keys = N.keys()
    del D_keys[D_keys.index('time')]  # remove time so we just get bin widths
    del D_keys[D_keys.index('rawtime')]
    del D_keys[D_keys.index('Total')]  # remove time so we just get bin widths
    D_min = deepcopy(D_keys)
    D_min = np.array(D_min, dtype=float)
    D_min = np.sort(D_min)

    # upper edge on last bin is 21.29 microns, therefore append it on the end
    D_max = np.append(D_min[1:], 21.29)

    # mid bin
    D = (D_max + D_min) / 2.0
    logD = np.log10(D)

    # bin widths
    dD = D_max - D_min
    dlogD = np.log10(D_max) - np.log10(D_min)

    return D, dD, logD, dlogD

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

    # pm10
    site = 'NK'
    ceil_id = 'CL31-D'
    # ceil = ceil_id + '_BSC_' + site
    ceil = ceil_id + '_' + site

    site_bsc = {ceil: FOcon.site_bsc[ceil]}
    # site_bsc = {ceil: FOcon.site_bsc[ceil], 'CL31-E_BSC_NK': 27.0 - 23.2}

    # day list
    # clear sky days (5 Feb 2015 - 31 Dec 2016)
    # daystrList = ['20160504']
    # daystrList = ['20150414', '20150415', '20150421', '20150611', '20160504']
    daystrList = ['20150415']

    days_iterate = eu.dateList_to_datetime(daystrList)

    # ceilometer gate number to use for backscatter comparison
    # 1 - noisy
    # 2 - more stable
    # see Kotthaus et al (2016) for more.
    ceil_gate_num = 2

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

        # def read_clearFlo_obsr_small():

        # read in number distribution from observed data ---------------------------

        # read in N data and extract relevent values
        # make N at all heights away from surface nan to be safe
        filepath = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/ClearfLo/' \
                   'man-dmps_n-kensington_20120114_r0.na'

        # read in the ClearfLo data
        dmps_dNdlogD, dmps_header_bins = read_dmps(filepath)

        # calculate bin centres (D_mid)

        dmps_D, dmps_dD, dmps_logD, dmps_dlogD = calc_bin_parameters_dmps(dmps_dNdlogD)

        # # check mid, widths start and ends are ok
        # for i in range(len(widths_end)):
        #     print str(widths_start[i]) + ' to ' + str(widths_end[i]) + ': mid = ' + str(D_mid[i]) + '; width = ' + str(widths[i])


        # -----------------------------------------------------------------------

        # Read in aps data (larger radii: 0.5 to 20 microns)
        filepath_aps = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/ClearfLo/' \
                       'man-aps_n-kensington_20120110_r0.na'

        # read in aps data (dictionary format)
        # header bins will be lower bound of the bin, not the bin mid! (header_bins != aps_D)
        aps_N, aps_header_bins = read_aps(filepath_aps)

        # calculate bin parameters
        # for the DMPS, the headers correspond to the min bin edge. Therefore, the header along is the bin max.
        aps_D, aps_dD, aps_logD, aps_dlogD = calc_bin_parameters_aps(aps_N)

        # -----------------------------------------------------------------------

        # calc dNdlogD from N, for aps data (coarse)
        aps_dNdlogD = {}
        for idx in range(len(aps_D)):

            # get bin name (start of bin for aps data annoyingly...), dlogD and D for each data column
            # convert bin ranges from microns to nm
            bin_i = aps_header_bins[idx]
            dlogD_i = aps_dlogD[idx] * 1e3
            D_i = aps_D[idx] * 1e3

            aps_dNdlogD[str(D_i)] = aps_N[bin_i]/ dlogD_i

        # -----------------------------------------------------------------------

        # merge the two dNdlogD datasets together...
        # set up so data resolution is 15 mins

        # time range - APS time res: 5 min, DMPS time res: ~12 min
        start_time = np.min([aps_N['time'][0], dmps_dNdlogD['time'][0]])
        end_time = np.max([aps_N['time'][-1], dmps_dNdlogD['time'][-1]])
        time_range = eu.date_range(start_time, end_time, 15, 'minutes')

        # binned shape = [time, bin]
        dNdlogD = {'time': time_range,
                   'binned': np.empty([len(time_range), len(dmps_D)+len(aps_D)])}
        dNdlogD['binned'][:] = np.nan

        # insert the dmps data first, take average of all data within the time period
        for t in range(len(time_range[:-1])):

            # find data for this time
            binary = np.logical_and(dmps_dNdlogD['time'] > time_range[t], dmps_dNdlogD['time'] < time_range[t + 1])

            for D_idx in range(len(dmps_D)): # smaller data fill the first columns

                # enumerate might be useful here...

                # bin str
                D_i_str = str(dmps_D[D_idx])

                # create mean of all data within the time period and store
                dNdlogD['binned'][t, D_idx] = np.nanmean(dmps_dNdlogD[D_i_str][binary])

            for D_idx in range(len(aps_D)): # smaller data fill the first columns

                # bin str
                D_i_str = str(dmps_D[D_idx])

                # idx in new dNlogD array (data will go after the dmps data)
                D_binned_idx = D_idx + len(dmps_D)

                # create mean of all data within the time period and store
                dNdlogD['binned'][t, D_binned_idx] = np.nanmean(aps_dNdlogD[D_i_str][binary])




        # -----------------------------------------------------------------------

        # already have the times ready. Convert the dD np array to str and use them ask keys
        # set up a 'sum' entry for Nv to sum up all the values for a single time. Will need it to calc Dv
        # also set up a 'sum' entry for N_data (basically Ntot), again, will need it to calc Dv
        dVdD = {'time': dNdlogD['time'], 'rawtime': dNdlogD['rawtime'],
                'sum': np.empty(len(dNdlogD['time'])), 'Dv': np.empty(len(dNdlogD['time']))}
        dVdD['sum'][:] = np.nan
        dVdD['Dv'][:] = np.nan

        dNdD = {'time': dNdlogD['time'], 'rawtime': dNdlogD['rawtime'], 'sum': np.empty(len(dNdlogD['time']))}
        dNdD['sum'][:] = np.nan

        # to help calculate Dv (2 parts of an equation)
        dNdlogD['Dv'] = np.empty(len(dNdlogD['time']))
        # dNdlogD['y3'] = np.empty(len(dNdlogD['time']))
        # dNdlogD['x4'][:] = np.nan # part of equation to calc Dv
        dNdlogD['Dv'][:] = np.nan # 2nd part of eqn for Dv

        # Dv_data = {'time': dNdlogD['time'], 'rawtime': dNdlogD['rawtime'], 'Dv': np.empty(len(dNdlogD['time']))}
        # Dv_data['Dv'][:] = np.nan
        dVdlogD = {'time': dNdlogD['time'], 'rawtime': dNdlogD['rawtime'], 'Dv': np.empty(len(dNdlogD['time']))}  # tester
        dVdlogD['Dv'][:] = np.nan  # tester
        dNdlogD['sum'] = np.empty(len(dNdlogD['time']))
        dNdlogD['sum'][:] = np.nan

        # calc dN/dD (n_N(Dp)) from dN/dlogD: Seinfeld and Pandis 2007 eqn 8.18
        for D_i in D:
            key = str(D_i)  # key for dictionary

            dNdD[key] = dNdlogD[key] / (2.303 * D_i)  # calc nN(D) from nN(dlogD)

            # -----------------------------------------------------------------------
            # calc dV/dD from dN/dD (eqn 8.6)
            dVdD[key] = (np.pi / 6.0) * (D_i ** 3.0) * dNdD[key]

        # first extract and store the value for the current t, from each bin
        # create a sum entry for dVdD, adding together all values for a single time (sigma Nv)
        # create a sum entry for dNdD (Ntot or sigma Ni)
        for t in range(len(dNdD['time'])):  # was Nv_logD_data
            x4 = []
            y3 = []

            for D_i, dlogD_i in zip(D, dlogD):
                # total_dN += [dNdD[key][t]]  # turn dN/dD into dN (hopefully)
                # total_dV += [dVdD[key][t]]  # just dV/dD data
                key = str(D_i)
                x4 += [dNdlogD[key][t] * dlogD_i * (D_i ** 4.0)]  # turn dN/dD into dN (hopefully)
                y3 += [dNdlogD[key][t] * dlogD_i * (D_i ** 3.0)]  # just dV/dD data


            # once all bins for time t have been calculated, sum them up
            dNdlogD['Dv'][t] = np.sum(x4)/np.sum(y3)  # Ntotal for time(hopefully)

        # # first extract and store the value for the current t, from each bin
        # # create a sum entry for dVdD, adding together all values for a single time (sigma Nv)
        # # create a sum entry for dNdD (Ntot or sigma Ni)
        # for t in range(len(dNdD['time'])):  # was Nv_logD_data
        #     total_dN = []
        #     total_dV = []
        #     for D_i, dD_i in zip(D, dD):
        #         key = str(D_i)
        #         # total_dN += [dNdD[key][t]]  # turn dN/dD into dN (hopefully)
        #         # total_dV += [dVdD[key][t]]  # just dV/dD data
        #         total_dN += [dNdD[key][t] * dD_i]  # turn dN/dD into dN (hopefully)
        #         total_dV += [dVdD[key][t] * dD_i]  # just dV/dD data
        #
        #     dNdD['sum'][t] = np.sum(total_dN)  # Ntotal for time(hopefully)
        #     dVdD['sum'][t] = np.sum(total_dV)  # sum of dv/dD for all size bins for later
        #
        # # calculate volume mean diameter
        # for t in range(len(dVdD['time'])):
        #     # dVdD['Dv'][t] = (dVdD['sum'][t] / (dNdD['sum'][t] * (np.pi / 6.0))) ** 1. / 3.
        #     dVdD['Dv'][t] = (dVdD['sum'][t] / (dNdD['sum'][t] * (np.pi / 6.0))) ** 1. / 3.


        # data is 2D array [time, bin]
        # allow for soem time based stats to be used
        dVdlogD['binned'] = np.squeeze(np.array([
                            [np.array([dVdD[str(D_i)][j] for D_i in D])]
                            for j in range(len(dNdD['time']))]))



        # plt.semilogx(D, (np.array(dVdlogD['binned'][0]) * 1e-3) / 2)

        # median, IQRs
        dVdlogD['median'] = np.nanmedian(dVdlogD['binned'], axis=0)
        dVdlogD['25th'] = np.nanpercentile(dVdlogD['binned'], 75, axis=0)
        dVdlogD['75th'] = np.nanpercentile(dVdlogD['binned'], 25, axis=0)

        # plot volume distribution for data (median with IQR)
        fig = plt.figure()
        plt.semilogx(D*1e-3, dVdlogD['median'], label='median', color='blue')
        plt.fill_between(D*1e-3, dVdlogD['25th'], dVdlogD['75th'], alpha=0.5, facecolor='blue', label='IQR')
        plt.ylabel('dV/dlogD')
        plt.xlabel('D [microns]')
        # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
        plt.legend()
        plt.savefig(savedir + 'aerosol_distributions/dVdlogD_v_D_clearflo_winter.png')
        # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

        # ------------------------------------------------

        # create 2D array of data
        dNdlogD['binned'] = np.squeeze(np.array([
                            [np.array([dNdD[str(D_i)][j] for D_i in D])]
                            for j in range(len(dNdD['time']))]))

        # median, IQRs
        dNdlogD['median'] = np.nanmedian(dNdlogD['binned'], axis=0)
        dNdlogD['25th'] = np.nanpercentile(dNdlogD['binned'], 75, axis=0)
        dNdlogD['75th'] = np.nanpercentile(dNdlogD['binned'], 25, axis=0)

        # plot volume distribution for data (median with IQR)
        fig = plt.figure()
        plt.semilogx(D*1e-3, dNdlogD['median'], label='median', color='blue')
        plt.fill_between(D*1e-3, dNdlogD['25th'], dNdlogD['75th'], alpha=0.5, facecolor='blue', label='IQR')
        plt.ylabel('dN/dlogD')
        plt.xlabel('D [microns]')
        # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
        plt.legend()
        plt.savefig(savedir + 'aerosol_distributions/dNdlogD_v_D_clearflo_winter.png')
        # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

        # ------------------------------------------------

        # plot time series of r
        fig = plt.figure()
        # plt.plot((dVdD['Dv'] * 1e-3) / 2, label='rv using dN/dD')
        plt.plot((dNdlogD['Dv'] * 1e-3) / 2, label='rv using dN/dlogD')
        plt.ylabel('microns')
        # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
        plt.legend()
        plt.savefig(savedir + 'aerosol_distributions/rv_Ben_help.png')
        # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

    return

if __name__ == '__main__':
    main()

print 'END PROGRAM'