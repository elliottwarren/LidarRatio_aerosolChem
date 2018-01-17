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
from matplotlib.dates import DateFormatter
import pickle
import netCDF4 as nc

import numpy as np
import datetime as dt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pandas

from copy import deepcopy
import colorsys

import ellUtils as eu
from forward_operator import FOUtils as FO
from forward_operator import FOconstants as FOcon


# Reading and reformatting

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

    """
    ToDo - Remove the overlapping radius bins so the calculation of dD and dlogD are accurate below 28 nm, see dmps 
    datafile. 
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

def calc_bin_parameters_aps(N, units='nm'):

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
    D_min = np.sort(D_min) * 1e3

    # upper edge on last bin is 21.29 microns, therefore append it on the end
    D_max = np.append(D_min[1:], 21.29*1e3)

    # mid bin
    D = (D_max + D_min) / 2.0
    logD = np.log10(D)

    # bin widths
    dD = D_max - D_min
    dlogD = np.log10(D_max) - np.log10(D_min)

    # # convert from microns to nm
    # didn't work for the dlogD one...
    # if units == 'nm':
    #
    #     D *= 1e3
    #     dD *= 1e3
    #     logD *= 1e3
    #     dlogD *= 1e3

    return D, dD, logD, dlogD

def calc_bin_parameters_smps(N, units='nm'):

    """
    Calculate bin parameters for the smps data
    headers are floats within a list, unlike aps which were keys within a dictionary
    ToDo make aps headers start as floats within a list
    :param N:
    :return:
    """

    # header is the mid bin D
    D = N['headers']

    # bin max -> the bin + half way to the next bin
    D_diffs = D[1:] - D[:-1] # checked
    D_max = D[:-1] + (D_diffs / 2.0) # checked

    # upper edge difference for the last bin is assumed be equal to the upper edge difference of the second to last bin
    #   therefore add the upper edge difference of the second to last bin, to the last bin.
    D_max = np.append(D_max, D[-1] + (D_diffs[-1]/2.0)) # checked
    # lower edge difference for the first bin is assumed to be equal to the lower edge difference of the second bin,
    #   therefore subtract the lower edge difference of the second bin, from the first bin.
    # lower edge of subsequent bins = upper edge of the previous bin, hence using D_max[:-1]
    D_min = np.append(D[0] - (D_diffs[0]/2.0), D_max[:-1]) # checked

    # bin parameters
    logD = np.log10(D)

    # bin widths
    dD = D_max - D_min
    dlogD = np.log10(D_max) - np.log10(D_min)

    # # convert from microns to nm
    # didn't work for the dlogD one...
    # if units == 'nm':
    #
    #     D *= 1e3
    #     dD *= 1e3
    #     logD *= 1e3
    #     dlogD *= 1e3

    N['D'] = D
    N['dD'] = dD
    N['logD'] = logD
    N['dlogD'] = dlogD

    return N

def calc_bin_parameters_grimm(N, units='nm'):

    """
    Calculate bin parameters for the GRIMM data
    grimm dictionary is set up a lot like the smps
    :param N:
    :return:
    """

    # header is the mid bin D
    D = N['D']

    # bin max -> the bin + half way to the next bin
    D_diffs = D[1:] - D[:-1] # checked
    D_max = D[:-1] + (D_diffs / 2.0) # checked

    # upper edge difference for the last bin is assumed be equal to the upper edge difference of the second to last bin
    #   therefore add the upper edge difference of the second to last bin, to the last bin.
    D_max = np.append(D_max, D[-1] + (D_diffs[-1]/2.0)) # checked
    # lower edge difference for the first bin is assumed to be equal to the lower edge difference of the second bin,
    #   therefore subtract the lower edge difference of the second bin, from the first bin.
    # lower edge of subsequent bins = upper edge of the previous bin, hence using D_max[:-1]
    D_min = np.append(D[0] - (D_diffs[0]/2.0), D_max[:-1]) # checked

    # bin parameters
    logD = np.log10(D)

    # bin widths
    dD = D_max - D_min
    dlogD = np.log10(D_max) - np.log10(D_min)

    N['D'] = D
    N['dD'] = dD
    N['logD'] = logD
    N['dlogD'] = dlogD

    return N

# processing / conversions

def calc_dNdlogD_from_N_aps(aps_N, aps_D, aps_dlogD, aps_header_bins):

    """
    Calculate dNdlogD from the basic N data, for the aps instrument
    :param aps_N:
    :param aps_D:
    :param aps_dlogD:
    :param aps_header_bins:
    :return: dNdlogD
    """

    aps_dNdlogD = {'time': aps_N['time']}
    for idx in range(len(aps_D)):
        # get bin name (start of bin for aps data annoyingly...), dlogD and D for each data column
        # convert bin ranges from microns to nm
        bin_i = aps_header_bins[idx]
        dlogD_i = aps_dlogD[idx]
        D_i = aps_D[idx]

        aps_dNdlogD[str(D_i)] = aps_N[bin_i] / dlogD_i

    return aps_dNdlogD

def calc_dNdlogD_from_N_smps(smps_N):

    """
    Calculate dNdlogD from the basic N data, for the smps instrument
    :param smps_N:
    :return: dNdlogD
    """

    smps_dNdlogD = {'time': smps_N['Date'],
                    'D': smps_N['D'],
                    'dD': smps_N['dD'],
                    'dlogD': smps_N['dlogD'],
                    'logD': smps_N['logD'],
                    'binned': np.empty(smps_N['binned'].shape)}
    smps_dNdlogD['binned'][:] = np.nan

    for D_idx, dlogD_i in enumerate(smps_N['dlogD']):
        smps_dNdlogD['binned'][:, D_idx] = smps_N['binned'][:, D_idx] / dlogD_i


    # for idx in range(len(smps_D)):
    #     # get bin name (start of bin for smps data annoyingly...), dlogD and D for each data column
    #     # convert bin ranges from microns to nm
    #     bin_i = smps_header_bins[idx]
    #     dlogD_i = smps_dlogD[idx]
    #     D_i = smps_D[idx]
    #
    #     smps_dNdlogD[str(D_i)] = smps_N[bin_i] / dlogD_i

    return smps_dNdlogD

def calc_dNdlogD_from_N_grimm(grimm_N):

    """
    Calculate dNdlogD from the basic N data, for the grimm instrument
    :param grimm_N:
    :param grimm_D:
    :param grimm_dlogD:
    :param grimm_header_bins:
    :return: dNdlogD
    """

    # set up dNdlogD dicionary
    grimm_dNdlogD = {'time': grimm_N['time'],
                     'D': grimm_N['D'],
                     'dD': grimm_N['dD'],
                     'dlogD': grimm_N['dlogD'],
                     'logD': grimm_N['logD'],
                    'binned': np.empty(grimm_N['binned'].shape)}
    grimm_dNdlogD['binned'][:] = np.nan

    # convert
    for D_idx, dlogD_i in enumerate(grimm_N['dlogD']):
        grimm_dNdlogD['binned'][:, D_idx] = grimm_N['binned'][:, D_idx] / dlogD_i

    return grimm_dNdlogD

def merge_dmps_aps_dNdlogD(dmps_dNdlogD, dmps_D, aps_dNdlogD, aps_D, timeRes=60):

    """
    Merge the dmps and aps dNdlogD datasets together, such a a 2D array called 'binned' is in the new dNdlogD array
    :param dmps_dNdlogD:
    :param dmps_D:
    :param aps_dNdlogD:
    :param aps_D:
    :param timeRes: time resolution of output data in minutes
    :return:
    """

    # time range - APS time res: 5 min, DMPS time res: ~12 min
    start_time = np.min([aps_dNdlogD['time'][0], dmps_dNdlogD['time'][0]])
    end_time = np.max([aps_dNdlogD['time'][-1], dmps_dNdlogD['time'][-1]])
    time_range = eu.date_range(start_time, end_time, timeRes, 'minutes')

    # binned shape = [time, bin]
    dNdlogD = {'time': time_range,
               'binned': np.empty([len(time_range), len(dmps_D) + len(aps_D)])}
    dNdlogD['binned'][:] = np.nan

    # # keep N values too
    # N = {'time': time_range,
    #      'binned': np.empty([len(time_range), len(dmps_D) + len(aps_D)])}
    # N['binned'][:] = np.nan

    # insert the dmps data first, take average of all data within the time period
    for t in range(len(time_range)):

        for D_idx in range(len(dmps_D)):  # smaller data fill the first columns

            # enumerate might be useful here...

            # find data for this time
            binary = np.logical_and(dmps_dNdlogD['time'] > time_range[t],
                                    dmps_dNdlogD['time'] < time_range[t] + dt.timedelta(minutes=timeRes))

            # bin str
            D_i_str = str(dmps_D[D_idx])

            # create mean of all data within the time period and store
            dNdlogD['binned'][t, D_idx] = np.nanmean(dmps_dNdlogD[D_i_str][binary])


        for D_idx in range(len(aps_D)):  # smaller data fill the first columns

            # find data for this time
            binary = np.logical_and(aps_dNdlogD['time'] > time_range[t],
                                    aps_dNdlogD['time'] < time_range[t] + dt.timedelta(minutes=timeRes))

            # bin str
            D_i_str = str(aps_D[D_idx])

            # idx in new dNlogD array (data will go after the dmps data)
            D_binned_idx = D_idx + len(dmps_D)

            # create mean of all data within the time period and store
            dNdlogD['binned'][t, D_binned_idx] = np.nanmean(aps_dNdlogD[D_i_str][binary])



    return dNdlogD

def merge_smps_grimm_dNdlogD(smps_dNdlogD, grimm_dNdlogD, timeRes=60):

    """
    Merge the dmps and aps dNdlogD datasets together, such a a 2D array called 'binned' is in the new dNdlogD array
    :param smps_dNdlogD:
    :param grimm_dNdlogD:
    :param timeRes: time resolution of output data in minutes
    :return:
    """

    # time range - APS time res: 5 min, DMPS time res: ~12 min
    start_time = np.min([smps_dNdlogD['time'][0], grimm_dNdlogD['time'][0]])
    end_time = np.max([smps_dNdlogD['time'][-1], grimm_dNdlogD['time'][-1]])
    time_range = eu.date_range(start_time, end_time, timeRes, 'minutes')

    # trim the grimm data that overlaps with the dmps (as smps has higher D binning resolution)
    # find which bins DO NOT overlap, and keep those.
    grimm_idx = np.where(grimm_dNdlogD['D'] >= smps_dNdlogD['D'][-1])

    # trim
    grimm_dNdlogD['D'] = grimm_dNdlogD['D'][grimm_idx]
    grimm_dNdlogD['dD'] = grimm_dNdlogD['dD'][grimm_idx]
    grimm_dNdlogD['dlogD'] = grimm_dNdlogD['dlogD'][grimm_idx]
    grimm_dNdlogD['logD'] = grimm_dNdlogD['logD'][grimm_idx]
    grimm_dNdlogD['binned'] = grimm_dNdlogD['binned'][:, grimm_idx]


    # binned shape = [time, bin]
    dNdlogD = {'time': time_range,
               'binned': np.empty([len(time_range), len(smps_dNdlogD['D']) + len(grimm_dNdlogD['D'])])}
    dNdlogD['binned'][:] = np.nan


    # insert the smps data first, take average of all data within the time period
    for t in range(len(time_range)):

        for D_idx in range(len(smps_D)):  # smaller data fill the first columns

            # enumerate might be useful here...

            # find data for this time
            binary = np.logical_and(smps_dNdlogD['time'] > time_range[t],
                                    smps_dNdlogD['time'] < time_range[t] + dt.timedelta(minutes=timeRes))

            # bin str
            D_i_str = str(smps_D[D_idx])

            # create mean of all data within the time period and store
            dNdlogD['binned'][t, D_idx] = np.nanmean(smps_dNdlogD[D_i_str][binary])


        for D_idx in range(len(grimm_D)):  # smaller data fill the first columns

            # find data for this time
            binary = np.logical_and(grimm_dNdlogD['time'] > time_range[t],
                                    grimm_dNdlogD['time'] < time_range[t] + dt.timedelta(minutes=timeRes))

            # bin str
            D_i_str = str(grimm_D[D_idx])

            # idx in new dNlogD array (data will go after the smps data)
            D_binned_idx = D_idx + len(smps_D)

            # create mean of all data within the time period and store
            dNdlogD['binned'][t, D_binned_idx] = np.nanmean(grimm_dNdlogD[D_i_str][binary])



    return dNdlogD, grimm_dNdlogD

def hourly_rh_threshold_pickle_save(dN, dVdlogD, dNdlogD, RH, D, dD, pickledir, RHthresh=60.0, equate='lt'):

    """
    Create hourly version of the aerosol data for saving. Also only take data based on an RH threshold
    :param dN:
    :param dVdlogD:
    :param dNdlogD:
    :param RH: relative humidity
    :param D:
    :param dD:
    :param pickledir:
    :param RHthresh:
    :param equate:
    :return: N_hourly [dict]
    """

    # create hourly WXT and dN data
    # remove dN data where RH was above a certain threshold
    date_range = eu.date_range(dN['time'][0], dN['time'][-1] + dt.timedelta(hours=1), 1, 'hours')
    N_hourly = {'dN': dN['binned'], 'time': dN['time'], 'D': D, 'dD': dD}
    N_hourly['dV/dlogD'] = deepcopy(dVdlogD['binned'])
    N_hourly['dN/dlogD'] = deepcopy(dNdlogD['binned'])
    N_hourly['RH'] =  np.empty(len(dN['time']))
    N_hourly['RH'][:] = np.nan
    for t, time_t in enumerate(dN['time'][:-1]):
        bool = np.logical_and(np.array(RH['time']) > time_t, np.array(RH['time']) < time_t + dt.timedelta(hours=1))

        med_RH = np.nanmean(RH['RH'][bool]) # RH
        med_RR = np.nanmean(RH['RR'][bool]) # rain rate in together with RH
        N_hourly['RH'][t] = med_RH

        # only keep values that are less than the RH threshold by np.nan values above it
        # only keep values if there was no rain present during the period
        if equate == 'lt':
            if (med_RH > RHthresh) | (med_RR > 0.0):
                N_hourly['dN'][t, :] = np.nan
                N_hourly['dV/dlogD'][t, :] = np.nan
                N_hourly['dN/dlogD'][t, :] = np.nan

        # only keep values that are greater than the threshold by np.nan values below it
        # only keep values if there was no rain present during the period
        elif equate == 'gt':
            if (med_RH < RHthresh) | (med_RR > 0.0):
                N_hourly['dN'][t, :] = np.nan
                N_hourly['dV/dlogD'][t, :] = np.nan
                N_hourly['dN/dlogD'][t, :] = np.nan


    # save dN data in pickle form
    # save time, Dv and N as a pickle
    # with open(pickledir + 'dN_dmps_aps_clearfloWinter.pickle', 'wb') as handle:
    #     pickle.dump(dN, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # NOTE! DO NOT USE protocol=HIGHEST..., loading from python in linux does not work for some uknown reason...
    with open(pickledir + 'N_hourly_dmps_aps_clearfloWinter_'+equate+str(RHthresh)+'_cut.pickle', 'wb') as handle:
        pickle.dump(N_hourly, handle)

    return N_hourly

# plotting

def quick_plot_dV(N_hourly, N_hourly_50, dVdlogD, savedir):

    # median, IQRs
    N_hourly['median'] = np.nanmedian(N_hourly['dV/dlogD'], axis=0)
    N_hourly['25th'] = np.nanpercentile(N_hourly['dV/dlogD'], 25, axis=0)
    N_hourly['75th'] = np.nanpercentile(N_hourly['dV/dlogD'], 75, axis=0)

    N_hourly_50['median'] = np.nanmedian(N_hourly_50['dV/dlogD'], axis=0)
    N_hourly_50['25th'] = np.nanpercentile(N_hourly_50['dV/dlogD'], 25, axis=0)
    N_hourly_50['75th'] = np.nanpercentile(N_hourly_50['dV/dlogD'], 75, axis=0)

    # median, IQRs
    dVdlogD['median'] = np.nanmedian(dVdlogD['binned'], axis=0)
    dVdlogD['25th'] = np.nanpercentile(dVdlogD['binned'], 25, axis=0)
    dVdlogD['75th'] = np.nanpercentile(dVdlogD['binned'], 75, axis=0)

    # plot volume distribution for data (median with IQR)
    # ok to use D from N_hourly as the bins are the same as the normal D variable
    fig = plt.figure(figsize=(5, 2.5))
    plt.semilogx(N_hourly['D']*1e-3, N_hourly['median']*1e-10, label=r'$RH < 60\%$', color='red')
    # plt.semilogx(N_hourly['D']*1e-3, N_hourly_50['median'], label=r'$RH < 50\%$', color='green')
    plt.semilogx(N_hourly['D']*1e-3, dVdlogD['median']*1e-10, label=r'$all \/\/data$', color='blue')
    plt.vlines(0.5, 0, 3, linestyle='--', alpha=0.5)
    plt.fill_between(N_hourly['D']*1e-3, N_hourly['25th']*1e-10, N_hourly['75th']*1e-10, alpha=0.2, facecolor='red', label=r'$RH < 60\% \/\/IQR$')
    # plt.fill_between(N_hourly['D']*1e-3, N_hourly_50['25th'], N_hourly_50['75th'], alpha=0.2, facecolor='green', label=r'$RH < 60\% IQR$')
    plt.fill_between(N_hourly['D'] * 1e-3, dVdlogD['25th']*1e-10, dVdlogD['75th']*1e-10, alpha=0.2, facecolor='blue', label=r'$all \/\/data \/\/IQR$')
    plt.ylabel('dV/dlogD '+r'$[1^{10}\/nm^{3}\/ cm^{-3}]$', labelpad=0)
    plt.xlabel(r'$Diameter \/\/[\mu m]$', labelpad=-3)
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir + 'aerosol_distributions/dVdlogD_v_D_clearflo_winter_combined_below60_and50_withRaincut.png')
    plt.close(fig)
    # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')




    return

def quick_plot_dN(N_hourly, N_hourly_50, dVdlogD, savedir):

    # median, IQRs
    N_hourly['median'] = np.nanmedian(N_hourly['dV/dlogD'], axis=0)
    N_hourly['25th'] = np.nanpercentile(N_hourly['dV/dlogD'], 25, axis=0)
    N_hourly['75th'] = np.nanpercentile(N_hourly['dV/dlogD'], 75, axis=0)

    N_hourly_50['median'] = np.nanmedian(N_hourly_50['dV/dlogD'], axis=0)
    N_hourly_50['25th'] = np.nanpercentile(N_hourly_50['dV/dlogD'], 25, axis=0)
    N_hourly_50['75th'] = np.nanpercentile(N_hourly_50['dV/dlogD'], 75, axis=0)

    # median, IQRs
    dVdlogD['median'] = np.nanmedian(dVdlogD['binned'], axis=0)
    dVdlogD['25th'] = np.nanpercentile(dVdlogD['binned'], 25, axis=0)
    dVdlogD['75th'] = np.nanpercentile(dVdlogD['binned'], 75, axis=0)

    # plot volume distribution for data (median with IQR)
    # ok to use D from N_hourly as the bins are the same as the normal D variable
    fig = plt.figure(figsize=(5, 2.5))
    plt.semilogx(N_hourly['D']*1e-3, N_hourly['median']*1e-10, label=r'$RH < 60\%$', color='red')
    # plt.semilogx(N_hourly['D']*1e-3, N_hourly_50['median'], label=r'$RH < 50\%$', color='green')
    plt.semilogx(N_hourly['D']*1e-3, dVdlogD['median']*1e-10, label=r'$all \/\/data$', color='blue')
    plt.vlines(0.5, 0, 3, linestyle='--', alpha=0.5)
    plt.fill_between(N_hourly['D']*1e-3, N_hourly['25th']*1e-10, N_hourly['75th']*1e-10, alpha=0.2, facecolor='red', label=r'$RH < 60\% \/\/IQR$')
    # plt.fill_between(N_hourly['D']*1e-3, N_hourly_50['25th'], N_hourly_50['75th'], alpha=0.2, facecolor='green', label=r'$RH < 60\% IQR$')
    plt.fill_between(N_hourly['D'] * 1e-3, dVdlogD['25th']*1e-10, dVdlogD['75th']*1e-10, alpha=0.2, facecolor='blue', label=r'$all \/\/data \/\/IQR$')
    plt.ylabel('dV/dlogD '+r'$[1^{10}\/nm^{3}\/ cm^{-3}]$', labelpad=0)
    plt.xlabel(r'$Diameter \/\/[\mu m]$', labelpad=-3)
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir + 'aerosol_distributions/dVdlogD_v_D_clearflo_winter_combined_below60_and50_withRaincut.png')
    plt.close(fig)
    # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')




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
    rhdatadir = maindir + 'data/L1/'
    pickledir = maindir + 'data/pickle/'

    # site and instruments
    # site_ins = {'site_short':'NK', 'site_long': 'North_Kensington', 'DMPS': True, 'APS': True, 'SMPS': False}
    site_ins = {'site_short':'Ch', 'site_long': 'Chilbolton', 'DMPS': False, 'APS': False, 'SMPS': True}

    # RH data
    site_rh = {'WXT_KSK': np.nan}
    rh_inst_site = site_rh.keys()[0]

    # time resolution of output data in minutes
    timeRes = 60

    # ==============================================================================
    # Read data
    # ==============================================================================

    # read in RH data
    # get all RH filenames and store them in a list
    # add rain rate [RR] to those extracted
    rhdays = eu.date_range(dt.datetime(2012, 01, 9), dt.datetime(2012, 02, 8), 1, 'days')
    RHfilepaths = [rhdatadir + rh_inst_site+'_'+day.strftime('%Y%j')+'_1min.nc' for day in rhdays]
    RH = eu.netCDF_read(RHfilepaths, vars=['RH', 'RR', 'time'])
    RH['time'] -= dt.timedelta(minutes=1) # change time from 'obs end' to 'start of obs', same as the other datasets
    RH['RH'][RH['RH'] == -999] = np.nan # remove bad data with nans
    RH['RR'][RH['RR'] == -999] = np.nan  # remove bad data with nans


    # read in number distribution from observed data ---------------------------

    # read in N data and extract relevent values
    # make N at all heights away from surface nan to be safe
    filepath = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/ClearfLo/' \
               'man-dmps_n-kensington_20120114_r0.na'

    # read in the ClearfLo data
    dmps_dNdlogD, dmps_header_bins = read_dmps(filepath)

    # remove the overlapping bins in both data and the headers!
    # VERY specific for the DMPS
    if filepath == 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/ClearfLo/' \
               'man-dmps_n-kensington_20120114_r0.na':
        for overlap_bin_i in ['22.7801', '25.2956', '28.0728']:

            del dmps_dNdlogD[overlap_bin_i]

            idx = np.where(dmps_header_bins == overlap_bin_i)
            dmps_header_bins = np.delete(dmps_header_bins, idx)
    else:
        raise ValueError('Script is designed for DMPS data read in - edit script before continuing!')

    # calculate bin centres and ranges for the dmps (D_mid)
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
    aps_D, aps_dD, aps_logD, aps_dlogD = calc_bin_parameters_aps(aps_N, units='nm')

    # calc dNdlogD from N, for aps data (coarse)
    aps_dNdlogD = calc_dNdlogD_from_N_aps(aps_N, aps_D, aps_dlogD, aps_header_bins)

    # -----------------------------------------------------------------------

    # Read in SMPS data (assuming all SMPS files share the same sort of format)
    filepath_SMPS = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/NPL/' \
                   'SMPS_Size_'+site_ins['site_long']+'_Annual_Ratified_2016_v01.xls'

    data_frame = pandas.read_excel(filepath_SMPS, 'Data')

    # remove the first column (date) and last 2 cols (empty col and total)
    N_raw = np.asarray(data_frame)
    smps_N = {'Date': np.array([i.to_datetime() for i in N_raw[:, 0]]),
              'binned': np.array(N_raw[:, 1:-2]),
              'headers': np.array(list(data_frame)[1:-2])}

    # get bin parameters
    smps_N = calc_bin_parameters_smps(smps_N, units='nm')

    # calc dNdlogD from N, for aps data
    smps_dNdlogD = calc_dNdlogD_from_N_smps(smps_N)

    # -----------------------------------------------------------------------

    # Read in the GRIMM EDM data
    # 2016 data only available from months 01 - 09 inclusively
    dirs = [maindir + 'data/Chilbolton/2016/' + '%02d'  % i + '/*.nc' for i in range(1, 10)]

    #grimmdays = eu.date_range(dt.datetime(2016, 1, 1), dt.datetime(2016, 9, 24), 1, 'days')
    grimmdays = eu.date_range(dt.datetime(2016, 1, 1), dt.datetime(2016, 9, 24), 1, 'days')
    grimmpaths = [maindir + 'data/Chilbolton/2016/'+day.strftime('%m')+'/' +
                   'cfarr-grimm_chilbolton_'+day.strftime('%Y%m%d')+'.nc' for day in grimmdays]

    # get sizes of arrys, so the multiple file read in data can be reshaped properly
    test = eu.netCDF_read(grimmpaths[0], vars=['particle_diameter', 'number_concentration_of_ambient_aerosol_in_air', 'time'])
    num_D_bins = len(test['particle_diameter'])

    # Read in all data from across multiple files
    # WARNING! GRIMM data is in m-3 not cm-3 like the other instruments! Therefore it is converted to cm-3 here
    # Reshape data that has been concatonated into vectors, into the proper 2D arrays
    raw = eu.netCDF_read(grimmpaths, vars=['number_concentration_of_ambient_aerosol_in_air', 'time'])

    grimm_N = {}
    grimm_N['binned'] = \
        raw['number_concentration_of_ambient_aerosol_in_air'].reshape(
                 (len(raw['number_concentration_of_ambient_aerosol_in_air'])/num_D_bins,
                  num_D_bins))
    grimm_N['binned'] *= 1e-06 # convert from m-3 to cm-3
    # multiple read in would just keep concatonating the diameters over and over
    # convert from microns to nm
    grimm_N['D'] = test['particle_diameter'] *1e03
    grimm_N['time'] = raw['time']

    # get bin parameters
    grimm_N = calc_bin_parameters_grimm(grimm_N, units='nm')

    # calc dNdlogD from N, for grimm data
    grimm_dNdlogD = calc_dNdlogD_from_N_grimm(grimm_N)

    # -----------------------------------------------------------------------

    # merge the dmps and APS dNdlogD datasets together...
    # set up so data resolution is 15 mins
    dNdlogD = merge_smps_grimm_dNdlogD(smps_dNdlogD, grimm_dNdlogD, timeRes=timeRes)

    # merge the aerosol parameters together too
    D = np.append(dmps_D, aps_D)
    logD = np.append(dmps_logD, aps_logD)
    dD = np.append(dmps_dD, aps_dD)
    dlogD = np.append(dmps_dlogD, aps_dlogD)

    # -----------------------------------------------------------------------

    # merge the dmps and APS dNdlogD datasets together...
    # set up so data resolution is 15 mins
    dNdlogD = merge_dmps_aps_dNdlogD(dmps_dNdlogD, dmps_D, aps_dNdlogD, aps_D, timeRes=timeRes)

    # merge the aerosol parameters together too
    D = np.append(dmps_D, aps_D)
    logD = np.append(dmps_logD, aps_logD)
    dD = np.append(dmps_dD, aps_dD)
    dlogD = np.append(dmps_dlogD, aps_dlogD)

    # ==============================================================================
    # Main processing of data
    # ==============================================================================


    # Set up the other aerosol distribution arrays in a similar style to dNdlogD (dictionaries with a 'binned' array)
    # ToDo keep 'sum' for now, though it originally served a prupose it is redunden, check if it might still be useful
    dVdD = {'time': dNdlogD['time'], 'binned': np.empty(dNdlogD['binned'].shape),
            'sum': np.empty(len(dNdlogD['time'])), 'Dv': np.empty(len(dNdlogD['time']))}
    dVdD['sum'][:] = np.nan
    dVdD['binned'][:] = np.nan
    dVdD['Dv'][:] = np.nan

    dNdD = {'time': dNdlogD['time'], 'sum': np.empty(len(dNdlogD['time'])), 'binned': np.empty(dNdlogD['binned'].shape)}
    dNdD['binned'][:] = np.nan
    dNdD['sum'][:] = np.nan

    # just the number of particles in each bin. Store all the diameter variables too, so it can be saved later
    dN = {'time': dNdlogD['time'], 'binned': np.empty(dNdlogD['binned'].shape), 'D': D, 'dD': dD}
    dN['binned'][:] = np.nan

    # Volume mean diameter
    dNdlogD['Dv'] = np.empty(len(dNdlogD['time']))
    dNdlogD['Dv'][:] = np.nan # 2nd part of eqn for Dv
    dNdlogD['Dv_accum'] = np.empty(len(dNdlogD['time']))
    dNdlogD['Dv_accum'][:] = np.nan # 2nd part of eqn for Dv


    dVdlogD = {'time': dNdlogD['time'], 'binned': np.empty(dNdlogD['binned'].shape),
               'Dv': np.empty(len(dNdlogD['time']))}  # tester
    dVdlogD['Dv'][:] = np.nan  # tester
    dVdlogD['binned'][:] = np.nan

    dNdlogD['Ntot'] = np.empty(len(dNdlogD['time'])) # total number of particles
    dNdlogD['Ntot'][:] = np.nan
    dNdlogD['Ntot_accum'] = np.empty(len(dNdlogD['time']))
    dNdlogD['Ntot_accum'][:] = np.nan

    # --------------------------------------

    # calc dN/dD (n_N(Dp)) from dN/dlogD: Seinfeld and Pandis 2007 eqn 8.18
    for i, D_i in enumerate(D):
        # key = str(D_i)  # key for dictionary

        logD_i = logD[i]

        # dNdD[key] = dNdlogD[key] / (2.303 * D_i)  # calc nN(D) from nN(dlogD)
        dNdD['binned'][:, i] = dNdlogD['binned'][:, i] / (2.303 * D_i)  # calc nN(D) from nN(dlogD)



        # -----------------------------------------------------------------------
        # calc dV/dD from dN/dD (eqn 8.6)
        dVdD['binned'][:, i] = (np.pi / 6.0) * (D_i ** 3.0) * dNdD['binned'][:, i]

        # calc dV/dlogD from dN/dlogD (eqn 8.6)
        dVdlogD['binned'][:, i] = (np.pi / 6.0) * (D_i ** 3.0) * dNdlogD['binned'][:, i]

    # calc dN from dN/dD(Actual number of particles per bin)
    for i, dD_i in enumerate(dD):

        dN['binned'][:, i] = dNdD['binned'][:, i] * dD_i

    # extract out only aerosol data based on the RH threshold and if it is less or more than it.
    N_hourly = hourly_rh_threshold_pickle_save(dN, dVdlogD, dNdlogD, RH, D, dD, pickledir, RHthresh=60.0, equate='lt')
    N_hourly_50 = hourly_rh_threshold_pickle_save(dN, dVdlogD, dNdlogD, RH, D, dD, pickledir, RHthresh=50.0, equate='lt')

    # quickplot what it looks like
    quick_plot_dV(N_hourly, N_hourly_50, dVdlogD, savedir)

    # # calc dN/dD (n_N(Dp)) from dN/dlogD: Seinfeld and Pandis 2007 eqn 8.18
    # first extract and store the value for the current t, from each bin
    # create a sum entry for dVdD, adding together all values for a single time (sigma Nv)
    # create a sum entry for dNdD (Ntot or sigma Ni)
    for t in range(len(dNdD['time'])):

        # total_dN += [dNdD[key][t]]  # turn dN/dD into dN (hopefully)
        # total_dV += [dVdD[key][t]]  # just dV/dD data
        x4 = dNdlogD['binned'][t, :] * dlogD * (D ** 4.0)  # turn dN/dD into dN (hopefully)
        y3 = dNdlogD['binned'][t, :] * dlogD * (D ** 3.0)  # just dV/dD data


        # once all bins for time t have been calculated, sum them up
        dNdlogD['Dv'][t] = np.sum(x4)/np.sum(y3)


    # ==============================================================================
    # Find Dv and N for the accum. range, for the aerFo
    # ==============================================================================

    # NOTE! D is in nm
    # try 0.02 to 0.7 microns first
    # accum_minD, accum_minD_idx, _ = eu.nearest(D, 40.0)
    # accum_maxD, accum_maxD_idx, _ = eu.nearest(D, 2000.0)
    accum_minD, accum_minD_idx, _ = eu.nearest(D, 1000.0)
    accum_maxD, accum_maxD_idx, _ = eu.nearest(D, 10000.0)
    accum_range_idx = range(accum_minD_idx, accum_maxD_idx + 1)

    # Get volume mean diameter
    for t in range(len(dNdD['time'])):

        # total_dN += [dNdD[key][t]]  # turn dN/dD into dN (hopefully)
        # total_dV += [dVdD[key][t]]  # just dV/dD data
        x4 = dNdlogD['binned'][t, accum_range_idx] * dlogD[accum_range_idx] * (D[accum_range_idx] ** 4.0)  # turn dN/dD into dN (hopefully)
        y3 = dNdlogD['binned'][t, accum_range_idx] * dlogD[accum_range_idx] * (D[accum_range_idx] ** 3.0)  # just dV/dD data

        # once all bins for time t have been calculated, sum them up
        dNdlogD['Dv_accum'][t] = np.sum(x4)/np.sum(y3)

    # # plot time series of r for defined acumm range
    # fig = plt.figure()
    # # plt.plot((dVdD['Dv'] * 1e-3) / 2, label='rv using dN/dD')
    # plt.plot_date(dNdlogD['time'], (dNdlogD['Dv_accum'] * 1e-3) / 2, label='rv using dN/dlogD', linestyle='-', fmt='-')
    # plt.ylabel('rv [microns]')
    # plt.xlabel('Date [dd/mm]')
    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
    # # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    # # plt.legend()
    # plt.savefig(savedir + 'aerosol_distributions/rv_accum_0p02_0p7.png')
    # plt.close(fig)
    # # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

    # Get total number of particles (eqn 8.9)
    for t in range(len(dNdD['time'])):

        dNdlogD['Ntot_accum'][t] = np.sum(dNdlogD['binned'][t, accum_range_idx] * dlogD[accum_range_idx])
        dNdlogD['Ntot'][t] = np.sum(dNdlogD['binned'][t, :] * dlogD)

    np.nanmean(dNdlogD['Ntot_accum'])

    # plot time series of N for defined acumm range
    fig = plt.figure()
    # plt.plot((dVdD['Dv'] * 1e-3) / 2, label='rv using dN/dD')
    plt.plot_date(dNdlogD['time'], dNdlogD['Ntot_accum'], linestyle='-', fmt='-')
    plt.ylabel('Ntot [cm-3]')
    plt.xlabel('Date [dd/mm]')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.savefig(savedir + 'aerosol_distributions/Ntot_accum_0p02_0p7.png')
    plt.close(fig)
    # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

    # save time, Dv and N as a pickle
    pickle_save = {'time': dNdlogD['time'], 'Ntot': dNdlogD['Ntot_accum'], 'Dv': dNdlogD['Dv_accum']}
    with open(pickledir + 'accum_Ntot_Dv_clearfloWinter.pickle', 'wb') as handle:
        pickle.dump(pickle_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # ==============================================================================
    # RH against r and N
    # ==============================================================================

    # subsample r or N based on RH

    # get coarsened RH data
    # plt histogram of it

    RH_15min = {'time': dNdlogD['time'], 'RH': np.empty(len(dNdlogD['time']))}
    RH_15min['RH'][:] = np.nan

    for t, time_i in enumerate(RH_15min['time']):

        # find the time for this period in the original 1min data
        binary = np.logical_and(RH['time'] > time_i,
                                RH['time'] < time_i + dt.timedelta(minutes=15))

        # create mean of all data within the time period and store
        RH_15min['RH'][t] = np.nanmean(RH['RH'][binary])

    # RH line plot of clearflo during winter
    fig, ax = plt.subplots(1,1)
    plt.plot_date(RH_15min['time'], RH_15min['RH'], linestyle='-', fmt='-')
    plt.xlabel('Date [dd/mm]')
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
    plt.ylabel('RH [%]')
    plt.suptitle('RH during clearflo winter iop')
    plt.savefig(savedir + 'aerosol_distributions/RH_lineplot.png')
    plt.close(fig)

    # -----------------------------------------------

    # quickly make a daily RH timeseries

    date_range = eu.date_range(dt.datetime(2012, 1, 10), dt.datetime(2012, 2, 8), 1, 'days')
    RH_day = {'time': date_range, 'RH': np.empty(len(date_range))}
    RH_day['RH'][:] = np.nan

    for t, time_i in enumerate(RH_day['time']):

        # find the time for this period in the original 1min data
        binary = np.logical_and(RH['time'] > time_i,
                                RH['time'] < time_i + dt.timedelta(days=1))

        # create mean of all data within the time period and store
        RH_day['RH'][t] = np.nanmean(RH['RH'][binary])

    # find driest day, quickplot data
    # np.where(RH_day['RH'] == np.nanmin(RH_day['RH']))
    # plt.plot(RH_day['RH'])

    # ------------------------------------------------

    # subsample dNdlogD based on RH

    # threshold
    thresh = 40.0

    low_idx  = np.logical_and(RH_15min['RH'] < thresh, ~np.isnan(dNdlogD['Dv_accum']))
    high_idx = np.logical_and(RH_15min['RH'] > thresh, ~np.isnan(dNdlogD['Dv_accum']))

    low_data = dNdlogD['Dv_accum'][low_idx]
    high_data = dNdlogD['Dv_accum'][high_idx]

    low_n = len(low_data)
    high_n = len(high_data)

    low_str  = '< '+str(thresh)+'%' + ': n = ' + str(low_n)
    high_str = '> '+str(thresh)+'%' + ': n = ' + str(high_n)

    nbins = 25

    fig, ax = plt.subplots(1,1)

    for data, lab in zip([low_data, high_data], [low_str, high_str]):
        n, bins = np.histogram(data, nbins, density=1)
        pdfx = np.zeros(n.size)
        pdfy = np.zeros(n.size)
        for k in range(n.size):
            pdfx[k] = 0.5 * (bins[k] + bins[k + 1])
            pdfy[k] = n[k]

        plt.plot(pdfx*1e-3, pdfy, label=lab)

    plt.legend()
    plt.ylabel('P(Dv)')
    plt.xlabel('D [microns]')
    plt.savefig(savedir + 'aerosol_distributions/RH_pdfs_'+str(thresh)+'thresh.png')
    plt.close(fig)

    # ==============================================================================
    # Plotting
    # ==============================================================================

    # median, IQRs
    dVdlogD['median'] = np.nanmedian(dVdlogD['binned'], axis=0)
    dVdlogD['25th'] = np.nanpercentile(dVdlogD['binned'], 25, axis=0)
    dVdlogD['75th'] = np.nanpercentile(dVdlogD['binned'], 75, axis=0)

    # plot volume distribution for data (median with IQR)
    fig = plt.figure(figsize=(5, 2.5))
    plt.semilogx(D*1e-3, dVdlogD['median'], label='median', color='blue')
    plt.vlines(0.5, 0, 3e10, linestyle='--', alpha=0.5)
    plt.fill_between(D*1e-3, dVdlogD['25th'], dVdlogD['75th'], alpha=0.5, facecolor='blue', label='IQR')
    plt.ylabel('dV/dlogD [nm3 cm-3]')
    plt.xlabel('D [microns]')
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir + 'aerosol_distributions/dVdlogD_v_D_clearflo_winter_combined2.png')
    plt.close(fig)
    # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

    # ------------------------------------------------

    # median, IQRs
    dNdlogD['median'] = np.nanmedian(dNdlogD['binned'], axis=0)
    dNdlogD['25th'] = np.nanpercentile(dNdlogD['binned'], 25, axis=0)
    dNdlogD['75th'] = np.nanpercentile(dNdlogD['binned'], 75, axis=0)

    # plot volume distribution for data (median with IQR)
    fig = plt.figure(figsize=(5, 2.5))
    plt.semilogx(D, dNdlogD['median'], label='median', color='blue')
    plt.vlines(500, 0, np.nanmax(dNdlogD['25th']), linestyle='--', alpha=0.5)
    plt.fill_between(D, dNdlogD['25th'], dNdlogD['75th'], alpha=0.5, facecolor='blue', label='IQR')
    plt.ylabel('dN/dlogD [cm-3]')
    plt.xlabel('D [nm]')
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir + 'aerosol_distributions/dNdlogD_v_D_clearflo_winter_combo.png')
    plt.close(fig)
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