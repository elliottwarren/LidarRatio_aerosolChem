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
    convert from microns to nm
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

def merge_dmps_aps_dNdlogD(dmps_dNdlogD, dmps_D, aps_dNdlogD, aps_D):

    """
    Merge the dmps and aps dNdlogD datasets together, such a a 2D array called 'binned' is in the new dNdlogD array
    :param dmps_dNdlogD:
    :param dmps_D:
    :param aps_dNdlogD:
    :param aps_D:
    :return:
    """

    # time range - APS time res: 5 min, DMPS time res: ~12 min
    start_time = np.min([aps_dNdlogD['time'][0], dmps_dNdlogD['time'][0]])
    end_time = np.max([aps_dNdlogD['time'][-1], dmps_dNdlogD['time'][-1]])
    time_range = eu.date_range(start_time, end_time, 15, 'minutes')

    # binned shape = [time, bin]
    dNdlogD = {'time': time_range,
               'binned': np.empty([len(time_range), len(dmps_D) + len(aps_D)])}
    dNdlogD['binned'][:] = np.nan

    # insert the dmps data first, take average of all data within the time period
    for t in range(len(time_range)):

        for D_idx in range(len(dmps_D)):  # smaller data fill the first columns

            # enumerate might be useful here...

            # find data for this time
            # find data for this time
            binary = np.logical_and(dmps_dNdlogD['time'] > time_range[t],
                                    dmps_dNdlogD['time'] < time_range[t] + dt.timedelta(minutes=15))

            # bin str
            D_i_str = str(dmps_D[D_idx])

            # create mean of all data within the time period and store
            dNdlogD['binned'][t, D_idx] = np.nanmean(dmps_dNdlogD[D_i_str][binary])

        for D_idx in range(len(aps_D)):  # smaller data fill the first columns

            # find data for this time
            binary = np.logical_and(aps_dNdlogD['time'] > time_range[t],
                                    aps_dNdlogD['time'] < time_range[t] + dt.timedelta(minutes=15))

            # bin str
            D_i_str = str(aps_D[D_idx])

            # idx in new dNlogD array (data will go after the dmps data)
            D_binned_idx = D_idx + len(dmps_D)

            # create mean of all data within the time period and store
            dNdlogD['binned'][t, D_binned_idx] = np.nanmean(aps_dNdlogD[D_i_str][binary])

    return dNdlogD

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


    # read in RH data



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
    aps_D, aps_dD, aps_logD, aps_dlogD = calc_bin_parameters_aps(aps_N, units='nm')


    # calc dNdlogD from N, for aps data (coarse)
    aps_dNdlogD = calc_dNdlogD_from_N_aps(aps_N, aps_D, aps_dlogD, aps_header_bins)

    # -----------------------------------------------------------------------

    # merge the two dNdlogD datasets together...
    # set up so data resolution is 15 mins
    dNdlogD = merge_dmps_aps_dNdlogD(dmps_dNdlogD, dmps_D, aps_dNdlogD, aps_D)

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
    accum_minD, accum_minD_idx, _ = eu.nearest(D, 20.0)
    accum_maxD, accum_maxD_idx, _ = eu.nearest(D, 700.0)
    accum_range_idx = range(accum_minD_idx, accum_maxD_idx + 1)

    # Get volume mean diameter
    for t in range(len(dNdD['time'])):

        # total_dN += [dNdD[key][t]]  # turn dN/dD into dN (hopefully)
        # total_dV += [dVdD[key][t]]  # just dV/dD data
        x4 = dNdlogD['binned'][t, accum_range_idx] * dlogD[accum_range_idx] * (D[accum_range_idx] ** 4.0)  # turn dN/dD into dN (hopefully)
        y3 = dNdlogD['binned'][t, accum_range_idx] * dlogD[accum_range_idx] * (D[accum_range_idx] ** 3.0)  # just dV/dD data

        # once all bins for time t have been calculated, sum them up
        dNdlogD['Dv_accum'][t] = np.sum(x4)/np.sum(y3)

    # plot time series of r for defined acumm range
    fig = plt.figure()
    # plt.plot((dVdD['Dv'] * 1e-3) / 2, label='rv using dN/dD')
    plt.plot((dNdlogD['Dv_accum'] * 1e-3) / 2, label='rv using dN/dlogD')
    plt.ylabel('microns')
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.savefig(savedir + 'aerosol_distributions/rv_accum_0p02_0p7.png')
    # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

    # Get total number of particles (eqn 8.9)
    for t in range(len(dNdD['time'])):

        dNdlogD['Ntot_accum'][t] = np.sum(dNdlogD['binned'][t, accum_range_idx] * dlogD[accum_range_idx])

    # plot time series of N for defined acumm range
    fig = plt.figure()
    # plt.plot((dVdD['Dv'] * 1e-3) / 2, label='rv using dN/dD')
    plt.plot(dNdlogD['Ntot_accum'])
    plt.ylabel('Ntot [cm-3]')
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.savefig(savedir + 'aerosol_distributions/Ntot_accum_0p02_0p7.png')
    # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')



    # ==============================================================================
    # Plotting
    # ==============================================================================

    # median, IQRs
    dVdlogD['median'] = np.nanmedian(dVdlogD['binned'], axis=0)
    dVdlogD['25th'] = np.nanpercentile(dVdlogD['binned'], 75, axis=0)
    dVdlogD['75th'] = np.nanpercentile(dVdlogD['binned'], 25, axis=0)

    # plot volume distribution for data (median with IQR)
    fig = plt.figure()
    plt.semilogx(D*1e-3, dVdlogD['median'], label='median', color='blue')
    plt.vlines(0.5, 0, 3e10, linestyle='--', alpha=0.5)
    plt.fill_between(D*1e-3, dVdlogD['25th'], dVdlogD['75th'], alpha=0.5, facecolor='blue', label='IQR')
    plt.ylabel('dV/dlogD [nm3 cm-3]')
    plt.xlabel('D [microns]')
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.savefig(savedir + 'aerosol_distributions/dVdlogD_v_D_clearflo_winter_combined2.png')
    # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

    # ------------------------------------------------

    # median, IQRs
    dNdlogD['median'] = np.nanmedian(dNdlogD['binned'], axis=0)
    dNdlogD['25th'] = np.nanpercentile(dNdlogD['binned'], 75, axis=0)
    dNdlogD['75th'] = np.nanpercentile(dNdlogD['binned'], 25, axis=0)

    # plot volume distribution for data (median with IQR)
    fig = plt.figure()
    plt.semilogx(D, dNdlogD['median'], label='median', color='blue')
    plt.vlines(500, 0, np.nanmax(dNdlogD['25th']), linestyle='--', alpha=0.5)
    plt.fill_between(D, dNdlogD['25th'], dNdlogD['75th'], alpha=0.5, facecolor='blue', label='IQR')
    plt.ylabel('dN/dlogD [cm-3]')
    plt.xlabel('D [nm]')
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.savefig(savedir + 'aerosol_distributions/dNdlogD_v_D_clearflo_winter_combo.png')
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