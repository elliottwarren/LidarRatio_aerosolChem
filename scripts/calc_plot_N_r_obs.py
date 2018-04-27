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

def read_grimm_and_met_vars(maindir, year):

    """
    Read in the GRIMM data for Chilbolton AND the meteorological data(RH, Tair, pressure)
    from the instrument's additional equipment
    :param maindir:
    :param year:
    :return: grimm_N
    """

    # Read in the GRIMM EDM data
    # 2016 data only available from months 01 - 09 inclusively
    dirs = [maindir + 'data/Chilbolton/'+str(year)+'/' + '%02d'  % i + '/*.nc' for i in range(1, 10)]

    #grimmdays = eu.date_range(dt.datetime(2016, 1, 1), dt.datetime(2016, 9, 24), 1, 'days')
    grimmdays = eu.date_range(dt.datetime(year, 1, 1), dt.datetime(year, 9, 24), 1, 'days')
    grimmpaths = [maindir + 'data/Chilbolton/'+str(year)+'/'+day.strftime('%m')+'/' +
                   'cfarr-grimm_chilbolton_'+day.strftime('%Y%m%d')+'.nc' for day in grimmdays]

    # get sizes of arrys, so the multiple file read in data can be reshaped properly
    test = eu.netCDF_read(grimmpaths[0], vars=['particle_diameter', 'number_concentration_of_ambient_aerosol_in_air', 'time'])
    num_D_bins = len(test['particle_diameter'])

    # Read in all data from across multiple files
    # WARNING! GRIMM data is in m-3 not cm-3 like the other instruments! Therefore it is converted to cm-3 here
    # Reshape data that has been concatonated into vectors, into the proper 2D arrays
    # RH [%]; Tair [K]; pressure [Pa]
    raw = eu.netCDF_read(grimmpaths,
                vars=['number_concentration_of_ambient_aerosol_in_air',
                      'relative_humidity', 'air_temperature', 'air_pressure', 'wind_speed', 'wind_from_direction', 'time'])

    grimm_N = {}
    grimm_N['binned'] = \
        raw['number_concentration_of_ambient_aerosol_in_air'].reshape(
                 (len(raw['number_concentration_of_ambient_aerosol_in_air'])/num_D_bins,
                  num_D_bins))

    # QAQC GRIMM data
    # remove bad data for the entire row, if any one value in the row is bad (fill value for bad data = -999)
    idx = np.where(grimm_N['binned'] < 0.0)[0] # juts the row [0]
    grimm_N['binned'][idx, :] = np.nan

    # some unrealistically high values can occur in the highest bins
    idx = np.where(grimm_N['binned'][:, -1] > 0.0001)[0] # 100 cm-3 where D = 32 microns...
    grimm_N['binned'][idx, :] = np.nan

    # convert from m-3 to cm-3
    grimm_N['binned'] *= 1e-06
    # multiple read in would just keep concatonating the diameters over and over
    grimm_N['D'] = test['particle_diameter'] *1e03 # convert from microns to nm
    grimm_N['time'] = raw['time']

    # Meteorological vars Relative humidity (with consistent variable names to RH read from LUMO)
    met_vars = {'time': grimm_N['time'],
                'RH': raw['relative_humidity'],
                'Tair': raw['air_temperature'],
                'pressure': raw['air_pressure'],
                'RH_frac': raw['relative_humidity']/100.0}

    return grimm_N, met_vars

def read_clearflo_aps(filepath_aps):

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

def read_routine_aps(filepath_aps):

    """
    Read in the routinely taken APS data, given to us by ERG
    :param filepath_aps:
    :return: N [dictionary with each header as a key]
    :return: header_mid: the headers [list of strings]
    """

    # APS headers
    filepath_aps_headers = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/ERG/' \
                    'APS species definitions.xlsx'

    # read main data
    N_raw = np.genfromtxt(filepath_aps, delimiter=',', dtype='|S20')
    pro_time = np.array([dt.datetime.strptime(i, '%d-%m-%y %H:%M') for i in N_raw[1:, 0]])

    # get headers and idx for just the particle size data - ignore the first column as it is <0.523
    raw_headers = np.array([i.split('@')[0] for i in N_raw[0]])
    particle_bool = np.array([True if (((i[:2] == 'P1') | (i[:2] == 'P2')) & (i != 'P165')) # SKIP 165 - 1st header
                              else False for i in raw_headers])
    particle_idx = np.where(particle_bool == True)
    raw_headers_particles = raw_headers[particle_idx]

    # read in APS header definitions - ignore the first column as it is <0.523
    header_defs = pandas.read_excel(filepath_aps_headers, skiprows=1)
    header_defs_array = np.asarray(header_defs)
    header_split = np.array([i[1].split(' ') for i in header_defs_array])
    header_orig = np.array([str(i[0]) for i in header_defs_array])
    # header_mid = np.array([str(i[3]) for i in header_split])
    header_mid = np.array([float(i[3]) for i in header_split])

    # convert header units from microns to nanometres
    header_mid *= 1.0e3

    # if the headers line up
    if header_orig[0] == raw_headers_particles[0]:

        # Turn N data into a numpy array with float values and np.nan where originally it was ''
        # There are diagnostic variables in the later columns that we don't want to extract.
        #     Also, we skip the first header, as it is not a well defined bin (less than and not a bin centre),
        #     hence N_raw[1:, 2:len(header_mid)+2].

        replace_idx = np.where(N_raw == '')
        N_raw[replace_idx] = np.nan
        N_processed = np.array(N_raw[1:, 2:len(header_mid)+2], dtype=float)

        # create N
        N = {'binned': N_processed,
             'time': pro_time,
             'headers': header_mid}

    else:
        raise ValueError('Headers do not allign between APS data and APS definitions!')

    return N

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

def calc_bin_parameters_aps_clearflo(aps_N, units='nm'):

    """
    Calculate bin parameters for the aps data
    :param N:
    :return:
    """

    D_keys = aps_N.keys()
    for key in ['time', 'rawtime', 'Total']:
        if key in D_keys:
            del D_keys[D_keys.index(key)]

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

def calc_bin_parameters_smps_dN(N, units='nm'):

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


    # calc dN

    # # header is the mid bin D
    # D = N['headers']
    # N = smps_N

    # bin max -> the bin + half way to the next bin
    N_diffs = N['binned'][:, 1:] - N['binned'][:, :-1] # checked
    N_max = N['binned'][:, :-1] + (N_diffs / 2.0) # checked

    # upper edge difference for the last bin is assumed be equal to the upper edge difference of the second to last bin
    #   therefore add the upper edge difference of the second to last bin, to the last bin.
    end = (N['binned'][:, -1] + (N_diffs[:, -1]/2.0))
    end = end.reshape((len(end), 1))
    N_max = np.hstack((N_max, end)) # checked
    # lower edge difference for the first bin is assumed to be equal to the lower edge difference of the second bin,
    #   therefore subtract the lower edge difference of the second bin, from the first bin.
    # lower edge of subsequent bins = upper edge of the previous bin, hence using D_max[:-1]
    start = (N['binned'][:, 0] - (N_diffs[:, 0]/2.0))
    start = start.reshape(len(start), 1)
    end = N_max[:, :-1]
    N_min = np.hstack((start, end)) # checked

    dN = N_max - N_min

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
    N['binned'] = dN
    # N['dN'] = dN


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

def calc_bin_parameters_general(N, units='nm'):

    """
    Calculate bin parameters for the data
    headers are floats within a list, unlike aps which were keys within a dictionary
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

# conversions / merging

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

    smps_dNdlogD = {'time': smps_N['time'],
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

def calc_dNdlogD_from_N_smps_with_dN(smps_N):

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

def calc_dNdlogD_from_N_general(N):

    """
    Calculate dNdlogD from the basic N data, for the smps instrument
    :param smps_N:
    :return: dNdlogD
    """

    dNdlogD = {'time': N['time'],
                    'D': N['D'],
                    'dD': N['dD'],
                    'dlogD': N['dlogD'],
                    'logD': N['logD'],
                    'binned': np.empty(N['binned'].shape)}
    dNdlogD['binned'][:] = np.nan

    for D_idx, dlogD_i in enumerate(N['dlogD']):
        dNdlogD['binned'][:, D_idx] = N['binned'][:, D_idx] / dlogD_i

    return dNdlogD

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
                                    dmps_dNdlogD['time'] <= time_range[t] + dt.timedelta(minutes=timeRes))

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
    Merge the smps and grimm dNdlogD datasets together, such a a 2D array called 'binned' is in the new dNdlogD array
    :param smps_dNdlogD:
    :param grimm_dNdlogD:
    :param timeRes: time resolution of output data in minutes
    :return:

    generalised so it could take 2 different instruments. Searches a subsample of the time array, based on what times
    have been used in previous iterations of t. Otherwise the code takes ages to run!
    NOTE: ASSUMES data is sorted in time order (data is ok to have missing times in it)
    """

    # time range - ToDo should probably swap the min and maxes around to ensure times have BOTH sets of data present
    start_time = np.max([smps_dNdlogD['time'][0], grimm_dNdlogD['time'][0]])
    end_time = np.min([smps_dNdlogD['time'][-1], grimm_dNdlogD['time'][-1]])
    # original
    # start_time = np.min([smps_dNdlogD['time'][0], grimm_dNdlogD['time'][0]])
    # end_time = np.max([smps_dNdlogD['time'][-1], grimm_dNdlogD['time'][-1]])
    time_range = eu.date_range(start_time, end_time, timeRes, 'minutes')

    # trim the grimm data that overlaps with the dmps (as smps has higher D binning resolution)
    # find which bins DO NOT overlap, and keep those.
    grimm_idx = np.where(grimm_dNdlogD['D'] >= smps_dNdlogD['D'][-1])[0]

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

    smps_skip_idx = 0
    grimm_skip_idx = 0

    # insert the smps data first, take average of all data within the time period
    for t in range(len(time_range)):

        ## 1. SMPS

        # find data for this time
        # currently set subsample to be arbitrarily 200 elements above the previous sampled element's idx.
        smps_binary = np.logical_and(smps_dNdlogD['time'][smps_skip_idx:smps_skip_idx+200] > time_range[t],
                                     smps_dNdlogD['time'][smps_skip_idx:smps_skip_idx+200] <= time_range[t] + dt.timedelta(minutes=timeRes))

        # get idx location of the current subsample (smps_skip_idx:smps_skip_idx+200) and add on how many idx elements
        #   have been skipped so far (smps_skip_idx), such that smps_idx = location of elements in the time period
        #   within the entire array [:]
        smps_idx = np.where(smps_binary == True)[0] + smps_skip_idx

        for D_idx in range(len(smps_dNdlogD['D'])):  # smaller data fill the first columns

            # create mean of all data within the time period and store
            dNdlogD['binned'][t, D_idx] = np.nanmean(smps_dNdlogD['binned'][smps_idx, D_idx])

        # change skip idx to miss data already used in averaging
        #   so the entire time array doesn't need to be searched through for each iteration of t
        if smps_idx.size != 0:
            smps_skip_idx = smps_idx[-1] + 1

        # -------------

        ## 2. GRIMM

        # find data for this time
        grimm_binary = np.logical_and(grimm_dNdlogD['time'][grimm_skip_idx:grimm_skip_idx+200] > time_range[t],
                                      grimm_dNdlogD['time'][grimm_skip_idx:grimm_skip_idx+200] <= time_range[t] + dt.timedelta(minutes=timeRes))
        # idx = np.where(binary == True)[0]

        grimm_idx = np.where(grimm_binary == True)[0] + grimm_skip_idx

        for D_idx in range(len(grimm_dNdlogD['D'])):  # smaller data fill the first columns

            # idx in new dNlogD array (data will go after the smps data)
            D_binned_idx = D_idx + len(smps_dNdlogD['D'])

            # create mean of all data within the time period and store
            # dNdlogD['binned'][t, D_binned_idx] = np.nanmean(grimm_dNdlogD['binned'][grimm_binary, D_idx])
            dNdlogD['binned'][t, D_binned_idx] = np.nanmean(grimm_dNdlogD['binned'][grimm_idx, D_idx])

        # change skip idx to miss data already used in averaging
        if grimm_idx.size != 0:
            grimm_skip_idx = grimm_idx[-1] + 1

    # merge the dimaeter bins together too
    for param in ['D', 'dD', 'dlogD', 'logD']:
        dNdlogD[param] = np.append(smps_dNdlogD[param], grimm_dNdlogD[param])

    # keep the index position of the original instruments D, within the combined D range
    smps_D_idx = np.arange(len(smps_dNdlogD['D']))
    grimm_D_idx = np.arange(smps_D_idx[-1] + 1, len(dNdlogD['D']))

    return dNdlogD, grimm_dNdlogD, smps_D_idx, grimm_D_idx

def merge_small_large_dNdlogD(small_dNdlogD, large_dNdlogD, timeRes=60):

    """
    Merge the small and large dNdlogD datasets together, such a a 2D array called 'binned' is in the new dNdlogD array
    :param small_dNdlogD:
    :param large_dNdlogD:
    :param timeRes: time resolution of output data in minutes
    :return:

    generalised so it could take 2 different instruments. Searches a subsample of the time array, based on what times
    have been used in previous iterations of t. Otherwise the code takes ages to run!
    NOTE: ASSUMES data is sorted in time order (data is ok to have missing times in it)
    """

    # time range - make sure mins = 0
    start_time = np.max([small_dNdlogD['time'][0], large_dNdlogD['time'][0]])
    end_time = np.min([small_dNdlogD['time'][-1], large_dNdlogD['time'][-1]])
    start_time -= dt.timedelta(minutes=start_time.minute)
    end_time -= dt.timedelta(minutes=end_time.minute)

    # original
    # start_time = np.min([small_dNdlogD['time'][0], large_dNdlogD['time'][0]])
    # end_time = np.max([small_dNdlogD['time'][-1], large_dNdlogD['time'][-1]])
    time_range = eu.date_range(start_time, end_time, timeRes, 'minutes')

    # trim the grimm data that overlaps with the dmps (as smps has higher D binning resolution)
    # find which bins DO NOT overlap, and keep those.
    large_idx = np.where(large_dNdlogD['D'] >= small_dNdlogD['D'][-1])[0]

    # trim
    large_dNdlogD['D'] = large_dNdlogD['D'][large_idx]
    large_dNdlogD['dD'] = large_dNdlogD['dD'][large_idx]
    large_dNdlogD['dlogD'] = large_dNdlogD['dlogD'][large_idx]
    large_dNdlogD['logD'] = large_dNdlogD['logD'][large_idx]
    large_dNdlogD['binned'] = large_dNdlogD['binned'][:, large_idx]


    # binned shape = [time, bin]
    dNdlogD = {'time': time_range,
               'binned': np.empty([len(time_range), len(small_dNdlogD['D']) + len(large_dNdlogD['D'])])}
    dNdlogD['binned'][:] = np.nan

    small_skip_idx = 0
    large_skip_idx = 0

    # insert the smps data first, take average of all data within the time period
    for t in range(len(time_range)):

        ## 1. SMPS

        # find data for this time
        # currently set subsample to be arbitrarily 200 elements above the previous sampled element's idx.
        small_binary = np.logical_and(small_dNdlogD['time'][small_skip_idx:small_skip_idx+200] > time_range[t],
                                     small_dNdlogD['time'][small_skip_idx:small_skip_idx+200] <= time_range[t] + dt.timedelta(minutes=timeRes))

        # get idx location of the current subsample (small_skip_idx:small_skip_idx+200) and add on how many idx elements
        #   have been skipped so far (small_skip_idx), such that small_idx = location of elements in the time period
        #   within the entire array [:]
        small_idx = np.where(small_binary == True)[0] + small_skip_idx

        for D_idx in range(len(small_dNdlogD['D'])):  # smaller data fill the first columns

            # create mean of all data within the time period and store
            dNdlogD['binned'][t, D_idx] = np.nanmean(small_dNdlogD['binned'][small_idx, D_idx])

        # change skip idx to miss data already used in averaging
        #   so the entire time array doesn't need to be searched through for each iteration of t
        if small_idx.size != 0:
            small_skip_idx = small_idx[-1] + 1

        # -------------

        ## 2. GRIMM

        # find data for this time
        large_binary = np.logical_and(large_dNdlogD['time'][large_skip_idx:large_skip_idx+200] >= time_range[t],
                                      large_dNdlogD['time'][large_skip_idx:large_skip_idx+200] < time_range[t] + dt.timedelta(minutes=timeRes))
        # idx = np.where(binary == True)[0]

        large_idx = np.where(large_binary == True)[0] + large_skip_idx

        for D_idx in range(len(large_dNdlogD['D'])):  # smaller data fill the first columns

            # idx in new dNlogD array (data will go after the smps data)
            D_binned_idx = D_idx + len(small_dNdlogD['D'])

            # create mean of all data within the time period and store
            # dNdlogD['binned'][t, D_binned_idx] = np.nanmean(large_dNdlogD['binned'][large_binary, D_idx])
            dNdlogD['binned'][t, D_binned_idx] = np.nanmean(large_dNdlogD['binned'][large_idx, D_idx])

        # change skip idx to miss data already used in averaging
        if large_idx.size != 0:
            large_skip_idx = large_idx[-1] + 1

    # merge the dimaeter bins together too
    for param in ['D', 'dD', 'dlogD', 'logD']:
        dNdlogD[param] = np.append(small_dNdlogD[param], large_dNdlogD[param])

    # keep the index position of the original instruments D, within the combined D range
    small_D_idx = np.arange(len(small_dNdlogD['D']))
    large_D_idx = np.arange(small_D_idx[-1] + 1, len(dNdlogD['D']))

    return dNdlogD, large_dNdlogD, small_D_idx, large_D_idx

# processing / calculations

def get_size_range_idx(D, D_min, D_max):

    """
    Get the size range idx for all diameters between the given minimum and maximum diameters.
    :return: accum_range_idx

    Function would work if diameter was replaced with radius
    """

    accum_minD, accum_minD_idx, _ = eu.nearest(D, D_min)
    accum_maxD, accum_maxD_idx, _ = eu.nearest(D, D_max)
    accum_range_idx = range(accum_minD_idx, accum_maxD_idx + 1)

    return accum_range_idx

def calc_volume_and_number_mean_diameter(size_range_idx, dNdD, dNdlogD, dlogD, D, units='nm'):

    """
    Calcualted volume and number mean diameter

    :param size_range_idx: idx for the size range
    :param dNdD:
    :param dNdlogD:
    :param dlogD:
    :param D:
    :return: Dv: volume mean diameter
    :return: Dn: number mean diameter

    Units are in nm be default but should match the units used by the variable 'D' and 'dlogD'
    NOTE: N * D = sum of all the particles D's.
    """

    # set up arrays to fill
    # Dv = volume mean diameter
    Dv = np.empty(len(dNdlogD['time']))
    Dv[:] = np.nan # 2nd part of eqn for Dv

    # Dn = arithmetic mean diameter
    Dn = np.empty(len(dNdlogD['time']))
    Dn[:] = np.nan # 2nd part of eqn for Dv

    # Dg = geometric mean diameter
    Dg = np.empty(len(dNdlogD['time']))
    Dg[:] = np.nan # 2nd part of eqn for Dv

    # Get volume mean diameter
    for t in range(len(dNdD['time'])):

        # 1. Volume mean diameter
        # calculate two parts of the eq. first
        # # dNdlogD * dlogD * D = N of original bin
        # 1/6 * pi * D**3 = V (volume sphere)
        #   below = sum(V * D) / sum(V)
        x1 = dNdlogD['binned'][t, size_range_idx] * dlogD[size_range_idx] * (D[size_range_idx] ** 4.0)  # V * D
        y1 = dNdlogD['binned'][t, size_range_idx] * dlogD[size_range_idx] * (D[size_range_idx] ** 3.0)  # just V

        # once all bins for time t have been calculated, sum them up
        Dv[t] = np.sum(x1)/np.sum(y1)

        # 2. Number mean diameter
        # calculate two parts of the eq. first
        # dNdlogD * dlogD * D = N of original bin
        #   -> sum(N * D) / sum(N) = number mean

        x2 = dNdlogD['binned'][t, size_range_idx] * dlogD[size_range_idx] * D[size_range_idx] # N* D
        y2 = dNdlogD['binned'][t, size_range_idx] * dlogD[size_range_idx]

        Dn[t] = np.sum(x2)/np.sum(y2)


        # gemoetric number mean diameter
        # source 1: CMD or Dg:
        #    http://www.tsi.com/uploadedFiles/_Site_Root/Products/Literature/Application_Notes/PR-001-RevA_Aerosol-Statistics-AppNote.pdf
        # source 2: section 1.3: http://eodg.atm.ox.ac.uk/user/grainger/research/aerosols.pdf
        x3 = D[size_range_idx] ** (dNdlogD['binned'][t, size_range_idx] * dlogD[size_range_idx]) #[D1**n1, D2**n2 ...] (TSI notes)
        y3 = dNdlogD['binned'][t, size_range_idx] * dlogD[size_range_idx] # [n1, n2 ... nN]

        Dg[t] = np.prod(x3) ** (1.0/np.sum(y3))


    return Dv, Dn

def calc_geometric_mean_and_stdev_radius(size_range_idx, dNdD, dNdlogD, dlogD, D, units='nm'):

    """
    Calcualted the geometric mean and geometric standard deviation of the aerosol distribution with respect to RADIUS
    as required by SOCRATES

    :param size_range_idx: idx for the size range
    :param dNdD:
    :param dNdlogD:
    :param dlogD:
    :param D:
    :return: r_g: geometric mean of the radii distribution
    :return: stdev_g_r: geometric standard deviation of the radii distribution

    Units are in nm be default but should match the units used by the variable 'D' and 'dlogD'.
    xn and yn are parts of the main equations.
    NOTE: N * D = sum of all the particles D's.
    """

    # calculated r parameters from D
    def calc_r_parameters(D):

        """
        Calculate radii parameters from the diameter
        :param D:
        :return: r parameters: r, dr, logr, dlogr
        """

        # calc r from D
        r = D/2.0

        # bin max -> the bin + half way to the next bin
        r_diffs = r[1:] - r[:-1]  # checked
        r_max = r[:-1] + (r_diffs / 2.0)  # checked

        # upper edge difference for the last bin is assumed be equal to the upper edge difference of the second to last bin
        #   therefore add the upper edge difference of the second to last bin, to the last bin.
        r_max = np.append(r_max, r[-1] + (r_diffs[-1] / 2.0))  # checked
        # lower edge difference for the first bin is assumed to be equal to the lower edge difference of the second bin,
        #   therefore subtract the lower edge difference of the second bin, from the first bin.
        # lower edge of subsequent bins = upper edge of the previous bin, hence using r_max[:-1]
        r_min = np.append(r[0] - (r_diffs[0] / 2.0), r_max[:-1])  # checked

        # bin parameters
        logr = np.log10(r)

        # bin widths
        dr = r_max - r_min
        dlogr = np.log10(r_max) - np.log10(r_min)

        return r, dr, logr, dlogr

    # calculat r parameters from D and log(D)
    r, dr, logr, dlogr = calc_r_parameters(D)

    # create dNdlogr to calculate geometric standard deviation lower down
    dNdlogr = (dNdlogD['binned'] * dlogD) / dlogr

    # Dg = geometric mean diameter
    r_g = np.empty(len(dNdlogD['time']))
    r_g[:] = np.nan # 2nd part of eqn for Dv

    # stdevg = geometric standard deviation RADIUS
    stdev_g_r = np.empty(len(dNdlogD['time']))
    stdev_g_r[:] = np.nan

    for t in range(len(dNdD['time'])):

        # # 1. Gemoetric number mean diameter
        # # source 1: CMD or Dg:
        # #    http://www.tsi.com/uploadedFiles/_Site_Root/Products/Literature/Application_Notes/PR-001-RevA_Aerosol-Statistics-AppNote.pdf
        # # source 2: section 1.3: http://eodg.atm.ox.ac.uk/user/grainger/research/aerosols.pdf
        # x1 = dNdlogD['binned'][t, size_range_idx] * dlogD[size_range_idx] # [n1, n2 ... nN]
        # x2 = (1.0 / np.sum(x1))  # 1/N
        #
        # # [(D1**(n1*1/N), (D2**(n2*1/N) ...] (rearanged eq from TSI notes as otherwise the computer cant store the number properly!)
        # x3 = (D[size_range_idx] ** ((dNdlogD['binned'][t, size_range_idx] * dlogD[size_range_idx]) * x2))
        #
        # # geometric mean diameter
        # Dg = np.prod(x3)
        #
        # # geometric mean radius
        # r_g[t] = Dg / 2.0

        # 1. Gemoetric number mean diameter
        # source 1: CMD or Dg:
        #    http://www.tsi.com/uploadedFiles/_Site_Root/Products/Literature/Application_Notes/PR-001-RevA_Aerosol-Statistics-AppNote.pdf
        # source 2: section 1.3: http://eodg.atm.ox.ac.uk/user/grainger/research/aerosols.pdf
        x1 = dNdlogr[t, size_range_idx] * dlogr[size_range_idx] # [n1, n2 ... nN]
        x2 = (1.0 / np.sum(x1))  # 1/N

        # [(D1**(n1*1/N), (D2**(n2*1/N) ...] (rearanged eq from TSI notes as otherwise the computer cant store the number properly!)
        x3 = (r[size_range_idx] ** ((dNdlogr[t, size_range_idx] * dlogr[size_range_idx]) * x2))

        # geometric mean diameter
        r_g[t] = np.prod(x3)


        # 2. Geometric standard deviation
        #   same sources as above
        y1 = (dNdlogr[t, size_range_idx] * dlogr[size_range_idx]) * ((logr[size_range_idx] - np.log10(r_g[t])) ** 2.0) # Ni * (diff in r from mean)**2

        y2 = np.sum(dNdlogr[t, size_range_idx] * dlogr[size_range_idx]) - 1 # N - 1

        # as the data uses log10, use that base to recover stdev_g_r
        stdev_g_r[t] = 10 ** ((np.sum(y1) / y2) ** 0.5)

    return r_g, stdev_g_r

def hourly_rh_threshold_pickle_save_fast(dN, dVdlogD, dNdlogD, met_vars, D, dD, pickledir, savestr, smps_D_idx, grimm_D_idx,
                                         subsample=True, RHthresh=60.0, equate='lt', save=True, extra=''):

    """
    Create hourly version of the aerosol data for saving. Also only take data based on an RH threshold
    :param dN:
    :param dVdlogD:
    :param dNdlogD:
    :param met_vars:
    :param D:
    :param dD:
    :param pickledir:
    :param savestr:
    :param smps_D_idx:
    :param grimm_D_idx:
    :param subsample:
    :param RHthresh:
    :param equate:
    :param save [bool]: Turn saving on/off
    :param extra: extra string to change the savename
    :return:
    """

    # create hourly WXT and dN data
    # remove dN data where RH was above a certain threshold
    # date_range = eu.date_range(dN['time'][0], dN['time'][-1] + dt.timedelta(hours=1), 1, 'hours')
    N_hourly = {'time': dN['time'], 'D': D, 'dD': dD, 'smps_idx': smps_D_idx, 'grimm_idx': grimm_D_idx}

    # copy the already avergaed up aerosol variables into N_hourly.
    N_hourly['dN'] = deepcopy(dN['binned'])
    N_hourly['dV/dlogD'] = deepcopy(dVdlogD['binned'])
    N_hourly['dN/dlogD'] = deepcopy(dNdlogD['binned'])
    N_hourly['Dv_lt2p5'] = deepcopy(dNdlogD['Dv_lt2p5'])
    N_hourly['Dn_lt2p5'] = deepcopy(dNdlogD['Dn_lt2p5'])
    N_hourly['Dv_2p5_10'] = deepcopy(dNdlogD['Dv_2p5_10'])
    N_hourly['Dn_2p5_10'] = deepcopy(dNdlogD['Dn_2p5_10'])



    # set up empty arrays for met data tp fill
    for met_key in ['RH', 'RH_frac', 'Tair', 'press']:
        N_hourly[met_key] = np.empty(len(dN['time']))
        N_hourly[met_key][:] = np.nan

    # keep average of met variables in a separate dictionary for later diagnostics
    met_vars_avg = {}
    for var in ['RH', 'RH_frac', 'Tair', 'press']:
        met_vars_avg[var] = np.empty(len(dN['time']))
        met_vars_avg[var][:] = np.nan

    met_skip_idx = 0

    for t, time_t in enumerate(dN['time'][:-1]):
        bool = np.logical_and(np.array(met_vars['time'][met_skip_idx:met_skip_idx+200]) > time_t,
                              np.array(met_vars['time'][met_skip_idx:met_skip_idx+200]) < time_t + dt.timedelta(hours=1))

        met_idx = np.where(bool == True)[0] + met_skip_idx

        for var in ['RH', 'RH_frac','Tair', 'press']:

            # average of variable for this time period, t
            var_i = np.nanmean(met_vars[var][met_idx])

            # store in N_hourly for picle saving
            N_hourly[var][t] = var_i

            # store met_vars_avg for diagnosis later
            met_vars_avg[var][t] = var_i

        if subsample == True:
            # only keep values that are less than the RH threshold by np.nan values above it
            # only keep values if there was no rain present during the period
            # do not remove the RH, Tair or pressure values
            if equate == 'lt':
                if (met_vars_avg['RH'][t] > RHthresh): #| (med_RR > 0.0):
                    N_hourly['dN'][t, :] = np.nan
                    N_hourly['dV/dlogD'][t, :] = np.nan
                    N_hourly['dN/dlogD'][t, :] = np.nan
                    N_hourly['Dv_lt2p5'][t] = np.nan
                    N_hourly['Dn_lt2p5'][t] = np.nan
                    N_hourly['Dv_2p5_10'][t] = np.nan
                    N_hourly['Dn_2p5_10'][t] = np.nan

            # only keep values that are greater than the threshold by np.nan values below it
            # only keep values if there was no rain present during the period
            elif equate == 'gt':
                if (met_vars_avg['RH'][t] < RHthresh): #| (med_RR > 0.0):
                    N_hourly['dN'][t, :] = np.nan
                    N_hourly['dV/dlogD'][t, :] = np.nan
                    N_hourly['dN/dlogD'][t, :] = np.nan
                    N_hourly['Dv_lt2p5'][t] = np.nan
                    N_hourly['Dn_lt2p5'][t] = np.nan
                    N_hourly['Dv_2p5_10'][t] = np.nan
                    N_hourly['Dn_2p5_10'][t] = np.nan

        # change skip idx to miss data already used in averaging
        #   so the entire time array doesn't need to be searched through for each iteration of t
        if met_idx.size != 0:
            met_skip_idx = met_idx[-1] + 1

    # save dN data in pickle form
    # save time, Dv and N as a pickle
    # with open(pickledir + 'dN_dmps_aps_clearfloWinter.pickle', 'wb') as handle:
    #     pickle.dump(dN, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if save == True:

        # change the savename, based on whether subsampling took place
        if subsample == True:
            samplesamplestr = '_'+equate+str(RHthresh)+'_cut'
        else:
            subsamplestr = ''

        # save the data
        # savestr will have the site name and instruments used in it
        # NOTE! DO NOT USE protocol=HIGHEST..., loading from python in linux does not work for some uknown reason...
        with open(pickledir + 'N_hourly_'+savestr+subsamplestr+'_'+extra+'.pickle', 'wb') as handle:
            pickle.dump(N_hourly, handle)

        print 'N_hourly_'+savestr+subsamplestr+'_'+extra+'.pickle'+'saved!'

    return N_hourly, met_vars

# plotting

def quick_plot_dV(N_hourly, N_hourly_50, dVdlogD, savestr, savedir, saveFile=False):

    """


    :param N_hourly:
    :param N_hourly_50:
    :param dVdlogD:
    :param args: can include savestr, savedir.
    :param save:
    :return:
    """

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
    fig = plt.figure(figsize=(6, 3.5))
    plt.semilogx(N_hourly['D']*1e-3, N_hourly['median']*1e-10, label=r'$RH < 60\%$', color='red')
    plt.semilogx(N_hourly['D']*1e-3, N_hourly_50['median']*1e-10, label=r'$RH < 50\%$', color='green')
    plt.semilogx(N_hourly['D']*1e-3, dVdlogD['median']*1e-10, label=r'$all \/\/data$', color='blue')
    #  plt.vlines(0.5, 0, 3, linestyle='--', alpha=0.5) # DMPS and APS (NK ClearfLo)?
    # plt.vlines(0.604, 0, 3, linestyle='--', alpha=0.5) # SMPS and GRIMM (Ch)
    plt.fill_between(N_hourly['D']*1e-3, N_hourly['25th']*1e-10, N_hourly['75th']*1e-10, alpha=0.2, facecolor='red', label=r'$RH < 60\% \/\/IQR$')
    plt.fill_between(N_hourly['D']*1e-3, N_hourly_50['25th']*1e-10, N_hourly_50['75th']*1e-10, alpha=0.2, facecolor='green', label=r'$RH < 50\% IQR$')
    plt.fill_between(N_hourly['D'] * 1e-3, dVdlogD['25th']*1e-10, dVdlogD['75th']*1e-10, alpha=0.2, facecolor='blue', label=r'$all \/\/data \/\/IQR$')
    plt.ylabel('dV/dlogD '+r'$[1^{10}\/nm^{3}\/ cm^{-3}]$', labelpad=0)
    plt.xlabel(r'$Diameter \/\/[\mu m]$', labelpad=-3)
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend(loc='best', fontsize='8')
    plt.tight_layout()
    if saveFile == True:
        plt.savefig(savedir + 'aerosol_distributions/dVdlogD_v_D_'+savestr+'_combined_below60_and50.png')
        plt.close(fig)

    return

def quick_plot_dN(N_hourly, N_hourly_50, dNdlogD, savestr, savedir):

    # median, IQRs
    N_hourly['median'] = np.nanmedian(N_hourly['dN/dlogD'], axis=0)
    N_hourly['25th'] = np.nanpercentile(N_hourly['dN/dlogD'], 25, axis=0)
    N_hourly['75th'] = np.nanpercentile(N_hourly['dN/dlogD'], 75, axis=0)

    N_hourly_50['median'] = np.nanmedian(N_hourly_50['dN/dlogD'], axis=0)
    N_hourly_50['25th'] = np.nanpercentile(N_hourly_50['dN/dlogD'], 25, axis=0)
    N_hourly_50['75th'] = np.nanpercentile(N_hourly_50['dN/dlogD'], 75, axis=0)

    # median, IQRs
    dNdlogD['median'] = np.nanmedian(dNdlogD['binned'], axis=0)
    dNdlogD['25th'] = np.nanpercentile(dNdlogD['binned'], 25, axis=0)
    dNdlogD['75th'] = np.nanpercentile(dNdlogD['binned'], 75, axis=0)

    # unit_conversion
    unit_conv = 1

    # plot volume distribution for data (median with IQR)
    # ok to use D from N_hourly as the bins are the same as the normal D variable
    fig = plt.figure(figsize=(5, 2.5))
    plt.semilogx(N_hourly['D']*1e-3, N_hourly['median']*unit_conv, label=r'$RH < 60\%$', color='red')
    plt.semilogx(N_hourly['D']*1e-3, N_hourly_50['median']*unit_conv, label=r'$RH < 50\%$', color='green')
    plt.semilogx(N_hourly['D']*1e-3, dNdlogD['median']*unit_conv, label=r'$all \/\/data$', color='blue')
    plt.vlines(0.604, 0, 4000, linestyle='--', alpha=0.5)
    plt.fill_between(N_hourly['D']*1e-3, N_hourly['25th']*unit_conv, N_hourly['75th']*unit_conv, alpha=0.2, facecolor='red', label=r'$RH < 60\% \/\/IQR$')
    plt.fill_between(N_hourly['D']*1e-3, N_hourly_50['25th']*unit_conv, N_hourly_50['75th']*unit_conv, alpha=0.2, facecolor='green', label=r'$RH < 50\% \/\/IQR$')
    plt.fill_between(N_hourly['D']*1e-3, dNdlogD['25th']*unit_conv, dNdlogD['75th']*unit_conv, alpha=0.2, facecolor='blue', label=r'$all \/\/data \/\/IQR$')
    plt.ylabel('dN/dlogD '+r'$[1e9 cm^{-3}]$', labelpad=0)
    plt.xlabel(r'$Diameter \/\/[\mu m]$', labelpad=-3)
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    # plt.xlim([0.01, 0.6])
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir + 'aerosol_distributions/dNdlogD_v_D_'+savestr+'.png')
    # plt.savefig(savedir + 'aerosol_distributions/dNdlogD_v_D_'+savestr+'_below60_and50.png')
    plt.close(fig)
    # plt.savefig(savedir + 'aerosol_distributions/rv_Claire_help2.png')

    return

def quick_plot_stdev_g_r(dNdlogD, D_min, D_max, site_meta, savestr):

    """
    Quick plot of the geometric standard deviation
    :param dNdlogD:
    :param D_min:
    :param D_max:
    :param site_meta:
    :param savestr: string with site name short and number distribution instrments used
    :return:
    """

    # plot histogram of r_g
    dataset = dNdlogD['stdev_g_r'][~np.isnan(dNdlogD['stdev_g_r'])]
    stdev_g_r_mean = np.nanmean(dNdlogD['stdev_g_r'])

    plt.hist(dataset, bins=20)
    date_range_str = dNdlogD['time'][0].strftime('%Y/%m/%d') + ' - ' + dNdlogD['time'][-1].strftime('%Y/%m/%d')
    size_range_radii_str = str(D_min/2.0) + '-' + str(D_max/2.0) + 'nm'
    plt.suptitle(site_meta['site_long'] + '; '+date_range_str + ';\n radii range: ' + size_range_radii_str + '; mean: ' + str(stdev_g_r_mean))
    plt.ylabel('frequency')
    plt.xlabel('geometric standard deviation')
    plt.savefig(savedir + 'geometric_stdev_mean/' + 'geo_stdev_r_'+ savestr + '_' + size_range_radii_str +'.png')
    plt.close()

    return

def quick_plot_r_g(dNdlogD, D_min, D_max, site_meta, savestr):

    """
    Quick plot the geometric mean radius

    :param dNdlogD:
    :param D_min:
    :param D_max:
    :return:
    """

    # plot histogram of r_g
    dataset = dNdlogD['r_g'][~np.isnan(dNdlogD['r_g'])]
    r_g_mean = np.nanmean(dNdlogD['r_g'])

    plt.hist(dataset, bins=20)
    date_range_str = dNdlogD['time'][0].strftime('%Y/%m/%d') + ' - ' + dNdlogD['time'][-1].strftime('%Y/%m/%d')
    size_range_radii_str = str(D_min/2.0) + '-' + str(D_max/2.0) + 'nm'
    plt.suptitle(site_meta['site_long'] + '; '+date_range_str + ';\n radii range: ' + size_range_radii_str + '; mean: ' + str(r_g_mean))
    plt.ylabel('frequency')
    plt.xlabel('geometric mean radius [nm]')
    plt.savefig(savedir + 'geometric_stdev_mean/' + 'geo_mean_r_' + savestr + '_' + size_range_radii_str +'.png')
    plt.close()

    return

if __name__ == '__main__':


    # ==============================================================================
    # Setup
    # ==============================================================================

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/'
    wxtdir = maindir + 'data/L1/'

    savedir = maindir + 'figures/number_concentration/'

    # data
    rhdatadir = maindir + 'data/L1/'
    pickledir = maindir + 'data/pickle/'

    # year of data - old
    years = ['2016']

    # site and instruments
    # period: 'routine' if the obs are routinely taken; 'long_term' if a long period is present; 'ClearfLo' if clearflo.
    # site_meta = {'site_short':'NK', 'site_long': 'North_Kensington', 'period': 'ClearfLo',
    #           'DMPS': True, 'APS': True, 'SMPS': False, 'GRIMM': False}

    # # NK: long term SMPS only
    # site_meta = {'site_short':'NK', 'site_long': 'North_Kensington', 'period': 'long_term'}
    # site_ins = {'DMPS': False, 'APS': False, 'SMPS': True, 'GRIMM': False}

    # NK: SMPS and APS
    site_meta = {'site_short':'NK', 'site_long': 'North_Kensington', 'period': 'long_term'}
    site_ins = {'DMPS': False, 'APS': True, 'SMPS': True, 'GRIMM': False}

    # site_meta = {'site_short':'Ch', 'site_long': 'Chilbolton', 'period': 'routine',
    #             'DMPS': False, 'APS': False, 'SMPS': True, 'GRIMM': True}


    # time resolution of output data in minutes
    timeRes = 60

    # save string
    savestr = site_meta['site_short'] + ''.join(['_'+i if site_ins[i] == True else '' for i in site_ins.iterkeys()])

    # ==============================================================================
    # Read met data
    # ==============================================================================

    if site_meta['period'] == 'ClearfLo':
        # RH data for ClearfLo
        site_rh = {'WXT_KSK': np.nan}
        rh_inst_site = site_rh.keys()[0]

        # read in RH data
        # get all RH filenames and store them in a list
        # add rain rate [RR] to those extracted
        rhdays = eu.date_range(dt.datetime(2012, 01, 9), dt.datetime(2012, 02, 8), 1, 'days')
        RHfilepaths = [rhdatadir + rh_inst_site+'_'+day.strftime('%Y%j')+'_1min.nc' for day in rhdays]
        RH = eu.netCDF_read(RHfilepaths, vars=['RH', 'RR', 'time'])
        RH['time'] -= dt.timedelta(minutes=1) # change time from 'obs end' to 'start of obs', same as the other datasets
        RH['RH'][RH['RH'] == -999] = np.nan # remove bad data with nans
        RH['RR'][RH['RR'] == -999] = np.nan  # remove bad data with nans

    if (site_meta['site_long'] == 'North_Kensington') & (site_meta['period'] == 'long_term'):

        # # one year
        # year = years[0] # read in just one years worth of data. Can always create a list of years if more is needed.
        # wxt_filepath = wxtdir + 'WXT_KSSW_'+year+'_15min.nc'
        # met_vars = eu.netCDF_read(wxt_filepath, vars=['RH', 'press', 'Tair', 'time'])
        # met_vars['RH_frac'] = met_vars['RH'] * 0.01

        # 2014 - 2016 inc.
        wxt_filepaths = [wxtdir + 'WXT_KSSW_'+year+'_15min.nc' for year in ['2014', '2015', '2016']]
        met_vars = eu.netCDF_read(wxt_filepaths, vars=['RH', 'press', 'Tair', 'time'])
        met_vars['RH_frac'] = met_vars['RH'] * 0.01


    # ==============================================================================
    # Read in number distribution data
    # ==============================================================================

    # ------------------------------------------
    # Smaller size ranges
    # ------------------------------------------

    ## DMPS
    if site_ins['DMPS'] == True:

        if site_meta['period'] == 'ClearfLo':

            # read in N data and extract relevent values
            # make N at all heights away from surface nan to be safe
            filepath = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/ClearfLo/' \
                       'man-dmps_n-kensington_20120114_r0.na'

            # read in the ClearfLo data
            dmps_dNdlogD, dmps_header_bins = read_dmps(filepath)

            # remove the overlapping bins in both data and the headers!
            # VERY specific for the DMPS when combining with APS data from ClearfLo
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
    ## SMPS
    if site_ins['SMPS'] == True:

        if site_meta['period'] == 'routine':

            # Read in SMPS data (assuming all SMPS files share the same sort of format)
            filepath_SMPS = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/NPL/' \
                           'SMPS_Size_'+site_meta['site_long']+'_Annual_Ratified_2016_v01.xls'

            data_frame = pandas.read_excel(filepath_SMPS, 'Data')

            # remove the first column (date) and last 2 cols (empty col and total)
            N_raw = np.asarray(data_frame)
            smps_N = {'Date': np.array([i.to_datetime() for i in N_raw[:, 0]]),
                      'binned': np.array(N_raw[:, 1:-2]),
                      'headers': np.array(list(data_frame)[1:-2])}

        # NK long SMPS dataset (.xls) 2007 - 2016 inclusively
        if (site_meta['period'] == 'long_term') & (site_meta['site_short'] == 'NK'):

            # Read in SMPS data (assuming all SMPS files share the same sort of format)
            filepath_SMPS = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/ERG/' \
                            'SMPSData_NK_2007-2016.xlsx'

            # years to read in
            years = [str(i) for i in range(2014, 2017)]

            print 'NK SMPS long-term read in...'

            for year in years:

                print 'year = '+ year

                data_frame = pandas.read_excel(filepath_SMPS, year)

                N_raw = np.asarray(data_frame)
                smps_N_year = {'time': np.array([i.to_datetime() for i in N_raw[:, 0]]),
                              'binned': np.array(N_raw[:, 1:-2]),
                              'headers': np.array(list(data_frame)[1:-2])}

                # store data
                if year == years[0]:
                    smps_N = deepcopy(smps_N_year)
                else:

                    # update smps_N with data for this year
                    smps_N['binned'] = np.vstack((smps_N['binned'], smps_N_year['binned']))
                    smps_N['time'] = np.append(smps_N['time'], smps_N_year['time'])
                    smps_N['headers'] = smps_N_year['headers']


        # get bin parameters
        smps_N = calc_bin_parameters_smps(smps_N, units='nm')

        # calc dNdlogD from N, for aps data
        smps_dNdlogD = calc_dNdlogD_from_N_smps(smps_N)

    # ------------------------------------------
    # Larger size ranges
    # ------------------------------------------

    ## APS
    if site_ins['APS'] == True:

        if site_meta['period'] == 'ClearfLo':

            # Read in aps data (larger radii: 0.5 to 20 microns)
            filepath_aps = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/ClearfLo/' \
                           'man-aps_n-kensington_20120110_r0.na'

            # read in aps data (dictionary format)
            # header bins will be lower bound of the bin, not the bin mid! (header_bins != aps_D)
            aps_N, aps_header_bins = read_clearflo_aps(filepath_aps)

            # calculate bin parameters # OLD WAY with headers being strings of the header bins e.g. '10.4', '11.8' ...
            # The headers correspond to the min bin edge. Therefore, the header along is the bin max.
            aps_D, aps_dD, aps_logD, aps_dlogD = calc_bin_parameters_aps_clearflo(aps_N, units='nm')

            # calc dNdlogD from N, for aps data (coarse)
            aps_dNdlogD = calc_dNdlogD_from_N_aps(aps_N, aps_D, aps_dlogD, aps_header_bins)

            # calc dNdlogD from N, for aps data (coarse)
            aps_dNdlogD = calc_dNdlogD_from_N_aps(aps_N, aps_D, aps_dlogD, aps_header_bins)


        if (site_meta['period'] == 'long_term') & (site_meta['site_short'] == 'NK'):

            # main data [# cm-3]
            filepath_aps = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/ERG/' \
                           'APS data NK 2014-2017.csv'

            # read in the data
            aps_N = read_routine_aps(filepath_aps)

            # calculate bin parameters
            # The headers correspond to the min bin edge. Therefore, the header along is the bin max.
            aps_N = calc_bin_parameters_general(aps_N, units='nm')

            # calc dNdlogD from N, for aps data (coarse)
            aps_dNdlogD = calc_dNdlogD_from_N_general(aps_N)


    # -----------------------------------------------------------------------
    ## GRIMM
    if site_ins['GRIMM'] == True:
        # # Read in the GRIMM EDM data
        # # 2016 data only available from months 01 - 09 inclusively
        grimm_N, met_vars = read_grimm_and_met_vars(maindir, year)

        # get bin parameters and add them to the main dictionary
        grimm_N = calc_bin_parameters_grimm(grimm_N, units='nm')

        # calc dNdlogD from N, for grimm data
        grimm_dNdlogD = calc_dNdlogD_from_N_grimm(grimm_N)

    # ==============================================================================
    # Merge datasets
    # ==============================================================================
    if (site_ins['SMPS'] == True) & (site_ins['GRIMM'] == True):
        # merge the dmps and APS dNdlogD datasets together...
        # set up so data resolution is the same
        # grimm D bins gets trimmed so bring it back out
        dNdlogD, grimm_dNlogD, smps_D_idx, grimm_D_idx = merge_smps_grimm_dNdlogD(smps_dNdlogD, grimm_dNdlogD, timeRes=timeRes)

        D = dNdlogD['D']
        dD = dNdlogD['dD']
        dlogD = dNdlogD['dlogD']
        logD = dNdlogD['logD']

        # extra save str for figures
        savestr = site_meta['site_short'] + '_SMPS_GRIMM'

    # -----------------------------------------------------------------------
    if (site_ins['DMPS'] == True) & (site_ins['APS'] == True):
        # merge the dmps and APS dNdlogD datasets together...
        # set up so data resolution is 15 mins
        dNdlogD = merge_dmps_aps_dNdlogD(dmps_dNdlogD, dmps_D, aps_dNdlogD, aps_D, timeRes=timeRes)

        # merge the aerosol parameters together too
        D = np.append(dmps_D, aps_D)
        logD = np.append(dmps_logD, aps_logD)
        dD = np.append(dmps_dD, aps_dD)
        dlogD = np.append(dmps_dlogD, aps_dlogD)

    # -----------------------------------------------------------------------
    # if only SMPS is set True
    if (site_ins['SMPS'] == True) & (np.sum(site_ins.values()) == 1):

        # As time resolution is already in 15 mins, create a 'Time' entry from 'Date'
        dNdlogD = smps_dNdlogD
        dNdlogD['time'] = smps_N['Date']

        D = smps_N['D']
        logD = smps_N['logD']
        dD = smps_N['dD']
        dlogD = smps_N['dlogD']

    # -----------------------------------------------------------------------
    # e.g. NK long term data 2014 - 2016 inc.
    if (site_ins['SMPS'] == True) & (site_ins['APS'] == True):

        # delete overlapping bins and merge
        dNdlogD, aps_dNdlogD_2, small_D_idx, large_D_idx = merge_small_large_dNdlogD(smps_dNdlogD, aps_dNdlogD, timeRes=timeRes)

        D = dNdlogD['D']
        logD = dNdlogD['logD']
        dD = dNdlogD['dD']
        dlogD = dNdlogD['dlogD']

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
    dNdlogD['Dn_accum'] = np.empty(len(dNdlogD['time']))
    dNdlogD['Dn_accum'][:] = np.nan # 2nd part of eqn for Dv

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

    # Calculate the volume and number mean diameters for 0 - 2.5 micron range (Dv and Dn respectively)
    # Get size range
    D_min = 0.0 # use lowest value in D as the lower limit
    D_max = 2500.0 # 2.5 microns
    size_range_idx = get_size_range_idx(D, D_min, D_max)

    dNdlogD['Dv_lt2p5'], dNdlogD['Dn_lt2p5'] = calc_volume_and_number_mean_diameter(size_range_idx, dNdD, dNdlogD, dlogD, D)

    # calculate the volume and number mean diameters for the 2.5 - 10 micron (Dv and Dn respectively)
    D_min = 2500.0 # 2.5 microns
    D_max = 10000.0 # 10 microns
    dNdlogD['Dv_2p5_10'], dNdlogD['Dn_2p5_10'] = calc_volume_and_number_mean_diameter(size_range_idx, dNdD, dNdlogD, dlogD, D)

    # calculate the volume and number mean diameters for below 10 micron (Dv and Dn respectively)
    D_min = 0.0 # 0 microns
    D_max = 10000.0 # 10 microns
    dNdlogD['Dv_10'], dNdlogD['Dn_10'] = calc_volume_and_number_mean_diameter(size_range_idx, dNdD, dNdlogD, dlogD, D)


    # extract out only aerosol data based on the RH threshold and if it is less or more than it.
    # Necessary if the measurements were taken at ambient RH and not dried before measureing
    N_hourly, met_avg = hourly_rh_threshold_pickle_save_fast(dN, dVdlogD, dNdlogD, met_vars, D, dD, pickledir, savestr,
                                    small_D_idx, large_D_idx, subsample=False, save=True, extra='20140101-20161231')

    # N_hourly_50, _ = hourly_rh_threshold_pickle_save_fast(dN, dVdlogD, dNdlogD, met_vars, D, dD, pickledir, savestr,
    #                                 smps_D_idx, grimm_D_idx, subsample=True, RHthresh=50.0, equate='lt', save=False)

    # # # quickplot what it looks like
    # # quick_plot_dV(N_hourly, N_hourly_50, dVdlogD, savestr, savedir)
    # quick_plot_dV(N_hourly, N_hourly_50, dVdlogD, savestr, savedir, saveFile=False)
    # quick_plot_dN(N_hourly, N_hourly_50, dNdlogD, savestr, savedir)

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
    # Find Dv, Dn and N for the accum. range, for the aerFo
    # ==============================================================================

    # Dv = mean volume diameter
    # Dn = mean number diameter

    # NOTE! D is in nm
    # calculate the volume and number mean diameters for the given diameter range (Dv and Dn respectively)
    D_min = 40.0
    D_max = 700.0

    # get the idx for the accumulation range
    accum_range_idx = get_size_range_idx(D, D_min, D_max)

    # calculate the mean volume and mean number diameter
    dNdlogD['Dv_accum'], dNdlogD['Dn_accum'] = calc_volume_and_number_mean_diameter(accum_range_idx, dNdD, dNdlogD, dlogD, D)

    # calculate the geometric mean and geometric standard deviation of the radius
    dNdlogD['r_g'], dNdlogD['stdev_g_r'] = calc_geometric_mean_and_stdev_radius(accum_range_idx, dNdD, dNdlogD, dlogD, D, units='nm')

    # statistics and plots on r_g and stdev_g_r
    quick_plot_r_g(dNdlogD, D_min, D_max, site_meta, savestr)
    quick_plot_stdev_g_r(dNdlogD, D_min, D_max, site_meta, savestr)

    # np.nanmean(dNdlogD['r_g'])
    # np.nanmean(dNdlogD['stdev_g_r'])

    plt.scatter(dNdlogD['r_g'][~np.isnan(dNdlogD['r_g'])], dNdlogD['stdev_g_r'][~np.isnan(dNdlogD['stdev_g_r'])])

    # plot time series of r for defined acumm range
    fig = plt.figure()
    # plt.plot((dVdD['Dv'] * 1e-3) / 2, label='rv using dN/dD')
    plt.plot_date(dNdlogD['time'], (dNdlogD['Dn_accum'] * 1e-3) / 2, label='rv using dN/dlogD', linestyle='-', fmt='-')
    plt.ylabel('rv [microns]')
    plt.xlabel('Date [dd/mm]')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    # plt.legend()
    # plt.savefig(savedir + 'aerosol_distributions/rv_accum_0p02_0p7.png')
    # plt.close(fig)

    # # Get total number of particles (eqn 8.9)
    # for t in range(len(dNdD['time'])):
    #
    #     dNdlogD['Ntot_accum'][t] = np.sum(dNdlogD['binned'][t, accum_range_idx] * dlogD[accum_range_idx])
    #     dNdlogD['Ntot'][t] = np.sum(dNdlogD['binned'][t, :] * dlogD)
    #
    # np.nanmean(dNdlogD['Ntot_accum'])
    #

    #
    # # save time, Dv and N as a pickle
    # pickle_save = {'time': dNdlogD['time'], 'Ntot': dNdlogD['Ntot_accum'], 'Dv': dNdlogD['Dv_accum']}
    # with open(pickledir + 'accum_Ntot_Dv_'+savestr+'.pickle', 'wb') as handle:
    #     pickle.dump(pickle_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    plt.savefig(savedir + 'aerosol_distributions/RH_lineplot_'+savestr+'.png')
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
    plt.savefig(savedir + 'aerosol_distributions/RH_pdfs_'+str(thresh)+'thresh_'+savestr+'.png')
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
    plt.savefig(savedir + 'aerosol_distributions/dVdlogD_v_D_.png')
    '+savestr+'
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
    plt.vlines(604, 0, np.nanmax(dNdlogD['25th']), linestyle='--', alpha=0.5)
    plt.fill_between(D, dNdlogD['25th'], dNdlogD['75th'], alpha=0.5, facecolor='blue', label='IQR')
    plt.ylabel('dN/dlogD [cm-3]')
    plt.xlabel('D [nm]')
    # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir + 'aerosol_distributions/dNdlogD_v_D_'+savestr+'.png')
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


print 'END PROGRAM'