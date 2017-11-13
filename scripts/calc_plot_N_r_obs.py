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
    daystrList = ['20150414', '20150415', '20150421', '20150611', '20160504']
    # daystrList = ['20150415']

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

        dNdlogD_raw = np.genfromtxt(filepath, delimiter=' ', skip_header=74, dtype=float, names=True)
        # N_raw = np.genfromtxt(filepath, delimiter=',', skip_header=75, dtype=float, names=True)

        # work around with the headers
        # VERY specific use with the TSI APS from NK during clearfLo
        headers = read_headers(filepath)

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

        # -----------------------------------------------------------------------
        # 2. calculate bin centres (D_mid)

        # estimate bin windths as only the centre diameters have been given
        # keep D_keys as a list of bins later on, to calculate n_v(log D)
        D_keys = dNdlogD.keys()
        del D_keys[D_keys.index('time')]  # remove time so we just get bin widths
        del D_keys[D_keys.index('rawtime')] # remove time so we just get bin widths
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

        # # check mid, widths start and ends are ok
        # for i in range(len(widths_end)):
        #     print str(widths_start[i]) + ' to ' + str(widths_end[i]) + ': mid = ' + str(D_mid[i]) + '; width = ' + str(widths[i])

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
        #     for key, dlogD_i in zip(D_keys, dlogD):
        #         # total_dN += [dNdD[key][t]]  # turn dN/dD into dN (hopefully)
        #         # total_dV += [dVdD[key][t]]  # just dV/dD data
        #         total_dN += [dNdlogD[key][t] * dlogD_i * (D_i ** 4.0)]  # turn dN/dD into dN (hopefully)
        #         total_dV += [dVdD[key][t]]  # just dV/dD data
        #
        #     dNdD['sum'][t] = np.sum(total_dN)  # Ntotal for time(hopefully)
        #     dVdD['sum'][t] = np.sum(total_dV)  # sum of dv/dD for all size bins for later

        # calculate volume mean diameter
        for t in range(len(dVdD['time'])):
            # dVdD['Dv'][t] = (dVdD['sum'][t] / (dNdD['sum'][t] * (np.pi / 6.0))) ** 1. / 3.
            dVdD['Dv'][t] = (dNdD['sum'][t] / (dNdD['sum'][t] * (np.pi / 6.0))) ** 1. / 3.

        fig = plt.figure()
        # plt.plot((dVdD['Dv'] * 1e-3) / 2, label='rv using dN/dD')
        plt.plot((dNdlogD['Dv'] * 1e-3) / 2, label='rv using dN/dlogD')
        plt.ylabel('microns')
        # plt.plot(Dv_logD_data['Dv'], label='using Nv(logD)')
        plt.legend()
        plt.savefig(savedir + 'aerosol_distributions/rv_Ben_help.png')

    return

if __name__ == '__main__':
    main()

print 'END PROGRAM'