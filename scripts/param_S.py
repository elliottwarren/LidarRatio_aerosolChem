"""
Parameterise the lidar ratio using linear fit, polyfits and LOWESS

Created by Elliott Warren Fri 07 Sept 2018
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import datetime as dt
from dateutil import tz

import ellUtils as eu
from forward_operator import FOconstants as FOcon

# import lowess from the api
lowess = sm.nonparametric.lowess

if __name__ == '__main__':

    # -----------------------------
    # Setup
    # -----------------------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = maindir + '/data/npy/S_timeseries/'
    savedir = maindir + 'figures/S_plots/'


    # year of data
    # for NK - only got 2014 and 2015 with APS data
    years = ['2014']
    year = years[0]

    ceil_lam = 905
    ceil_lambda_str = str(ceil_lam) + 'nm'

    # -----------------------------
    # Read
    # -----------------------------

    filename1 = 'NK_SMPS_APS_PM10_withSoot_2014_905nm.npy'
    filename2 = 'NK_SMPS_APS_PM10_withSoot_2015_905nm.npy'
    data1 = np.load(datadir + filename1).flat[0]
    data2 = np.load(datadir + filename2).flat[0]

    # optics = data['optics']
    S = np.append(data1['optics']['S'], data2['optics']['S'])
    RH = np.append(data1['met']['RH'], data2['met']['RH'])
    CBLK_N_weight_pm10 = np.append(data1['N_weight']['CBLK'], data2['N_weight']['CBLK'])
    # met = data['met']
    # dN = data['dN']
    # N_weight_pm10 = data['N_weight']
    # pm10_mass = data['pm10_mass']
    # time = data['met']['time']

    # -----------------------------
    # Plotting
    # -----------------------------

    # data for fitting
    x = np.array(RH)
    y = np.array(S)
    idx = np.isfinite(x) & np.isfinite(y)
    x_plot_range = np.nanmin(x[idx]),np.nanmax(x[idx])+1.0 # range to plot over

    # linear fit
    lin_coeffs = np.polyfit(x[idx], y[idx], 1)
    lin_f = np.poly1d(lin_coeffs)
    lin_fit = lin_f(x_plot_range)

    # cubic
    cubic_coeffs = np.polyfit(x[idx], y[idx], 3)
    cubic_f = np.poly1d(cubic_coeffs)
    cubic_fit = cubic_f(x_plot_range)

    # LOWESS
    it = 3
    frac = 0.1

    # NOTE: lowess(y,then x), opposite way round for some reason...
    lowess_fit = lowess(y, x, frac=frac, it=it)

    # SCATTER - S vs RH
    fig, ax = plt.subplots(1,1,figsize=(8, 4))
    key = 'CBLK'
    scat = ax.scatter(RH, S, c=CBLK_N_weight_pm10*100.0, vmin= 0.0, vmax = 25.0,
                      s=15, edgecolors='black', linewidths=0.5)
    # plot the fits
    plt.plot(x_plot_range, lin_fit, label='linear fit')
    plt.plot(x_plot_range, cubic_fit, label='cubic fit')
    plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], '--', color='red', alpha=1.0, label='LOWESS (frac=0.1)')

    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label('[%]', labelpad=-20, y=1.1, rotation=0)
    plt.xlabel(r'$RH \/[\%]$')
    plt.ylabel(r'$Lidar Ratio \/[sr]$')
    plt.ylim([10.0, 90.0])
    plt.xlim([20.0, 100.0])
    plt.tight_layout()
    plt.suptitle(ceil_lambda_str)
    plt.legend(loc='best')
    plt.savefig(savedir + 'S_vs_RH_NK_2014-2015_'+key+'_'+ceil_lambda_str+'_multipleFits.png')
    # plt.close(fig)











    print 'END PROGRAM'


