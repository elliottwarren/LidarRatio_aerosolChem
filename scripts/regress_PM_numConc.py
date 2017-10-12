"""
Scatter and regress PM against number concentration at NK

Created by Elliott Wed 11 Oct 2017
"""

import numpy as np
import ellUtils as eu
import datetime as dt

from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.colors as colors


def main():


    # -------------
    # Setup
    # -------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = maindir + 'data/'
    savedir = maindir + 'figures/number_concentration/'

    # can be '15min','Hr', or 'daily'
    tres = 'Hr'

    site = 'NK'

    # -------------
    # Read
    # -------------

    # get filenames
    pm10fname = 'PM10_' + site + '_'+tres+'_20100101-20160731.csv'
    pm10filepath = datadir + 'LAQN/' + pm10fname

    pm2p5fname = 'PM2p5_' + site + '_'+tres+'_20100101-20160731.csv'
    pm2p5filepath = datadir + 'LAQN/' + pm2p5fname

    numConcfname = 'numConc_' + site + '_' + tres + '_20100101-20160731.csv'
    numConcfilepath = datadir + 'DEFRA/' + numConcfname

    pm10rawData = np.genfromtxt(pm10filepath, delimiter=',', skip_header=1, dtype="|S20")
    pm2p5rawData = np.genfromtxt(pm2p5filepath, delimiter=',', skip_header=1, dtype="|S20")
    numConcrawData = np.genfromtxt(numConcfilepath, delimiter=',', skip_header=5, dtype="|S20")

    # create better timestamp for numConc
    #   - replace instances of 24:00:00 with 00:00:00 as hour in datetime can only be 0-23.
    #   - join two columns together to make 1 timestamp string for each measurement
    #   - take an hour off each entry, as these timestamps relate to the END of the hour -> they need to
    #          match the PM data which is the start of the hour
    rawtime = [i[0] + ' ' + i[1].replace('24:00:00', '00:00:00') for i in numConcrawData]
    time_endHr = np.array([dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') for i in rawtime])
    time_strtHr = time_endHr - dt.timedelta(hours=1)

    # process
    # some '' instances in the data so replace these with np.nan
    pm2p5 = {'PM2p5': np.array([np.nan if i[3] == '' else i[3] for i in pm2p5rawData], dtype=float),
             'time': np.array([dt.datetime.strptime(i[2], '%d/%m/%Y %H:%M') for i in pm2p5rawData]),
             'site': pm2p5rawData[0][0],
             'units': pm2p5rawData[0][4]}

    pm10 = {'PM10': np.array([np.nan if i[3] == '' else i[3] for i in pm10rawData], dtype=float),
            'time': np.array([dt.datetime.strptime(i[2], '%d/%m/%Y %H:%M') for i in pm10rawData]),
            'site': pm10rawData[0][0],
            'units': 'ug m-3'}

    # ensure numConcrawtime is used to create numConc['time']
    numConc = {'numConc': np.array([np.nan if i[2] == 'No data' else i[2] for i in numConcrawData], dtype=float),
               'time': time_strtHr,
               'site': site,
               'units': 'cm-3'}

    # check time lengths match and therefore assume times match fine
    if (numConc['time'] - pm10['time']).max() != dt.timedelta(0):
        raise ValueError('length of time in pm10 and pm2p5 differ!')

    if (numConc['time'] - pm10['time']).max() != dt.timedelta(0):
        raise ValueError('numConc and pm10 times differ!')

    # create coarse/fine mode ratio
    # replace instances of Inf as a consequence of taking the ratio, with nan
    pmRatio = np.array((pm10['PM10'] - pm2p5['PM2p5'])/ pm2p5['PM2p5'])
    pm_dates = np.array([i.date() for i in pm10['time']]) # would be the same as pm10 or pm2.5 time dates()

    idx = np.where(np.isinf(pmRatio) == True)
    pmRatio[idx] = np.nan

    # -------------
    # Plot
    # -------------

    # x = pm2p5['PM2p5']
    x = pm10['PM10']
    # x = pmRatio
    y = numConc['numConc']

    corr = spearmanr(x, y)

    # remove nan pairings
    # find pairs first
    bool = np.logical_or(np.isnan(x),np.isnan(y))
    idx_good = np.where(bool == False)
    xy = np.vstack([x[idx_good], y[idx_good]])

    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # scatter

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    # ax1.axhline(1.0, linestyle='--', alpha=0.5, xmin=0.0, xmax=1)
    ax1.scatter(x, y, c=z, s=4, edgecolor='',
                            norm = colors.PowerNorm(gamma=1./2.))
    #                            norm = colors.PowerNorm(vmin = 1e-5, vmax = 2e-4,gamma=1./2.))
    ax1.set_ylim([0.0, 80000])
    ax1.set_xlim([-3, 100])
    ax1.set_ylabel('Number concentration ' + numConc['units'])
    ax1.set_xlabel('PM 10 ' + pm2p5['units'])
    plt.suptitle('PM 10: 01/01/2010-31/07/2016 hourly, @' + site + '\n'
                  'r=' + '{:1.2f}'.format(corr[0]))
    # ax1.legend()

    plt.subplots_adjust(left=0.2)
    plt.savefig(savedir + 'pm10_numConc_'+tres+'.png')
    plt.close(fig)


    print 'END PROGRAM'

    return


if __name__ == '__main__':
    main()