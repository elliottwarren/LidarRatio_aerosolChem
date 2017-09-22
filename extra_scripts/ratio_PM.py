"""
Read in and plot PM10/PM2.5 for an LAQN site in London

Created by Elliott Fri 22/09/2017
"""

import numpy as np
import ellUtils as eu
import datetime as dt
import matplotlib.pyplot as plt

def main():

    # -------------
    # Setup
    # -------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = maindir + 'data/LAQN/'
    savedir = maindir + 'figures/extras/PMratio/'

    # -------------
    # Read
    # -------------

    # get filename
    pm10fname = 'PM10_MR_15min_20161005-20170917.csv'
    pm10filepath = datadir + pm10fname

    pm2p5fname = 'PM2p5_MR_15min_20161005-20170917.csv'
    pm2p5filepath = datadir + pm2p5fname

    pm10rawData = np.genfromtxt(pm10filepath, delimiter=',', skip_header=1, dtype="|S20")
    pm2p5rawData = np.genfromtxt(pm2p5filepath, delimiter=',', skip_header=1, dtype="|S20")

    # process
    # some '' instances in the data so replace these with np.nan
    pm2p5 = {'PM2p5': np.array([np.nan if i[3] == '' else i[3] for i in pm2p5rawData], dtype=float),
             'time': np.array([dt.datetime.strptime(i[2], '%d/%m/%Y %H:%M') for i in pm2p5rawData]),
             'site': pm2p5rawData[0][0],
             'units': pm2p5rawData[0][4]}

    pm10 = {'PM10': np.array([np.nan if i[3] == '' else i[3] for i in pm10rawData], dtype=float),
            'time': np.array([dt.datetime.strptime(i[2], '%d/%m/%Y %H:%M') for i in pm10rawData]),
            'site': pm10rawData[0][0],
            'units': pm10rawData[0][4]}

    # check time lengths match and therefore assume times match fine
    if len(pm10['time']) != len(pm2p5['time']):
        raise ValueError('length of time in pm10 and pm2p5 differ!')

    pmRatio = np.array(pm10['PM10']/pm2p5['PM2p5'])

    # -------------
    # Plot
    # -------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot_date(pm10['time'], pmRatio, label='PM10/PM2.5', linestyle='-', fmt='-')
    ax1.set_ylabel(pm10['units'])
    ax1.legend()

    ax2.plot_date(pm10['time'], pm10['PM10'], label='PM10', linestyle='-', fmt='-')
    ax2.plot_date(pm10['time'], pm10['PM10'], label='PM2.5', linestyle='-', fmt='-')
    ax2.set_ylabel(pm10['units'])
    ax2.set_xlabel('Date [YYYY-MM]')
    ax2.legend()

    plt.subplots_adjust()

    plt.savefig(savedir + 'pmRatio_10v2p5.png')









    print 'END PROGRAM'

    return


if __name__ == '__main__':
    main()