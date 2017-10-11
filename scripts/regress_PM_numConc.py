"""
Scatter and regress PM against number concentration at NK

Created by Elliott Wed 11 Oct 2017
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
    savedir = maindir + 'figures/numer_concentration/'

    # can be '15min','Hr', or 'daily'
    tres = 'Hr'

    # -------------
    # Read
    # -------------

    # get filenames
    # MR
    pm10fname = 'PM10_MR_'+tres+'_20161005-20170917.csv'
    pm10filepath = datadir + pm10fname

    pm2p5fname = 'PM2p5_MR_'+tres+'_20161005-20170917.csv'
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

    # create coarse/fine mode ratio
    pmRatio = np.array((pm10['PM10'] - pm2p5['PM2p5'])/ pm2p5['PM2p5'])
    pm_dates = np.array([i.date() for i in pm10['time']]) # would be the same as pm10 or pm2.5 time dates()

    # create unique datetime.dates (rounded to days)
    day_dates = eu.date_range(pm10['time'][0], pm10['time'][-1], 1, 'days') # datetimes
    day_dates = np.array([i.date() for i in day_dates]) # day dates from datetimes

    # create median and IQRs
    stats_ratio = {'median': [], 'q25': [], 'q75': [], 'mean': [], 'stddev': [], 'IQR': []}
    stats_pm2p5 = {'median': [], 'q25': [], 'q75': [], 'mean': [], 'stddev': [], 'IQR': []}
    stats_pm10 = {'median': [], 'q25': [], 'q75': [], 'mean': [], 'stddev': [], 'IQR': []}

    for stats_dir in [stats_ratio, stats_pm2p5, stats_pm10]:
        for key in stats_dir.iterkeys():
            stats_dir[key] = np.empty(len(day_dates))
            stats_dir[key][:] = np.nan

    # time match and calc stats
    # calc stats for pm ratio, pm2.5 and pm10 - zipping pairs up the right data to the right stats directory
    for d, day in enumerate(day_dates):
        for stats_dir, data in zip([stats_ratio, stats_pm2p5, stats_pm10], [pmRatio, pm2p5['PM2p5'], pm10['PM10']]):

            idx = np.where(pm_dates == day)[0]

            stats_dir['median'][d] = np.nanmedian(data[idx])
            stats_dir['q25'][d] = np.nanpercentile(data[idx], 25)
            stats_dir['q75'][d] = np.nanpercentile(data[idx], 75)
            stats_dir['IQR'][d] = stats_dir['q75'][d] - stats_dir['q25'][d]
            stats_dir['mean'][d] = np.nanmean(data[idx])
            stats_dir['stddev'][d] = np.nanstd(data[idx])


    # -------------
    # Plot
    # -------------

    # simple line plot

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.axhline(1.0, linestyle='--', alpha=0.5, xmin=0.0, xmax=1)
    ax1.plot_date(pm10['time'], pmRatio, label='(PM10-PM2.5)/PM2.5', linestyle='-', fmt='-')
    ax1.set_ylim([0.0, 3.0])
    ax1.set_xlim([pm10['time'][0], pm10['time'][-1]])
    ax1.set_ylabel(pm10['units'])
    ax1.legend()

    ax2.plot_date(pm10['time'], pm10['PM10'], label='PM10', linestyle='-', fmt='-')
    ax2.plot_date(pm2p5['time'], pm2p5['PM2p5'], label='PM2.5', linestyle='-', fmt='-')
    ax2.set_ylabel(pm10['units'])
    ax2.set_xlim([pm10['time'][0], pm10['time'][-1]])
    ax2.set_xlabel('Date [YYYY-MM]')
    ax2.legend()

    plt.subplots_adjust()
    plt.savefig(savedir + 'pmRatio_10v2p5_'+tres+'.png')
    plt.close(fig)


    # Median and IQR - line version

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.axhline(1.0, linestyle='--', alpha=0.5, xmin=0.0, xmax=1)

    ax1.fill_between(day_dates, stats_ratio['q25'], stats_ratio['q75'],
                    alpha=0.2, facecolor='blue')
    ax1.plot_date(day_dates, stats_ratio['median'], label='(PM10-PM2.5)/PM2.5', linestyle='-', fmt='-')

    ax1.set_ylim([0.0, 3.0])
    ax1.set_xlim([pm10['time'][0], pm10['time'][-1]])
    ax1.set_ylabel(pm10['units'])
    ax1.legend()

    ax2.fill_between(day_dates, stats_pm2p5['q25'], stats_pm2p5['q75'],
                    alpha=0.5, facecolor='orange')
    ax2.plot_date(day_dates, stats_pm2p5['median'], label='PM2.5', linestyle='-', fmt='-', color='orange')

    ax2.fill_between(day_dates, stats_pm10['q25'], stats_pm10['q75'],
                    alpha=0.2, facecolor='blue')
    ax2.plot_date(day_dates, stats_pm10['median'], label='PM10', linestyle='-', fmt='-', color='blue')

    # ax2.plot_date(day_dates, stats_pm2p5['median'], label='PM2.5', linestyle='-', fmt='-', color='orange')
    # ax2.plot_date(day_dates, stats_pm2p5['q25'], linestyle='--', fmt='-', color='orange')
    # ax2.plot_date(day_dates, stats_pm2p5['q75'], linestyle='--', fmt='-', color='orange')
    #
    # ax2.plot_date(day_dates, stats_pm10['median'], label='PM10', linestyle='-', fmt='-', color='blue')
    # ax2.plot_date(day_dates, stats_pm10['q25'], linestyle='--', fmt='-', color='blue')
    # ax2.plot_date(day_dates, stats_pm10['q75'], linestyle='--', fmt='-', color='blue')

    #ax2.plot_date(pm10['time'], pm10['PM10'], label='PM10', linestyle='-', fmt='-')
    #ax2.plot_date(pm2p5['time'], pm2p5['PM2p5'], label='PM2.5', linestyle='-', fmt='-')
    ax2.set_ylabel(pm10['units'])
    ax2.set_xlim([pm10['time'][0], pm10['time'][-1]])
    ax2.set_xlabel('Date [YYYY-MM]')
    ax2.legend()

    plt.subplots_adjust()
    plt.savefig(savedir + 'pmRatio_med_iqr_'+tres+'.png')
    plt.close(fig)






    print 'END PROGRAM'

    return


if __name__ == '__main__':
    main()