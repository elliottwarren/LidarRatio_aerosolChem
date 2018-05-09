"""
Quickly scatter observed RH at different sites to diagnose KSSW.

Created by Elliott Warren - Wed 02 May 2018
"""


import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import ellUtils as eu

# ------------------------------
# Setup
# ------------------------------


# directories
maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/L1/'

savedir = maindir + 'figures/RH_compare/'

# year of data - old
year = 2014
year_str = str(year)

sites = {'KSSW': 'WXT', 'IML': 'Davis', 'BFCL': 'Davis'}

# time resolution to average up to in minutes
timeRes = 60

#----------------------
# Read
# ---------------------

data = {}

for site, ins in sites.iteritems():

    datapath = datadir + ins + '_' + site + '_' + year_str + '_15min.nc'

    data[site] = eu.netCDF_read(datapath)



# sort the times out so they all match up
start_time = dt.datetime(year, 1, 1)
end_time = dt.datetime(year, 12, 31)
time_range = eu.date_range(start_time, end_time, 60, 'minutes')

# set up time processed dictionary for all sites to be stored in
data_pro = {}

for site in sites.iterkeys():

    # get variables to average
    vars = ['RH', 'Tair']

    # set up dictionary entry for this site
    data_pro[site] = {'time': time_range}
    for var in vars:
        data_pro[site][var] = np.empty(len(time_range))
        data_pro[site][var][:] = np.nan

    # time fill
    for t, time_t in enumerate(time_range):

        bool = np.logical_and(data[site]['time'] >= time_t,
                              data[site]['time'] <  time_t + dt.timedelta(minutes=timeRes))

        for var in vars:
            data_pro[site][var][t] = np.nanmean(data[site][var][bool])


# calculate vapour pressure
for site in sites.iterkeys():
    e_s_hpa = 6.112 * (np.exp((17.67 * data_pro[site]['Tair']) / (data_pro[site]['Tair'] + 243.5))) # [hPa]
    e_s = e_s_hpa * 100.0 # [Pa]

    # coonvert RH to [fraction]
    RH_frac = data_pro[site]['RH'] / 100.0

    data_pro[site]['e'] = RH_frac * e_s # [Pa]


# -------------------
# Plotting
# -------------------

site_a = 'KSSW'
site_b = 'BFCL'
var = 'e'
units = '[Pa]'
# lims = [20, 100]
lims = [np.nanmin(np.append(data_pro[site_a]['e'], data_pro[site_b]['e'])),
        np.nanmax(np.append(data_pro[site_a]['e'], data_pro[site_b]['e']))]

plt.figure(figsize=(5,5))
plt.scatter(data_pro[site_a][var], data_pro[site_b][var], s=5, alpha=0.2)
plt.plot(lims, lims, '--', color='grey')
plt.xlabel(site_a+' ' + var + ' ' + units)
plt.ylabel(site_b+' ' + var + ' ' + units)
plt.suptitle(year_str)
plt.xlim(lims)
plt.ylim(lims)
plt.savefig(savedir + var + '_' + site_a +'_vs_'+site_b+'_'+year_str+'.png')


print 'END PROGRAM'