"""
Compare the climatology of f(RH) between sites

Created by Elliott Mon 05/03/2018
Based on monthly_f_RH_creation.py
"""

import numpy as np
import ellUtils as eu
import matplotlib.pyplot as plt
import pickle
import datetime as dt

# Reading


if __name__ == '__main__':

    # ------------------------------------------
    # Setup
    # ------------------------------------------
    # site information
    # site_ins = {'site_short': 'NK', 'site_long': 'North Kensington',
    #             'ceil_lambda': 0.905e-06, 'land-type': 'urban'}
    site_ins = {'ceil_lambda': 0.905e-06, 'land-type': 'rural'}
    # site_ins = {'site_short':'Ha', 'site_long': 'Harwell',
    #             'ceil_lambda': 0.905e-06, 'land-type': 'rural'}

    ceil_lambda_nm_str = str(site_ins['ceil_lambda'] * 1e9) + 'nm'

    # User set args
    # band that read_spec_bands() uses to find the correct band
    #! Manually set
    band = 1

    # saveF(RH)?
    saveFRH = True

    # -------------------------

    # directories
    savedir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/figures/Mie/daily_f(RH)/'
    datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/L1/'
    N_dir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/pickle/'
    fRHdir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/Mie/daily_f(RH)/'
    pickleloaddir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/Mie/pickle/'

    # date resolution
    dataRes = 'daily'

    # Plotting different months
    month_colours = {'blue': [1, 2, 12], 'green': [3, 4, 5], 'red': [6, 7, 8], 'orange': [9, 10, 11]}
    month_ls = {'-': [1, 4, 7, 10], '--': [2, 5, 8, 11], '-.': [12, 3, 6, 9]}

    year = 2015

    # ---------------------------------------------------
    # Read, Process and save f(RH)
    # ---------------------------------------------------

    # RH
    #RH_path = datadir + 'WXT_KSSW_'+str(year)+'_15min.nc'
    #RH_obs = eu.netCDF_read(RH_path)

    # N(D) #use the Dn_accum.
    N_path = N_dir + 'accum_Ntot_Dn_NK_SMPS_'+str(year)+'.npy'
    N = np.load(N_path).flat[0]
    r_accum_m = N['Dn_accum'] / 2 / 1e9 # conv. to [m] from [nm]

    # f(RH) read in
    f_RH_data = {}
    rel_vol_species = {}
    rel_vol_time = {}

    for site in ['NK', 'Ch', 'Ha']:

        # read in f(RH) LUTs for each site
        path = fRHdir + dataRes + '_f(RH)_' + site + '_905.0nm.nc'

        # eu.netCDF_info(path)

        rawdata = eu.netCDF_read(path)

        # .shape(time, 5 radii sample, RH)
        #   5 radii sample = np.array([0.07, 0.11, 0.15, 1.0, 3.0]), same as in the radii_range_micron variable [microns]
        f_RH_data[site] = rawdata['f(RH) MURK']

        if site == 'NK':
            time = rawdata['times']
            RH = rawdata['Relative Humidity']
            radii_range_nm = rawdata['radii_range_nm']
            radii_range_micron = radii_range_nm*1e-3

        # Read in relative volume of each species for each site
        filename = pickleloaddir + site + '_daily_aerosol_relative_volume.pickle'
        with open(filename, 'rb') as handle:
            pickle_load_in = pickle.load(handle)
        rel_vol_species[site] = pickle_load_in['pm10_rel_vol']
        rel_vol_time[site] = pickle_load_in['time']

    # append f(RH) and species together for the boxplotting - order of concatonation for both needs to be the same!
    # f_RH_all.shape(time, radii, RH)
    f_RH_all = np.concatenate((f_RH_data['NK'], f_RH_data['Ha'], f_RH_data['Ch']), axis=0)
    rel_vol_species_all = {}
    # rel_vol_species_all[species_i].shape(time)
    for species_i in rel_vol_species['NK'].iterkeys():
        rel_vol_species_all[species_i] = \
            np.concatenate((rel_vol_species['NK'][species_i],
                            rel_vol_species['Ha'][species_i],
                            rel_vol_species['Ch'][species_i]), axis=0)

    # quick plot hist to figure out how big to make each boxplot bin
    nan_bool = np.isnan(rel_vol_species_all['CORG'])
    data_hist = rel_vol_species_all['CORG'][~nan_bool]
    plt.hist(data_hist)
    lower_percentile = np.round(np.percentile(data_hist, 33) * 100.0)
    upper_percentile = np.round(np.percentile(data_hist, 66) * 100.0)

    # check sample size in each f_RH set
    n = {}
    for site in ['NK', 'Ch', 'Ha']:
        idx = np.where(~np.isnan(f_RH_data[site][:, 1, 0]))[0]
        n[site] = int(idx.shape[0])
    # ---------------------------------------------------
    # Plotting
    # ---------------------------------------------------

    # BOX PLOT - f(RH) binned by RH, then by soot

    for species_i in rel_vol_species_all.iterkeys():

        # figure out boxplot ranges for species based on percentiles
        nan_bool = np.isnan(rel_vol_species_all[species_i])
        data_hist = rel_vol_species_all[species_i][~nan_bool]
        # plt.hist(data_hist)
        lower_percentile = np.round(np.percentile(data_hist, 33) * 100.0) # [%]
        upper_percentile = np.round(np.percentile(data_hist, 66) * 100.0) # [%]

        if lower_percentile == 0.0:
            lower_percentile = 1.0

        for radii_idx, radii_i in enumerate(radii_range_micron):

            ## 1. set up bins to divide the data [%]
            rh_bin_starts = np.array([0.0, 60.0, 70.0, 80.0, 90.0])
            rh_bin_ends = np.append(rh_bin_starts[1:], 100.0)

            # # set up limit for species last bin to be inf [fraction] # OLD way for soot
            # species_starts = np.array([0.0, 0.04, 0.08])
            # species_ends = np.append(species_starts[1:], np.inf)
            # species_bins_num = len(species_starts)

            # set up limit for species last bin to be inf [fraction]
            species_starts = np.array([0.0, lower_percentile/100.0, upper_percentile/100.0])
            species_ends = np.append(species_starts[1:], np.inf)
            species_bins_num = len(species_starts)

            # variables to help plot the legend
            species_starts_str = [str(int(i*100.0)) for i in species_starts]
            species_ends_str = [str(int(i*100.0)) for i in species_ends[:-1]] + ['100']
            species_legend_str = [i+'-'+j+' %' for i, j in zip(species_starts_str, species_ends_str)]
            species_colours = ['blue', 'orange', 'red']

            # positions for each boxplot (1/6, 3/6, 5/6 into each bin, given 3 species groups)
            #   and widths for each boxplot
            pos = []
            widths = []
            mid = []

            for i, (rh_s, rh_e) in enumerate(zip(rh_bin_starts, rh_bin_ends)):

                bin_6th = (rh_e-rh_s) * 1.0/6.0 # 1/6th of the current bin width
                pos += [[rh_s + bin_6th, rh_s +(3*bin_6th), rh_s+(5*bin_6th)]] #1/6, 3/6, 5/6 into each bin for the species boxplots
                widths += [bin_6th]
                mid += [rh_s +(3*bin_6th)]


            # Split the data - keep them in lists to preserve the order when plotting
            # bin_range_str will match each set of lists in rh_binned
            rh_split = {'binned': [], 'mean': [], 'n': [], 'bin_range_str': [], 'pos': []}

            for i, (rh_s, rh_e) in enumerate(zip(rh_bin_starts, rh_bin_ends)):

                # bin range
                rh_split['bin_range_str'] += [str(int(rh_s)) + '-' + str(int(rh_e))]

                # the list of lists for this RH bin (the actual data, the mean and sample number)
                rh_bin_i = []
                rh_bin_mean_i = []
                rh_bin_n_i = []

                # # extract out all S values that occured for this RH range and their corresponding CBLK weights
                # rh_bool = np.logical_and(met['RH'] >= rh_s, met['RH'] < rh_e)
                # S_rh_i = S[rh_bool]
                # N_weight_cblk_rh_i = N_weight_pm10['CBLK'][rh_bool]

                # extract out data for this RH
                rh_bool = np.logical_and(RH * 100.0 >= rh_s, RH * 100.0 < rh_e)
                f_RH_i = f_RH_all[:, radii_idx, rh_bool]
                rel_vol_species_rh_i = rel_vol_species_all[species_i]

                # idx of binned data
                for species_s, species_e in zip(species_starts, species_ends):

                    # booleon for the species data, for this rh subsample
                    species_bool = np.logical_and(rel_vol_species_rh_i >= species_s, rel_vol_species_rh_i < species_e)
                    f_RH_i_species_j = f_RH_i[species_bool] # species subsample from the rh subsample


                    # store the values for this bin
                    rh_bin_i += [f_RH_i_species_j] # the of subsample
                    rh_bin_mean_i += [np.mean(f_RH_i_species_j)] # mean of of subsample
                    rh_bin_n_i += [len(f_RH_i_species_j)] # number of subsample

                # add each set of rh_bins onto the full set of rh_bins
                rh_split['binned'] += [rh_bin_i]
                rh_split['mean'] += [rh_bin_mean_i]
                rh_split['n'] += [rh_bin_n_i]


            ## 2. Start the boxplots
            # whis=[10, 90] wont work if the q1 or q3 extend beyond the whiskers... (the one bin with n=3...)
            fig = plt.figure(figsize=(7, 3.5))
            ax = plt.gca()
            # fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
            # plt.hold(True)
            for j, (rh_bin_j, bin_range_str_j) in enumerate(zip(rh_split['binned'], rh_split['bin_range_str'])):

                bp = plt.boxplot(list(rh_bin_j), widths=widths[j], positions=pos[j], whis=[5, 95], sym='x')

                # colour the boxplots
                for c, colour_c in enumerate(species_colours):

                    # some parts of the boxplots are in two parts (e.g. 2 caps for each boxplot) therefore make an x_idx
                    #   for each pair
                    c_pair_idx = range(2*c, (2*c)+(len(species_colours)-1))

                    plt.setp(bp['boxes'][c], color=colour_c)
                    plt.setp(bp['medians'][c], color=colour_c)
                    [plt.setp(bp['caps'][i], color=colour_c) for i in c_pair_idx]
                    [plt.setp(bp['whiskers'][i], color=colour_c) for i in c_pair_idx]
                    #[plt.setp(bp['fliers'][i], color=colour_c) for i in c_pair_idx]

            print 'test'
            # add sample number at the top of each box
            (y_min, y_max) = ax.get_ylim()
            upperLabels = [str(np.round(n, 2)) for n in np.hstack(rh_split['n'])]
            for tick in range(len(np.hstack(pos))):
                k = tick % 3
                ax.text(np.hstack(pos)[tick], y_max - (y_max * (0.05)*(k+1)), upperLabels[tick],
                         horizontalalignment='center', size='x-small')

            ## 3. Prettify boxplot (legend, vertical lines, sample size at top)
            # prettify
            ax.set_xlim([0.0, 100.0])
            ax.set_xticks(mid)
            ax.set_xticklabels(rh_split['bin_range_str'])
            ax.set_ylabel(r'$f(RH)$')
            ax.set_xlabel(r'$RH \/[\%]$')
            ax.set_yscale('log')
            plt.suptitle(species_i + ': NK, Ch, Ha; radii='+str(radii_range_micron[radii_idx])+' microns')

            # add vertical dashed lines to split the groups up
            (y_min, y_max) = ax.get_ylim()
            for rh_e in rh_bin_ends:
                plt.vlines(rh_e, y_min, y_max, alpha=0.3, color='grey', linestyle='--')

            # draw temporary lines to create a legend
            lin=[]
            for c, colour_c in enumerate(species_colours):
                # lin_i, = plt.plot([np.nanmean(S),np.nanmean(S)],color=colour_c) # plot line with matching colour
                lin_i, = plt.plot([1.0, 1.0],color=colour_c) # plot line with matching colour
                lin += [lin_i] # keep the line handle in a list for the legend plotting
            plt.legend(lin, species_legend_str, fontsize=10, loc=(0.03,0.68))
            [i.set_visible(False) for i in lin] # set the line to be invisible

            plt.tight_layout()
            plt.subplots_adjust(top=0.90)

            ## 4. Save fig as unique image
            savepath = savedir + 'combined_'+species_i+'_'+str(radii_range_micron[radii_idx])+'.png'

            plt.savefig(savepath)
            plt.close(fig)


    print 'END PROGRRAM'