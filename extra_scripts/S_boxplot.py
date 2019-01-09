"""
Quickly plot the S as boxplots using this version of matplotlib

"""

import matplotlib.pyplot as plt

import numpy as np

import os

maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
savedir = maindir + 'figures/S_plots/'
npydir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/VBSharedFolder/temp_trans/'

wave = '355nm'

# Read ------------------------------------------------------------------------------------------------

# load in previously calculated S data
# filename = pickledir+ 'NK_SMPS_APS_PM10_withSoot_'+year_str+'_'+ceil_lambda_str+'_hysteresis_shapecorr.npy'
filename = npydir+'NK_SMPS_APS_PM10_withSoot_2015_'+wave+'_hysteresis_shapecorr.npy'
npy_load_in = np.load(filename).flat[0]

S = npy_load_in['optics']['S']
met = npy_load_in['met']
N_weight_pm10 = npy_load_in['N_weight']

# # Read ------------------------------------------------------------------------------------------------
#
# # load in previously calculated S data
# # filename = pickledir+ 'NK_SMPS_APS_PM10_withSoot_'+year_str+'_'+ceil_lambda_str+'_hysteresis_shapecorr.npy'
# filename = npydir+'NK_SMPS_APS_PM10_withSoot_2014_905nm_hysteresis_shapecorr.npy'
# filename2 = npydir+'NK_SMPS_APS_PM10_withSoot_2015_905nm_hysteresis_shapecorr.npy'
# npy_load_in = np.load(filename).flat[0]
# npy_load_in2 = np.load(filename2).flat[0]
#
# # optics = npy_load_in['optics']
# # S = optics['S']
#
# S = np.append(npy_load_in['optics']['S'], npy_load_in['optics']['S'])
#
# # met = npy_load_in['met']
# met = {key: np.append(npy_load_in['met'][key], npy_load_in2['met'][key]) for key in npy_load_in['met'].iterkeys()}
# N_weight_pm10 = {key: np.append(npy_load_in['N_weight'][key], npy_load_in2['N_weight'][key]) for key in
#                  npy_load_in['N_weight'].iterkeys()}

# plot ------------------------------------------------------------------------------------------------

# BOX PLOT - S binned by RH, then by soot

# ## 1. set up bins to divide the data [%] # old
# rh_bin_starts = np.array([0.0, 60.0, 70.0, 80.0, 90.0])
# rh_bin_ends = np.append(rh_bin_starts[1:], 100.0)

## 1. set up bins to divide the data [%]
rh_bin_starts = np.array([0.0, 60.0, 70.0, 80.0])
rh_bin_ends = np.append(rh_bin_starts[1:], 90.0)

# set up limit for soot last bin to be inf [fraction]
soot_starts = np.array([0.0, 0.08, 0.16])
soot_ends = np.append(soot_starts[1:], np.inf)
soot_bins_num = len(soot_starts)

# variables to help plot the legend
soot_starts_str = [str(int(i * 100.0)) for i in soot_starts]
soot_ends_str = [str(int(i * 100.0)) for i in soot_ends[:-1]] + ['100']
soot_legend_str = [i + '-' + j + ' %' for i, j in zip(soot_starts_str, soot_ends_str)]
soot_colours = ['blue', 'orange', 'red']

# positions for each boxplot (1/6, 3/6, 5/6 into each bin, given 3 soot groups)
#   and widths for each boxplot
pos = []
widths = []
mid = []

for i, (rh_s, rh_e) in enumerate(zip(rh_bin_starts, rh_bin_ends)):
    bin_6th = (rh_e - rh_s) * 1.0 / 6.0  # 1/6th of the current bin width
    pos += [[rh_s + bin_6th, rh_s + (3 * bin_6th),
             rh_s + (5 * bin_6th)]]  # 1/6, 3/6, 5/6 into each bin for the soot boxplots
    widths += [bin_6th]
    mid += [rh_s + (3 * bin_6th)]

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

    # extract out all S values that occured for this RH range and their corresponding CBLK weights
    rh_bool = np.logical_and(met['RH'] >= rh_s, met['RH'] < rh_e)
    S_rh_i = S[rh_bool]
    # N_weight_cblk_rh_i = N_weight_pm10['CBLK'][rh_bool]
    N_weight_cblk_rh_i = N_weight_pm10['CBLK'][rh_bool]

    # idx of binned data
    for soot_s, soot_e in zip(soot_starts, soot_ends):
        # booleon for the soot data, for this rh subsample
        soot_bool = np.logical_and(N_weight_cblk_rh_i >= soot_s, N_weight_cblk_rh_i < soot_e)
        S_rh_i_soot_j = S_rh_i[soot_bool]  # soot subsample from the rh subsample
        # remove nans
        S_rh_i_soot_j = S_rh_i_soot_j[~np.isnan(S_rh_i_soot_j)]

        # store the values for this bin
        rh_bin_i += [S_rh_i_soot_j]  # the of subsample
        rh_bin_mean_i += [np.mean(S_rh_i_soot_j)]  # mean of of subsample
        rh_bin_n_i += [len(S_rh_i_soot_j)]  # number of subsample

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

    bp = plt.boxplot(list(rh_bin_j), widths=widths[j], positions=pos[j], whis=[5, 95], sym='')

    # colour the boxplots
    for c, colour_c in enumerate(soot_colours):
        # some parts of the boxplots are in two parts (e.g. 2 caps for each boxplot) therefore make an x_idx
        #   for each pair
        c_pair_idx = range(2 * c, (2 * c) + (len(soot_colours) - 1))

        plt.setp(bp['boxes'][c], color=colour_c)
        plt.setp(bp['medians'][c], color=colour_c)
        [plt.setp(bp['caps'][i], color=colour_c) for i in c_pair_idx]
        [plt.setp(bp['whiskers'][i], color=colour_c) for i in c_pair_idx]
        # [plt.setp(bp['fliers'][i], color=colour_c) for i in c_pair_idx]

# # add sample number at the top of each box
# (y_min, y_max) = ax.get_ylim()
# upperLabels = [str(np.round(n, 2)) for n in np.hstack(rh_split['n'])]
# for tick in range(len(np.hstack(pos))):
#     k = tick % 3
#     ax.text(np.hstack(pos)[tick], y_max - (y_max * (0.05) * (k + 1)), upperLabels[tick],
#             horizontalalignment='center', size='x-small')


## 3. Prettify boxplot (legend, vertical lines, sample size at top)
# prettify
ax.set_xlim([0.0, 90.0])
ax.set_xticks(mid)
ax.set_xticklabels(rh_split['bin_range_str'])
ax.set_ylabel(r'$S \/[sr]$')
ax.set_xlabel(r'$RH \/[\%]$')
ax.set_ylim([15, 80])

# add sample number at below each box
(y_min, y_max) = ax.get_ylim()
upperLabels = [str(np.round(n, 2)) for n in np.hstack(rh_split['n'])]
for tick in range(len(np.hstack(pos))):
    k = tick % 3
    ax.text(np.hstack(pos)[tick], y_min + (y_min * (0.08) * (k + 1)), upperLabels[tick],
            horizontalalignment='center', size='x-small')

# add vertical dashed lines to split the groups up
(y_min, y_max) = ax.get_ylim()
for rh_e in rh_bin_ends:
    plt.vlines(rh_e, y_min, y_max, alpha=0.3, color='grey', linestyle='--')

# draw temporary lines to create a legend
lin = []
for c, colour_c in enumerate(soot_colours):
    lin_i, = plt.plot([np.nanmean(S), np.nanmean(S)], color=colour_c)  # plot line with matching colour
    lin += [lin_i]  # keep the line handle in a list for the legend plotting
plt.legend(lin, soot_legend_str, fontsize=10, loc=(0.02, 0.68))
[i.set_visible(False) for i in lin]  # set the line to be invisible

plt.tight_layout()

## 4. Save fig as unique image
i = 1
savepath = savedir + 'S_vs_RH_binnedSoot_2015_NK_boxplot_'+wave+'_' + str(
    i) + '.png'
while os.path.exists(savepath) == True:
    i += 1
    savepath = savedir + 'S_vs_RH_binnedSoot_2015_NK_boxplot_'+wave+'_' + str(
        i) + '.png'

plt.savefig(savepath)