"""
Plot the aerFO runs together. aerFO runs are done in beta_m_from_mod_and_aerObs.py

Created by Elliott Wed 30 May 2018
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import spearmanr
from scipy.stats import pearsonr

import numpy as np

if __name__ == '__main__':

    # ---------------------------
    # Setup
    # ---------------------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/'
    savedir = maindir + 'figures/model_eval/summary_boxplots/'
    npydir = datadir + 'npy/model_eval/'

    # use this to retain the plotting order
    run_plot_number = [str(i) for i in range(0, 9)]
    run_plot_position = range(1, 10)
    run_plot_order = ['run'+i for i in run_plot_number]

    # --------------------------
    # Read
    # --------------------------

    # NK
    NK_run_files = {'run0': 'NK_CTRL_run0.npy',
                    'run1': 'NK_pm10aerinput_run1.npy',
                    'run2': 'NK_obs_N_obs_r_accum_run2.npy',
                    'run3': 'NK_obs_N_obs_r_fine_accum_run3.npy',
                    'run4': 'NK_obs_N_obs_r_fine_accum_coarse_run4.npy',
                    'run5': 'NK_pm10aerinput_Q_extdry_fRH_hourly_obs_run5.npy',
                    'run6': 'NK_obs_N_obs_r_accum-S_hourly_run6.npy',
                    'run7': 'NK_obs_N_obs_r_accum-Q_extdry_fRH_hourly_obs_run7.npy',
                    'run8': 'NK_obs_N_obs_r_accum_corase_-Q_extdry_fRH_hourly_obs_run8.npy'}

    # load in all the data into one dictionary
    NK_runs = {run_i: np.load(npydir + file_i).flat[0] for run_i, file_i in NK_run_files.iteritems()}

    # gather absolute error (AE) and mean absolute error (MAE) statistics
    AE = []
    MAE = []
    diff = []
    rel_err = []
    n = []
    rs_mod = []
    rs_obs = []
    rs_beta = []
    for run_i in run_plot_order:
        # extract out all values that are not nan and store in AE
        good_idx = ~np.isnan(NK_runs[run_i]['statistics']['back_diff_norm'])
        AE += [np.abs(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]]
        MAE += np.nanmean([np.abs(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]])
        n += [np.sum(good_idx)] # sum of bools
        diff += [np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]]
        rel_err += [(np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx])/
                    (np.array(NK_runs[run_i]['statistics']['back_mod'])[good_idx])]


    # ----------------------------------------------------------------------------

    for run_i in run_plot_order:

        x = np.array(NK_runs[run_i]['statistics']['back_obs'])
        if run_i == 'run0':
            y = np.array(NK_runs[run_i]['statistics']['aer_mod'])
        else:
            y = np.array(NK_runs[run_i]['statistics']['aer_obs'])
        idx = np.isfinite(x) & np.isfinite(y)
        r, p = pearsonr(x[idx], y[idx])
        rso, ps = spearmanr(x[idx], y[idx])
        rs_obs += [rso]

        x = np.array(NK_runs[run_i]['statistics']['back_mod'])#[run1_good_idx]
        y = np.array(NK_runs[run_i]['statistics']['aer_obs'])#[run1_good_idx]
        idx = np.isfinite(x) & np.isfinite(y)
        r, p = pearsonr(x[idx], y[idx])
        rsm, ps = spearmanr(x[idx], y[idx])
        rs_mod += [rsm]

        x = np.array(NK_runs[run_i]['statistics']['back_mod'])  # [run1_good_idx]
        y = np.array(NK_runs[run_i]['statistics']['back_obs'])  # [run1_good_idx]
        idx = np.isfinite(x) & np.isfinite(y)
        r, p = pearsonr(x[idx], y[idx])
        rsb, ps = spearmanr(x[idx], y[idx])
        rs_beta += [rsb]

    # extra stats for paper

    # run0
    x = np.array(NK_runs['run0']['statistics']['back_obs'])#[run1_good_idx]
    y = np.array(NK_runs['run0']['statistics']['aer_obs'])#[run1_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    r, p = pearsonr(x[idx], y[idx])
    rs0o, ps = spearmanr(x[idx], y[idx])

    x = np.array(NK_runs['run0']['statistics']['back_mod'])#[run1_good_idx]
    y = np.array(NK_runs['run0']['statistics']['aer_mod'])#[run1_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    r, p = pearsonr(x[idx], y[idx])
    rs0m, ps = spearmanr(x[idx], y[idx])



    # run1 - PM10 as m
    x = np.array(NK_runs['run1']['statistics']['back_obs'])#[run1_good_idx]
    y = np.array(NK_runs['run1']['statistics']['aer_obs'])#[run1_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    r, p = pearsonr(x[idx], y[idx])
    rs1o, ps = spearmanr(x[idx], y[idx])

    x = np.array(NK_runs['run1']['statistics']['back_mod'])#[run1_good_idx]
    y = np.array(NK_runs['run1']['statistics']['aer_obs'])#[run1_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    r, p = pearsonr(x[idx], y[idx])
    rs1m, ps = spearmanr(x[idx], y[idx])
    # -----------------------------------

    x = np.array(NK_runs['run2']['statistics']['back_obs'])#[run2_good_idx]
    y = np.array(NK_runs['run2']['statistics']['aer_obs'])#[run2_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    r2m, p2 = pearsonr(x[idx], y[idx])
    rs2o, ps2 = spearmanr(x[idx], y[idx])

    x = np.array(NK_runs['run2']['statistics']['back_mod'])#[run2_good_idx]
    y = np.array(NK_runs['run2']['statistics']['aer_obs'])#[run2_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    r2o, p2 = pearsonr(x[idx], y[idx])
    rs2m, ps2 = spearmanr(x[idx], y[idx])

    # -----------------------------------

    aer_i = 'run5'
    x = np.array(NK_runs[aer_i]['statistics']['back_obs'])#[run2_good_idx]
    y = np.array(NK_runs[aer_i]['statistics']['aer_obs'])#[run2_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    r2m, p2 = pearsonr(x[idx], y[idx])
    rso, ps2 = spearmanr(x[idx], y[idx])
    print 'obs'
    print spearmanr(x[idx], y[idx])

    x = np.array(NK_runs[aer_i]['statistics']['back_mod'])#[run2_good_idx]
    y = np.array(NK_runs[aer_i]['statistics']['aer_obs'])#[run2_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    rsm, ps2 = spearmanr(x[idx], y[idx])
    print 'mod'
    print spearmanr(x[idx], y[idx])

    x = np.array(NK_runs[aer_i]['statistics']['back_mod'])#[run2_good_idx]
    y = np.array(NK_runs[aer_i]['statistics']['back_obs'])#[run2_good_idx]
    idx = np.isfinite(x) & np.isfinite(y)
    rsm, ps2 = spearmanr(x[idx], y[idx])
    print 'back'
    print spearmanr(x[idx], y[idx])

    # ---------------------------
    # Plotting
    # ---------------------------

    # 1. Box plot
    fig = plt.figure(figsize=(7, 3.5))
    ax = plt.gca()

    plt.boxplot(rel_err, whis=[5, 95], sym='')

    # prettify
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    #ax.set_ylim([0.0, 3.0e-05]) # use if y scale is linear

    # ax.set_yscale('log')
    # ax.set_ylabel(r'$Absolute \/\/Error\/\/\/ \mathrm{(|\beta_m - \beta_o|)}$')
    # ax.set_ylim([1e-8, 5e-05]) # use if y scale is linear
    # ax.set_ylabel(r'$Error\/\/\/ \mathrm{(\beta_m - \beta_o)}$')
    ax.set_ylabel(r'$Normalised\/\/\/Error\/\/\/ \mathrm{(\beta_m - \beta_o)/\beta_m}$')
    #ax.set_ylim([5e-05, -5e-05])  # use if y scale is linear
    ax.set_xlabel(r'$experiment\/\/number$')
    # set the positions of 1,2,3,4... to 0,1,2,3 ...
    plt.xticks(run_plot_position, run_plot_number)
    #plt.suptitle('NK')

    # add sample size at the top of plot for each box and whiskers
    # pos_t = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(n_i, 2)) for n_i in n]
    weights = ['bold', 'semibold']
    pos = np.arange(len(n)) + 1
    top = ax.get_ylim()[1]
    for tick, label in zip(range(len(pos)), ax.get_xticklabels()):
        k = tick % 2
        ax.text(pos[tick], top - (top * 0.08), upperLabels[tick],
                 horizontalalignment='center', size='x-small')

    plt.tight_layout()
    plt.savefig(savedir + 'NK_run_normalised_diff.png')



















    print 'END PROGRAM'