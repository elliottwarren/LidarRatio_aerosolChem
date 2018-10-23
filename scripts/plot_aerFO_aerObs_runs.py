"""
Plot the aerFO runs together. aerFO runs are done in beta_m_from_mod_and_aerObs.py

Created by Elliott Wed 30 May 2018
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import ellUtils as eu
import datetime as dt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from matplotlib.dates import  DateFormatter

if __name__ == '__main__':

    # ---------------------------
    # Setup
    # ---------------------------

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/'
    datadir = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/'
    savedir = maindir + 'figures/model_eval/summary_boxplots/'
    S_savedir = maindir + 'figures/model_eval/S_plots/'
    npydir = datadir + 'npy/model_eval/'

    # use this to retain the plotting order
    max_num_runs = 8
    run_plot_number = [str(i) for i in range(0, max_num_runs)]
    run_plot_position = range(1, max_num_runs+1)
    run_plot_order = ['run'+i for i in run_plot_number]

    # --------------------------
    # Read
    # --------------------------

    # NK
    # old set - pre hyst and shape corr
    # NK_run_files = {'run0': 'NK_CTRL_obs_N_obs_r_fine_accum_corase_-Q_extdry_fRH_hourly_obs_hourly_S_run0.npy',
    #                 'run1': 'NK_pm10aerinput_run1.npy',
    #                 'run2': 'NK_obs_N_obs_r_accum_run2.npy',
    #                 'run3': 'NK_obs_N_obs_r_fine_accum_run3.npy',
    #                 'run4': 'NK_obs_N_obs_r_fine_accum_coarse_run4.npy',
    #                 'run5': 'NK_pm10aerinput_Q_extdry_fRH_hourly_obs_run5.npy',
    #                 'run6': 'NK_obs_N_obs_r_accum-S_hourly_run6.npy',
    #                 'run7': 'NK_obs_N_obs_r_accum-Q_extdry_fRH_hourly_obs_run7.npy',
    #                 'run8': 'NK_obs_N_obs_r_accum_corase_-S_hourly_run8.npy',
    #                 'run9': 'NK_obs_N_obs_r_accum_corase_-Q_extdry_fRH_hourly_obs_run9.npy',
    #                 'run10': 'NK_obs_N_obs_r_accum_corase_-Q_extdry_fRH_hourly_obs-cheat_S_param_run10.npy'}

    NK_run_files = {'run0': 'run0_NK_pm10aerinput.npy',
                    'run1': 'run1_NK_obs_N_pm10_r_accum.npy',
                    'run2': 'run2_NK_pm10_N_obs_r_accum.npy',
                    'run3': 'run3_NK_obs_N_obs_r_accum.npy',
                    'run4': 'run4_NK_obs_N_obs_r_fine_accum.npy',
                    'run5': 'run5_NK_obs_N_obs_r_fine_accum_coarse.npy',
                    'run6': 'run6_NK_CTRL_obs_N_obs_r_fine_accum_corase_-Q_extdry_fRH_hourly_obs.npy',
                    'run7': 'run7_NK_CTRL_obs_N_obs_r_fine_accum_corase_-Q_extdry_fRH_hourly_obs_hourly_S.npy'}

    # 'run0': 'NK_CTRL_run0.npy',
    # 'run7': 'NK_obs_N_obs_r_accum-Q_extdry_fRH_hourly_obs_run7.npy',
    # 'run8': 'NK_obs_N_obs_r_accum_corase_-Q_extdry_fRH_hourly_obs_run8.npy'

    # load in all the data into one dictionary
    NK_runs = {run_i: np.load(npydir + file_i).flat[0] for run_i, file_i in NK_run_files.iteritems()}

    # find when data across all runs was present
    # 1.find bad entries for each run
    # 2.find unique bad entries (idxs that need to be missed)
    # 3.find which idxs were not mentioned in the unique bad entires as a booleon (using a complete idx range (0 - 175)
    # 4.turn booleon into idxs
    bad_idxs = np.array([])

    for run_i in run_plot_order[:]: # not run0 (CTRL has a different dataset)
        bad_date_idx = np.where(np.isnan(NK_runs[run_i]['statistics']['back_diff_norm']))
        bad_idxs = np.append(bad_idxs, bad_date_idx)
    bad_idxs.astype(int)
    # get unique idxs
    bad_idx_unique = np.unique(bad_idxs)
    all_idx = np.arange(175.0)
    keep = [True if i not in bad_idx_unique else False for i in all_idx]
    good_across_all_runs_idx = np.where(np.array(keep) == True)


    # # gather absolute error (AE) and mean absolute error (MAE) statistics
    # AE = []
    # MAE = []
    # diff = []
    # rel_err = []
    # n = []
    # rs_mod = []
    # rs_obs = []
    # rs_beta = []
    # for run_i in run_plot_order:
    #     # extract out all values that are not nan and store in AE
    #     good_idx = ~np.isnan(NK_runs[run_i]['statistics']['back_diff_norm'])
    #     AE += [np.abs(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]]
    #     MAE += np.nanmean([np.abs(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]])
    #     n += [np.sum(good_idx)] # sum of bools
    #     diff += [np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]]
    #     rel_err += [(np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx])/
    #                 (np.array(NK_runs[run_i]['statistics']['back_obs'])[good_idx])]

    # gather absolute error (AE) and mean absolute error (MAE) statistics
    AE = []
    MAE = []
    diff = []
    mean_diff = []
    rel_err = []
    n = []
    rs_mod = []
    rs_obs = []
    rs_beta = []
    stddev = []
    R_spearman = []
    p_spearman = []
    R_pearson = []
    p_pearson = []
    r_fine = []
    r_accum = []
    r_coarse = []
    spc_stddev = []

    # for run_i in [run_plot_order[0]]:
    #     # extract out all values that are not nan and store in AE
    #     good_idx = ~np.isnan(NK_runs[run_i]['statistics']['back_diff_norm'])
    #     AE += [np.abs(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]]
    #     MAE += np.nanmean([np.abs(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]])
    #     n += [np.sum(good_idx)] # sum of bools
    #     diff += [np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx]]
    #     rel_err += [(np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_idx])/
    #                 (np.array(NK_runs[run_i]['statistics']['back_obs'])[good_idx])]

    for run_i in run_plot_order:
        # extract out all values that are not nan and store in AE
        good_idx = ~np.isnan(NK_runs[run_i]['statistics']['back_diff_norm'])
        AE += [np.abs(NK_runs[run_i]['statistics']['back_diff_norm'])[good_across_all_runs_idx]]
        MAE += [np.nanmean([np.abs(NK_runs[run_i]['statistics']['back_diff_norm'])[good_across_all_runs_idx]])]
        mean_diff += [np.nanmean(np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_across_all_runs_idx])]
        n += [len(good_across_all_runs_idx[0])] # sum of bools
        diff += [np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_across_all_runs_idx]]
        rel_err += [(np.array(NK_runs[run_i]['statistics']['back_diff_norm'])[good_across_all_runs_idx])/
                    (np.array(NK_runs[run_i]['statistics']['back_obs'])[good_across_all_runs_idx])]
        stddev += [np.nanstd(np.array(NK_runs[run_i]['statistics']['back_mod'])[good_across_all_runs_idx])/
                   np.nanstd(np.array(NK_runs[run_i]['statistics']['back_obs'])[good_across_all_runs_idx])]

        # # remove the last 5 hours of 04/06/2015 which seem to be massive outliers.
        # idx = np.array([92, 93, 94, 95, 96]) # np.where(keep > 5.0e-6)[0] # [92, 93, 94, 95, 96]
        # times[idx]
        # a = np.array(NK_runs[run_i]['statistics']['back_mod'])[good_across_all_runs_idx]
        # b = np.array(NK_runs[run_i]['statistics']['back_obs'])[good_across_all_runs_idx]
        # a[idx] = np.nan
        # b[idx] = np.nan
        # spc_stddev += [np.nanstd(a)/np.nanstd(b)]


        # correlations
        r, p = pearsonr(np.array(NK_runs[run_i]['statistics']['back_mod'])[good_across_all_runs_idx],
                        np.array(NK_runs[run_i]['statistics']['back_obs'])[good_across_all_runs_idx])
        R_pearson += [r]
        p_pearson += [p]

        r, p = spearmanr(np.array(NK_runs[run_i]['statistics']['back_mod'])[good_across_all_runs_idx],
                             np.array(NK_runs[run_i]['statistics']['back_obs'])[good_across_all_runs_idx])
        R_spearman += [r]
        p_spearman += [p]

    # ----------------------------------------------------------------------------

    # # Correlation and such for each of the runs
    # for run_i in run_plot_order:
    #
    #     x = np.array(NK_runs[run_i]['statistics']['back_obs'])
    #     # if run_i == 'run0':
    #     #     y = np.array(NK_runs[run_i]['statistics']['aer_mod'])
    #     # else:
    #     #     y = np.array(NK_runs[run_i]['statistics']['aer_obs'])
    #     y = np.array(NK_runs[run_i]['statistics']['aer_obs'])
    #     idx = np.isfinite(x) & np.isfinite(y)
    #     r, p = pearsonr(x[idx], y[idx])
    #     rso, ps = spearmanr(x[idx], y[idx])
    #     rs_obs += [rso]
    #
    #     x = np.array(NK_runs[run_i]['statistics']['back_mod'])#[run1_good_idx]
    #     y = np.array(NK_runs[run_i]['statistics']['aer_obs'])#[run1_good_idx]
    #     idx = np.isfinite(x) & np.isfinite(y)
    #     r, p = pearsonr(x[idx], y[idx])
    #     rsm, ps = spearmanr(x[idx], y[idx])
    #     rs_mod += [rsm]
    #
    #     x = np.array(NK_runs[run_i]['statistics']['back_mod'])  # [run1_good_idx]
    #     y = np.array(NK_runs[run_i]['statistics']['back_obs'])  # [run1_good_idx]
    #     idx = np.isfinite(x) & np.isfinite(y)
    #     r, p = pearsonr(x[idx], y[idx])
    #     rsb, ps = spearmanr(x[idx], y[idx])
    #     rs_beta += [rsb]


    # ---------------------------
    # Plotting
    # ---------------------------

    # 1. Box plot
    fig = plt.figure(figsize=(7, 3.5))
    ax = plt.gca()

    plt.axhline(0.0, linestyle='--')
    plt.boxplot(diff, whis=[5, 95], sym='')
    plt.scatter(run_plot_position, mean_diff, marker='o', color='blue', edgecolor='blue')

    # prettify
    # ax.set_yscale('log')
    # ax.set_ylabel(r'$Absolute \/\/Error\/\/\/ \mathrm{(|\beta_m - \beta_o|)}$')
    # ax.set_ylim([1e-8, 1e-05]) # use if y scale is linear
    plt.axis('tight')
    ax.set_ylabel(r'$Error\/\/\/ \mathrm{(\beta_m - \beta_o)}$')
    # ax.set_ylabel(r'$Normalised\/\/\/Error\/\/\/ \mathrm{(\beta_m - \beta_o)/\beta_o}$')
    #ax.set_ylim([5e-05, -5e-05])  # use if y scale is linear
    ax.set_xlabel(r'$experiment\/\/number$')
    # set the positions of 1,2,3,4... to 0,1,2,3 ...
    plt.xticks(run_plot_position, run_plot_number)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.set_ylim([-3.0e-06, 4.0e-06]) # use if y scale is linear
    #plt.suptitle('NK')

    # add sample size at the top of plot for each box and whiskers
    # pos_t = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(n_i, 2)) for n_i in n]
    weights = ['bold', 'semibold']
    pos = np.arange(len(n)) + 1
    top = ax.get_ylim()[1]
    for tick, label in zip(range(len(pos)), ax.get_xticklabels()):
        k = tick % 2
        # ax.text(pos[tick], top - (top * 0.08), upperLabels[tick], # not AE
        ax.text(pos[tick], top - (top * 0.1), upperLabels[tick], #AE
                 horizontalalignment='center', size='x-small')

    plt.tight_layout()
    plt.savefig(savedir + 'NK_run_diff_tight.png')


    # ------------------------------------------


    # # 2. plot cheat_S vs hourly_S vs monthly_S
    #
    # # 2.1 Calculate cheat_S - aerFO total extinction coefficent / observed backscatter
    # print 'calculating cheat S using run9!'
    # print 'taking hourly S from run0!'
    # aerFO_ext_coeff_tot = NK_runs['run9']['statistics']['aer_ext_coeff_tot']
    # aer_FO_months = np.array([i.month for i in NK_runs['run9']['statistics']['datetime']])
    # aerFO_RH = NK_runs['run9']['statistics']['rh_mod']
    # cheat_S = (np.array(aerFO_ext_coeff_tot)*np.array(np.array(NK_runs['run9']['statistics']['transmission']))/
    #            np.array(NK_runs['run9']['statistics']['back_obs']))
    #
    # hourly_S = NK_runs['run0']['statistics']['S'] # taken straight from calculations
    # monthly_S = NK_runs['run9']['statistics']['S'] # taken from monthly LUT of S, as a function of RH
    #
    # # linear fit to cheat S
    # x = np.array(aerFO_RH)
    # y = np.array(cheat_S)
    # idx = np.isfinite(x) & np.isfinite(y)
    # m, b = np.polyfit(x[idx], y[idx], 1)
    #
    # # # create run 10 from 9!
    # # # cheat parameterisation
    # cheat_S_param = np.array([(m * i) + b for i in aerFO_RH])
    # # # attenuated backscatter using new S -> \beta_m = ext_coeff * T / S
    # # run10_beta = (np.array(aerFO_ext_coeff_tot)*np.array(np.array(NK_runs['run9']['statistics']['transmission'])))/\
    # #              cheat_S_param
    # # # calculate \beta_m - \beta_o
    # # run10_beta_diff = run10_beta - np.array(NK_runs['run9']['statistics']['back_obs'])
    #
    # # 2.2 Plot data
    # fig = plt.figure(figsize=(7, 3.5))
    # ax = plt.gca()
    #
    # plt.scatter(aerFO_RH, monthly_S, label='monthly S', s=4)
    # plt.scatter(aerFO_RH, hourly_S, label='hourly S', s=4)
    # plt.scatter(aerFO_RH, cheat_S, label='\'cheat\' S', s=4)
    # plt.scatter(aerFO_RH, cheat_S_param, label='\'cheat\' S param', s=4)
    # # plt.plot(RH_range, (m * RH_range) + b, label='\'cheat\' S param')
    #
    # #ax.plot(np.array(x_lim), m * np.array(x_lim) + b, ls='-', color='blue', label='obs')
    #
    # plt.legend(loc='best')
    # plt.xlabel('RH [%]')
    # plt.ylabel('S [sr]')
    #
    # plt.tight_layout()
    # plt.savefig(S_savedir+'hourly_S_vs_monthly_S_vs_cheat_S.png')


    # --------------------------

    # Check what is causing the 5 bad hours...

    x = np.array(NK_runs['run0']['statistics']['rh_mod'])
    y = np.array(NK_runs['run0']['statistics']['back_diff_norm'])
    idx = np.isfinite(x) & np.isfinite(y)
    r, p = pearsonr(x[idx], y[idx])
    rs, ps = spearmanr(x[idx], y[idx])
    # find the high diff points
    high_idx = np.where(y[idx] > 4.0e-06)[0]
    y[idx][high_idx]

    # # check transmission - seems ok
    # filename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/L1/CL31-D_CCW30_NK_2015_15min.nc'
    # data = eu.netCDF_read(filename)
    # _, t_idx, _ = eu.nearest(data['time'], dt.datetime(2015, 6, 4))
    # plt.figure()
    # plt.plot_date(data['time'][t_idx-(24*4):t_idx+(48*4)], data['transmission'][t_idx-(24*4):t_idx+(48*14)])
    # plt.ylabel('transmission for CL31-NK')
    # plt.xlabel('date [YYYY:mm:DD]')

    # check RH trend at the time
    # filename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/L1/Davis_IMU_2015_15min.nc'
    filename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/data/L1/WXT_KSSW_2015_15min.nc'
    data = eu.netCDF_read(filename)
    _, t_idx, _ = eu.nearest(data['time'], dt.datetime(2015, 6, 4))
    all_range = np.arange(t_idx, t_idx+(110*1))

    _, ts_idx, _ = eu.nearest(data['time'], dt.datetime(2015, 6, 4, 20, 0, 0))
    _, te_idx, _ = eu.nearest(data['time'], dt.datetime(2015, 6, 5, 0,  0, 0))
    focus_range = np.arange(ts_idx, te_idx)

    plt.figure(figsize=(11, 4))
    ax = plt.gca()
    # 96 = 1 day of 15 min data
    # plt.plot_date(data['time'][t_idx:t_idx+(110*1)], data['RH'][t_idx:t_idx+(110*1)], color='blue') # all data
    # plt.plot_date(data['time'][ts_idx:te_idx], data['RH'][ts_idx:te_idx], color='red') # just those 5 hours
    plt.plot_date(data['time'][all_range], data['RH'][all_range], color='blue') # all data
    plt.plot_date(data['time'][focus_range], data['RH'][focus_range], color='red') # just those 5 hours
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H'))
    #ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.ylabel('RH [%]')
    plt.xlabel('date [mm/DD HH]')
    plt.savefig('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/figures/diags/' + \
                'RH_NK_zoom.png')

    # RH_idx = np.isfinite(data['RH'])
    # plt.hist(data['RH'][RH_idx], bins=100)

    # check wind speed
    plt.figure(figsize=(11, 4))
    ax = plt.gca()
    plt.plot_date(data['time'][all_range], data['WS'][all_range], color='blue')
    plt.plot_date(data['time'][focus_range], data['WS'][focus_range], color='red')
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H'))
    plt.ylabel('wind speed [m s-1]')
    plt.xlabel('date [mm/DD HH]')
    plt.savefig('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/figures/diags/' + \
                'wind_speed_NK_20160604_zoom.png')

    # check wind direction and speed to see if its a sea breeze front
    plt.figure(figsize=(11, 4))
    ax = plt.gca()
    plt.plot_date(data['time'][all_range], data['dir'][all_range], color='blue')
    plt.plot_date(data['time'][focus_range], data['dir'][focus_range], color='red')
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H'))
    plt.ylabel('wind speed [m s-1]')
    plt.xlabel('date [mm/DD HH]')
    plt.savefig('C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/MorningBL/figures/diags/' + \
                'wind_dir_NK_20160604_zoom.png')

    fig = plt.figure()
    ax = plt.gca()

    plt.scatter(x,y)

    plt.xlabel('RH')
    plt.ylabel('diff')
    plt.ylim([np.nanmin(NK_runs['run0']['statistics']['back_diff_norm']),
              np.nanmax(NK_runs['run0']['statistics']['back_diff_norm'])])
    eu.add_at(ax, 'Pearson r=%1.5s, p=%1.1e; Spearman r=%1.5s, p=%1.1e; ' % (r, p, rs, ps), loc=1)
    plt.tight_layout()

    # -------------------------------------------------------------

    # # the 8 mini panel beta_m vs beta_o plots
    # fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(7,3.5))
    #
    # for ax_i, run_num_i in zip(ax.flatten(), run_plot_number):
    #
    #     run_i = 'run'+str(run_num_i)
    #
    #     data = NK_runs[run_i]['statistics']
    #
    #     # 1:1 line
    #     ax_i.plot([1.0e-9, 1.0e-3], [1.0e-9, 1.0e-3], linestyle='--', color='black')
    #
    #     # beta_o vs beta_m
    #     ax_i.scatter(data['back_obs'], data['back_mod'], s=4)
    #     ax_i.set_xscale("log")
    #     ax_i.set_yscale("log")
    #
    #     ax_i.set_xlim(5.0e-7, 1.0e-5)
    #     ax_i.set_ylim(1.0e-07, 2.0e-5)
    #
    #     # needed as the first plot seems to plot its minor ticks for some reason...
    #     ax_i.yaxis.set_minor_formatter(plt.NullFormatter())
    #
    #     plt.text(0.05, 0.85, str(run_num_i), transform=ax_i.transAxes)
    #
    # plt.subplots_adjust(bottom=0.2)
    # ax0 = eu.fig_majorAxis(fig)
    # ax0.set_ylabel(r'$\beta_m \/\/[m^{-1} sr^{-1}]$', labelpad=8)
    # ax0.set_xlabel(r'$\beta_o \/\/[m^{-1} sr^{-1}]$', labelpad=5)
    #
    # plt.savefig(savedir + 'beta_o_vs_beta_m.png')

    print 'END PROGRAM'